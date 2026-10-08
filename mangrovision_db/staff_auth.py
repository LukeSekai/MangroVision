"""Email-verified staff authentication using the existing private Postgres accounts.

Challenge cookies carry a random secret; only its SHA-256 hash and an HMAC of
the code live in the database. Codes and password inputs are never persisted.
All limits, attempts and consumption are durable across app restarts/workers.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import math
import re
import secrets
from datetime import datetime, timedelta, timezone

from .activity import append_activity
from .compat import get_connection
from .passwords import hash_password, verify_password
from .staff_email import send_verification_email

CODE_MINUTES = 10
RESEND_SECONDS = 60
MAX_ATTEMPTS = 5


class AuthError(Exception):
    def __init__(self, message: str, status: int = 400, retry_after: int | None = None,
                 *, field: str | None = None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after
        self.field = field


def _now():
    return datetime.now(timezone.utc)


def _date(value):
    return datetime.fromisoformat(value) if isinstance(value, str) else value


def _digest(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _fingerprint(user: dict) -> str:
    return _digest(json.dumps([user['password_hash'], user['full_name'], user['email']]))


def mask_email(email: str) -> str:
    local, _, domain = email.partition('@')
    return f'{local[:1]}***@{domain}' if domain else 'your registered email'


def validate_email(email: str) -> str:
    value = email.strip().lower()
    if len(value) > 254 or not re.fullmatch(r'[^\s@]+@[^\s@]+\.[^\s@]+', value):
        raise AuthError('Enter a valid email address.')
    return value


def _deliverable(email: str) -> bool:
    try:
        validate_email(email)
    except AuthError:
        return False
    return not email.lower().endswith(('.local', '.invalid', '.test'))


def validate_new_password(password: str | None) -> dict:
    if not password:
        raise AuthError('Enter a new password.')
    if not 12 <= len(password) <= 128 or not password.strip():
        raise AuthError('Use a password of 12–128 characters.')
    return {'password': password}


def _limit(conn, key: str, maximum: int) -> None:
    now = _now()
    row = conn.execute("""
        INSERT INTO staff_auth_limits(bucket_key, window_start, attempts) VALUES (?, ?, 1)
        ON CONFLICT (bucket_key) DO UPDATE SET
            attempts = CASE WHEN staff_auth_limits.window_start <= ? THEN 1 ELSE staff_auth_limits.attempts + 1 END,
            window_start = CASE WHEN staff_auth_limits.window_start <= ? THEN ? ELSE staff_auth_limits.window_start END
        RETURNING attempts, window_start
    """, (_digest(key), now, now-timedelta(minutes=10), now-timedelta(minutes=10), now)).fetchone()
    if row['attempts'] > maximum:
        wait = max(1, math.ceil((_date(row['window_start'])+timedelta(minutes=10)-now).total_seconds()))
        raise AuthError('Too many requests. Please try again later.', 429, wait)


def check_request_limit(identity: str, address: str) -> None:
    conn = get_connection()
    try:
        conn.execute('DELETE FROM staff_auth_limits WHERE window_start < ?', (_now()-timedelta(days=1),))
        # Do not trust forwarded headers as a source of client identity.
        _limit(conn, 'address:'+address, 30)
        _limit(conn, 'identity:'+identity.strip().lower(), 10)
        conn.commit()
    except AuthError:
        conn.commit()  # Persist throttled attempts, including concurrent requests.
        raise
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _issue(user: dict | None, purpose: str, *, email: str, pending=None) -> tuple[str, dict]:
    token = secrets.token_urlsafe(32)
    code = f'{secrets.randbelow(1_000_000):06d}'
    token_hash = _digest(token)
    key = _digest(email.strip().lower())
    now = _now()
    conn = get_connection()
    try:
        # The per-address lock also serializes decoy recovery requests.
        conn.execute('SELECT pg_advisory_xact_lock(?)', (int(key[:15], 16),))
        if user:
            current = conn.execute('SELECT * FROM users WHERE id = ? FOR UPDATE', (user['id'],)).fetchone()
            if not current or not hmac.compare_digest(_fingerprint(dict(current)), _fingerprint(user)):
                raise AuthError('Account details changed. Please start again.', 401)
            if not _deliverable(user['email']):
                raise AuthError('This account needs a real email address. Contact your administrator.', 409)
        recent = conn.execute("""
            SELECT created_at, purpose, credential_fingerprint FROM staff_auth_challenges
            WHERE request_key = ? AND created_at > ? ORDER BY created_at DESC
        """, (key, now-timedelta(hours=1))).fetchall()
        fingerprint = _fingerprint(user) if user else ''
        last_same_flow = next((row for row in recent if row['purpose'] == purpose and row['credential_fingerprint'] == fingerprint), None)
        if last_same_flow:
            wait = RESEND_SECONDS - (now-_date(last_same_flow['created_at'])).total_seconds()
            if wait > 0:
                raise AuthError('Please wait before requesting another code.', 429, math.ceil(wait))
        if len(recent) >= 10:
            raise AuthError('Too many verification emails. Please try again later.', 429, 3600)
        conn.execute('DELETE FROM staff_auth_challenges WHERE expires_at < ?', (now-timedelta(days=1),))
        conn.execute("""
            UPDATE staff_auth_challenges SET consumed_at = ?
            WHERE request_key = ? AND purpose = ? AND consumed_at IS NULL
        """, (now, key, purpose))
        conn.execute("""
            INSERT INTO staff_auth_challenges (
                token_hash, user_id, purpose, request_key, code_hash, credential_fingerprint,
                pending_changes, created_at, expires_at
            ) VALUES (?, ?, ?, ?, ?, ?, CAST(? AS jsonb), ?, ?)
        """, (token_hash, user['id'] if user else None, purpose, key,
              hmac.new(token.encode(), code.encode(), hashlib.sha256).hexdigest(),
              _fingerprint(user) if user else '', json.dumps(pending or {}), now, now+timedelta(minutes=CODE_MINUTES)))
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    if user:
        try:
            send_verification_email(token, code, user['email'], purpose)
        except Exception:
            _mark_delivery(token_hash, False)
            raise AuthError('The verification email could not be sent. Please try again in a minute or contact your administrator.', 503) from None
    _mark_delivery(token_hash, True)
    return token, {'verification_required':True, 'email_hint':mask_email(email),
                   'expires_in':CODE_MINUTES*60, 'resend_after':RESEND_SECONDS}


def _mark_delivery(token_hash: str, success: bool) -> None:
    conn = get_connection()
    try:
        conn.execute('UPDATE staff_auth_challenges SET delivered = ?, consumed_at = CASE WHEN ? THEN consumed_at ELSE ? END WHERE token_hash = ?',
                     (success, success, _now(), token_hash))
        conn.commit()
    finally:
        conn.close()


def begin_login(username: str, password: str, address: str) -> tuple[str, dict]:
    import planting_database as db
    check_request_limit(username, address)
    try:
        user = db.authenticate_user(username, password, record_login=False, explain_errors=True)
    except db.StaffCredentialError as error:
        raise AuthError(str(error), 401, field=error.field) from None
    if not user:
        raise AuthError('Invalid username or password.', 401)
    return _issue(user, 'login', email=user['email'])


def begin_recovery(email: str, address: str) -> tuple[str, dict]:
    email = validate_email(email)
    check_request_limit(email, address)
    conn = get_connection()
    try:
        rows = conn.execute('SELECT * FROM users WHERE lower(email) = ?', (email,)).fetchall()
    finally:
        conn.close()
    # The same challenge response for unknown addresses prevents enumeration.
    user = dict(rows[0]) if len(rows) == 1 else None
    return _issue(user, 'recovery', email=email)


def begin_account_change(user: dict, current_password: str,
                         password: str, address: str) -> tuple[str, dict]:
    check_request_limit(str(user['id']), address)
    if not verify_password(user.get('password_hash') or '', current_password)[0]:
        raise AuthError('Your current password is incorrect.', 401)
    changes = validate_new_password(password)
    changes['password_hash'] = hash_password(changes.pop('password'))
    return _issue(user, 'settings', email=user['email'], pending=changes)


def resend(token: str, purpose: str, address: str, *, user_id: int | None = None) -> tuple[str, dict]:
    conn = get_connection()
    try:
        row = conn.execute('SELECT * FROM staff_auth_challenges WHERE token_hash = ? AND purpose = ?', (_digest(token), purpose)).fetchone()
        if not row or row['consumed_at'] or _date(row['expires_at']) <= _now() or (user_id is not None and row['user_id'] != user_id):
            raise AuthError('Verification expired. Please start again.', 401)
        user = conn.execute('SELECT * FROM users WHERE id = ?', (row['user_id'],)).fetchone() if row['user_id'] else None
        if not user or not hmac.compare_digest(row['credential_fingerprint'], _fingerprint(dict(user))):
            raise AuthError('Verification expired. Please start again.', 401)
        pending = json.loads(row['pending_changes'])
    finally:
        conn.close()
    check_request_limit(str(user['id']), address)
    return _issue(dict(user), purpose, email=user['email'], pending=pending)


def finish(token: str, code: str, purpose: str, *, user_id: int | None = None,
           password: str | None = None) -> tuple[dict, str | None]:
    import planting_database as db
    code = code.strip()
    if not re.fullmatch(r'[0-9]{6}', code):
        raise AuthError('Enter the six-digit code from your email.')
    recovery_changes = validate_new_password(password) if purpose == 'recovery' else None
    conn = get_connection()
    now = _now()
    try:
        lookup = conn.execute('SELECT user_id FROM staff_auth_challenges WHERE token_hash = ?', (_digest(token),)).fetchone()
        # Consistent lock order with issuance: user, then challenge.
        user = conn.execute('SELECT * FROM users WHERE id = ? FOR UPDATE', (lookup['user_id'],)).fetchone() if lookup and lookup['user_id'] else None
        row = conn.execute('SELECT * FROM staff_auth_challenges WHERE token_hash = ? FOR UPDATE', (_digest(token),)).fetchone()
        if (not row or row['purpose'] != purpose or row['consumed_at'] or not row['delivered']
                or _date(row['expires_at']) <= now or row['attempts'] >= MAX_ATTEMPTS
                or (user_id is not None and row['user_id'] != user_id)):
            raise AuthError('Verification expired or unavailable. Please start again.', 401)
        candidate = hmac.new(token.encode(), code.encode(), hashlib.sha256).hexdigest()
        valid = hmac.compare_digest(row['code_hash'], candidate)
        if not valid or not user:
            attempts = row['attempts'] + 1
            conn.execute('UPDATE staff_auth_challenges SET attempts = ?, consumed_at = ? WHERE token_hash = ?',
                         (attempts, now if attempts >= MAX_ATTEMPTS else None, row['token_hash']))
            conn.commit()
            raise AuthError('Too many incorrect codes. Please start again.' if attempts >= MAX_ATTEMPTS else 'Incorrect verification code.', 401)
        user = dict(user)
        if not hmac.compare_digest(row['credential_fingerprint'], _fingerprint(user)):
            conn.execute('UPDATE staff_auth_challenges SET consumed_at = ? WHERE token_hash = ?', (now, row['token_hash']))
            conn.commit()
            raise AuthError('Account details changed. Please start again.', 401)
        changes = recovery_changes if purpose == 'recovery' else json.loads(row['pending_changes'])
        if purpose != 'login':
            # Earlier settings challenges may contain a username. Only a pending
            # password hash is usable now; username-only challenges must restart.
            new_hash = hash_password(changes['password']) if 'password' in changes else changes.get('password_hash')
            if not new_hash:
                raise AuthError('Request a new code to change your password.', 409)
            conn.execute('UPDATE users SET password_hash = ? WHERE id = ?', (new_hash, user['id']))
            conn.execute("UPDATE auth_sessions SET revoked_at = ? WHERE subject_type = 'user' AND subject_id = ? AND revoked_at IS NULL", (now, user['id']))
            conn.execute('UPDATE staff_auth_challenges SET consumed_at = ? WHERE user_id = ? AND consumed_at IS NULL', (now, user['id']))
            append_activity(conn, action='auth.credentials_changed', actor_type='staff', actor_user_id=user['id'],
                            summary='Staff account password changed after email verification.',
                            details={'via':purpose, 'username_changed':False,
                                     'password_changed':new_hash != user['password_hash']})
            session = None
        else:
            conn.execute('UPDATE users SET last_login = ? WHERE id = ?', (now, user['id']))
            session = db._create_session('user', user['id'], connection=conn)
        conn.execute('UPDATE staff_auth_challenges SET consumed_at = ? WHERE token_hash = ?', (now, row['token_hash']))
        conn.commit()
        return {'id':user['id'], 'full_name':user['full_name'], 'role':user.get('role', 'planner'), 'email':user['email']}, session
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
