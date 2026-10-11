"""Codes authenticate one stable participant slot; existing point ownership stays intact."""
from __future__ import annotations

import base64
import hashlib
import hmac
import os
import re
import secrets
from datetime import datetime, timedelta, timezone
from cryptography.fernet import InvalidToken

from .compat import get_connection
from .appointment_email import _cipher
from .organization_accounts import is_registration_pending

ALPHABET = 'ABCDEFGHJKLMNPQRSTUVWXYZ23456789'


class CodeError(ValueError):
    def __init__(self, message, status=400, retry_after=None):
        super().__init__(message)
        self.status = status
        self.retry_after = retry_after


def normalize_code(value):
    code = re.sub(r'[\s-]', '', str(value or '')).upper()
    if len(code) != 8 or any(char not in ALPHABET for char in code):
        raise CodeError('Enter your eight-character planter code, for example K7MP-4R9X.')
    return code


def _digest(code):
    # The existing stable private encryption key also protects short codes from
    # offline guessing if a database copy is disclosed. Never store raw SHA(code).
    _cipher()  # Validate configuration before any writes.
    key = base64.urlsafe_b64decode(os.environ['LIKE_EMAIL_ENCRYPTION_KEY'].strip())
    return hmac.new(key, b'mangrovision:planter-code:v1:' + code.encode('ascii'), hashlib.sha256).hexdigest()


def _fresh_code(conn):
    for _ in range(20):
        value = ''.join(secrets.choice(ALPHABET) for _ in range(8))
        digest = _digest(value)
        if not conn.execute('SELECT 1 FROM planter_access_codes WHERE code_digest = ?', (digest,)).fetchone():
            return digest, _cipher().encrypt(value.encode('ascii')).decode('ascii')
    raise ValueError('Could not generate a unique planter code. Try again.')


def _account(conn, planter_id):
    row = conn.execute('SELECT * FROM planters WHERE id = ? FOR UPDATE', (planter_id,)).fetchone()
    if not row or row['merged_into_planter_id'] is not None:
        raise CodeError('Organization account not found.', 404)
    if row['status'] != 'active' or is_registration_pending(dict(row)):
        raise CodeError('Create or reactivate this organization’s access before issuing codes.', 409)
    return dict(row)


def code_list(conn, planter_id):
    rows = conn.execute('''SELECT participant_slot, encrypted_code, revoked_at FROM planter_access_codes
        WHERE planter_id = ? ORDER BY participant_slot''', (planter_id,)).fetchall()
    result = []
    for row in rows:
        code = None
        if row['revoked_at'] is None:
            try:
                value = _cipher().decrypt(row['encrypted_code'].encode('ascii')).decode('ascii')
            except (InvalidToken, UnicodeError) as error:
                raise CodeError('The planter codes could not be opened. Check the stable email encryption key.', 503) from error
            code = value[:4] + '-' + value[4:]
        result.append({'participant_slot': row['participant_slot'], 'code': code,
                       'revoked': row['revoked_at'] is not None})
    return result


def ensure_codes(conn, planter_id):
    """Idempotent issuance only. Never change slots, devices, sessions or revoked codes."""
    account = _account(conn, planter_id)
    slots = conn.execute('''SELECT op.slot FROM organization_participants op
        LEFT JOIN planter_access_codes pc ON pc.planter_id = op.planter_id AND pc.participant_slot = op.slot
        WHERE op.planter_id = ? AND op.slot <= ? AND pc.planter_id IS NULL ORDER BY op.slot''',
        (planter_id, account['participant_count'])).fetchall()
    for slot in slots:
        digest, encrypted = _fresh_code(conn)
        conn.execute('''INSERT INTO planter_access_codes(planter_id, participant_slot, code_digest, encrypted_code)
            VALUES (?, ?, ?, ?)''', (planter_id, slot['slot'], digest, encrypted))
    return code_list(conn, planter_id)


def create_code_account(values, user_id):
    """Complete a reserved owner or create a new owner and codes in one transaction."""
    import planting_database as planting
    from .readable_credentials import available_username
    conn = get_connection()
    try:
        org = conn.execute('SELECT name FROM organizations WHERE id = ? FOR UPDATE',
                           (values['organization_id'],)).fetchone()
        if not org:
            raise CodeError('Selected organization was not found.', 404)
        # Internal legacy columns remain populated for compatibility with saved
        # records. These credentials are never returned and cannot sign in /field.
        username = available_username(conn, org['name'], values['organization_id'])
        planter_id = planting.create_planter(**values, username=username,
            password=secrets.token_urlsafe(40), created_by_user_id=user_id, _connection=conn)
        codes = ensure_codes(conn, planter_id)
        conn.commit()
        return {'planter_id': planter_id, 'codes': codes}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _delivery_guard(conn, organization_id):
    sending = conn.execute('''SELECT 1 FROM like_appointment_emails e
        JOIN like_appointment_requests r ON r.id = e.request_id
        JOIN planting_schedules s ON s.id = r.schedule_id
        WHERE s.organization_id = ? AND e.status = 'sending' LIMIT 1''', (organization_id,)).fetchone()
    if sending:
        raise CodeError('An access email is being delivered. Try again when delivery finishes.', 409)


def refresh_pending_code_emails(conn, organization_id, codes):
    """Replace pending credential messages without sending mail or changing schedules."""
    from .appointment_email import _message, _encrypted_message
    rows = conn.execute('''SELECT e.id, e.message_kind, r.id AS request_id, r.schedule_id
        FROM like_appointment_emails e JOIN like_appointment_requests r ON r.id = e.request_id
        JOIN planting_schedules s ON s.id = r.schedule_id
        WHERE s.organization_id = ? AND e.status = 'pending' FOR UPDATE OF e''', (organization_id,)).fetchall()
    for row in rows:
        request = dict(conn.execute('SELECT * FROM like_appointment_requests WHERE id = ?', (row['request_id'],)).fetchone())
        if request['appointment_type'] != 'tree_planting':
            continue
        schedule = dict(conn.execute('SELECT * FROM planting_schedules WHERE id = ?', (row['schedule_id'],)).fetchone())
        message = _message(request, schedule, login={'codes': codes}, kind=row['message_kind'])
        conn.execute('UPDATE like_appointment_emails SET encrypted_message = ? WHERE id = ?',
                     (_encrypted_message(message), row['id']))


def manage_codes(planter_id, *, issue=False, slot=None, action=None):
    conn = get_connection()
    try:
        account = _account(conn, planter_id)
        if issue or action:
            _delivery_guard(conn, account['organization_id'])
        if issue:
            codes = ensure_codes(conn, planter_id)
        elif action:
            if action not in {'regenerate', 'revoke'}:
                raise CodeError('Choose regenerate or revoke.')
            row = conn.execute('''SELECT participant_slot FROM planter_access_codes
                WHERE planter_id = ? AND participant_slot = ? FOR UPDATE''', (planter_id, slot)).fetchone()
            if not row:
                raise CodeError('Issue participant codes before changing this code.', 404)
            if action == 'regenerate':
                digest, encrypted = _fresh_code(conn)
                conn.execute('''UPDATE planter_access_codes SET code_digest = ?, encrypted_code = ?,
                    revoked_at = NULL, updated_at = CURRENT_TIMESTAMP
                    WHERE planter_id = ? AND participant_slot = ?''', (digest, encrypted, planter_id, slot))
            else:
                conn.execute('''UPDATE planter_access_codes SET revoked_at = CURRENT_TIMESTAMP,
                    updated_at = CURRENT_TIMESTAMP WHERE planter_id = ? AND participant_slot = ?''', (planter_id, slot))
            # Revocation must terminate that participant's existing access too.
            # Device identity and assigned points are preserved for safe recovery.
            conn.execute('''UPDATE auth_sessions SET revoked_at = CURRENT_TIMESTAMP
                WHERE subject_type = 'planter' AND subject_id = ? AND participant_slot = ? AND revoked_at IS NULL''',
                (planter_id, slot))
            codes = code_list(conn, planter_id)
        else:
            codes = code_list(conn, planter_id)
        if issue or action:
            refresh_pending_code_emails(conn, account['organization_id'], codes)
        conn.commit()
        return {'planter_id': planter_id, 'codes': codes, 'participant_count': account['participant_count']}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _limit(conn, key, maximum):
    now = datetime.now(timezone.utc)
    row = conn.execute('''INSERT INTO planter_auth_limits(bucket_key, window_start, attempts) VALUES (?, ?, 1)
        ON CONFLICT(bucket_key) DO UPDATE SET
            attempts = CASE WHEN planter_auth_limits.window_start <= ? THEN 1 ELSE planter_auth_limits.attempts + 1 END,
            window_start = CASE WHEN planter_auth_limits.window_start <= ? THEN excluded.window_start ELSE planter_auth_limits.window_start END
        RETURNING attempts, window_start''',
        (hashlib.sha256(key.encode()).hexdigest(), now, now - timedelta(minutes=10), now - timedelta(minutes=10))).fetchone()
    if row['attempts'] > maximum:
        start = datetime.fromisoformat(row['window_start'])
        raise CodeError('Too many sign-in attempts. Wait a few minutes and try again.', 429,
                        max(1, int((start + timedelta(minutes=10) - now).total_seconds())))


def authenticate_code(code, device_key, *, known_device_keys=(), resume_device=False, address='unknown'):
    """Authenticate and bind only the slot identified by this code, atomically."""
    import planting_database as planting
    normalized = normalize_code(code)
    digest = _digest(normalized)
    conn = get_connection()
    try:
        # Durable limits work across backend workers. The peer limit allows group
        # arrivals through a shared proxy; the separate code limit stops guessing.
        try:
            conn.execute("DELETE FROM planter_auth_limits WHERE window_start < CURRENT_TIMESTAMP - INTERVAL '1 day'")
            _limit(conn, 'peer:' + address, 600)
            _limit(conn, 'code:' + digest, 20)
        finally:
            conn.commit()
        found = conn.execute('SELECT planter_id, participant_slot FROM planter_access_codes WHERE code_digest = ?', (digest,)).fetchone()
        if not found:
            raise CodeError('This planter code is incorrect or no longer valid. Check it with your group leader or LGU staff.', 401)
        account = _account(conn, found['planter_id'])
        binding = conn.execute('''SELECT pc.revoked_at, op.device_key_hash FROM planter_access_codes pc
            JOIN organization_participants op ON op.planter_id = pc.planter_id AND op.slot = pc.participant_slot
            WHERE pc.planter_id = ? AND pc.participant_slot = ? AND pc.code_digest = ? FOR UPDATE OF pc, op''',
            (found['planter_id'], found['participant_slot'], digest)).fetchone()
        slot = found['participant_slot']
        if not binding or binding['revoked_at'] is not None or slot > account['participant_count']:
            raise CodeError('This planter code is incorrect or no longer valid. Check it with your group leader or LGU staff.', 401)
        if not isinstance(device_key, str) or not 16 <= len(device_key) <= 200:
            raise CodeError('A saved device identity is required. Reload this page and try again.', 409)
        keys = [device_key] if resume_device else [device_key, *known_device_keys]
        candidates = {hashlib.sha256(key.encode()).hexdigest(): key for key in keys
                      if isinstance(key, str) and 16 <= len(key) <= 200}
        identity = device_key
        if binding['device_key_hash']:
            identity = candidates.get(binding['device_key_hash'])
            if not identity:
                raise CodeError('This code belongs to a participant already using another browser. Use that browser or your device recovery code, or ask LGU staff to reset your device.', 409)
        else:
            if resume_device:
                raise CodeError('This device recovery code does not match a saved device. Use normal sign-in after an LGU reset.', 409)
            device_hash = hashlib.sha256(identity.encode()).hexdigest()
            if conn.execute('SELECT 1 FROM organization_participants WHERE planter_id = ? AND device_key_hash = ?',
                            (account['id'], device_hash)).fetchone():
                identity = secrets.token_hex(24)
                device_hash = hashlib.sha256(identity.encode()).hexdigest()
            conn.execute('UPDATE organization_participants SET device_key_hash = ? WHERE planter_id = ? AND slot = ?',
                         (device_hash, account['id'], slot))
        token = planting._create_session('planter', account['id'], slot, connection=conn)
        conn.execute('UPDATE planters SET last_login_at = CURRENT_TIMESTAMP WHERE id = ?', (account['id'],))
        account['participant_slot'] = slot
        conn.commit()
        return account, identity, token
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
