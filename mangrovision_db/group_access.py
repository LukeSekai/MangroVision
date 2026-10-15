"""One activity code, permanent participant usernames, and one current session.

The username is an identifier, not a second secret. Members possessing the group
code may resume a named participant, with explicit confirmation to replace an
active browser. All slot, capacity, and session changes commit together.
"""
from __future__ import annotations

import hashlib
import json
import re
from datetime import datetime, timedelta, timezone

from cryptography.fernet import InvalidToken

from .appointment_email import _cipher
from .compat import get_connection
from .planter_codes import CodeError, _account, _digest, _limit, normalize_code
from .request_context import planter_session_token


class GroupAccessError(CodeError):
    def __init__(self, message, status=400, *, field=None, kind=None, **details):
        super().__init__(message, status)
        self.field = field
        self.kind = kind
        self.details = details


class SharedAccessSetupError(GroupAccessError):
    def __init__(self):
        super().__init__(
            'Shared planter access needs database update 20261015_0019 before planting requests can be approved. '
            'Finish the database setup and restart the server, then try again.',
            503, kind='shared_access_setup_required')


def require_shared_access_schema(conn):
    """Detect an incomplete rollout before creating accounts or credentials."""
    ready = conn.execute("""SELECT
        to_regclass('activity_access_codes') IS NOT NULL
        AND to_regclass('participant_browser_bindings') IS NOT NULL
        AND NOT EXISTS (
            SELECT 1 FROM (VALUES
                ('organization_participants', 'participant_username'),
                ('organization_participants', 'username_key'),
                ('organization_participants', 'current_session_hash'),
                ('auth_sessions', 'planting_schedule_id'),
                ('auth_sessions', 'activity_code_digest'),
                ('auth_sessions', 'browser_key_hash')
            ) AS required(relation_name, column_name)
            WHERE NOT EXISTS (
                SELECT 1 FROM pg_catalog.pg_attribute
                WHERE attrelid = to_regclass(required.relation_name)
                    AND attname = required.column_name AND attnum > 0 AND NOT attisdropped
            )
        ) AS ready""").fetchone()
    if not ready or not ready['ready']:
        raise SharedAccessSetupError()


def username(value):
    display = str(value or '').strip()
    key = display.lower()
    if not re.fullmatch(r'[a-z0-9][a-z0-9._-]{2,31}', key):
        raise GroupAccessError('Use 3–32 letters or numbers. Dots, underscores and hyphens are allowed.', field='username')
    return display, key


def browser_hash(value):
    if not re.fullmatch(r'[a-f0-9]{48}', str(value or '')):
        raise GroupAccessError('Allow website storage so this browser can remember your participant.', field='browser_key')
    return hashlib.sha256(value.encode()).hexdigest()


def ensure_activity_code(conn, schedule):
    """Create access once; no code rotation or reassignment on repeated bookings."""
    import secrets
    import planting_database as planting
    from .organization_accounts import is_registration_pending
    from .readable_credentials import available_username
    from .planter_codes import ALPHABET

    if schedule.get('appointment_type', 'tree_planting') != 'tree_planting':
        return None
    if schedule['status'] not in {'confirmed', 'in_progress'}:
        return None
    require_shared_access_schema(conn)
    # Same lock order as appointment approval: organization, owner, access row.
    org_id = int(schedule['organization_id'])
    conn.execute('SELECT id FROM organizations WHERE id = ? FOR UPDATE', (org_id,)).fetchone()
    owner = conn.execute('''SELECT * FROM planters WHERE organization_id = ?
        AND merged_into_planter_id IS NULL FOR UPDATE''', (org_id,)).fetchone()
    count = int(schedule.get('expected_planters') or 1)
    if not 1 <= count <= 10000:
        raise ValueError('The planting activity needs a participant count between 1 and 10,000.')
    if owner and owner['status'] != 'active':
        raise ValueError('Reactivate this organization before creating planting access.')
    if not owner or is_registration_pending(dict(owner)):
        owner_id = planting.create_planter(full_name=schedule['organization'], organization_id=org_id,
            username=available_username(conn, schedule['organization'], org_id),
            password=secrets.token_urlsafe(40), participant_count=count, _connection=conn)
    else:
        owner_id = owner['id']
        # More participants on a later booking adds slots; existing allocations stay put.
        conn.execute('UPDATE planters SET participant_count = GREATEST(participant_count, ?) WHERE id = ?', (count, owner_id))
        conn.execute('''INSERT INTO organization_participants(planter_id, slot)
            SELECT ?, generate_series(1, ?) ON CONFLICT DO NOTHING''', (owner_id, count))
    existing = conn.execute('SELECT * FROM activity_access_codes WHERE schedule_id = ? FOR UPDATE', (schedule['id'],)).fetchone()
    if not existing:
        for _ in range(20):
            raw = ''.join(secrets.choice(ALPHABET) for _ in range(8))
            digest = _digest(raw)
            if not conn.execute('SELECT 1 FROM activity_access_codes WHERE code_digest = ?', (digest,)).fetchone():
                break
        else:
            raise ValueError('Could not create a unique organization code. Try again.')
        conn.execute('''INSERT INTO activity_access_codes(schedule_id, planter_id, code_digest, encrypted_code)
            VALUES (?, ?, ?, ?)''', (schedule['id'], owner_id, digest, _cipher().encrypt(raw.encode()).decode()))
    return activity_code(conn, schedule['id'])


def activity_code(conn, schedule_id):
    row = conn.execute('''SELECT ac.*, s.start_at, s.end_at, s.title, s.organization_id,
        s.status, s.expected_planters FROM activity_access_codes ac
        JOIN planting_schedules s ON s.id = ac.schedule_id WHERE ac.schedule_id = ?''', (schedule_id,)).fetchone()
    if not row:
        return None
    value = None
    if row['revoked_at'] is None:
        try:
            raw = _cipher().decrypt(row['encrypted_code'].encode()).decode()
        except InvalidToken as error:
            raise GroupAccessError('The organization code could not be opened. Check the stable encryption key.', 503) from error
        value = raw[:4] + '-' + raw[4:]
    return {key: row[key] for key in ('schedule_id', 'planter_id', 'start_at', 'end_at', 'title', 'status', 'expected_planters')} | {
        'code': value, 'revoked': row['revoked_at'] is not None}


def _open_activity(conn, digest):
    row = conn.execute('''SELECT ac.*, s.organization_id, s.title, s.start_at, s.end_at,
        s.status, s.expected_planters, s.appointment_type,
        s.start_at <= clock_timestamp() AS started, s.end_at > clock_timestamp() AS open
        FROM activity_access_codes ac JOIN planting_schedules s ON s.id = ac.schedule_id
        WHERE ac.code_digest = ? FOR SHARE OF s, ac''', (digest,)).fetchone()
    if not row or row['revoked_at'] is not None:
        raise GroupAccessError('This organization code is incorrect or no longer valid.', 401, field='access_code')
    if row['status'] not in {'confirmed', 'in_progress'} or row['appointment_type'] != 'tree_planting' or not row['open']:
        raise GroupAccessError('This planting activity has ended or is unavailable. Your saved progress is kept.', 403)
    if not row['started']:
        raise GroupAccessError('This code opens at the activity’s scheduled start time.', 403)
    return dict(row)


def _proof_slot(conn, account_id, browser_digest, previous_code):
    import planting_database as planting
    token = planter_session_token.get()
    if token:
        row = conn.execute('''SELECT participant_slot FROM auth_sessions WHERE subject_type = 'planter'
            AND subject_id = ? AND token_hash = ? AND revoked_at IS NULL AND expires_at > clock_timestamp()''',
            (account_id, planting._hash_session_token(token))).fetchone()
        if row:
            return row['participant_slot']
    row = conn.execute('''SELECT slot FROM organization_participants
        WHERE planter_id = ? AND device_key_hash = ?''', (account_id, browser_digest)).fetchone()
    if row:
        return row['slot']
    if previous_code:
        digest = _digest(normalize_code(previous_code))
        row = conn.execute('''SELECT participant_slot FROM planter_access_codes WHERE planter_id = ?
            AND code_digest = ? AND revoked_at IS NULL''', (account_id, digest)).fetchone()
        if not row:
            raise GroupAccessError('Check your previous individual code. It must belong to this organization.', 401, field='previous_code')
        return row['participant_slot']
    return None


def _switch_challenge(activity, key, browser_digest, active_hash):
    payload = {'schedule': activity['schedule_id'], 'digest': activity['code_digest'], 'username': key,
               'browser': browser_digest, 'session': active_hash,
               'expires': (datetime.now(timezone.utc) + timedelta(minutes=5)).timestamp()}
    return _cipher().encrypt(json.dumps(payload).encode()).decode()


def authenticate_group(access_code, name, browser_key, *, mode='return', switch_token=None, previous_code=None, address='unknown'):
    import planting_database as planting
    display, key = username(name)
    browser_digest = browser_hash(browser_key)
    digest = _digest(normalize_code(access_code))
    conn = get_connection()
    try:
        require_shared_access_schema(conn)
        # Group arrivals share one code and may share one proxy IP. Limit individual
        # username attempts instead of locking out 100 valid users on the same code.
        try:
            _limit(conn, 'group-peer:' + address, 2400)
            _limit(conn, 'group-user:' + digest + ':' + key, 30)
        finally:
            conn.commit()
        activity = _open_activity(conn, digest)
        account = _account(conn, activity['planter_id'])
        if account['organization_id'] != activity['organization_id']:
            raise GroupAccessError('This activity’s organization access needs to be corrected.', 409)
        bound = conn.execute('''SELECT op.* FROM participant_browser_bindings b
            JOIN organization_participants op ON op.planter_id = b.planter_id AND op.slot = b.participant_slot
            WHERE b.planter_id = ? AND b.browser_key_hash = ?''', (account['id'], browser_digest)).fetchone()
        participant = conn.execute('''SELECT * FROM organization_participants
            WHERE planter_id = ? AND username_key = ? FOR UPDATE''', (account['id'], key)).fetchone()
        if mode == 'first':
            if bound and bound['username_key']:
                raise GroupAccessError('This browser has already joined. Use Returning planter.', 409,
                    kind='already_joined', participant_username=bound['participant_username'])
            if participant:
                raise GroupAccessError('That username is already used in this organization. Return with it or choose another.', 409, field='username')
            slot = _proof_slot(conn, account['id'], browser_digest, previous_code)
            if slot is None:
                # Existing anonymous participants must prove their old slot before
                # naming it; never silently give somebody else's progress to a newcomer.
                participant = conn.execute('''SELECT op.* FROM organization_participants op
                    WHERE op.planter_id = ? AND op.username_key IS NULL AND op.device_key_hash IS NULL
                    AND NOT EXISTS (SELECT 1 FROM auth_sessions old WHERE old.subject_type = 'planter'
                        AND old.subject_id = op.planter_id AND old.participant_slot = op.slot)
                    AND NOT EXISTS (SELECT 1 FROM planter_assignment_points pap
                        JOIN planter_assignments pa ON pa.id = pap.assignment_id
                        WHERE pa.planter_id = op.planter_id AND pap.participant_slot = op.slot
                            AND (to_jsonb(pap)->>'status' IN ('completed', 'planted')
                                 OR to_jsonb(pap)->>'completed_at' IS NOT NULL))
                    ORDER BY op.slot LIMIT 1 FOR UPDATE''', (account['id'],)).fetchone()
            else:
                participant = conn.execute('''SELECT * FROM organization_participants
                    WHERE planter_id = ? AND slot = ? FOR UPDATE''', (account['id'], slot)).fetchone()
            if not participant or participant['username_key']:
                raise GroupAccessError('No unused participant slot is available. If you planted before, use your previous individual code once to keep your existing points.', 409, kind='existing_participant')
        elif mode != 'return':
            raise GroupAccessError('Choose First time or Returning planter.')
        elif not participant:
            raise GroupAccessError('Username not found in this organization. Check its spelling. Returning sign-in never creates a participant.', 401, field='username')
        slot = participant['slot']
        enrolled = conn.execute('''SELECT 1 FROM auth_sessions WHERE subject_type = 'planter'
            AND subject_id = ? AND participant_slot = ? AND planting_schedule_id = ? LIMIT 1''',
            (account['id'], slot, activity['schedule_id'])).fetchone()
        count = conn.execute('''SELECT count(DISTINCT participant_slot) AS total FROM auth_sessions
            WHERE subject_type = 'planter' AND subject_id = ? AND planting_schedule_id = ?''',
            (account['id'], activity['schedule_id'])).fetchone()['total']
        if not enrolled and count >= int(activity['expected_planters'] or account['participant_count']):
            raise GroupAccessError('This activity’s participant places are full. Returning participants can still sign in.', 409)
        active = conn.execute('''SELECT token_hash, browser_key_hash FROM auth_sessions
            WHERE token_hash = ? AND subject_type = 'planter' AND revoked_at IS NULL
            AND expires_at > clock_timestamp()''', (participant['current_session_hash'],)).fetchone()
        if active and not session_details(conn, active['token_hash']):
            active = None
        if active and active['browser_key_hash'] != browser_digest:
            challenge = _switch_challenge(activity, key, browser_digest, active['token_hash'])
            try:
                proof = json.loads(_cipher().decrypt((switch_token or '').encode()))
                valid = (proof['schedule'] == activity['schedule_id'] and proof['digest'] == digest
                    and proof['username'] == key and proof['browser'] == browser_digest
                    and proof['session'] == active['token_hash'] and proof['expires'] > datetime.now(timezone.utc).timestamp())
            except (InvalidToken, ValueError, KeyError, TypeError):
                valid = False
            if not valid:
                raise GroupAccessError('This participant is already signed in. Switch access to this browser?', 409,
                    kind='session_in_use', switch_token=challenge)
        if mode == 'first':
            conn.execute('''UPDATE organization_participants SET participant_username = ?, username_key = ?
                WHERE planter_id = ? AND slot = ?''', (display, key, account['id'], slot))
        conn.execute('''UPDATE auth_sessions SET revoked_at = CURRENT_TIMESTAMP WHERE subject_type = 'planter'
            AND subject_id = ? AND participant_slot = ? AND revoked_at IS NULL''', (account['id'], slot))
        token = planting._create_session('planter', account['id'], slot, connection=conn)
        session_hash = planting._hash_session_token(token)
        conn.execute('''UPDATE auth_sessions SET planting_schedule_id = ?, activity_code_digest = ?,
            browser_key_hash = ? WHERE token_hash = ?''', (activity['schedule_id'], digest, browser_digest, session_hash))
        conn.execute('''UPDATE organization_participants SET current_session_hash = ?
            WHERE planter_id = ? AND slot = ?''', (session_hash, account['id'], slot))
        conn.execute('''INSERT INTO participant_browser_bindings(planter_id, browser_key_hash, participant_slot)
            VALUES (?, ?, ?) ON CONFLICT (planter_id, browser_key_hash) DO UPDATE
            SET participant_slot = excluded.participant_slot''', (account['id'], browser_digest, slot))
        conn.execute('UPDATE planters SET last_login = CURRENT_TIMESTAMP WHERE id = ?', (account['id'],))
        if not session_details(conn, session_hash):
            raise GroupAccessError('This planting activity has ended. Your saved progress is kept.', 403)
        conn.commit()
        account.update(participant_slot=slot, participant_username=display if mode == 'first' else participant['participant_username'],
                       planting_schedule_id=activity['schedule_id'], activity_end_at=activity['end_at'], activity_title=activity['title'])
        return account, token
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def session_details(conn, token_hash):
    row = conn.execute('''SELECT op.participant_username, s.id AS planting_schedule_id,
        s.end_at AS activity_end_at, s.title AS activity_title
        FROM auth_sessions auth JOIN organization_participants op
            ON op.planter_id = auth.subject_id AND op.slot = auth.participant_slot
        JOIN activity_access_codes ac ON ac.schedule_id = auth.planting_schedule_id AND ac.planter_id = auth.subject_id
        JOIN planting_schedules s ON s.id = ac.schedule_id
        JOIN planters p ON p.id = auth.subject_id AND p.organization_id = s.organization_id
        WHERE auth.token_hash = ? AND auth.subject_type = 'planter' AND auth.revoked_at IS NULL
            AND auth.expires_at > clock_timestamp() AND op.current_session_hash = auth.token_hash
            AND ac.revoked_at IS NULL AND ac.code_digest = auth.activity_code_digest
            AND s.status IN ('confirmed', 'in_progress') AND p.status = 'active'
            AND s.start_at <= clock_timestamp() AND s.end_at > clock_timestamp()''', (token_hash,)).fetchone()
    return dict(row) if row else None


def guard_point_write(conn, planter_id, slot, point_ids):
    """Serialize a field write with switching, cancellation, and code replacement."""
    import planting_database as planting
    token_hash = planting._hash_session_token(planter_session_token.get())
    session = conn.execute('SELECT planting_schedule_id FROM auth_sessions WHERE token_hash = ?', (token_hash,)).fetchone()
    if not session or session['planting_schedule_id'] is None:
        raise GroupAccessError('Sign in with the organization code and your username.', 401)
    conn.execute('SELECT id FROM planting_schedules WHERE id = ? FOR SHARE', (session['planting_schedule_id'],)).fetchone()
    conn.execute('SELECT schedule_id FROM activity_access_codes WHERE schedule_id = ? FOR SHARE', (session['planting_schedule_id'],)).fetchone()
    conn.execute('SELECT slot FROM organization_participants WHERE planter_id = ? AND slot = ? FOR UPDATE', (planter_id, slot)).fetchone()
    details = session_details(conn, token_hash)
    owner = conn.execute('SELECT subject_id, participant_slot FROM auth_sessions WHERE token_hash = ?', (token_hash,)).fetchone()
    if not details or owner['subject_id'] != planter_id or owner['participant_slot'] != slot:
        raise GroupAccessError('This login has ended or moved to another browser. Sign in again. Your progress is kept.', 401)
    marks = ','.join('?' for _ in point_ids)
    rows = conn.execute(f'''SELECT pap.id FROM planter_assignment_points pap
        JOIN planter_assignments pa ON pa.id = pap.assignment_id
        WHERE pap.id IN ({marks}) AND pap.released_at IS NULL AND pa.planter_id = ?
            AND pap.participant_slot = ? AND pa.planting_schedule_id = ?''',
        (*point_ids, planter_id, slot, details['planting_schedule_id'])).fetchall()
    if {row['id'] for row in rows} != set(point_ids):
        raise GroupAccessError('Only points assigned to this participant and planting activity can be updated.', 403)


def manage_activity_code(schedule_id, *, action=None, owner_id=None, organization_id=None):
    """Staff read/issue one code; replacing it revokes this activity's sessions."""
    import secrets
    from .planter_codes import ALPHABET
    conn = get_connection()
    try:
        schedule = conn.execute('SELECT * FROM planting_schedules WHERE id = ? FOR UPDATE', (schedule_id,)).fetchone()
        if not schedule:
            raise GroupAccessError('Planting activity not found.', 404)
        schedule = dict(schedule)
        if organization_id is not None and schedule['organization_id'] != organization_id:
            raise GroupAccessError('This activity belongs to another organization.', 403)
        if schedule['appointment_type'] != 'tree_planting':
            raise GroupAccessError('Only tree planting activities have organization access.', 409)
        if owner_id is not None:
            account = _account(conn, owner_id)
            if account['organization_id'] != schedule['organization_id']:
                raise GroupAccessError('This activity belongs to another organization.', 403)
        if action:
            lock_access_emails(conn, schedule_id)
            if schedule['status'] not in {'confirmed', 'in_progress'}:
                raise GroupAccessError('Choose a confirmed or ongoing planting activity.', 409)
            ensure_activity_code(conn, schedule)
            if action in {'replace', 'revoke'}:
                if action == 'replace':
                    raw = ''.join(secrets.choice(ALPHABET) for _ in range(8))
                    conn.execute('''UPDATE activity_access_codes SET code_digest = ?, encrypted_code = ?,
                        revoked_at = NULL, updated_at = CURRENT_TIMESTAMP WHERE schedule_id = ?''',
                        (_digest(raw), _cipher().encrypt(raw.encode()).decode(), schedule_id))
                else:
                    conn.execute('UPDATE activity_access_codes SET revoked_at = CURRENT_TIMESTAMP WHERE schedule_id = ?', (schedule_id,))
                conn.execute('''UPDATE auth_sessions SET revoked_at = CURRENT_TIMESTAMP
                    WHERE subject_type = 'planter' AND planting_schedule_id = ? AND revoked_at IS NULL''', (schedule_id,))
                _refresh_pending_messages(conn, schedule)
            elif action != 'issue':
                raise GroupAccessError('Choose issue, replace or revoke.')
        result = activity_code(conn, schedule_id)
        conn.commit()
        return {'access': result}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def lock_access_emails(conn, schedule_id):
    # Lock pending rows too. A worker uses SKIP LOCKED and cannot start sending
    # the old message between our delivery check and ciphertext refresh.
    website = conn.execute('''SELECT e.status FROM like_appointment_emails e
        JOIN like_appointment_requests r ON r.id = e.request_id
        WHERE r.schedule_id = ? FOR UPDATE OF e''', (schedule_id,)).fetchall()
    manual = conn.execute('''SELECT status FROM activity_schedule_emails
        WHERE schedule_id = ? FOR UPDATE''', (schedule_id,)).fetchall()
    if any(row['status'] == 'sending' for row in [*website, *manual]):
        raise GroupAccessError('An access email is being delivered. Retry after delivery finishes.', 409)


def _refresh_pending_messages(conn, schedule):
    from .appointment_email import _message, _encrypted_message
    from .schedule_notifications import activity_message
    access = activity_code(conn, schedule['id'])
    rows = conn.execute('''SELECT e.id AS email_id, e.message_kind, r.* FROM like_appointment_emails e
        JOIN like_appointment_requests r ON r.id = e.request_id
        WHERE r.schedule_id = ? AND e.status = 'pending' FOR UPDATE OF e''', (schedule['id'],)).fetchall()
    for row in rows:
        message = _message(dict(row), schedule, login={'group_code': access}, kind=row['message_kind'])
        conn.execute('UPDATE like_appointment_emails SET encrypted_message = ? WHERE id = ?', (_encrypted_message(message), row['email_id']))
    row = conn.execute("SELECT id, message_kind FROM activity_schedule_emails WHERE schedule_id = ? AND status = 'pending' FOR UPDATE", (schedule['id'],)).fetchone()
    if row:
        message = activity_message(None, schedule, row['message_kind'], access=access)
        conn.execute('UPDATE activity_schedule_emails SET encrypted_message = ? WHERE id = ?', (_encrypted_message(message), row['id']))
