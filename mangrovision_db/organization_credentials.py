"""Reviewed, atomic conversion of existing organization logins.

This operation changes only username/password_hash. It does not create accounts,
redistribute points, reset devices, revoke sessions, or send messages.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from uuid import uuid4

from .readable_credentials import organization_password, organization_slug

RULE = 'organization-readable-v1'
ACTION = 'organization_account.credentials_standardized'


def _inventory(conn, *, lock=False):
    rows = conn.execute('''SELECT p.id, p.organization_id, o.name AS organization,
        p.username, p.password_hash, p.status, p.participant_count, p.merged_into_planter_id
        FROM planters p LEFT JOIN organizations o ON o.id = p.organization_id
        ORDER BY p.id''' + (' FOR UPDATE OF p' if lock else '')).fetchall()
    return [dict(row) for row in rows]


def _fingerprint(rows):
    return hashlib.sha256(json.dumps(rows, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def _converted_ids(conn):
    rows = conn.execute('''SELECT details FROM activity_logs WHERE action = ?
        AND details ->> 'credential_rule' = ?''', (ACTION, RULE)).fetchall()
    return {int((json.loads(row['details']) if isinstance(row['details'], str) else row['details'])['planter_id'])
            for row in rows}


def prepare_conversion(conn):
    """Read-only plan. The caller must encrypt its passwords before saving it."""
    rows = _inventory(conn)
    converted = _converted_ids(conn)
    occupied = {row['username'].lower(): row['id'] for row in rows if row['username']}
    entries = []
    skipped = []
    for row in rows:
        reason = None
        if row['merged_into_planter_id'] is not None:
            reason = 'Merged historical account'
        elif not row['organization_id'] or not row['organization']:
            reason = 'No linked organization'
        elif not row['username'] or not row['password_hash']:
            reason = 'Organization has no login yet'
        elif row['id'] in converted:
            reason = 'Already converted'
        if reason:
            skipped.append({'planter_id': row['id'], 'organization': row['organization'], 'reason': reason})
            continue
        base = organization_slug(row['organization'], row['organization_id'])
        username = base
        suffix = 2
        while username in occupied and occupied[username] != row['id']:
            username = f'{base}-{suffix}'
            suffix += 1
        if occupied.get(row['username'].lower()) == row['id']:
            occupied.pop(row['username'].lower())
        occupied[username] = row['id']
        entries.append({'planter_id': row['id'], 'organization_id': row['organization_id'],
                        'organization': row['organization'], 'status': row['status'],
                        'old_username': row['username'], 'username': username,
                        'password': organization_password(row['organization']),
                        'expected_state': _fingerprint([row])})
    return {'credential_rule': RULE, 'batch_id': str(uuid4()),
            'prepared_at': datetime.now(timezone.utc).isoformat(), 'accounts': entries, 'skipped': skipped}


def public_preview(plan):
    return {'credential_rule': plan['credential_rule'], 'batch_id': plan['batch_id'],
            'prepared_at': plan['prepared_at'],
            'accounts': [{key: entry[key] for key in ('planter_id', 'organization', 'status', 'old_username', 'username')}
                         for entry in plan['accounts']], 'skipped': plan['skipped']}


def apply_conversion(conn, plan):
    """Caller owns commit/rollback; any conflict cancels the entire batch."""
    from planting_database import _hash_password, append_activity

    if plan.get('credential_rule') != RULE or not plan.get('batch_id'):
        raise ValueError('Unsupported organization login conversion plan.')
    # Serialize account changes, including staff creation and website approval.
    conn.execute("SET LOCAL lock_timeout = '10s'")
    conn.execute('LOCK TABLE organizations IN SHARE MODE')
    conn.execute('LOCK TABLE planters IN SHARE ROW EXCLUSIVE MODE')
    done = conn.execute('''SELECT details FROM activity_logs WHERE action = ?
        AND details ->> 'batch_id' = ?''', (ACTION, plan['batch_id'])).fetchall()
    if done:
        completed = {int((json.loads(row['details']) if isinstance(row['details'], str) else row['details'])['planter_id'])
                     for row in done}
        if completed != {entry['planter_id'] for entry in plan['accounts']}:
            raise ValueError('This conversion batch has incomplete audit records. Review it before continuing.')
        return {'updated': 0, 'already_applied': True}
    rows = {row['id']: row for row in _inventory(conn, lock=True)}
    converted = _converted_ids(conn)
    planned_ids = [entry['planter_id'] for entry in plan['accounts']]
    if len(set(planned_ids)) != len(planned_ids):
        raise ValueError('The conversion plan contains duplicate accounts.')
    proposed = {entry['planter_id']: entry['username'].lower() for entry in plan['accounts']}
    if len(set(proposed.values())) != len(proposed):
        raise ValueError('The conversion plan contains duplicate usernames.')
    for entry in plan['accounts']:
        row = rows.get(entry['planter_id'])
        if not row or entry['planter_id'] in converted or _fingerprint([row]) != entry['expected_state']:
            raise ValueError('An organization account changed after preparation. Prepare and review a fresh plan.')
        if not row['username'] or not row['password_hash'] or not row['organization_id'] or row['merged_into_planter_id']:
            raise ValueError('This account is not eligible for login conversion.')
        occupied = conn.execute('SELECT id FROM planters WHERE lower(username) = lower(?) AND id <> ?',
                                (entry['username'], entry['planter_id'])).fetchone()
        if occupied and proposed.get(occupied['id'], entry['username'].lower()) == entry['username'].lower():
            raise ValueError('A planned username is now in use. Prepare and review a fresh plan.')
        pending = conn.execute('''SELECT e.id FROM like_appointment_emails e
            JOIN like_appointment_requests r ON r.id = e.request_id
            JOIN planting_schedules s ON s.id = r.schedule_id
            WHERE s.organization_id = ? AND e.status <> 'sent' FOR UPDATE OF e''',
                               (entry['organization_id'],)).fetchone()
        if pending:
            raise ValueError('An organization has an unsent booking email. Finish delivery before changing its login.')
    for entry in plan['accounts']:
        conn.execute('UPDATE planters SET username = ?, password_hash = ? WHERE id = ?',
                     (entry['username'], _hash_password(entry['password']), entry['planter_id']))
        append_activity(conn, action=ACTION, actor_type='system',
                        organization_id=entry['organization_id'],
                        summary=f"Organization login updated for {entry['organization']}.",
                        details={'planter_id': entry['planter_id'], 'credential_rule': RULE,
                                 'batch_id': plan['batch_id'], 'old_username': entry['old_username'],
                                 'new_username': entry['username']})
    return {'updated': len(plan['accounts']), 'already_applied': False}
