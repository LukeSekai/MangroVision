"""Preview or prepare shared codes for existing future planting activities.

Never sends mail or prints codes. Preparation requires a reviewed fingerprint,
verified encrypted backup, and a new encrypted handover in backups/ or tmp/.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from planting_database import _get_connection, append_activity
from mangrovision_db.appointment_email import _cipher
from mangrovision_db.group_access import ensure_activity_code, _refresh_pending_messages, lock_access_emails


def preview(conn):
    rows = [dict(row) for row in conn.execute('''SELECT id, organization_id, organization,
        title, start_at, end_at, expected_planters, status, appointment_type
        FROM planting_schedules WHERE appointment_type = 'tree_planting'
            AND status IN ('confirmed', 'in_progress') AND end_at > clock_timestamp()
        ORDER BY id''').fetchall()]
    fingerprint = hashlib.sha256(json.dumps(rows, sort_keys=True).encode()).hexdigest()
    return rows, fingerprint


def private_path(value):
    path = Path(value).resolve()
    if not any(path.is_relative_to(ROOT / folder) for folder in ('tmp', 'backups')):
        raise ValueError('Private artifacts must stay under backups/ or tmp/.')
    return path


def prepare(conn, expected_fingerprint):
    rows, fingerprint = preview(conn)
    if fingerprint != expected_fingerprint:
        raise ValueError('Activities changed. Review a fresh preview.')
    result = []
    for row in rows:
        current = conn.execute('SELECT * FROM planting_schedules WHERE id = ? FOR UPDATE', (row['id'],)).fetchone()
        if not current or any(current[key] != value for key, value in row.items()):
            raise ValueError('An activity changed. Review a fresh preview.')
        lock_access_emails(conn, row['id'])
        access = ensure_activity_code(conn, dict(current))
        _refresh_pending_messages(conn, dict(current))
        append_activity(conn, action='activity.group_access_prepared', actor_type='system',
            organization_id=row['organization_id'], summary='Prepared shared organization access for a planting activity.',
            details={'schedule_id': row['id'], 'expected_participants': row['expected_planters']})
        result.append({'organization': row['organization'], 'access': access})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepare', action='store_true')
    parser.add_argument('--expected-fingerprint')
    parser.add_argument('--verified-backup')
    parser.add_argument('--encrypted-handover')
    args = parser.parse_args()
    conn = None
    try:
        conn = _get_connection()
        rows, fingerprint = preview(conn)
        if not args.prepare:
            print(json.dumps({'activities': len(rows), 'schedule_ids': [row['id'] for row in rows],
                'expected_participants': sum(row['expected_planters'] or 0 for row in rows),
                'fingerprint': fingerprint, 'database_changed': False}))
            return 0
        if not all((args.expected_fingerprint, args.verified_backup, args.encrypted_handover)):
            raise ValueError('Preparation requires the reviewed fingerprint, verified backup, and encrypted handover.')
        backup, output = private_path(args.verified_backup), private_path(args.encrypted_handover)
        if backup.suffix != '.enc' or not backup.is_file() or backup.stat().st_size == 0:
            raise ValueError('The verified encrypted backup is missing.')
        if output.exists():
            raise ValueError('Choose a new handover path; existing private files must not be replaced.')
        result = prepare(conn, args.expected_fingerprint)
        output.parent.mkdir(parents=True, exist_ok=True)
        with output.open('xb') as artifact:
            artifact.write(_cipher().encrypt(json.dumps(result).encode()))
        conn.commit()
        print(f'Prepared {len(result)} activity codes. Encrypted handover saved. No emails sent.')
        return 0
    except Exception as error:
        if conn:
            conn.rollback()
        print(f'Group access preparation did not finish ({type(error).__name__}); private details suppressed.', file=sys.stderr)
        return 1
    finally:
        if conn:
            conn.close()


if __name__ == '__main__':
    raise SystemExit(main())
