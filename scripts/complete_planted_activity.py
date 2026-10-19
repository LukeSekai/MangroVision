"""Reconcile one fully planted activity. Defaults to a read-only preview."""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mangrovision_db.activity_rules import complete_fully_planted_activity
from mangrovision_db.compat import get_connection


def planting_fingerprint(conn, schedule_id):
    queries = [
        'SELECT * FROM planter_assignments WHERE planting_schedule_id = ? ORDER BY id',
        '''SELECT p.* FROM planter_assignment_points p JOIN planter_assignments a ON a.id = p.assignment_id
           WHERE a.planting_schedule_id = ? ORDER BY p.id''',
        '''SELECT p.* FROM planting_events p JOIN planter_assignments a ON a.id = p.assignment_id
           WHERE a.planting_schedule_id = ? ORDER BY p.id''',
        'SELECT * FROM activity_schedule_emails WHERE schedule_id = ? ORDER BY id',
        '''SELECT e.* FROM like_appointment_emails e JOIN like_appointment_requests r ON r.id = e.request_id
           WHERE r.schedule_id = ? ORDER BY e.id''',
    ]
    records = [[dict(row) for row in conn.execute(query, (schedule_id,)).fetchall()] for query in queries]
    return hashlib.sha256(json.dumps(records, sort_keys=True, default=str).encode()).hexdigest()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--schedule-id', type=int, required=True)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    conn = get_connection()
    try:
        if not args.apply:
            conn.execute('SET TRANSACTION READ ONLY')
        conn.execute("SET LOCAL statement_timeout = '15s'")
        current = conn.execute('SELECT id, status, appointment_type FROM planting_schedules WHERE id = ?'
                               + (' FOR UPDATE' if args.apply else ''), (args.schedule_id,)).fetchone()
        if not current:
            raise ValueError('Planting activity not found.')
        counts = dict(conn.execute('''SELECT COUNT(p.id) AS assigned_points,
            COUNT(p.id) FILTER (WHERE p.status = 'completed') AS planted_points
            FROM planter_assignments a JOIN planter_assignment_points p ON p.assignment_id = a.id
            WHERE a.planting_schedule_id = ?''', (args.schedule_id,)).fetchone())
        report = {'schedule_id': args.schedule_id, 'before_status': current['status'], **counts,
                  'eligible': current['appointment_type'] == 'tree_planting'
                  and current['status'] in {'confirmed', 'in_progress'}
                  and counts['assigned_points'] > 0 and counts['assigned_points'] == counts['planted_points']}
        if args.apply:
            before = planting_fingerprint(conn, args.schedule_id)
            changed = complete_fully_planted_activity(conn, args.schedule_id)
            after = planting_fingerprint(conn, args.schedule_id)
            if before != after:
                raise RuntimeError('Planting or email records changed; correction rolled back.')
            report.update(changed=changed is not None, planting_and_email_records_unchanged=True,
                          after_status=conn.execute('SELECT status FROM planting_schedules WHERE id = ?',
                                                    (args.schedule_id,)).fetchone()['status'])
            conn.commit()
            folder = ROOT / 'run_logs'
            folder.mkdir(exist_ok=True)
            filename = f"activity-completion-{args.schedule_id}-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json"
            (folder / filename).write_text(json.dumps(report, indent=2), encoding='utf-8')
        else:
            conn.rollback()
        print(json.dumps(report))
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


if __name__ == '__main__':
    try:
        main()
    except Exception as error:
        print(f'Activity correction failed ({type(error).__name__}); no private details printed.', file=sys.stderr)
        sys.exit(1)
