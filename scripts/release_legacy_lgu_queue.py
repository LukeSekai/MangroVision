"""Convert unfinished staff replacement reservations into ordinary land releases.

Run without arguments to preview counts; pass --apply after reviewing them.
Completed plantings and organization assignments are never changed.
"""
from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import planting_database as db
from mangrovision_db.site_actions import release_dead_locations


def pending_reservations():
    conn = db._get_connection()
    try:
        rows = [dict(row) for row in conn.execute('''SELECT r.planting_event_id, r.version,
            r.approved_by, pe.planting_point_id AS id, pp.latitude, pp.longitude,
            pe.project_site_id AS source_site_id
            FROM replanting_requests r JOIN planting_events pe ON pe.id = r.planting_event_id
            JOIN planting_points pp ON pp.id = pe.planting_point_id
            WHERE r.assignment_id IS NULL AND r.replacement_event_id IS NULL
              AND pp.deleted_at IS NULL AND pp.status = 'planned' AND pp.death_at IS NULL
              AND EXISTS (SELECT 1 FROM point_death_records d WHERE d.planting_event_id = pe.id)
              AND NOT EXISTS (SELECT 1 FROM planter_assignment_points pap
                  WHERE pap.planting_point_id = pp.id AND pap.released_at IS NULL)
              AND COALESCE(pe.closure_reason, '') <> 'completion_reversed'
              AND NOT EXISTS (SELECT 1 FROM planting_events newer
                  WHERE newer.planting_point_id = pp.id AND (newer.planted_at, newer.id) > (pe.planted_at, pe.id)
                    AND COALESCE(newer.closure_reason, '') <> 'completion_reversed')
            ORDER BY r.planting_event_id''').fetchall()]
        return [row for row in db._resolve_point_project_sites(conn, rows) if row['source_site_id'] is not None]
    finally:
        conn.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    try:
        rows = pending_reservations()
        print(f'Unfinished reservations still inside active sites: {len(rows)}')
        if not args.apply:
            return
        conn = db._get_connection()
        try:
            original_history = db._project_site_planting_history(conn)
        finally:
            conn.close()
        groups = defaultdict(list)
        for row in rows:
            groups[row['approved_by']].append(row)
        released = 0
        for user_id, group in groups.items():
            for start in range(0, len(group), 500):
                batch = group[start:start + 500]
                result = release_dead_locations([row['planting_event_id'] for row in batch],
                    {str(row['planting_event_id']): row['version'] for row in batch}, user_id, prepared=True)
                released += result['released_count']
        print(f'Released locations: {released}; planting and death history preserved.')
        print(f'Unfinished reservations still inside active sites: {len(pending_reservations())}')
        conn = db._get_connection()
        try:
            if db._project_site_planting_history(conn) != original_history:
                raise ValueError('Historical site outcomes changed during conversion; inspect concurrent activity.')
        finally:
            conn.close()
        print('Historical site planting counts, deaths, and survival unchanged: yes')
    except Exception as error:
        # Database errors can include connection strings and private values.
        print(f'Conversion stopped ({type(error).__name__}); no connection details displayed.', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
