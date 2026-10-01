"""Apply the user's confirmed 50-additional-deaths correction and snapshot old ages.

Run without --apply to preview. Changes are atomic and require the exact known
record identity/counts. No measured heights or historical manual stages change.
"""
import argparse
import json
import sys
from datetime import datetime, time, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from sqlalchemy import text
from mangrovision_db.compat import get_engine
from mangrovision_db.monitoring_progress import age_snapshot, MANILA


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--apply', action='store_true')
    args = parser.parse_args()
    with get_engine().connect() as conn:
        transaction = conn.begin()
        try:
            conn.execute(text("SET LOCAL lock_timeout = '5s'"))
            conn.execute(text("SET LOCAL statement_timeout = '30s'"))
            target = conn.execute(text('SELECT * FROM mangrovision.organization_monitoring_records WHERE id=29 FOR UPDATE')).mappings().one()
            assert target['organization_id'] == 9
            assert target['monitored_at'].astimezone(MANILA).date().isoformat() == '2026-09-12'
            assert (target['alive_count'], target['dead_count']) in {(283, 50), (148, 185)}
            baseline = conn.execute(text('SELECT alive_count, dead_count FROM mangrovision.organization_monitoring_records WHERE id=28 AND organization_id=9')).mappings().one()
            assert (baseline['alive_count'], baseline['dead_count']) == (198, 135)
            rows = conn.execute(text('SELECT id,organization_id,monitored_at,alive_count,dead_count FROM mangrovision.organization_monitoring_records WHERE growth_snapshot IS NULL OR id=29 ORDER BY id')).mappings().all()
            for row in rows:
                observed = row['monitored_at'].astimezone(MANILA)
                cutoff = datetime.combine(observed.date() + timedelta(days=1), time.min, tzinfo=MANILA)
                plantings = [dict(p) for p in conn.execute(text("""SELECT pe.id, pe.planted_at, pe.species
                    FROM mangrovision.planting_events pe JOIN mangrovision.planters p ON p.id=pe.planter_id
                    WHERE p.organization_id=:org AND pe.planted_at < :cutoff
                      AND COALESCE(pe.closure_reason,'') <> 'completion_reversed'"""),
                    {'org': row['organization_id'], 'cutoff': cutoff}).mappings()]
                snapshot = age_snapshot(plantings, observed, row['alive_count'] + row['dead_count'])
                if row['alive_count'] == 0:
                    snapshot['label'] = 'No living seedlings'
                if row['id'] == 29:
                    assert len(plantings) == 333
                    count_snapshot = {'event_ids': sorted(p['id'] for p in plantings), 'total_planted': 333,
                                      'previous_dead_count': 135, 'new_planted_count': 0,
                                      'correction': 'User confirmed 50 additional deaths on 2026-09-12; previous stored alive=283, dead=50.'}
                    conn.execute(text('''UPDATE mangrovision.organization_monitoring_records SET
                        alive_count=148, dead_count=185, new_dead_count=50, alive_before_count=198,
                        baseline_record_id=28, count_snapshot=CAST(:counts AS jsonb), growth_snapshot=CAST(:growth AS jsonb)
                        WHERE id=29 AND organization_id=9'''), {'counts':json.dumps(count_snapshot), 'growth':json.dumps(snapshot)})
                else:
                    conn.execute(text('UPDATE mangrovision.organization_monitoring_records SET growth_snapshot=CAST(:growth AS jsonb) WHERE id=:id AND growth_snapshot IS NULL'), {'growth':json.dumps(snapshot), 'id':row['id']})
            print(f'Prepared {len(rows)} age snapshots; record 29: 185 cumulative dead, 148 alive, 50 newly dead.')
            if args.apply:
                transaction.commit()
                print('Changes committed.')
            else:
                transaction.rollback()
                print('Preview only: all changes rolled back.')
        except Exception:
            transaction.rollback()
            raise


if __name__ == '__main__':
    main()
