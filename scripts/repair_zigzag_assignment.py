"""Preview or repair one untouched assignment's participant point locations.

Never moves coordinates, changes counts, or edits assignments with planting
history. --apply saves a recovery snapshot before updating participant_slot.
"""
import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from sqlalchemy import text
from mangrovision_db import get_engine
from mangrovision_db.planting_order import zigzag_assignment_points


def repair_plan(points):
    if not points or any(
        p['assignment_status'] != 'pending' or p['point_status'] != 'planned'
        or p['completed_at'] is not None
        or (p['status_changed_at'] is not None and p['status_changed_at'] != p['assigned_at'])
        or p['death_at'] is not None or p['deleted_at'] is not None or p['has_history']
        for p in points
    ):
        raise ValueError('Repair requires an entirely untouched, pending assignment with no planting history.')
    counts = Counter(p['participant_slot'] for p in points)
    if any(slot is None or slot < 1 for slot in counts):
        raise ValueError('All points must already belong to valid participant slots.')
    slots = [slot for slot, count in sorted(counts.items()) for _ in range(count)]
    ordered = zigzag_assignment_points(points)
    return [{**point, 'new_slot': slot} for point, slot in zip(ordered, slots)]


def plot_plan(plan, path):
    import math
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    lat0 = min(p['latitude'] for p in plan)
    lon0 = min(p['longitude'] for p in plan)
    scale = 111_320
    x = [(p['longitude'] - lon0) * scale * math.cos(math.radians(lat0)) for p in plan]
    y = [(p['latitude'] - lat0) * scale for p in plan]
    fig, axes = plt.subplots(1, 2, figsize=(12, 7), layout='constrained')
    for ax, key, title in zip(axes, ('participant_slot', 'new_slot'), ('Before: blocks of points', 'After: zigzag point locations')):
        for slot in sorted({p[key] for p in plan}):
            indexes = [i for i, p in enumerate(plan) if p[key] == slot]
            ax.scatter([x[i] for i in indexes], [y[i] for i in indexes], s=100,
                       color=plt.get_cmap('tab10')((slot - 1) % 10), label=f'Participant {slot}')
            for i in indexes:
                ax.text(x[i], y[i], str(slot), ha='center', va='center', color='white', fontsize=7)
        ax.set(title=title, xlabel='East (metres)', ylabel='North (metres)', aspect='equal')
        ax.grid(alpha=.15)
    fig.suptitle(f'Same {len(plan)} planting locations; participant counts preserved\nColours show ownership; no route or visit-order lines', fontsize=13)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assignment-id', type=int, required=True)
    parser.add_argument('--apply', action='store_true')
    parser.add_argument('--plot', action='store_true')
    args = parser.parse_args()
    with get_engine().begin() as c:
        if not args.apply:
            c.execute(text('SET TRANSACTION READ ONLY'))
        c.execute(text("SET LOCAL lock_timeout = '5s'"))
        c.execute(text("SET LOCAL statement_timeout = '15s'"))
        query = '''SELECT pa.id, pa.status, pl.participant_count, o.name
            FROM mangrovision.planter_assignments pa JOIN mangrovision.planters pl ON pl.id=pa.planter_id
            JOIN mangrovision.organizations o ON o.id=pl.organization_id WHERE pa.id=:id'''
        assignment = c.execute(text(query + (' FOR UPDATE OF pl, pa' if args.apply else '')),
                               {'id': args.assignment_id}).mappings().one()
        if assignment['status'] != 'active':
            raise ValueError('Only an active assignment may be repaired.')
        query = '''SELECT pp.id, pp.analysis_id, pp.point_num, pp.latitude, pp.longitude,
            pp.status AS point_status, pp.death_at, pp.deleted_at,
            pap.id AS assignment_point_id, pap.participant_slot, pap.sequence_num,
            pap.status AS assignment_status, pap.completed_at, pap.status_changed_at, pap.assigned_at,
            EXISTS(SELECT 1 FROM mangrovision.planting_events pe WHERE pe.planting_point_id=pp.id) AS has_history
            FROM mangrovision.planter_assignment_points pap
            JOIN mangrovision.planting_points pp ON pp.id=pap.planting_point_id
            WHERE pap.assignment_id=:id ORDER BY pp.id'''
        points = [dict(r) for r in c.execute(text(query + (' FOR UPDATE OF pp, pap' if args.apply else '')),
                                            {'id': args.assignment_id}).mappings()]
        plan = repair_plan(points)
        if max(p['new_slot'] for p in plan) > assignment['participant_count']:
            raise ValueError('A participant slot exceeds the account participant count.')
        changes = [p for p in plan if p['participant_slot'] != p['new_slot']]
        output = ROOT / 'validation_outputs'
        output.mkdir(exist_ok=True)
        if args.apply and changes:
            snapshot = output / f'zigzag-assignment-{args.assignment_id}-before-{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}.json'
            with snapshot.open('x', encoding='utf-8') as handle:
                json.dump({'assignment': dict(assignment), 'points': plan}, handle, indent=2, default=str)
            print('Recovery snapshot:', snapshot)
            c.execute(text('''UPDATE mangrovision.planter_assignment_points
                SET participant_slot=:slot
                WHERE id=:id AND assignment_id=:assignment_id AND status='pending' '''),
                [{'slot': p['new_slot'], 'id': p['assignment_point_id'], 'assignment_id': args.assignment_id} for p in changes])
            verified = dict(c.execute(text('''SELECT id, participant_slot FROM mangrovision.planter_assignment_points
                WHERE assignment_id=:id'''), {'id': args.assignment_id}).all())
            assert verified == {p['assignment_point_id']: p['new_slot'] for p in plan}
        print(json.dumps({'assignment': args.assignment_id, 'organization': assignment['name'],
                          'points': len(points), 'changed_owners': len(changes),
                          'shares': dict(sorted(Counter(p['new_slot'] for p in plan).items())),
                          'applied': args.apply}))
    if args.plot:
        path = output / f'zigzag-assignment-{args.assignment_id}.png'
        plot_plan(plan, path)
        print('Preview:', path)


if __name__ == '__main__':
    main()
