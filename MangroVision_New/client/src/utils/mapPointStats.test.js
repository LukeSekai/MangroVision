import test from 'node:test';
import assert from 'node:assert/strict';
import { countMapPointStatuses, getMapPointStatus } from './mapPointStats.js';

function assertReconciled(counts) {
  const { mapped, ...statuses } = counts;
  assert.equal(Object.values(statuses).reduce((sum, value) => sum + value, 0), mapped);
}

test('eroded points remain mapped and appear in the unavailable breakdown', () => {
  const points = [
    ...Array.from({ length: 1853 }, () => ({ planting_status: 'planned' })),
    ...Array.from({ length: 245 }, () => ({ assigned_planter_name: 'Organization', assignment_status: 'pending' })),
    ...Array.from({ length: 1100 }, () => ({ planting_status: 'planted' })),
    ...Array.from({ length: 20 }, () => ({ assignment_status: 'skipped' })),
    ...Array.from({ length: 246 }, () => ({ planting_status: 'planned', eroded_unavailable: true })),
  ];
  const counts = countMapPointStatuses(points);
  assert.deepEqual(counts, {
    mapped: 3464, planned: 1853, assigned: 245, planted: 1100, dead: 0, skipped: 20, unavailable: 246,
  });
  assertReconciled(counts);
});

test('overlapping assignment, planting and erosion flags count each point once', () => {
  const cases = [
    [{ assigned_planter_name: 'Organization', assignment_status: 'pending', planting_status: 'planted' }, 'planted'],
    [{ assignment_status: 'completed', planting_status: 'planned' }, 'planted'],
    [{ assignment_status: 'skipped', planting_status: 'planted' }, 'skipped'],
    [{ assignment_status: 'completed', planting_status: 'skipped' }, 'skipped'],
    [{ assigned_planter_name: 'Organization', assignment_status: 'pending', eroded_unavailable: true }, 'unavailable'],
    [{ planting_status: 'planned', inside_eroded_zone: true }, 'unavailable'],
    [{ planting_status: 'planted', eroded_unavailable: true }, 'planted'],
    [{ planting_status: 'skipped', eroded_unavailable: true }, 'skipped'],
    [{ planting_status: 'planted', death_at: '2026-09-18' }, 'dead'],
    [{ assignment_status: 'skipped', death_at: '2026-09-18', eroded_unavailable: true }, 'dead'],
    [{ planting_status: 'planned', assigned_planter_id: 7, assigned_planter_name: '' }, 'assigned'],
  ];
  for (const [point, expectedStatus] of cases) {
    const counts = countMapPointStatuses([point]);
    assert.equal(counts[expectedStatus], 1, JSON.stringify(point));
    assertReconciled(counts);
  }
});

test('removing an eroded zone returns a point to its planting or assignment status', () => {
  const points = [{ planting_status: 'planned' }, { planting_status: 'planned', assigned_planter_name: 'Organization' }];
  const blocked = countMapPointStatuses(points.map((point) => ({ ...point, eroded_unavailable: true })));
  assert.equal(blocked.unavailable, 2);
  const released = countMapPointStatuses(points);
  assert.equal(released.planned, 1);
  assert.equal(released.assigned, 1);
  assert.equal(released.unavailable, 0);
  assertReconciled(blocked);
  assertReconciled(released);
});

test('empty and saved-analysis point data reconcile without assignment fields', () => {
  assert.deepEqual(countMapPointStatuses([]), {
    mapped: 0, planned: 0, assigned: 0, planted: 0, dead: 0, skipped: 0, unavailable: 0,
  });
  const counts = countMapPointStatuses([
    { status: 'planned' }, { status: 'planted' }, { status: 'skipped' }, {},
  ]);
  assert.deepEqual(counts, {
    mapped: 4, planned: 2, assigned: 0, planted: 1, dead: 0, skipped: 1, unavailable: 0,
  });
  assertReconciled(counts);
});

test('API status drives the map and deleted points never inflate its totals', () => {
  const points = [
    { map_status: 'planned' }, { map_status: 'assigned' },
    { map_status: 'planted' }, { map_status: 'dead', planting_status: 'planted' },
    { map_status: 'skipped' }, { map_status: 'unavailable' },
    { map_status: 'dead', deleted_at: '2026-10-07' },
    { map_status: 'planned', is_deleted: true },
  ];
  assert.deepEqual(countMapPointStatuses(points), {
    mapped: 6, planned: 1, assigned: 1, planted: 1, dead: 1, skipped: 1, unavailable: 1,
  });
  assert.deepEqual(points.map(getMapPointStatus), [
    'planned', 'assigned', 'planted', 'dead', 'skipped', 'unavailable', null, null,
  ]);
});
