import test from 'node:test';
import assert from 'node:assert/strict';
import { filterMapPoints } from './mapPointFilters.js';

const points = [{ id: 1, death_at: null }, { id: 2, death_at: '2026-09-18' }, { id: 3 }];

test('map pages keep identified dead plants visible alongside other statuses', () => {
  assert.deepEqual(filterMapPoints(points), points);
  assert.deepEqual(filterMapPoints(points, { deadOnly: true }).map((p) => p.id), [2]);
});

test('recording a death keeps the location and site ownership until an explicit release', () => {
  const planted = { id: 10, planting_status: 'planted', source_project_site_id: 5 };
  const dead = { ...planted, death_at: '2026-10-14', map_status: 'dead' };
  assert.deepEqual(filterMapPoints([dead]), [dead]);
  assert.equal(filterMapPoints([dead])[0].source_project_site_id, planted.source_project_site_id);
  assert.deepEqual(filterMapPoints([{ id: 11, map_status: 'dead' }], { deadOnly: true }).map((p) => p.id), [11]);
});

test('explicitly deleted points stay hidden even if their death history exists', () => {
  assert.deepEqual(filterMapPoints([{ id: 2, death_at: '2026-09-18', deleted_at: '2026-10-14' }]), []);
});

test('a released point leaves the dead filter and returns to available map points', () => {
  const released = [{ id: 2, death_at: null, planting_status: 'planned', assigned_organization_id: null }];
  assert.deepEqual(filterMapPoints(released, { deadOnly: true }), []);
  assert.deepEqual(filterMapPoints(released), released);
});
