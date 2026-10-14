import test from 'node:test';
import assert from 'node:assert/strict';
import { buildSiteMapPreview, groupSitePreviewPoints } from './siteMapPreview.js';

const layers = {
  satellite: { url: 'https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}', maxNativeZoom: 21 },
  orthophoto: { url: '/tiles/FINAL/{z}/{x}/{y}.png', maxNativeZoom: 22,
    bounds: [[10.779044152226641, 122.62368885567372], [10.782065588714076, 122.6286308280576]] },
};
const area = (lon, lat, span = 0.0002) => ({ type: 'Polygon', coordinates: [[
  [lon, lat], [lon + span, lat], [lon + span, lat + span], [lon, lat + span], [lon, lat],
]] });

test('site map previews use actual geographic tiles and keep the whole boundary centered with padding', () => {
  const preview = buildSiteMapPreview(area(122.625, 10.780), layers);
  assert.ok(preview.tiles.some((tile) => tile.kind === 'satellite'));
  assert.ok(preview.tiles.some((tile) => tile.kind === 'orthophoto' && /^\/tiles\/FINAL\/\d+\/\d+\/\d+\.png$/.test(tile.url)));
  const pairs = [...preview.path.matchAll(/[ML]([\d.]+),([\d.]+)/g)].map((match) => [Number(match[1]), Number(match[2])]);
  assert.ok(pairs.every(([x, y]) => x >= 17.99 && x <= 302.01 && y >= 17.99 && y <= 162.01));
  assert.equal((Math.min(...pairs.map(([x]) => x)) + Math.max(...pairs.map(([x]) => x))) / 2, 160);
  assert.equal((Math.min(...pairs.map(([, y]) => y)) + Math.max(...pairs.map(([, y]) => y))) / 2, 90);
  assert.ok(preview.tiles.every((tile) => tile.x < 320 && tile.x + tile.size > 0 && tile.y < 180 && tile.y + tile.size > 0));
  const neighbor = buildSiteMapPreview(area(122.627, 10.781), layers);
  assert.notDeepEqual(preview.tiles.map((tile) => tile.url), neighbor.tiles.map((tile) => tile.url));
});

test('sites outside drone coverage use their own satellite location without unrelated orthophoto tiles', () => {
  const preview = buildSiteMapPreview(area(122.4, 10.7), layers);
  assert.ok(preview.tiles.length > 0);
  assert.ok(preview.tiles.every((tile) => tile.kind === 'satellite'));
});

test('before and after release share a map frame so retained locations stay aligned', () => {
  const original = area(122.625, 10.780);
  const kept = area(122.62505, 10.78005, 0.00003);
  const points = [{ id: 1, longitude: 122.62506, latitude: 10.78006, planting_status: 'planted' }];
  const before = buildSiteMapPreview(original, { ...layers, points });
  const after = buildSiteMapPreview(kept, { ...layers, points, frameGeometry: original });
  assert.equal(after.zoom, before.zoom);
  assert.deepEqual(after.tiles, before.tiles);
  assert.deepEqual(after.markers, before.markers);
  assert.notEqual(after.path, before.path);
  assert.notEqual(buildSiteMapPreview(kept, { ...layers, points }).zoom, before.zoom);
});

test('map boundaries preserve polygon holes and separate parts of a multi-polygon', () => {
  const polygon = area(122.625, 10.780);
  polygon.coordinates.push(area(122.62505, 10.78005, 0.00003).coordinates[0]);
  const preview = buildSiteMapPreview({ type: 'MultiPolygon', coordinates: [polygon.coordinates, area(122.6253, 10.7803).coordinates] }, layers);
  assert.equal((preview.path.match(/M/g) || []).length, 3);
  assert.equal((preview.path.match(/Z/g) || []).length, 3);
});

test('tiny sites use high-resolution drone tiles while satellite tiles stay within their native zoom', () => {
  const preview = buildSiteMapPreview(area(122.625, 10.780, 0.000005), layers);
  assert.equal(preview.zoom, 22);
  assert.ok(preview.tiles.filter((tile) => tile.kind === 'satellite').every((tile) => tile.url.endsWith('z=21') && tile.size === 512));
  assert.ok(preview.tiles.filter((tile) => tile.kind === 'orthophoto').every((tile) => tile.size === 256));
});

test('missing, degenerate or invalid site geometry has no fabricated map preview', () => {
  for (const geometry of [null, {}, { type: 'Polygon', coordinates: [] },
    { type: 'Point', coordinates: [122, 10] }, area(122, 10, 0), area(200, 10), area(122, 95)]) {
    assert.equal(buildSiteMapPreview(geometry, layers), null);
  }
});

test('point previews use resolved site ownership, keep deaths and erosion, and exclude duplicate or invalid locations', () => {
  const location = { longitude: 122.6251, latitude: 10.7801 };
  const groups = groupSitePreviewPoints([
    { id: 1, ...location, source_project_site_id: 7, project_site_id: 9, source_organization_id: 3 },
    { id: '1', ...location, source_project_site_id: '7' },
    { id: 2, ...location, source_site_id: 7, death_at: '2026-10-12' },
    { id: 3, ...location, source_project_site_id: 7, eroded_unavailable: true },
    { id: 4, ...location, source_project_site_id: 9, source_organization_id: 3 },
    { id: 5, ...location, source_project_site_id: 7, deleted_at: '2026-10-12' },
    { id: 6, ...location, source_project_site_id: 7, inside_visible_map: false },
    { id: 7, ...location, source_organization_id: 3, project_site_id: 7 },
    { id: 8, ...location, source_project_site_id: 7, latitude: null },
    { id: 9, ...location, source_project_site_id: 7, longitude: NaN },
  ]);
  assert.deepEqual(groups.get('7').map((point) => point.id), [1, 2, 3]);
  assert.deepEqual(groups.get('9').map((point) => point.id), [4]);
  assert.equal(groups.size, 2);
});

test('planting dots use the boundary’s map projection and main-map species and lifecycle colors', () => {
  const location = { longitude: 122.6251, latitude: 10.7801 };
  const preview = buildSiteMapPreview(area(122.625, 10.780), { ...layers, points: [
    { id: 1, ...location, species: 'rhizophora' },
    { id: 2, ...location, assignment_status: 'pending', species: 'rhizophora' },
    { id: 3, ...location, planting_status: 'planted', species: 'rhizophora' },
    { id: 4, ...location, death_at: '2026-10-12', planting_status: 'planted' },
    { id: 5, ...location, eroded_unavailable: true },
    { id: 6, ...location, planting_status: 'skipped' },
    { id: 7, ...location, deleted_at: '2026-10-12' },
    { id: 8, ...location, latitude: null },
    { id: 9, ...location, longitude: 122.7 },
  ] });
  assert.deepEqual(preview.markers.map(({ id, status, color }) => ({ id, status, color })), [
    { id: 1, status: 'planned', color: '#db2777' },
    { id: 2, status: 'assigned', color: '#2563eb' },
    { id: 3, status: 'planted', color: '#eab308' },
    { id: 4, status: 'dead', color: '#7f1d1d' },
    { id: 5, status: 'unavailable', color: '#f97316' },
    { id: 6, status: 'skipped', color: '#9ca3af' },
  ]);
  assert.ok(preview.markers.every(({ x, y }) => Math.abs(x - 160) < 0.001 && Math.abs(y - 90) < 0.001));
});
