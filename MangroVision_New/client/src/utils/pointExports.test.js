import test from 'node:test';
import assert from 'node:assert/strict';
import { filterExportPoints, pointExportFilename, pointExportOptions, pointExportPayload,
  pointExportScope, pointExportName } from './pointExports.js';

const points = [
  { id: 1, point_num: 1, analysis_id: 52, image_name: 'Analysis 2', latitude: 10.78, longitude: 122.62,
    source_project_site_id: 4, source_project_site_name: 'North site', species: 'bungalon', map_status: 'planted' },
  { id: 2, point_num: 2, analysis_id: 52, image_name: 'Analysis 2', latitude: 10.79, longitude: 122.63,
    source_project_site_id: 5, source_project_site_name: 'South site', species: 'rhizophora',
    analysis_species: 'bungalon', planted_species: 'Rhizophora', map_status: 'dead', death_at: '2026-10-01' },
  { id: 3, point_num: 1, analysis_id: 60, image_name: 'Analysis 3', latitude: 10.8, longitude: 122.64,
    source_project_site_id: null, source_site_id: null, project_site_name: 'Old site', site_zone_id: 4,
    species: 'bungalon', map_status: 'unavailable', eroded_unavailable: true, availability_reason: 'Inside an eroded zone' },
  { id: 4, point_num: 2, analysis_id: 60, image_name: 'Analysis 3', latitude: 10.81, longitude: 122.65,
    source_project_site_id: null, species: null, map_status: 'planned' },
  { id: 5, point_num: 3, analysis_id: 60, source_project_site_id: 4, species: 'bungalon', deleted_at: '2026-10-02' },
];
const ids = (rows) => rows.map((point) => point.id);

test('exports combine current site, analysis and species filters without changing map points', () => {
  assert.deepEqual(ids(filterExportPoints(points)), [1, 2, 3, 4]);
  assert.deepEqual(ids(filterExportPoints(points, { siteId: '4' })), [1]);
  assert.deepEqual(ids(filterExportPoints(points, { analysisId: '52' })), [1, 2]);
  assert.deepEqual(ids(filterExportPoints(points, { species: 'rhizophora' })), [2]);
  assert.deepEqual(ids(filterExportPoints(points, { siteId: '4', analysisId: '52', species: 'bungalon' })), [1]);
  assert.deepEqual(ids(filterExportPoints(points, { siteId: '4', species: 'rhizophora' })), []);
  assert.equal(points.length, 5);
});

test('released and legacy locations remain selectable without stale assignment attribution or species guesses', () => {
  assert.deepEqual(ids(filterExportPoints(points, { siteId: 'unrecorded' })), [3, 4]);
  assert.deepEqual(ids(filterExportPoints(points, { species: 'unrecorded' })), [4]);
  const [released] = pointExportPayload([points[2]], ['No current project site']).waypoints;
  assert.equal(released.project_site_id, null);
  assert.equal(released.project_site_name, null);
  assert.equal(released.status, 'unavailable');
  assert.equal(released.eroded_unavailable, true);
  assert.equal(released.availability_reason, 'Inside an eroded zone');
});

test('selectors use saved display numbers and distinct IDs even when image names repeat', () => {
  const options = pointExportOptions(points, [{ id: 52, display_number: 2, source_image_name: 'DJI.JPG' },
    { id: 60, display_number: 3, source_image_name: 'DJI.JPG' }]);
  assert.deepEqual(options.analyses, [{ value: '52', label: 'Analysis 2 — DJI.JPG' },
    { value: '60', label: 'Analysis 3 — DJI.JPG' }]);
  assert.equal(options.sites.length, 3);
  assert.equal(options.species.length, 3);
  assert.deepEqual(pointExportScope({ siteId: '4', analysisId: '52', species: 'bungalon' }, options),
    ['North site', 'Analysis 2 — DJI.JPG', 'Bungalon']);
});

test('other recorded species share a neutral choice while files preserve their planting snapshots', () => {
  const planted = { ...points[0], planted_species: 'Legacy species', species: null, analysis_species: 'bungalon' };
  const another = { ...planted, id: 6, planted_species: 'Another legacy species' };
  const planned = { ...planted, map_status: 'planned' };
  assert.deepEqual(ids(filterExportPoints([planted, another, ...points], { species: 'other-recorded' })), [1, 6]);
  assert.deepEqual(pointExportOptions([planted, another]).species,
    [{ value: 'other-recorded', label: 'Other recorded species' }]);
  assert.equal(pointExportPayload([planted], []).waypoints[0].species, 'legacy species');
  assert.equal(pointExportPayload([planned], []).waypoints[0].species, 'bungalon');
});

test('dead seedlings retain erosion membership independently of their primary status', () => {
  const point = { ...points[1], eroded_unavailable: false, inside_eroded_zone: true };
  const [waypoint] = pointExportPayload([point], []).waypoints;
  assert.equal(waypoint.status, 'dead');
  assert.equal(waypoint.eroded_unavailable, true);
  assert.equal(waypoint.species, 'rhizophora');
});

test('all formats receive the same selected rows, status, coordinates and attribution', () => {
  const selected = filterExportPoints(points, { analysisId: '52' });
  const payload = pointExportPayload(selected, ['All sites', 'Analysis 2', 'All species']);
  assert.deepEqual(payload.waypoints.map((point) => point.point_id), [1, 2]);
  assert.equal(payload.waypoints[1].species, 'rhizophora');
  assert.equal(payload.waypoints[1].status, 'dead');
  assert.equal(payload.waypoints[1].lat, points[1].latitude);
  assert.equal(payload.waypoints[1].lon, points[1].longitude);
  assert.equal(payload.waypoints[1].project_site_name, 'South site');
  const date = '2026-10-21T16:30:00Z'; // October 22 in Manila.
  const options = pointExportOptions(points, [{ id: 52, display_number: 2 }]);
  for (const format of ['csv', 'gpx', 'kmz', 'geojson']) {
    assert.equal(pointExportFilename({ siteId: '4' }, format, options, date),
      `MangroVision_north-site_site-4_planting-points_2026-10-22.${format}`);
  }
  assert.equal(pointExportFilename({}, 'csv', options, date), 'MangroVision_all-saved-points_planting-points_2026-10-22.csv');
  assert.match(pointExportName({ analysisId: '52' }, options, date), /MangroVision — Analysis 2 — Planting points — 2026-10-22/);
  assert.doesNotMatch(pointExportFilename({ species: '../../unsafe/name' }, 'csv'), /[\\/]/);
});

test('export filenames, point names and GPS IDs stay readable, unique and free of missing-value labels', () => {
  const selected = [points[0], { ...points[0], id: 6, analysis_id: 60 }, { latitude: 10.8, longitude: 122.7 }];
  const payload = pointExportPayload(selected, ['All points'], [{ id: 52, display_number: 2 }, { id: 60, display_number: 3 }]);
  assert.equal(payload.waypoints[0].name, 'Analysis 2 — Point 001 (ID 1)');
  assert.equal(payload.waypoints[1].name, 'Analysis 3 — Point 001 (ID 6)');
  assert.deepEqual(payload.waypoints.map((point) => point.navigation_name), ['MV-000001', 'MV-000006', 'MV-A0-P003']);
  assert.ok(payload.waypoints.every((point) => !/undefined|null|NaN/.test(point.name)));
  const options = { sites: [{ value: '4', label: '../../North: site? "sample"'.repeat(30) }] };
  const filename = pointExportFilename({ siteId: '4' }, 'kmz', options, '2026-10-22T00:00:00Z');
  assert.match(filename, /^[A-Za-z0-9_-]+\.kmz$/);
  assert.ok(filename.length <= 170);
  assert.ok(pointExportName({ siteId: '4' }, options).length <= 240);
});
