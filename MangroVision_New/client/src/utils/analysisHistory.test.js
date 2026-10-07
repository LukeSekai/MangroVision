import test from 'node:test';
import assert from 'node:assert/strict';
import { canViewSavedAnalysisOnMap, savedAnalysisMapContext } from './analysisMapContext.js';
import {
  DEFAULT_HISTORY_FILTERS, analysisDateOptions, canopyCoverage, filterAnalysisHistory, formatAnalysisDate, formatAnalysisNumber,
} from './analysisHistory.js';

const filter = (rows, changes) => filterAnalysisHistory(rows, { ...DEFAULT_HISTORY_FILTERS, ...changes }, '2026-10-07');
const ids = rows => rows.map(row => row.id);
const rows = [
  { id: 1, image_name: 'Analysis 1', source_image_name: 'QJBJ.JPG', analyzed_at: '2026-10-06T18:30:00Z', plantable_area_m2: 0, canopy_area_m2: 0, canopy_coverage_pct: 0 },
  { id: 2, image_name: 'Analysis 2', source_image_name: 'ITBH.JPG', analyzed_at: '2026-10-01T08:00:00+08:00', plantable_area_m2: 99.9, canopy_coverage_pct: 24.99 },
  { id: 3, image_name: 'Analysis 3', source_image_name: 'JTZS.JPG', analyzed_at: '2026-09-30T18:00:00Z', plantable_area_m2: 100, canopy_coverage_pct: 25 },
  { id: 4, image_name: 'Analysis 4', source_image_name: 'PZQN.JPG', analyzed_at: '2026-09-08T00:00:00+08:00', plantable_area_m2: 499.9, canopy_coverage_pct: 50 },
  { id: 5, image_name: 'Analysis 5', source_image_name: 'Older.JPG', analyzed_at: '2026-09-07T23:59:59+08:00', plantable_area_m2: 500, canopy_coverage_pct: 75 },
  { id: 6, image_name: 'Legacy', analyzed_at: 'invalid', plantable_area_m2: null, canopy_coverage_pct: null },
];

test('date filters use Philippine calendar days with inclusive boundaries', () => {
  assert.deepEqual(ids(filter(rows, { date: 'today' })), [1]);
  assert.deepEqual(ids(filter(rows, { date: '7' })), [1, 2, 3]);
  assert.deepEqual(ids(filter(rows, { date: '30' })), [1, 2, 3, 4]);
  assert.deepEqual(ids(filter(rows, { date: '2026-10' })), [1, 2, 3]);
  assert.deepEqual(analysisDateOptions(rows), [['2026-10', 'October 2026'], ['2026-09', 'September 2026']]);
  assert.deepEqual(filter([{ id: 7 }], { date: 'today' }), []);
  assert.equal(formatAnalysisDate(undefined), 'Date unavailable');
});

test('area and coverage remain available as sorting choices without filtering records out', () => {
  assert.deepEqual(DEFAULT_HISTORY_FILTERS, { search: '', date: 'all', sort: 'newest' });
  assert.deepEqual(ids(filter(rows, { sort: 'plantable-asc' })), [1, 2, 3, 4, 5, 6]);
  assert.deepEqual(ids(filter(rows, { sort: 'canopy-desc' })), [5, 4, 3, 2, 1, 6]);
});

test('filename search intersects filters and also accepts saved analysis names', () => {
  assert.deepEqual(ids(filter(rows, { search: ' jtzs ', date: '7' })), [3]);
  assert.deepEqual(ids(filter(rows, { search: 'Analysis 3', date: 'today' })), []);
  assert.deepEqual(ids(filter(rows, { search: 'ANALYSIS 3' })), [3]);
});

test('sorts place missing measurements last, retain stable ties, and leave the source list unchanged', () => {
  const original = ids(rows);
  assert.deepEqual(ids(filter(rows, { sort: 'oldest' })), [5, 4, 3, 2, 1, 6]);
  assert.deepEqual(ids(filter(rows, { sort: 'plantable-desc' })), [5, 4, 3, 2, 1, 6]);
  assert.deepEqual(ids(filter(rows, { sort: 'canopy-asc' })), [1, 2, 3, 4, 5, 6]);
  assert.deepEqual(ids(rows), original);
  assert.deepEqual(ids(filter([rows[1], { ...rows[1], id: 7 }], {})), [7, 2]);
});

test('legacy coverage can be calculated from area without treating unknown measurements as zero', () => {
  assert.equal(canopyCoverage({ canopy_area_m2: '125', total_area_m2: 500 }), 25);
  assert.equal(canopyCoverage({ canopy_area_m2: 0, total_area_m2: 500 }), 0);
  assert.equal(canopyCoverage({ canopy_area_m2: null, total_area_m2: 500 }), null);
  assert.equal(canopyCoverage({ canopy_area_m2: 125, total_area_m2: 0 }), null);
  assert.equal(formatAnalysisNumber(null), '—');
  assert.equal(formatAnalysisNumber(0), '0');
});

test('saved map areas keep the original polygon or multipolygon without stale image or point overlays', () => {
  const polygon = { type: 'Polygon', coordinates: [[[122.6, 10.8], [122.601, 10.8001], [122.6008, 10.801], [122.6001, 10.8009], [122.6, 10.8]]] };
  for (const footprint of [polygon, { type: 'MultiPolygon', coordinates: [polygon.coordinates] }]) {
    const analysis = { analysis_id: 52, map: { available: true, analysis_footprint: footprint, coordinates: [],
      analysis_overlay: { image_data_url: 'old-preview' }, safe_points_geojson: { features: ['stale-point'] } } };
    const context = savedAnalysisMapContext(analysis);
    assert.equal(canViewSavedAnalysisOnMap(analysis), true);
    assert.deepEqual(context.map.analysis_footprint, footprint);
    assert.equal(context.preserveMapView, false);
    assert.equal(savedAnalysisMapContext(analysis, { preserveView: true }).preserveMapView, true);
    assert.equal(context.map.analysis_overlay, undefined);
    assert.equal(context.map.safe_points_geojson, undefined);
    assert.equal(context.map.image_center_feature, null);
  }
});

test('older saved analyses can focus on their saved GPS center without fabricating an area boundary', () => {
  const analysis = { analysis_id: 12, map: { available: true, coordinates: [] },
    metadata: { image_center_lat: '10.780405', image_center_lon: '122.626975' } };
  assert.equal(canViewSavedAnalysisOnMap(analysis), true);
  const context = savedAnalysisMapContext(analysis);
  assert.equal(context.map.analysis_footprint, null);
  assert.deepEqual(context.map.image_center_feature.geometry, { type: 'Point', coordinates: [122.626975, 10.780405] });
});

test('missing or invalid GPS coordinates do not produce a map action at zero latitude or longitude', () => {
  for (const latitude of [null, undefined, '', ' ', true, 'bad', 91]) {
    assert.equal(canViewSavedAnalysisOnMap({ map: { available: true },
      metadata: { image_center_lat: latitude, image_center_lon: 122.6 } }), false);
  }
  assert.equal(canViewSavedAnalysisOnMap({ map: { available: false },
    metadata: { image_center_lat: 10.8, image_center_lon: 122.6 } }), false);
});
