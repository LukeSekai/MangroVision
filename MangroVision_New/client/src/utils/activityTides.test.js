import test from 'node:test';
import assert from 'node:assert/strict';
import { activityTideSummary, assessGraphWindow } from './plantingTides.js';

const base = Date.parse('2026-10-11T00:00:00Z');
const at = (hour) => new Date(base + hour * 3600000).toISOString();
const payload = {
  available: true, stale: false, fetched_at: base / 1000, datum: 'MSL', datum_reference: { provider: 'test' }, series_available: true,
  heights: [0, 0, 4, 0, 0].map((height_m, hour) => ({ timestamp: base / 1000 + hour * 3600, height_m })),
  extremes: [{ type: 'High', timestamp: base / 1000 + 7200, height_m: 4 }],
};
const summary = (changes = {}) => activityTideSummary({ payload, startAt: at(1), endAt: at(3), now: base, ...changes });

test('duration uses threshold crossings rather than a high-tide peak timestamp', () => {
  const result = summary();
  assert.equal(result.periods.length, 1);
  assert.equal(result.peaks.length, 1);
  assert.ok(Math.abs(result.periods[0].duration_minutes - 72) < 1e-6);
  assert.ok(Math.abs(result.periods[0].start - (base + 1.4 * 3600000)) < 1);
  assert.equal(result.periods[0].incomplete, false);
  assert.equal(result.status, assessGraphWindow({ payload, startAt: at(1), endAt: at(3), now: base }).status);
});

test('only periods overlapping the activity are displayed', () => {
  const result = summary({ startAt: at(0), endAt: at(1) });
  assert.equal(result.status, 'safe');
  assert.deepEqual(result.periods, []);
});

test('stale, gapped and uncovered forecasts cannot produce a duration or safety result', () => {
  for (const changes of [
    { now: base + 6 * 3600000 },
    { payload: { ...payload, stale: true } },
    { startAt: at(5), endAt: at(6) },
    { payload: { ...payload, heights: payload.heights.filter((_, i) => i !== 2) } },
    { payload: { ...payload, heights: payload.heights.map((row, i) => i === 2 ? { ...row, height_m: null } : row) } },
  ]) {
    const result = summary(changes);
    assert.equal(result.status, 'unknown');
    assert.deepEqual(result.periods, []);
    assert.deepEqual(result.peaks, []);
  }
});

test('a period extending beyond forecast coverage is labeled incomplete', () => {
  const result = summary({ startAt: at(0), endAt: at(1), payload: { ...payload,
    heights: [4, 0, 0, 0, 0].map((height_m, hour) => ({ timestamp: base / 1000 + hour * 3600, height_m })),
  } });
  assert.equal(result.periods.length, 1);
  assert.equal(result.periods[0].incomplete, true);
});
