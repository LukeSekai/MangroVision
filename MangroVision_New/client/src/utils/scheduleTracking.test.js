import test from 'node:test';
import assert from 'node:assert/strict';
import { SCHEDULE_TRACKING_GROUPS, scheduleProgressStatus, scheduleStatusForProgress, scheduleTrackingState } from './scheduleTracking.js';

const now = '2026-10-15T09:00:00+08:00';
const activity = (status, changes = {}) => ({
  status, start_at: '2026-10-15T07:00:00+08:00', end_at: '2026-10-15T08:00:00+08:00', ...changes,
});

test('exactly three progress groups retain approval as a separate property', () => {
  assert.deepEqual(SCHEDULE_TRACKING_GROUPS.map(({ key }) => key), ['completed', 'pending', 'cancelled']);
  for (const status of ['requested', 'tentative', 'confirmed', 'in_progress', 'completed', 'cancelled']) {
    const tracking = scheduleTrackingState(activity(status), now);
    assert.equal(Number(tracking.completed) + Number(tracking.pending) + Number(tracking.cancelled), 1);
    assert.equal(tracking[scheduleProgressStatus(status)], true);
  }
  for (const status of ['requested', 'tentative']) {
    assert.equal(scheduleTrackingState(activity(status), now).needs_confirmation, true);
    assert.equal(scheduleTrackingState(activity(status), now).approved, false);
  }
  for (const status of ['confirmed', 'in_progress', 'completed']) {
    assert.equal(scheduleTrackingState(activity(status), now).approved, true);
    assert.equal(scheduleTrackingState(activity(status), now).needs_confirmation, false);
  }
  assert.deepEqual(scheduleTrackingState(activity('cancelled'), now), {
    completed: false, pending: false, cancelled: true, needs_confirmation: false, approved: false, missed: false,
  });
});

test('saving pending progress preserves confirmed and ongoing planting eligibility', () => {
  for (const status of ['requested', 'tentative', 'confirmed', 'in_progress']) {
    assert.equal(scheduleProgressStatus(status), 'pending');
    assert.equal(scheduleStatusForProgress('pending', status), status);
    assert.equal(scheduleStatusForProgress('completed', status), 'completed');
    assert.equal(scheduleStatusForProgress('cancelled', status), 'cancelled');
  }
  assert.equal(scheduleStatusForProgress('pending'), 'requested');
  assert.equal(scheduleStatusForProgress('pending', 'completed'), 'completed');
  assert.equal(scheduleStatusForProgress('cancelled', 'completed'), 'completed');
  assert.equal(scheduleStatusForProgress('pending', 'cancelled'), 'requested');
});

test('missed uses the end of the activity, including unfinished planting', () => {
  for (const status of ['requested', 'tentative', 'confirmed', 'in_progress']) {
    assert.equal(scheduleTrackingState(activity(status, { planting_started: true }), now).missed, true);
  }
  assert.equal(scheduleTrackingState(activity('confirmed'), '2026-10-15T07:30:00+08:00').missed, false);
  assert.equal(scheduleTrackingState(activity('confirmed'), '2026-10-15T08:00:00+08:00').missed, false);
  assert.equal(scheduleTrackingState(activity('confirmed'), '2026-10-15T08:00:01+08:00').missed, true);
});

test('completed and cancelled activities never enter the missed queue', () => {
  for (const status of ['completed', 'cancelled']) {
    assert.equal(scheduleTrackingState(activity(status), now).missed, false);
  }
});

test('equivalent UTC and Philippine timestamps produce the same missed result', () => {
  const local = activity('confirmed');
  const utc = activity('confirmed', { end_at: '2026-10-15T00:00:00Z' });
  const legacy = { status: 'confirmed', date: '2026-10-15', end_time: '08:00' };
  assert.deepEqual(scheduleTrackingState(local, now), scheduleTrackingState(utc, now));
  assert.deepEqual(scheduleTrackingState(local, now), scheduleTrackingState(legacy, now));
});

test('date-only records use the Philippine day and do not invent an end time today', () => {
  const afterMidnight = '2026-10-14T17:00:00Z'; // October 15, 01:00 in the Philippines.
  assert.equal(scheduleTrackingState({ status: 'confirmed', date: '2026-10-14' }, afterMidnight).missed, true);
  assert.equal(scheduleTrackingState({ status: 'confirmed', date: '2026-10-15' }, afterMidnight).missed, false);
});

test('invalid and missing timing information cannot prove an activity was missed', () => {
  for (const row of [
    { status: 'confirmed' },
    { status: 'confirmed', date: '2026-02-31' },
    { status: 'confirmed', date: '2020-01-01', end_at: 'unknown' },
    { status: 'confirmed', date: '2020-01-01', end_time: '99:99' },
  ]) assert.equal(scheduleTrackingState(row, now).missed, false);
  assert.equal(scheduleTrackingState(activity('confirmed'), 'invalid').missed, false);
});

test('tracking labels preserve saved status, planting history and planning locks', () => {
  const row = Object.freeze(activity('confirmed', {
    planting_started: true, recorded_planting_count: 8, linked_assignment_count: 1,
  }));
  const snapshot = JSON.stringify(row);
  const result = scheduleTrackingState(row, now);
  assert.equal(result.missed, true);
  assert.equal(result.approved, true);
  assert.equal(result.pending, true);
  assert.equal(JSON.stringify(row), snapshot);
});
