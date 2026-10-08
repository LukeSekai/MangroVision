import test from 'node:test';
import assert from 'node:assert/strict';
import { groupCalendarSchedules } from './calendarScheduleGroups.js';

const schedule = (id, overrides = {}) => ({
  id, organization_id: id, organization_name: `Organization ${id}`,
  title: 'Tentative follow-up planting activity', scheduled_date: '2026-10-14',
  start_at: '2026-10-14T07:30:14+08:00', end_at: '2026-10-14T11:30:14+08:00',
  start_time: '07:30', end_time: '11:30', status: 'tentative', ...overrides,
});

test('three matching organization schedules share one card and retain every record', () => {
  const rows = [schedule(35), schedule(40), schedule(45)];
  const groups = groupCalendarSchedules(rows);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].organizationCount, 3);
  assert.deepEqual(groups[0].schedules, rows);
  assert.equal(groups[0].schedule, rows[0]);
});

test('different dates, times, durations, titles and statuses remain separate', () => {
  const rows = [
    schedule(1),
    schedule(2, { scheduled_date: '2026-10-15', start_at: '2026-10-15T07:30:14+08:00', end_at: '2026-10-15T11:30:14+08:00' }),
    schedule(3, { start_at: '2026-10-14T08:30:14+08:00' }),
    schedule(4, { end_at: '2026-10-14T12:30:14+08:00' }),
    schedule(5, { title: 'Shoreline cleanup' }),
    schedule(6, { status: 'confirmed' }),
    schedule(7, { appointment_type: 'field_visit' }),
  ];
  assert.equal(groupCalendarSchedules(rows).length, rows.length);
});

test('timezone representations of the same activity and harmless title spacing group together', () => {
  const rows = [schedule(1), schedule(2, {
    title: '  TENTATIVE   follow-up planting activity ',
    start_at: '2026-10-13T23:30:14Z', end_at: '2026-10-14T03:30:14Z',
  })];
  assert.equal(groupCalendarSchedules(rows).length, 1);
});

test('multiple records for one organization stay accessible without inflating its count', () => {
  const rows = [schedule(1), schedule(2, { organization_id: 1, organization_name: 'Organization 1' })];
  const [group] = groupCalendarSchedules(rows);
  assert.equal(group.organizationCount, 1);
  assert.equal(group.schedules.length, 2);
});

test('records with incomplete times are kept separately and empty input stays empty', () => {
  const rows = [1, 2].map((id) => schedule(id, { start_at: null, start_time: '', end_at: null, end_time: '' }));
  assert.equal(groupCalendarSchedules(rows).length, 2);
  assert.deepEqual(groupCalendarSchedules([]), []);
});
