import test, { afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { appointmentEndTime, bookingErrors, bookingPayload, bookingPolicy, manilaToday } from './booking.js';
import { assessPlantingBooking, bookingTideEvents, plantingBookingWindows } from './bookingTides.js';
import { assessGraphWindow } from '../utils/plantingTides.js';

const defaultHoursPolicy = bookingPolicy.enforcePlantingHours;
afterEach(() => { bookingPolicy.enforcePlantingHours = defaultHoursPolicy; });

const valid = { organization: ' School ', contact_name: ' Coordinator ', phone: '09123456789', email: 'coordinator@example.org', appointment_type: 'tree_planting',
  date: manilaToday(new Date(Date.now() + 7 * 86400000)), start_time: '08:00', end_time: '10:00', participants: '20', notes: '', consent: true, website: '' };

test('calendar dates follow Manila across a UTC date boundary', () => {
  assert.equal(manilaToday(new Date('2026-10-06T18:00:00Z')), '2026-10-07');
});

test('booking payload sends explicit Manila times and keeps the retry key', () => {
  const body = bookingPayload(valid, 'retry-key');
  assert.equal(body.organization, 'School');
  assert.equal(body.start_at, `${valid.date}T08:00:00+08:00`);
  assert.equal(body.end_at, `${valid.date}T10:00:00+08:00`);
  assert.equal(body.participants, 20);
  assert.equal(body.submission_key, 'retry-key');
  assert.equal(body.email, 'coordinator@example.org');
  assert.equal(body.appointment_type, 'tree_planting');
  assert.equal(body.title, 'Tree planting');
});

test('invalid booking requests fail before transmission', () => {
  for (const changes of [{ consent: false }, { date: '2020-01-01' }, { start_time: '22:00' },
    { participants: '0' }, { participants: '1.5' }, { organization: '' }, { phone: '' },
    { email: '' }, { email: 'broken' }, { appointment_type: '' }, { appointment_type: 'other' }]) {
    assert.throws(() => bookingPayload({ ...valid, ...changes }, 'retry-key'));
  }
});

test('each appointment choice retains its type and has a useful default title', () => {
  for (const [type, label] of [['field_visit', 'Field visit'], ['clean_up_drive', 'Clean-up drive'], ['tree_planting', 'Tree planting']]) {
    const body = bookingPayload({ ...valid, appointment_type: type, title: 'A cached custom title' }, 'retry-key');
    assert.equal(body.appointment_type, type);
    assert.equal(body.title, label);
  }
});

test('booking errors identify each correction and only validate the active step', () => {
  assert.deepEqual(Object.keys(bookingErrors({ ...valid, organization: '  ', phone: 'abc', email: 'broken' }, 1)), ['organization', 'phone', 'email']);
  assert.deepEqual(Object.keys(bookingErrors({ ...valid, email: 'broken', start_time: '23:00' }, 2)), ['start_time']);
  assert.throws(() => bookingPayload({ ...valid, start_time: '23:00' }, 'key'), (error) => Boolean(error.fieldErrors.start_time));
});

test('two-hour ends are derived for all appointment types, including noon and date limits', () => {
  for (const [start, end] of [['07:00', '09:00'], ['11:30', '13:30'], ['21:59', '23:59'], ['22:00', ''], ['23:00', ''], ['bad', '']]) {
    assert.equal(appointmentEndTime(start), end);
  }
  for (const appointment_type of ['tree_planting', 'field_visit', 'clean_up_drive']) {
    const body = bookingPayload({ ...valid, appointment_type, start_time: '07:00', end_time: '20:00' }, 'key');
    assert.equal(body.end_at, `${valid.date}T09:00:00+08:00`);
  }
  assert.equal(bookingErrors({ ...valid, date: '2030-02-30' }, 2).date, 'Choose a valid date.');
});

const tideDay = '2030-01-02';
const tideBase = Date.parse(`${tideDay}T00:00:00+08:00`);
// Short daytime low-water window, followed by lower water on later days. The shared
// estimated reference is calculated from this series, never hard-coded here.
const tidePayload = {
  available: true, stale: false, fetched_at: tideBase / 1000, datum: 'MSL', datum_reference: 'test', series_available: true,
  heights: Array.from({ length: 73 }, (_, hour) => ({ timestamp: tideBase / 1000 + hour * 3600,
    height_m: hour >= 24 || hour === 8 || hour === 9 ? -1 : hour === 7 || hour === 10 ? 0 : 1 })),
};
const windows = (changes = {}) => plantingBookingWindows({ payload: tidePayload, date: tideDay, now: tideBase, ...changes });

test('suggested windows and the latest start cover a full two-hour visit using the staff graph rule', () => {
  const result = windows();
  assert.equal(result.status, 'available');
  const window = result.windows[0];
  assert.equal(window.start, tideBase + 7 * 3600000);
  assert.equal(window.end, tideBase + 10 * 3600000);
  assert.equal(window.end - window.latestStart, 2 * 3600000);
  assert.equal(window.start % (15 * 60000), 0);
  assert.equal(window.end % (15 * 60000), 0);
  for (const startAt of [window.start, window.latestStart]) {
    assert.equal(assessGraphWindow({ payload: tidePayload, startAt, endAt: startAt + 2 * 3600000, now: tideBase }).status, 'safe');
  }
  assert.equal(assessGraphWindow({ payload: tidePayload, startAt: window.latestStart + 3600000,
    endAt: window.end + 3600000, now: tideBase }).status, 'unsafe');
});

test('a 9:45 AM visit can fit below the planting limit before the 3 PM low tide', () => {
  const payload = { ...tidePayload,
    heights: tidePayload.heights.map((row, hour) => ({ ...row,
      height_m: 0.8 + 0.7 * Math.cos((hour - 3) * Math.PI / 12) })),
    extremes: [{ timestamp: tideBase / 1000 + 3 * 3600, type: 'High', height_m: 1.5 },
      { timestamp: tideBase / 1000 + 15 * 3600, type: 'Low', height_m: 0.1 }],
  };
  const assessment = assessPlantingBooking({ payload, date: tideDay, startTime: '09:45', now: tideBase });
  assert.equal(assessment.status, 'safe');
  assert.ok(assessment.maximum_m < assessment.threshold_m);
  const start = tideBase + (9 * 60 + 45) * 60000;
  assert.ok(windows({ payload }).windows.some((window) => start >= window.start && start <= window.latestStart));
  const events = bookingTideEvents({ payload, date: tideDay, now: tideBase });
  assert.deepEqual(events.filter((event) => event.type === 'low').map((event) => event.at), [tideBase + 15 * 3600000]);
  assert.equal(events.some((event) => event.at === start), false);
});

test('missing, stale, invalid, gapped and out-of-horizon forecasts cannot suggest a planting time', () => {
  for (const changes of [
    { payload: null }, { payload: { ...tidePayload, stale: true } }, { now: tideBase + 6 * 3600000 },
    { date: '2030-01-12' }, { date: '2030-02-30' },
    { payload: { ...tidePayload, datum: 'Unknown' } },
    { payload: { ...tidePayload, heights: tidePayload.heights.filter((_, i) => i !== 8) } },
    { payload: { ...tidePayload, heights: tidePayload.heights.map((row, i) => i === 8 ? { ...row, height_m: null } : row) } },
  ]) assert.deepEqual(windows(changes).windows, []);
});

test('a forecast starting at 8 AM still suggests fully covered visits on its first Philippine date', () => {
  const payload = { ...tidePayload, heights: tidePayload.heights.slice(8).map((row) => ({ ...row, height_m: 0.2 })) };
  for (const enforceHours of [false, true]) {
    bookingPolicy.enforcePlantingHours = enforceHours;
    const result = windows({ payload });
    assert.equal(result.status, 'available');
    assert.equal(result.partialCoverage, true);
    assert.equal(result.windows[0].startTime, '08:00');
    assert.equal(result.windows[0].end, tideBase + (enforceHours ? 17 * 60 : 23 * 60 + 45) * 60000);
    for (const window of result.windows) {
      for (const startAt of [window.start, window.latestStart]) {
        assert.equal(assessGraphWindow({ payload, startAt, endAt: startAt + 2 * 3600000, now: tideBase }).status, 'safe');
      }
    }
    assert.equal(assessPlantingBooking({ payload, date: tideDay, startTime: '07:00', now: tideBase }).status, 'unknown');
  }
});

test('partial forecasts never extend suggestions past their last sample or fill a short window', () => {
  const heights = tidePayload.heights.slice(8, 12).map((row) => ({ ...row, height_m: 0.2 }));
  const result = windows({ payload: { ...tidePayload, heights } });
  assert.equal(result.status, 'available');
  assert.equal(result.partialCoverage, true);
  assert.equal(result.windows[0].startTime, '08:00');
  assert.equal(result.windows[0].end, tideBase + 11 * 3600000);
  assert.equal(result.windows[0].latestStart, tideBase + 9 * 3600000);
  const short = windows({ payload: { ...tidePayload, heights: heights.slice(0, 2) } });
  assert.equal(short.status, 'unavailable');
  assert.deepEqual(short.windows, []);
  assert.match(short.reason, /only part of this date/);
});

test('booking high and low tides follow Philippine dates, sort chronologically and reject invalid events', () => {
  const event = (hour, type, height_m = 0.2) => ({ timestamp: tideBase / 1000 + hour * 3600, type, height_m });
  const payload = { ...tidePayload, extremes: [event(21, 'High', 1.2), event(15, 'Low'), event(0, 'Low', 0),
    event(-1, 'High'), event(24, 'Low'), event(10, 'Other'), event(12, 'Low', null), { type: 'Low', height_m: 0.1 }] };
  const events = bookingTideEvents({ payload, date: tideDay, now: tideBase });
  assert.deepEqual(events.map(({ at, type }) => [at, type]), [[tideBase, 'low'],
    [tideBase + 15 * 3600000, 'low'], [tideBase + 21 * 3600000, 'high']]);
  assert.equal(events[0].height_m, 0);
  for (const changes of [{ payload: null }, { payload: { ...payload, stale: true } },
    { now: tideBase + 6 * 3600000 }, { date: '2030-01-12' }, { date: '2030-02-30' }]) {
    assert.deepEqual(bookingTideEvents({ payload, date: tideDay, now: tideBase, ...changes }), []);
  }
});

test('suggestions exclude past times, nighttime, and windows too short for two hours', () => {
  bookingPolicy.enforcePlantingHours = true;
  const now = tideBase + 8 * 3600000 + 5 * 60000;
  const fresh = { ...tidePayload, fetched_at: now / 1000 };
  for (const window of windows({ now, payload: fresh }).windows) {
    assert.ok(window.start > now);
    assert.ok(window.start >= tideBase + 6 * 3600000);
    assert.ok(window.end <= tideBase + 17 * 3600000);
  }
  const late = tideBase + 11 * 3600000;
  assert.equal(windows({ now: late, payload: { ...fresh, fetched_at: late / 1000 } }).status, 'none');
});

test('all-day lower water is clipped to LIKE hours and a latest start of 3 PM', () => {
  bookingPolicy.enforcePlantingHours = true;
  const payload = { ...tidePayload, heights: tidePayload.heights.map((row) => ({ ...row, height_m: 0.2 })) };
  const result = windows({ payload });
  assert.equal(result.windows.length, 1);
  assert.equal(result.windows[0].startTime, '06:00');
  assert.equal(result.windows[0].end, tideBase + 17 * 3600000);
  assert.equal(result.windows[0].latestStart, tideBase + 15 * 3600000);
  for (const startTime of ['04:30', '05:59', '15:01', '16:00', '17:00']) {
    assert.equal(assessPlantingBooking({ payload, date: tideDay, startTime, now: tideBase }).status, 'closed');
    assert.match(bookingErrors({ ...valid, start_time: startTime }, 2).start_time, /6 AM to 5 PM/);
  }
  for (const startTime of ['06:00', '15:00']) {
    assert.equal(assessPlantingBooking({ payload, date: tideDay, startTime, now: tideBase }).status, 'safe');
    assert.equal(bookingErrors({ ...valid, start_time: startTime }, 2).start_time, undefined);
  }
});

test('high water within opening hours splits availability and blocks a visit spanning the peak', () => {
  const payload = { ...tidePayload, heights: tidePayload.heights.map((row, i) => ({ ...row, height_m: i === 11 ? 1 : 0 })) };
  const result = windows({ payload });
  assert.equal(result.windows.length, 2);
  assert.equal(result.windows[0].end, tideBase + 10 * 3600000);
  assert.equal(result.windows[1].startTime, '12:00');
  assert.equal(assessPlantingBooking({ payload, date: tideDay, startTime: '10:00', now: tideBase }).status, 'unsafe');
  assert.equal(assessPlantingBooking({ payload, date: tideDay, startTime: '12:00', now: tideBase }).status, 'safe');
});

test('testing permits early and evening visits while preserving duration and tide checks', () => {
  const payload = { ...tidePayload, fetched_at: tideBase / 1000 - 3600,
    heights: tidePayload.heights.map((row) => ({ ...row, height_m: 0.2 })) };
  const result = windows({ payload, now: tideBase - 3600000 });
  assert.equal(result.windows[0].startTime, '00:00');
  assert.equal(result.windows[0].end, tideBase + (23 * 60 + 45) * 60000);
  for (const startTime of ['00:00', '04:30', '05:59', '15:01', '16:45', '19:00', '21:59']) {
    assert.equal(assessPlantingBooking({ payload, date: tideDay, startTime, now: tideBase - 3600000 }).status, 'safe');
    assert.equal(bookingErrors({ ...valid, start_time: startTime }, 2).start_time, undefined);
    const body = bookingPayload({ ...valid, start_time: startTime }, 'test-key');
    assert.equal(body.end_at, `${valid.date}T${appointmentEndTime(startTime)}:00+08:00`);
  }
  assert.equal(assessPlantingBooking({ payload, date: tideDay, startTime: '22:00', now: tideBase }).status, 'closed');
});

test('unknown forecasts remain available for LGU review outside the usual hours during testing', () => {
  assert.equal(assessPlantingBooking({ payload: null, date: tideDay, startTime: '06:00', now: tideBase }).status, 'unknown');
  assert.equal(assessPlantingBooking({ payload: null, date: tideDay, startTime: '04:30', now: tideBase }).status, 'unknown');
  assert.equal(bookingErrors({ ...valid, appointment_type: 'field_visit', start_time: '04:30' }, 2).start_time, undefined);
});

test('fractional tide crossings round inward rather than extending into higher water', () => {
  const fractional = { ...tidePayload, heights: tidePayload.heights.map((row, i) => ({ ...row,
    height_m: i === 7 ? -0.2 : i === 10 ? 0.4 : row.height_m })) };
  const result = windows({ payload: fractional });
  assert.equal(result.status, 'available');
  assert.equal(result.windows[0].startTime, '07:00');
  assert.equal(result.windows[0].end, tideBase + 9.5 * 3600000);
  assert.equal(assessGraphWindow({ payload: fractional, startAt: result.windows[0].start,
    endAt: result.windows[0].end, now: tideBase }).status, 'safe');
  const brief = { ...tidePayload, heights: tidePayload.heights.map((row, i) => ({ ...row,
    height_m: i === 7 || i === 10 ? 1 : i === 8 || i === 9 ? 0 : row.height_m })) };
  assert.equal(windows({ payload: brief }).status, 'none');
});
