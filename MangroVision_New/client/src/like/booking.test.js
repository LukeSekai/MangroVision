import test from 'node:test';
import assert from 'node:assert/strict';
import { bookingPayload, manilaToday } from './booking.js';

const valid = { organization: ' School ', contact_name: ' Coordinator ', phone: '09123456789', email: 'coordinator@example.org', appointment_type: 'tree_planting',
  title: '', date: manilaToday(new Date(Date.now() + 7 * 86400000)), start_time: '08:00', end_time: '10:00', participants: '20', notes: '', consent: true, website: '' };

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
  for (const changes of [{ consent: false }, { date: '2020-01-01' }, { end_time: '07:00' },
    { participants: '0' }, { participants: '1.5' }, { organization: '' }, { phone: '' },
    { email: '' }, { email: 'broken' }, { appointment_type: '' }, { appointment_type: 'other' }]) {
    assert.throws(() => bookingPayload({ ...valid, ...changes }, 'retry-key'));
  }
});

test('each appointment choice retains its type and has a useful default title', () => {
  for (const [type, label] of [['field_visit', 'Field visit'], ['clean_up_drive', 'Clean-up drive'], ['tree_planting', 'Tree planting']]) {
    const body = bookingPayload({ ...valid, appointment_type: type }, 'retry-key');
    assert.equal(body.appointment_type, type);
    assert.equal(body.title, label);
  }
});
