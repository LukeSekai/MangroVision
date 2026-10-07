import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { controlError, serverFieldErrors, submissionError } from './formValidation.js';
import { POINT_STATUS_LABELS, ASSIGNMENT_STATUS_LABELS, pointStatusLabel } from './pointStatus.js';
import { reportPeriodFieldErrors } from './restorationReports.js';

const dom = new JSDOM();
const input = (attributes, value = '') => {
  const control = dom.window.document.createElement('input');
  Object.entries(attributes).forEach(([key, setting]) => control.setAttribute(key, setting));
  control.value = value;
  return control;
};

test('counts explain missing, fractional and out-of-range values beside the field', () => {
  const count = input({ type: 'number', required: '', min: '1', max: '10', step: '1' });
  assert.equal(controlError(count, 'Participants'), 'Enter participants.');
  count.value = '0';
  assert.equal(controlError(count, 'Participants'), 'Enter 1 or more for participants.');
  count.value = '11';
  assert.equal(controlError(count, 'Participants'), 'Enter 10 or less for participants.');
  count.value = '2.5';
  assert.equal(controlError(count, 'Participants'), 'Enter a whole number for participants.');
  count.value = '10';
  assert.equal(controlError(count, 'Participants'), '');
});

test('optional, disabled and read-only controls do not block submission', () => {
  assert.equal(controlError(input({ type: 'number', min: '0' }), 'Goal'), '');
  assert.equal(controlError(input({ required: '', disabled: '' }), 'Organization'), '');
  assert.equal(controlError(input({ required: '', readonly: '' }), 'Recorded value'), '');
  assert.equal(controlError(input({ required: '' }, '   '), 'Activity title'), 'Enter activity title.');
});

test('password length and verification format provide specific corrections', () => {
  assert.equal(controlError(input({ minlength: '12' }, 'short'), 'New password'), 'Use at least 12 characters for new password.');
  const code = input({ pattern: '[0-9]{6}', title: 'Enter the six-digit code from your email.' }, '123');
  assert.equal(controlError(code, 'Email verification code'), 'Enter the six-digit code from your email.');
});

test('server validation preserves field locations and supplies a correction', () => {
  const fields = { count: { label: 'Participants', aliases: ['expected_planters'] } };
  const error = submissionError([{ loc: ['body', 'expected_planters'], type: 'greater_than_equal', ctx: { ge: 0 }, msg: 'Input should be greater than or equal to 0' }]);
  assert.deepEqual(serverFieldErrors(error, fields), { count: 'Enter 0 or more for participants.' });
  assert.deepEqual(serverFieldErrors(submissionError([{ loc: ['body', 'expected_planters'], type: 'int_from_float' }]), fields), { count: 'Enter a whole number for participants.' });
  assert.deepEqual(serverFieldErrors(new Error('Failed to fetch'), fields), {});
  assert.deepEqual(serverFieldErrors(new Error('Session expired. Sign in again.'), fields), {});
});

test('point aliases share labels while whole assignment completion remains distinct', () => {
  assert.equal(POINT_STATUS_LABELS.pending, POINT_STATUS_LABELS.assigned);
  assert.equal(POINT_STATUS_LABELS.completed, POINT_STATUS_LABELS.planted);
  assert.equal(pointStatusLabel('completed'), 'Planted');
  assert.equal(ASSIGNMENT_STATUS_LABELS.completed, 'Completed');
  assert.equal(pointStatusLabel('eroded_unavailable'), 'Unavailable');
  assert.equal(pointStatusLabel('not_assigned'), 'Planned');
});

test('report date errors identify the editable field without enabling an invalid period', () => {
  const today = '2026-10-07';
  assert.deepEqual(reportPeriodFieldErrors({ dateFrom: '2026-10-07', dateTo: '2026-10-01' }, today), { dateTo: 'Choose an end date on or after the start date.' });
  assert.deepEqual(reportPeriodFieldErrors({ dateFrom: '', dateTo: '2026-10-08' }, today), { dateFrom: 'Choose a valid start date.', dateTo: 'Choose today or an earlier end date.' });
  assert.deepEqual(reportPeriodFieldErrors({ dateFrom: '2026-02-30', dateTo: '2026-03-01' }, today), { dateFrom: 'Choose a valid start date.' });
  assert.deepEqual(reportPeriodFieldErrors({ dateFrom: '2026-10-01', dateTo: today }, today), {});
});
