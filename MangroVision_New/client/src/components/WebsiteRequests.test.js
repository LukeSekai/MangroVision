import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

let server, dom, root, createRoot, RequestReview;
const globals = new Map();
const request = {
  id: 7, reference: 'LIKE-REVIEW-TEST', organization: 'Coastal volunteers', contact_name: 'Test coordinator',
  phone: '09123456789', email: 'coordinator@example.org', title: 'Coastal field visit',
  appointment_type: 'field_visit', participants: 10, status: 'pending', notes: 'Meet at the gate.',
  start_at: '2035-06-12T08:00:00+08:00', end_at: '2035-06-12T09:30:00+08:00',
};
let calls, saved, assessed;

before(async () => {
  dom = new JSDOM('<div id="root"></div>', { pretendToBeVisual: true, url: 'http://localhost' });
  for (const [key, value] of Object.entries({ window: dom.window, document: dom.window.document,
    navigator: dom.window.navigator, HTMLElement: dom.window.HTMLElement, Element: dom.window.Element,
    IS_REACT_ACT_ENVIRONMENT: true, requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window), fetch: globalThis.fetch,
  })) {
    globals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  window.HTMLElement.prototype.scrollTo = () => {};
  ({ createRoot } = await import('react-dom/client'));
  server = await createServer({ configFile: false, appType: 'custom', server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    cacheDir: 'node_modules/.vite-website-review-tests', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ RequestReview } = await server.ssrLoadModule('/src/components/WebsiteRequests.jsx'));
});
after(async () => {
  await server?.close(); dom?.window.close();
  for (const [key, descriptor] of globals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
  }
});
beforeEach(() => {
  root = createRoot(document.getElementById('root'));
  calls = []; saved = []; assessed = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, body: JSON.parse(options.body) });
    return Response.json({ id: request.id, status: 'confirmed' });
  };
});
afterEach(async () => { await act(async () => root.unmount()); });

async function render(overrides = {}) {
  await act(async () => root.render(createElement(RequestReview, {
    request, organizations: [{ id: 4, name: request.organization }], schedules: [],
    assessmentFor: (window) => { assessed.push(window); return { status: 'safe', reason: 'Safe forecast.' }; },
    renderAdvice: () => null, onClose: () => {}, onSaved: (message) => saved.push(message), ...overrides,
  })));
}
async function change(name, value) {
  const control = document.querySelector(`[name="${name}"]`);
  assert.ok(control, `Missing ${name} control`);
  await act(async () => {
    const prototype = control.tagName === 'SELECT' ? window.HTMLSelectElement.prototype
      : control.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(prototype, 'value').set.call(control, value);
    control.dispatchEvent(new window.Event(control.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true }));
  });
}
async function check(name) { await act(async () => document.querySelector(`[name="${name}"]`).click()); }
async function submit() {
  await act(async () => document.querySelector('form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true })));
}

test('pending review shows submitted details without editors or request cancellation', async () => {
  await render();
  assert.equal(document.querySelector('.modal-backdrop').parentElement, document.body, 'Dialog must escape the workspace header stacking context');
  assert.ok(document.querySelector('.website-review-dialog'));
  assert.match(document.body.textContent, /Coastal volunteers|coordinator@example.org|Meet at the gate/);
  for (const name of ['organization', 'email', 'date', 'startTime', 'endTime']) assert.equal(document.querySelector(`[name="${name}"]`), null);
  assert.deepEqual([...document.querySelector('[name="action"]').options].map((option) => option.textContent),
    ['Confirm requested schedule', 'Reschedule and confirm', 'Decline request']);
  assert.doesNotMatch(document.body.textContent, /Cancel request/);
  assert.ok([...document.querySelectorAll('button')].find((button) => button.textContent === 'Close'));
  await submit();
  assert.equal(calls.length, 0, 'Contact agreement is required before confirming');
  await check('contacted'); await submit();
  assert.equal(calls.length, 1);
  assert.deepEqual(calls[0].body, { action: 'confirmed', contacted: true, decision_notes: null, organization: request.organization, organization_id: 4 });
  assert.equal(saved.length, 1);
});

test('rescheduling reveals time controls and sends the agreed replacement window', async () => {
  await render();
  const original = document.querySelector('.website-request-detail-wide:last-child').textContent;
  await change('action', 'rescheduled');
  assert.equal(document.querySelector('[name="date"]').value, '2035-06-12');
  await change('date', '2035-06-15'); await change('startTime', '10:00'); await change('endTime', '11:30');
  assert.equal(document.querySelector('.website-request-detail-wide:last-child').textContent, original);
  await check('contacted'); await submit();
  assert.equal(calls[0].body.action, 'confirmed');
  assert.equal(calls[0].body.start_at, '2035-06-15T10:00:00+08:00');
  assert.equal(calls[0].body.end_at, '2035-06-15T11:30:00+08:00');
  assert.equal(calls[0].body.organization, request.organization);
  assert.equal(calls[0].body.email, undefined);
});

test('returning from reschedule to confirmation ignores the draft times and reassesses the original', async () => {
  await render({ request: { ...request, appointment_type: 'tree_planting' } });
  await change('action', 'rescheduled'); await change('date', '2035-07-01');
  await check('contacted');
  await change('action', 'confirmed');
  assert.equal(document.querySelector('[name="date"]'), null);
  assert.equal(document.querySelector('[name="contacted"]').checked, false);
  assert.deepEqual(assessed.at(-1), { start_at: request.start_at, end_at: request.end_at });
  await check('contacted'); await submit();
  assert.equal(calls[0].body.start_at, undefined);
  assert.equal(calls[0].body.end_at, undefined);
});

test('editing reschedule times requires renewed agreement and warning acknowledgement', async () => {
  await render({ request: { ...request, appointment_type: 'tree_planting' },
    assessmentFor: () => ({ status: 'unsafe', reason: 'High water.' }) });
  await change('action', 'rescheduled');
  await check('contacted'); await check('acceptCaution');
  await change('startTime', '07:00');
  assert.equal(document.querySelector('[name="contacted"]').checked, false);
  assert.equal(document.querySelector('[name="acceptCaution"]').checked, false);
  await submit(); assert.equal(calls.length, 0);
  await check('contacted'); await submit(); assert.equal(calls.length, 0);
  await check('acceptCaution'); await submit(); assert.equal(calls.length, 1);
});

test('declining a request requires a reason and submits no schedule changes', async () => {
  await render();
  await change('action', 'declined');
  assert.equal(document.querySelector('[name="contacted"]'), null);
  assert.equal(document.querySelector('[name="date"]'), null);
  await submit(); assert.equal(calls.length, 0);
  await change('decisionNotes', 'Staff are unavailable.'); await submit();
  assert.deepEqual(calls[0].body, { action: 'declined', contacted: false, decision_notes: 'Staff are unavailable.' });
  assert.match(saved[0], /declined/);
});

test('a past request prompts explicit rescheduling instead of exposing editable times by default', async () => {
  await render({ request: { ...request, start_at: '2020-06-12T08:00:00+08:00', end_at: '2020-06-12T09:30:00+08:00' } });
  await check('contacted'); await submit();
  assert.equal(calls.length, 0);
  assert.match(document.body.textContent, /requested time has passed.*Reschedule and confirm/);
  assert.equal(document.querySelector('[name="date"]'), null);
  await change('action', 'rescheduled');
  await change('date', '2035-06-12'); await change('endTime', '07:00');
  await check('contacted'); await submit();
  assert.equal(calls.length, 0);
  assert.match(document.body.textContent, /end time later/);
  await change('endTime', '09:00'); await check('contacted'); await submit();
  assert.equal(calls.length, 1);
});

test('an older request without email can obtain the missing recipient before confirmation', async () => {
  await render({ request: { ...request, email: null } });
  assert.match(document.body.textContent, /older request has no email/);
  await check('contacted'); await submit(); assert.equal(calls.length, 0);
  await change('email', 'coordinator@example.org'); await submit();
  assert.equal(calls[0].body.email, 'coordinator@example.org');
  assert.equal(calls[0].body.start_at, undefined);
});
