import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

let server, dom, root, createRoot, AppointmentForm, FieldActivities, AuthScreen, OrganizationAccess, usePlanterAuthStore;
const globals = new Map();
before(async () => {
  dom = new JSDOM('<div id="root"></div>', { pretendToBeVisual: true, url: 'http://localhost' });
  for (const [key, value] of Object.entries({ window: dom.window, document: dom.window.document, navigator: dom.window.navigator,
    HTMLElement: dom.window.HTMLElement, Element: dom.window.Element, SVGElement: dom.window.SVGElement,
    localStorage: dom.window.localStorage, sessionStorage: dom.window.sessionStorage,
    IS_REACT_ACT_ENVIRONMENT: true, requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window), fetch: globalThis.fetch,
  })) {
    globals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  ({ createRoot } = await import('react-dom/client'));
  window.HTMLElement.prototype.scrollIntoView = () => {};
  server = await createServer({ configFile: false, appType: 'custom', server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    cacheDir: 'node_modules/.vite-activity-booking-tests', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ AppointmentForm } = await server.ssrLoadModule('/src/like/LikeWebsite.jsx'));
  ({ default: FieldActivities } = await server.ssrLoadModule('/src/components/FieldActivities.jsx'));
  ({ AuthScreen } = await server.ssrLoadModule('/src/pages/FieldApp.jsx'));
  ({ default: OrganizationAccess } = await server.ssrLoadModule('/src/components/OrganizationAccess.jsx'));
  ({ usePlanterAuthStore } = await server.ssrLoadModule('/src/stores/planterAuthStore.js'));
});
after(async () => {
  await server?.close(); dom?.window.close();
  for (const [key, descriptor] of globals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
  }
});
beforeEach(() => {
  root = createRoot(document.getElementById('root'));
  globalThis.fetch = async () => assert.fail('Unexpected network call');
  usePlanterAuthStore.setState({ status: 'idle', error: '', isAuthenticated: false, planter: null, token: null });
});
afterEach(async () => { await act(async () => root.unmount()); });
const settle = async () => act(async () => { await new Promise((resolve) => setTimeout(resolve, 40)); });
async function change(name, value) {
  const control = document.querySelector(`[name="${name}"]`);
  await act(async () => {
    const prototype = control.tagName === 'SELECT' ? window.HTMLSelectElement.prototype : window.HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(prototype, 'value').set.call(control, value);
    control.dispatchEvent(new window.Event(control.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true }));
  });
}
async function submit() { await act(async () => document.querySelector('form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true }))); await settle(); }

test('website step errors identify and focus the fields without transmitting a request', async () => {
  await act(async () => root.render(createElement(AppointmentForm)));
  await submit();
  assert.equal(document.activeElement.name, 'appointment_type');
  assert.equal(document.querySelector('[name="email"]').getAttribute('aria-invalid'), 'true');
  const email = document.querySelector('[name="email"]');
  assert.equal(document.getElementById(email.getAttribute('aria-describedby')).textContent, 'Enter an email address.');
  await change('appointment_type', 'tree_planting'); await change('organization', 'Example organization');
  await change('contact_name', 'Coordinator'); await change('phone', 'abc1234'); await change('email', 'broken');
  await submit();
  assert.equal(document.activeElement.name, 'phone');
  assert.match(document.body.textContent, /valid phone number/);
  assert.match(document.body.textContent, /email address/);
});

test('a network retry keeps website inputs and reuses the same submission key', async () => {
  const requests = [];
  globalThis.fetch = async (_url, options) => {
    requests.push(JSON.parse(options.body));
    if (requests.length === 1) throw new TypeError('Network unavailable');
    return { ok: true, json: async () => ({ reference: 'LIKE-TEST', status: 'pending' }) };
  };
  await act(async () => root.render(createElement(AppointmentForm)));
  for (const [name, value] of Object.entries({ appointment_type: 'tree_planting', organization: 'Example organization', contact_name: 'Coordinator', phone: '09123456789', email: 'head@example.org' })) await change(name, value);
  await submit();
  const date = new Date(Date.now() + 7 * 86400000).toISOString().slice(0, 10);
  await submit();
  const dateControl = document.querySelector('[name="date"]');
  assert.equal(document.getElementById(dateControl.getAttribute('aria-describedby')).textContent, 'Choose your preferred date.');
  for (const [name, value] of Object.entries({ date, start_time: '08:00', end_time: '10:00', participants: '20' })) await change(name, value);
  await act(async () => document.querySelector('[name="consent"]').click());
  await submit();
  assert.match(document.body.textContent, /Check your connection and retry/);
  assert.equal(document.querySelector('[name="participants"]').value, '20');
  await submit();
  assert.equal(requests.length, 2);
  assert.equal(requests[0].submission_key, requests[1].submission_key);
  assert.match(document.querySelector('[role="status"]').textContent, /LIKE-TEST/);
});

test('planter refresh displays revised dates and cancellations, with tide coverage clearly unavailable', async () => {
  const start = new Date(Date.now() + 20 * 86400000).toISOString();
  const end = new Date(Date.parse(start) + 3600000).toISOString();
  let loads = 0;
  globalThis.fetch = async (url) => ({ ok: true, json: async () => String(url).includes('/activities') ? { activities: [{
    id: 1, title: 'Organization planting', appointment_type: 'tree_planting', start_at: start, end_at: end,
    status: ++loads === 1 ? 'confirmed' : 'cancelled', change_notice: { rescheduled: true, changed_at: new Date().toISOString(), previous_start_at: new Date(Date.parse(start) - 86400000).toISOString() },
  }] } : { available: false, message: 'Forecast not available yet.' } });
  await act(async () => root.render(createElement(FieldActivities, { organizationId: 4 })));
  await settle();
  assert.match(document.body.textContent, /Rescheduled/);
  assert.match(document.body.textContent, /Use the current schedule/);
  assert.match(document.body.textContent, /Tide advisory unavailable/);
  await act(async () => [...document.querySelectorAll('button')].find((button) => button.textContent === 'Refresh activities').click());
  await settle();
  assert.match(document.body.textContent, /This activity is cancelled/);
  assert.doesNotMatch(document.body.textContent, /No elevated-water overlap forecast/);
});

test('field access needs only a planter code, without recovery or registration', async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, body: JSON.parse(options.body) });
    return Response.json({ planter: { id: 4, username: 'oton-youth', participant_slot: 1 } });
  };
  await act(async () => root.render(createElement(AuthScreen)));
  await settle();
  assert.equal(calls.length, 0);
  assert.doesNotMatch(document.body.textContent, /Register|Create Organization Account/);
  assert.match(document.body.textContent, /any browser or device/);
  assert.doesNotMatch(document.body.textContent, /recovery|reset your device/i);
  assert.equal(document.querySelector('[name="recovery_code"]'), null);
  assert.equal(document.querySelector('[name="password"]'), null);
  assert.equal(document.querySelector('[name="username"]'), null);
  await change('access_code', 'K7MP-4R9X');
  await submit();
  assert.equal(calls.length, 1);
  assert.match(calls[0].url, /\/planter-auth\/login$/);
  assert.equal(calls[0].body.access_code, 'K7MP-4R9X');
  assert.equal(calls[0].body.password, undefined);
  assert.equal(calls[0].body.username, undefined);
  assert.deepEqual(calls[0].body, { access_code: 'K7MP-4R9X' });
});

test('staff create individual codes with participant count and no credentials', async () => {
  const calls = [];
  let created = 0;
  globalThis.fetch = async (url, options) => {
    calls.push({ url, body: JSON.parse(options.body) });
    return Response.json({ id: 4, codes: [{ participant_slot: 1, code: 'K7MP-4R9X', revoked: false }] });
  };
  await act(async () => root.render(createElement(OrganizationAccess, {
    organization: { organization_id: 7, organization_name: 'Oton Youth', registration_pending: true },
    expectedParticipants: 20, onCreated: () => { created += 1; },
  })));
  assert.equal(document.querySelector('[name="username"]'), null);
  assert.equal(document.querySelector('[name="password"]'), null);
  assert.equal(document.querySelector('[name="participant_count"]').value, '20');
  await submit();
  assert.equal(calls.length, 1);
  assert.match(calls[0].url, /\/api\/planters\/$/);
  assert.equal(calls[0].body.organization_id, 7);
  assert.equal(calls[0].body.participant_count, 20);
  assert.equal(calls[0].body.password, undefined);
  assert.equal(calls[0].body.username, undefined);
  assert.equal(created, 1);
  assert.match(document.body.textContent, /K7MP-4R9X/);
  assert.match(document.body.textContent, /Planter codes created/);
});

test('existing codes are read without issuance and replacement requires confirmation', async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, method: options.method || 'GET', body: options.body ? JSON.parse(options.body) : null });
    return Response.json({ planter_id: 4, codes: [{ participant_slot: 1, code: calls.at(-1).method === 'POST' ? 'G6FR-3VD8' : 'K7MP-4R9X', revoked: false }] });
  };
  await act(async () => root.render(createElement(OrganizationAccess, {
    organization: { id: 4, organization_id: 7, organization_name: 'Oton Youth', participant_count: 1, registration_pending: false },
  })));
  assert.equal(calls.length, 0);
  const click = async (label) => act(async () => [...document.querySelectorAll('button')].find((button) => button.textContent === label).click());
  await click('Show planter codes'); await settle();
  assert.equal(calls[0].method, 'GET');
  assert.match(document.body.textContent, /K7MP-4R9X/);
  await click('Replace code');
  assert.equal(calls.length, 1);
  await click('Confirm'); await settle();
  assert.match(calls[1].url, /\/4\/participants\/1\/access-code$/);
  assert.deepEqual(calls[1].body, { action: 'regenerate' });
  assert.match(document.body.textContent, /G6FR-3VD8/);
});
