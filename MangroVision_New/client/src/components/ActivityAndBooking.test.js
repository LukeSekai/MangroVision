import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

let server, dom, root, createRoot, AppointmentForm, FieldActivities, AuthScreen, OrganizationAccess, usePlanterAuthStore, bookingPolicy;
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
  ({ bookingPolicy } = await server.ssrLoadModule('/src/like/booking.js'));
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
  bookingPolicy.enforcePlantingHours = false;
  bookingPolicy.enforceActivityAccessWindow = false;
  localStorage.clear(); sessionStorage.clear();
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
  globalThis.fetch = async (url, options) => {
    if (String(url).includes('/api/tides/forecast')) return Response.json({ available: false });
    requests.push(JSON.parse(options.body));
    if (requests.length === 1) throw new TypeError('Network unavailable');
    return { ok: true, json: async () => ({ reference: 'LIKE-TEST', status: 'pending' }) };
  };
  await act(async () => root.render(createElement(AppointmentForm)));
  for (const [name, value] of Object.entries({ appointment_type: 'tree_planting', organization: 'Example organization', contact_name: 'Coordinator', phone: '09123456789', email: 'head@example.org' })) await change(name, value);
  await submit();
  assert.equal(document.querySelector('[name="title"]'), null);
  assert.doesNotMatch(document.body.textContent, /Activity name/);
  const date = new Date(Date.now() + 7 * 86400000).toISOString().slice(0, 10);
  await submit();
  const dateControl = document.querySelector('[name="date"]');
  assert.equal(document.getElementById(dateControl.getAttribute('aria-describedby')).textContent, 'Choose your preferred date.');
  for (const [name, value] of Object.entries({ date, start_time: '08:00', participants: '20' })) await change(name, value);
  assert.equal(document.querySelector('[name="end_time"]').value, '10:00');
  assert.equal(document.querySelector('[name="end_time"]').readOnly, true);
  await act(async () => document.querySelector('[name="consent"]').click());
  await submit();
  assert.match(document.body.textContent, /Check your connection and retry/);
  assert.equal(document.querySelector('[name="participants"]').value, '20');
  await submit();
  assert.equal(requests.length, 2);
  assert.equal(requests[0].submission_key, requests[1].submission_key);
  assert.equal(requests[0].title, 'Tree planting');
  assert.equal(requests[0].end_at, `${date}T10:00:00+08:00`);
  assert.match(document.querySelector('[role="status"]').textContent, /LIKE-TEST/);
});

async function openWebsiteSchedule(type = 'tree_planting') {
  await act(async () => root.render(createElement(AppointmentForm)));
  for (const [name, value] of Object.entries({ appointment_type: type, organization: 'Example organization', contact_name: 'Coordinator', phone: '09123456789', email: 'head@example.org' })) await change(name, value);
  await submit();
}
function websiteForecast() {
  const date = new Intl.DateTimeFormat('en-CA', { timeZone: 'Asia/Manila', year: 'numeric', month: '2-digit', day: '2-digit' }).format(new Date(Date.now() + 2 * 86400000));
  const base = Date.parse(`${date}T00:00:00+08:00`);
  return { date, payload: { available: true, stale: false, fetched_at: Date.now() / 1000,
    datum: 'MSL', datum_reference: 'test', series_available: true, source: 'Open-Meteo Marine API',
    heights: Array.from({ length: 73 }, (_, hour) => ({ timestamp: base / 1000 + hour * 3600,
      height_m: hour >= 24 || hour === 8 || hour === 9 ? -1 : hour === 7 || hour === 10 ? 0 : 1 })),
  } };
}

test('website uses a tide suggestion and blocks sending when the whole visit does not fit', async () => {
  const { date, payload } = websiteForecast();
  let sent = 0;
  globalThis.fetch = async (url) => { if (url.includes('/appointments')) sent++; return Response.json(payload); };
  await openWebsiteSchedule();
  assert.match(document.body.textContent, /Choose a date to see suggested planting times/);
  await change('date', date); await settle();
  assert.match(document.body.textContent, /7:00 AM – 10:00 AM/);
  assert.match(document.body.textContent, /Start by 8:00 AM/);
  await act(async () => [...document.querySelectorAll('button')].find((button) => button.textContent === 'Use 7:00 AM').click());
  assert.equal(document.querySelector('[name="start_time"]').value, '07:00');
  assert.equal(document.querySelector('[name="end_time"]').value, '09:00');
  assert.match(document.body.textContent, /full two-hour visit fits/);
  await change('start_time', '09:00');
  assert.equal(document.querySelector('[name="end_time"]').value, '11:00');
  assert.match(document.body.textContent, /rise above the planting limit during your two-hour visit/);
  await change('participants', '20');
  await act(async () => document.querySelector('[name="consent"]').click());
  await submit();
  assert.equal(sent, 0);
  assert.equal(document.querySelector('[name="start_time"]').getAttribute('aria-invalid'), 'true');
  await change('date', '2039-01-01');
  assert.match(document.body.textContent, /does not cover the full booking period/);
  assert.equal(document.querySelector('.like-tide-windows'), null);
  assert.equal(document.querySelector('.like-tide-match'), null);
  assert.equal(document.querySelector('[name="start_time"]').value, '09:00');
});

test('LIKE offers 6 AM to 5 PM during all-day low water and rejects an early or late visit', async () => {
  bookingPolicy.enforcePlantingHours = true;
  const { date, payload } = websiteForecast();
  payload.heights = payload.heights.map((row) => ({ ...row, height_m: 0.2 }));
  let sent = 0;
  globalThis.fetch = async (url) => { if (url.includes('/appointments')) sent++; return Response.json(payload); };
  await openWebsiteSchedule();
  await change('date', date); await settle();
  assert.match(document.body.textContent, /6:00 AM – 5:00 PM/);
  assert.match(document.body.textContent, /Start by 3:00 PM/);
  assert.equal(document.querySelector('[name="start_time"]').min, '06:00');
  assert.equal(document.querySelector('[name="start_time"]').max, '15:00');
  await change('participants', '20');
  await act(async () => document.querySelector('[name="consent"]').click());
  for (const start of ['04:30', '15:15']) {
    await change('start_time', start); await submit();
    assert.match(document.body.textContent, /6 AM to 5 PM/);
    assert.equal(sent, 0);
    assert.equal(document.querySelector('.like-tide-match'), null);
  }
  await change('start_time', '15:00');
  assert.equal(document.querySelector('[name="end_time"]').value, '17:00');
  assert.ok(document.querySelector('.like-tide-match'));
});

test('testing allows the 4:45 PM booking through the form without closed-hours messages', async () => {
  const { date, payload } = websiteForecast();
  payload.heights = payload.heights.map((row) => ({ ...row, height_m: 0.2 }));
  const requests = [];
  globalThis.fetch = async (url, options) => {
    if (!url.includes('/appointments')) return Response.json(payload);
    requests.push(JSON.parse(options.body));
    return Response.json({ reference: 'LIKE-EVENING', status: 'pending' });
  };
  await openWebsiteSchedule();
  await change('date', date); await settle();
  const start = document.querySelector('[name="start_time"]');
  assert.equal(start.min, '');
  assert.equal(start.max, '21:59');
  await change('start_time', '16:45');
  assert.equal(document.querySelector('[name="end_time"]').value, '18:45');
  assert.doesNotMatch(document.body.textContent, /6 AM–5 PM|at or after 6 AM|finish by 5 PM|6 AM to 5 PM/);
  await change('participants', '20');
  await act(async () => document.querySelector('[name="consent"]').click());
  await submit();
  assert.equal(requests.length, 1);
  assert.equal(requests[0].start_at, `${date}T16:45:00+08:00`);
  assert.equal(requests[0].end_at, `${date}T18:45:00+08:00`);
  assert.match(document.querySelector('[role="status"]').textContent, /LIKE-EVENING/);
});

test('failed refresh hides old tide suggestions and recovery restores them without losing chosen times', async () => {
  const { date, payload } = websiteForecast();
  let attempts = 0;
  globalThis.fetch = async () => {
    if (++attempts === 2) throw new TypeError('Network unavailable');
    return Response.json(payload);
  };
  await openWebsiteSchedule();
  await change('date', date); await settle();
  await change('start_time', '07:00');
  const refresh = () => [...document.querySelectorAll('button')].find((button) => button.textContent === 'Refresh tides');
  await act(async () => refresh().click()); await settle();
  assert.equal(document.querySelector('.like-tide-windows'), null);
  assert.equal(document.querySelector('.like-tide-match'), null);
  assert.match(document.body.textContent, /could not load the tide forecast/);
  assert.equal(document.querySelector('[name="end_time"]').value, '09:00');
  await act(async () => refresh().click()); await settle();
  assert.notEqual(document.querySelector('.like-tide-windows'), null);
  assert.equal(document.querySelector('[name="start_time"]').value, '07:00');
});

test('other website activities also use two-hour slots without fetching planting guidance', async () => {
  await openWebsiteSchedule('field_visit');
  assert.equal(document.querySelector('.like-tide-guide'), null);
  await change('start_time', '11:30');
  assert.equal(document.querySelector('[name="end_time"]').value, '13:30');
  await change('start_time', '22:00');
  await submit();
  assert.match(document.body.textContent, /before 10 PM/);
  assert.equal(document.activeElement.name, 'date');
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

test('field access uses a shared code and username, with first-time and returning choices', async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, body: JSON.parse(options.body) });
    return Response.json({ planter: { id: 4, participant_username: 'Ana', participant_slot: 1, planting_schedule_id: 3 } });
  };
  localStorage.setItem('mv_last_planter_username', 'Ana');
  await act(async () => root.render(createElement(AuthScreen)));
  await settle();
  assert.equal(calls.length, 0);
  assert.doesNotMatch(document.body.textContent, /Register|Create Organization Account/);
  assert.match(document.body.textContent, /assigned points and saved progress/);
  assert.match(document.body.textContent, /First time/);
  assert.match(document.body.textContent, /Returning planter/);
  assert.doesNotMatch(document.body.textContent, /recovery|reset your device/i);
  assert.equal(document.querySelector('[name="recovery_code"]'), null);
  assert.equal(document.querySelector('[name="password"]'), null);
  assert.ok(document.querySelector('[name="username"]'));
  await change('access_code', 'K7MP-4R9X');
  await change('username', 'Ana');
  await submit();
  assert.equal(calls.length, 1);
  assert.match(calls[0].url, /\/planter-auth\/login$/);
  assert.equal(calls[0].body.access_code, 'K7MP4R9X');
  assert.equal(calls[0].body.password, undefined);
  assert.equal(calls[0].body.username, 'Ana');
  assert.equal(calls[0].body.mode, 'return');
  assert.match(calls[0].body.browser_key, /^[a-f0-9]{48}$/);
});

test('planter sign-in places code errors beside the input and clears service errors when editing', async () => {
  let attempt = 0;
  globalThis.fetch = async () => ++attempt === 1
    ? Response.json({ detail: [{ loc: ['body', 'device_key'], type: 'missing', msg: 'Field required' }] }, { status: 422 })
    : Response.json({ detail: [{ loc: ['body', 'access_code'], type: 'credential_error', msg: 'Check this code with your group leader.' }] }, { status: 401 });
  await act(async () => root.render(createElement(AuthScreen)));
  await change('access_code', 'k7mp–4r9x');
  await submit();
  assert.match(document.querySelector('[role="alert"]').textContent, /temporarily unavailable/);
  assert.equal(document.querySelector('[name="access_code"]').getAttribute('aria-invalid'), 'false');
  await change('access_code', 'ABCD-EFGH');
  assert.equal(document.querySelector('.field-error'), null);
  await submit();
  await settle();
  assert.equal(document.activeElement.name, 'access_code');
  assert.match(document.querySelector('.form-field-error').textContent, /group leader/);
  assert.equal(document.querySelector('.field-error'), null);
  assert.equal(document.querySelector('.form-error-summary'), null);
});

test('staff issue one code for a 100-person activity', async () => {
  const calls = [];
  let created = 0;
  globalThis.fetch = async (url, options) => {
    calls.push({ url, body: options.body ? JSON.parse(options.body) : null });
    return Response.json({ access: options.body ? { code: 'K7MP-4R9X', start_at: '2026-10-15T00:00:00Z', end_at: '2026-10-15T03:00:00Z' } : null });
  };
  await act(async () => root.render(createElement(OrganizationAccess, {
    organization: { organization_id: 7, organization_name: 'Oton Youth', registration_pending: true },
    activity: { id: 3, title: 'Tree planting', expected_planters: 100 }, onCreated: () => { created += 1; },
  })));
  assert.equal(document.querySelector('[name="username"]'), null);
  assert.equal(document.querySelector('[name="password"]'), null);
  assert.match(document.body.textContent, /One code for all 100 participants/);
  await act(async () => [...document.querySelectorAll('button')].find((button) => button.textContent === 'Show organization code').click());
  await settle();
  await act(async () => [...document.querySelectorAll('button')].find((button) => button.textContent === 'Create organization code').click());
  await settle();
  assert.equal(calls.length, 2);
  assert.match(calls[1].url, /\/organizations\/7\/activities\/3\/access$/);
  assert.deepEqual(calls[1].body, { action: 'issue' });
  assert.equal(created, 1);
  assert.match(document.body.textContent, /K7MP-4R9X/);
  assert.match(document.body.textContent, /organization code is ready/);
  assert.match(document.body.textContent, /before and after the scheduled planting time/);
  assert.doesNotMatch(document.body.textContent, /Opens .*Closes/);
});

test('existing codes are read without issuance and replacement requires confirmation', async () => {
  bookingPolicy.enforceActivityAccessWindow = true;
  const calls = [];
  globalThis.fetch = async (url, options) => {
    calls.push({ url, method: options.method || 'GET', body: options.body ? JSON.parse(options.body) : null });
    return Response.json({ access: { code: calls.at(-1).method === 'POST' ? 'G6FR-3VD8' : 'K7MP-4R9X', revoked: false,
      start_at: '2026-10-15T00:00:00Z', end_at: '2026-10-15T03:00:00Z' } });
  };
  await act(async () => root.render(createElement(OrganizationAccess, {
    organization: { id: 4, organization_id: 7, organization_name: 'Oton Youth', participant_count: 1, registration_pending: false },
    activity: { id: 3, title: 'Tree planting', expected_planters: 1 },
  })));
  assert.equal(calls.length, 0);
  const click = async (label) => act(async () => [...document.querySelectorAll('button')].find((button) => button.textContent === label).click());
  await click('Show organization code'); await settle();
  assert.equal(calls[0].method, 'GET');
  assert.match(document.body.textContent, /K7MP-4R9X/);
  assert.match(document.body.textContent, /Opens .*Closes/);
  await click('Replace code');
  assert.equal(calls.length, 1);
  assert.equal(document.querySelector('.modal-title').textContent, 'Replace organization code?');
  assert.equal(document.activeElement.textContent, 'Cancel');
  await click('Cancel');
  assert.equal(calls.length, 1);
  await click('Replace code');
  await act(async () => document.querySelector('.modal-actions .btn-primary').click()); await settle();
  assert.match(calls[1].url, /\/organizations\/7\/activities\/3\/access$/);
  assert.deepEqual(calls[1].body, { action: 'replace' });
  assert.match(document.body.textContent, /G6FR-3VD8/);
});

test('switch here requires modal confirmation and Cancel keeps the existing login', async () => {
  const calls = [];
  globalThis.fetch = async (_url, options) => {
    const body = JSON.parse(options.body); calls.push(body);
    return body.switch_token ? Response.json({ planter: { id: 4, participant_slot: 1, participant_username: 'Ana', planting_schedule_id: 3 } })
      : Response.json({ detail: { kind: 'session_in_use', message: 'Switch access?', switch_token: 'fictional-ticket' } }, { status: 409 });
  };
  await act(async () => root.render(createElement(AuthScreen)));
  await act(async () => [...document.querySelectorAll('button')].find((button) => button.textContent === 'Returning planter').click());
  await change('access_code', 'K7MP-4R9X'); await change('username', 'Ana'); await submit();
  assert.equal(calls.length, 1); assert.equal(usePlanterAuthStore.getState().isAuthenticated, false);
  assert.match(document.querySelector('[role="dialog"]').textContent, /other login.*will end/);
  await act(async () => document.querySelector('.modal-actions .btn-secondary').click());
  assert.equal(document.querySelector('[role="dialog"]'), null);
  assert.equal(calls.length, 1);
  assert.equal(document.querySelector('[name="username"]').value, 'Ana');
  await change('username', 'Ben');
  await change('username', 'Ana'); await submit();
  await act(async () => [...document.querySelectorAll('button')].find((button) => button.textContent === 'Switch here').click());
  await settle();
  assert.equal(calls.at(-1).switch_token, 'fictional-ticket');
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, true);
});

test('a failed browser switch closes the review and keeps the credentials for retry', async () => {
  globalThis.fetch = async (_url, options) => JSON.parse(options.body).switch_token
    ? Response.json({ detail: 'This switch request expired. Sign in again.' }, { status: 409 })
    : Response.json({ detail: { kind: 'session_in_use', message: 'Switch access?', switch_token: 'fictional-ticket' } }, { status: 409 });
  await act(async () => root.render(createElement(AuthScreen)));
  await act(async () => [...document.querySelectorAll('button')].find((button) => button.textContent === 'Returning planter').click());
  await change('access_code', 'K7MP-4R9X'); await change('username', 'Ana'); await submit();
  await act(async () => document.querySelector('.modal-actions .btn-primary').click()); await settle();
  assert.equal(document.querySelector('[role="dialog"]'), null);
  assert.match(document.querySelector('[role="alert"]').textContent, /expired/);
  assert.equal(document.querySelector('[name="username"]').value, 'Ana');
  assert.equal(document.querySelector('[name="access_code"]').value, 'K7MP-4R9X');
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, false);
});

test('first-time signup on a remembered browser redirects to its existing username', async () => {
  globalThis.fetch = async (url) => url.endsWith('/join')
    ? Response.json({ activity: { title: 'Tree planting', organization_name: 'Test group' } })
    : Response.json({ detail: { kind: 'already_joined', message: 'Return instead.', participant_username: 'Ana' } }, { status: 409 });
  await act(async () => root.render(createElement(AuthScreen)));
  await act(async () => [...document.querySelectorAll('button')].find((button) => button.textContent === 'First time').click());
  await change('access_code', 'K7MP-4R9X'); await submit();
  await change('username', 'AnotherName'); await submit();
  assert.equal(document.querySelector('[name="username"]').value, 'Ana');
  assert.match(document.body.textContent, /already joined/);
  assert.equal([...document.querySelectorAll('button')].find((button) => button.textContent === 'Returning planter').getAttribute('aria-pressed'), 'true');
});

test('first visit asks only for code, then requires a unique username inside Field', async () => {
  const calls = [];
  globalThis.fetch = async (url, options) => {
    const body = JSON.parse(options.body); calls.push({ url, body });
    if (url.endsWith('/join')) return Response.json({ activity: { title: 'Tree planting', organization_name: 'Test group' } });
    if (body.username === 'Taken') return Response.json({ detail: [{ loc: ['body', 'username'], type: 'credential_error', msg: 'That username is already used in this organization.' }] }, { status: 409 });
    return Response.json({ planter: { id: 4, participant_username: body.username, participant_slot: 1, planting_schedule_id: 3 } });
  };
  await act(async () => root.render(createElement(AuthScreen)));
  assert.equal(document.querySelector('[name="username"]'), null);
  assert.equal(document.querySelector('[name="previous_code"]'), null);
  assert.doesNotMatch(document.body.textContent, /previous individual/i);
  assert.equal(document.querySelector('[name="access_code"]').value, '');
  await submit();
  assert.equal(calls.length, 0);
  await change('access_code', 'K7MP-4R9X'); await submit();
  assert.deepEqual(calls[0].body, { access_code: 'K7MP4R9X' });
  assert.match(calls[0].url, /\/join$/);
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, false);
  assert.ok(document.querySelector('.field-shell .field-header'));
  assert.equal(document.activeElement.id, 'field-username-heading');
  assert.match(document.body.textContent, /Test group/);
  assert.equal(document.querySelector('[name="access_code"]'), null);
  await submit();
  assert.equal(calls.length, 1);
  await change('username', 'Taken'); await submit();
  assert.match(document.querySelector('.form-field-error').textContent, /already used/);
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, false);
  await change('username', 'Ana'); await submit();
  assert.equal(calls.at(-1).body.mode, 'first');
  assert.equal(calls.at(-1).body.access_code, 'K7MP4R9X');
  assert.equal(calls.at(-1).body.previous_code, undefined);
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, true);
});

test('leaving username setup allows another code without reserving a participant', async () => {
  let calls = 0;
  globalThis.fetch = async () => { calls++; return Response.json({ activity: { title: 'Tree planting', organization_name: 'Test group' } }); };
  await act(async () => root.render(createElement(AuthScreen)));
  await change('access_code', 'K7MP-4R9X'); await submit();
  await change('username', 'Unfinished');
  await act(async () => [...document.querySelectorAll('button')].find((button) => button.textContent === 'Back to sign in').click());
  assert.equal(document.querySelector('[name="username"]'), null);
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, false);
  assert.equal(calls, 1);
  await change('access_code', 'G6FR-3VD8'); await submit();
  assert.equal(document.querySelector('[name="username"]').value, '');
  assert.equal(calls, 2);
});


test('a future planting code explains its opening time and keeps the code entry editable',async () => {
  const message='It is not your planting schedule yet. Sign-in opens on Oct 20, 2026 at 08:00 AM (Philippine time).';
  globalThis.fetch=async()=>Response.json({ detail:{ kind:'activity_not_started',field:'access_code',message } },{ status:403 });
  await act(async()=>root.render(createElement(AuthScreen)));
  await change('access_code','K7MP-4R9X'); await submit();
  assert.match(document.body.textContent,/not your planting schedule yet/);
  assert.match(document.body.textContent,/08:00 AM/);
  assert.equal(document.querySelector('[name="access_code"]').value,'K7MP-4R9X');
  assert.equal(document.querySelector('[name="username"]'),null);
  assert.equal(usePlanterAuthStore.getState().isAuthenticated,false);
});

test('a code that expires during username setup returns to the code screen with its reason',async () => {
  globalThis.fetch=async(url)=>url.endsWith('/join')
    ? Response.json({ activity:{ title:'Tree planting',organization_name:'Test group' } })
    : Response.json({ detail:{ kind:'code_expired',field:'access_code',message:'This organization code expired when your planting activity ended. Your saved progress is kept.' } },{ status:403 });
  await act(async()=>root.render(createElement(AuthScreen)));
  await change('access_code','K7MP-4R9X'); await submit();
  await change('username','Ana'); await submit();
  assert.ok(document.querySelector('[name="access_code"]'));
  assert.equal(document.querySelector('[name="username"]'),null);
  assert.match(document.body.textContent,/code expired/);
  assert.doesNotMatch(document.body.textContent,/service is temporarily unavailable/);
});

test('returning with an unregistered username explains how to join without creating a participant',async () => {
  globalThis.fetch=async()=>Response.json({ detail:{ kind:'username_not_found',field:'username',
    message:'Username not found for this organization. Check your username and organization code. If you have not chosen a username here yet, choose First time.' } },{ status:401 });
  await act(async()=>root.render(createElement(AuthScreen)));
  await act(async()=>[...document.querySelectorAll('button')].find((button)=>button.textContent==='Returning planter').click());
  await change('access_code','K7MP-4R9X'); await change('username','Ana'); await submit();
  assert.match(document.body.textContent,/Username not found for this organization/);
  assert.match(document.body.textContent,/choose First time/);
  assert.equal(document.querySelector('[name="username"]').getAttribute('aria-invalid'),'true');
  assert.equal(usePlanterAuthStore.getState().isAuthenticated,false);
});

test('a temporary database failure explains the reason and can be retried without reloading',async () => {
  let failed=true;
  globalThis.fetch=async()=>failed
    ? Response.json({ detail:{ kind:'sign_in_database_unavailable',message:'The sign-in database could not be reached. Please try again shortly.' } },{ status:503 })
    : Response.json({ activity:{ title:'Tree planting',organization_name:'Test group' } });
  await act(async()=>root.render(createElement(AuthScreen)));
  await change('access_code','K7MP-4R9X'); await submit();
  assert.match(document.body.textContent,/database could not be reached/);
  assert.equal(document.querySelector('[name="access_code"]').value,'K7MP-4R9X');
  failed=false; await submit();
  assert.match(document.body.textContent,/Choose your username/);
  assert.ok(document.querySelector('[name="username"]'));
  assert.doesNotMatch(document.body.textContent,/database could not be reached/);
});
