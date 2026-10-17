import test, { before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

let server, dom, usePlanterAuthStore, requests, loginFailure, offline, unauthorized;
const originals = new Map();
const code = 'K7MP-4R9X';
const points = [{ assignment_point_id: 11, status: 'completed' }, { assignment_point_id: 12, status: 'pending' }];
before(async () => {
  dom = new JSDOM('', { url: 'https://field.example.test' });
  for (const [key, value] of Object.entries({ document: dom.window.document, localStorage: dom.window.localStorage,
    sessionStorage: dom.window.sessionStorage, fetch: async (input, init = {}) => {
      if (offline) throw new TypeError('Network unavailable');
      const path = new URL(input, 'https://field.example.test').pathname;
      const body = init.body ? JSON.parse(init.body) : null;
      requests.push({ path, body });
      if (path.endsWith('/join')) return loginFailure ? loginFailure() : Response.json({ activity: { title: 'Tree planting', organization_name: 'Test group' } });
      if (path.endsWith('/login') && loginFailure) return loginFailure();
      if (path.endsWith('/logout')) return Response.json({ status: 'ok' });
      if (!path.endsWith('/login') && unauthorized) return Response.json({ detail: 'Login ended.' }, { status: 401 });
      return Response.json({ planter: { id: 7, organization_id: 5, participant_count: 10, participant_slot: 1,
        participant_username: body?.username || 'Ana', planting_schedule_id: 2 },
        ...(path.endsWith('/field-points') ? { points, project_sites: { features: [] },
          site_access: [{ label: 'LIKE entrance', entrance: [10.77999592, 122.625672] }] } : {}) });
    } })) {
    originals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  server = await createServer({ configFile: false, appType: 'custom', server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    cacheDir: 'node_modules/.vite-planter-code-tests', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ usePlanterAuthStore } = await server.ssrLoadModule('/src/stores/planterAuthStore.js'));
});
after(async () => {
  await server?.close(); dom?.window.close();
  for (const [key, descriptor] of originals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
  }
});
beforeEach(() => {
  localStorage.clear(); sessionStorage.clear();
  for (const entry of document.cookie.split(';')) document.cookie = `${entry.split('=')[0].trim()}=; Max-Age=0; Path=/`;
  requests = []; loginFailure = null; offline = unauthorized = false;
  usePlanterAuthStore.setState({ token: null, planter: null, isAuthenticated: false, status: 'idle', error: '' });
});
const login = (options) => usePlanterAuthStore.getState().login(code, 'Ana', options);

test('code entry opens username setup without authenticating or storing the code', async () => {
  const activity = await usePlanterAuthStore.getState().checkOrganizationCode(code);
  assert.equal(activity.organization_name, 'Test group');
  assert.deepEqual(requests[0], { path: '/api/planter-auth/join', body: { access_code: 'K7MP4R9X' } });
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, false);
  assert.equal(usePlanterAuthStore.getState().planter, null);
  assert.equal(localStorage.length, 0);
  assert.equal(sessionStorage.length, 0);
  await login({ mode: 'first' });
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, true);
});

test('invalid code and unavailable service do not open username setup', async () => {
  loginFailure = () => Response.json({ detail: [{ loc: ['body', 'access_code'], type: 'credential_error', msg: 'Incorrect organization code.' }] }, { status: 401 });
  await assert.rejects(usePlanterAuthStore.getState().checkOrganizationCode(code), /Incorrect organization code/);
  offline = true;
  await assert.rejects(usePlanterAuthStore.getState().checkOrganizationCode(code), /Check your connection/);
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, false);
});

test('shared code and username are sent with a persistent browser ID and no PIN', async () => {
  await login({ mode: 'first' });
  const body = requests[0].body;
  assert.equal(body.access_code, 'K7MP4R9X'); assert.equal(body.username, 'Ana'); assert.equal(body.mode, 'first');
  assert.match(body.browser_key, /^[a-f0-9]{48}$/);
  assert.equal(body.password, undefined); assert.equal(body.pin, undefined);
  for (const storage of [localStorage, sessionStorage]) for (let n = 0; n < storage.length; n += 1) {
    assert.ok(!storage.getItem(storage.key(n)).includes(code));
    assert.ok(!storage.getItem(storage.key(n)).includes('K7MP4R9X'));
  }
});

test('logout preserves browser identity and username for another normal tab or fresh app load', async () => {
  await login({ mode: 'first' });
  const browser = requests[0].body.browser_key;
  await usePlanterAuthStore.getState().logout();
  assert.equal(localStorage.getItem('mv_planter_user'), null);
  assert.equal(localStorage.getItem('mv_last_planter_username'), 'Ana');
  await login();
  assert.equal(requests.at(-1).body.browser_key, browser);
  assert.equal(requests.at(-1).body.mode, 'return');
  assert.equal(usePlanterAuthStore.getState().planter.participant_slot, 1);
});

test('another browser sends the same username with its own identity to resume progress', async () => {
  await login(); const first = requests[0].body.browser_key;
  const other = new JSDOM('', { url: 'https://field.example.test' });
  try {
    globalThis.document = other.window.document; globalThis.localStorage = other.window.localStorage;
    globalThis.sessionStorage = other.window.sessionStorage;
    await login();
    assert.notEqual(requests.at(-1).body.browser_key, first);
    assert.equal(requests.at(-1).body.username, 'Ana');
  } finally {
    globalThis.document = dom.window.document; globalThis.localStorage = dom.window.localStorage;
    globalThis.sessionStorage = dom.window.sessionStorage; other.window.close();
  }
});

test('active login conflict stays unsigned until an explicit switch confirmation', async () => {
  loginFailure = () => Response.json({ detail: { kind: 'session_in_use', message: 'Switch access?', switch_token: 'fictional-ticket' } }, { status: 409 });
  await assert.rejects(login(), (error) => error.kind === 'session_in_use' && error.switchToken === 'fictional-ticket');
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, false);
  assert.equal(usePlanterAuthStore.getState().error, '');
  loginFailure = null;
  await login({ switchToken: 'fictional-ticket' });
  assert.equal(requests.at(-1).body.switch_token, 'fictional-ticket');
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, true);
  assert.ok(!JSON.stringify(localStorage).includes('fictional-ticket'));
});

test('existing browser signup conflict tells the form the username to resume', async () => {
  loginFailure = () => Response.json({ detail: { kind: 'already_joined', message: 'Return instead.', participant_username: 'Ana' } }, { status: 409 });
  await assert.rejects(login({ mode: 'first' }), (error) => error.kind === 'already_joined' && error.participantUsername === 'Ana');
});

test('invalid username errors retain the field information', async () => {
  loginFailure = () => Response.json({ detail: [{ loc: ['body', 'username'], type: 'credential_error', msg: 'Username not found.' }] }, { status: 401 });
  await assert.rejects(login(), (error) => error.message === 'Username not found.' && error.detail[0].loc.includes('username'));
});

test('session expiry or switching clears the workspace without deleting browser identity', async () => {
  await login(); const identity = requests[0].body.browser_key;
  unauthorized = true; await usePlanterAuthStore.getState().hydrateSession();
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, false);
  assert.match(usePlanterAuthStore.getState().error, /ended or moved/);
  unauthorized = false; await login();
  assert.equal(requests.at(-1).body.browser_key, identity);
  unauthorized = true;
  await assert.rejects(usePlanterAuthStore.getState().markPointStatus(11, 'completed'), /Login ended/);
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, false);
});

test('cookie backup keeps the same identity when local storage is unavailable', async () => {
  await login(); const identity = requests[0].body.browser_key;
  globalThis.localStorage = { getItem() { throw new Error('Blocked'); }, setItem() { throw new Error('Blocked'); }, removeItem() {} };
  try { await login(); assert.equal(requests.at(-1).body.browser_key, identity); }
  finally { globalThis.localStorage = dom.window.localStorage; }
});

test('when all browser storage is blocked, first signup makes no server request', async () => {
  const storage = globalThis.localStorage;
  const cookie = Object.getOwnPropertyDescriptor(document, 'cookie');
  globalThis.localStorage = { getItem() { return null; }, setItem() { throw new Error('Blocked'); }, removeItem() {} };
  Object.defineProperty(document, 'cookie', { configurable: true, get() { return ''; }, set() {} });
  try {
    await assert.rejects(login({ mode: 'first' }), /Allow cookies or browser storage/);
    assert.equal(requests.length, 0);
  } finally {
    globalThis.localStorage = storage;
    if (cookie) Object.defineProperty(document, 'cookie', cookie); else delete document.cookie;
  }
});

test('connection errors and throttling remain clear and retryable', async () => {
  offline = true; await assert.rejects(login(), /Check your connection/); offline = false;
  loginFailure = () => Response.json({ detail: [] }, { status: 429, headers: { 'Retry-After': '125' } });
  await assert.rejects(login(), /Try again in 3 minutes/);
  loginFailure = () => new Response('Unavailable', { status: 502 });
  await assert.rejects(login(), /sign-in service is temporarily unavailable/);
  loginFailure = null; await login();
  const workspace = await usePlanterAuthStore.getState().fetchFieldPoints();
  assert.deepEqual(workspace.points, points);
  assert.deepEqual(workspace.siteAccess, [{ label: 'LIKE entrance', entrance: [10.77999592, 122.625672] }]);
});
