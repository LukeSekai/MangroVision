import test, { before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { cwd } from 'node:process';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

let server, dom, usePlanterAuthStore, requests, unauthorized, rejectLogin, offline, loginFailure;
const savedGlobals = new Map();
const firstCode = 'K7MP-4R9X';
const secondCode = 'ABCD-EFGH';
const savedPoints = [
  { assignment_point_id: 11, status: 'completed' },
  { assignment_point_id: 12, status: 'pending' },
];

before(async () => {
  dom = new JSDOM('', { url: 'https://field.example.test' });
  const globals = {
    document: dom.window.document,
    localStorage: dom.window.localStorage,
    sessionStorage: dom.window.sessionStorage,
    fetch: async (input, init = {}) => {
      if (offline) throw new TypeError('Network unavailable');
      const path = new URL(input, 'https://field.example.test').pathname;
      const body = init.body ? JSON.parse(init.body) : null;
      requests.push({ path, body });
      if (path.endsWith('/login') && loginFailure) return loginFailure();
      if (path.endsWith('/login') && rejectLogin) return Response.json({ detail: [
        { loc: ['body', 'access_code'], type: 'credential_error', msg: 'This planter code is incorrect or no longer valid.' },
      ] }, { status: 401 });
      if (path.endsWith('/logout')) return Response.json({ status: 'ok' });
      if (path.endsWith('/session') && unauthorized) return Response.json({}, { status: 401 });
      return Response.json({ planter: { id: 7, participant_count: 10,
        participant_slot: body?.access_code === secondCode.replace('-', '') ? 2 : 1 },
        ...(path.endsWith('/field-points') ? { points: savedPoints, project_sites: { features: [] } } : {}),
      });
    },
  };
  for (const [key, value] of Object.entries(globals)) {
    savedGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  server = await createServer({ configFile: false, appType: 'custom',
    server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    cacheDir: 'node_modules/.vite-planter-code-tests', optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ usePlanterAuthStore } = await server.ssrLoadModule('/src/stores/planterAuthStore.js'));
});

after(async () => {
  await server?.close();
  dom?.window.close();
  for (const [key, descriptor] of savedGlobals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
});

beforeEach(() => {
  localStorage.clear();
  sessionStorage.clear();
  requests = [];
  unauthorized = rejectLogin = offline = false;
  loginFailure = null;
  usePlanterAuthStore.setState({ token: null, planter: null, isAuthenticated: false, status: 'idle', error: '' });
});

test('sign-in sends only the personal code and keeps credentials out of browser storage', async () => {
  await usePlanterAuthStore.getState().login(` ${firstCode} `);
  assert.deepEqual(requests[0].body, { access_code: 'K7MP4R9X' });
  assert.equal(usePlanterAuthStore.getState().planter.participant_slot, 1);
  for (const storage of [localStorage, sessionStorage]) {
    for (let index = 0; index < storage.length; index += 1) {
      const key = storage.key(index);
      assert.ok(!`${key} ${storage.getItem(key)}`.includes(firstCode));
      assert.ok(!key.startsWith('mv_participant_device'));
    }
  }
});

test('sign-in after doing nothing and logging out resumes the same participant on a fresh app load', async () => {
  await usePlanterAuthStore.getState().login(firstCode);
  await usePlanterAuthStore.getState().logout();
  assert.equal(localStorage.getItem('mv_planter_user'), null);
  const module = server.moduleGraph.getModuleById(`${cwd().replaceAll('\\', '/')}/src/stores/planterAuthStore.js`);
  server.moduleGraph.invalidateModule(module);
  const fresh = (await server.ssrLoadModule('/src/stores/planterAuthStore.js')).usePlanterAuthStore;
  assert.equal(fresh.getState().isAuthenticated, false);
  await fresh.getState().login(firstCode);
  assert.deepEqual(requests.at(-1).body, { access_code: 'K7MP4R9X' });
  assert.equal(fresh.getState().planter.participant_slot, 1);
});

test('a fresh browser on a different field domain uses the same code without recovery', async () => {
  await usePlanterAuthStore.getState().login(firstCode);
  await usePlanterAuthStore.getState().logout();
  const anotherBrowser = new JSDOM('', { url: 'https://new-field-link.example.test' });
  try {
    globalThis.document = anotherBrowser.window.document;
    globalThis.localStorage = anotherBrowser.window.localStorage;
    globalThis.sessionStorage = anotherBrowser.window.sessionStorage;
    await usePlanterAuthStore.getState().login(firstCode);
    assert.deepEqual(requests.at(-1).body, { access_code: 'K7MP4R9X' });
    assert.equal(usePlanterAuthStore.getState().planter.participant_slot, 1);
    assert.equal(localStorage.getItem('mv_participant_device'), null);
  } finally {
    globalThis.document = dom.window.document;
    globalThis.localStorage = dom.window.localStorage;
    globalThis.sessionStorage = dom.window.sessionStorage;
    anotherBrowser.window.close();
  }
});

test('changing codes on one browser selects the participant identified by each code', async () => {
  await usePlanterAuthStore.getState().login(firstCode);
  assert.equal(usePlanterAuthStore.getState().planter.participant_slot, 1);
  await usePlanterAuthStore.getState().logout();
  await usePlanterAuthStore.getState().login(secondCode);
  assert.equal(usePlanterAuthStore.getState().planter.participant_slot, 2);
  await usePlanterAuthStore.getState().logout();
  await usePlanterAuthStore.getState().login(firstCode);
  assert.equal(usePlanterAuthStore.getState().planter.participant_slot, 1);
});

test('session expiry and cleared site data do not require a saved device identity', async () => {
  await usePlanterAuthStore.getState().login(firstCode);
  unauthorized = true;
  await usePlanterAuthStore.getState().hydrateSession();
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, false);
  localStorage.clear();
  sessionStorage.clear();
  await usePlanterAuthStore.getState().login(firstCode);
  assert.equal(usePlanterAuthStore.getState().planter.participant_slot, 1);
  assert.deepEqual(requests.at(-1).body, { access_code: 'K7MP4R9X' });
});

test('loading assignments after code sign-in restores saved points and planting progress', async () => {
  await usePlanterAuthStore.getState().login(firstCode);
  const result = await usePlanterAuthStore.getState().fetchFieldPoints();
  assert.deepEqual(result.points, savedPoints);
  assert.deepEqual(result.projectSites, []);
  assert.equal(usePlanterAuthStore.getState().planter.participant_slot, 1);
});

test('blocked browser storage does not prevent code sign-in', async () => {
  const blockedStorage = {
    getItem() { throw new Error('Storage blocked'); },
    setItem() { throw new Error('Storage blocked'); },
    removeItem() { throw new Error('Storage blocked'); },
  };
  globalThis.localStorage = globalThis.sessionStorage = blockedStorage;
  try {
    await usePlanterAuthStore.getState().login(firstCode);
    assert.equal(usePlanterAuthStore.getState().isAuthenticated, true);
    assert.deepEqual(requests.at(-1).body, { access_code: 'K7MP4R9X' });
  } finally {
    globalThis.localStorage = dom.window.localStorage;
    globalThis.sessionStorage = dom.window.sessionStorage;
  }
});

test('incorrect or revoked codes show the credential error and allow a corrected retry', async () => {
  rejectLogin = true;
  await assert.rejects(usePlanterAuthStore.getState().login(firstCode), (error) => {
    assert.match(error.message, /incorrect or no longer valid/);
    assert.equal(error.status, 401);
    assert.match(error.detail[0].msg, /incorrect or no longer valid/);
    return true;
  });
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, false);
  assert.equal(usePlanterAuthStore.getState().status, 'error');
  rejectLogin = false;
  await usePlanterAuthStore.getState().login(secondCode);
  assert.equal(usePlanterAuthStore.getState().error, '');
  assert.equal(usePlanterAuthStore.getState().planter.participant_slot, 2);
});

test('a connection failure leaves sign-in retryable with a clear error', async () => {
  offline = true;
  await assert.rejects(usePlanterAuthStore.getState().login(firstCode), /Check your connection/);
  assert.equal(usePlanterAuthStore.getState().status, 'error');
  offline = false;
  await usePlanterAuthStore.getState().login(firstCode);
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, true);
});

test('mobile pasted dashes and spaces authenticate with the same canonical code', async () => {
  for (const code of ['k7mp–4r9x', ' K7MP—4R9X ', 'K7MP\u20114R9X', 'K7MP 4R9X']) {
    await usePlanterAuthStore.getState().login(code);
    assert.deepEqual(requests.at(-1).body, { access_code: 'K7MP4R9X' });
    assert.equal(usePlanterAuthStore.getState().planter.participant_slot, 1);
  }
});

test('an outdated backend explains service failure instead of blaming the code', async () => {
  loginFailure = () => Response.json({ detail: [{ loc: ['body', 'device_key'], type: 'missing', msg: 'Field required' }] }, { status: 422 });
  await assert.rejects(usePlanterAuthStore.getState().login(firstCode), /temporarily unavailable.*contact LGU staff/);
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, false);
});

test('throttling gives the retry time and server failures stay separate from code errors', async () => {
  loginFailure = () => Response.json({ detail: [] }, { status: 429, headers: { 'Retry-After': '125' } });
  await assert.rejects(usePlanterAuthStore.getState().login(firstCode), /Try again in 3 minutes/);
  loginFailure = () => new Response('Unavailable', { status: 502 });
  await assert.rejects(usePlanterAuthStore.getState().login(firstCode), /sign-in service is temporarily unavailable/);
  loginFailure = () => Response.json({ planter: {} });
  await assert.rejects(usePlanterAuthStore.getState().login(firstCode), /Could not verify your sign-in/);
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, false);
});
