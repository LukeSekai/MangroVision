import test, { before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { cwd } from 'node:process';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

let server, dom, usePlanterAuthStore, requests, unauthorized;
const savedGlobals = new Map();

before(async () => {
  dom = new JSDOM('', { url: 'https://field.example.test' });
  const globals = {
    localStorage: dom.window.localStorage,
    sessionStorage: dom.window.sessionStorage,
    fetch: async (input, init = {}) => {
      const path = new URL(input, 'https://field.example.test').pathname;
      const body = init.body ? JSON.parse(init.body) : null;
      requests.push({ path, body });
      if (path.endsWith('/logout')) return Response.json({ status: 'ok' });
      if (path.endsWith('/session') && unauthorized) return Response.json({}, { status: 401 });
      return Response.json({ planter: { id: 7, username: 'shared', participant_count: 10, participant_slot: 1 } });
    },
  };
  for (const [key, value] of Object.entries(globals)) {
    savedGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  server = await createServer({ configFile: false, appType: 'custom',
    server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    cacheDir: 'node_modules/.vite-planter-device-tests', optimizeDeps: { noDiscovery: true, include: [] },
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
  unauthorized = false;
  usePlanterAuthStore.setState({ token: null, planter: null, isAuthenticated: false, status: 'idle', error: '' });
});

test('registration records a device identity that survives logout and a fresh app load', async () => {
  await usePlanterAuthStore.getState().register({ username: 'shared', password: 'test', organization_id: 1, participant_count: 10 });
  const identity = requests[0].body.device_key;
  assert.ok(identity.length >= 16);
  assert.equal(localStorage.getItem('mv_participant_device'), identity);
  await usePlanterAuthStore.getState().logout();
  assert.equal(localStorage.getItem('mv_planter_user'), null);
  assert.equal(localStorage.getItem('mv_participant_device'), identity);
  const module = server.moduleGraph.getModuleById(`${cwd().replaceAll('\\', '/')}/src/stores/planterAuthStore.js`);
  server.moduleGraph.invalidateModule(module);
  const fresh = (await server.ssrLoadModule('/src/stores/planterAuthStore.js')).usePlanterAuthStore;
  assert.equal(fresh.getState().isAuthenticated, false);
  await fresh.getState().login('shared', 'test');
  assert.equal(requests.at(-1).body.device_key, identity);
  assert.equal(fresh.getState().planter.participant_slot, 1);
});

test('session expiry preserves the device identity for the next login', async () => {
  await usePlanterAuthStore.getState().login('shared', 'test');
  const identity = requests.at(-1).body.device_key;
  unauthorized = true;
  await usePlanterAuthStore.getState().hydrateSession();
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, false);
  assert.equal(localStorage.getItem('mv_participant_device'), identity);
  await usePlanterAuthStore.getState().login('shared', 'test');
  assert.equal(requests.at(-1).body.device_key, identity);
});

test('a separate browser gets a distinct identity instead of claiming the first device share', async () => {
  await usePlanterAuthStore.getState().login('shared', 'test');
  const firstIdentity = requests.at(-1).body.device_key;
  await usePlanterAuthStore.getState().logout();
  const anotherBrowser = new JSDOM('', { url: 'https://field.example.test' });
  try {
    globalThis.localStorage = anotherBrowser.window.localStorage;
    globalThis.sessionStorage = anotherBrowser.window.sessionStorage;
    await usePlanterAuthStore.getState().login('shared', 'test');
    assert.notEqual(requests.at(-1).body.device_key, firstIdentity);
    assert.equal(dom.window.localStorage.getItem('mv_participant_device'), firstIdentity);
  } finally {
    globalThis.localStorage = dom.window.localStorage;
    globalThis.sessionStorage = dom.window.sessionStorage;
    anotherBrowser.window.close();
  }
});
