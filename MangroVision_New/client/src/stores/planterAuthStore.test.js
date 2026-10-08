import test, { before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { cwd } from 'node:process';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
import { participantRecoveryCode } from '../utils/participantDevice.js';

let server, dom, usePlanterAuthStore, requests, unauthorized, rejectResume;
const savedGlobals = new Map();

before(async () => {
  dom = new JSDOM('', { url: 'https://field.example.test' });
  const globals = {
    document: dom.window.document,
    localStorage: dom.window.localStorage,
    sessionStorage: dom.window.sessionStorage,
    fetch: async (input, init = {}) => {
      const path = new URL(input, 'https://field.example.test').pathname;
      const body = init.body ? JSON.parse(init.body) : null;
      requests.push({ path, body });
      if (body?.resume_device && rejectResume) return Response.json({ detail: 'This recovery code is not linked to this organization.' }, { status: 409 });
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
  for (const cookie of document.cookie.split(';')) document.cookie = `${cookie.split('=')[0].trim()}=; Max-Age=0; Path=/`;
  sessionStorage.clear();
  requests = [];
  unauthorized = false;
  rejectResume = false;
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

test('repeated sign-ins on one Vercel origin reuse one identity across shared links', async () => {
  const hostedBrowser = new JSDOM('', { url: 'https://mangrovision-test.vercel.app/field' });
  try {
    globalThis.document = hostedBrowser.window.document;
    globalThis.localStorage = hostedBrowser.window.localStorage;
    globalThis.sessionStorage = hostedBrowser.window.sessionStorage;
    let originalIdentity;
    for (let visit = 0; visit < 10; visit += 1) {
      hostedBrowser.reconfigure({ url: 'https://mangrovision-test.vercel.app/field?share=visit-' + visit });
      await usePlanterAuthStore.getState().login('shared', 'test');
      const body = requests.at(-1).body;
      originalIdentity ??= body.device_key;
      assert.equal(body.device_key, originalIdentity);
      assert.equal(body.participant_slot, null);
      assert.equal(body.recover_slot, false);
      assert.equal(body.resume_device, false);
      assert.equal(usePlanterAuthStore.getState().planter.participant_slot, 1);
      await usePlanterAuthStore.getState().logout();
      assert.equal(localStorage.getItem('mv_participant_device:shared'), originalIdentity);
      assert.equal(localStorage.getItem('mv_planter_user'), null);
    }
    assert.equal(requests.filter((request) => request.path.endsWith('/login')).length, 10);
  } finally {
    globalThis.document = dom.window.document;
    globalThis.localStorage = dom.window.localStorage;
    globalThis.sessionStorage = dom.window.sessionStorage;
    hostedBrowser.window.close();
  }
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
    globalThis.document = anotherBrowser.window.document;
    globalThis.localStorage = anotherBrowser.window.localStorage;
    globalThis.sessionStorage = anotherBrowser.window.sessionStorage;
    await usePlanterAuthStore.getState().login('shared', 'test');
    assert.notEqual(requests.at(-1).body.device_key, firstIdentity);
    assert.equal(dom.window.localStorage.getItem('mv_participant_device'), firstIdentity);
  } finally {
    globalThis.document = dom.window.document;
    globalThis.localStorage = dom.window.localStorage;
    globalThis.sessionStorage = dom.window.sessionStorage;
    anotherBrowser.window.close();
  }
});

test('the cookie backup restores the same participant after local storage is lost', async () => {
  await usePlanterAuthStore.getState().login('shared', 'test');
  const identity = requests.at(-1).body.device_key;
  await usePlanterAuthStore.getState().logout();
  localStorage.clear();
  await usePlanterAuthStore.getState().login('shared', 'test');
  assert.equal(requests.at(-1).body.device_key, identity);
  assert.equal(localStorage.getItem('mv_participant_device:shared'), identity);
});

test('existing browser identities are retained and backed up instead of replaced', async () => {
  const identity = 'ab'.repeat(24);
  localStorage.setItem('mv_participant_device', identity);
  await usePlanterAuthStore.getState().login('shared', 'test');
  assert.equal(requests.at(-1).body.device_key, identity);
  assert.ok(document.cookie.includes(identity));
});

test('a recovery code carries the original participant to a different field domain', async () => {
  await usePlanterAuthStore.getState().login('shared', 'test');
  const identity = requests.at(-1).body.device_key;
  const code = participantRecoveryCode('shared');
  await usePlanterAuthStore.getState().logout();
  const changedLink = new JSDOM('', { url: 'https://new-field-link.example.test' });
  try {
    globalThis.document = changedLink.window.document;
    globalThis.localStorage = changedLink.window.localStorage;
    globalThis.sessionStorage = changedLink.window.sessionStorage;
    await usePlanterAuthStore.getState().login('shared', 'test', null, false, code);
    assert.equal(requests.at(-1).body.device_key, identity);
    assert.equal(requests.at(-1).body.resume_device, true);
    await usePlanterAuthStore.getState().logout();
    await usePlanterAuthStore.getState().login('shared', 'test');
    assert.equal(requests.at(-1).body.device_key, identity);
    assert.equal(requests.at(-1).body.resume_device, false);
  } finally {
    globalThis.document = dom.window.document;
    globalThis.localStorage = dom.window.localStorage;
    globalThis.sessionStorage = dom.window.sessionStorage;
    changedLink.window.close();
  }
});

test('a rejected recovery code does not replace the saved browser identity', async () => {
  await usePlanterAuthStore.getState().login('shared', 'test');
  const identity = requests.at(-1).body.device_key;
  await usePlanterAuthStore.getState().logout();
  rejectResume = true;
  await assert.rejects(usePlanterAuthStore.getState().login('shared', 'test', null, false, `MV1-${'ab'.repeat(24)}`), /not linked/);
  assert.equal(participantRecoveryCode('shared'), `MV1-${identity}`);
  rejectResume = false;
  await usePlanterAuthStore.getState().login('shared', 'test');
  assert.equal(requests.at(-1).body.device_key, identity);
});

test('an incomplete recovery code is rejected before a device can consume a slot', async () => {
  await assert.rejects(usePlanterAuthStore.getState().login('shared', 'test', null, false, 'MV1-short'), /complete device recovery code/);
  assert.equal(requests.length, 0);
  assert.equal(usePlanterAuthStore.getState().status, 'error');
});

test('cookie persistence works when local storage is unavailable', async () => {
  globalThis.localStorage = {
    getItem() { throw new Error('Storage blocked'); },
    setItem() { throw new Error('Storage blocked'); },
    removeItem() { throw new Error('Storage blocked'); },
  };
  try {
    await usePlanterAuthStore.getState().login('shared', 'test');
    const identity = requests.at(-1).body.device_key;
    await usePlanterAuthStore.getState().logout();
    await usePlanterAuthStore.getState().login('shared', 'test');
    assert.equal(requests.at(-1).body.device_key, identity);
    assert.equal(usePlanterAuthStore.getState().isAuthenticated, true);
  } finally { globalThis.localStorage = dom.window.localStorage; }
});

test('sign-in stops before taking a slot when neither storage method persists', async () => {
  globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
  globalThis.document = { cookie: '', location: { protocol: 'https:' } };
  Object.defineProperty(globalThis.document, 'cookie', { get: () => '', set: () => {} });
  try {
    await assert.rejects(usePlanterAuthStore.getState().login('shared', 'test'), /Allow cookies or browser storage/);
    assert.equal(requests.length, 0);
  } finally {
    globalThis.document = dom.window.document;
    globalThis.localStorage = dom.window.localStorage;
  }
});

test('restoring one organization preserves other organization identities', async () => {
  await usePlanterAuthStore.getState().login('first_org', 'test');
  const original = requests.at(-1).body.device_key;
  await usePlanterAuthStore.getState().logout();
  await usePlanterAuthStore.getState().login('second_org', 'test', null, false, `MV1-${'cd'.repeat(24)}`);
  await usePlanterAuthStore.getState().logout();
  await usePlanterAuthStore.getState().login('first_org', 'test');
  assert.equal(requests.at(-1).body.device_key, original);
});
