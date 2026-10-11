import test, { before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { cwd } from 'node:process';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
import { participantRecoveryCode } from '../utils/participantDevice.js';

let server, dom, usePlanterAuthStore, requests, unauthorized, rejectResume, returnedIdentity, returnedUsername, rejectLogin;
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
      if (path.endsWith('/login') && rejectLogin) return Response.json({ detail: 'Incorrect planter code' }, { status: 401 });
      if (body?.resume_device && rejectResume) return Response.json({ detail: 'This recovery code is not linked to this organization.' }, { status: 409 });
      if (path.endsWith('/logout')) return Response.json({ status: 'ok' });
      if (path.endsWith('/session') && unauthorized) return Response.json({}, { status: 401 });
      return Response.json({ planter: { id: 7, username: returnedUsername || 'shared', participant_count: 10, participant_slot: 1 }, ...(returnedIdentity ? { device_key: returnedIdentity } : {}) });
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
  returnedIdentity = null;
  returnedUsername = null;
  rejectLogin = false;
  usePlanterAuthStore.setState({ token: null, planter: null, isAuthenticated: false, status: 'idle', error: '' });
});

test('first sign-in records a device identity that survives logout and a fresh app load', async () => {
  await usePlanterAuthStore.getState().login('K7MP-4R9X');
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
  await fresh.getState().login('K7MP-4R9X');
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
      await usePlanterAuthStore.getState().login('K7MP-4R9X');
      const body = requests.at(-1).body;
      originalIdentity ??= body.device_key;
      assert.equal(body.device_key, originalIdentity);
      assert.equal(body.access_code, 'K7MP-4R9X');
      assert.equal(body.username, undefined);
      assert.equal(body.password, undefined);
      assert.equal(body.resume_device, false);
      assert.equal(usePlanterAuthStore.getState().planter.participant_slot, 1);
      await usePlanterAuthStore.getState().logout();
      assert.equal(localStorage.getItem('mv_participant_device:participant-7-1'), originalIdentity);
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
  await usePlanterAuthStore.getState().login('K7MP-4R9X');
  const identity = requests.at(-1).body.device_key;
  unauthorized = true;
  await usePlanterAuthStore.getState().hydrateSession();
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, false);
  assert.equal(localStorage.getItem('mv_participant_device'), identity);
  await usePlanterAuthStore.getState().login('K7MP-4R9X');
  assert.equal(requests.at(-1).body.device_key, identity);
});

test('a separate browser gets a distinct identity instead of claiming the first device share', async () => {
  await usePlanterAuthStore.getState().login('K7MP-4R9X');
  const firstIdentity = requests.at(-1).body.device_key;
  await usePlanterAuthStore.getState().logout();
  const anotherBrowser = new JSDOM('', { url: 'https://field.example.test' });
  try {
    globalThis.document = anotherBrowser.window.document;
    globalThis.localStorage = anotherBrowser.window.localStorage;
    globalThis.sessionStorage = anotherBrowser.window.sessionStorage;
    await usePlanterAuthStore.getState().login('K7MP-4R9X');
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
  await usePlanterAuthStore.getState().login('K7MP-4R9X');
  const identity = requests.at(-1).body.device_key;
  await usePlanterAuthStore.getState().logout();
  localStorage.clear();
  await usePlanterAuthStore.getState().login('K7MP-4R9X');
  assert.equal(requests.at(-1).body.device_key, identity);
  assert.equal(localStorage.getItem('mv_participant_device:participant-7-1'), identity);
});

test('existing browser identities are retained and backed up instead of replaced', async () => {
  const identity = 'ab'.repeat(24);
  localStorage.setItem('mv_participant_device', identity);
  await usePlanterAuthStore.getState().login('K7MP-4R9X');
  assert.equal(requests.at(-1).body.device_key, identity);
  assert.ok(document.cookie.includes(identity));
});

test('a recovery code carries the original participant to a different field domain', async () => {
  await usePlanterAuthStore.getState().login('K7MP-4R9X');
  const identity = requests.at(-1).body.device_key;
  const code = participantRecoveryCode('participant-7-1');
  await usePlanterAuthStore.getState().logout();
  const changedLink = new JSDOM('', { url: 'https://new-field-link.example.test' });
  try {
    globalThis.document = changedLink.window.document;
    globalThis.localStorage = changedLink.window.localStorage;
    globalThis.sessionStorage = changedLink.window.sessionStorage;
    await usePlanterAuthStore.getState().login('K7MP-4R9X', code);
    assert.equal(requests.at(-1).body.device_key, identity);
    assert.equal(requests.at(-1).body.resume_device, true);
    await usePlanterAuthStore.getState().logout();
    returnedIdentity = identity;
    await usePlanterAuthStore.getState().login('K7MP-4R9X');
    assert.ok(requests.at(-1).body.known_device_keys.includes(identity));
    assert.equal(requests.at(-1).body.resume_device, false);
  } finally {
    globalThis.document = dom.window.document;
    globalThis.localStorage = dom.window.localStorage;
    globalThis.sessionStorage = dom.window.sessionStorage;
    changedLink.window.close();
  }
});

test('a rejected recovery code does not replace the saved browser identity', async () => {
  await usePlanterAuthStore.getState().login('K7MP-4R9X');
  const identity = requests.at(-1).body.device_key;
  await usePlanterAuthStore.getState().logout();
  rejectResume = true;
  await assert.rejects(usePlanterAuthStore.getState().login('K7MP-4R9X', `MV1-${'ab'.repeat(24)}`), /not linked/);
  assert.equal(participantRecoveryCode('participant-7-1'), `MV1-${identity}`);
  rejectResume = false;
  await usePlanterAuthStore.getState().login('K7MP-4R9X');
  assert.equal(requests.at(-1).body.device_key, identity);
});

test('an incomplete recovery code is rejected before a device can consume a slot', async () => {
  await assert.rejects(usePlanterAuthStore.getState().login('K7MP-4R9X', 'MV1-short'), /complete device recovery code/);
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
    await usePlanterAuthStore.getState().login('K7MP-4R9X');
    const identity = requests.at(-1).body.device_key;
    await usePlanterAuthStore.getState().logout();
    await usePlanterAuthStore.getState().login('K7MP-4R9X');
    assert.equal(requests.at(-1).body.device_key, identity);
    assert.equal(usePlanterAuthStore.getState().isAuthenticated, true);
  } finally { globalThis.localStorage = dom.window.localStorage; }
});

test('sign-in stops before taking a slot when neither storage method persists', async () => {
  globalThis.localStorage = { getItem: () => null, setItem: () => {}, removeItem: () => {} };
  globalThis.document = { cookie: '', location: { protocol: 'https:' } };
  Object.defineProperty(globalThis.document, 'cookie', { get: () => '', set: () => {} });
  try {
    await assert.rejects(usePlanterAuthStore.getState().login('K7MP-4R9X'), /Allow cookies or browser storage/);
    assert.equal(requests.length, 0);
  } finally {
    globalThis.document = dom.window.document;
    globalThis.localStorage = dom.window.localStorage;
  }
});

test('restoring one organization preserves other organization identities', async () => {
  await usePlanterAuthStore.getState().login('K7MP-4R9X');
  const original = requests.at(-1).body.device_key;
  await usePlanterAuthStore.getState().logout();
  await usePlanterAuthStore.getState().login('G6FR-3VD8', `MV1-${'cd'.repeat(24)}`);
  await usePlanterAuthStore.getState().logout();
  await usePlanterAuthStore.getState().login('K7MP-4R9X');
  assert.equal(requests.at(-1).body.device_key, original);
});

test('a renamed login keeps its organization device while another group keeps its own identity', async () => {
  const otherGroup = 'ab'.repeat(24);
  const original = 'cd'.repeat(24);
  localStorage.setItem('mv_participant_device', otherGroup);
  localStorage.setItem('mv_participant_device:old-oton', original);
  returnedIdentity = original;
  returnedUsername = 'oton-youth';
  await usePlanterAuthStore.getState().login('K7MP-4R9X');
  assert.ok(requests.at(-1).body.known_device_keys.includes(original));
  assert.equal(localStorage.getItem('mv_participant_device:participant-7-1'), original);
  assert.equal(localStorage.getItem('mv_participant_device'), otherGroup);
  assert.equal(participantRecoveryCode('participant-7-1'), `MV1-${original}`);
  await usePlanterAuthStore.getState().logout();
  await usePlanterAuthStore.getState().login('K7MP-4R9X');
  assert.equal(requests.at(-1).body.device_key, otherGroup);
  assert.ok(requests.at(-1).body.known_device_keys.includes(original));
});

test('a rejected renamed login retains old identities for a corrected retry', async () => {
  const identity = 'cd'.repeat(24);
  localStorage.setItem('mv_participant_device:old-oton', identity);
  rejectLogin = true;
  await assert.rejects(usePlanterAuthStore.getState().login('G6FR-3VD8'), /Incorrect planter code/);
  assert.equal(localStorage.getItem('mv_participant_device:participant-7-1'), null);
  rejectLogin = false;
  returnedIdentity = identity;
  returnedUsername = 'oton-youth';
  await usePlanterAuthStore.getState().login('K7MP-4R9X');
  assert.ok(requests.at(-1).body.known_device_keys.includes(identity));
  assert.equal(localStorage.getItem('mv_participant_device:participant-7-1'), identity);
});

test('renamed identity recovery uses cookies when local storage is unavailable', async () => {
  const identity = 'cd'.repeat(24);
  document.cookie = `mv_participant_device%3Aold-oton=${identity}; Path=/`;
  globalThis.localStorage = {
    getItem() { throw new Error('Storage blocked'); },
    setItem() { throw new Error('Storage blocked'); },
    removeItem() { throw new Error('Storage blocked'); },
  };
  try {
    returnedIdentity = identity;
    returnedUsername = 'oton-youth';
    await usePlanterAuthStore.getState().login('K7MP-4R9X');
    assert.ok(requests.at(-1).body.known_device_keys.includes(identity));
    assert.equal(participantRecoveryCode('participant-7-1'), `MV1-${identity}`);
  } finally { globalThis.localStorage = dom.window.localStorage; }
});

test('a retained session binds its old device to the renamed username before showing a recovery code', async () => {
  const otherGroup = 'ab'.repeat(24);
  const original = 'cd'.repeat(24);
  localStorage.setItem('mv_participant_device', otherGroup);
  localStorage.setItem('mv_participant_device:old-oton', original);
  localStorage.setItem('mv_planter_user', JSON.stringify({ id: 7, username: 'old-oton', participant_slot: 1 }));
  returnedUsername = 'oton-youth';
  await usePlanterAuthStore.getState().hydrateSession();
  assert.equal(usePlanterAuthStore.getState().isAuthenticated, true);
  assert.equal(participantRecoveryCode('participant-7-1'), `MV1-${original}`);
  assert.equal(localStorage.getItem('mv_participant_device'), otherGroup);
  assert.equal(localStorage.getItem('mv_participant_device:participant-7-1'), original);
});

test('access codes are sent only to the sign-in endpoint and are never saved in browser storage', async () => {
  const code = 'K7MP-4R9X';
  await usePlanterAuthStore.getState().login(code);
  assert.equal(requests[0].body.access_code, code);
  const stored = Array.from({ length: localStorage.length }, (_, index) => [localStorage.key(index), localStorage.getItem(localStorage.key(index))]);
  assert.ok(!JSON.stringify(stored).includes(code));
  assert.ok(!document.cookie.includes(code));
});
