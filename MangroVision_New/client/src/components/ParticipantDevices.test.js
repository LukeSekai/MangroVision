import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
import { participantDeviceKey, participantRecoveryCode } from '../utils/participantDevice.js';

let server, dom, root, createRoot, ParticipantDevices, ParticipantRecovery, requests, released, copied;
const savedGlobals = new Map();
const planter = { id: 24, full_name: 'Test organization', username: 'test_org', participant_slot: 1 };

before(async () => {
  dom = new JSDOM('<div id="root"></div>', { url: 'https://devices.example.test', pretendToBeVisual: true });
  const globals = {
    window: dom.window, document: dom.window.document, localStorage: dom.window.localStorage,
    IS_REACT_ACT_ENVIRONMENT: true,
    navigator: { userAgent: dom.window.navigator.userAgent, clipboard: { writeText: async (value) => { copied = value; } } },
    requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
    cancelAnimationFrame: dom.window.cancelAnimationFrame.bind(dom.window),
    fetch: async (input, init = {}) => {
      requests.push({ path: input, method: init.method || 'GET' });
      if (init.method === 'POST') { released = true; return Response.json({ status: 'reset', slot: 1 }); }
      return Response.json({ participant_count: 3, registered_devices: released ? 1 : 2, available_devices: released ? 2 : 1,
        devices: [
          { slot: 1, registered: !released, assigned_points: 10, completed_points: 2, last_seen_at: '2026-10-08T01:00:00+00:00' },
          { slot: 2, registered: true, assigned_points: 10, completed_points: 0, last_seen_at: null },
          { slot: 3, registered: false, assigned_points: 10, completed_points: 0, last_seen_at: null },
        ],
      });
    },
  };
  for (const [key, value] of Object.entries(globals)) {
    savedGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  ({ createRoot } = await import('react-dom/client'));
  server = await createServer({ configFile: false, appType: 'custom',
    server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    cacheDir: 'node_modules/.vite-participant-devices-tests', optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ default: ParticipantDevices } = await server.ssrLoadModule('/src/components/ParticipantDevices.jsx'));
  ({ default: ParticipantRecovery } = await server.ssrLoadModule('/src/components/ParticipantRecovery.jsx'));
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
  requests = []; released = false; copied = '';
  localStorage.clear();
  for (const cookie of document.cookie.split(';')) document.cookie = `${cookie.split('=')[0].trim()}=; Max-Age=0; Path=/`;
  root = createRoot(document.getElementById('root'));
});
afterEach(async () => { await act(async () => root.unmount()); });

function button(label) {
  return [...document.querySelectorAll('button')].find((element) => element.textContent === label);
}

test('staff see occupied slots and reset only after confirming the chosen participant', async () => {
  await act(async () => root.render(createElement(ParticipantDevices, { planter })));
  assert.match(document.body.textContent, /2 of 3 device slots used/);
  assert.equal(document.querySelectorAll('.participant-device-list li').length, 3);
  assert.match(document.body.textContent, /10 points · 2 planted/);
  await act(async () => button('Reset device slot').click());
  assert.ok(document.querySelector('[role="dialog"]'));
  assert.equal(requests.filter((request) => request.method === 'POST').length, 0);
  await act(async () => button('Cancel').click());
  assert.equal(document.querySelector('[role="dialog"]'), null);
  assert.equal(requests.filter((request) => request.method === 'POST').length, 0);
  await act(async () => button('Reset device slot').click());
  await act(async () => document.querySelector('[role="dialog"] .btn-primary').click());
  assert.deepEqual(requests.filter((request) => request.method === 'POST'), [
    { path: '/api/planters/24/participants/1/reset-device', method: 'POST' },
  ]);
  assert.match(document.body.textContent, /1 of 3 device slots used/);
  assert.match(document.body.textContent, /points and planting history are preserved/);
});

test('the recovery dialog copies the existing participant identity', async () => {
  const identity = participantDeviceKey(planter.username);
  const code = participantRecoveryCode(planter.username);
  await act(async () => root.render(createElement(ParticipantRecovery, { planter, onClose: () => {} })));
  assert.equal(document.querySelector('input').value, code);
  assert.equal(document.querySelector('input').readOnly, true);
  await act(async () => button('Copy recovery code').click());
  assert.equal(copied, `MV1-${identity}`);
  assert.match(document.body.textContent, /organization's username and password/);
});

test('a missing browser identity never produces a misleading new recovery code', async () => {
  await act(async () => root.render(createElement(ParticipantRecovery, { planter, onClose: () => {} })));
  assert.equal(document.querySelector('input'), null);
  assert.match(document.querySelector('[role="alert"]').textContent, /Ask the LGU/);
  assert.equal(localStorage.length, 0);
  assert.equal(document.cookie, '');
});
