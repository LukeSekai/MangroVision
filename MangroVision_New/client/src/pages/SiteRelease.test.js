import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { MemoryRouter } from 'react-router-dom';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

let server, Workspace, LGUSites, store, dom, root, requests, candidates, conflict, refreshed;
const previous = new Map();

before(async () => {
  server = await createServer({ configFile: false, appType: 'custom',
    server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    optimizeDeps: { noDiscovery: true, include: [] },
  });
  Workspace = (await server.ssrLoadModule('/src/pages/MonitoringMapWorkspace.jsx')).default;
  LGUSites = (await server.ssrLoadModule('/src/pages/LGUSites.jsx')).default;
  store = (await server.ssrLoadModule('/src/stores/mapStore.js')).useMapStore;
});
after(async () => { await server?.close(); });

beforeEach(() => {
  dom = new JSDOM('<body><div id="root"></div></body>', { url: 'http://localhost/' });
  for (const [key, value] of Object.entries({ window: dom.window, document: dom.window.document,
    navigator: dom.window.navigator, HTMLElement: dom.window.HTMLElement, localStorage: dom.window.localStorage,
    IS_REACT_ACT_ENVIRONMENT: true })) {
    previous.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  requests = []; conflict = false; refreshed = [];
  candidates = [{ planting_event_id: 10, planting_point_id: 1, version: 0, can_release: true }];
  previous.set('fetch', Object.getOwnPropertyDescriptor(globalThis, 'fetch'));
  globalThis.fetch = async (url, options = {}) => {
    const path = String(url);
    requests.push({ path, body: options.body ? JSON.parse(options.body) : null });
    if (path.endsWith('/lgu-sites')) return { ok: true, json: async () => ({ account: { organization_id: 9, username: 'lgu', access_code: 'TEST-CODE' }, summary: { site_count: 0 }, sites: [] }) };
    if (options.method === 'POST') return { ok: !conflict, json: async () => conflict
      ? { detail: 'Location changed. Reload the map before releasing it.' } : { released_count: 1 } };
    return { ok: true, json: async () => ({ points: candidates }) };
  };
  store.setState({ points: [{ id: 1, planting_status: 'planted', death_at: '2026-10-01', source_project_site_id: 1 }],
    projectSites: { type: 'FeatureCollection', features: [] }, monitoringOrganizationId: null,
    replantingSelectedPointIds: [1], replantingSelectionMode: 'click', loadingPoints: false,
    fetchPoints: async () => {
      refreshed.push('points');
      store.setState({ points: [{ id: 1, planting_status: 'planned', death_at: null, source_project_site_id: null }] });
    },
    fetchStats: async () => { refreshed.push('stats'); }, fetchZones: async () => { refreshed.push('zones'); },
  });
  root = createRoot(document.getElementById('root'));
});
afterEach(async () => {
  await act(async () => root.unmount());
  dom.window.close();
  for (const [key, descriptor] of previous) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
  previous.clear();
});

const settle = () => new Promise((resolve) => setTimeout(resolve, 25));
async function mount(Component = Workspace) {
  await act(async () => { root.render(createElement(MemoryRouter, null, createElement(Component))); await settle(); });
}
async function click(label) {
  const scope = document.querySelector('[role="dialog"]') || document;
  const button = [...scope.querySelectorAll('button')].find((item) => item.textContent.trim() === label);
  assert.ok(button, label);
  await act(async () => { button.click(); await settle(); });
}

test('Monitoring releases selected event versions and refreshes points, counts, and boundaries', async () => {
  await mount();
  assert.doesNotMatch(document.body.textContent, /LGU replanting|Review for replanting/i);
  await click('Release from project site');
  assert.match(document.querySelector('[role="dialog"]').textContent, /Original planting records and survival rates stay with their sites/);
  assert.match(document.querySelector('[role="dialog"]').textContent, /Planned.*planting again/);
  assert.deepEqual([...document.querySelector('.modal-actions').querySelectorAll('button')].map(button => button.textContent), ['Cancel', 'Release 1 location']);
  assert.equal(document.querySelector('.modal-body button'), null);
  await click('Release 1 location');
  const writes = requests.filter((request) => request.body);
  assert.deepEqual(writes, [{ path: '/api/monitoring/release-locations', body: {
    planting_event_ids: [10], versions: { 10: 0 },
  } }]);
  assert.equal(document.querySelector('[role="dialog"]'), null);
  assert.match(document.body.textContent, /released from the project site and now Planned/);
  assert.deepEqual(refreshed.sort(), ['points', 'stats', 'zones']);
  assert.deepEqual(store.getState().replantingSelectedPointIds, []);
  assert.equal(store.getState().points[0].death_at, null);
});

test('a stale release displays the conflict and keeps the current points and selection', async () => {
  conflict = true;
  await mount();
  await click('Release from project site');
  await click('Release 1 location');
  assert.match(document.querySelector('[role="alert"]').textContent, /Reload the map/);
  assert.deepEqual(refreshed, []);
  assert.deepEqual(store.getState().replantingSelectedPointIds, [1]);
  assert.ok(store.getState().points[0].death_at);
});

test('ineligible deaths cannot submit a release', async () => {
  candidates[0].can_release = false;
  await mount();
  await click('Release from project site');
  assert.match(document.querySelector('[role="alert"]').textContent, /no longer available for release/);
  const confirm = [...document.querySelector('[role="dialog"]').querySelectorAll('button')]
    .find((button) => button.textContent === 'Release 1 location');
  assert.equal(confirm.disabled, true);
  await act(async () => document.querySelector('[role="dialog"]').dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Enter', bubbles: true })));
  assert.equal(requests.filter((request) => request.body).length, 0);
  await click('Cancel');
  assert.equal(document.querySelector('[role="dialog"]'), null);
});

test('an eroded release candidate shows the short replanting modal and cannot submit', async () => {
  candidates[0].can_release = false;
  candidates[0].eroded_unavailable = true;
  await act(async () => root.render(createElement(MemoryRouter, null, createElement(Workspace))));
  await click('Release from project site');
  assert.equal(document.querySelector('.modal-title').textContent, 'Replanting blocked');
  assert.equal(document.querySelector('.modal-body p').textContent,
    'Replanting is unavailable because this location is inside an eroded zone.');
  assert.deepEqual([...document.querySelectorAll('.modal-actions button')].map((button) => button.textContent), ['OK']);
  assert.equal(requests.some((request) => request.body), false);
  await click('OK');
  assert.equal(document.querySelector('.modal-title'), null);
});

test('LGU Sites keeps Monitoring navigation and provides permanent credentials without replacement controls', async () => {
  await mount(LGUSites);
  await act(settle);
  assert.equal(document.querySelector('h1').textContent, 'LGU Sites');
  assert.ok(document.querySelector('a[href="/monitoring/lgu-sites"]'));
  assert.ok(document.querySelector('a[href="/monitoring/map"]'));
  assert.doesNotMatch(document.body.textContent, /replant|replacement|Approve/i);
  assert.equal(requests[0].path, '/api/monitoring/lgu-sites');
  assert.match(document.body.textContent, /LGU planter account/);
  assert.doesNotMatch(document.body.textContent, /TEST-CODE/);
  await click('Show code');
  assert.match(document.body.textContent, /TEST-CODE/);
});
