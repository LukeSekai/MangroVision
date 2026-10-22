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
  candidates = [{ planting_event_id: 10, planting_point_id: 1, project_site_id: 1, version: 0, can_release: true }];
  previous.set('fetch', Object.getOwnPropertyDescriptor(globalThis, 'fetch'));
  globalThis.fetch = async (url, options = {}) => {
    const path = String(url);
    requests.push({ path, body: options.body ? JSON.parse(options.body) : null });
    if (path.endsWith('/lgu-sites')) return { ok: true, json: async () => ({ account: { organization_id: 9, username: 'lgu', access_code: 'TEST-CODE' }, summary: { site_count: 0 }, sites: [] }) };
    if (options.method === 'POST') return { ok: !conflict, json: async () => conflict
      ? { detail: 'Location changed. Reload the map before releasing it.' } : { released_count: 1 } };
    return { ok: true, json: async () => ({ points: candidates }) };
  };
  store.setState({ points: [{ id: 1, planting_status: 'planted', death_at: '2026-10-01', source_project_site_id: 1 },
    { id: 2, planting_status: 'planted', death_at: '2026-10-02', source_project_site_id: 2 }],
    projectSites: { type: 'FeatureCollection', features: [1, 2].map((id) => ({ id,
      properties: { id, name: id === 1 ? 'First site' : 'Second site', organization_id: 9, organization_name: 'Same organization',
        mortality_warning: { active: true, mortality_rate_pct: 50, total_planted: 10, dead_count: 5 } } })) },
    monitoringOrganizationId: null, monitoringProjectSiteId: null, showDeadPointsOnly: false,
    replantingSelectedPointIds: [], replantingSelectionMode: 'click', loadingPoints: false,
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
async function mount(Component = Workspace, { entry = '/monitoring/map?project_site_id=1&focus=site_points', select = true } = {}) {
  await act(async () => { root.render(createElement(MemoryRouter, { initialEntries: [entry] }, createElement(Component))); await settle(); });
  if (Component === Workspace && select) await act(async () => store.getState().toggleReplantingPoint(1));
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
    project_site_id: 1, planting_event_ids: [10], versions: { 10: 0 },
  } }]);
  assert.equal(requests[0].path, '/api/monitoring/release-candidates?project_site_id=1');
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

test('mortality panel lists only clickable site names and requires a site before release selection', async () => {
  await mount(Workspace, { entry: '/monitoring/map', select: false });
  const list = document.querySelector('.monitoring-map-site-list');
  assert.deepEqual([...list.querySelectorAll('button')].map((button) => button.textContent), ['First site', 'Second site']);
  assert.doesNotMatch(list.textContent, /High mortality|recorded deaths|50%/);
  assert.ok([...document.querySelectorAll('.monitoring-map-tools button')].every((button) => button.disabled));
  await act(async () => {
    store.getState().toggleReplantingPoint(1);
    store.getState().paintReplantingPoints([1, 2], 'select');
  });
  assert.deepEqual(store.getState().replantingSelectedPointIds, []);
  await click('Second site');
  assert.equal(document.querySelector('#monitoring-map-project-site').value, '2');
  assert.equal(store.getState().monitoringProjectSiteId, 2);
  assert.ok([...document.querySelectorAll('.monitoring-map-tools button')].every((button) => !button.disabled));
});

test('switching sites clears selection and only allows the new site even within the same organization', async () => {
  await mount();
  await click('Brush select');
  await click('Second site');
  assert.equal(store.getState().replantingSelectionMode, 'click');
  assert.deepEqual(store.getState().replantingSelectedPointIds, []);
  await act(async () => {
    store.getState().toggleReplantingPoint(1);
    store.getState().paintReplantingPoints([1, 2], 'select');
  });
  assert.deepEqual(store.getState().replantingSelectedPointIds, [2]);
  candidates = [{ planting_event_id: 20, planting_point_id: 2, project_site_id: 2, version: 3, can_release: true }];
  await click('Release from project site');
  assert.match(document.querySelector('[role="dialog"]').textContent, /Second site/);
  await click('Release 1 location');
  assert.deepEqual(requests.find((request) => request.body).body,
    { project_site_id: 2, planting_event_ids: [20], versions: { 20: 3 } });
});

test('a candidate from another site cannot be released even if its point ID matches', async () => {
  candidates[0].project_site_id = 2;
  await mount();
  await click('Release from project site');
  assert.match(document.querySelector('[role="alert"]').textContent, /another project site/);
  const confirm = [...document.querySelectorAll('.modal-actions button')].find((button) => button.textContent === 'Release 1 location');
  assert.equal(confirm.disabled, true);
  await act(async () => confirm.click());
  assert.equal(requests.some((request) => request.body), false);
});

test('an unknown site in the URL leaves release tools unavailable', async () => {
  await mount(Workspace, { entry: '/monitoring/map?project_site_id=999', select: false });
  assert.equal(store.getState().monitoringProjectSiteId, null);
  assert.ok([...document.querySelectorAll('.monitoring-map-tools button')].every((button) => button.disabled));
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
  await mount();
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
