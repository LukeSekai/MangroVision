import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement, useState } from 'react';
import { MemoryRouter, Route, useNavigate } from 'react-router-dom';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

let server, dom, root, createRoot, RetainedRoutes, Panel, PanelCard, useMapStore, useProcessingStore;
const pages = new Map();
const originalGlobals = new Map();
let requests, responses;

before(async () => {
  dom = new JSDOM('<div id="root"></div>', { url: 'http://workspace.example', pretendToBeVisual: true });
  const globals = {
    window: dom.window, document: dom.window.document, localStorage: dom.window.localStorage,
    Event: dom.window.Event, IS_REACT_ACT_ENVIRONMENT: true,
    requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
    cancelAnimationFrame: dom.window.cancelAnimationFrame.bind(dom.window),
    fetch: async (input) => {
      const path = new URL(input, window.location.origin).pathname;
      requests.push(path);
      if (responses.has(path)) return Response.json(responses.get(path));
      if (['/api/planters/', '/api/assignments/'].includes(path)) return Response.json([]);
      if (path === '/api/planters/dashboard') return Response.json({ active_planters: 0 });
      if (path === '/api/planter-auth/organizations') return Response.json({ organizations: [] });
      if (path === '/api/share/field-link') return Response.json({ active: false });
      throw new Error(`Unexpected panel test request: ${path}`);
    },
  };
  for (const [key, value] of Object.entries(globals)) {
    originalGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  // Give the collapse animation a real measurement despite JSDOM having no layout.
  Object.defineProperty(window.HTMLElement.prototype, 'scrollHeight', { configurable: true, get: () => 120 });
  ({ createRoot } = await import('react-dom/client'));
  server = await createServer({ configFile: false, appType: 'custom',
    server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    cacheDir: 'node_modules/.vite-panel-tests', optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ Panel, PanelCard } = await server.ssrLoadModule('/src/components/Panel.jsx'));
  ({ default: RetainedRoutes } = await server.ssrLoadModule('/src/components/RetainedRoutes.jsx'));
  for (const page of ['MapAnalytics', 'ImageProcessing', 'PlanterManagement', 'ErodedZoneEditor', 'MonitoringMapWorkspace']) {
    const module = await server.ssrLoadModule(`/src/pages/${page}.jsx`);
    pages.set(page, module.default);
  }
  ({ useMapStore } = await server.ssrLoadModule('/src/stores/mapStore.js'));
  ({ useProcessingStore } = await server.ssrLoadModule('/src/stores/processingStore.js'));
  const { useAuthStore } = await server.ssrLoadModule('/src/stores/authStore.js');
  useAuthStore.setState({ token: 'cookie', hydrated: true, isAuthenticated: true });
  useMapStore.setState({ fetchStats: async () => {}, fetchPoints: async () => {}, fetchZones: async () => {} });
});

after(async () => {
  await server?.close();
  dom?.window.close();
  for (const [key, descriptor] of originalGlobals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
});

beforeEach(() => {
  requests = [];
  responses = new Map();
  useMapStore.getState().resetWorkspaceData();
  useMapStore.setState({ stats: { analyses: [], points: [], total_analyses: 0 } });
  useProcessingStore.getState().reset();
  root = createRoot(document.getElementById('root'));
});
afterEach(async () => { await act(async () => root.unmount()); });

function Navigation() {
  const navigate = useNavigate();
  return createElement('nav', null,
    createElement('button', { 'data-nav': 'away', onClick: () => navigate('/away') }, 'Away'),
    createElement('button', { 'data-nav': 'back', onClick: () => navigate('/page') }, 'Back'));
}

async function settle(ms = 25) {
  await act(async () => { await new Promise((resolve) => window.setTimeout(resolve, ms)); });
}

async function render(Component) {
  await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ['/page'] },
    createElement(Navigation),
    createElement(RetainedRoutes, { paths: ['/page', '/away'] },
      createElement(Route, { path: '/page', element: createElement(Component) }),
      createElement(Route, { path: '/away', element: createElement('p', null, 'Another page') }),
    ))));
  await settle();
}

async function click(button) {
  assert.ok(button);
  await act(async () => button.click());
  await settle();
}

const headers = () => [...document.querySelectorAll('.panel-card-header')];
const expanded = () => headers().filter((header) => header.getAttribute('aria-expanded') === 'true');

test('Quick Assign shows all 100 OTON points without the false no-available warning', async () => {
  responses.set('/api/planters/', [{ id: 23, organization_id: 21, organization_name: 'OTON',
    status: 'active', participant_count: 10, registration_pending: false }]);
  responses.set('/api/planter-auth/organizations', { organizations: [{ id: 21, name: 'OTON' }] });
  useMapStore.setState({
    projectSites: { features: [{ type: 'Feature', id: 14,
      properties: { name: 'OTON', organization_id: 21, point_count: 100 } }] },
    points: Array.from({ length: 100 }, (_, index) => ({
      id: index + 1, point_num: index + 1, analysis_id: 1, latitude: 10.5,
      longitude: 123.5 + index * .00001, source_project_site_id: 14,
      source_organization_id: 21, planting_status: 'planned', species: 'Rhizophora',
    })),
  });
  await render(pages.get('PlanterManagement'));
  const quickAssign = headers().find((header) => header.textContent.includes('Quick Assign'));
  if (quickAssign.getAttribute('aria-expanded') === 'false') await click(quickAssign);
  const organization = document.querySelector('select');
  await act(async () => {
    organization.value = '21';
    organization.dispatchEvent(new window.Event('change', { bubbles: true }));
  });
  await settle();
  assert.match(document.body.textContent, /100 available points in this project site/);
  assert.doesNotMatch(document.body.textContent, /No points are available to assign in/);
  const count = document.getElementById('organization-point-count');
  assert.equal(count.disabled, false);
  assert.equal(count.value, '100');
  const assign = [...document.querySelectorAll('button')].find((button) => /Assign 100 Points/.test(button.textContent));
  assert.ok(assign);
  assert.equal(assign.disabled, false);
});

for (const name of ['MapAnalytics', 'ImageProcessing', 'PlanterManagement', 'ErodedZoneEditor', 'MonitoringMapWorkspace']) {
  test(`${name} opens one section at a time and preserves all-closed state on return`, async () => {
    await render(pages.get(name));
    const panel = document.querySelector('.floating-panel');
    assert.ok(headers().length >= 2);
    assert.equal(expanded().length, 1);
    for (const header of headers()) {
      if (header.getAttribute('aria-expanded') === 'false') await click(header);
      assert.deepEqual(expanded(), [header]);
      for (const sibling of headers()) {
        const body = document.getElementById(sibling.getAttribute('aria-controls'));
        assert.equal(body.getAttribute('aria-hidden'), String(sibling !== header));
        assert.equal(body.hasAttribute('inert'), sibling !== header);
      }
    }
    await click(expanded()[0]);
    assert.equal(expanded().length, 0);
    const readsBefore = requests.length;
    await click(document.querySelector('[data-nav="away"]'));
    await click(document.querySelector('[data-nav="back"]'));
    assert.equal(document.querySelector('.floating-panel'), panel);
    assert.equal(expanded().length, 0);
    assert.equal(requests.length, readsBefore);
  });
}

test('switching sections retains configuration inputs and the selected section on return', async () => {
  await render(pages.get('ImageProcessing'));
  const [upload, configuration] = headers();
  await click(configuration);
  const species = document.querySelector('.config-list select');
  assert.ok(species);
  await act(async () => { species.value = 'bungalon'; species.dispatchEvent(new Event('change', { bubbles: true })); });
  await click(upload);
  await click(configuration);
  assert.equal(document.querySelector('.config-list select').value, 'bungalon');
  await click(document.querySelector('[data-nav="away"]'));
  await click(document.querySelector('[data-nav="back"]'));
  assert.deepEqual(expanded(), [configuration]);
  assert.equal(document.querySelector('.config-list select').value, 'bungalon');
});

test('a programmatic panel change closes the previous section', async () => {
  function Workflow() {
    const [openKey, setOpenKey] = useState('configuration');
    return createElement(Panel, { title: 'Workflow', openKey, onOpenKeyChange: setOpenKey },
      createElement(PanelCard, { panelKey: 'upload', title: 'Upload' }, 'Upload fields'),
      createElement(PanelCard, { panelKey: 'configuration', title: 'Configuration' }, 'Configuration fields'),
      createElement('button', { 'data-reset': true, onClick: () => setOpenKey('upload') }, 'New image'));
  }
  await render(Workflow);
  assert.match(expanded()[0].textContent, /Configuration/);
  await click(document.querySelector('[data-reset]'));
  assert.equal(expanded().length, 1);
  assert.match(expanded()[0].textContent, /Upload/);
});

test('quickly reopening a section cancels its pending collapse animation', async () => {
  await render(pages.get('MapAnalytics'));
  const [overview, legend] = headers();
  await act(async () => legend.click());
  await act(async () => overview.click());
  await settle(60);
  assert.deepEqual(expanded(), [overview]);
  const body = document.getElementById(overview.getAttribute('aria-controls'));
  assert.equal(body.style.height, '120px');
});
