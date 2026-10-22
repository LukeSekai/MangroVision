import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement, useState } from 'react';
import { MemoryRouter, Route, useLocation, useNavigate } from 'react-router-dom';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
import { manilaDateKey } from '../utils/monitoringSchedule.js';
import { STAFF_TOURS } from '../guides/tours.js';

let server, dom, root, createRoot, RetainedRoutes, Panel, PanelCard, useMapStore, useProcessingStore, originalStartProcess;
let PlantingMapWorkspace, Sidebar, MapView, Modal;
const pages = new Map();
const originalGlobals = new Map();
const windowErrors = [];
let requests, responses, mutations;

before(async () => {
  dom = new JSDOM('<div id="root"></div>', { url: 'http://workspace.example', pretendToBeVisual: true });
  dom.window.addEventListener('error', (event) => { windowErrors.push(event.error); event.preventDefault(); });
  const globals = {
    window: dom.window, document: dom.window.document, localStorage: dom.window.localStorage,
    Event: dom.window.Event, Element: dom.window.Element, IS_REACT_ACT_ENVIRONMENT: true,
    FileReader: dom.window.FileReader, File: dom.window.File, FormData: dom.window.FormData,
    requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
    cancelAnimationFrame: dom.window.cancelAnimationFrame.bind(dom.window),
    ResizeObserver: class { observe() {} disconnect() {} },
    fetch: async (input, options) => {
      const path = new URL(input, window.location.origin).pathname;
      requests.push(path);
      if (options?.method === 'POST' && !['/api/zones/validate-boundary', '/api/project-sites/validate-boundary'].includes(path)) mutations.push({ path, body: options.body instanceof FormData ? Object.fromEntries(options.body) : JSON.parse(options.body || '{}') });
      if (responses.has(path)) {
        const value = responses.get(path);
        return typeof value === 'function' ? value(options, new URL(input, window.location.origin)) : value instanceof Response ? value : Response.json(value);
      }
      if (path === '/api/planting-schedules') return Response.json({ schedules: [] });
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
  // JSDOM has no paint engine. Keep Leaflet's real Canvas renderer, polygon
  // hit-testing and mouse events, supplying only dimensions and paint methods.
  window.HTMLCanvasElement.prototype.getContext = function () {
    return new Proxy({}, { get: (target, key) => target[key] ?? (() => {}) });
  };
  Object.defineProperty(window.HTMLElement.prototype, 'clientWidth', {
    configurable: true, get() { return this.classList.contains('map-container') ? 1200 : 0; },
  });
  Object.defineProperty(window.HTMLElement.prototype, 'clientHeight', {
    configurable: true, get() { return this.classList.contains('map-container') ? 800 : 0; },
  });
  ({ createRoot } = await import('react-dom/client'));
  server = await createServer({ configFile: false, appType: 'custom',
    server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    cacheDir: 'node_modules/.vite-panel-tests', optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ Panel, PanelCard } = await server.ssrLoadModule('/src/components/Panel.jsx'));
  ({ default: RetainedRoutes } = await server.ssrLoadModule('/src/components/RetainedRoutes.jsx'));
  ({ default: PlantingMapWorkspace } = await server.ssrLoadModule('/src/components/PlantingMapWorkspace.jsx'));
  ({ default: Sidebar } = await server.ssrLoadModule('/src/components/Sidebar.jsx'));
  ({ default: MapView } = await server.ssrLoadModule('/src/components/MapView.jsx'));
  ({ default: Modal } = await server.ssrLoadModule('/src/components/Modal.jsx'));
  for (const page of ['MapAnalytics', 'ImageProcessing', 'PlanterManagement', 'ErodedZoneEditor', 'MonitoringMapWorkspace', 'OrganizationMonitoring']) {
    const module = await server.ssrLoadModule(`/src/pages/${page}.jsx`);
    pages.set(page, module.default);
  }
  ({ useMapStore } = await server.ssrLoadModule('/src/stores/mapStore.js'));
  ({ useProcessingStore } = await server.ssrLoadModule('/src/stores/processingStore.js'));
  originalStartProcess = useProcessingStore.getState().startProcess;
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
  windowErrors.length = 0;
  requests = [];
  responses = new Map([['/api/analyses/', []], ['/api/project-sites', { features: [] }], ['/api/planters/map-points', []]]);
  mutations = [];
  useMapStore.getState().resetWorkspaceData();
  useMapStore.setState({ stats: { analyses: [], points: [], total_analyses: 0 } });
  useProcessingStore.getState().reset();
  useProcessingStore.setState({ startProcess: originalStartProcess });
  root = createRoot(document.getElementById('root'));
});
afterEach(async () => {
  await act(async () => root.unmount());
  await new Promise((resolve) => dom.window.requestAnimationFrame(resolve));
  assert.deepEqual(windowErrors, [], 'map teardown must not leave callbacks drawing on a removed canvas');
});

function Navigation() {
  const navigate = useNavigate();
  const location = useLocation();
  return createElement('nav', null,
    createElement('span', { 'data-current-path': true }, location.pathname),
    createElement('button', { 'data-nav': 'away', onClick: () => navigate('/away') }, 'Away'),
    createElement('button', { 'data-nav': 'back', onClick: () => navigate('/page') }, 'Back'),
    createElement('button', { 'data-nav': 'browser-back', onClick: () => navigate(-1) }, 'Browser Back'));
}

async function renderPlantingWorkspace(entry = '/map') {
  const tools = { '/map': 'MapAnalytics', '/zones': 'ErodedZoneEditor', '/planters': 'PlanterManagement', '/processing': 'ImageProcessing' };
  useMapStore.setState({ layerVisibility: { points: true, orthophoto: true, siteZones: true, projectSites: true, forbidden: true, eroded: true, warnings: true } });
  await act(async () => root.render(createElement(MemoryRouter, { initialEntries: [entry] },
    createElement(Sidebar), createElement(MapView),
    createElement(PlantingMapWorkspace, null,
      createElement(RetainedRoutes, { paths: [...Object.keys(tools), '/dashboard'] },
        ...Object.entries(tools).map(([path, page]) => createElement(Route, { key: path, path, element: createElement(pages.get(page)) })),
        createElement(Route, { path: '/dashboard', element: createElement('p', { 'data-dashboard': true }, 'Dashboard') }),
      )))));
  await settle();
}

const toolLink = (path) => document.querySelector(`.planting-tool-nav a[href="${path}"]`);
const layerInput = (label) => [...document.querySelectorAll('.leaflet-control-layers label')]
  .find((element) => element.textContent.trim() === label)?.querySelector('input');

test('Planting Map switches all four real tools in one panel while keeping forms and the map', async () => {
  await renderPlantingWorkspace();
  const map = useMapStore.getState().mapInstance;
  const frame = document.querySelector('.planting-workspace-panel');
  assert.deepEqual([...document.querySelectorAll('.planting-tool-nav a')].map((link) => link.textContent),
    ['Overview', 'Zone Editor', 'Assign Points', 'Analyze Image']);
  assert.equal(document.querySelectorAll('.floating-panel').length, 1);
  assert.equal(document.querySelector('.sidebar-nav a[href="/zones"]'), null);
  assert.equal(document.querySelector('.sidebar-nav a[href="/planters"]'), null);
  assert.equal(document.querySelector('.sidebar-nav a[href="/processing"]'), null);
  assert.equal(document.querySelector('[data-guide-card="layers"]'), null);

  await click(toolLink('/processing'));
  await click([...document.querySelectorAll('[data-guide-panel="Analyze Image"] .panel-card-header')]
    .find((header) => header.textContent.includes('Configuration')));
  const species = document.querySelector('.config-list select');
  await act(async () => { species.value = 'bungalon'; species.dispatchEvent(new Event('change', { bubbles: true })); });
  await click(toolLink('/zones'));
  assert.equal(document.querySelector('[data-guide-panel="Zone Editor"]').style.display, '');
  await click(toolLink('/planters'));
  assert.ok(document.querySelector('#assign-organization'));
  assert.equal(document.querySelector('.sidebar-nav a[href="/map"]').classList.contains('active'), true);
  await click(toolLink('/processing'));
  assert.equal(document.querySelector('.config-list select'), species);
  assert.equal(species.value, 'bungalon');
  assert.equal(document.querySelector('.planting-workspace-panel'), frame);
  assert.equal(document.querySelectorAll('.floating-panel').length, 1);
  assert.equal(useMapStore.getState().mapInstance, map);

  await click(document.querySelector('.panel-toggle'));
  assert.equal(frame.hasAttribute('inert'), true);
  await click(document.querySelector('.sidebar-nav a[href="/dashboard"]'));
  assert.ok(document.querySelector('[data-dashboard]'));
  assert.equal(document.getElementById('planting-map-workspace').hasAttribute('inert'), false);
  await click(document.querySelector('.sidebar-nav a[href="/map"]'));
  assert.equal(frame.classList.contains('floating-panel-hidden'), true);
  await click(document.querySelector('.panel-toggle'));
  assert.equal(frame.hasAttribute('inert'), false);
  await click(toolLink('/processing'));
  assert.equal(species.value, 'bungalon');
  assert.equal(mutations.length, 0);
});

test('direct assignment links work and the workspace still locks Zone Editor during processing', async () => {
  await renderPlantingWorkspace('/planters?section=assign');
  assert.equal(toolLink('/planters').getAttribute('aria-current'), 'page');
  assert.equal(document.querySelector('#assign-organization').closest('[inert]'), null);
  await click(toolLink('/processing'));
  await act(async () => useProcessingStore.setState({ processing: true, stage: 'Detecting canopy' }));
  const zoneLink = toolLink('/zones');
  assert.equal(zoneLink.getAttribute('aria-disabled'), 'true');
  await click(zoneLink);
  assert.equal(toolLink('/processing').getAttribute('aria-current'), 'page');
  await act(async () => useProcessingStore.setState({ processing: false }));
  await click(zoneLink);
  assert.equal(toolLink('/zones').getAttribute('aria-current'), 'page');
  assert.equal(mutations.length, 0);
});

test('upper-left layers stay in sync with shared map state and survive tool changes', async () => {
  await renderPlantingWorkspace();
  const map = useMapStore.getState().mapInstance;
  assert.ok(document.querySelector('.leaflet-top.leaflet-left [data-guide-layers]'));
  const labels = { points: 'Planting Points', orthophoto: 'Drone Orthomosaic', siteZones: 'Assignment Zones', projectSites: 'Project Sites', forbidden: 'Forbidden Zones', eroded: 'Eroded Zones', warnings: 'Warning Zones' };
  for (const [key, label] of Object.entries(labels)) {
    const input = layerInput(label);
    assert.equal(input.checked, true);
    await click(input);
    assert.equal(useMapStore.getState().layerVisibility[key], false);
    assert.equal(input.checked, false);
    await act(async () => useMapStore.getState().showLayers([key]));
    // Leaflet rebuilds the checkbox DOM after programmatic visibility changes.
    assert.equal(layerInput(label).checked, true);
  }
  await click(layerInput('Drone Orthomosaic'));
  await click(layerInput('Warning Zones'));
  await click(layerInput('OpenStreetMap'));
  await click(toolLink('/zones'));
  await click(toolLink('/processing'));
  await click(toolLink('/map'));
  assert.equal(layerInput('Drone Orthomosaic').checked, false);
  assert.equal(layerInput('Warning Zones').checked, false);
  assert.equal(layerInput('OpenStreetMap').checked, true);
  assert.equal(useMapStore.getState().mapInstance, map);
  assert.equal(mutations.length, 0);
});

function zoneFixtures() {
  const rectangle = (id, west, south, east, north, properties = {}) => ({
    type: 'Feature', id, properties,
    geometry: { type: 'Polygon', coordinates: [[[west, south], [east, south], [east, north], [west, north], [west, south]]] },
  });
  const collection = (...features) => ({ type: 'FeatureCollection', features });
  const eroded = rectangle(1, 122.6250, 10.7799, 122.6252, 10.7801,
    { name: 'Eroded zone 1', description: 'Washed-out ground.' });
  const secondEroded = rectangle(2, 122.6254, 10.7799, 122.6256, 10.7801);
  const warning = rectangle(3, 122.6250, 10.7803, 122.6252, 10.7805,
    { id: 3, name: 'Deep mud', warning_label: 'Difficult access', severity: 'high', notes: '<img src=x onerror=alert(1)>' });
  const forbidden = rectangle(4, 122.6254, 10.7803, 122.6256, 10.7805,
    { name: 'Do not plant', label: 'Legacy label' });
  const site = rectangle(5, 122.6247, 10.7796, 122.6259, 10.7808,
    { name: 'Community site', organization_name: 'Community', organization_id: 11 });
  const assignment = rectangle(6, 122.6248, 10.7797, 122.6258, 10.7807,
    { name: 'Assignment 6' });
  responses.set('/api/project-sites', collection(site));
  useMapStore.setState({
    center: [122.6253, 10.7800], zoom: 18, pointsLoaded: true,
    points: [{ id: 101, point_num: 101, latitude: 10.7800, longitude: 122.6251,
      planting_status: 'planned', species: 'bungalon', source_project_site_id: 5 }],
    erodedZones: collection(eroded, secondEroded), warningZones: collection(warning),
    forbiddenZones: collection(forbidden), projectSites: collection(site), siteZones: collection(assignment),
  });
  return { eroded, secondEroded, warning, forbidden, site, assignment };
}

function boundaryFor(map, feature) {
  let found;
  map.eachLayer((layer) => { if (layer.feature === feature) found = layer; });
  return found;
}

async function hoverMap(map, layer, latitude, longitude) {
  await settle(45); // Allow Leaflet's hover throttle to expire.
  const point = map.latLngToContainerPoint([latitude, longitude]);
  await act(async () => layer._renderer._container.dispatchEvent(new window.MouseEvent('mousemove', {
    bubbles: true, clientX: point.x, clientY: point.y,
  })));
}

for (const pointerType of ['mouse', 'touch']) {
test(`monitoring release brush requires a held primary pointer and ignores hidden dialogs (${pointerType})`, async () => {
  zoneFixtures();
  const base = useMapStore.getState().points[0];
  useMapStore.setState({ showDeadPointsOnly: true, replantingSelectionMode: 'click', points: [
    { ...base, death_at: '2026-10-20', planting_status: 'planted' },
    { ...base, id: 102, point_num: 102, longitude: 122.6253, planting_status: 'planted' },
    { ...base, id: 103, point_num: 103, longitude: 122.6257, death_at: '2026-10-20', planting_status: 'planted' },
  ] });
  function RetainedDialog() {
    const [open, setOpen] = useState(false);
    return createElement('div', null,
      createElement('button', { onClick: () => setOpen(true) }, 'Show test dialog'),
      createElement(Modal, { open, keepMounted: true, title: 'Site details', onCancel: () => setOpen(false) }));
  }
  await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ['/monitoring/map?project_site_id=5&focus=site_points'] },
    createElement(MapView), createElement(pages.get('MonitoringMapWorkspace')), createElement(RetainedDialog))));
  await settle();
  const map = useMapStore.getState().mapInstance;
  const button = (label) => [...document.querySelectorAll('button')].find(element => element.textContent.trim() === label);
  const pointer = async (type, longitude, buttons = 0, target = map.getContainer()) => {
    const position = map.latLngToContainerPoint([base.latitude, longitude]);
    const event = new window.MouseEvent(type, { bubbles: true, clientX: position.x, clientY: position.y, buttons });
    Object.defineProperty(event, 'pointerType', { value: pointerType });
    Object.defineProperty(event, 'pointerId', { value: 1 });
    await act(async () => { target.dispatchEvent(event); await settle(); });
  };
  const drag = async (start, end = start) => {
    await pointer('pointerdown', start, 1);
    await pointer('pointermove', end, 1);
    await pointer('pointerup', end, 0, window);
  };
  assert.ok(document.querySelector('[role="dialog"]').closest('[hidden]'));
  await click(button('Brush select'));
  await pointer('pointermove', base.longitude);
  await pointer('pointermove', 122.6257);
  assert.deepEqual(useMapStore.getState().replantingSelectedPointIds, [], 'hovering in Brush select must not paint');
  await pointer('pointermove', base.longitude, 1);
  assert.deepEqual(useMapStore.getState().replantingSelectedPointIds, [], 'the press must begin on the map');
  await drag(base.longitude);
  await pointer('pointermove', 122.6257);
  assert.deepEqual(useMapStore.getState().replantingSelectedPointIds, [101], 'painting stops when the button is released outside the map');
  await drag(122.6257);
  assert.deepEqual(useMapStore.getState().replantingSelectedPointIds, [101, 103]);
  assert.match(document.querySelector('.monitoring-map-actions').textContent, /2 selected.*Release from project site/);
  const markers = new Map();
  map.eachLayer(marker => { if (marker.options.mgPointId != null) markers.set(marker.options.mgPointId, marker); });
  assert.equal(markers.get(101).options.fillColor, '#7c3aed');
  assert.equal(markers.get(103).options.fillColor, '#7c3aed');
  assert.equal(markers.has(102), false, 'living points stay excluded by the dead-only filter');
  await click(button('Brush erase'));
  await pointer('pointermove', base.longitude);
  assert.deepEqual(useMapStore.getState().replantingSelectedPointIds, [101, 103], 'hovering must not erase');
  await drag(base.longitude);
  assert.deepEqual(useMapStore.getState().replantingSelectedPointIds, [103]);
  await click(button('Show test dialog'));
  await drag(122.6257);
  assert.deepEqual(useMapStore.getState().replantingSelectedPointIds, [103], 'a visible dialog still pauses the brush');
  await click(document.querySelector('.modal-close'));
  await drag(122.6257);
  assert.deepEqual(useMapStore.getState().replantingSelectedPointIds, []);
  await click(button('Move / click'));
  await act(async () => markers.get(101).fire('click'));
  assert.deepEqual(useMapStore.getState().replantingSelectedPointIds, [101]);
  await click(button('Brush select'));
  await click(button('Clear'));
  for (const reason of ['blur', 'pointercancel', 'pointerleave', 'lostpointercapture']) {
    await pointer('pointerdown', 122.6254, 1);
    if (reason === 'blur') await act(async () => window.dispatchEvent(new window.Event('blur')));
    else await pointer(reason, 122.6254, 0, reason === 'pointercancel' ? window : map.getContainer());
    await pointer('pointermove', base.longitude, 1);
    assert.deepEqual(useMapStore.getState().replantingSelectedPointIds, [], `${reason} must end the stroke`);
  }
  await act(async () => document.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
  assert.equal(useMapStore.getState().replantingSelectionMode, 'click');
  assert.equal(document.querySelector('.replanting-brush'), null);
  assert.equal(map.dragging.enabled(), true);
  assert.equal(mutations.length, 0, 'selection must never release locations without confirmation');
});
}

test('clicking a mortality site focuses its map and excludes other sites from release selection', async () => {
  const { site } = zoneFixtures();
  site.properties.mortality_warning = { active: true, mortality_rate_pct: 50 };
  const secondSite = { ...site, id: 7, properties: { ...site.properties, name: 'Second project site' },
    geometry: { type: 'Polygon', coordinates: [[[122.626, 10.779], [122.627, 10.779], [122.627, 10.781], [122.626, 10.781], [122.626, 10.779]]] } };
  const base = useMapStore.getState().points[0];
  useMapStore.setState({ projectSites: { type: 'FeatureCollection', features: [site, secondSite] }, points: [
    { ...base, death_at: '2026-10-20', planting_status: 'planted', source_organization_id: 11 },
    { ...base, id: 102, longitude: 122.6265, death_at: '2026-10-20', planting_status: 'planted', source_organization_id: 11, source_project_site_id: 7 },
  ] });
  await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ['/monitoring/map'] },
    createElement(MapView), createElement(pages.get('MonitoringMapWorkspace')))));
  await settle();
  const siteButton = [...document.querySelectorAll('.monitoring-map-site-list button')].find((button) => button.textContent === 'Second project site');
  await click(siteButton);
  const map = useMapStore.getState().mapInstance;
  assert.equal(document.querySelector('#monitoring-map-project-site').value, '7');
  assert.ok(map.getBounds().contains([10.7800, 122.6265]));
  const markers = [];
  map.eachLayer((layer) => { if (layer.options?.mgPointId != null) markers.push(layer.options.mgPointId); });
  assert.deepEqual(markers, [102]);
  await act(async () => useMapStore.getState().paintReplantingPoints([101, 102], 'select'));
  assert.deepEqual(useMapStore.getState().replantingSelectedPointIds, [102]);
  assert.equal(mutations.length, 0);
});

test('recording a death keeps its marker and site membership across all planting tools until release', async () => {
  const zones = zoneFixtures();
  const planted = { ...useMapStore.getState().points[0], planting_status: 'planted',
    map_status: 'planted', source_organization_id: 11, death_at: null };
  const otherLiving = { ...planted, id: 102, point_num: 102, latitude: 10.7802, longitude: 122.6253 };
  useMapStore.setState({ points: [planted, otherLiving] });
  responses.set('/api/planters/map-points/101/death', { death_at: '2026-10-14',
    death_reason: 'Not determined', death_reason_category: 'unknown', death_notes: '' });
  await renderPlantingWorkspace('/zones');
  const map = useMapStore.getState().mapInstance;
  const markerFor = (id) => {
    let found;
    map.eachLayer((layer) => { if (layer.options?.mgPointId === id) found = layer; });
    return found;
  };
  const siteCount = (status) => {
    const feature = useMapStore.getState().projectSites.features.find((site) => site.id === zones.site.id);
    const content = boundaryFor(map, feature).getPopup().getContent();
    return content().querySelector(`.project-site-info-stat.is-${status} dd`).textContent;
  };
  assert.equal(markerFor(101).options.mgDead, false);
  await act(async () => useMapStore.getState().markPointDead(101, 'unknown', ''));
  // A cached monitoring filter must not hide living or dead markers in the
  // planting tools, including while navigating away and returning.
  await act(async () => useMapStore.setState({ showDeadPointsOnly: true }));
  for (const path of ['/zones', '/map', '/planters', '/processing', '/zones']) {
    await click(toolLink(path));
    assert.equal(useMapStore.getState().mapInstance, map);
    const deadMarker = markerFor(101);
    assert.ok(deadMarker, `the dead seedling must remain visible on ${path}`);
    assert.equal(deadMarker.options.mgDead, true);
    assert.equal(deadMarker.options.fillColor, '#7f1d1d');
    assert.match(deadMarker.getPopup().getContent(), /Cause:.*Not determined/);
    assert.ok(markerFor(102), `living seedlings must remain visible on ${path}`);
    assert.equal(siteCount('dead'), '1');
    assert.equal(siteCount('planted'), '1');
    assert.equal(useMapStore.getState().points.find((point) => point.id === 101).source_project_site_id, 5);
  }
  // Explicit release refreshes the physical point's current status and owner;
  // the location can remain available on the map without belonging to the site.
  await act(async () => useMapStore.setState({ points: [
    { ...planted, planting_status: 'planned', map_status: 'planned', source_project_site_id: null,
      source_site_id: null, source_organization_id: null }, otherLiving,
  ] }));
  assert.equal(markerFor(101).options.mgDead, false);
  assert.equal(siteCount('dead'), '0');
  assert.equal(siteCount('planted'), '1');
  assert.deepEqual(requests.filter((path) => /release|reset-to-planned|replanting/.test(path)), []);
});

test('project-site hover shows analytics above assignment zones while point hover keeps individual details', async () => {
  const zones = zoneFixtures();
  zones.site.properties.planting_history = { total:3, planted:2, dead:1, survival_rate_pct:66.67 };
  const base = useMapStore.getState().points[0];
  useMapStore.setState({ points: [
    { ...base, planting_status: 'planted' },
    { ...base, id: 102, point_num: 102, latitude: 10.7801, longitude: 122.6253, planting_status: 'planted' },
    { ...base, id: 103, point_num: 103, latitude: 10.7802, longitude: 122.6255, planting_status: 'planted', death_at: '2026-10-18' },
    { ...base, id: 104, point_num: 104, latitude: 10.7803, longitude: 122.6257 },
  ] });
  await renderPlantingWorkspace();
  const map = useMapStore.getState().mapInstance;
  const site = boundaryFor(map, zones.site);
  const assignment = boundaryFor(map, zones.assignment);
  let marker;
  map.eachLayer((layer) => { if (layer.options?.mgPointId === 101) marker = layer; });
  assert.equal(site._renderer, assignment._renderer);
  assert.equal(site._renderer, marker._renderer);
  const initialRequests = [...requests];
  // This location is inside both boundaries and between the planting points.
  await hoverMap(map, site, 10.77985, 122.6253);
  const card = document.querySelector('.project-site-tooltip .project-site-info');
  assert.ok(card, 'the project site must receive hover ahead of its assignment zone');
  assert.equal(card.querySelector('.project-site-info-title').textContent, 'Community site');
  assert.equal(card.querySelector('.project-site-info-total strong').textContent, '3');
  assert.equal(card.querySelector('.project-site-info-current summary').textContent, 'Current site locations · 4');
  assert.equal(card.querySelector('.is-planted dd').textContent, '2');
  assert.equal(card.querySelector('.is-dead dd').textContent, '1');
  assert.equal(card.querySelector('.project-site-info-survival dd').textContent, '66.67%');
  assert.equal(document.querySelector('.site-zone-tooltip'), null);
  await hoverMap(map, site, base.latitude, base.longitude);
  assert.equal(marker.isTooltipOpen(), true);
  assert.equal(document.querySelector('.project-site-tooltip'), null);
  assert.match(document.querySelector('.point-status-tooltip-title').textContent, /Point #101/);
  await hoverMap(map, site, 10.77985, 122.6253);
  assert.ok(document.querySelector('.project-site-tooltip .project-site-info-survival'));
  assert.deepEqual(requests, initialRequests, 'hovering must reuse the already loaded data');
  assert.equal(mutations.length, 0);
});

test('site history in the map popup retains deaths after releasing and reassigning their locations', async () => {
  const zones = zoneFixtures();
  zones.site.properties.planting_history = { total: 3, planted: 2, dead: 1, survival_rate_pct: 66.67 };
  const base = useMapStore.getState().points[0];
  const living = { ...base, planting_status: 'planted' };
  const dead = { ...base, id: 102, point_num: 102, longitude: 122.6253,
    planting_status: 'planted', death_at: '2026-10-18' };
  const otherLiving = { ...living, id: 103, point_num: 103, longitude: 122.6255 };
  useMapStore.setState({ points: [living, dead, otherLiving] });
  await renderPlantingWorkspace();
  const map = useMapStore.getState().mapInstance;
  const assertHistory = (currentTotal, currentDead) => {
    const card = boundaryFor(map, zones.site).getPopup().getContent()();
    assert.equal(card.querySelector('.project-site-info-history .project-site-info-total strong').textContent, '3');
    assert.equal(card.querySelector('.project-site-info-history .is-dead dd').textContent, '1');
    assert.equal(card.querySelector('.project-site-info-survival dd').textContent, '66.67%');
    assert.equal(card.querySelector('.project-site-info-current summary').textContent, `Current site locations · ${currentTotal}`);
    assert.equal(card.querySelector('.project-site-info-current .is-dead dd').textContent, String(currentDead));
  };
  assertHistory(3, 1);
  const released = { ...dead, death_at: null, planting_status: 'planned', source_project_site_id: null,
    source_site_id: null, source_organization_id: null };
  await act(async () => useMapStore.setState({ points: [living, released, otherLiving] }));
  assertHistory(2, 0);
  const replanted = { ...released, planting_status: 'planted', source_project_site_id: 7, source_organization_id: 12 };
  await act(async () => useMapStore.setState({ points: [living, replanted, otherLiving] }));
  assertHistory(2, 0);
  assert.equal(mutations.length, 0);
});

test('recording and correcting a death refresh the site historical chart immediately', async () => {
  const zones = zoneFixtures();
  zones.site.properties.planting_history = { total: 2, planted: 2, dead: 0, survival_rate_pct: 100 };
  const base = { ...useMapStore.getState().points[0], planting_status: 'planted' };
  useMapStore.setState({ points: [base, { ...base, id: 102, longitude: 122.6253 }] });
  await renderPlantingWorkspace();
  const map = useMapStore.getState().mapInstance;
  const siteCard = () => {
    const feature = useMapStore.getState().projectSites.features.find((site) => site.id === zones.site.id);
    return boundaryFor(map, feature).getPopup().getContent()();
  };
  assert.equal(siteCard().querySelector('.project-site-info-history .is-dead dd').textContent, '0');
  const updated = { ...zones.site, properties: { ...zones.site.properties,
    planting_history: { total: 2, planted: 1, dead: 1, survival_rate_pct: 50 } } };
  responses.set('/api/planters/map-points/101/death', { death_at: '2026-10-18', death_reason_category: 'waves' });
  responses.set('/api/project-sites', (options) => {
    assert.equal(options.cache, 'reload');
    return Response.json({ type: 'FeatureCollection', features: [updated] });
  });
  await act(async () => useMapStore.getState().markPointDead(101, 'waves', ''));
  assert.equal(siteCard().querySelector('.project-site-info-history .is-dead dd').textContent, '1');
  assert.equal(siteCard().querySelector('.project-site-info-survival dd').textContent, '50%');
  responses.set('/api/planters/map-points/101/restore', { ok: true });
  responses.set('/api/project-sites', { type: 'FeatureCollection', features: [zones.site] });
  await act(async () => useMapStore.getState().restorePointToPlanted(101));
  assert.equal(siteCard().querySelector('.project-site-info-history .is-dead dd').textContent, '0');
  assert.equal(siteCard().querySelector('.project-site-info-survival dd').textContent, '100%');
  assert.equal(requests.filter((path) => path === '/api/project-sites').length, 2);
});

test('site mortality retains site analytics without orange polygons or warnings on unaffected plants', async () => {
  const zones = zoneFixtures();
  zones.site.properties.mortality_warning = { active: true, total_planted: 100, dead_count: 30, mortality_rate_pct: 30 };
  const base = useMapStore.getState().points[0];
  const warning = { survival_warning: true, warning_zone_ids: [-5], warning_severity: 'high',
    warning_summary: 'High mortality (30% dead). Check before planting.' };
  useMapStore.setState({ points: [
    { ...base, ...warning },
    { ...base, ...warning, id: 102, longitude: 122.6253, planting_status: 'planted' },
    { ...base, ...warning, id: 103, longitude: 122.6255, planting_status: 'planted', death_at: '2026-10-20' },
    { ...base, ...warning, id: 104, longitude: 122.6257, warning_zone_ids: [-5, 3] },
    { ...base, id: 105, longitude: 122.6254, survival_warning: true, warning_zone_ids: [],
      warning_summary: 'Wave exposure: Wave-caused death recorded here. Check conditions before planting.' },
  ] });
  await renderPlantingWorkspace();
  const map = useMapStore.getState().mapInstance;
  const site = boundaryFor(map, zones.site);
  assert.equal(site.options.color, '#0284c7');
  const checkPointColors = () => map.eachLayer((marker) => {
    if (marker.options.mgPointId == null) return;
    assert.equal(marker.options.fillColor, marker.options.mgBaseColor);
    assert.equal(marker.options.color, [104, 105].includes(marker.options.mgPointId) ? '#f59e0b' : '#000000');
  });
  checkPointColors();
  await act(async () => map.setZoom(20, { animate: false }));
  checkPointColors();
  await act(async () => map.setZoom(18, { animate: false }));
  assert.equal(document.querySelector('.project-site-danger-label'), null);
  await hoverMap(map, site, 10.77985, 122.6253);
  const card = document.querySelector('.project-site-tooltip .project-site-info');
  assert.ok(card);
  assert.match(card.textContent, /Community site/);
  assert.match(card.textContent, /High mortality · 30%/);
  assert.ok(card.querySelector('.project-site-info-survival'));
  assert.equal(document.querySelector('.project-site-danger-label'), null);
  await hoverMap(map, site, 10.7800, 122.6251);
  assert.ok(document.querySelector('.point-status-tooltip'));
  assert.doesNotMatch(document.querySelector('.point-status-tooltip').textContent, /Warning:|High mortality/);
  assert.equal(document.querySelector('.project-site-tooltip'), null);
  assert.equal(document.querySelector('.project-site-danger-label'), null);
  await hoverMap(map, site, 10.7800, 122.6254);
  assert.match(document.querySelector('.point-status-tooltip').textContent, /Wave exposure: Wave-caused death recorded here/);
  assert.doesNotMatch(document.querySelector('.point-status-tooltip').textContent, /High mortality/);
  await hoverMap(map, site, 10.7800, 122.6253);
  assert.match(document.querySelector('.point-status-tooltip').textContent, /Planted/i);
  assert.doesNotMatch(document.querySelector('.point-status-tooltip').textContent, /Warning:|High mortality|Wave exposure/);
  await act(async () => useMapStore.setState({ projectSites: { type: 'FeatureCollection', features: [] } }));
  assert.equal(document.querySelector('.project-site-danger-label'), null);
  assert.equal(mutations.length, 0);
});

test('zone hover labels work through the point canvas and identify overlapping boundaries', async () => {
  const zones = zoneFixtures();
  await renderPlantingWorkspace('/zones');
  const map = useMapStore.getState().mapInstance;
  const eroded = boundaryFor(map, zones.eroded);
  let marker;
  map.eachLayer((layer) => { if (layer.options?.mgPointId === 101) marker = layer; });
  assert.ok(eroded);
  assert.ok(marker);
  assert.ok(eroded._renderer === marker._renderer, 'zones must share point-canvas hit-testing');
  await hoverMap(map, eroded, 10.78006, 122.62506);
  assert.equal(document.querySelector('.leaflet-tooltip .zone-tooltip-title')?.textContent, 'Eroded zone 1');
  assert.match(document.querySelector('.leaflet-tooltip')?.textContent, /Washed-out ground/);

  await hoverMap(map, eroded, 10.7800, 122.6251);
  assert.equal(marker.isTooltipOpen(), true);
  const point = map.latLngToContainerPoint([10.7800, 122.6251]);
  await act(async () => marker._renderer._container.dispatchEvent(new window.MouseEvent('click', {
    bubbles: true, clientX: point.x, clientY: point.y,
  })));
  assert.equal(useMapStore.getState().selectedPointId, 101);
  map.closePopup();

  const warning = boundaryFor(map, zones.warning);
  await hoverMap(map, warning, 10.7804, 122.6251);
  assert.equal(document.querySelector('.leaflet-tooltip .zone-tooltip-title')?.textContent, 'Deep mud');
  const info = document.querySelector('.leaflet-tooltip');
  assert.match(info.textContent, /Difficult access.*Severity: high/);
  assert.match(info.textContent, /<img src=x onerror=alert\(1\)>/);
  assert.equal(info.querySelector('img'), null, 'saved notes must remain text');
  await hoverMap(map, boundaryFor(map, zones.forbidden), 10.7804, 122.6255);
  assert.equal(document.querySelector('.leaflet-tooltip .zone-tooltip-title')?.textContent, 'Do not plant');
  assert.match(document.querySelector('.leaflet-tooltip').textContent, /No planting allowed/);
  await hoverMap(map, boundaryFor(map, zones.secondEroded), 10.7800, 122.6255);
  assert.equal(document.querySelector('.leaflet-tooltip .zone-tooltip-title')?.textContent, 'Eroded Zone 2');
  assert.equal(mutations.length, 0);
});

test('each zone list locates its own boundary, reveals hidden layers and keeps deletion separate', async () => {
  const zones = zoneFixtures();
  await renderPlantingWorkspace('/zones');
  const map = useMapStore.getState().mapInstance;
  const frames = [];
  const fitBounds = map.fitBounds.bind(map);
  map.fitBounds = (bounds, options) => { frames.push({ bounds, options }); return fitBounds(bounds, options); };
  const findButton = (name) => document.querySelector(`.zone-locate[aria-label="Show ${name} on map"]`);
  const openList = async (title) => {
    const header = [...document.querySelectorAll('.panel-card-header')].find((element) => element.textContent.includes(title));
    if (header.getAttribute('aria-expanded') !== 'true') await click(header);
  };
  for (const [title, name, feature, visibility] of [
    ['Eroded Zones', 'Eroded zone 1', zones.eroded, 'eroded'],
    ['Eroded Zones', 'Eroded Zone 2', zones.secondEroded, 'eroded'],
    ['Warning Zones', 'Deep mud', zones.warning, 'warnings'],
    ['Project Sites', 'Community site', zones.site, 'projectSites'],
  ]) {
    await openList(title);
    await act(async () => useMapStore.getState().toggleLayer(visibility));
    assert.equal(useMapStore.getState().layerVisibility[visibility], false);
    await click(findButton(name));
    const boundary = boundaryFor(map, feature);
    assert.ok(frames.at(-1).bounds.equals(boundary.getBounds()));
    assert.equal(useMapStore.getState().layerVisibility[visibility], true);
    assert.equal(findButton(name).getAttribute('aria-pressed'), 'true');
    assert.equal(boundary.options.weight, 4);
    assert.ok(map.getBounds().contains(boundary.getBounds()));
    assert.ok(map.latLngToContainerPoint(boundary.getBounds().getCenter()).x < map.getSize().x - 436);
    assert.equal(boundary.isTooltipOpen(), true);
    assert.equal(document.querySelectorAll('.zone-locate[aria-pressed="true"]').length, 1);
    assert.equal(frames.at(-1).options.paddingBottomRight[0], 436);
  }
  assert.equal(boundaryFor(map, zones.eroded).options.weight, 2);
  // Repeated selection must navigate again after a user has panned away.
  const frameCount = frames.length;
  await click(findButton('Community site'));
  assert.equal(frames.length, frameCount + 1);
  await openList('Eroded Zones');
  await click(document.querySelector('[aria-label="Delete Eroded zone 1"]'));
  assert.match(document.querySelector('[role="dialog"]').textContent, /Delete "Eroded zone 1"/);
  assert.equal(frames.length, frameCount + 1, 'delete must not move the map');
  assert.equal(mutations.length, 0);
});

test('automatic mortality zones appear in Warning Zones and cannot be deleted manually', async () => {
  const zones = zoneFixtures();
  zones.warning.id = -5;
  Object.assign(zones.warning.properties, { id: -5, name: 'Community site', automatic: true,
    project_site_id: 5, warning_type: 'high_mortality', warning_label: 'High mortality (at least 30%)', notes: '30 of 100 planted seedlings have recorded deaths.' });
  zones.site.properties.mortality_warning = { active: true, total_planted: 100, dead_count: 30, mortality_rate_pct: 30 };
  await renderPlantingWorkspace('/zones');
  const header = [...document.querySelectorAll('.panel-card-header')].find(element => element.textContent.includes('Warning Zones'));
  if (header.getAttribute('aria-expanded') !== 'true') await click(header);
  const row = document.querySelector('.warning-zone-row');
  assert.match(row.textContent, /Automatic.*High mortality/);
  assert.equal(row.querySelector('[aria-label^="Delete"]'), null);
  await click(row.querySelector('.zone-locate'));
  assert.equal(boundaryFor(useMapStore.getState().mapInstance, zones.warning), undefined);
  const outline = boundaryFor(useMapStore.getState().mapInstance, zones.site);
  assert.equal(outline.isTooltipOpen(), true);
  assert.equal(outline.options.fillOpacity, 0, 'focusing the warning must not shade the points');
  const card = document.querySelector('.project-site-tooltip .project-site-info');
  assert.ok(card, 'The warning opens the general project-site card');
  assert.match(card.textContent, /Community site/);
  assert.match(card.textContent, /High mortality · 30%/);
  assert.ok(card.querySelector('.project-site-info-survival'));
  assert.ok(card.querySelector('.project-site-info-chart'));
  assert.equal(document.querySelector('.zone-tooltip-warning'), null);
  assert.doesNotMatch(card.textContent, /Severity:|All recorded causes|Review this area/);
  assert.equal(mutations.length, 0);
});

const namedButton = (name) => [...document.querySelectorAll('[data-guide-panel="Zone Editor"] button')]
  .find((button) => button.textContent.trim() === name);

async function drawZone(kind, { start = true, corners = [[10.7799, 122.6250], [10.7799, 122.6252], [10.7801, 122.6252], [10.7801, 122.6250]] } = {}) {
  if (start) {
    const drawHeader = [...document.querySelectorAll('.panel-card-header')].find((element) => element.textContent.includes('Draw Zone'));
    if (drawHeader.getAttribute('aria-expanded') !== 'true') await click(drawHeader);
    await click(namedButton(kind === 'warning' ? 'Warning' : 'Eroded'));
    await click(namedButton(kind === 'warning' ? 'Draw New Warning Zone' : 'Draw New Eroded Zone'));
  }
  const map = useMapStore.getState().mapInstance;
  const container = map.getContainer();
  const pointer = (type, latlng) => {
    const point = map.latLngToContainerPoint(latlng);
    const event = new window.MouseEvent(type, { bubbles: true, button: 0, clientX: point.x, clientY: point.y });
    Object.defineProperty(event, 'pointerId', { value: 1 });
    Object.defineProperty(event, 'pointerType', { value: 'mouse' });
    return event;
  };
  await act(async () => container.dispatchEvent(pointer('pointerdown', corners[0])));
  await settle(375);
  for (const corner of corners.slice(1)) {
    await act(async () => container.dispatchEvent(pointer('pointermove', corner)));
  }
  await act(async () => window.dispatchEvent(pointer('pointerup', corners[0])));
  await settle();
}

for (const kind of ['warning', 'eroded']) {
  test(`${kind} drawing outside the map blocks Save and can be redrawn inside`, async () => {
    zoneFixtures();
    const checks = [];
    responses.set('/api/zones/validate-boundary', (options) => {
      checks.push(JSON.parse(options.body));
      return Response.json({ detail: 'This zone extends outside the mapped area. Redraw it fully inside the drone map.' }, { status: 400 });
    });
    await renderPlantingWorkspace('/zones');
    await drawZone(kind);
    const label = kind === 'warning' ? 'Save Warning Zone' : 'Save Zone';
    assert.equal(namedButton(label).disabled, true);
    assert.match(document.querySelector('.save-form [role="alert"]').textContent, /outside the mapped area.*Redraw/);
    assert.equal(checks.length, 1);
    const ring = checks[0].geometry.coordinates[0];
    assert.equal(ring.length, 5);
    assert.deepEqual(ring[0], ring.at(-1));
    assert.ok(ring[0][0] > 122 && ring[0][1] < 11, 'send GeoJSON longitude/latitude ordering');
    await click(namedButton(label));
    assert.equal(mutations.length, 0);

    responses.set('/api/zones/validate-boundary', { valid: true });
    responses.set(`/api/zones/${kind === 'warning' ? 'warnings' : 'eroded'}`, { id: 8 });
    await click(namedButton('Redraw zone'));
    assert.equal(document.querySelector('.save-form'), null);
    await drawZone(kind, { start: false });
    assert.equal(namedButton(label).disabled, false);
    assert.equal(document.querySelector('.save-form [role="alert"]'), null);
    await click(namedButton(label));
    assert.equal(mutations.length, 1);
    assert.equal(mutations[0].path, `/api/zones/${kind === 'warning' ? 'warnings' : 'eroded'}`);
    assert.match(document.querySelector('[data-guide-panel="Zone Editor"]').textContent, /zone saved|Zone saved/);
  });
}

test('failed map-boundary checks keep Save disabled and offer a retry', async () => {
  zoneFixtures();
  responses.set('/api/zones/validate-boundary', Response.json({ detail: 'Could not check the map boundary. Please try again.' }, { status: 503 }));
  await renderPlantingWorkspace('/zones');
  await drawZone('warning');
  assert.equal(namedButton('Save Warning Zone').disabled, true);
  assert.match(document.querySelector('.save-form [role="alert"]').textContent, /Could not check the map boundary/);
  responses.set('/api/zones/validate-boundary', { valid: true });
  await click(namedButton('Try again'));
  await settle();
  assert.equal(namedButton('Save Warning Zone').disabled, false);
  assert.equal(requests.filter((path) => path === '/api/zones/validate-boundary').length, 2);
  assert.equal(mutations.length, 0);
});

test('eroded overlap confirmation cancels without a write, then saves the reviewed draft once', async () => {
  zoneFixtures();
  responses.set('/api/zones/validate-boundary', { valid: true });
  const saves = [];
  responses.set('/api/zones/eroded', (options) => {
    saves.push(options);
    return options.headers['X-Confirm-Seedling-Overlap'] === 'true'
      ? Response.json({ id: 8 })
      : Response.json({ detail: { code: 'seedling_overlap' } }, { status: 409 });
  });
  await renderPlantingWorkspace('/zones');
  await drawZone('eroded');
  await click(namedButton('Save Zone'));
  assert.equal(saves.length, 1);
  assert.equal(document.querySelector('.modal-title').textContent, 'Save eroded zone?');
  assert.equal(document.querySelector('.modal-body p').textContent,
    'This area contains recorded seedlings. Their records will be kept, but new planting and replanting here will be blocked.');
  await click([...document.querySelectorAll('.modal-actions button')].find((button) => button.textContent === 'Cancel'));
  assert.equal(saves.length, 1, 'Cancel makes no save request');
  assert.ok(document.querySelector('.save-form'), 'keep the unsaved draft for review');
  await click(namedButton('Save Zone'));
  await click([...document.querySelectorAll('.modal-actions button')].find((button) => button.textContent === 'Save zone'));
  assert.equal(saves.length, 3);
  assert.deepEqual(JSON.parse(saves[2].body), JSON.parse(saves[1].body));
  assert.equal(saves[2].headers['X-Confirm-Seedling-Overlap'], 'true');
  assert.equal(document.querySelector('.modal-title'), null);
  assert.equal(document.querySelector('.save-form'), null);
});

test('eroded vertex editing stays a draft, preserves holes, and patches only the chosen zone', async () => {
  const zones = zoneFixtures();
  const geometry = { type: 'MultiPolygon', coordinates: [zones.eroded.geometry.coordinates,
    [[[122.6253,10.78],[122.6254,10.78],[122.6254,10.7801],[122.6253,10.7801],[122.6253,10.78]],
      [[122.62532,10.78002],[122.62538,10.78002],[122.62538,10.78008],[122.62532,10.78008],[122.62532,10.78002]]]] };
  zones.eroded.geometry = geometry;
  responses.set('/api/zones/validate-boundary', { valid: true });
  const saves = [];
  responses.set('/api/zones/eroded/1', (options) => { saves.push(options); return Response.json({ id: 1 }); });
  await renderPlantingWorkspace('/zones');
  const openList = async () => click(document.querySelector('[data-guide-card="Eroded Zones"] .panel-card-header'));
  await openList();
  await click(document.querySelector('[aria-label="Edit Eroded zone 1"]'));
  assert.equal(document.querySelector('[data-guide-card="Draw Zone"] .panel-card-header').getAttribute('aria-expanded'), 'true');
  assert.equal(document.querySelectorAll('.zone-vertex-handle').length, 12);
  const handle = document.querySelector('.zone-vertex-handle');
  await act(async () => handle.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true })));
  await settle();
  assert.deepEqual(zones.eroded.geometry, geometry, 'the saved feature must remain unchanged');
  await click(namedButton('Cancel'));
  assert.equal(saves.length, 0);
  assert.equal(document.querySelectorAll('.zone-vertex-handle').length, 0);
  await openList();
  await click(document.querySelector('[aria-label="Edit Eroded zone 1"]'));
  await settle();
  await click(namedButton('Save Zone'));
  assert.equal(saves.length, 1);
  assert.equal(saves[0].method, 'PATCH');
  assert.deepEqual(JSON.parse(saves[0].body).geometry, geometry, 'preserve all parts and holes');
});

test('a failed confirmed erosion save keeps the draft and exposes its error', async () => {
  zoneFixtures();
  responses.set('/api/zones/validate-boundary', { valid: true });
  responses.set('/api/zones/eroded', (options) => options.headers['X-Confirm-Seedling-Overlap'] === 'true'
    ? Response.json({ detail: 'Could not save the zone. Please try again.' }, { status: 503 })
    : Response.json({ detail: { code: 'seedling_overlap' } }, { status: 409 }));
  await renderPlantingWorkspace('/zones');
  await drawZone('eroded');
  await click(namedButton('Save Zone'));
  await click([...document.querySelectorAll('.modal-actions button')].find((button) => button.textContent === 'Save zone'));
  assert.equal(document.querySelector('.modal-title'), null);
  assert.ok(document.querySelector('.save-form'));
  assert.match(document.querySelector('.save-form [role="alert"]').textContent, /entries are kept; please retry/i);
  assert.equal(namedButton('Save Zone').disabled, false);
});

test('cancelling a pending map-boundary check ignores its late response', async () => {
  zoneFixtures();
  let finishCheck;
  responses.set('/api/zones/validate-boundary', () => new Promise((resolve) => { finishCheck = resolve; }));
  await renderPlantingWorkspace('/zones');
  await drawZone('eroded');
  assert.equal(namedButton('Checking boundary...').disabled, true);
  assert.match(document.querySelector('.save-form').textContent, /Checking the zone boundary/);
  await click(namedButton('Cancel'));
  await act(async () => finishCheck(Response.json({ valid: true })));
  await settle();
  assert.equal(document.querySelector('.save-form'), null);
  assert.doesNotMatch(document.querySelector('[data-guide-panel="Zone Editor"]').textContent, /Ready to save|Checking the zone boundary/);
  assert.ok(namedButton('Draw New Eroded Zone'));
  assert.equal(mutations.length, 0);
});

test('a timed-out boundary check ends with a retryable error instead of enabling Save', async () => {
  zoneFixtures();
  responses.set('/api/zones/validate-boundary', (options) => new Promise((resolve, reject) => {
    options.signal.addEventListener('abort', () => reject(new window.DOMException('Timed out', 'AbortError')), { once: true });
  }));
  const originalTimeout = window.setTimeout;
  window.setTimeout = (callback, delay, ...args) => originalTimeout.call(window, callback, delay === 15000 ? 1 : delay, ...args);
  try {
    await renderPlantingWorkspace('/zones');
    await drawZone('eroded');
    assert.equal(namedButton('Save Zone').disabled, true);
    assert.match(document.querySelector('.save-form [role="alert"]').textContent, /Check your connection and try again/);
    assert.ok(namedButton('Try again'));
    assert.equal(mutations.length, 0);
  } finally {
    window.setTimeout = originalTimeout;
  }
});

test('project sites come from assignment and cannot be drawn or claimed separately in Zone Editor', async () => {
  zoneFixtures(); await renderPlantingWorkspace('/zones');
  const header=[...document.querySelectorAll('.panel-card-header')].find((item) => item.textContent.includes('Draw Zone'));
  if (header.getAttribute('aria-expanded') !== 'true') await click(header);
  assert.equal(namedButton('Project Site'),undefined);
  assert.equal(namedButton('Draw New Project Site'),undefined);
  assert.equal(document.getElementById('project-site-organization'),null);
  assert.ok(namedButton('Draw New Eroded Zone'));
  assert.equal(mutations.length,0);
});

test('an outline must begin from left to right and can then close back to the left', async () => {
  zoneFixtures();
  responses.set('/api/zones/validate-boundary', { valid: true });
  await renderPlantingWorkspace('/zones');
  await drawZone('eroded', { corners: [[10.7799, 122.6252], [10.7799, 122.6250], [10.7801, 122.6250], [10.7801, 122.6252]] });
  assert.equal(document.querySelector('.save-form'), null);
  assert.match(document.querySelector('[data-guide-panel="Zone Editor"] [role="alert"]').textContent, /Start on the left side and drag right/);
  assert.equal(requests.includes('/api/zones/validate-boundary'), false);
  await drawZone('eroded', { start: false });
  assert.equal(namedButton('Save Zone').disabled, false);
  assert.equal(requests.filter((path) => path === '/api/zones/validate-boundary').length, 1);
  assert.equal(mutations.length, 0);
});

function siteActionPlan(zones, overrides = {}) {
  return { site_id: 5, name: 'Community site', version: 'a'.repeat(64),
    counts: { living: 1, dead: 1, unplanted: 2, release: 3, historical_planted: 2, historical_dead: 1 },
    can_release: true, can_delete: false,
    release_reason: '', delete_reason: 'Delete is available only when all seedlings are recorded dead and no unplanted points remain.',
    before_geometry: zones.site.geometry, after_geometry: zones.eroded.geometry,
    points: [{ id:101, point_num:101, latitude:10.78, longitude:122.6251, status:'planted' }], ...overrides };
}

test('Project Sites search finds WVSU at the end and keeps each site’s map and actions distinct', async () => {
  const zones = zoneFixtures();
  const otherSites = Array.from({ length: 9 }, (_, index) => ({ ...zones.site, id: 30 + index,
    properties: { name: `Area ${index + 1}`, organization_name: `Volunteer team ${index + 1}`, organization_id: 30 + index } }));
  const fullName = { ...zones.site, id: 19, properties: { name: 'West Visayas State University', organization_name: 'West Visayas State University', organization_id: 28 } };
  const wvsu = { ...zones.site, id: 11, properties: { name: 'WVSU', organization_name: 'WVSU', organization_id: 10 } };
  useMapStore.setState({ projectSites: { type: 'FeatureCollection', features: [zones.site, ...otherSites, fullName, wvsu] } });
  responses.set('/api/project-sites/11/actions', siteActionPlan(zones, { site_id: 11, name: 'WVSU' }));
  await renderPlantingWorkspace('/zones');
  const card = document.querySelector('[data-guide-card="Project Sites"]');
  await click(card.querySelector('.panel-card-header'));
  assert.equal(card.querySelectorAll('.project-site-row').length, 12);
  assert.equal(card.querySelector('.panel-card-badge').textContent, '12');
  const input = card.querySelector('[aria-label="Search project sites"]');
  const search = async (value) => act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(input, value);
    input.dispatchEvent(new window.Event('input', { bubbles: true }));
  });
  await search(' WvsU ');
  assert.equal(card.querySelectorAll('.project-site-row').length, 1);
  assert.equal(card.querySelector('.zone-name').textContent, 'WVSU');
  assert.equal(card.querySelector('[role="status"]').textContent.trim(), '1 of 12 sites');
  const map = useMapStore.getState().mapInstance;
  assert.ok(boundaryFor(map, wvsu));
  assert.ok(boundaryFor(map, fullName), 'search must not remove other map areas');
  let focused;
  map.on('zonefocus', (event) => { focused = event.feature; });
  await click(card.querySelector('[aria-label="Show WVSU on map"]'));
  assert.equal(focused, wvsu);
  await click(card.querySelector('[aria-label="Actions for WVSU"]'));
  await settle();
  assert.ok(requests.includes('/api/project-sites/11/actions'));
  assert.equal(document.querySelector('.site-actions-menu').getAttribute('aria-label'), 'Actions for WVSU');
  await act(async () => document.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
  await search('state UNIVERSITY');
  assert.equal(card.querySelectorAll('.project-site-row').length, 1);
  assert.equal(card.querySelector('.zone-name').textContent, 'West Visayas State University');
  await search('volunteer team 2');
  assert.equal(card.querySelector('.zone-name').textContent, 'Area 2');
  await search('not a known site');
  assert.equal(card.querySelectorAll('.project-site-row').length, 0);
  assert.match(card.querySelector('[role="status"]').textContent, /No sites match/);
  await click(card.querySelector('[aria-label="Clear site search"]'));
  assert.equal(card.querySelectorAll('.project-site-row').length, 12);
  assert.equal(input.value, '');
  assert.equal(document.activeElement, input);
  assert.equal(useMapStore.getState().projectSites.features.length, 12);
  assert.equal(mutations.length, 0);
});

test('project-site dots show Release and guarded Delete without moving the map', async () => {
  const zones = zoneFixtures();
  responses.set('/api/project-sites/5/actions', siteActionPlan(zones));
  await renderPlantingWorkspace('/zones');
  const header = [...document.querySelectorAll('.panel-card-header')].find((item) => item.textContent.includes('Project Sites'));
  await click(header);
  const trigger = document.querySelector('[aria-label="Actions for Community site"]');
  assert.equal(document.querySelector('[aria-label="Delete Community site"]'), null);
  const map = useMapStore.getState().mapInstance;
  const center = map.getCenter();
  await click(trigger);
  await settle();
  assert.ok(map.getCenter().equals(center));
  const items = [...document.querySelectorAll('.site-actions-menu [role="menuitem"]')];
  assert.deepEqual(items.map((item) => item.textContent), ['Release area', 'Delete site']);
  assert.equal(items[0].getAttribute('aria-disabled'), 'false');
  assert.equal(items[1].getAttribute('aria-disabled'), 'true');
  assert.match(document.querySelector('.site-actions-menu').textContent, /all seedlings are recorded dead/);
  await click(items[1]);
  assert.equal(document.querySelector('[role="dialog"]'), null);
  await click(items[0]);
  const dialog = document.querySelector('[role="dialog"]');
  assert.equal(dialog.closest('.modal-backdrop').parentElement, document.body);
  assert.equal(document.querySelector('.planting-workspace-panel').contains(dialog), false);
  assert.equal(dialog.querySelector('.modal-title').textContent, 'Release area');
  assert.equal(dialog.querySelector('.site-action-name').textContent, 'Community site');
  assert.ok(dialog.querySelector('[aria-label="Close modal"]'));
  assert.match(dialog.textContent, /Site history stays saved/);
  assert.match(dialog.textContent, /2 planted · 1 recorded deaths/);
  assert.equal(dialog.querySelectorAll('.site-action-boundaries [role="img"]').length, 2);
  assert.equal(mutations.length, 0);
  await click([...dialog.querySelectorAll('button')].find((item) => item.textContent === 'Cancel'));
  assert.equal(document.querySelector('[role="dialog"]'), null);
  assert.equal(document.activeElement, trigger);
  assert.equal(mutations.length, 0);
  await click(trigger);
  await settle();
  await click(document.querySelector('.site-actions-menu [role="menuitem"]'));
  await click(document.querySelector('[role="dialog"] [aria-label="Close modal"]'));
  assert.equal(document.querySelector('[role="dialog"]'), null);
  assert.equal(document.activeElement, trigger);
  assert.equal(mutations.length, 0);
});

test('release confirms the preview version once and refreshes shared map data', async () => {
  const zones = zoneFixtures();
  responses.set('/api/project-sites/5/actions', siteActionPlan(zones));
  responses.set('/api/project-sites/5/release', { status:'released', id:5 });
  await renderPlantingWorkspace('/zones');
  await click([...document.querySelectorAll('.panel-card-header')].find((item) => item.textContent.includes('Project Sites')));
  await click(document.querySelector('[aria-label="Actions for Community site"]'));
  await settle();
  await click(document.querySelector('.site-actions-menu [role="menuitem"]'));
  const confirm = [...document.querySelector('[role="dialog"]').querySelectorAll('button')].find((item) => item.textContent === 'Release area');
  await click(confirm);
  assert.equal(mutations.filter((item) => item.path === '/api/project-sites/5/release').length, 1);
  assert.deepEqual(mutations.find((item) => item.path === '/api/project-sites/5/release').body, { version:'a'.repeat(64) });
  assert.equal(document.querySelector('[role="dialog"]'), null);
  assert.match(document.querySelector('[data-guide-panel="Zone Editor"]').textContent, /Planting history preserved/);
});

test('a changed release preview blocks confirmation until the updated area is reviewed', async () => {
  const zones = zoneFixtures();
  responses.set('/api/project-sites/5/actions', siteActionPlan(zones));
  responses.set('/api/project-sites/5/release', Response.json({ detail:'This site changed after the preview. Review the updated counts and boundary before confirming.' }, { status:409 }));
  await renderPlantingWorkspace('/zones');
  await click([...document.querySelectorAll('.panel-card-header')].find((item) => item.textContent.includes('Project Sites')));
  await click(document.querySelector('[aria-label="Actions for Community site"]'));
  await settle();
  await click(document.querySelector('.site-actions-menu [role="menuitem"]'));
  await click([...document.querySelector('[role="dialog"]').querySelectorAll('button')].find((item) => item.textContent === 'Release area'));
  assert.match(document.querySelector('[role="dialog"] [role="alert"]').textContent, /changed after the preview/);
  assert.equal([...document.querySelector('[role="dialog"]').querySelectorAll('button')].some((item) => item.textContent === 'Release area'), false);
  responses.set('/api/project-sites/5/actions', siteActionPlan(zones, { version:'b'.repeat(64) }));
  await click([...document.querySelector('[role="dialog"]').querySelectorAll('button')].find((item) => item.textContent === 'Review updated area'));
  await settle();
  assert.ok([...document.querySelector('[role="dialog"]').querySelectorAll('button')].some((item) => item.textContent === 'Release area'));
  assert.equal(mutations.length, 1);
});

test('all-dead sites offer Delete and send its reviewed version without losing history wording', async () => {
  const zones = zoneFixtures();
  responses.set('/api/project-sites/5/actions', siteActionPlan(zones, {
    can_delete:true, after_geometry:null,
    counts:{ living:0, dead:2, unplanted:0, release:2, historical_planted:2, historical_dead:2 },
  }));
  let deleted = 0;
  responses.set('/api/project-sites/5', (options, url) => {
    assert.equal(options.method, 'DELETE');
    assert.equal(url.searchParams.get('version'), 'a'.repeat(64));
    deleted += 1;
    return Response.json({ status:'deleted', id:5 });
  });
  await renderPlantingWorkspace('/zones');
  await click([...document.querySelectorAll('.panel-card-header')].find((item) => item.textContent.includes('Project Sites')));
  await click(document.querySelector('[aria-label="Actions for Community site"]'));
  await settle();
  const item = [...document.querySelectorAll('.site-actions-menu [role="menuitem"]')].find((element) => element.textContent === 'Delete site');
  assert.equal(item.getAttribute('aria-disabled'), 'false');
  await click(item);
  assert.match(document.querySelector('[role="dialog"]').textContent, /Planting records, deaths and survival rate are kept for this site/);
  await click([...document.querySelector('[role="dialog"]').querySelectorAll('button')].find((element) => element.textContent === 'Delete site'));
  assert.equal(deleted, 1);
  assert.equal(document.querySelector('[role="dialog"]'), null);
});

test('site menus support keyboard dismissal and failed checks cannot enable actions', async () => {
  zoneFixtures();
  responses.set('/api/project-sites/5/actions', Response.json({ detail:'Could not check this site. Please try again.' }, { status:503 }));
  await renderPlantingWorkspace('/zones');
  await click([...document.querySelectorAll('.panel-card-header')].find((item) => item.textContent.includes('Project Sites')));
  const trigger = document.querySelector('[aria-label="Actions for Community site"]');
  await click(trigger);
  await settle();
  assert.match(document.querySelector('.site-actions-menu [role="alert"]').textContent, /Could not check this site/);
  assert.ok([...document.querySelectorAll('.site-actions-menu [role="menuitem"]')].every((item) => item.getAttribute('aria-disabled') === 'true'));
  await act(async () => document.dispatchEvent(new window.KeyboardEvent('keydown', { key:'ArrowDown', bubbles:true })));
  assert.equal(document.activeElement.textContent, 'Delete site');
  await act(async () => document.dispatchEvent(new window.KeyboardEvent('keydown', { key:'Escape', bubbles:true })));
  assert.equal(document.querySelector('.site-actions-menu'), null);
  assert.ok(document.activeElement === trigger);
  assert.equal(mutations.length, 0);
});

async function settle(ms = 25) {
  await act(async () => { await new Promise((resolve) => window.setTimeout(resolve, ms)); });
}

async function render(Component, entry = '/page') {
  await act(async () => root.render(createElement(MemoryRouter, { initialEntries: [entry] },
    createElement(Navigation),
    createElement(RetainedRoutes, { paths: ['/page', '/away', '/map'] },
      createElement(Route, { path: '/page', element: createElement(Component) }),
      createElement(Route, { path: '/away', element: createElement('p', null, 'Another page') }),
      createElement(Route, { path: '/map', element: createElement('section', { id: 'planting-map-route' }, createElement(pages.get('MapAnalytics'))) }),
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

for (const [id, page] of [
  ['map', 'MapAnalytics'], ['zones', 'ErodedZoneEditor'],
  ['processing', 'ImageProcessing'], ['planters', 'PlanterManagement'],
  ['monitoring-map', 'MonitoringMapWorkspace'], ['monitoring', 'OrganizationMonitoring'],
]) {
  test(`${id} guide targets match the real screen without submitting records`, async () => {
    const tour = STAFF_TOURS.find((item) => item.id === id);
    responses.set('/api/monitoring/organizations', { organizations: [] });
    if (['map', 'zones', 'processing', 'planters'].includes(id)) await renderPlantingWorkspace(tour.path);
    else await render(pages.get(page));
    for (const step of tour.steps.filter((item) => item.path === tour.path)) {
      // This panel is intentionally absent until an organization is selected.
      if (id === 'planters' && step.target.endsWith('[data-guide-card="Organization code"]')) continue;
      const control = document.querySelector(step.target);
      assert.ok(control, `Missing guide target for ${id}: ${step.target}`);
      const header = control.closest('.panel-card')?.querySelector(':scope > .panel-card-header');
      if (header?.getAttribute('aria-expanded') === 'false') await click(header);
      assert.equal(control.closest('[inert], [aria-hidden="true"]'), null, step.target);
    }
    assert.equal(mutations.length, 0);
    assert.ok(!requests.some((path) => /\/access$/.test(path)), 'Guides must not reveal a code');
  });
}

test('Assign Points opens the form without the old sharing, report or project-site controls', async () => {
  await render(pages.get('PlanterManagement'));
  assert.equal(headers().some((header) => header.textContent.includes('Overview')), false);
  assert.match(expanded()[0].textContent, /Assign available points/);
  assert.equal(requests.includes('/api/planters/dashboard'), false);
  assert.equal(document.querySelector('.assignment-activity-report'), null);
  assert.equal(document.querySelector('#assign-site'), null);
  assert.doesNotMatch(document.body.textContent, /Field Share Link|Cloudflare|View Activity Report/);
  assert.equal(requests.some(path => path.startsWith('/api/share/')), false);
  assert.equal(document.querySelector('.par-backdrop'), null);
  assert.equal(mutations.length, 0);
});

function assignmentFixture(activities = [{ id:31,project_site_id:null }], locations = null) {
  responses.set('/api/planters/', [{ id:24,organization_id:22,status:'active',participant_count:10,registration_pending:false }]);
  responses.set('/api/planter-auth/organizations', { organizations:[{ id:22,name:'CICT' }] });
  responses.set('/api/planting-schedules', { schedules:activities.map((item) => ({ organization_id:22,
    appointment_type:'tree_planting',status:'confirmed',title:`Planting ${item.id}`,date:'2090-01-01',...item })) });
  responses.set('/api/planters/organizations/22/assignments', { project_site_id:77,assignment_ids:[1] });
  useMapStore.setState({ points:locations || [1,2].map((id) => ({ id,point_num:id,analysis_id:1,
    latitude:10.780,longitude:122.625+id*.00001,planting_status:'planned',species:id===1?'bungalon':'rhizophora' })) });
}
async function pickAssignmentPoint(id) {
  let marker;
  useMapStore.getState().mapInstance.eachLayer((layer) => { if (layer.options?.sitePointId===id) marker=layer; });
  assert.ok(marker,`Available point ${id}`);
  await act(async () => marker.fire('click',{ originalEvent:new window.MouseEvent('click') }));
  await settle();
}

  for (const scenario of ['none', 'requested', 'cancelled', 'completed', 'other organization', 'other activity type']) {
    test(`Assign Points blocks ordinary organizations without an eligible activity (${scenario})`, async () => {
      const activities = scenario === 'none' ? [] : [{ id: 31,
        ...(scenario === 'other organization' ? { organization_id: 99 }
          : scenario === 'other activity type' ? { appointment_type: 'clean_up_drive' }
            : { status: scenario }) }];
      assignmentFixture(activities);
      responses.set('/api/planters/', [{ id: 24, organization_id: 22, status: 'active',
        permanent_access: false, registration_pending: false }]);
      await renderPlantingWorkspace('/planters?organization_id=22');
      assert.ok(document.getElementById('assign-activity').disabled);
      assert.match(document.body.textContent, /Schedule and confirm a tree-planting activity for this organization/);
      assert.equal(document.querySelector('.planting-point-selector'), null);
      await act(async () => useMapStore.setState({ assignmentSelectedPointIds: [1] }));
      const assign = [...document.querySelectorAll('button')].find((button) => button.textContent === 'Assign 1 point');
      assert.ok(assign.disabled, 'retained or injected selections cannot bypass the activity requirement');
      await click(assign);
      assert.equal(document.querySelector('.modal-body'), null);
      assert.equal(mutations.length, 0);
    });
  }

for (const status of ['none', 'confirmed', 'in_progress', 'requested', 'cancelled', 'completed']) {
  test(`LGU assigns available points without a schedule link (${status})`, async () => {
    assignmentFixture(status === 'none' ? [] : [{ id: 31, status }]);
    responses.set('/api/planters/', [{ id: 24, organization_id: 22, status: 'active',
      permanent_access: true, registration_pending: false }]);
    await renderPlantingWorkspace('/planters?organization_id=22');
    assert.equal(document.getElementById('assign-activity'), null);
    assert.match(document.body.textContent, /LGU can assign available points without scheduling an activity/);
    await pickAssignmentPoint(1);
    await click([...document.querySelectorAll('button')].find((button) => button.textContent === 'Assign 1 point'));
    await click(document.querySelector('.modal-actions .btn-primary'));
    assert.deepEqual(mutations[0].body, { planting_point_ids: [1] });
  });
}

test('cancelling the selected activity disables assignment even with points already selected', async () => {
  assignmentFixture();
  await renderPlantingWorkspace('/planters?organization_id=22');
  await pickAssignmentPoint(1);
  responses.set('/api/planting-schedules', { schedules: [{ id: 31, organization_id: 22,
    appointment_type: 'tree_planting', status: 'cancelled', title: 'Cancelled planting' }] });
  await act(async () => window.dispatchEvent(new window.Event('mv:data-changed')));
  await settle();
  assert.equal(document.querySelector('.planting-point-selector'), null);
  assert.ok([...document.querySelectorAll('button')].find((button) => button.textContent === 'Assign 1 point').disabled);
  assert.equal(mutations.length, 0);
});

test('Assign Points starts empty and only selected map points become planting work', async () => {
  assignmentFixture(); await renderPlantingWorkspace('/planters?organization_id=22');
  assert.match(document.body.textContent,/2 available points/);
  assert.equal(document.getElementById('organization-point-count'),null);
  assert.deepEqual(useMapStore.getState().assignmentSelectedPointIds,[]);
  await pickAssignmentPoint(2);
  assert.deepEqual(useMapStore.getState().assignmentSelectedPointIds,[2]);
  await click([...document.querySelectorAll('button')].find((item) => item.textContent==='Assign 1 point'));
  assert.equal(mutations.length, 0);
  assert.match(document.querySelector('.modal-body').textContent, /1 point.*CICT.*Planting 31/);
  assert.equal(document.activeElement.textContent, 'Cancel');
  await click(document.querySelector('.modal-actions .btn-secondary'));
  assert.deepEqual(useMapStore.getState().assignmentSelectedPointIds, [2]);
  await click([...document.querySelectorAll('button')].find((item) => item.textContent==='Assign 1 point'));
  await click(document.querySelector('.modal-actions .btn-primary'));
  assert.deepEqual(mutations[0],{ path:'/api/planters/organizations/22/assignments',body:{ planting_point_ids:[2],planting_schedule_id:31 } });
});

test('Assign Points matches main-map point sizes through zoom and restores species colors after selection', async () => {
  assignmentFixture(undefined, [1, 2].map((id) => ({ id, point_num: id, analysis_id: 1,
    latitude: 10.780, longitude: 122.625 + id * .0001, planting_status: 'planned',
    species: id === 1 ? 'bungalon' : 'rhizophora' })));
  await renderPlantingWorkspace('/planters?organization_id=22');
  const map = useMapStore.getState().mapInstance;
  const layers = () => {
    const overlay = new Map(), base = new Map();
    map.eachLayer((layer) => {
      if (layer.options?.sitePointId) overlay.set(layer.options.sitePointId, layer);
      if (layer.options?.mgPointId) base.set(layer.options.mgPointId, layer);
    });
    return { overlay, base };
  };
  for (const zoom of [17, 19, 21, 22]) {
    await act(async () => map.setZoom(zoom, { animate: false }));
    const { overlay, base } = layers();
    for (const id of [1, 2]) {
      assert.equal(overlay.get(id).getRadius(), base.get(id).getRadius());
      assert.equal(overlay.get(id).options.weight, base.get(id).options.weight);
      assert.equal(overlay.get(id).options.fillColor, id === 1 ? '#16a34a' : '#db2777');
    }
  }
  await pickAssignmentPoint(2);
  await act(async () => map.setZoom(18, { animate: false }));
  const { overlay, base } = layers();
  assert.equal(overlay.get(2).options.fillColor, '#7c3aed');
  assert.equal(overlay.get(2).getRadius(), base.get(2).getRadius());
  await pickAssignmentPoint(2);
  assert.equal(layers().overlay.get(2).options.fillColor, '#db2777');
  assert.equal(mutations.length, 0);
});

test('the Planting Map starts with its overview and has no Next steps cards', async () => {
  await render(pages.get('MapAnalytics'));
  assert.equal(document.querySelector('.next-actions'), null);
  assert.match(expanded()[0].textContent, /Overview/);
});

test('selected mixed species are submitted together without a site or species override', async () => {
  assignmentFixture(); await renderPlantingWorkspace('/planters?organization_id=22');
  await pickAssignmentPoint(1); await pickAssignmentPoint(2);
  await click([...document.querySelectorAll('button')].find((item) => item.textContent==='Assign 2 points'));
  assert.equal(mutations.length, 0);
  await click(document.querySelector('.modal-actions .btn-primary'));
  assert.deepEqual(mutations[0].body,{ planting_point_ids:[1,2],planting_schedule_id:31 });
});

for (const pointerType of ['mouse', 'touch']) {
test(`Assign Points brushes only during a held primary stroke (${pointerType})`, async () => {
  const locations = [
    { id: 1, point_num: 1, analysis_id: 1, latitude: 10.78, longitude: 122.625, planting_status: 'planned', species: 'bungalon' },
    { id: 2, point_num: 2, analysis_id: 1, latitude: 10.78, longitude: 122.6257, planting_status: 'planned', species: 'rhizophora' },
  ];
  assignmentFixture(undefined, locations);
  await renderPlantingWorkspace('/planters?organization_id=22');
  const map = useMapStore.getState().mapInstance;
  await act(async () => map.setView([10.78, 122.62535], 20, { animate: false }));
  const container = map.getContainer();
  const captures = new Set();
  container.setPointerCapture = (id) => captures.add(id);
  container.hasPointerCapture = (id) => captures.has(id);
  container.releasePointerCapture = (id) => captures.delete(id);
  const selected = () => useMapStore.getState().assignmentSelectedPointIds;
  const tool = (label) => [...document.querySelectorAll('.planting-point-selector button')].find((item) => item.textContent === label);
  const pointer = async (type, longitude, buttons = 0, target = container, button = 0, pointerId = 1) => {
    const point = map.latLngToContainerPoint([10.78, longitude]);
    const event = new window.MouseEvent(type, { bubbles: true, cancelable: true,
      clientX: point.x, clientY: point.y, buttons, button });
    Object.defineProperties(event, { pointerType: { value: pointerType }, pointerId: { value: pointerId } });
    await act(async () => target.dispatchEvent(event));
    return event;
  };
  await click(tool('Brush select'));
  assert.match(document.querySelector('.planting-point-selector').textContent, /Hold the left mouse button and drag/);
  assert.equal(map.dragging.enabled(), false);
  await pointer('pointermove', locations[0].longitude);
  await pointer('pointermove', locations[1].longitude, 1);
  assert.deepEqual(selected(), [], 'hovering or entering with the button already held must not begin a stroke');
  await pointer('pointerdown', locations[0].longitude, 2, container, 2);
  await pointer('pointermove', locations[0].longitude, 2);
  assert.deepEqual(selected(), [], 'the secondary button must not paint');
  await pointer('pointerdown', locations[0].longitude, 1);
  assert.deepEqual(selected(), [1]);
  assert.ok(captures.has(1));
  assert.match(document.querySelector('.planting-point-selector').textContent, /Release to stop brushing/);
  await pointer('pointerup', locations[0].longitude, 0, window);
  await pointer('click', locations[0].longitude);
  await pointer('pointermove', locations[1].longitude);
  assert.deepEqual(selected(), [1], 'release outside the map and the compatibility click must not latch painting');
  assert.equal(captures.size, 0);
  await pointer('pointerdown', locations[1].longitude, 1);
  await pointer('pointermove', locations[0].longitude, 1);
  await pointer('pointerup', locations[0].longitude);
  assert.deepEqual(new Set(selected()), new Set([1, 2]), 'a new held drag may add more points');
  await click(tool('Brush erase'));
  await pointer('pointermove', locations[0].longitude);
  assert.deepEqual(new Set(selected()), new Set([1, 2]), 'hovering must not erase');
  await pointer('pointerdown', locations[0].longitude, 1);
  await pointer('pointerup', locations[0].longitude);
  await pointer('pointermove', locations[1].longitude);
  assert.deepEqual(selected(), [2], 'erasing stops on release');
  await click(tool('Clear'));
  await click(tool('Brush select'));
  for (const reason of ['blur', 'pointercancel', 'pointerleave', 'lostpointercapture', 'zoom', 'control']) {
    await pointer('pointerdown', 122.62535, 1);
    if (reason === 'blur') await act(async () => window.dispatchEvent(new window.Event('blur')));
    else if (reason === 'zoom') await act(async () => { map.fire('zoomstart'); map.fire('zoomend'); });
    else if (reason === 'control') await pointer('pointermove', locations[0].longitude, 1, container.querySelector('.leaflet-control-zoom-in'));
    else await pointer(reason, 122.62535, 0, reason === 'pointercancel' ? window : container);
    await pointer('pointermove', locations[0].longitude, 1);
    assert.deepEqual(selected(), [], `${reason} must end the stroke until a fresh press`);
    assert.equal(captures.size, 0);
  }
  if (pointerType === 'mouse') {
    await pointer('pointerdown', 122.62535, 1);
    await pointer('pointermove', locations[0].longitude, 0);
    await pointer('pointermove', locations[0].longitude, 1);
    assert.deepEqual(selected(), [], 'missing pointerup is recovered from the released-button state');
  }
  await pointer('pointerdown', 122.62535, 1);
  await pointer('pointerup', 122.62535, 0, window, 0, 2);
  await pointer('pointermove', locations[1].longitude, 1);
  assert.deepEqual(selected(), [2], 'a different pointer must not end the active stroke');
  await act(async () => document.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
  assert.equal(tool('Move / click').getAttribute('aria-pressed'), 'true');
  assert.equal(map.dragging.enabled(), true);
  assert.equal(captures.size, 0);
  assert.equal(document.querySelector('.seedling-selection-brush'), null);
  await pointer('pointermove', locations[0].longitude, 1);
  assert.deepEqual(selected(), [2]);
  assert.equal(mutations.length, 0, 'brushing selects points without assigning them');
});
}

test('point assignment requires a choice between multiple confirmed activities', async () => {
  assignmentFixture([{ id:31,project_site_id:null },{ id:32,project_site_id:null }]);
  await renderPlantingWorkspace('/planters?organization_id=22');
  assert.equal(document.querySelector('.planting-point-selector'),null);
  await act(async () => {
    const activity=document.getElementById('assign-activity');activity.value='32';
    activity.dispatchEvent(new window.Event('change',{ bubbles:true }));
  });
  await settle(); await pickAssignmentPoint(1);
  await click([...document.querySelectorAll('button')].find((item) => item.textContent==='Assign 1 point'));
  await click(document.querySelector('.modal-actions .btn-primary'));
  assert.equal(mutations[0].body.planting_schedule_id,32);
  assert.equal(mutations[0].body.site_zone_id,undefined);
});

test('leaving Assign Points removes its brush and returning retains the organization', async () => {
  assignmentFixture(); await renderPlantingWorkspace('/planters?organization_id=22');
  const map=useMapStore.getState().mapInstance;
  await pickAssignmentPoint(1);
  await click([...document.querySelectorAll('button')].find((item) => item.textContent==='Brush select'));
  assert.equal(map.dragging.enabled(),false);
  const container = map.getContainer(), captures = new Set();
  container.setPointerCapture = (id) => captures.add(id);
  container.hasPointerCapture = (id) => captures.has(id);
  container.releasePointerCapture = (id) => captures.delete(id);
  const bounds = container.getBoundingClientRect();
  const down = new window.MouseEvent('pointerdown', { bubbles: true, cancelable: true,
    button: 0, buttons: 1, clientX: bounds.left + 10, clientY: bounds.top + 10 });
  Object.defineProperties(down, { pointerType: { value: 'mouse' }, pointerId: { value: 7 } });
  await act(async () => container.dispatchEvent(down));
  assert.ok(captures.has(7), 'start a held stroke before leaving the retained tool');
  await click(toolLink('/map'));
  assert.equal(map.dragging.enabled(),true);
  assert.equal(captures.size,0);
  assert.equal(document.querySelector('.seedling-selection-brush'),null);
  let overlays=0;map.eachLayer((layer) => { if(layer.options?.sitePointId) overlays+=1; });
  assert.equal(overlays,0);
  await click(toolLink('/planters'));
  assert.equal(document.getElementById('assign-organization').value,'22');
  assert.equal(map.dragging.enabled(),true);
  assert.equal([...document.querySelectorAll('.planting-point-selector button')]
    .find((item) => item.textContent === 'Move / click').getAttribute('aria-pressed'), 'true');
  assert.deepEqual(useMapStore.getState().assignmentSelectedPointIds,[1]);
  assert.equal(mutations.length,0);
});

test('review-analysis links open history when there is no current result', async () => {
  await render(pages.get('ImageProcessing'), '/page?action=review');
  assert.ok(document.querySelector('.analysis-history-modal'));
  assert.match(document.querySelector('.analysis-history-modal').textContent, /Image Analysis History/);
  assert.equal(mutations.length, 0);
});

const savedImages = [
  { id: 52, image_name: 'Analysis 52', source_image_name: 'QJBJ.JPG', analyzed_at: '2026-10-07T08:00:00+08:00',
    can_delete: true, deletion_locked: false,
    plantable_area_m2: 175, canopy_area_m2: 90, canopy_coverage_pct: 30, hexagon_count: 50, result_preview_url: 'https://preview.example/QJBJ.webp' },
  { id: 51, image_name: 'Analysis 51', source_image_name: 'ITBH.JPG', analyzed_at: '2026-09-30T08:00:00+08:00',
    can_delete: true, deletion_locked: false,
    plantable_area_m2: 0, canopy_area_m2: 350, canopy_coverage_pct: 75, hexagon_count: 0, result_preview_url: null },
];

async function changeHistorySelect(id, value) {
  await act(async () => {
    const select = document.getElementById(id);
    select.value = value;
    select.dispatchEvent(new window.Event('change', { bubbles: true }));
  });
}

async function changeHistorySearch(value) {
  await act(async () => {
    const input = document.getElementById('analysis-history-search');
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(input, value);
    input.dispatchEvent(new window.Event('input', { bubbles: true }));
  });
}

const savedFootprint = { type: 'Polygon', coordinates: [[
  [122.629, 10.779], [122.630, 10.7791], [122.6301, 10.780], [122.6292, 10.7802], [122.629, 10.779],
]] };

function mockSavedReview() {
  responses.set('/api/analyses/', savedImages);
  responses.set('/api/analyses/52', { analysis_id: 52, uploaded_file_name: 'Analysis 52', source_image_name: 'QJBJ.JPG',
    metrics: { plantable_area_m2: 175, canopy_area_m2: 90, canopy_coverage_pct: 30 },
    map: { available: true, coordinates: [], analysis_footprint: savedFootprint },
    images: { visualization_preview_url: 'https://preview.example/QJBJ.webp' }, exports: {} });
}

function mockAreaCheck({ status = 'inside', count = 12, earlier = [savedImages[1]] } = {}) {
  responses.set('/api/analyses/preflight', { status, can_process: true, estimated_inside_pct: 60,
    latitude: 10.779, longitude: 122.629, footprint_calibrated: true,
    map: { available: true, analysis_footprint: savedFootprint } });
  responses.set('/api/analyses/area-context', { saved_point_count: count, analyses: earlier });
  const starts = [];
  useProcessingStore.setState({ startProcess: async parameters => {
    starts.push(parameters); useProcessingStore.setState({ processing: true });
  } });
  return starts;
}

async function uploadImage(name = 'QJBJ.JPG') {
  const input = document.querySelector('input[type="file"]');
  const file = new File(['fixture'], name, { type: 'image/jpeg' });
  Object.defineProperty(input, 'files', { configurable: true, value: [file] });
  await act(async () => input.dispatchEvent(new Event('change', { bubbles: true })));
  await settle();
}

const processButton = () => [...document.querySelectorAll('.process-action button')].find(button => /Run Analysis|Review Partial Coverage/.test(button.textContent));
const areaDialog = () => document.querySelector('.image-area-confirmation');

test('corrected photo footprint reaches the actual green Leaflet border during upload, results and saved review', async () => {
  mockAreaCheck({ count: 0, earlier: [] });
  const mapData = { available: true, analysis_footprint: savedFootprint,
    analysis_footprint_quality: 'metric_projected_image_corners',
    match: { success: true, photometric_refined: true, projection_rebuilt: true,
      projection_rotation_source: 'photometric_affine' } };
  responses.set('/api/analyses/preflight', { status: 'inside', can_process: true,
    latitude: 10.779, longitude: 122.629, footprint_calibrated: true, map: mapData });
  await renderPlantingWorkspace('/processing');
  const map = useMapStore.getState().mapInstance;
  const boundary = () => {
    const found = [];
    map.eachLayer((layer) => {
      if (String(layer.getTooltip?.()?.getContent()).startsWith('Projected photo boundary')) found.push(layer);
    });
    assert.equal(found.length, 1);
    return found[0];
  };
  await uploadImage('DJI_0312.JPG');
  assert.deepEqual(boundary().toGeoJSON().geometry, savedFootprint);
  assert.equal(boundary().options.color, '#16a34a');
  assert.match(boundary().getTooltip().getContent(), /approximate/);
  // The raw GPS centre is deliberately different from this skewed boundary.
  const result = { map: mapData, metadata: { image_center_lat: 10.778, image_center_lon: 122.627 },
    metrics: {}, images: {}, exports: {} };
  await act(async () => useProcessingStore.setState({ result, overlayOpen: false }));
  await settle();
  assert.deepEqual(boundary().toGeoJSON().geometry, savedFootprint);
  await act(async () => useMapStore.getState().setSavedAnalysisBoundary({ ...result, analysis_id: 77 }));
  assert.deepEqual(boundary().toGeoJSON().geometry, savedFootprint);
});

test('occupied-area warning lists existing points and images; Cancel stops and Continue starts once', async () => {
  const starts = mockAreaCheck();
  await render(pages.get('ImageProcessing'));
  await uploadImage();
  await click(processButton());
  assert.equal(starts.length, 0);
  assert.match(areaDialog().textContent, /12 saved planting points/);
  assert.match(areaDialog().textContent, /ITBH.JPG/);
  assert.equal(areaDialog().closest('.modal-backdrop').parentElement, document.body);
  assert.deepEqual(mutations.find(row => row.path === '/api/analyses/area-context').body.footprint, savedFootprint);
  await click(areaDialog().querySelector('.btn-secondary'));
  assert.equal(areaDialog(), null);
  assert.equal(starts.length, 0);
  responses.set('/api/analyses/area-context', { saved_point_count: 13, analyses: [savedImages[1]] });
  await click(processButton());
  assert.match(areaDialog().textContent, /13 saved planting points/);
  await click(areaDialog().querySelector('.btn-primary'));
  assert.equal(starts.length, 1);
  assert.equal(starts[0].file.name, 'QJBJ.JPG');
  assert.equal(starts[0].allow_partial_map_overlap, false);
  assert.equal(areaDialog(), null);
});

test('an empty area starts directly; earlier analysis with no saved points still asks for confirmation', async () => {
  const starts = mockAreaCheck({ count: 0 });
  await render(pages.get('ImageProcessing'));
  await uploadImage();
  await click(processButton());
  assert.match(areaDialog().textContent, /This area has already been analyzed/);
  assert.equal(starts.length, 0);
  await click(areaDialog().querySelector('.btn-secondary'));
  responses.set('/api/analyses/area-context', { saved_point_count: 0, analyses: [] });
  await click(processButton());
  assert.equal(areaDialog(), null);
  assert.equal(starts.length, 1);
});

test('repeat images explain zero new points and require explicit repeat approval', async () => {
  const starts = mockAreaCheck({ count: 0, earlier: [] });
  responses.set('/api/analyses/area-context', { saved_point_count: 0, analyses: [], repeat_analyses: [savedImages[0]] });
  await render(pages.get('ImageProcessing'));
  await uploadImage();
  await click(processButton());
  assert.match(areaDialog().textContent, /This image has already been analyzed/);
  assert.match(areaDialog().textContent, /No planting points will be added/);
  assert.equal(starts.length, 0);
  await click(areaDialog().querySelector('.btn-secondary'));
  assert.equal(starts.length, 0);
  await click(processButton());
  await click(areaDialog().querySelector('.btn-primary'));
  assert.equal(starts.length, 1);
  assert.equal(starts[0].allow_repeat_image_analysis, true);
});

test('repeat results show zero new planting points and allow saving their analysis history', async () => {
  useProcessingStore.setState({ result: { uploaded_file_name:'QJBJ.JPG', can_save:true,
    repeat_image:{ points_not_added:true, analyses:[savedImages[0]] },
    metrics:{ hexagon_count:0 }, map:{ available:true,coordinates:[] }, images:{}, exports:{} }, overlayOpen:true });
  await render(pages.get('ImageProcessing'));
  const overlay = document.querySelector('.rs-backdrop');
  assert.match(overlay.textContent, /New Planting Points0/);
  assert.match(overlay.querySelector('.rs-repeat-notice').textContent, /No planting points will be added/);
  assert.equal(overlay.querySelector('.rs-save-btn').disabled, false);
});

test('failed area checks explain how to retry and do not start analysis', async () => {
  const starts = mockAreaCheck();
  responses.set('/api/analyses/area-context', Response.json({ detail: 'Could not check existing planting data. Please try Run Analysis again.' }, { status: 503 }));
  await render(pages.get('ImageProcessing'));
  await uploadImage();
  await click(processButton());
  assert.equal(starts.length, 0);
  assert.equal(areaDialog(), null);
  assert.match(document.querySelector('.process-error[role="alert"]').textContent, /try Run Analysis again/);
  responses.set('/api/analyses/area-context', { saved_point_count: 5, analyses: [] });
  await click(processButton());
  assert.equal(document.querySelector('.process-error[role="alert"]'), null);
  assert.match(areaDialog().textContent, /5 saved planting points/);
});

test('partial map approval and occupied-area approval are both required', async () => {
  const starts = mockAreaCheck({ status: 'partial' });
  await render(pages.get('ImageProcessing'));
  await uploadImage();
  assert.equal(starts.length, 0);
  const partialDialog = [...document.querySelectorAll('.modal-card')].find(dialog => dialog.textContent.includes('Part of this image is outside the map'));
  await click(partialDialog.querySelector('.btn-primary'));
  assert.equal(starts.length, 0);
  await click(processButton());
  assert.ok(areaDialog());
  assert.equal(starts.length, 0);
  await click(areaDialog().querySelector('.btn-primary'));
  assert.equal(starts.length, 1);
  assert.equal(starts[0].allow_partial_map_overlap, true);
});

test('failed upload area checks can retry the selected image without starting analysis', async () => {
  const starts = mockAreaCheck();
  const valid = responses.get('/api/analyses/preflight');
  responses.set('/api/analyses/preflight', Response.json({ detail: 'Could not check existing planting data. Use Retry location check to try again.' }, { status: 503 }));
  await render(pages.get('ImageProcessing'));
  await uploadImage();
  assert.equal(processButton().disabled, true);
  assert.equal(starts.length, 0);
  responses.set('/api/analyses/preflight', valid);
  await click([...document.querySelectorAll('button')].find(button => button.textContent === 'Retry location check'));
  assert.equal(processButton().disabled, false);
  await click(processButton());
  assert.ok(areaDialog());
  assert.equal(starts.length, 0);
});

test('replacing an image cancels an in-flight area check and ignores its stale response', async () => {
  const starts = mockAreaCheck();
  let finish;
  responses.set('/api/analyses/area-context', () => new Promise(resolve => { finish = resolve; }));
  await render(pages.get('ImageProcessing'));
  await uploadImage();
  await click(processButton());
  await uploadImage('OTHER.JPG');
  await act(async () => finish(Response.json({ saved_point_count: 0, analyses: [] })));
  await settle();
  assert.equal(starts.length, 0);
  assert.equal(areaDialog(), null);
  responses.set('/api/analyses/area-context', { saved_point_count: 3, analyses: [] });
  await click(processButton());
  await click(areaDialog().querySelector('.btn-primary'));
  assert.equal(starts[0].file.name, 'OTHER.JPG');
});

test('history omits earlier-area sections and opens saved results from their cards', async () => {
  mockSavedReview();
  responses.set('/api/analyses/', [
    { ...savedImages[0], area_history_available: true, previous_analyses: [savedImages[1]] },
    { ...savedImages[1], area_history_available: false, previous_analyses: [] },
  ]);
  responses.set('/api/analyses/51', { analysis_id: 51, uploaded_file_name: 'Analysis 51',
    source_image_name: 'ITBH.JPG', metrics: {}, map: { available: false }, images: {}, exports: {} });
  await render(pages.get('ImageProcessing'), '/page?action=review');
  const cards = document.querySelectorAll('.analytics-history-item');
  assert.equal(document.querySelector('.analysis-history-area'), null);
  assert.doesNotMatch(document.querySelector('.analytics-history-list').textContent, /Earlier analyses in this area|Area history unavailable/);
  assert.match(cards[1].textContent, /ITBH.JPG/);
  assert.equal(document.querySelector('button button'), null);
  await click(cards[1].querySelector('.analysis-history-open'));
  assert.match(document.querySelector('.rs-backdrop').textContent, /Analysis 51/);
  assert.equal(requests.includes('/api/analyses/52'), false);
});

test('history filters combine, display result previews and open the matching saved analysis', async () => {
  mockSavedReview();
  await render(pages.get('ImageProcessing'), '/page?action=review');
  const list = () => document.querySelector('.analysis-history-modal .analytics-history-list');
  assert.equal(list().children.length, 2);
  assert.equal(list().querySelector('img').getAttribute('alt'), 'Detection result for QJBJ.JPG');
  assert.match(list().textContent, /No saved preview/);
  assert.equal(document.querySelectorAll('.analysis-history-filters select').length, 2);
  assert.equal(document.getElementById('analysis-history-plantable'), null);
  assert.equal(document.getElementById('analysis-history-canopy'), null);
  await changeHistorySelect('analysis-history-date', '2026-10');
  await changeHistorySelect('analysis-history-sort', 'canopy-desc');
  assert.equal(list().children.length, 1);
  assert.match(list().textContent, /QJBJ.JPG/);
  assert.match(document.querySelector('.analysis-history-toolbar').textContent, /1 of 2 saved analyses/);
  await click(list().querySelector('.analysis-history-open'));
  assert.equal(document.querySelector('.analysis-history-modal'), null);
  assert.ok(requests.includes('/api/analyses/52'));
  assert.ok(!requests.includes('/api/analyses/51'));
  assert.match(document.querySelector('.rs-backdrop').textContent, /Analysis 52/);
});

for (const action of ['X', 'Back to history', 'Browser Back', 'Escape']) {
  test(`${action} returns a saved review to history with the selected date and sort order`, async () => {
    mockSavedReview();
    await render(pages.get('ImageProcessing'), '/page?action=review');
    await changeHistorySelect('analysis-history-date', '2026-10');
    await changeHistorySelect('analysis-history-sort', 'plantable-desc');
    await click(document.querySelector('.analysis-history-open'));
    assert.equal(document.querySelector('.analysis-history-modal'), null);
    assert.ok(document.querySelector('.rs-backdrop'));
    if (action === 'X') await click(document.querySelector('.rs-close'));
    else if (action === 'Back to history') await click(document.querySelector('.rs-history-navigation button'));
    else if (action === 'Browser Back') await click(document.querySelector('[data-nav="browser-back"]'));
    else {
      await act(async () => window.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape' })));
      await settle();
    }
    assert.equal(document.querySelector('.rs-backdrop'), null);
    assert.ok(document.querySelector('.analysis-history-modal'));
    assert.equal(document.getElementById('analysis-history-date').value, '2026-10');
    assert.equal(document.getElementById('analysis-history-sort').value, 'plantable-desc');
    assert.equal(document.querySelectorAll('.analytics-history-item').length, 1);
    assert.match(document.querySelector('.analytics-history-list').textContent, /QJBJ.JPG/);
    await click([...document.querySelectorAll('.analysis-history-modal button')].find(button => button.textContent === 'Close history'));
    assert.equal(document.querySelector('.analysis-history-modal'), null);
  });
}

for (const source of ['review link', 'history button after visiting the map']) {
test(`View area in map stays in Analyze Image with the saved footprint from the ${source}`, async () => {
  mockSavedReview();
  const existingPoints = [{ id: 900, analysis_id: 10, latitude: 10.8, longitude: 122.6, planting_status: 'planted' }];
  useMapStore.setState({ points: existingPoints });
  if (source === 'review link') {
    await render(pages.get('ImageProcessing'), '/page?action=review');
  } else {
    await render(pages.get('ImageProcessing'), '/map');
    await click(document.querySelector('[data-nav="back"]'));
    await click(document.querySelector('.analysis-history-trigger'));
  }
  await changeHistorySelect('analysis-history-date', '2026-10');
  await changeHistorySelect('analysis-history-sort', 'canopy-desc');
  await click(document.querySelector('.analysis-history-open'));
  const viewMap = document.querySelector('.rs-view-map');
  assert.equal(viewMap.textContent.trim(), 'View area in map');
  assert.equal(viewMap.disabled, false);
  await click(viewMap);
  assert.equal(document.querySelector('[data-current-path]').textContent, '/page');
  assert.equal(document.querySelector('.rs-backdrop'), null);
  assert.equal(Boolean(document.querySelector('.analysis-history-modal')), false, 'View area in map must close history rather than reopen it over the map');
  const context = useMapStore.getState().currentAnalysis;
  assert.equal(context.analysis_id, 52);
  assert.equal(context.saved, true);
  assert.equal(context.preserveMapView, false);
  assert.deepEqual(context.map.analysis_footprint, savedFootprint);
  assert.equal(context.map.analysis_overlay, undefined);
  assert.equal(context.map.safe_points_geojson, undefined);
  assert.deepEqual(useMapStore.getState().points, existingPoints);
  assert.equal(mutations.length, 0);
  await click(document.querySelector('[data-nav="browser-back"]'));
  const history = document.querySelector('.analysis-history-modal');
  assert.ok(history);
  assert.notEqual(history.closest('.modal-backdrop').style.display, 'none');
  assert.equal(document.getElementById('analysis-history-date').value, '2026-10');
  assert.equal(document.getElementById('analysis-history-sort').value, 'canopy-desc');
});
}

test('completed analysis removes shortcut cards while keeping the result review button', async () => {
  useProcessingStore.setState({ result: { analysis_id: 69, saved: true, uploaded_file_name: 'Analysis 35', metrics: {}, map: {}, exports: {} } });
  await render(pages.get('ImageProcessing'));
  assert.equal(document.querySelector('.next-actions'), null);
  assert.ok([...document.querySelectorAll('.process-action button')].some(button => button.textContent.includes('Review analysis')));
});

test('a saved analysis without map coordinates explains why View area in map is unavailable', async () => {
  mockSavedReview();
  const detail = responses.get('/api/analyses/52');
  responses.set('/api/analyses/52', { ...detail, map: { available: false, coordinates: [] } });
  await render(pages.get('ImageProcessing'), '/page?action=review');
  await click(document.querySelector('.analysis-history-open'));
  const viewMap = document.querySelector('.rs-view-map');
  assert.equal(viewMap.disabled, true);
  assert.match(document.getElementById(viewMap.getAttribute('aria-describedby')).textContent, /No map location was saved/);
  await click(viewMap);
  assert.ok(document.querySelector('.rs-backdrop'));
});

test('filtered history preserves delete confirmation and filters when cancellation returns to history', async () => {
  responses.set('/api/analyses/', savedImages);
  await render(pages.get('ImageProcessing'), '/page?action=review');
  await changeHistorySelect('analysis-history-date', '2026-09');
  await click(document.querySelector('.analytics-history-delete'));
  assert.equal(document.querySelector('.analysis-history-modal'), null);
  assert.match(document.querySelector('.modal-card-danger').textContent, /Delete "Analysis 51"/);
  await click([...document.querySelectorAll('.modal-card-danger button')].find(button => button.textContent === 'Cancel'));
  assert.equal(document.getElementById('analysis-history-date').value, '2026-09');
  assert.match(document.querySelector('.analytics-history-list').textContent, /ITBH.JPG/);
  assert.doesNotMatch(document.querySelector('.analytics-history-list').textContent, /QJBJ.JPG/);
  assert.equal(mutations.length, 0);
});

test('field-record analyses show a lock instead of Delete and remain reviewable', async () => {
  mockSavedReview();
  responses.set('/api/analyses/', [
    { ...savedImages[0], can_delete: false, deletion_locked: true, delete_lock_reason: 'This analysis has field records and cannot be deleted.' },
    savedImages[1],
  ]);
  await render(pages.get('ImageProcessing'), '/page?action=review');
  const cards = document.querySelectorAll('.analytics-history-item');
  assert.equal(cards[0].querySelector('.analytics-history-delete'), null);
  assert.match(cards[0].querySelector('.analysis-history-lock').textContent, /Locked—contains field records/);
  const open = cards[0].querySelector('.analysis-history-open');
  assert.equal(document.getElementById(open.getAttribute('aria-describedby')).className, 'analysis-history-lock');
  assert.ok(cards[1].querySelector('.analytics-history-delete'));
  await click(open);
  assert.ok(document.querySelector('.rs-backdrop'));
  assert.equal(mutations.length, 0);
});

test('history without a verified deletion policy requires refresh instead of offering Delete', async () => {
  responses.set('/api/analyses/', [{ ...savedImages[0], can_delete: undefined, deletion_locked: undefined }]);
  await render(pages.get('ImageProcessing'), '/page?action=review');
  assert.equal(document.querySelector('.analytics-history-delete'), null);
  assert.match(document.querySelector('.analysis-history-lock').textContent, /Refresh to check deletion availability/);
  assert.equal(mutations.length, 0);
});

test('an analysis used after history loaded rejects stale deletion and returns as locked', async () => {
  mockSavedReview();
  responses.set('/api/analyses/52', options => {
    assert.equal(options.method, 'DELETE');
    responses.set('/api/analyses/', [{ ...savedImages[0], can_delete: false, deletion_locked: true }]);
    return Response.json({ detail: 'This analysis contains assigned points or field records and cannot be deleted.' }, { status: 409 });
  });
  await render(pages.get('ImageProcessing'), '/page?action=review');
  await click(document.querySelector('.analytics-history-delete'));
  await click([...document.querySelectorAll('.modal-card-danger button')].find(button => button.textContent === 'Delete analysis'));
  assert.equal(document.querySelector('.modal-card-danger'), null);
  assert.ok(document.querySelector('.analysis-history-modal'));
  assert.equal(document.querySelector('.analytics-history-delete'), null);
  assert.match(document.querySelector('.analysis-history-modal').textContent, /cannot be deleted/);
  assert.equal(mutations.length, 0); // DELETE is tracked separately by this harness.
  assert.equal(requests.filter(path => path === '/api/analyses/52').length, 1);
});

test('history shows an empty filtered state, resets filters and handles a broken thumbnail', async () => {
  responses.set('/api/analyses/', savedImages);
  await render(pages.get('ImageProcessing'), '/page?action=review');
  await changeHistorySearch('no-such-image');
  assert.match(document.querySelector('.analytics-history-list').textContent, /No analyses match these filters/);
  await click([...document.querySelectorAll('.analysis-history-toolbar button')].find(button => button.textContent === 'Reset filters'));
  assert.equal(document.querySelectorAll('.analytics-history-item').length, 2);
  await act(async () => document.querySelector('.analysis-history-thumbnail img').dispatchEvent(new window.Event('error')));
  assert.match(document.querySelector('.analysis-history-thumbnail').textContent, /Preview unavailable/);
  assert.equal(document.querySelector('.analysis-history-open').disabled, false);
});

test('monitoring review keeps the draft on Back and confirms with one busy button', async () => {
  responses.set('/api/monitoring/organizations', { organizations: [
    { id: 1, name: 'Due organization', total_planted: 100, monitoring_available: true, next_monitoring_date: manilaDateKey() },
  ] });
  responses.set('/api/monitoring/organizations/1/visit-context', { organization_id: 1, monitored_date: manilaDateKey(),
    total_planted: 100, previous_dead_count: 0, alive_before_count: 100, baseline_record_id: null,
    latest_record: { health_status: 'good', actions_taken: 'Checked living seedlings and cleared debris.' } });
  responses.set('/api/monitoring/organizations/1/planting-locations', { points: [] });
  let finishSave;
  responses.set('/api/monitoring/organization-records', () => new Promise(resolve => { finishSave = resolve; }));
  await render(pages.get('OrganizationMonitoring'));
  await click(document.querySelector('.org-monitoring-organization-card'));
  await settle();
  assert.equal(document.querySelector('.org-monitoring-submit').disabled, false);
  await act(async () => document.querySelector('.org-monitoring-form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true })));
  assert.equal(mutations.length, 0);
  assert.equal(document.querySelector('.modal-title').textContent, 'Save monitoring visit?');
  assert.match(document.querySelector('.monitoring-save-review').textContent, /Due organization.*0 newly dead/);
  assert.equal(document.activeElement.textContent, 'Back');
  await click(document.querySelector('.modal-actions .btn-secondary'));
  assert.equal(document.querySelector('#actions-taken').value, 'Checked living seedlings and cleared debris.');
  await act(async () => document.querySelector('.org-monitoring-form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true })));
  const confirm = document.querySelector('.modal-actions .btn-primary');
  await click(confirm);
  assert.equal(confirm.disabled, true);
  assert.equal(document.querySelectorAll('.modal-spinner').length, 1);
  assert.equal(document.querySelector('.org-monitoring-form'), null);
  assert.equal(mutations.length, 1);
  await click(confirm);
  assert.equal(mutations.length, 1);
  await act(async () => finishSave(Response.json({ organization_name: 'Due organization' })));
  await settle();
  assert.equal(document.querySelector('.modal-spinner'), null);
});

test('death-location review keeps notes on Back and preserves the draft after a failed save', async () => {
  const { default: Editor } = await server.ssrLoadModule('/src/components/DeathLocationEditor.jsx');
  responses.set('/api/monitoring/organizations/1/planting-locations', { points: [] });
  const writes = [];
  let saved = 0;
  responses.set('/api/monitoring/organization-records/7/death-locations', (options) => {
    writes.push(JSON.parse(options.body));
    return Response.json({ detail: 'Unable to save' }, { status: 503 });
  });
  await act(async () => root.render(createElement(Editor, { record: { id: 7, organization_id: 1,
    monitored_at: '2026-10-01', reported_dead_count: 2, dead_planting_event_ids: [10], location_version: 4 },
    onClose: () => {}, onSaved: () => { saved++; } })));
  await settle();
  const note = document.querySelector('[name="correction_note"]');
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLTextAreaElement.prototype, 'value').set.call(note, 'Checked the mapped location with field notes.');
    note.dispatchEvent(new window.Event('input', { bubbles: true }));
  });
  const review = async () => act(async () => document.querySelector('form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true })));
  await review();
  assert.equal(writes.length, 0);
  assert.equal(document.querySelector('.modal-title').textContent, 'Save death locations?');
  assert.match(document.querySelector('.modal-body').textContent, /2 reported dead.*1 located.*1 still to locate/);
  await click(document.querySelector('.modal-actions .btn-secondary'));
  assert.equal(document.querySelector('[name="correction_note"]').value, 'Checked the mapped location with field notes.');
  await review(); await click(document.querySelector('.modal-actions .btn-primary'));
  assert.equal(writes.length, 1);
  assert.equal(saved, 0);
  assert.deepEqual(writes[0], { dead_planting_event_ids: [10], expected_version: 4,
    correction_note: 'Checked the mapped location with field notes.' });
  assert.equal(document.querySelector('.modal-title').textContent, 'Identify or correct death locations');
  assert.match(document.querySelector('[role="alert"]').textContent, /entries are kept; please retry/);
  assert.equal(document.querySelector('[name="correction_note"]').value, writes[0].correction_note);
});

test('monitoring tabs separate today and overdue visits, hide locked organizations and support keyboard navigation', async () => {
  const today = manilaDateKey();
  const yesterday = manilaDateKey(new Date(Date.now() - 86_400_000));
  responses.set('/api/monitoring/organizations', { organizations: [
    { id: 1, name: 'Due organization', total_planted: 100, monitoring_available: true, next_monitoring_date: today },
    { id: 2, name: 'Upcoming organization', total_planted: 100, monitoring_available: false, next_monitoring_date: '2099-10-20' },
    { id: 3, name: 'Late organization', total_planted: 100, monitoring_available: true, next_monitoring_date: yesterday },
    { id: 4, name: 'Empty organization', total_planted: 0, monitoring_available: false },
  ] });
  await render(pages.get('OrganizationMonitoring'));
  const list = () => document.querySelector('.org-monitoring-organization-list');
  assert.match(list().textContent, /Due organization/);
  assert.doesNotMatch(list().textContent, /Upcoming organization|Late organization|Empty organization|Locked/);
  assert.equal(document.getElementById('monitoring-tab-today').getAttribute('aria-selected'), 'true');
  assert.equal(document.querySelector('.monitoring-due-filter'), null);
  await click(document.getElementById('monitoring-tab-overdue'));
  assert.match(list().textContent, /Late organization/);
  assert.match(list().textContent, /1 day overdue/);
  assert.doesNotMatch(list().textContent, /Due organization|Upcoming organization/);
  await act(async () => document.getElementById('monitoring-tab-overdue').dispatchEvent(new window.KeyboardEvent('keydown', { key: 'ArrowLeft', bubbles: true })));
  assert.equal(document.getElementById('monitoring-tab-today').getAttribute('aria-selected'), 'true');
  assert.equal(document.activeElement.id, 'monitoring-tab-today');
  assert.equal(mutations.length, 0);
});

test('existing dashboard due links open overdue work when no visits are due today', async () => {
  responses.set('/api/monitoring/organizations', { organizations: [
    { id: 1, name: 'Overdue owner', total_planted: 100, monitoring_available: true, next_monitoring_date: manilaDateKey(new Date(Date.now() - 86_400_000)) },
  ] });
  await render(pages.get('OrganizationMonitoring'), '/page?filter=due');
  assert.equal(document.getElementById('monitoring-tab-overdue').getAttribute('aria-selected'), 'true');
  assert.match(document.querySelector('#monitoring-visit-queue').textContent, /Overdue owner/);
});

test('site follow-up links show only the requested site and exclude its upcoming owner from visit tabs', async () => {
  responses.set('/api/monitoring/organizations', { organizations: [
    { id: 1, name: 'Other organization', total_planted: 100, monitoring_available: true, next_monitoring_date: manilaDateKey() },
    { id: 2, name: 'Site owner', total_planted: 100, monitoring_available: false, next_monitoring_date: '2099-10-20' },
  ] });
  responses.set('/api/project-sites', { features: [
    { id: 12, properties: { name: 'North planting site', organization_id: 2, organization_name: 'Site owner' } },
    { id: 13, properties: { name: 'Other site', organization_id: 1, organization_name: 'Other organization' } },
  ] });
  responses.set('/api/project-sites/12', { type: 'Feature', id: 12,
    properties: { name: 'North planting site', organization_id: 2, organization_name: 'Site owner' } });
  await render(pages.get('OrganizationMonitoring'), '/page?project_site_id=12');
  const context = document.querySelector('.monitoring-site-context');
  assert.match(context.textContent, /Follow-up for North planting site/);
  assert.match(context.textContent, /Monitoring visits cover all of its planting sites/);
  const list = () => document.querySelector('.org-monitoring-organization-list');
  assert.doesNotMatch(list().textContent, /Site owner|Other organization/);
  assert.match(document.querySelector('#monitoring-visit-queue').textContent, /No monitoring visits due today/);
  assert.equal(context.querySelector('a').getAttribute('href'), '/map?project_site_id=12&focus=site_points');
  const sites = () => document.querySelector('.monitoring-sites-section');
  assert.match(sites().textContent, /Site owner|North planting site/);
  assert.doesNotMatch(sites().textContent, /Other organization|Other site/);
  assert.equal(sites().querySelector('.panel-card-header'), null);
  assert.equal(sites().closest('.modal-backdrop'), null);
  await click([...context.querySelectorAll('button')].find((button) => button.textContent === 'Show all organizations'));
  assert.equal(document.querySelector('.monitoring-site-context'), null);
  assert.match(list().textContent, /Other organization/);
  assert.doesNotMatch(list().textContent, /Site owner/);
  assert.match(sites().textContent, /Other organization/);
  assert.equal(mutations.length, 0);
});

test('failed site lookups never show unrelated organizations and can be retried', async () => {
  responses.set('/api/monitoring/organizations', { organizations: [
    { id: 1, name: 'Other organization', total_planted: 100, monitoring_available: true, next_monitoring_date: manilaDateKey() },
  ] });
  responses.set('/api/project-sites/12', new Response(JSON.stringify({ detail: 'Project site not found.' }), { status: 404 }));
  await render(pages.get('OrganizationMonitoring'), '/page?project_site_id=12');
  assert.match(document.querySelector('.monitoring-site-context [role="alert"]').textContent, /Project site not found/);
  assert.doesNotMatch(document.querySelector('.org-monitoring-organization-list').textContent, /Other organization/);
  assert.ok([...document.querySelectorAll('.org-monitoring-kpis strong')].every((element) => element.textContent === '—'));
  assert.doesNotMatch(document.querySelector('.monitoring-sites-section').textContent, /Other organization/);
  responses.set('/api/project-sites/12', { type: 'Feature', id: 12,
    properties: { name: 'Restored site', organization_id: 1, organization_name: 'Other organization' } });
  await click([...document.querySelectorAll('.monitoring-site-context button')].find((button) => button.textContent === 'Retry'));
  assert.match(document.querySelector('.monitoring-site-context').textContent, /Restored site/);
  assert.match(document.querySelector('.org-monitoring-organization-list').textContent, /Other organization/);
  assert.equal(mutations.length, 0);
});

test('Sites to Monitor shows each site’s map area, never requests analysis pictures, and keeps five sites per page', async () => {
  responses.set('/api/monitoring/organizations', { organizations: [
    { id: 1, name: 'Site owner', total_planted: 100, alive_seedlings: 90, monitoring_available: true, next_monitoring_date: manilaDateKey() },
  ] });
  const sites = Array.from({ length: 12 }, (_, index) => {
    const lon = 122.624 + index * 0.00025;
    return { id: index + 1,
      properties: { name: `Planting area ${index + 1}`, organization_id: 1, organization_name: 'Site owner', point_count: 20 },
      geometry: index === 1 ? null : { type: 'Polygon', coordinates: [[
        [lon, 10.780], [lon + 0.00015, 10.780], [lon + 0.00015, 10.78015], [lon, 10.78015], [lon, 10.780],
      ]] },
    };
  });
  responses.set('/api/project-sites', { features: sites });
  responses.set('/api/planters/map-points', [
    { id: 11, source_project_site_id: 1, latitude: 10.78004, longitude: 122.62403, species: 'rhizophora' },
    { id: 12, source_project_site_id: 1, latitude: 10.78007, longitude: 122.62405, planting_status: 'planted' },
    { id: 13, source_site_id: 1, latitude: 10.78009, longitude: 122.62407, death_at: '2026-10-12' },
    { id: 14, source_project_site_id: 1, latitude: 10.78011, longitude: 122.62409, eroded_unavailable: true },
    { id: 15, source_project_site_id: 1, latitude: 10.78012, longitude: 122.62411, assignment_status: 'pending' },
    { id: '11', source_project_site_id: '1', latitude: 10.78004, longitude: 122.62403 },
    { id: 16, source_project_site_id: 1, latitude: 10.78007, longitude: 122.62405, deleted_at: '2026-10-12' },
    { id: 17, source_project_site_id: 6, source_organization_id: 1, latitude: 10.78007, longitude: 122.6253, planting_status: 'planted' },
    { id: 18, source_organization_id: 1, latitude: 10.78007, longitude: 122.62405 },
  ]);
  responses.set('/api/analyses/', [{ id: 50, project_site_id: 1, analyzed_at: '2026-10-12', result_preview_url: '/area-one.jpg' },
    { id: 51, project_site_id: 6, analyzed_at: '2026-10-12', result_preview_url: '/area-six.jpg' }]);
  await render(pages.get('OrganizationMonitoring'));
  const section = () => document.querySelector('.monitoring-sites-section');
  const rows = () => [...section().querySelectorAll('.monitoring-site-row')];
  const pagination = () => section().querySelector('.monitoring-sites-pagination');
  const next = () => [...pagination().querySelectorAll('button')].find((button) => button.textContent === 'Next');
  const previous = () => [...pagination().querySelectorAll('button')].find((button) => button.textContent === 'Previous');
  assert.equal(rows().length, 5);
  const firstTile = rows()[0].querySelector('img').getAttribute('src');
  assert.match(firstTile, /^https:\/\/mt\d\.google\.com\/vt\//);
  assert.match(rows()[0].querySelector('[data-map-layer="orthophoto"]').getAttribute('src'), /^\/tiles\/FINAL\//);
  assert.ok(rows()[0].querySelector('.monitoring-site-map-outline path').getAttribute('d'));
  const pointDots = () => [...rows()[0].querySelectorAll('.monitoring-site-map-point')];
  assert.deepEqual(pointDots().map((dot) => dot.getAttribute('data-point-id')), ['11', '12', '13', '14', '15']);
  assert.deepEqual(pointDots().map((dot) => dot.getAttribute('fill')), ['#db2777', '#eab308', '#7f1d1d', '#f97316', '#2563eb']);
  assert.match(rows()[0].querySelector('[role="img"]').getAttribute('aria-label'), /Planting area 1.*boundary outlined/);
  assert.match(rows()[0].querySelector('[role="img"]').getAttribute('aria-label'), /5 planting points/);
  assert.equal(requests.filter((path) => path === '/api/planters/map-points').length, 1);
  assert.equal(requests.includes('/api/analyses/'), false);
  assert.equal(rows()[0].querySelector('.monitoring-site-open').getAttribute('aria-label'), 'View site Planting area 1');
  assert.equal(rows()[0].querySelector('.monitoring-site-history-button').textContent, '');
  assert.equal(rows()[0].querySelector('a'), null);
  assert.match(rows()[1].textContent, /Site boundary unavailable/);
  assert.equal(previous().disabled, true);
  assert.match(pagination().textContent, /Showing 1–5 of 12 sites/);
  await click(next());
  assert.equal(rows().length, 5);
  assert.equal(rows()[0].querySelector('h3').textContent, 'Planting area 6');
  assert.notEqual(rows()[0].querySelector('img').getAttribute('src'), firstTile);
  assert.deepEqual(pointDots().map((dot) => dot.getAttribute('data-point-id')), ['17']);
  assert.equal(requests.filter((path) => path === '/api/planters/map-points').length, 1);
  const savedSection = section(), savedRow = rows()[0], savedTile = rows()[0].querySelector('img');
  const readsBeforeReturn = requests.length;
  await click(document.querySelector('[data-nav="away"]'));
  await click(document.querySelector('[data-nav="back"]'));
  assert.equal(section(), savedSection);
  assert.equal(rows()[0], savedRow);
  assert.equal(rows()[0].querySelector('img'), savedTile);
  assert.match(pagination().textContent, /Page 2 of 3/);
  assert.equal(requests.length, readsBeforeReturn);
  assert.doesNotMatch(section().textContent, /Loading monitoring sites|Refreshing sites/);
  await click(next());
  assert.equal(rows().length, 2);
  assert.equal(next().disabled, true);
  assert.match(pagination().textContent, /Page 3 of 3/);
  await click(previous());
  assert.equal(rows()[0].querySelector('h3').textContent, 'Planting area 6');
  await act(async () => rows()[0].querySelectorAll('img').forEach((tile) => tile.dispatchEvent(new window.Event('error'))));
  assert.match(rows()[0].textContent, /Map imagery unavailable/);
  assert.ok(rows()[0].querySelector('.monitoring-site-map-outline path'));
  assert.ok(rows()[0].querySelector('.monitoring-site-open'));
  await click(section().querySelector('.monitoring-sites-header button'));
  assert.ok(rows()[0].querySelector('img'));
  assert.doesNotMatch(rows()[0].textContent, /Map imagery unavailable/);
  assert.equal(requests.includes('/api/analyses/'), false);
  // If sites disappear while this page is open, the remaining last page stays reachable.
  responses.set('/api/project-sites', { features: sites.slice(0, 3) });
  await act(async () => window.dispatchEvent(new Event('mv:data-changed')));
  await settle();
  assert.equal(rows().length, 3);
  assert.match(pagination().textContent, /Page 1 of 1/);
  assert.equal(next().disabled, true);
  assert.equal(previous().disabled, true);
  assert.equal(mutations.length, 0);
});

test('Sites to Monitor keeps area previews available when points fail, and Refresh restores the dots', async () => {
  responses.set('/api/monitoring/organizations', { organizations: [] });
  responses.set('/api/project-sites', { features: [{ id: 1, properties: { name: 'North site' },
    geometry: { type: 'Polygon', coordinates: [[[122.624, 10.780], [122.62415, 10.780],
      [122.62415, 10.78015], [122.624, 10.78015], [122.624, 10.780]]] } }] });
  responses.set('/api/planters/map-points', () => Response.json({}, { status: 503 }));
  await render(pages.get('OrganizationMonitoring'));
  const section = () => document.querySelector('.monitoring-sites-section');
  assert.match(section().querySelector('[role="alert"]').textContent, /Could not load planting points/);
  assert.ok(section().querySelector('.monitoring-site-map-outline path'));
  assert.equal(section().querySelectorAll('.monitoring-site-map-point').length, 0);
  responses.set('/api/planters/map-points', [{ id: 98, source_project_site_id: 1, latitude: 10.78007, longitude: 122.62405, planting_status: 'planted' }]);
  await click(section().querySelector('.monitoring-sites-header button'));
  assert.equal(section().querySelector('[role="alert"]'), null);
  assert.equal(section().querySelector('.monitoring-site-map-point').getAttribute('data-point-id'), '98');
  assert.equal(mutations.length, 0);
});

test('site selection uses the monitoring modal layout and switches to cached history without changing the list', async () => {
  responses.set('/api/monitoring/organizations', { organizations: [
    { id: 1, name: 'North organization', total_planted: 100, alive_seedlings: 90, monitoring_record_count: 1, monitoring_available: true, next_monitoring_date: manilaDateKey() },
    { id: 2, name: 'South organization', total_planted: 200, alive_seedlings: 150, monitoring_record_count: 1, monitoring_available: true, next_monitoring_date: manilaDateKey() },
  ] });
  const geometry = { type: 'Polygon', coordinates: [[[122.624, 10.780], [122.62415, 10.780],
    [122.62415, 10.78015], [122.624, 10.78015], [122.624, 10.780]]] };
  responses.set('/api/project-sites', { features: [
    { id: 1, geometry, properties: { name: 'North area', organization_id: 1, organization_name: 'North organization',
      notes: 'Enter from the north path.', centroid_lat: 10.78007, centroid_lon: 122.62407 } },
    { id: 2, geometry, properties: { name: 'South area', organization_id: 2, organization_name: 'South organization' } },
  ] });
  responses.set('/api/planters/map-points', [{ id: 98, source_project_site_id: 1, latitude: 10.78007, longitude: 122.62405, planting_status: 'planted' }]);
  const historyOwners = [];
  responses.set('/api/monitoring/organization-records', (_options, url) => {
    const owner = Number(url.searchParams.get('organization_id'));
    historyOwners.push(owner);
    return Response.json({ records: [{ id: owner * 100, organization_id: owner, monitored_at: '2026-10-09',
      alive_count: 90, dead_count: 10, new_dead_count: 0, survival_rate_pct: 90, health_status: 'good',
      actions_taken: `Visit actions for organization ${owner}`, reported_dead_count: owner === 2 ? 1 : 0,
      located_dead_count: 0, unlocated_dead_count: owner === 2 ? 1 : 0, inspector_name: 'LGU officer' }], next_before_id: null });
  });
  responses.set('/api/monitoring/organizations/2/planting-locations', { points: [] });
  await render(pages.get('OrganizationMonitoring'));
  const section = () => document.querySelector('.monitoring-sites-section');
  const row = (id) => section().querySelector(`[data-site-id="${id}"]`);
  await click(row(1).querySelector('.monitoring-site-open'));
  const dialog = () => document.querySelector('.monitoring-site-modal');
  const detail = () => dialog().querySelector('.monitoring-site-details');
  const info = () => detail().querySelector('.monitoring-site-detail-content');
  const history = () => detail().querySelector('.monitoring-site-history');
  const clock = () => detail().querySelector('.monitoring-site-detail-actions .monitoring-site-history-button');
  assert.equal(document.getElementById('planting-map-route'), null);
  assert.equal(dialog().getAttribute('role'), 'dialog');
  assert.equal(dialog().getAttribute('aria-modal'), 'true');
  assert.equal(dialog().classList.contains('org-monitoring-visit-modal'), true);
  assert.equal(dialog().closest('.modal-backdrop').hidden, false);
  assert.equal(document.getElementById(dialog().getAttribute('aria-labelledby')).textContent, 'Site details');
  assert.equal(section().contains(dialog()), false);
  assert.equal(section().querySelector('.monitoring-sites-list').hidden, false);
  assert.equal(section().querySelectorAll('.monitoring-site-row').length, 2);
  assert.equal(info().hidden, false);
  assert.match(info().textContent, /Enter from the north path/);
  assert.match(info().textContent, /10.780070° N, 122.624070° E/);
  assert.equal(info().querySelector('.monitoring-site-map-large svg').getAttribute('viewBox'), '0 0 800 450');
  assert.equal(info().querySelectorAll('.monitoring-site-map-point').length, 1);
  assert.deepEqual(historyOwners, []);
  const savedMap = info().querySelector('.monitoring-site-map-large');
  await click(clock());
  assert.equal(info().hidden, true);
  assert.equal(history().hidden, false);
  assert.match(history().textContent, /Visit actions for organization 1/);
  assert.equal(dialog().querySelector('.modal-title').textContent, 'Monitoring history');
  assert.equal(document.querySelector('.org-monitoring-history-modal'), null);
  assert.deepEqual(historyOwners, [1]);
  await click(clock());
  assert.equal(info().hidden, false);
  assert.equal(history().hidden, true);
  assert.equal(info().querySelector('.monitoring-site-map-large'), savedMap);
  await click(clock());
  assert.deepEqual(historyOwners, [1]);
  const savedHistory = history(), readsBeforeReturn = requests.length;
  await click(document.querySelector('[data-nav="away"]'));
  await click(document.querySelector('[data-nav="back"]'));
  assert.equal(history(), savedHistory);
  assert.equal(history().hidden, false);
  assert.match(history().textContent, /Visit actions for organization 1/);
  assert.equal(requests.length, readsBeforeReturn);
  await click(dialog().querySelector('.modal-action-btn'));
  assert.equal(section().querySelector('.monitoring-sites-list').hidden, false);
  assert.equal(dialog().closest('.modal-backdrop').hidden, true);
  await click(row(1).querySelector('.monitoring-site-history-button'));
  assert.equal(dialog().closest('.modal-backdrop').hidden, false);
  assert.deepEqual(historyOwners, [1]);
  await act(async () => document.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
  await settle();
  assert.equal(dialog().closest('.modal-backdrop').hidden, true);
  assert.equal(document.activeElement, row(1).querySelector('.monitoring-site-open'));
  await click(row(2).querySelector('.monitoring-site-history-button'));
  assert.match(detail().querySelector('h3').textContent, /South area/);
  assert.deepEqual(historyOwners, [1, 2]);
  assert.match(history().textContent, /Visit actions for organization 2/);
  assert.doesNotMatch(history().textContent, /Visit actions for organization 1/);
  await click(history().querySelector('.org-monitoring-location-action'));
  assert.equal(dialog().closest('.modal-backdrop').hidden, true);
  const locationEditor = [...document.querySelectorAll('.modal-card')]
    .find((card) => card.querySelector('.modal-title')?.textContent === 'Identify or correct death locations');
  assert.ok(locationEditor);
  assert.equal(locationEditor.closest('.modal-backdrop').hidden, false);
  const titles = [...document.querySelectorAll('.modal-title')].map((title) => title.id);
  assert.equal(new Set(titles).size, titles.length);
  await act(async () => document.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
  await settle();
  assert.equal(dialog().closest('.modal-backdrop').hidden, false);
  assert.equal(dialog().querySelector('.modal-title').textContent, 'Monitoring history');
  assert.equal(mutations.length, 0);
});

for (const name of ['MapAnalytics', 'ImageProcessing', 'PlanterManagement', 'ErodedZoneEditor', 'MonitoringMapWorkspace']) {
  test(`${name} opens one section at a time and preserves all-closed state on return`, async () => {
    await render(pages.get(name));
    const panel = document.querySelector('.floating-panel');
    if (name === 'PlanterManagement') assert.equal(headers().length, 1);
    else assert.ok(headers().length >= 2);
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
test('mortality warnings align in monitoring cards and update after a death correction', async () => {
  responses.set('/api/monitoring/organizations', { organizations: [
    { id: 1, name: 'Site owner', total_planted: 100, alive_seedlings: 70, monitoring_available: true },
  ] });
  const site = { id: 1, properties: { id: 1, name: 'High mortality site', organization_id: 1,
    organization_name: 'Site owner', point_count: 100,
    mortality_warning: { active: true, total_planted: 100, dead_count: 30, mortality_rate_pct: 30 } },
    geometry: { type: 'Polygon', coordinates: [[[122.625,10.78],[122.626,10.78],[122.626,10.781],[122.625,10.781],[122.625,10.78]]] } };
  responses.set('/api/project-sites', { features: [site] });
  await render(pages.get('OrganizationMonitoring'));
  const row = document.querySelector('.monitoring-site-row');
  assert.match(row.textContent, /Warning: High mortality/);
  assert.match(row.textContent, /30 of 100 planted seedlings have recorded deaths \(30% mortality\)/);
  assert.equal(row.querySelector('.monitoring-site-map-outline path[fill-rule="evenodd"]').getAttribute('stroke'), '#15803d');
  await click(row.querySelector('.monitoring-site-open'));
  assert.match(document.querySelector('.monitoring-site-details').textContent, /All recorded causes are included/);
  site.properties.mortality_warning = { active: false, total_planted: 100, dead_count: 29, mortality_rate_pct: 29 };
  await act(async () => window.dispatchEvent(new window.Event('mv:data-changed')));
  await settle();
  assert.equal(document.querySelector('.monitoring-site-mortality-warning'), null);
});
