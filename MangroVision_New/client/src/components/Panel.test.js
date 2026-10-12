import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement, useState } from 'react';
import { MemoryRouter, Route, useNavigate } from 'react-router-dom';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
import { manilaDateKey } from '../utils/monitoringSchedule.js';

let server, dom, root, createRoot, RetainedRoutes, Panel, PanelCard, useMapStore, useProcessingStore, originalStartProcess;
let PlantingMapWorkspace, Sidebar, MapView;
const pages = new Map();
const originalGlobals = new Map();
let requests, responses, mutations;

before(async () => {
  dom = new JSDOM('<div id="root"></div>', { url: 'http://workspace.example', pretendToBeVisual: true });
  const globals = {
    window: dom.window, document: dom.window.document, localStorage: dom.window.localStorage,
    Event: dom.window.Event, IS_REACT_ACT_ENVIRONMENT: true,
    FileReader: dom.window.FileReader, File: dom.window.File, FormData: dom.window.FormData,
    requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
    cancelAnimationFrame: dom.window.cancelAnimationFrame.bind(dom.window),
    ResizeObserver: class { observe() {} disconnect() {} },
    fetch: async (input, options) => {
      const path = new URL(input, window.location.origin).pathname;
      requests.push(path);
      if (options?.method === 'POST') mutations.push({ path, body: options.body instanceof FormData ? Object.fromEntries(options.body) : JSON.parse(options.body || '{}') });
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
  requests = [];
  responses = new Map([['/api/analyses/', []], ['/api/project-sites', { features: [] }], ['/api/planters/map-points', []]]);
  mutations = [];
  useMapStore.getState().resetWorkspaceData();
  useMapStore.setState({ stats: { analyses: [], points: [], total_analyses: 0 } });
  useProcessingStore.getState().reset();
  useProcessingStore.setState({ startProcess: originalStartProcess });
  root = createRoot(document.getElementById('root'));
});
afterEach(async () => { await act(async () => root.unmount()); });

function Navigation() {
  const navigate = useNavigate();
  return createElement('nav', null,
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

test('Assign Points opens the form without a duplicate overview and keeps activity reports accessible', async () => {
  await render(pages.get('PlanterManagement'));
  assert.equal(headers().some((header) => header.textContent.includes('Overview')), false);
  assert.match(expanded()[0].textContent, /Assign available points/);
  assert.equal(requests.includes('/api/planters/dashboard'), false);
  await click(document.querySelector('.assignment-activity-report'));
  assert.equal(document.querySelector('[role="dialog"]').getAttribute('aria-labelledby'), 'par-title');
  assert.equal(document.querySelector('#par-title').textContent, 'Organization Activity');
  await click(document.querySelector('[aria-label="Close report"]'));
  assert.equal(document.querySelector('.par-backdrop'), null);
  assert.equal(mutations.length, 0);
});

test('Quick Assign shows available points and explains invalid counts only when assigning', async () => {
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
  const quickAssign = headers().find((header) => header.textContent.includes('Assign available points'));
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
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(count, '1.5');
    count.dispatchEvent(new window.Event('input', { bubbles: true }));
    count.focus();
    assign.focus();
  });
  assert.equal(count.getAttribute('aria-invalid'), 'false');
  assert.equal(assign.disabled, false);
  await click(assign);
  assert.equal(count.getAttribute('aria-invalid'), 'true');
  assert.match(document.getElementById(count.getAttribute('aria-describedby')).textContent, /whole number from 1 to 100/);
  assert.equal(mutations.length, 0);
});

test('the Planting Map starts with its overview and has no Next steps cards', async () => {
  await render(pages.get('MapAnalytics'));
  assert.equal(document.querySelector('.next-actions'), null);
  assert.match(expanded()[0].textContent, /Overview/);
});

test('Quick Assign submits CICT mixed species together using their recorded species', async () => {
  responses.set('/api/planters/', [{ id: 24, organization_id: 22, organization_name: 'CICT',
    status: 'active', participant_count: 10, registration_pending: false }]);
  responses.set('/api/planter-auth/organizations', { organizations: [{ id: 22, name: 'CICT' }] });
  responses.set('/api/planters/organizations/22/assignments', { assignment_id: 1, assignment_ids: [1, 2] });
  useMapStore.setState({
    projectSites: { features: [{ type: 'Feature', id: 15,
      properties: { name: 'CICT', organization_id: 22, point_count: 103 } }] },
    points: Array.from({ length: 103 }, (_, index) => ({
      id: index + 1, point_num: index + 1, analysis_id: index < 99 ? 148 : 170, latitude: 10.5,
      longitude: 122.5 + index * .00001, source_project_site_id: 15,
      source_organization_id: 22, planting_status: 'planned', species: index < 99 ? 'bungalon' : 'rhizophora',
    })),
  });
  await render(pages.get('PlanterManagement'));
  const quickAssign = headers().find((header) => header.textContent.includes('Assign available points'));
  if (quickAssign.getAttribute('aria-expanded') === 'false') await click(quickAssign);
  await act(async () => {
    const organization = document.querySelector('select');
    organization.value = '22';
    organization.dispatchEvent(new window.Event('change', { bubbles: true }));
  });
  await settle();
  assert.match(document.body.textContent, /99 Bungalon · 4 Rhizophora/);
  const assign = [...document.querySelectorAll('button')].find((button) => /Assign 103 Points/.test(button.textContent));
  assert.equal(assign.disabled, false);
  await click(assign);
  assert.equal(mutations.length, 1);
  assert.equal(mutations[0].path, '/api/planters/organizations/22/assignments');
  assert.deepEqual({ ...mutations[0].body, planting_point_ids: [...mutations[0].body.planting_point_ids].sort((a, b) => a - b) },
    { planting_point_ids: Array.from({ length: 103 }, (_, i) => i + 1), site_zone_id: 15 });
  assert.match(document.body.textContent, /Assigned 103 points to CICT/);
});

test('Quick Assign links points to the selected planting activity and requires a choice when ambiguous', async () => {
  responses.set('/api/planters/', [{ id: 24, organization_id: 22, organization_name: 'Demo group', status: 'active', participant_count: 10 }]);
  responses.set('/api/planter-auth/organizations', { organizations: [{ id: 22, name: 'Demo group' }] });
  responses.set('/api/planting-schedules', { schedules: [31, 32].map((id) => ({
    id, organization_id: 22, project_site_id: 15, appointment_type: 'tree_planting',
    status: 'confirmed', title: `Activity ${id}`, date: '2090-01-01',
  })) });
  responses.set('/api/planters/organizations/22/assignments', { assignment_id: 1, assignment_ids: [1] });
  useMapStore.setState({ projectSites: { features: [{ id: 15, properties: { name: 'Demo site', organization_id: 22 } }] },
    points: [{ id: 1, point_num: 1, analysis_id: 1, latitude: 10.5, longitude: 122.5,
      source_project_site_id: 15, source_organization_id: 22, planting_status: 'planned', species: 'bungalon' }] });
  await render(pages.get('PlanterManagement'));
  await act(async () => {
    const organization = document.getElementById('assign-organization');
    organization.value = '22'; organization.dispatchEvent(new window.Event('change', { bubbles: true }));
  });
  await settle();
  const assign = [...document.querySelectorAll('button')].find((button) => /Assign 1 Point/.test(button.textContent));
  await click(assign);
  assert.equal(mutations.length, 0);
  assert.match(document.body.textContent, /Choose the planting activity/);
  await act(async () => {
    const activity = document.getElementById('assign-activity');
    activity.value = '32'; activity.dispatchEvent(new window.Event('change', { bubbles: true }));
  });
  await click(assign);
  assert.equal(mutations.length, 1);
  assert.equal(mutations[0].body.planting_schedule_id, 32);
});

test('review-analysis links open history when there is no current result', async () => {
  await render(pages.get('ImageProcessing'), '/page?action=review');
  assert.ok(document.querySelector('.analysis-history-modal'));
  assert.match(document.querySelector('.analysis-history-modal').textContent, /Image Analysis History/);
  assert.equal(mutations.length, 0);
});

const savedImages = [
  { id: 52, image_name: 'Analysis 52', source_image_name: 'QJBJ.JPG', analyzed_at: '2026-10-07T08:00:00+08:00',
    plantable_area_m2: 175, canopy_area_m2: 90, canopy_coverage_pct: 30, hexagon_count: 50, result_preview_url: 'https://preview.example/QJBJ.webp' },
  { id: 51, image_name: 'Analysis 51', source_image_name: 'ITBH.JPG', analyzed_at: '2026-09-30T08:00:00+08:00',
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
test(`View area in map opens the exact saved footprint from the ${source} without needing planting points`, async () => {
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
  assert.ok(document.querySelector('#planting-map-route'));
  assert.equal(document.querySelector('#planting-map-route').style.display, '');
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
