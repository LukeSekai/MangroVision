import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { Activity, act, createElement } from 'react';
import { JSDOM } from 'jsdom';
import { build } from 'vite';
import { writeFile, unlink } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import { pid } from 'node:process';
import { PLANNED_SPECIES_COLORS, POINT_STATUS_COLORS } from '../utils/pointStatus.js';

let dom, root, createRoot, SavedPointExports, leaflet, requests, downloads, responder;
const maps = [], removedMaps = [];
const windowErrors = [];
const canvasStrokes = [], canvasFills = [];
const bundlePath = resolve(`node_modules/.point-export-tests-${pid}.mjs`);
const originalGlobals = new Map();
const points = [
  { id: 1, analysis_id: 52, point_num: 1, image_name: 'Analysis 2', latitude: 10.78, longitude: 122.624,
    source_project_site_id: 4, source_project_site_name: 'North site', species: 'bungalon', map_status: 'planted' },
  { id: 2, analysis_id: 60, point_num: 2, image_name: 'Analysis 3', latitude: 10.7802, longitude: 122.625,
    source_project_site_id: null, species: 'rhizophora', map_status: 'planned' },
];
const props = { points, loaded: true, loading: false, analyses: [{ id: 52, display_number: 2, source_image_name: 'DJI.JPG' }] };

before(async () => {
  dom = new JSDOM('<div id="root"></div>', { url: 'http://workspace.example', pretendToBeVisual: true });
  dom.window.addEventListener('error', (event) => { windowErrors.push(event.error); event.preventDefault(); });
  // Exercise the real Leaflet map/markers and bounds without a raster graphics
  // dependency or network imagery in this isolated DOM fixture.
  dom.window.HTMLCanvasElement.prototype.getContext = function () {
    const context = { canvas: this, path: [],
      beginPath() { this.path = []; },
      arc(...args) { this.path.push(['arc', ...args]); },
      moveTo(...args) { this.path.push(['move', ...args]); },
      lineTo(...args) { this.path.push(['line', ...args]); },
      stroke() { canvasStrokes.push({ color: this.strokeStyle, path: [...this.path] }); },
      fill() { canvasFills.push({ color: this.fillStyle, path: [...this.path] }); },
    };
    return new Proxy(context, { get: (target, key) => key in target ? target[key] : () => {} });
  };
  Object.defineProperty(dom.window.HTMLElement.prototype, 'clientWidth', {
    get() { return this.classList.contains('point-export-map') ? 600 : 0; },
  });
  Object.defineProperty(dom.window.HTMLElement.prototype, 'clientHeight', {
    get() { return this.classList.contains('point-export-map') ? 360 : 0; },
  });
  const globals = { window: dom.window, document: dom.window.document, navigator: dom.window.navigator,
    HTMLElement: dom.window.HTMLElement, SVGElement: dom.window.SVGElement, IS_REACT_ACT_ENVIRONMENT: true,
    requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
    cancelAnimationFrame: dom.window.cancelAnimationFrame.bind(dom.window),
    ResizeObserver: class { observe() {} disconnect() {} },
    fetch: async (url, init) => { requests.push({ url, ...init }); return responder(url, init); },
  };
  for (const [key, value] of Object.entries(globals)) {
    originalGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  dom.window.HTMLAnchorElement.prototype.click = function () { downloads.push(this.download); };
  ({ default: leaflet } = await import('leaflet'));
  leaflet.Map.addInitHook(function () { maps.push(this); this.on('unload', () => removedMaps.push(this)); });
  ({ createRoot } = await import('react-dom/client'));
  const bundle = await build({ configFile: false, logLevel: 'silent',
    ssr: { external: ['react', 'react/jsx-runtime', 'react-dom', 'leaflet'] },
    build: { ssr: resolve('src/components/SavedPointExports.jsx'), write: false, minify: false,
      rollupOptions: { output: { format: 'es', inlineDynamicImports: true } } },
  });
  await writeFile(bundlePath, bundle.output.find((item) => item.type === 'chunk' && item.isEntry).code);
  ({ default: SavedPointExports } = await import(pathToFileURL(bundlePath).href));
});
after(async () => {
  await unlink(bundlePath).catch(() => {});
  dom?.window.close();
  for (const [key, descriptor] of originalGlobals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
});
beforeEach(() => {
  requests = []; downloads = []; maps.length = 0; removedMaps.length = 0; windowErrors.length = 0;
  canvasStrokes.length = 0; canvasFills.length = 0;
  responder = async () => new Response('synthetic export');
  root = createRoot(document.getElementById('root'));
});
afterEach(async () => {
  await act(async () => root.unmount());
  await new Promise((resolve) => dom.window.requestAnimationFrame(resolve));
  assert.deepEqual(windowErrors, [], 'closed preview must not leave callbacks drawing on a removed canvas');
});
const render = async (updates = {}) => act(async () => root.render(createElement(SavedPointExports, { ...props, ...updates })));
const button = (text) => [...document.querySelectorAll('button')].find((element) => element.getAttribute('aria-label') === text || element.textContent === text);
const click = async (text) => act(async () => button(text).click());
const chooseScope = async (kind) => act(async () => document.querySelector(`input[type="radio"][value="${kind}"]`).click());
const select = async (value) => act(async () => {
  const control = document.querySelector('select');
  control.value = value;
  control.dispatchEvent(new dom.window.Event('change', { bubbles: true }));
});
const markers = () => {
  const result = [];
  maps.at(-1).eachLayer((layer) => { if (layer instanceof leaflet.CircleMarker) result.push(layer); });
  return result;
};
const locations = () => markers().map((marker) => { const location = marker.getLatLng(); return [location.lat, location.lng]; });

test('the sidebar starts with formats and opens scope controls with an independent map', async () => {
  await render();
  assert.equal(document.querySelector('select'), null);
  assert.equal(document.querySelector('[role="dialog"]'), null);
  assert.equal(maps.length, 0);
  for (const format of ['CSV', 'GPX', 'KMZ', 'GeoJSON']) assert.ok(button(`Export ${format}`));
  assert.equal(document.querySelectorAll('.point-export-format').length, 4);
  await click('Export CSV');
  assert.equal(document.querySelector('.modal-backdrop').parentElement, document.body);
  assert.equal(document.querySelectorAll('input[type="radio"]').length, 4);
  assert.equal(document.querySelector('select'), null);
  assert.equal(maps.length, 1);
  assert.equal(markers().length, points.length);
  assert.deepEqual(new Set(locations().map((location) => location.join(','))),
    new Set(points.map((point) => [point.latitude, point.longitude].join(','))));
  assert.equal(button('Download CSV').disabled, false);
  assert.ok(maps[0].getBounds().contains([points[0].latitude, points[0].longitude]));
  await click('Cancel');
  assert.equal(removedMaps.length, 1);
});

test('scope changes clear old choices and markers until a value is chosen', async () => {
  await render();
  await click('Export CSV');
  await chooseScope('site');
  assert.equal(markers().length, 0);
  assert.equal(button('Download CSV').disabled, true);
  assert.match(document.body.textContent, /Choose a project site to preview its points/);
  await select('4');
  assert.deepEqual(locations(), [[points[0].latitude, points[0].longitude]]);
  assert.match(document.querySelector('.point-export-selection').textContent, /1 pointNorth site/);
  await chooseScope('species');
  assert.equal(document.querySelector('select').value, '');
  assert.equal(markers().length, 0);
  await select('rhizophora');
  assert.deepEqual(locations(), [[points[1].latitude, points[1].longitude]]);
  await chooseScope('all');
  assert.equal(document.querySelector('select'), null);
  assert.equal(markers().length, 2);
  assert.equal(requests.length, 0);
});

test('solid point colors prioritize dead, erosion and planted over species without losing their details', async () => {
  const mixed = [
    points[0], { ...points[1], map_status: 'assigned' },
    { ...points[0], id: 3, species: null, planted_species: 'Legacy species', inside_eroded_zone: true,
      eroded_unavailable: false, availability_reason: 'Inside an eroded zone' },
    { ...points[1], id: 4, map_status: 'dead', death_at: '2026-10-20' },
    { ...points[0], id: 5, map_status: 'dead', death_at: '2026-10-20', inside_eroded_zone: true },
    { ...points[0], id: 6, species: null, map_status: 'planned' },
    { ...points[0], id: 7, map_status: 'planned' },
    { ...points[0], id: 8, species: 'Legacy species', map_status: 'planned' },
    { ...points[1], id: 9, map_status: 'planted' },
  ];
  await render({ points: mixed });
  await click('Export CSV');
  const byId = new Map(markers().map((marker) => [marker.options.mvPointId, marker]));
  assert.equal(byId.size, mixed.length, 'coloring must not duplicate exported locations');
  assert.equal(byId.get(1).options.fillColor, POINT_STATUS_COLORS.planted, 'planting overrides the Bungalon color');
  assert.equal(byId.get(2).options.fillColor, PLANNED_SPECIES_COLORS.rhizophora, 'assigned points keep their species fill');
  assert.equal(byId.get(3).options.fillColor, POINT_STATUS_COLORS.unavailable, 'erosion overrides planting even when unavailable is false');
  assert.equal(byId.get(4).options.fillColor, POINT_STATUS_COLORS.dead, 'death overrides the Rhizophora color');
  assert.equal(byId.get(5).options.fillColor, POINT_STATUS_COLORS.dead, 'death overrides erosion and the Bungalon color');
  assert.equal(byId.get(6).options.fillColor, '#64748b');
  assert.equal(byId.get(7).options.fillColor, PLANNED_SPECIES_COLORS.bungalon);
  assert.equal(byId.get(8).options.fillColor, '#64748b');
  assert.equal(byId.get(9).options.fillColor, POINT_STATUS_COLORS.planted, 'planting overrides the Rhizophora color');
  assert.ok([...byId.values()].every((marker) => marker.options.color === '#173421' && marker.options.weight === 0.5),
    'all solid dots use the same neutral outline');
  const tooltip = byId.get(5).getTooltip().getContent().textContent;
  assert.match(tooltip, /Bungalon/);
  assert.match(tooltip, /Dead/);
  assert.match(tooltip, /Inside an eroded zone/);
  const legend = document.querySelector('[aria-label="Export point colors"]');
  for (const label of ['Dead', 'Inside eroded zone', 'Planted', 'Bungalon', 'Rhizophora', 'Other / unrecorded species']) assert.ok(legend.textContent.includes(label));
  assert.match(document.body.textContent, /Color priority: dead, then eroded, then planted, then species/);

  canvasStrokes.length = 0; canvasFills.length = 0;
  await act(async () => {
    maps[0].setZoom(18, { animate: false });
    await new Promise((resolve) => dom.window.requestAnimationFrame(resolve));
  });
  for (const color of [POINT_STATUS_COLORS.dead, POINT_STATUS_COLORS.unavailable, POINT_STATUS_COLORS.planted,
    PLANNED_SPECIES_COLORS.bungalon, PLANNED_SPECIES_COLORS.rhizophora, '#64748b']) {
    assert.ok(canvasFills.some((fill) => fill.color === color && fill.path.some((command) => command[0] === 'arc')),
      `the map canvas must keep solid ${color} dots after zooming`);
  }
  assert.ok(canvasStrokes.every((stroke) => stroke.path.every((command) => command[0] === 'arc')),
    'solid dots must not add death crosses');
  assert.equal(byId.get(5).getRadius(), byId.get(1).getRadius());
  await click('Download CSV');
  const payload = JSON.parse(requests[0].body);
  assert.equal(payload.waypoints.length, mixed.length);
  const both = payload.waypoints.find((point) => point.point_id === 5);
  assert.equal(both.status, 'dead');
  assert.equal(both.eroded_unavailable, true);
  assert.equal(both.species, 'bungalon');
});

for (const format of ['CSV', 'GPX', 'KMZ', 'GeoJSON']) {
  test(`${format} downloads exactly the points shown on the preview map`, async () => {
    await render();
    await click(`Export ${format}`);
    await chooseScope('analysis');
    await select('52');
    const previewed = locations();
    // A background refresh must not replace the reviewed snapshot.
    await render({ points: [] });
    await click(`Download ${format}`);
    assert.equal(requests.length, 1);
    const body = JSON.parse(requests[0].body);
    assert.deepEqual(body.waypoints.map((point) => [point.lat, point.lon]), previewed);
    assert.deepEqual(body.waypoints.map((point) => point.point_id), [1]);
    assert.equal(body.waypoints[0].project_site_id, 4);
    const extension = format.toLowerCase();
    assert.match(downloads[0], new RegExp(`^MangroVision_analysis-2-dji-jpg_analysis-52_planting-points_\\d{4}-\\d{2}-\\d{2}\\.${extension}$`));
    assert.equal(body.filename_stem, downloads[0].replace(/\.[^.]+$/, ''));
    assert.match(body.export_name, /MangroVision — Analysis 2 — DJI.JPG — Planting points/);
    assert.equal(body.waypoints[0].name, 'Analysis 2 — Point 001 (ID 1)');
    assert.equal(body.waypoints[0].navigation_name, 'MV-000001');
    assert.match(requests[0].url, new RegExp(`/api/export/${format.toLowerCase()}$`));
    assert.match(document.body.textContent, /1 planting point exported as/);
    assert.equal(removedMaps.length, 1);
  });
}

test('panning and refitting the map keep the same export membership', async () => {
  await render();
  await click('Export KMZ');
  await act(async () => maps[0].setView([10.8, 122.7], 17));
  assert.equal(markers().length, 2);
  await click('Fit selected points');
  assert.ok(maps[0].getBounds().contains([points[0].latitude, points[0].longitude]));
  await click('Download KMZ');
  assert.equal(JSON.parse(requests[0].body).waypoints.length, 2);
});

for (const ctrlKey of [false, true]) {
  test(`${ctrlKey ? 'Ctrl / trackpad pinch' : 'mouse wheel'} zoom changes the preview map and cancels browser zoom`, async () => {
    await render();
    await click('Export CSV');
    await act(async () => maps[0].setZoom(18, { animate: false }));
    const mapElement = document.querySelector('.point-export-map');
    const wheel = new dom.window.WheelEvent('wheel', { deltaY: -120, deltaMode: 0,
      clientX: 300, clientY: 180, ctrlKey, bubbles: true, cancelable: true });
    await act(async () => {
      mapElement.dispatchEvent(wheel);
      await new Promise((resolve) => setTimeout(resolve, 80));
    });
    assert.equal(wheel.defaultPrevented, true, 'the browser must not zoom or scroll the page over the map');
    assert.ok(maps[0].getZoom() > 18, 'the gesture must zoom the preview map');
    assert.equal(markers().length, 2);
    const outside = new dom.window.WheelEvent('wheel', { deltaY: -120, ctrlKey, bubbles: true, cancelable: true });
    document.querySelector('.point-export-controls').dispatchEvent(outside);
    assert.equal(outside.defaultPrevented, false, 'controls outside the map retain normal browser behavior');
    await click('Download CSV');
    assert.equal(JSON.parse(requests[0].body).waypoints.length, 2);
  });
}

test('closing immediately after a wheel gesture cancels pending map zoom', async () => {
  await render();
  await click('Export CSV');
  const wheel = new dom.window.WheelEvent('wheel', { deltaY: -120, ctrlKey: true, bubbles: true, cancelable: true });
  await act(async () => document.querySelector('.point-export-map').dispatchEvent(wheel));
  await click('Cancel');
  await act(async () => { await new Promise((resolve) => setTimeout(resolve, 80)); });
  assert.equal(removedMaps.length, 1);
  assert.equal(document.querySelector('[role="dialog"]'), null);
});

test('tile failure keeps the selected points visible and downloadable', async () => {
  await render();
  await click('Export CSV');
  await act(async () => maps[0].eachLayer((layer) => { if (layer instanceof leaflet.TileLayer) layer.fire('tileerror'); }));
  assert.match(document.body.textContent, /Some map imagery could not load/);
  assert.equal(markers().length, 2);
  assert.equal(button('Download CSV').disabled, false);
});

test('a failed download keeps the selected scope and map ready for retry', async () => {
  responder = async () => Response.json({ detail: 'Try again shortly.' }, { status: 503 });
  await render();
  await click('Export GPX');
  await chooseScope('site');
  await select('4');
  await click('Download GPX');
  assert.match(document.body.textContent, /Try again shortly/);
  assert.equal(downloads.length, 0);
  assert.equal(document.querySelector('select').value, '4');
  assert.equal(markers().length, 1);
  responder = async () => new Response('gpx');
  await click('Download GPX');
  assert.equal(downloads.length, 1);
});

test('in-flight exports reject duplicate submission and abort on close or hiding', async () => {
  let release;
  responder = async () => new Promise((resolve) => { release = resolve; });
  await render();
  await click('Export KMZ');
  await act(async () => { button('Download KMZ').click(); button('Download KMZ').click(); });
  assert.equal(requests.length, 1);
  assert.equal(document.querySelector('.modal-action-btn.is-busy').disabled, true);
  assert.ok(document.querySelector('fieldset').disabled);
  await act(async () => root.render(null));
  assert.equal(requests[0].signal.aborted, true);
  assert.equal(removedMaps.length, 1);
  await act(async () => release(new Response('kml')));
  assert.equal(downloads.length, 0);
});

test('failed initial loading has retry and prevents opening an incomplete preview', async () => {
  let retried = 0;
  await render({ loaded: false, points: [], onRetry: () => { retried += 1; } });
  assert.match(document.body.textContent, /Saved points could not be loaded/);
  assert.equal(button('Export CSV').disabled, true);
  await click('Retry loading points');
  assert.equal(retried, 1);
  await render({ loading: true, loaded: false });
  assert.match(document.body.textContent, /Loading saved points/);
  assert.equal(button('Export CSV').disabled, true);
  assert.equal(maps.length, 0);
});

test('a retained hidden page removes its map and restores the reviewed scope on return', async () => {
  let release;
  responder = async () => new Promise((resolve) => { release = resolve; });
  const page = (mode) => createElement(Activity, { mode }, createElement(SavedPointExports, props));
  await act(async () => root.render(page('visible')));
  await click('Export GeoJSON');
  await chooseScope('species');
  await select('rhizophora');
  await click('Download GeoJSON');
  await act(async () => root.render(page('hidden')));
  assert.equal(requests[0].signal.aborted, true);
  assert.equal(removedMaps.length, 1);
  assert.equal(document.querySelector('.modal-backdrop').style.display, 'none');
  await act(async () => release(new Response('cancelled')));
  await act(async () => root.render(page('visible')));
  assert.equal(document.querySelector('select').value, 'rhizophora');
  assert.equal(button('Download GeoJSON').disabled, false);
  assert.equal(markers().length, 1);
  assert.equal(maps.length, 2);
  assert.equal(downloads.length, 0);
  await click('Cancel');
  assert.equal(document.querySelector('[role="dialog"]'), null);
});

test('Escape closes the preview and Enter on a scope choice does not download', async () => {
  await render();
  button('Export CSV').focus();
  await click('Export CSV');
  const radio = document.querySelector('input[type="radio"]');
  await act(async () => radio.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'Enter', bubbles: true })));
  assert.equal(requests.length, 0);
  await act(async () => document.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
  assert.equal(document.querySelector('[role="dialog"]'), null);
  assert.equal(document.activeElement, button('Export CSV'));
});
