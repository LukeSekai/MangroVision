import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement } from 'react';
import { JSDOM } from 'jsdom';
import { build } from 'vite';
import { readFile, writeFile, unlink } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import process from 'node:process';

let dom, root, createRoot, FieldApp, store, leaflet, site, rows, requests, marks, watches, gps, locate;
const maps = [], savedGlobals = new Map();
const bundlePath = resolve(`node_modules/.field-navigation-${process.pid}.mjs`);
const fixtures = [100, 99, 98, 97].map((point_num, index) => ({
  assignment_point_id: index + 1, point_num, assignment_status: 'pending',
  latitude: 10.78012, longitude: 122.62707 + index * 0.00003,
  analysis_id: 1, organization_id: 1, sequence_num: index + 1,
}));

before(async () => {
  [site] = JSON.parse(await readFile(new URL('../../../api/data/site_access_routes.json', import.meta.url), 'utf8'));
  dom = new JSDOM('<div id="root"></div>', { url: 'http://field.example.test', pretendToBeVisual: true });
  dom.window.SVGSVGElement.prototype.createSVGRect = () => ({});
  Object.defineProperty(dom.window.HTMLElement.prototype, 'clientWidth', { get() { return this.classList.contains('field-map') ? 390 : 0; } });
  Object.defineProperty(dom.window.HTMLElement.prototype, 'clientHeight', { get() { return this.classList.contains('field-map') ? 500 : 0; } });
  Object.defineProperty(dom.window.navigator, 'geolocation', { value: {
    getCurrentPosition: (success, error) => locate(success, error),
    watchPosition: (success, error) => { const id = Symbol(); watches.set(id, { success, error }); return id; },
    clearWatch: id => watches.delete(id),
  } });
  const globals = {
    window: dom.window, document: dom.window.document, navigator: dom.window.navigator,
    localStorage: dom.window.localStorage, sessionStorage: dom.window.sessionStorage,
    HTMLElement: dom.window.HTMLElement, SVGElement: dom.window.SVGElement,
    IS_REACT_ACT_ENVIRONMENT: true,
    requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
    cancelAnimationFrame: dom.window.cancelAnimationFrame.bind(dom.window),
    ResizeObserver: class { observe() {} disconnect() {} },
    fetch: async (input, options = {}) => {
      assert.equal(input, '/api/routing/compute');
      const request = JSON.parse(options.body);
      requests.push(request);
      return Response.json({ origin: [request.origin_lat, request.origin_lon],
        target: [request.dest_lat, request.dest_lon],
        entrance: site.access_path.at(-1), access_start: site.access_path[0],
        site_area: site.site_area, planting_area: site.planting_area,
        polyline: site.access_path, segments: [{ kind: 'road', polyline: site.access_path }],
        route_source: 'site_route',
      });
    },
  };
  for (const [key, value] of Object.entries(globals)) {
    savedGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  ({ default: leaflet } = await import(pathToFileURL(resolve('node_modules/leaflet/dist/leaflet-src.js')).href));
  leaflet.Map.addInitHook(function () { maps.push(this); });
  ({ createRoot } = await import('react-dom/client'));
  const bundle = await build({ configFile: false, logLevel: 'silent',
    ssr: { external: ['react', 'react/jsx-runtime', 'react-dom', 'leaflet'] },
    build: { ssr: resolve('src/pages/FieldApp.jsx'), write: false, minify: false,
      rollupOptions: { output: { format: 'es' } } },
    plugins: [{ name: 'field-navigation-fixtures', enforce: 'pre', transform(code, id) {
      const path = id.replaceAll('\\', '/');
      if (/\/components\/(GuideButton|FieldActivities|ActivityFeed)\.jsx$/.test(path)) return 'export default function Fixture() { return null; }';
      if (path.endsWith('/pages/FieldApp.jsx')) return `${code}\nexport { usePlanterAuthStore as TestStore };`;
    } }],
  });
  await writeFile(bundlePath, bundle.output.find(item => item.type === 'chunk' && item.isEntry).code);
  ({ default: FieldApp, TestStore: store } = await import(pathToFileURL(bundlePath).href));
});

after(async () => {
  await unlink(bundlePath).catch(() => {}); dom?.window.close();
  for (const [key, descriptor] of savedGlobals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
  }
});
beforeEach(() => {
  maps.length = 0; rows = structuredClone(fixtures); requests = []; marks = []; watches = new Map();
  gps = { coords: { latitude: site.bridge_start[0], longitude: site.bridge_start[1], accuracy: 5 } };
  locate = success => success(gps);
  localStorage.clear(); sessionStorage.clear();
  store.setState({ isAuthenticated: true, token: 'fixture', planter: { id: 1, organization_id: 1, participant_slot: 1, full_name: 'Fixture planter' },
    hydrateSession: async () => {},
    fetchFieldPoints: async () => ({ points: structuredClone(rows), projectSites: [],
      siteAccess: [{ name: site.name, label: 'LIKE entrance', entrance: site.access_path.at(-1), site_area: site.site_area }] }),
    markPointStatus: async (id, status) => {
      marks.push({ id, status });
      rows = rows.map(point => point.assignment_point_id === id ? { ...point, assignment_status: status } : point);
    },
  });
  root = createRoot(document.getElementById('root'));
});
afterEach(async () => { await act(async () => root.unmount()); assert.equal(watches.size, 0); });

async function render() {
  await act(async () => { root.render(createElement(FieldApp)); });
  await act(async () => { await new Promise(resolve => setTimeout(resolve, 0)); });
}
function button(text) { return [...document.querySelectorAll('button')].find(element => element.textContent === text); }
function nextButton() { return document.querySelector('.field-work-actions .field-btn-primary'); }
async function click(element) { assert.ok(element, 'Expected control'); await act(async () => element.click()); }
function activeLabel() { return document.querySelector('.field-navigation-control span')?.textContent; }
function orangeLines() { return Object.values(maps.at(-1)._layers).filter(layer => layer instanceof leaflet.Polyline && layer.options.color === '#ea580c'); }
async function selectPoint(num) {
  await click(document.querySelector('.field-list-toggle'));
  await click([...document.querySelectorAll('.field-point-row')].find(row => row.querySelector('strong').textContent === `Point #${num}`));
}

test('Next starts navigation immediately and the orange overlay connects every assigned point', async () => {
  await render();
  await click(nextButton());
  assert.equal(document.querySelector('.field-overlay[role="dialog"]'), null);
  assert.equal(activeLabel(), 'Point #100');
  assert.equal(orangeLines().length, 4, 'Bridge-to-active guide plus three point-to-point legs');
  assert.match(nextButton().textContent, /#99/);
  assert.equal(watches.size, 1);
  await click(nextButton());
  assert.equal(activeLabel(), 'Point #99');
  assert.equal(orangeLines().length, 3);
  assert.equal(requests.length, 1, 'The same physical bridge approach is reused');
  assert.equal(document.querySelector('.field-overlay[role="dialog"]'), null);
  assert.deepEqual(marks, [], 'Navigating does not mark points as planted');
  assert.ok(rows.every(point => point.assignment_status === 'pending'));
});

test('starting from a chosen marker still covers all points and Next reaches the end without cycling', async () => {
  await render();
  await selectPoint(99);
  await click(button('Navigate to this point'));
  assert.equal(activeLabel(), 'Point #99');
  for (const num of [100, 98, 97]) { await click(nextButton()); assert.equal(activeLabel(), `Point #${num}`); }
  assert.equal(nextButton().disabled, true);
  assert.equal(nextButton().textContent, 'Last point in route');
  assert.deepEqual(marks, []);
});

test('Next and the orange path preserve the planned walk when an upcoming point becomes unavailable', async () => {
  const offsets = [0, 2, 5, -3];
  rows = fixtures.map((point, index) => ({ ...point, longitude: fixtures[0].longitude + offsets[index] * 0.00003 }));
  await render(); await selectPoint(100); await click(button('Navigate to this point'));
  assert.match(nextButton().textContent, /#97/);
  rows[3].assignment_status = 'completed';
  await click(document.querySelector('[aria-label="Refresh assignments"]'));
  assert.match(nextButton().textContent, /#99/);
  const nextLeg = orangeLines()[1].getLatLngs();
  assert.equal(nextLeg[0].lng, rows[0].longitude);
  assert.equal(nextLeg[1].lng, rows[1].longitude);
  await click(nextButton());
  assert.equal(activeLabel(), 'Point #99');
  assert.match(nextButton().textContent, /#98/, 'Replanning must keep the still-unvisited point');
  await click(nextButton());
  assert.equal(activeLabel(), 'Point #98');
  assert.equal(nextButton().disabled, true);
  assert.deepEqual(marks, []);
});

test('the entrance-nearest point has an actionable map overlay before requesting GPS or road directions', async () => {
  rows[0].latitude -= 0.00003;
  await render();
  const overlay = document.querySelector('.field-start-point-overlay');
  assert.ok(overlay);
  assert.match(overlay.textContent, /Start at Point #100/);
  assert.match(overlay.textContent, /Nearest to LIKE entrance/);
  assert.match(nextButton().textContent, /#100/);
  assert.equal(requests.length, 0);
  await click(overlay.querySelector('button'));
  assert.equal(activeLabel(), 'Point #100');
  assert.equal(document.querySelector('.field-start-point-overlay'), null);
  assert.equal(document.querySelector('.field-overlay[role="dialog"]'), null);
  assert.equal(requests.length, 1);
  assert.deepEqual(marks, []);
});

test('the screenshot route and Next follow 95, 100, 99, 94, 88 through to 46 without returning', async () => {
  const positions = [[95,330,255],[100,270,291],[99,270,357],[94,330,323],[88,390,357],
    [82,448,323],[76,507,357],[70,566,323],[58,686,323],[46,802,323]];
  rows = positions.map(([point_num, x, y]) => ({ ...fixtures[0], point_num, assignment_point_id: point_num,
    latitude: 10.78014 - (y - 255) * 0.0000001, longitude: 122.62707 + (x - 270) * 0.0000001 }));
  await render(); await selectPoint(95); await click(button('Navigate to this point'));
  const firstLeg = orangeLines()[1].getLatLngs();
  assert.equal(firstLeg[1].lng, rows[1].longitude);
  assert.equal(firstLeg[1].lat, rows[1].latitude);
  for (const [num] of positions.slice(1)) {
    await click(nextButton());
    assert.equal(activeLabel(), `Point #${num}`);
  }
  assert.equal(nextButton().disabled, true);
  assert.equal(requests.length, 1);
  assert.deepEqual(marks, []);
});

test('recording planting keeps navigation active so the next point needs only one tap', async () => {
  await render(); await click(nextButton());
  await selectPoint(100);
  await click(button('Mark as Planted'));
  await click(button('Yes, mark as planted'));
  assert.equal(activeLabel(), 'Point #100');
  assert.deepEqual(marks, [{ id: 1, status: 'completed' }]);
  await click(button('Continue'));
  await click(nextButton());
  assert.equal(activeLabel(), 'Point #99');
  assert.equal(document.querySelector('.field-overlay[role="dialog"]'), null);
  assert.equal(watches.size, 1);
});

test('completed and eroded future points disappear from the route after assignments refresh', async () => {
  await render(); await click(nextButton());
  rows[1].assignment_status = 'completed'; rows[2].inside_eroded_zone = true;
  await click(document.querySelector('[aria-label="Refresh assignments"]'));
  assert.match(nextButton().textContent, /#97/);
  assert.equal(orangeLines().length, 2);
  await click(nextButton());
  assert.equal(activeLabel(), 'Point #97');
});

test('stopping while Next waits for GPS cannot restart navigation after the late fix arrives', async () => {
  await render(); await click(nextButton());
  let delayed;
  locate = success => { delayed = success; };
  await click(nextButton());
  await click(button('Stop navigation'));
  assert.equal(activeLabel(), undefined);
  await act(async () => delayed(gps));
  assert.equal(activeLabel(), undefined);
  assert.equal(watches.size, 0);
  assert.equal(nextButton().disabled, false);
});

test('a failed GPS fix on Next leaves the active route available for retry', async () => {
  await render(); await click(nextButton());
  locate = (success, error) => error({ message: 'Location unavailable.' });
  await click(nextButton());
  assert.equal(activeLabel(), 'Point #100');
  assert.match(document.querySelector('[role="alert"]').textContent, /Location unavailable/);
  assert.match(nextButton().textContent, /#99/);
  assert.equal(nextButton().disabled, false);
  assert.equal(watches.size, 1);
});
