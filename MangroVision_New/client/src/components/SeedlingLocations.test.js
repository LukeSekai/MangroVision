import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { cwd, pid } from 'node:process';
import { pathToFileURL } from 'node:url';
import { writeFile, unlink } from 'node:fs/promises';
import { resolve } from 'node:path';
import { act, createElement, useEffect, useState } from 'react';
import { JSDOM } from 'jsdom';
import { build } from 'vite';

let dom, root, createRoot, SeedlingLocations, Modal, latestSelection, requests, cancelled;
const bundlePath = resolve(`node_modules/.seedling-brush-${pid}.mjs`);
const maps = [];
const savedGlobals = new Map();
const points = Array.from({ length: 120 }, (_, index) => ({
  planting_event_id: index + 1, reference: 'Seedling ' + (index + 1),
  latitude: 10.001 + Math.floor(index / 12) * .000008,
  longitude: 122.001 + (index % 12) * .000008,
  selectable: true,
}));
points.push({ ...points[0], planting_event_id: 200, reference: 'Unavailable seedling',
  latitude: points[0].latitude + .000001, selectable: false });

before(async () => {
  dom = new JSDOM('<div id="root"></div>', { url: 'https://monitoring.example.test', pretendToBeVisual: true });
  dom.window.SVGSVGElement.prototype.createSVGRect = () => ({});
  Object.defineProperty(dom.window.HTMLElement.prototype, 'clientWidth', {
    get() { return this.classList.contains('seedling-location-map') ? 800 : 0; },
  });
  Object.defineProperty(dom.window.HTMLElement.prototype, 'clientHeight', {
    get() { return this.classList.contains('seedling-location-map') ? 340 : 0; },
  });
  const globals = {
    window: dom.window, document: dom.window.document, navigator: dom.window.navigator,
    HTMLElement: dom.window.HTMLElement, SVGElement: dom.window.SVGElement,
    IS_REACT_ACT_ENVIRONMENT: true,
    requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
    cancelAnimationFrame: dom.window.cancelAnimationFrame.bind(dom.window),
    ResizeObserver: class { observe() {} disconnect() {} },
    fetch: async (input, init = {}) => {
      requests.push({ path: input, method: init.method || 'GET' });
      return Response.json({ points });
    },
  };
  for (const [key, value] of Object.entries(globals)) {
    savedGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  const { default: leaflet } = await import(pathToFileURL(cwd() + '/node_modules/leaflet/dist/leaflet-src.js').href);
  leaflet.Map.addInitHook(function () { maps.push(this); });
  ({ createRoot } = await import('react-dom/client'));
  const bundle = await build({
    configFile: false, logLevel: 'silent',
    ssr: { external: ['react', 'react/jsx-runtime', 'react-dom', 'leaflet'] },
    build: { ssr: resolve('src/components/SeedlingLocations.jsx'), write: false, minify: false,
      rollupOptions: { output: { format: 'es' } },
    },
    plugins: [{ name: 'include-monitoring-dialog', enforce: 'pre', transform(code, id) {
      if (id.replaceAll('\\', '/').endsWith('/components/SeedlingLocations.jsx')) {
        return `${code}\nexport { default as TestModal } from './Modal.jsx';`;
      }
    } }],
  });
  await writeFile(bundlePath, bundle.output.find((item) => item.type === 'chunk' && item.isEntry).code);
  ({ default: SeedlingLocations, TestModal: Modal } = await import(pathToFileURL(bundlePath).href));
});

after(async () => {
  await unlink(bundlePath).catch(() => {});
  dom?.window.close();
  for (const [key, descriptor] of savedGlobals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
});

beforeEach(() => {
  maps.length = 0;
  requests = [];
  cancelled = 0;
  latestSelection = [];
  root = createRoot(document.getElementById('root'));
});
afterEach(async () => { await act(async () => root.unmount()); });

function cancelDialog() { cancelled += 1; }

function Editor({ initialSelected = [], modal = false, ...props }) {
  const [selected, setSelected] = useState(initialSelected);
  useEffect(() => { latestSelection = selected; }, [selected]);
  const editor = createElement(SeedlingLocations, { organizationId: 1, monitoredAt: '2026-10-08',
    selected, onChange: setSelected, maxSelected: 100, ...props });
  return modal ? createElement(Modal, { open: true, title: 'Monitoring visit',
    onCancel: cancelDialog }, editor) : editor;
}

function button(label) {
  return [...document.querySelectorAll('button')].find((element) => element.textContent === label);
}

function marker(label) { return document.querySelector('path[aria-label="' + label + '"]'); }
function map() { return maps.at(-1); }
function position(point) { return map().latLngToContainerPoint([point.latitude, point.longitude]); }
function pointer(type, point, options = {}) {
  const event = new dom.window.MouseEvent(type, { bubbles: true, cancelable: true,
    clientX: point.x, clientY: point.y, button: 0, ...options });
  Object.defineProperties(event, {
    pointerType: { value: options.pointerType || 'mouse' }, pointerId: { value: options.pointerId || 1 },
  });
  (options.target || map().getContainer()).dispatchEvent(event);
}

function mapClick(point, options = {}) {
  pointer('pointerdown', point, options);
  pointer('pointerup', point, options);
  pointer('click', point, options);
}

async function sweepAll() {
  await act(async () => {
    for (let row = 0; row < 10; row += 1) {
      pointer('pointermove', position(points[row * 12]));
      pointer('pointermove', position(points[row * 12 + 11]));
    }
  });
}

test('choosing the brush does not paint until a map click, then sweeps respect the death limit', async () => {
  await act(async () => root.render(createElement(Editor)));
  const firstMarker = marker('Seedling 1');
  await act(async () => button('Brush select').click());
  assert.equal(map().dragging.enabled(), false);
  assert.match(document.body.textContent, /Brush ready/);
  await sweepAll();
  assert.deepEqual(latestSelection, []);
  await act(async () => mapClick(position(points[0]), { target: firstMarker }));
  assert.match(document.body.textContent, /Brush active/);
  await sweepAll();
  assert.equal(latestSelection.length, 100);
  assert.equal(new Set(latestSelection).size, 100);
  assert.equal(latestSelection.includes(200), false);
  assert.match(document.body.textContent, /100 of 100 deaths identified/);
  assert.equal(marker('Seedling 1'), firstMarker, 'painting should preserve marker elements');
  assert.equal(firstMarker.getAttribute('aria-pressed'), 'true');
  assert.equal(marker('Unavailable seedling').getAttribute('aria-disabled'), 'true');
  await sweepAll();
  assert.equal(latestSelection.length, 100);
  assert.ok(requests.every((request) => request.method === 'GET'));
});

test('a second map click stops painting and another click starts a separate stroke', async () => {
  await act(async () => root.render(createElement(Editor)));
  await act(async () => button('Brush select').click());
  await act(async () => mapClick(position(points[0])));
  assert.deepEqual(latestSelection, [1]);
  await act(async () => mapClick(position(points[119])));
  assert.match(document.body.textContent, /Brush ready/);
  await act(async () => pointer('pointermove', position(points[60])));
  assert.deepEqual(latestSelection, [1], 'the stop click and later movement must not paint');
  await act(async () => mapClick(position(points[119])));
  assert.deepEqual(latestSelection, [1, 120], 'restarting must not paint the path across the paused gap');
});

test('brush erase removes crossed selections and click mode still makes individual corrections', async () => {
  await act(async () => root.render(createElement(Editor, { initialSelected: [1, 2, 120], maxSelected: 3 })));
  await act(async () => button('Brush erase').click());
  await act(async () => {
    pointer('pointermove', position(points[0]));
    pointer('pointermove', position(points[1]));
  });
  assert.deepEqual(latestSelection, [1, 2, 120]);
  await act(async () => {
    mapClick(position(points[0]));
    pointer('pointermove', position(points[1]));
  });
  assert.deepEqual(latestSelection, [120]);
  await act(async () => {
    mapClick(position(points[119]));
    pointer('pointermove', position(points[119]));
  });
  assert.deepEqual(latestSelection, [120], 'erase must stop without removing the point under the stop click');
  await act(async () => button('Move / click').click());
  assert.equal(map().dragging.enabled(), true);
  await act(async () => marker('Seedling 1').dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true })));
  assert.deepEqual(latestSelection, [120, 1]);
  await act(async () => marker('Seedling 1').dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: ' ', bubbles: true })));
  assert.deepEqual(latestSelection, [120]);
});

test('Escape finishes painting without closing the monitoring dialog', async () => {
  await act(async () => root.render(createElement(Editor, { modal: true })));
  await act(async () => button('Brush select').click());
  await act(async () => mapClick(position(points[0])));
  await act(async () => document.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
  assert.equal(button('Move / click').getAttribute('aria-pressed'), 'true');
  assert.equal(cancelled, 0);
  assert.equal(map().dragging.enabled(), true);
  assert.equal(document.querySelector('.seedling-selection-brush'), null);
  await act(async () => document.dispatchEvent(new dom.window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
  assert.equal(cancelled, 1);
});

test('touch painting requires an active drag and releases pointer capture when it finishes', async () => {
  await act(async () => root.render(createElement(Editor)));
  const captured = new Set();
  const container = map().getContainer();
  container.setPointerCapture = (id) => captured.add(id);
  container.hasPointerCapture = (id) => captured.has(id);
  container.releasePointerCapture = (id) => captured.delete(id);
  await act(async () => button('Brush select').click());
  await act(async () => pointer('pointermove', position(points[0]), { pointerType: 'touch' }));
  assert.deepEqual(latestSelection, []);
  await act(async () => {
    pointer('pointerdown', position(points[0]), { pointerType: 'touch' });
    pointer('pointermove', position(points[1]), { pointerType: 'touch' });
    pointer('pointerup', position(points[1]), { pointerType: 'touch' });
    pointer('click', position(points[1]), { pointerType: 'touch' });
    pointer('pointermove', position(points[119]), { pointerType: 'touch' });
    pointer('pointermove', position(points[119]));
  });
  assert.ok(latestSelection.includes(1) && latestSelection.includes(2));
  assert.equal(latestSelection.includes(120), false);
  assert.equal(latestSelection.includes(200), false);
  assert.equal(captured.size, 0);
  assert.match(document.body.textContent, /Brush ready/);
});

test('zero deaths, saving and read-only field sheets cannot paint new selections', async () => {
  await act(async () => root.render(createElement(Editor, { maxSelected: 0 })));
  assert.equal(button('Brush select').disabled, true);
  await act(async () => marker('Seedling 1').dispatchEvent(new dom.window.MouseEvent('click', { bubbles: true })));
  assert.deepEqual(latestSelection, []);
  await act(async () => root.render(createElement(Editor)));
  await act(async () => button('Brush select').click());
  await act(async () => mapClick(position(points[0])));
  const beforeSaving = [...latestSelection];
  await act(async () => root.render(createElement(Editor, { disabled: true })));
  assert.equal(map().dragging.enabled(), true);
  await act(async () => pointer('pointermove', position(points[0])));
  assert.deepEqual(latestSelection, beforeSaving);
  await act(async () => root.render(createElement(Editor, { readOnly: true })));
  assert.equal(button('Brush select'), undefined);
  assert.equal(marker('Seedling 1').hasAttribute('role'), false);
  assert.ok(button('Print visible area'));
});

test('moving over zoom controls does not paint or join separate brush strokes', async () => {
  await act(async () => root.render(createElement(Editor)));
  await act(async () => button('Brush select').click());
  await act(async () => {
    mapClick(position(points[0]));
    pointer('pointermove', position(points[60]), { target: document.querySelector('.leaflet-control-zoom-in') });
    mapClick(position(points[60]), { target: document.querySelector('.leaflet-control-attribution') });
    pointer('pointermove', position(points[119]));
  });
  assert.deepEqual(latestSelection, [1, 120]);
});

test('leaving the map breaks the sweep and losing focus pauses painting', async () => {
  await act(async () => root.render(createElement(Editor)));
  await act(async () => button('Brush select').click());
  await act(async () => {
    mapClick(position(points[0]));
    pointer('pointerleave', position(points[0]));
    pointer('pointermove', position(points[119]));
  });
  assert.deepEqual(latestSelection, [1, 120]);
  await act(async () => window.dispatchEvent(new dom.window.Event('blur')));
  await act(async () => pointer('pointermove', position(points[60])));
  assert.deepEqual(latestSelection, [1, 120]);
  assert.match(document.body.textContent, /Brush ready/);
});
