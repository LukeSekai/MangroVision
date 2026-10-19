import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

let server, dom, root, createRoot, FieldNavigationNotice, timers, stopped, retried;
const globals = new Map();
const route = { pointNum: 95, entrance: [10.77999592, 122.625672], route_source: 'site_route' };

before(async () => {
  dom = new JSDOM('<div id="root"></div>', { pretendToBeVisual: true });
  for (const [key, value] of Object.entries({ window: dom.window, document: dom.window.document, IS_REACT_ACT_ENVIRONMENT: true })) {
    globals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  ({ createRoot } = await import('react-dom/client'));
  server = await createServer({ configFile: false, appType: 'custom',
    server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    cacheDir: 'node_modules/.vite-field-navigation-tests', optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ default: FieldNavigationNotice } = await server.ssrLoadModule('/src/components/FieldNavigationNotice.jsx'));
  window.setTimeout = (callback, delay) => { const id = timers.size + 1; timers.set(id, { callback, delay }); return id; };
  window.clearTimeout = (id) => timers.delete(id);
});

after(async () => {
  await server?.close(); dom?.window.close();
  for (const [key, descriptor] of globals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
  }
});
beforeEach(() => { timers = new Map(); stopped = 0; retried = 0; root = createRoot(document.getElementById('root')); });
afterEach(async () => { await act(async () => root.unmount()); assert.equal(timers.size, 0); });

async function render(overrides = {}) {
  await act(async () => root.render(createElement(FieldNavigationNotice, { route, onStop: () => { stopped++; }, ...overrides })));
}

test('the path hint closes after seven seconds while navigation stays active', async () => {
  await render();
  assert.match(document.querySelector('[role="status"]').textContent, /Follow the white bridge.*Point #95/);
  assert.doesNotMatch(document.body.textContent, /Directions to planting point|Google Maps|km|bearing/);
  const timer = [...timers.values()][0];
  assert.equal(timer.delay, 7000);
  await act(async () => timer.callback());
  assert.equal(document.querySelector('.field-navigation-hint'), null);
  assert.match(document.querySelector('[aria-label="Active navigation"]').textContent, /Point #95/);
  assert.equal(stopped, 0);
});

test('dismissing the hint preserves navigation and the stop control still works', async () => {
  await render();
  await act(async () => document.querySelector('[aria-label="Dismiss path hint"]').click());
  assert.equal(document.querySelector('.field-navigation-hint'), null);
  assert.equal(stopped, 0);
  await act(async () => document.querySelector('.field-navigation-control button').click());
  assert.equal(stopped, 1);
});

test('GPS interruptions remain visible after the optional hint closes', async () => {
  await render({ gpsError: 'GPS updates paused.' });
  await act(async () => [...timers.values()][0].callback());
  assert.equal(document.querySelector('[role="status"]').textContent, 'GPS updates paused.');
});

test('the road failure reason and retry stay visible after the path hint closes', async () => {
  await render({ route: { ...route, route_source: 'partial_route', road_route_error: {
    kind: 'routing_not_configured', message: 'Road navigation is not configured. Contact LGU staff.',
  } }, onRetry: () => { retried++; } });
  await act(async () => [...timers.values()][0].callback());
  assert.equal(document.querySelector('.field-navigation-hint'), null);
  assert.match(document.querySelector('[role="alert"]').textContent, /Road navigation is not configured.*Only the mapped site approach is shown/);
  await act(async () => document.querySelector('.field-navigation-road-error button').click());
  assert.equal(retried, 1);
  assert.equal(stopped, 0);
});

test('legacy unavailable responses still explain the missing road and disable retry while busy', async () => {
  await render({ route: { pointNum: 1, route_source: 'point_guidance' }, onRetry: () => { retried++; }, retryBusy: true });
  assert.match(document.querySelector('[role="alert"]').textContent, /Road directions.*unavailable.*No road route to this point is shown/);
  const retry = document.querySelector('.field-navigation-road-error button');
  assert.equal(retry.disabled, true);
  await act(async () => retry.click());
  assert.equal(retried, 0);
});
