import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement, useEffect, useState } from 'react';
import { createRoot } from 'react-dom/client';
import { MemoryRouter, Route, useLocation, useNavigate } from 'react-router-dom';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

let server, dom, root, MapLink, MapReturnNavigation, MonitoringHeader, RetainedRoutes, PlantingMapWorkspace, useMapNavigation;
let currentLocation;
const previous = new Map();

before(async () => {
  server = await createServer({ configFile: false, appType: 'custom',
    server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ default: MapLink } = await server.ssrLoadModule('/src/components/MapLink.jsx'));
  ({ default: MapReturnNavigation } = await server.ssrLoadModule('/src/components/MapReturnNavigation.jsx'));
  ({ default: MonitoringHeader } = await server.ssrLoadModule('/src/components/MonitoringHeader.jsx'));
  ({ default: RetainedRoutes } = await server.ssrLoadModule('/src/components/RetainedRoutes.jsx'));
  ({ default: PlantingMapWorkspace } = await server.ssrLoadModule('/src/components/PlantingMapWorkspace.jsx'));
  ({ default: useMapNavigation } = await server.ssrLoadModule('/src/utils/useMapNavigation.js'));
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
  root = createRoot(document.getElementById('root'));
});
afterEach(async () => {
  await act(async () => root.unmount()); dom.window.close();
  for (const [key, descriptor] of previous) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
  }
  previous.clear();
});

function Source() {
  const [filter, setFilter] = useState('All sites');
  const navigateMap = useMapNavigation();
  return createElement('section', null,
    createElement('button', { onClick: () => setFilter('Coastal site') }, 'Choose site'),
    createElement('output', { 'data-filter': true }, filter),
    createElement(MapLink, { to: '/map?project_site_id=12&focus=site_points', state: { mapFocusSite: { id: 12 } } }, 'View site on map'),
    createElement(MapLink, { to: '/monitoring/map' }, 'Show monitoring map'),
    createElement('button', { onClick: () => navigateMap('/map') }, 'View location'));
}
function History() {
  const location = useLocation();
  useEffect(() => { currentLocation = location; }, [location]);
  const navigate = useNavigate();
  return createElement('div', null,
    createElement('button', { onClick: () => navigate(-1) }, 'Browser back'),
    createElement('button', { onClick: () => navigate(1) }, 'Browser forward'),
    createElement('button', { onClick: () => navigate('/map') }, 'Planting Map sidebar'));
}
async function mount(entry = '/dashboard', header = false) {
  await act(async () => root.render(createElement(MemoryRouter, { initialEntries: [entry] },
    createElement(History), createElement(MapReturnNavigation),
    header ? createElement(MonitoringHeader, { title: 'Page title', description: 'Page description' },
      createElement('button', null, 'Download Monitoring Report'))
      : createElement(PlantingMapWorkspace, null,
        createElement(RetainedRoutes, { paths: ['/dashboard', '/scheduling', '/organizations', '/monitoring', '/monitoring/replanting', '/map', '/zones', '/monitoring/map'] },
          ...['/dashboard', '/scheduling', '/organizations', '/monitoring', '/monitoring/replanting'].map((path) =>
            createElement(Route, { key: path, path, element: createElement(Source) })),
          createElement(Route, { path: '/map', element: createElement('p', null, 'Map view') }),
          createElement(Route, { path: '/zones', element: createElement('p', null, 'Zone editor') }),
          createElement(Route, { path: '/monitoring/map', element: createElement('p', null, 'Monitoring map') }))))));
}
async function click(text) {
  const element = [...document.querySelectorAll('a, button')].find((item) => item.textContent.trim() === text && !item.closest('[style*="display: none"]'));
  assert.ok(element, `Missing action: ${text}`);
  await act(async () => element.click());
}

test('return restores the source URL, router state and retained filter, including after browser navigation', async () => {
  await mount({ pathname: '/monitoring', search: '?project_site_id=12&tab=overdue', hash: '#visits', state: { selectedVisit: 7 } });
  await click('Choose site');
  const retainedOutput = document.querySelector('[data-filter]');
  await click('View site on map');
  assert.equal(currentLocation.search, '?project_site_id=12&focus=site_points');
  assert.deepEqual(currentLocation.state.mapFocusSite, { id: 12 });
  assert.equal(document.querySelectorAll('.map-return-navigation a').length, 1);
  await click('Back to Monitoring');
  assert.equal(currentLocation.search, '?project_site_id=12&tab=overdue');
  assert.equal(currentLocation.hash, '#visits');
  assert.deepEqual(currentLocation.state, { selectedVisit: 7 });
  assert.equal(document.querySelector('[data-filter]'), retainedOutput);
  assert.equal(retainedOutput.textContent, 'Coastal site');
  await click('Browser back');
  assert.equal(currentLocation.pathname, '/map');
  assert.match(document.querySelector('.map-return-navigation').textContent, /Back to Monitoring/);
  await click('Browser forward');
  assert.equal(currentLocation.pathname, '/monitoring');
});

test('Dashboard, Scheduling, Organizations and LGU links identify the source and persist across map tools', async () => {
  // Exercise the real Link wrapper for each distinct source instead of guessing history length.
  for (const [pathname, label] of [['/dashboard', 'Dashboard'], ['/scheduling', 'Scheduling'], ['/organizations', 'Organizations'], ['/monitoring/replanting', 'LGU Replanting']]) {
    await act(async () => root.unmount()); root = createRoot(document.getElementById('root'));
    await mount(pathname); await click('View site on map');
    assert.match(document.querySelector('.map-return-navigation').textContent, new RegExp(`Back to ${label}`));
    await click('Zone Editor');
    assert.equal(currentLocation.pathname, '/zones');
    await click(`Back to ${label}`);
    assert.equal(currentLocation.pathname, pathname);
  }
});

test('LGU location action and Show map both return to the LGU queue', async () => {
  await mount('/monitoring/replanting');
  await click('View location');
  await click('Back to LGU Replanting');
  await click('Show monitoring map');
  assert.equal(currentLocation.pathname, '/monitoring/map');
  await click('Back to LGU Replanting');
  assert.equal(currentLocation.pathname, '/monitoring/replanting');
});

test('direct map opens have no stale source, and direct monitoring maps have a safe return', async () => {
  await mount('/dashboard'); await click('View site on map'); await click('Planting Map sidebar');
  assert.equal(document.querySelector('.map-return-navigation'), null);
});

test('refresh restores the map return target from router state and invalid destinations are ignored', async () => {
  await mount({ pathname: '/map', state: { mapReturnTo: { pathname: '/scheduling', search: '?status=completed', state: { calendarMonth: '2026-09' } } } });
  await click('Back to Scheduling');
  assert.equal(currentLocation.search, '?status=completed');
  assert.deepEqual(currentLocation.state, { calendarMonth: '2026-09' });
});

test('direct monitoring map has a single return and unsafe source state cannot create an external link', async () => {
  await mount({ pathname: '/monitoring/map', state: { mapReturnTo: { pathname: 'https://example.com' } } });
  assert.equal(document.querySelectorAll('.map-return-navigation a').length, 1);
  await click('Back to Monitoring');
  assert.equal(currentLocation.pathname, '/monitoring');
});

test('both Monitoring headers use identical navigation and map action styling with the right active page', async () => {
  await mount('/monitoring', true);
  const shape = document.querySelector('.org-monitoring-page-header').innerHTML;
  assert.equal(document.querySelector('.monitoring-section-navigation [aria-current="page"]').textContent, 'Monitoring visits');
  assert.equal(document.querySelector('.org-monitoring-page-actions a.is-primary').textContent.trim(), 'Show map');
  await click('LGU Replanting');
  assert.equal(document.querySelector('.org-monitoring-page-header').innerHTML, shape);
  assert.equal(document.querySelector('.monitoring-section-navigation [aria-current="page"]').textContent, 'LGU Replanting');
  await click('Show map');
  assert.equal(currentLocation.state.mapReturnTo.pathname, '/monitoring/replanting');
});
