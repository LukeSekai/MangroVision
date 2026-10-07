import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement } from 'react';
import { MemoryRouter, Route, useNavigate } from 'react-router-dom';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

let server, dom, root, createRoot, RetainedRoutes, Dashboard, ActivityFeed, Scheduling, MapAnalytics, useMapStore;
let requests, scheduleRows, mapPoints;
const originalGlobals = new Map();
const rows = [1, 2].map((id) => ({
  analysis_id: id, analysis_number: id, image_name: `Analysis ${id}`,
  total_area_m2: 1000, plantable_area_m2: 600, danger_area_m2: 200,
  canopy_coverage_pct: 20,
}));

before(async () => {
  dom = new JSDOM('<div id="root"></div>', { url: 'http://workspace.example', pretendToBeVisual: true });
  const globals = {
    window: dom.window, document: dom.window.document, localStorage: dom.window.localStorage,
    Event: dom.window.Event, IS_REACT_ACT_ENVIRONMENT: true,
    requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
    cancelAnimationFrame: dom.window.cancelAnimationFrame.bind(dom.window),
    ResizeObserver: class { observe() {} disconnect() {} unobserve() {} },
  };
  for (const [key, value] of Object.entries(globals)) {
    originalGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  originalGlobals.set('fetch', Object.getOwnPropertyDescriptor(globalThis, 'fetch'));
  window.fetch = async (input, options = {}) => {
    const url = new URL(input instanceof Request ? input.url : input, window.location.origin);
    requests.push(url.pathname + url.search);
    if (url.pathname === '/api/planters/map-points/1/death' && options.method === 'PATCH') {
      mapPoints = mapPoints.map((point) => ({ ...point, map_status: 'dead', death_at: '2026-10-07' }));
      return Response.json({ death_at: '2026-10-07' });
    }
    if (url.pathname === '/api/planters/map-points/1/restore') {
      mapPoints = mapPoints.map((point) => ({ ...point, map_status: 'planted', death_at: null }));
      return Response.json({ ok: true });
    }
    if (url.pathname === '/api/planters/map-points/1/reset-to-planned') {
      mapPoints = mapPoints.map((point) => ({ ...point, map_status: 'unavailable', death_at: null, planting_status: 'planned' }));
      return Response.json({ ok: true });
    }
    if (url.pathname === '/api/planters/map-points') return Response.json(mapPoints);
    if (url.pathname === '/api/analyses/stats') return Response.json({ total_analyses: 1, points: mapPoints });
    if (url.pathname === '/api/dashboard/settings') return Response.json({ year: Number(url.searchParams.get('year')), annual_planting_target: 4000, min_survival_target_pct: 80 });
    if (url.pathname.startsWith('/api/dashboard/')) return Response.json({
      as_of: '2026-10-06T01:00:00Z', filter_options: { sites: [{ id: 1, name: 'Test Site' }] }, suitability: rows,
      lifecycle: mapPoints.map((point) => ({
        key: { planned: 'available', planted: 'planted_unverified' }[point.map_status] || point.map_status,
        value: 1,
      })),
    });
    if (url.pathname === '/api/activity/staff') return Response.json({ items: [{
      id: 1, summary: 'A retained activity record', created_at: '2026-10-06T01:00:00Z', actor_type: 'system',
    }] });
    if (url.pathname === '/api/planting-schedules') return Response.json({ schedules: scheduleRows, organizations: [], project_sites: [] });
    if (url.pathname === '/api/tides/forecast') return Response.json({ available: false, message: 'Forecast unavailable', events: [] });
    throw new Error(`Unexpected test request: ${url.pathname}`);
  };
  ({ createRoot } = await import('react-dom/client'));
  server = await createServer({ configFile: false, appType: 'custom',
    server: { middlewareMode: true, hmr: false, watch: null },
    cacheDir: 'node_modules/.vite-navigation-tests', optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ default: RetainedRoutes } = await server.ssrLoadModule('/src/components/RetainedRoutes.jsx'));
  ({ default: Dashboard } = await server.ssrLoadModule('/src/pages/Dashboard.jsx'));
  ({ default: ActivityFeed } = await server.ssrLoadModule('/src/components/ActivityFeed.jsx'));
  ({ default: Scheduling } = await server.ssrLoadModule('/src/pages/Scheduling.jsx'));
  ({ default: MapAnalytics } = await server.ssrLoadModule('/src/pages/MapAnalytics.jsx'));
  ({ useMapStore } = await server.ssrLoadModule('/src/stores/mapStore.js'));
  const { useAuthStore } = await server.ssrLoadModule('/src/stores/authStore.js');
  useAuthStore.setState({ token: 'cookie', hydrated: true, isAuthenticated: true });
  const { installSecureFetch } = await server.ssrLoadModule('/src/utils/secureFetch.js');
  installSecureFetch();
  globalThis.fetch = window.fetch;
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
  scheduleRows = [];
  mapPoints = [];
  window.dispatchEvent(new Event('mv:invalidate-reads'));
  root = createRoot(document.getElementById('root'));
});
afterEach(async () => { await act(async () => root.unmount()); });

function Navigation() {
  const navigate = useNavigate();
  return createElement('nav', null, ...['dashboard', 'activity', 'scheduling', 'map'].map((page) =>
    createElement('button', { key: page, 'data-page': page, onClick: () => navigate(`/${page}`) }, page)));
}

async function settle() {
  await act(async () => { await new Promise((resolve) => window.setTimeout(resolve, 25)); });
}

async function render(page = 'dashboard') {
  await act(async () => root.render(createElement(MemoryRouter, { initialEntries: [`/${page}`] },
    createElement(Navigation),
    createElement(RetainedRoutes, { paths: ['/dashboard', '/activity', '/scheduling', '/map'] },
      createElement(Route, { path: '/dashboard', element: createElement(Dashboard) }),
      createElement(Route, { path: '/activity', element: createElement(ActivityFeed) }),
      createElement(Route, { path: '/scheduling', element: createElement(Scheduling) }),
      createElement(Route, { path: '/map', element: createElement(MapAnalytics) }),
    ))));
  await settle();
}

async function click(selector) {
  const button = typeof selector === 'string' ? document.querySelector(selector) : selector;
  assert.ok(button, `Missing button: ${selector}`);
  await act(async () => button.click());
  await settle();
}

const count = (path) => requests.filter((url) => url.split('?')[0] === path).length;

test('map and donut pick up mutations made while their retained page is hidden', async () => {
  mapPoints = [{ id: 1, map_status: 'planted', planting_status: 'planted' }];
  await render('map');
  const statValue = (label) => [...document.querySelectorAll('.stat-card')]
    .find((card) => card.querySelector('.stat-label')?.textContent === label)
    ?.querySelector('.stat-value')?.textContent;
  const legend = () => document.querySelector('.dash-lifecycle-legend').textContent;
  assert.equal(statValue('Planted'), '1');
  await click('[data-page="dashboard"]');
  assert.match(legend(), /Planted/);
  await click('[data-page="activity"]');
  await act(async () => { await useMapStore.getState().markPointDead(1, 'other', ''); });
  await click('[data-page="dashboard"]');
  assert.match(legend(), /Dead/);
  assert.doesNotMatch(legend(), /Planted/);
  assert.equal(count('/api/dashboard/overview'), 2);
  await click('[data-page="map"]');
  assert.equal(statValue('Planted'), '0');
  assert.equal(statValue('Dead'), '1');
  assert.equal(statValue('Mapped Points'), '1');
  const requestsBeforeReturn = requests.length;
  await click('[data-page="dashboard"]');
  await click('[data-page="map"]');
  assert.equal(requests.length, requestsBeforeReturn);
});

test('recording death, restoring and releasing a point update map counts immediately', async () => {
  mapPoints = [{ id: 1, map_status: 'planted', planting_status: 'planted', eroded_unavailable: true }];
  await render('map');
  const statValue = (label) => [...document.querySelectorAll('.stat-card')]
    .find((card) => card.querySelector('.stat-label')?.textContent === label)
    ?.querySelector('.stat-value')?.textContent;
  assert.equal(statValue('Planted'), '1');
  await act(async () => { await useMapStore.getState().markPointDead(1, 'other', ''); });
  assert.equal(statValue('Dead'), '1');
  assert.equal(statValue('Planted'), '0');
  await act(async () => { await useMapStore.getState().restorePointToPlanted(1); });
  assert.equal(statValue('Dead'), '0');
  assert.equal(statValue('Planted'), '1');
  await act(async () => { await useMapStore.getState().resetPointToPlanned(1); });
  assert.equal(statValue('Planted'), '0');
  assert.equal(statValue('Unavailable'), '1');
  assert.equal(statValue('Mapped Points'), '1');
});

test('matching calendar activities open all organizations and edit the selected schedule', async () => {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: 'Asia/Manila', year: 'numeric', month: '2-digit', day: '2-digit',
  }).formatToParts(new Date());
  const dateParts = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  const date = `${dateParts.year}-${dateParts.month}-${dateParts.day}`;
  scheduleRows = [1, 2, 3].map((id) => ({
    id, organization_id: id, organization_name: `Organization ${id}`,
    project_site_id: id, project_site_name: `Planting area ${id}`,
    title: 'Tentative follow-up planting activity', date,
    start_at: `${date}T07:30:00+08:00`, end_at: `${date}T11:30:00+08:00`,
    start_time: '07:30', end_time: '11:30', status: 'tentative',
    expected_participants: 20, seedlings: 60, inspection_interval_days: 14,
  }));
  await render('scheduling');
  const cards = document.querySelectorAll('.schedule-calendar-event');
  assert.equal(cards.length, 1);
  assert.match(cards[0].textContent, /3 organizations/);
  await click(cards[0]);
  const organizations = document.querySelectorAll('.schedule-group-item');
  assert.equal(organizations.length, 3);
  organizations.forEach((item, index) => {
    assert.match(item.textContent, new RegExp(`Organization ${index + 1}`));
    assert.match(item.textContent, new RegExp(`Planting area ${index + 1}`));
  });
  await click(organizations[1].querySelector('.schedule-group-edit'));
  assert.equal(document.querySelector('.schedule-form input[list="schedule-organizations"]').value, 'Organization 2');
  assert.equal(document.querySelector('.schedule-form input[maxlength="180"]').value, 'Tentative follow-up planting activity');
  assert.equal(requests.filter((url) => !url.startsWith('/api/planting-schedules') && !url.startsWith('/api/tides/forecast')).length, 0);
});

test('dashboard tabs and sidebar navigation retain the selected analysis, filters and DOM', async () => {
  await render();
  await click('#dash-tab-sites');
  const dashboard = document.querySelector('main.dash');
  const siteFilter = document.querySelector('.dash-filters select');
  await act(async () => { siteFilter.value = '1'; siteFilter.dispatchEvent(new Event('change', { bubbles: true })); });
  await settle();
  const analysis = document.querySelector('#dash-panel-sites select');
  assert.ok(analysis);
  await act(async () => { analysis.value = '2'; analysis.dispatchEvent(new Event('change', { bubbles: true })); });
  assert.equal(analysis.value, '2');
  dashboard.scrollTop = 275;
  await click('#dash-tab-overview');
  const readsBeforeReturn = requests.length;
  await click('#dash-tab-sites');
  assert.equal(requests.length, readsBeforeReturn);
  assert.equal(document.querySelector('#dash-panel-sites select').value, '2');
  await click('[data-page="activity"]');
  assert.equal(count('/api/activity/staff'), 1);
  const beforeReturn = requests.length;
  await click('[data-page="dashboard"]');
  assert.equal(requests.length, beforeReturn);
  assert.equal(document.querySelector('main.dash'), dashboard);
  assert.equal(dashboard.scrollTop, 275);
  assert.equal(document.querySelector('#dash-tab-sites').getAttribute('aria-selected'), 'true');
  assert.equal(document.querySelector('.dash-filters select').value, '1');
  assert.equal(document.querySelector('#dash-panel-sites select').value, '2');
});

test('browser focus, visibility and opening Planting Goals do not reload the dashboard', async () => {
  await render();
  const beforeFocus = requests.length;
  await act(async () => {
    window.dispatchEvent(new Event('focus'));
    document.dispatchEvent(new Event('visibilitychange'));
  });
  await settle();
  assert.equal(requests.length, beforeFocus);
  assert.equal(count('/api/dashboard/overview'), 1);
  assert.equal(document.querySelector('.dash-refresh-button'), null);
  await click('.dash-goals-button');
  assert.equal(document.querySelector('.dash-goals-button').getAttribute('aria-expanded'), 'true');
  assert.equal(document.querySelector('#dashboard-planting-goals').style.display, '');
  assert.equal(document.querySelector('#dash-panel-overview').style.display, '');
  assert.equal(count('/api/dashboard/overview'), 1);
  assert.equal(count('/api/dashboard/settings'), 1);
});

test('unsaved planting goals survive closing the panel, changing tabs and returning to the dashboard', async () => {
  await render();
  await click('#dash-tab-sites');
  await click('#dash-tab-overview');
  await click('.dash-goals-button');
  const year = document.querySelector('.dash-goals-year select');
  await act(async () => { year.value = '2025'; year.dispatchEvent(new Event('change', { bubbles: true })); });
  await settle();
  const target = document.querySelector('.dash-goals input');
  assert.equal(target.value, '4000');
  await act(async () => {
    const setValue = Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set;
    setValue.call(target, '4999');
    target.dispatchEvent(new Event('input', { bubbles: true }));
  });
  assert.equal(target.value, '4999');
  const beforeTabs = requests.length;
  await click('#dash-tab-sites');
  await click('.dash-goals-button');
  await click('#dash-tab-overview');
  await click('.dash-goals-button');
  assert.equal(requests.length, beforeTabs);
  assert.equal(document.querySelector('.dash-goals input').value, '4999');
  await click('[data-page="activity"]');
  const beforePageReturn = requests.length;
  await click('[data-page="dashboard"]');
  assert.equal(requests.length, beforePageReturn);
  assert.equal(document.querySelector('.dash-goals-year select').value, '2025');
  assert.equal(document.querySelector('.dash-goals input').value, '4999');
});

test('Activity keeps its loaded history on return and its manual Refresh loads again', async () => {
  await render('activity');
  const feed = document.querySelector('.activity-feed');
  assert.match(feed.textContent, /A retained activity record/);
  await click('[data-page="dashboard"]');
  await click('[data-page="activity"]');
  assert.equal(document.querySelector('.activity-feed'), feed);
  assert.equal(count('/api/activity/staff'), 1);
  await click('.activity-feed-toolbar button');
  assert.equal(count('/api/activity/staff'), 2);
});

test('Scheduling retains its calendar and tide forecast when returning, and Refresh reloads both', async () => {
  await render('scheduling');
  assert.equal(count('/api/planting-schedules'), 1);
  assert.equal(count('/api/tides/forecast'), 1);
  await click('[data-page="dashboard"]');
  await click('[data-page="scheduling"]');
  await act(async () => {
    window.dispatchEvent(new Event('focus'));
    document.dispatchEvent(new Event('visibilitychange'));
  });
  await settle();
  assert.equal(count('/api/planting-schedules'), 1);
  assert.equal(count('/api/tides/forecast'), 1);
  await click('.schedule-refresh');
  assert.equal(count('/api/planting-schedules'), 2);
  assert.equal(count('/api/tides/forecast'), 2);
});
