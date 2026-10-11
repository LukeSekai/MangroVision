import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement } from 'react';
import { MemoryRouter, Route, useLocation, useNavigate } from 'react-router-dom';
import { JSDOM } from 'jsdom';
import { build } from 'vite';
import { writeFile, unlink } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import process from 'node:process';

let dom, root, createRoot, RetainedRoutes, Dashboard, ActivityFeed, Scheduling, MapAnalytics, useMapStore;
let requests, scheduleRows, mapPoints, appointmentRows, approvals, useAuthStore;
let goalSaves, goalSaveFailure, savedGoalSettings;
let followUpRows;
const bundlePath = resolve(`node_modules/.retained-routes-${process.pid}.mjs`);
const originalGlobals = new Map();
const rows = [1, 2].map((id) => ({
  analysis_id: id, analysis_number: id, image_name: `Analysis ${id}`,
  total_area_m2: 1000, plantable_area_m2: 600, danger_area_m2: 200,
  canopy_coverage_pct: 20,
}));

before(async () => {
  dom = new JSDOM('<div id="root"></div>', { url: 'http://workspace.example', pretendToBeVisual: true });
  dom.window.HTMLElement.prototype.scrollIntoView = function () {};
  dom.window.HTMLElement.prototype.scrollTo = function () {};
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
    if (url.pathname === '/api/dashboard/settings') {
      if (options.method === 'PUT') {
        const settings = JSON.parse(options.body);
        goalSaves.push(settings);
        if (goalSaveFailure) return Response.json({ detail: 'Please try saving again.' }, { status: 503 });
        savedGoalSettings.set(settings.year, settings);
        return Response.json(settings);
      }
      const year = Number(url.searchParams.get('year'));
      return Response.json(savedGoalSettings.get(year) || { year, annual_planting_target: 4000, min_survival_target_pct: 80 });
    }
    if (url.pathname.startsWith('/api/dashboard/')) return Response.json({
      as_of: '2026-10-06T01:00:00Z', filter_options: { sites: [{ id: 1, name: 'Test Site' }] }, suitability: rows,
      kpis: { sites_requiring_attention: { value: followUpRows.filter((site) => site.reasons.length).length } },
      site_attention: followUpRows,
      lifecycle: mapPoints.map((point) => ({
        key: { planned: 'available', planted: 'planted_unverified' }[point.map_status] || point.map_status,
        value: 1,
      })),
    });
    if (url.pathname === '/api/activity/staff') return Response.json({ items: [{
      id: 1, summary: 'A retained activity record', created_at: '2026-10-06T01:00:00Z', actor_type: 'system',
    }] });
    if (url.pathname === '/api/planting-schedules') return Response.json({ schedules: scheduleRows, organizations: [], project_sites: [] });
    if (url.pathname === '/api/like-appointments/summary') return Response.json({ pending_count: appointmentRows.filter((row) => row.status === 'pending').length });
    if (url.pathname === '/api/like-appointments') return Response.json({ requests: appointmentRows });
    if (url.pathname.endsWith('/review') && url.pathname.startsWith('/api/like-appointments/')) {
      const decision = JSON.parse(options.body || '{}');
      approvals.push(decision);
      appointmentRows[0] = { ...appointmentRows[0], status: 'confirmed', schedule_id: 11, email_status: 'pending' };
      return Response.json({ status: 'confirmed' });
    }
    if (url.pathname === '/api/tides/forecast') return Response.json({ available: false, message: 'Forecast unavailable', events: [] });
    throw new Error(`Unexpected test request: ${url.pathname}`);
  };
  ({ createRoot } = await import('react-dom/client'));
  const bundle = await build({
    configFile: false, logLevel: 'silent',
    ssr: { external: ['react', 'react/jsx-runtime', 'react-dom', 'react-router-dom', 'zustand', 'recharts', 'leaflet'] },
    build: { ssr: resolve('src/components/RetainedRoutes.jsx'), write: false, minify: false,
      rollupOptions: { output: { format: 'es', inlineDynamicImports: true } },
    },
    plugins: [{ name: 'include-retained-workflows', enforce: 'pre', transform(code, id) {
      if (id.replaceAll('\\', '/').endsWith('/components/RetainedRoutes.jsx')) {
        return `${code}
export { default as Dashboard } from '../pages/Dashboard.jsx';
export { default as ActivityFeed } from './ActivityFeed.jsx';
export { default as Scheduling } from '../pages/Scheduling.jsx';
export { default as MapAnalytics } from '../pages/MapAnalytics.jsx';
export { useMapStore } from '../stores/mapStore.js';
export { useAuthStore } from '../stores/authStore.js';
export { installSecureFetch } from '../utils/secureFetch.js';`;
      }
    } }],
  });
  await writeFile(bundlePath, bundle.output.find((item) => item.type === 'chunk' && item.isEntry).code);
  const modules = await import(pathToFileURL(bundlePath).href);
  ({ default: RetainedRoutes, Dashboard, ActivityFeed, Scheduling, MapAnalytics, useMapStore, useAuthStore } = modules);
  useAuthStore.setState({ token: 'cookie', hydrated: true, isAuthenticated: true });
  const { installSecureFetch } = modules;
  installSecureFetch();
  globalThis.fetch = window.fetch;
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
  requests = [];
  scheduleRows = [];
  mapPoints = [];
  appointmentRows = [];
  approvals = [];
  goalSaves = [];
  goalSaveFailure = false;
  savedGoalSettings = new Map();
  followUpRows = [];
  useAuthStore.setState({ user: null });
  window.dispatchEvent(new Event('mv:invalidate-reads'));
  root = createRoot(document.getElementById('root'));
});
afterEach(async () => { await act(async () => root.unmount()); });

function Navigation() {
  const navigate = useNavigate();
  const location = useLocation();
  return createElement('nav', null, ...['dashboard', 'activity', 'scheduling', 'map'].map((page) =>
    createElement('button', { key: page, 'data-page': page, onClick: () => navigate(`/${page}`) }, page)),
    createElement('output', { 'data-location': true }, location.pathname + location.search));
}

async function settle() {
  await act(async () => { await new Promise((resolve) => window.setTimeout(resolve, 25)); });
}

async function render(page = 'dashboard') {
  await act(async () => root.render(createElement(MemoryRouter, { initialEntries: [`/${page}`] },
    createElement(Navigation),
    createElement(RetainedRoutes, { paths: ['/dashboard', '/activity', '/scheduling', '/map', '/monitoring'] },
      createElement(Route, { path: '/dashboard', element: createElement(Dashboard) }),
      createElement(Route, { path: '/activity', element: createElement(ActivityFeed) }),
      createElement(Route, { path: '/scheduling', element: createElement(Scheduling) }),
      createElement(Route, { path: '/map', element: createElement(MapAnalytics) }),
      createElement(Route, { path: '/monitoring', element: createElement('p', null, 'Monitoring destination') }),
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

test('follow-up card lists affected sites and links to their exact map and monitoring context', async () => {
  followUpRows = [
    { site_id: 1, site_name: 'North planting site', reasons: ['overdue_inspections', 'survival_below_target'], overdue_inspections: 4, survival_rate_pct: 55 },
    { site_id: 2, site_name: 'Warning site', reasons: ['warning_exposure'], warning_points: 12 },
    { site_id: 3, site_name: 'Healthy site', reasons: [] },
  ];
  await render();
  const requestCount = requests.length;
  await click('.dash-kpi-action');
  const dialog = document.querySelector('.dash-follow-up-dialog');
  assert.ok(dialog);
  for (const text of [/North planting site/, /Monitoring overdue.*4 pending inspections/s, /Survival below target.*55%/s]) {
    assert.match(dialog.textContent, text);
  }
  assert.match(dialog.textContent, /Warning zone exposure.*12 planting points/s);
  assert.doesNotMatch(dialog.textContent, /Healthy site/);
  assert.equal(dialog.querySelectorAll('.dash-follow-up-site').length, 2);
  assert.equal(requests.length, requestCount, 'Opening the list uses the already loaded reporting scope');
  await click(dialog.querySelector('[aria-label="View Warning site on the map"]'));
  assert.equal(document.querySelector('[data-location]').textContent, '/map?project_site_id=2&focus=risk_areas');
  assert.equal(document.querySelector('.dash-follow-up-dialog'), null);
  await click('[data-page="dashboard"]');
  assert.equal(document.querySelector('.dash-follow-up-dialog'), null);
  await click('.dash-kpi-action');
  await click('[aria-label="Open Monitoring for North planting site"]');
  assert.equal(document.querySelector('[data-location]').textContent, '/monitoring?project_site_id=1');
  assert.equal(document.querySelector('.dash-follow-up-dialog'), null);
  assert.equal(goalSaves.length, 0);
});

test('zero follow-up sites has an honest empty state and Escape restores focus to the card', async () => {
  await render();
  const button = document.querySelector('.dash-kpi-action');
  button.focus();
  await click(button);
  assert.match(document.querySelector('.dash-follow-up-dialog').textContent, /No sites need follow-up for the selected filters/);
  assert.equal(document.querySelector('.dash-follow-up-dialog a'), null);
  await act(async () => document.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
  assert.equal(document.querySelector('.dash-follow-up-dialog'), null);
  assert.equal(document.activeElement, button);
});

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
  assert.equal(requests.filter((url) => !url.startsWith('/api/planting-schedules') && !url.startsWith('/api/tides/forecast') && !url.startsWith('/api/like-appointments')).length, 0);
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

test('planting goals appear first, close after saving and reopen with saved values', async () => {
  useAuthStore.setState({ user: { id: 1, role: 'lgu' } });
  await render();
  await click('.dash-goals-button');
  const goals = document.querySelector('#dashboard-planting-goals');
  const appointmentNotice = document.querySelector('.appointment-notice');
  const nextActions = document.querySelector('.dash > .next-actions');
  const follows = window.Node.DOCUMENT_POSITION_FOLLOWING;
  assert.ok(goals.compareDocumentPosition(appointmentNotice) & follows);
  assert.ok(appointmentNotice.compareDocumentPosition(nextActions) & follows);
  const year = document.querySelector('.dash-goals-year select');
  await act(async () => { year.value = '2025'; year.dispatchEvent(new Event('change', { bubbles: true })); });
  await settle();
  const target = document.querySelector('.dash-goals input');
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(target, '4999');
    target.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await click('.dash-goals button[type="submit"]');
  assert.deepEqual(goalSaves, [{ year: 2025, annual_planting_target: 4999, min_survival_target_pct: 80 }]);
  assert.equal(document.querySelector('.dash-goals-button').getAttribute('aria-expanded'), 'false');
  assert.equal(goals.style.display, 'none');
  assert.match(document.querySelector('.dash > [role="status"]').textContent, /2025 were saved/);
  assert.equal(document.activeElement, document.querySelector('.dash-goals-button'));
  await click('.dash-goals-button');
  assert.equal(goals.style.display, '');
  assert.equal(document.querySelector('.dash-goals input').value, '4999');
  assert.equal(document.querySelector('.dash-goals-notice'), null);
});

test('a failed planting goals save keeps the panel open and preserves the draft', async () => {
  goalSaveFailure = true;
  await render();
  await click('.dash-goals-button');
  const target = document.querySelector('.dash-goals input');
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(target, '4999');
    target.dispatchEvent(new Event('input', { bubbles: true }));
  });
  await click('.dash-goals button[type="submit"]');
  assert.equal(goalSaves.length, 1);
  assert.equal(document.querySelector('.dash-goals-button').getAttribute('aria-expanded'), 'true');
  assert.equal(document.querySelector('#dashboard-planting-goals').style.display, '');
  assert.equal(document.querySelector('.dash-goals input').value, '4999');
  assert.match(document.querySelector('.dash-goals').textContent, /Planting goals not saved/);
  assert.equal(document.querySelector('.dash-goals-notice'), null);
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

test('recorded planting locks planning inputs and deletion but permits notes and progress', async () => {
  useAuthStore.setState({ user: { id: 1, role: 'lgu' } });
  scheduleRows = [{ id: 12, title: 'Demo planting', organization: 'Demo group', organization_id: 4,
    start_at: '2090-01-01T08:00:00+08:00', end_at: '2090-01-01T10:00:00+08:00',
    date: '2090-01-01', start_time: '08:00', end_time: '10:00', inspection_interval_days: 14,
    status: 'confirmed', source: 'staff', appointment_type: 'tree_planting', planting_started: true,
    recorded_planting_count: 1, linked_assignment_count: 1, edit_lock_reason: 'Planting recorded.' }];
  await render('scheduling');
  const view = [...document.querySelectorAll('button')].find((button) => button.textContent === 'List');
  await click(view);
  const remove = [...document.querySelectorAll('button')].find((button) => button.textContent === 'Delete');
  assert.ok(remove.disabled);
  await click([...document.querySelectorAll('button')].find((button) => button.textContent === 'Update notes / status'));
  const form = document.querySelector('.schedule-form');
  for (const field of ['organizationName', 'title', 'date', 'startTime', 'endTime', 'expectedParticipants', 'seedlings']) {
    const input = [...form.querySelectorAll('input')].find((element) => element.name === field);
    assert.ok(input?.disabled, `${field} must be locked`);
  }
  assert.equal(form.querySelector('textarea').disabled, false);
  const status = form.querySelector('select');
  assert.deepEqual([...status.options].map((option) => option.value), ['confirmed', 'in_progress', 'completed']);
  assert.match(form.textContent, /Planning details are locked/);
});

test('LGU dashboard button opens pending website requests and approval stays in Scheduling', async () => {
  useAuthStore.setState({ user: { id: 1, role: 'lgu' } });
  const start = new Date(Date.now() + 3 * 86400000);
  start.setUTCHours(0, 0, 0, 0);
  appointmentRows = [{ id: 1, reference: 'LIKE-TEST', organization: 'Test school', contact_name: 'Coordinator',
    phone: '09123456789', email: 'test@example.org', appointment_type: 'field_visit', title: 'Field visit',
    start_at: start.toISOString(), end_at: new Date(start.getTime() + 7200000).toISOString(), participants: 20, status: 'pending' }];
  await render();
  const notice = document.querySelector('.appointment-notice');
  assert.match(notice.textContent, /1 pending request/);
  assert.equal(notice.getAttribute('href'), '/scheduling?requests=pending');
  assert.equal(document.querySelector('.website-review'), null);
  await click(notice);
  assert.match(document.querySelector('#website-requests').textContent, /LIKE-TEST/);
  assert.equal(document.activeElement.id, 'website-requests');
  await click('.website-review-button');
  assert.match(document.querySelector('.website-request-advice').textContent, /staff availability/);
  assert.equal(document.querySelector('.website-request-advice .planting-badge'), null);
  await click('input[name="contacted"]');
  const confirm = [...document.querySelectorAll('.modal-card button')].find((button) => button.textContent === 'Confirm and create schedule');
  await click(confirm);
  assert.equal(approvals.length, 1);
  assert.equal(approvals[0].email, 'test@example.org');
  assert.equal(approvals[0].contacted, true);
  assert.match(document.querySelector('.website-request-notice').textContent, /email is queued/);
  await click('[data-page="dashboard"]');
  assert.match(document.querySelector('.appointment-notice').textContent, /0 pending requests/);
  await click('.appointment-notice');
  assert.equal(document.querySelector('.website-request-toolbar .is-active').textContent, 'Pending requests');
});
