// Server rendering verifies content and fixed heights, not browser appearance.
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { createRoot } from 'react-dom/client';
import { MemoryRouter } from 'react-router-dom';
import { JSDOM } from 'jsdom';
import { build } from 'vite';
import { writeFile, unlink } from 'node:fs/promises';
import { resolve } from 'node:path';
import { pathToFileURL } from 'node:url';
import process from 'node:process';
import { selectAnalysis, analysisPieData } from '../utils/dashboardAnalyses.js';
import { dashboardSectionsForTab } from '../utils/dashboardLoading.js';

const rows = Array.from({ length: 10 }, (_, i) => ({
  analysis_id: 130 + i * 2,
  analysis_number: i + 1,
  image_name: `Analysis ${i + 1}`,
  site_name: 'A project site with a long descriptive name',
  analyzed_at: '2026-09-10T10:00:00Z',
  total_area_m2: 1000, plantable_area_m2: 600, danger_area_m2: 200,
  canopy_coverage_pct: i === 5 ? null : 20,
}));
const bundlePath = resolve(`node_modules/.dashboard-health-${process.pid}.mjs`);
let previousStorage;
let SitesTab;
let Dashboard, OverviewTab, OperationsTab, EcologyTab, PlantingGoalsForm, PlantingGoalsPanel;

before(async () => {
  previousStorage = Object.getOwnPropertyDescriptor(globalThis, 'localStorage');
  Object.defineProperty(globalThis, 'localStorage', {
    configurable: true, value: { getItem: () => null, removeItem: () => {} },
  });
  const bundle = await build({
    configFile: false, logLevel: 'silent',
    ssr: { external: ['react', 'react/jsx-runtime', 'react-dom', 'react-router-dom', 'zustand', 'recharts'] },
    build: { ssr: resolve('src/pages/Dashboard.jsx'), write: false, minify: false,
      rollupOptions: { output: { format: 'es' } },
    },
    plugins: [{ name: 'expose-dashboard-tabs', enforce: 'pre', transform(code, id) {
      if (id.replaceAll('\\', '/').endsWith('/pages/Dashboard.jsx')) return `${code}\nexport { SitesTab, OverviewTab, OperationsTab, EcologyTab, PlantingGoalsForm, PlantingGoalsPanel };`;
    } }],
  });
  await writeFile(bundlePath, bundle.output.find((item) => item.type === 'chunk' && item.isEntry).code);
  const module = await import(pathToFileURL(bundlePath).href);
  ({ SitesTab, OverviewTab, OperationsTab, EcologyTab, PlantingGoalsForm, PlantingGoalsPanel } = module);
  Dashboard = module.default;
});
after(async () => {
  await unlink(bundlePath).catch(() => {});
  if (previousStorage) Object.defineProperty(globalThis, 'localStorage', previousStorage);
  else delete globalThis.localStorage;
});

test('analysis selection uses saved names and falls back after filtering', () => {
  const first = selectAnalysis([...rows].reverse());
  const selected = selectAnalysis(rows, String(rows[6].analysis_id));
  assert.equal(first.selected.label, 'Analysis 1');
  assert.equal(selected.selected.label, 'Analysis 7');
  assert.equal(first.analyses.length, 10);
  assert.equal(selectAnalysis(rows.slice(0, 2), rows[6].analysis_id).selected.label, 'Analysis 1');
  assert.equal(selectAnalysis([]).selected, null);
});

test('area pie includes the remaining area and never adds canopy twice', () => {
  const { areaSlices, canopySlices } = analysisPieData(rows[0]);
  assert.deepEqual(areaSlices.map((row) => row.value), [600, 200, 200]);
  assert.equal(areaSlices.reduce((sum, row) => sum + row.value, 0), 1000);
  assert.deepEqual(canopySlices.map((row) => row.value), [20, 80]);
});

test('missing and invalid measurements are not turned into valid pies', () => {
  assert.deepEqual(analysisPieData(rows[5]).canopySlices, []);
  assert.deepEqual(analysisPieData({ ...rows[0], total_area_m2: 0 }).areaSlices, []);
  assert.deepEqual(analysisPieData({ ...rows[0], plantable_area_m2: null }).areaSlices, []);
  assert.deepEqual(analysisPieData({ ...rows[0], danger_area_m2: 900 }).areaSlices, []);
  assert.deepEqual(analysisPieData({ ...rows[0], canopy_coverage_pct: 101 }).canopySlices, []);
});

test('zero and full canopy coverage remain valid measurements', () => {
  assert.deepEqual(analysisPieData({ ...rows[0], canopy_coverage_pct: 0 }).canopySlices.map((row) => row.value), [0, 100]);
  assert.deepEqual(analysisPieData({ ...rows[0], canopy_coverage_pct: 100 }).canopySlices.map((row) => row.value), [100, 0]);
  assert.deepEqual(analysisPieData({ ...rows[0], total_area_m2: 800 }).areaSlices.map((row) => row.value), [600, 200]);
});

test('Project Sites compares ownership and all current statuses without analysis charts', () => {
  const html = render(SitesTab, { data: {
    summary: { project_sites: 1, mapped_points: 21, sites_with_warnings: 1 },
    suitability: rows,
    project_sites: [{ id: 7, name: 'Coastal site', organization_name: 'Coastal guardians', point_count: 21,
      geometry: { type: 'Polygon', coordinates: [] },
      point_status_counts: { planned: 1, assigned: 2, planted: 3, dead: 4, skipped: 5, unavailable: 6 },
      eroded_point_count: 6,
      warnings: [{ warning_type: 'deep_mud', label: 'Deep mud', severity: 'high', count: 8 }],
    }],
  }, onViewHistory: () => {} });
  assert.match(html, /Compare project sites/);
  assert.match(html, /Coastal guardians/);
  assert.match(html, /Current locations across all planting dates/);
  assert.match(html, /In eroded zones: <strong>6/);
  assert.match(html, /Deep mud: <strong>8/);
  assert.match(html, /View planting history/);
  assert.doesNotMatch(html, /Choose an analysis|Canopy coverage|Plantable area|Risk area|Assignments<|Analyses</);
  const dom = new JSDOM(html);
  const document = dom.window.document;
  assert.equal(document.querySelectorAll('table').length, 1);
  assert.deepEqual([...document.querySelectorAll('.dash-site-statuses > div')].map((entry) => entry.textContent),
    ['Planned1', 'Assigned2', 'Planted3', 'Dead4', 'Skipped5', 'Unavailable6']);
  assert.ok(document.querySelector('a[href="/map?project_site_id=7&focus=site_points"]'));
  assert.ok(document.querySelector('a[href="/monitoring?project_site_id=7"]'));
  assert.ok(document.querySelector('a[href="/processing?action=review"]'));
  dom.window.close();
});

test('empty Project Sites directs users to the current assignment workflow', () => {
  const html = render(SitesTab, { data: {} });
  assert.match(html, /No project sites in this view/);
  assert.match(html, /Assign Points/);
  assert.doesNotMatch(html, /Save an image analysis|Add a project site in the Zone Editor/);
});

const render = (Component, props = {}) => renderToStaticMarkup(createElement(MemoryRouter, null, createElement(Component, props)));

test('dashboard opens Planting Goals from the header without a goals tab or refresh control', () => {
  const html = render(Dashboard);
  assert.equal((html.match(/role="tab"/g) || []).length, 4);
  assert.match(html, /class="dash-goals-button" aria-expanded="false" aria-controls="dashboard-planting-goals"/);
  assert.match(html, /Planting Goals/);
  assert.doesNotMatch(html, /dashboard-settings|Close settings|>Settings</);
  assert.doesNotMatch(html, /dash-tab-goals|dash-panel-goals|Refresh data|dash-refresh-button/);
  assert.doesNotMatch(html, /Data Checks|Dashboard confidence|Analysis freshness|Quality trend/);
});

test('dashboard report tabs no longer show Information to complete notices', () => {
  const notices = { missing_site_links: 20, missing_event_site_links: 1,
    missing_species_links: 2, missing_event_species_links: 7, skips_missing_reason: 4 };
  for (const Component of [OperationsTab, EcologyTab, SitesTab]) {
    const html = render(Component, { data: {}, notices, noticesError: 'Network failure' });
    assert.doesNotMatch(html, /Information to complete|dash-record-notice|Missing-information check unavailable/);
    assert.doesNotMatch(html, /Review planting assignments|Review image analyses/);
  }
});

test('overview shows total organizations as a plain summary card', () => {
  const html = render(OverviewTab, { data: {
    kpis: { total_organizations: { value: 12, scope: 'global' }, sites_requiring_attention: { total: 5, value: 2 } },
    site_attention: [{ site_name: 'Site A', reasons: ['overdue_inspections'], overdue_inspections: 3 }],
  } });
  assert.match(html, /Total organizations/);
  assert.match(html, /Registered organizations, including LGU/);
  const dom = new JSDOM(html);
  const card = [...dom.window.document.querySelectorAll('.dash-kpi')]
    .find((element) => element.querySelector('.dash-kpi-label').textContent === 'Total organizations');
  assert.equal(card.tagName, 'ARTICLE');
  assert.equal(card.querySelector('.dash-kpi-value').textContent, '12');
  assert.equal(card.querySelector('button, a, [role="button"]'), null);
  assert.equal(card.hasAttribute('tabindex'), false);
  dom.window.close();
  assert.doesNotMatch(html, /Sites needing follow-up|dash-kpi-action|View sites and next steps/);
  assert.doesNotMatch(html, /Project sites that need follow-up|Site A|Inspections are past due/);
  assert.doesNotMatch(html, /Seedlings alive after inspection by project site|Seedlings alive after \d+ days/);
  assert.doesNotMatch(render(OperationsTab, { data: {} }), /class="dash-kpis"/);
});

test('donut status labels and total match Map Analytics including deaths and erosion', () => {
  const html = render(OverviewTab, { data: { lifecycle: [
    { key: 'available', value: 1853 }, { key: 'assigned', value: 245 },
    { key: 'planted_unverified', value: 900 }, { key: 'verified_alive', value: 100 },
    { key: 'dead', value: 100 }, { key: 'skipped', value: 20 },
    { key: 'unavailable', value: 246 },
  ] } });
  for (const [label, value] of [
    ['Planned', '1,853'], ['Assigned', '245'], ['Planted', '1,000'],
    ['Dead', '100'], ['Skipped', '20'], ['Unavailable', '246'],
  ]) {
    assert.match(html, new RegExp(`<strong>${label}</strong><small>[^<]*</small></div><b>${value}</b>`));
  }
  assert.match(html, /<strong>3,464<\/strong><span>Total points<\/span>/);
  assert.match(html, /across all dates/);
  assert.match(html, /Choose all project sites to compare with Planting Map/);
});

test('released sites keep Monitoring access without a misleading map link or invented counts', () => {
  const html = render(SitesTab, { data: { project_sites: [
    { id: 9, name: 'Released site', area_released_at: '2026-10-01', geometry: null,
      point_count: 0, point_status_counts: {}, eroded_point_count: 0, warnings: [] },
  ] } });
  assert.match(html, /Area released/);
  assert.match(html, /No current boundary/);
  assert.match(html, /No organization recorded/);
  assert.match(html, /No mapped warnings/);
  assert.match(html, /href="\/monitoring\?project_site_id=9"/);
  assert.doesNotMatch(html, /href="\/map\?/);
  const dom = new JSDOM(html);
  assert.equal(dom.window.document.querySelector('.dash-site-statuses dd').textContent, '—');
  dom.window.close();
});

test('planting goals retain saved targets without the unused inspection weekday controls', () => {
  const html = render(PlantingGoalsPanel, { year: 2026, settings: {
    year: 2026, annual_planting_target: 2000, min_survival_target_pct: 80, inspection_weekdays: [2, 5],
  } });
  assert.match(html, /Seedlings to plant this year/);
  assert.match(html, /Target percentage of seedlings alive/);
  assert.match(html, /Planting goals/);
  assert.match(html, /across all project sites/);
  assert.match(html, /Reporting year/);
  assert.match(html, /value="2026" selected/);
  assert.match(html, /Save planting goals/);
  assert.doesNotMatch(html, /LGU inspection|Monday|Tuesday|Friday|type="checkbox"|Save settings/);
  assert.match(html, /value="2000"/);
  assert.match(html, /value="80"/);
});

test('the goals panel uses its own year selector while the report tabs remain separate', () => {
  for (const tab of ['overview', 'operations', 'ecology', 'sites']) {
    assert.deepEqual(dashboardSectionsForTab(tab), [tab]);
  }
  const html = render(PlantingGoalsPanel, { year: 2025, settings: {
    year: 2025, annual_planting_target: 1500, min_survival_target_pct: null,
  } });
  assert.match(html, /value="2025" selected/);
  assert.match(html, /value="1500"/);
  assert.doesNotMatch(html, /Choose what to view|Project site<|Download Report|dashboard date range/);
});

test('goal editing waits for saved targets and disables saving while loading', () => {
  const missing = render(PlantingGoalsForm, { year: 2026, loading: true });
  assert.match(missing, /Loading/);
  assert.doesNotMatch(missing, /<form|Save planting goals/);
  const failed = render(PlantingGoalsForm, { year: 2026, error: 'Network failure' });
  assert.match(failed, /Planting goals could not be loaded/);
  assert.doesNotMatch(failed, /<form/);
  const refreshing = render(PlantingGoalsForm, {
    year: 2026, loading: true, settings: { year: 2026, annual_planting_target: 2000 },
  });
  assert.match(refreshing, /disabled=""[^>]*>Save planting goals/);
});


test('Seedling Health separates historical planting performance from current map status', () => {
  const html = render(EcologyTab, { data: {
    summary: { total: 860, planted: 860, dead: 0, survival_rate_pct: 100 },
    planting_performance: {
      summary: { total: 1145, planted: 860, dead: 285, unknown: 0, survival_rate_pct: 75.11 },
      replanting: { awaiting_replanting: 20, awaiting_review: 12, replacements_completed: 15 },
      batches: [
        { planted_date: '2026-05-16', organization_name: 'Coastal guardians', site_name: 'Coastal site', species: 'Bungalon', planting_type: 'original', total: 100, planted: 80, dead: 20, unknown: 0, survival_rate_pct: 80 },
        { planted_date: '2026-08-01', organization_name: 'Coastal guardians', site_name: 'Coastal site', species: 'Bungalon', planting_type: 'replacement', total: 20, planted: 20, dead: 0, unknown: 0, survival_rate_pct: 100 },
      ],
    },
    survival_cohorts: [{ interval_days: 30, alive: 496, dead: 234 }],
  } });
  assert.match(html, /Planting performance/);
  assert.match(html, /Current mapped seedlings/);
  assert.match(html, /all planting dates/);
  for (const value of ['1,145', '860', '285']) assert.match(html, new RegExp(value));
  assert.equal((html.match(/class="dash-kpi dash-kpi-/g) || []).length, 4);
  assert.match(html, /75.1%/);
  for (const label of ['Seedlings planted', 'Without recorded death', 'Recorded deaths', 'Recorded survival']) assert.match(html, new RegExp(label));
  assert.doesNotMatch(html, /LGU replanting|Replanting progress|dash-replanting-kpis/i);
  const dom = new JSDOM(html);
  const batchTable = [...dom.window.document.querySelectorAll('table')].find((table) => table.caption?.textContent === 'Survival by planting batch');
  const rows = [...batchTable.querySelectorAll('tbody tr')].map((row) => row.textContent);
  assert.match(rows[0], /Original.*1008020.*80%/);
  assert.match(rows[1], /Replacement.*20200.*100%/);
  dom.window.close();
  assert.doesNotMatch(html, /Checked 30 days|days after planting|Show results from|Inspection age|species-inspection-round|site-inspection-round/);
  assert.doesNotMatch(html, /496 alive|234<|Inspections completed/);
});

test('Planting Work omits assignment waiting ages and keeps its other charts', () => {
  const html = render(OperationsTab, { data: { backlog_aging: [{ label: '0–7 days', count: 5 }] } });
  assert.match(html, /Planting work by organization/);
  assert.match(html, /Planting work over time/);
  assert.doesNotMatch(html, /How long assigned points have been waiting|Days since assignment|Time waiting/);
});

test('planting goals require review, Cancel preserves values, and repeated confirmation submits once', async () => {
  const dom = new JSDOM('<body><div id="root"></div></body>', { url: 'http://dashboard.example', pretendToBeVisual: true });
  const previous = new Map(), writes = [], saved = [];
  let finish;
  for (const [key, value] of Object.entries({ window: dom.window, document: dom.window.document,
    IS_REACT_ACT_ENVIRONMENT: true, fetch: async (_url, options) => {
      writes.push(JSON.parse(options.body));
      return new Promise((resolve) => { finish = resolve; });
    } })) {
    previous.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  const root = createRoot(document.getElementById('root'));
  try {
    await act(async () => root.render(createElement(PlantingGoalsForm, { year: 2026,
      settings: { year: 2026, annual_planting_target: 2000, min_survival_target_pct: 85 }, onSaved: (value) => saved.push(value) })));
    const submit = async () => act(async () => document.querySelector('.dash-goals').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true })));
    await submit();
    assert.equal(writes.length, 0);
    assert.match(document.querySelector('.modal-body').textContent, /2026.*2000.*85%/);
    assert.equal(document.activeElement.textContent, 'Cancel');
    await act(async () => document.querySelector('.modal-actions .btn-secondary').click());
    assert.equal(document.querySelector('[name="annualTarget"]').value, '2000');
    assert.equal(writes.length, 0);
    await submit();
    const confirm = document.querySelector('.modal-actions .btn-primary');
    await act(async () => { confirm.click(); confirm.click(); });
    assert.equal(confirm.disabled, true);
    assert.deepEqual(writes, [{ year: 2026, annual_planting_target: 2000, min_survival_target_pct: 85 }]);
    await act(async () => finish(Response.json(writes[0])));
    assert.equal(saved.length, 1);
    assert.equal(document.querySelector('[role="dialog"]'), null);
  } finally {
    await act(async () => root.unmount());
    for (const [key, descriptor] of previous) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
    }
    dom.window.close();
  }
});

test('Seedling Health shows non-cumulative monthly death counts and zero-death months', () => {
  const html = render(EcologyTab, { data: {
    summary: { total: 100, planted: 50, dead: 50, survival_rate_pct: 50 },
    monthly_deaths: [
      { period_start: '2026-02-01', deaths: 7 },
      { period_start: '2026-03-01', deaths: 0 },
      { period_start: '2026-04-01', deaths: 4 },
    ],
    mortality_includes_unlocated: false,
  } });
  assert.match(html, /Seedling deaths by month/);
  assert.match(html, /New deaths recorded in each month/);
  assert.match(html, /Monthly number of seedlings recorded dead/);
  assert.match(html, /Number of dead seedlings/);
  assert.match(html, /Deaths without identified locations are excluded/);
  const dom = new JSDOM(html);
  const table = [...dom.window.document.querySelectorAll('table')]
    .find((element) => element.caption?.textContent === 'Seedling deaths by month data');
  assert.ok(table);
  assert.deepEqual([...table.querySelectorAll('tbody tr')].map((row) =>
    [...row.querySelectorAll('td')].map((cell) => cell.textContent)), [
    ['Feb 2026', '7'], ['Mar 2026', '0'], ['Apr 2026', '4'],
  ]);
  dom.window.close();
  assert.match(render(EcologyTab, { data: {} }), /No recorded death history yet/);
});

test('Project Sites requests a current snapshot and restores the site and dates in planting history', async () => {
  const dom = new JSDOM('<div id="root"></div>', { url: 'http://dashboard.example', pretendToBeVisual: true });
  const requests = [];
  const previous = new Map();
  const globals = {
    window: dom.window, document: dom.window.document, IS_REACT_ACT_ENVIRONMENT: true,
    ResizeObserver: class { observe() {} disconnect() {} },
    fetch: async (input, options = {}) => {
      assert.equal(options.method || 'GET', 'GET');
      const url = new URL(input, dom.window.location.origin);
      requests.push(url);
      if (url.pathname === '/api/dashboard/settings') return Response.json({ year: 2026 });
      assert.match(url.pathname, /^\/api\/dashboard\/(overview|sites|ecology)$/);
      return Response.json({ filter_options: { sites: [{ id: 7, name: 'Coastal site' }] },
        summary: { project_sites: 0, mapped_points: 0, sites_with_warnings: 0 }, project_sites: [] });
    },
  };
  for (const [key, value] of Object.entries(globals)) {
    previous.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  let root;
  try {
    root = createRoot(dom.window.document.querySelector('#root'));
    await act(async () => root.render(createElement(MemoryRouter, null, createElement(Dashboard))));
    const document = dom.window.document;
    const dates = [...document.querySelectorAll('input[type="date"]')].map((input) => input.value);
    assert.equal(dates.length, 2);
    const chooseSite = document.querySelector('.dash-filters select');
    await act(async () => {
      chooseSite.value = '7';
      chooseSite.dispatchEvent(new dom.window.Event('change', { bubbles: true }));
    });
    await act(async () => document.querySelector('#dash-tab-sites').click());
    assert.equal(document.querySelectorAll('input[type="date"]').length, 0);
    assert.equal(document.querySelector('.dash-report-button'), null);
    assert.match(document.querySelector('.dash-filter-count').textContent, /Current snapshot/);
    const snapshot = requests.filter((url) => url.pathname.endsWith('/sites')).at(-1);
    assert.equal(snapshot.searchParams.get('site_id'), '7');
    assert.equal(snapshot.searchParams.has('date_from'), false);
    assert.equal(snapshot.searchParams.has('date_to'), false);
    await act(async () => [...document.querySelectorAll('.dash-site-comparison button')]
      .find((button) => button.textContent === 'View planting history').click());
    assert.equal(document.querySelector('#dash-tab-ecology').getAttribute('aria-selected'), 'true');
    assert.deepEqual([...document.querySelectorAll('input[type="date"]')].map((input) => input.value), dates);
    assert.equal(document.querySelector('.dash-filters select').value, '7');
    assert.ok(document.querySelector('.dash-report-button'));
    const history = requests.filter((url) => url.pathname.endsWith('/ecology')).at(-1);
    assert.equal(history.searchParams.get('site_id'), '7');
    assert.equal(history.searchParams.get('date_from'), dates[0]);
    assert.equal(history.searchParams.get('date_to'), dates[1]);
    await act(async () => document.querySelector('#dash-tab-sites').click());
    await act(async () => [...document.querySelectorAll('.dash-filter-actions button')]
      .find((button) => button.textContent === 'Show all sites').click());
    assert.equal(document.querySelector('.dash-filters select').value, '');
    assert.equal(requests.filter((url) => url.pathname.endsWith('/sites')).at(-1).searchParams.has('site_id'), false);
  } finally {
    if (root) await act(async () => root.unmount());
    for (const [key, descriptor] of previous) {
      if (descriptor) Object.defineProperty(globalThis, key, descriptor);
      else delete globalThis[key];
    }
    dom.window.close();
  }
});
