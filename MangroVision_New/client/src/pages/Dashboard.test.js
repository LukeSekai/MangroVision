// Server rendering verifies content and fixed heights, not browser appearance.
import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
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

test('site charts stay compact and removed sections are absent', () => {
  const html = renderToStaticMarkup(createElement(MemoryRouter, null, createElement(SitesTab, {
    data: { suitability: rows },
  })));
  assert.equal((html.match(/class="dash-area-pie"/g) || []).length, 2);
  assert.match(html, /Choose an analysis/);
  assert.match(html, /60%/);
  assert.match(html, /Image area \(m²\)/);
  assert.doesNotMatch(html, /Unique footprint|Footprint union|Analysis chart pages/);
  assert.match(html, /Analysis 10/); // All analyses remain available in the tables.
  assert.doesNotMatch(html, /Analysis 130|Analysis 148/);
  assert.doesNotMatch(html, /Map drill-down|Next high and low tide|Expected tide height/);
});

test('empty site data renders without an analysis selector', () => {
  const html = renderToStaticMarkup(createElement(MemoryRouter, null, createElement(SitesTab, { data: {} })));
  assert.doesNotMatch(html, /Choose an analysis/);
  assert.match(html, /Save an image analysis/);
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

test('overview keeps progress and follow-up totals without the follow-up table or health comparisons', () => {
  const html = render(OverviewTab, { data: {
    kpis: { sites_requiring_attention: { total: 5, value: 2 } },
    site_attention: [{ site_name: 'Site A', reasons: ['overdue_inspections'], overdue_inspections: 3 }],
  } });
  assert.match(html, /Sites needing follow-up/);
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

test('analysis dates and boundary information share the existing sites table', () => {
  const html = render(SitesTab, { data: { suitability: [
    { ...rows[0], footprint_quality: 'approximate_coverage_rectangle' },
    { ...rows[1], footprint_quality: 'projected' },
  ] } });
  assert.match(html, /Map boundary source/);
  assert.match(html, /Estimated boundary/);
  assert.match(html, /Boundary saved with the analysis/);
  assert.doesNotMatch(html, /approximate_coverage_rectangle|Analysis freshness/);
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


test('Seedling Health shows four overall cards including map-based survival', () => {
  const html = render(EcologyTab, { data: {
    summary: { total: 1145, planted: 860, dead: 285, survival_rate_pct: 75.11 },
    survival_cohorts: [{ interval_days: 30, alive: 496, dead: 234 }],
    species_outcomes: [{ name: 'Mangrove', total: 1145, planted: 860, dead: 285 }],
  } });
  assert.match(html, /Overall seedling totals/);
  assert.match(html, /all planting dates/);
  for (const value of ['1,145', '860', '285']) assert.match(html, new RegExp(value));
  assert.equal((html.match(/class="dash-kpi dash-kpi-/g) || []).length, 4);
  assert.match(html, /75.1%/);
  for (const label of ['Total seedlings', 'Planted', 'Dead', 'Survival rate']) assert.match(html, new RegExp(label));
  assert.doesNotMatch(html, /Recorded survival|Awaiting health record|Recorded alive|Health recorded/);
  assert.doesNotMatch(html, /Checked 30 days|days after planting|Show results from|Inspection age|species-inspection-round|site-inspection-round/);
  assert.doesNotMatch(html, /496 alive|234<|Inspections completed/);
});
