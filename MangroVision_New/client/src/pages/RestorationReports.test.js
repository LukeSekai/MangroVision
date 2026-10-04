import test, { before, after } from 'node:test';
import assert from 'node:assert/strict';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { createServer } from 'vite';
import { buildRestorationReport } from '../utils/restorationReports.js';

let server;
let ReportDocument, ReportsPage, ExportActions;
before(async () => {
  server = await createServer({ configFile: false, appType: 'custom',
    server: { middlewareMode: true, hmr: false, watch: null },
    cacheDir: 'node_modules/.vite-report-tests', optimizeDeps: { noDiscovery: true, include: [] },
    plugins: [{ name: 'report-test-staff', load(id) {
      // Render with a synthetic officer; do not hydrate a real browser session.
      if (id.replaceAll('\\', '/').endsWith('/stores/authStore.js')) return "export const useAuthStore = (selector) => selector({ user: { full_name: 'Test LGU officer' } });";
    } }],
  });
  const module = await server.ssrLoadModule('/src/pages/RestorationReports.jsx');
  ReportDocument = module.RestorationReportDocument;
  ReportsPage = module.default;
  ExportActions = module.ReportExportActions;
});
after(async () => { await server?.close(); });

const filters = { dateFrom: '2026-07-01', dateTo: '2026-09-30', siteId: '4' };
const envelope = { period: { from: filters.dateFrom, to: filters.dateTo },
  filter_options: { sites: [{ id: 4, name: 'Nasugban' }] },
};
const metadata = { generatedAt: '4 Oct 2026, 10:00 am', preparedBy: 'LGU Staff', remarks: '' };
const render = (report, meta = metadata) => renderToStaticMarkup(createElement(ReportDocument, { report, metadata: meta }));

test('report document identifies the site, dates, preparer and observation denominator', () => {
  const report = buildRestorationReport('monitoring', { ecology: { ...envelope,
    summary: { interval_days: 14 }, survival_cohorts: [{ interval_days: 14, alive: 8, dead: 2, missing: 1, due: 20, inspected: 11 }],
  } }, filters);
  const html = render(report);
  assert.match(html, /Nasugban/);
  assert.match(html, /(?:1 Jul 2026|Jul 1, 2026)/);
  assert.match(html, /(?:30 Sept? 2026|Sept? 30, 2026)/);
  assert.match(html, /LGU Staff/);
  assert.match(html, /10 alive or dead seedlings/);
  assert.match(html, /80%/);
  assert.match(html, /55%/);
  assert.match(html, /<thead>/);
  assert.match(html, /scope="col"/);
  assert.doesNotMatch(html, /Certified|Financial|Hectares restored/);
});

test('planting preview names seedlings and shows month totals without implying planting happened on the first day', () => {
  const report = buildRestorationReport('planting', { overview: { ...envelope,
    kpis: { seedlings_planted: { value: 95 } },
    planting_progress: [{ period_start: '2026-07-01', planted: 95, cumulative_planted: 95 }],
  } }, filters);
  const html = render(report);
  assert.match(html, />Seedlings planted<\/th>/);
  assert.match(html, />Cumulative seedlings planted<\/th>/);
  assert.match(html, /<td[^>]*>Jul 2026<\/td>/);
  assert.doesNotMatch(html, /Planting events|planting event|Period cumulative/);
});

test('empty reports explain missing data rather than presenting fabricated survival', () => {
  const report = buildRestorationReport('monitoring', { ecology: { ...envelope } }, filters);
  const html = render(report);
  assert.match(html, /N\/A/);
  assert.match(html, /No records for this selection/);
  assert.doesNotMatch(html, /100%|0%/);
});

test('report text is escaped and remarks are included in the printable document', () => {
  const report = buildRestorationReport('planting', { overview: { ...envelope,
    kpis: { seedlings_planted: { value: 2 } }, planting_progress: [],
  } }, filters);
  report.site = '<img src=x onerror=alert(1)>';
  const html = render(report, { ...metadata, remarks: '<script>alert(1)</script>\nInspect erosion next week.' });
  assert.doesNotMatch(html, /<script>|onerror=alert\(1\)>/);
  assert.match(html, /&lt;script&gt;/);
  assert.match(html, /Inspect erosion next week/);
  assert.match(html, /Report remarks/);
});

test('reports screen updates automatically and keeps tabs and filters available during loading', () => {
  const html = renderToStaticMarkup(createElement(MemoryRouter, null, createElement(ReportsPage)));
  assert.equal((html.match(/aria-pressed=/g) || []).length, 4);
  assert.match(html, /Year to date/);
  assert.match(html, /Project site/);
  assert.match(html, /Preparing your report/);
  assert.match(html, /Reports update automatically/);
  assert.doesNotMatch(html, /Generate report|Budget utilization|Certification|Download CSV|disabled/);
});

test('report exports offer separate Print, Save as PDF and CSV buttons', () => {
  const renderActions = (pdfBusy) => renderToStaticMarkup(createElement(ExportActions, { pdfBusy }));
  const html = renderActions(false);
  assert.equal((html.match(/<button/g) || []).length, 3);
  assert.match(html, />Print<\/button>/);
  assert.match(html, />Save as PDF<\/button>/);
  assert.match(html, />Download CSV<\/button>/);
  assert.doesNotMatch(html, /Print \/ Save PDF|disabled/);
  const busy = renderActions(true);
  assert.match(busy, /Saving PDF/);
  assert.equal((busy.match(/disabled/g) || []).length, 1);
});
