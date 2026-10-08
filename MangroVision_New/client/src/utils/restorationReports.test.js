import test from 'node:test';
import assert from 'node:assert/strict';
import { createApiReadCache } from './apiReadCache.js';
import {
  buildRestorationReport, downloadReportPdf, formatReportValue, loadReportSource, manilaDay, reportCsv,
  reportPdfPayload, reportPeriod, reportRequests, validateReportPeriod,
} from './restorationReports.js';

const filters = { dateFrom: '2026-07-01', dateTo: '2026-09-30', siteId: '' };
const envelope = {
  period: { from: '2026-07-01T00:00:00+08:00', to: '2026-09-30T23:59:59+08:00' },
  as_of: '2026-10-04T09:00:00+08:00',
  filter_options: { sites: [{ id: 4, name: 'Nasugban' }] },
};

test('quarter presets work across the year boundary and use Philippine dates', () => {
  assert.deepEqual(reportPeriod('last-quarter', '2026-01-02'), { dateFrom: '2025-10-01', dateTo: '2025-12-31' });
  assert.deepEqual(reportPeriod('quarter', '2026-10-04'), { dateFrom: '2026-10-01', dateTo: '2026-10-04' });
  assert.equal(manilaDay('2026-09-30T18:00:00Z'), '2026-10-01');
  assert.equal(manilaDay('2026-09-30T23:30:00'), '2026-09-30');
  assert.equal(manilaDay('invalid'), '');
});

test('invalid, reversed, missing and future periods are rejected', () => {
  assert.equal(validateReportPeriod(filters, '2026-10-04'), '');
  assert.match(validateReportPeriod({ ...filters, dateFrom: '2026-02-30' }), /valid/);
  assert.match(validateReportPeriod({ ...filters, dateFrom: '2026-10-01' }), /before/);
  assert.match(validateReportPeriod({ ...filters, dateTo: '2026-10-05' }, '2026-10-04'), /future/);
  assert.match(validateReportPeriod({ ...filters, dateTo: '' }), /valid/);
});

test('reports request only their existing sources and apply the project-site filter', () => {
  const requests = reportRequests('mortality', { ...filters, siteId: '4' });
  assert.equal(requests.length, 2);
  assert.match(requests[0].path, /site_id=4/);
  assert.equal(requests[1].path, '/api/monitoring/replanting');
  assert.deepEqual(reportRequests('planting', filters).map((item) => item.key), ['overview']);
  assert.deepEqual(reportRequests('monitoring', filters).map((item) => item.key), ['ecology', 'organizationVisits']);
  assert.deepEqual(reportRequests('monitoring', { ...filters, siteId: '4' }).map((item) => item.key), ['ecology']);
  assert.throws(() => reportRequests('financial', filters), /supported/);
});

test('visit export follows pagination, filters the period and stops after older visits', async () => {
  const pages = [
    { records: [{ id: 4, monitored_at: '2026-10-02' }], next_before_id: 4 },
    { records: [{ id: 3, monitored_at: '2026-09-15' }, { id: 2, monitored_at: '2026-08-15' }], next_before_id: 2 },
    { records: [{ id: 1, monitored_at: '2026-06-15' }], next_before_id: 1 },
  ];
  const calls = [];
  const result = await loadReportSource({ key: 'organizationVisits', path: '/visits?limit=1000' }, filters, {
    fetcher: async (path, options) => { calls.push(path); assert.equal(options.cache, 'no-store'); return { ok: true, json: async () => pages[calls.length - 1] }; },
  });
  assert.deepEqual(result.records.map((record) => record.id), [3, 2]);
  assert.deepEqual(calls, ['/visits?limit=1000', '/visits?limit=1000&before_id=4', '/visits?limit=1000&before_id=2']);
});

test('failed or incomplete visit pagination cannot produce a partial report', async () => {
  const request = { key: 'organizationVisits', path: '/visits?limit=1000' };
  const bad = async () => ({ ok: true, json: async () => ({ records: [{ id: 4, monitored_at: '2026-09-15' }], next_before_id: 4 }) });
  await assert.rejects(loadReportSource(request, filters, { fetcher: bad }), /all organization visits/);
  await assert.rejects(loadReportSource(request, filters, { fetcher: async () => ({ ok: true, json: async () => ({}) }) }), /incomplete/);
  await assert.rejects(loadReportSource(request, filters, { fetcher: async () => ({ ok: false, json: async () => ({ detail: 'Sign in again.' }) }) }), /Sign in again/);
});

test('planting accomplishment counts recorded seedlings planted, including replacements, rather than assignments', () => {
  const overview = { ...envelope, kpis: { seedlings_planted: { value: 17 }, assigned_backlog: { value: 350 } }, planting_progress: [{ period_start: '2026-07-01', planted: 17, assigned: 95, cumulative_planted: 17, target_cumulative: 900 }] };
  const report = buildRestorationReport('planting', { overview }, filters);
  assert.equal(report.stats[0].value, 17);
  assert.equal(report.stats[0].label, 'Seedlings planted');
  assert.deepEqual(report.sections[0].columns.map((item) => item.key), ['period_start', 'planted', 'cumulative_planted']);
  assert.match(report.notes.join(' '), /replacement/i);
  assert.throws(() => buildRestorationReport('planting', { overview: {} }, filters), /incomplete/);
});

test('planting wording and month labels agree in the preview data, CSV and PDF snapshot', () => {
  const report = buildRestorationReport('planting', { overview: { ...envelope,
    kpis: { seedlings_planted: { value: 1111 } },
    planting_progress: [{ period_start: '2026-07-01', planted: 10, cumulative_planted: 10 }],
  } }, filters);
  const csv = reportCsv(report);
  const pdf = reportPdfPayload(report);
  assert.deepEqual(pdf.sections[0].columns.map((column) => column.label), ['Month', 'Seedlings planted', 'Cumulative seedlings planted']);
  assert.deepEqual(pdf.sections[0].rows[0], ['Jul 2026', '10', '10']);
  assert.match(csv, /"Month","Seedlings planted","Cumulative seedlings planted"/);
  assert.match(csv, /"Jul 2026","10","10"/);
  for (const content of [csv, JSON.stringify(pdf)]) {
    assert.doesNotMatch(content, /planting events?|Period cumulative/i);
    assert.match(content, /Replacement seedlings/);
    assert.match(content, /partial months/);
  }
});

test('overall report uses current planting counts and ignores age-specific health results', () => {
  const ecology = { ...envelope,
    summary: { total: 20, planted: 18, alive: 6, dead: 2, missing: 1, uninspected: 11 },
    survival_cohorts: [{ interval_days: 30, alive: 60, dead: 20, due: 200, inspected: 90 }],
  };
  const report = buildRestorationReport('monitoring', { ecology }, filters);
  assert.deepEqual(report.stats.map((row) => row.value), [20, 18, 2, 90]);
  assert.deepEqual(report.stats.map((row) => row.label), ['Total seedlings', 'Planted', 'Dead', 'Survival rate']);
  assert.equal(report.sections[0].columns.length, 4);
  assert.match(report.notes.join(' '), /all planting dates/);
  assert.doesNotMatch(JSON.stringify(report), /30-day|Inspection age|Recorded survival|Awaiting health record|Health recorded/);
  assert.equal(formatReportValue(null, 'percent'), 'N/A');
  assert.equal(formatReportValue(0, 'percent'), '0%');
});

test('overall site results do not add historical deaths or repeat inspection rounds', () => {
  const current = { total: 12, planted: 8, dead: 4 };
  const ecology = { ...envelope, summary: current,
    survival_cohorts: [{ interval_days: 28, alive: 60, dead: 40, carried_dead: 30 }],
    site_outcomes: [{ site_name: 'Nasugban', ...current }],
  };
  const report = buildRestorationReport('monitoring', { ecology }, filters);
  assert.equal(report.stats[0].value, 12);
  assert.equal(report.sections[1].rows[0].dead, 4);
  assert.equal(report.sections[1].rows[0].planted, 8);
  assert.match(report.notes.join(' '), /Each currently planted or dead mapped location counts once/);
  assert.doesNotMatch(JSON.stringify(report.sections), /28-day|Earlier deaths/);
});

test('height annex appears only for recorded measurements with a sample', () => {
  const ecology = { ...envelope, growth: [{ period_start: '2026-09-01', average_height_cm: 23.5, sample_size: 8 }, { average_height_cm: null, sample_size: 0 }] };
  const report = buildRestorationReport('monitoring', { ecology }, filters);
  assert.equal(report.sections.at(-1).rows.length, 1);
  assert.match(report.sections.at(-1).title, /height/);
  assert.doesNotMatch(buildRestorationReport('monitoring', { ecology: { ...ecology, growth: [] } }, filters).sections.map((item) => item.title).join(' '), /height/);
});

test('organization visit balances stay separate from verified survival and are excluded for a selected site', () => {
  const ecology = { ...envelope, summary: { total: 20, planted: 18, alive: 6, dead: 2, missing: 1, uninspected: 11 } };
  const organizationVisits = { records: [
    { organization_name: 'Nasugban', monitored_at: '2026-09-15', alive_count: 80, dead_count: 20, reported_dead_count: 3, health_status: 'fair', actions_taken: 'Inspect erosion', inspector_name: 'Officer' },
    { organization_name: 'Nasugban', monitored_at: '2026-06-15', alive_count: 90, dead_count: 10 },
  ] };
  const report = buildRestorationReport('monitoring', { ecology, organizationVisits }, filters);
  assert.equal(report.stats[0].value, 20);
  const visits = report.sections.find((section) => section.title === 'Organization-level monitoring visits');
  assert.equal(visits.rows.length, 1);
  assert.equal(visits.rows[0].alive_count, 80);
  assert.equal(visits.rows[0].health_label, 'Fair');
  assert.match(report.notes.join(' '), /must not be summed/);
  assert.equal(buildRestorationReport('monitoring', { ecology, organizationVisits }, { ...filters, siteId: '4' }).sections.some((section) => section.title === visits.title), false);
});

test('mortality counts reported deaths once and filters location detail by death date and site', () => {
  const ecology = { ...envelope, mortality_causes: [{ label: 'Erosion', deaths: 8, percent_of_deaths: 100 }], mortality_includes_unlocated: false };
  const replanting = { points: [
    { project_site_id: 4, death_at: '2026-08-01', replanting_status: 'approved' },
    { project_site_id: 4, death_at: '2026-09-30T18:00:00Z', replanting_status: 'completed' }, // October 1 in Manila.
    { project_site_id: 4, death_at: '2026-07-20', replanting_status: 'completed' },
    { project_site_id: 9, death_at: '2026-08-02', replanting_status: 'assigned' },
    { project_site_id: 4, death_at: '2026-06-30', replanting_status: 'awaiting_review' },
  ] };
  const report = buildRestorationReport('mortality', { ecology, replanting }, { ...filters, siteId: '4' });
  assert.equal(report.site, 'Nasugban');
  assert.deepEqual(report.stats.map((item) => item.value), [8, 2, 1, 1]);
  assert.equal(report.sections[1].rows.length, 2);
  assert.equal(report.sections[1].rows[0].status_label, 'Replanting approved; awaiting assignment');
  assert.equal(report.sections[1].columns[0].label, 'Seedling record ID');
  assert.match(report.notes.join(' '), /after the period end/);
  assert.match(report.notes.join(' '), /Differences in these dates can produce different counts/);
  assert.match(report.notes.join(' '), /individual dead seedlings, rather than distinct physical points/);
  assert.match(report.notes.join(' '), /excluded/);
  assert.throws(() => buildRestorationReport('mortality', { ecology }, filters), /incomplete/);
});

test('organization wording identifies unassigned groups and keeps point statuses separate from seedling totals', () => {
  const report = buildRestorationReport('organizations', { operations: { ...envelope,
    summary: { completed: 3, pending: 10, skipped: 1 },
    workload: [{ organization_id: null, organization_name: 'Nasugban', completed: 3, pending: 10, skipped: 1 }],
    assignments: [{ organization_id: null, site_name: 'Nasugban', completed: 3, pending: 10, skipped: 1 }],
  } }, filters);
  assert.equal(report.stats[0].label, 'Organization groups');
  assert.equal(report.stats[1].label, 'Points marked planted');
  assert.equal(report.sections[0].rows[0].organization_name, 'No organization assigned (Nasugban)');
  assert.equal(report.sections[1].rows[0].organization_name, 'No organization assigned (Nasugban)');
  assert.match(report.notes.join(' '), /number of seedlings planted/);
  assert.match(report.notes.join(' '), /current active assignments/);
});

test('organization report keeps current backlog separate from period accomplishments', () => {
  const operations = { ...envelope, summary: { completed: 12, pending: 20, skipped: 2 }, workload: [{ organization_name: 'Nasugban', completed: 12, pending: 20, skipped: 2, total: 34 }] };
  const report = buildRestorationReport('organizations', { operations }, filters);
  assert.deepEqual(report.stats.map((item) => item.value), [1, 12, 20, 2]);
  assert.doesNotMatch(report.sections[0].columns.map((item) => item.label).join(' '), /Total|Attendance/);
  assert.match(report.notes.join(' '), /current active assignments/);
});

test('CSV handles Unicode, quotes and newlines while neutralizing spreadsheet formulas', () => {
  const report = buildRestorationReport('organizations', { operations: { ...envelope, workload: [{ organization_name: '  =HYPERLINK("https://invalid")', completed: 0, pending: 0, skipped: 0 }] } }, filters);
  const csv = reportCsv(report, { preparedBy: '@SUM(1)', generatedAt: 'Oct 4, 2026', remarks: '+malicious\nUnicode: ñ and “quotes”' });
  assert.equal(csv.charCodeAt(0), 0xfeff);
  assert.match(csv, /"' {2}=HYPERLINK\(""https:\/\/invalid""\)"/);
  assert.match(csv, /"'@SUM\(1\)"/);
  assert.match(csv, /"'\+malicious\nUnicode: ñ/);
  assert.match(csv, /"0"/);
  assert.match(csv, /"N\/A"/);
  assert.equal(csv.endsWith('\r\n'), true);
});

test('PDF exports overall totals with formatted dates and optional remarks', () => {
  const report = buildRestorationReport('monitoring', { ecology: { ...envelope,
    summary: { total: 20, planted: 20, alive: 0, dead: 0, missing: 1, uninspected: 19 },
  } }, filters);
  const payload = reportPdfPayload(report, { preparedBy: 'LGU Staff', generatedAt: 'Oct 4, 2026', remarks: 'Inspect erosion.\nFollow up next week.' });
  assert.equal(payload.report_type, 'monitoring');
  assert.equal(payload.period_from, '2026-07-01');
  assert.deepEqual(payload.stats.map((row) => row.value), ['20', '20', '0', '100%']);
  assert.equal(payload.remarks, 'Inspect erosion.\nFollow up next week.');
  assert.deepEqual(payload.sections[0].rows[0], report.sections[0].columns.map((column) => formatReportValue(report.sections[0].rows[0][column.key], column.type)));
});

test('Save as PDF requests a PDF attachment directly and passes cancellation to the API', async () => {
  const report = buildRestorationReport('planting', { overview: envelope }, filters);
  const controller = new AbortController();
  const blob = new Blob(['%PDF-test'], { type: 'application/pdf' });
  const result = await downloadReportPdf(report, { remarks: 'Follow up.' }, {
    base: 'https://example.invalid', signal: controller.signal,
    fetcher: async (path, options) => {
      assert.equal(path, 'https://example.invalid/api/export/report/pdf');
      assert.equal(options.method, 'POST');
      assert.equal(options.signal, controller.signal);
      assert.equal(options.cache, 'no-store');
      assert.equal(JSON.parse(options.body).remarks, 'Follow up.');
      return { ok: true, headers: new Headers({ 'Content-Type': 'application/pdf' }), blob: async () => blob };
    },
  });
  assert.equal(result, blob);
});

test('PDF errors keep server authentication messages and reject non-PDF responses', async () => {
  const report = buildRestorationReport('planting', { overview: envelope }, filters);
  await assert.rejects(downloadReportPdf(report, {}, { fetcher: async () => ({ ok: false, json: async () => ({ detail: 'Invalid or expired LGU session.' }) }) }), /LGU session/);
  await assert.rejects(downloadReportPdf(report, {}, { fetcher: async () => ({ ok: true, headers: new Headers({ 'Content-Type': 'text/html' }) }) }), /did not return a PDF/);
});

test('a PDF download through the shared fetch wrapper does not trigger a record refresh', async () => {
  const report = buildRestorationReport('planting', { overview: envelope }, filters);
  const controller = new AbortController();
  let changes = 0;
  const cache = createApiReadCache({
    origin: 'https://example.invalid',
    fetcher: async () => new Response('%PDF-test', { headers: { 'Content-Type': 'application/pdf' } }),
    onMutation: () => { changes += 1; controller.abort(); },
  });
  const pdf = await downloadReportPdf(report, {}, { signal: controller.signal, fetcher: cache.fetch });
  assert.equal(await pdf.text(), '%PDF-test');
  assert.equal(changes, 0);
  assert.equal(controller.signal.aborted, false);
  // Genuine writes still notify pages to refresh their records.
  await cache.fetch('/api/assignments', { method: 'POST' });
  assert.equal(changes, 1);
});


test('overall survival uses map totals, shows no rate for empty data and allows zero survival', () => {
  const report = (summary) => buildRestorationReport('monitoring', { ecology: { ...envelope, summary } }, filters);
  assert.equal(report({ planted: 860, dead: 285 }).stats[3].value, 75.11);
  assert.equal(report({ planted: 0, dead: 0 }).stats[3].value, null);
  assert.equal(report({ planted: 0, dead: 10 }).stats[3].value, 0);
});
