import { pointStatusLabel } from './pointStatus.js';

const TIMEZONE = 'Asia/Manila';

export const REPORT_TYPES = [
  { id: 'planting', title: 'Planting accomplishment', description: 'Seedlings planted during the reporting period, including replacement seedlings.' },
  { id: 'monitoring', title: 'Survival and monitoring', description: 'Seedling inspection results, monitoring visits and recorded height measurements.' },
  { id: 'mortality', title: 'Mortality and replanting', description: 'Reported seedling deaths and replanting progress at identified locations.' },
  { id: 'organizations', title: 'Organization activity', description: 'Planted and skipped points, with current assigned work.' },
];

export function manilaDay(value = new Date()) {
  if (typeof value === 'string' && /^\d{4}-\d{2}-\d{2}$/.test(value)) return value;
  if (!value) return '';
  // Database timestamps without an offset represent local application time.
  if (typeof value === 'string' && /^\d{4}-\d{2}-\d{2}[T ]/.test(value) && !/(Z|[+-]\d{2}:?\d{2})$/i.test(value)) return value.slice(0, 10);
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  const parts = Object.fromEntries(new Intl.DateTimeFormat('en-CA', {
    timeZone: TIMEZONE, year: 'numeric', month: '2-digit', day: '2-digit',
  }).formatToParts(date).map((part) => [part.type, part.value]));
  return `${parts.year}-${parts.month}-${parts.day}`;
}

export function reportPeriod(preset, today = manilaDay()) {
  const [year, month] = today.split('-').map(Number);
  if (preset === 'year') return { dateFrom: `${year}-01-01`, dateTo: today };
  const quarterMonth = Math.floor((month - 1) / 3) * 3 + 1;
  if (preset === 'last-quarter') {
    const first = new Date(Date.UTC(year, quarterMonth - 4, 1));
    const last = new Date(Date.UTC(year, quarterMonth - 1, 0));
    return { dateFrom: first.toISOString().slice(0, 10), dateTo: last.toISOString().slice(0, 10) };
  }
  return { dateFrom: `${year}-${String(quarterMonth).padStart(2, '0')}-01`, dateTo: today };
}

const isReportDay = (value) => /^\d{4}-\d{2}-\d{2}$/.test(value || '')
    && !Number.isNaN(Date.parse(`${value}T00:00:00Z`))
    && new Date(`${value}T00:00:00Z`).toISOString().slice(0, 10) === value;

export function reportPeriodFieldErrors(filters, today = manilaDay()) {
  const errors = {};
  if (!isReportDay(filters.dateFrom)) errors.dateFrom = 'Choose a valid start date.';
  if (!isReportDay(filters.dateTo)) errors.dateTo = 'Choose a valid end date.';
  if (!Object.keys(errors).length && filters.dateFrom > filters.dateTo) errors.dateTo = 'Choose an end date on or after the start date.';
  if (isReportDay(filters.dateFrom) && filters.dateFrom > today) errors.dateFrom = 'Choose today or an earlier start date.';
  if (isReportDay(filters.dateTo) && filters.dateTo > today) errors.dateTo = 'Choose today or an earlier end date.';
  return errors;
}

export function validateReportPeriod(filters, today = manilaDay()) {
  if (!isReportDay(filters.dateFrom) || !isReportDay(filters.dateTo)) return 'Choose valid start and end dates.';
  if (filters.dateFrom > filters.dateTo) return 'The start date must be on or before the end date.';
  if (filters.dateTo > today) return 'The reporting period cannot end in the future.';
  return '';
}

export function reportRequests(type, filters) {
  const query = new URLSearchParams({ date_from: filters.dateFrom, date_to: filters.dateTo, bucket: 'month' });
  if (filters.siteId) query.set('site_id', filters.siteId);
  const dashboard = (section) => ({ key: section, path: `/api/dashboard/${section}?${query}` });
  if (type === 'planting') return [dashboard('overview')];
  if (type === 'organizations') return [dashboard('operations')];
  if (type === 'monitoring') return [dashboard('ecology'), ...(!filters.siteId ? [{ key: 'organizationVisits', path: '/api/monitoring/organization-records?limit=1000' }] : [])];
  if (type === 'mortality') return [dashboard('ecology'), { key: 'replanting', path: '/api/monitoring/replanting' }];
  throw new Error('Choose a supported report type.');
}

export async function loadReportSource(request, filters, { base = '', signal, fetcher = globalThis.fetch } = {}) {
  const read = async (path) => {
    const response = await fetcher(`${base}${path}`, { signal, cache: 'no-store' });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : 'Could not load report records. Please try again.');
    return payload;
  };
  let payload = await read(request.path);
  if (request.key !== 'organizationVisits') return payload;
  const records = [];
  const seen = new Set();
  while (true) {
    if (!Array.isArray(payload.records)) throw new Error('Organization visit records are incomplete. Please try again.');
    records.push(...payload.records);
    const lastDay = manilaDay(payload.records.at(-1)?.monitored_at);
    // The API sorts visits newest first. Older pages cannot contain period visits.
    if (!payload.next_before_id || (lastDay && lastDay < filters.dateFrom)) break;
    const cursor = Number(payload.next_before_id);
    if (!Number.isInteger(cursor) || cursor <= 0 || seen.has(cursor) || !payload.records.length) throw new Error('Could not load all organization visits. Please try again.');
    seen.add(cursor);
    payload = await read(`${request.path}&before_id=${cursor}`);
  }
  return { records: records.filter((record) => {
    const day = manilaDay(record.monitored_at);
    return day && day >= filters.dateFrom && day <= filters.dateTo;
  }) };
}

export function reportNumber(value) {
  if (value === null || value === undefined || value === '') return null;
  const number = Number(value);
  return Number.isFinite(number) ? number : null;
}

export function formatReportValue(value, type = 'text') {
  if (value === null || value === undefined || value === '') return 'N/A';
  if (type === 'date' || type === 'month') {
    const day = manilaDay(value);
    const options = type === 'month' ? { month: 'short', year: 'numeric' } : { dateStyle: 'medium' };
    return day ? new Intl.DateTimeFormat('en-PH', { ...options, timeZone: 'UTC' }).format(new Date(`${day}T00:00:00Z`)) : 'N/A';
  }
  if (['count', 'percent', 'decimal', 'coordinate'].includes(type)) {
    const number = reportNumber(value);
    if (number === null) return 'N/A';
    if (type === 'coordinate') return number.toFixed(7);
    const formatted = new Intl.NumberFormat('en-PH', { maximumFractionDigits: type === 'count' ? 0 : 1 }).format(number);
    return type === 'percent' ? `${formatted}%` : formatted;
  }
  return String(value);
}

const rowsOf = (value) => Array.isArray(value) ? value : [];
const column = (key, label, type = 'text') => ({ key, label, type });
const stat = (label, value, type = 'count', hint = '') => ({ label, value: reportNumber(value), type, hint });
const percent = (numerator, denominator) => denominator > 0 ? numerator / denominator * 100 : null;
const sum = (rows, key) => rows.reduce((total, row) => total + (reportNumber(row[key]) ?? 0), 0);

function inspectionRow(row) {
  const alive = reportNumber(row.alive) ?? 0;
  const dead = reportNumber(row.dead) ?? 0;
  const due = reportNumber(row.due) ?? 0;
  const completed = reportNumber(row.inspected_due ?? row.inspected) ?? 0;
  return {
    ...row, alive, dead, missing: reportNumber(row.missing) ?? 0,
    round: `${row.interval_days}-day`, due, completed,
    uninspected: Math.max(0, due - completed), sample: alive + dead,
    survival: percent(alive, alive + dead), coverage: percent(completed, due),
    carried_dead: reportNumber(row.carried_dead) ?? 0,
  };
}

const REPLANTING_LABELS = {
  awaiting_review: 'Awaiting LGU review', approved: 'Replanting approved; awaiting assignment',
  assigned: 'Assigned for replanting', completed: 'Replacement seedling planted',
};

const HEALTH_LABELS = { excellent: 'Excellent', good: 'Good', fair: 'Fair', poor: 'Poor', critical: 'Critical' };
const responsibleGroup = (row) => {
  if (row.organization_id !== null) return row.organization_name;
  const site = row.site_name || row.organization_name;
  return site && site !== 'No organization assigned' ? `No organization assigned (${site})` : 'No organization assigned';
};

export function buildRestorationReport(type, data, filters) {
  const definition = REPORT_TYPES.find((report) => report.id === type);
  if (!definition) throw new Error('Choose a supported report type.');
  const source = data.overview || data.operations || data.ecology;
  if (!source || !source.period) throw new Error('Report data is incomplete. Refresh and try again.');
  const report = {
    id: type, title: `${definition.title} report`, description: definition.description,
    period: { from: manilaDay(source.period.from), to: manilaDay(source.period.to) },
    site: source.filter_options?.sites?.find((site) => String(site.id) === String(filters.siteId))?.name
      || (filters.siteId ? `Project site #${filters.siteId}` : 'All project sites'),
    asOf: source.as_of, stats: [], sections: [], notes: [],
  };

  if (type === 'planting') {
    report.stats = [stat('Seedlings planted', source.kpis?.seedlings_planted?.value, 'count', 'Planted during the reporting period, including replacements')];
    report.sections = [{
      title: 'Seedlings planted by month',
      columns: [column('period_start', 'Month', 'month'), column('planted', 'Seedlings planted', 'count'), column('cumulative_planted', 'Cumulative seedlings planted', 'count')],
      rows: rowsOf(source.planting_progress),
    }];
    report.notes = [
      'Each recorded seedling planting in the selected dates counts once. Assigning a point alone does not count as planting.',
      'Replacement seedlings are included when planted, even at reused points. The total includes seedlings that later died.',
      'Monthly totals cover selected dates, including partial months. Cumulative seedlings planted starts at the beginning of the reporting period.',
    ];
  }

  if (type === 'monitoring') {
    const cohorts = rowsOf(source.survival_cohorts).map(inspectionRow);
    const primary = cohorts.find((row) => row.interval_days === source.summary?.interval_days);
    report.stats = [
      stat('Observed survival rate', primary?.survival, 'percent', primary ? `${primary.round} inspection · ${primary.sample} alive or dead seedlings` : 'No inspection data available'),
      stat('Inspection completion rate', primary?.coverage, 'percent', primary ? `${primary.round} inspection · ${primary.completed} of ${primary.due} due inspections completed` : 'No inspection data available'),
      stat('Seedlings awaiting inspection', primary?.uninspected, 'count', primary ? `${primary.round} inspection · at the reporting period end` : 'No inspection data available'),
      stat('Missing seedlings', primary?.missing, 'count', primary ? `${primary.round} inspection` : 'No inspection data available'),
    ];
    report.sections = [{
      title: 'Seedling inspection results by age',
      columns: [column('round', 'Inspection age'), column('alive', 'Alive', 'count'), column('dead', 'Dead', 'count'), column('missing', 'Missing', 'count'), column('due', 'Inspections due', 'count'), column('completed', 'Inspections completed', 'count'), column('uninspected', 'Awaiting inspection', 'count'), column('sample', 'Alive + dead', 'count'), column('survival', 'Survival rate', 'percent'), column('coverage', 'Completion rate', 'percent'), column('carried_dead', 'Earlier deaths', 'count')],
      rows: cohorts,
    }, {
      title: 'Seedling inspection results by project site',
      columns: [column('site_name', 'Project site'), column('round', 'Inspection age'), column('alive', 'Alive', 'count'), column('dead', 'Dead', 'count'), column('missing', 'Missing', 'count'), column('uninspected', 'Awaiting inspection', 'count'), column('sample', 'Alive + dead', 'count'), column('survival', 'Survival rate', 'percent'), column('coverage', 'Completion rate', 'percent')],
      rows: rowsOf(source.site_outcomes).map(inspectionRow),
    }, {
      title: 'Seedling inspection results by species',
      columns: [column('species_name', 'Species'), column('round', 'Inspection age'), column('alive', 'Alive', 'count'), column('dead', 'Dead', 'count'), column('missing', 'Missing', 'count'), column('uninspected', 'Awaiting inspection', 'count'), column('sample', 'Alive + dead', 'count'), column('survival', 'Survival rate', 'percent'), column('coverage', 'Completion rate', 'percent')],
      rows: rowsOf(source.species_outcomes).map(inspectionRow),
    }];
    const growth = rowsOf(source.growth).filter((row) => reportNumber(row.average_height_cm) !== null && reportNumber(row.sample_size) > 0);
    if (growth.length) report.sections.push({
      title: 'Recorded height measurements',
      columns: [column('period_start', 'Measurement month', 'month'), column('average_height_cm', 'Average measured seedling height (cm)', 'decimal'), column('sample_size', 'Number of measurements', 'count')],
      rows: growth,
    });
    report.notes = [
      'Results cover seedlings planted in the selected dates, using LGU inspections recorded through the reporting period end. Inspection age is the scheduled number of days after planting; actual visits may occur later. Keep each age separate when reading the totals.',
      'Alive, dead and missing are seedling counts. Observed survival rate = alive ÷ (alive + dead) × 100. Missing seedlings and those awaiting inspection are excluded from this rate. N/A means the rate cannot be calculated from the available records.',
      'Inspection completion rate = inspections completed ÷ inspections due × 100. It measures how much inspection work was completed by the reporting period end.',
      'Earlier deaths are already included in the dead seedling count at later inspection ages. They are excluded from inspections due and completed; do not add them to the dead count again.',
    ];
    if (growth.length) report.notes.push('Height figures use recorded measurements of alive seedlings, grouped by inspection month. They are measured heights, not growth forecasts; repeated observations may include the same seedling.');
    if (!filters.siteId && data.organizationVisits) {
      const visits = rowsOf(data.organizationVisits.records).filter((record) => {
        const day = manilaDay(record.monitored_at);
        return day && day >= filters.dateFrom && day <= filters.dateTo;
      }).map((record) => ({ ...record, health_label: HEALTH_LABELS[record.health_status] || record.health_status }));
      report.sections.push({
        title: 'Organization-level monitoring visits',
        columns: [column('organization_name', 'Organization'), column('monitored_at', 'Visit date', 'date'), column('alive_count', 'Reported alive seedlings', 'count'), column('dead_count', 'Cumulative reported deaths', 'count'), column('reported_dead_count', 'Deaths reported this visit', 'count'), column('death_reason', 'Reported cause of death'), column('health_label', 'Reported seedling health')],
        rows: visits,
      });
      if (visits.some((visit) => visit.actions_taken)) report.sections.push({
        title: 'Recorded field follow-up',
        columns: [column('organization_name', 'Organization'), column('monitored_at', 'Visit date', 'date'), column('actions_taken', 'Recorded actions'), column('inspector_name', 'Inspector')],
        rows: visits.filter((visit) => visit.actions_taken),
      });
      report.notes.push('Organization visits show reported alive seedling counts and cumulative deaths across the organization’s planting history. These balances repeat across visits and must not be summed or combined with individual inspection results. Deaths reported this visit is the additional count reported at that visit.');
    }
    if (filters.siteId) report.notes.push('Organization-wide monitoring visits are excluded from a site-filtered report because their counts do not identify a particular project site.');
  }

  if (type === 'mortality') {
    if (!data.replanting || !Array.isArray(data.replanting.points)) throw new Error('Replacement records are incomplete. Refresh and try again.');
    const causes = rowsOf(source.mortality_causes);
    const replacements = data.replanting.points.filter((row) => {
      const day = manilaDay(row.death_at);
      return day && day >= filters.dateFrom && day <= filters.dateTo
        && (!filters.siteId || String(row.project_site_id) === String(filters.siteId));
    }).map((row) => ({ ...row, status_label: REPLANTING_LABELS[row.replanting_status] || 'Status unavailable' }));
    report.stats = [
      stat('Reported seedling deaths', sum(causes, 'deaths'), 'count', source.mortality_includes_unlocated ? 'Includes deaths without identified locations' : 'Deaths linked to the selected project site'),
      stat('Dead seedlings with locations', replacements.length, 'count', 'Recorded death dates within the reporting period'),
      stat('Replacement seedlings planted', replacements.filter((row) => row.replanting_status === 'completed').length, 'count', 'For seedlings in the location table; status when generated'),
      stat('Replacement seedlings pending', replacements.filter((row) => row.replanting_status !== 'completed').length, 'count', 'For seedlings in the location table; status when generated'),
    ];
    report.sections = [{
      title: 'Reported seedling deaths by cause',
      columns: [column('label', 'Reported cause of death'), column('deaths', 'Seedling deaths', 'count'), column('percent_of_deaths', 'Share of reported deaths', 'percent')],
      rows: causes,
    }, {
      title: 'Dead seedlings with locations and current replanting status',
      columns: [column('planting_event_id', 'Seedling record ID', 'count'), column('point_num', 'Point number', 'count'), column('project_site_name', 'Project site'), column('organization_name', 'Planting organization'), column('death_at', 'Recorded death date', 'date'), column('latitude', 'Latitude', 'coordinate'), column('longitude', 'Longitude', 'coordinate'), column('status_label', 'Current replanting status')],
      rows: replacements,
    }];
    report.notes = [
      'Cause totals use the date a death was reported in a monitoring visit, or the recorded death date for individual records. The location table uses each seedling’s recorded death date. Differences in these dates can produce different counts.',
      'The location table lists individual dead seedlings, rather than distinct physical points. Different seedlings may have been planted at the same point over time. Replanting status is current when generated, including work completed after the period end.',
      source.mortality_includes_unlocated
        ? 'Reported deaths include deaths without identified locations. The cause totals and location table are separate views of death records; do not add their counts together.'
        : 'Deaths without identified locations are excluded from the cause totals when a project site is selected because their site is unknown.',
      'Replanting approval and assignment are preparation steps. Replacement seedlings planted counts cases with a recorded replacement planting confirmation.',
    ];
  }

  if (type === 'organizations') {
    const workload = rowsOf(source.workload).map((row) => ({ ...row, organization_name: responsibleGroup(row) }));
    report.stats = [
      stat('Organization groups', workload.length, 'count', 'Responsible organizations and unassigned groups listed'),
      stat('Points marked planted', source.summary?.completed, 'count', 'Assignment completions in the reporting period'),
      stat(`${pointStatusLabel('pending')} points (current)`, source.summary?.pending, 'count', 'Assigned points awaiting planting when the report was generated'),
      stat('Skipped assignment points', source.summary?.skipped, 'count', 'Marked skipped in the reporting period'),
    ];
    report.sections = [{
      title: 'Assignment point status by responsible organization',
      columns: [column('organization_name', 'Responsible organization / group'), column('completed', 'Points marked planted in period', 'count'), column('pending', `${pointStatusLabel('pending')} points (current)`, 'count'), column('skipped', 'Points skipped in period', 'count')],
      rows: workload,
    }, {
      title: 'Assignment detail',
      columns: [column('title', 'Assignment'), column('organization_name', 'Responsible organization / group'), column('site_name', 'Project site'), column('species', 'Species'), column('completed', 'Points marked planted in period', 'count'), column('pending', `${pointStatusLabel('pending')} points (current)`, 'count'), column('skipped', 'Points skipped in period', 'count')],
      rows: rowsOf(source.assignments).map((row) => ({ ...row, organization_name: responsibleGroup(row) })),
    }];
    report.notes = [
      'Planted and skipped figures count assignment points with those recorded status dates within the reporting period. Assigned figures count points awaiting planting in current active assignments when the report was generated.',
      'Use the planting accomplishment report for the number of seedlings planted. This report summarizes assignment point statuses; released, removed or reused points can make the totals differ.',
      'Rows are grouped by the organization responsible for each project site. Sites without a responsible organization appear as unassigned groups. These groups do not represent attendance or participant counts.',
    ];
  }
  return report;
}

export function reportFilename(report, extension) {
  return `mangrovision-${report.id}-${report.period.from}-to-${report.period.to}.${extension}`;
}

export function reportPdfPayload(report, { generatedAt, preparedBy = 'LGU staff', remarks = '' } = {}) {
  return {
    report_type: report.id, title: report.title, description: report.description,
    period_from: report.period.from, period_to: report.period.to,
    site: report.site, generated_at: generatedAt || 'N/A', prepared_by: preparedBy || 'LGU staff', remarks,
    stats: report.stats.map((stat) => ({ label: stat.label, value: formatReportValue(stat.value, stat.type), hint: stat.hint || '' })),
    sections: report.sections.map((section) => ({
      title: section.title, columns: section.columns,
      rows: section.rows.map((row) => section.columns.map((column) => formatReportValue(row[column.key], column.type))),
    })),
    notes: report.notes,
  };
}

export async function downloadReportPdf(report, metadata, { base = '', signal, fetcher = fetch } = {}) {
  const response = await fetcher(`${base}/api/export/report/pdf`, {
    method: 'POST', headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(reportPdfPayload(report, metadata)), cache: 'no-store', signal,
  });
  if (!response.ok) {
    const error = await response.json().catch(() => null);
    throw new Error(typeof error?.detail === 'string' ? error.detail : 'Could not save the PDF. Please try again.');
  }
  if (!response.headers.get('Content-Type')?.toLowerCase().startsWith('application/pdf')) {
    throw new Error('The server did not return a PDF. Please try again.');
  }
  return response.blob();
}

export function reportCsv(report, { generatedAt, preparedBy = '', remarks = '' } = {}) {
  const cell = (value) => {
    let text = value === null || value === undefined ? 'N/A' : String(value);
    // Names, notes and assignment titles can contain spreadsheet formulas.
    if (typeof value !== 'number' && /^\s*[=+\-@]/.test(text)) text = `'${text}`;
    return `"${text.replaceAll('"', '""')}"`;
  };
  const rows = [
    [report.title], ['Project site', report.site], ['Period start', report.period.from], ['Period end', report.period.to],
    ['Generated at (Asia/Manila)', generatedAt || 'N/A'], ['Prepared by', preparedBy || 'N/A'], [],
    ['Summary metric', 'Value', 'Scope'], ...report.stats.map((item) => [item.type === 'percent' ? `${item.label} (%)` : item.label, item.value ?? 'N/A', item.hint]), [],
  ];
  for (const section of report.sections) {
    rows.push([section.title], section.columns.map((item) => item.type === 'percent' ? `${item.label} (%)` : item.label));
    for (const row of section.rows) rows.push(section.columns.map((item) => item.type === 'month' ? formatReportValue(row[item.key], item.type) : row[item.key] ?? 'N/A'));
    if (!section.rows.length) rows.push(['No records for this selection.']);
    rows.push([]);
  }
  rows.push(['Reading these figures'], ...report.notes.map((note) => [note]));
  if (remarks.trim()) rows.push([], ['Report remarks', remarks.trim()]);
  return `\uFEFF${rows.map((row) => row.map(cell).join(',')).join('\r\n')}\r\n`;
}
