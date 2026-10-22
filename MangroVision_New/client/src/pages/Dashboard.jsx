import { Activity, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { Link } from 'react-router-dom';
import MapLink from '../components/MapLink';
import GrowthGuide from '../components/GrowthGuide';
import AppointmentNotice from '../components/AppointmentNotice';
import RestorationReportDialog from '../components/RestorationReportDialog';
import Modal from '../components/Modal';
import { DASHBOARD_ENDPOINTS, dashboardSectionsForTab } from '../utils/dashboardLoading';
import {
  Bar,
  BarChart,
  Cell,
  CartesianGrid,
  ComposedChart,
  LabelList,
  Legend,
  Line,
  LineChart,
  Pie,
  PieChart,
  ReferenceLine,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import './Dashboard.css';
import NextActions from '../components/NextActions';
import { POINT_STATUS_LABELS, POINT_STATUS_COLORS } from '../utils/pointStatus';
import useFormFeedback from '../utils/useFormFeedback';
import { FieldError, FormErrorSummary } from '../components/FormFeedback';
import { submissionError } from '../utils/formValidation';

const API = import.meta.env.VITE_API_BASE || '';
const TIMEZONE = 'Asia/Manila';

const TABS = [
  { id: 'overview', label: 'Overview' },
  { id: 'operations', label: 'Planting Work' },
  { id: 'ecology', label: 'Seedling Health' },
  { id: 'sites', label: 'Project Sites' },
];

const COLORS = {
  not_assigned: '#0f766e',
  available: '#16a34a',
  assigned: '#2563eb',
  planted_unverified: '#d97706',
  verified_alive: '#059669',
  verified_dead: '#dc2626',
  alive: '#059669',
  dead: '#dc2626',
  skipped: '#64748b',
  unavailable: '#ea580c',
  mapped: '#0f766e',
  planted: '#16a34a',
  pending: '#2563eb',
  completed: '#059669',
  danger: '#ef4444',
  canopy: '#7c3aed',
  coverage: '#0891b2',
  target: '#7c3aed',
  missing: '#94a3b8',
};

const LIFECYCLE_LABELS = POINT_STATUS_LABELS;

const LIFECYCLE_HINTS = {
  not_assigned: 'All open points not yet given to an organization',
  assigned: 'Given to an organization but not yet planted',
  planted: 'All seedlings currently recorded as planted',
  dead: 'Confirmed dead during monitoring',
  skipped: 'Could not be planted',
  unavailable: 'Excluded from planting because of site conditions',
};

const COUNT_FORMAT = new Intl.NumberFormat(undefined, { maximumFractionDigits: 0 });
const DECIMAL_FORMAT = new Intl.NumberFormat(undefined, { maximumFractionDigits: 1 });
const COMPACT_FORMAT = new Intl.NumberFormat(undefined, { notation: 'compact', maximumFractionDigits: 1 });

function arrayOf(value) {
  return Array.isArray(value) ? value : [];
}

function numberOrNull(value) {
  if (value === null || value === undefined || value === '') return null;
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : null;
}

function firstNumber(...values) {
  for (const value of values) {
    const parsed = numberOrNull(value);
    if (parsed !== null) return parsed;
  }
  return null;
}

function buildOrganizationWorkload(data) {
  const assignments = arrayOf(data?.assignments);

  if (assignments.length) {
    const grouped = new Map();

    assignments.forEach((row) => {
      const organizationId = firstNumber(row.organization_id, row.site_id);
      const organizationName = row.organization_name
        || row.organization?.name
        || row.site_name
        || row.project_site_name
        || 'No organization assigned';
      const key = organizationId !== null
        ? `organization:${organizationId}`
        : `organization-name:${organizationName.trim().toLocaleLowerCase()}`;
      const pending = firstNumber(row.pending, 0) ?? 0;
      const completed = firstNumber(row.completed, 0) ?? 0;
      const skipped = firstNumber(row.skipped, 0) ?? 0;
      const total = firstNumber(row.total, pending + completed + skipped) ?? 0;
      const current = grouped.get(key) || {
        organization_id: organizationId,
        organization_name: organizationName,
        organization_label: organizationName,
        pending: 0,
        completed: 0,
        skipped: 0,
        total: 0,
      };

      grouped.set(key, {
        ...current,
        pending: current.pending + pending,
        completed: current.completed + completed,
        skipped: current.skipped + skipped,
        total: current.total + total,
      });
    });

    const organizationRows = [...grouped.values()].filter((row) => row.total > 0);
    if (organizationRows.length) return [...organizationRows,
      ...arrayOf(data?.workload).filter((row) => row.owner_type === 'lgu')
        .map((row) => ({ ...row, organization_label: row.organization_name }))];
  }

  return arrayOf(data?.workload).map((row) => {
    const organizationName = row.organization_name
      || row.organization?.name
      || row.site_name
      || row.project_site_name
      || 'No organization assigned';
    return {
      ...row,
      organization_name: organizationName,
      organization_label: organizationName,
    };
  });
}

function formatCount(value) {
  const parsed = numberOrNull(value);
  return parsed === null ? '—' : COUNT_FORMAT.format(parsed);
}

function formatPercent(value) {
  const parsed = numberOrNull(value);
  return parsed === null ? '—' : `${DECIMAL_FORMAT.format(parsed)}%`;
}

function formatDate(value, includeTime = false) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat(undefined, includeTime
    ? { dateStyle: 'medium', timeStyle: 'short', timeZone: TIMEZONE }
    : { dateStyle: 'medium', timeZone: TIMEZONE }).format(date);
}

function formatShortDate(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat(undefined, {
    month: 'short', day: 'numeric', timeZone: TIMEZONE,
  }).format(date);
}

function formatMonth(value) {
  if (!value) return '—';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat(undefined, {
    month: 'short', year: 'numeric', timeZone: TIMEZONE,
  }).format(date);
}

function dateInManila(date = new Date()) {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: TIMEZONE,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(date);
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}`;
}

function makeYtdFilters() {
  const today = dateInManila();
  return {
    dateFrom: `${today.slice(0, 4)}-01-01`,
    dateTo: today,
    siteId: '',
  };
}

function makePlantingGoalsForm(settings) {
  return {
    annualTarget: settings?.annual_planting_target ?? '',
    survivalTarget: settings?.min_survival_target_pct ?? '',
  };
}

function optionId(option) {
  if (option === null || option === undefined) return '';
  if (typeof option !== 'object') return String(option);
  return String(option.id ?? option.value ?? option.key ?? '');
}

function optionLabel(option) {
  if (option === null || option === undefined) return 'Unknown';
  if (typeof option !== 'object') return String(option);
  return String(option.name ?? option.title ?? option.label ?? option.id ?? 'Unknown');
}

async function fetchJson(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw submissionError(body?.detail || body?.message, `Request failed (${response.status})`);
  }
  return body && typeof body === 'object' ? body : {};
}

function EmptyState({ title, children }) {
  return (
    <div className="dash-empty" role="status">
      <div className="dash-empty-mark" aria-hidden="true">○</div>
      <strong>{title}</strong>
      <p>{children}</p>
    </div>
  );
}

function LoadingState() {
  return (
    <div className="dash-loading" role="status" aria-live="polite">
      <span className="dash-spinner" aria-hidden="true" />
      <span>Loading dashboard statistics…</span>
    </div>
  );
}

function ErrorBanner({ message, compact = false, title = 'Could not load this data.' }) {
  return (
    <div className={`dash-error${compact ? ' is-compact' : ''}`} role="alert">
      <strong>{title}</strong> {message || 'Please try again.'}
    </div>
  );
}

function DatasetBoundary({ loading, error, data, children }) {
  if (loading && !data) return <LoadingState />;
  if (error && !data) return <ErrorBanner message={error} />;
  return (
    <>
      {error ? <ErrorBanner compact message={`${error} Showing the last successful result.`} /> : null}
      {children}
    </>
  );
}

function DataTable({ caption, columns, rows }) {
  return (
    <div className="dash-table-wrap">
      <table className="dash-table">
        <caption className="sr-only">{caption}</caption>
        <thead>
          <tr>{columns.map((column) => <th key={column.key} scope="col">{column.label}</th>)}</tr>
        </thead>
        <tbody>
          {rows.map((row, index) => (
            <tr key={row.id ?? row.key ?? row.assignment_id ?? `${caption}-${index}`}>
              {columns.map((column) => (
                <td key={column.key} data-label={column.label}>
                  {column.render ? column.render(row, index) : (row[column.key] ?? '—')}
                </td>
              ))}
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}

function Card({ title, subtitle, actions, className = '', children }) {
  return (
    <section className={`dash-card ${className}`.trim()}>
      <div className="dash-card-head">
        <div>
          <h2>{title}</h2>
          {subtitle ? <p>{subtitle}</p> : null}
        </div>
        {actions ? <div className="dash-card-actions">{actions}</div> : null}
      </div>
      <div className="dash-card-body">{children}</div>
    </section>
  );
}

function ChartCard({
  title,
  subtitle,
  actions,
  footer,
  data,
  tableData,
  columns,
  emptyTitle = 'Not enough data yet',
  emptyHint = 'Records will appear here when the required source data is available.',
  chartLabel,
  xLabel,
  yLabel,
  rightYLabel,
  height = 300,
  className = '',
  children,
}) {
  const chartRows = arrayOf(data);
  const rows = tableData === undefined ? chartRows : arrayOf(tableData);
  const hasAxisLabels = Boolean(xLabel || yLabel || rightYLabel);
  return (
    <Card title={title} subtitle={subtitle} actions={actions} className={className}>
      {chartRows.length ? (
        <>
          <div className={`dash-chart${hasAxisLabels ? ' has-axis-labels' : ''}${rightYLabel ? ' has-right-axis-label' : ''}`} style={{ height }} role="img" aria-label={chartLabel || title} tabIndex="0">
            {hasAxisLabels ? (
              <>
                {yLabel ? <span className="dash-axis-label dash-axis-label-y">{yLabel}</span> : null}
                <div className="dash-chart-plot">{children}</div>
                {rightYLabel ? <span className="dash-axis-label dash-axis-label-y dash-axis-label-right">{rightYLabel}</span> : null}
                {xLabel ? <span className="dash-axis-label dash-axis-label-x">{xLabel}</span> : null}
              </>
            ) : children}
          </div>
          {footer || null}
          {columns?.length && rows.length ? (
            <details className="dash-table-disclosure">
              <summary>View data table</summary>
              <DataTable caption={`${title} data`} columns={columns} rows={rows} />
            </details>
          ) : null}
        </>
      ) : <EmptyState title={emptyTitle}>{emptyHint}</EmptyState>}
    </Card>
  );
}

function KpiCard({ label, value, hint, tone = 'green', progress = null }) {
  const safeProgress = firstNumber(progress);
  return (
    <article className={`dash-kpi dash-kpi-${tone}`}>
      <span className="dash-kpi-accent" aria-hidden="true" />
      <div className="dash-kpi-label">{label}</div>
      <div className="dash-kpi-value">{value}</div>
      <div className="dash-kpi-hint">{hint || 'As of today'}</div>
      {safeProgress !== null ? (
        <div className="dash-kpi-progress" aria-label={`${label}: ${formatPercent(safeProgress)}`}>
          <span style={{ width: `${Math.max(0, Math.min(100, safeProgress))}%` }} />
        </div>
      ) : null}
    </article>
  );
}

function KpiStrip({ children, className = '' }) {
  return <div className={`dash-kpis ${className}`.trim()}>{children}</div>;
}

function LifecycleChart({ rows }) {
  const chartRows = rows.map((row) => ({
    ...row,
    name: LIFECYCLE_LABELS[row.key] || row.label,
    value: firstNumber(row.value, row.count, 0) ?? 0,
    color: POINT_STATUS_COLORS[row.key] || COLORS[row.key] || '#64748b',
  }));
  const total = chartRows.reduce((sum, row) => sum + row.value, 0);
  return (
    <div className="dash-lifecycle-visual">
      <div className="dash-lifecycle-donut">
        <ResponsiveContainer width="100%" height="100%">
          <PieChart>
            <Pie
              data={chartRows}
              dataKey="value"
              nameKey="name"
              cx="50%"
              cy="50%"
              innerRadius="58%"
              outerRadius="82%"
              paddingAngle={2}
              stroke="#ffffff"
              strokeWidth={2}
            >
              {chartRows.map((row) => <Cell key={row.key} fill={row.color} />)}
            </Pie>
            <Tooltip formatter={(value, name) => [formatCount(value), name]} />
          </PieChart>
        </ResponsiveContainer>
        <div className="dash-lifecycle-total" aria-hidden="true">
          <strong>{formatCount(total)}</strong>
          <span>Total points</span>
        </div>
      </div>
      <div className="dash-lifecycle-legend" aria-label="Planting status legend">
        {chartRows.map((row) => (
          <div className="dash-lifecycle-legend-row" key={row.key}>
            <span className="dash-lifecycle-swatch" style={{ backgroundColor: row.color }} aria-hidden="true" />
            <div>
              <strong>{row.name}</strong>
              <small>{LIFECYCLE_HINTS[row.key] || 'Current planting status'}</small>
            </div>
            <b>{formatCount(row.value)}</b>
            <span className="dash-lifecycle-percent">{total ? formatPercent((row.value / total) * 100) : '0%'}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

function OverviewTab({ data }) {
  const kpis = data?.kpis || {};
  const planted = kpis.seedlings_planted || {};
  const organizations = kpis.total_organizations || {};
  const lifecycleCounts = Object.fromEntries(arrayOf(data?.lifecycle).map((row) => [
    row.key,
    firstNumber(row?.value, row?.count, 0) ?? 0,
  ]));
  const lifecycle = [
    { key: 'not_assigned', value: lifecycleCounts.available || 0 },
    { key: 'assigned', value: lifecycleCounts.assigned || 0 },
    {
      key: 'planted',
      value: (lifecycleCounts.planted_unverified || 0) + (lifecycleCounts.verified_alive || 0),
    },
    { key: 'dead', value: lifecycleCounts.dead || 0 },
    { key: 'skipped', value: lifecycleCounts.skipped || 0 },
    { key: 'unavailable', value: lifecycleCounts.unavailable || 0 },
  ].filter((row) => row.value > 0);
  const annualTarget = firstNumber(planted.target);
  const progress = arrayOf(data?.planting_progress).reduce((rows, row) => {
    const lastCumulative = firstNumber(rows.at(-1)?.cumulative_planted, 0) ?? 0;
    const reportedCumulative = firstNumber(row.cumulative_planted, lastCumulative) ?? lastCumulative;
    const cumulativePlanted = Math.max(lastCumulative, reportedCumulative);
    return [...rows, {
      ...row,
      label: formatShortDate(row.period_start ?? row.date ?? row.week),
      cumulative_planted: cumulativePlanted,
      remaining_to_target: annualTarget === null ? null : Math.max(annualTarget - cumulativePlanted, 0),
    }];
  }, []);
  const progressHasActivity = progress.length > 0 && (
    annualTarget !== null || progress.some((row) => firstNumber(row.planted, row.cumulative_planted, 0) > 0)
  );
  const plantingScope = planted.scope || 'year_to_date';
  const plantingHint = plantingScope === 'year_to_date'
    ? numberOrNull(planted.target) === null
      ? 'This year so far · no yearly goal set'
      : `This year so far · goal: ${formatCount(planted.target)} seedlings`
    : plantingScope === 'filtered'
      ? 'Based on the selected project site'
      : 'Based on the selected dates';

  return (
    <div className="dash-tab-panel">
      <KpiStrip>
        <KpiCard
          label="Seedlings planted"
          value={formatCount(planted.value)}
          hint={plantingHint}
          progress={planted.progress_pct}
        />
        <KpiCard
          label="Total organizations"
          value={formatCount(organizations.value)}
          hint="Registered organizations, including LGU"
          tone="blue"
        />
      </KpiStrip>

      <div className="dash-grid">
        <ChartCard
          title="Current status of planting locations"
          subtitle="Current map locations for the selected project site, across all dates. Choose all project sites to compare with Planting Map. Released locations return to Planned; their original deaths stay in history."
          data={lifecycle}
          chartLabel="Cards showing the current status of all planting points"
          columns={[
            { key: 'label', label: 'Planting status', render: (row) => LIFECYCLE_LABELS[row.key] || row.label },
            { key: 'value', label: 'Planting points', render: (row) => formatCount(row.value ?? row.count) },
          ]}
          emptyHint="Planting points will appear here after a map analysis is saved."
          height={310}
        >
          <LifecycleChart rows={lifecycle} />
        </ChartCard>

        <ChartCard
          title="Planting progress"
          subtitle={annualTarget === null
            ? 'The total planted line only moves upward as more seedlings are recorded.'
            : `${formatCount(annualTarget)}-seedling annual goal. The seedlings still needed line moves down as planting increases.`}
          data={progressHasActivity ? progress : []}
          tableData={progress}
          chartLabel="Lines showing total seedlings planted increasing and seedlings still needed for the annual goal decreasing"
          xLabel="Week"
          yLabel="Number of seedlings"
          columns={[
            { key: 'period_start', label: 'Week', render: (row) => formatDate(row.period_start) },
            { key: 'planted', label: 'Planted this week', render: (row) => formatCount(row.planted) },
            { key: 'cumulative_planted', label: 'Total planted so far', render: (row) => formatCount(row.cumulative_planted) },
            { key: 'remaining_to_target', label: 'Still needed for annual goal', render: (row) => formatCount(row.remaining_to_target) },
          ]}
          emptyHint="Planting progress will appear after planted seedlings are recorded."
        >
          <ResponsiveContainer width="100%" height="100%">
            <LineChart data={progress} margin={{ top: 12, right: 18, bottom: 34, left: 18 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="label" tick={{ fontSize: 11 }} minTickGap={22} />
              <YAxis tickFormatter={(value) => COMPACT_FORMAT.format(value)} width={48} />
              <Tooltip formatter={(value, name) => [formatCount(value), name]} labelFormatter={(_, payload) => formatDate(payload?.[0]?.payload?.period_start)} />
              <Legend verticalAlign="top" height={34} />
              <Line type="monotone" dataKey="cumulative_planted" name="Total planted so far" stroke="#047857" strokeWidth={3} dot={false} />
              {annualTarget !== null ? <Line type="monotone" dataKey="remaining_to_target" name="Still needed for annual goal" stroke={COLORS.target} strokeWidth={2.5} strokeDasharray="7 4" connectNulls dot={false} /> : null}
            </LineChart>
          </ResponsiveContainer>
        </ChartCard>

      </div>
    </div>
  );
}

function WrappedCategoryTick({ x, y, payload }) {
  const lines = String(payload?.value || 'No organization assigned')
    .split(/\s+/)
    .reduce((result, word) => {
      const currentLine = result.at(-1) || '';
      const combined = currentLine ? `${currentLine} ${word}` : word;
      return combined.length <= 22
        ? [...result.slice(0, -1), combined]
        : [...result, word];
    }, []);
  return (
    <g transform={`translate(${x},${y})`}>
      <text x={-8} y={0} textAnchor="end" fill="#6b7280" fontSize={11}>
        {lines.map((line, index) => (
          <tspan key={`${line}-${index}`} x={-8} dy={index === 0 ? `${-0.55 * (lines.length - 1)}em` : '1.1em'}>
            {line}
          </tspan>
        ))}
      </text>
    </g>
  );
}

function PercentageBarLabel({ x, y, width, height, value }) {
  const labelInsideBar = Number(width) >= 52;
  return (
    <text
      x={labelInsideBar ? Number(x) + Number(width) - 8 : Number(x) + Number(width) + 8}
      y={Number(y) + Number(height) / 2}
      dy="0.35em"
      textAnchor={labelInsideBar ? 'end' : 'start'}
      fill={labelInsideBar ? '#ffffff' : '#047857'}
      fontSize={11}
      fontWeight={700}
    >
      {formatPercent(value)}
    </text>
  );
}

function OperationsTab({ data }) {
  const workload = buildOrganizationWorkload(data)
    .sort((a, b) => (firstNumber(b.total, b.pending + b.completed + b.skipped, 0) ?? 0) - (firstNumber(a.total, a.pending + a.completed + a.skipped, 0) ?? 0));
  const throughput = arrayOf(data?.throughput).map((row) => ({ ...row, label: formatShortDate(row.period_start) }));
  const throughputHasActivity = throughput.some((row) => (
    firstNumber(row.assigned, 0) + firstNumber(row.completed, 0) + firstNumber(row.skipped, 0)
  ) > 0);
  const assignments = arrayOf(data?.assignments);

  return (
    <div className="dash-tab-panel">
      <div className="dash-grid">
        <ChartCard
          title="Planting work by organization and LGU"
          subtitle="Planting work grouped by the responsible organization, with LGU replacement work shown separately."
          className="dash-card-full"
          data={workload}
          chartLabel="Stacked horizontal bars showing planting work by organization"
          xLabel="Number of planting points"
          yLabel="Organization"
          height={Math.max(300, Math.min(560, workload.length * 70 + 90))}
          columns={[
            { key: 'organization_name', label: 'Organization', render: (row) => row.organization_name || 'No organization assigned' },
            { key: 'pending', label: 'Not yet planted', render: (row) => formatCount(row.pending) },
            { key: 'completed', label: 'Planted', render: (row) => formatCount(row.completed) },
            { key: 'skipped', label: 'Could not be planted', render: (row) => formatCount(row.skipped) },
            { key: 'total', label: 'Total', render: (row) => formatCount(row.total) },
          ]}
          emptyHint="Assign planting points to an organization to show its planting work here."
        >
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={workload} layout="vertical" margin={{ top: 8, right: 18, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" allowDecimals={false} />
              <YAxis type="category" dataKey="organization_label" width={160} tick={<WrappedCategoryTick />} interval={0} />
              <Tooltip formatter={(value, name) => [formatCount(value), name]} />
              <Legend verticalAlign="top" height={34} />
              <Bar dataKey="pending" name="Not yet planted" stackId="work" fill={COLORS.pending} />
              <Bar dataKey="completed" name="Planted" stackId="work" fill={COLORS.completed} />
              <Bar dataKey="skipped" name="Could not be planted" stackId="work" fill={COLORS.skipped} radius={[0, 4, 4, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Planting work over time"
          subtitle="Points assigned, planted, or unable to be planted during the selected dates."
          data={throughputHasActivity ? throughput : []}
          tableData={throughput}
          chartLabel="Time series showing assignment throughput"
          xLabel="Week"
          yLabel="Number of planting points"
          className="dash-card-full"
          columns={[
            { key: 'period_start', label: 'Period', render: (row) => formatDate(row.period_start) },
            { key: 'assigned', label: 'Assigned', render: (row) => formatCount(row.assigned) },
            { key: 'completed', label: 'Planted', render: (row) => formatCount(row.completed) },
            { key: 'skipped', label: 'Could not be planted', render: (row) => formatCount(row.skipped) },
          ]}
          emptyHint="Dated assignment activity is required before throughput can be shown."
        >
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={throughput} margin={{ top: 12, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="label" minTickGap={22} tick={{ fontSize: 11 }} />
              <YAxis allowDecimals={false} width={44} />
              <Tooltip formatter={(value, name) => [formatCount(value), name]} labelFormatter={(_, payload) => formatDate(payload?.[0]?.payload?.period_start)} />
              <Legend verticalAlign="top" height={34} />
              <Bar dataKey="assigned" name="Assigned" fill={COLORS.assigned} radius={[3, 3, 0, 0]} />
              <Bar dataKey="completed" name="Planted" fill={COLORS.completed} radius={[3, 3, 0, 0]} />
              <Line type="monotone" dataKey="skipped" name="Could not be planted" stroke={COLORS.skipped} strokeWidth={2} />
            </ComposedChart>
          </ResponsiveContainer>
        </ChartCard>

        <Card title="Planting assignments" subtitle="Assignments with no planting locations are marked below and are not counted in planting totals." className="dash-card-full">
          {assignments.length ? (
            <DataTable
              caption="Filtered assignments"
              rows={assignments}
              columns={[
                { key: 'title', label: 'Assignment', render: (row) => <strong>{row.title || `Assignment ${row.assignment_id}`}</strong> },
                { key: 'site_name', label: 'Project site', render: (row) => row.site_name || 'Not linked' },
                { key: 'planter_name', label: 'Planter', render: (row) => row.planter_name || 'Unassigned' },
                { key: 'species', label: 'Species', render: (row) => row.species || 'Not recorded' },
                { key: 'pending', label: 'Not yet planted', render: (row) => formatCount(row.pending) },
                { key: 'completed', label: 'Planted', render: (row) => formatCount(row.completed) },
                { key: 'overdue', label: 'Past due', render: (row) => formatCount(row.overdue) },
                { key: 'status', label: 'Status', render: (row) => (
                  <span className={`dash-status is-${row.stale_zero_points ? 'warning' : 'neutral'}`}>
                    {row.stale_zero_points ? 'No planting locations' : (row.status || 'Not recorded')}
                  </span>
                ) },
              ]}
            />
          ) : <EmptyState title="No assignments match these filters">Try a broader period or create assignments from mapped planting locations.</EmptyState>}
        </Card>
      </div>
    </div>
  );
}

function StackedColumnCountLabel({ x, y, width, height, value }) {
  if (!value || height < 18) return null;
  return (
    <text x={x + width / 2} y={y + height / 2} textAnchor="middle" dominantBaseline="central" fill="#ffffff" fontSize={11} fontWeight={700}>
      {formatCount(value)}
    </text>
  );
}

function OutcomesChart({ rows, labelKey = 'name', verticalBars = false, plantedLabel = 'Planted' }) {
  const hasUnknown = rows.some((row) => row.unknown > 0);
  return (
    <ResponsiveContainer width="100%" height="100%">
      <BarChart data={rows} layout={verticalBars ? 'horizontal' : 'vertical'} margin={verticalBars ? { top: 24, right: 20, bottom: 8, left: 0 } : { top: 8, right: 28, bottom: 8, left: -14 }}>
        <CartesianGrid strokeDasharray="3 3" horizontal={verticalBars} vertical={!verticalBars} />
        {verticalBars ? <XAxis type="category" dataKey={labelKey} interval={0} tickMargin={10} height={40} /> : <XAxis type="number" allowDecimals={false} />}
        {verticalBars ? <YAxis type="number" allowDecimals={false} tickFormatter={formatCount} width={54} /> : <YAxis type="category" dataKey={labelKey} width={150} interval={0} tick={<WrappedCategoryTick />} />}
        <Tooltip formatter={(value, name) => [formatCount(value), name]} />
        <Legend verticalAlign="top" height={34} />
        <Bar dataKey="planted" name={plantedLabel} stackId="outcome" fill={COLORS.planted} maxBarSize={verticalBars ? 90 : undefined}>
          {verticalBars ? <LabelList dataKey="planted" content={<StackedColumnCountLabel />} /> : <LabelList dataKey="planted" position="center" fill="#ffffff" fontSize={11} fontWeight={700} formatter={formatCount} />}
        </Bar>
        <Bar dataKey="dead" name="Dead" stackId="outcome" fill={COLORS.dead} radius={verticalBars ? [4, 4, 0, 0] : [0, 4, 4, 0]} maxBarSize={verticalBars ? 90 : undefined}>
          {verticalBars ? <LabelList dataKey="dead" content={<StackedColumnCountLabel />} /> : <LabelList dataKey="dead" position="center" fill="#ffffff" fontSize={11} fontWeight={700} formatter={formatCount} />}
          {verticalBars && !hasUnknown ? <LabelList dataKey="total" position="top" fill="#14532d" fontSize={12} fontWeight={700} formatter={formatCount} /> : null}
        </Bar>
        {hasUnknown ? <Bar dataKey="unknown" name="Outcome not recorded" stackId="outcome" fill="#94a3b8" /> : null}
      </BarChart>
    </ResponsiveContainer>
  );
}

function EcologyTab({ data }) {
  const currentSummary = data?.summary || {};
  const performance = data?.planting_performance || {};
  const summary = performance.summary || {};
  const batches = arrayOf(performance.batches);
  const species = arrayOf(performance.species_outcomes);
  const sites = arrayOf(performance.site_outcomes);
  const speciesChart = species.filter((row) => firstNumber(row.total, 0) > 0);
  const siteChart = sites.filter((row) => firstNumber(row.total, 0) > 0);
  const healthRows = [
    { name: 'Planted', value: firstNumber(currentSummary.planted, 0), color: COLORS.planted },
    { name: 'Dead', value: firstNumber(currentSummary.dead, 0), color: COLORS.dead },
  ];
  const mortality = arrayOf(data?.mortality_causes).map((row) => ({ ...row, name: row.label || row.cause || 'Other' }));
  const monthlyDeaths = arrayOf(data?.monthly_deaths).map((row) => ({ ...row, label: formatMonth(row.period_start) }));
  const deathScope = data?.mortality_includes_unlocated === false
    ? 'Deaths without identified locations are excluded while location or species filters apply.'
    : 'Includes deaths whose locations are still unknown.';
  const seedlingGrowthGroups = arrayOf(data?.organization_growth?.growth_groups);
  const planterOutcomes = arrayOf(performance.planter_outcomes)
    .map((row) => ({ ...row, name: row.planter_name || row.name || `Planter ${row.planter_id ?? ''}` }));

  const healthColumns = [
    { key: 'total', label: 'Seedlings planted', render: (row) => formatCount(row.total) },
    { key: 'planted', label: 'Without recorded death', render: (row) => formatCount(row.planted) },
    { key: 'dead', label: 'Recorded deaths', render: (row) => formatCount(row.dead) },
    { key: 'unknown', label: 'Outcome not recorded', render: (row) => formatCount(row.unknown) },
    { key: 'survival_rate_pct', label: 'Recorded survival', render: (row) => formatPercent(row.survival_rate_pct) },
  ];
  const outcomeColumns = [{ key: 'name', label: 'Group' }, ...healthColumns];
  const planterOutcomeColumns = [
    { key: 'name', label: 'Planter' },
    { key: 'site_name', label: 'Project site', render: (row) => row.site_name || 'Not linked' },
    { key: 'species', label: 'Species', render: (row) => row.species || 'Not recorded' },
    ...healthColumns,
  ];

  return (
    <div className="dash-tab-panel">
      <div className="dash-method-note">
        <strong>Planting performance.</strong> Includes seedlings planted during the selected dates, with their recorded outcomes as of today. Replanting keeps the original death and counts the replacement as a new seedling. Seedlings without a recorded death may still be awaiting inspection.
      </div>
      <KpiStrip>
        <KpiCard label="Seedlings planted" value={formatCount(summary.total)} hint="Actual plantings in the selected dates, including replacements" tone="blue" />
        <KpiCard label="Without recorded death" value={formatCount(summary.planted)} hint="Includes seedlings awaiting inspection" />
        <KpiCard label="Recorded deaths" value={formatCount(summary.dead)} hint="Retained after site release or later planting" tone="red" />
        <KpiCard label="Recorded survival" value={formatPercent(summary.survival_rate_pct)} hint={`${formatCount(summary.planted)} without a recorded death out of ${formatCount(summary.total)} planted`} tone="emerald" progress={summary.survival_rate_pct} />
      </KpiStrip>
      {summary.unknown > 0 ? <p className="dash-method-note">{formatCount(summary.unknown)} earlier planting outcomes are missing. Survival is unavailable for affected groups until those records are complete.</p> : null}

      <div className="dash-grid">
        <Card title="Survival by planting batch" className="dash-card-full"
          subtitle="Grouped by planting date, organization, project site, species, and planting activity. Original and replacement plantings have separate results.">
          {batches.length ? <DataTable caption="Survival by planting batch" rows={batches} columns={[
            { key: 'planted_date', label: 'Planting date', render: (row) => formatShortDate(row.planted_date) },
            { key: 'organization_name', label: 'Organization' },
            { key: 'site_name', label: 'Project site' },
            { key: 'species', label: 'Species' },
            { key: 'planting_type', label: 'Planting', render: (row) => row.planting_type === 'replacement' ? 'Replacement' : 'Original' },
            ...healthColumns,
          ]} /> : <EmptyState title="No planting batches in these dates">Choose dates containing recorded plantings to review their results.</EmptyState>}
        </Card>

        <ChartCard
          title="Current mapped seedlings"
          subtitle="Current planted and dead locations across all planting dates, matching the map. Released locations appear as Planned; their original deaths stay in planting performance above."
          data={healthRows.some((row) => row.value > 0) ? healthRows : []}
          tableData={healthRows}
          chartLabel="Counts of current planted and dead seedlings"
          xLabel="Number of seedlings"
          yLabel="Planting status"
          className="dash-card-full"
          columns={[
            { key: 'name', label: 'Planting status' },
            { key: 'value', label: 'Seedlings', render: (row) => formatCount(row.value) },
          ]}
          emptyHint="Record a planting to see overall seedling totals."
        >
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={healthRows} layout="vertical" margin={{ top: 12, right: 42, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" allowDecimals={false} />
              <YAxis type="category" dataKey="name" width={160} interval={0} tick={<WrappedCategoryTick />} />
              <Tooltip formatter={(value) => [formatCount(value), 'Seedlings']} />
              <Bar dataKey="value" name="Seedlings" radius={[0, 5, 5, 0]} maxBarSize={38}>
                {healthRows.map((row) => <Cell key={row.name} fill={row.color} />)}
                <LabelList dataKey="value" position="right" formatter={formatCount} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Seedlings by species"
          subtitle="Recorded outcomes for seedlings planted during the selected dates. Earlier deaths remain included after replanting."
          data={speciesChart}
          tableData={species}
          chartLabel="Vertical stacked columns showing planting-batch outcomes for each species"
          xLabel="Species"
          yLabel="Number of seedlings"
          height={420}
          columns={outcomeColumns.map((column) => column.key === 'name' ? { ...column, label: 'Species' } : column)}
          emptyHint="Record plantings with a species to compare totals here."
        >
          <OutcomesChart rows={speciesChart} verticalBars plantedLabel="Without recorded death" />
        </ChartCard>

        <ChartCard
          title="Seedlings by project site"
          subtitle="Recorded outcomes for seedlings planted during the selected dates, including original and replacement plantings."
          data={siteChart}
          tableData={sites}
          chartLabel="Counts of planting-batch outcomes in each project site"
          xLabel="Number of seedlings"
          yLabel="Project site"
          height={Math.max(300, Math.min(560, sites.length * 70 + 90))}
          columns={outcomeColumns.map((column) => column.key === 'name' ? { ...column, label: 'Project site' } : column)}
          emptyHint="Record plantings linked to project sites to compare totals here."
        >
          <OutcomesChart rows={siteChart} plantedLabel="Without recorded death" />
        </ChartCard>

        <ChartCard
          title="Seedling deaths by month"
          subtitle={`New deaths recorded in each month during the selected dates, counted once. ${deathScope}`}
          data={monthlyDeaths}
          chartLabel="Monthly number of seedlings recorded dead"
          xLabel="Month"
          yLabel="Number of dead seedlings"
          className="dash-card-full"
          columns={[
            { key: 'period_start', label: 'Month', render: (row) => formatMonth(row.period_start) },
            { key: 'deaths', label: 'Seedlings recorded dead', render: (row) => formatCount(row.deaths) },
          ]}
          emptyTitle="No recorded death history yet"
          emptyHint="Monthly counts will appear when the recorded death history is available for these dates."
        >
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={monthlyDeaths} margin={{ top: 12, right: 16, bottom: 8, left: 0 }}>
              <CartesianGrid strokeDasharray="3 3" vertical={false} />
              <XAxis dataKey="label" minTickGap={22} tick={{ fontSize: 11 }} />
              <YAxis allowDecimals={false} width={44} />
              <Tooltip formatter={(value) => [formatCount(value), 'Seedlings recorded dead']}
                labelFormatter={(_, payload) => formatMonth(payload?.[0]?.payload?.period_start)} />
              <Legend verticalAlign="top" height={34} />
              <Bar dataKey="deaths" name="Seedlings recorded dead" fill={COLORS.dead} radius={[3, 3, 0, 0]} />
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Why seedlings died"
          subtitle={`Deaths reported during the selected period, counted once across monitoring visits and identified plants. ${deathScope}`}
          data={mortality}
          chartLabel="Bars showing deaths by cause and a line showing each cause's percentage of all deaths"
          xLabel="Number of dead seedlings and share of all deaths (%)"
          yLabel="Cause"
          className="dash-card-full"
          height={Math.max(300, Math.min(520, mortality.length * 42 + 90))}
          columns={[
            { key: 'name', label: 'Cause' },
            { key: 'deaths', label: 'Seedlings recorded dead', render: (row) => formatCount(row.deaths) },
            { key: 'percent_of_deaths', label: 'Share of all deaths', render: (row) => formatPercent(row.percent_of_deaths) },
          ]}
          emptyHint="This graph will appear after an inspector records a dead seedling and selects a cause."
        >
          <ResponsiveContainer width="100%" height="100%">
            <ComposedChart data={mortality} layout="vertical" margin={{ top: 12, right: 26, bottom: 8, left: 18 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis xAxisId="count" type="number" allowDecimals={false} />
              <XAxis xAxisId="percent" type="number" orientation="top" domain={[0, 100]} unit="%" />
              <YAxis type="category" dataKey="name" width={124} tick={{ fontSize: 11 }} />
              <Tooltip formatter={(value, name, item) => [item.dataKey === 'percent_of_deaths' ? formatPercent(value) : formatCount(value), name]} />
              <Legend verticalAlign="top" height={34} />
              <Bar xAxisId="count" dataKey="deaths" name="Seedlings recorded dead" fill={COLORS.dead} radius={[0, 5, 5, 0]} />
              <Line xAxisId="percent" type="linear" dataKey="percent_of_deaths" name="Share of all deaths" stroke={COLORS.target} strokeWidth={2.5} />
            </ComposedChart>
          </ResponsiveContainer>
        </ChartCard>

        <ChartCard
          title="Living seedlings by growth group"
          subtitle="All organizations combined. Latest visit counts include plantings awaiting inspection."
          actions={<Link to="/monitoring">Record a visit →</Link>}
          data={seedlingGrowthGroups}
          chartLabel="Bar graph showing how many living seedlings are in each growth group across all organizations"
          xLabel="Number of living seedlings"
          yLabel="Growth group"
          className="dash-card-full"
          height={Math.max(240, seedlingGrowthGroups.length * 64 + 100)}
          columns={[
            { key: 'growth_group', label: 'Growth group' },
            { key: 'seedling_count', label: 'Living seedlings', render: (row) => formatCount(row.seedling_count) },
          ]}
          emptyTitle="No planting ages yet"
          emptyHint="This graph will appear after planting dates or monitoring visits are recorded."
          footer={<>
            <p className="dash-growth-note">
              Counts use seedlings recorded as alive at the latest visit, plus planted seedlings awaiting a visit. Survival of uninspected seedlings is not yet verified.
            </p>
            {firstNumber(data?.organization_growth?.unclassified_living_count, 0) > 0 ? (
              <p className="dash-growth-note">Some seedlings are not shown because their planting date was not recorded.</p>
            ) : null}
            <GrowthGuide />
          </>}
        >
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={seedlingGrowthGroups} layout="vertical" margin={{ top: 12, right: 56, bottom: 8, left: 12 }}>
              <CartesianGrid strokeDasharray="3 3" horizontal={false} />
              <XAxis type="number" allowDecimals={false} />
              <YAxis type="category" dataKey="growth_group" width={130} tick={{ fontSize: 12 }} />
              <Tooltip formatter={(value) => [formatCount(value), 'Living seedlings']} />
              <Bar dataKey="seedling_count" name="Living seedlings" fill={COLORS.planted} radius={[0, 5, 5, 0]} barSize={28}>
                <LabelList dataKey="seedling_count" position="right" formatter={formatCount} fill="#14532d" fontSize={12} fontWeight={700} />
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </ChartCard>

        <Card title="Survival by organization" className="dash-card-full"
          subtitle="Each organization appears once. LGU combines all its sites using total seedling counts.">
          <DataTable caption="Planting performance by organization" rows={arrayOf(performance.organization_outcomes)} columns={[
            { key: 'organization_name', label: 'Organization' }, ...healthColumns,
          ]} />
        </Card>

        <details className="dash-insight-drawer">
          <summary>More information about planting results</summary>
          <div className="dash-insight-drawer-body">
            <p className="dash-caveat">
              Planting conditions, species, and planting dates affect outcomes. Use these totals to review work at each project site.
            </p>
            {planterOutcomes.length ? (
              <DataTable caption="Planting performance by planter, project site, and species" rows={planterOutcomes} columns={planterOutcomeColumns} />
            ) : <EmptyState title="No planting results yet">Results will appear here after plantings are recorded.</EmptyState>}
          </div>
        </details>
      </div>
    </div>
  );
}

const SITE_STATUSES = ['planned', 'assigned', 'planted', 'dead', 'skipped', 'unavailable'];

function SitesTab({ data, onViewHistory }) {
  const summary = data?.summary || {};
  const sites = arrayOf(data?.project_sites);
  return (
    <div className="dash-tab-panel">
      <KpiStrip className="dash-site-kpis">
        <KpiCard label="Project sites" value={formatCount(summary.project_sites)} hint="Sites in this view, including released sites" />
        <KpiCard label="Mapped locations" value={formatCount(summary.mapped_points)} hint="Currently owned by these sites" tone="blue" />
        <KpiCard label="Sites with warnings" value={formatCount(summary.sites_with_warnings)} hint="Mapped warnings or locations in eroded zones" tone="amber" />
      </KpiStrip>
      <Card title="Compare project sites" className="dash-site-comparison"
        subtitle="Current locations across all planting dates. Each location counts once; released locations leave their former site's totals."
        actions={onViewHistory ? <button type="button" className="dash-link-button" onClick={onViewHistory}>View planting history</button> : null}>
        {sites.length ? <DataTable caption="Current project site comparison" rows={sites} columns={[
          { key: 'name', label: 'Project site', render: (site) => <div className="dash-site-name">
            <strong>{site.name || `Site ${site.id}`}</strong>
            {site.area_released_at ? <span className="dash-status">Area released</span> : null}
          </div> },
          { key: 'organization_name', label: 'Organization', render: (site) => site.organization_name || 'No organization recorded' },
          { key: 'point_count', label: 'Locations', render: (site) => <strong>{formatCount(site.point_count)}</strong> },
          { key: 'point_status_counts', label: 'Current status', render: (site) => <dl className="dash-site-statuses">
            {SITE_STATUSES.map((status) => <div key={status}>
              <dt><i aria-hidden="true" style={{ backgroundColor: POINT_STATUS_COLORS[status] }} />{POINT_STATUS_LABELS[status]}</dt>
              <dd>{formatCount(site.point_status_counts?.[status])}</dd>
            </div>)}
          </dl> },
          { key: 'warnings', label: 'Warnings', render: (site) => <div className="dash-site-warnings">
            {site.eroded_point_count > 0 || arrayOf(site.warnings).length ? <ul>
              {site.eroded_point_count > 0 ? <li>In eroded zones: <strong>{formatCount(site.eroded_point_count)}</strong> locations</li> : null}
              {arrayOf(site.warnings).map((warning) => <li key={`${warning.warning_type}-${warning.severity}`}>
                {warning.warning_type === 'high_mortality' ? 'High recorded mortality' : warning.label}
                : <strong>{formatCount(warning.count)}</strong> locations<small> · {warning.severity}</small>
              </li>)}
            </ul> : <span>No mapped warnings</span>}
          </div> },
          { key: 'actions', label: 'Open', render: (site) => <div className="dash-site-actions">
            {site.geometry ? <MapLink className="dash-table-map-link"
              to={`/map?project_site_id=${encodeURIComponent(site.id)}&focus=site_points`}
              state={{ mapFocusSite: site }} aria-label={`View ${site.name || `Site ${site.id}`} on the map`}>View map</MapLink>
              : <span className="dash-site-no-boundary">No current boundary</span>}
            <Link className="dash-table-map-link" to={`/monitoring?project_site_id=${encodeURIComponent(site.id)}`}
              aria-label={`Open Monitoring for ${site.name || `Site ${site.id}`}`}>Monitoring</Link>
          </div> },
        ]} /> : <EmptyState title="No project sites in this view">Choose another project site, or use Assign Points to create a site when assigning available locations.</EmptyState>}
        <p className="dash-site-scope">Unclaimed locations appear on Planting Map. Warning categories can overlap; each category counts a location once. Mortality warnings use recorded planting history.</p>
        <p className="dash-site-analysis-link">Image area and canopy measurements are available in <Link to="/processing?action=review">Analyze Image</Link>.</p>
      </Card>
    </div>
  );
}

function PlantingGoalsForm({ settings, year, loading, error, onSaved }) {
  const feedback = useFormFeedback({
    annualTarget: { label: 'Yearly planting goal', aliases: ['annual_planting_target'] },
    survivalTarget: { label: 'Target percentage alive', aliases: ['min_survival_target_pct'] },
  });
  const [form, setForm] = useState(() => makePlantingGoalsForm(settings));
  const appliedSettings = useRef(JSON.stringify(settings));
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState('');
  const [review, setReview] = useState(null);
  const submitting = useRef(false);

  useEffect(() => {
    // Cached API reads create fresh objects even when saved values are unchanged.
    // Preserve the user's draft when returning to this retained form.
    const settingsSnapshot = JSON.stringify(settings);
    if (appliedSettings.current === settingsSnapshot) return;
    appliedSettings.current = settingsSnapshot;
    const refreshedForm = makePlantingGoalsForm(settings);
    queueMicrotask(() => setForm(refreshedForm));
  }, [settings]);

  const save = async (event) => {
    event.preventDefault();
    if (submitting.current || loading) return;
    setFormError('');
    if (!feedback.validate()) return;
    const annualTarget = form.annualTarget === '' ? null : Number(form.annualTarget);
    const survivalTarget = form.survivalTarget === '' ? null : Number(form.survivalTarget);
    if (annualTarget !== null && (!Number.isInteger(annualTarget) || annualTarget < 0)) {
      feedback.reject({ annualTarget: 'Enter zero or a positive whole number for the yearly planting goal, or leave it blank.' });
      return;
    }
    if (survivalTarget !== null && (!Number.isFinite(survivalTarget) || survivalTarget < 0 || survivalTarget > 100)) {
      feedback.reject({ survivalTarget: 'Enter a target percentage from 0 to 100, or leave it blank.' });
      return;
    }
    setReview({ year, annual_planting_target: annualTarget, min_survival_target_pct: survivalTarget });
  };

  const confirmSave = async () => {
    if (!review || submitting.current || loading) return;
    submitting.current = true;
    setSaving(true);
    try {
      const result = await fetchJson('/api/dashboard/settings', {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(review),
      });
      onSaved(result);
    } catch (saveError) {
      if (!feedback.fromServer(saveError)) setFormError(saveError.message || 'Could not save planting goals.');
    } finally {
      submitting.current = false;
      setReview(null);
      setSaving(false);
    }
  };

  if (!settings && !error) return <LoadingState />;
  if (error && !settings) return <ErrorBanner compact title="Planting goals could not be loaded" message="Reload the page to try again before editing goals for this reporting year." />;
  return (
    <><form className="dash-goals" noValidate onChangeCapture={feedback.onChange} onSubmit={save}>
      <FormErrorSummary feedback={feedback} />
      {error ? <ErrorBanner compact message={error} /> : null}
      {formError ? <ErrorBanner compact title="Planting goals not saved." message={formError} /> : null}
      <div className="dash-goals-grid">
        <label>
          <span>Seedlings to plant this year</span>
          <input {...feedback.props('annualTarget')} type="number" min="0" step="1" placeholder="Optional" value={form.annualTarget} onChange={(event) => setForm((current) => ({ ...current, annualTarget: event.target.value }))} />
              <FieldError feedback={feedback} field="annualTarget" />
        </label>
        <label>
          <span>Target percentage of seedlings alive (%)</span>
          <input {...feedback.props('survivalTarget')} type="number" min="0" max="100" step="0.1" placeholder="Optional" value={form.survivalTarget} onChange={(event) => setForm((current) => ({ ...current, survivalTarget: event.target.value }))} />
              <FieldError feedback={feedback} field="survivalTarget" />
        </label>
      </div>
      <div className="dash-goals-actions">
        <span>{settings?.updated_at ? `Last updated ${formatDate(settings.updated_at, true)}` : 'Targets are optional.'}</span>
        <button className="dash-primary-button" type="submit" disabled={saving || loading}>{saving ? 'Saving…' : 'Save planting goals'}</button>
      </div>
    </form>
    {review && createPortal(<Modal open title="Save planting goals?" variant="warning" busy={saving}
      initialFocus="cancel" confirmLabel="Save goals" cancelLabel="Cancel" confirmDisabled={loading}
      onConfirm={confirmSave} onCancel={() => { if (!saving) setReview(null); }}>
      <p>Update the goals for <strong>{review.year}</strong> across all project sites?</p>
      <p>Seedlings to plant: <strong>{review.annual_planting_target ?? 'Not set'}</strong><br />
        Target alive: <strong>{review.min_survival_target_pct == null ? 'Not set' : `${review.min_survival_target_pct}%`}</strong></p>
    </Modal>, document.body)}</>
  );
}

function PlantingGoalsPanel({ year, onYearChange, settings, loading, error, onSaved }) {
  return (
    <Card title="Planting goals" subtitle="Set annual planting and survival targets across all project sites.">
      <div className="dash-goals-year">
        <label>
          <span>Reporting year</span>
          <select value={year} onChange={(event) => onYearChange(Number(event.target.value))}>
            {Array.from({ length: 201 }, (_, index) => 2000 + index).map((value) => (
              <option key={value} value={value}>{value}</option>
            ))}
          </select>
        </label>
      </div>
      <PlantingGoalsForm key={year} year={year} settings={settings} loading={loading} error={error} onSaved={onSaved} />
    </Card>
  );
}

export default function Dashboard() {
  const [activeTab, setActiveTab] = useState('overview');
  const [visitedTabs, setVisitedTabs] = useState(['overview']);
  const reportView = useRef('');
  const [reportSelection, setReportSelection] = useState(null);
  const [goalsYear, setGoalsYear] = useState(() => Number(dateInManila().slice(0, 4)));
  const [goalsOpen, setGoalsOpen] = useState(false);
  const [goalsVisited, setGoalsVisited] = useState(false);
  const [goalsNotice, setGoalsNotice] = useState('');
  const goalsButton = useRef(null);
  const [filters, setFilters] = useState(makeYtdFilters);
  const [datasets, setDatasets] = useState({ overview: null, operations: null, ecology: null, sites: null });
  const [loading, setLoading] = useState({ overview: true, operations: false, ecology: false, sites: false });
  const [errors, setErrors] = useState({});
  const [dataRevision, setDataRevision] = useState(0);
  const [annualGoals, setAnnualGoals] = useState({});
  const [todayManila] = useState(dateInManila);

  const currentSitesView = activeTab === 'sites';
  const invalidPeriod = !currentSitesView && Boolean(filters.dateFrom && filters.dateTo && filters.dateFrom > filters.dateTo);
  const settingsYear = useMemo(() => {
    const toYear = Number(String(filters.dateTo || '').slice(0, 4));
    if (Number.isInteger(toYear) && toYear >= 2000) return toYear;
    return Number(dateInManila().slice(0, 4));
  }, [filters.dateTo]);
  const settingsYears = useMemo(() => (
    [...new Set([settingsYear, ...(goalsOpen ? [goalsYear] : [])])]
  ), [settingsYear, goalsOpen, goalsYear]);

  const queryString = useMemo(() => {
    const query = new URLSearchParams();
    if (!currentSitesView && filters.dateFrom) query.set('date_from', filters.dateFrom);
    if (!currentSitesView && filters.dateTo) query.set('date_to', filters.dateTo);
    if (filters.siteId) query.set('site_id', filters.siteId);
    query.set('bucket', 'week');
    return query.toString();
  }, [filters, currentSitesView]);


  useEffect(() => {
    if (invalidPeriod) {
      queueMicrotask(() => {
        setDatasets({ overview: null, operations: null, ecology: null, sites: null });
        setLoading({ overview: false, operations: false, ecology: false, sites: false });
      });
      return undefined;
    }
    // Shared API reads retain unchanged reports. Read again on activation so
    // writes made while this retained page was hidden invalidate its view too.
    let active = true;
    const sections = dashboardSectionsForTab(activeTab);
    const view = `${dataRevision}:${queryString}`;
    const changedView = reportView.current !== view;
    reportView.current = view;
    queueMicrotask(() => {
      if (active && changedView) {
        setDatasets({ overview: null, operations: null, ecology: null, sites: null });
        setLoading({ overview: false, operations: false, ecology: false, sites: false });
        setErrors({});
      }
    });
    sections.forEach((key) => {
      queueMicrotask(() => {
        if (active) setLoading((current) => ({ ...current, [key]: true }));
      });
      fetchJson(`${DASHBOARD_ENDPOINTS[key]}?${queryString}`)
        .then((result) => {
          if (active) {
            setDatasets((current) => ({ ...current, [key]: result }));
            setErrors((current) => ({ ...current, [key]: '' }));
          }
        })
        .catch((error) => {
          if (error.name !== 'AbortError') {
            if (active) setErrors((current) => ({ ...current, [key]: error.message }));
          }
        })
        .finally(() => {
          if (active) setLoading((current) => ({ ...current, [key]: false }));
        });
    });
    return () => { active = false; };
  }, [activeTab, queryString, dataRevision, invalidPeriod]);

  useEffect(() => {
    const onDataChanged = () => setDataRevision((value) => value + 1);
    window.addEventListener('mv:data-changed', onDataChanged);
    return () => window.removeEventListener('mv:data-changed', onDataChanged);
  }, []);

  useEffect(() => {
    let active = true;
    // The goals panel has its own reporting year. Opening it must not change
    // the year or reload the report currently displayed in the dashboard.
    settingsYears.forEach((year) => {
      queueMicrotask(() => {
        if (active) setAnnualGoals((current) => ({ ...current, [year]: { ...current[year], loading: true, error: '' } }));
      });
      fetchJson(`/api/dashboard/settings?year=${year}`)
        .then((result) => {
          const record = { data: result, error: '', loading: false };
          if (active) setAnnualGoals((current) => ({ ...current, [year]: record }));
        })
        .catch((error) => {
          if (error.name === 'AbortError') return;
          const record = { error: error.message || 'Could not load planting goals.', loading: false };
          if (active) setAnnualGoals((current) => ({ ...current, [year]: { ...current[year], ...record } }));
        });
    });
    return () => { active = false; };
  }, [dataRevision, settingsYears]);



  const filterOptions = useMemo(() => (
    datasets.overview?.filter_options
    || datasets.operations?.filter_options
    || datasets.ecology?.filter_options
    || datasets.sites?.filter_options
    || {}
  ), [datasets]);
  const activeSettings = annualGoals[settingsYear]?.data || null;
  const goalsRecord = annualGoals[goalsYear];

  const sites = arrayOf(filterOptions.sites);
  const defaultFilters = makeYtdFilters();
  const activeFilterCount = [
    filters.dateFrom !== defaultFilters.dateFrom || filters.dateTo !== defaultFilters.dateTo,
    Boolean(filters.siteId),
  ].filter(Boolean).length;
  const asOf = datasets[activeTab]?.as_of || datasets.overview?.as_of;

  const changeFilter = (key, value) => {
    setFilters((current) => ({ ...current, [key]: value }));
  };

  return (
    <main className="dash">
      <header className="dash-header">
        <div>
          <div className="dash-eyebrow">Mangrove restoration dashboard</div>
          <h1>Mangrove restoration overview</h1>
          <p>See planting progress, seedling survival, project sites, and registered organizations.</p>
        </div>
        <div className="dash-header-meta">
          <span>{asOf ? `Updated ${formatDate(asOf, true)}` : `Reporting in ${TIMEZONE}`}</span>
          <button ref={goalsButton} type="button" className="dash-goals-button" aria-expanded={goalsOpen}
            aria-controls="dashboard-planting-goals" onClick={() => {
              setGoalsNotice('');
              setGoalsVisited(true);
              setGoalsOpen((current) => !current);
            }}>
            Planting Goals
          </button>
        </div>
      </header>

      {goalsVisited && <Activity mode={goalsOpen ? 'visible' : 'hidden'}>
        <section id="dashboard-planting-goals" className="dash-goals-panel" aria-label="Planting goals">
          <PlantingGoalsPanel year={goalsYear} onYearChange={setGoalsYear} settings={goalsRecord?.data || null}
            loading={goalsRecord?.loading ?? true} error={goalsRecord?.error || ''} onSaved={(saved) => {
              setAnnualGoals((current) => ({ ...current, [goalsYear]: { data: saved, error: '', loading: false } }));
              setGoalsNotice(`Planting and survival goals for ${goalsYear} were saved.`);
              setGoalsOpen(false);
              goalsButton.current?.focus({ preventScroll: true });
            }} />
        </section>
      </Activity>}

      {goalsNotice && <div className="dash-success dash-goals-notice" role="status">{goalsNotice}</div>}

      <AppointmentNotice />

      <NextActions actions={[
        { label: 'Review analysis', to: '/processing?action=review', description: 'Review the current result or open image analysis history.' },
        { label: 'Assign available points', to: '/planters?section=assign', description: 'Choose an organization and reserve planting locations.' },
        { label: 'Record planting', to: '/field', reloadDocument: true, description: 'Open the planter page to record assigned locations as planted.' },
        { label: 'Inspect plants due', to: '/monitoring?filter=due', description: 'Find organizations due for a monitoring visit.' },
      ]} />

      <section className="dash-filter-shell" aria-labelledby="dashboard-filters-title">
        <div className="dash-filter-head">
          <div>
            <h2 id="dashboard-filters-title">{currentSitesView ? 'Choose a project site' : 'Choose what to view'}</h2>
            <p>{currentSitesView ? 'Current locations and warnings across all planting dates.' : 'Select a date range and, if needed, one project site.'}</p>
          </div>
          <div className="dash-filter-actions">
            <span className="dash-filter-count">{currentSitesView ? 'Current snapshot' : activeFilterCount ? 'Custom view' : 'This year so far'}</span>
            {currentSitesView ? <button type="button" onClick={() => changeFilter('siteId', '')}>Show all sites</button> : <button type="button" onClick={() => setFilters(makeYtdFilters())}>Show this year</button>}
            {!currentSitesView && <button type="button" className="dash-report-button" aria-haspopup="dialog" disabled={invalidPeriod} onClick={() => setReportSelection({
              ...filters, dateFrom: filters.dateFrom || defaultFilters.dateFrom, dateTo: filters.dateTo || defaultFilters.dateTo,
              type: activeTab === 'operations' ? 'organizations' : activeTab === 'ecology' ? 'monitoring' : 'planting',
            })}>Download Report</button>}
          </div>
        </div>
        <div className={`dash-filters${currentSitesView ? ' dash-site-filter' : ''}`}>
          {!currentSitesView && <>
          <label>
            <span>From</span>
            <input type="date" value={filters.dateFrom} max={filters.dateTo && filters.dateTo < todayManila ? filters.dateTo : todayManila} onChange={(event) => changeFilter('dateFrom', event.target.value)} />
          </label>
          <label>
            <span>To</span>
            <input type="date" value={filters.dateTo} min={filters.dateFrom || undefined} max={todayManila} aria-invalid={invalidPeriod} aria-describedby={invalidPeriod ? 'dashboard-period-error' : undefined} onChange={(event) => changeFilter('dateTo', event.target.value)} />
            {invalidPeriod && <span id="dashboard-period-error" className="form-field-error" role="alert">Choose an end date on or after the start date.</span>}
          </label>
          </>}
          <label>
            <span>Project site</span>
            <select value={filters.siteId} onChange={(event) => changeFilter('siteId', event.target.value)}>
              <option value="">All project sites</option>
              {sites.map((option) => <option key={optionId(option)} value={optionId(option)}>{optionLabel(option)}</option>)}
            </select>
          </label>
        </div>
      </section>

      <nav className="dash-tabs" role="tablist" aria-label="Dashboard sections">
        {TABS.map((tab) => (
          <button
            key={tab.id}
            id={`dash-tab-${tab.id}`}
            type="button"
            role="tab"
            aria-selected={activeTab === tab.id}
            aria-controls={`dash-panel-${tab.id}`}
            className={activeTab === tab.id ? 'is-active' : ''}
            onClick={() => {
              setVisitedTabs((current) => current.includes(tab.id) ? current : [...current, tab.id]);
              setActiveTab(tab.id);
            }}
          >
            {tab.label}
          </button>
        ))}
      </nav>

      {visitedTabs.map((tab) => <Activity key={tab} mode={activeTab === tab ? 'visible' : 'hidden'}>
      <section id={`dash-panel-${tab}`} role="tabpanel" aria-labelledby={`dash-tab-${tab}`} className="dash-content">
        {tab === 'overview' ? (
          <DatasetBoundary loading={loading.overview} error={errors.overview} data={datasets.overview}>
            <OverviewTab data={datasets.overview || {}} />
          </DatasetBoundary>
        ) : null}
        {tab === 'operations' ? (
          <DatasetBoundary loading={loading.operations} error={errors.operations} data={datasets.operations}>
            <OperationsTab data={datasets.operations || {}} />
          </DatasetBoundary>
        ) : null}
        {tab === 'ecology' ? (
          <DatasetBoundary loading={loading.ecology} error={errors.ecology} data={datasets.ecology}>
            <EcologyTab data={datasets.ecology || {}} settings={activeSettings} />
          </DatasetBoundary>
        ) : null}
        {tab === 'sites' ? (
          <DatasetBoundary loading={loading.sites} error={errors.sites} data={datasets.sites}>
            <SitesTab data={datasets.sites || {}} onViewHistory={() => {
              setVisitedTabs((current) => current.includes('ecology') ? current : [...current, 'ecology']);
              setActiveTab('ecology');
            }} />
          </DatasetBoundary>
        ) : null}
      </section>
      </Activity>)}
      {reportSelection && <RestorationReportDialog initialSelection={reportSelection} initialSites={sites} onClose={() => setReportSelection(null)} />}
    </main>
  );
}
