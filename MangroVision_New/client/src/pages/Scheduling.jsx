import { useEffect, useMemo, useRef, useState } from 'react';
import {
  CartesianGrid,
  Line,
  LineChart,
  ReferenceLine,
  ReferenceArea,
  ResponsiveContainer,
  Tooltip,
  XAxis,
  YAxis,
} from 'recharts';
import Modal from '../components/Modal';
import useTideForecasts from '../utils/useTideForecasts';
import {
  PLANTING_STATES, assessGraphWindow, assessGraphTime, chartLevelSeries, finiteNumber, forecastIssue,
  forecastPlantingGuide, guideLevelState, coloredTideSeries,
} from '../utils/plantingTides';
import { useAuthStore } from '../stores/authStore';
import './Scheduling.css';

const API = import.meta.env.VITE_API_BASE || '';
const TIMEZONE = 'Asia/Manila';

function arrayOf(value) {
  return Array.isArray(value) ? value : [];
}

function numberOrNull(value) {
  return finiteNumber(value);
}

function firstNumber(...values) {
  for (const value of values) {
    const parsed = numberOrNull(value);
    if (parsed !== null) return parsed;
  }
  return null;
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

function formatDate(value, includeTime = false) {
  if (!value) return 'â€”';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat(undefined, includeTime
    ? { dateStyle: 'medium', timeStyle: 'short', timeZone: TIMEZONE }
    : { dateStyle: 'medium', timeZone: TIMEZONE }).format(date);
}

function formatShortDate(value) {
  if (!value) return 'â€”';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return new Intl.DateTimeFormat(undefined, {
    month: 'short', day: 'numeric', timeZone: TIMEZONE,
  }).format(date);
}

function formatClock(value) {
  if (!value) return 'â€”';
  const match = String(value).match(/^(\d{1,2}):(\d{2})/);
  if (!match) return String(value);
  const date = new Date(Date.UTC(2000, 0, 1, Number(match[1]), Number(match[2])));
  return new Intl.DateTimeFormat(undefined, {
    hour: 'numeric', minute: '2-digit', timeZone: 'UTC',
  }).format(date);
}

function formatCount(value) {
  const parsed = numberOrNull(value);
  return parsed === null ? 'â€”' : new Intl.NumberFormat().format(parsed);
}

function formatHeight(value) {
  const parsed = numberOrNull(value);
  return parsed === null ? 'Height unavailable' : `${parsed.toFixed(2)} m`;
}

async function fetchJson(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  const body = await response.json().catch(() => ({}));
  if (!response.ok) {
    const detail = body?.detail ?? body?.message;
    const messages = (Array.isArray(detail) ? detail : [detail])
      .filter((item) => item !== null && item !== undefined && item !== '')
      .map((item) => {
        if (typeof item === 'string') return item;
        if (typeof item !== 'object') return String(item);
        const field = arrayOf(item.loc)
          .filter((part) => part !== 'body')
          .join('.');
        const message = item.msg ?? item.message ?? item.detail;
        if (message) return field ? `${field}: ${message}` : String(message);
        try {
          return JSON.stringify(item);
        } catch {
          return 'The server rejected one of the submitted values.';
        }
      });
    throw new Error(messages.join('; ') || `Request failed (${response.status})`);
  }
  return body && typeof body === 'object' ? body : {};
}

function normalizeTideEvents(payload) {
  const source = arrayOf(payload?.extremes).length
    ? payload.extremes
    : arrayOf(payload?.predictions).length
      ? payload.predictions
      : arrayOf(payload?.data);
  return source.filter((row) => row && typeof row === 'object').map((row, index) => {
    const rawTime = row.occurred_at ?? row.datetime ?? row.date ?? row.time ?? row.timestamp;
    const occurredAt = typeof rawTime === 'number' && rawTime < 1_000_000_000_000
      ? rawTime * 1000
      : rawTime;
    return {
      ...row,
      id: row.id ?? index,
      occurred_at: occurredAt,
      height_m: firstNumber(row.height_m, row.height, row.value),
      tide_type: row.tide_type ?? row.type ?? row.kind ?? 'Prediction',
    };
  }).filter((row) => Number.isFinite(new Date(row.occurred_at).getTime()) && row.height_m !== null)
    .sort((a, b) => new Date(a.occurred_at).getTime() - new Date(b.occurred_at).getTime());
}

function tideKind(row) {
  const value = String(row?.tide_type ?? row?.type ?? row?.kind ?? '').toLowerCase();
  if (value.includes('high')) return 'high';
  if (value.includes('low')) return 'low';
  return 'prediction';
}

function tidePlantingState(tide, guide) {
  const kind = tideKind(tide);
  if (kind === 'high') return 'unsafe';
  if (kind === 'low') return 'safe';
  return guideLevelState(tide?.height_m, guide);
}

function selectionKey(selection) {
  return `${selection?.date || ''}|${selection?.startTime || ''}|${selection?.endTime || ''}`;
}

function requiresTimeCautionBeforeSave(selection, assessment, acceptedKey) {
  return assessment?.status === 'unsafe' && acceptedKey !== selectionKey(selection);
}

function assessPlantingTime({ payload, startAt, endAt, now = Date.now() }) {
  const start = Date.parse(startAt);
  const end = endAt ? Date.parse(endAt) : null;
  if (!Number.isFinite(start) || (endAt && (!Number.isFinite(end) || end <= start))) {
    return { status: 'unknown', label: 'Set a valid activity time', reason: 'The end time must be later than the start time.' };
  }
  const highTide = normalizeTideEvents(payload).find((tide) => {
    if (tideKind(tide) !== 'high') return false;
    const occurredAt = new Date(tide.occurred_at).getTime();
    return end === null ? Math.abs(occurredAt - start) < 60_000 : occurredAt >= start && occurredAt <= end;
  });
  if (highTide) return {
    status: 'unsafe',
    label: 'Not safe to plant',
    reason: `High tide is expected at ${tideClock(highTide)} during this activity. Choose a low-tide time if possible.`,
    high_tide_at: highTide.occurred_at,
  };
  return end === null
    ? assessGraphTime({ payload, now, at: startAt, allowOutdated: true })
    : assessGraphWindow({ payload, now, startAt, endAt, allowOutdated: true });
}

function normalizeOrganization(source) {
  const properties = source?.properties || {};
  return {
    ...source,
    id: source?.id ?? source?.organization_id ?? source?.value ?? properties.id,
    name: source?.name ?? source?.organization_name ?? source?.label ?? source?.title ?? properties.name ?? '',
  };
}

function normalizeProjectSite(source) {
  const properties = source?.properties || {};
  const organization = source?.organization || properties.organization;
  return {
    ...source,
    id: source?.id ?? source?.project_site_id ?? properties.id,
    name: source?.name ?? source?.project_site_name ?? properties.name ?? `Planting area ${source?.id ?? properties.id ?? ''}`,
    organization_id: source?.organization_id ?? organization?.id ?? properties.organization_id ?? null,
    organization_name: source?.organization_name ?? (typeof organization === 'string' ? organization : organization?.name) ?? properties.organization_name ?? '',
  };
}

function scheduleDate(schedule) {
  if (schedule?.scheduled_date ?? schedule?.date) return schedule.scheduled_date ?? schedule.date;
  const parsed = schedule?.start_at ? new Date(schedule.start_at) : null;
  return parsed && !Number.isNaN(parsed.getTime()) ? dateInManila(parsed) : '';
}

function scheduleStartTime(schedule) {
  if (schedule?.start_time ?? schedule?.time) return schedule.start_time ?? schedule.time;
  const parsed = schedule?.start_at ? new Date(schedule.start_at) : null;
  return parsed && !Number.isNaN(parsed.getTime())
    ? new Intl.DateTimeFormat('en-GB', {
      hour: '2-digit', minute: '2-digit', hour12: false, timeZone: TIMEZONE,
    }).format(parsed)
    : '';
}

function normalizeSchedule(schedule) {
  const organization = schedule?.organization;
  return {
    ...schedule,
    id: schedule?.id ?? schedule?.schedule_id,
    organization_id: schedule?.organization_id ?? (typeof organization === 'object' ? organization?.id : null),
    organization_name: schedule?.organization_name ?? (typeof organization === 'string' ? organization : organization?.name) ?? '',
    title: schedule?.title ?? schedule?.event_title ?? 'Tree-planting activity',
    contact: schedule?.contact ?? schedule?.contact_person ?? '',
    project_site_id: schedule?.project_site_id ?? schedule?.site_id ?? null,
    project_site_name: schedule?.project_site_name ?? schedule?.site_name ?? schedule?.project?.name ?? '',
    scheduled_date: scheduleDate(schedule),
    start_time: scheduleStartTime(schedule),
    end_time: schedule?.end_time ?? '',
    expected_participants: schedule?.expected_participants ?? schedule?.expected_planters ?? schedule?.participants ?? null,
    seedlings: schedule?.seedlings ?? schedule?.expected_seedlings ?? schedule?.seedling_count ?? null,
    inspection_interval_days: schedule?.inspection_interval_days ?? schedule?.interval_days ?? null,
    status: schedule?.status ?? 'requested',
    notes: schedule?.notes ?? '',
  };
}

function makeScheduleForm(schedule = null) {
  const startAt = schedule?.start_at ? new Date(schedule.start_at) : null;
  const endAt = schedule?.end_at ? new Date(schedule.end_at) : null;
  return {
    organizationId: String(schedule?.organization_id ?? ''),
    organizationName: schedule?.organization_name ?? '',
    title: schedule?.title ?? '',
    contact: schedule?.contact ?? '',
    date: schedule?.scheduled_date ?? (startAt && !Number.isNaN(startAt.getTime()) ? dateInManila(startAt) : ''),
    startTime: schedule?.start_time ?? (startAt && !Number.isNaN(startAt.getTime())
      ? new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: TIMEZONE }).format(startAt)
      : ''),
    endTime: schedule?.end_time ?? (endAt && !Number.isNaN(endAt.getTime())
      ? new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: TIMEZONE }).format(endAt)
      : ''),
    expectedParticipants: schedule?.expected_participants ?? '',
    seedlings: schedule?.seedlings ?? '',
    inspectionIntervalDays: 14,
    status: schedule?.status ?? 'requested',
    notes: schedule?.notes ?? '',
  };
}

function scheduleStatusLabel(value) {
  if (String(value).toLowerCase() === 'requested') return 'Pending';
  const clean = String(value || 'requested').replaceAll('_', ' ');
  return clean.charAt(0).toUpperCase() + clean.slice(1);
}

function scheduleTimeLabel(schedule) {
  const start = scheduleStartTime(schedule);
  const end = schedule?.end_time;
  if (!start) return 'Time not set';
  return end ? `${formatClock(start)} - ${formatClock(end)}` : formatClock(start);
}

function calendarCells(monthKey) {
  const [year, month] = String(monthKey).split('-').map(Number);
  if (!year || !month) return [];
  const first = new Date(Date.UTC(year, month - 1, 1));
  const firstSunday = new Date(first);
  firstSunday.setUTCDate(first.getUTCDate() - first.getUTCDay());
  return Array.from({ length: 42 }, (_, index) => {
    const date = new Date(firstSunday);
    date.setUTCDate(firstSunday.getUTCDate() + index);
    return {
      date: date.toISOString().slice(0, 10),
      day: date.getUTCDate(),
      inMonth: date.getUTCMonth() === month - 1,
    };
  });
}

function shiftMonth(monthKey, amount) {
  const [year, month] = String(monthKey).split('-').map(Number);
  const shifted = new Date(Date.UTC(year, month - 1 + amount, 1));
  return shifted.toISOString().slice(0, 7);
}

function monthLabel(monthKey) {
  const [year, month] = String(monthKey).split('-').map(Number);
  return new Intl.DateTimeFormat(undefined, {
    month: 'long', year: 'numeric', timeZone: 'UTC',
  }).format(new Date(Date.UTC(year, month - 1, 1)));
}

function sameOrganization(site, schedule) {
  const scheduleId = schedule?.organization_id;
  const siteId = site?.organization_id;
  if (scheduleId !== null && scheduleId !== undefined && siteId !== null && siteId !== undefined) {
    return String(scheduleId) === String(siteId);
  }
  const scheduleName = String(schedule?.organization_name || '').trim().toLocaleLowerCase();
  const siteName = String(site?.organization_name || '').trim().toLocaleLowerCase();
  return Boolean(scheduleName && siteName && scheduleName === siteName);
}

function Message({ type = 'error', children }) {
  return <div className={`schedule-message is-${type}`} role={type === 'error' ? 'alert' : 'status'}>{children}</div>;
}

function Loading({ label }) {
  return <div className="schedule-loading" role="status"><span aria-hidden="true" />{label}</div>;
}

function PlantingBadge({ assessment }) {
  const state = PLANTING_STATES[assessment?.status] || PLANTING_STATES.unknown;
  if (!assessment || assessment.status === 'unknown') return <span className="schedule-tide-pending" title={assessment?.reason}>
    {assessment?.label || state.label}
  </span>;
  return <span className="schedule-planting-badge" style={{ color: state.color, borderColor: state.color }} title={assessment?.reason}>
    <span aria-hidden="true">{state.symbol}</span> {state.label}
  </span>;
}

function PlantingLegend() {
  return <div className="schedule-planting-legend" aria-label="Planting color guide">
    {['safe', 'unsafe'].map((status) => <PlantingBadge key={status} assessment={{ status }} />)}
    <small>Colors help you plan when to plant. Each activity also shows whether it is pending, confirmed or completed.</small>
  </div>;
}

function ScheduleCalendarEvent({ schedule, assessment, onOpen }) {
  const state = PLANTING_STATES[assessment.status] || PLANTING_STATES.unknown;
  return <button type="button" className={`schedule-calendar-event planting-${assessment.status}`} onClick={() => onOpen(schedule)}
    aria-haspopup="dialog" aria-label={`${schedule.title}, ${scheduleTimeLabel(schedule)}. ${state.label}. View activity details.`}>
    <span className="schedule-entry-top"><time>{scheduleStartTime(schedule) ? formatClock(scheduleStartTime(schedule)) : 'Time not set'}</time></span>
    <strong className="schedule-entry-title">{schedule.title}</strong>
  </button>;
}

function tideClock(tide) {
  return new Intl.DateTimeFormat(undefined, { hour: 'numeric', minute: '2-digit', timeZone: TIMEZONE }).format(new Date(tide.occurred_at));
}

function TideTime({ tide, guide, onOpen }) {
  const status = tidePlantingState(tide, guide);
  const state = PLANTING_STATES[status];
  if (onOpen) return <button type="button" className={`schedule-calendar-tide is-compact planting-${status}`} onClick={() => onOpen(tide)}
    aria-haspopup="dialog" aria-label={`${scheduleStatusLabel(tideKind(tide))} tide at ${tideClock(tide)}. ${state.label}. View tide details.`}>
    <time>{tideClock(tide)}</time>
    <span>{scheduleStatusLabel(tideKind(tide))} tide</span>
  </button>;
  return <div className={`schedule-calendar-tide planting-${status}`} title={`${formatDate(tide.occurred_at, true)}, ${formatHeight(tide.height_m)}. ${state.label} using the graph's estimated planting limit.`}>
    <span>{scheduleStatusLabel(tideKind(tide))} tide · {formatDate(tide.occurred_at, true).split(',').pop()?.trim()}</span>
    <span>{state.label}</span>
  </div>;
}

function assessSelectedTime(form, payload, now = Date.now()) {
  if (!form.date || (!form.startTime && !form.endTime)) return { status: 'unknown' };
  const startTime = form.startTime || form.endTime;
  return assessPlantingTime({ payload, now,
    startAt: `${form.date}T${startTime}+08:00`,
    endAt: form.startTime && form.endTime ? `${form.date}T${form.endTime}+08:00` : null,
  });
}

function HighTideCaution({ selection, onContinue, onBack }) {
  if (!selection) return null;
  return <Modal open title="High tide caution" variant="warning"
    confirmLabel="Keep this time" cancelLabel="Change time" onConfirm={onContinue} onCancel={onBack}>
    <div className="schedule-tide-caution">
      <p><strong>Water is expected to be too high for planting at your chosen time.</strong></p>
      <p>{formatDate(`${selection.date}T12:00:00+08:00`)} · {[selection.startTime, selection.endTime].filter(Boolean).map(formatClock).join(' - ')} (Philippine time)</p>
      <p>{selection.assessment?.reason || 'The graph marks this time as not safe to plant.'}</p>
      <p>You can choose another time or keep it and continue filling out the schedule.</p>
    </div>
  </Modal>;
}

function CalendarEntryDetails({ entry, assessment, guide, onClose, onEdit }) {
  if (!entry) return null;
  const schedule = entry.kind === 'schedule' ? entry.schedule : null;
  const tide = entry.kind === 'tide' ? entry.tide : null;
  const advice = schedule ? assessment : { status: tidePlantingState(tide, guide) };
  return <Modal open title={schedule ? schedule.title : `${scheduleStatusLabel(tideKind(tide))} tide`}
    className="modal-card-wide schedule-details-modal" variant="info"
    confirmLabel={schedule ? 'Edit activity' : 'Close'} cancelLabel={schedule ? 'Close' : null}
    onCancel={onClose} onConfirm={schedule ? () => { onClose(); onEdit(schedule); } : onClose}>
    <div className="schedule-entry-details">
      <div className="schedule-detail-when">
        <strong>{schedule ? scheduleTimeLabel(schedule) : tideClock(tide)}</strong>
        <span>{schedule ? formatDate(`${scheduleDate(schedule)}T12:00:00+08:00`) : formatDate(tide.occurred_at)} · Philippine time</span>
      </div>
      <div className="schedule-detail-advice">
        <PlantingBadge assessment={advice} />
        <p>{schedule ? assessment.reason : 'Uses the same estimated planting limit as the graph. Check your planting area before going.'}</p>
      </div>
      {schedule ? <dl>
        <div><dt>Status</dt><dd>{scheduleStatusLabel(schedule.status)}</dd></div>
        <div><dt>Organization</dt><dd>{schedule.organization_name || 'Not set'}</dd></div>
        <div><dt>Planting area</dt><dd>{schedule.project_site_name || 'No planting area chosen'}</dd></div>
        <div><dt>Participants</dt><dd>{formatCount(schedule.expected_participants)}</dd></div>
        <div><dt>Seedlings</dt><dd>{formatCount(schedule.seedlings)}</dd></div>
        <div><dt>Plant checks</dt><dd>{schedule.inspection_interval_days ? `Every ${formatCount(schedule.inspection_interval_days)} days` : 'Not set'}</dd></div>
        {schedule.contact ? <div><dt>Contact</dt><dd>{schedule.contact}</dd></div> : null}
        {schedule.notes ? <div><dt>Notes</dt><dd>{schedule.notes}</dd></div> : null}
      </dl> : <dl>
        <div><dt>Water level</dt><dd>{formatHeight(tide.height_m)}</dd></div>
        <div><dt>Estimated planting limit</dt><dd>{formatHeight(guide.threshold)}</dd></div>
      </dl>}
    </div>
  </Modal>;
}

function TideForecast({ payload, tides, loading, error, site, onRetry }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(timer);
  }, []);
  const samples = chartLevelSeries(payload);
  const rows = samples.length ? samples : tides.map((row) => ({ ...row, occurred_at: new Date(row.occurred_at).getTime() }));
  const issue = forecastIssue(payload, now);
  // Keep the visible curve colored from the displayed readings even when they
  // need refreshing. Scheduling assessments still require a current forecast.
  const guide = forecastPlantingGuide({ site, payload, now, allowOutdated: true });
  const { threshold, preview, percentage } = guide;
  const usable = guide.status !== 'unknown';
  const chartData = coloredTideSeries(rows, guide);
  const values = rows.map((row) => row.height_m).filter((value) => value !== null);
  if (threshold !== null) values.push(threshold);
  const lowest = Math.min(0, ...values);
  const highest = Math.max(0, ...values);
  const padding = Math.max(0.15, (highest - lowest) * 0.15);
  const domain = [Math.floor((lowest - padding) * 10) / 10, Math.ceil((highest + padding) * 10) / 10];
  const datumLabel = payload?.datum === 'MSL' ? 'meters from average sea level' : 'meters; starting level not confirmed';
  const referenceLabel = preview ? 'Estimated planting limit' : 'Measured ground height';
  const statusLabel = (height) => {
    const state = guideLevelState(height, guide);
    return preview && state !== 'unknown'
      ? state === 'safe' ? 'Safe to plant (estimated)' : 'Not safe to plant (estimated)'
      : PLANTING_STATES[state].label;
  };

  return (
    <section className="schedule-panel schedule-tide-panel" aria-labelledby="standalone-tide-title">
      <div className="schedule-panel-head">
        <div>
          <span className="schedule-step">Planting conditions</span>
          <h2 id="standalone-tide-title">When to plant</h2>
        </div>
        <div className="schedule-tide-tools">
          <button type="button" onClick={onRetry} disabled={loading}>{loading ? 'Loading tides...' : 'Refresh tides'}</button>
        </div>
      </div>
      {issue && !loading ? <Message type="warning">{issue}{payload?.stale && payload.message ? ` ${payload.message}` : ''}</Message> : null}
      {loading ? <Loading label="Loading the tide forecast..." /> : null}
      {error ? <Message>{error}</Message> : null}
      {!loading && !error && !chartData.length ? (
        <div className="schedule-empty"><strong>Tide forecast unavailable</strong><p>{payload?.message || 'No water levels are available yet. Please refresh the tides.'}</p></div>
      ) : null}
      <div className="schedule-graph-legend" aria-label="Planting color guide">
        <span className="is-safe"><i aria-hidden="true" />{preview ? 'Safe to plant (estimated)' : 'Safe to plant'}</span>
        <span className="is-unsafe"><i aria-hidden="true" />{preview ? 'Not safe to plant (estimated)' : 'Not safe to plant'}</span>
        {!usable ? <span className="is-unknown"><i aria-hidden="true" />Tide forecast unavailable</span> : null}
        <span className="schedule-reference-key">{referenceLabel}{threshold !== null ? `: ${threshold.toFixed(2)} m` : ': unknown'}</span>
      </div>
      {chartData.length ? (
        <>
          <div className="schedule-tide-chart" role="img" aria-label={`Expected water level (${datumLabel}); Date and time (Philippine time). ${referenceLabel}: ${threshold === null ? 'unknown' : `${threshold.toFixed(2)} m`}. Green shows estimated planting times. Red means avoid planting. Ground should be underwater no more than 30% of the time. Check the planting area before going.`}>
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ top: 24, right: 18, bottom: 30, left: 14 }}>
                <CartesianGrid stroke="#dbe4e7" strokeDasharray="3 5" vertical={false} />
                <XAxis dataKey="occurred_at" type="number" domain={['dataMin', 'dataMax']} tickFormatter={formatShortDate} minTickGap={40}
                  label={{ value: 'Date and time (Philippine time)', position: 'bottom', offset: 12, fill: '#334155', fontSize: 12 }} />
                <YAxis domain={domain} tickFormatter={(value) => `${Number(value.toFixed(2))} m`} width={64}
                  label={{ value: 'Water level (meters)', angle: -90, position: 'insideLeft', dx: -10, style: { textAnchor: 'middle', fontSize: 11 }, fill: '#334155' }} />
                {usable && threshold !== null ? <>
                  <ReferenceArea y1={domain[0]} y2={threshold} fill={guide.status === 'safe' ? '#16a34a' : '#dc2626'} fillOpacity={0.07} />
                  <ReferenceArea y1={threshold} y2={domain[1]} fill="#dc2626" fillOpacity={0.07} />
                  <ReferenceLine y={threshold} stroke="#475569" strokeWidth={1.5} strokeDasharray="6 4" />
                </> : null}
                <Tooltip content={({ active, payload: points, label }) => {
                  const row = points?.find((point) => point.value !== null)?.payload;
                  return active && row ? <div className="schedule-tide-tooltip">
                    <strong>{formatDate(label, true)}</strong>
                    <span>Water level: {formatHeight(row.height_m)}</span>
                    <span style={{ color: PLANTING_STATES[guideLevelState(row.height_m, guide)].color }}>{statusLabel(row.height_m)}</span>
                    {threshold !== null ? <small>{referenceLabel}: {formatHeight(threshold)}</small> : null}
                  </div> : null;
                }} />
                {Object.entries(PLANTING_STATES).map(([state, style]) => (
                  <Line key={state} type="linear" dataKey={state} name={style.label} stroke={samples.length ? style.color : 'none'} strokeWidth={3}
                    connectNulls={false} isAnimationActive={false} dot={samples.length ? false : { r: 3, fill: style.color }} activeDot={{ r: 5, strokeWidth: 2 }} />
                ))}
              </LineChart>
            </ResponsiveContainer>
          </div>
          <div className={`schedule-inundation-summary is-${usable && !preview ? guide.status : 'unknown'}`} role="status">
            <strong>{preview ? 'Estimated planting guide' : guide.label}</strong>
            <span>{percentage !== null
              ? `${percentage.toFixed(1)}% underwater / ${(100 - percentage).toFixed(1)}% above water during the dates shown`
              : 'We cannot estimate time underwater yet'}</span>
            <p>{preview
              ? 'The dashed line is an estimated planting limit. Green suggests a good time to plant; red means avoid planting. This guide uses an estimated ground height. Check the actual ground, water and weather at your planting area before going.'
              : 'Green means water is below the ground or just reaches it, and the area is expected to be underwater no more than 30% of the time. The ground must also be at or above average sea level. Red means avoid planting. Check local conditions before going.'}</p>
          </div>
          <p className="schedule-threshold-note">Time the ground is underwater:</p>
          <div className="schedule-inundation-scale" aria-label="How much time the ground is underwater">
            <span className="is-safe"><strong>0–30%</strong> Good for planting</span>
            <span className="is-marginal"><strong>&gt;30–50%</strong> Check the area first</span>
            <span className="is-unsafe"><strong>&gt;50%</strong> Avoid planting</span>
            <span className="is-unsafe"><strong>100%</strong> Underwater the whole time</span>
          </div>
          <details className="schedule-table-fallback">
            <summary>See water levels in a table</summary>
            <div className="schedule-table-wrap">
              <table>
                <thead><tr><th scope="col">Date and time (Philippine time)</th><th scope="col">Water level ({datumLabel})</th><th scope="col">Planting advice{preview ? ' (estimated)' : ''}</th></tr></thead>
                <tbody>{rows.map((tide) => (
                  <tr key={tide.occurred_at}>
                    <td>{formatDate(tide.occurred_at, true)}</td><td>{formatHeight(tide.height_m)}</td>
                    <td style={{ color: PLANTING_STATES[guideLevelState(tide.height_m, guide)].color }}>{statusLabel(tide.height_m)}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          </details>
        </>
      ) : null}
      <footer className="schedule-tide-source">
        <span>{payload?.datum === 'MSL' ? 'Water heights are measured from average sea level.' : 'The starting level for water measurements has not been confirmed.'}</span>
        <span>Choose ground that is underwater no more than 30% of the time. These estimates cover only the dates shown; check the area over time too.</span>
        <a href="https://cms.zsl.org/sites/default/files/2023-02/1%20Manual%20-%20Community-based%20Mangrove%20Rehabilitation.pdf" target="_blank" rel="noreferrer">Why 30%? Read the planting guide</a>
        {payload?.attribution ? <span>{payload.attribution_url
          ? <a href={payload.attribution_url} target="_blank" rel="noreferrer">{payload.attribution}</a> : payload.attribution}</span> : null}
      </footer>
    </section>
  );
}

export default function Scheduling() {
  const token = useAuthStore((state) => state.token);
  const [apiData, setApiData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [reloadKey, setReloadKey] = useState(0);
  const loadedSchedules = useRef(null);
  const [tideRetryKey, setTideRetryKey] = useState(0);
  const [now, setNow] = useState(() => Date.now());
  const [view, setView] = useState('calendar');
  const [calendarEntry, setCalendarEntry] = useState(null);
  const [month, setMonth] = useState(() => dateInManila().slice(0, 7));
  const [formOpen, setFormOpen] = useState(false);
  const [editingSchedule, setEditingSchedule] = useState(null);
  const [form, setForm] = useState(makeScheduleForm);
  const [formError, setFormError] = useState('');
  const [saving, setSaving] = useState(false);
  const [timeCaution, setTimeCaution] = useState(null);
  const lastWarnedTime = useRef(null);
  const acceptedUnsafeTime = useRef(null);
  const saveInFlight = useRef(false);
  const [notice, setNotice] = useState('');
  const [deletingId, setDeletingId] = useState(null);
  const [assigningSchedule, setAssigningSchedule] = useState(null);
  const [assignmentSiteId, setAssignmentSiteId] = useState('');
  const [assignmentError, setAssignmentError] = useState('');
  const [assignmentSaving, setAssignmentSaving] = useState(false);

  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(timer);
  }, []);

  useEffect(() => {
    const key = `${token}:${reloadKey}`;
    if (loadedSchedules.current === key) return undefined;
    const controller = new AbortController();
    if (!token) {
      queueMicrotask(() => {
        setApiData(null);
        setLoading(false);
        setError('Please sign in again to view schedules.');
      });
      return () => controller.abort();
    }
    queueMicrotask(() => {
      if (!controller.signal.aborted) {
        setLoading(true);
        setError('');
      }
    });
    fetchJson('/api/planting-schedules', { signal: controller.signal })
      .then((payload) => {
        if (!controller.signal.aborted) {
          loadedSchedules.current = key;
          setApiData(payload);
        }
      })
      .catch((loadError) => {
        if (loadError.name !== 'AbortError') setError(loadError.message || 'Could not load planting schedules.');
      })
      .finally(() => {
        if (!controller.signal.aborted) { loadedSchedules.current = key; setLoading(false); }
      });
    return () => controller.abort();
  }, [reloadKey, token]);

  useEffect(() => {
    const refresh = () => setReloadKey((value) => value + 1);
    window.addEventListener('mv:data-changed', refresh);
    return () => window.removeEventListener('mv:data-changed', refresh);
  }, []);

  const refreshSchedules = () => {
    window.dispatchEvent(new Event('mv:invalidate-reads'));
    setReloadKey((value) => value + 1);
    setTideRetryKey((value) => value + 1);
  };

  const schedules = useMemo(() => {
    const source = Array.isArray(apiData) ? apiData : arrayOf(apiData?.schedules).length ? apiData.schedules : arrayOf(apiData?.items);
    return source.map(normalizeSchedule).sort((a, b) => (
      `${a.scheduled_date}T${a.start_time}`.localeCompare(`${b.scheduled_date}T${b.start_time}`)
    ));
  }, [apiData]);
  const organizations = useMemo(() => {
    const merged = new Map();
    arrayOf(apiData?.organizations).forEach((source) => {
      const organization = normalizeOrganization(source);
      if (organization.id !== null && organization.id !== undefined && organization.name) merged.set(String(organization.id), organization);
    });
    schedules.forEach((schedule) => {
      if (schedule.organization_id !== null && schedule.organization_id !== undefined && schedule.organization_name) {
        const key = String(schedule.organization_id);
        if (!merged.has(key)) merged.set(key, { id: schedule.organization_id, name: schedule.organization_name });
      }
    });
    return [...merged.values()].sort((a, b) => a.name.localeCompare(b.name));
  }, [apiData, schedules]);
  const projectSites = useMemo(() => arrayOf(apiData?.project_sites)
    .map(normalizeProjectSite)
    .filter((site) => site.id !== null && site.id !== undefined)
    .sort((a, b) => a.name.localeCompare(b.name)), [apiData]);
  const forecastKey = '10.7800,122.6253';
  const forecastRecords = useTideForecasts([forecastKey], tideRetryKey);
  const activeForecast = forecastRecords[forecastKey];
  const tidePayload = activeForecast?.payload;
  const tideLoading = !activeForecast || activeForecast.loading;
  const tideError = activeForecast?.error || '';
  const calendarGuide = useMemo(() => {
    return forecastPlantingGuide({ payload: tidePayload, now, allowOutdated: true });
  }, [tidePayload, now]);
  useEffect(() => {
    if (!formOpen || saving) return;
    const selection = { date: form.date, startTime: form.startTime, endTime: form.endTime };
    const assessment = assessSelectedTime(selection, tidePayload, now);
    if (assessment.status !== 'unsafe') {
      lastWarnedTime.current = null;
      acceptedUnsafeTime.current = null;
      return;
    }
    const key = selectionKey(selection);
    if (lastWarnedTime.current === key || !requiresTimeCautionBeforeSave(selection, assessment, acceptedUnsafeTime.current)) return;
    let cancelled = false;
    queueMicrotask(() => {
      if (cancelled) return;
      lastWarnedTime.current = key;
      setTimeCaution({ ...selection, assessment });
    });
    return () => { cancelled = true; };
  }, [formOpen, saving, tidePayload, now, form.date, form.startTime, form.endTime]);
  const tides = useMemo(() => normalizeTideEvents(tidePayload), [tidePayload]);
  const tidesByDate = useMemo(() => {
    const grouped = new Map();
    tides.forEach((event) => {
      const key = dateInManila(new Date(event.occurred_at));
      if (!grouped.has(key)) grouped.set(key, []);
      grouped.get(key).push(event);
    });
    return grouped;
  }, [tides]);
  const assessmentFor = (schedule) => {
    if (!tidePayload && tideLoading) return { status: 'unknown', label: 'Loading tide forecast...', reason: 'Checking the same water levels shown in the graph.' };
    if (!tidePayload && tideError) return { status: 'unknown', label: 'Refresh tides to check this activity', reason: tideError };
    return assessPlantingTime({ payload: tidePayload, now,
      startAt: schedule?.start_at || `${schedule?.scheduled_date}T${schedule?.start_time}+08:00`,
      endAt: schedule?.end_at || `${schedule?.scheduled_date}T${schedule?.end_time}+08:00`,
    });
  };
  const calendarDays = useMemo(() => calendarCells(month), [month]);
  const calendarWeeks = useMemo(() => Array.from({ length: 6 }, (_, index) => calendarDays.slice(index * 7, index * 7 + 7)), [calendarDays]);
  const schedulesByDate = useMemo(() => {
    const grouped = new Map();
    schedules.forEach((schedule) => {
      if (!schedule.scheduled_date) return;
      if (!grouped.has(schedule.scheduled_date)) grouped.set(schedule.scheduled_date, []);
      grouped.get(schedule.scheduled_date).push(schedule);
    });
    return grouped;
  }, [schedules]);
  const assignmentSites = useMemo(() => assigningSchedule
    ? projectSites.filter((site) => sameOrganization(site, assigningSchedule))
    : [], [assigningSchedule, projectSites]);

  const today = dateInManila();
  const upcoming = schedules.filter((schedule) => schedule.scheduled_date >= today && !['completed', 'cancelled'].includes(String(schedule.status).toLowerCase()));
  const confirmed = upcoming.filter((schedule) => String(schedule.status).toLowerCase() === 'confirmed').length;
  const selectedDateTides = form.date ? arrayOf(tidesByDate.get(form.date)) : [];

  const openCreate = () => {
    lastWarnedTime.current = null;
    acceptedUnsafeTime.current = null;
    setTimeCaution(null);
    setEditingSchedule(null);
    setForm(makeScheduleForm());
    setFormError('');
    setNotice('');
    setFormOpen(true);
  };

  const openEdit = (schedule) => {
    lastWarnedTime.current = null;
    acceptedUnsafeTime.current = null;
    setTimeCaution(null);
    setEditingSchedule(schedule);
    setForm(makeScheduleForm(schedule));
    setFormError('');
    setNotice('');
    setFormOpen(true);
  };

  const closeForm = () => {
    if (saving) return;
    setFormOpen(false);
    setEditingSchedule(null);
    setFormError('');
    setTimeCaution(null);
    acceptedUnsafeTime.current = null;
  };

  const persistSchedule = async (request) => {
    if (!request || saveInFlight.current) return;
    saveInFlight.current = true;
    setSaving(true);
    try {
      await fetchJson(request.path, {
        method: request.method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request.body),
      });
      setNotice(request.method === 'PUT' ? 'Planting schedule updated.' : 'Planting schedule created. Once confirmed, you can choose a planting area for it.');
      setFormOpen(false);
      setEditingSchedule(null);
      setMonth(request.body.date.slice(0, 7));
      setReloadKey((value) => value + 1);
    } catch (saveError) {
      setFormError(saveError.message || 'Could not save the planting schedule.');
    } finally {
      setTimeCaution(null);
      setSaving(false);
      saveInFlight.current = false;
    }
  };

  const saveSchedule = async (event) => {
    event?.preventDefault?.();
    if (saveInFlight.current || timeCaution) return;
    setFormError('');
    if (!token) {
      setFormError('Sign in as LGU staff to save planting schedules.');
      return;
    }
    const organizationName = form.organizationName.trim();
    const organization = organizations.find((option) => (
      option.name.trim().toLocaleLowerCase() === organizationName.toLocaleLowerCase()
    ));
    if (!organizationName || !form.title.trim() || !form.date || !form.startTime || !form.endTime) {
      setFormError('Enter the organization, activity title, date, and start and end times.');
      return;
    }
    if (form.endTime <= form.startTime) {
      setFormError('End time must be later than the start time.');
      return;
    }
    const intervalDays = 14;
    const participants = form.expectedParticipants === '' ? null : Number(form.expectedParticipants);
    const seedlings = form.seedlings === '' ? null : Number(form.seedlings);
    if (participants !== null && (!Number.isInteger(participants) || participants < 0)) {
      setFormError('Expected participants must be a whole number or blank.');
      return;
    }
    if (seedlings !== null && (!Number.isInteger(seedlings) || seedlings < 0)) {
      setFormError('Expected seedlings must be a whole number or blank.');
      return;
    }
    const selection = { date: form.date, startTime: form.startTime, endTime: form.endTime };
    const timeAssessment = assessSelectedTime(selection, tidePayload, now);
    const unsafeKey = selectionKey(selection);
    if (requiresTimeCautionBeforeSave(selection, timeAssessment, acceptedUnsafeTime.current)) {
      lastWarnedTime.current = unsafeKey;
      setTimeCaution({ ...selection, assessment: timeAssessment });
      return;
    }
    const body = {
      ...(organization ? { organization_id: organization.id } : {}),
      organization: organization?.name || organizationName,
      title: form.title.trim(),
      contact: form.contact.trim() || null,
      date: form.date,
      start_time: form.startTime,
      end_time: form.endTime,
      // Use the canonical API names. The response still exposes the shorter
      // aliases for display, but create/update requests should not depend on
      // alias fields that older running API processes may reject as extras.
      expected_planters: participants,
      expected_seedlings: seedlings,
      inspection_interval_days: intervalDays,
      status: form.status,
      notes: form.notes.trim() || null,
    };
    const path = editingSchedule
      ? `/api/planting-schedules/${encodeURIComponent(editingSchedule.id)}`
      : '/api/planting-schedules';
    const request = { path, method: editingSchedule ? 'PUT' : 'POST', body };
    await persistSchedule(request);
  };

  const openAssignment = (schedule) => {
    setAssigningSchedule(schedule);
    setAssignmentSiteId('');
    setAssignmentError('');
  };

  const assignSite = async () => {
    setAssignmentError('');
    if (!token || !assigningSchedule) {
      setAssignmentError('Please sign in again.');
      return;
    }
    if (String(assigningSchedule.status).toLowerCase() !== 'confirmed' || assigningSchedule.project_site_id) {
      setAssignmentError('Only a confirmed schedule without a planting area can be assigned.');
      return;
    }
    const site = assignmentSites.find((option) => String(option.id) === String(assignmentSiteId));
    if (!site) {
      setAssignmentError('Choose a planting area owned by this organization.');
      return;
    }
    setAssignmentSaving(true);
    try {
      await fetchJson(`/api/planting-schedules/${encodeURIComponent(assigningSchedule.id)}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_site_id: site.id }),
      });
      setAssigningSchedule(null);
      setAssignmentSiteId('');
      setNotice(`Planting area â€œ${site.name}â€ assigned to the confirmed schedule.`);
      setReloadKey((value) => value + 1);
    } catch (saveError) {
      setAssignmentError(saveError.message || 'Could not assign the planting area.');
    } finally {
      setAssignmentSaving(false);
    }
  };

  const deleteSchedule = async (schedule) => {
    if (!token) {
      setError('Sign in as LGU staff to delete planting schedules.');
      return;
    }
    if (!window.confirm(`Delete â€œ${schedule.title}â€ from the planting schedule?`)) return;
    setDeletingId(schedule.id);
    setError('');
    try {
      await fetchJson(`/api/planting-schedules/${encodeURIComponent(schedule.id)}`, { method: 'DELETE' });
      setNotice('Planting schedule deleted.');
      setReloadKey((value) => value + 1);
    } catch (deleteError) {
      setError(deleteError.message || 'Could not delete the planting schedule.');
    } finally {
      setDeletingId(null);
    }
  };

  return (
    <main className="scheduling-page">
      <header className="schedule-header">
        <div>
          <div className="schedule-eyebrow">Community planting</div>
          <h1>Scheduling</h1>
          <p>Check the water levels, then plan planting activities with your partner organizations.</p>
        </div>
        <button type="button" className="schedule-refresh" onClick={refreshSchedules} disabled={loading}>â†» {loading ? 'Refreshingâ€¦' : 'Refresh schedules'}</button>
      </header>

      <TideForecast payload={tidePayload} tides={tides} loading={tideLoading} error={tideError} onRetry={() => setTideRetryKey((value) => value + 1)} />

      <section className="schedule-panel" aria-labelledby="organization-schedules-title">
        <div className="schedule-panel-head schedule-toolbar">
          <div>
            <span className="schedule-step">Plan a date, confirm it, then choose a planting area</span>
            <h2 id="organization-schedules-title">Organization planting schedules</h2>
            <p>After confirming an activity, choose one of the organization’s planting areas. You can add planting areas in the Zone Editor.</p>
          </div>
          <button type="button" className="schedule-primary" onClick={openCreate}>+ Add new schedule</button>
        </div>

        {error ? <Message>{error}</Message> : null}
        {notice ? <Message type="success">{notice}</Message> : null}

        <div className="schedule-kpis">
          <article><span>Upcoming activities</span><strong>{formatCount(upcoming.length)}</strong><small>Pending or confirmed</small></article>
          <article><span>Confirmed</span><strong>{formatCount(confirmed)}</strong><small>Ready to choose a planting area</small></article>
          <article><span>Expected participants</span><strong>{formatCount(upcoming.reduce((sum, row) => sum + (numberOrNull(row.expected_participants) || 0), 0))}</strong><small>Across upcoming activities</small></article>
          <article><span>Expected seedlings</span><strong>{formatCount(upcoming.reduce((sum, row) => sum + (numberOrNull(row.seedlings) || 0), 0))}</strong><small>Across upcoming activities</small></article>
        </div>

        <PlantingLegend />
        <p className="schedule-threshold-note">Colors follow the graph above. Click an activity or tide time for details.</p>
        <div className="schedule-view-head">
          <div className="schedule-view-toggle" aria-label="Schedule view">
            <button type="button" className={view === 'calendar' ? 'is-active' : ''} onClick={() => setView('calendar')}>Calendar</button>
            <button type="button" className={view === 'list' ? 'is-active' : ''} onClick={() => setView('list')}>List</button>
          </div>
          {view === 'calendar' ? (
            <div className="schedule-month-nav">
              <button type="button" onClick={() => setMonth((current) => shiftMonth(current, -1))} aria-label="Previous month">&lsaquo;</button>
              <strong>{monthLabel(month)}</strong>
              <button type="button" onClick={() => setMonth((current) => shiftMonth(current, 1))} aria-label="Next month">&rsaquo;</button>
              <button type="button" onClick={() => setMonth(today.slice(0, 7))}>Today</button>
            </div>
          ) : null}
        </div>

        {loading && !apiData ? <Loading label="Loading organization schedulesâ€¦" /> : view === 'calendar' ? (
          <div className="schedule-calendar" role="grid" aria-label={`Planting schedule for ${monthLabel(month)}`}>
            <div className="schedule-calendar-row is-header" role="row">
              {['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'].map((day) => <div role="columnheader" key={day}>{day}</div>)}
            </div>
            {calendarWeeks.map((week) => (
              <div className="schedule-calendar-row" role="row" key={week[0]?.date}>
                {week.map((cell) => {
                  const daySchedules = arrayOf(schedulesByDate.get(cell.date));
                  const dayTides = arrayOf(tidesByDate.get(cell.date));
                  return (
                    <article className={`schedule-calendar-day${cell.inMonth ? '' : ' is-outside'}${cell.date === today ? ' is-today' : ''}`} role="gridcell" key={cell.date}>
                      <time dateTime={cell.date}>{cell.day}</time>
                      <div className="schedule-calendar-events">
                        {daySchedules.map((schedule) => <ScheduleCalendarEvent key={schedule.id} schedule={schedule} assessment={assessmentFor(schedule)} onOpen={(item) => setCalendarEntry({ kind: 'schedule', schedule: item })} />)}
                        {dayTides.slice(0, 4).map((tide) => (
                          <TideTime key={`${tide.occurred_at}-${tideKind(tide)}`} tide={tide} guide={calendarGuide} onOpen={(item) => setCalendarEntry({ kind: 'tide', tide: item })} />
                        ))}
                      </div>
                    </article>
                  );
                })}
              </div>
            ))}
          </div>
        ) : schedules.length ? (
          <div className="schedule-table-wrap">
            <table>
              <caption className="sr-only">Organization planting schedules</caption>
              <thead><tr><th scope="col">Date</th><th scope="col">Activity</th><th scope="col">Planting area</th><th scope="col">Days between plant checks</th><th scope="col">Status</th><th scope="col">Planting advice</th><th scope="col">Actions</th></tr></thead>
              <tbody>{schedules.map((schedule) => {
                const canAssign = String(schedule.status).toLowerCase() === 'confirmed' && !schedule.project_site_id;
                return (
                  <tr key={schedule.id}>
                    <td><strong>{formatDate(`${schedule.scheduled_date}T12:00:00+08:00`)}</strong><small>{scheduleTimeLabel(schedule)}</small></td>
                    <td><strong>{schedule.title}</strong><small>{schedule.organization_name}</small><small>{formatCount(schedule.expected_participants)} participants</small></td>
                    <td>{schedule.project_site_name || (canAssign ? 'Choose a planting area' : 'No planting area chosen')}</td>
                    <td>{schedule.inspection_interval_days ? `Every ${formatCount(schedule.inspection_interval_days)} days` : 'Not set'}</td>
                    <td><span className={`schedule-status is-${String(schedule.status).toLowerCase()}`}>{scheduleStatusLabel(schedule.status)}</span></td>
                    <td><PlantingBadge assessment={assessmentFor(schedule)} /><small>{assessmentFor(schedule).reason}</small></td>
                    <td><div className="schedule-row-actions">
                      {canAssign ? <button type="button" className="is-primary" onClick={() => openAssignment(schedule)}>Choose area</button> : null}
                      <button type="button" onClick={() => openEdit(schedule)}>Edit</button>
                      <button type="button" className="is-danger" onClick={() => deleteSchedule(schedule)} disabled={String(deletingId) === String(schedule.id)}>{String(deletingId) === String(schedule.id) ? 'Deletingâ€¦' : 'Delete'}</button>
                    </div></td>
                  </tr>
                );
              })}</tbody>
            </table>
          </div>
        ) : (
          <div className="schedule-empty"><strong>No planting schedules yet</strong><p>Check the tide forecast, then create the first organization planting schedule.</p></div>
        )}
      </section>

      <CalendarEntryDetails entry={calendarEntry} assessment={calendarEntry?.kind === 'schedule' ? assessmentFor(calendarEntry.schedule) : null}
        guide={calendarGuide} onClose={() => setCalendarEntry(null)} onEdit={openEdit} />

      <HighTideCaution selection={timeCaution}
        onContinue={() => {
          acceptedUnsafeTime.current = selectionKey(timeCaution);
          setTimeCaution(null);
        }}
        onBack={() => setTimeCaution(null)} />

      <Modal
        open={formOpen && !timeCaution}
        title={editingSchedule ? 'Edit planting schedule' : 'Add new planting schedule'}
        confirmLabel={editingSchedule ? 'Save changes' : 'Create schedule'}
        cancelLabel="Close"
        busy={saving}
        onConfirm={saveSchedule}
        onCancel={closeForm}
        className="modal-card-wide schedule-form-modal"
        variant="info"
      >
        <form className="schedule-form" onSubmit={saveSchedule}>
          {formError ? <Message>{formError}</Message> : null}
          <div className="schedule-form-assessment">
            <PlantingBadge assessment={assessmentFor({ ...editingSchedule,
              start_at: `${form.date}T${form.startTime}+08:00`, end_at: `${form.date}T${form.endTime}+08:00`,
            })} />
            <p>{assessmentFor({ ...editingSchedule, start_at: `${form.date}T${form.startTime}+08:00`, end_at: `${form.date}T${form.endTime}+08:00` }).reason}</p>
          </div>
          <div className="schedule-form-grid">
            <label>
              <span>Organization *</span>
              <input
                value={form.organizationName}
                list="schedule-organizations"
                maxLength="200"
                placeholder="Enter or select an organization"
                onChange={(event) => {
                  const value = event.target.value;
                  const matching = organizations.find((organization) => organization.name.trim().toLocaleLowerCase() === value.trim().toLocaleLowerCase());
                  setForm((current) => ({ ...current, organizationName: value, organizationId: matching ? String(matching.id) : '' }));
                }}
                disabled={Boolean(editingSchedule?.project_site_id)}
                required
              />
              <datalist id="schedule-organizations">
                {organizations.map((organization) => <option key={organization.id} value={organization.name} />)}
              </datalist>
              {!organizations.length ? <small>Enter the organization name. After saving, you can add planting areas for it.</small> : null}
              {editingSchedule?.project_site_id ? <small>This activity already has a planting area, so its organization cannot be changed.</small> : null}
            </label>
            <label>
              <span>Activity title *</span>
              <input value={form.title} maxLength="180" onChange={(event) => setForm((current) => ({ ...current, title: event.target.value }))} required />
            </label>
            <label>
              <span>Contact</span>
              <input value={form.contact} maxLength="160" placeholder="Name, phone, or email" onChange={(event) => setForm((current) => ({ ...current, contact: event.target.value }))} />
            </label>
            <label>
              <span>Date *</span>
              <input type="date" value={form.date} onChange={(event) => setForm((current) => ({ ...current, date: event.target.value }))} required />
            </label>
            <label>
              <span>Start time *</span>
              <input type="time" value={form.startTime} onChange={(event) => setForm((current) => ({ ...current, startTime: event.target.value }))} required />
            </label>
            <label>
              <span>End time *</span>
              <input type="time" value={form.endTime} onChange={(event) => setForm((current) => ({ ...current, endTime: event.target.value }))} required />
            </label>
            <label>
              <span>Expected participants</span>
              <input type="number" min="0" step="1" value={form.expectedParticipants} onChange={(event) => setForm((current) => ({ ...current, expectedParticipants: event.target.value }))} />
            </label>
            <label>
              <span>Expected seedlings</span>
              <input type="number" min="0" step="1" value={form.seedlings} onChange={(event) => setForm((current) => ({ ...current, seedlings: event.target.value }))} />
            </label>
            <label>
              <span>Days between plant checks</span>
              <input type="number" value="14" readOnly aria-readonly="true" />
              <small>Monitoring is due every 14 days from the actual planting date.</small>
            </label>
            <label>
              <span>Status *</span>
              <select value={form.status} onChange={(event) => setForm((current) => ({ ...current, status: event.target.value }))} required>
                <option value="requested">Pending</option>
                <option value="confirmed">Confirmed</option>
                {editingSchedule ? <option value="tentative">Tentative</option> : null}
                {editingSchedule ? <option value="completed">Completed</option> : null}
                {editingSchedule ? <option value="cancelled">Cancelled</option> : null}
              </select>
            </label>
            <label className="schedule-form-notes">
              <span>Notes</span>
              <textarea rows="3" maxLength="2000" value={form.notes} onChange={(event) => setForm((current) => ({ ...current, notes: event.target.value }))} />
            </label>
          </div>
          {form.date ? (
            <div className="schedule-form-tides">
              <span>Tide forecast for {formatDate(`${form.date}T12:00:00+08:00`)}</span>
              {selectedDateTides.length ? selectedDateTides.map((tide) => (
                <TideTime key={`${tide.occurred_at}-${tideKind(tide)}`} tide={tide} guide={calendarGuide} />
              )) : <small>High and low tide times are not available for this date yet.</small>}
            </div>
          ) : null}
          <p className="schedule-form-footnote">Choose a planting area after confirming the activity. All times are Philippine time.</p>
        </form>
      </Modal>

      <Modal
        open={Boolean(assigningSchedule)}
        title="Choose planting area"
        confirmLabel="Choose area"
        cancelLabel="Close"
        busy={assignmentSaving}
        onConfirm={assignSite}
        onCancel={() => {
          if (!assignmentSaving) setAssigningSchedule(null);
        }}
        variant="success"
      >
        <div className="schedule-assignment-form">
          <p><strong>{assigningSchedule?.title}</strong> is confirmed for {assigningSchedule?.organization_name}.</p>
          {assignmentError ? <Message>{assignmentError}</Message> : null}
          <label>
            <span>Organization’s planting area *</span>
            <select value={assignmentSiteId} onChange={(event) => setAssignmentSiteId(event.target.value)}>
              <option value="">Select a planting area</option>
              {assignmentSites.map((site) => <option key={site.id} value={site.id}>{site.name}</option>)}
            </select>
          </label>
          {!assignmentSites.length ? <small>This organization has no planting areas yet. Add one in the Zone Editor first.</small> : null}
        </div>
      </Modal>
    </main>
  );
}
