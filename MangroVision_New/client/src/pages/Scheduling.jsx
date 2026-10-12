import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useLocation } from 'react-router-dom';
import Modal from '../components/Modal';
import TideChart from '../components/TideChart';
import useFormFeedback from '../utils/useFormFeedback';
import { FieldError, FormErrorSummary, FormMessage } from '../components/FormFeedback';
import { formRequest } from '../utils/formValidation';
import WebsiteRequests from '../components/WebsiteRequests';
import useTideForecasts from '../utils/useTideForecasts';
import {
  COASTAL_FORECAST_KEY, PLANTING_STATES, assessGraphWindow, assessGraphTime, chartLevelSeries, dayLevelSeries, finiteNumber, forecastIssue,
  forecastPlantingGuide, guideLevelState,
} from '../utils/plantingTides';
import { useAuthStore } from '../stores/authStore';
import { groupCalendarSchedules } from '../utils/calendarScheduleGroups';
import { APPOINTMENT_TYPES, appointmentTypeLabel, isPlantingAppointment } from '../utils/appointmentTypes';
import { bookingErrors } from '../like/booking';
import { scheduleContactFields, scheduleContactSummary } from '../utils/scheduleContact';
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
  return formRequest(`${API}${path}`, options, 'Could not save the activity. Check the details and retry.');
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

function scheduleEndTime(schedule) {
  if (schedule?.end_time) return schedule.end_time;
  const parsed = schedule?.end_at ? new Date(schedule.end_at) : null;
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
    end_time: scheduleEndTime(schedule),
    expected_participants: schedule?.expected_participants ?? schedule?.expected_planters ?? schedule?.participants ?? null,
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
    appointmentType: schedule ? schedule.appointment_type || 'tree_planting' : '',
    ...scheduleContactFields(schedule || {}),
    date: schedule?.scheduled_date || (startAt && !Number.isNaN(startAt.getTime()) ? dateInManila(startAt) : ''),
    startTime: schedule?.start_time || (startAt && !Number.isNaN(startAt.getTime())
      ? new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: TIMEZONE }).format(startAt)
      : ''),
    endTime: schedule?.end_time || (endAt && !Number.isNaN(endAt.getTime())
      ? new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', hour12: false, timeZone: TIMEZONE }).format(endAt)
      : ''),
    expectedParticipants: schedule?.expected_participants ?? '',
    inspectionIntervalDays: 14,
    status: schedule?.status ?? 'requested',
    notes: schedule?.notes ?? '',
  };
}

function scheduleStatusLabel(value) {
  if (String(value).toLowerCase() === 'requested') return 'Pending confirmation';
  const clean = String(value || 'requested').replaceAll('_', ' ');
  return clean.charAt(0).toUpperCase() + clean.slice(1);
}

function scheduleTimeLabel(schedule) {
  const start = scheduleStartTime(schedule);
  const end = scheduleEndTime(schedule);
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
  </div>;
}

function ScheduleCalendarEvent({ group, assessment, onOpen }) {
  const schedule = group.schedule;
  const grouped = group.schedules.length > 1;
  const organizationLabel = grouped
    ? group.organizationCount === group.schedules.length
      ? `${group.organizationCount} organizations`
      : `${group.schedules.length} schedules`
    : schedule.organization_name || 'Organization not set';
  const state = PLANTING_STATES[assessment.status] || PLANTING_STATES.unknown;
  return <button type="button" className={`schedule-calendar-event planting-${assessment.status}`} onClick={() => onOpen(group)}
    aria-haspopup="dialog" aria-label={`${schedule.title}, ${scheduleTimeLabel(schedule)}. ${organizationLabel}. ${isPlantingAppointment(schedule) ? state.label : appointmentTypeLabel(schedule.appointment_type)}. View activity details.`}>
    <span className="schedule-entry-top"><time>{scheduleTimeLabel(schedule)}</time></span>
    <strong className="schedule-entry-title">{schedule.title}</strong>
    <small>{appointmentTypeLabel(schedule.appointment_type)}</small>
    <span className="schedule-entry-organizations">{organizationLabel}</span>
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
    <strong>{scheduleStatusLabel(tideKind(tide))} tide</strong>
    <time dateTime={new Date(tide.occurred_at).toISOString()}>{tideClock(tide)}</time>
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

function ScheduleDetails({ schedule, showOrganization = true }) {
  return <dl>
    <div><dt>Appointment type</dt><dd>{appointmentTypeLabel(schedule.appointment_type)}</dd></div>
    <div><dt>Status</dt><dd>{scheduleStatusLabel(schedule.status)}</dd></div>
    <div><dt>Source</dt><dd>{schedule.source === 'website' ? 'LIKE website' : 'Staff-entered'}</dd></div>
    {showOrganization ? <div><dt>Organization</dt><dd>{schedule.organization_name || 'Not set'}</dd></div> : null}
    {isPlantingAppointment(schedule) ? <div><dt>Planting area</dt><dd>{schedule.project_site_name || 'No planting area chosen'}</dd></div> : null}
    <div><dt>Participants</dt><dd>{formatCount(schedule.expected_participants)}</dd></div>
    {isPlantingAppointment(schedule) ? <div><dt>Plant checks</dt><dd>{schedule.inspection_interval_days ? `Every ${formatCount(schedule.inspection_interval_days)} days` : 'Not set'}</dd></div> : null}
    {schedule.contact ? <div><dt>Contact</dt><dd>{schedule.contact}</dd></div> : null}
    {schedule.notification_email ? <div><dt>Notification email</dt><dd>{schedule.notification_email}</dd></div> : null}
    {schedule.notification_delivery ? <div><dt>Organization email</dt><dd>{schedule.notification_delivery.awaiting_retry ? 'Delivery failed; queued for automatic retry'
      : schedule.notification_delivery.status === 'sent' ? 'Sent' : schedule.notification_delivery.status === 'sending' ? 'Sending' : 'Queued'}</dd></div> : null}
    {schedule.notes ? <div><dt>Notes</dt><dd>{schedule.notes}</dd></div> : null}
  </dl>;
}

function CalendarEntryDetails({ entry, assessment, guide, payload, now, onClose, onEdit }) {
  if (!entry) return null;
  const schedule = entry.kind === 'schedule' ? entry.schedule : null;
  const schedules = schedule ? arrayOf(entry.schedules) : [];
  const grouped = schedules.length > 1;
  const tide = entry.kind === 'tide' ? entry.tide : null;
  const advice = schedule ? assessment : { status: tidePlantingState(tide, guide) };
  const editSchedule = (item) => { onClose(); onEdit(item); };
  return <Modal open title={schedule ? schedule.title : `${scheduleStatusLabel(tideKind(tide))} tide`}
    className={`modal-card-wide schedule-details-modal${grouped ? ' is-grouped' : ''}${tide ? ' is-tide' : ''}`} variant="info"
    confirmLabel={schedule && !grouped ? 'Edit activity' : 'Close'} cancelLabel={schedule && !grouped ? 'Close' : null}
    onCancel={onClose} onConfirm={schedule && !grouped ? () => editSchedule(schedule) : onClose}>
    <div className="schedule-entry-details">
      <div className="schedule-detail-when">
        <strong>{schedule ? scheduleTimeLabel(schedule) : tideClock(tide)}</strong>
        <span>{schedule ? formatDate(`${scheduleDate(schedule)}T12:00:00+08:00`) : formatDate(tide.occurred_at)} · Philippine time</span>
      </div>
      <div className="schedule-detail-advice">
        {schedule && !isPlantingAppointment(schedule) ? <strong>{appointmentTypeLabel(schedule.appointment_type)}</strong> : <PlantingBadge assessment={advice} />}
        <p>{schedule ? assessment.reason : 'Uses the same estimated planting limit as the graph. Check your planting area before going.'}</p>
      </div>
      {grouped ? <>
        <p className="schedule-group-summary">{schedules.length} separate activities at this time. View or edit each organization's activity below.</p>
        <div className="schedule-group-list" role="region" aria-label="Organization activities" tabIndex={0}>
          {schedules.map((item) => <section className="schedule-group-item" key={item.id}>
            <div className="schedule-group-item-header">
              <h3>{item.organization_name || 'Organization not set'}</h3>
              <button type="button" className="schedule-group-edit" onClick={() => editSchedule(item)}
                aria-label={`Edit activity for ${item.organization_name || 'this organization'}`}>Edit activity</button>
            </div>
            <ScheduleDetails schedule={item} showOrganization={false} />
          </section>)}
        </div>
      </> : schedule ? <ScheduleDetails schedule={schedule} /> : <><dl>
        <div><dt>Water level</dt><dd>{formatHeight(tide.height_m)}</dd></div>
        <div><dt>Estimated planting limit</dt><dd>{formatHeight(guide?.threshold)}</dd></div>
      </dl><DailyTideDetails tide={tide} payload={payload} guide={guide} now={now} /></>}
    </div>
  </Modal>;
}

function DailyTideDetails({ tide, payload, guide, now }) {
  const [openedAt] = useState(Date.now);
  const day = dateInManila(new Date(tide.occurred_at));
  const rows = dayLevelSeries(payload, day);
  const validRows = rows.filter((row) => row.height_m !== null);
  const dayStart = Date.parse(`${day}T00:00:00+08:00`);
  const dayEnd = dayStart + 86400_000;
  const issue = payload ? forecastIssue(payload, now ?? openedAt) || (guide.status === 'unknown' ? 'Some hourly water levels are missing.' : '') : '';
  const tides = normalizeTideEvents(payload).filter((event) => dateInManila(new Date(event.occurred_at)) === day);
  return <section className="schedule-tide-day-detail" aria-label={`Daily tide forecast for ${day}`}>
    <h3>Water levels for this day</h3>
    {issue ? <p className="schedule-tide-day-warning" role="status">{issue}</p> : null}
    {validRows.length >= 2 ? <>
      <div className="schedule-graph-legend" aria-label="Daily planting color guide">
        <span className="is-safe"><i aria-hidden="true" />Water low enough</span>
        <span className="is-unsafe"><i aria-hidden="true" />Water too high</span>
      </div>
      <TideChart rows={rows} guide={guide} tides={tides} day={day} selectedTide={tide}
        datumLabel={payload.datum === 'MSL' ? 'meters from average sea level' : 'meters; starting level not confirmed'} />
      {validRows[0].occurred_at > dayStart || validRows.at(-1).occurred_at < dayEnd
        ? <p className="schedule-tide-day-note">Some hours are outside this forecast.</p> : null}
    </> : <p className="schedule-tide-day-note">Hourly water levels are unavailable for this day.</p>}
  </section>;
}

function TideForecast({ payload, tides = [], loading, error, site, onRetry }) {
  const [now, setNow] = useState(() => Date.now());
  useEffect(() => {
    const timer = window.setInterval(() => setNow(Date.now()), 60_000);
    return () => window.clearInterval(timer);
  }, []);
  const samples = chartLevelSeries(payload);
  const rows = samples.length ? samples : tides.map((row) => ({ ...row, occurred_at: new Date(row.occurred_at).getTime() }));
  // Keep the visible curve colored from the displayed readings even when they
  // need refreshing. Scheduling assessments still require a current forecast.
  const guide = forecastPlantingGuide({ site, payload, now, allowOutdated: true });
  const issue = forecastIssue(payload, now) || (guide.status === 'unknown'
    ? 'Some water levels are missing. Refresh tides to see the forecast.' : '');
  const { threshold, preview, percentage } = guide;
  const usable = guide.status !== 'unknown';
  const tideEvents = tides.length ? tides : normalizeTideEvents(payload);
  const datumLabel = payload?.datum === 'MSL' ? 'meters from average sea level' : 'meters; starting level not confirmed';
  const referenceLabel = preview ? 'Estimated planting limit' : 'Measured ground height';
  const statusLabel = (height) => {
    const state = guideLevelState(height, guide);
    return state === 'safe' ? 'Water low enough · Good time to plant'
      : state === 'unsafe' ? guide.status === 'safe' ? 'Water too high · Avoid planting' : 'Avoid planting'
        : 'Planting guidance unavailable';
  };

  return (
    <section className="schedule-panel schedule-tide-panel" aria-labelledby="standalone-tide-title">
      <div className="schedule-panel-head">
        <div>
          <h2 id="standalone-tide-title">When to plant</h2>
          <p>Estimated times. Check water and weather at the site.</p>
        </div>
        <div className="schedule-tide-tools">
          <button type="button" onClick={onRetry} disabled={loading}>{loading ? 'Loading tides...' : 'Refresh tides'}</button>
        </div>
      </div>
      {issue && !loading && !error ? <Message type="warning">{issue}</Message> : null}
      {loading ? <Loading label="Loading the tide forecast..." /> : null}
      {error ? <Message>{error}</Message> : null}
      {!loading && !error && rows.length ? (
        <div className="schedule-tide-chart-section">
          <div className="schedule-graph-legend" aria-label="Planting color guide">
            <span className="is-safe"><i aria-hidden="true" />Water low enough · Good time to plant</span>
            <span className="is-unsafe"><i aria-hidden="true" />{guide.status === 'safe' ? 'Water too high' : 'Avoid planting'}</span>
            {!usable ? <span className="is-unknown"><i aria-hidden="true" />Time unavailable</span> : null}
            <span className="schedule-reference-key">{referenceLabel}{threshold !== null ? `: ${threshold.toFixed(2)} m` : ': unknown'}</span>
          </div>
          <TideChart rows={rows} guide={guide} tides={tideEvents} datumLabel={datumLabel}
            connectPoints={samples.filter((row) => row.height_m !== null).length >= 2} />
          <details className="schedule-table-fallback">
            <summary>Water levels</summary>
            <div className="schedule-table-wrap">
              <table>
                <thead><tr><th scope="col">Date and time (Philippine time)</th><th scope="col">Water level ({datumLabel})</th><th scope="col">Planting advice{preview ? ' (estimated)' : ''}</th></tr></thead>
                <tbody>{rows.filter((row) => row.height_m !== null).map((tide) => (
                  <tr key={tide.occurred_at}>
                    <td>{formatDate(tide.occurred_at, true)}</td><td>{formatHeight(tide.height_m)}</td>
                    <td style={{ color: PLANTING_STATES[guideLevelState(tide.height_m, guide)].color }}>{statusLabel(tide.height_m)}</td>
                  </tr>
                ))}</tbody>
              </table>
            </div>
          </details>
        </div>
      ) : null}
      <details className="schedule-estimate-details">
        <summary>How these times are estimated</summary>
        <div>
          <p>High tide marks the highest water level. Low tide marks the lowest. The graph's colors compare water levels with the planting limit.</p>
          <p>{preview
            ? 'This guide uses an estimated ground height. Check the actual ground height and water at your site.'
            : 'This guide uses the measured ground height for this site.'}</p>
          <dl>
            <div><dt>{referenceLabel}</dt><dd>{threshold === null ? 'Unavailable' : formatHeight(threshold)}</dd></div>
            <div><dt>Estimated time underwater</dt><dd>{percentage === null ? 'Unavailable' : `${percentage.toFixed(1)}% of the full forecast`}</dd></div>
          </dl>
          <p>Green suggests a planting time when water is below the limit. The ground must also be at or above average sea level and underwater no more than 30% of the time. A short forecast cannot confirm long-term site conditions.</p>
          <p>{payload?.datum === 'MSL' ? 'Water heights are measured from average sea level.' : 'The starting level for water measurements has not been confirmed.'}</p>
          <a href="https://cms.zsl.org/sites/default/files/2023-02/1%20Manual%20-%20Community-based%20Mangrove%20Rehabilitation.pdf" target="_blank" rel="noreferrer">Read the planting guide</a>
        </div>
      </details>
      {payload?.attribution ? <footer className="schedule-tide-source">
        {payload.attribution_url
          ? <a href={payload.attribution_url} target="_blank" rel="noreferrer">{payload.attribution}</a> : payload.attribution}
      </footer> : null}
    </section>
  );
}

export default function Scheduling() {
  const location = useLocation();
  const token = useAuthStore((state) => state.token);
  const [apiData, setApiData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [reloadKey, setReloadKey] = useState(0);
  const [reviewRequest, setReviewRequest] = useState(null);
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
  const feedback = useFormFeedback({
    organizationName: { label: 'Organization', aliases: ['organization', 'organization_id'], serverTerms: ['organization'] },
    appointmentType: { label: 'Appointment type', aliases: ['appointment_type'], serverTerms: ['appointment type'] },
    contactName: { label: 'Contact person', aliases: ['contact_name'], serverTerms: ['contact person'] },
    phone: { label: 'Phone number', serverTerms: ['phone number'] },
    email: { label: 'Email', aliases: ['notification_email'], serverTerms: ['email address', 'notification email'] },
    date: { label: 'Date' },
    startTime: { label: 'Start time', aliases: ['start_time', 'start_at'] },
    endTime: { label: 'End time', aliases: ['end_time', 'end_at'], serverTerms: ['end time', 'end_time'], validate: (value) => value && form.startTime && value <= form.startTime ? 'Choose an end time later than the start time on the same day.' : '' },
    expectedParticipants: { label: 'Participants', aliases: ['expected_planters', 'expected_participants'], serverTerms: ['participant count'] },
    status: { label: 'Activity status' },
    notes: { label: 'Notes' },
  });
  const assignmentFeedback = useFormFeedback({ site: { label: 'Planting area', aliases: ['project_site_id'], serverTerms: ['planting area', 'project site'] } });
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
  const formPlanting = form.appointmentType === 'tree_planting';
  const originalContact = scheduleContactFields(editingSchedule || {});

  useEffect(() => {
    if (new URLSearchParams(location.search).get('requests') !== 'pending') return undefined;
    const frame = window.requestAnimationFrame(() => {
      const panel = document.getElementById('website-requests');
      panel?.scrollIntoView({ block: 'start' });
      panel?.focus({ preventScroll: true });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [location.key, location.search, loading]);

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
  const approvedSchedules = useMemo(() => schedules.filter((schedule) => (
    ['confirmed', 'in_progress', 'completed'].includes(String(schedule.status).toLowerCase())
  )), [schedules]);
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
  const forecastKey = COASTAL_FORECAST_KEY;
  const forecastRecords = useTideForecasts([forecastKey], tideRetryKey);
  const activeForecast = forecastRecords[forecastKey];
  const tidePayload = activeForecast?.payload;
  const tideLoading = !activeForecast || activeForecast.loading;
  const tideError = activeForecast?.error || '';
  const calendarGuide = useMemo(() => {
    return forecastPlantingGuide({ payload: tidePayload, now, allowOutdated: true });
  }, [tidePayload, now]);
  useEffect(() => {
    if (!formOpen || saving || editingSchedule?.planting_started || !formPlanting) return;
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
  }, [formOpen, saving, editingSchedule, formPlanting, tidePayload, now, form.date, form.startTime, form.endTime]);
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
    if (schedule && !isPlantingAppointment(schedule)) return { status: 'unknown', label: appointmentTypeLabel(schedule.appointment_type), reason: 'Check staff availability and site conditions with LIKE.' };
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
    groupCalendarSchedules(approvedSchedules).forEach((group) => {
      const schedule = group.schedule;
      if (!schedule.scheduled_date) return;
      if (!grouped.has(schedule.scheduled_date)) grouped.set(schedule.scheduled_date, []);
      grouped.get(schedule.scheduled_date).push(group);
    });
    return grouped;
  }, [approvedSchedules]);
  const assignmentSites = useMemo(() => assigningSchedule
    ? projectSites.filter((site) => sameOrganization(site, assigningSchedule))
    : [], [assigningSchedule, projectSites]);

  const today = dateInManila();
  const upcoming = approvedSchedules.filter((schedule) => schedule.scheduled_date >= today && String(schedule.status).toLowerCase() !== 'completed');
  const confirmed = upcoming.filter((schedule) => String(schedule.status).toLowerCase() === 'confirmed').length;
  const selectedDateTides = form.date ? arrayOf(tidesByDate.get(form.date)) : [];

  const openCreate = () => {
    feedback.clear();
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
    feedback.clear();
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
    feedback.clear();
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
      const saved = await fetchJson(request.path, {
        method: request.method,
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(request.body),
      });
      const emailNotice = saved?.email_update?.status === 'missing_recipient'
        ? ' No email was queued. Add the organization’s notification email to send the current details. Planters can view the saved schedule.'
        : saved?.email_update ? (saved.email_update.kind === 'cancellation' ? ' A cancellation email is queued for the organization.'
          : saved.email_update.kind === 'update' ? ' An email with the revised details is queued for the organization.'
            : ' A confirmation email is queued for the organization.') : '';
      setNotice((request.method === 'PUT' ? 'Appointment updated.' : 'Appointment created.') + emailNotice);
      setFormOpen(false);
      setEditingSchedule(null);
      setMonth(request.body.date.slice(0, 7));
      setReloadKey((value) => value + 1);
    } catch (saveError) {
      if (request.method === 'POST' && (saveError.uncertain || saveError.status >= 500)) {
        setFormError('Could not verify whether the new activity was saved. Check List before submitting it again to avoid a duplicate. Your entries are kept.');
      } else if (!feedback.fromServer(saveError)) setFormError(saveError.message || 'Could not save the planting schedule.');
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
    const commonErrors = bookingErrors({
      organization: form.organizationName, contact_name: form.contactName, phone: form.phone, email: form.email,
      appointment_type: form.appointmentType, date: form.date, start_time: form.startTime, end_time: form.endTime,
      participants: form.expectedParticipants, consent: true,
    });
    const fieldNames = { organization: 'organizationName', contact_name: 'contactName', appointment_type: 'appointmentType',
      start_time: 'startTime', end_time: 'endTime', participants: 'expectedParticipants' };
    const errors = Object.fromEntries(Object.entries(commonErrors)
      .filter(([field]) => !editingSchedule || (['phone', 'email'].includes(field) && form[field].trim()))
      .map(([field, message]) => [fieldNames[field] || field, message]));
    if (!feedback.validate(errors)) return;
    const organizationName = form.organizationName.trim();
    const organization = organizations.find((option) => (
      option.name.trim().toLocaleLowerCase() === organizationName.toLocaleLowerCase()
    ));
    if (form.endTime <= form.startTime) {
      feedback.reject({ endTime: 'Choose an end time later than the start time on the same day.' });
      return;
    }
    const intervalDays = editingSchedule?.inspection_interval_days ?? 14;
    const participants = form.expectedParticipants === '' ? null : Number(form.expectedParticipants);
    if (participants !== null && (!Number.isInteger(participants) || participants < 0)) {
      feedback.reject({ expectedParticipants: 'Enter a whole number of 0 or more, or leave this blank.' });
      return;
    }
    const selection = { date: form.date, startTime: form.startTime, endTime: form.endTime };
    const timeAssessment = assessSelectedTime(selection, tidePayload, now);
    const unsafeKey = selectionKey(selection);
    if (!editingSchedule?.planting_started && formPlanting && requiresTimeCautionBeforeSave(selection, timeAssessment, acceptedUnsafeTime.current)) {
      lastWarnedTime.current = unsafeKey;
      setTimeCaution({ ...selection, assessment: timeAssessment });
      return;
    }
    const body = {
      ...(organization ? { organization_id: organization.id } : {}),
      organization: organization?.name || organizationName,
      ...(editingSchedule ? {
        contact: form.contactName === originalContact.contactName && form.phone === originalContact.phone
          ? editingSchedule.contact || null : scheduleContactSummary(form.contactName, form.phone),
        ...(editingSchedule.source === 'website' ? {} : { notification_email: form.email.trim() || null }),
      } : {
        appointment_type: form.appointmentType,
        contact_name: form.contactName.trim(), phone: form.phone.trim(), email: form.email.trim(),
      }),
      date: form.date,
      start_time: form.startTime,
      end_time: form.endTime,
      start_at: `${form.date}T${form.startTime}:00+08:00`,
      end_at: `${form.date}T${form.endTime}:00+08:00`,
      // Use the canonical API names. The response still exposes the shorter
      // aliases for display, but create/update requests should not depend on
      // alias fields that older running API processes may reject as extras.
      expected_planters: participants,
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
    if (!isPlantingAppointment(schedule)) return;
    assignmentFeedback.clear();
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
    if (!isPlantingAppointment(assigningSchedule)) {
      setAssignmentError('Only tree-planting appointments need a planting area.');
      return;
    }
    if (String(assigningSchedule.status).toLowerCase() !== 'confirmed' || assigningSchedule.project_site_id) {
      setAssignmentError('Only a confirmed schedule without a planting area can be assigned.');
      return;
    }
    const site = assignmentSites.find((option) => String(option.id) === String(assignmentSiteId));
    if (!site) {
      assignmentFeedback.reject({ site: 'Choose a planting area owned by this organization. Add one in the Zone Editor if none are listed.' });
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
      if (!assignmentFeedback.fromServer(saveError)) setAssignmentError(saveError.message || 'Could not assign the planting area.');
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
          <h1>Scheduling</h1>
          <p>Plan activities and review pending requests.</p>
        </div>
        <button type="button" className="schedule-refresh" onClick={refreshSchedules} disabled={loading}>â†» {loading ? 'Refreshingâ€¦' : 'Refresh schedules'}</button>
      </header>

      <WebsiteRequests token={token} reloadKey={reloadKey} organizations={organizations} schedules={schedules}
        assessmentFor={assessmentFor} renderAdvice={(assessment) => <PlantingBadge assessment={assessment} />}
        reviewRequest={reviewRequest} onReview={setReviewRequest}
        onClose={() => setReviewRequest(null)} onChanged={() => setReloadKey((value) => value + 1)} />

      <TideForecast payload={tidePayload} tides={tides} loading={tideLoading} error={tideError} onRetry={() => setTideRetryKey((value) => value + 1)} />

      <section className="schedule-panel" aria-labelledby="organization-schedules-title">
        <div className="schedule-panel-head schedule-toolbar">
          <div>
            <h2 id="organization-schedules-title">Schedules</h2>
            <p>Calendar shows approved activities. List shows all schedules.</p>
          </div>
          <button type="button" className="schedule-primary" onClick={openCreate}>+ Add schedule</button>
        </div>

        {error ? <Message>{error}</Message> : null}
        <FormMessage success={notice} />

        <div className="schedule-kpis">
          <article><span>Upcoming schedules</span><strong>{formatCount(upcoming.length)}</strong><small>By organization</small></article>
          <article><span>Confirmed schedules</span><strong>{formatCount(confirmed)}</strong><small>LGU-approved activities</small></article>
          <article><span>Expected participants</span><strong>{formatCount(upcoming.reduce((sum, row) => sum + (numberOrNull(row.expected_participants) || 0), 0))}</strong><small>Across upcoming activities</small></article>
        </div>

        <PlantingLegend />
        <p className="schedule-threshold-note">Click an activity or tide time for details.</p>
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
                        {daySchedules.map((group) => <ScheduleCalendarEvent key={group.key} group={group} assessment={assessmentFor(group.schedule)} onOpen={(item) => setCalendarEntry({ kind: 'schedule', schedule: item.schedule, schedules: item.schedules })} />)}
                        {dayTides.map((tide) => (
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
              <caption className="sr-only">Organization schedules</caption>
              <thead><tr><th scope="col">Date</th><th scope="col">Activity</th><th scope="col">Planting area</th><th scope="col">Days between plant checks</th><th scope="col">Status</th><th scope="col">Activity guidance</th><th scope="col">Actions</th></tr></thead>
              <tbody>{schedules.map((schedule) => {
                const planting = isPlantingAppointment(schedule);
                const canAssign = planting && String(schedule.status).toLowerCase() === 'confirmed' && !schedule.project_site_id;
                return (
                  <tr key={schedule.id}>
                    <td><strong>{formatDate(`${schedule.scheduled_date}T12:00:00+08:00`)}</strong><small>{scheduleTimeLabel(schedule)}</small></td>
                    <td><strong>{schedule.title}</strong><small>{schedule.organization_name}</small><small>{formatCount(schedule.expected_participants)} participants</small><small className="schedule-entry-source">{schedule.source === 'website' ? 'LIKE website' : 'Staff-entered'}</small></td>
                    <td>{planting ? schedule.project_site_name || (canAssign ? 'Choose a planting area' : 'No planting area chosen') : 'Not needed'}</td>
                    <td>{planting ? schedule.inspection_interval_days ? `Every ${formatCount(schedule.inspection_interval_days)} days` : 'Not set' : 'Not applicable'}</td>
                    <td><span className={`schedule-status is-${String(schedule.status).toLowerCase()}`}>{scheduleStatusLabel(schedule.status)}</span></td>
                    <td>{planting ? <PlantingBadge assessment={assessmentFor(schedule)} /> : <strong>{appointmentTypeLabel(schedule.appointment_type)}</strong>}<small>{assessmentFor(schedule).reason}</small></td>
                    <td><div className="schedule-row-actions">
                      {canAssign ? <button type="button" className="is-primary" onClick={() => openAssignment(schedule)}>Choose area</button> : null}
                      <button type="button" onClick={() => openEdit(schedule)}>{schedule.planting_started ? 'Update notes / status' : 'Edit'}</button>
                      <button type="button" className="is-danger" onClick={() => deleteSchedule(schedule)} disabled={String(deletingId) === String(schedule.id) || schedule.planting_started || schedule.source === 'website' || schedule.status !== 'requested' || schedule.linked_assignment_count > 0} title={schedule.planting_started ? schedule.edit_lock_reason : schedule.status !== 'requested' ? 'Cancel confirmed activities to notify the organization and keep their history.' : schedule.source === 'website' ? 'Cancel this booking to notify the requester and preserve its history.' : schedule.linked_assignment_count > 0 ? 'Remove unplanted assignments before deleting this activity.' : undefined}>{String(deletingId) === String(schedule.id) ? 'Deleting...' : 'Delete'}</button>
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
        guide={calendarGuide} payload={tidePayload} now={now} onClose={() => setCalendarEntry(null)} onEdit={openEdit} />

      <HighTideCaution selection={timeCaution}
        onContinue={() => {
          acceptedUnsafeTime.current = selectionKey(timeCaution);
          setTimeCaution(null);
        }}
        onBack={() => setTimeCaution(null)} />

      {formOpen && createPortal(<Modal
        open={!timeCaution}
        title={editingSchedule ? 'Edit appointment' : 'Add appointment'}
        confirmLabel={editingSchedule ? 'Save changes' : 'Create schedule'}
        cancelLabel="Close"
        busy={saving}
        onConfirm={saveSchedule}
        onCancel={closeForm}
        className="modal-card-wide schedule-form-modal"
        variant="info"
      >
        <form className="schedule-form" noValidate onChangeCapture={feedback.onChange} onSubmit={saveSchedule}>
          <FormErrorSummary feedback={feedback} />
          <FormMessage error={formError} loading={saving ? 'Saving the activity…' : ''} onRetry={formError.startsWith('Could not verify whether') ? undefined : saveSchedule} busy={saving} />
          {editingSchedule?.planting_started ? <p role="status">Planting has been recorded for this activity. Planning details are locked. You can correct contact details and notes, or update progress.</p> : null}
          {editingSchedule?.source === 'website' ? <p>Agree on revised details with the requester before saving. Changing the date, time, or expected participants queues an email. Cancelling queues a cancellation notice.</p> : null}
          {editingSchedule?.source !== 'website' ? <p>Confirmations, schedule changes, and cancellations are emailed to the contact person.</p> : null}
          {formPlanting ? <div className="schedule-form-assessment">
            <PlantingBadge assessment={assessmentFor({ ...editingSchedule,
              start_at: `${form.date}T${form.startTime}+08:00`, end_at: `${form.date}T${form.endTime}+08:00`,
            })} />
            <p>{assessmentFor({ ...editingSchedule, start_at: `${form.date}T${form.startTime}+08:00`, end_at: `${form.date}T${form.endTime}+08:00` }).reason}</p>
          </div> : form.appointmentType ? <p>{appointmentTypeLabel(form.appointmentType)}: check staff availability and site conditions with LIKE.</p> : null}
          <div className="schedule-form-grid">
            <label>
              <span>Organization *</span>
              <input {...feedback.props('organizationName')}
                value={form.organizationName}
                list="schedule-organizations"
                maxLength="200"
                placeholder="Enter or select an organization"
                onChange={(event) => {
                  const value = event.target.value;
                  const matching = organizations.find((organization) => organization.name.trim().toLocaleLowerCase() === value.trim().toLocaleLowerCase());
                  setForm((current) => ({ ...current, organizationName: value, organizationId: matching ? String(matching.id) : '' }));
                }}
                disabled={Boolean(editingSchedule && editingSchedule.status !== 'requested')}
                required
              />
              <FieldError feedback={feedback} field="organizationName" />
              <datalist id="schedule-organizations">
                {organizations.map((organization) => <option key={organization.id} value={organization.name} />)}
              </datalist>
              {!organizations.length ? <small>Enter the organization name. After saving, you can add planting areas for it.</small> : null}
              {editingSchedule?.project_site_id ? <small>This activity already has a planting area, so its organization cannot be changed.</small> : null}
            </label>
            <label>
              <span>Appointment type *</span>
              <select disabled={Boolean(editingSchedule)} {...feedback.props('appointmentType')} value={form.appointmentType} onChange={(event) => {
                acceptedUnsafeTime.current = null;
                lastWarnedTime.current = null;
                setForm((current) => ({ ...current, appointmentType: event.target.value }));
              }} required>
                <option value="">Choose an appointment type</option>
                {APPOINTMENT_TYPES.map((type) => <option key={type.value} value={type.value}>{type.label}</option>)}
              </select>
              <FieldError feedback={feedback} field="appointmentType" />
            </label>
            <label>
              <span>Contact person{!editingSchedule ? ' *' : ''}</span>
              <input {...feedback.props('contactName')} value={form.contactName} maxLength={Math.max(100, originalContact.contactName.length)} autoComplete="name" readOnly={editingSchedule?.source === 'website'} required={!editingSchedule} onChange={(event) => setForm((current) => ({ ...current, contactName: event.target.value }))} />
              <FieldError feedback={feedback} field="contactName" />
            </label>
            <label>
              <span>Phone number{!editingSchedule ? ' *' : ''}</span>
              <input {...feedback.props('phone')} type="tel" value={form.phone} minLength={editingSchedule ? undefined : 7} maxLength={Math.max(32, originalContact.phone.length)} autoComplete="tel" readOnly={editingSchedule?.source === 'website'} required={!editingSchedule} onChange={(event) => setForm((current) => ({ ...current, phone: event.target.value }))} />
              <FieldError feedback={feedback} field="phone" />
            </label>
            <label>
              <span>Email{!editingSchedule ? ' *' : ''}</span>
              <input {...feedback.props('email')} type="email" maxLength={254} value={form.email} autoComplete="email" readOnly={editingSchedule?.source === 'website'} required={!editingSchedule} onChange={(event) => setForm((current) => ({ ...current, email: event.target.value }))} />
              <FieldError feedback={feedback} field="email" />
            </label>
            <label>
              <span>Date *</span>
              <input disabled={Boolean(editingSchedule?.planting_started)} {...feedback.props('date')} type="date" min={editingSchedule ? undefined : today} value={form.date} onChange={(event) => setForm((current) => ({ ...current, date: event.target.value }))} required />
              <FieldError feedback={feedback} field="date" />
            </label>
            <label>
              <span>Start time *</span>
              <input disabled={Boolean(editingSchedule?.planting_started)} {...feedback.props('startTime')} type="time" value={form.startTime} onChange={(event) => setForm((current) => ({ ...current, startTime: event.target.value }))} required />
              <FieldError feedback={feedback} field="startTime" />
            </label>
            <label>
              <span>End time *</span>
              <input disabled={Boolean(editingSchedule?.planting_started)} {...feedback.props('endTime')} type="time" value={form.endTime} onChange={(event) => setForm((current) => ({ ...current, endTime: event.target.value }))} required />
              <FieldError feedback={feedback} field="endTime" />
            </label>
            <label>
              <span>Participants{!editingSchedule ? ' *' : ''}</span>
              <input disabled={Boolean(editingSchedule?.planting_started)} {...feedback.props('expectedParticipants')} type="number" min={editingSchedule ? '0' : '1'} max={editingSchedule ? undefined : '10000'} required={!editingSchedule} step="1" value={form.expectedParticipants} onChange={(event) => setForm((current) => ({ ...current, expectedParticipants: event.target.value }))} />
              <FieldError feedback={feedback} field="expectedParticipants" />
            </label>
            {formPlanting ? <label>
              <span>Days between plant checks</span>
              <input type="number" value={editingSchedule?.inspection_interval_days ?? 14} readOnly aria-readonly="true" />
              <small>Monitoring is due every 14 days from the actual planting date.</small>
            </label> : null}
            <label>
              <span>Status *</span>
              <select {...feedback.props('status')} value={form.status} onChange={(event) => setForm((current) => ({ ...current, status: event.target.value }))} required>
                {!editingSchedule?.planting_started ? <option value="requested">Pending confirmation</option> : null}
                {!editingSchedule?.planting_started || editingSchedule.status === 'confirmed' ? <option value="confirmed">Confirmed</option> : null}
                {editingSchedule && !editingSchedule.planting_started ? <option value="tentative">Tentative</option> : null}
                {editingSchedule && editingSchedule.status !== 'completed' ? <option value="in_progress">In progress</option> : null}
                {editingSchedule && !['requested', 'confirmed', 'tentative', 'in_progress', 'completed', 'cancelled'].includes(editingSchedule.status) ? <option value={editingSchedule.status}>{scheduleStatusLabel(editingSchedule.status)}</option> : null}
                {editingSchedule ? <option value="completed">Completed</option> : null}
                {editingSchedule && !editingSchedule.planting_started ? <option value="cancelled">Cancelled</option> : null}
              </select>
              <FieldError feedback={feedback} field="status" />
            </label>
            <label className="schedule-form-notes">
              <span>Notes</span>
              <textarea {...feedback.props('notes')} rows="3" maxLength="2000" value={form.notes} onChange={(event) => setForm((current) => ({ ...current, notes: event.target.value }))} />
              <FieldError feedback={feedback} field="notes" />
            </label>
          </div>
          {form.date && formPlanting ? (
            <div className="schedule-form-tides">
              <span>Tide forecast for {formatDate(`${form.date}T12:00:00+08:00`)}</span>
              {selectedDateTides.length ? selectedDateTides.map((tide) => (
                <TideTime key={`${tide.occurred_at}-${tideKind(tide)}`} tide={tide} guide={calendarGuide} />
              )) : <small>High and low tide times are not available for this date yet.</small>}
            </div>
          ) : null}
          <p className="schedule-form-footnote">{formPlanting ? 'Choose a planting area after confirming the activity. ' : ''}All times are Philippine time.</p>
        </form>
      </Modal>, document.body)}

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
        <div className="schedule-assignment-form" onChangeCapture={assignmentFeedback.onChange}>
          <FormErrorSummary feedback={assignmentFeedback} />
          <p><strong>{assigningSchedule?.title}</strong> is confirmed for {assigningSchedule?.organization_name}.</p>
          {assignmentError ? <Message>{assignmentError}</Message> : null}
          <label>
            <span>Organization’s planting area *</span>
            <select {...assignmentFeedback.props('site')} required value={assignmentSiteId} onChange={(event) => setAssignmentSiteId(event.target.value)}>
              <option value="">Select a planting area</option>
              {assignmentSites.map((site) => <option key={site.id} value={site.id}>{site.name}</option>)}
            </select>
            <FieldError feedback={assignmentFeedback} field="site" />
          </label>
          {!assignmentSites.length ? <small>This organization has no planting areas yet. Add one in the Zone Editor first.</small> : null}
        </div>
      </Modal>
    </main>
  );
}
