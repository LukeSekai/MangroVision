import { useCallback, useEffect, useRef, useState } from 'react';
import { formRequest } from '../utils/formValidation';
import useTideForecasts from '../utils/useTideForecasts';
import { activityTideSummary, COASTAL_FORECAST_KEY, forecastIssue } from '../utils/plantingTides';
import { appointmentTypeLabel } from '../utils/appointmentTypes';
import { FormMessage } from './FormFeedback';
import './FieldActivities.css';

const API = import.meta.env.VITE_API_BASE || '';
const dateTime = (value) => new Intl.DateTimeFormat('en-PH', { timeZone: 'Asia/Manila', dateStyle: 'medium', timeStyle: 'short' }).format(new Date(value));
const time = (value) => new Intl.DateTimeFormat('en-PH', { timeZone: 'Asia/Manila', hour: 'numeric', minute: '2-digit' }).format(new Date(value));
const durationLabel = (minutes) => {
  const rounded = Math.max(1, Math.round(minutes));
  return [rounded >= 60 ? `${Math.floor(rounded / 60)} hr` : '', rounded % 60 ? `${rounded % 60} min` : ''].filter(Boolean).join(' ');
};

export function ActivityTides({ activity, record, now, onRetry }) {
  if (activity.status === 'cancelled') return <p className="field-activity-cancelled">This activity is cancelled. Contact staff before arranging another visit.</p>;
  if (Date.parse(activity.end_at) < now) return <p>This activity time has passed. Current forecasts do not assess past field conditions.</p>;
  if (!record || record.loading) return <p role="status">Loading tide advisory…</p>;
  const payload = record.payload;
  const summary = activityTideSummary({ payload, startAt: activity.start_at, endAt: activity.end_at, now });
  const issue = forecastIssue(payload, now);
  return <div className={`field-activity-tides is-${summary.status}`}>
    {record.error ? <FormMessage error={record.error} onRetry={onRetry} /> : null}
    <strong>{summary.status === 'unknown' ? 'Tide advisory unavailable'
      : summary.periods.length ? 'Elevated water overlaps this activity' : 'No elevated-water overlap forecast'}</strong>
    {summary.status === 'unknown' ? <p>{summary.reason} Check again closer to the activity if it is beyond the forecast dates.</p> : <>
      <p>High tide peaks on this date: {summary.peaks.length ? summary.peaks.map((peak) => time(peak.at)).join(', ') : 'No peak time is available in this forecast.'}</p>
      {summary.periods.length ? <ul>{summary.periods.map((period) => <li key={period.start}>
        Estimated elevated-water period: {dateTime(period.start)} – {dateTime(period.end)}.
        {' '}{period.incomplete ? 'At least ' : 'About '}{durationLabel(period.duration_minutes)}{period.incomplete ? '; the period continues beyond the available readings' : ''}.
      </li>)}</ul> : <p>Water is predicted to remain at or below the estimated reference during this activity.</p>}
      <small>Uses the same coastal forecast and estimated reference as staff Scheduling ({summary.threshold_m.toFixed(2)} m above MSL). This is not a measured ground level. Confirm site conditions with staff.</small>
    </>}
    <p className="field-forecast-freshness">{payload?.fetched_at ? `Forecast fetched ${dateTime(payload.fetched_at * 1000)} · ${issue ? 'Needs refresh' : 'Current'}` : 'No current forecast received.'}</p>
    {payload?.source ? <small>Source: {payload.source}</small> : null}
    {payload?.attribution_url && /^https:\/\//.test(payload.attribution_url) ? <a href={payload.attribution_url} target="_blank" rel="noreferrer">Forecast attribution</a> : null}
    {issue && !record.error ? <button type="button" onClick={onRetry}>Refresh tide forecast</button> : null}
  </div>;
}

export default function FieldActivities({ organizationId }) {
  const [activities, setActivities] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [loadedAt, setLoadedAt] = useState(null);
  const [refreshKey, setRefreshKey] = useState(0);
  const [tideRetryKey, setTideRetryKey] = useState(0);
  const [now, setNow] = useState(Date.now);
  const attemptedForecast = useRef(null);
  const forecasts = useTideForecasts(activities.some((activity) => activity.status !== 'cancelled' && Date.parse(activity.end_at) >= now) ? [COASTAL_FORECAST_KEY] : [], tideRetryKey);
  const retryTides = useCallback(() => setTideRetryKey((key) => key + 1), []);
  useEffect(() => {
    const controller = new AbortController();
    let inFlight = false;
    const load = async () => {
      if (inFlight || controller.signal.aborted) return;
      inFlight = true;
      try {
        const result = await formRequest(`${API}/api/planter-auth/me/activities`, { signal: controller.signal, cache: 'no-store' }, 'Could not load your organization’s activities.');
        if (!controller.signal.aborted) { setActivities(result.activities || []); setError(''); setLoadedAt(Date.now()); }
      } catch (issue) {
        if (!controller.signal.aborted) setError(issue.message);
      } finally { inFlight = false; if (!controller.signal.aborted) setLoading(false); }
    };
    void load();
    const timer = window.setInterval(() => { setNow(Date.now()); void load(); }, 60000);
    const focus = () => { setNow(Date.now()); void load(); };
    window.addEventListener('focus', focus);
    return () => { controller.abort(); window.clearInterval(timer); window.removeEventListener('focus', focus); };
  }, [organizationId, refreshKey]);
  // Refresh an expired forecast through the shared six-hour backend cache.
  const payload = forecasts[COASTAL_FORECAST_KEY]?.payload;
  useEffect(() => {
    if (!payload || payload.stale || now - payload.fetched_at * 1000 < 6 * 3600000 || attemptedForecast.current === payload.fetched_at) return;
    attemptedForecast.current = payload.fetched_at;
    const timer = window.setTimeout(retryTides, 0);
    return () => window.clearTimeout(timer);
  }, [now, payload, retryTides]); // Forecast retries remain explicit after a provider failure.
  const changed = activities.some((activity) => activity.change_notice || activity.status === 'cancelled');
  return <details className="field-activities" open={changed || undefined}>
    <summary>Your activities <span>{loading ? 'Loading…' : `${activities.length} activities`}{changed ? ' · Schedule updates' : ''}</span></summary>
    <div className="field-activities-content">
      <div className="field-activities-toolbar"><small>Philippine time · {loadedAt ? `Checked ${time(loadedAt)}` : 'Checking schedules'}</small>
        <button type="button" onClick={() => { setLoading(true); setRefreshKey((key) => key + 1); retryTides(); }} disabled={loading}>Refresh activities</button></div>
      <FormMessage error={error ? `${error}${activities.length ? ' The details below could be out of date.' : ''}` : ''} onRetry={() => { setLoading(true); setRefreshKey((key) => key + 1); }} busy={loading} />
      {loading && !loadedAt ? <p role="status">Loading your activity details…</p> : !activities.length && !error ? <p>No confirmed activities yet. Staff will confirm your organization’s schedule.</p> : null}
      {activities.map((activity) => <article className="field-activity-card" key={activity.id}>
        <h3>{activity.title}</h3><span className={`field-activity-status is-${activity.status}`}>{activity.status.replaceAll('_', ' ')}</span>
        <p><strong>{dateTime(activity.start_at)} – {dateTime(activity.end_at)}</strong></p>
        <p>{appointmentTypeLabel(activity.appointment_type)}{activity.project_site_name ? ` · ${activity.project_site_name}` : ''}{activity.expected_planters != null ? ` · ${activity.expected_planters} participants` : ''}</p>
        {activity.change_notice ? <p className="field-activity-change" role="status"><strong>{activity.change_notice.rescheduled ? 'Rescheduled' : 'Activity updated'}</strong> · {dateTime(activity.change_notice.changed_at)}.
          {activity.change_notice.rescheduled && activity.change_notice.previous_start_at ? ` Previous start: ${dateTime(activity.change_notice.previous_start_at)}. Use the current schedule above.` : ' Use the current details above.'}</p> : null}
        <ActivityTides activity={activity} record={forecasts[COASTAL_FORECAST_KEY]} now={now} onRetry={retryTides} />
      </article>)}
    </div>
  </details>;
}
