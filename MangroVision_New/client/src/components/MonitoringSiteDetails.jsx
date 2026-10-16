import { useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import Modal from './Modal';
import SiteMapPreview from './SiteMapPreview';
import OrganizationHistory from './OrganizationHistory';
import { countMapPointStatuses } from '../utils/mapPointStats';
import { POINT_STATUS_COLORS } from '../utils/pointStatus';
import { projectSiteAreaM2, projectSiteId } from '../utils/projectSiteInfo';
import { monitoringSchedule, monitoringScheduleLabel } from '../utils/monitoringSchedule';

const count = (value) => value == null ? '—' : Number(value).toLocaleString();
function date(value) {
  if (!value) return 'Not recorded';
  return new Date(/^\d{4}-\d{2}-\d{2}$/.test(value) ? `${value}T12:00:00+08:00` : value).toLocaleDateString('en-PH', {
    timeZone: 'Asia/Manila', month: 'short', day: 'numeric', year: 'numeric',
  });
}
const STATUSES = [['planned', 'Planned'], ['assigned', 'Assigned'], ['planted', 'Planted'],
  ['dead', 'Dead'], ['skipped', 'Skipped'], ['unavailable', 'Unavailable']];

export function SiteHistoryButton({ name, onClick, active, disabled = false }) {
  const label = active ? `Show site details for ${name}` : `View visit history for ${name}`;
  return <button type="button" className="monitoring-site-history-button" onClick={onClick}
    aria-label={label} title={label} aria-pressed={active} disabled={disabled}>
    <svg width="19" height="19" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true">
      <circle cx="12" cy="12" r="9" /><path d="M12 7v5l3 2" />
    </svg>
  </button>;
}

export default function MonitoringSiteDetails({ site, organization, points, pointsAvailable, today, view,
  active, refreshKey, onViewChange, onClose, onChanged }) {
  const props = site.properties || {};
  const name = props.name || `Site ${projectSiteId(site)}`;
  const owner = props.organization_name || props.organization || organization?.name || 'No organization assigned';
  const stats = useMemo(() => countMapPointStatuses(points), [points]);
  const area = projectSiteAreaM2(site.geometry);
  const schedule = monitoringSchedule(organization, today);
  const latest = organization?.latest_record;
  const heading = useRef(null);
  const focusedView = useRef(null);
  const [editingLocations, setEditingLocations] = useState(false);

  useEffect(() => {
    if (!active) { focusedView.current = null; return; }
    if (focusedView.current === view) return;
    focusedView.current = view;
    const body = heading.current?.closest('.modal-body');
    if (body) body.scrollTop = 0;
    heading.current?.focus({ preventScroll: true });
  }, [active, view]);

  return createPortal(<Modal open={active && !editingLocations} keepMounted
    title={view === 'history' ? 'Monitoring history' : 'Site details'} variant="info"
    cancelLabel="Close" onCancel={onClose}
    className="modal-card-wide org-monitoring-visit-modal monitoring-site-modal">
    <div id="monitoring-site-details" className="monitoring-site-details">
    <header className="monitoring-site-detail-header org-monitoring-form-organization">
      <div>
        <h3 ref={heading} tabIndex={-1}>{name}</h3>
        <p className="monitoring-site-owner">{owner}</p>
      </div>
      <div className="monitoring-site-detail-actions">
        <span className={`monitoring-schedule-status is-${schedule.status}`}>{monitoringScheduleLabel(schedule)}</span>
        <SiteHistoryButton name={name} active={view === 'history'} disabled={!organization}
          onClick={() => onViewChange(view === 'history' ? 'details' : 'history')} />
      </div>
    </header>

    <div className="monitoring-site-detail-content" hidden={view !== 'details'} inert={view !== 'details'}>
      {props.big_wave_risk?.active === true && <div className="monitoring-site-wave-warning" role="note">
        <strong>Danger: This area is prone to big waves.</strong>
        <span>{count(props.big_wave_risk.wave_death_points)} of {count(props.big_wave_risk.planted_points)} planted locations ({props.big_wave_risk.percentage}%) have recorded big-wave deaths. This uses site history, including past plantings.</span>
      </div>}
      <section className="monitoring-site-detail-map" aria-label={`Location of ${name}`}>
        <div className="monitoring-site-detail-map-heading"><h4>Site map</h4>
          <span>{pointsAvailable ? `${count(stats.mapped)} mapped points` : 'Planting points unavailable'}</span></div>
        <SiteMapPreview key={refreshKey} site={site} name={name} points={points} large />
        <p className="monitoring-site-detail-map-caption">The outlined area and planting points show this site’s location.</p>
        {props.centroid_lat != null && props.centroid_lon != null ? <p className="monitoring-site-coordinates">
          {Number(props.centroid_lat).toFixed(6)}° N, {Number(props.centroid_lon).toFixed(6)}° E
        </p> : null}
      </section>
      <div className="monitoring-site-detail-info">
        <section aria-labelledby="monitoring-site-point-title">
          <h4 id="monitoring-site-point-title">Site information</h4>
          <dl className="monitoring-site-information">
            <div><dt>Area</dt><dd>{area == null ? '—' : `${area.toLocaleString('en-PH', { maximumFractionDigits: 1 })} m²`}</dd></div>
            <div><dt>Organization</dt><dd>{owner}</dd></div>
            {props.created_at ? <div><dt>Created</dt><dd>{date(props.created_at)}</dd></div> : null}
          </dl>
          <div className="monitoring-site-point-counts">
            {STATUSES.map(([status, label]) => <div key={status}>
              <span><i style={{ background: POINT_STATUS_COLORS[status] }} aria-hidden="true" />{label}</span>
              <strong>{count(pointsAvailable ? stats[status] : null)}</strong>
            </div>)}
          </div>
          {props.notes ? <p className="monitoring-site-notes">{props.notes}</p> : null}
        </section>
        <section className="monitoring-site-organization-summary" aria-labelledby="monitoring-site-visit-title">
          <h4 id="monitoring-site-visit-title">Organization monitoring</h4>
          <p>Visit totals cover all planting sites managed by {owner}.</p>
          <dl className="monitoring-site-metrics">
            <div><dt>Seedlings planted</dt><dd>{count(organization?.total_planted ?? organization?.current_planted_points)}</dd></div>
            <div><dt>Alive now</dt><dd>{count(organization?.alive_seedlings)}</dd></div>
            <div><dt>Survival</dt><dd>{latest?.survival_rate_pct == null ? '—' : `${latest.survival_rate_pct}%`}</dd></div>
            <div><dt>Recorded visits</dt><dd>{count(organization?.monitoring_record_count)}</dd></div>
            <div><dt>Last visit</dt><dd>{date(latest?.monitored_at)}</dd></div>
            <div><dt>Next visit</dt><dd>{organization?.next_monitoring_date ? date(organization.next_monitoring_date) : 'Not scheduled'}</dd></div>
          </dl>
          {latest?.actions_taken ? <p className="monitoring-site-notes"><strong>Latest LGU actions</strong>{latest.actions_taken}</p> : null}
        </section>
      </div>
    </div>

    <section className="monitoring-site-history" hidden={view !== 'history'} inert={view !== 'history'} aria-label={`Visit history for ${name}`}>
      <div className="monitoring-site-history-intro"><h4>Monitoring visits</h4>
        <p>Visits are recorded for {owner} across its planting sites.</p></div>
      {organization ? <OrganizationHistory key={organization.id} organization={organization}
        active={active && view === 'history'} refreshKey={refreshKey} onChanged={onChanged} onEditingChange={setEditingLocations} />
        : <p className="org-monitoring-muted">This site has no organization assigned to its monitoring visits.</p>}
    </section>
    </div>
  </Modal>, document.body);
}
