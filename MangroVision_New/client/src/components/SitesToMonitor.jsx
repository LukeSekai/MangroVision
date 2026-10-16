import { useEffect, useMemo, useRef, useState } from 'react';
import { manilaDateKey, monitoringSchedule, monitoringScheduleLabel } from '../utils/monitoringSchedule';
import { projectSiteAreaM2, projectSiteId } from '../utils/projectSiteInfo';
import SiteMapPreview from './SiteMapPreview';
import MonitoringSiteDetails, { SiteHistoryButton } from './MonitoringSiteDetails';
import { groupSitePreviewPoints } from '../utils/siteMapPreview';
import './SitesToMonitor.css';

const API = import.meta.env.VITE_API_BASE || '';
const PAGE_SIZE = 5;
const count = (value) => value == null ? '—' : Number(value).toLocaleString();
function dateLabel(value) {
  if (!value) return 'Not scheduled';
  const day = /^\d{4}-\d{2}-\d{2}$/.test(value) ? value : manilaDateKey(value);
  if (!day) return 'Date unavailable';
  return new Date(`${day}T12:00:00+08:00`).toLocaleDateString('en-PH', {
    timeZone: 'Asia/Manila', month: 'short', day: 'numeric', year: 'numeric',
  });
}

export default function SitesToMonitor({ organizations, requestedSiteId, token, today, onRefresh }) {
  const [sites, setSites] = useState([]);
  const [points, setPoints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [hasLoaded, setHasLoaded] = useState(false);
  const [error, setError] = useState('');
  const [pointsError, setPointsError] = useState('');
  const [revision, setRevision] = useState(0);
  const [page, setPage] = useState(1);
  const [selectedSiteId, setSelectedSiteId] = useState('');
  const [siteView, setSiteView] = useState('list');
  const loadedSnapshot = useRef(null);
  const section = useRef(null);

  useEffect(() => {
    if (!token) return undefined;
    // Retained routes pause and resume effects. A completed snapshot survives
    // that lifecycle; only Refresh or a data-change revision starts a new read.
    if (loadedSnapshot.current?.token === token && loadedSnapshot.current?.revision === revision) return undefined;
    const controller = new AbortController();
    async function read(path) {
      const response = await fetch(`${API}${path}`, { signal: controller.signal, cache: 'no-store' });
      const data = await response.json();
      if (!response.ok) throw new Error('Could not load monitoring sites.');
      return data;
    }
    async function load() {
      setLoading(true);
      const [siteResult, pointResult] = await Promise.allSettled([
        read('/api/project-sites'), read('/api/planters/map-points'),
      ]);
      if (controller.signal.aborted) return;
      if (siteResult.status === 'fulfilled' && Array.isArray(siteResult.value.features)) {
        setSites(siteResult.value.features);
        setError('');
      } else setError('Could not load monitoring sites. Try again.');
      if (pointResult.status === 'fulfilled' && Array.isArray(pointResult.value)) {
        setPoints(pointResult.value);
        setPointsError('');
      } else {
        setPoints([]);
        setPointsError('Could not load planting points. Select Refresh to try again.');
      }
      loadedSnapshot.current = { token, revision };
      setHasLoaded(true);
      setLoading(false);
    }
    void load();
    return () => controller.abort();
  }, [token, revision]);

  useEffect(() => {
    const refresh = () => setRevision((value) => value + 1);
    window.addEventListener('mv:data-changed', refresh);
    return () => window.removeEventListener('mv:data-changed', refresh);
  }, []);

  const pointsBySite = useMemo(() => groupSitePreviewPoints(points), [points]);

  const owners = new Map(organizations.map((organization) => [String(organization.id), organization]));
  const visible = requestedSiteId ? sites.filter((site) => String(projectSiteId(site)) === String(requestedSiteId)) : sites;
  const pageCount = Math.max(1, Math.ceil(visible.length / PAGE_SIZE));
  const currentPage = Math.min(page, pageCount);
  if (page !== currentPage) setPage(currentPage);
  const start = (currentPage - 1) * PAGE_SIZE;
  const pageSites = visible.slice(start, start + PAGE_SIZE);
  const selectedSite = visible.find((site) => String(projectSiteId(site)) === selectedSiteId);
  const detailsVisible = Boolean(selectedSite && siteView !== 'list');
  const initialLoading = loading && !hasLoaded;
  const openSite = (id, view) => {
    setSelectedSiteId(String(id));
    setSiteView(view);
  };
  const closeSite = () => {
    setSiteView('list');
    window.requestAnimationFrame(() => {
      const row = [...(section.current?.querySelectorAll('.monitoring-site-row') || [])]
        .find((element) => element.dataset.siteId === selectedSiteId);
      row?.querySelector('.monitoring-site-open')?.focus({ preventScroll: true });
    });
  };
  const refresh = () => {
    setRevision((value) => value + 1);
    void onRefresh?.();
  };
  const retry = <button type="button" className="monitoring-sites-retry" disabled={loading} onClick={refresh}>Try again</button>;

  return <section ref={section} className="monitoring-sites-section" data-guide-card="sites-to-monitor" aria-labelledby="monitoring-sites-title" aria-busy={loading}>
    <header className="monitoring-sites-header">
      <div><h2 id="monitoring-sites-title">Sites to Monitor <span>{initialLoading ? '—' : visible.length}</span></h2>
        <p>Browse planting areas and mapped points. Counts and monitoring dates cover each site’s organization.</p></div>
      <button type="button" className="monitoring-sites-retry" disabled={loading || !token} onClick={refresh}>Refresh</button>
    </header>
    {error ? <p className="org-monitoring-message is-error" role="alert">{error} {retry}</p> : null}
    {pointsError && !error ? <p className="org-monitoring-message is-error" role="alert">{pointsError}</p> : null}
    {loading && hasLoaded ? <p className="org-monitoring-muted" role="status">Refreshing sites…</p> : null}
    <div className="monitoring-sites-list">
      {initialLoading ? <p className="monitoring-sites-empty" role="status">Loading monitoring sites…</p>
        : error && !sites.length ? null : !visible.length ? <div className="monitoring-sites-empty"><strong>No sites to monitor yet</strong><p>Planting areas will appear here once a project site is created.</p></div>
          : pageSites.map((site) => {
            const id = projectSiteId(site);
            const props = site.properties || {};
            const name = props.name || `Site ${id}`;
            const owner = owners.get(String(props.organization_id));
            const schedule = monitoringSchedule(owner, today);
            const area = projectSiteAreaM2(site.geometry);
            const latest = owner?.latest_record;
            return <article key={id} className="monitoring-site-row" data-site-id={id}>
              <button type="button" className="monitoring-site-open" aria-label={`View site ${name}`}
                aria-haspopup="dialog" aria-controls="monitoring-site-details" onClick={() => openSite(id, 'details')} />
              <SiteMapPreview key={`${revision}:${id}:${JSON.stringify(site.geometry)}`} site={site} name={name} points={pointsBySite.get(String(id)) || []} />
              <div className="monitoring-site-main">
                <h3>{name}</h3>
                {props.big_wave_risk?.active === true && <p className="monitoring-site-wave-warning" role="note">
                  <strong>Danger: This area is prone to big waves.</strong>
                  <span>{count(props.big_wave_risk.wave_death_points)} of {count(props.big_wave_risk.planted_points)} planted locations ({props.big_wave_risk.percentage}%) have recorded big-wave deaths.</span>
                </p>}
                <p className="monitoring-site-owner">{props.organization_name || props.organization || owner?.name || 'No organization assigned'}</p>
                <p className="monitoring-site-meta">
                  {area != null ? <span>{area.toLocaleString('en-PH', { maximumFractionDigits: 1 })} m² area</span> : null}
                  {props.point_count != null ? <span>{count(props.point_count)} mapped points</span> : null}
                  <span>{latest ? `Last visit ${dateLabel(latest.monitored_at)}` : 'No visits recorded'}</span>
                </p>
                <dl className="monitoring-site-metrics">
                  <div><dt>Planted</dt><dd>{count(owner?.total_planted ?? owner?.current_planted_points)}</dd></div>
                  <div><dt>Alive now</dt><dd>{count(owner?.alive_seedlings)}</dd></div>
                  <div><dt>Survival</dt><dd>{latest?.survival_rate_pct == null ? '—' : `${latest.survival_rate_pct}%`}</dd></div>
                  <div><dt>Next visit</dt><dd>{dateLabel(owner?.next_monitoring_date)}</dd></div>
                </dl>
              </div>
              <div className="monitoring-site-actions">
                <span className={`monitoring-schedule-status is-${schedule.status}`}>{monitoringScheduleLabel(schedule)}</span>
                <span className="monitoring-site-view-hint" aria-hidden="true">View site <span>›</span></span>
                <SiteHistoryButton name={name} disabled={!owner} onClick={() => openSite(id, 'history')} />
              </div>
            </article>;
          })}
    </div>
    {!initialLoading && visible.length > 0 ? <nav className="monitoring-sites-pagination" aria-label="Sites to Monitor pages">
      <p role="status" aria-live="polite">Showing {start + 1}–{Math.min(start + PAGE_SIZE, visible.length)} of {visible.length} sites</p>
      <div><button type="button" onClick={() => setPage(currentPage - 1)} disabled={currentPage === 1}>Previous</button>
        <span>Page {currentPage} of {pageCount}</span>
        <button type="button" onClick={() => setPage(currentPage + 1)} disabled={currentPage === pageCount}>Next</button></div>
    </nav> : null}
    {selectedSite ? <MonitoringSiteDetails key={selectedSiteId} site={selectedSite}
      organization={owners.get(String(selectedSite.properties?.organization_id))}
      points={pointsBySite.get(selectedSiteId) || []} pointsAvailable={!pointsError}
      today={today} view={siteView} active={detailsVisible} refreshKey={revision}
      onViewChange={setSiteView} onClose={closeSite} onChanged={onRefresh} /> : null}
  </section>;
}
