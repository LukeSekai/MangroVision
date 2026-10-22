import { lazy, Suspense, useCallback, useEffect, useRef, useState } from 'react';
import { Link } from 'react-router-dom';
import MonitoringHeader from '../components/MonitoringHeader';
import MapLink from '../components/MapLink';
import Modal from '../components/Modal';
import { FormMessage } from '../components/FormFeedback';
import { useMapStore } from '../stores/mapStore';
import useMapNavigation from '../utils/useMapNavigation';
import { manilaDay } from '../utils/restorationReports';
import './OrganizationMonitoring.css';
import './LGUSites.css';

const API = import.meta.env.VITE_API_BASE || '';
const SeedlingLocations = lazy(() => import('../components/SeedlingLocations'));
const count = (value) => value == null ? '—' : Number(value).toLocaleString('en-PH');
const rate = (value) => value == null ? 'Not available yet' : `${value}%`;
const date = (value) => value ? new Date(value).toLocaleDateString('en-PH', { timeZone: 'Asia/Manila' }) : 'Not recorded';
const CAUSES = [['unknown', 'Not determined'], ['barnacles', 'Barnacles'], ['waves', 'Big waves'], ['disease', 'Disease'],
  ['animal_damage', 'Animal damage'], ['storm', 'Storm'], ['drying_out', 'Drying out'], ['vandalism', 'Vandalism'], ['other', 'Other']];

function siteDueStatus(site, today = manilaDay()) {
  if (!site.next_monitoring_date) return site.total_planted ? 'No living seedlings to monitor' : 'Awaiting planting';
  return site.next_monitoring_date < today ? 'Overdue' : site.next_monitoring_date === today ? 'Due today' : 'Upcoming';
}

async function read(path, options = {}) {
  const response = await fetch(`${API}${path}`, { cache: 'no-store', ...options });
  const payload = await response.json();
  if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : 'Could not load LGU Sites. Please try again.');
  return payload;
}

export default function LGUSites() {
  const [data, setData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [showCode, setShowCode] = useState(false);
  const [filter, setFilter] = useState('all');
  const [active, setActive] = useState(null);
  const [form, setForm] = useState(null);
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState('');
  const [visitReview, setVisitReview] = useState(null);
  const requestGeneration = useRef(0);
  const contextGeneration = useRef(0);
  const activeRef = useRef(null);
  const busyRef = useRef(false);
  const navigateMap = useMapNavigation();

  const invalidate = useCallback(() => { ++requestGeneration.current; ++contextGeneration.current; }, []);
  const load = useCallback(async () => {
    const generation = ++requestGeneration.current;
    setLoading(true);
    try {
      const payload = await read('/api/monitoring/lgu-sites');
      if (generation === requestGeneration.current) { setData(payload); setError(''); }
    } catch (failure) { if (generation === requestGeneration.current) setError(failure.message); }
    finally { if (generation === requestGeneration.current) setLoading(false); }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => { void load(); }, 0);
    const refresh = () => { if (!activeRef.current && !busyRef.current) void load(); };
    window.addEventListener('mv:data-changed', refresh);
    return () => { window.clearTimeout(timer); invalidate();
      window.removeEventListener('mv:data-changed', refresh); };
  }, [load, invalidate]);

  function close() {
    if (busyRef.current) return;
    setVisitReview(null);
    ++contextGeneration.current; activeRef.current = null; setActive(null); setFormError('');
  }

  async function open(site, view, day = manilaDay()) {
    setVisitReview(null);
    const generation = ++contextGeneration.current;
    activeRef.current = { id: site.project_site_id, view };
    setActive({ site, view, loading: true }); setFormError('');
    if (view === 'record') setForm({ monitored_at: day, dead_planting_event_ids: [], health_status: 'good',
      actions_taken: '', average_height_cm: '', death_reason_category: 'unknown', death_notes: '' });
    try {
      const context = await read(`/api/monitoring/lgu-sites/${site.project_site_id}/monitoring?monitored_at=${day}`);
      if (generation === contextGeneration.current) setActive({ site: context, view, loading: false });
    } catch (failure) { if (generation === contextGeneration.current) { setFormError(failure.message); setActive({ site, view, loading: false, failed: true }); } }
  }

  async function save(event) {
    event.preventDefault();
    if (busyRef.current || active.loading || active.failed) return;
    setVisitReview({ siteId: active.site.project_site_id, siteName: active.site.site_name,
      payload: { ...form, dead_planting_event_ids: [...form.dead_planting_event_ids], version: active.site.version,
        average_height_cm: form.average_height_cm === '' ? null : Number(form.average_height_cm) } });
  }

  async function confirmSave() {
    if (!visitReview || busyRef.current) return;
    const reviewed = visitReview;
    busyRef.current = true; setBusy(true); setFormError('');
    try {
      await read(`/api/monitoring/lgu-sites/${reviewed.siteId}/monitoring`, { method: 'POST',
        headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(reviewed.payload) });
      setNotice(`Monitoring saved for ${reviewed.siteName}. Other sites keep their own schedules.`);
      activeRef.current = null; setActive(null);
      await load();
      const map = useMapStore.getState();
      void map.fetchPoints({ force: true }); void map.fetchStats({ force: true }); void map.fetchZones({ force: true });
    } catch (failure) { setFormError(failure.message); }
    finally { busyRef.current = false; setBusy(false); setVisitReview(null); }
  }

  function viewMap(site) {
    const map = useMapStore.getState();
    map.setMonitoringOrganizationId(data.account.organization_id);
    const coordinates = site.geometry?.type === 'Polygon' ? site.geometry.coordinates[0] : site.geometry?.coordinates?.[0]?.[0];
    if (coordinates?.length) {
      const center = coordinates.reduce((sum, point) => [sum[0] + point[0], sum[1] + point[1]], [0, 0]).map((value) => value / coordinates.length);
      map.setView(center, 21);
    }
    navigateMap('/monitoring/map');
  }

  const sites = (data?.sites || []).filter((site) => filter === 'all' || (filter === 'due'
    ? ['Due today', 'Overdue'].includes(siteDueStatus(site)) : siteDueStatus(site) === 'Upcoming'));
  const summary = loading || error ? {} : data?.summary || {};
  const account = data?.account;
  return <main className="org-monitoring-page org-monitoring-panel lgu-sites-page"><div className="org-monitoring-content">
    <MonitoringHeader title="LGU Sites" description="One LGU organization, with planting and monitoring recorded separately for each site.">
      {account && <MapLink to={`/planters?section=assign&organization_id=${account.organization_id}`}>Assign LGU points</MapLink>}
    </MonitoringHeader>
    <div className="org-monitoring-kpis" aria-label="Combined LGU results">
      <article><span>LGU sites</span><strong>{count(summary.site_count)}</strong><small>Separate planting areas</small></article>
      <article><span>Seedlings planted</span><strong>{count(summary.total_planted)}</strong><small>All LGU sites combined</small></article>
      <article><span>Recorded deaths</span><strong>{count(summary.dead_count)}</strong><small>Retained after site release</small></article>
      <article><span>LGU recorded survival</span><strong>{rate(summary.survival_rate_pct)}</strong><small>Calculated from combined seedling counts</small></article>
    </div>
    <FormMessage loading={loading ? 'Loading LGU sites…' : ''} error={error} success={notice} onRetry={load} busy={loading} />
    {account && <section className="lgu-sites-access" data-guide-card="lgu-access" aria-label="Permanent LGU planter credentials">
      <div className="lgu-sites-account-heading"><h2>LGU planter account</h2><span className="lgu-sites-permanent">Permanent</span></div>
      <dl><div><dt>Username</dt><dd>{account.username}</dd></div><div><dt>Organization code</dt><dd><code>{showCode ? account.access_code : '••••-••••'}</code></dd></div></dl>
      <div className="lgu-sites-buttons"><button type="button" aria-label={showCode ? 'Hide organization code' : 'Show organization code'} aria-pressed={showCode} onClick={() => setShowCode((value) => !value)}>{showCode ? 'Hide code' : 'Show code'}</button><Link to="/field">Open planter app</Link></div>
    </section>}
    <section data-guide-card="lgu-sites" className="lgu-sites-workspace">
      <div className="lgu-sites-heading"><div><h2>Planting sites</h2><p>Visits follow each site’s planting dates at 14-day intervals. Weekend checks move to Monday.</p></div>
        <label>Show sites<select value={filter} onChange={(event) => setFilter(event.target.value)}><option value="all">All sites</option><option value="due">Due today or overdue</option><option value="upcoming">Upcoming monitoring</option></select></label></div>
      {!loading && !error && !sites.length && <p className="monitoring-visit-empty">{data?.sites?.length ? 'No sites match this selection.' : 'No LGU sites yet. Assign available planting points to LGU to form its project site.'}</p>}
      <div className="lgu-sites-grid">{sites.map((site) => <article className="lgu-site-card" key={site.project_site_id}>
        <header><h3>{site.site_name}</h3><span className={`lgu-site-status ${site.monitoring_available ? 'is-due' : ''}`}>{site.area_released_at ? 'Area released' : siteDueStatus(site)}</span></header>
        <dl><div><dt>Seedlings planted</dt><dd>{count(site.total_planted)}</dd></div><div><dt>Recorded deaths</dt><dd>{count(site.dead_count)}</dd></div>
          <div><dt>Recorded survival</dt><dd>{rate(site.survival_rate_pct)}</dd></div><div><dt>First planting</dt><dd>{date(site.first_planted_at)}</dd></div>
          <div><dt>Last monitoring</dt><dd>{date(site.latest_record?.monitored_at)}</dd></div><div><dt>Next monitoring</dt><dd>{site.next_monitoring_date ? date(site.next_monitoring_date) : 'Not scheduled'}</dd></div></dl>
        <div className="lgu-sites-buttons"><button type="button" className="is-primary" disabled={!site.monitoring_available} onClick={() => void open(site, 'record')}>Record monitoring</button>
          <button type="button" onClick={() => void open(site, 'history')}>View history</button><button type="button" disabled={!site.geometry} onClick={() => viewMap(site)}>Show on map</button>
</div>
      </article>)}</div>
    </section>
    {active && <Modal key={visitReview ? 'site-review' : 'site-form'} open title={visitReview ? 'Save site monitoring?' : `${active.view === 'record' ? 'Record monitoring' : 'Monitoring history'} · ${active.site.site_name}`}
      variant={visitReview ? 'warning' : 'default'}
      className={visitReview ? '' : 'modal-card-wide'} busy={busy} onCancel={() => { if (!busy) { if (visitReview) setVisitReview(null); else close(); } }}
      cancelLabel={visitReview ? 'Back' : 'Close'} initialFocus={visitReview ? 'cancel' : 'confirm'}
      confirmLabel="Save monitoring" onConfirm={visitReview ? confirmSave : undefined}>
      {visitReview && <div className="monitoring-save-review">
        <p><strong>{visitReview.siteName}</strong> · {visitReview.payload.monitored_at}</p>
        <p>{visitReview.payload.dead_planting_event_ids.length} newly dead · Overall health: {visitReview.payload.health_status}</p>
        {visitReview.payload.dead_planting_event_ids.length > 0 && <p>Cause: {CAUSES.find(([value]) => value === visitReview.payload.death_reason_category)?.[1]}</p>}
        <p>Save this visit for this LGU site?</p>
      </div>}
      {active.loading ? <p role="status">Loading site monitoring…</p> : active.view === 'history' ? <div className="lgu-site-history">
        {!active.site.history?.length && <p>No monitoring visits recorded for this site.</p>}
        {active.site.history?.map((visit) => <article key={visit.id}><h3>{date(visit.monitored_at)}</h3><p>{count(visit.alive_count)} without recorded death · {count(visit.dead_count)} recorded dead · {count(visit.new_dead_count)} newly dead</p>
          <p>Overall health: {visit.health_status}{visit.average_height_cm != null ? ` · Measured height: ${visit.average_height_cm} cm` : ''}</p><p>{visit.actions_taken}</p><small>Recorded by {visit.inspector_name}</small></article>)}
      </div> : visitReview ? null : <form onSubmit={save} className="lgu-site-form">
        <p>This visit covers <strong>{active.site.site_name}</strong>. Select every newly dead seedling on its map; enter zero deaths by leaving the selection empty.</p>
        <fieldset disabled={busy || active.failed}><label>Monitoring date<input type="date" required max={manilaDay()} value={form.monitored_at} onChange={(event) => void open(active.site, 'record', event.target.value)} /></label>
          <p>{count(active.site.alive_count)} seedlings without recorded death before this visit · {form.dead_planting_event_ids.length} newly dead selected</p>
          <Suspense fallback={<p role="status">Loading planting map…</p>}><SeedlingLocations providedPoints={active.site.points || []} selected={form.dead_planting_event_ids} onChange={(ids) => setForm({ ...form, dead_planting_event_ids: ids })} disabled={busy} /></Suspense>
          {form.dead_planting_event_ids.length > 0 && <><label>Cause of death<select value={form.death_reason_category} onChange={(event) => setForm({ ...form, death_reason_category: event.target.value })}>{CAUSES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}</select></label>
            <label>Death notes <small>optional</small><textarea maxLength={500} value={form.death_notes} onChange={(event) => setForm({ ...form, death_notes: event.target.value })} /></label></>}
          <label>Overall health<select value={form.health_status} onChange={(event) => setForm({ ...form, health_status: event.target.value })}>{['excellent', 'good', 'fair', 'poor', 'critical'].map((value) => <option key={value} value={value}>{value}</option>)}</select></label>
          <label>Average measured height (cm) <small>optional</small><input type="number" min="0" max="1000" step="any" value={form.average_height_cm} onChange={(event) => setForm({ ...form, average_height_cm: event.target.value })} /></label>
          <label>LGU actions and observations<textarea required maxLength={2000} value={form.actions_taken} onChange={(event) => setForm({ ...form, actions_taken: event.target.value })} /></label>
          <button className="btn btn-primary" type="submit" disabled={busy || !active.site.monitoring_available}>{busy ? 'Saving…' : 'Save site monitoring'}</button>
        </fieldset>
      </form>}
      <FormMessage error={formError} />
      {formError && <button type="button" className="btn" disabled={busy} onClick={() => void open(active.site, active.view, form?.monitored_at)}>Reload this site</button>}
    </Modal>}
  </div></main>;
}
