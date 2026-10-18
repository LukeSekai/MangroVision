import { useCallback, useEffect, useRef, useState } from 'react';
import MonitoringHeader from '../components/MonitoringHeader';
import useMapNavigation from '../utils/useMapNavigation';
import Modal from '../components/Modal';
import ReplantingApproval from '../components/ReplantingApproval';
import { FormMessage } from '../components/FormFeedback';
import { useMapStore } from '../stores/mapStore';
import { manilaDay } from '../utils/restorationReports';
import { REPLANTING_STAGES, replantingProgress, replantingRows, replantingStatusLabel, replantingVersions } from '../utils/lguReplanting';
import './OrganizationMonitoring.css';
import './LGUReplanting.css';

const API = import.meta.env.VITE_API_BASE || '';
const PAGE_SIZE = 10;
const date = (value) => value ? new Date(value).toLocaleDateString('en-PH', { timeZone: 'Asia/Manila' }) : '—';

export function ReplantingSummary({ progress }) {
  return <div className="org-monitoring-kpis lgu-replanting-kpis" aria-label="Replanting progress">
    <article><span>Awaiting review</span><strong>{progress.awaiting_review ?? '—'}</strong><small>Recorded deaths awaiting LGU approval</small></article>
    <article><span>Ready for LGU replanting</span><strong>{progress.ready_lgu ?? '—'}</strong><small>Blue assigned points awaiting actual planting</small></article>
    <article><span>Replacements planted</span><strong>{progress.replacements_completed ?? '—'}</strong><small>Actual replacement plantings recorded</small></article>
  </div>;
}

export default function LGUReplanting() {
  const navigateMap = useMapNavigation();
  const [rows, setRows] = useState([]);
  const [sites, setSites] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [stage, setStage] = useState('ready_lgu');
  const [site, setSite] = useState('');
  const [search, setSearch] = useState('');
  const [page, setPage] = useState(1);
  const [selectedIds, setSelectedIds] = useState([]);
  const [reviewing, setReviewing] = useState(false);
  const [planting, setPlanting] = useState(false);
  const [busy, setBusy] = useState(false);
  const [formError, setFormError] = useState('');
  const [form, setForm] = useState({ planted_date: manilaDay(), project_site_id: '', species: '', initial_height_cm: '', initial_condition: '', initial_notes: '' });
  const formRef = useRef(null);
  const loadGeneration = useRef(0);
  const busyRef = useRef(false);
  const dialogRef = useRef(false);
  const cancelPendingLoad = useCallback(() => { ++loadGeneration.current; }, []);
  useEffect(() => { dialogRef.current = reviewing || planting; }, [reviewing, planting]);

  const load = useCallback(async () => {
    const generation = ++loadGeneration.current;
    setLoading(true);
    try {
      const responses = await Promise.all([
        fetch(`${API}/api/monitoring/replanting`, { cache: 'reload' }),
        fetch(`${API}/api/project-sites`, { cache: 'reload' }),
      ]);
      const payloads = await Promise.all(responses.map((response) => response.json()));
      if (responses.some((response) => !response.ok)) throw new Error(payloads.find((payload) => payload.detail)?.detail || 'Could not load LGU replanting.');
      if (generation !== loadGeneration.current) return;
      setRows(payloads[0].points || []);
      setSites((payloads[1].features || []).filter((feature) => feature.geometry && !feature.properties?.deleted_at && !feature.properties?.area_released_at));
      setError('');
      setSelectedIds([]);
    } catch (failure) {
      if (generation === loadGeneration.current) setError(failure.message || 'Could not load LGU replanting.');
    } finally {
      if (generation === loadGeneration.current) setLoading(false);
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => { void load(); }, 0);
    const refresh = () => { if (!busyRef.current && !dialogRef.current) void load(); };
    window.addEventListener('mv:data-changed', refresh);
    return () => { window.clearTimeout(timer); window.removeEventListener('mv:data-changed', refresh); cancelPendingLoad(); };
  }, [load, cancelPendingLoad]);

  const filtered = replantingRows(rows, { stage, site, search });
  const pages = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const activePage = Math.min(page, pages);
  const visible = filtered.slice((activePage - 1) * PAGE_SIZE, activePage * PAGE_SIZE);
  const selected = rows.filter((row) => selectedIds.includes(row.planting_event_id));
  const progress = replantingProgress(site ? rows.filter((row) => String(row.project_site_id) === site) : rows);
  const canSelect = (row) => stage === 'awaiting_review' ? row.can_approve : stage === 'ready_lgu' && row.can_record_replanting;

  function changeScope(setter, value) {
    setter(value); setPage(1); setSelectedIds([]);
  }

  function toggle(row) {
    setSelectedIds((ids) => ids.includes(row.planting_event_id)
      ? ids.filter((id) => id !== row.planting_event_id)
      : ids.length < 500 ? [...ids, row.planting_event_id] : ids);
  }

  function refreshMap() {
    const map = useMapStore.getState();
    void map.fetchPoints({ force: true });
    void map.fetchStats({ force: true });
    void map.fetchZones({ force: true });
  }

  function openPlanting() {
    setForm({ planted_date: manilaDay(), project_site_id: String(selected[0]?.project_site_id || ''),
      species: selected.every((row) => row.species === selected[0]?.species) ? selected[0]?.species || '' : '',
      initial_height_cm: '', initial_condition: '', initial_notes: '' });
    setFormError(''); setPlanting(true);
  }

  async function recordPlanting(event) {
    event.preventDefault();
    if (busyRef.current || !selected.length) return;
    busyRef.current = true; setBusy(true); setFormError('');
    try {
      const response = await fetch(`${API}/api/monitoring/replanting/lgu-plantings`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ ...form, project_site_id: Number(form.project_site_id),
          initial_height_cm: form.initial_height_cm === '' ? null : Number(form.initial_height_cm),
          initial_condition: form.initial_condition || null,
          planting_event_ids: selected.map((row) => row.planting_event_id), versions: replantingVersions(selected) }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(typeof payload.detail === 'string' ? payload.detail : 'Could not record replacement planting. Check the fields and try again.');
      setPlanting(false);
      setNotice(`${payload.replanted_points} replacement seedling${payload.replanted_points === 1 ? '' : 's'} recorded. Their map locations are now planted; original deaths remain in history.`);
      await load(); refreshMap();
    } catch (failure) {
      setFormError(failure.message || 'Could not record replacement planting.');
    } finally {
      busyRef.current = false; setBusy(false);
    }
  }

  function viewOnMap(row) {
    const map = useMapStore.getState();
    map.setView([Number(row.longitude), Number(row.latitude)], 22);
    map.setSelectedPoint(row.planting_point_id);
    navigateMap('/map');
  }

  return <main className="org-monitoring-page org-monitoring-panel lgu-replanting-page"><div className="org-monitoring-content">
    <MonitoringHeader title="LGU Replanting"
      description="Review dead locations, prepare replacement work, and record seedlings planted by LGU staff." />
    <ReplantingSummary progress={loading || error ? {} : progress} />
    <FormMessage loading={loading ? 'Loading replanting records…' : ''} error={error} success={notice} onRetry={load} busy={loading} />
    <section className="lgu-replanting-workspace" data-guide-card="lgu-replanting">
      <p className="org-monitoring-intro">Approval assigns the location to the LGU queue. Blue means awaiting LGU planting; yellow means a replacement has been recorded. Original deaths and survival results stay in their planting batches.</p>
      <div className="lgu-replanting-filters">
        <label>Project site<select value={site} onChange={(event) => changeScope(setSite, event.target.value)}><option value="">All project sites</option>
          {sites.map((feature) => <option key={feature.id ?? feature.properties.id} value={feature.id ?? feature.properties.id}>{feature.properties.name}</option>)}</select></label>
        <label>Find a location<input type="search" value={search} placeholder="Search photo, point, site, or original organization" onChange={(event) => changeScope(setSearch, event.target.value)} /></label>
      </div>
      <div className="org-monitoring-visit-tabs" role="group" aria-label="Replanting stage">
        {REPLANTING_STAGES.map(([value, label]) => <button type="button" key={value} aria-pressed={stage === value} className={stage === value ? 'is-active' : ''}
          onClick={() => changeScope(setStage, value)}>{label}</button>)}
      </div>
      <div className="lgu-replanting-actions">
        <span>{selected.length} selected · {filtered.length} location{filtered.length === 1 ? '' : 's'}</span>
        {stage !== 'completed' && <>
          <button type="button" disabled={loading || !!error || !visible.some(canSelect)} onClick={() => setSelectedIds((ids) => [...new Set([...ids, ...visible.filter(canSelect).map((row) => row.planting_event_id)])].slice(0, 500))}>Select this page</button>
          <button type="button" disabled={!selected.length || loading} onClick={() => setSelectedIds([])}>Clear selection</button>
          <button type="button" className="is-primary" disabled={!selected.length || loading || !!error} onClick={() => stage === 'awaiting_review' ? setReviewing(true) : openPlanting()}>
            {stage === 'awaiting_review' ? 'Review for LGU replanting' : 'Record replacement planting'}</button>
        </>}
      </div>
      {!loading && !error && <div className="lgu-replanting-table-wrap"><table>
        <caption className="sr-only">LGU replanting locations</caption>
        <thead><tr><th scope="col">Select</th><th scope="col">Location</th><th scope="col">Project site</th><th scope="col">Original planting</th><th scope="col">Status</th><th scope="col">Map</th></tr></thead>
        <tbody>{visible.map((row) => <tr key={row.planting_event_id}>
          <td>{stage !== 'completed' ? <input type="checkbox" aria-label={`Select ${row.analysis_name} point ${row.point_num}`} checked={selectedIds.includes(row.planting_event_id)} disabled={!canSelect(row)} onChange={() => toggle(row)} /> : '—'}</td>
          <td><strong>{row.analysis_name} · Point {row.point_num}</strong><small>{row.species || 'Species not recorded'}</small></td>
          <td>{row.project_site_name || 'No linked site'}</td><td>{row.organization_name || 'LGU'}<small>Death recorded {date(row.death_at)}</small></td>
          <td><span className={`lgu-replanting-status is-${row.unavailable_reason ? 'unavailable' : row.replanting_status}`}>{replantingStatusLabel(row)}</span>
            {row.replacement_planted_at && <small>{date(row.replacement_planted_at)}{row.replacement_staff_name ? ` · ${row.replacement_staff_name}` : ''}</small>}</td>
          <td><button type="button" onClick={() => viewOnMap(row)}>View on map</button></td>
        </tr>)}</tbody>
      </table>{!visible.length && <p className="monitoring-visit-empty">No locations in this selection.</p>}</div>}
      <div className="lgu-replanting-pagination"><button type="button" disabled={activePage <= 1} onClick={() => setPage(activePage - 1)}>Previous</button><span>Page {activePage} of {pages}</span><button type="button" disabled={activePage >= pages} onClick={() => setPage(activePage + 1)}>Next</button></div>
    </section>
    {reviewing && <ReplantingApproval points={selected.map((row) => ({ ...row, id: row.planting_point_id }))} onClose={() => setReviewing(false)} onApproved={(count) => {
      setReviewing(false); setNotice(`${count} locations assigned to the LGU replanting queue.`); setStage('ready_lgu'); setPage(1); void load(); refreshMap();
    }} />}
    {planting && <Modal open title={`Record ${selected.length} replacement planting${selected.length === 1 ? '' : 's'}`} busy={busy} confirmLabel="Record planted replacements" onConfirm={() => formRef.current?.requestSubmit()} onCancel={() => setPlanting(false)}>
      <p>Record this after LGU staff have planted the replacement seedlings. Each replacement starts its own planting record.</p>
      <form ref={formRef} onSubmit={recordPlanting} className="lgu-replanting-form">
        <label>Project site<select required value={form.project_site_id} onChange={(event) => setForm({ ...form, project_site_id: event.target.value })}><option value="">Choose the site containing all selected points</option>{sites.map((feature) => <option key={feature.id ?? feature.properties.id} value={feature.id ?? feature.properties.id}>{feature.properties.name}</option>)}</select></label>
        <label>Planting date<input required type="date" max={manilaDay()} value={form.planted_date} onChange={(event) => setForm({ ...form, planted_date: event.target.value })} /></label>
        <label>Species<select required value={form.species} onChange={(event) => setForm({ ...form, species: event.target.value })}><option value="">Choose species</option><option value="Rhizophora">Rhizophora</option><option value="Bungalon">Bungalon</option></select></label>
        <label>Initial height (cm) <small>optional</small><input type="number" min="0" max="1000" step="any" value={form.initial_height_cm} onChange={(event) => setForm({ ...form, initial_height_cm: event.target.value })} /></label>
        <label>Initial condition <small>optional</small><select value={form.initial_condition} onChange={(event) => setForm({ ...form, initial_condition: event.target.value })}><option value="">Not recorded</option>{['healthy', 'fair', 'stressed', 'unknown'].map((value) => <option key={value} value={value}>{value[0].toUpperCase() + value.slice(1)}</option>)}</select></label>
        <label>Notes <small>optional</small><textarea maxLength={1000} value={form.initial_notes} onChange={(event) => setForm({ ...form, initial_notes: event.target.value })} /></label>
        <FormMessage error={formError} />
        {formError && <button type="button" className="btn" disabled={busy} onClick={() => { setPlanting(false); void load(); }}>Reload queue</button>}
      </form>
    </Modal>}
  </div></main>;
}
