import { useCallback, useEffect, useMemo, useState } from 'react';
import { PanelCard } from './Panel';
import { SeedlingMap } from './SeedlingLocations';
import { toggleSeedling } from '../utils/monitoringLocations';

const API = import.meta.env.VITE_API_BASE || '';
const STATUS = { awaiting_review: 'Awaiting LGU review', approved: 'Approved · awaiting assignment', assigned: 'Replacement assigned', completed: 'Replacement planted' };

export default function ReplantingPanel({ organizations, refreshKey, onChanged }) {
  const [points, setPoints] = useState([]);
  const [status, setStatus] = useState('awaiting_review');
  const [source, setSource] = useState('');
  const [selected, setSelected] = useState([]);
  const [organization, setOrganization] = useState('');
  const [assignmentDate, setAssignmentDate] = useState('');
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const load = useCallback(async () => {
    setLoading(true); setError('');
    try {
      const response = await fetch(`${API}/api/monitoring/replanting`);
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Could not load replacement work.');
      setPoints(payload.points || []); setSelected([]);
    } catch (failure) { setError(failure.message); }
    finally { setLoading(false); }
  }, []);
  useEffect(() => { const timer = setTimeout(() => void load(), 0); return () => clearTimeout(timer); }, [load, refreshKey]);
  const filtered = useMemo(() => points.filter((p) => (!status || p.replanting_status === status)
    && (!source || String(p.organization_id) === source)).map((p) => ({ ...p, selectable: p.replanting_status === 'approved' })), [points, source, status]);
  async function mutate(path, body, message) {
    if (busy) return;
    setBusy(true); setError(''); setNotice('');
    try {
      const response = await fetch(`${API}/api/monitoring/replanting/${path}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(body),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Could not update replacement work.');
      setNotice(message); setOrganization(''); setAssignmentDate('');
      await load(); onChanged?.();
    } catch (failure) { setError(failure.message); }
    finally { setBusy(false); }
  }
  function toggle(id) { if (!busy) setSelected((current) => toggleSeedling(current, id)); }
  return <PanelCard panelKey="needs-replanting" title="Needs replanting" className="replanting-panel">
    <p>Review identified deaths, then choose an organization for replacement planting. A replacement is recorded as planted when its assignment is completed.</p>
    <p className="replanting-map-legend"><span style={{ color: '#dc2626' }}>● Awaiting review</span> · <span style={{ color: '#b45309' }}>● Approved</span> · <span style={{ color: '#2563eb' }}>● Assigned</span> · <span style={{ color: '#166534' }}>● Planted</span> · <span style={{ color: '#7c3aed' }}>● Selected</span></p>
    <div className="seedling-location-filters">
      <label>Replacement stage<select value={status} disabled={busy} onChange={(e) => { setStatus(e.target.value); setSelected([]); }}><option value="">All stages</option>{Object.entries(STATUS).map(([value, label]) => <option key={value} value={value}>{label} ({points.filter((p) => p.replanting_status === value).length})</option>)}</select></label>
      <label>Original organization<select value={source} disabled={busy} onChange={(e) => { setSource(e.target.value); setSelected([]); }}><option value="">All organizations</option>{organizations.map((o) => <option key={o.id} value={o.id}>{o.name}</option>)}</select></label>
      <button type="button" onClick={load} disabled={busy || loading}>Refresh locations</button>
    </div>
    {error ? <p role="alert" className="org-monitoring-message is-error">{error}</p> : null}
    {notice ? <p role="status" className="org-monitoring-message is-success">{notice}</p> : null}
    {loading ? <p role="status">Loading replacement locations…</p> : <>
      {filtered.length ? <><SeedlingMap points={filtered} selected={selected} onToggle={toggle} />
        <div className="seedling-location-checklist">{filtered.map((p) => <div key={p.planting_event_id} className="replanting-row">
          <label>{p.selectable ? <input type="checkbox" checked={selected.includes(p.planting_event_id)} onChange={() => toggle(p.planting_event_id)} disabled={busy} /> : null}
            <span><strong>{p.analysis_name} · Point {p.point_num}</strong><small>{p.project_site_name} · {p.organization_name} · {p.species}<br />{STATUS[p.replanting_status]}{p.assignment_id ? ` · Assignment ${p.assignment_id}` : ''}</small></span></label>
          {p.replanting_status === 'awaiting_review' ? <button type="button" disabled={busy} onClick={() => mutate(`${p.planting_event_id}/approve`, { expected_version: p.version }, 'Location approved for replacement planting.')}>Approve for replacement</button> : null}
        </div>)}</div></> : <p>No locations at this stage. Deaths with unknown locations appear here after they are identified in Monitoring History.</p>}
      {selected.length > 0 ? <form className="replanting-assignment" onSubmit={(e) => { e.preventDefault(); if (!organization || !assignmentDate) return;
        void mutate('assign', { planting_event_ids: selected, versions: Object.fromEntries(points.filter((p) => selected.includes(p.planting_event_id)).map((p) => [p.planting_event_id, p.version])), organization_id: Number(organization), assignment_date: assignmentDate }, 'Replacement assignment created. Confirm planting through the existing assignment workflow.');
      }}>
        <p>{selected.length} approved locations selected. Choose an organization authorized for their project sites. Existing species will be kept.</p>
        <div className="seedling-location-filters"><label>Assign replacement work to<select required value={organization} onChange={(e) => setOrganization(e.target.value)} disabled={busy}><option value="">Select an organization</option>{organizations.map((o) => <option key={o.id} value={o.id}>{o.name}</option>)}</select></label>
          <label>Planting work date<input type="date" required value={assignmentDate} onChange={(e) => setAssignmentDate(e.target.value)} disabled={busy} /></label>
          <button type="submit" disabled={busy || !organization || !assignmentDate}>{busy ? 'Saving…' : 'Create replacement assignment'}</button></div>
      </form> : null}
    </>}
  </PanelCard>;
}
