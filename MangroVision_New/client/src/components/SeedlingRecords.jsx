import { useMemo, useState } from 'react';
import { PanelCard } from './Panel';
import './SeedlingRecords.css';

const API = import.meta.env.VITE_API_BASE || '';
const today = () => {
  const parts = Object.fromEntries(new Intl.DateTimeFormat('en-US', {
    timeZone: 'Asia/Manila', year: 'numeric', month: '2-digit', day: '2-digit',
  }).formatToParts(new Date()).map((part) => [part.type, part.value]));
  return `${parts.year}-${parts.month}-${parts.day}`;
};

function availablePoint(point, siteId) {
  return Number(point.source_project_site_id) === Number(siteId)
    && point.planting_status === 'planned'
    && point.assigned_planter_id == null
    && !point.deleted_at && !point.death_at
    && !point.eroded_unavailable && !point.inside_eroded_zone;
}

export default function SeedlingRecords({ points, projectSites, onSaved, disabled = false }) {
  const sites = projectSites?.features || [];
  const [siteId, setSiteId] = useState('');
  const [search, setSearch] = useState('');
  const [selectedIds, setSelectedIds] = useState([]);
  const [plantingDate, setPlantingDate] = useState(today);
  const [species, setSpecies] = useState('');
  const [height, setHeight] = useState('');
  const [condition, setCondition] = useState('');
  const [notes, setNotes] = useState('');
  const [confirmed, setConfirmed] = useState(false);
  const [busy, setBusy] = useState(false);
  const [message, setMessage] = useState('');
  const [error, setError] = useState('');

  const available = useMemo(() => points.filter((point) => availablePoint(point, siteId)), [points, siteId]);
  const visible = useMemo(() => available.filter((point) => (
    !search || `${point.point_num} ${point.image_name || ''}`.toLowerCase().includes(search.toLowerCase())
  )).slice(0, 100), [available, search]);

  const savePlanting = async (event) => {
    event.preventDefault();
    setError(''); setMessage('');
    if (!confirmed) { setError('Confirm that these seedlings were actually planted.'); return; }
    if (!selectedIds.length || !siteId || !species || !plantingDate) {
      setError('Choose a site, points, planting date, and species.'); return;
    }
    setBusy(true);
    try {
      const response = await fetch(`${API}/api/seedlings/lgu-plantings`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          planting_point_ids: selectedIds, project_site_id: Number(siteId),
          planted_date: plantingDate, species,
          initial_height_cm: height === '' ? null : Number(height),
          initial_condition: condition || null, initial_notes: notes || null,
        }),
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || 'LGU planting could not be recorded.');
      setMessage(`${payload.planted_points} seedling${payload.planted_points === 1 ? '' : 's'} recorded under LGU.`);
      setSelectedIds([]); setConfirmed(false); setHeight(''); setCondition(''); setNotes('');
      await onSaved?.();
    } catch (saveError) {
      setError(saveError.message || 'LGU planting could not be recorded.');
    } finally { setBusy(false); }
  };

  return (
    <PanelCard title="LGU planting" defaultOpen={false}>
      <p className="seedling-intro">Choose the mapped points the LGU actually planted. These count under LGU planting history even when the project site also has an organization.</p>
      <form className="seedling-form" onSubmit={savePlanting}>
        <label>Project site<select value={siteId} onChange={(event) => { setSiteId(event.target.value); setSelectedIds([]); }} disabled={disabled || busy}>
          <option value="">Select a site</option>
          {sites.map((site) => <option key={site.id ?? site.properties?.id} value={site.id ?? site.properties?.id}>{site.properties?.name || `Site ${site.id}`}</option>)}
        </select></label>
        <label>Find point<input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Point number or analysis name" disabled={!siteId || busy} /></label>
        <div className="seedling-point-list" aria-label="Available mapped planting points">
          {siteId && visible.length === 0 && <span>No available points match this search.</span>}
          {visible.map((point) => <label key={point.id}><input type="checkbox" checked={selectedIds.includes(point.id)} disabled={busy || disabled}
            onChange={() => setSelectedIds((current) => current.includes(point.id) ? current.filter((id) => id !== point.id) : [...current, point.id])} />
            Point #{point.point_num} · {point.image_name || 'Saved analysis'}</label>)}
        </div>
        {siteId && <small>{available.length} available in this site · showing up to 100 matching points · {selectedIds.length} selected</small>}
        <label>Actual planting date<input type="date" value={plantingDate} max={today()} onChange={(event) => setPlantingDate(event.target.value)} required /></label>
        <label>Species<select value={species} onChange={(event) => setSpecies(event.target.value)} required><option value="">Choose species</option><option>Bungalon</option><option>Rhizophora</option><option>Api-Api</option></select></label>
        <label>Initial height (cm, optional)<input type="number" min="0" max="1000" step="0.1" value={height} onChange={(event) => setHeight(event.target.value)} /></label>
        <label>Initial condition (optional)<select value={condition} onChange={(event) => setCondition(event.target.value)}><option value="">Not recorded</option><option value="healthy">Healthy</option><option value="fair">Fair</option><option value="stressed">Stressed</option><option value="unknown">Unknown</option></select></label>
        <label>Notes (optional)<textarea value={notes} onChange={(event) => setNotes(event.target.value)} maxLength={1000} /></label>
        <label className="seedling-confirm"><input type="checkbox" checked={confirmed} onChange={(event) => setConfirmed(event.target.checked)} /> I confirm these seedlings were planted on the selected date.</label>
        <button className="btn btn-primary btn-sm" type="submit" disabled={disabled || busy || !selectedIds.length}>{busy ? 'Saving…' : `Record ${selectedIds.length} LGU planting${selectedIds.length === 1 ? '' : 's'}`}</button>
      </form>
      {error && <p role="alert" className="seedling-error">{error}</p>}{message && <p role="status" className="seedling-success">{message}</p>}
    </PanelCard>
  );
}
