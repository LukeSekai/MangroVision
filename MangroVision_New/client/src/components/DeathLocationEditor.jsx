import { lazy, Suspense, useState } from 'react';
import Modal from './Modal';

const SeedlingLocations = lazy(() => import('./SeedlingLocations'));
const API = import.meta.env.VITE_API_BASE || '';

export default function DeathLocationEditor({ record, onClose, onSaved }) {
  const [selected, setSelected] = useState(record.dead_planting_event_ids || []);
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const reported = record.reported_dead_count;
  const removed = (record.dead_planting_event_ids || []).some((id) => !selected.includes(id));
  async function save(event) {
    event.preventDefault();
    if (busy || selected.length > reported || (removed && !note.trim())) return;
    setBusy(true); setError('');
    try {
      const response = await fetch(`${API}/api/monitoring/organization-records/${record.id}/death-locations`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dead_planting_event_ids: selected, expected_version: record.location_version, correction_note: note.trim() }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Could not save locations.');
      onSaved(payload);
    } catch (failure) { setError(failure.message); }
    finally { setBusy(false); }
  }
  return <Modal open title="Identify or correct death locations" className="modal-card-wide" variant="info" busy={busy} cancelLabel="Back to history" onCancel={onClose}>
    <form onSubmit={save}>
      <p>Visit: {new Date(record.monitored_at).toLocaleDateString('en-PH', { timeZone: 'Asia/Manila' })}. Saving locations keeps this visit’s death count, survival rate, and next visit date unchanged.</p>
      <p aria-live="polite"><strong>{reported} reported dead · {selected.length} located · {Math.max(0, reported - selected.length)} still to locate</strong></p>
      <Suspense fallback={<p>Loading locations…</p>}><SeedlingLocations organizationId={record.organization_id} recordId={record.id} selected={selected} onChange={setSelected} disabled={busy} maxSelected={reported} /></Suspense>
      <label>Location notes {removed ? '(required for corrections)' : '(optional)'}<textarea className="org-monitoring-input" value={note} onChange={(e) => setNote(e.target.value)} maxLength={2000} required={removed} disabled={busy} /></label>
      {selected.length > reported ? <p role="alert">Select no more than {reported} seedlings, the number reported in this visit.</p> : null}
      {error ? <p className="org-monitoring-message is-error" role="alert">{error} Close and reopen this visit to load the latest locations.</p> : null}
      <button className="org-monitoring-submit" type="submit" disabled={busy || selected.length > reported || (removed && !note.trim())}>{busy ? 'Saving…' : 'Save locations'}</button>
    </form>
  </Modal>;
}
