import { lazy, Suspense, useState } from 'react';
import Modal from './Modal';
import useFormFeedback from '../utils/useFormFeedback';
import { formRequest } from '../utils/formValidation';
import { FieldError, FormMessage } from './FormFeedback';

const SeedlingLocations = lazy(() => import('./SeedlingLocations'));
const API = import.meta.env.VITE_API_BASE || '';

export default function DeathLocationEditor({ record, onClose, onSaved }) {
  const [selected, setSelected] = useState(record.dead_planting_event_ids || []);
  const [note, setNote] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const reported = record.reported_dead_count;
  const removed = (record.dead_planting_event_ids || []).some((id) => !selected.includes(id));
  const feedback = useFormFeedback({ correction_note: { label: 'Location notes' } });
  async function save(event) {
    event.preventDefault();
    if (busy) return;
    if (!feedback.validate()) return;
    if (selected.length > reported) { setError(`Select no more than ${reported} seedlings.`); return; }
    setBusy(true); setError('');
    try {
      const payload = await formRequest(`${API}/api/monitoring/organization-records/${record.id}/death-locations`, {
        method: 'PUT', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ dead_planting_event_ids: selected, expected_version: record.location_version, correction_note: note.trim() }),
      });
      onSaved(payload);
    } catch (failure) { if (!feedback.fromServer(failure)) setError(failure.status === 409 ? `${failure.message} Close and reopen this visit to load its latest locations.` : failure.message); }
    finally { setBusy(false); }
  }
  return <Modal open title="Identify or correct death locations" className="modal-card-wide" variant="info" busy={busy} cancelLabel="Back to history" onCancel={onClose}>
    <form noValidate onChangeCapture={feedback.onChange} aria-busy={busy} onSubmit={save}>
      <p>Visit: {new Date(record.monitored_at).toLocaleDateString('en-PH', { timeZone: 'Asia/Manila' })}. Saving locations keeps this visit’s death count, survival rate, and next visit date unchanged.</p>
      <p aria-live="polite"><strong>{reported} reported dead · {selected.length} located · {Math.max(0, reported - selected.length)} still to locate</strong></p>
      <Suspense fallback={<p>Loading locations…</p>}><SeedlingLocations organizationId={record.organization_id} recordId={record.id} selected={selected} onChange={setSelected} disabled={busy} maxSelected={reported} /></Suspense>
      <label>Location notes {removed ? '(required for corrections)' : '(optional)'}<textarea {...feedback.props('correction_note')} className="org-monitoring-input" value={note} onChange={(e) => setNote(e.target.value)} maxLength={2000} required={removed} disabled={busy} /><FieldError feedback={feedback} field="correction_note" /></label>
      {selected.length > reported ? <p role="alert">Select no more than {reported} seedlings, the number reported in this visit.</p> : null}
      <FormMessage error={error} loading={busy ? 'Saving locations…' : ''} />
      <button className="org-monitoring-submit" type="submit" disabled={busy || selected.length > reported}>{busy ? 'Saving…' : error ? 'Retry save locations' : 'Save locations'}</button>
    </form>
  </Modal>;
}
