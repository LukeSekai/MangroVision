import { useEffect, useState } from 'react';
import Modal from './Modal';

const API = import.meta.env.VITE_API_BASE || '';

export default function ReplantingApproval({ points, onClose, onApproved }) {
  const [candidates, setCandidates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const selectedKey = JSON.stringify(points.map((point) => Number(point.id)).sort((a, b) => a - b));

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        const response = await fetch(`${API}/api/monitoring/replanting`, { signal: controller.signal });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || 'Could not load the death record.');
        const ids = JSON.parse(selectedKey);
        const current = (payload.points || []).filter((item) => ids.includes(Number(item.planting_point_id))
          && item.replanting_status === 'awaiting_review');
        if (current.length !== ids.length) throw new Error('Some selected points are no longer awaiting approval. Close this window and refresh the map.');
        if (!controller.signal.aborted) setCandidates(current);
      } catch (failure) {
        if (!controller.signal.aborted) setError(failure.message);
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void load();
    return () => controller.abort();
  }, [selectedKey]);

  async function approve() {
    if (busy || loading || candidates.length !== points.length) return;
    setBusy(true); setError('');
    try {
      const response = await fetch(`${API}/api/monitoring/replanting/approve`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ planting_event_ids: candidates.map((item) => item.planting_event_id),
          versions: Object.fromEntries(candidates.map((item) => [item.planting_event_id, item.version])) }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Could not approve this location.');
      onApproved(payload.approved_count);
    } catch (failure) {
      setError(failure.message);
    } finally {
      setBusy(false);
    }
  }

  return <Modal open title={`Approve ${points.length} dead plant${points.length === 1 ? '' : 's'} for replanting`} variant="info" busy={busy}
    cancelLabel="Close" onCancel={onClose}>
    <p>Release the selected locations from their current organizations for new planting assignments.</p>
    <p>Each original organization keeps its recorded deaths, alive count, and survival statistics.</p>
    {loading ? <p role="status">Loading death record…</p> : null}
    {error ? <p className="analytics-error" role="alert">{error}</p> : null}
    <button type="button" className="btn btn-primary" disabled={loading || busy || candidates.length !== points.length} onClick={approve}>
      {busy ? 'Approving…' : `Approve ${points.length} selected`}
    </button>
  </Modal>;
}
