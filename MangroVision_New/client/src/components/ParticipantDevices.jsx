import { useCallback, useEffect, useState } from 'react';
import Modal from './Modal';
import './ParticipantDevices.css';

const API = import.meta.env.VITE_API_BASE || '';

function activityTime(value) {
  if (!value) return 'No sign-in recorded';
  const date = new Date(value);
  return Number.isNaN(date.getTime()) ? 'Unknown' : date.toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' });
}

export default function ParticipantDevices({ planter }) {
  const [summary, setSummary] = useState(null);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [message, setMessage] = useState('');
  const [resetSlot, setResetSlot] = useState('');
  const [resetTarget, setResetTarget] = useState(null);

  const loadDevices = useCallback(async (signal) => {
    try {
      const response = await fetch(`${API}/api/planters/${planter.id}/participants`, { signal });
      const data = await response.json();
      if (!response.ok) throw new Error(data.detail || 'Could not load participant devices.');
      if (signal?.aborted) return;
      setSummary(data);
      setResetSlot((current) => data.devices.some((device) => device.registered && String(device.slot) === current)
        ? current : String(data.devices.find((device) => device.registered)?.slot || ''));
    } catch (problem) {
      if (!signal?.aborted) setError(problem.message);
    } finally {
      if (!signal?.aborted) setLoading(false);
    }
  }, [planter.id]);

  useEffect(() => {
    const controller = new AbortController();
    queueMicrotask(() => {
      if (!controller.signal.aborted) loadDevices(controller.signal);
    });
    return () => controller.abort();
  }, [loadDevices]);

  const resetDevice = async () => {
    const slot = resetTarget;
    setBusy(true);
    setError('');
    setMessage('');
    try {
      const response = await fetch(`${API}/api/planters/${planter.id}/participants/${slot}/reset-device`, { method: 'POST' });
      const result = await response.json();
      if (!response.ok) throw new Error(result.detail || 'Could not reset device.');
      setResetTarget(null);
      setMessage(`Participant ${slot} is ready for recovery. Their points and planting history are preserved. On the replacement browser, choose the LGU reset option and enter ${slot}.`);
      setLoading(true);
      await loadDevices();
    } catch (problem) { setError(problem.message); }
    finally { setBusy(false); }
  };

  return <div className="participant-devices">
    <p className="text-sm">Each browser profile on a field link counts as a device. Different links or browsers on the same phone can use extra slots. Use a recovery code when the link changes.</p>
    <div className="participant-device-summary">
      <strong>{summary ? `${summary.registered_devices} of ${summary.participant_count} device slots used` : 'Participant devices'}</strong>
      <button type="button" className="btn btn-ghost btn-sm" disabled={loading || busy} onClick={() => { setLoading(true); setError(''); loadDevices(); }}>Refresh</button>
    </div>
    {loading && <p role="status" className="text-sm">Loading device slots…</p>}
    {error && <p role="alert" className="assign-message assign-error">{error}</p>}
    {summary && <>
      <ul className="participant-device-list" aria-label={`${planter.full_name} participant device slots`}>
        {summary.devices.map((device) => <li key={device.slot}>
          <div className="participant-device-heading"><strong>Participant {device.slot}</strong><span>{device.registered ? 'Registered' : 'Available'}</span></div>
          <span>{device.assigned_points} points · {device.completed_points} planted</span>
          <span>Slot last active: {activityTime(device.last_seen_at)}</span>
        </li>)}
      </ul>
      <p className="text-sm">Reset only a slot you have identified as a replaced or duplicate browser. Reset signs out that participant and preserves their points.</p>
      <label className="form-label" htmlFor="participant-reset-slot">Participant to recover</label>
      <select id="participant-reset-slot" className="form-input" value={resetSlot} onChange={(event) => setResetSlot(event.target.value)} disabled={busy || loading}>
        <option value="">Select a registered participant</option>
        {summary.devices.filter((device) => device.registered).map((device) => <option key={device.slot} value={device.slot}>Participant {device.slot} · {device.completed_points} planted</option>)}
      </select>
      <button type="button" className="btn btn-secondary btn-sm" disabled={!resetSlot || loading || busy} onClick={() => setResetTarget(Number(resetSlot))}>Reset device slot</button>
    </>}
    {message && <p role="status" className="text-sm">{message}</p>}
    <Modal open={resetTarget !== null} title={`Reset Participant ${resetTarget}'s device?`} variant="warning" confirmLabel="Reset device slot" busy={busy} onConfirm={resetDevice} onCancel={() => setResetTarget(null)}>
      <p>This signs out Participant {resetTarget} on their existing browser. Their assigned points and planting history stay saved.</p>
      <p>The returning participant must choose the LGU reset option and enter {resetTarget} to resume those same points.</p>
    </Modal>
  </div>;
}
