import { useEffect, useRef, useState } from 'react';
import Modal from './Modal';
import { REPLANTING_BLOCKED } from '../utils/erosionMessages';
import './ReleaseDeadLocations.css';

const API = import.meta.env.VITE_API_BASE || '';

export default function ReleaseDeadLocations({ points, projectSite, onClose, onReleased }) {
  const [candidates, setCandidates] = useState([]);
  const [loading, setLoading] = useState(true);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const submitting = useRef(false);
  const selectedKey = JSON.stringify(points.map((point) => Number(point.id)).sort((a, b) => a - b));
  const projectSiteId = projectSite?.id;

  useEffect(() => {
    const controller = new AbortController();
    async function load() {
      try {
        if (projectSiteId == null) throw new Error('Choose a project site before releasing locations.');
        const response = await fetch(`${API}/api/monitoring/release-candidates?project_site_id=${encodeURIComponent(projectSiteId)}`, { signal: controller.signal });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || 'Could not load the death record.');
        const ids = JSON.parse(selectedKey);
        const selected = (payload.points || []).filter((item) => ids.includes(Number(item.planting_point_id)));
        if (selected.some((item) => Number(item.project_site_id) !== Number(projectSiteId))) throw new Error('Some selected locations belong to another project site. Close this window and select this site’s points again.');
        if (selected.some((item) => item.eroded_unavailable || item.inside_eroded_zone)) throw new Error(REPLANTING_BLOCKED);
        const current = selected.filter((item) => item.can_release);
        if (current.length !== ids.length) throw new Error('Some selected points are no longer available for release. Close this window and refresh the map.');
        if (!controller.signal.aborted) setCandidates(current);
      } catch (failure) {
        if (!controller.signal.aborted) setError(failure.message);
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void load();
    return () => controller.abort();
  }, [selectedKey, projectSiteId]);

  async function release() {
    if (submitting.current || loading || projectSiteId == null || !points.length || candidates.length !== points.length) return;
    submitting.current = true;
    setBusy(true); setError('');
    try {
      const response = await fetch(`${API}/api/monitoring/release-locations`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ project_site_id: Number(projectSiteId), planting_event_ids: candidates.map((item) => item.planting_event_id),
          versions: Object.fromEntries(candidates.map((item) => [item.planting_event_id, item.version])) }),
      });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Could not release these locations.');
      onReleased(payload.released_count);
    } catch (failure) {
      setError(failure.message);
    } finally {
      submitting.current = false;
      setBusy(false);
    }
  }

  if (error === REPLANTING_BLOCKED) return <Modal open title="Replanting blocked" variant="warning"
    confirmLabel="OK" cancelLabel={null} onConfirm={onClose} onCancel={onClose}><p>{REPLANTING_BLOCKED}</p></Modal>;

  return <Modal open title={`Release ${points.length} dead location${points.length === 1 ? '' : 's'}?`}
    variant="info" busy={busy} className="release-locations-modal" initialFocus="cancel"
    cancelLabel="Cancel" onCancel={onClose} onConfirm={release}
    confirmLabel={`Release ${points.length} location${points.length === 1 ? '' : 's'}`}
    confirmDisabled={loading || !points.length || candidates.length !== points.length}>
    <p>These locations will leave <strong>{projectSite?.name}</strong> and become <strong>Planned</strong>, ready for planting again.</p>
    <p className="release-locations-history">Original planting records and survival rates stay with their sites.</p>
    {loading ? <p className="release-locations-status" role="status">Checking selected locations…</p> : null}
    {error ? <p className="analytics-error" role="alert">{error}</p> : null}
  </Modal>;
}
