import { useEffect, useRef, useState } from 'react';
import Modal from './Modal';
import { growthStageLabel } from '../utils/mangroveGrowth';
import AutomaticGrowth from './AutomaticGrowth';
import DeathLocationEditor from './DeathLocationEditor';

const API = import.meta.env.VITE_API_BASE || '';
const dateLabel = (value) => new Date(value).toLocaleDateString(undefined, {
  timeZone: 'Asia/Manila', month: 'short', day: 'numeric', year: 'numeric',
});

export default function OrganizationHistory({ organization, onClose, onChanged }) {
  const [editing, setEditing] = useState(null);
  const [records, setRecords] = useState([]);
  const [beforeId, setBeforeId] = useState(null);
  const [nextId, setNextId] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [retry, setRetry] = useState(0);
  const loadedPage = useRef(null);

  useEffect(() => {
    const key = `${organization.id}:${beforeId}:${retry}`;
    if (loadedPage.current === key) return undefined;
    const controller = new AbortController();
    async function load() {
      setLoading(true);
      setError('');
      try {
        const query = new URLSearchParams({ organization_id: organization.id, limit: 100 });
        if (beforeId !== null) query.set('before_id', beforeId);
        const response = await fetch(`${API}/api/monitoring/organization-records?${query}`, { signal: controller.signal });
        const payload = await response.json();
        if (!response.ok || !Array.isArray(payload.records)) throw new Error(payload.detail || 'Could not load visit history.');
        if (controller.signal.aborted) return;
        setRecords((current) => [...new Map([...(beforeId === null ? [] : current), ...payload.records].map((record) => [record.id, record])).values()]);
        setNextId(payload.next_before_id ?? null);
      } catch (failure) {
        if (!controller.signal.aborted) setError(failure.message || 'Could not load visit history.');
      } finally {
        if (!controller.signal.aborted) { loadedPage.current = key; setLoading(false); }
      }
    }
    void load();
    return () => controller.abort();
  }, [organization.id, beforeId, retry]);

  useEffect(() => {
    const refresh = () => {
      setBeforeId(null);
      setRetry((value) => value + 1);
    };
    window.addEventListener('mv:data-changed', refresh);
    return () => window.removeEventListener('mv:data-changed', refresh);
  }, []);

  if (editing) return <DeathLocationEditor key={editing.id} record={editing} onClose={() => setEditing(null)} onSaved={(updated) => {
    setRecords((current) => current.map((item) => item.id === updated.id ? updated : item));
    setEditing(null); onChanged?.();
  }} />;
  return <Modal open title={`Monitoring history · ${organization.name}`} className="modal-card-wide org-monitoring-history-modal"
    variant="info" confirmLabel="Close history" cancelLabel={null} onConfirm={onClose} onCancel={onClose}>
    <div className="org-monitoring-history-visits">
      {!loading && !error && !records.length ? <p className="org-monitoring-muted">No monitoring visits recorded for this organization yet.</p> : null}
      {records.map((record) => <article key={record.id} className="org-monitoring-record">
        <div className="org-monitoring-record-head">
          <strong>{dateLabel(record.monitored_at)}</strong>
          <span className={`org-monitoring-health is-${record.health_status}`}>{record.health_status || 'Not recorded'}</span>
        </div>
        <div className="org-monitoring-record-metrics">
          <span><strong>{record.alive_count + record.dead_count}</strong> seedlings counted</span>
          <span><strong>{record.alive_count}</strong> alive</span>
          <span><strong>{record.dead_count}</strong> total dead</span>
          {record.new_dead_count !== null && record.new_dead_count !== undefined ? <span><strong>{record.new_dead_count}</strong> newly dead this visit</span> : null}
          <span><strong>{record.survival_rate_pct ?? '—'}%</strong> survival</span>
          <span><strong>{record.growth_snapshot?.label || growthStageLabel(record.growth_stage)}</strong> {record.growth_snapshot ? 'automatic estimate' : 'previously recorded stage'}</span>
          {record.average_height_cm !== null && record.average_height_cm !== undefined
            ? <span><strong>{record.average_height_cm} cm</strong> recorded average height</span> : null}
        </div>
        {record.growth_snapshot ? <details className="growth-guide"><summary>Growth at this visit</summary><AutomaticGrowth snapshot={record.growth_snapshot} alive={record.alive_count} guide={false} /></details> : null}
        <p><strong>LGU actions:</strong> {record.actions_taken}</p>
        {record.reported_dead_count > 0 ? <p><strong>Cause of death:</strong> {record.death_reason || 'Not determined'}{record.death_reason_notes ? ` — ${record.death_reason_notes}` : ''}</p> : null}
        {record.location_review_required ? <p>Location matching needs review: this older visit has inconsistent death counts.</p> : <div>
          <p>{record.reported_dead_count ?? 0} reported dead · {record.located_dead_count ?? 0} located · {record.unlocated_dead_count ?? 0} still to locate</p>
          {record.reported_dead_count > 0 ? <button type="button" onClick={() => setEditing(record)}>{record.unlocated_dead_count > 0 ? 'Identify remaining locations' : 'Review death locations'}</button> : null}
        </div>}
        <small>Recorded by {record.inspector_name || 'LGU staff'}</small>
      </article>)}
      {loading ? <p role="status">Loading visit history...</p> : null}
      {error ? <div className="org-monitoring-message is-error" role="alert">{error} <button type="button" onClick={() => setRetry((value) => value + 1)}>Try again</button></div> : null}
      {!loading && !error && nextId !== null ? <button type="button" className="org-monitoring-load-older" onClick={() => setBeforeId(nextId)}>Load older visits</button> : null}
    </div>
  </Modal>;
}
