import { useCallback, useEffect, useState } from 'react';
import './ActivityFeed.css';

const API = import.meta.env.VITE_API_BASE || '';

function actorLabel(item) {
  if (item.actor_type === 'staff') return item.staff_name || 'LGU staff';
  if (item.actor_type === 'planter') {
    const name = item.planter_name || item.organization_name || 'Organization participant';
    return item.participant_slot ? `${name} · Participant ${item.participant_slot}` : name;
  }
  return 'System';
}

export default function ActivityFeed({ scope = 'staff' }) {
  const [items, setItems] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [hasMore, setHasMore] = useState(false);

  const load = useCallback(async (beforeId = null) => {
    setLoading(true);
    setError('');
    try {
      const query = beforeId ? `?before_id=${beforeId}` : '';
      const response = await fetch(`${API}/api/activity/${scope}${query}`, { cache: 'no-store' });
      const payload = await response.json();
      if (!response.ok) throw new Error(payload.detail || 'Activity could not be loaded.');
      const next = payload.items || [];
      setItems((current) => beforeId ? [...current, ...next] : next);
      setHasMore(next.length === 50);
    } catch (loadError) {
      setError(loadError.message || 'Activity could not be loaded.');
    } finally {
      setLoading(false);
    }
  }, [scope]);

  useEffect(() => {
    const timer = window.setTimeout(() => { void load(); }, 0);
    return () => window.clearTimeout(timer);
  }, [load]);

  return <section className="activity-feed" aria-labelledby={`activity-feed-title-${scope}`}>
    <div className="activity-feed-toolbar">
      <div>
        <span className="activity-feed-eyebrow">History</span>
        <h2 id={`activity-feed-title-${scope}`}>Recent activity</h2>
        <p>Newest records appear first.</p>
      </div>
      <button type="button" onClick={() => load()} disabled={loading}>{loading ? 'Refreshing…' : 'Refresh activity'}</button>
    </div>
    {error && <p role="alert" className="activity-feed-error">{error}</p>}
    {loading && items.length === 0 && <p className="activity-feed-state" role="status">Loading activity…</p>}
    {!loading && items.length === 0 && !error && <p className="activity-feed-state">No activity has been recorded yet.</p>}
    {items.length > 0 && <ol className="activity-feed-list">{items.map((item) => <li key={item.id}>
      <div className="activity-feed-entry">
        <span className="activity-feed-marker" aria-hidden="true" />
        <div className="activity-feed-content">
          <strong>{item.summary}</strong>
          <div className="activity-feed-meta">
            <span>{actorLabel(item)}</span>
            {item.actor_type === 'planter' && item.planter_name && item.organization_name && <span>{item.organization_name}</span>}
            {item.project_site_name && <span>{item.project_site_name}</span>}
            {item.point_num != null && <span>Point {item.point_num}</span>}
          </div>
        </div>
        <time dateTime={item.created_at}>{new Date(item.created_at).toLocaleString(undefined, { dateStyle: 'medium', timeStyle: 'short' })}</time>
      </div>
    </li>)}</ol>}
    {hasMore && <div className="activity-feed-footer"><button type="button" className="activity-feed-more" onClick={() => load(items.at(-1)?.id)} disabled={loading}>{loading ? 'Loading…' : 'Load more activity'}</button></div>}
  </section>;
}
