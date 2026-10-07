import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import './AppointmentNotice.css';

const API = import.meta.env.VITE_API_BASE || '';

export default function AppointmentNotice() {
  const role = useAuthStore((state) => state.user?.role?.toLowerCase());
  const allowed = ['admin', 'lgu', 'planner'].includes(role);
  const [count, setCount] = useState(null);
  const [error, setError] = useState(false);
  useEffect(() => {
    if (!allowed) return undefined;
    const controller = new AbortController();
    let busy = false;
    const load = async () => {
      if (busy) return;
      busy = true;
      try {
        const response = await fetch(`${API}/api/like-appointments/summary`, { signal: controller.signal, cache: 'no-store' });
        if (!response.ok) throw new Error('Could not load appointments.');
        const data = await response.json();
        if (!controller.signal.aborted) { setCount(data.pending_count); setError(false); }
      } catch (issue) {
        if (issue.name !== 'AbortError') setError(true);
      } finally { busy = false; }
    };
    void load();
    const timer = window.setInterval(load, 30000);
    window.addEventListener('focus', load);
    window.addEventListener('mv:data-changed', load);
    return () => {
      controller.abort(); window.clearInterval(timer);
      window.removeEventListener('focus', load); window.removeEventListener('mv:data-changed', load);
    };
  }, [allowed]);
  if (!allowed) return null;
  return <Link className={`appointment-notice${count > 0 ? ' has-pending' : ''}`} to="/scheduling?requests=pending">
    <span className="appointment-notice-icon" aria-hidden="true">↗</span>
    <span><strong>LIKE appointment requests</strong><small>{error ? 'Count unavailable · Open Scheduling to review' : count === null ? 'Checking pending requests…' : `${count} pending ${count === 1 ? 'request' : 'requests'} · Review in Scheduling`}</small></span>
    <span className="appointment-notice-count" aria-label={error || count === null ? 'Pending count unavailable' : `${count} pending`}>{error || count === null ? '–' : count}</span>
  </Link>;
}
