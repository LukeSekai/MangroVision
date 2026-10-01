import { useCallback, useEffect, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import './NotificationBell.css';

const API = import.meta.env.VITE_API_BASE || '';
const PAGE_SIZE = 5;

function notificationDate(value) {
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return '';
  return new Intl.DateTimeFormat('en-PH', {
    timeZone: 'Asia/Manila', month: 'short', day: 'numeric', year: 'numeric',
  }).format(date);
}

export default function NotificationBell() {
  const navigate = useNavigate();
  const rootRef = useRef(null);
  const lastIds = useRef(new Set());
  const loadedOnce = useRef(false);
  const lastRefreshDay = useRef('');
  const [items, setItems] = useState([]);
  const [unread, setUnread] = useState(0);
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState('unread');
  const [visibleCount, setVisibleCount] = useState(PAGE_SIZE);
  const [toast, setToast] = useState(null);

  const load = useCallback(async (sync = false) => {
    try {
      const response = await fetch(`${API}/api/notifications${sync ? '/refresh' : ''}`, {
        method: sync ? 'POST' : 'GET', cache: 'no-store',
      });
      if (!response.ok) return;
      const payload = await response.json();
      const next = Array.isArray(payload.items) ? payload.items : [];
      const fresh = loadedOnce.current
        ? next.find((item) => !item.read_at && !lastIds.current.has(item.id))
        : null;
      if (fresh) setToast(fresh);
      loadedOnce.current = true;
      lastIds.current = new Set(next.map((item) => item.id));
      setItems(next);
      setUnread(payload.unread_count || 0);
    } catch {
      // Keep the workspace usable during a transient notification failure.
    }
  }, []);

  useEffect(() => {
    const tick = () => {
      const parts = Object.fromEntries(new Intl.DateTimeFormat('en-US', {
        timeZone: 'Asia/Manila', year: 'numeric', month: '2-digit', day: '2-digit',
      }).formatToParts(new Date()).map((part) => [part.type, part.value]));
      const today = `${parts.year}-${parts.month}-${parts.day}`;
      const sync = lastRefreshDay.current !== today;
      lastRefreshDay.current = today;
      void load(sync);
    };
    tick();
    const timer = window.setInterval(tick, 60_000);
    window.addEventListener('focus', tick);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener('focus', tick);
    };
  }, [load]);

  useEffect(() => {
    if (!toast) return undefined;
    const timer = window.setTimeout(() => setToast(null), 7000);
    return () => window.clearTimeout(timer);
  }, [toast]);

  useEffect(() => {
    if (!open) return undefined;
    const dismiss = (event) => {
      if (event.key === 'Escape') setOpen(false);
      if (event.type === 'pointerdown' && !rootRef.current?.contains(event.target)) setOpen(false);
    };
    document.addEventListener('keydown', dismiss);
    document.addEventListener('pointerdown', dismiss);
    return () => {
      document.removeEventListener('keydown', dismiss);
      document.removeEventListener('pointerdown', dismiss);
    };
  }, [open]);

  const select = async (item) => {
    setOpen(false);
    setToast(null);
    if (!item.read_at) {
      try {
        const response = await fetch(`${API}/api/notifications/${item.id}/read`, { method: 'POST' });
        if (response.ok) {
          setItems((current) => current.map((row) => row.id === item.id
            ? { ...row, read_at: new Date().toISOString() } : row));
          setUnread((current) => Math.max(0, current - 1));
        }
      } catch { /* Navigation still works if marking read fails. */ }
    }
    if (item.target_path?.startsWith('/')) navigate(item.target_path);
  };

  const filteredItems = filter === 'unread' ? items.filter((item) => !item.read_at) : items;
  const visibleItems = filteredItems.slice(0, visibleCount);

  return (
    <div className="notification-root" ref={rootRef}>
      <button type="button" className="notification-button" aria-label={`Notifications, ${unread} unread`}
        aria-haspopup="dialog" aria-expanded={open} onClick={() => setOpen((value) => !value)}>
        <svg width="21" height="21" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.9" aria-hidden="true">
          <path d="M18 8a6 6 0 0 0-12 0c0 7-3 7-3 9h18c0-2-3-2-3-9M10 21h4" />
        </svg>
        {unread > 0 && <span className="notification-count">{unread > 99 ? '99+' : unread}</span>}
      </button>
      {open && <div className="notification-panel" role="dialog" aria-label="Notifications">
        <div className="notification-panel-head">
          <div><strong>Notifications</strong><span>{unread ? `${unread} need your attention` : 'All caught up'}</span></div>
          <button type="button" className="notification-panel-close" aria-label="Close notifications" onClick={() => setOpen(false)}>×</button>
        </div>
        <div className="notification-filters" aria-label="Notification view">
          <button type="button" className={filter === 'unread' ? 'is-active' : ''} aria-pressed={filter === 'unread'}
            onClick={() => { setFilter('unread'); setVisibleCount(PAGE_SIZE); }}>Unread <span>{unread}</span></button>
          <button type="button" className={filter === 'all' ? 'is-active' : ''} aria-pressed={filter === 'all'}
            onClick={() => { setFilter('all'); setVisibleCount(PAGE_SIZE); }}>All <span>{items.length}</span></button>
        </div>
        <div className="notification-list">
          {visibleItems.length === 0 && <div className="notification-empty">
            <span aria-hidden="true">✓</span>
            <strong>{filter === 'unread' ? 'You’re all caught up' : 'No notifications yet'}</strong>
            <p>{filter === 'unread' ? 'New updates will appear here.' : 'Updates about planting and monitoring will appear here.'}</p>
          </div>}
          {visibleItems.map((item) => <button type="button" key={item.id}
            className={`notification-item${item.read_at ? '' : ' is-unread'}`} onClick={() => select(item)}>
            <span className="notification-item-dot" aria-hidden="true" />
            <span className="notification-item-content">
              <span className="notification-item-title">{item.title}</span>
              <span className="notification-item-preview">{item.body}</span>
              <span className="notification-item-date">{notificationDate(item.created_at)}</span>
            </span>
            <span className="notification-item-arrow" aria-hidden="true">›</span>
          </button>)}
        </div>
        {filteredItems.length > visibleCount && <button type="button" className="notification-more"
          onClick={() => setVisibleCount((current) => current + PAGE_SIZE)}>
          Show more ({filteredItems.length - visibleCount} remaining)
        </button>}
      </div>}
      {toast && <div className="notification-toast" role="status">
        <button type="button" className="notification-toast-close" aria-label="Dismiss notification" onClick={() => setToast(null)}>×</button>
        <button type="button" className="notification-toast-body" onClick={() => select(toast)}>
          <small>New notification</small><strong>{toast.title}</strong>
        </button>
      </div>}
    </div>
  );
}
