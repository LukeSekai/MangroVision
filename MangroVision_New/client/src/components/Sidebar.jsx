import { useEffect, useMemo, useState } from 'react';
import { NavLink, useLocation } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import { PLANTING_TOOLS } from './plantingWorkspaceContext';
import Logo from './Logo';
import './Sidebar.css';

const NAV_ITEMS = [
  {
    to: '/dashboard',
    label: 'Dashboard',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="3" width="7" height="9" rx="1" />
        <rect x="14" y="3" width="7" height="5" rx="1" />
        <rect x="14" y="12" width="7" height="9" rx="1" />
        <rect x="3" y="16" width="7" height="5" rx="1" />
      </svg>
    ),
  },
  {
    to: '/map',
    label: 'Planting Map',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <polygon points="1 6 1 22 8 18 16 22 23 18 23 2 16 6 8 2 1 6" />
        <line x1="8" y1="2" x2="8" y2="18" />
        <line x1="16" y1="6" x2="16" y2="22" />
      </svg>
    ),
  },
  {
    to: '/scheduling',
    label: 'Scheduling',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <rect x="3" y="5" width="18" height="16" rx="2" />
        <line x1="16" y1="3" x2="16" y2="7" />
        <line x1="8" y1="3" x2="8" y2="7" />
        <line x1="3" y1="11" x2="21" y2="11" />
        <path d="m9 16 2 2 4-4" />
      </svg>
    ),
  },
  {
    to: '/monitoring',
    label: 'Monitoring',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M22 12h-4l-3 9L9 3l-3 9H2" />
      </svg>
    ),
  },
  {
    to: '/activity',
    label: 'Activity Logs',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
        <path d="M4 4h16v16H4zM8 9h8M8 13h8M8 17h5" />
      </svg>
    ),
  },
];

function SidebarClock() {
  const [now, setNow] = useState(() => new Date());

  useEffect(() => {
    const timer = window.setInterval(() => setNow(new Date()), 1000);
    return () => window.clearInterval(timer);
  }, []);

  const dateLabel = useMemo(
    () => now.toLocaleDateString(undefined, {
      weekday: 'short',
      month: 'short',
      day: 'numeric',
    }),
    [now],
  );
  const timeLabel = useMemo(
    () => now.toLocaleTimeString(undefined, {
      hour: 'numeric',
      minute: '2-digit',
      second: '2-digit',
    }),
    [now],
  );

  return (
    <div className="sidebar-clock" title={`${dateLabel} ${timeLabel}`}>
      <div className="sidebar-clock-icon" aria-hidden="true">
        <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
          <circle cx="12" cy="12" r="9" />
          <path d="M12 7v5l3 2" />
        </svg>
      </div>
      <div className="sidebar-clock-text">
        <span className="sidebar-clock-date">{dateLabel}</span>
        <span className="sidebar-clock-time">{timeLabel}</span>
      </div>
    </div>
  );
}

export default function Sidebar() {
  const user = useAuthStore((s) => s.user);
  const logout = useAuthStore((s) => s.logout);
  const { pathname } = useLocation();
  const inPlantingWorkspace = PLANTING_TOOLS.some((tool) => tool.to === pathname);

  return (
    <aside className="sidebar">
      <div className="sidebar-top">
        <div className="sidebar-logo">
          <span className="sidebar-logo-chip" aria-hidden="true">
            <Logo variant="icon" size={28} className="sidebar-logo-icon" />
          </span>
          <span className="sidebar-brand-name">MangroVision</span>
        </div>

        <nav className="sidebar-nav">
          {NAV_ITEMS.map((item) => {
            return (
              <NavLink
                key={item.to}
                to={item.to}
                end={item.to === '/map'}
                aria-current={item.to === '/map' && inPlantingWorkspace ? 'page' : undefined}
                className={({ isActive }) =>
                  `sidebar-nav-item ${isActive || (item.to === '/map' && inPlantingWorkspace) ? 'active' : ''}`
                }
                title={item.label}
              >
                {item.icon}
                <span className="sidebar-nav-label">{item.label}</span>
              </NavLink>
            );
          })}
        </nav>
      </div>

      <div className="sidebar-bottom">
        <SidebarClock />
        <NavLink to="/account" className="sidebar-user" title="Account settings" aria-label="Account settings">
          <div className="sidebar-avatar">
            {user?.full_name?.charAt(0)?.toUpperCase() || 'U'}
          </div>
          <span className="sidebar-user-name">{user?.full_name}</span>
        </NavLink>
        <button className="sidebar-nav-item" onClick={logout} title="Sign out">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
            <polyline points="16 17 21 12 16 7" />
            <line x1="21" y1="12" x2="9" y2="12" />
          </svg>
          <span className="sidebar-nav-label">Sign out</span>
        </button>
      </div>
    </aside>
  );
}
