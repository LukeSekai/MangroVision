import { useState } from 'react';
import { Link, NavLink, useLocation } from 'react-router-dom';
import { useAuthStore } from '../stores/authStore';
import { PLANTING_TOOLS } from './plantingWorkspaceContext';
import Logo from './Logo';
import LogoutConfirmation from './LogoutConfirmation';
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
    to: '/organizations',
    label: 'Organizations',
    icon: (
      <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <circle cx="9" cy="8" r="3" />
        <path d="M3 21v-2a6 6 0 0 1 12 0v2M16 5a3 3 0 0 1 0 6M21 21v-2a6 6 0 0 0-4-5.66" />
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

export default function Sidebar() {
  const [logoutOpen, setLogoutOpen] = useState(false);
  const logout = useAuthStore((s) => s.logout);
  const { pathname } = useLocation();
  const inPlantingWorkspace = PLANTING_TOOLS.some((tool) => tool.to === pathname);

  return (
    <aside className="sidebar">
      <div className="sidebar-top">
        <Link className="sidebar-logo" to="/map" aria-label="MangroVision — Planting Map" title="Open Planting Map">
          <span className="sidebar-logo-chip" aria-hidden="true">
            <Logo variant="icon" size={28} className="sidebar-logo-icon" />
          </span>
          <span className="sidebar-brand-name">MangroVision</span>
        </Link>

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
        <button type="button" className="sidebar-nav-item" onClick={() => setLogoutOpen(true)} title="Sign out">
          <svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
            <polyline points="16 17 21 12 16 7" />
            <line x1="21" y1="12" x2="9" y2="12" />
          </svg>
          <span className="sidebar-nav-label">Sign out</span>
        </button>
      </div>
      <LogoutConfirmation open={logoutOpen} onCancel={() => setLogoutOpen(false)} onConfirm={() => {
        logout();
        setLogoutOpen(false);
      }} />
    </aside>
  );
}
