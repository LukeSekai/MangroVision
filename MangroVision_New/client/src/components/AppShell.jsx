import { useEffect, useState } from 'react';
import { useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import MapView from './MapView';
import Modal from './Modal';
import Logo from './Logo';
import ProcessingIndicator from './ProcessingIndicator';
import NotificationBell from './NotificationBell';
import { useAuthStore } from '../stores/authStore';
import { useMapStore } from '../stores/mapStore';
import './AppShell.css';

export default function AppShell({ children }) {
  const { pathname } = useLocation();
  const showMap = !['/monitoring', '/dashboard', '/account'].includes(pathname);
  const [mapVisited, setMapVisited] = useState(showMap);
  if (showMap && !mapVisited) setMapVisited(true);
  const user = useAuthStore((s) => s.user);
  const [welcomeOpen, setWelcomeOpen] = useState(false);

  useEffect(() => {
    let timer;
    const refresh = () => {
      window.clearTimeout(timer);
      timer = window.setTimeout(() => {
        if (!useAuthStore.getState().isAuthenticated || !showMap) return;
        const { fetchPoints, fetchStats, fetchZones } = useMapStore.getState();
        void fetchPoints();
        void fetchStats();
        void fetchZones();
      }, 50);
    };
    const reset = () => {
      window.clearTimeout(timer);
      useMapStore.getState().resetWorkspaceData();
    };
    window.addEventListener('mv:data-changed', refresh);
    window.addEventListener('mv:session-changed', reset);
    return () => {
      window.clearTimeout(timer);
      window.removeEventListener('mv:data-changed', refresh);
      window.removeEventListener('mv:session-changed', reset);
    };
  }, [showMap]);

  useEffect(() => {
    if (sessionStorage.getItem('mv_show_welcome') === '1') {
      const timer = window.setTimeout(() => setWelcomeOpen(true), 0);
      return () => window.clearTimeout(timer);
    }
  }, [user?.id]);

  const closeWelcome = () => {
    sessionStorage.removeItem('mv_show_welcome');
    setWelcomeOpen(false);
  };

  return (
    <div className="app-shell">
      <Sidebar />
      <main className="app-main">
        <header className="workspace-header">
          <span>MangroVision workspace</span>
          <NotificationBell />
        </header>
        {mapVisited && <div className="map-layer" style={{ display: showMap ? undefined : 'none' }}>
          <MapView />
        </div>}
        <div className="content-layer">
          {children}
        </div>
      </main>
      <Modal
        open={welcomeOpen}
        title={`Welcome back${user?.full_name ? `, ${user.full_name}` : ''}`}
        confirmLabel="Continue"
        cancelLabel=""
        onConfirm={closeWelcome}
        icon={<Logo variant="icon" size={36} alt="MangroVision" />}
      >
        <p>Your MangroVision workspace is ready.</p>
      </Modal>
      <ProcessingIndicator />
    </div>
  );
}
