import { useEffect, useState } from 'react';
import Sidebar from './Sidebar';
import MapView from './MapView';
import Modal from './Modal';
import ProcessingIndicator from './ProcessingIndicator';
import { useAuthStore } from '../stores/authStore';
import './AppShell.css';

export default function AppShell({ children }) {
  const user = useAuthStore((s) => s.user);
  const [welcomeOpen, setWelcomeOpen] = useState(false);

  useEffect(() => {
    if (sessionStorage.getItem('mv_show_welcome') === '1') {
      setWelcomeOpen(true);
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
        <div className="map-layer">
          <MapView />
        </div>
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
      >
        <p>Your MangroVision workspace is ready.</p>
      </Modal>
      <ProcessingIndicator />
    </div>
  );
}
