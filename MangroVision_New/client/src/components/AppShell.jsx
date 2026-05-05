import { useLocation } from 'react-router-dom';
import Sidebar from './Sidebar';
import MapView from './MapView';
import './AppShell.css';

export default function AppShell({ children }) {
  const location = useLocation();

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
    </div>
  );
}
