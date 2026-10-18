import { NavLink } from 'react-router-dom';
import MapLink from './MapLink';
import '../pages/OrganizationMonitoring.css';

export default function MonitoringHeader({ title, description, mapTo = '/monitoring/map', children }) {
  return <>
    <header className="org-monitoring-page-header">
      <div><span className="org-monitoring-eyebrow">Mangrove care</span><h1>{title}</h1><p>{description}</p></div>
      <div className="org-monitoring-page-actions">
        {children}
        <MapLink className="is-primary" to={mapTo}>
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="m3 5 6-2 6 2 6-2v16l-6 2-6-2-6 2V5Zm6-2v16m6-14v16" /></svg>
          Show map
        </MapLink>
      </div>
    </header>
    <nav className="monitoring-section-navigation" aria-label="Monitoring sections">
      <NavLink to="/monitoring" end>Monitoring visits</NavLink>
      <NavLink to="/monitoring/replanting">LGU Replanting</NavLink>
    </nav>
  </>;
}
