import { Link, useLocation } from 'react-router-dom';
import { mapReturnTarget } from '../utils/mapNavigation';
import './MapReturnNavigation.css';

export default function MapReturnNavigation() {
  const target = mapReturnTarget(useLocation());
  if (!target) return null;
  return <nav className="map-return-navigation" aria-label="Return from map">
    <Link to={{ pathname: target.pathname, search: target.search, hash: target.hash }} state={target.state}>
      <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="m12 5-7 7 7 7M5 12h14" /></svg>
      <span>Back to {target.label}</span>
    </Link>
  </nav>;
}
