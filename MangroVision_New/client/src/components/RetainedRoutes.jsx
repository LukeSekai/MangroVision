import { Activity, useState } from 'react';
import { Routes, useLocation } from 'react-router-dom';

// Keep only visited pages. Activity preserves their DOM and state while
// pausing effects, so hidden pages cannot take over the shared map or dialogs.
export default function RetainedRoutes({ paths, children }) {
  const location = useLocation();
  const retain = paths.includes(location.pathname);
  const [locations, setLocations] = useState(() => retain ? [location] : []);
  const saved = locations.find((entry) => entry.pathname === location.pathname);
  if (retain && saved !== location) {
    setLocations((current) => saved
      ? current.map((entry) => entry.pathname === location.pathname ? location : entry)
      : [...current, location]);
  }

  return <>
    {locations.map((entry) => <Activity key={entry.pathname}
      mode={entry.pathname === location.pathname ? 'visible' : 'hidden'}>
      <Routes location={entry}>{children}</Routes>
    </Activity>)}
    {!retain && <Routes>{children}</Routes>}
  </>;
}
