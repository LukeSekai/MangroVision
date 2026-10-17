import { useEffect, useState } from 'react';

export default function FieldNavigationNotice({ route, onStop, gpsError }) {
  const [hintOpen, setHintOpen] = useState(true);

  useEffect(() => {
    const timer = window.setTimeout(() => setHintOpen(false), 7000);
    return () => window.clearTimeout(timer);
  }, []);

  return (
    <div className="field-navigation">
      {hintOpen && <div className="field-navigation-hint" role="status">
        <span>
          {route.route_source === 'partial_route' && 'Road directions to the site are unavailable. '}
          {route.entrance
            ? `Follow the white bridge, then the orange line from Point #${route.pointNum} through your assigned points. Use marked planting lanes.`
            : `Follow the marked path to Point #${route.pointNum}.`}
        </span>
        <button type="button" onClick={() => setHintOpen(false)} aria-label="Dismiss path hint">×</button>
      </div>}
      {gpsError && <div className="field-navigation-gps-error" role="status">{gpsError}</div>}
      <div className="field-navigation-control" aria-label="Active navigation">
        <span>Point #{route.pointNum}</span>
        <button type="button" onClick={onStop}>Stop navigation</button>
      </div>
    </div>
  );
}
