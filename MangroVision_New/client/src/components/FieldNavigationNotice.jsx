import { useEffect, useState } from 'react';

export default function FieldNavigationNotice({ route, onStop, gpsError, onRetry, retryBusy = false }) {
  const [hintOpen, setHintOpen] = useState(true);
  const roadUnavailable = Boolean(route.road_route_error) || ['partial_route', 'point_guidance'].includes(route.route_source);

  useEffect(() => {
    const timer = window.setTimeout(() => setHintOpen(false), 7000);
    return () => window.clearTimeout(timer);
  }, []);

  return (
    <div className="field-navigation">
      {hintOpen && <div className="field-navigation-hint" role="status">
        <span>
          {route.entrance
            ? `Follow the white bridge, then the orange line from Point #${route.pointNum} through your assigned points. Use marked planting lanes.`
            : `Follow the marked path to Point #${route.pointNum}.`}
        </span>
        <button type="button" onClick={() => setHintOpen(false)} aria-label="Dismiss path hint">×</button>
      </div>}
      {roadUnavailable && <div className="field-navigation-road-error" role="alert">
        <span>{route.road_route_error?.message || 'Road directions from your location are unavailable.'}
          {' '}{route.entrance ? 'Only the mapped site approach is shown.' : 'No road route to this point is shown.'}
        </span>
        {onRetry && <button type="button" onClick={onRetry} disabled={retryBusy}>
          {retryBusy ? 'Retrying…' : 'Retry road route'}
        </button>}
      </div>}
      {gpsError && <div className="field-navigation-gps-error" role="status">{gpsError}</div>}
      <div className="field-navigation-control" aria-label="Active navigation">
        <span>Point #{route.pointNum}</span>
        <button type="button" onClick={onStop}>Stop navigation</button>
      </div>
    </div>
  );
}
