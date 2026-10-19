import { useMemo, useState } from 'react';
import { GOOGLE_SATELLITE_URL, GOOGLE_SATELLITE_MAX_NATIVE_ZOOM } from '../config/googleBasemap';
import { ORTHOPHOTO_BOUNDS, ORTHOPHOTO_MAX_NATIVE_ZOOM, ORTHOPHOTO_TILE_URL } from '../config/mapTiles';
import { buildSiteMapPreview } from '../utils/siteMapPreview';

export default function SiteMapPreview({ site, name, points = [], large = false, frameGeometry }) {
  const mortalityWarning = site.properties?.mortality_warning?.active === true;
  const [failedTiles, setFailedTiles] = useState(new Set());
  const preview = useMemo(() => buildSiteMapPreview(site.geometry, {
    satellite: { url: GOOGLE_SATELLITE_URL, maxNativeZoom: GOOGLE_SATELLITE_MAX_NATIVE_ZOOM },
    orthophoto: { url: ORTHOPHOTO_TILE_URL, maxNativeZoom: ORTHOPHOTO_MAX_NATIVE_ZOOM, bounds: ORTHOPHOTO_BOUNDS },
    points, frameGeometry,
    width: large ? 800 : 320, height: large ? 450 : 180, padding: large ? 28 : 18,
  }), [site.geometry, points, large, frameGeometry]);
  const radius = preview ? Math.max(1.2, Math.min(2.6, (preview.zoom - 15) * 0.4 + 1.1)) : 0;
  return <div className={`monitoring-site-thumbnail monitoring-site-map-preview${large ? ' monitoring-site-map-large' : ''}`} role="img"
    aria-label={preview ? `Map of ${name} with its site boundary outlined and ${preview.markers.length} planting points${mortalityWarning ? '. Warning: High mortality.' : ''}` : `Map preview unavailable for ${name}`}>
    {preview ? <>
      {preview.tiles.filter((tile) => !failedTiles.has(tile.key)).map((tile) => (
        <img key={tile.key} src={tile.url} alt="" aria-hidden="true" className="monitoring-site-map-tile"
          loading="lazy" decoding="async" draggable="false" data-map-layer={tile.kind}
          style={{ left: `${tile.x / preview.width * 100}%`, top: `${tile.y / preview.height * 100}%`,
            width: `${tile.size / preview.width * 100}%`, height: `${tile.size / preview.height * 100}%` }}
          onError={() => setFailedTiles((current) => new Set([...current, tile.key]))} />
      ))}
      <svg viewBox={`0 0 ${preview.width} ${preview.height}`} aria-hidden="true" className="monitoring-site-map-outline">
        <path d={preview.path} fill="none" stroke="white" strokeWidth="5" strokeLinejoin="round" />
        <path d={preview.path} fill={mortalityWarning ? 'none' : '#22c55e'} fillOpacity="0.12" fillRule="evenodd"
          stroke={mortalityWarning ? '#d97706' : '#15803d'} strokeWidth="2.5" strokeLinejoin="round"
          strokeDasharray={mortalityWarning ? '8 5' : undefined} />
        {preview.markers.map((point) => <circle key={point.id} className="monitoring-site-map-point"
          data-point-id={point.id} data-point-status={point.status} cx={point.x} cy={point.y} r={radius}
          fill={point.color} fillOpacity="0.95" stroke="#17251b" strokeWidth="0.65" />)}
      </svg>
      {failedTiles.size === preview.tiles.length ? <span className="monitoring-site-map-unavailable">Map imagery unavailable</span> : null}
      <span className="monitoring-site-map-attribution">© Google</span>
    </> : <span className="monitoring-site-no-preview">Site boundary unavailable</span>}
  </div>;
}
