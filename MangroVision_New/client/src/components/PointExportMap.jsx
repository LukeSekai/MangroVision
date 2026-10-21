import { useEffect, useMemo, useRef, useState } from 'react';
import L from 'leaflet';
import { cachedTileLayer } from '../utils/cachedTileLayer';
import { createGoogleSatelliteLayer } from '../config/googleBasemap';
import { ORTHOPHOTO_TILE_URL, ORTHOPHOTO_BOUNDS, ORTHOPHOTO_MAX_NATIVE_ZOOM } from '../config/mapTiles';
import { pointStatusLabel } from '../utils/pointStatus';
import { createPointCanvas } from '../utils/pointCanvas';
import exportPalette from '../config/pointExportStyles.json';
import { pointExportSpeciesLabel } from '../utils/pointExports';
import 'leaflet/dist/leaflet.css';

const radiusAt = (zoom) => zoom >= 21 ? 3 : zoom >= 19 ? 2 : 1.4;
const speciesKey = (value) => String(value || '').trim().toLowerCase();
const speciesLabel = (value) => value ? pointExportSpeciesLabel(speciesKey(value)) : 'Species not recorded';
const eroded = (point) => Boolean(point.eroded_unavailable || ['unavailable', 'eroded_unavailable'].includes(point.status));
const colorKey = (point) => {
  if (point.status === 'dead') return 'dead';
  if (eroded(point)) return 'eroded';
  if (['planted', 'completed'].includes(point.status)) return 'planted';
  const key = speciesKey(point.species);
  return Object.hasOwn(exportPalette.species, key) ? key : 'unknown';
};
const colorDefinition = (key) => exportPalette.statuses[key] || exportPalette.species[key] || exportPalette.unknown;
const colorRank = { dead: 3, eroded: 2, planted: 1 };
export default function PointExportMap({ waypoints, emptyMessage, onReady }) {
  const container = useRef(null);
  const mapRef = useRef(null);
  const groupRef = useRef(null);
  const boundsRef = useRef(null);
  const [imageryError, setImageryError] = useState(false);
  const colors = useMemo(() => {
    const selected = new Set(waypoints.map(colorKey));
    return [...Object.entries(exportPalette.statuses), ...Object.entries(exportPalette.species), ['unknown', exportPalette.unknown]]
      .filter(([key]) => selected.has(key));
  }, [waypoints]);

  useEffect(() => {
    // This read-only map never changes workspace selection or saved position.
    const renderer = createPointCanvas();
    const map = L.map(container.current, { renderer, preferCanvas: true, maxZoom: 23, scrollWheelZoom: true });
    mapRef.current = map;
    map.fitBounds(ORTHOPHOTO_BOUNDS);
    const reportTileError = () => setImageryError(true);
    const base = createGoogleSatelliteLayer({ maxZoom: 23 }).on('tileerror', reportTileError).addTo(map);
    const ortho = cachedTileLayer(ORTHOPHOTO_TILE_URL, {
      bounds: ORTHOPHOTO_BOUNDS, maxNativeZoom: ORTHOPHOTO_MAX_NATIVE_ZOOM,
      maxZoom: 23, minZoom: 10, noWrap: true,
    }).on('tileerror', reportTileError).addTo(map);
    groupRef.current = L.featureGroup().addTo(map);
    L.control.scale({ imperial: false }).addTo(map);
    map.on('zoomend', () => groupRef.current?.eachLayer((marker) => marker.setRadius(radiusAt(map.getZoom()))));
    const observer = new ResizeObserver(() => map.invalidateSize({ pan: false }));
    observer.observe(container.current);
    const frame = requestAnimationFrame(() => map.invalidateSize({ pan: false }));
    return () => {
      cancelAnimationFrame(frame);
      observer.disconnect();
      base.off('tileerror', reportTileError);
      ortho.off('tileerror', reportTileError);
      map.scrollWheelZoom.disable();
      // Leaflet removes the wheel listener without clearing its queued zoom.
      clearTimeout(map.scrollWheelZoom._timer);
      map.stop();
      // Remove paths while their canvas renderer is still attached. Removing
      // the renderer first can let path teardown schedule a draw on a dead canvas.
      groupRef.current?.remove();
      L.Util.cancelAnimFrame(renderer._redrawRequest);
      renderer.remove();
      map.remove();
      mapRef.current = null;
      groupRef.current = null;
      boundsRef.current = null;
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current, group = groupRef.current;
    group.clearLayers();
    const coordinates = [];
    // Draw higher-priority conditions last so ordinary dots do not cover them.
    const ordered = [...waypoints].sort((a, b) => (colorRank[colorKey(a)] || 0) - (colorRank[colorKey(b)] || 0));
    for (const point of ordered) {
      const position = [point.lat, point.lon];
      coordinates.push(position);
      const label = document.createElement('span');
      const insideErosion = eroded(point);
      label.textContent = [point.name, speciesLabel(point.species), pointStatusLabel(point.status),
        insideErosion ? point.availability_reason || 'Inside an eroded zone' : null].filter(Boolean).join(' · ');
      L.circleMarker(position, {
        radius: radiusAt(map.getZoom()),
        color: exportPalette.outline_color, weight: 0.5,
        fillColor: colorDefinition(colorKey(point)).color, fillOpacity: 1,
        mvPointId: point.point_id,
      }).bindTooltip(label).addTo(group);
    }
    boundsRef.current = coordinates.length ? L.latLngBounds(coordinates) : null;
    if (boundsRef.current) map.fitBounds(boundsRef.current, { padding: [28, 28], maxZoom: 22, animate: false });
    else map.fitBounds(ORTHOPHOTO_BOUNDS, { animate: false });
    onReady?.(waypoints);
  }, [waypoints, onReady]);

  const fit = () => {
    if (boundsRef.current) mapRef.current?.fitBounds(boundsRef.current, { padding: [28, 28], maxZoom: 22 });
  };

  return <>
    <div className="point-export-map-frame">
      <div ref={container} className="point-export-map" role="region" aria-label={`Map of ${waypoints.length.toLocaleString()} points selected for export`} />
      {waypoints.length > 0 && <button type="button" className="point-export-fit" onClick={fit}>Fit selected points</button>}
      {!waypoints.length && <div className="point-export-map-empty" role="status">{emptyMessage}</div>}
    </div>
    {imageryError && <p className="point-export-hint" role="status">Some map imagery could not load. Point locations are still shown.</p>}
    {colors.length > 0 && <ul className="point-export-legend" aria-label="Export point colors">{colors.map(([key, definition]) => <li key={key}>
      <span aria-hidden="true" style={{ background: definition.color }} />{definition.label}
    </li>)}</ul>}
  </>;
}
