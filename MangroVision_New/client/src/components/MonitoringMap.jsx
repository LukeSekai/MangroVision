import { useEffect, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { ORTHOPHOTO_BOUNDS, ORTHOPHOTO_MAX_NATIVE_ZOOM, ORTHOPHOTO_TILE_URL } from '../config/mapTiles';
import { bindProjectSiteInfo } from '../utils/projectSiteInfo';
import './ProjectSiteInfo.css';

const API = import.meta.env.VITE_API_BASE || '';
const siteStyle = { color: '#15803d', weight: 2, fillColor: '#22c55e', fillOpacity: 0.16 };

// An independent reference map: no editing controls or shared map-store changes.
export default function MonitoringMap() {
  const container = useRef(null);
  const mapRef = useRef(null);
  const sitesLayer = useRef(null);
  const [sites, setSites] = useState([]);
  const [selected, setSelected] = useState('');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [retry, setRetry] = useState(0);

  useEffect(() => {
    const map = L.map(container.current, { maxZoom: 23, scrollWheelZoom: true }).fitBounds(ORTHOPHOTO_BOUNDS);
    mapRef.current = map;
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', {
      attribution: '&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a> contributors',
      maxNativeZoom: 19, maxZoom: 23,
    }).addTo(map);
    const orthophoto = L.tileLayer(ORTHOPHOTO_TILE_URL, {
      bounds: ORTHOPHOTO_BOUNDS, maxNativeZoom: ORTHOPHOTO_MAX_NATIVE_ZOOM,
      maxZoom: 23, minZoom: 10, noWrap: true,
    }).addTo(map);
    L.control.layers({}, { 'Drone imagery': orthophoto }, { position: 'topright' }).addTo(map);
    L.control.scale({ imperial: false }).addTo(map);
    sitesLayer.current = L.featureGroup().addTo(map);
    const observer = new ResizeObserver(() => map.invalidateSize({ pan: false }));
    observer.observe(container.current);
    return () => {
      observer.disconnect();
      map.remove();
      mapRef.current = null;
      sitesLayer.current = null;
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    async function loadSites() {
      setLoading(true);
      setError('');
      try {
        const response = await fetch(`${API}/api/project-sites`, { signal: controller.signal });
        const payload = await response.json();
        if (!response.ok || !Array.isArray(payload.features)) throw new Error('Could not load project sites. Please try again.');
        if (controller.signal.aborted) return;
        const valid = payload.features.filter((site) => site?.geometry && ['Polygon', 'MultiPolygon'].includes(site.geometry.type));
        setSites(valid);
        setSelected('');
      } catch (failure) {
        if (!controller.signal.aborted) setError(failure.message || 'Could not load project sites.');
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    void loadSites();
    return () => controller.abort();
  }, [retry]);

  useEffect(() => {
    const map = mapRef.current;
    const group = sitesLayer.current;
    if (!map || !group) return;
    group.clearLayers();
    let selectedLayer;
    sites.forEach((site, index) => {
      const key = String(index);
      try {
        const layer = L.geoJSON(site, { style: { ...siteStyle, weight: selected === key ? 4 : 2, fillOpacity: selected === key ? 0.25 : 0.1 } });
        bindProjectSiteInfo(layer, site);
        layer.on('click', () => setSelected(key));
        group.addLayer(layer);
        if (selected === key) selectedLayer = layer;
      } catch {
        // One malformed geometry must not stop the other sites from displaying.
      }
    });
    const bounds = (selectedLayer || group).getBounds();
    if (bounds.isValid()) {
      map.fitBounds(bounds, { padding: [32, 32], maxZoom: 20 });
      selectedLayer?.openPopup();
    }
  }, [sites, selected]);

  return <div className="org-monitoring-reference-map">
    <div className="org-monitoring-map-toolbar">
      <label>Project site
        <select value={selected} onChange={(event) => setSelected(event.target.value)} disabled={loading || !sites.length}>
          <option value="">All project sites</option>
          {sites.map((site, index) => <option key={index} value={String(index)}>
            {site.properties?.name || `Project site ${index + 1}`}{site.properties?.organization_name ? ` — ${site.properties.organization_name}` : ''}
          </option>)}
        </select>
      </label>
      <p>Hover a site to see its information, or select or tap its boundary for details.</p>
    </div>
    {loading ? <p role="status">Loading project sites...</p> : null}
    {error ? <div className="org-monitoring-message is-error" role="alert">{error} <button type="button" onClick={() => setRetry((value) => value + 1)}>Try again</button></div> : null}
    {!loading && !error && !sites.length ? <p role="status">No project sites have been mapped yet.</p> : null}
    <div ref={container} className="org-monitoring-map-canvas" aria-label="Project site reference map" />
  </div>;
}
