import { useEffect, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { useMapStore } from '../stores/mapStore';
import { ORTHOPHOTO_MAX_NATIVE_ZOOM, ORTHOPHOTO_TILE_URL } from '../config/mapTiles';
import './MapView.css';

// Fix Leaflet default icon paths
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// Admin-side palette. Mirrors the planter view so the same point reads the
// same colour on both maps:
//   - planned   = green  (no planter assigned yet)
//   - assigned  = blue   (assigned but not yet planted)
//   - planted   = yellow (planter completed this point in the field)
//   - completed = yellow (DB enum equivalent of planted; still distinct from
//                         the never-touched "planned" green so reviewers can
//                         see at a glance which points are done)
const STATUS_COLORS = {
  planned: '#16a34a',
  assigned: '#2563eb',
  planted: '#eab308',
  completed: '#eab308',
  skipped: '#9ca3af',
};

const PREVIEW_COLORS = {
  safe: '#0f9d58',
  forbidden: '#c62828',
  eroded: '#ef6c00',
  canopy: '#7c3aed',
};

// Keep admin map point sizing in sync with the Delete Points map so points
// don't visually jump larger/smaller while moving between sections.
const getPointRadius = (zoom, selected = false) => {
  const base = zoom >= 22
    ? 3.5
    : zoom >= 21
      ? 2.4
      : zoom >= 20
        ? 1.6
        : zoom >= 19
          ? 1.05
          : zoom >= 18
            ? 0.8
            : zoom >= 17
              ? 0.65
              : 0.55;
  return selected ? Math.max(base + 1.2, base * 1.8) : base;
};

const getPointWeight = (zoom, selected = false) => {
  if (selected) return zoom >= 20 ? 1.8 : 1.2;
  if (zoom >= 22) return 1.1;
  if (zoom >= 21) return 0.75;
  if (zoom >= 20) return 0.45;
  if (zoom >= 19) return 0.2;
  return 0;
};

const getPreviewFilteredRadius = (zoom) => Math.max(0.6, getPointRadius(zoom) - 0.35);
const getAnalysisPointRadius = (zoom) => Math.max(0.45, getPointRadius(zoom) * 0.45);

export default function MapView() {
  const location = useLocation();
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const layersRef = useRef({});
  const pointsFingerprintRef = useRef('');
  // True while we're applying an external store change to the map. Lets the
  // moveend handler ignore that programmatic move so we don't write the same
  // view back to the store (which would cause a render loop).
  const skipNextSyncRef = useRef(false);

  const center = useMapStore((s) => s.center);
  const zoom = useMapStore((s) => s.zoom);
  const points = useMapStore((s) => s.points);
  const forbiddenZones = useMapStore((s) => s.forbiddenZones);
  const erodedZones = useMapStore((s) => s.erodedZones);
  const setSelectedPoint = useMapStore((s) => s.setSelectedPoint);
  const layerVisibility = useMapStore((s) => s.layerVisibility);
  const currentAnalysis = useMapStore((s) => s.currentAnalysis);
  const deletionSelectedPointIds = useMapStore((s) => s.deletionSelectedPointIds);
  const toggleDeletionPoint = useMapStore((s) => s.toggleDeletionPoint);
  const isPointDeletionMode = location.pathname === '/points';

  // Initialize map
  useEffect(() => {
    if (mapRef.current || !containerRef.current) return undefined;

    // Read the initial view once. The map writes user pans/zooms back into
    // mapStore, so subscribing this initialization effect to center/zoom would
    // tear Leaflet down and recreate it on every interaction.
    const initial = useMapStore.getState();

    const map = L.map(containerRef.current, {
      center: [initial.center[1], initial.center[0]],
      zoom: initial.zoom,
      maxZoom: 24,
      zoomControl: false,
      attributionControl: false,
    });

    // Tile layers
    const satellite = L.tileLayer(
      'https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
      { subdomains: '0123', maxZoom: 21, attribution: '&copy; Google' }
    );
    const osm = L.tileLayer(
      'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
      { maxZoom: 19, attribution: '&copy; OpenStreetMap' }
    );
    const orthophoto = L.tileLayer(
      ORTHOPHOTO_TILE_URL,
      { maxZoom: 24, maxNativeZoom: ORTHOPHOTO_MAX_NATIVE_ZOOM, opacity: 1.0, errorTileUrl: '', minZoom: 10 }
    );

    satellite.addTo(map);
    orthophoto.addTo(map);

    L.control.layers(
      { 'Satellite': satellite, 'OpenStreetMap': osm },
      { 'Orthophoto': orthophoto },
      { position: 'topright', collapsed: true }
    ).addTo(map);

    L.control.zoom({ position: 'bottomright' }).addTo(map);
    L.control.scale({ position: 'bottomleft', metric: true, imperial: false }).addTo(map);
    L.control.attribution({ position: 'bottomleft', prefix: false }).addTo(map);

    // Data layers
    const pointLayer = L.layerGroup().addTo(map);
    const forbiddenLayer = L.geoJSON(null, {
      style: { color: '#dc2626', weight: 2, fillColor: '#dc2626', fillOpacity: 0.15, dashArray: '6 4' },
    }).addTo(map);
    const erodedLayer = L.geoJSON(null, {
      style: { color: '#ea580c', weight: 2, fillColor: '#ea580c', fillOpacity: 0.15, dashArray: '6 4' },
    }).addTo(map);
    const processingLayer = L.layerGroup().addTo(map);
    const processingOverlayLayer = L.layerGroup().addTo(map);
    const processingCenterLayer = L.layerGroup().addTo(map);
    const processingFilteredLayer = L.layerGroup().addTo(map);

    // Store refs
    layersRef.current = {
      orthophoto,
      pointLayer,
      forbiddenLayer,
      erodedLayer,
      processingLayer,
      processingOverlayLayer,
      processingCenterLayer,
      processingFilteredLayer,
    };
    mapRef.current = map;
    useMapStore.getState().setMapInstance(map);

    // Persist user pans + zooms back to the store so other map-bearing pages
    // (PointDeletion in particular) inherit the same view when they mount.
    map.on('moveend', () => {
      if (skipNextSyncRef.current) {
        // This moveend was triggered by us applying a store change to the
        // map. Don't echo it back to the store.
        skipNextSyncRef.current = false;
        return;
      }
      const c = map.getCenter();
      useMapStore.getState().setView([c.lng, c.lat], map.getZoom());
    });

    // Rescale point markers when zoom changes so they don't clump when zoomed out
    map.on('zoomend', () => {
      const z = map.getZoom();
      const rFiltered = getPreviewFilteredRadius(z);
      if (layersRef.current.pointLayer) {
        layersRef.current.pointLayer.eachLayer((m) => {
          const selected = Boolean(m.options.mgDeletionSelected);
          if (typeof m.setStyle === 'function') {
            m.setStyle({
              color: selected ? '#7f1d1d' : '#000000',
              weight: getPointWeight(z, selected),
            });
          }
          if (typeof m.setRadius === 'function') m.setRadius(getPointRadius(z, selected));
        });
      }
      if (layersRef.current.processingLayer) {
        const rAnalysis = getAnalysisPointRadius(z);
        layersRef.current.processingLayer.eachLayer((m) => {
          if (typeof m.setStyle === 'function') m.setStyle({ weight: 0 });
          if (typeof m.setRadius === 'function') m.setRadius(rAnalysis);
        });
      }
      if (layersRef.current.processingFilteredLayer) {
        layersRef.current.processingFilteredLayer.eachLayer((m) => {
          if (typeof m.setRadius === 'function') m.setRadius(rFiltered);
        });
      }
    });

    // Resize handler
    const observer = new ResizeObserver(() => map.invalidateSize());
    observer.observe(containerRef.current);

    const { fetchPoints, fetchZones } = useMapStore.getState();
    fetchPoints();
    fetchZones();

    return () => {
      observer.disconnect();
      map.remove();
      mapRef.current = null;
      useMapStore.getState().setMapInstance(null);
    };
  }, []);

  // When the store's view changes (e.g. PointDeletion wrote to it on its own
  // moveend), bring this map to the same view. The skip ref + a rough equality
  // check below prevent feedback loops with the moveend handler above.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const c = map.getCenter();
    const sameLng = Math.abs(c.lng - center[0]) < 1e-7;
    const sameLat = Math.abs(c.lat - center[1]) < 1e-7;
    const sameZoom = map.getZoom() === zoom;
    if (sameLng && sameLat && sameZoom) return;
    skipNextSyncRef.current = true;
    map.setView([center[1], center[0]], zoom, { animate: false });
  }, [center, zoom]);

  // Sync layer visibility
  useEffect(() => {
    const map = mapRef.current;
    const layers = layersRef.current;
    if (!map || !layers.pointLayer) return;

    const toggle = (layer, visible) => {
      if (visible && !map.hasLayer(layer)) map.addLayer(layer);
      if (!visible && map.hasLayer(layer)) map.removeLayer(layer);
    };

    toggle(layers.pointLayer, layerVisibility.points);
    toggle(layers.orthophoto, layerVisibility.orthophoto);
    toggle(layers.forbiddenLayer, layerVisibility.forbidden);
    toggle(layers.erodedLayer, layerVisibility.eroded);
  }, [layerVisibility]);

  // Sync points — skip the redraw entirely when only unrelated store fields
  // changed (e.g. stats refresh after a tab switch with no actual point data change).
  useEffect(() => {
    const map = mapRef.current;
    const layer = layersRef.current.pointLayer;
    if (!map || !layer) return;

    // Build a lightweight fingerprint: id + status for every point.
    // If it matches the last render we skip the clear-and-redraw so the map
    // view stays rock-steady when pages call fetchPoints() on mount.
    const deletionSelectedIds = new Set(deletionSelectedPointIds.map(Number));
    const deletionSelectionKey = isPointDeletionMode
      ? [...deletionSelectedIds].sort((a, b) => a - b).join(',')
      : '';
    const fingerprint = `${isPointDeletionMode ? 'delete' : 'normal'}:${deletionSelectionKey}:` + points
      .map((p) => {
        const status = p.assigned_planter_name
          ? (p.assignment_status === 'completed' ? 'completed' : 'assigned')
          : (p.planting_status || 'planned');
        return `${p.id}:${status}`;
      })
      .join(',');

    if (fingerprint === pointsFingerprintRef.current) return;
    pointsFingerprintRef.current = fingerprint;

    layer.clearLayers();

    // Popup status badge background. completed / planted now use the same
    // amber tint so the point's badge matches the yellow marker.
    const statusBg = { planned: '#dcfce7', assigned: '#dbeafe', planted: '#fef3c7', completed: '#fef3c7' };

    points.forEach((p) => {
      const status = p.assigned_planter_name
        ? (p.assignment_status === 'completed' ? 'completed' : 'assigned')
        : (p.planting_status || 'planned');
      const pointId = Number(p.id);
      const isDeletionSelected = isPointDeletionMode && deletionSelectedIds.has(pointId);

      const color = isDeletionSelected ? '#ef4444' : (STATUS_COLORS[status] || STATUS_COLORS.planned);

      const z = map.getZoom();
      const marker = L.circleMarker([p.latitude, p.longitude], {
        radius: getPointRadius(z, isDeletionSelected),
        fillColor: color,
        color: isDeletionSelected ? '#7f1d1d' : '#000000',
        weight: getPointWeight(z, isDeletionSelected),
        fillOpacity: isDeletionSelected ? 0.96 : 0.92,
        opacity: 0.95,
        mgDeletionSelected: isDeletionSelected,
      });

      marker.bindPopup(`
        <div style="font-family:'Inter',sans-serif;min-width:180px;">
          <div style="font-weight:700;font-size:14px;margin-bottom:4px;">Point #${p.point_num}</div>
          <div style="font-size:12px;color:#6b7280;margin-bottom:8px;">${p.image_name || ''}</div>
          <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
            <span style="font-size:10px;padding:2px 8px;border-radius:99px;background:${isDeletionSelected ? '#fee2e2' : (statusBg[status] || '#f3f4f6')};font-weight:700;text-transform:uppercase;color:${isDeletionSelected ? '#991b1b' : 'inherit'};">${isDeletionSelected ? 'selected' : status}</span>
            ${p.assigned_planter_name ? `<span style="font-size:12px;color:#6b7280;">${p.assigned_planter_name}</span>` : ''}
          </div>
          <div style="font-size:11px;color:#9ca3af;margin-top:8px;font-family:monospace;">${p.latitude.toFixed(7)}, ${p.longitude.toFixed(7)}</div>
        </div>
      `, { maxWidth: 260 });

      marker.on('click', () => {
        if (isPointDeletionMode) {
          if (Number.isFinite(pointId)) toggleDeletionPoint(pointId);
          return;
        }
        setSelectedPoint(p.id);
      });
      marker.addTo(layer);
    });

    // preserved across every tab switch. maxZoom: 22 ≈ 5 m scale.
  }, [points, setSelectedPoint, isPointDeletionMode, deletionSelectedPointIds, toggleDeletionPoint]);

  // Sync zones
  useEffect(() => {
    const { forbiddenLayer, erodedLayer } = layersRef.current;
    if (forbiddenZones && forbiddenLayer) {
      forbiddenLayer.clearLayers();
      try { forbiddenLayer.addData(forbiddenZones); } catch { /* skip */ }
    }
    if (erodedZones && erodedLayer) {
      erodedLayer.clearLayers();
      try { erodedLayer.addData(erodedZones); } catch { /* skip */ }
    }
  }, [forbiddenZones, erodedZones]);

  // Sync current unsaved analysis preview
  useEffect(() => {
    const map = mapRef.current;
    const {
      processingLayer,
      processingOverlayLayer,
      processingCenterLayer,
      processingFilteredLayer,
    } = layersRef.current;
    if (
      !map ||
      !processingLayer ||
      !processingOverlayLayer ||
      !processingCenterLayer ||
      !processingFilteredLayer
    ) return;

    processingLayer.clearLayers();
    processingOverlayLayer.clearLayers();
    processingCenterLayer.clearLayers();
    processingFilteredLayer.clearLayers();

    if (!currentAnalysis?.map?.available) return;

    const analysisOverlay = currentAnalysis.map.analysis_overlay;
    const safeFeatures = currentAnalysis.map.safe_points_geojson?.features || [];
    const forbiddenFeatures = currentAnalysis.map.forbidden_filtered_geojson?.features || [];
    const erodedFeatures = currentAnalysis.map.eroded_filtered_geojson?.features || [];
    const orthophotoCanopyFeatures = currentAnalysis.map.orthophoto_canopy_filtered_geojson?.features || [];
    const centerFeature = currentAnalysis.map.image_center_feature;

    if (Array.isArray(analysisOverlay?.bounds)) {
      if (analysisOverlay.image_data_url) {
        L.imageOverlay(analysisOverlay.image_data_url, analysisOverlay.bounds, {
          opacity: analysisOverlay.opacity ?? 0.72,
          interactive: false,
          zIndex: 320,
        }).addTo(processingOverlayLayer);
      }
    }

    safeFeatures.forEach((feature) => {
      const [lon, lat] = feature.geometry.coordinates;
      const props = feature.properties || {};
      L.circleMarker([lat, lon], {
        radius: getAnalysisPointRadius(map.getZoom()),
        color: '#000000',
        weight: 1,
        fillColor: PREVIEW_COLORS.safe,
        fillOpacity: 0.85,
      })
        .bindPopup(`
          <div style="font-family:'Inter',sans-serif;min-width:180px;">
            <div style="font-weight:700;font-size:14px;margin-bottom:6px;">${props.name || 'Planting Point'}</div>
            <div style="font-size:12px;color:#4b5563;margin-bottom:6px;">Unsaved analysis preview</div>
            <div style="font-size:12px;color:#111827;">${lat.toFixed(7)}, ${lon.toFixed(7)}</div>
            <div style="font-size:12px;color:#6b7280;margin-top:6px;">Buffer: ${props.buffer_m ?? '-'} m</div>
            <div style="font-size:12px;color:#6b7280;">Area: ${props.area_m2 ?? '-'} m²</div>
          </div>
        `)
        .addTo(processingLayer);
    });

    forbiddenFeatures.forEach((feature) => {
      const [lon, lat] = feature.geometry.coordinates;
      const props = feature.properties || {};
      L.circleMarker([lat, lon], {
        radius: getPreviewFilteredRadius(map.getZoom()),
        color: PREVIEW_COLORS.forbidden,
        weight: 2,
        fillColor: '#fecaca',
        fillOpacity: 0.9,
      })
        .bindPopup(`
          <div style="font-family:'Inter',sans-serif;min-width:170px;">
            <div style="font-weight:700;font-size:14px;margin-bottom:6px;color:${PREVIEW_COLORS.forbidden};">Filtered Preview Point</div>
            <div style="font-size:12px;color:#111827;">${lat.toFixed(7)}, ${lon.toFixed(7)}</div>
            <div style="font-size:12px;color:#6b7280;margin-top:6px;">${props.reason || 'Inside forbidden zone'}</div>
          </div>
        `)
        .addTo(processingFilteredLayer);
    });

    erodedFeatures.forEach((feature) => {
      const [lon, lat] = feature.geometry.coordinates;
      const props = feature.properties || {};
      L.circleMarker([lat, lon], {
        radius: getPreviewFilteredRadius(map.getZoom()),
        color: PREVIEW_COLORS.eroded,
        weight: 2,
        fillColor: '#fed7aa',
        fillOpacity: 0.9,
      })
        .bindPopup(`
          <div style="font-family:'Inter',sans-serif;min-width:170px;">
            <div style="font-weight:700;font-size:14px;margin-bottom:6px;color:${PREVIEW_COLORS.eroded};">Filtered Preview Point</div>
            <div style="font-size:12px;color:#111827;">${lat.toFixed(7)}, ${lon.toFixed(7)}</div>
            <div style="font-size:12px;color:#6b7280;margin-top:6px;">${props.reason || 'Inside eroded zone'}</div>
          </div>
        `)
        .addTo(processingFilteredLayer);
    });

    orthophotoCanopyFeatures.forEach((feature) => {
      const [lon, lat] = feature.geometry.coordinates;
      const props = feature.properties || {};
      L.circleMarker([lat, lon], {
        radius: getPreviewFilteredRadius(map.getZoom()),
        color: PREVIEW_COLORS.canopy,
        weight: 2,
        fillColor: '#ddd6fe',
        fillOpacity: 0.9,
      })
        .bindPopup(`
          <div style="font-family:'Inter',sans-serif;min-width:170px;">
            <div style="font-weight:700;font-size:14px;margin-bottom:6px;color:${PREVIEW_COLORS.canopy};">Filtered Preview Point</div>
            <div style="font-size:12px;color:#111827;">${lat.toFixed(7)}, ${lon.toFixed(7)}</div>
            <div style="font-size:12px;color:#6b7280;margin-top:6px;">${props.reason || 'Overlaps canopy in orthophoto recheck'}</div>
          </div>
        `)
        .addTo(processingFilteredLayer);
    });

    if (centerFeature?.geometry?.coordinates) {
      const [lon, lat] = centerFeature.geometry.coordinates;
      const props = centerFeature.properties || {};
      L.marker([lat, lon])
        .bindPopup(`
          <div style="font-family:'Inter',sans-serif;min-width:180px;">
            <div style="font-weight:700;font-size:14px;margin-bottom:6px;">${props.name || 'Image Center'}</div>
            <div style="font-size:12px;color:#111827;">${lat.toFixed(7)}, ${lon.toFixed(7)}</div>
            <div style="font-size:12px;color:#6b7280;margin-top:6px;">Altitude: ${props.altitude_m ?? '-'} m</div>
            <div style="font-size:12px;color:#6b7280;">Heading: ${props.heading ?? '-'}°</div>
          </div>
        `)
        .addTo(processingCenterLayer);
    }

    const previewLatLngs = safeFeatures
      .concat(forbiddenFeatures)
      .concat(erodedFeatures)
      .concat(orthophotoCanopyFeatures)
      .map((feature) => {
        const [lon, lat] = feature.geometry.coordinates;
        return [lat, lon];
      });

    if (centerFeature?.geometry?.coordinates) {
      const [lon, lat] = centerFeature.geometry.coordinates;
      previewLatLngs.push([lat, lon]);
    }

    if (previewLatLngs.length > 0) {
      map.fitBounds(L.latLngBounds(previewLatLngs), {
        padding: [60, 420, 60, 100],
        maxZoom: 20,
        animate: true,
      });
    }
  }, [currentAnalysis]);

  return <div ref={containerRef} className="map-container" />;
}
