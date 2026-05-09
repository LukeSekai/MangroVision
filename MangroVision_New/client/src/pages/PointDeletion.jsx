import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { Panel, PanelCard } from '../components/Panel';
import Modal from '../components/Modal';
import { ORTHOPHOTO_MAX_NATIVE_ZOOM, ORTHOPHOTO_TILE_URL } from '../config/mapTiles';
import { useMapStore } from '../stores/mapStore';
import './PointDeletion.css';

const STATUS_COLORS = {
  planned: '#16a34a',
  assigned: '#2563eb',
  planted: '#d97706',
  completed: '#059669',
  skipped: '#9ca3af',
};

const getPointRadius = (zoom, selected = false) => {
  const base = zoom >= 22 ? 3.5 : zoom >= 21 ? 2.4 : zoom >= 20 ? 1.6 : zoom >= 19 ? 1.05 : zoom >= 18 ? 0.8 : zoom >= 17 ? 0.65 : 0.55;
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

function getPointStatus(point) {
  if (point.assigned_planter_name) {
    return point.assignment_status === 'completed' ? 'completed' : 'assigned';
  }
  return point.planting_status || 'planned';
}

export default function PointDeletion() {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const layersRef = useRef({});
  const fittedRef = useRef(false);

  const points = useMapStore((s) => s.points);
  const loadingPoints = useMapStore((s) => s.loadingPoints);
  const fetchPoints = useMapStore((s) => s.fetchPoints);
  const fetchStats = useMapStore((s) => s.fetchStats);
  const fetchZones = useMapStore((s) => s.fetchZones);
  const forbiddenZones = useMapStore((s) => s.forbiddenZones);
  const erodedZones = useMapStore((s) => s.erodedZones);
  const deletePoints = useMapStore((s) => s.deletePoints);

  const [selectedIds, setSelectedIds] = useState(() => new Set());
  const [confirmOpen, setConfirmOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  const selectablePoints = useMemo(
    () => points.filter((point) => (
      Number.isFinite(Number(point.id)) &&
      Number.isFinite(Number(point.latitude)) &&
      Number.isFinite(Number(point.longitude))
    )),
    [points],
  );

  const selectedPoints = useMemo(
    () => selectablePoints.filter((point) => selectedIds.has(Number(point.id))),
    [selectablePoints, selectedIds],
  );

  const togglePoint = useCallback((pointId) => {
    const numericId = Number(pointId);
    if (!Number.isFinite(numericId)) return;
    setNotice('');
    setError('');
    setSelectedIds((previous) => {
      const next = new Set(previous);
      if (next.has(numericId)) {
        next.delete(numericId);
      } else {
        next.add(numericId);
      }
      return next;
    });
  }, []);

  useEffect(() => {
    fetchPoints();
    fetchStats();
    fetchZones();
  }, [fetchPoints, fetchStats, fetchZones]);

  useEffect(() => {
    const validIds = new Set(selectablePoints.map((point) => Number(point.id)));
    setSelectedIds((previous) => {
      const next = new Set([...previous].filter((pointId) => validIds.has(pointId)));
      return next.size === previous.size ? previous : next;
    });
  }, [selectablePoints]);

  useEffect(() => {
    if (mapRef.current || !containerRef.current) return undefined;

    const map = L.map(containerRef.current, {
      center: [10.78, 122.6253],
      zoom: 21,
      maxZoom: 24,
      zoomControl: false,
      attributionControl: false,
    });

    const satellite = L.tileLayer(
      'https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
      { subdomains: '0123', maxZoom: 21, attribution: '&copy; Google' },
    );
    const osm = L.tileLayer(
      'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
      { maxZoom: 19, attribution: '&copy; OpenStreetMap' },
    );
    const orthophoto = L.tileLayer(
      ORTHOPHOTO_TILE_URL,
      { maxZoom: 24, maxNativeZoom: ORTHOPHOTO_MAX_NATIVE_ZOOM, opacity: 1.0, errorTileUrl: '', minZoom: 10 },
    );

    satellite.addTo(map);
    orthophoto.addTo(map);

    L.control.layers(
      { Satellite: satellite, OpenStreetMap: osm },
      { Orthophoto: orthophoto },
      { position: 'topright', collapsed: true },
    ).addTo(map);
    L.control.zoom({ position: 'bottomright' }).addTo(map);
    L.control.scale({ position: 'bottomleft', metric: true, imperial: false }).addTo(map);

    const pointLayer = L.layerGroup().addTo(map);
    const forbiddenLayer = L.geoJSON(null, {
      style: { color: '#dc2626', weight: 2, fillColor: '#dc2626', fillOpacity: 0.15, dashArray: '6 4' },
    }).addTo(map);
    const erodedLayer = L.geoJSON(null, {
      style: { color: '#ea580c', weight: 2, fillColor: '#ea580c', fillOpacity: 0.15, dashArray: '6 4' },
    }).addTo(map);

    map.on('zoomend', () => {
      const zoom = map.getZoom();
      pointLayer.eachLayer((marker) => {
        const isSelected = Boolean(marker.options.mgSelected);
        if (typeof marker.setRadius === 'function') {
          marker.setRadius(getPointRadius(zoom, isSelected));
        }
        if (typeof marker.setStyle === 'function') {
          marker.setStyle({ weight: getPointWeight(zoom, isSelected) });
        }
      });
    });

    const observer = new ResizeObserver(() => map.invalidateSize());
    observer.observe(containerRef.current);

    layersRef.current = { pointLayer, forbiddenLayer, erodedLayer };
    mapRef.current = map;

    return () => {
      observer.disconnect();
      map.remove();
      mapRef.current = null;
    };
  }, []);

  useEffect(() => {
    const { forbiddenLayer, erodedLayer } = layersRef.current;
    if (forbiddenLayer && forbiddenZones) {
      forbiddenLayer.clearLayers();
      try { forbiddenLayer.addData(forbiddenZones); } catch { /* skip malformed geojson */ }
    }
    if (erodedLayer && erodedZones) {
      erodedLayer.clearLayers();
      try { erodedLayer.addData(erodedZones); } catch { /* skip malformed geojson */ }
    }
  }, [forbiddenZones, erodedZones]);

  useEffect(() => {
    const map = mapRef.current;
    const layer = layersRef.current.pointLayer;
    if (!map || !layer) return;

    layer.clearLayers();
    const zoom = map.getZoom();

    selectablePoints.forEach((point) => {
      const pointId = Number(point.id);
      const status = getPointStatus(point);
      const isSelected = selectedIds.has(pointId);
      const color = isSelected ? '#ef4444' : (STATUS_COLORS[status] || STATUS_COLORS.planned);

      const marker = L.circleMarker([point.latitude, point.longitude], {
        radius: getPointRadius(zoom, isSelected),
        fillColor: color,
        color: isSelected ? '#7f1d1d' : '#000000',
        weight: getPointWeight(zoom, isSelected),
        fillOpacity: isSelected ? 0.96 : 0.9,
        opacity: 0.98,
        mgSelected: isSelected,
      });

      marker.bindPopup(`
        <div style="font-family:'Inter',sans-serif;min-width:190px;">
          <div style="font-weight:800;font-size:14px;margin-bottom:4px;">Point #${point.point_num}</div>
          <div style="font-size:12px;color:#6b7280;margin-bottom:8px;">${point.image_name || ''}</div>
          <div style="display:flex;align-items:center;gap:6px;flex-wrap:wrap;">
            <span style="font-size:10px;padding:2px 8px;border-radius:99px;background:${isSelected ? '#fee2e2' : '#f3f4f6'};color:${isSelected ? '#991b1b' : '#374151'};font-weight:800;text-transform:uppercase;">${isSelected ? 'selected' : status}</span>
            ${point.assigned_planter_name ? `<span style="font-size:12px;color:#6b7280;">${point.assigned_planter_name}</span>` : ''}
          </div>
          <div style="font-size:11px;color:#9ca3af;margin-top:8px;font-family:monospace;">${Number(point.latitude).toFixed(7)}, ${Number(point.longitude).toFixed(7)}</div>
        </div>
      `, { maxWidth: 260 });
      marker.on('click', () => togglePoint(pointId));
      marker.addTo(layer);
    });

    if (selectablePoints.length > 0 && !fittedRef.current) {
      const bounds = L.latLngBounds(selectablePoints.map((point) => [point.latitude, point.longitude]));
      map.fitBounds(bounds, { padding: [60, 420, 60, 100], maxZoom: 21, animate: true });
      fittedRef.current = true;
    }
  }, [selectablePoints, selectedIds, togglePoint]);

  const selectVisiblePoints = () => {
    const map = mapRef.current;
    if (!map) return;
    const bounds = map.getBounds();
    const visibleIds = selectablePoints
      .filter((point) => bounds.contains([point.latitude, point.longitude]))
      .map((point) => Number(point.id));
    setNotice('');
    setError('');
    setSelectedIds((previous) => new Set([...previous, ...visibleIds]));
  };

  const removeSelectedPoint = (pointId) => {
    setSelectedIds((previous) => {
      const next = new Set(previous);
      next.delete(Number(pointId));
      return next;
    });
  };

  const confirmDelete = async () => {
    if (!selectedIds.size) return;
    setDeleting(true);
    setError('');
    setNotice('');
    try {
      const payload = await deletePoints([...selectedIds]);
      setConfirmOpen(false);
      setSelectedIds(new Set());
      await Promise.all([fetchPoints(), fetchStats()]);
      const deletedCount = payload.deleted_count || 0;
      setNotice(deletedCount === 1 ? '1 point deleted.' : `${deletedCount} points deleted.`);
    } catch (deleteError) {
      setError(deleteError.message || 'Point deletion failed');
    } finally {
      setDeleting(false);
    }
  };

  const selectedCount = selectedIds.size;

  return (
    <div className="point-cleanup-page">
      <div ref={containerRef} className="point-cleanup-map" />

      <Panel title="Delete Points" subtitle={`${selectablePoints.length} mapped points`}>
        <PanelCard
          title="Selection"
          badge={selectedCount}
          icon={
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18" />
              <path d="M8 6V4h8v2" />
              <path d="M19 6l-1 14H6L5 6" />
            </svg>
          }
        >
          <div className="cleanup-selection-row">
            <div>
              <div className="cleanup-selection-count">{selectedCount}</div>
              <div className="cleanup-selection-label">selected</div>
            </div>
            <div className="cleanup-actions">
              <button type="button" className="btn btn-secondary btn-sm" onClick={selectVisiblePoints} disabled={!selectablePoints.length}>
                Select visible
              </button>
              <button type="button" className="btn btn-ghost btn-sm" onClick={() => setSelectedIds(new Set())} disabled={!selectedCount}>
                Clear
              </button>
            </div>
          </div>
          <button
            type="button"
            className="btn btn-danger btn-lg cleanup-delete-button"
            onClick={() => setConfirmOpen(true)}
            disabled={!selectedCount || deleting}
          >
            Delete selected
          </button>
          {loadingPoints && <div className="cleanup-muted">Loading points...</div>}
          {notice && <div className="cleanup-notice">{notice}</div>}
          {error && <div className="cleanup-error">{error}</div>}
        </PanelCard>

        <PanelCard
          title="Selected Points"
          badge={selectedCount}
          defaultOpen
          icon={
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 11l3 3L22 4" />
              <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
            </svg>
          }
        >
          {selectedPoints.length === 0 ? (
            <div className="cleanup-muted">No points selected.</div>
          ) : (
            <div className="cleanup-selected-list">
              {selectedPoints.slice(0, 30).map((point) => {
                const status = getPointStatus(point);
                return (
                  <div key={point.id} className="cleanup-selected-item">
                    <div className="cleanup-selected-main">
                      <div className="cleanup-selected-title">Point #{point.point_num}</div>
                      <div className="cleanup-selected-meta">{point.image_name || 'Saved analysis'}</div>
                    </div>
                    <span className={`badge badge-${status}`}>{status}</span>
                    <button
                      type="button"
                      className="btn btn-ghost btn-icon cleanup-remove-button"
                      onClick={() => removeSelectedPoint(point.id)}
                      title="Remove from selection"
                    >
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                    </button>
                  </div>
                );
              })}
              {selectedPoints.length > 30 && (
                <div className="cleanup-muted">+{selectedPoints.length - 30} more selected</div>
              )}
            </div>
          )}
        </PanelCard>
      </Panel>

      <Modal
        open={confirmOpen}
        title={selectedCount === 1 ? 'Delete selected point?' : `Delete ${selectedCount} selected points?`}
        variant="danger"
        confirmLabel={selectedCount === 1 ? 'Delete point' : 'Delete points'}
        busy={deleting}
        onConfirm={confirmDelete}
        onCancel={() => setConfirmOpen(false)}
      >
        <p>This removes the selected planting point records from the map and any planter assignments that use them.</p>
      </Modal>
    </div>
  );
}
