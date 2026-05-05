import { useEffect, useState, useCallback } from 'react';
import L from 'leaflet';
import { useMapStore } from '../stores/mapStore';
import { Panel, PanelCard } from '../components/Panel';
import './ErodedZoneEditor.css';

const API = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export default function ErodedZoneEditor() {
  const erodedZones = useMapStore((s) => s.erodedZones);
  const forbiddenZones = useMapStore((s) => s.forbiddenZones);
  const fetchZones = useMapStore((s) => s.fetchZones);
  const mapInstance = useMapStore((s) => s.mapInstance);

  const [deleting, setDeleting] = useState(null);
  const [drawing, setDrawing] = useState(false);
  const [drawLayer, setDrawLayer] = useState(null);
  const [drawnCoords, setDrawnCoords] = useState(null);
  const [zoneName, setZoneName] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveMsg, setSaveMsg] = useState('');

  useEffect(() => { fetchZones(); }, [fetchZones]);

  const erodedFeatures = erodedZones?.features || [];
  const forbiddenFeatures = forbiddenZones?.features || [];

  // Start drawing mode
  const startDrawing = useCallback(() => {
    if (!mapInstance) return;
    setDrawing(true);
    setSaveMsg('');
    setDrawnCoords(null);

    // Temporary drawing layer
    const layer = L.featureGroup().addTo(mapInstance);
    setDrawLayer(layer);

    // Change cursor
    mapInstance.getContainer().style.cursor = 'crosshair';

    const points = [];
    let polyline = null;

    const onClick = (e) => {
      points.push([e.latlng.lat, e.latlng.lng]);

      // Draw a marker for each vertex
      L.circleMarker(e.latlng, {
        radius: 5, fillColor: '#ea580c', color: '#fff', weight: 2, fillOpacity: 1,
      }).addTo(layer);

      // Update the polyline preview
      if (polyline) layer.removeLayer(polyline);
      if (points.length > 1) {
        polyline = L.polyline(points, {
          color: '#ea580c', weight: 2, dashArray: '6 4',
        }).addTo(layer);
      }
    };

    const onDblClick = (e) => {
      L.DomEvent.stopPropagation(e);
      L.DomEvent.preventDefault(e);

      if (points.length < 3) {
        setSaveMsg('Need at least 3 points for a polygon');
        return;
      }

      // Close the polygon
      const closed = [...points, points[0]];

      // Clear the drawing aids and show the final polygon
      layer.clearLayers();
      L.polygon(closed, {
        color: '#ea580c', weight: 2, fillColor: '#ea580c', fillOpacity: 0.2,
      }).addTo(layer);

      // Store coordinates in GeoJSON format [lng, lat]
      setDrawnCoords(closed.map(([lat, lng]) => [lng, lat]));

      // Clean up events
      mapInstance.off('click', onClick);
      mapInstance.off('dblclick', onDblClick);
      mapInstance.getContainer().style.cursor = '';
    };

    mapInstance.on('click', onClick);
    mapInstance.on('dblclick', onDblClick);

    // Store cleanup functions
    layer._cleanupFn = () => {
      mapInstance.off('click', onClick);
      mapInstance.off('dblclick', onDblClick);
      mapInstance.getContainer().style.cursor = '';
    };
  }, [mapInstance]);

  // Cancel drawing
  const cancelDrawing = useCallback(() => {
    if (drawLayer) {
      if (drawLayer._cleanupFn) drawLayer._cleanupFn();
      if (mapInstance && mapInstance.hasLayer(drawLayer)) {
        mapInstance.removeLayer(drawLayer);
      }
    }
    setDrawing(false);
    setDrawLayer(null);
    setDrawnCoords(null);
    setZoneName('');
  }, [drawLayer, mapInstance]);

  // Save the drawn zone
  const saveZone = async () => {
    if (!drawnCoords) return;
    setSaving(true);
    setSaveMsg('');
    try {
      const feature = {
        type: 'Feature',
        properties: { name: zoneName || `Eroded Zone ${erodedFeatures.length + 1}` },
        geometry: {
          type: 'Polygon',
          coordinates: [drawnCoords],
        },
      };

      const res = await fetch(`${API}/api/zones/eroded`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(feature),
      });

      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Failed to save zone');
      }

      setSaveMsg('Zone saved');
      cancelDrawing();
      fetchZones();
    } catch (err) {
      setSaveMsg(`Error: ${err.message}`);
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteEroded = async (index) => {
    setDeleting(index);
    try {
      const res = await fetch(`${API}/api/zones/eroded/${index}`, { method: 'DELETE' });
      if (res.ok) fetchZones();
    } catch (err) {
      console.error('Delete failed:', err);
    } finally {
      setDeleting(null);
    }
  };

  const handleClearAll = async () => {
    try {
      const res = await fetch(`${API}/api/zones/eroded`, {
        method: 'PUT',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ features: [] }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Failed to clear zones');
      }
      fetchZones();
      setSaveMsg('All eroded zones cleared');
    } catch (err) {
      setSaveMsg(`Error: ${err.message}`);
    }
  };

  const handleExportGeoJSON = () => {
    const blob = new Blob([JSON.stringify(erodedZones || { type: 'FeatureCollection', features: [] }, null, 2)], {
      type: 'application/geo+json',
    });
    const url = window.URL.createObjectURL(blob);
    const link = document.createElement('a');
    link.href = url;
    link.download = 'eroded_zones.geojson';
    document.body.appendChild(link);
    link.click();
    link.remove();
    window.URL.revokeObjectURL(url);
  };

  const getZoneName = (feature, index, prefix) => {
    return feature?.properties?.name || feature?.properties?.label || `${prefix} Zone ${index + 1}`;
  };

  const getZoneArea = (feature) => {
    const area = feature?.properties?.area_m2 || feature?.properties?.area;
    return area ? `${Number(area).toFixed(1)} m²` : null;
  };

  return (
    <Panel title="Zone Editor" subtitle={`${erodedFeatures.length + forbiddenFeatures.length} zones total`}>
      {/* Draw tools */}
      <PanelCard
        title="Draw Zone"
        icon={<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 19l7-7 3 3-7 7-3-3z"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/><path d="M2 2l7.586 7.586"/><circle cx="11" cy="11" r="2"/></svg>}
      >
        {!drawing ? (
          <div>
            <button className="btn btn-primary btn-sm" style={{ width: '100%' }} onClick={startDrawing}>
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
              Draw New Eroded Zone
            </button>
            <p className="text-sm" style={{ color: 'var(--text-muted)', marginTop: 8, lineHeight: 1.5 }}>
              Click points on the map to draw a polygon. Double-click to finish.
            </p>
          </div>
        ) : !drawnCoords ? (
          <div>
            <div className="drawing-active">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ea580c" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
              <span>Drawing active — click on the map</span>
            </div>
            <button className="btn btn-ghost btn-sm" style={{ width: '100%', marginTop: 8 }} onClick={cancelDrawing}>
              Cancel
            </button>
          </div>
        ) : (
          <div className="save-form">
            <div className="form-group">
              <label className="form-label">Zone Name</label>
              <input
                className="form-input"
                type="text"
                placeholder={`Eroded Zone ${erodedFeatures.length + 1}`}
                value={zoneName}
                onChange={(e) => setZoneName(e.target.value)}
              />
            </div>
            <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
              <button className="btn btn-primary btn-sm" style={{ flex: 1 }} onClick={saveZone} disabled={saving}>
                {saving ? 'Saving...' : 'Save Zone'}
              </button>
              <button className="btn btn-ghost btn-sm" onClick={cancelDrawing}>Cancel</button>
            </div>
            {saveMsg && <p className="text-sm" style={{ color: saveMsg.startsWith('Error') ? '#991b1b' : 'var(--color-completed)', marginTop: 6 }}>{saveMsg}</p>}
          </div>
        )}
      </PanelCard>

      {/* Eroded Zones */}
      <PanelCard
        title="Eroded Zones"
        badge={erodedFeatures.length}
        icon={<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ea580c" strokeWidth="2"><path d="M12 22s-8-4.5-8-11.8A8 8 0 0 1 12 2a8 8 0 0 1 8 8.2c0 7.3-8 11.8-8 11.8z"/><circle cx="12" cy="10" r="3"/></svg>}
      >
        <div className="zone-actions">
          <button className="btn btn-secondary btn-sm" onClick={handleExportGeoJSON} disabled={!erodedFeatures.length}>
            Export GeoJSON
          </button>
          <button className="btn btn-ghost btn-sm" onClick={handleClearAll} disabled={!erodedFeatures.length}>
            Clear All
          </button>
        </div>
        <div className="zone-list">
          {erodedFeatures.length === 0 ? (
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No eroded zones defined</p>
          ) : (
            erodedFeatures.map((f, i) => (
              <div key={i} className="zone-row">
                <div className="zone-dot" style={{ background: '#ea580c' }} />
                <div className="zone-info">
                  <span className="zone-name">{getZoneName(f, i, 'Eroded')}</span>
                  {getZoneArea(f) && <span className="zone-area">{getZoneArea(f)}</span>}
                </div>
                <button
                  className="btn btn-ghost btn-sm btn-icon"
                  onClick={() => handleDeleteEroded(i)}
                  disabled={deleting === i}
                  title="Delete zone"
                >
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                    <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                  </svg>
                </button>
              </div>
            ))
          )}
        </div>
      </PanelCard>

      {/* Forbidden Zones */}
      <PanelCard
        title="Forbidden Zones"
        badge={forbiddenFeatures.length}
        icon={<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#dc2626" strokeWidth="2"><circle cx="12" cy="12" r="10"/><line x1="4.93" y1="4.93" x2="19.07" y2="19.07"/></svg>}
        defaultOpen={false}
      >
        <div className="zone-list">
          {forbiddenFeatures.length === 0 ? (
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No forbidden zones defined</p>
          ) : (
            forbiddenFeatures.map((f, i) => (
              <div key={i} className="zone-row">
                <div className="zone-dot" style={{ background: '#dc2626' }} />
                <div className="zone-info">
                  <span className="zone-name">{getZoneName(f, i, 'Forbidden')}</span>
                  {getZoneArea(f) && <span className="zone-area">{getZoneArea(f)}</span>}
                </div>
                <span className="badge badge-forbidden" style={{ fontSize: '0.5625rem' }}>Read-only</span>
              </div>
            ))
          )}
        </div>
      </PanelCard>
    </Panel>
  );
}
