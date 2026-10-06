import { useEffect, useRef, useState, useCallback } from 'react';
import L from 'leaflet';
import { useMapStore } from '../stores/mapStore';
import { useProcessingStore } from '../stores/processingStore';
import { useAuthStore } from '../stores/authStore';
import { Panel, PanelCard } from '../components/Panel';
import Modal from '../components/Modal';
import './ErodedZoneEditor.css';

const API = import.meta.env.VITE_API_BASE || '';

// Color used for in-progress polygon drawing — keyed to the zone kind so the
// admin sees the right preview color while clicking points on the map.
const KIND_COLORS = {
  eroded: '#ea580c',
  site: '#0ea5e9',
  warning: '#f59e0b',
};

const WARNING_TYPES = [
  ['deep_mud', 'Deep mud / difficult access'],
  ['unstable_sediment', 'Unstable sediment'],
  ['tidal_exposure', 'High tidal exposure'],
  ['wave_exposure', 'Wave exposure'],
  ['low_survival_confidence', 'Low survival confidence'],
  ['planner_warning', 'Planner warning'],
  ['other', 'Other'],
];

const WARNING_SEVERITIES = [
  ['low', 'Low'],
  ['medium', 'Medium'],
  ['high', 'High'],
];

function pointInsidePolygon(latitude, longitude, vertices) {
  if (!Number.isFinite(latitude) || !Number.isFinite(longitude) || vertices.length < 3) return false;
  let inside = false;
  for (let current = 0, previous = vertices.length - 1; current < vertices.length; previous = current++) {
    const [currentLat, currentLon] = vertices[current];
    const [previousLat, previousLon] = vertices[previous];
    const crosses = ((currentLat > latitude) !== (previousLat > latitude))
      && (longitude < ((previousLon - currentLon) * (latitude - currentLat))
        / ((previousLat - currentLat) || Number.EPSILON) + currentLon);
    if (crosses) inside = !inside;
  }
  return inside;
}

function coveredPointCount(vertices, points) {
  if (vertices.length < 3) return 0;
  return points.reduce((total, point) => (
    pointInsidePolygon(Number(point.latitude), Number(point.longitude), vertices)
      ? total + 1
      : total
  ), 0);
}

export default function ErodedZoneEditor() {
  const loaded = useRef(false);
  const adminToken = useAuthStore((s) => s.token);
  const erodedZones = useMapStore((s) => s.erodedZones);
  const forbiddenZones = useMapStore((s) => s.forbiddenZones);
  const warningZones = useMapStore((s) => s.warningZones);
  const projectSites = useMapStore((s) => s.projectSites);
  const fetchZones = useMapStore((s) => s.fetchZones);
  const fetchPoints = useMapStore((s) => s.fetchPoints);
  const points = useMapStore((s) => s.points);
  const mapInstance = useMapStore((s) => s.mapInstance);
  const processing = useProcessingStore((s) => s.processing);
  const processingStage = useProcessingStore((s) => s.stage);

  const [deleting, setDeleting] = useState(null);
  const [pendingDeleteIndex, setPendingDeleteIndex] = useState(null);
  const [clearAllOpen, setClearAllOpen] = useState(false);
  const [clearAllBusy, setClearAllBusy] = useState(false);
  const [drawing, setDrawing] = useState(false);
  const [drawLayer, setDrawLayer] = useState(null);
  const [drawnCoords, setDrawnCoords] = useState(null);
  const [coveredCount, setCoveredCount] = useState(0);
  const [zoneName, setZoneName] = useState('');
  // Stable project sites are parent boundaries; assignment zones remain
  // auto-derived and are displayed independently on the map.
  const [drawingKind, setDrawingKind] = useState('eroded');
  const [projectSiteOrganizationId, setProjectSiteOrganizationId] = useState('');
  const [zoneNotes, setZoneNotes] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveMsgText, setSaveMsgText] = useState('');
  const [saveMsgIsError, setSaveMsgIsError] = useState(false);
  const [warningType, setWarningType] = useState('deep_mud');
  const [warningSeverity, setWarningSeverity] = useState('medium');
  const [warningDeleting, setWarningDeleting] = useState(null);
  const [pendingWarningDelete, setPendingWarningDelete] = useState(null);
  const [projectSiteDeleting, setProjectSiteDeleting] = useState(null);
  const [pendingProjectSiteDelete, setPendingProjectSiteDelete] = useState(null);
  const [projectSiteDeleteError, setProjectSiteDeleteError] = useState('');
  const zoneLocked = processing;
  const warningFeatures = warningZones?.features || [];
  const projectSiteFeatures = projectSites?.features || [];
  const organizationOptions = (() => {
    const sources = projectSites?.organizations || projectSites?.organization_options || [];
    const options = new Map();
    sources.forEach((organization) => {
      const id = organization?.id ?? organization?.organization_id;
      const name = organization?.name ?? organization?.organization_name;
      if (id !== null && id !== undefined && name) options.set(String(id), { id, name });
    });
    projectSiteFeatures.forEach((feature) => {
      const props = feature?.properties || {};
      const id = props.organization_id;
      const name = props.organization_name;
      if (id !== null && id !== undefined && name) options.set(String(id), { id, name });
    });
    return [...options.values()].sort((a, b) => String(a.name).localeCompare(String(b.name)));
  })();
  const selectedProjectOrganization = organizationOptions.find(
    (organization) => String(organization.id) === String(projectSiteOrganizationId),
  );

  useEffect(() => {
    if (loaded.current) return;
    loaded.current = true;
    fetchZones();
    fetchPoints();
  }, [fetchPoints, fetchZones]);

  const erodedFeatures = erodedZones?.features || [];
  const forbiddenFeatures = forbiddenZones?.features || [];

  // Start freehand drawing mode. Holding for a short moment prevents an
  // accidental pan from becoming a project boundary; after the hold the
  // editor traces the user's pointer and continuously counts enclosed points.
  const startDrawing = useCallback(() => {
    if (zoneLocked) {
      setSaveMsgText('Zone editing is locked while image processing is running.');
      setSaveMsgIsError(true);
      return;
    }
    if (drawingKind === 'site' && !projectSiteOrganizationId) {
      setSaveMsgText('Select the organization that owns this project site before drawing it.');
      setSaveMsgIsError(true);
      return;
    }
    if (!mapInstance) return;
    setDrawing(true);
    setSaveMsgText('');
    setSaveMsgIsError(false);
    setDrawnCoords(null);
    setCoveredCount(0);

    const layer = L.featureGroup().addTo(mapInstance);
    setDrawLayer(layer);
    const container = mapInstance.getContainer();
    const previewColor = KIND_COLORS[drawingKind] || KIND_COLORS.eroded;
    const trace = [];
    const draggingWasEnabled = mapInstance.dragging.enabled();
    let holdTimer = null;
    let activePointerId = null;
    let tracing = false;
    let lastContainerPoint = null;

    if (draggingWasEnabled) mapInstance.dragging.disable();
    container.style.cursor = 'crosshair';
    container.style.touchAction = 'none';
    setSaveMsgText('Press and hold on the map, then drag around the points you want.');

    const eventLatLng = (event) => {
      const containerPoint = L.DomEvent.getMousePosition(event, container);
      return { containerPoint, latlng: mapInstance.containerPointToLatLng(containerPoint) };
    };

    const renderTrace = () => {
      layer.clearLayers();
      if (trace.length > 1) {
        L.polyline(trace, { color: previewColor, weight: 3, dashArray: '6 4' }).addTo(layer);
      }
      if (trace.length > 2) {
        L.polygon(trace, {
          color: previewColor,
          weight: 2,
          fillColor: previewColor,
          fillOpacity: 0.2,
        }).addTo(layer);
      }
      const count = coveredPointCount(trace, points);
      setCoveredCount(count);
      setSaveMsgText(`${count} mapped point${count === 1 ? '' : 's'} currently covered. Release to finish.`);
      setSaveMsgIsError(false);
    };

    const appendTracePoint = (event) => {
      const { containerPoint, latlng } = eventLatLng(event);
      if (lastContainerPoint && containerPoint.distanceTo(lastContainerPoint) < 5) return;
      lastContainerPoint = containerPoint;
      trace.push([latlng.lat, latlng.lng]);
      renderTrace();
    };

    const clearHold = () => {
      if (holdTimer !== null) window.clearTimeout(holdTimer);
      holdTimer = null;
    };

    const finishTrace = (event) => {
      if (event.pointerId !== activePointerId) return;
      clearHold();
      if (!tracing) {
        setSaveMsgText('Hold for a moment, then drag around the wanted points.');
        setSaveMsgIsError(true);
        activePointerId = null;
        return;
      }
      if (trace.length < 3) {
        trace.length = 0;
        layer.clearLayers();
        setCoveredCount(0);
        setSaveMsgText('Trace a larger loop before releasing.');
        setSaveMsgIsError(true);
        tracing = false;
        activePointerId = null;
        return;
      }

      const closed = [...trace, trace[0]];
      const count = coveredPointCount(trace, points);
      layer.clearLayers();
      L.polygon(closed, {
        color: previewColor,
        weight: 2,
        fillColor: previewColor,
        fillOpacity: 0.24,
      }).addTo(layer);
      setCoveredCount(count);
      setDrawnCoords(closed.map(([lat, lng]) => [lng, lat]));
      setSaveMsgText(`Zone ready: ${count} mapped point${count === 1 ? '' : 's'} covered. Review or save it.`);
      setSaveMsgIsError(false);
      layer._cleanupFn();
    };

    const onPointerDown = (event) => {
      if (activePointerId !== null || event.button > 0) return;
      event.preventDefault();
      activePointerId = event.pointerId;
      lastContainerPoint = null;
      try { container.setPointerCapture(event.pointerId); } catch { /* optional */ }
      holdTimer = window.setTimeout(() => {
        tracing = true;
        container.style.cursor = 'grabbing';
        appendTracePoint(event);
      }, 350);
    };

    const onPointerMove = (event) => {
      if (!tracing || event.pointerId !== activePointerId) return;
      event.preventDefault();
      appendTracePoint(event);
    };

    const onContextMenu = (event) => event.preventDefault();
    container.addEventListener('pointerdown', onPointerDown);
    container.addEventListener('pointermove', onPointerMove);
    container.addEventListener('contextmenu', onContextMenu);
    window.addEventListener('pointerup', finishTrace);
    window.addEventListener('pointercancel', finishTrace);

    layer._cleanupFn = () => {
      clearHold();
      container.removeEventListener('pointerdown', onPointerDown);
      container.removeEventListener('pointermove', onPointerMove);
      container.removeEventListener('contextmenu', onContextMenu);
      window.removeEventListener('pointerup', finishTrace);
      window.removeEventListener('pointercancel', finishTrace);
      container.style.cursor = '';
      container.style.touchAction = '';
      if (draggingWasEnabled) mapInstance.dragging.enable();
    };
  }, [mapInstance, zoneLocked, drawingKind, projectSiteOrganizationId, points]);

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
    setCoveredCount(0);
    setZoneName('');
    setZoneNotes('');
    setProjectSiteOrganizationId('');
    setWarningType('deep_mud');
    setWarningSeverity('medium');
    setSaveMsgText('');
    setSaveMsgIsError(false);
  }, [drawLayer, mapInstance]);

  useEffect(() => {
    if (!zoneLocked) return undefined;
    const timer = window.setTimeout(() => {
      if (drawing) cancelDrawing();
      setSaveMsgText('Zone editing is locked while image processing is running.');
      setSaveMsgIsError(true);
    }, 0);
    return () => window.clearTimeout(timer);
  }, [zoneLocked, drawing, cancelDrawing]);

  useEffect(() => () => {
    if (!drawLayer) return;
    if (drawLayer._cleanupFn) drawLayer._cleanupFn();
    if (mapInstance && mapInstance.hasLayer(drawLayer)) {
      mapInstance.removeLayer(drawLayer);
    }
  }, [drawLayer, mapInstance]);

  // Save the drawn zone to the appropriate source. Project sites are stable
  // parent areas while assignment polygons remain automatically derived.
  const saveZone = async () => {
    if (!drawnCoords) return;
    if (zoneLocked) {
      setSaveMsgText('Zone editing is locked while image processing is running.');
      setSaveMsgIsError(true);
      return;
    }
    setSaving(true);
    setSaveMsgText('');
    setSaveMsgIsError(false);
    try {
      const geometry = { type: 'Polygon', coordinates: [drawnCoords] };

      if (drawingKind === 'site') {
        if (!adminToken) throw new Error('Your LGU session has expired. Sign in again to save a project site.');
        if (!projectSiteOrganizationId) throw new Error('Select the organization that owns this project site.');
        const name = zoneName.trim() || selectedProjectOrganization?.name || `Project Site ${projectSiteFeatures.length + 1}`;
        const res = await fetch(`${API}/api/project-sites`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            organization_id: Number(projectSiteOrganizationId),
            name,
            geometry,
            notes: zoneNotes.trim() || null,
          }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || 'Failed to save project site');
        }
        cancelDrawing();
        setSaveMsgText('Project site saved');
        setSaveMsgIsError(false);
        await Promise.all([fetchZones(), fetchPoints()]);
      } else if (drawingKind === 'warning') {
        const name = zoneName.trim() || `Warning Zone ${warningFeatures.length + 1}`;
        const res = await fetch(`${API}/api/zones/warnings`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            name,
            geometry,
            warning_type: warningType,
            severity: warningSeverity,
            notes: zoneNotes,
          }),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          throw new Error(err.detail || 'Failed to save warning zone');
        }
        cancelDrawing();
        setSaveMsgText('Warning zone saved');
        setSaveMsgIsError(false);
        await Promise.all([fetchZones(), fetchPoints()]);
      } else {
        const feature = {
          type: 'Feature',
          properties: { name: zoneName || `Eroded Zone ${erodedFeatures.length + 1}` },
          geometry,
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
        cancelDrawing();
        setSaveMsgText('Zone saved');
        setSaveMsgIsError(false);
        fetchZones();
        fetchPoints();
      }
    } catch (err) {
      setSaveMsgText(err.message || 'Failed to save zone');
      setSaveMsgIsError(true);
    } finally {
      setSaving(false);
    }
  };

  const handleDeleteProjectSite = async (site) => {
    const siteId = site?.id ?? site?.properties?.id;
    if (!siteId) return;
    if (zoneLocked) {
      setPendingProjectSiteDelete(null);
      setSaveMsgText('Zone editing is locked while image processing is running.');
      setSaveMsgIsError(true);
      return;
    }
    setProjectSiteDeleting(siteId);
    setProjectSiteDeleteError('');
    setSaveMsgText('');
    setSaveMsgIsError(false);
    try {
      if (!adminToken) throw new Error('Your LGU session has expired. Sign in again to delete a project site.');
      const res = await fetch(
        `${API}/api/project-sites/${siteId}`,
        { method: 'DELETE' },
      );
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Failed to delete project site');
      }
      setPendingProjectSiteDelete(null);
      await Promise.all([fetchZones(), fetchPoints()]);
    } catch (err) {
      setProjectSiteDeleteError(err.message || 'Failed to delete project site');
    } finally {
      setProjectSiteDeleting(null);
    }
  };

  const handleDeleteWarning = async (zone) => {
    if (!zone) return;
    if (zoneLocked) {
      setPendingWarningDelete(null);
      setSaveMsgText('Zone editing is locked while image processing is running.');
      setSaveMsgIsError(true);
      return;
    }
    setWarningDeleting(zone.id);
    setSaveMsgText('');
    setSaveMsgIsError(false);
    try {
      const res = await fetch(`${API}/api/zones/warnings/${zone.id}`, { method: 'DELETE' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Failed to delete warning zone');
      }
      setPendingWarningDelete(null);
      await Promise.all([fetchZones(), fetchPoints()]);
    } catch (err) {
      setSaveMsgText(err.message || 'Failed to delete warning zone');
      setSaveMsgIsError(true);
    } finally {
      setWarningDeleting(null);
    }
  };

  const handleDeleteEroded = async (index) => {
    if (zoneLocked) {
      setPendingDeleteIndex(null);
      setSaveMsgText('Zone editing is locked while image processing is running.');
      setSaveMsgIsError(true);
      return;
    }
    setDeleting(index);
    setSaveMsgText('');
    setSaveMsgIsError(false);
    try {
      const zone = erodedFeatures[index];
      const zoneId = zone?.id ?? zone?.properties?.id;
      if (!zoneId) throw new Error('This erosion zone has no stable database ID. Refresh the map and try again.');
      const res = await fetch(`${API}/api/zones/eroded/${zoneId}`, { method: 'DELETE' });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Failed to delete zone');
      }
      setPendingDeleteIndex(null);
      fetchZones();
      fetchPoints();
    } catch (err) {
      setSaveMsgText(err.message || 'Failed to delete zone');
      setSaveMsgIsError(true);
    } finally {
      setDeleting(null);
    }
  };

  const handleClearAll = async () => {
    if (zoneLocked) {
      setClearAllOpen(false);
      setSaveMsgText('Zone editing is locked while image processing is running.');
      setSaveMsgIsError(true);
      return;
    }
    setClearAllBusy(true);
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
      setClearAllOpen(false);
      fetchZones();
      fetchPoints();
      setSaveMsgText('All eroded zones cleared');
      setSaveMsgIsError(false);
    } catch (err) {
      setSaveMsgText(err.message || 'Failed to clear zones');
      setSaveMsgIsError(true);
    } finally {
      setClearAllBusy(false);
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
    <Panel title="Zone Editor" subtitle={`${projectSiteFeatures.length + erodedFeatures.length + forbiddenFeatures.length + warningFeatures.length} zones total`}>
      {zoneLocked && (
        <div className="zone-lock-banner">
          <strong>Zone editing locked</strong>
          <span>
            Image processing is running{processingStage ? `: ${processingStage}` : ''}.
          </span>
        </div>
      )}
      {/* Draw tools */}
      <PanelCard
        title="Draw Zone"
        icon={<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 19l7-7 3 3-7 7-3-3z"/><path d="M18 13l-1.5-7.5L2 2l3.5 14.5L13 18l5-5z"/><path d="M2 2l7.586 7.586"/><circle cx="11" cy="11" r="2"/></svg>}
      >
        {!drawing ? (
          <div>
            <div className="zone-kind-row" role="radiogroup" aria-label="Zone kind">
              <button
                type="button"
                role="radio"
                aria-checked={drawingKind === 'eroded'}
                className={`zone-kind-btn ${drawingKind === 'eroded' ? 'zone-kind-btn-active' : ''}`}
                onClick={() => {
                  setDrawingKind('eroded');
                  setProjectSiteOrganizationId('');
                }}
                disabled={zoneLocked}
              >
                <span className="zone-kind-swatch" style={{ background: KIND_COLORS.eroded }} aria-hidden="true" />
                Eroded
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={drawingKind === 'site'}
                className={`zone-kind-btn ${drawingKind === 'site' ? 'zone-kind-btn-active' : ''}`}
                onClick={() => setDrawingKind('site')}
                disabled={zoneLocked}
              >
                <span className="zone-kind-swatch" style={{ background: KIND_COLORS.site }} aria-hidden="true" />
                Project Site
              </button>
              <button
                type="button"
                role="radio"
                aria-checked={drawingKind === 'warning'}
                className={`zone-kind-btn ${drawingKind === 'warning' ? 'zone-kind-btn-active' : ''}`}
                onClick={() => {
                  setDrawingKind('warning');
                  setProjectSiteOrganizationId('');
                }}
                disabled={zoneLocked}
              >
                <span className="zone-kind-swatch" style={{ background: KIND_COLORS.warning }} aria-hidden="true" />
                Warning
              </button>
            </div>
            {drawingKind === 'site' ? (
              <div className="form-group project-site-owner-first">
                <label className="form-label" htmlFor="project-site-organization">Organization owner *</label>
                <select
                  id="project-site-organization"
                  className="form-input"
                  value={projectSiteOrganizationId}
                  onChange={(event) => {
                    const organizationId = event.target.value;
                    const organization = organizationOptions.find(
                      (option) => String(option.id) === String(organizationId),
                    );
                    setProjectSiteOrganizationId(organizationId);
                    setZoneName(organization?.name || '');
                    setSaveMsgText('');
                    setSaveMsgIsError(false);
                  }}
                  disabled={zoneLocked}
                  required
                >
                  <option value="">Select an organization before drawing</option>
                  {organizationOptions.map((organization) => (
                    <option key={organization.id} value={organization.id}>{organization.name}</option>
                  ))}
                </select>
                <small className="project-site-owner-help">
                  Organizations appear here after their first planting schedule is created.
                </small>
              </div>
            ) : null}
            <button
              className="btn btn-primary btn-sm"
              style={{ width: '100%' }}
              onClick={startDrawing}
              disabled={zoneLocked || (drawingKind === 'site' && !projectSiteOrganizationId)}
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><line x1="12" y1="5" x2="12" y2="19"/><line x1="5" y1="12" x2="19" y2="12"/></svg>
              {drawingKind === 'warning'
                ? 'Draw New Warning Zone'
                : drawingKind === 'site'
                  ? 'Draw New Project Site'
                  : 'Draw New Eroded Zone'}
            </button>
            <p className="text-sm" style={{ color: 'var(--text-muted)', marginTop: 8, lineHeight: 1.5 }}>
              {drawingKind === 'warning'
                ? 'Warning zones mark plantable points with expert notes such as deep mud, difficult access, or low survival confidence. They do not block assignment.'
                : drawingKind === 'site'
                  ? 'Long-press the map, then drag a loop around the wanted points. The live count helps you adjust the boundary before saving.'
                  : 'Click points on the map to draw an erosion polygon. Covered planting points stay visible but become Not Available for Planting. Removing the zone returns unassigned points to Planned. Click the first green point to close it.'}
            </p>
            {saveMsgText && (
              <p className="text-sm" style={{ color: saveMsgIsError ? '#991b1b' : 'var(--color-completed)', marginTop: 6 }}>
                {saveMsgText}
              </p>
            )}
          </div>
        ) : !drawnCoords ? (
          <div>
            <div className="drawing-active">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ea580c" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
              <span>Long-press, then drag around the wanted points</span>
            </div>
            <div className="zone-covered-count" role="status" aria-live="polite">
              <strong>{coveredCount}</strong>
              <span>mapped point{coveredCount === 1 ? '' : 's'} covered</span>
            </div>
            <button className="btn btn-ghost btn-sm" style={{ width: '100%', marginTop: 8 }} onClick={cancelDrawing}>
              Cancel
            </button>
            {saveMsgText && (
              <p className="text-sm" style={{ color: saveMsgIsError ? '#991b1b' : 'var(--color-completed)', marginTop: 6 }}>
                {saveMsgText}
              </p>
            )}
          </div>
        ) : (
          <div className="save-form">
            <div className="zone-covered-count is-ready" role="status">
              <strong>{coveredCount}</strong>
              <span>mapped point{coveredCount === 1 ? '' : 's'} inside this boundary</span>
            </div>
            <div className="form-group">
              <label className="form-label">
                {drawingKind === 'site' ? 'Project Site Name' : 'Zone Name'}
              </label>
              <input
                className="form-input"
                type="text"
                placeholder={drawingKind === 'warning'
                  ? `Warning Zone ${warningFeatures.length + 1}`
                  : drawingKind === 'site'
                    ? selectedProjectOrganization?.name || `Project Site ${projectSiteFeatures.length + 1}`
                    : `Eroded Zone ${erodedFeatures.length + 1}`}
                value={zoneName}
                onChange={(e) => setZoneName(e.target.value)}
              />
            </div>
            {drawingKind === 'warning' && (
              <>
                <div className="form-group">
                  <label className="form-label">Warning Type</label>
                  <select
                    className="form-input"
                    value={warningType}
                    onChange={(e) => setWarningType(e.target.value)}
                  >
                    {WARNING_TYPES.map(([value, label]) => (
                      <option key={value} value={value}>{label}</option>
                    ))}
                  </select>
                </div>
                <div className="form-group">
                  <label className="form-label">Severity</label>
                  <select
                    className="form-input"
                    value={warningSeverity}
                    onChange={(e) => setWarningSeverity(e.target.value)}
                  >
                    {WARNING_SEVERITIES.map(([value, label]) => (
                      <option key={value} value={value}>{label}</option>
                    ))}
                  </select>
                </div>
              </>
            )}
            {(drawingKind === 'warning' || drawingKind === 'site') && (
              <div className="form-group">
                <label className="form-label">Notes (optional)</label>
                <textarea
                  className="form-input"
                  rows={2}
                  placeholder={drawingKind === 'site'
                    ? 'Project location, restoration phase, access notes, or local site name'
                    : 'e.g. mud is too deep for planters during low tide, access only from the west'}
                  value={zoneNotes}
                  onChange={(e) => setZoneNotes(e.target.value)}
                />
              </div>
            )}
            <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
              <button className="btn btn-primary btn-sm" style={{ flex: 1 }} onClick={saveZone} disabled={saving || zoneLocked}>
                {saving
                  ? 'Saving...'
                  : drawingKind === 'warning'
                    ? 'Save Warning Zone'
                    : drawingKind === 'site'
                      ? 'Save Project Site'
                      : 'Save Zone'}
              </button>
              <button className="btn btn-ghost btn-sm" onClick={cancelDrawing}>Cancel</button>
            </div>
            {saveMsgText && (
              <p className="text-sm" style={{ color: saveMsgIsError ? '#991b1b' : 'var(--color-completed)', marginTop: 6 }}>
                {saveMsgText}
              </p>
            )}
          </div>
        )}
      </PanelCard>

      {/* Stable parent sites used for long-term LGU reporting. */}
      <PanelCard
        title="Project Sites"
        badge={projectSiteFeatures.length}
        icon={<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={KIND_COLORS.site} strokeWidth="2"><path d="M3 12l9-9 9 9-9 9-9-9z"/><path d="M8 12h8"/><path d="M12 8v8"/></svg>}
      >
        <div className="zone-list">
          {projectSiteFeatures.length === 0 ? (
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              No stable project sites yet. Draw one to group assignments and inspections across seasons.
            </p>
          ) : (
            projectSiteFeatures.map((feature, index) => {
              const props = feature.properties || {};
              const siteId = feature.id ?? props.id;
              return (
                <div key={siteId ?? index} className="zone-row project-site-row">
                  <div className="zone-dot" style={{ background: KIND_COLORS.site }} />
                  <div className="zone-info">
                    <span className="zone-name">{props.name || `Project Site ${index + 1}`}</span>
                    <span className="zone-area">
                      {props.organization_name ? `Owned by ${props.organization_name}` : 'Legacy site owner not recorded'}
                      {props.notes ? ` - ${props.notes}` : ''}
                      {props.assignment_count != null ? ` - ${props.assignment_count} assignments` : ''}
                    </span>
                  </div>
                  <button
                    className="btn btn-ghost btn-sm btn-icon"
                    onClick={() => {
                      setProjectSiteDeleteError('');
                      setPendingProjectSiteDelete(feature);
                    }}
                    disabled={projectSiteDeleting === siteId || zoneLocked}
                    title={`Delete ${props.name || 'project site'}`}
                    aria-label={`Delete ${props.name || 'project site'}`}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                    </svg>
                  </button>
                </div>
              );
            })
          )}
        </div>
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
          <button className="btn btn-ghost btn-sm" onClick={() => setClearAllOpen(true)} disabled={!erodedFeatures.length || zoneLocked}>
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
                  onClick={() => setPendingDeleteIndex(i)}
                  disabled={deleting === i || zoneLocked}
                  title={`Delete ${getZoneName(f, i, 'Eroded')}`}
                  aria-label={`Delete ${getZoneName(f, i, 'Eroded')}`}
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

      {/* Warning Zones (non-blocking expert/planner caution areas) */}
      <PanelCard
        title="Warning Zones"
        badge={warningFeatures.length}
        icon={<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" strokeWidth="2"><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>}
      >
        <div className="zone-list">
          {warningFeatures.length === 0 ? (
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              No warning zones yet. Draw one to tag plantable points with expert caution notes.
            </p>
          ) : (
            warningFeatures.map((f) => {
              const props = f.properties || {};
              const severity = props.severity || 'medium';
              return (
                <div key={props.id} className="zone-row warning-zone-row">
                  <div className="zone-dot" style={{ background: KIND_COLORS.warning }} />
                  <div className="zone-info">
                    <span className="zone-name">{props.name}</span>
                    <span className="zone-area">
                      {(props.warning_label || 'Planner warning')} - {severity}
                      {props.notes ? ` - ${props.notes}` : ''}
                    </span>
                  </div>
                  <span className={`warning-severity warning-severity-${severity}`}>
                    {severity}
                  </span>
                  <button
                    className="btn btn-ghost btn-sm btn-icon"
                    onClick={() => setPendingWarningDelete(props)}
                    disabled={warningDeleting === props.id || zoneLocked}
                    title={`Delete ${props.name}`}
                    aria-label={`Delete ${props.name}`}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                    </svg>
                  </button>
                </div>
              );
            })
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
      <Modal
        open={pendingDeleteIndex !== null}
        title={`Delete "${pendingDeleteIndex !== null ? getZoneName(erodedFeatures[pendingDeleteIndex], pendingDeleteIndex, 'Eroded') : ''}"?`}
        variant="danger"
        confirmLabel="Delete zone"
        cancelLabel="Cancel"
        busy={deleting !== null && deleting === pendingDeleteIndex}
        onConfirm={() => handleDeleteEroded(pendingDeleteIndex)}
        onCancel={() => { if (deleting === null) setPendingDeleteIndex(null); }}
      >
        <p>This permanently removes the eroded zone from the map. This action cannot be undone.</p>
      </Modal>

      <Modal
        open={clearAllOpen}
        title={`Clear all ${erodedFeatures.length} eroded zone${erodedFeatures.length !== 1 ? 's' : ''}?`}
        variant="danger"
        confirmLabel="Clear all zones"
        cancelLabel="Cancel"
        busy={clearAllBusy}
        onConfirm={handleClearAll}
        onCancel={() => { if (!clearAllBusy) setClearAllOpen(false); }}
      >
        <p>This permanently removes all drawn eroded zones. This action cannot be undone.</p>
      </Modal>

      <Modal
        open={pendingProjectSiteDelete !== null}
        title={`Delete "${pendingProjectSiteDelete?.properties?.name || 'project site'}"?`}
        variant="danger"
        confirmLabel="Delete project site"
        cancelLabel="Cancel"
        busy={projectSiteDeleting !== null}
        onConfirm={() => handleDeleteProjectSite(pendingProjectSiteDelete)}
        onCancel={() => {
          if (projectSiteDeleting === null) {
            setPendingProjectSiteDelete(null);
            setProjectSiteDeleteError('');
          }
        }}
      >
        <p>
          You can delete this boundary if no planting has been recorded. Unplanted assignments
          will be released and their points will return to planned. Saved image analyses and
          mapped points will remain available.
        </p>
        {projectSiteDeleteError ? (
          <p role="alert" style={{ color: '#991b1b', fontWeight: 600, marginTop: 10 }}>
            {projectSiteDeleteError}
          </p>
        ) : null}
      </Modal>

      <Modal
        open={pendingWarningDelete !== null}
        title={`Delete "${pendingWarningDelete?.name || ''}"?`}
        variant="danger"
        confirmLabel="Delete warning zone"
        cancelLabel="Cancel"
        busy={warningDeleting !== null && pendingWarningDelete && warningDeleting === pendingWarningDelete.id}
        onConfirm={() => handleDeleteWarning(pendingWarningDelete)}
        onCancel={() => { if (warningDeleting === null) setPendingWarningDelete(null); }}
      >
        <p>
          This removes the warning annotation. Points inside it are <strong>not</strong> deleted
          and remain plantable.
        </p>
      </Modal>

    </Panel>
  );
}
