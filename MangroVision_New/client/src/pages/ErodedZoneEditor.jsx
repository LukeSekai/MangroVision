import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import L from 'leaflet';
import { useMapStore } from '../stores/mapStore';
import { useProcessingStore } from '../stores/processingStore';
import { Panel, PanelCard } from '../components/Panel';
import Modal from '../components/Modal';
import useFormFeedback from '../utils/useFormFeedback';
import { FieldError, FormErrorSummary, FormMessage } from '../components/FormFeedback';
import { submissionError } from '../utils/formValidation';
import { zoneName as getZoneName } from '../utils/zoneInfo';
import ProjectSiteActions from '../components/ProjectSiteActions';
import { zoneVertexEditor } from '../utils/zoneVertexEditor';
import { geometryCoversPoint } from '../utils/zoneGeometry';
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
  const erodedZones = useMapStore((s) => s.erodedZones);
  const warningZones = useMapStore((s) => s.warningZones);
  const projectSites = useMapStore((s) => s.projectSites);
  const fetchZones = useMapStore((s) => s.fetchZones);
  const fetchPoints = useMapStore((s) => s.fetchPoints);
  const points = useMapStore((s) => s.points);
  const mapInstance = useMapStore((s) => s.mapInstance);
  const processing = useProcessingStore((s) => s.processing);
  const processingStage = useProcessingStore((s) => s.stage);
  const previousProcessingLock = useRef(null);

  const [deleting, setDeleting] = useState(null);
  const [pendingDeleteIndex, setPendingDeleteIndex] = useState(null);
  const [clearAllOpen, setClearAllOpen] = useState(false);
  const [clearAllBusy, setClearAllBusy] = useState(false);
  const [drawing, setDrawing] = useState(false);
  const [selectedZone, setSelectedZone] = useState(null);
  const [openCard, setOpenCard] = useState('Draw Zone');
  const [drawLayer, setDrawLayer] = useState(null);
  const [drawnCoords, setDrawnCoords] = useState(null);
  const [editingZone, setEditingZone] = useState(null);
  const [editedGeometry, setEditedGeometry] = useState(null);
  const [pendingErodedSave, setPendingErodedSave] = useState(null);
  const submitting = useRef(false);
  const draftGeometry = useMemo(() => editedGeometry || (drawnCoords
    ? { type: 'Polygon', coordinates: [drawnCoords] } : null), [editedGeometry, drawnCoords]);
  const [boundaryStatus, setBoundaryStatus] = useState('idle');
  const [boundaryRetry, setBoundaryRetry] = useState(0);
  const [coveredCount, setCoveredCount] = useState(0);
  const [zoneName, setZoneName] = useState('');
  // Stable project sites are parent boundaries; assignment zones remain
  // auto-derived and are displayed independently on the map.
  const [drawingKind, setDrawingKind] = useState('eroded');
  const [zoneNotes, setZoneNotes] = useState('');
  const [saving, setSaving] = useState(false);
  const [saveMsgText, setSaveMsgText] = useState('');
  const [saveMsgIsError, setSaveMsgIsError] = useState(false);
  const [warningType, setWarningType] = useState('deep_mud');
  const [warningSeverity, setWarningSeverity] = useState('medium');
  const feedback = useFormFeedback({
    name: { label: 'Zone name' },
    notes: { label: 'Notes' },
    warning_type: { label: 'Warning type' },
    severity: { label: 'Severity' },
  });
  const [warningDeleting, setWarningDeleting] = useState(null);
  const [pendingWarningDelete, setPendingWarningDelete] = useState(null);
  const [siteActionsId, setSiteActionsId] = useState(null);
  const [siteActionMessage, setSiteActionMessage] = useState('');
  const [siteSearch, setSiteSearch] = useState('');
  const siteSearchInput = useRef(null);
  const zoneLocked = processing;
  const warningFeatures = warningZones?.features || [];
  const projectSiteFeatures = projectSites?.features || [];
  const siteQuery = siteSearch.trim().toLowerCase();
  const matchingProjectSites = projectSiteFeatures.map((feature, index) => ({ feature, index }))
    .filter(({ feature, index }) => !siteQuery || [getZoneName(feature, index, 'site'), feature.properties?.organization_name]
      .some((value) => String(value || '').toLowerCase().includes(siteQuery)));
  const coveredPoints = useMemo(() => points.filter((point) => !point.deleted_at
    && !point.is_deleted && point.inside_visible_map !== false), [points]);

  useEffect(() => {
    if (loaded.current) return;
    loaded.current = true;
    fetchZones();
    fetchPoints();
  }, [fetchPoints, fetchZones]);

  const erodedFeatures = erodedZones?.features || [];

  useEffect(() => {
    if (!draftGeometry) return undefined;
    const controller = new AbortController();
    let active = true;
    const timeout = window.setTimeout(() => controller.abort(), 15000);
    async function check() {
      try {
        const response = await fetch(`${API}/api/zones/validate-boundary`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ geometry: draftGeometry }),
          signal: controller.signal,
        });
        const result = await response.json().catch(() => ({}));
        if (!active) return;
        const valid = response.ok && result.valid === true;
        setBoundaryStatus(valid ? 'valid' : response.status === 400 ? 'invalid' : 'failed');
        setSaveMsgText(valid ? 'Zone boundary checked. Ready to save.'
          : typeof result.detail === 'string' ? result.detail : 'Could not check the map boundary. Please try again.');
        setSaveMsgIsError(!valid);
        setCoveredCount(coveredPoints.filter((point) => geometryCoversPoint(draftGeometry, point)).length);
        drawLayer?.eachLayer((layer) => layer.setStyle?.({ color: valid ? KIND_COLORS[drawingKind] : '#dc2626',
          fillColor: valid ? KIND_COLORS[drawingKind] : '#dc2626', weight: 2, fillOpacity: .24 }));
      } catch {
        if (active) { setBoundaryStatus('failed'); setSaveMsgText('Could not check the map boundary. Check your connection and try again.'); setSaveMsgIsError(true); }
      } finally { window.clearTimeout(timeout); }
    }
    void check();
    return () => { active = false; controller.abort(); window.clearTimeout(timeout); };
  }, [draftGeometry, drawingKind, drawLayer, boundaryRetry, coveredPoints]);

  // Start freehand drawing mode. Holding for a short moment prevents an
  // accidental pan from becoming a project boundary; after the hold the
  // editor traces the user's pointer and continuously counts enclosed points.
  const startDrawing = useCallback(() => {
    if (zoneLocked) {
      setSaveMsgText('Zone editing is locked while image processing is running.');
      setSaveMsgIsError(true);
      return;
    }
    if (!mapInstance) return;
    setDrawing(true);
    setSaveMsgText('');
    setSaveMsgIsError(false);
    setDrawnCoords(null);
    setBoundaryStatus('idle');
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
    let startingContainerPoint = null;
    let beganTowardRight = false;

    if (draggingWasEnabled) mapInstance.dragging.disable();
    container.style.cursor = 'crosshair';
    container.style.touchAction = 'none';
    setSaveMsgText('Start on the left side. Press and hold, drag right, then trace the rest of the outline.');

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
      const count = coveredPointCount(trace, coveredPoints);
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
      const count = coveredPointCount(trace, coveredPoints);
      layer.clearLayers();
      L.polygon(closed, {
        color: previewColor,
        weight: 2,
        fillColor: previewColor,
        fillOpacity: 0.24,
      }).addTo(layer);
      setCoveredCount(count);
      setDrawnCoords(closed.map(([lat, lng]) => [lng, lat]));
      setBoundaryStatus('checking');
      setSaveMsgText('Checking the zone boundary...');
      setSaveMsgIsError(false);
      layer._cleanupFn();
    };

    const onPointerDown = (event) => {
      if (activePointerId !== null || event.button > 0) return;
      event.preventDefault();
      activePointerId = event.pointerId;
      lastContainerPoint = null;
      startingContainerPoint = eventLatLng(event).containerPoint;
      beganTowardRight = false;
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
      if (!beganTowardRight) {
        const movementX = eventLatLng(event).containerPoint.x - startingContainerPoint.x;
        if (movementX < -5) {
          trace.length = 0;
          layer.clearLayers();
          setCoveredCount(0);
          setSaveMsgText('Start on the left side and drag right to begin the outline.');
          setSaveMsgIsError(true);
          tracing = false;
          activePointerId = null;
          if (container.hasPointerCapture?.(event.pointerId)) container.releasePointerCapture(event.pointerId);
          return;
        }
        if (movementX < 5) return;
        beganTowardRight = true;
      }
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
  }, [mapInstance, zoneLocked, drawingKind, coveredPoints]);

  // Cancel drawing
  const cancelDrawing = useCallback(() => {
    if (drawLayer) {
      if (drawLayer._cleanupFn) drawLayer._cleanupFn();
      if (mapInstance && mapInstance.hasLayer(drawLayer)) {
        mapInstance.removeLayer(drawLayer);
      }
    }
    setDrawing(false);
    setEditingZone(null);
    setEditedGeometry(null);
    setPendingErodedSave(null);
    setDrawLayer(null);
    setDrawnCoords(null);
    setBoundaryStatus('idle');
    setCoveredCount(0);
    setZoneName('');
    setZoneNotes('');
    setWarningType('deep_mud');
    setWarningSeverity('medium');
    setSaveMsgText('');
    setSaveMsgIsError(false);
  }, [drawLayer, mapInstance]);

  const editZone = (feature) => {
    if (drawing || zoneLocked || !mapInstance) return;
    setDrawingKind('eroded');
    setOpenCard('Draw Zone');
    setEditingZone(feature);
    setEditedGeometry(JSON.parse(JSON.stringify(feature.geometry)));
    setDrawing(true);
    setZoneName(feature.properties?.name || '');
    setBoundaryStatus('checking');
    setSaveMsgText(''); setSaveMsgIsError(false);
    const layer = zoneVertexEditor(mapInstance, feature.geometry, (geometry) => {
      setEditedGeometry(geometry); setBoundaryStatus('checking');
    });
    setDrawLayer(layer);
    mapInstance.fitBounds(layer.getBounds(), { padding: [24, 24], maxZoom: 22 });
  };

  useEffect(() => {
    const staleLock = !zoneLocked && previousProcessingLock.current !== false
      && /(?:zone|project-site).*locked.*image processing/i.test(saveMsgText);
    previousProcessingLock.current = zoneLocked;
    if (!zoneLocked && !staleLock) return undefined;
    const timer = window.setTimeout(() => {
      if (zoneLocked && drawing) cancelDrawing();
      // The live banner above already explains an active lock. Do not retain a
      // processing warning as a form error after the operation has finished.
      // Keep fresh server rejections: another user's job may still be active.
      if (staleLock) { setSaveMsgText(''); setSaveMsgIsError(false); }
    }, 0);
    return () => window.clearTimeout(timer);
  }, [zoneLocked, drawing, cancelDrawing, saveMsgText]);

  useEffect(() => {
    if (drawLayer && mapInstance && !mapInstance.hasLayer(drawLayer)) drawLayer.addTo(mapInstance);
    return () => {
      if (!drawLayer) return;
      if (drawLayer._cleanupFn) drawLayer._cleanupFn();
      if (mapInstance && mapInstance.hasLayer(drawLayer)) {
        mapInstance.removeLayer(drawLayer);
      }
    };
  }, [drawLayer, mapInstance]);

  // Save the drawn zone to the appropriate source. Project sites are stable
  // parent areas while assignment polygons remain automatically derived.
  const saveZone = async (confirmed = false) => {
    if (!draftGeometry || submitting.current) return;
    if (boundaryStatus !== 'valid') return;
    if (zoneLocked) {
      setSaveMsgText('Zone editing is locked while image processing is running.');
      setSaveMsgIsError(true);
      return;
    }
    if (!feedback.validate()) return;
    submitting.current = true;
    setSaving(true);
    setSaveMsgText('');
    setSaveMsgIsError(false);
    try {
      const geometry = draftGeometry;
      if (drawingKind === 'warning') {
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
          throw submissionError(err.detail, 'Failed to save warning zone. Please try again.', res.status);
        }
        cancelDrawing();
        setSaveMsgText('Warning zone saved');
        setSaveMsgIsError(false);
        await Promise.all([fetchZones(), fetchPoints()]);
      } else {
        const feature = confirmed === true && pendingErodedSave ? pendingErodedSave : {
          type: 'Feature',
          properties: { ...editingZone?.properties, name: zoneName || `Eroded Zone ${erodedFeatures.length + 1}` },
          geometry,
        };
        const zoneId = editingZone?.id ?? editingZone?.properties?.id;
        const res = await fetch(`${API}/api/zones/eroded${zoneId != null ? `/${zoneId}` : ''}`, {
          method: zoneId != null ? 'PATCH' : 'POST',
          headers: { 'Content-Type': 'application/json', ...(confirmed === true ? { 'X-Confirm-Seedling-Overlap': 'true' } : {}) },
          body: JSON.stringify(feature),
        });
        if (!res.ok) {
          const err = await res.json().catch(() => ({}));
          if (res.status === 409 && err.detail?.code === 'seedling_overlap') {
            setPendingErodedSave(feature);
            return;
          }
          throw submissionError(err.detail, 'Failed to save zone. Please try again.', res.status);
        }
        cancelDrawing();
        setSaveMsgText('Zone saved');
        setSaveMsgIsError(false);
        await Promise.all([fetchZones({ force: true }), fetchPoints({ force: true })]);
        window.dispatchEvent(new Event('mv:data-changed'));
      }
    } catch (err) {
      if (confirmed === true) setPendingErodedSave(null);
      if (!feedback.fromServer(err)) {
        setSaveMsgText(err.message || 'Failed to save zone');
        setSaveMsgIsError(true);
      }
    } finally {
      submitting.current = false;
      setSaving(false);
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

  const focusZone = (feature, kind) => {
    if (!mapInstance || drawing) return;
    setSelectedZone(feature);
    mapInstance.fire('zonefocus', { feature, kind });
  };

  const getZoneArea = (feature) => {
    const area = feature?.properties?.area_m2 || feature?.properties?.area;
    return area ? `${Number(area).toFixed(1)} m²` : null;
  };

  return (
    <Panel plantingTool title="Zone Editor" openKey={openCard} onOpenKeyChange={setOpenCard}
      subtitle={`${projectSiteFeatures.length + erodedFeatures.length + warningFeatures.length} zones total`}>
      {zoneLocked && (
        <div className="zone-lock-banner">
          <strong>Zone editing locked</strong>
          <span>
            Image processing is running{processingStage ? `: ${processingStage}` : ''}.
          </span>
        </div>
      )}
      <PanelCard title="Draw Zone" panelKey="Draw Zone">
        {!drawing ? <div>
          <div className="zone-kind-row" role="radiogroup" aria-label="Zone kind">
            {[['eroded', 'Eroded'], ['warning', 'Warning']].map(([value, label]) => <button key={value} type="button" role="radio"
              aria-checked={drawingKind === value} className={`zone-kind-btn ${drawingKind === value ? 'zone-kind-btn-active' : ''}`}
              disabled={zoneLocked} onClick={() => setDrawingKind(value)}>
              <span className="zone-kind-swatch" style={{ background: KIND_COLORS[value] }} aria-hidden="true" />{label}
            </button>)}
          </div>
          <button type="button" className="btn btn-primary btn-sm" style={{ width: '100%', marginTop: 8 }} disabled={zoneLocked || !mapInstance} onClick={startDrawing}>
            {drawingKind === 'warning' ? 'Draw New Warning Zone' : 'Draw New Eroded Zone'}
          </button>
          <p className="text-sm">{drawingKind === 'warning' ? 'Mark hazards inside the drone map. Warning zones do not block assignment.'
            : 'Draw an eroded area inside the drone map. Covered points become unavailable for planting.'}</p>
          <p className="text-sm">Project sites are formed automatically when planting points are assigned.</p>
        </div> : !draftGeometry ? <div>
          <div className="drawing-active">Start on the left. Hold, drag right, then trace the outline.</div>
          <div className="zone-covered-count" role="status"><strong>{coveredCount}</strong><span>mapped points covered</span></div>
          <button type="button" className="btn btn-ghost btn-sm" onClick={cancelDrawing}>Cancel</button>
        </div> : <div className="save-form" onChangeCapture={feedback.onChange}>
          <FormErrorSummary feedback={feedback} />
          <FormMessage error={saveMsgIsError ? saveMsgText : ''} success={!saveMsgIsError ? saveMsgText : ''}
            loading={saving ? 'Saving zone...' : boundaryStatus === 'checking' ? 'Checking the zone boundary...' : ''} />
          {['invalid', 'failed'].includes(boundaryStatus) && <div className="zone-actions">
            {boundaryStatus === 'failed' && <button type="button" className="btn btn-secondary btn-sm" disabled={zoneLocked}
              onClick={() => { setBoundaryStatus('checking'); setBoundaryRetry((value) => value + 1); }}>Try again</button>}
            <button type="button" className="btn btn-secondary btn-sm" disabled={zoneLocked}
              onClick={() => {
                const original = editingZone, name = zoneName;
                cancelDrawing(); startDrawing();
                setEditingZone(original); setZoneName(name);
              }}>Redraw zone</button>
          </div>}
          <div className="zone-covered-count" role="status"><strong>{coveredCount}</strong><span>mapped points inside this boundary</span></div>
          <label className="form-group"><span className="form-label">Zone Name</span>
            <input {...feedback.props('name')} className="form-input" value={zoneName} onChange={(event) => setZoneName(event.target.value)} />
            <FieldError feedback={feedback} field="name" />
          </label>
          {drawingKind === 'warning' && <>
            <label className="form-group"><span className="form-label">Warning Type</span>
              <select {...feedback.props('warning_type')} className="form-input" value={warningType} onChange={(event) => setWarningType(event.target.value)}>
                {WARNING_TYPES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select><FieldError feedback={feedback} field="warning_type" />
            </label>
            <label className="form-group"><span className="form-label">Severity</span>
              <select {...feedback.props('severity')} className="form-input" value={warningSeverity} onChange={(event) => setWarningSeverity(event.target.value)}>
                {WARNING_SEVERITIES.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
              </select><FieldError feedback={feedback} field="severity" />
            </label>
            <label className="form-group"><span className="form-label">Notes (optional)</span>
              <textarea {...feedback.props('notes')} className="form-input" rows={2} value={zoneNotes} onChange={(event) => setZoneNotes(event.target.value)} />
              <FieldError feedback={feedback} field="notes" />
            </label>
          </>}
          <div className="zone-actions">
            <button type="button" className="btn btn-primary btn-sm" onClick={saveZone} disabled={saving || zoneLocked || boundaryStatus !== 'valid'}>
              {saving ? 'Saving...' : boundaryStatus === 'checking' ? 'Checking boundary...'
                : drawingKind === 'warning' ? 'Save Warning Zone' : 'Save Zone'}
            </button><button type="button" className="btn btn-ghost btn-sm" disabled={saving} onClick={cancelDrawing}>Cancel</button>
          </div>
        </div>}
        {(!drawing || !draftGeometry) && <FormMessage error={saveMsgIsError ? saveMsgText : ''} success={!saveMsgIsError ? saveMsgText : ''}
          loading={saving ? 'Saving zone...' : boundaryStatus === 'checking' ? 'Checking the zone boundary...' : ''} />}
      </PanelCard>

      {/* Stable parent sites used for long-term LGU reporting. */}
      <PanelCard
        title="Project Sites"
        panelKey="Project Sites"
        badge={projectSiteFeatures.length}
        icon={<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke={KIND_COLORS.site} strokeWidth="2"><path d="M3 12l9-9 9 9-9 9-9-9z"/><path d="M8 12h8"/><path d="M12 8v8"/></svg>}
      >
        <FormMessage success={siteActionMessage} />
        {projectSiteFeatures.length > 0 && <>
          <div className="project-site-search">
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" aria-hidden="true">
              <circle cx="10.5" cy="10.5" r="6.5" /><path d="m16 16 4 4" />
            </svg>
            <input ref={siteSearchInput} type="search" aria-label="Search project sites" placeholder="Search sites or organizations"
              value={siteSearch} onChange={(event) => { setSiteSearch(event.target.value); setSiteActionsId(null); }} />
            {siteSearch && <button type="button" aria-label="Clear site search"
              onClick={() => { setSiteSearch(''); siteSearchInput.current?.focus(); }}>Clear</button>}
          </div>
          {siteQuery && matchingProjectSites.length > 0 && <p className="project-site-search-count" role="status">
            {matchingProjectSites.length} of {projectSiteFeatures.length} sites
          </p>}
        </>}
        <div className="zone-list">
          {projectSiteFeatures.length === 0 ? (
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              No project sites yet. Assign planting points to form a site.
            </p>
          ) : matchingProjectSites.length === 0 ? (
            <p className="project-site-search-empty" role="status">No sites match “{siteSearch.trim()}”.</p>
          ) : (
            matchingProjectSites.map(({ feature, index }) => {
              const props = feature.properties || {};
              const siteId = feature.id ?? props.id;
              return (
                <div key={siteId ?? index} className="zone-row project-site-row">
                  <button type="button" className="zone-locate" onClick={() => focusZone(feature, 'site')}
                    aria-label={`Show ${getZoneName(feature, index, 'site')} on map`}
                    aria-pressed={selectedZone === feature} disabled={!mapInstance || drawing || !feature.geometry}>
                  <span className="zone-dot" style={{ background: KIND_COLORS.site }} aria-hidden="true" />
                  <span className="zone-info">
                    <span className="zone-name">{getZoneName(feature, index, 'site')}</span>
                    <span className="zone-area">
                      {props.organization_name ? `Owned by ${props.organization_name}` : 'Legacy site owner not recorded'}
                      {props.notes ? ` - ${props.notes}` : ''}
                      {props.assignment_count != null ? ` - ${props.assignment_count} assignments` : ''}
                      {props.area_released_at ? ' - Area fully released' : ''}
                    </span>
                  </span>
                  </button>
                  <ProjectSiteActions site={feature} open={siteActionsId === siteId}
                    onToggle={(open) => setSiteActionsId(open ? siteId : null)} locked={zoneLocked || drawing}
                    onComplete={async (action) => {
                      setSiteActionsId(null);
                      setSiteActionMessage(action === 'release' ? 'Area released. Planting history preserved.' : 'Site removed from the active map. History preserved.');
                      setSaveMsgText(action === 'release' ? 'Area released. Planting history preserved.' : 'Site deleted from the active map. History preserved.');
                      setSaveMsgIsError(false);
                      await Promise.all([fetchZones(), fetchPoints()]);
                      window.dispatchEvent(new Event('mv:data-changed'));
                    }} />
                </div>
              );
            })
          )}
        </div>
      </PanelCard>

      {/* Eroded Zones */}
      <PanelCard
        title="Eroded Zones"
        panelKey="Eroded Zones"
        badge={erodedFeatures.length}
        icon={<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#ea580c" strokeWidth="2"><path d="M12 22s-8-4.5-8-11.8A8 8 0 0 1 12 2a8 8 0 0 1 8 8.2c0 7.3-8 11.8-8 11.8z"/><circle cx="12" cy="10" r="3"/></svg>}
      >
        <div className="zone-actions">
          <button className="btn btn-secondary btn-sm" onClick={handleExportGeoJSON} disabled={!erodedFeatures.length}>
            Export GeoJSON
          </button>
          <button className="btn btn-ghost btn-sm" onClick={() => setClearAllOpen(true)} disabled={!erodedFeatures.length || zoneLocked || drawing}>
            Clear All
          </button>
        </div>
        <div className="zone-list">
          {erodedFeatures.length === 0 ? (
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No eroded zones defined</p>
          ) : (
            erodedFeatures.map((f, i) => (
              <div key={i} className="zone-row">
                <button type="button" className="zone-locate" onClick={() => focusZone(f, 'eroded')}
                  aria-label={`Show ${getZoneName(f, i, 'eroded')} on map`}
                  aria-pressed={selectedZone === f} disabled={!mapInstance || drawing}>
                <span className="zone-dot" style={{ background: '#ea580c' }} aria-hidden="true" />
                <span className="zone-info">
                  <span className="zone-name">{getZoneName(f, i, 'eroded')}</span>
                  {getZoneArea(f) && <span className="zone-area">{getZoneArea(f)}</span>}
                </span>
                </button>
                <button type="button" className="btn btn-ghost btn-sm" disabled={zoneLocked || drawing || !mapInstance}
                  onClick={() => editZone(f)} aria-label={`Edit ${getZoneName(f, i, 'eroded')}`}>Edit</button>
                <button
                  className="btn btn-ghost btn-sm btn-icon"
                  onClick={() => setPendingDeleteIndex(i)}
                  disabled={deleting === i || zoneLocked || drawing}
                  title={`Delete ${getZoneName(f, i, 'eroded')}`}
                  aria-label={`Delete ${getZoneName(f, i, 'eroded')}`}
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
        panelKey="Warning Zones"
        badge={warningFeatures.length}
        icon={<svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="#f59e0b" strokeWidth="2"><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0z"/><path d="M12 9v4"/><path d="M12 17h.01"/></svg>}
      >
        <div className="zone-list">
          {warningFeatures.length === 0 ? (
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              No warning zones yet. Draw one to tag plantable points with expert caution notes.
            </p>
          ) : (
            warningFeatures.map((f, i) => {
              const props = f.properties || {};
              const severity = props.severity || 'medium';
              return (
                <div key={props.id ?? i} className="zone-row warning-zone-row">
                  <button type="button" className="zone-locate" onClick={() => focusZone(f, 'warning')}
                    aria-label={`Show ${getZoneName(f, i, 'warning')} on map`}
                    aria-pressed={selectedZone === f} disabled={!mapInstance || drawing}>
                  <span className="zone-dot" style={{ background: KIND_COLORS.warning }} aria-hidden="true" />
                  <span className="zone-info">
                    <span className="zone-name">{getZoneName(f, i, 'warning')}</span>
                    <span className="zone-area">
                      {props.automatic ? 'Automatic - ' : ''}{(props.warning_label || 'Planner warning')} - {severity}
                      {props.notes ? ` - ${props.notes}` : ''}
                    </span>
                  </span>
                  <span className={`warning-severity warning-severity-${severity}`}>
                    {severity}
                  </span>
                  </button>
                  {!props.automatic && <button
                    className="btn btn-ghost btn-sm btn-icon"
                    onClick={() => setPendingWarningDelete(props)}
                    disabled={warningDeleting === props.id || zoneLocked}
                    title={`Delete ${getZoneName(f, i, 'warning')}`}
                    aria-label={`Delete ${getZoneName(f, i, 'warning')}`}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="3 6 5 6 21 6"/><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2"/>
                    </svg>
                  </button>}
                </div>
              );
            })
          )}
        </div>
      </PanelCard>

      <Modal open={pendingErodedSave !== null} title="Save eroded zone?" variant="warning"
        confirmLabel="Save zone" cancelLabel="Cancel" busy={saving} initialFocus="cancel"
        onConfirm={() => saveZone(true)} onCancel={() => { if (!saving) setPendingErodedSave(null); }}>
        <p>This area contains recorded seedlings. Their records will be kept, but new planting and replanting here will be blocked.</p>
      </Modal>

      <Modal
        open={pendingDeleteIndex !== null}
        title={`Delete "${pendingDeleteIndex !== null ? getZoneName(erodedFeatures[pendingDeleteIndex], pendingDeleteIndex, 'eroded') : ''}"?`}
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
