import { useEffect, useRef, useState, useCallback, useMemo } from 'react';
import L from 'leaflet';
import { useMapStore } from '../stores/mapStore';
import { useProcessingStore } from '../stores/processingStore';
import { useAuthStore } from '../stores/authStore';
import { Panel, PanelCard } from '../components/Panel';
import Modal from '../components/Modal';
import useFormFeedback from '../utils/useFormFeedback';
import { FieldError, FormErrorSummary, FormMessage } from '../components/FormFeedback';
import { submissionError } from '../utils/formValidation';
import { zoneName as getZoneName } from '../utils/zoneInfo';
import { focusMapArea, organizationAreaFeatures } from '../utils/organizationArea';
import { filterMonitoringPoints } from '../utils/monitoringOrganizationFilter';
import ProjectSiteActions from '../components/ProjectSiteActions';
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
  const warningZones = useMapStore((s) => s.warningZones);
  const projectSites = useMapStore((s) => s.projectSites);
  const siteZones = useMapStore((s) => s.siteZones);
  const showLayers = useMapStore((s) => s.showLayers);
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
  const [selectedZone, setSelectedZone] = useState(null);
  const [drawLayer, setDrawLayer] = useState(null);
  const [drawnCoords, setDrawnCoords] = useState(null);
  const [boundaryStatus, setBoundaryStatus] = useState('idle');
  const [boundaryRetry, setBoundaryRetry] = useState(0);
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
  const feedback = useFormFeedback({
    organization_id: { label: 'Organization owner', serverTerms: ['organization'] },
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
  const ownerHasProjectSite = projectSiteFeatures.some((feature) => (
    String(feature.properties?.organization_id) === String(projectSiteOrganizationId)
    && ['Polygon', 'MultiPolygon'].includes(feature.geometry?.type)
  ));
  const coveredPoints = useMemo(() => (drawingKind === 'site'
    ? ownerHasProjectSite
      ? filterMonitoringPoints(points, projectSiteOrganizationId)
      : points.filter((point) => {
        const ownerId = point.source_organization_id ?? point.assigned_organization_id;
        return ownerId == null || String(ownerId) === String(projectSiteOrganizationId);
      })
    : points)
    .filter((point) => !point.deleted_at && !point.is_deleted && point.inside_visible_map !== false),
  [points, drawingKind, projectSiteOrganizationId, ownerHasProjectSite]);

  const chooseProjectOrganization = (organizationId) => {
    setProjectSiteOrganizationId(organizationId);
    feedback.clear();
    const organization = organizationOptions.find((option) => String(option.id) === String(organizationId));
    setZoneName(organization?.name || '');
    setSaveMsgText('');
    setSaveMsgIsError(false);
    setBoundaryStatus(drawnCoords ? 'checking' : 'idle');
    if (!organizationId) return;
    showLayers(['projectSites', 'siteZones', 'points']);
    const features = organizationAreaFeatures({ ...projectSites, released_areas: null }, siteZones, organizationId);
    if (!features.length) {
      setSaveMsgText('Draw this organization’s first site on unclaimed land inside the drone map.');
      return;
    }
    try {
      const bounds = L.geoJSON({ type: 'FeatureCollection', features }).getBounds();
      if (!focusMapArea(mapInstance, bounds)) throw new Error('Area unavailable');
    } catch {
      setSaveMsgText('Could not show the organization’s area. Refresh the map and try again.');
      setSaveMsgIsError(true);
    }
  };

  useEffect(() => {
    if (loaded.current) return;
    loaded.current = true;
    fetchZones();
    fetchPoints();
  }, [fetchPoints, fetchZones]);

  const erodedFeatures = erodedZones?.features || [];

  useEffect(() => {
    if (!drawnCoords || (drawingKind === 'site' && !projectSiteOrganizationId)) return undefined;
    const controller = new AbortController();
    let active = true;
    const timeout = window.setTimeout(() => controller.abort(), 15000);
    const checkBoundary = async () => {
      try {
        const response = await fetch(`${API}${drawingKind === 'site' ? '/api/project-sites/validate-boundary' : '/api/zones/validate-boundary'}`, {
          method: 'POST', headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ geometry: { type: 'Polygon', coordinates: [drawnCoords] },
            ...(drawingKind === 'site' ? { organization_id: Number(projectSiteOrganizationId) } : {}),
          }),
          signal: controller.signal,
        });
        const result = await response.json().catch(() => ({}));
        if (!active) return;
        const valid = response.ok && result.valid === true;
        setBoundaryStatus(valid ? 'valid' : response.status === 400 ? 'invalid' : 'failed');
        setSaveMsgText(valid ? 'Zone boundary checked. Ready to save.'
          : typeof result.detail === 'string' ? result.detail : 'Could not check the map boundary. Please try again.');
        setSaveMsgIsError(!valid);
        setCoveredCount(coveredPointCount(drawnCoords.map(([longitude, latitude]) => [latitude, longitude]), coveredPoints));
        drawLayer?.eachLayer((layer) => layer.setStyle?.({
          color: valid ? KIND_COLORS[drawingKind] : '#dc2626',
          fillColor: valid ? KIND_COLORS[drawingKind] : '#dc2626',
        }));
      } catch {
        if (!active) return;
        setBoundaryStatus('failed');
        setSaveMsgText('Could not check the map boundary. Check your connection and try again.');
        setSaveMsgIsError(true);
      } finally {
        window.clearTimeout(timeout);
      }
    };
    checkBoundary();
    return () => { active = false; controller.abort(); window.clearTimeout(timeout); };
  }, [drawnCoords, drawingKind, drawLayer, boundaryRetry, projectSiteOrganizationId, coveredPoints]);

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
  }, [mapInstance, zoneLocked, drawingKind, projectSiteOrganizationId, coveredPoints]);

  // Cancel drawing
  const cancelDrawing = useCallback(({ keepOwner = false } = {}) => {
    if (drawLayer) {
      if (drawLayer._cleanupFn) drawLayer._cleanupFn();
      if (mapInstance && mapInstance.hasLayer(drawLayer)) {
        mapInstance.removeLayer(drawLayer);
      }
    }
    setDrawing(false);
    setDrawLayer(null);
    setDrawnCoords(null);
    setBoundaryStatus('idle');
    setCoveredCount(0);
    setZoneName('');
    setZoneNotes('');
    if (!keepOwner) setProjectSiteOrganizationId('');
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
    if (boundaryStatus !== 'valid') return;
    if (zoneLocked) {
      setSaveMsgText('Zone editing is locked while image processing is running.');
      setSaveMsgIsError(true);
      return;
    }
    if (!feedback.validate()) return;
    if (drawingKind === 'site' && !projectSiteOrganizationId) {
      feedback.reject({ organization_id: 'Choose the organization that owns this project site.' });
      return;
    }
    setSaving(true);
    setSaveMsgText('');
    setSaveMsgIsError(false);
    try {
      const geometry = { type: 'Polygon', coordinates: [drawnCoords] };

      if (drawingKind === 'site') {
        if (!adminToken) throw new Error('Your LGU session has expired. Sign in again to save a project site.');
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
          throw submissionError(err.detail, 'Failed to save project site. Please try again.', res.status);
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
          throw submissionError(err.detail, 'Failed to save warning zone. Please try again.', res.status);
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
          throw submissionError(err.detail, 'Failed to save zone. Please try again.', res.status);
        }
        cancelDrawing();
        setSaveMsgText('Zone saved');
        setSaveMsgIsError(false);
        fetchZones();
        fetchPoints();
      }
    } catch (err) {
      if (!feedback.fromServer(err)) {
        setSaveMsgText(err.message || 'Failed to save zone');
        setSaveMsgIsError(true);
      }
    } finally {
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
    <Panel plantingTool title="Zone Editor" subtitle={`${projectSiteFeatures.length + erodedFeatures.length + warningFeatures.length} zones total`}>
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
                <select {...feedback.props('organization_id')}
                  id="project-site-organization"
                  className="form-input"
                  value={projectSiteOrganizationId}
                  onChange={(event) => chooseProjectOrganization(event.target.value)}
                  disabled={zoneLocked}
                  required
                >
                  <option value="">Select an organization before drawing</option>
                  {organizationOptions.map((organization) => (
                    <option key={organization.id} value={organization.id}>{organization.name}</option>
                  ))}
                </select>
                <FieldError feedback={feedback} field="organization_id" />
                <small className="project-site-owner-help">
                  Organizations appear here after their first planting schedule is created.
                </small>
              </div>
            ) : null}
            <button
              className="btn btn-primary btn-sm"
              style={{ width: '100%' }}
              onClick={startDrawing}
              disabled={zoneLocked || !mapInstance || (drawingKind === 'site' && !projectSiteOrganizationId)}
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
                ? 'Draw inside the drone map. Warning zones mark hazards such as deep mud or difficult access. They do not block assignment.'
                : drawingKind === 'site'
                  ? ownerHasProjectSite
                    ? 'Draw inside this organization’s area or released land available to other organizations.'
                    : 'Choose the owner, then draw its first site on unclaimed land inside the drone map. Avoid other organizations’ areas.'
                  : 'Start on the left, drag right, then trace a closed area inside the drone map. Covered points become unavailable for planting.'}
            </p>
            <FormMessage error={saveMsgIsError ? saveMsgText : ""} success={!saveMsgIsError ? saveMsgText : ""} loading={saving ? "Saving zone..." : ""} />
          </div>
        ) : !drawnCoords ? (
          <div>
            <div className="drawing-active">
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="#ea580c" strokeWidth="2"><circle cx="12" cy="12" r="10"/><path d="M12 6v6l4 2"/></svg>
              <span>Start on the left. Hold, drag right, then trace the outline.</span>
            </div>
            <div className="zone-covered-count" role="status" aria-live="polite">
              <strong>{coveredCount}</strong>
              <span>mapped point{coveredCount === 1 ? '' : 's'} covered</span>
            </div>
            <button className="btn btn-ghost btn-sm" style={{ width: '100%', marginTop: 8 }} onClick={cancelDrawing}>
              Cancel
            </button>
            <FormMessage error={saveMsgIsError ? saveMsgText : ""} success={!saveMsgIsError ? saveMsgText : ""} loading={saving ? "Saving zone..." : ""} />
          </div>
        ) : (
          <div className="save-form" onChangeCapture={feedback.onChange}>
            <FormErrorSummary feedback={feedback} />
            <FormMessage error={saveMsgIsError ? saveMsgText : ""}
              loading={boundaryStatus === 'checking' ? 'Checking the zone boundary...' : ""} />
            {['invalid', 'failed'].includes(boundaryStatus) && <div className="zone-actions" style={{ marginTop: 8 }}>
              {boundaryStatus === 'failed' && <button type="button" className="btn btn-secondary btn-sm" disabled={zoneLocked}
                onClick={() => {
                  setBoundaryStatus('checking'); setBoundaryRetry((value) => value + 1);
                  setSaveMsgText('Checking the zone boundary...'); setSaveMsgIsError(false);
                }}>Try again</button>}
              <button type="button" className="btn btn-secondary btn-sm" disabled={zoneLocked || (drawingKind === 'site' && !projectSiteOrganizationId)}
                onClick={() => { cancelDrawing({ keepOwner: drawingKind === 'site' }); startDrawing(); }}>Redraw zone</button>
            </div>}
            {drawingKind === 'site' && <label className="form-group">
              <span className="form-label">Organization owner *</span>
              <select {...feedback.props('organization_id')} className="form-input" required disabled={saving} value={projectSiteOrganizationId}
                onChange={(event) => chooseProjectOrganization(event.target.value)}>
                <option value="">Select an organization</option>
                {organizationOptions.map((organization) => <option key={organization.id} value={organization.id}>{organization.name}</option>)}
              </select>
              <FieldError feedback={feedback} field="organization_id" />
            </label>}
            <div className={`zone-covered-count${boundaryStatus === 'valid' ? ' is-ready' : ''}`} role="status">
              <strong>{coveredCount}</strong>
              <span>mapped point{coveredCount === 1 ? '' : 's'} inside this boundary</span>
            </div>
            <div className="form-group">
              <label className="form-label">
                {drawingKind === 'site' ? 'Project Site Name' : 'Zone Name'}
              </label>
              <input {...feedback.props('name')}
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
              <FieldError feedback={feedback} field="name" />
            </div>
            {drawingKind === 'warning' && (
              <>
                <div className="form-group">
                  <label className="form-label">Warning Type</label>
                  <select {...feedback.props('warning_type')}
                    className="form-input"
                    value={warningType}
                    onChange={(e) => setWarningType(e.target.value)}
                  >
                    {WARNING_TYPES.map(([value, label]) => (
                      <option key={value} value={value}>{label}</option>
                    ))}
                  </select>
              <FieldError feedback={feedback} field="warning_type" />
                </div>
                <div className="form-group">
                  <label className="form-label">Severity</label>
                  <select {...feedback.props('severity')}
                    className="form-input"
                    value={warningSeverity}
                    onChange={(e) => setWarningSeverity(e.target.value)}
                  >
                    {WARNING_SEVERITIES.map(([value, label]) => (
                      <option key={value} value={value}>{label}</option>
                    ))}
                  </select>
              <FieldError feedback={feedback} field="severity" />
                </div>
              </>
            )}
            {(drawingKind === 'warning' || drawingKind === 'site') && (
              <div className="form-group">
                <label className="form-label">Notes (optional)</label>
                <textarea {...feedback.props('notes')}
                  className="form-input"
                  rows={2}
                  placeholder={drawingKind === 'site'
                    ? 'Project location, restoration phase, access notes, or local site name'
                    : 'e.g. mud is too deep for planters during low tide, access only from the west'}
                  value={zoneNotes}
                  onChange={(e) => setZoneNotes(e.target.value)}
                />
              <FieldError feedback={feedback} field="notes" />
              </div>
            )}
            <div style={{ display: 'flex', gap: 6, marginTop: 8 }}>
              <button className="btn btn-primary btn-sm" style={{ flex: 1 }} onClick={saveZone} disabled={saving || zoneLocked || boundaryStatus !== 'valid'}>
                {saving
                  ? 'Saving...'
                  : boundaryStatus === 'checking'
                    ? 'Checking boundary...'
                  : drawingKind === 'warning'
                    ? 'Save Warning Zone'
                    : drawingKind === 'site'
                      ? 'Save Project Site'
                      : 'Save Zone'}
              </button>
              <button className="btn btn-ghost btn-sm" onClick={cancelDrawing}>Cancel</button>
            </div>
            <FormMessage success={!saveMsgIsError && boundaryStatus !== 'checking' ? saveMsgText : ""} loading={saving ? "Saving zone..." : ""} />
          </div>
        )}
      </PanelCard>

      {/* Stable parent sites used for long-term LGU reporting. */}
      <PanelCard
        title="Project Sites"
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
              No stable project sites yet. Draw one to group assignments and inspections across seasons.
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
                <button type="button" className="zone-locate" onClick={() => focusZone(f, 'eroded')}
                  aria-label={`Show ${getZoneName(f, i, 'eroded')} on map`}
                  aria-pressed={selectedZone === f} disabled={!mapInstance || drawing}>
                <span className="zone-dot" style={{ background: '#ea580c' }} aria-hidden="true" />
                <span className="zone-info">
                  <span className="zone-name">{getZoneName(f, i, 'eroded')}</span>
                  {getZoneArea(f) && <span className="zone-area">{getZoneArea(f)}</span>}
                </span>
                </button>
                <button
                  className="btn btn-ghost btn-sm btn-icon"
                  onClick={() => setPendingDeleteIndex(i)}
                  disabled={deleting === i || zoneLocked}
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
                      {(props.warning_label || 'Planner warning')} - {severity}
                      {props.notes ? ` - ${props.notes}` : ''}
                    </span>
                  </span>
                  <span className={`warning-severity warning-severity-${severity}`}>
                    {severity}
                  </span>
                  </button>
                  <button
                    className="btn btn-ghost btn-sm btn-icon"
                    onClick={() => setPendingWarningDelete(props)}
                    disabled={warningDeleting === props.id || zoneLocked}
                    title={`Delete ${getZoneName(f, i, 'warning')}`}
                    aria-label={`Delete ${getZoneName(f, i, 'warning')}`}
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
