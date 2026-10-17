import { getPlanterColor } from '../utils/planterColors';
import { zigzagPoints } from '../utils/organizationAssignment';
import { navigableFieldPoint as canNavigateNextPoint, navigationPointQueue, navigationSegments, navigationStartPoint, retargetSiteRoute, upcomingNavigationPoints } from '../utils/fieldNavigation';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import L from 'leaflet';
import { cachedTileLayer } from '../utils/cachedTileLayer';
import 'leaflet/dist/leaflet.css';
import Modal from '../components/Modal';
import Logo from '../components/Logo';
import ActivityFeed from '../components/ActivityFeed';
import GuideButton from '../components/GuideButton';
import { bindProjectSiteInfo, emptyProjectSiteCounts, projectSitePointCounts, projectSiteStyle } from '../utils/projectSiteInfo';
import { POINT_STATUS_LABELS as STATUS_LABEL, POINT_STATUS_COLORS as STATUS_COLOR } from '../utils/pointStatus';
import useFormFeedback from '../utils/useFormFeedback';
import { FieldError } from '../components/FormFeedback';
import FieldActivities from '../components/FieldActivities';
import FieldNavigationNotice from '../components/FieldNavigationNotice';
import { usePlanterAuthStore } from '../stores/planterAuthStore';
import AuthScreen from '../components/PlanterAuthScreen';
import { ORTHOPHOTO_BOUNDS, ORTHOPHOTO_MAX_NATIVE_ZOOM, ORTHOPHOTO_TILE_URL } from '../config/mapTiles';
import { createGoogleSatelliteLayer } from '../config/googleBasemap';
import './FieldApp.css';
import '../components/ProjectSiteInfo.css';

export { AuthScreen };

const API_BASE = import.meta.env.VITE_API_BASE || '';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// Field-side palette. The planter assignment lifecycle on the database side
// uses 'pending' (assigned to me, not yet planted) and 'completed' (I have
// planted it). Both 'planned' and 'assigned' map to blue so any not-yet-done
// state is visually identical on the planter map. 'completed' / 'planted'
// is the freshly-planted state and renders YELLOW so it stands clearly
// apart from blue assigned points.
const COMPLETED_ASSIGNMENT_STATUSES = new Set(['planted', 'completed']);
const FINAL_ASSIGNMENT_STATUSES = new Set(['planted', 'completed', 'skipped']);

function canMarkPointCompleted(point) {
  return Boolean(point)
    && !point.eroded_unavailable && !point.inside_eroded_zone
    && !COMPLETED_ASSIGNMENT_STATUSES.has(point.assignment_status);
}

function getFieldPointStatus(point) {
  const assignmentStatus = point?.assignment_status;
  if (FINAL_ASSIGNMENT_STATUSES.has(assignmentStatus)) return assignmentStatus;
  if (point?.eroded_unavailable || point?.inside_eroded_zone) return 'eroded_unavailable';
  return assignmentStatus || 'pending';
}

function parseProjectSiteGeometry(value) {
  if (!value) return null;
  if (typeof value === 'object') return value;
  try {
    return JSON.parse(value);
  } catch {
    return null;
  }
}

const getFieldPointRadius = (zoom) => {
  if (zoom >= 23) return 8;
  if (zoom >= 22) return 6;
  if (zoom >= 21) return 5;
  return 4;
};

const getFieldPointWeight = (zoom) => {
  return zoom >= 22 ? 2 : 1.5;
};

function getCurrentLocation(options = {}) {
  return new Promise((resolve, reject) => {
    if (!('geolocation' in navigator)) {
      reject(new Error('Location is not supported on this device.'));
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve({ coordinates: [pos.coords.latitude, pos.coords.longitude], accuracy: pos.coords.accuracy }),
      (err) => reject(new Error(err.message || 'Could not get your location.')),
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 10000, ...options },
    );
  });
}

async function fetchRoute(origin, dest, travelMode = 'walking', accuracy = null) {
  const response = await fetch(`${API_BASE}/api/routing/compute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      origin_lat: origin[0],
      origin_lon: origin[1],
      dest_lat: dest[0],
      dest_lon: dest[1],
      travel_mode: travelMode,
      origin_accuracy_m: Number.isFinite(accuracy) ? accuracy : null,
    }),
  });
  if (!response.ok) {
    let detail = 'Could not compute route.';
    try {
      const payload = await response.json();
      if (payload?.detail) detail = payload.detail;
    } catch { /* ignore parse errors */ }
    throw new Error(detail);
  }
  return response.json();
}

function PointsMap({
  points,
  projectSites,
  selectedId,
  onSelect,
  route,
  userLocation,
  locationAccuracy,
  focusRequest,
  completionSelectedIds = [],
  startPoint,
  onStartNavigation,
  routeBusy,
}) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const layerRef = useRef(null);
  const zoneLayerRef = useRef(null);
  const markersRef = useRef(new Map());
  const routeLayerRef = useRef(null);
  const fittedRouteRef = useRef(null);
  const userMarkerRef = useRef(null);
  const hasFitBoundsRef = useRef(false);
  const hasFitPointsRef = useRef(false);
  const lastFocusRef = useRef(null);

  useEffect(() => {
    if (mapRef.current || !containerRef.current) return;
    const map = L.map(containerRef.current, {
      center: [10.78, 122.6253],
      zoom: 17,
      maxZoom: 24,
      zoomControl: false,
      attributionControl: true,
    });

    map.attributionControl.setPrefix(false);
    L.control.zoom({ position: 'bottomleft' }).addTo(map);
    L.control.scale({ imperial: false, position: 'bottomleft' }).addTo(map);
    map.createPane('orthophotoPane');
    map.getPane('orthophotoPane').style.zIndex = 250;

    createGoogleSatelliteLayer({ maxZoom: 24 }).addTo(map);
    cachedTileLayer(ORTHOPHOTO_TILE_URL, {
      pane: 'orthophotoPane',
      bounds: ORTHOPHOTO_BOUNDS,
      noWrap: true,
      minZoom: 14,
      maxZoom: 24,
      maxNativeZoom: ORTHOPHOTO_MAX_NATIVE_ZOOM,
      tms: false,
      errorTileUrl: 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
      opacity: 1,
    }).addTo(map);
    mapRef.current = map;
    zoneLayerRef.current = L.layerGroup().addTo(map);
    layerRef.current = L.layerGroup().addTo(map);
    const markers = markersRef.current;

    map.on('zoomend', () => {
      const zoom = map.getZoom();
      const radius = getFieldPointRadius(zoom);
      const weight = getFieldPointWeight(zoom);
      markersRef.current.forEach((marker) => {
        const hasWarning = Boolean(marker.options.mgWarning);
        const highlighted = marker.options.mgSelected || marker.options.mgCompletionSelected;
        marker.setRadius(highlighted ? radius + 2 : radius);
        marker.setStyle({ weight: hasWarning || highlighted ? Math.max(weight, 2) : weight });
      });
    });

    const resizeObserver = new ResizeObserver(() => map.invalidateSize({ pan: false }));
    resizeObserver.observe(containerRef.current);
    return () => {
      resizeObserver.disconnect();
      map.remove();
      mapRef.current = null;
      layerRef.current = null;
      zoneLayerRef.current = null;
      markers.clear();
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    const layer = layerRef.current;
    const zoneLayer = zoneLayerRef.current;
    if (!map || !layer || !zoneLayer) return;

    layer.clearLayers();
    zoneLayer.clearLayers();
    markersRef.current.clear();

    const siteBoundaries = [];
    const sitePointCounts = projectSitePointCounts(points, { scope: 'participant' });
    projectSites.forEach((feature) => {
      const properties = feature?.properties || {};
      const siteId = feature?.id ?? properties.id;
      const geometry = parseProjectSiteGeometry(feature?.geometry);
      if (siteId == null || !geometry) return;
      const boundary = L.geoJSON({ type: 'Feature', properties: {}, geometry }, {
        style: projectSiteStyle(feature, {
          color: '#22c55e',
          weight: 2.5,
          fillColor: '#16a34a',
          fillOpacity: 0.12,
          dashArray: '7 5',
        }),
      }).addTo(zoneLayer);
      bindProjectSiteInfo(boundary, { ...feature, geometry }, {
        counts: sitePointCounts.get(String(siteId)) || emptyProjectSiteCounts(),
        scope: 'participant',
        popupOptions: { autoPanPaddingTopLeft: [12, 120] },
      });
      siteBoundaries.push(boundary);
    });

    const validPoints = points.filter(
      (p) => typeof p.latitude === 'number' && typeof p.longitude === 'number',
    );

    const completionSelectedSet = new Set(completionSelectedIds.map(Number));
    validPoints.forEach((point) => {
      const isFinalStatus = ['planted', 'completed', 'skipped'].includes(point.assignment_status);
      const hasWarning = Boolean(point.survival_warning) && !isFinalStatus;
      const isCompletionSelected = completionSelectedSet.has(Number(point.assignment_point_id));
      const displayStatus = getFieldPointStatus(point);
      const color = ['pending', 'assigned'].includes(displayStatus) ? getPlanterColor(point.organization_id) : (STATUS_COLOR[displayStatus] || '#2563eb');
      const zoom = map.getZoom();
      const marker = L.circleMarker([point.latitude, point.longitude], {
        radius: isCompletionSelected ? getFieldPointRadius(zoom) + 0.9 : getFieldPointRadius(zoom),
        color: isCompletionSelected ? '#4c1d95' : (hasWarning ? '#f59e0b' : '#0f172a'),
        weight: isCompletionSelected ? 1.4 : (hasWarning ? Math.max(getFieldPointWeight(zoom), 1.2) : getFieldPointWeight(zoom)),
        fillColor: isCompletionSelected ? '#8b5cf6' : color,
        fillOpacity: isCompletionSelected ? 0.98 : 0.9,
        mgWarning: hasWarning,
        mgCompletionSelected: isCompletionSelected,
      });
      marker.on('click', () => onSelect(point.assignment_point_id));
      marker.bindTooltip(`Step ${point.visit_order || point.sequence_num} · Point #${point.point_num}`);
      marker.addTo(layer);
      // A generous invisible hit area makes small planting locations easier to tap.
      L.circleMarker([point.latitude, point.longitude], {
        radius: 14, stroke: false, fillOpacity: 0,
      }).on('click', () => onSelect(point.assignment_point_id)).addTo(layer);
      markersRef.current.set(point.assignment_point_id, marker);
    });

    if (startPoint) {
      const content = document.createElement('div');
      const title = document.createElement('strong');
      title.textContent = `Start at Point #${startPoint.point.point_num}`;
      const hint = document.createElement('span');
      hint.textContent = `Nearest to ${startPoint.label}`;
      const button = document.createElement('button');
      button.type = 'button';
      button.textContent = routeBusy ? 'Finding route…' : 'Navigate to this point';
      button.disabled = routeBusy;
      button.addEventListener('click', () => onStartNavigation(startPoint.point));
      content.append(title, hint, button);
      L.DomEvent.disableClickPropagation(content);
      L.DomEvent.disableScrollPropagation(content);
      L.tooltip({ permanent: true, interactive: true, direction: 'top', offset: [0, -12], className: 'field-start-point-overlay' })
        .setLatLng([startPoint.point.latitude, startPoint.point.longitude]).setContent(content).addTo(layer);
    }

    if (!hasFitPointsRef.current && validPoints.length > 0) {
      if (validPoints.length === 1) {
        // Field work starts at the actual assigned location, at planting scale.
        const only = validPoints[0];
        map.setView([only.latitude, only.longitude], 23, { animate: false });
      } else {
        const bounds = L.latLngBounds(validPoints.map((p) => [p.latitude, p.longitude]));
        map.fitBounds(bounds, { paddingTopLeft: [100, startPoint ? 140 : 40], paddingBottomRight: [100, 40], maxZoom: 23 });
      }
      hasFitBoundsRef.current = true;
      hasFitPointsRef.current = true;
    } else if (!hasFitBoundsRef.current && siteBoundaries.length > 0) {
      const bounds = L.featureGroup(siteBoundaries).getBounds();
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [35, 35], maxZoom: 19 });
        hasFitBoundsRef.current = true;
      }
    }
  }, [points, projectSites, onSelect, completionSelectedIds, startPoint, onStartNavigation, routeBusy]);

  useEffect(() => {
    markersRef.current.forEach((marker, id) => {
      const map = mapRef.current;
      const zoom = map?.getZoom() ?? 17;
      const isCompletionSelected = Boolean(marker.options.mgCompletionSelected);
      marker.options.mgSelected = id === selectedId;
      marker.setStyle({
        weight: id === selectedId || isCompletionSelected ? Math.max(1.2, getFieldPointWeight(zoom) + 0.8) : getFieldPointWeight(zoom),
        color: id === selectedId ? '#dc2626' : (isCompletionSelected ? '#4c1d95' : '#0f172a'),
      });
      marker.setRadius(id === selectedId || isCompletionSelected ? getFieldPointRadius(zoom) + 2 : getFieldPointRadius(zoom));
    });
    const selected = points.find((p) => p.assignment_point_id === selectedId);
    const map = mapRef.current;
    if (selected && map) {
      map.setView([selected.latitude, selected.longitude], Math.max(map.getZoom(), 22));
    }
  }, [selectedId, points]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (routeLayerRef.current) {
      routeLayerRef.current.eachLayer(layer => layer.unbindTooltip?.());
      routeLayerRef.current.remove();
      routeLayerRef.current = null;
    }

    if (route && (route.polyline?.length >= 2 || route.target)) {
      const routeGroup = L.layerGroup().addTo(map);
      const segments = navigationSegments(route, userLocation || route.origin, locationAccuracy);
      const bounds = L.latLngBounds([]);
      segments.forEach((segment) => {
        const isGuidance = segment.kind === 'guidance';
        const line = L.polyline(segment.polyline, {
          color: isGuidance ? '#ea580c' : '#4169e1', weight: 4, opacity: 0.95,
          dashArray: isGuidance ? '8 8' : null, lineCap: 'round', lineJoin: 'round',
        }).bindTooltip(isGuidance ? 'Follow your assigned points in order using marked lanes' : 'Follow the mapped access path and bridge').addTo(routeGroup);
        bounds.extend(line.getBounds());
      });

      if (route.entrance && segments.some(segment => segment.kind === 'road')) {
        L.circleMarker(route.entrance, { radius: 6, color: '#475569', fillColor: '#fff', fillOpacity: 1 })
          .bindTooltip('Bridge exit to planting area').addTo(routeGroup);
        bounds.extend(route.entrance);
      }
      if (route.access_start && segments.some(segment => segment.kind === 'road'
        && segment.polyline.some(([lat, lon]) => lat === route.access_start[0] && lon === route.access_start[1]))) {
        L.circleMarker(route.access_start, { radius: 8, color: '#15803d', fillOpacity: 0.8 })
          .bindTooltip('Access road starts here').addTo(routeGroup);
        bounds.extend(route.access_start);
      }
      if (route.target) {
        L.circleMarker(route.target, { radius: 8, color: '#f97316', fillOpacity: 0.2 })
          .bindTooltip(`Point #${route.pointNum}`, { permanent: true }).addTo(routeGroup);
        bounds.extend(route.target);
      }
      if (userLocation || route.origin) bounds.extend(userLocation || route.origin);

      routeLayerRef.current = routeGroup;
      const fitKey = route.navigationId ?? route;
      if (bounds.isValid() && fittedRouteRef.current !== fitKey) {
        fittedRouteRef.current = fitKey;
        const noticeHeight = containerRef.current?.parentElement?.querySelector('.field-navigation')?.offsetHeight || 0;
        map.fitBounds(bounds, { paddingTopLeft: [60, 30],
          paddingBottomRight: [30, Math.min(noticeHeight + 30, map.getSize().y * 0.4)], maxZoom: 23 });
      }
    }
  }, [route, userLocation, locationAccuracy]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (!userLocation) {
      userMarkerRef.current?.marker.unbindTooltip();
      userMarkerRef.current?.group.remove();
      userMarkerRef.current = null;
      return;
    }
    if (!userMarkerRef.current) {
      const group = L.layerGroup().addTo(map);
      const accuracyCircle = L.circle(userLocation, { radius: 0, color: '#2563eb',
        weight: 1, fillOpacity: 0.08, interactive: false }).addTo(group);
      const marker = L.circleMarker(userLocation, {
        radius: 7,
        color: '#ffffff',
        weight: 2,
        fillColor: '#2563eb',
        fillOpacity: 1,
      }).addTo(group);
      userMarkerRef.current = { group, marker, accuracyCircle, navigating: null };
    }
    const current = userMarkerRef.current;
    current.marker.setLatLng(userLocation);
    current.accuracyCircle.setLatLng(userLocation).setRadius(Number.isFinite(locationAccuracy) ? locationAccuracy : 0);
    if (current.navigating !== Boolean(route)) {
      current.marker.unbindTooltip().bindTooltip('You are here', { permanent: Boolean(route), direction: 'top' });
      current.navigating = Boolean(route);
    }
  }, [userLocation, locationAccuracy, route]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !focusRequest || lastFocusRef.current === focusRequest) return;
    lastFocusRef.current = focusRequest;
    if (focusRequest.kind === 'location' && userLocation) {
      map.setView(userLocation, 20);
    } else if (focusRequest.kind === 'point') {
      map.setView(focusRequest.coordinates, Math.max(map.getZoom(), 22));
    } else if (focusRequest.kind === 'points') {
      const locations = points.filter((p) => Number.isFinite(p.latitude) && Number.isFinite(p.longitude));
      if (locations.length) {
        map.fitBounds(L.latLngBounds(locations.map((p) => [p.latitude, p.longitude])), { padding: [40, 40], maxZoom: 23 });
      } else if (projectSites.length) {
        const bounds = L.geoJSON(projectSites).getBounds();
        if (bounds.isValid()) map.fitBounds(bounds, { padding: [40, 40], maxZoom: 20 });
      }
    }
    // A focus request is an explicit button action, not a live GPS-follow mode.
  }, [focusRequest, userLocation, points, projectSites]);

  return <div ref={containerRef} className="field-map" />;
}

function PointActionSheet({ point, open, onClose, onNavigate, onMark, busy, actionError, routeBusy, routeError }) {
  const [view, setView] = useState('actions');
  const dialogRef = useRef(null);

  useEffect(() => {
    if (!open) return;
    const previousFocus = document.activeElement;
    dialogRef.current?.focus();
    const handleKeyDown = (event) => {
      if (event.key === 'Escape') onClose();
      if (event.key !== 'Tab') return;
      const controls = dialogRef.current?.querySelectorAll('button:not([disabled])');
      if (!controls?.length) return;
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && (document.activeElement === first || document.activeElement === dialogRef.current)) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => {
      document.removeEventListener('keydown', handleKeyDown);
      if (previousFocus?.isConnected) previousFocus.focus();
    };
  }, [open, onClose]);

  const status = point?.assignment_status;
  const isPlanted = status === 'planted' || status === 'completed';
  const isSkipped = status === 'skipped';
  const isUnavailable = Boolean(point?.eroded_unavailable || point?.inside_eroded_zone)
    && !isPlanted
    && !isSkipped;
  const displayStatus = isUnavailable ? 'eroded_unavailable' : status;
  const hasPlannerWarning = Boolean(point?.survival_warning);
  const warningSeverity = point?.warning_severity || 'medium';
  const warningSummary = point?.warning_summary || 'Planner warning';
  const isFinal = isPlanted || isSkipped;
  const badgeStatus = isUnavailable ? 'eroded' : isSkipped ? 'skipped' : isPlanted ? 'planted' : 'pending';
  const badgeLabel = isUnavailable ? STATUS_LABEL.unavailable : isSkipped ? STATUS_LABEL.skipped : isPlanted ? STATUS_LABEL.planted : STATUS_LABEL.assigned;

  return (
    <>
      <div
        className={`field-overlay-backdrop ${open ? 'field-overlay-backdrop-open' : ''}`}
        onClick={onClose}
        aria-hidden={!open}
      />
      <div
        ref={dialogRef}
        tabIndex={-1}
        className={`field-overlay ${open ? 'field-overlay-open' : ''}`}
        role="dialog"
        aria-modal={open}
        aria-hidden={!open}
        aria-label={point ? `Point ${point.point_num} actions` : 'Point actions'}
      >
        <button
          type="button"
          className="field-overlay-close"
          onClick={onClose}
          aria-label="Close"
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round">
            <line x1="6" y1="6" x2="18" y2="18" />
            <line x1="18" y1="6" x2="6" y2="18" />
          </svg>
        </button>

        {point && view === 'actions' && (
          <div className="field-overlay-body">
            <div className="field-overlay-header">
              <div className="field-overlay-title">Step {point.visit_order || point.sequence_num} · Point #{point.point_num}</div>
              <span className={`field-badge field-badge-${badgeStatus}`}>
                {badgeLabel}
              </span>
            </div>
            <div className="field-overlay-actions">
              <button type="button" className="field-btn field-btn-primary" onClick={() => onMark(point, 'completed')} disabled={busy || isFinal || isUnavailable}>
                {isPlanted ? 'Already Planted' : busy ? 'Saving…' : 'Mark as Planted'}
              </button>
              <button type="button" className="field-btn field-btn-secondary field-btn-route" onClick={() => onNavigate(point)} disabled={routeBusy || isUnavailable || isFinal}>
                {routeBusy ? 'Finding route…' : 'Navigate to this point'}
              </button>
              <button type="button" className="field-btn field-btn-outline" onClick={() => setView('details')}>View Details</button>
              <button type="button" className="field-btn field-btn-skip" onClick={() => onMark(point, 'skipped')} disabled={busy || isFinal || isUnavailable}>
                {isSkipped ? 'Skipped' : busy ? 'Saving…' : 'Skip Point'}
              </button>
            </div>
            {isUnavailable && (
              <div className="field-warning">
                This point is inside an eroded zone and is currently Unavailable for planting. It will return to Planned when the LGU/Admin removes the zone.
              </div>
            )}
            {hasPlannerWarning && (
              <div className="field-warning">
                Planner warning ({warningSeverity}): {warningSummary}
              </div>
            )}
            {actionError && <div className="field-error">{actionError}</div>}
            {routeError && <div className="field-error" role="alert">{routeError}</div>}
          </div>
        )}

        {point && view === 'details' && (
          <div className="field-overlay-body">
            <button
              type="button"
              className="field-overlay-back"
              onClick={() => setView('actions')}
            >
              ← Back
            </button>
            <div className="field-overlay-title">Step {point.visit_order || point.sequence_num} · Point #{point.point_num}</div>
            <div className="field-overlay-detail-rows">
              <div className="field-overlay-detail-row">
                <span>Coordinates</span>
                <span>{point.latitude.toFixed(6)}, {point.longitude.toFixed(6)}</span>
              </div>
              {point.title && (
                <div className="field-overlay-detail-row">
                  <span>Assignment</span>
                  <span>{point.title}</span>
                </div>
              )}
              {point.species && (
                <div className="field-overlay-detail-row">
                  <span>Species</span>
                  <span>{point.species}</span>
                </div>
              )}
              {point.image_name && (
                <div className="field-overlay-detail-row">
                  <span>Source image</span>
                  <span>{point.image_name}</span>
                </div>
              )}
              {typeof point.buffer_m === 'number' && (
                <div className="field-overlay-detail-row">
                  <span>Buffer</span>
                  <span>{point.buffer_m.toFixed(1)} m</span>
                </div>
              )}
              {point.assigned_date && (
                <div className="field-overlay-detail-row">
                  <span>Assigned</span>
                  <span>{point.assigned_date}</span>
                </div>
              )}
              {point.notes && (
                <div className="field-overlay-detail-row">
                  <span>Notes</span>
                  <span>{point.notes}</span>
                </div>
              )}
              <div className="field-overlay-detail-row">
                <span>Status</span>
                <span>{STATUS_LABEL[displayStatus] || displayStatus}</span>
              </div>
              {hasPlannerWarning && (
                <div className="field-overlay-detail-row">
                  <span>Warning</span>
                  <span>{warningSeverity}: {warningSummary}</span>
                </div>
              )}
            </div>
          </div>
        )}
      </div>
    </>
  );
}

export default function FieldApp() {
  const isAuthenticated = usePlanterAuthStore((s) => s.isAuthenticated);
  const planter = usePlanterAuthStore((s) => s.planter);
  const hydrateSession = usePlanterAuthStore((s) => s.hydrateSession);
  const logout = usePlanterAuthStore((s) => s.logout);
  const fetchFieldPoints = usePlanterAuthStore((s) => s.fetchFieldPoints);
  const markPointStatus = usePlanterAuthStore((s) => s.markPointStatus);
  const markAllPointsCompleted = usePlanterAuthStore((s) => s.markAllPointsCompleted);

  const [points, setPoints] = useState([]);
  const [projectSites, setProjectSites] = useState([]);
  const [siteAccess, setSiteAccess] = useState([]);
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState('');
  const [selectedId, setSelectedId] = useState(null);
  const [markBusy, setMarkBusy] = useState(false);
  const [markError, setMarkError] = useState('');
  const [route, setRoute] = useState(null);
  const [userLocation, setUserLocation] = useState(null);
  const [locationAccuracy, setLocationAccuracy] = useState(null);
  const [locationBusy, setLocationBusy] = useState(false);
  const [locationError, setLocationError] = useState('');
  const [focusRequest, setFocusRequest] = useState(null);
  const [pointListOpen, setPointListOpen] = useState(false);
  const [pointFilter, setPointFilter] = useState('all');
  const [routeBusy, setRouteBusy] = useState(false);
  const [routeError, setRouteError] = useState('');
  const [navigationLocationError, setNavigationLocationError] = useState('');
  const [welcomeOpen, setWelcomeOpen] = useState(false);
  // 'register' = first-time sign-up → greet with "Welcome".
  // 'login'    = returning planter   → greet with "Welcome back".
  const [welcomeKind, setWelcomeKind] = useState('login');

  // Two-step plant flow: confirmation modal asks before the API call,
  // success modal acknowledges after. Both are state-driven so they survive
  // tab switches in the same window.
  const [pendingMarkPoint, setPendingMarkPoint] = useState(null);
  const [pendingMarkStatus, setPendingMarkStatus] = useState('completed');
  const [pendingSkipReason, setPendingSkipReason] = useState('');
  const skipFeedback = useFormFeedback({ skip_reason: { label: 'Reason for skipping', serverTerms: ['skip reason', 'skip_reason'] } });
  const [completedMarkPoint, setCompletedMarkPoint] = useState(null);
  const [completedMarkStatus, setCompletedMarkStatus] = useState('completed');
  const [markAllChooseMode, setMarkAllChooseMode] = useState(false);
  const [selectedMarkAllPointIds, setSelectedMarkAllPointIds] = useState([]);
  const [pendingMarkAllOpen, setPendingMarkAllOpen] = useState(false);
  const [markAllBusy, setMarkAllBusy] = useState(false);
  const [markAllError, setMarkAllError] = useState('');
  const [completedMarkAllResult, setCompletedMarkAllResult] = useState(null);

  const [avatarMenuOpen, setAvatarMenuOpen] = useState(false);
  const [activityOpen, setActivityOpen] = useState(false);
  const avatarMenuRef = useRef(null);
  const reloadSequence = useRef(0);
  const navigationRequest = useRef(0);
  const navigationRoute = useMemo(() => route ? {
    ...route, upcomingPoints: upcomingNavigationPoints(points, route),
  } : null, [route, points]);
  const recommendedStart = useMemo(() => navigationStartPoint(points, siteAccess), [points, siteAccess]);

  useEffect(() => () => { navigationRequest.current += 1; }, [isAuthenticated]);

  useEffect(() => {
    if (!avatarMenuOpen) return undefined;
    const handlePointerDown = (event) => {
      if (avatarMenuOpen && avatarMenuRef.current && !avatarMenuRef.current.contains(event.target)) {
        setAvatarMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handlePointerDown);
    document.addEventListener('touchstart', handlePointerDown);
    return () => {
      document.removeEventListener('mousedown', handlePointerDown);
      document.removeEventListener('touchstart', handlePointerDown);
    };
  }, [avatarMenuOpen]);

  const handleSelect = useCallback((id) => {
    if (markAllChooseMode) {
      const point = points.find((p) => p.assignment_point_id === id);
      if (!canMarkPointCompleted(point)) {
        setMarkAllError('Choose an available point that is not already planted.');
        return;
      }
      setMarkAllError('');
      setSelectedId(null);
      setSelectedMarkAllPointIds((currentIds) => {
        const numericId = Number(id);
        const selectedIds = new Set(currentIds.map(Number));
        if (selectedIds.has(numericId)) selectedIds.delete(numericId);
        else selectedIds.add(numericId);
        return [...selectedIds];
      });
      return;
    }
    setPointListOpen(false);
    setRouteError('');
    setSelectedId(id);
  }, [markAllChooseMode, points]);

  const handleCloseSheet = useCallback(() => {
    setSelectedId(null);
    setMarkError('');
  }, []);

  const handleStartNavigation = useCallback(async (point, continueNavigation = false) => {
    setRouteError('');
    if (point?.eroded_unavailable || point?.inside_eroded_zone) {
      setRouteError('This point is Unavailable for planting while it remains inside an eroded zone.');
      return;
    }
    const latitude = Number(point?.latitude);
    const longitude = Number(point?.longitude);
    if (!canNavigateNextPoint(point)) {
      setRouteError('This planting point has no valid map location.');
      return;
    }
    const request = ++navigationRequest.current;
    const remainingPoints = continueNavigation && route ? upcomingNavigationPoints(points, route) : points;
    const pointIds = (continueNavigation ? remainingPoints
      : navigationPointQueue(remainingPoints, point.assignment_point_id, recommendedStart?.entrance))
      .map(candidate => candidate.assignment_point_id);
    setRouteBusy(true);
    if (!continueNavigation) setRoute(null);
    try {
      let location;
      try {
        location = await getCurrentLocation({ maximumAge: 0 });
      } catch (locationError) {
        throw new Error(`${locationError.message || 'Could not get your GPS position.'} Allow location access to navigate from where you are now.`);
      }
      const origin = location.coordinates;
      const accuracy = location.accuracy;
      if (request !== navigationRequest.current) return;
      setUserLocation(origin);
      setLocationAccuracy(accuracy);
      const data = (continueNavigation && retargetSiteRoute(route, point, origin, accuracy))
        || await fetchRoute(origin, [latitude, longitude], 'walking', accuracy);
      if (request !== navigationRequest.current) return;
      setNavigationLocationError('');
      setRoute({ ...data, origin, pointNum: point.point_num, assignmentPointId: point.assignment_point_id,
        pointIds, navigationId: continueNavigation ? route.navigationId : request });
      setPointListOpen(false);
      setSelectedId(null);
      if (continueNavigation) setFocusRequest({ kind: 'point', coordinates: [latitude, longitude] });
    } catch (error) {
      if (request !== navigationRequest.current) return;
      // Never silently replace denied/unavailable GPS with an organization
      // base or the access road. Navigation starts at this participant's fix.
      if (!continueNavigation) setRoute(null);
      setRouteError(error.message || 'Could not display navigation. Please retry.');
    } finally {
      if (request === navigationRequest.current) setRouteBusy(false);
    }
  }, [points, route, recommendedStart]);

  useEffect(() => {
    if (!isAuthenticated || !route || !navigator.geolocation) return undefined;
    const watch = navigator.geolocation.watchPosition((position) => {
      setUserLocation([position.coords.latitude, position.coords.longitude]);
      setLocationAccuracy(position.coords.accuracy);
      setNavigationLocationError('');
    }, () => {
      setNavigationLocationError('GPS updates paused. Enable location access to update your position.');
    }, { enableHighAccuracy: true, maximumAge: 1000, timeout: 20000 });
    return () => navigator.geolocation.clearWatch(watch);
  }, [route, isAuthenticated]);

  const handleLocate = async () => {
    setLocationBusy(true);
    setLocationError('');
    try {
      const location = await getCurrentLocation({ maximumAge: 0 });
      setUserLocation(location.coordinates);
      setLocationAccuracy(location.accuracy);
      setFocusRequest({ kind: 'location' });
    } catch (error) {
      setLocationError(error.message || 'Could not find your location. Enable location access and try again.');
    } finally {
      setLocationBusy(false);
    }
  };

  const handleClearRoute = useCallback(() => {
    navigationRequest.current += 1;
    setRoute(null);
    setRouteBusy(false);
    setRouteError('');
  }, []);

  useEffect(() => {
    hydrateSession();
  }, [hydrateSession]);

  useEffect(() => {
    if (!isAuthenticated) return undefined;
    const check = () => { if (document.visibilityState !== 'hidden') void hydrateSession(); };
    const timer = window.setInterval(check, 30000);
    window.addEventListener('focus', check);
    document.addEventListener('visibilitychange', check);
    return () => {
      window.clearInterval(timer);
      window.removeEventListener('focus', check);
      document.removeEventListener('visibilitychange', check);
    };
  }, [isAuthenticated, hydrateSession]);

  const reload = useCallback(async () => {
    if (!isAuthenticated) return;
    const sequence = ++reloadSequence.current;
    setLoading(true);
    setLoadError('');
    try {
      const workspace = await fetchFieldPoints();
      if (sequence !== reloadSequence.current) return;
      setPoints(zigzagPoints(workspace.points).map((point, index) => ({ ...point, visit_order: index + 1 })));
      setProjectSites(workspace.projectSites);
      setSiteAccess(workspace.siteAccess || []);
    } catch (error) {
      if (sequence === reloadSequence.current) setLoadError(error.message || 'Could not load points');
    } finally {
      if (sequence === reloadSequence.current) setLoading(false);
    }
  }, [isAuthenticated, fetchFieldPoints]);

  useEffect(() => {
    const timer = window.setTimeout(() => { void reload(); }, 0);
    return () => {
      window.clearTimeout(timer);
      reloadSequence.current += 1;
    };
  }, [reload]);

  useEffect(() => {
    if (isAuthenticated && sessionStorage.getItem('mv_field_show_welcome') === '1') {
      // Use the kind flag set by the auth store to pick "Welcome" vs
      // "Welcome back". Default to 'login' if the flag is missing so an
      // unknown state still produces the safer "Welcome back" greeting.
      const kind = sessionStorage.getItem('mv_field_welcome_kind') || 'login';
      const timer = window.setTimeout(() => {
        setWelcomeKind(kind === 'register' ? 'register' : 'login');
        setWelcomeOpen(true);
      }, 0);
      return () => window.clearTimeout(timer);
    }
  }, [isAuthenticated, planter?.id]);

  const closeWelcome = () => {
    sessionStorage.removeItem('mv_field_show_welcome');
    sessionStorage.removeItem('mv_field_welcome_kind');
    setWelcomeOpen(false);
  };

  const selected = useMemo(
    () => points.find((p) => p.assignment_point_id === selectedId) || null,
    [points, selectedId],
  );
  const markAllCandidatePoints = points.filter(canMarkPointCompleted);
  const markAllCandidateCount = markAllCandidatePoints.length;
  const selectedMarkAllPointSet = new Set(selectedMarkAllPointIds.map(Number));
  const selectedMarkAllPoints = points.filter((point) => (
    selectedMarkAllPointSet.has(Number(point.assignment_point_id))
    && canMarkPointCompleted(point)
  ));
  const selectedMarkAllCount = selectedMarkAllPoints.length;
  const assignedCount = points.length;
  const pending = points.filter((p) => !FINAL_ASSIGNMENT_STATUSES.has(p.assignment_status)).length;
  const plantedCount = points.filter((p) => COMPLETED_ASSIGNMENT_STATUSES.has(p.assignment_status)).length;
  const skippedCount = points.filter((p) => p.assignment_status === 'skipped').length;
  const unavailableCount = points.filter((p) => getFieldPointStatus(p) === 'eroded_unavailable').length;
  const nextPoint = route ? navigationRoute.upcomingPoints[0] : recommendedStart?.point || points.find(canNavigateNextPoint);
  const progress = assignedCount ? Math.round(plantedCount / assignedCount * 100) : 0;
  const visibleListPoints = points.filter((point) => {
    if (pointFilter === 'all') return true;
    const status = getFieldPointStatus(point);
    if (pointFilter === 'pending') return canNavigateNextPoint(point);
    if (pointFilter === 'planted') return COMPLETED_ASSIGNMENT_STATUSES.has(status);
    return status === pointFilter;
  });
  const assignedZones = projectSites.map((feature) => ({
    id: feature.id ?? feature.properties?.id,
    name: feature.properties?.name || 'Organization zone',
  }));

  // PointActionSheet now opens the confirmation modal instead of calling
  // the API directly, so a planter never marks a point as planted by
  // accident. We also close the bottom action sheet immediately so the
  // confirmation modal isn't half-hidden behind it on phone screens.
  // If the planter cancels, they re-tap the marker to reopen the sheet.
  const handleMark = (point, status) => {
    skipFeedback.clear();
    setMarkError('');
    if (point?.eroded_unavailable || point?.inside_eroded_zone) {
      setMarkError('This point is Unavailable for planting while it remains inside an eroded zone.');
      return;
    }
    setSelectedId(null);
    setPendingMarkStatus(status === 'skipped' ? 'skipped' : 'completed');
    setPendingSkipReason('');
    setPendingMarkPoint(point);
  };

  const cancelMark = () => {
    if (markBusy) return;
    setPendingMarkPoint(null);
    setPendingSkipReason('');
  };

  const confirmMark = async () => {
    const point = pendingMarkPoint;
    if (!point) return;
    if (pendingMarkStatus === 'skipped' && !pendingSkipReason.trim()) {
      skipFeedback.reject({ skip_reason: 'Describe why this point cannot be planted, such as blocked access, deep mud, or unsafe tide.' });
      return;
    }
    setMarkBusy(true);
    setMarkError('');
    try {
      await markPointStatus(
        point.assignment_point_id,
        pendingMarkStatus,
        pendingMarkStatus === 'skipped' ? pendingSkipReason.trim() : null,
      );
      await reload();
      setRouteError('');
      // Close the confirmation modal AND the action sheet, then open the
      // success modal so the planter sees explicit acknowledgement.
      setPendingMarkPoint(null);
      setPendingSkipReason('');
      setSelectedId(null);
      setCompletedMarkStatus(pendingMarkStatus);
      setCompletedMarkPoint(point);
    } catch (error) {
      setMarkError(error.message || 'Could not update point status');
    } finally {
      setMarkBusy(false);
    }
  };

  const closeCompletionModal = () => {
    setCompletedMarkPoint(null);
  };

  const handleStartMarkAll = () => {
    setMarkAllError('');
    setMarkError('');
    setSelectedId(null);
    if (markAllCandidateCount === 0) {
      setMarkAllError('There are no available Assigned or Skipped points to mark as Planted.');
      return;
    }
    handleClearRoute();
    setMarkAllChooseMode(true);
    setPointListOpen(true);
    setPointFilter('all');
    setSelectedMarkAllPointIds([]);
  };

  const cancelMarkAll = () => {
    if (markAllBusy) return;
    setMarkAllError('');
    setPendingMarkAllOpen(false);
  };

  const confirmMarkAll = async () => {
    const selectedIds = selectedMarkAllPoints.map((point) => point.assignment_point_id);
    if (!selectedIds.length) {
      setMarkAllError('Choose at least one completed point first.');
      return;
    }
    setMarkAllBusy(true);
    setMarkAllError('');
    try {
      const result = await markAllPointsCompleted(selectedIds);
      await reload();
      setRoute(null);
      setRouteError('');
      setPendingMarkAllOpen(false);
      setMarkAllChooseMode(false);
      setSelectedMarkAllPointIds([]);
      setCompletedMarkAllResult(result);
    } catch (error) {
      setMarkAllError(error.message || 'Could not mark all points completed');
    } finally {
      setMarkAllBusy(false);
    }
  };

  const closeMarkAllCompletion = () => {
    setCompletedMarkAllResult(null);
  };

  const reviewMarkAllSelection = () => {
    setMarkAllError('');
    if (selectedMarkAllCount === 0) {
      setMarkAllError('Choose at least one completed point first.');
      return;
    }
    setPendingMarkAllOpen(true);
  };

  const cancelMarkAllSelection = () => {
    if (markAllBusy) return;
    setMarkAllChooseMode(false);
    setSelectedMarkAllPointIds([]);
    setPendingMarkAllOpen(false);
    setMarkAllError('');
  };

  if (!isAuthenticated) {
    return <AuthScreen />;
  }

  return (
    <div className="field-shell">
      <header className="field-header">
        <div className="field-header-brand">
          <Logo variant="icon" size={30} alt="MangroVision" />
          <span className="field-header-brand-name">MangroVision</span>
        </div>
        <div className="field-header-actions" ref={avatarMenuRef}>
          <GuideButton mode="field" />
          <button
            type="button"
            className={`field-avatar-button${avatarMenuOpen ? ' field-avatar-button-open' : ''}`}
            onClick={() => setAvatarMenuOpen((value) => !value)}
            aria-haspopup="menu"
            aria-expanded={avatarMenuOpen}
            aria-label="Account menu"
          >
            {(planter?.full_name || 'P').trim().charAt(0).toUpperCase()}
          </button>
          {avatarMenuOpen && (
            <div className="field-avatar-menu" role="menu">
              <div className="field-avatar-menu-header">
                <div className="field-avatar-menu-avatar" aria-hidden="true">
                  {(planter?.full_name || 'P').trim().charAt(0).toUpperCase()}
                </div>
                <div className="field-avatar-menu-info">
                  <div className="field-avatar-menu-name">{planter?.full_name || 'Planter'}</div>
                  <div className="field-avatar-menu-role">
                    {`${planter?.participant_username || 'Planter'} · Participant ${planter?.participant_slot || '—'} of ${planter?.participant_count || 1}`}
                  </div>
                </div>
              </div>
              <button type="button" role="menuitem" className="field-avatar-menu-item"
                onClick={() => { setAvatarMenuOpen(false); setActivityOpen(true); }}>
                Activity Logs
              </button>
              <button
                type="button"
                role="menuitem"
                className="field-avatar-menu-item field-avatar-menu-item-danger"
                onClick={() => {
                  setAvatarMenuOpen(false);
                  handleClearRoute();
                  setUserLocation(null);
                  setLocationAccuracy(null);
                  logout();
                }}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <path d="M9 21H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h4" />
                  <polyline points="16 17 21 12 16 7" />
                  <line x1="21" y1="12" x2="9" y2="12" />
                </svg>
                Sign out
              </button>
            </div>
          )}
        </div>
      </header>

      <FieldActivities key={planter?.organization_id ?? planter?.id} organizationId={planter?.organization_id} />

      {activityOpen && <div className="field-activity-overlay" role="dialog" aria-label="Organization activity">
        <div className="field-activity-head"><strong>Organization activity</strong><button type="button" onClick={() => setActivityOpen(false)}>Close</button></div>
        <ActivityFeed scope="field" />
      </div>}

      {loadError && <div className="field-error field-error-inline" role="alert">{loadError} <button type="button" onClick={reload} disabled={loading}>Retry</button></div>}
      {locationError && <div className="field-error field-error-inline" role="alert">{locationError} <button type="button" onClick={() => setLocationError('')}>Dismiss</button></div>}
      {markAllError && !pendingMarkAllOpen && (
        <div className="field-error field-error-inline">{markAllError}</div>
      )}

      <div className={'field-map-wrap' + (route ? ' field-map-wrap-navigation' : '')}>
        <PointsMap
          points={points}
          projectSites={projectSites}
          selectedId={selectedId}
          onSelect={handleSelect}
          route={navigationRoute}
          userLocation={userLocation}
          locationAccuracy={locationAccuracy}
          focusRequest={focusRequest}
          completionSelectedIds={selectedMarkAllPointIds}
          startPoint={!route && !markAllChooseMode && !selected ? recommendedStart : null}
          onStartNavigation={handleStartNavigation}
          routeBusy={routeBusy}
        />

        <div className="field-map-tools" aria-label="Map controls">
          <button type="button" onClick={() => setFocusRequest({ kind: 'points' })} disabled={loading || (!points.length && !projectSites.length)}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="M8 3H3v5m13-5h5v5M3 16v5h5m13-5v5h-5"/><circle cx="12" cy="12" r="3"/></svg>
            My points
          </button>
          <button type="button" onClick={handleLocate} disabled={locationBusy}>
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><circle cx="12" cy="12" r="7"/><circle cx="12" cy="12" r="2"/><path d="M12 1v4m0 14v4M1 12h4m14 0h4"/></svg>
            {locationBusy ? 'Locating…' : 'My location'}
          </button>
          {userLocation && locationAccuracy != null && !route && <span className="field-gps-accuracy">GPS ±{Math.round(locationAccuracy)} m</span>}
        </div>
        {points.length > 0 && !route && <div className="field-map-legend" aria-label="Point colors">
          <span><i style={{ background: getPlanterColor(planter?.organization_id) }} />{STATUS_LABEL.assigned}</span>
          <span><i style={{ background: '#eab308' }} />Planted</span>
          <span><i style={{ background: '#9ca3af' }} />Skipped</span>
        </div>}

        {route && <FieldNavigationNotice
          key={`navigation-${route.navigationId}`}
          route={navigationRoute}
          onStop={handleClearRoute}
          gpsError={navigationLocationError}
        />}

        {routeError && !selected && (
          <div className="field-route-error" role="alert">{routeError}</div>
        )}

        {!loading && points.length === 0 && !loadError && (
          <div className="field-empty-state" role="status">
            <svg width="36" height="36" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
              <circle cx="12" cy="10" r="3" />
            </svg>
            <p className="field-empty-title">No points assigned yet</p>
            <p className="field-empty-sub">Ask your team admin to assign you some planting points.</p>
          </div>
        )}

      </div>

      <section className={'field-work-panel' + (pointListOpen ? ' field-work-panel-expanded' : '')} aria-label="Your planting work">
        <div className="field-work-heading">
          <div className="field-work-identity">
            <span className="field-work-eyebrow">PARTICIPANT {planter?.participant_slot || '—'} · YOUR ASSIGNMENT</span>
            <h1>{planter?.organization_name || planter?.full_name || 'Your organization'}</h1>
            <span className="field-work-site">{assignedZones.map((zone) => zone.name).join(' · ') || 'Waiting for a project site'}</span>
          </div>
          <button type="button" className="field-refresh" onClick={reload} disabled={loading || markBusy || markAllBusy} aria-label="Refresh assignments" title="Refresh assignments">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="M20 7v5h-5M4 17v-5h5"/><path d="M6 7a7 7 0 0 1 12-1l2 6M4 12l2 6a7 7 0 0 0 12-1"/></svg>
          </button>
        </div>
        <div className="field-progress-copy" role="status">
          <span>{loading ? 'Refreshing your points…' : plantedCount + ' of ' + assignedCount + ' planted'}</span>
          <span>{pending - unavailableCount} to plant{skippedCount > 0 ? ' · ' + skippedCount + ' skipped' : ''}{unavailableCount > 0 ? ' · ' + unavailableCount + ' unavailable' : ''}</span>
        </div>
        <div className="field-progress-track" role="progressbar" aria-label="Planting progress" aria-valuemin={0} aria-valuemax={100} aria-valuenow={progress} aria-valuetext={plantedCount + ' of ' + assignedCount + ' points planted'}>
          <span style={{ width: progress + '%' }} />
        </div>
        <div className="field-work-actions">
          {markAllChooseMode ? <>
            <button type="button" className="field-btn field-btn-primary" onClick={reviewMarkAllSelection} disabled={loading || markAllBusy || !selectedMarkAllCount}>Review {selectedMarkAllCount || ''} {selectedMarkAllCount === 1 ? 'point' : 'points'}</button>
            <button type="button" className="field-btn field-btn-outline" onClick={cancelMarkAllSelection} disabled={markAllBusy}>Cancel selection</button>
          </> : <>
            <button type="button" className="field-btn field-btn-primary" disabled={!nextPoint || loading || markBusy || markAllBusy || routeBusy} onClick={() => { if (nextPoint) void handleStartNavigation(nextPoint, Boolean(route)); }}>
              {routeBusy ? 'Finding route…' : nextPoint ? 'Next point · #' + nextPoint.point_num : route ? 'Last point in route' : assignedCount ? 'No assigned points awaiting planting' : 'Waiting for points'}
            </button>
            <button type="button" className="field-btn field-btn-outline" onClick={handleStartMarkAll} disabled={loading || markBusy || markAllBusy || routeBusy || !markAllCandidateCount}>Choose points to mark</button>
          </>}
        </div>
        <button type="button" className="field-list-toggle" onClick={() => setPointListOpen((value) => !value)} aria-expanded={pointListOpen} aria-controls="field-point-list">
          <span>{markAllChooseMode ? selectedMarkAllCount + ' selected · ' + markAllCandidateCount + ' available' : 'My points (' + assignedCount + ')'}</span>
          <span>{pointListOpen ? 'Hide list' : 'Show list'} <span aria-hidden="true">{pointListOpen ? '⌄' : '⌃'}</span></span>
        </button>
        {pointListOpen && <div id="field-point-list" className="field-point-list-panel">
          {markAllChooseMode ? <div className="field-selection-help">
            <span>Choose only the locations you have planted.</span>
            <button type="button" onClick={() => setSelectedMarkAllPointIds(selectedMarkAllCount === markAllCandidateCount ? [] : markAllCandidatePoints.map((point) => point.assignment_point_id))} disabled={markAllBusy}>
              {selectedMarkAllCount === markAllCandidateCount ? 'Clear selection' : 'Select available'}
            </button>
          </div> : <div className="field-list-filters" aria-label="Filter your points">
            {[
              ['all', 'All', assignedCount], ['pending', STATUS_LABEL.assigned, pending - unavailableCount],
              ['planted', 'Planted', plantedCount], ['skipped', 'Skipped', skippedCount],
              ...(unavailableCount ? [['eroded_unavailable', 'Unavailable', unavailableCount]] : []),
            ].map(([filter, label, count]) => <button type="button" key={filter} aria-pressed={pointFilter === filter} onClick={() => setPointFilter(filter)}>{label} <span>{count}</span></button>)}
          </div>}
          <div className="field-point-list" aria-label={markAllChooseMode ? 'Select planted points' : 'Assigned planting points'}>
            {(markAllChooseMode ? points : visibleListPoints).map((point) => {
              const status = getFieldPointStatus(point);
              const checked = selectedMarkAllPointSet.has(Number(point.assignment_point_id));
              const color = ['pending', 'planned', 'assigned'].includes(status) ? getPlanterColor(planter?.organization_id) : STATUS_COLOR[status];
              return <button type="button" key={point.assignment_point_id} className={'field-point-row' + (checked && markAllChooseMode ? ' field-point-row-selected' : '')}
                onClick={() => handleSelect(point.assignment_point_id)} disabled={markAllChooseMode && (!canMarkPointCompleted(point) || markAllBusy)} aria-pressed={markAllChooseMode ? checked : undefined}>
                <span className="field-point-order" style={{ '--point-color': color }}>{markAllChooseMode ? checked ? '✓' : '○' : point.visit_order || point.sequence_num}</span>
                <span className="field-point-row-info"><strong>Point #{point.point_num}</strong><span>{point.species || 'Planting location'}{point.survival_warning ? ' · Planner warning' : ''}</span></span>
                <span className="field-point-row-status">{COMPLETED_ASSIGNMENT_STATUSES.has(status) ? 'Planted' : STATUS_LABEL[status] || status}</span>
                {!markAllChooseMode && <span aria-hidden="true">›</span>}
              </button>;
            })}
            {!markAllChooseMode && !visibleListPoints.length && <p className="field-list-empty">No points in this filter.</p>}
          </div>
        </div>}
      </section>

      {selected && <PointActionSheet
        key={`point-${selectedId}`}
        point={selected}
        open={Boolean(selected)}
        onClose={handleCloseSheet}
        onNavigate={handleStartNavigation}
        onMark={handleMark}
        busy={markBusy}
        actionError={markError}
        routeBusy={routeBusy}
        routeError={routeError}
      />}

      <Modal
        open={welcomeOpen}
        title={`${welcomeKind === 'register' ? 'Welcome' : 'Welcome back'}${planter?.full_name ? `, ${planter.full_name}` : ''}`}
        variant="success"
        confirmLabel="Let's get started"
        cancelLabel=""
        onConfirm={closeWelcome}
        icon={(
          <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <path d="M11 20A7 7 0 0 1 4 13H1a10 10 0 0 0 10 10v-3z" />
            <path d="M21 4c-4 0-9 1.5-11 6-2 4.5 0 11 0 11s6.5 2 11 0c4.5-2 6-7 6-11C27 5.5 25 4 21 4z" transform="translate(-4, 0)" />
          </svg>
        )}
      >
        <p>
          {welcomeKind === 'register'
            ? `You joined as ${planter?.participant_username}, Participant ${planter?.participant_slot}. Use the same username to return to your assigned points and planting progress.`
            : `Welcome back, ${planter?.participant_username}. Your assigned points and saved planting progress are ready.`}
        </p>
      </Modal>

      <Modal
        open={pendingMarkAllOpen}
        title="Mark selected points as completed?"
        variant="warning"
        confirmLabel="Yes, mark selected completed"
        cancelLabel="Not yet"
        busy={markAllBusy}
        onConfirm={confirmMarkAll}
        onCancel={cancelMarkAll}
      >
        <p>
          Are you sure these
          {' '}{selectedMarkAllCount} selected
          {' '}{selectedMarkAllCount === 1 ? 'point is' : 'points are'} all completed?
          This will mark only the selected
          {' '}{selectedMarkAllCount === 1 ? 'point' : 'points'} as planted.
        </p>
        {markAllError && (
          <p style={{ color: '#dc2626', marginTop: 8 }}>{markAllError}</p>
        )}
      </Modal>

      <Modal
        open={Boolean(pendingMarkPoint)}
        title={`${pendingMarkStatus === 'skipped' ? 'Skip' : 'Mark'} point${pendingMarkPoint?.point_num ? ` #${pendingMarkPoint.point_num}` : ''}${pendingMarkStatus === 'skipped' ? '?' : ' as planted?'}`}
        variant="warning"
        confirmLabel={pendingMarkStatus === 'skipped' ? 'Yes, skip point' : 'Yes, mark as planted'}
        cancelLabel="Not yet"
        busy={markBusy}
        onConfirm={confirmMark}
        onCancel={cancelMark}
      >
        {pendingMarkStatus === 'skipped' ? (
          <>
            <p>
              Confirm that this planting point should be skipped for now.
              The marker will turn gray on your map and the admin's records
              will update accordingly.
            </p>
            <label className="form-label" htmlFor="field-skip-reason">Reason for skipping</label>
            <textarea
              {...skipFeedback.props('skip_reason')}
              id="field-skip-reason"
              className="form-input"
              rows={3}
              required
              maxLength={500}
              value={pendingSkipReason}
              onChange={(event) => { skipFeedback.onChange(event); setPendingSkipReason(event.target.value); }}
              placeholder="For example: deep mud, blocked access, or unsafe tide"
              disabled={markBusy}
            />
            <FieldError feedback={skipFeedback} field="skip_reason" />
          </>
        ) : (
          <p>
            Confirm that you have planted a mangrove seedling at this point.
            The marker will turn yellow on your map and the admin's records
            will update accordingly.
          </p>
        )}
        {markError && (
          <p style={{ color: '#dc2626', marginTop: 8 }}>{markError}</p>
        )}
      </Modal>

      <Modal
        open={Boolean(completedMarkAllResult)}
        title="Selected points recorded"
        variant="success"
        confirmLabel="Continue"
        cancelLabel=""
        onConfirm={closeMarkAllCompletion}
      >
        <p>
          {completedMarkAllResult?.updated_points || 0}
          {' '}{(completedMarkAllResult?.updated_points || 0) === 1 ? 'point has' : 'points have'}
          {' '}been recorded as planted from your selected points. They now show in yellow on your field map.
        </p>
        {completedMarkAllResult?.unavailable_points > 0 && (
          <p>
            {completedMarkAllResult.unavailable_points} unavailable
            {' '}{completedMarkAllResult.unavailable_points === 1 ? 'point was' : 'points were'}
            {' '}left unchanged.
          </p>
        )}
      </Modal>

      <Modal
        open={Boolean(completedMarkPoint)}
        title={completedMarkStatus === 'skipped' ? 'Point skipped' : 'Point planted!'}
        variant="success"
        confirmLabel="Continue"
        cancelLabel=""
        onConfirm={closeCompletionModal}
      >
        {completedMarkStatus === 'skipped' ? (
          <p>
            Point
            {completedMarkPoint?.point_num ? ` #${completedMarkPoint.point_num}` : ''}
            {' '}has been recorded as skipped. It now shows in gray on your field map.
          </p>
        ) : (
        <p>
          Great work — point
          {completedMarkPoint?.point_num ? ` #${completedMarkPoint.point_num}` : ''}
          {' '}has been recorded as planted. It now shows in yellow on your
          field map.
        </p>
        )}
      </Modal>
    </div>
  );
}
