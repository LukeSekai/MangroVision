import { getPlanterColor } from '../utils/planterColors';
import { zigzagPoints } from '../utils/organizationAssignment';
import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import Modal from '../components/Modal';
import Logo from '../components/Logo';
import ActivityFeed from '../components/ActivityFeed';
import { usePlanterAuthStore } from '../stores/planterAuthStore';
import { ORTHOPHOTO_MAX_NATIVE_ZOOM, ORTHOPHOTO_TILE_URL } from '../config/mapTiles';
import { createGoogleSatelliteLayer } from '../config/googleBasemap';
import './FieldApp.css';

const API_BASE = import.meta.env.VITE_API_BASE || '';

delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

const STATUS_LABEL = {
  planned: 'Planned',
  assigned: 'Assigned',
  planted: 'Planted',
  completed: 'Completed',
  skipped: 'Skipped',
  eroded_unavailable: 'Not Available for Planting',
};

// Field-side palette. The planter assignment lifecycle on the database side
// uses 'pending' (assigned to me, not yet planted) and 'completed' (I have
// planted it). Both 'planned' and 'assigned' map to blue so any not-yet-done
// state is visually identical on the planter map. 'completed' / 'planted'
// is the freshly-planted state and renders YELLOW so it stands clearly
// apart from blue assigned points.
const STATUS_COLOR = {
  planned: '#2563eb',    // blue — still pending
  assigned: '#2563eb',   // blue — still pending
  pending: '#2563eb',    // blue — DB enum for "still pending"
  planted: '#eab308',    // yellow — planter has marked this complete
  completed: '#eab308',  // yellow — DB enum for "planted"
  skipped: '#9ca3af',
  eroded_unavailable: '#f97316',
};

const COMPLETED_ASSIGNMENT_STATUSES = new Set(['planted', 'completed']);
const FINAL_ASSIGNMENT_STATUSES = new Set(['planted', 'completed', 'skipped']);

function canMarkPointCompleted(point) {
  return Boolean(point)
    && !point.eroded_unavailable
    && !COMPLETED_ASSIGNMENT_STATUSES.has(point.assignment_status);
}

function canNavigateNextPoint(point) {
  return !FINAL_ASSIGNMENT_STATUSES.has(point.assignment_status)
    && !point.eroded_unavailable && !point.inside_eroded_zone;
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
  if (zoom >= 22) return 3.5;
  if (zoom >= 21) return 2.4;
  if (zoom >= 20) return 1.6;
  if (zoom >= 19) return 1.05;
  if (zoom >= 18) return 0.8;
  if (zoom >= 17) return 0.65;
  return 0.55;
};

const getFieldPointWeight = (zoom) => {
  if (zoom >= 22) return 1.1;
  if (zoom >= 21) return 0.75;
  if (zoom >= 20) return 0.45;
  if (zoom >= 19) return 0.2;
  return 0;
};

function getCurrentLocation(options = {}) {
  return new Promise((resolve, reject) => {
    if (!('geolocation' in navigator)) {
      reject(new Error('Location is not supported on this device.'));
      return;
    }
    navigator.geolocation.getCurrentPosition(
      (pos) => resolve([pos.coords.latitude, pos.coords.longitude]),
      (err) => reject(new Error(err.message || 'Could not get your location.')),
      { enableHighAccuracy: true, timeout: 15000, maximumAge: 10000, ...options },
    );
  });
}

async function fetchRoute(origin, dest, travelMode = 'walking') {
  const response = await fetch(`${API_BASE}/api/routing/compute`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({
      origin_lat: origin[0],
      origin_lon: origin[1],
      dest_lat: dest[0],
      dest_lon: dest[1],
      travel_mode: travelMode,
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

function AuthScreen() {
  const register = usePlanterAuthStore((s) => s.register);
  const login = usePlanterAuthStore((s) => s.login);
  const storeStatus = usePlanterAuthStore((s) => s.status);
  const storeError = usePlanterAuthStore((s) => s.error);

  const [mode, setMode] = useState('login');
  const [localError, setLocalError] = useState('');
  const [organizations, setOrganizations] = useState([]);
  const [organizationsLoading, setOrganizationsLoading] = useState(true);
  const [form, setForm] = useState({
    participant_count: 1,
    participant_slot: '',
    recover_slot: false,
    organization_id: '',
    username: '',
    password: '',
    phone: '',
    base_label: '',
  });

  const update = (field) => (event) => setForm((f) => ({ ...f, [field]: event.target.value }));
  const submitting = storeStatus === 'loading';

  useEffect(() => {
    let active = true;
    fetch(`${API_BASE}/api/planter-auth/organizations`)
      .then(async (response) => {
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.detail || 'Could not load organizations.');
        if (active) setOrganizations(Array.isArray(payload.organizations) ? payload.organizations : []);
      })
      .catch((error) => {
        if (active) setLocalError(error.message || 'Could not load organizations.');
      })
      .finally(() => {
        if (active) setOrganizationsLoading(false);
      });
    return () => { active = false; };
  }, []);

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLocalError('');
    try {
      if (mode === 'login') {
        await login(form.username.trim(), form.password, Number(form.participant_slot) || null, form.recover_slot);
      } else {
        if (!form.organization_id || !form.username.trim() || !form.password) {
          throw new Error('Organization, username, and password are required.');
        }
        await register({
          full_name: organizations.find((organization) => Number(organization.id) === Number(form.organization_id))?.name || 'Organization',
          participant_count: Number(form.participant_count),
          username: form.username.trim(),
          password: form.password,
          organization_id: Number(form.organization_id),
          phone: form.phone.trim(),
          base_label: form.base_label.trim(),
        });
      }
    } catch (error) {
      setLocalError(error.message || 'Something went wrong');
    }
  };

  return (
    <div className="field-auth-screen">
      <div className="field-auth-card">
        <div className="field-auth-brand">
          <Logo variant="lockup" size={44} alt="MangroVision" />
          <span className="field-auth-brand-suffix">Field</span>
        </div>
        <p className="field-auth-subtitle">
          Organization {mode === 'login' ? 'sign in' : 'registration'}
        </p>

        <div className="field-tab-row">
          <button
            type="button"
            className={`field-tab ${mode === 'login' ? 'field-tab-active' : ''}`}
            onClick={() => { setMode('login'); setLocalError(''); }}
          >
            Sign In
          </button>
          <button
            type="button"
            className={`field-tab ${mode === 'register' ? 'field-tab-active' : ''}`}
            onClick={() => { setMode('register'); setLocalError(''); }}
          >
            Register
          </button>
        </div>

        <form className="field-form" onSubmit={handleSubmit}>
          {mode === 'register' && (
            <>
              <label className="field-label">
                Organization
                <select
                  className="field-input"
                  value={form.organization_id}
                  onChange={update('organization_id')}
                  disabled={organizationsLoading || organizations.length === 0}
                  required
                >
                  <option value="">
                    {organizationsLoading
                      ? 'Loading organizations...'
                      : organizations.length
                        ? 'Select your organization'
                        : 'No organizations available'}
                  </option>
                  {organizations.map((organization) => (
                    <option key={organization.id} value={organization.id}>{organization.name}</option>
                  ))}
                </select>
                {!organizationsLoading && organizations.length === 0 && (
                  <small className="field-label-help">Ask the LGU to create the organization through Scheduling first.</small>
                )}
              </label>
            </>
          )}

          {mode === 'login' && (
            <>
              <small className="field-label-help">Use your organization's shared username and password. Each new device automatically receives its own participant slot, up to the organization's participant count.</small>
              <label className="field-label">
                <span><input type="checkbox" checked={form.recover_slot} onChange={(event) => setForm((current) => ({ ...current, recover_slot: event.target.checked }))} /> I am replacing a device after an LGU reset</span>
              </label>
              {form.recover_slot && (
                <label className="field-label">
                  Participant number reset by the LGU
                  <input className="field-input" type="number" min="1" max="10000" required value={form.participant_slot} onChange={update('participant_slot')} />
                  <small className="field-label-help">Only use this to recover your previous points on a replacement phone.</small>
                </label>
              )}
            </>
          )}
          <label className="field-label">
            Shared username
            <input
              className="field-input"
              type="text"
              value={form.username}
              onChange={update('username')}
              autoComplete="username"
              required
            />
          </label>

          <label className="field-label">
            Password
            <input
              className="field-input"
              type="password"
              value={form.password}
              onChange={update('password')}
              autoComplete={mode === 'login' ? 'current-password' : 'new-password'}
              required
            />
          </label>

          {mode === 'register' && (
            <>
              <label className="field-label">
                Number of participants / planters
                <input className="field-input" type="number" min="1" max="10000" step="1" required value={form.participant_count} onChange={update('participant_count')} />
                <small className="field-label-help">Register once for your organization. Each device uses this shared login and receives its own share of the assigned points.</small>
              </label>
              <label className="field-label">
                Phone (optional)
                <input
                  className="field-input"
                  type="tel"
                  value={form.phone}
                  onChange={update('phone')}
                  autoComplete="tel"
                />
              </label>
              <label className="field-label">
                Home base label (optional)
                <input
                  className="field-input"
                  type="text"
                  value={form.base_label}
                  onChange={update('base_label')}
                  placeholder="e.g. Leganes barangay hall"
                />
              </label>
            </>
          )}

          {(localError || storeError) && (
            <div className="field-error">{localError || storeError}</div>
          )}

          <button
            className="field-submit"
            type="submit"
            disabled={submitting || (mode === 'register' && (organizationsLoading || organizations.length === 0))}
          >
            {submitting ? 'Please wait...' : mode === 'login' ? 'Sign In' : 'Create Organization Account'}
          </button>
        </form>
      </div>
    </div>
  );
}

function PointsMap({
  points,
  projectSites,
  selectedId,
  onSelect,
  route,
  userLocation,
  completionSelectedIds = [],
}) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const layerRef = useRef(null);
  const zoneLayerRef = useRef(null);
  const markersRef = useRef(new Map());
  const routeLayerRef = useRef(null);
  const userMarkerRef = useRef(null);
  const hasFitBoundsRef = useRef(false);

  useEffect(() => {
    if (mapRef.current || !containerRef.current) return;
    const map = L.map(containerRef.current, {
      center: [10.78, 122.6253],
      zoom: 17,
      maxZoom: 24,
      zoomControl: true,
      attributionControl: true,
    });

    map.attributionControl.setPrefix(false);
    map.createPane('orthophotoPane');
    map.getPane('orthophotoPane').style.zIndex = 250;

    createGoogleSatelliteLayer({ maxZoom: 24 }).addTo(map);
    L.tileLayer(ORTHOPHOTO_TILE_URL, {
      pane: 'orthophotoPane',
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

    map.on('zoomend', () => {
      const zoom = map.getZoom();
      const radius = getFieldPointRadius(zoom);
      const weight = getFieldPointWeight(zoom);
      markersRef.current.forEach((marker) => {
        const hasWarning = Boolean(marker.options.mgWarning);
        marker.setRadius(radius);
        marker.setStyle({ weight: hasWarning ? Math.max(weight, 1.2) : weight });
      });
    });

    return () => {
      map.remove();
      mapRef.current = null;
      layerRef.current = null;
      zoneLayerRef.current = null;
      markersRef.current.clear();
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
    projectSites.forEach((feature) => {
      const properties = feature?.properties || {};
      const siteId = feature?.id ?? properties.id;
      const geometry = parseProjectSiteGeometry(feature?.geometry);
      if (siteId == null || !geometry) return;
      const boundary = L.geoJSON({ type: 'Feature', properties: {}, geometry }, {
        style: {
          color: '#22c55e',
          weight: 2.5,
          fillColor: '#16a34a',
          fillOpacity: 0.12,
          dashArray: '7 5',
        },
      }).addTo(zoneLayer);
      boundary.bindTooltip(
        `${properties.name || 'Organization zone'}${properties.organization_name ? ` · ${properties.organization_name}` : ''}`,
        { sticky: true },
      );
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
      markersRef.current.set(point.assignment_point_id, marker);
    });

    if (!hasFitBoundsRef.current && validPoints.length > 0) {
      if (validPoints.length === 1) {
        // Single point: center on it at a comfortable zoom that stays
        // inside the orthophoto's native resolution (so it doesn't go
        // pixelated on first load).
        const only = validPoints[0];
        map.setView([only.latitude, only.longitude], 19, { animate: false });
      } else {
        const bounds = L.latLngBounds(validPoints.map((p) => [p.latitude, p.longitude]));
        map.fitBounds(bounds, { padding: [40, 40], maxZoom: 19 });
      }
      hasFitBoundsRef.current = true;
    } else if (!hasFitBoundsRef.current && siteBoundaries.length > 0) {
      const bounds = L.featureGroup(siteBoundaries).getBounds();
      if (bounds.isValid()) {
        map.fitBounds(bounds, { padding: [35, 35], maxZoom: 19 });
        hasFitBoundsRef.current = true;
      }
    }
  }, [points, projectSites, onSelect, completionSelectedIds]);

  useEffect(() => {
    markersRef.current.forEach((marker, id) => {
      const map = mapRef.current;
      const zoom = map?.getZoom() ?? 17;
      const isCompletionSelected = Boolean(marker.options.mgCompletionSelected);
      marker.setStyle({
        weight: id === selectedId || isCompletionSelected ? Math.max(1.2, getFieldPointWeight(zoom) + 0.8) : getFieldPointWeight(zoom),
        color: id === selectedId ? '#dc2626' : (isCompletionSelected ? '#4c1d95' : '#0f172a'),
      });
      marker.setRadius(id === selectedId || isCompletionSelected ? getFieldPointRadius(zoom) + 0.7 : getFieldPointRadius(zoom));
    });
    const selected = points.find((p) => p.assignment_point_id === selectedId);
    const map = mapRef.current;
    if (selected && map) {
      map.panTo([selected.latitude, selected.longitude]);
    }
  }, [selectedId, points]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (routeLayerRef.current) {
      routeLayerRef.current.remove();
      routeLayerRef.current = null;
    }

    if (route && (route.polyline?.length >= 2 || route.target)) {
      const routeGroup = L.layerGroup().addTo(map);
      // Main routed path — royal blue for high-contrast visibility against
      // the satellite/orthophoto basemap. Solid line.
      const line = L.polyline(route.polyline || [], {
        color: '#4169e1',
        weight: 4,
        opacity: 0.95,
        lineCap: 'round',
        lineJoin: 'round',
      }).addTo(routeGroup);
      let bounds = line.getBounds();

      if (route.entrance && route.route_source !== 'within_site') {
        L.marker(route.entrance).bindTooltip('White-road entrance', { permanent: true })
          .addTo(routeGroup);
        bounds.extend(route.entrance);
      }
      if (route.target) {
        L.circleMarker(route.target, { radius: 8, color: '#f97316', fillOpacity: 0.2 })
          .bindTooltip('Selected planting point').addTo(routeGroup);
        bounds.extend(route.target);
      }

      routeLayerRef.current = routeGroup;
      if (bounds.isValid()) map.fitBounds(bounds, { padding: [60, 60], maxZoom: 20 });
    }
  }, [route]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;

    if (userMarkerRef.current) {
      userMarkerRef.current.remove();
      userMarkerRef.current = null;
    }

    if (userLocation) {
      const marker = L.circleMarker(userLocation, {
        radius: 7,
        color: '#ffffff',
        weight: 2,
        fillColor: '#2563eb',
        fillOpacity: 1,
      }).addTo(map);
      userMarkerRef.current = marker;
    }
  }, [userLocation]);

  return <div ref={containerRef} className="field-map" />;
}

function PointActionSheet({ point, open, onClose, onNavigate, onMark, busy, actionError, routeBusy }) {
  const [view, setView] = useState('actions');

  useEffect(() => {
    if (open) setView('actions');
  }, [open, point?.assignment_point_id]);

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
  const badgeLabel = isUnavailable ? 'Not Available' : isSkipped ? 'Skipped' : isPlanted ? 'Planted' : 'Pending';

  return (
    <>
      <div
        className={`field-overlay-backdrop ${open ? 'field-overlay-backdrop-open' : ''}`}
        onClick={onClose}
        aria-hidden={!open}
      />
      <div
        className={`field-overlay ${open ? 'field-overlay-open' : ''}`}
        role="dialog"
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
              <button
                type="button"
                className="field-btn field-btn-secondary"
                onClick={() => setView('details')}
              >
                View Details
              </button>
              <button
                type="button"
                className="field-btn field-btn-secondary"
                onClick={() => onNavigate(point)}
                disabled={routeBusy || isUnavailable}
              >
                {routeBusy ? 'Routing…' : 'Show In-App Route'}
              </button>
              <button
                type="button"
                className="field-btn field-btn-primary"
                onClick={() => onMark(point, 'completed')}
                disabled={busy || isFinal || isUnavailable}
              >
                {isPlanted ? 'Already Planted' : busy ? 'Saving…' : 'Mark as Planted'}
              </button>
              <button
                type="button"
                className="field-btn field-btn-skip"
                onClick={() => onMark(point, 'skipped')}
                disabled={busy || isFinal || isUnavailable}
              >
                {isSkipped ? 'Skipped' : busy ? 'Saving...' : 'Skip Point'}
              </button>
            </div>
            {isUnavailable && (
              <div className="field-warning">
                This point is inside an eroded zone and is currently Not Available for Planting. It will return to Planned when the LGU/Admin removes the zone.
              </div>
            )}
            {hasPlannerWarning && (
              <div className="field-warning">
                Planner warning ({warningSeverity}): {warningSummary}
              </div>
            )}
            {actionError && <div className="field-error">{actionError}</div>}
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
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState('');
  const [selectedId, setSelectedId] = useState(null);
  const [markBusy, setMarkBusy] = useState(false);
  const [markError, setMarkError] = useState('');
  const [hintDismissed, setHintDismissed] = useState(false);
  const [route, setRoute] = useState(null);
  const [userLocation, setUserLocation] = useState(null);
  const [routeBusy, setRouteBusy] = useState(false);
  const [routeError, setRouteError] = useState('');
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
  const [fabMenuOpen, setFabMenuOpen] = useState(false);
  const avatarMenuRef = useRef(null);
  const fabMenuRef = useRef(null);

  useEffect(() => {
    if (!avatarMenuOpen && !fabMenuOpen) return undefined;
    const handlePointerDown = (event) => {
      if (avatarMenuOpen && avatarMenuRef.current && !avatarMenuRef.current.contains(event.target)) {
        setAvatarMenuOpen(false);
      }
      if (fabMenuOpen && fabMenuRef.current && !fabMenuRef.current.contains(event.target)) {
        setFabMenuOpen(false);
      }
    };
    document.addEventListener('mousedown', handlePointerDown);
    document.addEventListener('touchstart', handlePointerDown);
    return () => {
      document.removeEventListener('mousedown', handlePointerDown);
      document.removeEventListener('touchstart', handlePointerDown);
    };
  }, [avatarMenuOpen, fabMenuOpen]);

  const handleSelect = useCallback((id) => {
    setHintDismissed(true);
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
    setSelectedId(id);
  }, [markAllChooseMode, points]);

  const handleCloseSheet = useCallback(() => {
    setSelectedId(null);
    setMarkError('');
  }, []);

  const planterBaseLat = planter?.base_lat;
  const planterBaseLon = planter?.base_lon;

  const handleStartNavigation = useCallback(async (point) => {
    setRouteError('');
    if (point?.eroded_unavailable || point?.inside_eroded_zone) {
      setRouteError('This point is Not Available for Planting while it remains inside an eroded zone.');
      return;
    }
    const latitude = Number(point?.latitude);
    const longitude = Number(point?.longitude);
    if (!Number.isFinite(latitude) || !Number.isFinite(longitude)) {
      setRouteError('This planting point has no valid map location.');
      return;
    }
    setRouteBusy(true);
    try {
      let origin;
      try {
        origin = await getCurrentLocation();
      } catch (locationError) {
        const baseLat = Number(planterBaseLat);
        const baseLon = Number(planterBaseLon);
        const hasConfiguredBase = planterBaseLat !== null && planterBaseLat !== undefined
          && planterBaseLat !== '' && planterBaseLon !== null
          && planterBaseLon !== undefined && planterBaseLon !== '';
        if (hasConfiguredBase && Number.isFinite(baseLat) && Number.isFinite(baseLon)) {
          origin = [baseLat, baseLon];
        } else {
          throw new Error(
            `${locationError.message || 'Could not get your location.'} Allow location access or ask the LGU to configure your base location.`,
          );
        }
      }
      setUserLocation(origin);
      const data = await fetchRoute(origin, [latitude, longitude], 'walking');
      setRoute(data);
      setSelectedId(null);
    } catch (error) {
      // Navigation remains inside MangroVision. Never hand planter location
      // data to a third-party navigation page as an implicit fallback.
      setRoute(null);
      setRouteError(error.message || 'Could not display the in-app route.');
    } finally {
      setRouteBusy(false);
    }
  }, [planterBaseLat, planterBaseLon]);

  const handleClearRoute = useCallback(() => {
    setRoute(null);
    setRouteError('');
  }, []);

  useEffect(() => {
    hydrateSession();
  }, [hydrateSession]);

  const reload = useCallback(async () => {
    if (!isAuthenticated) return;
    setLoading(true);
    setLoadError('');
    try {
      const workspace = await fetchFieldPoints();
      setPoints(zigzagPoints(workspace.points).map((point, index) => ({ ...point, visit_order: index + 1 })));
      setProjectSites(workspace.projectSites);
    } catch (error) {
      setLoadError(error.message || 'Could not load points');
    } finally {
      setLoading(false);
    }
  }, [isAuthenticated, fetchFieldPoints]);

  useEffect(() => {
    reload();
  }, [reload]);

  useEffect(() => {
    if (isAuthenticated && sessionStorage.getItem('mv_field_show_welcome') === '1') {
      // Use the kind flag set by the auth store to pick "Welcome" vs
      // "Welcome back". Default to 'login' if the flag is missing so an
      // unknown state still produces the safer "Welcome back" greeting.
      const kind = sessionStorage.getItem('mv_field_welcome_kind') || 'login';
      setWelcomeKind(kind === 'register' ? 'register' : 'login');
      setWelcomeOpen(true);
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
    setMarkError('');
    if (point?.eroded_unavailable || point?.inside_eroded_zone) {
      setMarkError('This point is Not Available for Planting while it remains inside an eroded zone.');
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
      setMarkError('Briefly explain why this point is being skipped.');
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
      setRoute(null);
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
      setMarkAllError('There are no available pending or skipped points to mark completed.');
      return;
    }
    setRoute(null);
    setMarkAllChooseMode(true);
    setSelectedMarkAllPointIds([]);
    setHintDismissed(true);
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
                    {`Participant ${planter?.participant_slot || '—'} of ${planter?.participant_count || 1}`}
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

      {activityOpen && <div className="field-activity-overlay" role="dialog" aria-label="Organization activity">
        <div className="field-activity-head"><strong>Organization activity</strong><button type="button" onClick={() => setActivityOpen(false)}>Close</button></div>
        <ActivityFeed scope="field" />
      </div>}

      {loadError && <div className="field-error field-error-inline">{loadError}</div>}
      {markAllError && !pendingMarkAllOpen && (
        <div className="field-error field-error-inline">{markAllError}</div>
      )}

      <div className="field-map-wrap">
        <PointsMap
          points={points}
          projectSites={projectSites}
          selectedId={selectedId}
          onSelect={handleSelect}
          route={route}
          userLocation={userLocation}
          completionSelectedIds={selectedMarkAllPointIds}
        />

        {assignedZones.length > 0 && (
          <div className="field-zone-summary" role="status">
            <strong>{planter?.organization_name || 'Your organization'}</strong>
            <span>Participant {planter?.participant_slot} · {points.length} assigned points</span>
            <span>{assignedZones.map((zone) => zone.name).join(' · ')}</span>
            <button type="button" className="field-btn field-btn-secondary"
              disabled={!points.some(canNavigateNextPoint)}
              onClick={() => {
                const nextPoint = points.find(canNavigateNextPoint);
                if (nextPoint) setSelectedId(nextPoint.assignment_point_id);
              }}>
              Next point
            </button>
          </div>
        )}

        {route && (
          <div className="field-route-banner" role="status">
            <div className="field-route-banner-info">
              <span className="field-route-banner-label">
                {route.route_source === 'site_entrance' ? 'Walking route to entrance'
                  : route.route_source === 'within_site' ? 'Within planting site' : 'Walking route'}
              </span>
              <span className="field-route-banner-stats">
                {route.distance_label} · {route.duration_label}
              </span>
              {route.navigation_note && <span className="field-route-banner-warning">{route.navigation_note}</span>}
            </div>
            <button
              type="button"
              className="field-btn field-btn-ghost field-route-banner-clear"
              onClick={handleClearRoute}
            >
              Clear
            </button>
          </div>
        )}

        {routeError && (
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

        {markAllChooseMode && (
          <div className="field-hint-pill field-hint-pill-action" role="status">
            {selectedMarkAllCount > 0
              ? `${selectedMarkAllCount} selected - tap Review when done`
              : 'Choose completed points'}
          </div>
        )}

        {!markAllChooseMode && !hintDismissed && points.length > 0 && !route && (
          <div className="field-hint-pill" role="status">
            Tap a marker to see options
          </div>
        )}

      </div>

      {!markAllChooseMode && points.length > 0 && (
        <div className="field-fab-wrap" ref={fabMenuRef}>
          {fabMenuOpen && (
            <div className="field-fab-menu" role="menu">
              <button
                type="button"
                role="menuitem"
                className="field-fab-menu-item"
                onClick={() => {
                  setFabMenuOpen(false);
                  handleStartMarkAll();
                }}
                disabled={loading || markAllBusy || markBusy}
              >
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                  <polyline points="20 6 9 17 4 12" />
                </svg>
                Mark All as Completed
              </button>
            </div>
          )}
          <button
            type="button"
            className="field-fab"
            onClick={() => setFabMenuOpen((value) => !value)}
            disabled={loading || markAllBusy || markBusy}
            aria-haspopup="menu"
            aria-expanded={fabMenuOpen}
            aria-label="Mark points"
            title="Mark points"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.6" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <polyline points="20 6 9 17 4 12" />
            </svg>
            <span className="field-fab-label">Mark Points</span>
          </button>
        </div>
      )}

      {markAllChooseMode && (
        <div className="field-fab-actions">
          <button
            type="button"
            className="field-btn field-btn-mark-all"
            onClick={reviewMarkAllSelection}
            disabled={loading || markAllBusy || markBusy}
          >
            <svg className="field-btn-mark-all-icon" width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <polyline points="20 6 9 17 4 12" />
            </svg>
            <span>
              {selectedMarkAllCount > 0
                ? `Review ${selectedMarkAllCount} Point${selectedMarkAllCount === 1 ? '' : 's'}`
                : 'Choose points'}
            </span>
          </button>
          <button
            type="button"
            className="field-btn field-btn-ghost field-fab-cancel"
            onClick={cancelMarkAllSelection}
            disabled={markAllBusy}
          >
            Cancel
          </button>
        </div>
      )}

      <section className="field-bottom-bar" aria-label="Planting summary">
        <div className="field-bottom-greeting">
          <span className="field-bottom-greeting-hi">Hi,</span>
          <span className="field-bottom-greeting-name">{planter?.full_name || 'planter'}</span>
        </div>
        <div className="field-bottom-stats" role="group" aria-label="Assignment summary">
          <div className="field-bottom-stat field-bottom-stat-total">
            <span className="field-bottom-stat-value">{assignedCount}</span>
            <span className="field-bottom-stat-label">{assignedCount === 1 ? 'point' : 'points'} assigned</span>
          </div>
          <div className="field-bottom-stat field-bottom-stat-pending">
            <span className="field-bottom-stat-dot" aria-hidden="true" />
            <span className="field-bottom-stat-value">{pending}</span>
            <span className="field-bottom-stat-label">pending</span>
          </div>
          <div className="field-bottom-stat field-bottom-stat-planted">
            <span className="field-bottom-stat-dot" aria-hidden="true" />
            <span className="field-bottom-stat-value">{plantedCount}</span>
            <span className="field-bottom-stat-label">planted</span>
          </div>
          <div className="field-bottom-stat field-bottom-stat-skipped">
            <span className="field-bottom-stat-dot" aria-hidden="true" />
            <span className="field-bottom-stat-value">{skippedCount}</span>
            <span className="field-bottom-stat-label">skipped</span>
          </div>
        </div>
      </section>

      <PointActionSheet
        point={selected}
        open={Boolean(selected)}
        onClose={handleCloseSheet}
        onNavigate={handleStartNavigation}
        onMark={handleMark}
        busy={markBusy}
        actionError={markError}
        routeBusy={routeBusy}
      />

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
            ? 'Your planter account is ready. Your assigned planting points will appear on the map below.'
            : 'Your field map is ready. Tap any marker to see what to do next.'}
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
              id="field-skip-reason"
              className="form-input"
              rows={3}
              maxLength={500}
              value={pendingSkipReason}
              onChange={(event) => setPendingSkipReason(event.target.value)}
              placeholder="For example: deep mud, blocked access, or unsafe tide"
              disabled={markBusy}
            />
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
        title="All assigned points completed"
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
