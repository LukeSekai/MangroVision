import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { usePlanterAuthStore } from '../stores/planterAuthStore';
import { ORTHOPHOTO_MAX_NATIVE_ZOOM, ORTHOPHOTO_TILE_URL } from '../config/mapTiles';
import './FieldApp.css';

const API_BASE = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

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
};

const STATUS_COLOR = {
  planned: '#2563eb',
  assigned: '#2563eb',
  planted: '#d97706',
  completed: '#059669',
  skipped: '#9ca3af',
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
    } catch (_) { /* ignore parse errors */ }
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
  const [form, setForm] = useState({
    full_name: '',
    username: '',
    password: '',
    phone: '',
    base_label: '',
  });

  const update = (field) => (event) => setForm((f) => ({ ...f, [field]: event.target.value }));
  const submitting = storeStatus === 'loading';

  const handleSubmit = async (event) => {
    event.preventDefault();
    setLocalError('');
    try {
      if (mode === 'login') {
        await login(form.username.trim(), form.password);
      } else {
        if (!form.full_name.trim() || !form.username.trim() || !form.password) {
          throw new Error('Name, username, and password are required.');
        }
        await register({
          full_name: form.full_name.trim(),
          username: form.username.trim(),
          password: form.password,
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
        <h1 className="field-auth-title">MangroVision Field</h1>
        <p className="field-auth-subtitle">
          Planter {mode === 'login' ? 'sign in' : 'registration'}
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
            <label className="field-label">
              Full name
              <input
                className="field-input"
                type="text"
                value={form.full_name}
                onChange={update('full_name')}
                autoComplete="name"
                required
              />
            </label>
          )}

          <label className="field-label">
            Username
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

          <button className="field-submit" type="submit" disabled={submitting}>
            {submitting ? 'Please wait...' : mode === 'login' ? 'Sign In' : 'Create Account'}
          </button>
        </form>
      </div>
    </div>
  );
}

function PointsMap({ points, selectedId, onSelect, route, userLocation }) {
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const layerRef = useRef(null);
  const markersRef = useRef(new Map());
  const routeLayerRef = useRef(null);
  const userMarkerRef = useRef(null);

  useEffect(() => {
    if (mapRef.current || !containerRef.current) return;
    const map = L.map(containerRef.current, {
      center: [10.78, 122.6253],
      zoom: 17,
      maxZoom: 24,
      zoomControl: true,
      attributionControl: false,
    });
    L.tileLayer(
      'https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
      {
        maxZoom: 22,
        maxNativeZoom: 19,
        attribution: 'Tiles © Esri',
      },
    ).addTo(map);
    L.tileLayer(ORTHOPHOTO_TILE_URL, {
      minZoom: 14,
      maxZoom: 24,
      maxNativeZoom: ORTHOPHOTO_MAX_NATIVE_ZOOM,
      tms: false,
      errorTileUrl: 'data:image/gif;base64,R0lGODlhAQABAIAAAAAAAP///yH5BAEAAAAALAAAAAABAAEAAAIBRAA7',
      opacity: 1,
    }).addTo(map);
    mapRef.current = map;
    layerRef.current = L.layerGroup().addTo(map);

    return () => {
      map.remove();
      mapRef.current = null;
      layerRef.current = null;
      markersRef.current.clear();
    };
  }, []);

  useEffect(() => {
    const map = mapRef.current;
    const layer = layerRef.current;
    if (!map || !layer) return;

    layer.clearLayers();
    markersRef.current.clear();

    const validPoints = points.filter(
      (p) => typeof p.latitude === 'number' && typeof p.longitude === 'number',
    );

    validPoints.forEach((point) => {
      const color = STATUS_COLOR[point.assignment_status] || '#2563eb';
      const marker = L.circleMarker([point.latitude, point.longitude], {
        radius: 9,
        color: '#0f172a',
        weight: 2,
        fillColor: color,
        fillOpacity: 0.9,
      });
      marker.on('click', () => onSelect(point.assignment_point_id));
      marker.addTo(layer);
      markersRef.current.set(point.assignment_point_id, marker);
    });

    if (validPoints.length > 0) {
      const bounds = L.latLngBounds(validPoints.map((p) => [p.latitude, p.longitude]));
      map.fitBounds(bounds, { padding: [40, 40], maxZoom: 19 });
    }
  }, [points, onSelect]);

  useEffect(() => {
    markersRef.current.forEach((marker, id) => {
      marker.setStyle({
        weight: id === selectedId ? 4 : 2,
        color: id === selectedId ? '#dc2626' : '#0f172a',
      });
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

    if (route?.polyline?.length >= 2) {
      const latlngs = route.polyline;
      const line = L.polyline(latlngs, {
        color: '#0f766e',
        weight: 5,
        opacity: 0.9,
        lineCap: 'round',
        lineJoin: 'round',
      }).addTo(map);
      routeLayerRef.current = line;
      map.fitBounds(line.getBounds(), { padding: [60, 60], maxZoom: 19 });
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
              <div className="field-overlay-title">Point #{point.point_num}</div>
              <span className={`field-badge field-badge-${isPlanted ? 'planted' : 'pending'}`}>
                {isPlanted ? 'Planted' : 'Pending'}
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
                disabled={routeBusy}
              >
                {routeBusy ? 'Routing…' : 'Open Navigation'}
              </button>
              <button
                type="button"
                className="field-btn field-btn-primary"
                onClick={() => onMark(point, 'completed')}
                disabled={busy || isPlanted}
              >
                {isPlanted ? 'Already Planted' : busy ? 'Saving…' : 'Mark as Planted'}
              </button>
            </div>
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
            <div className="field-overlay-title">Point #{point.point_num}</div>
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
                <span>{STATUS_LABEL[status] || status}</span>
              </div>
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

  const [points, setPoints] = useState([]);
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

  const handleSelect = useCallback((id) => {
    setSelectedId(id);
    setHintDismissed(true);
  }, []);

  const handleCloseSheet = useCallback(() => {
    setSelectedId(null);
    setMarkError('');
  }, []);

  const handleStartNavigation = useCallback(async (point) => {
    setRouteError('');
    setRouteBusy(true);
    try {
      const origin = await getCurrentLocation();
      setUserLocation(origin);
      const data = await fetchRoute(origin, [point.latitude, point.longitude], 'walking');
      setRoute(data);
      setSelectedId(null);
    } catch (error) {
      setRouteError(error.message || 'Could not start navigation.');
    } finally {
      setRouteBusy(false);
    }
  }, []);

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
      const list = await fetchFieldPoints();
      setPoints(list);
    } catch (error) {
      setLoadError(error.message || 'Could not load points');
    } finally {
      setLoading(false);
    }
  }, [isAuthenticated, fetchFieldPoints]);

  useEffect(() => {
    reload();
  }, [reload]);

  const selected = useMemo(
    () => points.find((p) => p.assignment_point_id === selectedId) || null,
    [points, selectedId],
  );

  const handleMark = async (point, status) => {
    setMarkBusy(true);
    setMarkError('');
    try {
      await markPointStatus(point.assignment_point_id, status);
      await reload();
    } catch (error) {
      setMarkError(error.message || 'Could not update point status');
    } finally {
      setMarkBusy(false);
    }
  };

  if (!isAuthenticated) {
    return <AuthScreen />;
  }

  const pending = points.filter((p) => p.assignment_status !== 'planted' && p.assignment_status !== 'completed').length;
  const plantedCount = points.filter((p) => p.assignment_status === 'planted' || p.assignment_status === 'completed').length;

  return (
    <div className="field-shell">
      <header className="field-header">
        <div>
          <div className="field-greeting">Hi, {planter?.full_name || 'planter'}</div>
          <div className="field-counts">
            {pending} pending · {plantedCount} planted
          </div>
        </div>
        <div className="field-header-actions">
          <button className="field-btn field-btn-ghost" onClick={reload} disabled={loading}>
            {loading ? 'Refreshing...' : 'Refresh'}
          </button>
          <button className="field-btn field-btn-ghost" onClick={logout}>
            Sign out
          </button>
        </div>
      </header>

      {loadError && <div className="field-error field-error-inline">{loadError}</div>}

      <div className="field-map-wrap">
        <PointsMap
          points={points}
          selectedId={selectedId}
          onSelect={handleSelect}
          route={route}
          userLocation={userLocation}
        />

        {route && (
          <div className="field-route-banner" role="status">
            <div className="field-route-banner-info">
              <span className="field-route-banner-label">Walking route</span>
              <span className="field-route-banner-stats">
                {route.distance_label} · {route.duration_label}
              </span>
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

        {!hintDismissed && points.length > 0 && !route && (
          <div className="field-hint-pill" role="status">
            Tap a marker to see options
          </div>
        )}
      </div>

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
    </div>
  );
}
