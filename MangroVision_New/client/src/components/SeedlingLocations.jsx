import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { ORTHOPHOTO_TILE_URL, ORTHOPHOTO_BOUNDS, ORTHOPHOTO_MAX_NATIVE_ZOOM } from '../config/mapTiles';
import { filterSeedlings, seedlingMarkerStyle, toggleSeedling, visibleSeedlings } from '../utils/monitoringLocations';
import { paintReplantingSelection } from '../utils/replantingBrush';
import { attachSeedlingBrush } from '../utils/seedlingBrush';
import './SeedlingLocations.css';

const API = import.meta.env.VITE_API_BASE || '';
const date = (value) => new Date(value).toLocaleDateString('en-PH', { timeZone: 'Asia/Manila' });
const reference = (p) => p.reference || `${p.analysis_name} · Point ${p.point_num}`;
const REPLACEMENT_COLORS = { awaiting_review: '#dc2626', approved: '#b45309', assigned: '#2563eb', completed: '#166534' };

function FieldSheet({ snapshot, onDone }) {
  useEffect(() => {
    const after = () => onDone();
    window.addEventListener('afterprint', after);
    const frame = requestAnimationFrame(() => window.print());
    return () => { cancelAnimationFrame(frame); window.removeEventListener('afterprint', after); };
  }, [onDone]);
  const { bounds: b, points } = snapshot;
  const xy = (p) => [30 + (p.longitude - b.west) / (b.east - b.west || 1) * 640,
    30 + (b.north - p.latitude) / (b.north - b.south || 1) * 380];
  return createPortal(<section className="seedling-field-sheet">
    <h1>MangroVision monitoring field sheet</h1>
    <p>Prepared {date(new Date())} · {points.length} planting locations · Inspector: __________________</p>
    <p>Visit date: __________________ · Mark dead seedlings by their analysis and point reference.</p>
    <svg viewBox="0 0 700 450" aria-label="Numbered planting map of the selected area">
      <rect x="30" y="30" width="640" height="380" fill="white" stroke="#64748b" />
      {[1, 2, 3].map((n) => <g key={n} stroke="#e2e8f0"><path d={`M${30 + n * 160},30 V410`} /><path d={`M30,${30 + n * 95} H670`} /></g>)}
      <text x="640" y="20" fontSize="12">↑ North</text>
      <text x="30" y="20" fontSize="10">{b.north.toFixed(6)}° N</text>
      <text x="30" y="435" fontSize="10">{b.south.toFixed(6)}° N · {b.west.toFixed(6)}° E to {b.east.toFixed(6)}° E</text>
      {points.map((p) => { const [x, y] = xy(p); return <g key={p.planting_event_id}>
        <circle cx={x} cy={y} r="3" fill="#166534" />
        <text x={x + 5} y={y - 5} fontSize="8">{p.analysis_name?.replace('Analysis ', 'A')}–P{p.point_num}</text>
      </g>; })}
    </svg>
    <p>Map labels: A3–P17 = Analysis 3 · Point 17. Coordinates identify mapped planting locations.</p>
    <table><thead><tr><th>Location</th><th>Site / assignment</th><th>Coordinates</th><th>Planted</th><th>Dead?</th><th>Notes</th></tr></thead>
      <tbody>{points.map((p) => <tr key={p.planting_event_id}><td>{reference(p)}</td><td>{p.project_site_name}<br />{p.assignment_title || p.assignment_id}</td>
        <td>{Number(p.latitude).toFixed(6)}, {Number(p.longitude).toFixed(6)}</td><td>{date(p.planted_at)}</td><td>☐</td><td>________________</td></tr>)}</tbody></table>
    <button type="button" className="seedling-no-print" onClick={onDone}>Close print preview</button>
  </section>, document.body);
}

function markerAppearance(point, zoom, chosen) {
  const color = chosen ? '#7c3aed' : REPLACEMENT_COLORS[point.replanting_status]
    || (point.selectable === false ? '#64748b' : '#166534');
  return { ...seedlingMarkerStyle(zoom, chosen), color: chosen ? '#5b21b6' : color,
    fillColor: color, fillOpacity: point.selectable === false && !chosen ? 0.55 : 0.88 };
}

export function SeedlingMap({ points, selected = [], onToggle, onBounds, selectionMode = 'click', onPaint, onStopPainting, onPaintingChange }) {
  const container = useRef(null);
  const mapRef = useRef(null);
  const layer = useRef(null);
  const callback = useRef(onBounds);
  const actions = useRef({ onToggle, selectionMode, selected });
  const brushCleanup = useRef(null);
  useEffect(() => { callback.current = onBounds; }, [onBounds]);
  useEffect(() => { actions.current = { onToggle, selectionMode, selected }; }, [onToggle, selectionMode, selected]);
  useEffect(() => {
    const map = L.map(container.current).fitBounds(ORTHOPHOTO_BOUNDS);
    mapRef.current = map;
    L.tileLayer('https://tile.openstreetmap.org/{z}/{x}/{y}.png', { attribution: '© OpenStreetMap contributors', maxZoom: 23, maxNativeZoom: 19 }).addTo(map);
    L.tileLayer(ORTHOPHOTO_TILE_URL, { bounds: ORTHOPHOTO_BOUNDS, maxNativeZoom: ORTHOPHOTO_MAX_NATIVE_ZOOM, maxZoom: 23 }).addTo(map);
    layer.current = L.featureGroup().addTo(map);
    L.control.scale({ imperial: false }).addTo(map);
    const report = () => { const b = map.getBounds(); callback.current?.({ north: b.getNorth(), south: b.getSouth(), east: b.getEast(), west: b.getWest() }); };
    const resizeMarkers = () => {
      layer.current?.eachLayer((marker) => {
        const style = seedlingMarkerStyle(map.getZoom(), marker.options.mvSelected);
        marker.setRadius(style.radius);
        marker.setStyle({ weight: style.weight });
      });
    };
    map.on('moveend', report);
    map.on('zoomend', resizeMarkers);
    const observer = new ResizeObserver(() => map.invalidateSize());
    observer.observe(container.current);
    return () => {
      brushCleanup.current?.();
      observer.disconnect();
      map.remove();
      mapRef.current = null;
      layer.current = null;
    };
  }, []);
  useEffect(() => {
    const group = layer.current;
    group.clearLayers();
    visibleSeedlings(points).forEach((point) => {
      const chosen = actions.current.selected.includes(point.planting_event_id);
      const marker = L.circleMarker([point.latitude, point.longitude], {
        ...markerAppearance(point, mapRef.current.getZoom(), chosen),
        mvSelected: chosen, mvPoint: point,
      });
      const label = document.createElement('span');
      label.textContent = reference(point);
      marker.bindTooltip(label);
      const toggle = () => {
        const current = actions.current;
        if (point.selectable !== false || current.selected.includes(point.planting_event_id)) {
          current.onToggle?.(point.planting_event_id);
        }
      };
      marker.on('click', () => { if (actions.current.selectionMode === 'click') toggle(); });
      marker.on('add', () => {
        const element = marker.getElement();
        if (!element) return;
        element.setAttribute('aria-label', reference(point));
        element.addEventListener('keydown', (event) => {
          if (event.key === 'Enter' || event.key === ' ') {
            event.preventDefault();
            toggle();
          }
        });
      });
      group.addLayer(marker);
    });
  }, [points]);
  useEffect(() => {
    const chosenIds = new Set(selected);
    layer.current.eachLayer((marker) => {
      const point = marker.options.mvPoint;
      const chosen = chosenIds.has(point.planting_event_id);
      if (marker.options.mvSelected !== chosen) {
        marker.options.mvSelected = chosen;
        const style = markerAppearance(point, mapRef.current.getZoom(), chosen);
        marker.setRadius(style.radius);
        marker.setStyle(style);
      }
      const element = marker.getElement();
      if (!element) return;
      const interactive = Boolean(onToggle) && (point.selectable !== false || chosen);
      element.setAttribute('tabindex', interactive ? '0' : '-1');
      if (onToggle) {
        element.setAttribute('role', 'button');
        element.setAttribute('aria-pressed', String(chosen));
        element.setAttribute('aria-disabled', String(!interactive));
      } else {
        element.removeAttribute('role');
        element.removeAttribute('aria-pressed');
        element.removeAttribute('aria-disabled');
      }
    });
  }, [points, selected, onToggle]);
  useEffect(() => {
    const valid = visibleSeedlings(points);
    if (valid.length) mapRef.current.fitBounds(L.latLngBounds(valid.map((p) => [p.latitude, p.longitude])), { padding: [30, 30], maxZoom: 22 });
  }, [points]);
  useEffect(() => {
    if (selectionMode === 'click' || !onPaint) return undefined;
    const cleanup = attachSeedlingBrush(mapRef.current, points, selectionMode, onPaint, onStopPainting, onPaintingChange);
    brushCleanup.current = cleanup;
    return cleanup;
  }, [points, selectionMode, onPaint, onStopPainting, onPaintingChange]);
  return <div ref={container} className="seedling-location-map" aria-label="Mapped seedling locations" />;
}

export default function SeedlingLocations({ organizationId, monitoredAt, recordId, selected = [], onChange, readOnly = false, disabled = false, maxSelected = Infinity }) {
  const [points, setPoints] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState('');
  const [retry, setRetry] = useState(0);
  const loadedLocations = useRef(null);
  const selectedRef = useRef(selected);
  const [selectionMode, setSelectionMode] = useState('click');
  const [painting, setPainting] = useState(false);
  const stopPainting = useCallback(() => setSelectionMode('click'), []);
  useEffect(() => { selectedRef.current = selected; }, [selected]);
  const [filters, setFilters] = useState({ site: '', assignment: '', search: '' });
  const [bounds, setBounds] = useState(null);
  const [printSnapshot, setPrintSnapshot] = useState(null);
  const closePrint = useCallback(() => setPrintSnapshot(null), []);
  useEffect(() => {
    const key = `${organizationId}:${monitoredAt}:${recordId}:${retry}`;
    if (loadedLocations.current === key) return undefined;
    const controller = new AbortController();
    async function load() {
      setLoading(true); setError('');
      try {
        const query = new URLSearchParams();
        if (monitoredAt) query.set('monitored_at', monitoredAt);
        if (recordId) query.set('record_id', recordId);
        const response = await fetch(`${API}/api/monitoring/organizations/${organizationId}/planting-locations?${query}`, { signal: controller.signal });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || 'Could not load planting locations.');
        if (!controller.signal.aborted) setPoints(payload.points || []);
      } catch (failure) { if (!controller.signal.aborted) setError(failure.message); }
      finally { if (!controller.signal.aborted) { loadedLocations.current = key; setLoading(false); } }
    }
    void load();
    return () => controller.abort();
  }, [organizationId, monitoredAt, recordId, retry]);
  const filtered = useMemo(() => filterSeedlings(points, filters), [points, filters]);
  const visible = visibleSeedlings(filtered, bounds);
  const toggle = useCallback((id) => {
    if (disabled || !onChange) return;
    const next = toggleSeedling(selectedRef.current, id, maxSelected);
    if (next !== selectedRef.current) {
      selectedRef.current = next;
      onChange(next);
    }
  }, [disabled, onChange, maxSelected]);
  const paint = useCallback((ids, mode) => {
    if (disabled || !onChange) return;
    const next = paintReplantingSelection(selectedRef.current, ids, mode, maxSelected);
    if (next !== selectedRef.current) {
      selectedRef.current = next;
      onChange(next);
    }
  }, [disabled, onChange, maxSelected]);
  const activeMode = readOnly || disabled || (selectionMode === 'select' && maxSelected <= 0) ? 'click' : selectionMode;
  const choices = (key, labelKey) => [...new Map(points.filter((p) => p[key] != null).map((p) => [p[key], p[labelKey] || String(p[key])])).entries()];
  return <div className="seedling-locations">
    {readOnly ? <div className="seedling-location-filters">
      <label>Project site<select value={filters.site} onChange={(e) => setFilters({ ...filters, site: e.target.value, assignment: '' })}><option value="">All sites</option>{choices('project_site_id', 'project_site_name').map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select></label>
      <label>Assignment<select value={filters.assignment} onChange={(e) => setFilters({ ...filters, assignment: e.target.value })}><option value="">All assignments</option>{choices('assignment_id', 'assignment_title').map(([id, name]) => <option key={id} value={id}>{name}</option>)}</select></label>
    </div> : null}
    {loading ? <p role="status">Loading planting locations…</p> : error ? <p role="alert">{error} <button type="button" onClick={() => setRetry(retry + 1)}>Retry</button></p> : <>
      {!readOnly ? <>
        <div className="seedling-selection-tools" role="group" aria-label="Dead seedling selection tool">
          {[['click', 'Move / click'], ['select', 'Brush select'], ['deselect', 'Brush erase']].map(([mode, label]) => <button key={mode} type="button"
            className={activeMode === mode ? 'is-active' : ''} aria-pressed={activeMode === mode}
            disabled={disabled || (mode === 'select' && maxSelected <= 0) || (mode === 'deselect' && !selected.length)}
            onClick={() => setSelectionMode(mode)}>{label}</button>)}
        </div>
        <p className="seedling-selection-help" aria-live="polite">{activeMode === 'click'
          ? 'Click a dead seedling to select it, or use Brush select to sweep over several. Click a purple point again to deselect it.'
          : <><strong>{painting ? 'Brush active.' : 'Brush ready.'}</strong>{' '}{painting
            ? `Move over ${activeMode === 'select' ? 'dead seedlings to select them' : 'purple points to erase them'}. Click the map again to stop.`
            : `Click the map to start ${activeMode === 'select' ? 'selecting dead seedlings' : 'erasing selected points'}; click again to stop.`}
            {' '}On a touch screen, drag across points and lift your finger to stop. Press Esc or choose Move / click to finish.</>}
          {' '}Purple means selected. Grey points are unavailable for this visit.</p>
      </> : <p>Zoom to a small section, then print its numbered map and matching checklist.</p>}
      {!readOnly && Number.isFinite(maxSelected) ? <p role="status">{maxSelected === 0 ? 'Enter the total deaths above before selecting locations.' : `${selected.length} of ${maxSelected} deaths identified.${selected.length >= maxSelected ? ' All reported deaths are located. Deselect a point to choose another.' : ''}`}</p> : null}
      <SeedlingMap points={filtered} selected={selected} onToggle={readOnly || disabled ? undefined : toggle} onBounds={setBounds}
        selectionMode={activeMode} onPaint={readOnly || disabled ? undefined : paint} onStopPainting={stopPainting} onPaintingChange={setPainting} />
      {readOnly ? <><div className="seedling-location-toolbar"><span>{visible.length} points in this map view</span>
        <button type="button" disabled={!bounds || !visible.length} onClick={() => setPrintSnapshot({ bounds, points: visible })}>Print visible area</button></div>
      <div className="seedling-location-checklist">
        {filtered.map((p) => <label key={p.planting_event_id}>
          <span><strong>{reference(p)}</strong><small>{p.project_site_name || 'No site'} · Planted {date(p.planted_at)} · {Number(p.latitude).toFixed(6)}, {Number(p.longitude).toFixed(6)}{p.linked_record_id && p.linked_record_id !== recordId ? ' · Already counted in another visit' : ''}</small></span>
        </label>)}
        {!filtered.length ? <p>No planting locations match these filters.</p> : null}
      </div></> : null}
    </>}
    {printSnapshot ? <FieldSheet snapshot={printSnapshot} onDone={closePrint} /> : null}
  </div>;
}
