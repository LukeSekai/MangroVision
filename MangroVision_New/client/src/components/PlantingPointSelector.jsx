import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import L from 'leaflet';
import { attachSeedlingBrush } from '../utils/seedlingBrush';
import { paintReplantingSelection } from '../utils/replantingBrush';
import { MAX_SITE_POINTS } from '../utils/projectSiteSelection';
import { getPointRadius, getPointWeight } from '../utils/pointMarkerStyle';
import { createPointCanvas } from '../utils/pointCanvas';
import { PLANNED_SPECIES_COLORS, POINT_STATUS_COLORS } from '../utils/pointStatus';
import './SeedlingLocations.css';

export default function PlantingPointSelector({ map, points: available, selected, onChange, disabled }) {
  const [mode, setMode] = useState('click');
  const [painting, setPainting] = useState(false);
  const markers = useRef(null);
  const live = useRef({ selected, mode, disabled, onChange });
  const selectable = useMemo(() => available.map((point) => ({ ...point, planting_event_id: Number(point.id) })), [available]);
  const stopPainting = useCallback(() => setMode('click'), []);
  const refreshMarkers = useCallback(() => {
    if (!map) return;
    const ids = new Set(live.current.selected), zoom = map.getZoom();
    markers.current?.eachLayer((marker) => {
      const chosen = ids.has(marker.options.sitePointId);
      marker.setRadius(getPointRadius(zoom, chosen));
      marker.setStyle({ weight: getPointWeight(zoom, chosen),
        color: chosen ? '#4c1d95' : '#000000',
        fillColor: chosen ? '#7c3aed' : marker.options.siteSpeciesColor,
        fillOpacity: chosen ? .96 : .92,
      });
    });
  }, [map]);
  // React Activity reconnects effects when a retained tool becomes visible.
  // Resume in click mode so returning to the map never resumes an old brush.
  useEffect(() => {
    const timer = window.setTimeout(stopPainting, 0);
    return () => window.clearTimeout(timer);
  }, [map, stopPainting]);
  useEffect(() => { live.current = { selected, mode, disabled, onChange }; }, [selected, mode, disabled, onChange]);

  useEffect(() => {
    if (!map) return undefined;
    const pane = map.getPane('projectSiteSelection') || map.createPane('projectSiteSelection');
    pane.style.zIndex = '620';
    const renderer = createPointCanvas({ pane: 'projectSiteSelection' });
    const group = L.featureGroup().addTo(map);
    markers.current = group;
    available.forEach((point) => {
      const id = Number(point.id);
      const species = String(point.species || '').trim().toLowerCase();
      const marker = L.circleMarker([point.latitude, point.longitude], {
        renderer, radius: getPointRadius(map.getZoom()), weight: getPointWeight(map.getZoom()),
        color: '#000000', fillColor: PLANNED_SPECIES_COLORS[species] || POINT_STATUS_COLORS.planned, fillOpacity: .92,
        siteSpeciesColor: PLANNED_SPECIES_COLORS[species] || POINT_STATUS_COLORS.planned,
        bubblingMouseEvents: false, sitePointId: id,
      });
      const label = document.createElement('span');
      const speciesLabel = species === 'bungalon' ? 'Bungalon' : species === 'rhizophora' ? 'Rhizophora' : 'Species not recorded';
      label.textContent = `Point ${point.point_num ?? id} · ${speciesLabel} · Click to select`;
      marker.bindTooltip(label);
      marker.on('click', (event) => {
        const current = live.current;
        L.DomEvent.stopPropagation(event.originalEvent);
        if (current.mode !== 'click' || current.disabled) return;
        const next = paintReplantingSelection(current.selected, [id], current.selected.includes(id) ? 'deselect' : 'select', MAX_SITE_POINTS);
        live.current.selected = next;
        current.onChange(next);
      });
      marker.addTo(group);
    });
    refreshMarkers();
    map.on('zoomend', refreshMarkers);
    return () => {
      map.off('zoomend', refreshMarkers);
      group.remove();
      L.Util?.cancelAnimFrame(renderer._redrawRequest);
      renderer.remove();
      markers.current = null;
    };
  }, [map, available, refreshMarkers]);

  useEffect(() => {
    refreshMarkers();
  }, [selected, available, refreshMarkers]);

  useEffect(() => {
    if (!map || mode === 'click' || disabled) return undefined;
    return attachSeedlingBrush(map, selectable, mode,
      (ids, operation) => {
        const current = live.current;
        const next = paintReplantingSelection(current.selected, ids, operation, MAX_SITE_POINTS);
        live.current.selected = next;
        current.onChange(next);
      },
      stopPainting, setPainting, { holdToPaint: true });
  }, [map, mode, disabled, selectable, stopPainting]);

  return <div className="planting-point-selector">
    <div className="project-site-selection-tools" role="group" aria-label="Point selection tool">
      {[['click', 'Move / click'], ['select', 'Brush select'], ['deselect', 'Brush erase']].map(([value, label]) =>
        <button key={value} type="button" aria-pressed={mode === value} disabled={disabled} onClick={() => setMode(value)}>{label}</button>)}
    </div>
    {mode !== 'click' && <p className="text-sm" role="status">{painting ? 'Keep dragging. Release to stop brushing.' : 'Hold the left mouse button and drag. Release to stop. Escape returns to Move / click.'}</p>}
    <div className="project-site-selection-count" role="status"><strong>{selected.length} selected</strong>
      <button type="button" className="btn btn-ghost btn-sm" disabled={disabled || !selected.length} onClick={() => onChange([])}>Clear</button></div>
    {!available.length && <p className="text-sm">No unclaimed planned points are available. Release dead locations from their old site before selecting them here.</p>}
    {selected.length === MAX_SITE_POINTS && <p className="text-sm">Selection limit: {MAX_SITE_POINTS.toLocaleString()} points.</p>}
  </div>;
}
