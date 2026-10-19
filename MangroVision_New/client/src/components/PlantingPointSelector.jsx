import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import L from 'leaflet';
import { attachSeedlingBrush } from '../utils/seedlingBrush';
import { paintReplantingSelection } from '../utils/replantingBrush';
import { MAX_SITE_POINTS } from '../utils/projectSiteSelection';
import './SeedlingLocations.css';

export default function PlantingPointSelector({ map, points: available, selected, onChange, disabled }) {
  const [mode, setMode] = useState('click');
  const [painting, setPainting] = useState(false);
  const markers = useRef(null);
  const live = useRef({ selected, mode, disabled, onChange });
  const selectable = useMemo(() => available.map((point) => ({ ...point, planting_event_id: Number(point.id) })), [available]);
  const stopPainting = useCallback(() => setMode('click'), []);
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
    const renderer = L.canvas({ pane: 'projectSiteSelection' });
    const group = L.featureGroup().addTo(map);
    markers.current = group;
    available.forEach((point) => {
      const id = Number(point.id);
      const marker = L.circleMarker([point.latitude, point.longitude], {
        renderer, radius: 3.5, weight: 1, color: '#166534', fillColor: '#16a34a', fillOpacity: .7,
        bubblingMouseEvents: false, sitePointId: id,
      });
      const label = document.createElement('span');
      label.textContent = `Point ${point.point_num ?? id} · Click to select`;
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
    return () => {
      group.remove();
      L.Util?.cancelAnimFrame(renderer._redrawRequest);
      renderer.remove();
      markers.current = null;
    };
  }, [map, available]);

  useEffect(() => {
    const ids = new Set(selected);
    markers.current?.eachLayer((marker) => {
      const chosen = ids.has(marker.options.sitePointId);
      marker.setRadius(chosen ? 5 : 3.5);
      marker.setStyle({ color: chosen ? '#5b21b6' : '#166534', fillColor: chosen ? '#7c3aed' : '#16a34a', fillOpacity: chosen ? 1 : .7 });
    });
  }, [selected, available]);

  useEffect(() => {
    if (!map || mode === 'click' || disabled) return undefined;
    return attachSeedlingBrush(map, selectable, mode,
      (ids, operation) => {
        const current = live.current;
        const next = paintReplantingSelection(current.selected, ids, operation, MAX_SITE_POINTS);
        live.current.selected = next;
        current.onChange(next);
      },
      stopPainting, setPainting);
  }, [map, mode, disabled, selectable, stopPainting]);

  return <div className="planting-point-selector">
    <div className="project-site-selection-tools" role="group" aria-label="Point selection tool">
      {[['click', 'Move / click'], ['select', 'Brush select'], ['deselect', 'Brush erase']].map(([value, label]) =>
        <button key={value} type="button" aria-pressed={mode === value} disabled={disabled} onClick={() => setMode(value)}>{label}</button>)}
    </div>
    {mode !== 'click' && <p className="text-sm" role="status">{painting ? 'Move over the points. Click again to stop.' : 'Click the map to start brushing. Escape returns to Move / click.'}</p>}
    <div className="project-site-selection-count" role="status"><strong>{selected.length} selected</strong>
      <button type="button" className="btn btn-ghost btn-sm" disabled={disabled || !selected.length} onClick={() => onChange([])}>Clear</button></div>
    {!available.length && <p className="text-sm">No unclaimed planned points are available. Release dead locations from their old site before selecting them here.</p>}
    {selected.length === MAX_SITE_POINTS && <p className="text-sm">Selection limit: {MAX_SITE_POINTS.toLocaleString()} points.</p>}
  </div>;
}
