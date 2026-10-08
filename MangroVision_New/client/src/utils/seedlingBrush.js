import { visibleSeedlings } from './monitoringLocations';
import { pointsAlongBrush, REPLANTING_BRUSH_RADIUS } from './replantingBrush';

// Keep screen-space painting independent of React renders and marker updates.
export function attachSeedlingBrush(map, points, mode, onPaint, onStop) {
  const container = map.getContainer();
  const brush = document.createElement('div');
  brush.className = 'seedling-selection-brush is-' + mode;
  brush.style.width = brush.style.height = REPLANTING_BRUSH_RADIUS * 2 + 'px';
  brush.setAttribute('aria-hidden', 'true');
  container.appendChild(brush);
  container.classList.add('is-brushing');
  const previousTouchAction = container.style.touchAction;
  container.style.touchAction = 'none';
  const handlers = [map.dragging, map.doubleClickZoom, map.boxZoom, map.touchZoom]
    .filter((handler) => handler?.enabled());
  handlers.forEach((handler) => handler.disable());
  const eligible = visibleSeedlings(points).filter((point) => mode === 'deselect' || point.selectable !== false);
  let projected = [];
  let previous = null;
  let moving = false;
  let activeTouch = null;
  const reset = () => { previous = null; brush.style.display = 'none'; };
  const project = () => {
    projected = eligible.map((point) => ({ id: point.planting_event_id,
      ...map.latLngToContainerPoint([point.latitude, point.longitude]) }));
    reset();
  };
  const sweep = (event) => {
    if (moving || (event.pointerType === 'touch' && event.pointerId !== activeTouch)
        || event.target.closest?.('.leaflet-control') || event.buttons > 1) {
      reset(); return;
    }
    const position = map.mouseEventToContainerPoint(event);
    const size = map.getSize();
    if (position.x < 0 || position.y < 0 || position.x > size.x || position.y > size.y) {
      reset(); return;
    }
    brush.style.display = 'block';
    brush.style.left = position.x + 'px';
    brush.style.top = position.y + 'px';
    const ids = pointsAlongBrush(projected, previous || position, position);
    previous = position;
    if (ids.length) onPaint(ids, mode);
  };
  const down = (event) => {
    if (event.target.closest?.('.leaflet-control') || event.button !== 0) return;
    if (event.pointerType === 'touch') {
      if (activeTouch !== null) return;
      activeTouch = event.pointerId;
      container.setPointerCapture(event.pointerId);
      event.preventDefault();
    }
    previous = null;
    sweep(event);
  };
  const up = (event) => {
    if (activeTouch !== null) {
      if (activeTouch !== event.pointerId) return;
      if (container.hasPointerCapture(activeTouch)) container.releasePointerCapture(activeTouch);
      activeTouch = null;
    }
    reset();
  };
  const blur = () => {
    if (activeTouch !== null && container.hasPointerCapture(activeTouch)) container.releasePointerCapture(activeTouch);
    activeTouch = null;
    reset();
  };
  const escape = (event) => {
    if (event.key !== 'Escape') return;
    event.preventDefault();
    // Finish the brush before the containing monitoring dialog handles Escape.
    event.stopPropagation();
    onStop();
  };
  const movingStart = () => { moving = true; reset(); };
  const movingEnd = () => { moving = false; project(); };
  project();
  map.on('movestart zoomstart', movingStart);
  map.on('moveend zoomend resize', movingEnd);
  container.addEventListener('pointermove', sweep);
  container.addEventListener('pointerdown', down);
  container.addEventListener('pointerup', up);
  container.addEventListener('pointercancel', up);
  container.addEventListener('pointerleave', reset);
  document.addEventListener('keydown', escape, true);
  window.addEventListener('blur', blur);
  let cleaned = false;
  return () => {
    if (cleaned) return;
    cleaned = true;
    blur();
    map.off('movestart zoomstart', movingStart);
    map.off('moveend zoomend resize', movingEnd);
    container.removeEventListener('pointermove', sweep);
    container.removeEventListener('pointerdown', down);
    container.removeEventListener('pointerup', up);
    container.removeEventListener('pointercancel', up);
    container.removeEventListener('pointerleave', reset);
    document.removeEventListener('keydown', escape, true);
    window.removeEventListener('blur', blur);
    handlers.forEach((handler) => handler.enable());
    container.style.touchAction = previousTouchAction;
    container.classList.remove('is-brushing');
    brush.remove();
  };
}
