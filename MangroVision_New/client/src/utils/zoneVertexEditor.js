import L from 'leaflet';

// Edit every ring of Polygon/MultiPolygon drafts, preserving holes and parts.
// The saved map layer is untouched until the caller submits the draft.
export function zoneVertexEditor(map, original, onChange) {
  const geometry = JSON.parse(JSON.stringify(original));
  const group = L.featureGroup().addTo(map);
  const outline = L.geoJSON(geometry, { style: { color: '#ea580c', fillOpacity: .24 } }).addTo(group);
  const polygons = geometry.type === 'Polygon' ? [geometry.coordinates] : geometry.coordinates;
  const notify = () => {
    outline.clearLayers().addData(geometry);
    onChange(JSON.parse(JSON.stringify(geometry)));
  };
  polygons.forEach((rings) => rings.forEach((ring) => ring.slice(0, -1).forEach(([lng, lat], index) => {
    const marker = L.marker([lat, lng], {
      draggable: true, keyboard: true, title: 'Move zone vertex: drag or use arrow keys', alt: 'Move zone vertex: drag or use arrow keys',
      icon: L.divIcon({ className: 'zone-vertex-handle', iconSize: [12, 12] }),
    }).addTo(group);
    const update = () => {
      const position = marker.getLatLng();
      ring[index] = [position.lng, position.lat];
      if (index === 0) ring[ring.length - 1] = [...ring[0]];
      notify();
    };
    marker.on('dragend', update);
    const keydown = (event) => {
      const directions = { ArrowLeft: [-2, 0], ArrowRight: [2, 0], ArrowUp: [0, -2], ArrowDown: [0, 2] };
      if (!directions[event.key]) return;
      event.preventDefault(); event.stopPropagation();
      const point = map.latLngToContainerPoint(marker.getLatLng()).add(directions[event.key]);
      marker.setLatLng(map.containerPointToLatLng(point));
      update();
    };
    const bindKeyboard = () => marker.getElement()?.addEventListener('keydown', keydown);
    bindKeyboard();
    marker.on('add', bindKeyboard);
    marker.on('remove', () => marker.getElement()?.removeEventListener('keydown', keydown));
  })));
  return group;
}
