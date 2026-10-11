// Leaflet emits overlay events for both checkbox clicks and programmatic layer
// changes. Suppress the latter so temporary monitoring filters cannot overwrite
// the user's shared layer preferences.
export function connectMapLayerVisibility(map, layers, store) {
  const keys = new Map(Object.entries(layers).map(([key, layer]) => [layer, key]));
  let synchronizing = false;
  const onOverlayChange = (event) => {
    const key = keys.get(event.layer);
    if (!key || synchronizing) return;
    const visible = event.type === 'overlayadd';
    const state = store.getState();
    if (state.layerVisibility[key] !== visible) state.toggleLayer(key);
  };
  map.on('overlayadd overlayremove', onOverlayChange);

  return {
    sync(visibility) {
      synchronizing = true;
      try {
        for (const [key, layer] of Object.entries(layers)) {
          if (visibility[key] && !map.hasLayer(layer)) map.addLayer(layer);
          if (!visibility[key] && map.hasLayer(layer)) map.removeLayer(layer);
        }
      } finally {
        synchronizing = false;
      }
    },
    dispose() { map.off('overlayadd overlayremove', onOverlayChange); },
  };
}
