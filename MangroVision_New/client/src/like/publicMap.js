import L from 'leaflet';
import 'leaflet/dist/leaflet.css';
import { createGoogleSatelliteLayer } from '../config/googleBasemap';
import { ORTHOPHOTO_BOUNDS, ORTHOPHOTO_MAX_NATIVE_ZOOM, ORTHOPHOTO_TILE_URL } from '../config/mapTiles';

// Intentionally independent of the staff map: only two raster tile layers,
// with no database reads, account/session requests, or editable map features.
export function createPublicMap(container, { onLoading, onError }) {
  const reducedMotion = window.matchMedia('(prefers-reduced-motion: reduce)').matches;
  const map = L.map(container, {
    minZoom: 10,
    maxZoom: ORTHOPHOTO_MAX_NATIVE_ZOOM,
    zoomControl: false,
    zoomAnimation: !reducedMotion,
    fadeAnimation: !reducedMotion,
  });
  let observer;
  try {
    map.attributionControl.setPrefix(false);
    map.createPane('orthophotoPane');
    map.getPane('orthophotoPane').style.zIndex = 250;

    const background = createGoogleSatelliteLayer({ maxZoom: ORTHOPHOTO_MAX_NATIVE_ZOOM });
    const orthophoto = L.tileLayer(ORTHOPHOTO_TILE_URL, {
      pane: 'orthophotoPane',
      bounds: ORTHOPHOTO_BOUNDS,
      maxNativeZoom: ORTHOPHOTO_MAX_NATIVE_ZOOM,
      maxZoom: ORTHOPHOTO_MAX_NATIVE_ZOOM,
      minZoom: 10,
      noWrap: true,
      updateWhenIdle: true,
      updateWhenZooming: false,
      attribution: 'LIKE orthophoto',
    });
    const loadingLayers = new Set();
    const trackLoading = (layer, loading) => {
      if (loading) loadingLayers.add(layer);
      else loadingLayers.delete(layer);
      onLoading(loadingLayers.size > 0);
    };
    for (const layer of [background, orthophoto]) {
      layer.on({ loading: () => trackLoading(layer, true), load: () => trackLoading(layer, false), tileerror: onError });
    }
    background.addTo(map);
    orthophoto.addTo(map);
    L.control.zoom({ position: 'bottomright' }).addTo(map);
    L.control.scale({ position: 'bottomleft', imperial: false }).addTo(map);

    const reset = () => map.fitBounds(ORTHOPHOTO_BOUNDS, { padding: [24, 24], maxZoom: 19, animate: false });
    reset();
    if (typeof ResizeObserver !== 'undefined') {
      observer = new ResizeObserver(() => map.invalidateSize({ pan: false }));
      observer.observe(container);
    }
    return {
      reset,
      retry: () => { background.redraw(); orthophoto.redraw(); },
      destroy: () => { observer?.disconnect(); map.remove(); },
    };
  } catch (issue) {
    observer?.disconnect();
    map.remove();
    throw issue;
  }
}
