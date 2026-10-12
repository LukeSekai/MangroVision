import L from 'leaflet';

export const GOOGLE_SATELLITE_URL =
  'https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}';
export const GOOGLE_SATELLITE_MAX_NATIVE_ZOOM = 21;

export function createGoogleSatelliteLayer({ maxZoom = 22 } = {}) {
  return L.tileLayer(GOOGLE_SATELLITE_URL, {
    subdomains: ['0', '1', '2', '3'],
    minZoom: 0,
    maxNativeZoom: GOOGLE_SATELLITE_MAX_NATIVE_ZOOM,
    maxZoom,
    noWrap: true,
    attribution: '&copy; Google Maps',
  });
}
