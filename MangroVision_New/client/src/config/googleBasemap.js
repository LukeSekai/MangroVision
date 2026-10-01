import L from 'leaflet';

const GOOGLE_SATELLITE_URL =
  'https://mt{s}.google.com/vt/lyrs=s&x={x}&y={y}&z={z}';

export function createGoogleSatelliteLayer({ maxZoom = 22 } = {}) {
  return L.tileLayer(GOOGLE_SATELLITE_URL, {
    subdomains: ['0', '1', '2', '3'],
    minZoom: 0,
    maxNativeZoom: 21,
    maxZoom,
    noWrap: true,
    attribution: '&copy; Google Maps',
  });
}
