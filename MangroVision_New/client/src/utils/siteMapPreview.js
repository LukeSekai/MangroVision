import { getMapPointStatus } from './mapPointStats.js';
import { PLANNED_SPECIES_COLORS, POINT_STATUS_COLORS } from './pointStatus.js';

const TILE_SIZE = 256;
const MAX_LATITUDE = 85.0511287798066;

function project([longitude, latitude]) {
  const lat = Math.max(-MAX_LATITUDE, Math.min(MAX_LATITUDE, latitude)) * Math.PI / 180;
  // The same Web Mercator / XYZ coordinates used by the main Leaflet map.
  // https://wiki.openstreetmap.org/wiki/Slippy_map_tilenames
  return [(longitude + 180) / 360, (1 - Math.asinh(Math.tan(lat)) / Math.PI) / 2];
}

function projectedPolygons(geometry) {
  const polygons = geometry?.type === 'Polygon' ? [geometry.coordinates]
    : geometry?.type === 'MultiPolygon' ? geometry.coordinates : null;
  if (!Array.isArray(polygons) || !polygons.length) return null;
  const valid = polygons.every((polygon) => Array.isArray(polygon) && polygon.length
    && polygon.every((ring) => Array.isArray(ring) && ring.length >= 3
      && ring.every((point) => Array.isArray(point) && point.length >= 2
        && Number.isFinite(point[0]) && Number.isFinite(point[1])
        && Math.abs(point[0]) <= 180 && Math.abs(point[1]) <= 90)));
  return valid ? polygons.map((polygon) => polygon.map((ring) => ring.map(project))) : null;
}

function tileUrl(template, z, x, y) {
  return template.replace('{s}', String((x + y) % 4))
    .replace('{z}', String(z)).replace('{x}', String(x)).replace('{y}', String(y));
}

function coordinates(point) {
  if (point.longitude == null || point.latitude == null
    || String(point.longitude).trim() === '' || String(point.latitude).trim() === '') return null;
  const longitude = Number(point.longitude), latitude = Number(point.latitude);
  return Number.isFinite(longitude) && Number.isFinite(latitude)
    && Math.abs(longitude) <= 180 && Math.abs(latitude) <= 90 ? [longitude, latitude] : null;
}

export function groupSitePreviewPoints(points) {
  const groups = new Map();
  const seen = new Map();
  for (const point of points) {
    // Use the same resolved site ownership as the staff map's site counts.
    const owner = point.source_project_site_id ?? point.source_site_id;
    if (owner == null || String(owner).trim() === '' || getMapPointStatus(point) === null
      || point.inside_visible_map === false || !coordinates(point)) continue;
    const siteId = String(owner).trim();
    if (!groups.has(siteId)) { groups.set(siteId, []); seen.set(siteId, new Set()); }
    const pointId = point.planting_point_id ?? point.id;
    if (pointId != null) {
      if (seen.get(siteId).has(String(pointId))) continue;
      seen.get(siteId).add(String(pointId));
    }
    groups.get(siteId).push(point);
  }
  return groups;
}

export function buildSiteMapPreview(geometry, { satellite, orthophoto, points = [], width = 320, height = 180, padding = 18 }) {
  const polygons = projectedPolygons(geometry);
  if (!polygons) return null;
  const { minX, maxX, minY, maxY } = polygons.flat(2).reduce((bounds, [x, y]) => ({
    minX: Math.min(bounds.minX, x), maxX: Math.max(bounds.maxX, x),
    minY: Math.min(bounds.minY, y), maxY: Math.max(bounds.maxY, y),
  }), { minX: Infinity, maxX: -Infinity, minY: Infinity, maxY: -Infinity });
  if (maxX <= minX || maxY <= minY) return null;
  const scale = Math.min((width - padding * 2) / (maxX - minX), (height - padding * 2) / (maxY - minY));
  const zoom = Math.max(0, Math.min(22, Math.floor(Math.log2(scale / TILE_SIZE))));
  const worldSize = TILE_SIZE * 2 ** zoom;
  const left = (minX + maxX) / 2 * worldSize - width / 2;
  const top = (minY + maxY) / 2 * worldSize - height / 2;
  const screen = ([x, y]) => [x * worldSize - left, y * worldSize - top];
  const path = polygons.flatMap((polygon) => polygon.map((ring) => (
    ring.map((point, index) => `${index ? 'L' : 'M'}${screen(point).map((value) => value.toFixed(2)).join(',')}`).join(' ') + ' Z'
  ))).join(' ');

  const markers = points.flatMap((point, index) => {
    const location = coordinates(point);
    const status = getMapPointStatus(point);
    if (!location || status === null || point.inside_visible_map === false) return [];
    const [x, y] = screen(project(location));
    if (x < 0 || x > width || y < 0 || y > height) return [];
    const species = typeof point.species === 'string' ? point.species.toLowerCase() : '';
    const color = (status === 'planned' ? PLANNED_SPECIES_COLORS[species] : null)
      || POINT_STATUS_COLORS[status] || POINT_STATUS_COLORS.planned;
    return [{ id: point.planting_point_id ?? point.id ?? index, x, y, status, color }];
  });

  const tiles = [];
  function addLayer(layer, kind) {
    if (!layer?.url) return;
    const nativeZoom = Math.min(zoom, layer.maxNativeZoom);
    const tileSize = TILE_SIZE * 2 ** (zoom - nativeZoom);
    const tileCount = 2 ** nativeZoom;
    const bounds = layer.bounds ? [
      project([layer.bounds[0][1], layer.bounds[1][0]]),
      project([layer.bounds[1][1], layer.bounds[0][0]]),
    ] : null;
    for (let y = Math.max(0, Math.floor(top / tileSize)); y <= Math.min(tileCount - 1, Math.ceil((top + height) / tileSize) - 1); y += 1) {
      for (let x = Math.max(0, Math.floor(left / tileSize)); x <= Math.min(tileCount - 1, Math.ceil((left + width) / tileSize) - 1); x += 1) {
        // Orthophoto requests follow the same coverage restriction as the main map.
        if (bounds && ((x + 1) / tileCount <= bounds[0][0] || x / tileCount >= bounds[1][0]
          || (y + 1) / tileCount <= bounds[0][1] || y / tileCount >= bounds[1][1])) continue;
        tiles.push({ key: `${kind}:${nativeZoom}:${x}:${y}`, kind,
          url: tileUrl(layer.url, nativeZoom, x, y),
          x: x * tileSize - left, y: y * tileSize - top, size: tileSize,
        });
      }
    }
  }
  addLayer(satellite, 'satellite');
  if (zoom >= 10) addLayer(orthophoto, 'orthophoto');
  return { width, height, zoom, path, tiles, markers };
}
