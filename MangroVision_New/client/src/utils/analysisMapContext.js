// Saved points already live in the main planting layer. Keep just their
// original photo boundary (or GPS center for older records), without replaying stale preview markers or
// short-lived raster URLs over the orthophoto.
function gpsCoordinate(value, limit) {
  if (value == null || (typeof value === 'string' && !value.trim()) || typeof value === 'boolean') return null;
  const number = Number(value);
  return Number.isFinite(number) && Math.abs(number) <= limit ? number : null;
}

function hasFootprint(footprint) {
  const rings = footprint?.type === 'Polygon' ? [footprint.coordinates?.[0]]
    : footprint?.type === 'MultiPolygon' && Array.isArray(footprint.coordinates) ? footprint.coordinates.map(polygon => polygon?.[0]) : [];
  return Boolean(rings?.length) && rings.every(ring => Array.isArray(ring) && ring.length >= 4
    && ring.every(coordinate => Array.isArray(coordinate)
      && gpsCoordinate(coordinate[0], 180) !== null && gpsCoordinate(coordinate[1], 90) !== null));
}

export function savedAnalysisMapContext(analysis, { preserveView = false } = {}) {
  if (!analysis?.map?.available) return null;
  const footprint = hasFootprint(analysis.map.analysis_footprint) ? analysis.map.analysis_footprint : null;
  const latitude = gpsCoordinate(analysis.metadata?.image_center_lat, 90);
  const longitude = gpsCoordinate(analysis.metadata?.image_center_lon, 180);
  const center = !footprint && latitude !== null && longitude !== null ? {
    type: 'Feature',
    properties: { name: 'Saved image center' },
    geometry: { type: 'Point', coordinates: [longitude, latitude] },
  } : null;
  return {
    analysis_id: analysis.analysis_id,
    uploaded_file_name: analysis.uploaded_file_name,
    saved: true,
    preserveMapView: preserveView,
    map: {
      available: Boolean(footprint || center),
      match: analysis.map.match,
      analysis_footprint: footprint,
      image_center_feature: center,
      analysis_footprint_quality: analysis.map.analysis_footprint_quality,
      location_status: analysis.map.location_status,
    },
  };
}

export function canViewSavedAnalysisOnMap(analysis) {
  return Boolean(savedAnalysisMapContext(analysis)?.map.available);
}

export function hasEstimatedAlignment(match) {
  if (!match?.success) return true;
  const source = match.projection_rotation_source || '';
  return source.startsWith('exif_') || source === 'heading_fallback';
}
