// Saved points already live in the main planting layer. Keep just their
// original photo boundary here, without replaying stale preview markers or
// short-lived raster URLs over the orthophoto.
export function savedAnalysisMapContext(analysis, { preserveView = false } = {}) {
  if (!analysis?.map?.available) return null;
  return {
    analysis_id: analysis.analysis_id,
    uploaded_file_name: analysis.uploaded_file_name,
    saved: true,
    preserveMapView: preserveView,
    map: {
      available: true,
      match: analysis.map.match,
      analysis_footprint: analysis.map.analysis_footprint,
      analysis_footprint_quality: analysis.map.analysis_footprint_quality,
      location_status: analysis.map.location_status,
    },
  };
}

export function hasEstimatedAlignment(match) {
  if (!match?.success) return true;
  const source = match.projection_rotation_source || '';
  return source.startsWith('exif_') || source === 'heading_fallback';
}
