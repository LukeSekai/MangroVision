export function filterSeedlings(points, { site = '', assignment = '', search = '' } = {}) {
  const query = search.trim().toLowerCase();
  return points.filter((point) => (!site || String(point.project_site_id) === site)
    && (!assignment || String(point.assignment_id) === assignment)
    && (!query || `${point.reference} ${point.analysis_name} ${point.point_num}`.toLowerCase().includes(query)));
}

export function toggleSeedling(ids, id, maximum = Infinity) {
  if (ids.includes(id)) return ids.filter((value) => value !== id);
  return new Set(ids).size >= maximum ? ids : [...new Set([...ids, id])];
}

export function visibleSeedlings(points, bounds) {
  return points.filter((p) => Number.isFinite(Number(p.latitude)) && p.latitude !== null
    && Number.isFinite(Number(p.longitude)) && p.longitude !== null
    && (!bounds || (p.latitude >= bounds.south && p.latitude <= bounds.north
      && p.longitude >= bounds.west && p.longitude <= bounds.east)));
}

export function deathLocationCounts(ids, total) {
  const located = new Set(ids).size;
  const reported = Number(total);
  return { located, unlocated: reported - located, reported,
    valid: total != null && String(total).trim() !== '' && Number.isInteger(reported) && reported >= located };
}

export function seedlingMarkerStyle(zoom, selected = false) {
  const level = Number.isFinite(zoom) ? zoom : 18;
  const radius = level >= 22 ? 2.4 : level >= 20 ? 1.9 : level >= 18 ? 1.5 : 1.1;
  return {
    radius: selected ? radius + 1.3 : radius,
    weight: selected ? 1.2 : level >= 20 ? 0.5 : 0.35,
  };
}
