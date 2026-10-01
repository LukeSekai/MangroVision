// Visit order only; assignment selection below determines point ownership.
export function zigzagPoints(points) {
  const grids = new Map();
  const missing = [];
  for (const point of points) {
    if (point.latitude == null || point.longitude == null
      || !Number.isFinite(Number(point.latitude)) || !Number.isFinite(Number(point.longitude))) {
      missing.push(point);
      continue;
    }
    const gridId = Number(point.analysis_id || 0);
    if (!grids.has(gridId)) grids.set(gridId, new Map());
    const rows = grids.get(gridId);
    const row = Math.round(Number(point.latitude) * 10000000);
    if (!rows.has(row)) rows.set(row, []);
    rows.get(row).push(point);
  }
  const tie = (a, b) => Number(a.point_num || 0) - Number(b.point_num || 0) || Number(a.id) - Number(b.id);
  const ordered = [];
  for (const gridId of [...grids.keys()].sort((a, b) => a - b)) {
    const rows = grids.get(gridId);
    [...rows.keys()].sort((a, b) => b - a).forEach((row, index) => {
      const direction = index % 2 ? -1 : 1;
      ordered.push(...rows.get(row).sort((a, b) => direction * (Number(a.longitude) - Number(b.longitude)) || tie(a, b)));
    });
  }
  return [...ordered, ...missing.sort(tie)];
}

function axisIndices(values) {
  const keys = [...new Set(values)].sort((a, b) => a - b);
  if (keys.length < 2) return new Map(keys.map(key => [key, 0]));
  const gaps = keys.slice(1).map((key, i) => key - keys[i]);
  const smallest = gaps.reduce((a, b) => Math.min(a, b), Infinity);
  const shortGaps = gaps.filter(gap => gap <= smallest * 1.5);
  const step = shortGaps.reduce((a, b) => a + b, 0) / shortGaps.length;
  return new Map(keys.map(key => [key, Math.floor((key - keys[0]) / step + 0.5)]));
}

// Mirrors zigzag_assignment_points in mangrovision_db/planting_order.py.
// Each strip alternates actual point locations across two adjacent levels.
export function zigzagAssignmentPoints(points) {
  const grids = new Map();
  const missing = [];
  for (const point of points) {
    if (point.latitude == null || point.longitude == null
      || !Number.isFinite(Number(point.latitude)) || !Number.isFinite(Number(point.longitude))) {
      missing.push(point);
      continue;
    }
    const gridId = Number(point.analysis_id || 0);
    if (!grids.has(gridId)) grids.set(gridId, []);
    grids.get(gridId).push(point);
  }
  const quantize = value => Math.floor(Number(value) * 10000000 + 0.5);
  const tie = (a, b) => Number(a.point_num || 0) - Number(b.point_num || 0) || Number(a.id) - Number(b.id);
  const ordered = [];
  for (const gridId of [...grids.keys()].sort((a, b) => a - b)) {
    const grid = grids.get(gridId);
    const rows = axisIndices(grid.map(p => quantize(p.latitude)));
    const columns = axisIndices(grid.map(p => quantize(p.longitude)));
    const key = p => {
      const row = rows.get(quantize(p.latitude));
      const column = columns.get(quantize(p.longitude));
      return [Math.floor(row / 2), (row + column) % 2, column];
    };
    ordered.push(...grid.sort((a, b) => {
      const ka = key(a), kb = key(b);
      return ka[0] - kb[0] || ka[1] - kb[1] || ka[2] - kb[2] || tie(a, b);
    }));
  }
  return [...ordered, ...missing.sort(tie)];
}

// The server revalidates ownership and availability under a transaction lock.
export function availableOrganizationPoints(points, organizationId, siteId) {
  if (organizationId == null || siteId == null) return [];
  return zigzagAssignmentPoints(points.filter((point) => (
    Number(point.source_organization_id) === Number(organizationId)
    && Number(point.source_project_site_id) === Number(siteId)
    && point.planting_status === 'planned'
    && point.assigned_planter_id == null
    && !point.deleted_at
    && !point.death_at
    && !point.eroded_unavailable
    && !point.inside_eroded_zone
  )));
}
