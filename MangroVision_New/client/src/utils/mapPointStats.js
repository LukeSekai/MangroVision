export function getMapPointStatus(point) {
  if (point.deleted_at || point.is_deleted) return null;
  // The API supplies the same classification used by dashboard aggregates.
  if (point.map_status) return point.map_status;
  if (point.death_at) return 'dead';
  const plantingStatus = point.planting_status ?? point.status;
  if (point.assignment_status === 'skipped' || plantingStatus === 'skipped') return 'skipped';
  if (point.assignment_status === 'completed' || plantingStatus === 'planted') return 'planted';
  if (point.eroded_unavailable || point.inside_eroded_zone) return 'unavailable';
  if (point.assignment_status === 'pending' || point.assignment_id != null
    || point.assigned_planter_id != null || point.assigned_planter_name) return 'assigned';
  return 'planned';
}

export function countMapPointStatuses(points) {
  const counts = {
    mapped: 0,
    planned: 0,
    assigned: 0,
    planted: 0,
    dead: 0,
    skipped: 0,
    unavailable: 0,
  };

  for (const point of points) {
    const status = getMapPointStatus(point);
    if (status === null) continue;
    counts.mapped += 1;
    counts[status] += 1;
  }

  return counts;
}
