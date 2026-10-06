export function countMapPointStatuses(points) {
  const counts = {
    mapped: points.length,
    planned: 0,
    assigned: 0,
    completed: 0,
    skipped: 0,
    unavailable: 0,
  };

  for (const point of points) {
    const plantingStatus = point.planting_status ?? point.status;
    // Match the map's status priority so each saved point belongs to one group.
    if (point.assignment_status === 'skipped' || plantingStatus === 'skipped') {
      counts.skipped += 1;
    } else if (point.assignment_status === 'completed' || plantingStatus === 'planted') {
      counts.completed += 1;
    } else if (point.eroded_unavailable || point.inside_eroded_zone) {
      counts.unavailable += 1;
    } else if (point.assigned_planter_name) {
      counts.assigned += 1;
    } else {
      counts.planned += 1;
    }
  }

  return counts;
}
