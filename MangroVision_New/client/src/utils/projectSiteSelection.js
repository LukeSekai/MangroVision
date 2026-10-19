export const MAX_SITE_POINTS = 10000;

export function availableSitePoints(points) {
  return points.filter((point) => point.planting_status === 'planned'
    && point.source_project_site_id == null && point.source_organization_id == null
    && point.assigned_planter_id == null && point.assignment_id == null
    && !point.deleted_at && !point.is_deleted && !point.death_at && !point.planted_at
    && !point.eroded_unavailable && !point.inside_eroded_zone && !point.inside_forbidden_zone
    && point.inside_visible_map !== false
    && point.latitude != null && point.longitude != null
    && Number.isFinite(Number(point.latitude)) && Number.isFinite(Number(point.longitude)));
}
