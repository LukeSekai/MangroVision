export function filterMonitoringPoints(points, organizationId, projectSiteId = null) {
  if (organizationId == null && projectSiteId == null) return points;
  return points.filter((point) => {
    const ownerId = point.source_organization_id ?? point.assigned_organization_id;
    const siteId = point.source_project_site_id ?? point.source_site_id;
    return (organizationId == null || (ownerId != null && Number(ownerId) === Number(organizationId)))
      && (projectSiteId == null || (siteId != null && Number(siteId) === Number(projectSiteId)));
  });
}

export function filterMonitoringFeatures(collection, organizationId) {
  if (organizationId == null || !Array.isArray(collection?.features)) return collection;
  return {
    ...collection,
    features: collection.features.filter((feature) => (
      feature.properties?.organization_id != null
      && Number(feature.properties.organization_id) === Number(organizationId)
    )),
  };
}

export function monitoringOrganizations(projectSites) {
  const organizations = new Map();
  for (const feature of projectSites?.features || []) {
    const id = feature.properties?.organization_id;
    const name = feature.properties?.organization_name || feature.properties?.organization;
    if (id != null && name) organizations.set(Number(id), String(name));
  }
  return [...organizations].map(([id, name]) => ({ id, name }))
    .sort((a, b) => a.name.localeCompare(b.name));
}
