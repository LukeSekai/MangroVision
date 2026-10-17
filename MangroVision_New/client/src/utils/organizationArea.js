// Stable organization sites define their area. Assignment boundaries provide
// a map focus for an organization's first site, not a limit on that new site.
export function organizationAreaFeatures(projectSites, assignmentZones, organizationId) {
  if (organizationId == null || String(organizationId).trim() === '') return [];
  const owned = (collection) => (collection?.features || []).filter((feature) => (
    feature.properties?.organization_id != null
    && String(feature.properties.organization_id) === String(organizationId)
    && ['Polygon', 'MultiPolygon'].includes(feature.geometry?.type)
  ));
  const sites = owned(projectSites);
  return [...(sites.length ? sites : owned(assignmentZones)), ...(projectSites?.released_areas?.features || [])];
}

export function focusMapArea(map, bounds) {
  if (!map || !bounds?.isValid()) return false;
  map.closePopup();
  map.eachLayer((layer) => layer.closeTooltip?.());
  map.stop();
  const panel = document.querySelector('.planting-workspace-panel:not(.floating-panel-hidden)')
    || document.querySelector('[data-guide-panel="Zone Editor"].floating-panel:not(.floating-panel-hidden)');
  const rightPadding = map.getSize().x >= 900 ? (panel?.offsetWidth || 400) + 36 : 36;
  map.fitBounds(bounds, {
    paddingTopLeft: [48, 48], paddingBottomRight: [rightPadding, 48],
    maxZoom: Math.min(22, map.getMaxZoom()), animate: false,
  });
  return true;
}
