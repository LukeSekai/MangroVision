import { useEffect, useMemo, useRef } from 'react';
import { useLocation } from 'react-router-dom';
import L from 'leaflet';
import { cachedTileLayer } from '../utils/cachedTileLayer';
import 'leaflet/dist/leaflet.css';
import { useMapStore } from '../stores/mapStore';
import { ORTHOPHOTO_BOUNDS, ORTHOPHOTO_MAX_NATIVE_ZOOM, ORTHOPHOTO_TILE_URL } from '../config/mapTiles';
import { createGoogleSatelliteLayer } from '../config/googleBasemap';
import { getPlanterColor, getPlanterTint } from '../utils/planterColors';
import { hasEstimatedAlignment } from '../utils/analysisMapContext';
import { filterMapPoints } from '../utils/mapPointFilters';
import { getMapPointStatus } from '../utils/mapPointStats';
import { bindProjectSiteInfo, emptyProjectSiteCounts, projectSiteId, projectSitePointCounts, projectSiteStyle } from '../utils/projectSiteInfo';
import { POINT_STATUS_COLORS as STATUS_COLORS, PLANNED_SPECIES_COLORS as SPECIES_COLORS, pointStatusLabel } from '../utils/pointStatus';
import { filterMonitoringFeatures, filterMonitoringPoints } from '../utils/monitoringOrganizationFilter';
import { pointsAlongBrush, REPLANTING_BRUSH_RADIUS } from '../utils/replantingBrush';
import { connectMapLayerVisibility } from '../utils/mapLayerVisibility';
import { bindZoneInfo, zoneName } from '../utils/zoneInfo';
import { focusMapArea } from '../utils/organizationArea';
import './MapView.css';
import './ProjectSiteInfo.css';

// Fix Leaflet default icon paths
delete L.Icon.Default.prototype._getIconUrl;
L.Icon.Default.mergeOptions({
  iconRetinaUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon-2x.png',
  iconUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-icon.png',
  shadowUrl: 'https://unpkg.com/leaflet@1.9.4/dist/images/marker-shadow.png',
});

// Admin-side palette. Mirrors the planter view so the same point reads the
// same colour on both maps:
//   - planned   = green  (no planter assigned yet)
//   - assigned  = blue   (assigned but not yet planted)
//   - planted   = yellow (planter completed this point in the field)
//   - completed = yellow (DB enum equivalent of planted; still distinct from
//                         the never-touched "planned" green so reviewers can
//                         see at a glance which points are done)
// Species-driven palette for *planned* (not yet assigned) points. Rhizophora
// requires 2 m spacing and is shown in magenta; Bungalon requires 1 m and is green. Falls back to
// the generic 'planned' green when the analysis didn't record a species
// (legacy data). Status colors above still take precedence for any point
// that has already moved past 'planned'.
const SPECIES_LABELS = {
  rhizophora: 'Rhizophora — 2 m spacing',
  bungalon: 'Bungalon — 1 m spacing',
};

const M_PER_DEG_LAT = 111320;
const CROSS_SPECIES_MIN_SPACING_M = 2.0;

const PREVIEW_COLORS = {
  safe: '#0f9d58',
  erodedUnavailable: '#f97316',
  forbidden: '#c62828',
  danger: '#dc2626',
  canopy: '#7c3aed',
  spacing: '#2563eb',
};

const PROJECT_SITE_STYLE = {
  color: '#0284c7',
  weight: 2.5,
  fillColor: '#0ea5e9',
  fillOpacity: 0.04,
  dashArray: '10 5',
};

const FOCUSED_PROJECT_SITE_STYLE = {
  color: '#047857',
  weight: 4,
  fillColor: '#10b981',
  fillOpacity: 0.12,
  dashArray: null,
};

// All boundaries share the points' renderer for hover hit-testing. Keep
// points above hazards, then project-site analytics above assignment summaries.
function orderZoneLayers(layers) {
  for (const key of ['warningLayer', 'erodedLayer', 'forbiddenLayer', 'projectSiteLayer', 'siteZoneLayer']) {
    layers[key]?.eachLayer((layer) => layer.bringToBack());
  }
}

const getPointRadius = (zoom, selected = false) => {
  const base = zoom >= 22
    ? 3.5
    : zoom >= 21
      ? 2.4
      : zoom >= 20
        ? 1.6
        : zoom >= 19
          ? 1.05
          : zoom >= 18
            ? 0.8
            : zoom >= 17
              ? 0.65
              : 0.55;
  return selected ? Math.max(base + 1.2, base * 1.8) : base;
};

const getPointWeight = (zoom, selected = false) => {
  if (selected) return zoom >= 20 ? 1.8 : 1.2;
  if (zoom >= 22) return 1.1;
  if (zoom >= 21) return 0.75;
  if (zoom >= 20) return 0.45;
  if (zoom >= 19) return 0.2;
  return 0;
};

const getPreviewFilteredRadius = (zoom) => Math.max(0.6, getPointRadius(zoom) - 0.35);
const getAnalysisPointRadius = (zoom) => Math.max(0.45, getPointRadius(zoom) * 0.45);

function escapeHtml(value) {
  return String(value ?? '')
    .replace(/&/g, '&amp;')
    .replace(/</g, '&lt;')
    .replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;')
    .replace(/'/g, '&#39;');
}

function localDistanceM(a, b) {
  const lat1 = Number(a.latitude);
  const lon1 = Number(a.longitude);
  const lat2 = Number(b.latitude);
  const lon2 = Number(b.longitude);
  if (![lat1, lon1, lat2, lon2].every(Number.isFinite)) return Infinity;
  const refLat = (lat1 + lat2) / 2;
  const mPerDegLon = M_PER_DEG_LAT * Math.max(0.2, Math.cos((refLat * Math.PI) / 180));
  const dLatM = (lat1 - lat2) * M_PER_DEG_LAT;
  const dLonM = (lon1 - lon2) * mPerDegLon;
  return Math.hypot(dLatM, dLonM);
}

function speciesKey(point) {
  return typeof point.species === 'string' ? point.species.toLowerCase() : '';
}

function filterCrossSpeciesDisplayPoints(allPoints) {
  const greenBlockers = allPoints.filter((point) => {
    const status = getPointDisplayStatus(point);
    if (status !== 'planned') return false;
    const key = speciesKey(point);
    return key === 'bungalon' || !key;
  });
  if (!greenBlockers.length) return allPoints;

  return allPoints.filter((point) => {
    if (getPointDisplayStatus(point) !== 'planned' || speciesKey(point) !== 'rhizophora') {
      return true;
    }
    return !greenBlockers.some(
      (greenPoint) => localDistanceM(point, greenPoint) < CROSS_SPECIES_MIN_SPACING_M,
    );
  });
}

function getPointDisplayStatus(point) {
  const status = getMapPointStatus(point);
  if (status === 'unavailable') return 'eroded_unavailable';
  if (status === 'planted' && point.assignment_status === 'completed') return 'completed';
  return status;
}

function formatStatusLabel(status) {
  return pointStatusLabel(status);
}

export default function MapView() {
  const location = useLocation();
  const containerRef = useRef(null);
  const mapRef = useRef(null);
  const layersRef = useRef({});
  const pointsFingerprintRef = useRef('');
  const focusedProjectSiteRef = useRef('');
  const focusedMonitoringOrganizationRef = useRef('');
  const focusedRiskAreasRef = useRef('');
  // True while we're applying an external store change to the map. Lets the
  // moveend handler ignore that programmatic move so we don't write the same
  // view back to the store (which would cause a render loop).
  const skipNextSyncRef = useRef(false);

  const center = useMapStore((s) => s.center);
  const zoom = useMapStore((s) => s.zoom);
  const points = useMapStore((s) => s.points);
  const pointsLoaded = useMapStore((s) => s.pointsLoaded);
  const sitePointCounts = useMemo(() => projectSitePointCounts(points), [points]);
  // Species spacing is independent of selection and page mode. Calculate it
  // once per dataset, rather than repeating pairwise checks on each selection.
  const spacedPoints = useMemo(() => filterCrossSpeciesDisplayPoints(points), [points]);
  // Include every field: refreshed coordinates, ownership and popup details
  // must update even when IDs and planting statuses have not changed.
  const pointsDataKey = useMemo(() => JSON.stringify(spacedPoints), [spacedPoints]);
  const forbiddenZones = useMapStore((s) => s.forbiddenZones);
  const erodedZones = useMapStore((s) => s.erodedZones);
  const warningZones = useMapStore((s) => s.warningZones);
  const siteZones = useMapStore((s) => s.siteZones);
  const projectSites = useMapStore((s) => s.projectSites);
  const siteZoneMortality = useMapStore((s) => s.siteZoneMortality);
  const setSelectedPoint = useMapStore((s) => s.setSelectedPoint);
  const layerVisibility = useMapStore((s) => s.layerVisibility);
  const showLayers = useMapStore((s) => s.showLayers);
  const currentAnalysis = useMapStore((s) => s.currentAnalysis);
  const assignmentSelectedPointIds = useMapStore((s) => s.assignmentSelectedPointIds);
  const assignmentOrganizationId = useMapStore((s) => s.assignmentOrganizationId);
  const assignmentProjectSiteId = useMapStore((s) => s.assignmentProjectSiteId);
  const monitoringOrganizationId = useMapStore((s) => s.monitoringOrganizationId);
  const showDeadPointsOnly = useMapStore((s) => s.showDeadPointsOnly);
  const toggleReplantingPoint = useMapStore((s) => s.toggleReplantingPoint);
  const replantingSelectedPointIds = useMapStore((s) => s.replantingSelectedPointIds);
  const replantingSelectionMode = useMapStore((s) => s.replantingSelectionMode);
  const isAnalyticsMode = location.pathname === '/map';
  const isMonitoringMapMode = location.pathname === '/monitoring/map';
  // On /planters the admin needs to see WHO owns each assigned point at a
  // glance, so we colour assigned points by organization instead of the single
  // "assigned = blue" used everywhere else (Map Analytics, Delete Points, …).
  const isPlanterMode = location.pathname === '/planters';

  // Initialize map
  useEffect(() => {
    if (mapRef.current || !containerRef.current) return undefined;

    // Read the initial view once. The map writes user pans/zooms back into
    // mapStore, so subscribing this initialization effect to center/zoom would
    // tear Leaflet down and recreate it on every interaction.
    const initial = useMapStore.getState();
    pointsFingerprintRef.current = '';

    const map = L.map(containerRef.current, {
      center: [initial.center[1], initial.center[0]],
      zoom: initial.zoom,
      maxZoom: ORTHOPHOTO_MAX_NATIVE_ZOOM,
      zoomControl: false,
      attributionControl: true,
      preferCanvas: true,
    });

    map.attributionControl.setPosition('bottomleft');
    map.attributionControl.setPrefix(false);

    // Keep the drone orthomosaic visually above every background basemap.
    map.createPane('orthophotoPane');
    map.getPane('orthophotoPane').style.zIndex = 250;

    const googleSatellite = createGoogleSatelliteLayer({
      maxZoom: ORTHOPHOTO_MAX_NATIVE_ZOOM,
    });

    const osm = cachedTileLayer(
      'https://tile.openstreetmap.org/{z}/{x}/{y}.png',
      {
        maxNativeZoom: 19,
        maxZoom: ORTHOPHOTO_MAX_NATIVE_ZOOM,
        attribution: '&copy; OpenStreetMap contributors',
      }
    );

    // High-resolution drone orthomosaic
    const orthophoto = cachedTileLayer(
      ORTHOPHOTO_TILE_URL,
      {
        pane: 'orthophotoPane',
        maxZoom: ORTHOPHOTO_MAX_NATIVE_ZOOM,
        maxNativeZoom: ORTHOPHOTO_MAX_NATIVE_ZOOM,
        opacity: 1.0,
        errorTileUrl: '',
        minZoom: 10,
        bounds: ORTHOPHOTO_BOUNDS,
        noWrap: true,
        keepBuffer: 3,
        updateWhenIdle: true,
        updateWhenZooming: false,
      }
    );

    googleSatellite.addTo(map);
    orthophoto.addTo(map);

    L.control.zoom({ position: 'bottomright' }).addTo(map);
    L.control.scale({ position: 'bottomleft', metric: true, imperial: false }).addTo(map);


    // Data layers
    const pointLayer = L.layerGroup().addTo(map);

    const forbiddenLayer = L.geoJSON(null, {
      pane: 'overlayPane',
      style: { color: '#dc2626', weight: 2, fillColor: '#dc2626', fillOpacity: 0.15, dashArray: '6 4' },
    }).addTo(map);
    const erodedLayer = L.geoJSON(null, {
      pane: 'overlayPane',
      style: { color: '#ea580c', weight: 2, fillColor: '#ea580c', fillOpacity: 0.15, dashArray: '6 4' },
    }).addTo(map);
    const warningLayer = L.geoJSON(null, {
      pane: 'overlayPane',
      style: (feature) => projectSiteStyle(feature, {
        color: '#f59e0b', weight: 2, fillColor: '#f59e0b', fillOpacity: 0.14, dashArray: '4 4',
      }),
    }).addTo(map);

    // Assignment zones are derived from each assignment. They remain neutral:
    // legacy death records do not establish verified ecological survival.
    const siteZoneLayer = L.geoJSON(null, {
      pane: 'overlayPane',
      style: { color: '#475569', weight: 2, fillColor: '#475569', fillOpacity: 0.10 },
    });
    const projectSiteLayer = L.geoJSON(null, {
      // Share the points' canvas so Leaflet can hit-test both. A canvas in a
      // separate lower pane cannot receive events through the point canvas.
      pane: 'overlayPane',
      style: (feature) => projectSiteStyle(feature, PROJECT_SITE_STYLE),
    });

    const baseLayerControl = L.control.layers(
      { 'Google Satellite': googleSatellite, OpenStreetMap: osm },
      {
        'Drone Orthomosaic': orthophoto,
        'Planting Points': pointLayer,
        'Assignment Zones': siteZoneLayer,
        'Project Sites': projectSiteLayer,
        'Forbidden Zones': forbiddenLayer,
        'Eroded Zones': erodedLayer,
        'Warning Zones': warningLayer,
      },
      { position: 'topleft', collapsed: true }
    ).addTo(map);
    baseLayerControl.getContainer().dataset.guideLayers = 'true';
    const layerVisibilityControl = connectMapLayerVisibility(map, {
      orthophoto, points: pointLayer, siteZones: siteZoneLayer,
      projectSites: projectSiteLayer, forbidden: forbiddenLayer,
      eroded: erodedLayer, warnings: warningLayer,
    }, useMapStore);

    // Bottom-left legend so admins can read what each dashed colour means
    // without having to hover every polygon. Sits above the scale bar.
    const legendControl = L.control({ position: 'bottomleft' });
    legendControl.onAdd = () => {
      const div = L.DomUtil.create('div', 'map-zone-legend');
      div.innerHTML = `
        <div class="map-zone-legend-row">
          <span class="map-zone-legend-swatch map-zone-legend-swatch-project" aria-hidden="true"></span>
          <span class="map-zone-legend-label">Project site</span>
        </div>
        <div class="map-zone-legend-row">
          <span class="map-zone-legend-swatch map-zone-legend-swatch-site" aria-hidden="true"></span>
          <span class="map-zone-legend-label">Assignment zone</span>
        </div>
        <div class="map-zone-legend-row">
          <span class="map-zone-legend-swatch map-zone-legend-swatch-forbidden" aria-hidden="true"></span>
          <span class="map-zone-legend-label">Forbidden zone</span>
        </div>
        <div class="map-zone-legend-row">
          <span class="map-zone-legend-swatch map-zone-legend-swatch-eroded" aria-hidden="true"></span>
          <span class="map-zone-legend-label">Eroded zone</span>
        </div>
        <div class="map-zone-legend-row">
          <span class="map-zone-legend-swatch map-zone-legend-swatch-warning" aria-hidden="true"></span>
          <span class="map-zone-legend-label">Warning zone</span>
        </div>
        <div class="map-zone-legend-row">
          <span class="map-zone-legend-swatch map-zone-legend-swatch-mortality" aria-hidden="true"></span>
          <span class="map-zone-legend-label">High mortality (≥30%)</span>
        </div>
      `;
      // Stop map drag/zoom from firing when the user interacts with the legend
      L.DomEvent.disableClickPropagation(div);
      L.DomEvent.disableScrollPropagation(div);
      return div;
    };
    legendControl.addTo(map);
    const processingLayer = L.layerGroup().addTo(map);
    const processingFootprintLayer = L.layerGroup().addTo(map);
    const processingCenterLayer = L.layerGroup().addTo(map);
    const processingFilteredLayer = L.layerGroup().addTo(map);

    // Store refs
    layersRef.current = {
      baseLayerControl,
      layerVisibilityControl,
      orthophoto,
      pointLayer,
      forbiddenLayer,
      erodedLayer,
      warningLayer,
      siteZoneLayer,
      projectSiteLayer,
      processingLayer,
      processingFootprintLayer,
      processingCenterLayer,
      processingFilteredLayer,
    };
    mapRef.current = map;
    useMapStore.getState().setMapInstance(map);

    let focusedZone = null;
    map.on('zonefocus', ({ feature, kind }) => {
      const key = { site: 'projectSiteLayer', eroded: 'erodedLayer', warning: 'warningLayer', forbidden: 'forbiddenLayer' }[kind];
      const group = layersRef.current[key];
      const layer = group?.getLayers().find((candidate) => candidate.feature === feature);
      const bounds = layer?.getBounds();
      if (!bounds?.isValid()) return;

      if (focusedZone) focusedZone.group.resetStyle(focusedZone.layer);
      focusedZone = { group, layer };
      const visibilityKey = { site: 'projectSites', eroded: 'eroded', warning: 'warnings', forbidden: 'forbidden' }[kind];
      useMapStore.getState().showLayers([visibilityKey]);
      if (!map.hasLayer(group)) group.addTo(map);
      layer.setStyle({ weight: 4, fillOpacity: projectSiteStyle(feature, { fillOpacity: 0.22 }).fillOpacity });
      orderZoneLayers(layersRef.current);
      focusMapArea(map, bounds);
      layer.openTooltip(bounds.getCenter());
    });

    // Persist user pans + zooms back to the store so other map-bearing pages
    // inherit the same view when they mount.
    map.on('moveend', () => {
      if (skipNextSyncRef.current) {
        // This moveend was triggered by us applying a store change to the
        // map. Don't echo it back to the store.
        skipNextSyncRef.current = false;
        return;
      }
      const c = map.getCenter();
      useMapStore.getState().setView([c.lng, c.lat], map.getZoom());
    });

    // Rescale point markers when zoom changes so they don't clump when zoomed out
    map.on('zoomend', () => {
      const z = map.getZoom();
      const rFiltered = getPreviewFilteredRadius(z);
      if (layersRef.current.pointLayer) {
        layersRef.current.pointLayer.eachLayer((m) => {
          const selected = Boolean(m.options.mgAssignmentSelected);
          const warning = Boolean(m.options.mgWarning);
          if (typeof m.setStyle === 'function') {
            m.setStyle({
              color: selected ? '#7f1d1d' : (warning ? '#f59e0b' : '#000000'),
              weight: warning ? Math.max(getPointWeight(z, selected), z >= 20 ? 1.1 : 0.75) : getPointWeight(z, selected),
            });
          }
          if (typeof m.setRadius === 'function') m.setRadius(getPointRadius(z, selected));
        });
      }
      if (layersRef.current.processingLayer) {
        const rAnalysis = getAnalysisPointRadius(z);
        layersRef.current.processingLayer.eachLayer((m) => {
          if (typeof m.setStyle === 'function') m.setStyle({ weight: 0 });
          if (typeof m.setRadius === 'function') m.setRadius(rAnalysis);
        });
      }
      if (layersRef.current.processingFilteredLayer) {
        layersRef.current.processingFilteredLayer.eachLayer((m) => {
          if (typeof m.setRadius === 'function') m.setRadius(rFiltered);
        });
      }
    });

    // Resize handler
    const observer = new ResizeObserver(() => map.invalidateSize());
    observer.observe(containerRef.current);

    const { fetchPoints, fetchZones } = useMapStore.getState();
    fetchPoints();
    fetchZones();

    return () => {
      observer.disconnect();
      layerVisibilityControl.dispose();
      map.remove();
      mapRef.current = null;
      useMapStore.getState().setMapInstance(null);
    };
  }, []);

  // Leaflet adds its own classes to this element. Toggle the page class
  // directly so React does not replace those classes during route changes.
  useEffect(() => {
    containerRef.current?.classList.toggle('monitoring-map-container', isMonitoringMapMode);
  }, [isMonitoringMapMode]);

  // When the store's view changes, bring this map to the same view. The skip
  // ref + a rough equality check below prevent feedback loops with the
  // moveend handler above.
  useEffect(() => {
    const map = mapRef.current;
    if (!map) return;
    const c = map.getCenter();
    const sameLng = Math.abs(c.lng - center[0]) < 1e-7;
    const sameLat = Math.abs(c.lat - center[1]) < 1e-7;
    const sameZoom = map.getZoom() === zoom;
    if (sameLng && sameLat && sameZoom) return;
    skipNextSyncRef.current = true;
    map.setView([center[1], center[0]], zoom, { animate: false });
  }, [center, zoom]);

  // Sync layer visibility
  useEffect(() => {
    const map = mapRef.current;
    const layers = layersRef.current;
    if (!map || !layers.pointLayer) return;

    const showSharedZones = !isMonitoringMapMode || monitoringOrganizationId == null;
    layers.layerVisibilityControl.sync({
      ...layerVisibility,
      forbidden: layerVisibility.forbidden && showSharedZones,
      eroded: layerVisibility.eroded && showSharedZones,
      warnings: layerVisibility.warnings && showSharedZones,
    });
    orderZoneLayers(layers);
  }, [layerVisibility, isMonitoringMapMode, monitoringOrganizationId]);

  // Sync points — skip the redraw entirely when only unrelated store fields
  // changed (e.g. stats refresh after a tab switch with no actual point data change).
  useEffect(() => {
    const map = mapRef.current;
    const layer = layersRef.current.pointLayer;
    if (!map || !layer) return;

    // Skip rebuilding markers only when both the complete dataset and the
    // selection/filter context match the last render.
    const assignmentSelectedIds = new Set(assignmentSelectedPointIds.map(Number));
    const modeKey = isMonitoringMapMode
      ? `monitor:${showDeadPointsOnly}:${monitoringOrganizationId ?? 'all'}`
      : (isPlanterMode ? 'planters' : isAnalyticsMode ? 'analytics' : 'normal');
    const scopedPoints = isMonitoringMapMode
      ? filterMonitoringPoints(spacedPoints, monitoringOrganizationId)
      : spacedPoints;
    const displayPoints = filterMapPoints(scopedPoints, {
      deadOnly: isMonitoringMapMode && showDeadPointsOnly,
    }).filter((point) => point.inside_visible_map !== false);
    const assignmentScopeKey = isPlanterMode
      ? `${assignmentOrganizationId ?? ''}:${assignmentProjectSiteId ?? ''}`
      : '';
    const fingerprint = `${modeKey}:${assignmentScopeKey}:${pointsDataKey}`;

    if (fingerprint === pointsFingerprintRef.current) return;
    pointsFingerprintRef.current = fingerprint;

    layer.clearLayers();

    // Popup status badge background. completed / planted now use the same
    // amber tint so the point's badge matches the yellow marker.
    const statusBg = {
      planned: '#dcfce7',
      assigned: '#dbeafe',
      planted: '#fef3c7',
      completed: '#fef3c7',
      skipped: '#f1f5f9',
      dead: '#fecaca',
      eroded_unavailable: '#ffedd5',
    };

    const currentReplantingSelection = new Set(useMapStore.getState().replantingSelectedPointIds);
    displayPoints.forEach((p) => {
      const status = getPointDisplayStatus(p);
      const statusLabel = formatStatusLabel(status);
      const pointId = Number(p.id);
      const isAssignmentSelected = isPlanterMode && assignmentSelectedIds.has(pointId);
      const isReplantingSelected = isMonitoringMapMode && currentReplantingSelection.has(p.id);
      const belongsToAnotherOrganization = isPlanterMode
        && assignmentOrganizationId != null
        && p.source_organization_id != null
        && Number(p.source_organization_id) !== Number(assignmentOrganizationId);
      const belongsToAnotherProjectSite = isPlanterMode
        && (assignmentProjectSiteId == null ? p.source_project_site_id != null : (
          p.source_project_site_id == null || Number(p.source_project_site_id) !== Number(assignmentProjectSiteId)
        ));
      const outsideAssignmentScope = belongsToAnotherOrganization || belongsToAnotherProjectSite;
      const hasWarning = Boolean(p.survival_warning);
      // Negative zone IDs identify automatic site mortality warnings. Keep
      // their points' ordinary outlines; the amber site boundary carries it.
      const mortalityWarningOnly = p.warning_zone_ids?.length > 0
        && p.warning_zone_ids.every((id) => Number(id) < 0);
      const showWarningRing = hasWarning && !mortalityWarningOnly;

      // In Planter Management, override the single "assigned = blue" with a
      // organization hue so the admin can identify the owner without hovering.
      const usePlanterColor = isPlanterMode && status === 'assigned' && p.assigned_planter_id != null;
      // Planned (not-yet-assigned) points are colored by the species the
      // analysis recorded — rhizophora magenta, bungalon green. Status colors
      // still win for assigned / planted / dead etc., so the planting
      // lifecycle stays visible after a point leaves the 'planned' bucket.
      const speciesKey = typeof p.species === 'string' ? p.species.toLowerCase() : null;
      const speciesColor = status === 'planned' && speciesKey && SPECIES_COLORS[speciesKey]
        ? SPECIES_COLORS[speciesKey]
        : null;
      const baseStatusColor = usePlanterColor
        ? getPlanterColor(p.assigned_organization_id)
        : (speciesColor || STATUS_COLORS[status] || STATUS_COLORS.planned);
      const color = isAssignmentSelected || isReplantingSelected ? '#7c3aed' : baseStatusColor;
      const badgeBg = isAssignmentSelected || isReplantingSelected
        ? '#ede9fe'
        : (usePlanterColor ? getPlanterTint(p.assigned_organization_id) : (statusBg[status] || '#f3f4f6'));

      const z = map.getZoom();
      const highlight = isAssignmentSelected || isReplantingSelected;
      const markerStroke = highlight
        ? '#4c1d95'
        : showWarningRing
          ? '#f59e0b'
          : '#000000';
      const markerWeight = showWarningRing
        ? Math.max(getPointWeight(z, highlight), z >= 20 ? 1.1 : 0.75)
        : getPointWeight(z, highlight);
      const marker = L.circleMarker([p.latitude, p.longitude], {
        radius: getPointRadius(z, highlight),
        fillColor: color,
        color: markerStroke,
        weight: markerWeight,
        fillOpacity: outsideAssignmentScope ? 0.2 : (highlight ? 0.96 : 0.92),
        opacity: 0.95,
        mgAssignmentSelected: highlight,
        mgWarning: showWarningRing,
        mgDead: status === 'dead',
        mgPointId: p.id,
        mgBaseColor: baseStatusColor,
        mgBaseStroke: showWarningRing ? '#f59e0b' : '#000000',
        mgOutsideAssignmentScope: outsideAssignmentScope,
      });

      const pointTitle = escapeHtml(p.point_num);
      const imageName = escapeHtml(p.image_name || '');
      const planterName = escapeHtml(p.planted_by_user_id
        ? `LGU${p.planted_by_name ? ` (${p.planted_by_name})` : ''}`
        : p.assigned_planter_name || '');
      const safeStatusLabel = escapeHtml(statusLabel);
      const isDeadPoint = status === 'dead';
      const deathReason = escapeHtml(p.death_reason || 'No reason recorded');
      const deathAt = escapeHtml(p.death_at ? String(p.death_at).slice(0, 19).replace('T', ' ') : '');
      const warningSummary = escapeHtml(p.warning_summary || 'Planner warning');
      const warningSeverity = escapeHtml(p.warning_severity || 'medium');
      const tooltipWarning = hasWarning
        ? `<div class="point-status-tooltip-detail point-status-tooltip-warning">Warning: ${warningSummary}</div>`
        : '';
      // Species hint — visible on every point that knows its species, so the
      // planner can hover any green/yellow dot and see "Bungalon (1 m)" or
      // "Rhizophora (2 m)" without opening the popup.
      const speciesLabel = speciesKey ? SPECIES_LABELS[speciesKey] : '';
      const tooltipSpecies = speciesLabel
        ? `<div class="point-status-tooltip-detail" style="text-transform:capitalize;">${escapeHtml(speciesLabel)}</div>`
        : '';
      const tooltipDetail = isDeadPoint
        ? `<div class="point-status-tooltip-detail">Cause: ${deathReason}</div>`
        : planterName
          ? `<div class="point-status-tooltip-detail">${planterName}</div>`
          : '';
      const tooltipSelected = isAssignmentSelected
        ? '<div class="point-status-tooltip-detail point-status-tooltip-selected">Selected for assignment</div>'
        : '';
      const assignmentScopeWarning = outsideAssignmentScope
        ? `<div class="point-status-tooltip-detail point-status-tooltip-warning">Not assignable to the selected organization: this point belongs to ${escapeHtml(p.source_organization_name || p.source_project_site_name || 'outside the selected project site')}.</div>`
        : '';
      const deathDetail = isDeadPoint
        ? `
          <div style="margin-top:8px;padding:8px;border-radius:8px;background:${badgeBg};border-left:4px solid ${baseStatusColor};">
            <div style="font-size:11px;font-weight:800;color:${baseStatusColor};text-transform:uppercase;letter-spacing:.03em;">Dead</div>
            <div style="font-size:12px;color:#374151;margin-top:3px;"><strong>Cause:</strong> ${deathReason}</div>
            ${deathAt ? `<div style="font-size:11px;color:#6b7280;margin-top:3px;">${deathAt}</div>` : ''}
          </div>
        `
        : '';
      const warningDetail = hasWarning
        ? `
          <div style="margin-top:8px;padding:8px;border-radius:8px;background:#fffbeb;border-left:4px solid #f59e0b;">
            <div style="font-size:11px;font-weight:800;color:#92400e;text-transform:uppercase;letter-spacing:.03em;">Warning (${warningSeverity})</div>
            <div style="font-size:12px;color:#374151;margin-top:3px;">${warningSummary}</div>
          </div>
        `
        : '';
      const monitorHint = isMonitoringMapMode && isDeadPoint
        ? '<div style="margin-top:8px;padding:6px 8px;border-radius:6px;background:#fff7ed;border:1px dashed #fdba74;font-size:11px;color:#9a3412;font-weight:600;">Click to select this dead point for release from its project site.</div>'
        : '';

      marker.bindTooltip(`
        <div class="point-status-tooltip-inner">
          <div class="point-status-tooltip-title">Point #${pointTitle}</div>
          <div class="point-status-tooltip-status" style="border-left-color:${baseStatusColor};">
            ${safeStatusLabel}
          </div>
          ${tooltipSpecies}
          ${tooltipDetail}
          ${tooltipWarning}
          ${tooltipSelected}
          ${assignmentScopeWarning}
        </div>
      `, {
        className: 'point-status-tooltip',
        direction: 'top',
        opacity: 0.98,
        sticky: true,
      });

      marker.bindPopup(`
        <div style="font-family:'Inter',sans-serif;min-width:180px;">
          <div style="font-weight:700;font-size:14px;margin-bottom:4px;">Point #${pointTitle}</div>
          <div style="font-size:12px;color:#6b7280;margin-bottom:8px;">${imageName}</div>
          <div style="display:flex;gap:6px;align-items:center;flex-wrap:wrap;">
            <span style="font-size:10px;padding:2px 8px;border-radius:99px;background:${badgeBg};font-weight:700;text-transform:uppercase;color:${isDeadPoint ? '#111827' : 'inherit'};">${statusLabel}</span>
            ${planterName ? `<span style="font-size:12px;color:#6b7280;">${planterName}</span>` : ''}
          </div>
          ${speciesLabel ? `<div style="margin-top:8px;padding:6px 8px;border-radius:6px;background:${speciesKey === 'rhizophora' ? '#fce7f3' : '#dcfce7'};border-left:4px solid ${speciesKey === 'rhizophora' ? '#db2777' : '#16a34a'};font-size:12px;color:#374151;"><strong>${escapeHtml(speciesLabel)}</strong></div>` : ''}
          ${deathDetail}
          ${warningDetail}
          ${monitorHint}
          <div style="font-size:11px;color:#9ca3af;margin-top:8px;font-family:monospace;">${p.latitude.toFixed(7)}, ${p.longitude.toFixed(7)}</div>
        </div>
      `, { maxWidth: 260 });

      marker.on('click', () => {
        if (isMonitoringMapMode && isDeadPoint) {
          map.closePopup();
          if (useMapStore.getState().replantingSelectionMode === 'click') toggleReplantingPoint(p.id);
          return;
        }
        // The assignment selector handles eligible point clicks and brushing
        // in its own pane. These base markers retain point details.
        if (isPlanterMode) return;
        setSelectedPoint(p.id);
      });
      marker.addTo(layer);
    });

    // preserved across every tab switch. maxZoom: 22 ≈ 5 m scale.
  }, [
    spacedPoints,
    pointsDataKey,
    setSelectedPoint,
    isPlanterMode,
    isMonitoringMapMode,
    isAnalyticsMode,
    showDeadPointsOnly,
    toggleReplantingPoint,
    assignmentSelectedPointIds,
    assignmentOrganizationId,
    assignmentProjectSiteId,
    monitoringOrganizationId,
  ]);

  // Recolor changed selections in place, keeping brush movement smooth.
  useEffect(() => {
    if ((!isMonitoringMapMode && !isPlanterMode) || !mapRef.current) return;
    const selected = new Set((isPlanterMode ? assignmentSelectedPointIds : replantingSelectedPointIds).map(Number));
    const z = mapRef.current.getZoom();
    layersRef.current.pointLayer?.eachLayer((marker) => {
      if (isMonitoringMapMode && !marker.options.mgDead) return;
      const chosen = selected.has(Number(marker.options.mgPointId));
      if (marker.options.mgAssignmentSelected === chosen) return;
      marker.options.mgAssignmentSelected = chosen;
      marker.setRadius(getPointRadius(z, chosen));
      marker.setStyle({
        fillColor: chosen ? '#7c3aed' : marker.options.mgBaseColor,
        color: chosen ? '#4c1d95' : marker.options.mgBaseStroke,
        fillOpacity: marker.options.mgOutsideAssignmentScope ? 0.2 : chosen ? 0.96 : 0.92,
        weight: marker.options.mgWarning ? Math.max(getPointWeight(z, chosen), z >= 20 ? 1.1 : 0.75) : getPointWeight(z, chosen),
      });
    });
  }, [replantingSelectedPointIds, assignmentSelectedPointIds, isMonitoringMapMode, isPlanterMode, points,
    showDeadPointsOnly, assignmentOrganizationId, assignmentProjectSiteId]);

  // Brush selection uses screen-space sweeps, independent of marker redraws.
  useEffect(() => {
    const map = mapRef.current;
    if (!map || !isMonitoringMapMode || replantingSelectionMode === 'click' || !layerVisibility.points) return;
    const container = map.getContainer();
    const brush = document.createElement('div');
    brush.className = `replanting-brush is-${replantingSelectionMode}`;
    brush.style.width = brush.style.height = `${REPLANTING_BRUSH_RADIUS * 2}px`;
    brush.setAttribute('aria-hidden', 'true');
    container.appendChild(brush);
    container.classList.add('is-brushing');
    const previousTouchAction = container.style.touchAction;
    container.style.touchAction = 'none';
    const handlers = [map.dragging, map.doubleClickZoom, map.boxZoom, map.touchZoom].filter((handler) => handler?.enabled());
    handlers.forEach((handler) => handler.disable());
    let previous = null;
    let moving = false;
    let activePointer = null;
    let frame = null;
    const pending = new Set();
    const eligible = filterMonitoringPoints(filterCrossSpeciesDisplayPoints(points), monitoringOrganizationId)
      .filter((point) => point.death_at);
    let projected = [];
    const project = () => {
      projected = eligible.map((point) => ({ id: point.id, ...map.latLngToContainerPoint([point.latitude, point.longitude]) }));
      previous = null;
    };
    project();
    const flush = () => {
      frame = null;
      const ids = [...pending];
      pending.clear();
      useMapStore.getState().paintReplantingPoints(ids, replantingSelectionMode);
    };
    const reset = () => { previous = null; brush.style.display = 'none'; };
    const hasOpenDialog = () => [...document.querySelectorAll('[role="dialog"]')].some((dialog) => {
      if (dialog.closest('[hidden], [inert], [aria-hidden="true"]')) return false;
      // Retained pages can also hide an ancestor with display:none while
      // keeping the dialog's DOM mounted for a later return.
      for (let node = dialog; node; node = node.parentElement) {
        const style = window.getComputedStyle(node);
        if (style.display === 'none' || style.visibility === 'hidden' || style.visibility === 'collapse') return false;
      }
      return true;
    });
    const endStroke = () => {
      const pointerId = activePointer;
      activePointer = null;
      if (frame !== null) cancelAnimationFrame(frame);
      frame = null;
      if (pending.size) flush();
      if (pointerId !== null && container.hasPointerCapture?.(pointerId)) container.releasePointerCapture(pointerId);
      reset();
    };
    const sweep = (event) => {
      const target = document.elementFromPoint?.(event.clientX, event.clientY) || event.target;
      if (moving || !container.contains(target) || target.closest?.('.leaflet-control') || hasOpenDialog()) {
        endStroke(); return;
      }
      const position = map.mouseEventToContainerPoint(event);
      const size = map.getSize();
      if (position.x < 0 || position.y < 0 || position.x > size.x || position.y > size.y) { endStroke(); return; }
      brush.style.display = 'block';
      brush.style.left = `${position.x}px`;
      brush.style.top = `${position.y}px`;
      // Hover only previews the brush. A stroke must start on this map, and
      // mouse/pen movement must retain the primary button throughout.
      if (activePointer === null || event.pointerId !== activePointer) { previous = null; return; }
      if (event.pointerType !== 'touch' && (event.buttons & 1) === 0) { endStroke(); return; }
      pointsAlongBrush(projected, previous || position, position).forEach((id) => pending.add(id));
      previous = position;
      if (pending.size && frame === null) frame = requestAnimationFrame(flush);
    };
    const down = (event) => {
      if (moving || activePointer !== null || event.isPrimary === false || event.button !== 0
          || event.target.closest?.('.leaflet-control') || hasOpenDialog()) return;
      previous = null;
      activePointer = event.pointerId;
      container.setPointerCapture?.(event.pointerId);
      event.preventDefault();
      sweep(event);
    };
    const up = (event) => {
      if (activePointer === event.pointerId) endStroke();
    };
    const escape = (event) => {
      if (event.key === 'Escape') useMapStore.getState().setReplantingSelectionMode('click');
    };
    const movingStart = () => { moving = true; endStroke(); };
    const movingEnd = () => { moving = false; project(); };
    map.closePopup();
    map.on('movestart zoomstart', movingStart);
    map.on('moveend zoomend resize', movingEnd);
    container.addEventListener('pointermove', sweep);
    container.addEventListener('pointerdown', down);
    window.addEventListener('pointerup', up);
    window.addEventListener('pointercancel', up);
    container.addEventListener('lostpointercapture', up);
    container.addEventListener('pointerleave', endStroke);
    document.addEventListener('keydown', escape);
    window.addEventListener('blur', endStroke);
    return () => {
      if (frame !== null) cancelAnimationFrame(frame);
      pending.clear();
      map.off('movestart zoomstart', movingStart);
      map.off('moveend zoomend resize', movingEnd);
      container.removeEventListener('pointermove', sweep);
      container.removeEventListener('pointerdown', down);
      window.removeEventListener('pointerup', up);
      window.removeEventListener('pointercancel', up);
      container.removeEventListener('lostpointercapture', up);
      container.removeEventListener('pointerleave', endStroke);
      document.removeEventListener('keydown', escape);
      window.removeEventListener('blur', endStroke);
      if (activePointer !== null && container.hasPointerCapture?.(activePointer)) container.releasePointerCapture(activePointer);
      handlers.forEach((handler) => handler.enable());
      container.style.touchAction = previousTouchAction;
      container.classList.remove('is-brushing');
      brush.remove();
    };
  }, [isMonitoringMapMode, replantingSelectionMode, layerVisibility.points, points, monitoringOrganizationId]);

  // Sync zones
  useEffect(() => {
    for (const [key, collection, kind] of [
      ['forbiddenLayer', forbiddenZones, 'forbidden'],
      ['erodedLayer', erodedZones, 'eroded'],
      ['warningLayer', warningZones, 'warning'],
    ]) {
      const group = layersRef.current[key];
      if (!group) continue;
      group.clearLayers();
      (collection?.features || []).forEach((feature, index) => {
        try {
          const automaticMortality = kind === 'warning' && feature.properties?.automatic
            && feature.properties.warning_type === 'high_mortality';
          const parentSite = automaticMortality ? projectSites?.features?.find((site) =>
            String(projectSiteId(site)) === String(feature.properties.project_site_id)) : null;
          const layer = L.geoJSON(feature, { ...group.options, interactive: !automaticMortality || Boolean(parentSite) });
          layer.eachLayer((boundary) => {
            if (parentSite) {
              // Automatic mortality belongs to the project's existing card;
              // do not open a second, detailed warning-zone popup over it.
              bindProjectSiteInfo(boundary, parentSite, {
                counts: pointsLoaded
                  ? (sitePointCounts.get(String(projectSiteId(parentSite))) || emptyProjectSiteCounts())
                  : null,
              });
            } else if (!automaticMortality) bindZoneInfo(boundary, feature, index, kind);
            group.addLayer(boundary);
          });
        } catch { /* Skip a malformed boundary without changing list indexes. */ }
      });
    }
    orderZoneLayers(layersRef.current);
  }, [forbiddenZones, erodedZones, warningZones, projectSites, sitePointCounts, pointsLoaded]);

  // Sync assignment zones. Legacy current/death counts are operational
  // context only, so polygons use a neutral style and never claim survival.
  useEffect(() => {
    const { siteZoneLayer } = layersRef.current;
    if (!siteZoneLayer) return;
    siteZoneLayer.clearLayers();
    if (!siteZones || !Array.isArray(siteZones.features)) return;

    const mortalityById = new Map(
      (siteZoneMortality || []).map((row) => [row.id, row]),
    );

    const visibleSiteZones = isMonitoringMapMode
      ? filterMonitoringFeatures(siteZones, monitoringOrganizationId)
      : siteZones;
    visibleSiteZones.features.forEach((feature) => {
      const props = feature?.properties || {};
      const stats = mortalityById.get(props.id) || mortalityById.get(feature.id);
      const total = stats?.total ?? 0;
      const currentPlanted = stats?.alive ?? 0;
      const recordedDeaths = stats?.dead ?? 0;
      const fillColor = '#64748b';

      const layer = L.geoJSON(feature, {
        pane: 'overlayPane',
        style: {
          color: fillColor,
          weight: 2,
          fillColor,
          fillOpacity: 0.07,
        },
      });

      const safeName = escapeHtml(props.organization_name || props.planter_name || props.name || 'Assignment zone');
      const summaryHtml = total > 0
        ? `<div class="site-zone-tooltip-stat">
             <strong>${currentPlanted}</strong> currently planted without a death record
             <span class="site-zone-tooltip-meta">
               (${recordedDeaths} recorded deaths · ${total} planted records; not verified survival)
             </span>
           </div>`
        : `<div class="site-zone-tooltip-stat site-zone-tooltip-empty">
             No planted points in this zone yet
           </div>`;

      layer.bindTooltip(
        `<div class="site-zone-tooltip-inner">
           <div class="site-zone-tooltip-title">${safeName}</div>
           ${summaryHtml}
         </div>`,
        { className: 'site-zone-tooltip', sticky: true, direction: 'top' },
      );

      layer.addTo(siteZoneLayer);
    });
    orderZoneLayers(layersRef.current);
  }, [siteZones, siteZoneMortality, isMonitoringMapMode, monitoringOrganizationId]);

  // Stable project sites are parent boundaries; assignment zones continue to
  // render independently inside them for batch-level drill-downs.
  useEffect(() => {
    const { projectSiteLayer } = layersRef.current;
    if (!projectSiteLayer) return;
    projectSiteLayer.clearLayers();
    if (!projectSites || !Array.isArray(projectSites.features)) return;
    const visibleProjectSites = isMonitoringMapMode
      ? filterMonitoringFeatures(projectSites, monitoringOrganizationId)
      : projectSites;
    for (const feature of visibleProjectSites.features) {
      try { projectSiteLayer.addData(feature); } catch { /* skip malformed site */ }
    }
    projectSiteLayer.getLayers().forEach((layer, index) => {
      bindProjectSiteInfo(layer, { ...layer.feature, properties: {
        ...layer.feature.properties, name: zoneName(layer.feature, index, 'site'),
      } }, {
        map: mapRef.current, createTooltip: L.tooltip,
        counts: pointsLoaded
          ? (sitePointCounts.get(String(projectSiteId(layer.feature))) || emptyProjectSiteCounts())
          : null,
      });
    });
    orderZoneLayers(layersRef.current);
  }, [projectSites, sitePointCounts, pointsLoaded, isMonitoringMapMode, monitoringOrganizationId]);

  // Show mapped points and the photo boundary on the unobstructed basemap.
  useEffect(() => {
    const map = mapRef.current;
    const {
      processingLayer,
      processingFootprintLayer,
      processingCenterLayer,
      processingFilteredLayer,
    } = layersRef.current;
    if (
      !map ||
      !processingLayer ||
      !processingFootprintLayer ||
      !processingCenterLayer ||
      !processingFilteredLayer
    ) return;

    processingLayer.clearLayers();
    processingFootprintLayer.clearLayers();
    processingCenterLayer.clearLayers();
    processingFilteredLayer.clearLayers();

    if (!currentAnalysis?.map?.available) return;

    const analysisFootprint = currentAnalysis.map.analysis_footprint;
    const locationStatus = currentAnalysis.map.location_status || 'inside';
    const safeFeatures = currentAnalysis.map.safe_points_geojson?.features || [];
    const forbiddenFeatures = currentAnalysis.map.forbidden_filtered_geojson?.features || [];
    const postSnapDangerFeatures = currentAnalysis.map.post_snap_danger_filtered_geojson?.features || [];
    const orthophotoCanopyFeatures = currentAnalysis.map.orthophoto_canopy_filtered_geojson?.features || [];
    const spacingFilteredFeatures = currentAnalysis.map.spacing_filtered_geojson?.features || [];
    const centerFeature = currentAnalysis.map.image_center_feature;

    if (analysisFootprint?.type === 'Polygon' || analysisFootprint?.type === 'MultiPolygon') {
      const footprintColor = locationStatus === 'outside'
        ? '#dc2626'
        : locationStatus === 'partial'
          ? '#d97706'
          : '#16a34a';
      const footprintLabel = locationStatus === 'outside'
        ? 'Selected image is outside the GIS map'
        : locationStatus === 'partial'
          ? 'Selected image partially overlaps the GIS map'
          : hasEstimatedAlignment(currentAnalysis.map.match)
            ? 'Projected photo boundary (approximate alignment; not a stitching seam)'
            : 'Projected photo boundary';
      L.geoJSON(
        {
          type: 'Feature',
          properties: { location_status: locationStatus },
          geometry: analysisFootprint,
        },
        {
          style: {
            color: footprintColor,
            weight: 3,
            fill: !currentAnalysis.saved,
            fillColor: footprintColor,
            fillOpacity: currentAnalysis.saved ? 0 : 0.12,
            dashArray: locationStatus === 'inside' ? null : '7 5',
          },
          onEachFeature: (_feature, layer) => {
            layer.bindTooltip(footprintLabel, {
              className: 'zone-tooltip',
              sticky: true,
              direction: 'top',
            });
          },
        },
      ).addTo(processingFootprintLayer);
    }

    safeFeatures.forEach((feature) => {
      const [lon, lat] = feature.geometry.coordinates;
      const props = feature.properties || {};
      const erodedUnavailable = Boolean(props.eroded_unavailable);
      const previewColor = erodedUnavailable
        ? PREVIEW_COLORS.erodedUnavailable
        : PREVIEW_COLORS.safe;
      const previewStatus = erodedUnavailable
        ? pointStatusLabel('unavailable')
        : pointStatusLabel('planned');
      L.circleMarker([lat, lon], {
        radius: getAnalysisPointRadius(map.getZoom()),
        color: '#000000',
        weight: 1,
        fillColor: previewColor,
        fillOpacity: 0.85,
      })
        .bindPopup(`
          <div style="font-family:'Inter',sans-serif;min-width:180px;">
            <div style="font-weight:700;font-size:14px;margin-bottom:6px;">${props.name || 'Planting Point'}</div>
            <div style="font-size:12px;color:#4b5563;margin-bottom:6px;">Unsaved analysis preview</div>
            <div style="font-size:12px;font-weight:700;color:${previewColor};margin-bottom:6px;">${previewStatus}</div>
            <div style="font-size:12px;color:#111827;">${lat.toFixed(7)}, ${lon.toFixed(7)}</div>
            <div style="font-size:12px;color:#6b7280;margin-top:6px;">Buffer: ${props.buffer_m ?? '-'} m</div>
            <div style="font-size:12px;color:#6b7280;">Area: ${props.area_m2 ?? '-'} m²</div>
          </div>
        `)
        .addTo(processingLayer);
    });

    forbiddenFeatures.forEach((feature) => {
      const [lon, lat] = feature.geometry.coordinates;
      const props = feature.properties || {};
      L.circleMarker([lat, lon], {
        radius: getPreviewFilteredRadius(map.getZoom()),
        color: PREVIEW_COLORS.forbidden,
        weight: 2,
        fillColor: '#fecaca',
        fillOpacity: 0.9,
      })
        .bindPopup(`
          <div style="font-family:'Inter',sans-serif;min-width:170px;">
            <div style="font-weight:700;font-size:14px;margin-bottom:6px;color:${PREVIEW_COLORS.forbidden};">Filtered Preview Point</div>
            <div style="font-size:12px;color:#111827;">${lat.toFixed(7)}, ${lon.toFixed(7)}</div>
            <div style="font-size:12px;color:#6b7280;margin-top:6px;">${props.reason || 'Inside forbidden zone'}</div>
          </div>
        `)
        .addTo(processingFilteredLayer);
    });

    postSnapDangerFeatures.forEach((feature) => {
      const [lon, lat] = feature.geometry.coordinates;
      const props = feature.properties || {};
      L.circleMarker([lat, lon], {
        radius: getPreviewFilteredRadius(map.getZoom()),
        color: PREVIEW_COLORS.danger,
        weight: 2,
        fillColor: '#fee2e2',
        fillOpacity: 0.9,
      })
        .bindPopup(`
          <div style="font-family:'Inter',sans-serif;min-width:170px;">
            <div style="font-weight:700;font-size:14px;margin-bottom:6px;color:${PREVIEW_COLORS.danger};">Filtered Preview Point</div>
            <div style="font-size:12px;color:#111827;">${lat.toFixed(7)}, ${lon.toFixed(7)}</div>
            <div style="font-size:12px;color:#6b7280;margin-top:6px;">${props.reason || 'Final snapped point too close to danger buffer'}</div>
          </div>
        `)
        .addTo(processingFilteredLayer);
    });

    orthophotoCanopyFeatures.forEach((feature) => {
      const [lon, lat] = feature.geometry.coordinates;
      const props = feature.properties || {};
      L.circleMarker([lat, lon], {
        radius: getPreviewFilteredRadius(map.getZoom()),
        color: PREVIEW_COLORS.canopy,
        weight: 2,
        fillColor: '#ddd6fe',
        fillOpacity: 0.9,
      })
        .bindPopup(`
          <div style="font-family:'Inter',sans-serif;min-width:170px;">
            <div style="font-weight:700;font-size:14px;margin-bottom:6px;color:${PREVIEW_COLORS.canopy};">Filtered Preview Point</div>
            <div style="font-size:12px;color:#111827;">${lat.toFixed(7)}, ${lon.toFixed(7)}</div>
            <div style="font-size:12px;color:#6b7280;margin-top:6px;">${props.reason || 'Overlaps canopy in orthophoto recheck'}</div>
          </div>
        `)
        .addTo(processingFilteredLayer);
    });

    spacingFilteredFeatures.forEach((feature) => {
      const [lon, lat] = feature.geometry.coordinates;
      const props = feature.properties || {};
      L.circleMarker([lat, lon], {
        radius: getPreviewFilteredRadius(map.getZoom()),
        color: PREVIEW_COLORS.spacing,
        weight: 2,
        fillColor: '#bfdbfe',
        fillOpacity: 0.9,
      })
        .bindPopup(`
          <div style="font-family:'Inter',sans-serif;min-width:170px;">
            <div style="font-weight:700;font-size:14px;margin-bottom:6px;color:${PREVIEW_COLORS.spacing};">Filtered Preview Point</div>
            <div style="font-size:12px;color:#111827;">${lat.toFixed(7)}, ${lon.toFixed(7)}</div>
            <div style="font-size:12px;color:#6b7280;margin-top:6px;">${props.reason || 'Too close to another planting point'}</div>
          </div>
        `)
        .addTo(processingFilteredLayer);
    });

    if (centerFeature?.geometry?.coordinates
      && (!centerFeature.properties?.outside_map_bounds || currentAnalysis.preflight)) {
      const [lon, lat] = centerFeature.geometry.coordinates;
      const props = centerFeature.properties || {};
      L.marker([lat, lon])
        .bindPopup(`
          <div style="font-family:'Inter',sans-serif;min-width:180px;">
            <div style="font-weight:700;font-size:14px;margin-bottom:6px;">${props.name || 'Image Center'}</div>
            <div style="font-size:12px;color:#111827;">${lat.toFixed(7)}, ${lon.toFixed(7)}</div>
            <div style="font-size:12px;color:#6b7280;margin-top:6px;">Altitude: ${props.altitude_m ?? '-'} m</div>
            <div style="font-size:12px;color:#6b7280;">Heading: ${props.heading ?? '-'}°</div>
          </div>
        `)
        .addTo(processingCenterLayer);
    }

    const previewLatLngs = safeFeatures
      .concat(forbiddenFeatures)
      .concat(postSnapDangerFeatures)
      .concat(orthophotoCanopyFeatures)
      .concat(spacingFilteredFeatures)
      .map((feature) => {
        const [lon, lat] = feature.geometry.coordinates;
        return [lat, lon];
      });

    const footprintCoordinates = analysisFootprint?.type === 'Polygon'
      ? analysisFootprint.coordinates?.[0]
      : analysisFootprint?.type === 'MultiPolygon'
        ? analysisFootprint.coordinates?.flatMap((polygon) => polygon?.[0] || [])
        : [];
    (footprintCoordinates || []).forEach((coordinate) => {
      const [lon, lat] = coordinate || [];
      if (Number.isFinite(Number(lat)) && Number.isFinite(Number(lon))) {
        previewLatLngs.push([Number(lat), Number(lon)]);
      }
    });

    if (centerFeature?.geometry?.coordinates
      && (!centerFeature.properties?.outside_map_bounds || currentAnalysis.preflight)) {
      const [lon, lat] = centerFeature.geometry.coordinates;
      previewLatLngs.push([lat, lon]);
    }

    if (previewLatLngs.length > 0 && !currentAnalysis.preserveMapView) {
      map.fitBounds(L.latLngBounds(previewLatLngs), {
        paddingTopLeft: [60, 60],
        paddingBottomRight: [map.getSize().x >= 900 ? 450 : 36, 60],
        maxZoom: 20,
        animate: true,
      });
    }
  }, [currentAnalysis]);

  useEffect(() => {
    const map = mapRef.current;
    if (!map || !isPlanterMode || assignmentProjectSiteId == null) return;
    const site = projectSites?.features?.find((feature) => Number(feature.id ?? feature.properties?.id) === Number(assignmentProjectSiteId));
    if (!site?.geometry) return;
    const bounds = L.geoJSON(site).getBounds();
    if (!bounds.isValid()) return;
    showLayers(['points', 'projectSites']);
    skipNextSyncRef.current = true;
    map.fitBounds(bounds, { paddingTopLeft: [60, 60], paddingBottomRight: [map.getSize().x >= 900 ? 450 : 36, 60], maxZoom: 20, animate: true });
  }, [assignmentProjectSiteId, isPlanterMode, projectSites, showLayers]);

  useEffect(() => {
    if (!isMonitoringMapMode) {
      focusedMonitoringOrganizationRef.current = '';
      return;
    }
    const map = mapRef.current;
    const features = projectSites?.features;
    if (!map || !Array.isArray(features) || !features.length) return;
    const focusKey = monitoringOrganizationId == null ? 'all' : String(monitoringOrganizationId);
    if (focusKey === focusedMonitoringOrganizationRef.current) return;
    if (focusKey === 'all' && !focusedMonitoringOrganizationRef.current) return;
    const visibleFeatures = filterMonitoringFeatures(projectSites, monitoringOrganizationId).features;
    if (!visibleFeatures.length) return;
    const bounds = L.geoJSON({ type: 'FeatureCollection', features: visibleFeatures }).getBounds();
    if (!bounds.isValid()) return;
    focusedMonitoringOrganizationRef.current = focusKey;
    if (monitoringOrganizationId != null) showLayers(['points', 'projectSites']);
    skipNextSyncRef.current = true;
    map.fitBounds(bounds, {
      paddingTopLeft: [60, 60],
      paddingBottomRight: [map.getSize().x >= 900 ? 450 : 36, 60],
      maxZoom: 20,
      animate: true,
    });
  }, [isMonitoringMapMode, monitoringOrganizationId, projectSites, showLayers]);

  // Dashboard links frame the site's actual planting points. Risk links frame
  // just its risk points, with the site boundary as a no-points fallback.
  useEffect(() => {
    const map = mapRef.current;
    const params = new URLSearchParams(location.search);
    const projectSiteId = params.get('project_site_id');
    const focusTarget = params.get('focus');

    if (!projectSiteId) {
      layersRef.current.projectSiteLayer?.setStyle((feature) => projectSiteStyle(feature, PROJECT_SITE_STYLE));
      focusedProjectSiteRef.current = '';
      return;
    }
    if (!map) return;

    const fallback = location.state?.mapFocusSite;
    const feature = projectSites?.features?.find((candidate) => (
      String(candidate?.id ?? candidate?.properties?.id ?? '') === String(projectSiteId)
    )) || (String(fallback?.id ?? '') === String(projectSiteId)
      ? { id: fallback.id, geometry: fallback.geometry, properties: fallback }
      : null);
    if (!feature) return;

    layersRef.current.projectSiteLayer?.eachLayer((siteLayer) => {
      const layerSiteId = siteLayer.feature?.id ?? siteLayer.feature?.properties?.id;
      const isFocused = String(layerSiteId ?? '') === String(projectSiteId);
      siteLayer.setStyle(projectSiteStyle(siteLayer.feature, isFocused ? FOCUSED_PROJECT_SITE_STYLE : PROJECT_SITE_STYLE));
      // Keep the boundary behind point markers even when it is highlighted.
      siteLayer.bringToBack();
    });

    const sitePoints = spacedPoints.filter((point) => (
      point.inside_visible_map !== false
      && String(point.source_project_site_id ?? '') === String(projectSiteId)
      && Number.isFinite(Number(point.latitude))
      && Number.isFinite(Number(point.longitude))
    ));
    const riskPoints = focusTarget === 'risk_areas'
      ? sitePoints.filter((point) => point.survival_warning)
      : [];
    const targetPoints = riskPoints.length ? riskPoints : sitePoints;
    const focusKey = `${location.key}:${projectSiteId}:${focusTarget || 'site'}:${targetPoints.length ? 'points' : 'boundary'}`;
    if (focusedProjectSiteRef.current === focusKey) return;

    showLayers(focusTarget === 'risk_areas'
      ? ['points', 'projectSites', 'warnings']
      : ['points', 'projectSites']);

    const frame = window.requestAnimationFrame(() => {
      map.invalidateSize({ pan: false });
      let bounds = null;
      if (targetPoints.length) {
        bounds = L.latLngBounds(targetPoints.map((point) => [Number(point.latitude), Number(point.longitude)]));
      } else if (feature.geometry) {
        try { bounds = L.geoJSON(feature).getBounds(); } catch { /* Use the stored centroid below. */ }
      }
      if (bounds?.isValid()) {
        const rightPadding = map.getSize().x >= 900 ? 420 : 36;
        map.fitBounds(bounds, {
          paddingTopLeft: [72, 72],
          paddingBottomRight: [rightPadding, 72],
          maxZoom: 21,
          animate: false,
        });
        focusedProjectSiteRef.current = focusKey;
        return;
      }

      const latitude = Number(feature.properties?.centroid_lat);
      const longitude = Number(feature.properties?.centroid_lon);
      if (Number.isFinite(latitude) && Number.isFinite(longitude)) {
        map.setView([latitude, longitude], 20, { animate: false });
        focusedProjectSiteRef.current = focusKey;
      }
    });

    return () => window.cancelAnimationFrame(frame);
  }, [location.key, location.search, location.state, projectSites, showLayers, spacedPoints]);

  // The Project Sites dashboard can open the map directly from the risk-area
  // chart. Show the warning and planting-point layers, then frame every mapped
  // warning area so the chart can be inspected spatially.
  useEffect(() => {
    const map = mapRef.current;
    const focusTarget = new URLSearchParams(location.search).get('focus');

    if (focusTarget !== 'risk_areas' || new URLSearchParams(location.search).has('project_site_id')) {
      focusedRiskAreasRef.current = '';
      return;
    }
    if (!map || !Array.isArray(warningZones?.features) || !warningZones.features.length) return;

    const focusKey = `${location.key}:risk_areas`;
    if (focusedRiskAreasRef.current === focusKey) return;

    try {
      const bounds = L.geoJSON(warningZones).getBounds();
      if (!bounds.isValid()) return;

      showLayers(['points', 'warnings']);
      const rightPadding = map.getSize().x >= 900 ? 420 : 36;
      skipNextSyncRef.current = true;
      map.fitBounds(bounds, {
        paddingTopLeft: [72, 72],
        paddingBottomRight: [rightPadding, 72],
        maxZoom: 20,
        animate: true,
      });
      focusedRiskAreasRef.current = focusKey;
    } catch {
      // Keep the current map view when a warning boundary is malformed.
    }
  }, [location.key, location.search, showLayers, warningZones]);

  return <div ref={containerRef} className="map-container" />;
}
