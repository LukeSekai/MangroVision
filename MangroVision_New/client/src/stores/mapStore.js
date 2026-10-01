import { create } from 'zustand';
import { savedAnalysisMapContext } from '../utils/analysisMapContext';
import { paintReplantingSelection } from '../utils/replantingBrush';

const API = import.meta.env.VITE_API_BASE || '';
const MAP_VIEW_STORAGE_KEY = 'mv_admin_map_view';
const DEFAULT_MAP_CENTER = [122.6253, 10.7800]; // Leganes Katunggan Park, Iloilo
// Around the Leaflet 50 m scale-bar range at this latitude.
const DEFAULT_MAP_ZOOM = 18;

function readStoredMapView() {
  if (typeof window === 'undefined') {
    return { center: DEFAULT_MAP_CENTER, zoom: DEFAULT_MAP_ZOOM };
  }
  try {
    const stored = JSON.parse(window.localStorage.getItem(MAP_VIEW_STORAGE_KEY) || 'null');
    const center = Array.isArray(stored?.center) ? stored.center.map(Number) : null;
    const zoom = Number(stored?.zoom);
    if (
      center?.length === 2 &&
      center.every(Number.isFinite) &&
      Number.isFinite(zoom)
    ) {
      return { center, zoom };
    }
  } catch {
    // Ignore malformed saved view and use the default below.
  }
  return { center: DEFAULT_MAP_CENTER, zoom: DEFAULT_MAP_ZOOM };
}

function persistMapView(center, zoom) {
  if (typeof window === 'undefined') return;
  try {
    window.localStorage.setItem(MAP_VIEW_STORAGE_KEY, JSON.stringify({ center, zoom }));
  } catch {
    // localStorage can be unavailable in private/restricted contexts.
  }
}

const initialMapView = readStoredMapView();

export const useMapStore = create((set, get) => ({
  // Map viewport
  center: initialMapView.center,
  zoom: initialMapView.zoom,
  setView: (center, zoom) => {
    persistMapView(center, zoom);
    set({ center, zoom });
  },

  // Stats
  resetWorkspaceData: () => set({
    stats: null, points: [], loadingStats: false, loadingPoints: false,
    forbiddenZones: null, erodedZones: null, siteZones: null,
    projectSites: null, siteZoneMortality: null, warningZones: null,
    selectedPointId: null, assignmentSelectedPointIds: [],
    assignmentOrganizationId: null, assignmentProjectSiteId: null,
    monitoringOrganizationId: null,
    replantingSelectedPointIds: [], replantingSelectionMode: 'click',
    monitoringSelectedPointId: null, currentAnalysis: null,
  }),
  stats: null,
  loadingStats: false,
  fetchStats: async ({ force = false } = {}) => {
    set({ loadingStats: true });
    try {
      const res = await fetch(`${API}/api/analyses/stats`, { cache: force ? 'reload' : 'default' });
      if (!res.ok) throw new Error(`Could not load statistics (${res.status}).`);
      const data = await res.json();
      set({ stats: data, loadingStats: false });
    } catch (err) {
      console.error('Failed to fetch stats:', err);
      set({ loadingStats: false });
    }
  },

  // Map points (all planting points with assignment info)
  points: [],
  loadingPoints: false,
  fetchPoints: async ({ force = false } = {}) => {
    set({ loadingPoints: true });
    try {
      const res = await fetch(`${API}/api/planters/map-points`, { cache: force ? 'reload' : 'default' });
      if (!res.ok) throw new Error(`Could not load map points (${res.status}).`);
      const data = await res.json();
      set({ points: data, loadingPoints: false });
    } catch (err) {
      console.error('Failed to fetch points:', err);
      set({ loadingPoints: false });
    }
  },
  // Zones
  forbiddenZones: null,
  erodedZones: null,
  // Site zones — auto-derived from planter assignments by the backend
  // (one zone per assignment, polygon = convex hull of its points). Used
  // for per-assignment survival tracking on the map. There is no manual
  // draw flow anymore — assignments ARE the zones.
  siteZones: null,
  projectSites: null,
  siteZoneMortality: null,
  warningZones: null,
  fetchZones: async ({ force = false } = {}) => {
    const paths = [
      '/api/zones/forbidden',
      '/api/zones/eroded',
      '/api/zones/sites',
      '/api/zones/sites/mortality',
      '/api/zones/warnings',
      '/api/project-sites',
    ];
    const results = await Promise.allSettled(paths.map(async (path) => {
      const response = await fetch(`${API}${path}`, { cache: force ? 'reload' : 'default' });
      if (!response.ok) throw new Error(`${path} returned ${response.status}`);
      return response.json();
    }));
    results.forEach((result, index) => {
      if (result.status === 'rejected') {
        console.error(`Failed to fetch ${paths[index]}:`, result.reason);
      }
    });
    set((state) => {
      const value = (index) => (
        results[index].status === 'fulfilled' ? results[index].value : undefined
      );
      const mortality = value(3);
      return {
        forbiddenZones: value(0) ?? state.forbiddenZones,
        erodedZones: value(1) ?? state.erodedZones,
        siteZones: value(2) ?? state.siteZones,
        siteZoneMortality: mortality ? (mortality.zones || []) : state.siteZoneMortality,
        warningZones: value(4) ?? state.warningZones,
        projectSites: value(5) ?? state.projectSites,
      };
    });
  },
  fetchSiteZoneMortality: async () => {
    try {
      const res = await fetch(`${API}/api/zones/sites/mortality`);
      const data = await res.json();
      set({ siteZoneMortality: data?.zones || [] });
    } catch (err) {
      console.error('Failed to fetch site zone mortality:', err);
    }
  },

  // Selected point
  selectedPointId: null,
  setSelectedPoint: (id) => set({ selectedPointId: id }),

  // Map Analytics death filter and reviewed release of planting locations.
  showDeadPointsOnly: false,
  setShowDeadPointsOnly: (value) => set({ showDeadPointsOnly: value }),
  monitoringOrganizationId: null,
  setMonitoringOrganizationId: (value) => set({
    monitoringOrganizationId: value == null || value === '' ? null : Number(value),
    replantingSelectedPointIds: [],
  }),
  replantingSelectedPointIds: [],
  replantingSelectionMode: 'click',
  setReplantingSelectionMode: (mode) => set({ replantingSelectionMode: mode }),
  paintReplantingPoints: (ids, mode) => set((state) => {
    const eligible = new Set(state.points.filter((point) => point.death_at).map((point) => point.id));
    const selection = paintReplantingSelection(state.replantingSelectedPointIds, ids.filter((id) => eligible.has(id)), mode);
    return selection === state.replantingSelectedPointIds ? state : { replantingSelectedPointIds: selection };
  }),
  toggleReplantingPoint: (id) => set((state) => ({
    replantingSelectedPointIds: state.replantingSelectedPointIds.includes(id)
      ? state.replantingSelectedPointIds.filter((pointId) => pointId !== id)
      : state.replantingSelectedPointIds.length < 500 ? [...state.replantingSelectedPointIds, id] : state.replantingSelectedPointIds,
  })),
  clearReplantingSelection: () => set({ replantingSelectedPointIds: [] }),

  // Planter Management batch assignment selection. Lives in the shared map
  // store so MapView can highlight selected points and toggle them on click.
  // One selected point and many selected points use the same flow.
  assignmentSelectedPointIds: [],
  assignmentOrganizationId: null,
  assignmentProjectSiteId: null,
  setAssignmentScope: (organizationId = null, projectSiteId = null) => set({
    assignmentOrganizationId: organizationId == null ? null : Number(organizationId),
    assignmentProjectSiteId: projectSiteId == null ? null : Number(projectSiteId),
  }),
  toggleAssignmentPoint: (id) => {
    const numericId = Number(id);
    if (!Number.isFinite(numericId)) return;
    set((state) => {
      const selected = new Set(state.assignmentSelectedPointIds.map(Number));
      if (selected.has(numericId)) selected.delete(numericId);
      else selected.add(numericId);
      return { assignmentSelectedPointIds: [...selected] };
    });
  },
  clearAssignmentSelection: () => set({ assignmentSelectedPointIds: [] }),
  setAssignmentSelection: (ids) => set({ assignmentSelectedPointIds: ids }),

  // Monitoring: which planted/dead point the user clicked. The Monitoring page
  // listens for this to open its "mark dead / restore" modal so the click flow
  // on /monitoring stays distinct from /points and the default selectedPoint.
  monitoringSelectedPointId: null,
  setMonitoringSelectedPoint: (id) => {
    const numericId = id == null ? null : Number(id);
    set({ monitoringSelectedPointId: Number.isFinite(numericId) ? numericId : null });
  },
  clearMonitoringSelectedPoint: () => set({ monitoringSelectedPointId: null }),

  markPointDead: async (pointId, reasonCategory, notes) => {
    const numericId = Number(pointId);
    if (!Number.isFinite(numericId)) throw new Error('Invalid point id.');
    const res = await fetch(`${API}/api/planters/map-points/${numericId}/death`, {
      method: 'PATCH',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        reason_category: reasonCategory,
        notes: notes || '',
      }),
    });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(payload.detail || 'Could not mark point dead.');
    set((state) => ({
      points: state.points.map((point) => {
        if (Number(point.id) !== numericId) return point;
        return {
          ...point,
          death_at: payload.death_at,
          death_reason: payload.death_reason,
          death_reason_category: payload.death_reason_category,
          death_notes: payload.death_notes,
        };
      }),
    }));
    return payload;
  },

  restorePointToPlanted: async (pointId) => {
    const numericId = Number(pointId);
    if (!Number.isFinite(numericId)) throw new Error('Invalid point id.');
    const res = await fetch(`${API}/api/planters/map-points/${numericId}/restore`, {
      method: 'POST',
    });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(payload.detail || 'Could not restore point.');
    set((state) => ({
      points: state.points.map((point) => {
        if (Number(point.id) !== numericId) return point;
        return {
          ...point,
          death_at: null,
          death_reason: null,
          death_reason_category: null,
          death_notes: null,
        };
      }),
    }));
    return payload;
  },

  resetPointToPlanned: async (pointId) => {
    const numericId = Number(pointId);
    if (!Number.isFinite(numericId)) throw new Error('Invalid point id.');
    const res = await fetch(`${API}/api/planters/map-points/${numericId}/reset-to-planned`, {
      method: 'POST',
    });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(payload.detail || 'Could not reset point to planned.');
    set((state) => ({
      points: state.points.map((point) => {
        if (Number(point.id) !== numericId) return point;
        // Release the assignment fields too — without this the marker keeps
        // ranking as 'completed' (yellow) because assignment_status wins
        // over planting_status in MapView.getPointDisplayStatus.
        return {
          ...point,
          planting_status: 'planned',
          planted_at: null,
          planted_date: null,
          death_at: null,
          death_reason: null,
          death_reason_category: null,
          death_notes: null,
          assigned_planter_id: null,
          assigned_planter_name: null,
          assignment_id: null,
          assignment_point_id: null,
          assignment_status: null,
          assignment_title: null,
          assignment_date: null,
          sequence_num: null,
        };
      }),
    }));
    return payload;
  },

  // Current processing preview or the selected saved photo's boundary.
  currentAnalysis: null,
  setCurrentAnalysis: (analysis) => set({ currentAnalysis: analysis }),
  setSavedAnalysisBoundary: (analysis, options) => set({
    currentAnalysis: savedAnalysisMapContext(analysis, options),
  }),
  clearCurrentAnalysis: () => set({ currentAnalysis: null }),

  // Optimistic append of a just-saved analysis to every view that reads from
  // the store — Map Analytics counters, analysis history, and the map markers.
  // Expects the full `result` payload returned by /api/analyses/process-stream
  // plus the /api/analyses/save response so we can synthesize the saved
  // analysis entry and its planting points without re-fetching the dataset.
  appendSavedAnalysis: ({ result, savePayload }) => set((state) => {
    if (!result || !savePayload?.analysis_id) return state;

    const analysisId = savePayload.analysis_id;
    const metrics = result.metrics || {};
    const coords = result.map?.coordinates || [];
    const imageName = result.uploaded_file_name || 'Saved Analysis';
    const nowIso = new Date().toISOString();
    const pointSpecies = result.parameters?.species || null;
    const plantingDistanceM = result.parameters?.planting_distance_m ?? null;

    const newAnalysisEntry = {
      id: analysisId,
      image_name: imageName,
      analyzed_at: nowIso,
      // History stores every retained geotagged point. The safe count is a
      // separate, dynamic metric that excludes erosion-unavailable points.
      hexagon_count: metrics.hexagon_count ?? coords.length,
      plantable_area_m2: metrics.plantable_area_m2 ?? 0,
    };

    const synthesizedPoints = coords.map((row) => ({
      id: `new-${analysisId}-${row.point_num}`,
      analysis_id: analysisId,
      image_name: imageName,
      point_num: row.point_num,
      latitude: row.latitude,
      longitude: row.longitude,
      buffer_m: row.buffer_m ?? null,
      area_m2: row.area_m2 ?? null,
      status: 'planned',
      planting_status: 'planned',
      species: pointSpecies,
      planting_distance_m: plantingDistanceM,
      planted_at: null,
      planted_date: null,
      assigned_planter_name: null,
      assignment_status: null,
      inside_eroded_zone: Boolean(row.eroded_unavailable),
      erosion_advisory: Boolean(row.eroded_unavailable),
      eroded_unavailable: Boolean(row.eroded_unavailable),
      availability_status: row.availability_status || 'available',
      availability_reason: row.eroded_unavailable ? 'Inside an eroded zone' : null,
      survival_warning: false,
      warning_zone_ids: [],
      warning_zone_names: [],
      warning_reasons: [],
      warning_severity: null,
      warning_summary: null,
    }));

    const prevStats = state.stats || {};
    const prevAnalyses = (prevStats.analyses || []).filter((a) => a.id !== analysisId);
    const prevStatsPoints = (prevStats.points || []).filter(
      (p) => !String(p?.id ?? '').startsWith(`new-${analysisId}-`),
    );
    const prevMapPoints = state.points.filter(
      (p) => !String(p?.id ?? '').startsWith(`new-${analysisId}-`),
    );

    const priorAnalysesCount = prevStats.total_analyses ?? prevAnalyses.length;
    const priorMappedCount = prevStats.total_mapped_points ?? prevStatsPoints.length;

    const nextStats = {
      ...prevStats,
      analyses: [newAnalysisEntry, ...prevAnalyses],
      points: [...prevStatsPoints, ...synthesizedPoints],
      total_analyses: priorAnalysesCount + 1,
      total_mapped_points: priorMappedCount + (savePayload.new_points ?? synthesizedPoints.length),
      total_coverage_m2:
        (prevStats.total_coverage_m2 || 0) + (metrics.total_area_m2 || 0),
      total_plantable_m2:
        (prevStats.total_plantable_m2 || 0) + (metrics.plantable_area_m2 || 0),
      total_danger_m2:
        (prevStats.total_danger_m2 || 0) + (metrics.danger_area_m2 || 0),
      total_canopies:
        (prevStats.total_canopies || 0) + (metrics.canopy_count || 0),
    };

    return {
      stats: nextStats,
      points: [...prevMapPoints, ...synthesizedPoints],
    };
  }),

  // Map instance ref (set by MapView)
  mapInstance: null,
  setMapInstance: (map) => set({ mapInstance: map }),

  // Layer visibility
  layerVisibility: {
    points: true,
    orthophoto: true,
    siteZones: true,
    projectSites: true,
    forbidden: true,
    eroded: true,
    warnings: true,
  },
  toggleLayer: (layerName) => {
    const current = get().layerVisibility;
    set({ layerVisibility: { ...current, [layerName]: !current[layerName] } });
  },
  showLayers: (layerNames = []) => {
    const current = get().layerVisibility;
    const next = { ...current };
    layerNames.forEach((layerName) => {
      if (Object.prototype.hasOwnProperty.call(next, layerName)) next[layerName] = true;
    });
    set({ layerVisibility: next });
  },

  // Active panel mode
  activeMode: 'analytics',
  setActiveMode: (mode) => set({ activeMode: mode }),
}));
