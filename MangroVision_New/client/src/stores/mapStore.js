import { create } from 'zustand';

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
  stats: null,
  loadingStats: false,
  fetchStats: async () => {
    set({ loadingStats: true });
    try {
      const res = await fetch(`${API}/api/analyses/stats`);
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
  fetchPoints: async () => {
    set({ loadingPoints: true });
    try {
      const res = await fetch(`${API}/api/planters/map-points`);
      const data = await res.json();
      set({ points: data, loadingPoints: false });
    } catch (err) {
      console.error('Failed to fetch points:', err);
      set({ loadingPoints: false });
    }
  },
  deletePoints: async (pointIds) => {
    const ids = [...new Set((pointIds || []).map((id) => Number(id)).filter((id) => Number.isFinite(id)))];
    if (!ids.length) throw new Error('Select at least one planting point.');

    const res = await fetch(`${API}/api/planters/map-points`, {
      method: 'DELETE',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ point_ids: ids }),
    });
    const payload = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(payload.detail || 'Point deletion failed');
    }

    const deletedIds = new Set((payload.deleted_point_ids || []).map((id) => Number(id)));
    set((state) => ({
      points: state.points.filter((point) => !deletedIds.has(Number(point.id))),
      selectedPointId: deletedIds.has(Number(state.selectedPointId)) ? null : state.selectedPointId,
      deletionSelectedPointIds: state.deletionSelectedPointIds
        .map(Number)
        .filter((id) => !deletedIds.has(id)),
      stats: state.stats
        ? {
            ...state.stats,
            points: (state.stats.points || []).filter((point) => !deletedIds.has(Number(point.id))),
            total_mapped_points: Math.max(
              0,
              (state.stats.total_mapped_points ?? state.points.length) - deletedIds.size,
            ),
          }
        : state.stats,
    }));
    return payload;
  },

  // Zones
  forbiddenZones: null,
  erodedZones: null,
  fetchZones: async () => {
    try {
      const [fRes, eRes] = await Promise.all([
        fetch(`${API}/api/zones/forbidden`),
        fetch(`${API}/api/zones/eroded`),
      ]);
      const [forbidden, eroded] = await Promise.all([fRes.json(), eRes.json()]);
      set({ forbiddenZones: forbidden, erodedZones: eroded });
    } catch (err) {
      console.error('Failed to fetch zones:', err);
    }
  },

  // Selected point
  selectedPointId: null,
  setSelectedPoint: (id) => set({ selectedPointId: id }),

  // Delete Points selection. This lives in the shared map store so the
  // delete workflow can use the already-mounted main map instead of creating
  // a second Leaflet instance that reloads tiles on navigation.
  deletionSelectedPointIds: [],
  toggleDeletionPoint: (id) => {
    const numericId = Number(id);
    if (!Number.isFinite(numericId)) return;
    set((state) => {
      const selected = new Set(state.deletionSelectedPointIds.map(Number));
      if (selected.has(numericId)) selected.delete(numericId);
      else selected.add(numericId);
      return { deletionSelectedPointIds: [...selected] };
    });
  },
  addDeletionPoints: (ids) => {
    const nextIds = (ids || []).map(Number).filter(Number.isFinite);
    if (!nextIds.length) return;
    set((state) => {
      const selected = new Set(state.deletionSelectedPointIds.map(Number));
      nextIds.forEach((id) => selected.add(id));
      return { deletionSelectedPointIds: [...selected] };
    });
  },
  removeDeletionPoint: (id) => {
    const numericId = Number(id);
    if (!Number.isFinite(numericId)) return;
    set((state) => ({
      deletionSelectedPointIds: state.deletionSelectedPointIds
        .map(Number)
        .filter((selectedId) => selectedId !== numericId),
    }));
  },
  clearDeletionSelection: () => set({ deletionSelectedPointIds: [] }),
  pruneDeletionSelection: (validIds) => {
    const valid = new Set((validIds || []).map(Number).filter(Number.isFinite));
    set((state) => ({
      deletionSelectedPointIds: state.deletionSelectedPointIds
        .map(Number)
        .filter((selectedId) => valid.has(selectedId)),
    }));
  },

  // Current unsaved processing preview
  currentAnalysis: null,
  setCurrentAnalysis: (analysis) => set({ currentAnalysis: analysis }),
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

    const newAnalysisEntry = {
      id: analysisId,
      image_name: imageName,
      analyzed_at: nowIso,
      hexagon_count: metrics.safe_hexagon_count ?? metrics.hexagon_count ?? coords.length,
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
      assigned_planter_name: null,
      assignment_status: null,
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
    forbidden: true,
    eroded: true,
  },
  toggleLayer: (layerName) => {
    const current = get().layerVisibility;
    set({ layerVisibility: { ...current, [layerName]: !current[layerName] } });
  },

  // Active panel mode
  activeMode: 'analytics',
  setActiveMode: (mode) => set({ activeMode: mode }),
}));
