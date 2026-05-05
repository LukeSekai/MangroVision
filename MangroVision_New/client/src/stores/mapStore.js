import { create } from 'zustand';

const API = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

export const useMapStore = create((set, get) => ({
  // Map viewport
  center: [122.6253, 10.7800],  // Leganes Katunggan Park, Iloilo
  zoom: 17,
  setView: (center, zoom) => set({ center, zoom }),

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
