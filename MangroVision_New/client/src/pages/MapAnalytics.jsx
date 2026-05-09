import { useEffect, useState } from 'react';
import { Panel, PanelCard } from '../components/Panel';
import { useMapStore } from '../stores/mapStore';
import Modal from '../components/Modal';
import ResultsOverlay from './ResultsOverlay';
import './ResultsOverlay.css';
import './ImageProcessing.css';
import './MapAnalytics.css';

const API = import.meta.env.VITE_API_BASE || '';

function downloadBlob(blob, fileName) {
  const url = window.URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.URL.revokeObjectURL(url);
}

export default function MapAnalytics() {
  const stats = useMapStore((s) => s.stats);
  const fetchStats = useMapStore((s) => s.fetchStats);
  const points = useMapStore((s) => s.points);
  const fetchPoints = useMapStore((s) => s.fetchPoints);
  const layerVisibility = useMapStore((s) => s.layerVisibility);
  const toggleLayer = useMapStore((s) => s.toggleLayer);

  const [busyExport, setBusyExport] = useState('');
  const [deleteError, setDeleteError] = useState('');
  const [exportError, setExportError] = useState('');
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const [loadingAnalysisId, setLoadingAnalysisId] = useState(null);
  const [analysisLoadError, setAnalysisLoadError] = useState('');
  const [pendingDeleteId, setPendingDeleteId] = useState(null);
  const [deleteBusy, setDeleteBusy] = useState(false);

  useEffect(() => {
    fetchStats();
    fetchPoints();
  }, [fetchPoints, fetchStats]);

  const analyses = stats?.analyses || [];
  const allSavedPoints = stats?.points || [];

  const planned = points.filter((point) => !point.assigned_planter_name && point.planting_status !== 'planted').length;
  const assigned = points.filter((point) => point.assigned_planter_name && point.assignment_status !== 'completed').length;
  const completed = points.filter((point) => point.assignment_status === 'completed' || point.planting_status === 'planted').length;

  const handleDeleteAnalysis = async (analysisId) => {
    setDeleteError('');
    setDeleteBusy(true);
    try {
      const response = await fetch(`${API}/api/analyses/${analysisId}`, { method: 'DELETE' });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || 'Delete failed');
      }
      setPendingDeleteId(null);
      fetchStats();
      fetchPoints();
    } catch (error) {
      setDeleteError(error.message || 'Delete failed');
    } finally {
      setDeleteBusy(false);
    }
  };

  const pendingDeleteAnalysis = analyses.find((a) => a.id === pendingDeleteId);

  const handleOpenAnalysis = async (analysisId) => {
    if (loadingAnalysisId) return;
    setLoadingAnalysisId(analysisId);
    setAnalysisLoadError('');
    try {
      const response = await fetch(`${API}/api/analyses/${analysisId}`);
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || 'Could not load analysis');
      }
      const detail = await response.json();
      setSelectedAnalysis(detail);
    } catch (error) {
      setAnalysisLoadError(error.message || 'Could not load analysis');
    } finally {
      setLoadingAnalysisId(null);
    }
  };

  const handleExportSelected = async (format) => {
    if (!selectedAnalysis?.exports?.waypoints?.length) return;
    const response = await fetch(`${API}/api/export/${format}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        waypoints: selectedAnalysis.exports.waypoints,
        image_name: selectedAnalysis.uploaded_file_name,
        detection_mode: selectedAnalysis.detection_mode,
      }),
    });
    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Could not export ${format.toUpperCase()}`);
    }
    const blob = await response.blob();
    downloadBlob(blob, `mangrovision_${selectedAnalysis.uploaded_file_name}.${format}`);
  };

  const handleExport = async (format) => {
    if (!allSavedPoints.length) return;

    setBusyExport(format);
    try {
      const response = await fetch(`${API}/api/export/${format}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          waypoints: allSavedPoints.map((point) => ({
            lat: point.latitude,
            lon: point.longitude,
            point_num: point.point_num,
            buffer_m: point.buffer_m,
            area_m2: point.area_m2,
            status: point.status,
            name: point.image_name ? `${point.image_name}-${String(point.point_num).padStart(3, '0')}` : undefined,
          })),
          image_name: 'all_analyses',
        }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `Export failed (${format})`);
      }

      const blob = await response.blob();
      downloadBlob(blob, `mangrovision_all_points.${format}`);
    } catch (error) {
      setExportError(error.message || 'Export failed');
    } finally {
      setBusyExport('');
    }
  };

  return (
    <Panel title="Map Analytics" subtitle={`${allSavedPoints.length} saved planting points`}>
      <PanelCard
        title="Overview"
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <rect x="3" y="3" width="7" height="7" rx="1" />
            <rect x="14" y="3" width="7" height="7" rx="1" />
            <rect x="3" y="14" width="7" height="7" rx="1" />
            <rect x="14" y="14" width="7" height="7" rx="1" />
          </svg>
        }
      >
        <div className="stats-grid">
          <div className="stat-card"><div className="stat-label">Analyses</div><div className="stat-value">{stats?.total_analyses ?? '-'}</div></div>
          <div className="stat-card"><div className="stat-label">Mapped Points</div><div className="stat-value">{stats?.total_mapped_points ?? allSavedPoints.length}</div></div>
          <div className="stat-card"><div className="stat-label">Planned</div><div className="stat-value" style={{ color: 'var(--color-planned)' }}>{planned}</div></div>
          <div className="stat-card"><div className="stat-label">Assigned</div><div className="stat-value" style={{ color: 'var(--color-assigned)' }}>{assigned}</div></div>
          <div className="stat-card"><div className="stat-label">Completed</div><div className="stat-value" style={{ color: 'var(--color-completed)' }}>{completed}</div></div>
          <div className="stat-card"><div className="stat-label">Active Planters</div><div className="stat-value">{stats?.active_planters ?? '-'}</div></div>
        </div>
      </PanelCard>

      <PanelCard
        title="Legend"
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="10" />
            <path d="M12 16v-4" />
            <path d="M12 8h.01" />
          </svg>
        }
      >
        <div className="legend-list">
          <div className="legend-item"><span className="legend-dot" style={{ background: '#16a34a' }} /><span>Planned</span></div>
          <div className="legend-item"><span className="legend-dot" style={{ background: '#2563eb' }} /><span>Assigned</span></div>
          <div className="legend-item"><span className="legend-dot" style={{ background: '#d97706' }} /><span>Planted</span></div>
          <div className="legend-item"><span className="legend-dot" style={{ background: '#059669' }} /><span>Completed</span></div>
          <div className="legend-item"><span className="legend-dot legend-dot-outline" style={{ borderColor: '#dc2626' }} /><span>Forbidden Zone</span></div>
          <div className="legend-item"><span className="legend-dot legend-dot-outline" style={{ borderColor: '#ea580c' }} /><span>Eroded Zone</span></div>
        </div>
      </PanelCard>

      <PanelCard
        title="Layers"
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <polygon points="12 2 2 7 12 12 22 7 12 2" />
            <polyline points="2 17 12 22 22 17" />
            <polyline points="2 12 12 17 22 12" />
          </svg>
        }
      >
        <div className="layer-toggles">
          <label className="layer-toggle">
            <input type="checkbox" checked={layerVisibility.points} onChange={() => toggleLayer('points')} />
            <span>Planting Points</span>
          </label>
          <label className="layer-toggle">
            <input type="checkbox" checked={layerVisibility.orthophoto} onChange={() => toggleLayer('orthophoto')} />
            <span>Orthophoto Overlay</span>
          </label>
          <label className="layer-toggle">
            <input type="checkbox" checked={layerVisibility.forbidden} onChange={() => toggleLayer('forbidden')} />
            <span>Forbidden Zones</span>
          </label>
          <label className="layer-toggle">
            <input type="checkbox" checked={layerVisibility.eroded} onChange={() => toggleLayer('eroded')} />
            <span>Eroded Zones</span>
          </label>
        </div>
      </PanelCard>

      <PanelCard
        title="Coverage"
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
            <circle cx="12" cy="10" r="3" />
          </svg>
        }
        defaultOpen={false}
      >
        <div className="info-rows">
          <div className="info-row"><span className="info-label">Total area analyzed</span><span className="info-value">{stats?.total_coverage_m2 ? `${(stats.total_coverage_m2 / 10000).toFixed(2)} ha` : '-'}</span></div>
          <div className="info-row"><span className="info-label">Plantable area</span><span className="info-value">{stats?.total_plantable_m2 ? `${stats.total_plantable_m2.toFixed(1)} m²` : '-'}</span></div>
          <div className="info-row"><span className="info-label">Danger area</span><span className="info-value">{stats?.total_danger_m2 ? `${stats.total_danger_m2.toFixed(1)} m²` : '-'}</span></div>
          <div className="info-row"><span className="info-label">Canopies detected</span><span className="info-value">{stats?.total_canopies ?? '-'}</span></div>
        </div>
      </PanelCard>

      <PanelCard
        title="Analysis History"
        badge={analyses.length}
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 3v5h5" />
            <path d="M3.05 13A9 9 0 1 0 6 5.3L3 8" />
            <path d="M12 7v5l3 3" />
          </svg>
        }
        defaultOpen={false}
      >
        {deleteError && <div className="analytics-error">{deleteError}</div>}
        {analysisLoadError && <div className="analytics-error">{analysisLoadError}</div>}
        <div className="analytics-history-list">
          {analyses.length === 0 ? (
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No saved analyses yet.</p>
          ) : (
            analyses.map((analysis) => {
              const isLoading = loadingAnalysisId === analysis.id;
              return (
                <div
                  key={analysis.id}
                  className={`analytics-history-item analytics-history-clickable ${isLoading ? 'analytics-history-loading' : ''}`}
                  role="button"
                  tabIndex={0}
                  onClick={() => handleOpenAnalysis(analysis.id)}
                  onKeyDown={(event) => {
                    if (event.key === 'Enter' || event.key === ' ') {
                      event.preventDefault();
                      handleOpenAnalysis(analysis.id);
                    }
                  }}
                  aria-label={`Open summary for ${analysis.image_name}`}
                >
                  <div className="analytics-history-main">
                    <div className="analytics-history-title">{analysis.image_name}</div>
                    <div className="analytics-history-meta">
                      {analysis.analyzed_at?.slice(0, 10)} · {analysis.hexagon_count} pts · {analysis.plantable_area_m2?.toFixed(1)} m²
                    </div>
                  </div>
                  <span className="analytics-history-hint" aria-hidden="true">
                    {isLoading ? (
                      'Loading…'
                    ) : (
                      <>
                        View Details
                        <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                          <line x1="5" y1="12" x2="19" y2="12" />
                          <polyline points="12 5 19 12 12 19" />
                        </svg>
                      </>
                    )}
                  </span>
                  <button
                    className="btn btn-ghost btn-sm btn-icon analytics-history-delete"
                    onClick={(event) => {
                      event.stopPropagation();
                      setPendingDeleteId(analysis.id);
                    }}
                    title={`Delete ${analysis.image_name}`}
                    aria-label={`Delete analysis ${analysis.image_name}`}
                  >
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                      <polyline points="3 6 5 6 21 6" />
                      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                    </svg>
                  </button>
                </div>
              );
            })
          )}
        </div>
      </PanelCard>

      <PanelCard
        title="Export Saved Points"
        badge={allSavedPoints.length}
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="7 10 12 15 17 10" />
            <line x1="12" y1="15" x2="12" y2="3" />
          </svg>
        }
        defaultOpen={false}
      >
        {exportError && <div className="analytics-error">{exportError}</div>}
        <div className="export-grid">
          {['csv', 'gpx', 'kml', 'geojson'].map((fmt) => (
            <button
              key={fmt}
              className="btn btn-secondary btn-sm"
              onClick={() => { setExportError(''); handleExport(fmt); }}
              disabled={!allSavedPoints.length || Boolean(busyExport)}
            >
              {busyExport === fmt ? 'Exporting…' : fmt.toUpperCase()}
            </button>
          ))}
        </div>
      </PanelCard>

      <ResultsOverlay
        open={Boolean(selectedAnalysis)}
        result={selectedAnalysis}
        originalPreview={null}
        saving={false}
        saved
        saveError=""
        onClose={() => setSelectedAnalysis(null)}
        onSave={() => {}}
        onExport={handleExportSelected}
      />

      <Modal
        open={Boolean(pendingDeleteId)}
        title={`Delete "${pendingDeleteAnalysis?.image_name || 'this analysis'}"?`}
        variant="danger"
        confirmLabel="Delete analysis"
        cancelLabel="Cancel"
        busy={deleteBusy}
        onConfirm={() => handleDeleteAnalysis(pendingDeleteId)}
        onCancel={() => { if (!deleteBusy) setPendingDeleteId(null); }}
      >
        <p>This permanently removes the analysis and all its saved planting points from the database. This action cannot be undone.</p>
      </Modal>
    </Panel>
  );
}
