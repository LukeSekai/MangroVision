import { useEffect, useMemo, useState } from 'react';
import { Panel, PanelCard } from '../components/Panel';
import { useMapStore } from '../stores/mapStore';
import { countMapPointStatuses } from '../utils/mapPointStats';
import Modal from '../components/Modal';
import { POINT_STATUS_COLORS } from '../utils/pointStatus';
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
  const [exportError, setExportError] = useState('');
  const [pendingExportFormat, setPendingExportFormat] = useState(null);
  const [completedExportFormat, setCompletedExportFormat] = useState(null);

  useEffect(() => {
    fetchStats();
    fetchPoints();
  }, [fetchPoints, fetchStats]);

  const allSavedPoints = stats?.points || [];
  const { mapped, planned, assigned, planted, dead, skipped, unavailable } = useMemo(
    () => countMapPointStatuses(points),
    [points],
  );

  const requestSavedPointsExport = (format) => {
    if (!allSavedPoints.length || busyExport) return;
    setExportError('');
    setPendingExportFormat(format);
  };

  const cancelSavedPointsExport = () => {
    if (busyExport) return;
    setPendingExportFormat(null);
  };

  const performSavedPointsExport = async () => {
    const format = pendingExportFormat;
    if (!format || !allSavedPoints.length) return;

    setBusyExport(format);
    setExportError('');
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
      setPendingExportFormat(null);
      setCompletedExportFormat(format);
    } catch (error) {
      setExportError(error.message || 'Export failed');
    } finally {
      setBusyExport('');
    }
  };

  return (
    <Panel title="Planting Map" subtitle={`${mapped} saved planting points`} initialOpenKey="overview">
      <PanelCard
        title="Overview"
        panelKey="overview"
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
          <div className="stat-card"><div className="stat-label">Mapped Points</div><div className="stat-value">{mapped}</div></div>
          <div className="stat-card"><div className="stat-label">Planned</div><div className="stat-value" style={{ color: 'var(--color-planned)' }}>{planned}</div></div>
          <div className="stat-card"><div className="stat-label">Assigned</div><div className="stat-value" style={{ color: 'var(--color-assigned)' }}>{assigned}</div></div>
          <div className="stat-card"><div className="stat-label">Planted</div><div className="stat-value" style={{ color: 'var(--color-completed)' }}>{planted}</div></div>
          <div className="stat-card"><div className="stat-label">Dead</div><div className="stat-value" style={{ color: '#7f1d1d' }}>{dead}</div></div>
          <div className="stat-card"><div className="stat-label">Skipped</div><div className="stat-value" style={{ color: '#6b7280' }}>{skipped}</div></div>
          <div className="stat-card analytics-unavailable-stat">
            <div className="stat-label">Unavailable</div>
            <div className="stat-value" style={{ color: '#f97316' }}>{unavailable}</div>
            <div className="stat-sub">Points in eroded zones, unavailable for planting.</div>
          </div>
        </div>
      </PanelCard>

      <PanelCard
        title="Legend"
        panelKey="legend"
        defaultOpen={false}
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
          <div className="legend-item"><span className="legend-dot" style={{ background: '#db2777' }} /><span>Rhizophora</span></div>
          <div className="legend-item"><span className="legend-dot" style={{ background: '#2563eb' }} /><span>Assigned</span></div>
          <div className="legend-item"><span className="legend-dot" style={{ background: POINT_STATUS_COLORS.planted }} /><span>Planted</span></div>
          <div className="legend-item"><span className="legend-dot" style={{ background: '#9ca3af' }} /><span>Skipped</span></div>
          <div className="legend-item"><span className="legend-dot" style={{ background: '#f97316' }} /><span>Unavailable for planting (eroded zone)</span></div>
          <div className="legend-item"><span className="legend-dot" style={{ background: '#7f1d1d' }} /><span>Dead (review in Monitoring)</span></div>
          <div className="legend-item"><span className="legend-dot legend-dot-outline" style={{ borderColor: '#16a34a', background: 'rgba(22, 163, 74, 0.16)' }} /><span>Coverage Zone</span></div>
          <div className="legend-item"><span className="legend-dot legend-dot-outline" style={{ borderColor: '#0284c7', background: 'rgba(14, 165, 233, 0.08)' }} /><span>Project Site</span></div>
          <div className="legend-item"><span className="legend-dot legend-dot-outline" style={{ borderColor: '#dc2626' }} /><span>Forbidden Zone</span></div>
          <div className="legend-item"><span className="legend-dot legend-dot-outline" style={{ borderColor: '#ea580c' }} /><span>Eroded Zone</span></div>
          <div className="legend-item"><span className="legend-dot legend-dot-outline" style={{ borderColor: '#f59e0b' }} /><span>Warning Zone</span></div>
        </div>
      </PanelCard>

      <PanelCard
        title="Layers"
        panelKey="layers"
        defaultOpen={false}
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
            <input type="checkbox" checked={layerVisibility.siteZones} onChange={() => toggleLayer('siteZones')} />
            <span>Assignment Zones</span>
          </label>
          <label className="layer-toggle">
            <input type="checkbox" checked={layerVisibility.projectSites} onChange={() => toggleLayer('projectSites')} />
            <span>Project Sites</span>
          </label>
          <label className="layer-toggle">
            <input type="checkbox" checked={layerVisibility.forbidden} onChange={() => toggleLayer('forbidden')} />
            <span>Forbidden Zones</span>
          </label>
          <label className="layer-toggle">
            <input type="checkbox" checked={layerVisibility.eroded} onChange={() => toggleLayer('eroded')} />
            <span>Eroded Zones</span>
          </label>
          <label className="layer-toggle">
            <input type="checkbox" checked={layerVisibility.warnings} onChange={() => toggleLayer('warnings')} />
            <span>Warning Zones</span>
          </label>
        </div>
      </PanelCard>

      <PanelCard
        title="Coverage"
        panelKey="coverage"
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
          <div className="info-row"><span className="info-label">Plantable area</span><span className="info-value">{stats?.total_plantable_m2 ? `${stats.total_plantable_m2.toFixed(1)} m2` : '-'}</span></div>
          <div className="info-row"><span className="info-label">Danger area</span><span className="info-value">{stats?.total_danger_m2 ? `${stats.total_danger_m2.toFixed(1)} m2` : '-'}</span></div>
        </div>
      </PanelCard>

      <PanelCard
        title="Export Saved Points"
        badge={allSavedPoints.length}
        panelKey="export"
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
              onClick={() => requestSavedPointsExport(fmt)}
              disabled={!allSavedPoints.length || Boolean(busyExport)}
            >
              {busyExport === fmt ? 'Exporting...' : fmt.toUpperCase()}
            </button>
          ))}
        </div>
      </PanelCard>

      <Modal
        open={Boolean(pendingExportFormat)}
        title={`Export all saved points as ${pendingExportFormat?.toUpperCase() || ''}?`}
        variant="info"
        confirmLabel={`Download ${pendingExportFormat?.toUpperCase() || ''}`.trim()}
        cancelLabel="Cancel"
        busy={Boolean(busyExport)}
        onConfirm={performSavedPointsExport}
        onCancel={cancelSavedPointsExport}
      >
        <p>
          {allSavedPoints.length} saved planting point{allSavedPoints.length === 1 ? '' : 's'}
          {' '}across all analyses will be bundled into a single
          {' '}<strong>.{pendingExportFormat || 'file'}</strong> download.
        </p>
        <p>The file will be saved to your browser downloads folder.</p>
        {exportError && (
          <p style={{ color: '#dc2626', marginTop: 8 }}>{exportError}</p>
        )}
      </Modal>

      <Modal
        open={Boolean(completedExportFormat)}
        title={`${completedExportFormat?.toUpperCase() || 'File'} downloaded`}
        variant="success"
        confirmLabel="Got it"
        cancelLabel=""
        onConfirm={() => setCompletedExportFormat(null)}
      >
        <p>
          {allSavedPoints.length} planting point{allSavedPoints.length === 1 ? '' : 's'} exported as
          {' '}<strong>mangrovision_all_points.{completedExportFormat}</strong>.
          {' '}Check your downloads folder.
        </p>
      </Modal>
    </Panel>
  );
}
