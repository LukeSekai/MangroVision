import { useEffect, useState } from 'react';
import { TILESET_PATH } from '../config/mapTiles';
import './ResultsOverlay.css';

const ICON_AREA = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="18" height="18" rx="2" />
    <path d="M3 9h18M9 3v18" />
  </svg>
);
const ICON_PERCENT = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <line x1="19" y1="5" x2="5" y2="19" />
    <circle cx="6.5" cy="6.5" r="2.5" />
    <circle cx="17.5" cy="17.5" r="2.5" />
  </svg>
);
const ICON_TREE = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 2l4 6h-3v4h4l4 6H3l4-6h4V8H8z" />
    <path d="M12 18v4" />
  </svg>
);
const ICON_PIN = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M21 10c0 7-9 13-9 13S3 17 3 10a9 9 0 0 1 18 0z" />
    <circle cx="12" cy="10" r="3" />
  </svg>
);
const ICON_LEAF = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M20 4C8 4 4 12 4 20c8 0 16-4 16-16z" />
    <path d="M4 20L14 10" />
  </svg>
);
const ICON_CHIP = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="6" y="6" width="12" height="12" rx="1.5" />
    <path d="M9 2v4M15 2v4M9 18v4M15 18v4M2 9h4M2 15h4M18 9h4M18 15h4" />
  </svg>
);
const ICON_GAUGE = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M12 14a4 4 0 1 0-3.1-6.5" />
    <path d="M12 14l5-5" />
    <path d="M4 20a10 10 0 1 1 16 0" />
  </svg>
);
const ICON_GRID = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <rect x="3" y="3" width="7" height="7" />
    <rect x="14" y="3" width="7" height="7" />
    <rect x="3" y="14" width="7" height="7" />
    <rect x="14" y="14" width="7" height="7" />
  </svg>
);
const ICON_CLOCK = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="9" />
    <polyline points="12 7 12 12 15 14" />
  </svg>
);
const ICON_RULER = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M3 17L17 3l4 4L7 21z" />
    <path d="M7 13l2 2M11 9l2 2M15 5l2 2" />
  </svg>
);
const ICON_WARN = (
  <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <path d="M10.3 3.86l-8.1 14.02A2 2 0 0 0 3.94 21h16.12a2 2 0 0 0 1.74-3.12L13.7 3.86a2 2 0 0 0-3.4 0z" />
    <line x1="12" y1="9" x2="12" y2="13" />
    <line x1="12" y1="17" x2="12.01" y2="17" />
  </svg>
);
const ICON_TARGET = (
  <svg width="12" height="12" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
    <circle cx="12" cy="12" r="9" />
    <circle cx="12" cy="12" r="5" />
    <circle cx="12" cy="12" r="1.5" />
  </svg>
);

function fmt(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toFixed(digits);
}

function formatProcessingTime(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return '—';
  const value = Number(seconds);
  if (value < 60) return `${value.toFixed(1)}s`;
  const mins = Math.floor(value / 60);
  const secs = Math.round(value - mins * 60);
  return `${mins}m ${secs}s`;
}

function prettyDetectionMode(mode) {
  if (!mode) return '—';
  if (mode === 'ai') return 'AI (Detectree2)';
  if (mode === 'hsv') return 'HSV Colour';
  return String(mode).toUpperCase();
}

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

function downloadText(content, fileName, mime = 'text/plain;charset=utf-8') {
  downloadBlob(new Blob([content], { type: mime }), fileName);
}

function downloadDataUrl(dataUrl, fileName) {
  const link = document.createElement('a');
  link.href = dataUrl;
  link.download = fileName;
  document.body.appendChild(link);
  link.click();
  link.remove();
}

function MetricItem({ icon, label, children, primary = false }) {
  return (
    <div className="metric-item">
      <span className="metric-icon">{icon}</span>
      <div className="metric-info">
        <span className="metric-label">{label}</span>
        <span className={`metric-value ${primary ? 'metric-value-primary' : ''}`}>{children}</span>
      </div>
    </div>
  );
}

export default function ResultsOverlay({
  open,
  result,
  originalPreview,
  saving,
  saved,
  saveError,
  onClose,
  onSave,
  onExport,
}) {
  const [visible, setVisible] = useState(false);
  const [closing, setClosing] = useState(false);
  const [busyExport, setBusyExport] = useState(null);
  const [exportError, setExportError] = useState('');
  const [showCoords, setShowCoords] = useState(false);

  useEffect(() => {
    if (open) {
      setVisible(true);
      setClosing(false);
    } else if (visible) {
      setClosing(true);
      const timer = setTimeout(() => {
        setVisible(false);
        setClosing(false);
      }, 220);
      return () => clearTimeout(timer);
    }
    return undefined;
  }, [open, visible]);

  useEffect(() => {
    if (!visible) return undefined;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    const onKey = (event) => {
      if (event.key === 'Escape') onClose();
    };
    window.addEventListener('keydown', onKey);
    return () => {
      document.body.style.overflow = previousOverflow;
      window.removeEventListener('keydown', onKey);
    };
  }, [visible, onClose]);

  if (!visible || !result) return null;

  const metrics = result.metrics || {};
  const metadata = result.metadata || {};
  const mapInfo = result.map || {};
  const warnings = result.messages?.warnings || [];
  const infos = result.messages?.info || [];
  const overlaps = result.overlaps?.analyses || [];
  const coordinateRows = mapInfo.coordinates || [];
  const canopyAreaM2 = metrics.canopy_area_m2 ?? 0;
  const exportsDisabled = !result.exports?.waypoints?.length;
  const baseName = (result.uploaded_file_name || 'mangrovision').replace(/\.[^/.]+$/, '');

  const handleBackdropClick = (event) => {
    if (event.target === event.currentTarget) onClose();
  };

  const runExport = async (format) => {
    if (!onExport) return;
    setBusyExport(format);
    setExportError('');
    try {
      await onExport(format);
    } catch (err) {
      setExportError(err?.message || `Could not export ${format.toUpperCase()}`);
    } finally {
      setBusyExport(null);
    }
  };

  const handleDownloadViz = () => {
    const dataUrl = result.images?.visualization_data_url;
    if (!dataUrl) return;
    downloadDataUrl(dataUrl, `${baseName}_visualization.png`);
  };

  const handleDownloadJson = () => {
    const payload = result.exports?.json_results;
    if (!payload) return;
    downloadText(JSON.stringify(payload, null, 2), `${baseName}_results.json`, 'application/json');
  };

  const matchSuccess = mapInfo.match?.success;
  const matchConfidence = mapInfo.match?.confidence;
  const safePointCount = (mapInfo.safe_points_geojson?.features || []).length;
  const forbiddenFilteredCount = metrics.forbidden_filtered_count ?? 0;
  const erodedFilteredCount = metrics.eroded_filtered_count ?? 0;
  const orthophotoCanopyFilteredCount = metrics.orthophoto_canopy_filtered_count ?? 0;
  const clippedOutsideCount = metrics.clipped_outside_orthophoto ?? 0;
  const duplicateFilteredCount = metrics.duplicate_filtered_count ?? 0;
  const displayedHeading = metadata.detected_heading ?? metadata.camera_heading;
  const displayedHeadingSource = matchSuccess ? 'Auto-aligned orthophoto' : metadata.heading_source;
  const visibleTileset = TILESET_PATH
    ? TILESET_PATH.split('/').map((segment) => decodeURIComponent(segment)).join('/')
    : '';

  return (
    <div
      className={`rs-backdrop ${closing ? 'rs-closing' : ''}`}
      role="dialog"
      aria-modal="true"
      aria-label="Analysis results"
      onMouseDown={handleBackdropClick}
    >
      <div className={`rs-modal ${closing ? 'rs-closing' : ''}`}>
        <button type="button" className="rs-close" onClick={onClose} aria-label="Close results">
          <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
            <line x1="18" y1="6" x2="6" y2="18" />
            <line x1="6" y1="6" x2="18" y2="18" />
          </svg>
        </button>

        <header className="rs-header">
          <div>
            <div className="rs-eyebrow">Analysis Complete</div>
            <h2 className="rs-title">{result.uploaded_file_name || 'Drone Image'}</h2>
          </div>
        </header>

        <div className="rs-layout">
          <section className="rs-image-col">
            <div className="rs-image-card">
              <div className="rs-image-label">Detection Overlay</div>
              {result.images?.visualization_data_url ? (
                <img
                  src={result.images.visualization_data_url}
                  alt="Detected canopy visualization"
                  className="rs-image"
                />
              ) : (
                <div className="rs-image-placeholder">
                  Preview image is only kept for the active session.
                </div>
              )}
            </div>
            <div className="rs-image-card">
              <div className="rs-image-label">Original Image</div>
              {/*
                Prefer the backend's original_data_url over the browser-side
                FileReader preview because OpenCV (backend) ignores EXIF
                Orientation while the browser respects it. Mixing the two
                in adjacent panels makes the AI overlay look "shifted"
                relative to the original even though the pixels match. Use
                the backend image so both panels show the exact pixels the
                detector processed, in the same orientation. The FileReader
                preview is kept only as a last-resort fallback for the
                pre-result loading state.
              */}
              {result.images?.original_data_url || originalPreview ? (
                <img
                  src={result.images?.original_data_url || originalPreview}
                  alt="Original drone input"
                  className="rs-image"
                />
              ) : (
                <div className="rs-image-placeholder">
                  Original drone image was not persisted.
                </div>
              )}
            </div>
            <p className="rs-image-caption">
              Legend: purple canopy, red danger buffer, light green planting buffer, orange overlap warning, dark green planting core.
            </p>
          </section>

          <section className="rs-metrics-col">
            {warnings.length > 0 && (
              <div className="warning-badges rs-full">
                {warnings.map((warning, index) => (
                  <div key={`ov-warn-${index}`} className="warning-badge">
                    <span className="warning-badge-icon">{ICON_WARN}</span>
                    <span className="warning-badge-text">{warning}</span>
                  </div>
                ))}
              </div>
            )}

            <div className="metric-card metric-card-primary">
              <div className="metric-card-title">Coverage</div>
              <div className="metric-grid metric-grid-primary">
                <MetricItem icon={ICON_AREA} label="Total Mangrove Area" primary>
                  {fmt(canopyAreaM2, 1)} m²
                  <span className="metric-value-sub">({fmt(canopyAreaM2 / 10000, 3)} ha)</span>
                </MetricItem>
                <MetricItem icon={ICON_PERCENT} label="Canopy Coverage" primary>
                  {fmt(metrics.canopy_coverage_pct, 2)}%
                </MetricItem>
                <MetricItem icon={ICON_TREE} label="Canopy Components" primary>
                  {metrics.canopy_count ?? 0}
                </MetricItem>
              </div>
            </div>

            <div className="metric-card metric-card-primary">
              <div className="metric-card-title">Planting</div>
              <div className="metric-grid metric-grid-primary">
                <MetricItem icon={ICON_PIN} label="Safe Planting Points" primary>
                  {metrics.safe_hexagon_count ?? metrics.hexagon_count ?? 0}
                </MetricItem>
                <MetricItem icon={ICON_LEAF} label="Plantable Area" primary>
                  {fmt(metrics.plantable_area_m2, 1)} m²
                  <span className="metric-value-sub">({fmt(metrics.plantable_percentage, 1)}%)</span>
                </MetricItem>
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-card-title">Detection</div>
              <div className="metric-grid">
                <MetricItem icon={ICON_CHIP} label="Method">
                  {prettyDetectionMode(result.detection_mode)}
                </MetricItem>
                <MetricItem icon={ICON_TARGET} label="Model">
                  {metrics.model_name || (result.detection_mode === 'ai' ? 'Detectree2' : 'HSV')}
                </MetricItem>
                <MetricItem icon={ICON_GAUGE} label="Confidence">
                  {metrics.ai_confidence_threshold !== undefined
                    ? fmt(metrics.ai_confidence_threshold, 2)
                    : '—'}
                </MetricItem>
                <MetricItem icon={ICON_TREE} label="AI Instances">
                  {metrics.ai_instance_count ?? metrics.canopy_count ?? 0}
                </MetricItem>
                <MetricItem icon={ICON_PIN} label="Ortho Match">
                  {matchSuccess
                    ? `${Math.round((matchConfidence || 0) * 100)}%`
                    : 'Fallback'}
                </MetricItem>
                <MetricItem icon={ICON_WARN} label="Below Conf.">
                  {metrics.ai_below_confidence_detections ?? 0}
                </MetricItem>
                <MetricItem icon={ICON_AREA} label="Max Filtered">
                  {metrics.ai_rejected_too_large_detections ?? 0}
                </MetricItem>
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-card-title">Processing</div>
              <div className="metric-grid">
                <MetricItem icon={ICON_GRID} label="Tiles Analyzed">
                  {metrics.tile_count ?? 0}
                </MetricItem>
                <MetricItem icon={ICON_CLOCK} label="Run Time">
                  {formatProcessingTime(metrics.processing_time_sec)}
                </MetricItem>
                <MetricItem icon={ICON_RULER} label="GSD">
                  {metrics.gsd_m_per_pixel
                    ? `${fmt(metrics.gsd_m_per_pixel * 100, 2)} cm/px`
                    : '—'}
                </MetricItem>
                <MetricItem icon={ICON_AREA} label="Coverage">
                  {metrics.coverage_m
                    ? `${fmt(metrics.coverage_m[0], 1)} × ${fmt(metrics.coverage_m[1], 1)} m`
                    : '—'}
                </MetricItem>
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-card-title">Metadata</div>
              <div className="metric-grid">
                <MetricItem icon={ICON_PIN} label="GPS">
                  {metadata.gps_valid ? 'Valid (EXIF)' : 'Missing / Fallback'}
                </MetricItem>
                <MetricItem icon={ICON_CHIP} label="Drone">
                  {metadata.camera?.model || result.parameters?.drone_to_use?.replace(/_/g, ' ') || '—'}
                </MetricItem>
                <MetricItem icon={ICON_RULER} label="Altitude">
                  {metrics.altitude_m ? `${fmt(metrics.altitude_m, 1)} m` : '—'}
                </MetricItem>
                <MetricItem icon={ICON_TARGET} label="Heading">
                  {displayedHeading !== undefined && displayedHeading !== null
                    ? `${fmt(displayedHeading, 1)}°`
                    : '—'}
                  {displayedHeadingSource && (
                    <span className="metric-value-sub">{displayedHeadingSource}</span>
                  )}
                </MetricItem>
              </div>
              {infos.length > 0 && (
                <div className="message-stack">
                  {infos.map((info, index) => (
                    <div key={`ov-info-${index}`} className="analysis-message analysis-info">
                      {info}
                    </div>
                  ))}
                </div>
              )}
              {overlaps.length > 0 && (
                <div className="analysis-sublist">
                  <div className="analysis-subtitle">Overlapping Prior Analyses</div>
                  {overlaps.map((ov, index) => (
                    <div key={`ov-overlap-${index}`} className="analysis-subrow">
                      <span>{ov.image_name || `Analysis ${ov.analysis_id}`}</span>
                      <span>{ov.overlap_count} pt{ov.overlap_count === 1 ? '' : 's'}</span>
                    </div>
                  ))}
                </div>
              )}
            </div>

            <div className="metric-card">
              <div className="metric-card-title">Geotagged Map Review</div>
              <div className="metric-grid">
                <MetricItem icon={ICON_PIN} label="Safe Points">
                  {safePointCount}
                </MetricItem>
                <MetricItem icon={ICON_WARN} label="Forbidden Filtered">
                  {forbiddenFilteredCount}
                </MetricItem>
                <MetricItem icon={ICON_WARN} label="Eroded Filtered">
                  {erodedFilteredCount}
                </MetricItem>
                <MetricItem icon={ICON_WARN} label="Canopy Recheck">
                  {orthophotoCanopyFilteredCount}
                </MetricItem>
                <MetricItem icon={ICON_WARN} label="Clipped Outside">
                  {clippedOutsideCount}
                </MetricItem>
                <MetricItem icon={ICON_WARN} label="Duplicates Skipped">
                  {duplicateFilteredCount}
                </MetricItem>
                <MetricItem icon={ICON_TARGET} label="Alignment">
                  {matchSuccess
                    ? `Auto (${Math.round((matchConfidence || 0) * 100)}%)`
                    : 'Heading fallback'}
                  {matchSuccess && (
                    <span className="metric-value-sub">
                      {mapInfo.match?.ortho_name || 'Matched orthophoto'}
                      {mapInfo.match?.center_drift_m !== undefined && mapInfo.match?.center_drift_m !== null
                        ? ` · drift ${fmt(mapInfo.match.center_drift_m, 2)} m`
                        : ''}
                      {mapInfo.match?.center_offset_east_m !== undefined && mapInfo.match?.center_offset_east_m !== null
                        && mapInfo.match?.center_offset_north_m !== undefined && mapInfo.match?.center_offset_north_m !== null
                        ? ` · E ${fmt(mapInfo.match.center_offset_east_m, 2)} m, N ${fmt(mapInfo.match.center_offset_north_m, 2)} m`
                        : ''}
                      {visibleTileset ? ` · tiles ${visibleTileset}` : ''}
                    </span>
                  )}
                </MetricItem>
              </div>
            </div>

            {!exportsDisabled && (
              <div className="metric-card rs-full">
                <div className="metric-card-title">Exports</div>
                <div className="export-grid">
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={handleDownloadViz}
                    disabled={!result.images?.visualization_data_url}
                  >
                    Visualization PNG
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={handleDownloadJson}
                    disabled={!result.exports?.json_results}
                  >
                    JSON Data
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => runExport('csv')}
                    disabled={Boolean(busyExport)}
                  >
                    {busyExport === 'csv' ? 'Exporting…' : 'CSV'}
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => runExport('gpx')}
                    disabled={Boolean(busyExport)}
                  >
                    {busyExport === 'gpx' ? 'Exporting…' : 'GPX'}
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => runExport('kml')}
                    disabled={Boolean(busyExport)}
                  >
                    {busyExport === 'kml' ? 'Exporting…' : 'KML'}
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => runExport('geojson')}
                    disabled={Boolean(busyExport)}
                  >
                    {busyExport === 'geojson' ? 'Exporting…' : 'GeoJSON'}
                  </button>
                </div>
                {exportError && (
                  <div className="process-error" style={{ marginTop: 8 }}>{exportError}</div>
                )}
              </div>
            )}

            {coordinateRows.length > 0 && (
              <div className="metric-card rs-full">
                <div
                  className="metric-card-title"
                  style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}
                >
                  <span>Planting Coordinates ({coordinateRows.length})</span>
                  <button
                    type="button"
                    className="btn btn-ghost btn-sm"
                    onClick={() => setShowCoords((prev) => !prev)}
                  >
                    {showCoords ? 'Hide' : 'Show'}
                  </button>
                </div>
                {showCoords && (
                  <div className="coordinate-table-wrap">
                    <table className="coordinate-table">
                      <thead>
                        <tr>
                          <th>#</th>
                          <th>Latitude</th>
                          <th>Longitude</th>
                          <th>Pixel</th>
                        </tr>
                      </thead>
                      <tbody>
                        {coordinateRows.map((row) => (
                          <tr key={`ov-coord-${row.point_num}`}>
                            <td>{row.point_num}</td>
                            <td>{fmt(row.latitude, 6)}</td>
                            <td>{fmt(row.longitude, 6)}</td>
                            <td>
                              {row.pixel_x}, {row.pixel_y}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </div>
            )}

            {result.can_save && (
              <div className="rs-save-block rs-full">
                {saved ? (
                  <div className="rs-save-success">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                    Saved ✓ — this analysis is now in Map Analytics.
                  </div>
                ) : (
                  <>
                    <button
                      type="button"
                      className="rs-save-btn"
                      onClick={onSave}
                      disabled={saving}
                    >
                      {saving ? (
                        <>
                          <span className="rs-spinner" aria-hidden="true" />
                          Saving…
                        </>
                      ) : (
                        'Save Analysis'
                      )}
                    </button>
                    {saveError && (
                      <div className="rs-save-error">
                        <span>{saveError}</span>
                        <button type="button" className="rs-retry" onClick={onSave} disabled={saving}>
                          Retry
                        </button>
                      </div>
                    )}
                  </>
                )}
              </div>
            )}
          </section>
        </div>
      </div>
    </div>
  );
}
