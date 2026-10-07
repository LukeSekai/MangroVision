import { useEffect, useState } from 'react';
import { createPortal } from 'react-dom';
import Modal from '../components/Modal';
import Logo from '../components/Logo';
import './ResultsOverlay.css';

// Per-format metadata for the confirmation/success modals. Keeping it inline
// next to the consumer so the labels and descriptions stay close to the
// buttons that trigger them.
const EXPORT_FORMAT_META = {
  jpeg: {
    label: 'Visualization JPEG',
    extension: 'jpg',
    description: 'Annotated drone image with detected canopies, danger buffers, and planting points.',
  },
  json: {
    label: 'JSON Data',
    extension: 'json',
    description: 'Raw analysis result for programmatic use or thesis appendix.',
  },
  csv: {
    label: 'CSV',
    extension: 'csv',
    description: 'Spreadsheet-friendly waypoint table (latitude, longitude, point #, area).',
  },
  gpx: {
    label: 'GPX',
    extension: 'gpx',
    description: 'GPX waypoints that load on handheld GPS units and field-mapping apps.',
  },
  kml: {
    label: 'KML',
    extension: 'kml',
    description: 'KML overlay that opens in Google Earth and Google My Maps.',
  },
  geojson: {
    label: 'GeoJSON',
    extension: 'geojson',
    description: 'GeoJSON for QGIS, Mapbox, Leaflet, and other GIS tools.',
  },
};

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

function fmtCount(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '—';
  return Number(value).toLocaleString();
}

function formatProcessingTime(seconds) {
  if (seconds === null || seconds === undefined || Number.isNaN(Number(seconds))) return '—';
  const value = Number(seconds);
  if (value < 60) return `${value.toFixed(1)}s`;
  const mins = Math.floor(value / 60);
  const secs = Math.round(value - mins * 60);
  return `${mins}m ${secs}s`;
}

function displayDroneModel(metadata, parameters) {
  const model = metadata.camera?.model || parameters?.drone_to_use;
  if (!model) return '—';
  // FC7703 is the camera identifier in this project's DJI Mini 4K photos.
  if (['FC7703', 'DJI_FC7703', 'DJI_MINI_4K'].includes(String(model).toUpperCase())) {
    return 'DJI Mini 4K';
  }
  return String(model).replace(/_/g, ' ');
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
  onBack,
  onViewMap,
  canViewMap = false,
  onSave,
  onExport,
}) {
  const [visible, setVisible] = useState(open);
  if (open && !visible) setVisible(true);
  const closing = visible && !open;
  const [busyExport, setBusyExport] = useState(null);
  const [exportError, setExportError] = useState('');
  const [showCoords, setShowCoords] = useState(false);

  // Two-step export flow:
  //   pendingExportFormat — user clicked an export button, confirmation modal asks
  //   completedExportFormat — file finished downloading, success modal acknowledges
  const [pendingExportFormat, setPendingExportFormat] = useState(null);
  const [completedExportFormat, setCompletedExportFormat] = useState(null);

  useEffect(() => {
    if (open || !visible) return undefined;
    const timer = setTimeout(() => setVisible(false), 220);
    return () => clearTimeout(timer);
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
  const overlaps = result.overlaps?.analyses || [];
  const coordinateRows = mapInfo.coordinates || [];
  const canopyAreaM2 = metrics.canopy_area_m2;
  const canopyAreaHa = Number.isFinite(Number(canopyAreaM2)) ? Number(canopyAreaM2) / 10000 : null;
  const exportsDisabled = !result.exports?.waypoints?.length;
  const baseName = (result.uploaded_file_name || 'mangrovision').replace(/\.[^/.]+$/, '');

  const handleBackdropClick = (event) => {
    if (event.target === event.currentTarget) onClose();
  };

  // Each export now goes through:
  //   1. requestExport(format)  → opens the confirmation modal (no I/O yet)
  //   2. performExport()        → runs after the user confirms; downloads the file
  //   3. completedExportFormat  → success modal opens automatically
  const requestExport = (format) => {
    if (!format || !EXPORT_FORMAT_META[format]) return;
    setExportError('');
    setPendingExportFormat(format);
  };

  const cancelExport = () => {
    if (busyExport) return;
    setPendingExportFormat(null);
  };

  const performExport = async () => {
    const format = pendingExportFormat;
    if (!format) return;

    setBusyExport(format);
    setExportError('');
    try {
      if (format === 'jpeg') {
        const dataUrl = result.images?.visualization_image_url;
        if (!dataUrl) throw new Error('Visualization image is not available.');
        downloadDataUrl(dataUrl, `${baseName}_visualization.jpg`);
      } else if (format === 'json') {
        const payload = result.exports?.json_results;
        if (!payload) throw new Error('JSON results payload is not available.');
        downloadText(
          JSON.stringify(payload, null, 2),
          `${baseName}_results.json`,
          'application/json',
        );
      } else {
        if (!onExport) throw new Error('Export handler is not wired up.');
        await onExport(format);
      }
      setPendingExportFormat(null);
      setCompletedExportFormat(format);
    } catch (err) {
      setExportError(err?.message || `Could not export ${format.toUpperCase()}`);
    } finally {
      setBusyExport(null);
    }
  };

  const closeCompletedExportModal = () => setCompletedExportFormat(null);

  const hasMatchInfo = mapInfo.match && typeof mapInfo.match.success === 'boolean';
  const matchSuccess = hasMatchInfo && mapInfo.match.success;
  const vegetationHeadingRefined = [
    'vegetation_heading_metric_anchor',
    'vegetation_heading_scale_metric_anchor',
  ].includes(mapInfo.match?.projection_rotation_source);
  const vegetationScaleRefined = mapInfo.match?.projection_rotation_source
    === 'vegetation_scale_metric_anchor';
  const edgeHeadingRefined = mapInfo.match?.projection_rotation_source
    === 'exif_heading_metric_edge_alignment';
  const displayedHeading = metadata.detected_heading ?? metadata.camera_heading;
  const displayedHeadingSource = matchSuccess
    ? vegetationScaleRefined
      ? 'DJI EXIF heading with vegetation-calibrated footprint'
      : vegetationHeadingRefined
      ? 'Vegetation-refined orthophoto alignment'
      : edgeHeadingRefined
      ? 'DJI EXIF heading with orthophoto edge correction'
      : 'Auto-aligned orthophoto'
    : metadata.heading_source;

  return createPortal(
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

        {(onBack || onViewMap) && <nav className="rs-history-navigation" aria-label="Analysis navigation">
          {onBack && <button type="button" className="btn btn-ghost btn-sm" onClick={onBack}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true"><path d="m12 19-7-7 7-7M5 12h14" /></svg>
            Back to history
          </button>}
          {onViewMap && <button type="button" className="btn btn-secondary btn-sm rs-view-map" onClick={onViewMap}
            disabled={!canViewMap} aria-describedby={!canViewMap ? 'rs-map-unavailable' : undefined}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="m3 5 6-2 6 2 6-2v16l-6 2-6-2-6 2V5Zm6-2v16m6-14v16" /></svg>
            View area in map
          </button>}
          {onViewMap && !canViewMap && <span id="rs-map-unavailable" className="rs-map-unavailable">No map location was saved for this analysis.</span>}
        </nav>}
        <header className="rs-header">
          <Logo variant="icon" size={32} className="rs-header-logo" alt="MangroVision" />
          <div>
            <div className="rs-eyebrow">Analysis Complete</div>
            <h2 className="rs-title">{result.uploaded_file_name || 'Drone Image'}</h2>
          </div>
        </header>

        <div className="rs-layout">
          <section className="rs-image-col">
            <div className="rs-image-card">
              <div className="rs-image-label">Detection Overlay</div>
              {result.images?.visualization_preview_url || result.images?.visualization_image_url ? (
                <img
                  src={result.images.visualization_preview_url || result.images.visualization_image_url}
                  alt="Detected canopy visualization"
                  className="rs-image"
                  decoding="async"
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
                Prefer the backend's private original image URL over the browser-side
                FileReader preview because OpenCV (backend) ignores EXIF
                Orientation while the browser respects it. Mixing the two
                in adjacent panels makes the AI overlay look "shifted"
                relative to the original even though the pixels match. Use
                the backend image so both panels show the exact pixels the
                detector processed, in the same orientation. The FileReader
                preview is kept only as a last-resort fallback for the
                pre-result loading state.
              */}
              {result.images?.original_preview_url || result.images?.original_image_url || originalPreview ? (
                <img
                  src={result.images?.original_preview_url || result.images?.original_image_url || originalPreview}
                  alt="Original drone input"
                  className="rs-image"
                  loading="lazy"
                  decoding="async"
                />
              ) : (
                <div className="rs-image-placeholder">
                  Original drone image was not persisted.
                </div>
              )}
            </div>
            <p className="rs-image-caption">
              Legend: purple canopy, red danger buffer, green available planting hexagons, orange eroded/unavailable planting hexagons.
            </p>
          </section>

          <section className="rs-metrics-col">
            <div className="metric-card metric-card-primary">
              <div className="metric-card-title">Coverage</div>
              <div className="metric-grid metric-grid-primary">
                <MetricItem icon={ICON_AREA} label="Detected Canopy Area" primary>
                  {fmt(canopyAreaM2, 1)} m²
                  <span className="metric-value-sub">({fmt(canopyAreaHa, 3)} ha)</span>
                </MetricItem>
                <MetricItem icon={ICON_PERCENT} label="Canopy Coverage" primary>
                  {fmt(metrics.canopy_coverage_pct, 2)}%
                </MetricItem>
              </div>
            </div>

            <div className="metric-card metric-card-primary">
              <div className="metric-card-title">Planting</div>
              <div className="metric-grid metric-grid-primary">
                <MetricItem icon={ICON_PIN} label={result.repeat_image?.points_not_added ? 'New Planting Points' : 'Safe Planting Points'} primary>
                  {metrics.safe_hexagon_count ?? metrics.hexagon_count ?? 0}
                </MetricItem>
                <MetricItem icon={ICON_LEAF} label="Plantable Area" primary>
                  {fmt(metrics.plantable_area_m2, 1)} m²
                  <span className="metric-value-sub">({fmt(metrics.plantable_percentage, 1)}%)</span>
                </MetricItem>
              </div>
            </div>

            <div className="metric-card">
              <div className="metric-card-title">Processing</div>
              <div className="metric-grid">
                <MetricItem icon={ICON_GRID} label="Tiles Analyzed">
                  {fmtCount(metrics.tile_count)}
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
                  {displayDroneModel(metadata, result.parameters)}
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

            {!exportsDisabled && (
              <div className="metric-card rs-full">
                <div className="metric-card-title">Exports</div>
                <div className="export-grid">
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => requestExport('jpeg')}
                    disabled={!result.images?.visualization_image_url || Boolean(busyExport)}
                  >
                    Visualization JPEG
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => requestExport('json')}
                    disabled={!result.exports?.json_results || Boolean(busyExport)}
                  >
                    JSON Data
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => requestExport('csv')}
                    disabled={Boolean(busyExport)}
                  >
                    {busyExport === 'csv' ? 'Exporting…' : 'CSV'}
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => requestExport('gpx')}
                    disabled={Boolean(busyExport)}
                  >
                    {busyExport === 'gpx' ? 'Exporting…' : 'GPX'}
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => requestExport('kml')}
                    disabled={Boolean(busyExport)}
                  >
                    {busyExport === 'kml' ? 'Exporting…' : 'KML'}
                  </button>
                  <button
                    type="button"
                    className="btn btn-secondary btn-sm"
                    onClick={() => requestExport('geojson')}
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

            {result.repeat_image?.points_not_added && <div className="rs-repeat-notice rs-full" role="status">
              <strong>Repeat image analysis</strong>
              <p>No planting points will be added. Existing planting records are kept. You can save this result in analysis history.</p>
            </div>}

            {result.can_save && (
              <div className="rs-save-block rs-full">
                {saved ? (
                  <div className="rs-save-success">
                    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.5" strokeLinecap="round" strokeLinejoin="round">
                      <polyline points="20 6 9 17 4 12" />
                    </svg>
                    Saved - this analysis is now in Analyze Image history.
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

      <Modal
        open={Boolean(pendingExportFormat)}
        title={`Download as ${EXPORT_FORMAT_META[pendingExportFormat]?.label || pendingExportFormat?.toUpperCase()}?`}
        variant="info"
        confirmLabel={`Download ${EXPORT_FORMAT_META[pendingExportFormat]?.extension?.toUpperCase() || ''}`.trim()}
        cancelLabel="Cancel"
        busy={Boolean(busyExport)}
        onConfirm={performExport}
        onCancel={cancelExport}
      >
        <p>{EXPORT_FORMAT_META[pendingExportFormat]?.description}</p>
        <p>The file will be saved to your browser's downloads folder.</p>
        {exportError && (
          <p style={{ color: '#dc2626', marginTop: 8 }}>{exportError}</p>
        )}
      </Modal>

      <Modal
        open={Boolean(completedExportFormat)}
        title={`${EXPORT_FORMAT_META[completedExportFormat]?.label || completedExportFormat?.toUpperCase() || 'File'} downloaded`}
        variant="success"
        confirmLabel="Got it"
        cancelLabel=""
        onConfirm={closeCompletedExportModal}
      >
        <p>
          The {EXPORT_FORMAT_META[completedExportFormat]?.label || completedExportFormat?.toUpperCase()}
          {' '}file has been saved to your downloads folder.
        </p>
      </Modal>
    </div>,
    document.body,
  );
}
