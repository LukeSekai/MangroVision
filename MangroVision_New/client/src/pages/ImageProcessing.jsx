import { useEffect, useRef, useState } from 'react';
import { Panel, PanelCard } from '../components/Panel';
import Modal from '../components/Modal';
import Logo from '../components/Logo';
import { useAuthStore } from '../stores/authStore';
import { useMapStore } from '../stores/mapStore';
import { useProcessingStore } from '../stores/processingStore';
import ResultsOverlay from './ResultsOverlay';
import './ResultsOverlay.css';
import './ImageProcessing.css';
import './MapAnalytics.css';

const API = import.meta.env.VITE_API_BASE || '';

const DEFAULT_ALTITUDE = 6.0;
const DEFAULT_DRONE_MODEL = 'GENERIC_4K';
const DEFAULT_AI_CONFIDENCE = 0.80;

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

export default function ImageProcessing() {
  const user = useAuthStore((s) => s.user);
  const setCurrentAnalysis = useMapStore((s) => s.setCurrentAnalysis);
  const clearCurrentAnalysis = useMapStore((s) => s.clearCurrentAnalysis);
  const setSavedAnalysisBoundary = useMapStore((s) => s.setSavedAnalysisBoundary);
  const appendSavedAnalysis = useMapStore((s) => s.appendSavedAnalysis);
  const fetchPoints = useMapStore((s) => s.fetchPoints);
  const fetchStats = useMapStore((s) => s.fetchStats);
  const stats = useMapStore((s) => s.stats);

  // All processing state lives in the store so it survives navigation.
  const processing = useProcessingStore((s) => s.processing);
  const stage = useProcessingStore((s) => s.stage);
  const progress = useProcessingStore((s) => s.progress);
  const result = useProcessingStore((s) => s.result);
  const error = useProcessingStore((s) => s.error);
  const saving = useProcessingStore((s) => s.saving);
  const saveError = useProcessingStore((s) => s.saveError);
  const overlayOpen = useProcessingStore((s) => s.overlayOpen);
  const previewUrl = useProcessingStore((s) => s.previewUrl);
  const storedFileName = useProcessingStore((s) => s.fileName);
  const startProcess = useProcessingStore((s) => s.startProcess);
  const saveCurrentAnalysis = useProcessingStore((s) => s.saveCurrentAnalysis);
  const resetProcessing = useProcessingStore((s) => s.reset);
  const setOverlayOpen = useProcessingStore((s) => s.setOverlayOpen);

  const fileRef = useRef(null);
  const preflightRequestRef = useRef(0);
  const [file, setFile] = useState(null);
  const [locationCheck, setLocationCheck] = useState(null);
  const [locationChecking, setLocationChecking] = useState(false);
  const [locationCheckError, setLocationCheckError] = useState('');
  const [locationBlockModalOpen, setLocationBlockModalOpen] = useState(false);
  const [partialConfirmOpen, setPartialConfirmOpen] = useState(false);
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const [loadingAnalysisId, setLoadingAnalysisId] = useState(null);
  const [analysisLoadError, setAnalysisLoadError] = useState('');
  const [pendingDeleteId, setPendingDeleteId] = useState(null);
  const [deleteBusy, setDeleteBusy] = useState(false);
  const [deleteError, setDeleteError] = useState('');
  // Species choice drives the planting-point spacing. The two supported
  // mangrove species have different field spacing requirements:
  //   - Bungalon    → 1.0 m between adjacent plants
  //   - Rhizophora  → 2.0 m between adjacent plants
  // The pipeline uses an edge-share (flat-top) hex tessellation, where the
  // nearest-neighbour distance is hexagon_size * sqrt(3). To hit the species
  // target spacing T exactly we set hexagon_size = T / sqrt(3).
  const [species, setSpecies] = useState('rhizophora');
  const [clearModalOpen, setClearModalOpen] = useState(false);

  const SPECIES_SPACING_M = { bungalon: 1.0, rhizophora: 2.0 };
  const targetSpacingM = SPECIES_SPACING_M[species] ?? 2.0;
  const hexagonSize = Number((targetSpacingM / Math.sqrt(3)).toFixed(4));

  // Canopy danger buffer is fixed at 2 m regardless of species. The buffer
  // represents the no-plant exclusion zone around detected canopies — that
  // distance is set by canopy/tree biology, not by the planting spacing of
  // the new seedlings, so it should not shrink to 1 m when bungalon is
  // selected. Users can still override in the input below.
  const [canopyBuffer, setCanopyBuffer] = useState(2.0);

  const [uploadOpen, setUploadOpen] = useState(true);
  const [configOpen, setConfigOpen] = useState(true);
  const [historyOpen, setHistoryOpen] = useState(false);

  const handleHistoryOpenChange = (next) => {
    setHistoryOpen(next);
    if (next) {
      setUploadOpen(false);
      setConfigOpen(false);
    }
  };

  const handleUploadOpenChange = (next) => {
    setUploadOpen(next);
    if (next) {
      setConfigOpen(true);
      setHistoryOpen(false);
    }
  };

  // The preview shown in the upload panel: prefer the freshly-selected local
  // file's preview if one exists; otherwise rehydrate from the store so users
  // who navigate back mid-processing still see their thumbnail.
  const preview = previewUrl;
  const displayedFileName = file?.name || storedFileName;
  const analyses = stats?.analyses || [];
  const pendingDeleteAnalysis = analyses.find((analysis) => analysis.id === pendingDeleteId);

  useEffect(() => {
    fetchStats();
  }, [fetchStats]);

  const inspectSelectedImage = async (selectedFile) => {
    const requestId = preflightRequestRef.current + 1;
    preflightRequestRef.current = requestId;
    setLocationChecking(true);
    setLocationCheck(null);
    setLocationCheckError('');
    setLocationBlockModalOpen(false);
    setPartialConfirmOpen(false);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('altitude', String(DEFAULT_ALTITUDE));
      formData.append('drone_model', DEFAULT_DRONE_MODEL);
      const response = await fetch(`${API}/api/analyses/preflight`, {
        method: 'POST',
        body: formData,
      });
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) {
        throw new Error(payload.detail || 'Could not verify this image on the GIS map.');
      }
      if (preflightRequestRef.current !== requestId) return;
      setLocationCheck(payload);
      setLocationBlockModalOpen(
        payload.status === 'outside' || payload.status === 'no_gps',
      );
      if (payload.can_process && payload.map?.available) {
        setCurrentAnalysis({
          preflight: true,
          uploaded_file_name: selectedFile.name,
          map: payload.map,
          metadata: {
            image_center_lat: payload.latitude,
            image_center_lon: payload.longitude,
            location_preflight: payload,
          },
        });
      } else {
        clearCurrentAnalysis();
      }
    } catch (preflightError) {
      if (preflightRequestRef.current !== requestId) return;
      setLocationCheckError(
        preflightError.message || 'Could not verify this image on the GIS map.',
      );
      clearCurrentAnalysis();
    } finally {
      if (preflightRequestRef.current === requestId) setLocationChecking(false);
    }
  };

  const handleFileSelect = (event) => {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;

    // Local file blob is needed to actually start the upload. Store gets the
    // preview URL via a quick FileReader pass inside startProcess.
    setFile(selectedFile);
    clearCurrentAnalysis();
    inspectSelectedImage(selectedFile);

    const reader = new FileReader();
    reader.onload = (loadEvent) => {
      // Pre-populate the preview in the store so it's visible immediately
      // (the same image is sent again on Process, no extra cost).
      useProcessingStore.setState({
        previewUrl: loadEvent.target?.result || '',
        fileName: selectedFile.name,
        result: null,
        error: '',
        overlayOpen: false,
      });
    };
    reader.readAsDataURL(selectedFile);
  };

  const handleClear = () => {
    preflightRequestRef.current += 1;
    setFile(null);
    setLocationCheck(null);
    setLocationChecking(false);
    setLocationCheckError('');
    setLocationBlockModalOpen(false);
    setPartialConfirmOpen(false);
    clearCurrentAnalysis();
    resetProcessing();
    if (fileRef.current) fileRef.current.value = '';
  };

  const requestClear = () => {
    if (!file && !result && !preview) return;
    setClearModalOpen(true);
  };

  const confirmClear = () => {
    handleClear();
    setClearModalOpen(false);
  };

  const startConfirmedProcess = (allowPartialMapOverlap = false) => {
    if (!file || processing) return;
    setPartialConfirmOpen(false);
    // Fire-and-forget: the store handles the fetch, the AbortController, and
    // updating processing/stage/progress/result/error in its own state. We do
    // not await this, so navigating away does NOT cancel the request.
    startProcess({
      file,
      altitude: DEFAULT_ALTITUDE,
      drone_model: DEFAULT_DRONE_MODEL,
      canopy_buffer: canopyBuffer,
      hexagon_size: hexagonSize,
      ai_confidence: DEFAULT_AI_CONFIDENCE,
      ai_runtime_tuning: {},
      species,
      allow_partial_map_overlap: allowPartialMapOverlap,
    }).then(() => {
      // After processing settles, see if the result wants to live on the map.
      const finalResult = useProcessingStore.getState().result;
      if (finalResult?.map?.available) {
        setCurrentAnalysis(finalResult);
      }
    });
  };

  const handleProcess = () => {
    if (!file || processing || locationChecking || !locationCheck?.can_process) return;
    if (locationCheck.status === 'partial') {
      setPartialConfirmOpen(true);
      return;
    }
    startConfirmedProcess(false);
  };

  const handleSaveToDatabase = async () => {
    const outcome = await saveCurrentAnalysis(user?.id ?? null);
    if (outcome) {
      appendSavedAnalysis({
        result: outcome.savedResult,
        savePayload: outcome.savePayload,
      });
      setSavedAnalysisBoundary({
        ...outcome.savedResult,
        analysis_id: outcome.savePayload.analysis_id,
      }, { preserveView: true });
      setOverlayOpen(false);
      await Promise.all([fetchPoints(), fetchStats()]);
    }
  };

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
      setSavedAnalysisBoundary(detail);
    } catch (openError) {
      setAnalysisLoadError(openError.message || 'Could not load analysis');
    } finally {
      setLoadingAnalysisId(null);
    }
  };

  const handleDeleteAnalysis = async (analysisId) => {
    if (!analysisId || deleteBusy) return;
    setDeleteError('');
    setDeleteBusy(true);
    try {
      const response = await fetch(`${API}/api/analyses/${analysisId}`, { method: 'DELETE' });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || 'Delete failed');
      }
      setPendingDeleteId(null);
      await Promise.all([fetchStats(), fetchPoints()]);
      clearCurrentAnalysis();
    } catch (deleteAnalysisError) {
      setDeleteError(deleteAnalysisError.message || 'Delete failed');
    } finally {
      setDeleteBusy(false);
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
    if (!result?.exports?.waypoints?.length) return;

    const response = await fetch(`${API}/api/export/${format}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        waypoints: result.exports.waypoints,
        image_name: result.uploaded_file_name,
        detection_mode: result.detection_mode,
      }),
    });

    if (!response.ok) {
      const payload = await response.json().catch(() => ({}));
      throw new Error(payload.detail || `Could not export ${format.toUpperCase()}`);
    }

    const blob = await response.blob();
    const extension = format === 'csv' ? 'csv' : format;
    downloadBlob(blob, `mangrovision_${result.uploaded_file_name}.${extension}`);
  };

  return (
    <Panel
      title="Image Processing"
      subtitle={result?.uploaded_file_name || displayedFileName || 'Analyze drone imagery'}
      accordion={false}
    >
      <PanelCard
        title="Upload Image"
        open={uploadOpen}
        onOpenChange={handleUploadOpenChange}
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M21 15v4a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2v-4" />
            <polyline points="17 8 12 3 7 8" />
            <line x1="12" y1="3" x2="12" y2="15" />
          </svg>
        }
      >
        <input
          ref={fileRef}
          type="file"
          accept="image/jpeg,image/jpg,image/png"
          onChange={handleFileSelect}
          className="file-input"
          id="image-upload"
          disabled={processing}
        />
        <label htmlFor="image-upload" className={`upload-area ${processing ? 'upload-disabled' : ''}`}>
          {preview ? (
            <img src={preview} alt="Preview" className="upload-preview" />
          ) : (
            <div className="upload-placeholder">
              <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="var(--text-muted)" strokeWidth="1.5">
                <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
                <circle cx="8.5" cy="8.5" r="1.5" />
                <polyline points="21 15 16 10 5 21" />
              </svg>
              <span>Click to select a drone image</span>
              <span className="upload-hint">JPEG or PNG with GPS EXIF required</span>
            </div>
          )}
        </label>
        {displayedFileName && !processing && (
          <div className="file-info">
            <span className="file-name">{displayedFileName}</span>
            <button className="btn btn-ghost btn-sm" onClick={requestClear}>Clear</button>
          </div>
        )}
        {file && (
          <div
            className={`image-location-card is-${locationCheck?.status || (locationChecking ? 'checking' : 'error')}`}
            role="status"
            aria-live="polite"
          >
            <div className="image-location-head">
              <span>GIS location</span>
              <strong>
                {locationChecking && 'Checking...'}
                {!locationChecking && locationCheck?.status === 'inside' && 'Inside map'}
                {!locationChecking && locationCheck?.status === 'partial' && 'Partial coverage'}
                {!locationChecking && locationCheck?.status === 'outside' && 'Outside map'}
                {!locationChecking && locationCheck?.status === 'no_gps' && 'GPS missing'}
                {!locationChecking && !locationCheck && 'Check failed'}
              </strong>
            </div>
            {locationChecking && (
              <p>Reading GPS metadata and locating the image footprint on the map...</p>
            )}
            {!locationChecking && locationCheckError && <p>{locationCheckError}</p>}
            {!locationChecking && locationCheck && (
              <>
                <p>{locationCheck.message}</p>
                {Number.isFinite(Number(locationCheck.latitude))
                  && Number.isFinite(Number(locationCheck.longitude)) && (
                    <div className="image-location-details">
                      <span>
                        <small>Map area</small>
                        <strong>{locationCheck.location_label || 'Mapped GIS area'}</strong>
                      </span>
                      <span>
                        <small>Exact center</small>
                        <strong>
                          {Number(locationCheck.latitude).toFixed(6)}, {' '}
                          {Number(locationCheck.longitude).toFixed(6)}
                        </strong>
                      </span>
                      {Array.isArray(locationCheck.coverage_m) && (
                        <span>
                          <small>
                            {locationCheck.footprint_calibrated
                              ? 'Calibrated coverage'
                              : 'Estimated coverage'}
                          </small>
                          <strong>
                            {Number(locationCheck.coverage_m[0]).toFixed(1)} m × {' '}
                            {Number(locationCheck.coverage_m[1]).toFixed(1)} m
                          </strong>
                        </span>
                      )}
                      {locationCheck.status === 'partial' && (
                        <span>
                          <small>Estimated inside map</small>
                          <strong>{locationCheck.estimated_inside_pct}%</strong>
                        </span>
                      )}
                    </div>
                  )}
              </>
            )}
          </div>
        )}
      </PanelCard>

      <PanelCard
        title="Configuration"
        open={configOpen}
        onOpenChange={setConfigOpen}
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        }
      >
        <div className="config-list">
          <div className="config-row">
            <span className="config-label">AI Confidence</span>
            <span className="config-value">{DEFAULT_AI_CONFIDENCE.toFixed(2)}</span>
          </div>
          <div className="config-row config-row-input">
            <label className="config-label" htmlFor="canopy-buffer">Danger Buffer (m)</label>
            <input
              id="canopy-buffer"
              className="form-input config-input"
              type="number"
              min="0.5"
              max="5.0"
              step="0.1"
              value={canopyBuffer}
              onChange={(event) => setCanopyBuffer(Number(event.target.value))}
              disabled={processing}
            />
          </div>
          <div className="config-row config-row-input">
            <label className="config-label" htmlFor="species-select">Species</label>
            <select
              id="species-select"
              className="form-input config-input"
              value={species}
              onChange={(event) => setSpecies(event.target.value)}
              disabled={processing}
            >
              <option value="bungalon">Bungalon — 1 m spacing</option>
              <option value="rhizophora">Rhizophora — 2 m spacing</option>
            </select>
          </div>
          <div className="config-row">
            <span className="config-label">Planting Point Distance</span>
            <span className="config-value">{targetSpacingM.toFixed(1)} m</span>
          </div>
        </div>
      </PanelCard>

      {processing && (
        <div className="progress-card">
          <div className="progress-brand">
            <Logo variant="icon" size={40} alt="MangroVision" />
          </div>
          <div className="progress-header">
            <span className="progress-label">{stage}</span>
            <span className="progress-pct">{progress}%</span>
          </div>
          <div className="progress-bar-track">
            <div className="progress-bar-fill" style={{ width: `${progress}%` }} />
          </div>
          <p className="progress-tip">
            Processing keeps running if you switch tabs or open another page — a small status badge appears at the bottom-right while it works.
          </p>
        </div>
      )}

      {!processing && (
        <div className="process-action">
          <button
            className="btn btn-primary btn-lg"
            style={{ width: '100%' }}
            onClick={handleProcess}
            disabled={!file || locationChecking || !locationCheck?.can_process}
          >
            {locationChecking
              ? 'Checking Image Location...'
              : locationCheck?.status === 'partial'
                ? 'Review Partial Coverage'
                : 'Run Analysis'}
          </button>
          {error && <div className="process-error">{error}</div>}
        </div>
      )}

      {result && (
        <div className="process-action">
          <button
            type="button"
            className="btn btn-primary btn-lg"
            style={{ width: '100%' }}
            onClick={() => setOverlayOpen(true)}
          >
            Show Summary
          </button>
        </div>
      )}

      {result && (
        <div className="process-action">
          <button
            type="button"
            className="btn btn-secondary btn-lg"
            style={{ width: '100%' }}
            onClick={requestClear}
          >
            Process Another Image
          </button>
        </div>
      )}

      <PanelCard
        title="Analysis History"
        badge={analyses.length}
        open={historyOpen}
        onOpenChange={handleHistoryOpenChange}
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M3 3v5h5" />
            <path d="M3.05 13A9 9 0 1 0 6 5.3L3 8" />
            <path d="M12 7v5l3 3" />
          </svg>
        }
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
                      {analysis.analyzed_at?.slice(0, 10)} - {analysis.hexagon_count} pts - {analysis.plantable_area_m2?.toFixed(1)} m2
                    </div>
                  </div>
                  <span className="analytics-history-hint" aria-hidden="true">
                    {isLoading ? (
                      'Loading...'
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

      <Modal
        open={partialConfirmOpen}
        title="Part of this image is outside the map"
        variant="warning"
        confirmLabel="Continue Processing"
        cancelLabel="Cancel"
        onConfirm={() => startConfirmedProcess(true)}
        onCancel={() => setPartialConfirmOpen(false)}
      >
        <p>
          Only about {locationCheck?.estimated_inside_pct ?? 0}% of the estimated image footprint
          is inside the supported GIS map. Continue only if you want MangroVision to analyze the
          image and keep results that can be mapped safely.
        </p>
        <p className="image-location-modal-coordinates">
          Image center: {Number(locationCheck?.latitude).toFixed(6)}, {' '}
          {Number(locationCheck?.longitude).toFixed(6)}
        </p>
      </Modal>

      <Modal
        open={locationBlockModalOpen}
        title={locationCheck?.status === 'outside'
          ? 'Image is outside map bounds'
          : 'Image GPS location is required'}
        variant="warning"
        confirmLabel="OK"
        onConfirm={() => setLocationBlockModalOpen(false)}
      >
        {locationCheck?.status === 'outside' ? (
          <>
            <p>
              This image&apos;s GPS coordinates are outside the supported map area.
              It cannot be analyzed.
            </p>
            <p className="image-location-modal-coordinates">
              Image location: {Number(locationCheck?.latitude).toFixed(6)}, {' '}
              {Number(locationCheck?.longitude).toFixed(6)}
            </p>
          </>
        ) : (
          <p>
            This image does not contain valid latitude and longitude metadata,
            so its map location cannot be verified and it cannot be analyzed.
          </p>
        )}
      </Modal>

      <Modal
        open={clearModalOpen}
        title={result ? 'Discard current analysis?' : 'Remove selected image?'}
        variant="danger"
        confirmLabel={result ? 'Discard analysis' : 'Remove image'}
        onConfirm={confirmClear}
        onCancel={() => setClearModalOpen(false)}
      >
        {result ? (
          <p>This clears the current preview from the workspace. Saved analyses stay in the database.</p>
        ) : (
          <p>This removes the selected image from the upload panel.</p>
        )}
      </Modal>

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

      <ResultsOverlay
        open={overlayOpen && Boolean(result)}
        result={result}
        originalPreview={preview}
        saving={saving}
        saved={Boolean(result?.saved)}
        saveError={saveError}
        onClose={() => setOverlayOpen(false)}
        onSave={handleSaveToDatabase}
        onExport={handleExport}
      />
    </Panel>
  );
}
