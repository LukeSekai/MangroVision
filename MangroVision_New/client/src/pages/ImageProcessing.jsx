import { useEffect, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import { useLocation, useNavigate, useNavigationType } from 'react-router-dom';
import { Panel, PanelCard } from '../components/Panel';
import Modal from '../components/Modal';
import ImageAnalysisHistory from '../components/ImageAnalysisHistory';
import Logo from '../components/Logo';
import { useAuthStore } from '../stores/authStore';
import { useMapStore } from '../stores/mapStore';
import { useProcessingStore } from '../stores/processingStore';
import { readAnalysisResponse } from '../utils/processingJobs';
import { canViewSavedAnalysisOnMap } from '../utils/analysisMapContext';
import { formatAnalysisDate } from '../utils/analysisHistory';
import ResultsOverlay from './ResultsOverlay';
import NextActions from '../components/NextActions';
import useFormFeedback from '../utils/useFormFeedback';
import { FieldError, FormErrorSummary } from '../components/FormFeedback';
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
  const location = useLocation();
  const navigate = useNavigate();
  const navigationType = useNavigationType();
  const reviewNavigation = useRef(null);
  const feedback = useFormFeedback({ canopy_buffer: { label: 'Danger buffer (m)' } });
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
  const resultPreviewUrl = useProcessingStore((s) => s.resultPreviewUrl);
  const storedFileName = useProcessingStore((s) => s.fileName);
  const startProcess = useProcessingStore((s) => s.startProcess);
  const saveCurrentAnalysis = useProcessingStore((s) => s.saveCurrentAnalysis);
  const resetProcessing = useProcessingStore((s) => s.reset);
  const setOverlayOpen = useProcessingStore((s) => s.setOverlayOpen);

  const fileRef = useRef(null);
  const pendingUploadRef = useRef(null);
  const preflightRequestRef = useRef(0);
  const areaCheckRequestRef = useRef(null);
  const [file, setFile] = useState(null);
  const [locationCheck, setLocationCheck] = useState(null);
  const [locationChecking, setLocationChecking] = useState(false);
  const [locationCheckError, setLocationCheckError] = useState('');
  const [locationBlockModalOpen, setLocationBlockModalOpen] = useState(false);
  const [partialConfirmOpen, setPartialConfirmOpen] = useState(false);
  const [partialApproved, setPartialApproved] = useState(false);
  const [areaChecking, setAreaChecking] = useState(false);
  const [areaCheckError, setAreaCheckError] = useState('');
  const [areaConfirmation, setAreaConfirmation] = useState(null);
  const [selectedAnalysis, setSelectedAnalysis] = useState(null);
  const [loadingAnalysisId, setLoadingAnalysisId] = useState(null);
  const [analysisLoadError, setAnalysisLoadError] = useState('');
  const [pendingDeleteId, setPendingDeleteId] = useState(null);
  const [pendingDeleteAnalysis, setPendingDeleteAnalysis] = useState(null);
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

  const [openPanel, setOpenPanel] = useState('upload');
  const [historyOpen, setHistoryOpen] = useState(false);

  // The preview shown in the upload panel: prefer the freshly-selected local
  // file's preview if one exists; otherwise rehydrate from the store so users
  // who navigate back mid-processing still see their thumbnail.
  const preview = previewUrl;
  const displayedFileName = file?.name || storedFileName;
  const analyses = stats?.analyses || [];
  const reviewedLocation = useRef(null);
  useEffect(() => {
    if (reviewNavigation.current || selectedAnalysis || location.state?.analysisReviewId) return;
    if (new URLSearchParams(location.search).get('action') !== 'review' || reviewedLocation.current === location.key) return;
    reviewedLocation.current = location.key;
    if (result) setOverlayOpen(true);
    else queueMicrotask(() => setHistoryOpen(true));
  }, [location.key, location.search, location.state, selectedAnalysis, result, setOverlayOpen]);

  useEffect(() => {
    const review = reviewNavigation.current;
    if (!review) return;
    if (review.viewingMap) {
      if (navigationType === 'POP' && location.key === review.returnKey) {
        reviewNavigation.current = null;
        queueMicrotask(() => setHistoryOpen(true));
      }
      return;
    }
    if (location.state?.analysisReviewId === review.id) {
      review.entered = true;
    } else if (review.entered && navigationType === 'POP' && location.key === review.returnKey) {
      reviewNavigation.current = null;
      queueMicrotask(() => {
        setSelectedAnalysis(null);
        setHistoryOpen(true);
      });
    }
  }, [location.key, location.state, navigationType]);
  const loaded = useRef(false);

  useEffect(() => {
    if (loaded.current) return;
    loaded.current = true;
    fetchStats();
  }, [fetchStats]);

  const inspectSelectedImage = async (selectedFile) => {
    areaCheckRequestRef.current?.abort();
    areaCheckRequestRef.current = null;
    setAreaChecking(false);
    setAreaCheckError('');
    setAreaConfirmation(null);
    const requestId = preflightRequestRef.current + 1;
    preflightRequestRef.current = requestId;
    setLocationChecking(true);
    setLocationCheck(null);
    setLocationCheckError('');
    setLocationBlockModalOpen(false);
    setPartialConfirmOpen(false);
    setPartialApproved(false);

    try {
      const formData = new FormData();
      formData.append('image', selectedFile);
      formData.append('altitude', String(DEFAULT_ALTITUDE));
      formData.append('drone_model', DEFAULT_DRONE_MODEL);
      const response = await fetch(`${API}/api/analyses/preflight`, {
        method: 'POST',
        body: formData,
      });
      const payload = await readAnalysisResponse(response);
      if (preflightRequestRef.current !== requestId) return;
      setLocationCheck(payload);
      setLocationBlockModalOpen(
        payload.status === 'outside' || payload.status === 'no_gps',
      );
      setPartialConfirmOpen(payload.status === 'partial');
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

  const selectImage = (selectedFile) => {
    resetProcessing();
    // Local file blob is needed to actually start the upload. Store gets the
    // preview URL via a quick FileReader pass inside startProcess.
    setFile(selectedFile);
    clearCurrentAnalysis();
    inspectSelectedImage(selectedFile);
    const requestId = preflightRequestRef.current;

    const reader = new FileReader();
    reader.onload = (loadEvent) => {
      if (preflightRequestRef.current !== requestId) return;
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

  const handleFileSelect = (event) => {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile || processing || saving) return;
    if (result && !result.saved) {
      pendingUploadRef.current = selectedFile;
      setClearModalOpen(true);
      return;
    }
    selectImage(selectedFile);
  };

  const clearUploadFields = () => {
    preflightRequestRef.current += 1;
    areaCheckRequestRef.current?.abort();
    areaCheckRequestRef.current = null;
    setAreaChecking(false);
    setAreaCheckError('');
    setAreaConfirmation(null);
    setFile(null);
    setLocationCheck(null);
    setLocationChecking(false);
    setLocationCheckError('');
    setLocationBlockModalOpen(false);
    setPartialConfirmOpen(false);
    setPartialApproved(false);
    if (fileRef.current) fileRef.current.value = '';
  };

  const handleClear = () => {
    clearUploadFields();
    clearCurrentAnalysis();
    resetProcessing();
    setOpenPanel('upload');
  };

  const requestClear = () => {
    if (!file && !result && !preview) return;
    if (result?.saved) {
      handleClear();
      return;
    }
    setClearModalOpen(true);
  };

  const confirmClear = () => {
    const nextFile = pendingUploadRef.current;
    pendingUploadRef.current = null;
    handleClear();
    setClearModalOpen(false);
    if (nextFile) selectImage(nextFile);
  };

  const startConfirmedProcess = (allowPartialMapOverlap = false, allowRepeatImageAnalysis = false) => {
    if (!file || processing) return;
    if (!feedback.validate()) { setOpenPanel('configuration'); return; }
    setPartialConfirmOpen(false);
    setAreaConfirmation(null);
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
      allow_repeat_image_analysis: allowRepeatImageAnalysis,
    }).then(() => {
      // After processing settles, see if the result wants to live on the map.
      const finalResult = useProcessingStore.getState().result;
      if (finalResult) {
        clearUploadFields();
        setOpenPanel('upload');
      }
      if (finalResult?.map?.available) {
        setCurrentAnalysis(finalResult);
      }
    });
  };

  const handleProcess = async () => {
    if (!file || processing || locationChecking || areaCheckRequestRef.current || !locationCheck?.can_process) return;
    if (!feedback.validate()) { setOpenPanel('configuration'); return; }
    if (locationCheck.status === 'partial' && !partialApproved) {
      setPartialConfirmOpen(true);
      return;
    }
    const controller = new AbortController();
    areaCheckRequestRef.current = controller;
    setAreaChecking(true);
    setAreaCheckError('');
    setAreaConfirmation(null);
    try {
      const response = await fetch(`${API}/api/analyses/area-context`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ footprint: locationCheck.map?.analysis_footprint,
          image_name: file.name, latitude: locationCheck.latitude, longitude: locationCheck.longitude,
          ...locationCheck.image_identity,
        }),
        signal: controller.signal, cache: 'no-store',
      });
      const context = await readAnalysisResponse(response);
      if (controller.signal.aborted) return;
      if (!Array.isArray(context.analyses) || !Number.isInteger(context.saved_point_count) || context.saved_point_count < 0) {
        throw new Error('Could not check existing planting data. Please try Run Analysis again.');
      }
      if (context.saved_point_count > 0 || context.analyses.length > 0 || context.repeat_analyses?.length > 0) {
        setAreaConfirmation({ ...context, requestId: preflightRequestRef.current });
      } else {
        startConfirmedProcess(locationCheck.status === 'partial');
      }
    } catch (checkError) {
      if (!controller.signal.aborted) setAreaCheckError(checkError.message || 'Could not check existing planting data. Please try Run Analysis again.');
    } finally {
      if (areaCheckRequestRef.current === controller) {
        areaCheckRequestRef.current = null;
        setAreaChecking(false);
      }
    }
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
      setHistoryOpen(false);
      setSelectedAnalysis(detail);
      setSavedAnalysisBoundary(detail);
      reviewNavigation.current = { id: analysisId, returnKey: location.key, entered: false };
      navigate(`${location.pathname}${location.search}${location.hash}`, {
        state: { ...location.state, analysisReviewId: analysisId },
      });
    } catch (openError) {
      setAnalysisLoadError(openError.message || 'Could not load analysis');
    } finally {
      setLoadingAnalysisId(null);
    }
  };

  const handleReturnToHistory = () => {
    const review = reviewNavigation.current;
    reviewNavigation.current = null;
    setSelectedAnalysis(null);
    setHistoryOpen(true);
    if (review?.entered && location.state?.analysisReviewId === review.id) navigate(-1);
  };

  const handleViewAnalysisOnMap = () => {
    if (!canViewSavedAnalysisOnMap(selectedAnalysis)) return;
    setSavedAnalysisBoundary(selectedAnalysis, { preserveView: false });
    const review = reviewNavigation.current;
    reviewNavigation.current = review ? { ...review, viewingMap: true } : null;
    setSelectedAnalysis(null);
    setHistoryOpen(false);
    setOverlayOpen(false);
    // Replace the review entry so browser Back returns to the history list.
    navigate('/map', { replace: true });
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
      setHistoryOpen(true);
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

  const confirmationAnalyses = areaConfirmation?.repeat_analyses?.length
    ? areaConfirmation.repeat_analyses : areaConfirmation?.analyses || [];

  return (
    <Panel
      plantingTool
      title="Analyze Image"
      subtitle={result?.uploaded_file_name || displayedFileName || 'Analyze drone imagery'}
      openKey={openPanel}
      onOpenKeyChange={setOpenPanel}
    >
      {result && <NextActions compact actions={[
        { label: 'Review analysis', onClick: () => setOverlayOpen(true), description: result.saved ? 'Open the saved analysis summary.' : 'Check the overlay and save the analysis.' },
        ...(result.saved ? [{ label: 'Assign available points', to: '/planters?section=assign', description: 'Choose an organization and project site.' }] : []),
      ]} />}
      <PanelCard
        title="Upload Image"
        panelKey="upload"
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
          disabled={processing || saving}
        />
        <label htmlFor="image-upload" className={`upload-area ${processing || saving ? 'upload-disabled' : ''}`}>
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
              <strong>Image GPS (EXIF)</strong>
            </div>
            {locationChecking && (
              <p>Checking image location...</p>
            )}
            {!locationChecking && locationCheckError && <>
              <p>{locationCheckError}</p>
              <button type="button" className="btn btn-secondary btn-sm" disabled={processing || saving}
                onClick={() => inspectSelectedImage(file)}>Retry location check</button>
            </>}
            {!locationChecking && locationCheck && (
              <>
                {locationCheck.status === 'no_gps' && <p>No GPS coordinates found in this image.</p>}
                {locationCheck.latitude != null && locationCheck.longitude != null
                  && Number.isFinite(Number(locationCheck.latitude))
                  && Number.isFinite(Number(locationCheck.longitude)) && (
                    <div className="image-location-details">
                      <span>
                        <small>Latitude</small>
                        <strong>{Number(locationCheck.latitude).toFixed(6)}</strong>
                      </span>
                      <span>
                        <small>Longitude</small>
                        <strong>{Number(locationCheck.longitude).toFixed(6)}</strong>
                      </span>
                    </div>
                  )}
              </>
            )}
          </div>
        )}
      </PanelCard>

      <PanelCard
        title="Configuration"
        panelKey="configuration"
        defaultOpen={false}
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
              {...feedback.props('canopy_buffer')}
              id="canopy-buffer"
              className="form-input config-input"
              type="number"
              min="0.5"
              max="5.0"
              step="0.1"
              value={canopyBuffer}
              onChange={(event) => { feedback.onChange(event); setCanopyBuffer(event.target.value); }}
              required
              disabled={processing || saving}
            />
            <FieldError feedback={feedback} field="canopy_buffer" />
          </div>
          <div className="config-row config-row-input">
            <label className="config-label" htmlFor="species-select">Species</label>
            <select
              id="species-select"
              className="form-input config-input"
              value={species}
              onChange={(event) => setSpecies(event.target.value)}
              disabled={processing || saving}
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
        <FormErrorSummary feedback={feedback} />
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

      {!processing && !result && (
        <div className="process-action">
          <button
            className="btn btn-primary btn-lg"
            style={{ width: '100%' }}
            onClick={handleProcess}
            disabled={!file || locationChecking || areaChecking || !locationCheck?.can_process}
          >
            {locationChecking
              ? 'Checking Image Location...'
              : areaChecking ? 'Checking Existing Planting Data...'
              : locationCheck?.status === 'partial' && !partialApproved
                ? 'Review Partial Coverage'
                : 'Run Analysis'}
          </button>
          {error && <div className="process-error">{error}</div>}
          {areaCheckError && <div className="process-error" role="alert">{areaCheckError}</div>}
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
            Review analysis
          </button>
        </div>
      )}

      <div className="process-action">
        <button
          type="button"
          className="btn btn-secondary btn-lg analysis-history-trigger"
          onClick={() => {
            setAnalysisLoadError('');
            setDeleteError('');
            setHistoryOpen(true);
            fetchStats({ force: true });
          }}
        >
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
            <path d="M3 3v5h5" />
            <path d="M3.05 13A9 9 0 1 0 6 5.3L3 8" />
            <path d="M12 7v5l3 3" />
          </svg>
          <span>View Image Analysis History</span>
          <span className="analysis-history-count">{analyses.length}</span>
        </button>
      </div>

      <ImageAnalysisHistory open={historyOpen} analyses={analyses}
        onClose={() => setHistoryOpen(false)} onOpen={handleOpenAnalysis}
        onDelete={(analysis) => {
          setHistoryOpen(false);
          setPendingDeleteId(analysis.id);
          setPendingDeleteAnalysis(analysis);
        }}
        openingId={loadingAnalysisId} deleteBusy={deleteBusy} error={analysisLoadError || deleteError}
      />

      {createPortal(<Modal
        open={Boolean(areaConfirmation)}
        title={areaConfirmation?.repeat_analyses?.length ? 'This image has already been analyzed' : areaConfirmation?.saved_point_count ? 'Planting points already exist in this area' : 'This area has already been analyzed'}
        variant="warning"
        confirmLabel="Continue analysis"
        cancelLabel="Cancel"
        className="image-area-confirmation"
        onConfirm={() => {
          if (areaConfirmation?.requestId !== preflightRequestRef.current) return;
          setAreaConfirmation(null);
          startConfirmedProcess(locationCheck.status === 'partial', Boolean(areaConfirmation.repeat_analyses?.length));
        }}
        onCancel={() => setAreaConfirmation(null)}
      >
        {areaConfirmation?.repeat_analyses?.length > 0 && <p><strong>No planting points will be added.</strong> You can run the analysis again and save its result. The earlier analysis and its planting records will be kept.</p>}
        {areaConfirmation?.saved_point_count > 0 && <p><strong>{areaConfirmation.saved_point_count.toLocaleString()} saved planting {areaConfirmation.saved_point_count === 1 ? 'point is' : 'points are'}</strong> inside this image's mapped area.</p>}
        {confirmationAnalyses.length > 0 && <>
          <p>{areaConfirmation?.repeat_analyses?.length ? 'Previous runs of this image:' : 'Previous analyses covering this area:'}</p>
          <ul className="image-area-previous-analyses">{confirmationAnalyses.map(item => (
            <li key={item.id}>
              <strong>{item.source_image_name || item.image_name}</strong>
              <span>{item.image_name !== item.source_image_name && `${item.image_name} · `}{formatAnalysisDate(item.analyzed_at)}</span>
            </li>
          ))}</ul>
        </>}
        {!locationCheck?.footprint_calibrated && <p className="image-area-estimate">The image boundary is estimated. Review its position on the map if needed.</p>}
        <p>{areaConfirmation?.repeat_analyses?.length ? 'Do you wish to run this image again?' : 'Do you wish to continue? Existing planting records will be kept, and duplicate planting points will still be excluded.'}</p>
      </Modal>, document.body)}

      <Modal
        open={partialConfirmOpen}
        title="Part of this image is outside the map"
        variant="warning"
        confirmLabel="Use Mapped Portion"
        cancelLabel="Cancel"
        onConfirm={() => {
          setPartialApproved(true);
          setPartialConfirmOpen(false);
        }}
        onCancel={() => setPartialConfirmOpen(false)}
      >
        <p>
          Only about {locationCheck?.estimated_inside_pct ?? 0}% of the estimated image footprint
          is inside the supported GIS map. Continue only if you want MangroVision to analyze the
          image. Only the mapped portion and its planting points will appear on the map.
        </p>
        <p className="image-location-modal-coordinates">
          Image center: {Number(locationCheck?.latitude).toFixed(6)}, {' '}
          {Number(locationCheck?.longitude).toFixed(6)}
        </p>
      </Modal>

      <Modal
        open={locationBlockModalOpen}
        title={locationCheck?.status === 'outside'
          ? 'Invalid image: outside map bounds'
          : 'Image GPS location is required'}
        variant="warning"
        confirmLabel="OK"
        onConfirm={() => setLocationBlockModalOpen(false)}
      >
        {locationCheck?.status === 'outside' ? (
          <>
            <p>
              This image does not overlap the visible map area. It cannot be analyzed.
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
        onCancel={() => {
          if (pendingUploadRef.current && fileRef.current) fileRef.current.value = '';
          pendingUploadRef.current = null;
          setClearModalOpen(false);
        }}
      >
        {result ? (
          <p>This discards the unsaved analysis so you can select a new image. Saved analyses stay in the database.</p>
        ) : (
          <p>This removes the selected image from the upload panel.</p>
        )}
      </Modal>

      {selectedAnalysis && <ResultsOverlay
        open
        result={selectedAnalysis}
        originalPreview={null}
        saving={false}
        saved
        saveError=""
        onClose={handleReturnToHistory}
        onBack={handleReturnToHistory}
        onViewMap={handleViewAnalysisOnMap}
        canViewMap={canViewSavedAnalysisOnMap(selectedAnalysis)}
        onSave={() => {}}
        onExport={handleExportSelected}
      />}

      <Modal
        open={Boolean(pendingDeleteId)}
        title={`Delete "${pendingDeleteAnalysis?.image_name || 'this analysis'}"?`}
        variant="danger"
        confirmLabel="Delete analysis"
        cancelLabel="Cancel"
        busy={deleteBusy}
        onConfirm={() => handleDeleteAnalysis(pendingDeleteId)}
        onCancel={() => {
          if (deleteBusy) return;
          setPendingDeleteId(null);
          setDeleteError('');
          setHistoryOpen(true);
        }}
      >
        <p>This permanently removes the analysis and all its saved planting points from the database. This action cannot be undone.</p>
        {deleteError && <div className="analytics-error" role="alert">{deleteError}</div>}
      </Modal>

      <ResultsOverlay
        open={overlayOpen && Boolean(result)}
        result={result}
        originalPreview={resultPreviewUrl}
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
