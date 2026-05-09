import { useRef, useState } from 'react';
import { Panel, PanelCard } from '../components/Panel';
import Modal from '../components/Modal';
import { useAuthStore } from '../stores/authStore';
import { useMapStore } from '../stores/mapStore';
import { useProcessingStore } from '../stores/processingStore';
import ResultsOverlay from './ResultsOverlay';
import './ResultsOverlay.css';
import './ImageProcessing.css';

const API = import.meta.env.VITE_API_BASE || '';

const DEFAULT_ALTITUDE = 6.0;
const DEFAULT_DRONE_MODEL = 'GENERIC_4K';
const DEFAULT_AI_CONFIDENCE = 0.87;

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
  const appendSavedAnalysis = useMapStore((s) => s.appendSavedAnalysis);

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
  const [file, setFile] = useState(null);
  const [canopyBuffer, setCanopyBuffer] = useState(2.0);
  const [hexagonSize, setHexagonSize] = useState(1.0);
  const [clearModalOpen, setClearModalOpen] = useState(false);

  // The preview shown in the upload panel: prefer the freshly-selected local
  // file's preview if one exists; otherwise rehydrate from the store so users
  // who navigate back mid-processing still see their thumbnail.
  const preview = previewUrl;
  const displayedFileName = file?.name || storedFileName;

  const handleFileSelect = (event) => {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;

    // Local file blob is needed to actually start the upload. Store gets the
    // preview URL via a quick FileReader pass inside startProcess.
    setFile(selectedFile);
    clearCurrentAnalysis();

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
    setFile(null);
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

  const handleProcess = () => {
    if (!file || processing) return;
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
    }).then(() => {
      // After processing settles, see if the result wants to live on the map.
      const finalResult = useProcessingStore.getState().result;
      if (finalResult?.map?.available) {
        setCurrentAnalysis(finalResult);
      }
    });
  };

  const handleSaveToDatabase = async () => {
    const outcome = await saveCurrentAnalysis(user?.id ?? null);
    if (outcome) {
      appendSavedAnalysis({
        result: outcome.savedResult,
        savePayload: outcome.savePayload,
      });
    }
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
    >
      <PanelCard
        title="Upload Image"
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
              <span className="upload-hint">JPEG or PNG with GPS EXIF preferred</span>
            </div>
          )}
        </label>
        {displayedFileName && !processing && (
          <div className="file-info">
            <span className="file-name">{displayedFileName}</span>
            <button className="btn btn-ghost btn-sm" onClick={requestClear}>Clear</button>
          </div>
        )}
      </PanelCard>

      <PanelCard
        title="Configuration"
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
            <circle cx="12" cy="12" r="3" />
            <path d="M19.4 15a1.65 1.65 0 0 0 .33 1.82l.06.06a2 2 0 0 1-2.83 2.83l-.06-.06a1.65 1.65 0 0 0-1.82-.33 1.65 1.65 0 0 0-1 1.51V21a2 2 0 0 1-4 0v-.09A1.65 1.65 0 0 0 9 19.4a1.65 1.65 0 0 0-1.82.33l-.06.06a2 2 0 0 1-2.83-2.83l.06-.06A1.65 1.65 0 0 0 4.68 15a1.65 1.65 0 0 0-1.51-1H3a2 2 0 0 1 0-4h.09A1.65 1.65 0 0 0 4.6 9a1.65 1.65 0 0 0-.33-1.82l-.06-.06a2 2 0 0 1 2.83-2.83l.06.06A1.65 1.65 0 0 0 9 4.68a1.65 1.65 0 0 0 1-1.51V3a2 2 0 0 1 4 0v.09a1.65 1.65 0 0 0 1 1.51 1.65 1.65 0 0 0 1.82-.33l.06-.06a2 2 0 0 1 2.83 2.83l-.06.06a1.65 1.65 0 0 0-.33 1.82V9a1.65 1.65 0 0 0 1.51 1H21a2 2 0 0 1 0 4h-.09a1.65 1.65 0 0 0-1.51 1z" />
          </svg>
        }
        defaultOpen={!result}
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
              max="2.0"
              step="0.1"
              value={canopyBuffer}
              onChange={(event) => setCanopyBuffer(Number(event.target.value))}
              disabled={processing}
            />
          </div>
          <div className="config-row config-row-input">
            <label className="config-label" htmlFor="hexagon-size">Planting Hexagon Size (m)</label>
            <input
              id="hexagon-size"
              className="form-input config-input"
              type="number"
              min="0.3"
              max="2.0"
              step="0.1"
              value={hexagonSize}
              onChange={(event) => setHexagonSize(Number(event.target.value))}
              disabled={processing}
            />
          </div>
        </div>
      </PanelCard>

      {processing && (
        <div className="progress-card">
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
            disabled={!file}
          >
            Run Analysis
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
