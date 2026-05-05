import { useEffect, useRef, useState } from 'react';
import { Panel, PanelCard } from '../components/Panel';
import { useAuthStore } from '../stores/authStore';
import { useMapStore } from '../stores/mapStore';
import ResultsOverlay from './ResultsOverlay';
import './ResultsOverlay.css';
import './ImageProcessing.css';

const API = import.meta.env.VITE_API_BASE || 'http://localhost:8000';

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

function formatMaybeNumber(value, digits = 1) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) return '-';
  return Number(value).toFixed(digits);
}

export default function ImageProcessing() {
  const user = useAuthStore((s) => s.user);
  const setCurrentAnalysis = useMapStore((s) => s.setCurrentAnalysis);
  const clearCurrentAnalysis = useMapStore((s) => s.clearCurrentAnalysis);
  const appendSavedAnalysis = useMapStore((s) => s.appendSavedAnalysis);

  const fileRef = useRef(null);
  const abortRef = useRef(null);

  const [file, setFile] = useState(null);
  const [preview, setPreview] = useState(null);
  const [processing, setProcessing] = useState(false);
  const [stage, setStage] = useState(null);
  const [progress, setProgress] = useState(0);
  const [result, setResult] = useState(null);
  const [error, setError] = useState('');
  const [saveError, setSaveError] = useState('');
  const [saving, setSaving] = useState(false);
  const [overlayOpen, setOverlayOpen] = useState(false);

  const [canopyBuffer, setCanopyBuffer] = useState(1.0);
  const [hexagonSize, setHexagonSize] = useState(1.0);

  useEffect(() => {
    return () => {
      if (abortRef.current) abortRef.current.abort();
      clearCurrentAnalysis();
    };
  }, [clearCurrentAnalysis]);

  const handleFileSelect = (event) => {
    const selectedFile = event.target.files?.[0];
    if (!selectedFile) return;

    setFile(selectedFile);
    setError('');
    setSaveError('');
    setResult(null);
    clearCurrentAnalysis();

    const reader = new FileReader();
    reader.onload = (loadEvent) => setPreview(loadEvent.target?.result || null);
    reader.readAsDataURL(selectedFile);
  };

  const handleClear = () => {
    setFile(null);
    setPreview(null);
    setProcessing(false);
    setStage(null);
    setProgress(0);
    setResult(null);
    setError('');
    setSaveError('');
    setOverlayOpen(false);
    clearCurrentAnalysis();
    if (fileRef.current) fileRef.current.value = '';
  };

  const handleProcess = async () => {
    if (!file) return;

    setProcessing(true);
    setError('');
    setSaveError('');
    setResult(null);
    clearCurrentAnalysis();
    setStage('Uploading image...');
    setProgress(1);

    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const formData = new FormData();
      formData.append('image', file);
      formData.append('altitude', String(DEFAULT_ALTITUDE));
      formData.append('drone_model', DEFAULT_DRONE_MODEL);
      formData.append('canopy_buffer', String(canopyBuffer));
      formData.append('hexagon_size', String(hexagonSize));
      formData.append('ai_confidence', String(DEFAULT_AI_CONFIDENCE));
      formData.append('ai_runtime_tuning', JSON.stringify({}));

      const response = await fetch(`${API}/api/analyses/process-stream`, {
        method: 'POST',
        body: formData,
        signal: controller.signal,
      });

      if (!response.ok || !response.body) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || `Processing failed (${response.status})`);
      }

      const reader = response.body.getReader();
      const decoder = new TextDecoder('utf-8');
      let buffer = '';
      let finalPayload = null;
      let streamError = null;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        buffer += decoder.decode(value, { stream: true });

        let boundary = buffer.indexOf('\n\n');
        while (boundary !== -1) {
          const rawEvent = buffer.slice(0, boundary);
          buffer = buffer.slice(boundary + 2);

          const dataLine = rawEvent
            .split('\n')
            .find((line) => line.startsWith('data:'));
          if (dataLine) {
            const jsonText = dataLine.slice(5).trim();
            if (jsonText) {
              let event;
              try {
                event = JSON.parse(jsonText);
              } catch {
                event = null;
              }
              if (event) {
                if (event.type === 'progress') {
                  if (typeof event.stage === 'string') setStage(event.stage);
                  if (typeof event.pct === 'number') setProgress(event.pct);
                } else if (event.type === 'result') {
                  finalPayload = event.payload;
                } else if (event.type === 'error') {
                  streamError = event.detail || 'Processing failed';
                }
              }
            }
          }
          boundary = buffer.indexOf('\n\n');
        }
      }

      if (streamError) throw new Error(streamError);
      if (!finalPayload) throw new Error('Processing ended without a result');

      setStage('Complete!');
      setProgress(100);
      setResult(finalPayload);
      setOverlayOpen(true);
      if (finalPayload.map?.available) {
        setCurrentAnalysis(finalPayload);
      }
    } catch (processError) {
      if (processError.name !== 'AbortError') {
        setStage(null);
        setProgress(0);
        setError(processError.message || 'Processing failed');
      }
    } finally {
      abortRef.current = null;
      setProcessing(false);
    }
  };

  const handleSaveToDatabase = async () => {
    if (!result?.analysis_key || saving || result.saved) return;

    setSaving(true);
    setSaveError('');
    try {
      const response = await fetch(`${API}/api/analyses/save`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          analysis_key: result.analysis_key,
          user_id: user?.id ?? null,
        }),
      });

      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(payload.detail || 'Save failed');
      }

      const payload = await response.json();
      const savedResult = {
        ...result,
        saved: true,
        analysis_id: payload.analysis_id,
        save_summary: {
          analysis_id: payload.analysis_id,
          new_points: payload.new_points,
          skipped_duplicates: payload.skipped_duplicates,
        },
      };
      setResult(savedResult);
      appendSavedAnalysis({ result: savedResult, savePayload: payload });
    } catch (saveAnalysisError) {
      setSaveError(saveAnalysisError.message || 'Save failed');
    } finally {
      setSaving(false);
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

  const exportButtonsDisabled = !result?.exports?.waypoints?.length;
  const metricSummary = result?.metrics;
  const coordinateRows = result?.map?.coordinates || [];
  const warnings = result?.messages?.warnings || [];
  const infos = result?.messages?.info || [];
  const overlaps = result?.overlaps?.analyses || [];
  const saveSummary = result?.save_summary;

  return (
    <Panel
      title="Image Processing"
      subtitle={result?.uploaded_file_name ? result.uploaded_file_name : 'Analyze drone imagery'}
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
        {file && !processing && (
          <div className="file-info">
            <span className="file-name">{file.name}</span>
            <button className="btn btn-ghost btn-sm" onClick={handleClear}>Clear</button>
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
            <span className="config-value">0.80</span>
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
          <p className="progress-tip">The map stays interactive while the analysis runs in the background.</p>
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
            onClick={handleClear}
          >
            Process Another Image
          </button>
        </div>
      )}

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
