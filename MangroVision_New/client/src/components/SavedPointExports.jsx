import { lazy, Suspense, useCallback, useEffect, useId, useMemo, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import Modal from './Modal';
import { EMPTY_EXPORT_FILTERS, exportablePoints, filterExportPoints, pointExportFilename, pointExportOptions,
  pointExportName, pointExportPayload, pointExportScope } from '../utils/pointExports';
import './SavedPointExports.css';

const API = import.meta.env.VITE_API_BASE || '';
const PointExportMap = lazy(() => import('./PointExportMap'));
const FORMATS = [
  { id: 'csv', label: 'CSV', description: 'Spreadsheet' },
  { id: 'gpx', label: 'GPX', description: 'GPS navigation' },
  { id: 'kmz', label: 'KMZ', description: 'Google Earth' },
  { id: 'geojson', label: 'GeoJSON', description: 'GIS data' },
];
const SCOPES = [
  { id: 'all', label: 'All points' },
  { id: 'site', label: 'Project site', key: 'siteId', list: 'sites', prompt: 'Choose a project site' },
  { id: 'species', label: 'Species', key: 'species', list: 'species', prompt: 'Choose a species' },
  { id: 'analysis', label: 'Saved analysis', key: 'analysisId', list: 'analyses', prompt: 'Choose a saved analysis' },
];

function downloadBlob(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export default function SavedPointExports({ points, analyses, loaded, loading, onRetry }) {
  const [session, setSession] = useState(null);
  const [scopeKind, setScopeKind] = useState('all');
  const [scopeValue, setScopeValue] = useState('');
  const [completed, setCompleted] = useState(null);
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [mappedWaypoints, setMappedWaypoints] = useState(null);
  const mapReady = useCallback((waypoints) => setMappedWaypoints(waypoints), []);
  const controllerRef = useRef(null);
  const scopeName = useId();
  const definition = SCOPES.find((scope) => scope.id === scopeKind);
  const filters = useMemo(() => ({ ...EMPTY_EXPORT_FILTERS,
    ...(definition.key ? { [definition.key]: scopeValue } : {}),
  }), [definition, scopeValue]);
  const selectionReady = scopeKind === 'all' || Boolean(scopeValue);
  const selected = useMemo(() => session && selectionReady ? filterExportPoints(session.points, filters) : [],
    [session, selectionReady, filters]);
  const scope = useMemo(() => session ? pointExportScope(filters, session.options) : [], [session, filters]);
  const filename = session ? pointExportFilename(filters, session.format, session.options, session.exportedAt) : '';
  const exportName = session ? pointExportName(filters, session.options, session.exportedAt) : '';
  const preview = useMemo(() => pointExportPayload(selected, scope, session?.analyses, {
    export_name: exportName, filename_stem: filename.replace(/\.[^.]+$/, ''), exported_at: session?.exportedAt,
  }), [selected, scope, session, exportName, filename]);
  const total = useMemo(() => exportablePoints(points).length, [points]);
  const scopeLabel = scopeKind === 'all' ? 'All saved points'
    : session?.options[definition.list].find((option) => option.value === scopeValue)?.label;
  const downloadLabel = FORMATS.find((item) => item.id === session?.format)?.label;

  useEffect(() => () => controllerRef.current?.abort(), []);

  const open = (format) => {
    if (busy || loading || !loaded || !total) return;
    setError('');
    setMappedWaypoints(null);
    setScopeKind('all');
    setScopeValue('');
    // Keep the map and download on the same snapshot throughout this dialog.
    setSession({ format, points: exportablePoints(points), analyses,
      options: pointExportOptions(points, analyses), exportedAt: new Date().toISOString() });
  };
  const close = () => {
    if (controllerRef.current) return;
    setSession(null);
    setError('');
  };
  const changeScope = (kind) => {
    if (controllerRef.current) return;
    setScopeKind(kind);
    setScopeValue('');
    setError('');
  };
  const perform = async () => {
    if (!session || !selectionReady || !selected.length || mappedWaypoints !== preview.waypoints || controllerRef.current) return;
    const controller = new AbortController();
    controllerRef.current = controller;
    setBusy(true);
    setError('');
    const snapshot = { format: session.format, count: selected.length, filename, payload: preview };
    try {
      const response = await fetch(`${API}/api/export/${snapshot.format}`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(snapshot.payload), signal: controller.signal,
      });
      if (!response.ok) {
        const payload = await response.json().catch(() => ({}));
        throw new Error(typeof payload.detail === 'string' ? payload.detail : 'Could not export the selected points. Please try again.');
      }
      const blob = await response.blob();
      if (controller.signal.aborted) return;
      downloadBlob(blob, snapshot.filename);
      setCompleted({ format: downloadLabel, count: snapshot.count, filename: snapshot.filename });
      setSession(null);
    } catch (failure) {
      if (!controller.signal.aborted) setError(failure.message || 'Could not export the selected points. Please try again.');
    } finally {
      setBusy(false);
      controllerRef.current = null;
    }
  };

  return <>
    <p className="point-export-intro">Choose a format, then select and preview points on the map.</p>
    {loading && <p className="point-export-hint" role="status">Loading saved points…</p>}
    {!loaded && !loading && <div className="analytics-error" role="alert">Saved points could not be loaded.
      <button type="button" className="btn btn-secondary btn-sm" onClick={onRetry}>Retry loading points</button>
    </div>}
    {loaded && !loading && !total && <p className="point-export-hint">Save an analysis to export planting points.</p>}
    <div className="point-export-formats">
      {FORMATS.map((format) => <button key={format.id} type="button" aria-label={`Export ${format.label}`}
        className="point-export-format" disabled={busy || loading || !loaded || !total || Boolean(session)} onClick={() => open(format.id)}>
        <strong>{format.label}</strong><span>{format.description}</span>
        <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="m9 5 7 7-7 7" /></svg>
      </button>)}
    </div>
    {(session || completed) && createPortal(<>
      <Modal open={Boolean(session)} title={`Export ${downloadLabel || ''}`} className="modal-card-wide point-export-dialog"
        variant="info" initialFocus="cancel" confirmLabel={`Download ${downloadLabel || ''}`.trim()} cancelLabel="Cancel"
        busy={busy} confirmDisabled={!selectionReady || !selected.length || mappedWaypoints !== preview.waypoints} onConfirm={perform} onCancel={close}>
        {session && <>
          <p className="point-export-dialog-intro">Choose which points to include. Review their locations before downloading.</p>
          {session.format === 'kmz' && <p className="point-export-hint">The KMZ file includes the same point colors shown in the preview for Google Earth.</p>}
          {session.format === 'gpx' && <p className="point-export-hint">For GPS navigation: short, unique waypoint IDs with readable species, status and site details. Flag colors depend on the receiving app or device.</p>}
          {session.format === 'geojson' && <p className="point-export-hint">Includes color fields for GIS apps. Appearance depends on the app opening the file. Choose KMZ for bundled marker styling.</p>}
          <div className="point-export-layout" onKeyDown={(event) => { if (event.key === 'Enter') event.stopPropagation(); }}>
            <div className="point-export-controls">
              <fieldset className="point-export-scopes" disabled={busy}>
                <legend>What would you like to export?</legend>
                <div className="point-export-scope-grid">{SCOPES.map((item) => <label key={item.id} className={scopeKind === item.id ? 'is-selected' : ''}>
                  <input type="radio" name={scopeName} value={item.id} checked={scopeKind === item.id} onChange={() => changeScope(item.id)} />
                  <span>{item.label}</span>
                </label>)}</div>
              </fieldset>
              {definition.key && <label className="point-export-choice">{definition.label}
                <select value={scopeValue} disabled={busy} onChange={(event) => { setScopeValue(event.target.value); setError(''); }}>
                  <option value="">{definition.prompt}</option>
                  {session.options[definition.list].map((option) => <option key={option.value} value={option.value}>{option.label}</option>)}
                </select>
              </label>}
              <div className="point-export-selection" role="status">
                <strong>{selected.length.toLocaleString()} <span>point{selected.length === 1 ? '' : 's'}</span></strong>
                <span>{selectionReady ? scopeLabel : definition.prompt}</span>
              </div>
              {selectionReady && !selected.length && <p className="point-export-hint">No saved points in this selection. Choose another option.</p>}
              {selectionReady && selected.length > 0 && <div className="point-export-file"><span>Download filename</span><strong>{filename}</strong></div>}
              <p className="point-export-hint">Red: dead. Orange: eroded. Yellow: planted. Green: Bungalon. Magenta: Rhizophora. Color priority: dead, then eroded, then planted, then species. Hover for species and erosion details.</p>
            </div>
            <section className="point-export-preview" aria-label="Export map preview">
              <header><h3>Map preview</h3><span>{selected.length.toLocaleString()} points</span></header>
              <Suspense fallback={<p role="status">Loading map preview…</p>}>
                <PointExportMap waypoints={preview.waypoints} onReady={mapReady}
                  emptyMessage={selectionReady ? 'No points in this selection.' : `${definition.prompt} to preview its points.`} />
              </Suspense>
              <p>Pan or zoom to inspect. Moving the map does not change the export.</p>
            </section>
          </div>
          {error && <div className="analytics-error" role="alert">{error}</div>}
        </>}
      </Modal>
      <Modal open={Boolean(completed)} title={`${completed?.format || 'File'} downloaded`}
        variant="success" confirmLabel="Got it" cancelLabel="" onConfirm={() => setCompleted(null)}>
        <p>{completed?.count.toLocaleString()} planting point{completed?.count === 1 ? '' : 's'} exported as <strong className="point-export-filename">{completed?.filename}</strong>. Check your downloads folder.</p>
      </Modal>
    </>, document.body)}
  </>;
}
