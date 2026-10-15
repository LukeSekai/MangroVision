import { useEffect, useId, useRef, useState } from 'react';
import { createPortal } from 'react-dom';
import Modal from './Modal';
import './ImageAnalysisHistory.css';
import {
  DEFAULT_HISTORY_FILTERS, HISTORY_SORT_OPTIONS,
  analysisDateOptions, canopyCoverage, filterAnalysisHistory, formatAnalysisDate, formatAnalysisNumber,
} from '../utils/analysisHistory.js';

const API = import.meta.env.VITE_API_BASE || '';
const PAGE_SIZE = 5;

function ResultThumbnail({ analysis }) {
  const [failed, setFailed] = useState(false);
  const name = analysis.source_image_name || analysis.image_name;
  return (
    <span className="analysis-history-thumbnail">
      {analysis.result_preview_url && !failed ? (
        <img src={analysis.result_preview_url} alt={`Detection result for ${name}`}
          loading="lazy" decoding="async" onError={() => setFailed(true)} />
      ) : (
        <span className="analysis-history-no-preview">
          <svg width="22" height="22" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.6" aria-hidden="true">
            <rect x="3" y="3" width="18" height="18" rx="3" /><circle cx="8" cy="8" r="1" />
            <path d="m3 17 5-5 4 4 4-6 5 7" />
          </svg>
          {failed ? 'Preview unavailable' : 'No saved preview'}
        </span>
      )}
    </span>
  );
}

export default function ImageAnalysisHistory({ open, analyses, onClose, onOpen, onDelete, openingId, deleteBusy, error }) {
  const [entries, setEntries] = useState(null);
  const [filters, setFilters] = useState(DEFAULT_HISTORY_FILTERS);
  const [loading, setLoading] = useState(false);
  const [loadError, setLoadError] = useState('');
  const [refresh, setRefresh] = useState(0);
  const [page, setPage] = useState(1);
  const listRef = useRef(null);
  const lockDescriptionId = useId();

  useEffect(() => {
    if (!open) return;
    const controller = new AbortController();
    async function loadHistory() {
      setLoading(true);
      setLoadError('');
      try {
        const response = await fetch(`${API}/api/analyses/?include_previews=true`, {
          signal: controller.signal, cache: 'no-store',
        });
        if (!response.ok) throw new Error('History unavailable');
        const data = await response.json();
        if (!Array.isArray(data)) throw new Error('Invalid history response');
        if (!controller.signal.aborted) setEntries(data);
      } catch {
        if (!controller.signal.aborted) setLoadError('Could not refresh saved results. Try Refresh to load the latest history and previews.');
      } finally {
        if (!controller.signal.aborted) setLoading(false);
      }
    }
    loadHistory();
    return () => controller.abort();
  }, [open, refresh]);

  const saved = entries ?? analyses;
  const filtered = filterAnalysisHistory(saved, filters);
  const pageCount = Math.max(1, Math.ceil(filtered.length / PAGE_SIZE));
  const currentPage = Math.min(page, pageCount);
  if (page !== currentPage) setPage(currentPage);
  const start = (currentPage - 1) * PAGE_SIZE;
  const pageEntries = filtered.slice(start, start + PAGE_SIZE);
  const months = analysisDateOptions(saved);
  const hasFilters = ['search', 'date'].some(key => filters[key] !== DEFAULT_HISTORY_FILTERS[key]);
  const updateFilter = (key, value) => {
    setFilters(current => ({ ...current, [key]: value }));
    setPage(1);
  };
  const resetFilters = () => {
    setFilters(DEFAULT_HISTORY_FILTERS);
    setPage(1);
  };
  const options = values => values.map(([value, label]) => <option key={value} value={value}>{label}</option>);

  useEffect(() => {
    if (listRef.current) listRef.current.scrollTop = 0;
  }, [currentPage, filters, open]);

  return createPortal(
    <Modal open={open} title="Image Analysis History" variant="info" className="analysis-history-modal"
      cancelLabel="Close history" onCancel={onClose}
      icon={<svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
        <path d="M3 3v5h5" /><path d="M3.05 13A9 9 0 1 0 6 5.3L3 8" /><path d="M12 7v5l3 3" />
      </svg>}
    >
      <div className="analysis-history-controls">
        <p className="analysis-history-intro">Find a saved image and select its result to review the analysis.</p>
        <label className="analysis-history-search" htmlFor="analysis-history-search">
          <span>Search images</span>
          <input id="analysis-history-search" type="search" placeholder="Filename or analysis name"
            value={filters.search} onChange={event => updateFilter('search', event.target.value)} />
        </label>
        <div className="analysis-history-filters">
          <label htmlFor="analysis-history-date"><span>Analysis date</span>
            <select id="analysis-history-date" value={filters.date} onChange={event => updateFilter('date', event.target.value)}>
              {options([['all', 'All dates'], ['today', 'Today'], ['7', 'Past 7 days'], ['30', 'Past 30 days'], ['90', 'Past 90 days']])}
              {months.length > 0 && <optgroup label="By month">{options(months)}</optgroup>}
            </select>
          </label>
          <label htmlFor="analysis-history-sort"><span>Sort by</span>
            <select id="analysis-history-sort" value={filters.sort} onChange={event => updateFilter('sort', event.target.value)}>
              {options(HISTORY_SORT_OPTIONS)}
            </select>
          </label>
        </div>
        <div className="analysis-history-toolbar">
          <p role="status" aria-live="polite">{loading ? 'Loading saved results…' : `${filtered.length} of ${saved.length} saved ${saved.length === 1 ? 'analysis' : 'analyses'}`}</p>
          <div>
            {hasFilters && <button type="button" className="btn btn-ghost btn-sm" onClick={resetFilters}>Reset filters</button>}
            <button type="button" className="btn btn-ghost btn-sm" onClick={() => setRefresh(current => current + 1)} disabled={loading || Boolean(openingId)}>Refresh</button>
          </div>
        </div>
        {(loadError || error) && <p className="analytics-error" role="alert">{error || loadError}</p>}
      </div>
      <div ref={listRef} id="analysis-history-list" className="analytics-history-list" aria-busy={loading || Boolean(openingId)}>
        {filtered.length === 0 ? (
          <div className="analysis-history-empty">
            <strong>{loading ? 'Loading history…' : saved.length ? 'No analyses match these filters' : loadError ? 'History could not be loaded' : 'No saved analyses yet'}</strong>
            <p>{loading ? 'Saved results will appear here.' : saved.length ? 'Change the filters or search another filename.' : loadError ? 'Select Refresh to try again.' : 'Analyze an image and save its result to see it here.'}</p>
          </div>
        ) : pageEntries.map(analysis => {
          const isOpening = openingId === analysis.id;
          const name = analysis.source_image_name || analysis.image_name;
          const coverage = canopyCoverage(analysis);
          const canDelete = analysis.can_delete === true && !analysis.deletion_locked;
          const locked = analysis.deletion_locked === true || analysis.can_delete === false;
          const lockId = `${lockDescriptionId}-${analysis.id}`;
          return (
            <div key={analysis.id} className={`analytics-history-item ${isOpening ? 'analytics-history-loading' : ''}`}>
              <button type="button" className="analysis-history-open" onClick={() => onOpen(analysis.id)}
                disabled={loading || Boolean(openingId)} aria-label={`Open summary for ${name}`}
                aria-describedby={canDelete ? undefined : lockId}>
                <ResultThumbnail key={`${analysis.id}-${analysis.result_preview_url || ''}`} analysis={analysis} />
                <span className="analytics-history-main">
                  <span className="analytics-history-title">{name}</span>
                  <span className="analytics-history-meta">
                    {analysis.image_name !== name && <span>{analysis.image_name} · </span>}
                    {formatAnalysisDate(analysis.analyzed_at)}
                  </span>
                  {!canDelete && <span id={lockId} className="analysis-history-lock"
                    title={analysis.delete_lock_reason || (locked
                      ? 'This analysis has assigned points or field records and cannot be deleted.'
                      : 'Refresh history to check deletion availability.')}>
                    <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                      <rect x="4" y="10" width="16" height="11" rx="2" /><path d="M8 10V7a4 4 0 0 1 8 0v3" />
                    </svg>
                    {locked ? 'Locked—contains field records' : 'Refresh to check deletion availability'}
                  </span>}
                  <span className="analysis-history-metrics">
                    <span><span>Plantable area</span><strong>{formatAnalysisNumber(analysis.plantable_area_m2)}{analysis.plantable_area_m2 != null && ' m²'}</strong></span>
                    <span><span>Canopy coverage</span><strong>{formatAnalysisNumber(coverage)}{coverage !== null && '%'}</strong></span>
                    <span><span>Detected canopy</span><strong>{formatAnalysisNumber(analysis.canopy_area_m2)}{analysis.canopy_area_m2 != null && ' m²'}</strong></span>
                    <span><span>Planting points</span><strong>{formatAnalysisNumber(analysis.hexagon_count, 0)}</strong></span>
                  </span>
                </span>
                <span className="analytics-history-hint" aria-hidden="true">{isOpening ? 'Loading…' : 'Review'}
                  <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="m9 5 7 7-7 7" /></svg>
                </span>
              </button>
              {canDelete && <button type="button" className="btn btn-ghost btn-sm btn-icon analytics-history-delete"
                onClick={() => onDelete(analysis)} title={`Delete ${name}`} aria-label={`Delete analysis ${name}`}
                disabled={loading || Boolean(openingId) || deleteBusy}>
                <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" aria-hidden="true">
                  <polyline points="3 6 5 6 21 6" /><path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
                </svg>
              </button>}
            </div>
          );
        })}
      </div>
      {filtered.length > 0 && <nav className="analysis-history-pagination" aria-label="Analysis history pages">
        <p role="status" aria-live="polite">Showing {start + 1}–{Math.min(start + PAGE_SIZE, filtered.length)} of {filtered.length} {filtered.length === 1 ? 'analysis' : 'analyses'}</p>
        <div>
          <button type="button" className="btn btn-secondary btn-sm" aria-controls="analysis-history-list"
            onClick={() => setPage(currentPage - 1)} disabled={currentPage === 1 || loading || Boolean(openingId) || deleteBusy}>Previous</button>
          <span>Page {currentPage} of {pageCount}</span>
          <button type="button" className="btn btn-secondary btn-sm" aria-controls="analysis-history-list"
            onClick={() => setPage(currentPage + 1)} disabled={currentPage === pageCount || loading || Boolean(openingId) || deleteBusy}>Next</button>
        </div>
      </nav>}
    </Modal>, document.body,
  );
}
