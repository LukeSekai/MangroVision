import { useEffect, useRef, useState } from 'react';
import Logo from '../components/Logo';
import { useAuthStore } from '../stores/authStore';
import {
  REPORT_TYPES, buildRestorationReport, downloadReportPdf, formatReportValue, loadReportSource, manilaDay,
  reportCsv, reportFilename, reportPeriod, reportRequests, validateReportPeriod,
} from '../utils/restorationReports';
import './RestorationReports.css';

const API = import.meta.env.VITE_API_BASE || '';
const selectionKey = (selection) => JSON.stringify(selection);

function saveDownload(blob, filename) {
  const url = URL.createObjectURL(blob);
  const link = document.createElement('a');
  link.href = url;
  link.download = filename;
  document.body.appendChild(link);
  link.click();
  link.remove();
  window.setTimeout(() => URL.revokeObjectURL(url), 1000);
}

export function ReportExportActions({ onCsv, onPrint, onPdf, pdfBusy }) {
  return <div className="reports-export-actions">
    <button type="button" className="reports-button" onClick={onCsv}>Download CSV</button>
    <button type="button" className="reports-button" onClick={onPrint}>Print</button>
    <button type="button" className="reports-button reports-button-primary" disabled={pdfBusy} onClick={onPdf}>{pdfBusy ? 'Saving PDF…' : 'Save as PDF'}</button>
  </div>;
}

function ReportIcon({ type = 'planting', size = 22 }) {
  const paths = {
    planting: <><path d="M14 2H6a2 2 0 0 0-2 2v16a2 2 0 0 0 2 2h12a2 2 0 0 0 2-2V8Z" /><path d="M14 2v6h6M8 13h8M8 17h5" /></>,
    monitoring: <><path d="M22 12h-4l-3 9L9 3l-3 9H2" /></>,
    mortality: <><path d="M20 7v5h-5M4 17v-5h5" /><path d="M6.1 7a7 7 0 0 1 11.6-2L20 8M4 16l2.3 3a7 7 0 0 0 11.6-2" /></>,
    organizations: <><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2M22 21v-2a4 4 0 0 0-3-3.9" /><circle cx="9" cy="7" r="4" /><path d="M16 3.1a4 4 0 0 1 0 7.8" /></>,
  };
  return <svg width={size} height={size} viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">{paths[type]}</svg>;
}

function ReportTable({ section }) {
  return (
    <section className="report-section">
      <h3>{section.title}</h3>
      <div className="report-table-scroll">
        <table className="report-table">
          <caption className="reports-sr-only">{section.title}</caption>
          <thead><tr>{section.columns.map((column) => <th key={column.key} scope="col" className={['text', 'date', 'month'].includes(column.type) ? '' : 'report-numeric'}>{column.label}</th>)}</tr></thead>
          <tbody>
            {section.rows.map((row, index) => <tr key={index}>{section.columns.map((column) => <td key={column.key} className={['text', 'date', 'month'].includes(column.type) ? '' : 'report-numeric'}>{formatReportValue(row[column.key], column.type)}</td>)}</tr>)}
            {!section.rows.length && <tr><td className="report-empty" colSpan={section.columns.length}>No records for this selection.</td></tr>}
          </tbody>
        </table>
      </div>
    </section>
  );
}

export function RestorationReportDocument({ report, metadata }) {
  return (
    <article className="report-document" aria-label="Report preview">
      <header className="report-document-header">
        <div className="report-brand"><Logo size={48} /><div><strong>MangroVision</strong><span>Restoration program records</span></div></div>
        <span className="report-document-label">LGU restoration report</span>
      </header>
      <div className="report-document-title"><h2>{report.title}</h2><p>{report.description}</p></div>
      <dl className="report-metadata">
        <div><dt>Reporting period</dt><dd>{formatReportValue(report.period.from, 'date')} – {formatReportValue(report.period.to, 'date')}</dd></div>
        <div><dt>Project site</dt><dd>{report.site}</dd></div>
        <div><dt>Generated · Asia/Manila</dt><dd>{metadata.generatedAt}</dd></div>
        <div><dt>Prepared by</dt><dd>{metadata.preparedBy || 'LGU staff'}</dd></div>
      </dl>
      <div className="report-stats">{report.stats.map((stat) => <div className="report-stat" key={stat.label}><span>{stat.label}</span><strong>{formatReportValue(stat.value, stat.type)}</strong>{stat.hint && <small>{stat.hint}</small>}</div>)}</div>
      {report.sections.map((section) => <ReportTable key={section.title} section={section} />)}
      {metadata.remarks.trim() && <section className="report-section report-remarks"><h3>Report remarks</h3><p>{metadata.remarks.trim()}</p></section>}
      <section className="report-reading-notes"><h3>Reading these figures</h3><ul>{report.notes.map((note) => <li key={note}>{note}</li>)}</ul></section>
      <footer className="report-document-footer"><span>MangroVision · {report.site}</span><span>{formatReportValue(report.period.from, 'date')} – {formatReportValue(report.period.to, 'date')}</span></footer>
    </article>
  );
}

export default function RestorationReportWorkspace({ initialSelection, initialSites = [] }) {
  const user = useAuthStore((state) => state.user);
  const [selection, setSelection] = useState(() => ({ type: 'planting', ...reportPeriod('year'), siteId: '', ...initialSelection }));
  const [preset, setPreset] = useState(() => {
    const year = reportPeriod('year');
    return (!initialSelection?.dateFrom || initialSelection.dateFrom === year.dateFrom)
      && (!initialSelection?.dateTo || initialSelection.dateTo === year.dateTo) ? 'year' : 'custom';
  });
  const [view, setView] = useState({ busy: true, error: '', snapshot: null });
  const [sites, setSites] = useState(initialSites);
  const [remarks, setRemarks] = useState('');
  const [recordRevision, setRecordRevision] = useState(0);
  const [pdfBusy, setPdfBusy] = useState(false);
  const [exportError, setExportError] = useState('');
  const pdfController = useRef(null);
  const originalTitle = useRef(null);
  const invalidPeriod = validateReportPeriod(selection);
  const dirty = Boolean(view.snapshot && view.snapshot.key !== selectionKey(selection));
  const outdated = Boolean(view.snapshot && view.snapshot.revision !== recordRevision);
  const loading = !invalidPeriod && (view.busy || dirty || outdated);
  const exportReady = Boolean(view.snapshot && !loading && !invalidPeriod);

  // Cancel an export if its report selection changes or the dialog is closed.
  useEffect(() => () => pdfController.current?.abort(), [selection, recordRevision]);

  useEffect(() => {
    const controller = new AbortController();
    if (invalidPeriod) {
      queueMicrotask(() => {
        if (!controller.signal.aborted) setView({ busy: false, error: '', snapshot: null });
      });
      return () => controller.abort();
    }
    async function load() {
      if (controller.signal.aborted) return;
      setView((current) => ({ ...current, busy: true, error: '' }));
      try {
        const entries = await Promise.all(reportRequests(selection.type, selection).map(async ({ key, path }) => {
          const payload = await loadReportSource({ key, path }, selection, { base: API, signal: controller.signal });
          return [key, payload];
        }));
        if (controller.signal.aborted) return;
        const data = Object.fromEntries(entries);
        const report = buildRestorationReport(selection.type, data, selection);
        const source = data.overview || data.operations || data.ecology;
        setSites(source.filter_options?.sites || []);
        setView({ busy: false, error: '', snapshot: {
          key: selectionKey(selection), revision: recordRevision, report,
          generatedAt: new Intl.DateTimeFormat('en-PH', { dateStyle: 'medium', timeStyle: 'short', timeZone: 'Asia/Manila' }).format(new Date()),
        } });
      } catch (error) {
        if (!controller.signal.aborted) setView({ busy: false, error: error.message || 'Could not generate the report.', snapshot: null });
      }
    }
    // Let date edits settle briefly and cancel requests for a previous selection.
    const timer = window.setTimeout(() => { void load(); }, 200);
    return () => { window.clearTimeout(timer); controller.abort(); };
  }, [selection, recordRevision, invalidPeriod]);

  useEffect(() => {
    const markOutdated = () => setRecordRevision((current) => current + 1);
    const beforePrint = () => document.body.classList.add('mv-report-printing');
    const afterPrint = () => {
      document.body.classList.remove('mv-report-printing');
      if (originalTitle.current !== null) {
        document.title = originalTitle.current;
        originalTitle.current = null;
      }
    };
    window.addEventListener('mv:data-changed', markOutdated);
    window.addEventListener('beforeprint', beforePrint);
    window.addEventListener('afterprint', afterPrint);
    return () => {
      window.removeEventListener('mv:data-changed', markOutdated);
      window.removeEventListener('beforeprint', beforePrint);
      window.removeEventListener('afterprint', afterPrint);
      afterPrint();
    };
  }, []);

  const metadata = {
    generatedAt: view.snapshot?.generatedAt || '',
    preparedBy: user?.full_name || 'LGU staff', remarks,
  };
  const change = (key, value) => {
    setExportError('');
    setView((current) => ({ ...current, busy: true, error: '' }));
    setSelection((current) => ({ ...current, [key]: value }));
  };
  const changePeriod = (value) => {
    setExportError('');
    setPreset(value);
    if (value !== 'custom') {
      setView((current) => ({ ...current, busy: true, error: '' }));
      setSelection((current) => ({ ...current, ...reportPeriod(value) }));
    }
  };
  const download = () => {
    if (!exportReady) return;
    saveDownload(new Blob([reportCsv(view.snapshot.report, metadata)], { type: 'text/csv;charset=utf-8' }), reportFilename(view.snapshot.report, 'csv'));
  };
  const savePdf = async () => {
    if (!exportReady || pdfBusy) return;
    const report = view.snapshot.report;
    const controller = new AbortController();
    pdfController.current = controller;
    setPdfBusy(true);
    setExportError('');
    try {
      const blob = await downloadReportPdf(report, metadata, { base: API, signal: controller.signal });
      if (!controller.signal.aborted) saveDownload(blob, reportFilename(report, 'pdf'));
    } catch (error) {
      if (!controller.signal.aborted) setExportError(error.message || 'Could not save the PDF. Please try again.');
    } finally {
      if (pdfController.current === controller) {
        pdfController.current = null;
        setPdfBusy(false);
      }
    }
  };
  const print = () => {
    if (!exportReady) return;
    originalTitle.current = document.title;
    document.title = reportFilename(view.snapshot.report, 'pdf').replace(/\.pdf$/, '');
    document.body.classList.add('mv-report-printing');
    try { window.print(); } catch {
      document.title = originalTitle.current;
      originalTitle.current = null;
      document.body.classList.remove('mv-report-printing');
      setExportError('Printing is unavailable in this browser. Use Save as PDF to download a copy.');
    }
  };

  return (
    <div className="reports-page">
      <div className="reports-workspace">
        <div className="reports-controls" role="group" aria-label="Report filters">
          <label className="reports-type-select">Report type<select value={selection.type} onChange={(event) => change('type', event.target.value)}>{REPORT_TYPES.map((type) => <option key={type.id} value={type.id}>{type.title}</option>)}</select></label>
          <div className="reports-filter-grid">
            <label>Reporting period<select value={preset} onChange={(event) => changePeriod(event.target.value)}><option value="year">Year to date</option><option value="quarter">This quarter</option><option value="last-quarter">Last quarter</option><option value="custom">Custom dates</option></select></label>
            <label>From<input type="date" value={selection.dateFrom} max={manilaDay()} required onChange={(event) => { setPreset('custom'); change('dateFrom', event.target.value); }} /></label>
            <label>To<input type="date" value={selection.dateTo} max={manilaDay()} required onChange={(event) => { setPreset('custom'); change('dateTo', event.target.value); }} /></label>
            <label>Project site<select value={selection.siteId} onChange={(event) => change('siteId', event.target.value)}><option value="">All project sites</option>{selection.siteId && !sites.some((site) => String(site.id) === String(selection.siteId)) && <option value={selection.siteId}>Project site #{selection.siteId}</option>}{sites.map((site) => <option key={site.id} value={site.id}>{site.name}</option>)}</select></label>
          </div>
          <p className="reports-filter-hint" role={invalidPeriod ? 'alert' : undefined}>{invalidPeriod || 'Reports update automatically when you change the selection.'}</p>
        </div>
        {view.error && <div className="reports-alert" role="alert">{view.error}{!view.snapshot && <button type="button" className="reports-button" onClick={() => setRecordRevision((current) => current + 1)}>Retry</button>}</div>}
        {loading && <div className="reports-loading" role="status"><ReportIcon /> Preparing your report…</div>}
        {exportReady && <>
          <div className="reports-toolbar"><div><span className="reports-eyebrow">Document preview</span><p>Print this report or download a PDF copy.</p></div><ReportExportActions onCsv={download} onPrint={print} onPdf={savePdf} pdfBusy={pdfBusy} /></div>
          {exportError && <div className="reports-alert" role="alert">{exportError}</div>}
          {pdfBusy && <p className="reports-export-status" role="status">Preparing your PDF download…</p>}
          <label className="reports-remarks-input">Report remarks <span>(optional)</span><textarea rows={2} value={remarks} maxLength={5000} placeholder="Add field findings, follow-up actions or explanations for this report." onChange={(event) => setRemarks(event.target.value)} /></label>
          <RestorationReportDocument report={view.snapshot.report} metadata={metadata} />
        </>}
      </div>
    </div>
  );
}
