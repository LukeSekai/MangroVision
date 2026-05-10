import { useEffect, useMemo, useState } from 'react';
import Modal from '../components/Modal';
import './PlanterActivityReport.css';

const API = import.meta.env.VITE_API_BASE || '';
const PLANTERS_PER_PAGE = 6;

const STATUS_LABEL = {
  active: 'Active',
  completed: 'Completed',
  archived: 'Archived',
};

const AVATAR_PALETTE = [
  '#16a34a', '#eab308', '#2563eb', '#9333ea', '#0891b2',
  '#dc2626', '#0d9488', '#d97706', '#7c3aed', '#0369a1',
];

function avatarColorFor(name = '') {
  let hash = 0;
  for (let i = 0; i < name.length; i += 1) hash = (hash * 31 + name.charCodeAt(i)) >>> 0;
  return AVATAR_PALETTE[hash % AVATAR_PALETTE.length];
}

function initialsFor(name = '') {
  const parts = name.trim().split(/\s+/).filter(Boolean);
  if (parts.length === 0) return '??';
  if (parts.length === 1) return parts[0].slice(0, 2).toUpperCase();
  return (parts[0][0] + parts[parts.length - 1][0]).toUpperCase();
}

function formatDate(isoOrDate) {
  if (!isoOrDate) return '—';
  try {
    const d = new Date(isoOrDate);
    if (Number.isNaN(d.getTime())) return String(isoOrDate);
    return d.toLocaleDateString(undefined, {
      weekday: 'short', year: 'numeric', month: 'short', day: 'numeric',
    });
  } catch {
    return String(isoOrDate);
  }
}

function pct(numerator, denominator) {
  if (!denominator) return 0;
  return Math.max(0, Math.min(100, Math.round((numerator / denominator) * 100)));
}

function planterStatusFlavor(row) {
  if (row.planter.status !== 'active') {
    return { key: 'inactive', label: 'Inactive', subLabel: 'No active sessions', dotColor: '#94a3b8' };
  }
  if (row.assignments.length === 0) {
    return { key: 'idle', label: 'Idle', subLabel: 'No assignments yet', dotColor: '#64748b' };
  }
  if (row.totalPoints > 0 && row.completedPoints >= row.totalPoints) {
    return { key: 'completed', label: 'Completed', subLabel: 'All points planted', dotColor: '#16a34a' };
  }
  if (row.completedPoints > 0) {
    return { key: 'in-progress', label: 'In Progress', subLabel: `${row.pendingPoints} still pending`, dotColor: '#eab308' };
  }
  return { key: 'ready', label: 'Ready', subLabel: 'Ready to plant', dotColor: '#0d9488' };
}

export default function PlanterActivityReport({ open, onClose }) {
  const [planters, setPlanters] = useState([]);
  const [assignments, setAssignments] = useState([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState('');
  const [page, setPage] = useState(0);

  const [deactivateTarget, setDeactivateTarget] = useState(null);
  const [deactivateBusy, setDeactivateBusy] = useState(false);
  const [deactivateError, setDeactivateError] = useState('');

  const [detailsTarget, setDetailsTarget] = useState(null);

  const refresh = async () => {
    setLoading(true);
    setError('');
    try {
      const [planterList, assignmentList] = await Promise.all([
        fetch(`${API}/api/planters/?include_inactive=true`).then((r) => r.json()),
        fetch(`${API}/api/assignments/`).then((r) => r.json()),
      ]);
      setPlanters(Array.isArray(planterList) ? planterList : []);
      setAssignments(Array.isArray(assignmentList) ? assignmentList : []);
    } catch (err) {
      setError(err?.message || 'Could not load activity data');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    if (!open) return;
    setPage(0);
    refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [open]);

  useEffect(() => {
    if (!open) return undefined;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = 'hidden';
    // Tag <body> while the report is open so a CSS rule can hide UI
    // chrome (the floating Panel "Hide" toggle in particular) that
    // would otherwise sit on top of the modal backdrop.
    document.body.classList.add('par-open');
    const handleKey = (event) => {
      if (event.key === 'Escape' && !deactivateTarget && !detailsTarget) onClose?.();
    };
    document.addEventListener('keydown', handleKey);
    return () => {
      document.body.style.overflow = previousOverflow;
      document.body.classList.remove('par-open');
      document.removeEventListener('keydown', handleKey);
    };
  }, [open, onClose, deactivateTarget, detailsTarget]);

  const planterRows = useMemo(() => {
    if (!planters.length) return [];
    const byPlanter = new Map();
    for (const planter of planters) {
      byPlanter.set(planter.id, {
        planter,
        assignments: [],
        totalPoints: 0,
        pendingPoints: 0,
        completedPoints: 0,
        skippedPoints: 0,
        activeAssignments: 0,
        completedAssignments: 0,
        latestDate: null,
      });
    }
    for (const a of assignments) {
      const row = byPlanter.get(a.planter_id);
      if (!row) continue;
      row.assignments.push(a);
      row.totalPoints += a.total_points || 0;
      row.pendingPoints += a.pending_points || 0;
      row.completedPoints += a.completed_points || 0;
      row.skippedPoints += a.skipped_points || 0;
      if (a.status === 'active') row.activeAssignments += 1;
      else if (a.status === 'completed') row.completedAssignments += 1;
      const date = a.assignment_date || a.created_at;
      if (date && (!row.latestDate || date > row.latestDate)) row.latestDate = date;
    }
    return Array.from(byPlanter.values()).sort((a, b) => {
      const aActive = a.planter.status === 'active' ? 0 : 1;
      const bActive = b.planter.status === 'active' ? 0 : 1;
      if (aActive !== bActive) return aActive - bActive;
      return b.completedPoints - a.completedPoints;
    });
  }, [planters, assignments]);

  const totals = useMemo(() => {
    let totalPoints = 0;
    let completedPoints = 0;
    let pendingPoints = 0;
    let skippedPoints = 0;
    let activeAssignments = 0;
    let completedAssignments = 0;
    for (const a of assignments) {
      totalPoints += a.total_points || 0;
      completedPoints += a.completed_points || 0;
      pendingPoints += a.pending_points || 0;
      skippedPoints += a.skipped_points || 0;
      if (a.status === 'active') activeAssignments += 1;
      else if (a.status === 'completed') completedAssignments += 1;
    }
    const activePlanters = planters.filter((p) => p.status === 'active').length;
    return {
      activePlanters,
      totalPlanters: planters.length,
      totalPoints,
      completedPoints,
      pendingPoints,
      skippedPoints,
      activeAssignments,
      completedAssignments,
      totalAssignments: assignments.length,
    };
  }, [assignments, planters]);

  const totalPages = Math.max(1, Math.ceil(planterRows.length / PLANTERS_PER_PAGE));
  const safePage = Math.min(page, totalPages - 1);
  const pageStart = safePage * PLANTERS_PER_PAGE;
  const pageRows = planterRows.slice(pageStart, pageStart + PLANTERS_PER_PAGE);

  const requestDeactivate = (planter) => {
    setDeactivateError('');
    setDeactivateTarget(planter);
  };

  const cancelDeactivate = () => {
    if (deactivateBusy) return;
    setDeactivateTarget(null);
  };

  const confirmDeactivate = async () => {
    if (!deactivateTarget) return;
    setDeactivateBusy(true);
    setDeactivateError('');
    try {
      const res = await fetch(`${API}/api/planters/${deactivateTarget.id}`, { method: 'DELETE' });
      if (!res.ok) {
        const payload = await res.json().catch(() => ({}));
        throw new Error(payload.detail || 'Deactivate failed');
      }
      setDeactivateTarget(null);
      await refresh();
    } catch (err) {
      setDeactivateError(err?.message || 'Deactivate failed');
    } finally {
      setDeactivateBusy(false);
    }
  };

  if (!open) return null;

  return (
    <div className="par-backdrop" role="presentation" onMouseDown={onClose}>
      <div
        className="par-card"
        role="dialog"
        aria-modal="true"
        aria-labelledby="par-title"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <header className="par-header">
          <div>
            <div className="par-eyebrow">Admin Report</div>
            <h2 id="par-title" className="par-title">Planter Activity</h2>
          </div>
          <button
            type="button"
            className="par-close"
            onClick={onClose}
            aria-label="Close report"
          >
            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round">
              <line x1="18" y1="6" x2="6" y2="18" />
              <line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </header>

        <div className="par-body">
          {loading && <div className="par-status">Loading activity…</div>}
          {error && !loading && <div className="par-error">{error}</div>}

          {!loading && !error && (
            <>
              <section className="par-summary">
                <SummaryStat label="Active planters"   value={totals.activePlanters}   accent="green"  />
                <SummaryStat label="Total assignments" value={totals.totalAssignments} accent="blue"   />
                <SummaryStat label="Total points"      value={totals.totalPoints}      accent="slate"  />
                <SummaryStat label="Planted"           value={totals.completedPoints}  accent="amber"  />
                <SummaryStat label="Pending"           value={totals.pendingPoints}    accent="indigo" />
                <SummaryStat label="Skipped"           value={totals.skippedPoints}    accent="gray"   />
              </section>

              {planterRows.length === 0 ? (
                <div className="par-status">
                  No planters registered yet. Activity will appear here once
                  planters are added and assigned points.
                </div>
              ) : (
                <>
                  <section className="par-grid">
                    {pageRows.map((row) => (
                      <PlanterReportCard
                        key={row.planter.id}
                        row={row}
                        onDeactivate={() => requestDeactivate(row.planter)}
                        onSeeDetails={() => setDetailsTarget(row)}
                      />
                    ))}
                  </section>

                  {totalPages > 1 && (
                    <Pagination
                      page={safePage}
                      totalPages={totalPages}
                      onChange={setPage}
                      visibleCount={pageRows.length}
                      totalCount={planterRows.length}
                    />
                  )}
                </>
              )}
            </>
          )}
        </div>
      </div>

      <Modal
        open={Boolean(deactivateTarget)}
        title={`Deactivate ${deactivateTarget?.full_name || 'planter'}?`}
        variant="danger"
        confirmLabel="Deactivate"
        cancelLabel="Cancel"
        busy={deactivateBusy}
        onConfirm={confirmDeactivate}
        onCancel={cancelDeactivate}
      >
        <p>
          {deactivateTarget?.full_name || 'This planter'} will no longer be able to sign in
          to the field app until reactivated.
        </p>
        <p>Existing assignments and saved planting points are not affected.</p>
        {deactivateError && (
          <p style={{ color: '#dc2626', marginTop: 8 }}>{deactivateError}</p>
        )}
      </Modal>

      <Modal
        open={Boolean(detailsTarget)}
        title={detailsTarget ? `${detailsTarget.planter.full_name} — Details` : 'Planter details'}
        variant="info"
        confirmLabel="Close"
        cancelLabel=""
        onConfirm={() => setDetailsTarget(null)}
      >
        {detailsTarget && (
          <>
            <p>
              <strong>Base:</strong> {detailsTarget.planter.base_label || 'Not set'}<br />
              <strong>Phone:</strong> {detailsTarget.planter.phone || 'Not set'}<br />
              <strong>Status:</strong> {detailsTarget.planter.status === 'active' ? 'Active' : 'Inactive'}
            </p>
            <p>
              <strong>Assignments:</strong> {detailsTarget.assignments.length}
              {' '}({detailsTarget.activeAssignments} active, {detailsTarget.completedAssignments} completed)<br />
              <strong>Total planting points:</strong> {detailsTarget.totalPoints}<br />
              <strong>Planted:</strong> {detailsTarget.completedPoints}
              {' '}({pct(detailsTarget.completedPoints, detailsTarget.totalPoints)}%)<br />
              <strong>Pending:</strong> {detailsTarget.pendingPoints}<br />
              <strong>Skipped:</strong> {detailsTarget.skippedPoints}
            </p>
          </>
        )}
      </Modal>
    </div>
  );
}

function SummaryStat({ label, value, accent }) {
  return (
    <div className={`par-summary-stat par-summary-${accent}`}>
      <div className="par-summary-value">{value}</div>
      <div className="par-summary-label">{label}</div>
    </div>
  );
}

function PlanterReportCard({ row, onDeactivate, onSeeDetails }) {
  const { planter, assignments, totalPoints, completedPoints } = row;
  const flavor = planterStatusFlavor(row);
  const completion = pct(completedPoints, totalPoints);
  const initials = initialsFor(planter.full_name);
  const avatarColor = avatarColorFor(planter.full_name || `planter-${planter.id}`);

  const sorted = [...assignments].sort((a, b) => {
    const ad = a.assignment_date || a.created_at || '';
    const bd = b.assignment_date || b.created_at || '';
    return bd.localeCompare(ad);
  });
  const recent = sorted.slice(0, 4);
  const hidden = sorted.length - recent.length;

  return (
    <article className={`par-planter par-planter-${flavor.key}`}>
      <header className="par-planter-head">
        <div className="par-avatar" style={{ background: avatarColor }} aria-hidden="true">
          {initials}
        </div>
        <div className="par-planter-identity">
          <div className="par-planter-name">{planter.full_name}</div>
          <div className="par-planter-meta">
            {planter.base_label || `Planter #${planter.id}`}
            {planter.phone ? ` · ${planter.phone}` : ''}
          </div>
        </div>
        <div className="par-status-block">
          <span className={`par-status-pill par-status-pill-${flavor.key}`}>
            <span className="par-status-pill-dot" style={{ background: flavor.dotColor }} aria-hidden="true" />
            {flavor.label}
          </span>
          <span className="par-status-sub">
            <span className="par-status-sub-dot" style={{ background: flavor.dotColor }} aria-hidden="true" />
            {flavor.subLabel}
          </span>
        </div>
      </header>

      <div className="par-planter-body">
        <div className="par-planter-meta-row">
          <span>{formatDate(row.latestDate)}</span>
          <span>{totalPoints} planting point{totalPoints === 1 ? '' : 's'}</span>
        </div>

        <div className="par-items">
          <div className="par-items-head">
            <span>Assignments</span>
            <span className="par-items-head-qty">Points</span>
            <span className="par-items-head-progress">Progress</span>
          </div>

          {recent.length === 0 ? (
            <div className="par-items-empty">No assignments yet.</div>
          ) : (
            recent.map((a) => {
              const aPct = pct(a.completed_points || 0, a.total_points || 0);
              return (
                <div key={a.id} className="par-items-row">
                  <div className="par-items-name">
                    <div className="par-items-title">{a.title || `Assignment #${a.id}`}</div>
                    <div className="par-items-sub">{STATUS_LABEL[a.status] || a.status} · {formatDate(a.assignment_date)}</div>
                  </div>
                  <div className="par-items-qty">{a.total_points || 0}</div>
                  <div className="par-items-progress">
                    <div className="par-items-progress-track">
                      <div className="par-items-progress-fill" style={{ width: `${aPct}%` }} />
                    </div>
                    <div className="par-items-progress-text">{a.completed_points || 0}/{a.total_points || 0}</div>
                  </div>
                </div>
              );
            })
          )}
          {hidden > 0 && (
            <div className="par-items-more">+{hidden} more</div>
          )}
        </div>
      </div>

      <footer className="par-planter-footer">
        <div className="par-total-row">
          <div className="par-total-label">Total Planted</div>
          <div className="par-total-value">
            <span className="par-total-fraction">{completedPoints}/{totalPoints || 0}</span>
            <span className="par-total-pct">{completion}%</span>
          </div>
        </div>
        <div className="par-total-bar">
          <div className="par-total-fill" style={{ width: `${completion}%` }} />
        </div>

        <div className="par-actions">
          <button
            type="button"
            className="par-action par-action-secondary"
            onClick={onDeactivate}
            disabled={planter.status !== 'active'}
            title={planter.status !== 'active' ? 'Already inactive' : 'Deactivate this planter'}
          >
            {planter.status === 'active' ? 'Deactivate' : 'Inactive'}
          </button>
          <button
            type="button"
            className="par-action par-action-primary"
            onClick={onSeeDetails}
          >
            See Details
          </button>
        </div>
      </footer>
    </article>
  );
}

function Pagination({ page, totalPages, onChange, visibleCount, totalCount }) {
  const goPrev = () => onChange(Math.max(0, page - 1));
  const goNext = () => onChange(Math.min(totalPages - 1, page + 1));
  const start = page * PLANTERS_PER_PAGE + 1;
  const end = page * PLANTERS_PER_PAGE + visibleCount;

  // Build a small page-number row, capped so it doesn't overflow on small screens.
  const indices = Array.from({ length: totalPages }, (_, i) => i);

  return (
    <nav className="par-pagination" aria-label="Planter pages">
      <span className="par-pagination-info">
        Showing <strong>{start}–{end}</strong> of <strong>{totalCount}</strong> planters
      </span>
      <div className="par-pagination-controls">
        <button
          type="button"
          className="par-pagination-btn"
          onClick={goPrev}
          disabled={page === 0}
          aria-label="Previous page"
        >
          ←
        </button>
        {indices.map((i) => (
          <button
            key={i}
            type="button"
            className={`par-pagination-page${i === page ? ' is-active' : ''}`}
            onClick={() => onChange(i)}
            aria-current={i === page ? 'page' : undefined}
            aria-label={`Page ${i + 1}`}
          >
            {i + 1}
          </button>
        ))}
        <button
          type="button"
          className="par-pagination-btn"
          onClick={goNext}
          disabled={page >= totalPages - 1}
          aria-label="Next page"
        >
          →
        </button>
      </div>
    </nav>
  );
}
