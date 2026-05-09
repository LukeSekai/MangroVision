import { useEffect, useRef, useState } from 'react';
import { useMapStore } from '../stores/mapStore';
import { Panel, PanelCard } from '../components/Panel';
import Modal from '../components/Modal';
import './PlanterManagement.css';

function EditPlanterDialog({ planter, form, onChange, onSave, onClose, busy, error }) {
  const dialogRef = useRef(null);

  useEffect(() => {
    const el = dialogRef.current;
    if (!el) return;
    const firstInput = el.querySelector('input, select, textarea');
    firstInput?.focus();

    const handleKey = (e) => {
      if (e.key === 'Escape' && !busy) { onClose(); return; }
      if (e.key !== 'Tab') return;
      const focusables = el.querySelectorAll(
        'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled])',
      );
      if (!focusables.length) return;
      const first = focusables[0];
      const last = focusables[focusables.length - 1];
      if (e.shiftKey && document.activeElement === first) { e.preventDefault(); last.focus(); }
      else if (!e.shiftKey && document.activeElement === last) { e.preventDefault(); first.focus(); }
    };
    document.addEventListener('keydown', handleKey);
    return () => document.removeEventListener('keydown', handleKey);
  }, [busy, onClose]);

  return (
    <div className="planter-edit-backdrop" onMouseDown={busy ? undefined : onClose}>
      <div
        ref={dialogRef}
        className="planter-edit-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby="edit-planter-title"
        onMouseDown={(e) => e.stopPropagation()}
      >
        <div className="planter-edit-header">
          <h3 id="edit-planter-title" className="planter-edit-title">Edit {planter.full_name}</h3>
          <button
            type="button"
            className="btn btn-ghost btn-icon"
            onClick={onClose}
            aria-label="Close edit dialog"
            disabled={busy}
          >
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
              <line x1="18" y1="6" x2="6" y2="18" /><line x1="6" y1="6" x2="18" y2="18" />
            </svg>
          </button>
        </div>
        <div className="form-group">
          <label className="form-label">Full name</label>
          <input className="form-input" type="text" value={form.full_name} onChange={(e) => onChange({ full_name: e.target.value })} disabled={busy} />
        </div>
        <div className="form-group">
          <label className="form-label">Phone</label>
          <input className="form-input" type="text" value={form.phone} onChange={(e) => onChange({ phone: e.target.value })} disabled={busy} />
        </div>
        <div className="form-group">
          <label className="form-label">Base label</label>
          <input className="form-input" type="text" value={form.base_label} onChange={(e) => onChange({ base_label: e.target.value })} disabled={busy} />
        </div>
        <div className="form-group">
          <label className="form-label">Notes</label>
          <textarea className="form-input" value={form.notes} onChange={(e) => onChange({ notes: e.target.value })} rows={2} disabled={busy} />
        </div>
        <div className="form-group">
          <label className="form-label">Reset password (leave blank to keep current)</label>
          <input className="form-input" type="password" value={form.password} onChange={(e) => onChange({ password: e.target.value })} autoComplete="new-password" disabled={busy} />
        </div>
        <div className="form-group">
          <label className="form-label">Status</label>
          <select className="form-select" value={form.status} onChange={(e) => onChange({ status: e.target.value })} disabled={busy}>
            <option value="active">Active</option>
            <option value="inactive">Inactive</option>
          </select>
        </div>
        {error && <div className="assign-message assign-error">{error}</div>}
        <div className="planter-edit-actions">
          <button className="btn btn-ghost btn-sm" onClick={onClose} disabled={busy}>Cancel</button>
          <button className="btn btn-primary btn-sm" onClick={onSave} disabled={busy}>
            {busy ? 'Saving…' : 'Save'}
          </button>
        </div>
      </div>
    </div>
  );
}

const API = import.meta.env.VITE_API_BASE || '';

const STATUS_LABEL = {
  planned: 'Planned',
  assigned: 'Assigned',
  planted: 'Planted',
  completed: 'Completed',
  skipped: 'Skipped',
};

export default function PlanterManagement() {
  const points = useMapStore((s) => s.points);
  const fetchPoints = useMapStore((s) => s.fetchPoints);
  const selectedPointId = useMapStore((s) => s.selectedPointId);

  const [planters, setPlanters] = useState([]);
  const [assignments, setAssignments] = useState([]);
  const [dashStats, setDashStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [selectedPlanterId, setSelectedPlanterId] = useState(null);
  const [assignError, setAssignError] = useState('');
  const [assignSuccess, setAssignSuccess] = useState('');

  // Edit planter dialog state
  const [editingPlanter, setEditingPlanter] = useState(null);
  const [editForm, setEditForm] = useState({
    full_name: '', phone: '', base_label: '', notes: '', password: '', status: 'active',
  });
  const [editBusy, setEditBusy] = useState(false);
  const [editError, setEditError] = useState('');

  // Deactivate confirmation modal state
  const [deactivateTarget, setDeactivateTarget] = useState(null);
  const [deactivateBusy, setDeactivateBusy] = useState(false);

  // Archive assignment confirmation modal state
  const [archiveTarget, setArchiveTarget] = useState(null);
  const [archiveBusy, setArchiveBusy] = useState(false);

  // Reactivate busy tracking
  const [reactivateBusyId, setReactivateBusyId] = useState(null);

  // Per-assignment point-status drill down
  const [assignmentPointsCache, setAssignmentPointsCache] = useState({});
  const [pointStatusBusyId, setPointStatusBusyId] = useState(null);

  const loadData = async () => {
    setLoading(true);
    try {
      const [pRes, dRes, aRes] = await Promise.all([
        fetch(`${API}/api/planters/?include_inactive=true`),
        fetch(`${API}/api/planters/dashboard`),
        fetch(`${API}/api/assignments/?active_only=true`),
      ]);
      setPlanters(await pRes.json());
      setDashStats(await dRes.json());
      setAssignments(await aRes.json());
    } catch (err) {
      console.error('Failed to load planter data:', err);
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadData(); }, []);

  const selectedPoint = points.find((p) => p.id === selectedPointId);
  const activePlanters = planters.filter((p) => p.status === 'active');
  const inactivePlanters = planters.filter((p) => p.status !== 'active');

  const handleAssign = async () => {
    if (!selectedPointId || !selectedPlanterId) return;
    setAssignError('');
    setAssignSuccess('');
    try {
      const res = await fetch(`${API}/api/planters/assign-point`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          planter_id: selectedPlanterId,
          planting_point_id: selectedPointId,
          allow_reassign: false,
        }),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Assignment failed');
      }
      const data = await res.json();
      setAssignSuccess(`Assigned to ${data.planter_name}`);
      fetchPoints();
      loadData();
    } catch (err) {
      setAssignError(err.message);
    }
  };

  const handleArchive = async (id) => {
    setArchiveBusy(true);
    try {
      await fetch(`${API}/api/assignments/${id}/archive`, { method: 'POST' });
      setArchiveTarget(null);
      loadData();
      fetchPoints();
    } catch (err) {
      console.error('Archive failed:', err);
    } finally {
      setArchiveBusy(false);
    }
  };

  const openEdit = (planter) => {
    setEditingPlanter(planter);
    setEditForm({
      full_name: planter.full_name || '',
      phone: planter.phone || '',
      base_label: planter.base_label || '',
      notes: planter.notes || '',
      password: '',
      status: planter.status || 'active',
    });
    setEditError('');
  };

  const handleEditSave = async () => {
    if (!editingPlanter) return;
    setEditBusy(true);
    setEditError('');
    try {
      const payload = {
        full_name: editForm.full_name,
        phone: editForm.phone,
        base_label: editForm.base_label,
        notes: editForm.notes,
        status: editForm.status,
      };
      if (editForm.password) payload.password = editForm.password;
      const res = await fetch(`${API}/api/planters/${editingPlanter.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });
      if (!res.ok) {
        const err = await res.json().catch(() => ({}));
        throw new Error(err.detail || 'Update failed');
      }
      setEditingPlanter(null);
      loadData();
    } catch (err) {
      setEditError(err.message);
    } finally {
      setEditBusy(false);
    }
  };

  const handleDeactivate = (planter) => {
    setDeactivateTarget(planter);
  };

  const confirmDeactivate = async () => {
    if (!deactivateTarget) return;
    setDeactivateBusy(true);
    try {
      await fetch(`${API}/api/planters/${deactivateTarget.id}`, { method: 'DELETE' });
      setDeactivateTarget(null);
      loadData();
    } catch (err) {
      console.error('Deactivate failed:', err);
    } finally {
      setDeactivateBusy(false);
    }
  };

  const cancelDeactivate = () => {
    if (deactivateBusy) return;
    setDeactivateTarget(null);
  };

  const handleReactivate = async (planter) => {
    setReactivateBusyId(planter.id);
    try {
      await fetch(`${API}/api/planters/${planter.id}`, {
        method: 'PATCH',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ status: 'active' }),
      });
      loadData();
    } catch (err) {
      console.error('Reactivate failed:', err);
    } finally {
      setReactivateBusyId(null);
    }
  };

  const loadAssignmentPoints = async (assignmentId) => {
    if (assignmentPointsCache[assignmentId]) {
      setAssignmentPointsCache((prev) => ({ ...prev, [assignmentId]: undefined }));
      return;
    }
    try {
      const res = await fetch(`${API}/api/assignments/${assignmentId}/points`);
      if (!res.ok) throw new Error('Could not load points');
      const data = await res.json();
      setAssignmentPointsCache((prev) => ({ ...prev, [assignmentId]: data }));
    } catch (err) {
      console.error('Load assignment points failed:', err);
    }
  };

  const updateAssignmentPointStatus = async (assignmentId, pointRow, status) => {
    setPointStatusBusyId(pointRow.assignment_point_id);
    try {
      const res = await fetch(
        `${API}/api/assignments/points/${pointRow.assignment_point_id}/status`,
        {
          method: 'PATCH',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ status }),
        },
      );
      if (!res.ok) throw new Error('Status update failed');
      const refreshed = await fetch(`${API}/api/assignments/${assignmentId}/points`).then((r) => r.json());
      setAssignmentPointsCache((prev) => ({ ...prev, [assignmentId]: refreshed }));
      fetchPoints();
      loadData();
    } catch (err) {
      console.error('Point status update failed:', err);
    } finally {
      setPointStatusBusyId(null);
    }
  };

  return (
    <Panel title="Planter Management" subtitle={`${activePlanters.length} active planters`}>
      {dashStats && (
        <PanelCard
          title="Dashboard"
          icon={
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/></svg>
          }
        >
          <div className="planter-stats-grid">
            <div className="stat-card">
              <div className="stat-label">Active</div>
              <div className="stat-value">{dashStats.active_planters}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Assignments</div>
              <div className="stat-value">{dashStats.active_assignments}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Pending</div>
              <div className="stat-value">{dashStats.pending_assigned_points}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Completed</div>
              <div className="stat-value" style={{ color: 'var(--color-completed)' }}>{dashStats.completed_assigned_points}</div>
            </div>
          </div>
        </PanelCard>
      )}

      <PanelCard
        title="Quick Assign (Single Point)"
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
        }
      >
        {selectedPoint ? (
          <div className="assign-card">
            <div className="assign-point-info">
              <span className="font-semibold">Point #{selectedPoint.point_num}</span>
              <span className="text-sm" style={{ color: 'var(--text-secondary)' }}>
                {selectedPoint.image_name}
              </span>
            </div>
            <div className="form-group" style={{ marginTop: 10 }}>
              <label className="form-label">Assign to planter</label>
              <select
                className="form-select"
                value={selectedPlanterId || ''}
                onChange={(e) => setSelectedPlanterId(Number(e.target.value) || null)}
              >
                <option value="">Select a planter...</option>
                {activePlanters.map((p) => (
                  <option key={p.id} value={p.id}>{p.full_name}</option>
                ))}
              </select>
            </div>
            <button
              className="btn btn-primary btn-sm"
              style={{ marginTop: 8, width: '100%' }}
              onClick={handleAssign}
              disabled={!selectedPlanterId}
            >
              Assign Point
            </button>
            {assignError && <div className="assign-message assign-error">{assignError}</div>}
            {assignSuccess && <div className="assign-message assign-success">{assignSuccess}</div>}
          </div>
        ) : (
          <p className="text-sm" style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
            Click a planting point on the map to assign it to a planter.
          </p>
        )}
      </PanelCard>

      <PanelCard
        title="Roster"
        badge={planters.length}
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2"/><circle cx="9" cy="7" r="4"/><path d="M22 21v-2a4 4 0 0 0-3-3.87"/><path d="M16 3.13a4 4 0 0 1 0 7.75"/></svg>
        }
      >
        <div className="planter-list">
          {loading ? (
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>Loading...</p>
          ) : planters.length === 0 ? (
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>
              No planters yet. They will appear here after they register through the /field mobile app.
            </p>
          ) : (
            <>
              {activePlanters.map((p) => (
                <div key={p.id} className="planter-row">
                  <div className="planter-avatar">{p.full_name?.charAt(0)?.toUpperCase()}</div>
                  <div className="planter-info">
                    <span className="planter-name">{p.full_name}</span>
                    <span className="planter-meta">
                      {p.username ? `@${p.username} · ` : ''}{p.active_assignments || 0} assignments
                    </span>
                  </div>
                  <div className="planter-actions">
                    <button className="btn btn-ghost btn-sm" onClick={() => openEdit(p)}>Edit</button>
                    <button className="btn btn-ghost btn-sm" onClick={() => handleDeactivate(p)}>Deactivate</button>
                  </div>
                </div>
              ))}
              {inactivePlanters.length > 0 && (
                <div className="planter-inactive-heading">Inactive</div>
              )}
              {inactivePlanters.map((p) => (
                <div key={p.id} className="planter-row planter-row-inactive">
                  <div className="planter-avatar">{p.full_name?.charAt(0)?.toUpperCase()}</div>
                  <div className="planter-info">
                    <span className="planter-name">{p.full_name}</span>
                    <span className="planter-meta">Inactive</span>
                  </div>
                  <div className="planter-actions">
                    <button
                      className="btn btn-ghost btn-sm"
                      onClick={() => handleReactivate(p)}
                      disabled={reactivateBusyId === p.id}
                    >
                      {reactivateBusyId === p.id ? 'Reactivating…' : 'Reactivate'}
                    </button>
                  </div>
                </div>
              ))}
            </>
          )}
        </div>
      </PanelCard>

      <PanelCard
        title="Active Assignments"
        badge={assignments.length}
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 11l3 3L22 4"/><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11"/></svg>
        }
        defaultOpen={false}
      >
        <div className="assignment-list">
          {assignments.length === 0 ? (
            <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No active assignments</p>
          ) : (
            assignments.map((a) => {
              const expandedRows = assignmentPointsCache[a.id];
              const isExpanded = Array.isArray(expandedRows);
              return (
                <div key={a.id} className="assignment-row-group">
                  <div className="assignment-row">
                    <button
                      className="assignment-info assignment-info-button"
                      onClick={() => loadAssignmentPoints(a.id)}
                    >
                      <span className="assignment-title">{a.title || `Assignment #${a.id}`}</span>
                      <span className="assignment-meta">
                        {a.planter_name} · {a.pending_points}/{a.total_points} pending
                      </span>
                    </button>
                    <button
                      className="btn btn-ghost btn-sm btn-icon"
                      onClick={() => setArchiveTarget(a)}
                      title={`Archive ${a.title || `Assignment #${a.id}`}`}
                      aria-label={`Archive ${a.title || `Assignment #${a.id}`}`}
                    >
                      <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
                        <polyline points="21 8 21 21 3 21 3 8" /><rect x="1" y="3" width="22" height="5" /><line x1="10" y1="12" x2="14" y2="12" />
                      </svg>
                    </button>
                  </div>
                  {isExpanded && (
                    <div className="assignment-point-list">
                      {expandedRows.length === 0 ? (
                        <p className="text-sm" style={{ color: 'var(--text-muted)' }}>No points on this assignment.</p>
                      ) : (
                        expandedRows.map((row) => {
                          const busy = pointStatusBusyId === row.assignment_point_id;
                          return (
                            <div key={row.assignment_point_id} className="assignment-point-row">
                              <div className="assignment-point-info">
                                <span className="assignment-point-title">Point #{row.point_num}</span>
                                <span className="assignment-point-meta">
                                  {STATUS_LABEL[row.status] || row.status}
                                </span>
                              </div>
                              <div className="assignment-point-actions">
                                {row.status !== 'planted' && row.status !== 'completed' && (
                                  <button
                                    className="btn btn-ghost btn-sm"
                                    disabled={busy}
                                    onClick={() => updateAssignmentPointStatus(a.id, row, 'planted')}
                                  >
                                    Mark planted
                                  </button>
                                )}
                                {row.status !== 'skipped' && (
                                  <button
                                    className="btn btn-ghost btn-sm"
                                    disabled={busy}
                                    onClick={() => updateAssignmentPointStatus(a.id, row, 'skipped')}
                                  >
                                    Skip
                                  </button>
                                )}
                                {(row.status === 'planted' || row.status === 'skipped') && (
                                  <button
                                    className="btn btn-ghost btn-sm"
                                    disabled={busy}
                                    onClick={() => updateAssignmentPointStatus(a.id, row, 'assigned')}
                                  >
                                    Reset
                                  </button>
                                )}
                              </div>
                            </div>
                          );
                        })
                      )}
                    </div>
                  )}
                </div>
              );
            })
          )}
        </div>
      </PanelCard>

      {editingPlanter && (
        <EditPlanterDialog
          planter={editingPlanter}
          form={editForm}
          onChange={(updates) => setEditForm((f) => ({ ...f, ...updates }))}
          onSave={handleEditSave}
          onClose={() => { if (!editBusy) setEditingPlanter(null); }}
          busy={editBusy}
          error={editError}
        />
      )}

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
          {deactivateTarget?.full_name || 'This planter'} will no longer be able to sign in to
          the field app until reactivated.
        </p>
        <p>Existing assignments and saved planting points are not affected.</p>
      </Modal>

      <Modal
        open={Boolean(archiveTarget)}
        title={`Archive "${archiveTarget?.title || `Assignment #${archiveTarget?.id}`}"?`}
        variant="warning"
        confirmLabel="Archive assignment"
        cancelLabel="Cancel"
        busy={archiveBusy}
        onConfirm={() => handleArchive(archiveTarget.id)}
        onCancel={() => { if (!archiveBusy) setArchiveTarget(null); }}
      >
        <p>
          This archives the assignment for <strong>{archiveTarget?.planter_name}</strong>. The
          planter will no longer see it in the field app. Existing planting point records are not deleted.
        </p>
      </Modal>
    </Panel>
  );
}
