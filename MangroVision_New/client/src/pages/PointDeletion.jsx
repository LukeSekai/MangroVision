import { useEffect, useMemo, useState } from 'react';
import { Panel, PanelCard } from '../components/Panel';
import Modal from '../components/Modal';
import { useMapStore } from '../stores/mapStore';
import './PointDeletion.css';

function getPointStatus(point) {
  if (point.assigned_planter_name) {
    return point.assignment_status === 'completed' ? 'completed' : 'assigned';
  }
  return point.planting_status || 'planned';
}

export default function PointDeletion() {
  const points = useMapStore((s) => s.points);
  const loadingPoints = useMapStore((s) => s.loadingPoints);
  const fetchPoints = useMapStore((s) => s.fetchPoints);
  const fetchStats = useMapStore((s) => s.fetchStats);
  const fetchZones = useMapStore((s) => s.fetchZones);
  const deletePoints = useMapStore((s) => s.deletePoints);
  const mapInstance = useMapStore((s) => s.mapInstance);
  const selectedPointIds = useMapStore((s) => s.deletionSelectedPointIds);
  const addDeletionPoints = useMapStore((s) => s.addDeletionPoints);
  const removeDeletionPoint = useMapStore((s) => s.removeDeletionPoint);
  const clearDeletionSelection = useMapStore((s) => s.clearDeletionSelection);
  const pruneDeletionSelection = useMapStore((s) => s.pruneDeletionSelection);

  const [confirmOpen, setConfirmOpen] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');

  const selectablePoints = useMemo(
    () => points.filter((point) => (
      Number.isFinite(Number(point.id)) &&
      Number.isFinite(Number(point.latitude)) &&
      Number.isFinite(Number(point.longitude))
    )),
    [points],
  );

  const selectedIdSet = useMemo(
    () => new Set(selectedPointIds.map(Number)),
    [selectedPointIds],
  );

  const selectedPoints = useMemo(
    () => selectablePoints.filter((point) => selectedIdSet.has(Number(point.id))),
    [selectablePoints, selectedIdSet],
  );

  useEffect(() => {
    fetchPoints();
    fetchStats();
    fetchZones();
  }, [fetchPoints, fetchStats, fetchZones]);

  useEffect(() => () => clearDeletionSelection(), [clearDeletionSelection]);

  useEffect(() => {
    pruneDeletionSelection(selectablePoints.map((point) => Number(point.id)));
  }, [selectablePoints, pruneDeletionSelection]);

  const selectVisiblePoints = () => {
    const map = mapInstance || useMapStore.getState().mapInstance;
    if (!map) return;
    const bounds = map.getBounds();
    const visibleIds = selectablePoints
      .filter((point) => bounds.contains([point.latitude, point.longitude]))
      .map((point) => Number(point.id));
    setNotice('');
    setError('');
    addDeletionPoints(visibleIds);
  };

  const removeSelectedPoint = (pointId) => {
    removeDeletionPoint(pointId);
  };

  const confirmDelete = async () => {
    if (!selectedPointIds.length) return;
    setDeleting(true);
    setError('');
    setNotice('');
    try {
      const payload = await deletePoints(selectedPointIds);
      setConfirmOpen(false);
      clearDeletionSelection();
      await Promise.all([fetchPoints(), fetchStats()]);
      const deletedCount = payload.deleted_count || 0;
      setNotice(deletedCount === 1 ? '1 point deleted.' : `${deletedCount} points deleted.`);
    } catch (deleteError) {
      setError(deleteError.message || 'Point deletion failed');
    } finally {
      setDeleting(false);
    }
  };

  const selectedCount = selectedPointIds.length;

  return (
    <div className="point-cleanup-page">
      <Panel title="Delete Points" subtitle={`${selectablePoints.length} mapped points`}>
        <PanelCard
          title="Selection"
          badge={selectedCount}
          icon={
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M3 6h18" />
              <path d="M8 6V4h8v2" />
              <path d="M19 6l-1 14H6L5 6" />
            </svg>
          }
        >
          <div className="cleanup-selection-row">
            <div>
              <div className="cleanup-selection-count">{selectedCount}</div>
              <div className="cleanup-selection-label">selected</div>
            </div>
            <div className="cleanup-actions">
              <button type="button" className="btn btn-secondary btn-sm" onClick={selectVisiblePoints} disabled={!selectablePoints.length}>
                Select visible
              </button>
              <button type="button" className="btn btn-ghost btn-sm" onClick={clearDeletionSelection} disabled={!selectedCount}>
                Clear
              </button>
            </div>
          </div>
          <button
            type="button"
            className="btn btn-danger btn-lg cleanup-delete-button"
            onClick={() => setConfirmOpen(true)}
            disabled={!selectedCount || deleting}
          >
            Delete selected
          </button>
          {loadingPoints && <div className="cleanup-muted">Loading points...</div>}
          {notice && <div className="cleanup-notice">{notice}</div>}
          {error && <div className="cleanup-error">{error}</div>}
        </PanelCard>

        <PanelCard
          title="Selected Points"
          badge={selectedCount}
          defaultOpen
          icon={
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
              <path d="M9 11l3 3L22 4" />
              <path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" />
            </svg>
          }
        >
          {selectedPoints.length === 0 ? (
            <div className="cleanup-muted">No points selected.</div>
          ) : (
            <div className="cleanup-selected-list">
              {selectedPoints.slice(0, 30).map((point) => {
                const status = getPointStatus(point);
                return (
                  <div key={point.id} className="cleanup-selected-item">
                    <div className="cleanup-selected-main">
                      <div className="cleanup-selected-title">Point #{point.point_num}</div>
                      <div className="cleanup-selected-meta">{point.image_name || 'Saved analysis'}</div>
                    </div>
                    <span className={`badge badge-${status}`}>{status}</span>
                    <button
                      type="button"
                      className="btn btn-ghost btn-icon cleanup-remove-button"
                      onClick={() => removeSelectedPoint(point.id)}
                      title="Remove from selection"
                    >
                      <svg width="13" height="13" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.4" strokeLinecap="round">
                        <line x1="18" y1="6" x2="6" y2="18" />
                        <line x1="6" y1="6" x2="18" y2="18" />
                      </svg>
                    </button>
                  </div>
                );
              })}
              {selectedPoints.length > 30 && (
                <div className="cleanup-muted">+{selectedPoints.length - 30} more selected</div>
              )}
            </div>
          )}
        </PanelCard>
      </Panel>

      <Modal
        open={confirmOpen}
        title={selectedCount === 1 ? 'Delete selected point?' : `Delete ${selectedCount} selected points?`}
        variant="danger"
        confirmLabel={selectedCount === 1 ? 'Delete point' : 'Delete points'}
        busy={deleting}
        onConfirm={confirmDelete}
        onCancel={() => setConfirmOpen(false)}
      >
        <p>This removes the selected planting point records from the map and any planter assignments that use them.</p>
      </Modal>
    </div>
  );
}
