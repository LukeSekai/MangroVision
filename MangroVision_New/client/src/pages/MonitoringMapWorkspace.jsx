import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { Panel, PanelCard } from '../components/Panel';
import ReplantingApproval from '../components/ReplantingApproval';
import { useMapStore } from '../stores/mapStore';
import { filterMonitoringPoints, monitoringOrganizations } from '../utils/monitoringOrganizationFilter';
import './MonitoringMapWorkspace.css';

const SELECTION_MODES = [
  ['click', 'Move / click'],
  ['select', 'Brush select'],
  ['deselect', 'Brush erase'],
];

const LAYERS = [
  ['points', 'Planting points'],
  ['orthophoto', 'Orthophoto overlay'],
  ['siteZones', 'Assignment zones'],
  ['projectSites', 'Project sites'],
  ['forbidden', 'Forbidden zones'],
  ['eroded', 'Eroded zones'],
  ['warnings', 'Warning zones'],
];

export default function MonitoringMapWorkspace() {
  const points = useMapStore((state) => state.points);
  const projectSites = useMapStore((state) => state.projectSites);
  const monitoringOrganizationId = useMapStore((state) => state.monitoringOrganizationId);
  const setMonitoringOrganizationId = useMapStore((state) => state.setMonitoringOrganizationId);
  const loadingPoints = useMapStore((state) => state.loadingPoints);
  const layerVisibility = useMapStore((state) => state.layerVisibility);
  const toggleLayer = useMapStore((state) => state.toggleLayer);
  const showDeadPointsOnly = useMapStore((state) => state.showDeadPointsOnly);
  const setShowDeadPointsOnly = useMapStore((state) => state.setShowDeadPointsOnly);
  const selectedIds = useMapStore((state) => state.replantingSelectedPointIds);
  const clearSelection = useMapStore((state) => state.clearReplantingSelection);
  const selectionMode = useMapStore((state) => state.replantingSelectionMode);
  const setSelectionMode = useMapStore((state) => state.setReplantingSelectionMode);
  const fetchPoints = useMapStore((state) => state.fetchPoints);
  const fetchStats = useMapStore((state) => state.fetchStats);
  const fetchZones = useMapStore((state) => state.fetchZones);
  const [reviewing, setReviewing] = useState(false);
  const [notice, setNotice] = useState('');

  useEffect(() => () => {
    clearSelection();
    setSelectionMode('click');
    setShowDeadPointsOnly(false);
    setMonitoringOrganizationId(null);
  }, [clearSelection, setSelectionMode, setShowDeadPointsOnly, setMonitoringOrganizationId]);

  const visiblePoints = filterMonitoringPoints(points, monitoringOrganizationId);
  const organizations = monitoringOrganizations(projectSites);
  const deadPoints = visiblePoints.filter((point) => Boolean(point.death_at));
  const selected = deadPoints.filter((point) => selectedIds.includes(point.id));
  const plantedCount = visiblePoints.filter((point) => !point.death_at && (
    point.planting_status === 'planted' || point.assignment_status === 'completed'
  )).length;

  function revealPoints() {
    if (!layerVisibility.points) toggleLayer('points');
  }

  function handleApproval(count) {
    setReviewing(false);
    clearSelection();
    setNotice(`${count} location${count === 1 ? ' is' : 's are'} available for new planting assignments. The original monitoring history remains unchanged.`);
    void fetchPoints({ force: true });
    void fetchStats({ force: true });
    void fetchZones({ force: true });
  }

  return <div className="monitoring-map-workspace">
    <Link className="monitoring-map-back" to="/monitoring">← Monitoring</Link>
    <Panel title="Monitoring map" subtitle={`${deadPoints.length} dead plants · ${visiblePoints.length} mapped points`}>
      <div className="monitoring-map-organization">
        <label htmlFor="monitoring-map-organization">Show organization</label>
        <select id="monitoring-map-organization" value={monitoringOrganizationId ?? ''}
          onChange={(event) => {
            setReviewing(false);
            setSelectionMode('click');
            setMonitoringOrganizationId(event.target.value);
          }}>
          <option value="">All organizations</option>
          {organizations.map((organization) => <option key={organization.id} value={organization.id}>{organization.name}</option>)}
        </select>
      </div>
      <PanelCard panelKey="monitoring-map-review" title="Dead plants & replanting" badge={deadPoints.length} className="monitoring-map-review-card" defaultOpen>
        <label className="monitoring-map-check">
          <input type="checkbox" checked={showDeadPointsOnly} onChange={(event) => {
            setShowDeadPointsOnly(event.target.checked);
            if (event.target.checked) revealPoints();
          }} />
          <span>Show dead plants only</span>
        </label>
        <div className="monitoring-map-tool-label">Selection tool</div>
        <div className="monitoring-map-tools" role="group" aria-label="Dead plant selection tool">
          {SELECTION_MODES.map(([mode, label]) => <button key={mode} type="button"
            className={selectionMode === mode ? 'is-active' : ''} aria-pressed={selectionMode === mode}
            onClick={() => {
              setSelectionMode(mode);
              revealPoints();
            }}>{label}</button>)}
        </div>
        <p className="monitoring-map-instruction">{selectionMode === 'click'
          ? 'Click a dead point to select it. Use a brush for several points; selected points turn purple.'
          : `Move over dead points to ${selectionMode === 'select' ? 'select' : 'deselect'} them. Press Esc or choose Move / click to finish.`}</p>
        {loadingPoints && !points.length ? <p className="monitoring-map-help" role="status">Loading planting points…</p> : null}
        {!loadingPoints && !deadPoints.length ? <p className="monitoring-map-help">No dead plants are currently mapped.</p> : null}
        {selected.length >= 500 ? <p className="monitoring-map-help" role="status">500 points selected. Approve this batch before selecting more.</p> : null}
        {selected.length > 0 ? <div className="monitoring-map-actions">
          <strong>{selected.length} selected</strong>
          <button type="button" className="is-primary" onClick={() => { setSelectionMode('click'); setReviewing(true); }}>Review for replanting</button>
          <button type="button" onClick={clearSelection}>Clear</button>
        </div> : null}
        {notice ? <p className="monitoring-map-notice" role="status">{notice}</p> : null}
      </PanelCard>

      <PanelCard panelKey="monitoring-map-overview" title="Overview" defaultOpen={false}>
        <div className="monitoring-map-stats">
          <div><span>Mapped points</span><strong>{visiblePoints.length.toLocaleString()}</strong></div>
          <div><span>Planted, not dead</span><strong>{plantedCount.toLocaleString()}</strong></div>
          <div><span>Dead</span><strong>{deadPoints.length.toLocaleString()}</strong></div>
          <div><span>Selected</span><strong>{selected.length.toLocaleString()}</strong></div>
        </div>
        <p className="monitoring-map-help">Record new deaths during an organization monitoring visit. This map is for locating dead plants and reviewing replanting.</p>
        <Link className="monitoring-map-link" to="/monitoring">Record a monitoring visit →</Link>
      </PanelCard>

      <PanelCard panelKey="monitoring-map-legend" title="Legend" defaultOpen={false}>
        <div className="monitoring-map-legend">
          <div><i style={{ background: '#7f1d1d' }} />Dead</div>
          <div><i style={{ background: '#7c3aed' }} />Selected for replanting review</div>
          <div><i style={{ background: '#eab308' }} />Planted</div>
          <div><i style={{ background: '#2563eb' }} />Assigned</div>
          <div><i style={{ background: '#16a34a' }} />Planned</div>
        </div>
      </PanelCard>

      <PanelCard panelKey="monitoring-map-layers" title="Layers" defaultOpen={false}>
        <div className="monitoring-map-layers">
          {LAYERS.map(([key, label]) => <label key={key}>
            <input type="checkbox"
              checked={Boolean(layerVisibility[key]) && (monitoringOrganizationId == null || !['forbidden', 'eroded', 'warnings'].includes(key))}
              disabled={monitoringOrganizationId != null && ['forbidden', 'eroded', 'warnings'].includes(key)}
              onChange={() => toggleLayer(key)} />
            <span>{label}</span>
          </label>)}
        </div>
        {monitoringOrganizationId != null ? <p className="monitoring-map-help">Shared risk zones are hidden while viewing one organization.</p> : null}
      </PanelCard>
    </Panel>
    {reviewing && selected.length > 0 ? <ReplantingApproval
      key={selected.map((point) => point.id).join(',')} points={selected}
      onClose={() => setReviewing(false)} onApproved={handleApproval} /> : null}
  </div>;
}
