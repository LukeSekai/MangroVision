import { useEffect, useState } from 'react';
import { Link, useLocation, useNavigate } from 'react-router-dom';
import { Panel, PanelCard } from '../components/Panel';
import ReleaseDeadLocations from '../components/ReleaseDeadLocations';
import { projectSiteId } from '../utils/projectSiteInfo';
import { useMapStore } from '../stores/mapStore';
import { filterMonitoringFeatures, filterMonitoringPoints, monitoringOrganizations } from '../utils/monitoringOrganizationFilter';
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
  const location = useLocation();
  const navigate = useNavigate();
  const points = useMapStore((state) => state.points);
  const projectSites = useMapStore((state) => state.projectSites);
  const monitoringOrganizationId = useMapStore((state) => state.monitoringOrganizationId);
  const setMonitoringOrganizationId = useMapStore((state) => state.setMonitoringOrganizationId);
  const setMonitoringProjectSiteId = useMapStore((state) => state.setMonitoringProjectSiteId);
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

  const sites = filterMonitoringFeatures(projectSites, monitoringOrganizationId)?.features || [];
  const requestedSiteId = new URLSearchParams(location.search).get('project_site_id');
  const selectedSite = sites.find((site) => String(projectSiteId(site)) === requestedSiteId);
  const selectedSiteId = selectedSite ? projectSiteId(selectedSite) : null;
  const [releaseSiteId, setReleaseSiteId] = useState(selectedSiteId);
  if (releaseSiteId !== selectedSiteId) {
    setReleaseSiteId(selectedSiteId);
    setReviewing(false);
    setNotice('');
  }
  useEffect(() => {
    setMonitoringProjectSiteId(selectedSiteId);
  }, [selectedSiteId, setMonitoringProjectSiteId]);

  const visiblePoints = filterMonitoringPoints(points, monitoringOrganizationId, selectedSiteId);
  const organizations = monitoringOrganizations(projectSites);
  const mortalityWarnings = sites.filter((site) => site.properties?.mortality_warning?.active === true);
  const deadPoints = visiblePoints.filter((point) => Boolean(point.death_at));
  const selected = selectedSite ? deadPoints.filter((point) => selectedIds.includes(point.id)) : [];
  const plantedCount = visiblePoints.filter((point) => !point.death_at && (
    point.planting_status === 'planted' || point.assignment_status === 'completed'
  )).length;

  function revealPoints() {
    if (!layerVisibility.points) toggleLayer('points');
  }

  function chooseSite(value) {
    setReviewing(false);
    setNotice('');
    setMonitoringProjectSiteId(value);
    const params = new URLSearchParams(location.search);
    if (value == null || value === '') {
      params.delete('project_site_id');
      params.delete('focus');
    } else {
      params.set('project_site_id', value);
      params.set('focus', 'site_points');
    }
    navigate({ pathname: location.pathname, search: params.toString() });
  }

  function handleRelease(count) {
    setReviewing(false);
    clearSelection();
    setNotice(`${count} location${count === 1 ? ' is' : 's are'} released from the project site and now Planned. Another organization can include them in a new project site. Original deaths and survival results remain in history.`);
    void fetchPoints({ force: true });
    void fetchStats({ force: true });
    void fetchZones({ force: true });
  }

  return <div className="monitoring-map-workspace">
    <Panel title="Monitoring map" subtitle={`${deadPoints.length} dead plants · ${visiblePoints.length} mapped points`}>
      <div className="monitoring-map-organization">
        <label htmlFor="monitoring-map-organization">Show organization</label>
        <select id="monitoring-map-organization" value={monitoringOrganizationId ?? ''}
          onChange={(event) => {
            setReviewing(false);
            setSelectionMode('click');
            setMonitoringOrganizationId(event.target.value);
            chooseSite(null);
          }}>
          <option value="">All organizations</option>
          {organizations.map((organization) => <option key={organization.id} value={organization.id}>{organization.name}</option>)}
        </select>
      </div>
      {mortalityWarnings.length > 0 && <PanelCard panelKey="monitoring-mortality-warnings" title="Mortality warning zones" badge={mortalityWarnings.length} defaultOpen>
        <ul className="monitoring-map-site-list">{mortalityWarnings.map((site) => <li key={projectSiteId(site)}>
          <button type="button" aria-pressed={String(projectSiteId(site)) === String(selectedSiteId)}
            onClick={() => chooseSite(projectSiteId(site))}>{site.properties.name}</button>
        </li>)}</ul>
      </PanelCard>}
      <PanelCard panelKey="monitoring-map-review" title="Dead plants & site release" badge={selectedSite ? deadPoints.length : 0} className="monitoring-map-review-card" defaultOpen>
        <div className="monitoring-map-organization monitoring-map-site-picker">
          <label htmlFor="monitoring-map-project-site">Project site</label>
          <select id="monitoring-map-project-site" value={selectedSiteId ?? ''} onChange={(event) => chooseSite(event.target.value)}>
            <option value="">Choose a project site</option>
            {sites.map((site) => <option key={projectSiteId(site)} value={projectSiteId(site)}>{site.properties.name}</option>)}
          </select>
        </div>
        <label className="monitoring-map-check">
          <input type="checkbox" disabled={!selectedSite} checked={showDeadPointsOnly} onChange={(event) => {
            setShowDeadPointsOnly(event.target.checked);
            if (event.target.checked) revealPoints();
          }} />
          <span>Show dead plants only</span>
        </label>
        <div className="monitoring-map-tool-label">Selection tool</div>
        <div className="monitoring-map-tools" role="group" aria-label="Dead plant selection tool">
          {SELECTION_MODES.map(([mode, label]) => <button key={mode} type="button"
            disabled={!selectedSite}
            className={selectionMode === mode ? 'is-active' : ''} aria-pressed={selectionMode === mode}
            onClick={() => {
              setSelectionMode(mode);
              revealPoints();
            }}>{label}</button>)}
        </div>
        <p className="monitoring-map-instruction">{!selectedSite ? 'Choose a project site to select and release its dead plants.' : selectionMode === 'click'
          ? 'Click a dead point to select it. Use a brush for several points; selected points turn purple.'
          : `Hold the left mouse button and drag to ${selectionMode === 'select' ? 'select' : 'deselect'} dead points. Release to stop; Esc exits.`}</p>
        {loadingPoints && !points.length ? <p className="monitoring-map-help" role="status">Loading planting points…</p> : null}
        {!loadingPoints && selectedSite && !deadPoints.length ? <p className="monitoring-map-help">No dead plants are currently mapped in this project site.</p> : null}
        {selected.length >= 500 ? <p className="monitoring-map-help" role="status">500 points selected. Release this batch before selecting more.</p> : null}
        {selected.length > 0 ? <div className="monitoring-map-actions">
          <strong>{selected.length} selected</strong>
          <button type="button" className="is-primary" onClick={() => { setSelectionMode('click'); setReviewing(true); }}>Release from project site</button>
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
        <p className="monitoring-map-help">Record new deaths during an organization monitoring visit. This map is for locating dead plants and releasing them from project sites.</p>
        <Link className="monitoring-map-link" to="/monitoring">Record a monitoring visit →</Link>
      </PanelCard>

      <PanelCard panelKey="monitoring-map-legend" title="Legend" defaultOpen={false}>
        <div className="monitoring-map-legend">
          <div><i style={{ background: '#7f1d1d' }} />Dead</div>
          <div><i style={{ background: '#16a34a', border: '2px solid #f59e0b' }} />Recorded wave-damage warning</div>
          <div><i style={{ background: '#7c3aed' }} />Selected for release</div>
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
    {reviewing && selected.length > 0 ? <ReleaseDeadLocations
      key={`${selectedSiteId}:${selected.map((point) => point.id).join(',')}`} points={selected}
      projectSite={{ id: selectedSiteId, name: selectedSite.properties.name }}
      onClose={() => setReviewing(false)} onReleased={handleRelease} /> : null}
  </div>;
}
