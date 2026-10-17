import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { availableOrganizationPoints } from '../utils/organizationAssignment';
import { useMapStore } from '../stores/mapStore';
import { useProcessingStore } from '../stores/processingStore';
import { Panel, PanelCard } from '../components/Panel';
import Modal from '../components/Modal';
import OrganizationAccess from '../components/OrganizationAccess';
import { getPlanterColor } from '../utils/planterColors';
import { useLocation } from 'react-router-dom';
import useFormFeedback from '../utils/useFormFeedback';
import { FieldError, FormErrorSummary, FormMessage } from '../components/FormFeedback';
import { formRequest } from '../utils/formValidation';
import './PlanterManagement.css';

const API = import.meta.env.VITE_API_BASE || '';

const ASSIGNMENT_SPECIES_LABELS = {
  bungalon: 'Bungalon',
  rhizophora: 'Rhizophora',
  'api-api': 'Api-Api',
};

function normalizeAssignmentSpecies(species) {
  const key = String(species || '').trim().toLowerCase().replace(/_/g, '-');
  if (key === 'api api' || key === 'apiapi') return 'api-api';
  return ASSIGNMENT_SPECIES_LABELS[key] ? key : '';
}

export default function PlanterManagement() {
  const location = useLocation();
  const requestedSection = new URLSearchParams(location.search).get('section');
  const [openPanel, setOpenPanel] = useState('assign');
  const feedback = useFormFeedback({
    organization: { label: 'Organization', aliases: ['organization_id', 'site_zone_id', 'project_site_id'], serverTerms: ['project site'] },
    activity: { label: 'Planting activity', aliases: ['planting_schedule_id'], serverTerms: ['planting activity'] },
    point_count: { label: 'Number of points', serverTerms: ['point count'] },
  });
  useEffect(() => {
    if (requestedSection === 'assign') queueMicrotask(() => setOpenPanel(requestedSection));
  }, [requestedSection, location.key]);
  const points = useMapStore((s) => s.points);
  const loadingPoints = useMapStore((s) => s.loadingPoints);
  const fetchPoints = useMapStore((s) => s.fetchPoints);
  const fetchZones = useMapStore((s) => s.fetchZones);
  const projectSites = useMapStore((s) => s.projectSites);
  const setAssignmentSelection = useMapStore((s) => s.setAssignmentSelection);
  const clearAssignmentSelection = useMapStore((s) => s.clearAssignmentSelection);
  const setAssignmentScope = useMapStore((s) => s.setAssignmentScope);
  const processing = useProcessingStore((s) => s.processing);
  const processingStage = useProcessingStore((s) => s.stage);

  const [planters, setPlanters] = useState([]);
  const [organizations, setOrganizations] = useState([]);
  const [plantingSchedules, setPlantingSchedules] = useState([]);
  const [assignmentActivityChoice, setAssignmentActivityChoice] = useState('');
  const [loading, setLoading] = useState(true);

  // One unified assign form: one clicked point and many clicked points share
  // the same selected-id list and batch endpoint.
  const [assignOrganizationId, setAssignOrganizationId] = useState(null);
  const [assignmentCount, setAssignmentCount] = useState(null);
  const [assignBusy, setAssignBusy] = useState(false);
  const [assignError, setAssignError] = useState('');
  const [assignSuccess, setAssignSuccess] = useState('');
  const [noAvailableSite, setNoAvailableSite] = useState(null);

  const loadSequence = useRef(0);
  const dataLoaded = useRef(false);
  const loadData = useCallback(async () => {
    const sequence = ++loadSequence.current;
    setLoading(true);
    try {
      const [pRes, oRes, sRes] = await Promise.all([
        fetch(`${API}/api/planters/?include_inactive=true`),
        fetch(`${API}/api/planter-auth/organizations`),
        fetch(`${API}/api/planting-schedules`),
      ]);
      if (![pRes, oRes, sRes].every((response) => response.ok)) throw new Error('Could not load planter data.');
      const [nextPlanters, nextOrganizations, nextSchedules] = await Promise.all([pRes.json(), oRes.json(), sRes.json()]);
      if (sequence !== loadSequence.current) return;
      setPlanters(nextPlanters);
      setOrganizations(nextOrganizations.organizations);
      setPlantingSchedules(nextSchedules.schedules || []);
    } catch (err) {
      console.error('Failed to load planter data:', err);
    } finally {
      if (sequence === loadSequence.current) { dataLoaded.current = true; setLoading(false); }
    }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => {
      if (!dataLoaded.current) void loadData();
    }, 0);
    const refresh = () => { void loadData(); };
    window.addEventListener('mv:data-changed', refresh);
    return () => {
      window.clearTimeout(timer);
      loadSequence.current += 1;
      window.removeEventListener('mv:data-changed', refresh);
    };
  }, [loadData]);

  // Drop assignment selection on unmount so a forgotten selection doesn't
  // leak into other pages (the same map instance is shared).
  useEffect(() => () => {
    clearAssignmentSelection();
    setAssignmentScope(null, null);
  }, [clearAssignmentSelection, setAssignmentScope]);

  const activePlanters = planters.filter((p) => p.status === 'active');
  const assignmentOrganizations = organizations.map((organization) => {
    const account = planters.find((planter) => Number(planter.organization_id) === Number(organization.id));
    return {
      ...account,
      organization_id: organization.id,
      organization_name: organization.name,
      registration_pending: !account || account.registration_pending,
    };
  }).filter((organization) => !organization.status || organization.status === 'active');
  const selectedAssignmentPlanter = assignmentOrganizations.find(
    (organization) => Number(organization.organization_id) === Number(assignOrganizationId),
  );
  const organizationProjectSites = (projectSites?.features || []).filter((feature) => {
    const ownerId = feature?.properties?.organization_id;
    return selectedAssignmentPlanter?.organization_id != null
      && Number(ownerId) === Number(selectedAssignmentPlanter.organization_id);
  });
  const defaultSiteId = organizationProjectSites[0]?.id ?? organizationProjectSites[0]?.properties?.id ?? null;
  const activityOptions = plantingSchedules.filter((activity) =>
    Number(activity.organization_id) === Number(assignOrganizationId)
    && activity.appointment_type === 'tree_planting'
    && ['confirmed', 'in_progress'].includes(activity.status));
  const assignmentActivity = activityOptions.find((activity) => String(activity.id) === String(assignmentActivityChoice))
    || (activityOptions.length === 1 ? activityOptions[0] : null);
  const activitySite = organizationProjectSites.find((site) => Number(site.id ?? site.properties?.id) === Number(assignmentActivity?.project_site_id));
  const assignProjectSiteId = assignmentActivity?.project_site_id != null
    ? activitySite?.id ?? activitySite?.properties?.id ?? null : defaultSiteId;
  useEffect(() => {
    setAssignmentScope(selectedAssignmentPlanter?.organization_id ?? null, assignProjectSiteId);
  }, [assignProjectSiteId, selectedAssignmentPlanter?.organization_id, setAssignmentScope]);
  const assignmentLocked = processing;
  const scopeOrganizationId = selectedAssignmentPlanter?.organization_id == null ? null : Number(selectedAssignmentPlanter.organization_id);
  const scopeProjectSiteId = assignProjectSiteId == null ? null : Number(assignProjectSiteId);
  const availablePoints = useMemo(() => availableOrganizationPoints(points, scopeOrganizationId, scopeProjectSiteId),
    [points, scopeOrganizationId, scopeProjectSiteId]);
  const requestedCount = assignmentCount ?? availablePoints.length;
  const validCount = !loadingPoints && Number.isInteger(requestedCount) && requestedCount > 0 && requestedCount <= availablePoints.length;
  const pointIdsToAssign = useMemo(() => validCount ? availablePoints.slice(0, requestedCount).map((point) => point.id) : [],
    [availablePoints, requestedCount, validCount]);
  useEffect(() => { setAssignmentSelection(pointIdsToAssign); }, [pointIdsToAssign, setAssignmentSelection]);
  const selectedPointIdSet = new Set(pointIdsToAssign.map((id) => Number(id)));
  const selectedSpeciesCounts = {};
  for (const point of points) {
    if (!selectedPointIdSet.has(Number(point.id))) continue;
    const label = ASSIGNMENT_SPECIES_LABELS[normalizeAssignmentSpecies(point.species)] || 'Species not recorded';
    selectedSpeciesCounts[label] = (selectedSpeciesCounts[label] || 0) + 1;
  }
  const selectedSpeciesSummary = Object.entries(selectedSpeciesCounts)
    .map(([label, count]) => `${count} ${label}`).join(' · ');

  const warnIfNoAvailablePoints = (planter, site) => {
    const siteId = site?.id ?? site?.properties?.id;
    setAssignError('');
    setAssignSuccess('');
    setNoAvailableSite(
      planter && siteId != null && !loadingPoints
        && availableOrganizationPoints(points, planter.organization_id, siteId).length === 0
        ? { organizationName: planter.organization_name, siteName: site.properties?.name || 'this project site' }
        : null,
    );
  };

  const handleAssign = async () => {
    setAssignError('');
    setAssignSuccess('');
    if (assignmentLocked) {
      setAssignError('Planting point assignment is locked while image processing is running.');
      return;
    }
    if (!assignOrganizationId) {
      feedback.reject({ organization: 'Choose the organization receiving these points.' });
      return;
    }
    if (!assignProjectSiteId) {
      feedback.reject({ organization: 'This organization needs a mapped planting area. Create or review its area in Zone Editor before assigning points.' });
      return;
    }
    if (activityOptions.length && !assignmentActivity) {
      feedback.reject({ activity: 'Choose the planting activity receiving these points.' });
      return;
    }
    if (!pointIdsToAssign.length) {
      feedback.reject({ point_count: `Enter a whole number from 1 to ${availablePoints.length}.` });
      return;
    }
    if (!feedback.validate()) return;
    setAssignBusy(true);
    try {
      await formRequest(`${API}/api/planters/organizations/${assignOrganizationId}/assignments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          planting_point_ids: pointIdsToAssign,
          site_zone_id: assignProjectSiteId,
          ...(assignmentActivity ? { planting_schedule_id: assignmentActivity.id } : {}),
        }),
      });
      const planter = selectedAssignmentPlanter;
      setAssignSuccess(
        `Assigned ${pointIdsToAssign.length} point${pointIdsToAssign.length === 1 ? '' : 's'} to ${planter?.organization_name || 'organization'}.`,
      );
      clearAssignmentSelection();
      setAssignmentCount(null);
      // Refresh zones too — each new assignment auto-generates a site zone
      // polygon (convex hull of its points) that should appear on the map
      // immediately without a manual page reload.
      await Promise.all([fetchPoints(), fetchZones(), loadData()]);
    } catch (err) {
      if (!feedback.fromServer(err)) setAssignError(err.message);
    } finally {
      setAssignBusy(false);
    }
  };

  return (
    <Panel plantingTool title="Planting Assignments" subtitle={`${activePlanters.length} active organizations`} openKey={openPanel} onOpenKeyChange={setOpenPanel}>
      <PanelCard
        title="Assign available points"
        panelKey="assign"
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z"/><circle cx="12" cy="10" r="3"/></svg>
        }
      >
        {assignmentLocked && (
          <div className="assignment-lock-banner">
            <strong>Assignment locked</strong>
            <span>
              Image processing is running{processingStage ? `: ${processingStage}` : ''}.
            </span>
          </div>
        )}
        <div className="assign-card" onChangeCapture={feedback.onChange}>
          <FormErrorSummary feedback={feedback} />
          <p className="text-sm" style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
            Select an organization and choose how many points to assign. Each participant receives staggered point locations forming zigzag strips, divided equally. Short strips at the site boundary continue into the next strip.
          </p>

          <div className="form-group" style={{ marginTop: 10 }}>
            <label className="form-label" htmlFor="assign-organization">Assign to organization</label>
            <select {...feedback.props('organization')}
              required
              className="form-select"
              id="assign-organization"
              value={assignOrganizationId || ''}
              onChange={(e) => {
                const nextPlanterId = Number(e.target.value) || null;
                feedback.clear();
                const nextPlanter = assignmentOrganizations.find((organization) => Number(organization.organization_id) === nextPlanterId);
                setAssignOrganizationId(nextPlanterId);
                setAssignmentActivityChoice('');
                setAssignmentCount(null);
                const ownedSites = (projectSites?.features || []).filter((feature) => Number(feature.properties?.organization_id) === Number(nextPlanter?.organization_id));
                const activities = plantingSchedules.filter((activity) => Number(activity.organization_id) === nextPlanterId
                  && activity.appointment_type === 'tree_planting' && ['confirmed', 'in_progress'].includes(activity.status));
                const linkedSiteId = activities.length === 1 ? activities[0].project_site_id : null;
                const site = linkedSiteId != null
                  ? ownedSites.find((feature) => Number(feature.id ?? feature.properties?.id) === Number(linkedSiteId)) : ownedSites[0];
                const siteId = site?.id ?? site?.properties?.id ?? null;
                warnIfNoAvailablePoints(nextPlanter, site);
                clearAssignmentSelection();
                setAssignmentScope(nextPlanter?.organization_id ?? null, siteId);
              }}
              disabled={assignmentLocked || assignBusy || loading || loadingPoints || !projectSites}
            >
              <option value="">Select an organization...</option>
              {assignmentOrganizations.map((p) => (
                <option key={p.organization_id} value={p.organization_id}>
                  {p.organization_name}
                </option>
              ))}
            </select>
              <FieldError feedback={feedback} field="organization" />
            {selectedAssignmentPlanter?.organization_name && (
              <span className="text-sm" style={{ color: 'var(--color-primary)', lineHeight: 1.4 }}>
                <span style={{ display: 'inline-block', width: 10, height: 10, borderRadius: '50%', background: getPlanterColor(selectedAssignmentPlanter.organization_id), marginRight: 6 }} />
                {selectedAssignmentPlanter.registration_pending
                  ? 'Login not created yet · Points can be reserved now'
                  : `${selectedAssignmentPlanter.participant_count} participant slots · Shared organization code`}
              </span>
            )}
            {selectedAssignmentPlanter && !assignProjectSiteId && !loadingPoints && (
              <span className="text-sm" role="status">
                {assignmentActivity?.project_site_id != null
                  ? 'This activity’s planting area is unavailable. Review its area in Scheduling.'
                  : 'This organization has no mapped planting area. Create one in Zone Editor first.'}
              </span>
            )}
          </div>

          <div className="form-group" style={{ marginTop: 10 }}>
            <label className="form-label" htmlFor="assign-activity">Planting activity</label>
            <select {...feedback.props('activity')} id="assign-activity" className="form-select"
              value={assignmentActivity?.id || ''} required={activityOptions.length > 0}
              onChange={(event) => {
                setAssignmentActivityChoice(event.target.value);
                setAssignmentCount(null);
                setNoAvailableSite(null);
                feedback.clear();
              }}
              disabled={assignmentLocked || assignBusy || loading || !activityOptions.length}>
              <option value="">{activityOptions.length ? 'Select a planting activity' : 'Unscheduled planting'}</option>
              {activityOptions.map((activity) => <option key={activity.id} value={activity.id}>
                {activity.title} · {activity.date}
              </option>)}
            </select>
            <FieldError feedback={feedback} field="activity" />
            <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
              {assignmentActivity?.project_site_id == null && assignmentActivity
                ? "These points belong to the selected activity. Its planting area will use this organization's mapped area."
                : activityOptions.length ? 'These points belong to the selected activity.' : 'No confirmed planting activity is available for this site.'}
            </span>
          </div>

          <div className="form-group" style={{ marginTop: 10 }}>
            <label className="form-label" htmlFor="organization-point-count">Number of points to assign</label>
            <input {...feedback.props('point_count')} id="organization-point-count" className="form-input" type="number" required min="1" max={availablePoints.length} step="1" value={requestedCount || ''} onChange={(event) => setAssignmentCount(Number(event.target.value))} disabled={!assignProjectSiteId || assignBusy || assignmentLocked || loadingPoints || availablePoints.length === 0} />
              <FieldError feedback={feedback} field="point_count" />
            <span className="text-sm">
              {loadingPoints ? 'Checking available points…' : `${availablePoints.length} available points for this organization.`}
            </span>
            {assignProjectSiteId && !loadingPoints && availablePoints.length === 0 && (
              <span className="text-sm" role="status">No points are available to assign. Review this organization's mapped area in Zone Editor.</span>
            )}
          </div>

          {selectedAssignmentPlanter && (
            <div className="assign-point-info" role="status">
              <strong>{pointIdsToAssign.length} points{!selectedAssignmentPlanter.registration_pending && ` / ${selectedAssignmentPlanter.participant_count} participants`}</strong>
              {selectedSpeciesSummary && <span className="text-sm">{selectedSpeciesSummary}</span>}
              {selectedAssignmentPlanter.registration_pending ? (
                <span className="text-sm">Select a confirmed planting activity and prepare its organization code below. Its participant count determines the initial allocation.</span>
              ) : <span className="text-sm">Each participant receives {Math.floor(pointIdsToAssign.length / selectedAssignmentPlanter.participant_count)}{pointIdsToAssign.length % selectedAssignmentPlanter.participant_count ? `–${Math.ceil(pointIdsToAssign.length / selectedAssignmentPlanter.participant_count)}` : ''} points in a first batch. Later batches balance existing allocations.</span>}
              <span className="text-sm">All planted points count toward {selectedAssignmentPlanter.organization_name}.</span>
            </div>
          )}

          <button
            className="btn btn-primary btn-sm"
            style={{ marginTop: 8, width: '100%' }}
            onClick={handleAssign}
            disabled={
              assignmentLocked
              || assignBusy
              || !assignOrganizationId
              || !assignProjectSiteId
              || loadingPoints
              || availablePoints.length === 0
            }
          >
            {assignBusy
              ? 'Assigning…'
              : assignmentLocked
                ? 'Assignment Locked'
                : `Assign ${pointIdsToAssign.length || ''} Point${pointIdsToAssign.length === 1 ? '' : 's'}`.trim()}
          </button>
          <FormMessage error={assignError} success={assignSuccess} loading={assignBusy ? 'Assigning planting points…' : ''} />
        </div>
      </PanelCard>

      {selectedAssignmentPlanter && (
        <PanelCard title="Organization code" defaultOpen={selectedAssignmentPlanter.registration_pending}>
          <OrganizationAccess key={`${selectedAssignmentPlanter.organization_id}:${assignmentActivity?.id || ''}`} organization={selectedAssignmentPlanter}
            activity={assignmentActivity}
            onCreated={() => { void loadData(); }} />
        </PanelCard>
      )}

      <Modal
        open={Boolean(noAvailableSite)}
        title="No available points"
        variant="warning"
        cancelLabel="Close"
        onCancel={() => setNoAvailableSite(null)}
      >
        <p>
          No points are available to assign in <strong>{noAvailableSite?.siteName}</strong> for <strong>{noAvailableSite?.organizationName}</strong>.
        </p>
        <p>Review this organization's mapped area in Zone Editor, or choose another organization with available points.</p>
      </Modal>

    </Panel>
  );
}
