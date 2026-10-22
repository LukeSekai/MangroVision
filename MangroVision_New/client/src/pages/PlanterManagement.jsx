import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { Link, useLocation } from 'react-router-dom';
import { availableOrganizationPoints } from '../utils/organizationAssignment';
import { availableSitePoints } from '../utils/projectSiteSelection';
import { useMapStore } from '../stores/mapStore';
import { useProcessingStore } from '../stores/processingStore';
import { Panel, PanelCard } from '../components/Panel';
import Modal from '../components/Modal';
import OrganizationAccess from '../components/OrganizationAccess';
import PlantingPointSelector from '../components/PlantingPointSelector';
import useFormFeedback from '../utils/useFormFeedback';
import { FieldError, FormErrorSummary, FormMessage } from '../components/FormFeedback';
import { formRequest } from '../utils/formValidation';
import './PlanterManagement.css';

const API = import.meta.env.VITE_API_BASE || '';

export default function PlanterManagement() {
  const location = useLocation();
  const active = location.pathname === '/planters';
  const [openPanel, setOpenPanel] = useState('assign');
  const points = useMapStore((s) => s.points);
  const map = useMapStore((s) => s.mapInstance);
  const loadingPoints = useMapStore((s) => s.loadingPoints);
  const fetchPoints = useMapStore((s) => s.fetchPoints);
  const fetchZones = useMapStore((s) => s.fetchZones);
  const selected = useMapStore((s) => s.assignmentSelectedPointIds);
  const setSelection = useMapStore((s) => s.setAssignmentSelection);
  const clearSelection = useMapStore((s) => s.clearAssignmentSelection);
  const setScope = useMapStore((s) => s.setAssignmentScope);
  const processing = useProcessingStore((s) => s.processing);
  const stage = useProcessingStore((s) => s.stage);
  const [planters, setPlanters] = useState([]);
  const [organizations, setOrganizations] = useState([]);
  const [schedules, setSchedules] = useState([]);
  const [organizationId, setOrganizationId] = useState(() => new URLSearchParams(location.search).get('organization_id') || '');
  const [activityId, setActivityId] = useState(() => new URLSearchParams(location.search).get('activity_id') || '');
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState('');
  const [busy, setBusy] = useState(false);
  const [error, setError] = useState('');
  const [success, setSuccess] = useState('');
  const [assignmentReview, setAssignmentReview] = useState(null);
  const submitting = useRef(false);
  const sequence = useRef(0);
  const loaded = useRef(false);
  const initialized = useRef(false);
  const appliedNavigation = useRef(null);
  const invalidate = useCallback(() => { ++sequence.current; }, []);
  const feedback = useFormFeedback({
    organization: { label: 'Organization', aliases: ['organization_id'] },
    activity: { label: 'Planting activity', aliases: ['planting_schedule_id'], serverTerms: ['planting activity'] },
  });

  const load = useCallback(async () => {
    const current = ++sequence.current;
    setLoading(true);
    try {
      const responses = await Promise.all([
        fetch(`${API}/api/planters/?include_inactive=true`),
        fetch(`${API}/api/planter-auth/organizations`),
        fetch(`${API}/api/planting-schedules`),
      ]);
      if (!responses.every((response) => response.ok)) throw new Error('Could not load organizations and planting activities.');
      const [accounts, owners, activities] = await Promise.all(responses.map((response) => response.json()));
      if (current !== sequence.current) return;
      setPlanters(accounts); setOrganizations(owners.organizations || []); setSchedules(activities.schedules || []); setLoadError('');
    } catch (failure) { if (current === sequence.current) setLoadError(failure.message); }
    finally { if (current === sequence.current) { loaded.current = true; setLoading(false); } }
  }, []);

  useEffect(() => {
    const timer = window.setTimeout(() => { if (!loaded.current) void load(); }, 0);
    const refresh = () => { if (!submitting.current) void load(); };
    window.addEventListener('mv:data-changed', refresh);
    return () => { window.clearTimeout(timer); invalidate(); window.removeEventListener('mv:data-changed', refresh); };
  }, [load, invalidate]);

  useEffect(() => {
    const params = new URLSearchParams(location.search);
    if (location.pathname !== '/planters' || !params.has('organization_id')) return undefined;
    if (appliedNavigation.current === location.key) return undefined;
    const timer = window.setTimeout(() => {
      appliedNavigation.current = location.key;
      setOrganizationId(params.get('organization_id') || '');
      setActivityId(params.get('activity_id') || '');
      clearSelection(); setError(''); setSuccess('');
    }, 0);
    return () => window.clearTimeout(timer);
  }, [location.key, location.pathname, location.search, clearSelection]);

  useEffect(() => {
    if (!initialized.current) { initialized.current = true; clearSelection(); }
    return () => { setScope(null, null); };
  }, [clearSelection, setScope]);

  const choices = organizations.map((organization) => {
    const account = planters.find((item) => Number(item.organization_id) === Number(organization.id));
    return { ...account, organization_id: organization.id, organization_name: organization.name,
      registration_pending: !account || account.registration_pending };
  }).filter((item) => !item.status || item.status === 'active');
  const owner = choices.find((item) => String(item.organization_id) === String(organizationId));
  const activities = schedules.filter((activity) => !owner?.permanent_access
    && Number(activity.organization_id) === Number(organizationId)
    && activity.appointment_type === 'tree_planting' && ['confirmed', 'in_progress'].includes(activity.status));
  const activity = activities.find((item) => String(item.id) === String(activityId)) || (activities.length === 1 ? activities[0] : null);
  // Existing activities keep their recorded area; new activities claim their
  // selected points automatically. There is no site selector or creation form.
  const linkedSiteId = activity?.project_site_id == null ? null : Number(activity.project_site_id);
  const available = useMemo(() => linkedSiteId == null ? availableSitePoints(points)
    : availableOrganizationPoints(points, Number(organizationId), Number(linkedSiteId)), [points, organizationId, linkedSiteId]);
  const eligible = useMemo(() => new Set(available.map((point) => Number(point.id))), [available]);
  const validSelected = selected.filter((id) => eligible.has(Number(id)));
  const staleSelection = validSelected.length !== selected.length;
  const ready = Boolean(owner && (owner.permanent_access || activity));
  const locked = !active || processing || busy || loading || loadingPoints || Boolean(loadError) || !ready || Boolean(assignmentReview);
  useEffect(() => { setScope(active ? owner?.organization_id ?? null : null, active ? linkedSiteId : null); },
    [active, owner?.organization_id, linkedSiteId, setScope]);

  function resetSelection() { clearSelection(); feedback.clear(); setError(''); setSuccess(''); }
  async function refreshPoints() { await Promise.all([fetchPoints({ force: true }), fetchZones({ force: true }), load()]); }
  async function assign() {
    if (submitting.current || locked) return;
    setError(''); setSuccess('');
    if (!owner) { feedback.reject({ organization: 'Choose the organization receiving these points.' }); return; }
    if (!owner.permanent_access && !activity) { feedback.reject({ activity: 'Choose a confirmed tree-planting activity.' }); return; }
    if (staleSelection) { setError('Some selected points are no longer available. Review or clear the selection.'); return; }
    if (!validSelected.length) { setError('Select at least one available point on the map.'); return; }
    if (!feedback.validate()) return;
    setAssignmentReview({ organizationId: owner.organization_id, organizationName: owner.organization_name,
      activityId: activity?.id, activityTitle: activity?.title, linkedSiteId,
      pointIds: [...validSelected] });
  }

  async function confirmAssignment() {
    if (!assignmentReview || submitting.current || !active || processing || loading || loadingPoints) return;
    const reviewed = assignmentReview;
    if (staleSelection || reviewed.pointIds.some((id) => !eligible.has(Number(id)))
        || String(owner?.organization_id) !== String(reviewed.organizationId)
        || String(activity?.id) !== String(reviewed.activityId)) {
      setAssignmentReview(null);
      setError('The available points or activity changed. Review the selection again.');
      return;
    }
    submitting.current = true; setBusy(true);
    try {
      await formRequest(`${API}/api/planters/organizations/${reviewed.organizationId}/assignments`, {
        method: 'POST', headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ planting_point_ids: reviewed.pointIds, ...(reviewed.activityId != null ? { planting_schedule_id: reviewed.activityId } : {}) }),
      });
      setAssignmentReview(null);
      setSuccess(`Assigned ${reviewed.pointIds.length} point${reviewed.pointIds.length === 1 ? '' : 's'} to ${reviewed.organizationName}. Its project site is shown on the map.`);
      clearSelection();
      await refreshPoints();
    } catch (failure) { setAssignmentReview(null); if (!feedback.fromServer(failure)) setError(failure.message); }
    finally { submitting.current = false; setBusy(false); }
  }

  return <Panel plantingTool title="Planting Assignments" subtitle="Assign points and form the organization's project site" openKey={openPanel} onOpenKeyChange={setOpenPanel}>
    <PanelCard title="Assign available points" panelKey="assign">
      {processing && <div className="assignment-lock-banner"><strong>Assignment locked</strong><span>Image processing is running{stage ? `: ${stage}` : ''}.</span></div>}
      <FormMessage error={loadError} loading={loading ? 'Loading organizations and activities...' : ''} onRetry={load} />
      <div className="assign-card" onChangeCapture={feedback.onChange}>
        <FormErrorSummary feedback={feedback} />
        <p className="text-sm">Choose the organization and select available planting points. Assigning them automatically forms the project site.</p>
        <div className="form-group">
          <label className="form-label" htmlFor="assign-organization">Assign to organization</label>
          <select {...feedback.props('organization')} id="assign-organization" className="form-select" required value={organizationId}
            disabled={processing || busy || loading || Boolean(loadError) || Boolean(assignmentReview)} onChange={(event) => {
              setOrganizationId(event.target.value); setActivityId(''); resetSelection();
            }}>
            <option value="">Select an organization</option>
            {choices.map((item) => <option key={item.organization_id} value={item.organization_id}>{item.organization_name}</option>)}
          </select><FieldError feedback={feedback} field="organization" />
        </div>
        {owner && !owner.permanent_access && <div className="form-group">
          <label className="form-label" htmlFor="assign-activity">Planting activity</label>
          <select {...feedback.props('activity')} id="assign-activity" className="form-select" required value={activity?.id || ''}
            disabled={processing || busy || loading || !activities.length || Boolean(assignmentReview)} onChange={(event) => { setActivityId(event.target.value); resetSelection(); }}>
            <option value="">{activities.length ? 'Select a confirmed activity' : 'No confirmed tree-planting activity'}</option>
            {activities.map((item) => <option key={item.id} value={item.id}>{item.title} - {item.date || item.start_at?.slice(0, 10)}</option>)}
          </select><FieldError feedback={feedback} field="activity" />
          {!activities.length && <p className="text-sm">Schedule and confirm a tree-planting activity for this organization in <Link to="/scheduling">Scheduling</Link> before assigning points.</p>}
        </div>}
        {owner?.permanent_access && <p className="text-sm">LGU can assign available points without scheduling an activity.</p>}
        {ready && <>
          <p className="text-sm">{loadingPoints ? 'Checking available points...' : `${available.length} available points. Click a point or use the brush to select locations.`}</p>
          {active && <PlantingPointSelector key={`${organizationId}:${activity?.id || 'lgu'}`} map={map} points={available}
            selected={validSelected} onChange={setSelection} disabled={locked} />}
          {!map && <p role="status" className="text-sm">Loading the planting map...</p>}
          {staleSelection && <p role="alert" className="text-sm">Some selected points are no longer available. Review or clear the selection.</p>}
          <p className="text-sm">Selected points remain awaiting planting. Survival and monitoring start from recorded planting.</p>
        </>}
        <button type="button" className="btn btn-primary btn-sm" style={{ marginTop: 8, width: '100%' }} onClick={assign}
          disabled={locked || !validSelected.length || staleSelection}>{busy ? 'Assigning...' : `Assign ${validSelected.length || ''} point${validSelected.length === 1 ? '' : 's'}`.replace('  ', ' ')}</button>
        <FormMessage error={error} success={success} loading={busy ? 'Assigning points and saving the project site...' : ''} />
        <button type="button" className="btn btn-ghost btn-sm" disabled={busy || processing || loadingPoints} onClick={() => void refreshPoints()}>Refresh available points</button>
      </div>
    </PanelCard>
    {owner && <PanelCard title="Organization code" defaultOpen={owner.registration_pending}>
      {owner.permanent_access ? <p className="text-sm">LGU uses its permanent planter account. <Link to="/monitoring/lgu-sites">View credentials in LGU Sites</Link>.</p>
        : <OrganizationAccess key={`${owner.organization_id}:${activity?.id || ''}`} organization={owner} activity={activity} onCreated={() => void load()} />}
    </PanelCard>}
    <Modal open={Boolean(assignmentReview)} title="Assign selected points?" variant="warning"
      initialFocus="cancel" confirmLabel="Assign points" cancelLabel="Cancel" busy={busy}
      confirmDisabled={processing || loading || loadingPoints || Boolean(loadError) || staleSelection}
      onConfirm={confirmAssignment} onCancel={() => { if (!busy) setAssignmentReview(null); }}>
      {assignmentReview && <>
        <p>Assign <strong>{assignmentReview.pointIds.length} {assignmentReview.pointIds.length === 1 ? 'point' : 'points'}</strong> to <strong>{assignmentReview.organizationName}</strong>{assignmentReview.activityTitle ? <> for {assignmentReview.activityTitle}</> : null}?</p>
        <p>{assignmentReview.linkedSiteId == null ? 'These points will form the project site.' : 'These points will be added to the saved project site.'} Planting is recorded separately.</p>
      </>}
    </Modal>
  </Panel>;
}
