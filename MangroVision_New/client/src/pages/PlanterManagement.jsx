import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { availableOrganizationPoints } from '../utils/organizationAssignment';
import { countMapPointStatuses } from '../utils/mapPointStats';
import { useMapStore } from '../stores/mapStore';
import { useProcessingStore } from '../stores/processingStore';
import { useAuthStore } from '../stores/authStore';
import { Panel, PanelCard } from '../components/Panel';
import Modal from '../components/Modal';
import PlanterActivityReport from './PlanterActivityReport';
import ParticipantDevices from '../components/ParticipantDevices';
import { getPlanterColor } from '../utils/planterColors';
import { POINT_STATUS_LABELS as STATUS_LABEL } from '../utils/pointStatus';
import { useLocation } from 'react-router-dom';
import useFormFeedback from '../utils/useFormFeedback';
import { FieldError, FormErrorSummary } from '../components/FormFeedback';
import { submissionError } from '../utils/formValidation';
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
    organization: { label: 'Organization', aliases: ['organization_id'] },
    site: { label: 'Project site', aliases: ['site_zone_id'], serverTerms: ['project site'] },
    activity: { label: 'Planting activity', aliases: ['planting_schedule_id'], serverTerms: ['planting activity'] },
    point_count: { label: 'Number of points', serverTerms: ['point count'] },
  });
  useEffect(() => {
    if (requestedSection === 'assign') queueMicrotask(() => setOpenPanel(requestedSection));
  }, [requestedSection, location.key]);
  const points = useMapStore((s) => s.points);
  const pointCounts = useMemo(() => countMapPointStatuses(points), [points]);
  const loadingPoints = useMapStore((s) => s.loadingPoints);
  const fetchPoints = useMapStore((s) => s.fetchPoints);
  const fetchZones = useMapStore((s) => s.fetchZones);
  const projectSites = useMapStore((s) => s.projectSites);
  const setAssignmentSelection = useMapStore((s) => s.setAssignmentSelection);
  const clearAssignmentSelection = useMapStore((s) => s.clearAssignmentSelection);
  const setAssignmentScope = useMapStore((s) => s.setAssignmentScope);
  const processing = useProcessingStore((s) => s.processing);
  const processingStage = useProcessingStore((s) => s.stage);
  const adminToken = useAuthStore((s) => s.token);

  const [planters, setPlanters] = useState([]);
  const [organizations, setOrganizations] = useState([]);
  const [plantingSchedules, setPlantingSchedules] = useState([]);
  const [assignmentActivityChoice, setAssignmentActivityChoice] = useState('');
  const [dashStats, setDashStats] = useState(null);
  const [loading, setLoading] = useState(true);

  // One unified assign form: one clicked point and many clicked points share
  // the same selected-id list and batch endpoint.
  const [assignOrganizationId, setAssignOrganizationId] = useState(null);
  const [assignmentCount, setAssignmentCount] = useState(null);
  const [assignProjectSiteChoice, setAssignProjectSiteId] = useState(null);
  const [assignBusy, setAssignBusy] = useState(false);
  const [assignError, setAssignError] = useState('');
  const [assignSuccess, setAssignSuccess] = useState('');
  const [noAvailableSite, setNoAvailableSite] = useState(null);

  const [reportOpen, setReportOpen] = useState(false);

  const [shareLink, setShareLink] = useState(null);
  const [shareBusy, setShareBusy] = useState(false);
  const [shareError, setShareError] = useState('');
  const [shareCopied, setShareCopied] = useState('');
  const shareCopiedTimerRef = useRef(null);

  const loadSequence = useRef(0);
  const dataLoaded = useRef(false);
  const shareLoaded = useRef(null);

  const showShareCopied = (message) => {
    setShareCopied(message);
    if (shareCopiedTimerRef.current) window.clearTimeout(shareCopiedTimerRef.current);
    shareCopiedTimerRef.current = window.setTimeout(() => setShareCopied(''), 2200);
  };

  const loadShareStatus = useCallback(async ({ silent = false } = {}) => {
    if (!silent) setShareError('');
    if (!adminToken) {
      if (!silent) setShareError('Sign in again to manage share links.');
      return;
    }
    try {
      const res = await fetch(`${API}/api/share/field-link`);
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(payload.detail || 'Could not check share link.');
      setShareLink(payload);
      shareLoaded.current = adminToken;
    } catch (err) {
      console.error('Share link status failed:', err);
      if (!silent) setShareError(err.message || 'Could not check share link.');
    }
  }, [adminToken]);

  const copyTextToClipboard = async (text) => {
    if (!text) throw new Error('No link to copy.');
    if (navigator.clipboard?.writeText && window.isSecureContext) {
      await navigator.clipboard.writeText(text);
      return;
    }
    const textarea = document.createElement('textarea');
    textarea.value = text;
    textarea.setAttribute('readonly', '');
    textarea.style.position = 'fixed';
    textarea.style.top = '-9999px';
    document.body.appendChild(textarea);
    textarea.select();
    const copied = document.execCommand('copy');
    document.body.removeChild(textarea);
    if (!copied) throw new Error('Clipboard permission was blocked.');
  };

  const loadData = useCallback(async () => {
    const sequence = ++loadSequence.current;
    setLoading(true);
    try {
      const [pRes, dRes, oRes, sRes] = await Promise.all([
        fetch(`${API}/api/planters/?include_inactive=true`),
        fetch(`${API}/api/planters/dashboard`),
        fetch(`${API}/api/planter-auth/organizations`),
        fetch(`${API}/api/planting-schedules`),
      ]);
      if (![pRes, dRes, oRes, sRes].every((response) => response.ok)) throw new Error('Could not load planter data.');
      const [nextPlanters, nextStats, nextOrganizations, nextSchedules] = await Promise.all([pRes.json(), dRes.json(), oRes.json(), sRes.json()]);
      if (sequence !== loadSequence.current) return;
      setPlanters(nextPlanters);
      setOrganizations(nextOrganizations.organizations);
      setDashStats(nextStats);
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
      if (shareLoaded.current !== adminToken) void loadShareStatus({ silent: true });
    }, 0);
    const refresh = () => { void loadData(); };
    window.addEventListener('mv:data-changed', refresh);
    return () => {
      window.clearTimeout(timer);
      loadSequence.current += 1;
      window.removeEventListener('mv:data-changed', refresh);
    };
  }, [adminToken, loadData, loadShareStatus]);

  useEffect(() => () => {
    if (shareCopiedTimerRef.current) window.clearTimeout(shareCopiedTimerRef.current);
  }, []);

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
  const assignProjectSiteId = organizationProjectSites.some((site) => Number(site.id ?? site.properties?.id) === Number(assignProjectSiteChoice))
    ? assignProjectSiteChoice : defaultSiteId;
  const activityOptions = plantingSchedules.filter((activity) =>
    Number(activity.organization_id) === Number(assignOrganizationId)
    && Number(activity.project_site_id) === Number(assignProjectSiteId)
    && activity.appointment_type === 'tree_planting'
    && ['confirmed', 'in_progress'].includes(activity.status));
  const assignmentActivity = activityOptions.find((activity) => String(activity.id) === String(assignmentActivityChoice))
    || (activityOptions.length === 1 ? activityOptions[0] : null);
  useEffect(() => {
    setAssignmentScope(selectedAssignmentPlanter?.organization_id ?? null, assignProjectSiteId);
  }, [assignProjectSiteId, selectedAssignmentPlanter?.organization_id, setAssignmentScope]);
  const assignmentLocked = processing;
  const shareFieldUrl = shareLink?.field_url || '';
  const shareHosted = shareLink?.provider === 'hosted';
  const shareLinkActive = Boolean(shareLink?.active && shareLink?.field_url);

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
      feedback.reject({ site: 'Choose a project site owned by this organization.' });
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
      const res = await fetch(`${API}/api/planters/organizations/${assignOrganizationId}/assignments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          planting_point_ids: pointIdsToAssign,
          site_zone_id: assignProjectSiteId,
          ...(assignmentActivity ? { planting_schedule_id: assignmentActivity.id } : {}),
        }),
      });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw submissionError(payload.detail, 'Assignment failed. Please try again.');
      }
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

  const handleGenerateCloudflareLink = async () => {
    setShareBusy(true);
    setShareError('');
    setShareCopied('');
    if (!adminToken) {
      setShareError('Sign in again to manage share links.');
      setShareBusy(false);
      return;
    }
    try {
      const res = await fetch(
        `${API}/api/share/field-link/cloudflare`,
        { method: 'POST' },
      );
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(payload.detail || 'Could not generate Cloudflare link.');
      setShareLink(payload);
      if (payload.field_url) {
        try {
          await copyTextToClipboard(payload.field_url);
          showShareCopied(payload.provider === 'hosted' ? 'Field link copied.' : 'Cloudflare link copied.');
        } catch (copyError) {
          console.warn('Clipboard copy failed:', copyError);
          setShareError('Link generated, but clipboard permission was blocked.');
        }
      }
    } catch (err) {
      setShareError(err.message || 'Could not generate Cloudflare link.');
    } finally {
      setShareBusy(false);
    }
  };

  const handleCopyShareLink = async () => {
    setShareError('');
    try {
      await copyTextToClipboard(shareFieldUrl);
      showShareCopied('Link copied.');
    } catch (err) {
      setShareError(err.message || 'Could not copy link.');
    }
  };

  const handleStopCloudflareLink = async () => {
    setShareBusy(true);
    setShareError('');
    setShareCopied('');
    if (!adminToken) {
      setShareError('Sign in again to manage share links.');
      setShareBusy(false);
      return;
    }
    try {
      const res = await fetch(
        `${API}/api/share/field-link/stop`,
        { method: 'POST' },
      );
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(payload.detail || 'Could not stop Cloudflare link.');
      setShareLink(payload);
    } catch (err) {
      setShareError(err.message || 'Could not stop Cloudflare link.');
    } finally {
      setShareBusy(false);
    }
  };

  return (
    <Panel title="Planting Assignments" subtitle={`${activePlanters.length} active organizations`} openKey={openPanel} onOpenKeyChange={setOpenPanel}>
      {dashStats && (
        <PanelCard
          title="Overview"
          icon={
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/></svg>
          }
        >
          <div className="planter-stats-grid">
            <div className="stat-card">
              <div className="stat-label">Active organizations</div>
              <div className="stat-value">{dashStats.active_planters}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Active assignments</div>
              <div className="stat-value">{dashStats.active_assignments}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Planned</div>
              <div className="stat-value" style={{ color: 'var(--color-planned)' }}>{pointCounts.planned.toLocaleString()}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Assigned</div>
              <div className="stat-value" style={{ color: 'var(--color-assigned)' }}>{pointCounts.assigned.toLocaleString()}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">{STATUS_LABEL.completed}</div>
              <div className="stat-value" style={{ color: 'var(--color-completed)' }}>{pointCounts.planted.toLocaleString()}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Dead</div>
              <div className="stat-value" style={{ color: '#7f1d1d' }}>{pointCounts.dead.toLocaleString()}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Skipped</div>
              <div className="stat-value" style={{ color: '#6b7280' }}>{pointCounts.skipped.toLocaleString()}</div>
            </div>
            <div className="stat-card">
              <div className="stat-label">Unavailable</div>
              <div className="stat-value" style={{ color: '#f97316' }}>{pointCounts.unavailable.toLocaleString()}</div>
            </div>
          </div>
          <button
            type="button"
            className="btn btn-secondary btn-sm"
            style={{ marginTop: 12, width: '100%', display: 'inline-flex', alignItems: 'center', justifyContent: 'center', gap: 8 }}
            onClick={() => setReportOpen(true)}
          >
            <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
              <path d="M3 3v18h18" />
              <path d="M7 14l4-4 4 4 5-5" />
            </svg>
            View Activity Report
          </button>
        </PanelCard>
      )}

      <PanelCard
        title="Field Share Link"
        defaultOpen={false}
        icon={
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
            <path d="M10 13a5 5 0 0 0 7.07 0l2.83-2.83a5 5 0 0 0-7.07-7.07L11 4.93" />
            <path d="M14 11a5 5 0 0 0-7.07 0L4.1 13.83a5 5 0 1 0 7.07 7.07L13 19.07" />
          </svg>
        }
      >
        <div className="share-card">
          <div className="share-status-row">
            <span
              className={`share-status-dot ${shareLinkActive ? 'share-status-dot-active' : ''}`}
              aria-hidden="true"
            />
            <span>{shareHosted ? 'Field link available' : shareLinkActive ? 'Cloudflare active' : 'Cloudflare inactive'}</span>
          </div>

          <div className="share-link-box">
            <input
              className="share-link-input"
              value={shareFieldUrl}
              placeholder={shareHosted ? 'Field app link' : 'Generate a Cloudflare link'}
              readOnly
              aria-label="Field app share link"
            />
            <button
              type="button"
              className="btn btn-ghost btn-sm btn-icon"
              onClick={handleCopyShareLink}
              disabled={!shareFieldUrl}
              title="Copy field link"
              aria-label="Copy field link"
            >
              <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
                <rect x="9" y="9" width="13" height="13" rx="2" ry="2" />
                <path d="M5 15H4a2 2 0 0 1-2-2V4a2 2 0 0 1 2-2h9a2 2 0 0 1 2 2v1" />
              </svg>
            </button>
          </div>

          <div className="share-actions">
            <button
              type="button"
              className="btn btn-primary btn-sm share-action-primary"
              onClick={shareHosted ? handleCopyShareLink : handleGenerateCloudflareLink}
              disabled={shareBusy}
            >
              {shareBusy ? 'Working...' : shareHosted ? 'Copy Field Link' : 'Generate Cloudflare Link'}
            </button>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => loadShareStatus()}
              disabled={shareBusy}
            >
              Check Status
            </button>
            {shareLinkActive && !shareHosted && (
              <button
                type="button"
                className="btn btn-ghost btn-sm"
                onClick={handleStopCloudflareLink}
                disabled={shareBusy}
              >
                Stop Link
              </button>
            )}
          </div>

          {!shareHosted && <p className="text-sm">Keep the current link while participants are working. A newly generated link needs their saved device recovery codes to restore the same participant numbers.</p>}
          {shareError && <div className="assign-message assign-error">{shareError}</div>}
          {shareCopied && <div className="assign-message assign-success">{shareCopied}</div>}
        </div>
      </PanelCard>

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
                const site = (projectSites?.features || []).find((feature) => Number(feature.properties?.organization_id) === Number(nextPlanter?.organization_id));
                const siteId = site?.id ?? site?.properties?.id ?? null;
                setAssignProjectSiteId(siteId);
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
                  ? 'Account not registered yet · Points can be reserved now'
                  : `${selectedAssignmentPlanter.participant_count} participants · Shared organization account`}
              </span>
            )}
          </div>

          <div className="form-group" style={{ marginTop: 10 }}>
            <label className="form-label" htmlFor="assign-site">Project site</label>
            <select {...feedback.props('site')}
              required
              className="form-select"
              id="assign-site"
              value={assignProjectSiteId || ''}
              onChange={(e) => {
                const nextSiteId = Number(e.target.value) || null;
                feedback.clear();
                setAssignProjectSiteId(nextSiteId);
                setAssignmentActivityChoice('');
                setAssignmentCount(null);
                warnIfNoAvailablePoints(selectedAssignmentPlanter, organizationProjectSites.find((site) => Number(site.id ?? site.properties?.id) === nextSiteId));
                clearAssignmentSelection();
                setAssignmentScope(selectedAssignmentPlanter?.organization_id ?? null, nextSiteId);
              }}
              disabled={assignmentLocked || assignBusy || loadingPoints || !assignOrganizationId || organizationProjectSites.length === 0}
            >
              <option value="">
                {!assignOrganizationId
                  ? 'Select an organization first'
                  : organizationProjectSites.length
                    ? 'Select a project site'
                    : 'No project site for this organization'}
              </option>
              {organizationProjectSites.map((feature) => {
                const props = feature.properties || {};
                const id = feature.id ?? props.id;
                return <option key={id} value={id}>{props.name || `Project Site ${id}`}</option>;
              })}
            </select>
              <FieldError feedback={feedback} field="site" />
            <span className="text-sm" style={{ color: 'var(--text-muted)', lineHeight: 1.4 }}>
              The map automatically locates this organization's project site.
            </span>
          </div>

          <div className="form-group" style={{ marginTop: 10 }}>
            <label className="form-label" htmlFor="assign-activity">Planting activity</label>
            <select {...feedback.props('activity')} id="assign-activity" className="form-select"
              value={assignmentActivity?.id || ''} required={activityOptions.length > 0}
              onChange={(event) => setAssignmentActivityChoice(event.target.value)}
              disabled={assignmentLocked || assignBusy || loading || !activityOptions.length}>
              <option value="">{activityOptions.length ? 'Select a planting activity' : 'Unscheduled planting'}</option>
              {activityOptions.map((activity) => <option key={activity.id} value={activity.id}>
                {activity.title} · {activity.date}
              </option>)}
            </select>
            <FieldError feedback={feedback} field="activity" />
            <span className="text-sm" style={{ color: 'var(--text-muted)' }}>
              {activityOptions.length ? 'These points belong to the selected activity.' : 'No confirmed planting activity is available for this site.'}
            </span>
          </div>

          <div className="form-group" style={{ marginTop: 10 }}>
            <label className="form-label" htmlFor="organization-point-count">Number of points to assign</label>
            <input {...feedback.props('point_count')} id="organization-point-count" className="form-input" type="number" required min="1" max={availablePoints.length} step="1" value={requestedCount || ''} onChange={(event) => setAssignmentCount(Number(event.target.value))} disabled={!assignProjectSiteId || assignBusy || assignmentLocked || loadingPoints || availablePoints.length === 0} />
              <FieldError feedback={feedback} field="point_count" />
            <span className="text-sm">
              {loadingPoints ? 'Checking available points…' : `${availablePoints.length} available points in this project site.`}
            </span>
            {assignProjectSiteId && !loadingPoints && availablePoints.length === 0 && (
              <span className="text-sm" role="status">No points are available to assign. Choose another project site with available points.</span>
            )}
          </div>

          {selectedAssignmentPlanter && (
            <div className="assign-point-info" role="status">
              <strong>{pointIdsToAssign.length} points{!selectedAssignmentPlanter.registration_pending && ` / ${selectedAssignmentPlanter.participant_count} participants`}</strong>
              {selectedSpeciesSummary && <span className="text-sm">{selectedSpeciesSummary}</span>}
              {selectedAssignmentPlanter.registration_pending ? (
                <span className="text-sm">These points will appear when this organization registers. Its participant count at signup determines each participant’s share.</span>
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
          {assignError && <div className="assign-message assign-error">{assignError}</div>}
          {assignSuccess && <div className="assign-message assign-success">{assignSuccess}</div>}
        </div>
      </PanelCard>

      {selectedAssignmentPlanter && !selectedAssignmentPlanter.registration_pending && (
        <PanelCard title="Participant devices" defaultOpen={false}>
          <ParticipantDevices key={selectedAssignmentPlanter.id} planter={selectedAssignmentPlanter} />
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
        <p>Choose another project site with available points.</p>
      </Modal>

      <PlanterActivityReport
        open={reportOpen}
        onClose={() => setReportOpen(false)}
      />

    </Panel>
  );
}
