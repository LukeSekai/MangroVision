import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { availableOrganizationPoints } from '../utils/organizationAssignment';
import { useMapStore } from '../stores/mapStore';
import { useProcessingStore } from '../stores/processingStore';
import { useAuthStore } from '../stores/authStore';
import { Panel, PanelCard } from '../components/Panel';
import Modal from '../components/Modal';
import PlanterActivityReport from './PlanterActivityReport';
import { getPlanterColor } from '../utils/planterColors';
import './PlanterManagement.css';

const API = import.meta.env.VITE_API_BASE || '';

const STATUS_LABEL = {
  planned: 'Planned',
  assigned: 'Assigned',
  pending: 'Pending',
  planted: 'Planted',
  completed: 'Completed',
  skipped: 'Skipped',
  deleted: 'Deleted',
  eroded_unavailable: 'Not Available for Planting',
};

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
  const adminToken = useAuthStore((s) => s.token);

  const [planters, setPlanters] = useState([]);
  const [organizations, setOrganizations] = useState([]);
  const [assignments, setAssignments] = useState([]);
  const [dashStats, setDashStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [resetSlot, setResetSlot] = useState(1);
  const [deviceMessage, setDeviceMessage] = useState('');

  // One unified assign form: one clicked point and many clicked points share
  // the same selected-id list and batch endpoint.
  const [assignOrganizationId, setAssignOrganizationId] = useState(null);
  const [assignmentCount, setAssignmentCount] = useState(null);
  const [assignProjectSiteChoice, setAssignProjectSiteId] = useState(null);
  const [assignBusy, setAssignBusy] = useState(false);
  const [assignError, setAssignError] = useState('');
  const [assignSuccess, setAssignSuccess] = useState('');
  const [noAvailableSite, setNoAvailableSite] = useState(null);
  const activeAssignmentsHeaderRef = useRef(null);

  const [reportOpen, setReportOpen] = useState(false);

  const [shareLink, setShareLink] = useState(null);
  const [shareBusy, setShareBusy] = useState(false);
  const [shareError, setShareError] = useState('');
  const [shareCopied, setShareCopied] = useState('');
  const shareCopiedTimerRef = useRef(null);

  // Archive assignment confirmation modal state
  const [archiveTarget, setArchiveTarget] = useState(null);
  const [archiveBusy, setArchiveBusy] = useState(false);

  // Per-assignment point-status drill down
  const [assignmentPointsCache, setAssignmentPointsCache] = useState({});
  const assignmentPointsRef = useRef({});
  const loadSequence = useRef(0);
  const dataLoaded = useRef(false);
  const shareLoaded = useRef(null);
  useEffect(() => { assignmentPointsRef.current = assignmentPointsCache; }, [assignmentPointsCache]);
  const [pointStatusBusyId, setPointStatusBusyId] = useState(null);

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
      const [pRes, dRes, aRes, oRes] = await Promise.all([
        fetch(`${API}/api/planters/?include_inactive=true`),
        fetch(`${API}/api/planters/dashboard`),
        fetch(`${API}/api/assignments/?active_only=true`),
        fetch(`${API}/api/planter-auth/organizations`),
      ]);
      if (![pRes, dRes, aRes, oRes].every((response) => response.ok)) throw new Error('Could not load planter data.');
      const [nextPlanters, nextStats, nextAssignments, nextOrganizations] = await Promise.all([pRes.json(), dRes.json(), aRes.json(), oRes.json()]);
      if (sequence !== loadSequence.current) return;
      setPlanters(nextPlanters);
      setOrganizations(nextOrganizations.organizations);
      setDashStats(nextStats);
      setAssignments(nextAssignments);
      const openIds = Object.keys(assignmentPointsRef.current).filter((id) => assignmentPointsRef.current[id]);
      await Promise.all(openIds.map(async (id) => {
        const response = await fetch(`${API}/api/assignments/${id}/points`);
        if (!response.ok) return;
        const rows = await response.json();
        if (sequence === loadSequence.current) setAssignmentPointsCache((current) => (
          current[id] ? { ...current, [id]: rows } : current
        ));
      }));
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
  useEffect(() => {
    setAssignmentScope(selectedAssignmentPlanter?.organization_id ?? null, assignProjectSiteId);
  }, [assignProjectSiteId, selectedAssignmentPlanter?.organization_id, setAssignmentScope]);
  const assignmentLocked = processing;
  const shareFieldUrl = shareLink?.field_url || '';
  const shareCloudflareActive = Boolean(shareLink?.active && shareLink?.field_url);

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
  const selectedSpeciesKeys = Array.from(new Set(
    points
      .filter((point) => selectedPointIdSet.has(Number(point.id)))
      .map((point) => normalizeAssignmentSpecies(point.species))
      .filter(Boolean),
  ));
  const autoAssignSpecies = selectedSpeciesKeys.length === 1
    ? ASSIGNMENT_SPECIES_LABELS[selectedSpeciesKeys[0]]
    : '';

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

  const showActiveAssignments = () => {
    setNoAvailableSite(null);
    window.requestAnimationFrame(() => {
      const header = activeAssignmentsHeaderRef.current;
      if (header?.getAttribute('aria-expanded') === 'false') header.click();
      header?.focus();
      header?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    });
  };

  const handleAssign = async () => {
    setAssignError('');
    setAssignSuccess('');
    if (assignmentLocked) {
      setAssignError('Planting point assignment is locked while image processing is running.');
      return;
    }
    if (!assignOrganizationId) {
      setAssignError('Pick an organization.');
      return;
    }
    if (!assignProjectSiteId) {
      setAssignError("Select a project site owned by the organization.");
      return;
    }
    if (!pointIdsToAssign.length) {
      setAssignError('Enter a point count within the available points for this site.');
      return;
    }
    setAssignBusy(true);
    try {
      const res = await fetch(`${API}/api/planters/organizations/${assignOrganizationId}/assignments`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          planting_point_ids: pointIdsToAssign,
          species: autoAssignSpecies,
          site_zone_id: assignProjectSiteId,
        }),
      });
      const payload = await res.json().catch(() => ({}));
      if (!res.ok) {
        throw new Error(payload.detail || 'Assignment failed.');
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
      setAssignError(err.message);
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
          showShareCopied('Cloudflare link copied.');
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

  const handleArchive = async (id) => {
    setArchiveBusy(true);
    try {
      await fetch(`${API}/api/assignments/${id}/archive`, { method: 'POST' });
      setArchiveTarget(null);
      loadData();
      fetchPoints();
      fetchZones();
    } catch (err) {
      console.error('Archive failed:', err);
    } finally {
      setArchiveBusy(false);
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
    <Panel title="Planter Management" subtitle={`${activePlanters.length} active organizations`}>
      <button type="button" className="btn btn-secondary btn-sm" disabled={loading}
        onClick={() => {
          window.dispatchEvent(new Event('mv:invalidate-reads'));
          void loadData();
          void loadShareStatus({ silent: true });
          void fetchPoints();
          void fetchZones();
        }}>{loading ? 'Refreshing…' : 'Refresh data'}</button>
      {dashStats && (
        <PanelCard
          title="Dashboard"
          icon={
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M18 20V10"/><path d="M12 20V4"/><path d="M6 20v-6"/></svg>
          }
        >
          <div className="planter-stats-grid">
            <div className="stat-card">
              <div className="stat-label">Organizations</div>
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
            <div className="stat-card">
              <div className="stat-label">Skipped</div>
              <div className="stat-value" style={{ color: '#6b7280' }}>{dashStats.skipped_assigned_points || 0}</div>
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
              className={`share-status-dot ${shareCloudflareActive ? 'share-status-dot-active' : ''}`}
              aria-hidden="true"
            />
            <span>{shareCloudflareActive ? 'Cloudflare active' : 'Cloudflare inactive'}</span>
          </div>

          <div className="share-link-box">
            <input
              className="share-link-input"
              value={shareFieldUrl}
              placeholder="Generate a Cloudflare link"
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
              onClick={handleGenerateCloudflareLink}
              disabled={shareBusy}
            >
              {shareBusy ? 'Working...' : 'Generate Cloudflare Link'}
            </button>
            <button
              type="button"
              className="btn btn-secondary btn-sm"
              onClick={() => loadShareStatus()}
              disabled={shareBusy}
            >
              Check Status
            </button>
            {shareCloudflareActive && (
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

          {shareError && <div className="assign-message assign-error">{shareError}</div>}
          {shareCopied && <div className="assign-message assign-success">{shareCopied}</div>}
        </div>
      </PanelCard>

      <PanelCard
        title="Quick Assign"
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
        <div className="assign-card">
          <p className="text-sm" style={{ color: 'var(--text-muted)', lineHeight: 1.5 }}>
            Select an organization and choose how many points to assign. Each participant receives staggered point locations forming zigzag strips, divided equally. Short strips at the site boundary continue into the next strip.
          </p>

          <div className="form-group" style={{ marginTop: 10 }}>
            <label className="form-label">Assign to organization</label>
            <select
              className="form-select"
              value={assignOrganizationId || ''}
              onChange={(e) => {
                const nextPlanterId = Number(e.target.value) || null;
                const nextPlanter = assignmentOrganizations.find((organization) => Number(organization.organization_id) === nextPlanterId);
                setAssignOrganizationId(nextPlanterId);
                setAssignmentCount(null);
                const site = (projectSites?.features || []).find((feature) => Number(feature.properties?.organization_id) === Number(nextPlanter?.organization_id));
                const siteId = site?.id ?? site?.properties?.id ?? null;
                setAssignProjectSiteId(siteId);
                warnIfNoAvailablePoints(nextPlanter, site);
                setResetSlot(1);
                setDeviceMessage('');
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
            <label className="form-label">Project site</label>
            <select
              className="form-select"
              value={assignProjectSiteId || ''}
              onChange={(e) => {
                const nextSiteId = Number(e.target.value) || null;
                setAssignProjectSiteId(nextSiteId);
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
            <span className="text-sm" style={{ color: 'var(--text-muted)', lineHeight: 1.4 }}>
              The map automatically locates this organization's project site.
            </span>
          </div>

          <div className="form-group" style={{ marginTop: 10 }}>
            <label className="form-label" htmlFor="organization-point-count">Number of points to assign</label>
            <input id="organization-point-count" className="form-input" type="number" min="1" max={availablePoints.length} step="1" value={requestedCount || ''} onChange={(event) => setAssignmentCount(Number(event.target.value))} disabled={!assignProjectSiteId || assignBusy || assignmentLocked || loadingPoints || availablePoints.length === 0} />
            <span className="text-sm">
              {loadingPoints ? 'Checking available points…' : `${availablePoints.length} available points in this project site.`}
            </span>
            {assignProjectSiteId && !loadingPoints && availablePoints.length === 0 && (
              <span className="text-sm" role="status">No points are available to assign. Check Active Assignments to review existing allocations.</span>
            )}
            {!loadingPoints && availablePoints.length > 0 && !validCount && requestedCount > 0 && <span className="assign-error">Choose a whole number up to {availablePoints.length}.</span>}
          </div>

          {selectedAssignmentPlanter && (
            <div className="assign-point-info" role="status">
              <strong>{pointIdsToAssign.length} points{!selectedAssignmentPlanter.registration_pending && ` / ${selectedAssignmentPlanter.participant_count} participants`}</strong>
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
              || pointIdsToAssign.length === 0
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
        <PanelCard title="Participant device recovery" defaultOpen={false}>
          <p className="text-sm">If a participant changes phones, reset their slot, then sign in on the replacement phone. Their assigned points stay the same.</p>
          <label className="form-label" htmlFor="participant-reset-slot">Participant number</label>
          <input id="participant-reset-slot" className="form-input" type="number" min="1" max={selectedAssignmentPlanter.participant_count} value={resetSlot} onChange={(event) => setResetSlot(Number(event.target.value))} />
          <button className="btn btn-secondary btn-sm" disabled={assignBusy || resetSlot < 1 || resetSlot > selectedAssignmentPlanter.participant_count || !Number.isInteger(resetSlot)} onClick={async () => {
            setAssignBusy(true);
            try {
              const response = await fetch(`${API}/api/planters/${selectedAssignmentPlanter.id}/participants/${resetSlot}/reset-device`, { method: 'POST' });
              const result = await response.json();
              if (!response.ok) throw new Error(result.detail || 'Could not reset device.');
              setDeviceMessage(`Participant ${resetSlot} can now sign in on a replacement device.`);
            } catch (error) { setDeviceMessage(error.message); }
            finally { setAssignBusy(false); }
          }}>Reset device slot</button>
          {deviceMessage && <p role="status" className="text-sm">{deviceMessage}</p>}
        </PanelCard>
      )}

      <PanelCard
        title="Active Assignments"
        headerRef={activeAssignmentsHeaderRef}
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
                        {a.species ? ` · ${a.species}` : ''}
                        {(a.project_site_name || a.site_name) ? ` · ${a.project_site_name || a.site_name}` : ''}
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
                          const lifecycleStatus = row.assignment_status || row.status;
                          const isPlanted = lifecycleStatus === 'planted' || lifecycleStatus === 'completed';
                          const isSkipped = lifecycleStatus === 'skipped';
                          const isUnavailable = Boolean(row.eroded_unavailable) && !isPlanted && !isSkipped;
                          const rowStatus = isUnavailable ? 'eroded_unavailable' : lifecycleStatus;
                          const hasWarning = Boolean(row.survival_warning);
                          return (
                            <div key={row.assignment_point_id} className="assignment-point-row">
                              <div className="assignment-point-info">
                                <span className="assignment-point-title">Point #{row.point_num}</span>
                                <span className="assignment-point-meta">
                                  {STATUS_LABEL[rowStatus] || rowStatus}
                                  {row.released_at ? ' · Location released for replacement' : ''}
                                  {hasWarning ? ` - Warning: ${row.warning_severity || 'medium'}` : ''}
                                  {isSkipped && row.skip_reason ? ` - ${row.skip_reason}` : ''}
                                </span>
                              </div>
                              <div className="assignment-point-actions">
                                {!row.released_at && !isPlanted && !isUnavailable && (
                                  <button
                                    className="btn btn-ghost btn-sm"
                                    disabled={busy}
                                    onClick={() => updateAssignmentPointStatus(a.id, row, 'completed')}
                                  >
                                    Mark planted
                                  </button>
                                )}
                                {!row.released_at && !isSkipped && !isUnavailable && (
                                  <button
                                    className="btn btn-ghost btn-sm"
                                    disabled={busy}
                                    onClick={() => updateAssignmentPointStatus(a.id, row, 'skipped')}
                                  >
                                    Skip
                                  </button>
                                )}
                                {!row.released_at && (isPlanted || isSkipped) && (
                                  <button
                                    className="btn btn-ghost btn-sm"
                                    disabled={busy}
                                    onClick={() => updateAssignmentPointStatus(a.id, row, 'pending')}
                                  >
                                    Reset
                                  </button>
                                )}
                                {isUnavailable && (
                                  <span className="text-sm" style={{ color: '#c2410c' }}>
                                    Remove the eroded zone to return this point to Planned.
                                  </span>
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

      <Modal
        open={Boolean(noAvailableSite)}
        title="No available points"
        variant="warning"
        confirmLabel="View Active Assignments"
        cancelLabel="Close"
        onConfirm={showActiveAssignments}
        onCancel={() => setNoAvailableSite(null)}
      >
        <p>
          No points are available to assign in <strong>{noAvailableSite?.siteName}</strong> for <strong>{noAvailableSite?.organizationName}</strong>.
        </p>
        <p>Check Active Assignments to review points that have already been allocated, or choose another project site with available points.</p>
      </Modal>

      <PlanterActivityReport
        open={reportOpen}
        onClose={() => setReportOpen(false)}
      />

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
