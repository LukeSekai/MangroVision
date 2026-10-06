import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate } from 'react-router-dom';
import Modal from '../components/Modal';
import OrganizationHistory from '../components/OrganizationHistory';
import AutomaticGrowth from '../components/AutomaticGrowth';
import RestorationReportDialog from '../components/RestorationReportDialog';
import { growthStageLabel } from '../utils/mangroveGrowth';
import { monitoringCounts, visitFormFromLatest } from '../utils/organizationMonitoring';
import { deathLocationCounts } from '../utils/monitoringLocations';
import { PanelCard } from '../components/Panel';
import { useAuthStore } from '../stores/authStore';
import './OrganizationMonitoring.css';

const API = import.meta.env.VITE_API_BASE || '';
const MANILA_TIMEZONE = 'Asia/Manila';
const SeedlingLocations = lazy(() => import('../components/SeedlingLocations'));

const HEALTH_OPTIONS = [
  { value: 'excellent', label: 'Excellent' },
  { value: 'good', label: 'Good' },
  { value: 'fair', label: 'Fair' },
  { value: 'poor', label: 'Poor' },
  { value: 'critical', label: 'Critical' },
];

function manilaDateInputValue() {
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: MANILA_TIMEZONE,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(new Date());
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return `${values.year}-${values.month}-${values.day}`;
}

function formatDate(value) {
  if (!value) return 'No monitoring visit yet';
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return String(value);
  return date.toLocaleDateString(undefined, {
    timeZone: MANILA_TIMEZONE,
    month: 'short',
    day: 'numeric',
    year: 'numeric',
  });
}


function SummaryIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M3 3v18h18" /><path d="m7 16 4-5 3 3 5-7" />
    </svg>
  );
}

function VisitIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M12 20h9" /><path d="M16.5 3.5a2.12 2.12 0 0 1 3 3L7 19l-4 1 1-4Z" />
    </svg>
  );
}

function HistoryIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2">
      <path d="M3 12a9 9 0 1 0 3-6.7L3 8" /><path d="M3 3v5h5" /><path d="M12 7v5l3 2" />
    </svg>
  );
}

export default function OrganizationMonitoring() {
  const navigate = useNavigate();
  const token = useAuthStore((state) => state.token);
  const loadedWorkspace = useRef(null);
  const loadedVisit = useRef(null);
  const [visitRefreshKey, setVisitRefreshKey] = useState(0);
  const [organizations, setOrganizations] = useState([]);
  const [deathReasons, setDeathReasons] = useState([]);
  const [historyOrganization, setHistoryOrganization] = useState(null);
  const [selectedOrganizationId, setSelectedOrganizationId] = useState('');
  const [loading, setLoading] = useState(true);
  const [loadError, setLoadError] = useState('');
  const [saving, setSaving] = useState(false);
  const [formError, setFormError] = useState('');
  const [notice, setNotice] = useState('');
  const [fieldSheetOpen, setFieldSheetOpen] = useState(false);
  const [reportOpen, setReportOpen] = useState(false);
  const [fieldOrganization, setFieldOrganization] = useState('');
  const [selectedDeaths, setSelectedDeaths] = useState([]);
  const [visitContext, setVisitContext] = useState(null);
  const [visitLoading, setVisitLoading] = useState(false);
  const [visitError, setVisitError] = useState('');
  const prefilledOrganization = useRef(null);
  const [form, setForm] = useState({
    monitored_at: manilaDateInputValue(),
    dead_count: '0',
    death_reason_category: '',
    death_reason_notes: '',
    health_status: '',
    actions_taken: '',
  });

  const loadWorkspace = useCallback(async ({ quiet = false, force = false } = {}) => {
    if (force) setVisitRefreshKey((value) => value + 1);
    if (!token) {
      setLoadError('Sign in again to load organization monitoring.');
      setLoading(false);
      return;
    }
    if (!quiet) setLoading(true);
    try {
      const organizationResponse = await fetch(`${API}/api/monitoring/organizations`, {
        cache: force ? 'reload' : 'default',
      });
      const organizationPayload = await organizationResponse.json().catch(() => ({}));
      if (!organizationResponse.ok) {
        throw new Error(organizationPayload.detail || 'Could not load organizations.');
      }
      const nextOrganizations = Array.isArray(organizationPayload.organizations)
        ? organizationPayload.organizations
        : [];
      setOrganizations(nextOrganizations);
      setDeathReasons(organizationPayload.death_reason_options || []);
      setSelectedOrganizationId((current) => (
        nextOrganizations.some((organization) => String(organization.id) === String(current))
          ? current
          : ''
      ));
      setLoadError('');
    } catch (error) {
      setLoadError(error.message || 'Could not load organization monitoring.');
    } finally {
      loadedWorkspace.current = token;
      setLoading(false);
    }
  }, [token]);

  useEffect(() => {
    if (loadedWorkspace.current === token) return undefined;
    const timer = window.setTimeout(() => loadWorkspace(), 0);
    return () => window.clearTimeout(timer);
  }, [loadWorkspace, token]);

  useEffect(() => {
    const refresh = () => { void loadWorkspace({ quiet: true }); };
    window.addEventListener('mv:data-changed', refresh);
    return () => window.removeEventListener('mv:data-changed', refresh);
  }, [loadWorkspace]);

  const selectedOrganization = useMemo(
    () => organizations.find(
      (organization) => String(organization.id) === String(selectedOrganizationId),
    ) || null,
    [organizations, selectedOrganizationId],
  );

  const contextReady = visitContext?.organization_id === Number(selectedOrganizationId)
    && visitContext?.monitored_date === form.monitored_at && !visitLoading && !visitError;
  const locationCounts = deathLocationCounts(selectedDeaths, form.dead_count);
  const counts = monitoringCounts(visitContext?.total_planted, locationCounts.valid ? String(locationCounts.reported) : '', visitContext?.previous_dead_count);

  useEffect(() => {
    if (!selectedOrganizationId || !form.monitored_at) {
      loadedVisit.current = null;
      return;
    }
    const key = `${selectedOrganizationId}:${form.monitored_at}:${visitRefreshKey}`;
    if (loadedVisit.current === key) return;
    const controller = new AbortController();
    async function loadVisit() {
      setVisitLoading(true);
      setVisitError('');
      try {
        const query = new URLSearchParams({ monitored_at: form.monitored_at });
        const response = await fetch(`${API}/api/monitoring/organizations/${selectedOrganizationId}/visit-context?${query}`, { signal: controller.signal });
        const payload = await response.json();
        if (!response.ok) throw new Error(payload.detail || 'Could not load the latest monitoring visit.');
        if (controller.signal.aborted) return;
        setVisitContext(payload);
        if (prefilledOrganization.current !== selectedOrganizationId) {
          setForm(visitFormFromLatest(payload.latest_record, form.monitored_at));
          prefilledOrganization.current = selectedOrganizationId;
        }
      } catch (error) {
        if (!controller.signal.aborted) setVisitError(error.message);
      } finally {
        if (!controller.signal.aborted) { loadedVisit.current = key; setVisitLoading(false); }
      }
    }
    void loadVisit();
    return () => controller.abort();
  }, [selectedOrganizationId, form.monitored_at, visitRefreshKey]);

  const updateForm = (updates) => {
    if (updates.monitored_at !== undefined) setSelectedDeaths([]);
    if (updates.dead_count !== undefined && Number(updates.dead_count) < selectedDeaths.length) {
      setFormError(`Deselect locations first: ${selectedDeaths.length} seedlings are currently selected.`);
      return;
    }
    setForm((current) => ({ ...current, ...updates }));
    setFormError('');
    setNotice('');
  };

  const submitRecord = async (event) => {
    event.preventDefault();
    if (saving || !contextReady) return;
    setFormError('');
    setNotice('');
    if (!selectedOrganizationId) {
      setFormError('Select an organization.');
      return;
    }
    if (!locationCounts.valid) {
      setFormError('Enter a whole-number death count at least as large as the number of selected locations.');
      return;
    }
    if (counts.error) {
      setFormError(counts.error);
      return;
    }
    if (counts.newlyDead > 0 && !form.death_reason_category) {
      setFormError('Select the cause of death for the newly dead seedlings.');
      return;
    }
    if (!form.health_status) {
      setFormError('Select the organization’s overall plant health.');
      return;
    }
    if (!form.actions_taken.trim()) {
      setFormError('Describe what the LGU did during monitoring.');
      return;
    }

    setSaving(true);
    try {
      const response = await fetch(
        `${API}/api/monitoring/organization-records`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            organization_id: Number(selectedOrganizationId),
            monitored_at: form.monitored_at,
            new_dead_count: counts.newlyDead,
            dead_planting_event_ids: selectedDeaths,
            unlocated_dead_count: locationCounts.unlocated,
            death_reason_category: counts.newlyDead > 0 ? form.death_reason_category : null,
            death_reason_notes: counts.newlyDead > 0 ? form.death_reason_notes.trim() : '',
            baseline_record_id: visitContext.baseline_record_id,
            expected_alive_count: visitContext.alive_before_count,
            health_status: form.health_status,
            actions_taken: form.actions_taken.trim(),
          }),
        },
      );
      const payload = await response.json().catch(() => ({}));
      if (!response.ok) throw new Error(payload.detail || 'Could not save monitoring record.');
      setNotice(`${payload.organization_name} monitoring was recorded as one organization visit.`);
      setSelectedOrganizationId('');
      setSelectedDeaths([]);
      setForm((current) => ({
        ...current,
        dead_count: '0',
        death_reason_category: '',
        death_reason_notes: '',
        health_status: '',
        actions_taken: '',
      }));
      await loadWorkspace({ quiet: true });
    } catch (error) {
      setFormError(error.message || 'Could not save monitoring record.');
    } finally {
      setSaving(false);
    }
  };

  const openVisitModal = (organizationId) => {
    setSelectedDeaths([]);
    setForm(visitFormFromLatest(null, manilaDateInputValue()));
    prefilledOrganization.current = null;
    setVisitContext(null);
    setVisitLoading(true);
    setVisitError('');
    setSelectedOrganizationId(String(organizationId));
    setFormError('');
    setNotice('');
  };

  const closeVisitModal = () => {
    if (saving) return;
    setSelectedOrganizationId('');
    setFormError('');
  };

  return (
    <div className="org-monitoring-page org-monitoring-panel">
      <div className="org-monitoring-content">
      <header className="org-monitoring-page-header">
        <div>
          <span className="org-monitoring-eyebrow">Mangrove care</span>
          <h1>Monitoring</h1>
          <p>Track plant health, record visits, and review each organization’s progress.</p>
        </div>
        <div className="org-monitoring-page-actions">
          <button type="button" aria-haspopup="dialog" onClick={() => setReportOpen(true)}>Download Monitoring Report</button>
          <button type="button" onClick={() => setFieldSheetOpen(true)}>Prepare field sheet</button>
          <button type="button" onClick={() => loadWorkspace({ force: true })} disabled={loading}>Refresh</button>
          <button type="button" className="is-primary" onClick={() => navigate('/monitoring/map')}>
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" aria-hidden="true"><path d="m3 5 6-2 6 2 6-2v16l-6 2-6-2-6 2V5Zm6-2v16m6-14v16" /></svg>
            Show map
          </button>
        </div>
      </header>

      <div className="org-monitoring-kpis" aria-label="Monitoring overview">
        <article><span>Organizations</span><strong>{loading ? '—' : organizations.length}</strong><small>Participating in monitoring</small></article>
        <article><span>Seedlings planted</span><strong>{loading ? '—' : organizations.reduce((sum, item) => sum + Number(item.total_planted ?? item.current_planted_points ?? 0), 0).toLocaleString()}</strong><small>Includes recorded deaths and replacement plantings</small></article>
        <article><span>Recorded visits</span><strong>{loading ? '—' : organizations.reduce((sum, item) => sum + Number(item.monitoring_record_count || 0), 0).toLocaleString()}</strong><small>Monitoring records to date</small></article>
        <article><span>Awaiting first visit</span><strong>{loading ? '—' : organizations.filter((item) => !item.latest_record).length}</strong><small>Organizations with no visit recorded</small></article>
      </div>

      {loadError && <div className="org-monitoring-message is-error">{loadError}</div>}
      {notice && <div className="org-monitoring-message is-success" role="status">{notice}</div>}

      <PanelCard
        panelKey="organization-summary"
        title="Record a monitoring visit"
        icon={<SummaryIcon />}
        className="org-monitoring-summary-card"
      >
        {loading ? <div className="org-monitoring-muted">Loading organizations...</div> : null}
        {!loading && organizations.length === 0 ? (
          <div className="org-monitoring-muted">
            No organizations are registered yet. Create a planting schedule or organization first.
          </div>
        ) : null}
        {organizations.length > 0 ? (
          <div className="org-monitoring-organization-list">
            {organizations.map((organization) => {
              const organizationLatest = organization.latest_record || null;
              const hasPlantedSeedlings = Number(organization.total_planted ?? organization.current_planted_points ?? 0) > 0;
              const monitoringAvailable = organization.monitoring_available === true;
              return (
                <button
                  key={organization.id}
                  type="button"
                  className={`org-monitoring-organization-card${monitoringAvailable ? '' : ' is-not-due'}`}
                  onClick={() => openVisitModal(organization.id)}
                  disabled={!monitoringAvailable}
                >
                  <span className="org-monitoring-organization-head">
                    <strong>{organization.name}</strong>
                    <span>{monitoringAvailable ? 'Record visit →' : 'Locked'}</span>
                  </span>
                  {monitoringAvailable ? (
                    <span className="org-monitoring-organization-metrics">
                      <span><strong>{organization.total_planted ?? organization.current_planted_points ?? 0}</strong> planted</span>
                      <span><strong>{organization.monitoring_record_count || 0}</strong> visits</span>
                      <span><strong>{organization.alive_seedlings ?? 0}</strong> alive now</span>
                      <span>{organizationLatest?.growth_snapshot?.label || growthStageLabel(organizationLatest?.growth_stage)}</span>
                      <span><strong>{organizationLatest ? `${organizationLatest.survival_rate_pct ?? 0}%` : '—'}</strong> alive</span>
                    </span>
                  ) : (
                    <span className="org-monitoring-waiting-message">
                      <strong>{hasPlantedSeedlings ? 'Not time to monitor yet' : 'No seedlings to monitor'}</strong>
                      <small>{hasPlantedSeedlings && organization.next_monitoring_date
                        ? `Available on ${formatDate(organization.next_monitoring_date)}`
                        : 'Monitoring starts two weeks after planting.'}</small>
                    </span>
                  )}
                </button>
              );
            })}
          </div>
        ) : null}
      </PanelCard>

      <Modal
        open={Boolean(selectedOrganization)}
        title="Record monitoring visit"
        icon={<VisitIcon />}
        variant="info"
        cancelLabel="Cancel"
        busy={saving}
        onCancel={closeVisitModal}
        className="modal-card-wide org-monitoring-visit-modal"
      >
        <form className="org-monitoring-form" onSubmit={submitRecord}>
          {visitLoading ? <p role="status">Loading the latest saved visit...</p> : null}
          {visitError ? <p className="org-monitoring-message is-error" role="alert">{visitError} Close and reopen this organization to retry.</p> : null}
          <div className="org-monitoring-form-organization">
            <strong>{selectedOrganization?.name}</strong>
            <span>
              {selectedOrganization?.current_planted_points || 0} mapped planted points · {' '}
              {selectedOrganization?.monitoring_record_count || 0} previous visits
            </span>
          </div>

          <div className="org-monitoring-field org-monitoring-date-field">
            <label className="org-monitoring-label" htmlFor="monitoring-date">Monitoring date</label>
            <input
              id="monitoring-date"
              className="org-monitoring-input"
              type="date"
              max={manilaDateInputValue()}
              value={form.monitored_at}
              onChange={(event) => updateForm({ monitored_at: event.target.value })}
              disabled={saving}
              required
            />
          </div>

          <div className="org-monitoring-count-grid">
            <label>
              <span>Alive seedlings before this visit</span>
              <input className="org-monitoring-input" type="number" value={contextReady ? visitContext.alive_before_count : ''} readOnly />
              <small>Previous deaths are already deducted. New plantings are included.</small>
            </label>
            <label>
              <span>Newly dead seedlings</span>
              <input className="org-monitoring-input is-dead" type="number" min="0"
                max={visitContext?.alive_before_count ?? 0} step="1" value={form.dead_count}
                onChange={(event) => updateForm({ dead_count: event.target.value })}
                placeholder="Enter 0 if none" disabled={saving || !contextReady} required />
              <small>Enter the total new deaths since the last visit. Select up to this many seedlings below to identify their locations.</small>
            </label>
          </div>
          {Number(form.dead_count) > 0 ? <div className="org-monitoring-count-grid">
            <label><span>Cause of death</span>
              <select className="org-monitoring-input" value={form.death_reason_category}
                onChange={(event) => updateForm({ death_reason_category: event.target.value })} disabled={saving || !contextReady} required>
                <option value="">Select a cause…</option>
                {deathReasons.map((reason) => <option key={reason.value} value={reason.value}>{reason.label}</option>)}
              </select>
              <small>Applies to the new deaths reported in this visit, including those still to locate.</small>
            </label>
            <label><span>Cause notes (optional)</span>
              <textarea className="org-monitoring-input" rows="2" maxLength="500" value={form.death_reason_notes}
                onChange={(event) => updateForm({ death_reason_notes: event.target.value })} disabled={saving || !contextReady}
                placeholder="Describe what you observed." />
            </label>
          </div> : null}
          {selectedOrganization && contextReady ? <fieldset disabled={saving} className="seedling-location-fieldset">
            <legend>Locate dead seedlings</legend>
            <p aria-live="polite">{locationCounts.valid ? `${locationCounts.reported} reported dead · ${locationCounts.located} located · ${locationCounts.unlocated} still to locate` : 'Enter a whole number for total new deaths.'}</p>
            <Suspense fallback={<p>Loading locations…</p>}><SeedlingLocations key={`${selectedOrganization.id}-${form.monitored_at}`} organizationId={selectedOrganization.id} monitoredAt={form.monitored_at}
              selected={selectedDeaths} onChange={saving ? undefined : setSelectedDeaths} disabled={saving} maxSelected={locationCounts.valid ? locationCounts.reported : 0} /></Suspense>
          </fieldset> : null}
          <div className="org-monitoring-calculated" aria-live="polite">
            <div><span>Alive after this visit</span><strong>{contextReady ? counts.alive ?? '—' : '—'}</strong></div>
            <div><span>Survival rate</span><strong>{!contextReady || counts.survival === null ? '—' : `${counts.survival.toFixed(2)}%`}</strong></div>
            <p>{contextReady ? `${visitContext.total_planted} originally planted · ${visitContext.previous_dead_count} previously dead · ${counts.dead ?? '—'} total dead after this visit` : 'Loading saved seedling counts...'}</p>
          </div>
          {contextReady && counts.error
            ? <p className="org-monitoring-message is-error" role="status">{counts.error}</p> : null}

          <div className="org-monitoring-field org-monitoring-growth-field">
            <span className="org-monitoring-label">Automatic growth estimate</span>
            <AutomaticGrowth snapshot={contextReady ? visitContext.growth_snapshot : null} alive={counts.alive} />
            {contextReady && visitContext.latest_record ? <p className="org-monitoring-growth-help">
              Last saved visit: {formatDate(visitContext.latest_record.monitored_at)}. Next 2-week check: {formatDate(visitContext.next_monitoring_date)}.
              {' '}Plant health and LGU actions below are carried forward; update them for this visit.
            </p> : null}
            {contextReady && visitContext.count_conflict ? <p className="org-monitoring-message is-error">An earlier visit recorded more deaths than the latest visit. Earlier deaths have been kept in the total; review the history before saving.</p> : null}
          </div>

          <div className="org-monitoring-field">
            <label className="org-monitoring-label" htmlFor="overall-health">Overall plant health</label>
            <select
              id="overall-health"
              className="org-monitoring-input"
              value={form.health_status}
              onChange={(event) => updateForm({ health_status: event.target.value })}
              disabled={saving || !contextReady}
              required
            >
              <option value="">Select overall health...</option>
              {HEALTH_OPTIONS.map((option) => (
                <option key={option.value} value={option.value}>{option.label}</option>
              ))}
            </select>
          </div>

          <div className="org-monitoring-field org-monitoring-actions-field">
            <label className="org-monitoring-label" htmlFor="actions-taken">
              What did the LGU do during monitoring?
            </label>
            <textarea
              id="actions-taken"
              className="org-monitoring-input org-monitoring-textarea"
              rows="4"
              maxLength="2000"
              value={form.actions_taken}
              onChange={(event) => updateForm({ actions_taken: event.target.value })}
              placeholder="Example: Removed debris, replaced guards, and cleared blocked water flow."
              disabled={saving || !contextReady}
              required
            />
          </div>

          {formError && <div className="org-monitoring-message is-error">{formError}</div>}
          {notice && <div className="org-monitoring-message is-success">{notice}</div>}
          <button
            className="org-monitoring-submit"
            type="submit"
            disabled={saving || loading || !contextReady || !visitContext?.total_planted}
          >
            {saving ? 'Saving monitoring visit...' : 'Save monitoring visit'}
          </button>
        </form>
      </Modal>

      <PanelCard
        panelKey="organization-history"
        title="Monitoring History"
        icon={<HistoryIcon />}
        badge={organizations.length}
        className="org-monitoring-history-card"
      >
        {loading ? <p className="org-monitoring-muted">Loading organizations...</p> : null}
        {!loading && !organizations.length ? <p className="org-monitoring-muted">No organizations to display yet.</p> : null}
        <div className="org-monitoring-organization-list">
          {organizations.map((organization) => <button key={organization.id} type="button"
            className="org-monitoring-organization-card" onClick={() => setHistoryOrganization(organization)} aria-haspopup="dialog">
            <span className="org-monitoring-organization-head"><strong>{organization.name}</strong><span>View history →</span></span>
            <span className="org-monitoring-organization-metrics">
              <span><strong>{organization.monitoring_record_count || 0}</strong> visits</span>
              <span>{organization.latest_record ? `Last visit: ${formatDate(organization.latest_record.monitored_at)}` : 'No visits yet'}</span>
            </span>
          </button>)}
        </div>
      </PanelCard>
      </div>
      {historyOrganization ? <OrganizationHistory key={historyOrganization.id} organization={historyOrganization} onChanged={() => loadWorkspace({ quiet: true })} onClose={() => setHistoryOrganization(null)} /> : null}
      <Modal open={fieldSheetOpen} title="Prepare field sheet" className="modal-card-wide" variant="info" confirmLabel="Close" cancelLabel={null} onConfirm={() => setFieldSheetOpen(false)} onCancel={() => setFieldSheetOpen(false)}>
        <label>Organization<select className="org-monitoring-input" value={fieldOrganization} onChange={(e) => setFieldOrganization(e.target.value)}><option value="">Select an organization</option>{organizations.map((o) => <option key={o.id} value={o.id}>{o.name}</option>)}</select></label>
        {fieldSheetOpen && fieldOrganization ? <Suspense fallback={<p>Loading map…</p>}><SeedlingLocations key={fieldOrganization} organizationId={fieldOrganization} readOnly /></Suspense> : null}
      </Modal>
      {reportOpen && <RestorationReportDialog initialSelection={{ type: 'monitoring' }} onClose={() => setReportOpen(false)} />}
    </div>
  );
}
