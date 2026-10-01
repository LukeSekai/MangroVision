import { useEffect, useMemo, useState } from 'react';
import { Panel, PanelCard } from '../components/Panel';
import Modal from '../components/Modal';
import { useMapStore } from '../stores/mapStore';
import { useAuthStore } from '../stores/authStore';
import './Monitoring.css';

const API = import.meta.env.VITE_API_BASE || '';
const TIMEZONE = 'Asia/Manila';

const DEATH_REASONS = [
  { key: 'barnacles', label: 'Barnacles' },
  { key: 'waves', label: 'Waves' },
  { key: 'disease', label: 'Disease' },
  { key: 'animal_damage', label: 'Animal damage' },
  { key: 'storm', label: 'Storm' },
  { key: 'drying_out', label: 'Drying out' },
  { key: 'vandalism', label: 'Vandalism' },
  { key: 'other', label: 'Other' },
];

function getPointStatus(point) {
  if (point.death_at) return 'dead';
  if (point.assignment_status === 'skipped' || point.planting_status === 'skipped') return 'skipped';
  if (point.assignment_status === 'completed' || point.planting_status === 'planted') return 'planted';
  if (point.eroded_unavailable || point.inside_eroded_zone) return 'eroded_unavailable';
  if (point.assigned_planter_name) {
    return 'assigned';
  }
  return point.planting_status || 'planned';
}

function statusLabel(status) {
  const labels = {
    planned: 'Planned',
    assigned: 'Assigned',
    planted: 'Planted',
    completed: 'Planted',
    skipped: 'Skipped',
    alive: 'Alive',
    missing: 'Missing / not found',
    dead: 'Dead',
    eroded_unavailable: 'Not Available for Planting',
  };
  return labels[status] || String(status || 'Planned');
}

function formatDateTime(value) {
  if (!value) return '—';
  const d = new Date(value);
  if (Number.isNaN(d.getTime())) return String(value);
  return d.toLocaleString(undefined, {
    month: 'short', day: 'numeric', year: 'numeric',
    hour: 'numeric', minute: '2-digit',
    timeZone: TIMEZONE,
  });
}

function manilaDayNumber(value) {
  const date = new Date(value || '');
  if (Number.isNaN(date.getTime())) return null;
  const parts = new Intl.DateTimeFormat('en-CA', {
    timeZone: TIMEZONE,
    year: 'numeric',
    month: '2-digit',
    day: '2-digit',
  }).formatToParts(date);
  const values = Object.fromEntries(parts.map((part) => [part.type, part.value]));
  return Date.UTC(Number(values.year), Number(values.month) - 1, Number(values.day)) / 86_400_000;
}

function firstDefined(...values) {
  return values.find((value) => value !== undefined && value !== null);
}

function numericCount(...values) {
  const value = firstDefined(...values);
  const number = Number(value);
  return Number.isFinite(number) ? number : 0;
}

function arrayValue(...values) {
  return values.find(Array.isArray) || [];
}

function projectIdentity(value) {
  const id = firstDefined(
    value?.project_site_id,
    value?.site_id,
    value?.project_id,
    value?.id,
  );
  return id === undefined || id === null || id === '' ? 'unassigned' : String(id);
}

function normalizeProject(value = {}, index = 0) {
  const summary = value.summary || value.counts || {};
  const inspections = value.inspection_schedule || value.inspections || value.inspection_summary || summary.inspections || {};
  const verified = value.verified || value.verified_outcomes || summary.verified || {};
  const rawPlanters = arrayValue(value.planters, value.planter_names, summary.planters);
  const planterNames = rawPlanters
    .map((planter) => (typeof planter === 'string' ? planter : planter?.name || planter?.full_name))
    .filter(Boolean);
  const id = firstDefined(value.project_site_id, value.site_id, value.project_id, value.id);
  const organizationId = firstDefined(
    value.organization_id,
    value.owner_organization_id,
    value.project_site?.organization_id,
  );
  const organizationName = firstDefined(
    value.organization_name,
    value.organization,
    value.owner_organization_name,
    value.project_site?.organization_name,
    value.project_site?.organization,
  );
  const name = firstDefined(
    value.project_site_name,
    value.site_name,
    value.project_name,
    value.name,
    id == null ? 'Unassigned project site' : `Project Site ${id}`,
  );
  const alive = numericCount(verified.alive, verified.alive_count, value.latest_alive, value.verified_alive, summary.verified_alive);
  const dead = numericCount(verified.dead, verified.dead_count, value.latest_dead, value.verified_dead, summary.verified_dead);
  const missing = numericCount(verified.missing, verified.missing_count, value.latest_missing, value.verified_missing, summary.verified_missing);
  const planted = numericCount(
    value.planting_event_count,
    value.planted_seedlings,
    value.planted_count,
    value.planting_events,
    summary.planting_event_count,
    summary.planted_seedlings,
    summary.planted,
    value.total,
  );
  return {
    ...value,
    id,
    key: id == null ? `unassigned-${index}` : String(id),
    name,
    organizationId,
    organizationName,
    projectSiteIds: id == null ? [] : [id],
    projectSiteCount: id == null ? 0 : 1,
    planted,
    distinctPlantedPoints: numericCount(
      value.planted_point_count,
      value.distinct_planted_points,
      summary.planted_point_count,
      planted,
    ),
    planterCount: numericCount(value.planter_count, summary.planter_count, planterNames.length),
    planterNames,
    due: numericCount(inspections.due, inspections.due_count, value.due, value.due_inspections),
    overdue: numericCount(inspections.overdue, inspections.overdue_count, value.overdue, value.overdue_inspections),
    upcoming: numericCount(inspections.upcoming, inspections.upcoming_count, value.upcoming, value.upcoming_inspections),
    alive,
    dead,
    missing,
    unverified: numericCount(
      verified.unverified,
      verified.unverified_count,
      value.unverified,
      Math.max(0, planted - alive - dead - missing),
    ),
  };
}

function groupProjectsByOrganization(source) {
  const organizations = new Map();
  source.forEach((project, index) => {
    const organizationName = project.organizationName || project.name || 'Unassigned organization';
    const organizationKey = project.organizationId != null
      ? `organization:${project.organizationId}`
      : `organization-name:${String(organizationName).trim().toLowerCase()}`;
    if (!organizations.has(organizationKey)) {
      organizations.set(organizationKey, {
        key: organizationKey,
        id: project.organizationId,
        organizationId: project.organizationId,
        organizationName,
        name: organizationName,
        sites: new Map(),
      });
    }
    const organization = organizations.get(organizationKey);
    const siteKey = project.id == null ? `unassigned:${index}` : String(project.id);
    // A site may be present in both the primary payload and a compatibility
    // fallback. Keep one copy so organization totals never duplicate it.
    organization.sites.set(siteKey, project);
  });

  return [...organizations.values()].map((organization) => {
    const sites = [...organization.sites.values()];
    const planterNames = [...new Set(sites.flatMap((site) => site.planterNames || []))];
    const projectSiteIds = sites
      .map((site) => site.id)
      .filter((id) => id !== undefined && id !== null && id !== '');
    return {
      key: organization.key,
      id: organization.id,
      organizationId: organization.organizationId,
      organizationName: organization.organizationName,
      name: organization.name,
      sites,
      projectSiteIds,
      projectSiteCount: projectSiteIds.length,
      planted: sites.reduce((total, site) => total + site.planted, 0),
      distinctPlantedPoints: sites.reduce((total, site) => total + site.distinctPlantedPoints, 0),
      planterCount: planterNames.length || sites.reduce((total, site) => total + site.planterCount, 0),
      planterNames,
      due: sites.reduce((total, site) => total + site.due, 0),
      overdue: sites.reduce((total, site) => total + site.overdue, 0),
      upcoming: sites.reduce((total, site) => total + site.upcoming, 0),
      alive: sites.reduce((total, site) => total + site.alive, 0),
      dead: sites.reduce((total, site) => total + site.dead, 0),
      missing: sites.reduce((total, site) => total + site.missing, 0),
      unverified: sites.reduce((total, site) => total + site.unverified, 0),
    };
  });
}

function observationStatus(value) {
  const status = String(value || '').toLowerCase();
  return ['alive', 'dead', 'missing'].includes(status) ? status : 'unverified';
}

function roundState(round, asOfValue) {
  const explicitState = String(round?.state || round?.schedule_status || '').toLowerCase();
  if (explicitState === 'not_applicable') return 'not_applicable';
  if (explicitState === 'completed') return 'completed';
  if (
    round?.observation_id
    || round?.inspected_at
    || round?.completed_at
    || (round?.id && ['alive', 'dead', 'missing'].includes(String(round?.status || '').toLowerCase()) && round?.interval_days)
  ) return 'completed';
  if (Number(round?.days_overdue || 0) > 0) return 'overdue';
  const dueAt = new Date(
    round?.scheduled_for || round?.operational_due_at || round?.scheduled_at || round?.due_at || '',
  ).getTime();
  const asOf = new Date(asOfValue || '').getTime();
  if (Number.isFinite(dueAt) && Number.isFinite(asOf)) return dueAt <= asOf ? 'due' : 'upcoming';
  return explicitState || 'upcoming';
}

function roundTargetAt(round) {
  return firstDefined(
    round?.target_due_at,
    round?.exact_due_at,
    round?.scientific_due_at,
    round?.due_at,
  );
}

function roundFieldAt(round) {
  return firstDefined(
    round?.scheduled_for,
    round?.operational_due_at,
    round?.scheduled_at,
    round?.due_at,
  );
}

function inspectionRoundLabel(round) {
  const roundNumber = Number(firstDefined(round?.round_number, round?.sequence_no));
  const cadence = Number(firstDefined(
    round?.cadence_days,
    round?.inspection_interval_days,
    round?.recurring_interval_days,
  ));
  if (Number.isInteger(roundNumber) && roundNumber > 0 && Number.isInteger(cadence) && cadence > 0) {
    return `Round ${roundNumber} · every ${cadence} days`;
  }
  return `${round?.interval_days || cadence || '—'}-day round`;
}

function inspectionCountdown(round, asOfValue) {
  const scheduledDay = manilaDayNumber(roundFieldAt(round));
  const asOfDay = manilaDayNumber(asOfValue);
  if (scheduledDay === null || asOfDay === null) return '';
  const days = scheduledDay - asOfDay;
  if (days > 0) return `${days} day${days === 1 ? '' : 's'} remaining`;
  if (days < 0) return `${Math.abs(days)} day${days === -1 ? '' : 's'} overdue`;
  return 'Due today';
}

function pointIdentity(point) {
  return String(firstDefined(point?.planting_point_id, point?.point_id, point?.id, ''));
}

function eventIdentity(value) {
  return String(firstDefined(value?.planting_event_id, value?.event_id, ''));
}

function deathReasonLabel(key) {
  if (!key) return 'Not recorded';
  return DEATH_REASONS.find((reason) => reason.key === key)?.label
    || String(key).replaceAll('_', ' ').replace(/\b\w/g, (character) => character.toUpperCase());
}

function humanizeCode(value) {
  if (!value) return '';
  const text = String(value).replaceAll('_', ' ').trim();
  return text ? text.charAt(0).toUpperCase() + text.slice(1) : '';
}

function hasMeasuredHeight(value) {
  return value !== null && value !== undefined && value !== '' && Number.isFinite(Number(value));
}

function formatHeight(value) {
  if (!hasMeasuredHeight(value)) return '—';
  const height = Number(value);
  return `${height.toFixed(height % 1 === 0 ? 0 : 1)} cm`;
}

export default function Monitoring() {
  const adminToken = useAuthStore((s) => s.token);
  const points = useMapStore((s) => s.points);
  const fetchPoints = useMapStore((s) => s.fetchPoints);
  const fetchStats = useMapStore((s) => s.fetchStats);
  const fetchZones = useMapStore((s) => s.fetchZones);
  const loadingPoints = useMapStore((s) => s.loadingPoints);

  const monitoringSelectedPointId = useMapStore((s) => s.monitoringSelectedPointId);
  const setMonitoringSelectedPoint = useMapStore((s) => s.setMonitoringSelectedPoint);
  const clearMonitoringSelectedPoint = useMapStore((s) => s.clearMonitoringSelectedPoint);
  const resetPointToPlanned = useMapStore((s) => s.resetPointToPlanned);

  const [mortality, setMortality] = useState(null);
  const [mortalityError, setMortalityError] = useState('');

  // Scheduled observations are the source of truth for verified survival.
  // Legacy death history remains visible for operations and replanting, but
  // only scheduled LGU observations establish verified ecological outcomes.
  const [inspectionQueue, setInspectionQueue] = useState(null);
  const [inspectionLoading, setInspectionLoading] = useState(false);
  const [inspectionError, setInspectionError] = useState('');
  const [selectedInspection, setSelectedInspection] = useState(null);
  const [inspectionStatus, setInspectionStatus] = useState('');
  const [inspectionCondition, setInspectionCondition] = useState('');
  const [inspectionHeight, setInspectionHeight] = useState('');
  const [inspectionNotes, setInspectionNotes] = useState('');
  const [inspectionActionsTaken, setInspectionActionsTaken] = useState('');
  const [inspectionDeathReason, setInspectionDeathReason] = useState('');
  const [inspectionPhoto, setInspectionPhoto] = useState('');
  const [inspectionSubmitting, setInspectionSubmitting] = useState(false);
  const [inspectionFormError, setInspectionFormError] = useState('');
  const [recentObservations, setRecentObservations] = useState([]);
  const [observationsLoading, setObservationsLoading] = useState(false);
  const [observationsError, setObservationsError] = useState('');

  // Monitoring begins with stable LGU project sites. A project opens into a
  // searchable list of current planting events and their scheduled/completed
  // rounds. The normalizers below intentionally accept both the canonical
  // contract and the earlier site_* aliases used during the additive migration.
  const [projectRows, setProjectRows] = useState([]);
  const [projectsLoading, setProjectsLoading] = useState(false);
  const [projectsError, setProjectsError] = useState('');
  const [selectedProject, setSelectedProject] = useState(null);
  const [selectedQueueOrganizationKey, setSelectedQueueOrganizationKey] = useState(null);
  const [projectDetail, setProjectDetail] = useState(null);
  const [projectDetailLoading, setProjectDetailLoading] = useState(false);
  const [projectDetailError, setProjectDetailError] = useState('');
  const [projectSearch, setProjectSearch] = useState('');
  const [projectStatusFilter, setProjectStatusFilter] = useState('all');
  const [projectScheduleFilter, setProjectScheduleFilter] = useState('all');
  const [projectPlanterFilter, setProjectPlanterFilter] = useState('all');
  const [projectSpeciesFilter, setProjectSpeciesFilter] = useState('all');

  // A dead point can only be "Reset to planned" — opening the spot for a
  // fresh planting cycle. There is no un-die path: the death is permanent
  // and stays in mortality history regardless of what happens next on this
  // location.
  const [submitting, setSubmitting] = useState(false);
  const [formError, setFormError] = useState('');
  const [notice, setNotice] = useState('');
  // Tracks the dead-points list row currently being reset inline so we can
  // disable just that row's button instead of locking the whole list.
  const [resettingId, setResettingId] = useState(null);

  useEffect(() => {
    fetchPoints();
    fetchStats();
    fetchZones();
  }, [fetchPoints, fetchStats, fetchZones]);

  const refreshMortality = async () => {
    try {
      const res = await fetch(`${API}/api/planters/mortality-stats`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to load mortality stats');
      setMortality(data);
      setMortalityError('');
    } catch (err) {
      setMortalityError(err.message || 'Failed to load mortality stats');
    }
  };

  const [siteZones, setSiteZones] = useState([]);
  const [siteZonesError, setSiteZonesError] = useState('');
  const [zoneDetail, setZoneDetail] = useState(null);
  const [zoneDetailError, setZoneDetailError] = useState('');
  const [zoneDetailLoading, setZoneDetailLoading] = useState(false);
  const [breakdownOverlayOpen, setBreakdownOverlayOpen] = useState(false);
  // Paginated single-zone view in the breakdown overlay — one zone per page
  // so each gets its full breath (stats, cause bars, points table) instead
  // of being cramped into a vertical stack.
  const [zoneIndex, setZoneIndex] = useState(0);

  const refreshSiteZones = async () => {
    try {
      const res = await fetch(`${API}/api/zones/sites/mortality`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to load assignment zones');
      setSiteZones(Array.isArray(data?.zones) ? data.zones : []);
      setSiteZonesError('');
    } catch (err) {
      setSiteZonesError(err.message || 'Failed to load assignment zones');
    }
  };

  const refreshInspections = async () => {
    if (!adminToken) {
      setInspectionQueue(null);
      setInspectionError('Sign in again to load LGU inspections.');
      return;
    }
    setInspectionLoading(true);
    try {
      const res = await fetch(
        `${API}/api/monitoring/due`,
      );
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || 'Failed to load scheduled inspections');
      setInspectionQueue({
        ...data,
        observations_due: Array.isArray(data.observations_due)
          ? data.observations_due
          : Array.isArray(data.due)
            ? data.due
            : [],
      });
      setInspectionError('');
    } catch (err) {
      setInspectionError(err.message || 'Failed to load scheduled inspections');
    } finally {
      setInspectionLoading(false);
    }
  };

  const refreshRecentObservations = async () => {
    if (!adminToken) {
      setRecentObservations([]);
      setObservationsError('');
      return;
    }
    setObservationsLoading(true);
    try {
      const res = await fetch(
        `${API}/api/monitoring/observations`,
      );
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || 'Failed to load completed inspections');
      setRecentObservations(Array.isArray(data.observations) ? data.observations : []);
      setObservationsError('');
    } catch (err) {
      setObservationsError(err.message || 'Failed to load completed inspections');
    } finally {
      setObservationsLoading(false);
    }
  };

  const refreshProjects = async () => {
    if (!adminToken) {
      setProjectRows([]);
      setProjectsError('Sign in again to load project monitoring.');
      return;
    }
    setProjectsLoading(true);
    try {
      const res = await fetch(
        `${API}/api/monitoring/projects`,
      );
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || 'Failed to load monitoring projects');
      setProjectRows(arrayValue(data.projects, data.project_sites, data.sites, data.items));
      setProjectsError('');
    } catch (err) {
      // The fallback aggregation below still gives LGU staff a usable view
      // during additive migrations, while making the endpoint failure visible.
      setProjectRows([]);
      setProjectsError(err.message || 'Failed to load monitoring projects');
    } finally {
      setProjectsLoading(false);
    }
  };

  const refreshProjectDetail = async (project = selectedProject) => {
    if (!project || !adminToken) return;
    const projectIds = (project.projectSiteIds?.length
      ? project.projectSiteIds
      : [firstDefined(project.id, project.project_site_id, project.site_id)])
      .filter((id) => id !== undefined && id !== null && id !== '');
    if (projectIds.length === 0) {
      setProjectDetail({ project, points: [] });
      setProjectDetailError('');
      return;
    }
    setProjectDetailLoading(true);
    setProjectDetailError('');
    try {
      const details = await Promise.all(projectIds.map(async (projectId) => {
        const res = await fetch(
          `${API}/api/monitoring/projects/${encodeURIComponent(projectId)}`,
        );
        const data = await res.json().catch(() => ({}));
        if (!res.ok) throw new Error(data.detail || 'Failed to load organization seedlings');
        return data;
      }));
      const combine = (...keys) => details.flatMap((detail) => arrayValue(
        ...keys.map((key) => detail?.[key]),
      ));
      setProjectDetail({
        organization: project,
        projects: details,
        points: combine('points', 'planting_points', 'seedlings', 'planting_events'),
        observations: combine('observations', 'observation_history', 'inspection_history', 'history'),
        scheduled_rounds: combine('scheduled_rounds', 'inspection_rounds', 'due_rounds', 'schedule'),
      });
    } catch (err) {
      setProjectDetail(null);
      setProjectDetailError(err.message || 'Failed to load project seedlings');
    } finally {
      setProjectDetailLoading(false);
    }
  };

  const openProject = (project) => {
    setSelectedProject(project);
    setProjectDetail(null);
    setProjectDetailError('');
    setProjectSearch('');
    setProjectStatusFilter('all');
    setProjectScheduleFilter('all');
    setProjectPlanterFilter('all');
    setProjectSpeciesFilter('all');
    refreshProjectDetail(project);
  };

  const closeProject = () => {
    setSelectedProject(null);
    setProjectDetail(null);
    setProjectDetailError('');
    setProjectSearch('');
  };

  const openBreakdownOverlay = async () => {
    setBreakdownOverlayOpen(true);
    setZoneIndex(0);
    setZoneDetailError('');
    setZoneDetailLoading(true);
    try {
      const res = await fetch(`${API}/api/zones/sites/detail`);
      const data = await res.json();
      if (!res.ok) throw new Error(data.detail || 'Failed to load mortality detail');
      setZoneDetail(Array.isArray(data?.zones) ? data.zones : []);
    } catch (err) {
      setZoneDetailError(err.message || 'Failed to load mortality detail');
    } finally {
      setZoneDetailLoading(false);
    }
  };

  const closeBreakdownOverlay = () => {
    setBreakdownOverlayOpen(false);
  };

  useEffect(() => {
    const timer = window.setTimeout(() => {
      refreshMortality();
      refreshSiteZones();
      refreshInspections();
      refreshRecentObservations();
      refreshProjects();
    }, 0);
    return () => window.clearTimeout(timer);
  }, [adminToken]); // eslint-disable-line react-hooks/exhaustive-deps

  const openInspection = (item) => {
    const isCorrection = Boolean(item?.id || item?.observation_id || item?.inspected_at);
    setSelectedInspection(item);
    setInspectionStatus(
      isCorrection && ['alive', 'dead', 'missing'].includes(item?.status) ? item.status : '',
    );
    setInspectionCondition(isCorrection ? item?.condition || '' : '');
    setInspectionHeight(isCorrection ? item?.height_cm ?? '' : '');
    setInspectionNotes(isCorrection ? item?.notes || '' : '');
    setInspectionActionsTaken(isCorrection ? item?.actions_taken || '' : '');
    setInspectionDeathReason(isCorrection ? item?.death_reason_category || '' : '');
    setInspectionPhoto('');
    setInspectionFormError('');
  };

  const closeInspection = () => {
    if (inspectionSubmitting) return;
    setSelectedInspection(null);
    setInspectionFormError('');
    setInspectionPhoto('');
    setInspectionActionsTaken('');
  };

  const selectInspectionPhoto = (event) => {
    const file = event.target.files?.[0];
    if (!file) {
      setInspectionPhoto('');
      return;
    }
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type)) {
      setInspectionFormError('Use a JPEG, PNG, or WebP photo.');
      event.target.value = '';
      return;
    }
    if (file.size > 5 * 1024 * 1024) {
      setInspectionFormError('Inspection photos must be 5 MB or smaller.');
      event.target.value = '';
      return;
    }
    const reader = new FileReader();
    reader.onload = () => {
      setInspectionPhoto(typeof reader.result === 'string' ? reader.result : '');
      setInspectionFormError('');
    };
    reader.onerror = () => setInspectionFormError('Could not read that photo.');
    reader.readAsDataURL(file);
  };

  const submitInspection = async () => {
    if (!selectedInspection || !adminToken) return;
    if (!['alive', 'dead', 'missing'].includes(inspectionStatus)) {
      setInspectionFormError('Choose the status observed during this inspection.');
      return;
    }
    const height = inspectionStatus === 'alive' && inspectionHeight !== ''
      ? Number(inspectionHeight)
      : null;
    if (height !== null && (!Number.isFinite(height) || height < 0 || height > 5000)) {
      setInspectionFormError('Height must be between 0 and 5,000 cm.');
      return;
    }
    if (inspectionStatus === 'dead' && !inspectionDeathReason) {
      setInspectionFormError('Choose a cause of death.');
      return;
    }

    setInspectionSubmitting(true);
    setInspectionFormError('');
    try {
      const res = await fetch(
        `${API}/api/monitoring/observations`,
        {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            planting_event_id: selectedInspection.planting_event_id,
            interval_days: selectedInspection.interval_days,
            status: inspectionStatus,
            condition: inspectionStatus === 'alive' ? inspectionCondition || null : null,
            height_cm: height,
            notes: inspectionNotes.trim() || null,
            actions_taken: inspectionActionsTaken.trim() || null,
            death_reason_category: inspectionStatus === 'dead' ? inspectionDeathReason : null,
            photo_data_url: inspectionPhoto || null,
          }),
        },
      );
      const data = await res.json().catch(() => ({}));
      if (!res.ok) throw new Error(data.detail || 'Could not save the inspection');
      setNotice(
        `Point #${selectedInspection.point_num ?? selectedInspection.planting_point_id} `
        + `${selectedInspection.id || selectedInspection.observation_id ? 'corrected' : 'recorded'} as ${inspectionStatus}.`,
      );
      setSelectedInspection(null);
      setInspectionPhoto('');
      setInspectionActionsTaken('');
      await Promise.all([
        refreshInspections(),
        refreshRecentObservations(),
        refreshProjects(),
        selectedProject ? refreshProjectDetail(selectedProject) : Promise.resolve(),
        refreshMortality(),
        refreshSiteZones(),
        fetchPoints(),
        fetchStats(),
      ]);
    } catch (err) {
      setInspectionFormError(err.message || 'Could not save the inspection.');
    } finally {
      setInspectionSubmitting(false);
    }
  };

  // Tear down monitoring selection when navigating away.
  useEffect(() => () => clearMonitoringSelectedPoint(), [clearMonitoringSelectedPoint]);

  const selectedPoint = useMemo(
    () => points.find((p) => Number(p.id) === Number(monitoringSelectedPointId)) || null,
    [points, monitoringSelectedPointId],
  );

  const selectedStatus = selectedPoint ? getPointStatus(selectedPoint) : null;
  const isSelectionDead = selectedStatus === 'dead';

  useEffect(() => {
    if (!selectedPoint) return undefined;
    const timer = window.setTimeout(() => {
      setFormError('');
    }, 0);
    return () => window.clearTimeout(timer);
  }, [monitoringSelectedPointId]); // eslint-disable-line react-hooks/exhaustive-deps

  const visiblePoints = useMemo(
    () => points.filter((p) => Number.isFinite(Number(p.latitude)) && Number.isFinite(Number(p.longitude))),
    [points],
  );

  const counts = useMemo(() => {
    const c = { planned: 0, assigned: 0, planted: 0, dead: 0, skipped: 0 };
    for (const p of visiblePoints) {
      const s = getPointStatus(p);
      if (s in c) c[s] += 1;
    }
    return c;
  }, [visiblePoints]);
  const historicalDeathCount = typeof mortality?.dead_count === 'number'
    ? mortality.dead_count
    : 0;

  // Recent deaths reads from the immutable death history so it shows every
  // recorded death, even after the spot has been reset to planned. Each
  // entry carries is_currently_dead so we can tell live deaths apart from
  // historical-only entries (reset spots) — only live deaths offer the
  // inline "Reset to planned" action.
  const recentDeaths = useMemo(() => (
    Array.isArray(mortality?.recent_deaths) ? mortality.recent_deaths : []
  ), [mortality]);

  const inspectionRows = useMemo(
    () => (Array.isArray(inspectionQueue?.observations_due)
      ? inspectionQueue.observations_due
      : []),
    [inspectionQueue],
  );
  const inspectionSummary = inspectionQueue?.summary || {};

  const fallbackProjects = useMemo(() => {
    const groups = new Map();
    const ensureGroup = (source = {}) => {
      const identity = projectIdentity(source);
      if (!groups.has(identity)) {
        const id = firstDefined(source.project_site_id, source.site_id, source.project_id);
        groups.set(identity, {
          id,
          project_site_id: id,
          name: firstDefined(source.project_site_name, source.site_name, 'Unassigned project site'),
          organization_id: firstDefined(source.organization_id, source.owner_organization_id),
          organization_name: firstDefined(source.organization_name, source.organization, source.owner_organization_name),
          plantedKeys: new Set(),
          planters: new Set(),
          due: 0,
          overdue: 0,
          upcoming: 0,
          latestByEvent: new Map(),
        });
      }
      return groups.get(identity);
    };

    visiblePoints
      .filter((point) => point.planted_at || ['planted', 'dead'].includes(getPointStatus(point)))
      .forEach((point) => {
        const group = ensureGroup(point);
        group.plantedKeys.add(`point:${point.id}`);
        if (point.assigned_planter_name) group.planters.add(point.assigned_planter_name);
      });

    inspectionRows.forEach((round) => {
      const group = ensureGroup(round);
      group.plantedKeys.add(`event:${eventIdentity(round) || pointIdentity(round)}`);
      if (round.planter_name) group.planters.add(round.planter_name);
      const state = roundState(round, inspectionQueue?.as_of);
      if (state === 'overdue') group.overdue += 1;
      else if (state === 'due') group.due += 1;
      else if (state === 'upcoming') group.upcoming += 1;
    });

    recentObservations.forEach((observation) => {
      const group = ensureGroup(observation);
      const eventKey = eventIdentity(observation) || pointIdentity(observation);
      group.plantedKeys.add(`event:${eventKey}`);
      if (observation.planter_name) group.planters.add(observation.planter_name);
      const current = group.latestByEvent.get(eventKey);
      const currentTime = new Date(current?.inspected_at || 0).getTime();
      const nextTime = new Date(observation.inspected_at || 0).getTime();
      if (!current || nextTime >= currentTime) group.latestByEvent.set(eventKey, observation);
    });

    return [...groups.values()].map((group) => {
      const outcomes = { alive: 0, dead: 0, missing: 0 };
      group.latestByEvent.forEach((observation) => {
        const status = observationStatus(observation.status);
        if (status in outcomes) outcomes[status] += 1;
      });
      return normalizeProject({
        id: group.id,
        name: group.name,
        organization_id: group.organization_id,
        organization_name: group.organization_name,
        planted_seedlings: group.plantedKeys.size,
        planter_count: group.planters.size,
        planters: [...group.planters],
        inspections: { due: group.due, overdue: group.overdue, upcoming: group.upcoming },
        verified: outcomes,
      });
    });
  }, [inspectionQueue?.as_of, inspectionRows, recentObservations, visiblePoints]);

  const projects = useMemo(() => {
    const normalized = projectRows.map(normalizeProject);
    const source = normalized.length > 0 ? normalized : fallbackProjects;
    return groupProjectsByOrganization(source).sort((a, b) => (
      (b.overdue - a.overdue)
      || (b.due - a.due)
      || String(a.name).localeCompare(String(b.name))
    ));
  }, [fallbackProjects, projectRows]);

  const inspectionOrganizations = useMemo(() => {
    const projectBySite = new Map();
    projects.forEach((organization) => {
      organization.projectSiteIds.forEach((siteId) => {
        projectBySite.set(String(siteId), organization);
      });
    });
    const groups = new Map();
    const asOf = new Date(inspectionQueue?.as_of || '').getTime();
    inspectionRows.forEach((row) => {
      const project = projectBySite.get(projectIdentity(row));
      const plantedByLgu = row.planted_by_user_id != null || row.planting_source === 'lgu_direct';
      const organizationId = plantedByLgu ? 0 : firstDefined(row.organization_id, project?.organizationId);
      const organizationName = plantedByLgu ? 'LGU' : firstDefined(
        row.organization_name, row.organization, project?.organizationName,
        project?.name, 'Unassigned organization',
      );
      const key = organizationId != null
        ? `organization:${organizationId}`
        : `organization-name:${String(organizationName).trim().toLowerCase()}`;
      if (!groups.has(key)) {
        groups.set(key, {
          key,
          organizationId,
          name: organizationName,
          rows: [],
          pointIds: new Set(),
          due: 0,
          overdue: 0,
          upcoming: 0,
        });
      }
      const group = groups.get(key);
      group.rows.push(row);
      const pointId = firstDefined(row.planting_point_id, row.point_id);
      if (pointId != null) group.pointIds.add(String(pointId));
      const scheduledAt = new Date(roundFieldAt(row) || '').getTime();
      if (Number(row.days_overdue || 0) > 0) group.overdue += 1;
      if (Number.isFinite(scheduledAt) && Number.isFinite(asOf) && scheduledAt > asOf) {
        group.upcoming += 1;
      } else {
        group.due += 1;
      }
    });
    return [...groups.values()]
      .map((group) => ({ ...group, plantedPointCount: group.pointIds.size }))
      .sort((a, b) => (b.overdue - a.overdue) || (b.due - a.due) || a.name.localeCompare(b.name));
  }, [inspectionQueue?.as_of, inspectionRows, projects]);

  const selectedQueueOrganization = useMemo(
    () => inspectionOrganizations.find((group) => group.key === selectedQueueOrganizationKey) || null,
    [inspectionOrganizations, selectedQueueOrganizationKey],
  );

  const projectPoints = useMemo(() => {
    if (!selectedProject) return [];
    const selectedIdentities = new Set((selectedProject.projectSiteIds || []).map(String));
    const detailPoints = arrayValue(
      projectDetail?.points,
      projectDetail?.planting_points,
      projectDetail?.seedlings,
      projectDetail?.planting_events,
    );
    const fallbackPoints = visiblePoints.filter((point) => (
      selectedIdentities.has(projectIdentity(point))
      && (point.planted_at || ['planted', 'dead'].includes(getPointStatus(point)))
    ));
    const sourcePoints = detailPoints.length > 0 ? detailPoints : fallbackPoints;
    const detailObservations = arrayValue(
      projectDetail?.observations,
      projectDetail?.observation_history,
      projectDetail?.inspection_history,
      projectDetail?.history,
    );
    const detailRounds = arrayValue(
      projectDetail?.scheduled_rounds,
      projectDetail?.inspection_rounds,
      projectDetail?.due_rounds,
      projectDetail?.schedule,
    );

    return sourcePoints.map((point, index) => {
      const pointId = pointIdentity(point);
      const eventId = eventIdentity(point);
      const matchesRecord = (record) => {
        const recordEvent = eventIdentity(record);
        const recordPoint = pointIdentity(record);
        return (eventId && recordEvent === eventId) || (pointId && recordPoint === pointId);
      };
      const historyCandidates = [
        ...arrayValue(point.observations, point.observation_history, point.inspection_history, point.history, point.growth_history),
        ...detailObservations.filter(matchesRecord),
        ...recentObservations.filter(matchesRecord),
      ];
      const historyMap = new Map();
      historyCandidates.forEach((observation, observationIndex) => {
        const key = firstDefined(
          observation.id,
          observation.observation_id,
          `${eventIdentity(observation)}:${observation.interval_days}:${observationIndex}`,
        );
        historyMap.set(String(key), observation);
      });
      const history = [...historyMap.values()].sort((a, b) => (
        numericCount(a.interval_days) - numericCount(b.interval_days)
        || new Date(a.inspected_at || 0).getTime() - new Date(b.inspected_at || 0).getTime()
      ));
      const latest = history.length > 0 ? history[history.length - 1] : point.latest_observation || null;
      const latestHeightObservation = [...history]
        .reverse()
        .find((observation) => hasMeasuredHeight(observation.height_cm)) || null;

      const roundCandidates = [
        ...arrayValue(point.scheduled_rounds, point.inspection_rounds, point.due_rounds, point.schedule),
        ...detailRounds.filter(matchesRecord),
        ...inspectionRows.filter(matchesRecord),
      ];
      const roundMap = new Map();
      roundCandidates.forEach((round, roundIndex) => {
        const nestedObservation = round?.observation && typeof round.observation === 'object'
          ? round.observation
          : {};
        const mergedRound = { ...round, ...nestedObservation };
        const key = `${eventIdentity(mergedRound) || eventId}:${mergedRound.interval_days ?? roundIndex}`;
        roundMap.set(key, {
          ...point,
          ...mergedRound,
          status: firstDefined(mergedRound.status, mergedRound.observation_status),
          planting_event_id: firstDefined(mergedRound.planting_event_id, eventId),
          planting_point_id: firstDefined(mergedRound.planting_point_id, pointId),
          point_num: firstDefined(mergedRound.point_num, point.point_num, point.number),
          site_name: firstDefined(mergedRound.site_name, mergedRound.project_site_name, point.site_name, point.project_site_name),
          planter_name: firstDefined(mergedRound.planter_name, point.planter_name, point.assigned_planter_name),
          species: firstDefined(mergedRound.species, point.species, point.assignment_species),
          planted_at: firstDefined(mergedRound.planted_at, point.planted_at),
        });
      });
      const rounds = [...roundMap.values()].sort((a, b) => (
        numericCount(a.interval_days) - numericCount(b.interval_days)
      ));
      const roundStates = rounds.map((round) => roundState(round, inspectionQueue?.as_of));
      const scheduleState = roundStates.includes('overdue')
        ? 'overdue'
        : roundStates.includes('due')
          ? 'due'
          : roundStates.includes('upcoming')
            ? 'upcoming'
            : roundStates.includes('not_applicable')
              ? 'not_applicable'
              : history.length > 0
              ? 'completed'
              : 'none';
      const latestStatus = observationStatus(firstDefined(
        point.latest_status,
        point.latest_observation_status,
        latest?.status,
      ));
      return {
        ...point,
        key: eventId || pointId || String(index),
        planting_event_id: firstDefined(point.planting_event_id, point.event_id, latest?.planting_event_id),
        planting_point_id: firstDefined(point.planting_point_id, point.point_id, point.id),
        point_num: firstDefined(point.point_num, point.number, point.planting_point_id, point.id),
        planted_at: firstDefined(point.planted_at, point.planting_timestamp, latest?.planted_at),
        planter_name: firstDefined(point.planter_name, point.assigned_planter_name, latest?.planter_name),
        species: firstDefined(point.species, point.assignment_species, latest?.species),
        assignment_title: firstDefined(point.assignment_title, point.title, latest?.assignment_title),
        latest,
        latestHeightObservation,
        latestStatus,
        history,
        rounds,
        scheduleState,
      };
    }).sort((a, b) => numericCount(a.point_num) - numericCount(b.point_num));
  }, [inspectionQueue?.as_of, inspectionRows, projectDetail, recentObservations, selectedProject, visiblePoints]);

  const projectFilterOptions = useMemo(() => ({
    planters: [...new Set(projectPoints.map((point) => point.planter_name).filter(Boolean))].sort(),
    species: [...new Set(projectPoints.map((point) => point.species).filter(Boolean))].sort(),
  }), [projectPoints]);

  const filteredProjectPoints = useMemo(() => {
    const search = projectSearch.trim().toLowerCase();
    return projectPoints.filter((point) => {
      const searchable = [
        point.point_num,
        point.planter_name,
        point.species,
        point.assignment_title,
        point.latest?.death_reason_category,
        point.latest?.notes,
        point.latest?.actions_taken,
        ...point.history.flatMap((observation) => [
          observation.death_reason_category,
          observation.condition,
          observation.notes,
          observation.actions_taken,
        ]),
      ].filter(Boolean).join(' ').toLowerCase();
      return (!search || searchable.includes(search))
        && (projectStatusFilter === 'all' || point.latestStatus === projectStatusFilter)
        && (projectScheduleFilter === 'all' || point.scheduleState === projectScheduleFilter)
        && (projectPlanterFilter === 'all' || point.planter_name === projectPlanterFilter)
        && (projectSpeciesFilter === 'all' || point.species === projectSpeciesFilter);
    });
  }, [projectPlanterFilter, projectPoints, projectScheduleFilter, projectSearch, projectSpeciesFilter, projectStatusFilter]);

  const selectedDueInspection = useMemo(() => {
    if (!selectedPoint || isSelectionDead) return null;
    const asOf = new Date(inspectionQueue?.as_of || '').getTime();
    if (!Number.isFinite(asOf)) return null;
    return inspectionRows.find((row) => {
      const samePoint = Number(row.planting_point_id) === Number(selectedPoint.id);
      const dueAt = new Date(roundFieldAt(row) || '').getTime();
      return samePoint && Number.isFinite(dueAt) && dueAt <= asOf;
    }) || null;
  }, [inspectionQueue?.as_of, inspectionRows, isSelectionDead, selectedPoint]);

  const closeModal = () => {
    if (submitting) return;
    clearMonitoringSelectedPoint();
    setFormError('');
  };

  const openSelectedDueInspection = () => {
    if (!selectedDueInspection) {
      closeModal();
      return;
    }
    clearMonitoringSelectedPoint();
    openInspection(selectedDueInspection);
  };

  const handleInlineReset = async (point) => {
    if (!point || resettingId === point.id) return;
    setResettingId(point.id);
    try {
      await resetPointToPlanned(point.id);
      setNotice(`Point #${point.point_num} reset to planned — ready for a fresh assignment.`);
      await Promise.all([
        fetchPoints(),
        refreshMortality(),
        refreshSiteZones(),
        refreshProjects(),
        selectedProject ? refreshProjectDetail(selectedProject) : Promise.resolve(),
      ]);
    } catch (err) {
      setFormError(err.message || 'Could not reset point.');
    } finally {
      setResettingId(null);
    }
  };

  const submitDeadAction = async () => {
    if (!selectedPoint) return;
    setSubmitting(true);
    setFormError('');
    try {
      // The only action on a dead point is to reset it to planned (free the
      // spot for a fresh planting cycle). The death record is intentionally
      // kept in mortality history.
      await resetPointToPlanned(selectedPoint.id);
      setNotice(`Point #${selectedPoint.point_num} reset to planned — ready for a fresh assignment.`);
      await Promise.all([
        fetchPoints(),
        refreshMortality(),
        refreshSiteZones(),
        refreshProjects(),
        selectedProject ? refreshProjectDetail(selectedProject) : Promise.resolve(),
      ]);
      clearMonitoringSelectedPoint();
    } catch (err) {
      setFormError(err.message || 'Could not update point.');
    } finally {
      setSubmitting(false);
    }
  };

  return (
    <div className="monitoring-page">
      <Panel
        title="Monitoring"
        subtitle={`${counts.planted} planted · ${counts.dead} currently dead · ${counts.skipped} skipped`}
      >
        <MonitoringProjects
          projects={projects}
          loading={projectsLoading || inspectionLoading || observationsLoading || loadingPoints}
          projectError={projectsError}
          inspectionError={inspectionError}
          observationsError={observationsError}
          inspectionSummary={inspectionSummary}
          notice={notice}
          onOpen={openProject}
        />

        <PanelCard
          title="Monitoring guidance"
          defaultOpen={false}
          icon={(
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><circle cx="12" cy="12" r="10" /><path d="M12 16v-4" /><path d="M12 8h.01" /></svg>
          )}
        >
          <p className="monitoring-hint">
            Use <strong>LGU follow-up inspections</strong> to record alive, dead, or missing outcomes.
            A planted point with no death record is still unverified. Clicking a planted point shows whether
            a scheduled round is due. Dead points can be <strong>Reset to planned</strong> for a new planting
            cycle; their historical death record remains preserved.
          </p>
        </PanelCard>

        <PanelCard
          title="All operational points (not verified outcomes)"
          badge={visiblePoints.length}
          defaultOpen={false}
          icon={(
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" /><circle cx="12" cy="10" r="3" /></svg>
          )}
        >
          <div className="monitoring-counts">
            <StatusPill color="#16a34a" label="Planned" value={counts.planned} />
            <StatusPill color="#2563eb" label="Assigned" value={counts.assigned} />
            <StatusPill color="#eab308" label="Planted, unverified" value={counts.planted} />
            <StatusPill color="#7f1d1d" label="Currently dead" value={counts.dead} />
            <StatusPill color="#9ca3af" label="Skipped" value={counts.skipped} />
            <StatusPill color="#450a0a" label="Recorded deaths (history)" value={historicalDeathCount} />
          </div>
          {loadingPoints && <div className="monitoring-muted">Loading points...</div>}
          {notice && <div className="monitoring-notice">{notice}</div>}
        </PanelCard>

        <PanelCard
          title="Follow-up queue by organization"
          badge={inspectionOrganizations.length}
          defaultOpen={false}
          icon={(
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 11l3 3L22 4" /><path d="M21 12v7a2 2 0 0 1-2 2H5a2 2 0 0 1-2-2V5a2 2 0 0 1 2-2h11" /></svg>
          )}
        >
          {inspectionLoading && <div className="monitoring-muted">Loading inspection schedule...</div>}
          {inspectionError && <div className="monitoring-error">{inspectionError}</div>}
          {!inspectionLoading && !inspectionError && inspectionQueue && (
            inspectionOrganizations.length === 0 ? (
              <div className="monitoring-muted">
                No inspections are scheduled. New follow-ups appear after planting events are recorded.
              </div>
            ) : (
              <div className="monitoring-project-grid">
                {inspectionOrganizations.map((organization) => (
                  <button
                    key={organization.key}
                    type="button"
                    className="monitoring-project-card"
                    onClick={() => setSelectedQueueOrganizationKey(organization.key)}
                  >
                    <span className="monitoring-project-card-head">
                      <span>
                        <strong>{organization.name}</strong>
                        <small>
                          {organization.plantedPointCount} planted {organization.plantedPointCount === 1 ? 'point' : 'points'}
                          {` · ${organization.rows.length} scheduled follow-ups`}
                        </small>
                      </span>
                      <span className="monitoring-project-open">View details →</span>
                    </span>
                  </button>
                ))}
              </div>
            )
          )}
        </PanelCard>

        <PanelCard
          title="Completed LGU observations"
          badge={recentObservations.length}
          defaultOpen={false}
          icon={(
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 11l3 3L22 4" /><path d="M5 4h11v16H5z" /></svg>
          )}
        >
          <p className="monitoring-hint">
            Open a completed round to correct an entry. Corrections retain the same planting event
            and inspection round and update linked death history atomically.
          </p>
          {observationsLoading && <div className="monitoring-muted">Loading completed observations...</div>}
          {observationsError && <div className="monitoring-error">{observationsError}</div>}
          {!observationsLoading && !observationsError && recentObservations.length === 0 ? (
            <div className="monitoring-muted">No verified observations have been submitted yet.</div>
          ) : null}
          {!observationsLoading && recentObservations.length > 0 ? (
            <div className="monitoring-inspection-list">
              {recentObservations.slice(0, 20).map((item) => (
                <button
                  key={item.id ?? `${item.planting_event_id}-${item.interval_days}`}
                  type="button"
                  className="monitoring-inspection-row"
                  onClick={() => openInspection(item)}
                >
                  <span className="monitoring-inspection-main">
                    <strong>Point #{item.point_num ?? item.planting_point_id}</strong>
                    <span>{item.site_name || item.assignment_title || 'Unassigned project site'}</span>
                  </span>
                  <span className="monitoring-inspection-due">
                    <strong>{item.interval_days}-day · {statusLabel(item.status)}</strong>
                    <span>{formatDateTime(item.inspected_at)} · Edit</span>
                  </span>
                </button>
              ))}
              {recentObservations.length > 20 ? (
                <div className="monitoring-muted">Showing the 20 most recent of {recentObservations.length} observations.</div>
              ) : null}
            </div>
          ) : null}
        </PanelCard>

        <PanelCard
          title="Operational death history"
          badge={mortality?.dead_count ?? 0}
          defaultOpen={false}
          icon={(
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M9 12l2 2 4-4" /><circle cx="12" cy="12" r="10" /></svg>
          )}
        >
          {mortalityError && <div className="monitoring-error">{mortalityError}</div>}
          {!mortality ? (
            <div className="monitoring-muted">Loading mortality data...</div>
          ) : (
            <>
              <div className="monitoring-mortality-summary">
                <div>
                  <div className="monitoring-mortality-rate">{mortality.mortality_rate.toFixed(1)}%</div>
                  <div className="monitoring-mortality-rate-label">recorded operational mortality</div>
                </div>
                <div className="monitoring-mortality-fraction">
                  <span><strong>{mortality.dead_count}</strong> dead</span>
                  <span><strong>{mortality.planted_alive}</strong> currently planted with no death record</span>
                  <span><strong>{mortality.ever_planted}</strong> ever planted</span>
                </div>
              </div>
              <div className="monitoring-muted">
                This legacy operational ratio is not verified cohort survival. Use the decision dashboard for comparable inspection-round rates.
              </div>
              {mortality.breakdown.length === 0 ? (
                <div className="monitoring-muted">No deaths recorded yet.</div>
              ) : (
                <div className="monitoring-breakdown">
                  {mortality.breakdown.map((row) => (
                    <div key={row.category} className="monitoring-breakdown-row">
                      <div className="monitoring-breakdown-label">
                        <span>{row.label}</span>
                        <span className="monitoring-breakdown-pct">{row.percent_of_dead.toFixed(1)}%</span>
                      </div>
                      <div className="monitoring-breakdown-bar">
                        <div className="monitoring-breakdown-fill" style={{ width: `${row.percent_of_dead}%` }} />
                      </div>
                      <div className="monitoring-breakdown-meta">
                        {row.count} dead{mortality.ever_planted ? ` · ${row.percent_of_planted.toFixed(1)}% of all planted` : ''}
                      </div>
                    </div>
                  ))}
                </div>
              )}
              <button
                type="button"
                className="btn btn-primary btn-sm monitoring-breakdown-cta"
                onClick={openBreakdownOverlay}
              >
                View operational table by assignment zone
              </button>
            </>
          )}
        </PanelCard>

        <PanelCard
          title="Assignment zones"
          badge={siteZones.length}
          defaultOpen={false}
          icon={(
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 12l9-9 9 9-9 9-9-9z"/><circle cx="12" cy="12" r="2"/></svg>
          )}
        >
          {siteZonesError && <div className="monitoring-error">{siteZonesError}</div>}
          {siteZones.length === 0 ? (
            <div className="monitoring-muted">
              No assignment zones yet. Each planter assignment automatically creates one.
            </div>
          ) : (
            <div className="monitoring-site-zone-list">
              {siteZones.map((zone) => (
                <div key={zone.id} className="monitoring-site-zone-row">
                  <div className="monitoring-site-zone-info">
                    <div className="monitoring-site-zone-name">{zone.name}</div>
                    <div className="monitoring-site-zone-meta">
                      {zone.total > 0
                        ? `${zone.alive} currently planted with no death record · ${zone.dead} recorded deaths`
                        : 'No planted points yet'}
                    </div>
                  </div>
                  <span
                    className={`monitoring-site-zone-pill ${zone.dead > 0 ? 'is-has-deaths' : 'is-all-alive'}`}
                    title={`${zone.dead} recorded deaths among ${zone.total} operational planting records; not verified survival`}
                  >
                    {zone.total > 0 ? `${zone.dead} death${zone.dead === 1 ? '' : 's'}` : '—'}
                  </span>
                </div>
              ))}
            </div>
          )}
        </PanelCard>

        <PanelCard
          title="Recent operational death records"
          badge={recentDeaths.length}
          icon={(
            <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 12a9 9 0 1 1 9 9" /><path d="M12 7v5l3 2" /></svg>
          )}
          defaultOpen={false}
        >
          {recentDeaths.length === 0 ? (
            <div className="monitoring-muted">No deaths recorded yet.</div>
          ) : (
            <div className="monitoring-dead-list">
              {recentDeaths.map((p) => {
                const busy = resettingId === p.id;
                const stillDead = Boolean(p.is_currently_dead);
                return (
                  <div key={p.death_record_id || p.id} className="monitoring-dead-row">
                    <button
                      type="button"
                      className="monitoring-dead-row-main"
                      onClick={() => stillDead && setMonitoringSelectedPoint(p.id)}
                      title={stillDead ? 'Open point details' : 'Spot already reopened — historical record only'}
                      disabled={busy || !stillDead}
                    >
                      <div>
                        <div className="monitoring-dead-title">
                          Point #{p.point_num}
                          {!stillDead && (
                            <span className="monitoring-dead-tag" title="Spot reset to planned — death stays in mortality history">
                              spot reopened
                            </span>
                          )}
                        </div>
                        <div className="monitoring-dead-meta">{p.death_reason || 'No reason recorded'}</div>
                      </div>
                      <div className="monitoring-dead-when">{formatDateTime(p.death_at)}</div>
                    </button>
                    {stillDead && (
                      <button
                        type="button"
                        className="btn btn-ghost btn-sm monitoring-dead-reset"
                        onClick={() => handleInlineReset(p)}
                        disabled={busy}
                        title="Mark this location as available for re-assignment"
                      >
                        {busy ? 'Resetting…' : 'Reset to planned'}
                      </button>
                    )}
                  </div>
                );
              })}
            </div>
          )}
        </PanelCard>
      </Panel>

      <Modal
        open={Boolean(selectedQueueOrganization)}
        title={selectedQueueOrganization ? `${selectedQueueOrganization.name} follow-up queue` : 'Organization follow-up queue'}
        variant="info"
        cancelLabel="Close"
        onCancel={() => setSelectedQueueOrganizationKey(null)}
        className="modal-card-wide monitoring-project-modal"
      >
        {selectedQueueOrganization && (
          <div className="monitoring-organization-queue-detail">
            <p className="monitoring-hint">
              Select a due or overdue round to record an LGU observation. Upcoming rounds are shown
              for planning and become actionable on their scheduled field date.
            </p>
            <div className="monitoring-inspection-summary">
              <div><strong>{selectedQueueOrganization.due}</strong><span>Due total</span></div>
              <div><strong>{selectedQueueOrganization.overdue}</strong><span>Overdue subset</span></div>
              <div><strong>{selectedQueueOrganization.upcoming}</strong><span>Upcoming</span></div>
            </div>
            <div className="monitoring-inspection-list">
              {selectedQueueOrganization.rows.map((item) => {
                const overdue = Number(item.days_overdue || 0) > 0;
                const dueAt = new Date(roundFieldAt(item) || '').getTime();
                const queueAsOf = new Date(inspectionQueue?.as_of || '').getTime();
                const upcoming = Number.isFinite(dueAt)
                  && Number.isFinite(queueAsOf)
                  && dueAt > queueAsOf;
                return (
                  <button
                    key={`${item.planting_event_id}-${item.interval_days}`}
                    type="button"
                    className={`monitoring-inspection-row ${overdue ? 'is-overdue' : ''} ${upcoming ? 'is-upcoming' : ''}`}
                    onClick={() => {
                      setSelectedQueueOrganizationKey(null);
                      openInspection(item);
                    }}
                    disabled={upcoming}
                    title={upcoming ? 'This inspection round is not due yet.' : undefined}
                  >
                    <span className="monitoring-inspection-main">
                      <strong>Point #{item.point_num ?? item.planting_point_id}</strong>
                      <span>
                        {item.site_name || item.assignment_title || item.image_name || 'Unassigned project site'}
                        {item.species ? ` - ${item.species}` : ''}
                      </span>
                    </span>
                    <span className="monitoring-inspection-due">
                      <strong>{inspectionRoundLabel(item)}</strong>
                      <span>{inspectionCountdown(item, inspectionQueue?.as_of)}</span>
                      <small>Target {formatDateTime(roundTargetAt(item))}</small>
                      <small>LGU field date {formatDateTime(roundFieldAt(item))}</small>
                    </span>
                  </button>
                );
              })}
            </div>
          </div>
        )}
      </Modal>

      <Modal
        open={Boolean(selectedProject)}
        title={selectedProject?.name || 'Project monitoring'}
        variant="info"
        cancelLabel="Close"
        onCancel={closeProject}
        className="modal-card-wide monitoring-project-modal"
      >
        {selectedProject && (
          <MonitoringProjectDetail
            project={selectedProject}
            points={projectPoints}
            filteredPoints={filteredProjectPoints}
            filters={{
              search: projectSearch,
              status: projectStatusFilter,
              schedule: projectScheduleFilter,
              planter: projectPlanterFilter,
              species: projectSpeciesFilter,
            }}
            setFilters={{
              search: setProjectSearch,
              status: setProjectStatusFilter,
              schedule: setProjectScheduleFilter,
              planter: setProjectPlanterFilter,
              species: setProjectSpeciesFilter,
            }}
            filterOptions={projectFilterOptions}
            loading={projectDetailLoading}
            error={projectDetailError}
            notice={notice}
            asOf={inspectionQueue?.as_of}
            mapPoints={visiblePoints}
            onBack={closeProject}
            onInspect={openInspection}
            onViewMap={setMonitoringSelectedPoint}
          />
        )}
      </Modal>

      <Modal
        open={Boolean(selectedInspection)}
        title={selectedInspection
          ? `${selectedInspection.id || selectedInspection.observation_id ? 'Correct' : 'Inspect'} Point #${selectedInspection.point_num ?? selectedInspection.planting_point_id}`
          : 'Record follow-up inspection'}
        variant={inspectionStatus === 'dead' ? 'danger' : 'info'}
        confirmLabel={selectedInspection?.id || selectedInspection?.observation_id ? 'Save correction' : 'Save observation'}
        cancelLabel="Cancel"
        busy={inspectionSubmitting}
        onConfirm={submitInspection}
        onCancel={closeInspection}
      >
        {selectedInspection && (
          <div className="monitoring-form monitoring-inspection-form">
            <div className="monitoring-form-summary">
              <div><strong>Follow-up:</strong> {inspectionRoundLabel(selectedInspection)}</div>
              <div><strong>Planted:</strong> {formatDateTime(selectedInspection.planted_at)}</div>
              {roundTargetAt(selectedInspection) ? <div><strong>Exact target:</strong> {formatDateTime(roundTargetAt(selectedInspection))}</div> : null}
              {roundFieldAt(selectedInspection) ? <div><strong>LGU field date:</strong> {formatDateTime(roundFieldAt(selectedInspection))} ({inspectionCountdown(selectedInspection, inspectionQueue?.as_of)})</div> : null}
              {selectedInspection.id || selectedInspection.observation_id ? <div><strong>Previously inspected:</strong> {formatDateTime(selectedInspection.inspected_at)}</div> : null}
              <div><strong>Site:</strong> {selectedInspection.site_name || 'No project site assigned'}</div>
              {selectedInspection.planter_name && <div><strong>Planter:</strong> {selectedInspection.planter_name}</div>}
              {selectedInspection.species && <div><strong>Species:</strong> {selectedInspection.species}</div>}
            </div>

            <fieldset className="monitoring-inspection-statuses">
              <legend>Observed status</legend>
              {[
                ['alive', 'Alive'],
                ['dead', 'Dead'],
                ['missing', 'Missing / not found'],
              ].map(([value, label]) => (
                <label key={value} className={`monitoring-inspection-status is-${value}`}>
                  <input
                    type="radio"
                    name="inspection-status"
                    value={value}
                    checked={inspectionStatus === value}
                    onChange={(event) => setInspectionStatus(event.target.value)}
                    disabled={inspectionSubmitting}
                  />
                  <span>{label}</span>
                </label>
              ))}
            </fieldset>

            {!inspectionStatus && (
              <div className="monitoring-form-guidance">No status is preselected. Choose only what LGU staff observed during this round.</div>
            )}

            {inspectionStatus === 'alive' && (
              <>
                <label className="monitoring-form-label" htmlFor="inspection-condition">Condition (optional)</label>
                <select
                  id="inspection-condition"
                  className="monitoring-form-select"
                  value={inspectionCondition}
                  onChange={(event) => setInspectionCondition(event.target.value)}
                  disabled={inspectionSubmitting}
                >
                  <option value="">Not recorded</option>
                  <option value="healthy">Healthy</option>
                  <option value="stressed">Stressed</option>
                  <option value="damaged">Damaged</option>
                </select>
                <label className="monitoring-form-label" htmlFor="inspection-height">Height in cm (optional)</label>
                <input
                  id="inspection-height"
                  className="monitoring-form-select"
                  type="number"
                  min="0"
                  max="5000"
                  step="0.1"
                  value={inspectionHeight}
                  onChange={(event) => setInspectionHeight(event.target.value)}
                  disabled={inspectionSubmitting}
                  placeholder="e.g. 42.5"
                />
              </>
            )}

            {inspectionStatus === 'dead' && (
              <>
                <label className="monitoring-form-label" htmlFor="inspection-death-reason">Cause of death</label>
                <select
                  id="inspection-death-reason"
                  className="monitoring-form-select"
                  value={inspectionDeathReason}
                  onChange={(event) => setInspectionDeathReason(event.target.value)}
                  disabled={inspectionSubmitting}
                >
                  <option value="">Choose a recorded cause</option>
                  {DEATH_REASONS.map((reason) => (
                    <option key={reason.key} value={reason.key}>{reason.label}</option>
                  ))}
                </select>
              </>
            )}

            <label className="monitoring-form-label" htmlFor="inspection-photo">
              Photo <span className="monitoring-form-optional">(optional, max 5 MB)</span>
            </label>
            <input
              id="inspection-photo"
              className="monitoring-inspection-file"
              type="file"
              accept="image/jpeg,image/png,image/webp"
              onChange={selectInspectionPhoto}
              disabled={inspectionSubmitting}
            />
            {inspectionPhoto && <div className="monitoring-inspection-photo-ready">Photo ready to upload</div>}

            <label className="monitoring-form-label" htmlFor="inspection-notes">
              Notes <span className="monitoring-form-optional">(optional)</span>
            </label>
            <textarea
              id="inspection-notes"
              className="monitoring-form-textarea"
              rows={3}
              maxLength={500}
              value={inspectionNotes}
              onChange={(event) => setInspectionNotes(event.target.value)}
              disabled={inspectionSubmitting}
              placeholder="Condition, nearby damage, access issues, or other field context"
            />
            <div className="monitoring-form-count">{inspectionNotes.length}/500</div>

            <label className="monitoring-form-label" htmlFor="inspection-actions">
              Actions taken <span className="monitoring-form-optional">(optional)</span>
            </label>
            <textarea
              id="inspection-actions"
              className="monitoring-form-textarea"
              rows={3}
              maxLength={1000}
              value={inspectionActionsTaken}
              onChange={(event) => setInspectionActionsTaken(event.target.value)}
              disabled={inspectionSubmitting}
              placeholder="For example: installed a guard, cleared debris, stabilized the stake, or scheduled replanting"
            />
            <div className="monitoring-form-count">{inspectionActionsTaken.length}/1000</div>
            {inspectionFormError && <div className="monitoring-error">{inspectionFormError}</div>}
          </div>
        )}
      </Modal>

      <Modal
        open={Boolean(selectedPoint)}
        title={selectedPoint
          ? (isSelectionDead
            ? `Point #${selectedPoint.point_num} — Dead`
            : `Point #${selectedPoint.point_num} — Monitoring`)
          : 'Update point status'}
        variant="info"
        confirmLabel={isSelectionDead ? 'Reset to planned' : selectedDueInspection ? 'Open due inspection' : 'Close'}
        cancelLabel={isSelectionDead || selectedDueInspection ? 'Cancel' : ''}
        busy={submitting}
        onConfirm={isSelectionDead ? submitDeadAction : selectedDueInspection ? openSelectedDueInspection : closeModal}
        onCancel={closeModal}
      >
        {selectedPoint && (
          <div className="monitoring-form">
            <div className="monitoring-form-summary">
              <div><strong>Status:</strong> {statusLabel(selectedStatus)}</div>
              <div><strong>From:</strong> {selectedPoint.image_name || 'Saved analysis'}</div>
              {selectedPoint.assigned_planter_name && (
                <div><strong>Planter:</strong> {selectedPoint.assigned_planter_name}</div>
              )}
              {selectedPoint.planted_at && (
                <div><strong>Planted:</strong> {formatDateTime(selectedPoint.planted_at)}</div>
              )}
              {isSelectionDead && (
                <div><strong>Marked dead:</strong> {formatDateTime(selectedPoint.death_at)}</div>
              )}
            </div>

            {isSelectionDead ? (
              <>
                <div className="monitoring-existing-reason">
                  <div className="monitoring-existing-reason-label">Recorded cause</div>
                  <div className="monitoring-existing-reason-text">
                    {selectedPoint.death_reason || 'No reason recorded.'}
                  </div>
                </div>

                <div className="monitoring-restore-note">
                  <strong>Reset to planned</strong> frees this location for a fresh planting cycle.
                  The death is permanent and stays in the mortality breakdown forever — resetting only
                  reopens the spot, it does <em>not</em> erase the death record.
                </div>
              </>
            ) : (
              <div className="monitoring-restore-note">
                <strong>This planting is unverified until an LGU inspection is submitted.</strong>{' '}
                {selectedDueInspection
                  ? `Its ${selectedDueInspection.interval_days}-day inspection is due. Open that round to record an observed status.`
                  : 'No scheduled inspection is due yet. A missing death record is not treated as proof that the seedling is alive.'}
              </div>
            )}

            {formError && <div className="monitoring-error" style={{ marginTop: 8 }}>{formError}</div>}
          </div>
        )}
      </Modal>

      <Modal
        open={breakdownOverlayOpen}
        title="Operational death history by assignment zone"
        variant="info"
        confirmLabel="Close"
        cancelLabel=""
        onConfirm={closeBreakdownOverlay}
      >
        {zoneDetailError && <div className="monitoring-error">{zoneDetailError}</div>}
        {zoneDetailLoading && <div className="monitoring-muted">Loading mortality detail...</div>}
        {!zoneDetailLoading && !zoneDetailError && (zoneDetail?.length ?? 0) === 0 && (
          <div className="monitoring-muted">
            No assignment zones yet. Create a planter assignment to start tracking operational records.
          </div>
        )}
        {!zoneDetailLoading && Array.isArray(zoneDetail) && zoneDetail.length > 0 && (() => {
          const safeIndex = Math.min(Math.max(zoneIndex, 0), zoneDetail.length - 1);
          const zone = zoneDetail[safeIndex];
          const mortalityDenom = zone.alive + zone.dead;
          const recordedMortalityPct = mortalityDenom > 0
            ? Math.round((zone.dead / mortalityDenom) * 100)
            : null;
          const pillTone = recordedMortalityPct === null ? 'empty' : recordedMortalityPct > 0 ? 'bad' : 'good';
          const headline = zone.title
            || (zone.planter_name ? `${zone.planter_name} — Assignment #${zone.assignment_id}` : `Assignment #${zone.assignment_id}`);
          const hasAnyActivity = zone.alive + zone.dead + zone.skipped > 0;

          // Tally causes explicitly recorded by LGU staff; environmental
          // context is never inferred as a cause here.
          const causeCounts = new Map();
          for (const p of zone.points) {
            if (!p.death_at) continue;
            const key = (p.death_reason_category || 'other').toLowerCase();
            const label = (p.death_reason || '').split(':')[0].trim()
              || (DEATH_REASONS.find((r) => r.key === key)?.label)
              || 'Other';
            const entry = causeCounts.get(key) || { key, label, count: 0 };
            entry.count += 1;
            causeCounts.set(key, entry);
          }
          const causeRows = [...causeCounts.values()].sort((a, b) => b.count - a.count);
          const totalDeaths = causeRows.reduce((sum, r) => sum + r.count, 0);

          return (
            <div className="monitoring-zone-page">
              <div className="monitoring-zone-pager">
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  onClick={() => setZoneIndex((i) => Math.max(0, i - 1))}
                  disabled={safeIndex === 0}
                >
                  ← Previous
                </button>
                <div className="monitoring-zone-pager-info">
                  Zone <strong>{safeIndex + 1}</strong> of <strong>{zoneDetail.length}</strong>
                </div>
                <button
                  type="button"
                  className="btn btn-ghost btn-sm"
                  onClick={() => setZoneIndex((i) => Math.min(zoneDetail.length - 1, i + 1))}
                  disabled={safeIndex >= zoneDetail.length - 1}
                >
                  Next →
                </button>
              </div>

              <section className="monitoring-zone-card">
                <header className="monitoring-zone-card-head">
                  <div className="monitoring-zone-card-head-main">
                    <div className="monitoring-zone-card-name">{headline}</div>
                    <div className="monitoring-zone-card-tags">
                      {zone.planter_name && (
                        <span className="monitoring-zone-tag monitoring-zone-tag-planter">
                          {zone.planter_name}
                        </span>
                      )}
                      {zone.species && (
                        <span className="monitoring-zone-tag monitoring-zone-tag-species">
                          {zone.species}
                        </span>
                      )}
                      {zone.assignment_date && (
                        <span className="monitoring-zone-tag">
                          {zone.assignment_date}
                        </span>
                      )}
                    </div>
                  </div>
                  <span className={`monitoring-survival-pill monitoring-survival-pill-${pillTone}`}>
                    <span className="monitoring-survival-pill-value">
                      {recordedMortalityPct !== null ? `${recordedMortalityPct}%` : '—'}
                    </span>
                    <span className="monitoring-survival-pill-label">
                      {recordedMortalityPct !== null ? 'recorded mortality' : 'No planted records'}
                    </span>
                  </span>
                </header>

                <div className="monitoring-zone-stats">
                  <div className="monitoring-zone-stat monitoring-zone-stat-alive">
                    <div className="monitoring-zone-stat-value">{zone.alive}</div>
                    <div className="monitoring-zone-stat-label">Planted, no death record</div>
                  </div>
                  <div className="monitoring-zone-stat monitoring-zone-stat-dead">
                    <div className="monitoring-zone-stat-value">{zone.dead}</div>
                    <div className="monitoring-zone-stat-label">Dead</div>
                  </div>
                  <div className="monitoring-zone-stat monitoring-zone-stat-skipped">
                    <div className="monitoring-zone-stat-value">{zone.skipped}</div>
                    <div className="monitoring-zone-stat-label">Skipped</div>
                  </div>
                  <div className="monitoring-zone-stat monitoring-zone-stat-pending">
                    <div className="monitoring-zone-stat-value">{zone.pending}</div>
                    <div className="monitoring-zone-stat-label">Pending</div>
                  </div>
                  <div className="monitoring-zone-stat monitoring-zone-stat-total">
                    <div className="monitoring-zone-stat-value">{zone.total}</div>
                    <div className="monitoring-zone-stat-label">Total</div>
                  </div>
                </div>

                {totalDeaths > 0 && (
                  <div className="monitoring-zone-causes">
                    <div className="monitoring-zone-causes-title">
                      Cause of death · <span className="monitoring-zone-causes-sub">{totalDeaths} death{totalDeaths === 1 ? '' : 's'}</span>
                    </div>
                    <div className="monitoring-zone-causes-list">
                      {causeRows.map((row) => {
                        const pct = totalDeaths > 0 ? (row.count / totalDeaths) * 100 : 0;
                        return (
                          <div key={row.key} className="monitoring-zone-cause-row">
                            <div className="monitoring-zone-cause-head">
                              <span className="monitoring-zone-cause-label">{row.label}</span>
                              <span className="monitoring-zone-cause-meta">
                                <strong>{row.count}</strong>
                                <span className="monitoring-zone-cause-pct">{pct.toFixed(0)}%</span>
                              </span>
                            </div>
                            <div className="monitoring-zone-cause-bar">
                              <div className="monitoring-zone-cause-fill" style={{ width: `${pct}%` }} />
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </div>
                )}

                {!hasAnyActivity ? (
                  <div className="monitoring-zone-empty">
                    <svg width="32" height="32" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.5"><circle cx="12" cy="12" r="10"/><path d="M12 8v4"/><path d="M12 16h.01"/></svg>
                    <div>This assignment has no planted, dead, or skipped points yet — still pending.</div>
                  </div>
                ) : (
                  <div className="monitoring-zone-table-wrap">
                    <table className="monitoring-zone-table">
                      <thead>
                        <tr>
                          <th className="col-point">Point</th>
                          <th className="col-status">Status</th>
                          <th className="col-planted">Planted on</th>
                          <th className="col-dead">Recorded dead</th>
                          <th className="col-cause">Cause</th>
                        </tr>
                      </thead>
                      <tbody>
                        {zone.points.map((p) => (
                          <tr key={`${zone.assignment_id}-${p.id}`}>
                            <td className="col-point">#{p.point_num ?? p.id}</td>
                            <td className="col-status">
                              <span className={`monitoring-zone-status-chip monitoring-zone-status-chip-${(p.status || '').split(' ')[0]}`}>
                                {p.status}
                              </span>
                            </td>
                            <td className="col-planted">{p.planted_at ? formatDateTime(p.planted_at) : '—'}</td>
                            <td className="col-dead">{p.death_at ? formatDateTime(p.death_at) : '—'}</td>
                            <td className="col-cause">{p.death_reason || '—'}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                )}
              </section>
            </div>
          );
        })()}
      </Modal>
    </div>
  );
}

function MonitoringProjectDetail({
  project,
  points,
  filteredPoints,
  filters,
  setFilters,
  filterOptions,
  loading,
  error,
  notice,
  asOf,
  mapPoints,
  onBack,
  onInspect,
  onViewMap,
}) {
  const mapPointIds = new Set(mapPoints.map((point) => Number(point.id)));
  return (
    <>
      <button type="button" className="monitoring-back-projects" onClick={onBack}>
        ← Back to organizations
      </button>
      <PanelCard
        title={project.name}
        badge={points.length}
        panelKey="monitoring-primary"
        icon={(
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M12 22s8-4 8-12V5l-8-3-8 3v5c0 8 8 12 8 12z" /><path d="M9 12l2 2 4-4" /></svg>
        )}
      >
        <div className="monitoring-project-detail-intro">
          <div>
            <strong>{project.planted}</strong>
            <span>
              seedling planting cycles across {project.distinctPlantedPoints} mapped {project.distinctPlantedPoints === 1 ? 'point' : 'points'}
            </span>
          </div>
          <p>Select a due round to inspect it, or open a completed observation to make a traceable correction.</p>
        </div>
        {loading && <div className="monitoring-muted">Loading project seedlings...</div>}
        {error && points.length === 0 && <div className="monitoring-error">{error}</div>}

        <div className="monitoring-project-filters">
          <label className="monitoring-project-search">
            <span>Search seedlings</span>
            <input
              type="search"
              value={filters.search}
              onChange={(event) => setFilters.search(event.target.value)}
              placeholder="Point, planter, species, assignment..."
            />
          </label>
          <label>
            <span>Latest status</span>
            <select value={filters.status} onChange={(event) => setFilters.status(event.target.value)}>
              <option value="all">All statuses</option>
              <option value="unverified">Unverified</option>
              <option value="alive">Verified alive</option>
              <option value="dead">Verified dead</option>
              <option value="missing">Missing / not found</option>
            </select>
          </label>
          <label>
            <span>Schedule</span>
            <select value={filters.schedule} onChange={(event) => setFilters.schedule(event.target.value)}>
              <option value="all">All rounds</option>
              <option value="overdue">Overdue</option>
              <option value="due">Due now</option>
              <option value="upcoming">Upcoming</option>
              <option value="completed">Completed rounds only</option>
              <option value="not_applicable">Not applicable</option>
              <option value="none">No scheduled round</option>
            </select>
          </label>
          <label>
            <span>Planter</span>
            <select value={filters.planter} onChange={(event) => setFilters.planter(event.target.value)}>
              <option value="all">All planters</option>
              {filterOptions.planters.map((name) => <option key={name} value={name}>{name}</option>)}
            </select>
          </label>
          <label>
            <span>Species</span>
            <select value={filters.species} onChange={(event) => setFilters.species(event.target.value)}>
              <option value="all">All species</option>
              {filterOptions.species.map((species) => <option key={species} value={species}>{species}</option>)}
            </select>
          </label>
        </div>

        <div className="monitoring-project-result-count">
          Showing <strong>{filteredPoints.length}</strong> of <strong>{points.length}</strong> seedling planting cycles
        </div>

        {!loading && points.length === 0 && (
          <div className="monitoring-project-empty">
            <strong>No planted seedlings are recorded for this project.</strong>
            <span>Assigned or planned candidates do not enter ecological monitoring until a planting event exists.</span>
          </div>
        )}
        {points.length > 0 && filteredPoints.length === 0 && (
          <div className="monitoring-project-empty">
            <strong>No seedlings match these filters.</strong>
            <span>Clear the search or choose broader status, schedule, planter, and species filters.</span>
          </div>
        )}

        <div className="monitoring-seedling-list">
          {filteredPoints.map((point) => (
            <MonitoringSeedlingCard
              key={point.key}
              point={point}
              asOf={asOf}
              canViewMap={mapPointIds.has(Number(point.planting_point_id))}
              onInspect={onInspect}
              onViewMap={onViewMap}
            />
          ))}
        </div>
        {notice && <div className="monitoring-notice">{notice}</div>}
      </PanelCard>
    </>
  );
}

function MonitoringSeedlingCard({ point, asOf, canViewMap, onInspect, onViewMap }) {
  return (
    <details className={`monitoring-seedling-card is-${point.latestStatus}`}>
      <summary>
        <span className="monitoring-seedling-title">
          <strong>Point #{point.point_num}</strong>
          <small>
            {point.assignment_title || 'Planting event'}
            {point.planting_event_id ? ` · cycle #${point.planting_event_id}` : ''}
          </small>
        </span>
        <span className={`monitoring-seedling-status is-${point.latestStatus}`}>
          {point.latestStatus === 'unverified'
            ? 'Unverified'
            : point.latestStatus === 'missing'
              ? 'Missing / not found'
              : `Observed ${statusLabel(point.latestStatus).toLowerCase()}`}
        </span>
      </summary>
      <div className="monitoring-seedling-body">
        <div className="monitoring-seedling-facts">
          <div><span>Planted</span><strong>{formatDateTime(point.planted_at)}</strong></div>
          <div><span>Planter</span><strong>{point.planter_name || 'Not recorded'}</strong></div>
          <div><span>Species</span><strong>{point.species || 'Not recorded'}</strong></div>
          <div><span>Planting cycle</span><strong>{point.closed_at ? `Closed · ${humanizeCode(point.closure_reason) || formatDateTime(point.closed_at)}` : 'Current'}</strong></div>
          <div>
            <span>Latest measured height</span>
            <strong>
              {formatHeight(point.latestHeightObservation?.height_cm)}
              {point.latestHeightObservation ? ` · ${point.latestHeightObservation.interval_days}-day` : ''}
            </strong>
          </div>
        </div>

        <div className="monitoring-seedling-latest">
          <div className="monitoring-seedling-section-title">Latest LGU observation</div>
          {point.latest ? (
            <div className="monitoring-seedling-latest-grid">
              <div><span>Round</span><strong>{point.latest.interval_days}-day</strong></div>
              <div><span>Inspected</span><strong>{formatDateTime(point.latest.inspected_at)}</strong></div>
              <div><span>Condition</span><strong>{point.latest.condition || 'Not recorded'}</strong></div>
              <div><span>Cause of death</span><strong>{point.latest.status === 'dead' ? deathReasonLabel(point.latest.death_reason_category) : 'Not applicable'}</strong></div>
              <div className="is-wide"><span>Notes</span><strong>{point.latest.notes || 'No notes recorded'}</strong></div>
              <div className="is-wide"><span>Actions taken</span><strong>{point.latest.actions_taken || 'No actions recorded'}</strong></div>
            </div>
          ) : (
            <div className="monitoring-muted">No LGU observation yet. This seedling remains unverified.</div>
          )}
        </div>

        <div className="monitoring-seedling-schedule">
          <div className="monitoring-seedling-section-title">Scheduled follow-up rounds</div>
          {point.rounds.length === 0 ? (
            <div className="monitoring-muted">No outstanding or upcoming round was returned for this planting event.</div>
          ) : (
            <div className="monitoring-round-list">
              {point.rounds.map((round) => {
                const state = roundState(round, asOf);
                const canOpen = ['due', 'overdue', 'completed'].includes(state);
                return (
                  <div key={`${point.key}-${round.interval_days}`} className={`monitoring-round-row is-${state}`}>
                    <span>
                      <strong>{inspectionRoundLabel(round)}</strong>
                      <small>
                        {state === 'not_applicable'
                          ? (humanizeCode(round.not_applicable_reason || round.reason || round.state_reason) || 'Not applicable to this planting cycle')
                          : state === 'completed'
                            ? `Completed ${formatDateTime(round.inspected_at)}`
                            : `${inspectionCountdown(round, asOf)} · field ${formatDateTime(roundFieldAt(round))}`}
                      </small>
                      {state !== 'not_applicable' ? <small>Exact target {formatDateTime(roundTargetAt(round))}</small> : null}
                    </span>
                    <button
                      type="button"
                      className="btn btn-secondary btn-sm"
                      onClick={() => canOpen && onInspect(round)}
                      disabled={!canOpen}
                    >
                      {state === 'completed'
                        ? 'Correct'
                        : state === 'upcoming'
                          ? 'Not due yet'
                          : state === 'not_applicable'
                            ? 'Not applicable'
                            : 'Inspect'}
                    </button>
                  </div>
                );
              })}
            </div>
          )}
        </div>

        <MonitoringObservationHistory point={point} onInspect={onInspect} />

        {canViewMap && (
          <button
            type="button"
            className="btn btn-ghost btn-sm monitoring-view-map"
            onClick={() => onViewMap(point.planting_point_id)}
          >
            View location on map
          </button>
        )}
      </div>
    </details>
  );
}

function MonitoringObservationHistory({ point, onInspect }) {
  return (
    <div className="monitoring-seedling-history">
      <div className="monitoring-seedling-section-title">Height, growth, and observation history</div>
      {point.history.length === 0 ? (
        <div className="monitoring-muted">No completed observations yet.</div>
      ) : (
        <div className="monitoring-history-list">
          {point.history.map((observation, observationIndex) => {
            const previousHeightObservation = [...point.history.slice(0, observationIndex)]
              .reverse()
              .find((entry) => hasMeasuredHeight(entry.height_cm));
            const currentHeight = hasMeasuredHeight(observation.height_cm) ? Number(observation.height_cm) : null;
            const previousHeight = hasMeasuredHeight(previousHeightObservation?.height_cm)
              ? Number(previousHeightObservation.height_cm)
              : null;
            const growth = currentHeight !== null && previousHeight !== null
              ? currentHeight - previousHeight
              : null;
            return (
              <div key={observation.id ?? `${point.key}-${observation.interval_days}`} className={`monitoring-history-row is-${observationStatus(observation.status)}`}>
                <div className="monitoring-history-head">
                  <span><strong>{observation.interval_days}-day</strong> · {statusLabel(observation.status)}</span>
                  <button type="button" className="btn btn-ghost btn-sm" onClick={() => onInspect(observation)}>Correct</button>
                </div>
                <div className="monitoring-history-metrics">
                  <span>Height <strong>{formatHeight(observation.height_cm)}</strong></span>
                  <span>Growth <strong>{growth === null ? '—' : `${growth >= 0 ? '+' : ''}${growth.toFixed(1)} cm`}</strong></span>
                  <span>Condition <strong>{observation.condition || '—'}</strong></span>
                  <span>Cause <strong>{observation.status === 'dead' ? deathReasonLabel(observation.death_reason_category) : '—'}</strong></span>
                </div>
                {(observation.notes || observation.actions_taken) && (
                  <div className="monitoring-history-notes">
                    {observation.notes && <span><strong>Notes:</strong> {observation.notes}</span>}
                    {observation.actions_taken && <span><strong>Actions:</strong> {observation.actions_taken}</span>}
                  </div>
                )}
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
}

function MonitoringProjects({
  projects,
  loading,
  projectError,
  inspectionError,
  observationsError,
  inspectionSummary,
  notice,
  onOpen,
}) {
  return (
    <>
      <PanelCard
        title="How project monitoring works"
        defaultOpen={false}
        icon={(
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M3 21h18" /><path d="M5 21V7l7-4 7 4v14" /><path d="M9 21v-6h6v6" /></svg>
        )}
      >
        <p className="monitoring-hint">
          Choose an organization first, then review its planted seedlings and scheduled rounds across
          every project site it owns.
          Only completed <strong>LGU observations</strong> establish recorded outcomes. Alive and dead
          contribute to verified survival; missing/not-found is reported separately. Newly planted and
          uninspected seedlings remain unverified.
        </p>
        <div className="monitoring-project-global-summary" aria-label="Inspection schedule summary">
          <div><strong>{inspectionSummary?.due ?? 0}</strong><span>Due total</span></div>
          <div><strong>{inspectionSummary?.overdue ?? 0}</strong><span>Overdue subset</span></div>
          <div><strong>{inspectionSummary?.upcoming ?? 0}</strong><span>Upcoming</span></div>
        </div>
      </PanelCard>

      <PanelCard
        title="Organizations"
        badge={projects.length}
        panelKey="monitoring-primary"
        icon={(
          <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2"><path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" /><circle cx="12" cy="10" r="3" /></svg>
        )}
      >
        {loading && <div className="monitoring-muted">Loading project monitoring...</div>}
        {projectError && projects.length === 0 && <div className="monitoring-error">{projectError}</div>}
        {inspectionError && <div className="monitoring-error">{inspectionError}</div>}
        {observationsError && <div className="monitoring-error">{observationsError}</div>}
        {!loading && projects.length === 0 && !inspectionError && (
          <div className="monitoring-project-empty">
            <strong>No organizations have planted seedlings to monitor yet.</strong>
            <span>Create an organization-owned project site, link an assignment to it, and record field planting before monitoring begins.</span>
          </div>
        )}
        {projects.length > 0 && (
          <div className="monitoring-project-grid">
            {projects.map((project) => {
              return (
                <button
                  type="button"
                  key={project.key}
                  className="monitoring-project-card"
                  onClick={() => onOpen(project)}
                >
                  <span className="monitoring-project-card-head">
                    <span>
                      <strong>{project.name}</strong>
                      <small>
                        {project.distinctPlantedPoints} planted {project.distinctPlantedPoints === 1 ? 'point' : 'points'}
                      </small>
                    </span>
                    <span className="monitoring-project-open">View details →</span>
                  </span>
                </button>
              );
            })}
          </div>
        )}
        {notice && <div className="monitoring-notice">{notice}</div>}
      </PanelCard>
    </>
  );
}

function StatusPill({ color, label, value }) {
  return (
    <div className="monitoring-status-pill">
      <span className="monitoring-status-swatch" style={{ background: color }} aria-hidden="true" />
      <span className="monitoring-status-label">{label}</span>
      <span className="monitoring-status-value">{value}</span>
    </div>
  );
}
