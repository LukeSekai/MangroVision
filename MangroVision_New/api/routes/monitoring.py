"""LGU-authenticated planting inspection endpoints."""

from typing import Optional

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, ConfigDict, Field
from mangrovision_db.monitoring_locations import monitoring_points, reconcile_locations, list_replanting, approve_replanting, approve_replanting_batch, assign_replanting

from planting_database import (
    DEATH_REASON_CATEGORIES,
    get_organization_visit_context,
    record_organization_monitoring_visit,
    get_due_monitoring_inspections,
    get_monitoring_project_detail,
    get_user_by_session_token,
    list_monitoring_projects,
    list_monitoring_observations,
    list_organization_monitoring_records,
    list_organization_monitoring_summaries,
    upsert_monitoring_observation,
)

router = APIRouter()


def _require_lgu_user() -> dict:
    # A user session (rather than a planter session) is the permission
    # boundary for scientific monitoring. Existing deployments call this
    # role "planner"; it represents authenticated LGU staff in v2.
    user = get_user_by_session_token("")
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired LGU session.")
    if str(user.get("role") or "").lower() not in {"admin", "lgu", "planner"}:
        raise HTTPException(status_code=403, detail="LGU/admin permission is required.")
    return user


def _observation_error(error: ValueError) -> HTTPException:
    detail = str(error)
    lower = detail.lower()
    if "not found" in lower:
        status_code = 404
    elif any(
        phrase in lower
        for phrase in (
            "not due yet",
            "cycle is closed",
            "current planting cycle",
            "currently planted point",
            "removed planting point",
            "separate recorded death",
        )
    ):
        status_code = 409
    else:
        status_code = 400
    return HTTPException(status_code=status_code, detail=detail)


class MonitoringObservationCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    planting_event_id: int
    interval_days: int
    status: str
    condition: Optional[str] = None
    height_cm: Optional[float] = None
    notes: Optional[str] = None
    actions_taken: Optional[str] = None
    death_reason_category: Optional[str] = None
    photo_data_url: Optional[str] = None


class OrganizationMonitoringCreate(BaseModel):
    """One aggregate LGU monitoring visit for an entire organization."""

    model_config = ConfigDict(extra="forbid")

    organization_id: int
    monitored_at: Optional[str] = None
    new_dead_count: int
    baseline_record_id: Optional[int]
    expected_alive_count: int
    health_status: str
    actions_taken: str
    dead_planting_event_ids: list[int] = Field(default_factory=list)
    unlocated_dead_count: Optional[int] = Field(default=None, ge=0)
    death_reason_category: Optional[str] = None
    death_reason_notes: str = Field(default='', max_length=500)


@router.get("/organizations")
def monitoring_organizations():
    """Include historical cohort totals separately from current mapped ownership."""
    _require_lgu_user()
    return {
        "organizations": list_organization_monitoring_summaries(),
        "health_options": ["excellent", "good", "fair", "poor", "critical"],
        "monitoring_unit": "organization",
        "death_reason_options": [{'value': key, 'label': label} for key, label in DEATH_REASON_CATEGORIES.items()],
    }


@router.get("/organization-records")
def organization_monitoring_records(
    organization_id: Optional[int] = None,
    limit: int = Query(default=100, ge=1, le=1000),
    before_id: Optional[int] = Query(default=None, ge=1),
):
    _require_lgu_user()
    try:
        records = list_organization_monitoring_records(
            organization_id=organization_id,
            limit=limit,
            before_id=before_id,
        )
        return {
            "records": records,
            "next_before_id": records[-1]["id"] if len(records) == limit else None,
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/organization-records", status_code=201)
def submit_organization_monitoring_record(
    body: OrganizationMonitoringCreate,
):
    user = _require_lgu_user()
    try:
        return record_organization_monitoring_visit(
            organization_id=body.organization_id,
            monitored_at=body.monitored_at,
            new_dead_count=body.new_dead_count,
            baseline_record_id=body.baseline_record_id,
            expected_alive_count=body.expected_alive_count,
            health_status=body.health_status,
            actions_taken=body.actions_taken,
            inspector_user_id=int(user["id"]),
            dead_planting_event_ids=body.dead_planting_event_ids,
            unlocated_dead_count=body.unlocated_dead_count,
            death_reason_category=body.death_reason_category,
            death_reason_notes=body.death_reason_notes,
        )
    except ValueError as error:
        detail = str(error)
        status_code = 404 if "not found" in detail.lower() else 409 if (
            'Monitoring changed' in detail or 'Not time to monitor yet' in detail
        ) else 400
        raise HTTPException(status_code=status_code, detail=detail) from error


@router.get('/organizations/{organization_id}/visit-context')
def organization_visit_context(organization_id: int, monitored_at: Optional[str] = None):
    _require_lgu_user()
    try:
        return get_organization_visit_context(organization_id, monitored_at)
    except ValueError as error:
        raise HTTPException(status_code=404 if 'not found' in str(error) else 400, detail=str(error)) from error


@router.get("/due")
def due_monitoring_inspections(
    as_of: Optional[str] = None,
    site_id: Optional[int] = None,
    project_site_id: Optional[int] = None,
    assignment_id: Optional[int] = None,
    species: Optional[str] = None,
    planter_id: Optional[int] = None,
    upcoming_days: int = Query(default=30, ge=0, le=365),
):
    _require_lgu_user()
    try:
        return get_due_monitoring_inspections(
            as_of=as_of,
            site_id=site_id,
            project_site_id=project_site_id,
            assignment_id=assignment_id,
            species=species,
            planter_id=planter_id,
            upcoming_days=upcoming_days,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


class DeathLocationsUpdate(BaseModel):
    model_config = ConfigDict(extra='forbid')
    dead_planting_event_ids: list[int]
    expected_version: int = Field(ge=0)
    correction_note: str = Field(default='', max_length=2000)


class ReplantingReview(BaseModel):
    expected_version: int = Field(ge=0)


class ReplantingAssignment(BaseModel):
    model_config = ConfigDict(extra='forbid')
    planting_event_ids: list[int]
    versions: dict[str, int]
    organization_id: int = Field(gt=0)
    assignment_date: str


class ReplantingBatchReview(BaseModel):
    model_config = ConfigDict(extra='forbid')
    planting_event_ids: list[int] = Field(min_length=1, max_length=500)
    versions: dict[str, int]


@router.get('/organizations/{organization_id}/planting-locations')
def organization_planting_locations(organization_id: int, monitored_at: Optional[str] = None, record_id: Optional[int] = None):
    _require_lgu_user()
    try:
        return {'points': monitoring_points(organization_id, monitored_at, record_id)}
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.put('/organization-records/{record_id}/death-locations')
def update_death_locations(record_id: int, body: DeathLocationsUpdate):
    user = _require_lgu_user()
    try:
        return reconcile_locations(record_id, body.dead_planting_event_ids, body.expected_version,
                                   int(user['id']), body.correction_note)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get('/replanting')
def replanting_candidates(organization_id: Optional[int] = None):
    _require_lgu_user()
    return {'points': list_replanting(organization_id)}


@router.post('/replanting/{event_id}/approve')
def review_replanting(event_id: int, body: ReplantingReview):
    user = _require_lgu_user()
    try:
        return approve_replanting(event_id, body.expected_version, int(user['id']))
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post('/replanting/approve')
def review_replanting_batch(body: ReplantingBatchReview):
    user = _require_lgu_user()
    try:
        return approve_replanting_batch(body.planting_event_ids, body.versions, int(user['id']))
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.post('/replanting/assign')
def create_replanting_assignment(body: ReplantingAssignment):
    user = _require_lgu_user()
    from api.runtime_state import is_processing_active
    if is_processing_active():
        raise HTTPException(status_code=409, detail='Wait for image processing to finish before assigning.')
    try:
        return assign_replanting(body.planting_event_ids, body.versions, body.organization_id,
                                body.assignment_date, int(user['id']))
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error


@router.get("/observations")
def monitoring_observations(
    planting_event_id: Optional[int] = None,
    site_id: Optional[int] = None,
    project_site_id: Optional[int] = None,
    assignment_id: Optional[int] = None,
    species: Optional[str] = None,
    planter_id: Optional[int] = None,
    observation_status: Optional[str] = Query(default=None, alias="status"),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    limit: int = Query(default=100, ge=1, le=1000),
):
    _require_lgu_user()
    try:
        return {
            "observations": list_monitoring_observations(
                planting_event_id=planting_event_id,
                site_id=site_id,
                project_site_id=project_site_id,
                assignment_id=assignment_id,
                species=species,
                planter_id=planter_id,
                status=observation_status,
                date_from=date_from,
                date_to=date_to,
                limit=limit,
            )
        }
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/observations")
def submit_monitoring_observation(
    body: MonitoringObservationCreate,
):
    user = _require_lgu_user()
    try:
        return upsert_monitoring_observation(
            planting_event_id=body.planting_event_id,
            interval_days=body.interval_days,
            status=body.status,
            inspector_user_id=int(user["id"]),
            condition=body.condition,
            height_cm=body.height_cm,
            notes=body.notes,
            actions_taken=body.actions_taken,
            death_reason_category=body.death_reason_category,
            photo_data_url=body.photo_data_url,
        )
    except ValueError as error:
        raise _observation_error(error) from error


@router.get("/projects")
def monitoring_projects(
    as_of: Optional[str] = None,
    upcoming_days: int = Query(default=30, ge=0, le=365),
):
    _require_lgu_user()
    try:
        return list_monitoring_projects(as_of=as_of, upcoming_days=upcoming_days)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/projects/{project_site_id}")
def monitoring_project_detail(
    project_site_id: int,
    as_of: Optional[str] = None,
    upcoming_days: int = Query(default=30, ge=0, le=365),
):
    _require_lgu_user()
    try:
        result = get_monitoring_project_detail(
            project_site_id=project_site_id,
            as_of=as_of,
            upcoming_days=upcoming_days,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if result is None:
        raise HTTPException(status_code=404, detail="Project site not found.")
    return result
