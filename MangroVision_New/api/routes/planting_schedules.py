"""Authenticated LGU planting-activity schedule endpoints."""

from typing import Any, Optional

from fastapi import APIRouter, BackgroundTasks, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict
from mangrovision_db.activity_rules import ActivityConflict
from mangrovision_db.appointment_email import run_delivery_cycle

from planting_database import (
    create_planting_schedule,
    delete_planting_schedule,
    get_planting_schedule,
    get_user_by_session_token,
    list_organizations,
    list_planting_schedules,
    list_project_sites,
    update_planting_schedule,
)


router = APIRouter()


def _require_lgu_user() -> dict:
    user = get_user_by_session_token("")
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired LGU session.")
    if str(user.get("role") or "").lower() not in {"admin", "lgu", "planner"}:
        raise HTTPException(status_code=403, detail="LGU/admin permission is required.")
    return user


def _schedule_error(error: ValueError) -> HTTPException:
    detail = str(error)
    return HTTPException(
        status_code=(
            404 if "not found" in detail.lower()
            else 409 if isinstance(error, ActivityConflict) or "cadence conflict" in detail.lower()
            else 400
        ),
        detail=detail,
    )


class PlantingScheduleCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    organization: str
    organization_id: Optional[int] = None
    inspection_interval_days: int
    contact: Optional[str] = None
    title: str
    start_at: Optional[str] = None
    end_at: Optional[str] = None
    date: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    expected_planters: Optional[int] = None
    expected_participants: Optional[int] = None
    expected_seedlings: Optional[int] = None
    seedlings: Optional[int] = None
    status: str = "requested"
    notes: Optional[str] = None


class PlantingScheduleUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    project_site_id: Optional[int] = None
    organization: Optional[str] = None
    organization_id: Optional[int] = None
    inspection_interval_days: Optional[int] = None
    contact: Optional[str] = None
    title: Optional[str] = None
    start_at: Optional[str] = None
    end_at: Optional[str] = None
    date: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    expected_planters: Optional[int] = None
    expected_participants: Optional[int] = None
    expected_seedlings: Optional[int] = None
    seedlings: Optional[int] = None
    status: Optional[str] = None
    notes: Optional[str] = None


def _calendar_timestamp(day: Any, clock: Any, field: str) -> str:
    clean_day = str(day or "").strip()
    clean_clock = str(clock or "").strip()
    if not clean_day or not clean_clock:
        raise ValueError(f"date and {field} are required when timestamps are not supplied.")
    return f"{clean_day}T{clean_clock}"


def _create_values(body: PlantingScheduleCreate) -> dict:
    values = body.model_dump()
    start_at = values["start_at"] or _calendar_timestamp(
        values["date"], values["start_time"], "start_time",
    )
    end_at = values["end_at"] or _calendar_timestamp(
        values["date"], values["end_time"], "end_time",
    )
    expected_planters = (
        values["expected_planters"]
        if values["expected_planters"] is not None
        else values["expected_participants"]
    )
    expected_seedlings = (
        values["expected_seedlings"]
        if values["expected_seedlings"] is not None
        else values["seedlings"]
    )
    return {
        "organization": values["organization"],
        "organization_id": values["organization_id"],
        "inspection_interval_days": values["inspection_interval_days"],
        "contact": values["contact"],
        "title": values["title"],
        "start_at": start_at,
        "end_at": end_at,
        "expected_planters": expected_planters,
        "expected_seedlings": expected_seedlings,
        "status": values["status"],
        "notes": values["notes"],
    }


def _update_values(schedule_id: int, body: PlantingScheduleUpdate) -> dict:
    values = body.model_dump(exclude_unset=True)
    current = get_planting_schedule(schedule_id)
    if current is None:
        raise HTTPException(status_code=404, detail="Planting schedule not found.")

    updates = {
        key: value for key, value in values.items()
        if key not in {
            "date", "start_time", "end_time", "expected_participants", "seedlings",
        }
    }
    if any(key in values for key in ("date", "start_time", "end_time")):
        day = values.get("date", current["date"])
        if "start_at" not in values:
            updates["start_at"] = _calendar_timestamp(
                day, values.get("start_time", current["start_time"]), "start_time",
            )
        if "end_at" not in values:
            updates["end_at"] = _calendar_timestamp(
                day, values.get("end_time", current["end_time"]), "end_time",
            )
    if "expected_planters" not in values and "expected_participants" in values:
        updates["expected_planters"] = values["expected_participants"]
    if "expected_seedlings" not in values and "seedlings" in values:
        updates["expected_seedlings"] = values["seedlings"]
    return updates


def _project_site_options() -> list[dict]:
    options = []
    for feature in list_project_sites():
        props = feature.get("properties") or {}
        site_id = feature.get("id") or props.get("id")
        options.append({
            "id": int(site_id),
            "project_site_id": int(site_id),
            "name": props.get("name") or f"Project Site {site_id}",
            "organization_id": props.get("organization_id"),
            "organization": props.get("organization_name") or props.get("organization"),
            "organization_name": props.get("organization_name") or props.get("organization"),
            "inspection_interval_days": props.get("inspection_interval_days"),
            "notes": props.get("notes"),
            "centroid_lat": props.get("centroid_lat"),
            "centroid_lon": props.get("centroid_lon"),
            "tide_calibration": props.get("tide_calibration"),
        })
    return options


@router.get("")
@router.get("/", include_in_schema=False)
def planting_schedules(
    project_site_id: Optional[int] = None,
    schedule_status: Optional[str] = Query(default=None, alias="status"),
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
):
    _require_lgu_user()
    try:
        return {
            "timezone": "Asia/Manila",
            "schedules": list_planting_schedules(
                project_site_id=project_site_id,
                status=schedule_status,
                date_from=date_from,
                date_to=date_to,
            ),
            "project_sites": _project_site_options(),
            "organizations": list_organizations(),
        }
    except ValueError as error:
        raise _schedule_error(error) from error


@router.get("/{schedule_id}")
def planting_schedule(schedule_id: int):
    _require_lgu_user()
    result = get_planting_schedule(schedule_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Planting schedule not found.")
    return result


@router.post("", status_code=status.HTTP_201_CREATED)
@router.post("/", status_code=status.HTTP_201_CREATED, include_in_schema=False)
def add_planting_schedule(body: PlantingScheduleCreate):
    user = _require_lgu_user()
    try:
        return create_planting_schedule(
            **_create_values(body), created_by_user_id=int(user["id"]),
        )
    except ValueError as error:
        raise _schedule_error(error) from error


def _edit_planting_schedule(
    schedule_id: int,
    body: PlantingScheduleUpdate,
    background_tasks: BackgroundTasks,
):
    user = _require_lgu_user()
    try:
        result = update_planting_schedule(
            schedule_id,
            **_update_values(schedule_id, body),
            updated_by_user_id=int(user["id"]),
        )
    except ValueError as error:
        raise _schedule_error(error) from error
    if result is None:
        raise HTTPException(status_code=404, detail="Planting schedule not found.")
    if result.get('email_update'):
        background_tasks.add_task(run_delivery_cycle, request_id=result['email_update']['request_id'])
    return result


@router.put("/{schedule_id}")
def replace_planting_schedule(
    schedule_id: int,
    body: PlantingScheduleUpdate,
    background_tasks: BackgroundTasks,
):
    return _edit_planting_schedule(schedule_id, body, background_tasks)


@router.patch("/{schedule_id}")
def edit_planting_schedule(
    schedule_id: int,
    body: PlantingScheduleUpdate,
    background_tasks: BackgroundTasks,
):
    return _edit_planting_schedule(schedule_id, body, background_tasks)


@router.delete("/{schedule_id}")
def remove_planting_schedule(schedule_id: int):
    user = _require_lgu_user()
    try:
        removed = delete_planting_schedule(schedule_id, deleted_by_user_id=int(user['id']))
    except ValueError as error:
        raise _schedule_error(error) from error
    if not removed:
        raise HTTPException(status_code=404, detail="Planting schedule not found.")
    return {"status": "deleted", "id": schedule_id}
