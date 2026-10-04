"""Planter management endpoints."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Depends
from fastapi.responses import JSONResponse
from shapely.geometry import Point
from shapely.prepared import prep
from api.routes.monitoring import _require_lgu_user
from pydantic import BaseModel, Field
from mangrovision_db.organization_accounts import reset_participant_device

from api.runtime_state import is_processing_active
from planting_database import (
    _get_connection,
    _hash_password,
    create_planter,
    create_planter_assignment,
    create_organization_assignment,
    get_mortality_detail_table,
    get_mortality_stats,
    reset_planting_point_to_planned,
    get_planter,
    get_planter_dashboard_stats,
    get_planter_field_points,
    list_planter_assignment_map_points,
    list_planters,
    mark_planting_point_dead,
    restore_planting_point_to_planted,
)

router = APIRouter(dependencies=[Depends(_require_lgu_user)])


class AssignPointRequest(BaseModel):
    planter_id: int
    planting_point_id: int
    allow_reassign: bool = False
    travel_mode: str = "walking"


class CreatePlanterRequest(BaseModel):
    full_name: str = "Organization account"
    organization_id: int
    participant_count: int = Field(default=1, ge=1, le=10000)
    username: str
    password: str
    phone: str = ""
    base_label: str = ""
    base_lat: Optional[float] = None
    base_lon: Optional[float] = None
    notes: str = ""
    status: str = "active"


class UpdatePlanterRequest(BaseModel):
    full_name: Optional[str] = None
    phone: Optional[str] = None
    base_label: Optional[str] = None
    base_lat: Optional[float] = None
    base_lon: Optional[float] = None
    notes: Optional[str] = None
    status: Optional[str] = None
    password: Optional[str] = None


class CreateAssignmentRequest(BaseModel):
    planting_point_ids: List[int]
    assigned_by_user_id: Optional[int] = None
    title: str = ""
    assignment_date: str = ""
    travel_mode: str = "walking"
    notes: str = ""
    species: str = ""
    site_zone_id: Optional[int] = None


class MarkDeadRequest(BaseModel):
    reason_category: str
    notes: str = ""


@router.post("/organizations/{organization_id}/assignments")
def create_organization_assignment_endpoint(organization_id: int, body: CreateAssignmentRequest,
                                           user: dict = Depends(_require_lgu_user)):
    _raise_if_processing_active()
    try:
        assignment_id = create_organization_assignment(
            organization_id, body.planting_point_ids,
            assigned_by_user_id=int(user["id"]), title=body.title,
            assignment_date=body.assignment_date, travel_mode=body.travel_mode,
            notes=body.notes, species=body.species, site_zone_id=body.site_zone_id,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"assignment_id": assignment_id}


def _raise_if_processing_active():
    if is_processing_active():
        raise HTTPException(
            status_code=409,
            detail="Planting point assignment is locked while image processing is running.",
        )


@router.get("/")
def get_planters(include_inactive: bool = True):
    return list_planters(include_inactive=include_inactive)


@router.get("/dashboard")
def dashboard_stats():
    return get_planter_dashboard_stats()


@router.get("/mortality-stats")
def mortality_stats():
    return get_mortality_stats()


@router.get("/mortality-detail")
def mortality_detail():
    """Per-zone (per-assignment) detailed table for the Monitoring overlay."""
    return {"zones": get_mortality_detail_table()}


@router.patch("/map-points/{point_id}/death")
def mark_point_dead(point_id: int, body: MarkDeadRequest):
    try:
        return mark_planting_point_dead(
            point_id=point_id,
            reason_category=body.reason_category,
            notes=body.notes,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/map-points/{point_id}/restore")
def restore_point_to_planted(point_id: int):
    try:
        return restore_planting_point_to_planted(point_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.post("/map-points/{point_id}/reset-to-planned")
def reset_point_to_planned(point_id: int):
    try:
        return reset_planting_point_to_planned(point_id)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/map-points")
def map_points():
    # The domain adapter already normalizes dates, decimals and JSON values.
    # Avoid FastAPI recursively normalizing every field of thousands of points
    # again; JSONResponse retains the same JSON number precision and nulls.
    points = list_planter_assignment_map_points()
    # Preserve historical records in the API and database. The map hides
    # coordinates outside the same visible footprint used for new analyses.
    from api.routes.processing import _load_gis_coverage_geometry

    visible_map = prep(_load_gis_coverage_geometry())
    for point in points:
        try:
            point["inside_visible_map"] = visible_map.covers(
                Point(float(point["longitude"]), float(point["latitude"]))
            )
        except (KeyError, TypeError, ValueError):
            point["inside_visible_map"] = False
    return JSONResponse(points)


@router.post("/assign-point")
def assign_point(body: AssignPointRequest):
    raise HTTPException(status_code=410, detail="Individual assignment has been removed. Assign a batch to the organization.")


# MIGRATED FROM app.py planter management tab (create form)
@router.post("/")
def create_planter_endpoint(body: CreatePlanterRequest, user: dict = Depends(_require_lgu_user)):
    try:
        planter_id = create_planter(
            full_name=body.full_name,
            username=body.username,
            password=body.password,
            phone=body.phone,
            base_label=body.base_label,
            base_lat=body.base_lat,
            base_lon=body.base_lon,
            notes=body.notes,
            status=body.status,
            organization_id=body.organization_id,
            participant_count=body.participant_count,
            created_by_user_id=int(user["id"]),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return get_planter(planter_id)


@router.get("/{planter_id}")
def get_planter_endpoint(planter_id: int):
    planter = get_planter(planter_id)
    if not planter:
        raise HTTPException(status_code=404, detail="Planter not found")
    return planter


# MIGRATED FROM app.py planter edit flow
@router.patch("/{planter_id}")
def update_planter_endpoint(planter_id: int, body: UpdatePlanterRequest):
    planter = get_planter(planter_id)
    if not planter:
        raise HTTPException(status_code=404, detail="Planter not found")

    if planter.get("merged_into_planter_id") is not None:
        raise HTTPException(status_code=409, detail="This legacy account was merged into its organization login.")
    updates: list[tuple[str, object]] = []
    if body.full_name is not None and planter.get("organization_id") is None:
        full_name = body.full_name.strip()
        if not full_name:
            raise HTTPException(status_code=400, detail="Planter name cannot be empty.")
        updates.append(("full_name", full_name))
    if body.phone is not None:
        updates.append(("phone", body.phone.strip() or None))
    if body.base_label is not None:
        updates.append(("base_label", body.base_label.strip() or None))
    if body.base_lat is not None:
        updates.append(("base_lat", body.base_lat))
    if body.base_lon is not None:
        updates.append(("base_lon", body.base_lon))
    if body.notes is not None:
        updates.append(("notes", body.notes.strip() or None))
    if body.status is not None:
        status = body.status.strip().lower()
        if status not in {"active", "inactive"}:
            raise HTTPException(status_code=400, detail="Invalid planter status.")
        updates.append(("status", status))
    if body.password is not None and body.password:
        updates.append(("password_hash", _hash_password(body.password)))

    if not updates:
        return planter

    set_clause = ", ".join(f"{column} = ?" for column, _ in updates)
    values = [value for _, value in updates] + [planter_id]

    conn = _get_connection()
    conn.execute(f"UPDATE planters SET {set_clause} WHERE id = ?", values)
    conn.commit()
    conn.close()
    return get_planter(planter_id)


# MIGRATED FROM app.py soft-delete planter flow
@router.delete("/{planter_id}")
def deactivate_planter_endpoint(planter_id: int, hard: bool = False):
    planter = get_planter(planter_id)
    if not planter:
        raise HTTPException(status_code=404, detail="Planter not found")

    conn = _get_connection()
    if hard:
        conn.execute("DELETE FROM planters WHERE id = ?", (planter_id,))
    else:
        conn.execute("UPDATE planters SET status = 'inactive' WHERE id = ?", (planter_id,))
    conn.commit()
    conn.close()
    return {"status": "deleted" if hard else "deactivated", "id": planter_id}


# MIGRATED FROM app.py batch assignment flow (create_planter_assignment)
@router.post("/{planter_id}/assignments")
def create_assignment_endpoint(planter_id: int, body: CreateAssignmentRequest,
                               user: dict = Depends(_require_lgu_user)):
    """Divide selected point locations into balanced participant zigzag strips."""
    _raise_if_processing_active()
    planter = get_planter(planter_id)
    if not planter:
        raise HTTPException(status_code=404, detail="Planter not found")
    try:
        assignment_id = create_planter_assignment(
            planter_id=planter_id,
            planting_point_ids=body.planting_point_ids,
            assigned_by_user_id=int(user["id"]),
            title=body.title,
            assignment_date=body.assignment_date,
            travel_mode=body.travel_mode,
            notes=body.notes,
            species=body.species,
            site_zone_id=body.site_zone_id,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"assignment_id": assignment_id}


# MIGRATED FROM app.py show_planter_field_view
@router.get("/{planter_id}/field-points")
def field_points(planter_id: int):
    planter = get_planter(planter_id)
    if not planter:
        raise HTTPException(status_code=404, detail="Planter not found")
    return get_planter_field_points(planter_id)


@router.post("/{planter_id}/participants/{slot}/reset-device")
def reset_device(planter_id: int, slot: int):
    try:
        reset_participant_device(planter_id, slot)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"status": "reset", "slot": slot}
