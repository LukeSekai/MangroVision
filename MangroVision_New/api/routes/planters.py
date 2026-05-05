"""Planter management endpoints."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from planting_database import (
    _get_connection,
    _hash_password,
    assign_planting_point_to_planter,
    create_planter,
    create_planter_assignment,
    get_planter,
    get_planter_dashboard_stats,
    get_planter_field_points,
    list_planter_assignment_map_points,
    list_planters,
)

router = APIRouter()


class AssignPointRequest(BaseModel):
    planter_id: int
    planting_point_id: int
    allow_reassign: bool = False
    travel_mode: str = "walking"


class CreatePlanterRequest(BaseModel):
    full_name: str
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


@router.get("/")
def get_planters(include_inactive: bool = True):
    return list_planters(include_inactive=include_inactive)


@router.get("/dashboard")
def dashboard_stats():
    return get_planter_dashboard_stats()


@router.get("/map-points")
def map_points():
    return list_planter_assignment_map_points()


@router.post("/assign-point")
def assign_point(body: AssignPointRequest):
    return assign_planting_point_to_planter(
        planter_id=body.planter_id,
        planting_point_id=body.planting_point_id,
        allow_reassign=body.allow_reassign,
        travel_mode=body.travel_mode,
    )


# MIGRATED FROM app.py planter management tab (create form)
@router.post("/")
def create_planter_endpoint(body: CreatePlanterRequest):
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

    updates: list[tuple[str, object]] = []
    if body.full_name is not None:
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
def create_assignment_endpoint(planter_id: int, body: CreateAssignmentRequest):
    planter = get_planter(planter_id)
    if not planter:
        raise HTTPException(status_code=404, detail="Planter not found")
    try:
        assignment_id = create_planter_assignment(
            planter_id=planter_id,
            planting_point_ids=body.planting_point_ids,
            assigned_by_user_id=body.assigned_by_user_id,
            title=body.title,
            assignment_date=body.assignment_date,
            travel_mode=body.travel_mode,
            notes=body.notes,
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
