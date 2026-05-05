"""Planter self-service authentication endpoints (mobile /field app)."""

from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from planting_database import (
    _get_connection,
    authenticate_planter,
    create_planter,
    create_planter_session,
    get_planter,
    get_planter_by_session_token,
    get_planter_field_points,
    revoke_planter_session,
    update_assignment_point_status,
    update_planter_last_login,
)

router = APIRouter()


class UpdatePointStatusBody(BaseModel):
    status: str


def _require_planter(token: str) -> dict:
    planter = get_planter_by_session_token(token)
    if not planter:
        raise HTTPException(status_code=401, detail="Invalid or expired planter session")
    if planter.get("status") != "active":
        raise HTTPException(status_code=403, detail="This planter account is inactive.")
    return planter


def _assert_point_belongs_to_planter(assignment_point_id: int, planter_id: int) -> None:
    conn = _get_connection()
    row = conn.execute(
        """
        SELECT pa.planter_id
        FROM planter_assignment_points pap
        JOIN planter_assignments pa ON pa.id = pap.assignment_id
        WHERE pap.id = ?
        """,
        (assignment_point_id,),
    ).fetchone()
    conn.close()
    if row is None:
        raise HTTPException(status_code=404, detail="Assignment point was not found.")
    if row["planter_id"] != planter_id:
        raise HTTPException(status_code=403, detail="That point is not on your assignment.")


class PlanterRegisterRequest(BaseModel):
    full_name: str
    username: str
    password: str
    phone: str = ""
    base_label: str = ""
    base_lat: Optional[float] = None
    base_lon: Optional[float] = None


class PlanterLoginRequest(BaseModel):
    username: str
    password: str


def _planter_public_dict(planter: dict) -> dict:
    """Strip password_hash before returning a planter to the client."""
    safe = {key: value for key, value in planter.items() if key != "password_hash"}
    return safe


# MIGRATED FROM app.py planter self-registration flow
@router.post("/register")
def register_planter(body: PlanterRegisterRequest):
    try:
        planter_id = create_planter(
            full_name=body.full_name,
            username=body.username,
            password=body.password,
            phone=body.phone,
            base_label=body.base_label,
            base_lat=body.base_lat,
            base_lon=body.base_lon,
            status="active",
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    planter = get_planter(planter_id)
    token = create_planter_session(planter_id)
    update_planter_last_login(planter_id)
    return {"token": token, "planter": _planter_public_dict(planter)}


@router.post("/login")
def planter_login(body: PlanterLoginRequest):
    planter = authenticate_planter(body.username, body.password)
    if not planter:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if planter.get("status") != "active":
        raise HTTPException(status_code=403, detail="This planter account is inactive.")
    token = create_planter_session(planter["id"])
    update_planter_last_login(planter["id"])
    return {"token": token, "planter": _planter_public_dict(planter)}


@router.get("/session")
def planter_session(token: str):
    planter = get_planter_by_session_token(token)
    if not planter:
        raise HTTPException(status_code=401, detail="Invalid or expired planter session")
    return {"planter": _planter_public_dict(planter)}


@router.post("/logout")
def planter_logout(token: str):
    revoke_planter_session(token)
    return {"status": "ok"}


# MIGRATED FROM app.py show_planter_field_view (/field workspace)
@router.get("/me/field-points")
def my_field_points(token: str):
    planter = _require_planter(token)
    return {
        "planter": _planter_public_dict(planter),
        "points": get_planter_field_points(planter["id"]),
    }


# MIGRATED FROM app.py update_assignment_point_status (field mark-planted action)
@router.patch("/me/points/{assignment_point_id}/status")
def my_point_status(assignment_point_id: int, body: UpdatePointStatusBody, token: str):
    planter = _require_planter(token)
    _assert_point_belongs_to_planter(assignment_point_id, planter["id"])
    try:
        update_assignment_point_status(assignment_point_id, body.status)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"status": "updated"}
