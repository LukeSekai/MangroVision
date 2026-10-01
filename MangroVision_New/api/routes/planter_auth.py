"""Planter self-service authentication endpoints (mobile /field app)."""

from typing import List, Optional

from fastapi import APIRouter, HTTPException, Response
from pydantic import BaseModel, Field
from mangrovision_db.organization_accounts import claim_participant_slot

from planting_database import (
    authenticate_planter,
    create_planter,
    create_planter_session,
    get_planter,
    get_planter_by_session_token,
    get_planter_field_points,
    list_organizations,
    list_project_sites,
    mark_planter_points_completed,
    revoke_planter_session,
    update_assignment_point_status,
    update_planter_last_login,
)
from api.security import clear_planter_session, set_planter_session

router = APIRouter()


class UpdatePointStatusBody(BaseModel):
    status: str
    skip_reason: Optional[str] = None


class MarkPointsCompletedBody(BaseModel):
    assignment_point_ids: List[int]


def _require_planter() -> dict:
    planter = get_planter_by_session_token("")
    if not planter:
        raise HTTPException(status_code=401, detail="Invalid or expired planter session")
    if planter.get("status") != "active":
        raise HTTPException(status_code=403, detail="This planter account is inactive.")
    if planter.get("participant_slot") is None:
        raise HTTPException(status_code=401, detail="Sign in again to join a participant slot.")
    return planter


def _assert_point_belongs_to_planter(assignment_point_id: int, planter: dict) -> None:
    allowed = {point["assignment_point_id"] for point in
               get_planter_field_points(planter["id"], planter["participant_slot"])}
    if assignment_point_id not in allowed:
        raise HTTPException(status_code=403, detail="That point is not assigned to your participant slot.")


class PlanterRegisterRequest(BaseModel):
    full_name: str = "Organization account"
    username: str
    password: str
    organization_id: int
    participant_count: int = Field(default=1, ge=1, le=10000)
    device_key: str = Field(min_length=16, max_length=200)
    phone: str = ""
    base_label: str = ""
    base_lat: Optional[float] = None
    base_lon: Optional[float] = None


class PlanterLoginRequest(BaseModel):
    username: str
    password: str
    device_key: str = Field(min_length=16, max_length=200)
    participant_slot: Optional[int] = Field(default=None, ge=1, le=10000)
    recover_slot: bool = False


def _planter_public_dict(planter: dict) -> dict:
    """Strip password_hash before returning a planter to the client."""
    safe = {key: value for key, value in planter.items() if key != "password_hash"}
    return safe


# MIGRATED FROM app.py planter self-registration flow
@router.get("/organizations")
def planter_organizations():
    """Public organization choices used by the planter registration form."""
    return {"organizations": list_organizations()}


@router.post("/register")
def register_planter(body: PlanterRegisterRequest, response: Response):
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
            organization_id=body.organization_id,
            participant_count=body.participant_count,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    planter = get_planter(planter_id)
    slot = claim_participant_slot(planter_id, body.device_key)
    planter["participant_slot"] = slot
    token = create_planter_session(planter_id, slot)
    set_planter_session(response, token)
    update_planter_last_login(planter_id)
    return {"planter": _planter_public_dict(planter)}


@router.post("/login")
def planter_login(body: PlanterLoginRequest, response: Response):
    planter = authenticate_planter(body.username, body.password)
    if not planter:
        raise HTTPException(status_code=401, detail="Invalid username or password")
    if planter.get("status") != "active":
        raise HTTPException(status_code=403, detail="This planter account is inactive.")
    try:
        slot = claim_participant_slot(planter["id"], body.device_key, body.participant_slot, recover_slot=body.recover_slot)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    planter["participant_slot"] = slot
    token = create_planter_session(planter["id"], slot)
    set_planter_session(response, token)
    update_planter_last_login(planter["id"])
    return {"planter": _planter_public_dict(planter)}


@router.get("/session")
def planter_session():
    return {"planter": _planter_public_dict(_require_planter())}


@router.post("/logout")
def planter_logout(response: Response):
    revoke_planter_session("")
    clear_planter_session(response)
    return {"status": "ok"}


# MIGRATED FROM app.py show_planter_field_view (/field workspace)
@router.get("/me/field-points")
def my_field_points():
    planter = _require_planter()
    organization_id = planter.get("organization_id")
    project_sites = []
    if organization_id is not None:
        project_sites = [
            feature for feature in list_project_sites()
            if feature.get("properties", {}).get("organization_id") == int(organization_id)
        ]
    return {
        "planter": _planter_public_dict(planter),
        "points": get_planter_field_points(planter["id"], planter["participant_slot"]),
        # An organization's mapped working area is independent of whether
        # this particular planter has been assigned planting points yet.
        "project_sites": {
            "type": "FeatureCollection",
            "name": "organization_project_sites",
            "features": project_sites,
        },
    }


@router.post("/me/points/mark-all-completed")
def my_points_mark_all_completed(body: MarkPointsCompletedBody):
    planter = _require_planter()
    allowed = {point["assignment_point_id"] for point in
               get_planter_field_points(planter["id"], planter["participant_slot"])}
    if not set(body.assignment_point_ids).issubset(allowed):
        raise HTTPException(status_code=403, detail="Only points in your participant slot can be completed.")
    try:
        return mark_planter_points_completed(planter["id"], body.assignment_point_ids,
                                             participant_slot=planter["participant_slot"])
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


# MIGRATED FROM app.py update_assignment_point_status (field mark-planted action)
@router.patch("/me/points/{assignment_point_id}/status")
def my_point_status(assignment_point_id: int, body: UpdatePointStatusBody):
    planter = _require_planter()
    _assert_point_belongs_to_planter(assignment_point_id, planter)
    try:
        update_assignment_point_status(
            assignment_point_id, body.status, body.skip_reason,
            actor_planter_id=int(planter["id"]),
            participant_slot=int(planter["participant_slot"]),
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"status": "updated"}
