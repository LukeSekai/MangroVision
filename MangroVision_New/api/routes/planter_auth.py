"""Planter self-service authentication endpoints (mobile /field app)."""

from typing import List, Optional, Literal

from fastapi import APIRouter, HTTPException, Response, Request
from pydantic import BaseModel, Field
from mangrovision_db.planter_codes import CodeError
from mangrovision_db.group_access import authenticate_group, check_group_code, GroupAccessError
from mangrovision_db.planter_activities import list_planter_activities

from planting_database import (
    get_planter_by_session_token,
    get_planter_field_points,
    list_organizations,
    list_project_sites,
    mark_planter_points_completed,
    revoke_planter_session,
    update_assignment_point_status,
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
        raise HTTPException(status_code=401, detail="Sign in with your organization code and username.")
    if planter.get('planting_schedule_id') is None:
        raise HTTPException(status_code=401, detail='Use the activity’s organization code and choose your participant username. Your existing progress is kept.')
    return planter


def _assert_point_belongs_to_planter(assignment_point_id: int, planter: dict) -> None:
    allowed = {point["assignment_point_id"] for point in
               get_planter_field_points(planter["id"], planter["participant_slot"], planter['planting_schedule_id'])}
    if assignment_point_id not in allowed:
        raise HTTPException(status_code=403, detail="That point is not assigned to your participant slot.")


class PlanterJoinRequest(BaseModel):
    access_code: str = Field(min_length=1, max_length=32)


class PlanterLoginRequest(PlanterJoinRequest):
    username: str = Field(min_length=1, max_length=64)
    browser_key: str = Field(min_length=48, max_length=48)
    mode: Literal['first', 'return'] = 'return'
    switch_token: Optional[str] = Field(default=None, max_length=2048)
    previous_code: Optional[str] = Field(default=None, max_length=32)


def _planter_public_dict(planter: dict) -> dict:
    """Credentials and the former shared username are never public sign-in data."""
    safe = {key: value for key, value in planter.items()
            if key not in {"password_hash", "username", "encrypted_code", "code_digest", "access_code"}}
    return safe


# Organization choices are also used by the staff assignment panel.
@router.get("/organizations")
def planter_organizations():
    """Organization choices used by the staff assignment panel."""
    return {"organizations": list_organizations()}


@router.post("/register")
def register_planter():
    # Keep a clear response for older cached clients, with no account/device writes.
    raise HTTPException(status_code=410, detail="Use First time on the planter sign-in page with your organization code and a username.")


def _access_error(error):
    headers = {'Retry-After': str(error.retry_after)} if error.retry_after else None
    if isinstance(error, GroupAccessError) and error.kind:
        detail = {'kind': error.kind, 'message': str(error), **error.details}
    elif error.status < 500 and error.status != 429 and getattr(error, 'field', 'access_code'):
        detail = [{'loc': ['body', getattr(error, 'field', 'access_code')], 'type': 'credential_error', 'msg': str(error)}]
    else:
        detail = str(error)
    return HTTPException(status_code=error.status, detail=detail, headers=headers)


@router.post("/join")
def planter_join(body: PlanterJoinRequest, response: Response, request: Request):
    try:
        activity = check_group_code(body.access_code,
            address=request.client.host if request.client else 'unknown')
    except CodeError as error:
        raise _access_error(error) from error
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    response.headers['Cache-Control'] = 'no-store'
    return {"activity": activity}


@router.post("/login")
def planter_login(body: PlanterLoginRequest, response: Response, request: Request):
    try:
        planter, token = authenticate_group(body.access_code, body.username, body.browser_key,
            mode=body.mode, switch_token=body.switch_token, previous_code=body.previous_code,
            address=request.client.host if request.client else 'unknown')
    except CodeError as error:
        raise _access_error(error) from error
    except ValueError as error:
        raise HTTPException(status_code=503, detail=str(error)) from error
    set_planter_session(response, token)
    response.headers['Cache-Control'] = 'no-store'
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
@router.get('/me/activities')
def my_activities(response: Response):
    planter = _require_planter()
    response.headers['Cache-Control'] = 'no-store'
    return {'activities': list_planter_activities(planter.get('organization_id'))}


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
        "points": get_planter_field_points(planter["id"], planter["participant_slot"], planter['planting_schedule_id']),
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
               get_planter_field_points(planter["id"], planter["participant_slot"], planter['planting_schedule_id'])}
    if not set(body.assignment_point_ids).issubset(allowed):
        raise HTTPException(status_code=403, detail="Only points in your participant slot can be completed.")
    try:
        return mark_planter_points_completed(planter["id"], body.assignment_point_ids,
                                             participant_slot=planter["participant_slot"])
    except CodeError as error:
        raise _access_error(error) from error
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
    except CodeError as error:
        raise _access_error(error) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return {"status": "updated"}
