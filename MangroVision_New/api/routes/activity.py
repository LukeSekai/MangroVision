"""Role-scoped activity history."""

from fastapi import APIRouter, Depends, HTTPException, Query

from api.routes.monitoring import _require_lgu_user
from mangrovision_db.activity import list_activity
from planting_database import get_planter_by_session_token

router = APIRouter()


@router.get("/staff")
def staff_activity(before_id: int | None = Query(default=None, ge=1),
                   limit: int = Query(default=50, ge=1, le=100),
                   _user: dict = Depends(_require_lgu_user)):
    return {"items": list_activity(before_id=before_id, limit=limit)}


@router.get("/field")
def field_activity(before_id: int | None = Query(default=None, ge=1),
                   limit: int = Query(default=50, ge=1, le=100)):
    planter = get_planter_by_session_token("")
    if not planter or planter.get("organization_id") is None:
        raise HTTPException(status_code=401, detail="Sign in with an organization field account.")
    return {"items": list_activity(
        organization_id=int(planter["organization_id"]),
        before_id=before_id, limit=limit,
    )}
