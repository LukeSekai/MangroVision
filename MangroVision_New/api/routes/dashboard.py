"""Decision-dashboard aggregate and settings endpoints."""

from typing import Any, Callable, Optional

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from planting_database import (
    get_dashboard_ecology,
    get_dashboard_operations,
    get_dashboard_record_notices,
    get_dashboard_overview,
    get_dashboard_settings,
    get_dashboard_sites,
    get_user_by_session_token,
    update_dashboard_settings,
)

router = APIRouter()


def _dashboard_filters(
    date_from: Optional[str] = None,
    date_to: Optional[str] = None,
    site_id: Optional[int] = None,
    assignment_id: Optional[int] = None,
    species: Optional[str] = None,
    planter_id: Optional[int] = None,
    bucket: str = "week",
    as_of: Optional[str] = None,
) -> dict[str, Any]:
    """Shared filter contract for all pre-aggregated dashboard views."""
    return {
        "date_from": date_from,
        "date_to": date_to,
        "site_id": site_id,
        "assignment_id": assignment_id,
        "species": species,
        "planter_id": planter_id,
        "bucket": bucket,
        "as_of": as_of,
    }


def _aggregate(
    getter: Callable[..., dict],
    filters: dict[str, Any],
) -> dict:
    try:
        return getter(**filters)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.get("/overview")
def dashboard_overview(filters: dict[str, Any] = Depends(_dashboard_filters)):
    return _aggregate(get_dashboard_overview, filters)


@router.get("/operations")
def dashboard_operations(filters: dict[str, Any] = Depends(_dashboard_filters)):
    return _aggregate(get_dashboard_operations, filters)


@router.get("/ecology")
def dashboard_ecology(filters: dict[str, Any] = Depends(_dashboard_filters)):
    return _aggregate(get_dashboard_ecology, filters)


@router.get("/sites")
def dashboard_sites(filters: dict[str, Any] = Depends(_dashboard_filters)):
    return _aggregate(get_dashboard_sites, filters)


class DashboardSettingsUpdate(BaseModel):
    """Partial settings update; explicit null clears an optional target."""

    model_config = ConfigDict(extra="forbid")

    year: Optional[int] = None
    annual_planting_target: Optional[int] = None
    min_survival_target_pct: Optional[float] = None
    inspection_intervals_days: Optional[list[int]] = None
    inspection_weekdays: Optional[list[int]] = None


def _require_lgu_user() -> dict:
    user = get_user_by_session_token("")
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired LGU session.")
    if str(user.get("role") or "").lower() not in {"admin", "lgu", "planner"}:
        raise HTTPException(status_code=403, detail="LGU/admin permission is required.")
    return user


@router.get("/record-notices", dependencies=[Depends(_require_lgu_user)])
def dashboard_record_notices():
    return get_dashboard_record_notices()


@router.get("/settings")
def dashboard_settings(year: Optional[int] = None):
    try:
        return get_dashboard_settings(year)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.put("/settings")
def save_dashboard_settings(body: DashboardSettingsUpdate):
    user = _require_lgu_user()
    # Preserve the database layer's omitted-versus-null semantics. Missing
    # fields keep their current value, while an explicit null clears a target
    # (and resets inspection intervals to their defaults).
    values = body.model_dump(exclude_unset=True)
    year = values.pop("year", body.year)
    try:
        return update_dashboard_settings(
            year=year,
            updated_by_user_id=int(user["id"]),
            **values,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
