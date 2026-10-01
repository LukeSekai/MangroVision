"""Stable project-site GeoJSON CRUD endpoints."""

from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, status
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat

from api.runtime_state import is_processing_active
from planting_database import (
    create_project_site,
    delete_project_site,
    get_user_by_session_token,
    get_project_site,
    list_organizations,
    list_project_sites,
    update_project_site,
    set_project_site_tide_calibration,
)
from .tides import tides_forecast

router = APIRouter()


def _require_lgu_user() -> dict:
    """Protect project-site mutations with the existing LGU session boundary."""
    user = get_user_by_session_token("")
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired LGU session.")
    if str(user.get("role") or "").lower() not in {"admin", "lgu", "planner"}:
        raise HTTPException(status_code=403, detail="LGU/admin permission is required.")
    return user


def _raise_if_processing_active() -> None:
    if is_processing_active():
        raise HTTPException(
            status_code=409,
            detail="Project-site editing is locked while image processing is running.",
        )


class ProjectSiteCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    organization_id: int
    geometry: dict
    notes: Optional[str] = None
    analysis_ids: Optional[list[int]] = None
    assignment_ids: Optional[list[int]] = None


class ProjectSiteUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: Optional[str] = None
    geometry: Optional[dict] = None
    notes: Optional[str] = None
    analysis_ids: Optional[list[int]] = None
    assignment_ids: Optional[list[int]] = None


class TideCalibration(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    elevation_m: FiniteFloat = Field(strict=True)
    datum_reference: str = Field(min_length=1, max_length=200)
    survey_reference: str = Field(min_length=10, max_length=1000)
    datum_compatibility_confirmed: Literal[True]


@router.put("/{site_id}/tide-calibration")
def save_tide_calibration(site_id: int, body: Optional[TideCalibration] = None):
    _require_lgu_user()
    _raise_if_processing_active()
    site = get_project_site(site_id)
    if site is None:
        raise HTTPException(status_code=404, detail="Project site not found.")
    calibration = None
    if body is not None:
        props = site["properties"]
        lat, lon = props.get("centroid_lat"), props.get("centroid_lon")
        if lat is None or lon is None:
            raise HTTPException(status_code=400, detail="Site coordinates are unavailable.")
        forecast = tides_forecast(lat=lat, lon=lon, days=7)
        if (not forecast.get("available") or forecast.get("stale")
                or not forecast.get("series_available") or forecast.get("datum") != "MSL"
                or forecast.get("datum_reference") != body.datum_reference):
            raise HTTPException(status_code=409, detail="Forecast reference changed or is unavailable. Refresh tides and verify the survey datum again.")
        calibration = {
            "elevation_m": body.elevation_m,
            "datum_reference": body.datum_reference,
            "survey_reference": body.survey_reference,
            "datum_compatibility_confirmed": True,
            "forecast_lat": round(lat, 4), "forecast_lon": round(lon, 4),
            "source": forecast["source"],
            "recorded_at": datetime.now(timezone.utc).isoformat(),
        }
    try:
        result = set_project_site_tide_calibration(site_id, calibration)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if result is None:
        raise HTTPException(status_code=404, detail="Project site not found.")
    return result


@router.get("")
@router.get("/")
def project_sites():
    return {
        "type": "FeatureCollection",
        "name": "project_sites",
        "features": list_project_sites(),
        "organizations": list_organizations(),
    }


@router.get("/{site_id}")
def project_site(site_id: int):
    result = get_project_site(site_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Project site not found.")
    return result


@router.post("", status_code=status.HTTP_201_CREATED)
@router.post("/", status_code=status.HTTP_201_CREATED, include_in_schema=False)
def add_project_site(body: ProjectSiteCreate):
    _require_lgu_user()
    _raise_if_processing_active()
    try:
        return create_project_site(
            name=body.name,
            geometry=body.geometry,
            notes=body.notes or "",
            analysis_ids=body.analysis_ids,
            assignment_ids=body.assignment_ids,
            organization_id=body.organization_id,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


def _update_project_site(site_id: int, body: ProjectSiteUpdate):
    _require_lgu_user()
    _raise_if_processing_active()
    try:
        result = update_project_site(
            site_id=site_id,
            name=body.name,
            geometry=body.geometry,
            notes=body.notes,
            analysis_ids=body.analysis_ids,
            assignment_ids=body.assignment_ids,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    if result is None:
        raise HTTPException(status_code=404, detail="Project site not found.")
    return result


@router.put("/{site_id}")
def replace_project_site(site_id: int, body: ProjectSiteUpdate):
    return _update_project_site(site_id, body)


@router.patch("/{site_id}")
def edit_project_site(site_id: int, body: ProjectSiteUpdate):
    return _update_project_site(site_id, body)


@router.delete("/{site_id}")
def remove_project_site(site_id: int):
    _require_lgu_user()
    _raise_if_processing_active()
    try:
        removed = delete_project_site(site_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    if not removed:
        raise HTTPException(status_code=404, detail="Project site not found.")
    return {"status": "deleted", "id": site_id}
