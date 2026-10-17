"""Stable project-site GeoJSON CRUD endpoints."""

from datetime import datetime, timezone
from typing import Literal, Optional

from fastapi import APIRouter, HTTPException, Query, status
from pydantic import BaseModel, ConfigDict, Field, FiniteFloat
from shapely.geometry import shape
from shapely.ops import unary_union
from sqlalchemy.exc import SQLAlchemyError

from api.runtime_state import is_processing_active
from planting_database import (
    create_project_site,
    delete_project_site,
    get_user_by_session_token,
    get_project_site,
    list_organizations,
    list_project_sites,
    list_site_zones,
    update_project_site,
    set_project_site_tide_calibration,
)
from .tides import tides_forecast
from .zones import _checked_zone_polygon, _require_zone_inside_map
from mangrovision_db.site_actions import apply_site_action, available_released_areas, preview_site_actions

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


class ProjectSiteBoundary(BaseModel):
    organization_id: int = Field(gt=0)
    geometry: dict


def _require_organization_boundary(organization_id: int, geometry: dict):
    polygon = _checked_zone_polygon(geometry)
    _require_zone_inside_map(geometry)
    try:
        sites = [feature for feature in list_project_sites()
                 if (feature.get('geometry') or {}).get('type') in {'Polygon', 'MultiPolygon'}]
        assignments = list_site_zones()
        def owned(features):
            return [feature for feature in features
                    if str((feature.get("properties") or {}).get("organization_id")) == str(organization_id)]

        def polygons(features):
            output = []
            for feature in features:
                candidate = shape(feature.get("geometry") or {})
                if candidate.geom_type not in {"Polygon", "MultiPolygon"} or not candidate.is_valid or candidate.is_empty:
                    raise ValueError("Invalid recorded organization area")
                output.append(candidate)
            return output

        own_sites = owned(sites)
        # A first project site establishes the organization's boundary. Assignment
        # hulls and released land must not become a prerequisite for creating it.
        # Existing owners remain limited to their sites and available releases.
        if own_sites:
            allowed = unary_union(polygons(own_sites + available_released_areas()['features']))
            if not allowed.covers(polygon):
                raise HTTPException(status_code=400, detail="This site extends outside the selected organization's area. Redraw it inside that area.")
        # Shared edges are allowed; an area belonging to another owner is not.
        recorded_owners = {str((feature.get('properties') or {}).get('organization_id')) for feature in sites}
        other_areas = sites + [feature for feature in assignments
                              if str((feature.get('properties') or {}).get('organization_id')) not in recorded_owners]
        others = [feature for feature in other_areas
                  if (feature.get("properties") or {}).get("organization_id") is not None
                  and str(feature["properties"]["organization_id"]) != str(organization_id)]
        if others and unary_union(polygons(others)).intersection(polygon).area > 0:
            raise HTTPException(status_code=400, detail="This site overlaps another organization's area. Redraw it inside the selected organization's area.")
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=503, detail="Could not check the organization's area. Refresh the map and try again.") from error


@router.post("/validate-boundary")
def validate_project_site_boundary(body: ProjectSiteBoundary):
    _require_lgu_user()
    _raise_if_processing_active()
    _require_organization_boundary(body.organization_id, body.geometry)
    return {"valid": True}


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
        "released_areas": available_released_areas(),
    }


class SiteActionBody(BaseModel):
    version: str = Field(min_length=64, max_length=64)


@router.get('/{site_id}/actions')
def site_actions(site_id: int):
    _require_lgu_user()
    try:
        return preview_site_actions(site_id)
    except KeyError as error:
        raise HTTPException(status_code=404, detail='Project site not found.') from error
    except SQLAlchemyError as error:
        raise HTTPException(status_code=503, detail='Could not check this site. Please try again.') from error


@router.post('/{site_id}/release')
def release_project_site_area(site_id: int, body: SiteActionBody):
    user = _require_lgu_user()
    _raise_if_processing_active()
    try:
        return apply_site_action(site_id, 'release', body.version, int(user['id']))
    except KeyError as error:
        raise HTTPException(status_code=404, detail='Project site not found.') from error
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except SQLAlchemyError as error:
        raise HTTPException(status_code=503, detail='Could not release this area. Review the site and try again.') from error


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
    _require_organization_boundary(body.organization_id, body.geometry)
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
    if body.geometry is not None:
        existing = get_project_site(site_id)
        if existing is None:
            raise HTTPException(status_code=404, detail="Project site not found.")
        owner_id = existing.get("properties", {}).get("organization_id")
        if owner_id is None:
            raise HTTPException(status_code=400, detail="This project site has no organization owner. Assign its owner before changing the boundary.")
        _require_organization_boundary(int(owner_id), body.geometry)
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
def remove_project_site(site_id: int, version: str = Query(min_length=64, max_length=64)):
    user = _require_lgu_user()
    _raise_if_processing_active()
    try:
        removed = delete_project_site(site_id, expected_version=version, user_id=int(user['id']))
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    except SQLAlchemyError as error:
        raise HTTPException(status_code=503, detail='Could not delete this site. Review the site and try again.') from error
    if not removed:
        raise HTTPException(status_code=404, detail="Project site not found.")
    return {"status": "deleted", "id": site_id}
