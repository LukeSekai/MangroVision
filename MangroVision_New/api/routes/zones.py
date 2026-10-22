"""Zone data endpoints — forbidden, eroded, and site zones."""

from fastapi import APIRouter, HTTPException, Header
from pydantic import BaseModel
from typing import Any, Optional
import logging
import math

from shapely import get_coordinates
from shapely.geometry import shape

from api.runtime_state import is_processing_active
from planting_database import (
    create_warning_zone,
    delete_warning_zone,
    get_mortality_detail_table,
    get_site_zone_mortality,
    get_warning_zone,
    get_user_by_session_token,
    invalidate_eroded_zone_cache,
    list_site_zones,
    list_warning_zones,
    update_warning_zone,
)
from mangrovision_db.zones import (
    create_zone,
    feature_collection,
    replace_zone_collection,
    soft_delete_zone,
    update_zone,
)
from mangrovision_db.erosion import SeedlingOverlap

router = APIRouter()
logger = logging.getLogger(__name__)

OUTSIDE_MAP_MESSAGE = "This zone extends outside the mapped area. Redraw it fully inside the drone map."


def _load_zone_coverage():
    # Use the same irregular imagery/GIS intersection as planting points and
    # image analysis, rather than the rectangular tile bounds or satellite map.
    from canopy_detection.ortho_matcher import _ensure_active_ortho
    from canopy_detection.orthophoto_coverage import visible_gis_coverage

    return visible_gis_coverage(
        _ensure_active_ortho(), feature_collection("gis_coverage").get("features", [])
    )


def _checked_zone_polygon(geometry):
    try:
        if not isinstance(geometry, dict) or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
            raise ValueError("Not a polygon")
        polygon = shape(geometry)
        coordinates = get_coordinates(polygon)
        if (polygon.is_empty or not polygon.is_valid or polygon.area <= 0
                or any(not math.isfinite(x) or not math.isfinite(y)
                       or abs(x) > 180 or abs(y) > 90 for x, y in coordinates)):
            raise ValueError("Invalid polygon")
        return polygon
    except Exception as error:
        raise HTTPException(
            status_code=400,
            detail="The zone boundary is invalid. Draw a closed area without crossing its own outline.",
        ) from error


def _require_zone_inside_map(geometry, coverage=None):
    polygon = _checked_zone_polygon(geometry)
    try:
        coverage = _load_zone_coverage() if coverage is None else coverage
        if coverage is None or coverage.is_empty or not coverage.is_valid:
            raise ValueError("No valid map boundary")
        inside = coverage.covers(polygon)
    except Exception as error:
        logger.warning("Could not check the zone against the visible map boundary: %s", type(error).__name__)
        raise HTTPException(
            status_code=503, detail="Could not check the map boundary. Please try again.",
        ) from error
    if not inside:
        raise HTTPException(status_code=400, detail=OUTSIDE_MAP_MESSAGE)
    return coverage


class ZoneBoundaryBody(BaseModel):
    geometry: dict


@router.post("/validate-boundary")
def validate_zone_boundary(body: ZoneBoundaryBody):
    _raise_if_processing_active()
    _require_lgu_user()
    _require_zone_inside_map(body.geometry)
    return {"valid": True}

def _raise_if_processing_active():
    if is_processing_active():
        raise HTTPException(
            status_code=409,
            detail="Zone editing is locked while image processing is running.",
        )


def _require_lgu_user() -> dict:
    user = get_user_by_session_token("")
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired LGU session.")
    if str(user.get("role") or "").lower() not in {"admin", "lgu", "planner"}:
        raise HTTPException(status_code=403, detail="LGU/admin permission is required.")
    return user


@router.get("/forbidden")
def get_forbidden_zones():
    return feature_collection("forbidden")


@router.get("/eroded")
def get_eroded_zones():
    return feature_collection("eroded")


class SaveErodedBody(BaseModel):
    features: list[Any]


@router.put("/eroded")
def save_eroded_zones(body: SaveErodedBody, x_confirm_seedling_overlap: str = Header(default='')):
    _raise_if_processing_active()
    user = _require_lgu_user()
    # Check the entire replacement before the repository can save any feature.
    # An empty collection remains usable for clearing old zones without imagery.
    coverage = None
    for feature in body.features:
        coverage = _require_zone_inside_map(
            feature.get("geometry") if isinstance(feature, dict) else None, coverage,
        )
    try:
        options = {'confirm_overlap': True} if x_confirm_seedling_overlap == 'true' else {}
        collection = replace_zone_collection("eroded", body.features, int(user["id"]), **options)
    except SeedlingOverlap as error:
        raise HTTPException(409, detail={'code': 'seedling_overlap', 'message': str(error)}) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    invalidate_eroded_zone_cache()
    return {"status": "saved", "count": len(collection["features"]), **collection}


@router.delete("/eroded/{zone_id}")
def delete_eroded_zone(zone_id: int):
    _raise_if_processing_active()
    _require_lgu_user()
    if not soft_delete_zone("eroded", zone_id):
        raise HTTPException(status_code=404, detail="Eroded zone not found")
    invalidate_eroded_zone_cache()
    remaining = len(feature_collection("eroded")["features"])
    return {"status": "deleted", "id": zone_id, "remaining": remaining}


@router.post("/eroded")
def add_eroded_zone(feature: dict, x_confirm_seedling_overlap: str = Header(default='')):
    """Create one erosion zone with a stable database ID."""
    _raise_if_processing_active()
    user = _require_lgu_user()
    _require_zone_inside_map(feature.get("geometry"))
    try:
        options = {'confirm_overlap': True} if x_confirm_seedling_overlap == 'true' else {}
        saved = create_zone("eroded", feature, int(user["id"]), **options)
    except SeedlingOverlap as error:
        raise HTTPException(409, detail={'code': 'seedling_overlap', 'message': str(error)}) from error
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    invalidate_eroded_zone_cache()
    return {"status": "added", "zone": saved, "id": saved["id"]}


@router.patch('/eroded/{zone_id}')
def edit_eroded_zone(zone_id: int, feature: dict, x_confirm_seedling_overlap: str = Header(default='')):
    _raise_if_processing_active()
    _require_lgu_user()
    _require_zone_inside_map(feature.get('geometry'))
    try:
        saved = update_zone('eroded', zone_id, feature,
                            confirm_overlap=x_confirm_seedling_overlap == 'true')
    except KeyError as error:
        raise HTTPException(404, detail='Eroded zone not found') from error
    except SeedlingOverlap as error:
        raise HTTPException(409, detail={'code': 'seedling_overlap', 'message': str(error)}) from error
    except ValueError as error:
        raise HTTPException(400, detail=str(error)) from error
    invalidate_eroded_zone_cache()
    return {'status': 'saved', 'zone': saved, 'id': saved['id']}


# ── Site zones (auto-derived from planter assignments) ──────────────
# Site zones used to be admin-drawn polygons. They are now generated 1:1
# from planter_assignments — each assignment's points form a convex-hull
# polygon — so per-area survival tracking happens automatically whenever
# the planner hands out work. The manual create/update/delete endpoints
# have been removed; GET still returns a GeoJSON FeatureCollection so the
# existing map-rendering pipeline keeps working unchanged.

class WarningZoneCreate(BaseModel):
    name: str
    geometry: dict
    warning_type: str = "planner_warning"
    severity: str = "medium"
    notes: Optional[str] = ""


class WarningZoneUpdate(BaseModel):
    name: Optional[str] = None
    geometry: Optional[dict] = None
    warning_type: Optional[str] = None
    severity: Optional[str] = None
    notes: Optional[str] = None


@router.get("/sites")
def get_site_zones():
    return {
        "type": "FeatureCollection",
        "name": "site_zones",
        "features": list_site_zones(),
    }


@router.get("/sites/mortality")
def get_site_zones_mortality():
    """Per-zone alive/dead/total/survival_rate, one row per assignment."""
    return {"zones": get_site_zone_mortality()}


@router.get("/sites/detail")
def get_site_zones_detail():
    """Full per-zone breakdown used by the Monitoring mortality overlay.

    Returns each assignment-derived site zone with its per-point status,
    planted_at, death_at, and death_reason — including historical deaths whose
    points have since been reset-to-planned and detached from the assignment.
    """
    return {"zones": get_mortality_detail_table()}


# Non-blocking planner/expert warning zones. These mark plantable points with
# risk context (deep mud, difficult access, low survival confidence) without
# excluding the point from assignment.

@router.get("/warnings")
def get_warning_zones():
    """List manual zones and site summaries, not per-location wave histories.

    Wave-death warnings are supplied with planting points. Automatic mortality
    summaries do not classify every point in their site's geometry as dangerous.
    """
    return {
        "type": "FeatureCollection",
        "name": "warning_zones",
        "features": list_warning_zones(),
    }


@router.get("/warnings/{zone_id}")
def get_warning_zone_endpoint(zone_id: int):
    zone = get_warning_zone(zone_id)
    if not zone:
        raise HTTPException(status_code=404, detail="Warning zone not found")
    return zone


@router.post("/warnings")
def create_warning_zone_endpoint(body: WarningZoneCreate):
    _raise_if_processing_active()
    _require_lgu_user()
    _require_zone_inside_map(body.geometry)
    try:
        zone_id = create_warning_zone(
            body.name,
            body.geometry,
            warning_type=body.warning_type,
            severity=body.severity,
            notes=body.notes or "",
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    return get_warning_zone(zone_id)


@router.patch("/warnings/{zone_id}")
def update_warning_zone_endpoint(zone_id: int, body: WarningZoneUpdate):
    _raise_if_processing_active()
    _require_lgu_user()
    if get_warning_zone(zone_id) is None:
        raise HTTPException(status_code=404, detail="Warning zone not found")
    if zone_id < 0:
        raise HTTPException(status_code=409, detail='Automatic mortality warnings are updated from monitoring records and cannot be edited manually.')
    if body.geometry is not None:
        _require_zone_inside_map(body.geometry)
    try:
        return update_warning_zone(
            zone_id,
            name=body.name,
            polygon_geojson=body.geometry,
            warning_type=body.warning_type,
            severity=body.severity,
            notes=body.notes,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@router.delete("/warnings/{zone_id}")
def delete_warning_zone_endpoint(zone_id: int):
    _raise_if_processing_active()
    _require_lgu_user()
    if get_warning_zone(zone_id) is None:
        raise HTTPException(status_code=404, detail="Warning zone not found")
    if zone_id < 0:
        raise HTTPException(status_code=409, detail='Automatic mortality warnings remain while recorded site mortality is at least 30%. Correct the monitoring records to update them.')
    delete_warning_zone(zone_id)
    return {"status": "deleted", "id": zone_id}
