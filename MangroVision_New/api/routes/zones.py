"""Zone data endpoints — forbidden, eroded, and site zones."""

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any, Optional

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
)

router = APIRouter()

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
def save_eroded_zones(body: SaveErodedBody):
    _raise_if_processing_active()
    user = _require_lgu_user()
    try:
        collection = replace_zone_collection("eroded", body.features, int(user["id"]))
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
def add_eroded_zone(feature: dict):
    """Create one erosion zone with a stable database ID."""
    _raise_if_processing_active()
    user = _require_lgu_user()
    try:
        saved = create_zone("eroded", feature, int(user["id"]))
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
    invalidate_eroded_zone_cache()
    return {"status": "added", "zone": saved, "id": saved["id"]}


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
    delete_warning_zone(zone_id)
    return {"status": "deleted", "id": zone_id}
