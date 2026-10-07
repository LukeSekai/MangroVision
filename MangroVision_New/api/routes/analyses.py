"""Analysis endpoints."""

import json
from typing import Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from planting_database import (
    count_nearby_points,
    delete_analysis,
    find_overlapping_analyses,
    find_repeat_image_analyses,
    get_analysis_area_context,
    get_analysis_asset_urls,
    get_all_stats,
    get_analysis_by_id,
    get_user_by_session_token,
    list_analysis_points,
    list_analysis_summaries,
)
from waypoint_export import generate_geojson

router = APIRouter()


def _require_lgu_user() -> dict:
    user = get_user_by_session_token("")
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired LGU session.")
    if str(user.get("role") or "").lower() not in {"admin", "lgu", "planner"}:
        raise HTTPException(status_code=403, detail="LGU/admin permission is required.")
    return user


def _empty_feature_collection(name: str) -> dict:
    return {"type": "FeatureCollection", "name": name, "features": []}


def _stored_json_object(value) -> dict:
    """Accept both native JSONB and the compatibility repository's JSON text."""
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except (TypeError, json.JSONDecodeError):
            return {}
    return value if isinstance(value, dict) else {}


@router.get("/stats")
def get_stats():
    _require_lgu_user()
    return get_all_stats()


# MIGRATED FROM app.py analysis history drawer
@router.get("/")
def list_analyses(include_previews: bool = False):
    _require_lgu_user()
    return list_analysis_summaries(include_previews=include_previews)


@router.get("/overlapping")
def overlapping(lat: float, lon: float, radius: float = 15.0):
    _require_lgu_user()
    return find_overlapping_analyses(lat, lon, radius)


@router.get("/nearby-points")
def nearby_points_count(lat: float, lon: float, radius: float = 15.0):
    _require_lgu_user()
    return {"count": count_nearby_points(lat, lon, radius)}


class AnalysisAreaRequest(BaseModel):
    footprint: dict
    image_name: Optional[str] = Field(default=None, max_length=255)
    latitude: Optional[float] = Field(default=None, ge=-90, le=90)
    longitude: Optional[float] = Field(default=None, ge=-180, le=180)
    source_image_sha256: Optional[str] = Field(default=None, pattern=r'^[a-f0-9]{64}$')
    source_original_sha256: Optional[str] = Field(default=None, pattern=r'^[a-f0-9]{64}$')


@router.post("/area-context")
def analysis_area_context(request: AnalysisAreaRequest):
    """Refresh the occupied-area advisory before starting image processing."""
    _require_lgu_user()
    try:
        context = get_analysis_area_context(request.footprint)
        context['repeat_analyses'] = find_repeat_image_analyses(
            request.image_name, request.latitude, request.longitude,
            request.source_image_sha256, request.source_original_sha256,
        ) if request.image_name else []
        return context
    except (ValueError, TypeError, KeyError) as error:
        raise HTTPException(status_code=400, detail="The image boundary could not be checked. Select the image again to verify its location.") from error
    except Exception as error:
        raise HTTPException(status_code=503, detail="Could not check existing planting data. Please try Run Analysis again.") from error


# MIGRATED FROM app.py analysis detail view (points)
@router.get("/{analysis_id}/points")
def analysis_points(analysis_id: int, only_unassigned: bool = False):
    _require_lgu_user()
    points = list_analysis_points(analysis_id, only_unassigned=only_unassigned)
    if not points:
        # Return an empty list (200) — distinguishing "no analysis" vs "no points" is upstream.
        return []
    return points


@router.get("/{analysis_id}")
def analysis_detail(analysis_id: int):
    """
    Full payload for one saved analysis, shaped like the /process-stream
    result so the shared Results Summary overlay can render it directly.
    Persisted image previews and the optional detail snapshot are replayed
    when available; older rows return nulls for fields that were not stored.
    """
    _require_lgu_user()
    row = get_analysis_by_id(analysis_id)
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")

    points = list_analysis_points(analysis_id)
    image_name = row.get("image_name") or "Saved Analysis"
    gsd_cm = row.get("gsd_cm") or 0
    gsd_m = float(gsd_cm) / 100.0
    total_area = float(row.get("total_area_m2") or 0)
    eroded_unavailable_count = sum(
        1 for point in points if point.get("eroded_unavailable")
    )
    available_safe_point_count = max(
        0,
        len(points) - eroded_unavailable_count,
    )
    saved_detail = _stored_json_object(row.get("analysis_detail_json"))
    # Replay the footprint from the SAME transform that produced the points.
    # Never replace it with a north-up bounding box or a hull of surviving
    # points: either would lose the orientation and the photo's edge setback.
    saved_footprint = (
        _stored_json_object(saved_detail.get("analysis_footprint"))
        or _stored_json_object(row.get("footprint_geojson"))
    )
    if saved_footprint.get("type") not in {"Polygon", "MultiPolygon"}:
        saved_footprint = None

    coordinates = [
        {
            "point_num": p["point_num"],
            "latitude": p["latitude"],
            "longitude": p["longitude"],
            "pixel_x": int(p.get("pixel_x") or 0),
            "pixel_y": int(p.get("pixel_y") or 0),
            "buffer_m": p.get("buffer_m"),
            "area_m2": p.get("area_m2"),
            "eroded_unavailable": bool(p.get("eroded_unavailable")),
            "availability_status": p.get("availability_status") or "available",
            "status_label": (
                "Not Available for Planting"
                if p.get("eroded_unavailable")
                else "Planned"
            ),
        }
        for p in points
    ]

    waypoints = [
        {
            "lat": p["latitude"],
            "lon": p["longitude"],
            "point_num": p["point_num"],
            "buffer_m": p.get("buffer_m"),
            "area_m2": p.get("area_m2"),
            "status": (
                "eroded_unavailable"
                if p.get("eroded_unavailable")
                else p.get("status")
            ),
            "eroded_unavailable": bool(p.get("eroded_unavailable")),
            "availability_reason": p.get("availability_reason"),
            "name": f"{image_name}-{int(p['point_num']):03d}",
        }
        for p in points
    ]
    detection_mode = saved_detail.get("detection_mode") or "ai"
    export_metadata = {
        "image_name": image_name,
        "analyzed_at": row.get("analyzed_at"),
        "detection_mode": detection_mode,
        "total_points": len(points),
    }
    safe_points_geojson = (
        json.loads(generate_geojson(waypoints, export_metadata))
        if waypoints
        else _empty_feature_collection("MangroVision Planting Points")
    )
    saved_metrics = saved_detail.get("metrics") if isinstance(saved_detail.get("metrics"), dict) else {}
    saved_metadata = saved_detail.get("metadata") if isinstance(saved_detail.get("metadata"), dict) else {}
    saved_parameters = saved_detail.get("parameters") if isinstance(saved_detail.get("parameters"), dict) else {}
    saved_match = saved_detail.get("map_match")
    stored_assets = get_analysis_asset_urls(analysis_id)

    metrics = {
        **saved_metrics,
        "canopy_count": row.get("canopy_count") or 0,
        "canopy_area_m2": row.get("canopy_area_m2"),
        "canopy_coverage_pct": row.get("canopy_coverage_pct"),
        "danger_area_m2": row.get("danger_area_m2") or 0,
        "danger_percentage": row.get("danger_pct") or 0,
        "plantable_area_m2": row.get("plantable_area_m2") or 0,
        "plantable_percentage": row.get("plantable_pct") or 0,
        "hexagon_count": row.get("hexagon_count") or 0,
        # Erosion availability is dynamic. Recompute this value from the
        # current active zones so deleting a treated erosion polygon makes its
        # retained orange points immediately count as safe/Planned again.
        "safe_hexagon_count": available_safe_point_count,
        "forbidden_filtered_count": row.get("forbidden_filtered") or 0,
        "eroded_filtered_count": row.get("eroded_filtered") or 0,
        "eroded_unavailable_count": eroded_unavailable_count,
        "gsd_m_per_pixel": gsd_m,
        "coverage_m": [row.get("coverage_w_m") or 0, row.get("coverage_h_m") or 0],
        "total_area_m2": total_area,
        "altitude_m": row.get("altitude_m"),
        "ai_confidence_threshold": saved_metrics.get("ai_confidence_threshold", row.get("ai_confidence")),
    }
    for optional_key in (
        "tile_count",
        "processing_time_sec",
        "model_name",
        "ai_avg_confidence",
        "ai_instance_count",
        "ai_below_confidence_detections",
        "ai_rescued_low_confidence_detections",
        "ai_rescued_shadow_crown_detections",
        "ai_rejected_low_saturation_detections",
        "ai_rejected_low_saturation_components",
        "ai_seedling_supplement_count",
        "ai_seedling_candidate_count",
        "ai_rejected_too_large_detections",
        "ai_rejected_non_canopy_class_detections",
        "duplicate_filtered_count",
        "spacing_filtered_count",
        "post_snap_danger_filtered_count",
        "clipped_outside_orthophoto",
        "orthophoto_canopy_filtered_count",
    ):
        metrics.setdefault(optional_key, None)

    metadata_payload = {
        **saved_metadata,
        "has_exif": saved_metadata.get("has_exif", row.get("center_lat") is not None),
        "has_gps": saved_metadata.get("has_gps", row.get("center_lat") is not None),
        "gps_valid": saved_metadata.get("gps_valid", row.get("center_lat") is not None),
        "image_center_lat": row.get("center_lat"),
        "image_center_lon": row.get("center_lon"),
    }

    return {
        "analysis_id": analysis_id,
        "analysis_key": None,
        "uploaded_file_name": image_name,
        "source_image_name": saved_detail.get("source_image_name") or image_name,
        "repeat_image": saved_detail.get("repeat_image"),
        "detection_mode": detection_mode,
        "parameters": saved_parameters,
        "inputs": {},
        "metadata": metadata_payload,
        "messages": {"info": [], "warnings": []},
        "overlaps": {"analyses": [], "nearby_points": 0},
        "metrics": metrics,
        "images": {
            "original_image_url": (stored_assets.get("original") or {}).get("url"),
            "visualization_image_url": (stored_assets.get("visualization") or {}).get("url"),
            "original_preview_url": (
                stored_assets.get("original_preview") or stored_assets.get("original") or {}
            ).get("url"),
            "visualization_preview_url": (
                stored_assets.get("visualization_preview")
                or stored_assets.get("visualization")
                or {}
            ).get("url"),
            "url_expires_in_seconds": next(
                (
                    asset.get("expires_in_seconds")
                    for asset in stored_assets.values()
                    if asset.get("expires_in_seconds")
                ),
                None,
            ),
        },
        "map": {
            "available": row.get("center_lat") is not None,
            "match": saved_match,
            "analysis_footprint": saved_footprint,
            "analysis_footprint_quality": (
                saved_detail.get("analysis_footprint_quality")
                or row.get("footprint_quality")
                or "missing"
            ),
            "image_center_feature": None,
            "safe_points_geojson": safe_points_geojson,
            "forbidden_filtered_geojson": None,
            "eroded_filtered_geojson": None,
            "post_snap_danger_filtered_geojson": None,
            "orthophoto_canopy_filtered_geojson": None,
            "spacing_filtered_geojson": None,
            "coordinates": coordinates,
        },
        "exports": {
            "waypoints": waypoints,
            "metadata": export_metadata,
            "json_results": None,
        },
        "can_save": False,
        "saved": True,
        "analyzed_at": row.get("analyzed_at"),
    }


@router.delete("/{analysis_id}")
def remove_analysis(analysis_id: int):
    _require_lgu_user()
    try:
        delete_analysis(analysis_id)
    except ValueError as error:
        raise HTTPException(status_code=409, detail=str(error)) from error
    return {"status": "deleted", "id": analysis_id}
