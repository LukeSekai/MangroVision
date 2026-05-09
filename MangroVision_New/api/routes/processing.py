"""Processing endpoints that expose the full app.py canopy-analysis workflow."""

import base64
import copy
import inspect
import io
import json
import os
import queue as queue_module
import sys
import threading
import time
import traceback
from collections import OrderedDict
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import cv2
import numpy as np
from fastapi import APIRouter, File, Form, HTTPException, UploadFile
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

# Fix Windows console encoding because canopy_detection prints Unicode text.
if sys.platform == "win32" and hasattr(sys.stdout, "buffer") and hasattr(sys.stderr, "buffer"):
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Ensure parent modules are importable.
_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))
if str(_ROOT / "canopy_detection") not in sys.path:
    sys.path.insert(0, str(_ROOT / "canopy_detection"))

from canopy_detector_hexagon import HexagonDetector
from exif_extractor import ExifExtractor
from forbidden_zone_filter import ForbiddenZoneFilter
import ortho_matcher  # imported as module so we can read live ORTHO_GSD globals
from ortho_matcher import (
    drone_pixel_to_gps_via_heading,
    drone_pixel_to_gps_via_homography,
    gps_to_ortho_pixel,
    ortho_pixel_to_gps,
    is_inside_any_orthophoto,
    match_drone_to_ortho,
)
from planting_database import (
    count_nearby_points,
    find_overlapping_analyses,
    get_saved_point_locations,
    save_analysis,
)
from waypoint_export import generate_geojson, hexagons_to_waypoints

router = APIRouter()

_FORBIDDEN_ZONE_CANDIDATES = (
    _ROOT / "forbidden_zone_final.geojson",
    _ROOT / "forbidden_zones_final.geojson",
    _ROOT / "forbidden_zones.geojson",
)
_FORBIDDEN_ZONES_PATH = next(
    (path for path in _FORBIDDEN_ZONE_CANDIDATES if path.exists()),
    _FORBIDDEN_ZONE_CANDIDATES[0],
)
_ERODED_ZONES_PATH = _ROOT / "eroded_zones.geojson"
_TEMP_UPLOADS_DIR = _ROOT / "MangroVision_New" / "temp_uploads"
_PROCESSING_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_PROCESSING_CACHE_LIMIT = 24


class SaveProcessedAnalysisRequest(BaseModel):
    analysis_key: str
    user_id: Optional[int] = None


# Safety buffer applied to forbidden polygons before the point-in-polygon
# check. Two reasons it is non-zero:
#   1. Drone-pixel-to-GPS projection has 1-3 m residual error at low SIFT
#      match confidence, so a hexagon visually on a forbidden structure can
#      project to a lat/lon just outside the polygon. A 2 m buffer absorbs
#      that error and prevents false-safe classifications around pavilions,
#      walkways, and other structures.
#   2. Independently of projection error, planting right against a man-made
#      structure is undesirable; a small enforced gap is good practice.
# Eroded zones do not need the same defensive buffer because they are
# wide-area sediment polygons and a 2 m halo would cost too much plantable area.
_FORBIDDEN_SAFETY_BUFFER_M = 2.0
_ERODED_SAFETY_BUFFER_M = 0.0


def _load_zone_filters() -> tuple[ForbiddenZoneFilter, ForbiddenZoneFilter]:
    """Load current forbidden and eroded zone filters from disk."""
    return (
        ForbiddenZoneFilter(
            str(_FORBIDDEN_ZONES_PATH),
            safety_buffer_m=_FORBIDDEN_SAFETY_BUFFER_M,
        ),
        ForbiddenZoneFilter(
            str(_ERODED_ZONES_PATH),
            safety_buffer_m=_ERODED_SAFETY_BUFFER_M,
        ),
    )


def _set_processing_cache(analysis_key: str, payload: dict[str, Any]) -> None:
    """Store recent processing results so the manual save flow matches app.py review/save behavior."""
    _PROCESSING_CACHE[analysis_key] = payload
    _PROCESSING_CACHE.move_to_end(analysis_key)
    while len(_PROCESSING_CACHE) > _PROCESSING_CACHE_LIMIT:
        _PROCESSING_CACHE.popitem(last=False)


def _get_processing_cache(analysis_key: str) -> Optional[dict[str, Any]]:
    """Return cached processing payload for a previously reviewed analysis."""
    payload = _PROCESSING_CACHE.get(analysis_key)
    if payload is not None:
        _PROCESSING_CACHE.move_to_end(analysis_key)
    return payload


def _encode_image_data_url(image: np.ndarray, extension: str = ".png") -> str:
    """Encode a BGR image array as a browser-ready data URL."""
    success, encoded = cv2.imencode(extension, image)
    if not success:
        raise ValueError("Could not encode image for response payload")
    mime = "image/png" if extension.lower() == ".png" else "image/jpeg"
    b64 = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:{mime};base64,{b64}"


def _empty_feature_collection(name: str) -> dict[str, Any]:
    """Return a minimal empty GeoJSON feature collection."""
    return {
        "type": "FeatureCollection",
        "name": name,
        "features": [],
    }


def _waypoints_to_geojson(waypoints: list[dict[str, Any]], metadata: dict[str, Any]) -> dict[str, Any]:
    """Convert waypoint rows into a GeoJSON object."""
    if not waypoints:
        return _empty_feature_collection("MangroVision Planting Points")
    return json.loads(generate_geojson(waypoints, metadata))


def _filtered_hexagons_to_geojson(
    hexagons: list[dict[str, Any]],
    reason: str,
    name: str,
) -> dict[str, Any]:
    """Serialize filtered planting points so React can preview excluded markers on the map."""
    features = []
    for index, hexagon in enumerate(hexagons, 1):
        lat = hexagon.get("_gps_lat")
        lon = hexagon.get("_gps_lon")
        if lat is None or lon is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [float(lon), float(lat)],
                },
                "properties": {
                    "name": f"{name} #{index}",
                    "reason": reason,
                    "buffer_m": hexagon.get("buffer_radius_m"),
                    "area_m2": hexagon.get("area_m2", hexagon.get("area_sqm", 0)),
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "name": name,
        "features": features,
    }


_ORTHO_CANOPY_RECHECK_RADIUS_M = 0.45
_ORTHO_CANOPY_STRONG_RATIO = 0.12


def _orthophoto_pixel_context(latitude: float, longitude: float) -> Optional[dict[str, Any]]:
    """Return the orthophoto image and pixel containing a GPS coordinate."""
    try:
        registry = ortho_matcher._get_ortho_registry()
    except Exception:
        return None

    for entry in registry:
        try:
            to_entry_crs, _ = ortho_matcher._get_entry_transformers(entry)
            entry_x, entry_y = to_entry_crs.transform(longitude, latitude)
            west, east, south, north = ortho_matcher._ortho_bounds_utm(entry)
            if not (west <= entry_x <= east and south <= entry_y <= north):
                continue

            px = int(round((entry_x - entry["origin_x"]) / entry["gsd_x"]))
            py = int(round((entry["origin_y"] - entry_y) / entry["gsd_y"]))

            key = str(entry["path"])
            ortho_image = ortho_matcher._ortho_cache.get(key)
            if ortho_image is None:
                ortho_image = ortho_matcher._read_orthophoto_bgr(Path(entry["path"]))
                ortho_matcher._ortho_cache[key] = ortho_image

            if 0 <= px < ortho_image.shape[1] and 0 <= py < ortho_image.shape[0]:
                return {
                    "entry": entry,
                    "image": ortho_image,
                    "px": px,
                    "py": py,
                    "gsd": (float(entry["gsd_x"]) + float(entry["gsd_y"])) / 2.0,
                }
        except Exception:
            continue

    return None


def _orthophoto_canopy_recheck(latitude: float, longitude: float) -> Optional[dict[str, Any]]:
    """Detect obvious canopy/vegetation under a projected map point."""
    context = _orthophoto_pixel_context(latitude, longitude)
    if context is None:
        return None

    ortho_image = context["image"]
    px = int(context["px"])
    py = int(context["py"])
    ortho_gsd = max(float(context["gsd"]), 1e-9)
    radius_px = max(3, int(round(_ORTHO_CANOPY_RECHECK_RADIUS_M / ortho_gsd)))

    y1 = max(0, py - radius_px)
    y2 = min(ortho_image.shape[0], py + radius_px + 1)
    x1 = max(0, px - radius_px)
    x2 = min(ortho_image.shape[1], px + radius_px + 1)
    patch = ortho_image[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    hue = hsv[:, :, 0]
    sat = hsv[:, :, 1]
    val = hsv[:, :, 2]

    bgr = patch.astype(np.float32)
    blue = bgr[:, :, 0]
    green = bgr[:, :, 1]
    red = bgr[:, :, 2]
    excess_green = (2.0 * green) - red - blue

    bright_green = (
        (hue >= 20)
        & (hue <= 92)
        & (sat >= 45)
        & (val >= 45)
        & (green > blue + 20.0)
        & (green >= red - 8.0)
        & (excess_green > 6.0)
    )
    shadow_green = (
        (hue >= 28)
        & (hue <= 100)
        & (sat >= 30)
        & (val >= 20)
        & (val <= 150)
        & (green > red + 4.0)
        & (green > blue + 12.0)
        & (excess_green > 10.0)
    )
    strong_green = (
        (hue >= 22)
        & (hue <= 90)
        & (sat >= 55)
        & (val >= 45)
        & (green > blue + 28.0)
        & (green >= red - 5.0)
        & (excess_green > 14.0)
    )

    valid = (blue + green + red) > 30.0
    valid_count = int(np.count_nonzero(valid))
    if valid_count <= 0:
        return None

    cy = py - y1
    cx = px - x1
    center_strong = bool(strong_green[cy, cx]) if 0 <= cy < strong_green.shape[0] and 0 <= cx < strong_green.shape[1] else False
    green_mask = (bright_green | shadow_green) & valid
    strong_mask = strong_green & valid
    green_ratio = float(np.count_nonzero(green_mask) / valid_count)
    strong_ratio = float(np.count_nonzero(strong_mask) / valid_count)
    is_canopy = bool(center_strong or strong_ratio >= _ORTHO_CANOPY_STRONG_RATIO)

    return {
        "is_canopy": is_canopy,
        "orthophoto": context["entry"].get("name"),
        "ortho_px": px,
        "ortho_py": py,
        "green_ratio": green_ratio,
        "strong_green_ratio": strong_ratio,
        "center_strong_green": center_strong,
    }


def _image_center_feature(
    latitude: Optional[float],
    longitude: Optional[float],
    properties: Optional[dict[str, Any]] = None,
) -> Optional[dict[str, Any]]:
    """Build a GeoJSON point feature for the uploaded image center."""
    if latitude is None or longitude is None:
        return None
    return {
        "type": "Feature",
        "geometry": {
            "type": "Point",
            "coordinates": [float(longitude), float(latitude)],
        },
        "properties": properties or {},
    }


def _sanitize_match_result(match_result: Optional[dict[str, Any]]) -> Optional[dict[str, Any]]:
    """Strip non-JSON homography matrices from the orthophoto match payload."""
    if match_result is None:
        return None
    def _json_float(value):
        try:
            return None if value is None else float(value)
        except (TypeError, ValueError):
            return None

    def _json_int(value):
        try:
            return None if value is None else int(value)
        except (TypeError, ValueError):
            return None

    return {
        "success": bool(match_result.get("success")),
        "heading": _json_float(match_result.get("heading")),
        "confidence": _json_float(match_result.get("confidence")),
        "inliers": _json_int(match_result.get("inliers")),
        "total_matches": _json_int(match_result.get("total_matches")),
        "ortho_name": match_result.get("ortho_name"),
        "ortho_path": match_result.get("ortho_path"),
        "match_score": _json_float(match_result.get("match_score")),
        "center_drift_px": _json_float(match_result.get("center_drift_px")),
        "center_drift_m": _json_float(match_result.get("center_drift_m")),
        "center_offset_east_m": _json_float(match_result.get("center_offset_east_m")),
        "center_offset_north_m": _json_float(match_result.get("center_offset_north_m")),
        "gps_anchored": bool(match_result.get("gps_anchored", False)),
        "projection_rebuilt": bool(match_result.get("projection_rebuilt", False)),
        "projection_rotation_source": match_result.get("projection_rotation_source"),
        "heading_diff_vs_exif_deg": _json_float(match_result.get("heading_diff_vs_exif_deg")),
        "rejected_reason": match_result.get("rejected_reason"),
        "error": match_result.get("error"),
    }


# Master switch for visual ortho-matching. When False, the SIFT-based
# `match_drone_to_ortho` step is skipped entirely and every drone image is
# projected onto the ortho using only GPS center + EXIF heading + GSD-derived
# scale (the heading-based fallback that already lives in
# `drone_pixel_to_gps_via_heading`). Use this when the visual matcher
# converges at low confidence and produces worse projections than the
# straight metric path would. Flipping this to True restores SIFT.
_USE_SIFT_MATCHING = True

# SIFT-vs-EXIF heading disagreement is recorded as telemetry but is NEVER
# used to reject a SIFT match. Real-world DJI flights routinely show 10–30°
# compass-vs-visual gaps because the magnetic compass drifts near metal,
# water, and shorelines. SIFT, matching against the orthophoto, is the more
# trustworthy source of orientation when it has converged. SIFT's own gates
# (inlier count, RANSAC confidence, center-drift cap) are sufficient to
# decide validity.


def _angular_diff(a: float, b: float) -> float:
    """Smallest signed angular difference (a - b) in degrees, in [-180, 180).

    Handles wrap-around so 359 vs 1 reports as -2, not 358.
    """
    return (float(a) - float(b) + 180.0) % 360.0 - 180.0


def _post_process_match(
    match_result: dict[str, Any],
    drone_image: np.ndarray,
    center_lat: float,
    center_lon: float,
    camera_heading: Optional[float],
    drone_gsd: float,
) -> dict[str, Any]:
    """Trust SIFT's homography unmodified when it succeeds.

    Earlier versions of this function GPS-anchored the SIFT homography
    (forcing the drone center to map to the EXIF GPS coordinate) and even
    rebuilt scale from physical GSD. Both "corrections" turned out to push
    the drone overlay AWAY from the structures it should land on whenever
    the orthophoto's georeference is offset from the drone's GPS — which is
    exactly the case in the project's WebODM-built map. The visual ground
    truth here is the orthophoto, not the GPS metadata.

    SIFT matches drone pixels directly against the orthophoto pixels, so its
    homography places the drone overlay on the same pixels the dashed
    forbidden polygons are drawn on. We now respect that placement.

    The function still records telemetry (heading disagreement vs EXIF, and
    SIFT's reported center drift relative to GPS) so it's visible in the
    response, but no fields on the homography itself are altered.
    """
    if not match_result.get("success"):
        return match_result

    sift_heading = match_result.get("heading")
    if camera_heading is not None and sift_heading is not None:
        match_result["heading_diff_vs_exif_deg"] = float(
            _angular_diff(sift_heading, camera_heading)
        )

    # The match_result already carries center_drift_* from `match_drone_to_ortho`.
    # Surface a flag that downstream code / UI can use to know the homography
    # is the raw SIFT one (not anchored or rebuilt).
    match_result["gps_anchored"] = False
    match_result["projection_rebuilt"] = False
    match_result["projection_rotation_source"] = "sift_raw"
    return match_result


def _match_drone_to_ortho_robust(
    drone_image: np.ndarray,
    center_lat: float,
    center_lon: float,
    drone_gsd: float,
    camera_heading: Optional[float] = None,
) -> dict[str, Any]:
    """Run SIFT matching with a wider retry before allowing heading fallback.

    On a successful SIFT match, the SIFT homography is REPLACED with a clean
    physics-based reconstruction. SIFT contributes only the rotation angle
    (when its confidence is high enough); scale and translation come from
    the exact drone-to-ortho GSD ratio and the GPS-tagged image center.
    This avoids letting low-confidence SIFT matches put a 10-20 m offset
    on the drone overlay because SIFT's noisy scale or translation got used.

    Heading disagreement between SIFT and EXIF is recorded as telemetry
    only; SIFT's own gates inside `match_drone_to_ortho` decide validity.

    When `_USE_SIFT_MATCHING` is False, the SIFT step is skipped entirely
    and a synthetic failure is returned, forcing the caller to use
    `drone_pixel_to_gps_via_heading` for projection — pure GPS + EXIF
    heading + GSD, no visual matching at all.
    """
    if not _USE_SIFT_MATCHING:
        return {
            "success": False,
            "rejected_reason": "sift_disabled_by_config",
            "error": "Visual ortho-matching is disabled. Using GPS + EXIF heading projection.",
        }

    result = match_drone_to_ortho(
        drone_image=drone_image,
        center_lat=center_lat,
        center_lon=center_lon,
        drone_gsd=drone_gsd,
    )
    if result.get("success"):
        return _post_process_match(
            result, drone_image, center_lat, center_lon, camera_heading, drone_gsd
        )

    retry = match_drone_to_ortho(
        drone_image=drone_image,
        center_lat=center_lat,
        center_lon=center_lon,
        drone_gsd=drone_gsd,
        margin_factor=2.2,
        max_features=12000,
    )
    if retry.get("success"):
        retry["retry_used"] = True
        retry["first_error"] = result.get("error")
        return _post_process_match(
            retry, drone_image, center_lat, center_lon, camera_heading, drone_gsd
        )
    retry["first_error"] = result.get("error")
    return retry


def _build_georeferenced_overlay(
    image: np.ndarray,
    homography: np.ndarray,
    max_dimension: int = 1800,
) -> Optional[dict[str, Any]]:
    """Warp the processed drone overlay into north-up orthophoto space for Leaflet."""
    if not isinstance(image, np.ndarray) or not isinstance(homography, np.ndarray):
        return None

    image_h, image_w = image.shape[:2]
    if image_h <= 0 or image_w <= 0:
        return None

    corners = np.array(
        [[[0, 0]], [[image_w, 0]], [[image_w, image_h]], [[0, image_h]]],
        dtype=np.float64,
    )
    try:
        ortho_corners = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    except Exception:
        return None

    if not np.all(np.isfinite(ortho_corners)):
        return None

    padding_px = 4
    min_x = int(np.floor(float(np.min(ortho_corners[:, 0])) - padding_px))
    max_x = int(np.ceil(float(np.max(ortho_corners[:, 0])) + padding_px))
    min_y = int(np.floor(float(np.min(ortho_corners[:, 1])) - padding_px))
    max_y = int(np.ceil(float(np.max(ortho_corners[:, 1])) + padding_px))

    out_w = max_x - min_x
    out_h = max_y - min_y
    if out_w <= 0 or out_h <= 0:
        return None

    # Guard against a bad homography ballooning the response payload.
    scale = min(1.0, max_dimension / max(out_w, out_h))
    warp_w = max(1, int(round(out_w * scale)))
    warp_h = max(1, int(round(out_h * scale)))

    crop_and_scale = np.array(
        [
            [scale, 0.0, -min_x * scale],
            [0.0, scale, -min_y * scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    warp_matrix = crop_and_scale @ homography

    try:
        warped = cv2.warpPerspective(
            image,
            warp_matrix,
            (warp_w, warp_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=(0, 0, 0),
        )
        alpha = cv2.warpPerspective(
            np.full((image_h, image_w), 255, dtype=np.uint8),
            warp_matrix,
            (warp_w, warp_h),
            flags=cv2.INTER_NEAREST,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=0,
        )
    except Exception:
        return None

    overlay_bgra = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)
    overlay_bgra[:, :, 3] = alpha

    north_lat, west_lon = ortho_pixel_to_gps(min_x, min_y)
    south_lat, east_lon = ortho_pixel_to_gps(max_x, max_y)

    return {
        "image_data_url": _encode_image_data_url(overlay_bgra, ".png"),
        "bounds": [
            [float(south_lat), float(west_lon)],
            [float(north_lat), float(east_lon)],
        ],
        "opacity": 0.72,
    }


def _safe_float(value: Any) -> Optional[float]:
    """Coerce a value to float when possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compute_canopy_area_m2(results: dict[str, Any]) -> float:
    """Canopy area in m² from the raw mask and GSD."""
    mask = results.get("canopy_mask")
    gsd = results.get("gsd_m_per_pixel")
    if not isinstance(mask, np.ndarray) or gsd is None:
        return 0.0
    return float(np.count_nonzero(mask) * (float(gsd) ** 2))


def _compute_canopy_coverage_pct(results: dict[str, Any]) -> float:
    """Percentage of the image covered by canopy."""
    canopy_area = _compute_canopy_area_m2(results)
    total_area = float(results.get("total_area_m2") or 0.0)
    if total_area <= 0:
        return 0.0
    return round((canopy_area / total_area) * 100.0, 2)


# MIGRATED FROM app.py: _gps_to_drone_pixel_via_homography lines 2160-2166
def _gps_to_drone_pixel_via_homography(latitude, longitude, h_inverse):
    """Map GPS coordinates into the drone image using the inverse ortho homography."""
    ortho_x, ortho_y = gps_to_ortho_pixel(latitude, longitude)
    pt = np.array([[[ortho_x, ortho_y]]], dtype=np.float64)
    drone_pt = cv2.perspectiveTransform(pt, h_inverse)
    px, py = drone_pt[0, 0]
    return float(px), float(py)


# MIGRATED FROM app.py: _gps_to_drone_pixel_via_heading lines 2169-2192
def _gps_to_drone_pixel_via_heading(
    latitude,
    longitude,
    image_w,
    image_h,
    center_lat,
    center_lon,
    gsd,
    heading_deg,
):
    """Approximate GPS to drone-pixel conversion when homography is unavailable."""
    mpdlat = 111320.0
    mpdlon = 111320.0 * np.cos(np.radians(center_lat))

    east_m = (longitude - center_lon) * mpdlon
    north_m = (latitude - center_lat) * mpdlat

    heading_rad = np.radians(float(heading_deg))
    offset_x_m = east_m * np.cos(heading_rad) - north_m * np.sin(heading_rad)
    offset_y_m = east_m * np.sin(heading_rad) + north_m * np.cos(heading_rad)

    px = (offset_x_m / gsd) + (image_w / 2.0)
    py = (image_h / 2.0) - (offset_y_m / gsd)
    return float(px), float(py)


# MIGRATED FROM app.py: _build_zone_mask_in_drone_pixels lines 2195-2226
def _build_zone_mask_in_drone_pixels(
    image_shape,
    zone_polygons,
    gps_to_pixel,
    max_polygon_area_fraction: float = 0.45,
    max_total_area_fraction: float = 0.70,
):
    """Rasterize lat/lon exclusion polygons into the current drone-image pixel space."""
    h, w = image_shape
    zone_mask = np.zeros((h, w), dtype=np.uint8)
    image_area = max(1, int(h) * int(w))
    skipped_large = 0

    for zone in zone_polygons or []:
        geoms = list(zone.geoms) if hasattr(zone, "geoms") else [zone]
        for geom in geoms:
            if geom.is_empty or not hasattr(geom, "exterior"):
                continue

            exterior_pts = []
            for lon, lat in geom.exterior.coords:
                px, py = gps_to_pixel(lat, lon)
                if np.isfinite(px) and np.isfinite(py):
                    exterior_pts.append([int(round(px)), int(round(py))])

            if len(exterior_pts) < 3:
                continue

            candidate_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.fillPoly(candidate_mask, [np.array(exterior_pts, dtype=np.int32)], 255)

            for interior in getattr(geom, "interiors", []):
                hole_pts = []
                for lon, lat in interior.coords:
                    px, py = gps_to_pixel(lat, lon)
                    if np.isfinite(px) and np.isfinite(py):
                        hole_pts.append([int(round(px)), int(round(py))])
                if len(hole_pts) >= 3:
                    cv2.fillPoly(candidate_mask, [np.array(hole_pts, dtype=np.int32)], 0)

            candidate_pixels = int(np.count_nonzero(candidate_mask))
            if candidate_pixels <= 0:
                continue
            if candidate_pixels / image_area > max_polygon_area_fraction:
                skipped_large += 1
                continue

            zone_mask = cv2.bitwise_or(zone_mask, candidate_mask)

    total_fraction = int(np.count_nonzero(zone_mask)) / image_area
    if total_fraction > max_total_area_fraction:
        print(
            "[Mangrovision] Ignoring projected forbidden-zone mask because it covers "
            f"{total_fraction:.0%} of the drone image.",
            flush=True,
        )
        return np.zeros((h, w), dtype=np.uint8)

    if skipped_large:
        print(
            "[Mangrovision] Skipped "
            f"{skipped_large} oversized projected forbidden-zone polygon(s).",
            flush=True,
        )

    return zone_mask


# MIGRATED FROM app.py: _apply_forbidden_zone_canopy_exclusion lines 2229-2282
def _apply_forbidden_zone_canopy_exclusion(detector, results, forbidden_mask):
    """Remove canopy pixels inside structural forbidden zones and rebuild the dependent outputs.

    Two-stage exclusion so that user-traced forbidden polygons (which are
    typically slightly tighter than the actual man-made structure) still
    catch the canopy on the structure's edges:

      1. Dilate the forbidden mask by ~half the canopy buffer width so a
         few pixels of slop around the polygon get treated as forbidden too.
      2. After re-extracting polygons, drop any whole polygon whose area
         is ≥30 % inside the original (un-dilated) forbidden mask — these
         are slivers extending out of a structure that survived the pixel
         mask but conceptually still belong to the forbidden zone.
    """
    canopy_mask = results.get("canopy_mask")
    if not isinstance(canopy_mask, np.ndarray) or canopy_mask.shape != forbidden_mask.shape:
        results["_forbidden_canopy_removed_pixels"] = 0
        return results

    gsd = float(results.get("gsd_m_per_pixel") or 0.05) or 0.05
    buffer_m = float(results.get("canopy_buffer_m") or 1.0) or 1.0
    dilation_px = max(1, int(round(0.5 * buffer_m / gsd)))
    kernel_size = 2 * dilation_px + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    expanded_forbidden = cv2.dilate(forbidden_mask, kernel)

    filtered_canopy_mask = canopy_mask.copy()
    filtered_canopy_mask[expanded_forbidden > 0] = 0

    removed_pixels = int(np.count_nonzero(canopy_mask) - np.count_nonzero(filtered_canopy_mask))
    results["_forbidden_canopy_removed_pixels"] = removed_pixels
    if removed_pixels <= 0:
        return results

    previous_canopy_count = int(results.get("canopy_count", 0))
    canopy_polygons = detector.mask_to_polygons(filtered_canopy_mask, min_area_m2=0.04)

    # Stage 2: drop whole polygons that still overlap the forbidden zone.
    if canopy_polygons and forbidden_mask.any():
        forbidden_bool = forbidden_mask > 0
        kept_polygons = []
        for poly in canopy_polygons:
            try:
                if poly.is_empty or poly.exterior is None:
                    continue
                pts = np.array(poly.exterior.coords, dtype=np.int32)
                poly_mask = np.zeros_like(forbidden_mask)
                cv2.fillPoly(poly_mask, [pts], 1)
                poly_area_px = int(poly_mask.sum())
                if poly_area_px <= 0:
                    continue
                overlap_px = int(np.count_nonzero(poly_mask & forbidden_bool))
                if overlap_px / poly_area_px >= 0.30:
                    # Erase remaining canopy pixels for this polygon too.
                    cv2.fillPoly(filtered_canopy_mask, [pts], 0)
                    continue
                kept_polygons.append(poly)
            except Exception:
                kept_polygons.append(poly)
        canopy_polygons = kept_polygons
        rebuilt_mask = np.zeros_like(filtered_canopy_mask)
        for poly in canopy_polygons:
            try:
                if poly.is_empty or poly.exterior is None:
                    continue
                pts = np.array(poly.exterior.coords, dtype=np.int32)
                cv2.fillPoly(rebuilt_mask, [pts], 255)
            except Exception:
                continue
        filtered_canopy_mask = rebuilt_mask
    danger_zone, danger_mask = detector.create_danger_zones(
        canopy_polygons,
        filtered_canopy_mask,
        results["canopy_buffer_m"],
    )
    plantable_zone = detector.identify_plantable_zones(danger_zone)
    hexagons = detector.generate_hexagonal_planting_zones(
        plantable_zone,
        results["hexagon_size_m"],
        maximize_coverage=True,
        danger_mask=danger_mask,
        canopy_mask=filtered_canopy_mask,
    )

    total_area_m2 = float(results.get("total_area_m2", 0.0))
    gsd = float(results["gsd_m_per_pixel"])
    danger_area_m2 = danger_zone.area * (gsd ** 2) if not danger_zone.is_empty else 0.0
    plantable_area_m2 = plantable_zone.area * (gsd ** 2) if not plantable_zone.is_empty else 0.0

    results["canopy_polygons"] = canopy_polygons
    results["canopy_mask"] = filtered_canopy_mask
    results["canopy_count"] = len(canopy_polygons)
    results["danger_zone"] = danger_zone
    results["danger_mask"] = danger_mask
    results["danger_area_m2"] = danger_area_m2
    results["danger_percentage"] = (danger_area_m2 / total_area_m2 * 100) if total_area_m2 > 0 else 0.0
    results["plantable_zone"] = plantable_zone
    results["plantable_area_m2"] = plantable_area_m2
    results["plantable_percentage"] = (plantable_area_m2 / total_area_m2 * 100) if total_area_m2 > 0 else 0.0
    results["hexagons"] = hexagons
    results["hexagon_count"] = len(hexagons)
    results["_forbidden_canopy_removed_count"] = max(0, previous_canopy_count - len(canopy_polygons))

    ai_metadata = dict(results.get("ai_metadata") or {})
    ai_metadata["forbidden_zone_canopy_removed_pixels"] = removed_pixels
    results["ai_metadata"] = ai_metadata
    return results


def _emit(progress_cb: Optional[Any], stage: str, pct: int, **extra: Any) -> None:
    """Forward a progress event to the SSE queue when one is provided."""
    if progress_cb is None:
        return
    try:
        progress_cb({"type": "progress", "stage": stage, "pct": pct, **extra})
    except Exception:
        # Progress reporting must never break the actual pipeline.
        pass


# FIXED: previously the FastAPI route called detector.process_image(...) without
# a progress_callback, so the tile_setup / tile_progress / tile_done events that
# ProperDetectree2Detector already emits fell on the floor. The UI sat frozen at
# the 25% checkpoint for the entire 10-15 minute inference. This bridge
# subscribes to those detector events and forwards them to the SSE stream and
# stdout, restoring the progress contract between the original Mangrovision
# backend and the Mangrovision_New API.
def _make_detector_progress_bridge(
    progress_cb: Optional[Any],
    tile_band_start: int = 26,
    tile_band_end: int = 54,
):
    """Return a detector progress_callback that maps tile events to SSE + stdout."""
    state = {"total": 0}

    def _bridge(event: str, payload: dict[str, Any]) -> None:
        total = int(payload.get("total_tiles") or state["total"] or 0)
        current = int(payload.get("current_tile") or 0)
        if payload.get("total_tiles"):
            state["total"] = total

        if event == "tile_setup":
            _emit(
                progress_cb,
                f"Preparing {total} tiles for AI inference",
                tile_band_start,
                total_tiles=total,
                current_tile=0,
            )
            print(
                f"[Mangrovision] Tile setup — {total} tiles queued for inference",
                flush=True,
            )
            return

        if event == "tile_done":
            if total > 0:
                fraction = current / total
                pct = int(round(tile_band_start + (tile_band_end - tile_band_start) * fraction))
                pct = max(tile_band_start, min(tile_band_end, pct))
                stage = f"Analyzing tile {current} of {total} — {pct}% complete"
            else:
                pct = tile_band_start
                stage = f"Analyzing tile {current}..."
            _emit(
                progress_cb,
                stage,
                pct,
                current_tile=current,
                total_tiles=total,
            )
            # One stdout line per tile, flushed immediately so the terminal shows
            # live progress rather than a buffered dump at the end.
            if total > 0:
                print(
                    f"[Mangrovision] Tile {current}/{total} processed — {pct}% complete",
                    flush=True,
                )
            else:
                print(
                    f"[Mangrovision] Tile {current} processed",
                    flush=True,
                )
            return

        if event == "tile_complete":
            _emit(
                progress_cb,
                "AI inference complete, cleaning up detections",
                tile_band_end,
                total_tiles=total,
                current_tile=total,
            )
            print(
                f"[Mangrovision] AI inference finished — {payload.get('final_trees', 0)} trees retained",
                flush=True,
            )

    return _bridge


def _execute_canopy_workflow(
    temp_path: Path,
    uploaded_name: str,
    altitude: float,
    drone_model: str,
    canopy_buffer: float,
    hexagon_size: float,
    ai_confidence: float,
    detection_mode: Optional[str],
    ai_runtime_tuning: str,
    progress_cb: Optional[Any] = None,
) -> dict[str, Any]:
    """Run the full canopy-analysis workflow on an already-saved drone image.

    Returns the JSON-serializable response payload. When a progress callback
    is supplied, emits stage/percent events so SSE clients can show real
    pipeline progress instead of a simulated timer. The caller is responsible
    for cleaning up temp_path; exceptions raised here propagate unwrapped.
    """
    info_messages: list[str] = []
    warning_messages: list[str] = []
    workflow_start_time = time.time()

    _emit(progress_cb, "Checking AI availability", 2)
    # MIGRATED FROM app.py: main lines 4703-4710
    try:
        from detectree2_proper import ProperDetectree2Detector  # noqa: F401

        ai_available = True
    except ImportError:
        ai_available = False

    if detection_mode not in {"ai", "hsv"}:
        detection_mode = "ai" if ai_available else "hsv"
    elif detection_mode == "ai" and not ai_available:
        detection_mode = "hsv"
        warning_messages.append("AI detector is unavailable, so the workflow fell back to HSV detection.")

    try:
        ai_runtime_tuning_dict = json.loads(ai_runtime_tuning or "{}")
        if not isinstance(ai_runtime_tuning_dict, dict):
            ai_runtime_tuning_dict = {}
    except json.JSONDecodeError:
        ai_runtime_tuning_dict = {}
        warning_messages.append("Invalid ai_runtime_tuning payload was ignored.")

    _emit(progress_cb, "Reading EXIF metadata", 6)
    # MIGRATED FROM app.py: analyze_image lines 5038-5121
    metadata = ExifExtractor.extract_all_metadata(str(temp_path))
    image_gps = None
    image_center_lat = None
    image_center_lon = None
    altitude_to_use = altitude
    drone_to_use = drone_model
    gps_valid = False
    camera_heading = 0.0
    heading_source = "Default (North)"
    overlaps: list[dict[str, Any]] = []
    nearby_points = 0

    if metadata.get("has_gps"):
        gps = metadata["gps"]
        image_center_lat = gps["latitude"]
        image_center_lon = gps["longitude"]

        try:
            gps_inside_ortho = is_inside_any_orthophoto(image_center_lat, image_center_lon)
        except Exception:
            gps_inside_ortho = False

        if gps_inside_ortho:
            gps_valid = True
            image_gps = gps
            info_messages.append(
                f"GPS found inside map bounds: {image_center_lat:.6f}, {image_center_lon:.6f}."
            )

            overlaps = find_overlapping_analyses(image_center_lat, image_center_lon)
            if overlaps:
                nearby_points = count_nearby_points(image_center_lat, image_center_lon)
                warning_messages.append(
                    f"This area already has planting data nearby: {len(overlaps)} saved analyses and {nearby_points} saved planting points."
                )
        else:
            image_gps = gps
            warning_messages.append(
                f"GPS found outside the orthophoto bounds: {image_center_lat:.6f}, {image_center_lon:.6f}."
            )

        if "relative_altitude" in gps and gps["relative_altitude"] is not None:
            altitude_to_use = gps["relative_altitude"]
            info_messages.append(f"Using detected altitude: {altitude_to_use:.1f} m (AGL from DJI XMP).")
        elif "altitude" in gps and gps["altitude"] is not None:
            altitude_to_use = gps["altitude"]
            info_messages.append(f"Using detected altitude: {altitude_to_use:.1f} m (MSL from EXIF).")

        if metadata.get("camera"):
            drone_to_use = ExifExtractor.detect_drone_model(metadata["camera"])
            info_messages.append(f"Detected drone: {drone_to_use.replace('_', ' ')}.")

        if gps.get("heading") is not None:
            camera_heading = float(gps["heading"])
            heading_source = gps.get("heading_source") or "EXIF GPSImgDirection"
        info_messages.append(f"Using heading {camera_heading:.1f}° ({heading_source}).")
    else:
        warning_messages.append("No GPS data found in the image. Geotagged map outputs and database save are unavailable.")

    # MIGRATED FROM app.py: analyze_image lines 5129-5154
    try:
        canopy_code_mtime = int((_ROOT / "canopy_detection" / "canopy_detector_hexagon.py").stat().st_mtime)
    except Exception:
        canopy_code_mtime = 0
    try:
        proper_code_mtime = int((_ROOT / "canopy_detection" / "detectree2_proper.py").stat().st_mtime)
    except Exception:
        proper_code_mtime = 0
    try:
        ortho_code_mtime = int((_ROOT / "canopy_detection" / "ortho_matcher.py").stat().st_mtime)
    except Exception:
        ortho_code_mtime = 0
    try:
        forbidden_zones_mtime = int(_FORBIDDEN_ZONES_PATH.stat().st_mtime)
    except Exception:
        forbidden_zones_mtime = 0
    try:
        eroded_zones_mtime = int(_ERODED_ZONES_PATH.stat().st_mtime)
    except Exception:
        eroded_zones_mtime = 0

    tuning_fingerprint = json.dumps(ai_runtime_tuning_dict or {}, sort_keys=True)
    analysis_key = (
        f"{uploaded_name}_{altitude_to_use}_{drone_to_use}_{canopy_buffer}_"
        f"{hexagon_size}_{ai_confidence}_{detection_mode}_"
        f"{canopy_code_mtime}_{proper_code_mtime}_{ortho_code_mtime}_"
        f"{forbidden_zones_mtime}_{eroded_zones_mtime}_"
        f"{tuning_fingerprint}"
    )

    _emit(progress_cb, f"Initializing {detection_mode.upper()} detector", 12)
    # MIGRATED FROM app.py: analyze_image lines 5158-5276
    detector = HexagonDetector(
        altitude_m=altitude_to_use,
        drone_model=drone_to_use,
        ai_confidence=ai_confidence,
        detection_mode=detection_mode,
        camera_info=metadata.get("camera") or {},
    )

    if (
        ai_runtime_tuning_dict
        and getattr(detector, "ai_detector", None) is not None
        and hasattr(detector.ai_detector, "set_runtime_tuning")
    ):
        try:
            set_tuning_fn = detector.ai_detector.set_runtime_tuning
            accepted = set(inspect.signature(set_tuning_fn).parameters.keys())
            tuned_kwargs = {
                key: value for key, value in ai_runtime_tuning_dict.items() if key in accepted
            }
            if tuned_kwargs:
                set_tuning_fn(**tuned_kwargs)
        except Exception as tuning_error:
            warning_messages.append(
                f"Runtime tuning values could not be fully applied: {tuning_error}"
            )

    _emit(progress_cb, "Detecting canopy and generating planting zones", 25)
    # FIXED: the detector already streams per-tile progress; we now subscribe so
    # the UI and terminal move during the long inference phase instead of
    # freezing at 25% for the full 10-15 minutes of tiling + prediction.
    detector_progress_bridge = _make_detector_progress_bridge(progress_cb)
    results = detector.process_image(
        image_path=str(temp_path),
        canopy_buffer_m=canopy_buffer,
        hexagon_size_m=hexagon_size,
        progress_callback=detector_progress_bridge,
    )

    _emit(progress_cb, "Loading forbidden and eroded zone filters", 55)
    forbidden_filter, eroded_filter = _load_zone_filters()
    match_result = None
    results["_forbidden_filtered"] = 0
    results["_eroded_filtered"] = 0
    results["_forbidden_canopy_removed_pixels"] = 0
    results["_forbidden_canopy_removed_count"] = 0

    # MIGRATED FROM app.py: analyze_image lines 5294-5382
    if image_gps is not None and (forbidden_filter.zone_count > 0 or eroded_filter.zone_count > 0):
        _emit(progress_cb, "Matching drone image to orthophoto", 62)
        _gsd = results["gsd_m_per_pixel"]
        _w, _h = results["image_size"]

        match_result = _match_drone_to_ortho_robust(
            drone_image=results["image"],
            center_lat=image_center_lat,
            center_lon=image_center_lon,
            drone_gsd=_gsd,
            camera_heading=camera_heading,
        )

        if match_result.get("success"):
            _H = match_result["H"]

            def _px_to_gps(px, py):
                return drone_pixel_to_gps_via_homography(px, py, _H)

            try:
                _H_inv = np.linalg.inv(_H)

                def _gps_to_px(lat, lon):
                    return _gps_to_drone_pixel_via_homography(lat, lon, _H_inv)

            except Exception:

                def _gps_to_px(lat, lon):
                    return _gps_to_drone_pixel_via_heading(
                        lat,
                        lon,
                        _w,
                        _h,
                        image_center_lat,
                        image_center_lon,
                        _gsd,
                        camera_heading,
                    )

        else:

            def _px_to_gps(px, py):
                return drone_pixel_to_gps_via_heading(
                    px,
                    py,
                    _w,
                    _h,
                    image_center_lat,
                    image_center_lon,
                    _gsd,
                    camera_heading,
                )

            def _gps_to_px(lat, lon):
                return _gps_to_drone_pixel_via_heading(
                    lat,
                    lon,
                    _w,
                    _h,
                    image_center_lat,
                    image_center_lon,
                    _gsd,
                    camera_heading,
                )

        if forbidden_filter.zone_count > 0:
            _emit(progress_cb, "Removing canopy inside forbidden zones", 68)
            _forbidden_mask = _build_zone_mask_in_drone_pixels(
                results["image"].shape[:2],
                forbidden_filter.forbidden_polygons,
                _gps_to_px,
            )
            results = _apply_forbidden_zone_canopy_exclusion(
                detector,
                results,
                _forbidden_mask,
            )

        _emit(progress_cb, "Filtering planting points against zones", 74)
        _safe = []
        _forbidden_hexes = []
        _eroded_hexes = []
        for _hex in results["hexagons"]:
            _px, _py = _hex["center"]
            _lat, _lon = _px_to_gps(_px, _py)
            _hex["_gps_lat"] = _lat
            _hex["_gps_lon"] = _lon
            if not forbidden_filter.is_safe_location(_lat, _lon):
                _forbidden_hexes.append(_hex)
            elif not eroded_filter.is_safe_location(_lat, _lon):
                _eroded_hexes.append(_hex)
            else:
                _safe.append(_hex)

        # Drop hexagons that would land on top of already-saved planting
        # points in the DB, so re-analyzing the same area doesn't stack
        # redundant markers. Uses ~1 m tolerance (roughly one hexagon).
        _duplicate_hexes: list = []
        if _safe:
            _DUP_TOL_DEG = 0.00001  # ~1.1 m at equator
            _lats = [h["_gps_lat"] for h in _safe]
            _lons = [h["_gps_lon"] for h in _safe]
            _saved_pts = get_saved_point_locations(
                min(_lats) - _DUP_TOL_DEG,
                max(_lats) + _DUP_TOL_DEG,
                min(_lons) - _DUP_TOL_DEG,
                max(_lons) + _DUP_TOL_DEG,
            )
            if _saved_pts:
                _kept: list = []
                for _hex in _safe:
                    _h_lat = _hex["_gps_lat"]
                    _h_lon = _hex["_gps_lon"]
                    _is_dup = any(
                        abs(_h_lat - _slat) <= _DUP_TOL_DEG
                        and abs(_h_lon - _slon) <= _DUP_TOL_DEG
                        for _slat, _slon in _saved_pts
                    )
                    if _is_dup:
                        _duplicate_hexes.append(_hex)
                    else:
                        _kept.append(_hex)
                _safe = _kept
                if _duplicate_hexes:
                    info_messages.append(
                        f"{len(_duplicate_hexes)} planting points were skipped because saved points already exist at those locations."
                    )

        results["hexagons"] = _safe
        results["hexagon_count"] = len(_safe)
        results["_forbidden_filtered"] = len(_forbidden_hexes)
        results["_eroded_filtered"] = len(_eroded_hexes)
        results["_duplicate_filtered"] = len(_duplicate_hexes)
        results["_forbidden_hexagons"] = _forbidden_hexes
        results["_eroded_hexagons"] = _eroded_hexes
        results["_duplicate_hexagons"] = _duplicate_hexes

    # MIGRATED FROM app.py: analyze_image lines 5600-5937
    _emit(progress_cb, "Projecting planting points to GPS", 85)
    map_image_gps = image_gps
    map_center_lat = image_center_lat
    map_center_lon = image_center_lon
    width, height = results["image_size"]
    safe_hexagons = [copy.deepcopy(hexagon) for hexagon in results.get("hexagons", [])]
    forbidden_hexagons = [copy.deepcopy(hexagon) for hexagon in results.get("_forbidden_hexagons", [])]
    eroded_hexagons = [copy.deepcopy(hexagon) for hexagon in results.get("_eroded_hexagons", [])]
    orthophoto_canopy_hexagons: list[dict[str, Any]] = []
    forbidden_filtered_count = int(results.get("_forbidden_filtered", 0) or 0)
    eroded_filtered_count = int(results.get("_eroded_filtered", 0) or 0)
    orthophoto_canopy_filtered_count = 0
    clipped_outside_orthophoto = 0
    detected_heading = camera_heading
    analysis_overlay = None

    if map_image_gps is not None:
        gsd = results["gsd_m_per_pixel"]

        if match_result is None:
            match_result = _match_drone_to_ortho_robust(
                drone_image=results["image"],
                center_lat=map_center_lat,
                center_lon=map_center_lon,
                drone_gsd=gsd,
                camera_heading=camera_heading,
            )

        if match_result.get("success"):
            H_matrix = match_result["H"]
            detected_heading = float(match_result.get("heading") or camera_heading)

            def pixel_to_latlon(px, py):
                return drone_pixel_to_gps_via_homography(px, py, H_matrix)

        else:
            warning_messages.append(
                f"Auto-alignment not available ({match_result.get('error')}). Using heading-based fallback ({camera_heading:.1f}°)."
            )

            def pixel_to_latlon(px, py):
                return drone_pixel_to_gps_via_heading(
                    px,
                    py,
                    width,
                    height,
                    map_center_lat,
                    map_center_lon,
                    gsd,
                    camera_heading,
                )

        for hexagon in safe_hexagons:
            if "_gps_lat" not in hexagon:
                px, py = hexagon["center"]
                lat, lon = pixel_to_latlon(px, py)
                hexagon["_gps_lat"] = lat
                hexagon["_gps_lon"] = lon

        for hexagon in forbidden_hexagons:
            if "_gps_lat" not in hexagon:
                px, py = hexagon["center"]
                lat, lon = pixel_to_latlon(px, py)
                hexagon["_gps_lat"] = lat
                hexagon["_gps_lon"] = lon

        for hexagon in eroded_hexagons:
            if "_gps_lat" not in hexagon:
                px, py = hexagon["center"]
                lat, lon = pixel_to_latlon(px, py)
                hexagon["_gps_lat"] = lat
                hexagon["_gps_lon"] = lon

        before_clip = len(safe_hexagons)
        safe_hexagons = [
            hexagon
            for hexagon in safe_hexagons
            if is_inside_any_orthophoto(hexagon["_gps_lat"], hexagon["_gps_lon"])
        ]
        clipped_outside_orthophoto = before_clip - len(safe_hexagons)
        if clipped_outside_orthophoto > 0:
            info_messages.append(
                f"{clipped_outside_orthophoto} planting points were removed because they fall outside orthophoto coverage."
            )

        if safe_hexagons:
            rechecked_hexagons = []
            for hexagon in safe_hexagons:
                canopy_recheck = _orthophoto_canopy_recheck(
                    float(hexagon["_gps_lat"]),
                    float(hexagon["_gps_lon"]),
                )
                if canopy_recheck and canopy_recheck.get("is_canopy"):
                    hexagon["_orthophoto_canopy_recheck"] = canopy_recheck
                    orthophoto_canopy_hexagons.append(hexagon)
                else:
                    rechecked_hexagons.append(hexagon)
            safe_hexagons = rechecked_hexagons
            orthophoto_canopy_filtered_count = len(orthophoto_canopy_hexagons)
            if orthophoto_canopy_filtered_count > 0:
                info_messages.append(
                    f"{orthophoto_canopy_filtered_count} planting points were removed because they landed on canopy in the orthophoto recheck."
                )

    results["hexagons"] = safe_hexagons
    results["hexagon_count"] = len(safe_hexagons)

    # Render after final map/export filtering so the image-space preview shows
    # the same planting-point set as coordinates, exports, and map markers.
    _emit(progress_cb, "Rendering visualization overlay", 90)
    visualization_results = dict(results)
    visualization_results["hexagons"] = safe_hexagons
    visualization_results["hexagon_count"] = len(safe_hexagons)
    visualization_image = detector.visualize_results(visualization_results)

    if analysis_overlay is None and map_image_gps is not None and match_result and match_result.get("success"):
        analysis_overlay = _build_georeferenced_overlay(visualization_image, match_result["H"])

    _emit(progress_cb, "Building coordinate table and exports", 92)
    coordinate_rows = []
    for index, hexagon in enumerate(safe_hexagons, 1):
        coordinate_rows.append(
            {
                "point_num": index,
                "latitude": round(float(hexagon["_gps_lat"]), 7),
                "longitude": round(float(hexagon["_gps_lon"]), 7),
                "pixel_x": int(hexagon["center"][0]),
                "pixel_y": int(hexagon["center"][1]),
                "buffer_m": _safe_float(hexagon.get("buffer_radius_m")),
                "area_m2": round(float(hexagon.get("area_m2", hexagon.get("area_sqm", 0))), 2),
            }
        )

    export_metadata = {
        "image_name": uploaded_name,
        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
        "detection_mode": detection_mode,
        "total_points": len(safe_hexagons),
    }
    waypoints = hexagons_to_waypoints(safe_hexagons, image_name=uploaded_name)
    safe_points_geojson = _waypoints_to_geojson(waypoints, export_metadata)
    forbidden_filtered_geojson = _filtered_hexagons_to_geojson(
        forbidden_hexagons,
        "Inside forbidden zone",
        "Filtered Forbidden Points",
    )
    eroded_filtered_geojson = _filtered_hexagons_to_geojson(
        eroded_hexagons,
        "Inside eroded zone",
        "Filtered Eroded Points",
    )
    orthophoto_canopy_filtered_geojson = _filtered_hexagons_to_geojson(
        orthophoto_canopy_hexagons,
        "Overlaps canopy in orthophoto recheck",
        "Filtered Orthophoto Canopy Points",
    )

    image_center_feature = _image_center_feature(
        map_center_lat,
        map_center_lon,
        {
            "name": "Image Center",
            "altitude_m": altitude_to_use,
            "gsd_cm": round(results["gsd_m_per_pixel"] * 100, 3),
            "heading": round(float(detected_heading), 1),
        },
    )

    json_results = {
        "canopy_count": results["canopy_count"],
        "danger_area_m2": results["danger_area_m2"],
        "plantable_area_m2": results["plantable_area_m2"],
        "hexagon_count": results["hexagon_count"],
        "gsd": results["gsd_m_per_pixel"],
        "coverage": results["coverage_m"],
    }

    can_save = bool(map_image_gps is not None and map_center_lat is not None and map_center_lon is not None)
    ai_meta = results.get("ai_metadata") or {}

    _emit(progress_cb, "Encoding preview images", 97)
    response_payload = {
        "status": "success",
        "analysis_key": analysis_key,
        "uploaded_file_name": uploaded_name,
        "detection_mode": detection_mode,
        "ai_available": ai_available,
        "parameters": {
            "altitude": altitude,
            "drone_model": drone_model,
            "canopy_buffer": canopy_buffer,
            "hexagon_size": hexagon_size,
            "ai_confidence": ai_confidence,
            "ai_runtime_tuning": ai_runtime_tuning_dict,
            "altitude_to_use": altitude_to_use,
            "drone_to_use": drone_to_use,
        },
        "metadata": {
            "has_exif": bool(metadata.get("has_exif")),
            "has_gps": bool(metadata.get("has_gps")),
            "gps_valid": gps_valid,
            "image_center_lat": map_center_lat,
            "image_center_lon": map_center_lon,
            "heading_source": heading_source,
            "camera_heading": camera_heading,
            "detected_heading": detected_heading,
            "camera": metadata.get("camera"),
            "xmp": metadata.get("xmp"),
        },
        "messages": {
            "info": info_messages,
            "warnings": warning_messages,
        },
        "overlaps": {
            "analyses": overlaps,
            "nearby_points": nearby_points,
        },
        "metrics": {
            "canopy_count": results["canopy_count"],
            "canopy_area_m2": _compute_canopy_area_m2(results),
            "canopy_coverage_pct": _compute_canopy_coverage_pct(results),
            "danger_area_m2": results["danger_area_m2"],
            "danger_percentage": results["danger_percentage"],
            "plantable_area_m2": results["plantable_area_m2"],
            "plantable_percentage": results["plantable_percentage"],
            "hexagon_count": results["hexagon_count"],
            "safe_hexagon_count": len(safe_hexagons),
            "forbidden_filtered_count": forbidden_filtered_count,
            "eroded_filtered_count": eroded_filtered_count,
            "orthophoto_canopy_filtered_count": orthophoto_canopy_filtered_count,
            "duplicate_filtered_count": int(results.get("_duplicate_filtered", 0) or 0),
            "clipped_outside_orthophoto": clipped_outside_orthophoto,
            "forbidden_canopy_removed_pixels": int(results.get("_forbidden_canopy_removed_pixels", 0) or 0),
            "forbidden_canopy_removed_count": int(results.get("_forbidden_canopy_removed_count", 0) or 0),
            "gsd_m_per_pixel": results["gsd_m_per_pixel"],
            "gsd_specs": results.get("gsd_specs"),
            "coverage_m": results["coverage_m"],
            "total_area_m2": results["total_area_m2"],
            "altitude_m": results["altitude_m"],
            "tile_count": int(ai_meta.get("num_tiles_processed", 0) or 0),
            "processing_time_sec": round(time.time() - workflow_start_time, 2),
            "model_name": ai_meta.get("model_name"),
            "ai_confidence_threshold": float(ai_confidence),
            "ai_avg_confidence": _safe_float(ai_meta.get("avg_confidence")),
            "ai_instance_count": int(ai_meta.get("instance_tree_count", results["canopy_count"]) or 0),
            "ai_raw_detections": int(ai_meta.get("raw_detections", 0) or 0),
            "ai_model_candidate_detections": int(ai_meta.get("model_candidate_detections", 0) or 0),
            "ai_below_confidence_detections": int(ai_meta.get("below_confidence_detections", 0) or 0),
            "ai_rejected_non_green_detections": int(ai_meta.get("rejected_non_green_detections", 0) or 0),
            "ai_rejected_too_large_detections": int(ai_meta.get("rejected_too_large_detections", 0) or 0),
            "ai_rejected_too_small_detections": int(ai_meta.get("rejected_too_small_detections", 0) or 0),
            "ai_coverage_component_count": int(ai_meta.get("coverage_component_count", results["canopy_count"]) or 0),
            "ai_canopy_merge_added_m2": _safe_float(
                (float(ai_meta.get("canopy_merge_added_pixels", 0) or 0) * (results["gsd_m_per_pixel"] ** 2))
            ),
            "ai_canopy_merge_gap_m": _safe_float(ai_meta.get("canopy_merge_gap_m")),
            "ai_max_crown_m2": _safe_float(ai_meta.get("max_crown_m2")),
        },
        "images": {
            "original_data_url": _encode_image_data_url(results["image"]),
            "visualization_data_url": _encode_image_data_url(visualization_image),
        },
        "map": {
            "available": bool(map_image_gps is not None),
            "match": _sanitize_match_result(match_result),
            "image_center_feature": image_center_feature,
            "analysis_overlay": analysis_overlay,
            "safe_points_geojson": safe_points_geojson,
            "forbidden_filtered_geojson": forbidden_filtered_geojson,
            "eroded_filtered_geojson": eroded_filtered_geojson,
            "orthophoto_canopy_filtered_geojson": orthophoto_canopy_filtered_geojson,
            "coordinates": coordinate_rows,
        },
        "exports": {
            "waypoints": waypoints,
            "metadata": export_metadata,
            "json_results": json_results,
        },
        "can_save": can_save,
        "saved": False,
    }

    _set_processing_cache(
        analysis_key,
        {
            "analysis_key": analysis_key,
            "image_name": uploaded_name,
            "results": results,
            "safe_hexagons": safe_hexagons,
            "center_lat": map_center_lat,
            "center_lon": map_center_lon,
            "response_payload": response_payload,
        },
    )

    _emit(progress_cb, "Analysis complete", 100)
    return response_payload


async def _save_upload_to_temp(image: UploadFile) -> tuple[Path, str]:
    """Persist an uploaded drone image to the temp uploads dir and return its path."""
    _TEMP_UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
    uploaded_name = image.filename or "uploaded_image"
    temp_path = _TEMP_UPLOADS_DIR / f"{int(time.time() * 1000)}_{uploaded_name}"
    contents = await image.read()
    with open(temp_path, "wb") as temp_file:
        temp_file.write(contents)
    return temp_path, uploaded_name


def _cleanup_temp(temp_path: Path) -> None:
    """Best-effort removal of an uploaded temp file."""
    try:
        if temp_path.exists():
            temp_path.unlink()
    except Exception:
        pass


@router.post("/process")
async def process_image(
    image: UploadFile = File(...),
    altitude: float = Form(6.0),
    drone_model: str = Form("Autel_EVO_II_Pro"),
    canopy_buffer: float = Form(2.0),
    hexagon_size: float = Form(1.5),
    ai_confidence: float = Form(0.87),
    detection_mode: Optional[str] = Form(None),
    ai_runtime_tuning: str = Form("{}"),
):
    """Run the canopy-analysis workflow and return the full JSON response."""
    temp_path, uploaded_name = await _save_upload_to_temp(image)
    try:
        return _execute_canopy_workflow(
            temp_path=temp_path,
            uploaded_name=uploaded_name,
            altitude=altitude,
            drone_model=drone_model,
            canopy_buffer=canopy_buffer,
            hexagon_size=hexagon_size,
            ai_confidence=ai_confidence,
            detection_mode=detection_mode,
            ai_runtime_tuning=ai_runtime_tuning,
        )
    except HTTPException:
        raise
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=f"Processing failed: {error}") from error
    finally:
        _cleanup_temp(temp_path)


@router.post("/process-stream")
async def process_image_stream(
    image: UploadFile = File(...),
    altitude: float = Form(6.0),
    drone_model: str = Form("Autel_EVO_II_Pro"),
    canopy_buffer: float = Form(2.0),
    hexagon_size: float = Form(1.5),
    ai_confidence: float = Form(0.87),
    detection_mode: Optional[str] = Form(None),
    ai_runtime_tuning: str = Form("{}"),
):
    """Stream real pipeline progress as Server-Sent Events.

    Event types yielded as `data: {json}\\n\\n` lines:
      * {"type": "progress", "stage": str, "pct": int}  — periodic checkpoints
      * {"type": "result",   "payload": {...}}          — terminal success
      * {"type": "error",    "detail": str}             — terminal failure
    The stream closes after the terminal event.
    """
    temp_path, uploaded_name = await _save_upload_to_temp(image)
    events: queue_module.Queue = queue_module.Queue()

    def progress_cb(event: dict[str, Any]) -> None:
        events.put(event)

    def runner() -> None:
        try:
            payload = _execute_canopy_workflow(
                temp_path=temp_path,
                uploaded_name=uploaded_name,
                altitude=altitude,
                drone_model=drone_model,
                canopy_buffer=canopy_buffer,
                hexagon_size=hexagon_size,
                ai_confidence=ai_confidence,
                detection_mode=detection_mode,
                ai_runtime_tuning=ai_runtime_tuning,
                progress_cb=progress_cb,
            )
            events.put({"type": "result", "payload": payload})
        except Exception as error:
            traceback.print_exc()
            events.put({"type": "error", "detail": f"Processing failed: {error}"})
        finally:
            events.put({"type": "__done__"})

    thread = threading.Thread(target=runner, daemon=True)
    thread.start()

    def event_generator():
        try:
            while True:
                event = events.get()
                if event.get("type") == "__done__":
                    break
                yield f"data: {json.dumps(event)}\n\n"
                if event.get("type") in {"result", "error"}:
                    break
        finally:
            _cleanup_temp(temp_path)

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.post("/save")
def save_processed_analysis(body: SaveProcessedAnalysisRequest):
    """Persist a reviewed analysis after the React client confirms the results."""
    cached_payload = _get_processing_cache(body.analysis_key)
    if cached_payload is None:
        raise HTTPException(
            status_code=404,
            detail="Processing result not found. Run the analysis again before saving.",
        )

    center_lat = cached_payload.get("center_lat")
    center_lon = cached_payload.get("center_lon")
    if center_lat is None or center_lon is None:
        raise HTTPException(
            status_code=400,
            detail="This analysis has no GPS center, so it cannot be saved to the database.",
        )

    cached_images = (cached_payload.get("response_payload") or {}).get("images") or {}

    try:
        analysis_id, new_points, skipped = save_analysis(
            image_name=cached_payload["image_name"],
            center_lat=center_lat,
            center_lon=center_lon,
            results=cached_payload["results"],
            hexagons=cached_payload["safe_hexagons"],
            user_id=body.user_id,
            original_image=cached_images.get("original_data_url"),
            visualization_image=cached_images.get("visualization_data_url"),
        )
    except Exception as error:
        raise HTTPException(status_code=500, detail=f"Could not save analysis: {error}") from error

    cached_payload["response_payload"]["saved"] = True
    cached_payload["response_payload"]["analysis_id"] = analysis_id
    cached_payload["response_payload"]["save_summary"] = {
        "analysis_id": analysis_id,
        "new_points": new_points,
        "skipped_duplicates": skipped,
    }

    return {
        "status": "saved",
        "analysis_id": analysis_id,
        "new_points": new_points,
        "skipped_duplicates": skipped,
    }
