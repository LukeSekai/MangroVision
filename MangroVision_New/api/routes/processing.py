"""Processing endpoints that expose the full app.py canopy-analysis workflow."""

import base64
import copy
import hashlib
import inspect
import io
import json
import math
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
from shapely.geometry import Point, mapping, shape
from shapely.ops import unary_union

from api.runtime_state import processing_job

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
from gsd_calculator import GSDCalculator
import ortho_matcher  # imported as module so we can read live ORTHO_GSD globals
from ortho_matcher import (
    drone_pixel_to_gps_via_heading,
    drone_pixel_to_gps_via_homography,
    gps_to_ortho_pixel,
    ortho_pixel_to_gps,
    is_inside_any_orthophoto,
    is_inside_supported_map_bounds,
    match_drone_to_ortho,
)
from canopy_detection.orthophoto_coverage import (
    overlay_visibility_mask, point_visibility_flags, visible_gis_coverage,
)
from planting_database import (
    OutsideVisibleMapError,
    count_nearby_points,
    find_overlapping_analyses,
    get_analysis_by_id,
    get_saved_species_point_locations,
    get_user_by_session_token,
    save_analysis,
)
from mangrovision_db.config import get_settings
from mangrovision_db.storage import signed_download_url, upload_analysis_data_urls
from mangrovision_db.zones import (
    feature_collection,
    project_site_context,
    zone_revision_fingerprint,
)
from waypoint_export import generate_geojson, hexagons_to_waypoints

router = APIRouter()


def _require_lgu_user() -> dict:
    user = get_user_by_session_token("")
    if not user:
        raise HTTPException(status_code=401, detail="Invalid or expired LGU session.")
    if str(user.get("role") or "").lower() not in {"admin", "lgu", "planner"}:
        raise HTTPException(status_code=403, detail="LGU/admin permission is required.")
    return user

_FORBIDDEN_ZONE_CANDIDATES = (
    _ROOT / "new_forbidden.geojson",
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
_PREFLIGHT_ALIGNMENT_CACHE: "OrderedDict[str, dict[str, Any]]" = OrderedDict()
_PREFLIGHT_ALIGNMENT_CACHE_LIMIT = 24


class SaveProcessedAnalysisRequest(BaseModel):
    analysis_key: str
    user_id: Optional[int] = None


# Safety buffer applied to forbidden polygons before the point-in-polygon
# check. Two reasons it is non-zero:
#   1. Drone-pixel-to-GPS projection has 1-3 m residual error at low SIFT
#      match confidence, so a hexagon visually on a forbidden structure can
#      project to a lat/lon just outside the polygon. The buffer absorbs that
#      error and prevents false-safe classifications around pavilions,
#      walkways, and other structures.
#   2. Independently of projection error, planting right against a man-made
#      structure is undesirable; a small enforced gap is good practice.
# This safety distance is independent from species spacing. Bungalon can use
# 1 m planting points, but danger/forbidden buffers stay at 2 m unless the
# actual danger-buffer input is changed by the user.
# Eroded polygons remain visible as a GIS layer, but they are also hard
# exclusions for newly generated planting points. A point inside either a
# forbidden or eroded polygon must never reach the preview, export, or save
# payload.
_FORBIDDEN_SAFETY_BUFFER_M = 2.0
_ERODED_SAFETY_BUFFER_M = 0.0
_FORBIDDEN_CANOPY_EXCLUSION_ENABLED = (
    os.getenv("MANGROVISION_FORBIDDEN_CANOPY_EXCLUSION", "").strip().lower()
    in {"1", "true", "yes", "on"}
)


def _load_zone_filters() -> tuple[ForbiddenZoneFilter, ForbiddenZoneFilter]:
    """Load structural and erosion exclusions from PostGIS.

    The forbidden safety buffer is intentionally not species-scaled; species
    only controls planting-point spacing. Eroded polygons use their exact
    boundary (no additional halo) and exclude every covered planting point.
    """
    return (
        ForbiddenZoneFilter(
            safety_buffer_m=_FORBIDDEN_SAFETY_BUFFER_M,
            geojson_data=feature_collection("forbidden"),
        ),
        ForbiddenZoneFilter(
            safety_buffer_m=_ERODED_SAFETY_BUFFER_M,
            geojson_data=feature_collection("eroded"),
        ),
    )


def _load_gis_coverage_geometry() -> Any:
    """Intersect the GIS zone with pixels actually visible on the map."""
    return visible_gis_coverage(
        ortho_matcher._ensure_active_ortho(),
        feature_collection("gis_coverage").get("features", []),
    )


def _visible_map_point_features(collection: Any, coverage: Any) -> Any:
    """Keep map preview markers over visible orthophoto pixels only."""
    if not isinstance(collection, dict):
        return collection
    kept = []
    for feature in collection.get("features") or []:
        geometry = feature.get("geometry") or {}
        coordinates = geometry.get("coordinates") or []
        if geometry.get("type") != "Point" or len(coordinates) < 2:
            continue
        try:
            if coverage.covers(Point(float(coordinates[0]), float(coordinates[1]))):
                kept.append(feature)
        except (TypeError, ValueError):
            continue
    return {**collection, "features": kept}


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


def _pop_processing_cache(analysis_key: str) -> Optional[dict[str, Any]]:
    """Consume a reviewed processing result so it cannot be saved twice."""
    return _PROCESSING_CACHE.pop(analysis_key, None)


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _set_preflight_alignment_cache(file_digest: str, match: dict[str, Any]) -> None:
    cached = {
        key: copy.deepcopy(value)
        for key, value in match.items()
        if key not in {"H", "raw_H", "similarity_H", "projective_H", "validated_H"}
    }
    _PREFLIGHT_ALIGNMENT_CACHE[file_digest] = cached
    _PREFLIGHT_ALIGNMENT_CACHE.move_to_end(file_digest)
    while len(_PREFLIGHT_ALIGNMENT_CACHE) > _PREFLIGHT_ALIGNMENT_CACHE_LIMIT:
        _PREFLIGHT_ALIGNMENT_CACHE.popitem(last=False)


def _get_preflight_alignment_cache(file_digest: str) -> Optional[dict[str, Any]]:
    cached = _PREFLIGHT_ALIGNMENT_CACHE.get(file_digest)
    if cached is None:
        return None
    _PREFLIGHT_ALIGNMENT_CACHE.move_to_end(file_digest)
    return copy.deepcopy(cached)


def _analysis_storage_prefix(analysis_key: str) -> str:
    """Map a descriptive cache key to a fixed-length S3-safe object prefix."""
    digest = hashlib.sha256(analysis_key.encode("utf-8")).hexdigest()
    return f"previews/{digest}"


def _encode_image_data_url(
    image: np.ndarray,
    extension: str = ".png",
    *,
    jpeg_quality: int = 94,
) -> str:
    """Encode a BGR image array as a browser-ready data URL.

    Analysis photos are stored as high-quality JPEGs so a normal drone image
    does not expand into a 15-25 MiB PNG before it is sent to private storage.
    PNG remains available for transparent overlays and other lossless output.
    """
    normalized_extension = extension.lower()
    encode_params: list[int] = []
    if normalized_extension in {".jpg", ".jpeg"}:
        encode_params = [cv2.IMWRITE_JPEG_QUALITY, max(1, min(100, jpeg_quality))]
    success, encoded = cv2.imencode(extension, image, encode_params)
    if not success:
        raise ValueError("Could not encode image for response payload")
    mime = "image/png" if normalized_extension == ".png" else "image/jpeg"
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
_ORTHO_CANOPY_RECHECK_MAX_RADIUS_M = 0.75
_ORTHO_CANOPY_STRONG_RATIO = 0.22
_ORTHO_CANOPY_MIN_CLUSTER_M2 = 0.04
_RAW_MATCH_MAX_DRIFT_FOR_COORDS_M = 2.0
_RAW_MATCH_MIN_CONFIDENCE_FOR_COORDS = 0.55
_RAW_MATCH_MAX_SCALE_ERROR_RATIO = 0.06
_RAW_MATCH_STRONG_CONSTRAINED_MIN_CONFIDENCE = 0.35
_RAW_MATCH_BROAD_SUPPORT_MIN_INLIERS = 110
_RAW_MATCH_BROAD_SUPPORT_MIN_CONFIDENCE = 0.30
_RAW_MATCH_BROAD_SUPPORT_MIN_HULL_RATIO = 0.20
_RAW_MATCH_BROAD_SUPPORT_MIN_SPAN_RATIO = 0.45
_RAW_MATCH_BROAD_SUPPORT_MIN_GRID_CELLS = 6
# Altitude-derived GSD is only an estimate. A constrained visual match with
# substantial feature support, an accurate heading, and low center drift is a
# better image-to-map registration even when its observed footprint differs by
# a moderate amount from that estimate. This keeps map coordinates and the
# raster preview in one observed coordinate system without trusting weak or
# heading-inconsistent matches.
_RAW_MATCH_STRONG_CONSTRAINED_MAX_SCALE_ERROR_RATIO = 0.18
_RAW_MATCH_LOW_CONF_HEADING_DIFF_DEG = 20.0
_FORCE_METRIC_VISUAL_SCALE = False
_EDGE_REFINE_ENABLED = False
_EDGE_REFINE_MAX_SHIFT_M = 4.0
_EDGE_REFINE_MAX_DIM_PX = 520
_EDGE_REFINE_MIN_IMPROVEMENT = 0.02
# When SIFT cannot establish a trustworthy match, a small rigid refinement is
# still useful for low-texture coastal frames. It keeps the physical GSD fixed,
# permits only a tightly bounded correction around the DJI heading, and then
# corrects the common offset between the aircraft GPS tag and the actual
# ground footprint. These extra gates are intentionally stricter than the
# generic edge-refinement gate so a weak shoreline/texture coincidence cannot
# move saved planting points.
_FAILED_MATCH_EDGE_MIN_SCORE = 0.45
_FAILED_MATCH_EDGE_MIN_IMPROVEMENT = 0.03
_FAILED_MATCH_HEADING_SEARCH_DEG = 2.0
_FAILED_MATCH_HEADING_STEP_DEG = 0.5
_FAILED_MATCH_HEADING_MIN_SCORE_GAIN = 0.006
# When feature matching fails on a mostly uniform water/mud image, use the
# large shoreline/vegetation silhouette to refine only the EXIF heading. The
# GPS centre and physical GSD remain authoritative. Strict score/improvement
# gates keep this fallback from inventing a rotation on weak or uniform scenes.
_VEGETATION_HEADING_MAX_DIM_PX = 1000
_VEGETATION_HEADING_MIN_SCORE = 0.30
_VEGETATION_HEADING_MIN_IMPROVEMENT = 0.06
_VEGETATION_HEADING_MIN_MASK_FRACTION = 0.003
_VEGETATION_HEADING_MAX_MASK_FRACTION = 0.55
_VEGETATION_SCALE_MIN_FACTOR = 0.60
_VEGETATION_SCALE_MAX_FACTOR = 1.10
_VEGETATION_SCALE_COARSE_STEP = 0.05
_VEGETATION_SCALE_FINE_STEP = 0.01
_MATCH_WARNING_MAX_DRIFT_M = 0.75
_MATCH_WARNING_MIN_CONFIDENCE = 0.65
_MIN_ANALYSIS_ALTITUDE_M = 0.5
_MAX_ANALYSIS_ALTITUDE_M = 500.0
_MAX_CANOPY_BUFFER_M = 100.0
_MIN_HEXAGON_SIZE_M = 0.05
_MAX_HEXAGON_SIZE_M = 100.0
_M_PER_DEG_LAT = 111_320.0
# Planting points always need this much fully observed photo area between their
# centre and the uploaded raster edge. A registered orthophoto may verify the
# rest of a larger canopy buffer, but it must never erase this photo-edge gap.
_IMAGE_EDGE_MIN_CLEARANCE_M = 1.0
# Factor = sqrt(3) ≈ 1.732 so the edge-share (flat-top) lattice nearest-
# neighbour distance equals the species' target spacing exactly:
# hexagon_size R = target / sqrt(3), neighbour distance = R * sqrt(3) = target.
# This matches the gap-free tessellation in canopy_detector_hexagon.py
# (h_spacing = 1.5*R between columns, v_spacing = sqrt(3)*R within column).
_FINAL_POINT_SPACING_FACTOR = math.sqrt(3.0)
_FINAL_DUPLICATE_SPACING_RATIO = 0.92
_CROSS_SPECIES_MIN_SPACING_M = 2.0


def _local_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    ref_lat = (float(lat1) + float(lat2)) / 2.0
    m_per_deg_lon = _M_PER_DEG_LAT * max(0.2, math.cos(math.radians(ref_lat)))
    d_lat_m = (float(lat1) - float(lat2)) * _M_PER_DEG_LAT
    d_lon_m = (float(lon1) - float(lon2)) * m_per_deg_lon
    return math.sqrt((d_lat_m * d_lat_m) + (d_lon_m * d_lon_m))


def _normalize_species_key(species: Optional[str]) -> Optional[str]:
    key = (species or "").strip().lower()
    return key if key in SPECIES_SPACING_M else None


def _species_spacing_m(species: Optional[str], fallback: Any = None) -> Optional[float]:
    try:
        if fallback is not None:
            value = float(fallback)
            if value > 0:
                return value
    except (TypeError, ValueError):
        pass
    key = _normalize_species_key(species)
    return SPECIES_SPACING_M.get(key) if key else None


def _cross_species_threshold_m(
    current_species: Optional[str],
    current_spacing_m: Optional[float],
    existing_species: Optional[str],
    existing_spacing_m: Optional[float],
) -> Optional[float]:
    current_key = _normalize_species_key(current_species)
    existing_key = _normalize_species_key(existing_species)
    if current_key is None or existing_key is None or current_key == existing_key:
        return None

    distances = [
        distance
        for distance in (current_spacing_m, existing_spacing_m)
        if distance is not None and distance > 0
    ]
    return max(_CROSS_SPECIES_MIN_SPACING_M, max(distances) if distances else 0.0)


def _thin_hexagons_against_saved_species_points(
    hexagons: list[dict[str, Any]],
    species: Optional[str],
    spacing_m: Optional[float],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Remove preview/export candidates that violate saved cross-species spacing."""
    species_key = _normalize_species_key(species)
    if species_key is None or not hexagons:
        return list(hexagons), []

    gps_hexagons = [
        hexagon
        for hexagon in hexagons
        if hexagon.get("_gps_lat") is not None and hexagon.get("_gps_lon") is not None
    ]
    if not gps_hexagons:
        return list(hexagons), []

    current_spacing_m = _species_spacing_m(species_key, spacing_m)
    max_spacing_m = max(_CROSS_SPECIES_MIN_SPACING_M, current_spacing_m or 0.0)
    for other_species, other_spacing in SPECIES_SPACING_M.items():
        if other_species != species_key:
            max_spacing_m = max(max_spacing_m, other_spacing)
    bbox_pad_deg = (max_spacing_m / _M_PER_DEG_LAT) * 2.0
    lats = [float(hexagon["_gps_lat"]) for hexagon in gps_hexagons]
    lons = [float(hexagon["_gps_lon"]) for hexagon in gps_hexagons]

    try:
        saved_points = get_saved_species_point_locations(
            min(lats) - bbox_pad_deg,
            max(lats) + bbox_pad_deg,
            min(lons) - bbox_pad_deg,
            max(lons) + bbox_pad_deg,
        )
    except Exception:
        return list(hexagons), []

    conflicting_saved_points = []
    for saved in saved_points:
        saved_species = _normalize_species_key(saved.get("species"))
        if saved_species is None or saved_species == species_key:
            continue
        conflicting_saved_points.append(
            {
                "latitude": float(saved["latitude"]),
                "longitude": float(saved["longitude"]),
                "species": saved_species,
                "spacing_m": _species_spacing_m(
                    saved_species,
                    saved.get("planting_distance_m"),
                ),
            }
        )
    if not conflicting_saved_points:
        return list(hexagons), []

    kept: list[dict[str, Any]] = []
    filtered: list[dict[str, Any]] = []
    for hexagon in hexagons:
        lat = hexagon.get("_gps_lat")
        lon = hexagon.get("_gps_lon")
        if lat is None or lon is None:
            kept.append(hexagon)
            continue

        too_close = False
        for saved in conflicting_saved_points:
            threshold_m = _cross_species_threshold_m(
                species_key,
                current_spacing_m,
                saved["species"],
                saved["spacing_m"],
            )
            if threshold_m is None:
                continue
            if _local_distance_m(lat, lon, saved["latitude"], saved["longitude"]) < threshold_m:
                too_close = True
                break

        if too_close:
            filtered.append(hexagon)
        else:
            kept.append(hexagon)

    return kept, filtered


def _thin_hexagons_by_gps_spacing(
    hexagons: list[dict[str, Any]],
    min_spacing_m: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Final GPS-space duplicate guard after ortho matching/projection.

    The image-space generator creates a regular non-overlapping lattice, but
    orthophoto correction, snapped/global lattice fallback, and raw GPS fallback
    can move a few final coordinates onto the same practical spot. This guard
    removes only those near-duplicates so it does not punch small holes into
    otherwise valid planting rows.
    """
    if not hexagons or min_spacing_m <= 0:
        return list(hexagons), []

    gps_hexagons = [
        (idx, hexagon)
        for idx, hexagon in enumerate(hexagons)
        if hexagon.get("_gps_lat") is not None and hexagon.get("_gps_lon") is not None
    ]
    if len(gps_hexagons) < 2:
        return list(hexagons), []

    ref_lat = sum(float(hexagon["_gps_lat"]) for _, hexagon in gps_hexagons) / len(gps_hexagons)
    m_per_deg_lon = _M_PER_DEG_LAT * max(0.2, math.cos(math.radians(ref_lat)))
    # The detector and shared GPS lattice already enforce the biological
    # spacing. This final pass is mainly a guard for raw fallback points whose
    # projection moved them too close to an existing node; exact lattice
    # neighbours are exempted by snap_key below.
    effective_min_spacing_m = float(min_spacing_m) * _FINAL_DUPLICATE_SPACING_RATIO
    min_spacing_sq = effective_min_spacing_m * effective_min_spacing_m

    def priority(item: tuple[int, dict[str, Any]]) -> tuple[int, int]:
        idx, hexagon = item
        # Prefer regular snapped/global-lattice points when a raw fallback point
        # is too close to another candidate.
        source = str(hexagon.get("_gps_projection_source") or "")
        source_rank = 1 if source == "raw" else 0
        return (source_rank, idx)

    kept_indexed: list[tuple[int, dict[str, Any]]] = []
    filtered: list[dict[str, Any]] = []
    kept_xy: list[tuple[float, float, bool, Optional[tuple[int, int]]]] = []

    for original_idx, hexagon in sorted(enumerate(hexagons), key=priority):
        lat = hexagon.get("_gps_lat")
        lon = hexagon.get("_gps_lon")
        if lat is None or lon is None:
            kept_indexed.append((original_idx, hexagon))
            continue

        source = str(hexagon.get("_gps_projection_source") or "")
        snap_key = hexagon.get("_snap_key")
        lattice_stable = source != "raw" and isinstance(snap_key, tuple)
        x_m = float(lon) * m_per_deg_lon
        y_m = float(lat) * _M_PER_DEG_LAT
        too_close = False
        for kept_x, kept_y, kept_lattice_stable, kept_snap_key in kept_xy:
            if lattice_stable and kept_lattice_stable and snap_key != kept_snap_key:
                continue
            dx = x_m - kept_x
            dy = y_m - kept_y
            if (dx * dx) + (dy * dy) < min_spacing_sq:
                too_close = True
                break

        if too_close:
            filtered.append(hexagon)
            continue

        kept_indexed.append((original_idx, hexagon))
        kept_xy.append((x_m, y_m, lattice_stable, snap_key if lattice_stable else None))

    return [hexagon for _, hexagon in sorted(kept_indexed, key=lambda item: item[0])], filtered


def _match_quality_warning(match_result: Optional[dict[str, Any]]) -> Optional[str]:
    if not match_result or not match_result.get("success"):
        return None

    confidence = match_result.get("confidence")
    drift_m = match_result.get("center_drift_m")
    scale_error_pct = match_result.get("projection_scale_error_pct")
    scale_ratio = match_result.get("projection_scale_ratio")
    rebuilt = bool(match_result.get("projection_rebuilt"))
    low_conf = confidence is not None and float(confidence) < _MATCH_WARNING_MIN_CONFIDENCE
    high_drift = drift_m is not None and float(drift_m) > _MATCH_WARNING_MAX_DRIFT_M
    high_scale_error = (
        scale_error_pct is not None
        and float(scale_error_pct) > (_RAW_MATCH_MAX_SCALE_ERROR_RATIO * 100.0)
    )
    if not (rebuilt or low_conf or high_drift or high_scale_error):
        return None

    details: list[str] = []
    if confidence is not None:
        details.append(f"{float(confidence):.0%} confidence")
    if drift_m is not None:
        details.append(f"{float(drift_m):.2f} m center drift")
    if scale_ratio is not None and scale_error_pct is not None:
        details.append(
            f"{float(scale_ratio):.2f}x footprint scale ({float(scale_error_pct):.0f}% off)"
        )
    detail_text = ", ".join(details) if details else "approximate match quality"
    if rebuilt:
        return (
            f"Orthophoto match is approximate ({detail_text}); GPS-anchored projection was used for export coordinates."
        )
    return (
        f"Orthophoto match is approximate ({detail_text}); visual SIFT alignment was used, so verify the overlay before export."
    )


_OUTSIDE_GIS_MAP_DETAIL = (
    "Invalid image: its footprint is outside the visible GIS map area. Analysis was not started. "
    "Upload a geotagged image captured within the mapped project area."
)
_GIS_MAP_CHECK_FAILED_DETAIL = (
    "The GIS map boundary could not be verified, so analysis was not started. "
    "Please check the configured map data and try again."
)


def _display_homography_for_match(
    match_result: Optional[dict[str, Any]],
) -> Optional[np.ndarray]:
    """Return the same transform that owns the exported coordinates.

    Using a raw visual match for the preview after coordinate export has fallen
    back to a metric transform creates two incompatible footprints: valid map
    points can move onto structures or outside the image. ``H`` is therefore
    authoritative for both the map overlay and the raster preview. Older raw
    keys remain as defensive fallbacks for incomplete cached match payloads.
    """
    if not match_result or not match_result.get("success"):
        return None

    for key in ("H", "raw_H", "visual_similarity_H", "similarity_H"):
        candidate = match_result.get(key)
        if (
            isinstance(candidate, np.ndarray)
            and candidate.shape == (3, 3)
            and np.all(np.isfinite(candidate))
        ):
            return candidate
    return None


def _coordinate_homography_for_match(
    match_result: Optional[dict[str, Any]],
) -> Optional[np.ndarray]:
    """Return the exact transform used to draw the analyzed raster on the map.

    A GPS-anchored rebuilt ``H`` is still a complete, metric image-to-map
    transform. Falling back to a second EXIF-heading projection for planting
    coordinates while drawing the raster with ``H`` translates/rotates the
    same lattice into two different map locations. Coordinates and overlay
    must therefore share this one authoritative transform, whether it came
    from a trusted visual match or from the conservative metric rebuild.
    """
    return _display_homography_for_match(match_result)


def _overlay_homography_for_match(match_result: Optional[dict[str, Any]]) -> Optional[np.ndarray]:
    """Use visual registration when warping the analyzed image onto the map."""
    return _display_homography_for_match(match_result)


def _active_ortho_metric_gsd_xy() -> tuple[float, float]:
    """Return active orthophoto x/y pixel sizes in physical metres."""
    fallback = float(getattr(ortho_matcher, "ORTHO_GSD", 0.0) or 0.0)
    gsd_x = float(getattr(ortho_matcher, "ORTHO_GSD_M_X", 0.0) or fallback)
    gsd_y = float(getattr(ortho_matcher, "ORTHO_GSD_M_Y", 0.0) or fallback)
    return gsd_x, gsd_y


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
                    "gsd": (
                        float(entry.get("gsd_m_x", entry["gsd_x"]))
                        + float(entry.get("gsd_m_y", entry["gsd_y"]))
                    )
                    / 2.0,
                }
        except Exception:
            continue

    return None


def _orthophoto_canopy_recheck(
    latitude: float,
    longitude: float,
    safety_radius_m: Optional[float] = None,
) -> Optional[dict[str, Any]]:
    """Detect obvious canopy/vegetation too close to a projected map point."""
    context = _orthophoto_pixel_context(latitude, longitude)
    if context is None:
        return None

    ortho_image = context["image"]
    px = int(context["px"])
    py = int(context["py"])
    ortho_gsd = max(float(context["gsd"]), 1e-9)
    radius_m = max(_ORTHO_CANOPY_RECHECK_RADIUS_M, float(safety_radius_m or 0.0))
    radius_px = max(3, int(round(radius_m / ortho_gsd)))

    y1 = max(0, py - radius_px)
    y2 = min(ortho_image.shape[0], py + radius_px + 1)
    x1 = max(0, px - radius_px)
    x2 = min(ortho_image.shape[1], px + radius_px + 1)
    patch = ortho_image[y1:y2, x1:x2]
    if patch.size == 0:
        return None

    # A very fine orthophoto can make a 0.75 m check window thousands of
    # pixels wide. Downsample only that local patch, preserving the physical
    # radius while putting a hard upper bound on per-point CPU/memory work.
    center_x = px - x1
    center_y = py - y1
    max_patch_dim = 513
    patch_scale = min(
        1.0,
        max_patch_dim / max(1, patch.shape[0]),
        max_patch_dim / max(1, patch.shape[1]),
    )
    if patch_scale < 1.0:
        resized_width = max(3, int(round(patch.shape[1] * patch_scale)))
        resized_height = max(3, int(round(patch.shape[0] * patch_scale)))
        patch = cv2.resize(
            patch,
            (resized_width, resized_height),
            interpolation=cv2.INTER_AREA,
        )
        center_x = int(round(center_x * patch_scale))
        center_y = int(round(center_y * patch_scale))
        ortho_gsd = ortho_gsd / patch_scale
        radius_px = max(3, int(round(radius_m / ortho_gsd)))

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

    yy, xx = np.ogrid[:patch.shape[0], :patch.shape[1]]
    cy = center_y
    cx = center_x
    dist_px = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    circle = dist_px <= radius_px

    valid = ((blue + green + red) > 30.0) & circle
    valid_count = int(np.count_nonzero(valid))
    if valid_count <= 0:
        return None

    center_strong = bool(strong_green[cy, cx]) if 0 <= cy < strong_green.shape[0] and 0 <= cx < strong_green.shape[1] else False
    green_mask = (bright_green | shadow_green) & valid
    strong_mask = strong_green & valid
    green_ratio = float(np.count_nonzero(green_mask) / valid_count)
    strong_ratio = float(np.count_nonzero(strong_mask) / valid_count)
    strong_pixels = int(np.count_nonzero(strong_mask))
    min_cluster_px = max(8, int(round(_ORTHO_CANOPY_MIN_CLUSTER_M2 / (ortho_gsd ** 2))))

    nearest_canopy_m = None
    if strong_pixels > 0:
        nearest_canopy_m = float(np.min(dist_px[strong_mask]) * ortho_gsd)

    # A single green-tinted pixel at the point center is not enough evidence to
    # delete a planting point; wet mud, rocks, and compression artifacts all
    # produce isolated green pixels in the orthophoto. Require a small but real
    # cluster before this late recheck can punch a hole in the planting lattice.
    has_canopy_cluster = strong_pixels >= min_cluster_px
    is_center_canopy = bool(
        has_canopy_cluster
        and (center_strong or strong_ratio >= _ORTHO_CANOPY_STRONG_RATIO)
    )
    is_too_close = bool(
        nearest_canopy_m is not None
        and nearest_canopy_m <= radius_m
        and has_canopy_cluster
    )
    is_canopy = bool(is_center_canopy or is_too_close)

    return {
        "is_canopy": is_canopy,
        "orthophoto": context["entry"].get("name"),
        "ortho_px": px,
        "ortho_py": py,
        "safety_radius_m": radius_m,
        "nearest_canopy_m": nearest_canopy_m,
        "green_ratio": green_ratio,
        "strong_green_ratio": strong_ratio,
        "strong_green_pixels": strong_pixels,
        "min_cluster_pixels": min_cluster_px,
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
        "similarity_confidence": _json_float(match_result.get("similarity_confidence")),
        "similarity_inliers": _json_int(match_result.get("similarity_inliers")),
        "similarity_source_hull_ratio": _json_float(match_result.get("similarity_source_hull_ratio")),
        "similarity_source_span_x_ratio": _json_float(match_result.get("similarity_source_span_x_ratio")),
        "similarity_source_span_y_ratio": _json_float(match_result.get("similarity_source_span_y_ratio")),
        "similarity_source_grid_cells": _json_int(match_result.get("similarity_source_grid_cells")),
        "ortho_name": match_result.get("ortho_name"),
        "ortho_path": match_result.get("ortho_path"),
        "match_score": _json_float(match_result.get("match_score")),
        "center_drift_px": _json_float(match_result.get("center_drift_px")),
        "center_drift_m": _json_float(match_result.get("center_drift_m")),
        "center_offset_east_m": _json_float(match_result.get("center_offset_east_m")),
        "center_offset_north_m": _json_float(match_result.get("center_offset_north_m")),
        "projective_heading": _json_float(match_result.get("projective_heading")),
        "projective_center_drift_m": _json_float(match_result.get("projective_center_drift_m")),
        "gps_anchored": bool(match_result.get("gps_anchored", False)),
        "projection_constrained": bool(match_result.get("projection_constrained", False)),
        "registration_validated": bool(match_result.get("registration_validated", False)),
        "registration_model": match_result.get("registration_model"),
        "registration_inliers": _json_int(match_result.get("registration_inliers")),
        "registration_validation_inliers": _json_int(match_result.get("registration_validation_inliers")),
        "registration_median_error_m": _json_float(match_result.get("registration_median_error_m")),
        "registration_stability_m": _json_float(match_result.get("registration_stability_m")),
        "projection_metric_scale_normalized": bool(match_result.get("projection_metric_scale_normalized", False)),
        "projection_edge_refined": bool(match_result.get("projection_edge_refined", False)),
        "edge_refine_base_score": _json_float(match_result.get("edge_refine_base_score")),
        "edge_refine_score": _json_float(match_result.get("edge_refine_score")),
        "edge_refine_improvement": _json_float(match_result.get("edge_refine_improvement")),
        "edge_refine_shift_m": _json_float(match_result.get("edge_refine_shift_m")),
        "edge_heading_baseline_score": _json_float(match_result.get("edge_heading_baseline_score")),
        "edge_heading_score_gain": _json_float(match_result.get("edge_heading_score_gain")),
        "edge_heading_correction_deg": _json_float(match_result.get("edge_heading_correction_deg")),
        "vegetation_heading_score": _json_float(match_result.get("vegetation_heading_score")),
        "vegetation_heading_baseline_score": _json_float(match_result.get("vegetation_heading_baseline_score")),
        "vegetation_heading_improvement": _json_float(match_result.get("vegetation_heading_improvement")),
        "vegetation_heading_correction_deg": _json_float(match_result.get("vegetation_heading_correction_deg")),
        "vegetation_gsd_scale_factor": _json_float(match_result.get("vegetation_gsd_scale_factor")),
        "refined_gsd_m_per_pixel": _json_float(match_result.get("refined_gsd_m_per_pixel")),
        "projection_rebuilt": bool(match_result.get("projection_rebuilt", False)),
        "projection_rotation_source": match_result.get("projection_rotation_source"),
        "projection_rebuild_reason": match_result.get("projection_rebuild_reason"),
        "raw_projection_scale": _json_float(match_result.get("raw_projection_scale")),
        "expected_projection_scale": _json_float(match_result.get("expected_projection_scale")),
        "projection_scale_ratio": _json_float(match_result.get("projection_scale_ratio")),
        "projection_scale_error_pct": _json_float(match_result.get("projection_scale_error_pct")),
        "heading_diff_vs_exif_deg": _json_float(match_result.get("heading_diff_vs_exif_deg")),
        "rejected_reason": match_result.get("rejected_reason"),
        "sift_error": match_result.get("sift_error"),
        "error": match_result.get("error"),
    }


def _estimated_analysis_footprint(
    latitude: Optional[float],
    longitude: Optional[float],
    coverage_m: Any,
    heading_deg: Any,
) -> Optional[dict[str, Any]]:
    """Build an oriented coverage rectangle for overlap-safe area totals.

    The processing pipeline currently stores an image centre and physical
    width/height rather than four surveyed corner coordinates. Persisting this
    explicitly marked estimate is still preferable to summing overlapping
    image areas; callers must retain the quality label and avoid presenting it
    as a cadastral boundary.
    """
    if latitude is None or longitude is None:
        return None
    try:
        width_m, height_m = [float(value) for value in coverage_m[:2]]
        heading = math.radians(float(heading_deg or 0.0))
    except (TypeError, ValueError, IndexError):
        return None
    if width_m <= 0 or height_m <= 0:
        return None

    metres_per_degree_lat = 111_320.0
    metres_per_degree_lon = metres_per_degree_lat * max(
        0.01,
        math.cos(math.radians(float(latitude))),
    )
    ring = []
    for x_m, y_m in (
        (-width_m / 2.0, -height_m / 2.0),
        (width_m / 2.0, -height_m / 2.0),
        (width_m / 2.0, height_m / 2.0),
        (-width_m / 2.0, height_m / 2.0),
        (-width_m / 2.0, -height_m / 2.0),
    ):
        east_m = (x_m * math.cos(heading)) + (y_m * math.sin(heading))
        north_m = (-x_m * math.sin(heading)) + (y_m * math.cos(heading))
        ring.append([
            float(longitude) + (east_m / metres_per_degree_lon),
            float(latitude) + (north_m / metres_per_degree_lat),
        ])
    return {"type": "Polygon", "coordinates": [ring]}


_NO_GPS_DETAIL = (
    "No GPS location was found in this image. Analysis was not started because "
    "MangroVision cannot verify where the image belongs on the GIS map."
)
_PARTIAL_GIS_MAP_DETAIL = (
    "Part of this image is outside the supported GIS map area. Review the highlighted "
    "footprint and confirm whether you want to continue processing the mapped portion."
)


def _footprint_grid_points(
    footprint: Optional[dict[str, Any]],
    *,
    sample_count: int = 11,
) -> list[tuple[float, float]]:
    """Return a regular lat/lon grid covering an oriented rectangle footprint."""
    try:
        ring = footprint["coordinates"][0]
        corner_0, corner_1, _, corner_3 = ring[:4]
        count = max(2, int(sample_count))
        points: list[tuple[float, float]] = []
        for row in range(count):
            v = row / (count - 1)
            for column in range(count):
                u = column / (count - 1)
                longitude = (
                    float(corner_0[0])
                    + (u * (float(corner_1[0]) - float(corner_0[0])))
                    + (v * (float(corner_3[0]) - float(corner_0[0])))
                )
                latitude = (
                    float(corner_0[1])
                    + (u * (float(corner_1[1]) - float(corner_0[1])))
                    + (v * (float(corner_3[1]) - float(corner_0[1])))
                )
                points.append((latitude, longitude))
        return points
    except (KeyError, TypeError, ValueError, IndexError):
        return []


def _classify_footprint_against_gis(
    footprint: Optional[dict[str, Any]],
    latitude: float,
    longitude: float,
    coverage: Optional[Any] = None,
) -> dict[str, Any]:
    """Classify the complete image against the visible GIS map."""
    try:
        if coverage is None:
            coverage = _load_gis_coverage_geometry()
        requested = shape(footprint) if footprint else Point(longitude, latitude)
        if coverage.covers(requested):
            status, percentage = "inside", 100.
        elif footprint and coverage.intersection(requested).area > 0:
            status = "partial" if footprint else "inside"
            percentage = 100. * coverage.intersection(requested).area / requested.area
        else:
            status, percentage = "outside", 0.
        return {"status": status, "estimated_inside_pct": round(max(0., min(100., percentage)), 1)}
    except Exception as error:
        raise HTTPException(status_code=503, detail=_GIS_MAP_CHECK_FAILED_DETAIL) from error


def _project_site_context(
    latitude: float,
    longitude: float,
    footprint: Optional[dict[str, Any]],
) -> dict[str, Any]:
    """Identify named LGU project sites at or intersecting the image footprint."""
    try:
        return project_site_context(latitude, longitude, footprint)
    except Exception:
        return {"project_sites": [], "location_label": None}


def _build_image_location_preflight(
    metadata: dict[str, Any],
    altitude: float,
    drone_model: str,
) -> dict[str, Any]:
    """Build a detector-free image location and map-boundary assessment."""
    if not metadata.get("has_gps") or not metadata.get("gps"):
        return {
            "status": "no_gps",
            "can_process": False,
            "requires_confirmation": False,
            "message": _NO_GPS_DETAIL,
            "map": {"available": False, "location_status": "no_gps"},
        }

    gps = metadata["gps"]
    try:
        latitude = float(gps["latitude"])
        longitude = float(gps["longitude"])
    except (KeyError, TypeError, ValueError):
        return {
            "status": "no_gps",
            "can_process": False,
            "requires_confirmation": False,
            "message": _NO_GPS_DETAIL,
            "map": {"available": False, "location_status": "no_gps"},
        }
    if (
        not math.isfinite(latitude)
        or not math.isfinite(longitude)
        or not -90.0 <= latitude <= 90.0
        or not -180.0 <= longitude <= 180.0
    ):
        return {
            "status": "no_gps",
            "can_process": False,
            "requires_confirmation": False,
            "message": _NO_GPS_DETAIL,
            "map": {"available": False, "location_status": "no_gps"},
        }

    camera_info = metadata.get("camera") or {}
    resolved_drone_model = ExifExtractor.detect_drone_model(camera_info) if camera_info else drone_model

    resolved_altitude = float(altitude)
    detected_altitude = gps.get("relative_altitude")
    if detected_altitude is None:
        detected_altitude = gps.get("altitude")
    if _is_reasonable_altitude(detected_altitude):
        resolved_altitude = float(detected_altitude)

    heading = float(gps.get("heading") or 0.0) % 360.0
    try:
        image_width = int(metadata.get("image_width") or camera_info.get("image_width"))
        image_height = int(metadata.get("image_height") or camera_info.get("image_height"))
    except (TypeError, ValueError):
        image_width = 0
        image_height = 0

    footprint = None
    coverage_m = None
    gsd_m = None
    gsd_specs: dict[str, Any] = {}
    if image_width > 0 and image_height > 0:
        gsd_m, gsd_specs = GSDCalculator.calculate_gsd_from_metadata(
            altitude_m=resolved_altitude,
            camera_info=camera_info,
            drone_model=resolved_drone_model,
            image_width_px=image_width,
            image_height_px=image_height,
        )
        coverage_m = [float(image_width) * float(gsd_m), float(image_height) * float(gsd_m)]
        footprint = _estimated_analysis_footprint(
            latitude,
            longitude,
            coverage_m,
            heading,
        )

    boundary = _classify_footprint_against_gis(footprint, latitude, longitude)
    status = boundary["status"]
    site_context = _project_site_context(latitude, longitude, footprint)
    if not site_context["location_label"]:
        try:
            site_context["location_label"] = ortho_matcher._active_tileset_name().replace("_", " ")
        except Exception:
            site_context["location_label"] = "Mapped GIS area"

    if status == "inside":
        message = "The complete estimated image footprint is inside the supported GIS map."
    elif status == "partial":
        message = _PARTIAL_GIS_MAP_DETAIL
    else:
        message = _OUTSIDE_GIS_MAP_DETAIL

    center_feature = _image_center_feature(
        latitude,
        longitude,
        {
            "name": "Selected Image Center",
            "location_label": site_context["location_label"],
            "altitude_m": round(resolved_altitude, 2),
            "heading": round(heading, 2),
            "preflight": True,
            "outside_map_bounds": not _load_gis_coverage_geometry().covers(Point(longitude, latitude)),
        },
    )
    return {
        "status": status,
        "can_process": status in {"inside", "partial"},
        "requires_confirmation": status == "partial",
        "message": message,
        "latitude": latitude,
        "longitude": longitude,
        "location_label": site_context["location_label"],
        "project_sites": site_context["project_sites"],
        "altitude_m": round(resolved_altitude, 2),
        "heading_deg": round(heading, 2),
        "image_width_px": image_width or None,
        "image_height_px": image_height or None,
        "gsd_m_per_pixel": round(float(gsd_m), 8) if gsd_m is not None else None,
        "gsd_source": gsd_specs.get("source"),
        "coverage_m": [round(value, 2) for value in coverage_m] if coverage_m else None,
        "footprint_calibrated": (
            gsd_specs.get("source") == "exif_focal_known_sensor"
        ),
        "footprint_calibration_method": (
            "camera_metadata"
            if gsd_specs.get("source") == "exif_focal_known_sensor"
            else None
        ),
        **boundary,
        "map": {
            "available": True,
            "location_status": status,
            "image_center_feature": center_feature,
            "analysis_footprint": footprint,
        },
    }


def _calibrate_preflight_footprint(
    temp_path: Path,
    preflight: dict[str, Any],
) -> dict[str, Any]:
    """Replace the nominal preflight rectangle with image-to-map calibration."""
    if preflight.get("status") == "no_gps":
        return preflight
    latitude = _safe_float(preflight.get("latitude"))
    longitude = _safe_float(preflight.get("longitude"))
    nominal_gsd = _safe_float(preflight.get("gsd_m_per_pixel"))
    heading = _safe_float(preflight.get("heading_deg"))
    if any(
        value is None for value in (latitude, longitude, nominal_gsd, heading)
    ) or float(nominal_gsd) <= 0:
        return preflight

    image = cv2.imread(str(temp_path))
    if not isinstance(image, np.ndarray):
        return preflight
    file_digest: Optional[str] = None
    calibration: Optional[dict[str, Any]] = None
    try:
        file_digest = _file_sha256(temp_path)
        cached = _get_preflight_alignment_cache(file_digest)
        if (
            cached
            and cached.get("success")
            and cached.get("projection_rotation_source")
            == "vegetation_scale_metric_anchor"
        ):
            calibration = cached
    except Exception:
        file_digest = None

    if calibration is None:
        calibration = _refine_failed_match_with_vegetation_scale(
            {
                "success": False,
                "error": "Feature matching is deferred until full processing.",
            },
            image,
            float(latitude),
            float(longitude),
            float(nominal_gsd),
            float(heading),
        )
    if not calibration.get("success"):
        return preflight

    refined_gsd = _safe_float(calibration.get("refined_gsd_m_per_pixel"))
    refined_heading = _safe_float(calibration.get("heading"))
    if refined_gsd is None or refined_gsd <= 0 or refined_heading is None:
        return preflight

    image_height, image_width = image.shape[:2]
    coverage_m = [image_width * refined_gsd, image_height * refined_gsd]
    footprint = _estimated_analysis_footprint(
        float(latitude),
        float(longitude),
        coverage_m,
        refined_heading,
    )
    boundary = _classify_footprint_against_gis(
        footprint,
        float(latitude),
        float(longitude),
    )
    status = boundary["status"]
    if status == "inside":
        message = (
            "The calibrated image footprint is inside the supported GIS map."
        )
    elif status == "partial":
        message = _PARTIAL_GIS_MAP_DETAIL
    else:
        message = _OUTSIDE_GIS_MAP_DETAIL

    site_context = _project_site_context(
        float(latitude),
        float(longitude),
        footprint,
    )
    location_label = site_context.get("location_label") or preflight.get(
        "location_label"
    )
    center_feature = _image_center_feature(
        float(latitude),
        float(longitude),
        {
            "name": "Selected Image Center",
            "location_label": location_label,
            "altitude_m": preflight.get("altitude_m"),
            "heading": round(refined_heading, 2),
            "preflight": True,
            "footprint_calibrated": True,
            "outside_map_bounds": not _load_gis_coverage_geometry().covers(
                Point(float(longitude), float(latitude))
            ),
        },
    )
    nominal_coverage = preflight.get("coverage_m")
    preflight.update(
        {
            "status": status,
            "can_process": status in {"inside", "partial"},
            "requires_confirmation": status == "partial",
            "message": message,
            "location_label": location_label,
            "project_sites": site_context.get("project_sites") or preflight.get(
                "project_sites", []
            ),
            "heading_deg": round(refined_heading, 2),
            "gsd_m_per_pixel": round(refined_gsd, 8),
            "coverage_m": [round(value, 2) for value in coverage_m],
            "nominal_coverage_m": nominal_coverage,
            "footprint_scale_factor": round(
                float(calibration.get("vegetation_gsd_scale_factor") or 1.0),
                4,
            ),
            "footprint_calibrated": True,
            "footprint_calibration_method": "vegetation_scale",
            "alignment": _sanitize_match_result(calibration),
            **boundary,
            "map": {
                "available": True,
                "location_status": status,
                "image_center_feature": center_feature,
                "analysis_footprint": footprint,
                "match": _sanitize_match_result(calibration),
            },
        }
    )
    if file_digest is not None:
        try:
            _set_preflight_alignment_cache(file_digest, calibration)
        except Exception:
            pass
    return preflight


def _enforce_image_location_preflight(
    preflight: dict[str, Any],
    *,
    allow_partial_map_overlap: bool,
) -> None:
    status = preflight.get("status")
    if status == "no_gps":
        raise HTTPException(status_code=422, detail=_NO_GPS_DETAIL)
    if status == "outside":
        raise HTTPException(status_code=422, detail=_OUTSIDE_GIS_MAP_DETAIL)
    if status == "partial" and not allow_partial_map_overlap:
        raise HTTPException(status_code=409, detail=_PARTIAL_GIS_MAP_DETAIL)


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


def _build_metric_centered_homography(
    drone_image: np.ndarray,
    center_lat: float,
    center_lon: float,
    drone_gsd: float,
    heading_deg: float,
) -> Optional[np.ndarray]:
    """Build a GPS-centered drone-pixel to orthophoto-pixel transform."""
    if not isinstance(drone_image, np.ndarray) or drone_image.ndim < 2:
        return None
    if drone_gsd is None or drone_gsd <= 0:
        return None

    image_h, image_w = drone_image.shape[:2]
    if image_h <= 0 or image_w <= 0:
        return None

    try:
        center_ox, center_oy = gps_to_ortho_pixel(center_lat, center_lon)
        ortho_gsd_x, ortho_gsd_y = _active_ortho_metric_gsd_xy()
    except Exception:
        return None

    if ortho_gsd_x <= 0 or ortho_gsd_y <= 0:
        return None

    cx = image_w / 2.0
    cy = image_h / 2.0
    theta = np.radians(-float(heading_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    h00 = drone_gsd * cos_t / ortho_gsd_x
    h01 = drone_gsd * sin_t / ortho_gsd_x
    h02 = center_ox - (h00 * cx) - (h01 * cy)

    h10 = -drone_gsd * sin_t / ortho_gsd_y
    h11 = drone_gsd * cos_t / ortho_gsd_y
    h12 = center_oy - (h10 * cx) - (h11 * cy)

    return np.array(
        [
            [h00, h01, h02],
            [h10, h11, h12],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _build_metric_visual_homography(
    drone_image: np.ndarray,
    visual_homography: Any,
    drone_gsd: float,
    heading_deg: float,
) -> Optional[np.ndarray]:
    """Use visual translation/heading but force scale to the physical GSD."""
    if not isinstance(drone_image, np.ndarray) or drone_image.ndim < 2:
        return None
    if not isinstance(visual_homography, np.ndarray) or visual_homography.shape != (3, 3):
        return None
    if drone_gsd is None or drone_gsd <= 0:
        return None

    image_h, image_w = drone_image.shape[:2]
    if image_h <= 0 or image_w <= 0:
        return None

    try:
        ortho_gsd_x, ortho_gsd_y = _active_ortho_metric_gsd_xy()
        center_mapped = cv2.perspectiveTransform(
            np.array([[[image_w / 2.0, image_h / 2.0]]], dtype=np.float64),
            visual_homography,
        )[0, 0]
    except Exception:
        return None

    if ortho_gsd_x <= 0 or ortho_gsd_y <= 0 or not np.all(np.isfinite(center_mapped)):
        return None

    cx = image_w / 2.0
    cy = image_h / 2.0
    theta = np.radians(-float(heading_deg))
    cos_t = float(np.cos(theta))
    sin_t = float(np.sin(theta))

    h00 = drone_gsd * cos_t / ortho_gsd_x
    h01 = drone_gsd * sin_t / ortho_gsd_x
    h02 = float(center_mapped[0]) - (h00 * cx) - (h01 * cy)

    h10 = -drone_gsd * sin_t / ortho_gsd_y
    h11 = drone_gsd * cos_t / ortho_gsd_y
    h12 = float(center_mapped[1]) - (h10 * cx) - (h11 * cy)

    return np.array(
        [
            [h00, h01, h02],
            [h10, h11, h12],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _prepare_edge_image(gray: np.ndarray) -> np.ndarray:
    """Return a stable edge map for cross-date drone/orthophoto matching."""
    if gray.ndim != 2:
        return np.zeros(gray.shape[:2], dtype=np.uint8)
    try:
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced = clahe.apply(gray.astype(np.uint8))
    except Exception:
        enhanced = gray.astype(np.uint8)
    enhanced = cv2.GaussianBlur(enhanced, (3, 3), 0)
    nonzero = enhanced[enhanced > 8]
    median = float(np.median(nonzero)) if nonzero.size else float(np.median(enhanced))
    lower = int(max(20, min(120, median * 0.66)))
    upper = int(max(lower + 30, min(220, median * 1.45)))
    return cv2.Canny(enhanced, lower, upper)


def _refine_homography_by_edge_translation(
    drone_image: np.ndarray,
    homography: Any,
) -> tuple[Optional[np.ndarray], dict[str, Optional[float]]]:
    """Nudge a rigid/metric footprint by matching edges against the orthophoto.

    The refinement is translation-only. Rotation and scale stay fixed, so this
    cannot introduce the perspective bending that made some overlays line up in
    one area while drifting elsewhere.
    """
    diagnostics: dict[str, Optional[float]] = {
        "edge_refine_base_score": None,
        "edge_refine_score": None,
        "edge_refine_improvement": None,
        "edge_refine_shift_px": None,
        "edge_refine_shift_m": None,
    }
    if not isinstance(drone_image, np.ndarray) or drone_image.ndim < 2:
        return None, diagnostics
    if not isinstance(homography, np.ndarray) or homography.shape != (3, 3):
        return None, diagnostics

    ortho = None
    try:
        ortho = ortho_matcher.load_orthophoto()
        ortho_gsd = float(ortho_matcher.ORTHO_GSD or 0.0)
    except Exception:
        ortho = None
        ortho_gsd = 0.0
    if not isinstance(ortho, np.ndarray) or ortho.ndim < 2 or ortho_gsd <= 0:
        return None, diagnostics

    image_h, image_w = drone_image.shape[:2]
    if image_h <= 0 or image_w <= 0:
        return None, diagnostics

    corners = np.array(
        [[[0.0, 0.0]], [[float(image_w), 0.0]], [[float(image_w), float(image_h)]], [[0.0, float(image_h)]]],
        dtype=np.float64,
    )
    try:
        ortho_corners = cv2.perspectiveTransform(corners, homography).reshape(-1, 2)
    except Exception:
        return None, diagnostics
    if not np.all(np.isfinite(ortho_corners)):
        return None, diagnostics

    max_shift_px = max(4.0, _EDGE_REFINE_MAX_SHIFT_M / max(ortho_gsd, 1e-9))
    padding_px = max_shift_px + 10.0
    min_x = int(np.floor(float(np.min(ortho_corners[:, 0])) - padding_px))
    max_x = int(np.ceil(float(np.max(ortho_corners[:, 0])) + padding_px))
    min_y = int(np.floor(float(np.min(ortho_corners[:, 1])) - padding_px))
    max_y = int(np.ceil(float(np.max(ortho_corners[:, 1])) + padding_px))

    min_x = max(0, min_x)
    min_y = max(0, min_y)
    max_x = min(ortho.shape[1], max_x)
    max_y = min(ortho.shape[0], max_y)
    patch_w = max_x - min_x
    patch_h = max_y - min_y
    if patch_w < 80 or patch_h < 80:
        return None, diagnostics

    scale = min(1.0, _EDGE_REFINE_MAX_DIM_PX / max(patch_w, patch_h))
    patch = ortho[min_y:max_y, min_x:max_x]
    if scale < 1.0:
        scaled_size = (
            max(1, int(round(patch_w * scale))),
            max(1, int(round(patch_h * scale))),
        )
        patch = cv2.resize(patch, scaled_size, interpolation=cv2.INTER_AREA)
    else:
        scaled_size = (patch_w, patch_h)

    patch_gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
    patch_edges = _prepare_edge_image(patch_gray)
    if int(np.count_nonzero(patch_edges)) < 200:
        return None, diagnostics

    dist_to_ortho_edge = cv2.distanceTransform(
        (patch_edges == 0).astype(np.uint8),
        cv2.DIST_L2,
        3,
    )

    drone_gray = cv2.cvtColor(drone_image, cv2.COLOR_BGR2GRAY)
    drone_edges = _prepare_edge_image(drone_gray)
    if int(np.count_nonzero(drone_edges)) < 300:
        return None, diagnostics

    crop_scale = np.array(
        [
            [scale, 0.0, -min_x * scale],
            [0.0, scale, -min_y * scale],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )
    base_warp = crop_scale @ homography
    out_w, out_h = scaled_size
    max_shift_scaled = max(3, int(round(max_shift_px * scale)))
    coarse_step = max(3, int(round(max_shift_scaled / 6)))

    def _score_shift(dx: int, dy: int) -> Optional[float]:
        shift = np.array(
            [[1.0, 0.0, float(dx)], [0.0, 1.0, float(dy)], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        try:
            warped_edges = cv2.warpPerspective(
                drone_edges,
                shift @ base_warp,
                (out_w, out_h),
                flags=cv2.INTER_NEAREST,
                borderMode=cv2.BORDER_CONSTANT,
                borderValue=0,
            )
        except Exception:
            return None
        mask = warped_edges > 0
        edge_count = int(np.count_nonzero(mask))
        if edge_count < 150:
            return None
        distances = dist_to_ortho_edge[mask]
        if distances.size == 0:
            return None
        # Convert edge distance into a bounded closeness score. Two pixels at
        # the downsampled scale roughly covers unavoidable cross-date blur.
        return float(np.mean(np.exp(-distances / 2.0)))

    base_score = _score_shift(0, 0)
    if base_score is None:
        return None, diagnostics

    best = (0, 0, base_score)
    for dy in range(-max_shift_scaled, max_shift_scaled + 1, coarse_step):
        for dx in range(-max_shift_scaled, max_shift_scaled + 1, coarse_step):
            score = _score_shift(dx, dy)
            if score is not None and score > best[2]:
                best = (dx, dy, score)

    fine_step = max(1, coarse_step // 3)
    bx, by, _ = best
    fine_radius = max(coarse_step, 2 * fine_step)
    for dy in range(by - fine_radius, by + fine_radius + 1, fine_step):
        if abs(dy) > max_shift_scaled:
            continue
        for dx in range(bx - fine_radius, bx + fine_radius + 1, fine_step):
            if abs(dx) > max_shift_scaled:
                continue
            score = _score_shift(dx, dy)
            if score is not None and score > best[2]:
                best = (dx, dy, score)

    best_dx_scaled, best_dy_scaled, best_score = best
    improvement = best_score - base_score
    shift_px = float(np.hypot(best_dx_scaled, best_dy_scaled) / max(scale, 1e-9))
    shift_m = shift_px * ortho_gsd
    diagnostics.update(
        {
            "edge_refine_base_score": float(base_score),
            "edge_refine_score": float(best_score),
            "edge_refine_improvement": float(improvement),
            "edge_refine_shift_px": shift_px,
            "edge_refine_shift_m": float(shift_m),
        }
    )
    if improvement < _EDGE_REFINE_MIN_IMPROVEMENT or shift_px < 0.5:
        return None, diagnostics
    if shift_m > (_EDGE_REFINE_MAX_SHIFT_M + 0.25):
        return None, diagnostics

    dx_px = float(best_dx_scaled) / max(scale, 1e-9)
    dy_px = float(best_dy_scaled) / max(scale, 1e-9)
    translation = np.array(
        [[1.0, 0.0, dx_px], [0.0, 1.0, dy_px], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )
    return translation @ homography, diagnostics


def _homography_scale_diagnostics(
    homography: Any,
    drone_image: np.ndarray,
    drone_gsd: float,
) -> dict[str, Optional[float]]:
    """Compare a homography's local pixel scale against the GSD-derived scale."""
    diagnostics: dict[str, Optional[float]] = {
        "raw_projection_scale": None,
        "expected_projection_scale": None,
        "projection_scale_ratio": None,
        "projection_scale_error_pct": None,
    }
    if not isinstance(homography, np.ndarray) or homography.shape != (3, 3):
        return diagnostics
    if not isinstance(drone_image, np.ndarray) or drone_image.ndim < 2:
        return diagnostics
    if drone_gsd is None or drone_gsd <= 0:
        return diagnostics

    try:
        ortho_gsd_x, ortho_gsd_y = _active_ortho_metric_gsd_xy()
    except Exception:
        return diagnostics
    if ortho_gsd_x <= 0 or ortho_gsd_y <= 0:
        return diagnostics

    image_h, image_w = drone_image.shape[:2]
    if image_h <= 0 or image_w <= 0:
        return diagnostics

    probe_px = max(8.0, min(256.0, image_w * 0.20, image_h * 0.20))
    cx = image_w / 2.0
    cy = image_h / 2.0
    probes = np.array(
        [[[cx, cy]], [[cx + probe_px, cy]], [[cx, cy + probe_px]]],
        dtype=np.float64,
    )
    try:
        mapped = cv2.perspectiveTransform(probes, homography).reshape(3, 2)
    except Exception:
        return diagnostics
    if not np.all(np.isfinite(mapped)):
        return diagnostics

    center, east_probe, south_probe = mapped
    scale_x = float(np.linalg.norm(east_probe - center) / probe_px)
    scale_y = float(np.linalg.norm(south_probe - center) / probe_px)
    raw_scale = (scale_x + scale_y) / 2.0
    expected_x = float(drone_gsd) / ortho_gsd_x
    expected_y = float(drone_gsd) / ortho_gsd_y
    expected_scale = (expected_x + expected_y) / 2.0
    if raw_scale <= 0 or expected_scale <= 0:
        return diagnostics

    ratio = raw_scale / expected_scale
    diagnostics.update(
        {
            "raw_projection_scale": float(raw_scale),
            "expected_projection_scale": float(expected_scale),
            "projection_scale_ratio": float(ratio),
            "projection_scale_error_pct": float(abs(ratio - 1.0) * 100.0),
        }
    )
    return diagnostics


def _homography_heading_deg(homography: Any) -> Optional[float]:
    """Extract the map heading represented by a drone-pixel homography."""
    if not isinstance(homography, np.ndarray) or homography.shape != (3, 3):
        return None
    try:
        h00 = float(homography[0, 0])
        h10 = float(homography[1, 0])
        if not (np.isfinite(h00) and np.isfinite(h10)):
            return None
        return float(np.degrees(np.arctan2(h10, h00)) % 360.0)
    except Exception:
        return None


def _homography_center_diagnostics(
    homography: Any,
    drone_image: np.ndarray,
    center_lat: float,
    center_lon: float,
) -> dict[str, Optional[float]]:
    """Measure where the image center lands relative to the EXIF GPS center."""
    diagnostics: dict[str, Optional[float]] = {
        "center_drift_px": None,
        "center_drift_m": None,
        "center_offset_east_m": None,
        "center_offset_north_m": None,
    }
    if not isinstance(homography, np.ndarray) or homography.shape != (3, 3):
        return diagnostics
    if not isinstance(drone_image, np.ndarray) or drone_image.ndim < 2:
        return diagnostics

    image_h, image_w = drone_image.shape[:2]
    if image_h <= 0 or image_w <= 0:
        return diagnostics

    try:
        expected_x, expected_y = gps_to_ortho_pixel(center_lat, center_lon)
        ortho_gsd_x, ortho_gsd_y = _active_ortho_metric_gsd_xy()
        mapped = cv2.perspectiveTransform(
            np.array([[[image_w / 2.0, image_h / 2.0]]], dtype=np.float64),
            homography,
        )[0, 0]
    except Exception:
        return diagnostics

    if not np.all(np.isfinite(mapped)):
        return diagnostics

    dx_px = float(mapped[0] - expected_x)
    dy_px = float(mapped[1] - expected_y)
    east_m = dx_px * ortho_gsd_x
    north_m = -dy_px * ortho_gsd_y
    diagnostics.update(
        {
            "center_drift_px": float(np.hypot(dx_px, dy_px)),
            "center_drift_m": float(np.hypot(east_m, north_m)),
            "center_offset_east_m": float(east_m),
            "center_offset_north_m": float(north_m),
        }
    )
    return diagnostics


def _has_strong_constrained_visual_support(
    match_result: dict[str, Any],
    using_similarity: bool,
    heading_diff_vs_exif_deg: Optional[float],
) -> bool:
    """Return whether a constrained visual match is safe for coordinates.

    The primary path preserves the existing ratio-based gate. A second path
    accepts a lower inlier ratio only when many inliers are distributed across
    the image. This handles feature-rich scenes where extra unmatched water,
    mud, or changed vegetation lowers the ratio despite a well-constrained
    registration, while rejecting large but localized repetitive clusters.
    """
    if not using_similarity or heading_diff_vs_exif_deg is None:
        return False
    try:
        inliers = int(match_result.get("similarity_inliers") or 0)
        confidence = float(match_result.get("similarity_confidence") or 0.0)
        hull_ratio = float(match_result.get("similarity_source_hull_ratio") or 0.0)
        span_x_ratio = float(match_result.get("similarity_source_span_x_ratio") or 0.0)
        span_y_ratio = float(match_result.get("similarity_source_span_y_ratio") or 0.0)
        grid_cells = int(match_result.get("similarity_source_grid_cells") or 0)
        heading_diff = abs(float(heading_diff_vs_exif_deg))
    except (TypeError, ValueError):
        return False

    ratio_supported = (
        inliers >= 80
        and confidence >= _RAW_MATCH_STRONG_CONSTRAINED_MIN_CONFIDENCE
    )
    broadly_supported = (
        inliers >= _RAW_MATCH_BROAD_SUPPORT_MIN_INLIERS
        and confidence >= _RAW_MATCH_BROAD_SUPPORT_MIN_CONFIDENCE
        and hull_ratio >= _RAW_MATCH_BROAD_SUPPORT_MIN_HULL_RATIO
        and span_x_ratio >= _RAW_MATCH_BROAD_SUPPORT_MIN_SPAN_RATIO
        and span_y_ratio >= _RAW_MATCH_BROAD_SUPPORT_MIN_SPAN_RATIO
        and grid_cells >= _RAW_MATCH_BROAD_SUPPORT_MIN_GRID_CELLS
    )
    return bool((ratio_supported or broadly_supported) and heading_diff <= 10.0)


def _post_process_match(
    match_result: dict[str, Any],
    drone_image: np.ndarray,
    center_lat: float,
    center_lon: float,
    camera_heading: Optional[float],
    drone_gsd: float,
) -> dict[str, Any]:
    """Use raw SIFT only when it is good enough for coordinates.

    SIFT can visually align the overlay well but still drift from the EXIF/GPS
    center on WebODM maps. When drift or confidence falls outside coordinate
    limits, rebuild a metric GPS-anchored homography using the best available
    heading. The response keeps the raw telemetry so the UI can explain this.
    """
    if not match_result.get("success"):
        return match_result

    validated_h = match_result.get("validated_H")
    if match_result.get("registration_validated") and isinstance(validated_h, np.ndarray):
        center_diagnostics = _homography_center_diagnostics(
            validated_h, drone_image, center_lat, center_lon,
        )
        center_drift = center_diagnostics.get("center_drift_m")
        if center_drift is not None and center_drift <= 5.0:
            # This model has passed held-out, spatial-support, perspective and
            # stability checks. Preserve its measured ground registration;
            # resetting it to EXIF GPS would reintroduce the observed offset.
            match_result["projective_H"] = match_result.get("H")
            match_result["H"] = validated_h
            match_result.update(center_diagnostics)
            h, w = drone_image.shape[:2]
            probes = cv2.perspectiveTransform(
                np.float64([[[w / 2, h / 2]], [[w / 2 + 1, h / 2]]]),
                validated_h,
            ).reshape(2, 2)
            dx, dy = probes[1] - probes[0]
            match_result["heading"] = float(np.degrees(np.arctan2(dy, dx)) % 360)
            if camera_heading is not None:
                match_result["heading_diff_vs_exif_deg"] = _angular_diff(
                    match_result["heading"], camera_heading,
                )
            model = match_result.get("registration_model", "projective")
            match_result.update({
                "gps_anchored": False,
                "projection_rebuilt": False,
                "projection_constrained": model != "projective",
                "projection_rotation_source": f"sift_validated_{model}",
                "projection_metric_scale_normalized": False,
                "projection_edge_refined": False,
                "registration_median_error_m": float(match_result["registration_median_error_px"]) * ortho_matcher.ORTHO_GSD,
                "registration_stability_m": float(match_result["registration_stability_px"]) * ortho_matcher.ORTHO_GSD,
            })
            match_result.update(_homography_scale_diagnostics(validated_h, drone_image, drone_gsd))
            return match_result
        match_result["registration_validated"] = False

    projective_h = match_result.get("H")
    similarity_h = match_result.get("similarity_H")
    using_similarity = isinstance(similarity_h, np.ndarray) and similarity_h.shape == (3, 3)
    if using_similarity:
        match_result["projective_H"] = projective_h
        match_result["projection_constrained"] = True
        match_result["projective_heading"] = match_result.get("heading")
        for key in (
            "center_drift_px",
            "center_drift_m",
            "center_offset_east_m",
            "center_offset_north_m",
        ):
            match_result[f"projective_{key}"] = match_result.get(key)

        similarity_heading = _homography_heading_deg(similarity_h)
        if similarity_heading is not None:
            match_result["heading"] = similarity_heading
        metric_visual_h = None
        if _FORCE_METRIC_VISUAL_SCALE and match_result.get("heading") is not None:
            metric_visual_h = _build_metric_visual_homography(
                drone_image,
                similarity_h,
                drone_gsd,
                float(match_result["heading"]),
            )
        if metric_visual_h is not None:
            match_result["visual_similarity_H"] = similarity_h
            match_result["H"] = metric_visual_h
            match_result["projection_metric_scale_normalized"] = True
            if _EDGE_REFINE_ENABLED:
                refined_h, edge_diagnostics = _refine_homography_by_edge_translation(
                    drone_image,
                    metric_visual_h,
                )
                for key, value in edge_diagnostics.items():
                    if value is not None:
                        match_result[key] = value
                if refined_h is not None:
                    match_result["H"] = refined_h
                    match_result["projection_edge_refined"] = True
                else:
                    match_result["projection_edge_refined"] = False
            else:
                match_result["projection_edge_refined"] = False
        else:
            match_result["H"] = similarity_h
            match_result["projection_metric_scale_normalized"] = False
            match_result["projection_edge_refined"] = False

        center_diagnostics = _homography_center_diagnostics(
            match_result.get("H"),
            drone_image,
            center_lat,
            center_lon,
        )
        for key, value in center_diagnostics.items():
            if value is not None:
                match_result[key] = value
    else:
        match_result["projection_constrained"] = False
        match_result["projection_metric_scale_normalized"] = False
        match_result["projection_edge_refined"] = False

    sift_heading = match_result.get("heading")
    if camera_heading is not None and sift_heading is not None:
        match_result["heading_diff_vs_exif_deg"] = float(
            _angular_diff(sift_heading, camera_heading)
        )

    match_result["gps_anchored"] = False
    match_result["projection_rebuilt"] = False
    match_result["projection_rotation_source"] = (
        "sift_metric_similarity_edge"
        if match_result.get("projection_edge_refined")
        else "sift_metric_similarity"
        if match_result.get("projection_metric_scale_normalized")
        else ("sift_similarity" if using_similarity else "sift_projective")
    )

    scale_diagnostics = _homography_scale_diagnostics(
        match_result.get("H"),
        drone_image,
        drone_gsd,
    )
    for key, value in scale_diagnostics.items():
        if value is not None:
            match_result[key] = value

    center_drift_m = match_result.get("center_drift_m")
    confidence = match_result.get("confidence")
    projection_scale_ratio = scale_diagnostics.get("projection_scale_ratio")
    heading_diff_for_gate = match_result.get("heading_diff_vs_exif_deg")
    strong_constrained_match = (
        camera_heading is not None
        and _has_strong_constrained_visual_support(
            match_result,
            using_similarity=using_similarity,
            heading_diff_vs_exif_deg=heading_diff_for_gate,
        )
    )

    rebuild_reasons: list[str] = []
    if (
        center_drift_m is not None
        and float(center_drift_m) > _RAW_MATCH_MAX_DRIFT_FOR_COORDS_M
        and not (strong_constrained_match and float(center_drift_m) <= 4.0)
    ):
        rebuild_reasons.append(
            f"{float(center_drift_m):.2f} m center drift"
        )
    if (
        confidence is not None
        and float(confidence) < _RAW_MATCH_MIN_CONFIDENCE_FOR_COORDS
        and not strong_constrained_match
    ):
        rebuild_reasons.append(
            f"{float(confidence):.0%} confidence"
        )
    if projection_scale_ratio is not None:
        scale_error = abs(float(projection_scale_ratio) - 1.0)
        strong_match_scale_ok = (
            strong_constrained_match
            and scale_error <= _RAW_MATCH_STRONG_CONSTRAINED_MAX_SCALE_ERROR_RATIO
        )
        if scale_error > _RAW_MATCH_MAX_SCALE_ERROR_RATIO and not strong_match_scale_ok:
            rebuild_reasons.append(
                f"{float(projection_scale_ratio):.2f}x footprint scale"
            )

    needs_metric_anchor = bool(rebuild_reasons)
    if needs_metric_anchor:
        heading_diff = match_result.get("heading_diff_vs_exif_deg")
        confidence_value = None
        try:
            confidence_value = None if confidence is None else float(confidence)
        except (TypeError, ValueError):
            confidence_value = None
        low_conf_heading_disagrees = (
            camera_heading is not None
            and heading_diff is not None
            and confidence_value is not None
            and confidence_value < _MATCH_WARNING_MIN_CONFIDENCE
            and abs(float(heading_diff)) > _RAW_MATCH_LOW_CONF_HEADING_DIFF_DEG
        )
        heading_for_projection = (
            camera_heading
            if low_conf_heading_disagrees
            else (sift_heading if sift_heading is not None else camera_heading)
        )
        if heading_for_projection is not None:
            rebuilt_h = _build_metric_centered_homography(
                drone_image=drone_image,
                center_lat=center_lat,
                center_lon=center_lon,
                drone_gsd=drone_gsd,
                heading_deg=float(heading_for_projection),
            )
            if rebuilt_h is not None:
                match_result["raw_H"] = match_result.get("H")
                match_result["H"] = rebuilt_h
                match_result["gps_anchored"] = True
                match_result["projection_rebuilt"] = True
                match_result["projection_rotation_source"] = (
                    "exif_heading_metric_anchor"
                    if low_conf_heading_disagrees
                    else "sift_heading_metric_anchor"
                )
                match_result["projection_rebuild_reason"] = (
                    "raw match outside coordinate limits: "
                    + "; ".join(rebuild_reasons)
                )
    return match_result


def _vegetation_alignment_mask(image: np.ndarray) -> np.ndarray:
    """Extract a conservative green-vegetation silhouette for map alignment."""
    if not isinstance(image, np.ndarray) or image.ndim != 3 or image.shape[2] < 3:
        return np.zeros((0, 0), dtype=np.uint8)
    hsv = cv2.cvtColor(image[:, :, :3], cv2.COLOR_BGR2HSV)
    blue, green, red = cv2.split(image[:, :, :3].astype(np.int16))
    excess_green = (2 * green) - red - blue
    selected = (
        (hsv[:, :, 0] >= 24)
        & (hsv[:, :, 0] <= 95)
        & (hsv[:, :, 1] >= 42)
        & (hsv[:, :, 2] >= 35)
        & (excess_green >= 18)
    )
    mask = selected.astype(np.uint8) * 255
    return cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), dtype=np.uint8),
    )


def _centered_array_crop(
    array: np.ndarray,
    center_x: float,
    center_y: float,
    width: int,
    height: int,
) -> np.ndarray:
    """Return a fixed-size crop, padding only the portion outside the source."""
    width = max(1, int(width))
    height = max(1, int(height))
    left = int(round(float(center_x) - (width / 2.0)))
    top = int(round(float(center_y) - (height / 2.0)))
    output_shape = (height, width) + tuple(array.shape[2:])
    output = np.zeros(output_shape, dtype=array.dtype)

    source_left = max(0, left)
    source_top = max(0, top)
    source_right = min(array.shape[1], left + width)
    source_bottom = min(array.shape[0], top + height)
    if source_left >= source_right or source_top >= source_bottom:
        return output

    output[
        source_top - top:source_bottom - top,
        source_left - left:source_right - left,
    ] = array[source_top:source_bottom, source_left:source_right]
    return output


def _rotate_mask_for_heading(mask: np.ndarray, heading_deg: float) -> np.ndarray:
    """Rotate an image-space mask into north-up map space without clipping."""
    height, width = mask.shape[:2]
    center = (width / 2.0, height / 2.0)
    matrix = cv2.getRotationMatrix2D(center, -float(heading_deg), 1.0)
    cos_value = abs(float(matrix[0, 0]))
    sin_value = abs(float(matrix[0, 1]))
    output_width = max(1, int(math.ceil((height * sin_value) + (width * cos_value))))
    output_height = max(1, int(math.ceil((height * cos_value) + (width * sin_value))))
    matrix[0, 2] += (output_width / 2.0) - center[0]
    matrix[1, 2] += (output_height / 2.0) - center[1]
    return cv2.warpAffine(
        mask,
        matrix,
        (output_width, output_height),
        flags=cv2.INTER_NEAREST,
        borderMode=cv2.BORDER_CONSTANT,
        borderValue=0,
    )


def _binary_mask_cosine_score(first: np.ndarray, second: np.ndarray) -> float:
    """Measure overlap while normalizing for different vegetation quantities."""
    first_selected = first > 0
    second_selected = second > 0
    first_count = int(np.count_nonzero(first_selected))
    second_count = int(np.count_nonzero(second_selected))
    if first_count <= 0 or second_count <= 0:
        return 0.0
    intersection = int(np.count_nonzero(first_selected & second_selected))
    return float(intersection / math.sqrt(first_count * second_count))


def _refine_failed_match_with_vegetation_scale(
    failed_match: dict[str, Any],
    drone_image: np.ndarray,
    center_lat: float,
    center_lon: float,
    drone_gsd: float,
    camera_heading: Optional[float],
) -> dict[str, Any]:
    """Refine footprint scale after normal SIFT has failed.

    The GPS center and DJI EXIF heading both remain authoritative. Vegetation
    is useful for estimating how large the frame should be, but orthomosaic
    seams and shoreline change can create a false rotation match. Searching
    scale only prevents those map artifacts from turning the image away from
    its recorded camera orientation.
    """
    if camera_heading is None or not math.isfinite(float(camera_heading)):
        return failed_match
    if not isinstance(drone_image, np.ndarray) or drone_image.ndim != 3:
        return failed_match
    if not math.isfinite(float(drone_gsd)) or float(drone_gsd) <= 0:
        return failed_match

    try:
        active_entry = ortho_matcher.select_orthophoto(center_lat, center_lon)
        orthophoto = ortho_matcher.load_orthophoto()
        ortho_gsd = float(ortho_matcher.ORTHO_GSD or 0.0)
        center_x, center_y = gps_to_ortho_pixel(center_lat, center_lon)
    except Exception:
        return failed_match
    if (
        active_entry is None
        or not isinstance(orthophoto, np.ndarray)
        or orthophoto.ndim != 3
        or ortho_gsd <= 0
    ):
        return failed_match

    image_height, image_width = drone_image.shape[:2]
    ortho_scale = float(drone_gsd) / ortho_gsd
    target_width = int(round(image_width * ortho_scale))
    target_height = int(round(image_height * ortho_scale))
    if target_width < 48 or target_height < 48 or target_width > 6000 or target_height > 6000:
        return failed_match

    # Score on a bounded-size representation. Both the drone image and
    # orthophoto patch use the same downsampling factor, while candidate GSD
    # factors represent genuine changes to the estimated ground footprint.
    score_scale = min(
        1.0,
        _VEGETATION_HEADING_MAX_DIM_PX
        / (max(target_width, target_height) * _VEGETATION_SCALE_MAX_FACTOR),
    )
    nominal_score_width = max(32, int(round(target_width * score_scale)))
    nominal_score_height = max(32, int(round(target_height * score_scale)))
    nominal_drone_for_score = cv2.resize(
        drone_image,
        (nominal_score_width, nominal_score_height),
        interpolation=cv2.INTER_AREA if score_scale < 1.0 else cv2.INTER_LINEAR,
    )
    nominal_drone_mask = _vegetation_alignment_mask(nominal_drone_for_score)
    drone_fraction = float(
        np.count_nonzero(nominal_drone_mask) / max(1, nominal_drone_mask.size)
    )
    if not (
        _VEGETATION_HEADING_MIN_MASK_FRACTION
        <= drone_fraction
        <= _VEGETATION_HEADING_MAX_MASK_FRACTION
    ):
        return failed_match

    patch_side = max(
        96,
        int(
            math.ceil(
                math.hypot(target_width, target_height)
                * _VEGETATION_SCALE_MAX_FACTOR
            )
        )
        + 20,
    )
    half_side = patch_side / 2.0
    source_left = max(0.0, float(center_x) - half_side)
    source_top = max(0.0, float(center_y) - half_side)
    source_right = min(float(orthophoto.shape[1]), float(center_x) + half_side)
    source_bottom = min(float(orthophoto.shape[0]), float(center_y) + half_side)
    valid_fraction = (
        max(0.0, source_right - source_left)
        * max(0.0, source_bottom - source_top)
        / float(patch_side * patch_side)
    )
    if valid_fraction < 0.70:
        return failed_match

    ortho_patch = _centered_array_crop(
        orthophoto,
        center_x,
        center_y,
        patch_side,
        patch_side,
    )
    if score_scale < 1.0:
        scaled_side = max(64, int(round(patch_side * score_scale)))
        ortho_patch = cv2.resize(
            ortho_patch,
            (scaled_side, scaled_side),
            interpolation=cv2.INTER_AREA,
        )
    ortho_mask = _vegetation_alignment_mask(ortho_patch)
    tolerance_radius_px = max(1, int(round((0.20 / ortho_gsd) * score_scale)))
    tolerance_radius_px = min(tolerance_radius_px, 6)
    ortho_mask = cv2.dilate(
        ortho_mask,
        np.ones(
            ((2 * tolerance_radius_px) + 1, (2 * tolerance_radius_px) + 1),
            dtype=np.uint8,
        ),
    )
    if np.count_nonzero(ortho_mask) < 100:
        return failed_match

    patch_center_x = ortho_mask.shape[1] / 2.0
    patch_center_y = ortho_mask.shape[0] / 2.0

    mask_cache: dict[float, np.ndarray] = {1.0: nominal_drone_mask}

    def _mask_for_scale(scale_factor: float) -> np.ndarray:
        cache_key = round(float(scale_factor), 4)
        cached = mask_cache.get(cache_key)
        if cached is not None:
            return cached
        candidate_width = max(
            32,
            int(round(target_width * score_scale * float(scale_factor))),
        )
        candidate_height = max(
            32,
            int(round(target_height * score_scale * float(scale_factor))),
        )
        candidate_image = cv2.resize(
            drone_image,
            (candidate_width, candidate_height),
            interpolation=cv2.INTER_AREA,
        )
        candidate_mask = _vegetation_alignment_mask(candidate_image)
        mask_cache[cache_key] = candidate_mask
        return candidate_mask

    def _score_candidate(scale_factor: float, heading: float) -> float:
        rotated = _rotate_mask_for_heading(
            _mask_for_scale(scale_factor),
            heading,
        )
        comparison = _centered_array_crop(
            ortho_mask,
            patch_center_x,
            patch_center_y,
            rotated.shape[1],
            rotated.shape[0],
        )
        return _binary_mask_cosine_score(rotated, comparison)

    baseline_heading = float(camera_heading) % 360.0
    baseline_score = _score_candidate(1.0, baseline_heading)
    best_heading = baseline_heading
    best_scale_factor = 1.0
    best_score = baseline_score
    coarse_scales = np.arange(
        _VEGETATION_SCALE_MIN_FACTOR,
        _VEGETATION_SCALE_MAX_FACTOR + (_VEGETATION_SCALE_COARSE_STEP / 2.0),
        _VEGETATION_SCALE_COARSE_STEP,
    )
    for scale_factor in coarse_scales:
        score = _score_candidate(float(scale_factor), baseline_heading)
        if score > best_score:
            best_scale_factor = float(scale_factor)
            best_score = score

    # Polish the coarse maximum without increasing the expensive global grid.
    fine_scale_start = max(
        _VEGETATION_SCALE_MIN_FACTOR,
        best_scale_factor - _VEGETATION_SCALE_COARSE_STEP,
    )
    fine_scale_stop = min(
        _VEGETATION_SCALE_MAX_FACTOR,
        best_scale_factor + _VEGETATION_SCALE_COARSE_STEP,
    )
    fine_scales = np.arange(
        fine_scale_start,
        fine_scale_stop + (_VEGETATION_SCALE_FINE_STEP / 2.0),
        _VEGETATION_SCALE_FINE_STEP,
    )
    for scale_factor in fine_scales:
        score = _score_candidate(float(scale_factor), baseline_heading)
        if score > best_score:
            best_scale_factor = float(scale_factor)
            best_score = score

    improvement = float(best_score - baseline_score)
    if (
        best_score < _VEGETATION_HEADING_MIN_SCORE
        or improvement < _VEGETATION_HEADING_MIN_IMPROVEMENT
        or abs(best_scale_factor - 1.0) < 0.03
        or best_scale_factor
        <= (_VEGETATION_SCALE_MIN_FACTOR + (_VEGETATION_SCALE_FINE_STEP / 2.0))
        or best_scale_factor
        >= (_VEGETATION_SCALE_MAX_FACTOR - (_VEGETATION_SCALE_FINE_STEP / 2.0))
    ):
        return failed_match

    refined_gsd = float(drone_gsd) * float(best_scale_factor)
    homography = _build_metric_centered_homography(
        drone_image,
        center_lat,
        center_lon,
        refined_gsd,
        best_heading,
    )
    if homography is None:
        return failed_match

    refined = dict(failed_match)
    refined.update(
        {
            "success": True,
            "H": homography,
            "heading": float(best_heading),
            "confidence": float(best_score),
            "match_score": float(best_score),
            "ortho_name": active_entry.get("name"),
            "ortho_path": str(active_entry.get("path") or ""),
            "gps_anchored": True,
            "projection_constrained": True,
            "projection_metric_scale_normalized": True,
            "projection_edge_refined": False,
            "projection_rebuilt": True,
            "projection_rotation_source": "vegetation_scale_metric_anchor",
            "projection_rebuild_reason": (
                "SIFT unavailable; vegetation alignment retained the DJI EXIF "
                f"heading and refined footprint scale to {best_scale_factor:.2f}x"
            ),
            "heading_diff_vs_exif_deg": 0.0,
            "vegetation_heading_score": float(best_score),
            "vegetation_heading_baseline_score": float(baseline_score),
            "vegetation_heading_improvement": improvement,
            "vegetation_heading_correction_deg": 0.0,
            "vegetation_gsd_scale_factor": float(best_scale_factor),
            "refined_gsd_m_per_pixel": refined_gsd,
            "sift_error": failed_match.get("error"),
            "center_drift_px": 0.0,
            "center_drift_m": 0.0,
            "center_offset_east_m": 0.0,
            "center_offset_north_m": 0.0,
            "error": None,
            "rejected_reason": None,
        }
    )
    scale_diagnostics = _homography_scale_diagnostics(
        homography,
        drone_image,
        refined_gsd,
    )
    for key, value in scale_diagnostics.items():
        if value is not None:
            refined[key] = value
    return refined


def _refine_failed_match_with_edge_translation(
    failed_match: dict[str, Any],
    drone_image: np.ndarray,
    center_lat: float,
    center_lon: float,
    drone_gsd: float,
    camera_heading: Optional[float],
) -> dict[str, Any]:
    """Recover a failed feature match with conservative rigid edge alignment.

    Coastal images can contain mostly water and mud, leaving too few stable
    SIFT keypoints even though their shoreline is clear. In that case the
    EXIF heading and camera-derived GSD still define a reliable rigid
    footprint; only its centre may be displaced because the EXIF coordinate
    is the aircraft position rather than the exact ground-frame centre.

    The fallback never changes the physical image scale. It first tests the
    EXIF heading and nearby half-degree headings, then permits a correction
    only when an interior candidate is measurably better than the EXIF
    baseline. Every candidate must also pass the existing strict translation
    score and improvement gates. The bounded search prevents a coincidental
    shoreline edge from inventing a materially different orientation.
    """
    if camera_heading is None or not math.isfinite(float(camera_heading)):
        return failed_match
    if not isinstance(drone_image, np.ndarray) or drone_image.ndim != 3:
        return failed_match
    if not math.isfinite(float(drone_gsd)) or float(drone_gsd) <= 0:
        return failed_match

    try:
        active_entry = ortho_matcher.select_orthophoto(center_lat, center_lon)
    except Exception:
        active_entry = None
    if active_entry is None:
        return failed_match

    exif_heading = float(camera_heading) % 360.0
    heading_offsets = np.arange(
        -_FAILED_MATCH_HEADING_SEARCH_DEG,
        _FAILED_MATCH_HEADING_SEARCH_DEG + (_FAILED_MATCH_HEADING_STEP_DEG * 0.5),
        _FAILED_MATCH_HEADING_STEP_DEG,
    )
    candidates: list[dict[str, Any]] = []
    for raw_offset in heading_offsets:
        offset = float(raw_offset)
        candidate_heading = (exif_heading + offset) % 360.0
        metric_h = _build_metric_centered_homography(
            drone_image,
            center_lat,
            center_lon,
            float(drone_gsd),
            candidate_heading,
        )
        if metric_h is None:
            continue
        candidate_h, diagnostics = _refine_homography_by_edge_translation(
            drone_image,
            metric_h,
        )
        candidate_score = _safe_float(diagnostics.get("edge_refine_score"))
        candidate_improvement = _safe_float(
            diagnostics.get("edge_refine_improvement")
        )
        if (
            candidate_h is None
            or candidate_score is None
            or candidate_improvement is None
            or candidate_score < _FAILED_MATCH_EDGE_MIN_SCORE
            or candidate_improvement < _FAILED_MATCH_EDGE_MIN_IMPROVEMENT
        ):
            continue
        candidates.append(
            {
                "offset": offset,
                "heading": candidate_heading,
                "H": candidate_h,
                "score": candidate_score,
                "diagnostics": diagnostics,
            }
        )

    baseline = next(
        (candidate for candidate in candidates if abs(candidate["offset"]) < 1e-9),
        None,
    )
    if baseline is None:
        return failed_match

    selected = baseline
    best = max(candidates, key=lambda candidate: candidate["score"])
    score_gain = float(best["score"] - baseline["score"])
    best_at_search_boundary = (
        abs(float(best["offset"]))
        >= _FAILED_MATCH_HEADING_SEARCH_DEG - 1e-9
    )
    if (
        not best_at_search_boundary
        and abs(float(best["offset"])) >= (_FAILED_MATCH_HEADING_STEP_DEG - 1e-9)
        and score_gain >= _FAILED_MATCH_HEADING_MIN_SCORE_GAIN
    ):
        selected = best

    refined_h = selected["H"]
    edge_diagnostics = selected["diagnostics"]
    score = float(selected["score"])
    heading_correction = float(selected["offset"])
    selected_heading = float(selected["heading"])
    selected_score_gain = float(score - baseline["score"])

    center_diagnostics = _homography_center_diagnostics(
        refined_h,
        drone_image,
        center_lat,
        center_lon,
    )
    shift_m = _safe_float(edge_diagnostics.get("edge_refine_shift_m")) or 0.0
    refined = dict(failed_match)
    refined.update(
        {
            "success": True,
            "H": refined_h,
            "heading": selected_heading,
            "confidence": float(score),
            "match_score": float(score),
            "ortho_name": active_entry.get("name"),
            "ortho_path": str(active_entry.get("path") or ""),
            "gps_anchored": True,
            "projection_constrained": True,
            "projection_metric_scale_normalized": True,
            "projection_edge_refined": True,
            "projection_rebuilt": True,
            "projection_rotation_source": (
                "exif_heading_metric_edge_alignment"
                if abs(heading_correction) >= 1e-9
                else "exif_heading_metric_edge_translation"
            ),
            "projection_rebuild_reason": (
                "SIFT unavailable; retained physical GSD, corrected the DJI "
                f"EXIF heading by {heading_correction:+.1f} degrees, then "
                f"corrected the ground footprint by {shift_m:.2f} m using "
                "orthophoto edges"
                if abs(heading_correction) >= 1e-9
                else "SIFT unavailable; retained DJI EXIF heading and physical "
                f"GSD, then corrected the ground footprint by {shift_m:.2f} m "
                "using orthophoto edges"
            ),
            "heading_diff_vs_exif_deg": heading_correction,
            "edge_heading_baseline_score": float(baseline["score"]),
            "edge_heading_score_gain": selected_score_gain,
            "edge_heading_correction_deg": heading_correction,
            "sift_error": failed_match.get("error"),
            "error": None,
            "rejected_reason": None,
            **edge_diagnostics,
            **center_diagnostics,
        }
    )
    scale_diagnostics = _homography_scale_diagnostics(
        refined_h,
        drone_image,
        float(drone_gsd),
    )
    for key, value in scale_diagnostics.items():
        if value is not None:
            refined[key] = value
    return refined


def _match_drone_to_ortho_robust(
    drone_image: np.ndarray,
    center_lat: float,
    center_lon: float,
    drone_gsd: float,
    camera_heading: Optional[float] = None,
) -> dict[str, Any]:
    """Run SIFT matching with a wider retry before allowing heading fallback.

    A high-quality SIFT homography is used directly. A low-confidence or
    high-drift SIFT match contributes only its heading while scale/translation
    come from GSD and the GPS-tagged image center. This avoids letting noisy
    SIFT scale or translation move export coordinates away from the photo.

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
    vegetation_refined = _refine_failed_match_with_vegetation_scale(
        retry,
        drone_image,
        center_lat,
        center_lon,
        drone_gsd,
        camera_heading,
    )
    if vegetation_refined.get("success"):
        return vegetation_refined
    return _refine_failed_match_with_edge_translation(
        retry,
        drone_image,
        center_lat,
        center_lon,
        drone_gsd,
        camera_heading,
    )


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

    try:
        visible = overlay_visibility_mask(
            ortho_matcher._ensure_active_ortho(), min_x, min_y, out_w, out_h, warp_w, warp_h,
        )
    except Exception:
        return None  # Never draw an unverified overlay beyond the map edge.
    overlay_bgra = cv2.cvtColor(warped, cv2.COLOR_BGR2BGRA)
    overlay_bgra[:, :, 3] = cv2.bitwise_and(alpha, visible)

    north_lat, west_lon = ortho_pixel_to_gps(min_x, min_y)
    south_lat, east_lon = ortho_pixel_to_gps(max_x, max_y)

    footprint_ring = []
    try:
        for corner_x, corner_y in ortho_corners:
            corner_lat, corner_lon = ortho_pixel_to_gps(float(corner_x), float(corner_y))
            if not (math.isfinite(float(corner_lat)) and math.isfinite(float(corner_lon))):
                footprint_ring = []
                break
            footprint_ring.append([float(corner_lon), float(corner_lat)])
        if footprint_ring:
            footprint_ring.append(list(footprint_ring[0]))
    except Exception:
        footprint_ring = []

    return {
        "image_data_url": _encode_image_data_url(overlay_bgra, ".png"),
        "bounds": [
            [float(south_lat), float(west_lon)],
            [float(north_lat), float(east_lon)],
        ],
        "opacity": 0.72,
        "footprint_geojson": (
            {"type": "Polygon", "coordinates": [footprint_ring]}
            if len(footprint_ring) == 5 else None
        ),
    }


def _safe_float(value: Any) -> Optional[float]:
    """Coerce a value to float when possible."""
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _require_metric_value(
    name: str,
    value: Any,
    *,
    min_value: float,
    max_value: float,
    allow_zero: bool = False,
) -> float:
    """Validate user-controlled metric inputs before they become pixel sizes."""
    parsed = _safe_float(value)
    if parsed is None or not math.isfinite(parsed):
        raise HTTPException(status_code=400, detail=f"{name} must be a finite number.")
    if allow_zero and parsed == 0:
        return 0.0
    if parsed < min_value or parsed > max_value:
        raise HTTPException(
            status_code=400,
            detail=f"{name} must be between {min_value:g} and {max_value:g}.",
        )
    return float(parsed)


def _is_reasonable_altitude(value: Any) -> bool:
    parsed = _safe_float(value)
    return (
        parsed is not None
        and math.isfinite(parsed)
        and _MIN_ANALYSIS_ALTITUDE_M <= parsed <= _MAX_ANALYSIS_ALTITUDE_M
    )


def _compute_canopy_area_m2(results: dict[str, Any]) -> float:
    """Canopy area in m² from the raw mask, falling back to polygons."""
    mask = results.get("canopy_mask")
    gsd = results.get("gsd_m_per_pixel")
    if gsd is None:
        return 0.0

    gsd_m = float(gsd)
    if isinstance(mask, np.ndarray):
        mask_area_m2 = float(np.count_nonzero(mask) * (gsd_m ** 2))
        if mask_area_m2 > 0:
            return mask_area_m2

    polygons = [
        polygon
        for polygon in (results.get("canopy_polygons") or [])
        if polygon is not None and not getattr(polygon, "is_empty", True)
    ]
    if not polygons:
        return 0.0

    try:
        return float(unary_union(polygons).area * (gsd_m ** 2))
    except Exception:
        return float(sum(float(getattr(poly, "area", 0.0) or 0.0) for poly in polygons) * (gsd_m ** 2))


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


def _image_edge_safety_state(
    center: Optional[tuple[float, float]],
    image_w: int,
    image_h: int,
    gsd_m_per_pixel: float,
    canopy_buffer_m: float,
    core_radius_px: float,
    orthophoto_verified: bool,
) -> tuple[bool, bool]:
    """Return ``(reject, needs_full_ortho_recheck)`` for an image-edge point.

    A planting point must always retain the configured one-metre inward gap
    from the analyzed raster edge. The wider canopy-clearance radius may extend
    past that gap only when a trusted orthophoto registration can inspect the
    otherwise unseen neighborhood.
    """
    if center is None or image_w <= 0 or image_h <= 0:
        return True, False
    try:
        px, py = float(center[0]), float(center[1])
    except (TypeError, ValueError, IndexError):
        return True, False
    if not (np.isfinite(px) and np.isfinite(py)):
        return True, False
    if px < 0 or py < 0 or px >= image_w or py >= image_h:
        return True, False

    edge_distance_px = min(px, py, (image_w - 1) - px, (image_h - 1) - py)
    core_clearance_px = max(0.5, float(core_radius_px) + 0.5)
    mandatory_photo_clearance_px = max(
        core_clearance_px,
        _IMAGE_EDGE_MIN_CLEARANCE_M / max(float(gsd_m_per_pixel), 1e-9),
    )
    if edge_distance_px <= mandatory_photo_clearance_px:
        return True, False

    full_clearance_px = max(
        mandatory_photo_clearance_px,
        max(0.0, float(canopy_buffer_m)) / max(float(gsd_m_per_pixel), 1e-9),
    )
    if edge_distance_px <= full_clearance_px:
        if orthophoto_verified:
            return False, True
        return True, False
    return False, False


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


def _build_reachable_map_component_mask(
    forbidden_mask: Optional[np.ndarray],
    anchor: Optional[tuple[float, float]],
) -> Optional[np.ndarray]:
    """Return the non-forbidden image region connected to the camera center.

    Some mapped forbidden polygons are seawalls or revetments that form the
    physical edge of the supported map. The orthophoto itself is rectangular,
    so a raster-bounds check alone incorrectly accepts water on the far side of
    such a barrier. Keeping only the connected non-forbidden component that
    contains the geotagged image center makes the barrier act as the true map
    boundary while leaving ordinary isolated forbidden polygons unaffected.
    """
    if (
        not isinstance(forbidden_mask, np.ndarray)
        or forbidden_mask.ndim != 2
        or anchor is None
    ):
        return None

    open_pixels = (forbidden_mask == 0).astype(np.uint8)
    if not np.any(open_pixels):
        return np.zeros_like(forbidden_mask, dtype=np.uint8)

    component_count, labels, stats, _ = cv2.connectedComponentsWithStats(
        open_pixels,
        connectivity=8,
    )
    if component_count <= 1:
        return np.zeros_like(forbidden_mask, dtype=np.uint8)

    try:
        anchor_x = int(round(float(anchor[0])))
        anchor_y = int(round(float(anchor[1])))
    except (TypeError, ValueError, IndexError):
        return None
    anchor_x = min(max(anchor_x, 0), forbidden_mask.shape[1] - 1)
    anchor_y = min(max(anchor_y, 0), forbidden_mask.shape[0] - 1)
    anchor_label = int(labels[anchor_y, anchor_x])

    # A mapped structure may cover the exact center pixel. In that unusual
    # case, retain the largest non-forbidden component instead of disabling the
    # safety gate or guessing that every disconnected area is valid.
    if anchor_label == 0:
        component_areas = stats[1:, cv2.CC_STAT_AREA]
        if component_areas.size == 0:
            return np.zeros_like(forbidden_mask, dtype=np.uint8)
        anchor_label = int(np.argmax(component_areas)) + 1

    return ((labels == anchor_label).astype(np.uint8) * 255)


def _center_is_inside_map_component(
    map_component_mask: Optional[np.ndarray],
    center: Optional[tuple[float, float]],
) -> bool:
    """Return whether a point lies in the retained image-side map component."""
    if map_component_mask is None:
        return True
    if center is None:
        return False
    try:
        px, py = center
        cx = int(round(float(px)))
        cy = int(round(float(py)))
    except (TypeError, ValueError, IndexError):
        return False
    if (
        cx < 0
        or cy < 0
        or cy >= map_component_mask.shape[0]
        or cx >= map_component_mask.shape[1]
    ):
        return False
    return bool(map_component_mask[cy, cx] > 0)


def _reproject_hexagons_for_visualization(
    detector: Any,
    hexagons: list[dict[str, Any]],
    gps_to_pixel: Optional[Any],
    hexagon_size_m: float,
    gsd_m_per_pixel: float,
) -> list[dict[str, Any]]:
    """Build display-only hexagon geometry from final GPS coordinates."""
    display_hexagons = [copy.deepcopy(hexagon) for hexagon in hexagons]
    if gps_to_pixel is None:
        return display_hexagons

    buffer_radius_px = float(hexagon_size_m) / max(float(gsd_m_per_pixel), 1e-9)
    core_radius_px = buffer_radius_px * 0.2
    for hexagon in display_hexagons:
        lat = hexagon.get("_gps_lat")
        lon = hexagon.get("_gps_lon")
        if lat is None or lon is None:
            continue
        try:
            px, py = gps_to_pixel(float(lat), float(lon))
        except Exception:
            continue
        if not (np.isfinite(px) and np.isfinite(py)):
            continue
        px = float(px)
        py = float(py)
        hexagon["center"] = (px, py)
        hexagon["buffer"] = detector.create_hexagon(px, py, buffer_radius_px)
        hexagon["core"] = detector.create_hexagon(px, py, core_radius_px)
    return display_hexagons


def _polygon_intersects_mask(polygon: Any, mask: Optional[np.ndarray]) -> bool:
    """Return whether a pixel-space polygon touches a non-zero zone mask."""
    if not isinstance(mask, np.ndarray) or mask.ndim != 2 or not np.any(mask):
        return False

    try:
        exterior = getattr(polygon, "exterior", None)
        coordinates = exterior.coords if exterior is not None else polygon
        points = np.asarray(coordinates, dtype=np.float64).reshape(-1, 2)
    except Exception:
        return False
    if len(points) < 3 or not np.all(np.isfinite(points)):
        return False

    h, w = mask.shape
    x1 = max(0, int(math.floor(float(np.min(points[:, 0])))))
    y1 = max(0, int(math.floor(float(np.min(points[:, 1])))))
    x2 = min(w, int(math.ceil(float(np.max(points[:, 0])))) + 1)
    y2 = min(h, int(math.ceil(float(np.max(points[:, 1])))) + 1)
    if x1 >= x2 or y1 >= y2:
        return False

    zone_roi = mask[y1:y2, x1:x2]
    if not np.any(zone_roi):
        return False

    local_points = np.rint(points - np.array([x1, y1], dtype=np.float64)).astype(np.int32)
    polygon_roi = np.zeros(zone_roi.shape, dtype=np.uint8)
    cv2.fillPoly(polygon_roi, [local_points], 255)
    return bool(np.any((polygon_roi > 0) & (zone_roi > 0)))


def _hexagon_core_intersects_any_mask(
    hexagon: dict[str, Any],
    zone_masks: tuple[Optional[np.ndarray], ...],
) -> bool:
    """Check the actual planting core, not its outer spacing/visual buffer."""
    polygon = hexagon.get("core")
    if polygon is None:
        polygon = hexagon.get("buffer")
    if polygon is None:
        return False
    return any(_polygon_intersects_mask(polygon, mask) for mask in zone_masks)


def _hexagon_buffer_intersects_any_mask(
    hexagon: dict[str, Any],
    zone_masks: tuple[Optional[np.ndarray], ...],
) -> bool:
    """Check the full displayed planting marker against exclusion masks."""
    polygon = hexagon.get("buffer")
    if polygon is None:
        polygon = hexagon.get("core")
    if polygon is None:
        return False
    return any(_polygon_intersects_mask(polygon, mask) for mask in zone_masks)


def _build_authoritative_visualization_hexagons(
    detector: Any,
    safe_hexagons: list[dict[str, Any]],
    visualization_gps_to_pixel: Optional[Any],
    hexagon_size_m: float,
    gsd_m_per_pixel: float,
) -> list[dict[str, Any]]:
    """Render one preview marker for every final map/export planting point.

    ``safe_hexagons`` has already passed all forbidden, danger, coverage, and
    spacing filters. A display-only filter based on a different
    visual homography must not change that final membership: doing so makes the
    raster preview disagree with the orthophoto and exported coordinates.
    """
    visualization_hexagons = _reproject_hexagons_for_visualization(
        detector,
        safe_hexagons,
        visualization_gps_to_pixel,
        hexagon_size_m,
        gsd_m_per_pixel,
    )
    if len(visualization_hexagons) != len(safe_hexagons):
        raise RuntimeError("Visualization point count diverged from authoritative safe points")
    return visualization_hexagons


def _gps_exclusion_reason(
    latitude: float,
    longitude: float,
    forbidden_filter: Any,
    eroded_filter: Any,
    gis_coverage_geometry: Optional[Any] = None,
) -> Optional[str]:
    """Return the exclusion or map boundary covering a GPS point, if any."""
    if not forbidden_filter.is_safe_location(float(latitude), float(longitude)):
        return "forbidden"
    if not eroded_filter.is_safe_location(float(latitude), float(longitude)):
        return "eroded"
    if gis_coverage_geometry is not None:
        point = Point(float(longitude), float(latitude))
        if not gis_coverage_geometry.covers(point):
            return "outside_map"
    return None


def _reconcile_authoritative_hexagons_with_visual_zones(
    detector: Any,
    safe_hexagons: list[dict[str, Any]],
    visualization_gps_to_pixel: Optional[Any],
    hexagon_size_m: float,
    gsd_m_per_pixel: float,
    forbidden_mask: Optional[np.ndarray],
    eroded_mask: Optional[np.ndarray],
) -> tuple[
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
    list[dict[str, Any]],
]:
    """Apply one coherent visual transform, then veto all zone conflicts.

    A point whose displayed planting core intersects a projected forbidden or
    eroded zone is removed from both the authoritative map/export set and the
    image overlay. Checking the core matches the authoritative candidate rule;
    checking the outer spacing buffer globally would double-apply zone
    clearance.

    Edge-adjacent points are the exception. Their wider canopy-clearance area
    extends beyond the uploaded raster and is verified from the orthophoto, so
    registration uncertainty is highest there. For those explicitly tagged
    points, reject a full displayed marker that overlaps the already-buffered
    forbidden mask. This prevents a partially visible marker from landing on
    a structure without reducing interior core-safe planting coverage.
    """
    visual_hexagons = _build_authoritative_visualization_hexagons(
        detector,
        safe_hexagons,
        visualization_gps_to_pixel,
        hexagon_size_m,
        gsd_m_per_pixel,
    )
    kept_safe: list[dict[str, Any]] = []
    kept_visual: list[dict[str, Any]] = []
    forbidden_conflicts: list[dict[str, Any]] = []
    eroded_conflicts: list[dict[str, Any]] = []

    for safe_hexagon, visual_hexagon in zip(safe_hexagons, visual_hexagons):
        edge_forbidden_overlap = bool(
            safe_hexagon.get("_requires_full_edge_ortho_recheck")
        ) and _hexagon_buffer_intersects_any_mask(
            visual_hexagon,
            (forbidden_mask,),
        )
        if edge_forbidden_overlap or _hexagon_core_intersects_any_mask(
            visual_hexagon,
            (forbidden_mask,),
        ):
            conflict = copy.deepcopy(safe_hexagon)
            conflict["_visual_zone_conflict"] = "forbidden"
            forbidden_conflicts.append(conflict)
            continue
        if _hexagon_core_intersects_any_mask(
            visual_hexagon,
            (eroded_mask,),
        ):
            conflict = copy.deepcopy(safe_hexagon)
            conflict["_visual_zone_conflict"] = "eroded"
            eroded_conflicts.append(conflict)
            continue
        kept_safe.append(safe_hexagon)
        kept_visual.append(visual_hexagon)

    return kept_safe, kept_visual, forbidden_conflicts, eroded_conflicts


# MIGRATED FROM app.py: _apply_forbidden_zone_canopy_exclusion lines 2229-2282.
# Kept as an optional fallback only. The newer canopy cleanup rules already
# suppress many mud/structure false positives, so automatically erasing canopy
# inside forbidden polygons is disabled by default. Re-enable with
# MANGROVISION_FORBIDDEN_CANOPY_EXCLUSION=1 if structure false positives return.
def _apply_forbidden_zone_canopy_exclusion(detector, results, forbidden_mask):
    """Remove canopy pixels inside structural forbidden zones and rebuild the dependent outputs.

    Two-stage exclusion so that user-traced forbidden polygons (which are
    typically slightly tighter than the actual man-made structure) still
    catch the canopy on the structure's edges:

      1. Dilate the forbidden mask by the canopy buffer width so nearby
         bridge/roof/water-edge spillover gets treated as forbidden too.
      2. After re-extracting polygons, drop any whole polygon whose area
         is at least 10 % inside the original (un-dilated) forbidden mask - these
         are slivers extending out of a structure that survived the pixel
         mask but conceptually still belong to the forbidden zone.
    """
    canopy_mask = results.get("canopy_mask")
    if not isinstance(canopy_mask, np.ndarray) or canopy_mask.shape != forbidden_mask.shape:
        results["_forbidden_canopy_removed_pixels"] = 0
        return results

    gsd = float(results.get("gsd_m_per_pixel") or 0.05) or 0.05
    buffer_m = float(results.get("canopy_buffer_m") or 1.0) or 1.0
    dilation_px = max(1, int(round(buffer_m / gsd)))
    if not np.any(forbidden_mask):
        expanded_forbidden = np.zeros_like(forbidden_mask)
    else:
        image_h, image_w = forbidden_mask.shape[:2]
        image_diagonal_px = int(math.ceil(math.hypot(image_w, image_h)))
        if dilation_px >= image_diagonal_px:
            expanded_forbidden = np.full_like(forbidden_mask, 255)
        elif dilation_px > 256:
            distance_to_forbidden = cv2.distanceTransform(
                (forbidden_mask == 0).astype(np.uint8),
                cv2.DIST_L2,
                5,
            )
            expanded_forbidden = (
                ((distance_to_forbidden <= dilation_px) | (forbidden_mask > 0)).astype(np.uint8)
                * 255
            )
        else:
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
    # A strict 10% threshold catches bridge/roof spillover that survived pixel clipping.
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
                if overlap_px / poly_area_px >= 0.10:
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
        extra_mask=results.get("seedling_mask"),
        extra_buffer_m=results.get("seedling_buffer_m", 0.0),
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


SPECIES_SPACING_M: dict[str, float] = {
    "bungalon": 1.0,
    "rhizophora": 2.0,
}

def _resolve_hexagon_size(species: Optional[str], hexagon_size: float) -> tuple[float, Optional[str]]:
    """Translate a species choice into an authoritative hexagon_size.

    The frontend already converts species → hexagon_size before sending the
    request, but enforcing the mapping here as well means a stale or hand-
    rolled client can't bypass the agreed planting distance for each species.
    Returns the (hexagon_size, canonical_species_key) pair to use.
    """
    if not species:
        return float(hexagon_size), None
    key = str(species).strip().lower()
    target_spacing = SPECIES_SPACING_M.get(key)
    if target_spacing is None:
        return float(hexagon_size), None
    # For the edge-share (flat-top) lattice the nearest-neighbour distance
    # is R * sqrt(3). To hit the species target T exactly, set R = T / sqrt(3)
    # so neighbour distance = T. See _FINAL_POINT_SPACING_FACTOR at module top.
    return target_spacing / _FINAL_POINT_SPACING_FACTOR, key


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
    species: Optional[str] = None,
    allow_partial_map_overlap: bool = False,
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

    altitude = _require_metric_value(
        "altitude",
        altitude,
        min_value=_MIN_ANALYSIS_ALTITUDE_M,
        max_value=_MAX_ANALYSIS_ALTITUDE_M,
    )
    canopy_buffer = _require_metric_value(
        "canopy_buffer",
        canopy_buffer,
        min_value=0.0,
        max_value=_MAX_CANOPY_BUFFER_M,
        allow_zero=True,
    )
    hexagon_size = _require_metric_value(
        "hexagon_size",
        hexagon_size,
        min_value=_MIN_HEXAGON_SIZE_M,
        max_value=_MAX_HEXAGON_SIZE_M,
    )

    # Species → spacing enforcement. A valid species choice overrides the
    # hexagon_size value (frontend already does the math, but recomputing
    # server-side guarantees the agreed planting distance for each species
    # regardless of what the client sent).
    hexagon_size, canonical_species = _resolve_hexagon_size(species, hexagon_size)
    hexagon_size = _require_metric_value(
        "hexagon_size",
        hexagon_size,
        min_value=_MIN_HEXAGON_SIZE_M,
        max_value=_MAX_HEXAGON_SIZE_M,
    )
    if canonical_species:
        info_messages.append(
            f"Spacing locked to {SPECIES_SPACING_M[canonical_species]:.1f} m for {canonical_species}."
        )

    _emit(progress_cb, "Checking image location against the map", 2)
    # MIGRATED FROM app.py: analyze_image lines 5038-5121
    metadata = ExifExtractor.extract_all_metadata(str(temp_path))
    location_preflight = _build_image_location_preflight(metadata, altitude, drone_model)
    if location_preflight.get("can_process"):
        # Keep the map preview, metre-based detector geometry, and persisted
        # footprint on the same calibrated coordinate model.
        location_preflight = _calibrate_preflight_footprint(
            temp_path,
            location_preflight,
        )
    _enforce_image_location_preflight(
        location_preflight,
        allow_partial_map_overlap=allow_partial_map_overlap,
    )

    _emit(progress_cb, "Checking AI availability", 8)
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
    precomputed_match_result: Optional[dict[str, Any]] = None
    analysis_gsd_override: Optional[float] = None

    if metadata.get("has_gps"):
        gps = metadata["gps"]
        image_center_lat = gps["latitude"]
        image_center_lon = gps["longitude"]

        gps_valid = True
        image_gps = gps
        if location_preflight.get("status") == "partial":
            warning_messages.append(
                "The user confirmed processing for an image whose estimated footprint "
                "partially overlaps the GIS map."
            )
        else:
            info_messages.append(
                f"GPS footprint is inside map bounds: {image_center_lat:.6f}, {image_center_lon:.6f}."
            )

        overlaps = find_overlapping_analyses(image_center_lat, image_center_lon)
        nearby_points = count_nearby_points(image_center_lat, image_center_lon)
        if overlaps and nearby_points:
            warning_messages.append(
                f"This area already has planting data nearby: {len(overlaps)} saved analyses and {nearby_points} saved planting points."
            )

        detected_altitude = None
        detected_altitude_label = None
        if "relative_altitude" in gps and gps["relative_altitude"] is not None:
            detected_altitude = _safe_float(gps["relative_altitude"])
            detected_altitude_label = "AGL from DJI XMP"
        elif "altitude" in gps and gps["altitude"] is not None:
            detected_altitude = _safe_float(gps["altitude"])
            detected_altitude_label = "MSL from EXIF"

        if detected_altitude is not None:
            if _is_reasonable_altitude(detected_altitude):
                altitude_to_use = float(detected_altitude)
                info_messages.append(
                    f"Using detected altitude: {altitude_to_use:.1f} m ({detected_altitude_label})."
                )
            else:
                warning_messages.append(
                    "Ignored suspicious detected altitude "
                    f"{detected_altitude:g} m ({detected_altitude_label}); "
                    f"using form altitude {altitude_to_use:.1f} m instead."
                )

        if metadata.get("camera"):
            drone_to_use = ExifExtractor.detect_drone_model(metadata["camera"])
            info_messages.append(f"Detected drone: {drone_to_use.replace('_', ' ')}.")

        if gps.get("heading") is not None:
            camera_heading = float(gps["heading"])
            heading_source = gps.get("heading_source") or "EXIF GPSImgDirection"
        info_messages.append(f"Using heading {camera_heading:.1f}° ({heading_source}).")
    else:
        warning_messages.append("No GPS data found in the image. Geotagged map outputs and database save are unavailable.")

    # Calibrate the image footprint before generating metre-based danger
    # buffers and planting spacing. Previously matching ran only after all
    # hexagons were created, so a corrected map footprint could look smaller
    # while retaining point spacing calculated from the oversized nominal GSD.
    if image_gps is not None:
        early_match_image = cv2.imread(str(temp_path))
        if isinstance(early_match_image, np.ndarray):
            early_height, early_width = early_match_image.shape[:2]
            nominal_gsd, _ = GSDCalculator.calculate_gsd_from_metadata(
                altitude_m=altitude_to_use,
                camera_info=metadata.get("camera") or {},
                drone_model=drone_to_use,
                image_width_px=early_width,
                image_height_px=early_height,
            )
            _emit(progress_cb, "Calibrating image footprint to orthophoto", 10)
            # Preflight caches only a coarse GPS-centred footprint, not the
            # measured visual transform. Reconstructing H from that cache used
            # to skip SIFT and discard the ground-landmark alignment entirely.
            precomputed_match_result = _match_drone_to_ortho_robust(
                drone_image=early_match_image,
                center_lat=image_center_lat,
                center_lon=image_center_lon,
                drone_gsd=float(nominal_gsd),
                camera_heading=camera_heading,
            )
            refined_gsd = _safe_float(
                precomputed_match_result.get("refined_gsd_m_per_pixel")
            )
            if refined_gsd is not None and refined_gsd > 0:
                analysis_gsd_override = refined_gsd
                footprint_scale = float(
                    precomputed_match_result.get("vegetation_gsd_scale_factor")
                    or (refined_gsd / max(float(nominal_gsd), 1e-9))
                )
                info_messages.append(
                    "Orthophoto shoreline/vegetation calibration adjusted the "
                    f"image footprint to {footprint_scale:.2f}x the altitude-based estimate."
                )

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
    active_zones_revision = zone_revision_fingerprint()

    tuning_fingerprint = json.dumps(ai_runtime_tuning_dict or {}, sort_keys=True)
    analysis_key = (
        f"{uploaded_name}_{altitude_to_use}_{drone_to_use}_{canopy_buffer}_"
        f"{hexagon_size}_{ai_confidence}_{detection_mode}_"
        f"{canopy_code_mtime}_{proper_code_mtime}_{ortho_code_mtime}_"
        f"{active_zones_revision}_"
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
        gsd_override=analysis_gsd_override,
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

    _emit(progress_cb, "Loading GIS zone layers", 55)
    forbidden_filter, eroded_filter = _load_zone_filters()
    gis_coverage_geometry = _load_gis_coverage_geometry()
    if eroded_filter.zone_count > 0:
        info_messages.append(
            f"{eroded_filter.zone_count} eroded zone(s) loaded as planting-point exclusions."
        )
    match_result = precomputed_match_result
    results["_forbidden_filtered"] = 0
    results["_eroded_filtered"] = 0
    results["_post_snap_danger_filtered"] = 0
    results["_forbidden_canopy_removed_pixels"] = 0
    results["_forbidden_canopy_removed_count"] = 0
    results["_forbidden_canopy_exclusion_enabled"] = bool(_FORBIDDEN_CANOPY_EXCLUSION_ENABLED)
    final_lattice_refill_fn = None
    _forbidden_point_mask: Optional[np.ndarray] = None
    _eroded_point_mask: Optional[np.ndarray] = None
    _reachable_map_component_mask: Optional[np.ndarray] = None
    coordinate_homography: Optional[np.ndarray] = None
    coordinate_homography_available = False
    coordinate_homography_trusted = False

    # MIGRATED FROM app.py: analyze_image lines 5294-5382
    if image_gps is not None:
        _emit(progress_cb, "Matching drone image to orthophoto", 62)
        _gsd = results["gsd_m_per_pixel"]
        _w, _h = results["image_size"]

        if match_result is None:
            match_result = _match_drone_to_ortho_robust(
                drone_image=results["image"],
                center_lat=image_center_lat,
                center_lon=image_center_lon,
                drone_gsd=_gsd,
                camera_heading=camera_heading,
            )
        match_warning = _match_quality_warning(match_result)
        if match_warning and match_warning not in warning_messages:
            warning_messages.append(match_warning)

        coordinate_homography = _coordinate_homography_for_match(match_result)
        if coordinate_homography is not None:
            _H = coordinate_homography

            def _px_to_gps(px, py):
                return drone_pixel_to_gps_via_homography(px, py, _H)

            try:
                _H_inv = np.linalg.inv(_H)

                def _gps_to_px(lat, lon):
                    return _gps_to_drone_pixel_via_homography(lat, lon, _H_inv)

                coordinate_homography_available = True
                coordinate_homography_trusted = not bool(
                    match_result.get("projection_rebuilt")
                )
            except Exception:
                coordinate_homography = None
                coordinate_homography_available = False
                coordinate_homography_trusted = False

        if not coordinate_homography_available:

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
            _forbidden_point_mask = _build_zone_mask_in_drone_pixels(
                results["image"].shape[:2],
                forbidden_filter.buffered_polygons,
                _gps_to_px,
                max_polygon_area_fraction=0.90,
                max_total_area_fraction=0.95,
            )
            _reachable_map_component_mask = _build_reachable_map_component_mask(
                _forbidden_point_mask,
                (float(_w) * 0.5, float(_h) * 0.5),
            )
        if eroded_filter.zone_count > 0:
            _eroded_point_mask = _build_zone_mask_in_drone_pixels(
                results["image"].shape[:2],
                eroded_filter.buffered_polygons,
                _gps_to_px,
                max_polygon_area_fraction=0.90,
                max_total_area_fraction=0.95,
            )
        if forbidden_filter.zone_count > 0 and _FORBIDDEN_CANOPY_EXCLUSION_ENABLED:
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
        elif forbidden_filter.zone_count > 0:
            info_messages.append(
                "Forbidden-zone canopy cleanup is disabled; forbidden zones still filter planting points."
            )

        _emit(progress_cb, "Filtering planting points against zones", 74)

        # Snap each hexagon's GPS coords to a shared global hex lattice so
        # adjacent image analyses produce continuous, evenly-spaced points
        # instead of misaligned per-image grids. The lattice is defined in
        # METERS (not pixels) using a fixed lat/lon origin (0, 0) so every
        # analysis at the same site lands on the same nodes regardless of
        # per-image GSD or grid-phase choices in the placement algorithm.
        # Edge-share (flat-top) tessellation — matches the placement
        # geometry in canopy_detector_hexagon.py:
        #   - column spacing        = 1.5 * R_m
        #   - within-column spacing = sqrt(3) * R_m   (= target spacing)
        #   - odd columns shifted down by sqrt(3)/2 * R_m
        # Each hex snaps to its nearest lattice node (max ~R_m away). Two
        # hexes that snap to the same (col, row) cell are deduplicated —
        # this naturally merges Phase 2/3 fillers with Phase 1 hexes.
        _R_m = float(hexagon_size)
        _H_m = 1.5 * _R_m
        _V_m = math.sqrt(3.0) * _R_m
        _M_PER_DEG_LAT = 111_320.0
        _lat_ref = image_center_lat if image_center_lat is not None else 0.0
        _m_per_deg_lon = _M_PER_DEG_LAT * max(0.2, math.cos(math.radians(_lat_ref)))

        def _snap_to_nearest_lattice_node(my: float, mx: float) -> tuple[int, int, float, float]:
            """Find the truly nearest flat-top edge-share lattice node.

            The placement runs in image-pixel space and the global lattice runs
            in lat/lon meters, so the two grids share *shape* but their origins
            are not aligned. A simple `round(mx/H_m)` snap can put a placement
            hex into a lattice column of the wrong parity (off by V/2), which
            shows up as diagonal stripes of mis-positioned points. Checking
            both candidate columns (and both candidate rows within each column)
            and keeping the truly closest of the four nodes prevents that.
            """
            col_lo = int(math.floor(mx / _H_m))
            best = None
            for col_candidate in (col_lo, col_lo + 1):
                y_off = _V_m * 0.5 if (col_candidate & 1) else 0.0
                row_lo = int(math.floor((my - y_off) / _V_m))
                for row_candidate in (row_lo, row_lo + 1):
                    cell_mx = col_candidate * _H_m
                    cell_my = row_candidate * _V_m + y_off
                    dy = my - cell_my
                    dx = mx - cell_mx
                    dist_sq = dx * dx + dy * dy
                    if best is None or dist_sq < best[2]:
                        best = (row_candidate, col_candidate, dist_sq, cell_my, cell_mx)
            assert best is not None
            return best[0], best[1], best[3], best[4]

        _projected_hexes: list[dict[str, Any]] = []
        for _hex in results["hexagons"]:
            _px, _py = _hex["center"]
            _lat, _lon = _px_to_gps(_px, _py)

            _my = _lat * _M_PER_DEG_LAT
            _mx = _lon * _m_per_deg_lon
            _row, _col, _snap_my, _snap_mx = _snap_to_nearest_lattice_node(_my, _mx)

            _hex["_raw_gps_lat"] = _lat
            _hex["_raw_gps_lon"] = _lon
            _hex["_snap_key"] = (_row, _col)
            _hex["_gps_lat"] = _snap_my / _M_PER_DEG_LAT
            _hex["_gps_lon"] = _snap_mx / _m_per_deg_lon
            _projected_hexes.append(_hex)

        _safe = []
        _forbidden_hexes = []
        _eroded_hexes = []
        _post_snap_danger_hexes = []
        _outside_hexes = []
        _image_edge_hexes = []
        _lattice_seen: set[tuple[int, int]] = set()
        _lattice_dedup = 0
        _snap_fallback_count = 0
        _lattice_refill_added = 0
        # The raster danger mask already includes the configured canopy buffer
        # shown in red. Do not add another hidden 2 m around that mask here;
        # this late snap guard only prevents the marker center from landing on
        # the displayed danger area after GPS/lattice projection.
        _danger_safety_margin_px = 0.5

        _danger_distance_map = None
        _core_radius_px = (float(hexagon_size) / max(float(_gsd), 1e-9)) * 0.2
        danger_mask = results.get("danger_mask")
        if isinstance(danger_mask, np.ndarray) and danger_mask.ndim == 2:
            try:
                _danger_distance_map = cv2.distanceTransform(
                    (danger_mask == 0).astype(np.uint8),
                    cv2.DIST_L2,
                    5,
                )
            except Exception:
                _danger_distance_map = None

        def _distance_from_mask(mask: Optional[np.ndarray]) -> Optional[np.ndarray]:
            if mask is None:
                return None
            try:
                return cv2.distanceTransform(
                    (mask == 0).astype(np.uint8),
                    cv2.DIST_L2,
                    5,
                )
            except Exception:
                return None

        _forbidden_point_distance_map = _distance_from_mask(_forbidden_point_mask)
        _eroded_point_distance_map = _distance_from_mask(_eroded_point_mask)

        def _candidate_display_center(lat: float, lon: float) -> Optional[tuple[float, float]]:
            try:
                px, py = _gps_to_px(lat, lon)
                if np.isfinite(px) and np.isfinite(py):
                    return float(px), float(py)
            except Exception:
                return None
            return None

        def _apply_display_center(hexagon: dict[str, Any], center: Optional[tuple[float, float]]) -> None:
            if center is None:
                return
            px, py = center
            buffer_radius_px = float(hexagon_size) / max(float(_gsd), 1e-9)
            core_radius_px = buffer_radius_px * 0.2
            hexagon["center"] = (px, py)
            hexagon["buffer"] = detector.create_hexagon(px, py, buffer_radius_px)
            hexagon["core"] = detector.create_hexagon(px, py, core_radius_px)

        def _edge_safety_state(
            center: Optional[tuple[float, float]],
        ) -> tuple[bool, bool]:
            return _image_edge_safety_state(
                center,
                _w,
                _h,
                _gsd,
                canopy_buffer,
                _core_radius_px,
                coordinate_homography_available,
            )

        def _annotate_edge_recheck(
            hexagon: dict[str, Any],
            center: Optional[tuple[float, float]],
        ) -> None:
            _, needs_full_recheck = _edge_safety_state(center)
            if needs_full_recheck:
                hexagon["_requires_full_edge_ortho_recheck"] = True
            else:
                hexagon.pop("_requires_full_edge_ortho_recheck", None)

        def _center_hits_zone_mask(
            mask: Optional[np.ndarray],
            center: Optional[tuple[float, float]],
        ) -> bool:
            if mask is None or center is None:
                return False
            try:
                px, py = center
                cx = int(round(float(px)))
                cy = int(round(float(py)))
            except Exception:
                return False
            if cx < 0 or cy < 0 or cy >= mask.shape[0] or cx >= mask.shape[1]:
                return False
            return bool(mask[cy, cx] > 0)

        def _center_is_clear_of_zone_mask(
            mask: Optional[np.ndarray],
            distance_map: Optional[np.ndarray],
            center: Optional[tuple[float, float]],
            clearance_px: float,
        ) -> bool:
            if mask is None or center is None:
                return True
            try:
                px, py = center
                cx = int(round(float(px)))
                cy = int(round(float(py)))
            except Exception:
                return True
            if cx < 0 or cy < 0 or cy >= mask.shape[0] or cx >= mask.shape[1]:
                return True
            if mask[cy, cx] > 0:
                return False
            if distance_map is None:
                return True
            return float(distance_map[cy, cx]) > max(0.5, float(clearance_px))

        def _candidate_rejection(
            lat: float,
            lon: float,
            display_center: Optional[tuple[float, float]],
            source_center: Optional[tuple[float, float]] = None,
        ) -> Optional[str]:
            try:
                if not is_inside_any_orthophoto(lat, lon):
                    return "outside"
            except Exception:
                pass
            gps_exclusion = _gps_exclusion_reason(
                lat,
                lon,
                forbidden_filter,
                eroded_filter,
                gis_coverage_geometry,
            )
            if gps_exclusion is not None:
                return gps_exclusion
            centers_to_check = (display_center, source_center)
            if any(_center_hits_zone_mask(_forbidden_point_mask, center) for center in centers_to_check):
                return "forbidden"
            if any(
                not _center_is_clear_of_zone_mask(
                    _forbidden_point_mask,
                    _forbidden_point_distance_map,
                    center,
                    _core_radius_px + 0.5,
                )
                for center in centers_to_check
            ):
                return "forbidden"
            if any(_center_hits_zone_mask(_eroded_point_mask, center) for center in centers_to_check):
                return "eroded"
            if any(
                not _center_is_clear_of_zone_mask(
                    _eroded_point_mask,
                    _eroded_point_distance_map,
                    center,
                    _core_radius_px + 0.5,
                )
                for center in centers_to_check
            ):
                return "eroded"
            if any(
                not _center_is_inside_map_component(
                    _reachable_map_component_mask,
                    center,
                )
                for center in centers_to_check
                if center is not None
            ):
                return "outside_map"
            if display_center is None:
                return "outside_image"
            px, py = display_center
            if px < 0 or py < 0 or px >= _w or py >= _h:
                return "outside_image"
            edge_rejected, _ = _edge_safety_state(display_center)
            if edge_rejected:
                return "image_edge"
            if _danger_distance_map is not None:
                cx = int(round(px))
                cy = int(round(py))
                if (
                    cx < 0
                    or cy < 0
                    or cy >= _danger_distance_map.shape[0]
                    or cx >= _danger_distance_map.shape[1]
                ):
                    return "outside_image"
                # The danger mask already includes the visible red canopy
                # buffer. Reject when the planting core would touch that
                # displayed danger area after GPS/lattice projection.
                if float(_danger_distance_map[cy, cx]) <= (_core_radius_px + _danger_safety_margin_px):
                    return "danger"
            return None

        def _refill_safe_lattice_gaps() -> int:
            """Add safe global-lattice nodes that earlier snap/filter passes opened up.

            The first detector pass packs points in drone-pixel space. Once those
            points are snapped to the shared GPS lattice and filtered against
            forbidden/danger zones, valid lattice cells can be left empty.
            This refill scans the same final lattice over the image footprint and
            accepts nodes whose projected display center is still core-safe.
            """
            plantable_zone = results.get("plantable_zone")
            if plantable_zone is None or getattr(plantable_zone, "is_empty", True):
                return 0

            lat_lon_samples: list[tuple[float, float]] = []
            for _px, _py in (
                (0.0, 0.0),
                (float(_w), 0.0),
                (float(_w), float(_h)),
                (0.0, float(_h)),
                (float(_w) * 0.5, float(_h) * 0.5),
            ):
                try:
                    _lat, _lon = _px_to_gps(_px, _py)
                    if np.isfinite(_lat) and np.isfinite(_lon):
                        lat_lon_samples.append((float(_lat), float(_lon)))
                except Exception:
                    continue

            for _hex in _projected_hexes:
                for _lat_key, _lon_key in (
                    ("_raw_gps_lat", "_raw_gps_lon"),
                    ("_gps_lat", "_gps_lon"),
                ):
                    _lat = _hex.get(_lat_key)
                    _lon = _hex.get(_lon_key)
                    if _lat is None or _lon is None:
                        continue
                    try:
                        if np.isfinite(float(_lat)) and np.isfinite(float(_lon)):
                            lat_lon_samples.append((float(_lat), float(_lon)))
                    except Exception:
                        continue

            if not lat_lon_samples:
                return 0

            _mx_values = [_lon * _m_per_deg_lon for _lat, _lon in lat_lon_samples]
            _my_values = [_lat * _M_PER_DEG_LAT for _lat, _lon in lat_lon_samples]
            _min_mx = min(_mx_values) - (_H_m * 2.0)
            _max_mx = max(_mx_values) + (_H_m * 2.0)
            _min_my = min(_my_values) - (_V_m * 2.0)
            _max_my = max(_my_values) + (_V_m * 2.0)

            _col_min = int(math.floor(_min_mx / _H_m)) - 1
            _col_max = int(math.ceil(_max_mx / _H_m)) + 1
            _approx_row_min = int(math.floor(_min_my / _V_m)) - 2
            _approx_row_max = int(math.ceil(_max_my / _V_m)) + 2
            _estimated_nodes = max(0, _col_max - _col_min + 1) * max(
                0,
                _approx_row_max - _approx_row_min + 1,
            )
            if _estimated_nodes > 50_000:
                warning_messages.append(
                    "Skipped planting-point gap refill because the projected image footprint was unexpectedly large."
                )
                return 0

            _buffer_radius_px = float(hexagon_size) / max(float(_gsd), 1e-9)
            _core_radius_px = _buffer_radius_px * 0.2
            _added = 0

            for _col in range(_col_min, _col_max + 1):
                _y_off = _V_m * 0.5 if (_col & 1) else 0.0
                _row_min = int(math.floor((_min_my - _y_off) / _V_m)) - 1
                _row_max = int(math.ceil((_max_my - _y_off) / _V_m)) + 1
                _cell_mx = _col * _H_m

                for _row in range(_row_min, _row_max + 1):
                    _key = (_row, _col)
                    if _key in _lattice_seen:
                        continue

                    _cell_my = (_row * _V_m) + _y_off
                    _lat = _cell_my / _M_PER_DEG_LAT
                    _lon = _cell_mx / _m_per_deg_lon
                    _display_center = _candidate_display_center(_lat, _lon)
                    if _candidate_rejection(_lat, _lon, _display_center) is not None:
                        continue
                    if _display_center is None:
                        continue

                    _px, _py = _display_center
                    _candidate = detector._evaluate_hex_candidate(
                        plantable_zone,
                        _px,
                        _py,
                        _buffer_radius_px,
                        _core_radius_px,
                        danger_distance_map=_danger_distance_map,
                        placement_stats=None,
                    )
                    if _candidate is None:
                        continue

                    _candidate["_gps_lat"] = _lat
                    _candidate["_gps_lon"] = _lon
                    _candidate["_raw_gps_lat"] = _lat
                    _candidate["_raw_gps_lon"] = _lon
                    _candidate["_snap_key"] = _key
                    _candidate["_gps_projection_source"] = "refill"
                    _annotate_edge_recheck(_candidate, _display_center)
                    _safe.append(_candidate)
                    _lattice_seen.add(_key)
                    _added += 1

            return _added

        def _final_lattice_refill_candidates(
            current_safe_hexagons: list[dict[str, Any]],
            *,
            apply_orthophoto_recheck: bool,
        ) -> list[dict[str, Any]]:
            """Return extra safe lattice nodes missing after late cleanup."""
            plantable_zone = results.get("plantable_zone")
            if plantable_zone is None or getattr(plantable_zone, "is_empty", True):
                return []

            def _normal_key(value: Any) -> Optional[tuple[int, int]]:
                if isinstance(value, (list, tuple)) and len(value) == 2:
                    try:
                        return int(value[0]), int(value[1])
                    except (TypeError, ValueError):
                        return None
                return None

            seen_keys = {
                key
                for key in (
                    _normal_key(hexagon.get("_snap_key"))
                    for hexagon in current_safe_hexagons
                    if str(hexagon.get("_gps_projection_source") or "") != "raw"
                )
                if key is not None
            }

            lat_lon_samples: list[tuple[float, float]] = []
            for _px, _py in (
                (0.0, 0.0),
                (float(_w), 0.0),
                (float(_w), float(_h)),
                (0.0, float(_h)),
                (float(_w) * 0.5, float(_h) * 0.5),
            ):
                try:
                    _lat, _lon = _px_to_gps(_px, _py)
                    if np.isfinite(_lat) and np.isfinite(_lon):
                        lat_lon_samples.append((float(_lat), float(_lon)))
                except Exception:
                    continue

            for _hex in list(_projected_hexes) + list(current_safe_hexagons):
                for _lat_key, _lon_key in (
                    ("_raw_gps_lat", "_raw_gps_lon"),
                    ("_gps_lat", "_gps_lon"),
                ):
                    _lat = _hex.get(_lat_key)
                    _lon = _hex.get(_lon_key)
                    if _lat is None or _lon is None:
                        continue
                    try:
                        if np.isfinite(float(_lat)) and np.isfinite(float(_lon)):
                            lat_lon_samples.append((float(_lat), float(_lon)))
                    except Exception:
                        continue

            if not lat_lon_samples:
                return []

            _mx_values = [_lon * _m_per_deg_lon for _lat, _lon in lat_lon_samples]
            _my_values = [_lat * _M_PER_DEG_LAT for _lat, _lon in lat_lon_samples]
            _min_mx = min(_mx_values) - (_H_m * 2.0)
            _max_mx = max(_mx_values) + (_H_m * 2.0)
            _min_my = min(_my_values) - (_V_m * 2.0)
            _max_my = max(_my_values) + (_V_m * 2.0)

            _col_min = int(math.floor(_min_mx / _H_m)) - 1
            _col_max = int(math.ceil(_max_mx / _H_m)) + 1
            _approx_row_min = int(math.floor(_min_my / _V_m)) - 2
            _approx_row_max = int(math.ceil(_max_my / _V_m)) + 2
            _estimated_nodes = max(0, _col_max - _col_min + 1) * max(
                0,
                _approx_row_max - _approx_row_min + 1,
            )
            if _estimated_nodes > 50_000:
                return []

            _buffer_radius_px = float(hexagon_size) / max(float(_gsd), 1e-9)
            _core_radius_px = _buffer_radius_px * 0.2
            _added: list[dict[str, Any]] = []
            _total_columns = max(1, _col_max - _col_min + 1)
            _column_interval = max(1, _total_columns // 20)

            for _column_index, _col in enumerate(
                range(_col_min, _col_max + 1),
                1,
            ):
                _y_off = _V_m * 0.5 if (_col & 1) else 0.0
                _row_min = int(math.floor((_min_my - _y_off) / _V_m)) - 1
                _row_max = int(math.ceil((_max_my - _y_off) / _V_m)) + 1
                _cell_mx = _col * _H_m

                for _row in range(_row_min, _row_max + 1):
                    _key = (_row, _col)
                    if _key in seen_keys:
                        continue

                    _cell_my = (_row * _V_m) + _y_off
                    _lat = _cell_my / _M_PER_DEG_LAT
                    _lon = _cell_mx / _m_per_deg_lon
                    _display_center = _candidate_display_center(_lat, _lon)
                    if _candidate_rejection(_lat, _lon, _display_center) is not None:
                        continue
                    if _display_center is None:
                        continue

                    _px, _py = _display_center
                    _candidate = detector._evaluate_hex_candidate(
                        plantable_zone,
                        _px,
                        _py,
                        _buffer_radius_px,
                        _core_radius_px,
                        danger_distance_map=_danger_distance_map,
                        placement_stats=None,
                    )
                    if _candidate is None:
                        continue

                    if apply_orthophoto_recheck:
                        _, _needs_full_edge_recheck = _edge_safety_state(_display_center)
                        _recheck_radius_m = (
                            float(canopy_buffer)
                            if _needs_full_edge_recheck
                            else min(
                                float(canopy_buffer),
                                _ORTHO_CANOPY_RECHECK_MAX_RADIUS_M,
                            )
                        )
                        canopy_recheck = _orthophoto_canopy_recheck(
                            _lat,
                            _lon,
                            safety_radius_m=_recheck_radius_m,
                        )
                        if (
                            (_needs_full_edge_recheck and canopy_recheck is None)
                            or (canopy_recheck and canopy_recheck.get("is_canopy"))
                        ):
                            continue

                    _candidate["_gps_lat"] = _lat
                    _candidate["_gps_lon"] = _lon
                    _candidate["_raw_gps_lat"] = _lat
                    _candidate["_raw_gps_lon"] = _lon
                    _candidate["_snap_key"] = _key
                    _candidate["_gps_projection_source"] = "final_refill"
                    _annotate_edge_recheck(_candidate, _display_center)
                    _added.append(_candidate)
                    seen_keys.add(_key)

                if (
                    _column_index == 1
                    or _column_index == _total_columns
                    or _column_index % _column_interval == 0
                ):
                    _emit(
                        progress_cb,
                        f"Refilling safe lattice {_column_index}/{_total_columns}",
                        88 + int(round(_column_index / _total_columns)),
                        current_column=_column_index,
                        total_columns=_total_columns,
                    )

            return _added

        final_lattice_refill_fn = _final_lattice_refill_candidates

        for _hex in _projected_hexes:
            _snap_key = _hex.get("_snap_key")
            _source_center = _hex.get("center")
            _candidate_options = [
                ("snapped", _hex["_gps_lat"], _hex["_gps_lon"], _snap_key),
            ]
            if canonical_species is None:
                _candidate_options.append(("raw", _hex["_raw_gps_lat"], _hex["_raw_gps_lon"], None))
            _first_rejection = None
            _chosen = None

            for _source, _lat, _lon, _key in _candidate_options:
                if _key is not None and _key in _lattice_seen:
                    if _first_rejection is None:
                        _first_rejection = "duplicate"
                    continue

                _display_center = _candidate_display_center(_lat, _lon)
                _rejection = _candidate_rejection(_lat, _lon, _display_center, _source_center)
                if _rejection is not None:
                    if _first_rejection is None:
                        _first_rejection = _rejection
                    continue

                _chosen = (_source, _lat, _lon, _key, _display_center)
                break

            if _chosen is not None:
                _source, _lat, _lon, _key, _display_center = _chosen
                if _key is not None:
                    _lattice_seen.add(_key)
                elif _snap_key is not None and _snap_key in _lattice_seen:
                    _lattice_dedup += 1
                if _source == "raw":
                    _snap_fallback_count += 1
                _hex["_gps_lat"] = _lat
                _hex["_gps_lon"] = _lon
                _hex["_gps_projection_source"] = _source
                _apply_display_center(_hex, _display_center)
                _annotate_edge_recheck(_hex, _display_center)
                _safe.append(_hex)
                continue

            _reason = _first_rejection or "forbidden"
            if _reason == "eroded":
                _eroded_hexes.append(_hex)
            elif _reason == "image_edge":
                _image_edge_hexes.append(_hex)
            elif _reason in {"outside", "outside_image", "outside_map"}:
                _outside_hexes.append(_hex)
            elif _reason == "danger":
                _post_snap_danger_hexes.append(_hex)
            elif _reason == "duplicate":
                _lattice_dedup += 1
            else:
                _forbidden_hexes.append(_hex)

        _lattice_refill_added = _refill_safe_lattice_gaps()
        if _lattice_refill_added:
            info_messages.append(
                f"{_lattice_refill_added} additional planting points were added by refilling safe gaps on the shared GPS lattice."
            )

        if _lattice_dedup:
            info_messages.append(
                f"{_lattice_dedup} duplicate planting points were merged onto the shared global hex lattice."
            )
        if _snap_fallback_count:
            info_messages.append(
                f"{_snap_fallback_count} planting points kept their direct image projection because lattice snapping would have moved them into an unsafe or unmapped area."
            )
        if canonical_species:
            info_messages.append(
                f"{canonical_species} planting points were kept on the fixed {SPECIES_SPACING_M[canonical_species]:.1f} m species lattice; off-lattice fallback points were not used."
            )

        # Do not hide DB-nearby points during processing preview. Re-running
        # the same image replaces its old analysis at save time, and
        # save_analysis still deduplicates against other analyses before
        # persistence. Keeping the preview complete makes the visible overlay
        # match the actual safe-space placement instead of inheriting stale DB
        # gaps from an earlier run.
        _duplicate_hexes: list = []

        results["hexagons"] = _safe
        results["hexagon_count"] = len(_safe)
        results["_forbidden_filtered"] = len(_forbidden_hexes)
        results["_eroded_filtered"] = len(_eroded_hexes)
        results["_post_snap_danger_filtered"] = len(_post_snap_danger_hexes)
        results["_duplicate_filtered"] = len(_duplicate_hexes)
        results["_preclip_outside_filtered"] = len(_outside_hexes)
        results["_image_edge_filtered"] = len(_image_edge_hexes)
        results["_lattice_refill_added"] = _lattice_refill_added
        results["_forbidden_hexagons"] = _forbidden_hexes
        results["_eroded_hexagons"] = _eroded_hexes
        results["_post_snap_danger_hexagons"] = _post_snap_danger_hexes
        results["_image_edge_hexagons"] = _image_edge_hexes
        results["_duplicate_hexagons"] = _duplicate_hexes
        results["_preclip_outside_hexagons"] = _outside_hexes

    # MIGRATED FROM app.py: analyze_image lines 5600-5937
    _emit(progress_cb, "Projecting planting points to GPS", 85)
    map_image_gps = image_gps
    map_center_lat = image_center_lat
    map_center_lon = image_center_lon
    width, height = results["image_size"]
    safe_hexagons = [copy.deepcopy(hexagon) for hexagon in results.get("hexagons", [])]
    forbidden_hexagons = [copy.deepcopy(hexagon) for hexagon in results.get("_forbidden_hexagons", [])]
    eroded_hexagons = [copy.deepcopy(hexagon) for hexagon in results.get("_eroded_hexagons", [])]
    post_snap_danger_hexagons = [
        copy.deepcopy(hexagon) for hexagon in results.get("_post_snap_danger_hexagons", [])
    ]
    image_edge_hexagons = [
        copy.deepcopy(hexagon) for hexagon in results.get("_image_edge_hexagons", [])
    ]
    orthophoto_canopy_hexagons: list[dict[str, Any]] = []
    spacing_filtered_hexagons: list[dict[str, Any]] = []
    forbidden_filtered_count = int(results.get("_forbidden_filtered", 0) or 0)
    eroded_filtered_count = int(results.get("_eroded_filtered", 0) or 0)
    post_snap_danger_filtered_count = int(results.get("_post_snap_danger_filtered", 0) or 0)
    image_edge_filtered_count = int(results.get("_image_edge_filtered", 0) or 0)
    orthophoto_canopy_filtered_count = 0
    spacing_filtered_count = 0
    final_lattice_refill_added = 0
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
            match_warning = _match_quality_warning(match_result)
            if match_warning and match_warning not in warning_messages:
                warning_messages.append(match_warning)

        if match_result.get("success"):
            detected_heading = float(match_result.get("heading") or camera_heading)

        if coordinate_homography_available and coordinate_homography is not None:
            H_matrix = coordinate_homography

            def pixel_to_latlon(px, py):
                return drone_pixel_to_gps_via_homography(px, py, H_matrix)

        else:
            if not match_result.get("success"):
                warning_messages.append(
                    f"Auto-alignment not available ({match_result.get('error')}). "
                    f"Using heading-based fallback ({camera_heading:.1f}°)."
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

        # A raw visual H and a GPS-anchored rebuilt H both map final exported
        # coordinates into the orthophoto. Recheck either one so no final map
        # point lands on mapped canopy. Only the no-H EXIF-heading fallback is
        # too weak to support this spatial safety test.
        orthophoto_recheck_enabled = bool(coordinate_homography_available)

        _emit(
            progress_cb,
            f"Projecting {len(safe_hexagons)} planting points to GPS",
            85,
            total_points=len(safe_hexagons),
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

        for hexagon in post_snap_danger_hexagons:
            if "_gps_lat" not in hexagon:
                px, py = hexagon["center"]
                lat, lon = pixel_to_latlon(px, py)
                hexagon["_gps_lat"] = lat
                hexagon["_gps_lon"] = lon

        before_clip = len(safe_hexagons)
        if orthophoto_recheck_enabled:
            _emit(
                progress_cb,
                f"Checking orthophoto coverage for {len(safe_hexagons)} planting points",
                86,
                total_points=len(safe_hexagons),
                )
            coverage_checked = []
            coverage_interval = max(1, len(safe_hexagons) // 20)
            for index, hexagon in enumerate(safe_hexagons, 1):
                if is_inside_any_orthophoto(hexagon["_gps_lat"], hexagon["_gps_lon"]):
                    hexagon["_orthophoto_bounds_checked"] = True
                    coverage_checked.append(hexagon)
                if index == 1 or index == len(safe_hexagons) or index % coverage_interval == 0:
                    _emit(
                        progress_cb,
                        f"Checking orthophoto coverage {index}/{len(safe_hexagons)}",
                        86 + int(round(index / max(1, len(safe_hexagons)))),
                        current_point=index,
                        total_points=len(safe_hexagons),
                    )
            safe_hexagons = coverage_checked
        else:
            # Heading fallback is only an approximate image-to-GPS projection.
            # Do not run a second full orthophoto registry/raster lookup for every
            # point: it cannot be spatially trusted and can force a huge GeoTIFF
            # load after the UI has already reached 85%.
            _emit(
                progress_cb,
                "Using heading fallback; skipping unreliable orthophoto bounds and canopy checks",
                86,
                total_points=len(safe_hexagons),
            )
        clipped_outside_orthophoto = (
            before_clip
            - len(safe_hexagons)
            + int(results.get("_preclip_outside_filtered", 0) or 0)
        )
        if clipped_outside_orthophoto > 0:
            info_messages.append(
                f"{clipped_outside_orthophoto} planting points were removed because they fall outside the supported map area."
            )

        if orthophoto_recheck_enabled and safe_hexagons:
            recheck_interval = max(1, len(safe_hexagons) // 20)
            rechecked_hexagons = []
            for index, hexagon in enumerate(safe_hexagons, 1):
                needs_full_edge_recheck = bool(
                    hexagon.get("_requires_full_edge_ortho_recheck")
                )
                recheck_radius_m = (
                    float(canopy_buffer)
                    if needs_full_edge_recheck
                    else min(
                        float(canopy_buffer),
                        _ORTHO_CANOPY_RECHECK_MAX_RADIUS_M,
                    )
                )
                canopy_recheck = _orthophoto_canopy_recheck(
                    float(hexagon["_gps_lat"]),
                    float(hexagon["_gps_lon"]),
                    safety_radius_m=recheck_radius_m,
                )
                if canopy_recheck and canopy_recheck.get("is_canopy"):
                    hexagon["_orthophoto_canopy_recheck"] = canopy_recheck
                    orthophoto_canopy_hexagons.append(hexagon)
                elif needs_full_edge_recheck and canopy_recheck is None:
                    hexagon["_image_edge_recheck_unavailable"] = True
                    image_edge_hexagons.append(hexagon)
                else:
                    rechecked_hexagons.append(hexagon)
                if index == 1 or index == len(safe_hexagons) or index % recheck_interval == 0:
                    recheck_pct = 86 + int(round(2 * index / max(1, len(safe_hexagons))))
                    _emit(
                        progress_cb,
                        f"Rechecking orthophoto canopy {index}/{len(safe_hexagons)}",
                        min(88, recheck_pct),
                        current_point=index,
                        total_points=len(safe_hexagons),
                    )
            safe_hexagons = rechecked_hexagons
            orthophoto_canopy_filtered_count = len(orthophoto_canopy_hexagons)
            image_edge_filtered_count = len(image_edge_hexagons)
            if orthophoto_canopy_filtered_count > 0:
                info_messages.append(
                    f"{orthophoto_canopy_filtered_count} planting points were removed because they landed on or too close to canopy in the orthophoto recheck."
                )

    if post_snap_danger_filtered_count > 0:
        info_messages.append(
            f"{post_snap_danger_filtered_count} planting points were removed because final GPS snapping put them inside the displayed danger buffer."
        )
    if image_edge_filtered_count > 0:
        info_messages.append(
            f"{image_edge_filtered_count} planting points were removed to preserve the 1 m uploaded-image edge clearance or because the required wider orthophoto check was unavailable."
        )

    final_min_spacing_m = max(0.0, float(hexagon_size) * _FINAL_POINT_SPACING_FACTOR)
    if safe_hexagons:
        safe_hexagons, spacing_filtered_hexagons = _thin_hexagons_by_gps_spacing(
            safe_hexagons,
            final_min_spacing_m,
        )
        spacing_filtered_count = len(spacing_filtered_hexagons)
        if spacing_filtered_count > 0:
            info_messages.append(
                f"{spacing_filtered_count} planting points were removed as near-duplicate GPS fallbacks below {final_min_spacing_m * _FINAL_DUPLICATE_SPACING_RATIO:.2f} m."
            )

    if (
        map_image_gps is not None
        and final_lattice_refill_fn is not None
        and coordinate_homography_trusted
    ):
        final_refill_hexagons = final_lattice_refill_fn(
            safe_hexagons,
            apply_orthophoto_recheck=True,
        )
        if final_refill_hexagons:
            safe_hexagons.extend(final_refill_hexagons)
            final_lattice_refill_added = len(final_refill_hexagons)
            safe_hexagons, second_spacing_filtered = _thin_hexagons_by_gps_spacing(
                safe_hexagons,
                final_min_spacing_m,
            )
            if second_spacing_filtered:
                spacing_filtered_hexagons.extend(second_spacing_filtered)
                spacing_filtered_count = len(spacing_filtered_hexagons)
            info_messages.append(
                f"{final_lattice_refill_added} final safe planting gaps were filled on the 1 m shared lattice."
            )

    if canonical_species and safe_hexagons:
        safe_hexagons, saved_species_filtered = _thin_hexagons_against_saved_species_points(
            safe_hexagons,
            canonical_species,
            SPECIES_SPACING_M[canonical_species],
        )
        if saved_species_filtered:
            spacing_filtered_hexagons.extend(saved_species_filtered)
            spacing_filtered_count = len(spacing_filtered_hexagons)
            info_messages.append(
                f"{len(saved_species_filtered)} planting points were removed because they were within 2 m of saved points from another species."
            )

    # Final fail-safe: every downstream output (preview, coordinates, export,
    # and save cache) is built from this exact list. Recheck authoritative GPS
    # coordinates here so a future refill or fallback path cannot bypass the
    # earlier candidate gate.
    final_zone_safe: list[dict[str, Any]] = []
    final_forbidden_conflicts: list[dict[str, Any]] = []
    final_eroded_conflicts: list[dict[str, Any]] = []
    final_outside_map_conflicts: list[dict[str, Any]] = []
    for hexagon in safe_hexagons:
        latitude = hexagon.get("_gps_lat")
        longitude = hexagon.get("_gps_lon")
        if latitude is None or longitude is None:
            continue
        exclusion_reason = _gps_exclusion_reason(
            float(latitude),
            float(longitude),
            forbidden_filter,
            eroded_filter,
            gis_coverage_geometry,
        )
        if exclusion_reason == "forbidden":
            final_forbidden_conflicts.append(hexagon)
        elif exclusion_reason == "eroded":
            final_eroded_conflicts.append(hexagon)
        elif exclusion_reason == "outside_map" or not _center_is_inside_map_component(
            _reachable_map_component_mask,
            hexagon.get("center"),
        ):
            final_outside_map_conflicts.append(hexagon)
        else:
            final_zone_safe.append(hexagon)
    safe_hexagons = final_zone_safe
    if final_forbidden_conflicts:
        forbidden_hexagons.extend(final_forbidden_conflicts)
        forbidden_filtered_count += len(final_forbidden_conflicts)
    if final_eroded_conflicts:
        eroded_hexagons.extend(final_eroded_conflicts)
        eroded_filtered_count += len(final_eroded_conflicts)
    if final_outside_map_conflicts:
        clipped_outside_orthophoto += len(final_outside_map_conflicts)
        info_messages.append(
            f"{len(final_outside_map_conflicts)} additional planting point(s) were removed by the final map-boundary safety check."
        )

    # The overview polygon used for fast preflight is sub-metre resolution.
    # Check source pixels here as well, so tiny transparent holes and the
    # exact edge cannot reach previews, exports, or the save cache.
    if safe_hexagons:
        exact_flags = point_visibility_flags(
            ortho_matcher._ensure_active_ortho(),
            gis_coverage_geometry,
            [(item.get("_gps_lat"), item.get("_gps_lon")) for item in safe_hexagons],
        )
        exact_outside_count = len(safe_hexagons) - sum(exact_flags)
        safe_hexagons = [item for item, visible in zip(safe_hexagons, exact_flags) if visible]
        if exact_outside_count:
            clipped_outside_orthophoto += exact_outside_count
            info_messages.append(
                f"{exact_outside_count} planting point(s) were removed at the exact visible-map edge."
            )

    # Aligned points reaching this stage have already passed the orthophoto
    # bounds check above. Heading-fallback points intentionally skip that
    # unreliable raster lookup, and final lattice-refill points are checked in
    # their own candidate-rejection path. Repeating the registry/image lookup
    # here only adds latency and previously made the UI appear stuck at 85%.

    # Keep the final image-space center that was evaluated by the authoritative
    # coordinate path. Reprojecting these GPS points through a second, display-
    # only homography previously moved valid points onto structures or outside
    # the image. The same coordinate masks are used for this final core-only
    # invariant check, so map, exports, and preview retain identical membership.
    _emit(progress_cb, "Rendering visualization overlay", 90)
    (
        safe_hexagons,
        visualization_hexagons,
        visual_forbidden_conflicts,
        visual_eroded_conflicts,
    ) = _reconcile_authoritative_hexagons_with_visual_zones(
        detector,
        safe_hexagons,
        None,
        float(hexagon_size),
        float(results["gsd_m_per_pixel"]),
        _forbidden_point_mask,
        _eroded_point_mask,
    )

    if visual_forbidden_conflicts:
        forbidden_hexagons.extend(visual_forbidden_conflicts)
        forbidden_filtered_count += len(visual_forbidden_conflicts)
    if visual_eroded_conflicts:
        eroded_hexagons.extend(visual_eroded_conflicts)
        eroded_filtered_count += len(visual_eroded_conflicts)

    # Eroded conflicts are exclusions, so none may remain to be marked as
    # visible-but-unavailable in the final authoritative set.
    eroded_unavailable_count = 0

    visual_conflict_filtered_count = (
        len(visual_forbidden_conflicts) + len(visual_eroded_conflicts)
    )
    if visual_conflict_filtered_count:
        info_messages.append(
            f"{visual_conflict_filtered_count} planting point(s) were removed from "
            "both map and image outputs because the authoritative hexagon "
            "core intersected a forbidden or eroded zone."
        )

    results["_lattice_refill_added"] = (
        int(results.get("_lattice_refill_added", 0) or 0) + final_lattice_refill_added
    )
    results["_final_lattice_refill_added"] = final_lattice_refill_added
    results["hexagons"] = safe_hexagons
    results["hexagon_count"] = len(safe_hexagons)
    results["_forbidden_filtered"] = forbidden_filtered_count
    results["_eroded_filtered"] = eroded_filtered_count
    results["_spacing_filtered"] = spacing_filtered_count
    results["_spacing_filtered_hexagons"] = spacing_filtered_hexagons
    # Surface the chosen species and the resulting field spacing so
    # save_analysis() can persist them on the analysis row — the map can
    # then color points by species without re-deriving from hexagon_size.
    results["species"] = canonical_species
    results["planting_distance_m"] = (
        SPECIES_SPACING_M[canonical_species] if canonical_species else None
    )

    # The final authoritative set and its one-transform visualization now have
    # identical membership, so no display-only suppression or relocation is
    # needed during rendering.
    results["_overlay_zone_suppressed"] = 0
    results["_overlay_rendered_count"] = len(visualization_hexagons)
    results["_overlay_projection_fallback_count"] = 0
    results["_overlay_visual_conflict_filtered_count"] = visual_conflict_filtered_count
    visualization_results = dict(results)
    visualization_results["hexagons"] = visualization_hexagons
    visualization_results["hexagon_count"] = len(visualization_hexagons)
    visualization_image = detector.visualize_results(visualization_results)

    overlay_h = _overlay_homography_for_match(match_result)
    if analysis_overlay is None and map_image_gps is not None and overlay_h is not None:
        analysis_overlay = _build_georeferenced_overlay(visualization_image, overlay_h)

    _emit(progress_cb, "Building coordinate table and exports", 92)
    results["_eroded_unavailable"] = eroded_unavailable_count
    available_safe_hexagon_count = len(safe_hexagons)

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
                "eroded_unavailable": False,
                "availability_status": "available",
                "status_label": "Planned",
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
    post_snap_danger_filtered_geojson = _filtered_hexagons_to_geojson(
        post_snap_danger_hexagons,
        "Final snapped point inside the displayed danger buffer",
        "Filtered Final Danger Buffer Points",
    )
    image_edge_filtered_geojson = _filtered_hexagons_to_geojson(
        image_edge_hexagons,
        "Too close to uploaded image edge; canopy outside photo cannot be checked",
        "Filtered Image Edge Points",
    )
    orthophoto_canopy_filtered_geojson = _filtered_hexagons_to_geojson(
        orthophoto_canopy_hexagons,
        "Too close to canopy in orthophoto recheck",
        "Filtered Orthophoto Canopy Points",
    )
    spacing_filtered_geojson = _filtered_hexagons_to_geojson(
        spacing_filtered_hexagons,
        "Too close to another final planting point",
        "Filtered Too-Close Points",
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
    analysis_footprint = analysis_overlay.get("footprint_geojson") if analysis_overlay else None
    if analysis_footprint:
        footprint_quality = (
            "metric_projected_image_corners"
            if match_result and match_result.get("projection_rebuilt")
            else "matched_image_corners"
        )
    else:
        analysis_footprint = _estimated_analysis_footprint(
            map_center_lat,
            map_center_lon,
            results.get("coverage_m"),
            detected_heading,
        )
        footprint_quality = "estimated_oriented_rectangle" if analysis_footprint else "missing"
    results["footprint_geojson"] = analysis_footprint
    results["footprint_quality"] = footprint_quality
    visible_footprint = None
    if analysis_footprint:
        clipped_footprint = gis_coverage_geometry.intersection(shape(analysis_footprint))
        if clipped_footprint.geom_type in {"Polygon", "MultiPolygon"} and not clipped_footprint.is_empty:
            visible_footprint = mapping(clipped_footprint)

    json_results = {
        "canopy_count": results["canopy_count"],
        "danger_area_m2": results["danger_area_m2"],
        "plantable_area_m2": results["plantable_area_m2"],
        "hexagon_count": results["hexagon_count"],
        "safe_hexagon_count": available_safe_hexagon_count,
        "eroded_unavailable_count": eroded_unavailable_count,
        "gsd": results["gsd_m_per_pixel"],
        "coverage": results["coverage_m"],
    }

    can_save = bool(map_image_gps is not None and map_center_lat is not None and map_center_lon is not None)
    ai_meta = results.get("ai_metadata") or {}
    canopy_area_m2 = _compute_canopy_area_m2(results)
    canopy_coverage_pct = _compute_canopy_coverage_pct(results)
    processing_time_sec = round(time.time() - workflow_start_time, 2)
    results["canopy_area_m2"] = canopy_area_m2
    results["canopy_coverage_pct"] = canopy_coverage_pct
    results["ai_confidence"] = float(ai_confidence)

    _emit(progress_cb, "Encoding preview images", 97)
    # The cache key intentionally contains the source filename and serialized
    # settings, but those characters are not valid in every S3-compatible
    # object-key implementation. A digest gives Storage a short, URL-safe path
    # while preserving the original analysis key used by the save workflow.
    preview_assets = upload_analysis_data_urls(
        _encode_image_data_url(results["image"], ".jpg"),
        _encode_image_data_url(visualization_image, ".jpg"),
        prefix=_analysis_storage_prefix(analysis_key),
    )
    preview_asset_map = {asset.kind: asset for asset in preview_assets}

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
            "species": canonical_species,
            "planting_distance_m": (
                SPECIES_SPACING_M[canonical_species] if canonical_species else None
            ),
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
            "location_preflight": {
                key: value
                for key, value in location_preflight.items()
                if key != "map"
            },
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
            "canopy_area_m2": canopy_area_m2,
            "canopy_coverage_pct": canopy_coverage_pct,
            "danger_area_m2": results["danger_area_m2"],
            "danger_percentage": results["danger_percentage"],
            "plantable_area_m2": results["plantable_area_m2"],
            "plantable_percentage": results["plantable_percentage"],
            "hexagon_count": results["hexagon_count"],
            # Every accepted point is available for planting at analysis time;
            # danger, forbidden, and eroded conflicts were removed earlier.
            "safe_hexagon_count": available_safe_hexagon_count,
            "forbidden_filtered_count": forbidden_filtered_count,
            "eroded_filtered_count": eroded_filtered_count,
            "eroded_unavailable_count": eroded_unavailable_count,
            "post_snap_danger_filtered_count": post_snap_danger_filtered_count,
            "image_edge_filtered_count": image_edge_filtered_count,
            "orthophoto_canopy_filtered_count": orthophoto_canopy_filtered_count,
            "spacing_filtered_count": spacing_filtered_count,
            "duplicate_filtered_count": int(results.get("_duplicate_filtered", 0) or 0),
            "overlay_zone_suppressed_count": int(results.get("_overlay_zone_suppressed", 0) or 0),
            "overlay_rendered_count": int(
                results.get("_overlay_rendered_count", len(safe_hexagons)) or 0
            ),
            "overlay_projection_fallback_count": int(
                results.get("_overlay_projection_fallback_count", 0) or 0
            ),
            "overlay_visual_conflict_filtered_count": int(
                results.get("_overlay_visual_conflict_filtered_count", 0) or 0
            ),
            "outside_map_filtered_count": clipped_outside_orthophoto,
            # Retained for compatibility with existing saved analysis payloads.
            "clipped_outside_orthophoto": clipped_outside_orthophoto,
            "lattice_refill_added": int(results.get("_lattice_refill_added", 0) or 0),
            "final_lattice_refill_added": int(results.get("_final_lattice_refill_added", 0) or 0),
            "forbidden_canopy_exclusion_enabled": bool(results.get("_forbidden_canopy_exclusion_enabled", False)),
            "forbidden_canopy_removed_pixels": int(results.get("_forbidden_canopy_removed_pixels", 0) or 0),
            "forbidden_canopy_removed_count": int(results.get("_forbidden_canopy_removed_count", 0) or 0),
            "gsd_m_per_pixel": results["gsd_m_per_pixel"],
            "gsd_specs": results.get("gsd_specs"),
            "coverage_m": results["coverage_m"],
            "total_area_m2": results["total_area_m2"],
            "altitude_m": results["altitude_m"],
            "tile_count": int(ai_meta.get("num_tiles_processed", 0) or 0),
            "processing_time_sec": processing_time_sec,
            "model_name": ai_meta.get("model_name"),
            "ai_confidence_threshold": float(ai_confidence),
            "ai_avg_confidence": _safe_float(ai_meta.get("avg_confidence")),
            "ai_instance_count": int(ai_meta.get("instance_tree_count", results["canopy_count"]) or 0),
            "ai_raw_detections": int(ai_meta.get("raw_detections", 0) or 0),
            "ai_model_candidate_detections": int(ai_meta.get("model_candidate_detections", 0) or 0),
            "ai_below_confidence_detections": int(ai_meta.get("below_confidence_detections", 0) or 0),
            "ai_rescued_low_confidence_detections": int(ai_meta.get("rescued_low_confidence_detections", 0) or 0),
            "ai_rescued_shadow_crown_detections": int(ai_meta.get("rescued_shadow_crown_detections", 0) or 0),
            "ai_rejected_non_green_detections": int(ai_meta.get("rejected_non_green_detections", 0) or 0),
            "ai_rejected_low_saturation_detections": int(ai_meta.get("rejected_low_saturation_detections", 0) or 0),
            "ai_rejected_low_saturation_components": int(ai_meta.get("rejected_low_saturation_components", 0) or 0),
            "ai_seedling_supplement_count": int(ai_meta.get("seedling_supplement_count", 0) or 0),
            "ai_seedling_candidate_count": int(ai_meta.get("seedling_candidate_count", 0) or 0),
            "ai_seedling_micro_candidate_count": int(
                ai_meta.get("seedling_micro_candidate_count", 0) or 0
            ),
            "ai_seedling_micro_supplement_count": int(
                ai_meta.get("seedling_micro_supplement_count", 0) or 0
            ),
            "ai_seedling_micro_rejected_shape": int(
                ai_meta.get("seedling_micro_rejected_shape", 0) or 0
            ),
            "ai_seedling_micro_rejected_color": int(
                ai_meta.get("seedling_micro_rejected_color", 0) or 0
            ),
            "ai_seedling_micro_single_candidate_count": int(
                ai_meta.get("seedling_micro_single_candidate_count", 0) or 0
            ),
            "ai_seedling_micro_single_rejected_color": int(
                ai_meta.get("seedling_micro_single_rejected_color", 0) or 0
            ),
            "ai_seedling_micro_single_rejected_support": int(
                ai_meta.get("seedling_micro_single_rejected_support", 0) or 0
            ),
            "ai_seedling_micro_min_area_m2": _safe_float(
                ai_meta.get("seedling_micro_min_area_m2")
            ),
            "ai_seedling_contextual_candidate_count": int(
                ai_meta.get("seedling_contextual_candidate_count", 0) or 0
            ),
            "ai_seedling_contextual_supplement_count": int(
                ai_meta.get("seedling_contextual_supplement_count", 0) or 0
            ),
            "ai_seedling_contextual_rejected_neighbor": int(
                ai_meta.get("seedling_contextual_rejected_neighbor", 0) or 0
            ),
            "ai_seedling_fragment_duplicates_removed": int(
                ai_meta.get("seedling_fragment_duplicates_removed", 0) or 0
            ),
            "ai_seedling_cluster_rescue_count": int(
                ai_meta.get("seedling_cluster_rescue_count", 0) or 0
            ),
            "ai_seedling_cluster_rescue_rejected_existing": int(
                ai_meta.get("seedling_cluster_rescue_rejected_existing", 0) or 0
            ),
            "ai_seedling_cluster_rescue_rejected_shape": int(
                ai_meta.get("seedling_cluster_rescue_rejected_shape", 0) or 0
            ),
            "ai_seedling_cluster_rescue_rejected_color": int(
                ai_meta.get("seedling_cluster_rescue_rejected_color", 0) or 0
            ),
            "ai_seedling_linked_rejected_shape": int(
                ai_meta.get("seedling_linked_rejected_shape", 0) or 0
            ),
            "ai_seedling_ground_artifact_rejected_count": int(
                ai_meta.get("seedling_ground_artifact_rejected_count", 0) or 0
            ),
            "ai_seedling_water_context_candidate_count": int(
                ai_meta.get("seedling_water_context_candidate_count", 0) or 0
            ),
            "ai_seedling_water_context_verified_count": int(
                ai_meta.get("seedling_water_context_verified_count", 0) or 0
            ),
            "ai_seedling_water_context_rejected_count": int(
                ai_meta.get("seedling_water_context_rejected_count", 0) or 0
            ),
            "ai_seedling_neutral_water_context_candidate_count": int(
                ai_meta.get("seedling_neutral_water_context_candidate_count", 0) or 0
            ),
            "ai_seedling_neutral_water_context_verified_count": int(
                ai_meta.get("seedling_neutral_water_context_verified_count", 0) or 0
            ),
            "ai_seedling_neutral_water_context_rejected_count": int(
                ai_meta.get("seedling_neutral_water_context_rejected_count", 0) or 0
            ),
            "ai_seedling_buffer_m": _safe_float(
                ai_meta.get("seedling_buffer_m", results.get("seedling_buffer_m", 0.0))
            ),
            "ai_seedling_detection_mode": ai_meta.get("seedling_detection_mode"),
            "ai_seedling_classifier_available": bool(
                ai_meta.get("seedling_classifier_available", False)
            ),
            "ai_seedling_classifier_threshold": _safe_float(
                ai_meta.get("seedling_classifier_threshold")
            ),
            "ai_seedling_rejected_classifier": int(
                ai_meta.get("seedling_rejected_classifier", 0) or 0
            ),
            "ai_seedling_rejected_neighbor": int(
                ai_meta.get("seedling_rejected_neighbor", 0) or 0
            ),
            "ai_seedling_rejected_duplicate": int(
                ai_meta.get("seedling_rejected_duplicate", 0) or 0
            ),
            "ai_seedling_fallback_legacy_count": int(
                ai_meta.get("seedling_fallback_legacy_count", 0) or 0
            ),
            "ai_rejected_non_canopy_class_detections": int(ai_meta.get("rejected_non_canopy_class_detections", 0) or 0),
            "ai_class_candidate_counts": ai_meta.get("class_candidate_counts"),
            "ai_class_kept_counts": ai_meta.get("class_kept_counts"),
            "ai_canopy_class_ids": ai_meta.get("canopy_class_ids"),
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
            "original_image_url": signed_download_url(
                preview_asset_map["original"].object_key
            ),
            "visualization_image_url": signed_download_url(
                preview_asset_map["visualization"].object_key
            ),
            "original_preview_url": signed_download_url(
                preview_asset_map.get("original_preview", preview_asset_map["original"]).object_key
            ),
            "visualization_preview_url": signed_download_url(
                preview_asset_map.get(
                    "visualization_preview", preview_asset_map["visualization"]
                ).object_key
            ),
            "url_expires_in_seconds": get_settings().s3_presigned_url_ttl_seconds,
        },
        "map": {
            "available": bool(map_image_gps is not None),
            "location_status": location_preflight.get("status"),
            "match": _sanitize_match_result(match_result),
            "image_center_feature": image_center_feature,
            "analysis_footprint": visible_footprint,
            "analysis_footprint_quality": results["footprint_quality"],
            "analysis_overlay": analysis_overlay,
            "safe_points_geojson": _visible_map_point_features(safe_points_geojson, gis_coverage_geometry),
            "forbidden_filtered_geojson": _visible_map_point_features(forbidden_filtered_geojson, gis_coverage_geometry),
            "eroded_filtered_geojson": _visible_map_point_features(eroded_filtered_geojson, gis_coverage_geometry),
            "post_snap_danger_filtered_geojson": _visible_map_point_features(post_snap_danger_filtered_geojson, gis_coverage_geometry),
            "image_edge_filtered_geojson": _visible_map_point_features(image_edge_filtered_geojson, gis_coverage_geometry),
            "orthophoto_canopy_filtered_geojson": _visible_map_point_features(orthophoto_canopy_filtered_geojson, gis_coverage_geometry),
            "spacing_filtered_geojson": _visible_map_point_features(spacing_filtered_geojson, gis_coverage_geometry),
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
    results["_analysis_detail_json"] = json.dumps(
        {
            "detection_mode": response_payload["detection_mode"],
            "parameters": response_payload["parameters"],
            "metadata": response_payload["metadata"],
            "metrics": response_payload["metrics"],
            "map_match": response_payload["map"].get("match"),
            "analysis_footprint": analysis_footprint,
            "analysis_footprint_quality": results["footprint_quality"],
        },
        ensure_ascii=False,
    )

    _set_processing_cache(
        analysis_key,
        {
            "analysis_key": analysis_key,
            "image_name": uploaded_name,
            "results": results,
            "safe_hexagons": safe_hexagons,
            "center_lat": map_center_lat,
            "center_lon": map_center_lon,
            "stored_assets": preview_assets,
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


@router.post("/preflight")
async def preflight_image_location(
    image: UploadFile = File(...),
    altitude: float = Form(6.0),
    drone_model: str = Form("GENERIC_4K"),
):
    """Read image metadata and classify its GIS footprint without running AI."""
    _require_lgu_user()
    temp_path, uploaded_name = await _save_upload_to_temp(image)
    try:
        clean_altitude = _require_metric_value(
            "altitude",
            altitude,
            min_value=_MIN_ANALYSIS_ALTITUDE_M,
            max_value=_MAX_ANALYSIS_ALTITUDE_M,
        )
        metadata = ExifExtractor.extract_all_metadata(str(temp_path))
        result = _build_image_location_preflight(metadata, clean_altitude, drone_model)
        if result.get("can_process"):
            result = _calibrate_preflight_footprint(temp_path, result)
        result["uploaded_file_name"] = uploaded_name
        return result
    except HTTPException:
        raise
    except Exception as error:
        traceback.print_exc()
        raise HTTPException(
            status_code=400,
            detail=f"Could not inspect the selected image location: {error}",
        ) from error
    finally:
        _cleanup_temp(temp_path)


@router.post("/process")
async def process_image(
    image: UploadFile = File(...),
    altitude: float = Form(6.0),
    drone_model: str = Form("Autel_EVO_II_Pro"),
    canopy_buffer: float = Form(2.0),
    hexagon_size: float = Form(1.5),
    ai_confidence: float = Form(0.80),
    detection_mode: Optional[str] = Form(None),
    ai_runtime_tuning: str = Form("{}"),
    species: Optional[str] = Form(None),
    allow_partial_map_overlap: bool = Form(False),
):
    """Run the canopy-analysis workflow and return the full JSON response."""
    _require_lgu_user()
    temp_path, uploaded_name = await _save_upload_to_temp(image)
    try:
        with processing_job():
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
                species=species,
                allow_partial_map_overlap=allow_partial_map_overlap,
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
    ai_confidence: float = Form(0.80),
    detection_mode: Optional[str] = Form(None),
    ai_runtime_tuning: str = Form("{}"),
    species: Optional[str] = Form(None),
    allow_partial_map_overlap: bool = Form(False),
):
    """Stream real pipeline progress as Server-Sent Events.

    Event types yielded as `data: {json}\\n\\n` lines:
      * {"type": "progress", "stage": str, "pct": int}  — periodic checkpoints
      * {"type": "result",   "payload": {...}}          — terminal success
      * {"type": "error",    "detail": str}             — terminal failure
    The stream closes after the terminal event.
    """
    _require_lgu_user()
    temp_path, uploaded_name = await _save_upload_to_temp(image)
    events: queue_module.Queue = queue_module.Queue()

    def progress_cb(event: dict[str, Any]) -> None:
        events.put(event)

    def runner() -> None:
        try:
            with processing_job():
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
                    species=species,
                    allow_partial_map_overlap=allow_partial_map_overlap,
                )
            events.put({"type": "result", "payload": payload})
        except HTTPException as error:
            events.put({
                "type": "error",
                "detail": str(error.detail),
                "status_code": error.status_code,
            })
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
    user = _require_lgu_user()
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

    try:
        analysis_id, new_points, skipped = save_analysis(
            image_name=cached_payload["image_name"],
            center_lat=center_lat,
            center_lon=center_lon,
            results=cached_payload["results"],
            hexagons=cached_payload["safe_hexagons"],
            user_id=int(user["id"]),
            stored_assets=cached_payload.get("stored_assets") or [],
        )
    except OutsideVisibleMapError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        _pop_processing_cache(body.analysis_key)
        raise HTTPException(status_code=500, detail=f"Could not save analysis: {error}") from error

    cached_payload["response_payload"]["saved"] = True
    cached_payload["response_payload"]["analysis_id"] = analysis_id
    analysis_name = get_analysis_by_id(analysis_id)["image_name"]
    cached_payload["response_payload"]["source_image_name"] = cached_payload["image_name"]
    cached_payload["response_payload"]["uploaded_file_name"] = analysis_name
    cached_payload["response_payload"]["save_summary"] = {
        "analysis_id": analysis_id,
        "new_points": new_points,
        "skipped_duplicates": skipped,
    }
    _pop_processing_cache(body.analysis_key)

    return {
        "status": "saved",
        "analysis_id": analysis_id,
        "analysis_name": analysis_name,
        "new_points": new_points,
        "skipped_duplicates": skipped,
    }
