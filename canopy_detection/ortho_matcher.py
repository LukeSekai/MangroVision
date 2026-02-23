"""
ortho_matcher.py
================
Matches a drone image against the WebODM orthophoto GeoTIFF to:
1. Automatically detect the camera heading (yaw)
2. Map any drone-image pixel to an exact GPS coordinate

The orthophoto has a known GeoTransform (EPSG:32651 = WGS84 / UTM Zone 51N).
We extract the orthophoto patch that covers the drone image's approximate
footprint, match keypoints to find the homography, then convert pixels
through that homography + GeoTransform to GPS.

No manual heading input required.
"""

import cv2
import numpy as np
import pyproj
from PIL import Image as PILImage
from pathlib import Path
from typing import Optional, Tuple, Dict

# ─────────────────────────────────────────────────────────────────────────────
# GeoTIFF constants (read once from MAP/odm_orthophoto/odm_orthophoto.tif tags)
# ModelTiepointTag (33922): pixel(0,0) → UTM
# ModelPixelScaleTag (33550): metres per pixel
# ─────────────────────────────────────────────────────────────────────────────
ORTHO_ORIGIN_X = 458971.7055811524    # UTM Easting  of pixel (0,0) top-left
ORTHO_ORIGIN_Y = 1191822.928832103   # UTM Northing of pixel (0,0) top-left
ORTHO_GSD_X    = 0.04998046150839694  # m/pixel  (X → East)
ORTHO_GSD_Y    = 0.049985695278993825 # m/pixel  (Y → North, image Y increases downward)
ORTHO_GSD      = (ORTHO_GSD_X + ORTHO_GSD_Y) / 2.0
ORTHO_PATH     = Path(__file__).parent.parent / "MAP" / "odm_orthophoto" / "odm_orthophoto.tif"

# CRS transformers (lazy-initialized)
_to_utm   = None   # GPS (lon,lat) → UTM 51N (x,y)
_from_utm = None   # UTM 51N (x,y) → GPS (lon,lat)

def _get_transformers():
    global _to_utm, _from_utm
    if _to_utm is None:
        _to_utm   = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
        _from_utm = pyproj.Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
    return _to_utm, _from_utm


# ─────────────────────────────────────────────────────────────────────────────
# Coordinate helpers
# ─────────────────────────────────────────────────────────────────────────────

def gps_to_ortho_pixel(lat: float, lon: float) -> Tuple[float, float]:
    """GPS (lat,lon) → orthophoto pixel (px, py)"""
    to_utm, _ = _get_transformers()
    utm_x, utm_y = to_utm.transform(lon, lat)
    px = (utm_x - ORTHO_ORIGIN_X) / ORTHO_GSD_X
    py = (ORTHO_ORIGIN_Y - utm_y)  / ORTHO_GSD_Y
    return px, py


def ortho_pixel_to_gps(px: float, py: float) -> Tuple[float, float]:
    """Orthophoto pixel (px, py) → GPS (lat, lon)"""
    _, from_utm = _get_transformers()
    utm_x = ORTHO_ORIGIN_X + px * ORTHO_GSD_X
    utm_y = ORTHO_ORIGIN_Y - py * ORTHO_GSD_Y
    lon, lat = from_utm.transform(utm_x, utm_y)
    return lat, lon


# ─────────────────────────────────────────────────────────────────────────────
# Orthophoto image loader (cached)
# ─────────────────────────────────────────────────────────────────────────────
_ortho_cache: Optional[np.ndarray] = None

def load_orthophoto() -> Optional[np.ndarray]:
    """Load orthophoto as BGR numpy array (cached)."""
    global _ortho_cache
    if _ortho_cache is not None:
        return _ortho_cache
    if not ORTHO_PATH.exists():
        return None
    try:
        pil_img = PILImage.open(str(ORTHO_PATH)).convert("RGB")
        _ortho_cache = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        return _ortho_cache
    except Exception as e:
        print(f"[OrthoMatcher] Failed to load orthophoto: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main matching function
# ─────────────────────────────────────────────────────────────────────────────

def match_drone_to_ortho(
    drone_image: np.ndarray,
    center_lat: float,
    center_lon: float,
    drone_gsd: float,
    margin_factor: float = 1.6,
    max_features: int = 8000,
) -> Dict:
    """
    Match *drone_image* against the orthophoto patch that covers the same area.

    Returns a dict with:
        success   (bool)  – whether a valid homography was found
        H         (3×3)   – homography: drone pixel → ortho pixel
        heading   (float) – estimated camera heading in degrees (0=N, 90=E …)
        confidence(float) – 0–1 match quality score
        error     (str)   – error message if success=False
    """
    ortho = load_orthophoto()
    if ortho is None:
        return {"success": False, "error": "Orthophoto not found"}

    oh, ow = ortho.shape[:2]
    dh, dw = drone_image.shape[:2]

    # ── Step 1: Find drone centre in orthophoto pixel space ──────────────────
    cx_o, cy_o = gps_to_ortho_pixel(center_lat, center_lon)

    # ── Step 2: Calculate drone footprint in ortho pixels ────────────────────
    scale = drone_gsd / ORTHO_GSD              # drone pixel → ortho pixel
    half_w_o = int(dw * scale / 2 * margin_factor)
    half_h_o = int(dh * scale / 2 * margin_factor)

    x1 = max(0, int(cx_o - half_w_o))
    y1 = max(0, int(cy_o - half_h_o))
    x2 = min(ow, int(cx_o + half_w_o))
    y2 = min(oh, int(cy_o + half_h_o))

    if (x2 - x1) < 80 or (y2 - y1) < 80:
        return {"success": False, "error": "Drone footprint outside orthophoto bounds"}

    ortho_patch = ortho[y1:y2, x1:x2]

    # ── Step 3a: Resize drone image to ortho patch scale ─────────────────────
    drone_small = cv2.resize(
        drone_image,
        (int(dw * scale), int(dh * scale)),
        interpolation=cv2.INTER_LINEAR
    )

    # ── Step 3b: Crop drone_small to the area that OVERLAPS with the ortho ───
    # Find what region of the drone (at ortho scale) actually appears in the patch
    dcx = dw * scale / 2      # drone centre in drone_small coords
    dcy = dh * scale / 2
    # Patch covers ortho [x1..x2, y1..y2], drone covers [cx_o-dw*scale/2 .. cx_o+dw*scale/2]
    # but without rotation we can't know exactly, so use the patch bbox mapped back
    local_x1 = int(max(0, x1 - (cx_o - dw * scale / 2)))
    local_y1 = int(max(0, y1 - (cy_o - dh * scale / 2)))
    local_x2 = int(min(drone_small.shape[1], local_x1 + (x2 - x1)))
    local_y2 = int(min(drone_small.shape[0], local_y1 + (y2 - y1)))

    if (local_x2 - local_x1) > 50 and (local_y2 - local_y1) > 50:
        drone_crop  = drone_small[local_y1:local_y2, local_x1:local_x2]
        crop_offset = (local_x1, local_y1)   # offset in drone_small space
    else:
        drone_crop  = drone_small
        crop_offset = (0, 0)

    # ── Step 4: SIFT feature matching ────────────────────────────────────────
    try:
        sift = cv2.SIFT_create(nfeatures=max_features, contrastThreshold=0.02)
    except cv2.error:
        sift = cv2.ORB_create(nfeatures=max_features)

    gray1 = cv2.cvtColor(drone_crop,   cv2.COLOR_BGR2GRAY)
    gray2 = cv2.cvtColor(ortho_patch,  cv2.COLOR_BGR2GRAY)

    # Enhance contrast for better feature detection in natural scenes
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray1 = clahe.apply(gray1)
    gray2 = clahe.apply(gray2)

    kp1, des1 = sift.detectAndCompute(gray1, None)
    kp2, des2 = sift.detectAndCompute(gray2, None)

    if des1 is None or des2 is None or len(kp1) < 8 or len(kp2) < 8:
        return {
            "success": False,
            "error": f"Insufficient features: drone={len(kp1) if kp1 else 0}, ortho={len(kp2) if kp2 else 0}"
        }

    # ── Step 5: Match features ───────────────────────────────────────────────
    FLANN_INDEX_KDTREE = 1
    index_params  = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
    search_params = dict(checks=50)
    
    try:
        matcher = cv2.FlannBasedMatcher(index_params, search_params)
        matches_raw = matcher.knnMatch(des1, des2, k=2)
    except cv2.error:
        # ORB descriptors → BF matcher
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
        matches_raw = bf.knnMatch(des1, des2, k=2)

    # Lowe's ratio test (looser for natural/repetitive textures)
    good = [m for m, n in matches_raw if m.distance < 0.80 * n.distance]

    if len(good) < 6:
        return {
            "success": False,
            "error": f"Not enough good matches: {len(good)} (need ≥6). "
                     f"The image may be outside the orthophoto or too different in appearance."
        }

    # ── Step 6: Find homography (drone_crop → ortho_patch) ──────────────────
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H_crop, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inlier_count = int(mask.sum()) if mask is not None else 0
    confidence = inlier_count / len(good) if good else 0.0

    if H_crop is None or inlier_count < 6:
        return {
            "success": False,
            "error": f"Homography failed (inliers={inlier_count}, conf={confidence:.2f})"
        }

    # ── Step 7: Build full homography: full-res drone pixel → ortho pixel ───
    # Chain of transforms:
    #   full_drone  --[S_drone]--> drone_small  --[T_crop]--> drone_crop
    #                                            <inverse>
    #   drone_crop  --[H_crop]--> ortho_patch   --[T_patch]--> full_ortho
    #
    S_drone = np.array([[scale, 0, 0], [0, scale, 0], [0, 0, 1]], dtype=np.float64)

    # Translation: drone_small → drone_crop (subtract crop offset)
    ox, oy = crop_offset
    T_crop_inv = np.array([[1, 0, -ox], [0, 1, -oy], [0, 0, 1]], dtype=np.float64)

    # Translation: ortho_patch → full ortho (add patch offset x1, y1)
    T_patch = np.array([[1, 0, x1], [0, 1, y1], [0, 0, 1]], dtype=np.float64)

    # Final: full_drone → drone_crop → ortho_patch → full_ortho
    H_to_ortho = T_patch @ H_crop @ T_crop_inv @ S_drone

    # ── Step 9: Estimate camera heading from homography ─────────────────────
    # The rotation component of H: extract from top-left 2x2
    # For a pure rotation (ignoring scale/shear), angle = atan2(H[1,0], H[0,0])
    h00, h10 = H_to_ortho[0, 0], H_to_ortho[1, 0]
    heading_rad = np.arctan2(h10, h00)
    heading_deg = np.degrees(heading_rad) % 360

    return {
        "success": True,
        "H": H_to_ortho,
        "heading": heading_deg,
        "confidence": confidence,
        "inliers": inlier_count,
        "total_matches": len(good),
        "patch_bounds": (x1, y1, x2, y2),
        "error": None,
    }


# ─────────────────────────────────────────────────────────────────────────────
# GPS conversion using homography
# ─────────────────────────────────────────────────────────────────────────────

def drone_pixel_to_gps_via_homography(
    px: float, py: float, H: np.ndarray
) -> Tuple[float, float]:
    """
    Convert a drone-image pixel (px, py) to GPS (lat, lon)
    using the homography H (drone pixel → orthophoto pixel).
    """
    # Apply homography to get ortho pixel
    pt = np.array([[[px, py]]], dtype=np.float64)
    ortho_pt = cv2.perspectiveTransform(pt, H)
    ox, oy = ortho_pt[0, 0]
    return ortho_pixel_to_gps(ox, oy)


# ─────────────────────────────────────────────────────────────────────────────
# Fallback: heading-based conversion (used when matching fails)
# ─────────────────────────────────────────────────────────────────────────────

def drone_pixel_to_gps_via_heading(
    px: float, py: float,
    image_w: int, image_h: int,
    center_lat: float, center_lon: float,
    gsd: float,
    heading_deg: float,
) -> Tuple[float, float]:
    """
    Fallback pixel→GPS using camera heading + GSD.
    heading_deg: 0=North, 90=East (top of image points to this direction).
    """
    offset_x_px = px - image_w / 2
    offset_y_px = py - image_h / 2

    offset_x_m =  offset_x_px * gsd
    offset_y_m = -offset_y_px * gsd   # image Y is inverted vs. geographic North

    heading_rad = np.radians(-heading_deg)   # clockwise → CCW rotation
    rotated_x_m = offset_x_m * np.cos(heading_rad) - offset_y_m * np.sin(heading_rad)
    rotated_y_m = offset_x_m * np.sin(heading_rad) + offset_y_m * np.cos(heading_rad)

    mpdlat = 111320.0
    mpdlon = 111320.0 * np.cos(np.radians(center_lat))

    lat = center_lat + rotated_y_m / mpdlat
    lon = center_lon + rotated_x_m / mpdlon
    return lat, lon
