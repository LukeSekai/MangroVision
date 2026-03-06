"""
ortho_matcher.py
================
Matches a drone image against WebODM orthophoto GeoTIFFs to:
1. Automatically detect the camera heading (yaw)
2. Map any drone-image pixel to an exact GPS coordinate

Supports MULTIPLE orthophotos (3 map parts). The correct one is selected
automatically based on the drone image's GPS coordinates.

The orthophotos use EPSG:32651 = WGS84 / UTM Zone 51N.
"""

import cv2
import numpy as np
import pyproj
from PIL import Image as PILImage
from pathlib import Path
from typing import Optional, Tuple, Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# Multi-Orthophoto Registry
# Each entry: (name, path, origin_x, origin_y, gsd_x, gsd_y, width, height)
# Values read from GeoTIFF ModelTiepointTag (33922) & ModelPixelScaleTag (33550)
# ─────────────────────────────────────────────────────────────────────────────

_DESKTOP = Path.home() / "Desktop"

ORTHO_REGISTRY = [
    {
        "name": "1st MAP",
        "path": _DESKTOP / "WebODM" / "1st MAP" / "Task-of-2026-02-19T144959031Z-all (1)" / "odm_orthophoto" / "odm_orthophoto.tif",
        "origin_x": 458971.930938,      # UTM Easting  of pixel (0,0) top-left
        "origin_y": 1191823.193120,     # UTM Northing of pixel (0,0) top-left
        "gsd_x": 0.0499921871,          # m/pixel (X → East)
        "gsd_y": 0.0499914324,          # m/pixel (Y → North, image Y ↓)
        "width": 2467,
        "height": 3422,
    },
    {
        "name": "2nd MAP",
        "path": _DESKTOP / "WebODM" / "2nd MAP" / "Task-of-2026-02-26T144004414Z-all" / "odm_orthophoto" / "odm_orthophoto.tif",
        "origin_x": 458878.837225,
        "origin_y": 1191788.394178,
        "gsd_x": 0.0499902759,
        "gsd_y": 0.0499928374,
        "width": 2965,
        "height": 2718,
    },
    {
        "name": "3rd MAP",
        "path": _DESKTOP / "WebODM" / "3rd MAP" / "Task-of-2026-02-26T220654471Z-all" / "odm_orthophoto" / "odm_orthophoto.tif",
        "origin_x": 458847.596193,
        "origin_y": 1191710.998803,
        "gsd_x": 0.0499898637,
        "gsd_y": 0.0499937509,
        "width": 3835,
        "height": 3082,
    },
]

# Also check the old single-file fallback path
_LEGACY_PATH = Path(__file__).parent.parent / "MAP" / "odm_orthophoto" / "odm_orthophoto.tif"

# ─── Active orthophoto state (set by select_orthophoto) ──────────────────────
ORTHO_ORIGIN_X = 458971.930938
ORTHO_ORIGIN_Y = 1191823.193120
ORTHO_GSD_X    = 0.0499921871
ORTHO_GSD_Y    = 0.0499914324
ORTHO_GSD      = (ORTHO_GSD_X + ORTHO_GSD_Y) / 2.0
ORTHO_PATH     = _LEGACY_PATH      # overwritten by select_orthophoto()

# CRS transformers (lazy-initialized)
_to_utm   = None   # GPS (lon,lat) → UTM 51N (x,y)
_from_utm = None   # UTM 51N (x,y) → GPS (lon,lat)

def _get_transformers():
    global _to_utm, _from_utm
    if _to_utm is None:
        _to_utm   = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:32651", always_xy=True)
        _from_utm = pyproj.Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
    return _to_utm, _from_utm


def _ortho_bounds_utm(entry: dict) -> Tuple[float, float, float, float]:
    """Return (west, east, south, north) UTM bounds for an ortho entry."""
    west  = entry["origin_x"]
    north = entry["origin_y"]
    east  = west  + entry["width"]  * entry["gsd_x"]
    south = north - entry["height"] * entry["gsd_y"]
    return west, east, south, north


def is_inside_any_orthophoto(lat: float, lon: float) -> bool:
    """
    Check if a GPS coordinate falls inside ANY registered orthophoto
    AND has actual imagery (non-black pixel).  WebODM orthophotos have
    irregular boundaries — NoData regions are (0,0,0) black pixels.
    """
    to_utm, _ = _get_transformers()
    utm_x, utm_y = to_utm.transform(lon, lat)

    for entry in ORTHO_REGISTRY:
        w, e, s, n = _ortho_bounds_utm(entry)
        if not (w <= utm_x <= e and s <= utm_y <= n):
            continue
        # Compute pixel coords in this ortho
        px = int((utm_x - entry["origin_x"]) / entry["gsd_x"])
        py = int((entry["origin_y"] - utm_y) / entry["gsd_y"])
        if px < 0 or py < 0 or px >= entry["width"] or py >= entry["height"]:
            continue
        # Check if actual pixel has data (not NoData black)
        key = str(entry["path"])
        if key in _ortho_cache:
            ortho_img = _ortho_cache[key]
            b, g, r = ortho_img[py, px]
            if int(b) + int(g) + int(r) > 30:  # non-black threshold
                return True
        else:
            # If the ortho isn't loaded yet, try loading just this one
            p = Path(entry["path"])
            if p.exists():
                try:
                    pil_img = PILImage.open(str(p)).convert("RGB")
                    img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                    _ortho_cache[key] = img
                    b, g, r = img[py, px]
                    if int(b) + int(g) + int(r) > 30:
                        return True
                except Exception:
                    # Can't load → fall back to rectangle check
                    return True
            else:
                # File missing → accept rectangle check
                return True
    return False


def _get_ortho_pixel_bgr(lat: float, lon: float) -> Optional[Tuple[int, int, int]]:
    """
    Return the BGR pixel value at a GPS location from the best orthophoto.
    Returns None if outside all orthophotos or on NoData.
    """
    to_utm, _ = _get_transformers()
    utm_x, utm_y = to_utm.transform(lon, lat)

    for entry in ORTHO_REGISTRY:
        w, e, s, n = _ortho_bounds_utm(entry)
        if not (w <= utm_x <= e and s <= utm_y <= n):
            continue
        px = int((utm_x - entry["origin_x"]) / entry["gsd_x"])
        py = int((entry["origin_y"] - utm_y) / entry["gsd_y"])
        if px < 0 or py < 0 or px >= entry["width"] or py >= entry["height"]:
            continue
        key = str(entry["path"])
        if key not in _ortho_cache:
            p = Path(entry["path"])
            if p.exists():
                try:
                    pil_img = PILImage.open(str(p)).convert("RGB")
                    _ortho_cache[key] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception:
                    continue
            else:
                continue
        ortho_img = _ortho_cache[key]
        b, g, r = ortho_img[py, px]
        if int(b) + int(g) + int(r) <= 30:
            continue  # NoData
        return (int(b), int(g), int(r))
    return None


def is_ortho_pixel_vegetation(lat: float, lon: float, radius_px: int = 20,
                               threshold: float = 0.25) -> bool:
    """
    Check if a GPS location falls on or NEAR vegetation in the orthophoto.

    Parameters
    ----------
    radius_px : int
        Sampling radius in ortho pixels.  Default 20 → ~2 m diameter at
        5 cm/px GSD, which covers the 1 m hexagon plus a safety margin.
    threshold : float
        Fraction of valid pixels that must be vegetation to trigger True.
        Default 0.25 → if ≥25 % of the surrounding area is canopy,
        the spot is too close to existing trees for safe planting.

    Vegetation pixels are detected via three complementary HSV rules:
      1. Bright green canopy  :  H 20-95, S ≥ 25, V ≥ 40
      2. Dark / shadowed green:  H 20-95, S ≥ 20, V  10-39
      3. Dark saturated foliage:           S ≥ 40, V  10-60
         (catches very dark canopy whose hue wraps or is noisy)
    Additionally a simple G-dominance check is applied:
      G channel > R channel AND G channel > B channel AND G ≥ 30
    Any pixel matching EITHER the HSV rules OR the G-dominance rule is
    counted as vegetation.
    """
    to_utm, _ = _get_transformers()
    utm_x, utm_y = to_utm.transform(lon, lat)

    for entry in ORTHO_REGISTRY:
        w, e, s, n = _ortho_bounds_utm(entry)
        if not (w <= utm_x <= e and s <= utm_y <= n):
            continue
        cpx = int((utm_x - entry["origin_x"]) / entry["gsd_x"])
        cpy = int((entry["origin_y"] - utm_y) / entry["gsd_y"])
        if cpx < 0 or cpy < 0 or cpx >= entry["width"] or cpy >= entry["height"]:
            continue
        key = str(entry["path"])
        if key not in _ortho_cache:
            p = Path(entry["path"])
            if p.exists():
                try:
                    pil_img = PILImage.open(str(p)).convert("RGB")
                    _ortho_cache[key] = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
                except Exception:
                    continue
            else:
                continue
        ortho_img = _ortho_cache[key]
        oh, ow = ortho_img.shape[:2]

        # Sample a (2*radius+1)x(2*radius+1) patch
        y1 = max(0, cpy - radius_px)
        y2 = min(oh, cpy + radius_px + 1)
        x1 = max(0, cpx - radius_px)
        x2 = min(ow, cpx + radius_px + 1)
        patch = ortho_img[y1:y2, x1:x2]

        if patch.size == 0:
            continue

        # ── Mask out NoData / black pixels ──
        b_ch = patch[:, :, 0].astype(np.int32)
        g_ch = patch[:, :, 1].astype(np.int32)
        r_ch = patch[:, :, 2].astype(np.int32)
        not_black = (b_ch + g_ch + r_ch) > 30
        n_valid = int(not_black.sum())
        if n_valid == 0:
            continue

        # ── HSV-based vegetation detection ──
        patch_hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
        h = patch_hsv[:, :, 0]
        s = patch_hsv[:, :, 1]
        v = patch_hsv[:, :, 2]

        green_bright = (h >= 20) & (h <= 95) & (s >= 25) & (v >= 40)
        green_dark   = (h >= 20) & (h <= 95) & (s >= 20) & (v >= 10) & (v < 40)
        dark_sat     = (s >= 40) & (v >= 10) & (v <= 60)
        hsv_veg      = green_bright | green_dark | dark_sat

        # ── G-channel dominance (catches greenish tones HSV might miss) ──
        g_dom = (g_ch > r_ch) & (g_ch > b_ch) & (g_ch >= 30)

        veg_mask = (hsv_veg | g_dom) & not_black
        veg_ratio = np.count_nonzero(veg_mask) / n_valid
        return veg_ratio >= threshold

    return False


def get_combined_ortho_bounds_gps() -> Tuple[float, float, float, float]:
    """
    Return the combined bounding box of ALL orthophotos in GPS (lat/lon).
    Returns (min_lat, max_lat, min_lon, max_lon).
    """
    _, from_utm = _get_transformers()
    all_lats = []
    all_lons = []
    for entry in ORTHO_REGISTRY:
        w, e, s, n = _ortho_bounds_utm(entry)
        for utm_x, utm_y in [(w, s), (w, n), (e, s), (e, n)]:
            lon, lat = from_utm.transform(utm_x, utm_y)
            all_lats.append(lat)
            all_lons.append(lon)
    if not all_lats:
        return (0, 0, 0, 0)
    return min(all_lats), max(all_lats), min(all_lons), max(all_lons)


def select_orthophoto(lat: float, lon: float) -> Optional[dict]:
    """
    Pick the orthophoto whose bounds contain the given GPS coordinate.
    Updates the global ORTHO_* variables so every downstream function
    (gps_to_ortho_pixel, ortho_pixel_to_gps, etc.) works with the
    correct map part automatically.

    Returns the chosen registry entry, or None if no match.
    """
    global ORTHO_ORIGIN_X, ORTHO_ORIGIN_Y, ORTHO_GSD_X, ORTHO_GSD_Y, ORTHO_GSD, ORTHO_PATH

    to_utm, _ = _get_transformers()
    utm_x, utm_y = to_utm.transform(lon, lat)

    best = None
    best_margin = -1e30

    for entry in ORTHO_REGISTRY:
        if not entry["path"].exists():
            continue
        w, e, s, n = _ortho_bounds_utm(entry)
        # Margin = min distance from point to any edge (positive = inside)
        margin = min(utm_x - w, e - utm_x, utm_y - s, n - utm_y)
        if margin > best_margin:
            best_margin = margin
            best = entry

    if best is None:
        # Try legacy path as last resort
        if _LEGACY_PATH.exists():
            print("[OrthoMatcher] No registry match — using legacy path")
            ORTHO_PATH = _LEGACY_PATH
            return None
        print("[OrthoMatcher] No orthophoto found for (%.6f, %.6f)" % (lat, lon))
        return None

    # Set globals
    _activate_ortho(best)

    inside = best_margin > 0
    print(f"[OrthoMatcher] Selected '{best['name']}' for ({lat:.6f}, {lon:.6f}) "
          f"({'inside' if inside else 'nearest, %.0fm from edge' % abs(best_margin)})")
    return best


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
# Orthophoto image loader (cached per path)
# ─────────────────────────────────────────────────────────────────────────────
_ortho_cache: Dict[str, np.ndarray] = {}

def load_orthophoto() -> Optional[np.ndarray]:
    """Load the currently-selected orthophoto as BGR numpy array (cached)."""
    global _ortho_cache
    key = str(ORTHO_PATH)
    if key in _ortho_cache:
        return _ortho_cache[key]
    if not ORTHO_PATH.exists():
        return None
    try:
        print(f"[OrthoMatcher] Loading {ORTHO_PATH.name} …")
        pil_img = PILImage.open(str(ORTHO_PATH)).convert("RGB")
        img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        _ortho_cache[key] = img
        return img
    except Exception as e:
        print(f"[OrthoMatcher] Failed to load orthophoto: {e}")
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Internal: match against a single orthophoto (already selected via globals)
# ─────────────────────────────────────────────────────────────────────────────

def _match_single_ortho(
    drone_image: np.ndarray,
    center_lat: float,
    center_lon: float,
    drone_gsd: float,
    margin_factor: float = 1.6,
    max_features: int = 8000,
) -> Dict:
    """Match drone_image against the CURRENTLY SELECTED orthophoto."""
    ortho = load_orthophoto()
    if ortho is None:
        return {"success": False, "error": "Orthophoto not loaded", "inliers": 0, "confidence": 0}

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
        return {"success": False, "inliers": 0, "confidence": 0, "error": "Drone footprint outside orthophoto bounds"}

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
            "inliers": 0,
            "confidence": 0,
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

    if len(good) < 8:
        return {
            "success": False,
            "inliers": 0,
            "confidence": 0,
            "error": f"Not enough good matches: {len(good)} (need ≥8). "
                     f"The image may be outside the orthophoto or too different in appearance."
        }

    # ── Step 6: Find homography (drone_crop → ortho_patch) ──────────────────
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)

    H_crop, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    inlier_count = int(mask.sum()) if mask is not None else 0
    confidence = inlier_count / len(good) if good else 0.0

    if H_crop is None or inlier_count < 10:
        return {
            "success": False,
            "inliers": inlier_count,
            "confidence": confidence,
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

    # ── Step 8: Sanity check — corners must spread out, not collapse ─────────
    corners_drone = np.float32([
        [[0, 0]], [[dw, 0]], [[0, dh]], [[dw, dh]]
    ])
    corners_ortho = cv2.perspectiveTransform(corners_drone, H_to_ortho)
    cos = corners_ortho.reshape(4, 2)
    # The mapped area should be a reasonable fraction of the drone footprint
    spread_x = cos[:, 0].max() - cos[:, 0].min()
    spread_y = cos[:, 1].max() - cos[:, 1].min()
    expected_spread = min(dw, dh) * scale * 0.3  # at least 30% of expected footprint
    if spread_x < expected_spread or spread_y < expected_spread:
        return {
            "success": False,
            "inliers": inlier_count,
            "confidence": confidence,
            "error": f"Homography degenerate — mapped area too small "
                     f"({spread_x:.0f}x{spread_y:.0f} vs expected ≥{expected_spread:.0f}px)"
        }

    # Also check: centre of drone should map near the expected ortho centre
    center_mapped = cv2.perspectiveTransform(
        np.float32([[[dw/2, dh/2]]]), H_to_ortho
    )[0, 0]
    center_dist = np.hypot(center_mapped[0] - cx_o, center_mapped[1] - cy_o)
    max_center_drift = max(dw, dh) * scale * 1.5  # allow up to 1.5x footprint drift
    if center_dist > max_center_drift:
        return {
            "success": False,
            "inliers": inlier_count,
            "confidence": confidence,
            "error": f"Homography centre drifted {center_dist:.0f}px "
                     f"(max {max_center_drift:.0f}px)"
        }

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
# Public: try ALL orthophotos and return the best match
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
    Match *drone_image* against ALL available orthophoto maps and return the
    result with the highest number of inliers.

    Tries the best-fit orthophoto first (by GPS), then all others. This
    handles cases where the drone is near the edge of one map but overlaps
    better with another.

    Returns a dict with:
        success   (bool)  – whether a valid homography was found
        H         (3×3)   – homography: drone pixel → ortho pixel
        heading   (float) – estimated camera heading in degrees (0=N, 90=E …)
        confidence(float) – 0–1 match quality score
        error     (str)   – error message if success=False
    """
    # Sort orthophotos: best-fit first, then other available ones
    to_utm, _ = _get_transformers()
    utm_x, utm_y = to_utm.transform(center_lon, center_lat)

    scored_entries = []
    for entry in ORTHO_REGISTRY:
        if not entry["path"].exists():
            continue
        w, e, s, n = _ortho_bounds_utm(entry)
        margin = min(utm_x - w, e - utm_x, utm_y - s, n - utm_y)
        scored_entries.append((margin, entry))
    scored_entries.sort(key=lambda x: -x[0])  # highest margin first (most inside)

    if not scored_entries:
        return {"success": False, "error": "No orthophoto files found — check WebODM folder paths"}

    best_result = None
    best_inliers = -1
    errors = []

    for margin, entry in scored_entries:
        # Set globals for this orthophoto
        _activate_ortho(entry)
        tag = entry["name"]

        # Check if GPS is reasonably close (within 200m of ortho edge)
        if margin < -200:
            continue

        print(f"[OrthoMatcher] Trying '{tag}' (margin={margin:.0f}m) …")
        result = _match_single_ortho(
            drone_image, center_lat, center_lon, drone_gsd,
            margin_factor, max_features,
        )

        if result["success"] and result["inliers"] > best_inliers:
            best_result = result
            best_inliers = result["inliers"]
            best_result["ortho_name"] = tag
            # If we have a really good match, stop early
            if best_inliers >= 30 and result["confidence"] >= 0.35:
                print(f"[OrthoMatcher] Good match on '{tag}': {best_inliers} inliers, conf={result['confidence']:.0%}")
                break
        elif not result["success"]:
            errors.append(f"{tag}: {result['error']}")

    if best_result is not None:
        # Re-activate the winning orthophoto so pixel→GPS conversions use it
        for _, entry in scored_entries:
            if entry["name"] == best_result.get("ortho_name"):
                _activate_ortho(entry)
                break
        return best_result

    # All failed — return the most informative error
    return {
        "success": False,
        "error": "No match on any orthophoto. " + "; ".join(errors) if errors else "No orthophoto overlap",
    }


def _activate_ortho(entry: dict):
    """Set the global ORTHO_* variables to use a specific ortho entry."""
    global ORTHO_ORIGIN_X, ORTHO_ORIGIN_Y, ORTHO_GSD_X, ORTHO_GSD_Y, ORTHO_GSD, ORTHO_PATH
    ORTHO_ORIGIN_X = entry["origin_x"]
    ORTHO_ORIGIN_Y = entry["origin_y"]
    ORTHO_GSD_X    = entry["gsd_x"]
    ORTHO_GSD_Y    = entry["gsd_y"]
    ORTHO_GSD      = (ORTHO_GSD_X + ORTHO_GSD_Y) / 2.0
    ORTHO_PATH     = entry["path"]


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
