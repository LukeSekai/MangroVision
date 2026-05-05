"""
ortho_matcher.py
================
Matches a drone image against WebODM orthophoto GeoTIFFs to:
1. Automatically detect the camera heading (yaw)
2. Map any drone-image pixel to an exact GPS coordinate

Supports MULTIPLE orthophotos (3 map parts). The correct one is selected
automatically based on the drone image's GPS coordinates.

Orthophoto transforms are read from GeoTIFF metadata when available.
"""

import cv2
import numpy as np
import os
import pyproj
import rasterio
from pathlib import Path
from typing import Optional, Tuple, Dict, List

# ─────────────────────────────────────────────────────────────────────────────
# Multi-Orthophoto Registry
# Each entry: (name, path, origin_x, origin_y, gsd_x, gsd_y, width, height)
# Values read from GeoTIFF ModelTiepointTag (33922) & ModelPixelScaleTag (33550)
# ─────────────────────────────────────────────────────────────────────────────

_DESKTOP = Path.home() / "Desktop"
_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_ACTIVE_ORTHO_PATH = _DESKTOP / "WebODM" / "Practice_final_cut.tif"

_ORTHO_REGISTRY_SEEDS = [
    {
        "name": "1st MAP",
        "path": _DESKTOP / "WebODM" / "1st MAP" / "Task-of-2026-02-19T144959031Z-all (1)" / "odm_orthophoto" / "odm_orthophoto.tif",
    },
    {
        "name": "2nd MAP",
        "path": _DESKTOP / "WebODM" / "2nd MAP" / "Task-of-2026-02-26T144004414Z-all" / "odm_orthophoto" / "odm_orthophoto.tif",
    },
    {
        "name": "3rd MAP",
        "path": _DESKTOP / "WebODM" / "3rd MAP" / "Task-of-2026-02-26T220654471Z-all" / "odm_orthophoto" / "odm_orthophoto.tif",
    },
]

_FALLBACK_ORTHO_METADATA = {
    "1st MAP": {
        "origin_x": 458971.930938,
        "origin_y": 1191823.193120,
        "gsd_x": 0.0499921871,
        "gsd_y": 0.0499914324,
        "width": 2467,
        "height": 3422,
        "crs": "EPSG:32651",
    },
    "2nd MAP": {
        "origin_x": 458878.837225,
        "origin_y": 1191788.394178,
        "gsd_x": 0.0499902759,
        "gsd_y": 0.0499928374,
        "width": 2965,
        "height": 2718,
        "crs": "EPSG:32651",
    },
    "3rd MAP": {
        "origin_x": 458847.596193,
        "origin_y": 1191710.998803,
        "gsd_x": 0.0499898637,
        "gsd_y": 0.0499937509,
        "width": 3835,
        "height": 3082,
        "crs": "EPSG:32651",
    },
}

# Also check the old single-file fallback path
_LEGACY_PATH = Path(__file__).parent.parent / "MAP" / "odm_orthophoto" / "odm_orthophoto.tif"

_ORTHO_SEARCH_ROOTS = [
    _DESKTOP / "WebODM",
    Path(__file__).parent.parent / "MAP",
]

_ORTHO_DIR_HINTS = {"odm_orthophoto"}
_ORTHO_FILE_HINTS = ("ortho", "orthophoto", "merged", "final", "cut", "clipped")
_NON_ORTHO_DIR_HINTS = {"odm_dem"}
_NON_ORTHO_FILE_HINTS = ("dsm", "dtm", "dem")

# ─── Active orthophoto state (set by select_orthophoto) ──────────────────────
_ORTHO_REGISTRY_CACHE: Optional[List[dict]] = None
_ORTHO_REGISTRY_CACHE_KEY: Optional[str] = None
_ENTRY_TRANSFORMERS: Dict[str, Tuple[pyproj.Transformer, pyproj.Transformer]] = {}

ORTHO_ORIGIN_X = 0.0
ORTHO_ORIGIN_Y = 0.0
ORTHO_GSD_X = 0.0
ORTHO_GSD_Y = 0.0
ORTHO_GSD = 0.0
ORTHO_PATH = _LEGACY_PATH
ORTHO_CRS = "EPSG:32651"
_ACTIVE_ORTHO_ENTRY: Optional[dict] = None
_to_ortho_crs = None
_from_ortho_crs = None


def _fallback_entry(seed: dict) -> Optional[dict]:
    """Return baked-in metadata if GeoTIFF metadata is unavailable."""
    fallback = _FALLBACK_ORTHO_METADATA.get(seed["name"])
    if fallback is None:
        return None
    return {
        "name": seed["name"],
        "path": Path(seed["path"]),
        **fallback,
    }


def _read_env_file_value(key: str) -> Optional[str]:
    """Read backend map-source settings from local env files if present."""
    for env_path in (
        _PROJECT_ROOT / "MangroVision_New" / ".env.local",
        _PROJECT_ROOT / "MangroVision_New" / ".env",
    ):
        if not env_path.exists():
            continue
        try:
            for line in env_path.read_text(encoding="utf-8", errors="ignore").splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                name, value = line.split("=", 1)
                if name.strip() == key:
                    return value.strip().strip('"').strip("'")
        except Exception:
            continue
    return None


def _resolve_configured_path(raw_path: str) -> Path:
    expanded = os.path.expandvars(os.path.expanduser(raw_path))
    path = Path(expanded)
    if not path.is_absolute():
        path = _PROJECT_ROOT / path
    return path


def _configured_ortho_seed() -> Optional[dict]:
    """Return the active map GeoTIFF, if the system has one configured."""
    auto_discovery = (
        os.getenv("MANGROVISION_ORTHO_AUTO_DISCOVERY")
        or _read_env_file_value("MANGROVISION_ORTHO_AUTO_DISCOVERY")
        or ""
    ).strip().lower()
    if auto_discovery in {"1", "true", "yes", "on"}:
        return None

    configured_path = (
        os.getenv("MANGROVISION_ORTHO_PATH")
        or _read_env_file_value("MANGROVISION_ORTHO_PATH")
    )
    if configured_path:
        path = _resolve_configured_path(configured_path)
        name = (
            os.getenv("MANGROVISION_ORTHO_NAME")
            or _read_env_file_value("MANGROVISION_ORTHO_NAME")
            or path.stem
        )
        return {"name": name, "path": path}

    if _DEFAULT_ACTIVE_ORTHO_PATH.exists():
        return {"name": "Practice_final_cut", "path": _DEFAULT_ACTIVE_ORTHO_PATH}

    return None


def _read_orthophoto_bgr(path: Path) -> np.ndarray:
    """Load an orthophoto TIFF into a BGR uint8 array."""
    with rasterio.open(path) as dataset:
        bands = [1, 2, 3] if dataset.count >= 3 else [1]
        image = dataset.read(bands)

    if image.shape[0] == 1:
        image = np.repeat(image, 3, axis=0)

    rgb = np.moveaxis(image[:3], 0, -1)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)
    return cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)


def _read_ortho_metadata(seed: dict) -> Optional[dict]:
    """Read orthophoto bounds and pixel scale directly from the GeoTIFF.

    If the seeded path is missing (WebODM tasks sometimes get the orthophoto
    file renamed during export, e.g. `2nd.tif` instead of `odm_orthophoto.tif`),
    fall back to any *.tif sitting in the same `odm_orthophoto` folder so the
    registry self-heals across renames.
    """
    path = Path(seed["path"])
    if not path.exists():
        sibling_tifs = sorted(path.parent.glob("*.tif")) if path.parent.exists() else []
        if not sibling_tifs:
            return None
        path = sibling_tifs[0]
        print(f"[OrthoMatcher] '{seed['name']}': seed path missing, using sibling {path.name}")

    try:
        with rasterio.open(path) as dataset:
            if dataset.crs is None:
                raise ValueError("GeoTIFF has no CRS")

            transform = dataset.transform
            if abs(transform.b) > 1e-9 or abs(transform.d) > 1e-9:
                raise ValueError("rotated orthophoto transforms are not supported")

            return {
                "name": seed["name"],
                "path": path,
                "origin_x": float(transform.c),
                "origin_y": float(transform.f),
                "gsd_x": abs(float(transform.a)),
                "gsd_y": abs(float(transform.e)),
                "width": int(dataset.width),
                "height": int(dataset.height),
                "crs": dataset.crs.to_string(),
            }
    except Exception as exc:
        fallback = _fallback_entry(seed)
        if fallback is not None:
            print(f"[OrthoMatcher] Metadata read failed for '{seed['name']}', using fallback values: {exc}")
            return fallback
        print(f"[OrthoMatcher] Metadata read failed for '{seed['name']}': {exc}")
        return None


def _is_orthophoto_tif(path: Path) -> bool:
    """Return True for GeoTIFFs that look like orthophoto references."""
    suffix = path.suffix.lower()
    if suffix not in {".tif", ".tiff"}:
        return False

    stem = path.stem.lower()
    parts = [part.lower() for part in path.parts]
    if any(part in _NON_ORTHO_DIR_HINTS for part in parts):
        return False
    if any(hint in stem for hint in _NON_ORTHO_FILE_HINTS):
        return False
    if any(part in _ORTHO_DIR_HINTS for part in parts):
        return True
    if any(hint in stem for hint in _ORTHO_FILE_HINTS):
        return True

    # Some exported reference folders are named like FINAL MAP / TRY MAP and
    # contain a single georeferenced TIFF with a custom name.
    return any("map" in part for part in parts)


def _ortho_seed_name(path: Path, root: Path) -> str:
    """Build a readable, unique-ish name for a discovered orthophoto."""
    try:
        rel = path.relative_to(root)
    except ValueError:
        rel = path

    folders = list(rel.parts[:-1])
    lower_folders = [folder.lower() for folder in folders]
    if "odm_orthophoto" in lower_folders:
        folders = folders[:lower_folders.index("odm_orthophoto")]

    if not folders:
        return path.stem
    return " / ".join(folders)


def _discover_ortho_seeds() -> List[dict]:
    """Auto-discover orthophoto GeoTIFFs across the local map folders.

    This scans the whole Desktop/WebODM tree and the project MAP folder, so it
    catches layouts like:
    - 1st/1st/1st-orthophoto.tif
    - 1st MAP/Task-of-.../odm_orthophoto/odm_orthophoto.tif
    - 5th MAP/1/odm_orthophoto/1.tif
    - Map1/MAP-1-orthophoto.tif
    - right/Right Side Map/Map10/Map10-orthophoto.tif
    - six/6th/6th-orthophoto.tif
    """
    found: List[dict] = []
    seen_paths: set[str] = set()

    for root in _ORTHO_SEARCH_ROOTS:
        if not root.exists():
            continue
        for tif in root.rglob("*"):
            if not tif.is_file() or not _is_orthophoto_tif(tif):
                continue
            key = str(tif.resolve()).lower()
            if key in seen_paths:
                continue
            seen_paths.add(key)
            found.append({"name": _ortho_seed_name(tif, root), "path": tif})

    return found


def _get_ortho_registry() -> List[dict]:
    """Return all available orthophotos with live metadata."""
    global _ORTHO_REGISTRY_CACHE, _ORTHO_REGISTRY_CACHE_KEY

    configured_seed = _configured_ortho_seed()
    cache_key = (
        str(Path(configured_seed["path"]).resolve()).lower()
        if configured_seed is not None
        else "__auto_discovery__"
    )
    if _ORTHO_REGISTRY_CACHE is not None and _ORTHO_REGISTRY_CACHE_KEY == cache_key:
        return _ORTHO_REGISTRY_CACHE

    if configured_seed is not None:
        entry = _read_ortho_metadata(configured_seed)
        if entry is None:
            print(f"[OrthoMatcher] Configured orthophoto not available: {configured_seed['path']}")
            _ORTHO_REGISTRY_CACHE = []
        else:
            _ORTHO_REGISTRY_CACHE = [entry]
            print(f"[OrthoMatcher] Using configured orthophoto: {entry['name']} ({entry['path']})")
        _ORTHO_REGISTRY_CACHE_KEY = cache_key
        return _ORTHO_REGISTRY_CACHE

    seeds = list(_ORTHO_REGISTRY_SEEDS)
    seeds.append({"name": "Legacy MAP", "path": _LEGACY_PATH})

    # Append any discovered orthophotos that aren't already in the seed list.
    seen_paths = {str(Path(s["path"])).lower() for s in seeds}
    for discovered in _discover_ortho_seeds():
        key = str(discovered["path"]).lower()
        if key in seen_paths:
            continue
        seeds.append(discovered)
        seen_paths.add(key)

    registry: List[dict] = []
    seen_resolved: set[str] = set()
    for seed in seeds:
        entry = _read_ortho_metadata(seed)
        if entry is None:
            continue
        # Dedupe by the path the metadata reader actually used (handles the
        # case where a seed pointed at a missing filename and the sibling
        # fallback resolved to the same TIFF that auto-discovery also found).
        resolved = str(Path(entry["path"]).resolve()).lower()
        if resolved in seen_resolved:
            continue
        seen_resolved.add(resolved)
        registry.append(entry)

    _ORTHO_REGISTRY_CACHE = registry
    _ORTHO_REGISTRY_CACHE_KEY = cache_key
    return registry


def _get_entry_transformers(entry: dict) -> Tuple[pyproj.Transformer, pyproj.Transformer]:
    """Return GPS <-> orthophoto CRS transformers for an entry."""
    crs_key = entry["crs"]
    if crs_key not in _ENTRY_TRANSFORMERS:
        _ENTRY_TRANSFORMERS[crs_key] = (
            pyproj.Transformer.from_crs("EPSG:4326", crs_key, always_xy=True),
            pyproj.Transformer.from_crs(crs_key, "EPSG:4326", always_xy=True),
        )
    return _ENTRY_TRANSFORMERS[crs_key]


def _ensure_active_ortho() -> dict:
    """Ensure the module has an active orthophoto before pixel/GPS conversion."""
    global _ACTIVE_ORTHO_ENTRY
    if _ACTIVE_ORTHO_ENTRY is None:
        registry = _get_ortho_registry()
        if not registry:
            raise RuntimeError("No orthophoto metadata available")
        _activate_ortho(registry[0])
    return _ACTIVE_ORTHO_ENTRY


def _get_transformers():
    _ensure_active_ortho()
    return _to_ortho_crs, _from_ortho_crs


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
    for entry in _get_ortho_registry():
        to_entry_crs, _ = _get_entry_transformers(entry)
        entry_x, entry_y = to_entry_crs.transform(lon, lat)
        w, e, s, n = _ortho_bounds_utm(entry)
        if not (w <= entry_x <= e and s <= entry_y <= n):
            continue
        # Compute pixel coords in this ortho
        px = int((entry_x - entry["origin_x"]) / entry["gsd_x"])
        py = int((entry["origin_y"] - entry_y) / entry["gsd_y"])
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
                    img = _read_orthophoto_bgr(p)
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


def select_orthophoto(lat: float, lon: float) -> Optional[dict]:
    """
    Pick the orthophoto whose bounds contain the given GPS coordinate.
    Updates the global ORTHO_* variables so every downstream function
    (gps_to_ortho_pixel, ortho_pixel_to_gps, etc.) works with the
    correct map part automatically.

    Returns the chosen registry entry, or None if no match.
    """
    registry = _get_ortho_registry()
    if not registry:
        print("[OrthoMatcher] No orthophoto found for (%.6f, %.6f)" % (lat, lon))
        return None

    best = None
    best_margin = -1e30

    for entry in registry:
        to_entry_crs, _ = _get_entry_transformers(entry)
        entry_x, entry_y = to_entry_crs.transform(lon, lat)
        w, e, s, n = _ortho_bounds_utm(entry)
        # Margin = min distance from point to any edge (positive = inside)
        margin = min(entry_x - w, e - entry_x, entry_y - s, n - entry_y)
        if margin > best_margin:
            best_margin = margin
            best = entry

    if best is None:
        print("[OrthoMatcher] No orthophoto found for (%.6f, %.6f)" % (lat, lon))
        return None

    if best_margin < -200:
        print(
            f"[OrthoMatcher] No orthophoto close enough for ({lat:.6f}, {lon:.6f}) "
            f"(nearest edge distance {abs(best_margin):.0f}m)"
        )
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
        img = _read_orthophoto_bgr(ORTHO_PATH)
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
    center_dx_px = float(center_mapped[0] - cx_o)
    center_dy_px = float(center_mapped[1] - cy_o)
    center_offset_east_m = center_dx_px * ORTHO_GSD_X
    center_offset_north_m = -center_dy_px * ORTHO_GSD_Y
    center_dist = np.hypot(center_dx_px, center_dy_px)
    center_dist_m = center_dist * ORTHO_GSD
    footprint_max_px = max(dw * scale, dh * scale)
    # GPS can be a little noisy, but a valid visual match should still keep the
    # image center close to the tagged camera position. The previous 1.5x
    # footprint allowance could accept convincing false positives on repetitive
    # shoreline/roof textures.
    max_center_drift = min(footprint_max_px * 0.35, 25.0 / max(ORTHO_GSD, 1e-9))
    max_center_drift = max(max_center_drift, 5.0 / max(ORTHO_GSD, 1e-9))
    if center_dist > max_center_drift:
        return {
            "success": False,
            "inliers": inlier_count,
            "confidence": confidence,
            "center_drift_px": float(center_dist),
            "center_drift_m": float(center_dist_m),
            "center_offset_east_m": float(center_offset_east_m),
            "center_offset_north_m": float(center_offset_north_m),
            "center_max_drift_px": float(max_center_drift),
            "center_max_drift_m": float(max_center_drift * ORTHO_GSD),
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
        "center_drift_px": float(center_dist),
        "center_drift_m": float(center_dist_m),
        "center_offset_east_m": float(center_offset_east_m),
        "center_offset_north_m": float(center_offset_north_m),
        "center_max_drift_px": float(max_center_drift),
        "center_max_drift_m": float(max_center_drift * ORTHO_GSD),
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
    strongest result by inliers, confidence, and center-drift sanity.

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
    registry = _get_ortho_registry()

    scored_entries = []
    for entry in registry:
        to_entry_crs, _ = _get_entry_transformers(entry)
        entry_x, entry_y = to_entry_crs.transform(center_lon, center_lat)
        w, e, s, n = _ortho_bounds_utm(entry)
        margin = min(entry_x - w, e - entry_x, entry_y - s, n - entry_y)
        scored_entries.append((margin, entry))
    scored_entries.sort(key=lambda x: -x[0])  # highest margin first (most inside)

    inside_entries = [(margin, entry) for margin, entry in scored_entries if margin >= 0]
    if inside_entries:
        scored_entries = inside_entries

    if not scored_entries:
        return {"success": False, "error": "No orthophoto files found — check WebODM folder paths"}

    best_result = None
    best_entry = None
    best_score = -1.0
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

        if result["success"]:
            drift_ratio = result.get("center_drift_px", 0.0) / max(result.get("center_max_drift_px", 1.0), 1.0)
            score = result["inliers"] * (0.5 + result["confidence"]) / (1.0 + drift_ratio)
            result["match_score"] = float(score)
            result["ortho_name"] = tag
            result["ortho_path"] = str(entry["path"])
            result["ortho_margin_m"] = float(margin)
            print(
                f"[OrthoMatcher] Match on '{tag}': {result['inliers']} inliers, "
                f"conf={result['confidence']:.0%}, drift={result.get('center_drift_px', 0):.0f}px, "
                f"score={score:.1f}"
            )
            if score > best_score:
                best_result = result
                best_entry = entry
                best_score = score
        elif not result["success"]:
            errors.append(f"{tag}: {result['error']}")

    if best_result is not None:
        # Re-activate the winning orthophoto so pixel→GPS conversions use it
        if best_entry is not None:
            _activate_ortho(best_entry)
            print(
                f"[OrthoMatcher] Best match: '{best_entry['name']}' "
                f"score={best_score:.1f}, path={best_entry['path']}"
            )
        return best_result

    # All failed — return the most informative error
    return {
        "success": False,
        "error": "No match on any orthophoto. " + "; ".join(errors) if errors else "No orthophoto overlap",
    }


def _activate_ortho(entry: dict):
    """Set the global ORTHO_* variables to use a specific ortho entry."""
    global ORTHO_ORIGIN_X, ORTHO_ORIGIN_Y, ORTHO_GSD_X, ORTHO_GSD_Y, ORTHO_GSD, ORTHO_PATH, ORTHO_CRS
    global _ACTIVE_ORTHO_ENTRY, _to_ortho_crs, _from_ortho_crs
    ORTHO_ORIGIN_X = entry["origin_x"]
    ORTHO_ORIGIN_Y = entry["origin_y"]
    ORTHO_GSD_X    = entry["gsd_x"]
    ORTHO_GSD_Y    = entry["gsd_y"]
    ORTHO_GSD      = (ORTHO_GSD_X + ORTHO_GSD_Y) / 2.0
    ORTHO_PATH     = entry["path"]
    ORTHO_CRS      = entry["crs"]
    _ACTIVE_ORTHO_ENTRY = entry
    _to_ortho_crs, _from_ortho_crs = _get_entry_transformers(entry)


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
