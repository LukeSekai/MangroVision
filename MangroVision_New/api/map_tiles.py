"""Map tile serving for static XYZ pyramids and GeoTIFF-backed map folders."""

from functools import lru_cache
from pathlib import Path

from fastapi import APIRouter, HTTPException
from fastapi.responses import FileResponse, Response

router = APIRouter()

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_MAP_DIR = _PROJECT_ROOT / "MAP"
_TILE_SIZE = 256
_MAX_ZOOM = 30
_WEB_MERCATOR_HALF_WORLD = 20037508.342789244


def _is_relative_to(path: Path, parent: Path) -> bool:
    try:
        path.relative_to(parent)
        return True
    except ValueError:
        return False


def _media_type(extension: str) -> str:
    normalized = extension.lower()
    if normalized in {"jpg", "jpeg"}:
        return "image/jpeg"
    if normalized == "webp":
        return "image/webp"
    return "image/png"


def _tile_bounds_3857(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    world = _WEB_MERCATOR_HALF_WORLD * 2.0
    span = world / (2**z)
    left = -_WEB_MERCATOR_HALF_WORLD + (x * span)
    right = left + span
    top = _WEB_MERCATOR_HALF_WORLD - (y * span)
    bottom = top - span
    return left, bottom, right, top


def _static_tile_path(tileset: str, z: int, x: int, y: int, extension: str) -> Path | None:
    candidate = (_MAP_DIR / tileset / str(z) / str(x) / f"{y}.{extension.lower()}").resolve()
    if not _is_relative_to(candidate, _MAP_DIR.resolve()):
        return None
    return candidate if candidate.is_file() else None


@lru_cache(maxsize=64)
def _tileset_geotiff_path(tileset: str) -> str | None:
    tileset_dir = (_MAP_DIR / tileset).resolve()
    if not _is_relative_to(tileset_dir, _MAP_DIR.resolve()) or not tileset_dir.is_dir():
        return None

    candidates = sorted(
        path
        for pattern in ("*.tif", "*.tiff")
        for path in tileset_dir.rglob(pattern)
        if path.is_file() and not path.name.lower().endswith(".aux.xml")
    )
    if not candidates:
        return None

    return str(candidates[0])


@lru_cache(maxsize=1)
def _transparent_png() -> bytes:
    import cv2
    import numpy as np

    image = np.zeros((_TILE_SIZE, _TILE_SIZE, 4), dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", image)
    if not ok:
        raise RuntimeError("Could not encode transparent map tile")
    return encoded.tobytes()


def _encode_png_rgba(rgba_image) -> bytes:
    import cv2

    bgra_image = cv2.cvtColor(rgba_image, cv2.COLOR_RGBA2BGRA)
    ok, encoded = cv2.imencode(".png", bgra_image)
    if not ok:
        raise RuntimeError("Could not encode GeoTIFF map tile")
    return encoded.tobytes()


@lru_cache(maxsize=512)
def _render_geotiff_tile(path: str, mtime_ns: int, z: int, x: int, y: int) -> bytes:
    import numpy as np
    import rasterio
    from rasterio.enums import Resampling
    from rasterio.vrt import WarpedVRT
    from rasterio.windows import from_bounds

    del mtime_ns  # Cache key only; the path read below already points to the file.

    left, bottom, right, top = _tile_bounds_3857(z, x, y)
    with rasterio.open(path) as source:
        with WarpedVRT(source, crs="EPSG:3857", resampling=Resampling.lanczos) as vrt:
            vrt_left, vrt_bottom, vrt_right, vrt_top = vrt.bounds
            if right <= vrt_left or left >= vrt_right or top <= vrt_bottom or bottom >= vrt_top:
                return _transparent_png()

            indexes = [1, 2, 3] if vrt.count >= 3 else [1]
            clipped_left = max(left, vrt_left)
            clipped_bottom = max(bottom, vrt_bottom)
            clipped_right = min(right, vrt_right)
            clipped_top = min(top, vrt_top)

            span_x = right - left
            span_y = top - bottom
            col_start = max(0, int(np.floor(((clipped_left - left) / span_x) * _TILE_SIZE)))
            col_end = min(_TILE_SIZE, int(np.ceil(((clipped_right - left) / span_x) * _TILE_SIZE)))
            row_start = max(0, int(np.floor(((top - clipped_top) / span_y) * _TILE_SIZE)))
            row_end = min(_TILE_SIZE, int(np.ceil(((top - clipped_bottom) / span_y) * _TILE_SIZE)))
            if col_end <= col_start or row_end <= row_start:
                return _transparent_png()

            window = from_bounds(
                clipped_left,
                clipped_bottom,
                clipped_right,
                clipped_top,
                transform=vrt.transform,
            )
            clipped_data = vrt.read(
                indexes,
                window=window,
                out_shape=(len(indexes), row_end - row_start, col_end - col_start),
                masked=True,
                resampling=Resampling.lanczos,
            )

    data = np.ma.masked_array(
        np.zeros((len(indexes), _TILE_SIZE, _TILE_SIZE), dtype=clipped_data.dtype),
        mask=np.ones((len(indexes), _TILE_SIZE, _TILE_SIZE), dtype=bool),
    )
    data[:, row_start:row_end, col_start:col_end] = clipped_data

    mask = np.ma.getmaskarray(data)
    filled = np.ma.filled(data, 0)
    if len(indexes) == 1:
        filled = np.repeat(filled, 3, axis=0)
        mask = np.repeat(mask, 3, axis=0)

    rgb = np.moveaxis(filled[:3], 0, -1)
    if rgb.dtype != np.uint8:
        rgb = np.clip(rgb, 0, 255).astype(np.uint8)

    pixel_mask = np.any(np.moveaxis(mask[:3], 0, -1), axis=2)
    alpha = np.where(pixel_mask, 0, 255).astype(np.uint8)
    alpha[np.sum(rgb, axis=2) <= 3] = 0
    if not np.any(alpha):
        return _transparent_png()

    rgba = np.dstack((rgb, alpha))
    return _encode_png_rgba(rgba)


@router.get("/tiles/{tileset}/{z}/{x}/{y}.{extension}", include_in_schema=False)
def get_map_tile(tileset: str, z: int, x: int, y: int, extension: str):
    extension = extension.lower().lstrip(".")
    if extension not in {"png", "jpg", "jpeg", "webp"}:
        raise HTTPException(status_code=404, detail="Unsupported tile format")
    if z < 0 or z > _MAX_ZOOM or x < 0 or y < 0 or x >= 2**z or y >= 2**z:
        raise HTTPException(status_code=404, detail="Tile out of range")

    static_tile = _static_tile_path(tileset, z, x, y, extension)
    if static_tile is not None:
        return FileResponse(static_tile, media_type=_media_type(extension))

    geotiff_path = _tileset_geotiff_path(tileset)
    if geotiff_path is None or extension != "png":
        raise HTTPException(status_code=404, detail="Tile not found")

    path = Path(geotiff_path)
    try:
        tile = _render_geotiff_tile(str(path), path.stat().st_mtime_ns, z, x, y)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Could not render map tile: {exc}") from exc

    return Response(content=tile, media_type="image/png")
