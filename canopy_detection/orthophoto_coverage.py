"""The visible part of the active GeoTIFF, including its irregular edge."""

from functools import lru_cache
from pathlib import Path

import cv2
import numpy as np
import pyproj
import rasterio
from rasterio.enums import Resampling
from rasterio.windows import Window
from shapely.geometry import Polygon
from shapely.geometry import Point, shape
from shapely.ops import transform, unary_union
import math


@lru_cache(maxsize=4)
def visible_coverage_geometry(path: str, modified_ns: int):
    """Approximate visible raster coverage in WGS84 at sub-metre resolution."""
    del modified_ns  # Included in the cache key to refresh replaced orthophotos.
    with rasterio.open(path) as dataset:
        if dataset.crs is None:
            raise ValueError("The active orthophoto has no coordinate system.")
        scale = min(1., 2048 / max(dataset.width, dataset.height))
        width = max(1, round(dataset.width * scale))
        height = max(1, round(dataset.height * scale))
        out_shape = (height, width)
        visible = dataset.read_masks(1, out_shape=out_shape, resampling=Resampling.nearest) > 0
        # Map tiles also make black/no-image pixels transparent.
        bands = [1, 2, 3] if dataset.count >= 3 else [1]
        rgb = dataset.read(bands, out_shape=(len(bands), height, width), resampling=Resampling.nearest)
        visible &= np.sum(rgb.astype(np.uint16), axis=0) > 3
        contours, _ = cv2.findContours(visible.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        pixel_to_crs = dataset.transform * rasterio.Affine.scale(dataset.width / width, dataset.height / height)
        to_gps = pyproj.Transformer.from_crs(dataset.crs, "EPSG:4326", always_xy=True).transform
        polygons = []
        for contour in contours:
            if cv2.contourArea(contour) < 4:
                continue
            contour = cv2.approxPolyDP(contour, .75, True).reshape(-1, 2)
            if len(contour) < 3:
                continue
            polygon = Polygon([pixel_to_crs * (float(x), float(y)) for x, y in contour])
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if not polygon.is_empty:
                polygons.append(transform(to_gps, polygon))
        if not polygons:
            raise ValueError("The active orthophoto has no visible map pixels.")
        return unary_union(polygons)


def active_visible_coverage(ortho_entry):
    path = Path(ortho_entry["path"])
    return visible_coverage_geometry(str(path.resolve()), path.stat().st_mtime_ns)


def visible_gis_coverage(ortho_entry, features):
    """The authoritative GIS zones intersected with actual map imagery."""
    geometries = []
    for feature in features:
        raw = feature.get("geometry") or {}
        if raw.get("type") not in {"Polygon", "MultiPolygon"}:
            continue
        try:
            geometry = shape(raw)
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            if not geometry.is_empty:
                geometries.append(geometry)
        except Exception:
            continue
    if not geometries:
        raise RuntimeError("No valid active GIS coverage polygon is configured.")
    coverage = unary_union(geometries).intersection(active_visible_coverage(ortho_entry))
    if coverage.is_empty:
        raise RuntimeError("The active GIS coverage polygon has no visible imagery.")
    return coverage


def point_visibility_flags(ortho_entry, coverage, lat_lon_points):
    """Verify planting coordinates against GIS zones and exact GeoTIFF pixels."""
    points = list(lat_lon_points)
    flags = [False] * len(points)
    candidates = []
    for index, (latitude, longitude) in enumerate(points):
        try:
            lat, lon = float(latitude), float(longitude)
            if math.isfinite(lat) and math.isfinite(lon) and coverage.covers(Point(lon, lat)):
                candidates.append((index, lon, lat))
        except (TypeError, ValueError):
            continue
    if not candidates:
        return flags
    with rasterio.open(ortho_entry["path"]) as dataset:
        to_raster = pyproj.Transformer.from_crs("EPSG:4326", dataset.crs, always_xy=True)
        positions = [to_raster.transform(lon, lat) for _, lon, lat in candidates]
        bands = [1, 2, 3] if dataset.count >= 3 else [1]
        for (index, _, _), pixel in zip(
            candidates, dataset.sample(positions, indexes=bands, masked=True)
        ):
            flags[index] = bool(
                not np.ma.is_masked(pixel) and np.sum(pixel.astype(np.uint16)) > 3
            )
    return flags


def overlay_visibility_mask(ortho_entry, x: int, y: int, width: int, height: int,
                            out_width: int, out_height: int) -> np.ndarray:
    """Sample the same visibility rule as the map tiles in overlay coordinates."""
    with rasterio.open(ortho_entry["path"]) as dataset:
        window = Window(x, y, width, height)
        shape = (out_height, out_width)
        mask = dataset.read_masks(1, window=window, out_shape=shape,
                                  boundless=True, resampling=Resampling.nearest) > 0
        bands = [1, 2, 3] if dataset.count >= 3 else [1]
        rgb = dataset.read(bands, window=window, out_shape=(len(bands), *shape),
                           boundless=True, fill_value=0, resampling=Resampling.nearest)
        mask &= np.sum(rgb.astype(np.uint16), axis=0) > 3
        return (mask.astype(np.uint8) * 255)
