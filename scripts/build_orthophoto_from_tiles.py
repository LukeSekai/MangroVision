"""Build a georeferenced GeoTIFF from the highest-resolution XYZ tile level."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import rasterio
from PIL import Image
from rasterio.enums import ColorInterp
from rasterio.transform import from_origin
from rasterio.windows import Window


WEB_MERCATOR_RADIUS_M = 6_378_137.0
TILE_SIZE = 256


def _numeric_directories(root: Path) -> list[int]:
    return sorted(int(path.name) for path in root.iterdir() if path.is_dir() and path.name.isdigit())


def build_geotiff(tileset: Path, output: Path, zoom: int | None = None) -> None:
    zoom_levels = _numeric_directories(tileset)
    if not zoom_levels:
        raise ValueError(f"No numeric zoom directories found in {tileset}")

    selected_zoom = max(zoom_levels) if zoom is None else zoom
    zoom_root = tileset / str(selected_zoom)
    if not zoom_root.is_dir():
        raise ValueError(f"Zoom directory does not exist: {zoom_root}")

    x_values = _numeric_directories(zoom_root)
    if not x_values:
        raise ValueError(f"No numeric x directories found in {zoom_root}")

    tile_paths: dict[tuple[int, int], Path] = {}
    y_values: list[int] = []
    for x in x_values:
        for tile_path in (zoom_root / str(x)).glob("*.png"):
            if not tile_path.stem.isdigit():
                continue
            y = int(tile_path.stem)
            tile_paths[(x, y)] = tile_path
            y_values.append(y)
    if not y_values:
        raise ValueError(f"No PNG tiles found in {zoom_root}")

    x_min, x_max = min(x_values), max(x_values)
    y_min, y_max = min(y_values), max(y_values)
    width = (x_max - x_min + 1) * TILE_SIZE
    height = (y_max - y_min + 1) * TILE_SIZE

    world_span_m = 2.0 * math.pi * WEB_MERCATOR_RADIUS_M
    resolution = world_span_m / (TILE_SIZE * (2**selected_zoom))
    west = -math.pi * WEB_MERCATOR_RADIUS_M + x_min * TILE_SIZE * resolution
    north = math.pi * WEB_MERCATOR_RADIUS_M - y_min * TILE_SIZE * resolution

    output.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        output,
        "w",
        driver="GTiff",
        width=width,
        height=height,
        count=4,
        dtype="uint8",
        crs="EPSG:3857",
        transform=from_origin(west, north, resolution, resolution),
        tiled=True,
        blockxsize=TILE_SIZE,
        blockysize=TILE_SIZE,
        compress="deflate",
        predictor=2,
        interleave="pixel",
        BIGTIFF="IF_SAFER",
    ) as dataset:
        dataset.colorinterp = (
            ColorInterp.red,
            ColorInterp.green,
            ColorInterp.blue,
            ColorInterp.alpha,
        )
        dataset.update_tags(
            source_tileset=tileset.name,
            source_zoom=str(selected_zoom),
            xyz_tile_bounds=f"{x_min},{y_min},{x_max},{y_max}",
        )

        for (x, y), tile_path in sorted(tile_paths.items()):
            with Image.open(tile_path) as image:
                rgba = np.asarray(image.convert("RGBA"), dtype=np.uint8)
            if rgba.shape != (TILE_SIZE, TILE_SIZE, 4):
                raise ValueError(f"Unexpected tile dimensions for {tile_path}: {rgba.shape}")
            window = Window(
                (x - x_min) * TILE_SIZE,
                (y - y_min) * TILE_SIZE,
                TILE_SIZE,
                TILE_SIZE,
            )
            dataset.write(np.moveaxis(rgba, -1, 0), window=window)

    print(
        f"Built {output} from {len(tile_paths)} tiles at z{selected_zoom} "
        f"({width}x{height}px, {resolution:.6f} m/px)."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("tileset", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--zoom", type=int)
    args = parser.parse_args()
    build_geotiff(args.tileset, args.output, args.zoom)


if __name__ == "__main__":
    main()
