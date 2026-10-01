"""Replace authoritative GIS coverage zones from GeoJSON or active map bounds."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mangrovision_db.zones import replace_zone_collection


def from_geojson(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("type") == "FeatureCollection":
        return list(payload.get("features") or [])
    if payload.get("type") == "Feature":
        return [payload]
    if payload.get("type") in {"Polygon", "MultiPolygon"}:
        return [{"type": "Feature", "properties": {}, "geometry": payload}]
    raise ValueError("GIS coverage input must contain Polygon/MultiPolygon geometry")


def from_active_map() -> list[dict]:
    from canopy_detection import ortho_matcher

    bounds = ortho_matcher._active_tileset_bounds_latlon()
    if bounds is None:
        raise RuntimeError("Could not resolve active tile or orthophoto bounds")
    south, west, north, east = [float(value) for value in bounds]
    return [{
        "type": "Feature",
        "properties": {
            "name": "Active MangroVision GIS coverage",
            "source": "active-map-bounds",
        },
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [west, south], [east, south], [east, north], [west, north], [west, south]
            ]],
        },
    }]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--geojson", type=Path)
    args = parser.parse_args()
    try:
        features = from_geojson(args.geojson) if args.geojson else from_active_map()
        result = replace_zone_collection("gis_coverage", features)
    except Exception as error:
        print(f"GIS coverage sync failed: {error}", file=sys.stderr)
        return 1
    print(f"Saved {len(result['features'])} authoritative GIS coverage zone(s).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
