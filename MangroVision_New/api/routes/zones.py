"""Zone data endpoints — forbidden and eroded zones (GeoJSON)."""

import json
from pathlib import Path
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel
from typing import Any

router = APIRouter()

_ROOT = Path(__file__).resolve().parent.parent.parent.parent
_FORBIDDEN_CANDIDATES = (
    _ROOT / "forbidden_zone_final.geojson",
    _ROOT / "forbidden_zones_final.geojson",
    _ROOT / "forbidden_zones.geojson",
)
_FORBIDDEN_PATH = next((path for path in _FORBIDDEN_CANDIDATES if path.exists()), _FORBIDDEN_CANDIDATES[0])
_ERODED_PATH = _ROOT / "eroded_zones.geojson"


def _load_geojson(path: Path) -> dict:
    if not path.exists():
        return {"type": "FeatureCollection", "features": []}
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


@router.get("/forbidden")
def get_forbidden_zones():
    return _load_geojson(_FORBIDDEN_PATH)


@router.get("/eroded")
def get_eroded_zones():
    return _load_geojson(_ERODED_PATH)


class SaveErodedBody(BaseModel):
    features: list[Any]


@router.put("/eroded")
def save_eroded_zones(body: SaveErodedBody):
    geojson = {
        "type": "FeatureCollection",
        "name": "eroded_zones",
        "features": body.features,
    }
    with open(_ERODED_PATH, "w", encoding="utf-8") as f:
        json.dump(geojson, f, indent=2)
    return {"status": "saved", "count": len(body.features)}


@router.delete("/eroded/{index}")
def delete_eroded_zone(index: int):
    data = _load_geojson(_ERODED_PATH)
    features = data.get("features", [])
    if index < 0 or index >= len(features):
        raise HTTPException(status_code=404, detail="Zone index out of range")
    features.pop(index)
    data["features"] = features
    with open(_ERODED_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return {"status": "deleted", "remaining": len(features)}


@router.post("/eroded")
def add_eroded_zone(feature: dict):
    """Append a single GeoJSON Feature to the eroded zones file."""
    data = _load_geojson(_ERODED_PATH)
    features = data.get("features", [])
    features.append(feature)
    data["features"] = features
    with open(_ERODED_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)
    return {"status": "added", "total": len(features)}
