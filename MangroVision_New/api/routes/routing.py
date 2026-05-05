"""Routing endpoints — wraps Google Routes API for in-app navigation."""

import json
import os
import tomllib
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

router = APIRouter()

_SECRETS_PATH = Path(__file__).resolve().parents[3] / ".streamlit" / "secrets.toml"


def _load_google_routes_api_key() -> str:
    env_value = (os.getenv("GOOGLE_ROUTES_API_KEY") or "").strip()
    if env_value:
        return env_value
    try:
        with _SECRETS_PATH.open("rb") as fh:
            return str(tomllib.load(fh).get("google_routes_api_key") or "").strip()
    except (FileNotFoundError, OSError, tomllib.TOMLDecodeError):
        return ""


class RouteRequest(BaseModel):
    origin_lat: float
    origin_lon: float
    dest_lat: float
    dest_lon: float
    travel_mode: Optional[str] = "walking"


def _travel_mode_to_google(mode: str) -> str:
    normalized = (mode or "walking").strip().lower()
    return {
        "walking": "WALK",
        "driving": "DRIVE",
        "bicycling": "BICYCLE",
        "transit": "TRANSIT",
    }.get(normalized, "WALK")


def _parse_duration_seconds(value: Optional[str]) -> Optional[float]:
    if not value:
        return None
    try:
        return float(str(value).rstrip("s"))
    except ValueError:
        return None


def _format_duration(seconds: Optional[float]) -> str:
    if seconds is None:
        return "—"
    total = int(round(seconds))
    minutes, sec = divmod(total, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m"
    return f"{sec}s"


def _format_distance(meters: Optional[float]) -> str:
    if meters is None:
        return "—"
    if meters >= 1000:
        return f"{meters / 1000:.2f} km"
    return f"{meters:.0f} m"


def _decode_polyline(encoded: str) -> list[list[float]]:
    """Decode a Google encoded polyline into [[lat, lon], ...]."""
    coordinates: list[list[float]] = []
    index = 0
    lat = 0
    lon = 0
    length = len(encoded)

    while index < length:
        result = 0
        shift = 0
        while True:
            byte = ord(encoded[index]) - 63
            index += 1
            result |= (byte & 0x1F) << shift
            shift += 5
            if byte < 0x20:
                break
        lat += ~(result >> 1) if result & 1 else (result >> 1)

        result = 0
        shift = 0
        while True:
            byte = ord(encoded[index]) - 63
            index += 1
            result |= (byte & 0x1F) << shift
            shift += 5
            if byte < 0x20:
                break
        lon += ~(result >> 1) if result & 1 else (result >> 1)

        coordinates.append([lat / 1e5, lon / 1e5])

    return coordinates


@router.post("/compute")
def compute_route(body: RouteRequest):
    """Return polyline + distance + duration for a point-to-point route."""
    api_key = _load_google_routes_api_key()
    if not api_key:
        raise HTTPException(
            status_code=503,
            detail="Routing is not configured. Set GOOGLE_ROUTES_API_KEY on the server.",
        )

    request_body = {
        "origin": {
            "location": {
                "latLng": {
                    "latitude": body.origin_lat,
                    "longitude": body.origin_lon,
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": body.dest_lat,
                    "longitude": body.dest_lon,
                }
            }
        },
        "travelMode": _travel_mode_to_google(body.travel_mode or "walking"),
        "computeAlternativeRoutes": False,
        "languageCode": "en-US",
        "units": "METRIC",
    }

    http_request = Request(
        "https://routes.googleapis.com/directions/v2:computeRoutes",
        data=json.dumps(request_body).encode("utf-8"),
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "routes.distanceMeters,routes.duration,routes.polyline.encodedPolyline",
        },
    )

    try:
        with urlopen(http_request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as err:
        detail = err.read().decode("utf-8", errors="ignore")
        raise HTTPException(status_code=502, detail=f"Google Routes API error ({err.code}): {detail or err.reason}") from err
    except URLError as err:
        raise HTTPException(status_code=502, detail=f"Google Routes API connection error: {err.reason}") from err

    route = (payload.get("routes") or [None])[0]
    if not route:
        raise HTTPException(status_code=404, detail="No route found between those points.")

    encoded = ((route.get("polyline") or {}).get("encodedPolyline") or "").strip()
    if not encoded:
        raise HTTPException(status_code=404, detail="Route returned no polyline.")

    distance_m = route.get("distanceMeters")
    duration_s = _parse_duration_seconds(route.get("duration"))

    return {
        "polyline": _decode_polyline(encoded),
        "distance_m": distance_m,
        "duration_s": duration_s,
        "distance_label": _format_distance(distance_m),
        "duration_label": _format_duration(duration_s),
        "travel_mode": (body.travel_mode or "walking").lower(),
    }
