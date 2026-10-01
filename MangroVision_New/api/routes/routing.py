"""Routing endpoints — wraps Google Routes API for in-app navigation."""

import json
import os
import tomllib
from pathlib import Path
from typing import Optional
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.site_access import access_for_destination, distance_m, inside_area, path_distance, remaining_access_path

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
    origin_lat: float = Field(ge=-90, le=90)
    origin_lon: float = Field(ge=-180, le=180)
    dest_lat: float = Field(ge=-90, le=90)
    dest_lon: float = Field(ge=-180, le=180)
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


def _google_route(origin, destination, travel_mode):
    """Choose the shortest valid walking route returned by the provider."""
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
                    "latitude": origin[0],
                    "longitude": origin[1],
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": destination[0],
                    "longitude": destination[1],
                }
            }
        },
        "travelMode": _travel_mode_to_google(travel_mode),
        "computeAlternativeRoutes": True,
        "polylineQuality": "HIGH_QUALITY",
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
        raise HTTPException(status_code=502, detail=f"Walking directions are unavailable (provider status {err.code}). Please retry.") from err
    except URLError as err:
        raise HTTPException(status_code=502, detail=f"Google Routes API connection error: {err.reason}") from err

    candidates = []
    for route in payload.get('routes') or []:
        encoded = ((route.get('polyline') or {}).get('encodedPolyline') or '').strip()
        try:
            polyline = _decode_polyline(encoded)
        except (IndexError, ValueError):
            continue
        if len(polyline) < 2:
            continue
        # Reject snapping to a different road instead of drawing a bridge over
        # the unmapped gap. In particular this prevents the eastern pond detour.
        if distance_m(polyline[-1], destination) > 20 or distance_m(polyline[0], origin) > 30:
            continue
        meters = route.get('distanceMeters')
        if not isinstance(meters, (int, float)) or meters < 0:
            continue
        candidates.append((meters, _parse_duration_seconds(route.get('duration')), polyline))
    if not candidates:
        raise HTTPException(status_code=404, detail="No route found between those points.")
    return min(candidates, key=lambda item: (item[0], item[1] if item[1] is not None else float('inf')))


@router.post('/compute')
def compute_route(body: RouteRequest):
    origin, destination = [body.origin_lat, body.origin_lon], [body.dest_lat, body.dest_lon]
    mode = (body.travel_mode or 'walking').lower()
    site = access_for_destination(destination)
    note = None
    entrance = None
    if site:
        if mode != 'walking':
            raise HTTPException(status_code=422, detail='The mangrove access road is configured for walking navigation.')
        path = site['access_path']
        entrance = path[-1]
        if inside_area(origin, site['site_area']):
            # Do not send someone already planting back out onto public roads,
            # or draw an unverified straight route through mangroves/water.
            return {
                'polyline': [], 'target': destination, 'entrance': entrance,
                'distance_m': None, 'duration_s': None,
                'distance_label': f'{_format_distance(distance_m(origin, destination))} to point (straight-line)',
                'duration_label': 'Follow planting order', 'travel_mode': mode,
                'route_source': 'within_site',
                'navigation_note': 'You are inside the site. Follow the LGU-marked planting lanes in zigzag order; internal walking paths are not mapped.',
            }
        local_path = remaining_access_path(origin, path)
        if local_path is not None:
            polyline = local_path
            meters = path_distance(polyline)
            seconds = meters / 1.2
        else:
            meters, seconds, polyline = _google_route(origin, path[0], mode)
            access_meters = distance_m(polyline[-1], path[0]) + path_distance(path)
            polyline = [*polyline, *path]
            meters += access_meters
            seconds = seconds + access_meters/1.2 if seconds is not None else None
        note = f'Follow the white road to the entrance. Your point is {_format_distance(distance_m(entrance, destination))} beyond it (straight-line); use the LGU-marked planting lanes inside.'
    else:
        meters, seconds, polyline = _google_route(origin, destination, mode)
    return {
        'polyline': polyline, 'target': destination, 'entrance': entrance,
        'distance_m': meters, 'duration_s': seconds,
        'distance_label': _format_distance(meters), 'duration_label': _format_duration(seconds),
        'travel_mode': mode, 'route_source': 'site_entrance' if site else 'google_routes',
        'navigation_note': note,
    }
