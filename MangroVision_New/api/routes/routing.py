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
from dotenv import dotenv_values

from api.site_access import access_for_destination, distance_m, inside_area, inside_or_near_area, path_distance, remaining_access_path
from api.routing_usage import reserve_google_request

router = APIRouter()

_SECRETS_PATH = Path(__file__).resolve().parents[3] / ".streamlit" / "secrets.toml"
_ENV_PATH = Path(__file__).resolve().parents[3] / '.env'


def _load_google_routes_api_key() -> str:
    env_value = (os.getenv("GOOGLE_ROUTES_API_KEY") or "").strip()
    if env_value:
        return env_value
    # A newly entered laptop key can be picked up without restarting the API.
    local_value = (dotenv_values(_ENV_PATH).get('GOOGLE_ROUTES_API_KEY') or '').strip()
    if local_value:
        return local_value
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
    origin_accuracy_m: Optional[float] = Field(default=None, ge=0, allow_inf_nan=False)


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

    reserve_google_request(api_key)
    try:
        with urlopen(http_request, timeout=20) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as err:
        raise HTTPException(status_code=502, detail=f"Walking directions are unavailable (provider status {err.code}). Please retry.") from err
    except (URLError, TimeoutError) as err:
        reason = err.reason if isinstance(err, URLError) else 'timed out'
        raise HTTPException(status_code=502, detail=f"Google Routes API connection error: {reason}") from err

    candidates = []
    for route in payload.get('routes') or []:
        encoded = ((route.get('polyline') or {}).get('encodedPolyline') or '').strip()
        try:
            polyline = _decode_polyline(encoded)
        except (IndexError, ValueError):
            continue
        if len(polyline) < 2:
            continue
        # The access destination must still join the correct road, rather than
        # an eastern pond detour. The origin can be off-road (e.g. at home);
        # its marker stays separate from the mapped road geometry.
        if distance_m(polyline[-1], destination) > 20:
            continue
        meters = route.get('distanceMeters')
        if not isinstance(meters, (int, float)) or meters < 0:
            continue
        candidates.append((meters, _parse_duration_seconds(route.get('duration')), polyline))
    if not candidates:
        raise HTTPException(status_code=404, detail="No route found between those points.")
    return min(candidates, key=lambda item: (item[0], item[1] if item[1] is not None else float('inf')))


def _navigation_to_point(origin, destination, mode, road_path, source, site=None,
                         meters=None, seconds=None, note=None, road_sections=None):
    """Follow mapped roads, then show local guidance to the exact point.

    Remote GPS-to-road gaps remain separate. The final dashed guide starts at
    the known site entrance, or at the current GPS fix when already onsite.
    """
    segments = [{'kind': 'road', 'polyline': section}
                for section in (road_sections if road_sections is not None else [road_path])
                if len(section) >= 2]
    guide_start = origin if source == 'within_site' else site['access_path'][-1] if site else None
    if guide_start is None and road_path and distance_m(road_path[-1], destination) <= 20:
        guide_start = road_path[-1]
    if guide_start is not None and guide_start != destination:
        segments.append({'kind': 'guidance', 'polyline': [guide_start, destination]})
    navigation_path = [coordinate for segment in segments for coordinate in segment['polyline']]
    return {
        'origin': origin, 'target': destination,
        'polyline': road_path, 'segments': segments, 'navigation_path': navigation_path,
        'entrance': site['access_path'][-1] if site else None,
        'access_start': site['access_path'][0] if site else None,
        'site_area': site['site_area'] if site else None,
        # These quantities describe the mapped road leg, not an ETA along
        # unknown internal paths or direct GPS connections.
        'distance_m': meters, 'duration_s': seconds,
        'distance_label': _format_distance(meters), 'duration_label': _format_duration(seconds),
        'travel_mode': mode, 'route_source': source,
        'road_route_available': source not in {'partial_route', 'point_guidance', 'within_site'},
        'navigation_note': note or 'Follow the blue road route, then the orange dashed guide to your point using marked planting lanes.',
    }


def _unavailable_route_guidance(origin, destination, mode, site=None):
    """Keep the known access road and local point guide during provider outages."""
    result = _navigation_to_point(origin, destination, mode,
        site['access_path'] if site else [], 'partial_route' if site else 'point_guidance', site)
    result.update({
        'distance_label': 'Road directions unavailable', 'duration_label': None,
        'navigation_note': 'Road directions from your location are unavailable. Open Google Maps for road directions. '
            + ('The access road and orange dashed guide to your point are shown; follow marked planting lanes.'
               if site else 'No unmapped shortcut is drawn.'),
    })
    return result


@router.post('/compute')
def compute_route(body: RouteRequest):
    origin, destination = [body.origin_lat, body.origin_lon], [body.dest_lat, body.dest_lon]
    mode = (body.travel_mode or 'walking').lower()
    site = access_for_destination(destination)
    note = None
    entrance = None
    road_sections = None
    if site:
        if mode != 'walking':
            raise HTTPException(status_code=422, detail='The mangrove access road is configured for walking navigation.')
        path = site['access_path']
        entrance = path[-1]
        # GPS uncertainty may straddle the shoreline/site outline. Cap the
        # tolerance so a poor fix cannot classify a distant position as onsite.
        near_site = inside_or_near_area(origin, site['site_area'], max(3, min(body.origin_accuracy_m or 0, 20)))
        if near_site:
            note = (
                'You are near the site boundary; GPS may drift. '
                if not inside_area(origin, site['site_area']) else 'You are inside the site. '
            ) + 'The orange dashed line shows the direction to your point; follow marked planting lanes.'
            result = _navigation_to_point(origin, destination, mode, [], 'within_site', site, note=note)
            result['distance_label'] = f'{_format_distance(distance_m(origin, destination))} to point (straight-line)'
            result['duration_label'] = 'Follow planting order'
            return result
        local_path = remaining_access_path(origin, path)
        if local_path is not None:
            # remaining_access_path includes the actual GPS fix followed by
            # its projection on the road. Draw only the road from that projection.
            polyline = local_path[1:]
            meters = path_distance(polyline)
            seconds = meters / 1.2
        else:
            try:
                meters, seconds, polyline = _google_route(origin, path[0], mode)
            except HTTPException as error:
                if error.status_code not in (404, 502, 503):
                    raise
                return _unavailable_route_guidance(origin, destination, mode, site)
            access_meters = path_distance(path)
            # A provider endpoint can be up to 20 m from the traced road start;
            # keep these sections separate so the gap is never drawn as a road.
            road_sections = [polyline, path]
            polyline = [*polyline, *path]
            meters += access_meters
            seconds = seconds + access_meters/1.2 if seconds is not None else None
        note = 'Follow the blue road route to the entrance, then the orange dashed guide to your point using marked planting lanes.'
    else:
        try:
            meters, seconds, polyline = _google_route(origin, destination, mode)
        except HTTPException as error:
            if error.status_code not in (404, 502, 503):
                raise
            return _unavailable_route_guidance(origin, destination, mode)
    return _navigation_to_point(origin, destination, mode, polyline,
                                'site_route' if site else 'google_routes', site,
                                meters, seconds, note, road_sections)
