"""Environmental context endpoint — pulls free Open-Meteo daily archive data
for a site zone's centroid so the UI can show recent temperature, rainfall,
and wind alongside that zone's mortality numbers.

Key design points:
  - **No predictive claims.** This endpoint surfaces raw context only, so the
    dashboard can show "5 days above 35 °C in the last month" next to that
    zone's mortality. Building a real risk model would need ground-truthed
    training data and is explicitly out of scope.
  - **No API key required.** Open-Meteo's free tier is generous and stable
    for low-traffic admin tools (https://open-meteo.com/en/docs/historical-weather-api).
  - **In-memory cache** mirroring the tides.py pattern keeps daily fetches to
    one per zone per day even if the dashboard is reloaded.
  - **Polygon centroid** is computed with shapely (already a dependency) so
    the env data represents the middle of the zone, not a corner.
"""

from __future__ import annotations

import json as _json
import time
import urllib.error
import urllib.parse
import urllib.request
from datetime import date, timedelta
from threading import Lock
from typing import Any, Dict, Optional

from fastapi import APIRouter, HTTPException

from planting_database import get_site_zone

router = APIRouter()

_CACHE_LOCK = Lock()
_CACHE: Dict[str, Dict[str, Any]] = {}
# Daily archive doesn't move within a day — a 6 h TTL keeps the dashboard
# responsive after 5 PM updates without burning Open-Meteo's free quota.
_CACHE_TTL_SECONDS = 6 * 60 * 60

_HEAT_STRESS_C = 35.0           # daily max above this → counted as a heat-stress day
_HIGH_WIND_KMH = 50.0           # daily max wind above this → counted as a high-wind day
_RAIN_DAY_MM = 1.0              # daily precipitation above this → counted as a rainy day
_HIGH_WAVE_M = 1.0              # daily max wave height above this → counted as high-wave day
_SEA_HEAT_STRESS_C = 32.0       # daily max SST above this → counted as a sea-heat-stress day
                                # (mangrove propagules show stress from ~32 °C; bleaching
                                # corals near here go at 30 °C, so 32 is a defensible
                                # tropical-coastal proxy)

_OPEN_METEO_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_OPEN_METEO_MARINE_URL = "https://marine-api.open-meteo.com/v1/marine"


def _cache_key(lat: float, lon: float, days: int) -> str:
    return f"{round(lat, 3)}:{round(lon, 3)}:{days}"


def _polygon_centroid(geometry: dict) -> Optional[tuple]:
    """Compute the (lat, lon) centroid of a GeoJSON Polygon/MultiPolygon."""
    try:
        from shapely.geometry import shape
    except ImportError:
        return None
    try:
        geom = shape(geometry)
        if not geom.is_valid:
            geom = geom.buffer(0)
        c = geom.centroid
        # GeoJSON is (lon, lat) but we return (lat, lon) for consistency with
        # the rest of the app's lat/lon-first conventions.
        return (float(c.y), float(c.x))
    except Exception:
        return None


def _http_get_json(url: str, timeout: int = 10) -> Dict[str, Any]:
    req = urllib.request.Request(url, headers={"User-Agent": "MangroVision/1.0"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        body = resp.read().decode("utf-8")
    return _json.loads(body)


def _fetch_open_meteo_archive(lat: float, lon: float, days: int) -> Dict[str, Any]:
    """Daily land-weather archive: rainfall + wind. Air temp dropped — for
    mangroves, sea surface temperature (from the marine endpoint) is the more
    relevant heat-stress signal."""
    end = date.today() - timedelta(days=2)  # archive lags ~2 days
    start = end - timedelta(days=max(1, days - 1))
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join([
            "precipitation_sum",
            "wind_speed_10m_max",
            "wind_gusts_10m_max",
        ]),
        "timezone": "Asia/Manila",
    }
    return _http_get_json(_OPEN_METEO_ARCHIVE_URL + "?" + urllib.parse.urlencode(params))


def _fetch_open_meteo_marine(lat: float, lon: float, days: int) -> Optional[Dict[str, Any]]:
    """Daily marine series: wave height + sea surface temperature.

    Marine API only resolves to coastal grid cells; inland centroids return
    empty arrays. We swallow upstream errors and return None so the endpoint
    can still serve the land-weather portion.
    """
    end = date.today() - timedelta(days=2)
    start = end - timedelta(days=max(1, days - 1))
    params = {
        "latitude": f"{lat:.4f}",
        "longitude": f"{lon:.4f}",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
        "daily": ",".join([
            "wave_height_max",
            "sea_surface_temperature_max",
        ]),
        "timezone": "Asia/Manila",
    }
    try:
        return _http_get_json(_OPEN_METEO_MARINE_URL + "?" + urllib.parse.urlencode(params))
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError):
        return None
    except Exception:
        return None


def _summarize(
    archive_daily: Dict[str, Any],
    marine_daily: Optional[Dict[str, Any]],
) -> Dict[str, Any]:
    """Reduce daily series to the handful of numbers the dashboard cares about.

    Land vs marine variables come from different API endpoints; marine returns
    None when the centroid is inland. We expose `available: false` on the
    sea cards in that case so the UI can render an honest empty state.
    """
    times = archive_daily.get("time") or []
    rain = archive_daily.get("precipitation_sum") or []
    wind = archive_daily.get("wind_speed_10m_max") or []
    gust = archive_daily.get("wind_gusts_10m_max") or []

    def _safe(values, fn):
        nums = [v for v in values if isinstance(v, (int, float))]
        return fn(nums) if nums else None

    rain_days = sum(1 for v in rain if isinstance(v, (int, float)) and v >= _RAIN_DAY_MM)
    high_wind_days = sum(1 for v in wind if isinstance(v, (int, float)) and v >= _HIGH_WIND_KMH)

    summary: Dict[str, Any] = {
        "days_in_window": len(times),
        "rainfall_mm": {
            "total": _safe(rain, sum),
            "max_daily": _safe(rain, max),
            "rainy_days": rain_days,
            "rainy_day_threshold_mm": _RAIN_DAY_MM,
        },
        "wind_kmh": {
            "max_sustained": _safe(wind, max),
            "max_gust": _safe(gust, max),
            "high_wind_days": high_wind_days,
            "high_wind_threshold": _HIGH_WIND_KMH,
        },
    }

    # Marine (wave + sea surface temperature) — coastal grid only.
    if marine_daily and any(
        marine_daily.get(k) for k in ("wave_height_max", "sea_surface_temperature_max")
    ):
        wave = marine_daily.get("wave_height_max") or []
        sst = marine_daily.get("sea_surface_temperature_max") or []
        wave_days = sum(1 for v in wave if isinstance(v, (int, float)) and v >= _HIGH_WAVE_M)
        sst_heat_days = sum(
            1 for v in sst if isinstance(v, (int, float)) and v >= _SEA_HEAT_STRESS_C
        )
        summary["wave_height_m"] = {
            "available": True,
            "max_daily": _safe(wave, max),
            "avg_daily": _safe(wave, lambda xs: sum(xs) / len(xs)),
            "high_wave_days": wave_days,
            "high_wave_threshold": _HIGH_WAVE_M,
        }
        summary["sea_temperature_c"] = {
            "available": True,
            "max_daily": _safe(sst, max),
            "avg_daily": _safe(sst, lambda xs: sum(xs) / len(xs)),
            "heat_stress_days": sst_heat_days,
            "heat_stress_threshold": _SEA_HEAT_STRESS_C,
        }
    else:
        summary["wave_height_m"] = {"available": False}
        summary["sea_temperature_c"] = {"available": False}

    return summary


@router.get("/sites/{zone_id}/env-context")
def site_zone_env_context(zone_id: int, days: int = 30):
    """Return last `days` of weather context for a site zone's centroid.

    Response is intentionally descriptive ("5 heat-stress days") not
    predictive — see the module docstring for why.
    """
    if days < 1 or days > 90:
        raise HTTPException(status_code=400, detail="days must be between 1 and 90")

    zone = get_site_zone(zone_id)
    if zone is None:
        raise HTTPException(status_code=404, detail="Site zone not found")

    centroid = _polygon_centroid(zone.get("geometry") or {})
    if centroid is None:
        raise HTTPException(status_code=400, detail="Zone polygon could not be parsed")

    lat, lon = centroid
    cache_key = _cache_key(lat, lon, days)
    now = time.time()
    with _CACHE_LOCK:
        hit = _CACHE.get(cache_key)
        if hit and (now - hit["fetched_at"]) < _CACHE_TTL_SECONDS:
            return hit["payload"]

    try:
        archive_raw = _fetch_open_meteo_archive(lat, lon, days)
    except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError) as error:
        return {
            "configured": False,
            "zone_id": zone_id,
            "centroid": {"lat": lat, "lon": lon},
            "days": days,
            "message": f"Could not reach the Open-Meteo archive: {error}",
        }
    except Exception as error:  # last-resort guard
        return {
            "configured": False,
            "zone_id": zone_id,
            "centroid": {"lat": lat, "lon": lon},
            "days": days,
            "message": f"Unexpected error fetching environmental context: {error}",
        }

    # Marine fetch is best-effort; failure (or inland centroid) just means the
    # sea-temperature and wave cards render as "not available for this zone".
    marine_raw = _fetch_open_meteo_marine(lat, lon, days)

    summary = _summarize(
        archive_raw.get("daily") or {},
        (marine_raw or {}).get("daily") or {},
    )
    payload = {
        "configured": True,
        "zone_id": zone_id,
        "zone_name": (zone.get("properties") or {}).get("name"),
        "centroid": {"lat": lat, "lon": lon},
        "days": days,
        "source": "Open-Meteo Historical Weather + Marine APIs",
        "summary": summary,
    }

    with _CACHE_LOCK:
        _CACHE[cache_key] = {"fetched_at": now, "payload": payload}
    return payload
