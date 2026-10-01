"""Tide forecast proxy with resilient, last-known-good caching.

WorldTides is preferred when ``WORLDTIDES_API_KEY`` is configured. Otherwise,
the route uses Open-Meteo's keyless marine forecast and derives high/low
turning points from its hourly sea-level series. A successful response is
cached in memory and on disk so a short provider or network interruption does
not immediately remove tide information from the scheduling screen. No tide
values are fabricated by this module.
"""

from __future__ import annotations

import copy
import hashlib
import json
import logging
import math
import os
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from threading import Lock
from typing import Any, Dict, Optional

from fastapi import APIRouter, Query

router = APIRouter()
logger = logging.getLogger(__name__)

# Match the admin map's default center so tides line up with the visualized site.
_DEFAULT_LAT = 10.7800
_DEFAULT_LON = 122.6253
_WORLDTIDES_URL = "https://www.worldtides.info/api/v3"
_OPEN_METEO_URL = "https://marine-api.open-meteo.com/v1/marine"
_OPEN_METEO_MAX_FORECAST_DAYS = 8

_CACHE_LOCK = Lock()
_CACHE: Dict[str, Dict[str, Any]] = {}
# Fresh data is reused for six hours; a real response may be served for up to
# another day if the selected provider is temporarily unreachable.
_CACHE_TTL_SECONDS = 6 * 60 * 60
_STALE_CACHE_TTL_SECONDS = 24 * 60 * 60
_CACHE_PATH = Path(
    os.getenv(
        "MANGROVISION_TIDE_CACHE_PATH",
        str(Path(__file__).resolve().parents[2] / "run_logs" / "worldtides_forecast_cache.json"),
    )
)


def _reject_nonfinite_json(value: str) -> None:
    raise ValueError(f"Nonfinite JSON number: {value}")


def _cache_key(lat: float, lon: float, days: int) -> str:
    # Invalidate pre-series caches; use the same precision as provider requests.
    return f"series-v2:{lat:.4f}:{lon:.4f}:{days}"


def _load_disk_cache_unlocked() -> None:
    """Merge the small persistent cache into memory, ignoring corrupt files."""

    try:
        raw = json.loads(_CACHE_PATH.read_text(encoding="utf-8"), parse_constant=_reject_nonfinite_json)
    except (FileNotFoundError, OSError, ValueError, TypeError):
        return
    entries = raw.get("entries") if isinstance(raw, dict) else None
    if not isinstance(entries, dict):
        return
    for key, entry in entries.items():
        if not isinstance(entry, dict) or not isinstance(entry.get("payload"), dict):
            continue
        try:
            fetched_at = float(entry["fetched_at"])
        except (KeyError, TypeError, ValueError):
            continue
        current = _CACHE.get(str(key))
        if not math.isfinite(fetched_at) or fetched_at > time.time() + 60:
            continue
        if not current or fetched_at > float(current.get("fetched_at", 0)):
            _CACHE[str(key)] = {"fetched_at": fetched_at, "payload": entry["payload"]}


def _persist_cache_unlocked() -> None:
    """Best-effort atomic cache write; forecast delivery must not depend on it."""

    try:
        _CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
        temporary = _CACHE_PATH.with_suffix(_CACHE_PATH.suffix + ".tmp")
        temporary.write_text(
            json.dumps({"version": 1, "entries": _CACHE}, separators=(",", ":")),
            encoding="utf-8",
        )
        os.replace(temporary, _CACHE_PATH)
    except OSError:
        # The in-memory cache still protects the current process if the runtime
        # directory is read-only.
        return


def _read_cache(key: str, *, allow_stale: bool = False) -> Optional[Dict[str, Any]]:
    with _CACHE_LOCK:
        if key not in _CACHE:
            _load_disk_cache_unlocked()
        entry = _CACHE.get(key)
        if not entry:
            return None
        fetched_at = _finite(entry.get("fetched_at"))
        if fetched_at is None or fetched_at > time.time() + 60:
            return None
        age_seconds = max(0.0, time.time() - fetched_at)
        max_age = _STALE_CACHE_TTL_SECONDS if allow_stale else _CACHE_TTL_SECONDS
        if age_seconds > max_age:
            return None
        payload = copy.deepcopy(entry["payload"])

    if age_seconds > _CACHE_TTL_SECONDS:
        payload["stale"] = True
        payload["available"] = True
        source = str(payload.get("source") or "The tide provider")
        payload["message"] = (
            f"{source} is temporarily unreachable. Showing the last successful "
            "forecast; verify the fetched time before scheduling fieldwork."
        )
    return payload


def _write_cache(key: str, payload: Dict[str, Any]) -> None:
    with _CACHE_LOCK:
        _CACHE[key] = {"fetched_at": time.time(), "payload": copy.deepcopy(payload)}
        _persist_cache_unlocked()


def _request_worldtides(url: str, *, timeout: int = 15) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "MangroVision/2.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    data = json.loads(raw, parse_constant=_reject_nonfinite_json)
    if not isinstance(data, dict):
        raise ValueError("WorldTides returned an unexpected response")
    return data


def _request_open_meteo(url: str, *, timeout: int = 15) -> Dict[str, Any]:
    request = urllib.request.Request(
        url,
        headers={"Accept": "application/json", "User-Agent": "MangroVision/2.0"},
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        raw = response.read().decode("utf-8", errors="replace")
    data = json.loads(raw, parse_constant=_reject_nonfinite_json)
    if not isinstance(data, dict):
        raise ValueError("Open-Meteo returned an unexpected response")
    return data


def _finite(value: Any) -> Optional[float]:
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        number = float(value)
    except (ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _level_series(timestamps: Any, heights: Any) -> list[Dict[str, Any]]:
    if not isinstance(timestamps, list) or not isinstance(heights, list):
        raise ValueError("Missing water-level arrays")
    if len(timestamps) != len(heights):
        raise ValueError("Mismatched water-level arrays")
    rows = []
    previous = -1
    for raw_time, raw_height in zip(timestamps, heights):
        timestamp = _finite(raw_time)
        if timestamp is None or not 0 <= timestamp <= 253402300799 or timestamp <= previous:
            raise ValueError("Invalid or unordered water-level timestamps")
        previous = timestamp
        rows.append({"timestamp": timestamp, "height_m": _finite(raw_height)})
    return rows


def _series_metadata(heights: list, *, datum: Optional[str], reference: Any) -> dict:
    valid = [row for row in heights if row["height_m"] is not None]
    fingerprint = hashlib.sha256(json.dumps(reference, sort_keys=True).encode()).hexdigest()
    return {
        "heights": heights,
        "datum": datum or "Unknown",
        "datum_reference": fingerprint if datum == "MSL" else None,
        "coverage_start": valid[0]["timestamp"] if valid else None,
        "coverage_end": valid[-1]["timestamp"] if valid else None,
        "series_available": len(valid) >= 2,
        "fresh_for_seconds": _CACHE_TTL_SECONDS,
        "verification_url": "https://namria.gov.ph/kiosk/namria02.htm",
    }


def _open_meteo_extremes(data: Dict[str, Any]) -> list[Dict[str, Any]]:
    """Convert hourly sea levels into high/low turning points.

    Open-Meteo values are rounded, so flat peaks and troughs are represented
    as runs. Selecting the midpoint of each run avoids duplicate tide events.
    """

    hourly = data.get("hourly")
    if not isinstance(hourly, dict):
        raise ValueError("Open-Meteo did not return hourly sea-level data")
    timestamps = hourly.get("time")
    heights = hourly.get("sea_level_height_msl")
    if not isinstance(timestamps, list) or not isinstance(heights, list):
        raise ValueError("Open-Meteo did not return hourly sea-level data")
    if len(timestamps) != len(heights) or len(timestamps) < 2:
        raise ValueError("Open-Meteo returned incomplete sea-level data")

    # A missing sample separates segments so an outage cannot create an
    # artificial turning point across the gap.
    segments: list[list[tuple[int, float]]] = []
    current_segment: list[tuple[int, float]] = []
    for raw_timestamp, raw_height in zip(timestamps, heights):
        timestamp = _finite(raw_timestamp)
        height = _finite(raw_height)
        if timestamp is None or height is None:
            if current_segment:
                segments.append(current_segment)
                current_segment = []
            continue
        if current_segment and not 0 < timestamp - current_segment[-1][0] <= 3600:
            segments.append(current_segment)
            current_segment = []
        current_segment.append((timestamp, height))
    if current_segment:
        segments.append(current_segment)

    extremes: list[Dict[str, Any]] = []
    for points in segments:
        runs: list[tuple[int, int, float]] = []
        for index, (_, height) in enumerate(points):
            if runs and height == runs[-1][2]:
                start, _, run_height = runs[-1]
                runs[-1] = (start, index, run_height)
            else:
                runs.append((index, index, height))

        for run_index in range(1, len(runs) - 1):
            start, end, height = runs[run_index]
            previous_height = runs[run_index - 1][2]
            next_height = runs[run_index + 1][2]
            if height > previous_height and height > next_height:
                tide_type = "High"
            elif height < previous_height and height < next_height:
                tide_type = "Low"
            else:
                continue
            midpoint = (start + end) // 2
            extremes.append(
                {
                    "type": tide_type,
                    "timestamp": points[midpoint][0],
                    "height_m": height,
                }
            )
    return extremes


def _provider_error_code(error: BaseException) -> str:
    reason = getattr(error, "reason", None)
    if isinstance(reason, PermissionError) or isinstance(error, PermissionError):
        return "network_permission_denied"
    if isinstance(error, urllib.error.HTTPError):
        return "provider_http_error"
    if isinstance(error, (TimeoutError, urllib.error.URLError)):
        return "provider_unreachable"
    return "provider_response_error"


def _unavailable_payload(
    *, lat: float, lon: float, days: int, error_code: str, source: str
) -> Dict[str, Any]:
    return {
        "configured": True,
        "available": False,
        "source": source,
        "message": (
            "The tide forecast service cannot be reached from the MangroVision server right now. "
            "Check the server's internet access and try again."
        ),
        "error_code": error_code,
        "lat": lat,
        "lon": lon,
        "days": days,
        "extremes": [],
        "heights": [],
        "datum": "Unknown",
        "series_available": False,
        "stale": False,
    }


@router.get("/forecast")
def tides_forecast(
    lat: float = Query(_DEFAULT_LAT, ge=-90, le=90),
    lon: float = Query(_DEFAULT_LON, ge=-180, le=180),
    days: int = Query(7, ge=1, le=30),
):
    """Return sampled water levels plus the existing high/low tide events."""

    key = _cache_key(lat, lon, days)
    cached = _read_cache(key)
    provider_mode = "worldtides" if (os.getenv("WORLDTIDES_API_KEY") or "").strip() else "open-meteo"
    if cached is not None and cached.get("provider_mode", provider_mode) == provider_mode:
        return cached

    api_key = (os.getenv("WORLDTIDES_API_KEY") or "").strip()
    if api_key:
        params = urllib.parse.urlencode(
            {
                "extremes": "",
                "heights": "",
                "datum": "MSL",
                "localtime": "",
                "timezone": "",
                "step": 1800,
                "days": days,
                "lat": f"{lat:.4f}",
                "lon": f"{lon:.4f}",
                "key": api_key,
            }
        )
        try:
            data = _request_worldtides(f"{_WORLDTIDES_URL}?{params}")
            raw_status = data.get("status")
            try:
                provider_status = int(raw_status) if raw_status is not None else 200
            except (TypeError, ValueError, OverflowError):
                provider_status = 500
            if provider_status >= 400 or data.get("error"):
                raise ValueError(str(data.get("error") or "WorldTides rejected the request"))
            extremes = [
                {
                    "type": item.get("type"),
                    "date": item.get("date"),
                    "timestamp": _finite(item.get("dt")),
                    "height_m": _finite(item.get("height")),
                }
                for item in (data.get("extremes") or [])
                if isinstance(item, dict) and _finite(item.get("dt")) is not None
                and _finite(item.get("height")) is not None
            ]
            raw_heights = data.get("heights", [])
            if not isinstance(raw_heights, list) or any(not isinstance(row, dict) for row in raw_heights):
                raise ValueError("Malformed WorldTides heights")
            heights = _level_series(
                [row.get("dt") for row in raw_heights], [row.get("height") for row in raw_heights],
            )
            if not extremes and not any(row["height_m"] is not None for row in heights):
                raise ValueError("WorldTides returned no water levels")
            datum = data.get("responseDatum") if isinstance(data.get("responseDatum"), str) else None
            payload = {
                "configured": True,
                "available": True,
                "stale": False,
                "source": "WorldTides",
                "provider_mode": provider_mode,
                "fallback": False,
                "lat": lat,
                "lon": lon,
                "days": days,
                "station": data.get("station") or None,
                "timezone": data.get("timezone") or "UTC",
                "extremes": extremes,
                "fetched_at": int(time.time()),
                "attribution": str(data.get("copyright") or "Tide predictions by WorldTides"),
                "attribution_url": "https://www.worldtides.info",
                **_series_metadata(heights, datum=datum, reference={
                    "provider": "WorldTides", "datum": datum,
                    "station": data.get("station"), "atlas": data.get("atlas"),
                    "lat": data.get("lat", round(lat, 4)), "lon": data.get("lon", round(lon, 4)),
                }),
            }
            _write_cache(key, payload)
            return payload
        except (
            OSError,
            TimeoutError,
            urllib.error.HTTPError,
            urllib.error.URLError,
            json.JSONDecodeError,
            TypeError,
            ValueError,
        ) as error:
            # Try the independent provider before falling back to old readings.
            # Returning a stale WorldTides result here would hide a working
            # Open-Meteo forecast whenever a cache entry already exists.
            logger.warning("WorldTides refresh failed (%s); trying Open-Meteo", _provider_error_code(error))

    forecast_days = min(days, _OPEN_METEO_MAX_FORECAST_DAYS)
    params = urllib.parse.urlencode(
        {
            "latitude": f"{lat:.4f}",
            "longitude": f"{lon:.4f}",
            "hourly": "sea_level_height_msl",
            "timeformat": "unixtime",
            "timezone": "GMT",
            "forecast_days": forecast_days,
            "cell_selection": "sea",
        }
    )
    try:
        data = _request_open_meteo(f"{_OPEN_METEO_URL}?{params}")
        if data.get("error"):
            raise ValueError(str(data.get("reason") or "Open-Meteo rejected the request"))
        hourly = data.get("hourly")
        if not isinstance(hourly, dict):
            raise ValueError("Missing hourly water levels")
        units = data.get("hourly_units") or {}
        if not isinstance(units, dict) or units.get("sea_level_height_msl", "m") != "m":
            raise ValueError("Incompatible sea-level units")
        heights = _level_series(hourly.get("time"), hourly.get("sea_level_height_msl"))
        if len([row for row in heights if row["height_m"] is not None]) < 2:
            raise ValueError("Insufficient water levels")
        extremes = _open_meteo_extremes(data)
    except (
        OSError,
        TimeoutError,
        urllib.error.HTTPError,
        urllib.error.URLError,
        json.JSONDecodeError,
        TypeError,
        ValueError,
    ) as error:
        logger.warning("Open-Meteo tide refresh failed (%s): %s", _provider_error_code(error), error)
        stale = _read_cache(key, allow_stale=True)
        if stale is not None:
            stale["error_code"] = _provider_error_code(error)
            return stale
        return _unavailable_payload(
            lat=lat,
            lon=lon,
            days=forecast_days,
            error_code=_provider_error_code(error),
            source="Open-Meteo Marine API",
        )

    payload = {
        "configured": True,
        "available": True,
        "stale": False,
        "source": "Open-Meteo Marine API",
        "provider_mode": provider_mode,
        "fallback": bool(api_key),
        "lat": lat,
        "lon": lon,
        "model_lat": _finite(data.get("latitude")),
        "model_lon": _finite(data.get("longitude")),
        "days": forecast_days,
        "requested_days": days,
        "station": {"name": "Open-Meteo marine grid"},
        "timezone": data.get("timezone") or "GMT",
        "extremes": extremes,
        "fetched_at": int(time.time()),
        "attribution": "Marine forecast data by Open-Meteo",
        "attribution_url": "https://open-meteo.com/en/docs/marine-weather-api",
        "advisory": (
            "Modelled coastal tide heights are for scheduling guidance only, "
            "not coastal navigation."
        ),
        **_series_metadata(heights, datum="MSL", reference={
            "provider": "Open-Meteo", "datum": "global MSL",
            "lat": data.get("latitude"), "lon": data.get("longitude"),
        }),
    }
    if _finite(data.get("latitude")) is None or _finite(data.get("longitude")) is None:
        # Display the requested MSL variable, but do not calibrate against an
        # unidentified model cell (nor silently assume the requested land point).
        payload["datum_reference"] = None
    if days > forecast_days:
        payload["message"] = (
            f"The available marine model currently covers {forecast_days} days; "
            "the calendar will show tide markers within that verified window."
        )
    _write_cache(key, payload)
    return payload
