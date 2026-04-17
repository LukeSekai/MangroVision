"""
MangroVision planter-facing field app.
"""

import json
import math
import os
from datetime import datetime
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import folium
import streamlit as st
import streamlit.components.v1 as components
import streamlit_folium as st_folium
from branca.element import Element

from planting_database import (
    authenticate_planter,
    create_planter,
    update_planter_last_login,
    get_planter_field_points,
    list_planter_assignments,
    update_assignment_point_status,
    create_planter_session,
    get_planter_by_session_token,
    revoke_planter_session,
)


st.set_page_config(
    page_title="MangroVision Field App",
    page_icon="🌱",
    layout="centered",
    initial_sidebar_state="collapsed",
)

_FIELD_COMPONENT_DIR = Path(__file__).parent / "components" / "field_geolocator"
_FIELD_GEOLOCATOR = components.declare_component(
    "field_geolocator",
    path=str(_FIELD_COMPONENT_DIR),
)
_PLANTER_SESSION_QUERY_KEY = "planter_session"

st.markdown("""
<style>
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
        font-family: "Aptos", "Trebuchet MS", sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top center, rgba(36, 86, 60, 0.22) 0%, rgba(36, 86, 60, 0.10) 22%, rgba(237, 241, 238, 0) 44%),
            linear-gradient(180deg, #08100b 0%, #0f1d15 24%, #2c4538 48%, #cdd9d0 70%, #edf1ee 100%);
    }

    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        background: transparent !important;
    }

    [data-testid="stSidebar"] {
        display: none !important;
    }

    [data-testid="stAppViewContainer"] [data-testid="block-container"] {
        max-width: 760px;
        padding-top: 1rem;
        padding-bottom: 2rem;
    }

    .field-shell,
    .field-login,
    .field-register,
    .field-map-card,
    .navigation-card,
    .next-card,
    .point-card {
        background: linear-gradient(160deg, rgba(8, 16, 11, 0.96) 0%, rgba(19, 33, 25, 0.95) 100%);
        border: 1px solid rgba(120, 202, 149, 0.10);
        border-radius: 24px;
        padding: 1.15rem 1.2rem;
        box-shadow: 0 20px 40px rgba(8, 16, 11, 0.24);
        margin-bottom: 1rem;
    }

    .field-shell h1,
    .field-login h1,
    .field-register h2,
    .field-map-card h2,
    .navigation-card h2,
    .next-card h2,
    .point-card h3 {
        color: #f1f8f3;
        margin: 0;
    }

    .field-kicker {
        color: #8fd3a7;
        text-transform: uppercase;
        letter-spacing: 0.16rem;
        font-size: 0.76rem;
        font-weight: 800;
        margin-bottom: 0.45rem;
    }

    .field-shell p,
    .field-login p,
    .field-register p,
    .field-map-card p,
    .navigation-card p,
    .next-card p,
    .point-card p {
        color: #cadbcc;
        line-height: 1.6;
        margin: 0.5rem 0 0 0;
    }

    .field-metrics {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 0.95rem;
    }

    .field-metrics span {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(143, 211, 167, 0.12);
        color: #eef7f1;
        border-radius: 999px;
        padding: 0.45rem 0.76rem;
        font-size: 0.82rem;
        font-weight: 700;
    }

    .point-status {
        border-radius: 999px;
        padding: 0.35rem 0.72rem;
        font-size: 0.76rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
    }

    .point-status.pending {
        background: rgba(120, 202, 149, 0.14);
        color: #b7e4c5;
    }

    .point-status.completed {
        background: rgba(126, 200, 141, 0.18);
        color: #d6f3de;
    }

    .point-status.skipped {
        background: rgba(239, 108, 0, 0.18);
        color: #ffd2ac;
    }

    .point-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.8rem;
        flex-wrap: wrap;
    }

    .point-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.75rem;
    }

    .point-meta span {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(143, 211, 167, 0.10);
        color: #eef7f1;
        border-radius: 999px;
        padding: 0.35rem 0.66rem;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .point-coords {
        color: #eef7f1;
        font-size: 1rem;
        font-weight: 800;
        margin-top: 0.8rem;
    }

    .field-map-card,
    .navigation-card {
        padding-bottom: 0.95rem;
    }

    .field-map-copy,
    .navigation-copy {
        margin-top: 0.35rem;
    }

    .nav-helper {
        color: #bcd0c1;
        font-size: 0.88rem;
        line-height: 1.55;
        margin: 0.75rem 0 0.4rem 0;
    }

    .status-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.75rem;
        margin-bottom: 1rem;
    }

    .status-card {
        background: linear-gradient(160deg, rgba(8, 16, 11, 0.96) 0%, rgba(19, 33, 25, 0.95) 100%);
        border: 1px solid rgba(120, 202, 149, 0.10);
        border-radius: 20px;
        padding: 0.9rem 1rem;
        box-shadow: 0 18px 34px rgba(8, 16, 11, 0.18);
    }

    .status-label {
        color: #9fcab0;
        font-size: 0.74rem;
        font-weight: 800;
        letter-spacing: 0.12rem;
        text-transform: uppercase;
        margin-bottom: 0.4rem;
    }

    .status-value {
        color: #f1f8f3;
        font-size: 1.02rem;
        font-weight: 800;
        line-height: 1.25;
    }

    .status-subvalue {
        color: #bcd0c1;
        font-size: 0.84rem;
        line-height: 1.45;
        margin-top: 0.28rem;
    }

    .target-card,
    .queue-shell {
        background: linear-gradient(160deg, rgba(8, 16, 11, 0.98) 0%, rgba(19, 33, 25, 0.96) 100%);
        border: 1px solid rgba(120, 202, 149, 0.10);
        border-radius: 24px;
        padding: 1.1rem 1.15rem;
        box-shadow: 0 20px 40px rgba(8, 16, 11, 0.22);
        margin-bottom: 1rem;
    }

    .target-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.8rem;
        flex-wrap: wrap;
    }

    .target-card h2,
    .queue-shell h2 {
        color: #f1f8f3;
        margin: 0;
        line-height: 1.1;
    }

    .target-note {
        color: #cfe1d4;
        font-size: 0.95rem;
        line-height: 1.55;
        margin-top: 0.55rem;
    }

    .target-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.8rem;
    }

    .target-meta span {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(143, 211, 167, 0.12);
        color: #eef7f1;
        border-radius: 999px;
        padding: 0.38rem 0.7rem;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .queue-item {
        border-top: 1px solid rgba(120, 202, 149, 0.10);
        padding-top: 0.85rem;
        margin-top: 0.85rem;
    }

    .queue-item:first-of-type {
        border-top: none;
        padding-top: 0;
        margin-top: 0.75rem;
    }

    .queue-item-header {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.8rem;
        flex-wrap: wrap;
    }

    .queue-item-title {
        color: #f1f8f3;
        font-size: 1rem;
        font-weight: 800;
        line-height: 1.25;
    }

    .queue-item-copy {
        color: #cadbcc;
        font-size: 0.9rem;
        line-height: 1.55;
        margin-top: 0.35rem;
    }

    .queue-item-coords {
        color: #e7f4eb;
        font-size: 0.86rem;
        font-weight: 700;
        margin-top: 0.4rem;
    }

    .queue-browser-shell {
        margin-bottom: 0.35rem;
    }

    .queue-browser-copy {
        color: #cadbcc;
        font-size: 0.92rem;
        line-height: 1.55;
        margin-top: 0.4rem;
    }

    .queue-browser-card {
        min-height: 15.5rem;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
        margin-bottom: 0.7rem;
    }

    .queue-browser-top {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.8rem;
        flex-wrap: wrap;
    }

    .queue-browser-title {
        color: #f4fbf6;
        font-size: 1.22rem;
        font-weight: 800;
        line-height: 1.2;
        margin-top: 0.15rem;
    }

    .queue-browser-index {
        display: flex;
        gap: 0.45rem;
        flex-wrap: wrap;
        justify-content: flex-end;
    }

    .queue-browser-index span {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(143, 211, 167, 0.14);
        color: #eef7f1;
        border-radius: 999px;
        padding: 0.38rem 0.68rem;
        font-size: 0.78rem;
        font-weight: 800;
        letter-spacing: 0.05rem;
        text-transform: uppercase;
    }

    .queue-browser-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.45rem;
        margin-top: 0.9rem;
    }

    .queue-browser-meta span {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(143, 211, 167, 0.10);
        color: #dfeee3;
        border-radius: 999px;
        padding: 0.38rem 0.68rem;
        font-size: 0.82rem;
        font-weight: 700;
    }

    .queue-browser-coords {
        color: #f4fbf6;
        font-size: 1.08rem;
        font-weight: 800;
        margin-top: 0.95rem;
        letter-spacing: 0.01rem;
    }

    .queue-browser-note {
        color: #bcd0c1;
        font-size: 0.88rem;
        line-height: 1.55;
        margin-top: 0.8rem;
    }

    .queue-browser-arrow .stButton > button {
        min-height: 15.5rem;
        font-size: 1.65rem;
        padding: 0;
        border-radius: 22px !important;
    }

    .queue-browser-actions {
        margin-bottom: 0.95rem;
    }

    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(143, 211, 167, 0.10) !important;
        border-radius: 18px !important;
        padding: 1rem !important;
    }

    div[data-testid="stForm"] label,
    div[data-testid="stForm"] p,
    div[data-testid="stForm"] span,
    div[data-testid="stForm"] input {
        color: #eef7f1 !important;
    }

    div[data-testid="stForm"] [data-baseweb="input"] {
        background: rgba(255, 255, 255, 0.06) !important;
        border-color: rgba(143, 211, 167, 0.14) !important;
    }

    .stButton > button,
    .stLinkButton > a {
        width: 100%;
        border-radius: 16px !important;
        min-height: 3rem;
        font-weight: 700 !important;
    }

    .stButton > button {
        background: linear-gradient(135deg, #173728 0%, #24563c 100%);
        color: white;
        border: none;
    }

    .stLinkButton > a {
        background: linear-gradient(135deg, #24563c 0%, #2f7252 100%);
        color: white !important;
        border: none !important;
        display: flex;
        align-items: center;
        justify-content: center;
        text-decoration: none !important;
    }

    @media (max-width: 700px) {
        [data-testid="stAppViewContainer"] [data-testid="block-container"] {
            padding-top: 0.65rem;
            padding-left: 0.85rem;
            padding-right: 0.85rem;
        }

        .field-shell,
        .field-login,
        .field-register,
        .field-map-card,
        .navigation-card,
        .next-card,
        .point-card,
        .target-card,
        .queue-shell {
            padding: 1rem 0.95rem;
            border-radius: 22px;
        }

        .field-shell h1,
        .field-login h1 {
            font-size: 2rem;
        }

        .status-grid {
            grid-template-columns: 1fr;
        }
    }
</style>
""", unsafe_allow_html=True)


def _set_planter_auth_query(session_token: str | None):
    """Persist planter session token in the URL so refresh restores login."""
    try:
        if session_token:
            st.query_params[_PLANTER_SESSION_QUERY_KEY] = session_token
        else:
            st.query_params.pop(_PLANTER_SESSION_QUERY_KEY, None)
    except Exception:
        pass


def _restore_planter_auth_from_query():
    """Restore planter login state from a persisted session token."""
    if st.session_state.get("field_logged_in"):
        return
    try:
        session_token = st.query_params.get(_PLANTER_SESSION_QUERY_KEY, "")
    except Exception:
        return
    session_token = str(session_token or "").strip()
    if not session_token:
        return

    planter = get_planter_by_session_token(session_token)
    if planter:
        st.session_state.field_logged_in = True
        st.session_state.field_planter_id = planter.get("id")
        st.session_state.field_planter_name = planter.get("full_name")
        st.session_state.field_auth_session_token = session_token
        return

    _set_planter_auth_query(None)


def _clear_planter_auth_state(revoke_session: bool = True):
    """Clear planter auth state and optionally revoke the persisted session."""
    session_token = st.session_state.get("field_auth_session_token")
    if not session_token:
        try:
            session_token = str(st.query_params.get(_PLANTER_SESSION_QUERY_KEY, "") or "").strip()
        except Exception:
            session_token = ""
    if revoke_session and session_token:
        revoke_planter_session(session_token)

    st.session_state.field_logged_in = False
    st.session_state.field_planter_id = None
    st.session_state.field_planter_name = None
    st.session_state.field_auth_session_token = None
    st.session_state.field_nav_point_id = None
    st.session_state.field_current_location = None
    st.session_state.field_location_error = ""
    st.session_state.field_location_refresh_nonce = 0
    st.session_state.field_map_center = None
    st.session_state.field_map_zoom = None
    st.session_state.field_map_planter_id = None
    st.session_state.field_queue_point_id = None
    st.session_state.field_map_force_view = True
    _set_planter_auth_query(None)


def _init_auth_state():
    """Initialize planter auth session keys."""
    if "field_logged_in" not in st.session_state:
        st.session_state.field_logged_in = False
    if "field_planter_id" not in st.session_state:
        st.session_state.field_planter_id = None
    if "field_planter_name" not in st.session_state:
        st.session_state.field_planter_name = None
    if "field_nav_point_id" not in st.session_state:
        st.session_state.field_nav_point_id = None
    if "field_location_refresh_nonce" not in st.session_state:
        st.session_state.field_location_refresh_nonce = 0
    if "field_current_location" not in st.session_state:
        st.session_state.field_current_location = None
    if "field_location_error" not in st.session_state:
        st.session_state.field_location_error = ""
    if "field_map_center" not in st.session_state:
        st.session_state.field_map_center = None
    if "field_map_zoom" not in st.session_state:
        st.session_state.field_map_zoom = None
    if "field_map_planter_id" not in st.session_state:
        st.session_state.field_map_planter_id = None
    if "field_queue_point_id" not in st.session_state:
        st.session_state.field_queue_point_id = None
    if "field_map_force_view" not in st.session_state:
        st.session_state.field_map_force_view = True
    if "field_auth_session_token" not in st.session_state:
        st.session_state.field_auth_session_token = None
    _restore_planter_auth_from_query()


def _get_google_routes_api_key() -> str:
    """Return the Google Routes API key from Streamlit secrets or env."""
    key = ""
    try:
        key = st.secrets.get("google_routes_api_key", "")
    except Exception:
        key = ""
    if not key:
        key = os.getenv("GOOGLE_ROUTES_API_KEY", "")
    return (key or "").strip()


def _get_tile_server_base_url() -> str:
    """Return the tile server base URL for orthophoto tiles."""
    base_url = ""
    try:
        base_url = st.secrets.get("tile_server_base_url", "")
    except Exception:
        base_url = ""
    if not base_url:
        base_url = os.getenv("MANGROVISION_TILE_SERVER_BASE_URL", "http://localhost:8080")
    return (base_url or "http://localhost:8080").rstrip("/")


def _routes_travel_mode(travel_mode: str) -> str:
    """Normalize travel mode to a Google Routes API travel mode."""
    mode = _normalize_travel_mode(travel_mode)
    return {
        "walking": "WALK",
        "driving": "DRIVE",
        "bicycling": "BICYCLE",
        "transit": "TRANSIT",
    }.get(mode, "WALK")


def _normalize_travel_mode(travel_mode: str) -> str:
    """Normalize travel mode to a Google-supported mode."""
    mode = (travel_mode or "walking").strip().lower()
    if mode not in {"driving", "walking", "bicycling", "transit"}:
        return "walking"
    return mode


def _request_browser_geolocation() -> dict | None:
    """Return the latest browser geolocation from the custom component."""
    if not _FIELD_COMPONENT_DIR.exists():
        return None
    return _FIELD_GEOLOCATOR(
        refresh_nonce=int(st.session_state.field_location_refresh_nonce),
        default=None,
        key="field_geolocator_widget",
    )


def _decode_google_polyline(encoded: str) -> list[tuple[float, float]]:
    """Decode a Google encoded polyline into (lat, lon) tuples."""
    coordinates: list[tuple[float, float]] = []
    index = 0
    lat = 0
    lon = 0

    while index < len(encoded):
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

        coordinates.append((lat / 1e5, lon / 1e5))

    return coordinates


def _parse_duration_seconds(duration_value: str | None) -> float | None:
    """Parse a Google duration string like '542s' into seconds."""
    if not duration_value:
        return None
    try:
        return float(str(duration_value).rstrip("s"))
    except ValueError:
        return None


def _format_route_duration(duration_seconds: float | None) -> str:
    """Format route duration in a compact human-readable form."""
    if duration_seconds is None:
        return "Route duration unavailable"
    total_seconds = int(round(duration_seconds))
    minutes, seconds = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours > 0:
        return f"{hours}h {minutes}m"
    if minutes > 0:
        return f"{minutes}m"
    return f"{seconds}s"


def _format_route_distance(distance_meters: float | None) -> str:
    """Format route distance in meters or kilometers."""
    if distance_meters is None:
        return "Route distance unavailable"
    if distance_meters >= 1000:
        return f"{distance_meters / 1000:.2f} km"
    return f"{distance_meters:.0f} m"


def _approx_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Return an approximate distance in meters using a simple planar conversion."""
    lat_scale = 111320.0
    lon_scale = 111320.0 * math.cos(math.radians((lat1 + lat2) / 2.0))
    dy = (lat2 - lat1) * lat_scale
    dx = (lon2 - lon1) * lon_scale
    return (dx * dx + dy * dy) ** 0.5


def _format_location_timestamp(iso_value: str | None) -> str:
    """Format ISO timestamp into a compact field-friendly label."""
    if not iso_value:
        return "Location timestamp unavailable"
    try:
        dt = datetime.fromisoformat(str(iso_value).replace("Z", "+00:00"))
    except ValueError:
        return str(iso_value)
    return dt.strftime("%I:%M:%S %p")


@st.cache_data(ttl=30, show_spinner=False)
def _compute_google_route(
    origin_lat: float,
    origin_lon: float,
    dest_lat: float,
    dest_lon: float,
    travel_mode: str,
) -> dict | None:
    """Call Google Routes API and return a decoded route payload."""
    api_key = _get_google_routes_api_key()
    if not api_key:
        return None

    body = {
        "origin": {
            "location": {
                "latLng": {
                    "latitude": origin_lat,
                    "longitude": origin_lon,
                }
            }
        },
        "destination": {
            "location": {
                "latLng": {
                    "latitude": dest_lat,
                    "longitude": dest_lon,
                }
            }
        },
        "travelMode": _routes_travel_mode(travel_mode),
        "computeAlternativeRoutes": False,
        "languageCode": "en-US",
        "units": "METRIC",
    }
    payload = json.dumps(body).encode("utf-8")
    request = Request(
        "https://routes.googleapis.com/directions/v2:computeRoutes",
        data=payload,
        method="POST",
        headers={
            "Content-Type": "application/json",
            "X-Goog-Api-Key": api_key,
            "X-Goog-FieldMask": "routes.distanceMeters,routes.duration,routes.polyline.encodedPolyline",
        },
    )

    try:
        with urlopen(request, timeout=20) as response:
            response_payload = json.loads(response.read().decode("utf-8"))
    except HTTPError as err:
        detail = err.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"Google Routes API error ({err.code}): {detail or err.reason}") from err
    except URLError as err:
        raise RuntimeError(f"Google Routes API connection error: {err.reason}") from err

    route = (response_payload.get("routes") or [None])[0]
    if not route:
        return None

    encoded_polyline = ((route.get("polyline") or {}).get("encodedPolyline") or "").strip()
    if not encoded_polyline:
        return None

    return {
        "distance_meters": route.get("distanceMeters"),
        "duration_seconds": _parse_duration_seconds(route.get("duration")),
        "polyline_points": _decode_google_polyline(encoded_polyline),
    }


def _add_operational_map_layers(map_obj, orthophoto_name: str = "Orthophoto Overlay"):
    """Add the shared basemap stack so WebODM tiles sit above a satellite fallback."""
    tile_server_base_url = _get_tile_server_base_url()

    folium.TileLayer(
        tiles="https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}",
        attr="Google Satellite",
        name="Satellite Base",
        overlay=False,
        control=True,
        max_zoom=21,
    ).add_to(map_obj)

    folium.TileLayer(
        tiles=f"{tile_server_base_url}/FINAL%20MAP/{{z}}/{{x}}/{{y}}.jpg",
        attr="MangroVision Orthophoto | QGIS",
        name=orthophoto_name,
        overlay=True,
        control=True,
        max_zoom=20,
        min_zoom=10,
        show=True,
        opacity=1.0,
    ).add_to(map_obj)


def _style_layer_control(map_obj):
    """Apply a dark MangroVision skin to Folium layer controls."""
    map_obj.get_root().header.add_child(Element("""
    <style>
        .leaflet-control-layers-expanded {
            min-width: 220px;
            padding: 0.9rem 0.95rem 0.8rem 0.95rem !important;
            border-radius: 18px !important;
            border: 1px solid rgba(120, 202, 149, 0.16) !important;
            background: linear-gradient(160deg, rgba(8, 16, 11, 0.96) 0%, rgba(19, 33, 25, 0.95) 100%) !important;
            box-shadow: 0 18px 34px rgba(8, 16, 11, 0.30) !important;
            color: #eef7f1 !important;
            backdrop-filter: blur(10px);
        }

        .leaflet-control-layers-base,
        .leaflet-control-layers-overlays {
            display: grid;
            gap: 0.28rem;
            margin-top: 0.15rem;
        }

        .leaflet-control-layers label {
            display: flex !important;
            align-items: center;
            gap: 0.45rem;
            padding: 0.38rem 0.44rem;
            border-radius: 12px;
            color: #eef7f1 !important;
            font-size: 0.95rem;
            font-weight: 600;
            transition: background 0.18s ease;
        }

        .leaflet-control-layers label:hover {
            background: rgba(120, 202, 149, 0.08);
        }

        .leaflet-control-layers-separator {
            border-top: 1px solid rgba(120, 202, 149, 0.16) !important;
            margin: 0.45rem 0 !important;
        }

        .leaflet-control-layers-selector {
            accent-color: #7fd29b;
            transform: scale(1.05);
        }
    </style>
    """))


def _build_field_assignment_map(points, selected_point, current_location=None, route_info=None):
    """Build the stable base map used by the planter field workspace."""
    from folium.plugins import Fullscreen

    field_map = folium.Map(
        location=[10.7800, 122.6253],
        zoom_start=19,
        tiles=None,
        control_scale=True,
    )
    _add_operational_map_layers(field_map)
    Fullscreen(position="topleft", title="Expand map", title_cancel="Exit fullscreen").add_to(field_map)
    _style_layer_control(field_map)
    return field_map


def _build_field_assignment_layers(points, selected_point, current_location=None, route_info=None):
    """Build dynamic layers so GPS refreshes do not remount the base map."""
    if not points:
        return []

    sorted_points = sorted(points, key=lambda row: row["sequence_num"])
    feature_groups = []
    route_points = [(point["latitude"], point["longitude"]) for point in sorted_points]
    if len(route_points) >= 2:
        route_group = folium.FeatureGroup(name="Assignment Route", show=False)
        folium.PolyLine(
            locations=route_points,
            color="#8fd3a7",
            weight=3,
            opacity=0.72,
            tooltip="Assignment route order",
        ).add_to(route_group)
        feature_groups.append(route_group)

    if route_info and route_info.get("polyline_points"):
        live_route_group = folium.FeatureGroup(name="Live Route", show=True)
        live_route_points = [(lat, lon) for lat, lon in route_info["polyline_points"]]
        folium.PolyLine(
            locations=live_route_points,
            color="#00E5FF",
            weight=5,
            opacity=0.92,
            tooltip="Google Routes path",
        ).add_to(live_route_group)
        if selected_point:
            route_end_lat, route_end_lon = live_route_points[-1]
            target_gap_m = _approx_distance_m(
                route_end_lat,
                route_end_lon,
                selected_point["latitude"],
                selected_point["longitude"],
            )
            if target_gap_m >= 3:
                folium.PolyLine(
                    locations=[
                        [route_end_lat, route_end_lon],
                        [selected_point["latitude"], selected_point["longitude"]],
                    ],
                    color="#FFE082",
                    weight=4,
                    opacity=0.95,
                    dash_array="10, 10",
                    tooltip="Final direct approach",
                ).add_to(live_route_group)
        feature_groups.append(live_route_group)

    assignment_group = folium.FeatureGroup(name="Assigned Points", show=True)
    for point in sorted_points:
        status = point.get("assignment_status") or "pending"
        border_color = "#8fd3a7"
        fill_color = "#4CAF50"
        if status == "completed":
            border_color = "#A5D6A7"
            fill_color = "#81C784"
        elif status == "skipped":
            border_color = "#FFB74D"
            fill_color = "#FB8C00"

        radius = 7
        weight = 2
        tooltip = f"Point {point['sequence_num']:02d}"
        if selected_point and point["assignment_point_id"] == selected_point["assignment_point_id"]:
            radius = 10
            weight = 3
            border_color = "#FDD835"
            fill_color = "#FFEE58"
            tooltip = f"Navigation Target: Point {point['sequence_num']:02d}"

        folium.CircleMarker(
            location=[point["latitude"], point["longitude"]],
            radius=radius,
            color=border_color,
            fill=True,
            fillColor=fill_color,
            fillOpacity=0.88,
            weight=weight,
            tooltip=tooltip,
            popup=(
                f"<b>Point {point['sequence_num']:02d}</b><br>"
                f"{point['title']}<br>"
                f"{point['latitude']:.7f}, {point['longitude']:.7f}<br>"
                f"Status: {(point.get('assignment_status') or 'pending').title()}"
            ),
        ).add_to(assignment_group)
    feature_groups.append(assignment_group)

    if current_location:
        current_group = folium.FeatureGroup(name="Current Location", show=True)
        folium.CircleMarker(
            location=[current_location["lat"], current_location["lon"]],
            radius=8,
            color="#90CAF9",
            fill=True,
            fillColor="#42A5F5",
            fillOpacity=0.95,
            weight=2,
            tooltip="Your current location",
            popup=(
                "<b>Current location</b><br>"
                f"{current_location['lat']:.7f}, {current_location['lon']:.7f}"
            ),
        ).add_to(current_group)
        feature_groups.append(current_group)

    return feature_groups


def _default_field_map_view(points, selected_point, current_location=None):
    """Return the default center/zoom used the first time a planter opens the map."""
    if selected_point:
        return [selected_point["latitude"], selected_point["longitude"]], 20
    if current_location:
        return [current_location["lat"], current_location["lon"]], 19
    if points:
        first_point = sorted(points, key=lambda row: row["sequence_num"])[0]
        return [first_point["latitude"], first_point["longitude"]], 19
    return [10.7800, 122.6253], 18


def _google_maps_url(dest_lat: float, dest_lon: float, travel_mode: str = "walking") -> str:
    """Build a Google Maps directions URL for the planter's current location."""
    params = {
        "api": 1,
        "destination": f"{dest_lat:.7f},{dest_lon:.7f}",
        "travelmode": _normalize_travel_mode(travel_mode),
    }
    return "https://www.google.com/maps/dir/?" + urlencode(params)


def _link_button(label: str, url: str):
    """Render external navigation button."""
    if hasattr(st, "link_button"):
        st.link_button(label, url, use_container_width=True, type="primary")
    else:
        st.markdown(f"[{label}]({url})")


def _render_login() -> bool:
    """Render planter login screen."""
    _init_auth_state()
    if st.session_state.field_logged_in:
        return True

    now_str = datetime.now().strftime("%B %d, %Y | %I:%M %p")
    st.markdown(f"""
    <div class="field-login">
        <div class="field-kicker">MangroVision Field App</div>
        <h1>Planter Access</h1>
        <p>Sign in to open your assigned planting points, launch navigation to the next destination, and update completion status directly from the field.</p>
        <div class="field-metrics">
            <span>Mobile first</span>
            <span>One-tap navigation</span>
            <span>Completion tracking</span>
            <span>{now_str}</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    login_tab, register_tab = st.tabs(["Sign In", "Create Account"])

    with login_tab:
        with st.form("field_login_form", clear_on_submit=False):
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            login_clicked = st.form_submit_button("Open Field Workspace", use_container_width=True, type="primary")

        if login_clicked:
            planter = authenticate_planter(username.strip(), password)
            if planter:
                update_planter_last_login(planter["id"])
                session_token = create_planter_session(planter["id"])
                st.session_state.field_logged_in = True
                st.session_state.field_planter_id = planter["id"]
                st.session_state.field_planter_name = planter["full_name"]
                st.session_state.field_auth_session_token = session_token
                st.session_state.field_nav_point_id = None
                st.session_state.field_current_location = None
                st.session_state.field_location_error = ""
                st.session_state.field_location_refresh_nonce = 0
                st.session_state.field_map_center = None
                st.session_state.field_map_zoom = None
                st.session_state.field_map_planter_id = None
                st.session_state.field_queue_point_id = None
                st.session_state.field_map_force_view = True
                _set_planter_auth_query(session_token)
                st.rerun()
            else:
                st.error("Invalid planter username or password.")

    with register_tab:
        st.markdown("""
        <div class="field-register">
            <div class="field-kicker">Field Registration</div>
            <h2>Create Your Planter Account</h2>
            <p>Register here first, then your account will appear in the planner console for assignment management.</p>
        </div>
        """, unsafe_allow_html=True)
        with st.form("field_register_form", clear_on_submit=True):
            full_name = st.text_input("Full name")
            new_username = st.text_input("Username", key="register_username")
            new_password = st.text_input("Password", type="password", key="register_password")
            phone = st.text_input("Phone / contact")
            register_clicked = st.form_submit_button("Create Field Account", use_container_width=True, type="primary")

        if register_clicked:
            try:
                create_planter(
                    full_name=full_name,
                    username=new_username,
                    password=new_password,
                    phone=phone,
                )
                st.success("Planter account created. You can now sign in from the Sign In tab.")
            except ValueError as err:
                st.warning(str(err))

    return False


def _logout():
    """Clear planter auth session."""
    _clear_planter_auth_state(revoke_session=True)
    st.rerun()


def main():
    """Run the planter-facing field app."""
    if not _render_login():
        return

    planter_id = st.session_state.field_planter_id
    planter_name = st.session_state.field_planter_name or "Planter"
    field_points = get_planter_field_points(planter_id)
    active_assignments = list_planter_assignments(planter_id=planter_id, active_only=True)

    status_order = {"pending": 0, "completed": 1, "skipped": 2}
    ordered_points = sorted(
        field_points,
        key=lambda row: (
            status_order.get(row["assignment_status"], 9),
            row["assignment_date"],
            row["sequence_num"],
        ),
    )
    pending_points = [point for point in ordered_points if point["assignment_status"] == "pending"]
    completed_count = sum(1 for point in field_points if point["assignment_status"] == "completed")
    skipped_count = sum(1 for point in field_points if point["assignment_status"] == "skipped")
    next_point = pending_points[0] if pending_points else None
    selected_point = next(
        (
            point
            for point in ordered_points
            if point["assignment_point_id"] == st.session_state.field_nav_point_id
        ),
        None,
    )
    if not selected_point:
        selected_point = next_point or (ordered_points[0] if ordered_points else None)
        st.session_state.field_nav_point_id = (
            selected_point["assignment_point_id"] if selected_point else None
        )

    location_payload = _request_browser_geolocation()
    if isinstance(location_payload, dict):
        if location_payload.get("status") == "success":
            st.session_state.field_current_location = {
                "lat": float(location_payload["latitude"]),
                "lon": float(location_payload["longitude"]),
                "accuracy": float(location_payload.get("accuracy") or 0),
                "captured_at": location_payload.get("captured_at") or datetime.now().isoformat(timespec="seconds"),
            }
            st.session_state.field_location_error = ""
        elif location_payload.get("status") == "error":
            st.session_state.field_location_error = location_payload.get("message") or "Current location is unavailable."

    current_location = st.session_state.field_current_location
    default_map_center, default_map_zoom = _default_field_map_view(
        ordered_points,
        selected_point,
        current_location=current_location,
    )
    if st.session_state.field_map_planter_id != planter_id:
        st.session_state.field_map_planter_id = planter_id
        st.session_state.field_map_center = default_map_center
        st.session_state.field_map_zoom = default_map_zoom
        st.session_state.field_map_force_view = True
    if st.session_state.field_map_center is None:
        st.session_state.field_map_center = default_map_center
        st.session_state.field_map_force_view = True
    if st.session_state.field_map_zoom is None:
        st.session_state.field_map_zoom = default_map_zoom
        st.session_state.field_map_force_view = True

    route_info = None
    route_error = ""
    if selected_point and current_location and _get_google_routes_api_key():
        try:
            route_info = _compute_google_route(
                current_location["lat"],
                current_location["lon"],
                selected_point["latitude"],
                selected_point["longitude"],
                selected_point["travel_mode"],
            )
        except RuntimeError as err:
            route_error = str(err)

    header_col, logout_col = st.columns([0.78, 0.22], gap="medium")
    with header_col:
        st.markdown(f"""
        <div class="field-shell">
            <div class="field-kicker">Field Workspace</div>
            <h1>{planter_name}</h1>
            <p>Open navigation for the next assigned planting point, then mark that location completed or skipped once field work is done.</p>
            <div class="field-metrics">
                <span>{len(active_assignments)} active batches</span>
                <span>{len(pending_points)} pending points</span>
                <span>{completed_count} completed</span>
                <span>{skipped_count} skipped</span>
            </div>
        </div>
        """, unsafe_allow_html=True)
    with logout_col:
        st.button("Log Out", use_container_width=True, on_click=_logout)

    if not field_points:
        st.info("No active field assignments are available for this planter.")
        return

    selected_status = selected_point["assignment_status"] if selected_point else None
    route_distance_label = _format_route_distance(route_info.get("distance_meters")) if route_info else "Waiting for route"
    route_duration_label = _format_route_duration(route_info.get("duration_seconds")) if route_info else "Waiting for ETA"
    tracking_value = "Live tracking active" if current_location else "Waiting for GPS"
    tracking_subvalue = (
        f"Updated {_format_location_timestamp(current_location.get('captured_at'))}"
        + (
            f" | Accuracy {current_location['accuracy']:.0f} m"
            if current_location and current_location.get("accuracy")
            else ""
        )
        if current_location
        else (st.session_state.field_location_error or "Allow location access on the planter device.")
    )
    next_target_label = (
        f"Point {next_point['sequence_num']:02d}" if next_point else "All assigned points handled"
    )
    next_target_subvalue = next_point["title"] if next_point else "No pending points remain"
    route_mode_label = _normalize_travel_mode(selected_point["travel_mode"]) if selected_point else "walking"

    if selected_point:
        st.markdown(f"""
        <div class="target-card">
            <div class="target-header">
                <div>
                    <div class="field-kicker">Current Target</div>
                    <h2>Point {selected_point['sequence_num']:02d} - {selected_point['title']}</h2>
                </div>
                <div class="point-status {selected_status}">{selected_status}</div>
            </div>
            <p class="target-note">Keep this point as your active field destination. Use the large action buttons below after you reach or assess the site.</p>
            <div class="target-meta">
                <span>{selected_point['image_name']}</span>
                <span>{selected_point['assignment_date']}</span>
                <span>{route_mode_label}</span>
                <span>Point #{selected_point['point_num']}</span>
            </div>
            <div class="point-coords">{selected_point['latitude']:.7f}, {selected_point['longitude']:.7f}</div>
        </div>
        """, unsafe_allow_html=True)

        if next_point and selected_point["assignment_point_id"] != next_point["assignment_point_id"]:
            st.info(f"Recommended next planting point: Point {next_point['sequence_num']:02d} - {next_point['title']}")

        action_col1, action_col2, action_col3 = st.columns(3)
        with action_col1:
            primary_label = "Complete Current Target" if selected_status == "pending" else "Reset Current Target"
            if st.button(primary_label, key="selected_point_primary", use_container_width=True):
                if selected_status == "pending":
                    if st.session_state.field_nav_point_id == selected_point["assignment_point_id"]:
                        st.session_state.field_nav_point_id = None
                    update_assignment_point_status(selected_point["assignment_point_id"], "completed")
                else:
                    update_assignment_point_status(selected_point["assignment_point_id"], "pending")
                st.rerun()
        with action_col2:
            secondary_label = "Skip Current Target" if selected_status == "pending" else "Switch To Next Pending"
            secondary_disabled = selected_status != "pending" and not next_point
            if st.button(secondary_label, key="selected_point_secondary", use_container_width=True, disabled=secondary_disabled):
                if selected_status == "pending":
                    if st.session_state.field_nav_point_id == selected_point["assignment_point_id"]:
                        st.session_state.field_nav_point_id = None
                    update_assignment_point_status(selected_point["assignment_point_id"], "skipped")
                elif next_point:
                    st.session_state.field_nav_point_id = next_point["assignment_point_id"]
                st.rerun()
        with action_col3:
            _link_button(
                "Open In Google Maps",
                _google_maps_url(
                    selected_point["latitude"],
                    selected_point["longitude"],
                    selected_point["travel_mode"],
                ),
            )
    else:
        st.success("No pending planting points remain. This planter has completed all active field assignments.")

    st.markdown(f"""
    <div class="status-grid">
        <div class="status-card">
            <div class="status-label">Tracking</div>
            <div class="status-value">{tracking_value}</div>
            <div class="status-subvalue">{tracking_subvalue}</div>
        </div>
        <div class="status-card">
            <div class="status-label">Route Distance</div>
            <div class="status-value">{route_distance_label}</div>
            <div class="status-subvalue">{route_error or 'Distance from current planter location to the active target.'}</div>
        </div>
        <div class="status-card">
            <div class="status-label">Estimated Time</div>
            <div class="status-value">{route_duration_label}</div>
            <div class="status-subvalue">Updates automatically while GPS tracking is active.</div>
        </div>
        <div class="status-card">
            <div class="status-label">Recommended Next</div>
            <div class="status-value">{next_target_label}</div>
            <div class="status-subvalue">{next_target_subvalue}</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class="field-map-card">
        <div class="field-kicker">Field Map</div>
        <h2>Live Planting Route</h2>
        <p class="field-map-copy">GPS, route lines, and assigned markers keep updating in real time, but the viewport stays where the planter left it unless they explicitly recenter the map.</p>
    </div>
    """, unsafe_allow_html=True)
    location_col, refresh_col, recenter_col = st.columns([0.52, 0.24, 0.24], gap="small")
    with location_col:
        if current_location:
            st.caption(
                f"Live GPS: {current_location['lat']:.7f}, {current_location['lon']:.7f}"
                + (
                    f" | Accuracy {current_location['accuracy']:.0f} m"
                    if current_location.get("accuracy")
                    else ""
                )
                + f" | Updated {_format_location_timestamp(current_location.get('captured_at'))}"
            )
        elif st.session_state.field_location_error:
            st.caption(f"GPS status: {st.session_state.field_location_error}")
        else:
            st.caption("GPS status: waiting for browser location permission.")
    with refresh_col:
        if st.button("Re-sync GPS", key="refresh_field_location", use_container_width=True):
            st.session_state.field_location_refresh_nonce += 1
            st.rerun()
    with recenter_col:
        recenter_label = "Center On Target" if selected_point else "Center Map"
        if st.button(recenter_label, key="recenter_field_map", use_container_width=True):
            focus_center, focus_zoom = _default_field_map_view(
                ordered_points,
                selected_point,
                current_location=current_location,
            )
            st.session_state.field_map_center = focus_center
            st.session_state.field_map_zoom = focus_zoom
            st.session_state.field_map_force_view = True
            st.rerun()

    assignment_map = _build_field_assignment_map(
        ordered_points,
        selected_point,
        current_location=current_location,
        route_info=route_info,
    )
    if assignment_map is not None:
        assignment_layers = _build_field_assignment_layers(
            ordered_points,
            selected_point,
            current_location=current_location,
            route_info=route_info,
        )
        map_center_arg = None
        map_zoom_arg = None
        if st.session_state.field_map_force_view and st.session_state.field_map_center is not None:
            map_center_arg = tuple(st.session_state.field_map_center)
        if st.session_state.field_map_force_view and st.session_state.field_map_zoom is not None:
            map_zoom_arg = int(st.session_state.field_map_zoom)
        map_response = st_folium.st_folium(
            assignment_map,
            height=430,
            key="field_assignment_map",
            returned_objects=["center", "zoom"],
            center=map_center_arg,
            zoom=map_zoom_arg,
            feature_group_to_add=assignment_layers,
            layer_control=folium.LayerControl(collapsed=False),
            use_container_width=True,
        )
        st.session_state.field_map_force_view = False
        if isinstance(map_response, dict):
            center_payload = map_response.get("center") or {}
            if (
                isinstance(center_payload, dict)
                and center_payload.get("lat") is not None
                and center_payload.get("lng") is not None
            ):
                new_center = [
                    float(center_payload["lat"]),
                    float(center_payload["lng"]),
                ]
                if st.session_state.field_map_center != new_center:
                    st.session_state.field_map_center = new_center
            zoom_payload = map_response.get("zoom")
            if zoom_payload is not None:
                new_zoom = int(zoom_payload)
                if st.session_state.field_map_zoom != new_zoom:
                    st.session_state.field_map_zoom = new_zoom

    pending_queue = [point for point in ordered_points if point["assignment_status"] == "pending"]
    history_queue = [point for point in ordered_points if point["assignment_status"] != "pending"]
    queue_point = None
    queue_index = 0

    if not pending_queue:
        st.success("No pending points remain in the active queue.")
    else:
        pending_ids = [point["assignment_point_id"] for point in pending_queue]
        active_queue_point_id = st.session_state.field_queue_point_id
        if active_queue_point_id not in pending_ids:
            if selected_point and selected_point["assignment_status"] == "pending":
                active_queue_point_id = selected_point["assignment_point_id"]
            elif next_point:
                active_queue_point_id = next_point["assignment_point_id"]
            else:
                active_queue_point_id = pending_ids[0]
            st.session_state.field_queue_point_id = active_queue_point_id

        queue_index = pending_ids.index(st.session_state.field_queue_point_id)
        queue_point = pending_queue[queue_index]
        queue_is_selected = (
            selected_point is not None
            and queue_point["assignment_point_id"] == selected_point["assignment_point_id"]
        )

        st.markdown("""
        <div class="queue-shell queue-browser-shell">
            <div class="field-kicker">Pending Queue</div>
            <h2>Browse Remaining Planting Points</h2>
            <p class="queue-browser-copy">Review one pending location at a time, move left or right through the queue, and keep the action buttons fixed below the card so the planter only focuses on one destination.</p>
        </div>
        """, unsafe_allow_html=True)

        prev_col, card_col, next_col = st.columns([0.16, 0.68, 0.16], gap="small")
        with prev_col:
            st.markdown("<div style='height:5.2rem'></div>", unsafe_allow_html=True)
            if st.button(
                "←",
                key="field_queue_prev",
                use_container_width=True,
                disabled=len(pending_queue) <= 1,
            ):
                previous_index = (queue_index - 1) % len(pending_queue)
                st.session_state.field_queue_point_id = pending_queue[previous_index]["assignment_point_id"]
                st.rerun()
        with card_col:
            st.markdown(f"""
            <div class="point-card queue-browser-card">
                <div>
                    <div class="queue-browser-top">
                        <div>
                            <div class="field-kicker">Pending Point</div>
                            <div class="queue-browser-title">Point {queue_point['sequence_num']:02d} - {queue_point['title']}</div>
                        </div>
                        <div class="queue-browser-index">
                            <span>{queue_index + 1} of {len(pending_queue)}</span>
                            <span>{'current target' if queue_is_selected else 'pending'}</span>
                        </div>
                    </div>
                    <div class="queue-browser-meta">
                        <span>{queue_point['image_name']}</span>
                        <span>{queue_point['assignment_date']}</span>
                        <span>{_normalize_travel_mode(queue_point['travel_mode'])}</span>
                        <span>Point #{queue_point['point_num']}</span>
                    </div>
                </div>
                <div>
                    <div class="queue-browser-coords">{queue_point['latitude']:.7f}, {queue_point['longitude']:.7f}</div>
                    <div class="queue-browser-note">Use the side arrows to browse the pending queue without stacking multiple cards on the screen.</div>
                </div>
            </div>
            """, unsafe_allow_html=True)
        with next_col:
            st.markdown("<div style='height:5.2rem'></div>", unsafe_allow_html=True)
            if st.button(
                "→",
                key="field_queue_next",
                use_container_width=True,
                disabled=len(pending_queue) <= 1,
            ):
                next_index = (queue_index + 1) % len(pending_queue)
                st.session_state.field_queue_point_id = pending_queue[next_index]["assignment_point_id"]
                st.rerun()

        action_col1, action_col2, action_col3 = st.columns(3, gap="small")
        with action_col1:
            if queue_is_selected:
                st.button(
                    "Current Target",
                    key=f"field_nav_target_{queue_point['assignment_point_id']}",
                    use_container_width=True,
                    disabled=True,
                )
            else:
                if st.button(
                    "Make Current Target",
                    key=f"field_nav_target_{queue_point['assignment_point_id']}",
                    use_container_width=True,
                ):
                    st.session_state.field_nav_point_id = queue_point["assignment_point_id"]
                    st.session_state.field_queue_point_id = queue_point["assignment_point_id"]
                    st.rerun()
        with action_col2:
            if st.button(
                "Mark Complete",
                key=f"field_complete_{queue_point['assignment_point_id']}",
                use_container_width=True,
            ):
                if st.session_state.field_nav_point_id == queue_point["assignment_point_id"]:
                    st.session_state.field_nav_point_id = None
                update_assignment_point_status(queue_point["assignment_point_id"], "completed")
                st.rerun()
        with action_col3:
            if st.button(
                "Skip Point",
                key=f"field_skip_{queue_point['assignment_point_id']}",
                use_container_width=True,
            ):
                if st.session_state.field_nav_point_id == queue_point["assignment_point_id"]:
                    st.session_state.field_nav_point_id = None
                update_assignment_point_status(queue_point["assignment_point_id"], "skipped")
                st.rerun()

    if history_queue:
        with st.expander("History: completed and skipped points", expanded=False):
            for point in history_queue:
                history_status = point["assignment_status"]
                st.markdown(f"""
                <div class="point-card">
                    <div class="queue-item-header">
                        <div>
                            <div class="field-kicker">History Point</div>
                            <div class="queue-item-title">Point {point['sequence_num']:02d} - {point['title']}</div>
                        </div>
                        <div class="point-status {history_status}">{history_status}</div>
                    </div>
                    <div class="queue-item-copy">{point['image_name']} | {point['assignment_date']} | {_normalize_travel_mode(point['travel_mode'])}</div>
                    <div class="queue-item-coords">{point['latitude']:.7f}, {point['longitude']:.7f}</div>
                </div>
                """, unsafe_allow_html=True)
                history_col1, history_col2 = st.columns(2, gap="small")
                with history_col1:
                    if st.button(
                        "Set As Current Target",
                        key=f"history_nav_target_{point['assignment_point_id']}",
                        use_container_width=True,
                    ):
                        st.session_state.field_nav_point_id = point["assignment_point_id"]
                        st.rerun()
                with history_col2:
                    if st.button(
                        "Reset To Pending",
                        key=f"field_reset_{point['assignment_point_id']}",
                        use_container_width=True,
                    ):
                        update_assignment_point_status(point["assignment_point_id"], "pending")
                        st.rerun()


if __name__ == "__main__":
    main()
