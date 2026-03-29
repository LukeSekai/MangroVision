"""
MangroVision - AI-Powered Mangrove Planting Zone Analyzer
Beautiful Streamlit UI for the thesis project
"""

import sys
import math
import re
import copy
# Force UTF-8 output so emoji in print() don't crash on Windows (cp1252 terminals)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import streamlit as st
import cv2
import numpy as np
import pandas as pd
import inspect
from PIL import Image
from pathlib import Path
import io
import json
import base64
from datetime import datetime
import time
from urllib.parse import urlencode
import streamlit_folium as st_folium
import folium
from branca.element import Element
from pyproj import Transformer

# Add canopy_detection to path
sys.path.append(str(Path(__file__).parent / "canopy_detection"))

from canopy_detection.canopy_detector_hexagon import HexagonDetector
from canopy_detection.exif_extractor import ExifExtractor
from canopy_detection.ortho_matcher import (
    match_drone_to_ortho,
    drone_pixel_to_gps_via_homography,
    drone_pixel_to_gps_via_heading,
    select_orthophoto,
    is_inside_any_orthophoto,
)
from canopy_detection.forbidden_zone_filter import ForbiddenZoneFilter
from planting_database import (
    save_analysis, find_overlapping_analyses, count_nearby_points,
    get_all_stats, delete_analysis,
    authenticate_user, ensure_admin_user, update_last_login,
    create_user_session, get_user_by_session_token, revoke_user_session,
    list_planters, get_planter_dashboard_stats,
    list_planter_assignment_map_points, assign_planting_point_to_planter,
    list_planter_assignments,
    get_planter_field_points, update_assignment_point_status,
    get_assignment_points, archive_planter_assignment, delete_planter_assignment,
)
from waypoint_export import (
    generate_gpx, generate_kml, generate_geojson, hexagons_to_waypoints,
)

# Load forbidden zones (towers, bridges, houses) once at startup
_FORBIDDEN_ZONES_PATH = Path(__file__).parent / "forbidden_zones.geojson"
_forbidden_filter = ForbiddenZoneFilter(str(_FORBIDDEN_ZONES_PATH))

# Load eroded zones (user-drawn erosion areas) once at startup
_ERODED_ZONES_PATH = Path(__file__).parent / "eroded_zones.geojson"
_eroded_filter = ForbiddenZoneFilter(str(_ERODED_ZONES_PATH))  # Reuse same class
_LOGIN_BG_PATH = Path(__file__).parent / "assets" / "mangrovebg.jpg"


# Page configuration
st.set_page_config(
    page_title="MangroVision",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for beautiful modern design
st.markdown("""
<style>
    /* Main color scheme */
    :root {
        --primary-green: #0b120d;
        --secondary-green: #132119;
        --accent-green: #24563c;
        --highlight-green: #78ca95;
        --light-green: #eef3ef;
        --background: #edf1ee;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #0b120d 0%, #173728 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .main-header h1 {
        color: white;
        font-size: 2.8rem;
        font-weight: 700;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.2);
    }
    
    .main-header p {
        color: #d6eadb;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #08100b 0%, #102018 45%, #173728 100%);
        border-right: 1px solid rgba(255,255,255,0.08);
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }
    
    section[data-testid="stSidebar"] h3 {
        color: #ccebd7 !important;
        font-weight: 600;
    }
    
    section[data-testid="stSidebar"] label {
        color: #eff8f1 !important;
        font-weight: 500;
    }
    
    section[data-testid="stSidebar"] p {
        color: #d0e3d7 !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #eef7f0 !important;
    }

    /* Sidebar select boxes and inputs */
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: rgba(255, 255, 255, 0.08);
        color: #eef7f0;
    }
    
    section[data-testid="stSidebar"] input {
        background-color: rgba(255, 255, 255, 0.08);
        color: #eef7f0;
    }
    
    /* Sidebar help text */
    section[data-testid="stSidebar"] .stTooltipIcon {
        color: #7EC88D !important;
    }
    
    /* Sidebar slider labels */
    section[data-testid="stSidebar"] .stSlider > div > div > div {
        color: #C8E6C9 !important;
    }
    
    /* Card styling */
    .stat-card {
        background: white;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 2px 8px rgba(0,0,0,0.08);
        border-left: 4px solid #4A9D6F;
        margin-bottom: 1rem;
    }
    
    .stat-value {
        font-size: 2rem;
        font-weight: 700;
        color: #2D5F3F;
        margin: 0;
    }
    
    .stat-label {
        font-size: 0.9rem;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    /* Upload area */
    .uploadedFile {
        border: 2px dashed #4A9D6F;
        border-radius: 10px;
        padding: 2rem;
        background: #F5F7F5;
    }
    
    /* Button styling */
    .stButton > button {
        background: linear-gradient(135deg, #173728 0%, #24563c 100%);
        color: white;
        border: none;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        width: 100%;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 10px 18px rgba(23, 55, 40, 0.18);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #132119 0%, #24563c 100%);
        color: white;
        border: none;
        padding: 0.75rem 1.5rem;
        font-size: 1rem;
        font-weight: 600;
        border-radius: 8px;
        width: 100%;
    }
    
    /* Image container */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
    }
    
    /* Results section */
    .results-header {
        color: #2D5F3F;
        font-size: 1.8rem;
        font-weight: 700;
        margin: 2rem 0 1rem 0;
        border-bottom: 3px solid #4A9D6F;
        padding-bottom: 0.5rem;
    }
    
    /* Info box */
    .info-box {
        background: linear-gradient(135deg, #E3F2FD 0%, #BBDEFB 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #2196F3;
        margin: 1rem 0;
        color: #1565C0 !important;
    }
    
    .info-box strong {
        color: #0D47A1 !important;
    }
    
    /* Success box */
    .success-box {
        background: linear-gradient(135deg, #E8F5E9 0%, #C8E6C9 100%);
        padding: 1rem 1.5rem;
        border-radius: 10px;
        border-left: 4px solid #4CAF50;
        margin: 1rem 0;
        color: #2E7D32 !important;
    }
    
    .success-box strong {
        color: #1B5E20 !important;
    }
    
    /* Metric styling */
    div[data-testid="stMetricValue"] {
        font-size: 2rem;
        color: #1e4936;
        font-weight: 700;
    }
    
    div[data-testid="stMetricLabel"] {
        color: #666;
        font-weight: 600;
    }
    
    /* Divider */
    hr {
        margin: 2rem 0;
        border: none;
        height: 2px;
        background: linear-gradient(90deg, transparent, #4A9D6F, transparent);
    }
    
    /* Sidebar dividers */
    section[data-testid="stSidebar"] hr {
        background: linear-gradient(90deg, transparent, #4A9D6F, transparent);
        opacity: 0.5;
    }

    /* Form submit button (Sign In) should match green theme */
    .stFormSubmitButton > button {
        background: linear-gradient(135deg, #2D5F3F 0%, #4A9D6F 100%) !important;
        color: white !important;
        border: none !important;
        padding: 0.75rem 2rem;
        font-size: 1.1rem;
        font-weight: 600;
        border-radius: 8px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        transition: all 0.3s ease;
        width: 100%;
    }

    .stFormSubmitButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
        background: linear-gradient(135deg, #2D5F3F 0%, #4A9D6F 100%) !important;
    }

    .stFormSubmitButton > button:focus,
    .stFormSubmitButton > button:active {
        outline: none !important;
        border: 1px solid #7EC88D !important;
        box-shadow: 0 0 0 2px rgba(126, 200, 141, 0.35) !important;
        background: linear-gradient(135deg, #2D5F3F 0%, #4A9D6F 100%) !important;
    }

    /* Auth panel */
    .login-hero {
        padding: 2.1rem 2.15rem;
        border-radius: 30px;
        background: linear-gradient(165deg, rgba(8, 16, 11, 0.96) 0%, rgba(16, 32, 24, 0.94) 100%);
        border: 1px solid rgba(143, 211, 167, 0.12);
        box-shadow: 0 28px 50px rgba(8, 16, 11, 0.32);
        min-height: 100%;
    }

    .login-kicker {
        color: #98cfac;
        text-transform: uppercase;
        letter-spacing: 0.18rem;
        font-size: 0.78rem;
        font-weight: 800;
        margin-bottom: 0.6rem;
    }

    .login-hero h1 {
        color: #f3faf5;
        font-size: 3rem;
        line-height: 1.02;
        margin: 0;
    }

    .login-hero p {
        color: #cadbcc;
        font-size: 1rem;
        line-height: 1.7;
        margin: 0.9rem 0 0 0;
        max-width: 640px;
    }

    .login-badge-row {
        display: flex;
        flex-wrap: wrap;
        gap: 0.55rem;
        margin-top: 1.1rem;
    }

    .login-badge-row span {
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(143, 211, 167, 0.12);
        color: #eef7f1;
        border-radius: 999px;
        padding: 0.48rem 0.8rem;
        font-size: 0.82rem;
        font-weight: 700;
    }

    .login-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 0.95rem;
        margin-top: 1.25rem;
    }

    .login-grid-item {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(143, 211, 167, 0.10);
        border-radius: 18px;
        padding: 1rem;
    }

    .login-grid-item strong {
        display: block;
        color: #f1f8f3;
        font-size: 0.98rem;
        font-weight: 800;
        margin-bottom: 0.28rem;
    }

    .login-grid-item span {
        color: #c6dbcd;
        font-size: 0.92rem;
        line-height: 1.55;
    }

    .auth-shell {
        padding: 1.35rem;
        border-radius: 28px;
        background: linear-gradient(165deg, rgba(8, 16, 11, 0.96) 0%, rgba(16, 32, 24, 0.94) 100%);
        border: 1px solid rgba(143, 211, 167, 0.12);
        margin-bottom: 0.9rem;
        box-shadow: 0 24px 44px rgba(8, 16, 11, 0.30);
    }

    .auth-caption {
        color: #98cfac;
        text-transform: uppercase;
        letter-spacing: 0.16rem;
        font-size: 0.74rem;
        font-weight: 800;
        margin-bottom: 0.35rem;
    }

    .auth-title {
        color: #f3faf5;
        font-size: 2rem;
        font-weight: 800;
        margin: 0;
    }

    .auth-subtitle {
        color: #cadbcc;
        font-size: 0.97rem;
        margin-top: 0.28rem;
        margin-bottom: 0.9rem;
        line-height: 1.6;
    }

    .auth-time {
        color: #eef7f1;
        font-weight: 700;
        background: rgba(255, 255, 255, 0.06);
        border: 1px solid rgba(143, 211, 167, 0.12);
        border-radius: 14px;
        padding: 0.62rem 0.78rem;
        margin-bottom: 0.75rem;
    }

    /* Opaque login form card */
    div[data-testid="stForm"] {
        background: rgba(255, 255, 255, 0.04) !important;
        border: 1px solid rgba(143, 211, 167, 0.10) !important;
        border-radius: 18px !important;
        padding: 1rem !important;
    }

    div[data-testid="stForm"] label,
    div[data-testid="stForm"] p,
    div[data-testid="stForm"] span {
        color: #eef7f1 !important;
    }

    div[data-testid="stForm"] [data-baseweb="input"] {
        background: rgba(255, 255, 255, 0.06) !important;
        border-color: rgba(143, 211, 167, 0.14) !important;
    }

    div[data-testid="stForm"] input {
        color: #eef7f1 !important;
    }

    .auth-stage {
        text-align: center;
        color: #eef7f1;
        padding: 1.35rem 1rem;
        border-radius: 20px;
        background: linear-gradient(165deg, rgba(8, 16, 11, 0.96) 0%, rgba(16, 32, 24, 0.94) 100%);
        border: 1px solid rgba(143, 211, 167, 0.12);
        box-shadow: 0 24px 44px rgba(8, 16, 11, 0.30);
    }

    .account-card {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(126, 200, 141, 0.18);
        color: #eef7f0;
        padding: 0.95rem 0.9rem;
        border-radius: 18px;
        margin-bottom: 0.65rem;
    }

    .account-kicker {
        color: #b9eac8;
        text-transform: uppercase;
        letter-spacing: 0.13rem;
        font-size: 0.7rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }

    .account-name {
        color: #ffffff;
        font-size: 1.02rem;
        font-weight: 800;
        margin: 0;
    }

    .account-caption {
        color: #d7ece0;
        font-size: 0.84rem;
        line-height: 1.5;
        margin-top: 0.28rem;
    }

    /* Login input polish */
    div[data-testid="stTextInput"] input:focus,
    div[data-testid="stTextInput"] input[aria-invalid="true"] {
        border-color: #4A9D6F !important;
        box-shadow: 0 0 0 1px #4A9D6F !important;
        outline: none !important;
    }

    div[data-testid="stTextInput"] [data-baseweb="input"]:focus-within {
        border-color: #4A9D6F !important;
        box-shadow: 0 0 0 1px #4A9D6F !important;
    }

    /* Workspace shell */
    html, body, [data-testid="stAppViewContainer"], [data-testid="stAppViewContainer"] > .main {
        font-family: "Aptos", "Trebuchet MS", sans-serif;
    }

    [data-testid="stAppViewContainer"] {
        background:
            radial-gradient(circle at top center, rgba(36, 86, 60, 0.28) 0%, rgba(36, 86, 60, 0.12) 24%, rgba(237, 241, 238, 0) 44%),
            linear-gradient(180deg, #08100b 0%, #0f1d15 20%, #183126 34%, #395443 48%, #cad7ce 64%, #e7ede8 76%, #edf1ee 100%);
    }

    [data-testid="stHeader"],
    [data-testid="stToolbar"],
    [data-testid="stDecoration"] {
        background: transparent !important;
    }

    [data-testid="stAppViewContainer"] [data-testid="block-container"] {
        max-width: 1480px;
        padding-top: 1.35rem;
        padding-bottom: 2.25rem;
    }

    .main-header {
        background: linear-gradient(135deg, #08100b 0%, #102018 46%, #173728 100%);
        padding: 1.55rem 1.85rem;
        border-radius: 28px;
        margin-bottom: 1rem;
        box-shadow: 0 28px 50px rgba(8, 16, 11, 0.34);
        border: 1px solid rgba(195, 222, 203, 0.10);
    }

    .main-header-grid {
        display: grid;
        grid-template-columns: minmax(0, 1fr) auto;
        align-items: start;
        gap: 1.2rem;
    }

    .main-kicker {
        color: rgba(198, 226, 207, 0.86);
        text-transform: uppercase;
        letter-spacing: 0.18rem;
        font-size: 0.78rem;
        font-weight: 700;
        margin-bottom: 0.55rem;
    }

    .main-header h1 {
        font-size: 2.55rem;
        line-height: 1.08;
        margin-bottom: 0.35rem;
        text-shadow: none;
    }

    .main-header p {
        color: rgba(226, 239, 230, 0.92);
        font-size: 0.96rem;
        line-height: 1.62;
        max-width: 690px;
        margin: 0;
    }

    .header-meta {
        display: flex;
        flex-direction: column;
        gap: 0.55rem;
        align-items: flex-end;
        justify-content: flex-start;
        justify-self: end;
        max-width: none;
    }

    .header-badge {
        background: rgba(255, 255, 255, 0.12);
        color: #f2fbf5;
        border: 1px solid rgba(255, 255, 255, 0.16);
        padding: 0.55rem 0.8rem;
        border-radius: 999px;
        font-size: 0.84rem;
        font-weight: 600;
        backdrop-filter: blur(8px);
        width: max-content;
        max-width: 100%;
    }

    .main-header-notes {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        margin-top: 0.95rem;
    }

    .main-header-notes span {
        background: rgba(255, 255, 255, 0.08);
        color: #e4f3e8;
        border: 1px solid rgba(143, 211, 167, 0.14);
        border-radius: 999px;
        padding: 0.46rem 0.72rem;
        font-size: 0.79rem;
        font-weight: 700;
        line-height: 1.2;
    }

    .section-hero {
        background: linear-gradient(160deg, rgba(11, 18, 13, 0.96) 0%, rgba(19, 33, 25, 0.95) 100%);
        border: 1px solid rgba(120, 202, 149, 0.10);
        border-radius: 24px;
        padding: 1.2rem 1.35rem;
        margin: 0.4rem 0 1rem 0;
        box-shadow: 0 20px 40px rgba(8, 16, 11, 0.24);
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 1rem;
        flex-wrap: wrap;
    }

    .section-kicker {
        color: #8fd3a7;
        text-transform: uppercase;
        letter-spacing: 0.16rem;
        font-size: 0.76rem;
        font-weight: 800;
        margin-bottom: 0.42rem;
    }

    .section-hero h2 {
        color: #f1f8f3;
        margin: 0;
        font-size: 1.8rem;
        font-weight: 800;
    }

    .section-hero p {
        color: #cadbcc;
        margin: 0.42rem 0 0 0;
        max-width: 760px;
        line-height: 1.6;
    }

    .section-hero-badges {
        display: flex;
        gap: 0.5rem;
        flex-wrap: wrap;
        justify-content: flex-end;
    }

    .section-hero-badges span {
        background: rgba(255, 255, 255, 0.06);
        color: #e4f3e8;
        border: 1px solid rgba(143, 211, 167, 0.12);
        border-radius: 999px;
        padding: 0.48rem 0.78rem;
        font-size: 0.82rem;
        font-weight: 700;
    }

    .workspace-card,
    .preview-card {
        background: linear-gradient(160deg, rgba(11, 18, 13, 0.96) 0%, rgba(19, 33, 25, 0.95) 100%);
        border: 1px solid rgba(120, 202, 149, 0.10);
        border-radius: 24px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 20px 40px rgba(8, 16, 11, 0.22);
        margin-bottom: 1rem;
    }

    .control-card {
        background: rgba(255, 255, 255, 0.95);
        border: 1px solid rgba(19, 33, 25, 0.10);
        border-radius: 24px;
        padding: 1.15rem 1.25rem;
        box-shadow: 0 18px 38px rgba(19, 41, 32, 0.08);
        margin-bottom: 1rem;
    }

    .operations-strip {
        background: linear-gradient(160deg, rgba(11, 18, 13, 0.96) 0%, rgba(19, 33, 25, 0.95) 100%);
        border: 1px solid rgba(120, 202, 149, 0.10);
        border-radius: 24px;
        padding: 1rem 1.2rem;
        margin: 1rem 0 1rem 0;
        box-shadow: 0 20px 40px rgba(8, 16, 11, 0.22);
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.9rem;
        flex-wrap: wrap;
    }

    .operations-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
    }

    .operations-meta span {
        background: rgba(255, 255, 255, 0.06);
        color: #eef7f1;
        border: 1px solid rgba(143, 211, 167, 0.10);
        border-radius: 999px;
        padding: 0.42rem 0.72rem;
        font-size: 0.82rem;
        font-weight: 700;
    }

    .control-card .operations-meta span {
        background: #eef4f0;
        color: #173728;
        border: 1px solid rgba(19, 33, 25, 0.08);
    }

    .panel-kicker {
        color: #2f7252;
        text-transform: uppercase;
        letter-spacing: 0.14rem;
        font-size: 0.74rem;
        font-weight: 800;
        margin-bottom: 0.4rem;
    }

    .panel-title {
        color: #1e4936;
        font-size: 1.45rem;
        font-weight: 800;
        margin: 0;
    }

    .panel-copy {
        color: #54695e;
        line-height: 1.6;
        margin: 0.45rem 0 0 0;
    }

    .workspace-card .panel-kicker,
    .preview-card .panel-kicker,
    .operations-strip .panel-kicker {
        color: #8fd3a7;
    }

    .workspace-card .panel-title,
    .preview-card .panel-title,
    .operations-strip .panel-title {
        color: #f1f8f3;
    }

    .workspace-card .panel-copy,
    .preview-card .panel-copy,
    .operations-strip .panel-copy {
        color: #cadbcc;
    }

    .panel-caption {
        color: #667b73;
        font-size: 0.9rem;
        line-height: 1.6;
    }

    .layer-legend {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.85rem;
    }

    .layer-pill {
        display: inline-flex;
        align-items: center;
        gap: 0.4rem;
        padding: 0.44rem 0.72rem;
        border-radius: 999px;
        font-size: 0.82rem;
        font-weight: 700;
        border: 1px solid rgba(19, 33, 25, 0.10);
        background: rgba(255, 255, 255, 0.82);
        color: #173728;
    }

    .layer-dot {
        width: 0.62rem;
        height: 0.62rem;
        border-radius: 50%;
        display: inline-block;
    }

    .system-list {
        margin: 0.55rem 0 0 0;
        padding-left: 1rem;
        color: #486159;
        line-height: 1.75;
    }

    .system-list li {
        margin-bottom: 0.15rem;
    }

    .empty-preview {
        border: 1px dashed rgba(40, 92, 68, 0.24);
        border-radius: 20px;
        padding: 1.5rem 1.2rem;
        background: linear-gradient(135deg, #f6faf7 0%, #edf4f0 100%);
        color: #52675f;
    }

    .sidebar-brand {
        background: linear-gradient(155deg, #09100b 0%, #102018 55%, #173728 100%);
        padding: 1.2rem 1rem;
        border-radius: 18px;
        border: 1px solid rgba(157, 226, 189, 0.18);
        margin-bottom: 1rem;
    }

    .sidebar-brand h2 {
        color: white;
        margin: 0;
        font-size: 1.55rem;
        font-weight: 800;
    }

    .sidebar-brand p {
        color: #dbeee1 !important;
        margin: 0.35rem 0 0 0;
        font-size: 0.9rem;
    }

    .sidebar-brand span {
        color: #a8dfbb;
        text-transform: uppercase;
        letter-spacing: 0.14rem;
        font-size: 0.72rem;
        font-weight: 800;
    }

    section[data-testid="stFileUploader"] {
        background: linear-gradient(160deg, #0d1610 0%, #16271d 100%);
        border-radius: 18px;
        padding: 0.35rem;
        border: 1px dashed rgba(120, 202, 149, 0.24);
    }

    section[data-testid="stFileUploader"] * {
        color: #eef7f1 !important;
    }

    div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.94);
        border: 1px solid rgba(29, 75, 49, 0.08);
        border-radius: 18px;
        padding: 0.9rem 1rem;
        box-shadow: 0 12px 28px rgba(21, 45, 35, 0.08);
    }

    section[data-testid="stSidebar"] div[data-testid="stMetric"] {
        background: rgba(255, 255, 255, 0.08);
        border: 1px solid rgba(157, 226, 189, 0.14);
        box-shadow: none;
    }

    section[data-testid="stSidebar"] div[data-testid="stMetricLabel"],
    section[data-testid="stSidebar"] div[data-testid="stMetricValue"] {
        color: #f1faf4 !important;
    }

    section[data-testid="stSidebar"] .stButton > button {
        justify-content: flex-start;
        min-height: 3.5rem;
        background: rgba(255, 255, 255, 0.08);
        color: #f4fbf5;
        border: 1px solid rgba(223, 245, 230, 0.12);
        border-radius: 18px;
        box-shadow: none;
        padding: 0.85rem 1rem;
        font-size: 1rem;
    }

    section[data-testid="stSidebar"] .stButton > button:hover {
        background: rgba(255, 255, 255, 0.12);
        transform: none;
        box-shadow: none;
    }

    section[data-testid="stSidebar"] .stButton > button[kind="primary"] {
        background: linear-gradient(135deg, #1f4a34 0%, #2a6748 100%);
        border-color: rgba(143, 211, 167, 0.28);
        box-shadow: 0 14px 24px rgba(8, 16, 11, 0.22);
    }

    .time-chip {
        margin-top: 0.55rem;
        margin-bottom: 0.9rem;
        padding: 0.7rem 0.9rem;
        border-radius: 16px;
        background: #f4f8f5;
        border: 1px solid rgba(19, 33, 25, 0.10);
        color: #173728;
        box-shadow: 0 12px 26px rgba(19, 41, 32, 0.06);
        width: fit-content;
    }

    .time-chip span {
        display: block;
        font-size: 0.72rem;
        text-transform: uppercase;
        letter-spacing: 0.11rem;
        color: #5d7668;
        font-weight: 800;
        margin-bottom: 0.16rem;
    }

    .time-chip strong {
        font-size: 0.98rem;
    }

    .manager-card,
    .field-card {
        background: linear-gradient(160deg, rgba(11, 18, 13, 0.96) 0%, rgba(19, 33, 25, 0.95) 100%);
        border: 1px solid rgba(120, 202, 149, 0.10);
        border-radius: 24px;
        padding: 1.1rem 1.2rem;
        box-shadow: 0 18px 36px rgba(8, 16, 11, 0.20);
        margin-bottom: 1rem;
    }

    .manager-card h3,
    .field-card h3 {
        color: #f1f8f3;
        font-size: 1.35rem;
        font-weight: 800;
        margin: 0;
    }

    .manager-card p,
    .field-card p {
        color: #cadbcc;
        line-height: 1.6;
        margin: 0.45rem 0 0 0;
    }

    .manager-meta,
    .field-meta {
        display: flex;
        flex-wrap: wrap;
        gap: 0.5rem;
        margin-top: 0.85rem;
    }

    .manager-meta span,
    .field-meta span {
        background: rgba(255, 255, 255, 0.06);
        color: #eef7f1;
        border: 1px solid rgba(143, 211, 167, 0.10);
        border-radius: 999px;
        padding: 0.38rem 0.7rem;
        font-size: 0.8rem;
        font-weight: 700;
    }

    .field-card.is-pending {
        border-color: rgba(143, 211, 167, 0.16);
    }

    .field-card.is-completed {
        border-color: rgba(126, 200, 141, 0.30);
        background: linear-gradient(160deg, rgba(13, 31, 21, 0.96) 0%, rgba(27, 55, 38, 0.94) 100%);
    }

    .field-card.is-skipped {
        border-color: rgba(239, 108, 0, 0.28);
        background: linear-gradient(160deg, rgba(32, 22, 12, 0.96) 0%, rgba(52, 33, 12, 0.94) 100%);
    }

    .field-title-row {
        display: flex;
        justify-content: space-between;
        align-items: flex-start;
        gap: 0.85rem;
        flex-wrap: wrap;
    }

    .field-status {
        border-radius: 999px;
        padding: 0.38rem 0.72rem;
        font-size: 0.78rem;
        font-weight: 800;
        text-transform: uppercase;
        letter-spacing: 0.08rem;
    }

    .field-status.pending {
        background: rgba(120, 202, 149, 0.14);
        color: #b7e4c5;
    }

    .field-status.completed {
        background: rgba(126, 200, 141, 0.18);
        color: #d6f3de;
    }

    .field-status.skipped {
        background: rgba(239, 108, 0, 0.18);
        color: #ffd2ac;
    }

    .field-coords {
        color: #eef7f1;
        font-size: 1rem;
        font-weight: 700;
        margin-top: 0.75rem;
    }

    .field-caption {
        color: #bcd0c1;
        font-size: 0.9rem;
        line-height: 1.55;
        margin-top: 0.4rem;
    }

    .field-divider {
        height: 1px;
        background: linear-gradient(90deg, rgba(143, 211, 167, 0), rgba(143, 211, 167, 0.22), rgba(143, 211, 167, 0));
        margin: 0.85rem 0 0.9rem 0;
    }

    @media (max-width: 900px) {
        .main-header h1,
        .login-hero h1 {
            font-size: 2.05rem;
        }

        .login-grid {
            grid-template-columns: 1fr;
        }

        .main-header,
        .section-hero,
        .auth-shell,
        .login-hero,
        .manager-card,
        .field-card,
        .control-card,
        .preview-card,
        .operations-strip {
            padding: 1rem 1rem;
            border-radius: 22px;
        }

        .header-meta,
        .section-hero-badges,
        .operations-meta,
        .manager-meta,
        .field-meta {
            justify-content: flex-start;
        }

        .main-header-grid {
            grid-template-columns: 1fr;
        }

        .header-meta {
            align-items: flex-start;
            justify-self: start;
        }
    }

</style>
""", unsafe_allow_html=True)


_PLANNER_SESSION_QUERY_KEY = "planner_session"


def _set_auth_query(session_token: str | None):
    """Persist planner session token in the URL so browser refresh restores login."""
    try:
        if session_token:
            st.query_params[_PLANNER_SESSION_QUERY_KEY] = session_token
        else:
            st.query_params.pop(_PLANNER_SESSION_QUERY_KEY, None)
            st.query_params.pop("auth", None)
            st.query_params.pop("user", None)
    except Exception:
        pass


def _get_login_bg_data_uri() -> str:
    """Load local login placeholder image and return a data URI for CSS background-image."""
    try:
        image_bytes = _LOGIN_BG_PATH.read_bytes()
        ext = _LOGIN_BG_PATH.suffix.lower()
        mime = "image/svg+xml" if ext == ".svg" else "image/png"
        b64 = base64.b64encode(image_bytes).decode("ascii")
        return f"data:{mime};base64,{b64}"
    except Exception:
        # Fallback to a tiny embedded gradient SVG so login never looks blank.
        fallback_svg = (
            "<svg xmlns='http://www.w3.org/2000/svg' width='1200' height='520' viewBox='0 0 1200 520'>"
            "<defs><linearGradient id='g' x1='0' x2='1' y1='0' y2='1'>"
            "<stop offset='0%' stop-color='#173728'/><stop offset='100%' stop-color='#2f7650'/>"
            "</linearGradient></defs><rect width='1200' height='520' fill='url(#g)'/>"
            "<circle cx='210' cy='130' r='150' fill='rgba(126,200,141,0.18)'/>"
            "<circle cx='980' cy='390' r='180' fill='rgba(223,245,230,0.13)'/>"
            "</svg>"
        )
        b64 = base64.b64encode(fallback_svg.encode("utf-8")).decode("ascii")
        return f"data:image/svg+xml;base64,{b64}"


def _restore_auth_from_query():
    """Restore planner login state from a persisted session token."""
    if st.session_state.get('logged_in'):
        return
    try:
        session_token = st.query_params.get(_PLANNER_SESSION_QUERY_KEY, "")
    except Exception:
        return
    session_token = str(session_token or "").strip()
    if not session_token:
        return

    user = get_user_by_session_token(session_token)
    if user:
        st.session_state.logged_in = True
        st.session_state.user_id = user.get('id')
        st.session_state.username = user.get('full_name')
        st.session_state.last_login = user.get('last_login')
        st.session_state.auth_session_token = session_token
        return

    _set_auth_query(None)


def _clear_auth_state(revoke_session: bool = True):
    """Clear planner auth state and optionally revoke the persisted session."""
    session_token = st.session_state.get("auth_session_token")
    if not session_token:
        try:
            session_token = str(st.query_params.get(_PLANNER_SESSION_QUERY_KEY, "") or "").strip()
        except Exception:
            session_token = ""
    if revoke_session and session_token:
        revoke_user_session(session_token)

    st.session_state.logged_in = False
    st.session_state.user_id = None
    st.session_state.username = None
    st.session_state.last_login = None
    st.session_state.auth_session_token = None
    st.session_state.clear_login_fields = True
    st.session_state.show_login_success = False
    _set_auth_query(None)


def _init_auth_state():
    """Initialize authentication-related session keys."""
    if 'logged_in' not in st.session_state:
        st.session_state.logged_in = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'username' not in st.session_state:
        st.session_state.username = None
    if 'last_login' not in st.session_state:
        st.session_state.last_login = None
    if 'clear_login_fields' not in st.session_state:
        st.session_state.clear_login_fields = False
    if 'show_login_success' not in st.session_state:
        st.session_state.show_login_success = False
    if 'auth_session_token' not in st.session_state:
        st.session_state.auth_session_token = None
    _restore_auth_from_query()


def _render_login_screen() -> bool:
    """Render login UI; return True when authenticated."""
    _init_auth_state()
    if st.session_state.logged_in and not st.session_state.get('show_login_success', False):
        return True

    ensure_admin_user()
    now_str = datetime.now().strftime("%B %d, %Y | %I:%M:%S %p")
    login_bg_data_uri = _get_login_bg_data_uri()

    st.markdown(
        f"""
        <style>
            [data-testid="stAppViewContainer"] {{
                position: relative;
                overflow: hidden;
                isolation: isolate;
                background: transparent !important;
            }}

            [data-testid="stAppViewContainer"]::before {{
                content: "";
                position: fixed;
                inset: 0;
                background-image: url('{login_bg_data_uri}');
                background-size: cover;
                background-position: center;
                filter: blur(8px);
                transform: scale(1.06);
                z-index: -2;
                pointer-events: none;
            }}

            [data-testid="stAppViewContainer"]::after {{
                content: "";
                position: fixed;
                inset: 0;
                background: linear-gradient(135deg, rgba(6, 17, 14, 0.74) 0%, rgba(10, 23, 18, 0.68) 50%, rgba(17, 48, 33, 0.66) 100%);
                z-index: -1;
                pointer-events: none;
            }}

            [data-testid="stAppViewContainer"],
            [data-testid="stAppViewContainer"] > .main,
            [data-testid="stHeader"],
            [data-testid="stToolbar"] {{
                background: transparent !important;
                position: relative;
                z-index: 2;
            }}

            [data-testid="stAppViewContainer"] [data-testid="block-container"] {{
                position: relative;
                z-index: 3;
            }}
        </style>
        """,
        unsafe_allow_html=True,
    )

    if st.session_state.get('show_login_success', False):
        left, mid, right = st.columns([1, 1.12, 1])
        with mid:
            st.markdown(
                """
                <div class="auth-stage">
                    <h2 style="margin:0;">MangroVision</h2>
                    <p style="margin:0.4rem 0 0 0; color:#9fd0af;">Login successful</p>
                </div>
                """,
                unsafe_allow_html=True,
            )
        time.sleep(1.0)
        st.session_state.show_login_success = False
        st.rerun()
        return False

    hero_col, form_col = st.columns([1.12, 0.88], gap="large")

    with hero_col:
        st.markdown("""
        <div class="login-hero">
            <div class="login-kicker">MangroVision Planning System</div>
            <h1>Map-First Mangrove Planning Workspace</h1>
            <p>Authenticate to access the live orthophoto workspace for canopy analysis, exclusion zoning, planting layout generation, and field export.</p>
            <div class="login-badge-row">
                <span>Orthophoto overlays</span>
                <span>Exclusion zoning</span>
                <span>Field export ready</span>
            </div>
            <div class="login-grid">
                <div class="login-grid-item">
                    <strong>Operational mapping</strong>
                    <span>Review the planting site on the live map before you queue any drone frame for analysis.</span>
                </div>
                <div class="login-grid-item">
                    <strong>Layer-based review</strong>
                    <span>Check forbidden zones, eroded areas, analysis locations, and planting points directly from the workspace.</span>
                </div>
                <div class="login-grid-item">
                    <strong>Analysis workflow</strong>
                    <span>Run canopy processing only after the site is visually confirmed and the target image is staged.</span>
                </div>
                <div class="login-grid-item">
                    <strong>Deployment output</strong>
                    <span>Export planting coordinates for field navigation once the planting layout has been validated.</span>
                </div>
            </div>
        </div>
        """, unsafe_allow_html=True)

    with form_col:
        st.markdown(f"""
        <div class="auth-shell">
            <div class="auth-caption">Secure Access</div>
            <div class="auth-title">Workspace Login</div>
            <div class="auth-subtitle">Sign in to continue to the MangroVision operations dashboard.</div>
            <div class="auth-time">System time: {now_str}</div>
        </div>
        """, unsafe_allow_html=True)

        if st.session_state.get('clear_login_fields'):
            st.session_state.pop('login_username', None)
            st.session_state.pop('login_password', None)
            st.session_state.clear_login_fields = False

        with st.form("login_form", clear_on_submit=False):
            username = st.text_input("Username", placeholder="Enter username", key="login_username")
            password = st.text_input("Password", type="password", placeholder="Enter password", key="login_password")
            login_clicked = st.form_submit_button("Sign In", type="primary", use_container_width=True)

        if login_clicked:
            user = authenticate_user(username.strip(), password)
            if user:
                update_last_login(user['id'])
                session_token = create_user_session(user['id'])
                st.session_state.logged_in = True
                st.session_state.user_id = user['id']
                st.session_state.username = user['full_name']
                st.session_state.last_login = datetime.now().isoformat(timespec='seconds')
                st.session_state.auth_session_token = session_token
                st.session_state.clear_login_fields = True
                st.session_state.show_login_success = True
                _set_auth_query(session_token)
                st.rerun()
            else:
                st.error("Invalid username or password.")

    return False


def _render_user_panel():
    """Render a compact logged-in account panel in the sidebar."""
    st.markdown(
        f"""
        <div class="account-card">
            <div class="account-kicker">Account</div>
            <p class="account-name">{st.session_state.get('username', 'Unknown')}</p>
            <div class="account-caption">Authenticated access to the MangroVision planning workspace.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if st.button("Log Out", use_container_width=True):
        _clear_auth_state(revoke_session=True)
        st.rerun()


def _render_header_datetime_live():
    """Render realtime datetime (with seconds) for the main header."""
    def _clock_markup() -> str:
        now_str = datetime.now().strftime("%B %d, %Y | %I:%M:%S %p")
        return (
            "<div style='margin-top:0.55rem; margin-bottom:0.9rem; padding:0.65rem 0.9rem; "
            "border-radius:14px; background:rgba(255, 255, 255, 0.88); "
            "border:1px solid rgba(26, 70, 52, 0.10); color:#17392c; "
            "box-shadow:0 12px 26px rgba(19, 41, 32, 0.08); "
            f"font-weight:600; width:fit-content;'>🕒 {now_str}</div>"
        )

    # Re-render only this fragment every second when supported.
    if hasattr(st, "fragment"):
        @st.fragment(run_every="1s")
        def _clock_fragment():
            st.markdown(_clock_markup(), unsafe_allow_html=True)
        _clock_fragment()
    else:
        st.markdown(_clock_markup(), unsafe_allow_html=True)


def _render_workspace_clock():
    """Render a compact live clock for the analysis console."""
    def _clock_markup() -> str:
        now_str = datetime.now().strftime("%B %d, %Y | %I:%M:%S %p")
        return (
            "<div class='time-chip'>"
            "<span>Operations Time</span>"
            f"<strong>{now_str}</strong>"
            "</div>"
        )

    if hasattr(st, "fragment"):
        @st.fragment(run_every="1s")
        def _clock_fragment():
            st.markdown(_clock_markup(), unsafe_allow_html=True)
        _clock_fragment()
    else:
        st.markdown(_clock_markup(), unsafe_allow_html=True)


def _reload_eroded_filter():
    """Reload eroded zones from disk (called after saving new zones)"""
    global _eroded_filter
    _eroded_filter = ForbiddenZoneFilter(str(_ERODED_ZONES_PATH))


def _render_section_banner(kicker: str, title: str, subtitle: str, badges=None):
    """Render a reusable section banner for the workspace views."""
    badge_html = ""
    for badge in badges or []:
        badge_html += f"<span>{badge}</span>"

    st.markdown(
        f"""
        <div class="section-hero">
            <div>
                <div class="section-kicker">{kicker}</div>
                <h2>{title}</h2>
                <p>{subtitle}</p>
            </div>
            <div class="section-hero-badges">{badge_html}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _get_analysis_map_center(analyses):
    """Pick a sensible map center from saved analyses, or use the Leganes default."""
    lats = [a.get('center_lat') for a in analyses if a.get('center_lat') is not None]
    lons = [a.get('center_lon') for a in analyses if a.get('center_lon') is not None]
    if lats and lons:
        return [sum(lats) / len(lats), sum(lons) / len(lons)]
    return [10.7800, 122.6253]


def _add_operational_map_layers(map_obj, orthophoto_name: str = "Orthophoto Overlay"):
    """Add the shared basemap stack so WebODM tiles sit above a satellite fallback."""
    folium.TileLayer(
        tiles='https://mt1.google.com/vt/lyrs=s&x={x}&y={y}&z={z}',
        attr='Google Maps',
        name='Satellite Base',
        overlay=False,
        control=True,
        show=True,
        max_zoom=21,
    ).add_to(map_obj)

    folium.TileLayer(
        tiles="http://localhost:8080/FINAL%20MAP/{z}/{x}/{y}.jpg",
        attr='MangroVision Orthophoto | QGIS',
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


def _saved_point_marker_style(point_status: str | None) -> dict:
    """Return consistent map styling for saved planting-point states."""
    status = (point_status or "planned").strip().lower()
    if status == "planted":
        return {
            "border_color": "#F9A825",
            "fill_color": "#FFEE58",
            "label": "Planted",
        }
    if status == "skipped":
        return {
            "border_color": "#EF6C00",
            "fill_color": "#FFB74D",
            "label": "Skipped",
        }
    return {
        "border_color": "#1B5E20",
        "fill_color": "#4CAF50",
        "label": "Planned",
    }


def _build_workspace_overview_map(stats):
    """Build the overview map shown on the main planning workspace before upload."""
    from folium.plugins import Fullscreen

    analyses = stats.get('analyses', [])
    all_points = stats.get('points', [])
    center = _get_analysis_map_center(analyses)

    workspace_map = folium.Map(
        location=center,
        zoom_start=19 if analyses else 18,
        tiles=None,
        control_scale=True,
    )
    _add_operational_map_layers(workspace_map)

    if _forbidden_filter.forbidden_polygons:
        forbidden_group = folium.FeatureGroup(name='Forbidden Zones', show=True)
        for poly in _forbidden_filter.forbidden_polygons:
            coords = [(lat, lon) for lon, lat in poly.exterior.coords]
            folium.Polygon(
                locations=coords,
                color='#C62828',
                fill=True,
                fillColor='#E53935',
                fillOpacity=0.28,
                weight=2,
                tooltip='Forbidden Zone',
            ).add_to(forbidden_group)
        forbidden_group.add_to(workspace_map)

    if _eroded_filter.forbidden_polygons:
        eroded_group = folium.FeatureGroup(name='Eroded Zones', show=True)
        for poly in _eroded_filter.forbidden_polygons:
            coords = [(lat, lon) for lon, lat in poly.exterior.coords]
            folium.Polygon(
                locations=coords,
                color='#EF6C00',
                fill=True,
                fillColor='#FB8C00',
                fillOpacity=0.28,
                weight=2,
                tooltip='Eroded Zone',
            ).add_to(eroded_group)
        eroded_group.add_to(workspace_map)

    sorted_analyses = sorted(
        analyses,
        key=lambda row: row.get('analyzed_at') or "",
        reverse=True,
    )
    analysis_group = folium.FeatureGroup(name='Analysis Locations', show=True)
    for analysis in sorted_analyses[:12]:
        lat = analysis.get('center_lat')
        lon = analysis.get('center_lon')
        if lat is None or lon is None:
            continue
        folium.Marker(
            location=[lat, lon],
            popup=(
                f"<b>{analysis['image_name']}</b><br>"
                f"Captured: {analysis['analyzed_at']}<br>"
                f"Planting points: {analysis['hexagon_count']}<br>"
                f"Canopies: {analysis['canopy_count']}"
            ),
            tooltip=analysis['image_name'],
            icon=folium.Icon(color='blue', icon='camera', prefix='fa'),
        ).add_to(analysis_group)
    analysis_group.add_to(workspace_map)

    visible_points = list(all_points)
    if len(visible_points) > 450:
        step = max(1, int(np.ceil(len(visible_points) / 450)))
        visible_points = visible_points[::step]

    if visible_points:
        points_group = folium.FeatureGroup(name='Planting Zones', show=True)
        for point in visible_points:
            point_style = _saved_point_marker_style(point.get("status"))
            folium.CircleMarker(
                location=[point['latitude'], point['longitude']],
                radius=3,
                color=point_style["border_color"],
                fillColor=point_style["fill_color"],
                fillOpacity=0.82,
                weight=1,
                tooltip=f"{point['image_name']} · {point_style['label']}",
            ).add_to(points_group)
        points_group.add_to(workspace_map)

    Fullscreen(position="topleft", title="Expand map", title_cancel="Exit fullscreen").add_to(workspace_map)
    folium.LayerControl(collapsed=False).add_to(workspace_map)
    _style_layer_control(workspace_map)
    return workspace_map


def _build_google_maps_navigation_url(dest_lat, dest_lon, travel_mode="walking", origin_lat=None, origin_lon=None):
    """Build a Google Maps directions URL for one destination."""
    params = {
        "api": 1,
        "destination": f"{dest_lat:.7f},{dest_lon:.7f}",
        "travelmode": travel_mode or "walking",
    }
    if origin_lat is not None and origin_lon is not None:
        params["origin"] = f"{origin_lat:.7f},{origin_lon:.7f}"
    return "https://www.google.com/maps/dir/?" + urlencode(params)


def _render_navigation_button(label: str, url: str):
    """Render an external navigation button with a Streamlit fallback."""
    if hasattr(st, "link_button"):
        st.link_button(label, url, use_container_width=True, type="primary")
    else:
        st.markdown(f"[{label}]({url})")


def _assignment_points_to_waypoints(points, prefix: str):
    """Convert assignment point rows into export-ready waypoint dicts."""
    export_rows = []
    for point in points:
        export_rows.append({
            "point_num": point.get("sequence_num") or point.get("point_num"),
            "latitude": point["latitude"],
            "longitude": point["longitude"],
            "buffer_m": point.get("buffer_m"),
            "area_m2": point.get("area_m2"),
            "status": point.get("assignment_status") or point.get("status") or "pending",
        })
    return hexagons_to_waypoints(export_rows, image_name=prefix)


def _init_planter_management_state():
    """Initialize session state used by the interactive admin assignment map."""
    if "planter_management_selected_point_id" not in st.session_state:
        st.session_state.planter_management_selected_point_id = None
    if "planter_management_selected_planter_id" not in st.session_state:
        st.session_state.planter_management_selected_planter_id = None
    if "planter_management_last_map_event" not in st.session_state:
        st.session_state.planter_management_last_map_event = None
    if "planter_management_flash" not in st.session_state:
        st.session_state.planter_management_flash = None
    if "planter_management_planters" not in st.session_state:
        st.session_state.planter_management_planters = None
    if "planter_management_points" not in st.session_state:
        st.session_state.planter_management_points = None
    if "planter_management_points_by_id" not in st.session_state:
        st.session_state.planter_management_points_by_id = {}
    if "planter_management_assignments" not in st.session_state:
        st.session_state.planter_management_assignments = None
    if "planter_management_dashboard_stats" not in st.session_state:
        st.session_state.planter_management_dashboard_stats = None
    if "planter_management_points_version" not in st.session_state:
        st.session_state.planter_management_points_version = 0
    if "planter_management_base_map" not in st.session_state:
        st.session_state.planter_management_base_map = None
    if "planter_management_base_map_version" not in st.session_state:
        st.session_state.planter_management_base_map_version = None
    if "planter_management_pending_assignment" not in st.session_state:
        st.session_state.planter_management_pending_assignment = None
    if "planter_management_map_center" not in st.session_state:
        st.session_state.planter_management_map_center = None
    if "planter_management_map_zoom" not in st.session_state:
        st.session_state.planter_management_map_zoom = None


def _compute_planter_management_dashboard_stats(planters: list[dict], assignments: list[dict]) -> dict:
    """Compute dashboard counters from cached planter-management data."""
    active_planters = sum(1 for planter in planters if planter.get("status") == "active")
    active_assignments = sum(1 for assignment in assignments if assignment.get("status") == "active")
    pending_assigned_points = sum(
        int(assignment.get("pending_points") or 0)
        for assignment in assignments
        if assignment.get("status") == "active"
    )
    completed_assigned_points = sum(
        int(assignment.get("completed_points") or 0)
        for assignment in assignments
        if assignment.get("status") == "active"
    )
    return {
        "active_planters": active_planters,
        "active_assignments": active_assignments,
        "pending_assigned_points": pending_assigned_points,
        "completed_assigned_points": completed_assigned_points,
    }


def _refresh_planter_management_cache(
    *,
    reload_planters: bool = False,
    reload_points: bool = False,
    reload_assignments: bool = False,
    force: bool = False,
):
    """Load planter-management data into session state only when needed."""
    if force or reload_planters or st.session_state.planter_management_planters is None:
        st.session_state.planter_management_planters = list_planters()

    if force or reload_points or st.session_state.planter_management_points is None:
        points = list_planter_assignment_map_points()
        st.session_state.planter_management_points = points
        st.session_state.planter_management_points_by_id = {
            int(point["id"]): point for point in points
        }
        st.session_state.planter_management_points_version += 1

    if force or reload_assignments or st.session_state.planter_management_assignments is None:
        st.session_state.planter_management_assignments = list_planter_assignments()

    st.session_state.planter_management_dashboard_stats = _compute_planter_management_dashboard_stats(
        st.session_state.planter_management_planters or [],
        st.session_state.planter_management_assignments or [],
    )


def _invalidate_planter_management_base_map():
    """Force the cached admin assignment base map to rebuild on the next render."""
    st.session_state.planter_management_base_map = None
    st.session_state.planter_management_base_map_version = None


def _get_cached_planter_management_base_map(points: list[dict]):
    """Return a cached base map that only rebuilds when point data changes."""
    version = st.session_state.get("planter_management_points_version", 0)
    if (
        st.session_state.get("planter_management_base_map") is None
        or st.session_state.get("planter_management_base_map_version") != version
    ):
        st.session_state.planter_management_base_map = _build_planter_assignment_map(points)
        st.session_state.planter_management_base_map_version = version
    return st.session_state.planter_management_base_map


def _default_planter_management_map_view(points: list[dict]):
    """Return the default center and zoom for the admin assignment map."""
    if points:
        center_lats = [point.get("center_lat") for point in points if point.get("center_lat") is not None]
        center_lons = [point.get("center_lon") for point in points if point.get("center_lon") is not None]
        if center_lats and center_lons:
            return [sum(center_lats) / len(center_lats), sum(center_lons) / len(center_lons)], 19
        return [
            sum(float(point["latitude"]) for point in points) / len(points),
            sum(float(point["longitude"]) for point in points) / len(points),
        ], 19
    return [10.7800, 122.6253], 18


def _build_planter_assignment_selected_layers(selected_point: dict | None):
    """Return a lightweight highlight layer for the currently selected point."""
    if not selected_point:
        return []

    selected_group = folium.FeatureGroup(name="Selected Point", show=True)
    planting_status = (selected_point.get("planting_status") or "planned").strip().lower()
    status_label = "Planted" if planting_status == "planted" else "Selected"
    folium.CircleMarker(
        location=[float(selected_point["latitude"]), float(selected_point["longitude"])],
        radius=9,
        color="#F9A825",
        fill=True,
        fillColor="#FFEE58",
        fillOpacity=0.95,
        weight=3,
        tooltip=f"Point {int(selected_point['point_num']):03d} · {status_label}",
        popup=folium.Popup(
            f"<b>Point {int(selected_point['point_num']):03d}</b><br>{selected_point['image_name']}<br>{status_label}",
            max_width=320,
        ),
    ).add_to(selected_group)
    return [selected_group]


def _render_planter_assignment_map(
    base_map,
    *,
    selected_layers=None,
    center: list[float] | None = None,
    zoom: int | None = None,
):
    """Render the admin assignment map while preserving state when the installed component supports it."""
    returned_objects = [
        "last_clicked",
        "last_object_clicked",
        "last_object_clicked_popup",
        "center",
        "zoom",
    ]
    signature = inspect.signature(st_folium.st_folium)
    params = signature.parameters
    supports_advanced_overlay = all(
        name in params for name in ("feature_group_to_add", "center", "zoom", "layer_control")
    )

    render_kwargs = {
        "height": 620,
        "key": "planter_assignment_map",
        "returned_objects": returned_objects,
        "use_container_width": True,
    }

    if supports_advanced_overlay:
        if center is not None:
            render_kwargs["center"] = center
        if zoom is not None:
            render_kwargs["zoom"] = zoom
        if selected_layers:
            render_kwargs["feature_group_to_add"] = selected_layers
        render_kwargs["layer_control"] = folium.LayerControl(collapsed=False)
        return st_folium.st_folium(base_map, **render_kwargs)

    render_map = copy.deepcopy(base_map)
    for selected_layer in selected_layers or []:
        selected_layer.add_to(render_map)
    if center is not None:
        render_map.location = center
    if zoom is not None:
        render_map.options["zoom"] = zoom
        render_map.options["zoomStart"] = zoom
    folium.LayerControl(collapsed=False).add_to(render_map)
    _style_layer_control(render_map)
    return st_folium.st_folium(render_map, **render_kwargs)


def _queue_planter_management_assignment(point_id: int, planter_id: int, allow_reassign: bool):
    """Queue one assignment request so it can be processed before the next render."""
    st.session_state.planter_management_pending_assignment = {
        "point_id": int(point_id),
        "planter_id": int(planter_id),
        "allow_reassign": bool(allow_reassign),
    }


def _apply_planter_management_assignment_result(result: dict):
    """Update cached planter-management state after one successful assignment."""
    points_by_id = st.session_state.get("planter_management_points_by_id") or {}
    point = points_by_id.get(int(result["planting_point_id"]))
    if point:
        point["assigned_planter_id"] = result["planter_id"]
        point["assigned_planter_name"] = result["planter_name"]
        point["assignment_id"] = result["assignment_id"]
        point["assignment_title"] = result["assignment_title"]
        point["assignment_date"] = result["assignment_date"]
        point["assignment_status"] = result["assignment_status"]
        point["sequence_num"] = result["sequence_num"]

    planters = st.session_state.get("planter_management_planters") or []
    target_planter = next((planter for planter in planters if planter["id"] == result["planter_id"]), None)
    if target_planter:
        target_planter["pending_points"] = int(target_planter.get("pending_points") or 0) + 1
        if result.get("created_new_assignment"):
            target_planter["active_assignments"] = int(target_planter.get("active_assignments") or 0) + 1

    source_planter_id = result.get("source_planter_id")
    if source_planter_id is not None:
        source_planter = next((planter for planter in planters if planter["id"] == source_planter_id), None)
        if source_planter:
            source_status = (result.get("source_assignment_status") or "").strip().lower()
            if source_status == "pending":
                source_planter["pending_points"] = max(0, int(source_planter.get("pending_points") or 0) - 1)
            elif source_status == "completed":
                source_planter["completed_points"] = max(0, int(source_planter.get("completed_points") or 0) - 1)
            if result.get("source_assignment_deleted"):
                source_planter["active_assignments"] = max(0, int(source_planter.get("active_assignments") or 0) - 1)

    st.session_state.planter_management_points_version += 1
    _invalidate_planter_management_base_map()
    _refresh_planter_management_cache(reload_assignments=True)


def _process_pending_planter_management_assignment():
    """Process a queued assignment before rendering the admin assignment UI."""
    pending = st.session_state.get("planter_management_pending_assignment")
    if not pending:
        return

    try:
        result = assign_planting_point_to_planter(
            planter_id=int(pending["planter_id"]),
            planting_point_id=int(pending["point_id"]),
            assigned_by_user_id=st.session_state.get("user_id"),
            allow_reassign=bool(pending.get("allow_reassign")),
        )
        _apply_planter_management_assignment_result(result)
        if result["was_reassigned"]:
            message = (
                f"Point {result['point_num']:03d} was reassigned from "
                f"{result['reassigned_from_planter_name']} to {result['planter_name']}."
            )
        else:
            message = (
                f"Point {result['point_num']:03d} was assigned to "
                f"{result['planter_name']} in batch #{result['assignment_id']}."
            )
        st.session_state.planter_management_flash = ("success", message)
    except ValueError as err:
        st.session_state.planter_management_flash = ("warning", str(err))
    finally:
        st.session_state.planter_management_pending_assignment = None


def _point_click_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Approximate distance in meters between two close GPS points."""
    avg_lat = math.radians((lat1 + lat2) / 2.0)
    lat_m = (lat1 - lat2) * 111_320.0
    lon_m = (lon1 - lon2) * 111_320.0 * max(0.2, abs(math.cos(avg_lat)))
    return (lat_m * lat_m + lon_m * lon_m) ** 0.5


def _resolve_clicked_assignment_point(map_response: dict, points: list[dict]) -> tuple[int | None, str | None]:
    """Resolve the last clicked map marker to a planting point id."""
    if not isinstance(map_response, dict):
        return None, None

    object_clicked = map_response.get("last_object_clicked") or {}
    if object_clicked.get("lat") is not None and object_clicked.get("lng") is not None:
        clicked = object_clicked
    else:
        popup_html = map_response.get("last_object_clicked_popup")
        if popup_html:
            match = re.search(r'data-point-id="(\d+)"', str(popup_html))
            if match:
                point_id = int(match.group(1))
                return point_id, f"popup:{point_id}"
        clicked = map_response.get("last_clicked") or {}

    if clicked.get("lat") is None or clicked.get("lng") is None:
        return None, None

    clicked_lat = float(clicked["lat"])
    clicked_lon = float(clicked["lng"])
    nearest_point = None
    nearest_distance = None
    for point in points:
        distance_m = _point_click_distance_m(
            clicked_lat,
            clicked_lon,
            float(point["latitude"]),
            float(point["longitude"]),
        )
        if nearest_distance is None or distance_m < nearest_distance:
            nearest_distance = distance_m
            nearest_point = point

    if nearest_point and nearest_distance is not None and nearest_distance <= 6.0:
        point_id = int(nearest_point["id"])
        return point_id, f"coord:{point_id}:{round(clicked_lat, 7)}:{round(clicked_lon, 7)}"
    return None, None


def _build_planter_assignment_map(points: list[dict]):
    """Build the cached admin map used for assigning planting points."""
    from folium.plugins import Fullscreen

    if points:
        center_lats = [p.get("center_lat") for p in points if p.get("center_lat") is not None]
        center_lons = [p.get("center_lon") for p in points if p.get("center_lon") is not None]
        if center_lats and center_lons:
            map_center = [sum(center_lats) / len(center_lats), sum(center_lons) / len(center_lons)]
        else:
            map_center = [
                sum(float(p["latitude"]) for p in points) / len(points),
                sum(float(p["longitude"]) for p in points) / len(points),
            ]
    else:
        map_center = [10.7800, 122.6253]

    assignment_map = folium.Map(
        location=map_center,
        zoom_start=19 if points else 18,
        tiles=None,
        control_scale=True,
    )
    _add_operational_map_layers(assignment_map)

    unassigned_group = folium.FeatureGroup(name="Unassigned Points", show=True)
    assigned_group = folium.FeatureGroup(name="Assigned Points", show=True)
    planted_group = folium.FeatureGroup(name="Planted Points", show=True)

    for point in points:
        point_id = int(point["id"])
        assigned_planter = point.get("assigned_planter_name")
        planting_status = (point.get("planting_status") or "planned").strip().lower()

        radius = 5
        border_color = "#2E7D32"
        fill_color = "#4CAF50"
        marker_group = unassigned_group
        status_label = "Unassigned"

        if planting_status == "planted":
            radius = 6
            border_color = "#F9A825"
            fill_color = "#FFEE58"
            marker_group = planted_group
            status_label = "Planted"
        elif assigned_planter:
            radius = 6
            border_color = "#1565C0"
            fill_color = "#42A5F5"
            marker_group = assigned_group
            status_label = f"Assigned to {assigned_planter}"

        popup_html = f"""
        <div data-point-id="{point_id}">
            <b>Point {int(point['point_num']):03d}</b><br>
            {point['image_name']}<br>
            {float(point['latitude']):.7f}, {float(point['longitude']):.7f}<br>
            {status_label}
        </div>
        """

        folium.CircleMarker(
            location=[float(point["latitude"]), float(point["longitude"])],
            radius=radius,
            color=border_color,
            fill=True,
            fillColor=fill_color,
            fillOpacity=0.92,
            weight=2,
            tooltip=f"Point {int(point['point_num']):03d} · {status_label}",
            popup=folium.Popup(popup_html, max_width=320),
        ).add_to(marker_group)

    if points:
        unassigned_group.add_to(assignment_map)
        assigned_group.add_to(assignment_map)
        planted_group.add_to(assignment_map)

    Fullscreen(position="topleft", title="Expand map", title_cancel="Exit fullscreen").add_to(assignment_map)
    _style_layer_control(assignment_map)
    return assignment_map


def show_planter_management():
    """Planner-facing module for planter records and assignment creation."""
    _init_planter_management_state()
    _process_pending_planter_management_assignment()
    _refresh_planter_management_cache()

    dashboard_stats = st.session_state.get("planter_management_dashboard_stats") or {}
    planters = st.session_state.get("planter_management_planters") or []
    assignment_map_points = st.session_state.get("planter_management_points") or []
    assignment_points_by_id = st.session_state.get("planter_management_points_by_id") or {}
    assignments = st.session_state.get("planter_management_assignments") or []

    _render_section_banner(
        "Field Operations",
        "Planter Management",
        "Assign saved planting points from an interactive map, monitor planter workload, and keep field batches organized from the same operational console.",
        [
            f"{dashboard_stats['active_planters']} active planters",
            f"{dashboard_stats['active_assignments']} active assignments",
            f"{dashboard_stats['pending_assigned_points']} pending assigned points",
        ],
    )

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Active planters", dashboard_stats["active_planters"])
    m2.metric("Active assignments", dashboard_stats["active_assignments"])
    m3.metric("Pending assigned points", dashboard_stats["pending_assigned_points"])
    m4.metric("Completed assigned points", dashboard_stats["completed_assigned_points"])

    st.markdown("""
    <div class="manager-card">
        <div class="panel-kicker">Interactive Assignment Map</div>
        <h3>Assign Saved Planting Points</h3>
        <p>Select a planter first, then click a planting point on the map to review its details and assign or reassign it.</p>
    </div>
    """, unsafe_allow_html=True)
    st.info("Planter accounts are created in `field_app.py`. This admin panel now assigns saved points directly from the map.")

    flash_message = st.session_state.get("planter_management_flash")
    if flash_message:
        flash_kind, flash_text = flash_message
        if flash_kind == "success":
            st.success(flash_text)
        elif flash_kind == "warning":
            st.warning(flash_text)
        else:
            st.info(flash_text)
        st.session_state.planter_management_flash = None

    active_planters = [planter for planter in planters if planter["status"] == "active"]
    active_planter_ids = {planter["id"] for planter in active_planters}
    if (
        st.session_state.get("planter_management_selected_planter_id") is not None
        and st.session_state.get("planter_management_selected_planter_id") not in active_planter_ids
    ):
        st.session_state.planter_management_selected_planter_id = None

    selected_point_id = st.session_state.get("planter_management_selected_point_id")
    if selected_point_id is not None and selected_point_id not in assignment_points_by_id:
        st.session_state.planter_management_selected_point_id = None
        selected_point_id = None
    selected_point = assignment_points_by_id.get(selected_point_id)

    selector_col, helper_col = st.columns([0.7, 0.3], gap="large")
    with selector_col:
        selected_planter_id = st.selectbox(
            "Planter",
            options=[None] + [planter["id"] for planter in active_planters],
            format_func=lambda planter_id: (
                "Select a planter..."
                if planter_id is None
                else next(
                    (
                        f"{planter['full_name']} · {planter['pending_points']} pending · {planter['active_assignments']} active batches"
                        for planter in active_planters
                        if planter["id"] == planter_id
                    ),
                    "Unknown planter",
                )
            ),
            key="planter_management_selected_planter_id",
        )
    selected_planter = next((planter for planter in active_planters if planter["id"] == selected_planter_id), None)
    with helper_col:
        if selected_planter:
            base_text = "Base not set"
            if selected_planter.get("base_lat") is not None and selected_planter.get("base_lon") is not None:
                base_text = (
                    f"{selected_planter.get('base_label') or 'Base'} · "
                    f"{selected_planter['base_lat']:.6f}, {selected_planter['base_lon']:.6f}"
                )
            st.markdown(f"""
            <div class="manager-card">
                <div class="panel-kicker">Selected Planter</div>
                <h3>{selected_planter['full_name']}</h3>
                <div class="manager-meta">
                    <span>{selected_planter['pending_points']} pending points</span>
                    <span>{selected_planter['active_assignments']} active batches</span>
                </div>
                <p>{base_text}</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown("""
            <div class="manager-card">
                <div class="panel-kicker">Selection</div>
                <h3>No planter selected</h3>
                <p>Choose an active planter from the dropdown before assigning a planting point from the map.</p>
            </div>
            """, unsafe_allow_html=True)

    if not active_planters:
        st.info("No active planters are available for assignment.")
    elif not assignment_map_points:
        st.info("Save at least one analysis to the database before assigning field points.")
    else:
        if (
            st.session_state.get("planter_management_map_center") is None
            or st.session_state.get("planter_management_map_zoom") is None
        ):
            default_center, default_zoom = _default_planter_management_map_view(assignment_map_points)
            if st.session_state.get("planter_management_map_center") is None:
                st.session_state.planter_management_map_center = default_center
            if st.session_state.get("planter_management_map_zoom") is None:
                st.session_state.planter_management_map_zoom = default_zoom

        effective_selected_point_id = selected_point_id
        effective_selected_point = selected_point
        inline_click_warning = None
        pre_render_map_state = st.session_state.get("planter_assignment_map")
        if isinstance(pre_render_map_state, dict):
            pre_render_center = pre_render_map_state.get("center")
            if isinstance(pre_render_center, dict):
                if pre_render_center.get("lat") is not None and pre_render_center.get("lng") is not None:
                    st.session_state.planter_management_map_center = [
                        float(pre_render_center["lat"]),
                        float(pre_render_center["lng"]),
                    ]
            elif isinstance(pre_render_center, (list, tuple)) and len(pre_render_center) == 2:
                st.session_state.planter_management_map_center = [
                    float(pre_render_center[0]),
                    float(pre_render_center[1]),
                ]

            pre_render_zoom = pre_render_map_state.get("zoom")
            if isinstance(pre_render_zoom, (int, float)):
                st.session_state.planter_management_map_zoom = int(pre_render_zoom)

            pre_clicked_point_id, pre_click_event_key = _resolve_clicked_assignment_point(
                pre_render_map_state,
                assignment_map_points,
            )
            if (
                pre_clicked_point_id is not None
                and pre_click_event_key
                and pre_click_event_key != st.session_state.get("planter_management_last_map_event")
            ):
                st.session_state.planter_management_last_map_event = pre_click_event_key
                st.session_state.planter_management_selected_point_id = pre_clicked_point_id
                effective_selected_point_id = pre_clicked_point_id
                effective_selected_point = assignment_points_by_id.get(pre_clicked_point_id)
                if selected_planter is None:
                    inline_click_warning = "Please select a planter first."

        map_col, detail_col = st.columns([1.35, 0.65], gap="large")
        with map_col:
            assignment_map = _get_cached_planter_management_base_map(assignment_map_points)
            selected_layers = _build_planter_assignment_selected_layers(effective_selected_point)
            st.caption("Marker colors: green = planned, blue = assigned, yellow = planted or selected.")
            map_response = _render_planter_assignment_map(
                assignment_map,
                selected_layers=selected_layers,
                center=st.session_state.get("planter_management_map_center"),
                zoom=st.session_state.get("planter_management_map_zoom"),
            )

            response_center = map_response.get("center") if isinstance(map_response, dict) else None
            if isinstance(response_center, dict):
                if response_center.get("lat") is not None and response_center.get("lng") is not None:
                    st.session_state.planter_management_map_center = [
                        float(response_center["lat"]),
                        float(response_center["lng"]),
                    ]
            elif isinstance(response_center, (list, tuple)) and len(response_center) == 2:
                st.session_state.planter_management_map_center = [
                    float(response_center[0]),
                    float(response_center[1]),
                ]

            response_zoom = map_response.get("zoom") if isinstance(map_response, dict) else None
            if isinstance(response_zoom, (int, float)):
                st.session_state.planter_management_map_zoom = int(response_zoom)

            clicked_point_id, click_event_key = _resolve_clicked_assignment_point(
                map_response,
                assignment_map_points,
            )
            if (
                clicked_point_id is not None
                and click_event_key
                and click_event_key != st.session_state.get("planter_management_last_map_event")
            ):
                st.session_state.planter_management_last_map_event = click_event_key
                st.session_state.planter_management_selected_point_id = clicked_point_id
                effective_selected_point_id = clicked_point_id
                effective_selected_point = assignment_points_by_id.get(clicked_point_id)
                if selected_planter is None:
                    inline_click_warning = "Please select a planter first."

        with detail_col:
            selected_point = effective_selected_point or assignment_points_by_id.get(
                st.session_state.get("planter_management_selected_point_id")
            )

            if inline_click_warning:
                st.warning(inline_click_warning)

            if not selected_point:
                st.markdown("""
                <div class="manager-card">
                    <div class="panel-kicker">Point Details</div>
                    <h3>No point selected</h3>
                    <p>Click a planting point on the map to inspect its details and assign it to the selected planter.</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                planting_status = (selected_point.get("planting_status") or "planned").strip().lower()
                assignment_state = (
                    "Planted"
                    if planting_status == "planted"
                    else (
                        f"Assigned to {selected_point['assigned_planter_name']}"
                        if selected_point.get("assigned_planter_name")
                        else "Unassigned"
                    )
                )
                st.markdown(f"""
                <div class="manager-card">
                    <div class="field-title-row">
                        <div>
                            <div class="panel-kicker">Selected Point</div>
                            <h3>Point {int(selected_point['point_num']):03d}</h3>
                        </div>
                        <div class="field-status {'completed' if planting_status == 'planted' or selected_point.get('assigned_planter_id') else 'pending'}">
                            {assignment_state}
                        </div>
                    </div>
                    <div class="manager-meta">
                        <span>{selected_point['image_name']}</span>
                        <span>{selected_point['analyzed_at'][:10]}</span>
                        <span>{float(selected_point['buffer_m'] or 0):.1f} m buffer</span>
                    </div>
                    <div class="field-coords">{float(selected_point['latitude']):.7f}, {float(selected_point['longitude']):.7f}</div>
                    <p>Area {float(selected_point['area_m2'] or 0):.2f} m²</p>
                </div>
                """, unsafe_allow_html=True)

                if selected_point.get("assigned_planter_name"):
                    st.caption(
                        f"Current assignment: {selected_point['assigned_planter_name']}"
                        + (
                            f" · Batch {selected_point['assignment_title']}"
                            if selected_point.get("assignment_title")
                            else ""
                        )
                    )
                else:
                    st.caption(
                        "This planting point is currently marked as planted."
                        if planting_status == "planted"
                        else "This planting point is currently available for assignment."
                    )

                assign_disabled = selected_planter is None or (
                    selected_point.get("assigned_planter_id") == selected_planter_id
                )
                reassign_mode = (
                    selected_planter is not None
                    and selected_point.get("assigned_planter_id") is not None
                    and selected_point.get("assigned_planter_id") != selected_planter_id
                )

                if planting_status == "planted":
                    st.success("This point is already marked as planted and now appears as a yellow point on analytics maps.")
                elif selected_planter is None:
                    st.warning("Please select a planter first.")
                elif selected_point.get("assigned_planter_id") == selected_planter_id:
                    st.info(f"This point is already assigned to {selected_planter['full_name']}.")

                if (
                    planting_status != "planted"
                    and selected_planter is not None
                    and selected_point.get("assigned_planter_id") != selected_planter_id
                ):
                    button_label = (
                        f"Reassign To {selected_planter['full_name']}"
                        if reassign_mode
                        else f"Assign To {selected_planter['full_name']}"
                    )
                    st.button(
                        button_label,
                        key=f"assign_map_point_{selected_point['id']}_{selected_planter_id}",
                        type="primary",
                        use_container_width=True,
                        disabled=assign_disabled,
                        on_click=_queue_planter_management_assignment,
                        args=(selected_point["id"], selected_planter_id, reassign_mode),
                    )

    st.markdown("---")
    roster_col, assignment_list_col = st.columns([0.94, 1.06], gap="large")

    with roster_col:
        st.markdown("""
        <div class="manager-card">
            <div class="panel-kicker">Planter Roster</div>
            <h3>Registered Field Staff</h3>
            <p>Track field workload, account access, and base location readiness for every planter.</p>
        </div>
        """, unsafe_allow_html=True)

        if not planters:
            st.info("No planter profiles yet.")
        else:
            for planter in planters:
                base_text = "Base not set"
                if planter.get("base_lat") is not None and planter.get("base_lon") is not None:
                    base_text = f"{planter.get('base_label') or 'Base'} - {planter['base_lat']:.6f}, {planter['base_lon']:.6f}"
                st.markdown(f"""
                <div class="manager-card">
                    <div class="field-title-row">
                        <div>
                            <div class="panel-kicker">Planter</div>
                            <h3>{planter['full_name']}</h3>
                        </div>
                        <div class="field-status {'pending' if planter['status'] == 'active' else 'skipped'}">{planter['status']}</div>
                    </div>
                    <div class="manager-meta">
                        <span>{planter['active_assignments']} active assignments</span>
                        <span>{planter['pending_points']} pending points</span>
                        <span>{planter['completed_points']} completed points</span>
                    </div>
                    <p>Username: {planter.get('username') or 'No field login yet'}<br>{planter.get('phone') or 'No contact number saved'}<br>{base_text}</p>
                </div>
                """, unsafe_allow_html=True)

    with assignment_list_col:
        st.markdown("""
        <div class="manager-card">
            <div class="panel-kicker">Assignment Monitor</div>
            <h3>Active And Recent Batches</h3>
            <p>Export per-planter point lists, archive finished work, and clean up old batches once the field run is done.</p>
        </div>
        """, unsafe_allow_html=True)

        if not assignments:
            st.info("No planter assignments yet.")
        else:
            for assignment in assignments:
                assignment_points = get_assignment_points(assignment["id"])
                waypoint_rows = _assignment_points_to_waypoints(assignment_points, prefix=assignment["title"])
                export_meta = {
                    "image_name": assignment["title"],
                    "analyzed_at": assignment["created_at"],
                    "detection_mode": "field-assignment",
                    "total_points": len(waypoint_rows),
                }
                export_df = pd.DataFrame([
                    {
                        "Sequence": point["sequence_num"],
                        "Point #": point["point_num"],
                        "Latitude": f"{point['latitude']:.7f}",
                        "Longitude": f"{point['longitude']:.7f}",
                        "Status": point["assignment_status"],
                        "Source Image": point["image_name"],
                    }
                    for point in assignment_points
                ])

                st.markdown(f"""
                <div class="manager-card">
                    <div class="field-title-row">
                        <div>
                            <div class="panel-kicker">Assignment</div>
                            <h3>{assignment['title']}</h3>
                        </div>
                        <div class="field-status {'pending' if assignment['status'] == 'active' else 'completed'}">{assignment['status']}</div>
                    </div>
                    <div class="manager-meta">
                        <span>{assignment['planter_name']}</span>
                        <span>{assignment['assignment_date']}</span>
                        <span>{assignment['travel_mode']}</span>
                        <span>{assignment['total_points']} total points</span>
                        <span>{assignment['pending_points']} pending</span>
                        <span>{assignment['completed_points']} completed</span>
                    </div>
                    <p>{assignment.get('notes') or 'No batch notes.'}</p>
                </div>
                """, unsafe_allow_html=True)

                if assignment_points:
                    export_col1, export_col2, export_col3, export_col4 = st.columns(4)
                    with export_col1:
                        st.download_button(
                            "CSV",
                            data=export_df.to_csv(index=False),
                            file_name=f"assignment_{assignment['id']}.csv",
                            mime="text/csv",
                            use_container_width=True,
                            key=f"assignment_csv_{assignment['id']}",
                        )
                    with export_col2:
                        st.download_button(
                            "GPX",
                            data=generate_gpx(waypoint_rows, export_meta),
                            file_name=f"assignment_{assignment['id']}.gpx",
                            mime="application/gpx+xml",
                            use_container_width=True,
                            key=f"assignment_gpx_{assignment['id']}",
                        )
                    with export_col3:
                        st.download_button(
                            "KML",
                            data=generate_kml(waypoint_rows, export_meta),
                            file_name=f"assignment_{assignment['id']}.kml",
                            mime="application/vnd.google-earth.kml+xml",
                            use_container_width=True,
                            key=f"assignment_kml_{assignment['id']}",
                        )
                    with export_col4:
                        st.download_button(
                            "GeoJSON",
                            data=generate_geojson(waypoint_rows, export_meta),
                            file_name=f"assignment_{assignment['id']}.geojson",
                            mime="application/geo+json",
                            use_container_width=True,
                            key=f"assignment_geojson_{assignment['id']}",
                        )

                action_col1, action_col2 = st.columns(2)
                with action_col1:
                    archive_disabled = assignment["status"] == "archived"
                    if st.button(
                        "Archive Assignment" if not archive_disabled else "Already Archived",
                        key=f"archive_assignment_{assignment['id']}",
                        use_container_width=True,
                        disabled=archive_disabled,
                    ):
                        archive_planter_assignment(assignment["id"])
                        _refresh_planter_management_cache(
                            reload_planters=True,
                            reload_points=True,
                            reload_assignments=True,
                        )
                        _invalidate_planter_management_base_map()
                        st.rerun()
                with action_col2:
                    delete_disabled = assignment["status"] != "archived"
                    if st.button(
                        "Delete Archived",
                        key=f"delete_assignment_{assignment['id']}",
                        use_container_width=True,
                        disabled=delete_disabled,
                    ):
                        delete_planter_assignment(assignment["id"])
                        _refresh_planter_management_cache(
                            reload_planters=True,
                            reload_points=True,
                            reload_assignments=True,
                        )
                        _invalidate_planter_management_base_map()
                        st.rerun()


def show_planter_field_view():
    """Mobile-friendly field module for planters and route launching."""
    planters = [p for p in list_planters(include_inactive=False) if p["status"] == "active"]
    dashboard_stats = get_planter_dashboard_stats()

    _render_section_banner(
        "Field Navigation",
        "Planter Mobile View",
        "Use this responsive field view on a phone to open navigation for the next planting point, then mark it completed or skipped after deployment.",
        [
            f"{dashboard_stats['active_planters']} active planters",
            f"{dashboard_stats['active_assignments']} live batches",
            "Google Maps handoff",
        ],
    )

    if not planters:
        st.info("No active planters are available yet. Register a planter in Planter Management first.")
        return

    planter_options = {f"{p['full_name']} · {p['pending_points']} pending": p["id"] for p in planters}
    selected_planter_label = st.selectbox("Select planter", list(planter_options.keys()))
    selected_planter_id = planter_options[selected_planter_label]
    selected_planter = next((p for p in planters if p["id"] == selected_planter_id), None)
    field_points = get_planter_field_points(selected_planter_id)
    active_assignments = list_planter_assignments(planter_id=selected_planter_id, active_only=True)

    pending_count = sum(1 for point in field_points if point["assignment_status"] == "pending")
    completed_count = sum(1 for point in field_points if point["assignment_status"] == "completed")
    skipped_count = sum(1 for point in field_points if point["assignment_status"] == "skipped")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Active batches", len(active_assignments))
    c2.metric("Pending points", pending_count)
    c3.metric("Completed", completed_count)
    c4.metric("Skipped", skipped_count)

    if selected_planter and selected_planter.get("base_label"):
        st.caption(f"Field base: {selected_planter['base_label']}")

    if not field_points:
        st.info("This planter does not have any active field points yet.")
        return

    status_order = {"pending": 0, "completed": 1, "skipped": 2}
    field_points = sorted(
        field_points,
        key=lambda row: (
            status_order.get(row["assignment_status"], 9),
            row["assignment_date"],
            row["sequence_num"],
        ),
    )

    for point in field_points:
        status_class = point["assignment_status"]
        st.markdown(f"""
        <div class="field-card is-{status_class}">
            <div class="field-title-row">
                <div>
                    <div class="panel-kicker">Assignment Point</div>
                    <h3>Point {point['sequence_num']:02d} · {point['title']}</h3>
                </div>
                <div class="field-status {status_class}">{point['assignment_status']}</div>
            </div>
            <div class="field-meta">
                <span>{point['image_name']}</span>
                <span>{point['travel_mode']}</span>
                <span>Point #{point['point_num']}</span>
                <span>{point['assignment_date']}</span>
            </div>
            <div class="field-coords">{point['latitude']:.7f}, {point['longitude']:.7f}</div>
            <div class="field-caption">Buffer {point.get('buffer_m') or 0:.1f} m · Area {point.get('area_m2') or 0:.2f} m²</div>
            <div class="field-divider"></div>
        </div>
        """, unsafe_allow_html=True)

        nav_url = _build_google_maps_navigation_url(
            point["latitude"],
            point["longitude"],
            travel_mode=point.get("travel_mode") or "walking",
        )
        _render_navigation_button("Open Navigation", nav_url)

        action_col1, action_col2, action_col3 = st.columns(3)
        with action_col1:
            if st.button("Mark Complete", key=f"complete_{point['assignment_point_id']}", use_container_width=True):
                update_assignment_point_status(point["assignment_point_id"], "completed")
                st.rerun()
        with action_col2:
            if st.button("Skip Point", key=f"skip_{point['assignment_point_id']}", use_container_width=True):
                update_assignment_point_status(point["assignment_point_id"], "skipped")
                st.rerun()
        with action_col3:
            if st.button("Reset", key=f"reset_{point['assignment_point_id']}", use_container_width=True):
                update_assignment_point_status(point["assignment_point_id"], "pending")
                st.rerun()

def show_eroded_zone_editor():
    """
    Map-based editor for marking eroded zones.
    Users draw polygons on the orthophoto map to designate eroded areas
    that should be excluded from planting, even if space is available.
    """
    from folium.plugins import Draw, Fullscreen

    _render_section_banner(
        "Zone Editor",
        "Eroded Area Mapping",
        "Draw and save erosion exclusions directly on the orthophoto so unsafe planting sections are blocked before field deployment.",
        [
            f"{len(_eroded_filter.forbidden_polygons)} saved erosion polygons",
            f"{_forbidden_filter.zone_count} structural exclusions",
        ],
    )

    st.markdown("""
    <div class="info-box">
        <strong>🏜️ Eroded Zone Editor</strong><br>
        <span style="color: #1565C0;">
        Draw polygons on the map to mark <strong>eroded areas</strong> where mangroves
        should <strong>not</strong> be planted. These zones will be excluded from
        planting recommendations just like forbidden zones (towers, bridges, houses).<br><br>
        <strong>How to use:</strong><br>
        1. Use the polygon draw tool (▣) on the left side of the map<br>
        2. Click points on the map to define the eroded area boundary<br>
        3. Double-click to finish the polygon<br>
        4. Click <strong>"💾 Save Eroded Zones"</strong> to save<br>
        5. Saved zones will automatically be applied when analyzing drone images
        </span>
    </div>
    """, unsafe_allow_html=True)

    # Load existing eroded zones for display
    existing_zones = []
    if _ERODED_ZONES_PATH.exists():
        try:
            with open(_ERODED_ZONES_PATH, 'r', encoding='utf-8') as f:
                existing_data = json.load(f)
            existing_zones = existing_data.get('features', [])
        except Exception:
            existing_zones = []

    # Stats
    col1, col2 = st.columns(2)
    with col1:
        st.metric("🏜️ Eroded Zones Saved", len(existing_zones))
    with col2:
        st.metric("🚫 Forbidden Zones (structures)", _forbidden_filter.zone_count)

    st.markdown("---")

    # Default center: Leganes mangrove area
    center_lat = 10.7800
    center_lon = 122.6253

    # Create map with drawing tools
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=19,
        tiles=None,
        control_scale=True,
    )
    _add_operational_map_layers(m, orthophoto_name="Orthophoto Overlay")

    # Show existing eroded zones (orange)
    if existing_zones:
        eroded_group = folium.FeatureGroup(name='🏜️ Eroded Zones (saved)')
        for i, feature in enumerate(existing_zones):
            geom = feature.get('geometry', {})
            if geom.get('type') == 'Polygon':
                coords_raw = geom['coordinates'][0]
                coords = [(c[1], c[0]) for c in coords_raw]  # (lon,lat) → (lat,lon)
                folium.Polygon(
                    locations=coords,
                    color='#FF6F00',
                    fill=True,
                    fillColor='#FF6F00',
                    fillOpacity=0.35,
                    weight=2,
                    tooltip=f'🏜️ Eroded Zone #{i+1} (saved)',
                ).add_to(eroded_group)
        eroded_group.add_to(m)

    # Show existing forbidden zones (red) for reference
    if _forbidden_filter.forbidden_polygons:
        fz_group = folium.FeatureGroup(name='🚫 Forbidden Zones (structures)')
        for poly in _forbidden_filter.forbidden_polygons:
            coords = [(lat, lon) for lon, lat in poly.exterior.coords]
            folium.Polygon(
                locations=coords,
                color='red',
                fill=True,
                fillColor='red',
                fillOpacity=0.25,
                weight=1,
                tooltip='🚫 Forbidden Zone (tower/bridge/house)',
            ).add_to(fz_group)
        fz_group.add_to(m)

    # Add Draw control (polygon only)
    Draw(
        export=False,
        draw_options={
            'polyline': False,
            'rectangle': True,
            'circle': False,
            'circlemarker': False,
            'marker': False,
            'polygon': {
                'allowIntersection': False,
                'shapeOptions': {
                    'color': '#FF6F00',
                    'fillColor': '#FF6F00',
                    'fillOpacity': 0.4,
                    'weight': 3,
                },
            },
        },
        edit_options={'edit': False},
    ).add_to(m)

    Fullscreen(position="topleft", title="Expand map", title_cancel="Exit fullscreen").add_to(m)
    folium.LayerControl(collapsed=False).add_to(m)
    _style_layer_control(m)

    # Render map and capture drawn data
    map_output = st_folium.st_folium(
        m, height=600,
        key="eroded_zone_map",
        returned_objects=["all_drawings"],
        use_container_width=True,
    )

    # Process drawn polygons
    st.markdown("---")
    st.markdown("### 📝 Drawn Polygons")

    all_drawings = map_output.get("all_drawings") or []
    new_polygons = []
    for drawing in all_drawings:
        geom = drawing.get("geometry", {})
        if geom.get("type") == "Polygon":
            new_polygons.append(drawing)

    if new_polygons:
        st.success(f"✅ {len(new_polygons)} new polygon(s) drawn on map")
        for i, poly in enumerate(new_polygons):
            coords = poly['geometry']['coordinates'][0]
            n_pts = len(coords)
            st.write(f"  • Polygon #{i+1}: {n_pts} vertices")
    else:
        st.info("Draw polygons on the map using the polygon tool (▣) on the left, then click **Save** below.")

    # Save / Clear buttons
    btn_col1, btn_col2, btn_col3 = st.columns(3)

    with btn_col1:
        if st.button("💾 Save Eroded Zones", type="primary", use_container_width=True):
            # Merge new drawings with existing zones
            features = list(existing_zones)  # Keep existing
            for poly in new_polygons:
                feature = {
                    "type": "Feature",
                    "properties": {
                        "zone_type": "eroded",
                        "label": "Eroded Area",
                    },
                    "geometry": poly["geometry"],
                }
                features.append(feature)

            geojson_out = {
                "type": "FeatureCollection",
                "name": "eroded_zones",
                "features": features,
            }
            with open(_ERODED_ZONES_PATH, 'w', encoding='utf-8') as f:
                json.dump(geojson_out, f, indent=2)

            _reload_eroded_filter()
            st.success(f"✅ Saved {len(features)} eroded zone(s) to eroded_zones.geojson")
            st.rerun()

    with btn_col2:
        if st.button("🗑️ Clear ALL Eroded Zones", use_container_width=True):
            geojson_out = {
                "type": "FeatureCollection",
                "name": "eroded_zones",
                "features": [],
            }
            with open(_ERODED_ZONES_PATH, 'w', encoding='utf-8') as f:
                json.dump(geojson_out, f, indent=2)

            _reload_eroded_filter()
            st.success("🗑️ All eroded zones cleared!")
            st.rerun()

    with btn_col3:
        if existing_zones:
            eroded_str = json.dumps({
                "type": "FeatureCollection",
                "name": "eroded_zones",
                "features": existing_zones,
            }, indent=2)
            st.download_button(
                "📥 Export GeoJSON",
                data=eroded_str,
                file_name="eroded_zones.geojson",
                mime="application/geo+json",
                use_container_width=True,
            )

    # Delete one saved zone at a time (without clearing everything)
    if existing_zones:
        st.markdown("### 🧹 Delete Individual Eroded Zone")

        zone_labels = []
        for idx, feature in enumerate(existing_zones, 1):
            coords = (feature.get("geometry", {}) or {}).get("coordinates", [[]])[0]
            zone_labels.append(f"Zone #{idx} ({len(coords)} vertices)")

        sel_col1, sel_col2 = st.columns([3, 1])
        with sel_col1:
            selected_zone = st.selectbox(
                "Select saved zone to delete",
                options=zone_labels,
                key="eroded_zone_delete_select",
            )
        with sel_col2:
            st.markdown("<br>", unsafe_allow_html=True)
            if st.button(
                "🗑️ Delete Selected",
                key="delete_selected_eroded_zone",
                use_container_width=True,
            ):
                selected_index = zone_labels.index(selected_zone)
                updated_features = [
                    feature for i, feature in enumerate(existing_zones)
                    if i != selected_index
                ]
                geojson_out = {
                    "type": "FeatureCollection",
                    "name": "eroded_zones",
                    "features": updated_features,
                }
                with open(_ERODED_ZONES_PATH, "w", encoding="utf-8") as f:
                    json.dump(geojson_out, f, indent=2)

                _reload_eroded_filter()
                st.success(
                    f"✅ Deleted Zone #{selected_index + 1}. "
                    f"{len(updated_features)} zone(s) remain."
                )
                st.rerun()

    # Legend
    st.markdown("""
    <div style="background: #1E1E1E; padding: 1rem; border-radius: 8px; margin-top: 1rem;">
        <strong style="color: #fff;">Legend:</strong><br>
        <span style="color: #FF6F00;">■</span> <span style="color: #ccc;">Eroded Zones (no planting)</span><br>
        <span style="color: #FF0000;">■</span> <span style="color: #ccc;">Forbidden Zones - structures (no planting)</span><br>
        <span style="color: #4CAF50;">■</span> <span style="color: #ccc;">Safe for planting (shown in Analyze mode)</span>
    </div>
    """, unsafe_allow_html=True)


# ── Map Analytics View ──────────────────────────────────────────────

def show_map_analytics():
    """
    Dashboard showing all saved planting data across the entire map.
    Aggregate stats + interactive map with every planting point ever saved.
    """
    from folium.plugins import Fullscreen

    _render_section_banner(
        "System Analytics",
        "Planting History And Coverage",
        "Review saved analyses, inspect historical planting points, and export the accumulated field dataset from the full Leganes mapping system.",
        [
            "Orthophoto overview",
            "Saved analyses",
            "Field export ready",
        ],
    )

    stats = get_all_stats()
    analyses = stats.get('analyses', [])
    all_points = stats.get('points', [])

    if stats['total_analyses'] == 0:
        st.info(
            "📭 **No analyses saved yet.** Go to **Map Workspace**, "
            "upload an image, run detection, then click **💾 Save to database**. "
            "Saved results will appear here."
        )
        return

    # ── Aggregate Metrics ─────────────────────────────────────────
    st.markdown("### 🧮 Aggregate Statistics")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("📸 Total Analyses", stats['total_analyses'])
    m2.metric("🌱 Remaining Planting Points", stats['total_planting_points'])
    m3.metric("🟢 Total Plantable Area", f"{stats['total_plantable_m2']:.1f} m²")
    m4.metric("🌳 Total Canopies Detected", stats['total_canopies'])

    m5, m6, m7, m8 = st.columns(4)
    m5.metric("🟡 Planted Points", stats.get('total_planted_points', 0))
    m6.metric("🔴 Total Danger Area", f"{stats['total_danger_m2']:.1f} m²")
    m7.metric("📏 Total Coverage", f"{stats['total_coverage_m2']:.1f} m²")
    m8.metric("🚫 Forbidden-Filtered", stats['total_forbidden_filtered'])
    m9, m10 = st.columns(2)
    m9.metric("🏜️ Erosion-Filtered", stats['total_eroded_filtered'])
    m10.metric("🧭 All Mapped Points", stats.get('total_mapped_points', len(all_points)))

    st.markdown("---")

    # ── Full Map ──────────────────────────────────────────────────
    st.markdown("### 🗺️ All Planting Locations")
    st.info(
        f"Showing **{stats.get('total_mapped_points', len(all_points))}** mapped points from **{stats['total_analyses']}** analyses. "
        f"**{stats['total_planting_points']}** remain to plant and **{stats.get('total_planted_points', 0)}** are already planted. "
        f"Green = planned, yellow = planted."
    )

    # Determine map centre from the average of all analysis centres
    _lats = [a['center_lat'] for a in analyses if a['center_lat']]
    _lons = [a['center_lon'] for a in analyses if a['center_lon']]
    if _lats and _lons:
        _center = [sum(_lats) / len(_lats), sum(_lons) / len(_lons)]
    else:
        _center = [10.780, 122.625]  # Leganes default

    analytics_map = folium.Map(
        location=_center,
        zoom_start=18,
        tiles=None,
        control_scale=True,
    )
    _add_operational_map_layers(analytics_map)

    # Show forbidden zone polygons (red)
    if _forbidden_filter.forbidden_polygons:
        fz = folium.FeatureGroup(name='🚫 Forbidden Zones')
        for poly in _forbidden_filter.forbidden_polygons:
            coords = [(lat, lon) for lon, lat in poly.exterior.coords]
            folium.Polygon(
                locations=coords, color='red', fill=True,
                fillColor='red', fillOpacity=0.30, weight=2,
                tooltip='🚫 Forbidden Zone',
            ).add_to(fz)
        fz.add_to(analytics_map)

    # Show eroded zone polygons (orange)
    if _eroded_filter.forbidden_polygons:
        ez = folium.FeatureGroup(name='🏜️ Eroded Zones')
        for poly in _eroded_filter.forbidden_polygons:
            coords = [(lat, lon) for lon, lat in poly.exterior.coords]
            folium.Polygon(
                locations=coords, color='orange', fill=True,
                fillColor='orange', fillOpacity=0.30, weight=2,
                tooltip='🏜️ Eroded Zone',
            ).add_to(ez)
        ez.add_to(analytics_map)

    # Colour each analysis differently
    _colours = ['#4CAF50', '#2196F3', '#FF9800', '#E91E63', '#9C27B0',
                '#00BCD4', '#CDDC39', '#FF5722', '#607D8B', '#795548']

    for idx, analysis in enumerate(analyses):
        colour = _colours[idx % len(_colours)]
        grp = folium.FeatureGroup(name=f"📸 {analysis['image_name']} ({analysis['analyzed_at'][:10]})")

        # Image centre marker
        if analysis['center_lat'] and analysis['center_lon']:
            folium.Marker(
                location=[analysis['center_lat'], analysis['center_lon']],
                popup=f"📸 <b>{analysis['image_name']}</b><br>"
                      f"Date: {analysis['analyzed_at']}<br>"
                      f"Canopies: {analysis['canopy_count']}<br>"
                      f"Planting pts: {analysis['hexagon_count']}<br>"
                      f"Plantable: {analysis['plantable_area_m2']:.1f} m²",
                icon=folium.Icon(color='blue', icon='camera', prefix='fa'),
                tooltip=f"📸 {analysis['image_name']}",
            ).add_to(grp)

        grp.add_to(analytics_map)

    # All planting points as a single layer
    pts_grp = folium.FeatureGroup(name='🌱 All Planting Points')
    for pt in all_points:
        point_style = _saved_point_marker_style(pt.get("status"))
        folium.CircleMarker(
            location=[pt['latitude'], pt['longitude']],
            radius=3,
            color=point_style["border_color"],
            fillColor=point_style["fill_color"],
            fillOpacity=0.8,
            weight=1,
            tooltip=f"{point_style['label']} · {pt['image_name']} ({pt['analyzed_at'][:10]})",
            popup=f"🌱 GPS: {pt['latitude']:.7f}°, {pt['longitude']:.7f}°<br>"
                  f"Image: {pt['image_name']}<br>"
                  f"Status: {point_style['label']}<br>"
                  f"Buffer: {pt['buffer_m']}m | Area: {pt['area_m2']:.2f} m²",
        ).add_to(pts_grp)
    pts_grp.add_to(analytics_map)

    Fullscreen(position="topleft", title="Expand map", title_cancel="Exit fullscreen").add_to(analytics_map)
    folium.LayerControl(collapsed=False).add_to(analytics_map)
    _style_layer_control(analytics_map)
    st_folium.st_folium(
        analytics_map,
        height=650,
        key="analytics_map",
        returned_objects=[],
        use_container_width=True,
    )

    # ── Per-analysis breakdown table ──────────────────────────────
    st.markdown("---")
    st.markdown("### 📋 Analysis History")

    for a in analyses:
        with st.expander(f"📸 {a['image_name']} — {a['analyzed_at']} ({a['hexagon_count']} points)"):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("🌳 Canopies", a['canopy_count'])
            c2.metric("🌱 Planting Pts", a['hexagon_count'])
            c3.metric("🟢 Plantable", f"{a['plantable_area_m2']:.1f} m²")
            c4.metric("🔴 Danger", f"{a['danger_area_m2']:.1f} m²")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("📏 Coverage", f"{a['total_area_m2']:.1f} m²")
            c6.metric("📐 GSD", f"{a['gsd_cm']:.2f} cm/px" if a['gsd_cm'] else "—")
            c7.metric("🚫 Forbidden", a['forbidden_filtered'])
            c8.metric("🏜️ Eroded", a['eroded_filtered'])

            if a['center_lat'] and a['center_lon']:
                st.caption(f"📍 Centre: {a['center_lat']:.6f}°, {a['center_lon']:.6f}°")

            if st.button(f"🗑️ Delete this analysis", key=f"del_{a['id']}"):
                delete_analysis(a['id'])
                st.success("Deleted. Refresh the page to update.")
                st.rerun()

    # ── Bulk Export All Planting Points ────────────────────────────
    if all_points:
        st.markdown("---")
        st.markdown("### 📡 Export All Planting Points")
        st.info(f"Export all **{len(all_points)}** saved planting points for field navigation")

        _bulk_wps = hexagons_to_waypoints(all_points)
        _bulk_meta = {
            "image_name": "all_analyses",
            "analyzed_at": datetime.now().isoformat(timespec="seconds"),
            "total_points": len(_bulk_wps),
        }

        _bc1, _bc2, _bc3, _bc4 = st.columns(4)
        with _bc1:
            _bulk_csv = pd.DataFrame([{
                "Point #": w["point_num"],
                "Latitude": f"{w['lat']:.7f}",
                "Longitude": f"{w['lon']:.7f}",
                "Buffer (m)": w.get("buffer_m", ""),
                "Area (m²)": w.get("area_m2", ""),
                "Status": w.get("status", "planned"),
            } for w in _bulk_wps]).to_csv(index=False)
            st.download_button(
                label="📥 CSV",
                data=_bulk_csv,
                file_name="mangrovision_all_points.csv",
                mime="text/csv",
                use_container_width=True,
                key="bulk_csv",
            )
        with _bc2:
            st.download_button(
                label="📡 GPX (GPS)",
                data=generate_gpx(_bulk_wps, _bulk_meta),
                file_name="mangrovision_all_points.gpx",
                mime="application/gpx+xml",
                use_container_width=True,
                key="bulk_gpx",
            )
        with _bc3:
            st.download_button(
                label="🌍 KML (Google Earth)",
                data=generate_kml(_bulk_wps, _bulk_meta),
                file_name="mangrovision_all_points.kml",
                mime="application/vnd.google-earth.kml+xml",
                use_container_width=True,
                key="bulk_kml",
            )
        with _bc4:
            st.download_button(
                label="🗺️ GeoJSON (QGIS)",
                data=generate_geojson(_bulk_wps, _bulk_meta),
                file_name="mangrovision_all_points.geojson",
                mime="application/geo+json",
                use_container_width=True,
                key="bulk_geojson",
            )
        st.caption("💡 GPX works with Garmin, Locus Map, OsmAnd. KML opens in Google Earth. GeoJSON imports into QGIS.")


def main():

    if not _render_login_screen():
        return

    try:
        from canopy_detection.detectree2_proper import ProperDetectree2Detector  # noqa: F401
        ai_available = True
    except ImportError:
        ai_available = False

    detection_mode = "ai" if ai_available else "hsv"
    workflow_label = "Standard canopy mapping" if ai_available else "Backup canopy mapping"
    workspace_stats = get_all_stats()
    eroded_zone_count = len(_eroded_filter.forbidden_polygons)
    total_exclusions = _forbidden_filter.zone_count + eroded_zone_count
    workspace_modes = [
        "Map Workspace",
        "Map Analytics",
        "Eroded Zone Editor",
        "Planter Management",
        "Field Navigation",
    ]
    if st.session_state.get("workspace_mode") not in workspace_modes:
        st.session_state.workspace_mode = "Map Workspace"

    with st.sidebar:
        st.markdown("""
        <div class="sidebar-brand">
            <span>Navigation</span>
            <h2>MangroVision</h2>
            <p>Move between mapping, analytics, erosion editing, planter management, and the mobile field navigation view from one operational sidebar.</p>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("### Workspace Views")
        for option in workspace_modes:
            if st.button(
                option,
                key=f"workspace_mode_{option.lower().replace(' ', '_')}",
                use_container_width=True,
                type="primary" if st.session_state.workspace_mode == option else "secondary",
            ):
                st.session_state.workspace_mode = option
        mode = st.session_state.workspace_mode

        st.markdown("---")
        st.markdown("### System Snapshot")
        st.metric("Saved analyses", workspace_stats['total_analyses'])
        st.metric("Remaining points", workspace_stats['total_planting_points'])
        st.metric("Active exclusions", total_exclusions)

        ai_confidence = 0.75
        ai_runtime_tuning = {}
        altitude = 6.0
        drone_model = "GENERIC_4K"

        if mode == "Map Workspace":
            st.markdown("---")
            st.markdown("### Analysis Controls")
            canopy_buffer = st.slider(
                "Danger Zone Buffer (meters)",
                min_value=0.5,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Buffer distance around detected canopies (red zones)"
            )

            hexagon_size = st.slider(
                "Planting Hexagon Size (meters)",
                min_value=0.3,
                max_value=2.0,
                value=1.0,
                step=0.1,
                help="Size of hexagonal planting zones (green buffers)"
            )
            st.caption("These values are applied when you run a new image analysis.")
        else:
            canopy_buffer = 1.0
            hexagon_size = 1.0
            st.markdown("---")
            st.caption("Analysis controls appear here when you return to the map workspace.")

        st.markdown("---")
        _render_user_panel()

    st.markdown(f"""
    <div class="main-header">
        <div class="main-header-grid">
            <div>
                <div class="main-kicker">MangroVision Planning System</div>
                <h1>Leganes Mangrove Mapping Workspace</h1>
                <p>Professional geospatial workspace for canopy detection, exclusion zoning, planting-point generation, and field export.</p>
                <div class="main-header-notes">
                    <span>Map-first workflow</span>
                    <span>{_forbidden_filter.zone_count} forbidden zones</span>
                    <span>{eroded_zone_count} eroded zones</span>
                </div>
            </div>
            <div class="header-meta">
                <div class="header-badge">{workspace_stats['total_analyses']} saved analyses</div>
                <div class="header-badge">{workspace_stats['total_planting_points']} remaining planting points</div>
                <div class="header-badge">{total_exclusions} active exclusion polygons</div>
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    if mode == "Eroded Zone Editor":
        show_eroded_zone_editor()
        return

    if mode == "Map Analytics":
        show_map_analytics()
        return

    if mode == "Planter Management":
        show_planter_management()
        return

    if mode == "Field Navigation":
        show_planter_field_view()
        return

    uploaded_file = None
    image = None

    workspace_map = _build_workspace_overview_map(workspace_stats)
    st_folium.st_folium(
        workspace_map,
        height=860,
        key="workspace_overview_map",
        returned_objects=[],
        use_container_width=True,
    )
    st.markdown("""
    <div class="layer-legend">
        <span class="layer-pill"><span class="layer-dot" style="background:#1B5E20;"></span>Planting zones</span>
        <span class="layer-pill"><span class="layer-dot" style="background:#C62828;"></span>Forbidden structures</span>
        <span class="layer-pill"><span class="layer-dot" style="background:#EF6C00;"></span>Eroded areas</span>
        <span class="layer-pill"><span class="layer-dot" style="background:#1E88E5;"></span>Analysis locations</span>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="operations-strip">
        <div>
            <div class="panel-kicker">Operations Deck</div>
            <h3 class="panel-title">Analysis Queue And Site Review</h3>
            <p class="panel-copy">Keep the map as the operating surface, then stage one drone frame below it when the site boundary and exclusion layers are already confirmed.</p>
        </div>
        <div class="operations-meta">
            <span>{workflow_label}</span>
            <span>{canopy_buffer:.1f} m danger buffer</span>
            <span>{hexagon_size:.1f} m planting spacing</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    st.markdown(f"""
    <div class="control-card">
        <div class="panel-kicker">Analysis Console</div>
        <h3 class="panel-title">Queue A Drone Frame</h3>
        <p class="panel-copy">Use this console only after you verify the planting site on the map. The upload stays secondary so the workspace still reads like a mapping system.</p>
        <div class="operations-meta" style="margin-top:0.9rem;">
            <span>{workflow_label}</span>
            <span>{canopy_buffer:.1f} m danger buffer</span>
            <span>{hexagon_size:.1f} m planting spacing</span>
            <span>{altitude:.1f} m manual altitude</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    _render_workspace_clock()

    uploaded_file = st.file_uploader(
        "Select a drone image",
        type=["jpg", "jpeg", "png"],
        help="Upload a drone image of the mangrove area"
    )

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.caption(f"Queued frame: {image.size[0]} x {image.size[1]} pixels")
        if st.button("Run Analysis", type="primary"):
            st.session_state.run_analysis = True
            st.session_state.current_file = uploaded_file
            st.session_state.current_altitude = altitude
            st.session_state.current_drone_model = drone_model
            st.session_state.current_canopy_buffer = canopy_buffer
            st.session_state.current_hexagon_size = hexagon_size
            st.session_state.current_ai_confidence = ai_confidence
            st.session_state.current_detection_mode = detection_mode
            st.session_state.current_ai_runtime_tuning = ai_runtime_tuning

        st.markdown("""
        <div class="preview-card">
            <div class="panel-kicker">Image Preview</div>
            <h3 class="panel-title">Queued Drone Frame</h3>
            <p class="panel-copy">This frame is ready to process using the active planting geometry settings.</p>
        </div>
        """, unsafe_allow_html=True)
        st.image(image, caption="Queued image", width='stretch')
    else:
        st.caption("Load a drone frame only after the map layers and target site are visually confirmed.")

    # Run analysis if requested
    if uploaded_file is not None and st.session_state.get('run_analysis', False):
        analyze_image(
            st.session_state.current_file,
            st.session_state.current_altitude,
            st.session_state.current_drone_model,
            st.session_state.current_canopy_buffer,
            st.session_state.current_hexagon_size,
            st.session_state.current_ai_confidence,
            st.session_state.current_detection_mode,
            st.session_state.get('current_ai_runtime_tuning', {})
        )


def analyze_image(
    uploaded_file,
    altitude,
    drone_model,
    canopy_buffer,
    hexagon_size,
    ai_confidence,
    detection_mode='ai',
    ai_runtime_tuning=None,
):
    """Process the uploaded image using the canopy-analysis workflow."""
    
    # ── Progress bar for user feedback ─────────────────────────────
    progress_bar = st.progress(0, text="⏳ Preparing analysis...")
    tile_status = st.empty()
    
    with st.container():
        # Save uploaded file temporarily
        temp_path = Path("temp_upload.jpg")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        progress_bar.progress(5, text="📡 Extracting image metadata...")
        
        try:
            # STEP 1: Extract EXIF metadata (GPS, altitude, drone model)
            st.markdown("---")
            st.markdown("### 📡 Extracting Image Metadata")
            
            metadata = ExifExtractor.extract_all_metadata(str(temp_path))
            
            # Orthophoto map bounds — derived from ALL 3 WebODM orthophotos
            # 1st MAP: W=458971.9 E=459095.3 S=1191652.1 N=1191823.2
            # 2nd MAP: W=458878.8 E=459027.1 S=1191652.5 N=1191788.4
            # 3rd MAP: W=458847.6 E=459039.3 S=1191556.9 N=1191711.0
            bounds_utm = {
                'north': 1191823.193,    # 1st MAP top
                'south': 1191556.918,    # 3rd MAP bottom
                'east':  459095.262,     # 1st MAP right
                'west':  458847.596      # 3rd MAP left
            }
            
            # Convert to lat/lon for display
            transformer = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
            sw_lon, sw_lat = transformer.transform(bounds_utm['west'], bounds_utm['south'])
            ne_lon, ne_lat = transformer.transform(bounds_utm['east'], bounds_utm['north'])
            
            # GPS validation
            image_gps = None
            image_center_lat = None
            image_center_lon = None
            altitude_to_use = altitude
            drone_to_use = drone_model
            gps_valid = False
            
            if metadata.get('has_gps'):
                gps = metadata['gps']
                image_center_lat = gps['latitude']
                image_center_lon = gps['longitude']
                
                # Check if GPS is within orthophoto bounds
                if (sw_lat <= image_center_lat <= ne_lat and 
                    sw_lon <= image_center_lon <= ne_lon):
                    st.success(f"✅ GPS Found: {image_center_lat:.6f}°, {image_center_lon:.6f}° (INSIDE map bounds)")
                    gps_valid = True
                    image_gps = gps
                    
                    # ── Check for existing planting data in this area ──────
                    _overlaps = find_overlapping_analyses(image_center_lat, image_center_lon)
                    if _overlaps:
                        _existing_pts = count_nearby_points(image_center_lat, image_center_lon)
                        _names = ', '.join(set(o['image_name'] for o in _overlaps[:3]))
                        st.warning(
                            f"⚠️ **This area already has planting data!** "
                            f"{len(_overlaps)} previous analysis(es) found nearby "
                            f"({_names}), with {_existing_pts} planting points saved. "
                            f"Run detection, then save only if you want to update the database."
                        )
                else:
                    st.warning(f"⚠️ GPS Found: {image_center_lat:.6f}°, {image_center_lon:.6f}° (OUTSIDE map bounds)")
                    st.info("Map will show markers, but they may be outside the orthophoto area")
                    gps_valid = False  # Still process, but warn user
                    image_gps = gps
                
                # Use detected altitude if available
                if 'relative_altitude' in gps and gps['relative_altitude'] is not None:
                    altitude_to_use = gps['relative_altitude']
                    st.info(f"✈️ Using detected altitude: {altitude_to_use:.1f}m (AGL from EXIF)")
                elif 'altitude' in gps and gps['altitude'] is not None:
                    altitude_to_use = gps['altitude']
                    st.info(f"✈️ Using detected altitude: {altitude_to_use:.1f}m (MSL from EXIF)")
                
                # Auto-detect drone model
                if metadata.get('camera'):
                    drone_to_use = ExifExtractor.detect_drone_model(metadata['camera'])
                    st.info(f"📷 Detected drone: {drone_to_use.replace('_', ' ')}")
                
                # Automatic heading only (manual heading UI removed).
                camera_heading = 0.0
                heading_source = "Default (North)"
                if gps.get('heading') is not None:
                    camera_heading = float(gps['heading'])
                    heading_source = "EXIF GPSImgDirection"
                st.info(f"🧭 Using heading: **{camera_heading:.1f}°** ({heading_source})")
            else:
                st.error("❌ No GPS data found in image!")
                st.warning("Cannot geotag results on map without GPS coordinates.")
                st.info("Using manual altitude and drone model from sidebar.")
                camera_heading = 0  # Default if no GPS
            
            # STEP 2: Run detection with detected or manual parameters
            st.markdown("---")
            st.markdown("### 🔍 Running Detection Analysis")
            
            progress_bar.progress(15, text="🔍 Initializing analysis engine...")
            
            # Create a unique key for this analysis.
            # Include detector backend mtimes so code changes invalidate cached results.
            try:
                canopy_code_mtime = int((Path(__file__).parent / "canopy_detection" / "canopy_detector_hexagon.py").stat().st_mtime)
            except Exception:
                canopy_code_mtime = 0
            try:
                proper_code_mtime = int((Path(__file__).parent / "canopy_detection" / "detectree2_proper.py").stat().st_mtime)
            except Exception:
                proper_code_mtime = 0
            tuning_fingerprint = json.dumps(ai_runtime_tuning or {}, sort_keys=True)
            analysis_key = (
                f"{uploaded_file.name}_{altitude_to_use}_{drone_to_use}_{canopy_buffer}_"
                f"{hexagon_size}_{ai_confidence}_{detection_mode}_"
                f"{canopy_code_mtime}_{proper_code_mtime}_"
                f"{tuning_fingerprint}"
            )
            
            # Check if we've already run detection for this configuration
            if 'last_analysis_key' not in st.session_state or st.session_state.last_analysis_key != analysis_key:
                # Initialize the detector for the active canopy-analysis workflow.
                detector = HexagonDetector(
                    altitude_m=altitude_to_use, 
                    drone_model=drone_to_use,
                    ai_confidence=ai_confidence,
                    detection_mode=detection_mode
                )
                if (
                    ai_runtime_tuning
                    and getattr(detector, "ai_detector", None) is not None
                    and hasattr(detector.ai_detector, "set_runtime_tuning")
                ):
                    try:
                        set_tuning_fn = detector.ai_detector.set_runtime_tuning
                        accepted = set(inspect.signature(set_tuning_fn).parameters.keys())
                        tuned_kwargs = {
                            k: v for k, v in ai_runtime_tuning.items()
                            if k in accepted
                        }
                        if tuned_kwargs:
                            set_tuning_fn(**tuned_kwargs)
                    except Exception as _tuning_err:
                        st.warning(f"Runtime tuning values could not be fully applied: {_tuning_err}")
                
                progress_bar.progress(25, text="🌳 Detecting canopy zones... This may take a moment")
                detection_progress_state = {"total_tiles": None, "last_ui_tile": 0}

                def _estimate_total_tiles(image_file):
                    """Estimate full tile grid count before vegetation prefilter runs."""
                    if getattr(detector, "ai_detector", None) is None:
                        return None
                    image_preview = cv2.imread(str(image_file))
                    if image_preview is None:
                        return None
                    img_h, img_w = image_preview.shape[:2]
                    runtime_cfg = getattr(detector.ai_detector, "runtime_tuning", {}) or {}
                    tile_size = int(runtime_cfg.get("tile_size", 512))
                    overlap_val = runtime_cfg.get("tile_overlap", runtime_cfg.get("overlap", None))
                    if overlap_val is None:
                        overlap_px = 128
                    else:
                        overlap_f = float(overlap_val)
                        overlap_px = int(tile_size * overlap_f) if 0.0 <= overlap_f < 1.0 else int(overlap_f)
                    stride = max(1, tile_size - overlap_px)
                    x_count = len(range(0, img_w, stride))
                    y_count = len(range(0, img_h, stride))
                    return int(x_count * y_count)

                estimated_tiles = _estimate_total_tiles(temp_path)
                if estimated_tiles:
                    tile_status.info(
                        f"🧩 Tile estimate: about {estimated_tiles} total tiles. "
                        f"Checking vegetation tiles..."
                    )
                else:
                    tile_status.info("🧩 Tile counter initializing...")

                def _on_detection_progress(event, payload):
                    if not isinstance(payload, dict):
                        payload = {}
                    try:
                        if event == "tile_setup":
                            total_tiles = int(payload.get("total_tiles") or 0)
                            skipped_tiles = int(payload.get("skipped_tiles") or 0)
                            detection_progress_state["total_tiles"] = total_tiles
                            if total_tiles > 0:
                                setup_text = f"Detecting canopies... {total_tiles} tiles to process"
                                if skipped_tiles > 0:
                                    setup_text += f" ({skipped_tiles} skipped)"
                                progress_bar.progress(26, text=setup_text)
                                tile_status.info(
                                    f"Tile workload: {total_tiles} to process"
                                    + (f" ({skipped_tiles} skipped)" if skipped_tiles > 0 else "")
                                )
                        elif event == "tile_progress":
                            total_tiles = int(
                                payload.get("total_tiles")
                                or detection_progress_state.get("total_tiles")
                                or 0
                            )
                            current_tile = int(payload.get("current_tile") or 0)
                            if total_tiles > 0:
                                tile_fraction = min(1.0, max(0.0, current_tile / total_tiles))
                                bar_value = min(58, 26 + int(tile_fraction * 32))
                                update_every = max(1, total_tiles // 50)
                                show_tile_update = (
                                    current_tile == 1
                                    or current_tile == total_tiles
                                    or current_tile - detection_progress_state["last_ui_tile"] >= update_every
                                )
                                progress_bar.progress(
                                    bar_value,
                                    text=f"Detecting canopies... Tile {current_tile}/{total_tiles}",
                                )
                                if show_tile_update:
                                    tile_status.info(f"Processing tile {current_tile}/{total_tiles}")
                                    detection_progress_state["last_ui_tile"] = current_tile
                        elif event == "tile_complete":
                            total_tiles = int(
                                payload.get("total_tiles")
                                or detection_progress_state.get("total_tiles")
                                or 0
                            )
                            if total_tiles > 0:
                                progress_bar.progress(
                                    58,
                                    text=f"AI tile detection finished ({total_tiles}/{total_tiles})",
                                )
                                tile_status.success(f"Tile pass complete: {total_tiles}/{total_tiles}")
                    except Exception as _cb_err:
                        # UI progress updates must not interrupt image processing.
                        print(f"[Tile UI callback warning] {_cb_err}")
                # Process image
                results = detector.process_image(
                    image_path=str(temp_path),
                    canopy_buffer_m=canopy_buffer,
                    hexagon_size_m=hexagon_size,
                    progress_callback=_on_detection_progress,
                )

                if detection_progress_state.get("total_tiles") in (None, 0):
                    ai_meta = results.get("ai_metadata", {}) if isinstance(results, dict) else {}
                    fallback_tiles = int(
                        ai_meta.get("num_tiles")
                        or ai_meta.get("num_tiles_processed")
                        or 0
                    )
                    if fallback_tiles > 0:
                        tile_status.info(f"AI processed {fallback_tiles} tiles.")
                    else:
                        tile_status.info("Tile count unavailable for this detection run.")
                
                progress_bar.progress(60, text="⬡ Generating planting hexagons...")
                
                progress_bar.progress(65, text="🚫 Filtering forbidden & eroded zones...")
                
                # ── EARLY FORBIDDEN ZONE FILTERING ────────────────────────
                # Filter hexagons BEFORE visualization so the Visual Results
                # image also excludes planting points on bridges/towers/houses.
                results['_forbidden_filtered'] = 0
                if image_gps is not None and _forbidden_filter.zone_count > 0:
                    _gsd = results['gsd_m_per_pixel']
                    _w, _h = results['image_size']
                    
                    # Run ortho-matching (result is cached for reuse in map section)
                    _match_key = f"ortho_match_{uploaded_file.name}"
                    if _match_key not in st.session_state:
                        _match_result = match_drone_to_ortho(
                            drone_image=results['image'],
                            center_lat=image_center_lat,
                            center_lon=image_center_lon,
                            drone_gsd=_gsd,
                        )
                        st.session_state[_match_key] = _match_result
                    else:
                        _match_result = st.session_state[_match_key]
                    
                    # Build pixel→GPS converter
                    if _match_result['success']:
                        _H = _match_result['H']
                        def _px_to_gps(px, py):
                            return drone_pixel_to_gps_via_homography(px, py, _H)
                    else:
                        def _px_to_gps(px, py):
                            return drone_pixel_to_gps_via_heading(
                                px, py, _w, _h,
                                image_center_lat, image_center_lon,
                                _gsd, camera_heading
                            )
                    
                    # Filter: keep only hexagons outside forbidden AND eroded zones
                    _safe = []
                    _forbidden_hexes = []
                    _eroded_hexes = []
                    for _hex in results['hexagons']:
                        _px, _py = _hex['center']
                        _lat, _lon = _px_to_gps(_px, _py)
                        _hex['_gps_lat'] = _lat
                        _hex['_gps_lon'] = _lon
                        if not _forbidden_filter.is_safe_location(_lat, _lon):
                            _forbidden_hexes.append(_hex)
                        elif not _eroded_filter.is_safe_location(_lat, _lon):
                            _eroded_hexes.append(_hex)
                        else:
                            _safe.append(_hex)
                    
                    results['hexagons'] = _safe
                    results['hexagon_count'] = len(_safe)
                    results['_forbidden_filtered'] = len(_forbidden_hexes)
                    results['_eroded_filtered'] = len(_eroded_hexes)
                    results['_forbidden_hexagons'] = _forbidden_hexes
                    results['_eroded_hexagons'] = _eroded_hexes
                
                progress_bar.progress(75, text="🎨 Creating visualization...")
                
                # Create visualization (now with filtered hexagons)
                vis_image = detector.visualize_results(results)
                
                # Convert BGR to RGB for display
                vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
                
                # Store in session state
                st.session_state.last_analysis_key = analysis_key
                st.session_state.cached_results = results
                st.session_state.cached_vis_image = vis_image_rgb
                st.session_state.cached_detector = detector
                st.session_state.cached_image_gps = image_gps
                st.session_state.cached_image_center_lat = image_center_lat
                st.session_state.cached_image_center_lon = image_center_lon
            else:
                # Use cached results
                results = st.session_state.cached_results
                vis_image_rgb = st.session_state.cached_vis_image
                detector = st.session_state.cached_detector
                tile_status.info("Using cached analysis results (no live tile processing).")
            
            progress_bar.progress(90, text="📊 Preparing results...")
            
            # ✅ Analysis complete — fill progress bar
            progress_bar.progress(100, text="✅ Analysis complete!")
            
            # Display results
            st.markdown('<hr>', unsafe_allow_html=True)
            st.markdown('<h2 class="results-header">📊 Analysis Results</h2>', unsafe_allow_html=True)
            
            # Metrics row
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric(
                    label="🌳 Canopies Detected",
                    value=results['canopy_count']
                )
            
            with col2:
                st.metric(
                    label="🔴 Danger Zones",
                    value=f"{results['danger_area_m2']:.1f} m²",
                    delta=f"{results['danger_percentage']:.1f}%"
                )
            
            with col3:
                st.metric(
                    label="🟢 Plantable Area",
                    value=f"{results['plantable_area_m2']:.1f} m²",
                    delta=f"{results['plantable_percentage']:.1f}%"
                )
            
            with col4:
                st.metric(
                    label="⬡ Planting Hexagons",
                    value=results['hexagon_count']
                )
            
            st.markdown("---")
            
            # Image comparison
            st.markdown("### 🖼️ Visual Results")
            img_col1, img_col2 = st.columns(2, gap="large")
            
            with img_col1:
                st.markdown("**Original Image**")
                original_rgb = cv2.cvtColor(results['image'], cv2.COLOR_BGR2RGB)
                st.image(original_rgb, width='stretch')
            
            with img_col2:
                st.markdown("**Detected Zones**")
                st.image(vis_image_rgb, width='stretch')

            st.caption(
                "Legend: 🟣 Canopy Areas | 🔴 1m Danger Buffer | 🟢 1m Planting Buffer | "
                "🟠 Overlap Warning | 🟩 Planting Points"
            )
            
            st.markdown("---")
            
            # Detailed statistics
            st.markdown("### 📈 Detailed Statistics")
            
            stat_col1, stat_col2 = st.columns(2)
            
            with stat_col1:
                st.markdown(f"""
                **🗺️ Coverage Information:**
                - Ground Sample Distance: {results['gsd_m_per_pixel']*100:.3f} cm/pixel
                - Coverage Area: {results['coverage_m'][0]:.2f}m × {results['coverage_m'][1]:.2f}m
                - Total Area: {results['total_area_m2']:.2f} m²
                - Flight Altitude: {results['altitude_m']:.1f} meters
                """)
            
            with stat_col2:
                st.markdown(f"""
                **🌿 Detection Results:**
                - Mangrove Canopies: {results['canopy_count']}
                - Canopy Danger Buffer: {results['canopy_buffer_m']:.1f} meters (red zone)
                - Hexagon Buffer Size: {results['hexagon_size_m']:.1f} meters (green zone)
                - Planting Points: {results['hexagon_count']} (maximized coverage)
                """)
            
            # Add color legend explanation
            st.markdown("---")
            st.markdown("### 🎨 Color Legend Explanation")
            legend_col1, legend_col2, legend_col3, legend_col4, legend_col5 = st.columns(5)
            
            with legend_col1:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #800080 0%, #9932CC 100%); padding: 1rem; border-radius: 8px; text-align: center;'>
                    <h4 style='color: white; margin: 0;'>🟣 Canopies</h4>
                    <p style='color: white; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>Detected vegetation</p>
                </div>
                """, unsafe_allow_html=True)
            
            with legend_col2:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #DC143C 0%, #FF0000 100%); padding: 1rem; border-radius: 8px; text-align: center;'>
                    <h4 style='color: white; margin: 0;'>🔴 Danger Buffer</h4>
                    <p style='color: white; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>1m safety zone</p>
                </div>
                """, unsafe_allow_html=True)
            
            with legend_col3:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #90EE90 0%, #98FB98 100%); padding: 1rem; border-radius: 8px; text-align: center;'>
                    <h4 style='color: #2D5F3F; margin: 0;'>🟢 Planting Buffer</h4>
                    <p style='color: #2D5F3F; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>1m safe zones</p>
                </div>
                """, unsafe_allow_html=True)
            
            with legend_col4:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #FF8C00 0%, #FFA500 100%); padding: 1rem; border-radius: 8px; text-align: center;'>
                    <h4 style='color: white; margin: 0;'>🟠 Overlap</h4>
                    <p style='color: white; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>Buffer warning zone</p>
                </div>
                """, unsafe_allow_html=True)
            
            with legend_col5:
                st.markdown("""
                <div style='background: linear-gradient(135deg, #006400 0%, #228B22 100%); padding: 1rem; border-radius: 8px; text-align: center;'>
                    <h4 style='color: white; margin: 0;'>🟩 Planting Points</h4>
                    <p style='color: white; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>Exact locations</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Download section
            st.markdown("---")
            st.markdown("### ⬇️ Export Results")
            
            download_col1, download_col2, download_col3 = st.columns(3)
            
            with download_col1:
                # Convert visualization to bytes
                vis_pil = Image.fromarray(vis_image_rgb)
                buf = io.BytesIO()
                vis_pil.save(buf, format='PNG')
                
                st.download_button(
                    label="📥 Download Visualization",
                    data=buf.getvalue(),
                    file_name="mangrovision_analysis.png",
                    mime="image/png"
                )
            
            with download_col2:
                # JSON results
                json_results = {
                    'canopy_count': results['canopy_count'],
                    'danger_area_m2': results['danger_area_m2'],
                    'plantable_area_m2': results['plantable_area_m2'],
                    'hexagon_count': results['hexagon_count'],
                    'gsd': results['gsd_m_per_pixel'],
                    'coverage': results['coverage_m']
                }
                
                st.download_button(
                    label="📄 Download JSON Data",
                    data=json.dumps(json_results, indent=2),
                    file_name="mangrovision_data.json",
                    mime="application/json"
                )
            
            with download_col3:
                # GeoJSON export (for QGIS)
                _export_wps = hexagons_to_waypoints(
                    results.get('hexagons', []),
                    image_name=uploaded_file.name,
                )
                _export_meta = {
                    "image_name": uploaded_file.name,
                    "analyzed_at": datetime.now().isoformat(timespec="seconds"),
                    "detection_mode": detection_mode,
                }
                if _export_wps:
                    st.download_button(
                        label="🗺️ Download GeoJSON",
                        data=generate_geojson(_export_wps, _export_meta),
                        file_name=f"mangrovision_{uploaded_file.name}.geojson",
                        mime="application/geo+json",
                    )
                else:
                    st.info("🗺️ GeoJSON available after GPS mapping (below)")
            
            # STEP 3: Geotag results on orthophoto map
            # Use cached GPS data if available from previous run
            map_image_gps = st.session_state.get('cached_image_gps', image_gps)
            map_center_lat = st.session_state.get('cached_image_center_lat', image_center_lat)
            map_center_lon = st.session_state.get('cached_image_center_lon', image_center_lon)
            
            if map_image_gps is not None:
                st.markdown("---")
                st.markdown("### 🗺️ Geotagged Map View")
                st.info("📍 Showing GPS markers for safe planting locations - each green point shows exact coordinates where mangroves can be planted")
                
                # Extract detection data for mapping
                gsd = results['gsd_m_per_pixel']
                canopy_polygons = results['canopy_polygons']
                hexagons = results['hexagons']  # Already filtered by forbidden & eroded zones
                width, height = results['image_size']
                forbidden_filtered_count = results.get('_forbidden_filtered', 0)
                eroded_filtered_count = results.get('_eroded_filtered', 0)
                forbidden_hexagons = results.get('_forbidden_hexagons', [])
                eroded_hexagons = results.get('_eroded_hexagons', [])

                # ── AUTO-ALIGN: reuse cached ortho match ─────────────────
                match_key = f"ortho_match_{uploaded_file.name}"
                if match_key not in st.session_state:
                    with st.spinner("🔍 Auto-aligning drone image with orthophoto map…"):
                        match_result = match_drone_to_ortho(
                            drone_image=results['image'],
                            center_lat=map_center_lat,
                            center_lon=map_center_lon,
                            drone_gsd=gsd,
                        )
                    st.session_state[match_key] = match_result
                else:
                    match_result = st.session_state[match_key]

                if match_result['success']:
                    H_matrix = match_result['H']
                    detected_heading = match_result['heading']
                    match_conf = match_result['confidence']
                    st.success(
                        f"✅ Auto-alignment successful — "
                        f"{match_result['inliers']}/{match_result['total_matches']} inlier matches, "
                        f"confidence {match_conf:.0%}, "
                        f"detected heading {detected_heading:.1f}°"
                    )

                    def pixel_to_latlon(px, py):
                        return drone_pixel_to_gps_via_homography(px, py, H_matrix)
                else:
                    # Fallback: use heading-based conversion
                    st.warning(
                        f"⚠️ Auto-alignment not available ({match_result['error']}). "
                        f"Using heading-based fallback ({camera_heading}°). "
                        f"Adjust heading slider for better accuracy."
                    )
                    detected_heading = camera_heading

                    def pixel_to_latlon(px, py):
                        return drone_pixel_to_gps_via_heading(
                            px, py, width, height,
                            map_center_lat, map_center_lon,
                            gsd, camera_heading
                        )
                
                # Create orthophoto map centered on image location
                from folium.plugins import Fullscreen
                ortho_map = folium.Map(
                    location=[map_center_lat, map_center_lon],
                    zoom_start=20,
                    tiles=None,
                    control_scale=True,
                    max_bounds=True
                )
                _add_operational_map_layers(ortho_map)
                
                st.info(f"📍 Map shows GPS markers for each planting location from Visual Results")
                image_width_m = width * gsd
                image_height_m = height * gsd
                
                # Add image center marker
                folium.Marker(
                    location=[map_center_lat, map_center_lon],
                    popup=f"""📷 <b>Image Center</b><br>
                    Altitude: {altitude_to_use:.1f}m<br>
                    GSD: {gsd*100:.2f} cm/pixel<br>
                    Coverage: {image_width_m:.1f}m × {image_height_m:.1f}m<br>
                    Heading: {detected_heading:.1f}°""",
                    icon=folium.Icon(color='blue', icon='camera', prefix='fa'),
                    tooltip="📷 Image Location"
                ).add_to(ortho_map)
                
                # ── Hexagons are already filtered by forbidden & eroded zones ──
                # (filtering was done before visualization so Visual Results
                #  image also excludes forbidden/eroded zone hexagons)
                safe_hexagons = hexagons  # Already safe — filtered earlier

                _filter_msgs = []
                if forbidden_filtered_count > 0:
                    _filter_msgs.append(f"🚫 {forbidden_filtered_count} in forbidden zones (towers/bridges/houses)")
                if eroded_filtered_count > 0:
                    _filter_msgs.append(f"🏜️ {eroded_filtered_count} in eroded zones")
                if _filter_msgs:
                    st.warning(f"Planting points filtered out: {'; '.join(_filter_msgs)}. {len(safe_hexagons)} safe points remain.")

                # ── Draw forbidden zone polygons on the map (red) ─────────
                if _forbidden_filter.forbidden_polygons:
                    fz_group = folium.FeatureGroup(name='Forbidden Zones', show=False)
                    for poly in _forbidden_filter.forbidden_polygons:
                        # Shapely polygon coords are (lon, lat); Folium needs (lat, lon)
                        coords = [(lat, lon) for lon, lat in poly.exterior.coords]
                        folium.Polygon(
                            locations=coords,
                            color='red',
                            fill=True,
                            fillColor='red',
                            fillOpacity=0.35,
                            weight=2,
                            tooltip='🚫 Forbidden Zone (tower/bridge/house)',
                        ).add_to(fz_group)
                    fz_group.add_to(ortho_map)

                # ── Draw eroded zone polygons on the map (orange) ─────────
                if _eroded_filter.forbidden_polygons:
                    ez_group = folium.FeatureGroup(name='Eroded Zones', show=False)
                    for poly in _eroded_filter.forbidden_polygons:
                        coords = [(lat, lon) for lon, lat in poly.exterior.coords]
                        folium.Polygon(
                            locations=coords,
                            color='orange',
                            fill=True,
                            fillColor='orange',
                            fillOpacity=0.35,
                            weight=2,
                            tooltip='🏜️ Eroded Zone (erosion area — not plantable)',
                        ).add_to(ez_group)
                    ez_group.add_to(ortho_map)

                # Ensure hexagons have GPS coords (compute if not pre-computed)
                for hexagon in safe_hexagons:
                    if '_gps_lat' not in hexagon:
                        px, py = hexagon['center']
                        lat, lon = pixel_to_latlon(px, py)
                        hexagon['_gps_lat'] = lat
                        hexagon['_gps_lon'] = lon

                # ── CLIP: Remove planting points that land OUTSIDE orthophoto map ──
                _before_clip = len(safe_hexagons)
                safe_hexagons = [
                    h for h in safe_hexagons
                    if is_inside_any_orthophoto(h['_gps_lat'], h['_gps_lon'])
                ]
                _clipped_out = _before_clip - len(safe_hexagons)
                if _clipped_out > 0:
                    st.info(f"🗺️ {_clipped_out} planting points removed — outside orthophoto map coverage. {len(safe_hexagons)} remain.")

                # Add RED X markers for forbidden-filtered hexagons
                if forbidden_hexagons:
                    fz_pts = folium.FeatureGroup(name='Filtered Forbidden Points', show=False)
                    for fh in forbidden_hexagons:
                        lat = fh['_gps_lat']
                        lon = fh['_gps_lon']
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=4,
                            color='#B71C1C',
                            fillColor='#F44336',
                            fillOpacity=0.7,
                            weight=2,
                            tooltip='🚫 Filtered (forbidden zone)',
                            popup=f"""🚫 <b>Filtered Point</b><br>
                            Reason: Inside forbidden zone<br>
                            GPS: {lat:.7f}°, {lon:.7f}°""",
                        ).add_to(fz_pts)
                    fz_pts.add_to(ortho_map)

                # Add ORANGE X markers for eroded-filtered hexagons
                if eroded_hexagons:
                    ez_pts = folium.FeatureGroup(name='Filtered Eroded Points', show=False)
                    for eh in eroded_hexagons:
                        lat = eh['_gps_lat']
                        lon = eh['_gps_lon']
                        folium.CircleMarker(
                            location=[lat, lon],
                            radius=4,
                            color='#E65100',
                            fillColor='#FF9800',
                            fillOpacity=0.7,
                            weight=2,
                            tooltip='🏜️ Filtered (eroded zone)',
                            popup=f"""🏜️ <b>Filtered Point</b><br>
                            Reason: Inside eroded zone<br>
                            GPS: {lat:.7f}°, {lon:.7f}°""",
                        ).add_to(ez_pts)
                    ez_pts.add_to(ortho_map)

                # Add GREEN POINTS only for safe planting hexagons
                planting_group = folium.FeatureGroup(name='Planting Zones', show=True)
                for i, hexagon in enumerate(safe_hexagons):
                    lat = hexagon['_gps_lat']
                    lon = hexagon['_gps_lon']

                    folium.CircleMarker(
                        location=[lat, lon],
                        radius=4,
                        popup=f"""🌱 <b>Planting Point #{i+1}</b><br>
                        GPS: {lat:.7f}°, {lon:.7f}°<br>
                        Pixel: ({int(hexagon['center'][0])}, {int(hexagon['center'][1])})<br>
                        Buffer: {hexagon.get('buffer_radius_m', 'N/A')}m<br>
                        Area: {hexagon.get('area_m2', hexagon.get('area_sqm', 0)):.2f} m²""",
                        tooltip=f"🌱 Point #{i+1}",
                        color='#1B5E20',
                        fillColor='#4CAF50',
                        fillOpacity=0.8,
                        weight=2
                    ).add_to(planting_group)
                planting_group.add_to(ortho_map)
                
                # Add layer control
                Fullscreen(position="topleft", title="Expand map", title_cancel="Exit fullscreen").add_to(ortho_map)
                folium.LayerControl(collapsed=False).add_to(ortho_map)
                _style_layer_control(ortho_map)
                
                # Display map
                st_folium.st_folium(
                    ortho_map,
                    height=600,
                    key="geo_map",
                    returned_objects=[],
                    use_container_width=True,
                )
                
                # Show planting coordinates table
                st.markdown("---")
                st.markdown("### 📍 Planting Location Coordinates")
                st.info("ℹ️ Use Visual Results above to see where each point is located in the analyzed image")
                
                # Create coordinate dataframe with GPS (only safe hexagons)
                coord_data = []
                for i, hexagon in enumerate(safe_hexagons, 1):
                    lat = hexagon['_gps_lat']
                    lon = hexagon['_gps_lon']
                    px, py = hexagon['center']
                    coord_data.append({
                        "Point #": i,
                        "Latitude": f"{lat:.7f}",
                        "Longitude": f"{lon:.7f}",
                        "Pixel X": int(px),
                        "Pixel Y": int(py),
                        "Buffer (m)": hexagon['buffer_radius_m'],
                        "Area (m²)": round(hexagon['area_m2'], 2)
                    })
                
                if coord_data:
                    df = pd.DataFrame(coord_data)
                    st.dataframe(df, use_container_width=True)
                    
                    # ── Waypoint Export Buttons ────────────────────────
                    st.markdown("#### 📡 Export Waypoints for Field Navigation")
                    _wp_list = hexagons_to_waypoints(
                        safe_hexagons,
                        image_name=uploaded_file.name,
                    )
                    _wp_meta = {
                        "image_name": uploaded_file.name,
                        "analyzed_at": datetime.now().isoformat(timespec="seconds"),
                        "detection_mode": detection_mode,
                        "total_points": len(_wp_list),
                    }

                    _dl1, _dl2, _dl3, _dl4 = st.columns(4)
                    with _dl1:
                        csv = df.to_csv(index=False)
                        st.download_button(
                            label="📥 CSV",
                            data=csv,
                            file_name=f"planting_coordinates_{uploaded_file.name}.csv",
                            mime="text/csv",
                            use_container_width=True,
                        )
                    with _dl2:
                        st.download_button(
                            label="📡 GPX (GPS)",
                            data=generate_gpx(_wp_list, _wp_meta),
                            file_name=f"planting_waypoints_{uploaded_file.name}.gpx",
                            mime="application/gpx+xml",
                            use_container_width=True,
                        )
                    with _dl3:
                        st.download_button(
                            label="🌍 KML (Google Earth)",
                            data=generate_kml(_wp_list, _wp_meta),
                            file_name=f"planting_waypoints_{uploaded_file.name}.kml",
                            mime="application/vnd.google-earth.kml+xml",
                            use_container_width=True,
                        )
                    with _dl4:
                        st.download_button(
                            label="🗺️ GeoJSON (QGIS)",
                            data=generate_geojson(_wp_list, _wp_meta),
                            file_name=f"planting_waypoints_{uploaded_file.name}.geojson",
                            mime="application/geo+json",
                            use_container_width=True,
                        )
                    st.caption("💡 GPX works with Garmin, Locus Map, OsmAnd. KML opens in Google Earth. GeoJSON imports into QGIS.")
                
                if forbidden_filtered_count > 0:
                    st.success(f"✅ Analysis complete! {len(safe_hexagons)} safe planting locations shown on map ({forbidden_filtered_count} filtered out from forbidden zones). Use Visual Results to see exact positions.")
                else:
                    st.success(f"✅ Analysis complete! {len(safe_hexagons)} GPS-tagged planting locations shown on map. Use Visual Results to see exact positions.")
                
                # ── Manual save to database ──────────────────────────
                _save_key = f"saved_{st.session_state.get('last_analysis_key', '')}"
                if _save_key in st.session_state:
                    st.caption(f"💾 Already saved (Analysis #{st.session_state[_save_key]}).")
                else:
                    st.info("Results are not yet saved to Map Analytics.")
                    _save_btn_key = f"save_btn_{st.session_state.get('last_analysis_key', 'current')}"
                    if st.button("💾 Save to database", key=_save_btn_key, type="primary"):
                        try:
                            _aid, _new, _skipped = save_analysis(
                                image_name=uploaded_file.name,
                                center_lat=map_center_lat,
                                center_lon=map_center_lon,
                                results=results,
                                hexagons=safe_hexagons,
                                user_id=st.session_state.get('user_id'),
                            )
                            st.session_state[_save_key] = _aid
                            if _skipped > 0:
                                st.success(f"✅ Saved {_new} new planting points (Analysis #{_aid}). {_skipped} duplicates skipped.")
                            else:
                                st.success(f"✅ Planting data saved (Analysis #{_aid}, {_new} points).")
                            st.caption("Saved points are now available in **Map Analytics**.")
                        except Exception as _db_err:
                            st.warning(f"⚠️ Could not save to database: {_db_err}")
            else:
                st.warning("⚠️ GPS mapping skipped - no GPS data in image")
                st.success("✅ Analysis complete! Results ready for export.")
            
        except Exception as e:
            progress_bar.progress(100, text="❌ Error during analysis")
            st.error(f"❌ Error processing image: {str(e)}")
            st.exception(e)
        
        finally:
            # Cleanup temporary files
            if temp_path.exists():
                temp_path.unlink()


if __name__ == "__main__":
    main()


