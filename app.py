"""
MangroVision - AI-Powered Mangrove Planting Zone Analyzer
Beautiful Streamlit UI for the thesis project
"""

import sys
# Force UTF-8 output so emoji in print() don't crash on Windows (cp1252 terminals)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

import streamlit as st
import cv2
import numpy as np
import pandas as pd
from PIL import Image
from pathlib import Path
import io
import json
import streamlit_folium as st_folium
import folium
from pyproj import Transformer

# Add canopy_detection to path
sys.path.append(str(Path(__file__).parent / "canopy_detection"))

from canopy_detection.canopy_detector_hexagon import HexagonDetector
from canopy_detection.exif_extractor import ExifExtractor
from canopy_detection.auto_align import detect_camera_heading, snap_to_cardinal
from canopy_detection.flight_log_parser import FlightLogParser, snap_to_cardinal as snap_heading
from canopy_detection.reference_matcher import ReferencePointMatcher
from canopy_detection.ortho_matcher import (
    match_drone_to_ortho,
    drone_pixel_to_gps_via_homography,
    drone_pixel_to_gps_via_heading,
)
from canopy_detection.forbidden_zone_filter import ForbiddenZoneFilter

# Load forbidden zones (towers, bridges, houses) once at startup
_FORBIDDEN_ZONES_PATH = Path(__file__).parent / "forbidden_zones.geojson"
_forbidden_filter = ForbiddenZoneFilter(str(_FORBIDDEN_ZONES_PATH))

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
        --primary-green: #2D5F3F;
        --secondary-green: #4A9D6F;
        --accent-green: #7EC88D;
        --light-green: #C8E6C9;
        --background: #23395d;
    }
    
    /* Header styling */
    .main-header {
        background: linear-gradient(135deg, #2D5F3F 0%, #4A9D6F 100%);
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
        color: #C8E6C9;
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
    }
    
    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #1a1a1a 0%, #2d2d2d 100%);
    }
    
    section[data-testid="stSidebar"] > div {
        padding-top: 2rem;
    }
    
    section[data-testid="stSidebar"] h3 {
        color: #7EC88D !important;
        font-weight: 600;
    }
    
    section[data-testid="stSidebar"] label {
        color: #C8E6C9 !important;
        font-weight: 500;
    }
    
    section[data-testid="stSidebar"] p {
        color: #B0B0B0 !important;
    }
    
    section[data-testid="stSidebar"] .stMarkdown {
        color: #E0E0E0 !important;
    }
    
    /* Sidebar select boxes and inputs */
    section[data-testid="stSidebar"] .stSelectbox > div > div {
        background-color: #3a3a3a;
        color: #E0E0E0;
    }
    
    section[data-testid="stSidebar"] input {
        background-color: #3a3a3a;
        color: #E0E0E0;
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
        background: linear-gradient(135deg, #2D5F3F 0%, #4A9D6F 100%);
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
        box-shadow: 0 6px 12px rgba(0,0,0,0.15);
    }
    
    /* Download button */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #1976D2 0%, #2196F3 100%);
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
        color: #2D5F3F;
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
</style>
""", unsafe_allow_html=True)


def show_orthophoto_detection_viewer():
    """
    Display tree crown detection results from GeoJSON on orthophoto map
    """
    st.markdown("## 🗺️ Orthophoto Tree Crown Detection Results")
    
    st.markdown("""
    <div class="info-box">
        <strong>📍 View & Validate Tree Crown Detections</strong><br>
        <span style="color: #1565C0;">
        This mode displays tree crowns detected from the high-resolution orthophoto using
        the samgeo + detectree2 pipeline. Each detected crown is shown as a polygon with
        its area calculated in m².<br><br>
        
        To generate detection results, run:<br>
        <code>python canopy_detection/samgeo_ortho_detector.py</code>
        </span>
    </div>
    """, unsafe_allow_html=True)
    
    # Check for GeoJSON files
    output_dir = Path(__file__).parent / 'output_geojson'
    geojson_files = []
    if output_dir.exists():
        geojson_files = list(output_dir.glob('*.geojson')) + list(output_dir.glob('*.json'))
    geojson_files = sorted(geojson_files, key=lambda x: x.stat().st_mtime, reverse=True)
    
    if not geojson_files:
        st.warning("""
        ⚠️ No GeoJSON detection files found in `output_geojson/` folder.
        
        **To create detections:**
        1. Install samgeo: `pip install segment-geospatial`
        2. Run detection: `python canopy_detection/samgeo_ortho_detector.py`
        3. Refresh this page
        
        See **INSTALL_SAMGEO_DETECTREE2.md** for complete guide.
        """)
        return
    
    # GeoJSON file selector
    selected_geojson = st.selectbox(
        "📁 Select GeoJSON File",
        options=[f.name for f in geojson_files],
        index=0
    )
    
    geojson_path = output_dir / selected_geojson
    
    # Load GeoJSON
    with open(geojson_path, 'r') as f:
        geojson_data = json.load(f)
    
    features = geojson_data.get('features', [])
    
    if not features:
        st.error("❌ GeoJSON file contains no features")
        return
    
    # Calculate statistics
    tree_count = len(features)
    areas = []
    for feature in features:
        props = feature.get('properties', {})
        area = props.get('area_pixels', props.get('area', 0.0))  # Support both field names
        if area > 0:
            areas.append(area)
    
    total_area_px = sum(areas) if areas else 0.0
    avg_area_px = total_area_px / len(areas) if areas else 0.0
    min_area_px = min(areas) if areas else 0.0
    max_area_px = max(areas) if areas else 0.0
    
    # Display metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("🌳 Trees Detected", tree_count)
    
    with col2:
        st.metric("📏 Total Canopy Area", f"{total_area_px:.0f} px²")
    
    with col3:
        st.metric("📊 Average Crown Size", f"{avg_area_px:.0f} px²")
    
    with col4:
        st.metric("📐 Size Range", f"{min_area_px:.0f} - {max_area_px:.0f} px²")
    
    # Map display
    st.markdown("---")
    st.markdown("### 🗺️ Interactive Map")
    
    # Calculate center from features
    all_coords = []
    for feature in features:
        geom = feature.get('geometry', {})
        if geom.get('type') == 'Polygon':
            coords = geom.get('coordinates', [[]])[0]
            all_coords.extend(coords)
        elif geom.get('type') == 'MultiPolygon':
            for poly in geom.get('coordinates', []):
                all_coords.extend(poly[0] if poly else [])
    
    if all_coords:
        lons = [c[0] for c in all_coords]
        lats = [c[1] for c in all_coords]
        center_lat = sum(lats) / len(lats)
        center_lon = sum(lons) / len(lons)
    else:
        # Default to Leganes, Iloilo
        center_lat = 10.7826
        center_lon = 122.5942
    
    # Create map
    m = folium.Map(
        location=[center_lat, center_lon],
        zoom_start=18,
        tiles=None,
        control_scale=True
    )
    
    # Add tile layers
    folium.TileLayer(
        tiles="http://localhost:8080/{z}/{x}/{y}.jpg",
        attr='MangroVision Orthophoto',
        name='Orthophoto',
        overlay=False,
        control=True,
        max_zoom=22,
        min_zoom=15,
        show=True
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}',
        attr='Esri WorldImagery',
        name='Satellite',
        overlay=False,
        control=True,
        show=False
    ).add_to(m)
    
    folium.TileLayer(
        tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
        attr='© OpenStreetMap',
        name='OpenStreetMap',
        overlay=False,
        control=True,
        show=False
    ).add_to(m)
    
    # Add GeoJSON layer with green styling
    style_function = lambda x: {
        'fillColor': '#4CAF50',
        'color': '#1B5E20',
        'weight': 2,
        'fillOpacity': 0.4
    }
    
    highlight_function = lambda x: {
        'fillColor': '#81C784',
        'color': '#1B5E20',
        'weight': 3,
        'fillOpacity': 0.7
    }
    
    # Determine which field to use for area
    sample_props = features[0].get('properties', {}) if features else {}
    area_field = 'area_pixels' if 'area_pixels' in sample_props else 'area'
    area_label = 'Crown Area (px²):' if area_field == 'area_pixels' else 'Crown Area (m²):'
    
    folium.GeoJson(
        geojson_data,
        name='Tree Crowns',
        style_function=style_function,
        highlight_function=highlight_function,
        tooltip=folium.GeoJsonTooltip(
            fields=[area_field, 'tree_id', 'confidence'],
            aliases=[area_label, 'Tree ID:', 'Confidence:'],
            localize=True
        ),
        popup=folium.GeoJsonPopup(
            fields=[area_field, 'tree_id', 'confidence'],
            aliases=[area_label, 'Tree ID:', 'Confidence:'],
            localize=True
        )
    ).add_to(m)
    
    # Add layer control
    folium.LayerControl().add_to(m)
    
    # Display map
    st_folium.st_folium(m, width=1400, height=700, key="geojson_map", returned_objects=[])
    
    # Validation section
    st.markdown("---")
    st.markdown("### 🔍 Quality Validation")
    
    if st.button("📊 Run Validation Analysis"):
        try:
            from canopy_detection.validation_metrics import DetectionValidator
            
            with st.spinner("Running validation..."):
                validator = DetectionValidator(str(geojson_path))
                
                # Area statistics
                stats = validator.calculate_area_statistics()
                
                st.markdown("#### 📊 Crown Area Distribution")
                stat_col1, stat_col2, stat_col3 = st.columns(3)
                
                with stat_col1:
                    st.metric("Mean Area", f"{stats['mean_area_m2']:.2f} m²")
                    st.metric("Median Area", f"{stats['median_area_m2']:.2f} m²")
                
                with stat_col2:
                    st.metric("Std Deviation", f"{stats['std_area_m2']:.2f} m²")
                    st.metric("Min Area", f"{stats['min_area_m2']:.2f} m²")
                
                with stat_col3:
                    st.metric("Max Area", f"{stats['max_area_m2']:.2f} m²")
                    st.metric("Total Area", f"{stats['total_area_m2']:.1f} m²")
                
                # Size validation
                st.markdown("#### ✓ Size Validity Check")
                size_check = validator.validate_crown_sizes(min_expected_m2=0.5, max_expected_m2=50.0)
                
                if size_check['passed']:
                    st.success(f"✅ {size_check['valid_percent']:.1f}% of detections have realistic sizes ({size_check['expected_range_m2']} m²)")
                else:
                    st.warning(f"⚠️ Only {size_check['valid_percent']:.1f}% of detections have realistic sizes")
                
                val_col1, val_col2, val_col3 = st.columns(3)
                with val_col1:
                    st.metric("Valid Sizes", size_check['valid_size'])
                with val_col2:
                    st.metric("Too Small", size_check['too_small'])
                with val_col3:
                    st.metric("Too Large", size_check['too_large'])
                
                # Commission errors
                st.markdown("#### ❌ Commission Error Detection")
                commission = validator.detect_commission_errors(max_reasonable_area_m2=100.0)
                
                if commission['commission_rate_percent'] < 5:
                    st.success(f"✅ Low commission rate: {commission['commission_rate_percent']:.2f}%")
                elif commission['commission_rate_percent'] < 15:
                    st.warning(f"⚠️ Moderate commission rate: {commission['commission_rate_percent']:.2f}%")
                else:
                    st.error(f"❌ High commission rate: {commission['commission_rate_percent']:.2f}%")
                
                if commission['errors']:
                    with st.expander("View suspected errors"):
                        for error in commission['errors'][:10]:
                            st.write(f"• Feature #{error['feature_id']}: {error['reason']}")
                
        except ImportError:
            st.error("Validation module not found. Please ensure validation_metrics.py exists.")
        except Exception as e:
            st.error(f"Validation error: {str(e)}")
    
    # Download section
    st.markdown("---")
    st.markdown("### ⬇️  Export Data")
    
    dl_col1, dl_col2 = st.columns(2)
    
    with dl_col1:
        with open(geojson_path, 'r') as f:
            geojson_str = f.read()
        
        st.download_button(
            label="📥 Download GeoJSON",
            data=geojson_str,
            file_name=selected_geojson,
            mime="application/geo+json"
        )
    
    with dl_col2:
        # Create CSV of crown areas
        csv_data = "tree_id,area_m2,centroid_lat,centroid_lon\n"
        for idx, feature in enumerate(features):
            props = feature.get('properties', {})
            area = props.get('area', 0.0)
            geom = feature.get('geometry', {})
            
            # Calculate centroid (simplified)
            if geom.get('type') == 'Polygon':
                coords = geom.get('coordinates', [[]])[0]
                if coords:
                    avg_lon = sum(c[0] for c in coords) / len(coords)
                    avg_lat = sum(c[1] for c in coords) / len(coords)
                    csv_data += f"{idx+1},{area:.2f},{avg_lat:.7f},{avg_lon:.7f}\n"
        
        st.download_button(
            label="📄 Download CSV Summary",
            data=csv_data,
            file_name=f"{selected_geojson.replace('.geojson', '')}_summary.csv",
            mime="text/csv"
        )


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌿 MangroVision</h1>
        <p>AI-Powered Mangrove Planting Zone Analyzer for Leganes, Iloilo</p>
        <p style="font-size: 0.9rem; margin-top: 0.5rem;">Intelligent tree crown detection & safe zone mapping</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Mode selector
    st.markdown("---")
    mode = st.radio(
        "📋 Select Analysis Mode",
        options=["🖼️ Analyze Drone Image", "🗺️ View Orthophoto Tree Crown Detection"],
        index=0,
        horizontal=True
    )
    st.markdown("---")
    
    # Route to appropriate analysis mode
    if mode == "🗺️ View Orthophoto Tree Crown Detection":
        show_orthophoto_detection_viewer()
        return
    
    # Sidebar
    with st.sidebar:
        # Logo header
        st.markdown("""
        <div style="background: linear-gradient(135deg, #2D5F3F 0%, #4A9D6F 100%); 
                    padding: 1.5rem; border-radius: 10px; text-align: center; margin-bottom: 1.5rem;">
            <h1 style="color: white; margin: 0; font-size: 2rem;">🌿 MangroVision</h1>
            <p style="color: #C8E6C9; margin: 0.5rem 0 0 0; font-size: 0.9rem;">AI Planting Zone Analyzer</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ⚙️ Detection Settings")
        
        # Check if detectree2 is available
        try:
            from canopy_detection.detectree2_detector import Detectree2Detector
            detectree2_available = True
            st.success("🌳 **Smart Hybrid System** (HSV + AI) Ready")
        except ImportError:
            detectree2_available = False
            st.warning("🌳 Using **HSV detection** (detectree2 not installed)")
            st.caption("Install detectree2 for AI-powered hybrid detection")
        
        # Detection mode selector (NEW!)
        if detectree2_available:
            detection_mode = st.selectbox(
                "🧠 Detection Mode",
                ["hybrid", "ai", "hsv"],
                index=0,
                format_func=lambda x: {
                    "hybrid": "⭐ Smart Hybrid (HSV + AI) - RECOMMENDED",
                    "ai": "🤖 AI Only (may miss trees)",
                    "hsv": "⚡ HSV Only (fast, good coverage)"
                }[x],
                help="Hybrid merges HSV and AI for 90-95% accuracy. AI-only may miss shadows/small trees."
            )
        else:
            detection_mode = "hsv"
            st.caption("⚡ Mode: HSV Only (AI not available)")
        
        # AI confidence slider - Only show for AI/Hybrid modes
        if detection_mode in ['ai', 'hybrid']:
            ai_confidence = st.slider(
                "AI Confidence Threshold",
                min_value=0.3,
                max_value=0.9,
                value=0.5,
                step=0.05,
                help="Higher = only high-confidence detections. 0.5-0.6 is standard for good models."
            )
        else:
            ai_confidence = 0.5  # Default value for HSV mode
        
        # Model selection
        model_name = st.selectbox(
            "Detectree2 Model",
            ["benchmark", "paracou"],
            help="benchmark: General trees | paracou: Tropical forests (best for mangroves)"
        )
        
        # Species selection (placeholder for future)
        species = st.selectbox(
            "Mangrove Species",
            ["Bungalon (All Species)", "Bungalon - Avicennia", "Bungalon - Rhizophora"],
            help="Select the mangrove species to detect"
        )
        
        st.markdown("---")
        
        st.markdown("### 📏 Buffer Settings")
        
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
        
        st.markdown("---")
        
        st.markdown("### 🚁 Flight Parameters")
        
        altitude = st.number_input(
            "Flight Altitude (meters)",
            min_value=3.0,
            max_value=20.0,
            value=6.0,
            step=0.5,
            help="Drone flight altitude"
        )
        
        drone_model = st.selectbox(
            "Drone Model",
            ["GENERIC_4K", "DJI_MINI_3", "DJI_MAVIC_3", "DJI_AIR_2S", "DJI_PHANTOM_4"],
            help="Select your drone model for accurate GSD calculation"
        )
        
        st.markdown("---")
        
        st.markdown("""
        <div style="background: linear-gradient(135deg, #1E3A2E 0%, #2D5F3F 100%);
                    padding: 1.2rem; border-radius: 10px; border-left: 4px solid #7EC88D; margin: 1rem 0;">
            <strong style="color: #7EC88D; font-size: 1.1rem;">ℹ️ About</strong><br>
            <span style="color: #C8E6C9; line-height: 1.6;">MangroVision uses <strong>detectree2 AI</strong> (Mask R-CNN) for accurate tree crown detection and safe planting zone identification.</span>
            <br><br>
            <strong style="color: #7EC88D;">Technology:</strong><br>
            <span style="color: #C8E6C9;">🤖 Detectree2 - AI tree detection<br>
            � Specialized for tree crowns<br>
            🎯 State-of-the-art accuracy</span>
            <br><br>
            <strong style="color: #7EC88D;">Color Legend:</strong><br>
            <span style="color: #C8E6C9;">🔴 Red = Danger Zones (1m buffer)<br>
            🟢 Green = Planting Zones (1m hexagons)</span>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("---")
        st.markdown("<p style='color: #7EC88D; font-weight: 600;'>👥 Thesis Project 2026</p>", unsafe_allow_html=True)
        st.markdown("<p style='color: #C8E6C9;'>GIS-Based Analysis for Leganes</p>", unsafe_allow_html=True)
    
    # Main content area
    col1, col2 = st.columns([1, 1])
    
    with col1:
        st.markdown("### 📤 Upload Drone Image")
        
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png"],
            help="Upload a drone image of the mangrove area"
        )
        
        if uploaded_file is not None:
            # Display uploaded image
            image = Image.open(uploaded_file)
            st.markdown('<div class="image-container">', unsafe_allow_html=True)
            st.image(image, caption="📸 Original Drone Image", width='stretch')
            st.markdown('</div>', unsafe_allow_html=True)
            
            # Show image info
            st.info(f"📐 Image Size: {image.size[0]} × {image.size[1]} pixels")
    
    with col2:
        if uploaded_file is not None:
            st.markdown("### 🚀 Ready to Analyze")
            
            st.markdown("""
            <div class="success-box">
                <span style="color: #2E7D32; font-size: 1.05rem;">
                <strong>✅ Image loaded successfully!</strong><br>
                Click the button below to start detection.
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            # Analyze button
            if st.button("🔍 Detect Canopies & Generate Planting Zones", type="primary"):
                # Store that analysis was requested
                st.session_state.run_analysis = True
                st.session_state.current_file = uploaded_file
                st.session_state.current_altitude = altitude
                st.session_state.current_drone_model = drone_model
                st.session_state.current_canopy_buffer = canopy_buffer
                st.session_state.current_hexagon_size = hexagon_size
                st.session_state.current_ai_confidence = ai_confidence
                st.session_state.current_model_name = model_name
                st.session_state.current_detection_mode = detection_mode
        else:
            st.markdown("### 📋 Instructions")
            st.markdown("""
            <div class="info-box">
                <strong>How to use MangroVision:</strong><br><br>
                <span style="color: #1565C0; font-size: 1.05rem; line-height: 1.8;">
                1️⃣ Upload a drone image (left panel)<br>
                2️⃣ Adjust detection settings (sidebar)<br>
                3️⃣ Click 'Detect & Analyze' button<br>
                4️⃣ View results and download shapefiles<br>
                5️⃣ Import into QGIS for final mapping
                </span>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("### 🎯 Features")
            st.markdown("""
            <div style="color: #white; font-size: 1.05rem; line-height: 2;">
            • 🌳 <strong>AI Canopy Detection</strong> - Identifies mangrove trees<br>
            • 🔴 <strong>Danger Zone Mapping</strong> - 1m safety buffers<br>
            • 🟢 <strong>Planting Zone Generation</strong> - Hexagonal planting areas<br>
            • 📊 <strong>Statistical Analysis</strong> - Area calculations & metrics<br>
            • 🗺️ <strong>QGIS Export</strong> - Shapefile generation for GIS
            </div>
            """, unsafe_allow_html=True)

    # Run analysis if requested
    if uploaded_file is not None and st.session_state.get('run_analysis', False):
        analyze_image(
            st.session_state.current_file,
            st.session_state.current_altitude,
            st.session_state.current_drone_model,
            st.session_state.current_canopy_buffer,
            st.session_state.current_hexagon_size,
            st.session_state.current_ai_confidence,
            st.session_state.current_model_name,
            st.session_state.current_detection_mode
        )


def analyze_image(uploaded_file, altitude, drone_model, canopy_buffer, hexagon_size, ai_confidence, model_name, detection_mode='hybrid'):
    """Process the uploaded image using detectree2 AI detection"""
    
    with st.spinner("🔄 Analyzing image... This may take a moment..."):
        # Save uploaded file temporarily
        temp_path = Path("temp_upload.jpg")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # STEP 1: Extract EXIF metadata (GPS, altitude, drone model)
            st.markdown("---")
            st.markdown("### 📡 Extracting Image Metadata")
            
            metadata = ExifExtractor.extract_all_metadata(str(temp_path))
            
            # Orthophoto map bounds (UTM Zone 51N)
            bounds_utm = {
                'north': 1191825.078,
                'south': 1191650.359,
                'east': 459097.2828,
                'west': 458969.3065
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
                
                # AUTOMATIC HEADING DETECTION
                st.markdown("---")
                st.markdown("### 🧭 Camera Heading Detection")
                
                camera_heading = 0  # Default
                heading_source = "Default (North)"
                
                # Method 1: Flight Log/SRT File Upload (if available)
                st.markdown("#### 📁 Method 1: Flight Log (Auto)")
                with st.expander("ℹ️ Upload SRT file if available"):
                    st.markdown("""
                    **If you have DJI .SRT files:**
                    
                    These files contain GPS trajectory data that can be used to estimate heading.
                    However, for **hovering drones** (taking nadir photos), this may not be accurate.
                    
                    Upload your `.SRT` file to try automatic extraction.
                    """)
                
                flight_log = st.file_uploader(
                    "Upload DJI .SRT or flight log file (optional)",
                    type=['srt', 'txt', 'log', 'csv'],
                    help="Only works if drone was moving during capture",
                    key="flight_log_upload"
                )
                
                if flight_log is not None:
                    temp_path = Path("temp_flight_log") / flight_log.name
                    temp_path.parent.mkdir(exist_ok=True)
                    with open(temp_path, 'wb') as f:
                        f.write(flight_log.getvalue())
                    
                    parser = FlightLogParser()
                    extracted_heading = parser.extract_heading(str(temp_path))
                    
                    if extracted_heading is not None:
                        camera_heading = extracted_heading
                        heading_source = f"Flight Log ({flight_log.name})"
                        st.success(f"✅ Heading extracted: {camera_heading:.1f}° from {flight_log.name}")
                    else:
                        st.warning("⚠️ Could not extract heading (drone may have been hovering). Use landmark method after analysis.")
                
                # Method 2: Landmark-Based Alignment (RECOMMENDED)
                st.markdown("#### 🎯 Method 2: Landmark Alignment (RECOMMENDED)")
                st.info("""
                **Most Accurate Method:**
                1. First, run analysis with current heading
                2. After seeing Visual Results and map, identify a clear landmark (tower, building, path junction)
                3. Click the landmark in both views to auto-calculate rotation
                4. System will recalculate GPS coordinates with correct heading
                
                ⚡ This will be available after running the analysis below.
                """)
                
                # Store landmark alignment flag in session state
                if 'use_landmark_alignment' not in st.session_state:
                    st.session_state.use_landmark_alignment = False
                
                # Method 3: Manual Fine-tune (fallback)
                st.markdown("#### ⚙️ Method 3: Manual Adjustment (Fallback)")
                manual_override = st.checkbox("Use manual heading adjustment", value=False, key="manual_heading_override")
                
                if manual_override:
                    camera_heading = st.slider(
                        "Manual heading:",
                        min_value=0, max_value=359, value=int(camera_heading), step=1,
                        help="0°=North, 90°=East, 180°=South, 270°=West"
                    )
                    heading_source = "Manual Override"
                
                st.info(f"🧭 Using heading: **{camera_heading:.1f}°** ({heading_source})")
            else:
                st.error("❌ No GPS data found in image!")
                st.warning("Cannot geotag results on map without GPS coordinates.")
                st.info("Using manual altitude and drone model from sidebar.")
                camera_heading = 0  # Default if no GPS
            
            # STEP 2: Run detection with detected or manual parameters
            st.markdown("---")
            st.markdown("### 🔍 Running Detection Analysis")
            
            # Create a unique key for this analysis
            analysis_key = f"{uploaded_file.name}_{altitude_to_use}_{drone_to_use}_{canopy_buffer}_{hexagon_size}_{ai_confidence}_{model_name}_{detection_mode}"
            
            # Check if we've already run detection for this configuration
            if 'last_analysis_key' not in st.session_state or st.session_state.last_analysis_key != analysis_key:
                # Initialize detector with Smart Hybrid mode
                detector = HexagonDetector(
                    altitude_m=altitude_to_use, 
                    drone_model=drone_to_use,
                    ai_confidence=ai_confidence,
                    model_name=model_name,
                    detection_mode=detection_mode
                )
                
                # Process image
                results = detector.process_image(
                    image_path=str(temp_path),
                    canopy_buffer_m=canopy_buffer,
                    hexagon_size_m=hexagon_size
                )
                
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
                    
                    # Filter: keep only hexagons outside forbidden zones
                    _safe = []
                    _filtered = 0
                    for _hex in results['hexagons']:
                        _px, _py = _hex['center']
                        _lat, _lon = _px_to_gps(_px, _py)
                        _hex['_gps_lat'] = _lat
                        _hex['_gps_lon'] = _lon
                        if _forbidden_filter.is_safe_location(_lat, _lon):
                            _safe.append(_hex)
                        else:
                            _filtered += 1
                    
                    results['hexagons'] = _safe
                    results['hexagon_count'] = len(_safe)
                    results['_forbidden_filtered'] = _filtered
                
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
            
            img_col1, img_col2 = st.columns(2)
            
            with img_col1:
                st.markdown("**Original Image**")
                original_rgb = cv2.cvtColor(results['image'], cv2.COLOR_BGR2RGB)
                st.image(original_rgb, width='stretch')
            
            with img_col2:
                st.markdown("**Detected Zones**")
                st.markdown("""
                🟣 **Purple** = Canopy Areas | 🔴 **Red** = 1m Danger Buffer  
                🟢 **Light Green** = 1m Planting Buffer | 🟠 **Orange** = Overlap Warning
                🟩 **Dark Green** = Planting Points
                """)
                st.image(vis_image_rgb, width='stretch')
            
            # Helper for finding pixel coordinates
            with st.expander("📏 How to find pixel coordinates for landmark alignment"):
                img_h, img_w = results['image'].shape[:2]
                st.markdown(f"""
                **To use landmark-based heading calibration (see after map):**
                
                1. **Identify a clear landmark** visible in both image and map (tower, building, path)
                2. **Estimate pixel coordinates** in detected zones image:
                   - Image dimensions: {img_w} × {img_h} pixels
                   - Center point: ({img_w//2}, {img_h//2})
                   - Upper-left quarter: X ≈ {img_w//4}, Y ≈ {img_h//4}
                   - Upper-right quarter: X ≈ {img_w*3//4}, Y ≈ {img_h//4}
                   - Adjust based on visual position
                
                3. **For exact coordinates:** Download image, open in image viewer, hover over landmark
                4. **Find same landmark on map** (scroll down to map below)
                5. **Use Landmark Alignment tool** (after map) to auto-calculate heading
                
                💡 This eliminates manual guessing and provides accurate GPS transformation!
                """)
            
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
                    <p style='color: white; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>Warning zone</p>
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
                st.info("🗺️ Shapefile export for QGIS coming soon!")
            
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
                hexagons = results['hexagons']  # Already filtered by forbidden zones
                width, height = results['image_size']
                forbidden_filtered_count = results.get('_forbidden_filtered', 0)

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
                
                # ── Map Display Options ──────────────────────────────────
                show_forbidden_zones = st.checkbox(
                    "🚫 Show forbidden zone boundaries (red polygons)",
                    value=False,
                    help="Toggle visibility of forbidden zones on the map. Filtering is always active."
                )
                
                # Create orthophoto map centered on image location
                ortho_map = folium.Map(
                    location=[map_center_lat, map_center_lon],
                    zoom_start=20,
                    tiles=None,
                    control_scale=True,
                    max_bounds=True
                )
                
                # Add orthophoto tile layer (YOUR CUSTOM MAP)
                folium.TileLayer(
                    tiles="http://localhost:8080/{z}/{x}/{y}.jpg",
                    attr='MangroVision Orthophoto | WebODM',
                    name='Orthophoto',  
                    
                    overlay=False,
                    control=True,
                    max_zoom=22,
                    min_zoom=15,
                    show=True
                ).add_to(ortho_map)
                
                # Add OpenStreetMap reference layer (backup)
                folium.TileLayer(
                    tiles='https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png',
                    attr='© OpenStreetMap contributors',
                    name='OpenStreetMap',
                    overlay=False,
                    control=True,
                    opacity=1.0,
                    show=False
                ).add_to(ortho_map)
                
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
                
                # ── Hexagons are already filtered by forbidden zones ────
                # (filtering was done before visualization so Visual Results
                #  image also excludes forbidden zone hexagons)
                safe_hexagons = hexagons  # Already safe — filtered earlier

                if forbidden_filtered_count > 0:
                    st.warning(f"🚫 {forbidden_filtered_count} planting points filtered out (inside forbidden zones: towers, bridges, houses). {len(safe_hexagons)} safe points remain.")

                # ── Draw forbidden zone polygons on the map (red) ─────────
                if show_forbidden_zones and _forbidden_filter.forbidden_polygons:
                    fz_group = folium.FeatureGroup(name='🚫 Forbidden Zones')
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

                # Ensure hexagons have GPS coords (compute if not pre-computed)
                for hexagon in safe_hexagons:
                    if '_gps_lat' not in hexagon:
                        px, py = hexagon['center']
                        lat, lon = pixel_to_latlon(px, py)
                        hexagon['_gps_lat'] = lat
                        hexagon['_gps_lon'] = lon

                # Add GREEN POINTS only for safe planting hexagons
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
                    ).add_to(ortho_map)
                
                # Add layer control
                folium.LayerControl().add_to(ortho_map)
                
                # Display map
                st_folium.st_folium(ortho_map, width=1400, height=600, key="geo_map", returned_objects=[])
                
                # LANDMARK ALIGNMENT INTERFACE (optional precision override)
                st.markdown("---")
                st.markdown("### 🎯 Precision Override (Optional)")

                with st.expander("📍 Use this only if auto-alignment failed or markers are still off"):
                    st.markdown("""
                    **When to use this:**
                    - Auto-alignment shows a warning above (image outside orthophoto, or too few matches)
                    - Green dots are clearly shifted on the map

                    **Steps:**
                    1. Find a clear landmark visible in both Visual Results and the map (e.g. water tower corner)
                    2. Note its pixel coordinates from the Visual Results image
                    3. Right-click the same landmark on the map to get its GPS coordinates
                    4. Enter both below and click **Calculate Heading**
                    5. Copy the result into the **Manual Adjustment** slider above and re-run
                    """)

                    meters_per_degree_lat = 111320.0
                    meters_per_degree_lon = 111320.0 * np.cos(np.radians(map_center_lat))

                    st.warning("⚠️ Choose a landmark near the image edges for best accuracy")

                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("##### 📸 Landmark in Visual Results")
                        st.caption(f"Image size: {width}×{height} pixels")
                        landmark_px_x = st.number_input("Pixel X:", min_value=0, max_value=width, value=width//2, key="landmark_px_x")
                        landmark_px_y = st.number_input("Pixel Y:", min_value=0, max_value=height, value=height//2, key="landmark_px_y")

                    with col2:
                        st.markdown("##### 🗺️ Same Landmark on Map")
                        st.caption("Right-click the landmark on the map to read its GPS")
                        landmark_lat = st.number_input("Latitude:",  min_value=-90.0,  max_value=90.0,  value=map_center_lat, format="%.7f", key="landmark_lat")
                        landmark_lon = st.number_input("Longitude:", min_value=-180.0, max_value=180.0, value=map_center_lon, format="%.7f", key="landmark_lon")

                    if st.button("🧮 Calculate Heading from Landmark", type="primary"):
                        matcher = ReferencePointMatcher()
                        matcher.add_point_pair((landmark_px_x, landmark_px_y), (landmark_lat, landmark_lon))
                        calculated_heading = matcher.calculate_rotation(
                            image_center_px=(width / 2, height / 2),
                            map_center_latlon=(map_center_lat, map_center_lon),
                            gsd=gsd,
                            meters_per_degree_lat=meters_per_degree_lat,
                            meters_per_degree_lon=meters_per_degree_lon,
                        )
                        if calculated_heading is not None:
                            st.success(f"✅ Calculated heading: **{calculated_heading:.1f}°** — enter this in the Manual Adjustment slider above and re-run.")
                        else:
                            st.error("❌ Could not calculate heading. Please check your coordinates.")
                
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
                    
                    # Download CSV button
                    csv = df.to_csv(index=False)
                    st.download_button(
                        label="📥 Download Coordinate List (CSV)",
                        data=csv,
                        file_name=f"planting_coordinates_{uploaded_file.name}.csv",
                        mime="text/csv"
                    )
                
                if forbidden_filtered_count > 0:
                    st.success(f"✅ Analysis complete! {len(safe_hexagons)} safe planting locations shown on map ({forbidden_filtered_count} filtered out from forbidden zones). Use Visual Results to see exact positions.")
                else:
                    st.success(f"✅ Analysis complete! {len(safe_hexagons)} GPS-tagged planting locations shown on map. Use Visual Results to see exact positions.")
            else:
                st.warning("⚠️ GPS mapping skipped - no GPS data in image")
                st.success("✅ Analysis complete! Results ready for export.")
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.exception(e)
        
        finally:
            # Cleanup temporary files
            if temp_path.exists():
                temp_path.unlink()


if __name__ == "__main__":
    main()
