"""
MangroVision - AI-Powered Mangrove Planting Zone Analyzer
Beautiful Streamlit UI for the thesis project
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import io
import json

# Add canopy_detection to path
sys.path.append(str(Path(__file__).parent / "canopy_detection"))

from canopy_detection.canopy_detector_hexagon import HexagonDetector

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


def main():
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>🌿 MangroVision</h1>
        <p>AI-Powered Mangrove Planting Zone Analyzer for Leganes, Iloilo</p>
    </div>
    """, unsafe_allow_html=True)
    
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
            max_value=1.0,
            value=0.5,
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
            <span style="color: #C8E6C9; line-height: 1.6;">MangroVision uses AI to detect mangrove canopies and identify safe planting zones.</span>
            <br><br>
            <strong style="color: #7EC88D;">Color Legend:</strong><br>
            <span style="color: #C8E6C9;">🔴 Red = Danger Zones (1m buffer)<br>
            🟢 Green = Planting Zones (0.5m hexagons)</span>
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
                analyze_image(uploaded_file, altitude, drone_model, canopy_buffer, hexagon_size)
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


def analyze_image(uploaded_file, altitude, drone_model, canopy_buffer, hexagon_size):
    """Process the uploaded image and display results"""
    
    with st.spinner("🔄 Analyzing image... This may take a moment..."):
        # Save uploaded file temporarily
        temp_path = Path("temp_upload.jpg")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        try:
            # Initialize detector
            detector = HexagonDetector(altitude_m=altitude, drone_model=drone_model)
            
            # Process image
            results = detector.process_image(
                image_path=str(temp_path),
                canopy_buffer_m=canopy_buffer,
                hexagon_size_m=hexagon_size
            )
            
            # Create visualization
            vis_image = detector.visualize_results(results)
            
            # Convert BGR to RGB for display
            vis_image_rgb = cv2.cvtColor(vis_image, cv2.COLOR_BGR2RGB)
            
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
                🟢 **Light Green** = 0.5m Planting Buffer | � **Orange** = Overlap Warning  
                �🟩 **Dark Green** = Planting Points
                """)
                st.image(vis_image_rgb, width='stretch')
            
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
                    <p style='color: #2D5F3F; font-size: 0.9rem; margin: 0.5rem 0 0 0;'>0.5m safe zones</p>
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
            
            st.success("✅ Analysis complete! Results are ready for QGIS integration.")
            
        except Exception as e:
            st.error(f"❌ Error processing image: {str(e)}")
            st.exception(e)
        
        finally:
            # Cleanup
            if temp_path.exists():
                temp_path.unlink()


if __name__ == "__main__":
    main()
