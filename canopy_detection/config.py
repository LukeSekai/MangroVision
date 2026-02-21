"""
Configuration file for Mangrove Canopy Detection System
For Leganes Mangrove Planting Zone Analysis
"""

import os

# Base paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DRONE_IMAGES_DIR = os.path.join(BASE_DIR, "drone_images")
OUTPUT_DIR = os.path.join(BASE_DIR, "output")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create directories if they don't exist
os.makedirs(DRONE_IMAGES_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# Detection parameters
BUFFER_DISTANCE_METERS = 1.0  # 1 meter buffer around canopy danger zones
CONFIDENCE_THRESHOLD = 0.5    # Minimum confidence for tree detection

# Drone and Camera Settings (IMPORTANT: Configure for accurate measurements!)
# ============================================================================
FLIGHT_ALTITUDE_METERS = 6.0  # Fixed drone altitude: 6 meters, 90° angle

# Option 1: Use known drone model
DRONE_MODEL = 'GENERIC_4K'    # Options: DJI_MINI_3, DJI_MAVIC_3, DJI_AIR_2S, 
                              #          DJI_PHANTOM_4, GENERIC_4K

# Option 2: Manual camera specs (if known, overrides drone model)
USE_MANUAL_CAMERA_SPECS = False
MANUAL_SENSOR_WIDTH_MM = 6.3
MANUAL_FOCAL_LENGTH_MM = 4.5

# Ground Sample Distance (calculated automatically, or set manually)
# Run gsd_calculator.py to calculate this for your setup!
GSD_METERS_PER_PIXEL = None   # Set to None for auto-calculation
                              # Or set manually: e.g., 0.0092 for 7m altitude

# Image processing
IMAGE_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.tif', '.tiff']

# Model settings (will use pre-trained detectree2 model initially)
MODEL_TYPE = "mask_rcnn"      # Mask R-CNN for instance segmentation
DEVICE = "cpu"                # Change to "cuda" if GPU available

# GIS settings (for future integration)
DEFAULT_CRS = "EPSG:32651"    # UTM Zone 51N (Philippines)

# Output settings
SAVE_VISUALIZATIONS = True
SAVE_SHAPEFILES = True
SAVE_JSON = True

# Color coding for visualization
DANGER_ZONE_COLOR = (255, 0, 0)      # Red for canopy danger zones
BUFFER_ZONE_COLOR = (255, 165, 0)    # Orange for buffer zones
SAFE_ZONE_COLOR = (0, 255, 0)        # Green for safe planting zones
