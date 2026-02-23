"""
MangroVision - Map-Based Plantable Area Detection Backend
FastAPI backend for processing uploaded images and returning geographic coordinates
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import List, Tuple, Optional, Dict
import cv2
import numpy as np
from pathlib import Path
import json
from datetime import datetime
import sys

# Import your existing detector
sys.path.append(str(Path(__file__).parent / "canopy_detection"))
from canopy_detection.canopy_detector_hexagon import HexagonDetector

# GIS libraries for coordinate transformation
from pyproj import Transformer
import math

app = FastAPI(title="MangroVision Map API")

# Enable CORS for frontend access
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class DetectionResponse(BaseModel):
    """Response model for detection results"""
    success: bool
    plantable_areas_found: int
    markers: List[Dict]
    image_analyzed: str
    processing_time_ms: float
    error: Optional[str] = None


class OrthophotoMetadata:
    """
    Store your orthophoto's geographic metadata
    This should match your WebODM output
    """
    def __init__(self):
        # ✅ ACTUAL COORDINATES from WebODM output (UTM Zone 51N)
        # These are in UTM meters, will be converted to lat/lon automatically
        self.bounds_utm = {
            'north': 1191825.078,    # Top (northing)
            'south': 1191650.359,    # Bottom (northing)
            'east': 459097.2828,     # Right (easting)
            'west': 458969.3065      # Left (easting)
        }
        
        # Convert UTM to Lat/Lon for web map
        from pyproj import Transformer
        transformer = Transformer.from_crs("EPSG:32651", "EPSG:4326", always_xy=True)
        
        # Convert corners to lat/lon
        sw_lon, sw_lat = transformer.transform(self.bounds_utm['west'], self.bounds_utm['south'])
        ne_lon, ne_lat = transformer.transform(self.bounds_utm['east'], self.bounds_utm['north'])
        
        self.bounds = {
            'north': ne_lat,
            'south': sw_lat,
            'east': ne_lon,
            'west': sw_lon
        }
        
        # Orthophoto dimensions (approximate from tiles)
        self.width_px = 8192   # From tile structure
        self.height_px = 6144  # Estimated
        
        # Coordinate Reference System - UTM Zone 51N (Philippines)
        self.crs_from = "EPSG:32651"  # UTM Zone 51N
        self.crs_to = "EPSG:4326"     # WGS84 Lat/Lon (for web maps)
        
        # Ground Sample Distance (calculated from bounds)
        width_m = self.bounds_utm['east'] - self.bounds_utm['west']
        self.gsd_cm = (width_m * 100) / self.width_px  # Convert to cm/pixel
        
    def get_bounds_dict(self):
        """Return bounds for Leaflet map"""
        return {
            'southwest': [self.bounds['south'], self.bounds['west']],
            'northeast': [self.bounds['north'], self.bounds['east']]
        }


# Global metadata instance
ORTHO_METADATA = OrthophotoMetadata()


class GeoTransformer:
    """
    Transform pixel coordinates to geographic coordinates
    """
    
    @staticmethod
    def pixel_to_latlon(pixel_x: int, pixel_y: int, 
                        image_width: int, image_height: int,
                        metadata: OrthophotoMetadata) -> Tuple[float, float]:
        """
        Convert pixel coordinates in uploaded image to lat/lon
        
        Key Concept: Map the pixel position in the image to the geographic
        coordinates of the orthophoto bounds.
        
        Args:
            pixel_x: X coordinate in image (0 to image_width)
            pixel_y: Y coordinate in image (0 to image_height)
            image_width: Uploaded image width in pixels
            image_height: Uploaded image height in pixels
            metadata: Orthophoto metadata with bounds
            
        Returns:
            (latitude, longitude) tuple
        """
        
        # Calculate the position as a fraction of the image
        x_fraction = pixel_x / image_width
        y_fraction = pixel_y / image_height
        
        # Map to geographic coordinates
        # X maps to longitude (west to east)
        longitude = metadata.bounds['west'] + (
            x_fraction * (metadata.bounds['east'] - metadata.bounds['west'])
        )
        
        # Y maps to latitude (north to south) - NOTE: Y increases downward in images
        latitude = metadata.bounds['north'] - (
            y_fraction * (metadata.bounds['north'] - metadata.bounds['south'])
        )
        
        return latitude, longitude
    
    @staticmethod
    def get_center_coordinates(bbox_pixels: Tuple[int, int, int, int],
                              image_width: int, image_height: int,
                              metadata: OrthophotoMetadata) -> Tuple[float, float]:
        """
        Get center coordinates of a bounding box
        
        Args:
            bbox_pixels: (x, y, width, height) in pixels
            image_width: Total image width
            image_height: Total image height
            metadata: Orthophoto metadata
            
        Returns:
            (latitude, longitude) of the center point
        """
        x, y, w, h = bbox_pixels
        center_x = x + w / 2
        center_y = y + h / 2
        
        return GeoTransformer.pixel_to_latlon(
            center_x, center_y, image_width, image_height, metadata
        )


@app.get("/")
def read_root():
    """Health check endpoint"""
    return {
        "status": "online",
        "service": "MangroVision Map API",
        "version": "1.0.0"
    }


@app.get("/api/map/metadata")
def get_map_metadata():
    """
    Return orthophoto metadata for map initialization
    Frontend uses this to set map bounds and tile layer
    """
    return {
        "bounds": ORTHO_METADATA.get_bounds_dict(),
        "center": [
            (ORTHO_METADATA.bounds['north'] + ORTHO_METADATA.bounds['south']) / 2,
            (ORTHO_METADATA.bounds['east'] + ORTHO_METADATA.bounds['west']) / 2
        ],
        "tile_url": "/tiles/{z}/{x}/{y}.png",  # Update with your tile server URL
        "max_zoom": 20,
        "min_zoom": 15,
        "gsd_cm": ORTHO_METADATA.gsd_cm
    }


@app.post("/api/detect-plantable-area", response_model=DetectionResponse)
async def detect_plantable_area(
    image: UploadFile = File(...),
    drone_altitude: float = 6.0,
    drone_model: str = "GENERIC_4K"
):
    """
    Main endpoint: Upload image, detect plantable areas, return coordinates
    
    Process:
    1. Receive uploaded image
    2. Run canopy detection
    3. Identify plantable zones (hexagons)
    4. Convert pixel coordinates to lat/lon
    5. Return markers for the map
    """
    start_time = datetime.now()
    
    try:
        # 1. Read uploaded image
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image file")
        
        height, width = img.shape[:2]
        
        # 2. Initialize detector with your existing code
        detector = HexagonDetector(altitude_m=drone_altitude, drone_model=drone_model)
        detector.calculate_gsd(width, height)
        
        # 3. Detect canopies and generate plantable zones
        canopy_polygons, canopy_mask = detector.detect_canopies(img)
        
        # Create danger zones (buffers around canopies)
        danger_zones = detector.create_danger_zones(canopy_polygons, buffer_m=1.0)
        
        # Generate hexagonal planting zones
        hexagon_centers, hexagon_polygons = detector.generate_hexagon_grid(
            image_shape=(height, width),
            hexagon_spacing_m=0.5
        )
        
        # Filter plantable hexagons (not in danger zones)
        plantable_hexagons = detector.filter_plantable_hexagons(
            hexagon_polygons, 
            danger_zones
        )
        
        # 4. Convert plantable hexagons to geographic coordinates
        markers = []
        
        for idx, hexagon in enumerate(plantable_hexagons):
            # Get centroid of hexagon in pixels
            centroid = hexagon.centroid
            pixel_x = centroid.x
            pixel_y = centroid.y
            
            # Convert to lat/lon
            lat, lon = GeoTransformer.pixel_to_latlon(
                pixel_x, pixel_y, width, height, ORTHO_METADATA
            )
            
            markers.append({
                "id": idx,
                "latitude": lat,
                "longitude": lon,
                "area_m2": hexagon.area * (detector.gsd ** 2),  # Convert to real area
                "confidence": 1.0,
                "type": "plantable_zone"
            })
        
        # 5. Also add detected canopy markers for reference
        canopy_markers = []
        for idx, canopy in enumerate(canopy_polygons):
            centroid = canopy.centroid
            lat, lon = GeoTransformer.pixel_to_latlon(
                centroid.x, centroid.y, width, height, ORTHO_METADATA
            )
            canopy_markers.append({
                "id": f"canopy_{idx}",
                "latitude": lat,
                "longitude": lon,
                "type": "canopy_danger",
                "area_m2": canopy.area * (detector.gsd ** 2)
            })
        
        # Calculate processing time
        processing_time = (datetime.now() - start_time).total_seconds() * 1000
        
        # 6. Return response
        return DetectionResponse(
            success=True,
            plantable_areas_found=len(plantable_hexagons),
            markers=markers + canopy_markers,  # Combined markers
            image_analyzed=image.filename,
            processing_time_ms=processing_time
        )
        
    except Exception as e:
        return DetectionResponse(
            success=False,
            plantable_areas_found=0,
            markers=[],
            image_analyzed=image.filename if image else "unknown",
            processing_time_ms=0,
            error=str(e)
        )


@app.post("/api/detect-simple")
async def detect_simple(image: UploadFile = File(...)):
    """
    Simplified detection endpoint for testing
    Returns a single center point marker
    """
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        
        height, width = img.shape[:2]
        
        # Just return center point for testing
        center_lat, center_lon = GeoTransformer.pixel_to_latlon(
            width // 2, height // 2, width, height, ORTHO_METADATA
        )
        
        return {
            "success": True,
            "markers": [{
                "id": 0,
                "latitude": center_lat,
                "longitude": center_lon,
                "type": "test_marker",
                "message": "Center of uploaded image"
            }]
        }
    except Exception as e:
        return {"success": False, "error": str(e)}


if __name__ == "__main__":
    import uvicorn
    print("🌿 Starting MangroVision Map API...")
    print("📍 Configure your orthophoto bounds in OrthophotoMetadata class!")
    uvicorn.run(app, host="0.0.0.0", port=8000)
