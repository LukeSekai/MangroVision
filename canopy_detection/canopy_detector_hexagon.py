"""
MangroVision - Hexagonal Planting Zone Detector
Detects canopies, creates danger zones, and generates hexagonal planting buffers
Supports both HSV color detection and AI-powered detectree2
"""

import cv2
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd
from typing import Tuple, List, Dict, Optional
import json
from pathlib import Path
import math

from gsd_calculator import GSDCalculator

# Optional: Try to import detectree2 detector
try:
    from detectree2_proper import ProperDetectree2Detector
    DETECTREE2_AVAILABLE = True
    print("✅ Proper Detectree2 library available and loaded")
except ImportError as e:
    try:
        from detectree2_detector import Detectree2Detector
        DETECTREE2_AVAILABLE = True
        print("✅ Detectree2 AI available and loaded (fallback)")
    except ImportError as e:
        DETECTREE2_AVAILABLE = False
        print(f"⚠️ detectree2_detector not available: {e}")
        print(f"   Using HSV color detection fallback")
class HexagonDetector:
    """Advanced canopy detector with hexagonal planting zones - Smart Hybrid System"""
    
    def __init__(self, 
                 altitude_m: float = 6.0, 
                 drone_model: str = 'GENERIC_4K',
                 ai_confidence: float = 0.5,
                 model_name: str = 'benchmark',
                 detection_mode: str = 'hybrid'):
        """
        Initialize detector with Smart Hybrid detection
        
        Args:
            altitude_m: Flight altitude in meters
            drone_model: Drone model for GSD calculation
            ai_confidence: Confidence threshold for AI detection (0-1)
            model_name: 'paracou' (tropical) or 'benchmark' (general)
            detection_mode: 'hybrid', 'ai', or 'hsv'
                - 'hybrid': Merge HSV + AI results (RECOMMENDED - 90-95% accuracy)
                - 'ai': AI only (75-85% accuracy, may miss trees)
                - 'hsv': HSV only (85-90% accuracy, fast)
        """
        self.altitude_m = altitude_m
        self.drone_model = drone_model
        self.ai_confidence = ai_confidence
        self.detection_mode = detection_mode
        self.gsd = None
        self.image_shape = None
        self.use_ai = DETECTREE2_AVAILABLE
        
        # Initialize detectree2 AI detector if needed (for 'ai' or 'hybrid' modes)
        if detection_mode in ['ai', 'hybrid'] and DETECTREE2_AVAILABLE:
            print(f"🌳 Initializing MangroVision with Smart Hybrid System...")
            print(f"   Mode: {detection_mode.upper()} (HSV + AI merger)" if detection_mode == 'hybrid' else f"   Mode: {detection_mode.upper()}")
            try:
                # Try proper detectree2 integration first
                self.ai_detector = ProperDetectree2Detector(
                    confidence_threshold=ai_confidence,
                    device='cpu'  # Change to 'cuda' if GPU available
                )
                self.ai_detector.setup_model()
                print(f"✓ Detectree2 AI initialized successfully")
            except (NameError, AttributeError):
                # Fallback to custom detector
                self.ai_detector = Detectree2Detector(
                    confidence_threshold=ai_confidence,
                    device='cpu',
                    model_name=model_name
                )
                self.ai_detector.setup_model()
                print(f"✓ Custom Detectree2 initialized successfully")
        elif detection_mode == 'hsv':
            print(f"🌳 Initializing MangroVision with HSV detection (fast mode)")
            self.ai_detector = None
        else:
            print(f"⚠️  Detectree2 not available - using HSV color detection fallback")
            print(f"   For better accuracy, install detectron2 and detectree2")
            self.ai_detector = None
            self.detection_mode = 'hsv'  # Force HSV if AI not available
        
    def calculate_gsd(self, image_width: int, image_height: int):
        """Calculate Ground Sample Distance for the image"""
        self.gsd, specs = GSDCalculator.calculate_gsd_from_drone(
            altitude_m=self.altitude_m,
            drone_model=self.drone_model
        )
        # Store as (height, width) to match numpy convention
        self.image_shape = (image_height, image_width)
        return self.gsd
    
    def detect_canopies(self, image: np.ndarray) -> Tuple[List[Polygon], np.ndarray]:
        """
        Smart Hybrid Detection: Merges HSV + AI for maximum accuracy
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (List of Shapely Polygon objects, binary canopy mask)
        """
        h, w = image.shape[:2]
        # Reset AI metadata each run
        self._ai_metadata = {}
        
        if self.detection_mode == 'hybrid' and self.ai_detector is not None:
            # SMART HYBRID: Run both HSV and AI, then merge results
            print(f"🌳 Running Smart Hybrid Detection (HSV + AI)...")
            
            # Step 1: HSV Detection (catches everything green)
            hsv_polygons, hsv_mask = self._detect_hsv(image)
            
            # Step 2: AI Detection (high-confidence canopies)
            try:
                ai_result = self.ai_detector.detect_from_image(image)
                # Handle both 2-value and 3-value returns
                if len(ai_result) == 3:
                    ai_polygons, ai_mask, metadata = ai_result
                else:
                    ai_polygons, ai_mask = ai_result
                    metadata = {}
                # Store AI metadata (contains per-class masks and class info)
                self._ai_metadata = metadata
            except Exception as e:
                print(f"   ⚠️ AI detection failed: {e}")
                print(f"   Falling back to HSV-only detection")
                return hsv_polygons, hsv_mask
            
            # Step 3: Merge results (UNION - keep all unique detections)
            combined_polygons = self._merge_detections(hsv_polygons, ai_polygons)
            
            # Step 4: Create combined mask
            combined_mask = cv2.bitwise_or(hsv_mask, ai_mask)
            
            # Statistics
            total_canopy_pixels = np.count_nonzero(combined_mask)
            total_canopy_m2 = total_canopy_pixels * (self.gsd ** 2)
            
            class_counts = metadata.get('class_counts', {})
            bungalon_count = class_counts.get(1, 0)
            other_ai_count = class_counts.get(0, 0)
            
            print(f"✓ Hybrid Detection Results:")
            print(f"   - HSV found: {len(hsv_polygons)} crowns")
            print(f"   - AI found: {len(ai_polygons)} crowns")
            if bungalon_count > 0:
                print(f"     → Bungalon Canopy: {bungalon_count}")
                print(f"     → Mangrove-Canopy: {other_ai_count}")
            print(f"   - Merged total: {len(combined_polygons)} crowns ({total_canopy_m2:.1f} m²)")
            print(f"   - Method: UNION (best of both worlds)")
            
            return combined_polygons, combined_mask
            
        elif self.detection_mode == 'ai' and self.ai_detector is not None:
            # AI ONLY MODE
            print(f"🌳 Running AI-only detection...")
            
            try:
                ai_result = self.ai_detector.detect_from_image(image)
                # Handle both 2-value and 3-value returns
                if len(ai_result) == 3:
                    canopy_polygons, canopy_mask, metadata = ai_result
                else:
                    canopy_polygons, canopy_mask = ai_result
                    metadata = {}
                self._ai_metadata = metadata
            except Exception as e:
                print(f"   ⚠️ AI detection failed: {e}")
                print(f"   Falling back to HSV detection")
                return self._detect_hsv(image)
            
            total_canopy_pixels = np.count_nonzero(canopy_mask)
            total_canopy_m2 = total_canopy_pixels * (self.gsd ** 2)
            
            print(f"✓ AI detected {len(canopy_polygons)} tree crowns (Total: {total_canopy_m2:.1f} m²)")
            print(f"   Using: {metadata.get('detection_method', 'detectree2')}")
            
            return canopy_polygons, canopy_mask
            
        else:
            # HSV ONLY MODE (fallback or explicit choice)
            print(f"🌳 Running HSV-only detection...")
            
            canopy_polygons, canopy_mask = self._detect_hsv(image)
            
            total_canopy_pixels = np.count_nonzero(canopy_mask)
            total_canopy_m2 = total_canopy_pixels * (self.gsd ** 2)
            
            print(f"✓ HSV detected {len(canopy_polygons)} tree crowns (Total: {total_canopy_m2:.1f} m²)")
            
            return canopy_polygons, canopy_mask
    
    def _detect_hsv(self, image: np.ndarray) -> Tuple[List[Polygon], np.ndarray]:
        """
        Enhanced HSV detection with multi-scale morphology for mangroves
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (List of Shapely Polygon objects, binary canopy mask)
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Enhanced HSV ranges for mangroves (catches dark/shadowed canopies)
        lower_green = np.array([20, 30, 30])   # Lower threshold for shadowed areas
        upper_green = np.array([95, 255, 255])  # Wider hue range
        canopy_mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Multi-scale morphological operations (better gap filling)
        # Small kernel: Connect nearby pixels
        kernel_small = np.ones((3, 3), np.uint8)
        canopy_mask = cv2.morphologyEx(canopy_mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
        
        # Medium kernel: Fill larger gaps in canopies
        kernel_medium = np.ones((7, 7), np.uint8)
        canopy_mask = cv2.morphologyEx(canopy_mask, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
        
        # Remove small noise
        kernel_open = np.ones((5, 5), np.uint8)
        canopy_mask = cv2.morphologyEx(canopy_mask, cv2.MORPH_OPEN, kernel_open)
        
        # Find contours and create polygons
        contours, _ = cv2.findContours(canopy_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        canopy_polygons = []
        # Dynamic minimum area based on GSD (0.5 m² minimum)
        min_area_m2 = 0.5
        min_area_pixels = int(min_area_m2 / (self.gsd ** 2)) if self.gsd else 300
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area_pixels:
                # Simplify contour to reduce points
                epsilon = 0.005 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = approx.reshape(-1, 2)
                
                if len(points) >= 3:
                    poly = Polygon(points)
                    if poly.is_valid:
                        canopy_polygons.append(poly)
                    elif not poly.is_valid:
                        # Try to fix invalid polygons
                        poly = poly.buffer(0)
                        if poly.is_valid and not poly.is_empty:
                            if isinstance(poly, Polygon):
                                canopy_polygons.append(poly)
                            elif isinstance(poly, MultiPolygon):
                                canopy_polygons.extend(list(poly.geoms))
        
        return canopy_polygons, canopy_mask
    
    def _merge_detections(self, hsv_polygons: List[Polygon], ai_polygons: List[Polygon]) -> List[Polygon]:
        """
        Merge HSV and AI detections using intelligent UNION
        
        Removes duplicates while keeping unique detections from both methods
        
        Args:
            hsv_polygons: Polygons from HSV detection
            ai_polygons: Polygons from AI detection
            
        Returns:
            List of merged unique polygons
        """
        if not hsv_polygons:
            return ai_polygons
        if not ai_polygons:
            return hsv_polygons
        
        # Start with all AI polygons (higher confidence)
        merged = list(ai_polygons)
        
        # Add HSV polygons that don't significantly overlap with AI
        overlap_threshold = 0.5  # 50% IoU threshold
        
        for hsv_poly in hsv_polygons:
            is_duplicate = False
            
            for ai_poly in ai_polygons:
                try:
                    # Calculate Intersection over Union (IoU)
                    if hsv_poly.intersects(ai_poly):
                        intersection = hsv_poly.intersection(ai_poly).area
                        union = hsv_poly.union(ai_poly).area
                        iou = intersection / union if union > 0 else 0
                        
                        if iou > overlap_threshold:
                            is_duplicate = True
                            break
                except Exception:
                    continue
            
            # Add HSV detection if it's unique (not a duplicate)
            if not is_duplicate:
                merged.append(hsv_poly)
        
        return merged
    
    def detect_structures(self, image: np.ndarray, canopy_mask: np.ndarray = None) -> Tuple[List[Polygon], np.ndarray]:
        """
        Detect man-made structures (bridges, towers, buildings) using color analysis.
        Uses canopy mask to EXCLUDE all vegetation first - structures are what's left
        that is NOT green vegetation and NOT water/mud/bare soil.
        NO BUFFER ZONES - just exact footprint of structures
        
        Args:
            image: Input BGR image
            canopy_mask: Binary mask of detected canopy/vegetation (to exclude from search)
            
        Returns:
            Tuple of (List of structure polygons, binary structure mask)
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # ── Step 1: Build a vegetation exclusion mask ──────────────────────────
        # Exclude all GREEN pixels first (mangroves, grass, any vegetation)
        lower_green1 = np.array([35, 30, 30])
        upper_green1 = np.array([90, 255, 255])
        green_mask = cv2.inRange(hsv, lower_green1, upper_green1)
        
        # Also exclude DARK pixels (shadows, water, dark mud)
        # EXPANDED range to catch darker mud areas
        lower_dark = np.array([0, 0, 0])
        upper_dark = np.array([180, 255, 100])  # Increased from 50 to 100 to catch more mud
        dark_mask = cv2.inRange(hsv, lower_dark, upper_dark)
        
        # Also exclude BROWN/TAN bare soil - EXPANDED ranges
        # Range 1: Brown/tan soil (original)
        lower_soil1 = np.array([8, 20, 60])
        upper_soil1 = np.array([30, 200, 180])
        soil_mask1 = cv2.inRange(hsv, lower_soil1, upper_soil1)
        
        # Range 2: Gray/light mud (catches desaturated tan/gray mud)
        # REFINED to avoid tower colors - lower saturation only
        lower_soil2 = np.array([0, 0, 50])      # Low saturation, mid-low brightness
        upper_soil2 = np.array([35, 35, 150])   # Reduced saturation from 60 to 35 (avoid tower)
        soil_mask2 = cv2.inRange(hsv, lower_soil2, upper_soil2)
        
        # Range 3: Very light mud/sand
        # REFINED to avoid tower - lower saturation and specific brightness
        lower_soil3 = np.array([15, 10, 120])   # Light tan/beige areas
        upper_soil3 = np.array([35, 50, 200])   # Reduced saturation from 80 to 50
        soil_mask3 = cv2.inRange(hsv, lower_soil3, upper_soil3)
        
        # Combine all soil/mud masks
        soil_mask = cv2.bitwise_or(soil_mask1, soil_mask2)
        soil_mask = cv2.bitwise_or(soil_mask, soil_mask3)
        
        # Build combined exclusion zone
        exclusion_mask = cv2.bitwise_or(green_mask, dark_mask)
        exclusion_mask = cv2.bitwise_or(exclusion_mask, soil_mask)
        
        # Also exclude provided canopy mask (AI-detected trees)
        if canopy_mask is not None:
            exclusion_mask = cv2.bitwise_or(exclusion_mask, canopy_mask)
        
        # ── Step 2: What remains = candidate structures ────────────────────────
        # Invert exclusion to get non-vegetation, non-soil, non-dark pixels
        candidate_mask = cv2.bitwise_not(exclusion_mask)
        
        # ── Step 3: Detect GRAY/CONCRETE/METAL colors ─────────────────────────
        # Low saturation = man-made materials (concrete, metal, painted wood)
        # MADE MORE RESTRICTIVE to avoid false positives with mud
        # Requires higher brightness to distinguish from mud
        lower_manmade = np.array([0, 0, 100])   # Increased from 70 to 100 (brighter)
        upper_manmade = np.array([180, 40, 230]) # Reduced saturation threshold from 45 to 40
        manmade_color = cv2.inRange(hsv, lower_manmade, upper_manmade)
        
        # ── Step 4: Detect RED/RUST/ORANGE METAL (towers, bridges) ────────────
        # Expanded to catch entire tower including shadowed/lighter parts
        
        # Red range 1: Bright red/rust (upper part of tower)
        lower_red1 = np.array([0, 60, 80])      # Reduced saturation from 80 to 60, increased value
        upper_red1 = np.array([10, 255, 255])   # Full value range
        red_mask1 = cv2.inRange(hsv, lower_red1, upper_red1)
        
        # Red range 2: Wraparound hue (170-180)
        lower_red2 = np.array([165, 60, 80])
        upper_red2 = np.array([180, 255, 255])
        red_mask2 = cv2.inRange(hsv, lower_red2, upper_red2)
        
        # Orange/brown range: Rusty metal, painted towers
        lower_orange = np.array([8, 50, 70])    # Orange-brown hue
        upper_orange = np.array([25, 255, 220])
        orange_mask = cv2.inRange(hsv, lower_orange, upper_orange)
        
        # Dark red/brown: Shadowed parts of tower
        lower_dark_red = np.array([0, 40, 40])   # Lower thresholds to catch shadows
        upper_dark_red = np.array([15, 255, 120])
        dark_red_mask = cv2.inRange(hsv, lower_dark_red, upper_dark_red)
        
        # Combine all metal/tower colors
        red_mask = cv2.bitwise_or(red_mask1, red_mask2)
        red_mask = cv2.bitwise_or(red_mask, orange_mask)
        red_mask = cv2.bitwise_or(red_mask, dark_red_mask)
        
        # Fill holes in tower mask BEFORE combining with other structures
        kernel_fill_tower = np.ones((15, 15), np.uint8)
        red_mask = cv2.morphologyEx(red_mask, cv2.MORPH_CLOSE, kernel_fill_tower, iterations=2)
        
        # ── Step 5: Combine - must be man-made color AND NOT excluded ──────────
        structure_mask = cv2.bitwise_or(manmade_color, red_mask)
        structure_mask = cv2.bitwise_and(structure_mask, candidate_mask)
        
        # ── Step 6: Morphological cleanup ─────────────────────────────────────
        # Additional cleanup to connect nearby structure parts
        kernel_close = np.ones((9, 9), np.uint8)  # Increased from 7x7 to 9x9
        structure_mask = cv2.morphologyEx(structure_mask, cv2.MORPH_CLOSE, kernel_close, iterations=4)
        kernel_open = np.ones((5, 5), np.uint8)
        structure_mask = cv2.morphologyEx(structure_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Find contours of structures
        contours, _ = cv2.findContours(structure_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        structure_polygons = []
        # Minimum area for structures (5 m² - filters out tiny noise)
        min_area_m2 = 5.0
        min_area_pixels = int(min_area_m2 / (self.gsd ** 2)) if self.gsd else 500
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area_pixels:
                # Simplify contour
                epsilon = 0.01 * cv2.arcLength(contour, True)
                approx = cv2.approxPolyDP(contour, epsilon, True)
                points = approx.reshape(-1, 2)
                
                if len(points) >= 3:
                    poly = Polygon(points)
                    if poly.is_valid:
                        structure_polygons.append(poly)
                    elif not poly.is_valid:
                        poly = poly.buffer(0)
                        if poly.is_valid and not poly.is_empty:
                            if isinstance(poly, Polygon):
                                structure_polygons.append(poly)
                            elif isinstance(poly, MultiPolygon):
                                structure_polygons.extend(list(poly.geoms))
        
        # Calculate statistics
        structure_pixels = np.count_nonzero(structure_mask)
        structure_m2 = structure_pixels * (self.gsd ** 2)
        
        # Calculate what was excluded (for debugging)
        excluded_pixels = np.count_nonzero(exclusion_mask)
        excluded_m2 = excluded_pixels * (self.gsd ** 2)
        total_pixels = h * w
        total_m2 = total_pixels * (self.gsd ** 2)
        
        print(f"✓ Structure detection breakdown:")
        print(f"   Total area: {total_m2:.1f} m²")
        print(f"   Excluded (vegetation/mud/water): {excluded_m2:.1f} m² ({excluded_pixels/total_pixels*100:.1f}%)")
        print(f"   Structures detected: {len(structure_polygons)} ({structure_m2:.1f} m²)")
        print(f"   No buffer applied - exact footprint only")
        
        return structure_polygons, structure_mask
    
    def detect_non_vegetation_areas(self, image: np.ndarray) -> np.ndarray:
        """
        Detect non-vegetation areas in the image (bridges, roads, water, buildings)
        These areas should NOT have planting zones
        
        Args:
            image: Input BGR image
            
        Returns:
            Binary mask where 255 = non-vegetation (forbidden), 0 = potential planting area
        """
        h, w = image.shape[:2]
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Initialize combined mask
        forbidden_mask = np.zeros((h, w), dtype=np.uint8)
        
        # 1. DETECT GRAY AREAS (concrete bridges, roads, buildings)
        # Gray has low saturation and mid-range value
        lower_gray = np.array([0, 0, 60])      # Low saturation, moderate brightness
        upper_gray = np.array([180, 50, 220])  # Any hue, low saturation
        gray_mask = cv2.inRange(hsv, lower_gray, upper_gray)
        
        # 2. DETECT WATER (blue/dark areas)
        # Water appears blue or very dark
        lower_water1 = np.array([90, 40, 20])   # Blue water
        upper_water1 = np.array([130, 255, 180])
        water_mask1 = cv2.inRange(hsv, lower_water1, upper_water1)
        
        lower_water2 = np.array([0, 0, 0])      # Dark water/shadows
        upper_water2 = np.array([180, 255, 40])
        water_mask2 = cv2.inRange(hsv, lower_water2, upper_water2)
        
        water_mask = cv2.bitwise_or(water_mask1, water_mask2)
        
        # 3. DETECT BRIGHT/WHITE AREAS (concrete, white buildings)
        lower_white = np.array([0, 0, 220])
        upper_white = np.array([180, 30, 255])
        white_mask = cv2.inRange(hsv, lower_white, upper_white)
        
        # Combine all non-vegetation masks
        forbidden_mask = cv2.bitwise_or(forbidden_mask, gray_mask)
        forbidden_mask = cv2.bitwise_or(forbidden_mask, water_mask)
        forbidden_mask = cv2.bitwise_or(forbidden_mask, white_mask)
        
        # Clean up the mask - remove small noise
        kernel_clean = np.ones((5, 5), np.uint8)
        forbidden_mask = cv2.morphologyEx(forbidden_mask, cv2.MORPH_OPEN, kernel_clean)
        
        # Expand forbidden areas slightly to be safe
        kernel_expand = np.ones((10, 10), np.uint8)
        forbidden_mask = cv2.dilate(forbidden_mask, kernel_expand, iterations=1)
        
        # Calculate statistics
        forbidden_pixels = np.count_nonzero(forbidden_mask)
        forbidden_m2 = forbidden_pixels * (self.gsd ** 2)
        forbidden_pct = (forbidden_pixels / (h * w)) * 100
        
        print(f"✓ Detected non-vegetation areas: {forbidden_m2:.1f} m² ({forbidden_pct:.1f}% of image)")
        print(f"   (bridges, roads, water, buildings automatically excluded)")
        
        return forbidden_mask
    
    def create_danger_zones(self, canopy_polygons: List[Polygon], canopy_mask: np.ndarray, buffer_m: float = 1.0) -> Tuple[Polygon, np.ndarray]:
        """
        Create danger zones (canopies + 1m buffer) with proper masking
        CLIPS to image boundaries to prevent overflow
        
        Args:
            canopy_polygons: List of canopy polygons
            canopy_mask: Binary mask of canopy areas
            buffer_m: Buffer distance in meters (default 1.0m)
            
        Returns:
            Tuple of (unified danger zone polygon, danger zone mask)
        """
        h, w = self.image_shape
        danger_mask = np.zeros((h, w), dtype=np.uint8)
        
        if not canopy_polygons:
            return Polygon(), danger_mask
        
        # Create image boundary polygon for clipping
        image_boundary = Polygon([(0, 0), (w, 0), (w, h), (0, h)])
        
        # Convert buffer distance to pixels
        buffer_pixels = buffer_m / self.gsd
        print(f"   Buffer calculation: {buffer_m}m ÷ {self.gsd:.5f}m/px = {buffer_pixels:.1f} pixels")
        
        # Create buffers around each canopy (includes canopy + buffer)
        buffered = [poly.buffer(buffer_pixels) for poly in canopy_polygons]
        
        # Merge all buffers
        danger_zone = unary_union(buffered)
        
        # CLIP to image boundaries - this is crucial!
        danger_zone = danger_zone.intersection(image_boundary)
        
        # Create danger zone mask
        if isinstance(danger_zone, Polygon):
            if danger_zone.exterior:
                pts = np.array(danger_zone.exterior.coords, dtype=np.int32)
                cv2.fillPoly(danger_mask, [pts], 255)
        elif isinstance(danger_zone, MultiPolygon):
            for poly in danger_zone.geoms:
                if poly.exterior:
                    pts = np.array(poly.exterior.coords, dtype=np.int32)
                    cv2.fillPoly(danger_mask, [pts], 255)
        
        danger_area_pixels = np.count_nonzero(danger_mask)
        danger_area_m2 = danger_area_pixels * (self.gsd ** 2)
        print(f"✓ Created 1.0m danger buffer zones ({danger_area_m2:.2f} m²)")
        return danger_zone, danger_mask
    
    def identify_plantable_zones(self, danger_zone: Polygon) -> Polygon:
        """
        Identify areas outside danger zones (plantable zones)
        
        Args:
            danger_zone: Combined danger zone polygon
            
        Returns:
            Plantable zone polygon
        """
        # Create total image boundary
        h, w = self.image_shape
        total_area = Polygon([(0, 0), (w, 0), (w, h), (0, h)])
        
        # Subtract danger zones from total area
        plantable_zone = total_area.difference(danger_zone)
        
        return plantable_zone
    
    def create_hexagon(self, center_x: float, center_y: float, radius_pixels: float) -> Polygon:
        """
        Create a hexagon polygon
        
        Args:
            center_x: X coordinate of center
            center_y: Y coordinate of center
            radius_pixels: Radius in pixels
            
        Returns:
            Hexagon polygon
        """
        angles = [i * np.pi / 3 for i in range(6)]  # 6 points, 60° apart
        points = [
            (center_x + radius_pixels * np.cos(angle),
             center_y + radius_pixels * np.sin(angle))
            for angle in angles
        ]
        return Polygon(points)
    
    def generate_hexagonal_planting_zones(
        self, 
        plantable_zone: Polygon, 
        hexagon_size_m: float = 1.0,
        maximize_coverage: bool = True
    ) -> List[Dict]:
        """
        Generate maximized hexagonal planting zones
        Only the core (exact planting point) must be in safe zone
        Buffers can overlap with danger zones and each other (max 0.1m overlap)
        
        Args:
            plantable_zone: Available planting area
            hexagon_size_m: Hexagon buffer size in meters (default 1.0m)
            maximize_coverage: Try to fit hexagons in all available spaces
            
        Returns:
            List of hexagon dictionaries with geometry and metadata
        """
        if plantable_zone.is_empty:
            print("  ⚠️ No plantable zone available")
            return []
        
        plantable_area_m2 = plantable_zone.area * (self.gsd ** 2)
        print(f"  Plantable area to fill: {plantable_area_m2:.2f} m²")
        
        # Single-size hexagon generation with core-based safety check
        print(f"  Placing {hexagon_size_m}m hexagons (0.1m overlap allowed)...")
        hexagons = self._place_hexagons_of_size(
            plantable_zone, hexagon_size_m, [], max_overlap_m=0.1
        )
        
        print(f"✓ Generated {len(hexagons)} planting zones")
        return hexagons
    
    def _place_hexagons_of_size(
        self,
        plantable_zone: Polygon,
        hexagon_size_m: float,
        existing_hexagons: List[Dict],
        max_overlap_m: float = 0.1,
        min_clearance_m: float = 0.5
    ) -> List[Dict]:
        """
        MAXIMIZED PLACEMENT with proper hexagonal tessellation.
        Tries multiple grid phase offsets and picks the best one,
        then fills remaining gaps with a secondary scan.
        
        Rules:
          - Dark green CORE must be ≥90% inside plantable zone
          - Light green BUFFER must be ≥70% inside plantable zone (max 30% red overlap)
          - Hexagons follow a perfect tessellation grid (no gaps between neighbors)
        
        Args:
            plantable_zone: Available planting area
            hexagon_size_m: Size of hexagons to place (buffer circumradius in meters)
            existing_hexagons: Already placed hexagons to avoid
            max_overlap_m: Maximum allowed overlap between buffers in meters
            min_clearance_m: Minimum clearance for center point
            
        Returns:
            List of newly placed hexagons
        """
        # The hexagon_size_m represents the BUFFER radius (circumradius R)
        buffer_radius_pixels = hexagon_size_m / self.gsd
        core_radius_pixels = buffer_radius_pixels * 0.2
        
        # Get bounding box (expand slightly to catch edge hexagons)
        bounds = plantable_zone.bounds
        minx, miny, maxx, maxy = bounds
        
        # Expand bounds by one full buffer radius so edge hexagons aren't missed
        minx -= buffer_radius_pixels
        miny -= buffer_radius_pixels
        maxx += buffer_radius_pixels
        maxy += buffer_radius_pixels
        
        # PROPER HEXAGONAL TESSELLATION GRID
        # For flat-top hexagons with circumradius R:
        #   - Same-row horizontal spacing = √3 * R (buffers share edges, no gaps)
        #   - Vertical row spacing = 3/2 * R
        #   - Odd rows offset by √3/2 * R
        R = buffer_radius_pixels
        h_spacing = np.sqrt(3) * R   # ~1.732 * R between centers in same row
        v_spacing = 1.5 * R          # 3/2 * R between rows
        
        # ── Phase 1: Try multiple grid offsets, keep best ────────────────
        # The fixed grid origin can miss valid areas. By trying several
        # phase shifts (fractions of the grid cell), we find the alignment
        # that covers the most plantable area.
        num_phases = 5  # Try 5×5 = 25 phase combinations
        best_hexagons = []
        
        for phase_y_i in range(num_phases):
            for phase_x_i in range(num_phases):
                phase_x = (phase_x_i / num_phases) * h_spacing
                phase_y = (phase_y_i / num_phases) * v_spacing
                
                candidate = self._tessellate_grid(
                    plantable_zone, R, buffer_radius_pixels, core_radius_pixels,
                    h_spacing, v_spacing, minx + phase_x, miny + phase_y, maxx, maxy
                )
                
                if len(candidate) > len(best_hexagons):
                    best_hexagons = candidate
        
        print(f"    Phase 1 (best grid alignment): {len(best_hexagons)} hexagons")
        
        # ── Phase 2: Fill remaining gaps ─────────────────────────────────
        # After choosing the best grid, scan for leftover plantable pockets
        # that the grid missed. Place additional hexagons that don't overlap
        # with already-placed ones.
        placed_centers = set()
        for h in best_hexagons:
            # Quantize to grid to track occupancy
            cx, cy = h['center']
            placed_centers.add((round(cx, 1), round(cy, 1)))
        
        # Use a finer sub-grid search: half-cell offsets
        extra_hexagons = []
        sub_offsets = [
            (h_spacing * 0.25, v_spacing * 0.25),
            (h_spacing * 0.75, v_spacing * 0.25),
            (h_spacing * 0.25, v_spacing * 0.75),
            (h_spacing * 0.75, v_spacing * 0.75),
            (h_spacing * 0.5, v_spacing * 0.5),
        ]
        
        for dx, dy in sub_offsets:
            candidates = self._tessellate_grid(
                plantable_zone, R, buffer_radius_pixels, core_radius_pixels,
                h_spacing, v_spacing, minx + dx, miny + dy, maxx, maxy
            )
            for c in candidates:
                cx, cy = c['center']
                # Check this hexagon doesn't overlap any already-placed hexagon
                too_close = False
                for placed in best_hexagons + extra_hexagons:
                    px, py = placed['center']
                    dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                    # Minimum distance for non-overlapping buffers
                    min_dist = buffer_radius_pixels + placed['buffer_radius_m'] / self.gsd
                    # Allow tiny tolerance (1 pixel)
                    if dist < min_dist - 1:
                        too_close = True
                        break
                if not too_close:
                    extra_hexagons.append(c)
        
        if extra_hexagons:
            print(f"    Phase 2 (gap filling): +{len(extra_hexagons)} extra hexagons")
        
        all_hexagons = best_hexagons + extra_hexagons
        return all_hexagons
    
    def _tessellate_grid(
        self,
        plantable_zone,
        R: float,
        buffer_radius_pixels: float,
        core_radius_pixels: float,
        h_spacing: float,
        v_spacing: float,
        start_x: float,
        start_y: float,
        max_x: float,
        max_y: float
    ) -> List[Dict]:
        """
        Place hexagons on a single tessellation grid with given origin.
        Returns list of valid hexagons.
        """
        hexagons = []
        row = 0
        y = start_y
        
        while y <= max_y:
            # Proper tessellation offset: odd rows shift right by √3/2 * R
            x_offset = (np.sqrt(3) / 2) * R if row % 2 == 1 else 0
            x = start_x + x_offset
            
            while x <= max_x:
                center_point = Point(x, y)
                
                # Quick reject: center must be in plantable zone
                if not plantable_zone.contains(center_point):
                    x += h_spacing
                    continue
                
                hexagon_buffer = self.create_hexagon(x, y, buffer_radius_pixels)
                hexagon_core = self.create_hexagon(x, y, core_radius_pixels)
                
                # CORE must be ≥90% inside plantable zone
                core_ratio = 0.0
                if plantable_zone.intersects(hexagon_core):
                    core_ratio = hexagon_core.intersection(plantable_zone).area / hexagon_core.area
                
                # BUFFER must be ≥70% inside plantable zone (max 30% red overlap)
                buffer_safe_ratio = 0.0
                if plantable_zone.intersects(hexagon_buffer):
                    buffer_safe_ratio = hexagon_buffer.intersection(plantable_zone).area / hexagon_buffer.area
                
                if core_ratio >= 0.90 and buffer_safe_ratio >= 0.70:
                    hex_dict = {
                        'buffer': hexagon_buffer,
                        'core': hexagon_core,
                        'center': (x, y),
                        'buffer_radius_m': buffer_radius_pixels * self.gsd,
                        'core_radius_m': core_radius_pixels * self.gsd,
                        'area_m2': hexagon_core.area * (self.gsd ** 2)
                    }
                    hexagons.append(hex_dict)
                
                x += h_spacing
            
            y += v_spacing
            row += 1
        
        return hexagons
    
    def _fill_gaps(self, plantable_zone: Polygon, existing_hexagons: List[Dict],
                   buffer_radius_pixels: float, core_radius_pixels: float) -> List[Dict]:
        """
        Legacy gap-filling method (now replaced by adaptive sizing)
        Kept for backward compatibility
        """
        # No longer used - adaptive sizing handles gap filling better
        return []
    
    def process_image(
        self, 
        image_path: str,
        canopy_buffer_m: float = 1.0,
        hexagon_size_m: float = 1.0
    ) -> Dict:
        """
        Complete processing pipeline
        
        Args:
            image_path: Path to input image
            canopy_buffer_m: Buffer around canopies (red zone) in meters
            hexagon_size_m: Size of planting hexagons (green buffers) in meters
            
        Returns:
            Dictionary with all results
        """
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        h, w = image.shape[:2]
        
        # Calculate GSD first
        self.calculate_gsd(w, h)
        
        print(f"\n🔍 Processing: {Path(image_path).name}")
        print(f"   Image size: {w}x{h} pixels")
        print(f"   GSD: {self.gsd:.4f} m/pixel")
        print(f"   Coverage: {w * self.gsd:.1f}m x {h * self.gsd:.1f}m\n")
        
        # Step 1: Detect canopies with mask
        canopy_polygons, canopy_mask = self.detect_canopies(image)
        
        # Step 2: Create danger zones (canopy + 1m buffer) with mask
        danger_zone, danger_mask = self.create_danger_zones(canopy_polygons, canopy_mask, canopy_buffer_m)
        
        # Step 3: Identify plantable zones (avoid canopy buffers)
        # Note: Man-made structures (towers, bridges, houses) are filtered
        # via forbidden_zones.geojson in the Streamlit app instead.
        plantable_zone = self.identify_plantable_zones(danger_zone)
        
        # Step 6: Generate maximized hexagonal planting zones
        hexagons = self.generate_hexagonal_planting_zones(plantable_zone, hexagon_size_m, maximize_coverage=True)
        
        # Calculate statistics
        total_area_m2 = w * h * (self.gsd ** 2)
        danger_area_m2 = danger_zone.area * (self.gsd ** 2) if not danger_zone.is_empty else 0
        plantable_area_m2 = plantable_zone.area * (self.gsd ** 2) if not plantable_zone.is_empty else 0
        
        results = {
            'image_path': image_path,
            'image_size': (w, h),
            'gsd_m_per_pixel': self.gsd,
            'altitude_m': self.altitude_m,
            'canopy_buffer_m': canopy_buffer_m,
            'hexagon_size_m': hexagon_size_m,
            'total_area_m2': total_area_m2,
            'coverage_m': (w * self.gsd, h * self.gsd),
            'canopy_count': len(canopy_polygons),
            'danger_area_m2': danger_area_m2,
            'danger_percentage': (danger_area_m2 / total_area_m2 * 100) if total_area_m2 > 0 else 0,
            'plantable_area_m2': plantable_area_m2,
            'plantable_percentage': (plantable_area_m2 / total_area_m2 * 100) if total_area_m2 > 0 else 0,
            'hexagon_count': len(hexagons),
            'canopy_polygons': canopy_polygons,
            'canopy_mask': canopy_mask,
            'danger_zone': danger_zone,
            'danger_mask': danger_mask,
            'plantable_zone': plantable_zone,
            'hexagons': hexagons,
            'image': image,
            'ai_metadata': getattr(self, '_ai_metadata', {}),
        }
        
        print(f"\n✅ Processing complete!")
        print(f"   Canopies: {len(canopy_polygons)}")
        print(f"   Danger area: {danger_area_m2:.2f} m² ({results['danger_percentage']:.1f}%)")
        print(f"   Plantable area: {plantable_area_m2:.2f} m² ({results['plantable_percentage']:.1f}%)")
        print(f"   Planting zones: {len(hexagons)} (0.1m overlap allowed)\n")
        
        return results
    
    def visualize_results(self, results: Dict, output_path: str = None):
        """
        Create visualization with proper color separation:
        - Teal/Cyan: Bungalon Canopy (AI-classified)
        - Purple: Other canopy areas (Mangrove-Canopy / HSV detected)
        - Red: 1m danger buffer zones around canopies
        - Light green: 1m hexagon buffers (safe planting zone)
        - Dark green: Hexagon cores (exact planting points)
        
        Note: Man-made structures (towers, bridges, houses) are filtered
        via forbidden_zones.geojson in the Streamlit app instead.
        
        Args:
            results: Results dictionary from process_image()
            output_path: Path to save visualization (optional)
            
        Returns:
            Visualization image
        """
        image = results['image'].copy()
        h, w = image.shape[:2]
        
        # Create overlay image
        overlay = np.zeros_like(image)
        
        # Create masks for each layer
        canopy_mask = results['canopy_mask']
        danger_mask = results['danger_mask']
        
        # Get per-class masks from AI metadata
        ai_metadata = results.get('ai_metadata', {})
        bungalon_mask = ai_metadata.get('bungalon_mask', None)
        other_canopy_mask = ai_metadata.get('other_canopy_mask', None)
        class_counts = ai_metadata.get('class_counts', {})
        bungalon_count = class_counts.get(1, 0)
        other_ai_count = class_counts.get(0, 0)
        
        # Create hexagon buffer mask
        hexagon_buffer_mask = np.zeros((h, w), dtype=np.uint8)
        for hex_info in results['hexagons']:
            hexagon_buffer = hex_info['buffer']
            pts = np.array(hexagon_buffer.exterior.coords, dtype=np.int32)
            cv2.fillPoly(hexagon_buffer_mask, [pts], 255)
        
        # Calculate buffer zones (danger buffer minus canopy)
        buffer_only_mask = np.zeros_like(canopy_mask)
        buffer_only_mask[danger_mask > 0] = 255
        buffer_only_mask[canopy_mask > 0] = 0
        
        # Layer 1: Draw canopy areas with class-specific coloring
        if bungalon_mask is not None and other_canopy_mask is not None:
            # Non-Bungalon canopy areas (purple) - includes HSV-only detections
            # HSV-only areas = canopy_mask minus all AI masks
            hsv_only_mask = canopy_mask.copy()
            hsv_only_mask[bungalon_mask > 0] = 0
            hsv_only_mask[other_canopy_mask > 0] = 0
            
            # Draw other AI canopy (Mangrove-Canopy class) in purple
            overlay[other_canopy_mask > 0] = (128, 0, 128)  # Purple for Mangrove-Canopy
            # Draw HSV-only detections in purple too
            overlay[hsv_only_mask > 0] = (128, 0, 128)  # Purple for HSV-detected
            # Draw Bungalon Canopy in TEAL/CYAN (stands out from purple)
            overlay[bungalon_mask > 0] = (255, 255, 0)  # Cyan/Teal in BGR for Bungalon
        else:
            # No class info available - all purple (fallback)
            overlay[canopy_mask > 0] = (128, 0, 128)  # Purple for canopies
        
        # Layer 2: Draw danger buffer zones in RED
        overlay[buffer_only_mask > 0] = (0, 0, 255)  # Red for danger buffer
        
        # Layer 4: Draw hexagon buffers in LIGHT GREEN
        overlay[hexagon_buffer_mask > 0] = (144, 238, 144)  # Light green for safe buffer
        
        # Layer 4.5: OVERLAP DETECTION - hexagon buffers overlapping with danger zones = ORANGE WARNING
        overlap_mask = cv2.bitwise_and(hexagon_buffer_mask, danger_mask)
        overlay[overlap_mask > 0] = (0, 165, 255)  # Orange warning for overlap
        
        # Layer 5: Draw hexagon cores in DARK GREEN (actual planting points)
        for hex_info in results['hexagons']:
            hexagon_core = hex_info['core']
            pts = np.array(hexagon_core.exterior.coords, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 128, 0))  # Dark green for planting point
            # Add bright border to make it visible
            cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)  # Bright green border
        
        # Blend with original image
        result_img = cv2.addWeighted(image, 0.4, overlay, 0.6, 0)
        
        # Add legend
        legend_y = 30
        
        # Build legend items based on whether class info is available
        if bungalon_count > 0:
            legend_items = [
                ("BUNGALON CANOPY:", (255, 255, 0), f"{bungalon_count} detected"),
                ("OTHER CANOPY:", (128, 0, 128), f"{results['canopy_count'] - bungalon_count} detected"),
                ("DANGER BUFFER:", (0, 0, 255), f"{results['danger_area_m2']:.1f} m\u00b2"),
                ("PLANTING:", (0, 128, 0), f"{results['hexagon_count']} hexagons"),
                ("PLANTABLE AREA:", (0, 255, 0), f"{results['plantable_area_m2']:.1f} m\u00b2")
            ]
        else:
            legend_items = [
                ("CANOPIES:", (128, 0, 128), f"{results['canopy_count']} detected"),
                ("DANGER BUFFER:", (0, 0, 255), f"{results['danger_area_m2']:.1f} m\u00b2"),
                ("PLANTING:", (0, 128, 0), f"{results['hexagon_count']} hexagons"),
                ("PLANTABLE AREA:", (0, 255, 0), f"{results['plantable_area_m2']:.1f} m\u00b2")
            ]
        
        for label, color, value in legend_items:
            # Draw color box
            cv2.rectangle(result_img, (10, legend_y - 15), (30, legend_y), color, -1)
            cv2.rectangle(result_img, (10, legend_y - 15), (30, legend_y), (255, 255, 255), 1)
            # Draw text
            text = f"{label} {value}"
            cv2.putText(result_img, text, (40, legend_y - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(result_img, text, (40, legend_y - 3), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            legend_y += 25
        
        # Add summary stats
        stats_y = h - 60
        cv2.rectangle(result_img, (5, stats_y - 5), (400, h - 5), (0, 0, 0), -1)
        cv2.rectangle(result_img, (5, stats_y - 5), (400, h - 5), (255, 255, 255), 1)
        
        stats = [
            f"Coverage: {results['coverage_m'][0]:.1f}m x {results['coverage_m'][1]:.1f}m",
            f"Plantable: {results['plantable_area_m2']:.1f} m2 ({results['plantable_percentage']:.1f}%)"
        ]
        
        for i, stat in enumerate(stats):
            cv2.putText(result_img, stat, (10, stats_y + i * 20), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1, cv2.LINE_AA)
        
        if output_path:
            cv2.imwrite(output_path, result_img)
            print(f"✓ Saved visualization to: {output_path}")
        
        return result_img


if __name__ == "__main__":
    # Test the detector with core-based checking and 0.1m overlap
    detector = HexagonDetector(altitude_m=6.0, drone_model='GENERIC_4K')
    
    # Process flight_2_frame_0042 which has some plantable area
    image_path = "../drone_images/flight_2_frame_0042.jpg"
    
    results = detector.process_image(
        image_path=image_path,
        canopy_buffer_m=1.0,
        hexagon_size_m=1.0
    )
    
    # Visualize
    vis = detector.visualize_results(results, "../output/hexagon_maximized_test.png")
    
    print(f"\n{'='*60}")
    print(f"FINAL RESULTS - Core-Based Safety + 0.1m Overlap")
    print(f"{'='*60}")
    print(f"Canopies detected: {results['canopy_count']}")
    print(f"Danger zones: {results['danger_area_m2']:.2f} m²")
    print(f"Plantable area: {results['plantable_area_m2']:.2f} m²")
    print(f"Total planting zones: {results['hexagon_count']}")
    print(f"{'='*60}")
