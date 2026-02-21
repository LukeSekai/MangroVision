"""
MangroVision - Hexagonal Planting Zone Detector
Detects canopies, creates danger zones, and generates hexagonal planting buffers
"""

import cv2
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
import geopandas as gpd
from typing import Tuple, List, Dict
import json
from pathlib import Path
import math

from gsd_calculator import GSDCalculator


class HexagonDetector:
    """Advanced canopy detector with hexagonal planting zones"""
    
    def __init__(self, altitude_m: float = 6.0, drone_model: str = 'GENERIC_4K'):
        """
        Initialize detector with flight parameters
        
        Args:
            altitude_m: Flight altitude in meters
            drone_model: Drone model for GSD calculation
        """
        self.altitude_m = altitude_m
        self.drone_model = drone_model
        self.gsd = None
        self.image_shape = None
        
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
        Detect mangrove canopies from image and create canopy mask
        Currently using HSV color detection (will be replaced with detectree2)
        
        Args:
            image: Input BGR image
            
        Returns:
            Tuple of (List of Shapely Polygon objects, binary canopy mask)
        """
        h, w = image.shape[:2]
        
        # Convert to HSV for green vegetation detection
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define green color range - BROADER to catch all vegetation variations
        # Hue: 30-90 (yellow-green to blue-green, covers all vegetation tones)
        # Saturation: 40-255 (catch both bright and dull green)
        # Value: 40-255 (catch shadows and bright areas)
        lower_green = np.array([30, 40, 40])
        upper_green = np.array([90, 255, 255])
        
        # Create mask
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Morphological operations to MERGE nearby vegetation into unified canopies
        # Use LARGER kernels to connect nearby green areas
        kernel_large = np.ones((15, 15), np.uint8)  # Large kernel for merging
        kernel_small = np.ones((5, 5), np.uint8)    # Small kernel for cleaning
        
        # First: close gaps to merge nearby vegetation
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_large, iterations=2)
        # Second: remove noise
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small)
        # Third: dilate to ensure connected regions
        mask = cv2.dilate(mask, kernel_large, iterations=1)
        
        # Find contours
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Create clean canopy mask
        canopy_mask = np.zeros((h, w), dtype=np.uint8)
        
        # Convert contours to Shapely polygons
        canopy_polygons = []
        min_area_pixels = 5000  # Much higher threshold to avoid tiny fragments (was 500)
        
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_area_pixels:
                # Convert to polygon (same method as working detector)
                points = contour.reshape(-1, 2)
                if len(points) >= 3:
                    poly = Polygon(points)
                    if poly.is_valid:
                        canopy_polygons.append(poly)
                        # Draw on canopy mask
                        cv2.fillPoly(canopy_mask, [points.astype(np.int32)], 255)
        
        print(f"✓ Detected {len(canopy_polygons)} canopy regions")
        return canopy_polygons, canopy_mask
    
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
        hexagon_size_m: float = 0.5,
        maximize_coverage: bool = True
    ) -> List[Dict]:
        """
        Generate maximized hexagonal planting zones
        Only the core (exact planting point) must be in safe zone
        Buffers can overlap with danger zones and each other (max 0.1m overlap)
        
        Args:
            plantable_zone: Available planting area
            hexagon_size_m: Hexagon buffer size in meters (default 0.5m)
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
        max_overlap_m: float = 0.1
    ) -> List[Dict]:
        """
        Place hexagons of a specific size in plantable zone
        Core must be fully in plantable zone, buffers can overlap up to max_overlap_m
        
        Args:
            plantable_zone: Available planting area
            hexagon_size_m: Size of hexagons to place
            existing_hexagons: Already placed hexagons to avoid
            max_overlap_m: Maximum allowed overlap between buffers in meters (default 0.1m)
            
        Returns:
            List of newly placed hexagons
        """
        # The hexagon_size_m represents the BUFFER radius
        buffer_radius_pixels = hexagon_size_m / self.gsd
        core_radius_pixels = buffer_radius_pixels * 0.2
        
        # Get bounding box
        if isinstance(plantable_zone, MultiPolygon):
            bounds = plantable_zone.bounds
        else:
            bounds = plantable_zone.bounds
        
        minx, miny, maxx, maxy = bounds
        
        # Use MUCH finer grid spacing to catch narrow plantable strips
        # Space based on CORE size, not buffer (buffer can overlap)
        # Check every ~0.2-0.3m to find all possible locations
        search_spacing_m = 0.25  # Search every 0.25m to find narrow strips
        search_spacing_pixels = search_spacing_m / self.gsd
        
        h_spacing = search_spacing_pixels
        v_spacing = search_spacing_pixels * 0.866  # Hexagonal offset
        
        hexagons = []
        
        row = 0
        y = miny + core_radius_pixels  # Start from edge + core radius
        
        while y < maxy:
            # Offset every other row for better coverage
            x_offset = search_spacing_pixels * 0.5 if row % 2 == 1 else 0
            x = minx + core_radius_pixels + x_offset  # Start from edge + core radius
            
            while x < maxx:
                # Create hexagon buffer
                hexagon_buffer = self.create_hexagon(x, y, buffer_radius_pixels)
                hexagon_core = self.create_hexagon(x, y, core_radius_pixels)
                
                # CRITICAL: Only check if CORE (actual planting point) is in plantable zone
                # The buffer can overlap with danger zones - that's just spacing!
                # Only the exact planting location (core) must be safe
                is_in_plantable = False
                
                if isinstance(plantable_zone, MultiPolygon):
                    # Check if core is within plantable zone
                    is_in_plantable = plantable_zone.contains(hexagon_core) or \
                                     (plantable_zone.intersects(hexagon_core) and \
                                      hexagon_core.intersection(plantable_zone).area / hexagon_core.area >= 0.95)
                else:
                    # Check if core is within plantable zone
                    is_in_plantable = plantable_zone.contains(hexagon_core) or \
                                     (plantable_zone.intersects(hexagon_core) and \
                                      hexagon_core.intersection(plantable_zone).area / hexagon_core.area >= 0.95)
                
                # Check that cores don't overlap and buffers overlap by max 0.1m
                # CRITICAL: Check against BOTH existing hexagons AND newly placed ones
                has_overlap = False
                if is_in_plantable:
                    # Combine existing hexagons with newly placed hexagons
                    all_placed_hexagons = existing_hexagons + hexagons
                    
                    for placed in all_placed_hexagons:
                        px, py = placed['center']
                        distance = np.sqrt((x - px)**2 + (y - py)**2)
                        
                        # Calculate minimum distance:
                        # Distance between centers = sum of buffer radii - allowed overlap
                        placed_buffer_radius = placed['buffer_radius_m'] / self.gsd
                        max_overlap_pixels = max_overlap_m / self.gsd
                        
                        # Buffers can overlap by up to max_overlap_m
                        min_distance = (buffer_radius_pixels + placed_buffer_radius) - max_overlap_pixels
                        
                        if distance < min_distance:
                            has_overlap = True
                            break
                
                if is_in_plantable and not has_overlap:
                    hex_dict = {
                        'buffer': hexagon_buffer,
                        'core': hexagon_core,
                        'center': (x, y),
                        'buffer_radius_m': hexagon_size_m,
                        'core_radius_m': hexagon_size_m * 0.2,
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
        hexagon_size_m: float = 0.5
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
        
        # Step 3: Identify plantable zones
        plantable_zone = self.identify_plantable_zones(danger_zone)
        
        # Step 4: Generate maximized hexagonal planting zones
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
            'image': image
        }
        
        print(f"\n✅ Processing complete!")
        print(f"   Canopies: {len(canopy_polygons)}")
        print(f"   Danger area: {danger_area_m2:.2f} m² ({results['danger_percentage']:.1f}%)")
        print(f"   Plantable area: {plantable_area_m2:.2f} m² ({results['plantable_percentage']:.1f}%)")
        print(f"   Planting zones: {len(hexagons)} (0.1m overlap allowed)\n")
        
        return results
    
    def visualize_results(self, results: Dict, output_path: str = None):
        """
        Create visualization with proper color separation and overlap detection:
        - Purple: Canopy areas (detected vegetation)
        - Red: 1m danger buffer zones (no overlap)
        - Light green: 0.5m hexagon buffers (safe planting zone, no overlap)
        - Orange: Overlap between danger buffer and planting buffer (WARNING)
        - Dark green: Hexagon cores (exact planting points)
        
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
        
        # Detect OVERLAP between danger buffer and hexagon buffer
        overlap_mask = np.zeros_like(canopy_mask)
        overlap_mask[(buffer_only_mask > 0) & (hexagon_buffer_mask > 0)] = 255
        
        # Remove overlaps from individual masks to avoid double-coloring
        buffer_only_mask[overlap_mask > 0] = 0
        hexagon_buffer_mask[overlap_mask > 0] = 0
        
        # Layer 1: Draw canopy areas in PURPLE
        overlay[canopy_mask > 0] = (128, 0, 128)  # Purple for canopies
        
        # Layer 2: Draw danger buffer zones in RED (no overlap areas)
        overlay[buffer_only_mask > 0] = (0, 0, 255)  # Red for danger buffer
        
        # Layer 3: Draw hexagon buffers in LIGHT GREEN (no overlap areas)
        overlay[hexagon_buffer_mask > 0] = (144, 238, 144)  # Light green for safe buffer
        
        # Layer 4: Draw OVERLAP areas in ORANGE (warning)
        overlay[overlap_mask > 0] = (0, 165, 255)  # Orange for overlap/warning
        
        # Layer 5: Draw hexagon cores in DARK GREEN (actual planting points)
        for hex_info in results['hexagons']:
            hexagon_core = hex_info['core']
            pts = np.array(hexagon_core.exterior.coords, dtype=np.int32)
            cv2.fillPoly(overlay, [pts], (0, 128, 0))  # Dark green for planting point
            # Add bright border to make it visible
            cv2.polylines(overlay, [pts], True, (0, 255, 0), 2)  # Bright green border
        
        # Blend with original image
        result_img = cv2.addWeighted(image, 0.4, overlay, 0.6, 0)
        
        # Add legend with overlap indicator
        legend_y = 30
        overlap_count = np.count_nonzero(overlap_mask)
        overlap_area_m2 = overlap_count * (results['gsd_m_per_pixel'] ** 2)
        
        legend_items = [
            ("CANOPIES:", (128, 0, 128), f"{results['canopy_count']} detected"),
            ("DANGER BUFFER (1m):", (0, 0, 255), f"{results['danger_area_m2']:.1f} m2"),
            ("PLANTING BUFFER (0.5m):", (144, 238, 144), f"Safe zones"),
            ("OVERLAP WARNING:", (0, 165, 255), f"{overlap_area_m2:.1f} m2" if overlap_area_m2 > 0 else "None"),
            ("PLANTING POINTS:", (0, 255, 0), f"{results['hexagon_count']} locations"),
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
        hexagon_size_m=0.5
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
