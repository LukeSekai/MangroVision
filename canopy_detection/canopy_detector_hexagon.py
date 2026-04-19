"""
MangroVision - Hexagonal Planting Zone Detector
Detects canopies, creates danger zones, and generates hexagonal planting buffers
Supports HSV fallback and the official detectree2 integration
"""

import cv2
import numpy as np
from shapely.geometry import Point, Polygon, MultiPolygon
from shapely.ops import unary_union
from typing import Tuple, List, Dict, Optional, Callable, Any
from pathlib import Path
import math

from gsd_calculator import GSDCalculator

# Optional: Try to import the official detectree2 backend
try:
    from detectree2_proper import ProperDetectree2Detector
    DETECTREE2_AVAILABLE = True
    print("Proper Detectree2 library available and loaded")
except ImportError as e:
    DETECTREE2_AVAILABLE = False
    print(f"Detectree2 proper not available: {e}")
    print("   Using HSV color detection fallback")


class HexagonDetector:
    """Advanced canopy detector with hexagonal planting zones."""
    
    def __init__(self,
                 altitude_m: float = 6.0,
                 drone_model: str = 'GENERIC_4K',
                 ai_confidence: float = 0.80,
                 detection_mode: str = 'ai'):
        """
        Initialize the detector.

        Args:
            altitude_m: Flight altitude in meters
            drone_model: Drone model for GSD calculation
            ai_confidence: Confidence threshold for AI detection (0-1)
            detection_mode: 'ai' or 'hsv'
        """
        if detection_mode not in {'ai', 'hsv'}:
            detection_mode = 'ai'

        self.altitude_m = altitude_m
        self.drone_model = drone_model
        self.ai_confidence = 0.80 if detection_mode == 'ai' else ai_confidence
        self.detection_mode = detection_mode
        self.gsd = None
        self.image_shape = None
        self.ai_detector = None

        if detection_mode == 'ai' and abs(float(ai_confidence) - 0.80) > 1e-6:
            print(f"   AI confidence fixed at 0.80 (requested: {ai_confidence:.2f})")

        if detection_mode == 'ai' and DETECTREE2_AVAILABLE:
            print("Initializing MangroVision with AI detection system...")
            print(f"   Mode: {detection_mode.upper()}")
            try:
                self.ai_detector = ProperDetectree2Detector(
                    confidence_threshold=self.ai_confidence,
                    device='cpu'
                )
                self.ai_detector.setup_model()
                print("Detectree2 AI initialized successfully")
            except Exception as exc:
                print(f"AI detector initialization failed: {exc}")
                print("   Falling back to HSV detection")
                self.detection_mode = 'hsv'
                self.ai_detector = None
        elif detection_mode == 'hsv':
            print("Initializing MangroVision with HSV detection")
        else:
            print("Detectree2 not available - using HSV color detection fallback")
            self.detection_mode = 'hsv'

    def calculate_gsd(self, image_width: int, image_height: int):
        """Calculate Ground Sample Distance for the image"""
        self.gsd, specs = GSDCalculator.calculate_gsd_from_drone(
            altitude_m=self.altitude_m,
            drone_model=self.drone_model
        )
        # Store as (height, width) to match numpy convention
        self.image_shape = (image_height, image_width)
        return self.gsd

    def _run_ai_detection(
        self,
        image: np.ndarray,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ):
        """Call AI detector with backward-compatible kwargs."""
        detect_fn = self.ai_detector.detect_from_image
        attempts = [
            {"gsd": self.gsd, "progress_callback": progress_callback},
            {"gsd": self.gsd},
            {"progress_callback": progress_callback},
            {},
        ]

        last_type_error = None
        for kwargs in attempts:
            try:
                return detect_fn(image, **kwargs)
            except TypeError as err:
                last_type_error = err
                continue

        if last_type_error is not None:
            raise last_type_error
        return detect_fn(image)
    
    def detect_canopies(
        self,
        image: np.ndarray,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Tuple[List[Polygon], np.ndarray]:
        """
        Detect canopy polygons using AI when available, otherwise HSV.

        Args:
            image: Input BGR image

        Returns:
            Tuple of (List of Shapely Polygon objects, binary canopy mask)
        """
        self._ai_metadata = {}

        if self.detection_mode == 'ai' and self.ai_detector is not None:
            print("Running AI-only detection...")

            try:
                ai_result = self._run_ai_detection(
                    image,
                    progress_callback=progress_callback,
                )
                if isinstance(ai_result, (list, tuple)):
                    if len(ai_result) >= 4:
                        canopy_polygons, canopy_mask, metadata = ai_result[0], ai_result[1], ai_result[2]
                    elif len(ai_result) == 3:
                        canopy_polygons, canopy_mask, metadata = ai_result
                    else:
                        canopy_polygons, canopy_mask = ai_result[0], ai_result[1]
                        metadata = {}
                else:
                    raise ValueError(f"Unexpected AI result type: {type(ai_result)}")
                self._ai_metadata = metadata
            except Exception as e:
                print(f"   AI detection failed: {e}")
                print("   Falling back to HSV detection")
                return self._detect_hsv(image)

            total_canopy_pixels = np.count_nonzero(canopy_mask)
            total_canopy_m2 = total_canopy_pixels * (self.gsd ** 2)

            print(f"AI detected {len(canopy_polygons)} tree crowns (Total: {total_canopy_m2:.1f} m2)")
            print(f"   Using: {metadata.get('detection_method', 'detectree2')}")

            return canopy_polygons, canopy_mask

        print("Running HSV-only detection...")

        canopy_polygons, canopy_mask = self._detect_hsv(image)

        total_canopy_pixels = np.count_nonzero(canopy_mask)
        total_canopy_m2 = total_canopy_pixels * (self.gsd ** 2)

        print(f"HSV detected {len(canopy_polygons)} tree crowns (Total: {total_canopy_m2:.1f} m2)")

        return canopy_polygons, canopy_mask

    def _detect_hsv(self, image: np.ndarray) -> Tuple[List[Polygon], np.ndarray]:
        """
        HSV canopy detection with vegetation-strength validation.
        Prevents large open mud/water areas from being mislabeled as canopy.

        Args:
            image: Input BGR image

        Returns:
            Tuple of (List of Shapely Polygon objects, binary canopy mask)
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        sat = hsv[:, :, 1]

        # Excess Green + channel dominance checks:
        # mud/water can sit in a green-ish hue range, but usually lacks strong green dominance.
        bgr_f = image.astype(np.float32)
        b = bgr_f[:, :, 0]
        g = bgr_f[:, :, 1]
        r = bgr_f[:, :, 2]

        excess_green = (2.0 * g) - r - b
        exg_mask = (excess_green > 16.0).astype(np.uint8) * 255
        green_dom_mask = ((g > (r + 8.0)) & (g > (b + 6.0))).astype(np.uint8) * 255
        vegetation_strength_mask = cv2.bitwise_or(exg_mask, green_dom_mask)

        # Stricter canopy hue/sat/value windows.
        primary_green = cv2.inRange(hsv, np.array([25, 45, 30]), np.array([95, 255, 255]))
        yellow_green = cv2.inRange(hsv, np.array([18, 55, 40]), np.array([30, 255, 255]))
        dark_green = cv2.inRange(hsv, np.array([25, 25, 12]), np.array([95, 200, 90]))
        dark_green = cv2.bitwise_and(dark_green, vegetation_strength_mask)

        canopy_mask = cv2.bitwise_or(primary_green, yellow_green)
        canopy_mask = cv2.bitwise_or(canopy_mask, dark_green)
        canopy_mask = cv2.bitwise_and(canopy_mask, vegetation_strength_mask)

        # Remove very low-saturation regions that commonly represent mud/water.
        canopy_mask[sat < 22] = 0

        # Conservative morphology: keep local cleanup, avoid large-gap bridging.
        canopy_mask = cv2.morphologyEx(
            canopy_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8), iterations=1
        )
        canopy_mask = cv2.morphologyEx(
            canopy_mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8), iterations=1
        )

        canopy_polygons = []
        cleaned_mask = np.zeros_like(canopy_mask)

        # Dynamic minimum area based on GSD (0.5 m2 minimum).
        min_area_m2 = 0.5
        min_area_pixels = int(min_area_m2 / (self.gsd ** 2)) if self.gsd else 300

        # Component-level filtering preserves interior gaps better than contour filling.
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(canopy_mask, connectivity=8)
        for label_idx in range(1, num_labels):
            component_area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if component_area <= min_area_pixels:
                continue

            component_pixels = labels == label_idx
            strong_exg_ratio = float(np.count_nonzero((excess_green > 20.0) & component_pixels)) / component_area
            green_dom_ratio = float(np.count_nonzero((g > (r + 8.0)) & (g > (b + 6.0)) & component_pixels)) / component_area
            mean_sat = float(np.mean(sat[component_pixels]))

            if max(strong_exg_ratio, green_dom_ratio) < 0.30:
                continue
            if mean_sat < 30.0 and max(strong_exg_ratio, green_dom_ratio) < 0.45:
                continue

            cleaned_mask[component_pixels] = 255

        canopy_polygons = self.mask_to_polygons(cleaned_mask, min_area_m2=min_area_m2)

        return canopy_polygons, cleaned_mask

    def mask_to_polygons(self, mask: np.ndarray, min_area_m2: float = 0.5) -> List[Polygon]:
        """Convert a binary canopy mask into shapely polygons."""
        canopy_polygons: List[Polygon] = []
        min_area_pixels = int(min_area_m2 / (self.gsd ** 2)) if self.gsd else 300

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) <= min_area_pixels:
                continue
            epsilon = 0.005 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape(-1, 2)

            if len(points) >= 3:
                poly = Polygon(points)
                if poly.is_valid:
                    canopy_polygons.append(poly)
                elif not poly.is_valid:
                    poly = poly.buffer(0)
                    if poly.is_valid and not poly.is_empty:
                        if isinstance(poly, Polygon):
                            canopy_polygons.append(poly)
                        elif isinstance(poly, MultiPolygon):
                            canopy_polygons.extend(list(poly.geoms))

        return canopy_polygons

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
        print(f"   Buffer calculation: {buffer_m}m Ã· {self.gsd:.5f}m/px = {buffer_pixels:.1f} pixels")
        
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
        print(f"âœ“ Created 1.0m danger buffer zones ({danger_area_m2:.2f} mÂ2)")
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
        angles = [i * np.pi / 3 for i in range(6)]  # 6 points, 60Â° apart
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
        maximize_coverage: bool = True,
        danger_mask: Optional[np.ndarray] = None,
        canopy_mask: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        Generate maximized hexagonal planting zones
        Core-safe mode:
        - Planting core (dark green) must stay fully out of danger zones
        - Buffer may partially overlap danger zones (visual warning in orange)
        - Neighbor buffers do not overlap (0.0m)

        Args:
            plantable_zone: Available planting area
            hexagon_size_m: Hexagon buffer size in meters (default 1.0m)
            maximize_coverage: Try to fit hexagons in all available spaces
            danger_mask: Optional raster danger mask used to enforce core safety
            canopy_mask: Optional canopy mask; combined with danger mask for display-consistent core safety

        Returns:
            List of hexagon dictionaries with geometry and metadata
        """
        if plantable_zone.is_empty:
            print("  âš ï¸ No plantable zone available")
            return []
        
        plantable_area_m2 = plantable_zone.area * (self.gsd ** 2)
        print(f"  Plantable area to fill: {plantable_area_m2:.2f} mÂ2")
        
        # Single-size hexagon generation with core-safe no-overlap mode
        print(f"  Placing {hexagon_size_m}m hexagons (core-safe, 0.0m overlap)...")
        hexagons = self._place_hexagons_of_size(
            plantable_zone,
            hexagon_size_m,
            [],
            max_overlap_m=0.0,
            danger_mask=danger_mask,
            canopy_mask=canopy_mask
        )
        
        print(f"âœ“ Generated {len(hexagons)} planting zones")
        return hexagons
    
    def _place_hexagons_of_size(
        self,
        plantable_zone: Polygon,
        hexagon_size_m: float,
        existing_hexagons: List[Dict],
        max_overlap_m: float = 0.0,
        min_clearance_m: float = 0.5,
        danger_mask: Optional[np.ndarray] = None,
        canopy_mask: Optional[np.ndarray] = None
    ) -> List[Dict]:
        """
        MAXIMIZED PLACEMENT with proper hexagonal tessellation.
        Tries multiple grid phase offsets and picks the best one,
        then fills remaining gaps with a secondary scan.
        
        Rules:
          - Dark green CORE must be fully outside danger zone
          - Light green BUFFER should mostly stay in plantable zone (>=70%)
          - Neighbor buffers can overlap up to configured max_overlap_m
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

        # Raster-space guard: enforce core safety against the exact displayed danger mask.
        danger_distance_map = None
        if isinstance(danger_mask, np.ndarray) and danger_mask.ndim == 2:
            try:
                combined_danger = danger_mask.copy()
                if isinstance(canopy_mask, np.ndarray) and canopy_mask.ndim == 2 and canopy_mask.shape == danger_mask.shape:
                    combined_danger = cv2.bitwise_or(combined_danger, canopy_mask)
                safe_pixels = (combined_danger == 0).astype(np.uint8)
                danger_distance_map = cv2.distanceTransform(safe_pixels, cv2.DIST_L2, 5)
            except Exception:
                danger_distance_map = None

        placement_stats = {
            'center_outside': 0,
            'core_clearance_fail': 0,
            'core_ratio_fail': 0,
            'buffer_ratio_fail': 0,
            'accepted': 0
        }
        
        # CORNER-TOUCH HEXAGON GRID
        # For flat-top hexagons with circumradius R:
        #   - Same-row horizontal spacing = 2 * R
        #   - Vertical row spacing = âˆš3 * R
        #   - Odd rows shift right by 1 * R
        # This matches the separated layout shown in the preferred screenshot.
        R = buffer_radius_pixels
        h_spacing = 2.0 * R
        v_spacing = np.sqrt(3) * R
        
        # â”€â”€ Phase 1: Try multiple grid offsets, keep best â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # The fixed grid origin can miss valid areas. By trying several
        # phase shifts (fractions of the grid cell), we find the alignment
        # that covers the most plantable area.
        num_phases = 5  # Try 5Ã—5 = 25 phase combinations
        num_phases = max(num_phases, 7)
        best_hexagons = []
        
        for phase_y_i in range(num_phases):
            for phase_x_i in range(num_phases):
                phase_x = (phase_x_i / num_phases) * h_spacing
                phase_y = (phase_y_i / num_phases) * v_spacing
                
                candidate = self._tessellate_grid(
                    plantable_zone, R, buffer_radius_pixels, core_radius_pixels,
                    h_spacing, v_spacing, minx + phase_x, miny + phase_y, maxx, maxy,
                    danger_distance_map=danger_distance_map,
                    placement_stats=placement_stats
                )
                
                if len(candidate) > len(best_hexagons):
                    best_hexagons = candidate
        
        print(f"    Phase 1 (best grid alignment): {len(best_hexagons)} hexagons")
        
        # â”€â”€ Phase 2: Fill remaining gaps â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
            (h_spacing * 0.125, v_spacing * 0.5),
            (h_spacing * 0.875, v_spacing * 0.5),
            (h_spacing * 0.5, v_spacing * 0.125),
            (h_spacing * 0.5, v_spacing * 0.875),
        ]
        
        for dx, dy in sub_offsets:
            candidates = self._tessellate_grid(
                plantable_zone, R, buffer_radius_pixels, core_radius_pixels,
                h_spacing, v_spacing, minx + dx, miny + dy, maxx, maxy,
                danger_distance_map=danger_distance_map,
                placement_stats=placement_stats
            )
            for c in candidates:
                if not self._buffers_overlap(c['buffer'], best_hexagons + extra_hexagons):
                    extra_hexagons.append(c)
                elif placement_stats is not None:
                    placement_stats['buffer_overlap_fail'] = placement_stats.get('buffer_overlap_fail', 0) + 1
        
        if extra_hexagons:
            print(f"    Phase 2 (gap filling): +{len(extra_hexagons)} extra hexagons")

        dense_hexagons = self._dense_fill_remaining_gaps(
            plantable_zone,
            buffer_radius_pixels,
            core_radius_pixels,
            best_hexagons + extra_hexagons,
            danger_distance_map=danger_distance_map,
            placement_stats=placement_stats
        )
        if dense_hexagons:
            print(f"    Phase 3 (dense edge fill): +{len(dense_hexagons)} extra hexagons")

        print(
            "    Placement diagnostics: "
            f"accepted={placement_stats.get('accepted', 0)}, "
            f"center_outside={placement_stats.get('center_outside', 0)}, "
            f"core_clearance_fail={placement_stats.get('core_clearance_fail', 0)}, "
            f"core_ratio_fail={placement_stats.get('core_ratio_fail', 0)}, "
            f"buffer_ratio_fail={placement_stats.get('buffer_ratio_fail', 0)}, "
            f"buffer_overlap_fail={placement_stats.get('buffer_overlap_fail', 0)}"
        )
        
        all_hexagons = best_hexagons + extra_hexagons + dense_hexagons
        return all_hexagons

    def _buffers_overlap(
        self,
        candidate_buffer: Polygon,
        placed_hexagons: List[Dict],
        area_tolerance_px: float = 1.0,
    ) -> bool:
        """Allow touching edges or corners, but reject true area overlap."""
        for placed in placed_hexagons:
            try:
                overlap_area = candidate_buffer.intersection(placed['buffer']).area
                if overlap_area > area_tolerance_px:
                    return True
            except Exception:
                continue
        return False

    def _dense_fill_remaining_gaps(
        self,
        plantable_zone: Polygon,
        buffer_radius_pixels: float,
        core_radius_pixels: float,
        placed_hexagons: List[Dict],
        danger_distance_map: Optional[np.ndarray] = None,
        placement_stats: Optional[Dict[str, int]] = None,
    ) -> List[Dict]:
        """Greedy fine-grid fill for irregular pockets the main lattice misses."""
        if plantable_zone.is_empty:
            return []

        try:
            occupied_union = unary_union([h['buffer'] for h in placed_hexagons]) if placed_hexagons else Polygon()
            remaining_zone = plantable_zone.difference(occupied_union)
        except Exception:
            remaining_zone = plantable_zone

        if remaining_zone.is_empty:
            return []

        minx, miny, maxx, maxy = remaining_zone.bounds
        step_x = max(buffer_radius_pixels * 0.55, 6.0)
        step_y = max(buffer_radius_pixels * 0.48, 6.0)
        accepted: List[Dict] = []

        row = 0
        y = miny
        while y <= maxy:
            x = minx + (step_x * 0.5 if row % 2 == 1 else 0.0)
            while x <= maxx:
                if not remaining_zone.contains(Point(x, y)):
                    x += step_x
                    continue

                candidate = self._evaluate_hex_candidate(
                    plantable_zone,
                    x,
                    y,
                    buffer_radius_pixels,
                    core_radius_pixels,
                    danger_distance_map=danger_distance_map,
                    placement_stats=placement_stats
                )
                if candidate is not None:
                    if not self._buffers_overlap(candidate['buffer'], placed_hexagons + accepted):
                        accepted.append(candidate)
                    elif placement_stats is not None:
                        placement_stats['buffer_overlap_fail'] = placement_stats.get('buffer_overlap_fail', 0) + 1
                x += step_x
            y += step_y
            row += 1

        return accepted

    def _evaluate_hex_candidate(
        self,
        plantable_zone: Polygon,
        x: float,
        y: float,
        buffer_radius_pixels: float,
        core_radius_pixels: float,
        danger_distance_map: Optional[np.ndarray] = None,
        placement_stats: Optional[Dict[str, int]] = None,
    ) -> Optional[Dict]:
        """Validate a candidate center and return the ready-to-place hex."""
        center_point = Point(x, y)

        if not plantable_zone.contains(center_point):
            if placement_stats is not None:
                placement_stats['center_outside'] = placement_stats.get('center_outside', 0) + 1
            return None

        hexagon_buffer = self.create_hexagon(x, y, buffer_radius_pixels)
        hexagon_core = self.create_hexagon(x, y, core_radius_pixels)

        if danger_distance_map is not None:
            cx = int(round(x))
            cy = int(round(y))
            if (
                cx < 0 or cy < 0
                or cy >= danger_distance_map.shape[0]
                or cx >= danger_distance_map.shape[1]
                or float(danger_distance_map[cy, cx]) < (core_radius_pixels + 0.5)
            ):
                if placement_stats is not None:
                    placement_stats['core_clearance_fail'] = placement_stats.get('core_clearance_fail', 0) + 1
                return None

        core_ratio = 0.0
        if plantable_zone.intersects(hexagon_core):
            core_ratio = hexagon_core.intersection(plantable_zone).area / max(hexagon_core.area, 1e-9)

        buffer_safe_ratio = 0.0
        if plantable_zone.intersects(hexagon_buffer):
            buffer_safe_ratio = hexagon_buffer.intersection(plantable_zone).area / max(hexagon_buffer.area, 1e-9)

        if core_ratio < 0.99:
            if placement_stats is not None:
                placement_stats['core_ratio_fail'] = placement_stats.get('core_ratio_fail', 0) + 1
            return None
        if buffer_safe_ratio < 0.70:
            if placement_stats is not None:
                placement_stats['buffer_ratio_fail'] = placement_stats.get('buffer_ratio_fail', 0) + 1
            return None

        if placement_stats is not None:
            placement_stats['accepted'] = placement_stats.get('accepted', 0) + 1

        return {
            'buffer': hexagon_buffer,
            'core': hexagon_core,
            'center': (x, y),
            'buffer_radius_m': buffer_radius_pixels * self.gsd,
            'core_radius_m': core_radius_pixels * self.gsd,
            'area_m2': hexagon_core.area * (self.gsd ** 2)
        }
    
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
        max_y: float,
        danger_distance_map: Optional[np.ndarray] = None,
        placement_stats: Optional[Dict[str, int]] = None
    ) -> List[Dict]:
        """
        Place hexagons on a single tessellation grid with given origin.
        Returns list of valid hexagons.
        """
        hexagons = []
        row = 0
        y = start_y
        
        while y <= max_y:
            # Corner-touch offset: odd rows shift right by one radius.
            x_offset = R if row % 2 == 1 else 0.0
            x = start_x + x_offset
            
            while x <= max_x:
                hex_dict = self._evaluate_hex_candidate(
                    plantable_zone,
                    x,
                    y,
                    buffer_radius_pixels,
                    core_radius_pixels,
                    danger_distance_map=danger_distance_map,
                    placement_stats=placement_stats
                )
                if hex_dict is not None:
                    hexagons.append(hex_dict)
                
                x += h_spacing
            
            y += v_spacing
            row += 1
        
        return hexagons
    
    def process_image(
        self, 
        image_path: str,
        canopy_buffer_m: float = 1.0,
        hexagon_size_m: float = 1.0,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
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
        
        print(f"\nðŸ” Processing: {Path(image_path).name}")
        print(f"   Image size: {w}x{h} pixels")
        print(f"   GSD: {self.gsd:.4f} m/pixel")
        print(f"   Coverage: {w * self.gsd:.1f}m x {h * self.gsd:.1f}m\n")
        
        # Step 1: Detect canopies with mask
        canopy_polygons, canopy_mask = self.detect_canopies(
            image,
            progress_callback=progress_callback,
        )
        
        # Step 2: Create danger zones (canopy + 1m buffer) with mask
        danger_zone, danger_mask = self.create_danger_zones(canopy_polygons, canopy_mask, canopy_buffer_m)
        
        # Step 3: Identify plantable zones (avoid canopy buffers)
        # Note: Man-made structures (towers, bridges, houses) are filtered
        # via the forbidden-zone GeoJSON in the Streamlit app.
        # Note: Points outside orthophoto coverage are filtered in the
        # map section of app.py via is_inside_any_orthophoto().
        plantable_zone = self.identify_plantable_zones(danger_zone)
        
        # Step 6: Generate maximized hexagonal planting zones
        hexagons = self.generate_hexagonal_planting_zones(
            plantable_zone,
            hexagon_size_m,
            maximize_coverage=True,
            danger_mask=danger_mask,
            canopy_mask=canopy_mask
        )
        
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
        
        print(f"\nâœ… Processing complete!")
        print(f"   Canopies: {len(canopy_polygons)}")
        print(f"   Danger area: {danger_area_m2:.2f} mÂ2 ({results['danger_percentage']:.1f}%)")
        print(f"   Plantable area: {plantable_area_m2:.2f} mÂ2 ({results['plantable_percentage']:.1f}%)")
        print(f"   Planting zones: {len(hexagons)} (0.0m buffer overlap, core-safe)\n")
        
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
        via the forbidden-zone GeoJSON in the Streamlit app instead.
        
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

        # Layer 4.5: Overlap warning (buffer intersects danger zone)
        overlap_mask = cv2.bitwise_and(hexagon_buffer_mask, danger_mask)
        overlay[overlap_mask > 0] = (0, 165, 255)  # Orange warning
        
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
            print(f"âœ“ Saved visualization to: {output_path}")
        
        return result_img


