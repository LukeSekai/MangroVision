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


_AVAILABLE_HEX_BUFFER_COLOR = (144, 238, 144)
_AVAILABLE_HEX_CORE_COLOR = (0, 128, 0)
_AVAILABLE_HEX_BORDER_COLOR = (0, 255, 0)
_ERODED_HEX_BUFFER_COLOR = (80, 190, 255)
_ERODED_HEX_CORE_COLOR = (0, 140, 255)
_ERODED_HEX_BORDER_COLOR = (0, 80, 255)


def _hexagon_render_colors(hex_info: Dict) -> Tuple[tuple, tuple, tuple]:
    """Return orange colors for a point currently blocked by erosion."""
    if bool(
        hex_info.get("_eroded_unavailable")
        or hex_info.get("_inside_eroded_zone")
    ):
        return (
            _ERODED_HEX_BUFFER_COLOR,
            _ERODED_HEX_CORE_COLOR,
            _ERODED_HEX_BORDER_COLOR,
        )
    return (
        _AVAILABLE_HEX_BUFFER_COLOR,
        _AVAILABLE_HEX_CORE_COLOR,
        _AVAILABLE_HEX_BORDER_COLOR,
    )

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
                 ai_confidence: float = 0.87,
                 detection_mode: str = 'ai',
                 camera_info: Optional[Dict] = None,
                 gsd_override: Optional[float] = None):
        """
        Initialize the detector.

        Args:
            altitude_m: Flight altitude in meters
            drone_model: Drone model for GSD calculation
            ai_confidence: Confidence threshold for AI detection (0-1).
                Default 0.85 - a precision-leaning operating point above
                the selected checkpoint's 0.50 evaluation threshold, chosen
                to keep the planting pipeline clean of low-confidence AI
                proposals.
            detection_mode: 'ai' or 'hsv'
        """
        if detection_mode not in {'ai', 'hsv'}:
            detection_mode = 'ai'

        self.altitude_m = altitude_m
        self.drone_model = drone_model
        # Honor the caller-supplied confidence so the UI slider, the FastAPI
        # form default, and the model checkpoint can be tuned together. The
        # previous version hard-clamped this, which suppressed valid
        # detections from the new 1-class checkpoint that was tuned at 0.5.
        self.ai_confidence = float(ai_confidence)
        self.detection_mode = detection_mode
        self.camera_info = camera_info or {}
        self.gsd_override = (
            float(gsd_override)
            if gsd_override is not None
            and math.isfinite(float(gsd_override))
            and float(gsd_override) > 0
            else None
        )
        self.gsd = None
        self.image_shape = None
        self.ai_detector = None
        self._seedling_mask = None

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
        calculated_gsd, specs = GSDCalculator.calculate_gsd_from_metadata(
            altitude_m=self.altitude_m,
            camera_info=self.camera_info,
            drone_model=self.drone_model,
            image_width_px=image_width,
            image_height_px=image_height,
        )
        self.gsd = calculated_gsd
        if self.gsd_override is not None:
            specs = dict(specs)
            specs['nominal_gsd_m_per_pixel'] = float(calculated_gsd)
            specs['source'] = 'orthophoto_vegetation_calibration'
            self.gsd = float(self.gsd_override)
        # Store as (height, width) to match numpy convention
        self.image_shape = (image_height, image_width)
        self.gsd_specs = specs
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
        self._seedling_mask = np.zeros(image.shape[:2], dtype=np.uint8)

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
                        candidate_seedling_mask = ai_result[3]
                    elif len(ai_result) == 3:
                        canopy_polygons, canopy_mask, metadata = ai_result
                        candidate_seedling_mask = None
                    else:
                        canopy_polygons, canopy_mask = ai_result[0], ai_result[1]
                        metadata = {}
                        candidate_seedling_mask = None
                else:
                    raise ValueError(f"Unexpected AI result type: {type(ai_result)}")
                self._ai_metadata = metadata
                if (
                    isinstance(candidate_seedling_mask, np.ndarray)
                    and candidate_seedling_mask.shape == image.shape[:2]
                ):
                    self._seedling_mask = ((candidate_seedling_mask > 0).astype(np.uint8) * 255)
            except Exception as e:
                print(f"   AI detection failed: {e}")
                print("   Falling back to HSV detection")
                self._seedling_mask = np.zeros(image.shape[:2], dtype=np.uint8)
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

    def _detect_hsv(self, image: np.ndarray, min_area_m2: float = 0.12) -> Tuple[List[Polygon], np.ndarray]:
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

        # Dynamic minimum area based on GSD. Keep small real seedlings/canopies
        # while relying on vegetation-strength checks to reject specks.
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

    def mask_to_polygons(self, mask: np.ndarray, min_area_m2: float = 0.12) -> List[Polygon]:
        """Convert a binary canopy mask into shapely polygons."""
        canopy_polygons: List[Polygon] = []
        min_area_pixels = int(min_area_m2 / (self.gsd ** 2)) if self.gsd else 300

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if cv2.contourArea(contour) <= min_area_pixels:
                continue
            # Keep large canopy components faithful to the raster mask. A
            # pure perimeter-scaled epsilon over-simplifies big merged crowns
            # into diagonal/triangular polygons.
            epsilon = min(1.0, 0.001 * cv2.arcLength(contour, True))
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

    def create_danger_zones(
        self,
        canopy_polygons: List[Polygon],
        canopy_mask: np.ndarray,
        buffer_m: float = 2.0,
        extra_mask: Optional[np.ndarray] = None,
        extra_buffer_m: float = 0.0,
    ) -> Tuple[Polygon, np.ndarray]:
        """
        Create danger zones (canopies + configurable buffer) with proper masking
        CLIPS to image boundaries to prevent overflow

        Args:
            canopy_polygons: List of canopy polygons
            canopy_mask: Binary mask of canopy areas
            buffer_m: Buffer distance in meters (default 2.0m)
            extra_mask: Optional isolated seedling mask used only for safety
                buffering; it does not alter the mature-canopy mask.
            extra_buffer_m: Safety radius for ``extra_mask``. It is separate
                from the mature-canopy radius so tiny markers do not create a
                full-size canopy danger field.

        Returns:
            Tuple of (unified danger zone polygon, danger zone mask)
        """
        h, w = self.image_shape
        danger_mask = np.zeros((h, w), dtype=np.uint8)

        # Create image boundary polygon for clipping
        image_boundary = Polygon([(0, 0), (w, 0), (w, h), (0, h)])

        # Convert buffer distance to pixels
        # Keep seedling safety distance independent from the mature-canopy
        # buffer. Applying 2 m to hundreds of tiny markers merges them into a
        # continuous red field.
        buffer_pixels = max(0.0, float(buffer_m)) / self.gsd
        extra_buffer_pixels = max(0.0, float(extra_buffer_m)) / self.gsd
        print(f"   Buffer calculation: {buffer_m}m Ã· {self.gsd:.5f}m/px = {buffer_pixels:.1f} pixels")

        # Every canopy shown in the purple overlay must contribute to the red
        # danger buffer. Earlier versions skipped sub-0.1 m² polygons; that
        # made the preview look like "detected canopy with no safety zone" and
        # could leave refill candidates too close to visible vegetation.
        if isinstance(canopy_mask, np.ndarray) and canopy_mask.shape == danger_mask.shape:
            canopy_pixels = canopy_mask > 0
            seedling_pixels = (
                extra_mask > 0
                if isinstance(extra_mask, np.ndarray) and extra_mask.shape == danger_mask.shape
                else np.zeros_like(canopy_pixels)
            )
            if not np.any(canopy_pixels) and not np.any(seedling_pixels):
                return Polygon(), danger_mask
            if np.any(canopy_pixels):
                distance_from_canopy = cv2.distanceTransform(
                    (~canopy_pixels).astype(np.uint8),
                    cv2.DIST_L2,
                    5,
                )
                danger_mask[(distance_from_canopy <= buffer_pixels) | canopy_pixels] = 255
            if np.any(seedling_pixels):
                if extra_buffer_pixels > 0:
                    distance_from_seedlings = cv2.distanceTransform(
                        (~seedling_pixels).astype(np.uint8),
                        cv2.DIST_L2,
                        5,
                    )
                    danger_mask[
                        (distance_from_seedlings <= extra_buffer_pixels) | seedling_pixels
                    ] = 255
                else:
                    # Keep exact marker pixels unavailable without painting a
                    # mature-style halo around every tiny candidate.
                    danger_mask[seedling_pixels] = 255
            danger_polygons = self.mask_to_polygons(danger_mask, min_area_m2=0.0)
            danger_zone = unary_union(danger_polygons) if danger_polygons else Polygon()
            danger_zone = danger_zone.intersection(image_boundary)
        else:
            valid_canopy_polys = [
                poly for poly in canopy_polygons
                if poly is not None and not poly.is_empty
            ]
            if not valid_canopy_polys:
                return Polygon(), danger_mask

            buffered = [poly.buffer(buffer_pixels) for poly in valid_canopy_polys]
            danger_zone = unary_union(buffered).intersection(image_boundary)

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
        print(f"Created {buffer_m:.1f}m danger buffer zones ({danger_area_m2:.2f} m^2)")
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
        Create a flat-top hexagon polygon.

        Flat-top orientation (vertices at 0°, 60°, 120°, ... — left and right
        are single vertices, top and bottom are horizontal edges) tessellates
        without gaps in a column-based layout, which gives the dense uniform
        planting grid we want (horizontal rows of flat-topped hexes).

        Args:
            center_x: X coordinate of center
            center_y: Y coordinate of center
            radius_pixels: Circumradius in pixels (center to vertex)

        Returns:
            Hexagon polygon
        """
        # Start at 0° so the left and right of the hex are points (flat-top)
        # and the top/bottom edges are horizontal — perfect for edge-share columns.
        angles = [i * np.pi / 3 for i in range(6)]
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
        - Buffer may overlap danger zones (visual warning in orange)
        - Neighbor buffers do not overlap (0.0m)

        Args:
            plantable_zone: Available planting area
            hexagon_size_m: Hexagon buffer size in meters (default 1.0m)
            maximize_coverage: Try to fit hexagons in all available spaces
            danger_mask: Optional raster danger mask used to enforce core safety
            canopy_mask: Optional canopy mask retained for API compatibility

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
          - Light green BUFFER is a spacing/visual guide, not an exclusion test
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

        # Raster-space guard: enforce core safety against the cleaned danger mask.
        # Do not OR the raw canopy mask back in here: tiny AI specks that were
        # intentionally excluded from danger buffers would otherwise punch
        # one-cell holes into otherwise plantable 1 m lattices.
        danger_distance_map = None
        if isinstance(danger_mask, np.ndarray) and danger_mask.ndim == 2:
            try:
                safe_pixels = (danger_mask == 0).astype(np.uint8)
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
        
        # EDGE-SHARE HEXAGON GRID (flat-top, column-based)
        # For flat-top hexagons with circumradius R, the gap-free tessellation
        # uses:
        #   - Column horizontal spacing  = 1.5 * R
        #   - Same-column vertical spacing = sqrt(3) * R
        #   - Odd columns shift down by sqrt(3)/2 * R
        # Every neighbor pair is exactly sqrt(3)*R apart (nearest-neighbour
        # distance is identical to the pointy-top variant), and the hex buffers
        # share edges instead of leaving triangular corner-touch gaps. To hit a
        # species' target planting distance T, set hexagon_size = T / sqrt(3).
        R = buffer_radius_pixels
        h_spacing = 1.5 * R
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

        if core_ratio < 0.99:
            if placement_stats is not None:
                placement_stats['core_ratio_fail'] = placement_stats.get('core_ratio_fail', 0) + 1
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
        col = 0
        x = start_x

        while x <= max_x:
            # Edge-share offset: odd columns shift down by half the within-column
            # spacing (sqrt(3)/2 * R), so each hex slots into the gap between
            # its two diagonal neighbours in the column to its left/right.
            y_offset = v_spacing * 0.5 if col % 2 == 1 else 0.0
            y = start_y + y_offset

            while y <= max_y:
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

                y += v_spacing

            x += h_spacing
            col += 1

        return hexagons
    
    def process_image(
        self, 
        image_path: str,
        canopy_buffer_m: float = 2.0,
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

        # For the AI path, the detector returns an already-filtered raster
        # coverage mask. Keep that mask as the visual/source-of-truth layer;
        # rebuilding it from polygons can reintroduce simplification artifacts
        # on large merged canopy components.
        ai_mask_is_authoritative = bool(
            (getattr(self, "_ai_metadata", {}) or {}).get("detection_method")
        )
        if ai_mask_is_authoritative and isinstance(canopy_mask, np.ndarray):
            canopy_mask = ((canopy_mask > 0).astype(np.uint8) * 255)
        elif canopy_polygons:
            aligned_mask = np.zeros_like(canopy_mask)
            for poly in canopy_polygons:
                if poly.is_empty or poly.exterior is None:
                    continue
                pts = np.array(poly.exterior.coords, dtype=np.int32)
                cv2.fillPoly(aligned_mask, [pts], 255)

            # Post-rebuild smoothing. Kernel size of ~10 cm at typical drone
            # GSD (1-2 cm/px) is large enough to bridge sub-pixel polygon
            # rounding without merging genuinely-separate crowns. The kernel
            # is clamped so very-fine-GSD imagery doesn't get an oversized
            # close that erases real gaps.
            close_radius_m = 0.10
            if self.gsd and self.gsd > 0:
                close_px = int(round(close_radius_m / self.gsd))
            else:
                close_px = 5
            close_px = max(2, min(close_px, 20))
            kernel_size = 2 * close_px + 1
            kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE, (kernel_size, kernel_size)
            )
            aligned_mask = cv2.morphologyEx(
                aligned_mask, cv2.MORPH_CLOSE, kernel, iterations=1
            )

            canopy_mask = aligned_mask
        else:
            canopy_mask = np.zeros_like(canopy_mask)

        # Seedling candidates remain a separate evidence layer, but accepted
        # seedlings are still existing mangroves and must exclude planting
        # points. By default they inherit the user-selected canopy safety
        # radius. A numeric runtime override can deliberately use a smaller
        # seedling-specific radius (or zero) without altering mature polygons.
        seedling_buffer_m = float(canopy_buffer_m)
        if self.ai_detector is not None:
            configured_seedling_buffer = getattr(
                self.ai_detector,
                "runtime_tuning",
                {},
            ).get("seedling_buffer_m")
            if configured_seedling_buffer is not None:
                seedling_buffer_m = max(0.0, float(configured_seedling_buffer))

        # Step 2: Create danger zones (canopy + configured buffer) with mask
        danger_zone, danger_mask = self.create_danger_zones(
            canopy_polygons,
            canopy_mask,
            canopy_buffer_m,
            extra_mask=self._seedling_mask,
            extra_buffer_m=seedling_buffer_m,
        )
        
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
            'gsd_specs': getattr(self, 'gsd_specs', {}),
            'altitude_m': self.altitude_m,
            'canopy_buffer_m': canopy_buffer_m,
            'seedling_buffer_m': seedling_buffer_m,
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
            'seedling_mask': self._seedling_mask,
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
        - Red: configured danger buffer zones around canopies
        - Light/dark green: currently available planting markers
        - Light/dark orange: planting markers unavailable due to erosion
        
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
        seedling_mask = results.get('seedling_mask')
        hex_buffer_color = _AVAILABLE_HEX_BUFFER_COLOR
        hex_core_color = _AVAILABLE_HEX_CORE_COLOR
        eroded_buffer_color = _ERODED_HEX_BUFFER_COLOR
        eroded_core_color = _ERODED_HEX_CORE_COLOR
        eroded_count = sum(
            1
            for hex_info in results['hexagons']
            if bool(
                hex_info.get('_eroded_unavailable')
                or hex_info.get('_inside_eroded_zone')
            )
        )
        available_count = max(0, len(results['hexagons']) - eroded_count)
        planting_label = "AVAILABLE:" if eroded_count else "PLANTING:"
        
        # Get per-class masks from AI metadata
        ai_metadata = results.get('ai_metadata', {})
        bungalon_mask = ai_metadata.get('bungalon_mask', None)
        other_canopy_mask = ai_metadata.get('other_canopy_mask', None)
        class_counts = ai_metadata.get('class_counts', {})
        bungalon_count = class_counts.get(1, 0)
        other_ai_count = class_counts.get(0, 0)
        if isinstance(bungalon_mask, np.ndarray) and bungalon_mask.shape == canopy_mask.shape:
            bungalon_mask = cv2.bitwise_and(bungalon_mask, canopy_mask)
        if isinstance(other_canopy_mask, np.ndarray) and other_canopy_mask.shape == canopy_mask.shape:
            other_canopy_mask = cv2.bitwise_and(other_canopy_mask, canopy_mask)
        
        # Create hexagon buffer mask. Each hex is shrunk by HEX_VISUAL_SHRINK
        # around its center for the trihex-tile look — adjacent hexes leave
        # small triangular negative spaces between their corners instead of
        # edge-sharing. Pure rendering choice: placement geometry, lattice
        # spacing, and biological planting distance are all unchanged
        # (see _place_hexagons_of_size for the lattice math).
        HEX_VISUAL_SHRINK = 0.88
        hexagon_buffer_mask = np.zeros((h, w), dtype=np.uint8)
        eroded_hexagon_buffer_mask = np.zeros((h, w), dtype=np.uint8)
        for hex_info in results['hexagons']:
            full_hex = hex_info['buffer']
            cx, cy = hex_info['center']
            raw_coords = np.array(full_hex.exterior.coords)
            shrunk = (raw_coords - (cx, cy)) * HEX_VISUAL_SHRINK + (cx, cy)
            pts = shrunk.astype(np.int32)
            target_mask = (
                eroded_hexagon_buffer_mask
                if bool(
                    hex_info.get('_eroded_unavailable')
                    or hex_info.get('_inside_eroded_zone')
                )
                else hexagon_buffer_mask
            )
            cv2.fillPoly(target_mask, [pts], 255)
        
        # Calculate buffer zones (danger buffer minus canopy)
        buffer_only_mask = np.zeros_like(canopy_mask)
        buffer_only_mask[danger_mask > 0] = 255
        buffer_only_mask[canopy_mask > 0] = 0
        
        # Layer 1: Draw hexagon buffers in LIGHT GREEN. Keep this below canopy
        # and danger layers so dense planting grids do not hide detections.
        overlay[hexagon_buffer_mask > 0] = hex_buffer_color
        overlay[eroded_hexagon_buffer_mask > 0] = eroded_buffer_color

        # Layer 2: Draw canopy areas with class-specific coloring
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
        
        # Layer 3: Draw danger buffer zones in RED
        overlay[buffer_only_mask > 0] = (0, 0, 255)  # Red for danger buffer

        danger_contours, _ = cv2.findContours(danger_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if danger_contours:
            cv2.drawContours(overlay, danger_contours, -1, (0, 0, 255), 3)

        # Hybrid seedling candidates are rendered separately from mature
        # canopy geometry but use the same violet color for a uniform
        # vegetation overlay. They are drawn after the danger layer so the
        # reviewer can still see the exact candidate behind the safety buffer.
        if isinstance(seedling_mask, np.ndarray) and seedling_mask.shape == canopy_mask.shape:
            overlay[seedling_mask > 0] = (128, 0, 128)  # violet/purple in BGR

        # Layer 4.5: Overlap warning (buffer intersects danger zone)
        overlap_mask = cv2.bitwise_and(hexagon_buffer_mask, danger_mask)
        overlay[overlap_mask > 0] = (0, 165, 255)  # Orange warning
        
        # Layer 5: Draw hexagon cores in DARK GREEN (actual planting points)
        for hex_info in results['hexagons']:
            hexagon_core = hex_info['core']
            pts = np.array(hexagon_core.exterior.coords, dtype=np.int32)
            _, point_core_color, point_border_color = _hexagon_render_colors(hex_info)
            cv2.fillPoly(overlay, [pts], point_core_color)
            # Add bright border to make it visible
            cv2.polylines(overlay, [pts], True, point_border_color, 2)
        
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
                (planting_label, hex_core_color, f"{available_count if eroded_count else results['hexagon_count']} hexagons"),
                ("PLANTABLE AREA:", (0, 255, 0), f"{results['plantable_area_m2']:.1f} m\u00b2")
            ]
        else:
            legend_items = [
                ("CANOPIES:", (128, 0, 128), f"{results['canopy_count']} detected"),
                ("DANGER BUFFER:", (0, 0, 255), f"{results['danger_area_m2']:.1f} m\u00b2"),
                (planting_label, hex_core_color, f"{available_count if eroded_count else results['hexagon_count']} hexagons"),
                ("PLANTABLE AREA:", (0, 255, 0), f"{results['plantable_area_m2']:.1f} m\u00b2")
            ]
        if eroded_count:
            legend_items.insert(
                -1,
                ("ERODED / UNAVAILABLE:", eroded_core_color, f"{eroded_count} hexagons"),
            )
        
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


