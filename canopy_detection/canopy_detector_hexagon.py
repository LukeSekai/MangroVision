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
from collections import Counter

from gsd_calculator import GSDCalculator

# Optional: Try to import detectree2 detector
try:
    from detectree2_proper import ProperDetectree2Detector
    DETECTREE2_AVAILABLE = True
    print("âœ… Proper Detectree2 library available and loaded")
except ImportError as e:
    try:
        from detectree2_detector import Detectree2Detector
        DETECTREE2_AVAILABLE = True
        print("âœ… Detectree2 AI available and loaded (fallback)")
    except ImportError as e:
        DETECTREE2_AVAILABLE = False
        print(f"âš ï¸ detectree2_detector not available: {e}")
        print(f"   Using HSV color detection fallback")
class HexagonDetector:
    """Advanced canopy detector with hexagonal planting zones - Smart Hybrid System"""
    
    def __init__(self, 
                 altitude_m: float = 6.0, 
                 drone_model: str = 'GENERIC_4K',
                 ai_confidence: float = 0.75,
                 model_name: str = 'benchmark',
                 detection_mode: str = 'ai'):
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
        # Force AI-only evaluation path: disable HSV+AI merge mode.
        if detection_mode == 'hybrid':
            print("   Hybrid mode disabled for evaluation; forcing AI-only mode.")
            detection_mode = 'ai'

        self.altitude_m = altitude_m
        self.drone_model = drone_model
        # Keep AI confidence fixed at 0.75 for AI-driven modes.
        self.ai_confidence = 0.75 if detection_mode in ['ai', 'hybrid'] else ai_confidence
        self.detection_mode = detection_mode
        self.gsd = None
        self.image_shape = None
        self.use_ai = DETECTREE2_AVAILABLE
        self._last_ai_filter_stats = {}
        self._last_hsv_merge_stats = {}
        self._last_hexagon_placement_stats = {}

        if detection_mode in ['ai', 'hybrid'] and abs(float(ai_confidence) - 0.75) > 1e-6:
            print(f"   AI confidence fixed at 0.75 (requested: {ai_confidence:.2f})")

        # Initialize detectree2 AI detector if needed (for 'ai' or 'hybrid' modes)
        if detection_mode in ['ai', 'hybrid'] and DETECTREE2_AVAILABLE:
            print(f"ðŸŒ³ Initializing MangroVision with AI detection system...")
            print(f"   Mode: {detection_mode.upper()}")
            try:
                # Try proper detectree2 integration first
                self.ai_detector = ProperDetectree2Detector(
                    confidence_threshold=self.ai_confidence,
                    device='cpu'  # Change to 'cuda' if GPU available
                )
                self.ai_detector.setup_model()
                print(f"âœ“ Detectree2 AI initialized successfully")
            except Exception:
                # Fallback to custom detector
                try:
                    detector_cls = Detectree2Detector
                except NameError:
                    from detectree2_detector import Detectree2Detector as detector_cls

                self.ai_detector = detector_cls(
                    confidence_threshold=self.ai_confidence,
                    device='cpu',
                    model_name=model_name
                )
                self.ai_detector.setup_model()
                print(f"âœ“ Custom Detectree2 initialized successfully")
        elif detection_mode == 'hsv':
            print(f"ðŸŒ³ Initializing MangroVision with HSV detection (fast mode)")
            self.ai_detector = None
        else:
            print(f"âš ï¸  Detectree2 not available - using HSV color detection fallback")
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
        # Reset AI metadata each run
        self._ai_metadata = {}

        if self.detection_mode == 'hybrid' and self.ai_detector is not None:
            # SMART HYBRID: Run both HSV and AI, then merge results
            print(f"?? Running Smart Hybrid Detection (HSV + AI)...")

            # Step 1: HSV Detection (catches everything green)
            hsv_polygons, hsv_mask = self._detect_hsv(image)

            # Step 2: AI Detection (high-confidence canopies)
            try:
                try:
                    ai_result = self.ai_detector.detect_from_image(image, gsd=self.gsd)
                except TypeError:
                    ai_result = self.ai_detector.detect_from_image(image)
                # Handle 2, 3, or 4-value returns from different detector versions
                if isinstance(ai_result, (list, tuple)):
                    if len(ai_result) >= 4:
                        ai_polygons, ai_mask, metadata = ai_result[0], ai_result[1], ai_result[2]
                    elif len(ai_result) == 3:
                        ai_polygons, ai_mask, metadata = ai_result
                    else:
                        ai_polygons, ai_mask = ai_result[0], ai_result[1]
                        metadata = {}
                else:
                    raise ValueError(f"Unexpected AI result type: {type(ai_result)}")
                # Store AI metadata (contains per-class masks and class info)
                self._ai_metadata = metadata
            except Exception as e:
                print(f"   ?? AI detection failed: {e}")
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

            print(f"? Hybrid Detection Results:")
            print(f"   - HSV found: {len(hsv_polygons)} crowns")
            print(f"   - AI found: {len(ai_polygons)} crowns")
            if bungalon_count > 0:
                print(f"     ? Bungalon Canopy: {bungalon_count}")
                print(f"     ? Mangrove-Canopy: {other_ai_count}")
            print(f"   - Merged total: {len(combined_polygons)} crowns ({total_canopy_m2:.1f} m2)")
            print(f"   - Method: UNION (best of both worlds)")

            return combined_polygons, combined_mask

        elif self.detection_mode == 'ai' and self.ai_detector is not None:
            # AI ONLY MODE
            print(f"?? Running AI-only detection...")

            try:
                try:
                    ai_result = self.ai_detector.detect_from_image(image, gsd=self.gsd)
                except TypeError:
                    ai_result = self.ai_detector.detect_from_image(image)
                # Handle 2, 3, or 4-value returns from different detector versions
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
                print(f"   ?? AI detection failed: {e}")
                print(f"   Falling back to HSV detection")
                return self._detect_hsv(image)

            total_canopy_pixels = np.count_nonzero(canopy_mask)
            total_canopy_m2 = total_canopy_pixels * (self.gsd ** 2)

            print(f"? AI detected {len(canopy_polygons)} tree crowns (Total: {total_canopy_m2:.1f} m2)")
            print(f"   Using: {metadata.get('detection_method', 'detectree2')}")

            return canopy_polygons, canopy_mask

        else:
            # HSV ONLY MODE (fallback or explicit choice)
            print(f"?? Running HSV-only detection...")

            canopy_polygons, canopy_mask = self._detect_hsv(image)

            total_canopy_pixels = np.count_nonzero(canopy_mask)
            total_canopy_m2 = total_canopy_pixels * (self.gsd ** 2)

            print(f"? HSV detected {len(canopy_polygons)} tree crowns (Total: {total_canopy_m2:.1f} m2)")

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

        contours, _ = cv2.findContours(cleaned_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
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

        return canopy_polygons, cleaned_mask

    def _polygon_compactness(self, poly: Polygon) -> float:
        """Compactness in [0..1], where 1 is a perfect circle."""
        if poly is None or poly.is_empty:
            return 0.0
        perimeter = float(poly.length)
        if perimeter <= 0:
            return 0.0
        return float((4.0 * np.pi * float(poly.area)) / (perimeter * perimeter))

    def _polygons_to_mask(self, polygons: List[Polygon], image_shape: Tuple[int, int, int]) -> np.ndarray:
        """Rasterize polygon list into a binary mask."""
        mask = np.zeros(image_shape[:2], dtype=np.uint8)
        if not polygons:
            return mask
        for poly in polygons:
            if poly is None or poly.is_empty:
                continue
            parts = [poly] if isinstance(poly, Polygon) else list(poly.geoms)
            for part in parts:
                if part.is_empty or part.exterior is None:
                    continue
                pts = np.array(part.exterior.coords, dtype=np.int32)
                if len(pts) >= 3:
                    cv2.fillPoly(mask, [pts], 255)
        return mask

    def _clip_ai_metadata_masks(self, metadata: Dict, final_mask: np.ndarray) -> Dict:
        """Clip optional AI class masks so visualization matches final accepted canopy mask."""
        if not isinstance(metadata, dict):
            return {}
        clipped = dict(metadata)
        for key in ("bungalon_mask", "other_canopy_mask"):
            class_mask = clipped.get(key, None)
            if isinstance(class_mask, np.ndarray) and class_mask.shape[:2] == final_mask.shape[:2]:
                clipped[key] = cv2.bitwise_and(class_mask, final_mask)
        return clipped

    def _filter_ai_primary_polygons(self, ai_polygons: List[Polygon], image: np.ndarray) -> List[Polygon]:
        """
        AI-primary post-filter.
        Keeps valid AI detections and removes only obvious false blobs.
        Adds reject-reason diagnostics to inspect scale/GSD effects.
        """
        self._last_ai_filter_stats = {}
        if not ai_polygons:
            self._last_ai_filter_stats = {
                'input': 0,
                'kept': 0,
                'rejected': 0,
                'reasons': {}
            }
            return []

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        bgr_f = image.astype(np.float32)
        b = bgr_f[:, :, 0]
        g = bgr_f[:, :, 1]
        r = bgr_f[:, :, 2]
        excess_green = (2.0 * g) - r - b
        green_hue_like = (hue >= 30) & (hue <= 100) & (sat >= 20) & (val >= 20)
        vegetation_like = ((excess_green > 10.0) & (g > (r + 2.0))) | green_hue_like

        if self.gsd:
            min_area_m2 = 0.06
            soft_large_m2 = 4.0
            max_area_m2 = 18.0
            min_area_px = max(1.0, min_area_m2 / (self.gsd ** 2))
            soft_large_px = max(1.0, soft_large_m2 / (self.gsd ** 2))
            max_area_px = max(1.0, max_area_m2 / (self.gsd ** 2))
        else:
            min_area_m2 = None
            soft_large_m2 = None
            max_area_m2 = None
            min_area_px = 120.0
            soft_large_px = 5200.0
            max_area_px = 18000.0

        reasons = Counter()
        reject_logs: List[str] = []
        near_min_rejects = 0
        near_max_rejects = 0

        def _reject(reason: str, details: str):
            reasons[reason] += 1
            reject_logs.append(f"{reason}: {details}")

        keep: List[Polygon] = []
        for idx, poly in enumerate(ai_polygons):
            if poly is None or poly.is_empty or not poly.is_valid:
                _reject('invalid_geometry', f"idx={idx}")
                continue
            if poly.area <= 0:
                _reject('invalid_geometry', f"idx={idx}, area<=0")
                continue

            area_px = float(poly.area)
            area_m2 = (area_px * (self.gsd ** 2)) if self.gsd else None

            is_large = area_px >= soft_large_px

            if area_px < min_area_px:
                if area_px >= (0.7 * min_area_px):
                    near_min_rejects += 1
                area_txt = f"{area_m2:.3f}m2" if area_m2 is not None else f"{area_px:.0f}px2"
                thr_txt = f"{min_area_m2:.2f}m2" if min_area_m2 is not None else f"{min_area_px:.0f}px2"
                _reject('too_small', f"idx={idx}, area={area_txt}, min={thr_txt}")
                continue
            if area_px > max_area_px:
                if area_px <= (1.3 * max_area_px):
                    near_max_rejects += 1
                area_txt = f"{area_m2:.3f}m2" if area_m2 is not None else f"{area_px:.0f}px2"
                thr_txt = f"{max_area_m2:.2f}m2" if max_area_m2 is not None else f"{max_area_px:.0f}px2"
                _reject('too_large', f"idx={idx}, area={area_txt}, max={thr_txt}")
                continue

            compactness = self._polygon_compactness(poly)
            if is_large:
                compactness_floor = 0.010
            elif area_px >= (0.35 * soft_large_px):
                compactness_floor = 0.004
            else:
                compactness_floor = 0.0015

            if compactness < compactness_floor:
                _reject(
                    'low_compactness',
                    f"idx={idx}, compactness={compactness:.4f}, min={compactness_floor:.4f}"
                )
                continue

            poly_mask = self._polygons_to_mask([poly], image.shape)
            pix = poly_mask > 0
            pix_count = int(np.count_nonzero(pix))
            if pix_count == 0:
                _reject('invalid_geometry', f"idx={idx}, empty_raster")
                continue

            veg_ratio = float(np.count_nonzero(vegetation_like & pix)) / pix_count
            green_hue_ratio = float(np.count_nonzero(green_hue_like & pix)) / pix_count
            mean_sat = float(np.mean(sat[pix]))
            sat_p75 = float(np.percentile(sat[pix], 75))
            sat_std = float(np.std(sat[pix]))
            gray_std = float(np.std(gray[pix]))

            if is_large:
                min_veg_ratio = 0.09
                min_green_hue_ratio = 0.10
                sat_floor = 18.0
            elif area_px >= (0.35 * soft_large_px):
                min_veg_ratio = 0.06
                min_green_hue_ratio = 0.07
                sat_floor = 15.0
            else:
                min_veg_ratio = 0.03
                min_green_hue_ratio = 0.04
                sat_floor = 11.0

            if veg_ratio < min_veg_ratio or green_hue_ratio < min_green_hue_ratio:
                _reject(
                    'low_vegetation_ratio',
                    f"idx={idx}, veg={veg_ratio:.3f}<{min_veg_ratio:.3f}, green={green_hue_ratio:.3f}<{min_green_hue_ratio:.3f}"
                )
                continue

            if (mean_sat < sat_floor) and (sat_p75 < (sat_floor + 5.0)) and (veg_ratio < (min_veg_ratio + 0.12)):
                _reject(
                    'low_saturation',
                    f"idx={idx}, mean_sat={mean_sat:.1f}, p75_sat={sat_p75:.1f}, floor={sat_floor:.1f}"
                )
                continue

            # Large smooth blobs (water/mud-like) tend to be low texture.
            if is_large and sat_std < 10.0 and gray_std < 12.0 and green_hue_ratio < 0.16:
                _reject(
                    'low_saturation',
                    f"idx={idx}, smooth_blob sat_std={sat_std:.1f}, gray_std={gray_std:.1f}, green={green_hue_ratio:.3f}"
                )
                continue

            keep.append(poly)

        thresholds = {
            'min_area_m2': min_area_m2,
            'soft_large_m2': soft_large_m2,
            'max_area_m2': max_area_m2,
            'min_area_px': int(round(min_area_px)),
            'soft_large_px': int(round(soft_large_px)),
            'max_area_px': int(round(max_area_px)),
        }
        self._last_ai_filter_stats = {
            'input': len(ai_polygons),
            'kept': len(keep),
            'rejected': max(0, len(ai_polygons) - len(keep)),
            'gsd': self.gsd,
            'thresholds': thresholds,
            'near_min_rejects': near_min_rejects,
            'near_max_rejects': near_max_rejects,
            'reasons': dict(reasons),
            'rejection_log': reject_logs
        }

        print("   AI post-filter diagnostics:")
        if self.gsd:
            print(
                f"     GSD={self.gsd:.5f} m/px | size gate={min_area_m2:.2f}-{max_area_m2:.1f} m2 "
                f"({int(round(min_area_px))}-{int(round(max_area_px))} px2)"
            )
        else:
            print(f"     GSD unavailable | size gate={int(round(min_area_px))}-{int(round(max_area_px))} px2")
        if near_min_rejects > 0 or near_max_rejects > 0:
            print(
                f"     Near-threshold rejects: min={near_min_rejects}, max={near_max_rejects} "
                f"(altitude/GSD sanity-check)"
            )
        for reason in ('too_small', 'too_large', 'low_compactness', 'low_vegetation_ratio', 'low_saturation'):
            if reasons.get(reason, 0) > 0:
                print(f"     {reason}: {reasons[reason]}")

        max_log_lines = 30
        if reject_logs:
            print("     Sample reject details:")
            for line in reject_logs[:max_log_lines]:
                print(f"       - {line}")
            if len(reject_logs) > max_log_lines:
                print(f"       - ... {len(reject_logs) - max_log_lines} more")

        return keep

    def _merge_detections(
        self,
        hsv_polygons: List[Polygon],
        ai_polygons: List[Polygon],
        ai_mask: np.ndarray = None,
        weak_ai_mask: Optional[np.ndarray] = None,
        image: np.ndarray = None
    ) -> List[Polygon]:
        """
        Merge HSV and AI detections using intelligent UNION

        Removes duplicates while keeping unique detections from both methods
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
        
        # â”€â”€ Step 1: Build a vegetation exclusion mask â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        
        # â”€â”€ Step 2: What remains = candidate structures â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Invert exclusion to get non-vegetation, non-soil, non-dark pixels
        candidate_mask = cv2.bitwise_not(exclusion_mask)
        
        # â”€â”€ Step 3: Detect GRAY/CONCRETE/METAL colors â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Low saturation = man-made materials (concrete, metal, painted wood)
        # MADE MORE RESTRICTIVE to avoid false positives with mud
        # Requires higher brightness to distinguish from mud
        lower_manmade = np.array([0, 0, 100])   # Increased from 70 to 100 (brighter)
        upper_manmade = np.array([180, 40, 230]) # Reduced saturation threshold from 45 to 40
        manmade_color = cv2.inRange(hsv, lower_manmade, upper_manmade)
        
        # â”€â”€ Step 4: Detect RED/RUST/ORANGE METAL (towers, bridges) â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
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
        
        # â”€â”€ Step 5: Combine - must be man-made color AND NOT excluded â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        structure_mask = cv2.bitwise_or(manmade_color, red_mask)
        structure_mask = cv2.bitwise_and(structure_mask, candidate_mask)
        
        # â”€â”€ Step 6: Morphological cleanup â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # Additional cleanup to connect nearby structure parts
        kernel_close = np.ones((9, 9), np.uint8)  # Increased from 7x7 to 9x9
        structure_mask = cv2.morphologyEx(structure_mask, cv2.MORPH_CLOSE, kernel_close, iterations=4)
        kernel_open = np.ones((5, 5), np.uint8)
        structure_mask = cv2.morphologyEx(structure_mask, cv2.MORPH_OPEN, kernel_open, iterations=1)
        
        # Find contours of structures
        contours, _ = cv2.findContours(structure_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        structure_polygons = []
        # Minimum area for structures (5 mÂ2 - filters out tiny noise)
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
        
        print(f"âœ“ Structure detection breakdown:")
        print(f"   Total area: {total_m2:.1f} mÂ2")
        print(f"   Excluded (vegetation/mud/water): {excluded_m2:.1f} mÂ2 ({excluded_pixels/total_pixels*100:.1f}%)")
        print(f"   Structures detected: {len(structure_polygons)} ({structure_m2:.1f} mÂ2)")
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
        
        # NOTE: Do NOT exclude brown mud/bare soil â€” that's where mangroves
        # should be planted!  Only exclude water, concrete, and buildings.
        
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
        
        print(f"âœ“ Detected non-vegetation areas: {forbidden_m2:.1f} mÂ2 ({forbidden_pct:.1f}% of image)")
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
        self._last_hexagon_placement_stats = {}
        
        # PROPER HEXAGONAL TESSELLATION GRID
        # For flat-top hexagons with circumradius R:
        #   - Same-row horizontal spacing = âˆš3 * R (buffers share edges, no gaps)
        #   - Vertical row spacing = 3/2 * R
        #   - Odd rows offset by âˆš3/2 * R
        R = buffer_radius_pixels
        h_spacing = np.sqrt(3) * R   # ~1.732 * R between centers in same row
        v_spacing = 1.5 * R          # 3/2 * R between rows
        
        # â”€â”€ Phase 1: Try multiple grid offsets, keep best â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€â”€
        # The fixed grid origin can miss valid areas. By trying several
        # phase shifts (fractions of the grid cell), we find the alignment
        # that covers the most plantable area.
        num_phases = 5  # Try 5Ã—5 = 25 phase combinations
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
        ]
        
        for dx, dy in sub_offsets:
            candidates = self._tessellate_grid(
                plantable_zone, R, buffer_radius_pixels, core_radius_pixels,
                h_spacing, v_spacing, minx + dx, miny + dy, maxx, maxy,
                danger_distance_map=danger_distance_map,
                placement_stats=placement_stats
            )
            for c in candidates:
                cx, cy = c['center']
                # Check this hexagon doesn't overlap any already-placed hexagon
                too_close = False
                allowed_overlap_px = (max_overlap_m / self.gsd) if (self.gsd and max_overlap_m > 0) else 0.0
                for placed in best_hexagons + extra_hexagons:
                    px, py = placed['center']
                    dist = np.sqrt((cx - px)**2 + (cy - py)**2)
                    # Minimum distance for non-overlapping buffers
                    min_dist = buffer_radius_pixels + placed['buffer_radius_m'] / self.gsd
                    if dist < (min_dist - allowed_overlap_px):
                        too_close = True
                        break
                if not too_close:
                    extra_hexagons.append(c)
        
        if extra_hexagons:
            print(f"    Phase 2 (gap filling): +{len(extra_hexagons)} extra hexagons")

        self._last_hexagon_placement_stats = placement_stats
        print(
            "    Placement diagnostics: "
            f"accepted={placement_stats.get('accepted', 0)}, "
            f"center_outside={placement_stats.get('center_outside', 0)}, "
            f"core_clearance_fail={placement_stats.get('core_clearance_fail', 0)}, "
            f"core_ratio_fail={placement_stats.get('core_ratio_fail', 0)}, "
            f"buffer_ratio_fail={placement_stats.get('buffer_ratio_fail', 0)}"
        )
        
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
            # Proper tessellation offset: odd rows shift right by âˆš3/2 * R
            x_offset = (np.sqrt(3) / 2) * R if row % 2 == 1 else 0
            x = start_x + x_offset
            
            while x <= max_x:
                center_point = Point(x, y)
                
                # Quick reject: center must be in plantable zone
                if not plantable_zone.contains(center_point):
                    if placement_stats is not None:
                        placement_stats['center_outside'] = placement_stats.get('center_outside', 0) + 1
                    x += h_spacing
                    continue
                
                hexagon_buffer = self.create_hexagon(x, y, buffer_radius_pixels)
                hexagon_core = self.create_hexagon(x, y, core_radius_pixels)

                # Root-cause fix: enforce core safety in raster space against danger mask.
                # If center does not have at least core-radius clearance, the core would
                # visually land in red/purple danger areas due to vector/raster mismatch.
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
                        x += h_spacing
                        continue

                core_ratio = 0.0
                if plantable_zone.intersects(hexagon_core):
                    core_ratio = hexagon_core.intersection(plantable_zone).area / max(hexagon_core.area, 1e-9)

                buffer_safe_ratio = 0.0
                if plantable_zone.intersects(hexagon_buffer):
                    buffer_safe_ratio = hexagon_buffer.intersection(plantable_zone).area / max(hexagon_buffer.area, 1e-9)

                # Core must stay fully safe; buffer can overlap danger up to 30%.
                if core_ratio < 0.99:
                    if placement_stats is not None:
                        placement_stats['core_ratio_fail'] = placement_stats.get('core_ratio_fail', 0) + 1
                    x += h_spacing
                    continue
                if buffer_safe_ratio < 0.70:
                    if placement_stats is not None:
                        placement_stats['buffer_ratio_fail'] = placement_stats.get('buffer_ratio_fail', 0) + 1
                    x += h_spacing
                    continue

                if placement_stats is not None:
                    placement_stats['accepted'] = placement_stats.get('accepted', 0) + 1

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
        
        print(f"\nðŸ” Processing: {Path(image_path).name}")
        print(f"   Image size: {w}x{h} pixels")
        print(f"   GSD: {self.gsd:.4f} m/pixel")
        print(f"   Coverage: {w * self.gsd:.1f}m x {h * self.gsd:.1f}m\n")
        
        # Step 1: Detect canopies with mask
        canopy_polygons, canopy_mask = self.detect_canopies(image)
        
        # Step 2: Create danger zones (canopy + 1m buffer) with mask
        danger_zone, danger_mask = self.create_danger_zones(canopy_polygons, canopy_mask, canopy_buffer_m)
        
        # Step 3: Identify plantable zones (avoid canopy buffers)
        # Note: Man-made structures (towers, bridges, houses) are filtered
        # via forbidden_zones.geojson in the Streamlit app.
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


if __name__ == "__main__":
    # Test the detector with core-safe overlap placement
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
    print(f"FINAL RESULTS - Core-Safe Overlap Placement")
    print(f"{'='*60}")
    print(f"Canopies detected: {results['canopy_count']}")
    print(f"Danger zones: {results['danger_area_m2']:.2f} mÂ2")
    print(f"Plantable area: {results['plantable_area_m2']:.2f} mÂ2")
    print(f"Total planting zones: {results['hexagon_count']}")
    print(f"{'='*60}")



