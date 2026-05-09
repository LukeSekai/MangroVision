"""
MangroVision - Proper Detectree2 Integration
Uses the official detectree2 library for accurate tree crown delineation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable, Any
import torch
from shapely.geometry import Polygon
import geopandas as gpd

# Official detectree2 imports
from detectree2.models.train import setup_cfg
from detectree2.models.outputs import clean_crowns
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


class ProperDetectree2Detector:
    """
    Proper integration with detectree2 library
    Uses official detectree2 prediction pipeline for maximum accuracy
    """
    
    def __init__(self,
                 confidence_threshold: float = 0.87,
                 device: str = 'cpu'):
        """
        Initialize proper detectree2 detector

        The default confidence threshold is 0.85. The selected checkpoint
        was evaluated at SCORE_THRESH_TEST=0.5; running at 0.85
        is a precision-leaning operating point that keeps low-confidence
        AI proposals from contaminating the planting pipeline.

        Args:
            confidence_threshold: Minimum confidence score (0-1)
            device: 'cpu' or 'cuda'
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.predictor = None
        self.cfg = None
        self.model_path = None
        self.model_name = None
        self.runtime_tuning = {
            "tile_veg_threshold": 0.002,
            # Lowered to 0.01 m^2 (~11 cm crown diameter at typical drone GSD)
            # so even very small saplings and recently planted seedlings
            # survive the area gate. The strict_canopy_hsv gate below remains
            # the primary defense against tiny false positives.
            "min_crown_m2": 0.01,
            "max_crown_m2": 250.0,
            "cleanup_iou": 0.75,
            "fallback_nms_iou": 0.90,
            "tile_size": 512,
            "tile_overlap": 0.25,
            "use_clean_crowns": False,
            "merge_canopy_fragments": True,
            # Bumped from 0.45 m → 1.50 m so adjacent crowns whose buffers
            # would otherwise leave red triangular wedges in the visualization
            # get morph-closed into a continuous canopy mass. This affects
            # *only* the rasterized canopy mass and the danger-buffer outline;
            # individual crown counts (final_polygons) are unchanged.
            "canopy_merge_gap_m": 0.20,
            # Bumped from 0.25 m → 0.5 m so the AI mask absorbs slightly more
            # of the adjacent strict-green vegetation, eliminating the thin
            # purple-to-red mottling at canopy edges.
            "canopy_hsv_expansion_m": 0.10,
            # Conservative color validation: keep AI as the detector, but reject
            # predictions that have almost no mangrove-green pixels inside them.
            "strict_canopy_hsv": True,
            "detection_veg_min_ratio": 0.03,
        }
        
        print(f"🌳 Initializing Proper Detectree2 Library")
        print(f"   Using official detectree2 prediction pipeline")
        print(f"   Device: {device}")
        print(f"   Confidence threshold: {confidence_threshold}")

    def set_runtime_tuning(
        self,
        tile_veg_threshold: float = None,
        min_crown_m2: float = None,
        max_crown_m2: float = None,
        cleanup_iou: float = None,
        fallback_nms_iou: float = None,
        tile_size: float = None,
        tile_overlap: float = None,
        use_clean_crowns: bool = None,
        merge_canopy_fragments: bool = None,
        canopy_merge_gap_m: float = None,
        canopy_hsv_expansion_m: float = None,
        strict_canopy_hsv: bool = None,
        detection_veg_min_ratio: float = None,
    ):
        """Update non-threshold inference tuning parameters."""
        if tile_veg_threshold is not None:
            self.runtime_tuning["tile_veg_threshold"] = max(0.0, min(float(tile_veg_threshold), 1.0))
        if min_crown_m2 is not None:
            self.runtime_tuning["min_crown_m2"] = max(0.01, float(min_crown_m2))
        if max_crown_m2 is not None:
            self.runtime_tuning["max_crown_m2"] = max(self.runtime_tuning["min_crown_m2"], float(max_crown_m2))
        if cleanup_iou is not None:
            self.runtime_tuning["cleanup_iou"] = max(0.05, min(float(cleanup_iou), 0.95))
        if fallback_nms_iou is not None:
            self.runtime_tuning["fallback_nms_iou"] = max(0.05, min(float(fallback_nms_iou), 0.99))
        if tile_size is not None:
            self.runtime_tuning["tile_size"] = int(max(256, min(float(tile_size), 2048)))
        if tile_overlap is not None:
            self.runtime_tuning["tile_overlap"] = max(0.0, min(float(tile_overlap), 0.6))
        if use_clean_crowns is not None:
            self.runtime_tuning["use_clean_crowns"] = bool(use_clean_crowns)
        if merge_canopy_fragments is not None:
            self.runtime_tuning["merge_canopy_fragments"] = bool(merge_canopy_fragments)
        if canopy_merge_gap_m is not None:
            self.runtime_tuning["canopy_merge_gap_m"] = max(0.0, min(float(canopy_merge_gap_m), 2.0))
        if canopy_hsv_expansion_m is not None:
            self.runtime_tuning["canopy_hsv_expansion_m"] = max(0.0, min(float(canopy_hsv_expansion_m), 2.0))
        if strict_canopy_hsv is not None:
            self.runtime_tuning["strict_canopy_hsv"] = bool(strict_canopy_hsv)
        if detection_veg_min_ratio is not None:
            self.runtime_tuning["detection_veg_min_ratio"] = max(0.0, min(float(detection_veg_min_ratio), 0.95))
        
    def setup_model(self, model_path: str = None):
        """
        Setup detectree2 model with proper configuration
        
        Args:
            model_path: Path to model weights (.pth file)
        """
        print(f"⚙️ Setting up detectree2 model...")
        
        selected_model_name = None

        # Find model file
        if model_path is None:
            project_root = Path(__file__).parent.parent
            model_dir = project_root / 'models'
            try_model_dir = project_root / 'MangroVision_New' / 'try_model'

            # Prefer the newly selected refined checkpoint bundled with the
            # MangroVision_New workspace. Older local weights remain as
            # fallbacks so the app can still start if the try_model file is
            # missing on another machine.
            latest_model_garden = sorted(model_dir.glob("250312*.pth"))
            model_candidates = [
                (try_model_dir / 'model_final.pth', 'MangroVision New Try Model (2-class)'),
                (model_dir / 'custom_mangrove_model' / 'model_final.pth', 'Custom Mangrove Model (2-class)'),
                (model_dir / 'latest_mangrove_only' / 'model_final.pth', 'Latest Mangrove Only (1-class)'),
            ]
            model_candidates.extend((p, f"Model Garden ({p.name})") for p in latest_model_garden)
            model_candidates.extend([
                (model_dir / '230103_randresize_full.pth', 'Optimized Tropical (Zenodo 230103)'),
                (model_dir / 'detectree2_model.pth', 'Custom Model'),
                (model_dir / '230717_tropical_base.pth', 'Base Tropical (230717)'),
            ])
             
            for candidate, model_name in model_candidates:
                if candidate.exists():
                    model_path = str(candidate)
                    selected_model_name = model_name
                    print(f"   ✅ Loading {model_name}")
                    break
            
            if model_path is None:
                raise FileNotFoundError(
                    f"No detectree2 model found in {model_dir}\n"
                    f"Download from: https://github.com/PatBall1/detectree2/releases"
                )
        
        # Prefer official detectree2 setup_cfg path, then fall back to manual config.
        cfg = None
        try:
            cfg = setup_cfg(update_model=model_path)
        except TypeError:
            # Some versions may expose a different setup_cfg signature.
            try:
                cfg = setup_cfg(model_path)
            except Exception:
                cfg = None
        except Exception:
            cfg = None

        if cfg is None:
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
            ))
            cfg.MODEL.WEIGHTS = model_path

        # Set NUM_CLASSES per checkpoint family. latest_mangrove_only and the
        # tropical-base / model-garden checkpoints are 1-class (canopy only);
        # the legacy custom_mangrove_model directory holds 2-class checkpoints
        # (Mangrove canopy / Non mangrove).
        path_str = str(model_path).replace("\\", "/")
        if "latest_mangrove_only" in path_str:
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        elif "custom_mangrove_model" in path_str or "MangroVision_New/try_model" in path_str:
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        else:
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        # Ask Detectron2 for candidates down to the model's evaluation threshold,
        # then apply the user-facing confidence gate ourselves. This lets the
        # analysis report how many candidates were below the active threshold.
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = min(0.50, float(self.confidence_threshold))
        cfg.MODEL.DEVICE = self.device

        # Keep more proposals in dense canopies.
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000
        cfg.MODEL.RPN.NMS_THRESH = 0.6
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000
        cfg.INPUT.FORMAT = "BGR"
         
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        self.model_path = str(model_path)
        self.model_name = selected_model_name or Path(model_path).name
        
        print(f"✅ Detectree2 Model Loaded!")
        return self.predictor
    
    def _detect_vegetation_hsv(self, image: np.ndarray) -> np.ndarray:
        """
        Detect all vegetation using HSV color space (FAST pre-filter)
        
        Args:
            image: BGR image
            
        Returns:
            Binary mask of vegetation areas
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Range 1: yellow-green to pure green.
        lower_green1 = np.array([25, 30, 20])
        upper_green1 = np.array([85, 255, 255])
        mask1 = cv2.inRange(hsv, lower_green1, upper_green1)

        # Range 2: blue-green canopy tones in shadow.
        lower_green2 = np.array([85, 20, 20])
        upper_green2 = np.array([100, 255, 255])
        mask2 = cv2.inRange(hsv, lower_green2, upper_green2)

        vegetation_mask = cv2.bitwise_or(mask1, mask2)

        kernel = np.ones((5, 5), np.uint8)
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, kernel)
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel)
        
        return vegetation_mask

    def _detect_strict_canopy_green_hsv(self, image: np.ndarray) -> np.ndarray:
        """Return a stricter green vegetation mask for validating AI crowns."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]

        bgr_f = image.astype(np.float32)
        b = bgr_f[:, :, 0]
        g = bgr_f[:, :, 1]
        r = bgr_f[:, :, 2]
        excess_green = (2.0 * g) - r - b
        green_dominance = (g > (r + 8.0)) & (g > (b + 6.0))

        green_hue = (hue >= 24) & (hue <= 96) & (sat >= 35) & (val >= 20)
        yellow_green = (hue >= 18) & (hue < 32) & (sat >= 55) & (val >= 35)
        shadow_green = (hue >= 30) & (hue <= 100) & (sat >= 28) & (val >= 12) & (val <= 135)

        strength = (excess_green > 12.0) | green_dominance
        strict_mask = ((green_hue | yellow_green | shadow_green) & strength).astype(np.uint8) * 255

        kernel = np.ones((3, 3), np.uint8)
        strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        strict_mask = cv2.morphologyEx(strict_mask, cv2.MORPH_CLOSE, kernel, iterations=1)
        return strict_mask

    @staticmethod
    def _mask_to_polygons(mask: np.ndarray, min_area_px: float) -> List[Polygon]:
        """Convert a binary mask to polygons using a pixel-area cutoff."""
        polygons: List[Polygon] = []
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            if len(contour) < 3 or cv2.contourArea(contour) < min_area_px:
                continue
            # Large merged canopy components have long perimeters. Using a
            # perimeter-scaled epsilon without a cap turns natural mask edges
            # into coarse triangles, which then creates false canopy wedges.
            epsilon = min(1.0, 0.001 * cv2.arcLength(contour, True))
            approx = cv2.approxPolyDP(contour, epsilon, True)
            points = approx.reshape(-1, 2)
            if len(points) < 3:
                continue
            try:
                poly = Polygon(points)
                if not poly.is_valid:
                    poly = poly.buffer(0)
                if isinstance(poly, Polygon) and poly.is_valid and not poly.is_empty:
                    polygons.append(poly)
            except Exception:
                continue
        return polygons

    @staticmethod
    def _polygons_to_mask(polygons: List[Polygon], shape: Tuple[int, int]) -> np.ndarray:
        """Rasterize polygons into a binary mask."""
        h, w = shape
        mask = np.zeros((h, w), dtype=np.uint8)
        for poly in polygons:
            try:
                if poly is None or poly.is_empty or poly.exterior is None:
                    continue
                coords = np.array(poly.exterior.coords, dtype=np.int32)
                cv2.fillPoly(mask, [coords], 255)
            except Exception:
                continue
        return mask

    @staticmethod
    def _metric_kernel_size(distance_m: float, gsd: Optional[float], fallback_px: int = 31) -> int:
        """Convert a metric morphology distance to a bounded odd kernel size."""
        if gsd is not None and gsd > 0 and distance_m > 0:
            size = int(round(distance_m / gsd))
        else:
            size = fallback_px if distance_m > 0 else 0
        if size <= 0:
            return 0
        size = max(3, min(size, 151))
        if size % 2 == 0:
            size += 1
        return size

    def _merge_fragmented_canopy_mask(
        self,
        instance_mask: np.ndarray,
        strict_green_mask: Optional[np.ndarray],
        gsd: Optional[float],
        min_area_px: float,
    ) -> Tuple[List[Polygon], np.ndarray, Dict[str, Any]]:
        """Merge nearby AI fragments into canopy coverage components."""
        merge_enabled = bool(self.runtime_tuning.get("merge_canopy_fragments", True))
        if not merge_enabled or int(np.count_nonzero(instance_mask)) == 0:
            return (
                self._mask_to_polygons(instance_mask, min_area_px),
                instance_mask,
                {
                    "merge_canopy_fragments": merge_enabled,
                    "canopy_merge_gap_m": 0.0,
                    "canopy_hsv_expansion_m": 0.0,
                    "canopy_merge_added_pixels": 0,
                },
            )

        merge_gap_m = float(self.runtime_tuning.get("canopy_merge_gap_m", 0.20))
        hsv_expansion_m = float(self.runtime_tuning.get("canopy_hsv_expansion_m", 0.10))
        merge_kernel_size = self._metric_kernel_size(merge_gap_m, gsd)
        expansion_kernel_size = self._metric_kernel_size(hsv_expansion_m, gsd, fallback_px=17)

        merged_mask = instance_mask.copy()
        before_pixels = int(np.count_nonzero(merged_mask))

        if (
            strict_green_mask is not None
            and strict_green_mask.shape == merged_mask.shape
            and expansion_kernel_size > 0
        ):
            expansion_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (expansion_kernel_size, expansion_kernel_size),
            )
            nearby_ai = cv2.dilate(merged_mask, expansion_kernel, iterations=1)
            green_near_ai = cv2.bitwise_and(strict_green_mask, nearby_ai)
            merged_mask = cv2.bitwise_or(merged_mask, green_near_ai)

        if merge_kernel_size > 0:
            merge_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (merge_kernel_size, merge_kernel_size),
            )
            merged_mask = cv2.morphologyEx(merged_mask, cv2.MORPH_CLOSE, merge_kernel, iterations=1)

        coverage_polygons = self._mask_to_polygons(merged_mask, min_area_px)
        if not coverage_polygons:
            coverage_polygons = self._mask_to_polygons(instance_mask, min_area_px)
            merged_mask = instance_mask

        after_pixels = int(np.count_nonzero(merged_mask))
        return (
            coverage_polygons,
            merged_mask,
            {
                "merge_canopy_fragments": merge_enabled,
                "canopy_merge_gap_m": merge_gap_m,
                "canopy_hsv_expansion_m": hsv_expansion_m,
                "canopy_merge_kernel_px": merge_kernel_size,
                "canopy_hsv_expansion_kernel_px": expansion_kernel_size,
                "canopy_merge_added_pixels": max(0, after_pixels - before_pixels),
            },
        )
    
    def detect_from_image(self, 
                         image: np.ndarray,
                         gsd: float = None,
                         tile_size: int = 512,
                         overlap: float = 0.25,
                         progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None) -> Tuple[List[Polygon], np.ndarray, Dict]:
        """
        Detect tree crowns using detectree2 tiled inference.
        Uses official setup_cfg for model config and clean_crowns for overlap cleanup.
        
        Args:
            image: Input BGR image
            gsd: Ground Sample Distance (optional, for compatibility)
            tile_size: Size of tiles for detection (default 512px)
            overlap: Overlap fraction between tiles (default 0.25 = 25%)
            
        Returns:
            Tuple of (polygons, mask, metadata)
        """
        if self.predictor is None:
            self.setup_model()

        def _emit_progress(event: str, payload: Dict[str, Any]) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(event, payload)
            except Exception:
                # Progress reporting must never interrupt detection.
                pass
        
        tile_size = int(self.runtime_tuning.get("tile_size", tile_size))
        overlap = float(self.runtime_tuning.get("tile_overlap", overlap))
        overlap = max(0.0, min(overlap, 0.6))
        h, w = image.shape[:2]
        print(f"🌳 Running detectree2 on {w}x{h} image...")
        
        # PHASE 1: HSV Pre-filter - Find all vegetation (FAST)
        vegetation_mask = self._detect_vegetation_hsv(image)
        veg_pixels = np.count_nonzero(vegetation_mask)
        veg_percent = 100 * veg_pixels / (h * w)
        print(f"   Phase 1: HSV found {veg_percent:.1f}% vegetation coverage")
        
        veg_tile_threshold = float(self.runtime_tuning.get("tile_veg_threshold", 0.002))

        stride = max(1, int(tile_size * (1.0 - overlap)))
        tiles: List[Tuple[int, int, int, int]] = []
        tiles_checked = 0

        # PHASE 2: Select only tiles with sufficient vegetation coverage.
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                tiles_checked += 1

                if veg_tile_threshold <= 0.0:
                    tiles.append((x, y, x_end, y_end))
                    continue

                tile_veg_mask = vegetation_mask[y:y_end, x:x_end]
                veg_ratio = np.count_nonzero(tile_veg_mask) / ((x_end - x) * (y_end - y))
                if veg_ratio >= veg_tile_threshold:
                    tiles.append((x, y, x_end, y_end))

        # Safety fallback: if pre-filter rejects everything, scan all tiles.
        if not tiles:
            print("   [WARN] Vegetation pre-filter skipped all tiles; using full image tiling fallback")
            for y in range(0, h, stride):
                for x in range(0, w, stride):
                    x_end = min(x + tile_size, w)
                    y_end = min(y + tile_size, h)
                    tiles.append((x, y, x_end, y_end))

        skipped_tiles = max(0, tiles_checked - len(tiles))
        print(f"   Phase 2: Processing {len(tiles)} tiles with vegetation (skipped {skipped_tiles} empty)")
        print(f"   Tile size: {tile_size}px, overlap: {int(overlap*100)}%")
        _emit_progress(
            "tile_setup",
            {
                "total_tiles": len(tiles),
                "checked_tiles": int(tiles_checked),
                "skipped_tiles": int(skipped_tiles),
                "tile_size": int(tile_size),
                "overlap": float(overlap),
            },
        )
        
        # Run detection on each tile
        all_instances = []
        total_model_detections = 0
        below_confidence_detections = 0
        rejected_non_green = 0
        rejected_too_small = 0
        rejected_too_large = 0
        rejected_invalid_geometry = 0
        min_crown_m2 = float(self.runtime_tuning.get("min_crown_m2", 0.05))
        max_crown_m2 = float(self.runtime_tuning.get("max_crown_m2", 250.0))
        strict_canopy_hsv = bool(self.runtime_tuning.get("strict_canopy_hsv", True))
        detection_veg_min_ratio = float(self.runtime_tuning.get("detection_veg_min_ratio", 0.03))
        strict_green_mask = self._detect_strict_canopy_green_hsv(image) if strict_canopy_hsv else None
        gsd_used = float(gsd) if (gsd is not None and gsd > 0) else None
        if gsd_used is not None:
            min_area_px = max(20.0, min_crown_m2 / (gsd_used ** 2))
            max_area_px = max(1000.0, max_crown_m2 / (gsd_used ** 2))
        else:
            min_area_px = 80.0
            max_area_px = 120000.0
         
        for tile_idx, (x1, y1, x2, y2) in enumerate(tiles):
            current_tile = tile_idx + 1
            _emit_progress(
                "tile_progress",
                {
                    "current_tile": int(current_tile),
                    "total_tiles": len(tiles),
                },
            )
            if tile_idx % 10 == 0:
                print(f"   Tile {current_tile}/{len(tiles)}...")

            tile = image[y1:y2, x1:x2]

            with torch.no_grad():
                outputs = self.predictor(tile)

            instances = outputs["instances"].to("cpu")
            scores = instances.scores.numpy()
            masks = instances.pred_masks.numpy()

            # Store each detection with global coordinates and confidence.
            for i in range(len(scores)):
                total_model_detections += 1
                if scores[i] < self.confidence_threshold:
                    below_confidence_detections += 1
                    continue

                mask = masks[i].astype(np.uint8)
                if int(np.count_nonzero(mask)) <= 0:
                    continue

                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                for contour in contours:
                    local_detection_mask = np.zeros(mask.shape, dtype=np.uint8)
                    cv2.fillPoly(local_detection_mask, [contour], 1)
                    detection_pixels = int(np.count_nonzero(local_detection_mask))
                    if detection_pixels <= 0:
                        continue

                    if strict_canopy_hsv:
                        strict_tile = strict_green_mask[y1:y2, x1:x2]
                        green_pixels = int(np.count_nonzero((local_detection_mask > 0) & (strict_tile > 0)))
                        vegetation_ratio = green_pixels / detection_pixels
                        if vegetation_ratio < detection_veg_min_ratio:
                            rejected_non_green += 1
                            continue
                    else:
                        vegetation_ratio = 1.0

                    # Convert to global coordinates
                    contour_global = contour.copy()
                    contour_global[:, 0, 0] += x1
                    contour_global[:, 0, 1] += y1

                    # Convert to polygon
                    if len(contour_global) >= 3:
                        try:
                            points = contour_global.reshape(-1, 2)
                            poly = Polygon(points)
                            if not poly.is_valid:
                                poly = poly.buffer(0)
                            if not isinstance(poly, Polygon) or poly.is_empty:
                                rejected_invalid_geometry += 1
                                continue
                            if poly.area < min_area_px:
                                rejected_too_small += 1
                                continue
                            if poly.area > max_area_px:
                                rejected_too_large += 1
                                continue
                            all_instances.append({
                                'polygon': poly,
                                'score': scores[i],
                                'contour': contour_global,
                                'vegetation_ratio': vegetation_ratio,
                            })
                        except Exception:
                            rejected_invalid_geometry += 1
                            continue

            # Emit a post-inference tile_done event so subscribers can report
            # "tile N processed" rather than "tile N starting".
            _emit_progress(
                "tile_done",
                {
                    "current_tile": int(current_tile),
                    "total_tiles": len(tiles),
                    "tile_detections": int(len(scores)),
                },
            )

        print(f"   Found {len(all_instances)} detections")
        if below_confidence_detections:
            print(f"   Ignored {below_confidence_detections} candidates below confidence {self.confidence_threshold:.2f}")
        if rejected_non_green:
            print(f"   Rejected {rejected_non_green} non-green AI detections")
        if rejected_too_large:
            print(f"   Rejected {rejected_too_large} detections above max crown area {max_crown_m2:.1f} m2")

        # High-recall default: skip aggressive clean_crowns unless explicitly enabled.
        use_clean_crowns = bool(self.runtime_tuning.get("use_clean_crowns", False))
        if use_clean_crowns:
            final_polygons = self._clean_with_detectree2_outputs(all_instances)
            if not final_polygons:
                final_polygons = self._nms_polygons(
                    all_instances,
                    iou_threshold=float(self.runtime_tuning.get("fallback_nms_iou", 0.90))
                )
        else:
            final_polygons = self._nms_polygons(
                all_instances,
                iou_threshold=float(self.runtime_tuning.get("fallback_nms_iou", 0.90))
            )

        instance_mask = self._polygons_to_mask(final_polygons, (h, w))
        coverage_polygons, combined_mask, merge_metadata = self._merge_fragmented_canopy_mask(
            instance_mask=instance_mask,
            strict_green_mask=strict_green_mask,
            gsd=gsd_used,
            min_area_px=min_area_px,
        )

        print(f"   ✅ {len(final_polygons)} AI instances after cleanup")
        if merge_metadata.get("merge_canopy_fragments"):
            print(
                f"   Coverage merge produced {len(coverage_polygons)} canopy components "
                f"(+{merge_metadata.get('canopy_merge_added_pixels', 0)} px)"
            )
        _emit_progress(
            "tile_complete",
            {
                "total_tiles": len(tiles),
                "raw_detections": int(len(all_instances)),
                "final_trees": int(len(coverage_polygons)),
                "instance_trees": int(len(final_polygons)),
            },
        )

        if strict_canopy_hsv:
            print(f"   HSV validation retained {len(final_polygons)} AI canopy polygons")

        instance_pixels = int(np.count_nonzero(instance_mask))
        coverage_pixels = int(np.count_nonzero(combined_mask))
        area_factor = (gsd_used ** 2) if gsd_used is not None else 0.0
        
        metadata = {
            'num_tiles': len(tiles),
            'num_tiles_checked': int(tiles_checked),
            'num_tiles_processed': len(tiles),
            'num_tiles_skipped': int(skipped_tiles),
            'tile_size': tile_size,
            'overlap': overlap,
            'tile_veg_threshold': float(veg_tile_threshold),
            'model_candidate_detections': int(total_model_detections),
            'below_confidence_detections': int(below_confidence_detections),
            'raw_detections': len(all_instances),
            'total_ai_detections': len(all_instances),
            'instance_tree_count': len(final_polygons),
            'final_trees': len(coverage_polygons),
            'coverage_component_count': len(coverage_polygons),
            'num_detected_canopies': len(coverage_polygons),
            'rejected_non_green_detections': int(rejected_non_green),
            'rejected_too_small_detections': int(rejected_too_small),
            'rejected_too_large_detections': int(rejected_too_large),
            'rejected_invalid_geometry': int(rejected_invalid_geometry),
            'strict_canopy_hsv': strict_canopy_hsv,
            'detection_veg_min_ratio': detection_veg_min_ratio,
            'gsd_used': gsd_used,
            'min_crown_m2': min_crown_m2,
            'max_crown_m2': max_crown_m2,
            'instance_canopy_area_m2': instance_pixels * area_factor,
            'coverage_canopy_area_m2': coverage_pixels * area_factor,
            'model_path': self.model_path,
            'model_name': self.model_name,
            'cleanup_iou': float(self.runtime_tuning.get("cleanup_iou", 0.75)),
            'fallback_nms_iou': float(self.runtime_tuning.get("fallback_nms_iou", 0.90)),
            'use_clean_crowns': use_clean_crowns,
            'detection_method': 'detectree2_official'
        }
        metadata.update(merge_metadata)
         
        return coverage_polygons, combined_mask, metadata

    def _clean_with_detectree2_outputs(self, instances: List[Dict]) -> List[Polygon]:
        """Apply detectree2 clean_crowns overlap-cleaning on polygon outputs."""
        if not instances:
            return []
        try:
            crowns = gpd.GeoDataFrame(
                {
                    "geometry": [i["polygon"] for i in instances],
                    "Confidence_score": [float(i["score"]) for i in instances],
                },
                geometry="geometry",
            )
            cleanup_iou = float(self.runtime_tuning.get("cleanup_iou", 0.75))
            cleaned = clean_crowns(crowns, cleanup_iou, confidence=self.confidence_threshold)
            if cleaned is None or cleaned.empty:
                return []
            polygons: List[Polygon] = []
            for geom in cleaned.geometry:
                if geom is None or geom.is_empty:
                    continue
                if isinstance(geom, Polygon):
                    polygons.append(geom)
                else:
                    try:
                        polygons.extend([g for g in geom.geoms if isinstance(g, Polygon)])
                    except Exception:
                        continue
            return polygons
        except Exception as e:
            print(f"   ⚠️ detectree2 clean_crowns unavailable/failed: {e}")
            return []
    
    def _nms_polygons(self, instances: List[Dict], iou_threshold: float = 0.5) -> List[Polygon]:
        """
        Non-maximum suppression for polygons based on IoU
        """
        if len(instances) == 0:
            return []
        
        # Sort by confidence score
        instances = sorted(instances, key=lambda x: x['score'], reverse=True)
        
        keep = []
        
        for instance in instances:
            poly = instance['polygon']
            
            # Check IoU against all kept polygons
            should_keep = True
            for kept_poly in keep:
                try:
                    intersection = poly.intersection(kept_poly).area
                    union = poly.union(kept_poly).area
                    iou = intersection / union if union > 0 else 0
                    
                    if iou > iou_threshold:
                        should_keep = False
                        break
                except Exception:
                    continue
            
            if should_keep:
                keep.append(poly)
        
        return keep
