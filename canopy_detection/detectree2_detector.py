"""
MangroVision - AI-Powered Tree Canopy Structure Detector
Uses Detectron2 Mask R-CNN for canopy detection with shape-based filtering
Focuses on detecting tree crown STRUCTURES regardless of species
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional, Dict, Callable, Any
import torch
from shapely.geometry import Polygon
import gdown

# Detectron2 imports
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectron2 import model_zoo as d2_model_zoo


class Detectree2Detector:
    """
    AI-powered tree canopy detector using structural analysis
    Detects tree crowns based on shape, size, and texture patterns
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 device: str = 'cpu',
                 model_name: str = 'benchmark'):
        """
        Initialize tree canopy structure detector
        
        Args:
            confidence_threshold: Minimum confidence score (0-1)
            device: 'cpu' or 'cuda'
            model_name: Model identifier (for future use)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model_name = model_name
        self.predictor = None
        self.cfg = None
        self.model_path = None
        self.loaded_model_name = None
        self.runtime_tuning = {
            "tile_veg_threshold": 0.002,
            "min_crown_m2": 0.05,
            "max_crown_m2": 50.0,
            "separate_threshold_m2": 10.0,
            "overlap_validation_ratio": 0.15,
        }
        
        print(f"🌳 Initializing AI Tree Canopy Structure Detector")
        print(f"   Using Detectron2 Mask R-CNN for segmentation")
        print(f"   Device: {device}")
        print(f"   Confidence threshold: {confidence_threshold}")

    def set_runtime_tuning(
        self,
        tile_veg_threshold: float = None,
        min_crown_m2: float = None,
        max_crown_m2: float = None,
        separate_threshold_m2: float = None,
        overlap_validation_ratio: float = None,
    ):
        """Update non-threshold inference tuning parameters."""
        if tile_veg_threshold is not None:
            self.runtime_tuning["tile_veg_threshold"] = max(0.0, min(float(tile_veg_threshold), 1.0))
        if min_crown_m2 is not None:
            self.runtime_tuning["min_crown_m2"] = max(0.01, float(min_crown_m2))
        if max_crown_m2 is not None:
            self.runtime_tuning["max_crown_m2"] = max(self.runtime_tuning["min_crown_m2"], float(max_crown_m2))
        if separate_threshold_m2 is not None:
            self.runtime_tuning["separate_threshold_m2"] = max(0.05, float(separate_threshold_m2))
        if overlap_validation_ratio is not None:
            self.runtime_tuning["overlap_validation_ratio"] = max(0.0, min(float(overlap_validation_ratio), 1.0))
        
    def setup_model(self):
        """
        Setup Mask R-CNN with custom-trained mangrove model
        Priority: Custom Model > Detectree2 Tropical > COCO Base
        """
        print(f"⚙️ Setting up AI mangrove canopy detector...")
        
        model_dir = Path(__file__).parent.parent / 'models'
        
        # PRIORITY 1: Check for custom-trained Roboflow model (BEST)
        custom_model_path = model_dir / 'custom_mangrove_model' / 'model_final.pth'
        
        if custom_model_path.exists():
            # Load custom model trained on Roboflow dataset (Mangrove + Bungalon)
            print(f"   🎯 Loading CUSTOM-TRAINED Model")
            print(f"   📍 Trained on: YOUR Roboflow dataset (240 images)")
            print(f"   🌳 Base: detectree2 tropical forest model (transfer learning)")
            print(f"   🏆 Classes: Mangrove-Canopy, Bungalon Canopy")
            print(f"   📊 Performance: AP50=59.5%, AP=38.3% (Bungalon)")
            
            cfg = get_cfg()
            # Use R-101 architecture to match training configuration
            cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"  # R-101 (matches training)
            ))
            
            # Configure for 2 mangrove classes
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2  # Mangrove-Canopy, Bungalon Canopy
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
            cfg.MODEL.DEVICE = self.device
            
            # Optimize for dense canopy - INCREASE limits for more detections
            cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000  # Increased from 3000
            cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000  # Increased from 1000
            
            # Lower NMS threshold to keep more overlapping detections
            cfg.MODEL.RPN.NMS_THRESH = 0.6  # Default 0.7, lower = fewer removals
            
            # Load custom weights
            cfg.MODEL.WEIGHTS = str(custom_model_path)
            self.model_path = str(custom_model_path)
            self.loaded_model_name = "Custom Mangrove Model"
            
        else:
            # PRIORITY 2: Try detectree2 tropical models
            print(f"   ⚠️ Custom model not found at: {custom_model_path}")
            print(f"   📥 Falling back to detectree2 tropical models...")
            
            cfg = get_cfg()
            cfg.merge_from_file(model_zoo.get_config_file(
                "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"  # Base architecture
            ))
            
            # Configure for tree detection (1 class: tree crown)
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  
            cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
            cfg.MODEL.DEVICE = self.device
            
            # Optimize for dense canopy
            cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
            cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000
            cfg.MODEL.RPN.NMS_THRESH = 0.6
            
            # Try latest model-garden checkpoints first, then legacy tropical models.
            tropical_models = []
            latest_model_garden = sorted(model_dir.glob("250312*.pth"))
            tropical_models.extend([(p, f"Model Garden ({p.name})") for p in latest_model_garden])
            tropical_models.extend([
                (model_dir / '230103_randresize_full.pth', 'Optimized Tropical (Zenodo 230103)'),
                (model_dir / '230717_tropical_base.pth', 'Base Tropical (230717)')
            ])
            
            model_loaded = False
            for model_path, model_name in tropical_models:
                if model_path.exists():
                    print(f"   ✅ Loading {model_name}")
                    print(f"   📍 Trained on: Danum, Sepilok, Paracou tropical forests")
                    cfg.MODEL.WEIGHTS = str(model_path)
                    self.model_path = str(model_path)
                    self.loaded_model_name = model_name
                    model_loaded = True
                    break
            
            # PRIORITY 3: Fall back to COCO
            if not model_loaded:
                print(f"   ⚠️ No tropical model found in: {model_dir}")
                print(f"   ⚠️ Using COCO base weights (less accurate)")
                cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                    "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
                )
                self.model_path = cfg.MODEL.WEIGHTS
                self.loaded_model_name = "COCO Base"
        
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        
        print(f"✅ Detectree2 Model Loaded!")
        return self.predictor
    
    def analyze_canopy_structure(self, mask: np.ndarray, image: np.ndarray) -> dict:
        """
        Analyze if a segmented region matches tree canopy characteristics
        
        Args:
            mask: Binary mask of detected region
            image: Original image for texture/color analysis
            
        Returns:
            Dict with analysis results and scores
        """
        h, w = mask.shape
        
        # Calculate shape properties
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if len(contours) == 0:
            return {'is_canopy': False, 'score': 0.0}
        
        contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(contour)
        perimeter = cv2.arcLength(contour, True)
        
        # 1. SIZE FILTER: Tree crowns ~0.5m² to 50m² (relaxed range)
        min_area_px = 100   # ~0.13 m² at 0.0113 GSD
        max_area_px = 100000  # ~127 m² 
        if area < min_area_px or area > max_area_px:
            return {'is_canopy': False, 'score': 0.0, 'reason': 'size', 'area': area}
        
        # 2. SHAPE FILTER: Compactness (relaxed - trees can be irregular)
        compactness = (4 * np.pi * area) / (perimeter ** 2) if perimeter > 0 else 0
        # Very lenient - accept even elongated shapes
        if compactness < 0.15:  # Was 0.3, now much more lenient
            return {'is_canopy': False, 'score': 0.0, 'reason': 'elongated', 'compactness': compactness}
        
        # 3. TEXTURE FILTER: Skip for now - too restrictive
        # Trees have varied textures, don't filter on this
        
        # 4. COLOR FILTER: Should have green component (relaxed threshold)
        # Extract masked region from image
        masked_region = cv2.bitwise_and(image, image, mask=mask)
        hsv_region = cv2.cvtColor(masked_region, cv2.COLOR_BGR2HSV)
        hsv_region = hsv_region[mask > 0]
        
        if len(hsv_region) == 0:
            return {'is_canopy': False, 'score': 0.0}
        
        # Detect green hues (wide range for all vegetation)
        green_pixels = np.sum((hsv_region[:, 0] >= 20) & (hsv_region[:, 0] <= 100))
        green_ratio = green_pixels / len(hsv_region) if len(hsv_region) > 0 else 0
        
        # Very lenient - just needs SOME green (5%)
        if green_ratio < 0.05:
            return {'is_canopy': False, 'score': 0.0, 'reason': 'color', 'green_ratio': green_ratio}
        
        # Calculate overall canopy score (simplified)
        canopy_score = (
            (compactness * 0.3) +     # Shape
            (green_ratio * 0.7)       # Color is most important
        )
        
        return {
            'is_canopy': canopy_score > 0.2,  # Lower threshold
            'score': canopy_score,
            'compactness': compactness,
            'green_ratio': green_ratio,
            'area': area
        }
    
    def _separate_merged_crowns(self, mask: np.ndarray) -> List[np.ndarray]:
        """
        Attempt to separate merged tree crowns using watershed segmentation
        
        Args:
            mask: Binary mask potentially containing multiple merged crowns
            
        Returns:
            List of separated binary masks (or just [original] if separation fails)
        """
        try:
            from scipy import ndimage
            from skimage.segmentation import watershed
            from skimage.feature import peak_local_max
            
            # Distance transform: find peaks (tree centers)
            distance = ndimage.distance_transform_edt(mask)
            
            # Find local maxima (potential tree centers)
            # min_distance ensures trees are at least 20 pixels apart (~0.2m)
            local_peaks = peak_local_max(distance, min_distance=20, labels=mask > 0)
            
            if len(local_peaks) <= 1:
                # Only one peak found, can't separate
                return [mask]
            
            # Create markers for watershed
            markers = np.zeros_like(mask, dtype=int)
            for idx, peak in enumerate(local_peaks):
                markers[peak[0], peak[1]] = idx + 1
            
            # Apply watershed to separate crowns
            labels = watershed(-distance, markers, mask=mask > 0)
            
            # Extract individual masks
            separated_masks = []
            for label_id in range(1, labels.max() + 1):
                separated_mask = (labels == label_id).astype(np.uint8) * 255
                if np.count_nonzero(separated_mask) > 100:  # Minimum size
                    separated_masks.append(separated_mask)
            
            return separated_masks if len(separated_masks) > 1 else [mask]
            
        except (ImportError, Exception) as e:
            # If scipy/skimage not available or watershed fails, return original
            return [mask]
    
    def detect_from_image(
        self,
        image: np.ndarray,
        gsd: Optional[float] = None,
        progress_callback: Optional[Callable[[str, Dict[str, Any]], None]] = None,
    ) -> Tuple[List[Polygon], np.ndarray, dict]:
        """
        Detect tree crowns using detectree2 pre-trained model with tiled inference.
        
        Detectree2 was trained on small image tiles (~400x400px), so we tile
        the input image, run AI on each tile, then merge results.
        
        Args:
            image: Input BGR image (OpenCV format)
            
        Returns:
            Tuple of:
                - List of canopy polygons (Shapely Polygon objects)
                - Binary mask of all detected canopies
                - Detection metadata
        """
        if self.predictor is None:
            self.setup_model()

        def _emit_progress(event: str, payload: Dict[str, Any]) -> None:
            if progress_callback is None:
                return
            try:
                progress_callback(event, payload)
            except Exception:
                # Progress reporting should never break inference.
                pass
        
        h, w = image.shape[:2]
        print(f"🌳 Running detectree2 tree crown detection on {w}x{h} image...")
        
        # PHASE 1: HSV Pre-filter - Identify all vegetation areas
        print(f"   Phase 1: HSV vegetation detection...")
        vegetation_mask = self._detect_vegetation_hsv(image)
        veg_pixels = np.count_nonzero(vegetation_mask)
        veg_percent = 100 * veg_pixels / (h * w)
        print(f"   ✓ Found {veg_percent:.1f}% vegetation coverage")
        
        # PHASE 2: Detectree2 AI - Separate vegetation into individual crowns
        print(f"   Phase 2: AI crown structure detection...")
        
        # Tile parameters - OPTIMIZED for detectree2 training size
        # Model was trained on ~400px tiles, use similar size for best results
        tile_size = 512  # Reduced from 800 to match training data better
        overlap = 128  # 25% overlap (128/512) for good edge detection
        stride = tile_size - overlap
        
        # Generate tiles - keep nearly all non-empty vegetation tiles for better recall
        tiles = []
        tiles_checked = 0
        veg_tile_threshold = float(self.runtime_tuning.get("tile_veg_threshold", 0.002))
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                # Ensure tile is at least 200px
                if (x_end - x) < 200 or (y_end - y) < 200:
                    continue
                
                tiles_checked += 1
                
                # OPTIMIZATION: Check if tile has vegetation before processing
                tile_veg_mask = vegetation_mask[y:y_end, x:x_end]
                veg_pixels = np.count_nonzero(tile_veg_mask)
                tile_area = (x_end - x) * (y_end - y)
                veg_ratio = veg_pixels / tile_area if tile_area > 0 else 0
                
                # Very light prefilter to avoid missing sparse canopy clusters
                if veg_ratio >= veg_tile_threshold:
                    tiles.append((x, y, x_end, y_end))
        
        skipped_tiles = tiles_checked - len(tiles)
        print(f"   Tiling: {len(tiles)} tiles with vegetation (skipped {skipped_tiles} empty tiles)")
        print(f"   Tile size: {tile_size}px, overlap: {overlap}px")
        _emit_progress(
            "tile_setup",
            {
                "total_tiles": len(tiles),
                "checked_tiles": int(tiles_checked),
                "skipped_tiles": int(skipped_tiles),
                "tile_size": int(tile_size),
                "overlap": int(overlap),
            },
        )
        
        # Run AI on each tile - collect individual detections
        all_detections = []  # Store (polygon, score) for each detection
        total_detections = 0
        
        # GSD-based size thresholds (use detected GSD when available)
        gsd_used = gsd if (gsd is not None and gsd > 0) else 0.0113
        MIN_CROWN_M2 = float(self.runtime_tuning.get("min_crown_m2", 0.05))
        MAX_CROWN_M2 = float(self.runtime_tuning.get("max_crown_m2", 50.0))
        SEPARATE_THRESHOLD_M2 = float(self.runtime_tuning.get("separate_threshold_m2", 10.0))
        
        min_area_px = int(MIN_CROWN_M2 / (gsd_used ** 2))
        max_area_px = int(MAX_CROWN_M2 / (gsd_used ** 2))
        separate_px = int(SEPARATE_THRESHOLD_M2 / (gsd_used ** 2))
        
        for tile_idx, (x1, y1, x2, y2) in enumerate(tiles):
            current_tile = tile_idx + 1
            _emit_progress(
                "tile_progress",
                {
                    "current_tile": int(current_tile),
                    "total_tiles": len(tiles),
                },
            )
            if tile_idx % 5 == 0:
                print(f"   Processing tile {current_tile}/{len(tiles)}...")
            
            tile = image[y1:y2, x1:x2]
            
            with torch.no_grad():
                outputs = self.predictor(tile)
            
            instances = outputs["instances"].to("cpu")
            pred_masks = instances.pred_masks.numpy()
            scores = instances.scores.numpy()
            
            for i in range(len(scores)):
                if scores[i] < self.confidence_threshold:
                    continue
                
                total_detections += 1
                mask = pred_masks[i].astype(np.uint8) * 255
                
                # Size filter
                area = np.count_nonzero(mask)
                if area < 100 or area > max_area_px * 3:  # Filter extreme sizes
                    continue
                
                # Extract contours from this single detection
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    # Offset contour to full image coordinates
                    contour_offset = contour.copy()
                    contour_offset[:, 0, 0] += x1
                    contour_offset[:, 0, 1] += y1
                    
                    c_area = cv2.contourArea(contour_offset)
                    if c_area < min_area_px:
                        continue
                    
                    # Try watershed separation for large detections
                    if c_area > separate_px:
                        # Create mask at original position for watershed
                        full_mask = np.zeros((h, w), dtype=np.uint8)
                        cv2.drawContours(full_mask, [contour_offset], -1, 255, -1)
                        separated = self._separate_merged_crowns(full_mask)
                        
                        if len(separated) > 1:
                            for sep_mask in separated:
                                sep_contours, _ = cv2.findContours(sep_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                                for sc in sep_contours:
                                    sa = cv2.contourArea(sc)
                                    if min_area_px < sa < max_area_px:
                                        poly = self._contour_to_polygon_obj(sc)
                                        if poly:
                                            all_detections.append((poly, scores[i], sc))
                            continue
                    
                    # Normal detection - convert to polygon
                    if c_area < max_area_px:
                        poly = self._contour_to_polygon_obj(contour_offset)
                        if poly:
                            all_detections.append((poly, scores[i], contour_offset))
        
        print(f"   Found {total_detections} AI detections → {len(all_detections)} individual crowns")
        
        # PHASE 3: Vegetation Validation - Keep only crowns that overlap with HSV vegetation
        print(f"   Phase 3: Validating crowns against vegetation mask...")
        validated_detections = []
        filtered_non_vegetation = 0
        
        for poly, score, contour in all_detections:
            # Create mask for this crown
            crown_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.drawContours(crown_mask, [contour], -1, 255, -1)
            
            # Check overlap with vegetation mask
            overlap = cv2.bitwise_and(crown_mask, vegetation_mask)
            overlap_ratio = np.count_nonzero(overlap) / np.count_nonzero(crown_mask)
            
            # Require at least 25% overlap with vegetation (improves recall on sparse crowns)
            # Filters out non-vegetation false positives
            if overlap_ratio >= float(self.runtime_tuning.get("overlap_validation_ratio", 0.15)):
                validated_detections.append((poly, score))
            else:
                filtered_non_vegetation += 1
        
        print(f"   ✓ Validated: {len(validated_detections)} crowns, filtered: {filtered_non_vegetation} non-vegetation")
        
        # DISABLE NMS - Accept all validated detections
        # Dense canopy means trees touch each other, IoU-based NMS removes real trees
        # Visual overlap from tile edges is acceptable trade-off for complete coverage
        canopy_polygons = [poly for poly, score in validated_detections]
        kept_canopies = len(canopy_polygons)
        
        print(f"   ℹ️ NMS disabled - keeping all validated detections for dense canopy")
        
        # Create combined mask from final polygons
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for poly in canopy_polygons:
            coords = np.array(poly.exterior.coords, dtype=np.int32)
            cv2.fillPoly(combined_mask, [coords], 255)
        
        print(f"✅ Detectree2 Results:")
        print(f"   ✓ {kept_canopies} individual tree crowns detected")
        print(f"   Pipeline: HSV vegetation -> AI crown separation -> Validation")
        _emit_progress(
            "tile_complete",
            {
                "total_tiles": len(tiles),
                "raw_detections": int(total_detections),
                "final_trees": int(kept_canopies),
            },
        )
        
        metadata = {
            'num_tiles': len(tiles),
            'num_tiles_checked': int(tiles_checked),
            'num_tiles_processed': len(tiles),
            'num_tiles_skipped': int(skipped_tiles),
            'total_ai_detections': total_detections,
            'raw_detections': total_detections,
            'num_detected_canopies': kept_canopies,
            'final_trees': kept_canopies,
            'filtered_non_vegetation': filtered_non_vegetation,
            'vegetation_coverage_percent': float(veg_percent),
            'detection_method': 'hsv_guided_detectree2',
            'tile_size': tile_size,
            'overlap': overlap,
            'tile_veg_threshold': float(veg_tile_threshold),
            'gsd_used': float(gsd_used),
            'min_crown_m2': float(MIN_CROWN_M2),
            'max_crown_m2': float(MAX_CROWN_M2),
            'overlap_validation_ratio': float(self.runtime_tuning.get("overlap_validation_ratio", 0.15)),
            'model_path': self.model_path,
            'model_name': self.loaded_model_name,
        }
        
        return canopy_polygons, combined_mask, metadata
    
    def _detect_vegetation_hsv(self, image):
        """
        Detect all vegetation areas using HSV color space
        This creates a coarse vegetation mask that guides detectree2
        
        Args:
            image: BGR image
            
        Returns:
            Binary mask of vegetation areas
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        
        # Define green range for vegetation
        # Range 1: Yellowish-green to pure green (most vegetation)
        lower_green1 = np.array([25, 30, 20])
        upper_green1 = np.array([85, 255, 255])
        mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
        
        # Range 2: Blue-green (some mangroves, shadows)
        lower_green2 = np.array([85, 20, 20])
        upper_green2 = np.array([100, 255, 255])
        mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
        
        # Combine both green ranges
        vegetation_mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, kernel)
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel)
        
        return vegetation_mask
        green_ratio = np.sum(green_mask) / len(hsv_pixels)
        
        # Require at least 30% green pixels to be considered vegetation
        # This filters out boats, buildings, bare ground, etc.
        return green_ratio >= 0.30
    
    def _contour_to_polygon_obj(self, contour):
        """Convert contour to Shapely Polygon object"""
        epsilon = 0.003 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        points = approx.reshape(-1, 2)
        if len(points) >= 3:
            poly = Polygon(points)
            if poly.is_valid and not poly.is_empty:
                return poly
        return None
    
    def _nms_polygons(self, detections, iou_threshold=0.5):
        """
        Non-Maximum Suppression for polygons to remove duplicates from tile overlaps
        
        Args:
            detections: List of (polygon, score) tuples
            iou_threshold: IoU threshold for considering polygons as duplicates
            
        Returns:
            List of kept polygons (deduplicated)
        """
        if len(detections) == 0:
            return []
        
        # Sort by score (highest first)
        detections = sorted(detections, key=lambda x: x[1], reverse=True)
        
        kept = []
        
        for poly, score in detections:
            # Check if this polygon overlaps significantly with any kept polygon
            should_keep = True
            for kept_poly in kept:
                try:
                    intersection = poly.intersection(kept_poly)
                    union = poly.union(kept_poly)
                    iou = intersection.area / union.area if union.area > 0 else 0
                    
                    if iou > iou_threshold:
                        should_keep = False
                        break
                except:
                    # If shapely operations fail, keep it
                    pass
            
            if should_keep:
                kept.append(poly)
        
        return kept
    
    def detect_from_geotiff_samgeo(self, 
                                     geotiff_path: str,
                                     output_geojson: str = None,
                                     tile_width: int = 100,
                                     tile_height: int = 100,
                                     model_path: str = None) -> dict:
        """
        Detect tree crowns from GeoTIFF using samgeo.detectree2 wrapper.
        This is the RECOMMENDED method for accurate detection on orthophotos.
        
        Args:
            geotiff_path: Path to input GeoTIFF file (0.1-0.5m/pixel resolution)
            output_geojson: Path to save output GeoJSON (optional)
            tile_width: Width of tiles for inference (default: 100)
            tile_height: Height of tiles for inference (default: 100)
            model_path: Path to detectree2 model weights (optional)
            
        Returns:
            Dictionary containing:
                - 'geojson_path': Path to output GeoJSON
                - 'tree_count': Number of trees detected
                - 'total_area_m2': Total canopy area in square meters
                - 'metadata': Additional detection metadata
        """
        try:
            from samgeo.detectree2 import TreeCrownDelineator
        except ImportError:
            raise ImportError(
                "segment-geospatial not installed. "
                "Install with: pip install segment-geospatial\n"
                "See INSTALL_SAMGEO_DETECTREE2.md for complete guide"
            )
        
        print(f"🌳 Samgeo Detectree2 - GeoTIFF Tree Crown Detection")
        print(f"   Input: {geotiff_path}")
        print(f"   Tile size: {tile_width}x{tile_height}")
        
        # Determine model path
        if model_path is None:
            model_dir = Path(__file__).parent.parent / 'models'
            # Try new tropical model first, fall back to existing
            tropical_models = [
                model_dir / '230103_randresize_full.pth',
                model_dir / '230717_tropical_base.pth'
            ]
            for model in tropical_models:
                if model.exists():
                    model_path = str(model)
                    print(f"   Using model: {model.name}")
                    break
            
            if model_path is None:
                raise FileNotFoundError(
                    f"No detectree2 model found in {model_dir}\n"
                    f"Download from: https://zenodo.org/record/7123579"
                )
        
        # Set default output path
        if output_geojson is None:
            output_dir = Path(__file__).parent.parent / 'output_geojson'
            output_dir.mkdir(exist_ok=True)
            output_geojson = str(output_dir / 'detected_crowns.geojson')
        
        # Initialize TreeCrownDelineator
        print(f"   Initializing TreeCrownDelineator...")
        delineator = TreeCrownDelineator(
            model_path=model_path,
            device=self.device
        )
        
        # Run prediction with tiled inference
        print(f"   Running tiled inference...")
        delineator.predict(
            image_path=geotiff_path,
            output_path=output_geojson,
            tile_width=tile_width,
            tile_height=tile_height,
            output_format='geojson',
            min_area=0.5,  # Minimum area in m² (adjust for your needs)
            iou_threshold=0.5  # NMS threshold for overlapping detections
        )
        
        # Calculate statistics from GeoJSON
        stats = self._calculate_geojson_stats(output_geojson)
        
        print(f"✅ Detection complete!")
        print(f"   Trees detected: {stats['tree_count']}")
        print(f"   Total canopy area: {stats['total_area_m2']:.2f} m²")
        print(f"   Output: {output_geojson}")
        
        return {
            'geojson_path': output_geojson,
            'tree_count': stats['tree_count'],
            'total_area_m2': stats['total_area_m2'],
            'avg_crown_area_m2': stats['avg_crown_area_m2'],
            'metadata': {
                'tile_size': f"{tile_width}x{tile_height}",
                'model': Path(model_path).name,
                'device': self.device
            }
        }
    
    def _calculate_geojson_stats(self, geojson_path: str) -> dict:
        """Calculate statistics from output GeoJSON"""
        import json
        
        with open(geojson_path, 'r') as f:
            geojson = json.load(f)
        
        features = geojson.get('features', [])
        tree_count = len(features)
        
        if tree_count == 0:
            return {
                'tree_count': 0,
                'total_area_m2': 0.0,
                'avg_crown_area_m2': 0.0
            }
        
        # Calculate areas (features should have area property from samgeo)
        areas = []
        for feature in features:
            props = feature.get('properties', {})
            area = props.get('area', 0.0)  # Area in m²
            if area > 0:
                areas.append(area)
        
        total_area = sum(areas)
        avg_area = total_area / len(areas) if areas else 0.0
        
        return {
            'tree_count': tree_count,
            'total_area_m2': total_area,
            'avg_crown_area_m2': avg_area
        }
    



def test_detectree2():
    """Test function for detectree2 detector"""
    print("\n" + "="*70)
    print("TESTING DETECTREE2 DETECTOR")
    print("="*70 + "\n")
    
    # Create dummy test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:, :] = [34, 139, 34]  # Green background
    
    # Add some tree-like circles
    cv2.circle(test_image, (200, 200), 50, (0, 255, 0), -1)
    cv2.circle(test_image, (400, 300), 60, (0, 200, 0), -1)
    
    # Initialize detector
    detector = Detectree2Detector(confidence_threshold=0.5, device='cpu')
    
    # Run detection
    try:
        polygons, mask, metadata = detector.detect_from_image(test_image)
        print(f"\n✓ Detection successful!")
        print(f"   Detected {len(polygons)} canopy regions")
        print(f"   Metadata: {metadata}")
        return True
    except Exception as e:
        print(f"\n✗ Detection failed: {e}")
        return False


if __name__ == "__main__":
    test_detectree2()
