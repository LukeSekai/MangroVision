"""
MangroVision - Proper Detectree2 Integration
Uses the official detectree2 library for accurate tree crown delineation
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Dict
import torch
from shapely.geometry import Polygon
import json

# Official detectree2 imports
from detectree2.models.train import setup_cfg, get_tree_dicts
from detectree2.models.predict import predict_on_data
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo


class ProperDetectree2Detector:
    """
    Proper integration with detectree2 library
    Uses official detectree2 prediction pipeline for maximum accuracy
    """
    
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 device: str = 'cpu'):
        """
        Initialize proper detectree2 detector
        
        Args:
            confidence_threshold: Minimum confidence score (0-1)
            device: 'cpu' or 'cuda'
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.predictor = None
        self.cfg = None
        
        print(f"🌳 Initializing Proper Detectree2 Library")
        print(f"   Using official detectree2 prediction pipeline")
        print(f"   Device: {device}")
        print(f"   Confidence threshold: {confidence_threshold}")
        
    def setup_model(self, model_path: str = None):
        """
        Setup detectree2 model with proper configuration
        
        Args:
            model_path: Path to model weights (.pth file)
        """
        print(f"⚙️ Setting up detectree2 model...")
        
        # Find model file
        if model_path is None:
            model_dir = Path(__file__).parent.parent / 'models'
            
            # Look for detectree2 models in order of preference (BEST FIRST)
            model_candidates = [
                (model_dir / '230103_randresize_full.pth', 'Optimized Tropical (Zenodo 230103)'),
                (model_dir / 'detectree2_model.pth', 'Custom Model'),
                (model_dir / '230717_tropical_base.pth', 'Base Tropical (230717)')
            ]
            
            for candidate, model_name in model_candidates:
                if candidate.exists():
                    model_path = str(candidate)
                    print(f"   ✅ Loading {model_name}")
                    break
            
            if model_path is None:
                raise FileNotFoundError(
                    f"No detectree2 model found in {model_dir}\n"
                    f"Download from: https://github.com/PatBall1/detectree2/releases"
                )
        
        # Setup config using Mask R-CNN R101-FPN
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        ))
        
        # Detectree2 configuration
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only 1 class: tree crown
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.DEVICE = self.device
        cfg.MODEL.WEIGHTS = model_path
        
        # Optimize for dense canopy - INCREASE limits for more detections
        cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000  # Increased proposals
        cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000  # Keep more detections
        cfg.MODEL.RPN.NMS_THRESH = 0.6  # Lower NMS threshold
        cfg.TEST.DETECTIONS_PER_IMAGE = 1000
        
        # Input format
        cfg.INPUT.FORMAT = "BGR"
        
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        
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
        
        # Green vegetation ranges
        lower_green1 = np.array([25, 30, 20])
        upper_green1 = np.array([85, 255, 255])
        mask1 = cv2.inRange(hsv, lower_green1, upper_green1)
        
        lower_green2 = np.array([85, 20, 20])
        upper_green2 = np.array([100, 255, 255])
        mask2 = cv2.inRange(hsv, lower_green2, upper_green2)
        
        vegetation_mask = cv2.bitwise_or(mask1, mask2)
        
        # Clean up noise
        kernel = np.ones((5, 5), np.uint8)
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_OPEN, kernel)
        vegetation_mask = cv2.morphologyEx(vegetation_mask, cv2.MORPH_CLOSE, kernel)
        
        return vegetation_mask
    
    def detect_from_image(self, 
                         image: np.ndarray,
                         tile_size: int = 512,
                         overlap: float = 0.25) -> Tuple[List[Polygon], np.ndarray, Dict]:
        """
        Detect tree crowns using proper detectree2 tiled inference
        OPTIMIZED: 512px tiles with 25% overlap (matches training data size)
        
        Args:
            image: Input BGR image
            tile_size: Size of tiles for detection (default 512px, matches training)
            overlap: Overlap fraction between tiles (default 0.25 = 25%)
            
        Returns:
            Tuple of (polygons, mask, metadata)
        """
        if self.predictor is None:
            self.setup_model()
        
        h, w = image.shape[:2]
        print(f"🌳 Running detectree2 on {w}x{h} image...")
        
        # PHASE 1: HSV Pre-filter - Find all vegetation (FAST)
        vegetation_mask = self._detect_vegetation_hsv(image)
        veg_pixels = np.count_nonzero(vegetation_mask)
        veg_percent = 100 * veg_pixels / (h * w)
        print(f"   Phase 1: HSV found {veg_percent:.1f}% vegetation coverage")
        
        # Calculate stride based on overlap
        stride = int(tile_size * (1 - overlap))
        
        # PHASE 2: Generate tiles - ONLY process tiles with vegetation
        tiles = []
        tiles_checked = 0
        for y in range(0, h, stride):
            for x in range(0, w, stride):
                x_end = min(x + tile_size, w)
                y_end = min(y + tile_size, h)
                
                tiles_checked += 1
                
                # OPTIMIZATION: Skip tiles without vegetation
                tile_veg_mask = vegetation_mask[y:y_end, x:x_end]
                veg_ratio = np.count_nonzero(tile_veg_mask) / ((x_end - x) * (y_end - y))
                
                # Only process tiles with at least 10% vegetation
                if veg_ratio >= 0.10:
                    tiles.append((x, y, x_end, y_end))
        
        skipped_tiles = tiles_checked - len(tiles)
        print(f"   Phase 2: Processing {len(tiles)} tiles with vegetation (skipped {skipped_tiles} empty)")
        print(f"   Tile size: {tile_size}px, overlap: {int(overlap*100)}%")
        
        # Run detection on each tile
        all_instances = []
        
        for tile_idx, (x1, y1, x2, y2) in enumerate(tiles):
            if tile_idx % 10 == 0:
                print(f"   Tile {tile_idx+1}/{len(tiles)}...")
            
            tile = image[y1:y2, x1:x2]
            
            with torch.no_grad():
                outputs = self.predictor(tile)
            
            instances = outputs["instances"].to("cpu")
            scores = instances.scores.numpy()
            masks = instances.pred_masks.numpy()
            
            # Store each detection with global coordinates
            for i in range(len(scores)):
                if scores[i] >= self.confidence_threshold:
                    mask = masks[i].astype(np.uint8)
                    
                    # Find contours
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    
                    for contour in contours:
                        # Convert to global coordinates
                        contour_global = contour.copy()
                        contour_global[:, 0, 0] += x1
                        contour_global[:, 0, 1] += y1
                        
                        # Convert to polygon
                        if len(contour_global) >= 3:
                            try:
                                points = contour_global.reshape(-1, 2)
                                poly = Polygon(points)
                                if poly.is_valid and poly.area > 100:  # Min 100 pixels
                                    all_instances.append({
                                        'polygon': poly,
                                        'score': scores[i],
                                        'contour': contour_global
                                    })
                            except:
                                continue
        
        print(f"   Found {len(all_instances)} detections")
        
        # DISABLE NMS - Keep all detections for dense canopy
        final_polygons = [inst['polygon'] for inst in all_instances]
        
        print(f"   ✅ {len(final_polygons)} trees detected (NMS disabled for dense canopy)")
        
        # Create combined mask
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        for poly in final_polygons:
            try:
                coords = np.array(poly.exterior.coords, dtype=np.int32)
                cv2.fillPoly(combined_mask, [coords], 255)
            except:
                continue
        
        metadata = {
            'num_tiles': len(tiles),
            'tile_size': tile_size,
            'overlap': overlap,
            'raw_detections': len(all_instances),
            'final_trees': len(final_polygons),
            'detection_method': 'detectree2_official'
        }
        
        return final_polygons, combined_mask, metadata
    
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
                except:
                    continue
            
            if should_keep:
                keep.append(poly)
        
        return keep


# Convenience function for backward compatibility
def detect_trees_detectree2(image: np.ndarray, 
                            confidence: float = 0.5,
                            tile_size: int = 800) -> Tuple[List[Polygon], np.ndarray, Dict]:
    """
    Detect trees using proper detectree2 library
    
    Args:
        image: Input BGR image
        confidence: Confidence threshold
        tile_size: Tile size for detection
        
    Returns:
        Tuple of (polygons, mask, metadata)
    """
    detector = ProperDetectree2Detector(confidence_threshold=confidence)
    detector.setup_model()
    return detector.detect_from_image(image, tile_size=tile_size)
