"""
MangroVision - Proper Detectree2 Integration
Uses the official detectree2 library for accurate tree crown delineation
"""

import cv2
import json
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
                 confidence_threshold: float = 0.90,
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
        self.model_path = None
        self.model_name = None
        self.model_metadata: Dict[str, Any] = {}
        self.model_metadata_path = None
        self.num_classes = None
        self.class_names: List[str] = []
        self.canopy_class_ids: Optional[List[int]] = None
        self.excluded_class_ids: List[int] = []
        self.runtime_tuning = {
            "tile_veg_threshold": 0.002,
            "min_crown_m2": 0.05,
            "max_crown_m2": 60.0,
            "cleanup_iou": 0.75,
            "fallback_nms_iou": 0.90,
            "tile_size": 512,
            "tile_overlap": 0.25,
            "use_clean_crowns": False,
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

    def _load_model_metadata(self, model_path: Path) -> Tuple[Dict[str, Any], Optional[str]]:
        """Load optional sidecar metadata stored next to a checkpoint."""
        candidates = [
            model_path.with_suffix(".json"),
            model_path.parent / "model_metadata.json",
        ]
        for candidate in candidates:
            if not candidate.exists():
                continue
            try:
                with candidate.open("r", encoding="utf-8") as handle:
                    data = json.load(handle)
                if isinstance(data, dict):
                    return data, str(candidate)
            except Exception as exc:
                print(f"   [WARN] Failed to read model metadata from {candidate}: {exc}")
        return {}, None

    def _infer_num_classes(self, model_path: Path, metadata: Optional[Dict[str, Any]] = None) -> int:
        """Infer ROI class count from metadata or checkpoint weights."""
        if metadata:
            value = metadata.get("num_classes")
            if isinstance(value, int) and value > 0:
                return value

        try:
            checkpoint = torch.load(str(model_path), map_location="cpu")
            state = checkpoint.get("model", checkpoint)
            cls_score = state.get("roi_heads.box_predictor.cls_score.weight")
            if cls_score is not None and len(cls_score.shape) == 2:
                return max(1, int(cls_score.shape[0] - 1))
        except Exception as exc:
            print(f"   [WARN] Could not infer class count from checkpoint: {exc}")

        if "custom_mangrove_model" in str(model_path).replace("\\", "/"):
            return 2
        return 1

    def _resolve_canopy_class_ids(self, metadata: Dict[str, Any], num_classes: int) -> Optional[List[int]]:
        """Return the classes that should count as canopy for this app."""
        raw_ids = metadata.get("canopy_class_ids")
        if isinstance(raw_ids, list):
            class_ids = sorted(
                {
                    int(value)
                    for value in raw_ids
                    if isinstance(value, (int, float)) and 0 <= int(value) < num_classes
                }
            )
            if class_ids:
                return class_ids

        if num_classes == 1:
            return [0]

        return None

    def _resolve_excluded_class_ids(self, metadata: Dict[str, Any], class_names: List[str]) -> List[int]:
        """Map excluded class names into class ids when metadata provides them."""
        excluded_names = metadata.get("excluded_class_names")
        if not isinstance(excluded_names, list) or not class_names:
            return []

        excluded_ids: List[int] = []
        for index, class_name in enumerate(class_names):
            if class_name in excluded_names:
                excluded_ids.append(index)
        return excluded_ids
        
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
            model_dir = Path(__file__).parent.parent / 'models'

            # Prefer latest model-garden checkpoints first, then local custom/legacy.
            latest_model_garden = sorted(model_dir.glob("250312*.pth"))
            model_candidates = [(p, f"Model Garden ({p.name})") for p in latest_model_garden]
            model_candidates.extend([
                (model_dir / 'try_new_latest_model' / 'model_final.pth', 'Try New Latest Model'),
                (model_dir / 'custom_mangrove_model' / 'model_final.pth', 'Custom Mangrove Model'),
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

        model_path_obj = Path(model_path)
        metadata, metadata_path = self._load_model_metadata(model_path_obj)
        num_classes = self._infer_num_classes(model_path_obj, metadata)
        class_names = metadata.get("class_names") if isinstance(metadata.get("class_names"), list) else []
        canopy_class_ids = self._resolve_canopy_class_ids(metadata, num_classes)
        excluded_class_ids = self._resolve_excluded_class_ids(metadata, class_names)
        
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

        cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(num_classes)

        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
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
        self.model_metadata = metadata
        self.model_metadata_path = metadata_path
        self.num_classes = int(num_classes)
        self.class_names = [str(name) for name in class_names]
        self.canopy_class_ids = canopy_class_ids
        self.excluded_class_ids = excluded_class_ids
        print(f"   Classes configured: {self.num_classes}")
        if self.class_names:
            print(f"   Class names: {', '.join(self.class_names)}")
        if self.canopy_class_ids is not None:
            print(f"   Canopy class ids: {self.canopy_class_ids}")
        if self.excluded_class_ids:
            print(f"   Excluded class ids: {self.excluded_class_ids}")
        if self.model_metadata_path:
            print(f"   Metadata: {self.model_metadata_path}")
        
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
        score_pass_detections = 0
        filtered_non_canopy_detections = 0
        kept_class_counts: Dict[int, int] = {}
        min_crown_m2 = float(self.runtime_tuning.get("min_crown_m2", 0.05))
        max_crown_m2 = float(self.runtime_tuning.get("max_crown_m2", 60.0))
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
            pred_classes = (
                instances.pred_classes.numpy()
                if instances.has("pred_classes")
                else np.zeros(len(scores), dtype=np.int64)
            )

            # Store each detection with global coordinates and confidence.
            for i in range(len(scores)):
                if scores[i] >= self.confidence_threshold:
                    score_pass_detections += 1
                    class_id = int(pred_classes[i]) if i < len(pred_classes) else 0
                    if self.canopy_class_ids is not None and class_id not in self.canopy_class_ids:
                        filtered_non_canopy_detections += 1
                        continue

                    mask = masks[i].astype(np.uint8)
                    kept_class_counts[class_id] = kept_class_counts.get(class_id, 0) + 1
                    
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
                                if poly.is_valid and (min_area_px <= poly.area <= max_area_px):
                                    all_instances.append({
                                        'polygon': poly,
                                        'score': scores[i],
                                        'contour': contour_global,
                                        'class_id': class_id,
                                    })
                            except:
                                continue
         
        print(f"   Found {len(all_instances)} detections")

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

        print(f"   ✅ {len(final_polygons)} trees detected after cleanup")
        _emit_progress(
            "tile_complete",
            {
                "total_tiles": len(tiles),
                "raw_detections": int(len(all_instances)),
                "final_trees": int(len(final_polygons)),
            },
        )
         
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
            'num_tiles_checked': int(tiles_checked),
            'num_tiles_processed': len(tiles),
            'num_tiles_skipped': int(skipped_tiles),
            'tile_size': tile_size,
            'overlap': overlap,
            'tile_veg_threshold': float(veg_tile_threshold),
            'raw_detections': len(all_instances),
            'total_ai_detections': len(all_instances),
            'score_pass_detections': int(score_pass_detections),
            'filtered_non_canopy_detections': int(filtered_non_canopy_detections),
            'kept_class_counts': kept_class_counts,
            'final_trees': len(final_polygons),
            'num_detected_canopies': len(final_polygons),
            'num_classes': self.num_classes,
            'class_names': self.class_names,
            'canopy_class_ids': self.canopy_class_ids,
            'excluded_class_ids': self.excluded_class_ids,
            'gsd_used': gsd_used,
            'min_crown_m2': min_crown_m2,
            'max_crown_m2': max_crown_m2,
            'model_path': self.model_path,
            'model_name': self.model_name,
            'model_metadata_path': self.model_metadata_path,
            'cleanup_iou': float(self.runtime_tuning.get("cleanup_iou", 0.75)),
            'fallback_nms_iou': float(self.runtime_tuning.get("fallback_nms_iou", 0.90)),
            'use_clean_crowns': use_clean_crowns,
            'detection_method': 'detectree2_official'
        }
         
        return final_polygons, combined_mask, metadata

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
