"""
MangroVision - Proper Detectree2 Integration
Uses the official detectree2 library for accurate tree crown delineation
"""

import cv2
import json
import numpy as np
import os
from pathlib import Path
from typing import Tuple, List, Dict, Optional, Callable, Any
import torch
from shapely.geometry import Polygon
import geopandas as gpd

try:
    from .seedling_detector import SeedlingDetector, SeedlingDetectorConfig
    from .seedling_leaf_evidence import recover_yellow_leaf_clusters
except ImportError:  # The legacy applications import this file as a top-level module.
    from seedling_detector import SeedlingDetector, SeedlingDetectorConfig
    from seedling_leaf_evidence import recover_yellow_leaf_clusters

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
                 confidence_threshold: float = 0.80,
                 device: str = 'cpu'):
        """
        Initialize proper detectree2 detector

        The default confidence threshold is 0.80. Moderate-confidence
        proposals still pass through the water/structure and canopy-color
        gates below, so lowering the public threshold improves recall without
        accepting every ambiguous model proposal.

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
        self.class_names: List[str] = []
        self.canopy_class_ids: set[int] = {0}
        self.runtime_tuning = {
            "tile_veg_threshold": 0.002,
            # Raised from 0.01 m² → 0.05 m² (~25 cm crown diameter). The old
            # threshold let through pixel-cluster false positives on water
            # ripples and shadow, each of which became a 1-2 m red halo in the
            # planting overlay (especially painful at bungalon's 1 m spacing).
            # Genuine small saplings still pass; only single-cluster noise is
            # rejected. The strict_canopy_hsv gate below remains the primary
            # color-based defense.
            "min_crown_m2": 0.05,
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
            # Precision filter for AI masks that are technically green-ish but
            # visually read as water/mud/roof tint: low saturation, low excess
            # green, and only partial strict-green support.
            "reject_low_saturation_components": True,
            "water_component_max_saturation": 60.0,
            "water_component_max_exg": 28.0,
            "water_component_max_green_ratio": 0.80,
            # Recover small missed crowns without lowering the public
            # confidence gate globally. Below-threshold detections are kept
            # only when the color evidence is strongly canopy-like.
            "recall_confidence_floor": 0.75,
            "recall_veg_min_ratio": 0.85,
            "recall_min_saturation": 80.0,
            "recall_min_exg": 35.0,
            # Keep the precision behavior of the former 0.87 operating point:
            # proposals below this score must show canopy color/shape evidence,
            # even when the public confidence threshold is lower.
            "color_gate_confidence": 0.87,
            # A second recall path for small crowns in shadow or mixed water
            # backgrounds. These do not always have strong excess-green, but
            # they are compact and still saturated enough to differ from the
            # broad drab water/structure false positives.
            "shadow_recall_confidence_floor": 0.60,
            "shadow_recall_max_area_m2": 1.20,
            "shadow_recall_min_veg_ratio": 0.35,
            "shadow_recall_min_saturation": 90.0,
            "shadow_recall_min_exg": 15.0,
            # Detect seedlings that are too small for reliable instance-mask
            # proposals. These are tiny, compact yellow-green blobs in planted
            # rows. Strategy: accept small absolute sizes (real seedlings can
            # be just a few leaves) but demand strong local color contrast
            # against the surrounding mud — that's what separates a real
            # planted sprout from water ripple or wet-sand noise.
            "seedling_supplement": True,
            # 0.0008 m² ≈ 8 cm² ≈ 3 cm leaf cluster. Big enough to skip
            # single-pixel speckles, small enough to keep real seedlings.
            "seedling_min_area_m2": 0.0008,
            "seedling_max_area_m2": 0.025,
            "seedling_max_bbox_m": 0.45,
            # Absolute color floors — kept moderate so real seedlings in
            # partial shade aren't rejected. The real false-positive rejection
            # happens at the *local contrast* layer below.
            "seedling_min_saturation": 28.0,
            "seedling_min_value": 40.0,
            "seedling_min_exg": 10.0,
            "seedling_neighbor_radius_m": 1.25,
            # Contextual rescue is only for a weak fragment immediately beside
            # an already-qualified seedling.  Keeping this much tighter than
            # the row-support radius prevents unrelated mud speckles from
            # borrowing evidence from a real plant elsewhere in the row.
            "seedling_contextual_neighbor_radius_m": 0.45,
            # Bright leaves from one tiny plant are often separated by a
            # one-pixel JPEG/mud gap.  Group only that immediate fragmentation
            # before rendering one marker per physical plant.  A separate,
            # strict cluster ceiling recovers juvenile plants that are larger
            # than the legacy tiny-component cap without touching the mature
            # Detectree2 mask.
            "seedling_fragment_link_m": 0.03,
            "seedling_cluster_max_area_m2": 0.15,
            "seedling_cluster_max_bbox_m": 0.65,
            "seedling_cluster_min_pixels": 2,
            # Final hard-negative gate for smooth green water.  The gate is
            # evaluated on the complete linked leaf cluster, so normal,
            # contextual, micro, and cluster-rescue candidates cannot bypass
            # it.  Ring statistics exclude every green candidate pixel; this
            # keeps densely planted DJI_0990 seedlings from contaminating
            # their own background estimate.
            "seedling_water_context_rejection": True,
            "seedling_water_context_inner_radius_m": 0.16,
            "seedling_water_context_outer_radius_m": 0.40,
            "seedling_water_context_min_saturation": 32.0,
            "seedling_water_context_min_exg": 12.0,
            "seedling_water_context_min_chroma": 0.0,
            # A second water signature covers neutral gray-green water and
            # timber-edge reflections.  These backgrounds can be saturated
            # while having near-zero/negative green chroma, so they bypass the
            # green-water rule above.  The bounded ExG/chroma interval keeps
            # this path away from genuinely green vegetation backgrounds.
            "seedling_neutral_water_min_saturation": 40.0,
            "seedling_neutral_water_min_exg": -5.0,
            # Yellow/olive mud can have ExG above 22 while its green chroma
            # remains neutral. Cover that tint too; genuine leaves still pass
            # the independent object-to-background contrast check.
            "seedling_neutral_water_max_exg": 35.0,
            "seedling_neutral_water_min_chroma": -0.03,
            "seedling_neutral_water_max_chroma": 0.006,
            "seedling_water_min_saturation_gain": 18.0,
            "seedling_water_min_exg_gain": 12.0,
            "seedling_water_min_object_exg": 35.0,
            "seedling_water_min_object_chroma": 0.018,
            # A represented component previously skipped all size checks.  A
            # connected field of water glints could therefore collapse into
            # one accepted "seedling" spanning hundreds of square metres.
            "seedling_linked_max_area_m2": 0.60,
            "seedling_linked_max_bbox_m": 2.0,
            # Reverted to 2 neighbors. Real seedlings in planted rows are
            # clustered, but the local-contrast gate already rejects isolated
            # noise blobs without help from the neighbor count.
            "seedling_min_neighbors": 2,
            "seedling_weak_min_neighbors": 2,
            "seedling_duplicate_radius_m": 0.10,
            "seedling_marker_radius_m": 0.08,
            "seedling_local_contrast_radius_m": 0.45,
            # Bumped from 4 → 10 (brightness gain) and 6 → 18 (excess-green
            # gain). This is the key change: mud and water have near-zero
            # excess-green, so requiring a +18 ExG jump vs the local
            # neighborhood reliably distinguishes a green seedling from a
            # wet-mud highlight or shadow speckle.
            "seedling_min_local_value_gain": 10.0,
            "seedling_min_local_exg_gain": 18.0,
            "seedling_min_chroma": 0.012,
            # Micro-seedlings can occupy only 2–7 pixels at the observed
            # 0.97–1.49 cm/px GSD. They use a stricter all-features gate and
            # local green support instead of weakening the normal supplement.
            "seedling_micro_supplement": True,
            "seedling_micro_min_area_m2": 0.00005,
            "seedling_micro_max_area_m2": 0.0008,
            "seedling_micro_max_bbox_m": 0.22,
            "seedling_micro_min_saturation": 48.0,
            "seedling_micro_min_value": 55.0,
            "seedling_micro_min_exg": 16.0,
            "seedling_micro_min_chroma": 0.009,
            "seedling_micro_min_local_value_gain": 6.0,
            "seedling_micro_min_local_exg_gain": 12.0,
            "seedling_micro_min_neighbors": 1,
            "seedling_micro_support_radius_m": 0.06,
            "seedling_micro_min_support_pixels": 2,
            # ``None`` means accepted seedlings inherit the user-configured
            # canopy safety radius. They remain outside the mature-canopy mask
            # and area statistics, but they must still exclude unsafe planting
            # cells. A numeric runtime override (including 0) remains available.
            "seedling_buffer_m": None,
            # The mature-canopy detector remains unchanged.  Hybrid mode is
            # attempted only when a trained seedling checkpoint is available;
            # otherwise the existing supplement is used as a compatibility
            # fallback so deployment behavior does not silently disappear.
            "seedling_detection_mode": os.getenv("MANGROVISION_SEEDLING_DETECTION_MODE", "hybrid"),
            "seedling_classifier_path": os.getenv("MANGROVISION_SEEDLING_CLASSIFIER_PATH"),
            "seedling_classifier_threshold": 0.90,
            "seedling_isolated_classifier_threshold": 0.97,
            "seedling_crop_size_m": 0.75,
            "seedling_input_size": 64,
            "seedling_batch_size": 64,
        }
        self.last_seedling_mask = np.zeros((0, 0), dtype=np.uint8)
        
        print(f"🌳 Initializing Proper Detectree2 Library")
        print(f"   Using official detectree2 prediction pipeline")
        print(f"   Device: {device}")
        print(f"   Confidence threshold: {confidence_threshold}")

    @staticmethod
    def _load_model_metadata(model_path: str) -> Dict[str, Any]:
        metadata_path = Path(model_path).with_name("model_metadata.json")
        if not metadata_path.exists():
            return {}
        try:
            with open(metadata_path, "r", encoding="utf-8") as fh:
                metadata = json.load(fh)
            return metadata if isinstance(metadata, dict) else {}
        except Exception as exc:
            print(f"   ⚠️ Could not read model metadata at {metadata_path}: {exc}")
            return {}

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
        reject_low_saturation_components: bool = None,
        water_component_max_saturation: float = None,
        water_component_max_exg: float = None,
        water_component_max_green_ratio: float = None,
        recall_confidence_floor: float = None,
        recall_veg_min_ratio: float = None,
        recall_min_saturation: float = None,
        recall_min_exg: float = None,
        color_gate_confidence: float = None,
        shadow_recall_confidence_floor: float = None,
        shadow_recall_max_area_m2: float = None,
        shadow_recall_min_veg_ratio: float = None,
        shadow_recall_min_saturation: float = None,
        shadow_recall_min_exg: float = None,
        seedling_supplement: bool = None,
        seedling_min_area_m2: float = None,
        seedling_max_area_m2: float = None,
        seedling_max_bbox_m: float = None,
        seedling_min_saturation: float = None,
        seedling_min_value: float = None,
        seedling_min_exg: float = None,
        seedling_neighbor_radius_m: float = None,
        seedling_contextual_neighbor_radius_m: float = None,
        seedling_fragment_link_m: float = None,
        seedling_cluster_max_area_m2: float = None,
        seedling_cluster_max_bbox_m: float = None,
        seedling_cluster_min_pixels: int = None,
        seedling_water_context_rejection: bool = None,
        seedling_water_context_inner_radius_m: float = None,
        seedling_water_context_outer_radius_m: float = None,
        seedling_water_context_min_saturation: float = None,
        seedling_water_context_min_exg: float = None,
        seedling_water_context_min_chroma: float = None,
        seedling_neutral_water_min_saturation: float = None,
        seedling_neutral_water_min_exg: float = None,
        seedling_neutral_water_max_exg: float = None,
        seedling_neutral_water_min_chroma: float = None,
        seedling_neutral_water_max_chroma: float = None,
        seedling_water_min_saturation_gain: float = None,
        seedling_water_min_exg_gain: float = None,
        seedling_water_min_object_exg: float = None,
        seedling_water_min_object_chroma: float = None,
        seedling_linked_max_area_m2: float = None,
        seedling_linked_max_bbox_m: float = None,
        seedling_min_neighbors: float = None,
        seedling_weak_min_neighbors: float = None,
        seedling_duplicate_radius_m: float = None,
        seedling_marker_radius_m: float = None,
        seedling_local_contrast_radius_m: float = None,
        seedling_min_local_value_gain: float = None,
        seedling_min_local_exg_gain: float = None,
        seedling_min_chroma: float = None,
        seedling_micro_supplement: bool = None,
        seedling_micro_min_area_m2: float = None,
        seedling_micro_max_area_m2: float = None,
        seedling_micro_max_bbox_m: float = None,
        seedling_micro_min_saturation: float = None,
        seedling_micro_min_value: float = None,
        seedling_micro_min_exg: float = None,
        seedling_micro_min_chroma: float = None,
        seedling_micro_min_local_value_gain: float = None,
        seedling_micro_min_local_exg_gain: float = None,
        seedling_micro_min_neighbors: int = None,
        seedling_micro_support_radius_m: float = None,
        seedling_micro_min_support_pixels: int = None,
        seedling_buffer_m: float = None,
        seedling_detection_mode: str = None,
        seedling_classifier_path: str = None,
        seedling_classifier_threshold: float = None,
        seedling_isolated_classifier_threshold: float = None,
        seedling_crop_size_m: float = None,
        seedling_input_size: int = None,
        seedling_batch_size: int = None,
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
        if reject_low_saturation_components is not None:
            self.runtime_tuning["reject_low_saturation_components"] = bool(reject_low_saturation_components)
        if water_component_max_saturation is not None:
            self.runtime_tuning["water_component_max_saturation"] = max(0.0, min(float(water_component_max_saturation), 255.0))
        if water_component_max_exg is not None:
            self.runtime_tuning["water_component_max_exg"] = max(-255.0, min(float(water_component_max_exg), 510.0))
        if water_component_max_green_ratio is not None:
            self.runtime_tuning["water_component_max_green_ratio"] = max(0.0, min(float(water_component_max_green_ratio), 1.0))
        if recall_confidence_floor is not None:
            self.runtime_tuning["recall_confidence_floor"] = max(0.0, min(float(recall_confidence_floor), 0.99))
        if recall_veg_min_ratio is not None:
            self.runtime_tuning["recall_veg_min_ratio"] = max(0.0, min(float(recall_veg_min_ratio), 1.0))
        if recall_min_saturation is not None:
            self.runtime_tuning["recall_min_saturation"] = max(0.0, min(float(recall_min_saturation), 255.0))
        if recall_min_exg is not None:
            self.runtime_tuning["recall_min_exg"] = max(-255.0, min(float(recall_min_exg), 510.0))
        if color_gate_confidence is not None:
            self.runtime_tuning["color_gate_confidence"] = max(0.0, min(float(color_gate_confidence), 0.99))
        if shadow_recall_confidence_floor is not None:
            self.runtime_tuning["shadow_recall_confidence_floor"] = max(0.0, min(float(shadow_recall_confidence_floor), 0.99))
        if shadow_recall_max_area_m2 is not None:
            self.runtime_tuning["shadow_recall_max_area_m2"] = max(0.01, float(shadow_recall_max_area_m2))
        if shadow_recall_min_veg_ratio is not None:
            self.runtime_tuning["shadow_recall_min_veg_ratio"] = max(0.0, min(float(shadow_recall_min_veg_ratio), 1.0))
        if shadow_recall_min_saturation is not None:
            self.runtime_tuning["shadow_recall_min_saturation"] = max(0.0, min(float(shadow_recall_min_saturation), 255.0))
        if shadow_recall_min_exg is not None:
            self.runtime_tuning["shadow_recall_min_exg"] = max(-255.0, min(float(shadow_recall_min_exg), 510.0))
        if seedling_supplement is not None:
            self.runtime_tuning["seedling_supplement"] = bool(seedling_supplement)
        if seedling_min_area_m2 is not None:
            self.runtime_tuning["seedling_min_area_m2"] = max(0.0, float(seedling_min_area_m2))
        if seedling_max_area_m2 is not None:
            self.runtime_tuning["seedling_max_area_m2"] = max(0.0, float(seedling_max_area_m2))
        if seedling_max_bbox_m is not None:
            self.runtime_tuning["seedling_max_bbox_m"] = max(0.01, float(seedling_max_bbox_m))
        if seedling_min_saturation is not None:
            self.runtime_tuning["seedling_min_saturation"] = max(0.0, min(float(seedling_min_saturation), 255.0))
        if seedling_min_value is not None:
            self.runtime_tuning["seedling_min_value"] = max(0.0, min(float(seedling_min_value), 255.0))
        if seedling_min_exg is not None:
            self.runtime_tuning["seedling_min_exg"] = max(-255.0, min(float(seedling_min_exg), 510.0))
        if seedling_neighbor_radius_m is not None:
            self.runtime_tuning["seedling_neighbor_radius_m"] = max(0.05, float(seedling_neighbor_radius_m))
        if seedling_contextual_neighbor_radius_m is not None:
            self.runtime_tuning["seedling_contextual_neighbor_radius_m"] = max(
                0.05,
                min(float(seedling_contextual_neighbor_radius_m), 1.0),
            )
        if seedling_fragment_link_m is not None:
            self.runtime_tuning["seedling_fragment_link_m"] = max(
                0.0,
                min(float(seedling_fragment_link_m), 0.10),
            )
        if seedling_cluster_max_area_m2 is not None:
            self.runtime_tuning["seedling_cluster_max_area_m2"] = max(
                self.runtime_tuning.get("seedling_max_area_m2", 0.025),
                float(seedling_cluster_max_area_m2),
            )
        if seedling_cluster_max_bbox_m is not None:
            self.runtime_tuning["seedling_cluster_max_bbox_m"] = max(
                self.runtime_tuning.get("seedling_max_bbox_m", 0.45),
                float(seedling_cluster_max_bbox_m),
            )
        if seedling_cluster_min_pixels is not None:
            self.runtime_tuning["seedling_cluster_min_pixels"] = max(
                2,
                int(seedling_cluster_min_pixels),
            )
        if seedling_water_context_rejection is not None:
            self.runtime_tuning["seedling_water_context_rejection"] = bool(
                seedling_water_context_rejection
            )
        if seedling_water_context_inner_radius_m is not None:
            self.runtime_tuning["seedling_water_context_inner_radius_m"] = max(
                0.05,
                float(seedling_water_context_inner_radius_m),
            )
        if seedling_water_context_outer_radius_m is not None:
            self.runtime_tuning["seedling_water_context_outer_radius_m"] = max(
                self.runtime_tuning.get("seedling_water_context_inner_radius_m", 0.16)
                + 0.02,
                float(seedling_water_context_outer_radius_m),
            )
        if seedling_water_context_min_saturation is not None:
            self.runtime_tuning["seedling_water_context_min_saturation"] = max(
                0.0,
                min(float(seedling_water_context_min_saturation), 255.0),
            )
        if seedling_water_context_min_exg is not None:
            self.runtime_tuning["seedling_water_context_min_exg"] = max(
                -255.0,
                min(float(seedling_water_context_min_exg), 510.0),
            )
        if seedling_water_context_min_chroma is not None:
            self.runtime_tuning["seedling_water_context_min_chroma"] = max(
                -1.0,
                min(float(seedling_water_context_min_chroma), 1.0),
            )
        if seedling_neutral_water_min_saturation is not None:
            self.runtime_tuning["seedling_neutral_water_min_saturation"] = max(
                0.0,
                min(float(seedling_neutral_water_min_saturation), 255.0),
            )
        if seedling_neutral_water_min_exg is not None:
            self.runtime_tuning["seedling_neutral_water_min_exg"] = max(
                -255.0,
                min(float(seedling_neutral_water_min_exg), 510.0),
            )
        if seedling_neutral_water_max_exg is not None:
            self.runtime_tuning["seedling_neutral_water_max_exg"] = max(
                -255.0,
                min(float(seedling_neutral_water_max_exg), 510.0),
            )
        if seedling_neutral_water_min_chroma is not None:
            self.runtime_tuning["seedling_neutral_water_min_chroma"] = max(
                -1.0,
                min(float(seedling_neutral_water_min_chroma), 1.0),
            )
        if seedling_neutral_water_max_chroma is not None:
            self.runtime_tuning["seedling_neutral_water_max_chroma"] = max(
                -1.0,
                min(float(seedling_neutral_water_max_chroma), 1.0),
            )
        if seedling_water_min_saturation_gain is not None:
            self.runtime_tuning["seedling_water_min_saturation_gain"] = float(
                seedling_water_min_saturation_gain
            )
        if seedling_water_min_exg_gain is not None:
            self.runtime_tuning["seedling_water_min_exg_gain"] = float(
                seedling_water_min_exg_gain
            )
        if seedling_water_min_object_exg is not None:
            self.runtime_tuning["seedling_water_min_object_exg"] = float(
                seedling_water_min_object_exg
            )
        if seedling_water_min_object_chroma is not None:
            self.runtime_tuning["seedling_water_min_object_chroma"] = max(
                -1.0,
                min(float(seedling_water_min_object_chroma), 1.0),
            )
        if seedling_linked_max_area_m2 is not None:
            self.runtime_tuning["seedling_linked_max_area_m2"] = max(
                self.runtime_tuning.get("seedling_cluster_max_area_m2", 0.15),
                float(seedling_linked_max_area_m2),
            )
        if seedling_linked_max_bbox_m is not None:
            self.runtime_tuning["seedling_linked_max_bbox_m"] = max(
                self.runtime_tuning.get("seedling_cluster_max_bbox_m", 0.65),
                float(seedling_linked_max_bbox_m),
            )
        if seedling_min_neighbors is not None:
            self.runtime_tuning["seedling_min_neighbors"] = max(0, int(seedling_min_neighbors))
        if seedling_weak_min_neighbors is not None:
            self.runtime_tuning["seedling_weak_min_neighbors"] = max(0, int(seedling_weak_min_neighbors))
        if seedling_duplicate_radius_m is not None:
            self.runtime_tuning["seedling_duplicate_radius_m"] = max(0.0, float(seedling_duplicate_radius_m))
        if seedling_marker_radius_m is not None:
            self.runtime_tuning["seedling_marker_radius_m"] = max(0.01, float(seedling_marker_radius_m))
        if seedling_local_contrast_radius_m is not None:
            self.runtime_tuning["seedling_local_contrast_radius_m"] = max(0.05, float(seedling_local_contrast_radius_m))
        if seedling_min_local_value_gain is not None:
            self.runtime_tuning["seedling_min_local_value_gain"] = float(seedling_min_local_value_gain)
        if seedling_min_local_exg_gain is not None:
            self.runtime_tuning["seedling_min_local_exg_gain"] = float(seedling_min_local_exg_gain)
        if seedling_min_chroma is not None:
            self.runtime_tuning["seedling_min_chroma"] = max(-1.0, min(float(seedling_min_chroma), 1.0))
        if seedling_micro_supplement is not None:
            self.runtime_tuning["seedling_micro_supplement"] = bool(seedling_micro_supplement)
        if seedling_micro_min_area_m2 is not None:
            self.runtime_tuning["seedling_micro_min_area_m2"] = max(0.0, float(seedling_micro_min_area_m2))
        if seedling_micro_max_area_m2 is not None:
            self.runtime_tuning["seedling_micro_max_area_m2"] = max(
                self.runtime_tuning.get("seedling_micro_min_area_m2", 0.00005),
                float(seedling_micro_max_area_m2),
            )
        if seedling_micro_max_bbox_m is not None:
            self.runtime_tuning["seedling_micro_max_bbox_m"] = max(0.02, float(seedling_micro_max_bbox_m))
        if seedling_micro_min_saturation is not None:
            self.runtime_tuning["seedling_micro_min_saturation"] = max(
                0.0, min(float(seedling_micro_min_saturation), 255.0)
            )
        if seedling_micro_min_value is not None:
            self.runtime_tuning["seedling_micro_min_value"] = max(
                0.0, min(float(seedling_micro_min_value), 255.0)
            )
        if seedling_micro_min_exg is not None:
            self.runtime_tuning["seedling_micro_min_exg"] = max(
                -255.0, min(float(seedling_micro_min_exg), 510.0)
            )
        if seedling_micro_min_chroma is not None:
            self.runtime_tuning["seedling_micro_min_chroma"] = max(
                -1.0, min(float(seedling_micro_min_chroma), 1.0)
            )
        if seedling_micro_min_local_value_gain is not None:
            self.runtime_tuning["seedling_micro_min_local_value_gain"] = float(seedling_micro_min_local_value_gain)
        if seedling_micro_min_local_exg_gain is not None:
            self.runtime_tuning["seedling_micro_min_local_exg_gain"] = float(seedling_micro_min_local_exg_gain)
        if seedling_micro_min_neighbors is not None:
            self.runtime_tuning["seedling_micro_min_neighbors"] = max(0, int(seedling_micro_min_neighbors))
        if seedling_micro_support_radius_m is not None:
            self.runtime_tuning["seedling_micro_support_radius_m"] = max(
                0.03, min(float(seedling_micro_support_radius_m), 1.0)
            )
        if seedling_micro_min_support_pixels is not None:
            self.runtime_tuning["seedling_micro_min_support_pixels"] = max(
                1, int(seedling_micro_min_support_pixels)
            )
        if seedling_buffer_m is not None:
            self.runtime_tuning["seedling_buffer_m"] = max(0.0, min(float(seedling_buffer_m), 2.0))
        if seedling_detection_mode is not None:
            mode = str(seedling_detection_mode).strip().lower()
            self.runtime_tuning["seedling_detection_mode"] = (
                mode if mode in {"legacy", "hybrid", "off", "disabled"} else "hybrid"
            )
        if seedling_classifier_path is not None:
            self.runtime_tuning["seedling_classifier_path"] = str(seedling_classifier_path)
        if seedling_classifier_threshold is not None:
            self.runtime_tuning["seedling_classifier_threshold"] = max(
                0.5, min(float(seedling_classifier_threshold), 0.999)
            )
        if seedling_isolated_classifier_threshold is not None:
            self.runtime_tuning["seedling_isolated_classifier_threshold"] = max(
                self.runtime_tuning.get("seedling_classifier_threshold", 0.90),
                min(float(seedling_isolated_classifier_threshold), 0.999),
            )
        if seedling_crop_size_m is not None:
            self.runtime_tuning["seedling_crop_size_m"] = max(0.20, min(float(seedling_crop_size_m), 3.0))
        if seedling_input_size is not None:
            self.runtime_tuning["seedling_input_size"] = max(32, min(int(seedling_input_size), 256))
        if seedling_batch_size is not None:
            self.runtime_tuning["seedling_batch_size"] = max(1, min(int(seedling_batch_size), 512))
        
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

        self.model_metadata = self._load_model_metadata(model_path)
        self.class_names = list(self.model_metadata.get("class_names") or [])
        metadata_num_classes = self.model_metadata.get("num_classes")

        # Set NUM_CLASSES per checkpoint family. Prefer metadata when present
        # so two-class heads can be interpreted consistently during inference.
        path_str = str(model_path).replace("\\", "/")
        if metadata_num_classes is not None:
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = int(metadata_num_classes)
        elif "latest_mangrove_only" in path_str:
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1
        elif "custom_mangrove_model" in path_str or "MangroVision_New/try_model" in path_str:
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
        else:
            cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1

        metadata_canopy_ids = self.model_metadata.get("canopy_class_ids")
        if isinstance(metadata_canopy_ids, list) and metadata_canopy_ids:
            self.canopy_class_ids = {int(class_id) for class_id in metadata_canopy_ids}
        elif cfg.MODEL.ROI_HEADS.NUM_CLASSES == 1:
            self.canopy_class_ids = {0}
        elif "custom_mangrove_model" in path_str:
            self.canopy_class_ids = {0}
        elif "MangroVision_New/try_model" in path_str:
            self.canopy_class_ids = {1}
        else:
            self.canopy_class_ids = {0}

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
        if self.class_names:
            print(f"   Classes: {self.class_names}")
        print(f"   Canopy class IDs: {sorted(self.canopy_class_ids)}")
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
        # The seedling supplement step (run before this) writes very small
        # blobs (~30 cm²) into instance_mask. If we extract polygons from the
        # merged mask using the full crown floor (min_area_px), those seedling
        # blobs survive as purple pixels but disappear from the polygon list —
        # which means create_danger_zones() never buffers them, leaving small
        # detections without a red exclusion ring. Use a seedling-aware floor
        # so every visible canopy speck also becomes a buffer-able polygon.
        # Seedlings are returned through a separate mask. Keep the mature
        # canopy polygon floor independent so a tiny seedling marker cannot be
        # re-extracted as purple/red canopy coverage.
        polygon_min_area_px = float(min_area_px)

        merge_enabled = bool(self.runtime_tuning.get("merge_canopy_fragments", True))
        if not merge_enabled or int(np.count_nonzero(instance_mask)) == 0:
            return (
                self._mask_to_polygons(instance_mask, polygon_min_area_px),
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

        coverage_polygons = self._mask_to_polygons(merged_mask, polygon_min_area_px)
        if not coverage_polygons:
            coverage_polygons = self._mask_to_polygons(instance_mask, polygon_min_area_px)
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

    def _filter_low_saturation_components(
        self,
        canopy_mask: np.ndarray,
        image: np.ndarray,
        strict_green_mask: Optional[np.ndarray],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Remove drab green-ish components that are usually water/mud/roof tint."""
        if not bool(self.runtime_tuning.get("reject_low_saturation_components", True)):
            return canopy_mask, {
                "reject_low_saturation_components": False,
                "rejected_low_saturation_components": 0,
                "rejected_low_saturation_pixels": 0,
            }

        if int(np.count_nonzero(canopy_mask)) == 0:
            return canopy_mask, {
                "reject_low_saturation_components": True,
                "rejected_low_saturation_components": 0,
                "rejected_low_saturation_pixels": 0,
            }

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation = hsv[:, :, 1].astype(np.float32)
        bgr = image.astype(np.float32)
        excess_green = (2.0 * bgr[:, :, 1]) - bgr[:, :, 2] - bgr[:, :, 0]

        max_sat = float(self.runtime_tuning.get("water_component_max_saturation", 60.0))
        max_exg = float(self.runtime_tuning.get("water_component_max_exg", 28.0))
        max_green_ratio = float(self.runtime_tuning.get("water_component_max_green_ratio", 0.80))

        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
            (canopy_mask > 0).astype(np.uint8),
            connectivity=8,
        )
        filtered = np.zeros_like(canopy_mask)
        removed_components = 0
        removed_pixels = 0

        for label_idx in range(1, num_labels):
            area = int(stats[label_idx, cv2.CC_STAT_AREA])
            if area <= 0:
                continue
            component_pixels = labels == label_idx
            mean_sat = float(np.mean(saturation[component_pixels]))
            mean_exg = float(np.mean(excess_green[component_pixels]))
            if strict_green_mask is not None and strict_green_mask.shape == canopy_mask.shape:
                green_ratio = (
                    float(np.count_nonzero(component_pixels & (strict_green_mask > 0))) / area
                )
            else:
                green_ratio = 0.0

            if mean_sat <= max_sat and mean_exg <= max_exg and green_ratio <= max_green_ratio:
                removed_components += 1
                removed_pixels += area
                continue

            filtered[component_pixels] = 255

        return filtered, {
            "reject_low_saturation_components": True,
            "water_component_max_saturation": max_sat,
            "water_component_max_exg": max_exg,
            "water_component_max_green_ratio": max_green_ratio,
            "rejected_low_saturation_components": int(removed_components),
            "rejected_low_saturation_pixels": int(removed_pixels),
        }

    def _detect_seedling_supplement_mask(
        self,
        image: np.ndarray,
        base_mask: np.ndarray,
        gsd: Optional[float],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Add tiny seedling detections from compact yellow-green components."""
        if not bool(self.runtime_tuning.get("seedling_supplement", True)) or gsd is None or gsd <= 0:
            return np.zeros_like(base_mask), {
                "seedling_supplement": False,
                "seedling_supplement_count": 0,
                "seedling_supplement_pixels": 0,
            }

        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        hue = hsv[:, :, 0]
        sat = hsv[:, :, 1]
        val = hsv[:, :, 2]
        bgr = image.astype(np.float32)
        b = bgr[:, :, 0]
        g = bgr[:, :, 1]
        r = bgr[:, :, 2]
        excess_green = (2.0 * g) - r - b
        green_chroma = (g - np.maximum(r, b)) / np.maximum(r + g + b, 1.0)

        min_area_m2 = float(self.runtime_tuning.get("seedling_min_area_m2", 0.00005))
        max_area_m2 = float(self.runtime_tuning.get("seedling_max_area_m2", 0.025))
        max_bbox_m = float(self.runtime_tuning.get("seedling_max_bbox_m", 0.45))
        min_sat = float(self.runtime_tuning.get("seedling_min_saturation", 24.0))
        min_val = float(self.runtime_tuning.get("seedling_min_value", 35.0))
        min_exg = float(self.runtime_tuning.get("seedling_min_exg", 5.0))
        neighbor_radius_m = float(self.runtime_tuning.get("seedling_neighbor_radius_m", 1.25))
        contextual_neighbor_radius_m = float(
            self.runtime_tuning.get("seedling_contextual_neighbor_radius_m", 0.45)
        )
        fragment_link_m = float(
            self.runtime_tuning.get("seedling_fragment_link_m", 0.03)
        )
        cluster_max_area_m2 = float(
            self.runtime_tuning.get("seedling_cluster_max_area_m2", 0.15)
        )
        cluster_max_bbox_m = float(
            self.runtime_tuning.get("seedling_cluster_max_bbox_m", 0.65)
        )
        cluster_min_pixels = max(
            2,
            int(self.runtime_tuning.get("seedling_cluster_min_pixels", 2)),
        )
        water_context_rejection = bool(
            self.runtime_tuning.get("seedling_water_context_rejection", True)
        )
        water_context_inner_radius_m = float(
            self.runtime_tuning.get("seedling_water_context_inner_radius_m", 0.16)
        )
        water_context_outer_radius_m = max(
            water_context_inner_radius_m + 0.02,
            float(
                self.runtime_tuning.get(
                    "seedling_water_context_outer_radius_m",
                    0.40,
                )
            ),
        )
        water_context_min_sat = float(
            self.runtime_tuning.get("seedling_water_context_min_saturation", 32.0)
        )
        water_context_min_exg = float(
            self.runtime_tuning.get("seedling_water_context_min_exg", 12.0)
        )
        water_context_min_chroma = float(
            self.runtime_tuning.get("seedling_water_context_min_chroma", 0.0)
        )
        neutral_water_min_sat = float(
            self.runtime_tuning.get("seedling_neutral_water_min_saturation", 40.0)
        )
        neutral_water_min_exg = float(
            self.runtime_tuning.get("seedling_neutral_water_min_exg", -5.0)
        )
        neutral_water_max_exg = max(
            neutral_water_min_exg,
            float(self.runtime_tuning.get("seedling_neutral_water_max_exg", 35.0)),
        )
        neutral_water_min_chroma = float(
            self.runtime_tuning.get("seedling_neutral_water_min_chroma", -0.03)
        )
        neutral_water_max_chroma = max(
            neutral_water_min_chroma,
            float(
                self.runtime_tuning.get(
                    "seedling_neutral_water_max_chroma",
                    0.006,
                )
            ),
        )
        water_min_sat_gain = float(
            self.runtime_tuning.get("seedling_water_min_saturation_gain", 18.0)
        )
        water_min_exg_gain = float(
            self.runtime_tuning.get("seedling_water_min_exg_gain", 12.0)
        )
        water_min_object_exg = float(
            self.runtime_tuning.get("seedling_water_min_object_exg", 35.0)
        )
        water_min_object_chroma = float(
            self.runtime_tuning.get("seedling_water_min_object_chroma", 0.018)
        )
        linked_max_area_m2 = float(
            self.runtime_tuning.get("seedling_linked_max_area_m2", 0.60)
        )
        linked_max_bbox_m = float(
            self.runtime_tuning.get("seedling_linked_max_bbox_m", 2.0)
        )
        min_neighbors = int(self.runtime_tuning.get("seedling_min_neighbors", 2))
        weak_min_neighbors = int(self.runtime_tuning.get("seedling_weak_min_neighbors", 2))
        duplicate_radius_m = float(self.runtime_tuning.get("seedling_duplicate_radius_m", 0.10))
        marker_radius_m = float(self.runtime_tuning.get("seedling_marker_radius_m", 0.08))
        contrast_radius_m = float(self.runtime_tuning.get("seedling_local_contrast_radius_m", 0.45))
        min_value_gain = float(self.runtime_tuning.get("seedling_min_local_value_gain", 4.0))
        min_exg_gain = float(self.runtime_tuning.get("seedling_min_local_exg_gain", 6.0))
        min_chroma = float(self.runtime_tuning.get("seedling_min_chroma", 0.006))
        micro_enabled = bool(self.runtime_tuning.get("seedling_micro_supplement", True))
        micro_min_area_m2 = float(self.runtime_tuning.get("seedling_micro_min_area_m2", 0.00005))
        micro_max_area_m2 = float(self.runtime_tuning.get("seedling_micro_max_area_m2", 0.0008))
        micro_max_bbox_m = float(self.runtime_tuning.get("seedling_micro_max_bbox_m", 0.22))
        micro_min_sat = float(self.runtime_tuning.get("seedling_micro_min_saturation", 48.0))
        micro_min_val = float(self.runtime_tuning.get("seedling_micro_min_value", 55.0))
        micro_min_exg = float(self.runtime_tuning.get("seedling_micro_min_exg", 16.0))
        micro_min_chroma = float(self.runtime_tuning.get("seedling_micro_min_chroma", 0.009))
        micro_min_value_gain = float(
            self.runtime_tuning.get("seedling_micro_min_local_value_gain", 6.0)
        )
        micro_min_exg_gain = float(
            self.runtime_tuning.get("seedling_micro_min_local_exg_gain", 12.0)
        )
        micro_min_neighbors = int(self.runtime_tuning.get("seedling_micro_min_neighbors", 1))
        micro_support_radius_m = float(
            self.runtime_tuning.get("seedling_micro_support_radius_m", 0.06)
        )
        micro_min_support_pixels = int(
            self.runtime_tuning.get("seedling_micro_min_support_pixels", 2)
        )

        strong_seedling_pixels = (
            (hue >= 16)
            & (hue <= 65)
            & (sat >= max(min_sat, 35.0))
            & (val >= min_val)
            & (excess_green >= max(min_exg, 10.0))
            & (g > (r + 4.0))
            & (g > (b + 2.0))
            & (green_chroma >= min_chroma)
        )
        pale_seedling_pixels = (
            (hue >= 16)
            & (hue <= 72)
            & (sat >= min_sat)
            & (val >= max(min_val, 58.0))
            & (excess_green >= min_exg)
            & (g > (r + 2.0))
            & (g > (b + 1.0))
            & (green_chroma >= min_chroma)
        )
        seedling_pixels = (strong_seedling_pixels | pale_seedling_pixels).astype(np.uint8)

        # Yellow wood and mud highlights can have high saturation/ExG while
        # green barely exceeds red. In such frames virtually none of the
        # candidate pixels have a convincing green leaf core. Use the fraction
        # within candidate vegetation, not the whole image, so a sparse real
        # plant can supply its own evidence. Leaf-rich frames retain the pale
        # and contextual recovery paths needed for shaded/tiny seedlings.
        leaf_core_pixels = (
            (seedling_pixels > 0) & (green_chroma >= 0.04)
            & (sat >= 55) & (val >= 45)
        )
        leaf_core_fraction = float(np.count_nonzero(leaf_core_pixels)) / max(
            1, int(np.count_nonzero(seedling_pixels))
        )
        low_leaf_evidence = leaf_core_fraction < 0.01

        micro_support_kernel_size = self._metric_kernel_size(
            micro_support_radius_m,
            gsd,
            fallback_px=9,
        )
        micro_support = cv2.boxFilter(
            seedling_pixels.astype(np.float32),
            -1,
            (micro_support_kernel_size, micro_support_kernel_size),
            normalize=False,
            borderType=cv2.BORDER_REPLICATE,
        )
        micro_local_value = cv2.blur(
            val.astype(np.float32),
            (micro_support_kernel_size, micro_support_kernel_size),
        )
        micro_local_exg = cv2.blur(
            excess_green.astype(np.float32),
            (micro_support_kernel_size, micro_support_kernel_size),
        )

        contrast_kernel_size = self._metric_kernel_size(contrast_radius_m, gsd, fallback_px=47)
        local_value = cv2.blur(val.astype(np.float32), (contrast_kernel_size, contrast_kernel_size))
        local_exg = cv2.blur(excess_green.astype(np.float32), (contrast_kernel_size, contrast_kernel_size))

        duplicate_mask = base_mask > 0
        if duplicate_radius_m > 0:
            duplicate_kernel_size = self._metric_kernel_size(duplicate_radius_m, gsd, fallback_px=9)
            if duplicate_kernel_size > 0:
                duplicate_kernel = cv2.getStructuringElement(
                    cv2.MORPH_ELLIPSE,
                    (duplicate_kernel_size, duplicate_kernel_size),
                )
                duplicate_mask = cv2.dilate(
                    duplicate_mask.astype(np.uint8),
                    duplicate_kernel,
                    iterations=1,
                ) > 0

        normal_min_area_px = max(1, int(round(min_area_m2 / (gsd ** 2))))
        micro_min_area_px = max(1, int(round(micro_min_area_m2 / (gsd ** 2))))
        micro_max_area_px = max(micro_min_area_px, int(round(micro_max_area_m2 / (gsd ** 2))))
        min_area_px = micro_min_area_px if micro_enabled else normal_min_area_px
        max_area_px = max(min_area_px, int(round(max_area_m2 / (gsd ** 2))))
        max_bbox_px = max(3, int(round(max_bbox_m / gsd)))
        micro_max_bbox_px = max(3, int(round(micro_max_bbox_m / gsd)))
        water_context_inner_px = max(
            2,
            int(round(water_context_inner_radius_m / gsd)),
        )
        water_context_outer_px = max(
            water_context_inner_px + 2,
            int(round(water_context_outer_radius_m / gsd)),
        )
        # Sun glints can acquire saturated yellow-green fringes through the
        # camera/JPEG color processing. Require both a field of clipped glints
        # and immediate adjacency to one before applying this specialized gate.
        clipped_highlights = (r >= 245) & (g >= 245) & (np.abs(r - g) <= 10)
        highlight_radius_px = max(2, int(round(0.04 / gsd)))
        near_clipped_highlight = cv2.dilate(
            clipped_highlights.astype(np.uint8),
            cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (2 * highlight_radius_px + 1, 2 * highlight_radius_px + 1),
            ),
        )

        def water_context_metrics(
            center: tuple[float, float],
            object_sat: float,
            object_exg: float,
            object_chroma: float,
        ) -> dict[str, Any]:
            """Classify a candidate against a green-free local background ring."""
            if not water_context_rejection:
                return {
                    "has_ring": False,
                    "water_context": False,
                    "verified": True,
                    "reject": False,
                }

            center_x = int(max(0, min(image.shape[1] - 1, round(center[0]))))
            center_y = int(max(0, min(image.shape[0] - 1, round(center[1]))))
            x0 = max(0, center_x - water_context_outer_px)
            x1 = min(image.shape[1], center_x + water_context_outer_px + 1)
            y0 = max(0, center_y - water_context_outer_px)
            y1 = min(image.shape[0], center_y + water_context_outer_px + 1)
            offset_y, offset_x = np.ogrid[
                y0 - center_y : y1 - center_y,
                x0 - center_x : x1 - center_x,
            ]
            distance_sq = (offset_x * offset_x) + (offset_y * offset_y)
            ring = (
                (distance_sq >= (water_context_inner_px ** 2))
                & (distance_sq <= (water_context_outer_px ** 2))
                & (seedling_pixels[y0:y1, x0:x1] == 0)
            )
            # A median needs enough actual background pixels to be meaningful.
            # Near image edges or a solid canopy, skip this specialized gate
            # and leave the existing duplicate/color filters in charge.
            if int(np.count_nonzero(ring)) < 24:
                return {
                    "has_ring": False,
                    "water_context": False,
                    "verified": True,
                    "reject": False,
                }

            ring_sat = float(np.median(sat[y0:y1, x0:x1][ring]))
            ring_exg = float(np.median(excess_green[y0:y1, x0:x1][ring]))
            ring_chroma = float(np.median(green_chroma[y0:y1, x0:x1][ring]))
            sat_gain = float(object_sat - ring_sat)
            exg_gain = float(object_exg - ring_exg)
            green_water_context = bool(
                ring_sat >= water_context_min_sat
                and ring_exg >= water_context_min_exg
                and ring_chroma >= water_context_min_chroma
            )
            neutral_water_context = bool(
                ring_sat >= neutral_water_min_sat
                and neutral_water_min_exg <= ring_exg <= neutral_water_max_exg
                and neutral_water_min_chroma
                <= ring_chroma
                <= neutral_water_max_chroma
            )
            highlight_fraction = float(np.mean(clipped_highlights[y0:y1, x0:x1][ring]))
            specular_water_context = bool(
                highlight_fraction >= 0.08
                and near_clipped_highlight[center_y, center_x]
            )
            water_context = bool(
                green_water_context or neutral_water_context or specular_water_context
            )
            verified = bool(
                sat_gain >= water_min_sat_gain
                and exg_gain >= water_min_exg_gain
                and object_exg >= water_min_object_exg
                and object_chroma >= water_min_object_chroma
                and (not specular_water_context or object_chroma >= 0.04)
            )
            # Shadowed mud can fall just outside the neutral-water chroma
            # interval. Weakly green texture with almost no saturation gain
            # is not a leaf, even when it borrows a nearby candidate's support.
            shadow_mud_reject = bool(
                ring_sat >= 40 and ring_chroma < -0.03
                and neutral_water_min_exg <= ring_exg <= neutral_water_max_exg
                and object_chroma < 0.03 and object_exg < 35
                and sat_gain < 12
            )
            return {
                "has_ring": True,
                "water_context": water_context,
                "green_water_context": green_water_context,
                "neutral_water_context": neutral_water_context,
                "specular_water_context": specular_water_context,
                "highlight_fraction": highlight_fraction,
                "verified": verified,
                "reject": bool((water_context and not verified) or shadow_mud_reject),
                "shadow_mud_reject": shadow_mud_reject,
                "ring_saturation": ring_sat,
                "ring_exg": ring_exg,
                "ring_chroma": ring_chroma,
                "saturation_gain": sat_gain,
                "exg_gain": exg_gain,
            }

        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(seedling_pixels, 8)
        candidates: list[dict[str, Any]] = []
        rejected_existing = 0
        rejected_shape = 0
        rejected_color = 0
        micro_candidate_count = 0
        micro_rejected_shape = 0
        micro_rejected_color = 0
        micro_single_candidate_count = 0
        micro_single_rejected_color = 0
        micro_single_rejected_support = 0
        contextual_candidate_count = 0
        contextual_micro_candidate_count = 0

        for label_idx in range(1, num_labels):
            x, y, width, height, area = stats[label_idx].tolist()
            if area < min_area_px or area > max_area_px:
                rejected_shape += 1
                continue
            is_micro = bool(
                micro_enabled
                and area >= micro_min_area_px
                and area <= micro_max_area_px
            )
            if is_micro:
                micro_candidate_count += 1
            if width > max_bbox_px or height > max_bbox_px:
                rejected_shape += 1
                if is_micro:
                    micro_rejected_shape += 1
                    micro_candidate_count -= 1
                continue
            if is_micro and (width > micro_max_bbox_px or height > micro_max_bbox_px):
                rejected_shape += 1
                micro_rejected_shape += 1
                micro_candidate_count -= 1
                continue
            aspect = max(width, height) / max(1, min(width, height))
            if aspect > (4.0 if is_micro else 6.0):
                rejected_shape += 1
                if is_micro:
                    micro_rejected_shape += 1
                    micro_candidate_count -= 1
                continue

            component = labels[y : y + height, x : x + width] == label_idx
            if np.count_nonzero(component & duplicate_mask[y : y + height, x : x + width]) > 0:
                rejected_existing += 1
                if is_micro:
                    micro_candidate_count -= 1
                continue

            cx, cy = centroids[label_idx]
            component_sat = sat[y : y + height, x : x + width][component]
            component_val = val[y : y + height, x : x + width][component]
            component_exg = excess_green[y : y + height, x : x + width][component]
            component_chroma = green_chroma[y : y + height, x : x + width][component]
            mean_sat = float(np.mean(component_sat))
            mean_val = float(np.mean(component_val))
            mean_exg = float(np.mean(component_exg))
            mean_chroma = float(np.mean(component_chroma))
            image_h, image_w = image.shape[:2]
            center_x = int(max(0, min(image_w - 1, round(cx))))
            center_y = int(max(0, min(image_h - 1, round(cy))))
            value_gain = mean_val - float(local_value[center_y, center_x])
            exg_gain = mean_exg - float(local_exg[center_y, center_x])
            micro_value_gain = mean_val - float(micro_local_value[center_y, center_x])
            micro_exg_gain = mean_exg - float(micro_local_exg[center_y, center_x])

            absolute_strong_color = (
                mean_sat >= 55.0
                and mean_val >= 45.0
                and mean_exg >= 22.0
                and mean_chroma >= max(min_chroma, 0.012)
            )
            strong_color = absolute_strong_color and (
                exg_gain >= min_exg_gain or value_gain >= min_value_gain
            )
            pale_contrast = (
                mean_sat >= min_sat
                and mean_val >= 70.0
                and mean_exg >= 12.0
                and mean_chroma >= min_chroma
                and value_gain >= min_value_gain
                and exg_gain >= min_exg_gain
            )
            dark_mud_like = (
                mean_val < (float(local_value[center_y, center_x]) - 4.0)
                and exg_gain < (min_exg_gain + 2.0)
                and mean_sat < 75.0
            )
            micro_color = (
                mean_sat >= micro_min_sat
                and mean_val >= micro_min_val
                and mean_exg >= micro_min_exg
                and mean_chroma >= micro_min_chroma
                and max(value_gain, micro_value_gain) >= micro_min_value_gain
                and max(exg_gain, micro_exg_gain) >= micro_min_exg_gain
            )
            # Some real seedlings are weaker because a neighboring seedling
            # raises the local green average. Keep these as contextual-only
            # candidates: they may be rescued later by already-qualified
            # anchors, but can never validate one another.
            contextual_color = (
                not dark_mud_like
                and absolute_strong_color
                and (
                    max(value_gain, micro_value_gain) >= 1.0
                    or max(exg_gain, micro_exg_gain) >= 2.0
                )
            )
            contextual_only = False
            contextual_required_neighbors = 1
            micro_support_ok = True
            if is_micro and area == 1:
                micro_single_candidate_count += 1
                support_pixels = int(round(float(micro_support[center_y, center_x])))
                if support_pixels < micro_min_support_pixels:
                    micro_single_rejected_support += 1
                    micro_support_ok = False

            if is_micro and not (micro_color and micro_support_ok):
                if contextual_color:
                    contextual_only = True
                    contextual_candidate_count += 1
                    contextual_micro_candidate_count += 1
                else:
                    rejected_color += 1
                    micro_rejected_color += 1
                    if area == 1:
                        micro_single_rejected_color += 1
                    continue
            if not is_micro and (dark_mud_like or not (strong_color or pale_contrast)):
                if contextual_color:
                    contextual_only = True
                    contextual_candidate_count += 1
                else:
                    rejected_color += 1
                    continue

            required_neighbors = micro_min_neighbors if is_micro else min_neighbors
            if is_micro and area == 1:
                # The local-support gate above replaces component-neighbor
                # counting for a single-pixel candidate.
                required_neighbors = 0
            if strong_color:
                # Strong, compact, locally contrasted green evidence is enough
                # on its own. Sparse fields such as the upper-right of DJI_0990
                # contain real isolated seedlings, so requiring a companion
                # systematically removes exactly the smallest plants. Weaker
                # pale candidates still retain the row/neighbor requirement.
                required_neighbors = 0
            if not is_micro and (area <= 2 or not strong_color):
                required_neighbors = max(required_neighbors, weak_min_neighbors)
            if contextual_only:
                required_neighbors = int(contextual_required_neighbors)

            candidates.append(
                {
                    "center": (float(cx), float(cy)),
                    "area": int(area),
                    "bbox": (int(x), int(y), int(width), int(height)),
                    "required_neighbors": int(required_neighbors),
                    "strong_color": bool(strong_color),
                    "is_micro": is_micro,
                    "contextual_only": bool(contextual_only),
                    "mean_saturation": mean_sat,
                    "mean_value": mean_val,
                    "mean_exg": mean_exg,
                    "mean_chroma": mean_chroma,
                }
            )

        radius_px = max(1.0, neighbor_radius_m / gsd)
        radius_sq = radius_px * radius_px
        contextual_radius_px = max(1.0, contextual_neighbor_radius_m / gsd)
        contextual_radius_sq = contextual_radius_px * contextual_radius_px
        grid_cell = max(1, int(round(radius_px)))
        centers = [candidate["center"] for candidate in candidates]

        # Pass 1: evaluate ordinary candidates against other ordinary
        # candidates.  Contextual candidates are deliberately excluded so a
        # collection of weak mud speckles cannot bootstrap itself into a row.
        ordinary_cells: dict[tuple[int, int], list[int]] = {}
        ordinary_indices = [
            idx for idx, candidate in enumerate(candidates)
            if not candidate.get("contextual_only")
        ]
        for idx in ordinary_indices:
            cx, cy = centers[idx]
            key = (int(cx // grid_cell), int(cy // grid_cell))
            ordinary_cells.setdefault(key, []).append(idx)

        supported_ordinary_indices: list[int] = []
        for idx in ordinary_indices:
            cx, cy = centers[idx]
            required_neighbors = int(candidates[idx].get("required_neighbors", min_neighbors))
            if required_neighbors <= 0:
                supported_ordinary_indices.append(idx)
                continue
            cell_x = int(cx // grid_cell)
            cell_y = int(cy // grid_cell)
            neighbors = 0
            for nx in range(cell_x - 1, cell_x + 2):
                for ny in range(cell_y - 1, cell_y + 2):
                    for other_idx in ordinary_cells.get((nx, ny), []):
                        if other_idx == idx:
                            continue
                        ox, oy = centers[other_idx]
                        dx = cx - ox
                        dy = cy - oy
                        if (dx * dx) + (dy * dy) <= radius_sq:
                            neighbors += 1
                            if neighbors >= required_neighbors:
                                break
                    if neighbors >= required_neighbors:
                        break
                if neighbors >= required_neighbors:
                    break
            if neighbors >= required_neighbors:
                supported_ordinary_indices.append(idx)

        # Pass 2: weak nearby fragments may only use already-supported
        # ordinary candidates as anchors, and only inside the compact
        # contextual radius.  They never support one another.
        contextual_grid_cell = max(1, int(round(contextual_radius_px)))
        anchor_cells: dict[tuple[int, int], list[int]] = {}
        for idx in supported_ordinary_indices:
            cx, cy = centers[idx]
            key = (
                int(cx // contextual_grid_cell),
                int(cy // contextual_grid_cell),
            )
            anchor_cells.setdefault(key, []).append(idx)

        supported_contextual_indices: list[int] = []
        for idx, candidate in enumerate(candidates):
            if not candidate.get("contextual_only"):
                continue
            cx, cy = centers[idx]
            required_neighbors = int(candidate.get("required_neighbors", 1))
            cell_x = int(cx // contextual_grid_cell)
            cell_y = int(cy // contextual_grid_cell)
            neighbors = 0
            for nx in range(cell_x - 1, cell_x + 2):
                for ny in range(cell_y - 1, cell_y + 2):
                    for anchor_idx in anchor_cells.get((nx, ny), []):
                        ox, oy = centers[anchor_idx]
                        dx = cx - ox
                        dy = cy - oy
                        if (dx * dx) + (dy * dy) <= contextual_radius_sq:
                            neighbors += 1
                            if neighbors >= required_neighbors:
                                break
                    if neighbors >= required_neighbors:
                        break
                if neighbors >= required_neighbors:
                    break
            if neighbors >= required_neighbors:
                supported_contextual_indices.append(idx)

        supported_indices = supported_ordinary_indices + supported_contextual_indices

        supported_index_set = set(supported_indices)

        # One seedling frequently produces several disconnected green leaf
        # components.  Link only immediate (roughly one-pixel at DJI_0990's
        # GSD) fragments and render one marker at the original green-pixel
        # centroid.  This removes duplicate leaf markers without merging
        # genuinely separate nearby seedlings.
        if fragment_link_m > 0:
            fragment_kernel_size = self._metric_kernel_size(
                fragment_link_m,
                gsd,
                fallback_px=3,
            )
            fragment_kernel = cv2.getStructuringElement(
                cv2.MORPH_ELLIPSE,
                (fragment_kernel_size, fragment_kernel_size),
            )
            linked_seedling_pixels = cv2.dilate(
                seedling_pixels,
                fragment_kernel,
                iterations=1,
            )
        else:
            fragment_kernel_size = 1
            linked_seedling_pixels = seedling_pixels.copy()

        (
            linked_label_count,
            linked_labels,
            linked_stats,
            _,
        ) = cv2.connectedComponentsWithStats(linked_seedling_pixels, 8)

        represented_clusters: dict[int, list[int]] = {}
        unmapped_supported_indices: list[int] = []
        for idx in supported_indices:
            cx, cy = centers[idx]
            center_x = int(max(0, min(image.shape[1] - 1, round(cx))))
            center_y = int(max(0, min(image.shape[0] - 1, round(cy))))
            linked_label = int(linked_labels[center_y, center_x])
            if linked_label > 0:
                represented_clusters.setdefault(linked_label, []).append(idx)
            else:
                unmapped_supported_indices.append(idx)

        cluster_max_area_px = max(
            max_area_px,
            int(round(cluster_max_area_m2 / (gsd ** 2))),
        )
        cluster_max_bbox_px = max(
            max_bbox_px,
            int(round(cluster_max_bbox_m / gsd)),
        )
        linked_max_area_px = max(
            cluster_max_area_px,
            int(round(linked_max_area_m2 / (gsd ** 2))),
        )
        linked_max_bbox_px = max(
            cluster_max_bbox_px,
            int(round(linked_max_bbox_m / gsd)),
        )
        accepted_entries: list[dict[str, Any]] = []
        cluster_rescue_count = 0
        cluster_rescue_rejected_existing = 0
        cluster_rescue_rejected_shape = 0
        cluster_rescue_rejected_color = 0
        linked_rejected_shape = 0
        water_context_candidate_count = 0
        water_context_verified_count = 0
        water_context_rejected_count = 0
        water_context_ring_skipped = 0
        neutral_water_context_candidate_count = 0
        neutral_water_context_verified_count = 0
        neutral_water_context_rejected_count = 0
        specular_water_context_candidate_count = 0
        specular_water_context_rejected_count = 0
        shadow_mud_rejected_count = 0

        def record_water_context(metrics: dict[str, Any]) -> bool:
            """Update final-cluster water counters and return rejection state."""
            nonlocal water_context_candidate_count
            nonlocal water_context_verified_count
            nonlocal water_context_rejected_count
            nonlocal water_context_ring_skipped
            nonlocal neutral_water_context_candidate_count
            nonlocal neutral_water_context_verified_count
            nonlocal neutral_water_context_rejected_count
            nonlocal specular_water_context_candidate_count
            nonlocal specular_water_context_rejected_count
            nonlocal shadow_mud_rejected_count

            rejected = bool(metrics.get("reject"))
            shadow_mud_rejected_count += int(metrics.get("shadow_mud_reject", False))
            if not metrics.get("has_ring"):
                water_context_ring_skipped += 1
            if metrics.get("water_context"):
                water_context_candidate_count += 1
                if rejected:
                    water_context_rejected_count += 1
                else:
                    water_context_verified_count += 1
            if metrics.get("neutral_water_context"):
                neutral_water_context_candidate_count += 1
                if rejected:
                    neutral_water_context_rejected_count += 1
                else:
                    neutral_water_context_verified_count += 1
            if metrics.get("specular_water_context"):
                specular_water_context_candidate_count += 1
                if rejected:
                    specular_water_context_rejected_count += 1
            return rejected

        for linked_label in range(1, linked_label_count):
            linked_x, linked_y, linked_w, linked_h, _ = linked_stats[linked_label].tolist()
            linked_component = (
                linked_labels[
                    linked_y : linked_y + linked_h,
                    linked_x : linked_x + linked_w,
                ]
                == linked_label
            )
            original_component = linked_component & (
                seedling_pixels[
                    linked_y : linked_y + linked_h,
                    linked_x : linked_x + linked_w,
                ]
                > 0
            )
            original_y, original_x = np.nonzero(original_component)
            if original_x.size == 0:
                continue

            global_x = original_x + linked_x
            global_y = original_y + linked_y
            cluster_center = (float(np.mean(global_x)), float(np.mean(global_y)))
            cluster_area = int(original_x.size)
            cluster_width = int(np.max(original_x) - np.min(original_x) + 1)
            cluster_height = int(np.max(original_y) - np.min(original_y) + 1)
            cluster_aspect = max(cluster_width, cluster_height) / max(
                1,
                min(cluster_width, cluster_height),
            )
            cluster_sat = float(np.mean(sat[global_y, global_x]))
            cluster_val = float(np.mean(val[global_y, global_x]))
            cluster_exg = float(np.mean(excess_green[global_y, global_x]))
            cluster_chroma = float(np.mean(green_chroma[global_y, global_x]))

            represented = represented_clusters.get(linked_label)
            if represented:
                if (
                    cluster_area > linked_max_area_px
                    or cluster_width > linked_max_bbox_px
                    or cluster_height > linked_max_bbox_px
                    or cluster_aspect > 12.0
                ):
                    linked_rejected_shape += 1
                    continue

                # Prefer an ordinary strong component as the representative;
                # contextual fragments remain useful only when no ordinary
                # component in this physical plant survived.
                representative_idx = max(
                    represented,
                    key=lambda candidate_idx: (
                        not bool(candidates[candidate_idx].get("contextual_only")),
                        bool(candidates[candidate_idx].get("strong_color")),
                        int(candidates[candidate_idx].get("area", 0)),
                    ),
                )
                context_metrics = water_context_metrics(
                    cluster_center,
                    cluster_sat,
                    cluster_exg,
                    cluster_chroma,
                )
                if record_water_context(context_metrics):
                    continue
                accepted_entries.append(
                    {
                        "center": cluster_center,
                        "is_micro": bool(candidates[representative_idx].get("is_micro")),
                        "contextual_only": bool(
                            candidates[representative_idx].get("contextual_only")
                        ),
                        "cluster_rescue": False,
                        "mean_chroma": cluster_chroma,
                        "water_context": bool(
                            context_metrics.get("water_context")
                        ),
                    }
                )
                continue

            if (
                cluster_area < cluster_min_pixels
                or cluster_area > cluster_max_area_px
                or cluster_width > cluster_max_bbox_px
                or cluster_height > cluster_max_bbox_px
                or cluster_aspect > 6.0
            ):
                cluster_rescue_rejected_shape += 1
                continue

            if np.count_nonzero(
                original_component
                & duplicate_mask[
                    linked_y : linked_y + linked_h,
                    linked_x : linked_x + linked_w,
                ]
            ) > 0:
                cluster_rescue_rejected_existing += 1
                continue

            strict_cluster_color = (
                cluster_sat >= 55.0
                and cluster_val >= 45.0
                and cluster_exg >= 22.0
                and cluster_chroma >= max(min_chroma, 0.012)
            )
            if not strict_cluster_color:
                cluster_rescue_rejected_color += 1
                continue

            context_metrics = water_context_metrics(
                cluster_center,
                cluster_sat,
                cluster_exg,
                cluster_chroma,
            )
            if record_water_context(context_metrics):
                continue
            accepted_entries.append(
                {
                    "center": cluster_center,
                    "is_micro": bool(cluster_area <= micro_max_area_px),
                    "contextual_only": False,
                    "cluster_rescue": True,
                    "mean_chroma": cluster_chroma,
                    "water_context": bool(
                        context_metrics.get("water_context")
                    ),
                }
            )
            cluster_rescue_count += 1

        for idx in unmapped_supported_indices:
            candidate = candidates[idx]
            context_metrics = water_context_metrics(
                centers[idx],
                float(candidate.get("mean_saturation", 0.0)),
                float(candidate.get("mean_exg", 0.0)),
                float(candidate.get("mean_chroma", 0.0)),
            )
            if record_water_context(context_metrics):
                continue
            accepted_entries.append(
                {
                    "center": centers[idx],
                    "is_micro": bool(candidate.get("is_micro")),
                    "contextual_only": bool(candidate.get("contextual_only")),
                    "cluster_rescue": False,
                    "mean_chroma": float(candidate.get("mean_chroma", 0.0)),
                    "water_context": bool(
                        context_metrics.get("water_context")
                    ),
                }
            )

        represented_supported_count = len(represented_clusters) + len(
            unmapped_supported_indices
        )
        fragment_duplicates_removed = max(
            0,
            len(supported_indices) - represented_supported_count,
        )

        # Apply after every rescue/grouping path, so a stick fragment cannot
        # bypass this check by borrowing weak neighbors or cluster membership.
        low_leaf_rejected_count = 0
        if low_leaf_evidence:
            retained_entries = []
            for entry in accepted_entries:
                if entry["mean_chroma"] < 0.03:
                    low_leaf_rejected_count += 1
                    cluster_rescue_count -= int(entry.get("cluster_rescue", False))
                else:
                    retained_entries.append(entry)
            accepted_entries = retained_entries

        yellow_metadata = {"seedling_yellow_recovery_count": 0}
        if not low_leaf_evidence:
            recovered, yellow_metadata = recover_yellow_leaf_clusters(
                image, duplicate_mask, [entry["center"] for entry in accepted_entries], gsd,
            )
            accepted_entries.extend(recovered)

        supplement_mask = np.zeros_like(base_mask)
        marker_radius_px = max(2, int(round(marker_radius_m / gsd)))
        for entry in accepted_entries:
            cx, cy = entry["center"]
            cv2.circle(
                supplement_mask,
                (int(round(cx)), int(round(cy))),
                marker_radius_px,
                255,
                thickness=-1,
            )

        return supplement_mask, {
            "seedling_supplement": True,
            "seedling_supplement_count": int(len(accepted_entries)),
            "seedling_supplement_pixels": int(np.count_nonzero(supplement_mask)),
            "seedling_candidate_count": int(len(candidates)),
            "seedling_leaf_core_fraction": leaf_core_fraction,
            "seedling_low_leaf_evidence": low_leaf_evidence,
            "seedling_low_leaf_evidence_rejected_count": low_leaf_rejected_count,
            "seedling_shadow_mud_rejected_count": shadow_mud_rejected_count,
            **yellow_metadata,
            "seedling_rejected_existing": int(rejected_existing),
            "seedling_rejected_shape": int(rejected_shape),
            "seedling_rejected_color": int(rejected_color),
            "seedling_rejected_neighbor": int(len(candidates) - len(supported_indices)),
            "seedling_neighbor_radius_m": neighbor_radius_m,
            "seedling_contextual_neighbor_radius_m": contextual_neighbor_radius_m,
            "seedling_min_neighbors": int(min_neighbors),
            "seedling_weak_min_neighbors": int(weak_min_neighbors),
            "seedling_marker_radius_m": marker_radius_m,
            "seedling_fragment_link_m": fragment_link_m,
            "seedling_fragment_kernel_px": int(fragment_kernel_size),
            "seedling_fragment_duplicates_removed": int(fragment_duplicates_removed),
            "seedling_cluster_rescue_count": int(cluster_rescue_count),
            "seedling_cluster_rescue_rejected_existing": int(
                cluster_rescue_rejected_existing
            ),
            "seedling_cluster_rescue_rejected_shape": int(
                cluster_rescue_rejected_shape
            ),
            "seedling_cluster_rescue_rejected_color": int(
                cluster_rescue_rejected_color
            ),
            "seedling_cluster_max_area_m2": float(cluster_max_area_m2),
            "seedling_cluster_max_bbox_m": float(cluster_max_bbox_m),
            "seedling_linked_rejected_shape": int(linked_rejected_shape),
            "seedling_linked_max_area_m2": float(linked_max_area_m2),
            "seedling_linked_max_bbox_m": float(linked_max_bbox_m),
            "seedling_water_context_rejection": bool(water_context_rejection),
            "seedling_water_context_candidate_count": int(
                water_context_candidate_count
            ),
            "seedling_water_context_verified_count": int(
                water_context_verified_count
            ),
            "seedling_water_context_rejected_count": int(
                water_context_rejected_count
            ),
            "seedling_water_context_ring_skipped": int(
                water_context_ring_skipped
            ),
            "seedling_specular_water_context_candidate_count": int(
                specular_water_context_candidate_count
            ),
            "seedling_specular_water_context_rejected_count": int(
                specular_water_context_rejected_count
            ),
            "seedling_neutral_water_context_candidate_count": int(
                neutral_water_context_candidate_count
            ),
            "seedling_neutral_water_context_verified_count": int(
                neutral_water_context_verified_count
            ),
            "seedling_neutral_water_context_rejected_count": int(
                neutral_water_context_rejected_count
            ),
            "seedling_water_context_inner_radius_m": float(
                water_context_inner_radius_m
            ),
            "seedling_water_context_outer_radius_m": float(
                water_context_outer_radius_m
            ),
            "seedling_neutral_water_min_saturation": float(
                neutral_water_min_sat
            ),
            "seedling_neutral_water_min_exg": float(neutral_water_min_exg),
            "seedling_neutral_water_max_exg": float(neutral_water_max_exg),
            "seedling_neutral_water_min_chroma": float(
                neutral_water_min_chroma
            ),
            "seedling_neutral_water_max_chroma": float(
                neutral_water_max_chroma
            ),
            "seedling_micro_candidate_count": int(micro_candidate_count),
            "seedling_micro_rejected_shape": int(micro_rejected_shape),
            "seedling_micro_rejected_color": int(micro_rejected_color),
            "seedling_micro_supplement_count": int(
                sum(1 for entry in accepted_entries if entry.get("is_micro"))
            ),
            "seedling_micro_single_candidate_count": int(micro_single_candidate_count),
            "seedling_micro_single_rejected_color": int(micro_single_rejected_color),
            "seedling_micro_single_rejected_support": int(micro_single_rejected_support),
            "seedling_micro_min_area_m2": float(micro_min_area_m2),
            "seedling_contextual_candidate_count": int(contextual_candidate_count),
            "seedling_contextual_micro_candidate_count": int(
                contextual_micro_candidate_count
            ),
            "seedling_contextual_supplement_count": int(
                sum(1 for entry in accepted_entries if entry.get("contextual_only"))
            ),
            "seedling_contextual_rejected_neighbor": int(
                sum(
                    1
                    for idx, candidate in enumerate(candidates)
                    if candidate.get("contextual_only") and idx not in supported_index_set
                )
            ),
            "seedling_buffer_m": self.runtime_tuning.get("seedling_buffer_m"),
            "seedling_accepted_centers": [
                [float(entry["center"][0]), float(entry["center"][1])]
                for entry in accepted_entries
            ],
            "seedling_contextual_accepted_centers": [
                [float(entry["center"][0]), float(entry["center"][1])]
                for entry in accepted_entries
                if entry.get("contextual_only")
            ],
        }

    def _detect_hybrid_seedling_supplement_mask(
        self,
        image: np.ndarray,
        base_mask: np.ndarray,
        gsd: Optional[float],
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """Run the isolated learned seedling branch without altering base crowns."""
        empty_mask = np.zeros_like(base_mask)
        if not bool(self.runtime_tuning.get("seedling_supplement", True)):
            return empty_mask, {
                "seedling_detection_mode": "disabled",
                "seedling_classifier_available": False,
                "seedling_supplement_count": 0,
                "seedling_supplement_pixels": 0,
            }
        if gsd is None or gsd <= 0:
            return empty_mask, {
                "seedling_detection_mode": "fallback_legacy",
                "seedling_classifier_available": False,
                "seedling_classifier_error": "valid GSD is required",
            }

        config = SeedlingDetectorConfig.from_mapping(self.runtime_tuning)
        detector = SeedlingDetector(config=config, device=self.device)
        seedling_mask, _, metadata = detector.detect(image, base_mask, float(gsd))
        return seedling_mask, metadata
    
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
        rescued_low_confidence = 0
        rescued_shadow_crowns = 0
        rejected_non_canopy_class = 0
        rejected_non_green = 0
        rejected_low_saturation_detections = 0
        rejected_too_small = 0
        rejected_too_large = 0
        rejected_invalid_geometry = 0
        class_candidate_counts: Dict[int, int] = {}
        class_kept_counts: Dict[int, int] = {}
        min_crown_m2 = float(self.runtime_tuning.get("min_crown_m2", 0.05))
        max_crown_m2 = float(self.runtime_tuning.get("max_crown_m2", 250.0))
        strict_canopy_hsv = bool(self.runtime_tuning.get("strict_canopy_hsv", True))
        detection_veg_min_ratio = float(self.runtime_tuning.get("detection_veg_min_ratio", 0.03))
        strict_green_mask = self._detect_strict_canopy_green_hsv(image) if strict_canopy_hsv else None
        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        saturation_image = hsv_image[:, :, 1].astype(np.float32)
        bgr_image = image.astype(np.float32)
        excess_green_image = (
            (2.0 * bgr_image[:, :, 1]) - bgr_image[:, :, 2] - bgr_image[:, :, 0]
        )
        recall_confidence_floor = float(self.runtime_tuning.get("recall_confidence_floor", 0.75))
        recall_veg_min_ratio = float(self.runtime_tuning.get("recall_veg_min_ratio", 0.85))
        recall_min_saturation = float(self.runtime_tuning.get("recall_min_saturation", 80.0))
        recall_min_exg = float(self.runtime_tuning.get("recall_min_exg", 35.0))
        color_gate_confidence = float(self.runtime_tuning.get("color_gate_confidence", 0.87))
        shadow_recall_confidence_floor = float(
            self.runtime_tuning.get("shadow_recall_confidence_floor", 0.60)
        )
        shadow_recall_max_area_m2 = float(self.runtime_tuning.get("shadow_recall_max_area_m2", 1.20))
        shadow_recall_min_veg_ratio = float(
            self.runtime_tuning.get("shadow_recall_min_veg_ratio", 0.35)
        )
        shadow_recall_min_saturation = float(
            self.runtime_tuning.get("shadow_recall_min_saturation", 90.0)
        )
        shadow_recall_min_exg = float(self.runtime_tuning.get("shadow_recall_min_exg", 15.0))
        reject_low_saturation = bool(self.runtime_tuning.get("reject_low_saturation_components", True))
        low_sat_max = float(self.runtime_tuning.get("water_component_max_saturation", 60.0))
        low_exg_max = float(self.runtime_tuning.get("water_component_max_exg", 28.0))
        low_green_ratio_max = float(self.runtime_tuning.get("water_component_max_green_ratio", 0.80))
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
            classes = (
                instances.pred_classes.numpy()
                if instances.has("pred_classes")
                else np.zeros(len(scores), dtype=np.int64)
            )

            # Store each detection with global coordinates and confidence.
            for i in range(len(scores)):
                total_model_detections += 1
                class_id = int(classes[i]) if i < len(classes) else 0
                class_candidate_counts[class_id] = class_candidate_counts.get(class_id, 0) + 1
                if class_id not in self.canopy_class_ids:
                    rejected_non_canopy_class += 1
                    continue
                score = float(scores[i])
                score_below_threshold = score < float(self.confidence_threshold)
                score_below_color_gate = score < max(float(self.confidence_threshold), color_gate_confidence)
                min_recall_floor = min(recall_confidence_floor, shadow_recall_confidence_floor)
                if score_below_threshold and score < min_recall_floor:
                    below_confidence_detections += 1
                    continue

                mask = masks[i].astype(np.uint8)
                if int(np.count_nonzero(mask)) <= 0:
                    continue

                # Find contours
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                kept_candidate = False

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

                    saturation_tile = saturation_image[y1:y2, x1:x2]
                    excess_green_tile = excess_green_image[y1:y2, x1:x2]
                    detection_pixels_bool = local_detection_mask > 0
                    mean_saturation = float(np.mean(saturation_tile[detection_pixels_bool]))
                    mean_exg = float(np.mean(excess_green_tile[detection_pixels_bool]))
                    detection_area_m2 = (
                        detection_pixels * (gsd_used ** 2)
                        if gsd_used is not None
                        else 0.0
                    )

                    if (
                        reject_low_saturation
                        and mean_saturation <= low_sat_max
                        and mean_exg <= low_exg_max
                        and vegetation_ratio <= low_green_ratio_max
                    ):
                        rejected_low_saturation_detections += 1
                        continue

                    if score_below_color_gate:
                        strong_leaf_evidence = (
                            vegetation_ratio >= recall_veg_min_ratio
                            and mean_saturation >= recall_min_saturation
                            and mean_exg >= recall_min_exg
                        )
                        shadow_crown_evidence = (
                            not strong_leaf_evidence
                            and score >= shadow_recall_confidence_floor
                            and detection_area_m2 <= shadow_recall_max_area_m2
                            and vegetation_ratio >= shadow_recall_min_veg_ratio
                            and mean_saturation >= shadow_recall_min_saturation
                            and mean_exg >= shadow_recall_min_exg
                        )
                        if not (strong_leaf_evidence or shadow_crown_evidence):
                            continue
                    else:
                        shadow_crown_evidence = False

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
                                'score': score,
                                'class_id': class_id,
                                'contour': contour_global,
                                'vegetation_ratio': vegetation_ratio,
                                'mean_saturation': mean_saturation,
                                'mean_exg': mean_exg,
                                'rescued_low_confidence': score_below_threshold,
                                'color_gated': bool(score_below_color_gate),
                                'rescued_shadow_crown': bool(score_below_color_gate and shadow_crown_evidence),
                            })
                            kept_candidate = True
                            if score_below_threshold:
                                rescued_low_confidence += 1
                            if score_below_color_gate and shadow_crown_evidence:
                                rescued_shadow_crowns += 1
                            class_kept_counts[class_id] = class_kept_counts.get(class_id, 0) + 1
                        except Exception:
                            rejected_invalid_geometry += 1
                            continue

                if score_below_threshold and not kept_candidate:
                    below_confidence_detections += 1

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
        if rescued_low_confidence:
            print(f"   Rescued {rescued_low_confidence} below-threshold detections with strong canopy color")
        if rescued_shadow_crowns:
            print(f"   Rescued {rescued_shadow_crowns} compact shadow crown detections")
        if rejected_non_canopy_class:
            print(f"   Rejected {rejected_non_canopy_class} non-canopy class detections")
        if rejected_non_green:
            print(f"   Rejected {rejected_non_green} non-green AI detections")
        if rejected_low_saturation_detections:
            print(f"   Rejected {rejected_low_saturation_detections} drab water/structure-like detections")
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

        # Keep the mature-canopy result immutable.  Every seedling mode carries
        # its accepted markers separately so canopy morphology and water
        # filtering cannot erase tiny candidates drawn over mud.
        base_instance_mask = self._polygons_to_mask(final_polygons, (h, w))
        instance_mask = base_instance_mask
        seedling_mask = np.zeros_like(base_instance_mask)
        seedling_metadata: Dict[str, Any]
        seedling_mode = str(self.runtime_tuning.get("seedling_detection_mode", "hybrid")).lower()
        if seedling_mode == "legacy":
            legacy_seedling_mask, seedling_metadata = self._detect_seedling_supplement_mask(
                image=image,
                base_mask=base_instance_mask,
                gsd=gsd_used,
            )
            # Keep legacy supplement pixels separate from the mature-canopy
            # mask.  Merging a tiny marker into the canopy mask causes the
            # later morphology and low-saturation water filter to erase or
            # reshape it, especially when the marker is drawn over mud.
            seedling_mask = legacy_seedling_mask.copy()
            seedling_metadata["seedling_detection_mode"] = "legacy"
        elif seedling_mode in {"off", "disabled"}:
            seedling_metadata = {
                "seedling_detection_mode": "disabled",
                "seedling_classifier_available": False,
                "seedling_supplement_count": 0,
                "seedling_supplement_pixels": 0,
            }
        else:
            seedling_mask, seedling_metadata = self._detect_hybrid_seedling_supplement_mask(
                image=image,
                base_mask=base_instance_mask,
                gsd=gsd_used,
            )
            if seedling_metadata.get("seedling_detection_mode") == "fallback_legacy":
                # A missing/unloadable checkpoint must not remove the current
                # seedling behavior.  It also must not change the mature mask.
                legacy_seedling_mask, legacy_metadata = self._detect_seedling_supplement_mask(
                    image=image,
                    base_mask=base_instance_mask,
                    gsd=gsd_used,
                )
                # The fallback supplement is also kept separate so the
                # accepted tiny candidates bypass mature-canopy morphology
                # and water filtering while still contributing to safety
                # buffers through the returned seedling mask.
                seedling_mask = legacy_seedling_mask.copy()
                seedling_metadata.update(legacy_metadata)
                seedling_metadata["seedling_detection_mode"] = "fallback_legacy"
                seedling_metadata["seedling_fallback_legacy_count"] = int(
                    seedling_metadata.get("seedling_supplement_count", 0) or 0
                )

        # The separate seedling mask is used by planting safety geometry and
        # visualization, but is not merged into mature-canopy area.
        self.last_seedling_mask = seedling_mask.copy()

        coverage_polygons, combined_mask, merge_metadata = self._merge_fragmented_canopy_mask(
            instance_mask=instance_mask,
            strict_green_mask=strict_green_mask,
            gsd=gsd_used,
            min_area_px=min_area_px,
        )
        combined_mask, component_filter_metadata = self._filter_low_saturation_components(
            canopy_mask=combined_mask,
            image=image,
            strict_green_mask=strict_green_mask,
        )
        if component_filter_metadata.get("rejected_low_saturation_components", 0):
            # Re-extract only at the mature-canopy floor. Seedling markers are
            # kept in seedling_mask and do not belong in canopy polygons.
            coverage_polygons = self._mask_to_polygons(combined_mask, float(min_area_px))

        print(f"   ✅ {len(final_polygons)} AI instances after cleanup")
        if merge_metadata.get("merge_canopy_fragments"):
            print(
                f"   Coverage merge produced {len(coverage_polygons)} canopy components "
                f"(+{merge_metadata.get('canopy_merge_added_pixels', 0)} px)"
            )
        if component_filter_metadata.get("rejected_low_saturation_components", 0):
            print(
                f"   Removed {component_filter_metadata['rejected_low_saturation_components']} "
                "low-saturation water/structure component(s)"
            )
        if seedling_metadata.get("seedling_supplement_count", 0):
            print(
                f"   Added {seedling_metadata['seedling_supplement_count']} "
                "tiny seedling supplement detections"
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
            'rescued_low_confidence_detections': int(rescued_low_confidence),
            'rescued_shadow_crown_detections': int(rescued_shadow_crowns),
            'raw_detections': len(all_instances),
            'total_ai_detections': len(all_instances),
            'instance_tree_count': len(final_polygons),
            'final_trees': len(coverage_polygons),
            'coverage_component_count': len(coverage_polygons),
            'num_detected_canopies': len(coverage_polygons),
            'class_names': self.class_names,
            'canopy_class_ids': sorted(self.canopy_class_ids),
            'class_candidate_counts': {str(k): int(v) for k, v in sorted(class_candidate_counts.items())},
            'class_kept_counts': {str(k): int(v) for k, v in sorted(class_kept_counts.items())},
            'rejected_non_canopy_class_detections': int(rejected_non_canopy_class),
            'rejected_non_green_detections': int(rejected_non_green),
            'rejected_low_saturation_detections': int(rejected_low_saturation_detections),
            'rejected_too_small_detections': int(rejected_too_small),
            'rejected_too_large_detections': int(rejected_too_large),
            'rejected_invalid_geometry': int(rejected_invalid_geometry),
            'strict_canopy_hsv': strict_canopy_hsv,
            'detection_veg_min_ratio': detection_veg_min_ratio,
            'recall_confidence_floor': recall_confidence_floor,
            'recall_veg_min_ratio': recall_veg_min_ratio,
            'recall_min_saturation': recall_min_saturation,
            'recall_min_exg': recall_min_exg,
            'color_gate_confidence': color_gate_confidence,
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
        metadata.update(component_filter_metadata)
        metadata.update(seedling_metadata)
         
        # The fourth return value is optional for backward compatibility.  It
        # contains only isolated hybrid seedling markers; the first two values
        # remain the mature/legacy coverage outputs expected by callers.
        return coverage_polygons, combined_mask, metadata, seedling_mask

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
