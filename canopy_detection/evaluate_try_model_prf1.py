"""
Evaluate the active MangroVision try_model checkpoint with instance-level
precision, recall, and F1 at IoU=0.50 across confidence thresholds.

This script treats the held-out Roboflow COCO test split as a general
mangrove-canopy detection dataset. The active checkpoint's metadata declares
class index 1 as the canopy class, so predictions are filtered to class 1.

Run from the project root with the venv active:

    python canopy_detection/evaluate_try_model_prf1.py
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Iterable

import cv2
import numpy as np
import torch
from detectron2 import model_zoo
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from pycocotools.coco import COCO


ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = ROOT / "MangroVision_New" / "try_model" / "model_final.pth"
METADATA_PATH = ROOT / "MangroVision_New" / "try_model" / "model_metadata.json"
DATASET_ROOT = ROOT / "MangroVision_New" / "Practice_annotate.v1-1.coco-segmentation"
TEST_JSON = DATASET_ROOT / "test" / "_annotations.coco.json"
TEST_IMAGE_ROOT = DATASET_ROOT / "test"
OUTPUT_DIR = ROOT / "train_outputs" / "try_model_prf1"
OUTPUT_JSON = OUTPUT_DIR / "metrics_iou50_thresholds_test.json"
CONFIDENCE_THRESHOLDS = (0.50, 0.85, 0.87, 0.90)
BASE_SCORE_THRESHOLD = min(CONFIDENCE_THRESHOLDS)
IOU_THRESHOLD = 0.50


def load_canopy_class_ids() -> list[int]:
    with open(METADATA_PATH, "r", encoding="utf-8") as fh:
        metadata = json.load(fh)
    return [int(class_id) for class_id in metadata.get("canopy_class_ids", [1])]


def build_cfg() -> object:
    cfg = get_cfg()
    cfg.merge_from_file(
        model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
        )
    )
    cfg.MODEL.WEIGHTS = str(MODEL_PATH)
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = BASE_SCORE_THRESHOLD
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000
    cfg.MODEL.RPN.NMS_THRESH = 0.6
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.INPUT.FORMAT = "BGR"
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 512
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    return cfg


def coco_bbox_to_xyxy(bbox: Iterable[float]) -> np.ndarray:
    x, y, w, h = [float(v) for v in bbox]
    return np.array([x, y, x + w, y + h], dtype=np.float32)


def bbox_iou(a: np.ndarray, b: np.ndarray) -> float:
    x1 = max(float(a[0]), float(b[0]))
    y1 = max(float(a[1]), float(b[1]))
    x2 = min(float(a[2]), float(b[2]))
    y2 = min(float(a[3]), float(b[3]))
    inter_w = max(0.0, x2 - x1)
    inter_h = max(0.0, y2 - y1)
    inter = inter_w * inter_h
    area_a = max(0.0, float(a[2] - a[0])) * max(0.0, float(a[3] - a[1]))
    area_b = max(0.0, float(b[2] - b[0])) * max(0.0, float(b[3] - b[1]))
    union = area_a + area_b - inter
    return float(inter / union) if union > 0 else 0.0


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    a_bool = a.astype(bool)
    b_bool = b.astype(bool)
    inter = int(np.logical_and(a_bool, b_bool).sum())
    union = int(np.logical_or(a_bool, b_bool).sum())
    return float(inter / union) if union > 0 else 0.0


def greedy_match(predictions: list[dict], gt_items: list[dict], metric: str, iou_threshold: float) -> dict:
    matched_gt: set[int] = set()
    true_positive = 0
    false_positive = 0
    matches: list[dict] = []

    for pred_index, pred in enumerate(sorted(predictions, key=lambda item: item["score"], reverse=True)):
        best_gt_index = None
        best_iou = 0.0
        for gt_index, gt in enumerate(gt_items):
            if gt_index in matched_gt:
                continue
            if metric == "mask":
                current_iou = mask_iou(pred["mask"], gt["mask"])
            else:
                current_iou = bbox_iou(pred["bbox"], gt["bbox"])
            if current_iou > best_iou:
                best_iou = current_iou
                best_gt_index = gt_index

        if best_gt_index is not None and best_iou >= iou_threshold:
            matched_gt.add(best_gt_index)
            true_positive += 1
            matches.append(
                {
                    "prediction_rank": pred_index + 1,
                    "gt_index": best_gt_index,
                    "iou": best_iou,
                    "score": pred["score"],
                }
            )
        else:
            false_positive += 1

    false_negative = len(gt_items) - true_positive
    precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) else 0.0
    recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "matches": matches,
    }


def main() -> int:
    if not MODEL_PATH.exists():
        raise FileNotFoundError(MODEL_PATH)
    if not TEST_JSON.exists():
        raise FileNotFoundError(TEST_JSON)

    canopy_class_ids = set(load_canopy_class_ids())
    cfg = build_cfg()
    predictor = DefaultPredictor(cfg)
    coco = COCO(str(TEST_JSON))

    summaries = {
        f"{threshold:.2f}": {
            "ground_truth_instances": 0,
            "predicted_instances": 0,
            "bbox": {"tp": 0, "fp": 0, "fn": 0},
            "segm": {"tp": 0, "fp": 0, "fn": 0},
        }
        for threshold in CONFIDENCE_THRESHOLDS
    }
    per_image = {f"{threshold:.2f}": [] for threshold in CONFIDENCE_THRESHOLDS}

    for image_id in coco.getImgIds():
        image_info = coco.loadImgs([image_id])[0]
        image_path = TEST_IMAGE_ROOT / image_info["file_name"]
        image = cv2.imread(str(image_path))
        if image is None:
            raise FileNotFoundError(image_path)

        ann_ids = coco.getAnnIds(imgIds=[image_id], iscrowd=None)
        annotations = coco.loadAnns(ann_ids)
        gt_items = [
            {
                "bbox": coco_bbox_to_xyxy(ann["bbox"]),
                "mask": coco.annToMask(ann).astype(np.uint8),
            }
            for ann in annotations
        ]

        with torch.no_grad():
            outputs = predictor(image)
        instances = outputs["instances"].to("cpu")
        classes = instances.pred_classes.numpy() if instances.has("pred_classes") else np.zeros(len(instances), dtype=np.int64)
        scores = instances.scores.numpy() if instances.has("scores") else np.zeros(len(instances), dtype=np.float32)
        masks = instances.pred_masks.numpy() if instances.has("pred_masks") else []
        boxes = instances.pred_boxes.tensor.numpy() if instances.has("pred_boxes") else []

        predictions = []
        for idx, class_id in enumerate(classes):
            if int(class_id) not in canopy_class_ids:
                continue
            predictions.append(
                {
                    "score": float(scores[idx]),
                    "bbox": boxes[idx].astype(np.float32),
                    "mask": masks[idx].astype(np.uint8),
                }
            )

        for threshold in CONFIDENCE_THRESHOLDS:
            threshold_key = f"{threshold:.2f}"
            threshold_predictions = [
                prediction
                for prediction in predictions
                if prediction["score"] >= threshold
            ]
            bbox_result = greedy_match(threshold_predictions, gt_items, "bbox", IOU_THRESHOLD)
            segm_result = greedy_match(threshold_predictions, gt_items, "mask", IOU_THRESHOLD)

            summary = summaries[threshold_key]
            summary["ground_truth_instances"] += len(gt_items)
            summary["predicted_instances"] += len(threshold_predictions)
            for key in ("tp", "fp", "fn"):
                summary["bbox"][key] += bbox_result[key]
                summary["segm"][key] += segm_result[key]

            per_image[threshold_key].append(
                {
                    "image_id": image_id,
                    "file_name": image_info["file_name"],
                    "ground_truth_instances": len(gt_items),
                    "predicted_instances": len(threshold_predictions),
                    "bbox": {k: bbox_result[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")},
                    "segm": {k: segm_result[k] for k in ("tp", "fp", "fn", "precision", "recall", "f1")},
                }
            )

    for summary in summaries.values():
        for metric_key in ("bbox", "segm"):
            tp = summary[metric_key]["tp"]
            fp = summary[metric_key]["fp"]
            fn = summary[metric_key]["fn"]
            precision = tp / (tp + fp) if (tp + fp) else 0.0
            recall = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
            summary[metric_key].update(
                {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                }
            )

    result = {
        "model": str(MODEL_PATH.relative_to(ROOT)),
        "dataset": str(TEST_JSON.relative_to(ROOT)),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "device": cfg.MODEL.DEVICE,
        "base_score_threshold": cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST,
        "reported_score_thresholds": list(CONFIDENCE_THRESHOLDS),
        "iou_threshold": IOU_THRESHOLD,
        "prediction_class_filter": sorted(canopy_class_ids),
        "ground_truth_interpretation": "all COCO test annotations treated as general mangrove canopy instances",
        "summary_by_score_threshold": summaries,
        "per_image": per_image,
    }

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_JSON, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2)

    print(json.dumps(result["summary_by_score_threshold"], indent=2))
    print(f"Wrote {OUTPUT_JSON}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
