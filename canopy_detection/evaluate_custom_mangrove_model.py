"""
Evaluate models/custom_mangrove_model/model_final.pth against the
Practice_annotate.v1-1.coco-segmentation dataset, then write the real
validation and test metrics into models/custom_mangrove_model/model_metadata.json.

The Roboflow export uses two COCO categories:
    id 0 = "Mangrove-Canopy"   (super-category placeholder, no annotations)
    id 1 = "Bungalon Canopy"   (the only category with annotations)

The deployed model treats class index 0 as "Mangrove canopy" (and class 1 as
"Non mangrove", which has no test annotations here). We register the dataset
with a single thing_class — "Mangrove canopy" — and remap any annotation
whose category_id is 1 to match. The evaluation then directly compares the
model's class-0 predictions against the canopy annotations.

Run from project root with the venv active:

    python canopy_detection/evaluate_custom_mangrove_model.py
"""

import json
import sys
from datetime import datetime
from pathlib import Path

import torch
from detectron2.config import get_cfg
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultPredictor
from detectron2.evaluation import COCOEvaluator, inference_on_dataset
from detectron2.data import build_detection_test_loader
from detectron2 import model_zoo


_ROOT = Path(__file__).resolve().parent.parent
_MODEL_PATH = _ROOT / "models" / "custom_mangrove_model" / "model_final.pth"
_METADATA_PATH = _ROOT / "models" / "custom_mangrove_model" / "model_metadata.json"

_DATASET_ROOT = (
    _ROOT / "MangroVision_New" / "Practice_annotate.v1-1.coco-segmentation"
)
_REMAPPED_DIR = _ROOT / "train_outputs" / "custom_mangrove_eval" / "remapped_coco"


def _remap_coco_to_single_canopy_class(src_json: Path, dst_json: Path) -> int:
    """Rewrite a Roboflow COCO file so its single annotated class becomes
    category_id 0 with name 'Mangrove canopy'. Returns the annotation count.
    """
    with open(src_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    annotations = data.get("annotations") or []
    used_ids = sorted({int(a["category_id"]) for a in annotations})
    # Always declare BOTH model classes in the COCO categories block, even
    # though only category 0 ("Mangrove canopy") has ground-truth annotations
    # in this dataset. The model emits predictions for both classes; if the
    # dataset declares only one, COCOEvaluator's category-id assertion fails
    # the moment the model outputs a "Non mangrove" prediction.
    two_class_categories = [
        {"id": 0, "name": "Mangrove canopy", "supercategory": "none"},
        {"id": 1, "name": "Non mangrove", "supercategory": "none"},
    ]

    if not annotations:
        data["categories"] = two_class_categories
        dst_json.parent.mkdir(parents=True, exist_ok=True)
        with open(dst_json, "w", encoding="utf-8") as f:
            json.dump(data, f)
        return 0

    if len(used_ids) != 1:
        raise ValueError(
            f"Expected exactly one annotated category in {src_json}, found {used_ids}"
        )

    for ann in annotations:
        ann["category_id"] = 0

    data["categories"] = two_class_categories

    dst_json.parent.mkdir(parents=True, exist_ok=True)
    with open(dst_json, "w", encoding="utf-8") as f:
        json.dump(data, f)
    return len(annotations)


def _build_cfg() -> "object":
    cfg = get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file(
        "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"
    ))
    cfg.MODEL.WEIGHTS = str(_MODEL_PATH)
    # The deployed model is loaded as 2-class by detectree2_proper.py, but only
    # class 0 (Mangrove canopy) has matching ground truth in this dataset, so
    # we evaluate as a 1-class problem. Detectron2 needs NUM_CLASSES to match
    # the checkpoint's head shape; if the checkpoint was trained 2-class it
    # will load with NUM_CLASSES=2 and we'll project predictions ourselves.
    # In practice, setting NUM_CLASSES=2 here matches the production loader.
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 2
    # SCORE_THRESH_TEST must be low (~0.05) for COCO-style AP computation,
    # NOT 0.5. AP is the area under the precision-recall curve across ALL
    # confidence thresholds; if we filter at 0.5 before COCOEvaluator sees
    # the predictions, every true positive the model emitted at confidence
    # 0.05–0.50 is silently dropped and AP collapses to 0.
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.05
    cfg.MODEL.RPN.PRE_NMS_TOPK_TEST = 6000
    cfg.MODEL.RPN.POST_NMS_TOPK_TEST = 3000
    cfg.MODEL.RPN.NMS_THRESH = 0.6
    cfg.TEST.DETECTIONS_PER_IMAGE = 1000
    cfg.INPUT.FORMAT = "BGR"
    # Match the deployed training input size (512). Without these the default
    # ResizeShortestEdge in Detectron2's test pipeline upscales 432x432 inputs
    # past 800px and the CPU dataloader runs out of memory for Mask R-CNN.
    cfg.INPUT.MIN_SIZE_TEST = 512
    cfg.INPUT.MAX_SIZE_TEST = 512
    # Single-process dataloader so no worker forks duplicate the model memory.
    cfg.DATALOADER.NUM_WORKERS = 0
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    cfg.OUTPUT_DIR = str(_ROOT / "train_outputs" / "custom_mangrove_eval")
    Path(cfg.OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    return cfg


def _evaluate_split(cfg, predictor, split_name: str) -> dict:
    loader = build_detection_test_loader(cfg, split_name)
    evaluator = COCOEvaluator(split_name, output_dir=cfg.OUTPUT_DIR)
    metrics = inference_on_dataset(predictor.model, loader, evaluator)
    return {
        "bbox": {k: float(v) for k, v in (metrics.get("bbox") or {}).items() if v is not None},
        "segm": {k: float(v) for k, v in (metrics.get("segm") or {}).items() if v is not None},
    }


def main() -> int:
    if not _MODEL_PATH.exists():
        print(f"[ERROR] Model not found: {_MODEL_PATH}", file=sys.stderr)
        return 1
    if not _DATASET_ROOT.exists():
        print(f"[ERROR] Dataset not found: {_DATASET_ROOT}", file=sys.stderr)
        return 1

    print(f"[eval] Model:   {_MODEL_PATH}")
    print(f"[eval] Dataset: {_DATASET_ROOT}")
    print(f"[eval] CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"[eval] GPU: {torch.cuda.get_device_name(0)}")

    splits = {}
    for split in ("train", "valid", "test"):
        src_json = _DATASET_ROOT / split / "_annotations.coco.json"
        if not src_json.exists():
            print(f"[ERROR] Missing {src_json}", file=sys.stderr)
            return 1
        dst_json = _REMAPPED_DIR / split / "_annotations.coco.json"
        n = _remap_coco_to_single_canopy_class(src_json, dst_json)
        splits[split] = (dst_json, n)
        print(f"[eval] Remapped {split:5s}: {n:4d} annotations -> {dst_json}")

    dataset_names = {}
    for split, (json_path, _) in splits.items():
        ds_name = f"custom_mangrove_{split}"
        if ds_name in DatasetCatalog.list():
            DatasetCatalog.remove(ds_name)
            MetadataCatalog.remove(ds_name)
        register_coco_instances(
            ds_name, {}, str(json_path), str(_DATASET_ROOT / split)
        )
        # Register both classes the model emits even though only "Mangrove
        # canopy" has ground-truth annotations in this dataset. COCO evaluator
        # asserts that every predicted class id is < len(thing_classes), so we
        # must declare both. AP-Non mangrove will simply report 0 / no support.
        MetadataCatalog.get(ds_name).thing_classes = ["Mangrove canopy", "Non mangrove"]
        dataset_names[split] = ds_name

    cfg = _build_cfg()
    predictor = DefaultPredictor(cfg)

    valid_metrics = _evaluate_split(cfg, predictor, dataset_names["valid"])
    print("\n[eval] === VALIDATION ===")
    print(f"  segm AP   = {valid_metrics['segm'].get('AP', 'n/a')}")
    print(f"  segm AP50 = {valid_metrics['segm'].get('AP50', 'n/a')}")
    print(f"  bbox AP   = {valid_metrics['bbox'].get('AP', 'n/a')}")
    print(f"  bbox AP50 = {valid_metrics['bbox'].get('AP50', 'n/a')}")

    test_metrics = _evaluate_split(cfg, predictor, dataset_names["test"])
    print("\n[eval] === TEST (held out) ===")
    print(f"  segm AP   = {test_metrics['segm'].get('AP', 'n/a')}")
    print(f"  segm AP50 = {test_metrics['segm'].get('AP50', 'n/a')}")
    print(f"  bbox AP   = {test_metrics['bbox'].get('AP', 'n/a')}")
    print(f"  bbox AP50 = {test_metrics['bbox'].get('AP50', 'n/a')}")

    if not _METADATA_PATH.exists():
        print(f"[ERROR] Metadata file missing: {_METADATA_PATH}", file=sys.stderr)
        return 1

    with open(_METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    train_json, train_count = splits["train"]
    valid_json, valid_count = splits["valid"]
    test_json, test_count = splits["test"]

    metadata["train_annotations"] = str(_DATASET_ROOT / "train" / "_annotations.coco.json")
    metadata["valid_annotations"] = str(_DATASET_ROOT / "valid" / "_annotations.coco.json")
    metadata["test_annotations"] = str(_DATASET_ROOT / "test" / "_annotations.coco.json")
    metadata["dataset_split_counts"] = {
        "train_annotations": train_count,
        "valid_annotations": valid_count,
        "test_annotations": test_count,
    }

    metadata["selected_deploy_checkpoint"] = {
        "source_checkpoint": "models/custom_mangrove_model/model_final.pth",
        "selection_basis": "field-deployment recall (no validation-time selection recorded)",
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "valid_bbox": valid_metrics["bbox"],
        "valid_segm": valid_metrics["segm"],
    }
    metadata["held_out_test_results_for_selected_checkpoint"] = {
        "checkpoint": "model_final.pth",
        "dataset": str(_DATASET_ROOT / "test" / "_annotations.coco.json"),
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "bbox": test_metrics["bbox"],
        "segm": test_metrics["segm"],
    }
    metadata["test_results"] = {
        "bbox": test_metrics["bbox"],
        "segm": test_metrics["segm"],
    }

    with open(_METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n[eval] Wrote real metrics into {_METADATA_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
