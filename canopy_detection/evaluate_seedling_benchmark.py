"""Evaluate legacy versus hybrid seedling detection on a fixed manifest.

Manifest format (JSON Lines)::

    {"image": "frames/site_a_001.JPG", "gsd": 0.010,
     "split": "test", "area_m2": 800.0,
     "seedlings": [[120, 80], [340, 210]]}

The mature-canopy model is run identically in both modes.  Only the isolated
seedling branch changes.  A prediction is a true positive when it is within
``--match-distance-m`` of one unmatched annotated seedling.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Tuple

import cv2
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT / "canopy_detection") not in sys.path:
    sys.path.insert(0, str(ROOT / "canopy_detection"))

from detectree2_proper import ProperDetectree2Detector  # noqa: E402


def _points(value: Any) -> List[Tuple[float, float]]:
    result = []
    for item in value or []:
        if isinstance(item, dict):
            result.append((float(item["x"]), float(item["y"])))
        else:
            result.append((float(item[0]), float(item[1])))
    return result


def _predicted_centers(metadata: Dict[str, Any]) -> List[Tuple[float, float]]:
    accepted = metadata.get("seedling_accepted_candidates") or []
    if accepted:
        return [(float(item["x"]), float(item["y"])) for item in accepted]
    return [(float(item[0]), float(item[1])) for item in metadata.get("seedling_accepted_centers") or []]


def _score(predicted: List[Tuple[float, float]], truth: List[Tuple[float, float]], match_px: float) -> Dict[str, Any]:
    remaining = set(range(len(truth)))
    true_positive = 0
    matched_distances = []
    for px, py in predicted:
        best_index = None
        best_distance = None
        for truth_index in remaining:
            tx, ty = truth[truth_index]
            distance = float(((px - tx) ** 2 + (py - ty) ** 2) ** 0.5)
            if distance <= match_px and (best_distance is None or distance < best_distance):
                best_index = truth_index
                best_distance = distance
        if best_index is not None:
            remaining.remove(best_index)
            true_positive += 1
            matched_distances.append(best_distance)
    false_positive = len(predicted) - true_positive
    false_negative = len(truth) - true_positive
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    return {
        "predicted": len(predicted),
        "ground_truth": len(truth),
        "tp": true_positive,
        "fp": false_positive,
        "fn": false_negative,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "mean_match_distance_px": (
            float(np.mean(matched_distances)) if matched_distances else None
        ),
    }


def _run_mode(records: List[Dict[str, Any]], mode: str, model_path: str | None, confidence: float) -> Dict[str, Any]:
    total = {
        "tp": 0,
        "fp": 0,
        "fn": 0,
        "predicted": 0,
        "ground_truth": 0,
        "area_m2": 0.0,
        "mature_canopy_area_m2": 0.0,
        "mature_canopy_components": 0,
    }
    images = []
    for record in records:
        image_path = Path(record["image"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(f"could not read {image_path}")
        detector = ProperDetectree2Detector(confidence_threshold=confidence, device="cpu")
        if model_path:
            detector.setup_model(model_path=model_path)
        else:
            detector.setup_model()
        detector.set_runtime_tuning(
            seedling_detection_mode=mode,
            seedling_classifier_path=record.get("seedling_classifier_path"),
        )
        _, _, metadata, _ = detector.detect_from_image(image, gsd=float(record["gsd"]))
        truth = _points(record.get("seedlings"))
        predicted = _predicted_centers(metadata)
        match_px = float(record.get("match_distance_m", 0.15)) / float(record["gsd"])
        metrics = _score(predicted, truth, match_px)
        for key in ("tp", "fp", "fn", "predicted", "ground_truth"):
            total[key] += metrics[key]
        total["area_m2"] += float(record.get("area_m2", 0.0) or 0.0)
        total["mature_canopy_area_m2"] += float(
            metadata.get("coverage_canopy_area_m2", 0.0) or 0.0
        )
        total["mature_canopy_components"] += int(
            metadata.get("coverage_component_count", 0) or 0
        )
        images.append({"image": str(image_path), "metrics": metrics, "metadata": {
            "seedling_detection_mode": metadata.get("seedling_detection_mode"),
            "classifier_available": metadata.get("seedling_classifier_available", False),
            "candidate_count": metadata.get("seedling_candidate_count", 0),
            "rejected_classifier": metadata.get("seedling_rejected_classifier", 0),
            "rejected_duplicate": metadata.get("seedling_rejected_duplicate", 0),
            "mature_canopy_area_m2": metadata.get("coverage_canopy_area_m2", 0.0),
            "mature_canopy_components": metadata.get("coverage_component_count", 0),
        }})
    precision = total["tp"] / max(total["tp"] + total["fp"], 1)
    recall = total["tp"] / max(total["tp"] + total["fn"], 1)
    total["precision"] = precision
    total["recall"] = recall
    total["f1"] = 2 * precision * recall / max(precision + recall, 1e-12)
    total["false_positives_per_m2"] = total["fp"] / max(total["area_m2"], 1e-12)
    return {"mode": mode, "summary": total, "images": images}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--split", default="test")
    parser.add_argument("--model-path", default=None)
    parser.add_argument("--confidence", type=float, default=0.80)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()
    records = []
    with args.manifest.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            if str(record.get("split", "test")) == args.split:
                records.append(record)
    if not records:
        raise ValueError(f"manifest has no records for split {args.split!r}")
    result = {
        "manifest": str(args.manifest),
        "split": args.split,
        "legacy": _run_mode(records, "legacy", args.model_path, args.confidence),
        "hybrid": _run_mode(records, "hybrid", args.model_path, args.confidence),
    }
    legacy_summary = result["legacy"]["summary"]
    hybrid_summary = result["hybrid"]["summary"]
    legacy_area = float(legacy_summary.get("mature_canopy_area_m2", 0.0))
    legacy_components = float(legacy_summary.get("mature_canopy_components", 0.0))
    result["mature_canopy_regression"] = {
        "area_delta_m2": float(hybrid_summary["mature_canopy_area_m2"] - legacy_area),
        "area_delta_pct": (
            float((hybrid_summary["mature_canopy_area_m2"] - legacy_area) / legacy_area * 100.0)
            if legacy_area > 0 else 0.0
        ),
        "component_delta": int(
            hybrid_summary["mature_canopy_components"] - legacy_components
        ),
        "component_delta_pct": (
            float((hybrid_summary["mature_canopy_components"] - legacy_components) / legacy_components * 100.0)
            if legacy_components > 0 else 0.0
        ),
    }
    rendered = json.dumps(result, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered)


if __name__ == "__main__":
    main()
