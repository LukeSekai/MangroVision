"""
Batch evaluator for MangroVision canopy detection.

Runs the same detector settings across a folder of images and writes a CSV
summary so you can compare runs consistently before/after tuning.
"""

import argparse
import json
import time
from pathlib import Path
import sys

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(REPO_ROOT))
sys.path.append(str(REPO_ROOT / "canopy_detection"))


def parse_args():
    parser = argparse.ArgumentParser(description="Batch canopy evaluation for MangroVision.")
    parser.add_argument("--images-dir", required=True, help="Folder containing drone images.")
    parser.add_argument("--pattern", default="*.jpg", help="Glob pattern (default: *.jpg)")
    parser.add_argument("--output-csv", default="batch_eval_results.csv", help="Output CSV path.")
    parser.add_argument("--altitude-m", type=float, default=6.0)
    parser.add_argument("--drone-model", default="GENERIC_4K")
    parser.add_argument("--detection-mode", choices=["hybrid", "ai", "hsv"], default="ai")
    parser.add_argument("--model-name", default="paracou")
    parser.add_argument("--ai-confidence", type=float, default=0.75)
    parser.add_argument("--canopy-buffer-m", type=float, default=1.0)
    parser.add_argument("--hexagon-size-m", type=float, default=1.0)
    parser.add_argument("--tile-veg-threshold", type=float, default=0.002)
    parser.add_argument("--min-crown-m2", type=float, default=0.05)
    parser.add_argument("--max-crown-m2", type=float, default=60.0)
    parser.add_argument("--overlap-validation-ratio", type=float, default=0.15)
    parser.add_argument("--cleanup-iou", type=float, default=0.75)
    parser.add_argument("--fallback-nms-iou", type=float, default=0.90)
    return parser.parse_args()


def main():
    args = parse_args()

    try:
        from canopy_detection.canopy_detector_hexagon import HexagonDetector  # noqa: E402
    except Exception as exc:
        raise RuntimeError(
            "Could not import canopy detector. Install dependencies first: "
            "pip install -r requirements.txt"
        ) from exc

    images_dir = Path(args.images_dir)
    if not images_dir.exists():
        raise FileNotFoundError(f"Images directory not found: {images_dir}")

    image_paths = sorted(images_dir.glob(args.pattern))
    if not image_paths:
        raise FileNotFoundError(f"No images matched pattern '{args.pattern}' in {images_dir}")

    detector = HexagonDetector(
        altitude_m=args.altitude_m,
        drone_model=args.drone_model,
        ai_confidence=args.ai_confidence,
        model_name=args.model_name,
        detection_mode=args.detection_mode,
    )

    runtime_tuning = {
        "tile_veg_threshold": args.tile_veg_threshold,
        "min_crown_m2": args.min_crown_m2,
        "max_crown_m2": args.max_crown_m2,
        "overlap_validation_ratio": args.overlap_validation_ratio,
        "cleanup_iou": args.cleanup_iou,
        "fallback_nms_iou": args.fallback_nms_iou,
        "separate_threshold_m2": 10.0,
    }

    if (
        getattr(detector, "ai_detector", None) is not None
        and hasattr(detector.ai_detector, "set_runtime_tuning")
    ):
        try:
            import inspect

            accepted = set(inspect.signature(detector.ai_detector.set_runtime_tuning).parameters.keys())
            tuned_kwargs = {k: v for k, v in runtime_tuning.items() if k in accepted}
            if tuned_kwargs:
                detector.ai_detector.set_runtime_tuning(**tuned_kwargs)
        except Exception as exc:
            print(f"[warn] Could not apply runtime tuning: {exc}")

    rows = []
    for idx, image_path in enumerate(image_paths, start=1):
        print(f"[{idx}/{len(image_paths)}] Processing {image_path.name}")
        start = time.time()
        try:
            results = detector.process_image(
                image_path=str(image_path),
                canopy_buffer_m=args.canopy_buffer_m,
                hexagon_size_m=args.hexagon_size_m,
            )
            elapsed = time.time() - start
            meta = results.get("ai_metadata", {}) or {}
            rows.append(
                {
                    "image_name": image_path.name,
                    "seconds": round(elapsed, 3),
                    "canopy_count": results.get("canopy_count"),
                    "danger_area_m2": results.get("danger_area_m2"),
                    "plantable_area_m2": results.get("plantable_area_m2"),
                    "hexagon_count": results.get("hexagon_count"),
                    "detection_method": meta.get("detection_method"),
                    "model_name": meta.get("model_name"),
                    "model_path": meta.get("model_path"),
                    "num_tiles_checked": meta.get("num_tiles_checked"),
                    "num_tiles_processed": meta.get("num_tiles_processed", meta.get("num_tiles")),
                    "num_tiles_skipped": meta.get("num_tiles_skipped"),
                    "raw_detections": meta.get("raw_detections", meta.get("total_ai_detections")),
                    "final_trees": meta.get("final_trees", meta.get("num_detected_canopies")),
                    "gsd_used": meta.get("gsd_used", results.get("gsd_m_per_pixel")),
                    "runtime_tuning": json.dumps(runtime_tuning, sort_keys=True),
                }
            )
        except Exception as exc:
            elapsed = time.time() - start
            rows.append(
                {
                    "image_name": image_path.name,
                    "seconds": round(elapsed, 3),
                    "error": str(exc),
                    "runtime_tuning": json.dumps(runtime_tuning, sort_keys=True),
                }
            )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Saved batch evaluation to: {output_csv}")


if __name__ == "__main__":
    main()
