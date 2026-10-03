"""Offline seedling-only review; never loads the mature-canopy predictor or DB.

Run with the project's Python environment. Output contains original-photo
thumbnails, so keep it local just like the input images.
"""
import argparse
import csv
import hashlib
import html
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from canopy_detection.detectree2_proper import ProperDetectree2Detector
from canopy_detection.exif_extractor import ExifExtractor
from canopy_detection.gsd_calculator import GSDCalculator


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    detector = ProperDetectree2Detector()
    # Fail loudly if a future refactor attempts to invoke canopy inference.
    def forbidden(*a, **kw):
        raise RuntimeError("Mature-canopy inference is disabled for this review")
    detector.setup_model = forbidden
    detector.detect_from_image = forbidden
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
    paths.sort(key=lambda p: (p.stem != "LUXA2941", p.name))
    rows = []
    for index, path in enumerate(paths, 1):
        started = time.monotonic()
        row = {"file": path.name}
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError("Cannot decode image")
            metadata = ExifExtractor.extract_all_metadata(str(path))
            altitude = metadata["gps"].get("relative_altitude")
            if altitude is None or altitude <= 0:
                raise ValueError("Missing positive relative altitude; no scale guessed")
            gsd, _ = GSDCalculator.calculate_gsd_from_metadata(
                altitude, metadata["camera"], ExifExtractor.detect_drone_model(metadata["camera"]),
                image.shape[1], image.shape[0])
            empty = np.zeros(image.shape[:2], np.uint8)
            mask, result = detector._detect_hybrid_seedling_supplement_mask(image, empty, gsd)
            if result.get("seedling_detection_mode") == "fallback_legacy":
                mask, legacy = detector._detect_seedling_supplement_mask(image, empty, gsd)
                result.update(legacy)
            assert detector.predictor is None and not np.any(empty)
            centers = result.get("seedling_accepted_centers", [])
            record = {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                      "gsd": gsd, "shape": list(image.shape), "metadata": result,
                      "mature_canopy_inference": False, "canopy_exclusion_mask": "empty"}
            (args.output / (path.stem + ".json")).write_text(json.dumps(record, indent=2), encoding="utf-8")
            cv2.imwrite(str(args.output / (path.stem + "_mask.png")), mask)
            small = cv2.resize(image, (1200, round(1200*image.shape[0]/image.shape[1])))
            cv2.imwrite(str(args.output / (path.stem + "_original.jpg")), small)
            scale = 1200/image.shape[1]
            for x, y in centers:
                point = (round(x*scale), round(y*scale))
                cv2.circle(small, point, 4, (0, 0, 0), 2)
                cv2.circle(small, point, 3, (255, 0, 255), 1)
            cv2.imwrite(str(args.output / (path.stem + "_overlay.jpg")), small)
            row.update(status="ok", count=result.get("seedling_supplement_count", 0),
                       rejected=result.get("seedling_ground_artifact_rejected_count", 0),
                       mode=result.get("seedling_detection_mode"), gsd=gsd)
        except Exception as exc:
            row.update(status="error", error=str(exc))
        row["seconds"] = round(time.monotonic()-started, 2)
        rows.append(row)
        (args.output / "summary.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
        print(f"[{index}/{len(paths)}] {row}", flush=True)
    columns = ["file", "status", "count", "rejected", "mode", "gsd", "seconds", "error"]
    with (args.output / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    cards = []
    for row in rows:
        name = html.escape(row["file"])
        stem = html.escape(Path(row["file"]).stem, quote=True)
        if row["status"] == "ok":
            cards.append(f'<section><h2>{name}: {row["count"]} markers</h2><div><a href="{stem}_original.jpg"><img loading="lazy" src="{stem}_original.jpg"></a><a href="{stem}_overlay.jpg"><img loading="lazy" src="{stem}_overlay.jpg"></a></div><a href="{stem}.json">Detection metadata</a></section>')
        else:
            cards.append(f'<section><h2>{name}</h2><p>{html.escape(row["error"])}</p></section>')
    page = '<!doctype html><meta charset="utf-8"><title>Seedling-only review</title><style>body{font:16px system-ui;margin:24px;background:#f5f7f5}section{background:white;padding:16px;margin:20px 0}section div{display:flex;gap:8px}section div a{width:50%}img{width:100%}h2{font-size:18px}</style><h1>Seedling-only review</h1><p>Original (left), accepted centers in magenta (right). No mature-canopy inference, canopy exclusion, danger buffers, GIS clipping, or database writes. Tree foliage may also receive markers. Counts are not accuracy measurements.</p>' + ''.join(cards)
    (args.output / "index.html").write_text(page, encoding="utf-8")


if __name__ == "__main__":
    main()
