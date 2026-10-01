"""Create a leakage-safe seedling classifier dataset from point annotations.

The input is JSON Lines.  Each line describes one original flight image and
must contain an explicit split so adjacent frames can be kept together::

    {"image": "frames/site_a_001.JPG", "gsd": 0.010,
     "split": "train", "seedlings": [[120, 80]],
     "negatives": [[340, 210], [500, 90]]}

Coordinates are pixel ``[x, y]`` pairs.  Negative points should be selected
from mud, water, shadow, debris, and other known hard-negative regions.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Tuple

import cv2


def _extract_patch(image, x: float, y: float, side_px: int):
    side_px = max(16, int(side_px))
    half = side_px // 2
    pad = half + 2
    padded = cv2.copyMakeBorder(image, pad, pad, pad, pad, cv2.BORDER_REFLECT_101)
    cx = int(round(x)) + pad
    cy = int(round(y)) + pad
    return padded[cy - half : cy - half + side_px, cx - half : cx - half + side_px]


def _iter_points(value: Any) -> Iterable[Tuple[float, float]]:
    for point in value or []:
        if isinstance(point, dict):
            yield float(point["x"]), float(point["y"])
        else:
            yield float(point[0]), float(point[1])


def prepare_dataset(manifest_path: Path, output_dir: Path, crop_size_m: float = 0.75) -> Dict[str, int]:
    counts = {"seedling": 0, "negative": 0, "images": 0}
    output_dir.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            record = json.loads(line)
            image_path = Path(record["image"])
            if not image_path.exists():
                raise FileNotFoundError(f"line {line_number}: image not found: {image_path}")
            split = str(record.get("split", "")).strip().lower()
            if split not in {"train", "valid", "test"}:
                raise ValueError(f"line {line_number}: split must be train, valid, or test")
            gsd = float(record["gsd"])
            if gsd <= 0:
                raise ValueError(f"line {line_number}: gsd must be positive")
            image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
            if image is None:
                raise ValueError(f"line {line_number}: OpenCV could not read {image_path}")
            side_px = max(32, int(round(crop_size_m / gsd)))
            counts["images"] += 1
            for class_name, field_name in (("seedling", "seedlings"), ("negative", "negatives")):
                class_dir = output_dir / split / class_name
                class_dir.mkdir(parents=True, exist_ok=True)
                for point_index, (x, y) in enumerate(_iter_points(record.get(field_name))):
                    patch = _extract_patch(image, x, y, side_px)
                    if patch.size == 0:
                        continue
                    filename = f"{image_path.stem}_{line_number:05d}_{class_name}_{point_index:04d}.jpg"
                    destination = class_dir / filename
                    if not cv2.imwrite(str(destination), patch, [int(cv2.IMWRITE_JPEG_QUALITY), 97]):
                        raise IOError(f"could not write {destination}")
                    counts[class_name] += 1
    return counts


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True, help="JSONL point-annotation manifest")
    parser.add_argument("--output", type=Path, required=True, help="ImageFolder output directory")
    parser.add_argument("--crop-size-m", type=float, default=0.75)
    args = parser.parse_args()
    counts = prepare_dataset(args.manifest, args.output, args.crop_size_m)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
