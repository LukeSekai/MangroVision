"""Validate canopy cleanup changes using cached, full-resolution AI predictions.

Run review_canopies.py first. This replay uses the same NMS, appearance checks,
coverage merge and water filtering as production, without rerunning either AI
branch, alignment, planting generation or database operations.
"""

import argparse
import json
from pathlib import Path
import sys
import time

import cv2
import numpy as np
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from canopy_detection.detectree2_proper import ProperDetectree2Detector
from canopy_detection.seedling_leaf_evidence import filter_small_canopy_ground_artifacts
from scripts.review_canopies import component_records, save_previews, sha256, write_gallery, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('baseline', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--verify-baseline', action='store_true',
                        help='Require bit-identical masks before changing the production filter')
    args = parser.parse_args()
    if args.output.resolve() == args.baseline.resolve():
        raise ValueError('Keep the original audit unchanged; choose a separate output directory')
    args.output.mkdir(parents=True, exist_ok=True)
    cv2.setNumThreads(2)
    source_hashes = {str(p.relative_to(ROOT)): sha256(p) for p in [
        ROOT / 'canopy_detection/detectree2_proper.py', ROOT / 'canopy_detection/seedling_leaf_evidence.py',
        Path(__file__).resolve()]}
    manifest = json.loads((args.baseline / 'run_sources.json').read_text(encoding='utf-8'))
    if manifest['tuning'].get('use_clean_crowns'):
        raise ValueError('This replay requires the recorded NMS cleanup path')
    detector = ProperDetectree2Detector(confidence_threshold=manifest['confidence'])
    detector.runtime_tuning = manifest['tuning'].copy()
    rows, changes = [], []
    originals = json.loads((args.baseline / 'summary.json').read_text(encoding='utf-8'))
    for index, original in enumerate(originals, 1):
        if original['status'] != 'ok':
            raise ValueError(f"Incomplete baseline photo: {original['file']}")
        started = time.monotonic()
        path = args.input / original['file']
        record = json.loads((args.baseline / f'{path.stem}.json').read_text(encoding='utf-8'))
        if sha256(path) != record['sha256']:
            raise ValueError(f'Original photo changed: {path.name}')
        image = cv2.imread(str(path))
        if image is None or list(image.shape) != record['shape']:
            raise ValueError(f'Invalid original photo: {path.name}')
        gsd = original['gsd']
        candidates = [{**candidate, 'polygon': Polygon(candidate['polygon'])}
                      for candidate in record['pre_cleanup_candidates']]
        polygons = detector._nms_polygons(candidates, detector.runtime_tuning['fallback_nms_iou'])
        base_mask = detector._polygons_to_mask(polygons, image.shape[:2])
        validated, ground_metadata = filter_small_canopy_ground_artifacts(image, base_mask, gsd)
        if ground_metadata['canopy_ground_artifact_rejected_count']:
            rejected = detector._mask_to_polygons(cv2.subtract(base_mask, validated), 0.)
            polygons = [p for p in polygons if not any(r.covers(p.representative_point()) for r in rejected)]
        strict_green = (detector._detect_strict_canopy_green_hsv(image)
                        if detector.runtime_tuning['strict_canopy_hsv'] else None)
        minimum_area = max(20., detector.runtime_tuning['min_crown_m2'] / gsd**2)
        coverage, mask, merge_metadata = detector._merge_fragmented_canopy_mask(
            validated, strict_green, gsd, minimum_area)
        mask, water_metadata = detector._filter_low_saturation_components(mask, image, strict_green)
        if water_metadata['rejected_low_saturation_components']:
            coverage = detector._mask_to_polygons(mask, minimum_area)
        baseline_mask = cv2.imread(str(args.baseline / f'{path.stem}_mask.png'), 0)
        if baseline_mask is None or baseline_mask.shape != mask.shape:
            raise ValueError(f'Missing baseline mask: {path.name}')
        changed_pixels = int(np.count_nonzero(mask != baseline_mask))
        if args.verify_baseline and changed_pixels:
            raise AssertionError(f'Replay differs from original pipeline: {path.name}, {changed_pixels} pixels')
        components = component_records(image, mask, gsd)
        row = {**original, 'instances': len(polygons), 'components': len(components),
               'area_m2': float(np.count_nonzero(mask) * gsd**2),
               'coverage_pct': float(100 * np.count_nonzero(mask) / mask.size),
               'ground_rejected': ground_metadata['canopy_ground_artifact_rejected_count'],
               'seconds': round(time.monotonic() - started, 2)}
        count, labels, stats, _ = cv2.connectedComponentsWithStats((baseline_mask > 0).astype(np.uint8), 8)
        retained_components = []
        for component in record['accepted_components']:
            label = component['id']
            x, y, w, h, pixels = stats[label]
            selected = labels[y:y+h, x:x+w] == label
            retained = np.count_nonzero((mask[y:y+h, x:x+w] > 0) & selected)
            retained_components.append({**component, 'retained_fraction': float(retained / pixels)})
        change = {'file': path.name, 'changed_pixels': changed_pixels,
                  'removed_pixels': int(np.count_nonzero((baseline_mask > 0) & (mask == 0))),
                  'added_pixels': int(np.count_nonzero((baseline_mask == 0) & (mask > 0))),
                  'baseline_area_m2': original['area_m2'], 'new_area_m2': row['area_m2'],
                  'baseline_component_retention': retained_components, **ground_metadata}
        changes.append(change)
        rows.append(row)
        write_json(args.output / f'{path.stem}.json', {
            'file': path.name, 'sha256': record['sha256'], 'shape': record['shape'], 'row': row,
            'accepted_components': components, 'coverage_polygons': [list(p.exterior.coords) for p in coverage],
            'metadata': {**ground_metadata, **merge_metadata, **water_metadata},
            'model_inference': False, 'reuses_baseline_ai_predictions': True, 'seedling_inference': False,
        })
        save_previews(args.output, path, image, mask, components)
        if index % 10 == 0 or index == len(originals):
            print(f'[{index}/{len(originals)}] replayed; {sum(c["changed_pixels"] > 0 for c in changes)} changed photos', flush=True)
    write_json(args.output / 'summary.json', rows)
    write_json(args.output / 'changes.json', changes)
    write_json(args.output / 'run_sources.json', {
        'baseline': str(args.baseline.resolve()), 'baseline_manifest_sha256': sha256(args.baseline / 'run_sources.json'),
        'verified_bit_identical_baseline': args.verify_baseline,
        'model_inference': False, 'seedling_inference': False,
        'sources': source_hashes,
    })
    write_gallery(args.output, rows)
    print(f'Finished {len(rows)} photos; {sum(c["changed_pixels"] > 0 for c in changes)} changed', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
