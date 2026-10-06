"""Review mature-canopy inference on original photos without the seedling branch.

Outputs stay local. No API, database, alignment, or planting-point generation runs.
The gallery and overview sheets are a visual review, not an accuracy benchmark.
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
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from canopy_detection.detectree2_proper import ProperDetectree2Detector
from canopy_detection.exif_extractor import ExifExtractor
from canopy_detection.gsd_calculator import GSDCalculator


def sha256(path):
    with path.open('rb') as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def write_json(path, value):
    # Keep the previous checkpoint readable if the process is interrupted.
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False), encoding='utf-8')
    temporary.replace(path)


def component_records(image, mask, gsd):
    count, labels, stats, centers = cv2.connectedComponentsWithStats((mask > 0).astype(np.uint8), 8)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    b, g, r = cv2.split(image.astype(np.float32))
    exg = 2 * g - r - b
    records = []
    for label in range(1, count):
        x, y, width, height, pixels = map(int, stats[label])
        selected = labels[y:y + height, x:x + width] == label
        records.append({
            'id': label, 'center': centers[label].tolist(), 'bbox': [x, y, width, height],
            'pixels': pixels, 'area_m2': pixels * gsd ** 2,
            'mean_saturation': float(hsv[y:y + height, x:x + width, 1][selected].mean()),
            'mean_exg': float(exg[y:y + height, x:x + width][selected].mean()),
            'touches_edge': x == 0 or y == 0 or x + width == image.shape[1] or y + height == image.shape[0],
        })
    return records


def save_previews(output, path, image, mask, components):
    width = 1200
    height = round(width * image.shape[0] / image.shape[1])
    original = cv2.resize(image, (width, height), interpolation=cv2.INTER_AREA)
    selected = cv2.resize(mask, (width, height), interpolation=cv2.INTER_NEAREST) > 0
    overlay = original.copy()
    overlay[selected] = (0.50 * overlay[selected] + 0.50 * np.array([210, 40, 165])).astype(np.uint8)
    scale = width / image.shape[1]
    for component in components:
        center = tuple(round(v * scale) for v in component['center'])
        cv2.putText(overlay, str(component['id']), center, cv2.FONT_HERSHEY_SIMPLEX,
                    .42, (255, 255, 255), 2, cv2.LINE_AA)
        cv2.putText(overlay, str(component['id']), center, cv2.FONT_HERSHEY_SIMPLEX,
                    .42, (0, 0, 0), 1, cv2.LINE_AA)
    for name, pixels in [('original', original), ('overlay', overlay)]:
        if not cv2.imwrite(str(output / f'{path.stem}_{name}.jpg'), pixels):
            raise OSError('Could not write review preview')
    if not cv2.imwrite(str(output / f'{path.stem}_mask.png'), mask):
        raise OSError('Could not write canopy mask')
    # Full-resolution crops of the smallest components make isolated ground
    # mistakes visible without zooming a 4K image for every candidate.
    crops = []
    for component in sorted(components, key=lambda c: c['area_m2'])[:24]:
        x, y, w, h = component['bbox']
        margin = max(30, round(.4 / component['gsd'])) if 'gsd' in component else 60
        x0, y0 = max(0, x - margin), max(0, y - margin)
        x1, y1 = min(image.shape[1], x + w + margin), min(image.shape[0], y + h + margin)
        crop = image[y0:y1, x0:x1]
        annotated = crop.copy()
        inside = mask[y0:y1, x0:x1] > 0
        annotated[inside] = (.5 * annotated[inside] + .5 * np.array([210, 40, 165])).astype(np.uint8)
        tile = np.full((250, 500, 3), 255, np.uint8)
        for offset, view in [(0, crop), (250, annotated)]:
            factor = min(244 / view.shape[1], 216 / view.shape[0])
            resized = cv2.resize(view, (max(1, round(view.shape[1] * factor)), max(1, round(view.shape[0] * factor))))
            tile[30:30 + resized.shape[0], offset:offset + resized.shape[1]] = resized
        label = f"#{component['id']} ({round(component['center'][0])}, {round(component['center'][1])}) {component['area_m2']:.3f} m2"
        cv2.putText(tile, label, (5, 21), cv2.FONT_HERSHEY_SIMPLEX, .45, (30, 30, 30), 1, cv2.LINE_AA)
        crops.append(tile)
    if crops:
        columns = 2
        sheet = np.full((250 * ((len(crops) + columns - 1) // columns), 500 * columns, 3), 240, np.uint8)
        for index, tile in enumerate(crops):
            y, x = (index // columns) * 250, (index % columns) * 500
            sheet[y:y + 250, x:x + 500] = tile
        cv2.imwrite(str(output / f'{path.stem}_crops.jpg'), sheet)


def write_gallery(output, rows):
    cards = []
    for row in rows:
        name = html.escape(row['file'])
        stem = html.escape(Path(row['file']).stem, quote=True)
        if row['status'] == 'ok':
            cards.append(f'<section><h2>{name} — {row["components"]} components; {row["area_m2"]:.2f} m²</h2>'
                         f'<div class="pair"><a href="{stem}_original.jpg"><img loading="lazy" src="{stem}_original.jpg"></a>'
                         f'<a href="{stem}_overlay.jpg"><img loading="lazy" src="{stem}_overlay.jpg"></a></div>'
                         f'<p><a href="{stem}.json">Coordinates and filter trace</a> · '
                         f'<a href="{stem}_mask.png">Full-resolution mask</a>'
                         + (f' · <a href="{stem}_crops.jpg">Component close-ups</a>' if row['components'] else '') + '</p></section>')
        else:
            cards.append(f'<section><h2>{name}</h2><p>{html.escape(row["error"])}</p></section>')
    page = ('<!doctype html><html><head><meta charset="utf-8"><title>Canopy-only review</title>'
            '<style>body{font:16px system-ui;margin:24px;background:#f5f7f5;color:#18382a}'
            'section{background:white;padding:16px;margin:20px 0;border-radius:12px}'
            '.pair{display:flex;gap:8px}.pair a{width:50%}img{width:100%}h2{font-size:18px}'
            '@media(max-width:700px){.pair{display:block}.pair a{display:block;width:100%}}</style></head><body>'
            '<h1>Canopy-only review</h1><p>Original (left), accepted canopy coverage in purple (right). '
            'Numbers identify connected components. Seedling detection is disabled. '
            'Full-resolution photos, the current model and filters, confidence 0.80, and EXIF ground scale. '
            'No alignment, planting generation, danger buffers, or database writes. '
            'Counts and areas are detector outputs, not accuracy measurements.</p>' + ''.join(cards) + '</body></html>')
    (output / 'index.html').write_text(page, encoding='utf-8')
    with (output / 'summary.csv').open('w', newline='', encoding='utf-8') as stream:
        fields = ['file', 'status', 'instances', 'components', 'area_m2', 'coverage_pct', 'tiles',
                  'ground_rejected', 'gsd', 'seconds', 'error']
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    good = [row for row in rows if row['status'] == 'ok']
    for start in range(0, len(good), 12):
        sheet = np.full((4 * 210, 3 * 640, 3), 250, np.uint8)
        for index, row in enumerate(good[start:start + 12]):
            y, x = (index // 3) * 210, (index % 3) * 640
            for offset, suffix in [(0, 'original'), (320, 'overlay')]:
                preview = cv2.imread(str(output / f"{Path(row['file']).stem}_{suffix}.jpg"))
                if preview is not None:
                    preview = cv2.resize(preview, (320, 180))
                    sheet[y + 28:y + 208, x + offset:x + offset + 320] = preview
            title = f"{row['file']} | {row['components']} components | {row['area_m2']:.1f} m2"
            cv2.putText(sheet, title, (x + 5, y + 20), cv2.FONT_HERSHEY_SIMPLEX, .46, (30, 30, 30), 1, cv2.LINE_AA)
        cv2.imwrite(str(output / f'overview_{start // 12 + 1:02d}.jpg'), sheet)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('input', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--device', choices=['auto', 'cpu', 'cuda'], default='auto')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--only', nargs='+', help='Exact photo filenames or stems for a focused AI rerun')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    torch.set_num_threads(4)
    cv2.setNumThreads(2)
    detector = ProperDetectree2Detector(confidence_threshold=.80, device=device)
    detector.set_runtime_tuning(seedling_supplement=False, seedling_micro_supplement=False,
                                seedling_detection_mode='off')
    def forbidden(*a, **kw):
        raise RuntimeError('Seedling inference must remain disabled in a canopy-only review')
    detector._detect_seedling_supplement_mask = forbidden
    detector._detect_hybrid_seedling_supplement_mask = forbidden
    detector.setup_model(str(ROOT / 'MangroVision_New/try_model/model_final.pth'))
    if detector.predictor is None:
        raise RuntimeError('The real AI canopy checkpoint is required; no color fallback is allowed')
    model_file = Path(detector.model_path)
    sources = {str(path.relative_to(ROOT)): sha256(path) for path in [
        ROOT / 'canopy_detection/detectree2_proper.py', ROOT / 'canopy_detection/seedling_leaf_evidence.py',
        ROOT / 'canopy_detection/exif_extractor.py', ROOT / 'canopy_detection/gsd_calculator.py',
        model_file, model_file.with_name('model_metadata.json'), Path(__file__).resolve(),
    ]}
    manifest = {'device': device, 'torch': torch.__version__, 'confidence': .80, 'sources': sources,
                'tuning': detector.runtime_tuning, 'input': str(args.input.resolve()),
                'min_size_test': detector.cfg.INPUT.MIN_SIZE_TEST, 'max_size_test': detector.cfg.INPUT.MAX_SIZE_TEST}
    manifest_path = args.output / 'run_sources.json'
    if args.resume and manifest_path.exists() and json.loads(manifest_path.read_text(encoding='utf-8')) != manifest:
        raise RuntimeError('Review inputs or detector settings changed; use a new output directory')
    write_json(manifest_path, manifest)
    paths = sorted(p for p in args.input.iterdir() if p.suffix.lower() in {'.jpg', '.jpeg', '.png'})
    if args.only:
        requested = {Path(name).stem.upper() for name in args.only}
        available = {p.stem.upper() for p in paths}
        if requested - available:
            raise ValueError(f'Missing requested photos: {sorted(requested - available)}')
        paths = [p for p in paths if p.stem.upper() in requested]
    priorities = ['HTWO9123', 'ITBH6710', 'JTZS4533', 'LUXA2941', 'WAJG']
    paths.sort(key=lambda p: (next((i for i, prefix in enumerate(priorities) if p.stem.startswith(prefix)), len(priorities)), p.name))
    if args.limit:
        paths = paths[:args.limit]
    rows = []
    captured = []
    original_nms = detector._nms_polygons
    def record_candidates(instances, *a, **kw):
        captured[:] = [{**{key: val for key, val in instance.items() if key not in {'polygon', 'contour'}},
                       'polygon': list(map(list, instance['polygon'].exterior.coords))} for instance in instances]
        return original_nms(instances, *a, **kw)
    detector._nms_polygons = record_candidates
    for index, path in enumerate(paths, 1):
        started = time.monotonic()
        record_path = args.output / f'{path.stem}.json'
        if args.resume and record_path.exists():
            record = json.loads(record_path.read_text(encoding='utf-8'))
            if record.get('sha256') == sha256(path) and record['row']['status'] == 'ok':
                rows.append(record['row'])
                print(f'[{index}/{len(paths)}] resumed {path.name}', flush=True)
                continue
        row = {'file': path.name}
        captured.clear()
        try:
            image = cv2.imread(str(path))
            if image is None:
                raise ValueError('Cannot decode original image')
            exif = ExifExtractor.extract_all_metadata(str(path))
            altitude = exif['gps'].get('relative_altitude')
            if altitude is None or altitude <= 0:
                raise ValueError('No positive relative altitude in EXIF; ground scale was not guessed')
            gsd, scale_source = GSDCalculator.calculate_gsd_from_metadata(
                altitude, exif['camera'], ExifExtractor.detect_drone_model(exif['camera']),
                image.shape[1], image.shape[0])
            def progress(event, values):
                if event == 'tile_done' and values['current_tile'] % 20 == 0:
                    print(f"[{index}/{len(paths)}] {path.name}: tile {values['current_tile']}/{values['total_tiles']}", flush=True)
            polygons, mask, metadata, seedling_mask = detector.detect_from_image(image, gsd=gsd, progress_callback=progress)
            assert metadata['seedling_detection_mode'] == 'disabled'
            assert metadata['seedling_supplement_count'] == 0 and not np.any(seedling_mask)
            components = component_records(image, mask, gsd)
            save_previews(args.output, path, image, mask, components)
            row.update(status='ok', instances=metadata['instance_tree_count'], components=len(components),
                       area_m2=float(np.count_nonzero(mask) * gsd ** 2),
                       coverage_pct=float(100 * np.count_nonzero(mask) / mask.size),
                       tiles=metadata['num_tiles'], ground_rejected=metadata['canopy_ground_artifact_rejected_count'],
                       gsd=gsd, seconds=round(time.monotonic() - started, 2))
            record = {'file': path.name, 'sha256': sha256(path), 'shape': list(image.shape), 'row': row,
                      'scale_source': scale_source, 'metadata': metadata, 'accepted_components': components,
                      'pre_cleanup_candidates': captured.copy(),
                      'coverage_polygons': [list(map(list, polygon.exterior.coords)) for polygon in polygons],
                      'mature_canopy_inference': True, 'seedling_inference': False}
            write_json(record_path, record)
        except Exception as exc:
            row.update(status='error', seconds=round(time.monotonic() - started, 2), error=str(exc))
            print(f'ERROR {path.name}: {exc}', flush=True)
        rows.append(row)
        write_json(args.output / 'summary.json', rows)
        write_gallery(args.output, rows)
        print(f'[{index}/{len(paths)}] {json.dumps(row)}', flush=True)
    write_json(args.output / 'summary.json', rows)
    write_gallery(args.output, rows)
    print(f"Finished {len(rows)} photos: {sum(r['status'] == 'ok' for r in rows)} successful", flush=True)
    return 1 if any(row['status'] != 'ok' for row in rows) else 0


if __name__ == '__main__':
    raise SystemExit(main())
