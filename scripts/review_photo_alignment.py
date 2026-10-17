"""Review local automatic photo alignment without running AI or changing records."""
import argparse
import html
import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from shapely.geometry import shape

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / 'MangroVision_New')]
from api.routes import processing as p


def project_corners(image, matrix):
    height, width = image.shape[:2]
    return cv2.perspectiveTransform(np.float64([
        [[0, 0]], [[width, 0]], [[width, height]], [[0, height]],
    ]), matrix).reshape(-1, 2)


def write_jpeg(path, image, max_width=1200):
    if image.shape[1] > max_width:
        ratio = max_width / image.shape[1]
        image = cv2.resize(image, (max_width, round(image.shape[0] * ratio)), interpolation=cv2.INTER_AREA)
    if not cv2.imwrite(str(path), image, [cv2.IMWRITE_JPEG_QUALITY, 93]):
        raise RuntimeError(f'Could not write {path.name}')


def review(photo, output):
    image = cv2.imread(str(photo))
    if image is None:
        raise ValueError('The original image could not be opened.')
    metadata = p.ExifExtractor.extract_all_metadata(str(photo))
    gps = metadata.get('gps') or {}
    if any(gps.get(key) is None for key in ('latitude', 'longitude', 'heading')):
        raise ValueError('This review needs GPS and camera heading in the original photo.')
    height, width = image.shape[:2]
    gsd, _ = p.GSDCalculator.calculate_gsd_from_metadata(
        gps.get('relative_altitude', gps.get('altitude')), metadata['camera'],
        p.ExifExtractor.detect_drone_model(metadata['camera']), width, height,
    )
    match = p._get_image_alignment(photo, image, gps['latitude'], gps['longitude'], gsd, gps['heading'])
    matrix = p._coordinate_homography_for_match(match)
    if matrix is None:
        raise RuntimeError('Automatic alignment was not accepted for this image.')
    nominal = p._build_metric_centered_homography(image, gps['latitude'], gps['longitude'], gsd, gps['heading'])
    footprint = p._projected_image_footprint(image.shape, matrix)
    # Do not query organizations or project sites. This review isolates the
    # transform and uses a generous local coverage envelope to avoid clipping.
    with patch.object(p, '_project_site_context', return_value={'project_sites': []}), \
         patch.object(p, '_load_gis_coverage_geometry', return_value=shape(footprint).buffer(.01)):
        preflight = p._calibrate_preflight_footprint(photo, {
            'latitude': gps['latitude'], 'longitude': gps['longitude'],
            'gsd_m_per_pixel': gsd, 'heading_deg': gps['heading'],
            'coverage_m': [width * gsd, height * gsd],
        })
    overlay_h = p._overlay_homography_for_match(match)
    preview_matches = shape(preflight['map']['analysis_footprint']).equals(shape(footprint))
    if not preview_matches or not np.array_equal(matrix, overlay_h):
        raise RuntimeError('The location preview and result do not share the same transform.')
    reference = p.ortho_matcher.load_orthophoto()
    registered = cv2.warpPerspective(reference, np.linalg.inv(matrix), (width, height))
    blend = cv2.addWeighted(image, .5, registered, .5, 0)
    output.mkdir(parents=True, exist_ok=True)
    for name, pixels in [('photo.jpg', image), ('registered-reference.jpg', registered), ('blend.jpg', blend)]:
        write_jpeg(output / name, pixels)
    panels = []
    for title, pixels in [('Original photo', image), ('Orthophoto in photo coordinates', registered), ('50% blend', blend)]:
        panel = cv2.resize(pixels, (1000, round(height * 1000 / width)))
        panel = cv2.copyMakeBorder(panel, 35, 0, 0, 0, cv2.BORDER_CONSTANT, value=(245, 245, 245))
        cv2.putText(panel, title, (10, 24), 0, .65, (20, 40, 20), 1, cv2.LINE_AA)
        panels.append(panel)
    write_jpeg(output / 'comparison.jpg', np.vstack(panels))
    corrected, estimated = project_corners(image, matrix), project_corners(image, nominal)
    all_corners = np.vstack((corrected, estimated))
    left, top = np.maximum(0, np.floor(all_corners.min(axis=0) - 80).astype(int))
    right, bottom = np.minimum([reference.shape[1], reference.shape[0]], np.ceil(all_corners.max(axis=0) + 80).astype(int))
    map_image = reference[top:bottom, left:right].copy()
    offset = np.array([left, top])
    # Dashed amber is the former GPS estimate; green is the actual new polygon.
    old = np.vstack((estimated, estimated[0])) - offset
    for first, second in zip(old[:-1], old[1:]):
        distance = np.linalg.norm(second - first)
        for start in np.arange(0, distance, 24):
            a = first + (second - first) * (start / distance)
            b = first + (second - first) * (min(start + 12, distance) / distance)
            cv2.line(map_image, tuple(np.rint(a).astype(int)), tuple(np.rint(b).astype(int)), (0, 180, 245), 3, cv2.LINE_AA)
    cv2.polylines(map_image, [np.rint(corrected - offset).astype(np.int32)], True, (35, 170, 30), 4, cv2.LINE_AA)
    write_jpeg(output / 'map-boundary.jpg', map_image, max_width=1600)
    report = {
        'image': photo.name, 'method': match.get('projection_rotation_source'),
        'approximate': bool(match.get('projection_rebuilt')),
        'shared_preview_coordinates_overlay': bool(preview_matches),
        'center_offset_m': match.get('center_drift_m'),
        'photometric_score_before': match.get('photometric_baseline_score'),
        'photometric_score_after': match.get('photometric_score'),
        'improved_regions': match.get('photometric_improved_cells'),
        'fit_agreement_m': match.get('photometric_stability_m'),
        'accepted_H': matrix.tolist(), 'previous_H': nominal.tolist(),
        'footprint': footprint,
    }
    (output / 'alignment.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    title = html.escape(photo.name)
    page = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>{title} alignment review</title><style>
body{{margin:0;background:#f2f7f3;color:#183b2a;font:16px/1.5 system-ui,sans-serif}}main{{max-width:1300px;margin:32px auto;padding:0 20px}}h1{{margin-bottom:8px}}p{{max-width:90ch}}section{{background:white;padding:22px;border:1px solid #d4e2d8;border-radius:14px;margin:20px 0}}img{{width:100%;height:auto;display:block;border-radius:8px}}.compare{{position:relative}}.compare img+img{{position:absolute;inset:0;opacity:.5}}input{{width:100%}}.legend{{display:flex;gap:24px;flex-wrap:wrap}}.green{{color:#169c21}}.amber{{color:#986500}}details{{margin-top:18px}}a{{color:#166534}}
</style><main><h1>{title} — automatic alignment</h1>
<p>Local alignment review. The green border below uses the same corrected transform as the upload preview and planting coordinates. This remains an approximate image alignment. Saved analyses and live planting records were not changed.</p>
<section><h2>Map boundary</h2><p class="legend"><strong class="green">Green: corrected photo boundary</strong><span class="amber">Dashed amber: previous GPS estimate</span></p><img src="map-boundary.jpg" alt="Orthophoto showing the corrected green footprint and previous amber GPS footprint"></section>
<section><h2>Compare the photo and orthophoto</h2><p>Move the slider to compare matching features.</p><div class="compare"><img src="photo.jpg" alt="Original photo"><img id="reference" src="registered-reference.jpg" alt="Orthophoto projected into the same photo coordinates"></div><label>Orthophoto opacity <input type="range" min="0" max="100" value="50" oninput="document.getElementById('reference').style.opacity=this.value/100"></label><p><a href="comparison.jpg">Open the three-image comparison</a></p></section>
<details><summary>Verification details</summary><p>Preview, result overlay and planting coordinates share the accepted transform: <strong>{str(preview_matches).lower()}</strong>.</p><p><a href="alignment.json">Read local alignment measurements</a></p></details></main></html>'''
    (output / 'index.html').write_text(page, encoding='utf-8')
    print(json.dumps({key: value for key, value in report.items() if key not in {'accepted_H', 'previous_H', 'footprint'}}, indent=2))
    print(output.resolve() / 'index.html')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('image', type=Path)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    review(args.image, args.output)
