"""Local, read-only image regressions; writes diagnostic artifacts, never DB rows."""

import argparse
import json
import sys
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "MangroVision_New"), str(ROOT / "canopy_detection")]

from api.routes import processing as processing
from detectree2_proper import ProperDetectree2Detector
from exif_extractor import ExifExtractor
from gsd_calculator import GSDCalculator


def image_context(path):
    image = cv2.imread(str(path))
    metadata = ExifExtractor.extract_all_metadata(str(path))
    gps = metadata["gps"]
    gsd, _ = GSDCalculator.calculate_gsd_from_metadata(
        altitude_m=gps.get("relative_altitude", gps.get("altitude")),
        camera_info=metadata["camera"],
        drone_model=ExifExtractor.detect_drone_model(metadata["camera"]),
        image_width_px=image.shape[1], image_height_px=image.shape[0],
    )
    return image, gps, gsd


def seedlings(path, output):
    image, _, gsd = image_context(path)
    contexts = []

    def record(frame, event, result):
        if event == "return" and frame.f_code.co_name == "water_context_metrics":
            values = frame.f_locals
            contexts.append({
                "center": list(values["center"]),
                **{k: float(values[k]) for k in ("object_sat", "object_exg", "object_chroma")},
                **result,
            })

    detector = ProperDetectree2Detector()
    sys.setprofile(record)
    try:
        mask, metadata = detector._detect_seedling_supplement_mask(
            image, np.zeros(image.shape[:2], np.uint8), gsd,
        )
    finally:
        sys.setprofile(None)
    metadata.update({"gsd": gsd, "contexts": contexts})
    (output / f"{path.stem}_seedlings.json").write_text(json.dumps(metadata, indent=2))
    centers = metadata["seedling_accepted_centers"]
    display = cv2.resize(image, (1200, 675))
    for x, y in centers:
        cv2.circle(display, (round(x * 1200 / image.shape[1]), round(y * 675 / image.shape[0])), 3, (255, 0, 255), 1)
    cv2.imwrite(str(output / f"{path.stem}_seedlings.jpg"), display)
    # Contact sheets keep the actual source pixels visible at each accepted point.
    if "0585" in path.stem:
        selected = [c for c in centers if c[1] > 1400 or c[0] > 3100]
    else:
        selected = centers[::max(1, len(centers) // 48)][:48]
    selected = selected[:96]
    for start in range(0, len(selected), 48):
        batch = selected[start:start+48]
        sheet = np.full((6 * 140, 8 * 140, 3), 245, np.uint8)
        for i, (x, y) in enumerate(batch):
            crop = cv2.getRectSubPix(image, (80, 80), (float(x), float(y)))
            tile = cv2.resize(crop, (136, 112), interpolation=cv2.INTER_NEAREST)
            col, row = (i % 8) * 140, (i // 8) * 140
            sheet[row:row+112, col:col+136] = tile
            cv2.circle(sheet, (col+68, row+56), 10, (255, 0, 255), 1)
            cv2.putText(sheet, f"{round(x)},{round(y)}", (col+2,row+130), 0, .36, (0,0,0), 1)
        cv2.imwrite(str(output / f"{path.stem}_crops_{start}.jpg"), sheet)
    print(path.stem, {k:v for k,v in metadata.items() if isinstance(v,(int,float))}, flush=True)


def alignment(path, output):
    image, gps, gsd = image_context(path)
    matches = []
    original = cv2.findHomography
    original_selector = processing.ortho_matcher.select_registration

    def record(src, dst, *args, **kwargs):
        matches.append((src.copy(), dst.copy()))
        return original_selector(src, dst, *args, **kwargs)

    with patch.object(processing.ortho_matcher, "select_registration", side_effect=record):
        result = processing._match_drone_to_ortho_robust(
            image, gps["latitude"], gps["longitude"], gsd, gps.get("heading", 0),
        )
    serial = {k: v.tolist() if isinstance(v,np.ndarray) else v for k,v in result.items()}
    (output / f"{path.stem}_alignment.json").write_text(json.dumps(serial, indent=2))
    if not result.get("success") or not matches:
        print(path.stem, serial, flush=True)
        return
    src_full, dst_full = matches[0]
    np.savez(output / f"{path.stem}_matches.npz", src=src_full, dst=dst_full,
             current=result["H"], shape=np.array(image.shape), gsd=gsd,
             ortho_gsd=processing.ortho_matcher.ORTHO_GSD)
    ortho = processing.ortho_matcher.load_orthophoto()
    h,w = image.shape[:2]
    corners = np.float64([[[0,0]],[[w,0]],[[w,h]],[[0,h]]])
    transformed = cv2.perspectiveTransform(corners, result["H"]).reshape(-1,2)
    left,top = np.maximum(0,np.floor(transformed.min(0)-50).astype(int))
    right,bottom = np.minimum([ortho.shape[1],ortho.shape[0]],np.ceil(transformed.max(0)+50).astype(int))
    target = ortho[top:bottom,left:right]
    T = np.float64([[1,0,-left],[0,1,-top],[0,0,1]])
    summary = {}
    for kind in ("current", "previous", "affine", "projective"):
        if kind == "current":
            H = result["H"]
        elif kind == "previous":
            legacy_result = dict(result)
            legacy_result["registration_validated"] = False
            legacy_result["H"] = result.get("projective_H", result["H"])
            H = processing._post_process_match(
                legacy_result, image, gps["latitude"], gps["longitude"], gps.get("heading", 0), gsd,
            )["H"]
        elif kind == "affine":
            A, _ = cv2.estimateAffine2D(src_full, dst_full, method=cv2.RANSAC, ransacReprojThreshold=3, maxIters=10000, confidence=.999)
            H = np.vstack((A,[0,0,1]))
        else:
            H, _ = original(src_full,dst_full,cv2.RANSAC,3,maxIters=10000,confidence=.999)
        residual = np.linalg.norm(cv2.perspectiveTransform(src_full,H).reshape(-1,2)-dst_full.reshape(-1,2),axis=1)
        keep = residual < 5
        if kind == "current":
            current_inliers = keep.copy()
        summary[kind] = {
            "inliers": int(keep.sum()),
            "median_m": float(np.median(residual[keep])*processing.ortho_matcher.ORTHO_GSD) if keep.any() else None,
            "median_on_current_consensus_m": float(np.median(residual[current_inliers])*processing.ortho_matcher.ORTHO_GSD) if current_inliers.any() else None,
            "support": processing.ortho_matcher._spatial_support_metrics(src_full[keep],w,h),
        }
        registered = cv2.warpPerspective(image,T@H,(target.shape[1],target.shape[0]))
        valid = cv2.warpPerspective(np.full((h,w),255,np.uint8),T@H,(target.shape[1],target.shape[0]))>0
        blend=target.copy()
        blend[valid] = cv2.addWeighted(registered,.5,target,.5,0)[valid]
        cv2.imwrite(str(output / f"{path.stem}_{kind}_alignment.jpg"),blend)
    print(path.stem, json.dumps(summary), flush=True)
    (output / f"{path.stem}_alignment_comparison.json").write_text(json.dumps(summary,indent=2))


if __name__ == "__main__":
    parser=argparse.ArgumentParser()
    parser.add_argument("mode", choices=["seedlings","alignment"])
    parser.add_argument("images", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    for path in args.images:
        globals()[args.mode](path,args.output)
