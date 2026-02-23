"""Quick test for ortho_matcher - test multiple images/datasets"""
import sys
sys.path.insert(0, '.')
import cv2
from canopy_detection.ortho_matcher import match_drone_to_ortho, drone_pixel_to_gps_via_homography, gps_to_ortho_pixel

# For context: orthophoto bounds GPS
# Top-left:     lat=10.7813, lon=122.6247
# Bottom-right: lat=10.7798, lon=122.6258

test_cases = [
    ("drone_images/dataset_with_gps/frame_0049.jpg",      10.781000, 122.625397, 0.014),
    ("drone_images/dataset_with_gps_0030/frame_0049.jpg", 10.780497, 122.625200, 0.014),
]

for imgpath, lat, lon, gsd in test_cases:
    import os
    if not os.path.exists(imgpath):
        print(f"SKIP {imgpath} (not found)")
        continue
    drone_img = cv2.imread(imgpath)
    if drone_img is None:
        print(f"SKIP {imgpath} (could not read)")
        continue
    cx, cy = gps_to_ortho_pixel(lat, lon)
    print(f"\n--- {imgpath} ---")
    print(f"  Drone GPS: lat={lat}, lon={lon} -> ortho pixel ({cx:.0f},{cy:.0f})")
    result = match_drone_to_ortho(drone_img, lat, lon, gsd)
    if result['success']:
        h, w = drone_img.shape[:2]
        clat, clon = drone_pixel_to_gps_via_homography(w/2, h/2, result['H'])
        print(f"  OK: inliers={result['inliers']}/{result['total_matches']}, conf={result['confidence']:.1%}, heading={result['heading']:.1f}deg")
        print(f"  Center GPS check: predicted ({clat:.5f},{clon:.5f}), expected ({lat},{lon}), err_m={(abs(clat-lat)*111320):.1f}m lat, {(abs(clon-lon)*111320*0.98):.1f}m lon")
    else:
        print(f"  FAIL: {result['error']}")
