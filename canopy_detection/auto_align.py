"""
Automatic Image Alignment using Feature Matching
Matches drone image with orthophoto to auto-detect camera heading
"""

import cv2
import numpy as np
import requests
from io import BytesIO
from PIL import Image
import math


def download_orthophoto_tile(center_lat, center_lon, zoom=19, tile_size=640):
    """
    Download orthophoto tile from XYZ server for feature matching
    
    Args:
        center_lat: Center latitude
        center_lon: Center longitude
        zoom: Zoom level (higher = more detail)
        tile_size: Size of tile to download
        
    Returns:
        numpy array of orthophoto image
    """
    # Convert lat/lon to tile coordinates
    lat_rad = math.radians(center_lat)
    n = 2.0 ** zoom
    x_tile = int((center_lon + 180.0) / 360.0 * n)
    y_tile = int((1.0 - math.asinh(math.tan(lat_rad)) / math.pi) / 2.0 * n)
    
    # Download tile from local server
    url = f"http://localhost:8080/{zoom}/{x_tile}/{y_tile}.png"
    
    try:
        response = requests.get(url, timeout=10)
        if response.status_code == 200:
            img = Image.open(BytesIO(response.content))
            return np.array(img)
        else:
            print(f"Failed to download tile: {response.status_code}")
            return None
    except Exception as e:
        print(f"Error downloading orthophoto: {e}")
        return None


def detect_camera_heading(drone_image, center_lat, center_lon, max_dimension=800):
    """
    Automatically detect camera heading by matching with orthophoto
    
    Args:
        drone_image: BGR drone image (numpy array)
        center_lat: GPS latitude of image center
        center_lon: GPS longitude of image center
        max_dimension: Resize images to this max dimension for faster processing
        
    Returns:
        heading_degrees: Detected camera heading (0=North, 90=East, 180=South, 270=West)
        confidence: Confidence score (0-1)
        visualization: Debug image showing matches
    """
    print(f"\n🔍 Auto-detecting camera heading via feature matching...")
    
    # Step 1: Download orthophoto tile
    print(f"   Downloading orthophoto tile...")
    ortho = download_orthophoto_tile(center_lat, center_lon, zoom=19)
    
    if ortho is None:
        print(f"   ❌ Failed to download orthophoto")
        return None, 0.0, None
    
    # Convert orthophoto to BGR if needed
    if len(ortho.shape) == 2:
        ortho = cv2.cvtColor(ortho, cv2.COLOR_GRAY2BGR)
    elif ortho.shape[2] == 4:
        ortho = cv2.cvtColor(ortho, cv2.COLOR_RGBA2BGR)
    
    # Step 2: Resize images for faster processing
    h, w = drone_image.shape[:2]
    if max(h, w) > max_dimension:
        scale = max_dimension / max(h, w)
        drone_resized = cv2.resize(drone_image, None, fx=scale, fy=scale)
    else:
        drone_resized = drone_image.copy()
    
    h_o, w_o = ortho.shape[:2]
    if max(h_o, w_o) > max_dimension:
        scale_o = max_dimension / max(h_o, w_o)
        ortho_resized = cv2.resize(ortho, None, fx=scale_o, fy=scale_o)
    else:
        ortho_resized = ortho.copy()
    
    print(f"   Drone image: {drone_resized.shape[:2]}, Ortho: {ortho_resized.shape[:2]}")
    
    # Step 3: Convert to grayscale
    drone_gray = cv2.cvtColor(drone_resized, cv2.COLOR_BGR2GRAY)
    ortho_gray = cv2.cvtColor(ortho_resized, cv2.COLOR_BGR2GRAY)
    
    # Step 4: Detect features using ORB (fast and patent-free)
    print(f"   Detecting features...")
    orb = cv2.ORB_create(nfeatures=2000, scaleFactor=1.2, nlevels=8)
    
    kp1, des1 = orb.detectAndCompute(drone_gray, None)
    kp2, des2 = orb.detectAndCompute(ortho_gray, None)
    
    print(f"   Found {len(kp1)} features in drone image")
    print(f"   Found {len(kp2)} features in orthophoto")
    
    if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
        print(f"   ❌ Not enough features detected")
        return None, 0.0, None
    
    # Step 5: Match features
    print(f"   Matching features...")
    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)
    matches = bf.knnMatch(des1, des2, k=2)
    
    # Apply Lowe's ratio test
    good_matches = []
    for match_pair in matches:
        if len(match_pair) == 2:
            m, n = match_pair
            if m.distance < 0.75 * n.distance:
                good_matches.append(m)
    
    print(f"   Found {len(good_matches)} good matches")
    
    if len(good_matches) < 10:
        print(f"   ❌ Not enough good matches")
        return None, 0.0, None
    
    # Step 6: Extract matched point coordinates
    pts1 = np.float32([kp1[m.queryIdx].pt for m in good_matches])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good_matches])
    
    # Step 7: Calculate transformation matrix
    print(f"   Calculating rotation...")
    M, mask = cv2.estimateAffinePartial2D(pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=5.0)
    
    if M is None:
        print(f"   ❌ Could not estimate transformation")
        return None, 0.0, None
    
    # Extract rotation angle from affine matrix
    # M = [[cos(θ), -sin(θ), tx],
    #      [sin(θ),  cos(θ), ty]]
    rotation_rad = math.atan2(M[1, 0], M[0, 0])
    rotation_deg = math.degrees(rotation_rad)
    
    # Normalize to 0-360
    heading = rotation_deg % 360
    
    # Calculate confidence based on inliers
    inliers = np.sum(mask)
    confidence = inliers / len(good_matches)
    
    print(f"   ✓ Detected heading: {heading:.1f}° (confidence: {confidence:.2%})")
    
    # Step 8: Create visualization
    matches_img = cv2.drawMatches(
        drone_resized, kp1,
        ortho_resized, kp2,
        good_matches[:50],  # Show top 50 matches
        None,
        flags=cv2.DrawMatchesFlags_NOT_DRAW_SINGLE_POINTS
    )
    
    # Add text overlay
    cv2.putText(matches_img, f"Heading: {heading:.1f} degrees", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    cv2.putText(matches_img, f"Matches: {len(good_matches)} ({confidence:.0%} confidence)", (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    return heading, confidence, matches_img


def snap_to_cardinal(heading, threshold=22.5):
    """
    Snap heading to nearest cardinal direction
    
    Args:
        heading: Heading in degrees (0-360)
        threshold: Degrees tolerance for snapping
        
    Returns:
        Snapped heading (0, 90, 180, or 270)
    """
    cardinals = [0, 90, 180, 270]
    
    # Find closest cardinal
    min_diff = 360
    snapped = heading
    
    for cardinal in cardinals:
        diff = abs(heading - cardinal)
        # Handle wrap-around
        if diff > 180:
            diff = 360 - diff
        
        if diff < min_diff and diff <= threshold:
            min_diff = diff
            snapped = cardinal
    
    return snapped


if __name__ == "__main__":
    # Test with a sample image
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python auto_align.py <image_path> <lat> <lon>")
        sys.exit(1)
    
    image = cv2.imread(sys.argv[1])
    lat = float(sys.argv[2])
    lon = float(sys.argv[3])
    
    heading, conf, vis = detect_camera_heading(image, lat, lon)
    
    if heading is not None:
        snapped = snap_to_cardinal(heading)
        print(f"\nDetected: {heading:.1f}°")
        print(f"Snapped to: {snapped}° (cardinal direction)")
        print(f"Confidence: {conf:.2%}")
        
        cv2.imwrite("feature_matching_result.png", vis)
        print(f"Saved visualization to: feature_matching_result.png")
    else:
        print("Failed to detect heading")
