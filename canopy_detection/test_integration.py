"""
Test script to verify pure detectree2 AI detection integration
"""

import sys
from pathlib import Path
import numpy as np
import cv2

# Add canopy_detection to path
sys.path.append(str(Path(__file__).parent))

from canopy_detector_hexagon import HexagonDetector


def test_detectree2_ai():
    """Test detectree2 AI detection"""
    print("\n" + "="*70)
    print("DETECTREE2 AI DETECTION TEST")
    print("="*70 + "\n")
    
    # Create test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:, :] = [200, 200, 200]  # Gray background
    
    # Add green vegetation areas
    cv2.circle(test_image, (200, 200), 50, (50, 180, 50), -1)
    cv2.circle(test_image, (400, 300), 60, (40, 160, 40), -1)
    
    try:
        print("Creating MangroVision detector with detectree2 AI...")
        detector = HexagonDetector(
            altitude_m=6.0,
            drone_model='GENERIC_4K',
            ai_confidence=0.5,
            model_name='benchmark'
        )
        
        # Calculate GSD
        h, w = test_image.shape[:2]
        gsd = detector.calculate_gsd(w, h)
        print(f"GSD: {gsd:.5f} m/pixel\n")
        
        # Run detection
        print("Running detectree2 AI detection (this may take a moment)...")
        polygons, mask = detector.detect_canopies(test_image)
        
        print(f"\n✓ Detectree2 AI Detection Complete!")
        print(f"   Detected {len(polygons)} tree crowns")
        print(f"   Mask shape: {mask.shape}")
        
        return True
        
    except Exception as e:
        print(f"\n✗ Detection failed: {e}")
        print(f"   Please ensure detectree2 and detectron2 are properly installed")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("\n" + "="*70)
    print("MANGROVISION - PURE DETECTREE2 AI SYSTEM TEST")
    print("="*70)
    
    # Test detectree2 AI
    success = test_detectree2_ai()
    
    # Summary
    print("\n" + "="*70)
    print("TEST SUMMARY")
    print("="*70)
    print(f"{'✓' if success else '✗'} Detectree2 AI Detection: {'PASSED' if success else 'FAILED'}")
    print("="*70)
    
    if success:
        print("\n✓ MangroVision AI system is ready to use!")
        print("  - Detectree2 AI detection working")
        print("  - Tree crown detection operational")
        print("\nRun: streamlit run app.py")
    else:
        print("\n✗ Issues detected - please check detectree2 installation")

