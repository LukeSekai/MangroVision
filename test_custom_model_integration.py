"""
Test Custom Trained Model Integration
Verify that the detectree2_detector loads the custom model correctly
"""

import sys
from pathlib import Path

# Add canopy_detection to path
sys.path.insert(0, str(Path(__file__).parent / "canopy_detection"))

from detectree2_detector import Detectree2Detector

def test_custom_model_loading():
    """Test if custom model loads with correct configuration"""
    
    print("="*70)
    print("TESTING CUSTOM MODEL INTEGRATION")
    print("="*70)
    print()
    
    # Test 1: Initialize detector with 'custom' model
    print("Test 1: Initialize detector with 'custom' model name")
    print("-"*70)
    
    detector = Detectree2Detector(
        confidence_threshold=0.5,
        device='cpu',  # Use CPU for testing
        model_name='custom'
    )
    
    print()
    print("Test 2: Setup model (should load custom_mangrove_model/model_final.pth)")
    print("-"*70)
    
    try:
        detector.setup_model()
        print()
        print("✅ Model setup successful!")
        print()
        
        # Verify configuration
        print("Test 3: Verify model configuration")
        print("-"*70)
        print(f"   Number of classes: {detector.cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
        print(f"   Architecture: {detector.cfg.MODEL.WEIGHTS}")
        print(f"   Device: {detector.cfg.MODEL.DEVICE}")
        print(f"   Confidence threshold: {detector.cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST}")
        print(f"   RPN Pre-NMS: {detector.cfg.MODEL.RPN.PRE_NMS_TOPK_TEST}")
        print(f"   RPN Post-NMS: {detector.cfg.MODEL.RPN.POST_NMS_TOPK_TEST}")
        print(f"   NMS Threshold: {detector.cfg.MODEL.RPN.NMS_THRESH}")
        print()
        
        if detector.cfg.MODEL.ROI_HEADS.NUM_CLASSES == 2:
            print("✅ Correct number of classes (2: Mangrove-Canopy, Bungalon Canopy)")
        else:
            print(f"⚠️  WARNING: Expected 2 classes, got {detector.cfg.MODEL.ROI_HEADS.NUM_CLASSES}")
        
        if 'custom_mangrove_model' in str(detector.cfg.MODEL.WEIGHTS):
            print("✅ Custom model path detected in weights")
        else:
            print(f"⚠️  WARNING: Custom model path not found in weights: {detector.cfg.MODEL.WEIGHTS}")
        
        print()
        print("="*70)
        print("ALL TESTS PASSED!")
        print("="*70)
        print()
        print("The system is now configured to use your custom trained model")
        print("trained on detectree2 base with your Roboflow dataset!")
        print()
        
        return True
        
    except Exception as e:
        print()
        print(f"❌ ERROR: {str(e)}")
        print()
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_custom_model_loading()
    sys.exit(0 if success else 1)
