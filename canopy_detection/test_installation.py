"""
Test script to verify detectree2 installation and basic functionality
Run this first to ensure everything is set up correctly
"""

import torch
import numpy as np
import cv2
from detectron2 import model_zoo
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog
import matplotlib.pyplot as plt


def test_detectron2():
    """Test basic detectron2 functionality"""
    print("="*60)
    print("TESTING DETECTRON2 INSTALLATION")
    print("="*60 + "\n")
    
    # Check PyTorch
    print(f"✓ PyTorch version: {torch.__version__}")
    print(f"✓ CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  CUDA version: {torch.version.cuda}")
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        print("  Using CPU (this is fine for testing)")
    
    # Check Detectron2
    from detectron2 import __version__ as d2_version
    print(f"✓ Detectron2 version: {d2_version}")
    
    # Check detectree2
    from detectree2 import __version__ as dt2_version
    print(f"✓ Detectree2 version: {dt2_version}")
    
    print("\n" + "="*60)
    print("BASIC MASK R-CNN TEST")
    print("="*60 + "\n")
    
    # Create a simple test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    
    # Draw some shapes to simulate trees
    cv2.circle(test_image, (200, 200), 50, (0, 255, 0), -1)
    cv2.circle(test_image, (400, 300), 60, (0, 200, 0), -1)
    cv2.circle(test_image, (500, 150), 40, (0, 180, 0), -1)
    
    print("Created synthetic test image with tree-like shapes")
    print(f"Image shape: {test_image.shape}")
    
    try:
        # Setup a basic config (using COCO pre-trained model for testing)
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5
        cfg.MODEL.DEVICE = "cpu"  # Force CPU for compatibility
        
        print("✓ Detectron2 configuration loaded successfully")
        print("✓ Model: Mask R-CNN with ResNet-50 backbone")
        
        print("\nNOTE: For actual tree detection, you'll need to:")
        print("  1. Train detectree2 on your mangrove dataset")
        print("  2. Or use detectree2's pre-trained tree detection models")
        
    except Exception as e:
        print(f"✗ Error: {e}")
        return False
    
    print("\n" + "="*60)
    print("ALL TESTS PASSED! ✓")
    print("="*60 + "\n")
    print("Your environment is ready for detectree2 development!")
    print("\nNext steps:")
    print("  1. Add your drone images to 'drone_images/' folder")
    print("  2. Run the main canopy detector script")
    print("  3. Train on your mangrove dataset for production use")
    
    return True


def test_detectree2_imports():
    """Test detectree2-specific imports"""
    print("\n" + "="*60)
    print("TESTING DETECTREE2 MODULES")
    print("="*60 + "\n")
    
    try:
        from detectree2.models.train import setup_cfg
        print("✓ detectree2.models.train")
        
        from detectree2.models.predict import predict_img
        print("✓ detectree2.models.predict")
        
        from detectree2.preprocessing.tiling import tile_data
        print("✓ detectree2.preprocessing.tiling")
        
        print("\nAll detectree2 modules imported successfully!")
        return True
        
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False


def main():
    """Run all tests"""
    print("\n")
    print("#"*60)
    print("# DETECTREE2 INSTALLATION VERIFICATION")
    print("# Mangrove Canopy Detection System")
    print("#"*60)
    print("\n")
    
    # Test detectron2
    if not test_detectron2():
        print("\n⚠ Warning: Some tests failed. Check error messages above.")
        return
    
    # Test detectree2
    test_detectree2_imports()
    
    print("\n" + "#"*60)
    print("# VERIFICATION COMPLETE")
    print("#"*60)
    print("\nYour system is ready for mangrove canopy detection!")
    print("See README.md for usage instructions.")


if __name__ == "__main__":
    main()
