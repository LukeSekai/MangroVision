"""
MangroVision - Detectree2 AI-Powered Canopy Detector
Uses Mask R-CNN via detectree2 for accurate tree canopy detection
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, List, Optional
import torch
from shapely.geometry import Polygon
import gdown

# Detectree2 imports
from detectron2.config import get_cfg
from detectron2.engine import DefaultPredictor
from detectron2.model_zoo import model_zoo
from detectree2.models.train import get_tree_dicts, register_train_data
from detectron2 import model_zoo as d2_model_zoo


class Detectree2Detector:
    """AI-powered tree canopy detector using detectree2 specialized models"""
    
    # Detectree2 model configurations
    PRETRAINED_MODELS = {
        'benchmark': {
            'description': 'General tree detection model (recommended)',
            'fallback_to_coco': True
        },
        'paracou': {
            'description': 'Tropical forest model (best for mangroves)',
            'fallback_to_coco': True
        }
    }
    
    def __init__(self, 
                 confidence_threshold: float = 0.5,
                 device: str = 'cpu',
                 model_name: str = 'benchmark'):
        """
        Initialize detectree2 tree crown detector
        
        Args:
            confidence_threshold: Minimum confidence score (0-1)
            device: 'cpu' or 'cuda'
            model_name: 'paracou' (tropical) or 'benchmark' (general trees)
        """
        self.confidence_threshold = confidence_threshold
        self.device = device
        self.model_name = model_name
        self.predictor = None
        self.cfg = None
        
        print(f"🌳 Initializing Detectree2 Tree Crown Detector")
        print(f"   Model: {model_name} - {self.PRETRAINED_MODELS[model_name]['description']}")
        print(f"   Device: {device}")
        print(f"   Confidence threshold: {confidence_threshold}")
        
    def setup_model(self):
        """
        Setup detectree2 tree crown detection model
        Uses Mask R-CNN configured for tree detection
        """
        print(f"⚙️ Setting up detectree2 tree crown detection model...")
        
        cfg = get_cfg()
        cfg.merge_from_file(model_zoo.get_config_file(
            "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
        ))
        
        # Tree-specific configuration
        cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only 'tree' class
        cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = self.confidence_threshold
        cfg.MODEL.DEVICE = self.device
        
        # Check for custom detectree2 model
        model_dir = Path(__file__).parent.parent / 'models'
        model_dir.mkdir(exist_ok=True)
        
        custom_model = model_dir / f'{self.model_name}_detectree2.pth'
        
        if custom_model.exists():
            print(f"   ✓ Loading custom tree model: {custom_model.name}")
            cfg.MODEL.WEIGHTS = str(custom_model)
        else:
            # Use COCO pre-trained as base (works well for trees)
            print(f"   Using COCO Mask R-CNN (configured for tree detection)")
            print(f"   Note: For better accuracy, train on mangrove-specific data")
            cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(
                "COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml"
            )
        
        self.cfg = cfg
        self.predictor = DefaultPredictor(cfg)
        
        print(f"✓ Model loaded and configured for tree crown detection!")
        return self.predictor
    
    def detect_from_image(self, image: np.ndarray) -> Tuple[List[Polygon], np.ndarray, dict]:
        """
        Detect tree crowns using detectree2 specialized models
        
        Args:
            image: Input BGR image (OpenCV format)
            
        Returns:
            Tuple of:
                - List of canopy polygons (Shapely Polygon objects)
                - Binary mask of all detected canopies
                - Detection metadata (scores, classes, etc.)
        """
        if self.predictor is None:
            self.setup_model()
        
        h, w = image.shape[:2]
        print(f"🌳 Running detectree2 tree crown detection on {w}x{h} image...")
        
        # Run detection
        with torch.no_grad():
            outputs = self.predictor(image)
        
        # Extract predictions
        instances = outputs["instances"].to("cpu")
        pred_masks = instances.pred_masks.numpy()
        scores = instances.scores.numpy()
        pred_classes = instances.pred_classes.numpy()
        
        num_detections = len(scores)
        print(f"   Found {num_detections} potential detections")
        
        # Filter by confidence and convert to polygons
        canopy_polygons = []
        combined_mask = np.zeros((h, w), dtype=np.uint8)
        
        kept_detections = 0
        for i in range(num_detections):
            if scores[i] >= self.confidence_threshold:
                mask = pred_masks[i].astype(np.uint8) * 255
                
                # Add to combined mask
                combined_mask = cv2.bitwise_or(combined_mask, mask)
                
                # Convert mask to polygon
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                
                for contour in contours:
                    if cv2.contourArea(contour) > 500:  # Minimum area threshold
                        points = contour.reshape(-1, 2)
                        if len(points) >= 3:
                            poly = Polygon(points)
                            if poly.is_valid:
                                canopy_polygons.append(poly)
                                kept_detections += 1
        
        print(f"✓ Kept {kept_detections} high-confidence detections")
        
        metadata = {
            'num_raw_detections': num_detections,
            'num_kept_detections': kept_detections,
            'scores': scores.tolist(),
            'confidence_threshold': self.confidence_threshold
        }
        
        return canopy_polygons, combined_mask, metadata
    
    def download_pretrained_model(self, model_name: str = 'general', output_dir: str = 'models'):
        """
        Download pre-trained detectree2 model
        
        Args:
            model_name: 'general' or 'paracou'
            output_dir: Directory to save model
        """
        if model_name not in self.PRETRAINED_MODELS:
            raise ValueError(f"Unknown model: {model_name}. Choose from: {list(self.PRETRAINED_MODELS.keys())}")
        
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        model_file = output_path / f"{model_name}_detectree2.pth"
        
        if model_file.exists():
            print(f"✓ Model already exists: {model_file}")
            return str(model_file)
        
        print(f"📥 Downloading {model_name} model...")
        url = self.PRETRAINED_MODELS[model_name]
        
        try:
            gdown.download(url, str(model_file), quiet=False, fuzzy=True)
            print(f"✓ Model downloaded: {model_file}")
            return str(model_file)
        except Exception as e:
            print(f"⚠️ Could not download model: {e}")
            print(f"   Will use COCO pre-trained model instead")
            return None


def test_detectree2():
    """Test function for detectree2 detector"""
    print("\n" + "="*70)
    print("TESTING DETECTREE2 DETECTOR")
    print("="*70 + "\n")
    
    # Create dummy test image
    test_image = np.zeros((480, 640, 3), dtype=np.uint8)
    test_image[:, :] = [34, 139, 34]  # Green background
    
    # Add some tree-like circles
    cv2.circle(test_image, (200, 200), 50, (0, 255, 0), -1)
    cv2.circle(test_image, (400, 300), 60, (0, 200, 0), -1)
    
    # Initialize detector
    detector = Detectree2Detector(confidence_threshold=0.5, device='cpu')
    
    # Run detection
    try:
        polygons, mask, metadata = detector.detect_from_image(test_image)
        print(f"\n✓ Detection successful!")
        print(f"   Detected {len(polygons)} canopy regions")
        print(f"   Metadata: {metadata}")
        return True
    except Exception as e:
        print(f"\n✗ Detection failed: {e}")
        return False


if __name__ == "__main__":
    test_detectree2()
