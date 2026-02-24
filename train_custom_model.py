"""
Train Custom Mangrove Detection Model using Transfer Learning
Fine-tune detectree2 on your Roboflow dataset
"""

import os
import sys
from pathlib import Path
import json
import torch
from detectron2.data.datasets import register_coco_instances
from detectron2.data import DatasetCatalog, MetadataCatalog
from detectron2.config import get_cfg
from detectron2.engine import DefaultTrainer
from detectron2 import model_zoo
from detectron2.evaluation import COCOEvaluator
import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ========================================
# CONFIGURATION
# ========================================

# Dataset paths
DATASET_DIR = Path("training_data/roboflow_dataset")
TRAIN_ANNOTATIONS = DATASET_DIR / "train" / "_annotations.coco.json"
TRAIN_IMAGES = DATASET_DIR / "train"
VALID_ANNOTATIONS = DATASET_DIR / "valid" / "_annotations.coco.json"
VALID_IMAGES = DATASET_DIR / "valid"

# Output directory for trained model
OUTPUT_DIR = Path("models/custom_mangrove_model")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training parameters
MAX_ITERATIONS = 3000  # Adjust based on dataset size (240 images = ~3000 iterations is good)
LEARNING_RATE = 0.00025  # Lower learning rate for transfer learning
BATCH_SIZE = 2  # Images per batch (adjust based on GPU memory)
NUM_WORKERS = 4  # Data loading workers
CHECKPOINT_PERIOD = 500  # Save checkpoint every N iterations

# Model configuration
BASE_MODEL = "COCO-InstanceSegmentation/mask_rcnn_R_101_FPN_3x.yaml"  # Architecture (R-101 for detectree2)
# Use detectree2's tropical model as base (TRANSFER LEARNING FROM CANOPY-AWARE MODEL)
USE_DETECTREE2_BASE = True  # TRUE = Use detectree2 base model (better!)
DETECTREE2_MODEL_PATH = "models/230103_randresize_full.pth"  # Detectree2 tropical model

# ========================================
# REGISTER DATASETS
# ========================================

def register_datasets():
    """Register training and validation datasets with Detectron2"""
    
    logger.info("Registering datasets...")
    
    # Clear any existing registrations
    for dataset_name in ["mangrove_train", "mangrove_val"]:
        if dataset_name in DatasetCatalog.list():
            DatasetCatalog.remove(dataset_name)
            MetadataCatalog.remove(dataset_name)
    
    # Register training set
    register_coco_instances(
        "mangrove_train",
        {},
        str(TRAIN_ANNOTATIONS),
        str(TRAIN_IMAGES)
    )
    
    # Register validation set
    register_coco_instances(
        "mangrove_val",
        {},
        str(VALID_ANNOTATIONS),
        str(VALID_IMAGES)
    )
    
    # Get number of classes from annotations
    with open(TRAIN_ANNOTATIONS) as f:
        coco_data = json.load(f)
        num_classes = len(coco_data['categories'])
        class_names = [cat['name'] for cat in coco_data['categories']]
    
    # Set metadata
    MetadataCatalog.get("mangrove_train").set(thing_classes=class_names)
    MetadataCatalog.get("mangrove_val").set(thing_classes=class_names)
    
    logger.info(f"✅ Registered datasets with {num_classes} classes: {class_names}")
    
    return num_classes, class_names


# ========================================
# TRAINING CONFIGURATION
# ========================================

def setup_config(num_classes):
    """Setup Detectron2 configuration for training"""
    
    logger.info("Setting up training configuration...")
    
    cfg = get_cfg()
    
    # Load base model config
    cfg.merge_from_file(model_zoo.get_config_file(BASE_MODEL))
    
    # ========================================
    # DATA AUGMENTATION (crucial for 240 images!)
    # ========================================
    from detectron2.data import transforms as T
    
    logger.info("Enabling data augmentation for small dataset...")
    
    # Enable augmentation pipeline
    cfg.INPUT.MIN_SIZE_TRAIN = (640, 672, 704, 736, 768, 800)  # Random scale
    cfg.INPUT.MAX_SIZE_TRAIN = 1333
    cfg.INPUT.MIN_SIZE_TEST = 800
    cfg.INPUT.MAX_SIZE_TEST = 1333
    
    # Random flip
    cfg.INPUT.RANDOM_FLIP = "horizontal"  # Horizontal flip for aerial imagery
    
    # Color jitter (important for varying lighting conditions)
    cfg.INPUT.BRIGHTNESS = (0.8, 1.2)  # Brightness variation
    cfg.INPUT.CONTRAST = (0.8, 1.2)    # Contrast variation  
    cfg.INPUT.SATURATION = (0.8, 1.2)  # Saturation variation
    
    logger.info("   ✓ Random scaling (0.8-1.2x)")
    logger.info("   ✓ Horizontal flip")
    logger.info("   ✓ Brightness/Contrast/Saturation jitter")
    
    # ========================================
    # DATASET CONFIGURATION
    # ========================================
    
    # Dataset configuration
    cfg.DATASETS.TRAIN = ("mangrove_train",)
    cfg.DATASETS.TEST = ("mangrove_val",)
    
    # Dataloader
    cfg.DATALOADER.NUM_WORKERS = NUM_WORKERS
    
    # Model weights - USE DETECTREE2 BASE MODEL
    detectree2_path = Path(DETECTREE2_MODEL_PATH)
    
    if USE_DETECTREE2_BASE and detectree2_path.exists():
        logger.info(f"✅ Using detectree2 TROPICAL BASE MODEL (Transfer Learning)")
        logger.info(f"   Path: {DETECTREE2_MODEL_PATH}")
        logger.info(f"   Model already knows: Tree canopy structures from aerial imagery")
        logger.info(f"   Will learn: Your specific Mangrove vs Bungalon classes")
        cfg.MODEL.WEIGHTS = str(detectree2_path)
    elif USE_DETECTREE2_BASE and not detectree2_path.exists():
        logger.error(f"❌ DETECTREE2 MODEL NOT FOUND: {detectree2_path.absolute()}")
        logger.error(f"")
        logger.error(f"Please download the model:")
        logger.error(f"1. Visit: https://github.com/PatBall1/detectree2")
        logger.error(f"2. Download: 230103_randresize_full.pth from model zoo")
        logger.error(f"3. Place in: {detectree2_path.parent.absolute()}")
        logger.error(f"")
        raise FileNotFoundError(f"Detectree2 model not found: {detectree2_path}")
    else:
        logger.info(f"Using COCO pre-trained model (fallback)")
        cfg.MODEL.WEIGHTS = model_zoo.get_checkpoint_url(BASE_MODEL)
    
    # Model architecture
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = num_classes  # Your custom classes
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 128  # RoIs per image
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # Detection threshold
    
    # Training parameters
    cfg.SOLVER.IMS_PER_BATCH = BATCH_SIZE
    cfg.SOLVER.BASE_LR = LEARNING_RATE
    cfg.SOLVER.MAX_ITER = MAX_ITERATIONS
    cfg.SOLVER.STEPS = (int(MAX_ITERATIONS * 0.7), int(MAX_ITERATIONS * 0.9))  # LR decay steps
    cfg.SOLVER.GAMMA = 0.1  # LR decay factor
    cfg.SOLVER.CHECKPOINT_PERIOD = CHECKPOINT_PERIOD
    
    # Output directory
    cfg.OUTPUT_DIR = str(OUTPUT_DIR)
    
    # Device
    cfg.MODEL.DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    logger.info(f"✅ Configuration ready")
    logger.info(f"   Device: {cfg.MODEL.DEVICE}")
    logger.info(f"   Classes: {num_classes}")
    logger.info(f"   Iterations: {MAX_ITERATIONS}")
    logger.info(f"   Learning Rate: {LEARNING_RATE}")
    logger.info(f"   Batch Size: {BATCH_SIZE}")
    
    return cfg


# ========================================
# CUSTOM TRAINER WITH EVALUATION
# ========================================

class MangroveTrainer(DefaultTrainer):
    """Custom trainer with validation evaluation"""
    
    @classmethod
    def build_evaluator(cls, cfg, dataset_name, output_folder=None):
        if output_folder is None:
            output_folder = os.path.join(cfg.OUTPUT_DIR, "inference")
        return COCOEvaluator(dataset_name, output_dir=output_folder)


# ========================================
# MAIN TRAINING FUNCTION
# ========================================

def train_model():
    """Main training function"""
    
    print("\n" + "="*70)
    print("🌿 MangroVision - Custom Model Training")
    print("   Transfer Learning with Detectree2")
    print("="*70)
    
    # Verify dataset exists
    if not TRAIN_ANNOTATIONS.exists():
        logger.error(f"❌ Training annotations not found: {TRAIN_ANNOTATIONS}")
        logger.error("   Please run: python download_roboflow_dataset.py")
        return False
    
    if not VALID_ANNOTATIONS.exists():
        logger.error(f"❌ Validation annotations not found: {VALID_ANNOTATIONS}")
        return False
    
    # Check GPU availability
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        gpu_memory = torch.cuda.get_device_properties(0).total_memory / 1e9
        logger.info(f"✅ GPU available: {gpu_name} ({gpu_memory:.1f} GB)")
    else:
        logger.warning("⚠️ No GPU detected - training will be SLOW on CPU")
        logger.warning("   Consider using Google Colab with GPU for faster training")
        response = input("\n   Continue with CPU training? (y/n): ")
        if response.lower() != 'y':
            return False
    
    # Register datasets
    num_classes, class_names = register_datasets()
    
    # Setup configuration
    cfg = setup_config(num_classes)
    
    # Create trainer
    trainer = MangroveTrainer(cfg)
    trainer.resume_or_load(resume=False)
    
    # Start training
    print("\n" + "="*70)
    print("🚀 Starting Training...")
    print("="*70)
    print(f"   Training images: 240")
    print(f"   Validation images: 15")
    print(f"   Classes: {', '.join(class_names)}")
    print(f"   Iterations: {MAX_ITERATIONS}")
    print(f"   Estimated time: ~2-4 hours (GPU) or ~12-24 hours (CPU)")
    print("="*70)
    print("\n⏳ Training in progress... (check logs above)")
    print("   Press Ctrl+C to stop training\n")
    
    try:
        trainer.train()
        
        print("\n" + "="*70)
        print("✅ Training Complete!")
        print("="*70)
        print(f"   Model saved to: {OUTPUT_DIR}")
        print(f"   Final model: {OUTPUT_DIR / 'model_final.pth'}")
        print("\n📖 Next Steps:")
        print("   1. Test your model: python test_custom_model.py")
        print("   2. Update app.py to use your custom model")
        print("   3. Compare accuracy with base model")
        print("="*70)
        
        return True
        
    except KeyboardInterrupt:
        print("\n\n⚠️ Training interrupted by user")
        print(f"   Progress saved to: {OUTPUT_DIR}")
        print("   You can resume training later")
        return False
    
    except Exception as e:
        logger.error(f"\n❌ Training failed: {e}")
        logger.error("   Check the error above for details")
        return False


# ========================================
# ENTRY POINT
# ========================================

if __name__ == "__main__":
    print("\n")
    success = train_model()
    print("\n")
    
    if not success:
        sys.exit(1)
