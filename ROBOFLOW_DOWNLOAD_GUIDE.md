# 📥 Download Roboflow Dataset for Training

## Quick Guide: Get Your Annotated Dataset from Roboflow

---

## 🎯 Overview

You've annotated your mangrove images in Roboflow. Now you need to:
1. Download the dataset in the correct format (COCO)
2. Place it in your project structure
3. Train detectree2 with your custom data

---

## 📦 Method 1: Download via Roboflow Python API (Recommended)

### Step 1: Install Roboflow Package

```powershell
# Make sure your virtual environment is activated
cd C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env
.\Scripts\activate

# Install roboflow
pip install roboflow
```

### Step 2: Get Your API Key

1. Go to [Roboflow](https://roboflow.com/)
2. Log in to your account
3. Click your profile icon (top right)
4. Select "Account" → "Roboflow API"
5. Copy your **Private API Key**

### Step 3: Create Download Script

Create a file called `download_roboflow_dataset.py` in your MangroVision folder:

```python
"""
Download annotated dataset from Roboflow
"""
from roboflow import Roboflow
from pathlib import Path
import shutil

# Configuration
API_KEY = "LJti0618t62VdAV816QP"  # Replace with your actual API key
WORKSPACE_NAME = "Finding"  # Replace with your workspace name
PROJECT_NAME = "Practice_annotate"  # Replace with your project name
VERSION = 1  # Your dataset version number

# Output directory
OUTPUT_DIR = Path("training_data/roboflow_dataset")

def download_dataset():
    """Download dataset from Roboflow in COCO format"""
    
    print("🌿 MangroVision - Roboflow Dataset Downloader")
    print("=" * 60)
    
    # Initialize Roboflow
    print(f"\n📡 Connecting to Roboflow...")
    rf = Roboflow(api_key=API_KEY)
    
    # Get your project
    print(f"📂 Loading project: {WORKSPACE_NAME}/{PROJECT_NAME}")
    project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
    
    # Get specific version
    print(f"📊 Loading version {VERSION}...")
    dataset = project.version(VERSION)
    
    # Download in COCO format (best for detectree2/Mask R-CNN)
    print(f"\n⬇️ Downloading dataset in COCO format...")
    print(f"📁 Saving to: {OUTPUT_DIR}")
    
    # Create output directory
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # Download
    dataset.download(
        model_format="coco",  # COCO format for instance segmentation
        location=str(OUTPUT_DIR)
    )
    
    print("\n✅ Download complete!")
    print(f"\n📁 Dataset structure:")
    print(f"   {OUTPUT_DIR}/")
    print(f"   ├── train/")
    print(f"   │   ├── _annotations.coco.json")
    print(f"   │   └── *.jpg")
    print(f"   ├── valid/")
    print(f"   │   ├── _annotations.coco.json")
    print(f"   │   └── *.jpg")
    print(f"   └── test/  (if available)")
    
    # Show statistics
    train_dir = OUTPUT_DIR / "train"
    valid_dir = OUTPUT_DIR / "valid"
    
    if train_dir.exists():
        train_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
        print(f"\n📊 Training images: {len(train_images)}")
    
    if valid_dir.exists():
        valid_images = list(valid_dir.glob("*.jpg")) + list(valid_dir.glob("*.png"))
        print(f"📊 Validation images: {len(valid_images)}")
    
    print("\n✅ Ready for training!")
    print("   Next step: Run the training script")

if __name__ == "__main__":
    download_dataset()
```

### Step 4: Configure & Run

1. **Edit the script:**
   - Replace `YOUR_API_KEY_HERE` with your actual API key
   - Replace `your-workspace` with your Roboflow workspace name
   - Replace `mangrove-detection` with your project name
   - Update `VERSION` number if needed

2. **Find your project details:**
   - Go to your Roboflow project
   - Look at the URL: `roboflow.com/YOUR-WORKSPACE/YOUR-PROJECT/VERSION`
   - Copy those values

3. **Run the download:**

```powershell
cd MangroVision
python download_roboflow_dataset.py
```

---

## 📦 Method 2: Manual Download via Web Interface

### Step 1: Go to Your Roboflow Project

1. Log in to [Roboflow](https://roboflow.com/)
2. Select your mangrove detection project
3. Click on the version you want to download

### Step 2: Export in COCO Format

1. Click the **"Download"** button
2. Select format: **"COCO"** ⚠️ Important: NOT "YOLO" or "Pascal VOC"
3. Click **"Download ZIP"**
4. Save the ZIP file to your computer

### Step 3: Extract to Project

```powershell
# Create training data folder
cd MangroVision
mkdir training_data
mkdir training_data\roboflow_dataset

# Extract the downloaded ZIP to training_data\roboflow_dataset\
# You should have:
#   training_data\roboflow_dataset\
#   ├── train\
#   │   ├── _annotations.coco.json
#   │   └── (images)
#   └── valid\
#       ├── _annotations.coco.json
#       └── (images)
```

---

## 🗂️ Expected Dataset Structure

After download, you should have:

```
MangroVision/
├── training_data/
│   └── roboflow_dataset/
│       ├── train/
│       │   ├── _annotations.coco.json  ✅ COCO annotations
│       │   ├── image_001.jpg
│       │   ├── image_002.jpg
│       │   └── ...
│       ├── valid/
│       │   ├── _annotations.coco.json  ✅ COCO annotations
│       │   ├── image_050.jpg
│       │   └── ...
│       └── test/  (optional)
│           ├── _annotations.coco.json
│           └── ...
```

---

## ✅ Verify Download

Quick check to ensure everything is correct:

```python
# Save as verify_roboflow_dataset.py
import json
from pathlib import Path

def verify_dataset():
    """Verify that the dataset is properly formatted"""
    
    dataset_dir = Path("training_data/roboflow_dataset")
    
    print("🔍 Verifying Roboflow Dataset")
    print("=" * 60)
    
    # Check directories
    train_dir = dataset_dir / "train"
    valid_dir = dataset_dir / "valid"
    
    if not train_dir.exists():
        print("❌ train/ folder not found!")
        return False
    
    if not valid_dir.exists():
        print("❌ valid/ folder not found!")
        return False
    
    print("✅ Directories found")
    
    # Check annotations
    train_ann = train_dir / "_annotations.coco.json"
    valid_ann = valid_dir / "_annotations.coco.json"
    
    if not train_ann.exists():
        print("❌ train/_annotations.coco.json not found!")
        return False
    
    if not valid_ann.exists():
        print("❌ valid/_annotations.coco.json not found!")
        return False
    
    print("✅ Annotation files found")
    
    # Parse annotations
    with open(train_ann) as f:
        train_data = json.load(f)
    
    with open(valid_ann) as f:
        valid_data = json.load(f)
    
    # Show statistics
    print(f"\n📊 Training Set:")
    print(f"   Images: {len(train_data['images'])}")
    print(f"   Annotations: {len(train_data['annotations'])}")
    print(f"   Categories: {len(train_data['categories'])}")
    
    print(f"\n📊 Validation Set:")
    print(f"   Images: {len(valid_data['images'])}")
    print(f"   Annotations: {len(valid_data['annotations'])}")
    print(f"   Categories: {len(valid_data['categories'])}")
    
    # Show category names
    print(f"\n🏷️ Categories:")
    for cat in train_data['categories']:
        print(f"   - {cat['name']} (id: {cat['id']})")
    
    # Check for segmentation data
    sample_ann = train_data['annotations'][0]
    if 'segmentation' in sample_ann:
        print(f"\n✅ Segmentation data present (good for Mask R-CNN)")
    else:
        print(f"\n⚠️ Warning: No segmentation data found!")
    
    print("\n✅ Dataset is valid and ready for training!")
    return True

if __name__ == "__main__":
    verify_dataset()
```

Run verification:

```powershell
python verify_roboflow_dataset.py
```

---

## 🚀 Next Steps

Once your dataset is downloaded and verified:

### 1. Prepare for Training

Create `train_custom_model.py`:

```python
"""
Train custom mangrove detection model with Roboflow data
"""
from detectree2.models.train import setup_cfg, MyTrainer
from detectron2.data.datasets import register_coco_instances
from detectron2.engine import DefaultTrainer
from pathlib import Path
import os

def train_custom_model():
    """Train detectree2 model on custom Roboflow dataset"""
    
    # Register your dataset
    dataset_dir = Path("training_data/roboflow_dataset")
    
    # Register training set
    register_coco_instances(
        "mangrove_train",
        {},
        str(dataset_dir / "train" / "_annotations.coco.json"),
        str(dataset_dir / "train")
    )
    
    # Register validation set
    register_coco_instances(
        "mangrove_val",
        {},
        str(dataset_dir / "valid" / "_annotations.coco.json"),
        str(dataset_dir / "valid")
    )
    
    # Configure training
    from detectron2.config import get_cfg
    cfg = get_cfg()
    cfg.merge_from_file("detectree2/model_garden/configs/canopy_config.yaml")
    
    # Custom settings
    cfg.DATASETS.TRAIN = ("mangrove_train",)
    cfg.DATASETS.TEST = ("mangrove_val",)
    cfg.DATALOADER.NUM_WORKERS = 4
    cfg.MODEL.ROI_HEADS.BATCH_SIZE_PER_IMAGE = 256
    cfg.MODEL.ROI_HEADS.NUM_CLASSES = 1  # Only "tree" class
    cfg.SOLVER.IMS_PER_BATCH = 2  # Batch size (adjust for your GPU)
    cfg.SOLVER.BASE_LR = 0.001
    cfg.SOLVER.MAX_ITER = 3000  # Training iterations
    cfg.MODEL.DEVICE = "cuda"  # Use GPU
    
    # Output directory
    cfg.OUTPUT_DIR = "./models/custom_mangrove_model"
    os.makedirs(cfg.OUTPUT_DIR, exist_ok=True)
    
    # Start training
    print("🚀 Starting training...")
    trainer = DefaultTrainer(cfg)
    trainer.resume_or_load(resume=False)
    trainer.train()
    
    print(f"✅ Training complete! Model saved to: {cfg.OUTPUT_DIR}")

if __name__ == "__main__":
    train_custom_model()
```

### 2. Start Training

```powershell
# Make sure you have GPU support
python train_custom_model.py
```

Training will take several hours depending on:
- Number of images (100-200)
- Number of iterations (3000-5000)
- GPU speed (RTX 3050 Ti: ~4-8 hours)

### 3. Use Your Trained Model

After training, update your detector to use the custom model:

```python
# In app.py or detector
detector = Detectree2Detector(
    model_path="models/custom_mangrove_model/model_final.pth",
    confidence_threshold=0.5
)
```

---

## 🔧 Troubleshooting

### "No module named 'roboflow'"
```powershell
pip install roboflow
```

### "Invalid API key"
- Check that you copied the entire key
- Make sure you're using the **private** API key, not public
- Regenerate key if needed in Roboflow settings

### "Project not found"
- Double-check workspace name (case-sensitive)
- Double-check project name (use dashes, not spaces)
- Ensure you have access to the project

### "Download fails partway"
- Check internet connection
- Try manual download method instead
- Download smaller batches if dataset is very large

---

## 📚 Additional Resources

- [Roboflow Documentation](https://docs.roboflow.com/)
- [COCO Format Specification](https://cocodataset.org/#format-data)
- [Detectree2 Training Guide](https://github.com/PatBall1/detectree2)

---

## ✅ Summary

**To download your Roboflow dataset:**

```powershell
# 1. Install roboflow
pip install roboflow

# 2. Create and edit download_roboflow_dataset.py
# (Add your API key and project details)

# 3. Run download
python download_roboflow_dataset.py

# 4. Verify
python verify_roboflow_dataset.py

# 5. Train
python train_custom_model.py
```

**You're ready to train with your custom annotations! 🚀**
