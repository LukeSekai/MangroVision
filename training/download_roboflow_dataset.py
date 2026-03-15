"""
Download annotated dataset from Roboflow for MangroVision training
"""
from roboflow import Roboflow
from pathlib import Path
import json

# ========================================
# CONFIGURATION - EDIT THESE VALUES
# ========================================

API_KEY = "LJti0618t62VdAV816QP"  # Get from: roboflow.com -> Account -> Roboflow API
WORKSPACE_NAME = "finding"  # Your Roboflow workspace name (lowercase)
PROJECT_NAME = "practice_annotate-ygsoo"  # Your project name in Roboflow
VERSION = 1  # Dataset version number



# Output directory
OUTPUT_DIR = Path("training_data/roboflow_dataset")

# ========================================
# DOWNLOAD FUNCTION
# ========================================

def download_dataset():
    """Download dataset from Roboflow in COCO format"""
    
    print("=" * 70)
    print("🌿 MangroVision - Roboflow Dataset Downloader")
    print("=" * 70)
    
    # Validate configuration
    if API_KEY == "YOUR_API_KEY_HERE":
        print("\n❌ ERROR: Please edit this script and add your Roboflow API key!")
        print("\n📝 Steps:")
        print("   1. Go to https://roboflow.com/")
        print("   2. Click your profile icon → Account → Roboflow API")
        print("   3. Copy your Private API Key")
        print("   4. Edit this file and replace 'YOUR_API_KEY_HERE' with your key")
        print("   5. Also update WORKSPACE_NAME and PROJECT_NAME")
        return False
    
    if WORKSPACE_NAME == "your-workspace" or PROJECT_NAME == "mangrove-detection":
        print("\n⚠️ WARNING: You may need to update WORKSPACE_NAME and PROJECT_NAME")
        print(f"   Current workspace: {WORKSPACE_NAME}")
        print(f"   Current project: {PROJECT_NAME}")
        print("\n   Check your Roboflow URL: roboflow.com/WORKSPACE/PROJECT")
        response = input("\n   Continue anyway? (y/n): ")
        if response.lower() != 'y':
            return False
    
    try:
        # Initialize Roboflow
        print(f"\n📡 Connecting to Roboflow...")
        rf = Roboflow(api_key=API_KEY)
        
        # Get your project
        print(f"📂 Loading project: {WORKSPACE_NAME}/{PROJECT_NAME}")
        project = rf.workspace(WORKSPACE_NAME).project(PROJECT_NAME)
        
        # Get specific version
        print(f"📊 Loading version {VERSION}...")
        dataset = project.version(VERSION)
        
        # Create output directory
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
        
        # Download in COCO format (best for detectree2/Mask R-CNN)
        print(f"\n⬇️ Downloading dataset in COCO format...")
        print(f"📁 Saving to: {OUTPUT_DIR.absolute()}")
        print("\n⏳ This may take a few minutes depending on dataset size...")
        
        # Download
        dataset.download(
            model_format="coco",  # COCO format for instance segmentation
            location=str(OUTPUT_DIR)
        )
        
        print("\n✅ Download complete!")
        
        # Show structure
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
        show_statistics()
        
        print("\n" + "=" * 70)
        print("✅ Dataset ready for training!")
        print("=" * 70)
        print("\n📖 Next steps:")
        print("   1. Run: python verify_roboflow_dataset.py")
        print("   2. Then: python train_custom_model.py")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error downloading dataset: {e}")
        print("\n🔧 Troubleshooting:")
        print("   - Check your API key is correct")
        print("   - Check your workspace name (case-sensitive)")
        print("   - Check your project name (use dashes, not spaces)")
        print("   - Ensure you have internet connection")
        print("   - Try manual download from Roboflow web interface")
        return False


def show_statistics():
    """Display dataset statistics"""
    
    train_dir = OUTPUT_DIR / "train"
    valid_dir = OUTPUT_DIR / "valid"
    test_dir = OUTPUT_DIR / "test"
    
    print("\n📊 Dataset Statistics:")
    print("   " + "-" * 50)
    
    # Training set
    if train_dir.exists():
        train_images = list(train_dir.glob("*.jpg")) + list(train_dir.glob("*.png"))
        train_ann = train_dir / "_annotations.coco.json"
        
        if train_ann.exists():
            with open(train_ann) as f:
                data = json.load(f)
            print(f"   📸 Training images: {len(train_images)}")
            print(f"   🏷️  Training annotations: {len(data['annotations'])}")
    
    # Validation set
    if valid_dir.exists():
        valid_images = list(valid_dir.glob("*.jpg")) + list(valid_dir.glob("*.png"))
        valid_ann = valid_dir / "_annotations.coco.json"
        
        if valid_ann.exists():
            with open(valid_ann) as f:
                data = json.load(f)
            print(f"   📸 Validation images: {len(valid_images)}")
            print(f"   🏷️  Validation annotations: {len(data['annotations'])}")
    
    # Test set (optional)
    if test_dir.exists():
        test_images = list(test_dir.glob("*.jpg")) + list(test_dir.glob("*.png"))
        test_ann = test_dir / "_annotations.coco.json"
        
        if test_ann.exists():
            with open(test_ann) as f:
                data = json.load(f)
            print(f"   📸 Test images: {len(test_images)}")
            print(f"   🏷️  Test annotations: {len(data['annotations'])}")


# ========================================
# MAIN
# ========================================

if __name__ == "__main__":
    print("\n")
    success = download_dataset()
    print("\n")
    
    if not success:
        print("⚠️ Download did not complete. Please check the errors above.")
        print("📖 For help, see: ROBOFLOW_DOWNLOAD_GUIDE.md")
