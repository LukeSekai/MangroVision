"""
Verify Roboflow dataset is properly downloaded and formatted
"""
import json
from pathlib import Path

def verify_dataset():
    """Verify that the dataset is properly formatted for training"""
    
    dataset_dir = Path("training_data/roboflow_dataset")
    
    print("=" * 70)
    print("🔍 MangroVision - Dataset Verification")
    print("=" * 70)
    
    # Check main directory
    if not dataset_dir.exists():
        print("\n❌ Dataset directory not found!")
        print(f"   Expected: {dataset_dir.absolute()}")
        print("\n📝 Please run: python download_roboflow_dataset.py")
        return False
    
    print(f"\n✅ Dataset directory found: {dataset_dir.absolute()}")
    
    # Check subdirectories
    train_dir = dataset_dir / "train"
    valid_dir = dataset_dir / "valid"
    test_dir = dataset_dir / "test"
    
    if not train_dir.exists():
        print("❌ train/ folder not found!")
        return False
    print("✅ train/ folder exists")
    
    if not valid_dir.exists():
        print("❌ valid/ folder not found!")
        return False
    print("✅ valid/ folder exists")
    
    if test_dir.exists():
        print("✅ test/ folder exists (optional)")
    
    # Check annotation files
    train_ann = train_dir / "_annotations.coco.json"
    valid_ann = valid_dir / "_annotations.coco.json"
    
    if not train_ann.exists():
        print("❌ train/_annotations.coco.json not found!")
        return False
    print("✅ Training annotations found")
    
    if not valid_ann.exists():
        print("❌ valid/_annotations.coco.json not found!")
        return False
    print("✅ Validation annotations found")
    
    # Parse and validate annotations
    print("\n" + "=" * 70)
    print("📊 Dataset Statistics")
    print("=" * 70)
    
    try:
        with open(train_ann) as f:
            train_data = json.load(f)
        
        with open(valid_ann) as f:
            valid_data = json.load(f)
        
        # Training set statistics
        print(f"\n📦 Training Set:")
        print(f"   Images: {len(train_data['images'])}")
        print(f"   Annotations: {len(train_data['annotations'])}")
        print(f"   Categories: {len(train_data['categories'])}")
        
        if len(train_data['images']) == 0:
            print("   ⚠️ WARNING: No training images found!")
        elif len(train_data['images']) < 20:
            print("   ⚠️ WARNING: Very few training images. Consider adding more.")
        
        if len(train_data['annotations']) == 0:
            print("   ❌ ERROR: No annotations found!")
            return False
        
        # Validation set statistics
        print(f"\n📦 Validation Set:")
        print(f"   Images: {len(valid_data['images'])}")
        print(f"   Annotations: {len(valid_data['annotations'])}")
        print(f"   Categories: {len(valid_data['categories'])}")
        
        if len(valid_data['images']) == 0:
            print("   ⚠️ WARNING: No validation images found!")
        
        # Show categories
        print(f"\n🏷️ Categories:")
        for cat in train_data['categories']:
            print(f"   - {cat['name']} (id: {cat['id']})")
        
        # Check for segmentation data (required for Mask R-CNN)
        print(f"\n🔍 Annotation Format Check:")
        sample_ann = train_data['annotations'][0]
        
        if 'segmentation' in sample_ann:
            print(f"   ✅ Segmentation data present (polygon masks)")
            print(f"   ✅ Compatible with Mask R-CNN / detectree2")
            
            # Check segmentation format
            if isinstance(sample_ann['segmentation'], list) and len(sample_ann['segmentation']) > 0:
                seg = sample_ann['segmentation'][0]
                if isinstance(seg, list) and len(seg) >= 6:  # At least 3 points (x,y pairs)
                    print(f"   ✅ Polygon format valid")
                else:
                    print(f"   ⚠️ WARNING: Segmentation may be in wrong format")
        else:
            print(f"   ❌ ERROR: No segmentation data found!")
            print(f"   ❌ This dataset is not suitable for Mask R-CNN training")
            print(f"   📝 You need polygon annotations, not bounding boxes")
            return False
        
        if 'bbox' in sample_ann:
            print(f"   ✅ Bounding boxes present")
        
        # Calculate average annotations per image
        avg_train = len(train_data['annotations']) / max(len(train_data['images']), 1)
        avg_valid = len(valid_data['annotations']) / max(len(valid_data['images']), 1)
        
        print(f"\n📈 Average Annotations per Image:")
        print(f"   Training: {avg_train:.1f} trees/mangroves per image")
        print(f"   Validation: {avg_valid:.1f} trees/mangroves per image")
        
        if avg_train < 5:
            print(f"   ⚠️ Low annotation density. Are all trees labeled?")
        
        # Check for common issues
        print(f"\n🔧 Quality Checks:")
        
        # Check for images without annotations
        annotated_image_ids = set(ann['image_id'] for ann in train_data['annotations'])
        all_image_ids = set(img['id'] for img in train_data['images'])
        unannotated = all_image_ids - annotated_image_ids
        
        if len(unannotated) > 0:
            print(f"   ⚠️ {len(unannotated)} images have no annotations")
        else:
            print(f"   ✅ All images have at least one annotation")
        
        # Check for tiny annotations (may be errors)
        tiny_anns = [ann for ann in train_data['annotations'] if ann.get('area', 0) < 100]
        if len(tiny_anns) > 0:
            print(f"   ⚠️ {len(tiny_anns)} very small annotations (< 100 pixels)")
        
        # Success!
        print("\n" + "=" * 70)
        print("✅ Dataset is valid and ready for training!")
        print("=" * 70)
        
        print("\n📖 Next Steps:")
        print("   1. Review the statistics above")
        print("   2. If everything looks good, run: python train_custom_model.py")
        print("   3. Training will take several hours depending on dataset size")
        
        return True
        
    except json.JSONDecodeError as e:
        print(f"\n❌ Error parsing JSON: {e}")
        print("   The annotation file may be corrupted")
        return False
    except Exception as e:
        print(f"\n❌ Error: {e}")
        return False


if __name__ == "__main__":
    print("\n")
    success = verify_dataset()
    print("\n")
    
    if not success:
        print("⚠️ Verification failed. Please fix the issues above.")
        print("📖 For help, see: ROBOFLOW_DOWNLOAD_GUIDE.md")
