"""
Comprehensive Dataset Analysis for Better Training
Analyzes annotation quality, class balance, and identifies potential issues
"""

import json
from pathlib import Path
import numpy as np
from collections import Counter
import cv2

def analyze_dataset():
    """Comprehensive analysis of the Roboflow dataset"""
    
    print("=" * 80)
    print("📊 DATASET QUALITY ANALYSIS FOR TRAINING")
    print("=" * 80)
    print()
    
    dataset_dir = Path("training_data/roboflow_dataset")
    
    if not dataset_dir.exists():
        print("❌ Dataset not found. Run: python download_roboflow_dataset.py")
        return
    
    # Analyze both splits
    for split in ["train", "valid"]:
        print(f"\n{'='*80}")
        print(f"📁 Analyzing {split.upper()} split")
        print(f"{'='*80}\n")
        
        ann_path = dataset_dir / split / "_annotations.coco.json"
        img_dir = dataset_dir / split
        
        if not ann_path.exists():
            print(f"❌ {split} annotations not found")
            continue
        
        with open(ann_path) as f:
            data = json.load(f)
        
        analyze_split(data, img_dir, split)

def analyze_split(data, img_dir, split_name):
    """Analyze a single dataset split"""
    
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    images = {img['id']: img for img in data['images']}
    annotations = data['annotations']
    
    print(f"📝 Basic Statistics:")
    print(f"   Images: {len(images)}")
    print(f"   Annotations: {len(annotations)}")
    print(f"   Classes: {len(categories)}")
    print()
    
    # Class distribution
    print(f"📊 Class Distribution:")
    class_counts = Counter(ann['category_id'] for ann in annotations)
    total_anns = len(annotations)
    
    for cat_id, count in sorted(class_counts.items()):
        percentage = (count / total_anns) * 100
        class_name = categories[cat_id]
        print(f"   {class_name:25s} {count:5d} ({percentage:5.1f}%)")
    
    # Check for class imbalance
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        print()
        if imbalance_ratio > 3:
            print(f"⚠️  CLASS IMBALANCE DETECTED!")
            print(f"   Ratio: {imbalance_ratio:.1f}:1")
            print(f"   Recommendation: Consider data augmentation for minority class")
        else:
            print(f"✅ Classes are reasonably balanced (ratio: {imbalance_ratio:.1f}:1)")
    
    # Annotation size analysis
    print()
    print(f"📐 Annotation Size Analysis:")
    
    areas = []
    bbox_areas = []
    aspect_ratios = []
    polygon_vertices = []
    
    for ann in annotations:
        if 'segmentation' in ann and ann['segmentation']:
            # Get polygon vertices
            seg = ann['segmentation'][0] if isinstance(ann['segmentation'], list) else ann['segmentation']
            if isinstance(seg, list):
                polygon_vertices.append(len(seg) // 2)
        
        if 'area' in ann:
            areas.append(ann['area'])
        
        if 'bbox' in ann:
            x, y, w, h = ann['bbox']
            bbox_areas.append(w * h)
            if h > 0:
                aspect_ratios.append(w / h)
    
    if areas:
        print(f"   Segmentation Area:")
        print(f"      Mean:   {np.mean(areas):,.0f} px²")
        print(f"      Median: {np.median(areas):,.0f} px²")
        print(f"      Min:    {np.min(areas):,.0f} px²")
        print(f"      Max:    {np.max(areas):,.0f} px²")
        print(f"      Std:    {np.std(areas):,.0f} px²")
    
    if bbox_areas:
        print(f"   Bounding Box Area:")
        print(f"      Mean:   {np.mean(bbox_areas):,.0f} px²")
        print(f"      Median: {np.median(bbox_areas):,.0f} px²")
    
    if aspect_ratios:
        print(f"   Aspect Ratios (W/H):")
        print(f"      Mean:   {np.mean(aspect_ratios):.2f}")
        print(f"      Median: {np.median(aspect_ratios):.2f}")
        
        # Check for extreme aspect ratios
        extreme_ratios = [r for r in aspect_ratios if r < 0.3 or r > 3.0]
        if extreme_ratios:
            print(f"   ⚠️  {len(extreme_ratios)} annotations with extreme aspect ratios")
    
    if polygon_vertices:
        print(f"   Polygon Complexity:")
        print(f"      Mean vertices: {np.mean(polygon_vertices):.1f}")
        print(f"      Median vertices: {np.median(polygon_vertices):.0f}")
    
    # Check for small annotations
    if areas:
        small_annotations = [a for a in areas if a < 100]
        if small_annotations:
            print()
            print(f"⚠️  {len(small_annotations)} very small annotations (< 100 px²)")
            print(f"   These may be hard to detect. Consider removing or verifying.")
    
    # Check annotations per image
    print()
    print(f"📸 Annotations per Image:")
    
    img_ann_counts = Counter(ann['image_id'] for ann in annotations)
    anns_per_img = list(img_ann_counts.values())
    
    if anns_per_img:
        print(f"   Mean:   {np.mean(anns_per_img):.1f}")
        print(f"   Median: {np.median(anns_per_img):.0f}")
        print(f"   Min:    {np.min(anns_per_img)}")
        print(f"   Max:    {np.max(anns_per_img)}")
        
        # Images with no annotations
        images_with_no_anns = len(images) - len(img_ann_counts)
        if images_with_no_anns > 0:
            print(f"   ⚠️  {images_with_no_anns} images with NO annotations")
    
    # Image size analysis
    print()
    print(f"🖼️  Image Dimensions:")
    
    image_widths = [img['width'] for img in images.values()]
    image_heights = [img['height'] for img in images.values()]
    
    if image_widths:
        print(f"   Width:  {np.min(image_widths)} - {np.max(image_widths)} px (mean: {np.mean(image_widths):.0f})")
        print(f"   Height: {np.min(image_heights)} - {np.max(image_heights)} px (mean: {np.mean(image_heights):.0f})")
        
        # Check for inconsistent sizes
        if len(set(image_widths)) > 3 or len(set(image_heights)) > 3:
            print(f"   ⚠️  Images have varying dimensions (augmentation will resize)")
    
    print()


def generate_recommendations():
    """Generate training recommendations based on analysis"""
    
    print("\n" + "=" * 80)
    print("💡 TRAINING RECOMMENDATIONS")
    print("=" * 80)
    print()
    
    dataset_dir = Path("training_data/roboflow_dataset")
    ann_path = dataset_dir / "train" / "_annotations.coco.json"
    
    if not ann_path.exists():
        return
    
    with open(ann_path) as f:
        data = json.load(f)
    
    num_images = len(data['images'])
    num_annotations = len(data['annotations'])
    
    print(f"📊 Dataset Size: {num_images} training images")
    print()
    
    # Recommendations based on dataset size
    if num_images < 100:
        print("⚠️  VERY SMALL DATASET (< 100 images)")
        print("   Recommendations:")
        print("   • Enable STRONG data augmentation")
        print("   • Use higher learning rate decay")
        print("   • Train for more iterations (5000+)")
        print("   • Consider collecting more data")
    elif num_images < 500:
        print("✅ SMALL DATASET (100-500 images)")
        print("   Recommendations:")
        print("   • Enable moderate data augmentation")
        print("   • Use transfer learning (detectree2 base) ✓")
        print("   • Train for 3000-5000 iterations")
        print("   • Monitor for overfitting")
    else:
        print("✅ GOOD DATASET SIZE (500+ images)")
        print("   Recommendations:")
        print("   • Standard augmentation sufficient")
        print("   • Can train for longer (5000-10000 iterations)")
        print("   • Less risk of overfitting")
    
    print()
    print("🎯 Current Training Configuration:")
    print("   • Base Model: Detectree2 230103_randresize_full.pth")
    print("   • Architecture: Mask R-CNN R-101 FPN")
    print("   • Iterations: 3000")
    print("   • Learning Rate: 0.00025")
    print("   • Batch Size: 2")
    print()
    
    # Augmentation recommendations
    print("🔄 Data Augmentation Settings:")
    print("   Detectron2 includes built-in augmentation:")
    print("   ✓ Random flip (horizontal)")
    print("   ✓ Brightness adjustment")
    print("   ✓ Contrast adjustment")
    print("   ✓ Saturation adjustment")
    print("   ✓ Random crop/resize")
    print()
    print("   Additional augmentations in train_custom_model.py:")
    print("   • Rotation: Enabled")
    print("   • Scale jitter: 0.8-1.2x")
    print("   • Color jitter: Brightness, contrast, saturation")
    print()
    
    # Class-specific recommendations
    class_counts = Counter(ann['category_id'] for ann in data['annotations'])
    if class_counts:
        max_count = max(class_counts.values())
        min_count = min(class_counts.values())
        imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
        
        if imbalance_ratio > 3:
            print("⚠️  Class Imbalance Handling:")
            print("   • Consider focal loss (helps with imbalanced classes)")
            print("   • Or: Oversample minority class during training")
            print("   • Or: Apply stronger augmentation to minority class")
            print()


if __name__ == "__main__":
    analyze_dataset()
    generate_recommendations()
    
    print()
    print("=" * 80)
    print("✅ Analysis Complete!")
    print("=" * 80)
    print()
    print("Next steps:")
    print("1. Review the analysis above")
    print("2. Address any warnings (class imbalance, small annotations)")
    print("3. When ready: python train_custom_model.py")
    print()
