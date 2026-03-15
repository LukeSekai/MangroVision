"""
Visualize Dataset Annotations
Shows random samples from your dataset to verify annotation quality
"""

import json
from pathlib import Path
import cv2
import numpy as np
import random

def visualize_annotations(num_samples=10, split="train"):
    """
    Visualize random samples from the dataset
    
    Args:
        num_samples: Number of images to visualize
        split: 'train' or 'valid'
    """
    
    print("=" * 80)
    print(f"🖼️  DATASET ANNOTATION VISUALIZER")
    print("=" * 80)
    print()
    
    dataset_dir = Path("training_data/roboflow_dataset")
    ann_path = dataset_dir / split / "_annotations.coco.json"
    img_dir = dataset_dir / split
    
    if not ann_path.exists():
        print(f"❌ {split} annotations not found")
        return
    
    # Load annotations
    with open(ann_path) as f:
        data = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    images = {img['id']: img for img in data['images']}
    annotations = data['annotations']
    
    # Group annotations by image
    img_annotations = {}
    for ann in annotations:
        img_id = ann['image_id']
        if img_id not in img_annotations:
            img_annotations[img_id] = []
        img_annotations[img_id].append(ann)
    
    # Select random images that have annotations
    available_images = [img_id for img_id in img_annotations.keys()]
    
    if not available_images:
        print("❌ No annotated images found")
        return
    
    num_samples = min(num_samples, len(available_images))
    selected_images = random.sample(available_images, num_samples)
    
    print(f"📊 Dataset: {len(images)} images, {len(annotations)} annotations")
    print(f"🎲 Randomly selected {num_samples} images from {split} split")
    print()
    
    # Color map for classes
    colors = [
        (0, 255, 0),    # Green for Mangrove
        (255, 0, 0),    # Blue for Bungalon
        (0, 255, 255),  # Yellow
        (255, 0, 255),  # Magenta
    ]
    
    output_dir = Path("output") / "dataset_visualization"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    for idx, img_id in enumerate(selected_images, 1):
        img_info = images[img_id]
        img_path = img_dir / img_info['file_name']
        
        if not img_path.exists():
            print(f"⚠️  Image not found: {img_path}")
            continue
        
        # Load image
        image = cv2.imread(str(img_path))
        if image is None:
            print(f"⚠️  Could not load: {img_path}")
            continue
        
        # Create overlay
        overlay = image.copy()
        
        # Draw annotations
        img_anns = img_annotations[img_id]
        
        for ann in img_anns:
            cat_id = ann['category_id']
            cat_name = categories[cat_id]
            color = colors[cat_id % len(colors)]
            
            # Draw segmentation polygon
            if 'segmentation' in ann and ann['segmentation']:
                seg = ann['segmentation']
                if isinstance(seg, list) and seg:
                    polygon = seg[0] if isinstance(seg[0], list) else seg
                    
                    # Convert to numpy array
                    if len(polygon) >= 6:  # At least 3 points (x,y pairs)
                        pts = np.array(polygon).reshape(-1, 2).astype(np.int32)
                        
                        # Fill polygon with transparency
                        cv2.fillPoly(overlay, [pts], color)
                        
                        # Draw polygon outline
                        cv2.polylines(image, [pts], True, color, 2)
            
            # Draw bounding box
            if 'bbox' in ann:
                x, y, w, h = [int(v) for v in ann['bbox']]
                cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                
                # Add label
                label = f"{cat_name}"
                label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                cv2.rectangle(image, (x, y - label_size[1] - 5), 
                            (x + label_size[0], y), color, -1)
                cv2.putText(image, label, (x, y - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Blend overlay with original
        alpha = 0.3
        image = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
        
        # Add info text
        info_text = f"Image {idx}/{num_samples} | {img_info['file_name']} | {len(img_anns)} annotations"
        cv2.putText(image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(image, info_text, (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 1)
        
        # Save visualization
        output_path = output_dir / f"{split}_{idx:02d}_{img_info['file_name']}"
        cv2.imwrite(str(output_path), image)
        
        print(f"✅ {idx}/{num_samples}: {img_info['file_name']}")
        print(f"   Annotations: {len(img_anns)}")
        for ann in img_anns:
            cat_name = categories[ann['category_id']]
            area = ann.get('area', 0)
            print(f"      • {cat_name}: {area:.0f} px²")
    
    print()
    print(f"✅ Visualizations saved to: {output_dir.absolute()}")
    print()
    print("💡 Check these images to verify:")
    print("   • Annotations are accurate")
    print("   • Polygons match canopy boundaries")
    print("   • No missing or duplicate annotations")
    print("   • Classes are correctly labeled")
    print()


def visualize_class_examples(samples_per_class=5):
    """Show examples of each class"""
    
    print("=" * 80)
    print(f"🏷️  CLASS-SPECIFIC EXAMPLES")
    print("=" * 80)
    print()
    
    dataset_dir = Path("training_data/roboflow_dataset")
    ann_path = dataset_dir / "train" / "_annotations.coco.json"
    img_dir = dataset_dir / "train"
    
    if not ann_path.exists():
        print("❌ Training annotations not found")
        return
    
    with open(ann_path) as f:
        data = json.load(f)
    
    categories = {cat['id']: cat['name'] for cat in data['categories']}
    
    print(f"Found {len(categories)} classes:")
    for cat_id, cat_name in categories.items():
        print(f"   {cat_id}: {cat_name}")
    print()
    
    # Group annotations by category
    category_annotations = {cat_id: [] for cat_id in categories.keys()}
    for ann in data['annotations']:
        category_annotations[ann['category_id']].append(ann)
    
    # Show examples for each class
    for cat_id, cat_name in categories.items():
        anns = category_annotations[cat_id]
        print(f"\n{cat_name}: {len(anns)} annotations")
        
        if not anns:
            print("   ⚠️  No annotations for this class!")
            continue
        
        # Sample a few
        sample_anns = random.sample(anns, min(samples_per_class, len(anns)))
        
        for ann in sample_anns:
            area = ann.get('area', 0)
            bbox = ann.get('bbox', [0, 0, 0, 0])
            w, h = bbox[2], bbox[3]
            print(f"   • Area: {area:,.0f} px², Size: {w:.0f}×{h:.0f} px")


if __name__ == "__main__":
    print("🎨 MangroVision - Dataset Visualization Tool")
    print()
    
    # Visualize training set
    print("Visualizing TRAINING set...")
    visualize_annotations(num_samples=10, split="train")
    
    print()
    print("-" * 80)
    print()
    
    # Visualize validation set
    print("Visualizing VALIDATION set...")
    visualize_annotations(num_samples=5, split="valid")
    
    print()
    print("-" * 80)
    print()
    
    # Show class-specific examples
    visualize_class_examples(samples_per_class=5)
    
    print()
    print("=" * 80)
    print("✅ VISUALIZATION COMPLETE")
    print("=" * 80)
    print()
    print("Check: output/dataset_visualization/ for images")
    print()
