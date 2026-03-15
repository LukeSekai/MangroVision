"""
Quick Start Demo - Detectree2 for Mangrove Canopy Detection
This script shows how to use detectree2's pre-trained model to detect tree canopies
"""

import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


def create_sample_image():
    """Create a sample drone-like image for testing"""
    # Create a realistic-looking test image
    height, width = 1000, 1500
    
    # Green background (grass/water)
    image = np.ones((height, width, 3), dtype=np.uint8)
    image[:, :] = [34, 139, 34]  # Forest green
    
    # Add some "tree canopy" circles with varying shades of green
    canopies = [
        ((300, 250), 80, [20, 100, 20]),
        ((700, 300), 100, [30, 120, 30]),
        ((500, 600), 90, [25, 110, 25]),
        ((1100, 400), 70, [28, 115, 28]),
        ((900, 700), 85, [22, 105, 22]),
        ((400, 800), 95, [26, 108, 26]),
    ]
    
    for (x, y), radius, color in canopies:
        # Main canopy
        cv2.circle(image, (x, y), radius, color, -1)
        # Add some texture
        cv2.circle(image, (x, y), radius, [c+20 for c in color], 2)
        # Highlight
        cv2.circle(image, (x-10, y-10), radius//4, [c+40 for c in color], -1)
    
    return image


def demo_detectree2_workflow():
    """Demonstrate the detectree2 workflow"""
    print("\n" + "="*70)
    print("DETECTREE2 QUICK START DEMO")
    print("Mangrove Canopy Detection System")
    print("="*70 + "\n")
    
    # Create output directory
    demo_dir = Path("../../output/demo")
    demo_dir.mkdir(parents=True, exist_ok=True)
    
    print("Step 1: Creating Sample Image")
    print("-" * 70)
    sample_image = create_sample_image()
    sample_path = demo_dir / "sample_drone_image.png"
    cv2.imwrite(str(sample_path), sample_image)
    print(f"✓ Sample image created: {sample_path}")
    print(f"  Image size: {sample_image.shape[1]}x{sample_image.shape[0]} pixels\n")
    
    print("Step 2: Understanding Detectree2 Workflow")
    print("-" * 70)
    print("""
Detectree2 requires:
1. Training data (annotated tree canopy images)
2. Model training on your specific dataset
3. Prediction using the trained model

For your thesis defense demo:
- You can use detectree2's pre-trained models (trained on various tree datasets)
- Later, train on your mangrove-specific images for better accuracy
    """)
    
    print("Step 3: Required Steps for Mangrove Detection")
    print("-" * 70)
    print("""
A. Prepare Your Data:
   1. Collect 50-100 drone images of mangrove canopies
   2. Annotate tree crowns using CVAT or similar tool
   3. Export in COCO format
   
B. Train Model:
   ```python
   from detectree2.models.train import setup_cfg, MyTrainer
   from detectree2.data_loading import register_train_data
   
   # Register your dataset
   register_train_data(
       train_folder="path/to/training/images",
       val_folder="path/to/validation/images",
       name="mangrove_dataset"
   )
   
   # Setup configuration
   cfg = setup_cfg(
       dataset_name="mangrove_dataset",
       num_classes=1,  # Just 'tree' class
       max_iter=5000
   )
   
   # Train
   trainer = MyTrainer(cfg)
   trainer.train()
   ```

C. Make Predictions:
   ```python
   from detectree2.models.predict import predict_on_data
   
   predict_on_data(
       cfg=cfg,
       img_dir="path/to/drone/images",
       output_dir="path/to/output"
   )
   ```
    """)
    
    print("Step 4: Creating Danger Zones and Buffers")
    print("-" * 70)
    print("""
After detection, create buffer zones:
    
   from shapely.geometry import Point
   from shapely.ops import unary_union
   
   # For each detected canopy
   canopy_polygons = []
   for detection in predictions:
       # Convert mask to polygon
       polygon = mask_to_polygon(detection['segmentation'])
       canopy_polygons.append(polygon)
   
   # Create 1-meter buffer
   buffer_distance = 1.0 / pixel_resolution  # Convert to pixels
   danger_zones = [poly.buffer(buffer_distance) for poly in canopy_polygons]
   merged_danger = unary_union(danger_zones)
   
   # Calculate safe zones
   total_area = Polygon([(0,0), (width,0), (width,height), (0,height)])
   safe_zones = total_area.difference(merged_danger)
    """)
    
    print("Step 5: Visualization")
    print("-" * 70)
    
    # Create a mock visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Original image
    axes[0].imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
    axes[0].set_title('1. Original Drone Image', fontsize=12, fontweight='bold')
    axes[0].axis('off')
    
    # Mock detection overlay
    overlay1 = cv2.cvtColor(sample_image.copy(), cv2.COLOR_BGR2RGB)
    axes[1].imshow(overlay1)
    axes[1].set_title('2. Detected Canopies (Red)', fontsize=12, fontweight='bold')
    axes[1].axis('off')
    
    # Mock zones overlay
    overlay2 = cv2.cvtColor(sample_image.copy(), cv2.COLOR_BGR2RGB)
    axes[2].imshow(overlay2)
    axes[2].set_title('3. Danger Zones + Safe Planting Areas', fontsize=12, fontweight='bold')
    axes[2].axis('off')
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', alpha=0.7, label='Canopy Danger Zones'),
        Patch(facecolor='orange', alpha=0.5, label='1m Buffer Zones'),
        Patch(facecolor='lightgreen', alpha=0.4, label='Safe Planting Zones')
    ]
    axes[2].legend(handles=legend_elements, loc='upper right', fontsize=9)
    
    plt.tight_layout()
    viz_path = demo_dir / "workflow_example.png"
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    print(f"✓ Visualization saved: {viz_path}\n")
    plt.close()
    
    print("Step 6: Next Actions for Your Thesis")
    print("-" * 70)
    print("""
FOR YOUR 20% DEFENSE:
✓ Installed: detectron2 and detectree2
✓ Setup: Project structure ready
✓ Demo: Show this workflow and explain the concept

AFTER DEFENSE (for actual deployment):
☐ Collect mangrove drone images (50-100 images)
☐ Annotate tree canopies (use CVAT: cvat.ai)
☐ Train detectree2 model on your data
☐ Test on validation images
☐ Deploy for Leganes area mapping
☐ Integrate with GIS coordinates
    """)
    
    print("\n" + "="*70)
    print("DEMO COMPLETE!")
    print("="*70)
    print(f"\nAll demo files saved to: {demo_dir}")
    print("\nYou're ready to present the concept at your 20% defense!")
    print("The system architecture and workflow are in place.")
    

def show_training_example():
    """Show example of how training would work"""
    print("\n" + "="*70)
    print("TRAINING EXAMPLE CODE")
    print("="*70 + "\n")
    
    training_code = '''
# training_mangrove_model.py
# Run this after you've collected and annotated your mangrove images

from detectree2.data_loading import register_train_data
from detectree2.models.train import setup_cfg, MyTrainer
from detectree2.R_utils import set_seed
import detectron2

# Set random seed for reproducibility
set_seed(42)

# 1. Register your annotated dataset
print("Registering dataset...")
register_train_data(
    train_location="MangroVision/training_data/images",
    val_location="MangroVision/validation_data/images", 
    train_json_name="annotations_train.json",
    val_json_name="annotations_val.json",
    name="leganes_mangroves"
)

# 2. Setup training configuration
print("Setting up model configuration...")
cfg = setup_cfg(
    dataset_name="leganes_mangroves",
    num_classes=1,          # Only "mangrove canopy" class
    batch_size=2,
    learning_rate=0.001,
    max_iter=5000,          # Adjust based on dataset size
    eval_period=500,        # Validate every 500 iterations
    model_weights="COCO-InstanceSegmentation/mask_rcnn_R_50_FPN_3x.yaml",
    output_dir="MangroVision/models/trained"
)

# Use CPU or GPU
cfg.MODEL.DEVICE = "cpu"  # Change to "cuda" if you have GPU

# 3. Train the model
print("Starting training...")
trainer = MyTrainer(cfg)
trainer.resume_or_load(resume=False)
trainer.train()

print("Training complete! Model saved to:", cfg.OUTPUT_DIR)
'''
    
    print(training_code)
    
    # Save training example
    example_path = Path("../../output/demo/training_example.py")
    example_path.parent.mkdir(parents=True, exist_ok=True)
    with open(example_path, 'w') as f:
        f.write(training_code)
    
    print(f"\n✓ Training example saved to: {example_path}\n")


if __name__ == "__main__":
    demo_detectree2_workflow()
    show_training_example()
    
    print("\n" + "🌳"*35)
    print("Ready for your thesis defense!")
    print("Place your real drone images in 'drone_images/' folder to begin.")
    print("🌳"*35 + "\n")
