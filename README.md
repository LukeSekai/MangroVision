# Mangrove Canopy Detection System
## GIS-Based Safe Planting Zone Analyzer for Leganes

This system uses **detectree2** (Mask R-CNN) to detect mangrove tree canopies from 90-degree drone imagery and identifies safe planting zones for your thesis defense.

---

## 🎯 Project Overview

**Goal**: Detect mangrove canopies as "danger zones," create 1-meter buffer zones, and identify safe planting areas.

**Current Status**: 
- ✅ detectron2 and detectree2 installed
- ✅ Project structure created
- ⚠️ Ready for model training on your mangrove dataset

---

## 📁 Project Structure

```
MangroVision/
├── canopy_detection/
│   ├── config.py              # Configuration settings
│   ├── canopy_detector.py     # Main detection script
│   └── test_installation.py   # Installation verification
├── drone_images/              # Place your 90° drone shots here
├── output/                    # Processed results & visualizations
├── models/                    # Trained model weights
├── dataset_frames/            # Video frame extraction
└── flight_videos/            # Original drone videos
```

---

## 🚀 Quick Start

### 1. Verify Installation

```powershell
cd MangroVision\canopy_detection
C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\Scripts\python.exe test_installation.py
```

This will verify that detectron2 and detectree2 are working correctly.

### 2. Add Your Drone Images

Place your 90-degree drone shot images in the `drone_images/` folder:
- Supported formats: JPG, PNG, TIFF
- Recommended resolution: High resolution (e.g., 4K or 6000x4000px)
- Ensure images show clear tree canopy coverage

### 3. Run Canopy Detection

```powershell
C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\Scripts\python.exe canopy_detector.py
```

This will:
- Detect tree canopies in your images
- Create 1-meter buffer zones (danger zones)
- Identify safe planting zones
- Save visualizations and results to `output/`

---

## 🎓 For Your 20% Defense

### What the System Does:

1. **Canopy Detection**: Uses Mask R-CNN to segment tree canopies
2. **Danger Zone Mapping**: Marks detected canopies as no-planting areas
3. **Buffer Creation**: Adds 1-meter safety buffer around each canopy
4. **Safe Zone Identification**: Calculates remaining area suitable for planting

### Current Capabilities:

- ✅ Detects tree canopy shapes and boundaries
- ✅ Creates buffer zones around detections
- ✅ Calculates safe vs. danger zone percentages
- ✅ Generates visual overlay maps

### For Later (Post-Defense):

- Train model specifically on mangrove species
- Add species classification
- Integrate with GIS mapping for Leganes
- Add GPS coordinates and real-world measurements

---

## ⚙️ Configuration

Edit `config.py` to adjust:

```python
BUFFER_DISTANCE_METERS = 1.0      # Buffer size around canopies
CONFIDENCE_THRESHOLD = 0.5        # Detection confidence (0-1)
DEVICE = "cpu"                    # or "cuda" for GPU
DEFAULT_CRS = "EPSG:32651"        # Philippines UTM Zone 51N
```

---

## 🔧 Training Your Own Model

To train detectree2 specifically for mangrove detection:

### Step 1: Prepare Training Data

1. Collect 50-100 drone images of mangrove canopies
2. Annotate using tools like:
   - CVAT (Computer Vision Annotation Tool)
   - LabelMe
   - VGG Image Annotator (VIA)

3. Export annotations in COCO format

### Step 2: Register Dataset

```python
from detectree2.models.train import register_train_data

register_train_data(
    train_folder="path/to/training/images",
    val_folder="path/to/validation/images",
    name="mangrove_canopy"
)
```

### Step 3: Train Model

```python
from detectree2.models.train import setup_cfg, MyTrainer

cfg = setup_cfg(
    dataset_name="mangrove_canopy",
    num_classes=1,  # Just "tree" class for now
    max_iter=5000,
    eval_period=500
)

trainer = MyTrainer(cfg)
trainer.train()
```

---

## 📊 Output Files

After processing, you'll find in `output/`:

- `*_zones.png` - Visualization with danger/safe zones overlay
- `*_results.json` - Detection statistics and metadata
- `*_shapefile/` - GIS-compatible shapefiles (future)

---

## 🐛 Troubleshooting

### Issue: "No drone images found"
**Solution**: Add images to `drone_images/` folder

### Issue: Detection quality is poor
**Solution**: 
- Train on your specific mangrove dataset
- Adjust `CONFIDENCE_THRESHOLD` in config.py
- Ensure high-quality input images

### Issue: Out of memory
**Solution**: 
- Reduce image resolution
- Process fewer images at once
- Use CPU instead of GPU (set `DEVICE = "cpu"`)

---

## 📚 Key Technologies

- **detectron2**: Facebook's state-of-the-art detection framework
- **detectree2**: Specialized for tree crown detection
- **Mask R-CNN**: Instance segmentation model
- **GeoPandas**: Spatial data operations
- **Shapely**: Geometric operations for buffer zones

---

## 🎯 Next Steps for Thesis

1. ✅ **Phase 1 (20% Defense)**: Demonstrate canopy detection capability
2. **Phase 2**: Train on mangrove-specific dataset
3. **Phase 3**: Integrate with GIS for Leganes area mapping
4. **Phase 4**: Add species classification (if needed)
5. **Phase 5**: Deploy for field use with GPS integration

---

## 📝 Citations

If using for your thesis, cite:

```
Ball, J.G.C., Hickman, S.H.M., Jackson, T.D., Koay, X.J., Hirst, J., Jay, W., 
Archer, M., Aubry-Kientz, M., Vincent, G. and Coomes, D.A. (2023). 
Detectree2: A Python package for tree crown detection in RGB imagery using Mask R-CNN. 
Methods in Ecology and Evolution.
```

---

## 💡 Tips for Demo

1. Use clear, high-resolution images
2. Show before/after visualizations
3. Highlight the buffer zone concept
4. Explain safe vs. danger zone percentages
5. Mention future GIS integration plans

---

## 🤝 Support

For questions about:
- **detectree2**: https://github.com/PatBall1/detectree2
- **detectron2**: https://github.com/facebookresearch/detectron2
- **This project**: [Your contact info]

---

## 📄 License

[Your thesis license/academic use notice]

---

**Good luck with your 20% defense! 🌳🚁**
