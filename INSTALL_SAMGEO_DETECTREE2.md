# 🌳 Installing Samgeo + Detectree2 for MangroVision

**Complete installation guide for accurate tree crown detection using the segment-geospatial (samgeo) wrapper for detectree2**

---

## 📋 Overview

This guide will help you install:
- **PyTorch & TorchVision** - Deep learning framework
- **Detectron2** - Meta's object detection library
- **Detectree2** - Tree crown detection model
- **Segment-Geospatial (samgeo)** - Geospatial wrapper with easy-to-use API
- **Pre-trained tropical model** - For mangrove/tropical tree detection

---

## 🔧 Step 1: Install PyTorch & TorchVision

### For CPU (Recommended for most users)
```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

### For NVIDIA GPU (If you have CUDA-capable GPU)
```powershell
# For CUDA 11.8
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118

# For CUDA 12.1
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

**Verify PyTorch installation:**
```powershell
python -c "import torch; print(f'PyTorch {torch.__version__} installed')"
```

---

## 🔧 Step 2: Install Detectron2

Detectron2 requires compilation from source. Use Meta's official GitHub repository.

### Windows Installation
```powershell
# Install Visual Studio Build Tools first (if not already installed)
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Install detectron2 from GitHub
pip install git+https://github.com/facebookresearch/detectron2.git
```

### Alternative (Pre-built wheels)
If the GitHub installation fails, try pre-built wheels:
```powershell
pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
```

**Verify Detectron2 installation:**
```powershell
python -c "import detectron2; print(f'Detectron2 {detectron2.__version__} installed')"
```

---

## 🔧 Step 3: Install Segment-Geospatial (samgeo)

The `segment-geospatial` package includes a detectree2 wrapper with a simplified API.

```powershell
pip install segment-geospatial
```

This installs:
- `samgeo` - Main package
- `samgeo.detectree2` - Tree crown delineation module
- GDAL bindings for geospatial operations
- Other dependencies (geopandas, rasterio, etc.)

**Verify samgeo installation:**
```powershell
python -c "from samgeo import SamGeo; from samgeo.detectree2 import TreeCrownDelineator; print('Samgeo installed successfully')"
```

---

## 🔧 Step 4: Install Additional Dependencies

```powershell
pip install geopandas rasterio shapely pyproj
```

---

## 🌳 Step 5: Download Pre-trained Tropical Model

Download the detectree2 model trained on tropical forests (including mangroves).

### Option A: Direct Download from Zenodo
```powershell
# Download the tropical model (230103_randresize_full.pth)
# Link: https://zenodo.org/record/7123579

# Save to models/ folder
# OR use wget/curl
```

### Option B: Using Python (gdown)
```powershell
python canopy_detection/download_models.py
```

### Manual Download
1. Go to: https://zenodo.org/record/7123579
2. Download: `230103_randresize_full.pth` (or latest version)
3. Save to: `MangroVision/models/230103_randresize_full.pth`

**Available Models:**
- `230103_randresize_full.pth` - Full tropical model (recommended)
- `230717_tropical_base.pth` - Base tropical model (currently in use)
- Custom models can be trained on your own data

---

## 🧪 Step 6: Verify Complete Installation

Run the test script:

```powershell
python canopy_detection/test_samgeo_detectree2.py
```

Expected output:
```
✅ PyTorch installed: 2.0.0
✅ Detectron2 installed: 0.6
✅ Samgeo installed: 0.10.0
✅ TreeCrownDelineator available
✅ Tropical model found: models/230103_randresize_full.pth
🌳 All components installed successfully!
```

---

## 📁 Project Structure

```
MangroVision/
├── models/
│   ├── 230103_randresize_full.pth  ← New tropical model (recommended)
│   └── 230717_tropical_base.pth    ← Current model (backup)
├── canopy_detection/
│   ├── detectree2_detector.py      ← Updated with samgeo methods
│   ├── samgeo_ortho_detector.py    ← New: Orthophoto detection
│   └── validation_metrics.py       ← New: Quality validation
├── MAP/
│   └── odm_orthophoto/
│       └── odm_orthophoto.tif      ← Your orthophoto
└── output_geojson/
    └── detected_crowns.geojson     ← Detection results
```

---

## 🚀 Quick Start Usage

### Basic Detection on GeoTIFF

```python
from samgeo.detectree2 import TreeCrownDelineator

# Initialize delineator
delineator = TreeCrownDelineator(
    model_path="models/230103_randresize_full.pth",
    device="cpu"  # or "cuda" for GPU
)

# Run detection on high-res orthophoto
delineator.predict(
    image_path="MAP/odm_orthophoto/odm_orthophoto.tif",
    output_path="output_geojson/detected_crowns.geojson",
    tile_width=100,
    tile_height=100,
    output_format="geojson"
)

print(f"✅ Detection complete! Results saved to GeoJSON")
```

### In MangroVision Streamlit App

The app will automatically use samgeo if installed:

```powershell
streamlit run app.py
```

---

## 🔍 Troubleshooting

### Issue: "No module named 'detectron2'"
**Solution:** Reinstall detectron2 from GitHub
```powershell
pip uninstall detectron2
pip install git+https://github.com/facebookresearch/detectron2.git
```

### Issue: "GDAL not found"
**Solution:** Install GDAL using conda (easier on Windows)
```powershell
conda install -c conda-forge gdal
```

### Issue: "CUDA out of memory"
**Solution:** Use CPU device or reduce tile size
```python
delineator = TreeCrownDelineator(device="cpu")
# OR
delineator.predict(..., tile_width=50, tile_height=50)
```

### Issue: "Model file not found"
**Solution:** Download the model manually:
1. Visit: https://zenodo.org/record/7123579
2. Download `230103_randresize_full.pth`
3. Save to `MangroVision/models/`

---

## 📚 Additional Resources

- **Detectree2 Paper:** https://besjournals.onlinelibrary.wiley.com/doi/full/10.1111/2041-210X.13901
- **Samgeo Docs:** https://samgeo.gishub.org/
- **Detectron2 Docs:** https://detectron2.readthedocs.io/
- **PyTorch Docs:** https://pytorch.org/docs/

---

## ✅ Installation Complete!

You're now ready to detect tree crowns with state-of-the-art accuracy using:
- **Detectree2** for precise crown segmentation
- **Samgeo wrapper** for easy GeoTIFF processing
- **Pre-trained tropical model** optimized for mangroves

Next steps:
1. Run detection on your orthophoto: `python canopy_detection/samgeo_ortho_detector.py`
2. View results in Streamlit: `streamlit run app.py`
3. Validate detection quality: `python canopy_detection/validation_metrics.py`

Happy detecting! 🌳🌿
