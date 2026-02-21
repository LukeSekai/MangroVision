# 🌳 SETUP COMPLETE - QUICK REFERENCE GUIDE

## ✅ What's Been Installed

- **Python 3.13** environment
- **PyTorch 2.10.0** (CPU version)
- **Detectron2 0.6** (Facebook's detection framework)
- **Detectree2 2.1.2** (Tree crown detection)
- **GeoPandas, Shapely, Rasterio** (GIS tools)
- **OpenCV, NumPy, Matplotlib** (Image processing)

---

## 📁 Your Project Structure

```
MangroVision/
│
├── canopy_detection/
│   ├── config.py                    # Settings (buffer distance, thresholds, etc.)
│   ├── canopy_detector.py          # Main detection pipeline
│   ├── demo_quickstart.py          # Quick demo and examples
│   └── test_installation.py        # Verify installation
│
├── drone_images/                   # 📸 PUT YOUR DRONE SHOTS HERE
│   └── (add your 90° drone images)
│
├── output/                         # 📊 Results appear here
│   └── demo/
│       ├── sample_drone_image.png
│       ├── workflow_example.png
│       └── training_example.py
│
├── models/                         # 🧠 Trained model weights
│   └── (models will be saved here)
│
├── dataset_frames/                 # Video frames
├── flight_videos/                  # Original videos
├── extract_frames.py              # Frame extraction tool
└── README.md                       # Full documentation

```

---

## 🚀 Quick Commands

### 1. Test Installation
```powershell
cd MangroVision\canopy_detection
C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\Scripts\python.exe test_installation.py
```

### 2. Run Demo
```powershell
cd MangroVision\canopy_detection
C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\Scripts\python.exe demo_quickstart.py
```

### 3. Process Your Images (after adding to drone_images/)
```powershell
cd MangroVision\canopy_detection
C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\Scripts\python.exe canopy_detector.py
```

---

## 🎯 For Your 20% Defense

### What to Present:

1. **System Architecture**
   - Show the project folder structure
   - Explain Mask R-CNN and detectree2
   - Demonstrate the installed environment

2. **Workflow Concept**
   - Drone captures 90° overhead images
   - Detectree2 segments tree canopies
   - System creates 1m buffer (danger zone)
   - Remaining area = safe planting zones

3. **Demo Materials**
   - Show `output/demo/workflow_example.png`
   - Explain the color coding:
     - Red = Canopy danger zones
     - Orange = 1m buffer
     - Green = Safe planting areas

4. **Future Steps**
   - Collect 50-100 mangrove images
   - Annotate with CVAT
   - Train model on mangrove data
   - Deploy in Leganes area

---

## 📊 Key Parameters (in config.py)

```python
BUFFER_DISTANCE_METERS = 1.0      # Safety buffer around trees
CONFIDENCE_THRESHOLD = 0.5        # Detection confidence (0-1)
DEVICE = "cpu"                    # "cpu" or "cuda" (for GPU)
DEFAULT_CRS = "EPSG:32651"        # Philippines UTM Zone 51N
```

---

## 🔄 Typical Workflow

### Phase 1: Setup (DONE ✅)
- Install detectron2 & detectree2
- Create project structure
- Test installation

### Phase 2: Data Collection (NEXT)
1. Fly drone over mangrove area at 90° angle
2. Capture high-res images (recommended: 4K or higher)
3. Save to `drone_images/` folder
4. Ensure good lighting and clear canopy visibility

### Phase 3: Annotation (For custom model)
1. Use CVAT (cvat.ai) to annotate tree canopies
2. Draw polygons around each tree crown
3. Label as "mangrove" or "tree"
4. Export in COCO JSON format
5. Split into training (80%) and validation (20%)

### Phase 4: Training (After annotation)
```python
# Use the provided training_example.py
C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\Scripts\python.exe training_example.py
```

### Phase 5: Prediction & Analysis
```python
# Run on new images
C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\Scripts\python.exe canopy_detector.py
```

---

## 🛠️ Troubleshooting

### "No module named detectron2"
- Make sure you're using the correct Python:
  `C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\Scripts\python.exe`

### "Out of memory"
- Reduce image size before processing
- Set `DEVICE = "cpu"` in config.py
- Process one image at a time

### "Poor detection quality"
- Increase `CONFIDENCE_THRESHOLD` (e.g., 0.7)
- Use higher resolution images
- Train on your specific mangrove dataset

### "Images not found"
- Check that images are in `drone_images/` folder
- Verify file extensions (.jpg, .png, .tif)
- Use absolute paths if needed

---

## 📝 For Thesis Documentation

### System Requirements:
- Python 3.13+
- PyTorch 2.10+
- Detectron2 0.6
- Detectree2 2.1.2
- Minimum 8GB RAM
- GPU recommended (but not required)

### Input Requirements:
- Drone imagery: 90-degree overhead shots
- Format: JPEG, PNG, or TIFF
- Resolution: Minimum 2000x2000 pixels (higher is better)
- Coverage: Clear view of tree canopies

### Output Formats:
- Visualization images (PNG)
- GeoJSON with detection polygons
- Statistics (JSON format)
- Shapefiles for GIS integration

---

## 🎓 Academic Citations

```
Ball, J.G.C., et al. (2023). Detectree2: A Python package for 
tree crown detection in RGB imagery using Mask R-CNN. 
Methods in Ecology and Evolution.

He, K., et al. (2017). Mask R-CNN. 
Proceedings of the IEEE International Conference on Computer Vision (ICCV).
```

---

## 📞 Resources

- **Detectree2 Docs**: https://github.com/PatBall1/detectree2
- **Detectron2 Docs**: https://detectron2.readthedocs.io/
- **CVAT Annotation**: https://www.cvat.ai/
- **COCO Format**: https://cocodataset.org/#format-data

---

## ✨ Next Steps

1. **NOW**: Review the demo output in `output/demo/`
2. **TODAY**: Collect or organize your drone images
3. **THIS WEEK**: Practice your 20% defense presentation
4. **AFTER DEFENSE**: 
   - Annotate 50-100 images
   - Train your custom model
   - Test on validation data
   - Deploy for Leganes mapping

---

## 💡 Pro Tips

1. **Start Small**: Test on 5-10 images first
2. **High Quality**: Better images = better detections
3. **Consistent Altitude**: Keep drone height consistent
4. **Good Lighting**: Avoid shadows and overexposure
5. **Backup Everything**: Save raw images and outputs
6. **Document Process**: Take notes for your thesis

---

**🎉 Your system is ready! Good luck with your defense! 🎉**

Questions about the setup? Everything you need is in:
- `README.md` (full documentation)
- `demo_quickstart.py` (code examples)
- `config.py` (customizable settings)
