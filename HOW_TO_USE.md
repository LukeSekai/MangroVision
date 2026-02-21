# 🚁 STEP-BY-STEP OPERATION GUIDE
## Mangrove Canopy Detection System - How to Use

---

## 📋 PART 1: QUICK START (For 20% Defense Demo)

### Step 1: Verify Installation ✅
**What**: Check that everything is installed correctly  
**How**:
```powershell
cd C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\MangroVision\canopy_detection
C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\Scripts\python.exe test_installation.py
```
**Expected Output**: 
- ✓ All tests passed
- ✓ Detectron2 and Detectree2 versions displayed

---

### Step 2: Run the Demo 🎨
**What**: See how the system works with sample data  
**How**:
```powershell
C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\Scripts\python.exe demo_quickstart.py
```
**What You'll Get**:
- Sample drone image created
- Workflow visualization
- Training code examples
- Files saved in: `output/demo/`

**Look at the outputs**:
```powershell
# Open the output folder
explorer ..\..\output\demo
```

---

## 📸 PART 2: USING YOUR OWN DRONE IMAGES

### Step 3: Prepare Your Drone Images 🖼️
**What**: Add your 90-degree drone shots  
**How**:

1. **Take drone photos**:
   - Fly drone directly overhead (90° angle looking down)
   - Height: 10-50 meters recommended
   - Ensure clear view of tree canopies
   - Good lighting (avoid harsh shadows)
   - High resolution (4K or higher preferred)

2. **Copy images to the folder**:
   ```powershell
   # Navigate to drone_images folder
   cd C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\MangroVision\drone_images
   
   # Copy your images here (example)
   # Just drag and drop your JPG/PNG files into this folder
   ```

3. **Supported formats**:
   - ✅ JPG/JPEG
   - ✅ PNG
   - ✅ TIF/TIFF

4. **Verify images are there**:
   ```powershell
   Get-ChildItem C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\MangroVision\drone_images
   ```

---

### Step 4: Run Detection on Your Images 🔍
**What**: Process your drone images to detect canopies  
**How**:
```powershell
cd C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\MangroVision\canopy_detection
C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\Scripts\python.exe canopy_detector.py
```

**What Happens**:
1. Script finds all images in `drone_images/` folder
2. Loads each image
3. Detects tree canopies (currently uses pre-trained model)
4. Creates buffer zones (1 meter around each canopy)
5. Identifies safe planting zones
6. Saves results to `output/` folder

**While Running, You'll See**:
```
====================================================
MANGROVE CANOPY DETECTION SYSTEM
Leganes Safe Planting Zone Analyzer
====================================================

Found 3 image(s) to process

====================================================
Processing: drone_shot_001.jpg
====================================================

Loaded image: C:\...\drone_images\drone_shot_001.jpg
Image shape: (4000, 6000, 3)

Note: Using pre-trained detectree2 model.
For mangrove-specific detection, train on your labeled data.

Processing complete!
```

---

### Step 5: View Your Results 📊
**What**: Check the processed outputs  
**How**:
```powershell
# Open output folder
explorer C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\MangroVision\output
```

**You'll Find**:
- `[imagename]_zones.png` - Visualization with overlays
- `[imagename]_results.json` - Detection statistics

**Example JSON Output**:
```json
{
  "image_path": "drone_shot_001.jpg",
  "num_detections": 15,
  "confidence_threshold": 0.5,
  "buffer_distance_meters": 1.0
}
```

---

## ⚙️ PART 3: CUSTOMIZE SETTINGS

### Step 6: Adjust Configuration (Optional) 🔧
**What**: Change buffer distance, confidence, etc.  
**How**:

1. **Open config file**:
   ```powershell
   code C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\MangroVision\canopy_detection\config.py
   ```

2. **Common adjustments**:
   ```python
   # Change buffer distance (default is 1.0 meter)
   BUFFER_DISTANCE_METERS = 2.0  # Make it 2 meters instead
   
   # Change detection confidence (default is 0.5)
   CONFIDENCE_THRESHOLD = 0.7    # Higher = stricter detections
   
   # Change to GPU (if you have one)
   DEVICE = "cuda"  # Instead of "cpu"
   ```

3. **Save and rerun**:
   ```powershell
   C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\Scripts\python.exe canopy_detector.py
   ```

---

## 🎓 PART 4: FOR YOUR DEFENSE PRESENTATION

### Step 7: Prepare Your Demo 📽️
**What**: Show your system in action during defense  
**How**:

1. **Have these windows ready**:
   - PowerShell terminal (to run commands)
   - File Explorer showing project structure
   - Image viewer with results
   - Your presentation slides

2. **Demo Script**:
   ```
   "First, I verify the system is installed..."
   [Run test_installation.py]
   
   "Here's our project structure for analyzing Leganes mangrove areas..."
   [Show MangroVision folder]
   
   "The system processes 90-degree drone imagery..."
   [Show sample drone image]
   
   "It detects tree canopies and creates 1-meter buffer zones..."
   [Show workflow visualization]
   
   "Safe planting zones are identified automatically..."
   [Show zones output]
   
   "After the defense, we'll train this specifically on mangrove data..."
   [Mention training plan]
   ```

3. **Key Points to Emphasize**:
   - ✓ Uses state-of-the-art Mask R-CNN (detectree2)
   - ✓ Automated danger zone mapping
   - ✓ Configurable buffer distances
   - ✓ Future GIS integration planned
   - ✓ Will be trained on Leganes-specific data

---

## 🎯 PART 5: AFTER DEFENSE - TRAINING YOUR MODEL

### Step 8: Collect Training Data 📸
**What**: Gather images for mangrove-specific training  
**Target**: 50-100 high-quality images

**Collection Tips**:
- Various lighting conditions
- Different growth stages
- Multiple mangrove areas
- Consistent altitude
- Clear canopy visibility

---

### Step 9: Annotate Your Images 🖊️
**What**: Label tree canopies in your images  
**Tool**: CVAT (Computer Vision Annotation Tool)

**How**:
1. **Go to**: https://www.cvat.ai/ or run locally
2. **Create new project**: "Leganes Mangroves"
3. **Add labels**: "mangrove_canopy"
4. **Upload images**: Your 50-100 training images
5. **Draw polygons**: Around each tree canopy
6. **Export**: As COCO JSON format

**Folder structure for training**:
```
MangroVision/
├── training_data/
│   ├── images/              # 80% of your images
│   └── annotations_train.json
├── validation_data/
│   ├── images/              # 20% of your images
│   └── annotations_val.json
```

---

### Step 10: Train Your Model 🧠
**What**: Train detectree2 on your mangrove dataset  
**How**:

1. **Use the provided training script**:
   ```powershell
   cd C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\MangroVision\output\demo
   C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\Scripts\python.exe training_example.py
   ```

2. **Or create custom training script**:
   ```python
   from detectree2.data_loading import register_train_data
   from detectree2.models.train import setup_cfg, MyTrainer
   
   # Register your dataset
   register_train_data(
       train_location="MangroVision/training_data/images",
       val_location="MangroVision/validation_data/images",
       train_json_name="annotations_train.json",
       val_json_name="annotations_val.json",
       name="leganes_mangroves"
   )
   
   # Setup and train
   cfg = setup_cfg(
       dataset_name="leganes_mangroves",
       num_classes=1,
       max_iter=5000
   )
   
   trainer = MyTrainer(cfg)
   trainer.train()
   ```

3. **Training time**: 
   - CPU: Several hours
   - GPU: 30-60 minutes
   - Depends on dataset size

---

### Step 11: Use Your Trained Model 🚀
**What**: Apply your trained model to new images  
**How**:

1. **Update config.py**:
   ```python
   # Point to your trained model
   MODEL_PATH = "MangroVision/models/trained/model_final.pth"
   ```

2. **Run detection**:
   ```powershell
   C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\Scripts\python.exe canopy_detector.py
   ```

3. **Results will be more accurate** for mangrove-specific detection!

---

## 🔄 TYPICAL WORKFLOW SUMMARY

```
┌─────────────────────────────────────────────────┐
│ 1. ADD DRONE IMAGES to drone_images/ folder    │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ 2. RUN: python canopy_detector.py              │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ 3. SYSTEM PROCESSES:                            │
│    - Detects canopies                          │
│    - Creates buffers                           │
│    - Identifies safe zones                     │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ 4. CHECK RESULTS in output/ folder             │
│    - View visualizations                        │
│    - Read statistics                           │
└─────────────────────────────────────────────────┘
                      ↓
┌─────────────────────────────────────────────────┐
│ 5. ADJUST if needed (config.py)               │
│    - Change buffer distance                     │
│    - Adjust confidence threshold               │
└─────────────────────────────────────────────────┘
```

---

## ⚠️ TROUBLESHOOTING

### Problem: "No images found"
**Solution**: 
```powershell
# Check if images are in correct folder
Get-ChildItem C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\MangroVision\drone_images
# Make sure files have extensions: .jpg, .png, .tif
```

### Problem: "Module not found"
**Solution**:
```powershell
# Make sure you're using the correct Python
C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\Scripts\python.exe -m pip list
# Should show detectron2 and detectree2
```

### Problem: "Out of memory"
**Solution**:
- Reduce image size before processing
- Process fewer images at once
- Close other programs

### Problem: "Poor detection results"
**Solution**:
- This is expected with pre-trained model
- Train on your own mangrove data (Step 10)
- Adjust CONFIDENCE_THRESHOLD in config.py

---

## 📋 QUICK COMMAND REFERENCE

**Run Tests**:
```powershell
cd C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\MangroVision\canopy_detection
C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\Scripts\python.exe test_installation.py
```

**Run Demo**:
```powershell
C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\Scripts\python.exe demo_quickstart.py
```

**Process Images**:
```powershell
C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\Scripts\python.exe canopy_detector.py
```

**Open Output Folder**:
```powershell
explorer C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\MangroVision\output
```

**Edit Config**:
```powershell
code config.py
```

---

## 🎯 CHECKLIST FOR 20% DEFENSE

- [ ] Test installation (run test_installation.py)
- [ ] Run demo (run demo_quickstart.py)
- [ ] Review output visualizations
- [ ] Understand the workflow concept
- [ ] Prepare to explain Mask R-CNN
- [ ] Know your data collection plan
- [ ] Understand training process
- [ ] Plan GIS integration approach

---

## 📚 ADDITIONAL RESOURCES

- **Full Documentation**: See README.md
- **Quick Reference**: See SETUP_COMPLETE.md
- **Configuration**: Edit config.py
- **Training Examples**: Check output/demo/training_example.py

---

**🎓 You're ready to operate the system! Good luck with your defense! 🌳**
