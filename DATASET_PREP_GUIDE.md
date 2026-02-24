# Dataset Preparation Guide for Better Training

## 🎯 Overview
With 240 training images, proper dataset analysis and augmentation are **crucial** for good results.

---

## 🔧 Pre-Training Scripts (Run These First!)

### 1. **Analyze Dataset Quality**
```bash
python analyze_dataset.py
```

**What it does:**
- ✅ Checks class distribution (Mangrove vs Bungalon balance)
- ✅ Analyzes annotation sizes and quality
- ✅ Identifies potential issues (small annotations, missing data)
- ✅ Provides training recommendations
- ✅ Checks for class imbalance

**Run this to:**
- Understand your data before training
- Identify any annotation problems
- Get customized training recommendations

---

### 2. **Visualize Annotations**
```bash
python visualize_dataset.py
```

**What it does:**
- ✅ Shows 10 random training images with annotations overlaid
- ✅ Shows 5 validation images
- ✅ Displays examples of each class
- ✅ Saves visualizations to `output/dataset_visualization/`

**Run this to:**
- Verify annotations are correct
- Check polygon quality
- Ensure no mislabeled data
- Visually inspect canopy boundaries

---

## 🔄 Data Augmentation (Already Added!)

Your training script now includes **automatic augmentation**:

### Augmentations Applied:
✅ **Random scaling** (0.8x - 1.2x)  
✅ **Horizontal flip** (aerial images)  
✅ **Brightness variation** (0.8x - 1.2x)  
✅ **Contrast adjustment** (0.8x - 1.2x)  
✅ **Saturation changes** (0.8x - 1.2x)  

### Why This Helps:
- 240 images → **Effectively 1000+ images** with augmentation
- Model learns to handle varying lighting conditions
- Reduces overfitting
- Better generalization to new images

---

## 📊 Expected Dataset Analysis Output

When you run `analyze_dataset.py`, you'll see:

### Class Distribution
```
Mangrove-Canopy          1,500  (46.1%)
Bungalon Canopy          1,756  (53.9%)
```
- ✅ **Balanced** if ratio < 3:1
- ⚠️ **Imbalanced** if ratio > 3:1 (needs attention)

### Annotation Sizes
```
Mean area: 5,234 px²
Median: 4,120 px²
Min: 250 px² 
Max: 45,000 px²
```
- ⚠️ Small annotations (< 100 px²) may be hard to detect
- ✅ Consistent sizes = easier training

### Annotations per Image
```
Mean: 13.6 annotations/image
Median: 12 annotations/image
```
- Good for dense canopy detection

---

## ⚠️ Common Issues & Solutions

### Issue 1: Class Imbalance
**Problem:** One class has 3x more samples than another  

**Solutions:**
- Enable focal loss in training (helps minority class)
- Oversampleminority class during training
- Collect more data for minority class

### Issue 2: Small Annotations
**Problem:** Some mangrove canopies are < 100 pixels  

**Solutions:**
- Use higher resolution images
- Remove very small annotations (< 50 px²)
- Lower confidence threshold during inference

### Issue 3: Inconsistent Image Sizes
**Problem:** Images have different dimensions  

**Solutions:**
- ✅ Augmentation automatically handles this
- Model resizes images during training

### Issue 4: Poor Annotation Quality
**Problem:** Polygons don't match canopy boundaries  

**Solutions:**
- Review visualizations from `visualize_dataset.py`
- Re-annotate problem images in Roboflow
- Remove poor quality annotations

---

## 📈 Training Recommendations

### For 240 Images (Your Case):

**Iterations:** 3,000 - 5,000  
**Learning Rate:** 0.00025 (current)  
**Batch Size:** 2 (current)  
**Augmentation:** ✅ Enabled (strong)  
**Base Model:** Detectree2 tropical (transfer learning)  

### Expected Training Time:
- RTX 3050 Ti: ~20-30 minutes (3000 iterations)
- CPU: ~2-3 hours

### Expected Performance:
- **Good:** AP50 > 50% (usable)
- **Very Good:** AP50 > 60% (production-ready)
- **Excellent:** AP50 > 70% (high quality)

With detectree2 base + 240 images, expect **55-65% AP50**.

---

## 🚀 Recommended Workflow

### Before Training:

1. **Analyze your data**
   ```bash
   python analyze_dataset.py
   ```
   - Check for any warnings
   - Note class distribution

2. **Visualize annotations**
   ```bash
   python visualize_dataset.py
   ```
   - Review saved images in `output/dataset_visualization/`
   - Verify annotation quality

3. **Fix any issues**
   - Re-annotate if needed in Roboflow
   - Re-download dataset
   - Run analysis again

### During Training:

4. **Start training**
   ```bash
   python train_custom_model.py
   ```
   - Monitor loss (should decrease steadily)
   - Watch validation metrics

5. **Check progress**
   - Loss should drop from ~4.0 to < 1.0
   - AP50 should increase steadily
   - Save checkpoints every 500 iterations

### After Training:

6. **Evaluate model**
   - Check final AP50 score
   - Test on validation images
   - Integrate into MangroVision system

---

## 🎯 Quality Targets

### Minimum Viable:
- ✅ At least 100 training images
- ✅ Both classes represented
- ✅ Clean annotations (polygons match boundaries)
- ✅ AP50 > 40%

### Production Ready:
- ✅ 200+ training images ← **You have 240 ✓**
- ✅ Balanced classes
- ✅ High-quality annotations
- ✅ AP50 > 55%

### Excellent:
- ✅ 500+ training images
- ✅ Diverse conditions (lighting, seasons)
- ✅ Expert-level annotations
- ✅ AP50 > 65%

---

## 📝 Checklist Before Training

Run through this list:

- [ ] Downloaded dataset from Roboflow
- [ ] Ran `analyze_dataset.py` ← **Do this first!**
- [ ] Reviewed analysis output (no critical warnings)
- [ ] Ran `visualize_dataset.py` ← **Do this second!**
- [ ] Checked visualization images (annotations look good)
- [ ] Got detectree2 base model (230103_randresize_full.pth)
- [ ] Placed model in `models/230103_randresize_full.pth`
- [ ] Ready to train!

---

## 💡 Pro Tips

1. **Save intermediate checkpoints**
   - Training saves every 500 iterations
   - If training crashes, you can resume

2. **Monitor training curves**
   - Check `models/custom_mangrove_model/metrics.json`
   - Loss should decrease smoothly

3. **Compare checkpoints**
   - Test different checkpoints (500, 1000, 1500...)
   - Sometimes earlier checkpoints perform better

4. **Iterate on data**
   - If AP50 < 40%, add more training data
   - Focus on difficult cases (small canopies, dense areas)

5. **Use validation set**
   - Never train on validation images
   - Validation metrics show true performance

---

## 🤔 Need More Data?

If your model isn't performing well (AP50 < 40%):

1. **Collect more images**
   - Aim for 500+ total
   - Focus on challenging scenarios

2. **Improve annotations**
   - Ensure polygons are accurate
   - Add missing canopies
   - Remove false annotations

3. **Balance classes**
   - Equal samples of Mangrove and Bungalon
   - Or use weighted loss

4. **Increase diversity**
   - Different times of day
   - Different seasons
   - Different drone altitudes

---

## ✅ Summary

**Before your team member sends the detectree2 model:**

1. Run `python analyze_dataset.py` 
2. Run `python visualize_dataset.py`
3. Check output/dataset_visualization/ folder
4. Fix any issues found
5. Then you're ready to train!

**These scripts will help ensure you get the best possible training results from your 240 images.**

---

*Good dataset preparation = Better model performance!* 🎯
