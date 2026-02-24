# 🚀 MangroVision - Custom Model Training Guide

## ✅ What's Ready

Your annotated dataset from Roboflow is downloaded and verified:
- **240 training images** with 3,256 annotations
- **15 validation images** with 196 annotations
- **2 classes**: Mangrove-Canopy, Bungalon Canopy
- **Format**: COCO with polygon segmentation (ready for Mask R-CNN)

## ⚠️ Important: Training Requirements

### Current Setup:
- ✅ PyTorch installed (CPU version)
- ✅ Training scripts created
- ⚠️ **No GPU detected** - CPU training will be VERY SLOW

### Training Time Estimates:
- **With GPU** (RTX 3050 Ti / GTX 1080): 2-4 hours
- **With CPU** (your current setup): 24-48 hours ⚠️

---

## 🎯 Recommended Training Options

### **Option 1: Google Colab (RECOMMENDED) 🌟**

**Pros:**
- Free GPU access (Tesla T4)
- Fast training (2-4 hours)
- No local setup needed
- Easy to use

**Steps:**

1. **Upload your dataset** to Google Drive:
   - Zip: `training_data/roboflow_dataset`
   - Upload to Drive

2. **Create Colab Notebook:**
   ```python
   # Install dependencies
   !pip install torch torchvision
   !pip install 'git+https://github.com/facebookresearch/detectron2.git'
   
   # Mount Google Drive
   from google.colab import drive
   drive.mount('/content/drive')
   
   # Unzip dataset
   !unzip /content/drive/MyDrive/roboflow_dataset.zip -d /content/
   
   # Copy your train_custom_model.py to Colab
   # Run training
   !python train_custom_model.py
   ```

3. **Download trained model** back to your computer

**Tutorial:** https://colab.research.google.com/

---

### **Option 2: Local GPU Training**

If you have access to a laptop/PC with NVIDIA GPU:

1. **Check GPU:**
   ```powershell
   nvidia-smi
   ```

2. **Install CUDA PyTorch:**
   ```powershell
   pip uninstall torch torchvision torchaudio
   pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
   ```

3. **Install Detectron2:**
   ```powershell
   pip install 'git+https://github.com/facebookresearch/detectron2.git'
   ```

4. **Run training:**
   ```powershell
   cd MangroVision
   python train_custom_model.py
   ```

---

### **Option 3: CPU Training (Not Recommended)**

Only if you have 24-48 hours to spare:

1. **Install Detectron2** (try pre-built wheel):
   ```powershell
   pip install detectron2 -f https://dl.fbaipublicfiles.com/detectron2/wheels/cpu/torch2.0/index.html
   ```

2. **Run training:**
   ```powershell
   python train_custom_model.py
   ```

3. **Let it run overnight** (or for 2 days)

---

## 📖 After Training

Once training completes, you'll have:
- `models/custom_mangrove_model/model_final.pth` - Your trained model

### Test Your Model:
```powershell
python test_custom_model.py
```

### Use in Your App:

Update `app.py` or `canopy_detection/detectree2_detector.py`:

```python
detector = Detectree2Detector(
    model_path="models/custom_mangrove_model/model_final.pth",
    confidence_threshold=0.5
)
```

---

## 🎓 Training Parameters (in train_custom_model.py)

You can adjust these based on your needs:

```python
MAX_ITERATIONS = 3000  # More iterations = better learning (but longer)
LEARNING_RATE = 0.00025  # Lower = more careful learning
BATCH_SIZE = 2  # Higher = faster (but needs more GPU memory)
```

**Recommended for your dataset:**
- 240 images → 3000-5000 iterations
- Start with defaults, adjust if needed

---

## ❓ Which Option Should You Choose?

### Use **Google Colab** if:
- You want fast results (2-4 hours)
- You don't have a GPU
- You want free compute

### Use **Local GPU** if:
- You have NVIDIA GPU with 4GB+ VRAM
- You want full control
- You plan to train multiple times

### Use **CPU** if:
- You have 24+ hours available
- You just want to test the process
- No other option available

---

## 🚀 Quick Start (Google Colab)

I recommend starting with Google Colab. It's the fastest and easiest option.

**Ready to start?** Let me know which option you want to use, and I'll guide you through it!

---

## 📊 Expected Results

After training, you should see:
- **Detection accuracy**: 85-95% (better than base model)
- **Custom class recognition**: Your specific mangrove types
- **Reduced false positives**: Better understanding of your data

**Your model will be specialized for your specific:**
- Camera angles
- Lighting conditions
- Mangrove species
- Environmental conditions

This is why custom training is powerful! 🎯
