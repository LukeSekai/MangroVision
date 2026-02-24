# Quick Reference: What You Need from Team Member

## File Required
**Name:** `230103_randresize_full.pth`  
**Size:** ~200-300 MB  
**Type:** PyTorch model weights (.pth file)

## Where to Place It
```
MangroVision/models/230103_randresize_full.pth
```

Full path:
```
C:\Users\Lenovo-Pc\OneDrive\Thesis\thesis_env\MangroVision\models\230103_randresize_full.pth
```

## What It Is
- Detectree2 pre-trained tropical forest model
- Already trained on tree canopy detection
- Will be used as BASE for transfer learning
- Your 240 Roboflow images will fine-tune it for mangroves

## After You Get It
Run this command:
```bash
cd MangroVision
python train_custom_model.py
```

## Training Will:
- Load detectree2 base model (knows tree structures)
- Fine-tune on YOUR 240 mangrove images
- Learn to distinguish Mangrove-Canopy vs Bungalon Canopy
- Train for ~3000 iterations (~20 minutes on RTX 3050 Ti)
- Save to: `models/custom_mangrove_model/model_final.pth`

## Status
✅ Training script updated and ready  
✅ Unnecessary files cleaned up  
⏳ Waiting for model file via GDrive  

---
After training completes, the system will use your custom model automatically!
