"""
MangroVision - Download Detectree2 Pre-trained Models
Utility script to download and setup AI models
"""

import sys
from pathlib import Path

# Add canopy_detection to path
sys.path.append(str(Path(__file__).parent))

from detectree2_detector import Detectree2Detector


def download_models():
    """Download all available pre-trained models"""
    print("\n" + "="*70)
    print("DETECTREE2 MODEL DOWNLOADER")
    print("MangroVision AI Model Setup")
    print("="*70 + "\n")
    
    models_dir = Path(__file__).parent.parent / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 Models will be saved to: {models_dir}\n")
    
    detector = Detectree2Detector()
    
    print("Available models:")
    print("  1. 'general' - General purpose tree detection (trained on multiple forests)")
    print("  2. 'paracou' - Tropical forest model (good for mangroves)\n")
    
    # Option to download
    print("Note: Models are ~200MB each and will be downloaded from Zenodo")
    print("If download fails, the system will use COCO pre-trained model instead.\n")
    
    choice = input("Download pre-trained model? (general/paracou/skip) [skip]: ").strip().lower()
    
    if choice in ['general', 'paracou']:
        try:
            model_path = detector.download_pretrained_model(
                model_name=choice,
                output_dir=str(models_dir)
            )
            
            if model_path:
                print(f"\n✓ Model ready at: {model_path}")
                print(f"\nTo use this model in your detector:")
                print(f"detector = HexagonDetector(")
                print(f"    detection_method='ai',")
                print(f"    ai_confidence=0.5")
                print(f")")
            
        except Exception as e:
            print(f"\n⚠️ Download failed: {e}")
            print(f"System will use COCO pre-trained model (still works well)")
    else:
        print("\n✓ Skipping download")
        print("System will use COCO pre-trained Mask R-CNN (works fine for trees)")
    
    print("\n" + "="*70)
    print("SETUP COMPLETE")
    print("="*70)
    print("\nYou can now run MangroVision with AI detection:")
    print("  streamlit run app.py")
    print("\nIn the sidebar, select 'AI (detectree2)' for detection method\n")


if __name__ == "__main__":
    download_models()
