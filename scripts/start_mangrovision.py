#!/usr/bin/env python3
"""
Quick Start Script for MangroVision with AI Detection
Launches the Streamlit app with proper environment
"""

import subprocess
import sys
from pathlib import Path

def main():
    print("\n" + "="*70)
    print("🌿 MANGROVISION - AI-POWERED CANOPY DETECTION")
    print("="*70 + "\n")
    
    # Check if we're in the right directory
    app_file = Path("app.py")
    if not app_file.exists():
        print("⚠️  Error: app.py not found!")
        print("Please run this script from the MangroVision directory\n")
        return
    
    print("✓ Found app.py")
    print("\n🚀 Launching MangroVision...")
    print("\nFeatures available:")
    print("  • HSV Color Detection (Fast)")
    print("  • AI Detection via detectree2 (Accurate)")
    print("  • Hexagonal planting zone generation")
    print("  • Forbidden zone filtering")
    print("  • Interactive map visualization")
    print("\n" + "="*70)
    print("The app will open in your browser shortly...")
    print("Press Ctrl+C to stop the server")
    print("="*70 + "\n")
    
    # Launch streamlit
    try:
        subprocess.run([sys.executable, "-m", "streamlit", "run", "app.py"])
    except KeyboardInterrupt:
        print("\n\n✓ MangroVision stopped. Goodbye! 🌿\n")

if __name__ == "__main__":
    main()
