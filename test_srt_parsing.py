"""
Quick test to verify SRT parsing works with your DJI files
"""
import sys
from pathlib import Path

# Add canopy_detection to path
sys.path.append(str(Path(__file__).parent / "canopy_detection"))

from canopy_detection.flight_log_parser import FlightLogParser

def test_srt_files():
    """Test parsing of DJI SRT files in drone_images folder"""
    
    print("="*60)
    print("Testing DJI SRT File Parsing")
    print("="*60)
    
    drone_images_dir = Path("drone_images")
    srt_files = list(drone_images_dir.glob("*.srt"))
    
    if not srt_files:
        print("❌ No SRT files found in drone_images/")
        print("   Please add your DJI_*.srt files to the drone_images folder")
        return
    
    print(f"\nFound {len(srt_files)} SRT files:\n")
    
    parser = FlightLogParser()
    
    for srt_file in sorted(srt_files):
        print(f"\n{'='*60}")
        print(f"📁 File: {srt_file.name}")
        print(f"{'='*60}")
        
        heading = parser.extract_heading(str(srt_file))
        
        if heading is not None:
            print(f"\n✅ SUCCESS: Heading = {heading:.1f}°")
            
            # Show cardinal direction
            if 337.5 <= heading or heading < 22.5:
                direction = "North"
            elif 22.5 <= heading < 67.5:
                direction = "Northeast"
            elif 67.5 <= heading < 112.5:
                direction = "East"
            elif 112.5 <= heading < 157.5:
                direction = "Southeast"
            elif 157.5 <= heading < 202.5:
                direction = "South"
            elif 202.5 <= heading < 247.5:
                direction = "Southwest"
            elif 247.5 <= heading < 292.5:
                direction = "West"
            else:
                direction = "Northwest"
            
            print(f"   Direction: {direction}")
            print(f"   Ready to use in MangroVision! ✓")
        else:
            print(f"\n❌ FAILED: Could not extract heading")
            print(f"   This file may not contain GPS trajectory data")
    
    print(f"\n{'='*60}")
    print("Testing complete!")
    print("="*60)

if __name__ == "__main__":
    test_srt_files()
