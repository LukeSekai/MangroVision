"""Quick test to check GPS data and heading from all three drone image folders"""
import sys, os
from pathlib import Path
sys.path.append(str(Path(__file__).parent / "canopy_detection"))

from canopy_detection.exif_extractor import ExifExtractor
from canopy_detection.gps_trajectory import GPSTrajectoryAnalyzer

def test_folder(folder_name):
    extractor = ExifExtractor()
    analyzer = GPSTrajectoryAnalyzer()
    
    folder = Path(f"drone_images/{folder_name}")
    frames = sorted(folder.glob("*.jpg"))
    
    print(f"\n{'='*60}")
    print(f"📁 Folder: {folder_name} ({len(frames)} images)")
    print(f"{'='*60}")
    
    # Check GPS on first, middle, last frame
    samples = [frames[0], frames[len(frames)//2], frames[-1]]
    for f in samples:
        meta = extractor.extract_all_metadata(str(f))
        if meta.get("has_gps"):
            g = meta["gps"]
            print(f"  {f.name}: lat={g['latitude']:.7f}, lon={g['longitude']:.7f}, alt={g.get('altitude', '?')}m")
    
    # Sample every 10th frame for trajectory heading
    for i, f in enumerate(frames):
        if i % 10 == 0:
            meta = extractor.extract_all_metadata(str(f))
            if meta.get("has_gps"):
                g = meta["gps"]
                analyzer.add_point(g["latitude"], g["longitude"])
    
    heading = analyzer.get_flight_heading()
    if heading is not None:
        cardinals = [(0,"North"),(45,"NE"),(90,"East"),(135,"SE"),(180,"South"),(225,"SW"),(270,"West"),(315,"NW")]
        direction = min(cardinals, key=lambda c: min(abs(heading-c[0]), 360-abs(heading-c[0])))[1]
        print(f"\n  ✅ GPS Heading: {heading:.1f}° ({direction})")
    else:
        print(f"\n  ⚠️  Drone appears stationary - cannot calculate heading from GPS")
    
    return heading

if __name__ == "__main__":
    print("Testing GPS data across all drone image folders...")
    for folder in ["dataset_with_gps", "dataset_with_gps_0029", "dataset_with_gps_0030"]:
        test_folder(folder)
    print(f"\n{'='*60}")
    print("Done! Images are ready to use in MangroVision.")
    print("Upload any frame_*.jpg from any folder in the Streamlit app.")

