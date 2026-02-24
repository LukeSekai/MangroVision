"""
Test the hybrid HSV+AI approach for speed improvements
"""
import cv2
import time
from pathlib import Path
from canopy_detection.detectree2_detector import Detectree2Detector

# Load test image
test_image = Path("drone_images/dataset_with_gps/frame_0000.jpg")
image = cv2.imread(str(test_image))

print("=" * 60)
print("🚀 HYBRID HSV+AI APPROACH TEST")
print("=" * 60)
print(f"\n📸 Image: {image.shape[1]}x{image.shape[0]} pixels")

# Initialize detector
detector = Detectree2Detector(confidence_threshold=0.35, device='cpu')
detector.setup_model()

# Run detection with timing
print(f"\n⏱️  Running detection with optimizations...")
start_time = time.time()

polygons, mask, metadata = detector.detect_from_image(image)

end_time = time.time()
processing_time = end_time - start_time

print(f"\n" + "=" * 60)
print("📊 RESULTS")
print("=" * 60)
print(f"✅ Trees detected: {len(polygons)}")
print(f"⏱️  Processing time: {processing_time:.2f} seconds")
print(f"🔲 Tiles processed: {metadata.get('num_tiles', 'N/A')}")
print(f"🌿 Vegetation coverage: {metadata.get('vegetation_coverage_percent', 'N/A'):.1f}%")
print(f"🤖 AI detections: {metadata.get('total_ai_detections', 'N/A')}")
print(f"❌ Filtered non-vegetation: {metadata.get('filtered_non_vegetation', 'N/A')}")

# Estimate time saved
if 'num_tiles' in metadata:
    all_tiles_estimate = processing_time * (60 / metadata['num_tiles'])
    time_saved = all_tiles_estimate - processing_time
    print(f"\n💡 Estimated time if processing ALL tiles: {all_tiles_estimate:.2f}s")
    print(f"⚡ Time saved by skipping empty tiles: {time_saved:.2f}s ({time_saved/all_tiles_estimate*100:.1f}%)")

print("=" * 60)
