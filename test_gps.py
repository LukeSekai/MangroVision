import sys; sys.path.insert(0, '.')
from canopy_detection.exif_extractor import ExifExtractor
ex = ExifExtractor()
for img in ['drone_images/dataset_with_gps/frame_0049.jpg',
            'drone_images/dataset_with_gps_0030/frame_0049.jpg',
            'drone_images/dataset_with_gps_0029/frame_0049.jpg']:
    m = ex.extract_metadata(img)
    if m.get('has_gps'):
        g = m['gps']
        print(f"{img}: lat={g['latitude']:.7f}, lon={g['longitude']:.7f}")
    else:
        print(f"{img}: NO GPS")
