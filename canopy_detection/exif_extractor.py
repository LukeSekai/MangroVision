"""
EXIF Metadata Extractor for Drone Images
Extracts GPS coordinates, altitude, and camera information from drone photos
"""

from PIL import Image
from PIL.ExifTags import TAGS, GPSTAGS
from typing import Dict, Optional, Tuple
import os


class ExifExtractor:
    """Extract metadata from drone images"""
    
    @staticmethod
    def get_exif_data(image_path: str) -> Dict:
        """
        Extract all EXIF data from an image including GPS IFD
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary of EXIF data
        """
        try:
            image = Image.open(image_path)
            exif_data = {}
            
            # Get EXIF tags
            exif = image.getexif()
            if exif is None:
                return {}
            
            # Get standard EXIF tags
            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                exif_data[tag_name] = value
            
            # Get GPS IFD (Image File Directory) - this is where GPS data is stored
            try:
                gps_ifd = exif.get_ifd(0x8825)  # 0x8825 is the GPS IFD tag
                if gps_ifd:
                    gps_data = {}
                    for tag_id, value in gps_ifd.items():
                        tag_name = GPSTAGS.get(tag_id, tag_id)
                        gps_data[tag_name] = value
                    exif_data['GPSInfo'] = gps_data
            except (KeyError, AttributeError):
                # No GPS data available
                pass
            
            return exif_data
        except Exception as e:
            print(f"Error reading EXIF data: {e}")
            return {}
    
    @staticmethod
    def _convert_to_degrees(value) -> float:
        """
        Convert GPS coordinates to degrees in float format
        
        Args:
            value: GPS coordinate in degrees, minutes, seconds format
            
        Returns:
            Coordinate in decimal degrees
        """
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)
    
    @staticmethod
    def get_gps_info(exif_data: Dict) -> Optional[Dict]:
        """
        Extract GPS information from EXIF data
        
        Args:
            exif_data: Dictionary of EXIF data
            
        Returns:
            Dictionary with GPS info or None if not available
        """
        if 'GPSInfo' not in exif_data:
            return None
        
        gps_info = {}
        gps_parsed = exif_data['GPSInfo']
        
        # Check if gps_data is a dictionary (should be after get_exif_data processing)
        if not isinstance(gps_parsed, dict):
            return None
        
        # Extract latitude
        if 'GPSLatitude' in gps_parsed and 'GPSLatitudeRef' in gps_parsed:
            lat = ExifExtractor._convert_to_degrees(gps_parsed['GPSLatitude'])
            if gps_parsed['GPSLatitudeRef'] == 'S':
                lat = -lat
            gps_info['latitude'] = lat
        
        # Extract longitude
        if 'GPSLongitude' in gps_parsed and 'GPSLongitudeRef' in gps_parsed:
            lon = ExifExtractor._convert_to_degrees(gps_parsed['GPSLongitude'])
            if gps_parsed['GPSLongitudeRef'] == 'W':
                lon = -lon
            gps_info['longitude'] = lon
        
        # Extract altitude (relative to sea level)
        if 'GPSAltitude' in gps_parsed:
            alt_value = gps_parsed['GPSAltitude']
            # Handle both float and rational (tuple) formats
            if isinstance(alt_value, tuple):
                altitude = float(alt_value[0]) / float(alt_value[1]) if alt_value[1] != 0 else 0.0
            else:
                altitude = float(alt_value)
            
            # Check if below sea level (0 = above, 1 = below)
            if 'GPSAltitudeRef' in gps_parsed and gps_parsed['GPSAltitudeRef'] == 1:
                altitude = -altitude
            
            gps_info['altitude'] = altitude
        
        # DJI drones store relative altitude (above ground level) in a custom tag
        # This is the flight altitude we actually want for GSD calculation
        if 'RelativeAltitude' in gps_parsed:
            rel_alt = gps_parsed['RelativeAltitude']
            if isinstance(rel_alt, tuple):
                gps_info['relative_altitude'] = float(rel_alt[0]) / float(rel_alt[1]) if rel_alt[1] != 0 else 0.0
            else:
                gps_info['relative_altitude'] = float(rel_alt)
        
        # Extract compass heading/direction (GPSImgDirection)
        if 'GPSImgDirection' in gps_parsed:
            heading = gps_parsed['GPSImgDirection']
            if isinstance(heading, tuple):
                gps_info['heading'] = float(heading[0]) / float(heading[1]) if heading[1] != 0 else 0.0
            else:
                gps_info['heading'] = float(heading)
        
        return gps_info if gps_info else None
    
    @staticmethod
    def get_camera_info(exif_data: Dict) -> Dict:
        """
        Extract camera information from EXIF data
        
        Args:
            exif_data: Dictionary of EXIF data
            
        Returns:
            Dictionary with camera info
        """
        camera_info = {}
        
        # Camera make and model
        if 'Make' in exif_data:
            camera_info['make'] = exif_data['Make']
        if 'Model' in exif_data:
            camera_info['model'] = exif_data['Model']
        
        # Image dimensions
        if 'ExifImageWidth' in exif_data:
            camera_info['image_width'] = exif_data['ExifImageWidth']
        if 'ExifImageHeight' in exif_data:
            camera_info['image_height'] = exif_data['ExifImageHeight']
        
        # Focal length
        if 'FocalLength' in exif_data:
            focal = exif_data['FocalLength']
            # Handle both int and rational (tuple) formats
            if isinstance(focal, tuple):
                camera_info['focal_length_mm'] = focal[0] / focal[1]
            else:
                camera_info['focal_length_mm'] = float(focal)
        
        # ISO, aperture, shutter speed
        if 'ISOSpeedRatings' in exif_data:
            camera_info['iso'] = exif_data['ISOSpeedRatings']
        if 'FNumber' in exif_data:
            f = exif_data['FNumber']
            if isinstance(f, tuple):
                camera_info['aperture'] = f[0] / f[1]
            else:
                camera_info['aperture'] = float(f)
        if 'ExposureTime' in exif_data:
            exp = exif_data['ExposureTime']
            if isinstance(exp, tuple):
                camera_info['shutter_speed'] = f"1/{int(exp[1]/exp[0])}"
            else:
                camera_info['shutter_speed'] = exp
        
        # DateTime
        if 'DateTime' in exif_data:
            camera_info['datetime'] = exif_data['DateTime']
        
        return camera_info
    
    @staticmethod
    def extract_all_metadata(image_path: str) -> Dict:
        """
        Extract all relevant metadata from drone image
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dictionary with GPS, camera, and other metadata
        """
        # Get all EXIF data
        exif_data = ExifExtractor.get_exif_data(image_path)
        
        if not exif_data:
            return {
                'has_exif': False,
                'has_gps': False,
                'error': 'No EXIF data found in image'
            }
        
        # Extract GPS info
        gps_info = ExifExtractor.get_gps_info(exif_data)
        
        # Extract camera info
        camera_info = ExifExtractor.get_camera_info(exif_data)
        
        # Get image dimensions from PIL if not in EXIF
        image = Image.open(image_path)
        width, height = image.size
        
        metadata = {
            'has_exif': True,
            'has_gps': gps_info is not None,
            'gps': gps_info,
            'camera': camera_info,
            'image_width': camera_info.get('image_width', width),
            'image_height': camera_info.get('image_height', height),
            'file_path': image_path,
            'file_size_mb': os.path.getsize(image_path) / (1024 * 1024)
        }
        
        return metadata
    
    @staticmethod
    def detect_drone_model(camera_info: Dict) -> str:
        """
        Try to detect drone model from camera make/model
        
        Args:
            camera_info: Dictionary with camera information
            
        Returns:
            Drone model string that matches GSDCalculator.COMMON_DRONES keys
        """
        make = camera_info.get('make', '').upper()
        model = camera_info.get('model', '').upper()
        
        # DJI drone detection
        if 'DJI' in make or 'DJI' in model:
            if 'MINI 3' in model or 'MINI3' in model:
                return 'DJI_MINI_3'
            elif 'MAVIC 3' in model or 'MAVIC3' in model:
                return 'DJI_MAVIC_3'
            elif 'AIR 2S' in model or 'AIR2S' in model:
                return 'DJI_AIR_2S'
            elif 'PHANTOM 4' in model or 'PHANTOM4' in model:
                return 'DJI_PHANTOM_4'
        
        # Default to generic 4K
        return 'GENERIC_4K'
    
    @staticmethod
    def print_metadata_summary(metadata: Dict):
        """Print a formatted summary of extracted metadata"""
        print("\n" + "="*70)
        print("DRONE IMAGE METADATA")
        print("="*70)
        
        if not metadata.get('has_exif'):
            print("❌ No EXIF data found in image")
            return
        
        print(f"\n📸 Camera Information:")
        camera = metadata.get('camera', {})
        if 'make' in camera:
            print(f"   Make: {camera['make']}")
        if 'model' in camera:
            print(f"   Model: {camera['model']}")
        if 'focal_length_mm' in camera:
            print(f"   Focal Length: {camera['focal_length_mm']:.1f}mm")
        
        print(f"\n📏 Image Dimensions:")
        print(f"   {metadata['image_width']} × {metadata['image_height']} pixels")
        print(f"   File Size: {metadata['file_size_mb']:.2f} MB")
        
        if metadata.get('has_gps'):
            gps = metadata['gps']
            print(f"\n📍 GPS Information:")
            if 'latitude' in gps and 'longitude' in gps:
                print(f"   Latitude: {gps['latitude']:.6f}°")
                print(f"   Longitude: {gps['longitude']:.6f}°")
            if 'altitude' in gps:
                print(f"   Altitude (MSL): {gps['altitude']:.1f}m")
            if 'relative_altitude' in gps:
                print(f"   Relative Altitude (AGL): {gps['relative_altitude']:.1f}m")
        else:
            print(f"\n❌ No GPS data found in image")
        
        print("="*70 + "\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        metadata = ExifExtractor.extract_all_metadata(image_path)
        ExifExtractor.print_metadata_summary(metadata)
        
        # Detect drone model
        if metadata.get('camera'):
            drone_model = ExifExtractor.detect_drone_model(metadata['camera'])
            print(f"Detected Drone Model: {drone_model}")
    else:
        print("Usage: python exif_extractor.py <image_path>")
