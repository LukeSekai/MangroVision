"""
EXIF metadata extraction for drone images.
"""

import os
from typing import Dict, Optional

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS


class ExifExtractor:
    """Extract GPS and camera metadata from image EXIF tags."""

    @staticmethod
    def get_exif_data(image_path: str) -> Dict:
        """Return the raw EXIF payload plus decoded GPS tags when present."""
        try:
            image = Image.open(image_path)
            exif_data = {}
            exif = image.getexif()
            if exif is None:
                return {}

            for tag_id, value in exif.items():
                tag_name = TAGS.get(tag_id, tag_id)
                exif_data[tag_name] = value

            try:
                gps_ifd = exif.get_ifd(0x8825)
                if gps_ifd:
                    gps_data = {}
                    for tag_id, value in gps_ifd.items():
                        tag_name = GPSTAGS.get(tag_id, tag_id)
                        gps_data[tag_name] = value
                    exif_data['GPSInfo'] = gps_data
            except (KeyError, AttributeError):
                pass

            return exif_data
        except Exception as e:
            print(f"Error reading EXIF data: {e}")
            return {}

    @staticmethod
    def _convert_to_degrees(value) -> float:
        """Convert a GPS DMS tuple into decimal degrees."""
        d = float(value[0])
        m = float(value[1])
        s = float(value[2])
        return d + (m / 60.0) + (s / 3600.0)

    @staticmethod
    def get_gps_info(exif_data: Dict) -> Optional[Dict]:
        """Extract normalized GPS info from decoded EXIF data."""
        gps_parsed = exif_data.get('GPSInfo')
        if not isinstance(gps_parsed, dict):
            return None

        gps_info = {}

        if 'GPSLatitude' in gps_parsed and 'GPSLatitudeRef' in gps_parsed:
            lat = ExifExtractor._convert_to_degrees(gps_parsed['GPSLatitude'])
            if gps_parsed['GPSLatitudeRef'] == 'S':
                lat = -lat
            gps_info['latitude'] = lat

        if 'GPSLongitude' in gps_parsed and 'GPSLongitudeRef' in gps_parsed:
            lon = ExifExtractor._convert_to_degrees(gps_parsed['GPSLongitude'])
            if gps_parsed['GPSLongitudeRef'] == 'W':
                lon = -lon
            gps_info['longitude'] = lon

        if 'GPSAltitude' in gps_parsed:
            alt_value = gps_parsed['GPSAltitude']
            if isinstance(alt_value, tuple):
                altitude = float(alt_value[0]) / float(alt_value[1]) if alt_value[1] != 0 else 0.0
            else:
                altitude = float(alt_value)

            if gps_parsed.get('GPSAltitudeRef') == 1:
                altitude = -altitude
            gps_info['altitude'] = altitude

        if 'RelativeAltitude' in gps_parsed:
            rel_alt = gps_parsed['RelativeAltitude']
            if isinstance(rel_alt, tuple):
                gps_info['relative_altitude'] = float(rel_alt[0]) / float(rel_alt[1]) if rel_alt[1] != 0 else 0.0
            else:
                gps_info['relative_altitude'] = float(rel_alt)

        if 'GPSImgDirection' in gps_parsed:
            heading = gps_parsed['GPSImgDirection']
            if isinstance(heading, tuple):
                gps_info['heading'] = float(heading[0]) / float(heading[1]) if heading[1] != 0 else 0.0
            else:
                gps_info['heading'] = float(heading)

        return gps_info or None

    @staticmethod
    def get_camera_info(exif_data: Dict) -> Dict:
        """Extract camera metadata from EXIF tags."""
        camera_info = {}

        if 'Make' in exif_data:
            camera_info['make'] = exif_data['Make']
        if 'Model' in exif_data:
            camera_info['model'] = exif_data['Model']
        if 'ExifImageWidth' in exif_data:
            camera_info['image_width'] = exif_data['ExifImageWidth']
        if 'ExifImageHeight' in exif_data:
            camera_info['image_height'] = exif_data['ExifImageHeight']

        if 'FocalLength' in exif_data:
            focal = exif_data['FocalLength']
            if isinstance(focal, tuple):
                camera_info['focal_length_mm'] = focal[0] / focal[1]
            else:
                camera_info['focal_length_mm'] = float(focal)

        if 'ISOSpeedRatings' in exif_data:
            camera_info['iso'] = exif_data['ISOSpeedRatings']
        if 'FNumber' in exif_data:
            f_number = exif_data['FNumber']
            if isinstance(f_number, tuple):
                camera_info['aperture'] = f_number[0] / f_number[1]
            else:
                camera_info['aperture'] = float(f_number)
        if 'ExposureTime' in exif_data:
            exposure = exif_data['ExposureTime']
            if isinstance(exposure, tuple):
                camera_info['shutter_speed'] = f"1/{int(exposure[1] / exposure[0])}"
            else:
                camera_info['shutter_speed'] = exposure

        if 'DateTime' in exif_data:
            camera_info['datetime'] = exif_data['DateTime']

        return camera_info

    @staticmethod
    def extract_all_metadata(image_path: str) -> Dict:
        """Return the normalized metadata payload used by the app."""
        exif_data = ExifExtractor.get_exif_data(image_path)
        if not exif_data:
            return {
                'has_exif': False,
                'has_gps': False,
                'error': 'No EXIF data found in image',
            }

        gps_info = ExifExtractor.get_gps_info(exif_data)
        camera_info = ExifExtractor.get_camera_info(exif_data)

        image = Image.open(image_path)
        width, height = image.size

        return {
            'has_exif': True,
            'has_gps': gps_info is not None,
            'gps': gps_info,
            'camera': camera_info,
            'image_width': camera_info.get('image_width', width),
            'image_height': camera_info.get('image_height', height),
            'file_path': image_path,
            'file_size_mb': os.path.getsize(image_path) / (1024 * 1024),
        }

    @staticmethod
    def detect_drone_model(camera_info: Dict) -> str:
        """Map camera make/model strings to known drone presets."""
        make = camera_info.get('make', '').upper()
        model = camera_info.get('model', '').upper()

        if 'DJI' in make or 'DJI' in model:
            if 'MINI 3' in model or 'MINI3' in model:
                return 'DJI_MINI_3'
            if 'MAVIC 3' in model or 'MAVIC3' in model:
                return 'DJI_MAVIC_3'
            if 'AIR 2S' in model or 'AIR2S' in model:
                return 'DJI_AIR_2S'
            if 'PHANTOM 4' in model or 'PHANTOM4' in model:
                return 'DJI_PHANTOM_4'

        return 'GENERIC_4K'
