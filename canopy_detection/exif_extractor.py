"""
EXIF metadata extraction for drone images.
"""

import os
import re
from typing import Dict, Optional

from PIL import Image
from PIL.ExifTags import GPSTAGS, TAGS


class ExifExtractor:
    """Extract GPS and camera metadata from image EXIF tags."""

    _DJI_XMP_RE = re.compile(rb'(?:drone-dji|dji):([A-Za-z0-9_]+)="([^"]*)"')

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
                exif_ifd = exif.get_ifd(0x8769)
                if exif_ifd:
                    for tag_id, value in exif_ifd.items():
                        tag_name = TAGS.get(tag_id, tag_id)
                        exif_data[tag_name] = value
            except (KeyError, AttributeError):
                pass

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
    def _clean_text(value) -> str:
        """Normalize EXIF ASCII strings that DJI pads with NUL bytes."""
        if value is None:
            return ''
        if isinstance(value, bytes):
            value = value.decode('utf-8', errors='ignore')
        return str(value).replace('\x00', '').strip()

    @staticmethod
    def _as_float(value) -> Optional[float]:
        """Return a numeric EXIF/XMP value as float when possible."""
        if value is None:
            return None
        try:
            if isinstance(value, tuple):
                return float(value[0]) / float(value[1]) if value[1] != 0 else None
            return float(str(value).strip())
        except (TypeError, ValueError, ZeroDivisionError):
            return None

    @staticmethod
    def get_xmp_dji_data(image_path: str) -> Dict:
        """Extract DJI XMP attributes embedded in JPEG APP1 metadata."""
        try:
            with open(image_path, 'rb') as image_file:
                payload = image_file.read()
        except OSError:
            return {}

        xmp_data = {}
        for key, raw_value in ExifExtractor._DJI_XMP_RE.findall(payload):
            name = key.decode('utf-8', errors='ignore')
            text_value = raw_value.decode('utf-8', errors='ignore').strip()
            numeric_value = ExifExtractor._as_float(text_value)
            xmp_data[name] = numeric_value if numeric_value is not None else text_value
        return xmp_data

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
            parsed_heading = ExifExtractor._as_float(heading)
            if parsed_heading is not None:
                gps_info['heading'] = parsed_heading % 360.0
                gps_info['heading_source'] = 'EXIF GPSImgDirection'

        return gps_info or None

    @staticmethod
    def get_camera_info(exif_data: Dict) -> Dict:
        """Extract camera metadata from EXIF tags."""
        camera_info = {}

        if 'Make' in exif_data:
            camera_info['make'] = ExifExtractor._clean_text(exif_data['Make'])
        if 'Model' in exif_data:
            camera_info['model'] = ExifExtractor._clean_text(exif_data['Model'])
        if 'ExifImageWidth' in exif_data:
            camera_info['image_width'] = exif_data['ExifImageWidth']
        if 'ExifImageHeight' in exif_data:
            camera_info['image_height'] = exif_data['ExifImageHeight']

        if 'FocalLength' in exif_data:
            focal = ExifExtractor._as_float(exif_data['FocalLength'])
            if focal is not None:
                camera_info['focal_length_mm'] = focal
        if 'FocalLengthIn35mmFilm' in exif_data:
            focal_35 = ExifExtractor._as_float(exif_data['FocalLengthIn35mmFilm'])
            if focal_35 is not None:
                camera_info['focal_length_35mm'] = focal_35

        if 'ISOSpeedRatings' in exif_data:
            camera_info['iso'] = exif_data['ISOSpeedRatings']
        if 'FNumber' in exif_data:
            aperture = ExifExtractor._as_float(exif_data['FNumber'])
            if aperture is not None:
                camera_info['aperture'] = aperture
        if 'ExposureTime' in exif_data:
            exposure_value = ExifExtractor._as_float(exif_data['ExposureTime'])
            if exposure_value and exposure_value > 0:
                reciprocal = round(1.0 / exposure_value)
                camera_info['shutter_speed'] = f"1/{reciprocal}" if reciprocal > 1 else exposure_value

        if 'DateTime' in exif_data:
            camera_info['datetime'] = exif_data['DateTime']

        return camera_info

    @staticmethod
    def extract_all_metadata(image_path: str) -> Dict:
        """Return the normalized metadata payload used by the app."""
        exif_data = ExifExtractor.get_exif_data(image_path)
        xmp_data = ExifExtractor.get_xmp_dji_data(image_path)
        if not exif_data:
            return {
                'has_exif': False,
                'has_gps': False,
                'xmp': xmp_data,
                'error': 'No EXIF data found in image',
            }

        gps_info = ExifExtractor.get_gps_info(exif_data)
        camera_info = ExifExtractor.get_camera_info(exif_data)

        if gps_info is None:
            gps_info = {}

        rel_altitude = ExifExtractor._as_float(xmp_data.get('RelativeAltitude'))
        abs_altitude = ExifExtractor._as_float(xmp_data.get('AbsoluteAltitude'))
        if rel_altitude is not None:
            gps_info['relative_altitude'] = rel_altitude
        if abs_altitude is not None:
            gps_info['absolute_altitude'] = abs_altitude

        flight_yaw = ExifExtractor._as_float(xmp_data.get('FlightYawDegree'))
        gimbal_yaw = ExifExtractor._as_float(xmp_data.get('GimbalYawDegree'))
        if gps_info.get('heading') is None:
            if flight_yaw is not None:
                gps_info['heading'] = (flight_yaw + (gimbal_yaw or 0.0)) % 360.0
                gps_info['heading_source'] = 'DJI XMP FlightYawDegree'
            elif gimbal_yaw is not None:
                gps_info['heading'] = gimbal_yaw % 360.0
                gps_info['heading_source'] = 'DJI XMP GimbalYawDegree'

        for xmp_key, camera_key in (
            ('GimbalPitchDegree', 'gimbal_pitch_degree'),
            ('GimbalRollDegree', 'gimbal_roll_degree'),
            ('FlightPitchDegree', 'flight_pitch_degree'),
            ('FlightRollDegree', 'flight_roll_degree'),
        ):
            numeric_value = ExifExtractor._as_float(xmp_data.get(xmp_key))
            if numeric_value is not None:
                camera_info[camera_key] = numeric_value

        image = Image.open(image_path)
        width, height = image.size

        has_gps_coordinates = (
            gps_info.get('latitude') is not None
            and gps_info.get('longitude') is not None
        )

        return {
            'has_exif': True,
            'has_gps': has_gps_coordinates,
            'gps': gps_info or None,
            'camera': camera_info,
            'image_width': camera_info.get('image_width', width),
            'image_height': camera_info.get('image_height', height),
            'xmp': xmp_data,
            'file_path': image_path,
            'file_size_mb': os.path.getsize(image_path) / (1024 * 1024),
        }

    @staticmethod
    def detect_drone_model(camera_info: Dict) -> str:
        """Map camera make/model strings to known drone presets."""
        make = ExifExtractor._clean_text(camera_info.get('make', '')).upper()
        model = ExifExtractor._clean_text(camera_info.get('model', '')).upper()

        if 'DJI' in make or 'DJI' in model:
            if 'FC7703' in model:
                return 'DJI_FC7703'
            if 'MINI 3' in model or 'MINI3' in model:
                return 'DJI_MINI_3'
            if 'MAVIC 3' in model or 'MAVIC3' in model:
                return 'DJI_MAVIC_3'
            if 'AIR 2S' in model or 'AIR2S' in model:
                return 'DJI_AIR_2S'
            if 'PHANTOM 4' in model or 'PHANTOM4' in model:
                return 'DJI_PHANTOM_4'

        return 'GENERIC_4K'
