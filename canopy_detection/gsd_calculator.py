"""
Ground Sample Distance (GSD) calculator for drone imagery.
"""


class GSDCalculator:
    """Calculate image scale from known drone camera specifications."""

    COMMON_DRONES = {
        'DJI_MINI_3': {
            'sensor_width_mm': 6.3,
            'sensor_height_mm': 4.7,
            'focal_length_mm': 6.7,
            'image_width_px': 4000,
            'image_height_px': 3000,
        },
        'DJI_MAVIC_3': {
            'sensor_width_mm': 17.3,
            'sensor_height_mm': 13.0,
            # Physical focal length for the wide Hasselblad camera (24 mm format equivalent).
            'focal_length_mm': 12.29,
            'image_width_px': 5280,
            'image_height_px': 3956,
        },
        'DJI_AIR_2S': {
            'sensor_width_mm': 13.2,
            'sensor_height_mm': 8.8,
            # Physical focal length for the fixed wide camera (22 mm format equivalent).
            'focal_length_mm': 8.4,
            'image_width_px': 5472,
            'image_height_px': 3648,
        },
        'DJI_PHANTOM_4': {
            'sensor_width_mm': 13.2,
            'sensor_height_mm': 8.8,
            # Physical focal length for the 1-inch camera (24 mm format equivalent).
            'focal_length_mm': 8.8,
            'image_width_px': 5472,
            'image_height_px': 3648,
        },
        'GENERIC_4K': {
            'sensor_width_mm': 6.3,
            'sensor_height_mm': 4.7,
            'focal_length_mm': 4.5,
            'image_width_px': 3840,
            'image_height_px': 2160,
        },
        # DJI FC7703 4K camera. The camera reports a 4 mm physical focal
        # length and 24 mm 35 mm-equivalent focal length on 4000px-wide stills.
        'DJI_FC7703': {
            'sensor_width_mm': 6.3,
            'sensor_height_mm': 3.54,
            'focal_length_mm': 4.0,
            'image_width_px': 4000,
            'image_height_px': 2250,
        },
    }

    @staticmethod
    def calculate_gsd(altitude_m, sensor_width_mm, focal_length_mm, image_width_px):
        """Return meters per pixel for the given camera geometry."""
        sensor_width_m = sensor_width_mm / 1000.0
        focal_length_m = focal_length_mm / 1000.0
        return (altitude_m * sensor_width_m) / (focal_length_m * image_width_px)

    @staticmethod
    def calculate_gsd_from_drone(altitude_m, drone_model='GENERIC_4K'):
        """Return `(gsd, specs)` for a known drone model."""
        if drone_model not in GSDCalculator.COMMON_DRONES:
            print(f"Warning: Unknown drone model '{drone_model}', using GENERIC_4K")
            drone_model = 'GENERIC_4K'

        specs = GSDCalculator.COMMON_DRONES[drone_model]
        gsd = GSDCalculator.calculate_gsd(
            altitude_m=altitude_m,
            sensor_width_mm=specs['sensor_width_mm'],
            focal_length_mm=specs['focal_length_mm'],
            image_width_px=specs['image_width_px'],
        )
        return gsd, specs

    @staticmethod
    def calculate_gsd_from_metadata(
        altitude_m,
        camera_info=None,
        drone_model='GENERIC_4K',
        image_width_px=None,
        image_height_px=None,
    ):
        """Return `(gsd, specs)` using EXIF camera geometry when available.

        EXIF focal length plus 35 mm-equivalent focal length lets us estimate
        the active sensor width for the actual still frame. If those tags are
        missing, fall back to the known drone preset.
        """
        camera_info = camera_info or {}
        focal_mm = camera_info.get('focal_length_mm')
        focal_35mm = camera_info.get('focal_length_35mm')
        width_px = image_width_px or camera_info.get('image_width')
        height_px = image_height_px or camera_info.get('image_height')

        try:
            focal_mm = float(focal_mm)
            focal_35mm = float(focal_35mm)
            width_px = int(width_px)
            height_px = int(height_px) if height_px else None
        except (TypeError, ValueError):
            focal_mm = None
            focal_35mm = None
            width_px = None
            height_px = None

        if focal_mm and focal_35mm and width_px:
            # 35 mm-equivalent focal length is based on the 35 mm diagonal.
            # Convert that crop factor to active sensor width for the image's
            # aspect ratio; this handles 16:9 DJI stills better than a fixed
            # 4:3 sensor-width preset.
            if height_px:
                aspect_scale = width_px / ((width_px ** 2 + height_px ** 2) ** 0.5)
            else:
                aspect_scale = 36.0 / ((36.0 ** 2 + 24.0 ** 2) ** 0.5)
            sensor_diagonal_mm = 43.2666153 * focal_mm / focal_35mm
            sensor_width_mm = sensor_diagonal_mm * aspect_scale
            gsd = GSDCalculator.calculate_gsd(
                altitude_m=altitude_m,
                sensor_width_mm=sensor_width_mm,
                focal_length_mm=focal_mm,
                image_width_px=width_px,
            )
            return gsd, {
                'sensor_width_mm': sensor_width_mm,
                'focal_length_mm': focal_mm,
                'focal_length_35mm': focal_35mm,
                'image_width_px': width_px,
                'image_height_px': height_px,
                'source': 'exif_focal_35mm',
            }

        gsd, specs = GSDCalculator.calculate_gsd_from_drone(altitude_m, drone_model)
        specs = dict(specs)
        specs['source'] = f'drone_preset:{drone_model}'
        return gsd, specs
