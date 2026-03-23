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
            'focal_length_mm': 24,
            'image_width_px': 5280,
            'image_height_px': 3956,
        },
        'DJI_AIR_2S': {
            'sensor_width_mm': 13.2,
            'sensor_height_mm': 8.8,
            'focal_length_mm': 22,
            'image_width_px': 5472,
            'image_height_px': 3648,
        },
        'DJI_PHANTOM_4': {
            'sensor_width_mm': 13.2,
            'sensor_height_mm': 8.8,
            'focal_length_mm': 24,
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
