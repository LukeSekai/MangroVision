"""
Ground Sample Distance (GSD) Calculator for Drone Imagery
Calculates accurate real-world distances from image pixels
"""

import math

class GSDCalculator:
    """
    Calculate Ground Sample Distance for accurate real-world measurements
    """
    
    # Common drone camera specifications
    COMMON_DRONES = {
        'DJI_MINI_3': {
            'sensor_width_mm': 6.3,  # 1/1.3" sensor
            'sensor_height_mm': 4.7,
            'focal_length_mm': 6.7,
            'image_width_px': 4000,
            'image_height_px': 3000
        },
        'DJI_MAVIC_3': {
            'sensor_width_mm': 17.3,  # 4/3 CMOS
            'sensor_height_mm': 13.0,
            'focal_length_mm': 24,
            'image_width_px': 5280,
            'image_height_px': 3956
        },
        'DJI_AIR_2S': {
            'sensor_width_mm': 13.2,  # 1" sensor
            'sensor_height_mm': 8.8,
            'focal_length_mm': 22,
            'image_width_px': 5472,
            'image_height_px': 3648
        },
        'DJI_PHANTOM_4': {
            'sensor_width_mm': 13.2,  # 1" sensor
            'sensor_height_mm': 8.8,
            'focal_length_mm': 24,
            'image_width_px': 5472,
            'image_height_px': 3648
        },
        'GENERIC_4K': {
            'sensor_width_mm': 6.3,
            'sensor_height_mm': 4.7,
            'focal_length_mm': 4.5,
            'image_width_px': 3840,
            'image_height_px': 2160
        }
    }
    
    @staticmethod
    def calculate_gsd(altitude_m, sensor_width_mm, focal_length_mm, image_width_px):
        """
        Calculate Ground Sample Distance (GSD) - how many meters per pixel
        
        Formula: GSD = (altitude × sensor_width) / (focal_length × image_width)
        
        Args:
            altitude_m: Flight altitude in meters
            sensor_width_mm: Camera sensor width in millimeters
            focal_length_mm: Lens focal length in millimeters
            image_width_px: Image width in pixels
            
        Returns:
            GSD in meters per pixel
        """
        # Convert sensor width to meters
        sensor_width_m = sensor_width_mm / 1000.0
        # Convert focal length to meters
        focal_length_m = focal_length_mm / 1000.0
        
        # Calculate GSD
        gsd_m_per_pixel = (altitude_m * sensor_width_m) / (focal_length_m * image_width_px)
        
        return gsd_m_per_pixel
    
    @staticmethod
    def calculate_gsd_from_drone(altitude_m, drone_model='GENERIC_4K'):
        """
        Calculate GSD using predefined drone specifications
        """
        if drone_model not in GSDCalculator.COMMON_DRONES:
            print(f"Warning: Unknown drone model '{drone_model}', using GENERIC_4K")
            drone_model = 'GENERIC_4K'
        
        specs = GSDCalculator.COMMON_DRONES[drone_model]
        
        gsd = GSDCalculator.calculate_gsd(
            altitude_m=altitude_m,
            sensor_width_mm=specs['sensor_width_mm'],
            focal_length_mm=specs['focal_length_mm'],
            image_width_px=specs['image_width_px']
        )
        
        return gsd, specs
    
    @staticmethod
    def meters_to_pixels(distance_m, gsd_m_per_pixel):
        """Convert real-world meters to image pixels"""
        return distance_m / gsd_m_per_pixel
    
    @staticmethod
    def pixels_to_meters(pixels, gsd_m_per_pixel):
        """Convert image pixels to real-world meters"""
        return pixels * gsd_m_per_pixel
    
    @staticmethod
    def calculate_coverage(altitude_m, sensor_width_mm, sensor_height_mm, 
                          focal_length_mm, image_width_px, image_height_px):
        """
        Calculate the ground coverage area of the image
        
        Returns:
            dict with width_m, height_m, area_m2
        """
        gsd = GSDCalculator.calculate_gsd(
            altitude_m, sensor_width_mm, focal_length_mm, image_width_px
        )
        
        width_m = gsd * image_width_px
        height_m = gsd * image_height_px
        area_m2 = width_m * height_m
        
        return {
            'width_m': width_m,
            'height_m': height_m,
            'area_m2': area_m2,
            'gsd_cm_per_pixel': gsd * 100
        }
    
    @staticmethod
    def estimate_from_image_and_altitude(image_width_px, image_height_px, altitude_m):
        """
        Estimate GSD when camera specs are unknown
        Uses typical consumer drone values
        """
        # Typical values for consumer drones
        typical_sensor_width_mm = 6.3  # 1/2.3" sensor
        typical_focal_length_mm = 4.5
        
        gsd = GSDCalculator.calculate_gsd(
            altitude_m=altitude_m,
            sensor_width_mm=typical_sensor_width_mm,
            focal_length_mm=typical_focal_length_mm,
            image_width_px=image_width_px
        )
        
        print(f"⚠️ Using estimated GSD (no camera specs provided)")
        print(f"   Assumed: {typical_sensor_width_mm}mm sensor, {typical_focal_length_mm}mm lens")
        
        return gsd


def interactive_gsd_calculator():
    """
    Interactive calculator to help users determine their GSD
    """
    print("\n" + "="*70)
    print("GROUND SAMPLE DISTANCE (GSD) CALCULATOR")
    print("="*70 + "\n")
    
    print("Choose input method:")
    print("1. Known drone model (DJI, etc.)")
    print("2. Manual camera specifications")
    print("3. Auto-estimate from image + altitude")
    
    choice = input("\nEnter choice (1-3): ").strip()
    
    altitude = float(input("Enter flight altitude (meters): "))
    
    if choice == '1':
        print("\nAvailable drone models:")
        for i, model in enumerate(GSDCalculator.COMMON_DRONES.keys(), 1):
            print(f"  {i}. {model}")
        
        model_name = input("\nEnter model name or number: ").strip()
        if model_name.isdigit():
            model_name = list(GSDCalculator.COMMON_DRONES.keys())[int(model_name)-1]
        
        gsd, specs = GSDCalculator.calculate_gsd_from_drone(altitude, model_name)
        
        print(f"\n{'='*70}")
        print(f"Results for {model_name} at {altitude}m altitude:")
        print(f"{'='*70}")
        print(f"Ground Sample Distance: {gsd:.6f} m/pixel ({gsd*100:.3f} cm/pixel)")
        print(f"\nCamera Specs:")
        print(f"  Sensor: {specs['sensor_width_mm']}mm × {specs['sensor_height_mm']}mm")
        print(f"  Focal Length: {specs['focal_length_mm']}mm")
        print(f"  Resolution: {specs['image_width_px']}×{specs['image_height_px']}px")
        
        coverage = GSDCalculator.calculate_coverage(
            altitude, specs['sensor_width_mm'], specs['sensor_height_mm'],
            specs['focal_length_mm'], specs['image_width_px'], specs['image_height_px']
        )
        print(f"\nGround Coverage:")
        print(f"  Width: {coverage['width_m']:.2f}m")
        print(f"  Height: {coverage['height_m']:.2f}m")
        print(f"  Area: {coverage['area_m2']:.2f} m²")
        
    elif choice == '2':
        sensor_width = float(input("Sensor width (mm): "))
        focal_length = float(input("Focal length (mm): "))
        image_width = int(input("Image width (pixels): "))
        
        gsd = GSDCalculator.calculate_gsd(altitude, sensor_width, focal_length, image_width)
        
        print(f"\n{'='*70}")
        print(f"Ground Sample Distance: {gsd:.6f} m/pixel ({gsd*100:.3f} cm/pixel)")
        print(f"{'='*70}")
        
    else:
        image_width = int(input("Image width (pixels): "))
        image_height = int(input("Image height (pixels): "))
        
        gsd = GSDCalculator.estimate_from_image_and_altitude(image_width, image_height, altitude)
        
        print(f"\n{'='*70}")
        print(f"Estimated GSD: {gsd:.6f} m/pixel ({gsd*100:.3f} cm/pixel)")
        print(f"{'='*70}")
    
    print(f"\nAt 1 meter distance:")
    print(f"  1 meter = {GSDCalculator.meters_to_pixels(1.0, gsd):.1f} pixels")
    print(f"\nUse this GSD value in your config.py!")
    
    return gsd


if __name__ == "__main__":
    interactive_gsd_calculator()
