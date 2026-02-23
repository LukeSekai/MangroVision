"""
GPS trajectory analyzer for calculating flight direction from consecutive images.
If multiple images from same flight are available, calculate heading from GPS track.
"""

import math
from typing import List, Tuple, Optional
import numpy as np


class GPSTrajectoryAnalyzer:
    """Calculate flight direction from GPS coordinates of consecutive images."""
    
    def __init__(self):
        self.gps_points = []  # List of (lat, lon, timestamp) tuples
    
    def add_point(self, latitude: float, longitude: float, timestamp: Optional[float] = None):
        """
        Add GPS point from an image.
        
        Args:
            latitude: Latitude in degrees
            longitude: Longitude in degrees
            timestamp: Optional timestamp (for ordering if filenames don't sort chronologically)
        """
        self.gps_points.append((latitude, longitude, timestamp))
    
    def calculate_heading_between_points(self, 
                                        lat1: float, lon1: float,
                                        lat2: float, lon2: float) -> float:
        """
        Calculate bearing/heading between two GPS points.
        Uses the forward azimuth formula.
        
        Args:
            lat1, lon1: First point
            lat2, lon2: Second point
            
        Returns:
            Heading in degrees (0-360, where 0=North, 90=East, etc.)
        """
        # Convert to radians
        lat1_rad = math.radians(lat1)
        lat2_rad = math.radians(lat2)
        delta_lon_rad = math.radians(lon2 - lon1)
        
        # Calculate bearing
        x = math.sin(delta_lon_rad) * math.cos(lat2_rad)
        y = (math.cos(lat1_rad) * math.sin(lat2_rad) - 
             math.sin(lat1_rad) * math.cos(lat2_rad) * math.cos(delta_lon_rad))
        
        bearing_rad = math.atan2(x, y)
        bearing_deg = math.degrees(bearing_rad)
        
        # Normalize to 0-360
        bearing_deg = (bearing_deg + 360) % 360
        
        return bearing_deg
    
    def get_flight_heading(self) -> Optional[float]:
        """
        Calculate average flight heading from all GPS points.
        
        Returns:
            Average heading in degrees or None if insufficient points
        """
        if len(self.gps_points) < 2:
            print("Need at least 2 GPS points to calculate heading")
            return None
        
        # Sort by timestamp if available, otherwise assume order is correct
        if all(p[2] is not None for p in self.gps_points):
            sorted_points = sorted(self.gps_points, key=lambda p: p[2])
        else:
            sorted_points = self.gps_points
        
        headings = []
        for i in range(len(sorted_points) - 1):
            lat1, lon1, _ = sorted_points[i]
            lat2, lon2, _ = sorted_points[i + 1]
            
            heading = self.calculate_heading_between_points(lat1, lon1, lat2, lon2)
            headings.append(heading)
            
            print(f"  Segment {i+1}: ({lat1:.6f},{lon1:.6f}) → ({lat2:.6f},{lon2:.6f}) = {heading:.1f}°")
        
        # Calculate circular mean (important for angles!)
        # Can't just average 359° and 1° = 180°, should be 0°
        sin_sum = sum(math.sin(math.radians(h)) for h in headings)
        cos_sum = sum(math.cos(math.radians(h)) for h in headings)
        
        mean_heading_rad = math.atan2(sin_sum, cos_sum)
        mean_heading_deg = math.degrees(mean_heading_rad)
        mean_heading_deg = (mean_heading_deg + 360) % 360
        
        print(f"\n✓ Calculated flight heading: {mean_heading_deg:.1f}°")
        print(f"  (from {len(headings)} trajectory segments)")
        
        return mean_heading_deg
    
    def clear(self):
        """Clear all GPS points."""
        self.gps_points = []


def analyze_multiple_images(image_paths: List[str]) -> Optional[float]:
    """
    Analyze multiple images and calculate flight heading.
    
    Args:
        image_paths: List of paths to drone images
        
    Returns:
        Flight heading in degrees or None
    """
    from canopy_detection.exif_extractor import ExifExtractor
    
    extractor = ExifExtractor()
    analyzer = GPSTrajectoryAnalyzer()
    
    print(f"Analyzing {len(image_paths)} images for GPS trajectory...")
    
    for i, path in enumerate(image_paths):
        metadata = extractor.extract_metadata(path)
        
        if metadata.get('has_gps'):
            gps = metadata['gps']
            timestamp = metadata.get('datetime_timestamp')  # Unix timestamp if available
            
            analyzer.add_point(gps['latitude'], gps['longitude'], timestamp)
            print(f"  Image {i+1}: {gps['latitude']:.6f}, {gps['longitude']:.6f}")
        else:
            print(f"  Image {i+1}: No GPS data")
    
    return analyzer.get_flight_heading()


if __name__ == "__main__":
    print("GPS Trajectory Analyzer ready.")
    print("Usage:")
    print("  analyzer = GPSTrajectoryAnalyzer()")
    print("  analyzer.add_point(lat1, lon1)")
    print("  analyzer.add_point(lat2, lon2)")
    print("  heading = analyzer.get_flight_heading()")
