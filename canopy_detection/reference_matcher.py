"""
Reference point matching for automatic rotation detection.
User clicks same landmarks in image and map, system calculates rotation.
"""

import numpy as np
from typing import List, Tuple, Optional
import math


class ReferencePointMatcher:
    """Calculate rotation angle from matched reference points."""
    
    def __init__(self):
        self.image_points = []  # (x, y) in pixels
        self.map_points = []    # (lat, lon) in degrees
    
    def add_point_pair(self, image_xy: Tuple[float, float], map_latlon: Tuple[float, float]):
        """
        Add a matched pair of points.
        
        Args:
            image_xy: (x, y) pixel coordinates in image
            map_latlon: (lat, lon) GPS coordinates on map
        """
        self.image_points.append(image_xy)
        self.map_points.append(map_latlon)
    
    def calculate_rotation(self, 
                          image_center_px: Tuple[float, float],
                          map_center_latlon: Tuple[float, float],
                          gsd: float,
                          meters_per_degree_lat: float,
                          meters_per_degree_lon: float) -> Optional[float]:
        """
        Calculate rotation angle from matched points.
        
        Args:
            image_center_px: Center of image in pixels (width/2, height/2)
            map_center_latlon: GPS coordinates of image center (lat, lon)
            gsd: Ground Sample Distance in meters/pixel
            meters_per_degree_lat: Meters per degree latitude
            meters_per_degree_lon: Meters per degree longitude
            
        Returns:
            Rotation angle in degrees (0-360) or None if insufficient points
        """
        if len(self.image_points) < 2:
            print("Need at least 2 matched points")
            return None
        
        # Convert image points to offsets from center (in meters)
        image_offsets_m = []
        for px, py in self.image_points:
            offset_x_px = px - image_center_px[0]
            offset_y_px = py - image_center_px[1]
            offset_x_m = offset_x_px * gsd
            offset_y_m = -offset_y_px * gsd  # Y-flip
            image_offsets_m.append((offset_x_m, offset_y_m))
        
        # Convert map points to offsets from center (in meters)
        map_offsets_m = []
        center_lat, center_lon = map_center_latlon
        for lat, lon in self.map_points:
            delta_lat = lat - center_lat
            delta_lon = lon - center_lon
            offset_y_m = delta_lat * meters_per_degree_lat
            offset_x_m = delta_lon * meters_per_degree_lon
            map_offsets_m.append((offset_x_m, offset_y_m))
        
        # Calculate rotation angles for each point pair
        angles = []
        for i in range(len(image_offsets_m)):
            img_x, img_y = image_offsets_m[i]
            map_x, map_y = map_offsets_m[i]
            
            # Angle of image point from center
            img_angle = math.atan2(img_y, img_x)
            
            # Angle of map point from center
            map_angle = math.atan2(map_y, map_x)
            
            # Rotation needed to align image to map
            rotation_rad = map_angle - img_angle
            rotation_deg = math.degrees(rotation_rad)
            
            # Normalize to 0-360
            rotation_deg = rotation_deg % 360
            
            angles.append(rotation_deg)
            print(f"  Point {i+1}: Image angle={math.degrees(img_angle):.1f}°, Map angle={math.degrees(map_angle):.1f}°, Rotation={rotation_deg:.1f}°")
        
        # Use median to avoid outliers
        angles.sort()
        median_angle = angles[len(angles) // 2]
        
        print(f"\n✓ Calculated rotation: {median_angle:.1f}°")
        print(f"  (from {len(angles)} point pairs)")
        
        return median_angle
    
    def clear(self):
        """Clear all matched points."""
        self.image_points = []
        self.map_points = []
    
    def get_point_count(self) -> int:
        """Get number of matched point pairs."""
        return len(self.image_points)


def calculate_rotation_from_two_points(
    point1_image: Tuple[float, float],
    point2_image: Tuple[float, float],
    point1_map: Tuple[float, float],
    point2_map: Tuple[float, float]
) -> float:
    """
    Simplified calculation using just 2 points.
    
    Args:
        point1_image: (x1, y1) in pixels
        point2_image: (x2, y2) in pixels
        point1_map: (lat1, lon1)
        point2_map: (lat2, lon2)
        
    Returns:
        Rotation angle in degrees
    """
    # Vector in image space
    img_dx = point2_image[0] - point1_image[0]
    img_dy = point2_image[1] - point1_image[1]
    img_angle = math.atan2(-img_dy, img_dx)  # -dy because y-flip
    
    # Vector in map space (approximate - assumes small area)
    map_dx = point2_map[1] - point1_map[1]  # longitude
    map_dy = point2_map[0] - point1_map[0]  # latitude
    map_angle = math.atan2(map_dy, map_dx)
    
    # Rotation difference
    rotation_rad = map_angle - img_angle
    rotation_deg = math.degrees(rotation_rad)
    
    # Normalize to 0-360
    rotation_deg = rotation_deg % 360
    
    return rotation_deg


if __name__ == "__main__":
    print("Reference point matcher ready.")
    print("Usage:")
    print("  matcher = ReferencePointMatcher()")
    print("  matcher.add_point_pair((x1, y1), (lat1, lon1))")
    print("  matcher.add_point_pair((x2, y2), (lat2, lon2))")
    print("  angle = matcher.calculate_rotation(...)")
