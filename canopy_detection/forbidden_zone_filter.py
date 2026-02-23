"""
MangroVision - Forbidden Zone Filter
Filters out planting locations that fall within forbidden zones
(bridges, buildings, towers, roads, etc.)
"""

import json
from shapely.geometry import shape, Point, Polygon
from typing import List, Dict, Tuple
from pathlib import Path


class ForbiddenZoneFilter:
    """Filter planting locations based on forbidden zone polygons"""
    
    def __init__(self, geojson_path: str):
        """
        Load forbidden zones from GeoJSON file
        
        Args:
            geojson_path: Path to the forbidden_zones.geojson file
        """
        self.geojson_path = Path(geojson_path)
        self.forbidden_polygons = []
        self.zone_count = 0
        
        if not self.geojson_path.exists():
            print(f"⚠️ Warning: Forbidden zones file not found: {geojson_path}")
            print(f"   All locations will be marked as safe.")
            return
        
        try:
            with open(self.geojson_path, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
            
            # Convert GeoJSON features to Shapely polygons
            if 'features' in geojson_data:
                for feature in geojson_data['features']:
                    if feature.get('geometry') and feature['geometry'].get('type') in ['Polygon', 'MultiPolygon']:
                        poly = shape(feature['geometry'])
                        if poly.is_valid:
                            self.forbidden_polygons.append(poly)
                
                self.zone_count = len(self.forbidden_polygons)
                print(f"✅ Loaded {self.zone_count} forbidden zones from {self.geojson_path.name}")
            else:
                print(f"⚠️ Warning: No features found in {geojson_path}")
                
        except Exception as e:
            print(f"❌ Error loading forbidden zones: {e}")
            print(f"   All locations will be marked as safe.")
    
    def is_safe_location(self, latitude: float, longitude: float) -> bool:
        """
        Check if a coordinate is NOT in a forbidden zone
        
        Args:
            latitude: GPS latitude coordinate
            longitude: GPS longitude coordinate
            
        Returns:
            True if safe to plant (not in forbidden zone), False if forbidden
        """
        # If no forbidden zones loaded, all locations are safe
        if not self.forbidden_polygons:
            return True
        
        # Create point (Note: Shapely uses (lon, lat) order for geographic coordinates)
        point = Point(longitude, latitude)
        
        # Check if point is inside any forbidden zone
        for zone in self.forbidden_polygons:
            if zone.contains(point):
                return False  # Point is inside a forbidden zone
        
        return True  # Point is safe
    
    def filter_gaps(self, gap_list: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter a list of detected planting gaps
        
        Args:
            gap_list: List of gap dictionaries with 'lat' and 'lon' keys
                     Example: [{'lat': 14.5995, 'lon': 120.9842, ...}, ...]
        
        Returns:
            Tuple of (safe_gaps, filtered_gaps)
            - safe_gaps: Gaps that are safe to plant
            - filtered_gaps: Gaps that are in forbidden zones
        """
        safe_gaps = []
        filtered_gaps = []
        
        for gap in gap_list:
            lat = gap.get('lat')
            lon = gap.get('lon')
            
            if lat is None or lon is None:
                print(f"⚠️ Warning: Gap missing GPS coordinates: {gap}")
                continue
            
            if self.is_safe_location(lat, lon):
                safe_gaps.append(gap)
            else:
                filtered_gaps.append(gap)
                print(f"🚫 Filtered out gap at ({lat:.6f}, {lon:.6f}) - in forbidden zone")
        
        return safe_gaps, filtered_gaps
    
    def filter_hexagons_with_gps(self, hexagons: List[Dict], 
                                  pixel_to_gps_func) -> Tuple[List[Dict], List[Dict]]:
        """
        Filter hexagons by converting their pixel coordinates to GPS and checking forbidden zones
        
        Args:
            hexagons: List of hexagon dictionaries with 'center' (pixel coords)
            pixel_to_gps_func: Function that converts (px, py) to (lat, lon)
                              Example: lambda px, py: drone_pixel_to_gps_via_heading(...)
        
        Returns:
            Tuple of (safe_hexagons, forbidden_hexagons)
        """
        safe_hexagons = []
        forbidden_hexagons = []
        
        for i, hexagon in enumerate(hexagons):
            px, py = hexagon['center']
            
            # Convert pixel coordinates to GPS
            try:
                lat, lon = pixel_to_gps_func(px, py)
                
                # Store GPS coordinates in hexagon dict for later use
                hexagon['gps'] = {'lat': lat, 'lon': lon}
                
                # Check if location is safe
                if self.is_safe_location(lat, lon):
                    safe_hexagons.append(hexagon)
                else:
                    forbidden_hexagons.append(hexagon)
                    print(f"🚫 Filtered hexagon #{i+1} at ({lat:.6f}, {lon:.6f}) - in forbidden zone")
                    
            except Exception as e:
                print(f"⚠️ Error converting hexagon #{i+1} coordinates: {e}")
                # If conversion fails, include by default (safe mode)
                safe_hexagons.append(hexagon)
        
        return safe_hexagons, forbidden_hexagons
    
    def get_statistics(self) -> Dict:
        """
        Get statistics about loaded forbidden zones
        
        Returns:
            Dictionary with zone statistics
        """
        if not self.forbidden_polygons:
            return {
                'zone_count': 0,
                'total_area_m2': 0,
                'file_loaded': False
            }
        
        total_area = sum(poly.area for poly in self.forbidden_polygons)
        
        return {
            'zone_count': self.zone_count,
            'total_area_deg2': total_area,  # Area in square degrees (not accurate without projection)
            'file_loaded': True,
            'file_path': str(self.geojson_path)
        }
