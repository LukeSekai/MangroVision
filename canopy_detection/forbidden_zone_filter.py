"""
MangroVision forbidden-zone loader and point filter.
"""

import json
from pathlib import Path

from shapely.geometry import Point, shape


class ForbiddenZoneFilter:
    """Filter planting locations against exclusion polygons."""

    def __init__(self, geojson_path: str):
        self.geojson_path = Path(geojson_path)
        self.forbidden_polygons = []
        self.zone_count = 0

        if not self.geojson_path.exists():
            print(f"Warning: Forbidden zones file not found: {geojson_path}")
            print("   All locations will be marked as safe.")
            return

        try:
            with open(self.geojson_path, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)

            if 'features' not in geojson_data:
                print(f"Warning: No features found in {geojson_path}")
                return

            for feature in geojson_data['features']:
                geometry = feature.get('geometry') or {}
                if geometry.get('type') in ['Polygon', 'MultiPolygon']:
                    poly = shape(geometry)
                    if poly.is_valid:
                        self.forbidden_polygons.append(poly)

            self.zone_count = len(self.forbidden_polygons)
            print(f"Loaded {self.zone_count} forbidden zones from {self.geojson_path.name}")
        except Exception as e:
            print(f"Error loading forbidden zones: {e}")
            print("   All locations will be marked as safe.")

    def is_safe_location(self, latitude: float, longitude: float) -> bool:
        """Return `True` when the point is outside every exclusion polygon."""
        if not self.forbidden_polygons:
            return True

        point = Point(longitude, latitude)
        return not any(zone.contains(point) for zone in self.forbidden_polygons)
