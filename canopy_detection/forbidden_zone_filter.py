"""
MangroVision forbidden-zone loader and point filter.
"""

import json
from pathlib import Path

from shapely.geometry import Point, shape


# Approximate degrees per meter at a typical latitude. 1 degree of latitude is
# ~111,320 m anywhere; 1 degree of longitude shrinks toward the poles but for
# the project's mangrove sites near the equator the difference is < 2 percent,
# which is well inside the projection-error margin this buffer is meant to
# absorb. A degrees-per-meter approximation is used so the geometry stays in
# the GeoJSON's native CRS84 lat/lon space without a UTM round-trip.
_DEG_PER_METER = 1.0 / 111_320.0


class ForbiddenZoneFilter:
    """Filter planting locations against exclusion polygons.

    A non-zero ``safety_buffer_m`` expands every loaded polygon by that many
    meters before the contains check. This absorbs residual projection error
    from low-confidence SIFT matches (drone-pixel-to-GPS conversion can be
    1-3 m off when the homography RANSAC inlier ratio is low), and also
    enforces a sensible minimum distance from man-made structures.
    """

    def __init__(self, geojson_path: str, safety_buffer_m: float = 0.0):
        self.geojson_path = Path(geojson_path)
        self.safety_buffer_m = float(max(0.0, safety_buffer_m))
        self.forbidden_polygons = []
        self.buffered_polygons = []
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

            buffer_deg = self.safety_buffer_m * _DEG_PER_METER
            for feature in geojson_data['features']:
                geometry = feature.get('geometry') or {}
                if geometry.get('type') in ['Polygon', 'MultiPolygon']:
                    poly = shape(geometry)
                    if poly.is_valid:
                        self.forbidden_polygons.append(poly)
                        if buffer_deg > 0:
                            try:
                                self.buffered_polygons.append(poly.buffer(buffer_deg))
                            except Exception:
                                # Fall back to the unbuffered polygon if the
                                # geometry can't be inflated (degenerate ring,
                                # self-intersection, etc.).
                                self.buffered_polygons.append(poly)
                        else:
                            self.buffered_polygons.append(poly)

            self.zone_count = len(self.forbidden_polygons)
            buffer_note = (
                f" (with {self.safety_buffer_m:.1f} m safety buffer)"
                if self.safety_buffer_m > 0
                else ""
            )
            print(
                f"Loaded {self.zone_count} forbidden zones from "
                f"{self.geojson_path.name}{buffer_note}"
            )
        except Exception as e:
            print(f"Error loading forbidden zones: {e}")
            print("   All locations will be marked as safe.")

    def is_safe_location(self, latitude: float, longitude: float) -> bool:
        """Return `True` when the point is outside every (buffered) exclusion polygon."""
        if not self.buffered_polygons:
            return True

        point = Point(longitude, latitude)
        return not any(zone.contains(point) for zone in self.buffered_polygons)
