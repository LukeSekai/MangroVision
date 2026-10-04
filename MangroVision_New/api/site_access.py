"""Mapped access roads shared by the organization plots within a physical site."""
import json
import math
from functools import lru_cache
from pathlib import Path


def distance_m(a, b):
    lat1, lon1, lat2, lon2 = map(math.radians, [*a, *b])
    hav = math.sin((lat2-lat1)/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin((lon2-lon1)/2)**2
    return 6371000 * 2 * math.asin(math.sqrt(min(1, hav)))


def path_distance(path):
    return sum(distance_m(a, b) for a, b in zip(path, path[1:]))


def inside_area(point, polygon):
    """Ray crossing with coordinates consistently in latitude/longitude order."""
    x, y = point
    inside = False
    for a, b in zip(polygon, polygon[1:]):
        if (a[1] > y) != (b[1] > y) and x < (b[0]-a[0])*(y-a[1])/(b[1]-a[1])+a[0]:
            inside = not inside
    return inside


def segment_projection(point, a, b):
    """Closest position on a short geographic segment, in [lat, lon]."""
    scale = math.cos(math.radians(point[0]))
    dx, dy = b[0]-a[0], (b[1]-a[1])*scale
    length_sq = dx*dx + dy*dy
    t = max(0, min(1, ((point[0]-a[0])*dx + (point[1]-a[1])*scale*dy)/length_sq)) if length_sq else 0
    return [a[0]+t*(b[0]-a[0]), a[1]+t*(b[1]-a[1])]


def inside_or_near_area(point, polygon, tolerance_m=3):
    """Include the traced boundary and a small, explicit measurement tolerance.

    Planting coordinates and the manually traced physical-site outline do not
    coincide exactly. This classifies nearby positions without drawing a path
    or snapping their actual coordinates across the boundary.
    """
    if inside_area(point, polygon):
        return True
    return any(distance_m(point, segment_projection(point, a, b)) <= tolerance_m
               for a, b in zip(polygon, [*polygon[1:], polygon[0]]))


@lru_cache(maxsize=1)
def access_routes():
    return json.loads((Path(__file__).parent / 'data/site_access_routes.json').read_text(encoding='utf-8'))


def access_for_destination(destination):
    return next((site for site in access_routes() if inside_or_near_area(destination, site['site_area'])), None)


def remaining_access_path(origin, path, tolerance_m=12):
    """Join at the closest road segment when GPS is already on the access road.

    The small tolerance accommodates phone GPS and tracing error; it must not
    snap a position across a pond to skip the public-road route.
    """
    best = None
    for index, (a, b) in enumerate(zip(path, path[1:])):
        projection = segment_projection(origin, a, b)
        gap = distance_m(origin, projection)
        if best is None or gap < best[0]:
            best = (gap, index, projection)
    if best is None or best[0] > tolerance_m:
        return None
    return [origin, best[2], *path[best[1]+1:]]
