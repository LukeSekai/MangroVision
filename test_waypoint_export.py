"""
Test MangroVision Waypoint Export Module
========================================
Validates GPX, KML, and GeoJSON output for correctness.
Run:  python test_waypoint_export.py
"""

import sys, os

# Force UTF-8 on Windows terminals (cp1252 can't handle emoji)
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from waypoint_export import (
    generate_gpx, generate_kml, generate_geojson, hexagons_to_waypoints,
)
import xml.etree.ElementTree as ET
import json


# ── Sample data ──────────────────────────────────────────────────────

SAMPLE_WAYPOINTS = [
    {"lat": 10.78001, "lon": 122.62530, "name": "MV-001",
     "point_num": 1, "buffer_m": 1.0, "area_m2": 3.14, "status": "planned"},
    {"lat": 10.78015, "lon": 122.62545, "name": "MV-002",
     "point_num": 2, "buffer_m": 1.0, "area_m2": 2.98, "status": "planned"},
    {"lat": 10.78028, "lon": 122.62560, "name": "MV-003",
     "point_num": 3, "buffer_m": 1.0, "area_m2": 3.05, "status": "planned"},
]

SAMPLE_METADATA = {
    "image_name": "DJI_0042.jpg",
    "analyzed_at": "2026-03-15T04:30:00",
    "detection_mode": "hybrid",
    "total_points": 3,
}

SAMPLE_HEXAGONS = [
    {"_gps_lat": 10.78001, "_gps_lon": 122.62530, "center": (500, 400),
     "buffer_radius_m": 1.0, "area_m2": 3.14},
    {"_gps_lat": 10.78015, "_gps_lon": 122.62545, "center": (600, 450),
     "buffer_radius_m": 1.0, "area_m2": 2.98},
]

SAMPLE_DB_POINTS = [
    {"latitude": 10.78001, "longitude": 122.62530, "buffer_m": 1.0,
     "area_m2": 3.14, "status": "planned", "image_name": "DJI_0042.jpg"},
    {"latitude": 10.78028, "longitude": 122.62560, "buffer_m": 1.0,
     "area_m2": 3.05, "status": "planted", "image_name": "DJI_0042.jpg"},
]

passed = 0
failed = 0


def check(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name} — {detail}")


# =====================================================================
# 1. GPX TESTS
# =====================================================================
print("\n🧪 GPX Tests")
print("─" * 50)

gpx_str = generate_gpx(SAMPLE_WAYPOINTS, SAMPLE_METADATA)
check("GPX is non-empty string", isinstance(gpx_str, str) and len(gpx_str) > 100)
check("GPX has XML declaration", gpx_str.startswith("<?xml"))

# Parse and validate structure
gpx_ns = {"g": "http://www.topografix.com/GPX/1/1"}
gpx_root = ET.fromstring(gpx_str)
check("GPX root tag is 'gpx'", gpx_root.tag.endswith("gpx"))
check("GPX version is 1.1", gpx_root.get("version") == "1.1")

wpts = gpx_root.findall("g:wpt", gpx_ns)
check("GPX has 3 waypoints", len(wpts) == 3, f"got {len(wpts)}")

wpt1 = wpts[0]
check("First WPT lat correct", wpt1.get("lat") == "10.78001000",
      f"got {wpt1.get('lat')}")
check("First WPT lon correct", wpt1.get("lon") == "122.62530000",
      f"got {wpt1.get('lon')}")
check("First WPT name is MV-001",
      wpt1.find("g:name", gpx_ns).text == "MV-001")
check("First WPT has desc",
      wpt1.find("g:desc", gpx_ns) is not None)
check("First WPT sym is 'Flag, Green'",
      wpt1.find("g:sym", gpx_ns).text == "Flag, Green")

meta_el = gpx_root.find("g:metadata", gpx_ns)
check("GPX has metadata", meta_el is not None)
check("Metadata name correct",
      meta_el.find("g:name", gpx_ns).text == "MangroVision Planting Waypoints")
check("Metadata time correct",
      meta_el.find("g:time", gpx_ns).text == "2026-03-15T04:30:00")

# Empty waypoints
gpx_empty = generate_gpx([])
gpx_empty_root = ET.fromstring(gpx_empty)
wpts_empty = gpx_empty_root.findall("g:wpt", gpx_ns)
check("Empty GPX has 0 waypoints", len(wpts_empty) == 0)

# Single waypoint
gpx_single = generate_gpx([SAMPLE_WAYPOINTS[0]])
gpx_single_root = ET.fromstring(gpx_single)
check("Single-point GPX has 1 waypoint",
      len(gpx_single_root.findall("g:wpt", gpx_ns)) == 1)


# =====================================================================
# 2. KML TESTS
# =====================================================================
print("\n🧪 KML Tests")
print("─" * 50)

kml_str = generate_kml(SAMPLE_WAYPOINTS, SAMPLE_METADATA)
check("KML is non-empty string", isinstance(kml_str, str) and len(kml_str) > 100)
check("KML has XML declaration", kml_str.startswith("<?xml"))

kml_ns = {"k": "http://www.opengis.net/kml/2.2"}
kml_root = ET.fromstring(kml_str)
check("KML root tag is 'kml'", kml_root.tag.endswith("kml"))

doc = kml_root.find("k:Document", kml_ns)
check("KML has Document", doc is not None)

folder = doc.find("k:Folder", kml_ns)
check("KML has Folder", folder is not None)

placemarks = folder.findall("k:Placemark", kml_ns)
check("KML has 3 placemarks", len(placemarks) == 3, f"got {len(placemarks)}")

pm1 = placemarks[0]
check("First placemark name is MV-001",
      pm1.find("k:name", kml_ns).text == "MV-001")
check("First placemark has styleUrl",
      pm1.find("k:styleUrl", kml_ns).text == "#plantingPoint")

coords_text = pm1.find("k:Point/k:coordinates", kml_ns).text
check("First placemark coordinates correct",
      coords_text.startswith("122.62530000,10.78001000"),
      f"got {coords_text}")

# Style check
style = doc.find("k:Style[@id='plantingPoint']", kml_ns)
check("KML has plantingPoint style", style is not None)

# Empty
kml_empty = generate_kml([])
kml_empty_root = ET.fromstring(kml_empty)
kml_empty_folder = kml_empty_root.find("k:Document/k:Folder", kml_ns)
check("Empty KML has 0 placemarks",
      len(kml_empty_folder.findall("k:Placemark", kml_ns)) == 0)


# =====================================================================
# 3. GEOJSON TESTS
# =====================================================================
print("\n🧪 GeoJSON Tests")
print("─" * 50)

geojson_str = generate_geojson(SAMPLE_WAYPOINTS, SAMPLE_METADATA)
check("GeoJSON is non-empty string", isinstance(geojson_str, str) and len(geojson_str) > 50)

gj = json.loads(geojson_str)
check("GeoJSON type is FeatureCollection", gj["type"] == "FeatureCollection")
check("GeoJSON has 3 features", len(gj["features"]) == 3)

f1 = gj["features"][0]
check("First feature type is Feature", f1["type"] == "Feature")
check("First feature geometry is Point", f1["geometry"]["type"] == "Point")
check("First feature coordinates", f1["geometry"]["coordinates"] == [122.62530, 10.78001])
check("First feature name", f1["properties"]["name"] == "MV-001")
check("First feature buffer_m", f1["properties"]["buffer_m"] == 1.0)
check("Metadata image_name", gj["metadata"]["image_name"] == "DJI_0042.jpg")

# Empty
gj_empty = json.loads(generate_geojson([]))
check("Empty GeoJSON has 0 features", len(gj_empty["features"]) == 0)


# =====================================================================
# 4. HELPER: hexagons_to_waypoints
# =====================================================================
print("\n🧪 hexagons_to_waypoints Tests")
print("─" * 50)

# From hexagon format (app results)
wps = hexagons_to_waypoints(SAMPLE_HEXAGONS, image_name="DJI_0042.jpg")
check("Converts 2 hexagons", len(wps) == 2)
check("Lat from _gps_lat", wps[0]["lat"] == 10.78001)
check("Lon from _gps_lon", wps[0]["lon"] == 122.62530)
check("Name has image prefix", wps[0]["name"].startswith("DJI_0042"))
check("Buffer from buffer_radius_m", wps[0]["buffer_m"] == 1.0)

# From database format (Map Analytics)
wps_db = hexagons_to_waypoints(SAMPLE_DB_POINTS)
check("Converts 2 DB points", len(wps_db) == 2)
check("Lat from latitude", wps_db[0]["lat"] == 10.78001)
check("Lon from longitude", wps_db[0]["lon"] == 122.62530)
check("Status preserved", wps_db[1]["status"] == "planted")

# Empty list
wps_empty = hexagons_to_waypoints([])
check("Empty input → empty output", wps_empty == [])


# =====================================================================
# SUMMARY
# =====================================================================
print("\n" + "=" * 50)
total = passed + failed
if failed == 0:
    print(f"✅ All {passed} waypoint export tests passed!")
else:
    print(f"❌ {failed}/{total} tests FAILED, {passed} passed")
print("=" * 50)

sys.exit(0 if failed == 0 else 1)
