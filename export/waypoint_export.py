"""
MangroVision — GPS Waypoint Export Module
==========================================
Generate GPX, KML, and GeoJSON files from planting-point data
so field workers can navigate to exact planting locations.

No external dependencies — uses Python standard library only.
"""

import json
import xml.etree.ElementTree as ET
from datetime import datetime
from typing import List, Dict, Optional


# ====================================================================
#  GPX 1.1
# ====================================================================

_GPX_NS = "http://www.topografix.com/GPX/1/1"
_XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
_MV_NS = "http://mangrovision.local/gpx/ext/1"

def generate_gpx(
    waypoints: List[Dict],
    metadata: Optional[Dict] = None,
) -> str:
    """
    Build a GPX 1.1 XML string from a list of planting-point dicts.

    Parameters
    ----------
    waypoints : list[dict]
        Each dict must have ``lat`` and ``lon`` (floats).
        Optional keys: ``name``, ``point_num``, ``buffer_m``,
        ``area_m2``, ``status``.
    metadata : dict, optional
        ``image_name``, ``analyzed_at``, ``total_points``,
        ``detection_mode``.

    Returns
    -------
    str
        GPX 1.1 XML document (UTF-8, with XML declaration).
    """
    metadata = metadata or {}

    # Register namespaces so the output is clean
    ET.register_namespace("", _GPX_NS)
    ET.register_namespace("xsi", _XSI_NS)
    ET.register_namespace("mv", _MV_NS)

    # Use {namespace}tag notation — ET.register_namespace handles xmlns declarations.
    # Explicit xmlns:* attributes are NOT needed and would cause duplication.
    gpx = ET.Element(f"{{{_GPX_NS}}}gpx", {
        "version": "1.1",
        "creator": "MangroVision - AI Mangrove Planting Zone Analyzer",
        f"{{{_XSI_NS}}}schemaLocation": (
            f"{_GPX_NS} http://www.topografix.com/GPX/1/1/gpx.xsd"
        ),
    })

    # Shorthand for GPX-namespace tags
    def _g(tag):
        return f"{{{_GPX_NS}}}{tag}"

    # <metadata>
    meta_el = ET.SubElement(gpx, _g("metadata"))
    ET.SubElement(meta_el, _g("name")).text = "MangroVision Planting Waypoints"
    desc_parts = []
    if metadata.get("image_name"):
        desc_parts.append(f"Image: {metadata['image_name']}")
    if metadata.get("detection_mode"):
        desc_parts.append(f"Mode: {metadata['detection_mode']}")
    desc_parts.append(f"Points: {len(waypoints)}")
    ET.SubElement(meta_el, _g("desc")).text = " | ".join(desc_parts)

    author = ET.SubElement(meta_el, _g("author"))
    ET.SubElement(author, _g("name")).text = "MangroVision"

    time_str = metadata.get(
        "analyzed_at", datetime.now().isoformat(timespec="seconds")
    )
    ET.SubElement(meta_el, _g("time")).text = time_str

    # <wpt> elements
    for i, wp in enumerate(waypoints, 1):
        lat = wp.get("lat")
        lon = wp.get("lon")
        if lat is None or lon is None:
            continue

        wpt = ET.SubElement(gpx, _g("wpt"), {
            "lat": f"{lat:.8f}",
            "lon": f"{lon:.8f}",
        })

        wpt_name = wp.get("name", f"MV-{i:03d}")
        ET.SubElement(wpt, _g("name")).text = wpt_name

        # Description
        desc_lines = [f"Planting Point #{wp.get('point_num', i)}"]
        if wp.get("buffer_m") is not None:
            desc_lines.append(f"Buffer: {wp['buffer_m']}m")
        if wp.get("area_m2") is not None:
            desc_lines.append(f"Area: {wp['area_m2']:.2f} m\u00b2")
        if wp.get("status"):
            desc_lines.append(f"Status: {wp['status']}")
        ET.SubElement(wpt, _g("desc")).text = " | ".join(desc_lines)

        ET.SubElement(wpt, _g("sym")).text = "Flag, Green"
        ET.SubElement(wpt, _g("type")).text = "Planting Point"

        # MangroVision extensions
        ext = ET.SubElement(wpt, _g("extensions"))
        mv = ET.SubElement(ext, f"{{{_MV_NS}}}planting")
        if wp.get("buffer_m") is not None:
            ET.SubElement(mv, f"{{{_MV_NS}}}buffer_m").text = str(wp["buffer_m"])
        if wp.get("area_m2") is not None:
            ET.SubElement(mv, f"{{{_MV_NS}}}area_m2").text = f"{wp['area_m2']:.2f}"
        ET.SubElement(mv, f"{{{_MV_NS}}}point_num").text = str(wp.get("point_num", i))
        if wp.get("status"):
            ET.SubElement(mv, f"{{{_MV_NS}}}status").text = wp["status"]

    tree = ET.ElementTree(gpx)
    ET.indent(tree, space="  ")

    import io
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue().decode("utf-8")


# ====================================================================
#  KML 2.2
# ====================================================================

_KML_NS = "http://www.opengis.net/kml/2.2"

def generate_kml(
    waypoints: List[Dict],
    metadata: Optional[Dict] = None,
) -> str:
    """
    Build a KML 2.2 XML string from a list of planting-point dicts.

    Parameters
    ----------
    waypoints : list[dict]
        Same format as :func:`generate_gpx`.
    metadata : dict, optional
        Same format as :func:`generate_gpx`.

    Returns
    -------
    str
        KML 2.2 XML document (UTF-8, with XML declaration).
    """
    metadata = metadata or {}

    ET.register_namespace("", _KML_NS)

    kml = ET.Element("kml", {"xmlns": _KML_NS})
    doc = ET.SubElement(kml, "Document")

    # Document metadata
    doc_name = "MangroVision Planting Waypoints"
    if metadata.get("image_name"):
        doc_name += f" — {metadata['image_name']}"
    ET.SubElement(doc, "name").text = doc_name

    desc_parts = []
    if metadata.get("analyzed_at"):
        desc_parts.append(f"Analyzed: {metadata['analyzed_at']}")
    if metadata.get("detection_mode"):
        desc_parts.append(f"Detection mode: {metadata['detection_mode']}")
    desc_parts.append(f"Total points: {len(waypoints)}")
    ET.SubElement(doc, "description").text = " | ".join(desc_parts)

    # Green pin style
    style = ET.SubElement(doc, "Style", {"id": "plantingPoint"})
    icon_style = ET.SubElement(style, "IconStyle")
    ET.SubElement(icon_style, "color").text = "ff00aa00"  # Green (aaBBGGRR)
    ET.SubElement(icon_style, "scale").text = "1.0"
    icon = ET.SubElement(icon_style, "Icon")
    ET.SubElement(icon, "href").text = (
        "http://maps.google.com/mapfiles/kml/pushpin/grn-pushpin.png"
    )
    label_style = ET.SubElement(style, "LabelStyle")
    ET.SubElement(label_style, "scale").text = "0.8"

    # Folder for planting points
    folder = ET.SubElement(doc, "Folder")
    ET.SubElement(folder, "name").text = "MangroVision Planting Points"
    ET.SubElement(folder, "open").text = "1"

    for i, wp in enumerate(waypoints, 1):
        lat = wp.get("lat")
        lon = wp.get("lon")
        if lat is None or lon is None:
            continue

        pm = ET.SubElement(folder, "Placemark")
        pm_name = wp.get("name", f"MV-{i:03d}")
        ET.SubElement(pm, "name").text = pm_name
        ET.SubElement(pm, "styleUrl").text = "#plantingPoint"

        # Rich HTML description
        html = (
            "<![CDATA["
            "<table style='font-family:Arial;font-size:12px'>"
            f"<tr><td><b>Point</b></td><td>#{wp.get('point_num', i)}</td></tr>"
            f"<tr><td><b>Latitude</b></td><td>{lat:.7f}°</td></tr>"
            f"<tr><td><b>Longitude</b></td><td>{lon:.7f}°</td></tr>"
        )
        if wp.get("buffer_m") is not None:
            html += f"<tr><td><b>Buffer</b></td><td>{wp['buffer_m']}m</td></tr>"
        if wp.get("area_m2") is not None:
            html += f"<tr><td><b>Area</b></td><td>{wp['area_m2']:.2f} m²</td></tr>"
        if wp.get("status"):
            html += f"<tr><td><b>Status</b></td><td>{wp['status']}</td></tr>"
        html += "</table>]]>"
        ET.SubElement(pm, "description").text = html

        point = ET.SubElement(pm, "Point")
        ET.SubElement(point, "coordinates").text = f"{lon:.8f},{lat:.8f},0"

    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")

    import io
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue().decode("utf-8")


# ====================================================================
#  GeoJSON
# ====================================================================

def generate_geojson(
    waypoints: List[Dict],
    metadata: Optional[Dict] = None,
) -> str:
    """
    Build a GeoJSON FeatureCollection string from planting-point dicts.

    Parameters
    ----------
    waypoints : list[dict]
        Same format as :func:`generate_gpx`.
    metadata : dict, optional
        Same format as :func:`generate_gpx`.

    Returns
    -------
    str
        GeoJSON string (UTF-8).
    """
    metadata = metadata or {}

    features = []
    for i, wp in enumerate(waypoints, 1):
        lat = wp.get("lat")
        lon = wp.get("lon")
        if lat is None or lon is None:
            continue

        feature = {
            "type": "Feature",
            "geometry": {
                "type": "Point",
                "coordinates": [lon, lat],
            },
            "properties": {
                "name": wp.get("name", f"MV-{i:03d}"),
                "point_num": wp.get("point_num", i),
                "buffer_m": wp.get("buffer_m"),
                "area_m2": wp.get("area_m2"),
                "status": wp.get("status", "planned"),
            },
        }
        features.append(feature)

    collection = {
        "type": "FeatureCollection",
        "name": "MangroVision Planting Points",
        "metadata": {
            "generator": "MangroVision",
            "image_name": metadata.get("image_name"),
            "analyzed_at": metadata.get("analyzed_at"),
            "detection_mode": metadata.get("detection_mode"),
            "total_points": len(features),
        },
        "features": features,
    }

    return json.dumps(collection, indent=2, ensure_ascii=False)


# ====================================================================
#  Helper: Convert hexagon list → waypoint list
# ====================================================================

def hexagons_to_waypoints(
    hexagons: List[Dict],
    image_name: str = "",
) -> List[Dict]:
    """
    Convert MangroVision hexagon dicts (from ``results['hexagons']`` or
    the database ``planting_points`` rows) to the flat waypoint format
    expected by the generator functions.

    Parameters
    ----------
    hexagons : list[dict]
        Hexagons with ``_gps_lat``/``_gps_lon`` *or*
        ``latitude``/``longitude`` keys.
    image_name : str
        Used as a prefix for waypoint names.

    Returns
    -------
    list[dict]
        Waypoints ready for ``generate_gpx`` / ``generate_kml`` / etc.
    """
    prefix = image_name.split(".")[0][:8] if image_name else "MV"

    waypoints = []
    for i, h in enumerate(hexagons, 1):
        lat = h.get("_gps_lat") or h.get("latitude")
        lon = h.get("_gps_lon") or h.get("longitude")
        if lat is None or lon is None:
            continue

        waypoints.append({
            "lat": float(lat),
            "lon": float(lon),
            "name": f"{prefix}-{i:03d}",
            "point_num": h.get("point_num", i),
            "buffer_m": h.get("buffer_radius_m") or h.get("buffer_m"),
            "area_m2": h.get("area_m2") or h.get("area_sqm", 0),
            "status": h.get("status", "planned"),
        })

    return waypoints
