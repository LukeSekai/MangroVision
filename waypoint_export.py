"""
MangroVision — GPS Waypoint Export Module
==========================================
Generate GPX, KML, and GeoJSON files from planting-point data
so field workers can navigate to exact planting locations.

The KMZ packager uses the existing Pillow dependency for embedded marker icons.
"""

import json
import re
import unicodedata
import xml.etree.ElementTree as ET
from html import escape
from io import BytesIO
from zipfile import ZipFile, ZIP_DEFLATED
from datetime import datetime
from typing import List, Dict, Optional
from waypoint_styles import gpx_symbol, is_eroded, marker_icon_png, marker_style_id, point_export_style


# ====================================================================
#  GPX 1.1
# ====================================================================

_GPX_NS = "http://www.topografix.com/GPX/1/1"
_XSI_NS = "http://www.w3.org/2001/XMLSchema-instance"
_MV_NS = "http://mangrovision.local/gpx/ext/1"

# Optional current-location attribution supplied by saved-point exports.
# Keep processing previews' existing GeoJSON contract when these are absent.
_POINT_CONTEXT = ("point_id", "analysis_id", "image_name", "project_site_id", "project_site_name", "species", "navigation_name")


def safe_export_stem(value):
    text = unicodedata.normalize('NFKD', str(value or '')).encode('ascii', 'ignore').decode()
    return re.sub(r'[^A-Za-z0-9_-]+', '-', text).strip('-_')[:160] or 'planting_points'

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
    ET.SubElement(meta_el, _g("name")).text = metadata.get('export_name') or "MangroVision Planting Waypoints"
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

        wpt_name = wp.get('navigation_name') or wp.get("name") or f"MV-{i:03d}"
        ET.SubElement(wpt, _g("name")).text = wpt_name
        if wp.get('navigation_name') and wp.get('name'):
            ET.SubElement(wpt, _g('cmt')).text = wp['name']

        # Description
        desc_lines = [f"Planting Point #{wp.get('point_num', i)}"]
        if wp.get("buffer_m") is not None:
            desc_lines.append(f"Buffer: {wp['buffer_m']}m")
        if wp.get("area_m2") is not None:
            desc_lines.append(f"Area: {wp['area_m2']:.2f} m\u00b2")
        if wp.get("status"):
            desc_lines.append(f"Status: {wp['status']}")
        if wp.get("species"):
            desc_lines.append(f"Species: {wp['species']}")
        if wp.get('name'):
            desc_lines.append(wp['name'])
        if wp.get('project_site_name'):
            desc_lines.append(f"Project site: {wp['project_site_name']}")
        if is_eroded(wp):
            desc_lines.append(wp.get('availability_reason') or 'Inside an eroded zone')
        ET.SubElement(wpt, _g("desc")).text = " | ".join(desc_lines)

        ET.SubElement(wpt, _g("sym")).text = gpx_symbol(wp)
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
        for key in (*_POINT_CONTEXT, "eroded_unavailable", "availability_reason"):
            if wp.get(key) is not None:
                value = wp[key]
                ET.SubElement(mv, f"{{{_MV_NS}}}{key}").text = str(value).lower() if isinstance(value, bool) else str(value)
        for key, value in point_export_style(wp).items():
            if value is not None:
                ET.SubElement(mv, f"{{{_MV_NS}}}{key}").text = value

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
    *, icon_paths: Optional[Dict] = None,
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
    doc_name = metadata.get('export_name') or "MangroVision Planting Waypoints"
    if not metadata.get('export_name') and metadata.get("image_name"):
        doc_name += f" — {metadata['image_name']}"
    ET.SubElement(doc, "name").text = doc_name

    desc_parts = []
    if metadata.get("analyzed_at"):
        desc_parts.append(f"Analyzed: {metadata['analyzed_at']}")
    if metadata.get("detection_mode"):
        desc_parts.append(f"Detection mode: {metadata['detection_mode']}")
    desc_parts.append(f"Total points: {len(waypoints)}")
    desc_parts.append('Point colors: dead red; eroded orange; planted yellow; Bungalon green; '
                      'Rhizophora magenta; other/unrecorded gray. '
                      'Color priority: dead > eroded > planted > species. Species and erosion details remain recorded.')
    ET.SubElement(doc, "description").text = " | ".join(desc_parts)

    # Small, centered dots keep dense planting grids readable in Google Earth.
    styled_points = {marker_style_id(point): point for point in waypoints}
    for style_id, waypoint in styled_points.items():
        appearance = point_export_style(waypoint)
        style = ET.SubElement(doc, "Style", {"id": style_id})
        icon_style = ET.SubElement(style, "IconStyle")
        red, green, blue = (appearance['marker_color'][index:index+2] for index in (1, 3, 5))
        # Packaged icons already contain their color; leave them untinted.
        ET.SubElement(icon_style, "color").text = 'ffffffff' if icon_paths else f'ff{blue}{green}{red}'
        ET.SubElement(icon_style, "scale").text = '0.35'
        icon = ET.SubElement(icon_style, "Icon")
        ET.SubElement(icon, "href").text = (icon_paths[style_id] if icon_paths else
            'https://maps.google.com/mapfiles/kml/shapes/placemark_circle.png')
        ET.SubElement(icon_style, "hotSpot", {
            "x": "0.5", "y": "0.5", "xunits": "fraction", "yunits": "fraction",
        })
        label_style = ET.SubElement(style, "LabelStyle")
        ET.SubElement(label_style, "scale").text = "0"

    # Folder for planting points
    folder = ET.SubElement(doc, "Folder")
    ET.SubElement(folder, "name").text = metadata.get('export_name') or "MangroVision Planting Points"
    ET.SubElement(folder, "open").text = "1"

    for i, wp in enumerate(waypoints, 1):
        lat = wp.get("lat")
        lon = wp.get("lon")
        if lat is None or lon is None:
            continue

        pm = ET.SubElement(folder, "Placemark")
        pm_name = wp.get("name", f"MV-{i:03d}")
        ET.SubElement(pm, "name").text = pm_name
        ET.SubElement(pm, "styleUrl").text = f'#{marker_style_id(wp)}'

        # Rich HTML description
        html = (
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
            html += f"<tr><td><b>Status</b></td><td>{escape(str(wp['status']))}</td></tr>"
        for key, label in (('species', 'Species'), ('project_site_name', 'Project site'), ('image_name', 'Analysis')):
            if wp.get(key):
                html += f'<tr><td><b>{label}</b></td><td>{escape(str(wp[key]))}</td></tr>'
        if is_eroded(wp):
            html += f'<tr><td><b>Erosion</b></td><td>{escape(str(wp.get("availability_reason") or "Inside an eroded zone"))}</td></tr>'
        html += "</table>"
        ET.SubElement(pm, "description").text = html

        context = {key: wp[key] for key in (*_POINT_CONTEXT, "eroded_unavailable", "availability_reason")
                   if wp.get(key) is not None}
        context.update({key: value for key, value in point_export_style(wp).items() if value is not None})
        if context:
            data = ET.SubElement(pm, "ExtendedData")
            for key, value in context.items():
                item = ET.SubElement(data, "Data", {"name": key})
                ET.SubElement(item, "value").text = str(value).lower() if isinstance(value, bool) else str(value)

        point = ET.SubElement(pm, "Point")
        ET.SubElement(point, "coordinates").text = f"{lon:.8f},{lat:.8f},0"

    tree = ET.ElementTree(kml)
    ET.indent(tree, space="  ")

    import io
    buf = io.BytesIO()
    tree.write(buf, encoding="utf-8", xml_declaration=True)
    return buf.getvalue().decode("utf-8")


def generate_kmz(waypoints: List[Dict], metadata: Optional[Dict] = None) -> bytes:
    """Package one KML and its generated icons, with no remote icon references."""
    variants = {marker_style_id(point): point_export_style(point) for point in waypoints}
    paths = {style_id: f'icons/{style_id}.png' for style_id in variants}
    output = BytesIO()
    with ZipFile(output, 'w', compression=ZIP_DEFLATED) as archive:
        archive.writestr('doc.kml', generate_kml(waypoints, metadata, icon_paths=paths))
        for style_id, appearance in variants.items():
            archive.writestr(paths[style_id], marker_icon_png(appearance['marker_color']))
    return output.getvalue()


# ====================================================================
#  GeoJSON
# ====================================================================

def generate_geojson(
    waypoints: List[Dict],
    metadata: Optional[Dict] = None,
    *, include_style: bool = False,
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
                "eroded_unavailable": bool(wp.get("eroded_unavailable", False)),
                "availability_reason": wp.get("availability_reason"),
                **{key: wp[key] for key in _POINT_CONTEXT if key in wp},
            },
        }
        if include_style:
            appearance = point_export_style(wp)
            feature['properties'].update({
                'marker-color': appearance['marker_color'], 'marker-size': 'small',
                'marker-symbol': '',
                **appearance,
                'mangrovision_style': appearance,
            })
        features.append(feature)

    collection = {
        "type": "FeatureCollection",
        "name": metadata.get('export_name') or "MangroVision Planting Points",
        "metadata": {
            "generator": "MangroVision",
            "image_name": metadata.get("image_name"),
            "analyzed_at": metadata.get("analyzed_at"),
            "detection_mode": metadata.get("detection_mode"),
            "total_points": len(features),
            **{key: metadata[key] for key in ('export_name', 'exported_at') if metadata.get(key)},
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

        eroded_unavailable = bool(
            h.get("_eroded_unavailable", h.get("eroded_unavailable", False))
        )
        waypoints.append({
            "lat": float(lat),
            "lon": float(lon),
            "name": f"{prefix}-{i:03d}",
            "point_num": h.get("point_num", i),
            "buffer_m": h.get("buffer_radius_m") or h.get("buffer_m"),
            "area_m2": h.get("area_m2") or h.get("area_sqm", 0),
            "status": "eroded_unavailable" if eroded_unavailable else h.get("status", "planned"),
            "eroded_unavailable": eroded_unavailable,
            "availability_reason": (
                "Inside an eroded zone" if eroded_unavailable else None
            ),
        })

    return waypoints
