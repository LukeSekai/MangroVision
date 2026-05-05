"""Export endpoints — generate CSV, GPX, KML, and GeoJSON from point data."""

import csv
import io
from fastapi import APIRouter
from fastapi.responses import Response
from pydantic import BaseModel
from typing import Optional

from waypoint_export import generate_gpx, generate_kml, generate_geojson

router = APIRouter()


class WaypointItem(BaseModel):
    lat: float
    lon: float
    name: Optional[str] = None
    point_num: Optional[int] = None
    buffer_m: Optional[float] = None
    area_m2: Optional[float] = None
    status: Optional[str] = "planned"


class ExportRequest(BaseModel):
    waypoints: list[WaypointItem]
    image_name: Optional[str] = None
    detection_mode: Optional[str] = None


@router.post("/csv")
def export_csv(body: ExportRequest):
    # MIGRATED FROM app.py: analyze_image coordinate export lines 5836-5877
    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["Point #", "Latitude", "Longitude", "Buffer (m)", "Area (m²)", "Status"],
    )
    writer.writeheader()
    for index, waypoint in enumerate(body.waypoints, 1):
        writer.writerow(
            {
                "Point #": waypoint.point_num or index,
                "Latitude": f"{waypoint.lat:.7f}",
                "Longitude": f"{waypoint.lon:.7f}",
                "Buffer (m)": waypoint.buffer_m if waypoint.buffer_m is not None else "",
                "Area (m²)": f"{waypoint.area_m2:.2f}" if waypoint.area_m2 is not None else "",
                "Status": waypoint.status or "planned",
            }
        )
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=planting_points.csv"},
    )


@router.post("/gpx")
def export_gpx(body: ExportRequest):
    wps = [w.model_dump() for w in body.waypoints]
    meta = {"image_name": body.image_name, "detection_mode": body.detection_mode, "total_points": len(wps)}
    content = generate_gpx(wps, meta)
    return Response(content=content, media_type="application/gpx+xml",
                    headers={"Content-Disposition": "attachment; filename=planting_points.gpx"})


@router.post("/kml")
def export_kml(body: ExportRequest):
    wps = [w.model_dump() for w in body.waypoints]
    meta = {"image_name": body.image_name, "detection_mode": body.detection_mode, "total_points": len(wps)}
    content = generate_kml(wps, meta)
    return Response(content=content, media_type="application/vnd.google-earth.kml+xml",
                    headers={"Content-Disposition": "attachment; filename=planting_points.kml"})


@router.post("/geojson")
def export_geojson(body: ExportRequest):
    wps = [w.model_dump() for w in body.waypoints]
    meta = {"image_name": body.image_name, "detection_mode": body.detection_mode, "total_points": len(wps)}
    content = generate_geojson(wps, meta)
    return Response(content=content, media_type="application/geo+json",
                    headers={"Content-Disposition": "attachment; filename=planting_points.geojson"})
