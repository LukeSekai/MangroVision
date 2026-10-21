"""Export endpoints — generate CSV, GPX, KML, and GeoJSON from point data."""

import csv
import io
from fastapi import APIRouter, Depends
from fastapi.responses import Response
from pydantic import BaseModel, Field
from typing import Optional

from waypoint_export import generate_gpx, generate_kml, generate_kmz, generate_geojson, safe_export_stem
from waypoint_styles import point_export_style
from api.report_pdf import ReportPdfRequest, render_restoration_report_pdf
from api.routes.dashboard import _require_lgu_user

router = APIRouter()


@router.post("/report/pdf", dependencies=[Depends(_require_lgu_user)])
def export_report_pdf(body: ReportPdfRequest):
    return Response(
        content=render_restoration_report_pdf(body),
        media_type="application/pdf",
        headers={
            "Content-Disposition": f'attachment; filename="{body.filename}"',
            "Cache-Control": "no-store",
        },
    )


class WaypointItem(BaseModel):
    lat: float
    lon: float
    name: Optional[str] = None
    point_num: Optional[int] = None
    buffer_m: Optional[float] = None
    area_m2: Optional[float] = None
    status: Optional[str] = "planned"
    eroded_unavailable: bool = False
    availability_reason: Optional[str] = None
    point_id: Optional[int] = None
    analysis_id: Optional[int] = None
    image_name: Optional[str] = None
    project_site_id: Optional[int] = None
    project_site_name: Optional[str] = None
    species: Optional[str] = None
    navigation_name: Optional[str] = None


class ExportRequest(BaseModel):
    waypoints: list[WaypointItem]
    image_name: Optional[str] = None
    detection_mode: Optional[str] = None
    export_name: Optional[str] = Field(default=None, max_length=240)
    filename_stem: Optional[str] = Field(default=None, max_length=180)
    exported_at: Optional[str] = Field(default=None, max_length=40)


def _metadata(body):
    return {key: getattr(body, key) for key in ('image_name', 'detection_mode', 'export_name', 'exported_at')}


def _download_header(body, extension):
    return f'attachment; filename="{safe_export_stem(body.filename_stem)}.{extension}"'


@router.post("/csv")
def export_csv(body: ExportRequest):
    def text_cell(value):
        text = value or ""
        return f"'{text}" if text.lstrip().startswith(("=", "+", "-", "@")) else text

    output = io.StringIO()
    writer = csv.DictWriter(
        output,
        fieldnames=["Point #", "Point Name", "Navigation ID", "Latitude", "Longitude", "Buffer (m)", "Area (m²)", "Status",
                    "Point ID", "Analysis ID", "Analysis", "Project Site ID", "Project Site", "Species",
                    "Eroded / Unavailable", "Availability Reason", "Marker Color", "Color Category"],
    )
    writer.writeheader()
    for index, waypoint in enumerate(body.waypoints, 1):
        style = point_export_style(waypoint.model_dump())
        writer.writerow(
            {
                "Point #": waypoint.point_num or index,
                "Point Name": text_cell(waypoint.name),
                "Navigation ID": text_cell(waypoint.navigation_name),
                "Latitude": f"{waypoint.lat:.7f}",
                "Longitude": f"{waypoint.lon:.7f}",
                "Buffer (m)": waypoint.buffer_m if waypoint.buffer_m is not None else "",
                "Area (m²)": f"{waypoint.area_m2:.2f}" if waypoint.area_m2 is not None else "",
                "Status": text_cell(waypoint.status or "planned"),
                "Point ID": waypoint.point_id,
                "Analysis ID": waypoint.analysis_id,
                "Analysis": text_cell(waypoint.image_name),
                "Project Site ID": waypoint.project_site_id,
                "Project Site": text_cell(waypoint.project_site_name),
                "Species": text_cell(waypoint.species),
                "Eroded / Unavailable": waypoint.eroded_unavailable,
                "Availability Reason": text_cell(waypoint.availability_reason),
                "Marker Color": style['marker_color'],
                "Color Category": style['color_category'],
            }
        )
    return Response(
        content=output.getvalue(),
        media_type="text/csv",
        headers={"Content-Disposition": _download_header(body, 'csv')},
    )


@router.post("/gpx")
def export_gpx(body: ExportRequest):
    wps = [w.model_dump() for w in body.waypoints]
    meta = _metadata(body)
    content = generate_gpx(wps, meta)
    return Response(content=content, media_type="application/gpx+xml",
                    headers={"Content-Disposition": _download_header(body, 'gpx')})


@router.post("/kml")
def export_kml(body: ExportRequest):
    wps = [w.model_dump() for w in body.waypoints]
    meta = _metadata(body)
    content = generate_kml(wps, meta)
    return Response(content=content, media_type="application/vnd.google-earth.kml+xml",
                    headers={"Content-Disposition": _download_header(body, 'kml')})


@router.post("/geojson")
def export_geojson(body: ExportRequest):
    wps = [w.model_dump() for w in body.waypoints]
    meta = _metadata(body)
    content = generate_geojson(wps, meta, include_style=True)
    return Response(content=content, media_type="application/geo+json",
                    headers={"Content-Disposition": _download_header(body, 'geojson')})


@router.post('/kmz', dependencies=[Depends(_require_lgu_user)])
def export_kmz(body: ExportRequest):
    waypoints = [waypoint.model_dump() for waypoint in body.waypoints]
    metadata = _metadata(body)
    return Response(content=generate_kmz(waypoints, metadata), media_type='application/vnd.google-earth.kmz',
        headers={'Content-Disposition': _download_header(body, 'kmz'), 'Cache-Control': 'no-store'})
