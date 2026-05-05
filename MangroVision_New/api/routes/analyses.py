"""Analysis endpoints."""

from fastapi import APIRouter, HTTPException

from planting_database import (
    count_nearby_points,
    delete_analysis,
    find_overlapping_analyses,
    get_all_stats,
    get_analysis_by_id,
    list_analysis_points,
    list_analysis_summaries,
)

router = APIRouter()


@router.get("/stats")
def get_stats():
    return get_all_stats()


# MIGRATED FROM app.py analysis history drawer
@router.get("/")
def list_analyses():
    return list_analysis_summaries()


@router.get("/overlapping")
def overlapping(lat: float, lon: float, radius: float = 0.0015):
    return find_overlapping_analyses(lat, lon, radius)


@router.get("/nearby-points")
def nearby_points_count(lat: float, lon: float, radius: float = 0.0015):
    return {"count": count_nearby_points(lat, lon, radius)}


# MIGRATED FROM app.py analysis detail view (points)
@router.get("/{analysis_id}/points")
def analysis_points(analysis_id: int, only_unassigned: bool = False):
    points = list_analysis_points(analysis_id, only_unassigned=only_unassigned)
    if not points:
        # Return an empty list (200) — distinguishing "no analysis" vs "no points" is upstream.
        return []
    return points


@router.get("/{analysis_id}")
def analysis_detail(analysis_id: int):
    """
    Full payload for one saved analysis, shaped like the /process-stream
    result so the shared Results Summary overlay can render it directly.
    Images (drone photo + visualization overlay) are not persisted, so
    those fields are null — the overlay shows a placeholder for them.
    """
    row = get_analysis_by_id(analysis_id)
    if not row:
        raise HTTPException(status_code=404, detail="Analysis not found")

    points = list_analysis_points(analysis_id)
    image_name = row.get("image_name") or "Saved Analysis"
    gsd_cm = row.get("gsd_cm") or 0
    gsd_m = float(gsd_cm) / 100.0
    total_area = float(row.get("total_area_m2") or 0)

    coordinates = [
        {
            "point_num": p["point_num"],
            "latitude": p["latitude"],
            "longitude": p["longitude"],
            "pixel_x": int(p.get("pixel_x") or 0),
            "pixel_y": int(p.get("pixel_y") or 0),
            "buffer_m": p.get("buffer_m"),
            "area_m2": p.get("area_m2"),
        }
        for p in points
    ]

    waypoints = [
        {
            "lat": p["latitude"],
            "lon": p["longitude"],
            "point_num": p["point_num"],
            "buffer_m": p.get("buffer_m"),
            "area_m2": p.get("area_m2"),
            "status": p.get("status"),
            "name": f"{image_name}-{int(p['point_num']):03d}",
        }
        for p in points
    ]

    return {
        "analysis_id": analysis_id,
        "analysis_key": None,
        "uploaded_file_name": image_name,
        "detection_mode": "ai",
        "inputs": {},
        "metadata": {
            "has_exif": row.get("center_lat") is not None,
            "has_gps": row.get("center_lat") is not None,
            "gps_valid": row.get("center_lat") is not None,
            "image_center_lat": row.get("center_lat"),
            "image_center_lon": row.get("center_lon"),
            "heading_source": None,
            "camera_heading": None,
            "detected_heading": None,
            "camera": None,
        },
        "messages": {"info": [], "warnings": []},
        "overlaps": {"analyses": [], "nearby_points": 0},
        "metrics": {
            "canopy_count": row.get("canopy_count") or 0,
            "canopy_area_m2": 0.0,
            "canopy_coverage_pct": 0.0,
            "danger_area_m2": row.get("danger_area_m2") or 0,
            "danger_percentage": row.get("danger_pct") or 0,
            "plantable_area_m2": row.get("plantable_area_m2") or 0,
            "plantable_percentage": row.get("plantable_pct") or 0,
            "hexagon_count": row.get("hexagon_count") or 0,
            "safe_hexagon_count": row.get("hexagon_count") or 0,
            "forbidden_filtered_count": row.get("forbidden_filtered") or 0,
            "eroded_filtered_count": row.get("eroded_filtered") or 0,
            "duplicate_filtered_count": 0,
            "clipped_outside_orthophoto": 0,
            "gsd_m_per_pixel": gsd_m,
            "coverage_m": [row.get("coverage_w_m") or 0, row.get("coverage_h_m") or 0],
            "total_area_m2": total_area,
            "altitude_m": row.get("altitude_m"),
            "tile_count": 0,
            "processing_time_sec": 0,
            "model_name": None,
            "ai_confidence_threshold": row.get("ai_confidence"),
            "ai_avg_confidence": None,
        },
        "images": {
            "original_data_url": row.get("original_image"),
            "visualization_data_url": row.get("visualization_image"),
        },
        "map": {
            "available": row.get("center_lat") is not None,
            "match": None,
            "image_center_feature": None,
            "safe_points_geojson": None,
            "forbidden_filtered_geojson": None,
            "eroded_filtered_geojson": None,
            "coordinates": coordinates,
        },
        "exports": {
            "waypoints": waypoints,
            "metadata": {
                "image_name": image_name,
                "analyzed_at": row.get("analyzed_at"),
                "detection_mode": "ai",
                "total_points": len(points),
            },
            "json_results": None,
        },
        "can_save": False,
        "saved": True,
        "analyzed_at": row.get("analyzed_at"),
    }


@router.delete("/{analysis_id}")
def remove_analysis(analysis_id: int):
    delete_analysis(analysis_id)
    return {"status": "deleted", "id": analysis_id}
