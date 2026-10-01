"""PostGIS-backed authoritative map-zone repository."""

from __future__ import annotations

import json
from typing import Any

from shapely.geometry import MultiPolygon, mapping, shape
from shapely.validation import make_valid

from .compat import get_connection

ZONE_TYPES = {"gis_coverage", "eroded", "forbidden", "warning"}


def normalize_polygon(geometry: dict[str, Any]) -> dict[str, Any]:
    if not isinstance(geometry, dict) or geometry.get("type") not in {"Polygon", "MultiPolygon"}:
        raise ValueError("Zone geometry must be a GeoJSON Polygon or MultiPolygon.")
    parsed = shape(geometry)
    if not parsed.is_valid:
        parsed = make_valid(parsed)
    polygon_parts = []
    if parsed.geom_type == "Polygon":
        polygon_parts = [parsed]
    elif parsed.geom_type == "MultiPolygon":
        polygon_parts = list(parsed.geoms)
    elif parsed.geom_type == "GeometryCollection":
        polygon_parts = [part for part in parsed.geoms if part.geom_type == "Polygon"]
    if not polygon_parts:
        raise ValueError("Zone geometry does not contain a valid polygon.")
    normalized = MultiPolygon(polygon_parts)
    if normalized.is_empty or not normalized.is_valid:
        raise ValueError("Zone geometry is empty or invalid.")
    return mapping(normalized)


def _feature(row: dict) -> dict:
    raw_properties = row.get("properties") or {}
    properties = (
        json.loads(raw_properties)
        if isinstance(raw_properties, str)
        else dict(raw_properties)
    )
    properties.update({
        "id": int(row["id"]),
        "zone_type": row["zone_type"],
        "name": row["name"],
        "warning_type": row.get("warning_type"),
        "severity": row.get("severity"),
        "notes": row.get("notes"),
        "created_at": row.get("created_at"),
        "updated_at": row.get("updated_at"),
    })
    return {
        "type": "Feature",
        "id": int(row["id"]),
        "properties": properties,
        "geometry": json.loads(row["polygon_geojson"]),
    }


def classify_gis_footprint(
    footprint: dict[str, Any] | None,
    *,
    latitude: float | None = None,
    longitude: float | None = None,
) -> dict[str, Any]:
    """Classify a footprint against the union of active GIS coverage zones."""
    if footprint is not None:
        requested_sql = (
            "extensions.ST_Multi(extensions.ST_SetSRID("
            "extensions.ST_GeomFromGeoJSON(?), 4326))"
        )
        requested_params = (json.dumps(normalize_polygon(footprint)),)
        is_point = False
    elif latitude is not None and longitude is not None:
        requested_sql = "extensions.ST_SetSRID(extensions.ST_MakePoint(?, ?), 4326)"
        requested_params = (float(longitude), float(latitude))
        is_point = True
    else:
        raise ValueError("A footprint or latitude/longitude is required.")
    conn = get_connection()
    try:
        row = conn.execute(f"""
            WITH requested AS (
                SELECT {requested_sql} AS geometry
            ), coverage AS (
                SELECT extensions.ST_UnaryUnion(extensions.ST_Collect(geometry)) AS geometry
                FROM map_zones
                WHERE zone_type = 'gis_coverage' AND deleted_at IS NULL
            )
            SELECT
                CASE
                    WHEN coverage.geometry IS NULL THEN 'unconfigured'
                    WHEN extensions.ST_Covers(coverage.geometry, requested.geometry) THEN 'inside'
                    WHEN extensions.ST_Intersects(coverage.geometry, requested.geometry) THEN 'partial'
                    ELSE 'outside'
                END AS status,
                CASE
                    WHEN coverage.geometry IS NULL THEN 0.0
                    WHEN extensions.ST_Covers(coverage.geometry, requested.geometry) THEN 100.0
                    WHEN {str(is_point).upper()} THEN 0.0
                    ELSE 100.0 * extensions.ST_Area(
                        extensions.ST_Intersection(coverage.geometry, requested.geometry)::extensions.geography
                    ) / NULLIF(
                        extensions.ST_Area(requested.geometry::extensions.geography),
                        0.0
                    )
                END AS estimated_inside_pct
            FROM requested CROSS JOIN coverage
        """, requested_params).fetchone()
    finally:
        conn.close()
    if not row or row["status"] == "unconfigured":
        raise RuntimeError("No active GIS coverage zone is configured.")
    percentage = max(0.0, min(100.0, float(row["estimated_inside_pct"] or 0.0)))
    return {
        "status": row["status"],
        "estimated_inside_pct": round(percentage, 1),
    }


def project_site_context(
    latitude: float,
    longitude: float,
    footprint: dict[str, Any] | None,
) -> dict[str, Any]:
    """Return named sites covering the center or intersecting the footprint."""
    normalized = normalize_polygon(footprint) if footprint is not None else None
    footprint_sql = (
        "extensions.ST_Multi(extensions.ST_SetSRID(extensions.ST_GeomFromGeoJSON(?), 4326))"
        if normalized is not None
        else "extensions.ST_SetSRID(extensions.ST_MakePoint(?, ?), 4326)"
    )
    footprint_params: tuple[Any, ...] = (
        (json.dumps(normalized),)
        if normalized is not None
        else (float(longitude), float(latitude))
    )
    conn = get_connection()
    try:
        rows = conn.execute(f"""
            WITH requested AS (
                SELECT
                    extensions.ST_SetSRID(extensions.ST_MakePoint(?, ?), 4326) AS center,
                    {footprint_sql} AS footprint
            )
            SELECT
                ps.id,
                ps.name,
                o.name AS organization_name,
                extensions.ST_Covers(ps.geometry, requested.center) AS covers_center
            FROM project_sites ps
            CROSS JOIN requested
            LEFT JOIN organizations o ON o.id = ps.organization_id
            WHERE extensions.ST_Covers(ps.geometry, requested.center)
               OR extensions.ST_Intersects(ps.geometry, requested.footprint)
            ORDER BY covers_center DESC, ps.id
        """, (
            float(longitude),
            float(latitude),
            *footprint_params,
        )).fetchall()
    finally:
        conn.close()
    matches = [
        {
            "id": int(row["id"]),
            "name": row["name"],
            "organization_name": row["organization_name"],
        }
        for row in rows
    ]
    return {
        "project_sites": matches,
        "location_label": matches[0]["name"] if matches else None,
    }


def list_zones(zone_type: str) -> list[dict]:
    if zone_type not in ZONE_TYPES:
        raise ValueError("Unknown map-zone type.")
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT id, zone_type, name, warning_type, severity, notes,
                   properties, polygon_geojson, created_at, updated_at
            FROM map_zones
            WHERE zone_type = ? AND deleted_at IS NULL
            ORDER BY id
        """, (zone_type,)).fetchall()
        return [_feature(dict(row)) for row in rows]
    finally:
        conn.close()


def feature_collection(zone_type: str) -> dict:
    return {
        "type": "FeatureCollection",
        "name": f"{zone_type}_zones",
        "features": list_zones(zone_type),
    }


def zone_revision_fingerprint() -> str:
    """Return a compact cache key that changes after any active-zone edit."""
    conn = get_connection()
    try:
        row = conn.execute("""
            SELECT COUNT(*) AS zone_count,
                   COALESCE(MAX(EXTRACT(EPOCH FROM COALESCE(updated_at, created_at))), 0) AS changed_at
            FROM map_zones
            WHERE deleted_at IS NULL
        """).fetchone()
        return f"{int(row['zone_count'])}:{float(row['changed_at'])}"
    finally:
        conn.close()


def create_zone(
    zone_type: str,
    feature: dict,
    created_by_user_id: int | None = None,
) -> dict:
    if zone_type not in ZONE_TYPES:
        raise ValueError("Unknown map-zone type.")
    geometry = normalize_polygon(feature.get("geometry") or {})
    properties = dict(feature.get("properties") or {})
    name = str(properties.pop("name", "") or f"{zone_type.title()} zone").strip()[:160]
    warning_type = properties.pop("warning_type", None)
    severity = properties.pop("severity", None)
    notes = properties.pop("notes", None)
    properties.pop("id", None)
    properties.pop("zone_type", None)
    conn = get_connection()
    try:
        cursor = conn.execute("""
            INSERT INTO map_zones (
                zone_type, name, warning_type, severity, notes,
                properties, polygon_geojson, created_by_user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            zone_type,
            name,
            warning_type,
            severity,
            notes,
            json.dumps(properties, ensure_ascii=False),
            json.dumps(geometry, ensure_ascii=False),
            created_by_user_id,
        ))
        zone_id = int(cursor.lastrowid)
        conn.commit()
    finally:
        conn.close()
    return get_zone(zone_type, zone_id)


def get_zone(zone_type: str, zone_id: int) -> dict | None:
    conn = get_connection()
    try:
        row = conn.execute("""
            SELECT id, zone_type, name, warning_type, severity, notes,
                   properties, polygon_geojson, created_at, updated_at
            FROM map_zones
            WHERE id = ? AND zone_type = ? AND deleted_at IS NULL
        """, (int(zone_id), zone_type)).fetchone()
        return _feature(dict(row)) if row else None
    finally:
        conn.close()


def update_zone(zone_type: str, zone_id: int, feature: dict) -> dict:
    existing = get_zone(zone_type, zone_id)
    if not existing:
        raise KeyError(zone_id)
    geometry = normalize_polygon(feature.get("geometry") or existing["geometry"])
    properties = {**existing["properties"], **dict(feature.get("properties") or {})}
    name = str(properties.pop("name", existing["properties"].get("name", "Zone"))).strip()[:160]
    warning_type = properties.pop("warning_type", existing["properties"].get("warning_type"))
    severity = properties.pop("severity", existing["properties"].get("severity"))
    notes = properties.pop("notes", existing["properties"].get("notes"))
    for key in ("id", "zone_type", "created_at", "updated_at"):
        properties.pop(key, None)
    conn = get_connection()
    try:
        conn.execute("""
            UPDATE map_zones
            SET name = ?, warning_type = ?, severity = ?, notes = ?,
                properties = ?, polygon_geojson = ?, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND zone_type = ? AND deleted_at IS NULL
        """, (
            name,
            warning_type,
            severity,
            notes,
            json.dumps(properties, ensure_ascii=False),
            json.dumps(geometry, ensure_ascii=False),
            int(zone_id),
            zone_type,
        ))
        conn.commit()
    finally:
        conn.close()
    return get_zone(zone_type, zone_id)


def soft_delete_zone(zone_type: str, zone_id: int) -> bool:
    conn = get_connection()
    try:
        result = conn.execute("""
            UPDATE map_zones
            SET deleted_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND zone_type = ? AND deleted_at IS NULL
        """, (int(zone_id), zone_type))
        conn.commit()
        return result.rowcount > 0
    finally:
        conn.close()


def replace_zone_collection(
    zone_type: str,
    features: list[dict],
    user_id: int | None = None,
) -> dict:
    retained: set[int] = set()
    output: list[dict] = []
    for feature in features:
        raw_id = feature.get("id") or (feature.get("properties") or {}).get("id")
        if raw_id and get_zone(zone_type, int(raw_id)):
            saved = update_zone(zone_type, int(raw_id), feature)
        else:
            saved = create_zone(zone_type, feature, user_id)
        retained.add(int(saved["id"]))
        output.append(saved)

    conn = get_connection()
    try:
        if retained:
            placeholders = ",".join("?" for _ in retained)
            conn.execute(
                f"""
                UPDATE map_zones
                SET deleted_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE zone_type = ? AND deleted_at IS NULL
                  AND id NOT IN ({placeholders})
                """,
                (zone_type, *sorted(retained)),
            )
        else:
            conn.execute("""
                UPDATE map_zones
                SET deleted_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
                WHERE zone_type = ? AND deleted_at IS NULL
            """, (zone_type,))
        conn.commit()
    finally:
        conn.close()
    return {"type": "FeatureCollection", "name": f"{zone_type}_zones", "features": output}
