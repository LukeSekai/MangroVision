"""
MangroVision Planting Zone Database
Stores planting zones from each analysis in a local SQLite database.
Provides overlap detection and aggregate statistics for the map.
"""

import sqlite3
import json
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

_DB_PATH = Path(__file__).parent / "planting_zones.db"


def _get_connection() -> sqlite3.Connection:
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create tables if they don't exist."""
    conn = _get_connection()
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS analyses (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            image_name  TEXT    NOT NULL,
            analyzed_at TEXT    NOT NULL,
            center_lat  REAL,
            center_lon  REAL,
            altitude_m  REAL,
            gsd_cm      REAL,
            coverage_w  REAL,
            coverage_h  REAL,
            total_area_m2       REAL,
            canopy_count        INTEGER,
            danger_area_m2      REAL,
            plantable_area_m2   REAL,
            hexagon_count       INTEGER,
            forbidden_filtered  INTEGER DEFAULT 0,
            eroded_filtered     INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS planting_points (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
            point_num   INTEGER NOT NULL,
            latitude    REAL    NOT NULL,
            longitude   REAL    NOT NULL,
            pixel_x     INTEGER,
            pixel_y     INTEGER,
            buffer_m    REAL,
            area_m2     REAL
        );

        CREATE INDEX IF NOT EXISTS idx_points_latlon
            ON planting_points(latitude, longitude);

        CREATE INDEX IF NOT EXISTS idx_analyses_center
            ON analyses(center_lat, center_lon);
    """)
    conn.commit()
    conn.close()


# ── Save ────────────────────────────────────────────────────────────

# ~0.5 m proximity threshold for point dedup (in degrees)
_DEDUP_RADIUS_DEG = 0.000005  # ~0.55 m at equator

# ~15 m radius for matching an analysis to the same area
_ANALYSIS_MATCH_DEG = 0.00015  # ~15 m at equator


def _existing_point_set(conn, lat_min, lat_max, lon_min, lon_max) -> set:
    """
    Return a set of (rounded_lat, rounded_lon) for every planting point
    already in the DB within the given bounding box.
    Rounding to 7 decimal places (~1 cm) makes the proximity check fast.
    """
    rows = conn.execute("""
        SELECT latitude, longitude FROM planting_points
        WHERE latitude  BETWEEN ? AND ?
          AND longitude BETWEEN ? AND ?
    """, (lat_min, lat_max, lon_min, lon_max)).fetchall()
    return {(round(r['latitude'], 7), round(r['longitude'], 7)) for r in rows}


def _delete_analysis_rows(conn, analysis_id: int):
    """Remove an analysis and all its planting points (internal helper)."""
    conn.execute("DELETE FROM planting_points WHERE analysis_id = ?", (analysis_id,))
    conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))


def save_analysis(
    image_name: str,
    center_lat: Optional[float],
    center_lon: Optional[float],
    results: dict,
    hexagons: list,
) -> Tuple[int, int, int]:
    """
    Persist an analysis and its planting points.

    If an existing analysis covers the same area (centre within ~15 m),
    that old analysis row and its planting points are **replaced** so
    aggregate statistics don't double-count.

    Individual planting points are still deduplicated against points from
    *other* analyses (within ~0.5 m).

    Returns (analysis_id, new_points_count, skipped_duplicates_count).
    """
    conn = _get_connection()
    cur = conn.cursor()

    # ── Replace previous analysis for the same area ───────────────
    replaced_ids = []
    if center_lat is not None and center_lon is not None:
        old_rows = conn.execute("""
            SELECT id FROM analyses
            WHERE center_lat BETWEEN ? AND ?
              AND center_lon BETWEEN ? AND ?
        """, (
            center_lat - _ANALYSIS_MATCH_DEG, center_lat + _ANALYSIS_MATCH_DEG,
            center_lon - _ANALYSIS_MATCH_DEG, center_lon + _ANALYSIS_MATCH_DEG,
        )).fetchall()
        for row in old_rows:
            replaced_ids.append(row['id'])
            _delete_analysis_rows(conn, row['id'])

    # ── Insert new analysis row ───────────────────────────────────
    cur.execute("""
        INSERT INTO analyses
            (image_name, analyzed_at, center_lat, center_lon,
             altitude_m, gsd_cm, coverage_w, coverage_h,
             total_area_m2, canopy_count, danger_area_m2,
             plantable_area_m2, hexagon_count,
             forbidden_filtered, eroded_filtered)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        image_name,
        datetime.now().isoformat(timespec='seconds'),
        center_lat,
        center_lon,
        results.get('altitude_m'),
        results.get('gsd_m_per_pixel', 0) * 100,
        results.get('coverage_m', [0, 0])[0],
        results.get('coverage_m', [0, 0])[1],
        results.get('total_area_m2', 0),
        results.get('canopy_count', 0),
        results.get('danger_area_m2', 0),
        results.get('plantable_area_m2', 0),
        results.get('hexagon_count', 0),
        results.get('_forbidden_filtered', 0),
        results.get('_eroded_filtered', 0),
    ))
    analysis_id = cur.lastrowid

    # ── Deduplicate planting points against OTHER analyses ────────
    if hexagons:
        _all_lats = [h.get('_gps_lat', 0) for h in hexagons]
        _all_lons = [h.get('_gps_lon', 0) for h in hexagons]
        _existing = _existing_point_set(
            conn,
            min(_all_lats) - _DEDUP_RADIUS_DEG,
            max(_all_lats) + _DEDUP_RADIUS_DEG,
            min(_all_lons) - _DEDUP_RADIUS_DEG,
            max(_all_lons) + _DEDUP_RADIUS_DEG,
        )
    else:
        _existing = set()

    new_count = 0
    skipped = 0
    for i, h in enumerate(hexagons, 1):
        _lat = h.get('_gps_lat')
        _lon = h.get('_gps_lon')
        _key = (round(_lat, 7), round(_lon, 7)) if _lat and _lon else None

        if _key and _key in _existing:
            skipped += 1
            continue  # Duplicate — skip

        cur.execute("""
            INSERT INTO planting_points
                (analysis_id, point_num, latitude, longitude,
                 pixel_x, pixel_y, buffer_m, area_m2)
            VALUES (?,?,?,?,?,?,?,?)
        """, (
            analysis_id,
            i,
            _lat,
            _lon,
            int(h['center'][0]),
            int(h['center'][1]),
            h.get('buffer_radius_m'),
            h.get('area_m2', h.get('area_sqm', 0)),
        ))
        new_count += 1
        if _key:
            _existing.add(_key)  # Prevent intra-batch dupes too

    # Update analysis row with actual inserted count
    cur.execute(
        "UPDATE analyses SET hexagon_count = ? WHERE id = ?",
        (new_count, analysis_id)
    )

    conn.commit()
    conn.close()
    return analysis_id, new_count, skipped


# ── Overlap Detection ───────────────────────────────────────────────

def find_overlapping_analyses(
    center_lat: float,
    center_lon: float,
    radius_deg: float = 0.0015,  # ~150 m at equator
) -> List[dict]:
    """
    Return past analyses whose image centre is within *radius_deg*
    of the given point (simple bounding-box check).
    """
    conn = _get_connection()
    rows = conn.execute("""
        SELECT id, image_name, analyzed_at, center_lat, center_lon,
               hexagon_count, plantable_area_m2
        FROM analyses
        WHERE center_lat BETWEEN ? AND ?
          AND center_lon BETWEEN ? AND ?
        ORDER BY analyzed_at DESC
    """, (
        center_lat - radius_deg, center_lat + radius_deg,
        center_lon - radius_deg, center_lon + radius_deg,
    )).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def count_nearby_points(
    center_lat: float,
    center_lon: float,
    radius_deg: float = 0.0015,
) -> int:
    """Count planting points already saved near a GPS centre."""
    conn = _get_connection()
    row = conn.execute("""
        SELECT COUNT(*) AS cnt FROM planting_points
        WHERE latitude  BETWEEN ? AND ?
          AND longitude BETWEEN ? AND ?
    """, (
        center_lat - radius_deg, center_lat + radius_deg,
        center_lon - radius_deg, center_lon + radius_deg,
    )).fetchone()
    conn.close()
    return row['cnt'] if row else 0


# ── Aggregate Statistics ────────────────────────────────────────────

def get_all_stats() -> dict:
    """Return aggregate statistics across all saved analyses."""
    conn = _get_connection()

    summary = conn.execute("""
        SELECT
            COUNT(*)                    AS total_analyses,
            COALESCE(SUM(hexagon_count), 0)       AS total_planting_points,
            COALESCE(SUM(plantable_area_m2), 0)   AS total_plantable_m2,
            COALESCE(SUM(danger_area_m2), 0)      AS total_danger_m2,
            COALESCE(SUM(total_area_m2), 0)       AS total_coverage_m2,
            COALESCE(SUM(canopy_count), 0)        AS total_canopies,
            COALESCE(SUM(forbidden_filtered), 0)  AS total_forbidden_filtered,
            COALESCE(SUM(eroded_filtered), 0)     AS total_eroded_filtered
        FROM analyses
    """).fetchone()

    stats = dict(summary)

    # All planting points for map rendering
    points = conn.execute("""
        SELECT pp.latitude, pp.longitude, pp.buffer_m, pp.area_m2,
               a.image_name, a.analyzed_at
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        ORDER BY a.analyzed_at DESC
    """).fetchall()
    stats['points'] = [dict(p) for p in points]

    # Per-analysis breakdown
    analyses = conn.execute("""
        SELECT id, image_name, analyzed_at, center_lat, center_lon,
               canopy_count, hexagon_count, plantable_area_m2,
               danger_area_m2, total_area_m2, gsd_cm,
               forbidden_filtered, eroded_filtered
        FROM analyses ORDER BY analyzed_at DESC
    """).fetchall()
    stats['analyses'] = [dict(a) for a in analyses]

    conn.close()
    return stats


def get_all_planting_points() -> List[dict]:
    """Return every saved planting point with its analysis metadata."""
    conn = _get_connection()
    rows = conn.execute("""
        SELECT pp.latitude, pp.longitude, pp.buffer_m, pp.area_m2,
               pp.point_num, a.image_name, a.analyzed_at, a.id AS analysis_id
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        ORDER BY a.analyzed_at DESC, pp.point_num
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_analysis(analysis_id: int):
    """Remove an analysis and its points."""
    conn = _get_connection()
    conn.execute("DELETE FROM planting_points WHERE analysis_id = ?", (analysis_id,))
    conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    conn.commit()
    conn.close()


# Initialise on import
init_db()
