"""
MangroVision Planting Zone Database (Normalized 4-Table Schema)
================================================================
Tables:
  1. users             - Planner accounts (who operates the system)
  2. analyses          - Per-image analysis results
  3. planting_points   - Individual GPS-tagged planting locations
  4. exclusion_zones   - Forbidden & eroded areas (replaces loose GeoJSON)

All foreign keys use ON DELETE CASCADE so removing a user or analysis
automatically cleans up dependent rows.

SQLite with WAL journal mode for concurrent-read performance.
"""

import sqlite3
import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple

# ── Database path ───────────────────────────────────────────────────
_DB_PATH = Path(__file__).parent / "planting_zones.db"
_DB_INITIALIZED = False


# ====================================================================
#  Connection & Schema
# ====================================================================

def _get_connection() -> sqlite3.Connection:
    """Return a WAL-mode connection; create tables on first call."""
    global _DB_INITIALIZED
    conn = sqlite3.connect(str(_DB_PATH))
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row

    if not _DB_INITIALIZED:
        _create_tables(conn)
        _DB_INITIALIZED = True
    return conn


def _create_tables(conn: sqlite3.Connection):
    """Create the 4-table schema if it doesn't exist."""
    conn.executescript("""
        -- ============================================================
        -- 1. USERS  (the planner who operates MangroVision)
        -- ============================================================
        CREATE TABLE IF NOT EXISTS users (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name       TEXT    NOT NULL,
            email           TEXT    NOT NULL UNIQUE,
            role            TEXT    NOT NULL DEFAULT 'planner',
            organization    TEXT,
            password_hash   TEXT,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
            last_login      TEXT
        );

        -- ============================================================
        -- 2. ANALYSES  (one row per image analysis run)
        -- ============================================================
        CREATE TABLE IF NOT EXISTS analyses (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id             INTEGER REFERENCES users(id) ON DELETE SET NULL,
            image_name          TEXT    NOT NULL,
            analyzed_at         TEXT    NOT NULL,
            center_lat          REAL,
            center_lon          REAL,
            altitude_m          REAL,
            gsd_cm              REAL,
            coverage_w_m        REAL,
            coverage_h_m        REAL,
            total_area_m2       REAL,
            canopy_count        INTEGER,
            polygon_count       INTEGER DEFAULT 0,
            danger_area_m2      REAL,
            danger_pct          REAL,
            plantable_area_m2   REAL,
            plantable_pct       REAL,
            hexagon_count       INTEGER,
            ai_confidence       REAL,
            canopy_buffer_m     REAL,
            hexagon_size_m      REAL,
            forbidden_filtered  INTEGER DEFAULT 0,
            eroded_filtered     INTEGER DEFAULT 0
        );

        CREATE INDEX IF NOT EXISTS idx_analyses_user
            ON analyses(user_id);
        CREATE INDEX IF NOT EXISTS idx_analyses_center
            ON analyses(center_lat, center_lon);
        CREATE INDEX IF NOT EXISTS idx_analyses_date
            ON analyses(analyzed_at);

        -- ============================================================
        -- 3. PLANTING_POINTS  (individual GPS planting locations)
        -- ============================================================
        CREATE TABLE IF NOT EXISTS planting_points (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            analysis_id     INTEGER NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
            point_num       INTEGER NOT NULL,
            latitude        REAL    NOT NULL,
            longitude       REAL    NOT NULL,
            pixel_x         INTEGER,
            pixel_y         INTEGER,
            buffer_m        REAL,
            area_m2         REAL,
            status          TEXT    NOT NULL DEFAULT 'planned'
                            CHECK(status IN ('planned', 'planted', 'skipped'))
        );

        CREATE INDEX IF NOT EXISTS idx_points_analysis
            ON planting_points(analysis_id);
        CREATE INDEX IF NOT EXISTS idx_points_latlon
            ON planting_points(latitude, longitude);
        CREATE INDEX IF NOT EXISTS idx_points_status
            ON planting_points(status);

        -- ============================================================
        -- 4. EXCLUSION_ZONES  (forbidden + eroded areas)
        -- ============================================================
        CREATE TABLE IF NOT EXISTS exclusion_zones (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id             INTEGER REFERENCES users(id) ON DELETE SET NULL,
            zone_type           TEXT    NOT NULL
                                CHECK(zone_type IN ('forbidden', 'eroded')),
            geometry_geojson    TEXT    NOT NULL,
            reason              TEXT,
            created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_zones_type
            ON exclusion_zones(zone_type);
        CREATE INDEX IF NOT EXISTS idx_zones_user
            ON exclusion_zones(user_id);
    """)
    conn.commit()


def init_db():
    """Explicitly create tables (called on import)."""
    conn = _get_connection()
    conn.close()


# ====================================================================
#  User Management
# ====================================================================

def create_user(full_name: str, email: str, organization: str = None,
                password_hash: str = None) -> int:
    """Insert a new planner.  Returns the user id."""
    conn = _get_connection()
    cur = conn.execute("""
        INSERT INTO users (full_name, email, organization, password_hash)
        VALUES (?, ?, ?, ?)
    """, (full_name, email, organization, password_hash))
    uid = cur.lastrowid
    conn.commit()
    conn.close()
    return uid


def get_user_by_email(email: str) -> Optional[dict]:
    """Lookup a user by email.  Returns dict or None."""
    conn = _get_connection()
    row = conn.execute("SELECT * FROM users WHERE email = ?", (email,)).fetchone()
    conn.close()
    return dict(row) if row else None


def _hash_password(password: str) -> str:
    """Return SHA-256 password hash (hex)."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def get_user_by_name(full_name: str) -> Optional[dict]:
    """Lookup a user by full name. Returns dict or None."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE lower(full_name) = lower(?)",
        (full_name.strip(),),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def ensure_admin_user() -> int:
    """
    Ensure a default admin planner exists.
    Default credentials: username=admin, password=admin123
    """
    conn = _get_connection()
    row = conn.execute(
        "SELECT id FROM users WHERE lower(full_name) = 'admin'"
    ).fetchone()
    if row:
        uid = row['id']
    else:
        cur = conn.execute("""
            INSERT INTO users (full_name, email, role, organization, password_hash)
            VALUES (?, ?, ?, ?, ?)
        """, (
            'admin',
            'admin@mangrovision.local',
            'planner',
            'MangroVision',
            _hash_password('admin123'),
        ))
        uid = cur.lastrowid
        conn.commit()
    conn.close()
    return uid


def authenticate_user(username: str, password: str) -> Optional[dict]:
    """Authenticate using full_name + password. Returns user dict on success."""
    if not username or not password:
        return None

    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE lower(full_name) = lower(?)",
        (username.strip(),),
    ).fetchone()
    conn.close()

    if not row:
        return None

    user = dict(row)
    stored_hash = user.get('password_hash') or ''
    if not stored_hash:
        return None

    if stored_hash == _hash_password(password):
        return user

    return None


def get_or_create_default_user() -> int:
    """Return the id of the default planner; create if absent."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT id FROM users WHERE email = 'planner@mangrovision.local'"
    ).fetchone()
    if row:
        uid = row['id']
    else:
        cur = conn.execute("""
            INSERT INTO users (full_name, email, role, organization)
            VALUES ('MangroVision Planner', 'planner@mangrovision.local',
                    'planner', 'Leganes Municipal ENRO')
        """)
        uid = cur.lastrowid
        conn.commit()
    conn.close()
    return uid


def update_last_login(user_id: int):
    """Stamp the current datetime on the user's last_login field."""
    conn = _get_connection()
    conn.execute(
        "UPDATE users SET last_login = ? WHERE id = ?",
        (datetime.now().isoformat(timespec='seconds'), user_id),
    )
    conn.commit()
    conn.close()


def get_all_users() -> List[dict]:
    """Return all registered users."""
    conn = _get_connection()
    rows = conn.execute("SELECT * FROM users ORDER BY created_at").fetchall()
    conn.close()
    return [dict(r) for r in rows]


# ====================================================================
#  Analysis CRUD
# ====================================================================

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
    """Remove an analysis (CASCADE handles planting_points)."""
    conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))


def save_analysis(
    image_name: str,
    center_lat: Optional[float],
    center_lon: Optional[float],
    results: dict,
    hexagons: list,
    user_id: Optional[int] = None,
) -> Tuple[int, int, int]:
    """
    Persist an analysis and its planting points.

    If an existing analysis covers the same area (centre within ~15 m),
    that old row + points are **replaced** to avoid double-counting.

    Individual planting points are still deduplicated against points from
    *other* analyses (within ~0.5 m).

    Returns (analysis_id, new_points_count, skipped_duplicates_count).
    """
    conn = _get_connection()
    cur = conn.cursor()

    # Default to system planner if no user specified
    if user_id is None:
        user_id = get_or_create_default_user()

    # ── Replace previous analysis for the same area ───────────────
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
            _delete_analysis_rows(conn, row['id'])

    # ── Insert new analysis row ───────────────────────────────────
    cur.execute("""
        INSERT INTO analyses
            (user_id, image_name, analyzed_at, center_lat, center_lon,
             altitude_m, gsd_cm, coverage_w_m, coverage_h_m,
             total_area_m2, canopy_count, polygon_count,
             danger_area_m2, danger_pct,
             plantable_area_m2, plantable_pct,
             hexagon_count, ai_confidence,
             canopy_buffer_m, hexagon_size_m,
             forbidden_filtered, eroded_filtered)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        user_id,
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
        results.get('polygon_count', 0),
        results.get('danger_area_m2', 0),
        results.get('danger_percentage', 0),
        results.get('plantable_area_m2', 0),
        results.get('plantable_percentage', 0),
        results.get('hexagon_count', 0),
        results.get('ai_confidence'),
        results.get('canopy_buffer_m'),
        results.get('hexagon_size_m'),
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
            continue

        cur.execute("""
            INSERT INTO planting_points
                (analysis_id, point_num, latitude, longitude,
                 pixel_x, pixel_y, buffer_m, area_m2, status)
            VALUES (?,?,?,?,?,?,?,?,?)
        """, (
            analysis_id,
            i,
            _lat,
            _lon,
            int(h['center'][0]),
            int(h['center'][1]),
            h.get('buffer_radius_m'),
            h.get('area_m2', h.get('area_sqm', 0)),
            'planned',
        ))
        new_count += 1
        if _key:
            _existing.add(_key)

    # Update analysis row with actual inserted count
    cur.execute(
        "UPDATE analyses SET hexagon_count = ? WHERE id = ?",
        (new_count, analysis_id),
    )

    conn.commit()
    conn.close()
    return analysis_id, new_count, skipped


# ====================================================================
#  Overlap / Nearby Detection
# ====================================================================

def find_overlapping_analyses(
    center_lat: float,
    center_lon: float,
    radius_deg: float = 0.0015,
) -> List[dict]:
    """
    Return past analyses whose image centre is within *radius_deg*
    of the given point.
    """
    conn = _get_connection()
    rows = conn.execute("""
        SELECT a.id, a.image_name, a.analyzed_at, a.center_lat, a.center_lon,
               a.hexagon_count, a.plantable_area_m2,
               u.full_name AS planner_name
        FROM analyses a
        LEFT JOIN users u ON u.id = a.user_id
        WHERE a.center_lat BETWEEN ? AND ?
          AND a.center_lon BETWEEN ? AND ?
        ORDER BY a.analyzed_at DESC
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


# ====================================================================
#  Aggregate Statistics (Map Analytics page)
# ====================================================================

def get_all_stats() -> dict:
    """Return aggregate statistics across all saved analyses."""
    conn = _get_connection()

    summary = conn.execute("""
        SELECT
            COUNT(*)                                  AS total_analyses,
            COALESCE(SUM(hexagon_count), 0)           AS total_planting_points,
            COALESCE(SUM(plantable_area_m2), 0)       AS total_plantable_m2,
            COALESCE(SUM(danger_area_m2), 0)          AS total_danger_m2,
            COALESCE(SUM(total_area_m2), 0)           AS total_coverage_m2,
            COALESCE(SUM(canopy_count), 0)            AS total_canopies,
            COALESCE(SUM(forbidden_filtered), 0)      AS total_forbidden_filtered,
            COALESCE(SUM(eroded_filtered), 0)         AS total_eroded_filtered
        FROM analyses
    """).fetchone()

    stats = dict(summary)

    # All planting points for map rendering
    points = conn.execute("""
        SELECT pp.latitude, pp.longitude, pp.buffer_m, pp.area_m2,
               pp.status,
               a.image_name, a.analyzed_at
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        ORDER BY a.analyzed_at DESC
    """).fetchall()
    stats['points'] = [dict(p) for p in points]

    # Per-analysis breakdown (with planner name)
    analyses = conn.execute("""
        SELECT a.id, a.image_name, a.analyzed_at, a.center_lat, a.center_lon,
               a.canopy_count, a.polygon_count, a.hexagon_count,
               a.plantable_area_m2, a.plantable_pct,
               a.danger_area_m2, a.danger_pct,
               a.total_area_m2, a.gsd_cm,
               a.ai_confidence,
               a.canopy_buffer_m, a.hexagon_size_m,
               a.forbidden_filtered, a.eroded_filtered,
               u.full_name AS planner_name
        FROM analyses a
        LEFT JOIN users u ON u.id = a.user_id
        ORDER BY a.analyzed_at DESC
    """).fetchall()
    stats['analyses'] = [dict(a) for a in analyses]

    conn.close()
    return stats


def get_all_planting_points() -> List[dict]:
    """Return every saved planting point with its analysis metadata."""
    conn = _get_connection()
    rows = conn.execute("""
        SELECT pp.latitude, pp.longitude, pp.buffer_m, pp.area_m2,
               pp.point_num, pp.status,
               a.image_name, a.analyzed_at, a.id AS analysis_id,
               u.full_name AS planner_name
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        LEFT JOIN users u ON u.id = a.user_id
        ORDER BY a.analyzed_at DESC, pp.point_num
    """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_analysis(analysis_id: int):
    """Remove an analysis and its points (CASCADE handles planting_points)."""
    conn = _get_connection()
    conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    conn.commit()
    conn.close()


# ====================================================================
#  Planting Point Status
# ====================================================================

def update_point_status(point_id: int, status: str):
    """Update a single planting point's status (planned/planted/skipped)."""
    assert status in ('planned', 'planted', 'skipped'), f"Invalid status: {status}"
    conn = _get_connection()
    conn.execute(
        "UPDATE planting_points SET status = ? WHERE id = ?",
        (status, point_id),
    )
    conn.commit()
    conn.close()


def get_planting_summary() -> dict:
    """Return counts of points by status."""
    conn = _get_connection()
    rows = conn.execute("""
        SELECT status, COUNT(*) AS cnt
        FROM planting_points
        GROUP BY status
    """).fetchall()
    conn.close()
    return {r['status']: r['cnt'] for r in rows}


# ====================================================================
#  Exclusion Zones (replaces forbidden_zones.geojson + eroded_zones.geojson)
# ====================================================================

def save_exclusion_zone(
    zone_type: str,
    geometry_geojson: str,
    reason: str = None,
    user_id: Optional[int] = None,
) -> int:
    """
    Insert a single exclusion zone (forbidden or eroded).
    geometry_geojson is the GeoJSON geometry string for one polygon/feature.
    Returns the zone id.
    """
    assert zone_type in ('forbidden', 'eroded'), f"Invalid zone_type: {zone_type}"
    conn = _get_connection()
    if user_id is None:
        user_id = get_or_create_default_user()
    cur = conn.execute("""
        INSERT INTO exclusion_zones (user_id, zone_type, geometry_geojson, reason)
        VALUES (?, ?, ?, ?)
    """, (user_id, zone_type, geometry_geojson, reason))
    zid = cur.lastrowid
    conn.commit()
    conn.close()
    return zid


def get_exclusion_zones(zone_type: Optional[str] = None) -> List[dict]:
    """
    Return exclusion zones, optionally filtered by type.
    Each row includes the full geometry_geojson string.
    """
    conn = _get_connection()
    if zone_type:
        rows = conn.execute("""
            SELECT ez.*, u.full_name AS created_by_name
            FROM exclusion_zones ez
            LEFT JOIN users u ON u.id = ez.user_id
            WHERE ez.zone_type = ?
            ORDER BY ez.created_at
        """, (zone_type,)).fetchall()
    else:
        rows = conn.execute("""
            SELECT ez.*, u.full_name AS created_by_name
            FROM exclusion_zones ez
            LEFT JOIN users u ON u.id = ez.user_id
            ORDER BY ez.zone_type, ez.created_at
        """).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def delete_exclusion_zone(zone_id: int):
    """Remove a single exclusion zone."""
    conn = _get_connection()
    conn.execute("DELETE FROM exclusion_zones WHERE id = ?", (zone_id,))
    conn.commit()
    conn.close()


def clear_exclusion_zones(zone_type: str):
    """Remove all zones of a given type (e.g. clear all eroded zones)."""
    conn = _get_connection()
    conn.execute("DELETE FROM exclusion_zones WHERE zone_type = ?", (zone_type,))
    conn.commit()
    conn.close()


def export_exclusion_zones_geojson(zone_type: str) -> dict:
    """
    Build a GeoJSON FeatureCollection from stored exclusion zones.
    Compatible with the existing ForbiddenZoneFilter loader.
    """
    zones = get_exclusion_zones(zone_type)
    features = []
    for z in zones:
        try:
            geom = json.loads(z['geometry_geojson'])
        except (json.JSONDecodeError, TypeError):
            continue
        features.append({
            "type": "Feature",
            "properties": {
                "id": z['id'],
                "zone_type": z['zone_type'],
                "reason": z.get('reason'),
                "created_by": z.get('created_by_name'),
                "created_at": z.get('created_at'),
            },
            "geometry": geom,
        })
    return {
        "type": "FeatureCollection",
        "name": f"{zone_type}_zones",
        "features": features,
    }


def import_geojson_to_exclusion_zones(
    geojson_path: str,
    zone_type: str,
    user_id: Optional[int] = None,
) -> int:
    """
    Read a GeoJSON file and insert each Feature as an exclusion zone.
    Returns the number of zones imported.
    """
    path = Path(geojson_path)
    if not path.exists():
        return 0
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    features = data.get('features', [])
    count = 0
    for feat in features:
        geom = feat.get('geometry')
        if geom:
            props = feat.get('properties', {})
            reason = props.get('reason') or props.get('name') or zone_type
            save_exclusion_zone(
                zone_type=zone_type,
                geometry_geojson=json.dumps(geom),
                reason=reason,
                user_id=user_id,
            )
            count += 1
    return count


# ── Initialise on import ────────────────────────────────────────────
init_db()
