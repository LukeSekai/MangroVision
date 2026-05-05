"""
MangroVision planting database for users, analyses, planting points, and field assignments.
"""

import sqlite3
import hashlib
import math
import secrets
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
        _migrate_schema(conn)
        _DB_INITIALIZED = True
    return conn


def _create_tables(conn: sqlite3.Connection):
    """Create the current application tables if they don't exist."""
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
            eroded_filtered     INTEGER DEFAULT 0,
            original_image      TEXT,
            visualization_image TEXT
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
        -- 4. PLANTERS  (field staff who receive planting assignments)
        -- ============================================================
        CREATE TABLE IF NOT EXISTS planters (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            full_name       TEXT    NOT NULL,
            phone           TEXT,
            base_label      TEXT,
            base_lat        REAL,
            base_lon        REAL,
            status          TEXT    NOT NULL DEFAULT 'active'
                            CHECK(status IN ('active', 'inactive')),
            notes           TEXT,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_planters_status
            ON planters(status);

        -- ============================================================
        -- 5. PLANTER_ASSIGNMENTS  (batch assignment per planter)
        -- ============================================================
        CREATE TABLE IF NOT EXISTS planter_assignments (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            planter_id          INTEGER NOT NULL REFERENCES planters(id) ON DELETE CASCADE,
            assigned_by_user_id INTEGER REFERENCES users(id) ON DELETE SET NULL,
            title               TEXT    NOT NULL,
            assignment_date     TEXT    NOT NULL,
            travel_mode         TEXT    NOT NULL DEFAULT 'walking',
            status              TEXT    NOT NULL DEFAULT 'active'
                                CHECK(status IN ('active', 'completed', 'archived')),
            notes               TEXT,
            created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_planter_assignments_planter
            ON planter_assignments(planter_id);
        CREATE INDEX IF NOT EXISTS idx_planter_assignments_status
            ON planter_assignments(status);

        -- ============================================================
        -- 6. PLANTER_ASSIGNMENT_POINTS  (ordered point list)
        -- ============================================================
        CREATE TABLE IF NOT EXISTS planter_assignment_points (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            assignment_id       INTEGER NOT NULL REFERENCES planter_assignments(id) ON DELETE CASCADE,
            planting_point_id   INTEGER NOT NULL REFERENCES planting_points(id) ON DELETE CASCADE,
            sequence_num        INTEGER NOT NULL,
            status              TEXT    NOT NULL DEFAULT 'pending'
                                CHECK(status IN ('pending', 'completed', 'skipped')),
            completed_at        TEXT,
            notes               TEXT,
            UNIQUE(assignment_id, planting_point_id),
            UNIQUE(assignment_id, sequence_num)
        );

        CREATE INDEX IF NOT EXISTS idx_assignment_points_assignment
            ON planter_assignment_points(assignment_id);
        CREATE INDEX IF NOT EXISTS idx_assignment_points_point
            ON planter_assignment_points(planting_point_id);
        CREATE INDEX IF NOT EXISTS idx_assignment_points_status
            ON planter_assignment_points(status);

        -- ============================================================
        -- 7. AUTH_SESSIONS  (persisted browser sessions)
        -- ============================================================
        CREATE TABLE IF NOT EXISTS auth_sessions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_type        TEXT    NOT NULL
                                CHECK(subject_type IN ('user', 'planter')),
            subject_id          INTEGER NOT NULL,
            token_hash          TEXT    NOT NULL UNIQUE,
            created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
            revoked_at          TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_auth_sessions_subject
            ON auth_sessions(subject_type, subject_id);
        CREATE INDEX IF NOT EXISTS idx_auth_sessions_active
            ON auth_sessions(subject_type, token_hash, revoked_at);

    """)
    conn.commit()


def _table_columns(conn: sqlite3.Connection, table_name: str) -> set:
    """Return the set of column names for a table."""
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def _migrate_schema(conn: sqlite3.Connection):
    """Apply additive schema migrations for existing local databases."""
    analyses_columns = _table_columns(conn, "analyses")
    if "original_image" not in analyses_columns:
        conn.execute("ALTER TABLE analyses ADD COLUMN original_image TEXT")
    if "visualization_image" not in analyses_columns:
        conn.execute("ALTER TABLE analyses ADD COLUMN visualization_image TEXT")

    planter_columns = _table_columns(conn, "planters")
    if "username" not in planter_columns:
        conn.execute("ALTER TABLE planters ADD COLUMN username TEXT")
    if "password_hash" not in planter_columns:
        conn.execute("ALTER TABLE planters ADD COLUMN password_hash TEXT")
    if "last_login" not in planter_columns:
        conn.execute("ALTER TABLE planters ADD COLUMN last_login TEXT")

    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_planters_username ON planters(username)")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS auth_sessions (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            subject_type        TEXT    NOT NULL
                                CHECK(subject_type IN ('user', 'planter')),
            subject_id          INTEGER NOT NULL,
            token_hash          TEXT    NOT NULL UNIQUE,
            created_at          TEXT    NOT NULL DEFAULT (datetime('now')),
            revoked_at          TEXT
        )
    """)
    conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_subject ON auth_sessions(subject_type, subject_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_auth_sessions_active ON auth_sessions(subject_type, token_hash, revoked_at)")
    conn.commit()


def init_db():
    """Explicitly create tables (called on import)."""
    conn = _get_connection()
    conn.close()


# ====================================================================
#  User Management
# ====================================================================

def _hash_password(password: str) -> str:
    """Return SHA-256 password hash (hex)."""
    return hashlib.sha256(password.encode('utf-8')).hexdigest()


def _hash_session_token(token: str) -> str:
    """Return a stable hash for a browser session token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _create_session(subject_type: str, subject_id: int) -> str:
    """Create a persisted session token for a user or planter."""
    if subject_type not in {"user", "planter"}:
        raise ValueError("Invalid auth session subject type.")

    token = secrets.token_urlsafe(32)
    token_hash = _hash_session_token(token)

    conn = _get_connection()
    conn.execute("""
        INSERT INTO auth_sessions (subject_type, subject_id, token_hash)
        VALUES (?, ?, ?)
    """, (subject_type, int(subject_id), token_hash))
    conn.commit()
    conn.close()
    return token


def _get_subject_by_session(subject_type: str, token: str) -> Optional[dict]:
    """Return the matching user or planter for an active session token."""
    if subject_type not in {"user", "planter"}:
        raise ValueError("Invalid auth session subject type.")
    token = (token or "").strip()
    if not token:
        return None

    token_hash = _hash_session_token(token)
    conn = _get_connection()
    session_row = conn.execute("""
        SELECT subject_id
        FROM auth_sessions
        WHERE subject_type = ?
          AND token_hash = ?
          AND revoked_at IS NULL
    """, (subject_type, token_hash)).fetchone()
    if not session_row:
        conn.close()
        return None

    if subject_type == "user":
        subject_row = conn.execute(
            "SELECT * FROM users WHERE id = ?",
            (session_row["subject_id"],),
        ).fetchone()
    else:
        subject_row = conn.execute(
            "SELECT * FROM planters WHERE id = ?",
            (session_row["subject_id"],),
        ).fetchone()
    conn.close()
    return dict(subject_row) if subject_row else None


def _revoke_session(subject_type: str, token: str) -> None:
    """Revoke a persisted session token."""
    if subject_type not in {"user", "planter"}:
        raise ValueError("Invalid auth session subject type.")
    token = (token or "").strip()
    if not token:
        return

    conn = _get_connection()
    conn.execute("""
        UPDATE auth_sessions
        SET revoked_at = ?
        WHERE subject_type = ?
          AND token_hash = ?
          AND revoked_at IS NULL
    """, (
        datetime.now().isoformat(timespec="seconds"),
        subject_type,
        _hash_session_token(token),
    ))
    conn.commit()
    conn.close()


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


def create_user_session(user_id: int) -> str:
    """Create a persisted session token for a planner user."""
    return _create_session("user", user_id)


def get_user_by_session_token(token: str) -> Optional[dict]:
    """Return the planner user associated with an active session token."""
    return _get_subject_by_session("user", token)


def revoke_user_session(token: str) -> None:
    """Revoke a planner user's persisted session token."""
    _revoke_session("user", token)


def get_planter_by_username(username: str) -> Optional[dict]:
    """Lookup a planter by username. Returns dict or None."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM planters WHERE lower(username) = lower(?)",
        (username.strip(),),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def authenticate_planter(username: str, password: str) -> Optional[dict]:
    """Authenticate a planter using username and password."""
    if not username or not password:
        return None

    planter = get_planter_by_username(username)
    if not planter:
        return None

    stored_hash = planter.get("password_hash") or ""
    if not stored_hash:
        return None

    if stored_hash == _hash_password(password):
        return planter

    return None


def create_planter_session(planter_id: int) -> str:
    """Create a persisted session token for a planter."""
    return _create_session("planter", planter_id)


def get_planter_by_session_token(token: str) -> Optional[dict]:
    """Return the planter associated with an active session token."""
    return _get_subject_by_session("planter", token)


def revoke_planter_session(token: str) -> None:
    """Revoke a planter's persisted session token."""
    _revoke_session("planter", token)


def update_planter_last_login(planter_id: int):
    """Stamp the current datetime on the planter's last_login field."""
    conn = _get_connection()
    conn.execute(
        "UPDATE planters SET last_login = ? WHERE id = ?",
        (datetime.now().isoformat(timespec='seconds'), planter_id),
    )
    conn.commit()
    conn.close()


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
    original_image: Optional[str] = None,
    visualization_image: Optional[str] = None,
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

    # ── Replace previous analysis only when the SAME image is re-run
    #    in the same area. Different images in the same area stay
    #    side-by-side so their unique points are preserved. Per-point
    #    dedup still prevents overlapping markers.
    if center_lat is not None and center_lon is not None and image_name:
        old_rows = conn.execute("""
            SELECT id FROM analyses
            WHERE image_name = ?
              AND center_lat BETWEEN ? AND ?
              AND center_lon BETWEEN ? AND ?
        """, (
            image_name,
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
             forbidden_filtered, eroded_filtered,
             original_image, visualization_image)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
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
        original_image,
        visualization_image,
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


def get_saved_point_locations(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> List[Tuple[float, float]]:
    """
    Return the raw (lat, lon) of every saved planting point inside the bbox.
    Used at processing time to filter new analysis hexagons that would
    overlap points already in the database.
    """
    conn = _get_connection()
    rows = conn.execute("""
        SELECT latitude, longitude FROM planting_points
        WHERE latitude  BETWEEN ? AND ?
          AND longitude BETWEEN ? AND ?
    """, (lat_min, lat_max, lon_min, lon_max)).fetchall()
    conn.close()
    return [(float(r['latitude']), float(r['longitude'])) for r in rows]


# ====================================================================
#  Aggregate Statistics (Map Analytics page)
# ====================================================================

def get_all_stats() -> dict:
    """Return aggregate statistics across all saved analyses."""
    conn = _get_connection()

    summary = conn.execute("""
        SELECT
            COUNT(*)                                  AS total_analyses,
            COALESCE(SUM(plantable_area_m2), 0)       AS total_plantable_m2,
            COALESCE(SUM(danger_area_m2), 0)          AS total_danger_m2,
            COALESCE(SUM(total_area_m2), 0)           AS total_coverage_m2,
            COALESCE(SUM(canopy_count), 0)            AS total_canopies,
            COALESCE(SUM(forbidden_filtered), 0)      AS total_forbidden_filtered,
            COALESCE(SUM(eroded_filtered), 0)         AS total_eroded_filtered,
            COALESCE((SELECT COUNT(*) FROM planting_points WHERE status = 'planned'), 0) AS total_planting_points,
            COALESCE((SELECT COUNT(*) FROM planting_points), 0) AS total_mapped_points,
            COALESCE((SELECT COUNT(*) FROM planting_points WHERE status = 'planted'), 0) AS total_planted_points,
            COALESCE((SELECT COUNT(*) FROM planting_points WHERE status = 'skipped'), 0) AS total_skipped_points
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


def delete_analysis(analysis_id: int):
    """Remove an analysis and its points (CASCADE handles planting_points)."""
    conn = _get_connection()
    conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))
    conn.commit()
    conn.close()


# ====================================================================
#  Planter Management
# ====================================================================

def create_planter(
    full_name: str,
    username: str,
    password: str,
    phone: str = "",
    base_label: str = "",
    base_lat: Optional[float] = None,
    base_lon: Optional[float] = None,
    notes: str = "",
    status: str = "active",
) -> int:
    """Create a planter record and return the new id."""
    full_name = full_name.strip()
    if not full_name:
        raise ValueError("Planter name is required.")

    username = (username or "").strip().lower()
    if not username:
        raise ValueError("Planter username is required.")
    if not password:
        raise ValueError("Planter password is required.")

    status = (status or "active").strip().lower()
    if status not in {"active", "inactive"}:
        raise ValueError("Invalid planter status.")

    conn = _get_connection()
    existing = conn.execute(
        "SELECT id FROM planters WHERE lower(username) = lower(?)",
        (username,),
    ).fetchone()
    if existing:
        conn.close()
        raise ValueError("That planter username is already in use.")

    cur = conn.execute("""
        INSERT INTO planters (full_name, username, password_hash, phone, base_label, base_lat, base_lon, notes, status)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        full_name,
        username,
        _hash_password(password),
        phone.strip() if phone else None,
        base_label.strip() if base_label else None,
        base_lat,
        base_lon,
        notes.strip() if notes else None,
        status,
    ))
    conn.commit()
    planter_id = cur.lastrowid
    conn.close()
    return planter_id


def list_planters(include_inactive: bool = True) -> List[dict]:
    """Return planters with active-assignment and assigned-point counts."""
    conn = _get_connection()
    query = """
        SELECT
            p.*,
            COUNT(DISTINCT CASE WHEN pa.status = 'active' THEN pa.id END) AS active_assignments,
            COALESCE(SUM(CASE WHEN pa.status = 'active' AND pap.status = 'pending' THEN 1 ELSE 0 END), 0) AS pending_points,
            COALESCE(SUM(CASE WHEN pa.status = 'active' AND pap.status = 'completed' THEN 1 ELSE 0 END), 0) AS completed_points
        FROM planters p
        LEFT JOIN planter_assignments pa ON pa.planter_id = p.id
        LEFT JOIN planter_assignment_points pap ON pap.assignment_id = pa.id
    """
    params: Tuple = ()
    if not include_inactive:
        query += " WHERE p.status = 'active'"
    query += """
        GROUP BY p.id
        ORDER BY CASE WHEN p.status = 'active' THEN 0 ELSE 1 END, lower(p.full_name)
    """
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_planter(planter_id: int) -> Optional[dict]:
    """Return one planter row or None."""
    conn = _get_connection()
    row = conn.execute("SELECT * FROM planters WHERE id = ?", (planter_id,)).fetchone()
    conn.close()
    return dict(row) if row else None


def get_analysis_by_id(analysis_id: int) -> Optional[dict]:
    """Return one saved analysis row by id, or None if not found."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM analyses WHERE id = ?", (analysis_id,),
    ).fetchone()
    conn.close()
    return dict(row) if row else None


def list_analysis_summaries() -> List[dict]:
    """Return saved analyses for assignment selection."""
    conn = _get_connection()
    rows = conn.execute("""
        SELECT id, image_name, analyzed_at, hexagon_count, center_lat, center_lon
        FROM analyses
        ORDER BY analyzed_at DESC
    """).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def list_analysis_points(analysis_id: int, only_unassigned: bool = False) -> List[dict]:
    """Return planting points for one saved analysis."""
    conn = _get_connection()
    base_query = """
        SELECT
            pp.id,
            pp.analysis_id,
            pp.point_num,
            pp.latitude,
            pp.longitude,
            pp.buffer_m,
            pp.area_m2,
            pp.status,
            a.image_name,
            a.analyzed_at
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pp.analysis_id = ?
    """
    params: List = [analysis_id]
    if only_unassigned:
        base_query += """
          AND NOT EXISTS (
              SELECT 1
              FROM planter_assignment_points pap
              JOIN planter_assignments pa ON pa.id = pap.assignment_id
              WHERE pap.planting_point_id = pp.id
                AND pa.status = 'active'
          )
        """
    base_query += " ORDER BY pp.point_num ASC"
    rows = conn.execute(base_query, tuple(params)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_planter_dashboard_stats() -> dict:
    """Return high-level planter management counts."""
    conn = _get_connection()
    row = conn.execute("""
        SELECT
            COALESCE((SELECT COUNT(*) FROM planters WHERE status = 'active'), 0) AS active_planters,
            COALESCE((SELECT COUNT(*) FROM planter_assignments WHERE status = 'active'), 0) AS active_assignments,
            COALESCE((
                SELECT COUNT(*)
                FROM planter_assignment_points pap
                JOIN planter_assignments pa ON pa.id = pap.assignment_id
                WHERE pa.status = 'active'
                  AND pap.status = 'pending'
            ), 0) AS pending_assigned_points,
            COALESCE((
                SELECT COUNT(*)
                FROM planter_assignment_points pap
                JOIN planter_assignments pa ON pa.id = pap.assignment_id
                WHERE pa.status = 'active'
                  AND pap.status = 'completed'
            ), 0) AS completed_assigned_points
    """).fetchone()
    conn.close()
    return dict(row)


def _refresh_assignment_status(conn: sqlite3.Connection, assignment_id: int) -> None:
    """Refresh one assignment batch status after point-level changes."""
    row = conn.execute("""
        SELECT
            COUNT(*) AS total_points,
            COALESCE(SUM(CASE WHEN status = 'pending' THEN 1 ELSE 0 END), 0) AS pending_points
        FROM planter_assignment_points
        WHERE assignment_id = ?
    """, (assignment_id,)).fetchone()

    if not row or row["total_points"] == 0:
        conn.execute("DELETE FROM planter_assignments WHERE id = ?", (assignment_id,))
        return

    new_status = "completed" if row["pending_points"] == 0 else "active"
    conn.execute(
        "UPDATE planter_assignments SET status = ? WHERE id = ?",
        (new_status, assignment_id),
    )


def _distance_score(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Cheap local distance score for assignment ordering."""
    avg_lat = (lat1 + lat2) / 2.0
    lon_scale = max(0.2, abs(math.cos(math.radians(avg_lat))))
    d_lat = lat1 - lat2
    d_lon = (lon1 - lon2) * lon_scale
    return (d_lat * d_lat) + (d_lon * d_lon)


def _order_point_ids_for_planter(
    conn: sqlite3.Connection,
    planter: dict,
    point_ids: List[int],
) -> List[int]:
    """Order assignment points using a nearest-neighbor pass from planter base when available."""
    placeholders = ",".join("?" for _ in point_ids)
    rows = conn.execute(f"""
        SELECT id, point_num, latitude, longitude
        FROM planting_points
        WHERE id IN ({placeholders})
    """, tuple(point_ids)).fetchall()
    points = [dict(row) for row in rows]
    if len(points) <= 1:
        return [point["id"] for point in points]

    current_lat = planter.get("base_lat")
    current_lon = planter.get("base_lon")
    if current_lat is None or current_lon is None:
        points.sort(key=lambda row: row["point_num"])
        current_lat = points[0]["latitude"]
        current_lon = points[0]["longitude"]

    remaining = points[:]
    ordered_ids: List[int] = []

    while remaining:
        remaining.sort(
            key=lambda row: (
                _distance_score(current_lat, current_lon, row["latitude"], row["longitude"]),
                row["point_num"],
            )
        )
        next_point = remaining.pop(0)
        ordered_ids.append(next_point["id"])
        current_lat = next_point["latitude"]
        current_lon = next_point["longitude"]

    return ordered_ids


def create_planter_assignment(
    planter_id: int,
    planting_point_ids: List[int],
    assigned_by_user_id: Optional[int] = None,
    title: str = "",
    assignment_date: str = "",
    travel_mode: str = "walking",
    notes: str = "",
) -> int:
    """Create one assignment batch and attach ordered planting points."""
    point_ids = [int(pid) for pid in planting_point_ids if pid is not None]
    if not point_ids:
        raise ValueError("Select at least one planting point.")

    planter = get_planter(planter_id)
    if not planter:
        raise ValueError("Selected planter was not found.")

    assignment_date = (assignment_date or datetime.now().date().isoformat()).strip()
    travel_mode = (travel_mode or "walking").strip().lower()
    if travel_mode not in {"walking", "driving"}:
        raise ValueError("Invalid travel mode.")

    conn = _get_connection()
    placeholders = ",".join("?" for _ in point_ids)
    duplicate_rows = conn.execute(f"""
        SELECT
            pp.point_num,
            a.image_name,
            pa.title,
            pl.full_name AS planter_name
        FROM planter_assignment_points pap
        JOIN planter_assignments pa ON pa.id = pap.assignment_id
        JOIN planting_points pp ON pp.id = pap.planting_point_id
        JOIN analyses a ON a.id = pp.analysis_id
        JOIN planters pl ON pl.id = pa.planter_id
        WHERE pa.status = 'active'
          AND pap.planting_point_id IN ({placeholders})
        ORDER BY pp.point_num
    """, tuple(point_ids)).fetchall()
    if duplicate_rows:
        sample = duplicate_rows[0]
        conn.close()
        raise ValueError(
            f"Point #{sample['point_num']} from {sample['image_name']} is already assigned to {sample['planter_name']}."
        )

    ordered_point_ids = _order_point_ids_for_planter(conn, planter, point_ids)

    if not title.strip():
        title = f"{planter['full_name']} planting run {assignment_date}"

    cur = conn.execute("""
        INSERT INTO planter_assignments (
            planter_id, assigned_by_user_id, title, assignment_date, travel_mode, status, notes
        )
        VALUES (?, ?, ?, ?, ?, 'active', ?)
    """, (
        planter_id,
        assigned_by_user_id,
        title.strip(),
        assignment_date,
        travel_mode,
        notes.strip() if notes else None,
    ))
    assignment_id = cur.lastrowid

    for idx, point_id in enumerate(ordered_point_ids, start=1):
        conn.execute("""
            INSERT INTO planter_assignment_points (
                assignment_id, planting_point_id, sequence_num, status
            )
            VALUES (?, ?, ?, 'pending')
        """, (
            assignment_id,
            point_id,
            idx,
        ))

    conn.commit()
    conn.close()
    return assignment_id


def list_planter_assignment_map_points() -> List[dict]:
    """Return all planting points with their current active assignment, if any."""
    conn = _get_connection()
    rows = conn.execute("""
        WITH active_point_assignments AS (
            SELECT
                pap.id AS assignment_point_id,
                pap.assignment_id,
                pap.planting_point_id,
                pap.sequence_num,
                pap.status AS assignment_status,
                pa.planter_id,
                pa.title AS assignment_title,
                pa.assignment_date,
                pa.created_at,
                pl.full_name AS planter_name,
                ROW_NUMBER() OVER (
                    PARTITION BY pap.planting_point_id
                    ORDER BY pa.created_at DESC, pap.id DESC
                ) AS rn
            FROM planter_assignment_points pap
            JOIN planter_assignments pa ON pa.id = pap.assignment_id
            JOIN planters pl ON pl.id = pa.planter_id
            WHERE pa.status = 'active'
        )
        SELECT
            pp.id,
            pp.analysis_id,
            pp.point_num,
            pp.latitude,
            pp.longitude,
            pp.buffer_m,
            pp.area_m2,
            pp.status AS planting_status,
            a.image_name,
            a.analyzed_at,
            a.center_lat,
            a.center_lon,
            apa.assignment_point_id,
            apa.assignment_id,
            apa.sequence_num,
            apa.assignment_status,
            apa.planter_id AS assigned_planter_id,
            apa.planter_name AS assigned_planter_name,
            apa.assignment_title,
            apa.assignment_date
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        LEFT JOIN active_point_assignments apa
               ON apa.planting_point_id = pp.id
              AND apa.rn = 1
        ORDER BY a.analyzed_at DESC, pp.point_num ASC
    """).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def assign_planting_point_to_planter(
    planter_id: int,
    planting_point_id: int,
    assigned_by_user_id: Optional[int] = None,
    travel_mode: str = "walking",
    allow_reassign: bool = False,
) -> dict:
    """Assign one planting point to a planter, optionally reassigning it."""
    planter = get_planter(planter_id)
    if not planter:
        raise ValueError("Selected planter was not found.")
    if planter.get("status") != "active":
        raise ValueError("Selected planter is not active.")

    travel_mode = (travel_mode or "walking").strip().lower()
    if travel_mode not in {"walking", "driving"}:
        raise ValueError("Invalid travel mode.")

    conn = _get_connection()
    point_row = conn.execute("""
        SELECT
            pp.id,
            pp.point_num,
            pp.latitude,
            pp.longitude,
            pp.buffer_m,
            pp.area_m2,
            pp.status AS planting_status,
            a.image_name,
            a.analyzed_at
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pp.id = ?
    """, (planting_point_id,)).fetchone()
    if not point_row:
        conn.close()
        raise ValueError("Selected planting point was not found.")

    point = dict(point_row)
    if point.get("planting_status") == "planted":
        conn.close()
        raise ValueError(
            f"Point #{point['point_num']} from {point['image_name']} is already marked as planted."
        )

    existing_row = conn.execute("""
        WITH active_point_assignments AS (
            SELECT
                pap.id AS assignment_point_id,
                pap.assignment_id,
                pap.planting_point_id,
                pap.sequence_num,
                pap.status AS assignment_status,
                pa.planter_id,
                pa.title AS assignment_title,
                pa.assignment_date,
                pa.created_at,
                pl.full_name AS planter_name,
                ROW_NUMBER() OVER (
                    PARTITION BY pap.planting_point_id
                    ORDER BY pa.created_at DESC, pap.id DESC
                ) AS rn
            FROM planter_assignment_points pap
            JOIN planter_assignments pa ON pa.id = pap.assignment_id
            JOIN planters pl ON pl.id = pa.planter_id
            WHERE pa.status = 'active'
        )
        SELECT *
        FROM active_point_assignments
        WHERE planting_point_id = ?
          AND rn = 1
    """, (planting_point_id,)).fetchone()

    reassigned_from = None
    source_assignment_id = None
    source_planter_id = None
    source_assignment_status = None
    source_assignment_deleted = False
    if existing_row:
        existing = dict(existing_row)
        if existing["planter_id"] == planter_id:
            conn.close()
            raise ValueError(
                f"Point #{point['point_num']} from {point['image_name']} is already assigned to {planter['full_name']}."
            )
        if not allow_reassign:
            conn.close()
            raise ValueError(
                f"Point #{point['point_num']} from {point['image_name']} is already assigned to {existing['planter_name']}."
            )

        reassigned_from = existing["planter_name"]
        source_assignment_id = existing["assignment_id"]
        source_planter_id = existing["planter_id"]
        source_assignment_status = existing["assignment_status"]
        conn.execute(
            "DELETE FROM planter_assignment_points WHERE id = ?",
            (existing["assignment_point_id"],),
        )
        _refresh_assignment_status(conn, existing["assignment_id"])
        source_assignment_deleted = (
            conn.execute(
                "SELECT 1 FROM planter_assignments WHERE id = ?",
                (existing["assignment_id"],),
            ).fetchone() is None
        )

    target_assignment_row = conn.execute("""
        SELECT id, title, assignment_date
        FROM planter_assignments
        WHERE planter_id = ?
          AND status = 'active'
        ORDER BY assignment_date DESC, created_at DESC, id DESC
        LIMIT 1
    """, (planter_id,)).fetchone()

    assignment_date = datetime.now().date().isoformat()
    created_new_assignment = False
    if target_assignment_row:
        target_assignment_id = target_assignment_row["id"]
        target_assignment_title = target_assignment_row["title"]
        assignment_date = target_assignment_row["assignment_date"]
    else:
        target_assignment_title = f"{planter['full_name']} map assignment {assignment_date}"
        cur = conn.execute("""
            INSERT INTO planter_assignments (
                planter_id, assigned_by_user_id, title, assignment_date, travel_mode, status, notes
            )
            VALUES (?, ?, ?, ?, ?, 'active', ?)
        """, (
            planter_id,
            assigned_by_user_id,
            target_assignment_title,
            assignment_date,
            travel_mode,
            "Created from the interactive planter management map.",
        ))
        target_assignment_id = cur.lastrowid
        created_new_assignment = True

    next_sequence_row = conn.execute("""
        SELECT COALESCE(MAX(sequence_num), 0) + 1 AS next_sequence
        FROM planter_assignment_points
        WHERE assignment_id = ?
    """, (target_assignment_id,)).fetchone()
    next_sequence = next_sequence_row["next_sequence"] if next_sequence_row else 1

    conn.execute("""
        INSERT INTO planter_assignment_points (
            assignment_id, planting_point_id, sequence_num, status
        )
        VALUES (?, ?, ?, 'pending')
    """, (
        target_assignment_id,
        planting_point_id,
        next_sequence,
    ))
    _refresh_assignment_status(conn, target_assignment_id)
    conn.commit()
    conn.close()

    return {
        "assignment_id": target_assignment_id,
        "assignment_title": target_assignment_title,
        "planter_id": planter_id,
        "planter_name": planter["full_name"],
        "planting_point_id": planting_point_id,
        "point_num": point["point_num"],
        "image_name": point["image_name"],
        "assignment_date": assignment_date,
        "assignment_status": "pending",
        "sequence_num": next_sequence,
        "created_new_assignment": created_new_assignment,
        "was_reassigned": bool(reassigned_from),
        "reassigned_from_planter_name": reassigned_from,
        "source_assignment_id": source_assignment_id,
        "source_planter_id": source_planter_id,
        "source_assignment_status": source_assignment_status,
        "source_assignment_deleted": source_assignment_deleted,
    }


def list_planter_assignments(planter_id: Optional[int] = None, active_only: bool = False) -> List[dict]:
    """Return assignment batches with aggregate status counts."""
    conn = _get_connection()
    query = """
        SELECT
            pa.id,
            pa.planter_id,
            pa.title,
            pa.assignment_date,
            pa.travel_mode,
            pa.status,
            pa.notes,
            pa.created_at,
            p.full_name AS planter_name,
            p.base_label,
            p.base_lat,
            p.base_lon,
            u.full_name AS assigned_by_name,
            COUNT(pap.id) AS total_points,
            COALESCE(SUM(CASE WHEN pap.status = 'pending' THEN 1 ELSE 0 END), 0) AS pending_points,
            COALESCE(SUM(CASE WHEN pap.status = 'completed' THEN 1 ELSE 0 END), 0) AS completed_points,
            COALESCE(SUM(CASE WHEN pap.status = 'skipped' THEN 1 ELSE 0 END), 0) AS skipped_points
        FROM planter_assignments pa
        JOIN planters p ON p.id = pa.planter_id
        LEFT JOIN users u ON u.id = pa.assigned_by_user_id
        LEFT JOIN planter_assignment_points pap ON pap.assignment_id = pa.id
    """
    clauses = []
    params: List = []
    if planter_id is not None:
        clauses.append("pa.planter_id = ?")
        params.append(planter_id)
    if active_only:
        clauses.append("pa.status = 'active'")
    if clauses:
        query += " WHERE " + " AND ".join(clauses)
    query += """
        GROUP BY pa.id
        ORDER BY CASE WHEN pa.status = 'active' THEN 0 ELSE 1 END, pa.assignment_date DESC, pa.created_at DESC
    """
    rows = conn.execute(query, tuple(params)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_assignment_points(assignment_id: int) -> List[dict]:
    """Return ordered points for one assignment batch."""
    conn = _get_connection()
    rows = conn.execute("""
        SELECT
            pap.id AS assignment_point_id,
            pap.assignment_id,
            pap.sequence_num,
            pap.status AS assignment_status,
            pap.completed_at,
            pap.notes,
            pa.title,
            pa.travel_mode,
            pa.assignment_date,
            pa.status AS assignment_status_overall,
            p.id AS planter_id,
            p.full_name AS planter_name,
            p.base_label,
            p.base_lat,
            p.base_lon,
            pp.id AS planting_point_id,
            pp.point_num,
            pp.latitude,
            pp.longitude,
            pp.buffer_m,
            pp.area_m2,
            a.image_name,
            a.analyzed_at
        FROM planter_assignment_points pap
        JOIN planter_assignments pa ON pa.id = pap.assignment_id
        JOIN planters p ON p.id = pa.planter_id
        JOIN planting_points pp ON pp.id = pap.planting_point_id
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pap.assignment_id = ?
        ORDER BY pap.sequence_num ASC
    """, (assignment_id,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def get_planter_field_points(planter_id: int) -> List[dict]:
    """Return all active assignment points for one planter, ordered for field use."""
    conn = _get_connection()
    rows = conn.execute("""
        SELECT
            pap.id AS assignment_point_id,
            pap.assignment_id,
            pap.sequence_num,
            pap.status AS assignment_status,
            pap.completed_at,
            pa.title,
            pa.travel_mode,
            pa.assignment_date,
            p.full_name AS planter_name,
            p.base_label,
            p.base_lat,
            p.base_lon,
            pp.id AS planting_point_id,
            pp.point_num,
            pp.latitude,
            pp.longitude,
            pp.buffer_m,
            pp.area_m2,
            a.image_name,
            a.analyzed_at
        FROM planter_assignment_points pap
        JOIN planter_assignments pa ON pa.id = pap.assignment_id
        JOIN planters p ON p.id = pa.planter_id
        JOIN planting_points pp ON pp.id = pap.planting_point_id
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pa.planter_id = ?
          AND pa.status = 'active'
        ORDER BY pa.assignment_date DESC, pap.sequence_num ASC
    """, (planter_id,)).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def update_assignment_point_status(assignment_point_id: int, status: str) -> None:
    """Update one assignment point status and auto-close completed batches."""
    status = (status or "").strip().lower()
    if status not in {"pending", "completed", "skipped"}:
        raise ValueError("Invalid assignment point status.")

    completed_at = datetime.now().isoformat(timespec='seconds') if status == "completed" else None

    conn = _get_connection()
    row = conn.execute("""
        SELECT assignment_id, planting_point_id
        FROM planter_assignment_points
        WHERE id = ?
    """, (assignment_point_id,)).fetchone()
    if not row:
        conn.close()
        raise ValueError("Assignment point was not found.")

    conn.execute("""
        UPDATE planter_assignment_points
        SET status = ?, completed_at = ?
        WHERE id = ?
    """, (status, completed_at, assignment_point_id))

    planting_status = "planned"
    if status == "completed":
        planting_status = "planted"
    elif status == "skipped":
        planting_status = "skipped"

    conn.execute("""
        UPDATE planting_points
        SET status = ?
        WHERE id = ?
    """, (planting_status, row["planting_point_id"]))

    _refresh_assignment_status(conn, row["assignment_id"])

    conn.commit()
    conn.close()


def archive_planter_assignment(assignment_id: int) -> None:
    """Archive an assignment batch so it no longer appears in active field work."""
    conn = _get_connection()
    conn.execute(
        "UPDATE planter_assignments SET status = 'archived' WHERE id = ?",
        (assignment_id,),
    )
    conn.commit()
    conn.close()


def delete_planter_assignment(assignment_id: int) -> None:
    """Delete an assignment batch and its ordered points."""
    conn = _get_connection()
    conn.execute("DELETE FROM planter_assignments WHERE id = ?", (assignment_id,))
    conn.commit()
    conn.close()


# ── Initialise on import ────────────────────────────────────────────
init_db()
