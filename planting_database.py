"""
MangroVision planting database for users, analyses, planting points, and field assignments.
"""

from __future__ import annotations

import json
import hashlib
import math
import secrets
import base64
import binascii
import re
import uuid
from pathlib import Path
from datetime import date, datetime, time, timedelta, timezone
from typing import Any, List, Dict, Optional, Tuple
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

from mangrovision_db import DatabaseError, get_connection as _postgres_connection
from mangrovision_db.activity import append_activity
from mangrovision_db.config import get_settings
from mangrovision_db.monitoring_progress import age_snapshot, carried_counts
from mangrovision_db.passwords import hash_password, verify_password
from mangrovision_db.request_context import planter_session_token, staff_session_token
from mangrovision_db.storage import (
    StoredAsset,
    delete_assets,
    delete_object,
    signed_download_url,
    stale_object_keys,
    upload_analysis_data_urls,
)
from mangrovision_db.zones import normalize_polygon


class OutsideVisibleMapError(ValueError):
    """A planting point would be stored outside the active visible map."""

try:
    from shapely.geometry import Point, shape
except Exception:  # pragma: no cover - keeps DB import usable if geospatial deps are absent.
    Point = None
    shape = None

# ── Database path ───────────────────────────────────────────────────
_ERODED_ZONES_PATH = Path(__file__).parent / "eroded_zones.geojson"
_ERODED_ZONE_CACHE = {"expires_at": None, "polygons": []}
_WARNING_TYPE_LABELS = {
    "deep_mud": "Deep mud / difficult access",
    "unstable_sediment": "Unstable sediment",
    "tidal_exposure": "High tidal exposure",
    "wave_exposure": "Wave exposure",
    "low_survival_confidence": "Low survival confidence",
    "planner_warning": "Planner warning",
    "other": "Other",
}
_WARNING_SEVERITY_RANK = {"low": 1, "medium": 2, "high": 3}
try:
    _MANILA_TZ = ZoneInfo("Asia/Manila")
except ZoneInfoNotFoundError:  # Windows/offline installs may not bundle tzdata.
    _MANILA_TZ = timezone(timedelta(hours=8), name="Asia/Manila")
_DEFAULT_INSPECTION_INTERVALS = [30, 90, 180, 365]
_DEFAULT_OPERATIONAL_INSPECTION_INTERVAL_DAYS = 14
_ORGANIZATION_MONITORING_INTERVAL_DAYS = 14
# ISO weekday numbers: Monday=1 ... Sunday=7. These are preferred fieldwork
# days for planning; monitoring only moves when its 14-day date is a weekend.
_DEFAULT_INSPECTION_WEEKDAYS = [2, 5]
_VALID_MONITORING_STATUSES = {"alive", "dead", "missing"}
_VALID_ORGANIZATION_HEALTH_STATUSES = {
    "excellent", "good", "fair", "poor", "critical",
}
_VALID_PLANTING_SCHEDULE_STATUSES = {
    "draft", "requested", "tentative", "planned", "scheduled", "confirmed", "in_progress",
    "completed", "postponed", "cancelled",
}
_MAX_MONITORING_PHOTO_BYTES = 5 * 1024 * 1024


def _monitoring_workday(nominal_due_day: date) -> date:
    """Keep the 14-day planting anchor, but perform weekend visits on Monday."""
    weekday = nominal_due_day.weekday()
    if weekday == 5:
        return nominal_due_day + timedelta(days=2)
    if weekday == 6:
        return nominal_due_day + timedelta(days=1)
    return nominal_due_day


# ====================================================================
#  Connection & Schema
# ====================================================================

def _get_connection():
    """Return a PostgreSQL connection; Alembic owns all schema changes."""
    return _postgres_connection()


def _create_tables(conn: Any):
    """Create the current application tables if they don't exist."""
    raise RuntimeError("Runtime schema creation was removed; run 'alembic upgrade head'.")
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
            canopy_area_m2      REAL,
            canopy_coverage_pct REAL,
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
            visualization_image TEXT,
            analysis_detail_json TEXT
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
                            CHECK(status IN ('planned', 'planted', 'skipped')),
            planted_at      TEXT,
            planted_date    TEXT,
            deleted_at      TEXT,
            deletion_reason TEXT,
            deletion_batch_id TEXT,
            deletion_color  TEXT
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
            organization_id INTEGER REFERENCES organizations(id) ON DELETE RESTRICT,
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
            species             TEXT,
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

        -- ============================================================
        -- 8. SITE_ZONES  (admin-drawn plantable site polygons)
        -- ------------------------------------------------------------
        -- Named polygons covering distinct planting areas (e.g. "Site A —
        -- south berm"). Used to attribute mortality to a specific zone so
        -- the dashboard can show which areas perform better/worse. Stored
        -- in the DB rather than a GeoJSON file because mortality stats need
        -- a fast join with planting_points and we want per-zone metadata
        -- (notes, color override) without rewriting the whole file on edit.
        -- ============================================================
        CREATE TABLE IF NOT EXISTS site_zones (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT    NOT NULL,
            notes           TEXT,
            polygon_geojson TEXT    NOT NULL,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at      TEXT
        );

        -- ============================================================
        -- 9. WARNING_ZONES  (expert/planner caution polygons)
        -- ------------------------------------------------------------
        -- Non-blocking spatial annotations. Points inside these polygons
        -- remain plantable/assignable, but the UI surfaces a warning such
        -- as "deep mud" or "low survival confidence" for field judgment.
        -- ============================================================
        CREATE TABLE IF NOT EXISTS warning_zones (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            name            TEXT    NOT NULL,
            warning_type    TEXT    NOT NULL DEFAULT 'planner_warning',
            severity        TEXT    NOT NULL DEFAULT 'medium'
                            CHECK(severity IN ('low', 'medium', 'high')),
            notes           TEXT,
            polygon_geojson TEXT    NOT NULL,
            created_at      TEXT    NOT NULL DEFAULT (datetime('now')),
            updated_at      TEXT
        );

        -- ============================================================
        -- 10. POINT_DEATH_RECORDS  (immutable mortality history)
        -- ------------------------------------------------------------
        -- One row per *death event*. The current-state death_* columns on
        -- planting_points are convenient for the live map ("is this point
        -- right now dead?"), but they get wiped when a dead spot is reset
        -- to 'planned' for re-planting. Mortality reporting needs to count
        -- every death that ever happened — across planting cycles — so we
        -- mirror each mark-dead event here and treat this table as the
        -- source of truth for the mortality breakdown.
        --
        -- planter_id / planter_name / species are snapshotted from the
        -- assignment at death time so the per-planter/per-species stats
        -- survive a reset-to-planned that detaches the assignment.
        -- ============================================================
        CREATE TABLE IF NOT EXISTS point_death_records (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            planting_point_id   INTEGER NOT NULL REFERENCES planting_points(id) ON DELETE CASCADE,
            assignment_id       INTEGER,
            death_at            TEXT    NOT NULL,
            reason_category     TEXT    NOT NULL,
            reason_label        TEXT,
            notes               TEXT,
            planter_id          INTEGER,
            planter_name        TEXT,
            species             TEXT,
            created_at          TEXT    NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_death_records_point
            ON point_death_records(planting_point_id);
        CREATE INDEX IF NOT EXISTS idx_death_records_category
            ON point_death_records(reason_category);
        CREATE INDEX IF NOT EXISTS idx_death_records_death_at
            ON point_death_records(death_at);

    """)
    conn.commit()


def _table_columns(conn: Any, table_name: str) -> set:
    """Return the set of column names for a table."""
    raise RuntimeError("Runtime schema inspection was removed; Alembic owns the schema.")
    rows = conn.execute(f"PRAGMA table_info({table_name})").fetchall()
    return {row["name"] for row in rows}


def _clean_positive_interval(value: Any, field: str = "inspection_interval_days") -> int:
    try:
        interval = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a whole number of days.") from error
    if interval <= 0 or interval > 3650:
        raise ValueError(f"{field} must be between 1 and 3650 days.")
    return interval


def _organization_key(value: Any) -> str:
    """Return a stable case-insensitive lookup key without changing display text."""
    return " ".join(str(value or "").strip().split()).casefold()


def _clean_organization_name(value: Any) -> str:
    clean = " ".join(str(value or "").strip().split())
    if not clean:
        raise ValueError("organization is required.")
    return clean[:200]


def _register_organization(
    conn: Any,
    name: Any,
    inspection_interval_days: Any,
) -> Any:
    """Create/update the small canonical registry used by schedules and sites."""
    clean_name = _clean_organization_name(name)
    normalized_name = _organization_key(clean_name)
    cadence = _clean_positive_interval(inspection_interval_days)
    now = datetime.now(_MANILA_TZ).isoformat(timespec="seconds")
    conn.execute("""
        INSERT INTO organizations (
            name, normalized_name, inspection_interval_days, created_at, updated_at
        ) VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(normalized_name) DO UPDATE SET
            name = excluded.name,
            inspection_interval_days = excluded.inspection_interval_days,
            updated_at = excluded.updated_at
    """, (clean_name, normalized_name, cadence, now, now))
    return conn.execute(
        "SELECT * FROM organizations WHERE normalized_name = ?",
        (normalized_name,),
    ).fetchone()


def _resolve_registered_organization(
    conn: Any,
    name: Any,
    inspection_interval_days: Any,
    organization_id: Optional[Any] = None,
) -> Any:
    clean_name = _clean_organization_name(name)
    if organization_id is not None:
        try:
            clean_id = int(organization_id)
        except (TypeError, ValueError) as error:
            raise ValueError("organization_id must be a whole number or null.") from error
        existing = conn.execute(
            "SELECT * FROM organizations WHERE id = ?", (clean_id,),
        ).fetchone()
        if not existing:
            raise ValueError("Selected organization was not found.")
        if _organization_key(existing["name"]) != _organization_key(clean_name):
            raise ValueError("organization_id does not match the supplied organization name.")
    canonical = _register_organization(conn, clean_name, inspection_interval_days)
    if organization_id is not None and int(canonical["id"]) != int(organization_id):
        raise ValueError("organization_id does not match the supplied organization name.")
    return canonical


def _ensure_operational_coordination_schema(conn: Any) -> None:
    raise RuntimeError("Runtime schema migration was removed; run 'alembic upgrade head'.")
    """Install non-destructive organization, cadence, and nullable-site changes.

    SQLite cannot remove a NOT NULL constraint in place, so older schedule
    tables are rebuilt inside the caller's migration transaction. All legacy
    rows and primary keys are copied verbatim; only the site requirement is
    relaxed and the new snapshot columns are added.
    """
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS organizations (
            id                       INTEGER PRIMARY KEY AUTOINCREMENT,
            name                     TEXT NOT NULL,
            normalized_name          TEXT NOT NULL UNIQUE,
            inspection_interval_days INTEGER NOT NULL
                                     CHECK(inspection_interval_days BETWEEN 1 AND 3650),
            created_at               TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at               TEXT
        );

        CREATE TABLE IF NOT EXISTS organization_monitoring_records (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            organization_id       INTEGER NOT NULL REFERENCES organizations(id) ON DELETE RESTRICT,
            monitored_at          TEXT NOT NULL,
            alive_count           INTEGER NOT NULL CHECK(alive_count >= 0),
            dead_count            INTEGER NOT NULL CHECK(dead_count >= 0),
            average_height_cm     REAL NOT NULL CHECK(average_height_cm >= 0),
            health_status         TEXT NOT NULL CHECK(
                health_status IN ('excellent', 'good', 'fair', 'poor', 'critical')
            ),
            actions_taken         TEXT NOT NULL,
            inspector_user_id     INTEGER REFERENCES users(id) ON DELETE SET NULL,
            created_at            TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_organization_monitoring_org_date
            ON organization_monitoring_records(organization_id, monitored_at DESC);
        CREATE INDEX IF NOT EXISTS idx_organization_monitoring_date
            ON organization_monitoring_records(monitored_at DESC);
        CREATE INDEX IF NOT EXISTS idx_organizations_name
            ON organizations(name COLLATE NOCASE);
    """)

    site_columns = _table_columns(conn, "site_zones")
    if "organization_id" not in site_columns:
        conn.execute(
            "ALTER TABLE site_zones ADD COLUMN organization_id INTEGER "
            "REFERENCES organizations(id) ON DELETE RESTRICT"
        )
    if "inspection_interval_days" not in site_columns:
        conn.execute(
            "ALTER TABLE site_zones ADD COLUMN inspection_interval_days INTEGER "
            "CHECK(inspection_interval_days IS NULL OR "
            "inspection_interval_days BETWEEN 1 AND 3650)"
        )

    schedule_columns = _table_columns(conn, "planting_schedules")
    if "organization_id" not in schedule_columns:
        conn.execute(
            "ALTER TABLE planting_schedules ADD COLUMN organization_id INTEGER "
            "REFERENCES organizations(id) ON DELETE RESTRICT"
        )
    if "inspection_interval_days" not in schedule_columns:
        conn.execute(
            "ALTER TABLE planting_schedules ADD COLUMN inspection_interval_days INTEGER "
            "CHECK(inspection_interval_days IS NULL OR "
            "inspection_interval_days BETWEEN 1 AND 3650)"
        )

    project_site_info = next(
        (
            row for row in conn.execute("PRAGMA table_info(planting_schedules)").fetchall()
            if row["name"] == "project_site_id"
        ),
        None,
    )
    if project_site_info is not None and int(project_site_info["notnull"] or 0):
        conn.execute("DROP INDEX IF EXISTS idx_planting_schedules_site_time")
        conn.execute("DROP INDEX IF EXISTS idx_planting_schedules_status_time")
        conn.execute("ALTER TABLE planting_schedules RENAME TO planting_schedules_site_required")
        conn.execute("""
            CREATE TABLE planting_schedules (
                id                       INTEGER PRIMARY KEY AUTOINCREMENT,
                project_site_id          INTEGER REFERENCES site_zones(id) ON DELETE RESTRICT,
                organization             TEXT NOT NULL,
                contact                  TEXT,
                title                    TEXT NOT NULL,
                start_at                 TEXT NOT NULL,
                end_at                   TEXT NOT NULL,
                expected_planters        INTEGER CHECK(expected_planters IS NULL OR expected_planters >= 0),
                expected_seedlings       INTEGER CHECK(expected_seedlings IS NULL OR expected_seedlings >= 0),
                status                   TEXT NOT NULL DEFAULT 'requested',
                notes                    TEXT,
                created_by_user_id       INTEGER REFERENCES users(id) ON DELETE SET NULL,
                updated_by_user_id       INTEGER REFERENCES users(id) ON DELETE SET NULL,
                created_at               TEXT NOT NULL DEFAULT (datetime('now')),
                updated_at               TEXT,
                organization_id          INTEGER REFERENCES organizations(id) ON DELETE RESTRICT,
                inspection_interval_days INTEGER CHECK(
                    inspection_interval_days IS NULL OR inspection_interval_days BETWEEN 1 AND 3650
                )
            )
        """)
        conn.execute("""
            INSERT INTO planting_schedules (
                id, project_site_id, organization, contact, title, start_at, end_at,
                expected_planters, expected_seedlings, status, notes,
                created_by_user_id, updated_by_user_id, created_at, updated_at,
                organization_id, inspection_interval_days
            )
            SELECT
                id, project_site_id, organization, contact, title, start_at, end_at,
                expected_planters, expected_seedlings, status, notes,
                created_by_user_id, updated_by_user_id, created_at, updated_at,
                organization_id, inspection_interval_days
            FROM planting_schedules_site_required
        """)
        conn.execute("DROP TABLE planting_schedules_site_required")

    event_columns = _table_columns(conn, "planting_events")
    if "inspection_interval_days" not in event_columns:
        conn.execute(
            "ALTER TABLE planting_events ADD COLUMN inspection_interval_days INTEGER "
            "CHECK(inspection_interval_days IS NULL OR "
            "inspection_interval_days BETWEEN 1 AND 3650)"
        )

    settings_columns = _table_columns(conn, "dashboard_settings")
    if "inspection_weekdays_json" not in settings_columns:
        conn.execute(
            "ALTER TABLE dashboard_settings ADD COLUMN inspection_weekdays_json TEXT "
            "NOT NULL DEFAULT '[2, 5]'"
        )

    # Canonicalize legacy free-text schedule organizations without inventing
    # owners for unrelated legacy sites. Cadence falls back to the old primary
    # 30-day operational convention only where no snapshot ever existed.
    for row in conn.execute("""
        SELECT id, organization, inspection_interval_days
        FROM planting_schedules
        ORDER BY id
    """).fetchall():
        if not str(row["organization"] or "").strip():
            continue
        cadence = row["inspection_interval_days"] or _DEFAULT_OPERATIONAL_INSPECTION_INTERVAL_DAYS
        organization = _register_organization(conn, row["organization"], cadence)
        conn.execute("""
            UPDATE planting_schedules
            SET organization = ?, organization_id = ?,
                inspection_interval_days = COALESCE(inspection_interval_days, ?)
            WHERE id = ?
        """, (
            organization["name"], int(organization["id"]), int(cadence), int(row["id"]),
        ))

    # A legacy site can be attributed only when every linked historical
    # schedule names the same canonical organization. Ambiguous/unlinked sites
    # stay ownerless but remain readable and are never deleted.
    for site in conn.execute("SELECT id, organization_id FROM site_zones ORDER BY id").fetchall():
        if site["organization_id"] is None:
            linked = conn.execute("""
                SELECT DISTINCT organization_id
                FROM planting_schedules
                WHERE project_site_id = ? AND organization_id IS NOT NULL
            """, (int(site["id"]),)).fetchall()
            if len(linked) == 1:
                conn.execute(
                    "UPDATE site_zones SET organization_id = ? WHERE id = ?",
                    (int(linked[0]["organization_id"]), int(site["id"])),
                )
        conn.execute("""
            UPDATE site_zones
            SET inspection_interval_days = COALESCE(
                inspection_interval_days,
                (SELECT inspection_interval_days FROM organizations
                 WHERE id = site_zones.organization_id),
                ?
            )
            WHERE id = ?
        """, (_DEFAULT_OPERATIONAL_INSPECTION_INTERVAL_DAYS, int(site["id"])))

    conn.execute("CREATE INDEX IF NOT EXISTS idx_site_zones_organization ON site_zones(organization_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_planting_schedules_organization ON planting_schedules(organization_id, start_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_planting_schedules_site_time ON planting_schedules(project_site_id, start_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_planting_schedules_status_time ON planting_schedules(status, start_at)")


def _backfill_planting_events(conn: Any) -> int:
    raise RuntimeError("Legacy backfills only run through the cutover migration command.")
    """Idempotently snapshot legacy completed/planted rows as planting events.

    Existing rows are never deleted or rewritten.  A stable ``source_key`` makes
    the migration safe to run repeatedly, while a later replanting receives a
    new assignment-point id and therefore a distinct event.
    """
    before = conn.total_changes
    conn.execute("""
        INSERT OR IGNORE INTO planting_events (
            source_key, planting_point_id, assignment_point_id, assignment_id,
            site_zone_id, planter_id, species, planted_at, point_num,
            latitude, longitude, source, inspection_interval_days
        )
        SELECT
            'assignment-point:' || pap.id,
            pp.id,
            pap.id,
            pa.id,
            COALESCE(pa.site_zone_id, a.site_zone_id),
            pa.planter_id,
            COALESCE(NULLIF(TRIM(pa.species), ''), NULLIF(TRIM(a.species), '')),
            COALESCE(pap.completed_at, pp.planted_at, pp.planted_date, pa.assignment_date),
            pp.point_num,
            pp.latitude,
            pp.longitude,
            'legacy_backfill',
            COALESCE(
                (SELECT sz.inspection_interval_days FROM site_zones sz
                 WHERE sz.id = COALESCE(pa.site_zone_id, a.site_zone_id)),
                30
            )
        FROM planter_assignment_points pap
        JOIN planter_assignments pa ON pa.id = pap.assignment_id
        JOIN planting_points pp ON pp.id = pap.planting_point_id
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pap.status = 'completed'
          AND COALESCE(pap.completed_at, pp.planted_at, pp.planted_date, pa.assignment_date) IS NOT NULL
    """)
    conn.execute("""
        INSERT OR IGNORE INTO planting_events (
            source_key, planting_point_id, assignment_point_id, assignment_id,
            site_zone_id, planter_id, species, planted_at, point_num,
            latitude, longitude, source, inspection_interval_days
        )
        SELECT
            'legacy-point:' || pp.id || ':' || COALESCE(pp.planted_at, pp.planted_date),
            pp.id,
            NULL,
            NULL,
            a.site_zone_id,
            NULL,
            NULLIF(TRIM(a.species), ''),
            COALESCE(pp.planted_at, pp.planted_date),
            pp.point_num,
            pp.latitude,
            pp.longitude,
            'legacy_backfill',
            COALESCE(
                (SELECT sz.inspection_interval_days FROM site_zones sz
                 WHERE sz.id = a.site_zone_id),
                30
            )
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pp.status = 'planted'
          AND COALESCE(pp.planted_at, pp.planted_date) IS NOT NULL
          AND NOT EXISTS (
              SELECT 1
              FROM planter_assignment_points pap
              WHERE pap.planting_point_id = pp.id
                AND pap.status = 'completed'
          )
    """)
    return max(0, conn.total_changes - before)


def _install_dashboard_schema(conn: Any) -> None:
    raise RuntimeError("Runtime schema migration was removed; run 'alembic upgrade head'.")
    """Install the additive decision-dashboard schema and event triggers."""
    conn.executescript("""
        CREATE TABLE IF NOT EXISTS planting_events (
            id                  INTEGER PRIMARY KEY AUTOINCREMENT,
            source_key          TEXT NOT NULL UNIQUE,
            planting_point_id   INTEGER REFERENCES planting_points(id) ON DELETE SET NULL,
            assignment_point_id INTEGER REFERENCES planter_assignment_points(id) ON DELETE SET NULL,
            assignment_id       INTEGER REFERENCES planter_assignments(id) ON DELETE SET NULL,
            site_zone_id        INTEGER REFERENCES site_zones(id) ON DELETE SET NULL,
            planter_id          INTEGER REFERENCES planters(id) ON DELETE SET NULL,
            species             TEXT,
            planted_at          TEXT NOT NULL,
            point_num           INTEGER,
            latitude            REAL,
            longitude           REAL,
            source              TEXT NOT NULL DEFAULT 'field_completion',
            site_attribution_source TEXT NOT NULL DEFAULT 'planting_snapshot',
            inspection_interval_days INTEGER CHECK(
                inspection_interval_days IS NULL OR inspection_interval_days BETWEEN 1 AND 3650
            ),
            closed_at           TEXT,
            closure_reason      TEXT,
            created_at          TEXT NOT NULL DEFAULT (datetime('now'))
        );

        CREATE INDEX IF NOT EXISTS idx_planting_events_point
            ON planting_events(planting_point_id, planted_at);
        CREATE INDEX IF NOT EXISTS idx_planting_events_assignment
            ON planting_events(assignment_id, planted_at);
        CREATE INDEX IF NOT EXISTS idx_planting_events_site
            ON planting_events(site_zone_id, planted_at);
        CREATE INDEX IF NOT EXISTS idx_planting_events_planter
            ON planting_events(planter_id, planted_at);

        CREATE TABLE IF NOT EXISTS monitoring_observations (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            planting_event_id     INTEGER NOT NULL REFERENCES planting_events(id) ON DELETE CASCADE,
            interval_days         INTEGER NOT NULL CHECK(interval_days > 0),
            inspected_at          TEXT NOT NULL,
            status                TEXT NOT NULL CHECK(status IN ('alive', 'dead', 'missing')),
            condition             TEXT,
            height_cm             REAL CHECK(height_cm IS NULL OR height_cm >= 0),
            photo_path            TEXT,
            notes                 TEXT,
            actions_taken         TEXT,
            death_reason_category TEXT,
            inspector_user_id     INTEGER REFERENCES users(id) ON DELETE SET NULL,
            death_record_id       INTEGER REFERENCES point_death_records(id) ON DELETE SET NULL,
            created_at            TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at            TEXT,
            UNIQUE(planting_event_id, interval_days)
        );

        CREATE INDEX IF NOT EXISTS idx_monitoring_observations_event
            ON monitoring_observations(planting_event_id, interval_days);
        CREATE INDEX IF NOT EXISTS idx_monitoring_observations_date
            ON monitoring_observations(inspected_at);
        CREATE INDEX IF NOT EXISTS idx_monitoring_observations_status
            ON monitoring_observations(status);

        CREATE TABLE IF NOT EXISTS dashboard_settings (
            year                    INTEGER PRIMARY KEY,
            annual_planting_target  INTEGER CHECK(annual_planting_target IS NULL OR annual_planting_target >= 0),
            min_survival_target_pct REAL CHECK(min_survival_target_pct IS NULL OR (min_survival_target_pct >= 0 AND min_survival_target_pct <= 100)),
            inspection_intervals_json TEXT NOT NULL DEFAULT '[30, 90, 180, 365]',
            inspection_weekdays_json TEXT NOT NULL DEFAULT '[2, 5]',
            updated_at              TEXT NOT NULL DEFAULT (datetime('now')),
            updated_by_user_id      INTEGER REFERENCES users(id) ON DELETE SET NULL
        );

        CREATE TABLE IF NOT EXISTS organizations (
            id                       INTEGER PRIMARY KEY AUTOINCREMENT,
            name                     TEXT NOT NULL,
            normalized_name          TEXT NOT NULL UNIQUE,
            inspection_interval_days INTEGER NOT NULL
                                     CHECK(inspection_interval_days BETWEEN 1 AND 3650),
            created_at               TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at               TEXT
        );

        CREATE TABLE IF NOT EXISTS planting_schedules (
            id                    INTEGER PRIMARY KEY AUTOINCREMENT,
            project_site_id       INTEGER REFERENCES site_zones(id) ON DELETE RESTRICT,
            organization          TEXT NOT NULL,
            organization_id       INTEGER REFERENCES organizations(id) ON DELETE RESTRICT,
            inspection_interval_days INTEGER NOT NULL
                                     CHECK(inspection_interval_days BETWEEN 1 AND 3650),
            contact               TEXT,
            title                 TEXT NOT NULL,
            start_at              TEXT NOT NULL,
            end_at                TEXT NOT NULL,
            expected_planters     INTEGER CHECK(expected_planters IS NULL OR expected_planters >= 0),
            expected_seedlings    INTEGER CHECK(expected_seedlings IS NULL OR expected_seedlings >= 0),
            status                TEXT NOT NULL DEFAULT 'requested',
            notes                 TEXT,
            created_by_user_id    INTEGER REFERENCES users(id) ON DELETE SET NULL,
            updated_by_user_id    INTEGER REFERENCES users(id) ON DELETE SET NULL,
            created_at            TEXT NOT NULL DEFAULT (datetime('now')),
            updated_at            TEXT
        );

        CREATE INDEX IF NOT EXISTS idx_planting_schedules_site_time
            ON planting_schedules(project_site_id, start_at);
        CREATE INDEX IF NOT EXISTS idx_planting_schedules_status_time
            ON planting_schedules(status, start_at);
    """)

    _ensure_operational_coordination_schema(conn)

    event_columns = _table_columns(conn, "planting_events")
    if "closed_at" not in event_columns:
        conn.execute("ALTER TABLE planting_events ADD COLUMN closed_at TEXT")
    if "closure_reason" not in event_columns:
        conn.execute("ALTER TABLE planting_events ADD COLUMN closure_reason TEXT")
    if "site_attribution_source" not in event_columns:
        conn.execute("ALTER TABLE planting_events ADD COLUMN site_attribution_source TEXT NOT NULL DEFAULT 'planting_snapshot'")

    observation_columns = _table_columns(conn, "monitoring_observations")
    if "actions_taken" not in observation_columns:
        conn.execute("ALTER TABLE monitoring_observations ADD COLUMN actions_taken TEXT")

    # Trigger definitions are recreated so an older development build cannot
    # leave stale trigger SQL behind.  Dropping a trigger does not touch data.
    conn.executescript("""
        DROP TRIGGER IF EXISTS trg_assignment_point_insert_timestamps;
        CREATE TRIGGER trg_assignment_point_insert_timestamps
        AFTER INSERT ON planter_assignment_points
        BEGIN
            UPDATE planter_assignment_points
               SET assigned_at = COALESCE(NEW.assigned_at, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                   status_changed_at = COALESCE(NEW.status_changed_at, strftime('%Y-%m-%dT%H:%M:%fZ', 'now'))
             WHERE id = NEW.id;
        END;

        DROP TRIGGER IF EXISTS trg_assignment_point_status_change;
        CREATE TRIGGER trg_assignment_point_status_change
        AFTER UPDATE OF status ON planter_assignment_points
        WHEN OLD.status IS NOT NEW.status
        BEGIN
            UPDATE planter_assignment_points
               SET status_changed_at = strftime('%Y-%m-%dT%H:%M:%fZ', 'now')
             WHERE id = NEW.id;

            UPDATE planting_events
               SET closed_at = COALESCE(closed_at, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                   closure_reason = 'completion_reversed'
             WHERE id = (
                 SELECT pe.id FROM planting_events pe
                 WHERE pe.assignment_point_id = NEW.id
                   AND pe.closed_at IS NULL
                 ORDER BY pe.planted_at DESC, pe.id DESC LIMIT 1
             )
               AND OLD.status = 'completed'
               AND NEW.status != 'completed';

            INSERT OR IGNORE INTO planting_events (
                source_key, planting_point_id, assignment_point_id, assignment_id,
                site_zone_id, planter_id, species, planted_at, point_num,
                latitude, longitude, source, inspection_interval_days
            )
            SELECT
                'assignment-point:' || NEW.id || ':' ||
                    COALESCE(NEW.completed_at, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')) || ':' ||
                    lower(hex(randomblob(4))),
                pp.id,
                NEW.id,
                pa.id,
                COALESCE(pa.site_zone_id, a.site_zone_id),
                pa.planter_id,
                COALESCE(NULLIF(TRIM(pa.species), ''), NULLIF(TRIM(a.species), '')),
                COALESCE(NEW.completed_at, strftime('%Y-%m-%dT%H:%M:%fZ', 'now')),
                pp.point_num,
                pp.latitude,
                pp.longitude,
                'field_completion',
                COALESCE(
                    (SELECT sz.inspection_interval_days FROM site_zones sz
                     WHERE sz.id = COALESCE(pa.site_zone_id, a.site_zone_id)),
                    30
                )
            FROM planter_assignments pa
            JOIN planting_points pp ON pp.id = NEW.planting_point_id
            JOIN analyses a ON a.id = pp.analysis_id
            WHERE pa.id = NEW.assignment_id
              AND NEW.status = 'completed';
        END;
    """)


def _migrate_schema(conn: Any):
    raise RuntimeError("Runtime schema migration was removed; run 'alembic upgrade head'.")
    """Apply additive schema migrations for existing local databases."""
    analyses_columns = _table_columns(conn, "analyses")
    if "original_image" not in analyses_columns:
        conn.execute("ALTER TABLE analyses ADD COLUMN original_image TEXT")
    if "visualization_image" not in analyses_columns:
        conn.execute("ALTER TABLE analyses ADD COLUMN visualization_image TEXT")
    if "analysis_detail_json" not in analyses_columns:
        conn.execute("ALTER TABLE analyses ADD COLUMN analysis_detail_json TEXT")
    if "canopy_area_m2" not in analyses_columns:
        conn.execute("ALTER TABLE analyses ADD COLUMN canopy_area_m2 REAL")
    if "canopy_coverage_pct" not in analyses_columns:
        conn.execute("ALTER TABLE analyses ADD COLUMN canopy_coverage_pct REAL")
    if "species" not in analyses_columns:
        conn.execute("ALTER TABLE analyses ADD COLUMN species TEXT")
    if "planting_distance_m" not in analyses_columns:
        conn.execute("ALTER TABLE analyses ADD COLUMN planting_distance_m REAL")
    if "site_zone_id" not in analyses_columns:
        conn.execute("ALTER TABLE analyses ADD COLUMN site_zone_id INTEGER REFERENCES site_zones(id) ON DELETE SET NULL")
    if "footprint_geojson" not in analyses_columns:
        conn.execute("ALTER TABLE analyses ADD COLUMN footprint_geojson TEXT")
    if "footprint_quality" not in analyses_columns:
        conn.execute("ALTER TABLE analyses ADD COLUMN footprint_quality TEXT")

    point_columns = _table_columns(conn, "planting_points")
    if "planted_at" not in point_columns:
        conn.execute("ALTER TABLE planting_points ADD COLUMN planted_at TEXT")
    if "planted_date" not in point_columns:
        conn.execute("ALTER TABLE planting_points ADD COLUMN planted_date TEXT")
    if "deleted_at" not in point_columns:
        conn.execute("ALTER TABLE planting_points ADD COLUMN deleted_at TEXT")
    if "deletion_reason" not in point_columns:
        conn.execute("ALTER TABLE planting_points ADD COLUMN deletion_reason TEXT")
    if "deletion_batch_id" not in point_columns:
        conn.execute("ALTER TABLE planting_points ADD COLUMN deletion_batch_id TEXT")
    if "deletion_color" not in point_columns:
        conn.execute("ALTER TABLE planting_points ADD COLUMN deletion_color TEXT")
    if "death_at" not in point_columns:
        conn.execute("ALTER TABLE planting_points ADD COLUMN death_at TEXT")
    if "death_reason" not in point_columns:
        conn.execute("ALTER TABLE planting_points ADD COLUMN death_reason TEXT")
    if "death_reason_category" not in point_columns:
        conn.execute("ALTER TABLE planting_points ADD COLUMN death_reason_category TEXT")
    if "death_notes" not in point_columns:
        conn.execute("ALTER TABLE planting_points ADD COLUMN death_notes TEXT")

    planter_columns = _table_columns(conn, "planters")
    if "username" not in planter_columns:
        conn.execute("ALTER TABLE planters ADD COLUMN username TEXT")
    if "password_hash" not in planter_columns:
        conn.execute("ALTER TABLE planters ADD COLUMN password_hash TEXT")
    if "last_login" not in planter_columns:
        conn.execute("ALTER TABLE planters ADD COLUMN last_login TEXT")

    assignment_columns = _table_columns(conn, "planter_assignments")
    if "species" not in assignment_columns:
        conn.execute("ALTER TABLE planter_assignments ADD COLUMN species TEXT")
    if "site_zone_id" not in assignment_columns:
        conn.execute("ALTER TABLE planter_assignments ADD COLUMN site_zone_id INTEGER REFERENCES site_zones(id) ON DELETE SET NULL")

    assignment_point_columns = _table_columns(conn, "planter_assignment_points")
    if "assigned_at" not in assignment_point_columns:
        conn.execute("ALTER TABLE planter_assignment_points ADD COLUMN assigned_at TEXT")
    if "status_changed_at" not in assignment_point_columns:
        conn.execute("ALTER TABLE planter_assignment_points ADD COLUMN status_changed_at TEXT")
    if "skip_reason" not in assignment_point_columns:
        conn.execute("ALTER TABLE planter_assignment_points ADD COLUMN skip_reason TEXT")

    _install_dashboard_schema(conn)

    # Legacy planters remain valid; new self-service registrations supply a
    # stable organization link from the existing organization registry.
    planter_columns = _table_columns(conn, "planters")
    if "organization_id" not in planter_columns:
        conn.execute(
            "ALTER TABLE planters ADD COLUMN organization_id INTEGER "
            "REFERENCES organizations(id) ON DELETE RESTRICT"
        )
    # A historical planter can be attributed defensibly when every linked
    # project-site assignment names the same organization. Ambiguous or
    # unassigned legacy accounts intentionally remain null.
    conn.execute("""
        UPDATE planters
           SET organization_id = (
               SELECT MIN(sz.organization_id)
               FROM planter_assignments pa
               JOIN site_zones sz ON sz.id = pa.site_zone_id
               WHERE pa.planter_id = planters.id
                 AND sz.organization_id IS NOT NULL
           )
         WHERE organization_id IS NULL
           AND 1 = (
               SELECT COUNT(DISTINCT sz.organization_id)
               FROM planter_assignments pa
               JOIN site_zones sz ON sz.id = pa.site_zone_id
               WHERE pa.planter_id = planters.id
                 AND sz.organization_id IS NOT NULL
           )
    """)
    # Replace only system-generated placeholders; LGU-authored site names are
    # never overwritten. New sites already default to the organization name.
    conn.execute("""
        UPDATE site_zones
           SET name = (
               SELECT o.name FROM organizations o
               WHERE o.id = site_zones.organization_id
           )
         WHERE organization_id IS NOT NULL
           AND name GLOB 'Project Site [0-9]*'
    """)

    death_record_columns = _table_columns(conn, "point_death_records")
    if "assignment_id" not in death_record_columns:
        conn.execute("ALTER TABLE point_death_records ADD COLUMN assignment_id INTEGER")
    if "planting_event_id" not in death_record_columns:
        conn.execute("ALTER TABLE point_death_records ADD COLUMN planting_event_id INTEGER REFERENCES planting_events(id) ON DELETE SET NULL")
    if "monitoring_observation_id" not in death_record_columns:
        conn.execute("ALTER TABLE point_death_records ADD COLUMN monitoring_observation_id INTEGER REFERENCES monitoring_observations(id) ON DELETE SET NULL")

    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_planters_username ON planters(username)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_planters_organization ON planters(organization_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_points_planted_date ON planting_points(planted_date)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_points_deleted ON planting_points(deleted_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_points_deletion_batch ON planting_points(deletion_batch_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_points_death_category ON planting_points(death_reason_category)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_analyses_project_site ON analyses(site_zone_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_assignments_project_site ON planter_assignments(site_zone_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_assignment_points_assigned_at ON planter_assignment_points(assigned_at)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_death_records_planting_event ON point_death_records(planting_event_id)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_death_records_monitoring_observation ON point_death_records(monitoring_observation_id)")
    conn.execute("CREATE UNIQUE INDEX IF NOT EXISTS idx_death_records_one_per_planting_event ON point_death_records(planting_event_id) WHERE planting_event_id IS NOT NULL")

    conn.execute("""
        UPDATE planter_assignment_points
           SET assigned_at = COALESCE(
                   assigned_at,
                   (SELECT pa.created_at FROM planter_assignments pa WHERE pa.id = planter_assignment_points.assignment_id)
               ),
               status_changed_at = COALESCE(
                   status_changed_at,
                   completed_at,
                   (SELECT pa.created_at FROM planter_assignments pa WHERE pa.id = planter_assignment_points.assignment_id)
               )
         WHERE assigned_at IS NULL OR status_changed_at IS NULL
    """)
    _backfill_planting_events(conn)
    conn.execute("""
        UPDATE planting_events
        SET inspection_interval_days = COALESCE(
            inspection_interval_days,
            (SELECT sz.inspection_interval_days FROM site_zones sz
             WHERE sz.id = planting_events.site_zone_id),
            ?
        )
        WHERE inspection_interval_days IS NULL
    """, (_DEFAULT_OPERATIONAL_INSPECTION_INTERVAL_DAYS,))
    for footprint_row in conn.execute("""
        SELECT id, center_lat, center_lon, coverage_w_m, coverage_h_m
        FROM analyses
        WHERE footprint_geojson IS NULL
    """).fetchall():
        footprint = _coverage_rectangle_geojson(
            footprint_row["center_lat"],
            footprint_row["center_lon"],
            footprint_row["coverage_w_m"],
            footprint_row["coverage_h_m"],
        )
        conn.execute(
            "UPDATE analyses SET footprint_geojson = ?, footprint_quality = ? WHERE id = ?",
            (
                json.dumps(footprint) if footprint else None,
                "approximate_coverage_rectangle" if footprint else "unavailable",
                int(footprint_row["id"]),
            ),
        )

    # Backfill: if the death history table was just created on an older DB
    # that already has dead points (death_at set on planting_points), copy
    # those into the history table so the mortality breakdown stays correct
    # after the schema upgrade. We only insert when there is NO existing
    # history row for that point, so this is safe to run repeatedly.
    conn.execute("""
        INSERT INTO point_death_records (
            planting_point_id, assignment_id, death_at, reason_category, reason_label,
            notes, planter_id, planter_name, species
        )
        SELECT
            pp.id,
            (SELECT pa.id
               FROM planter_assignment_points pap
               JOIN planter_assignments pa ON pa.id = pap.assignment_id
              WHERE pap.planting_point_id = pp.id
                AND pap.status = 'completed'
              ORDER BY pap.completed_at DESC, pap.id DESC
              LIMIT 1),
            pp.death_at,
            COALESCE(pp.death_reason_category, 'other'),
            pp.death_reason,
            pp.death_notes,
            (SELECT pa.planter_id
               FROM planter_assignment_points pap
               JOIN planter_assignments pa ON pa.id = pap.assignment_id
              WHERE pap.planting_point_id = pp.id
                AND pap.status = 'completed'
              ORDER BY pap.completed_at DESC, pap.id DESC
              LIMIT 1),
            (SELECT pl.full_name
               FROM planter_assignment_points pap
               JOIN planter_assignments pa ON pa.id = pap.assignment_id
               JOIN planters pl ON pl.id = pa.planter_id
              WHERE pap.planting_point_id = pp.id
                AND pap.status = 'completed'
              ORDER BY pap.completed_at DESC, pap.id DESC
              LIMIT 1),
            (SELECT pa.species
               FROM planter_assignment_points pap
               JOIN planter_assignments pa ON pa.id = pap.assignment_id
              WHERE pap.planting_point_id = pp.id
                AND pap.status = 'completed'
              ORDER BY pap.completed_at DESC, pap.id DESC
              LIMIT 1)
        FROM planting_points pp
        WHERE pp.death_at IS NOT NULL
          AND pp.deleted_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM point_death_records pdr
              WHERE pdr.planting_point_id = pp.id
          )
    """)

    # If existing history rows predate the assignment_id column, backfill it
    # from the latest completed assignment that referenced the point. This
    # lets the auto-generated site-zone mortality stats attribute deaths to
    # the right assignment for historical data too.
    conn.execute("""
        UPDATE point_death_records
           SET assignment_id = (
               SELECT pa.id
                 FROM planter_assignment_points pap
                 JOIN planter_assignments pa ON pa.id = pap.assignment_id
                WHERE pap.planting_point_id = point_death_records.planting_point_id
                  AND pap.status = 'completed'
                ORDER BY pap.completed_at DESC, pap.id DESC
                LIMIT 1
           )
         WHERE assignment_id IS NULL
    """)
    legacy_deaths = conn.execute("""
        SELECT id, planting_point_id, assignment_id, death_at
        FROM point_death_records
        WHERE planting_event_id IS NULL
        ORDER BY death_at, id
    """).fetchall()
    for legacy_death in legacy_deaths:
        event_row = conn.execute("""
            SELECT pe.id
            FROM planting_events pe
            WHERE pe.planting_point_id = ?
              AND (? IS NULL OR pe.assignment_id = ?)
              AND pe.planted_at <= ?
              AND NOT EXISTS (
                  SELECT 1 FROM point_death_records used
                  WHERE used.planting_event_id = pe.id
              )
            ORDER BY pe.planted_at DESC, pe.id DESC
            LIMIT 1
        """, (
            legacy_death["planting_point_id"],
            legacy_death["assignment_id"],
            legacy_death["assignment_id"],
            legacy_death["death_at"],
        )).fetchone()
        if event_row:
            conn.execute(
                "UPDATE point_death_records SET planting_event_id = ? WHERE id = ?",
                (event_row["id"], legacy_death["id"]),
            )
    conn.execute("""
        UPDATE planting_events
           SET closed_at = COALESCE(
                   closed_at,
                   (SELECT MAX(pdr.death_at)
                    FROM point_death_records pdr
                    WHERE pdr.planting_event_id = planting_events.id)
               ),
               closure_reason = COALESCE(closure_reason, 'recorded_death')
         WHERE EXISTS (
             SELECT 1 FROM point_death_records pdr
             WHERE pdr.planting_event_id = planting_events.id
         )
    """)
    conn.execute("""
        UPDATE planting_points
        SET
            planted_at = (
                SELECT MAX(pap.completed_at)
                FROM planter_assignment_points pap
                WHERE pap.planting_point_id = planting_points.id
                  AND pap.status = 'completed'
                  AND pap.completed_at IS NOT NULL
            ),
            planted_date = substr((
                SELECT MAX(pap.completed_at)
                FROM planter_assignment_points pap
                WHERE pap.planting_point_id = planting_points.id
                  AND pap.status = 'completed'
                  AND pap.completed_at IS NOT NULL
            ), 1, 10)
        WHERE status = 'planted'
          AND planted_at IS NULL
          AND EXISTS (
              SELECT 1
              FROM planter_assignment_points pap
              WHERE pap.planting_point_id = planting_points.id
                AND pap.status = 'completed'
                AND pap.completed_at IS NOT NULL
          )
    """)
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
    _cleanup_same_species_spacing_conflicts(conn)
    _cleanup_cross_species_spacing_conflicts(conn)
    conn.commit()


def init_db():
    raise RuntimeError("Runtime schema creation was removed; run 'alembic upgrade head'.")
    """Explicitly create tables (called on import)."""
    conn = _get_connection()
    conn.close()


# ====================================================================
#  User Management
# ====================================================================

def _hash_password(password: str) -> str:
    """Return an Argon2id password hash."""
    return hash_password(password)


def _hash_session_token(token: str) -> str:
    """Return a stable hash for a browser session token."""
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


def _create_session(subject_type: str, subject_id: int, participant_slot: Optional[int] = None, *, connection=None) -> str:
    """Create a persisted session token for a user or planter."""
    if subject_type not in {"user", "planter"}:
        raise ValueError("Invalid auth session subject type.")

    token = secrets.token_urlsafe(32)
    token_hash = _hash_session_token(token)
    now = datetime.now(timezone.utc)
    settings = get_settings()
    lifetime = (
        timedelta(hours=settings.staff_session_hours)
        if subject_type == "user"
        else timedelta(days=settings.planter_session_days)
    )

    conn = connection if connection is not None else _get_connection()
    try:
        conn.execute("""
            INSERT INTO auth_sessions (
                subject_type, subject_id, token_hash, created_at, expires_at, last_seen_at, participant_slot
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (subject_type, int(subject_id), token_hash, now, now + lifetime, now, participant_slot))
        organization_id = None
        if subject_type == "planter":
            planter = conn.execute("SELECT organization_id FROM planters WHERE id = ?", (int(subject_id),)).fetchone()
            organization_id = planter["organization_id"] if planter else None
        append_activity(
            conn, action="auth.signed_in",
            actor_type="staff" if subject_type == "user" else "planter",
            actor_user_id=int(subject_id) if subject_type == "user" else None,
            actor_planter_id=int(subject_id) if subject_type == "planter" else None,
            participant_slot=participant_slot,
            organization_id=organization_id,
            summary="LGU staff signed in." if subject_type == "user" else "Organization participant signed in.",
        )
        if connection is None:
            conn.commit()
    except Exception:
        if connection is None:
            conn.rollback()
        raise
    finally:
        if connection is None:
            conn.close()
    return token


def _get_subject_by_session(subject_type: str, token: str) -> Optional[dict]:
    """Return the matching user or planter for an active session token."""
    if subject_type not in {"user", "planter"}:
        raise ValueError("Invalid auth session subject type.")
    if not token or token == "cookie":
        token = staff_session_token.get() if subject_type == "user" else planter_session_token.get()
    token = (token or "").strip()
    if not token:
        return None

    token_hash = _hash_session_token(token)
    conn = _get_connection()
    try:
        # Validate revocation/expiry on every request, using one read. Updating
        # the same session row on every tile/workspace read caused lock queues
        # and exhausted the small hosted database pool under concurrent use.
        if subject_type == "user":
            subject_columns = "subject.*"
            subject_join = "JOIN users subject ON subject.id = session.subject_id"
        else:
            subject_columns = "subject.*, organization.name AS organization_name"
            subject_join = """JOIN planters subject ON subject.id = session.subject_id
                LEFT JOIN organizations organization ON organization.id = subject.organization_id"""
        row = conn.execute(f"""
            SELECT {subject_columns}, session.id AS _auth_session_id,
                   session.participant_slot AS _auth_participant_slot,
                   (session.last_seen_at IS NULL OR session.last_seen_at
                       < CURRENT_TIMESTAMP - INTERVAL '5 minutes') AS _auth_heartbeat_due
            FROM auth_sessions session
            {subject_join}
            WHERE session.subject_type = ? AND session.token_hash = ?
              AND session.revoked_at IS NULL AND session.expires_at > CURRENT_TIMESTAMP
        """, (subject_type, token_hash)).fetchone()
        if not row:
            return None
        result = dict(row)
        session_id = result.pop("_auth_session_id")
        participant_slot = result.pop("_auth_participant_slot")
        # Compare timestamps in PostgreSQL: CompatRow converts them to strings.
        # The boolean is preserved and uses the DB clock consistently.
        heartbeat_due = result.pop("_auth_heartbeat_due")
        if heartbeat_due:
            conn.execute("""
                UPDATE auth_sessions SET last_seen_at = CURRENT_TIMESTAMP
                WHERE id = ? AND (last_seen_at IS NULL
                    OR last_seen_at < CURRENT_TIMESTAMP - INTERVAL '5 minutes')
            """, (session_id,))
            conn.commit()
        if subject_type == "planter":
            result["participant_slot"] = participant_slot
        return result
    finally:
        conn.close()


def _revoke_session(subject_type: str, token: str) -> None:
    """Revoke a persisted session token."""
    if subject_type not in {"user", "planter"}:
        raise ValueError("Invalid auth session subject type.")
    if not token or token == "cookie":
        token = staff_session_token.get() if subject_type == "user" else planter_session_token.get()
    token = (token or "").strip()
    if not token:
        return

    conn = _get_connection()
    try:
        revoked = conn.execute("""
            UPDATE auth_sessions
            SET revoked_at = ?
            WHERE subject_type = ?
              AND token_hash = ?
              AND revoked_at IS NULL
            RETURNING subject_id, participant_slot
        """, (
            datetime.now(timezone.utc),
            subject_type,
            _hash_session_token(token),
        )).fetchone()
        if revoked:
            organization_id = None
            if subject_type == "planter":
                planter = conn.execute("SELECT organization_id FROM planters WHERE id = ?", (revoked["subject_id"],)).fetchone()
                organization_id = planter["organization_id"] if planter else None
            append_activity(
                conn, action="auth.signed_out",
                actor_type="staff" if subject_type == "user" else "planter",
                actor_user_id=revoked["subject_id"] if subject_type == "user" else None,
                actor_planter_id=revoked["subject_id"] if subject_type == "planter" else None,
                participant_slot=revoked["participant_slot"],
                organization_id=organization_id,
                summary="LGU staff signed out." if subject_type == "user" else "Organization participant signed out.",
            )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
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
    """Return an existing staff account; never create default credentials."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT id FROM users ORDER BY CASE WHEN role IN ('admin', 'lgu') THEN 0 ELSE 1 END, id LIMIT 1"
    ).fetchone()
    conn.close()
    if not row:
        raise RuntimeError(
            "No staff account exists. Run scripts/bootstrap_admin.py with explicit credentials."
        )
    return int(row["id"])


def authenticate_user(username: str, password: str, *, record_login: bool = True) -> Optional[dict]:
    """Authenticate using full_name + password. Returns user dict on success."""
    if not username or not password:
        return None

    conn = _get_connection()
    row = conn.execute(
        "SELECT * FROM users WHERE lower(full_name) = lower(?)",
        (username.strip(),),
    ).fetchone()

    if not row:
        conn.close()
        return None

    user = dict(row)
    stored_hash = user.get('password_hash') or ''
    if not stored_hash:
        conn.close()
        return None

    valid, upgraded_hash = verify_password(stored_hash, password)
    if valid:
        updated = conn.execute(
            "UPDATE users SET password_hash = ?, last_login = CASE WHEN ? THEN CURRENT_TIMESTAMP ELSE last_login END WHERE id = ? AND password_hash = ?",
            (upgraded_hash or stored_hash, record_login, user["id"], stored_hash),
        )
        if not updated.rowcount:
            # A concurrent recovery must not be overwritten by this old login.
            conn.rollback()
            conn.close()
            return None
        conn.commit()
        conn.close()
        user["password_hash"] = upgraded_hash or stored_hash
        return user

    conn.close()
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
        """
        SELECT p.*, o.name AS organization_name
        FROM planters p
        LEFT JOIN organizations o ON o.id = p.organization_id
        WHERE lower(p.username) = lower(?)
        """,
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

    valid, upgraded_hash = verify_password(stored_hash, password)
    if valid:
        if upgraded_hash:
            conn = _get_connection()
            conn.execute(
                "UPDATE planters SET password_hash = ?, last_login = CURRENT_TIMESTAMP WHERE id = ?",
                (upgraded_hash, planter["id"]),
            )
            conn.commit()
            conn.close()
            planter["password_hash"] = upgraded_hash
        return planter

    return None


def create_planter_session(planter_id: int, participant_slot: Optional[int] = None) -> str:
    """Create a persisted session token for a planter."""
    return _create_session("planter", planter_id, participant_slot)


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
    """Return an existing staff user; never create an implicit account."""
    conn = _get_connection()
    row = conn.execute(
        "SELECT id FROM users ORDER BY id LIMIT 1"
    ).fetchone()
    conn.close()
    if not row:
        raise RuntimeError("No staff account exists; bootstrap an administrator first.")
    return int(row["id"])


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

# Proximity threshold for point dedup at save time. Must stay *below* the
# species' nearest-neighbour distance T, otherwise legitimate in-lattice
# neighbours look like duplicates and every other point in a saved grid gets
# silently dropped. Effective radius = max(_DEDUP_RADIUS_M, hexagon_size *
# _POINT_SPACING_FACTOR).
#   - hexagon_size = T / sqrt(3) (e.g. 0.577 m for bungalon, 1.155 m for
#     rhizophora; see _resolve_hexagon_size in api/routes/processing.py).
#   - Factor 0.5 → ~0.29 × T, well clear of the in-lattice spacing.
#   - Floor 0.5 m catches truly-coincident points from re-runs of the same
#     image, where sub-pixel jitter can offset two saves by < 10 cm.
# Previous value of 1.90 made the dedup radius 1.097 m for bungalon (target
# spacing 1.0 m), which killed roughly every other point at save time —
# leading to a saved grid that looked like 2 m spacing for bungalon.
_DEDUP_RADIUS_M = 0.5
_POINT_SPACING_FACTOR = 0.5
_SAME_SPECIES_DEDUP_SPACING_RATIO = 0.92
_M_PER_DEG_LAT = 111_320.0
_SPECIES_SPACING_M = {"bungalon": 1.0, "rhizophora": 2.0}
_CROSS_SPECIES_MIN_SPACING_M = 2.0

# ~15 m radius for matching an analysis to the same area
_ANALYSIS_MATCH_DEG = 0.00015  # ~15 m at equator


class _NearbyPointIndex:
    """Spatial hash for ~0.5 m proximity dedup of planting points.

    The previous implementation rounded lat/lon to 7 decimal places (~1 cm)
    and used set-membership, which only caught duplicates that landed on
    *exactly* the same sub-cm coordinate — useless for two analyses whose
    grids are offset by 10-50 cm. This index buckets points into cells of
    ~radius_m so a point matches if any neighbour cell holds a point within
    the real Haversine-equivalent distance.
    """

    def __init__(self, radius_m: float, ref_lat: float):
        self._radius_m = radius_m
        self._radius_m_sq = radius_m * radius_m
        cos_lat = max(0.2, math.cos(math.radians(ref_lat)))
        self._m_per_deg_lon = _M_PER_DEG_LAT * cos_lat
        # One cell ≈ one radius wide so neighbour search is at most 9 cells.
        self._cell_deg_lat = radius_m / _M_PER_DEG_LAT
        self._cell_deg_lon = radius_m / self._m_per_deg_lon
        self._cells: dict[tuple[int, int], list[tuple[float, float]]] = {}

    def _cell_key(self, lat: float, lon: float) -> tuple[int, int]:
        return (int(lat / self._cell_deg_lat), int(lon / self._cell_deg_lon))

    def add(self, lat: float, lon: float) -> None:
        self._cells.setdefault(self._cell_key(lat, lon), []).append((lat, lon))

    def has_neighbor(self, lat: float, lon: float) -> bool:
        cy, cx = self._cell_key(lat, lon)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cell = self._cells.get((cy + dy, cx + dx))
                if not cell:
                    continue
                for ex_lat, ex_lon in cell:
                    d_lat_m = (lat - ex_lat) * _M_PER_DEG_LAT
                    d_lon_m = (lon - ex_lon) * self._m_per_deg_lon
                    if d_lat_m * d_lat_m + d_lon_m * d_lon_m <= self._radius_m_sq:
                        return True
        return False


class _SpeciesSpacingIndex:
    """Spatial hash for saved points that can conflict with a new species run."""

    def __init__(self, radius_m: float, ref_lat: float):
        self._radius_m = radius_m
        self._radius_m_sq = radius_m * radius_m
        cos_lat = max(0.2, math.cos(math.radians(ref_lat)))
        self._m_per_deg_lon = _M_PER_DEG_LAT * cos_lat
        self._cell_deg_lat = radius_m / _M_PER_DEG_LAT
        self._cell_deg_lon = radius_m / self._m_per_deg_lon
        self._cells: dict[tuple[int, int], list[tuple[float, float, Optional[str], Optional[float]]]] = {}

    def _cell_key(self, lat: float, lon: float) -> tuple[int, int]:
        return (int(lat / self._cell_deg_lat), int(lon / self._cell_deg_lon))

    def add(
        self,
        lat: float,
        lon: float,
        species: Optional[str],
        planting_distance_m: Optional[float],
    ) -> None:
        self._cells.setdefault(self._cell_key(lat, lon), []).append(
            (lat, lon, _normalize_species_key(species), _species_spacing_m(species, planting_distance_m))
        )

    def has_conflict(
        self,
        lat: float,
        lon: float,
        species: Optional[str],
        planting_distance_m: Optional[float],
    ) -> bool:
        current_species = _normalize_species_key(species)
        current_spacing_m = _species_spacing_m(current_species, planting_distance_m)
        if current_species is None and current_spacing_m is None:
            return False

        cy, cx = self._cell_key(lat, lon)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                cell = self._cells.get((cy + dy, cx + dx))
                if not cell:
                    continue
                for ex_lat, ex_lon, ex_species, ex_spacing_m in cell:
                    threshold_m = _spacing_threshold_between_species(
                        current_species,
                        current_spacing_m,
                        ex_species,
                        ex_spacing_m,
                    )
                    if threshold_m is None:
                        continue
                    d_lat_m = (lat - ex_lat) * _M_PER_DEG_LAT
                    d_lon_m = (lon - ex_lon) * self._m_per_deg_lon
                    if d_lat_m * d_lat_m + d_lon_m * d_lon_m < threshold_m * threshold_m:
                        return True
        return False


def _normalize_species_key(species: Optional[str]) -> Optional[str]:
    key = (species or "").strip().lower()
    return key if key in _SPECIES_SPACING_M else None


def _species_spacing_m(
    species: Optional[str],
    fallback_distance_m: Optional[float] = None,
) -> Optional[float]:
    try:
        if fallback_distance_m is not None:
            distance = float(fallback_distance_m)
            if distance > 0:
                return distance
    except (TypeError, ValueError):
        pass
    key = _normalize_species_key(species)
    return _SPECIES_SPACING_M.get(key)


def _spacing_threshold_between_species(
    current_species: Optional[str],
    current_spacing_m: Optional[float],
    existing_species: Optional[str],
    existing_spacing_m: Optional[float],
) -> Optional[float]:
    current_species = _normalize_species_key(current_species)
    existing_species = _normalize_species_key(existing_species)

    # Same-species rows already use the small duplicate guard. Do not apply the
    # full biological spacing here, because exact 1 m / 2 m lattice neighbours
    # from the same species must remain valid.
    if current_species is not None and current_species == existing_species:
        return None

    distances = [
        distance
        for distance in (current_spacing_m, existing_spacing_m)
        if distance is not None and distance > 0
    ]
    if not distances:
        return None
    return max(_CROSS_SPECIES_MIN_SPACING_M, max(distances))


def _dedup_radius_for_results(results: dict) -> float:
    species_key = _normalize_species_key(results.get('species'))
    species_spacing_m = _species_spacing_m(
        species_key,
        results.get('planting_distance_m'),
    )
    if species_spacing_m is not None and species_spacing_m > 0:
        return max(_DEDUP_RADIUS_M, species_spacing_m * _SAME_SPECIES_DEDUP_SPACING_RATIO)

    try:
        planting_radius_m = float(results.get('hexagon_size_m') or 0)
    except (TypeError, ValueError):
        planting_radius_m = 0.0
    return max(_DEDUP_RADIUS_M, planting_radius_m * _POINT_SPACING_FACTOR)


def _existing_point_index(conn, lat_min, lat_max, lon_min, lon_max, radius_m: float) -> _NearbyPointIndex:
    """Build a spatial index of saved points inside the given bbox."""
    rows = conn.execute("""
        SELECT latitude, longitude FROM planting_points
        WHERE latitude  BETWEEN ? AND ?
          AND longitude BETWEEN ? AND ?
          AND deleted_at IS NULL
    """, (lat_min, lat_max, lon_min, lon_max)).fetchall()
    ref_lat = (lat_min + lat_max) / 2.0
    index = _NearbyPointIndex(radius_m, ref_lat)
    for r in rows:
        index.add(r['latitude'], r['longitude'])
    return index


def _existing_species_spacing_index(
    conn,
    lat_min,
    lat_max,
    lon_min,
    lon_max,
    radius_m: float,
) -> _SpeciesSpacingIndex:
    """Build a spatial index for enforcing species-to-species field spacing."""
    rows = conn.execute("""
        SELECT
            pp.latitude,
            pp.longitude,
            a.species,
            a.planting_distance_m
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pp.latitude  BETWEEN ? AND ?
          AND pp.longitude BETWEEN ? AND ?
          AND pp.deleted_at IS NULL
    """, (lat_min, lat_max, lon_min, lon_max)).fetchall()
    ref_lat = (lat_min + lat_max) / 2.0
    index = _SpeciesSpacingIndex(radius_m, ref_lat)
    for row in rows:
        index.add(
            row['latitude'],
            row['longitude'],
            row['species'],
            row['planting_distance_m'],
        )
    return index


def _planting_spacing_indexes(conn, hexagons: list, results: dict):
    """Use identical saved-point bounds and thresholds for preview and save."""
    species = _normalize_species_key(results.get('species'))
    spacing = _species_spacing_m(species, results.get('planting_distance_m'))
    dedup_radius = _dedup_radius_for_results(results)
    cross_radius = max(_CROSS_SPECIES_MIN_SPACING_M, spacing or 0.0)
    coordinates = [(h.get('_gps_lat'), h.get('_gps_lon')) for h in hexagons
                   if h.get('_gps_lat') is not None and h.get('_gps_lon') is not None]
    if not coordinates:
        return _NearbyPointIndex(dedup_radius, 0.0), _SpeciesSpacingIndex(cross_radius, 0.0)
    lats, lons = zip(*coordinates)
    pad = (max(dedup_radius, cross_radius) / _M_PER_DEG_LAT) * 2.0
    bounds = (min(lats)-pad, max(lats)+pad, min(lons)-pad, max(lons)+pad)
    return (
        _existing_point_index(conn, *bounds, dedup_radius),
        _existing_species_spacing_index(conn, *bounds, cross_radius),
    )


def filter_new_planting_hexagons(hexagons: list, results: dict) -> tuple[list, list]:
    """Return only points save would accept, before rendering/exporting a preview.

    Includes planned and planted locations, regardless of species. Database
    failures propagate so an unchecked preview cannot advertise occupied spots.
    This is read-only; no planting records or analyses are changed.
    """
    if not hexagons:
        return [], []
    conn = _get_connection()
    try:
        nearby, species_index = _planting_spacing_indexes(conn, hexagons, results)
    finally:
        conn.close()
    species = _normalize_species_key(results.get('species'))
    spacing = _species_spacing_m(species, results.get('planting_distance_m'))
    kept, filtered = [], []
    for hexagon in hexagons:
        lat, lon = hexagon.get('_gps_lat'), hexagon.get('_gps_lon')
        if lat is not None and lon is not None:
            if nearby.has_neighbor(lat, lon) or species_index.has_conflict(lat, lon, species, spacing):
                filtered.append(hexagon)
                continue
            nearby.add(lat, lon)
            species_index.add(lat, lon, species, spacing)
        kept.append(hexagon)
    return kept, filtered


def _local_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    ref_lat = (float(lat1) + float(lat2)) / 2.0
    m_per_deg_lon = _M_PER_DEG_LAT * max(0.2, math.cos(math.radians(ref_lat)))
    d_lat_m = (float(lat1) - float(lat2)) * _M_PER_DEG_LAT
    d_lon_m = (float(lon1) - float(lon2)) * m_per_deg_lon
    return math.sqrt((d_lat_m * d_lat_m) + (d_lon_m * d_lon_m))


def _cleanup_same_species_spacing_conflicts(conn: Any) -> int:
    """Soft-delete unassigned planned points that are too close to same-species points."""
    rows = conn.execute("""
        SELECT
            pp.id,
            pp.analysis_id,
            pp.latitude,
            pp.longitude,
            pp.status,
            LOWER(COALESCE(a.species, '')) AS species,
            a.planting_distance_m,
            a.analyzed_at,
            EXISTS (
                SELECT 1
                FROM planter_assignment_points pap
                JOIN planter_assignments pa ON pa.id = pap.assignment_id
                WHERE pap.planting_point_id = pp.id
                  AND pa.status = 'active'
            ) AS has_active_assignment
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pp.deleted_at IS NULL
          AND pp.latitude IS NOT NULL
          AND pp.longitude IS NOT NULL
          AND LOWER(COALESCE(a.species, '')) IN ('bungalon', 'rhizophora')
    """).fetchall()
    if len(rows) < 2:
        return 0

    def row_priority(row) -> tuple[int, str, int]:
        protected = bool(row["has_active_assignment"]) or row["status"] != "planned"
        return (0 if protected else 1, str(row["analyzed_at"] or ""), int(row["id"]))

    kept_rows = []
    delete_ids: set[int] = set()
    affected_analysis_ids: set[int] = set()

    for row in sorted(rows, key=row_priority):
        species = _normalize_species_key(row["species"])
        spacing_m = _species_spacing_m(species, row["planting_distance_m"])
        if spacing_m is None or spacing_m <= 0:
            kept_rows.append(row)
            continue

        threshold_m = max(_DEDUP_RADIUS_M, spacing_m * _SAME_SPECIES_DEDUP_SPACING_RATIO)
        threshold_sq = threshold_m * threshold_m
        ref_lat = float(row["latitude"])
        ref_lon = float(row["longitude"])
        conflict = False

        for kept in kept_rows:
            if _normalize_species_key(kept["species"]) != species:
                continue
            kept_spacing_m = _species_spacing_m(species, kept["planting_distance_m"])
            kept_threshold_m = max(
                _DEDUP_RADIUS_M,
                min(spacing_m, kept_spacing_m or spacing_m) * _SAME_SPECIES_DEDUP_SPACING_RATIO,
            )
            effective_threshold_sq = min(threshold_sq, kept_threshold_m * kept_threshold_m)
            ref_mid_lat = (ref_lat + float(kept["latitude"])) / 2.0
            m_per_deg_lon = _M_PER_DEG_LAT * max(0.2, math.cos(math.radians(ref_mid_lat)))
            d_lat_m = (ref_lat - float(kept["latitude"])) * _M_PER_DEG_LAT
            d_lon_m = (ref_lon - float(kept["longitude"])) * m_per_deg_lon
            if (d_lat_m * d_lat_m) + (d_lon_m * d_lon_m) < effective_threshold_sq:
                conflict = True
                break

        if conflict and row["status"] == "planned" and not row["has_active_assignment"]:
            delete_ids.add(int(row["id"]))
            affected_analysis_ids.add(int(row["analysis_id"]))
            continue

        kept_rows.append(row)

    if not delete_ids:
        return 0

    deleted_at = datetime.now().isoformat(timespec="seconds")
    batch_id = f"same-species-spacing-{deleted_at}"
    placeholders = ",".join("?" for _ in delete_ids)
    conn.execute(
        f"""
        UPDATE planting_points
           SET deleted_at = ?,
               deletion_reason = ?,
               deletion_batch_id = ?,
               deletion_color = ?
         WHERE id IN ({placeholders})
        """,
        (
            deleted_at,
            "Automatically removed: same-species planting point was closer than the saved spacing.",
            batch_id,
            "#9ca3af",
            *sorted(delete_ids),
        ),
    )

    for analysis_id in affected_analysis_ids:
        conn.execute(
            """
            UPDATE analyses
               SET hexagon_count = (
                   SELECT COUNT(*)
                   FROM planting_points
                   WHERE analysis_id = ?
                     AND deleted_at IS NULL
               )
             WHERE id = ?
            """,
            (analysis_id, analysis_id),
        )

    return len(delete_ids)


def _cleanup_cross_species_spacing_conflicts(conn: Any) -> int:
    """Soft-delete planned Rhizophora points that violate the 2 m species gap."""
    rows = conn.execute("""
        SELECT
            pp.id,
            pp.analysis_id,
            pp.latitude,
            pp.longitude,
            pp.status,
            LOWER(COALESCE(a.species, '')) AS species,
            EXISTS (
                SELECT 1
                FROM planter_assignment_points pap
                JOIN planter_assignments pa ON pa.id = pap.assignment_id
                WHERE pap.planting_point_id = pp.id
                  AND pa.status = 'active'
            ) AS has_active_assignment
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pp.deleted_at IS NULL
          AND pp.latitude IS NOT NULL
          AND pp.longitude IS NOT NULL
          AND LOWER(COALESCE(a.species, '')) IN ('bungalon', 'rhizophora')
    """).fetchall()

    bungalon_rows = [row for row in rows if row["species"] == "bungalon"]
    rhizophora_rows = [
        row
        for row in rows
        if row["species"] == "rhizophora"
        and row["status"] == "planned"
        and not row["has_active_assignment"]
    ]
    if not bungalon_rows or not rhizophora_rows:
        return 0

    delete_ids: set[int] = set()
    affected_analysis_ids: set[int] = set()
    for yellow in rhizophora_rows:
        for green in bungalon_rows:
            if _local_distance_m(
                yellow["latitude"],
                yellow["longitude"],
                green["latitude"],
                green["longitude"],
            ) < _CROSS_SPECIES_MIN_SPACING_M:
                delete_ids.add(int(yellow["id"]))
                affected_analysis_ids.add(int(yellow["analysis_id"]))
                break

    if not delete_ids:
        return 0

    deleted_at = datetime.now().isoformat(timespec="seconds")
    batch_id = f"cross-species-spacing-{deleted_at}"
    placeholders = ",".join("?" for _ in delete_ids)
    conn.execute(
        f"""
        UPDATE planting_points
           SET deleted_at = ?,
               deletion_reason = ?,
               deletion_batch_id = ?,
               deletion_color = ?
         WHERE id IN ({placeholders})
        """,
        (
            deleted_at,
            "Automatically removed: Rhizophora point was within 2 m of a Bungalon point.",
            batch_id,
            "#db2777",
            *sorted(delete_ids),
        ),
    )

    for analysis_id in affected_analysis_ids:
        conn.execute(
            """
            UPDATE analyses
               SET hexagon_count = (
                   SELECT COUNT(*)
                   FROM planting_points
                   WHERE analysis_id = ?
                     AND deleted_at IS NULL
               )
             WHERE id = ?
            """,
            (analysis_id, analysis_id),
        )

    return len(delete_ids)


def _delete_analysis_rows(conn, analysis_id: int):
    """Remove only an unworked analysis; preserve all field/history evidence."""
    protected = conn.execute("""
        SELECT 1
        FROM planting_points pp
        WHERE pp.analysis_id = ?
          AND (
              EXISTS(
                  SELECT 1 FROM planter_assignment_points pap
                  WHERE pap.planting_point_id = pp.id
              )
              OR EXISTS(
                  SELECT 1 FROM planting_events pe
                  WHERE pe.planting_point_id = pp.id
              )
              OR EXISTS(
                  SELECT 1 FROM point_death_records pdr
                  WHERE pdr.planting_point_id = pp.id
              )
          )
        LIMIT 1
    """, (int(analysis_id),)).fetchone()
    if protected:
        raise ValueError(
            "This analysis has assignment, planting, monitoring, or mortality history and cannot be replaced or deleted."
        )
    conn.execute("""
        INSERT INTO object_cleanup_jobs (object_key, reason)
        SELECT object_key, 'analysis_deleted'
        FROM analysis_assets
        WHERE analysis_id = ?
        ON CONFLICT (object_key) DO NOTHING
    """, (int(analysis_id),))
    conn.execute("DELETE FROM analyses WHERE id = ?", (analysis_id,))


def _enqueue_asset_cleanup(assets: List[StoredAsset], reason: str) -> None:
    """Persist compensation work without masking the original save failure."""
    if not assets:
        return
    cleanup_conn = None
    try:
        cleanup_conn = _get_connection()
        for asset in assets:
            cleanup_conn.execute("""
                INSERT INTO object_cleanup_jobs (object_key, reason)
                VALUES (?, ?)
                ON CONFLICT (object_key) DO NOTHING
            """, (asset.object_key, reason))
        cleanup_conn.commit()
    except Exception:
        if cleanup_conn is not None:
            try:
                cleanup_conn.rollback()
            except Exception:
                pass
    finally:
        if cleanup_conn is not None:
            try:
                cleanup_conn.close()
            except Exception:
                pass


def get_analysis_asset_urls(analysis_id: int) -> dict:
    """Return short-lived private URLs for a saved analysis."""
    conn = _get_connection()
    try:
        rows = conn.execute("""
            SELECT kind, object_key, content_type, byte_size, sha256
            FROM analysis_assets
            WHERE analysis_id = ? AND lifecycle_state = 'ready'
            ORDER BY kind
        """, (int(analysis_id),)).fetchall()
    finally:
        conn.close()
    assets = {}
    for row in rows:
        assets[row["kind"]] = {
            "url": signed_download_url(row["object_key"]),
            "content_type": row["content_type"],
            "byte_size": int(row["byte_size"]),
            "sha256": row["sha256"],
            "expires_in_seconds": get_settings().s3_presigned_url_ttl_seconds,
        }
    return assets


def process_object_cleanup_jobs(limit: int = 50) -> dict:
    """Retry durable S3 cleanup work after database commits."""
    conn = _get_connection()
    rows = conn.execute("""
        SELECT id, object_key
        FROM object_cleanup_jobs
        WHERE completed_at IS NULL
        ORDER BY id
        LIMIT ?
    """, (max(1, min(int(limit), 500)),)).fetchall()
    completed = 0
    for row in rows:
        try:
            delete_object(row["object_key"])
            conn.execute("""
                UPDATE object_cleanup_jobs
                SET attempts = attempts + 1, last_error = NULL,
                    completed_at = CURRENT_TIMESTAMP
                WHERE id = ?
            """, (row["id"],))
            completed += 1
        except Exception as error:
            conn.execute("""
                UPDATE object_cleanup_jobs
                SET attempts = attempts + 1, last_error = ?
                WHERE id = ?
            """, (str(error)[:1000], row["id"]))
    conn.commit()
    conn.close()
    return {"requested": len(rows), "completed": completed}


def cleanup_abandoned_analysis_previews(older_than_hours: int = 24) -> dict:
    """Delete expired preview objects that were never attached to an analysis."""
    hours = max(1, min(int(older_than_hours), 24 * 30))
    cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
    candidates = stale_object_keys("analyses/previews/", cutoff)
    if not candidates:
        return {"candidates": 0, "deleted": 0}
    registered: set[str] = set()
    conn = _get_connection()
    try:
        for start in range(0, len(candidates), 500):
            batch = candidates[start:start + 500]
            placeholders = ", ".join("?" for _ in batch)
            rows = conn.execute(
                f"SELECT object_key FROM analysis_assets WHERE object_key IN ({placeholders})",
                tuple(batch),
            ).fetchall()
            registered.update(row["object_key"] for row in rows)
    finally:
        conn.close()
    deleted = 0
    for object_key in candidates:
        if object_key in registered:
            continue
        delete_object(object_key)
        deleted += 1
    return {"candidates": len(candidates), "deleted": deleted}


def _coverage_rectangle_geojson(
    center_lat: Optional[float],
    center_lon: Optional[float],
    coverage_w_m: Optional[float],
    coverage_h_m: Optional[float],
) -> Optional[dict]:
    """Return an approximate north-up WGS84 footprint for legacy analyses."""
    try:
        lat = float(center_lat)
        lon = float(center_lon)
        width = float(coverage_w_m)
        height = float(coverage_h_m)
    except (TypeError, ValueError):
        return None
    if width <= 0 or height <= 0 or not (-90 <= lat <= 90) or not (-180 <= lon <= 180):
        return None
    half_lat = (height / 2.0) / 111_320.0
    meters_per_lon = max(1.0, 111_320.0 * math.cos(math.radians(lat)))
    half_lon = (width / 2.0) / meters_per_lon
    ring = [
        [lon - half_lon, lat - half_lat],
        [lon + half_lon, lat - half_lat],
        [lon + half_lon, lat + half_lat],
        [lon - half_lon, lat + half_lat],
        [lon - half_lon, lat - half_lat],
    ]
    return {"type": "Polygon", "coordinates": [ring]}


def _analysis_footprint_values(
    results: dict,
    center_lat: Optional[float],
    center_lon: Optional[float],
) -> tuple[Optional[str], Optional[str]]:
    """Extract an authoritative footprint, or create an honest approximation."""
    candidate = results.get("footprint_geojson") or results.get("_footprint_geojson")
    if isinstance(candidate, str):
        try:
            candidate = json.loads(candidate)
        except (TypeError, json.JSONDecodeError):
            candidate = None
    if isinstance(candidate, dict):
        try:
            serialized = _validate_site_polygon_payload(candidate)
            return serialized, str(results.get("footprint_quality") or "projected").strip()[:80]
        except ValueError:
            pass
    coverage = results.get("coverage_m") or [None, None]
    if not isinstance(coverage, (list, tuple)) or len(coverage) < 2:
        coverage = [None, None]
    approximate = _coverage_rectangle_geojson(center_lat, center_lon, coverage[0], coverage[1])
    if approximate:
        return json.dumps(approximate), "approximate_coverage_rectangle"
    return None, "unavailable"


def _matching_project_site_id(
    conn: Any,
    latitude: Optional[float],
    longitude: Optional[float],
) -> Optional[int]:
    """Return the stable site containing the center only when unambiguous."""
    if latitude is None or longitude is None:
        return None
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return None
    row = conn.execute("""
        SELECT MIN(id) AS site_id, COUNT(*) AS match_count
        FROM project_sites
        WHERE extensions.ST_Covers(
            geometry,
            extensions.ST_SetSRID(extensions.ST_MakePoint(?, ?), 4326)
        )
    """, (lon, lat)).fetchone()
    return int(row["site_id"]) if row and int(row["match_count"]) == 1 else None


def _resolve_point_project_sites(
    conn: Any,
    point_rows: List[Any],
) -> List[dict]:
    """Attach an unambiguous stable project site to point-shaped rows.

    An analysis link narrows the candidate site, but each point must be
    covered by that site's saved boundary. Unlinked analyses use the
    unambiguous polygon containing the point.
    """
    site_by_id = {}
    for row in conn.execute("""
        SELECT sz.id, sz.name, sz.organization_id, o.name AS organization_name
        FROM site_zones sz
        LEFT JOIN organizations o ON o.id = sz.organization_id
    """).fetchall():
        site = dict(row)
        site_by_id[int(site["id"])] = site

    point_items = [dict(row) for row in point_rows]
    unresolved = []
    for index, point in enumerate(point_items):
        site_id = point.get("source_project_site_id", point.get("source_site_id"))
        if point.get("latitude") is not None and point.get("longitude") is not None:
            unresolved.append((index, float(point["longitude"]), float(point["latitude"]), site_id))
    resolved_site_ids: dict[int, int] = {}
    if unresolved:
        values_sql = ", ".join("(?, ?, ?, CAST(? AS bigint))" for _ in unresolved)
        params = tuple(value for item in unresolved for value in item)
        matches = conn.execute(f"""
            WITH input_points(row_key, longitude, latitude, linked_site_id) AS (
                VALUES {values_sql}
            ), matched AS (
                SELECT ip.row_key, MIN(ps.id) AS site_id, COUNT(ps.id) AS match_count
                FROM input_points ip
                LEFT JOIN project_sites ps
                  ON (ip.linked_site_id IS NULL OR ps.id = ip.linked_site_id)
                  AND extensions.ST_Covers(
                      ps.geometry,
                      extensions.ST_SetSRID(
                          extensions.ST_MakePoint(ip.longitude, ip.latitude), 4326
                      )
                  )
                GROUP BY ip.row_key
            )
            SELECT row_key, site_id
            FROM matched
            WHERE match_count = 1
        """, params).fetchall()
        resolved_site_ids = {int(row["row_key"]): int(row["site_id"]) for row in matches}

    resolved = []
    for index, point in enumerate(point_items):
        site_id = resolved_site_ids.get(index)
        site = site_by_id.get(int(site_id)) if site_id is not None else None
        point["source_project_site_id"] = int(site_id) if site is not None else None
        point["source_site_id"] = int(site_id) if site is not None else None
        point["source_project_site_name"] = site.get("name") if site else None
        point["source_site_name"] = site.get("name") if site else None
        point["source_organization_id"] = site.get("organization_id") if site else None
        point["source_organization_name"] = site.get("organization_name") if site else None
        resolved.append(point)
    return resolved


def save_analysis(
    image_name: str,
    center_lat: Optional[float],
    center_lon: Optional[float],
    results: dict,
    hexagons: list,
    user_id: Optional[int] = None,
    original_image: Optional[str] = None,
    visualization_image: Optional[str] = None,
    stored_assets: Optional[List[StoredAsset]] = None,
) -> Tuple[int, int, int]:
    """Persist one analysis without placing image bytes in PostgreSQL."""
    # Last guard before any storage upload, same-image replacement, or insert.
    # A stale preview made before a map/zone update must be reprocessed.
    if hexagons:
        from canopy_detection import ortho_matcher
        from canopy_detection.orthophoto_coverage import (
            point_visibility_flags, visible_gis_coverage,
        )
        from mangrovision_db.zones import feature_collection

        active_ortho = ortho_matcher._ensure_active_ortho()
        coverage = visible_gis_coverage(
            active_ortho, feature_collection("gis_coverage").get("features", []),
        )
        visible_flags = point_visibility_flags(
            active_ortho, coverage,
            [(item.get("_gps_lat"), item.get("_gps_lon")) for item in hexagons],
        )
        invalid_count = len(hexagons) - sum(visible_flags)
        if invalid_count:
            raise OutsideVisibleMapError(
                f"{invalid_count} planting point(s) fall outside the visible map. "
                "Run the analysis again before saving."
            )
    stored_assets = list(stored_assets or upload_analysis_data_urls(original_image, visualization_image))
    conn = _get_connection()
    try:
        result = _save_analysis_with_connection(
            conn,
            image_name,
            center_lat,
            center_lon,
            results,
            hexagons,
            user_id=user_id,
            original_image=None,
            visualization_image=None,
        )
        analysis_id = result[0]
        asset_conn = _get_connection()
        try:
            for asset in stored_assets:
                asset_conn.execute("""
                    INSERT INTO analysis_assets (
                        analysis_id, kind, object_key, content_type,
                        byte_size, sha256, lifecycle_state
                    ) VALUES (?, ?, ?, ?, ?, ?, 'ready')
                """, (
                    analysis_id,
                    asset.kind,
                    asset.object_key,
                    asset.content_type,
                    asset.byte_size,
                    asset.sha256,
                ))
            asset_conn.commit()
        except BaseException:
            asset_conn.rollback()
            cleanup_conn = _get_connection()
            try:
                _delete_analysis_rows(cleanup_conn, analysis_id)
                cleanup_conn.commit()
            finally:
                cleanup_conn.close()
            raise
        finally:
            asset_conn.close()
        return result
    except BaseException:
        # Reprocessing can be rejected after inspecting downstream assignment
        # or monitoring history.  Roll back any earlier same-image deletions
        # and, importantly on Windows, close the handle before the exception
        # reaches callers that may immediately remove a temporary database.
        try:
            conn.rollback()
        except DatabaseError:
            pass
        _enqueue_asset_cleanup(stored_assets, "analysis_save_failed")
        delete_assets(stored_assets)
        raise
    finally:
        try:
            conn.close()
        except DatabaseError:
            pass


def _save_analysis_with_connection(
    conn: Any,
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
    *other* analyses using the same approximate spacing as the active
    processing preview.

    Returns (analysis_id, new_points_count, skipped_duplicates_count).
    """
    cur = conn.cursor()
    footprint_geojson, footprint_quality = _analysis_footprint_values(
        results, center_lat, center_lon
    )
    explicit_site_id = results.get("site_zone_id")
    if explicit_site_id is not None:
        try:
            analysis_site_zone_id = int(explicit_site_id)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid project site id for analysis.") from error
        if not conn.execute("SELECT 1 FROM site_zones WHERE id = ?", (analysis_site_zone_id,)).fetchone():
            raise ValueError("Selected project site for analysis was not found.")
    else:
        analysis_site_zone_id = _matching_project_site_id(conn, center_lat, center_lon)

    # Default to system planner if no user specified
    if user_id is None:
        user_id = get_or_create_default_user()

    # ── Replace previous analysis only when the SAME image is re-run
    #    in the same area. Different images in the same area stay
    #    side-by-side so their unique points are preserved. Per-point
    #    dedup still prevents overlapping markers.
    analysis_number = None
    if center_lat is not None and center_lon is not None and image_name:
        old_rows = conn.execute("""
            SELECT id, analysis_number FROM analyses
            WHERE COALESCE(analysis_detail_json ->> 'source_image_name', image_name) = ?
              AND center_lat BETWEEN ? AND ?
              AND center_lon BETWEEN ? AND ?
        """, (
            image_name,
            center_lat - _ANALYSIS_MATCH_DEG, center_lat + _ANALYSIS_MATCH_DEG,
            center_lon - _ANALYSIS_MATCH_DEG, center_lon + _ANALYSIS_MATCH_DEG,
        )).fetchall()
        for row in old_rows:
            if analysis_number is None:
                analysis_number = row['analysis_number']
            _delete_analysis_rows(conn, row['id'])

    # ── Insert new analysis row ───────────────────────────────────
    # Preserve source identity for reprocessing independently of the saved name.
    detail = results.get('_analysis_detail_json') or {}
    if isinstance(detail, str):
        detail = json.loads(detail)
    detail = dict(detail)
    detail['source_image_name'] = image_name
    # Reprocessing the same saved photo keeps its display number. Only new
    # saved analyses consume this counter, independent of internal row IDs.
    if analysis_number is None:
        analysis_number = conn.execute(
            "SELECT nextval('mangrovision.analysis_number_seq') AS number"
        ).fetchone()['number']
    cur.execute("""
        INSERT INTO analyses
            (user_id, image_name, analysis_number, analyzed_at, center_lat, center_lon,
             altitude_m, gsd_cm, coverage_w_m, coverage_h_m,
             total_area_m2, canopy_count, canopy_area_m2, canopy_coverage_pct, polygon_count,
             danger_area_m2, danger_pct,
             plantable_area_m2, plantable_pct,
             hexagon_count, ai_confidence,
             canopy_buffer_m, hexagon_size_m,
             forbidden_filtered, eroded_filtered,
             analysis_detail_json,
             species, planting_distance_m, footprint_geojson, footprint_quality,
             site_zone_id)
        VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
    """, (
        user_id,
        f"Analysis {analysis_number}",
        analysis_number,
        _manila_now().isoformat(timespec='seconds'),
        center_lat,
        center_lon,
        results.get('altitude_m'),
        results.get('gsd_m_per_pixel', 0) * 100,
        results.get('coverage_m', [0, 0])[0],
        results.get('coverage_m', [0, 0])[1],
        results.get('total_area_m2', 0),
        results.get('canopy_count', 0),
        results.get('canopy_area_m2'),
        results.get('canopy_coverage_pct'),
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
        json.dumps(detail),
        results.get('species'),
        results.get('planting_distance_m'),
        footprint_geojson,
        footprint_quality,
        analysis_site_zone_id,
    ))
    analysis_id = cur.lastrowid

    # ── Deduplicate planting points against OTHER analyses ────────
    # Real distance check (≤ half the species spacing — see _POINT_SPACING_FACTOR
    # above) so a second analysis whose grid is offset by ~5-50 cm from an
    # existing one merges instead of double-stacking. The radius must stay
    # below the in-lattice nearest-neighbour distance, otherwise legitimate
    # adjacent points in a single saved analysis would mark each other as
    # duplicates.
    #
    # Cross-species spacing is stricter and separate: Rhizophora (2 m)
    # must not sit inside a Bungalon/green row, and vice versa. Use the larger
    # biological spacing between the two species while leaving same-species
    # lattice neighbours alone.
    _analysis_species = _normalize_species_key(results.get('species'))
    _analysis_spacing_m = _species_spacing_m(
        _analysis_species,
        results.get('planting_distance_m'),
    )
    _existing, _existing_species_spacing = _planting_spacing_indexes(conn, hexagons, results)

    new_count = 0
    skipped = 0
    for i, h in enumerate(hexagons, 1):
        _lat = h.get('_gps_lat')
        _lon = h.get('_gps_lon')

        if _lat is not None and _lon is not None and _existing.has_neighbor(_lat, _lon):
            skipped += 1
            continue

        if (
            _lat is not None
            and _lon is not None
            and _existing_species_spacing.has_conflict(
                _lat,
                _lon,
                _analysis_species,
                _analysis_spacing_m,
            )
        ):
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
        if _lat is not None and _lon is not None:
            _existing.add(_lat, _lon)
            _existing_species_spacing.add(
                _lat,
                _lon,
                _analysis_species,
                _analysis_spacing_m,
            )

    _cleanup_same_species_spacing_conflicts(conn)
    _cleanup_cross_species_spacing_conflicts(conn)
    count_row = conn.execute(
        """
        SELECT COUNT(*) AS cnt
        FROM planting_points
        WHERE analysis_id = ?
          AND deleted_at IS NULL
        """,
        (analysis_id,),
    ).fetchone()
    new_count = int(count_row["cnt"] if count_row else 0)

    # Update analysis row with actual inserted count
    cur.execute(
        "UPDATE analyses SET hexagon_count = ? WHERE id = ?",
        (new_count, analysis_id),
    )

    conn.commit()
    return analysis_id, new_count, skipped


# ====================================================================
#  Overlap / Nearby Detection
# ====================================================================

def _nearby_search_deltas(center_lat: float, radius_m: float) -> tuple[float, float]:
    lat_delta = float(radius_m) / 111_320.0
    lon_scale = 111_320.0 * max(0.2, math.cos(math.radians(float(center_lat))))
    return lat_delta, float(radius_m) / lon_scale


def _approx_distance_m(lat_a: float, lon_a: float, lat_b: float, lon_b: float) -> float:
    lat_mid = (float(lat_a) + float(lat_b)) * 0.5
    dy = (float(lat_a) - float(lat_b)) * 111_320.0
    dx = (float(lon_a) - float(lon_b)) * 111_320.0 * max(
        0.2,
        math.cos(math.radians(lat_mid)),
    )
    return math.hypot(dx, dy)


def find_overlapping_analyses(
    center_lat: float,
    center_lon: float,
    radius_m: float = 15.0,
) -> List[dict]:
    """Return past analyses with active planting points near this GPS centre."""
    conn = _get_connection()
    rows = conn.execute("""
        WITH requested AS (
            SELECT extensions.ST_SetSRID(extensions.ST_MakePoint(?, ?), 4326) AS location
        )
        SELECT a.id, a.image_name, a.analyzed_at, a.center_lat, a.center_lon,
               a.hexagon_count, a.plantable_area_m2, u.full_name AS planner_name
        FROM analyses a
        CROSS JOIN requested
        LEFT JOIN users u ON u.id = a.user_id
        WHERE a.center_location && extensions.ST_Expand(
                  requested.location, ? / 111320.0
              )
          AND extensions.ST_DWithin(
                  a.center_location::extensions.geography,
                  requested.location::extensions.geography,
                  ?
              )
          AND EXISTS (
              SELECT 1
              FROM planting_points pp
              WHERE pp.analysis_id = a.id
                AND pp.deleted_at IS NULL
          )
        ORDER BY a.analyzed_at DESC
    """, (
        float(center_lon), float(center_lat), float(radius_m), float(radius_m),
    )).fetchall()
    conn.close()
    return [dict(row) for row in rows]


def count_nearby_points(
    center_lat: float,
    center_lon: float,
    radius_m: float = 15.0,
) -> int:
    """Count planting points already saved near a GPS centre."""
    conn = _get_connection()
    row = conn.execute("""
        WITH requested AS (
            SELECT extensions.ST_SetSRID(extensions.ST_MakePoint(?, ?), 4326) AS location
        )
        SELECT COUNT(*) AS point_count
        FROM planting_points pp
        CROSS JOIN requested
        WHERE pp.deleted_at IS NULL
          AND pp.location && extensions.ST_Expand(requested.location, ? / 111320.0)
          AND extensions.ST_DWithin(
                  pp.location::extensions.geography,
                  requested.location::extensions.geography,
                  ?
              )
    """, (
        float(center_lon), float(center_lat), float(radius_m), float(radius_m),
    )).fetchone()
    conn.close()
    return int(row["point_count"] if row else 0)


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
          AND deleted_at IS NULL
    """, (lat_min, lat_max, lon_min, lon_max)).fetchall()
    conn.close()
    return [(float(r['latitude']), float(r['longitude'])) for r in rows]


def get_saved_species_point_locations(
    lat_min: float,
    lat_max: float,
    lon_min: float,
    lon_max: float,
) -> List[dict]:
    """Return saved point coordinates with analysis species metadata."""
    conn = _get_connection()
    rows = conn.execute("""
        SELECT
            pp.latitude,
            pp.longitude,
            a.species,
            a.planting_distance_m
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pp.latitude  BETWEEN ? AND ?
          AND pp.longitude BETWEEN ? AND ?
          AND pp.deleted_at IS NULL
    """, (lat_min, lat_max, lon_min, lon_max)).fetchall()
    conn.close()
    return [
        {
            "latitude": float(row["latitude"]),
            "longitude": float(row["longitude"]),
            "species": row["species"],
            "planting_distance_m": row["planting_distance_m"],
        }
        for row in rows
    ]


def _load_eroded_polygons() -> List:
    """Load active erosion polygons from the authoritative PostGIS table."""
    if Point is None or shape is None:
        return []
    now = datetime.now(timezone.utc)
    expires_at = _ERODED_ZONE_CACHE.get("expires_at")
    if expires_at is not None and expires_at > now:
        return _ERODED_ZONE_CACHE["polygons"]
    polygons = []
    conn = _get_connection()
    try:
        rows = conn.execute("""
            SELECT polygon_geojson
            FROM map_zones
            WHERE zone_type = 'eroded' AND deleted_at IS NULL
            ORDER BY id
        """).fetchall()
        for row in rows:
            geometry = json.loads(row["polygon_geojson"])
            polygon = shape(geometry)
            if polygon.is_empty:
                continue
            if not polygon.is_valid:
                polygon = polygon.buffer(0)
            if polygon.is_valid and not polygon.is_empty:
                polygons.append(polygon)
    except Exception:
        polygons = []
    finally:
        conn.close()
    _ERODED_ZONE_CACHE["polygons"] = polygons
    _ERODED_ZONE_CACHE["expires_at"] = now + timedelta(seconds=1)
    return polygons


def invalidate_eroded_zone_cache() -> None:
    _ERODED_ZONE_CACHE["expires_at"] = None
    _ERODED_ZONE_CACHE["polygons"] = []


def _is_point_inside_eroded_zone(latitude, longitude) -> bool:
    """Return whether PostGIS says the point is covered by active erosion."""
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return False
    conn = _get_connection()
    try:
        row = conn.execute("""
            SELECT EXISTS (
                SELECT 1
                FROM map_zones mz
                WHERE mz.zone_type = 'eroded'
                  AND mz.deleted_at IS NULL
                  AND extensions.ST_Covers(
                      mz.geometry,
                      extensions.ST_SetSRID(
                          extensions.ST_MakePoint(?, ?),
                          4326
                      )
                  )
            ) AS is_covered
        """, (lon, lat)).fetchone()
        return bool(row and row["is_covered"])
    finally:
        conn.close()


def _eroded_point_ids(rows) -> set[int]:
    """Resolve erosion coverage for saved points in one indexed PostGIS query."""
    point_ids = sorted({
        int(dict(row)["id"])
        for row in rows
        if dict(row).get("id") is not None and not dict(row).get("deleted_at")
    })
    if not point_ids:
        return set()
    placeholders = ", ".join("?" for _ in point_ids)
    conn = _get_connection()
    try:
        covered = conn.execute(f"""
            SELECT DISTINCT pp.id
            FROM planting_points pp
            JOIN map_zones mz
              ON mz.zone_type = 'eroded'
             AND mz.deleted_at IS NULL
             AND extensions.ST_Covers(mz.geometry, pp.location)
            WHERE pp.id IN ({placeholders})
              AND pp.deleted_at IS NULL
        """, tuple(point_ids)).fetchall()
        return {int(row["id"]) for row in covered}
    finally:
        conn.close()


def _warning_type_label(warning_type: str) -> str:
    clean = (warning_type or "planner_warning").strip().lower()
    return _WARNING_TYPE_LABELS.get(clean, clean.replace("_", " ").title())


def _clean_warning_type(warning_type: Optional[str]) -> str:
    clean = (warning_type or "planner_warning").strip().lower()
    clean = clean.replace("-", "_").replace(" ", "_")
    if not clean:
        return "planner_warning"
    return clean[:48]


def _clean_warning_severity(severity: Optional[str]) -> str:
    clean = (severity or "medium").strip().lower()
    return clean if clean in _WARNING_SEVERITY_RANK else "medium"


def _load_warning_zone_geometries() -> List[dict]:
    """Load DB-backed warning polygons as shapely geometries."""
    if Point is None or shape is None:
        return []

    conn = _get_connection()
    try:
        rows = conn.execute("""
            SELECT id, name, warning_type, severity, notes, polygon_geojson
            FROM warning_zones
            ORDER BY id
        """).fetchall()
    except DatabaseError:
        rows = []
    finally:
        conn.close()

    zones = []
    for row in rows:
        try:
            geom = shape(json.loads(row["polygon_geojson"]))
            if geom.is_empty:
                continue
            if not geom.is_valid:
                geom = geom.buffer(0)
            if not geom.is_valid or geom.is_empty:
                continue
            zones.append({
                "id": int(row["id"]),
                "name": row["name"] or f"Warning Zone {row['id']}",
                "warning_type": row["warning_type"] or "planner_warning",
                "severity": _clean_warning_severity(row["severity"]),
                "notes": row["notes"] or "",
                "geometry": geom,
            })
        except Exception:
            continue
    return zones


def _matching_warning_zones(latitude, longitude, warning_zones: List[dict]) -> List[dict]:
    if Point is None or not warning_zones:
        return []
    try:
        lat = float(latitude)
        lon = float(longitude)
    except (TypeError, ValueError):
        return []

    point = Point(lon, lat)
    matches = []
    for zone in warning_zones:
        geom = zone.get("geometry")
        if geom is None:
            continue
        try:
            if geom.contains(point) or geom.touches(point):
                matches.append(zone)
        except Exception:
            continue
    return matches


def _resolve_point_species(item: dict) -> tuple[Optional[str], Optional[float]]:
    """Return (species_key, planting_distance_m) for a point.

    Only trusts the analysis's saved species column. We intentionally do NOT
    guess legacy rows from hexagon_size — the old default (1.5 m) doesn't map
    cleanly onto either of the two new species (Bungalon 1 m / Rhizophora
    2 m), and guessing wrong would mis-color every pre-existing analysis.
    Legacy points just stay species=None and the map falls back to the
    generic 'planned' green.
    """
    raw_species = (item.get("analysis_species") or "").strip().lower() or None
    distance = item.get("analysis_planting_distance_m")
    SPECIES_DEFAULTS = {"bungalon": 1.0, "rhizophora": 2.0}
    if raw_species in SPECIES_DEFAULTS:
        return raw_species, float(distance) if distance is not None else SPECIES_DEFAULTS[raw_species]
    return None, (float(distance) if distance is not None else None)


def _annotate_point_advisories(rows, *, eroded_point_ids=None) -> List[dict]:
    """Add dynamic erosion availability and planner-warning fields."""
    annotated = []
    warning_zones = _load_warning_zone_geometries()
    if eroded_point_ids is None:
        eroded_point_ids = _eroded_point_ids(rows)
    for row in rows:
        item = dict(row)
        species_key, distance_m = _resolve_point_species(item)
        item["species"] = species_key
        item["planting_distance_m"] = distance_m
        is_deleted = bool(item.get("deleted_at"))
        item["is_deleted"] = is_deleted
        if is_deleted:
            item["inside_eroded_zone"] = False
            item["erosion_advisory"] = False
            item["eroded_unavailable"] = False
            item["availability_status"] = "deleted"
            item["availability_reason"] = item.get("deletion_reason") or "Marked deleted"
            item["survival_warning"] = False
            item["warning_zone_ids"] = []
            item["warning_zone_names"] = []
            item["warning_reasons"] = []
            item["warning_severity"] = None
            item["warning_summary"] = None
            annotated.append(item)
            continue

        inside_eroded_zone = int(item["id"]) in eroded_point_ids
        item["inside_eroded_zone"] = inside_eroded_zone
        item["erosion_advisory"] = inside_eroded_zone
        item["eroded_unavailable"] = inside_eroded_zone
        item["availability_status"] = (
            "eroded_unavailable" if inside_eroded_zone else "available"
        )
        item["availability_reason"] = (
            "Inside an eroded zone" if inside_eroded_zone else None
        )
        matches = _matching_warning_zones(
            item.get("latitude"),
            item.get("longitude"),
            warning_zones,
        )
        item["survival_warning"] = bool(matches)
        item["warning_zone_ids"] = [zone["id"] for zone in matches]
        item["warning_zone_names"] = [zone["name"] for zone in matches]
        item["warning_reasons"] = [
            f"{_warning_type_label(zone['warning_type'])}: {zone['notes']}".strip(": ")
            for zone in matches
        ]
        if matches:
            primary = max(
                matches,
                key=lambda zone: _WARNING_SEVERITY_RANK.get(zone["severity"], 0),
            )
            primary_label = _warning_type_label(primary["warning_type"])
            item["warning_severity"] = primary["severity"]
            item["warning_summary"] = (
                f"{primary['name']} - {primary_label}"
                + (f": {primary['notes']}" if primary.get("notes") else "")
            )
        else:
            item["warning_severity"] = None
            item["warning_summary"] = None
        annotated.append(item)
    return annotated


def _deleted_point_message(point: dict) -> str:
    reason = (point.get("deletion_reason") or "No reason recorded").strip()
    return (
        f"Point #{point['point_num']} from {point['image_name']} was marked "
        f"deleted and cannot be assigned. Reason: {reason}"
    )


def _first_deleted_point(
    conn: Any,
    point_ids: List[int],
) -> Optional[dict]:
    if not point_ids:
        return None

    placeholders = ",".join("?" for _ in point_ids)
    row = conn.execute(f"""
        SELECT
            pp.id,
            pp.point_num,
            pp.deletion_reason,
            a.image_name
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pp.id IN ({placeholders})
          AND pp.deleted_at IS NOT NULL
        ORDER BY pp.point_num ASC
        LIMIT 1
    """, tuple(point_ids)).fetchone()
    return dict(row) if row else None


def _eroded_unavailable_message(point: dict) -> str:
    return (
        f"Point #{point['point_num']} from {point['image_name']} is inside an "
        "eroded zone and is not available for planting. Remove the eroded zone "
        "to return this point to Planned."
    )


def _first_eroded_unavailable_point(
    conn: Any,
    point_ids: List[int],
) -> Optional[dict]:
    """Return the first requested point currently covered by an eroded zone."""
    if not point_ids:
        return None

    placeholders = ",".join("?" for _ in point_ids)
    rows = conn.execute(f"""
        SELECT
            pp.id,
            pp.point_num,
            pp.latitude,
            pp.longitude,
            a.image_name
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pp.id IN ({placeholders})
          AND pp.deleted_at IS NULL
        ORDER BY pp.point_num ASC
    """, tuple(point_ids)).fetchall()
    for row in rows:
        point = dict(row)
        if _is_point_inside_eroded_zone(point["latitude"], point["longitude"]):
            return point
    return None


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
            COALESCE((SELECT COUNT(*) FROM planting_points WHERE status = 'planned' AND deleted_at IS NULL), 0) AS total_planting_points,
            COALESCE((SELECT COUNT(*) FROM planting_points WHERE deleted_at IS NULL), 0) AS total_mapped_points,
            COALESCE((SELECT COUNT(*) FROM planting_points WHERE status = 'planted' AND deleted_at IS NULL), 0) AS total_planted_points,
            COALESCE((SELECT COUNT(*) FROM planting_points WHERE status = 'skipped' AND deleted_at IS NULL), 0) AS total_skipped_points
        FROM analyses
    """).fetchone()

    stats = dict(summary)

    # All planting points for map rendering
    points = conn.execute("""
        SELECT pp.id, pp.point_num, pp.latitude, pp.longitude, pp.buffer_m, pp.area_m2,
               pp.status, pp.planted_at, pp.planted_date,
               pp.deleted_at, pp.deletion_reason, pp.deletion_batch_id,
               pp.deletion_color,
               a.image_name, a.analyzed_at
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pp.deleted_at IS NULL
        ORDER BY a.analyzed_at DESC
    """).fetchall()
    stats['points'] = _annotate_point_advisories(points)

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
    """Remove an analysis only when none of its points has field history."""
    conn = _get_connection()
    try:
        _delete_analysis_rows(conn, int(analysis_id))
        conn.commit()
    finally:
        conn.close()
    process_object_cleanup_jobs()


DEATH_REASON_CATEGORIES = {
    "unknown": "Not determined",
    "barnacles": "Barnacles",
    "waves": "Waves",
    "disease": "Disease",
    "animal_damage": "Animal damage",
    "storm": "Storm",
    "drying_out": "Drying out",
    "vandalism": "Vandalism",
    "other": "Other",
}


def mark_planting_point_dead(
    point_id: int,
    reason_category: str,
    notes: str = "",
) -> dict:
    """Mark a planted point as dead with a categorical reason + optional notes."""
    category_key = (reason_category or "").strip().lower()
    if category_key not in DEATH_REASON_CATEGORIES:
        raise ValueError(
            "Invalid death reason. Allowed: " + ", ".join(DEATH_REASON_CATEGORIES.keys())
        )

    notes_text = (notes or "").strip()
    if len(notes_text) > 500:
        notes_text = notes_text[:500]

    label = DEATH_REASON_CATEGORIES[category_key]
    full_reason = f"{label}: {notes_text}" if notes_text else label

    try:
        numeric_id = int(point_id)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid planting point id.") from error

    # NOTE: planting_points.status has a CHECK constraint that only allows
    # ('planned', 'planted', 'skipped'). Rather than rewrite the table to add
    # a 'dead' status, we keep status = 'planted' and treat
    # death_at IS NOT NULL as the canonical "dead" signal everywhere.

    conn = _get_connection()
    try:
        row = conn.execute("""
            SELECT id, status, deleted_at, death_at
            FROM planting_points
            WHERE id = ? FOR UPDATE
        """, (numeric_id,)).fetchone()

        if not row:
            raise ValueError("Planting point not found.")
        if row["deleted_at"]:
            raise ValueError("Cannot mark a deleted point as dead. Restore it first.")
        if row["status"] != "planted":
            raise ValueError("Only planted points can be marked dead.")
        if row["death_at"]:
            raise ValueError("This point is already marked dead. Restore it first.")

        death_at = _manila_now().isoformat(timespec="microseconds")
        conn.execute("""
            UPDATE planting_points
               SET death_at = ?,
                   death_reason = ?,
                   death_reason_category = ?,
                   death_notes = ?
             WHERE id = ?
        """, (death_at, full_reason, category_key, notes_text or None, numeric_id))

        # Mirror this death into the immutable history so mortality stats
        # survive a later reset-to-planned (which clears the current-state
        # death_* columns to free the spot for re-planting). Snapshot the
        # planter/species attribution from the most recent completed
        # assignment so per-planter / per-species breakdowns stay accurate
        # even after the assignment is detached.
        attribution_row = conn.execute("""
            SELECT pa.id AS assignment_id, pa.planter_id, pl.full_name AS planter_name, pa.species
              FROM planter_assignment_points pap
              JOIN planter_assignments pa ON pa.id = pap.assignment_id
              LEFT JOIN planters pl ON pl.id = pa.planter_id
             WHERE pap.planting_point_id = ?
               AND pap.status = 'completed'
             ORDER BY pap.completed_at DESC, pap.id DESC
             LIMIT 1
        """, (numeric_id,)).fetchone()
        snap_assignment_id = attribution_row["assignment_id"] if attribution_row else None
        snap_planter_id = attribution_row["planter_id"] if attribution_row else None
        snap_planter_name = attribution_row["planter_name"] if attribution_row else None
        snap_species = attribution_row["species"] if attribution_row else None
        planting_event_row = conn.execute("""
            SELECT id
            FROM planting_events
            WHERE planting_point_id = ?
            ORDER BY planted_at DESC, id DESC
            LIMIT 1
        """, (numeric_id,)).fetchone()
        planting_event_id = int(planting_event_row["id"]) if planting_event_row else None

        conn.execute("""
            INSERT INTO point_death_records (
                planting_point_id, assignment_id, planting_event_id, death_at,
                reason_category, reason_label, notes, planter_id, planter_name, species
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            numeric_id, snap_assignment_id, planting_event_id, death_at, category_key, label,
            notes_text or None, snap_planter_id, snap_planter_name, snap_species,
        ))
        if planting_event_id is not None:
            conn.execute("""
                UPDATE planting_events
                SET closed_at = COALESCE(closed_at, ?),
                    closure_reason = COALESCE(closure_reason, 'recorded_death')
                WHERE id = ?
            """, (death_at, planting_event_id))

        conn.commit()
    finally:
        conn.close()

    return {
        "status": "dead",
        "point_id": numeric_id,
        "death_at": death_at,
        "death_reason": full_reason,
        "death_reason_category": category_key,
        "death_notes": notes_text or None,
    }


def restore_planting_point_to_planted(point_id: int) -> dict:
    """Correct an erroneous manual death on the current planting cycle."""
    try:
        numeric_id = int(point_id)
    except (TypeError, ValueError) as error:
        raise ValueError("Invalid planting point id.") from error

    conn = _get_connection()
    try:
        row = conn.execute("""
            SELECT pp.id, pp.status, pp.death_at,
                   pe.id AS planting_event_id, pe.closure_reason
            FROM planting_points pp
            LEFT JOIN planting_events pe ON pe.id = (
                SELECT latest.id FROM planting_events latest
                WHERE latest.planting_point_id = pp.id
                ORDER BY latest.planted_at DESC, latest.id DESC LIMIT 1
            )
            WHERE pp.id = ? FOR UPDATE OF pp
        """, (numeric_id,)).fetchone()
        if not row:
            raise ValueError("Planting point not found.")
        if not row["death_at"]:
            raise ValueError("Only dead points can be restored to planted.")

        death_row = conn.execute("""
            SELECT id, monitoring_observation_id
            FROM point_death_records
            WHERE planting_event_id = ?
            ORDER BY death_at DESC, id DESC LIMIT 1
        """, (row["planting_event_id"],)).fetchone() if row["planting_event_id"] else None
        if row['planting_event_id'] and conn.execute("""SELECT 1 FROM monitoring_death_locations WHERE planting_event_id = ? AND revoked_at IS NULL
            UNION ALL SELECT 1 FROM replanting_requests WHERE planting_event_id = ?""",
            (row['planting_event_id'], row['planting_event_id'])).fetchone():
            raise ValueError('This death is linked to organization monitoring or replacement work. Review it in Monitoring History.')
        if death_row and death_row["monitoring_observation_id"] is not None:
            raise ValueError(
                "This death is linked to a monitoring observation; correct that observation instead."
            )

        conn.execute("""
            UPDATE planting_points
               SET death_at = NULL,
                   death_reason = NULL,
                   death_reason_category = NULL,
                   death_notes = NULL
             WHERE id = ? FOR UPDATE
        """, (numeric_id,))

        if death_row:
            conn.execute("DELETE FROM point_death_records WHERE id = ?", (death_row["id"],))
        if row["planting_event_id"] and row["closure_reason"] == "recorded_death":
            conn.execute("""
                UPDATE planting_events SET closed_at = NULL, closure_reason = NULL
                WHERE id = ?
            """, (row["planting_event_id"],))

        conn.commit()
    finally:
        conn.close()

    return {"status": "planted", "point_id": numeric_id}


def reset_planting_point_to_planned(point_id: int) -> dict:
    """Review replacement work without deleting original assignment evidence."""
    from mangrovision_db.monitoring_locations import approve_replanting
    user = get_user_by_session_token('')
    if not user or str(user.get('role', '')).lower() not in {'admin', 'lgu', 'planner'}:
        raise ValueError('LGU staff must approve replacement planting.')
    conn = _get_connection()
    try:
        event = conn.execute('SELECT id FROM planting_events WHERE planting_point_id = ? ORDER BY id DESC LIMIT 1', (int(point_id),)).fetchone()
        if not event:
            raise ValueError('Planting event was not found.')
    finally:
        conn.close()
    return approve_replanting(event['id'], 0, int(user['id']))


def get_mortality_stats() -> dict:
    """Aggregate mortality counts + per-reason breakdown across all death events.

    Mortality rate = dead / (planted_alive + dead). 'dead' is read from the
    immutable point_death_records history table, so a point that died and was
    later reset-to-planned (freed for a fresh planting cycle) still counts in
    mortality. Only restore_planting_point_to_planted (death-was-wrong
    correction) removes a history row.
    """
    conn = _get_connection()
    try:
        alive_row = conn.execute("""
            SELECT COALESCE(SUM(CASE WHEN status = 'planted' AND death_at IS NULL THEN 1 ELSE 0 END), 0) AS planted_alive
            FROM planting_points
            WHERE deleted_at IS NULL
        """).fetchone()

        dead_row = conn.execute("""
            SELECT COUNT(*) AS dead_count FROM point_death_records
        """).fetchone()

        reason_rows = conn.execute("""
            SELECT
                COALESCE(reason_category, 'other') AS category,
                COUNT(*) AS dead_count,
                MAX(death_at) AS most_recent_death
            FROM point_death_records
            GROUP BY COALESCE(reason_category, 'other')
            ORDER BY dead_count DESC
        """).fetchall()

        recent_rows = conn.execute("""
            SELECT
                pdr.id AS death_record_id,
                pdr.planting_point_id AS id,
                pp.point_num,
                pp.latitude,
                pp.longitude,
                pdr.death_at,
                pdr.reason_label AS death_reason,
                pdr.reason_category AS death_reason_category,
                pdr.notes AS death_notes,
                pdr.planter_name,
                pdr.species,
                a.image_name,
                -- is_currently_dead lets the UI tell historical deaths
                -- (now reset-to-planned, spot reopened, OR the point was
                -- later deleted) from live deaths (still flagged dead on
                -- the map). Only the latter offer the "Reset to planned"
                -- inline action.
                CASE
                    WHEN pp.death_at IS NOT NULL AND pp.deleted_at IS NULL THEN 1
                    ELSE 0
                END AS is_currently_dead
            FROM point_death_records pdr
            LEFT JOIN planting_points pp ON pp.id = pdr.planting_point_id
            LEFT JOIN analyses a ON a.id = pp.analysis_id
            ORDER BY pdr.death_at DESC
            LIMIT 20
        """).fetchall()

        # Per-species + per-planter survival.
        # Alive = currently-planted points attributed to their latest
        #         COMPLETED assignment's species/planter.
        # Dead  = every death event in history, attributed using the
        #         species/planter snapshotted at the time of death (so
        #         reset-to-planned doesn't strip the dead count off the
        #         original planter).
        species_alive_rows = conn.execute("""
            WITH planted_alive AS (
                SELECT
                    pp.id AS point_id,
                    pa.species,
                    ROW_NUMBER() OVER (
                        PARTITION BY pp.id
                        ORDER BY pap.completed_at DESC, pap.id DESC
                    ) AS rn
                FROM planting_points pp
                JOIN (SELECT * FROM planter_assignment_points WHERE released_at IS NULL) pap ON pap.planting_point_id = pp.id
                JOIN planter_assignments pa ON pa.id = pap.assignment_id
                WHERE pp.status = 'planted'
                  AND pp.death_at IS NULL
                  AND pp.deleted_at IS NULL
                  AND pap.status = 'completed'
            )
            SELECT
                COALESCE(NULLIF(TRIM(species), ''), 'Unspecified') AS species,
                COUNT(*) AS alive
            FROM planted_alive
            WHERE rn = 1
            GROUP BY species
        """).fetchall()

        species_dead_rows = conn.execute("""
            SELECT
                COALESCE(NULLIF(TRIM(species), ''), 'Unspecified') AS species,
                COUNT(*) AS dead
            FROM point_death_records
            GROUP BY species
        """).fetchall()

        planter_alive_rows = conn.execute("""
            WITH planted_alive AS (
                SELECT
                    pp.id AS point_id,
                    pa.planter_id,
                    ROW_NUMBER() OVER (
                        PARTITION BY pp.id
                        ORDER BY pap.completed_at DESC, pap.id DESC
                    ) AS rn
                FROM planting_points pp
                JOIN (SELECT * FROM planter_assignment_points WHERE released_at IS NULL) pap ON pap.planting_point_id = pp.id
                JOIN planter_assignments pa ON pa.id = pap.assignment_id
                WHERE pp.status = 'planted'
                  AND pp.death_at IS NULL
                  AND pp.deleted_at IS NULL
                  AND pap.status = 'completed'
            )
            SELECT
                pl.id AS planter_id,
                pl.full_name AS planter_name,
                COUNT(*) AS alive
            FROM planted_alive pa_inner
            JOIN planters pl ON pl.id = pa_inner.planter_id
            WHERE pa_inner.rn = 1
            GROUP BY pl.id, pl.full_name
        """).fetchall()

        planter_dead_rows = conn.execute("""
            SELECT
                pdr.planter_id,
                COALESCE(pdr.planter_name, pl.full_name) AS planter_name,
                COUNT(*) AS dead
            FROM point_death_records pdr
            LEFT JOIN planters pl ON pl.id = pdr.planter_id
            WHERE pdr.planter_id IS NOT NULL
            GROUP BY pdr.planter_id, COALESCE(pdr.planter_name, pl.full_name)
        """).fetchall()

        # Planting trend — last 30 days of planted_date counts. Used to draw a
        # tiny sparkline-style chart on the dashboard so admins can see if
        # planting cadence is going up or down.
        trend_rows = conn.execute("""
            SELECT
                planted_date AS day,
                COUNT(*) AS planted
            FROM planting_points
            WHERE planted_date IS NOT NULL
              AND deleted_at IS NULL
              AND planted_date >= CURRENT_DATE - 30
            GROUP BY planted_date
            ORDER BY day
        """).fetchall()
    finally:
        conn.close()

    planted_alive = int(alive_row["planted_alive"] or 0)
    dead_count = int(dead_row["dead_count"] or 0)
    ever_planted = planted_alive + dead_count
    mortality_rate = (dead_count / ever_planted * 100.0) if ever_planted else 0.0

    breakdown = []
    for row in reason_rows:
        count = int(row["dead_count"] or 0)
        category = row["category"] or "other"
        breakdown.append({
            "category": category,
            "label": DEATH_REASON_CATEGORIES.get(category, category.title()),
            "count": count,
            "percent_of_dead": (count / dead_count * 100.0) if dead_count else 0.0,
            "percent_of_planted": (count / ever_planted * 100.0) if ever_planted else 0.0,
            "most_recent_death": row["most_recent_death"],
        })

    survival_rate = (planted_alive / ever_planted * 100.0) if ever_planted else 0.0

    def _row_with_rate(row):
        alive = int(row["alive"] or 0)
        dead = int(row["dead"] or 0)
        total = alive + dead
        return {
            "alive": alive,
            "dead": dead,
            "total": total,
            "survival_rate": round((alive / total * 100.0) if total else 0.0, 2),
            "mortality_rate": round((dead / total * 100.0) if total else 0.0, 2),
        }

    species_totals: dict[str, dict] = {}
    for row in species_alive_rows:
        key = row["species"]
        species_totals.setdefault(key, {"alive": 0, "dead": 0})["alive"] = int(row["alive"] or 0)
    for row in species_dead_rows:
        key = row["species"]
        species_totals.setdefault(key, {"alive": 0, "dead": 0})["dead"] = int(row["dead"] or 0)

    species_breakdown = []
    for species, totals in species_totals.items():
        entry = _row_with_rate({"alive": totals["alive"], "dead": totals["dead"]})
        entry["species"] = species
        species_breakdown.append(entry)
    species_breakdown.sort(key=lambda r: r["total"], reverse=True)

    planter_totals: dict[int, dict] = {}
    for row in planter_alive_rows:
        pid = int(row["planter_id"])
        planter_totals.setdefault(pid, {
            "alive": 0, "dead": 0, "planter_name": row["planter_name"],
        })["alive"] = int(row["alive"] or 0)
    for row in planter_dead_rows:
        pid = int(row["planter_id"])
        bucket = planter_totals.setdefault(pid, {
            "alive": 0, "dead": 0, "planter_name": row["planter_name"],
        })
        bucket["dead"] = int(row["dead"] or 0)
        if not bucket.get("planter_name"):
            bucket["planter_name"] = row["planter_name"]

    planter_breakdown = []
    for pid, totals in planter_totals.items():
        entry = _row_with_rate({"alive": totals["alive"], "dead": totals["dead"]})
        entry["planter_id"] = pid
        entry["planter_name"] = totals["planter_name"]
        planter_breakdown.append(entry)
    planter_breakdown.sort(key=lambda r: r["total"], reverse=True)
    planter_breakdown = planter_breakdown[:20]

    planting_trend = [
        {"day": row["day"], "planted": int(row["planted"] or 0)}
        for row in trend_rows
    ]

    return {
        "planted_alive": planted_alive,
        "dead_count": dead_count,
        "ever_planted": ever_planted,
        "mortality_rate": round(mortality_rate, 2),
        "survival_rate": round(survival_rate, 2),
        "breakdown": breakdown,
        "recent_deaths": [dict(row) for row in recent_rows],
        "categories": [
            {"key": key, "label": label}
            for key, label in DEATH_REASON_CATEGORIES.items()
        ],
        "by_species": species_breakdown,
        "by_planter": planter_breakdown,
        "planting_trend": planting_trend,
    }


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
    organization_id: Optional[int] = None,
    participant_count: int = 1,
    created_by_user_id: Optional[int] = None,
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

    if not isinstance(participant_count, int) or not 1 <= participant_count <= 10000:
        raise ValueError("Participant count must be between 1 and 10000.")
    if organization_id is None:
        raise ValueError("Select an organization for the shared account.")
    password_hash = _hash_password(password)
    conn = _get_connection()
    try:
        clean_organization_id = None
        if organization_id is not None:
            try:
                clean_organization_id = int(organization_id)
            except (TypeError, ValueError) as error:
                raise ValueError("Select a valid organization.") from error
            if not conn.execute(
                "SELECT 1 FROM organizations WHERE id = ? FOR UPDATE", (clean_organization_id,),
            ).fetchone():
                raise ValueError("Selected organization was not found.")
        organization = conn.execute("SELECT name FROM organizations WHERE id = ?", (clean_organization_id,)).fetchone()
        full_name = organization["name"]
        from mangrovision_db.organization_accounts import is_registration_pending, allocate_reserved_points
        reserved = conn.execute("SELECT * FROM planters WHERE organization_id = ? AND merged_into_planter_id IS NULL FOR UPDATE", (clean_organization_id,)).fetchone()
        if reserved and not is_registration_pending(dict(reserved)):
            raise ValueError("This organization already has an account. Use its shared login.")
        if reserved and reserved["status"] != "active":
            raise ValueError("This organization is inactive. Ask the LGU to reactivate it.")
        existing = conn.execute(
            "SELECT id FROM planters WHERE lower(username) = lower(?)",
            (username,),
        ).fetchone()
        if existing:
            raise ValueError("That planter username is already in use.")

        if reserved:
            planter_id = int(reserved["id"])
            conn.execute("""UPDATE planters SET username = ?, password_hash = ?, phone = ?,
                base_label = ?, base_lat = ?, base_lon = ?, participant_count = ?, status = ?
                WHERE id = ?""", (username, password_hash, phone.strip() or None,
                base_label.strip() or None, base_lat, base_lon, participant_count, status, planter_id))
            allocate_reserved_points(conn, planter_id, participant_count)
            append_activity(conn, action="organization_account.created",
                actor_type="staff" if created_by_user_id is not None else "planter",
                actor_user_id=created_by_user_id,
                actor_planter_id=planter_id if created_by_user_id is None else None,
                organization_id=clean_organization_id,
                summary=f"Registered the organization account for {full_name} with its reserved points.",
                details={"planter_id": planter_id, "participant_count": participant_count})
            conn.commit()
            return planter_id

        cur = conn.execute("""
            INSERT INTO planters (
                full_name, organization_id, username, password_hash, phone,
                base_label, base_lat, base_lon, notes, status, participant_count
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            full_name,
            clean_organization_id,
            username,
            password_hash,
            phone.strip() if phone else None,
            base_label.strip() if base_label else None,
            base_lat,
            base_lon,
            notes.strip() if notes else None,
            status,
            participant_count,
        ))
        planter_id = cur.lastrowid
        conn.execute("INSERT INTO organization_participants(planter_id, slot) SELECT ?, generate_series(1, ?)", (planter_id, participant_count))
        append_activity(
            conn, action="organization_account.created",
            actor_type="staff" if created_by_user_id is not None else "planter",
            actor_user_id=created_by_user_id,
            actor_planter_id=planter_id if created_by_user_id is None else None,
            organization_id=clean_organization_id,
            summary=f"Created the organization account for {full_name}.",
            details={"planter_id": planter_id, "participant_count": participant_count},
        )
        conn.commit()
        return planter_id
    finally:
        conn.close()


def list_planters(include_inactive: bool = True) -> List[dict]:
    """Return planters with active-assignment and assigned-point counts."""
    conn = _get_connection()
    query = """
        SELECT
            p.*,
            (p.username IS NULL AND p.password_hash IS NULL) AS registration_pending,
            o.name AS organization_name,
            COUNT(DISTINCT CASE WHEN pa.status = 'active' THEN pa.id END) AS active_assignments,
            COALESCE(SUM(CASE WHEN pa.status = 'active' AND pap.status = 'pending' THEN 1 ELSE 0 END), 0) AS pending_points,
            COALESCE(SUM(CASE WHEN pa.status = 'active' AND pap.status = 'completed' THEN 1 ELSE 0 END), 0) AS completed_points
        FROM planters p
        LEFT JOIN organizations o ON o.id = p.organization_id
        LEFT JOIN planter_assignments pa ON pa.planter_id = p.id
        LEFT JOIN planter_assignment_points pap ON pap.assignment_id = pa.id
    """
    params: Tuple = ()
    query += " WHERE p.merged_into_planter_id IS NULL AND p.organization_id IS NOT NULL"
    if not include_inactive:
        query += " AND p.status = 'active'"
    query += """
        GROUP BY p.id, o.id
        ORDER BY CASE WHEN p.status = 'active' THEN 0 ELSE 1 END, lower(p.full_name)
    """
    rows = conn.execute(query, params).fetchall()
    conn.close()
    return [{key: value for key, value in dict(row).items() if key != "password_hash"} for row in rows]


def get_planter(planter_id: int) -> Optional[dict]:
    """Return one planter row or None."""
    conn = _get_connection()
    row = conn.execute("""
        SELECT p.*, o.name AS organization_name
        FROM planters p
        LEFT JOIN organizations o ON o.id = p.organization_id
        WHERE p.id = ?
    """, (planter_id,)).fetchone()
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
            pp.planted_at,
            pp.planted_date,
            pp.deleted_at,
            pp.deletion_reason,
            pp.deletion_batch_id,
            pp.deletion_color,
            a.image_name,
            a.analyzed_at,
            a.site_zone_id AS analysis_site_zone_id,
            a.species AS analysis_species
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pp.analysis_id = ?
          AND pp.deleted_at IS NULL
    """
    params: List = [analysis_id]
    if only_unassigned:
        base_query += """
          AND NOT EXISTS (
              SELECT 1
              FROM (SELECT * FROM planter_assignment_points WHERE released_at IS NULL) pap
              JOIN planter_assignments pa ON pa.id = pap.assignment_id
              WHERE pap.planting_point_id = pp.id
                AND pa.status = 'active'
          )
        """
    base_query += " ORDER BY pp.point_num ASC"
    rows = conn.execute(base_query, tuple(params)).fetchall()
    conn.close()
    # Erosion never removes a planting point from planner visibility. The
    # advisory fields let the UI render it as unavailable while assignment
    # creation remains server-side blocked until the active zone is removed.
    return _annotate_point_advisories(rows)


def get_planter_dashboard_stats() -> dict:
    """Return high-level planter management counts."""
    conn = _get_connection()
    row = conn.execute("""
        SELECT
            COALESCE((SELECT COUNT(*) FROM planters WHERE status = 'active' AND merged_into_planter_id IS NULL AND organization_id IS NOT NULL), 0) AS active_planters,
            COALESCE((SELECT COUNT(*) FROM planter_assignments WHERE status = 'active'), 0) AS active_assignments,
            COALESCE((
                SELECT COUNT(*)
                FROM (SELECT * FROM planter_assignment_points WHERE released_at IS NULL) pap
                JOIN planter_assignments pa ON pa.id = pap.assignment_id
                WHERE pa.status = 'active'
                  AND pap.status = 'pending'
            ), 0) AS pending_assigned_points,
            COALESCE((
                SELECT COUNT(*)
                FROM (SELECT * FROM planter_assignment_points WHERE released_at IS NULL) pap
                JOIN planter_assignments pa ON pa.id = pap.assignment_id
                WHERE pa.status IN ('active', 'completed', 'archived')
                  AND pap.status = 'completed'
            ), 0) AS completed_assigned_points,
            COALESCE((
                SELECT COUNT(*)
                FROM (SELECT * FROM planter_assignment_points WHERE released_at IS NULL) pap
                JOIN planter_assignments pa ON pa.id = pap.assignment_id
                WHERE pa.status = 'active'
                  AND pap.status = 'skipped'
            ), 0) AS skipped_assigned_points
    """).fetchone()
    conn.close()
    return dict(row)


def _refresh_assignment_status(conn: Any, assignment_id: int) -> None:
    """Refresh one assignment batch status after point-level changes."""
    row = conn.execute("""
        SELECT
            pa.status AS current_status,
            COUNT(pap.id) AS total_points,
            COALESCE(SUM(CASE WHEN pap.status = 'pending' THEN 1 ELSE 0 END), 0) AS pending_points
        FROM planter_assignments pa
        LEFT JOIN planter_assignment_points pap ON pap.assignment_id = pa.id
        WHERE pa.id = ?
        GROUP BY pa.id
    """, (assignment_id,)).fetchone()

    if not row:
        return
    if row["total_points"] == 0:
        has_history = conn.execute(
            "SELECT 1 FROM planting_events WHERE assignment_id = ? LIMIT 1",
            (assignment_id,),
        ).fetchone()
        if has_history:
            conn.execute(
                "UPDATE planter_assignments SET status = 'archived' WHERE id = ?",
                (assignment_id,),
            )
        else:
            conn.execute("DELETE FROM planter_assignments WHERE id = ?", (assignment_id,))
        return

    if row["current_status"] == "archived":
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
    conn: Any,
    planter: dict,
    point_ids: List[int],
) -> List[int]:
    """Group zigzag point locations before dividing participant ownership."""
    from mangrovision_db.planting_order import zigzag_assignment_points
    if not point_ids:
        return []
    placeholders = ",".join("?" for _ in point_ids)
    rows = conn.execute(f"""
        SELECT id, analysis_id, point_num, latitude, longitude
        FROM planting_points
        WHERE id IN ({placeholders})
    """, tuple(point_ids)).fetchall()
    points = [dict(row) for row in rows]
    return [point['id'] for point in zigzag_assignment_points(points)]


ALLOWED_ASSIGNMENT_SPECIES = ("Bungalon", "Rhizophora", "Api-Api")
_ASSIGNMENT_SPECIES_BY_KEY = {
    "bungalon": "Bungalon",
    "rhizophora": "Rhizophora",
    "api-api": "Api-Api",
    "api api": "Api-Api",
    "apiapi": "Api-Api",
}


def _canonical_assignment_species(species: Optional[str]) -> Optional[str]:
    key = (species or "").strip().lower().replace("_", "-")
    if not key:
        return None
    return _ASSIGNMENT_SPECIES_BY_KEY.get(key)


def _infer_assignment_species(conn: Any, point_ids: List[int]) -> Optional[str]:
    if not point_ids:
        return None

    placeholders = ",".join("?" for _ in point_ids)
    rows = conn.execute(f"""
        SELECT DISTINCT LOWER(TRIM(COALESCE(a.species, ''))) AS species
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pp.id IN ({placeholders})
          AND TRIM(COALESCE(a.species, '')) <> ''
    """, tuple(point_ids)).fetchall()

    species_values = {
        _canonical_assignment_species(row["species"])
        for row in rows
        if _canonical_assignment_species(row["species"])
    }
    if len(species_values) == 1:
        return next(iter(species_values))
    return None


def create_planter_assignment(
    planter_id: int,
    planting_point_ids: List[int],
    assigned_by_user_id: Optional[int] = None,
    title: str = "",
    assignment_date: str = "",
    travel_mode: str = "walking",
    notes: str = "",
    species: str = "",
    site_zone_id: Optional[int] = None,
    connection: Any = None,
) -> int:
    """Create a batch atomically; a supplied connection belongs to its caller."""
    conn = connection or _get_connection()
    try:
        assignment_id = _create_planter_assignment(
            planter_id, planting_point_ids, assigned_by_user_id, title,
            assignment_date, travel_mode, notes, species, site_zone_id, conn)
        if connection is None:
            conn.commit()
        return assignment_id
    except Exception:
        if connection is None:
            conn.rollback()
        raise
    finally:
        if connection is None:
            conn.close()


def _create_planter_assignment(
    planter_id: int,
    planting_point_ids: List[int],
    assigned_by_user_id: Optional[int] = None,
    title: str = "",
    assignment_date: str = "",
    travel_mode: str = "walking",
    notes: str = "",
    species: str = "",
    site_zone_id: Optional[int] = None,
    connection: Any = None,
) -> int:
    """Create one assignment batch and attach ordered planting points."""
    try:
        point_ids = list(dict.fromkeys(int(pid) for pid in planting_point_ids if pid is not None))
    except (TypeError, ValueError) as error:
        raise ValueError("Planting point ids must be whole numbers.") from error
    if not point_ids:
        raise ValueError("Select at least one planting point.")

    assignment_date = (assignment_date or datetime.now().date().isoformat()).strip()
    travel_mode = (travel_mode or "walking").strip().lower()
    if travel_mode not in {"walking", "driving"}:
        raise ValueError("Invalid travel mode.")

    species_value = _canonical_assignment_species(species)
    if species and not species_value:
        raise ValueError(
            "Species must be one of: " + ", ".join(ALLOWED_ASSIGNMENT_SPECIES) + "."
        )

    conn = connection or _get_connection()
    # Serialize batches for this organization before calculating each slot's share.
    row = conn.execute("SELECT * FROM planters WHERE id = ? FOR UPDATE", (planter_id,)).fetchone()
    if not row:
        raise ValueError("Selected planter was not found.")
    planter = dict(row)
    if planter["status"] != "active" or planter.get("merged_into_planter_id") is not None or planter.get("organization_id") is None:
        raise ValueError("Choose an active organization account.")
    if site_zone_id is None:
        sites = conn.execute("SELECT id FROM project_sites WHERE organization_id = ? ORDER BY id", (planter["organization_id"],)).fetchall()
        if len(sites) == 1:
            site_zone_id = sites[0]["id"]
    normalized_site_zone_id = None
    if planter.get("organization_id") is not None and site_zone_id is None:
        raise ValueError("Select a project site owned by this planter's organization.")
    if site_zone_id is not None:
        try:
            normalized_site_zone_id = int(site_zone_id)
        except (TypeError, ValueError) as error:
            raise ValueError("Invalid project site id.") from error
        selected_site = conn.execute(
            "SELECT id, organization_id FROM site_zones WHERE id = ?",
            (normalized_site_zone_id,),
        ).fetchone()
        if not selected_site:
            raise ValueError("Selected project site was not found.")
        if (
            planter.get("organization_id") is not None
            and selected_site["organization_id"] != planter["organization_id"]
        ):
            raise ValueError(
                "The selected project site belongs to a different organization than this planter."
            )
    if not species_value:
        species_value = _infer_assignment_species(conn, point_ids)
    if not species_value:
        raise ValueError(
            "Choose a species for this assignment or record one on its image analysis before assigning points."
        )

    placeholders = ",".join("?" for _ in point_ids)
    requested_rows = conn.execute(f"""
        SELECT
            pp.id,
            pp.point_num,
            pp.latitude,
            pp.longitude,
            pp.status,
            pp.death_at,
            pp.deleted_at,
            a.species AS analysis_species,
            a.site_zone_id AS source_site_id,
            source_site.name AS source_site_name,
            source_site.organization_id AS source_organization_id,
            source_org.name AS source_organization_name
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        LEFT JOIN site_zones source_site ON source_site.id = a.site_zone_id
        LEFT JOIN organizations source_org ON source_org.id = source_site.organization_id
        WHERE pp.id IN ({placeholders})
        ORDER BY pp.id
        FOR UPDATE OF pp
    """, point_ids).fetchall()
    requested_rows = _resolve_point_project_sites(conn, requested_rows)
    if len(requested_rows) != len(point_ids):
        found = {int(row["id"]) for row in requested_rows}
        missing = [point_id for point_id in point_ids if point_id not in found]
        raise ValueError(f"Unknown planting point id(s): {', '.join(map(str, missing))}.")
    missing_analysis_species = [
        row for row in requested_rows
        if not _canonical_assignment_species(row["analysis_species"])
    ]
    if missing_analysis_species:
        sample = missing_analysis_species[0]
        raise ValueError(
            f"Point #{sample['point_num']} has no valid species on its image analysis. "
            "Record the species there before assigning this point."
        )
    non_assignable = [
        row for row in requested_rows
        if row["status"] not in {"planned", "skipped"} or row["death_at"] is not None
    ]
    if non_assignable:
        sample = non_assignable[0]
        raise ValueError(
            f"Point #{sample['point_num']} is already planted or has mortality history; reset it before reassigning."
        )

    if normalized_site_zone_id is not None:
        wrong_site_points = [
            row for row in requested_rows
            if row["source_site_id"] is None
            or int(row["source_site_id"]) != normalized_site_zone_id
        ]
        if wrong_site_points:
            sample = wrong_site_points[0]
            if sample["source_site_id"] is None:
                raise ValueError(
                    f"Point #{sample['point_num']} is not inside exactly one project site. "
                    "Choose a point inside the selected organization's Zone Editor boundary."
                )
            source_owner = sample["source_organization_name"] or "another organization"
            source_site_name = sample["source_site_name"] or "another project site"
            raise ValueError(
                f"Point #{sample['point_num']} belongs to {source_site_name} ({source_owner}), "
                "not the selected project site. Choose a planter and project site from the same organization as the point."
            )

    deleted_point = _first_deleted_point(conn, point_ids)
    if deleted_point:
        raise ValueError(_deleted_point_message(deleted_point))

    eroded_point = _first_eroded_unavailable_point(conn, point_ids)
    if eroded_point:
        raise ValueError(_eroded_unavailable_message(eroded_point))

    duplicate_rows = conn.execute(f"""
        SELECT
            pp.point_num,
            a.image_name,
            pa.title,
            pl.full_name AS planter_name
        FROM (SELECT * FROM planter_assignment_points WHERE released_at IS NULL) pap
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
        raise ValueError(
            f"Point #{sample['point_num']} from {sample['image_name']} is already assigned to {sample['planter_name']}."
        )

    ordered_point_ids = _order_point_ids_for_planter(conn, planter, point_ids)

    if not title.strip():
        title = f"{planter['full_name']} planting run {assignment_date}"

    cur = conn.execute("""
        INSERT INTO planter_assignments (
            planter_id, assigned_by_user_id, title, assignment_date, travel_mode,
            status, species, notes, site_zone_id
        )
        VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)
    """, (
        planter_id,
        assigned_by_user_id,
        title.strip(),
        assignment_date,
        travel_mode,
        species_value,
        notes.strip() if notes else None,
        normalized_site_zone_id,
    ))
    assignment_id = cur.lastrowid

    from mangrovision_db.organization_accounts import allocation_slots
    counts = conn.execute("""SELECT pap.participant_slot, COUNT(*) AS total
        FROM (SELECT * FROM planter_assignment_points WHERE released_at IS NULL) pap JOIN planter_assignments pa ON pa.id = pap.assignment_id
        WHERE pa.planter_id = ? AND pa.status IN ('active', 'completed') GROUP BY pap.participant_slot""", (planter_id,)).fetchall()
    slots = allocation_slots(len(ordered_point_ids), planter["participant_count"],
                             {row["participant_slot"]: row["total"] for row in counts})
    for idx, point_id in enumerate(ordered_point_ids, start=1):
        conn.execute("""
            INSERT INTO planter_assignment_points (
                assignment_id, planting_point_id, sequence_num, status, participant_slot
            )
            VALUES (?, ?, ?, 'pending', ?)
        """, (
            assignment_id,
            point_id,
            idx,
            slots[idx - 1],
        ))

    # Review releases a location; every assignment path keeps that review linked.
    conn.execute(f"""UPDATE replanting_requests r SET assignment_id = ?, version = version + 1
        FROM planting_events pe WHERE pe.id = r.planting_event_id
        AND pe.planting_point_id IN ({placeholders}) AND r.assignment_id IS NULL
        AND r.replacement_event_id IS NULL""", [assignment_id, *point_ids])
    append_activity(
        conn, action="assignment.created", actor_type="staff" if assigned_by_user_id else "system",
        actor_user_id=assigned_by_user_id, actor_planter_id=planter_id,
        organization_id=planter["organization_id"], project_site_id=normalized_site_zone_id,
        summary=f"Assigned {len(ordered_point_ids)} planting points to {planter['full_name']}.",
        details={"assignment_id": assignment_id, "point_count": len(ordered_point_ids)},
    )
    return assignment_id


def create_organization_assignment(organization_id: int, planting_point_ids: List[int], **kwargs) -> int:
    """Reserve points for an organization, even before its shared login exists."""
    conn = _get_connection()
    try:
        organization = conn.execute("SELECT * FROM organizations WHERE id = ? FOR UPDATE",
                                    (organization_id,)).fetchone()
        if not organization:
            raise ValueError("Selected organization was not found.")
        account = conn.execute("""SELECT * FROM planters WHERE organization_id = ?
            AND merged_into_planter_id IS NULL FOR UPDATE""", (organization_id,)).fetchone()
        if account is None:
            # NULL credentials make this a reservation owner, unable to sign in.
            cursor = conn.execute("""INSERT INTO planters(full_name, organization_id, status, participant_count)
                VALUES (?, ?, 'active', 1)""", (organization["name"], organization_id))
            planter_id = int(cursor.lastrowid)
            conn.execute("INSERT INTO organization_participants(planter_id, slot) VALUES (?, 1)", (planter_id,))
        else:
            planter_id = int(account["id"])
        assignment_id = create_planter_assignment(planter_id, planting_point_ids, connection=conn, **kwargs)
        conn.commit()
        return assignment_id
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_planter_assignment_map_points() -> List[dict]:
    """Return all planting points with their current active assignment, if any.

    DISTINCT ON exposes one assignment per point to the planner. Filtering a
    ROW_NUMBER result badly underestimated this join and caused millions of
    repeated comparisons. Keep the priority/date/id ordering identical.
    """
    conn = _get_connection()
    rows = conn.execute("""
        WITH active_point_assignments AS (
            SELECT DISTINCT ON (pap.planting_point_id)
                pap.id AS assignment_point_id,
                pap.assignment_id,
                pap.planting_point_id,
                pap.sequence_num,
                pap.status AS assignment_status,
                pap.assigned_at,
                pap.status_changed_at,
                pap.skip_reason,
                pa.planter_id,
                pa.title AS assignment_title,
                pa.assignment_date,
                pa.species AS assignment_species,
                pa.site_zone_id,
                sz.name AS project_site_name,
                pa.created_at,
                pl.full_name AS planter_name,
                pl.organization_id
            FROM (SELECT * FROM planter_assignment_points WHERE released_at IS NULL) pap
            JOIN planter_assignments pa ON pa.id = pap.assignment_id
            JOIN planters pl ON pl.id = pa.planter_id
            LEFT JOIN site_zones sz ON sz.id = pa.site_zone_id
            WHERE pa.status IN ('active', 'completed')
               OR (pa.status = 'archived' AND pap.status = 'completed')
            ORDER BY pap.planting_point_id,
                CASE
                    WHEN pa.status = 'active' AND pap.status = 'pending' THEN 0
                    WHEN pap.status = 'completed' THEN 1
                    ELSE 2
                END,
                COALESCE(pap.completed_at, pa.created_at) DESC,
                pap.id DESC
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
            pp.planted_at,
            pp.planted_date,
            pp.deleted_at,
            pp.deletion_reason,
            pp.deletion_batch_id,
            pp.deletion_color,
            pp.death_at,
            pp.death_reason,
            pp.death_reason_category,
            pp.death_notes,
            a.image_name,
            a.analyzed_at,
            a.center_lat,
            a.center_lon,
            a.species AS analysis_species,
            a.planting_distance_m AS analysis_planting_distance_m,
            a.hexagon_size_m AS analysis_hexagon_size_m,
            source_site.id AS source_project_site_id,
            source_site.id AS source_site_id,
            source_site.name AS source_site_name,
            source_site.name AS source_project_site_name,
            source_site.organization_id AS source_organization_id,
            source_org.name AS source_organization_name,
            apa.assignment_point_id,
            apa.assignment_id,
            apa.sequence_num,
            apa.assignment_status,
            apa.assigned_at,
            apa.status_changed_at,
            apa.skip_reason,
            apa.planter_id AS assigned_planter_id,
            apa.organization_id AS assigned_organization_id,
            apa.planter_name AS assigned_planter_name,
            apa.assignment_title,
            apa.assignment_date,
            apa.assignment_species,
            apa.site_zone_id,
            apa.project_site_name,
            apa.project_site_name AS site_name,
            current_planting.source AS planting_source,
            current_planting.planted_by_user_id,
            planting_staff.full_name AS planted_by_name,
            EXISTS (
                SELECT 1 FROM map_zones mz
                WHERE mz.zone_type = 'eroded' AND mz.deleted_at IS NULL
                  AND extensions.ST_Covers(mz.geometry, pp.location)
            ) AS map_eroded_coverage
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        LEFT JOIN LATERAL (
            SELECT MIN(ps.id) AS site_id
            FROM project_sites ps
            WHERE (a.site_zone_id IS NULL OR ps.id = a.site_zone_id)
              AND extensions.ST_Covers(
                  ps.geometry,
                  extensions.ST_SetSRID(extensions.ST_MakePoint(pp.longitude, pp.latitude), 4326)
              )
            HAVING COUNT(*) = 1
        ) inferred_site ON TRUE
        LEFT JOIN site_zones source_site
          ON source_site.id = inferred_site.site_id
        LEFT JOIN organizations source_org ON source_org.id = source_site.organization_id
        LEFT JOIN active_point_assignments apa
               ON apa.planting_point_id = pp.id
        LEFT JOIN LATERAL (
            SELECT pe.source, pe.planted_by_user_id
            FROM planting_events pe
            WHERE pe.planting_point_id = pp.id AND pe.closed_at IS NULL
            ORDER BY pe.planted_at DESC, pe.id DESC LIMIT 1
        ) current_planting ON TRUE
        LEFT JOIN users planting_staff ON planting_staff.id = current_planting.planted_by_user_id
        WHERE pp.deleted_at IS NULL
        ORDER BY a.analyzed_at DESC, pp.point_num ASC
    """).fetchall()
    conn.close()
    # Ownership and erosion come from the same fresh read as the point data.
    # Keep the internal coverage flag out of the public response contract.
    resolved_rows = [dict(row) for row in rows]
    eroded_ids = {int(row['id']) for row in resolved_rows if row.pop('map_eroded_coverage')}
    return _annotate_point_advisories(resolved_rows, eroded_point_ids=eroded_ids)


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
            pp.planted_at,
            pp.planted_date,
            pp.deleted_at,
            pp.deletion_reason,
            pp.deletion_batch_id,
            pp.deletion_color,
            a.image_name,
            a.analyzed_at,
            a.site_zone_id AS analysis_site_zone_id,
            a.species AS analysis_species
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pp.id = ?
    """, (planting_point_id,)).fetchone()
    if not point_row:
        conn.close()
        raise ValueError("Selected planting point was not found.")

    if conn.execute("""SELECT 1 FROM replanting_requests r JOIN planting_events pe ON pe.id = r.planting_event_id
        WHERE pe.planting_point_id = ? AND r.replacement_event_id IS NULL""", (planting_point_id,)).fetchone():
        conn.close()
        raise ValueError('Assign this approved replacement through the organization assignment form in Planters.')
    point = dict(point_row)
    if point.get("deleted_at"):
        conn.close()
        raise ValueError(_deleted_point_message(point))
    if _is_point_inside_eroded_zone(point.get("latitude"), point.get("longitude")):
        conn.close()
        raise ValueError(_eroded_unavailable_message(point))
    if point.get("planting_status") == "planted":
        conn.close()
        raise ValueError(
            f"Point #{point['point_num']} from {point['image_name']} is already marked as planted."
        )
    analysis_species = _canonical_assignment_species(point.get("analysis_species"))
    if not analysis_species:
        conn.close()
        raise ValueError(
            f"Point #{point['point_num']} has no valid species on its image analysis. "
            "Record the species there before assigning this point."
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
            FROM (SELECT * FROM planter_assignment_points WHERE released_at IS NULL) pap
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
          AND site_zone_id IS NOT DISTINCT FROM ?
          AND LOWER(TRIM(species)) = LOWER(?)
        ORDER BY assignment_date DESC, created_at DESC, id DESC
        LIMIT 1
    """, (planter_id, point.get("analysis_site_zone_id"), analysis_species)).fetchone()

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
                planter_id, assigned_by_user_id, title, assignment_date, travel_mode,
                status, species, notes, site_zone_id
            )
            VALUES (?, ?, ?, ?, ?, 'active', ?, ?, ?)
        """, (
            planter_id,
            assigned_by_user_id,
            target_assignment_title,
            assignment_date,
            travel_mode,
            analysis_species,
            "Created from the interactive planter management map.",
            point.get("analysis_site_zone_id"),
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
            pa.species,
            pa.site_zone_id,
            sz.name AS project_site_name,
            sz.name AS site_name,
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
        LEFT JOIN site_zones sz ON sz.id = pa.site_zone_id
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
        GROUP BY pa.id, p.id, u.id, sz.id, sz.name
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
            pap.released_at,
            pap.released_by,
            pap.assigned_at,
            pap.status_changed_at,
            pap.skip_reason,
            pap.notes,
            pa.title,
            pa.species,
            pa.site_zone_id,
            sz.name AS project_site_name,
            sz.name AS site_name,
            pa.travel_mode,
            pa.assignment_date,
            pa.status AS assignment_status_overall,
            p.id AS planter_id,
            p.full_name AS planter_name,
            p.base_label,
            p.base_lat,
            p.base_lon,
            pp.id,
            pp.id AS planting_point_id,
            pp.analysis_id,
            pp.status AS planting_status,
            pp.deleted_at,
            pp.deletion_reason,
            pp.death_at,
            pp.point_num,
            pp.latitude,
            pp.longitude,
            pp.buffer_m,
            pp.area_m2,
            pp.planted_at,
            pp.planted_date,
            a.species AS analysis_species,
            a.planting_distance_m AS analysis_planting_distance_m,
            a.image_name,
            a.analyzed_at
        FROM planter_assignment_points pap
        JOIN planter_assignments pa ON pa.id = pap.assignment_id
        LEFT JOIN site_zones sz ON sz.id = pa.site_zone_id
        JOIN planters p ON p.id = pa.planter_id
        JOIN planting_points pp ON pp.id = pap.planting_point_id
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pap.assignment_id = ?
        ORDER BY pap.sequence_num ASC
    """, (assignment_id,)).fetchall()
    conn.close()
    return _annotate_point_advisories(rows)


def get_planter_field_points(planter_id: int, participant_slot: Optional[int] = None) -> List[dict]:
    """Return active and completed assignment points for one planter.

    Completed assignments are included so finished points remain on the
    field map (rendered as "planted" in yellow) instead of disappearing
    the moment the last point in a batch is completed. Use the
    'archived' status when an assignment really should be hidden.
    """
    conn = _get_connection()
    rows = conn.execute("""
        SELECT
            pap.id AS assignment_point_id,
            pap.assignment_id,
            pap.sequence_num,
            pap.participant_slot,
            pap.status AS assignment_status,
            pap.completed_at,
            pap.assigned_at,
            pap.status_changed_at,
            pap.skip_reason,
            pa.title,
            pa.species,
            pa.travel_mode,
            pa.assignment_date,
            pa.site_zone_id AS project_site_id,
            sz.name AS project_site_name,
            sz.polygon_geojson AS project_site_geometry,
            sz.organization_id AS project_site_organization_id,
            o.name AS organization_name,
            p.organization_id,
            p.full_name AS planter_name,
            p.base_label,
            p.base_lat,
            p.base_lon,
            pp.id,
            pp.id AS planting_point_id,
            pp.analysis_id,
            pp.status AS planting_status,
            pp.deleted_at,
            pp.deletion_reason,
            pp.death_at,
            pp.point_num,
            pp.latitude,
            pp.longitude,
            pp.buffer_m,
            pp.area_m2,
            pp.planted_at,
            pp.planted_date,
            a.species AS analysis_species,
            a.planting_distance_m AS analysis_planting_distance_m,
            a.image_name,
            a.analyzed_at
        FROM (SELECT * FROM planter_assignment_points WHERE released_at IS NULL) pap
        JOIN planter_assignments pa ON pa.id = pap.assignment_id
        JOIN planters p ON p.id = pa.planter_id
        LEFT JOIN site_zones sz ON sz.id = pa.site_zone_id
        LEFT JOIN organizations o ON o.id = p.organization_id
        JOIN planting_points pp ON pp.id = pap.planting_point_id
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pa.planter_id = ?
          AND (CAST(? AS INTEGER) IS NULL OR pap.participant_slot = ?)
          AND pa.status IN ('active', 'completed')
          AND (
                pa.site_zone_id IS NULL
                OR p.organization_id IS NULL
                OR sz.organization_id = p.organization_id
              )
        ORDER BY pa.assignment_date DESC, pap.sequence_num ASC
    """, (planter_id, participant_slot, participant_slot)).fetchall()
    conn.close()
    return _annotate_point_advisories(rows)


def update_assignment_point_status(
    assignment_point_id: int,
    status: str,
    skip_reason: Optional[str] = None,
    *,
    actor_user_id: Optional[int] = None,
    actor_planter_id: Optional[int] = None,
    participant_slot: Optional[int] = None,
) -> None:
    """Update one assignment point status and auto-close completed batches."""
    status = (status or "").strip().lower()
    if status not in {"pending", "completed", "skipped"}:
        raise ValueError("Invalid assignment point status.")
    clean_skip_reason = (skip_reason or "").strip() or None
    if clean_skip_reason and len(clean_skip_reason) > 500:
        clean_skip_reason = clean_skip_reason[:500]
    if status != "skipped":
        clean_skip_reason = None

    completed_at = _manila_now().isoformat(timespec='seconds') if status == "completed" else None
    planted_date = completed_at[:10] if completed_at else None

    conn = _get_connection()
    conn.execute('SELECT id FROM planting_points WHERE id = (SELECT planting_point_id FROM planter_assignment_points WHERE id = ?) FOR UPDATE', (assignment_point_id,)).fetchone()
    row = conn.execute("""
        SELECT
            pap.assignment_id,
            pap.planting_point_id,
            pap.status AS current_assignment_point_status,
            pa.planter_id,
            pa.site_zone_id,
            pl.organization_id,
            pp.point_num,
            pp.latitude,
            pp.longitude,
            pp.deleted_at,
            pp.deletion_reason,
            a.image_name
        FROM (SELECT * FROM planter_assignment_points WHERE released_at IS NULL) pap
        JOIN planter_assignments pa ON pa.id = pap.assignment_id
        JOIN planters pl ON pl.id = pa.planter_id
        JOIN planting_points pp ON pp.id = pap.planting_point_id
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pap.id = ?
    """, (assignment_point_id,)).fetchone()
    if not row:
        conn.close()
        raise ValueError("Assignment point was not found.")
    point = dict(row)
    if point['current_assignment_point_status'] == status:
        conn.close()
        return
    if point.get("deleted_at"):
        conn.close()
        raise ValueError(_deleted_point_message(point))
    if status in {"completed", "skipped"} and _is_point_inside_eroded_zone(
        point.get("latitude"),
        point.get("longitude"),
    ):
        conn.close()
        raise ValueError(_eroded_unavailable_message(point))
    if point.get("current_assignment_point_status") == "completed" and status != "completed":
        protected_history = conn.execute("""
            SELECT 1
            FROM planting_events pe
            WHERE pe.assignment_point_id = ?
              AND (
                  pe.replaces_event_id IS NOT NULL OR EXISTS(SELECT 1 FROM monitoring_observations mo WHERE mo.planting_event_id = pe.id)
                  OR EXISTS(SELECT 1 FROM point_death_records pdr WHERE pdr.planting_event_id = pe.id)
              )
            LIMIT 1
        """, (assignment_point_id,)).fetchone()
        if protected_history:
            conn.close()
            raise ValueError(
                "A completed point with monitoring or mortality history cannot be undone."
            )

    conn.execute("""
        UPDATE planter_assignment_points
        SET status = ?, completed_at = ?, skip_reason = ?
        WHERE id = ?
    """, (status, completed_at, clean_skip_reason, assignment_point_id))

    planting_status = "planned"
    if status == "completed":
        planting_status = "planted"
    elif status == "skipped":
        planting_status = "skipped"

    conn.execute("""
        UPDATE planting_points
        SET status = ?,
            planted_at = ?,
            planted_date = ?
        WHERE id = ?
    """, (planting_status, completed_at, planted_date, row["planting_point_id"]))

    _refresh_assignment_status(conn, row["assignment_id"])

    event = conn.execute("""
        SELECT id FROM planting_events WHERE assignment_point_id = ? AND closed_at IS NULL
        ORDER BY id DESC LIMIT 1
    """, (assignment_point_id,)).fetchone() if status == "completed" else None
    append_activity(
        conn, action=f"point.{status}",
        actor_type="planter" if actor_planter_id else "staff" if actor_user_id else "system",
        actor_user_id=actor_user_id, actor_planter_id=actor_planter_id,
        participant_slot=participant_slot, organization_id=row["organization_id"],
        project_site_id=row["site_zone_id"], planting_point_id=row["planting_point_id"],
        planting_event_id=event["id"] if event else None,
        summary=f"Point #{row['point_num']} was {'planted' if status == 'completed' else status}.",
        details={"assignment_id": row["assignment_id"], "skip_reason": clean_skip_reason},
    )

    conn.commit()
    conn.close()


def mark_planter_points_completed(planter_id: int, assignment_point_ids: List[int],
                                  participant_slot: Optional[int] = None) -> dict:
    """Mark selected available assignment points for one planter as completed."""
    selected_ids = [
        int(point_id)
        for point_id in dict.fromkeys(assignment_point_ids or [])
        if point_id is not None
    ]
    if not selected_ids:
        raise ValueError("Select at least one completed point.")

    completed_at = _manila_now().isoformat(timespec='seconds')
    planted_date = completed_at[:10]

    conn = _get_connection()
    placeholders = ",".join("?" for _ in selected_ids)
    conn.execute(f"""SELECT id FROM planting_points WHERE id IN
        (SELECT planting_point_id FROM planter_assignment_points WHERE id IN ({placeholders})) ORDER BY id FOR UPDATE""", selected_ids).fetchall()
    rows = conn.execute("""
        SELECT
            pap.id AS assignment_point_id,
            pap.assignment_id,
            pap.status AS assignment_status,
            pap.planting_point_id,
            pa.site_zone_id,
            pl.organization_id,
            pp.point_num,
            pp.latitude,
            pp.longitude,
            pp.deleted_at,
            pp.deletion_reason,
            a.image_name
        FROM (SELECT * FROM planter_assignment_points WHERE released_at IS NULL) pap
        JOIN planter_assignments pa ON pa.id = pap.assignment_id
        JOIN planters pl ON pl.id = pa.planter_id
        JOIN planting_points pp ON pp.id = pap.planting_point_id
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pa.planter_id = ?
          AND pa.status IN ('active', 'completed')
          AND pap.status != 'completed'
          AND pap.id IN ({placeholders})
        ORDER BY pa.assignment_date DESC, pap.sequence_num ASC
    """.format(placeholders=placeholders), (planter_id, *selected_ids)).fetchall()

    actionable = []
    unavailable_count = 0
    deleted_count = 0
    for row in rows:
        point = dict(row)
        if point.get("deleted_at"):
            deleted_count += 1
            continue
        if _is_point_inside_eroded_zone(point.get("latitude"), point.get("longitude")):
            unavailable_count += 1
            continue
        actionable.append(point)

    if not actionable:
        conn.close()
        return {
            "status": "updated",
            "updated_points": 0,
            "requested_points": len(selected_ids),
            "unavailable_points": unavailable_count,
            "deleted_points": deleted_count,
        }

    assignment_point_ids = [point["assignment_point_id"] for point in actionable]
    planting_point_ids = [point["planting_point_id"] for point in actionable]
    assignment_ids = sorted({point["assignment_id"] for point in actionable})

    assignment_placeholders = ",".join("?" for _ in assignment_point_ids)
    planting_placeholders = ",".join("?" for _ in planting_point_ids)

    conn.execute(f"""
        UPDATE planter_assignment_points
        SET status = 'completed',
            completed_at = ?
        WHERE id IN ({assignment_placeholders})
    """, (completed_at, *assignment_point_ids))

    conn.execute(f"""
        UPDATE planting_points
        SET status = 'planted',
            planted_at = ?,
            planted_date = ?
        WHERE id IN ({planting_placeholders})
    """, (completed_at, planted_date, *planting_point_ids))

    for assignment_id in assignment_ids:
        _refresh_assignment_status(conn, assignment_id)

    for point in actionable:
        event = conn.execute("""
            SELECT id FROM planting_events WHERE assignment_point_id = ? AND closed_at IS NULL
            ORDER BY id DESC LIMIT 1
        """, (point["assignment_point_id"],)).fetchone()
        append_activity(
            conn, action="point.completed", actor_type="planter", actor_planter_id=planter_id,
            participant_slot=participant_slot, organization_id=point["organization_id"],
            project_site_id=point["site_zone_id"], planting_point_id=point["planting_point_id"],
            planting_event_id=event["id"] if event else None,
            summary=f"Point #{point['point_num']} was planted.",
            details={"assignment_id": point["assignment_id"]},
        )

    conn.commit()
    conn.close()
    return {
        "status": "updated",
        "updated_points": len(actionable),
        "requested_points": len(selected_ids),
        "unavailable_points": unavailable_count,
        "deleted_points": deleted_count,
    }


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
    try:
        if conn.execute(
            "SELECT 1 FROM planting_events WHERE assignment_id = ? LIMIT 1",
            (int(assignment_id),),
        ).fetchone():
            raise ValueError(
                "This assignment has planting history and cannot be deleted; archive it instead."
            )
        conn.execute("DELETE FROM planter_assignments WHERE id = ?", (int(assignment_id),))
        conn.commit()
    finally:
        conn.close()


# ====================================================================
#  Site Zones — admin-drawn plantable area polygons + per-zone
#  mortality aggregation. The point-in-polygon spatial join uses
#  shapely (already a dependency for the canopy detector).
# ====================================================================

def _parse_site_zone_geometry(polygon_geojson: str):
    """Parse a stored polygon JSON into a shapely geometry. Returns None on failure."""
    try:
        from shapely.geometry import shape  # local import: shapely is heavy
    except ImportError:
        return None
    try:
        geom = shape(json.loads(polygon_geojson))
        if not geom.is_valid:
            geom = geom.buffer(0)
        return geom
    except Exception:
        return None


def _site_geometry_centroid(polygon_geojson: Optional[str]) -> tuple[Optional[float], Optional[float]]:
    """Return ``(latitude, longitude)`` for contextual map/tide lookup only."""
    if not polygon_geojson:
        return None, None
    geometry = _parse_site_zone_geometry(polygon_geojson)
    if geometry is None or geometry.is_empty:
        return None, None
    try:
        centroid = geometry.centroid
        return float(centroid.y), float(centroid.x)
    except (AttributeError, TypeError, ValueError):
        return None, None


def _validate_site_polygon_payload(polygon_geojson: dict) -> str:
    """Validate that the payload is a Polygon/MultiPolygon and return JSON string."""
    if not isinstance(polygon_geojson, dict):
        raise ValueError("Polygon must be a GeoJSON geometry object.")
    geom_type = polygon_geojson.get("type")
    if geom_type not in {"Polygon", "MultiPolygon"}:
        raise ValueError("Polygon must be a GeoJSON Polygon or MultiPolygon.")
    coords = polygon_geojson.get("coordinates")
    if not coords:
        raise ValueError("Polygon coordinates are missing.")
    normalized = normalize_polygon({"type": geom_type, "coordinates": coords})
    return json.dumps(normalized)


def list_site_zones() -> List[dict]:
    """Auto-derived site zones — one per planter assignment.

    Each active/completed/archived assignment with >= 3 unique point
    coordinates becomes a site zone whose polygon is the convex hull of
    its assigned points. There is no manual draw flow anymore; site
    zones are entirely a function of the assignment graph so per-area
    mortality tracking happens for free whenever a planner hands out work.
    """
    conn = _get_connection()
    try:
        assignment_rows = conn.execute("""
            WITH assignment_hulls AS (
                SELECT
                    a.id,
                    a.title,
                    a.species,
                    a.created_at,
                    a.assignment_date,
                    a.status,
                    a.planter_id,
                    pl.full_name AS planter_name,
                    pl.organization_id,
                    COUNT(pp.id) AS point_count,
                    extensions.ST_ConvexHull(
                        extensions.ST_Collect(pp.location)
                    ) AS hull
                FROM planter_assignments a
                JOIN planters pl ON pl.id = a.planter_id
                JOIN (SELECT * FROM planter_assignment_points WHERE released_at IS NULL) pap ON pap.assignment_id = a.id
                JOIN planting_points pp ON pp.id = pap.planting_point_id
                WHERE a.status IN ('active', 'completed', 'archived')
                  AND pp.deleted_at IS NULL
                  AND pp.location IS NOT NULL
                GROUP BY
                    a.id,
                    a.title,
                    a.species,
                    a.created_at,
                    a.assignment_date,
                    a.status,
                    a.planter_id,
                    pl.full_name,
                    pl.organization_id
            )
            SELECT
                *,
                extensions.ST_AsGeoJSON(hull) AS hull_geojson
            FROM assignment_hulls
            WHERE point_count >= 3
              AND extensions.ST_GeometryType(hull) = 'ST_Polygon'
            ORDER BY created_at DESC
        """).fetchall()
    finally:
        conn.close()

    features: List[dict] = []
    for arow in assignment_rows:
        aid = int(arow["id"])
        try:
            polygon_geojson = json.loads(arow["hull_geojson"])
        except (TypeError, ValueError, KeyError):
            continue

        title = (arow["title"] or "").strip()
        planter_name = (arow["planter_name"] or "").strip()
        if title and planter_name:
            display_name = f"{title} — {planter_name}"
        elif title:
            display_name = title
        elif planter_name:
            display_name = f"{planter_name} — Assignment #{aid}"
        else:
            display_name = f"Assignment #{aid}"

        features.append({
            "type": "Feature",
            "id": aid,
            "properties": {
                "id": aid,
                "name": display_name,
                "notes": arow["species"] or "",
                "created_at": arow["created_at"],
                "updated_at": None,
                "assignment_id": aid,
                "assignment_status": arow["status"],
                "assignment_date": arow["assignment_date"],
                "planter_id": int(arow["planter_id"]) if arow["planter_id"] is not None else None,
                "planter_name": planter_name or None,
                "organization_id": int(arow["organization_id"]) if arow["organization_id"] is not None else None,
                "species": arow["species"] or None,
                "point_count": int(arow["point_count"]),
                "is_auto_assignment_zone": True,
            },
            "geometry": polygon_geojson,
        })
    return features


def get_site_zone(zone_id: int) -> Optional[dict]:
    """Return a single auto-generated site zone (assignment-derived) by id."""
    target = int(zone_id)
    for feature in list_site_zones():
        if int(feature.get("id") or 0) == target:
            return feature
    return None


def create_site_zone(name: str, polygon_geojson: dict, notes: str = "") -> int:
    """Site zones are auto-generated from assignments — manual creation is disabled."""
    raise ValueError(
        "Site zones are auto-generated from planter assignments. "
        "Create an assignment instead of drawing a zone manually."
    )


def update_site_zone(
    zone_id: int,
    name: Optional[str] = None,
    polygon_geojson: Optional[dict] = None,
    notes: Optional[str] = None,
) -> Optional[dict]:
    """Site zones are auto-generated from assignments — manual edits are disabled."""
    raise ValueError(
        "Site zones are auto-generated from planter assignments and cannot be edited directly."
    )


def delete_site_zone(zone_id: int) -> None:
    """Site zones are auto-generated from assignments — manual deletion is disabled."""
    raise ValueError(
        "Site zones are auto-generated from planter assignments and cannot be deleted directly. "
        "Archive or delete the underlying assignment to remove the zone."
    )


def _warning_zone_feature(row: Any) -> Optional[dict]:
    try:
        geometry = json.loads(row["polygon_geojson"])
    except Exception:
        return None
    warning_type = _clean_warning_type(row["warning_type"])
    severity = _clean_warning_severity(row["severity"])
    return {
        "type": "Feature",
        "id": int(row["id"]),
        "properties": {
            "id": int(row["id"]),
            "name": row["name"],
            "warning_type": warning_type,
            "warning_label": _warning_type_label(warning_type),
            "severity": severity,
            "notes": row["notes"],
            "created_at": row["created_at"],
            "updated_at": row["updated_at"],
        },
        "geometry": geometry,
    }


def list_warning_zones() -> List[dict]:
    """Return planner warning polygons as GeoJSON Features."""
    conn = _get_connection()
    try:
        rows = conn.execute("""
            SELECT id, name, warning_type, severity, notes, polygon_geojson, created_at, updated_at
            FROM warning_zones
            ORDER BY name COLLATE NOCASE
        """).fetchall()
    finally:
        conn.close()

    features = []
    for row in rows:
        feature = _warning_zone_feature(row)
        if feature:
            features.append(feature)
    return features


def get_warning_zone(zone_id: int) -> Optional[dict]:
    """Return a single warning zone with parsed geometry."""
    conn = _get_connection()
    try:
        row = conn.execute("""
            SELECT id, name, warning_type, severity, notes, polygon_geojson, created_at, updated_at
            FROM warning_zones
            WHERE id = ?
        """, (int(zone_id),)).fetchone()
    finally:
        conn.close()
    return _warning_zone_feature(row) if row else None


def create_warning_zone(
    name: str,
    polygon_geojson: dict,
    warning_type: str = "planner_warning",
    severity: str = "medium",
    notes: str = "",
) -> int:
    """Create a non-blocking planner warning zone and return its id."""
    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("Warning zone name is required.")
    polygon_text = _validate_site_polygon_payload(polygon_geojson)
    conn = _get_connection()
    try:
        cur = conn.execute("""
            INSERT INTO map_zones (
                zone_type, name, warning_type, severity, notes, polygon_geojson
            ) VALUES ('warning', ?, ?, ?, ?, ?)
        """, (
            clean_name[:120],
            _clean_warning_type(warning_type),
            _clean_warning_severity(severity),
            (notes or "").strip() or None,
            polygon_text,
        ))
        conn.commit()
        return int(cur.lastrowid)
    finally:
        conn.close()


def update_warning_zone(
    zone_id: int,
    name: Optional[str] = None,
    polygon_geojson: Optional[dict] = None,
    warning_type: Optional[str] = None,
    severity: Optional[str] = None,
    notes: Optional[str] = None,
) -> Optional[dict]:
    """Patch a planner warning zone."""
    updates: list[tuple[str, object]] = []
    if name is not None:
        clean = name.strip()
        if not clean:
            raise ValueError("Warning zone name cannot be empty.")
        updates.append(("name", clean[:120]))
    if polygon_geojson is not None:
        updates.append(("polygon_geojson", _validate_site_polygon_payload(polygon_geojson)))
    if warning_type is not None:
        updates.append(("warning_type", _clean_warning_type(warning_type)))
    if severity is not None:
        updates.append(("severity", _clean_warning_severity(severity)))
    if notes is not None:
        updates.append(("notes", notes.strip() or None))
    if not updates:
        return get_warning_zone(zone_id)
    set_clause = ", ".join(f"{column} = ?" for column, _ in updates)
    values = [value for _, value in updates] + [datetime.now().isoformat(timespec="seconds"), int(zone_id)]
    conn = _get_connection()
    try:
        conn.execute(
            f"""UPDATE map_zones
                SET {set_clause}, updated_at = ?
                WHERE id = ? AND zone_type = 'warning' AND deleted_at IS NULL""",
            values,
        )
        conn.commit()
    finally:
        conn.close()
    return get_warning_zone(zone_id)


def delete_warning_zone(zone_id: int) -> None:
    conn = _get_connection()
    try:
        conn.execute("""
            UPDATE map_zones
            SET deleted_at = CURRENT_TIMESTAMP, updated_at = CURRENT_TIMESTAMP
            WHERE id = ? AND zone_type = 'warning' AND deleted_at IS NULL
        """, (int(zone_id),))
        conn.commit()
    finally:
        conn.close()


def get_mortality_detail_table() -> List[dict]:
    """Per-zone detailed table used by the Monitoring overlay.

    Each "zone" is a planter assignment (site zones are auto-derived from
    assignments now). For each zone we return every point that has ever been
    associated with it — currently linked AND historical deaths whose point
    has since been reset-to-planned (and detached). For each point we expose:

      * status      — planted / dead / skipped / pending / deleted /
                       'dead (spot reset)' for detached historical deaths
      * planted_at  — when the assignment_point was completed (or planting_points)
      * death_at    — when the point died (live death_at, else historical record)
      * death_reason — human label
      * death_reason_category — categorical key used by the breakdown

    Rows are sorted by point_num within each zone so the table reads
    consistently between renders.
    """
    conn = _get_connection()
    try:
        assignments = conn.execute("""
            SELECT a.id, a.title, a.species, a.status, a.assignment_date,
                   a.created_at, pl.full_name AS planter_name
            FROM planter_assignments a
            LEFT JOIN planters pl ON pl.id = a.planter_id
            WHERE a.status IN ('active', 'completed', 'archived')
            ORDER BY a.created_at DESC
        """).fetchall()

        live_rows = conn.execute("""
            SELECT pap.assignment_id,
                   pap.planting_point_id,
                   pap.status AS assignment_point_status,
                   pap.completed_at,
                   pp.point_num,
                   pp.status AS planting_status,
                   pp.planted_at,
                   pp.death_at,
                   pp.death_reason,
                   pp.death_reason_category,
                   pp.deleted_at
            FROM (SELECT * FROM planter_assignment_points WHERE released_at IS NULL) pap
            JOIN planting_points pp ON pp.id = pap.planting_point_id
        """).fetchall()

        # Most-recent death record per (assignment_id, point_id). Used to
        # surface the cause even when the live state has been reset, AND to
        # surface "detached" historical deaths (point no longer in the
        # assignment_points table).
        death_rows = conn.execute("""
            SELECT pdr.assignment_id,
                   pdr.planting_point_id,
                   pdr.death_at,
                   pdr.reason_label,
                   pdr.reason_category,
                   pp.point_num
            FROM point_death_records pdr
            LEFT JOIN planting_points pp ON pp.id = pdr.planting_point_id
            WHERE pdr.assignment_id IS NOT NULL
            ORDER BY pdr.death_at DESC, pdr.id DESC
        """).fetchall()
    finally:
        conn.close()

    deaths_by_key: dict[tuple[int, int], dict] = {}
    for row in death_rows:
        key = (int(row["assignment_id"]), int(row["planting_point_id"]))
        deaths_by_key.setdefault(key, {
            "death_at": row["death_at"],
            "reason_label": row["reason_label"],
            "reason_category": row["reason_category"],
            "point_num": row["point_num"],
        })

    live_by_assignment: dict[int, list[dict]] = {}
    for row in live_rows:
        live_by_assignment.setdefault(int(row["assignment_id"]), []).append(dict(row))

    result: List[dict] = []
    for arow in assignments:
        aid = int(arow["id"])
        title = (arow["title"] or "").strip()
        planter_name = arow["planter_name"]
        if title and planter_name:
            display_name = f"{title} — {planter_name}"
        elif title:
            display_name = title
        elif planter_name:
            display_name = f"Assignment #{aid} — {planter_name}"
        else:
            display_name = f"Assignment #{aid}"

        points: list[dict] = []
        seen_point_ids: set[int] = set()

        for cp in live_by_assignment.get(aid, []):
            pid = int(cp["planting_point_id"])
            seen_point_ids.add(pid)
            death_key = (aid, pid)
            death = deaths_by_key.get(death_key)
            is_deleted = bool(cp["deleted_at"])
            is_dead_live = bool(cp["death_at"]) and not is_deleted
            is_skipped = (cp["planting_status"] == "skipped") or (cp["assignment_point_status"] == "skipped")
            is_completed = cp["assignment_point_status"] == "completed"

            if is_deleted:
                status = "deleted"
            elif is_dead_live:
                status = "dead"
            elif is_skipped:
                status = "skipped"
            elif is_completed:
                status = "planted"
            else:
                status = "pending"

            planted_at = cp["planted_at"] or cp["completed_at"]
            death_at = cp["death_at"] or (death["death_at"] if death else None)
            death_reason = cp["death_reason"] or (death["reason_label"] if death else None)
            death_reason_category = cp["death_reason_category"] or (death["reason_category"] if death else None)

            points.append({
                "id": pid,
                "point_num": int(cp["point_num"]) if cp["point_num"] is not None else None,
                "status": status,
                "planted_at": planted_at,
                "death_at": death_at,
                "death_reason": death_reason,
                "death_reason_category": death_reason_category,
                "is_detached": False,
            })

        # Historical-only deaths: point was reset-to-planned and detached,
        # so it no longer appears in planter_assignment_points. Surface as
        # a virtual row so the user can still see the death tied to this
        # zone.
        for (death_aid, pid), death in deaths_by_key.items():
            if death_aid != aid or pid in seen_point_ids:
                continue
            points.append({
                "id": pid,
                "point_num": int(death["point_num"]) if death["point_num"] is not None else None,
                "status": "dead (spot reset)",
                "planted_at": None,
                "death_at": death["death_at"],
                "death_reason": death["reason_label"],
                "death_reason_category": death["reason_category"],
                "is_detached": True,
            })

        points.sort(key=lambda p: (p["point_num"] if p["point_num"] is not None else 1 << 30))

        alive = sum(1 for p in points if p["status"] == "planted")
        dead = sum(1 for p in points if p["status"].startswith("dead"))
        skipped = sum(1 for p in points if p["status"] == "skipped")
        pending = sum(1 for p in points if p["status"] == "pending")

        result.append({
            "assignment_id": aid,
            "name": display_name,
            "title": title or None,
            "planter_name": planter_name,
            "species": arow["species"] or None,
            "assignment_status": arow["status"],
            "assignment_date": arow["assignment_date"],
            "points": points,
            "alive": alive,
            "dead": dead,
            "skipped": skipped,
            "pending": pending,
            "total": len(points),
        })

    return result


def get_site_zone_mortality() -> List[dict]:
    """Per-assignment alive/dead breakdown for the auto-derived site zones.

    Site zones are now 1:1 with planter_assignments, so per-zone mortality is
    simply per-assignment mortality. Each death event is attributed to the
    assignment_id snapshotted at mark-dead time, so a death stays attached to
    the assignment that owned the point at the time it died — even after a
    reset-to-planned detaches the point and a new assignment picks it up.
    """
    conn = _get_connection()
    try:
        assignment_rows = conn.execute("""
            SELECT
                a.id,
                a.title,
                pl.full_name AS planter_name
            FROM planter_assignments a
            JOIN planters pl ON pl.id = a.planter_id
            WHERE a.status IN ('active', 'completed', 'archived')
        """).fetchall()

        alive_rows = conn.execute("""
            SELECT pap.assignment_id, COUNT(*) AS alive
            FROM (SELECT * FROM planter_assignment_points WHERE released_at IS NULL) pap
            JOIN planting_points pp ON pp.id = pap.planting_point_id
            WHERE pp.status = 'planted'
              AND pp.death_at IS NULL
              AND pp.deleted_at IS NULL
            GROUP BY pap.assignment_id
        """).fetchall()

        dead_rows = conn.execute("""
            SELECT assignment_id, COUNT(*) AS dead
            FROM point_death_records
            WHERE assignment_id IS NOT NULL
            GROUP BY assignment_id
        """).fetchall()
    finally:
        conn.close()

    alive_by_id = {int(row["assignment_id"]): int(row["alive"] or 0) for row in alive_rows}
    dead_by_id = {int(row["assignment_id"]): int(row["dead"] or 0) for row in dead_rows}

    result: List[dict] = []
    for arow in assignment_rows:
        aid = int(arow["id"])
        alive = alive_by_id.get(aid, 0)
        dead = dead_by_id.get(aid, 0)
        total = alive + dead
        title = (arow["title"] or "").strip()
        planter_name = (arow["planter_name"] or "").strip()
        if title and planter_name:
            display_name = f"{title} — {planter_name}"
        elif title:
            display_name = title
        elif planter_name:
            display_name = f"{planter_name} — Assignment #{aid}"
        else:
            display_name = f"Assignment #{aid}"
        result.append({
            "id": aid,
            "name": display_name,
            "alive": alive,
            "dead": dead,
            "total": total,
            "survival_rate": round((alive / total * 100.0) if total else 0.0, 2),
            "mortality_rate": round((dead / total * 100.0) if total else 0.0, 2),
        })
    return result


# ── Initialise on import ────────────────────────────────────────────
# Schema installation is intentionally not performed at import time. Run
# ``alembic upgrade head`` before starting the supported FastAPI application.


# ====================================================================
#  Decision dashboard: stable project sites, settings, monitoring, and
#  scientifically explicit aggregate metrics.
# ====================================================================

_UNSET = object()


def _manila_now() -> datetime:
    return datetime.now(_MANILA_TZ)


def _parse_dashboard_datetime(
    value: Optional[Any],
    *,
    end_of_day: bool = False,
) -> Optional[datetime]:
    if value is None or value == "":
        return None
    if isinstance(value, datetime):
        parsed = value
    elif isinstance(value, date):
        parsed = datetime.combine(value, time.max if end_of_day else time.min)
    else:
        text = str(value).strip()
        if not text:
            return None
        try:
            if len(text) == 10:
                parsed_date = date.fromisoformat(text)
                parsed = datetime.combine(parsed_date, time.max if end_of_day else time.min)
            else:
                parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError as error:
            raise ValueError(f"Invalid ISO date/time: {text}") from error
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=_MANILA_TZ)
    else:
        parsed = parsed.astimezone(_MANILA_TZ)
    return parsed


def _dashboard_period(
    date_from: Optional[Any] = None,
    date_to: Optional[Any] = None,
    bucket: str = "week",
    as_of: Optional[Any] = None,
) -> tuple[datetime, datetime, datetime, str]:
    as_of_dt = _parse_dashboard_datetime(as_of) or _manila_now()
    start = _parse_dashboard_datetime(date_from)
    end = _parse_dashboard_datetime(date_to, end_of_day=True)
    if start is None:
        start = datetime(as_of_dt.year, 1, 1, tzinfo=_MANILA_TZ)
    if end is None:
        end = as_of_dt
    if end > as_of_dt:
        end = as_of_dt
    if start > end:
        raise ValueError("date_from must be on or before date_to.")
    clean_bucket = (bucket or "week").strip().lower()
    if clean_bucket not in {"day", "week", "month"}:
        raise ValueError("bucket must be day, week, or month.")
    return start, end, as_of_dt, clean_bucket


def _iso_local(value: Optional[Any]) -> Optional[str]:
    parsed = _parse_dashboard_datetime(value)
    return parsed.isoformat(timespec="seconds") if parsed else None


def list_organizations() -> List[dict]:
    conn = _get_connection()
    try:
        return [
            {
                "id": int(row["id"]),
                "name": row["name"],
                "inspection_interval_days": int(row["inspection_interval_days"]),
                "created_at": row["created_at"],
                "updated_at": row["updated_at"],
            }
            for row in conn.execute("""
                SELECT id, name, inspection_interval_days, created_at, updated_at
                FROM organizations
                ORDER BY name COLLATE NOCASE, id
            """).fetchall()
        ]
    finally:
        conn.close()


def _clean_monitoring_count(value: Any, field_name: str) -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a whole number.")
    try:
        number = float(value)
        count = int(number)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError(f"{field_name} must be a whole number.") from error
    if not math.isfinite(number) or number != count or count < 0:
        raise ValueError(f"{field_name} must be a non-negative whole number.")
    return count


def _clean_monitoring_height(value: Any) -> float:
    try:
        height = float(value)
    except (TypeError, ValueError, OverflowError) as error:
        raise ValueError("average_height_cm must be a non-negative number.") from error
    if not math.isfinite(height) or height < 0:
        raise ValueError("average_height_cm must be a non-negative number.")
    return round(height, 2)


def _clean_organization_health_status(value: Any) -> str:
    clean = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if clean not in _VALID_ORGANIZATION_HEALTH_STATUSES:
        raise ValueError(
            "health_status must be one of: excellent, good, fair, poor, critical."
        )
    return clean


def _clean_growth_stage(value: Optional[str], alive: int, height: Optional[float]) -> Optional[str]:
    if value is None:
        if height is None:
            raise ValueError("Select the growth stage seen during this visit.")
        return None  # Preserve measured-height submissions from older clients.
    if value not in {"seedling", "young", "larger", "mixed", "not_checked", "no_living"}:
        raise ValueError("Select a valid growth stage.")
    if (alive == 0) != (value == "no_living"):
        raise ValueError("Growth stage must match whether living seedlings were found.")
    return value


def _organization_growth_dashboard(conn: Any, end: datetime) -> dict:
    """Combine latest visit survivors with plantings awaiting a visit."""
    rows = conn.execute("""
        SELECT r.alive_count, r.growth_snapshot
        FROM organizations o
        LEFT JOIN organization_monitoring_records r ON r.id = (
            SELECT latest.id FROM organization_monitoring_records latest
            WHERE latest.organization_id = o.id
              AND latest.monitored_at <= ?
            ORDER BY latest.monitored_at DESC, latest.id DESC LIMIT 1
        )
    """, (end.isoformat(),)).fetchall()
    groups = {
        'new': {
            'key': 'new-seedlings',
            'growth_group': 'New seedlings',
            'seedling_count': 0,
        },
        'growing': {
            'key': 'growing-seedlings',
            'growth_group': 'Growing seedlings',
            'seedling_count': 0,
        },
    }
    living_seedlings_counted = 0
    unclassified_living_count = 0
    missing_date_count = 0
    for source in rows:
        row = dict(source)
        snapshot = _monitoring_json(row.get('growth_snapshot')) or {}
        living_count = max(0, int(row.get('alive_count') or 0))
        missing_date_count += max(0, int(snapshot.get('missing_date_count') or 0))
        planted_by_group = {'new': 0, 'growing': 0}
        for cohort in snapshot.get('cohorts') or []:
            cycle = int(cohort.get('completed_cycles') or 0)
            group_key = 'new' if cycle == 0 else 'growing'
            planted_by_group[group_key] += max(0, int(cohort.get('planted_count') or 0))

        planted_with_dates = sum(planted_by_group.values())
        living_with_dates = min(living_count, planted_with_dates)
        unclassified_living_count += living_count - living_with_dates
        if not planted_with_dates:
            continue

        # Visits save one living total for an organization. Share that total
        # across its groups by planting count while keeping the exact living total.
        allocations = {
            key: living_with_dates * count // planted_with_dates
            for key, count in planted_by_group.items()
        }
        remaining = living_with_dates - sum(allocations.values())
        remainder_order = sorted(
            planted_by_group,
            key=lambda key: (
                -(living_with_dates * planted_by_group[key] % planted_with_dates),
                0 if key == 'new' else 1,
            ),
        )
        for key in remainder_order[:remaining]:
            allocations[key] += 1
        for key, count in allocations.items():
            groups[key]['seedling_count'] += count
        living_seedlings_counted += living_with_dates

    # These open planting events are included at their recorded age until a
    # visit covers them, without inventing a visit or a measured growth stage.
    unvisited = [dict(row) for row in conn.execute("""
        SELECT pe.planted_at, pe.species
        FROM planting_events pe
        JOIN planters p ON p.id = pe.planter_id
        LEFT JOIN organization_monitoring_records r ON r.id = (
            SELECT latest.id FROM organization_monitoring_records latest
            WHERE latest.organization_id = p.organization_id
              AND latest.monitored_at <= ?
            ORDER BY latest.monitored_at DESC, latest.id DESC LIMIT 1
        )
        WHERE p.organization_id IS NOT NULL AND pe.planted_at <= ?
          AND pe.closed_at IS NULL
          AND (r.id IS NULL OR pe.planted_at > r.monitored_at)
    """, (end.isoformat(), end.isoformat())).fetchall()]
    pending = age_snapshot(unvisited, end, len(unvisited))
    for cohort in pending['cohorts']:
        key = 'new' if cohort['completed_cycles'] == 0 else 'growing'
        groups[key]['seedling_count'] += cohort['planted_count']
        living_seedlings_counted += cohort['planted_count']
    missing_date_count += pending['missing_date_count']

    growth_groups = list(groups.values()) if living_seedlings_counted else []
    return {
        'growth_groups': growth_groups,
        'living_seedlings_counted': living_seedlings_counted,
        'unclassified_living_count': unclassified_living_count,
        'missing_planting_date_count': missing_date_count,
        'awaiting_visit_count': len(unvisited),
        'review_interval_days': 14,
    }


def _organization_monitoring_record_row(
    conn: Any,
    record_id: int,
) -> Optional[dict]:
    return _organization_monitoring_record_rows(conn, [record_id]).get(int(record_id))


def _organization_monitoring_record_rows(conn, record_ids):
    """Load visits and their death locations in two queries, regardless of count."""
    ids = list(dict.fromkeys(int(value) for value in record_ids))
    if not ids:
        return {}
    marks = ','.join('?' for _ in ids)
    rows = conn.execute(f"""
        SELECT
            omr.id,
            omr.organization_id,
            o.name AS organization_name,
            omr.monitored_at,
            omr.alive_count,
            omr.dead_count,
            omr.average_height_cm,
            omr.growth_stage,
            omr.growth_snapshot,
            omr.count_snapshot,
            omr.new_dead_count,
            omr.location_death_count,
            omr.location_version,
            omr.alive_before_count,
            omr.baseline_record_id,
            omr.health_status,
            omr.actions_taken,
            omr.inspector_user_id,
            u.full_name AS inspector_name,
            omr.created_at
        FROM organization_monitoring_records omr
        JOIN organizations o ON o.id = omr.organization_id
        LEFT JOIN users u ON u.id = omr.inspector_user_id
        WHERE omr.id IN ({marks})
    """, ids).fetchall()
    locations = {}
    for link in conn.execute(f'''SELECT record_id, planting_event_id
        FROM monitoring_death_locations
        WHERE record_id IN ({marks}) AND revoked_at IS NULL
        ORDER BY planting_event_id''', ids).fetchall():
        locations.setdefault(link['record_id'], []).append(link['planting_event_id'])
    return {row['id']: _format_organization_monitoring_record(row, locations.get(row['id'], []))
            for row in rows}


def _format_organization_monitoring_record(row, event_ids):
    result = dict(row)
    result['growth_snapshot'] = _monitoring_json(result.get('growth_snapshot'))
    result['count_snapshot'] = _monitoring_json(result.get('count_snapshot'))
    death_snapshot = result['count_snapshot'] or {}
    result['death_reason_category'] = death_snapshot.get('death_reason_category')
    result['death_reason'] = DEATH_REASON_CATEGORIES.get(result['death_reason_category'], 'Not determined')
    result['death_reason_notes'] = death_snapshot.get('death_reason_notes') or ''
    from mangrovision_db.monitoring_locations import location_summary_from_ids
    result.update(location_summary_from_ids(result, event_ids))
    observed = int(result["alive_count"]) + int(result["dead_count"])
    result["total_observed"] = observed
    result["survival_rate_pct"] = (
        round(int(result["alive_count"]) / observed * 100.0, 2)
        if observed else None
    )
    return result


def _monitoring_json(value):
    return json.loads(value) if isinstance(value, str) else value


def _organization_monitoring_timing(latest, plantings, observed, total_planted):
    """Find the next 14-day planting-age round, without shifting after late visits."""
    latest_date = None
    if latest:
        latest_at = _parse_dashboard_datetime(latest.get('monitored_at'))
        latest_date = latest_at.astimezone(_MANILA_TZ).date() if latest_at else None
    next_dates = []
    for planting in plantings:
        planted_at = _parse_dashboard_datetime(planting.get('planted_at'))
        if planted_at is None:
            continue
        planted_date = planted_at.astimezone(_MANILA_TZ).date()
        elapsed = (latest_date - planted_date).days if latest_date is not None else -1
        round_number = max(1, elapsed // _ORGANIZATION_MONITORING_INTERVAL_DAYS + 1)
        nominal_due_day = planted_date + timedelta(
            days=round_number * _ORGANIZATION_MONITORING_INTERVAL_DAYS
        )
        next_dates.append(_monitoring_workday(nominal_due_day))
    next_date = min(next_dates) if next_dates else None
    days_until = (
        max(0, (next_date - observed.date()).days)
        if next_date is not None else None
    )
    return {
        'next_monitoring_date': next_date.isoformat() if next_date else None,
        'monitoring_available': bool(
            int(total_planted or 0) > 0
            and next_date is not None
            and observed.astimezone(_MANILA_TZ).date() >= next_date
        ),
        'days_until_monitoring': days_until,
        'interval_days': _ORGANIZATION_MONITORING_INTERVAL_DAYS,
    }


def _organization_visit_context(conn, organization_id, monitored_at=None):
    observed = _parse_dashboard_datetime(monitored_at) or _manila_now()
    # A date input includes all planting activity on that Philippine calendar day.
    cutoff = datetime.combine(observed.date() + timedelta(days=1), time.min, tzinfo=observed.tzinfo)
    latest_row = conn.execute('''SELECT id FROM organization_monitoring_records
        WHERE organization_id = ? ORDER BY monitored_at DESC, id DESC LIMIT 1''', (organization_id,)).fetchone()
    latest = _organization_monitoring_record_row(conn, latest_row['id']) if latest_row else None
    if latest and observed.date() < _parse_dashboard_datetime(latest['monitored_at']).date():
        raise ValueError('Choose a visit date on or after the last saved monitoring visit.')
    rows = [dict(row) for row in conn.execute('''
        SELECT pe.id, pe.planted_at, pe.species
        FROM planting_events pe JOIN planters p ON p.id = pe.planter_id
        WHERE p.organization_id = ? AND pe.planted_at <= ?
          AND COALESCE(pe.closure_reason, '') <> 'completion_reversed'
        ORDER BY pe.planted_at, pe.id
    ''', (organization_id, min(cutoff - timedelta(microseconds=1), _manila_now()).isoformat())).fetchall()]
    totals = conn.execute('''SELECT COALESCE(MAX(alive_count + dead_count), 0) AS planted,
        COALESCE(MAX(dead_count), 0) AS dead FROM organization_monitoring_records
        WHERE organization_id = ?''', (organization_id,)).fetchone()
    counts = carried_counts(latest, rows, totals['planted'], totals['dead'])
    growth = age_snapshot(rows, observed, counts['total_planted'])
    timing = _organization_monitoring_timing(latest, rows, observed, counts['total_planted'])
    return {**counts, 'organization_id': organization_id, 'monitored_date': observed.date().isoformat(),
            'latest_record': latest, 'baseline_record_id': latest['id'] if latest else None,
            'growth_snapshot': growth, **timing}


def get_organization_visit_context(organization_id, monitored_at=None):
    conn = _get_connection()
    try:
        if not conn.execute('SELECT id FROM organizations WHERE id = ?', (organization_id,)).fetchone():
            raise ValueError('Selected organization was not found.')
        return _organization_visit_context(conn, organization_id, monitored_at)
    finally:
        conn.close()


def record_organization_monitoring_visit(*, organization_id, monitored_at, new_dead_count,
                                       baseline_record_id, expected_alive_count, health_status,
                                       actions_taken, inspector_user_id, dead_planting_event_ids=None,
                                       unlocated_dead_count=None, death_reason_category=None,
                                       death_reason_notes=''):
    from mangrovision_db.monitoring_locations import clean_event_ids, reconcile_on_connection, _record
    event_ids = clean_event_ids(dead_planting_event_ids)
    new_dead = _clean_monitoring_count(new_dead_count, 'Newly dead seedlings')
    unknown = new_dead - len(event_ids) if unlocated_dead_count is None else _clean_monitoring_count(unlocated_dead_count, 'Unlocated deaths')
    if unknown < 0 or len(event_ids) + unknown != new_dead:
        raise ValueError('Newly dead must equal selected seedlings plus deaths with unknown locations.')
    cause = str(death_reason_category or '').strip().lower()
    cause_notes = str(death_reason_notes or '').strip()
    if new_dead and cause not in DEATH_REASON_CATEGORIES:
        raise ValueError('Select a valid cause of death for the newly dead seedlings.')
    if len(cause_notes) > 500:
        raise ValueError('Death cause notes must be 500 characters or fewer.')
    health = _clean_organization_health_status(health_status)
    actions = str(actions_taken or '').strip()
    if not actions or len(actions) > 2000:
        raise ValueError('Describe the LGU actions in 1 to 2000 characters.')
    observed = _parse_dashboard_datetime(monitored_at)
    if not observed or observed.date() > _manila_now().date():
        raise ValueError('Choose a valid monitoring date that is not in the future.')
    conn = _get_connection()
    try:
        # Serialize visits for the same organization so two users cannot spend
        # the same alive baseline or overwrite a newer visit.
        if not conn.execute('SELECT id FROM organizations WHERE id = ? FOR UPDATE', (organization_id,)).fetchone():
            raise ValueError('Selected organization was not found.')
        context = _organization_visit_context(conn, organization_id, observed)
        if context['latest_record']:
            observed = max(observed, _parse_dashboard_datetime(context['latest_record']['monitored_at']))
        if context['baseline_record_id'] != baseline_record_id or context['alive_before_count'] != expected_alive_count:
            raise ValueError('Monitoring changed. Reopen the organization to load the latest saved counts.')
        if context['total_planted'] <= 0:
            raise ValueError('This organization has no planted seedlings to monitor yet.')
        if not context['monitoring_available']:
            raise ValueError(
                f"Not time to monitor yet. The next monitoring visit is on {context['next_monitoring_date']}."
            )
        if new_dead > context['alive_before_count']:
            raise ValueError(f"New deaths cannot exceed {context['alive_before_count']} alive seedlings.")
        alive = context['alive_before_count'] - new_dead
        dead = context['previous_dead_count'] + new_dead
        growth = context['growth_snapshot']
        if alive == 0:
            growth['label'] = 'No living seedlings'
        count_snapshot = {key: context[key] for key in ('total_planted', 'previous_dead_count', 'event_ids', 'new_planted_count')}
        if new_dead:
            count_snapshot.update(death_reason_category=cause, death_reason_notes=cause_notes)
        cursor = conn.execute('''INSERT INTO organization_monitoring_records (
            organization_id, monitored_at, alive_count, dead_count, average_height_cm,
            growth_stage, growth_snapshot, count_snapshot, new_dead_count, alive_before_count,
            baseline_record_id, health_status, actions_taken, inspector_user_id, created_at
        ) VALUES (?, ?, ?, ?, NULL, ?, CAST(? AS jsonb), CAST(? AS jsonb), ?, ?, ?, ?, ?, ?, ?)''', (
            organization_id, observed.isoformat(), alive, dead,
            'no_living' if alive == 0 else 'age_estimate', json.dumps(growth), json.dumps(count_snapshot),
            new_dead, context['alive_before_count'], baseline_record_id, health, actions,
            inspector_user_id, _manila_now().isoformat(),
        ))
        record_id = int(cursor.lastrowid)
        conn.execute('UPDATE organization_monitoring_records SET location_death_count = ? WHERE id = ?', (new_dead, record_id))
        if event_ids:
            reconcile_on_connection(conn, _record(conn, record_id), event_ids, inspector_user_id)
        result = _organization_monitoring_record_row(conn, record_id)
        organization = conn.execute("SELECT name FROM organizations WHERE id = ?",
                                    (organization_id,)).fetchone()
        append_activity(
            conn, action="monitoring.visit_recorded", actor_type="staff",
            actor_user_id=inspector_user_id, organization_id=organization_id,
            summary=f"LGU recorded a monitoring visit for {organization['name'] if organization else 'an organization'}.",
            details={"record_id": record_id, "alive_count": alive, "new_dead_count": new_dead,
                     "monitored_date": observed.date().isoformat()},
        )
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_organization_monitoring_records(
    organization_id: Optional[int] = None,
    limit: int = 100,
    before_id: Optional[int] = None,
) -> List[dict]:
    try:
        clean_limit = max(1, min(1000, int(limit)))
    except (TypeError, ValueError) as error:
        raise ValueError("limit must be a whole number.") from error

    conn = _get_connection()
    try:
        params: list[Any] = []
        conditions = []
        if organization_id is not None:
            try:
                clean_organization_id = int(organization_id)
            except (TypeError, ValueError) as error:
                raise ValueError("organization_id must be a whole number.") from error
            conditions.append("omr.organization_id = ?")
            params.append(clean_organization_id)
        if before_id is not None:
            try:
                clean_before_id = int(before_id)
            except (TypeError, ValueError) as error:
                raise ValueError("before_id must be a whole number.") from error
            if clean_before_id <= 0:
                raise ValueError("before_id must be positive.")
            conditions.append("""(omr.monitored_at, omr.id) < (
                SELECT monitored_at, id FROM organization_monitoring_records WHERE id = ?
            )""")
            params.append(clean_before_id)
        where = "WHERE " + " AND ".join(conditions) if conditions else ""
        rows = conn.execute(f"""
            SELECT omr.id
            FROM organization_monitoring_records omr
            {where}
            ORDER BY omr.monitored_at DESC, omr.id DESC
            LIMIT ?
        """, (*params, clean_limit)).fetchall()
        records = _organization_monitoring_record_rows(conn, [row['id'] for row in rows])
        return [records[row['id']] for row in rows if row['id'] in records]
    finally:
        conn.close()


def list_organization_monitoring_summaries() -> List[dict]:
    """Return each organization with current point context and its latest visit."""
    conn = _get_connection()
    try:
        rows = conn.execute("""
            SELECT
                o.id,
                o.name,
                o.inspection_interval_days,
                COALESCE(tracked.current_planted_points, 0) AS current_planted_points,
                COALESCE(history.record_count, 0) AS monitoring_record_count,
                COALESCE(history.recorded_total, 0) AS recorded_total,
                COALESCE(history.recorded_dead, 0) AS recorded_dead,
                history.latest_record_id
            FROM organizations o
            LEFT JOIN (
                SELECT
                    p.organization_id,
                    COUNT(DISTINCT pp.id) AS current_planted_points
                FROM planters p
                JOIN planter_assignments pa ON pa.planter_id = p.id
                JOIN (SELECT * FROM planter_assignment_points WHERE released_at IS NULL) pap ON pap.assignment_id = pa.id
                JOIN planting_points pp ON pp.id = pap.planting_point_id
                WHERE p.organization_id IS NOT NULL
                  AND pp.status = 'planted'
                  AND pp.deleted_at IS NULL
                GROUP BY p.organization_id
            ) tracked ON tracked.organization_id = o.id
            LEFT JOIN (
                SELECT
                    records.organization_id,
                    COUNT(*) AS record_count,
                    MAX(records.alive_count + records.dead_count) AS recorded_total,
                    MAX(records.dead_count) AS recorded_dead,
                    (
                        SELECT newest.id
                        FROM organization_monitoring_records newest
                        WHERE newest.organization_id = records.organization_id
                        ORDER BY newest.monitored_at DESC, newest.id DESC
                        LIMIT 1
                    ) AS latest_record_id
                FROM organization_monitoring_records records
                GROUP BY records.organization_id
            ) history ON history.organization_id = o.id
            ORDER BY o.name COLLATE NOCASE, o.id
        """).fetchall()
        latest_records = _organization_monitoring_record_rows(
            conn, [row['latest_record_id'] for row in rows if row['latest_record_id'] is not None])
        observed = _manila_now()
        plantings = {}
        if rows:
            for planting in conn.execute('''
                SELECT p.organization_id, pe.id, pe.planted_at, pe.species
                FROM planting_events pe JOIN planters p ON p.id = pe.planter_id
                WHERE p.organization_id IS NOT NULL AND pe.planted_at <= ?
                  AND COALESCE(pe.closure_reason, '') <> 'completion_reversed'
                ORDER BY pe.planted_at, pe.id
            ''', (observed.isoformat(),)).fetchall():
                plantings.setdefault(planting['organization_id'], []).append(dict(planting))
        summaries = []
        for row in rows:
            item = dict(row)
            item["id"] = int(item["id"])
            item["inspection_interval_days"] = int(item["inspection_interval_days"])
            item["current_planted_points"] = int(item["current_planted_points"] or 0)
            item["monitoring_record_count"] = int(item["monitoring_record_count"] or 0)
            latest_id = item.pop("latest_record_id", None)
            latest = item['latest_record'] = latest_records.get(latest_id)
            if latest and observed.date() < _parse_dashboard_datetime(latest['monitored_at']).date():
                raise ValueError('Choose a visit date on or after the last saved monitoring visit.')
            events = plantings.get(item['id'], [])
            counts = carried_counts(latest, events, item.pop('recorded_total'), item.pop('recorded_dead'))
            context = {**counts, **_organization_monitoring_timing(
                latest, events, observed, counts['total_planted'])}
            # Historical cohort totals survive release of the physical location.
            item['total_planted'] = context['total_planted']
            item['alive_seedlings'] = context['alive_before_count']
            item['previous_dead_count'] = context['previous_dead_count']
            item['next_monitoring_date'] = context['next_monitoring_date']
            item['monitoring_available'] = context['monitoring_available']
            item['days_until_monitoring'] = context['days_until_monitoring']
            item['monitoring_interval_days'] = context['interval_days']
            summaries.append(item)
        return summaries
    finally:
        conn.close()


def create_organization_monitoring_record(
    *,
    organization_id: int,
    alive_count: Any,
    dead_count: Any,
    average_height_cm: Any = None,
    growth_stage: Optional[str] = None,
    health_status: str,
    actions_taken: str,
    monitored_at: Optional[Any] = None,
    inspector_user_id: Optional[int] = None,
) -> dict:
    try:
        clean_organization_id = int(organization_id)
    except (TypeError, ValueError) as error:
        raise ValueError("organization_id must be a whole number.") from error
    clean_alive = _clean_monitoring_count(alive_count, "alive_count")
    clean_dead = _clean_monitoring_count(dead_count, "dead_count")
    if clean_alive + clean_dead <= 0:
        raise ValueError("Record at least one alive or dead plant.")
    clean_height = _clean_monitoring_height(average_height_cm) if average_height_cm is not None else None
    clean_stage = _clean_growth_stage(growth_stage, clean_alive, clean_height)
    clean_health = _clean_organization_health_status(health_status)
    clean_actions = str(actions_taken or "").strip()
    if not clean_actions:
        raise ValueError("actions_taken is required.")
    if len(clean_actions) > 2000:
        clean_actions = clean_actions[:2000]
    clean_monitored_at = _parse_dashboard_datetime(monitored_at) or _manila_now()
    if clean_monitored_at > _manila_now() + timedelta(minutes=5):
        raise ValueError("monitored_at cannot be in the future.")

    conn = _get_connection()
    try:
        if not conn.execute(
            "SELECT 1 FROM organizations WHERE id = ?",
            (clean_organization_id,),
        ).fetchone():
            raise ValueError("Selected organization was not found.")
        cursor = conn.execute("""
            INSERT INTO organization_monitoring_records (
                organization_id, monitored_at, alive_count, dead_count,
                average_height_cm, growth_stage, health_status, actions_taken,
                inspector_user_id, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            clean_organization_id,
            clean_monitored_at.isoformat(timespec="seconds"),
            clean_alive,
            clean_dead,
            clean_height,
            clean_stage,
            clean_health,
            clean_actions,
            inspector_user_id,
            _manila_now().isoformat(timespec="seconds"),
        ))
        record_id = int(cursor.lastrowid)
        conn.commit()
        return _organization_monitoring_record_row(conn, record_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def _project_site_feature(row: Any, counts: Optional[dict] = None) -> dict:
    try:
        geometry = json.loads(row["polygon_geojson"])
    except (TypeError, json.JSONDecodeError):
        geometry = None
    centroid_lat, centroid_lon = _site_geometry_centroid(row["polygon_geojson"])
    props = {
        "id": int(row["id"]),
        "name": row["name"],
        "notes": row["notes"],
        "organization_id": (
            int(row["organization_id"]) if row["organization_id"] is not None else None
        ),
        "organization_name": row["organization_name"],
        "organization": row["organization_name"],
        "inspection_interval_days": (
            int(row["inspection_interval_days"])
            if row["inspection_interval_days"] is not None else None
        ),
        "centroid_lat": centroid_lat,
        "centroid_lon": centroid_lon,
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
        "is_stable_project_site": True,
        "tide_calibration": json.loads(row["tide_calibration"]) if row.get("tide_calibration") else None,
    }
    props.update(counts or {})
    return {
        "type": "Feature",
        "id": int(row["id"]),
        "properties": props,
        "geometry": geometry,
    }


def _project_site_counts(conn: Any) -> dict[int, dict]:
    counts: dict[int, dict] = {}
    for row in conn.execute("""
        WITH point_membership AS (
            SELECT a.site_zone_id AS site_id, pp.id AS point_id
            FROM analyses a JOIN planting_points pp ON pp.analysis_id = a.id
            JOIN project_sites ps ON ps.id = a.site_zone_id
            WHERE extensions.ST_Covers(ps.geometry, pp.location)
            UNION
            SELECT pa.site_zone_id AS site_id, pp.id AS point_id
            FROM planter_assignments pa
            JOIN planter_assignment_points pap ON pap.assignment_id = pa.id
            JOIN planting_points pp ON pp.id = pap.planting_point_id
            WHERE pa.site_zone_id IS NOT NULL AND pap.released_at IS NULL
        ), point_counts AS (
            SELECT site_id, COUNT(*) AS point_count FROM point_membership GROUP BY site_id
        ), assignment_counts AS (
            SELECT site_zone_id, COUNT(*) AS assignment_count FROM planter_assignments GROUP BY site_zone_id
        ), analysis_counts AS (
            SELECT site_zone_id, COUNT(*) AS analysis_count FROM analyses GROUP BY site_zone_id
        ), schedule_counts AS (
            SELECT project_site_id, COUNT(*) AS schedule_count FROM planting_schedules GROUP BY project_site_id
        )
        SELECT sz.id, pa.assignment_count, a.analysis_count, pp.point_count, ps.schedule_count
        FROM site_zones sz
        LEFT JOIN assignment_counts pa ON pa.site_zone_id = sz.id
        LEFT JOIN analysis_counts a ON a.site_zone_id = sz.id
        LEFT JOIN point_counts pp ON pp.site_id = sz.id
        LEFT JOIN schedule_counts ps ON ps.project_site_id = sz.id
    """).fetchall():
        counts[int(row["id"])] = {
            "assignment_count": int(row["assignment_count"] or 0),
            "analysis_count": int(row["analysis_count"] or 0),
            "point_count": int(row["point_count"] or 0),
            "schedule_count": int(row["schedule_count"] or 0),
        }
    return counts


def list_project_sites() -> List[dict]:
    conn = _get_connection()
    try:
        rows = conn.execute("""
            SELECT sz.id, sz.name, sz.notes, sz.polygon_geojson,
                   sz.organization_id, o.name AS organization_name,
                   sz.inspection_interval_days,
                   sz.created_at, sz.updated_at, sz.tide_calibration
            FROM project_sites sz
            LEFT JOIN organizations o ON o.id = sz.organization_id
            ORDER BY sz.name COLLATE NOCASE, sz.id
        """).fetchall()
        counts = _project_site_counts(conn)
        return [_project_site_feature(row, counts.get(int(row["id"]))) for row in rows]
    finally:
        conn.close()


def get_project_site(site_id: int) -> Optional[dict]:
    conn = _get_connection()
    try:
        row = conn.execute("""
            SELECT sz.id, sz.name, sz.notes, sz.polygon_geojson,
                   sz.organization_id, o.name AS organization_name,
                   sz.inspection_interval_days,
                   sz.created_at, sz.updated_at, sz.tide_calibration
            FROM project_sites sz
            LEFT JOIN organizations o ON o.id = sz.organization_id
            WHERE sz.id = ?
        """, (int(site_id),)).fetchone()
        if not row:
            return None
        return _project_site_feature(row, _project_site_counts(conn).get(int(site_id)))
    finally:
        conn.close()


def _validate_link_ids(
    conn: Any,
    table: str,
    record_ids: Optional[List[int]],
) -> Optional[List[int]]:
    if record_ids is None:
        return None
    normalized = sorted({int(item) for item in record_ids})
    if not normalized:
        return []
    if table not in {"analyses", "planter_assignments"}:
        raise ValueError("Invalid project-site link target.")
    placeholders = ",".join("?" for _ in normalized)
    found = {
        int(row["id"])
        for row in conn.execute(
            f"SELECT id FROM {table} WHERE id IN ({placeholders})", normalized
        ).fetchall()
    }
    missing = [item for item in normalized if item not in found]
    if missing:
        label = "analysis" if table == "analyses" else "assignment"
        raise ValueError(f"Unknown {label} id(s): {', '.join(map(str, missing))}.")
    return normalized


def _replace_project_site_links(
    conn: Any,
    site_id: int,
    table: str,
    record_ids: Optional[List[int]],
) -> None:
    validated = _validate_link_ids(conn, table, record_ids)
    if validated is None:
        return
    conn.execute(f"UPDATE {table} SET site_zone_id = NULL WHERE site_zone_id = ?", (site_id,))
    if validated:
        placeholders = ",".join("?" for _ in validated)
        conn.execute(
            f"UPDATE {table} SET site_zone_id = ? WHERE id IN ({placeholders})",
            (site_id, *validated),
        )
        if table == "planter_assignments":
            conn.execute(f"""
                UPDATE planting_events
                SET site_zone_id = ?, site_attribution_source = 'manual_assignment_link'
                WHERE site_zone_id IS NULL
                  AND assignment_id IN ({placeholders})
            """, (site_id, *validated))
        else:
            conn.execute(f"""
                UPDATE planting_events
                SET site_zone_id = ?, site_attribution_source = 'manual_analysis_link'
                WHERE site_zone_id IS NULL
                  AND planting_point_id IN (
                      SELECT id FROM planting_points
                      WHERE analysis_id IN ({placeholders})
                  )
            """, (site_id, *validated))


def _auto_link_project_site(conn: Any, site_id: int, polygon_text: str) -> None:
    """Best-effort initial linking for currently ungrouped analyses/assignments."""
    conn.execute("""
        WITH unambiguous AS (
            SELECT a.id, MIN(ps.id) AS project_site_id
            FROM analyses a
            JOIN project_sites ps
              ON extensions.ST_Covers(ps.geometry, a.center_location)
            WHERE a.project_site_id IS NULL AND a.center_location IS NOT NULL
            GROUP BY a.id
            HAVING COUNT(ps.id) = 1 AND MIN(ps.id) = ?
        )
        UPDATE analyses a
        SET project_site_id = unambiguous.project_site_id
        FROM unambiguous
        WHERE a.id = unambiguous.id
    """, (int(site_id),))
    conn.execute("""
        UPDATE planting_events pe
        SET project_site_id = a.project_site_id,
            site_attribution_source = 'auto_unambiguous_analysis'
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        WHERE pe.project_site_id IS NULL
          AND pe.planting_point_id = pp.id
          AND a.project_site_id = ?
    """, (int(site_id),))
    conn.execute("""
        WITH assignment_centers AS (
            SELECT pa.id,
                   extensions.ST_Centroid(extensions.ST_Collect(pp.location)) AS center
            FROM planter_assignments pa
            JOIN planter_assignment_points pap ON pap.assignment_id = pa.id
            JOIN planting_points pp ON pp.id = pap.planting_point_id
            WHERE pa.project_site_id IS NULL
            GROUP BY pa.id
        ), unambiguous AS (
            SELECT ac.id, MIN(ps.id) AS project_site_id
            FROM assignment_centers ac
            JOIN project_sites ps ON extensions.ST_Covers(ps.geometry, ac.center)
            GROUP BY ac.id
            HAVING COUNT(ps.id) = 1 AND MIN(ps.id) = ?
        )
        UPDATE planter_assignments pa
        SET project_site_id = unambiguous.project_site_id
        FROM unambiguous
        WHERE pa.id = unambiguous.id
    """, (int(site_id),))
    conn.execute("""
        UPDATE planting_events pe
        SET project_site_id = pa.project_site_id,
            site_attribution_source = 'auto_unambiguous_assignment'
        FROM planter_assignments pa
        WHERE pe.project_site_id IS NULL
          AND pe.assignment_id = pa.id
          AND pa.project_site_id = ?
    """, (int(site_id),))


def create_project_site(
    name: str,
    geometry: dict,
    notes: str = "",
    analysis_ids: Optional[List[int]] = None,
    assignment_ids: Optional[List[int]] = None,
    organization_id: Optional[int] = None,
) -> dict:
    clean_name = (name or "").strip()
    if not clean_name:
        raise ValueError("Project site name is required.")
    polygon_text = _validate_site_polygon_payload(geometry)
    try:
        clean_organization_id = int(organization_id)
    except (TypeError, ValueError) as error:
        raise ValueError("organization_id is required and must be a whole number.") from error
    conn = _get_connection()
    try:
        organization = conn.execute(
            "SELECT * FROM organizations WHERE id = ? FOR SHARE", (clean_organization_id,),
        ).fetchone()
        if not organization:
            raise ValueError("Selected organization was not found.")
        cursor = conn.execute("""
            INSERT INTO site_zones (
                name, notes, polygon_geojson, organization_id,
                inspection_interval_days
            ) VALUES (?, ?, ?, ?, ?)
        """, (
            clean_name[:120], (notes or "").strip() or None, polygon_text,
            clean_organization_id, int(organization["inspection_interval_days"]),
        ))
        site_id = int(cursor.lastrowid)
        if analysis_ids is None and assignment_ids is None:
            _auto_link_project_site(conn, site_id, polygon_text)
        else:
            _replace_project_site_links(conn, site_id, "analyses", analysis_ids or [])
            _replace_project_site_links(conn, site_id, "planter_assignments", assignment_ids or [])
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_project_site(site_id)


def update_project_site(
    site_id: int,
    name: Optional[str] = None,
    geometry: Optional[dict] = None,
    notes: Optional[str] = None,
    analysis_ids: Optional[List[int]] = None,
    assignment_ids: Optional[List[int]] = None,
) -> Optional[dict]:
    conn = _get_connection()
    try:
        existing = conn.execute(
            "SELECT * FROM site_zones WHERE id = ? FOR UPDATE", (int(site_id),)
        ).fetchone()
        if not existing:
            return None
        updates: list[tuple[str, Any]] = []
        if name is not None:
            clean_name = name.strip()
            if not clean_name:
                raise ValueError("Project site name cannot be empty.")
            updates.append(("name", clean_name[:120]))
        if geometry is not None:
            updates.append(("polygon_geojson", _validate_site_polygon_payload(geometry)))
            # A measurement must be reviewed again after the site's area changes.
            conn.execute("UPDATE project_sites SET tide_calibration = NULL WHERE id = ?", (int(site_id),))
        if notes is not None:
            updates.append(("notes", notes.strip() or None))
        if updates:
            set_sql = ", ".join(f"{column} = ?" for column, _ in updates)
            conn.execute(
                f"UPDATE site_zones SET {set_sql}, updated_at = ? WHERE id = ?",
                (*[value for _, value in updates], _manila_now().isoformat(timespec="seconds"), int(site_id)),
            )
        _replace_project_site_links(conn, int(site_id), "analyses", analysis_ids)
        _replace_project_site_links(conn, int(site_id), "planter_assignments", assignment_ids)
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_project_site(site_id)


def set_project_site_tide_calibration(site_id: int, calibration: Optional[dict]) -> Optional[dict]:
    """Persist an API-validated measurement independently of site assignment."""
    conn = _get_connection()
    try:
        row = conn.execute(
            "SELECT polygon_geojson FROM project_sites WHERE id = ? FOR UPDATE", (int(site_id),),
        ).fetchone()
        if not row:
            return None
        if calibration is not None:
            lat, lon = _site_geometry_centroid(row["polygon_geojson"])
            if lat is None or lon is None or (
                round(lat, 4), round(lon, 4)
            ) != (calibration["forecast_lat"], calibration["forecast_lon"]):
                raise ValueError("Site location changed. Refresh its forecast before saving calibration.")
        conn.execute(
            "UPDATE project_sites SET tide_calibration = CAST(? AS jsonb), updated_at = ? WHERE id = ?",
            (json.dumps(calibration, allow_nan=False) if calibration is not None else None,
             _manila_now().isoformat(timespec="seconds"), int(site_id)),
        )
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return get_project_site(site_id)


def delete_project_site(site_id: int) -> bool:
    conn = _get_connection()
    try:
        linked = conn.execute("""
            SELECT
                EXISTS(SELECT 1 FROM analyses WHERE site_zone_id = ?) AS has_analyses,
                EXISTS(SELECT 1 FROM planter_assignments WHERE site_zone_id = ?) AS has_assignments,
                EXISTS(SELECT 1 FROM planting_events WHERE site_zone_id = ?) AS has_events,
                EXISTS(SELECT 1 FROM planting_schedules WHERE project_site_id = ?) AS has_schedules
        """, (int(site_id), int(site_id), int(site_id), int(site_id))).fetchone()
        if linked and any(int(linked[key] or 0) for key in linked.keys()):
            raise ValueError(
                "This project site is linked to analyses, assignments, planting history, or schedules and cannot be deleted."
            )
        cursor = conn.execute("DELETE FROM site_zones WHERE id = ?", (int(site_id),))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def _clean_schedule_status(value: Any) -> str:
    clean = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if clean not in _VALID_PLANTING_SCHEDULE_STATUSES:
        allowed = ", ".join(sorted(_VALID_PLANTING_SCHEDULE_STATUSES))
        raise ValueError(f"status must be one of: {allowed}.")
    return clean


def _clean_schedule_text(
    value: Any,
    field: str,
    *,
    required: bool = False,
    max_length: int = 1000,
) -> Optional[str]:
    clean = str(value or "").strip()
    if required and not clean:
        raise ValueError(f"{field} is required.")
    return clean[:max_length] or None


def _clean_schedule_count(value: Any, field: str) -> Optional[int]:
    if value is None or value == "":
        return None
    try:
        clean = int(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"{field} must be a whole number or null.") from error
    if clean < 0:
        raise ValueError(f"{field} must be zero or greater.")
    return clean


def _planting_schedule_row(conn: Any, schedule_id: int) -> Optional[dict]:
    row = conn.execute("""
        SELECT ps.*, sz.name AS project_site_name, sz.polygon_geojson,
               o.name AS canonical_organization_name
        FROM planting_schedules ps
        LEFT JOIN site_zones sz ON sz.id = ps.project_site_id
        LEFT JOIN organizations o ON o.id = ps.organization_id
        WHERE ps.id = ?
    """, (int(schedule_id),)).fetchone()
    if not row:
        return None
    start_at = _parse_dashboard_datetime(row["start_at"])
    end_at = _parse_dashboard_datetime(row["end_at"])
    has_site = row["project_site_id"] is not None
    centroid_lat, centroid_lon = (
        _site_geometry_centroid(row["polygon_geojson"])
        if has_site and row["polygon_geojson"] else (None, None)
    )
    location_context = None
    tide_context = None
    if has_site:
        location_context = {
            "project_site_id": int(row["project_site_id"]),
            "project_site_name": row["project_site_name"],
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "source": "stable_project_site_boundary",
            "use": "operational_scheduling_only",
            "mortality_causation": False,
        }
        tide_context = {
            "latitude": centroid_lat,
            "longitude": centroid_lon,
            "association": "project_site_centroid",
            "use": "operational_scheduling_only",
            "mortality_causation": False,
            "readings_persisted": False,
        }
    return {
        "id": int(row["id"]),
        "project_site_id": int(row["project_site_id"]) if has_site else None,
        "project_site_name": row["project_site_name"],
        "organization_id": (
            int(row["organization_id"]) if row["organization_id"] is not None else None
        ),
        "organization": row["canonical_organization_name"] or row["organization"],
        "organization_name": row["canonical_organization_name"] or row["organization"],
        "inspection_interval_days": (
            int(row["inspection_interval_days"])
            if row["inspection_interval_days"] is not None
            else _DEFAULT_OPERATIONAL_INSPECTION_INTERVAL_DAYS
        ),
        "contact": row["contact"],
        "title": row["title"],
        "start_at": start_at.isoformat(timespec="seconds") if start_at else row["start_at"],
        "end_at": end_at.isoformat(timespec="seconds") if end_at else row["end_at"],
        # Calendar aliases keep simple single-day form clients backward-compatible.
        "date": start_at.date().isoformat() if start_at else None,
        "start_time": start_at.strftime("%H:%M") if start_at else None,
        "end_time": end_at.strftime("%H:%M") if end_at else None,
        "expected_planters": row["expected_planters"],
        "expected_participants": row["expected_planters"],
        "expected_seedlings": row["expected_seedlings"],
        "seedlings": row["expected_seedlings"],
        "status": row["status"],
        "notes": row["notes"],
        "timezone": "Asia/Manila",
        "site_location_context": location_context,
        "tide_context": tide_context,
        "created_by_user_id": row["created_by_user_id"],
        "updated_by_user_id": row["updated_by_user_id"],
        "created_at": row["created_at"],
        "updated_at": row["updated_at"],
    }


def list_planting_schedules(
    project_site_id: Optional[int] = None,
    status: Optional[str] = None,
    date_from: Optional[Any] = None,
    date_to: Optional[Any] = None,
) -> List[dict]:
    clauses: list[str] = []
    params: list[Any] = []
    if project_site_id is not None:
        clauses.append("project_site_id = ?")
        params.append(int(project_site_id))
    if status:
        clauses.append("status = ?")
        params.append(_clean_schedule_status(status))
    start = _parse_dashboard_datetime(date_from)
    end = _parse_dashboard_datetime(date_to, end_of_day=True)
    if start and end and start > end:
        raise ValueError("date_from must be on or before date_to.")
    if start:
        clauses.append("start_at >= ?")
        params.append(start.isoformat(timespec="seconds"))
    if end:
        clauses.append("start_at <= ?")
        params.append(end.isoformat(timespec="seconds"))
    where = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    conn = _get_connection()
    try:
        ids = conn.execute(
            f"SELECT id FROM planting_schedules{where} ORDER BY start_at, id",
            params,
        ).fetchall()
        return [
            schedule
            for row in ids
            if (schedule := _planting_schedule_row(conn, int(row["id"]))) is not None
        ]
    finally:
        conn.close()


def get_planting_schedule(schedule_id: int) -> Optional[dict]:
    conn = _get_connection()
    try:
        return _planting_schedule_row(conn, int(schedule_id))
    finally:
        conn.close()


def create_planting_schedule(
    *,
    organization: str,
    inspection_interval_days: int,
    organization_id: Optional[int] = None,
    title: str,
    start_at: Any,
    end_at: Any,
    contact: Optional[str] = None,
    expected_planters: Optional[int] = None,
    expected_seedlings: Optional[int] = None,
    status: str = "requested",
    notes: Optional[str] = None,
    created_by_user_id: Optional[int] = None,
) -> dict:
    clean_organization = _clean_organization_name(organization)
    clean_cadence = _clean_positive_interval(inspection_interval_days)
    if clean_cadence != 14:
        raise ValueError("Plant checks are fixed at 14-day intervals.")
    clean_title = _clean_schedule_text(title, "title", required=True, max_length=200)
    clean_contact = _clean_schedule_text(contact, "contact", max_length=300)
    clean_notes = _clean_schedule_text(notes, "notes", max_length=2000)
    clean_start = _parse_dashboard_datetime(start_at)
    clean_end = _parse_dashboard_datetime(end_at)
    if clean_start is None or clean_end is None:
        raise ValueError("start_at and end_at are required.")
    if clean_end <= clean_start:
        raise ValueError("end_at must be after start_at.")
    clean_status = _clean_schedule_status(status or "requested")
    if clean_status not in {"requested", "confirmed"}:
        raise ValueError("New schedules must have status requested or confirmed.")
    clean_planters = _clean_schedule_count(expected_planters, "expected_planters")
    clean_seedlings = _clean_schedule_count(expected_seedlings, "expected_seedlings")
    now = _manila_now().isoformat(timespec="seconds")

    conn = _get_connection()
    try:
        canonical = _resolve_registered_organization(
            conn, clean_organization, clean_cadence, organization_id,
        )
        cursor = conn.execute("""
            INSERT INTO planting_schedules (
                project_site_id, organization, organization_id,
                inspection_interval_days, contact, title, start_at, end_at,
                expected_planters, expected_seedlings, status, notes,
                created_by_user_id, updated_by_user_id, created_at, updated_at
            ) VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            canonical["name"], int(canonical["id"]), clean_cadence,
            clean_contact, clean_title,
            clean_start.isoformat(timespec="seconds"), clean_end.isoformat(timespec="seconds"),
            clean_planters, clean_seedlings, clean_status, clean_notes,
            created_by_user_id, created_by_user_id, now, now,
        ))
        schedule_id = int(cursor.lastrowid)
        conn.commit()
        return _planting_schedule_row(conn, schedule_id)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def update_planting_schedule(
    schedule_id: int,
    *,
    project_site_id: Any = _UNSET,
    organization: Any = _UNSET,
    organization_id: Any = _UNSET,
    inspection_interval_days: Any = _UNSET,
    contact: Any = _UNSET,
    title: Any = _UNSET,
    start_at: Any = _UNSET,
    end_at: Any = _UNSET,
    expected_planters: Any = _UNSET,
    expected_seedlings: Any = _UNSET,
    status: Any = _UNSET,
    notes: Any = _UNSET,
    updated_by_user_id: Optional[int] = None,
) -> Optional[dict]:
    conn = _get_connection()
    try:
        current = conn.execute(
            "SELECT * FROM planting_schedules WHERE id = ? FOR UPDATE", (int(schedule_id),),
        ).fetchone()
        if not current:
            conn.rollback()
            return None
        values = dict(current)
        current_status = str(current["status"] or "").strip().lower()
        current_org_key = _organization_key(current["organization"])
        if organization is not _UNSET:
            requested_org = _clean_organization_name(organization)
            if current["project_site_id"] is not None and _organization_key(requested_org) != current_org_key:
                raise ValueError("A linked schedule cannot be moved to a different organization.")
            values["organization"] = requested_org
        if inspection_interval_days is not _UNSET:
            values["inspection_interval_days"] = _clean_positive_interval(inspection_interval_days)
            if values["inspection_interval_days"] != 14:
                raise ValueError("Plant checks are fixed at 14-day intervals.")
        if title is not _UNSET:
            values["title"] = _clean_schedule_text(
                title, "title", required=True, max_length=200,
            )
        if contact is not _UNSET:
            values["contact"] = _clean_schedule_text(contact, "contact", max_length=300)
        if notes is not _UNSET:
            values["notes"] = _clean_schedule_text(notes, "notes", max_length=2000)
        if start_at is not _UNSET:
            parsed = _parse_dashboard_datetime(start_at)
            if parsed is None:
                raise ValueError("start_at is required.")
            values["start_at"] = parsed.isoformat(timespec="seconds")
        if end_at is not _UNSET:
            parsed = _parse_dashboard_datetime(end_at)
            if parsed is None:
                raise ValueError("end_at is required.")
            values["end_at"] = parsed.isoformat(timespec="seconds")
        if _parse_dashboard_datetime(values["end_at"]) <= _parse_dashboard_datetime(values["start_at"]):
            raise ValueError("end_at must be after start_at.")
        if expected_planters is not _UNSET:
            values["expected_planters"] = _clean_schedule_count(
                expected_planters, "expected_planters",
            )
        if expected_seedlings is not _UNSET:
            values["expected_seedlings"] = _clean_schedule_count(
                expected_seedlings, "expected_seedlings",
            )
        if status is not _UNSET:
            values["status"] = _clean_schedule_status(status)
        cadence = _clean_positive_interval(
            values.get("inspection_interval_days")
            or _DEFAULT_OPERATIONAL_INSPECTION_INTERVAL_DAYS
        )
        requested_organization_id = (
            (values.get("organization_id") if organization is _UNSET else None)
            if organization_id is _UNSET
            else organization_id
        )
        canonical = _resolve_registered_organization(
            conn, values["organization"], cadence, requested_organization_id,
        )
        values["organization"] = canonical["name"]
        values["organization_id"] = int(canonical["id"])

        if project_site_id is not _UNSET:
            if project_site_id in (None, ""):
                values["project_site_id"] = None
            else:
                if current_status != "confirmed" or values["status"] != "confirmed":
                    raise ValueError(
                        "Confirm and save the schedule before assigning a project site in a separate step."
                    )
                try:
                    values["project_site_id"] = int(project_site_id)
                except (TypeError, ValueError) as error:
                    raise ValueError("project_site_id must be a whole number.") from error
                selected_site = conn.execute(
                    "SELECT * FROM site_zones WHERE id = ?", (values["project_site_id"],),
                ).fetchone()
                if not selected_site:
                    raise ValueError("Selected project site was not found.")
                if selected_site["organization_id"] is None:
                    raise ValueError("Selected legacy project site has no organization owner.")
                if (
                    int(selected_site["organization_id"]) != int(values["organization_id"])
                ):
                    raise ValueError("Selected project site belongs to a different organization.")

        effective_site_id = values.get("project_site_id")
        should_update_site_cadence = bool(
            effective_site_id is not None
            and values["status"] == "confirmed"
            and (
                project_site_id is not _UNSET
                or inspection_interval_days is not _UNSET
            )
        )
        if should_update_site_cadence:
            now = _manila_now()
            conflict = conn.execute("""
                SELECT id, title, inspection_interval_days
                FROM planting_schedules
                WHERE project_site_id = ?
                  AND id != ?
                  AND status IN ('confirmed', 'in_progress')
                  AND end_at >= ?
                  AND inspection_interval_days IS NOT NULL
                  AND inspection_interval_days != ?
                ORDER BY start_at, id
                LIMIT 1
            """, (
                int(effective_site_id), int(schedule_id),
                now.isoformat(timespec="seconds"), cadence,
            )).fetchone()
            if conflict:
                raise ValueError(
                    "Project-site cadence conflict: another active or future confirmed "
                    f"schedule for this site uses {int(conflict['inspection_interval_days'])} days."
                )
            conn.execute("""
                UPDATE site_zones
                SET inspection_interval_days = ?, updated_at = ?
                WHERE id = ?
            """, (
                cadence, now.isoformat(timespec="seconds"), int(effective_site_id),
            ))

        updated_at = _manila_now().isoformat(timespec="seconds")
        conn.execute("""
            UPDATE planting_schedules
               SET project_site_id = ?, organization = ?, organization_id = ?,
                   inspection_interval_days = ?, contact = ?, title = ?,
                   start_at = ?, end_at = ?, expected_planters = ?,
                   expected_seedlings = ?, status = ?, notes = ?,
                   updated_by_user_id = ?, updated_at = ?
             WHERE id = ?
        """, (
            values["project_site_id"], values["organization"], values["organization_id"],
            cadence, values["contact"], values["title"], values["start_at"], values["end_at"],
            values["expected_planters"], values["expected_seedlings"],
            values["status"], values["notes"], updated_by_user_id,
            updated_at, int(schedule_id),
        ))
        conn.commit()
        return _planting_schedule_row(conn, int(schedule_id))
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def delete_planting_schedule(schedule_id: int) -> bool:
    conn = _get_connection()
    try:
        cursor = conn.execute(
            "DELETE FROM planting_schedules WHERE id = ?", (int(schedule_id),),
        )
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def _clean_inspection_intervals(value: Optional[Any]) -> List[int]:
    if value is None:
        return list(_DEFAULT_INSPECTION_INTERVALS)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("inspection_intervals_days must be a list of days.") from error
    if not isinstance(value, (list, tuple)):
        raise ValueError("inspection_intervals_days must be a list of days.")
    try:
        intervals = sorted({int(day) for day in value})
    except (TypeError, ValueError) as error:
        raise ValueError("Inspection intervals must be whole numbers of days.") from error
    if not intervals or any(day <= 0 or day > 3650 for day in intervals):
        raise ValueError("Inspection intervals must be between 1 and 3650 days.")
    return intervals


def _clean_inspection_weekdays(value: Optional[Any]) -> List[int]:
    if value is None:
        return list(_DEFAULT_INSPECTION_WEEKDAYS)
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except json.JSONDecodeError as error:
            raise ValueError("inspection_weekdays must be a list of ISO weekdays.") from error
    if not isinstance(value, (list, tuple)):
        raise ValueError("inspection_weekdays must be a list of ISO weekdays.")
    try:
        weekdays = sorted({int(day) for day in value})
    except (TypeError, ValueError) as error:
        raise ValueError("Inspection weekdays must be whole numbers.") from error
    if not weekdays or any(day < 1 or day > 7 for day in weekdays):
        raise ValueError("Inspection weekdays must use ISO values 1 (Monday) through 7 (Sunday).")
    return weekdays


def _snap_forward_to_inspection_weekday(
    target_due_at: datetime,
    inspection_weekdays: List[int],
) -> datetime:
    weekdays = _clean_inspection_weekdays(inspection_weekdays)
    days_forward = min((weekday - target_due_at.isoweekday()) % 7 for weekday in weekdays)
    return target_due_at + timedelta(days=days_forward)


def _record_value(record: Any, key: str, default: Any = None) -> Any:
    try:
        value = record[key]
    except (KeyError, IndexError, TypeError):
        return default
    return default if value is None else value


def _event_inspection_cadence(event: Any) -> int:
    return _DEFAULT_OPERATIONAL_INSPECTION_INTERVAL_DAYS


def _round_timing(
    planted_at: datetime,
    interval_days: int,
    inspection_weekdays: List[int],
) -> dict:
    planted_day = planted_at.astimezone(_MANILA_TZ).date()
    due_day = planted_day + timedelta(days=int(interval_days))
    target_due_at = datetime.combine(due_day, time.min, tzinfo=_MANILA_TZ)
    scheduled_for = datetime.combine(_monitoring_workday(due_day), time.min, tzinfo=_MANILA_TZ)
    return {
        "target_due_at": target_due_at,
        "scheduled_for": scheduled_for,
        "due_at": scheduled_for,
    }


def _expected_operational_intervals(
    planted_at: datetime,
    cadence_days: int,
    cutoff: datetime,
    history: Optional[List[dict]] = None,
    active: bool = True,
) -> List[int]:
    """Lazily derive expected target ages for one immutable planting cycle."""
    cadence = _clean_positive_interval(cadence_days)
    valid_history = {
        int(row["interval_days"])
        for row in (history or [])
        if int(row.get("interval_days") or 0) > 0
    }
    highest_valid_history = max(valid_history, default=0)
    if not active:
        maximum_age = highest_valid_history
    else:
        elapsed_days = max(0, (cutoff.astimezone(_MANILA_TZ).date()
                               - planted_at.astimezone(_MANILA_TZ).date()).days)
        maximum_age = max(elapsed_days, highest_valid_history)
    return sorted(set(range(cadence, maximum_age + 1, cadence)) | valid_history)


def get_dashboard_record_notices() -> dict:
    """Small, global completeness counts for the relevant dashboard tabs.

    Keep source records and historical planting events separate: combining
    them into a single issue count would imply unique affected seedlings.
    """
    conn = _get_connection()
    try:
        row = conn.execute("""
            SELECT
                (SELECT COUNT(*) FROM analyses WHERE site_zone_id IS NULL) +
                (SELECT COUNT(*) FROM planter_assignments WHERE site_zone_id IS NULL) AS missing_site_links,
                (SELECT COUNT(*) FROM analyses WHERE NULLIF(TRIM(species), '') IS NULL) +
                (SELECT COUNT(*) FROM planter_assignments WHERE NULLIF(TRIM(species), '') IS NULL) AS missing_species_links,
                (SELECT COUNT(*) FROM planting_events WHERE site_zone_id IS NULL) AS missing_event_site_links,
                (SELECT COUNT(*) FROM planting_events WHERE NULLIF(TRIM(species), '') IS NULL) AS missing_event_species_links,
                (SELECT COUNT(*) FROM planter_assignment_points
                 WHERE status = 'skipped' AND NULLIF(TRIM(skip_reason), '') IS NULL) AS skips_missing_reason
        """).fetchone()
        return {'scope': 'global', **{key: int(row[key]) for key in row.keys()}}
    finally:
        conn.close()


def get_dashboard_settings(year: Optional[int] = None, *, _conn: Any = None) -> dict:
    selected_year = int(year or _manila_now().year)
    if selected_year < 2000 or selected_year > 2200:
        raise ValueError("year must be between 2000 and 2200.")
    conn = _conn if _conn is not None else _get_connection()
    try:
        row = conn.execute("SELECT * FROM dashboard_settings WHERE year = ?", (selected_year,)).fetchone()
    finally:
        if _conn is None:
            conn.close()
    if not row:
        return {
            "year": selected_year,
            "annual_planting_target": None,
            "min_survival_target_pct": None,
            "inspection_intervals_days": list(_DEFAULT_INSPECTION_INTERVALS),
            "inspection_weekdays": list(_DEFAULT_INSPECTION_WEEKDAYS),
            "updated_at": None,
        }
    try:
        intervals = _clean_inspection_intervals(row["inspection_intervals_json"])
    except ValueError:
        intervals = list(_DEFAULT_INSPECTION_INTERVALS)
    try:
        weekdays = _clean_inspection_weekdays(row["inspection_weekdays_json"])
    except (ValueError, IndexError):
        weekdays = list(_DEFAULT_INSPECTION_WEEKDAYS)
    return {
        "year": selected_year,
        "annual_planting_target": (
            int(row["annual_planting_target"])
            if row["annual_planting_target"] is not None else None
        ),
        "min_survival_target_pct": (
            float(row["min_survival_target_pct"])
            if row["min_survival_target_pct"] is not None else None
        ),
        "inspection_intervals_days": intervals,
        "inspection_weekdays": weekdays,
        "updated_at": row["updated_at"],
    }


def update_dashboard_settings(
    year: Optional[int] = None,
    annual_planting_target: Any = _UNSET,
    min_survival_target_pct: Any = _UNSET,
    inspection_intervals_days: Any = _UNSET,
    inspection_weekdays: Any = _UNSET,
    updated_by_user_id: Optional[int] = None,
) -> dict:
    current = get_dashboard_settings(year)
    target = current["annual_planting_target"] if annual_planting_target is _UNSET else annual_planting_target
    survival = current["min_survival_target_pct"] if min_survival_target_pct is _UNSET else min_survival_target_pct
    intervals = current["inspection_intervals_days"] if inspection_intervals_days is _UNSET else inspection_intervals_days
    weekdays = current["inspection_weekdays"] if inspection_weekdays is _UNSET else inspection_weekdays
    if target is not None:
        try:
            target = int(target)
        except (TypeError, ValueError) as error:
            raise ValueError("annual_planting_target must be a whole number or null.") from error
        if target < 0:
            raise ValueError("annual_planting_target cannot be negative.")
    if survival is not None:
        try:
            survival = float(survival)
        except (TypeError, ValueError) as error:
            raise ValueError("min_survival_target_pct must be a number or null.") from error
        if not 0 <= survival <= 100:
            raise ValueError("min_survival_target_pct must be between 0 and 100.")
    clean_intervals = _clean_inspection_intervals(intervals)
    clean_weekdays = _clean_inspection_weekdays(weekdays)
    updated_at = _manila_now().isoformat(timespec="seconds")
    conn = _get_connection()
    try:
        conn.execute("""
            INSERT INTO dashboard_settings (
                year, annual_planting_target, min_survival_target_pct,
                inspection_intervals_json, inspection_weekdays_json,
                updated_at, updated_by_user_id
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(year) DO UPDATE SET
                annual_planting_target = excluded.annual_planting_target,
                min_survival_target_pct = excluded.min_survival_target_pct,
                inspection_intervals_json = excluded.inspection_intervals_json,
                inspection_weekdays_json = excluded.inspection_weekdays_json,
                updated_at = excluded.updated_at,
                updated_by_user_id = excluded.updated_by_user_id
        """, (
            current["year"], target, survival, json.dumps(clean_intervals),
            json.dumps(clean_weekdays),
            updated_at, updated_by_user_id,
        ))
        conn.commit()
    finally:
        conn.close()
    return get_dashboard_settings(current["year"])


def _safe_monitoring_photo_path(photo_path: Optional[str]) -> Optional[str]:
    if photo_path is None:
        return None
    clean = str(photo_path).strip().replace("\\", "/")
    if not clean:
        return None
    path = Path(clean)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError("photo_path must be a safe relative path.")
    return clean[:500]


def save_monitoring_photo_data_url(photo_data_url: str) -> str:
    """Validate and persist a small JPEG/PNG inspection photo outside SQLite."""
    match = re.fullmatch(
        r"data:image/(jpeg|jpg|png|webp);base64,([A-Za-z0-9+/=\r\n]+)",
        (photo_data_url or "").strip(),
        flags=re.IGNORECASE,
    )
    if not match:
        raise ValueError("photo_data_url must be a base64 JPEG, PNG, or WebP data URL.")
    encoded = re.sub(r"\s+", "", match.group(2))
    if len(encoded) > ((_MAX_MONITORING_PHOTO_BYTES + 2) // 3) * 4 + 4:
        raise ValueError("Inspection photo must be 5 MB or smaller.")
    try:
        payload = base64.b64decode(encoded, validate=True)
    except (binascii.Error, ValueError) as error:
        raise ValueError("Inspection photo contains invalid base64 data.") from error
    if len(payload) > _MAX_MONITORING_PHOTO_BYTES:
        raise ValueError("Inspection photo must be 5 MB or smaller.")
    if payload.startswith(b"\xff\xd8\xff"):
        extension = "jpg"
    elif payload.startswith(b"\x89PNG\r\n\x1a\n"):
        extension = "png"
    elif len(payload) >= 12 and payload[:4] == b"RIFF" and payload[8:12] == b"WEBP":
        extension = "webp"
    else:
        raise ValueError("Inspection photo content is not a valid JPEG, PNG, or WebP image.")
    upload_dir = Path(__file__).parent / "MangroVision_New" / "monitoring_uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_name = f"inspection_{uuid.uuid4().hex}.{extension}"
    (upload_dir / file_name).write_bytes(payload)
    return f"monitoring_uploads/{file_name}"


def _monitoring_observation_row(conn: Any, observation_id: int) -> Optional[dict]:
    row = conn.execute("""
        SELECT
            mo.id,
            mo.planting_event_id,
            pe.planting_point_id,
            pe.point_num,
            pe.assignment_id,
            pa.title AS assignment_title,
            pe.site_zone_id AS site_id,
            pe.site_zone_id AS project_site_id,
            sz.name AS site_name,
            sz.name AS project_site_name,
            pe.planter_id,
            COALESCE(pl.full_name, CASE WHEN pe.planted_by_user_id IS NOT NULL THEN 'LGU' END) AS planter_name,
            pe.species,
            pe.planted_at,
            pe.inspection_interval_days,
            pe.closed_at,
            pe.closure_reason,
            mo.interval_days,
            mo.inspected_at,
            mo.status,
            mo.condition,
            mo.height_cm,
            mo.photo_path,
            mo.notes,
            mo.actions_taken,
            mo.death_reason_category,
            mo.inspector_user_id,
            u.full_name AS inspector_name,
            mo.death_record_id,
            mo.created_at,
            mo.updated_at
        FROM monitoring_observations mo
        JOIN planting_events pe ON pe.id = mo.planting_event_id
        LEFT JOIN planter_assignments pa ON pa.id = pe.assignment_id
        LEFT JOIN site_zones sz ON sz.id = pe.site_zone_id
        LEFT JOIN planters pl ON pl.id = pe.planter_id
        LEFT JOIN users u ON u.id = mo.inspector_user_id
        WHERE mo.id = ?
    """, (int(observation_id),)).fetchone()
    if not row:
        return None
    result = dict(row)
    planted_at = _parse_dashboard_datetime(result.get("planted_at"))
    interval_days = int(result["interval_days"])
    cadence_days = _event_inspection_cadence(result)
    if planted_at:
        target_due_at = planted_at + timedelta(days=interval_days)
        weekdays = get_dashboard_settings(target_due_at.year)["inspection_weekdays"]
        timing = _round_timing(planted_at, interval_days, weekdays)
        result.update({
            "round_number": (
                interval_days // cadence_days
                if interval_days > 0 and interval_days % cadence_days == 0 else None
            ),
            "target_due_at": timing["target_due_at"].isoformat(timespec="seconds"),
            "scheduled_for": timing["scheduled_for"].isoformat(timespec="seconds"),
            "due_at": timing["due_at"].isoformat(timespec="seconds"),
            "cadence": {
                "interval_days": cadence_days,
                "inspection_weekdays": weekdays,
            },
        })
    return result


def upsert_monitoring_observation(
    planting_event_id: int,
    interval_days: int,
    status: str,
    inspector_user_id: int,
    condition: Optional[str] = None,
    height_cm: Optional[float] = None,
    notes: Optional[str] = None,
    death_reason_category: Optional[str] = None,
    photo_path: Optional[str] = None,
    photo_data_url: Optional[str] = None,
    inspected_at: Optional[Any] = None,
    actions_taken: Optional[str] = None,
) -> dict:
    clean_status = (status or "").strip().lower()
    if clean_status not in _VALID_MONITORING_STATUSES:
        raise ValueError("status must be alive, dead, or missing.")
    try:
        event_id = int(planting_event_id)
        interval = int(interval_days)
    except (TypeError, ValueError) as error:
        raise ValueError("planting_event_id and interval_days must be whole numbers.") from error
    if interval <= 0:
        raise ValueError("interval_days must be a positive cumulative target age.")
    inspected_dt = _parse_dashboard_datetime(inspected_at) or _manila_now()
    if inspected_dt > _manila_now() + timedelta(minutes=5):
        raise ValueError("Inspection time cannot be in the future.")
    clean_condition = (condition or "").strip()[:120] or None
    clean_notes = (notes or "").strip()[:1000] or None
    clean_actions_taken = (actions_taken or "").strip()[:2000] or None
    clean_photo_path = _safe_monitoring_photo_path(photo_path)
    clean_height = None
    if height_cm is not None:
        try:
            clean_height = float(height_cm)
        except (TypeError, ValueError) as error:
            raise ValueError("height_cm must be a number.") from error
        if clean_height < 0 or clean_height > 10_000:
            raise ValueError("height_cm must be between 0 and 10000.")
    reason_key = (death_reason_category or "").strip().lower() or None
    if clean_status == "dead":
        if reason_key not in DEATH_REASON_CATEGORIES:
            raise ValueError(
                "A valid death_reason_category is required when status is dead."
            )
    else:
        reason_key = None

    created_photo_path = None
    if photo_data_url:
        clean_photo_path = save_monitoring_photo_data_url(photo_data_url)
        created_photo_path = clean_photo_path

    conn = None
    try:
        conn = _get_connection()
        conn.execute('SELECT id FROM planting_points WHERE id = (SELECT planting_point_id FROM planting_events WHERE id = ?) FOR UPDATE', (event_id,)).fetchone()
        event = conn.execute("""
            SELECT pe.*, pp.status AS point_status, pp.deleted_at, pp.death_at,
                   pp.death_reason, pp.death_reason_category,
                   pa.planter_id AS current_planter_id,
                   pl.full_name AS current_planter_name,
                   pa.species AS current_species
            FROM planting_events pe
            LEFT JOIN planting_points pp ON pp.id = pe.planting_point_id
            LEFT JOIN planter_assignments pa ON pa.id = pe.assignment_id
            LEFT JOIN planters pl ON pl.id = pa.planter_id
            WHERE pe.id = ?
            FOR UPDATE OF pe
        """, (event_id,)).fetchone()
        if not event:
            raise ValueError("Planting event was not found.")
        planted_dt = _parse_dashboard_datetime(event["planted_at"])
        if not planted_dt:
            raise ValueError("Planting event has an invalid planting timestamp.")
        existing = conn.execute("""
            SELECT * FROM monitoring_observations
            WHERE planting_event_id = ? AND interval_days = ?
        """, (event_id, interval)).fetchone()
        observed_dt = (
            _parse_dashboard_datetime(existing["inspected_at"])
            if existing is not None
            else inspected_dt
        ) or inspected_dt
        cadence_days = _event_inspection_cadence(event)
        scheduled_due_at = _round_timing(planted_dt, interval, [1])["due_at"]
        if existing is None and interval % cadence_days != 0:
            raise ValueError(
                f"interval_days must be a multiple of this planting cycle's "
                f"{cadence_days}-day inspection cadence."
            )
        if existing is None and inspected_dt < scheduled_due_at:
            raise ValueError(
                f"The round cannot be inspected before its scheduled weekday "
                f"({scheduled_due_at.isoformat(timespec='seconds')})."
            )
        if event["closed_at"] and not existing:
            raise ValueError("This planting cycle is closed and cannot receive a new inspection round.")

        point_id = event["planting_point_id"]
        death_record_id = existing["death_record_id"] if existing else None
        correcting_dead_to_non_outcome = bool(
            existing and existing["status"] == "dead" and clean_status != "dead"
        )
        if correcting_dead_to_non_outcome and conn.execute("""SELECT 1 FROM monitoring_death_locations WHERE planting_event_id = ? AND revoked_at IS NULL
            UNION ALL SELECT 1 FROM replanting_requests WHERE planting_event_id = ?""", (event_id, event_id)).fetchone():
            raise ValueError('Organization monitoring or replacement work is linked to this death; review its location history first.')
        if correcting_dead_to_non_outcome:
            linked_death = None
            if death_record_id is not None:
                linked_death = conn.execute(
                    "SELECT * FROM point_death_records WHERE id = ? AND planting_event_id = ?",
                    (death_record_id, event_id),
                ).fetchone()
            if linked_death:
                latest_event = conn.execute("""
                    SELECT id FROM planting_events
                    WHERE planting_point_id = ?
                    ORDER BY planted_at DESC, id DESC LIMIT 1
                """, (point_id,)).fetchone() if point_id is not None else None
                if (
                    latest_event
                    and int(latest_event["id"]) == event_id
                    and event["death_at"] == linked_death["death_at"]
                ):
                    conn.execute("""
                        UPDATE planting_points
                        SET death_at = NULL, death_reason = NULL,
                            death_reason_category = NULL, death_notes = NULL
                        WHERE id = ?
                    """, (point_id,))
                conn.execute("DELETE FROM point_death_records WHERE id = ?", (linked_death["id"],))
            death_record_id = None
            remaining_death = conn.execute(
                "SELECT 1 FROM point_death_records WHERE planting_event_id = ? LIMIT 1",
                (event_id,),
            ).fetchone()
            if not remaining_death and event["closure_reason"] in {"monitoring_death", "recorded_death"}:
                conn.execute("""
                    UPDATE planting_events SET closed_at = NULL, closure_reason = NULL
                    WHERE id = ?
                """, (event_id,))

        recorded_death_dt = _parse_dashboard_datetime(event["death_at"])
        if (
            clean_status in {"alive", "missing"}
            and recorded_death_dt
            and observed_dt >= recorded_death_dt
            and not correcting_dead_to_non_outcome
        ):
            raise ValueError(
                "This planting cycle was already recorded dead by this inspection time."
            )
        if clean_status == "dead":
            if point_id is None or event["deleted_at"]:
                raise ValueError("A removed planting point cannot receive a death observation.")
            latest_event = conn.execute("""
                SELECT id FROM planting_events
                WHERE planting_point_id = ?
                ORDER BY planted_at DESC, id DESC LIMIT 1
            """, (point_id,)).fetchone()
            correcting_existing_death = bool(existing and existing["status"] == "dead")
            if not correcting_existing_death:
                if not latest_event or int(latest_event["id"]) != event_id:
                    raise ValueError("Only the current planting cycle can be marked dead.")
                if event["point_status"] != "planted":
                    raise ValueError("Only a currently planted point can be marked dead.")

            death_at = (
                observed_dt.isoformat(timespec="microseconds")
                if correcting_existing_death
                else event["death_at"] or observed_dt.isoformat(timespec="microseconds")
            )
            label = DEATH_REASON_CATEGORIES[reason_key]
            full_reason = f"{label}: {clean_notes}" if clean_notes else label
            if not correcting_existing_death or (
                latest_event
                and int(latest_event["id"]) == event_id
                and event["death_at"]
            ):
                conn.execute("""
                    UPDATE planting_points
                    SET death_at = ?, death_reason = ?, death_reason_category = ?, death_notes = ?
                    WHERE id = ?
                """, (death_at, full_reason, reason_key, clean_notes, point_id))

            death_row = conn.execute(
                "SELECT * FROM point_death_records WHERE planting_event_id = ?",
                (event_id,),
            ).fetchone()
            if not death_row and event["death_at"]:
                unlinked = conn.execute("""
                    SELECT id FROM point_death_records
                    WHERE planting_point_id = ? AND planting_event_id IS NULL
                    ORDER BY death_at DESC, id DESC LIMIT 1
                """, (point_id,)).fetchone()
                if unlinked:
                    conn.execute(
                        "UPDATE point_death_records SET planting_event_id = ? WHERE id = ?",
                        (event_id, unlinked["id"]),
                    )
                    death_row = unlinked
            if not death_row:
                cursor = conn.execute("""
                    INSERT INTO point_death_records (
                        planting_point_id, assignment_id, planting_event_id,
                        death_at, reason_category, reason_label, notes,
                        planter_id, planter_name, species
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    point_id, event["assignment_id"], event_id, death_at,
                    reason_key, label, clean_notes,
                    event["planter_id"] or event["current_planter_id"],
                    event["current_planter_name"],
                    event["species"] or event["current_species"],
                ))
                death_record_id = int(cursor.lastrowid)
            else:
                death_record_id = int(death_row["id"])
                conn.execute("""
                    UPDATE point_death_records
                    SET reason_category = ?, reason_label = ?, notes = ?
                    WHERE id = ?
                """, (reason_key, label, clean_notes, death_record_id))
            conn.execute("""
                UPDATE planting_events
                SET closed_at = COALESCE(closed_at, ?),
                    closure_reason = CASE
                        WHEN closure_reason = 'death_reset_to_planned' THEN closure_reason
                        ELSE 'monitoring_death'
                    END
                WHERE id = ?
            """, (death_at, event_id))

        observed_text = observed_dt.isoformat(timespec="seconds")
        updated_text = inspected_dt.isoformat(timespec="seconds")
        conn.execute("""
            INSERT INTO monitoring_observations (
                planting_event_id, interval_days, inspected_at, status,
                condition, height_cm, photo_path, notes, actions_taken,
                death_reason_category, inspector_user_id, death_record_id,
                updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(planting_event_id, interval_days) DO UPDATE SET
                status = excluded.status,
                condition = excluded.condition,
                height_cm = excluded.height_cm,
                photo_path = COALESCE(excluded.photo_path, monitoring_observations.photo_path),
                notes = excluded.notes,
                actions_taken = excluded.actions_taken,
                death_reason_category = excluded.death_reason_category,
                inspector_user_id = excluded.inspector_user_id,
                death_record_id = excluded.death_record_id,
                updated_at = excluded.updated_at
        """, (
            event_id, interval, observed_text, clean_status, clean_condition,
            clean_height, clean_photo_path, clean_notes, clean_actions_taken, reason_key,
            int(inspector_user_id), death_record_id, updated_text,
        ))
        saved = conn.execute("""
            SELECT id FROM monitoring_observations
            WHERE planting_event_id = ? AND interval_days = ?
        """, (event_id, interval)).fetchone()
        if death_record_id is not None:
            conn.execute("""
                UPDATE point_death_records
                SET monitoring_observation_id = ?
                WHERE id = ?
            """, (int(saved["id"]), death_record_id))
        planter = conn.execute("SELECT organization_id FROM planters WHERE id = ?",
                               (event["planter_id"],)).fetchone() if event["planter_id"] else None
        append_activity(
            conn, action="monitoring.seedling_inspected", actor_type="staff",
            actor_user_id=int(inspector_user_id),
            organization_id=planter["organization_id"] if planter else None,
            project_site_id=event["site_zone_id"],
            planting_point_id=event["planting_point_id"], planting_event_id=event_id,
            summary=f"LGU recorded a {clean_status} inspection for point #{event['point_num']}.",
            details={"interval_days": interval, "status": clean_status,
                     "observation_id": int(saved["id"])},
        )
        conn.commit()
        return _monitoring_observation_row(conn, int(saved["id"]))
    except Exception:
        if conn is not None:
            conn.rollback()
        if created_photo_path:
            absolute_photo = Path(__file__).parent / "MangroVision_New" / created_photo_path
            try:
                absolute_photo.unlink(missing_ok=True)
            except OSError:
                pass
        raise
    finally:
        if conn is not None:
            conn.close()


def _resolve_project_site_filter(
    site_id: Optional[int],
    project_site_id: Optional[int],
) -> Optional[int]:
    if site_id is None and project_site_id is None:
        return None
    if site_id is not None and project_site_id is not None and int(site_id) != int(project_site_id):
        raise ValueError("site_id and project_site_id must identify the same project site.")
    return int(project_site_id if project_site_id is not None else site_id)


def list_monitoring_observations(
    planting_event_id: Optional[int] = None,
    site_id: Optional[int] = None,
    assignment_id: Optional[int] = None,
    species: Optional[str] = None,
    planter_id: Optional[int] = None,
    status: Optional[str] = None,
    date_from: Optional[Any] = None,
    date_to: Optional[Any] = None,
    limit: Optional[int] = None,
    project_site_id: Optional[int] = None,
) -> List[dict]:
    selected_site_id = _resolve_project_site_filter(site_id, project_site_id)
    clauses = []
    params: list[Any] = []
    mappings = [
        (planting_event_id, "mo.planting_event_id = ?"),
        (selected_site_id, "pe.site_zone_id = ?"),
        (assignment_id, "pe.assignment_id = ?"),
        (planter_id, "pe.planter_id = ?"),
    ]
    for value, clause in mappings:
        if value is not None:
            clauses.append(clause)
            params.append(int(value))
    if species:
        clauses.append("LOWER(TRIM(COALESCE(pe.species, ''))) = LOWER(TRIM(?))")
        params.append(species)
    if status:
        clean_status = status.strip().lower()
        if clean_status not in _VALID_MONITORING_STATUSES:
            raise ValueError("status must be alive, dead, or missing.")
        clauses.append("mo.status = ?")
        params.append(clean_status)
    start = _parse_dashboard_datetime(date_from)
    end = _parse_dashboard_datetime(date_to, end_of_day=True)
    if start:
        clauses.append("mo.inspected_at >= ?")
        params.append(start.isoformat())
    if end:
        clauses.append("mo.inspected_at <= ?")
        params.append(end.isoformat())
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    limit_sql = ""
    if limit is not None:
        try:
            clean_limit = int(limit)
        except (TypeError, ValueError) as error:
            raise ValueError("limit must be a whole number.") from error
        if clean_limit < 1 or clean_limit > 1000:
            raise ValueError("limit must be between 1 and 1000.")
        limit_sql = " LIMIT ?"
        params.append(clean_limit)
    conn = _get_connection()
    try:
        ids = conn.execute(f"""
            SELECT mo.id
            FROM monitoring_observations mo
            JOIN planting_events pe ON pe.id = mo.planting_event_id
            {where}
            ORDER BY mo.inspected_at DESC, mo.id DESC
            {limit_sql}
        """, params).fetchall()
        return [
            _monitoring_observation_row(conn, int(row["id"]))
            for row in ids
        ]
    finally:
        conn.close()


def get_due_monitoring_inspections(
    as_of: Optional[Any] = None,
    site_id: Optional[int] = None,
    assignment_id: Optional[int] = None,
    species: Optional[str] = None,
    planter_id: Optional[int] = None,
    upcoming_days: int = 30,
    project_site_id: Optional[int] = None,
    *,
    _conn: Any = None,
    _settings: Optional[dict] = None,
) -> dict:
    as_of_dt = _parse_dashboard_datetime(as_of) or _manila_now()
    selected_site_id = _resolve_project_site_filter(site_id, project_site_id)
    # Dashboard callers already have a connection and may have read this year's
    # settings. Reuse only within this request; no additional stale-data cache.
    settings = _settings if _settings is not None else get_dashboard_settings(as_of_dt.year, _conn=_conn)
    legacy_intervals = settings["inspection_intervals_days"]
    inspection_weekdays = settings["inspection_weekdays"]
    clauses = []
    params: list[Any] = []
    for value, clause in (
        (selected_site_id, "pe.site_zone_id = ?"),
        (assignment_id, "pe.assignment_id = ?"),
        (planter_id, "pe.planter_id = ?"),
    ):
        if value is not None:
            clauses.append(clause)
            params.append(int(value))
    if species:
        clauses.append("LOWER(TRIM(COALESCE(pe.species, ''))) = LOWER(TRIM(?))")
        params.append(species)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    conn = _conn if _conn is not None else _get_connection()
    try:
        events = conn.execute(f"""
            SELECT pe.*, pp.point_num AS current_point_num, a.image_name,
                   pp.status AS current_point_status,
                   pa.title AS assignment_title, sz.name AS site_name,
                   sz.organization_id, o.name AS organization_name,
                   latest.status AS latest_status,
                   death_obs.interval_days AS death_interval,
                   EXISTS(
                       SELECT 1 FROM point_death_records pdr
                       WHERE pdr.planting_event_id = pe.id
                   ) AS has_death_record,
                   CASE WHEN pe.id = (
                       SELECT newer.id FROM planting_events newer
                       WHERE newer.planting_point_id = pe.planting_point_id
                       ORDER BY newer.planted_at DESC, newer.id DESC LIMIT 1
                   ) THEN 1 ELSE 0 END AS is_current_cycle
            FROM planting_events pe
            LEFT JOIN planting_points pp ON pp.id = pe.planting_point_id
            LEFT JOIN analyses a ON a.id = pp.analysis_id
            LEFT JOIN planter_assignments pa ON pa.id = pe.assignment_id
            LEFT JOIN site_zones sz ON sz.id = pe.site_zone_id
            LEFT JOIN organizations o ON o.id = sz.organization_id
            LEFT JOIN monitoring_observations latest
                   ON latest.id = (
                       SELECT mo2.id FROM monitoring_observations mo2
                       WHERE mo2.planting_event_id = pe.id
                       ORDER BY mo2.interval_days DESC, mo2.inspected_at DESC, mo2.id DESC
                       LIMIT 1
                   )
            LEFT JOIN monitoring_observations death_obs
                   ON death_obs.id = (
                       SELECT mo3.id FROM monitoring_observations mo3
                       WHERE mo3.planting_event_id = pe.id AND mo3.status = 'dead'
                       ORDER BY mo3.interval_days, mo3.id LIMIT 1
                   )
            {where}
            ORDER BY pe.planted_at, pe.id
        """, params).fetchall()
        observed = {
            (int(row["planting_event_id"]), int(row["interval_days"]))
            for row in conn.execute(
                "SELECT planting_event_id, interval_days FROM monitoring_observations"
            ).fetchall()
        }
    finally:
        if _conn is None:
            conn.close()

    cutoff = as_of_dt + timedelta(days=max(0, min(int(upcoming_days), 365)))
    pending: list[dict] = []
    for event in events:
        if (
            event["closed_at"]
            or event["has_death_record"]
            or event["death_interval"] is not None
            or str(event["latest_status"] or "").lower() == "dead"
            or not event["is_current_cycle"]
            or event["current_point_status"] != "planted"
        ):
            continue
        planted_dt = _parse_dashboard_datetime(event["planted_at"])
        if not planted_dt or planted_dt > as_of_dt:
            continue
        cadence_days = _event_inspection_cadence(event)
        intervals = _expected_operational_intervals(
            planted_dt, cadence_days, cutoff, active=True,
        )
        for interval in intervals:
            if (int(event["id"]), interval) in observed:
                continue
            timing = _round_timing(planted_dt, interval, inspection_weekdays)
            target_due_at = timing["target_due_at"]
            scheduled_for = timing["scheduled_for"]
            overdue_days = max(0, (as_of_dt.date() - scheduled_for.date()).days)
            pending.append({
                "planting_event_id": int(event["id"]),
                "planting_point_id": (
                    int(event["planting_point_id"])
                    if event["planting_point_id"] is not None else None
                ),
                "point_num": (
                    int(event["point_num"] or event["current_point_num"])
                    if (event["point_num"] is not None or event["current_point_num"] is not None)
                    else None
                ),
                "image_name": event["image_name"],
                "site_id": int(event["site_zone_id"]) if event["site_zone_id"] is not None else None,
                "project_site_id": int(event["site_zone_id"]) if event["site_zone_id"] is not None else None,
                "site_name": event["site_name"],
                "project_site_name": event["site_name"],
                "organization_id": (
                    int(event["organization_id"])
                    if event["organization_id"] is not None else None
                ),
                "organization_name": event["organization_name"],
                "organization": event["organization_name"],
                "planting_source": event["source"],
                "planted_by_user_id": event["planted_by_user_id"],
                "assignment_id": int(event["assignment_id"]) if event["assignment_id"] is not None else None,
                "assignment_title": event["assignment_title"],
                "species": event["species"],
                "planted_at": planted_dt.isoformat(timespec="seconds"),
                "interval_days": interval,
                "inspection_interval_days": cadence_days,
                "round_number": interval // cadence_days,
                "target_due_at": target_due_at.isoformat(timespec="seconds"),
                "scheduled_for": scheduled_for.isoformat(timespec="seconds"),
                "due_at": scheduled_for.isoformat(timespec="seconds"),
                "cadence": {
                    "interval_days": cadence_days,
                    "inspection_weekdays": inspection_weekdays,
                },
                "days_overdue": overdue_days,
                "latest_status": event["latest_status"],
            })
    pending.sort(key=lambda item: (item["due_at"], item["planting_event_id"], item["interval_days"]))
    due = sum(1 for item in pending if _parse_dashboard_datetime(item["due_at"]) <= as_of_dt)
    overdue = sum(1 for item in pending if item["days_overdue"] > 0)
    upcoming = sum(1 for item in pending if _parse_dashboard_datetime(item["due_at"]) > as_of_dt)
    return {
        "as_of": as_of_dt.isoformat(timespec="seconds"),
        # Retained for scientific/reporting compatibility only. Operational
        # rounds above are event-specific cadence multiples.
        "intervals_days": legacy_intervals,
        "inspection_weekdays": inspection_weekdays,
        "summary": {"due": due, "overdue": overdue, "upcoming": upcoming},
        "observations_due": pending,
    }


def _monitoring_project_summary(
    project_site_id: int,
    project_site_name: str,
    notes: Optional[str],
    events: List[dict],
    latest_by_event: dict[int, dict],
    due_rows: List[dict],
    schedules: List[dict],
    as_of_dt: datetime,
) -> dict:
    statuses = [
        latest_by_event[int(event["id"])]["status"]
        for event in events
        if int(event["id"]) in latest_by_event
    ]
    alive = statuses.count("alive")
    dead = statuses.count("dead")
    missing = statuses.count("missing")
    verified = alive + dead
    overdue = sum(1 for row in due_rows if int(row.get("days_overdue") or 0) > 0)
    due = sum(
        1 for row in due_rows
        if (_parse_dashboard_datetime(row.get("due_at")) or as_of_dt) <= as_of_dt
    )
    upcoming = max(0, len(due_rows) - due)
    planting_point_ids = {
        int(event["planting_point_id"])
        for event in events if event.get("planting_point_id") is not None
    }
    planter_groups: dict[int, dict] = {}
    for event in events:
        if event.get("planter_id") is None:
            continue
        planter_id = int(event["planter_id"])
        planter = planter_groups.setdefault(planter_id, {
            "id": planter_id,
            "name": event.get("planter_name") or f"Planter {planter_id}",
            "planting_event_count": 0,
        })
        planter["planting_event_count"] += 1
    planters = sorted(
        planter_groups.values(),
        key=lambda row: (-int(row["planting_event_count"]), row["name"].casefold()),
    )
    upcoming_schedules = sum(
        1 for schedule in schedules
        if schedule.get("status") not in {"completed", "cancelled"}
        and (_parse_dashboard_datetime(schedule.get("end_at")) or as_of_dt) >= as_of_dt
    )
    return {
        "project_site_id": int(project_site_id),
        "site_id": int(project_site_id),
        "project_site_name": project_site_name,
        "site_name": project_site_name,
        "notes": notes,
        "planting_event_count": len(events),
        "planted_point_count": len(planting_point_ids),
        "planter_count": len(planters),
        "planters": planters,
        "schedule_count": len(schedules),
        "upcoming_schedule_count": upcoming_schedules,
        "latest_alive": alive,
        "latest_dead": dead,
        "latest_missing": missing,
        "latest_observed": len(statuses),
        "latest_unverified": max(0, len(events) - len(statuses)),
        "verified_sample_size": verified,
        "latest_descriptive_survival_rate_pct": (
            round(alive / verified * 100.0, 2) if verified else None
        ),
        "latest_status_scope": "mixed_inspection_rounds_latest_per_planting_event",
        "latest_status_cohort_comparable": False,
        "due_inspections": due,
        "overdue_inspections": overdue,
        "upcoming_inspections": upcoming,
        "totals": {
            "planting_events": len(events),
            "planted_points": len(planting_point_ids),
            "planters": len(planters),
            "schedules": len(schedules),
        },
        # Compatibility shape for the monitoring project cards. These are
        # verified observations, but they deliberately represent each event's
        # latest recorded round and therefore are not a same-age survival
        # cohort or a planter performance score.
        "verified": {
            "alive": alive,
            "dead": dead,
            "missing": missing,
            "unverified": max(0, len(events) - len(statuses)),
            "scope": "mixed_inspection_rounds_latest_per_planting_event",
            "cohort_comparable": False,
        },
        "latest_observations": {
            "alive": alive,
            "dead": dead,
            "missing": missing,
            "observed": len(statuses),
            "unverified": max(0, len(events) - len(statuses)),
            "verified_sample_size": verified,
            "descriptive_survival_rate_pct": (
                round(alive / verified * 100.0, 2) if verified else None
            ),
            "scope": "mixed_inspection_rounds_latest_per_planting_event",
            "cohort_comparable": False,
            "label": "Latest verified status by planting event (mixed inspection rounds)",
        },
        "inspection_schedule": {
            "due": due,
            "overdue": overdue,
            "upcoming": upcoming,
        },
    }


def list_monitoring_projects(
    as_of: Optional[Any] = None,
    upcoming_days: int = 30,
) -> dict:
    """List stable project sites that have at least one defensible planting event."""
    as_of_dt = _parse_dashboard_datetime(as_of) or _manila_now()
    try:
        clean_upcoming_days = max(0, min(int(upcoming_days), 365))
    except (TypeError, ValueError) as error:
        raise ValueError("upcoming_days must be a whole number.") from error

    conn = _get_connection()
    try:
        site_rows = [dict(row) for row in conn.execute("""
            SELECT sz.id, sz.name, sz.notes, sz.polygon_geojson,
                   sz.organization_id, o.name AS organization_name
            FROM site_zones sz
            LEFT JOIN organizations o ON o.id = sz.organization_id
            ORDER BY COALESCE(o.name, sz.name) COLLATE NOCASE, sz.name COLLATE NOCASE, sz.id
        """).fetchall()]
        raw_events = [dict(row) for row in conn.execute("""
            SELECT pe.*, COALESCE(pl.full_name, CASE WHEN pe.planted_by_user_id IS NOT NULL THEN 'LGU' END) AS planter_name
            FROM planting_events pe
            LEFT JOIN planters pl ON pl.id = pe.planter_id
            WHERE site_zone_id IS NOT NULL
              AND COALESCE(pe.closure_reason, '') != 'completion_reversed'
            ORDER BY pe.planted_at, pe.id
        """).fetchall()]
        raw_observations = [dict(row) for row in conn.execute("""
            SELECT mo.*
            FROM monitoring_observations mo
            JOIN planting_events pe ON pe.id = mo.planting_event_id
            WHERE pe.site_zone_id IS NOT NULL
            ORDER BY mo.interval_days, mo.inspected_at, mo.id
        """).fetchall()]
    finally:
        conn.close()

    events = [
        event for event in raw_events
        if (_parse_dashboard_datetime(event.get("planted_at")) or as_of_dt) <= as_of_dt
    ]
    event_ids = {int(event["id"]) for event in events}
    latest_by_event: dict[int, dict] = {}
    for observation in raw_observations:
        event_id = int(observation["planting_event_id"])
        inspected_at = _parse_dashboard_datetime(observation.get("inspected_at"))
        if event_id not in event_ids or (inspected_at and inspected_at > as_of_dt):
            continue
        current = latest_by_event.get(event_id)
        current_key = (
            int(current["interval_days"]), str(current["inspected_at"]), int(current["id"]),
        ) if current else None
        candidate_key = (
            int(observation["interval_days"]), str(observation["inspected_at"]), int(observation["id"]),
        )
        if current_key is None or candidate_key > current_key:
            latest_by_event[event_id] = observation

    due_payload = get_due_monitoring_inspections(
        as_of=as_of_dt, upcoming_days=clean_upcoming_days,
    )
    due_by_site: dict[int, list[dict]] = {}
    for row in due_payload["observations_due"]:
        if row.get("project_site_id") is not None:
            due_by_site.setdefault(int(row["project_site_id"]), []).append(row)

    projects = []
    for site in site_rows:
        site_id = int(site["id"])
        site_events = [event for event in events if int(event["site_zone_id"]) == site_id]
        if not site_events:
            continue
        site_schedules = list_planting_schedules(project_site_id=site_id)
        summary = _monitoring_project_summary(
            site_id, site["name"], site.get("notes"), site_events,
            latest_by_event, due_by_site.get(site_id, []), site_schedules, as_of_dt,
        )
        summary["organization_id"] = (
            int(site["organization_id"])
            if site.get("organization_id") is not None else None
        )
        summary["organization_name"] = site.get("organization_name")
        summary["organization"] = site.get("organization_name")
        centroid_lat, centroid_lon = _site_geometry_centroid(site.get("polygon_geojson"))
        summary["centroid_lat"] = centroid_lat
        summary["centroid_lon"] = centroid_lon
        projects.append(summary)

    projects.sort(key=lambda row: (
        -int(row["overdue_inspections"]), row["project_site_name"].casefold(),
    ))
    return {
        "as_of": as_of_dt.isoformat(timespec="seconds"),
        "timezone": "Asia/Manila",
        "upcoming_days": clean_upcoming_days,
        "projects": projects,
    }


def _scheduled_monitoring_rounds(
    event: dict,
    history: List[dict],
    inspection_weekdays: List[int],
    as_of_dt: datetime,
    project_site_id: int,
    project_site_name: str,
    upcoming_days: int = 30,
) -> List[dict]:
    """Lazily build recurring rounds through lookahead and valid history."""
    planted_at = _parse_dashboard_datetime(event.get("planted_at"))
    if planted_at is None:
        return []
    observations_by_interval = {
        int(observation["interval_days"]): observation
        for observation in history
    }
    closed_at = _parse_dashboard_datetime(event.get("closed_at"))
    has_recorded_death = bool(event.get("has_death_record")) or any(
        str(observation.get("status") or "").lower() == "dead"
        for observation in history
    )
    inactive_cycle = (
        (closed_at is not None and closed_at <= as_of_dt)
        or event.get("current_point_status") != "planted"
        or bool(event.get("deleted_at"))
        or has_recorded_death
        or not bool(event.get("is_current_cycle", True))
    )
    cadence_days = _event_inspection_cadence(event)
    cutoff = as_of_dt + timedelta(days=max(0, min(int(upcoming_days), 365)))
    intervals = _expected_operational_intervals(
        planted_at,
        cadence_days,
        cutoff,
        history=history,
        active=not inactive_cycle,
    )
    if inactive_cycle:
        # Closed/dead/superseded cycles retain their valid inspection history,
        # but never surface missed or future rounds as actionable field work.
        intervals = [
            interval for interval in intervals
            if interval in observations_by_interval
        ]
    rounds = []
    for interval in intervals:
        interval_days = int(interval)
        timing = _round_timing(planted_at, interval_days, inspection_weekdays)
        target_due_at = timing["target_due_at"]
        scheduled_for = timing["scheduled_for"]
        observation = observations_by_interval.get(interval_days)
        if observation is not None:
            state = "completed"
        elif scheduled_for < as_of_dt and scheduled_for.date() < as_of_dt.date():
            state = "overdue"
        elif scheduled_for <= as_of_dt:
            state = "due"
        else:
            state = "upcoming"
        days_overdue = (
            max(0, (as_of_dt.date() - scheduled_for.date()).days)
            if state == "overdue" else 0
        )
        row = {
            "planting_event_id": int(event["id"]),
            "planting_point_id": (
                int(event["planting_point_id"])
                if event.get("planting_point_id") is not None else None
            ),
            "point_num": (
                int(event["point_num"])
                if event.get("point_num") is not None else None
            ),
            "project_site_id": int(project_site_id),
            "site_id": int(project_site_id),
            "project_site_name": project_site_name,
            "site_name": project_site_name,
            "planter_id": (
                int(event["planter_id"])
                if event.get("planter_id") is not None else None
            ),
            "planter_name": event.get("planter_name"),
            "species": event.get("species"),
            "planted_at": planted_at.isoformat(timespec="seconds"),
            "interval_days": interval_days,
            "inspection_interval_days": cadence_days,
            "round_number": interval_days // cadence_days,
            "target_due_at": target_due_at.isoformat(timespec="seconds"),
            "scheduled_for": scheduled_for.isoformat(timespec="seconds"),
            "due_at": scheduled_for.isoformat(timespec="seconds"),
            "cadence": {
                "interval_days": cadence_days,
                "inspection_weekdays": inspection_weekdays,
            },
            "state": state,
            "schedule_status": state,
            "days_overdue": days_overdue,
            "is_actionable": state in {"due", "overdue"},
            "observation": observation,
        }
        if observation is not None:
            row.update({
                "id": int(observation["id"]),
                "observation_id": int(observation["id"]),
                "completed_at": observation.get("inspected_at"),
                "inspected_at": observation.get("inspected_at"),
                "observation_status": observation.get("status"),
                "status": observation.get("status"),
                "condition": observation.get("condition"),
                "height_cm": observation.get("height_cm"),
                "photo_path": observation.get("photo_path"),
                "notes": observation.get("notes"),
                "actions_taken": observation.get("actions_taken"),
                "death_reason_category": observation.get("death_reason_category"),
                "inspector_user_id": observation.get("inspector_user_id"),
                "inspector_name": observation.get("inspector_name"),
                "inspector": {
                    "id": observation.get("inspector_user_id"),
                    "name": observation.get("inspector_name"),
                },
            })
        rounds.append(row)
    return rounds


def get_monitoring_project_detail(
    project_site_id: int,
    as_of: Optional[Any] = None,
    upcoming_days: int = 30,
) -> Optional[dict]:
    """Return planting-cycle-first monitoring detail for one stable project site."""
    try:
        clean_site_id = int(project_site_id)
    except (TypeError, ValueError) as error:
        raise ValueError("project_site_id must be a whole number.") from error
    as_of_dt = _parse_dashboard_datetime(as_of) or _manila_now()
    try:
        clean_upcoming_days = max(0, min(int(upcoming_days), 365))
    except (TypeError, ValueError) as error:
        raise ValueError("upcoming_days must be a whole number.") from error

    conn = _get_connection()
    try:
        site = conn.execute("""
            SELECT sz.id, sz.name, sz.notes, sz.polygon_geojson,
                   sz.organization_id, o.name AS organization_name,
                   sz.inspection_interval_days, sz.created_at, sz.updated_at
            FROM site_zones sz
            LEFT JOIN organizations o ON o.id = sz.organization_id
            WHERE sz.id = ?
        """, (clean_site_id,)).fetchone()
        if not site:
            return None
        event_rows = [dict(row) for row in conn.execute("""
            SELECT pe.*, pp.status AS current_point_status, pp.deleted_at,
                   pa.title AS assignment_title,
                   COALESCE(pl.full_name, CASE WHEN pe.planted_by_user_id IS NOT NULL THEN 'LGU' END) AS planter_name,
                   a.image_name,
                   EXISTS(
                       SELECT 1 FROM point_death_records pdr
                       WHERE pdr.planting_event_id = pe.id
                   ) AS has_death_record,
                   CASE WHEN pe.id = (
                       SELECT newer.id FROM planting_events newer
                       WHERE newer.planting_point_id = pe.planting_point_id
                       ORDER BY newer.planted_at DESC, newer.id DESC LIMIT 1
                   ) THEN 1 ELSE 0 END AS is_current_cycle
            FROM planting_events pe
            LEFT JOIN planting_points pp ON pp.id = pe.planting_point_id
            LEFT JOIN analyses a ON a.id = pp.analysis_id
            LEFT JOIN planter_assignments pa ON pa.id = pe.assignment_id
            LEFT JOIN planters pl ON pl.id = pe.planter_id
            WHERE pe.site_zone_id = ?
              AND COALESCE(pe.closure_reason, '') != 'completion_reversed'
            ORDER BY pe.planted_at DESC, pe.id DESC
        """, (clean_site_id,)).fetchall()]
    finally:
        conn.close()

    events = [
        event for event in event_rows
        if (_parse_dashboard_datetime(event.get("planted_at")) or as_of_dt) <= as_of_dt
    ]
    observations = list_monitoring_observations(
        project_site_id=clean_site_id,
        date_to=as_of_dt,
        limit=None,
    )
    history_by_event: dict[int, list[dict]] = {}
    for observation in observations:
        history_by_event.setdefault(int(observation["planting_event_id"]), []).append(observation)
    for history in history_by_event.values():
        history.sort(key=lambda row: (
            int(row["interval_days"]), str(row["inspected_at"]), int(row["id"]),
        ))
    latest_by_event = {
        event_id: history[-1] for event_id, history in history_by_event.items() if history
    }
    due_payload = get_due_monitoring_inspections(
        as_of=as_of_dt,
        project_site_id=clean_site_id,
        upcoming_days=clean_upcoming_days,
    )
    due_by_event: dict[int, list[dict]] = {}
    for row in due_payload["observations_due"]:
        due_by_event.setdefault(int(row["planting_event_id"]), []).append(row)
    schedules = list_planting_schedules(project_site_id=clean_site_id)
    summary = _monitoring_project_summary(
        clean_site_id, site["name"], site["notes"], events,
        latest_by_event, due_payload["observations_due"], schedules, as_of_dt,
    )
    settings = get_dashboard_settings(as_of_dt.year)
    intervals = settings["inspection_intervals_days"]
    inspection_weekdays = settings["inspection_weekdays"]

    points = []
    all_scheduled_rounds = []
    for event in events:
        event_id = int(event["id"])
        history = history_by_event.get(event_id, [])
        scheduled_rounds = _scheduled_monitoring_rounds(
            event, history, inspection_weekdays, as_of_dt,
            clean_site_id, site["name"], clean_upcoming_days,
        )
        all_scheduled_rounds.extend(scheduled_rounds)
        growth_history = [
            {
                "observation_id": row["id"],
                "interval_days": row["interval_days"],
                "inspected_at": row["inspected_at"],
                "height_cm": row["height_cm"],
                "condition": row["condition"],
                "status": row["status"],
            }
            for row in history if row.get("height_cm") is not None
        ]
        points.append({
            "planting_event_id": event_id,
            "planting_point_id": (
                int(event["planting_point_id"])
                if event.get("planting_point_id") is not None else None
            ),
            "point_num": int(event["point_num"]) if event.get("point_num") is not None else None,
            "latitude": event.get("latitude"),
            "longitude": event.get("longitude"),
            "planted_at": _iso_local(event.get("planted_at")),
            "inspection_interval_days": _event_inspection_cadence(event),
            "source": event.get("source"),
            "closed_at": _iso_local(event.get("closed_at")),
            "closure_reason": event.get("closure_reason"),
            "current_point_status": event.get("current_point_status"),
            "point_deleted_at": event.get("deleted_at"),
            "image_name": event.get("image_name"),
            "project_site_id": clean_site_id,
            "assignment_id": (
                int(event["assignment_id"]) if event.get("assignment_id") is not None else None
            ),
            "assignment_title": event.get("assignment_title"),
            "planter": {
                "id": int(event["planter_id"]) if event.get("planter_id") is not None else None,
                "name": event.get("planter_name"),
            },
            "planter_id": int(event["planter_id"]) if event.get("planter_id") is not None else None,
            "planter_name": event.get("planter_name"),
            "species": event.get("species"),
            "latest_observation": latest_by_event.get(event_id),
            "observations": history,
            "inspection_history": history,
            "observation_history": history,
            "history": history,
            "growth_history": growth_history,
            "scheduled_rounds": scheduled_rounds,
            "inspection_rounds": scheduled_rounds,
            "due_rounds": due_by_event.get(event_id, []),
        })

    try:
        geometry = json.loads(site["polygon_geojson"])
    except (TypeError, json.JSONDecodeError):
        geometry = None
    centroid_lat, centroid_lon = _site_geometry_centroid(site["polygon_geojson"])
    return {
        "as_of": as_of_dt.isoformat(timespec="seconds"),
        "timezone": "Asia/Manila",
        "upcoming_days": clean_upcoming_days,
        "project_site": {
            "id": clean_site_id,
            "project_site_id": clean_site_id,
            "name": site["name"],
            "notes": site["notes"],
            "organization_id": (
                int(site["organization_id"]) if site["organization_id"] is not None else None
            ),
            "organization": site["organization_name"],
            "organization_name": site["organization_name"],
            "inspection_interval_days": (
                int(site["inspection_interval_days"])
                if site["inspection_interval_days"] is not None else None
            ),
            "geometry": geometry,
            "centroid_lat": centroid_lat,
            "centroid_lon": centroid_lon,
            "created_at": site["created_at"],
            "updated_at": site["updated_at"],
        },
        "summary": summary,
        "points": points,
        "observations": observations,
        "inspection_history": observations,
        "scheduled_rounds": all_scheduled_rounds,
        "inspection_rounds": all_scheduled_rounds,
        "due_rounds": due_payload["observations_due"],
        "inspection_intervals_days": intervals,
        "inspection_weekdays": inspection_weekdays,
        "schedules": schedules,
    }


def _dashboard_filter_options(conn: Any) -> dict:
    # Independent dropdown lists share one round trip. JSON aggregation keeps
    # their own ordering and does not create a cartesian join of the lists.
    row = conn.execute("""
        SELECT
            (SELECT jsonb_agg(jsonb_build_object('id', id, 'name', name) ORDER BY name, id)
             FROM site_zones) AS sites,
            (SELECT jsonb_agg(jsonb_build_object('id', id, 'title', title, 'site_id', site_zone_id)
                             ORDER BY assignment_date DESC, id DESC)
             FROM planter_assignments) AS assignments,
            (SELECT jsonb_agg(species ORDER BY species) FROM (
                SELECT NULLIF(TRIM(species), '') AS species FROM analyses
                UNION
                SELECT NULLIF(TRIM(species), '') AS species FROM planter_assignments
                UNION
                SELECT NULLIF(TRIM(species), '') AS species FROM planting_events
            ) species_options WHERE species IS NOT NULL) AS species,
            (SELECT jsonb_agg(jsonb_build_object('id', id, 'name', full_name) ORDER BY full_name, id)
             FROM planters) AS planters
    """).fetchone()
    return {key: _monitoring_json(row[key]) or [] for key in ('sites', 'assignments', 'species', 'planters')}


def _dashboard_envelope(
    conn: Any,
    start: datetime,
    end: datetime,
    as_of_dt: datetime,
    bucket: str,
    site_id: Optional[int],
    assignment_id: Optional[int],
    species: Optional[str],
    planter_id: Optional[int],
) -> dict:
    return {
        "as_of": as_of_dt.isoformat(timespec="seconds"),
        "period": {
            "from": start.isoformat(timespec="seconds"),
            "to": end.isoformat(timespec="seconds"),
            "bucket": bucket,
            "timezone": "Asia/Manila",
        },
        "applied_filters": {
            "site_id": int(site_id) if site_id is not None else None,
            "assignment_id": int(assignment_id) if assignment_id is not None else None,
            "species": species or None,
            "planter_id": int(planter_id) if planter_id is not None else None,
        },
        "filter_options": _dashboard_filter_options(conn),
    }


def _matches_dashboard_dimensions(
    row: Any,
    site_id: Optional[int],
    assignment_id: Optional[int],
    species: Optional[str],
    planter_id: Optional[int],
) -> bool:
    def _value(key: str):
        try:
            return row[key]
        except (KeyError, IndexError):
            return None

    if site_id is not None and _value("site_id") != int(site_id):
        return False
    if assignment_id is not None and _value("assignment_id") != int(assignment_id):
        return False
    if planter_id is not None and _value("planter_id") != int(planter_id):
        return False
    if species:
        if str(_value("species") or "").strip().casefold() != species.strip().casefold():
            return False
    return True


def _load_dashboard_events(conn: Any) -> List[dict]:
    return [dict(row) for row in conn.execute("""
        SELECT
            pe.id,
            pe.planting_point_id,
            pe.assignment_point_id,
            pe.assignment_id,
            pe.site_zone_id AS site_id,
            sz.name AS site_name,
            pe.planter_id,
            COALESCE(pl.full_name, CASE WHEN pe.planted_by_user_id IS NOT NULL THEN 'LGU' END) AS planter_name,
            pe.species,
            pe.planted_at,
            pe.inspection_interval_days,
            pe.closed_at,
            pe.closure_reason,
            pe.point_num,
            pe.latitude,
            pe.longitude,
            pa.title AS assignment_title,
            pp.analysis_id,
            a.image_name
        FROM planting_events pe
        LEFT JOIN planter_assignments pa ON pa.id = pe.assignment_id
        LEFT JOIN site_zones sz ON sz.id = pe.site_zone_id
        LEFT JOIN planters pl ON pl.id = pe.planter_id
        LEFT JOIN planting_points pp ON pp.id = pe.planting_point_id
        LEFT JOIN analyses a ON a.id = pp.analysis_id
        WHERE COALESCE(pe.closure_reason, '') != 'completion_reversed'
        ORDER BY pe.planted_at, pe.id
    """).fetchall()]


def _load_dashboard_observations(conn: Any) -> List[dict]:
    return [dict(row) for row in conn.execute("""
        SELECT mo.*, pe.site_zone_id AS site_id, pe.assignment_id,
               pe.planter_id, pe.species, pe.planted_at
        FROM monitoring_observations mo
        JOIN planting_events pe ON pe.id = mo.planting_event_id
        ORDER BY mo.inspected_at, mo.id
    """).fetchall()]


def _filter_dashboard_events(
    events: List[dict],
    start: datetime,
    end: datetime,
    site_id: Optional[int],
    assignment_id: Optional[int],
    species: Optional[str],
    planter_id: Optional[int],
) -> List[dict]:
    result = []
    for event in events:
        planted = _parse_dashboard_datetime(event.get("planted_at"))
        if not planted or planted < start or planted > end:
            continue
        if _matches_dashboard_dimensions(
            event, site_id, assignment_id, species, planter_id
        ):
            result.append(event)
    return result


def _monitoring_rollup(
    events: List[dict],
    observations: List[dict],
    intervals: List[int],
    as_of_dt: datetime,
) -> dict:
    event_ids = {int(event["id"]) for event in events}
    filtered_obs = [
        observation for observation in observations
        if int(observation["planting_event_id"]) in event_ids
        and (
            (_parse_dashboard_datetime(observation.get("inspected_at")) or as_of_dt)
            <= as_of_dt
        )
    ]
    obs_by_slot = {
        (int(row["planting_event_id"]), int(row["interval_days"])): row
        for row in filtered_obs
    }
    latest_by_event: dict[int, dict] = {}
    death_interval: dict[int, int] = {}
    death_observation: dict[int, dict] = {}
    for row in filtered_obs:
        event_id = int(row["planting_event_id"])
        existing = latest_by_event.get(event_id)
        if existing is None or (
            int(row["interval_days"]), str(row["inspected_at"]), int(row["id"])
        ) > (
            int(existing["interval_days"]), str(existing["inspected_at"]), int(existing["id"])
        ):
            latest_by_event[event_id] = row
        if row["status"] == "dead":
            row_interval = int(row["interval_days"])
            if event_id not in death_interval or row_interval < death_interval[event_id]:
                death_interval[event_id] = row_interval
                death_observation[event_id] = row

    due_slots: list[tuple[dict, int, datetime, Optional[dict]]] = []
    for event in events:
        planted = _parse_dashboard_datetime(event.get("planted_at"))
        if not planted:
            continue
        event_id = int(event["id"])
        closed_at = _parse_dashboard_datetime(event.get("closed_at"))
        for interval in intervals:
            due_at = _round_timing(planted, interval, [1])["due_at"]
            if due_at <= as_of_dt:
                if closed_at and event_id not in death_interval and due_at > closed_at:
                    continue
                observation = obs_by_slot.get((event_id, interval))
                if (
                    observation is None
                    and event_id in death_interval
                    and interval >= death_interval[event_id]
                ):
                    observation = dict(death_observation[event_id])
                    observation["interval_days"] = interval
                    observation["carried_forward"] = True
                due_slots.append(
                    (event, interval, due_at, observation)
                )

    cohorts = []
    for interval in intervals:
        slots = [slot for slot in due_slots if slot[1] == interval]
        statuses = [slot[3]["status"] for slot in slots if slot[3] is not None]
        alive = statuses.count("alive")
        dead = statuses.count("dead")
        missing = statuses.count("missing")
        carried_dead = sum(
            1 for slot in slots
            if slot[3] is not None and slot[3].get("carried_forward")
        )
        scheduled_slots = [
            slot for slot in slots
            if not (slot[3] is not None and slot[3].get("carried_forward"))
        ]
        inspected = sum(1 for slot in scheduled_slots if slot[3] is not None)
        verified = alive + dead
        due = len(scheduled_slots)
        cohorts.append({
            "interval_days": interval,
            "due": due,
            "inspected": inspected,
            "alive": alive,
            "dead": dead,
            "missing": missing,
            "carried_dead": carried_dead,
            "coverage_pct": round(inspected / due * 100.0, 2) if due else None,
            "survival_rate_pct": round(alive / verified * 100.0, 2) if verified else None,
        })

    scheduled_slots = [
        slot for slot in due_slots
        if not (slot[3] is not None and slot[3].get("carried_forward"))
    ]
    due_total = len(scheduled_slots)
    inspected_due = sum(1 for slot in scheduled_slots if slot[3] is not None)
    overdue = sum(
        1 for _, _, due_at, observation in due_slots
        if observation is None and due_at.date() < as_of_dt.date()
    )
    latest = list(latest_by_event.values())
    latest_alive = sum(1 for row in latest if row["status"] == "alive")
    latest_dead = sum(1 for row in latest if row["status"] == "dead")
    latest_missing = sum(1 for row in latest if row["status"] == "missing")
    verified = latest_alive + latest_dead
    return {
        "observations": filtered_obs,
        "latest_by_event": latest_by_event,
        "due_slots": due_slots,
        "cohorts": cohorts,
        "due_total": due_total,
        "inspected_due": inspected_due,
        "overdue": overdue,
        "latest_alive": latest_alive,
        "latest_dead": latest_dead,
        "latest_missing": latest_missing,
        "verified_sample": verified,
        "verified_survival_rate_pct": (
            round(latest_alive / verified * 100.0, 2) if verified else None
        ),
        "coverage_pct": (
            round(inspected_due / due_total * 100.0, 2) if due_total else None
        ),
    }


def _operational_monitoring_coverage(
    events: List[dict],
    observations: List[dict],
    pending_rounds: List[dict],
    as_of_dt: datetime,
) -> dict:
    """Summarize recurring event-cadence rounds without scientific cohorts.

    Scientific survival cohorts continue to use the configured reporting ages
    in ``_monitoring_rollup``. Operational coverage instead counts every valid
    cadence-multiple round completed by ``as_of`` plus every still-actionable
    round whose snapped LGU field date has arrived.
    """
    events_by_id = {int(event["id"]): event for event in events}
    completed_slots: set[tuple[int, int]] = set()
    for observation in observations:
        event_id = int(observation["planting_event_id"])
        event = events_by_id.get(event_id)
        if event is None:
            continue
        inspected_at = _parse_dashboard_datetime(observation.get("inspected_at"))
        planted_at = _parse_dashboard_datetime(event.get("planted_at"))
        if not inspected_at or not planted_at or inspected_at > as_of_dt:
            continue
        interval_days = int(observation.get("interval_days") or 0)
        cadence_days = _event_inspection_cadence(event)
        if (
            interval_days <= 0
            or interval_days % cadence_days != 0
            or inspected_at < planted_at + timedelta(days=interval_days)
        ):
            # Preserve readable legacy history, but do not present an invalid
            # legacy slot as completed operational work.
            continue
        completed_slots.add((event_id, interval_days))

    due_pending = []
    for row in pending_rounds:
        event_id = int(row["planting_event_id"])
        interval_days = int(row["interval_days"])
        due_at = _parse_dashboard_datetime(row.get("due_at"))
        if (
            event_id in events_by_id
            and due_at is not None
            and due_at <= as_of_dt
            and (event_id, interval_days) not in completed_slots
        ):
            due_pending.append(row)

    inspected_due = len(completed_slots)
    due_total = inspected_due + len(due_pending)
    overdue = sum(1 for row in due_pending if int(row.get("days_overdue") or 0) > 0)
    return {
        "rate_pct": (
            round(inspected_due / due_total * 100.0, 2) if due_total else None
        ),
        "inspected_due": inspected_due,
        "due_total": due_total,
        "overdue": overdue,
        "pending_due": due_pending,
    }


def _bucket_start(value: datetime, bucket: str) -> date:
    local = value.astimezone(_MANILA_TZ)
    if bucket == "day":
        return local.date()
    if bucket == "week":
        return local.date() - timedelta(days=local.weekday())
    return date(local.year, local.month, 1)


def _next_bucket(value: date, bucket: str) -> date:
    if bucket == "day":
        return value + timedelta(days=1)
    if bucket == "week":
        return value + timedelta(days=7)
    if value.month == 12:
        return date(value.year + 1, 1, 1)
    return date(value.year, value.month + 1, 1)


def _blank_bucket_series(start: datetime, end: datetime, bucket: str) -> list[dict]:
    cursor = _bucket_start(start, bucket)
    final = _bucket_start(end, bucket)
    rows = []
    while cursor <= final and len(rows) < 4000:
        rows.append({
            "period_start": cursor.isoformat(),
            "mapped": 0,
            "assigned": 0,
            "planted": 0,
            "cumulative_planted": 0,
            "target_cumulative": None,
        })
        cursor = _next_bucket(cursor, bucket)
    return rows


def _load_current_dashboard_points(conn: Any) -> List[dict]:
    return [dict(row) for row in conn.execute("""
        WITH ranked_assignment AS (
            SELECT DISTINCT ON (pap.planting_point_id)
                pap.id AS assignment_point_id,
                pap.planting_point_id,
                pap.assignment_id,
                pap.status AS assignment_point_status,
                pap.assigned_at,
                pap.status_changed_at,
                pap.skip_reason,
                pa.status AS assignment_status,
                pa.site_zone_id,
                pa.planter_id,
                pa.species AS assignment_species
            FROM (SELECT * FROM planter_assignment_points WHERE released_at IS NULL) pap
            JOIN planter_assignments pa ON pa.id = pap.assignment_id
            WHERE pa.status IN ('active', 'completed')
               OR (pa.status = 'archived' AND pap.status = 'completed')
            ORDER BY pap.planting_point_id,
                CASE
                    WHEN pa.status = 'active' AND pap.status = 'pending' THEN 0
                    WHEN pap.status = 'completed' THEN 1
                    ELSE 2
                END,
                COALESCE(pap.completed_at, pap.status_changed_at, pap.assigned_at, pa.created_at) DESC,
                pap.id DESC
        )
        SELECT
            pp.id,
            pp.analysis_id,
            pp.point_num,
            pp.latitude,
            pp.longitude,
            pp.status AS point_status,
            pp.planted_at,
            pp.death_at,
            pp.deleted_at,
            a.analyzed_at,
            a.image_name,
            COALESCE(ra.site_zone_id, a.site_zone_id) AS site_id,
            ra.assignment_id,
            ra.planter_id,
            COALESCE(NULLIF(TRIM(ra.assignment_species), ''), NULLIF(TRIM(a.species), '')) AS species,
            ra.assignment_point_id,
            ra.assignment_point_status,
            ra.assignment_status,
            ra.assigned_at,
            ra.status_changed_at,
            ra.skip_reason
        FROM planting_points pp
        JOIN analyses a ON a.id = pp.analysis_id
        LEFT JOIN ranked_assignment ra
               ON ra.planting_point_id = pp.id
    """).fetchall()]


def _warning_counts_by_site(points: List[dict], conn: Any) -> tuple[dict, set[int]]:
    counts: dict[Optional[int], int] = {}
    exposed_ids: set[int] = set()
    if Point is None:
        return counts, exposed_ids
    zones = []
    for row in conn.execute("SELECT polygon_geojson FROM warning_zones").fetchall():
        geometry = _parse_site_zone_geometry(row["polygon_geojson"])
        if geometry is not None:
            zones.append(geometry)
    if not zones:
        return counts, exposed_ids
    for point in points:
        if (
            point.get("deleted_at")
            or point.get("point_status") not in {"planned", "planted"}
            or point.get("latitude") is None
            or point.get("longitude") is None
        ):
            continue
        location = Point(float(point["longitude"]), float(point["latitude"]))
        if any(zone.covers(location) for zone in zones):
            point_id = int(point["id"])
            exposed_ids.add(point_id)
            key = int(point["site_id"]) if point.get("site_id") is not None else None
            counts[key] = counts.get(key, 0) + 1
    return counts, exposed_ids


def get_dashboard_overview(
    date_from: Optional[Any] = None,
    date_to: Optional[Any] = None,
    site_id: Optional[int] = None,
    assignment_id: Optional[int] = None,
    species: Optional[str] = None,
    planter_id: Optional[int] = None,
    bucket: str = "week",
    as_of: Optional[Any] = None,
) -> dict:
    start, end, as_of_dt, clean_bucket = _dashboard_period(
        date_from, date_to, bucket, as_of
    )
    conn = _get_connection()
    try:
        envelope = _dashboard_envelope(
            conn, start, end, as_of_dt, clean_bucket,
            site_id, assignment_id, species, planter_id,
        )
        all_events = _load_dashboard_events(conn)
        period_events = _filter_dashboard_events(
            all_events, start, end, site_id, assignment_id, species, planter_id
        )
        observations = _load_dashboard_observations(conn)
        settings = get_dashboard_settings(end.year, _conn=conn)
        operational_settings = (settings if end.year == as_of_dt.year
                                else get_dashboard_settings(as_of_dt.year, _conn=conn))
        intervals = settings["inspection_intervals_days"]
        # Overview state is evaluated as of the requested/current instant even
        # when the event cohort is narrowed by a historical date range.
        rollup = _monitoring_rollup(period_events, observations, intervals, as_of_dt)
        operational_due = get_due_monitoring_inspections(
            as_of=as_of_dt,
            site_id=site_id,
            assignment_id=assignment_id,
            species=species,
            planter_id=planter_id,
            upcoming_days=0,
            _conn=conn, _settings=operational_settings,
        )["observations_due"]
        operational_coverage = _operational_monitoring_coverage(
            period_events, observations, operational_due, as_of_dt,
        )
        primary_interval = intervals[0] if intervals else None
        primary_cohort = next(
            (
                cohort for cohort in rollup["cohorts"]
                if cohort["interval_days"] == primary_interval
            ),
            {
                "interval_days": primary_interval,
                "due": 0,
                "inspected": 0,
                "alive": 0,
                "dead": 0,
                "missing": 0,
                "coverage_pct": None,
                "survival_rate_pct": None,
            },
        )
        global_operational_due = (
            get_due_monitoring_inspections(as_of=as_of_dt, upcoming_days=0,
                                           _conn=conn, _settings=operational_settings)["observations_due"]
            if any(value is not None for value in (site_id, assignment_id, planter_id)) or species
            else operational_due
        )
        global_operational_coverage = _operational_monitoring_coverage(
            all_events, observations, global_operational_due, as_of_dt,
        )
        points = [
            row for row in _load_current_dashboard_points(conn)
            if _matches_dashboard_dimensions(
                row, site_id, assignment_id, species, planter_id
            )
        ]
        all_events_by_point: dict[int, list[dict]] = {}
        for event in all_events:
            if event.get("planting_point_id") is not None:
                all_events_by_point.setdefault(int(event["planting_point_id"]), []).append(event)
        all_obs_by_event: dict[int, list[dict]] = {}
        for observation in observations:
            all_obs_by_event.setdefault(int(observation["planting_event_id"]), []).append(observation)

        lifecycle_counts = {
            "available": 0,
            "assigned": 0,
            "planted_unverified": 0,
            "verified_alive": 0,
            "dead": 0,
            "skipped": 0,
            "unavailable": 0,
        }
        for point in points:
            # Soft-deleted/filtered candidates are historical removals, not a
            # stage in the current restoration lifecycle.
            if point.get("deleted_at"):
                continue
            if point.get("point_status") == "skipped" or point.get("assignment_point_status") == "skipped":
                lifecycle_counts["skipped"] += 1
                continue
            if point.get("point_status") == "planted":
                if point.get("death_at"):
                    lifecycle_counts["dead"] += 1
                    continue
                events_for_point = all_events_by_point.get(int(point["id"]), [])
                latest_event = max(
                    events_for_point,
                    key=lambda event: (str(event["planted_at"]), int(event["id"])),
                    default=None,
                )
                latest_observation = None
                if latest_event:
                    event_observations = [
                        observation
                        for observation in all_obs_by_event.get(int(latest_event["id"]), [])
                        if (
                            _parse_dashboard_datetime(observation.get("inspected_at"))
                            or as_of_dt
                        ) <= as_of_dt
                    ]
                    latest_observation = max(
                        event_observations,
                        key=lambda observation: (
                            int(observation["interval_days"]),
                            str(observation["inspected_at"]),
                            int(observation["id"]),
                        ),
                        default=None,
                    )
                if latest_observation and latest_observation["status"] == "alive":
                    lifecycle_counts["verified_alive"] += 1
                elif latest_observation and latest_observation["status"] == "dead":
                    lifecycle_counts["dead"] += 1
                else:
                    lifecycle_counts["planted_unverified"] += 1
                continue
            if (
                point.get("assignment_status") == "active"
                and point.get("assignment_point_status") == "pending"
            ):
                lifecycle_counts["assigned"] += 1
            else:
                lifecycle_counts["available"] += 1

        progress = _blank_bucket_series(start, end, clean_bucket)
        progress_by_date = {row["period_start"]: row for row in progress}
        for point in points:
            analyzed = _parse_dashboard_datetime(point.get("analyzed_at"))
            if analyzed and start <= analyzed <= end:
                key = _bucket_start(analyzed, clean_bucket).isoformat()
                if key in progress_by_date:
                    progress_by_date[key]["mapped"] += 1
            assigned = _parse_dashboard_datetime(point.get("assigned_at"))
            if assigned and start <= assigned <= end:
                key = _bucket_start(assigned, clean_bucket).isoformat()
                if key in progress_by_date:
                    progress_by_date[key]["assigned"] += 1
        for event in period_events:
            planted = _parse_dashboard_datetime(event["planted_at"])
            if planted:
                key = _bucket_start(planted, clean_bucket).isoformat()
                if key in progress_by_date:
                    progress_by_date[key]["planted"] += 1
        cumulative = 0
        annual_target_applicable = (
            start.year == end.year and start.date() == date(end.year, 1, 1)
            and site_id is None
            and assignment_id is None
            and not species
            and planter_id is None
        )
        target = settings["annual_planting_target"] if annual_target_applicable else None
        days_in_year = date(end.year, 12, 31).timetuple().tm_yday
        for row in progress:
            cumulative += row["planted"]
            row["cumulative_planted"] = cumulative
            period_date = date.fromisoformat(row["period_start"])
            if target is not None and period_date.year == end.year:
                elapsed_days = min(days_in_year, max(0, period_date.timetuple().tm_yday))
                row["target_cumulative"] = round(target * elapsed_days / days_in_year, 2)

        quality = conn.execute("""
            SELECT
                (SELECT COUNT(*) FROM planter_assignments pa WHERE NOT EXISTS (
                    SELECT 1 FROM planter_assignment_points pap WHERE pap.assignment_id = pa.id
                )) AS stale_zero,
                (SELECT COUNT(*) FROM analyses WHERE site_zone_id IS NULL) +
                (SELECT COUNT(*) FROM planter_assignments WHERE site_zone_id IS NULL) AS missing_site_links,
                (SELECT COUNT(*) FROM analyses WHERE NULLIF(TRIM(species), '') IS NULL) +
                (SELECT COUNT(*) FROM planter_assignments WHERE NULLIF(TRIM(species), '') IS NULL) AS missing_species_links,
                (SELECT COUNT(*) FROM planting_events WHERE site_zone_id IS NULL) AS missing_event_site_links,
                (SELECT COUNT(*) FROM planting_events WHERE NULLIF(TRIM(species), '') IS NULL) AS missing_event_species_links,
                (SELECT COUNT(*) FROM planting_points WHERE deleted_at IS NOT NULL) AS historical_removed_points,
                (SELECT COUNT(*) FROM analyses WHERE analyzed_at < ?) AS stale_analyses,
                (SELECT COUNT(*) FROM planter_assignment_points
                 WHERE status = 'skipped' AND NULLIF(TRIM(skip_reason), '') IS NULL) AS legacy_skips_missing_reason
        """, ((as_of_dt - timedelta(days=180)).isoformat(),)).fetchone()
        stale_zero = int(quality['stale_zero'])
        missing_site_links = int(quality['missing_site_links'])
        missing_species_links = int(quality['missing_species_links'])
        missing_event_site_links = int(quality['missing_event_site_links'])
        missing_event_species_links = int(quality['missing_event_species_links'])
        historical_removed_points = int(quality['historical_removed_points'])
        stale_analyses = int(quality['stale_analyses'])
        legacy_skips_missing_reason = int(quality['legacy_skips_missing_reason'])

        warning_counts, _ = _warning_counts_by_site(points, conn)
        represented_site_ids = {
            int(row["site_id"])
            for row in (*points, *period_events)
            if row.get("site_id") is not None
        }
        sites = [
            feature for feature in list_project_sites()
            if site_id is None or int(feature["id"]) == int(site_id)
        ]
        if assignment_id is not None or species or planter_id is not None:
            sites = [
                feature for feature in sites
                if int(feature["id"]) in represented_site_ids
            ]
        primary_outcomes_by_site: dict[int, list[str]] = {}
        for event, interval, _, observation in rollup["due_slots"]:
            if (
                interval == primary_interval
                and observation is not None
                and event.get("site_id") is not None
            ):
                primary_outcomes_by_site.setdefault(int(event["site_id"]), []).append(
                    observation["status"]
                )
        overdue_by_site: dict[int, int] = {}
        for pending_round in operational_coverage["pending_due"]:
            pending_site_id = pending_round.get("project_site_id")
            if int(pending_round.get("days_overdue") or 0) > 0 and pending_site_id is not None:
                key = int(pending_site_id)
                overdue_by_site[key] = overdue_by_site.get(key, 0) + 1
        site_attention = []
        min_survival = settings["min_survival_target_pct"]
        for feature in sites:
            sid = int(feature["id"])
            statuses = primary_outcomes_by_site.get(sid, [])
            alive = statuses.count("alive")
            dead = statuses.count("dead")
            inspected = alive + dead
            rate = round(alive / inspected * 100.0, 2) if inspected else None
            reasons = []
            if min_survival is not None and rate is not None and rate < min_survival:
                reasons.append("survival_below_target")
            if overdue_by_site.get(sid, 0):
                reasons.append("overdue_inspections")
            if warning_counts.get(sid, 0):
                reasons.append("warning_exposure")
            site_attention.append({
                "site_id": sid,
                "site_name": feature["properties"]["name"],
                "survival_rate_pct": rate,
                "interval_days": primary_interval,
                "alive": alive,
                "dead": dead,
                "inspected": inspected,
                "overdue_inspections": overdue_by_site.get(sid, 0),
                "warning_points": warning_counts.get(sid, 0),
                "reasons": reasons,
            })
        site_attention.sort(key=lambda row: (-len(row["reasons"]), row["site_name"].casefold()))

        backlog_points = [
            point for point in points
            if point.get("assignment_status") == "active"
            and point.get("assignment_point_status") == "pending"
            and not point.get("deleted_at")
        ]
        overdue_backlog = sum(
            1 for point in backlog_points
            if (
                _parse_dashboard_datetime(point.get("assigned_at"))
                and as_of_dt - _parse_dashboard_datetime(point.get("assigned_at")) > timedelta(days=14)
            )
        )
        planted_count = len(period_events)
        verified_alive = int(primary_cohort["alive"])
        verified_dead = int(primary_cohort["dead"])
        verified_sample = verified_alive + verified_dead
        requiring_attention = sum(1 for row in site_attention if row["reasons"])

        envelope.update({
            "kpis": {
                "seedlings_planted": {
                    "value": planted_count,
                    "target": target,
                    "scope": (
                        "year_to_date" if annual_target_applicable
                        else "filtered" if any((site_id, assignment_id, species, planter_id))
                        else "selected_period"
                    ),
                    "target_year": end.year if annual_target_applicable else None,
                    "progress_pct": (
                        round(planted_count / target * 100.0, 2) if target else None
                    ),
                },
                "available_locations": {"value": lifecycle_counts["available"]},
                "assigned_backlog": {"value": len(backlog_points), "overdue": overdue_backlog},
                "verified_survival": {
                    "rate_pct": primary_cohort["survival_rate_pct"],
                    "interval_days": primary_interval,
                    "alive": verified_alive,
                    "dead": verified_dead,
                    "inspected": verified_sample,
                },
                "inspection_coverage": {
                    "rate_pct": operational_coverage["rate_pct"],
                    "inspected_due": operational_coverage["inspected_due"],
                    "due_total": operational_coverage["due_total"],
                    "overdue": operational_coverage["overdue"],
                    "scope": "recurring_operational_rounds",
                },
                "sites_requiring_attention": {
                    "value": requiring_attention,
                    "total": len(site_attention),
                },
            },
            "lifecycle": [
                {"key": key, "label": label, "value": lifecycle_counts[key]}
                for key, label in (
                    ("available", "Available"),
                    ("assigned", "Assigned"),
                    ("planted_unverified", "Planted — unverified"),
                    ("verified_alive", "Verified alive"),
                    ("dead", "Dead"),
                    ("skipped", "Skipped"),
                    ("unavailable", "Unavailable"),
                )
            ],
            "planting_progress": progress,
            "site_attention": site_attention,
            "data_quality_summary": {
                "scope": "global",
                "stale_zero_point_assignments": stale_zero,
                "missing_site_links": missing_site_links,
                "missing_species_links": missing_species_links,
                "missing_event_site_links": missing_event_site_links,
                "missing_event_species_links": missing_event_species_links,
                "inspection_gaps": global_operational_coverage["overdue"],
                "stale_analyses": stale_analyses,
                "legacy_skips_missing_reason": legacy_skips_missing_reason,
                "historical_removed_points": historical_removed_points,
            },
            # No historical snapshots exist yet, so an empty trend is more
            # truthful than reconstructing fictitious past data-quality states.
            "data_quality_trends": [],
        })
        return envelope
    finally:
        conn.close()


def get_dashboard_operations(
    date_from: Optional[Any] = None,
    date_to: Optional[Any] = None,
    site_id: Optional[int] = None,
    assignment_id: Optional[int] = None,
    species: Optional[str] = None,
    planter_id: Optional[int] = None,
    bucket: str = "week",
    as_of: Optional[Any] = None,
) -> dict:
    start, end, as_of_dt, clean_bucket = _dashboard_period(
        date_from, date_to, bucket, as_of
    )
    conn = _get_connection()
    try:
        envelope = _dashboard_envelope(
            conn, start, end, as_of_dt, clean_bucket,
            site_id, assignment_id, species, planter_id,
        )
        assignment_rows = [dict(row) for row in conn.execute("""
            SELECT
                pa.id AS assignment_id,
                pa.title,
                pa.site_zone_id AS site_id,
                sz.name AS site_name,
                sz.organization_id,
                o.name AS organization_name,
                pa.planter_id,
                pl.full_name AS planter_name,
                pa.species,
                pa.status,
                pa.assignment_date,
                pa.created_at,
                COUNT(pap.id) AS total,
                COALESCE(SUM(CASE WHEN pa.status = 'active' AND pap.status = 'pending' AND pp.deleted_at IS NULL THEN 1 ELSE 0 END), 0) AS pending,
                COALESCE(SUM(CASE WHEN pap.status = 'completed' THEN 1 ELSE 0 END), 0) AS completed,
                COALESCE(SUM(CASE WHEN pap.status = 'skipped' THEN 1 ELSE 0 END), 0) AS skipped,
                MIN(CASE WHEN pap.status = 'pending' AND pp.deleted_at IS NULL THEN pap.assigned_at END) AS oldest_pending_at
            FROM planter_assignments pa
            JOIN planters pl ON pl.id = pa.planter_id
            LEFT JOIN site_zones sz ON sz.id = pa.site_zone_id
            LEFT JOIN organizations o ON o.id = sz.organization_id
            LEFT JOIN planter_assignment_points pap ON pap.assignment_id = pa.id
            LEFT JOIN planting_points pp ON pp.id = pap.planting_point_id
            GROUP BY pa.id, pl.id, sz.id, sz.name, sz.organization_id, o.id, o.name
            ORDER BY CASE WHEN pa.status = 'active' THEN 0 ELSE 1 END,
                     pa.assignment_date DESC, pa.id DESC
        """).fetchall()]
        assignments = []
        for row in assignment_rows:
            if not _matches_dashboard_dimensions(
                row, site_id, assignment_id, species, planter_id
            ):
                continue
            assignments.append({
                "assignment_id": int(row["assignment_id"]),
                "title": row["title"],
                "site_id": int(row["site_id"]) if row["site_id"] is not None else None,
                "site_name": row["site_name"],
                "organization_id": (
                    int(row["organization_id"])
                    if row["organization_id"] is not None else None
                ),
                "organization_name": row["organization_name"] or row["site_name"],
                "planter_id": int(row["planter_id"]),
                "planter_name": row["planter_name"],
                "species": row["species"],
                "status": row["status"],
                "total": 0,
                "all_time_total": int(row["total"] or 0),
                "pending": 0,
                "completed": 0,
                "skipped": 0,
                "overdue": 0,
                "stale_zero_points": int(row["total"] or 0) == 0,
            })

        point_rows = [dict(row) for row in conn.execute("""
            SELECT
                pap.id,
                pap.assignment_id,
                pap.status,
                pap.assigned_at,
                pap.completed_at,
                pap.status_changed_at,
                pap.skip_reason,
                pa.status AS assignment_overall_status,
                pa.site_zone_id AS site_id,
                pa.planter_id,
                pa.species,
                pp.deleted_at
                , pp.latitude
                , pp.longitude
            FROM planter_assignment_points pap
            JOIN planter_assignments pa ON pa.id = pap.assignment_id
            JOIN planting_points pp ON pp.id = pap.planting_point_id
        """).fetchall()]
        point_rows = [
            row for row in point_rows
            if not row.get("deleted_at")
            and _matches_dashboard_dimensions(
                row, site_id, assignment_id, species, planter_id
            )
        ]

        assignment_output_by_id = {
            int(row["assignment_id"]): row for row in assignments
        }
        for row in point_rows:
            output = assignment_output_by_id.get(int(row["assignment_id"]))
            assigned_at = _parse_dashboard_datetime(row.get("assigned_at"))
            completed_at = _parse_dashboard_datetime(row.get("completed_at") or row.get("status_changed_at"))
            skipped_at = _parse_dashboard_datetime(row.get("status_changed_at"))
            current_pending = (
                row["status"] == "pending"
                and row["assignment_overall_status"] == "active"
            )
            completed_in_period = bool(
                row["status"] == "completed" and completed_at and start <= completed_at <= end
            )
            skipped_in_period = bool(
                row["status"] == "skipped" and skipped_at and start <= skipped_at <= end
            )
            if output and (current_pending or completed_in_period or skipped_in_period):
                output["total"] += 1
                if current_pending:
                    output["pending"] += 1
                elif completed_in_period:
                    output["completed"] += 1
                elif skipped_in_period:
                    output["skipped"] += 1
            if not current_pending:
                continue
            if assigned_at and as_of_dt - assigned_at > timedelta(days=14):
                if output:
                    output["overdue"] += 1

        assignment_source_by_id = {
            int(row["assignment_id"]): row for row in assignment_rows
        }
        assignments = [
            row for row in assignments
            if row["total"] > 0
            or (
                row["stale_zero_points"]
                and (
                    (_parse_dashboard_datetime(
                        assignment_source_by_id[row["assignment_id"]].get("created_at")
                        or assignment_source_by_id[row["assignment_id"]].get("assignment_date")
                    ) or start)
                    >= start
                )
                and (
                    (_parse_dashboard_datetime(
                        assignment_source_by_id[row["assignment_id"]].get("created_at")
                        or assignment_source_by_id[row["assignment_id"]].get("assignment_date")
                    ) or end)
                    <= end
                )
            )
        ]

        aging = {
            "0_7": 0,
            "8_14": 0,
            "15_30": 0,
            "over_30": 0,
        }
        for row in point_rows:
            if row["status"] != "pending" or row["assignment_overall_status"] != "active":
                continue
            assigned_at = _parse_dashboard_datetime(row.get("assigned_at")) or as_of_dt
            age = max(0, (as_of_dt.date() - assigned_at.date()).days)
            if age <= 7:
                aging["0_7"] += 1
            elif age <= 14:
                aging["8_14"] += 1
            elif age <= 30:
                aging["15_30"] += 1
            else:
                aging["over_30"] += 1

        # Work is planned and reviewed by partner organization/project site.
        # Keep individual planter names in assignment detail, but group this
        # dashboard comparison by the organization responsible for the site.
        workload_by_organization: dict[object, dict] = {}
        for assignment in assignments:
            organization_id = assignment.get("organization_id")
            organization_name = assignment.get("organization_name") or "No organization assigned"
            organization_key = organization_id if organization_id is not None else organization_name
            bucket_row = workload_by_organization.setdefault(organization_key, {
                "organization_id": organization_id,
                "organization_name": organization_name,
                "pending": 0,
                "completed": 0,
                "skipped": 0,
                "total": 0,
            })
            for status in ("pending", "completed", "skipped"):
                bucket_row[status] += int(assignment.get(status) or 0)
            bucket_row["total"] += int(assignment.get("total") or 0)
        workload = sorted(
            workload_by_organization.values(),
            key=lambda row: (
                -row["pending"],
                -row["total"],
                row["organization_name"].casefold(),
            ),
        )

        throughput_base = [
            {
                "period_start": row["period_start"],
                "assigned": 0,
                "completed": 0,
                "skipped": 0,
            }
            for row in _blank_bucket_series(start, end, clean_bucket)
        ]
        throughput_map = {row["period_start"]: row for row in throughput_base}
        for row in point_rows:
            assigned_at = _parse_dashboard_datetime(row.get("assigned_at"))
            if assigned_at and start <= assigned_at <= end:
                key = _bucket_start(assigned_at, clean_bucket).isoformat()
                if key in throughput_map:
                    throughput_map[key]["assigned"] += 1
            if row["status"] == "completed":
                completed_at = _parse_dashboard_datetime(row.get("completed_at") or row.get("status_changed_at"))
                if completed_at and start <= completed_at <= end:
                    key = _bucket_start(completed_at, clean_bucket).isoformat()
                    if key in throughput_map:
                        throughput_map[key]["completed"] += 1
            elif row["status"] == "skipped":
                skipped_at = _parse_dashboard_datetime(row.get("status_changed_at"))
                if skipped_at and start <= skipped_at <= end:
                    key = _bucket_start(skipped_at, clean_bucket).isoformat()
                    if key in throughput_map:
                        throughput_map[key]["skipped"] += 1

        assigned_in_period = sum(
            1 for row in point_rows
            if (
                _parse_dashboard_datetime(row.get("assigned_at"))
                and start <= _parse_dashboard_datetime(row.get("assigned_at")) <= end
            )
        )
        total_pending = sum(row["pending"] for row in assignments)
        total_completed = sum(row["completed"] for row in assignments)
        total_skipped = sum(row["skipped"] for row in assignments)
        total_overdue = sum(row["overdue"] for row in assignments)
        envelope.update({
            "summary": {
                "assigned": assigned_in_period,
                "pending": total_pending,
                "completed": total_completed,
                "skipped": total_skipped,
                "overdue": total_overdue,
                "assigned_scope": "selected_period",
                "pending_scope": "current_state",
            },
            "workload": workload,
            "backlog_aging": [
                {"key": key, "label": label, "count": aging[key]}
                for key, label in (
                    ("0_7", "0–7 days"),
                    ("8_14", "8–14 days"),
                    ("15_30", "15–30 days"),
                    ("over_30", "Over 30 days"),
                )
            ],
            "throughput": throughput_base,
            "assignments": assignments,
        })
        return envelope
    finally:
        conn.close()


def _ecology_outcomes(
    events: List[dict],
    rollup: dict,
    group_key: str,
    name_key: Optional[str] = None,
) -> List[dict]:
    buckets: dict[tuple[Any, int], dict] = {}
    for event, interval, _, observation in rollup["due_slots"]:
        raw_key = event.get(group_key)
        dimension = raw_key if raw_key not in {None, ""} else "Unspecified"
        key = (dimension, int(interval))
        bucket = buckets.setdefault(key, {
            "alive": 0,
            "dead": 0,
            "missing": 0,
            "carried_dead": 0,
            "due": 0,
            "inspected_due": 0,
        })
        if observation is not None:
            bucket[observation["status"]] += 1
            if observation.get("carried_forward"):
                bucket["carried_dead"] += 1
        carried_forward = bool(observation and observation.get("carried_forward"))
        if not carried_forward:
            bucket["due"] += 1
        if observation is not None and not carried_forward:
            bucket["inspected_due"] += 1
    result = []
    for (dimension, interval), values in buckets.items():
        verified = values["alive"] + values["dead"]
        inspected = verified + values["missing"]
        row = {
            "interval_days": interval,
            "planting_age_days": interval,
            "age_band": f"{interval}-day inspection",
            "alive": values["alive"],
            "dead": values["dead"],
            "missing": values["missing"],
            "carried_dead": values["carried_dead"],
            "due": values["due"],
            "inspected_due": values["inspected_due"],
            "inspected": inspected,
            "sample_size": verified,
            "coverage_pct": (
                round(values["inspected_due"] / values["due"] * 100.0, 2)
                if values["due"] else None
            ),
            "survival_rate_pct": (
                round(values["alive"] / verified * 100.0, 2) if verified else None
            ),
            "small_sample": verified < 10,
        }
        if group_key == "species":
            row.update({"species": dimension, "species_name": dimension, "name": dimension})
        elif group_key == "site_id":
            numeric_key = int(dimension) if dimension != "Unspecified" else None
            site_name = next(
                (
                    event.get(name_key or "site_name")
                    for event in events
                    if event.get(group_key) == numeric_key and event.get(name_key or "site_name")
                ),
                "Unassigned site",
            )
            row.update({"site_id": numeric_key, "site_name": site_name, "name": site_name})
        result.append(row)
    result.sort(key=lambda row: (
        int(row["interval_days"]),
        -row["sample_size"],
        str(row.get("name") or "").casefold(),
    ))
    return result


def get_dashboard_ecology(
    date_from: Optional[Any] = None,
    date_to: Optional[Any] = None,
    site_id: Optional[int] = None,
    assignment_id: Optional[int] = None,
    species: Optional[str] = None,
    planter_id: Optional[int] = None,
    bucket: str = "week",
    as_of: Optional[Any] = None,
) -> dict:
    start, end, as_of_dt, clean_bucket = _dashboard_period(
        date_from, date_to, bucket, as_of
    )
    conn = _get_connection()
    try:
        envelope = _dashboard_envelope(
            conn, start, end, as_of_dt, clean_bucket,
            site_id, assignment_id, species, planter_id,
        )
        events = _filter_dashboard_events(
            _load_dashboard_events(conn), start, end,
            site_id, assignment_id, species, planter_id,
        )
        observations = _load_dashboard_observations(conn)
        intervals = get_dashboard_settings(end.year)["inspection_intervals_days"]
        rollup = _monitoring_rollup(events, observations, intervals, end)
        primary_interval = intervals[0] if intervals else None
        primary_cohort = next(
            (
                cohort for cohort in rollup["cohorts"]
                if cohort["interval_days"] == primary_interval
            ),
            {
                "interval_days": primary_interval,
                "due": 0,
                "inspected": 0,
                "alive": 0,
                "dead": 0,
                "missing": 0,
                "coverage_pct": None,
                "survival_rate_pct": None,
            },
        )
        event_by_id = {int(event["id"]): event for event in events}

        species_outcomes = _ecology_outcomes(events, rollup, "species")
        site_outcomes = _ecology_outcomes(events, rollup, "site_id", "site_name")

        from mangrovision_db.mortality import mortality_cause_counts
        cause_counts = mortality_cause_counts(conn, start, min(end, as_of_dt),
            site_id=site_id, assignment_id=assignment_id, species=species, planter_id=planter_id)
        total_deaths = sum(cause_counts.values())
        cumulative = 0
        mortality_causes = []
        for cause, deaths in sorted(cause_counts.items(), key=lambda item: (-item[1], item[0])):
            cumulative += deaths
            mortality_causes.append({
                "cause": cause,
                "label": DEATH_REASON_CATEGORIES.get(cause, cause.replace("_", " ").title()),
                "deaths": deaths,
                "percent_of_deaths": round(deaths / total_deaths * 100.0, 2) if total_deaths else None,
                "cumulative_pct": round(cumulative / total_deaths * 100.0, 2) if total_deaths else None,
            })

        growth_buckets = {
            row["period_start"]: {"values": [], "period_start": row["period_start"]}
            for row in _blank_bucket_series(start, end, clean_bucket)
        }
        for observation in rollup["observations"]:
            if observation["status"] != "alive" or observation.get("height_cm") is None:
                continue
            inspected_at = _parse_dashboard_datetime(observation["inspected_at"])
            if not inspected_at or inspected_at < start or inspected_at > end:
                continue
            key = _bucket_start(inspected_at, clean_bucket).isoformat()
            if key in growth_buckets:
                growth_buckets[key]["values"].append(float(observation["height_cm"]))
        growth = []
        for row in growth_buckets.values():
            values = row["values"]
            if values:
                growth.append({
                    "period_start": row["period_start"],
                    "average_height_cm": round(sum(values) / len(values), 2),
                    "sample_size": len(values),
                })

        # Planter outcomes are deliberately contextual: every row retains
        # site, species, planting age/inspection round, and coverage.
        contextual: dict[tuple, dict] = {}
        for event in events:
            for interval in intervals:
                key = (
                    event.get("planter_id"), event.get("site_id"),
                    event.get("species") or "Unspecified", interval,
                )
                row = contextual.setdefault(key, {
                    "planter_id": int(event["planter_id"]) if event.get("planter_id") is not None else None,
                    "planter_name": event.get("planter_name") or "Unattributed",
                    "site_id": int(event["site_id"]) if event.get("site_id") is not None else None,
                    "site_name": event.get("site_name") or "Unassigned site",
                    "project_site_name": event.get("site_name") or "Unassigned site",
                    "species": event.get("species") or "Unspecified",
                    "species_name": event.get("species") or "Unspecified",
                    "planting_age_days": interval,
                    "age_band": f"{interval}-day inspection",
                    "interval_days": interval,
                    "alive": 0,
                    "dead": 0,
                    "missing": 0,
                    "carried_dead": 0,
                    "due": 0,
                    "inspected_due": 0,
                })
        for event, interval, _, observation in rollup["due_slots"]:
            key = (
                event.get("planter_id"), event.get("site_id"),
                event.get("species") or "Unspecified", interval,
            )
            if key in contextual:
                if observation is not None:
                    contextual[key][observation["status"]] += 1
                    if observation.get("carried_forward"):
                        contextual[key]["carried_dead"] += 1
                carried_forward = bool(observation and observation.get("carried_forward"))
                if not carried_forward:
                    contextual[key]["due"] += 1
                if observation is not None and not carried_forward:
                    contextual[key]["inspected_due"] += 1
        planter_outcomes = []
        for row in contextual.values():
            inspected = row["alive"] + row["dead"] + row["missing"]
            sample = row["alive"] + row["dead"]
            if not row["due"] and not inspected:
                continue
            row["inspected"] = inspected
            row["sample_size"] = sample
            row["coverage_pct"] = (
                round(row["inspected_due"] / row["due"] * 100.0, 2)
                if row["due"] else None
            )
            row["survival_rate_pct"] = (
                round(row["alive"] / sample * 100.0, 2) if sample else None
            )
            row["small_sample"] = sample < 10
            row["contextual_only"] = True
            planter_outcomes.append(row)
        planter_outcomes.sort(key=lambda row: (
            row["planter_name"].casefold(), row["site_name"].casefold(),
            row["species"].casefold(), row["interval_days"],
        ))

        envelope.update({
            "summary": {
                "interval_days": primary_interval,
                "verified_survival_rate_pct": primary_cohort["survival_rate_pct"],
                "alive": primary_cohort["alive"],
                "dead": primary_cohort["dead"],
                "missing": primary_cohort["missing"],
                "inspected": primary_cohort["inspected"],
                "coverage_pct": primary_cohort["coverage_pct"],
                "inspected_due": primary_cohort["inspected"],
                "due_total": primary_cohort["due"],
            },
            "survival_cohorts": rollup["cohorts"],
            "species_outcomes": species_outcomes,
            "site_outcomes": site_outcomes,
            "planter_outcomes": planter_outcomes,
            "mortality_causes": mortality_causes,
            "mortality_includes_unlocated": site_id is None and assignment_id is None and not species and planter_id is None,
            "growth": growth,
            # This chart always combines the latest monitoring visit from every organization.
            "organization_growth": _organization_growth_dashboard(conn, min(end, as_of_dt)),
        })
        return envelope
    finally:
        conn.close()


def _unique_footprint_metrics(analyses: List[dict]) -> dict:
    if not analyses:
        return {
            "unique_footprint_area_m2": None,
            "unique_footprint_quality": "unavailable",
            "footprints_included": 0,
            "footprints_missing": 0,
        }
    if shape is None:
        return {
            "unique_footprint_area_m2": None,
            "unique_footprint_quality": "unavailable",
            "footprints_included": 0,
            "footprints_missing": len(analyses),
        }
    geometries = []
    quality_values = []
    missing = 0
    for analysis in analyses:
        try:
            payload = json.loads(analysis.get("footprint_geojson") or "null")
            geometry = shape(payload) if payload else None
            if geometry is None or geometry.is_empty or geometry.geom_type not in {"Polygon", "MultiPolygon"}:
                raise ValueError("invalid footprint")
            if not geometry.is_valid:
                geometry = geometry.buffer(0)
            if geometry.is_empty:
                raise ValueError("empty footprint")
            geometries.append(geometry)
            quality_values.append(str(analysis.get("footprint_quality") or "unavailable").lower())
        except Exception:
            missing += 1
    # A partial union would understate coverage and is therefore not reported.
    if not geometries or missing:
        return {
            "unique_footprint_area_m2": None,
            "unique_footprint_quality": "unavailable_partial" if geometries else "unavailable",
            "footprints_included": len(geometries),
            "footprints_missing": missing,
        }
    try:
        from shapely.ops import unary_union

        unioned = unary_union(geometries)
        try:
            from pyproj import Geod

            geod = Geod(ellps="WGS84")
            area_m2, _ = geod.geometry_area_perimeter(unioned)
            area_value = round(abs(float(area_m2)), 2)
        except (ImportError, ModuleNotFoundError):
            # Offline Windows deployments may lack pyproj.  Compute the area
            # of the already-unioned geometry in a local equirectangular
            # projection; unlike summing source rows this still removes all
            # overlaps and is sufficiently accurate at project-site scale.
            center_lat = float(unioned.centroid.y)
            meters_lat = 111_320.0
            meters_lon = 111_320.0 * max(0.01, math.cos(math.radians(center_lat)))

            def _ring_area(coords) -> float:
                projected = [(float(x) * meters_lon, float(y) * meters_lat) for x, y, *_ in coords]
                return abs(sum(
                    projected[index][0] * projected[(index + 1) % len(projected)][1]
                    - projected[(index + 1) % len(projected)][0] * projected[index][1]
                    for index in range(len(projected))
                )) / 2.0

            def _polygon_area(polygon) -> float:
                return max(
                    0.0,
                    _ring_area(polygon.exterior.coords)
                    - sum(_ring_area(interior.coords) for interior in polygon.interiors),
                )

            if unioned.geom_type == "Polygon":
                area_value = round(_polygon_area(unioned), 2)
            elif unioned.geom_type == "MultiPolygon":
                area_value = round(sum(_polygon_area(polygon) for polygon in unioned.geoms), 2)
            else:
                raise ValueError("Footprint union did not produce polygonal geometry.")
    except Exception:
        return {
            "unique_footprint_area_m2": None,
            "unique_footprint_quality": "unavailable",
            "footprints_included": len(geometries),
            "footprints_missing": 0,
        }
    estimated = any(
        quality not in {"authoritative", "surveyed", "exact"}
        for quality in quality_values
    )
    return {
        "unique_footprint_area_m2": area_value,
        "unique_footprint_quality": "estimated" if estimated else "authoritative",
        "footprints_included": len(geometries),
        "footprints_missing": 0,
    }


def get_dashboard_sites(
    date_from: Optional[Any] = None,
    date_to: Optional[Any] = None,
    site_id: Optional[int] = None,
    assignment_id: Optional[int] = None,
    species: Optional[str] = None,
    planter_id: Optional[int] = None,
    bucket: str = "week",
    as_of: Optional[Any] = None,
) -> dict:
    start, end, as_of_dt, clean_bucket = _dashboard_period(
        date_from, date_to, bucket, as_of
    )
    conn = _get_connection()
    try:
        envelope = _dashboard_envelope(
            conn, start, end, as_of_dt, clean_bucket,
            site_id, assignment_id, species, planter_id,
        )
        all_analyses = [dict(row) for row in conn.execute("""
            SELECT a.*, sz.name AS site_name
            FROM analyses a
            LEFT JOIN site_zones sz ON sz.id = a.site_zone_id
            ORDER BY a.analyzed_at, a.id
        """).fetchall()]
        dimension_points = [
            row for row in _load_current_dashboard_points(conn)
            if _matches_dashboard_dimensions(
                row, site_id, assignment_id, species, planter_id
            )
        ]
        dimension_analysis_ids = {int(row["analysis_id"]) for row in dimension_points}
        analyses = []
        for analysis in all_analyses:
            analyzed = _parse_dashboard_datetime(analysis.get("analyzed_at"))
            if not analyzed or analyzed < start or analyzed > end:
                continue
            if assignment_id is not None or planter_id is not None:
                if int(analysis["id"]) not in dimension_analysis_ids:
                    continue
            else:
                analysis_dimensions = {
                    "site_id": analysis.get("site_zone_id"),
                    "assignment_id": None,
                    "planter_id": None,
                    "species": analysis.get("species"),
                }
                if not _matches_dashboard_dimensions(
                    analysis_dimensions, site_id, None, species, None
                ):
                    continue
            analyses.append(analysis)

        suitability = []
        for analysis in analyses:
            suitability.append({
                "analysis_id": int(analysis["id"]),
                "analysis_number": analysis["analysis_number"],
                "image_name": analysis["image_name"],
                "site_id": int(analysis["site_zone_id"]) if analysis["site_zone_id"] is not None else None,
                "site_name": analysis["site_name"],
                "analyzed_at": analysis["analyzed_at"],
                "total_area_m2": float(analysis["total_area_m2"] or 0),
                "plantable_area_m2": float(analysis["plantable_area_m2"] or 0),
                "danger_area_m2": float(analysis["danger_area_m2"] or 0),
                "canopy_area_m2": float(analysis["canopy_area_m2"] or 0),
                "canopy_coverage_pct": (
                    float(analysis["canopy_coverage_pct"])
                    if analysis["canopy_coverage_pct"] is not None else None
                ),
                "footprint_quality": analysis["footprint_quality"] or "unavailable",
            })
        total_area = sum(row["total_area_m2"] for row in suitability)
        plantable_area = sum(row["plantable_area_m2"] for row in suitability)
        danger_area = sum(row["danger_area_m2"] for row in suitability)
        canopy_area = sum(row["canopy_area_m2"] for row in suitability)
        footprint = _unique_footprint_metrics(analyses)

        points = [
            row for row in dimension_points
            if not row.get("deleted_at")
            and row.get("point_status") in {"planned", "planted"}
            and row.get("latitude") is not None
            and row.get("longitude") is not None
        ]
        warning_buckets: dict[tuple[str, str], dict] = {}
        warning_exposed_ids: set[int] = set()
        warning_rows = conn.execute("""
            SELECT warning_type, severity, polygon_geojson
            FROM warning_zones
            ORDER BY warning_type, severity
        """).fetchall()
        if Point is not None:
            for warning in warning_rows:
                geometry = _parse_site_zone_geometry(warning["polygon_geojson"])
                if geometry is None:
                    continue
                warning_type = _clean_warning_type(warning["warning_type"])
                severity = _clean_warning_severity(warning["severity"])
                key = (warning_type, severity)
                output = warning_buckets.setdefault(key, {
                    "warning_type": warning_type,
                    "label": _warning_type_label(warning_type),
                    "severity": severity,
                    "count": 0,
                    "planned": 0,
                    "planted": 0,
                })
                for point in points:
                    if geometry.covers(Point(float(point["longitude"]), float(point["latitude"]))):
                        output["count"] += 1
                        output["planted" if point["point_status"] == "planted" else "planned"] += 1
                        warning_exposed_ids.add(int(point["id"]))
        warning_exposure = sorted(
            warning_buckets.values(),
            key=lambda row: (-_WARNING_SEVERITY_RANK.get(row["severity"], 0), -row["count"], row["label"]),
        )
        warning_points_by_site: dict[int, dict[str, int]] = {}
        for point in points:
            point_id = int(point["id"])
            point_site_id = point.get("site_id")
            if point_id not in warning_exposed_ids or point_site_id is None:
                continue
            site_warning = warning_points_by_site.setdefault(int(point_site_id), {
                "warning_point_count": 0,
                "warning_planned_points": 0,
                "warning_planted_points": 0,
            })
            site_warning["warning_point_count"] += 1
            status_key = (
                "warning_planted_points"
                if point.get("point_status") == "planted"
                else "warning_planned_points"
            )
            site_warning[status_key] += 1

        represented_site_ids = {
            int(row["site_id"])
            for row in dimension_points
            if row.get("site_id") is not None
        }
        represented_site_ids.update(
            int(row["site_zone_id"])
            for row in analyses
            if row.get("site_zone_id") is not None
        )
        project_sites = []
        for feature in list_project_sites():
            if site_id is not None and int(feature["id"]) != int(site_id):
                continue
            if (
                (assignment_id is not None or species or planter_id is not None)
                and int(feature["id"]) not in represented_site_ids
            ):
                continue
            properties = feature["properties"]
            site_geometry = _parse_site_zone_geometry(
                json.dumps(feature.get("geometry"))
            ) if feature.get("geometry") else None
            centroid = site_geometry.centroid if site_geometry is not None else None
            project_sites.append({
                "id": int(feature["id"]),
                "name": properties["name"],
                "notes": properties.get("notes"),
                "geometry": feature.get("geometry"),
                "centroid_lat": float(centroid.y) if centroid is not None else None,
                "centroid_lon": float(centroid.x) if centroid is not None else None,
                "assignment_count": int(properties.get("assignment_count") or 0),
                "analysis_count": int(properties.get("analysis_count") or 0),
                "point_count": int(properties.get("point_count") or 0),
                **warning_points_by_site.get(int(feature["id"]), {
                    "warning_point_count": 0,
                    "warning_planned_points": 0,
                    "warning_planted_points": 0,
                }),
            })
        allowed_assignments = set()
        for row in conn.execute("""
            SELECT id, site_zone_id AS site_id, planter_id, species,
                   COALESCE(created_at, assignment_date) AS occurred_at
            FROM planter_assignments
        """).fetchall():
            candidate = dict(row)
            candidate["assignment_id"] = int(row["id"])
            occurred_at = _parse_dashboard_datetime(row["occurred_at"])
            if (
                _matches_dashboard_dimensions(
                    candidate, site_id, assignment_id, species, planter_id
                )
                and (occurred_at is None or start <= occurred_at <= end)
            ):
                allowed_assignments.add(int(row["id"]))
        assignment_features = [
            feature for feature in list_site_zones()
            if int(feature["properties"].get("assignment_id") or feature["id"]) in allowed_assignments
        ]

        envelope.update({
            "summary": {
                "project_sites": len(project_sites),
                "project_sites_scope": "current_state",
                "analyses": len(analyses),
                "total_area_m2": round(total_area, 2),
                "unique_footprint_area_m2": footprint["unique_footprint_area_m2"],
                "unique_footprint_quality": footprint["unique_footprint_quality"],
                "footprint_quality": footprint["unique_footprint_quality"],
                "footprints_included": footprint["footprints_included"],
                "footprints_missing": footprint["footprints_missing"],
                "plantable_area_m2": round(plantable_area, 2),
                "danger_area_m2": round(danger_area, 2),
                "plantable_pct": round(plantable_area / total_area * 100.0, 2) if total_area else None,
                "danger_pct": round(danger_area / total_area * 100.0, 2) if total_area else None,
                "canopy_coverage_pct": round(canopy_area / total_area * 100.0, 2) if total_area else None,
                "warning_exposed_points": len(warning_exposed_ids),
                "warning_exposure_scope": "current_state",
            },
            "suitability": suitability,
            "warning_exposure": warning_exposure,
            "project_sites": project_sites,
            "assignment_zones": {
                "type": "FeatureCollection",
                "name": "assignment_zones",
                "features": assignment_features,
            },
            "environmental_context": {
                "availability": "unavailable",
                "contextual_only": True,
                "causal_interpretation": False,
                "message": "No combined environmental feed is configured; use per-site context endpoints when available.",
            },
            "tides": {
                "configured": False,
                "contextual_only": True,
                "causal_interpretation": False,
                "data": [],
            },
        })
        return envelope
    finally:
        conn.close()
