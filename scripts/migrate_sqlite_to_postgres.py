"""Resumable, audited SQLite/GeoJSON to PostgreSQL/PostGIS migration.

This is an offline cutover command, not application runtime code. It takes a
consistent SQLite snapshot, intentionally omits old sessions, uploads analysis
images to private S3-compatible storage, and preserves explicit durable IDs.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
import sys
import tempfile
import uuid
from collections.abc import Iterable
from contextlib import ExitStack, closing
from datetime import date, datetime, timezone
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mangrovision_db.config import get_settings
from mangrovision_db.storage import (
    delete_object,
    s3_client,
    upload_data_url,
)
from mangrovision_db.zones import normalize_polygon


TABLE_SPECS: tuple[tuple[str, str], ...] = (
    ("users", "users"),
    ("organizations", "organizations"),
    ("site_zones", "project_sites"),
    ("analyses", "analyses"),
    ("planting_points", "planting_points"),
    ("planters", "planters"),
    ("planter_assignments", "planter_assignments"),
    ("planter_assignment_points", "planter_assignment_points"),
    ("planting_events", "planting_events"),
    ("point_death_records", "point_death_records"),
    ("monitoring_observations", "monitoring_observations"),
    ("organization_monitoring_records", "organization_monitoring_records"),
    ("planting_schedules", "planting_schedules"),
    ("dashboard_settings", "dashboard_settings"),
)

TIMESTAMP_COLUMNS = {
    "analyzed_at", "assigned_at", "closed_at", "completed_at", "created_at",
    "death_at", "end_at", "inspected_at", "last_login", "last_seen_at",
    "monitored_at", "planted_at", "revoked_at", "start_at",
    "status_changed_at", "updated_at",
}
DATE_COLUMNS = {"assignment_date", "planted_date"}
JSON_COLUMNS = {
    "analysis_detail_json", "canopy_polygons_geojson", "footprint_geojson",
    "inspection_intervals_json", "inspection_weekdays_json", "polygon_geojson",
}
POLYGON_COLUMNS = {"footprint_geojson", "polygon_geojson"}
RENAMED_COLUMNS = {"site_zone_id": "project_site_id"}
SKIPPED_COLUMNS = {"original_image", "visualization_image"}
DEFERRED_COLUMNS = {("point_death_records", "monitoring_observation_id")}
IDENTITY_TABLES = {
    target for _, target in TABLE_SPECS if target != "dashboard_settings"
} | {"map_zones", "analysis_assets", "object_cleanup_jobs"}


class AssetImportError(RuntimeError):
    """Carry object keys that still need durable cleanup after rollback."""

    def __init__(self, message: str, cleanup_keys: list[str]):
        super().__init__(message)
        self.cleanup_keys = cleanup_keys


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot_sqlite(source: Path, destination: Path) -> None:
    source_uri = f"file:{source.resolve().as_posix()}?mode=ro"
    source_db = sqlite3.connect(source_uri, uri=True, timeout=30)
    target_db = sqlite3.connect(destination)
    try:
        source_db.backup(target_db)
    finally:
        target_db.close()
        source_db.close()


def source_connection(snapshot: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(f"file:{snapshot.as_posix()}?mode=ro", uri=True)
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA foreign_keys = ON")
    return connection


def table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?", (table,)
    ).fetchone() is not None


def parse_timestamp(value: Any, naive_zone: ZoneInfo) -> datetime | None:
    if value in (None, ""):
        return None
    if isinstance(value, datetime):
        parsed = value
    else:
        parsed = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=naive_zone)
    return parsed.astimezone(timezone.utc)


def parse_date(value: Any) -> date | None:
    if value in (None, ""):
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    return date.fromisoformat(str(value).strip()[:10])


def parse_json(value: Any, *, polygon: bool = False) -> str | None:
    if value in (None, ""):
        return None
    parsed = value if isinstance(value, (dict, list)) else json.loads(str(value))
    if polygon:
        parsed = normalize_polygon(parsed)
    return json.dumps(parsed, ensure_ascii=False, separators=(",", ":"))


def target_columns(connection, schema: str, table: str) -> set[str]:
    return set(connection.execute(text("""
        SELECT column_name
        FROM information_schema.columns
        WHERE table_schema = :schema AND table_name = :table
          AND is_generated = 'NEVER'
    """), {"schema": schema, "table": table}).scalars())


def transform_row(
    source_table: str,
    target_table: str,
    source_row: sqlite3.Row,
    allowed_columns: set[str],
    naive_zone: ZoneInfo,
) -> dict[str, Any]:
    transformed: dict[str, Any] = {}
    for source_column in source_row.keys():
        if source_column in SKIPPED_COLUMNS:
            continue
        target_column = RENAMED_COLUMNS.get(source_column, source_column)
        if target_column not in allowed_columns:
            continue
        if (target_table, target_column) in DEFERRED_COLUMNS:
            transformed[target_column] = None
            continue
        value = source_row[source_column]
        if target_column in TIMESTAMP_COLUMNS:
            value = parse_timestamp(value, naive_zone)
        elif target_column in DATE_COLUMNS:
            value = parse_date(value)
        elif target_column in JSON_COLUMNS:
            value = parse_json(value, polygon=target_column in POLYGON_COLUMNS)
        transformed[target_column] = value
    return transformed


def insert_row(connection, table: str, row: dict[str, Any]) -> bool:
    columns = list(row)
    column_sql = ", ".join(f'"{column}"' for column in columns)
    values_sql = ", ".join(
        f"CAST(:{column} AS jsonb)" if column in JSON_COLUMNS else f":{column}"
        for column in columns
    )
    result = connection.execute(
        text(
            f'INSERT INTO "{table}" ({column_sql}) VALUES ({values_sql}) '
            "ON CONFLICT DO NOTHING"
        ),
        row,
    )
    return result.rowcount > 0


def load_feature_collection(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(path.read_text(encoding="utf-8-sig"))
    if payload.get("type") == "FeatureCollection":
        return list(payload.get("features") or [])
    if payload.get("type") == "Feature":
        return [payload]
    if payload.get("type") in {"Polygon", "MultiPolygon"}:
        return [{"type": "Feature", "properties": {}, "geometry": payload}]
    raise ValueError(f"{path} is not a polygon GeoJSON layer")


def active_gis_coverage_feature() -> dict[str, Any]:
    from canopy_detection import ortho_matcher

    bounds = ortho_matcher._active_tileset_bounds_latlon()
    if bounds is None:
        raise RuntimeError("The active GIS tile/orthophoto bounds could not be resolved")
    south, west, north, east = [float(value) for value in bounds]
    return {
        "type": "Feature",
        "properties": {"name": "Active MangroVision GIS coverage"},
        "geometry": {
            "type": "Polygon",
            "coordinates": [[
                [west, south], [east, south], [east, north], [west, north], [west, south]
            ]],
        },
    }


def import_geojson_layer(
    connection,
    zone_type: str,
    path: Path | None,
    features: Iterable[dict[str, Any]],
) -> int:
    if not isinstance(features, list):
        features = list(features)
    source_hash = sha256_file(path) if path else hashlib.sha256(
        json.dumps(features, sort_keys=True).encode("utf-8")
    ).hexdigest()
    inserted = 0
    for index, feature in enumerate(features):
        properties = dict(feature.get("properties") or {})
        geometry = normalize_polygon(feature.get("geometry") or {})
        properties.update({
            "migration_source_sha256": source_hash,
            "migration_feature_index": str(index),
            "migration_source_path": str(path) if path else "active-gis-coverage",
        })
        row = {
            "zone_type": zone_type,
            "name": str(properties.pop("name", f"{zone_type.title()} zone {index + 1}")),
            "warning_type": properties.pop("warning_type", None),
            "severity": properties.pop("severity", None),
            "notes": properties.pop("notes", None),
            "properties": json.dumps(properties, ensure_ascii=False),
            "polygon_geojson": json.dumps(geometry, ensure_ascii=False),
        }
        result = connection.execute(text("""
            INSERT INTO map_zones (
                zone_type, name, warning_type, severity, notes,
                properties, polygon_geojson
            ) VALUES (
                :zone_type, :name, :warning_type, :severity, :notes,
                CAST(:properties AS jsonb), CAST(:polygon_geojson AS jsonb)
            )
            ON CONFLICT DO NOTHING
        """), row)
        inserted += max(0, result.rowcount)
    return inserted


def import_warning_zones(source: sqlite3.Connection, target) -> int:
    if not table_exists(source, "warning_zones"):
        return 0
    inserted = 0
    for row in source.execute("SELECT * FROM warning_zones ORDER BY id"):
        geometry = parse_json(row["polygon_geojson"], polygon=True)
        result = target.execute(text("""
            INSERT INTO map_zones (
                id, zone_type, name, warning_type, severity, notes,
                properties, polygon_geojson, created_at, updated_at
            ) VALUES (
                :id, 'warning', :name, :warning_type, :severity, :notes,
                '{}'::jsonb, CAST(:polygon_geojson AS jsonb), :created_at, :updated_at
            ) ON CONFLICT (id) DO NOTHING
        """), {
            "id": row["id"],
            "name": row["name"],
            "warning_type": row["warning_type"],
            "severity": row["severity"],
            "notes": row["notes"],
            "polygon_geojson": geometry,
            "created_at": parse_timestamp(row["created_at"], NAIVE_ZONE),
            "updated_at": parse_timestamp(row["updated_at"], NAIVE_ZONE),
        })
        inserted += max(0, result.rowcount)
    return inserted


def import_assets(source: sqlite3.Connection, target, settings) -> dict[str, int]:
    uploaded = 0
    skipped = 0
    uploaded_keys: list[str] = []
    try:
        for row in source.execute(
            "SELECT id, original_image, visualization_image FROM analyses ORDER BY id"
        ):
            analysis_id = int(row["id"])
            for kind, column in (("original", "original_image"), ("visualization", "visualization_image")):
                data_url = row[column]
                if not data_url:
                    continue
                exists = target.execute(text("""
                    SELECT 1 FROM analysis_assets WHERE analysis_id = :analysis_id AND kind = :kind
                """), {"analysis_id": analysis_id, "kind": kind}).first()
                if exists:
                    skipped += 1
                    continue
                asset = upload_data_url(f"migrated/{analysis_id}", kind, data_url)
                uploaded_keys.append(asset.object_key)
                target.execute(text("""
                    INSERT INTO analysis_assets (
                        analysis_id, kind, object_key, content_type,
                        byte_size, sha256, lifecycle_state
                    ) VALUES (
                        :analysis_id, :kind, :object_key, :content_type,
                        :byte_size, :sha256, 'ready'
                    ) ON CONFLICT (analysis_id, kind) DO NOTHING
                """), {
                    "analysis_id": analysis_id,
                    "kind": asset.kind,
                    "object_key": asset.object_key,
                    "content_type": asset.content_type,
                    "byte_size": asset.byte_size,
                    "sha256": asset.sha256,
                })
                uploaded += 1
    except Exception as error:
        cleanup_keys: list[str] = []
        for object_key in uploaded_keys:
            try:
                delete_object(object_key)
            except Exception:
                cleanup_keys.append(object_key)
        raise AssetImportError(str(error), cleanup_keys) from error
    return {"uploaded": uploaded, "already_present": skipped}


def reset_identity_sequences(connection, schema: str) -> None:
    for table in sorted(IDENTITY_TABLES):
        connection.execute(text(f"""
            SELECT setval(
                pg_get_serial_sequence('{schema}.{table}', 'id'),
                COALESCE((SELECT MAX(id) FROM "{table}"), 1),
                EXISTS (SELECT 1 FROM "{table}")
            )
        """))


def restore_deferred_references(source: sqlite3.Connection, target) -> None:
    if not table_exists(source, "point_death_records"):
        return
    for row in source.execute("""
        SELECT id, monitoring_observation_id
        FROM point_death_records
        WHERE monitoring_observation_id IS NOT NULL
    """):
        target.execute(text("""
            UPDATE point_death_records
            SET monitoring_observation_id = :observation_id
            WHERE id = :id
        """), {"id": row["id"], "observation_id": row["monitoring_observation_id"]})


def verify_assets(target, settings) -> dict[str, int]:
    checked = 0
    mismatched = 0
    client = s3_client()
    rows = target.execute(text("""
        SELECT object_key, byte_size, sha256 FROM analysis_assets ORDER BY id
    """)).mappings()
    for row in rows:
        head = client.head_object(Bucket=settings.s3_bucket, Key=row["object_key"])
        metadata_hash = (head.get("Metadata") or {}).get("sha256")
        if int(head["ContentLength"]) != int(row["byte_size"]) or metadata_hash != row["sha256"]:
            mismatched += 1
        checked += 1
    return {"checked": checked, "mismatched": mismatched}


def source_manifest(source: sqlite3.Connection) -> dict[str, Any]:
    counts = {
        table: int(source.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0])
        for table, _ in TABLE_SPECS
        if table_exists(source, table)
    }
    counts["auth_sessions_skipped"] = (
        int(source.execute("SELECT COUNT(*) FROM auth_sessions").fetchone()[0])
        if table_exists(source, "auth_sessions") else 0
    )
    counts["analysis_assets_expected"] = int(source.execute("""
        SELECT COALESCE(SUM(
            CASE WHEN original_image IS NOT NULL AND original_image <> '' THEN 1 ELSE 0 END
          + CASE WHEN visualization_image IS NOT NULL AND visualization_image <> '' THEN 1 ELSE 0 END
        ), 0)
        FROM analyses
    """).fetchone()[0])
    invalid_foreign_keys = [tuple(row) for row in source.execute("PRAGMA foreign_key_check")]
    return {"source_counts": counts, "sqlite_foreign_key_errors": invalid_foreign_keys}


def target_manifest(target, schema: str) -> dict[str, Any]:
    counts = {
        table: int(target.execute(text(f'SELECT COUNT(*) FROM "{table}"')).scalar_one())
        for table in sorted({target for _, target in TABLE_SPECS} | {"map_zones", "analysis_assets"})
    }
    invalid_geometries = int(target.execute(text("""
        SELECT COUNT(*) FROM (
            SELECT geometry FROM project_sites
            UNION ALL SELECT footprint FROM analyses WHERE footprint IS NOT NULL
            UNION ALL SELECT location FROM planting_points
            UNION ALL SELECT geometry FROM map_zones
        ) geometries
        WHERE NOT extensions.ST_IsValid(geometry)
    """)).scalar_one())
    unvalidated_constraints = int(target.execute(text("""
        SELECT COUNT(*)
        FROM pg_constraint c
        JOIN pg_namespace n ON n.oid = c.connamespace
        WHERE n.nspname = :schema AND NOT c.convalidated
    """), {"schema": schema}).scalar_one())
    return {
        "target_counts": counts,
        "invalid_postgis_geometries": invalid_geometries,
        "unvalidated_constraints": unvalidated_constraints,
        "schema": schema,
    }


def count_mismatches(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    source_counts = manifest.get("source_counts") or {}
    target_counts = manifest.get("target_counts") or {}
    mismatches = []
    for source_table, target_table in TABLE_SPECS:
        if source_table not in source_counts:
            continue
        expected = int(source_counts[source_table])
        actual = int(target_counts.get(target_table, -1))
        if expected != actual:
            mismatches.append({
                "source_table": source_table,
                "target_table": target_table,
                "expected": expected,
                "actual": actual,
            })
    expected_assets = int(source_counts.get("analysis_assets_expected", 0))
    actual_assets = int(target_counts.get("analysis_assets", -1))
    if expected_assets != actual_assets:
        mismatches.append({
            "source_table": "analyses image columns",
            "target_table": "analysis_assets",
            "expected": expected_assets,
            "actual": actual_assets,
        })
    return mismatches


def run(args: argparse.Namespace) -> dict[str, Any]:
    global NAIVE_ZONE
    NAIVE_ZONE = ZoneInfo(args.naive_timezone)
    source_path = args.source.resolve()
    if not source_path.is_file():
        raise FileNotFoundError(source_path)

    settings = get_settings()
    with (
        tempfile.TemporaryDirectory(prefix="mangrovision-migration-") as temp_dir,
        ExitStack() as cleanup,
    ):
        snapshot = Path(temp_dir) / "source.sqlite"
        snapshot_sqlite(source_path, snapshot)
        source_hash = sha256_file(snapshot)
        source = cleanup.enter_context(closing(source_connection(snapshot)))
        manifest = {
            "source_path": str(source_path),
            "source_sha256": source_hash,
            "naive_timestamp_timezone": args.naive_timezone,
            **source_manifest(source),
        }
        if manifest["sqlite_foreign_key_errors"]:
            raise RuntimeError("SQLite foreign-key validation failed; see the dry-run manifest")
        if args.dry_run:
            return manifest

        engine = create_engine(
            settings.migration_database_url,
            connect_args={"sslmode": settings.db_sslmode} if settings.db_sslmode else {},
        )
        cleanup.callback(engine.dispose)
        run_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"mangrovision:{source_hash}"))
        with engine.begin() as target:
            target.execute(text(f'SET LOCAL search_path TO "{settings.db_schema}", extensions, public'))
            revision = target.execute(text(
                f'SELECT version_num FROM "{settings.db_schema}".alembic_version'
            )).scalar_one_or_none()
            if revision != "20260911_0003":
                raise RuntimeError("Run 'alembic upgrade head' before importing data")
            existing = target.execute(text("""
                SELECT status FROM migration_imports WHERE source_sha256 = :source_sha256
            """), {"source_sha256": source_hash}).scalar_one_or_none()
            if existing == "complete" and not args.verify_only:
                raise RuntimeError("This frozen SQLite source has already been migrated")
            if args.verify_only:
                manifest.update(target_manifest(target, settings.db_schema))
                manifest["assets"] = verify_assets(target, settings)
                manifest["count_mismatches"] = count_mismatches(manifest)
                if (
                    manifest["invalid_postgis_geometries"]
                    or manifest["unvalidated_constraints"]
                    or manifest["assets"]["mismatched"]
                    or manifest["count_mismatches"]
                ):
                    raise RuntimeError("Target verification failed; inspect the manifest")
                return manifest
            target.execute(text("""
                INSERT INTO migration_imports (
                    id, source_sha256, source_path, status, manifest, last_error
                ) VALUES (
                    CAST(:id AS uuid), :source_sha256, :source_path,
                    'running', CAST(:manifest AS jsonb), NULL
                )
                ON CONFLICT (source_sha256) DO UPDATE
                    SET status = 'running', last_error = NULL
            """), {
                "id": run_id,
                "source_sha256": source_hash,
                "source_path": str(source_path),
                "manifest": json.dumps(manifest),
            })

        imported: dict[str, int] = {}
        try:
            with engine.begin() as target:
                target.execute(text(f'SET LOCAL search_path TO "{settings.db_schema}", extensions, public'))
                for source_table, target_table in TABLE_SPECS:
                    if not table_exists(source, source_table):
                        continue
                    allowed = target_columns(target, settings.db_schema, target_table)
                    imported_count = 0
                    for source_row in source.execute(f'SELECT * FROM "{source_table}" ORDER BY rowid'):
                        transformed = transform_row(
                            source_table, target_table, source_row, allowed, NAIVE_ZONE
                        )
                        imported_count += int(insert_row(target, target_table, transformed))
                    imported[target_table] = imported_count

                restore_deferred_references(source, target)
                imported["warning_zones"] = import_warning_zones(source, target)
                reset_identity_sequences(target, settings.db_schema)
                imported["gis_coverage"] = import_geojson_layer(
                    target, "gis_coverage", None, [active_gis_coverage_feature()]
                )
                for zone_type, path in (("eroded", args.eroded), ("forbidden", args.forbidden)):
                    if path and path.is_file():
                        imported[zone_type] = import_geojson_layer(
                            target, zone_type, path, load_feature_collection(path)
                        )
                reset_identity_sequences(target, settings.db_schema)

            with engine.begin() as target:
                target.execute(text(f'SET LOCAL search_path TO "{settings.db_schema}", extensions, public'))
                assets = import_assets(source, target, settings)

            with engine.begin() as target:
                target.execute(text(f'SET LOCAL search_path TO "{settings.db_schema}", extensions, public'))
                manifest["imported_rows"] = imported
                manifest["assets"] = assets
                manifest.update(target_manifest(target, settings.db_schema))
                manifest["asset_verification"] = verify_assets(target, settings)
                manifest["count_mismatches"] = count_mismatches(manifest)
                if (
                    manifest["invalid_postgis_geometries"]
                    or manifest["unvalidated_constraints"]
                    or manifest["asset_verification"]["mismatched"]
                    or manifest["count_mismatches"]
                ):
                    raise RuntimeError("Post-migration count, constraint, geometry, or asset verification failed")
                target.execute(text("""
                    UPDATE migration_imports
                    SET status = 'complete', completed_at = CURRENT_TIMESTAMP,
                        manifest = CAST(:manifest AS jsonb), last_error = NULL
                    WHERE source_sha256 = :source_sha256
                """), {
                    "manifest": json.dumps(manifest, default=str),
                    "source_sha256": source_hash,
                })
        except Exception as error:
            with engine.begin() as target:
                target.execute(text(f'SET LOCAL search_path TO "{settings.db_schema}", extensions, public'))
                for object_key in getattr(error, "cleanup_keys", []):
                    target.execute(text("""
                        INSERT INTO object_cleanup_jobs (object_key, reason, last_error)
                        VALUES (:object_key, 'migration_asset_compensation', :error)
                        ON CONFLICT (object_key) DO UPDATE
                            SET last_error = EXCLUDED.last_error,
                                completed_at = NULL
                    """), {"object_key": object_key, "error": str(error)[:1000]})
                target.execute(text("""
                    UPDATE migration_imports
                    SET status = 'failed', last_error = :error
                    WHERE source_sha256 = :source_sha256
                """), {"error": str(error)[:4000], "source_sha256": source_hash})
            raise
        return manifest


def parser() -> argparse.ArgumentParser:
    command = argparse.ArgumentParser(description=__doc__)
    command.add_argument(
        "--source",
        type=Path,
        default=Path(os.getenv("SQLITE_MIGRATION_SOURCE", "planting_zones.db")),
    )
    command.add_argument("--eroded", type=Path, default=ROOT / "eroded_zones.geojson")
    default_forbidden = next(
        (path for path in (ROOT / "new_forbidden.geojson", ROOT / "forbidden_zone_final.geojson") if path.exists()),
        ROOT / "forbidden_zones.geojson",
    )
    command.add_argument("--forbidden", type=Path, default=default_forbidden)
    command.add_argument("--naive-timezone", default="Asia/Manila")
    command.add_argument("--dry-run", action="store_true", help="Validate the frozen source only")
    command.add_argument("--verify-only", action="store_true", help="Verify an existing target import")
    command.add_argument("--manifest", type=Path, help="Write the JSON verification manifest")
    return command


def main() -> int:
    args = parser().parse_args()
    try:
        manifest = run(args)
    except Exception as error:
        print(f"Migration failed: {error}", file=sys.stderr)
        return 1
    rendered = json.dumps(manifest, indent=2, default=str)
    print(rendered)
    if args.manifest:
        args.manifest.parent.mkdir(parents=True, exist_ok=True)
        args.manifest.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
