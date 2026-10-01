"""Read-only integrity and count audit for a MangroVision SQLite source."""

from __future__ import annotations

import argparse
import hashlib
import json
import sqlite3
from pathlib import Path


def table_exists(connection: sqlite3.Connection, table: str) -> bool:
    return connection.execute(
        "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ?",
        (table,),
    ).fetchone() is not None


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def snapshot(source: Path, destination: Path) -> dict[str, object]:
    resolved_source = source.resolve(strict=True)
    resolved_destination = destination.resolve()
    if resolved_destination.exists():
        raise FileExistsError(resolved_destination)
    resolved_destination.parent.mkdir(parents=True, exist_ok=True)

    source_connection = sqlite3.connect(
        f"file:{resolved_source.as_posix()}?mode=ro",
        uri=True,
        timeout=30,
    )
    destination_connection = sqlite3.connect(resolved_destination)
    try:
        source_connection.backup(destination_connection)
    finally:
        destination_connection.close()
        source_connection.close()

    return {
        "path": str(resolved_destination),
        "bytes": resolved_destination.stat().st_size,
        "sha256": sha256_file(resolved_destination),
    }


def audit(source: Path) -> dict[str, object]:
    resolved = source.resolve(strict=True)
    connection = sqlite3.connect(
        f"file:{resolved.as_posix()}?mode=ro",
        uri=True,
        timeout=30,
    )
    try:
        tables = [
            str(row[0])
            for row in connection.execute(
                """
                SELECT name
                FROM sqlite_master
                WHERE type = 'table' AND name NOT LIKE 'sqlite_%'
                ORDER BY name
                """
            )
        ]
        counts = {
            table: int(
                connection.execute(f'SELECT COUNT(*) FROM "{table}"').fetchone()[0]
            )
            for table in tables
        }
        foreign_key_errors = [
            list(row) for row in connection.execute("PRAGMA foreign_key_check")
        ]
        quick_check = str(connection.execute("PRAGMA quick_check").fetchone()[0])

        asset_count = 0
        original_characters = 0
        visualization_characters = 0
        if table_exists(connection, "analyses"):
            row = connection.execute(
                """
                SELECT
                  COALESCE(SUM(
                    CASE WHEN original_image IS NOT NULL AND original_image <> '' THEN 1 ELSE 0 END
                    + CASE WHEN visualization_image IS NOT NULL AND visualization_image <> '' THEN 1 ELSE 0 END
                  ), 0),
                  COALESCE(SUM(LENGTH(original_image)), 0),
                  COALESCE(SUM(LENGTH(visualization_image)), 0)
                FROM analyses
                """
            ).fetchone()
            asset_count = int(row[0])
            original_characters = int(row[1])
            visualization_characters = int(row[2])

        return {
            "source": str(resolved),
            "source_bytes": resolved.stat().st_size,
            "quick_check": quick_check,
            "table_counts": counts,
            "foreign_key_error_count": len(foreign_key_errors),
            "foreign_key_errors_sample": foreign_key_errors[:20],
            "analysis_assets_expected": asset_count,
            "embedded_image_characters": {
                "original": original_characters,
                "visualization": visualization_characters,
            },
        }
    finally:
        connection.close()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source", nargs="?", type=Path, default=Path("planting_zones.db"))
    parser.add_argument(
        "--snapshot",
        type=Path,
        help="Create a consistent SQLite backup at this new path before auditing it",
    )
    args = parser.parse_args()
    result: dict[str, object] = {}
    audited_source = args.source
    if args.snapshot:
        result["snapshot"] = snapshot(args.source, args.snapshot)
        audited_source = args.snapshot
    result["audit"] = audit(audited_source)
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
