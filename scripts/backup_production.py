"""Create an encrypted PostgreSQL plus private-object backup archive."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tarfile
import tempfile
from datetime import datetime, timezone
from pathlib import Path, PurePosixPath

from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mangrovision_db.config import get_settings
from mangrovision_db.storage import s3_client


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def postgres_environment(database_url: str) -> tuple[dict[str, str], str]:
    parsed = make_url(database_url)
    environment = os.environ.copy()
    environment.update({
        "PGHOST": parsed.host or "127.0.0.1",
        "PGPORT": str(parsed.port or 5432),
        "PGUSER": parsed.username or "",
        "PGPASSWORD": parsed.password or "",
        "PGDATABASE": parsed.database or "",
    })
    sslmode = parsed.query.get("sslmode") or get_settings().db_sslmode
    if sslmode:
        environment["PGSSLMODE"] = sslmode
    return environment, parsed.database or "mangrovision"


def safe_object_path(root: Path, object_key: str) -> Path:
    parts = PurePosixPath(object_key).parts
    if not parts or any(part in {"", ".", ".."} for part in parts):
        raise ValueError(f"Unsafe object key: {object_key!r}")
    return root.joinpath(*parts)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "backups")
    args = parser.parse_args()
    passphrase = os.getenv("BACKUP_ENCRYPTION_PASSWORD", "")
    if len(passphrase) < 16:
        print("BACKUP_ENCRYPTION_PASSWORD must contain at least 16 characters.", file=sys.stderr)
        return 1
    pg_dump = shutil.which("pg_dump")
    openssl = shutil.which("openssl")
    if not pg_dump or not openssl:
        print("pg_dump and openssl must be installed and available on PATH.", file=sys.stderr)
        return 1

    settings = get_settings()
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    encrypted_path = args.output_dir / f"mangrovision-{timestamp}.tar.enc"
    with tempfile.TemporaryDirectory(prefix="mangrovision-backup-") as temp_name:
        temp = Path(temp_name)
        database_dump = temp / "database.dump"
        object_root = temp / "objects"
        object_root.mkdir()
        pg_env, database_name = postgres_environment(settings.migration_database_url)
        subprocess.run(
            [
                pg_dump,
                "--format=custom",
                "--schema",
                settings.db_schema,
                "--file",
                str(database_dump),
            ],
            env=pg_env,
            check=True,
        )

        objects: list[dict[str, object]] = []
        client = s3_client()
        paginator = client.get_paginator("list_objects_v2")
        for page in paginator.paginate(Bucket=settings.s3_bucket):
            for item in page.get("Contents", []):
                key = str(item["Key"])
                local_path = safe_object_path(object_root, key)
                local_path.parent.mkdir(parents=True, exist_ok=True)
                client.download_file(settings.s3_bucket, key, str(local_path))
                head = client.head_object(Bucket=settings.s3_bucket, Key=key)
                objects.append({
                    "key": key,
                    "byte_size": local_path.stat().st_size,
                    "sha256": digest(local_path),
                    "content_type": head.get("ContentType") or "application/octet-stream",
                    "cache_control": head.get("CacheControl"),
                    "metadata": head.get("Metadata") or {},
                })

        manifest = {
            "created_at": datetime.now(timezone.utc).isoformat(),
            "database": database_name,
            "schema": settings.db_schema,
            "database_dump_sha256": digest(database_dump),
            "bucket": settings.s3_bucket,
            "objects": objects,
        }
        (temp / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        archive = temp / "backup.tar"
        with tarfile.open(archive, "w") as tar:
            tar.add(database_dump, arcname="database.dump")
            tar.add(temp / "manifest.json", arcname="manifest.json")
            tar.add(object_root, arcname="objects")

        encryption_env = os.environ.copy()
        encryption_env["MANGROVISION_BACKUP_PASSPHRASE"] = passphrase
        subprocess.run([
            openssl, "enc", "-aes-256-cbc", "-salt", "-pbkdf2",
            "-in", str(archive), "-out", str(encrypted_path),
            "-pass", "env:MANGROVISION_BACKUP_PASSPHRASE",
        ], env=encryption_env, check=True)

    print(f"Encrypted backup created: {encrypted_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
