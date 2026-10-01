"""Restore an encrypted MangroVision PostgreSQL and object backup."""

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
from pathlib import Path, PurePosixPath

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mangrovision_db.config import get_settings
from mangrovision_db.storage import s3_client
from scripts.backup_production import postgres_environment


def digest(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(chunk)
    return result.hexdigest()


def safe_extract(archive: tarfile.TarFile, destination: Path) -> None:
    destination = destination.resolve()
    for member in archive.getmembers():
        candidate = (destination / member.name).resolve()
        if destination not in candidate.parents and candidate != destination:
            raise ValueError(f"Unsafe backup member: {member.name}")
        if member.issym() or member.islnk():
            raise ValueError("Backup archives may not contain links")
    archive.extractall(destination)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("backup", type=Path)
    parser.add_argument(
        "--confirm-restore",
        action="store_true",
        help="Required: replaces database objects represented by the backup",
    )
    args = parser.parse_args()
    if not args.confirm_restore:
        print("Restore refused without --confirm-restore.", file=sys.stderr)
        return 2
    if not args.backup.is_file():
        print(f"Backup not found: {args.backup}", file=sys.stderr)
        return 1
    passphrase = os.getenv("BACKUP_ENCRYPTION_PASSWORD", "")
    if not passphrase:
        print("BACKUP_ENCRYPTION_PASSWORD is required.", file=sys.stderr)
        return 1
    pg_restore = shutil.which("pg_restore")
    openssl = shutil.which("openssl")
    if not pg_restore or not openssl:
        print("pg_restore and openssl must be installed and available on PATH.", file=sys.stderr)
        return 1

    settings = get_settings()
    with tempfile.TemporaryDirectory(prefix="mangrovision-restore-") as temp_name:
        temp = Path(temp_name)
        archive_path = temp / "backup.tar"
        encryption_env = os.environ.copy()
        encryption_env["MANGROVISION_BACKUP_PASSPHRASE"] = passphrase
        subprocess.run([
            openssl, "enc", "-d", "-aes-256-cbc", "-pbkdf2",
            "-in", str(args.backup.resolve()), "-out", str(archive_path),
            "-pass", "env:MANGROVISION_BACKUP_PASSPHRASE",
        ], env=encryption_env, check=True)
        with tarfile.open(archive_path, "r") as archive:
            safe_extract(archive, temp)

        manifest = json.loads((temp / "manifest.json").read_text(encoding="utf-8"))
        if manifest.get("schema") != settings.db_schema:
            raise RuntimeError(
                "Backup schema does not match DB_SCHEMA; refusing to restore into an unexpected schema"
            )
        database_dump = temp / "database.dump"
        if digest(database_dump) != manifest["database_dump_sha256"]:
            raise RuntimeError("Database dump checksum does not match the backup manifest")
        for item in manifest["objects"]:
            relative = PurePosixPath(item["key"])
            local_path = temp / "objects" / Path(*relative.parts)
            if digest(local_path) != item["sha256"]:
                raise RuntimeError(f"Object checksum mismatch: {item['key']}")

        pg_env, _ = postgres_environment(settings.migration_database_url)
        subprocess.run([
            pg_restore, "--clean", "--if-exists", "--no-owner",
            "--exit-on-error", str(database_dump),
        ], env=pg_env, check=True)

        client = s3_client()
        for item in manifest["objects"]:
            relative = PurePosixPath(item["key"])
            local_path = temp / "objects" / Path(*relative.parts)
            extra_args = {
                "ContentType": item.get("content_type") or "application/octet-stream",
                "Metadata": item.get("metadata") or {},
            }
            if item.get("cache_control"):
                extra_args["CacheControl"] = item["cache_control"]
            client.upload_file(
                str(local_path),
                settings.s3_bucket,
                item["key"],
                ExtraArgs=extra_args,
            )
            head = client.head_object(Bucket=settings.s3_bucket, Key=item["key"])
            if int(head["ContentLength"]) != int(item["byte_size"]):
                raise RuntimeError(f"Restored object size mismatch: {item['key']}")
            expected_metadata = item.get("metadata") or {}
            if expected_metadata and (head.get("Metadata") or {}) != expected_metadata:
                raise RuntimeError(f"Restored object metadata mismatch: {item['key']}")

    print("Restore completed and all archived object checksums were verified.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
