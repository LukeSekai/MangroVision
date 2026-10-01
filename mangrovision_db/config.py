"""Environment-only configuration for database, storage, and sessions."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

from dotenv import load_dotenv


# Local development only. Hosting-provided environment variables always win.
load_dotenv(Path(__file__).resolve().parents[1] / ".env", override=False)


def _as_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _as_int(name: str, default: int, minimum: int = 1) -> int:
    try:
        return max(minimum, int(os.getenv(name, str(default))))
    except ValueError:
        return default


def _psycopg_url(value: str) -> str:
    value = value.strip()
    if value.startswith("postgres://"):
        return "postgresql+psycopg://" + value[len("postgres://") :]
    if value.startswith("postgresql://"):
        return "postgresql+psycopg://" + value[len("postgresql://") :]
    return value


@dataclass(frozen=True)
class Settings:
    database_url: str
    migration_database_url: str
    db_schema: str
    db_pool_size: int
    db_max_overflow: int
    db_sslmode: str
    s3_endpoint_url: str
    s3_region: str
    s3_access_key_id: str
    s3_secret_access_key: str
    s3_bucket: str
    s3_presigned_url_ttl_seconds: int
    s3_force_path_style: bool
    cookie_secure: bool
    cookie_domain: str | None
    trusted_origins: tuple[str, ...]
    staff_session_hours: int
    planter_session_days: int


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    database_url = _psycopg_url(
        os.getenv(
            "DATABASE_URL",
            "postgresql+psycopg://mangrovision:mangrovision@127.0.0.1:5432/mangrovision",
        )
    )
    migration_url = _psycopg_url(os.getenv("MIGRATION_DATABASE_URL", database_url))
    trusted = tuple(
        origin.strip().rstrip("/")
        for origin in os.getenv(
            "TRUSTED_ORIGINS",
            "http://localhost:5173,http://127.0.0.1:5173",
        ).split(",")
        if origin.strip()
    )
    cookie_domain = os.getenv("COOKIE_DOMAIN", "").strip() or None
    return Settings(
        database_url=database_url,
        migration_database_url=migration_url,
        db_schema=os.getenv("DB_SCHEMA", "mangrovision").strip() or "mangrovision",
        db_pool_size=_as_int("DB_POOL_SIZE", 5),
        db_max_overflow=_as_int("DB_MAX_OVERFLOW", 5, minimum=0),
        db_sslmode=os.getenv("DB_SSLMODE", "prefer").strip() or "prefer",
        s3_endpoint_url=os.getenv("S3_ENDPOINT_URL", "http://127.0.0.1:9000").rstrip("/"),
        s3_region=os.getenv("S3_REGION", "us-east-1"),
        s3_access_key_id=os.getenv("S3_ACCESS_KEY_ID", ""),
        s3_secret_access_key=os.getenv("S3_SECRET_ACCESS_KEY", ""),
        s3_bucket=os.getenv("S3_BUCKET", "analysis-images"),
        s3_presigned_url_ttl_seconds=_as_int("S3_PRESIGNED_URL_TTL_SECONDS", 900),
        s3_force_path_style=_as_bool("S3_FORCE_PATH_STYLE", True),
        cookie_secure=_as_bool("COOKIE_SECURE", False),
        cookie_domain=cookie_domain,
        trusted_origins=trusted,
        staff_session_hours=_as_int("STAFF_SESSION_HOURS", 12),
        planter_session_days=_as_int("PLANTER_SESSION_DAYS", 7),
    )
