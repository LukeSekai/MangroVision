"""Create the first LGU administrator from explicit environment values."""

from __future__ import annotations

import os
import sys
from pathlib import Path

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mangrovision_db.config import get_settings
from mangrovision_db.passwords import hash_password


def required(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} is required")
    return value


def main() -> int:
    try:
        full_name = required("BOOTSTRAP_ADMIN_USERNAME")
        email = required("BOOTSTRAP_ADMIN_EMAIL").lower()
        password = required("BOOTSTRAP_ADMIN_PASSWORD")
        if "@" not in email:
            raise RuntimeError("BOOTSTRAP_ADMIN_EMAIL must be a valid email address")
        if len(password) < 12:
            raise RuntimeError("BOOTSTRAP_ADMIN_PASSWORD must contain at least 12 characters")

        settings = get_settings()
        engine = create_engine(
            settings.migration_database_url,
            connect_args={"sslmode": settings.db_sslmode} if settings.db_sslmode else {},
        )
        with engine.begin() as connection:
            connection.execute(text(
                f'SET LOCAL search_path TO "{settings.db_schema}", extensions, public'
            ))
            existing = connection.execute(text("""
                SELECT id FROM users
                WHERE lower(full_name) = lower(:full_name) OR lower(email) = lower(:email)
            """), {"full_name": full_name, "email": email}).first()
            if existing:
                raise RuntimeError("An account with that name or email already exists")
            user_id = connection.execute(text("""
                INSERT INTO users (full_name, email, role, password_hash)
                VALUES (:full_name, :email, 'admin', :password_hash)
                RETURNING id
            """), {
                "full_name": full_name,
                "email": email,
                "password_hash": hash_password(password),
            }).scalar_one()
        engine.dispose()
    except Exception as error:
        print(f"Administrator bootstrap failed: {error}", file=sys.stderr)
        return 1

    print(f"Created administrator #{user_id} ({full_name}).")
    print("Remove BOOTSTRAP_ADMIN_* values from the environment now.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
