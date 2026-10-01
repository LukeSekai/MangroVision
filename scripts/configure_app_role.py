"""Enable the restricted PostgreSQL login using the password in DATABASE_URL.

The password never appears in SQLAlchemy logs or command-line arguments. Run this
once with an owner connection in MIGRATION_DATABASE_URL after Alembic creates the
``mangrovision`` role.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

from psycopg import sql
from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mangrovision_db.config import get_settings


def main() -> int:
    settings = get_settings()
    role_name = os.getenv("APP_DATABASE_ROLE", "mangrovision").strip()
    if role_name != "mangrovision":
        print("APP_DATABASE_ROLE must be 'mangrovision'.", file=sys.stderr)
        return 1

    app_url = make_url(settings.database_url)
    app_username = app_url.username or ""
    if app_username not in {role_name} and not app_username.startswith(f"{role_name}."):
        print(
            "DATABASE_URL must authenticate as mangrovision (or mangrovision.<project-ref> through Supavisor).",
            file=sys.stderr,
        )
        return 1
    if not app_url.password:
        print("DATABASE_URL must contain the new application-role password.", file=sys.stderr)
        return 1

    engine = create_engine(
        settings.migration_database_url,
        connect_args={"sslmode": settings.db_sslmode} if settings.db_sslmode else {},
    )
    raw = None
    try:
        with engine.connect() as connection:
            current_user = connection.execute(text("SELECT CURRENT_USER")).scalar_one()
            role_exists = connection.execute(
                text("SELECT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :role)"),
                {"role": role_name},
            ).scalar_one()
        if not role_exists:
            raise RuntimeError("Run 'alembic upgrade head' before configuring the application role")
        if current_user == role_name:
            raise RuntimeError("MIGRATION_DATABASE_URL must use the owner/migration account")

        raw = engine.raw_connection()
        with raw.cursor() as cursor:
            cursor.execute(
                sql.SQL("ALTER ROLE {} LOGIN PASSWORD {}").format(
                    sql.Identifier(role_name),
                    sql.Literal(app_url.password),
                )
            )
            cursor.execute(
                sql.SQL("ALTER ROLE {} SET search_path TO mangrovision, extensions, public").format(
                    sql.Identifier(role_name)
                )
            )
            cursor.execute(
                sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(
                    sql.Identifier(raw.driver_connection.info.dbname),
                    sql.Identifier(role_name),
                )
            )
        raw.commit()
    except Exception as error:
        if raw is not None:
            raw.rollback()
        print(f"Application-role configuration failed: {error}", file=sys.stderr)
        return 1
    finally:
        if raw is not None:
            raw.close()
        engine.dispose()

    print("Restricted application login configured; no credential was printed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
