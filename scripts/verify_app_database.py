"""Verify the restricted application database login without exposing credentials."""

from __future__ import annotations

import sys
from pathlib import Path

from sqlalchemy import create_engine, text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mangrovision_db.config import get_settings


def main() -> int:
    settings = get_settings()
    engine = create_engine(
        settings.database_url,
        connect_args={"sslmode": settings.db_sslmode} if settings.db_sslmode else {},
    )
    try:
        with engine.connect() as connection:
            user = connection.execute(text("SELECT current_user")).scalar_one()
            schema = connection.execute(text("SELECT current_schema()")).scalar_one()
            revision = connection.execute(
                text("SELECT version_num FROM mangrovision.alembic_version")
            ).scalar_one()
            table_count = connection.execute(
                text(
                    "SELECT count(*) FROM information_schema.tables "
                    "WHERE table_schema = :schema"
                ),
                {"schema": settings.db_schema},
            ).scalar_one()
            attributes = connection.execute(
                text(
                    "SELECT rolsuper, rolcreatedb, rolcreaterole, "
                    "rolreplication, rolbypassrls "
                    "FROM pg_roles WHERE rolname = current_user"
                )
            ).one()
            has_crud = connection.execute(
                text(
                    "SELECT has_table_privilege("
                    "current_user, :table_name, :privileges)"
                ),
                {
                    "table_name": f"{settings.db_schema}.users",
                    "privileges": "SELECT,INSERT,UPDATE,DELETE",
                },
            ).scalar_one()

        checks = {
            "LOGIN_USER_OK": user == "mangrovision",
            "SEARCH_PATH_OK": schema == settings.db_schema,
            "ALEMBIC_REVISION_PRESENT": bool(revision),
            "TABLES_VISIBLE": table_count > 0,
            "CRUD_GRANTS_OK": has_crud,
            "RESTRICTED_ROLE_OK": not any(attributes),
        }
        for name, passed in checks.items():
            print(f"{name}={passed}")
        print(f"ALEMBIC_REVISION={revision}")
        print(f"TABLE_COUNT={table_count}")
        return 0 if all(checks.values()) else 1
    except Exception as error:
        print(f"Application database verification failed: {error}", file=sys.stderr)
        return 1
    finally:
        engine.dispose()


if __name__ == "__main__":
    raise SystemExit(main())
