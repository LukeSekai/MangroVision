"""Verify the additive migration without changing records or printing secrets.

--rehearse runs upgrade/downgrade inside a transaction and rolls everything back.
Default verifies the applied revision, view access, nullable column and app reads.
"""

import argparse
import importlib.util
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from alembic.migration import MigrationContext
from alembic.operations import Operations
from sqlalchemy import create_engine, text

from mangrovision_db.config import get_settings


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rehearse", action="store_true")
    args = parser.parse_args()
    settings = get_settings()
    engine = create_engine(settings.migration_database_url,
                           connect_args={"sslmode": settings.db_sslmode, "connect_timeout": 10})
    with engine.connect() as conn:
        transaction = conn.begin()
        try:
            conn.execute(text("SET LOCAL lock_timeout = '5s'"))
            revision = conn.execute(text("SELECT version_num FROM mangrovision.alembic_version")).scalar_one()
            before = conn.execute(text("SELECT count(*) FROM mangrovision.project_sites")).scalar_one()
            if args.rehearse:
                assert revision == "20260904_0002", "Rehearsal requires the previous revision"
                spec = importlib.util.spec_from_file_location("calibration_migration", ROOT / "alembic/versions/20260911_0003_tide_calibration.py")
                migration = importlib.util.module_from_spec(spec)
                spec.loader.exec_module(migration)
                with Operations.context(MigrationContext.configure(conn)):
                    migration.upgrade()
                    assert conn.execute(text("SELECT count(*) FROM mangrovision.project_sites WHERE tide_calibration IS NULL")).scalar_one() == before
                    migration.downgrade()
                    migration.upgrade()
            else:
                assert revision == "20260911_0003", "Apply the tide calibration migration first"
            column = conn.execute(text("""
                SELECT data_type, is_nullable FROM information_schema.columns
                WHERE table_schema = 'mangrovision' AND table_name = 'project_sites' AND column_name = 'tide_calibration'
            """)).one()
            assert tuple(column) == ("jsonb", "YES")
            assert conn.execute(text("SELECT count(*) FROM mangrovision.site_zones")).scalar_one() == before
            print("Migration upgrade/downgrade rehearsal passed; all changes rolled back." if args.rehearse
                  else "Applied revision, nullable calibration and unchanged compatibility view verified.")
        finally:
            transaction.rollback()
    engine.dispose()
    if not args.rehearse:
        from planting_database import list_project_sites
        assert all("tide_calibration" in site["properties"] for site in list_project_sites())
        print("Project-site reads verified using the application's database role.")


if __name__ == "__main__":
    main()
