"""Optional surveyed tide calibration; no defaults or schedule workflow changes."""

from alembic import op

revision = "20260911_0003"
down_revision = "20260904_0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    # The existing private-schema access controls and compatibility view remain
    # untouched. This small, cohesive measurement document is not an FK store.
    op.execute("""
        ALTER TABLE mangrovision.project_sites
        ADD COLUMN tide_calibration JSONB,
        ADD CONSTRAINT ck_project_sites_tide_calibration CHECK (
            tide_calibration IS NULL OR (
                jsonb_typeof(tide_calibration) = 'object'
                AND tide_calibration ?& ARRAY[
                    'elevation_m', 'datum_reference', 'survey_reference',
                    'forecast_lat', 'forecast_lon', 'recorded_at'
                ]
                AND jsonb_typeof(tide_calibration->'elevation_m') = 'number'
                AND jsonb_typeof(tide_calibration->'datum_reference') = 'string'
            )
        )
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE mangrovision.project_sites
        DROP CONSTRAINT ck_project_sites_tide_calibration,
        DROP COLUMN tide_calibration
    """)
