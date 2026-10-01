"""Store observed growth categories without inventing height measurements."""

from alembic import op

revision = "20260912_0004"
down_revision = "20260911_0003"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("SET LOCAL statement_timeout = '30s'")
    op.execute("""
        ALTER TABLE mangrovision.organization_monitoring_records
        ALTER COLUMN average_height_cm DROP NOT NULL,
        ADD COLUMN growth_stage TEXT,
        ADD CONSTRAINT ck_monitoring_growth_stage CHECK (
            growth_stage IS NULL OR growth_stage IN (
                'seedling', 'young', 'larger', 'mixed', 'not_checked', 'no_living'
            )
        ),
        ADD CONSTRAINT ck_monitoring_growth_evidence CHECK (
            growth_stage IS NOT NULL OR average_height_cm IS NOT NULL
        ),
        ADD CONSTRAINT ck_monitoring_living_growth CHECK (
            growth_stage IS NULL OR
            (alive_count = 0 AND growth_stage = 'no_living') OR
            (alive_count > 0 AND growth_stage <> 'no_living')
        )
    """)


def downgrade() -> None:
    # Refuse a downgrade that would discard category-only visit evidence.
    op.execute("""
        ALTER TABLE mangrovision.organization_monitoring_records
        ALTER COLUMN average_height_cm SET NOT NULL,
        DROP CONSTRAINT ck_monitoring_living_growth,
        DROP CONSTRAINT ck_monitoring_growth_evidence,
        DROP CONSTRAINT ck_monitoring_growth_stage,
        DROP COLUMN growth_stage
    """)
