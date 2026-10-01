"""Snapshot automatic growth and distinguish new deaths from cumulative deaths."""
from alembic import op

revision = '20260912_0005'
down_revision = '20260912_0004'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("SET LOCAL statement_timeout = '30s'")
    op.execute("""
        ALTER TABLE mangrovision.organization_monitoring_records
        ADD COLUMN new_dead_count INTEGER CHECK (new_dead_count >= 0 AND new_dead_count <= dead_count),
        ADD COLUMN alive_before_count INTEGER CHECK (alive_before_count >= 0),
        ADD COLUMN baseline_record_id BIGINT REFERENCES mangrovision.organization_monitoring_records(id),
        ADD COLUMN growth_snapshot JSONB CHECK (growth_snapshot IS NULL OR jsonb_typeof(growth_snapshot) = 'object'),
        ADD COLUMN count_snapshot JSONB CHECK (count_snapshot IS NULL OR jsonb_typeof(count_snapshot) = 'object'),
        ADD CONSTRAINT ck_monitoring_progress_counts CHECK (
            new_dead_count IS NULL OR (
                alive_before_count IS NOT NULL AND
                alive_count = alive_before_count - new_dead_count AND
                growth_snapshot IS NOT NULL AND count_snapshot IS NOT NULL
            )
        ),
        DROP CONSTRAINT ck_monitoring_growth_stage,
        ADD CONSTRAINT ck_monitoring_growth_stage CHECK (
            growth_stage IS NULL OR growth_stage IN (
                'seedling', 'young', 'larger', 'mixed', 'not_checked', 'no_living', 'age_estimate'
            )
        )
    """)
    op.execute('CREATE INDEX idx_monitoring_baseline_record ON mangrovision.organization_monitoring_records(baseline_record_id)')


def downgrade():
    # Refuse removal while automatic snapshots exist; do not erase visit evidence.
    op.execute("""DO $$ BEGIN
        IF EXISTS (SELECT 1 FROM mangrovision.organization_monitoring_records WHERE new_dead_count IS NOT NULL) THEN
            RAISE EXCEPTION 'Automatic visit snapshots exist; preserve them before downgrading';
        END IF;
    END $$""")
    op.execute("""ALTER TABLE mangrovision.organization_monitoring_records
        DROP CONSTRAINT ck_monitoring_progress_counts,
        DROP COLUMN new_dead_count, DROP COLUMN alive_before_count,
        DROP COLUMN baseline_record_id, DROP COLUMN growth_snapshot, DROP COLUMN count_snapshot,
        DROP CONSTRAINT ck_monitoring_growth_stage,
        ADD CONSTRAINT ck_monitoring_growth_stage CHECK (growth_stage IS NULL OR growth_stage IN (
            'seedling', 'young', 'larger', 'mixed', 'not_checked', 'no_living'))
    """)
