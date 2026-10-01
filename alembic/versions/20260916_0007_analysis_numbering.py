"""Give saved analyses a display number independent of database row IDs."""

from alembic import op

revision = "20260916_0007"
down_revision = "20260912_0006"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("SET LOCAL statement_timeout = '60s'")
    op.execute("""
        ALTER TABLE mangrovision.analyses ADD COLUMN analysis_number BIGINT;
        CREATE SEQUENCE mangrovision.analysis_number_seq
            OWNED BY mangrovision.analyses.analysis_number;
        WITH numbered AS (
            SELECT id, ROW_NUMBER() OVER (ORDER BY analyzed_at, id) AS number
            FROM mangrovision.analyses
        )
        UPDATE mangrovision.analyses a
        SET analysis_number = numbered.number,
            analysis_detail_json = COALESCE(a.analysis_detail_json, '{}'::jsonb)
                || jsonb_build_object(
                    'previous_saved_name', a.image_name,
                    'source_image_name', COALESCE(
                        a.analysis_detail_json->>'source_image_name', a.image_name
                    )
                ),
            image_name = 'Analysis ' || numbered.number
        FROM numbered WHERE a.id = numbered.id;
        SELECT setval('mangrovision.analysis_number_seq',
            COALESCE((SELECT MAX(analysis_number) FROM mangrovision.analyses), 0) + 1,
            false);
        ALTER TABLE mangrovision.analyses
            ALTER COLUMN analysis_number SET NOT NULL,
            ALTER COLUMN analysis_number SET DEFAULT nextval('mangrovision.analysis_number_seq'),
            ADD CONSTRAINT uq_analyses_analysis_number UNIQUE (analysis_number),
            ADD CONSTRAINT ck_analyses_analysis_number_positive CHECK (analysis_number > 0);
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mangrovision') THEN
                GRANT USAGE, SELECT ON SEQUENCE mangrovision.analysis_number_seq TO mangrovision;
            END IF;
        END $$;
    """)


def downgrade() -> None:
    op.execute("""
        UPDATE mangrovision.analyses
        SET image_name = COALESCE(
            analysis_detail_json->>'previous_saved_name',
            analysis_detail_json->>'source_image_name', image_name
        );
        ALTER TABLE mangrovision.analyses DROP COLUMN analysis_number;
    """)
