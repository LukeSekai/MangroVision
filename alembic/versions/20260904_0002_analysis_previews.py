"""Allow compact private preview assets for analysis history."""

from alembic import op

revision = "20260904_0002"
down_revision = "20260831_0001"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET search_path TO mangrovision, extensions, public")
    op.execute(
        """
        ALTER TABLE analysis_assets
            DROP CONSTRAINT analysis_assets_kind_check,
            ADD CONSTRAINT analysis_assets_kind_check
                CHECK (kind IN (
                    'original', 'visualization',
                    'original_preview', 'visualization_preview'
                ))
        """
    )


def downgrade() -> None:
    op.execute("SET search_path TO mangrovision, extensions, public")
    op.execute(
        """
        INSERT INTO object_cleanup_jobs (object_key, reason)
        SELECT object_key, 'analysis_preview_migration_downgrade'
        FROM analysis_assets
        WHERE kind IN ('original_preview', 'visualization_preview')
        ON CONFLICT (object_key) DO NOTHING
        """
    )
    op.execute(
        """
        DELETE FROM analysis_assets
        WHERE kind IN ('original_preview', 'visualization_preview')
        """
    )
    op.execute(
        """
        ALTER TABLE analysis_assets
            DROP CONSTRAINT analysis_assets_kind_check,
            ADD CONSTRAINT analysis_assets_kind_check
                CHECK (kind IN ('original', 'visualization'))
        """
    )
