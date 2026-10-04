"""Use the established Supabase cloud sender without exposing Vault broadly."""

from alembic import op

revision = '20261004_0012'
down_revision = '20261004_0011'
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("SET LOCAL statement_timeout = '60s'")
    op.execute("""
        ALTER TABLE mangrovision.staff_auth_challenges ADD COLUMN email_claimed_at TIMESTAMPTZ;
        CREATE FUNCTION mangrovision.staff_email_transport() RETURNS JSONB
        LANGUAGE plpgsql SECURITY DEFINER SET search_path = '' AS $$
        DECLARE project_url TEXT; email_token TEXT;
        BEGIN
            IF pg_catalog.to_regnamespace('vault') IS NULL THEN RETURN NULL; END IF;
            SELECT decrypted_secret INTO project_url FROM vault.decrypted_secrets
                WHERE name = 'mangrovision_project_url';
            SELECT decrypted_secret INTO email_token FROM vault.decrypted_secrets
                WHERE name = 'mangrovision_staff_email_token';
            IF project_url IS NULL OR email_token IS NULL THEN RETURN NULL; END IF;
            RETURN pg_catalog.jsonb_build_object('url', rtrim(project_url, '/') || '/functions/v1/staff-verification', 'token', email_token);
        END $$;
        REVOKE ALL ON FUNCTION mangrovision.staff_email_transport() FROM PUBLIC;
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mangrovision') THEN
                GRANT EXECUTE ON FUNCTION mangrovision.staff_email_transport() TO mangrovision;
            END IF;
        END $$;
    """)


def downgrade() -> None:
    op.execute('DROP FUNCTION mangrovision.staff_email_transport()')
    op.execute('ALTER TABLE mangrovision.staff_auth_challenges DROP COLUMN email_claimed_at')
