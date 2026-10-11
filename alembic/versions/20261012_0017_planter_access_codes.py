"""Individual, persistent planter codes and durable sign-in limits."""
from alembic import op

revision = '20261012_0017'
down_revision = '20261011_0016'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("SET LOCAL statement_timeout = '60s'")
    op.execute("""
        CREATE TABLE mangrovision.planter_access_codes (
            planter_id BIGINT NOT NULL,
            participant_slot INTEGER NOT NULL,
            code_digest TEXT NOT NULL UNIQUE CHECK (length(code_digest) = 64),
            encrypted_code TEXT NOT NULL,
            revoked_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (planter_id, participant_slot),
            FOREIGN KEY (planter_id, participant_slot)
                REFERENCES mangrovision.organization_participants(planter_id, slot) ON DELETE CASCADE
        );
        CREATE TABLE mangrovision.planter_auth_limits (
            bucket_key TEXT PRIMARY KEY,
            window_start TIMESTAMPTZ NOT NULL,
            attempts INTEGER NOT NULL CHECK (attempts > 0)
        );
        ALTER TABLE mangrovision.planter_access_codes ENABLE ROW LEVEL SECURITY;
        ALTER TABLE mangrovision.planter_auth_limits ENABLE ROW LEVEL SECURITY;
        REVOKE ALL ON mangrovision.planter_access_codes, mangrovision.planter_auth_limits FROM PUBLIC;
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mangrovision') THEN
                GRANT SELECT, INSERT, UPDATE, DELETE ON mangrovision.planter_access_codes,
                    mangrovision.planter_auth_limits TO mangrovision;
                CREATE POLICY planter_codes_backend ON mangrovision.planter_access_codes TO mangrovision
                    USING (true) WITH CHECK (true);
                CREATE POLICY planter_limits_backend ON mangrovision.planter_auth_limits TO mangrovision
                    USING (true) WITH CHECK (true);
            END IF;
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
                REVOKE ALL ON mangrovision.planter_access_codes, mangrovision.planter_auth_limits FROM anon;
            END IF;
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
                REVOKE ALL ON mangrovision.planter_access_codes, mangrovision.planter_auth_limits FROM authenticated;
            END IF;
        END $$;
    """)


def downgrade():
    op.execute('DROP TABLE mangrovision.planter_auth_limits')
    op.execute('DROP TABLE mangrovision.planter_access_codes')
