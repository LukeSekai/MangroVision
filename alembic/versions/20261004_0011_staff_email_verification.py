"""Private, durable staff verification challenges and request limits."""

from alembic import op

revision = "20261004_0011"
down_revision = "20260929_0010"
branch_labels = None
depends_on = None

DDL = """
CREATE TABLE mangrovision.staff_auth_challenges (
    token_hash TEXT PRIMARY KEY CHECK (length(token_hash) = 64),
    user_id BIGINT REFERENCES mangrovision.users(id) ON DELETE CASCADE,
    purpose TEXT NOT NULL CHECK (purpose IN ('login', 'recovery', 'settings')),
    request_key TEXT NOT NULL,
    code_hash TEXT NOT NULL CHECK (length(code_hash) = 64),
    credential_fingerprint TEXT NOT NULL,
    pending_changes JSONB NOT NULL DEFAULT '{}'::jsonb,
    attempts INTEGER NOT NULL DEFAULT 0 CHECK (attempts BETWEEN 0 AND 5),
    delivered BOOLEAN NOT NULL DEFAULT false,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMPTZ NOT NULL,
    consumed_at TIMESTAMPTZ,
    CHECK (expires_at > created_at)
);
CREATE INDEX idx_staff_auth_challenges_account ON mangrovision.staff_auth_challenges (user_id, created_at DESC);
CREATE INDEX idx_staff_auth_challenges_request ON mangrovision.staff_auth_challenges (request_key, created_at DESC);
CREATE INDEX idx_staff_auth_challenges_expiry ON mangrovision.staff_auth_challenges (expires_at);

CREATE TABLE mangrovision.staff_auth_limits (
    bucket_key TEXT PRIMARY KEY CHECK (length(bucket_key) = 64),
    window_start TIMESTAMPTZ NOT NULL,
    attempts INTEGER NOT NULL CHECK (attempts > 0)
);
CREATE INDEX idx_staff_auth_limits_window ON mangrovision.staff_auth_limits (window_start);

REVOKE ALL ON mangrovision.staff_auth_challenges, mangrovision.staff_auth_limits FROM PUBLIC;
ALTER TABLE mangrovision.staff_auth_challenges ENABLE ROW LEVEL SECURITY;
ALTER TABLE mangrovision.staff_auth_limits ENABLE ROW LEVEL SECURITY;
DO $$ BEGIN
    IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mangrovision') THEN
        GRANT SELECT, INSERT, UPDATE, DELETE ON mangrovision.staff_auth_challenges,
            mangrovision.staff_auth_limits TO mangrovision;
        CREATE POLICY staff_challenges_backend ON mangrovision.staff_auth_challenges
            FOR ALL TO mangrovision USING (true) WITH CHECK (true);
        CREATE POLICY staff_limits_backend ON mangrovision.staff_auth_limits
            FOR ALL TO mangrovision USING (true) WITH CHECK (true);
    END IF;
END $$;
"""


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("SET LOCAL statement_timeout = '60s'")
    # Supabase roles exist in production; ordinary local Postgres may omit them.
    op.execute(DDL)
    op.execute("""DO $$ DECLARE role_name TEXT; BEGIN
        FOR role_name IN SELECT rolname FROM pg_roles WHERE rolname IN ('anon', 'authenticated') LOOP
            EXECUTE format('REVOKE ALL ON mangrovision.staff_auth_challenges, mangrovision.staff_auth_limits FROM %I', role_name);
        END LOOP;
    END $$;""")


def downgrade() -> None:
    op.execute("DROP TABLE mangrovision.staff_auth_challenges, mangrovision.staff_auth_limits")
