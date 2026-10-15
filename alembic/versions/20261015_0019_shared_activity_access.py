"""Activity group codes, permanent usernames, and remembered browsers."""
from alembic import op

revision = '20261015_0019'
down_revision = '20261014_0018'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("SET LOCAL statement_timeout = '60s'")
    op.execute("""
        ALTER TABLE mangrovision.organization_participants
            ADD COLUMN participant_username TEXT,
            ADD COLUMN username_key TEXT,
            ADD COLUMN current_session_hash TEXT,
            ADD CONSTRAINT participant_username_pair CHECK (
                (participant_username IS NULL AND username_key IS NULL) OR
                (participant_username IS NOT NULL AND username_key IS NOT NULL
                 AND username_key = lower(btrim(participant_username))
                 AND username_key ~ '^[a-z0-9][a-z0-9._-]{2,31}$'));
        CREATE UNIQUE INDEX participant_username_unique
            ON mangrovision.organization_participants(planter_id, username_key)
            WHERE username_key IS NOT NULL;
        CREATE TABLE mangrovision.activity_access_codes (
            schedule_id BIGINT PRIMARY KEY REFERENCES mangrovision.planting_schedules(id) ON DELETE CASCADE,
            planter_id BIGINT NOT NULL REFERENCES mangrovision.planters(id) ON DELETE CASCADE,
            code_digest TEXT NOT NULL UNIQUE CHECK (length(code_digest) = 64),
            encrypted_code TEXT NOT NULL,
            revoked_at TIMESTAMPTZ,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
        );
        CREATE INDEX activity_codes_planter_idx ON mangrovision.activity_access_codes(planter_id);
        CREATE TABLE mangrovision.participant_browser_bindings (
            planter_id BIGINT NOT NULL,
            browser_key_hash TEXT NOT NULL CHECK (length(browser_key_hash) = 64),
            participant_slot INTEGER NOT NULL,
            created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
            PRIMARY KEY (planter_id, browser_key_hash),
            FOREIGN KEY (planter_id, participant_slot)
                REFERENCES mangrovision.organization_participants(planter_id, slot) ON DELETE CASCADE
        );
        CREATE INDEX participant_browsers_slot_idx
            ON mangrovision.participant_browser_bindings(planter_id, participant_slot);
        ALTER TABLE mangrovision.auth_sessions
            ADD COLUMN planting_schedule_id BIGINT REFERENCES mangrovision.planting_schedules(id) ON DELETE CASCADE,
            ADD COLUMN activity_code_digest TEXT,
            ADD COLUMN browser_key_hash TEXT;
        CREATE INDEX field_session_activity_idx
            ON mangrovision.auth_sessions(planting_schedule_id, subject_id, participant_slot)
            WHERE planting_schedule_id IS NOT NULL;
        ALTER TABLE mangrovision.activity_access_codes ENABLE ROW LEVEL SECURITY;
        ALTER TABLE mangrovision.participant_browser_bindings ENABLE ROW LEVEL SECURITY;
        REVOKE ALL ON mangrovision.activity_access_codes, mangrovision.participant_browser_bindings FROM PUBLIC;
        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mangrovision') THEN
                GRANT SELECT, INSERT, UPDATE, DELETE ON mangrovision.activity_access_codes,
                    mangrovision.participant_browser_bindings TO mangrovision;
                CREATE POLICY activity_codes_backend ON mangrovision.activity_access_codes TO mangrovision
                    USING (true) WITH CHECK (true);
                CREATE POLICY participant_browsers_backend ON mangrovision.participant_browser_bindings TO mangrovision
                    USING (true) WITH CHECK (true);
            END IF;
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'anon') THEN
                REVOKE ALL ON mangrovision.activity_access_codes, mangrovision.participant_browser_bindings FROM anon;
            END IF;
            IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'authenticated') THEN
                REVOKE ALL ON mangrovision.activity_access_codes, mangrovision.participant_browser_bindings FROM authenticated;
            END IF;
        END $$;
    """)


def downgrade():
    raise RuntimeError('Shared access retains participant identities. Restore a verified backup to downgrade safely.')
