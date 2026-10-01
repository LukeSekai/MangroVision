"""One organization login, persistent device slots, and balanced point allocations.

Keep legacy accounts as inactive audit records. The oldest active account becomes
the shared login; assignment and planting history move to that account.
"""
from alembic import op

revision = '20260912_0006'
down_revision = '20260912_0005'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("SET LOCAL statement_timeout = '60s'")
    op.execute("""
        ALTER TABLE mangrovision.planters
          ADD COLUMN participant_count INTEGER NOT NULL DEFAULT 1 CHECK (participant_count BETWEEN 1 AND 10000),
          ADD COLUMN merged_into_planter_id BIGINT REFERENCES mangrovision.planters(id);
        CREATE INDEX idx_planters_merged_into ON mangrovision.planters(merged_into_planter_id);
        CREATE TEMP TABLE organization_account_merge ON COMMIT DROP AS
          SELECT id, organization_id,
            first_value(id) OVER (PARTITION BY organization_id ORDER BY (status = 'active') DESC, id) AS account_id,
            count(*) OVER (PARTITION BY organization_id) AS participants
          FROM mangrovision.planters WHERE organization_id IS NOT NULL;
        UPDATE mangrovision.planters p SET
          participant_count = m.participants,
          merged_into_planter_id = CASE WHEN p.id <> m.account_id THEN m.account_id END,
          status = CASE WHEN p.id <> m.account_id THEN 'inactive' ELSE p.status END
          FROM organization_account_merge m WHERE m.id = p.id;
        UPDATE mangrovision.planters p SET full_name = o.name
          FROM mangrovision.organizations o
          WHERE p.organization_id = o.id AND p.merged_into_planter_id IS NULL;
        UPDATE mangrovision.planter_assignments a SET planter_id = m.account_id
          FROM organization_account_merge m WHERE a.planter_id = m.id;
        UPDATE mangrovision.planting_events e SET planter_id = m.account_id
          FROM organization_account_merge m WHERE e.planter_id = m.id;
        UPDATE mangrovision.point_death_records d SET planter_id = m.account_id
          FROM organization_account_merge m WHERE d.planter_id = m.id;
        UPDATE mangrovision.auth_sessions SET revoked_at = CURRENT_TIMESTAMP
          WHERE subject_type = 'planter' AND revoked_at IS NULL;
        CREATE UNIQUE INDEX uq_planters_organization_account
          ON mangrovision.planters(organization_id)
          WHERE merged_into_planter_id IS NULL;
        CREATE TABLE mangrovision.organization_participants (
          planter_id BIGINT NOT NULL REFERENCES mangrovision.planters(id) ON DELETE CASCADE,
          slot INTEGER NOT NULL CHECK (slot > 0),
          device_key_hash TEXT,
          PRIMARY KEY (planter_id, slot),
          UNIQUE (planter_id, device_key_hash)
        );
        ALTER TABLE mangrovision.organization_participants ENABLE ROW LEVEL SECURITY;
        REVOKE ALL ON mangrovision.organization_participants FROM PUBLIC;
        DO $$ BEGIN
          IF EXISTS (SELECT 1 FROM pg_roles WHERE rolname = 'mangrovision') THEN
            GRANT SELECT, INSERT, UPDATE, DELETE ON mangrovision.organization_participants TO mangrovision;
            CREATE POLICY organization_participants_backend ON mangrovision.organization_participants
              TO mangrovision USING (true) WITH CHECK (true);
          END IF;
        END $$;
        INSERT INTO mangrovision.organization_participants(planter_id, slot)
          SELECT p.id, generate_series(1, p.participant_count)
          FROM mangrovision.planters p WHERE merged_into_planter_id IS NULL;
        ALTER TABLE mangrovision.auth_sessions ADD COLUMN participant_slot INTEGER CHECK (participant_slot > 0);
        ALTER TABLE mangrovision.planter_assignment_points
          ADD COLUMN participant_slot INTEGER NOT NULL DEFAULT 1 CHECK (participant_slot > 0);
        WITH allocation AS (
          SELECT pap.id, 1 + ((row_number() OVER (PARTITION BY pa.planter_id ORDER BY pa.id, pap.sequence_num) - 1)
            % p.participant_count)::integer AS slot
          FROM mangrovision.planter_assignment_points pap
          JOIN mangrovision.planter_assignments pa ON pa.id = pap.assignment_id
          JOIN mangrovision.planters p ON p.id = pa.planter_id
        ) UPDATE mangrovision.planter_assignment_points pap SET participant_slot = a.slot
          FROM allocation a WHERE a.id = pap.id;
        CREATE INDEX idx_assignment_points_participant
          ON mangrovision.planter_assignment_points(assignment_id, participant_slot);
    """)


def downgrade():
    raise RuntimeError('Organization accounts consolidate planting history. Restore a pre-migration backup to undo the conversion.')
