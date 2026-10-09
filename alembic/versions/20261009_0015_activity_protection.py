"""Link activity planting batches and distinguish revised booking emails."""

from alembic import op

revision = '20261009_0015'
down_revision = '20261007_0014'
branch_labels = None
depends_on = None


def upgrade():
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("SET LOCAL statement_timeout = '60s'")
    op.execute("""
        ALTER TABLE mangrovision.planter_assignments ADD COLUMN planting_schedule_id BIGINT
            REFERENCES mangrovision.planting_schedules(id) ON DELETE RESTRICT;
        CREATE INDEX idx_assignments_activity ON mangrovision.planter_assignments(planting_schedule_id);
        ALTER TABLE mangrovision.like_appointment_emails ADD COLUMN message_kind TEXT NOT NULL
            DEFAULT 'confirmation' CHECK (message_kind IN ('confirmation', 'update', 'cancellation'));
        WITH matches AS (
            SELECT pa.id, MIN(ps.id) AS schedule_id
            FROM mangrovision.planter_assignments pa
            JOIN mangrovision.planters pl ON pl.id = pa.planter_id
            JOIN mangrovision.planting_schedules ps ON ps.organization_id = pl.organization_id
                AND ps.project_site_id = pa.project_site_id
                AND ps.appointment_type = 'tree_planting'
                AND (ps.start_at AT TIME ZONE 'Asia/Manila')::date = pa.assignment_date
            GROUP BY pa.id HAVING COUNT(*) = 1
        )
        UPDATE mangrovision.planter_assignments pa SET planting_schedule_id = matches.schedule_id
            FROM matches WHERE pa.id = matches.id;
    """)
    op.execute("""
        CREATE FUNCTION mangrovision.lock_activity_on_planting() RETURNS trigger
        LANGUAGE plpgsql AS $$
        DECLARE activity_id BIGINT; activity_status TEXT;
        BEGIN
            SELECT planting_schedule_id INTO activity_id FROM mangrovision.planter_assignments
                WHERE id = NEW.assignment_id;
            IF activity_id IS NOT NULL THEN
                SELECT status INTO activity_status FROM mangrovision.planting_schedules
                    WHERE id = activity_id FOR UPDATE;
                IF activity_status NOT IN ('confirmed', 'in_progress', 'completed') THEN
                    RAISE EXCEPTION 'The linked planting activity is not confirmed.' USING ERRCODE = '23514';
                END IF;
            END IF;
            RETURN NEW;
        END $$;
        CREATE TRIGGER trg_lock_activity_on_planting BEFORE INSERT ON mangrovision.planting_events
            FOR EACH ROW EXECUTE FUNCTION mangrovision.lock_activity_on_planting();
    """)


def downgrade():
    op.execute("DROP TRIGGER trg_lock_activity_on_planting ON mangrovision.planting_events")
    op.execute("DROP FUNCTION mangrovision.lock_activity_on_planting()")
    op.execute("ALTER TABLE mangrovision.like_appointment_emails DROP COLUMN message_kind")
    op.execute("ALTER TABLE mangrovision.planter_assignments DROP COLUMN planting_schedule_id")
