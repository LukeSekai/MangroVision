"""Require a recorded species on assignments and planting events.

Revision ID: 20260929_0010
Revises: 20260928_0009
"""

from alembic import op


revision = "20260929_0010"
down_revision = "20260928_0009"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.execute("SET LOCAL lock_timeout = '5s'")
    op.execute("SET LOCAL statement_timeout = '60s'")
    op.execute("""
        UPDATE mangrovision.planting_events pe
        SET species = COALESCE(
            (SELECT NULLIF(BTRIM(pa.species), '')
             FROM mangrovision.planter_assignments pa WHERE pa.id = pe.assignment_id),
            (SELECT NULLIF(BTRIM(a.species), '')
             FROM mangrovision.planting_points pp
             JOIN mangrovision.analyses a ON a.id = pp.analysis_id
             WHERE pp.id = pe.planting_point_id)
        )
        WHERE NULLIF(BTRIM(pe.species), '') IS NULL
          AND (pe.assignment_id IS NOT NULL OR pe.planting_point_id IS NOT NULL);

        -- This isolated email test had no point or analysis from which to
        -- derive a species. Bungalon is explicit demonstration data here.
        UPDATE mangrovision.planting_events
        SET species = 'Bungalon'
        WHERE source = 'email_test'
          AND source_key = 'email-test:2026-09-29:bceed0b78ce8'
          AND NULLIF(BTRIM(species), '') IS NULL;

        DO $$ BEGIN
            IF EXISTS (SELECT 1 FROM mangrovision.planter_assignments
                       WHERE NULLIF(BTRIM(species), '') IS NULL)
                OR EXISTS (SELECT 1 FROM mangrovision.planting_events
                           WHERE NULLIF(BTRIM(species), '') IS NULL) THEN
                RAISE EXCEPTION 'Resolve missing planting species before applying this migration';
            END IF;
        END $$;

        ALTER TABLE mangrovision.planter_assignments
            ADD CONSTRAINT ck_planter_assignments_species_present
            CHECK (species IS NOT NULL AND BTRIM(species) <> '') NOT VALID;
        ALTER TABLE mangrovision.planting_events
            ADD CONSTRAINT ck_planting_events_species_present
            CHECK (species IS NOT NULL AND BTRIM(species) <> '') NOT VALID;
        ALTER TABLE mangrovision.planter_assignments
            VALIDATE CONSTRAINT ck_planter_assignments_species_present;
        ALTER TABLE mangrovision.planting_events
            VALIDATE CONSTRAINT ck_planting_events_species_present;
    """)


def downgrade() -> None:
    op.execute("""
        ALTER TABLE mangrovision.planting_events
            DROP CONSTRAINT ck_planting_events_species_present;
        ALTER TABLE mangrovision.planter_assignments
            DROP CONSTRAINT ck_planter_assignments_species_present;
    """)
