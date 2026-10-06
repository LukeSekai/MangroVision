"""The authorized reset operates on temporary tables with real foreign keys."""
import gzip
import hashlib
import json
import os

import pytest
from sqlalchemy import text

import planting_database as db
from scripts.reset_organization import prepare_reset, apply_reset, save_backup
from test_quick_assign_postgres import workspace, create_oton_boundary

pytestmark = pytest.mark.skipif(os.getenv('MANGROVISION_RUN_TEMP_ACCOUNT_TESTS') != '1',
                              reason='Temporary PostgreSQL checks are opt-in.')


@pytest.fixture()
def reset_workspace(workspace):
    for table in ('organization_monitoring_records','monitoring_observations',
                  'monitoring_death_locations','staff_notifications','email_reminders'):
        workspace.execute(text(f'CREATE TEMP TABLE {table} (LIKE mangrovision.{table} INCLUDING ALL)'))
    # Recreate production foreign keys against temporary parents. Never attach
    # test rows to production tables, identities, triggers or users.
    tables = ['organizations','planters','organization_participants','planter_assignments',
              'planter_assignment_points','planting_events','planting_points','project_sites',
              'analyses','users','auth_sessions','point_death_records','monitoring_observations',
              'organization_monitoring_records','monitoring_death_locations','replanting_requests',
              'activity_logs','planting_schedules','staff_notifications','email_reminders']
    fks = workspace.execute(text("""SELECT c.conname,child.relname AS child,pg_get_constraintdef(c.oid) AS definition
        FROM pg_constraint c JOIN pg_class child ON child.oid=c.conrelid
        JOIN pg_class parent ON parent.oid=c.confrelid JOIN pg_namespace ns ON ns.oid=child.relnamespace
        WHERE c.contype='f' AND ns.nspname='mangrovision'
          AND child.relname=ANY(:tables) AND parent.relname=ANY(:tables)"""),{'tables':tables}).mappings().all()
    for fk in fks:
        definition = fk['definition'].replace('mangrovision.','pg_temp.')
        workspace.execute(text(f"ALTER TABLE pg_temp.{fk['child']} ADD CONSTRAINT test_{fk['conname']} {definition}"))
    site = create_oton_boundary(workspace)
    workspace.execute(text("UPDATE organizations SET name='OTON',normalized_name='oton' WHERE id=2"))
    db.create_organization_assignment(2,list(range(96,196)),site_zone_id=site['id'])
    db.create_organization_assignment(1,[1],site_zone_id=1)
    # Reproduce the old orphaned state after the site's boundary was removed.
    workspace.execute(text('UPDATE planter_assignments SET project_site_id=NULL WHERE project_site_id=2'))
    workspace.execute(text('DELETE FROM project_sites WHERE id=2'))
    workspace.execute(text("""
        UPDATE planter_assignment_points SET status='completed',completed_at=CURRENT_TIMESTAMP
          WHERE planting_point_id BETWEEN 96 AND 105 OR planting_point_id=1;
        UPDATE planting_points SET status='planted',planted_at=CURRENT_TIMESTAMP,planted_date=CURRENT_DATE
          WHERE id BETWEEN 96 AND 105 OR id=1;
        INSERT INTO planting_events(id,source_key,planting_point_id,assignment_point_id,assignment_id,planter_id,species,planted_at)
          SELECT pp.id,'reset-test-'||pp.id,pp.id,pap.id,pa.id,pa.planter_id,'Rhizophora',CURRENT_TIMESTAMP
          FROM planting_points pp JOIN planter_assignment_points pap ON pap.planting_point_id=pp.id
          JOIN planter_assignments pa ON pa.id=pap.assignment_id
          WHERE pp.id BETWEEN 96 AND 105 OR pp.id=1;
        INSERT INTO auth_sessions(id,subject_type,subject_id,token_hash,expires_at)
          SELECT id,'planter',id,repeat(id::text,64),CURRENT_TIMESTAMP+INTERVAL '1 day' FROM planters;
        INSERT INTO email_reminders(id,event_key,recipient_email,subject,body,send_on,monitoring_due_date)
          VALUES (1,'monitoring:test','lgu@example.test','Monitoring',E'Due soon\n- OTON\n- NASUGBAN test',CURRENT_DATE,CURRENT_DATE);
    """))
    yield workspace


def test_reset_returns_all_100_points_to_planned_and_preserves_other_organization(reset_workspace,tmp_path):
    conn=reset_workspace
    snapshot,params=prepare_reset(conn,2,'OTON',lock=True)
    assert len(snapshot['planting_points'])==100
    assert len(snapshot['planting_events'])==10
    path,digest=save_backup(snapshot,2,tmp_path)
    with gzip.open(path,'rb') as backup:
        payload=backup.read()
    assert hashlib.sha256(payload).hexdigest()==digest
    assert len(json.loads(payload)['tables']['planting_points'])==100
    locations=conn.execute(text('SELECT id,latitude,longitude,analysis_id FROM planting_points ORDER BY id')).all()
    apply_reset(conn,snapshot,params)
    assert conn.scalar(text('SELECT COUNT(*) FROM organizations WHERE id=2'))==0
    assert conn.scalar(text('SELECT COUNT(*) FROM planters WHERE organization_id=2'))==0
    assert conn.scalar(text("SELECT COUNT(*) FROM planting_points WHERE id BETWEEN 96 AND 195 AND status='planned' AND planted_at IS NULL AND planted_date IS NULL"))==100
    assert conn.execute(text('SELECT id,latitude,longitude,analysis_id FROM planting_points ORDER BY id')).all()==locations
    assert conn.scalar(text("SELECT status FROM planting_points WHERE id=1"))=='planted'
    assert conn.scalar(text('SELECT COUNT(*) FROM planting_events'))==1
    other=conn.scalar(text('SELECT id FROM planters WHERE organization_id=1'))
    assert conn.scalar(text('SELECT COUNT(*) FROM auth_sessions WHERE subject_id=:other'),{'other':other})==1
    assert conn.scalar(text('SELECT COUNT(*) FROM planter_assignment_points'))==1
    assert conn.scalar(text('SELECT COUNT(*) FROM analyses'))==1
    body=conn.scalar(text('SELECT body FROM email_reminders WHERE id=1'))
    assert '- NASUGBAN test' in body
    assert '- OTON' not in body


def test_reset_refuses_mismatched_identity_and_existing_site(reset_workspace):
    with pytest.raises(ValueError,match='must both match'):
        prepare_reset(reset_workspace,2,'CICT',lock=True)
    with pytest.raises(ValueError,match='no project site'):
        prepare_reset(reset_workspace,1,'NASUGBAN test',lock=True)
    assert reset_workspace.scalar(text('SELECT COUNT(*) FROM planting_events'))==11


def test_reset_refuses_other_organization_replacement_chain(reset_workspace):
    reset_workspace.execute(text('UPDATE planting_events SET replaces_event_id=96 WHERE id=1'))
    with pytest.raises(ValueError,match='Other organization records depend'):
        prepare_reset(reset_workspace,2,'OTON',lock=True)
    assert reset_workspace.scalar(text('SELECT COUNT(*) FROM organizations'))==2
    assert reset_workspace.scalar(text('SELECT COUNT(*) FROM planting_events'))==11
