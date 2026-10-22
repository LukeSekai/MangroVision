"""Quick Assign regressions using temporary PostGIS tables, never live rows."""
import importlib
import os
from datetime import datetime, timedelta, timezone
from collections import Counter
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

import planting_database as db
from mangrovision_db.compat import CompatConnection, get_engine
from mangrovision_db.organization_accounts import claim_participant_slot

pytestmark = pytest.mark.skipif(
    os.getenv('MANGROVISION_RUN_TEMP_ACCOUNT_TESTS') != '1',
    reason='Temporary PostgreSQL checks are opt-in.',
)


@pytest.fixture()
def workspace(monkeypatch):
    with get_engine().connect() as connection:
        transaction = connection.begin()
        try:
            connection.execute(text("SET LOCAL statement_timeout = '20s'"))
            for table in ('organizations', 'planters', 'organization_participants',
                          'planter_assignments', 'planter_assignment_points',
                          'planting_events', 'point_death_records', 'auth_sessions',
                          'project_sites', 'analyses', 'planting_points', 'users',
                          'map_zones', 'replanting_requests', 'activity_logs', 'planting_schedules',
                          'activity_schedule_emails', 'like_appointment_emails', 'like_appointment_requests', 'activity_access_codes'):
                connection.execute(text(f'CREATE TEMP TABLE {table} (LIKE mangrovision.{table} INCLUDING ALL)'))
            connection.execute(text('ALTER TABLE pg_temp.planter_assignments ADD COLUMN IF NOT EXISTS planting_schedule_id BIGINT'))
            connection.execute(text('SET LOCAL search_path = pg_temp, mangrovision, extensions, public'))
            connection.execute(text('CREATE TEMP VIEW site_zones AS SELECT * FROM pg_temp.project_sites'))
            # Clone the real completion trigger, with every reference isolated.
            trigger_sql = connection.scalar(text(
                "SELECT pg_get_functiondef('mangrovision.maintain_planting_event()'::regprocedure)"))
            connection.execute(text(trigger_sql.replace('mangrovision', 'pg_temp')))
            connection.execute(text('''CREATE TRIGGER test_planting_event AFTER UPDATE OF status
                ON pg_temp.planter_assignment_points FOR EACH ROW
                EXECUTE FUNCTION pg_temp.maintain_planting_event()'''))

            class TempConnection(CompatConnection):
                def __init__(self):
                    super().__init__(connection)
                    self.savepoint = connection.begin_nested()

                def commit(self):
                    self.savepoint.commit()

                def rollback(self):
                    if self.savepoint.is_active:
                        self.savepoint.rollback()

                def close(self):
                    self.rollback()

            monkeypatch.setattr(db, '_get_connection', TempConnection)
            # Advisories are independent of assignment ownership and are empty here.
            monkeypatch.setattr(db, '_load_warning_zone_geometries', lambda: [])
            monkeypatch.setattr(db, '_is_point_inside_eroded_zone', lambda *_, **kwargs: False)
            connection.execute(text("""
                INSERT INTO organizations(id,name,normalized_name,inspection_interval_days)
                    VALUES (1,'NASUGBAN test','nasugban test',14),(2,'Other test','other test',14);
                INSERT INTO users(id,full_name,email,role) VALUES (1,'LGU test','lgu@example.test','lgu');
                INSERT INTO project_sites(id,name,organization_id,polygon_geojson) VALUES
                    (1,'NASUGBAN',1,'{"type":"Polygon","coordinates":[[[122,10],[123,10],[123,11],[122,11],[122,10]]]}'::jsonb);
                INSERT INTO analyses(id,analysis_number,image_name,analyzed_at,project_site_id,species)
                    VALUES (1,1,'quick-assign-test.jpg',CURRENT_TIMESTAMP,1,'Rhizophora');
                INSERT INTO planting_points(id,analysis_id,point_num,latitude,longitude)
                    SELECT value,1,value,10.5,
                        CASE WHEN value <= 95 THEN 122.5 + value * .00001 ELSE 123.5 END
                    FROM generate_series(1,350) value;
            """))
            yield connection
        finally:
            transaction.rollback()


def confirmed_activity(workspace, organization_id=1, site_id=1):
    return workspace.execute(text('''INSERT INTO planting_schedules(organization_id,organization,title,
        appointment_type,start_at,end_at,status,project_site_id)
        SELECT id,name,'Scheduled test planting','tree_planting',CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP+INTERVAL '2 hours','confirmed',:site FROM organizations WHERE id=:org
        RETURNING id'''), {'site': site_id, 'org': organization_id}).scalar_one()


def test_linked_image_does_not_expand_saved_boundary(workspace):
    points = db.list_planter_assignment_map_points()
    inside = [p for p in points if p['source_project_site_id'] == 1]
    assert len(points) == 350
    assert len(inside) == 95
    assert db.get_project_site(1)['properties']['point_count'] == 95
    conn = db._get_connection()
    resolved = db._resolve_point_project_sites(conn, [
        {'id': 1, 'latitude': 10.5, 'longitude': 122.5, 'source_site_id': 1},
        {'id': 2, 'latitude': 10.5, 'longitude': 123.5, 'source_site_id': 1},
        {'id': 3, 'latitude': 10, 'longitude': 122, 'source_site_id': 1},
    ])
    conn.close()
    assert [p['source_site_id'] for p in resolved] == [1, None, 1]
    workspace.execute(text('''INSERT INTO planting_schedules(id,organization_id,organization,title,appointment_type,start_at,end_at,status,project_site_id)
        VALUES(1,1,'NASUGBAN test','Boundary check planting','tree_planting',CURRENT_TIMESTAMP,
            CURRENT_TIMESTAMP+INTERVAL '2 hours','confirmed',1)'''))
    with pytest.raises(ValueError, match='not inside exactly one project site'):
        db.create_organization_assignment(1, [96], species='Rhizophora', site_zone_id=1, planting_schedule_id=1)
    # A failed batch must not leave a reservation account behind.
    assert db.list_planters() == []


def test_activity_without_area_reschedules_until_the_first_assigned_point_is_planted(workspace):
    from mangrovision_db.activity_rules import ActivityConflict
    activity = db.create_planting_schedule(organization='NASUGBAN test', organization_id=1,
        inspection_interval_days=14, title='Planting before area selection',
        start_at='2090-01-01T08:00:00+08:00', end_at='2090-01-01T10:00:00+08:00', status='confirmed')
    assignment_id = db.create_organization_assignments(1, [1, 2], site_zone_id=1,
        planting_schedule_id=activity['id'], assigned_by_user_id=1)[0]
    assert db.get_planting_schedule(activity['id'])['project_site_id'] == 1
    rescheduled = db.update_planting_schedule(activity['id'],
        start_at='2090-01-02T08:00:00+08:00', end_at='2090-01-02T10:00:00+08:00')
    assert rescheduled['date'] == '2090-01-02' and not rescheduled['planting_started']
    batch = workspace.execute(text('SELECT planter_id, assignment_date, planting_schedule_id FROM planter_assignments WHERE id=:id'),
                              {'id': assignment_id}).one()
    assert batch.assignment_date.isoformat() == '2090-01-02' and batch.planting_schedule_id == activity['id']
    point = workspace.execute(text('SELECT id FROM planter_assignment_points WHERE assignment_id=:id ORDER BY id LIMIT 1'),
                              {'id': assignment_id}).scalar_one()
    db.mark_planter_points_completed(batch.planter_id, [point])
    planted = db.get_planting_schedule(activity['id'])
    assert planted['recorded_planting_count'] == 1 and planted['planting_started']
    assert workspace.scalar(text("SELECT count(*) FROM planter_assignment_points WHERE assignment_id=:id AND status='pending'"),
                            {'id': assignment_id}) == 1
    with pytest.raises(ActivityConflict, match='already been recorded'):
        db.update_planting_schedule(activity['id'],
            start_at='2090-01-03T08:00:00+08:00', end_at='2090-01-03T10:00:00+08:00')
    assert db.get_planting_schedule(activity['id']) == planted


def test_failed_assignment_rolls_back_its_activity_area_link(workspace):
    activity = db.create_planting_schedule(organization='NASUGBAN test', organization_id=1,
        inspection_interval_days=14, title='Atomic assignment',
        start_at='2090-01-01T08:00:00+08:00', end_at='2090-01-01T10:00:00+08:00', status='confirmed')
    before = db.get_planting_schedule(activity['id'])
    with pytest.raises(ValueError, match='not inside exactly one project site'):
        db.create_organization_assignments(1, [96], site_zone_id=1, planting_schedule_id=activity['id'])
    assert db.get_planting_schedule(activity['id']) == before
    assert workspace.scalar(text('SELECT count(*) FROM planter_assignments')) == 0
    assert workspace.scalar(text("SELECT count(*) FROM activity_logs WHERE action='schedule.updated'")) == 0


@pytest.mark.parametrize('last_point_method', ['single', 'bulk'])
def test_final_planted_point_completes_only_its_activity_across_all_batches(workspace, last_point_method):
    from mangrovision_db.activity_rules import complete_fully_planted_activity, ActivityConflict
    activity = db.create_planting_schedule(organization='NASUGBAN test', organization_id=1,
        inspection_interval_days=14, title='Complete all batches', start_at='2090-01-01T08:00:00+08:00',
        end_at='2090-01-01T10:00:00+08:00', status='confirmed')
    conn = db._get_connection()
    assert complete_fully_planted_activity(conn, activity['id']) is None
    conn.close()
    batch = db.create_organization_assignments(1, [1, 2, 3], site_zone_id=1,
        planting_schedule_id=activity['id'])[0]
    owner = workspace.scalar(text('SELECT planter_id FROM planter_assignments WHERE id=:id'), {'id': batch})
    # Copy a domain-created batch, including its required species and metadata.
    original = dict(workspace.execute(text('SELECT * FROM planter_assignments WHERE id=:id'), {'id': batch}).mappings().one())
    original.pop('id')
    columns = ', '.join(original)
    values = ', '.join(':' + column for column in original)
    second = workspace.scalar(text(f'INSERT INTO planter_assignments ({columns}) VALUES ({values}) RETURNING id'), original)
    workspace.execute(text('UPDATE planter_assignment_points SET assignment_id=:batch WHERE planting_point_id=3'), {'batch': second})
    ids = dict(workspace.execute(text('SELECT planting_point_id, id FROM planter_assignment_points')).all())
    future = db.create_planting_schedule(organization='NASUGBAN test', organization_id=1,
        inspection_interval_days=14, title='Separate future planting', start_at='2091-01-01T08:00:00+08:00',
        end_at='2091-01-01T10:00:00+08:00', status='confirmed')
    db.create_organization_assignments(1, [4], site_zone_id=1, planting_schedule_id=future['id'])
    result = db.mark_planter_points_completed(owner, [ids[1], ids[2]])
    assert result['completed_activity_ids'] == []
    assert db.get_planting_schedule(activity['id'])['status'] == 'confirmed'
    email_count = workspace.scalar(text('SELECT count(*) FROM activity_schedule_emails'))
    result = (db.update_assignment_point_status(ids[3], 'completed') if last_point_method == 'single'
              else db.mark_planter_points_completed(owner, [ids[3]]))
    assert result['completed_activity_ids'] == [activity['id']]
    assert db.get_planting_schedule(activity['id'])['status'] == 'completed'
    assert db.get_planting_schedule(future['id'])['status'] == 'confirmed'
    assert workspace.scalar(text('SELECT count(*) FROM activity_schedule_emails')) == email_count
    conn = db._get_connection()
    assert complete_fully_planted_activity(conn, activity['id']) is None
    conn.close()
    assert workspace.scalar(text("SELECT count(*) FROM activity_logs WHERE details->>'source'='all_assigned_points_planted'")) == 1
    with pytest.raises(ActivityConflict, match='cannot be reopened'):
        db.update_assignment_point_status(ids[3], 'pending')
    assert workspace.scalar(text('SELECT status FROM planter_assignment_points WHERE id=:id'), {'id': ids[3]}) == 'completed'


@pytest.mark.parametrize('remaining_status', ['pending', 'skipped'])
def test_unplanted_points_prevent_automatic_activity_completion(workspace, remaining_status):
    activity = db.create_planting_schedule(organization='NASUGBAN test', organization_id=1,
        inspection_interval_days=14, title='Unfinished activity', start_at='2090-01-01T08:00:00+08:00',
        end_at='2090-01-01T10:00:00+08:00', status='confirmed')
    batch = db.create_organization_assignments(1, [1, 2], site_zone_id=1,
        planting_schedule_id=activity['id'])[0]
    ids = workspace.execute(text('SELECT id FROM planter_assignment_points WHERE assignment_id=:id ORDER BY id'), {'id': batch}).scalars().all()
    db.update_assignment_point_status(ids[0], 'completed')
    if remaining_status == 'skipped':
        db.update_assignment_point_status(ids[1], 'skipped', 'Unsafe footing')
        assert workspace.scalar(text('SELECT status FROM planter_assignments WHERE id=:id'), {'id': batch}) == 'completed'
    assert db.get_planting_schedule(activity['id'])['status'] == 'confirmed'


def test_assignment_links_to_activity_then_field_planting_locks_it(workspace):
    from mangrovision_db.activity_rules import ActivityConflict
    start = datetime.now(timezone(timedelta(hours=8))) + timedelta(days=3)
    activity = db.create_planting_schedule(organization='NASUGBAN test', organization_id=1,
        inspection_interval_days=14, title='Demo activity', start_at=start,
        end_at=start + timedelta(hours=2), status='confirmed')
    db.update_planting_schedule(activity['id'], project_site_id=1)
    assignment_id = db.create_organization_assignments(1, [1, 2], site_zone_id=1, planting_schedule_id=activity['id'])[0]
    linked = workspace.execute(text('SELECT planting_schedule_id, assignment_date, planter_id FROM planter_assignments WHERE id = :id'), {'id': assignment_id}).one()
    assert linked.planting_schedule_id == activity['id'] and linked.assignment_date == start.date()
    point = workspace.execute(text('SELECT id FROM planter_assignment_points WHERE assignment_id = :id ORDER BY id LIMIT 1'), {'id': assignment_id}).scalar_one()
    db.mark_planter_points_completed(linked.planter_id, [point])
    assert db.get_planting_schedule(activity['id'])['planting_started']
    with pytest.raises(ActivityConflict, match='already been recorded'):
        db.update_planting_schedule(activity['id'], title='Must remain unchanged')
    assert db.update_planting_schedule(activity['id'], notes='Corrected note')['notes'] == 'Corrected note'


def test_reservation_survives_retired_registration_endpoint(workspace, monkeypatch):
    activity = confirmed_activity(workspace)
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
    auth = importlib.import_module('api.routes.planter_auth')
    planters = importlib.import_module('api.routes.planters')
    app = FastAPI()
    app.include_router(auth.router, prefix='/api/planter-auth')
    app.include_router(planters.router, prefix='/api/planters')
    app.dependency_overrides[planters._require_lgu_user] = lambda: {'id': 1}
    client = TestClient(app)
    for ids in (list(range(1,51)), list(range(51,96))):
        response = client.post('/api/planters/organizations/1/assignments', json={
            'planting_point_ids': ids, 'site_zone_id': 1, 'species': 'Rhizophora', 'planting_schedule_id': activity,
        })
        assert response.status_code == 200, response.text
    reserved = db.list_planters()[0]
    assert reserved['registration_pending']
    assert reserved['username'] is None
    assert reserved['pending_points'] == 95
    assert 'password_hash' not in reserved
    with pytest.raises(ValueError, match='inactive'):
        claim_participant_slot(reserved['id'], 'pending-device-key-123')
    signup = client.post('/api/planter-auth/register', json={
        'organization_id': 1, 'username': 'nasugban_test', 'password': 'test-password',
        'participant_count': 10, 'device_key': 'nasugban-device-key-01',
    })
    assert signup.status_code == 410
    after = db.list_planters()[0]
    assert after['id'] == reserved['id'] and after['registration_pending']
    assert after['pending_points'] == 95
    points = db.get_planter_field_points(reserved['id'])
    assert {point['planting_point_id'] for point in points} == set(range(1, 96))
    with pytest.raises(ValueError, match='already assigned'):
        db.create_organization_assignment(1, [1], species='Rhizophora', site_zone_id=1, planting_schedule_id=activity)


def test_unlinked_points_use_only_unambiguous_polygons(workspace):
    workspace.execute(text('UPDATE analyses SET project_site_id = NULL'))
    assert len([p for p in db.list_planter_assignment_map_points() if p['source_site_id'] == 1]) == 95
    workspace.execute(text("""INSERT INTO project_sites(id,name,organization_id,polygon_geojson)
        SELECT 2,'Overlapping site',2,polygon_geojson FROM project_sites WHERE id=1"""))
    assert all(p['source_site_id'] is None for p in db.list_planter_assignment_map_points())


def create_oton_boundary(workspace):
    """One saved photo spans NASUGBAN's 95 points and OTON's next 100."""
    workspace.execute(text("""
        UPDATE organizations SET name = 'OTON test', normalized_name = 'oton test' WHERE id = 2;
        UPDATE planting_points SET longitude = 124.5 WHERE id > 195;
        ALTER TABLE pg_temp.project_sites ALTER COLUMN id RESTART WITH 2;
    """))
    return db.create_project_site(
        name='OTON', organization_id=2,
        geometry={'type': 'Polygon', 'coordinates': [
            [[123,10],[124,10],[124,11],[123,11],[123,10]],
        ]},
    )


def test_new_boundary_recognizes_points_from_image_linked_to_another_site(workspace):
    assert all(p['source_site_id'] is None for p in db.list_planter_assignment_map_points() if p['id'] > 95)
    oton = create_oton_boundary(workspace)
    assert oton['properties']['point_count'] == 100
    assert workspace.scalar(text('SELECT project_site_id FROM analyses WHERE id=1')) == 1
    points = db.list_planter_assignment_map_points()
    oton_points = [p for p in points if p['source_project_site_id'] == oton['id']]
    assert {p['id'] for p in oton_points} == set(range(96,196))
    assert {p['source_organization_id'] for p in oton_points} == {2}
    assert {p['planting_status'] for p in oton_points} == {'planned'}
    assert all(p['assigned_planter_id'] is None for p in oton_points)
    assert len([p for p in points if p['source_site_id'] == 1]) == 95
    assert db.get_project_site(1)['properties']['point_count'] == 95
    conn = db._get_connection()
    try:
        resolved = db._resolve_point_project_sites(conn, [
            {'id': 96, 'latitude': 10.5, 'longitude': 123.5, 'source_site_id': 1},
        ])
    finally:
        conn.close()
    assert resolved[0]['source_site_id'] == oton['id']
    assert resolved[0]['source_organization_id'] == 2


def test_quick_assign_accepts_oton_points_and_rejects_other_organization(workspace, monkeypatch):
    oton = create_oton_boundary(workspace)
    activity = confirmed_activity(workspace, organization_id=2, site_id=oton['id'])
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
    route = importlib.import_module('api.routes.planters')
    app = FastAPI()
    app.include_router(route.router, prefix='/api/planters')
    app.dependency_overrides[route._require_lgu_user] = lambda: {'id': 1}
    with TestClient(app) as client:
        wrong = client.post('/api/planters/organizations/2/assignments', json={
            'planting_point_ids': [1], 'site_zone_id': oton['id'], 'species': 'Rhizophora', 'planting_schedule_id': activity,
        })
        assert wrong.status_code == 400
        assigned = client.post('/api/planters/organizations/2/assignments', json={
            'planting_point_ids': list(range(96,196)), 'site_zone_id': oton['id'], 'species': 'Rhizophora', 'planting_schedule_id': activity,
        })
        assert assigned.status_code == 200, assigned.text
    account_id = db.create_planter('OTON', 'oton_test', 'test', organization_id=2, participant_count=10)
    assert Counter(p['participant_slot'] for p in db.get_planter_field_points(account_id)) == {slot:10 for slot in range(1,11)}
    points = db.list_planter_assignment_map_points()
    assert len([p for p in points if p['source_site_id'] == oton['id'] and p['assigned_planter_id'] == account_id]) == 100
    assert all(p['assigned_planter_id'] is None for p in points if p['source_site_id'] == 1)


def test_overlaps_preserve_covering_link_and_leave_other_overlaps_unresolved(workspace):
    oton = create_oton_boundary(workspace)
    workspace.execute(text("""INSERT INTO project_sites(id,name,organization_id,polygon_geojson)
        SELECT 3,'Overlapping NASUGBAN',2,polygon_geojson FROM project_sites WHERE id=1"""))
    assert len([p for p in db.list_planter_assignment_map_points() if p['source_site_id'] == 1]) == 95
    workspace.execute(text("""INSERT INTO project_sites(id,name,organization_id,polygon_geojson)
        SELECT 4,'Overlapping OTON',1,polygon_geojson FROM project_sites WHERE id=2"""))
    points = db.list_planter_assignment_map_points()
    assert all(p['source_site_id'] is None for p in points if 96 <= p['id'] <= 195)
    assert db.get_project_site(oton['id'])['properties']['point_count'] == 0
    conn = db._get_connection()
    try:
        assert db._resolve_point_project_sites(conn, [
            {'id': 96, 'latitude': 10.5, 'longitude': 123.5, 'source_site_id': 1},
        ])[0]['source_site_id'] is None
    finally:
        conn.close()


def create_cict_species_selection(workspace):
    """Reproduce CICT's 99 Bungalon and 4 Rhizophora points in one site."""
    workspace.execute(text("""
        UPDATE organizations SET name='CICT', normalized_name='cict' WHERE id=1;
        UPDATE project_sites SET name='CICT' WHERE id=1;
        UPDATE analyses SET species='bungalon', planting_distance_m=1 WHERE id=1;
        INSERT INTO analyses(id,analysis_number,image_name,analyzed_at,species,planting_distance_m)
            VALUES (2,2,'second-species-test.jpg',CURRENT_TIMESTAMP,'rhizophora',2);
        UPDATE planting_points SET longitude=122.5 + id * .00001 WHERE id<=103;
        UPDATE planting_points SET analysis_id=2 WHERE id BETWEEN 100 AND 103;
    """))
    return confirmed_activity(workspace)


def quick_assign_client(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
    route = importlib.import_module('api.routes.planters')
    app = FastAPI()
    app.include_router(route.router, prefix='/api/planters')
    app.dependency_overrides[route._require_lgu_user] = lambda: {'id': 1}
    return TestClient(app)


@pytest.mark.parametrize('count,registered', [(100, False), (103, True)])
def test_cict_mixed_species_assigns_every_selected_point_and_balances_devices(workspace, monkeypatch, count, registered):
    activity = create_cict_species_selection(workspace)
    if registered:
        db.create_planter('CICT', 'cict_test', 'test', organization_id=1, participant_count=10)
    with quick_assign_client(monkeypatch) as client:
        assigned = client.post('/api/planters/organizations/1/assignments', json={
            'planting_point_ids': list(range(1, count+1)), 'site_zone_id': 1, 'planting_schedule_id': activity,
        })
    assert assigned.status_code == 200, assigned.text
    payload = assigned.json()
    assert len(payload['assignment_ids']) == 2
    assert payload['assignment_id'] == payload['assignment_ids'][0]
    species_counts = workspace.execute(text("""
        SELECT pa.species, COUNT(*) FROM planter_assignments pa
        JOIN planter_assignment_points pap ON pap.assignment_id=pa.id GROUP BY pa.species
    """)).all()
    assert dict(species_counts) == {'Bungalon': 99, 'Rhizophora': count-99}
    if not registered:
        db.create_planter('CICT', 'cict_test', 'test', organization_id=1, participant_count=10)
    account = db.list_planters()[0]
    points = db.get_planter_field_points(account['id'])
    assert {p['id'] for p in points} == set(range(1, count+1))
    assert all(p['species'] == ('bungalon' if p['id'] < 100 else 'rhizophora') for p in points)
    shares = Counter(p['participant_slot'] for p in points)
    assert len(shares) == 10
    assert max(shares.values()) - min(shares.values()) <= 1
    assert sum(shares.values()) == count


def test_single_species_cict_assignment_still_accepts_recorded_bungalon(workspace, monkeypatch):
    activity = create_cict_species_selection(workspace)
    with quick_assign_client(monkeypatch) as client:
        response = client.post('/api/planters/organizations/1/assignments', json={
            'planting_point_ids': list(range(1,100)), 'site_zone_id': 1, 'species': 'BUNGALON', 'planting_schedule_id': activity,
        })
    assert response.status_code == 200, response.text
    assert len(response.json()['assignment_ids']) == 1
    assert workspace.scalar(text('SELECT species FROM planter_assignments')) == 'Bungalon'
    assert workspace.scalar(text('SELECT COUNT(*) FROM planter_assignment_points')) == 99


def test_genuinely_missing_species_blocks_the_entire_selection(workspace, monkeypatch):
    create_cict_species_selection(workspace)
    workspace.execute(text('UPDATE analyses SET species=NULL WHERE id=2'))
    with quick_assign_client(monkeypatch) as client:
        response = client.post('/api/planters/organizations/1/assignments', json={
            'planting_point_ids': list(range(1,104)), 'site_zone_id': 1,
        })
    assert response.status_code == 400
    assert 'Point #100 has no valid species on its image analysis' in response.json()['detail']
    for table in ('planters', 'organization_participants', 'planter_assignments', 'planter_assignment_points', 'activity_logs'):
        assert workspace.scalar(text(f'SELECT COUNT(*) FROM {table}')) == 0


@pytest.mark.parametrize('ids,species', [([1], 'Rhizophora'), ([1,100], 'Bungalon')])
def test_requested_species_cannot_relabel_recorded_points(workspace, monkeypatch, ids, species):
    create_cict_species_selection(workspace)
    with quick_assign_client(monkeypatch) as client:
        response = client.post('/api/planters/organizations/1/assignments', json={
            'planting_point_ids': ids, 'site_zone_id': 1, 'species': species,
        })
    assert response.status_code == 400
    assert 'do not all have the requested species' in response.json()['detail']
    assert workspace.scalar(text('SELECT COUNT(*) FROM planter_assignments')) == 0
    assert db.list_planters() == []


@pytest.mark.parametrize('failure', ['already_assigned', 'outside_site'])
def test_later_species_failure_rolls_back_earlier_species_batch(workspace, monkeypatch, failure):
    activity = create_cict_species_selection(workspace)
    if failure == 'already_assigned':
        db.create_organization_assignment(1, [100], site_zone_id=1, planting_schedule_id=activity)
        expected_assignments = 1
        error = 'already assigned'
    else:
        workspace.execute(text('UPDATE planting_points SET longitude=124.5 WHERE id=100'))
        expected_assignments = 0
        error = 'not inside exactly one project site'
    with quick_assign_client(monkeypatch) as client:
        response = client.post('/api/planters/organizations/1/assignments', json={
            'planting_point_ids': [1,100], 'site_zone_id': 1, 'planting_schedule_id': activity,
        })
    assert response.status_code == 400
    assert error in response.json()['detail']
    assert workspace.scalar(text('SELECT COUNT(*) FROM planter_assignments')) == expected_assignments
    assert workspace.scalar(text('SELECT COUNT(*) FROM activity_logs')) == expected_assignments
    assert workspace.scalar(text('SELECT COUNT(*) FROM planter_assignment_points WHERE planting_point_id=1')) == 0
    if not expected_assignments:
        assert db.list_planters() == []
