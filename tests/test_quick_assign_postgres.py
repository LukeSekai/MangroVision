"""Quick Assign regressions using temporary PostGIS tables, never live rows."""
import importlib
import os
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
            for table in ('organizations', 'planters', 'organization_participants',
                          'planter_assignments', 'planter_assignment_points',
                          'planting_events', 'point_death_records', 'auth_sessions',
                          'project_sites', 'analyses', 'planting_points', 'users',
                          'map_zones', 'replanting_requests', 'activity_logs', 'planting_schedules'):
                connection.execute(text(f'CREATE TEMP TABLE {table} (LIKE mangrovision.{table} INCLUDING ALL)'))
            connection.execute(text('SET LOCAL search_path = pg_temp, mangrovision, extensions, public'))
            connection.execute(text('CREATE TEMP VIEW site_zones AS SELECT * FROM pg_temp.project_sites'))

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
            monkeypatch.setattr(db, '_is_point_inside_eroded_zone', lambda *_: False)
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
    with pytest.raises(ValueError, match='not inside exactly one project site'):
        db.create_organization_assignment(1, [96], species='Rhizophora', site_zone_id=1)
    # A failed batch must not leave a reservation account behind.
    assert db.list_planters() == []


def test_reserve_then_register_and_open_field_account(workspace, monkeypatch):
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
            'planting_point_ids': ids, 'site_zone_id': 1, 'species': 'Rhizophora',
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
    assert signup.status_code == 200, signup.text
    assert signup.json()['planter']['id'] == reserved['id']
    assert 'password_hash' not in signup.json()['planter']
    assert not db.list_planters()[0]['registration_pending']
    points = db.get_planter_field_points(reserved['id'])
    assert len(points) == 95
    assert {p['planting_point_id'] for p in points} == set(range(1,96))
    shares = Counter(p['participant_slot'] for p in points)
    assert sorted(shares.values()) == [9]*5 + [10]*5
    token = db.create_planter_session(reserved['id'], 1)
    monkeypatch.setattr(auth, 'get_planter_by_session_token', lambda _: db.get_planter_by_session_token(token))
    field = client.get('/api/planter-auth/me/field-points')
    assert field.status_code == 200, field.text
    assert len(field.json()['points']) == 10
    assert {p['participant_slot'] for p in field.json()['points']} == {1}
    with pytest.raises(ValueError, match='already has an account'):
        db.create_planter('duplicate', 'different_user', 'test', organization_id=1)
    with pytest.raises(ValueError, match='already assigned'):
        db.create_organization_assignment(1, [1], species='Rhizophora', site_zone_id=1)


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
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
    route = importlib.import_module('api.routes.planters')
    app = FastAPI()
    app.include_router(route.router, prefix='/api/planters')
    app.dependency_overrides[route._require_lgu_user] = lambda: {'id': 1}
    with TestClient(app) as client:
        wrong = client.post('/api/planters/organizations/2/assignments', json={
            'planting_point_ids': [1], 'site_zone_id': oton['id'], 'species': 'Rhizophora',
        })
        assert wrong.status_code == 400
        assigned = client.post('/api/planters/organizations/2/assignments', json={
            'planting_point_ids': list(range(96,196)), 'site_zone_id': oton['id'], 'species': 'Rhizophora',
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
