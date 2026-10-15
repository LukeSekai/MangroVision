"""Real field updates, switching, and progress use TEMP clones with outer rollback."""
import importlib
import importlib.util
import os
from collections import Counter
from pathlib import Path

import pytest
from cryptography.fernet import Fernet
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

import planting_database as db
from mangrovision_db import group_access
from mangrovision_db.compat import CompatConnection, get_engine

pytestmark = pytest.mark.skipif(os.getenv('MANGROVISION_RUN_TEMP_ACCOUNT_TESTS') != '1', reason='Temporary PostgreSQL checks are opt-in.')


@pytest.fixture()
def workspace(monkeypatch):
    try:
        connection = get_engine().connect()
    except Exception as error:
        raise pytest.fail.Exception(f'Temporary database check could not connect ({type(error).__name__}); details suppressed.', pytrace=False) from None
    transaction = connection.begin()
    try:
        connection.execute(text("SET LOCAL statement_timeout = '20s'"))
        for table in ('organizations', 'planters', 'organization_participants', 'planter_assignments',
            'planter_assignment_points', 'planting_events', 'point_death_records', 'auth_sessions',
            'project_sites', 'analyses', 'planting_points', 'users', 'map_zones', 'replanting_requests',
            'activity_logs', 'planting_schedules', 'monitoring_observations', 'planter_auth_limits',
            'planter_access_codes'):
            connection.execute(text(f'CREATE TEMP TABLE {table} (LIKE mangrovision.{table} INCLUDING ALL)'))
        connection.execute(text('SET LOCAL search_path = pg_temp, mangrovision, extensions, public'))
        connection.execute(text('CREATE TEMP VIEW site_zones AS SELECT * FROM pg_temp.project_sites'))
        # Run the new migration only against session-local clones.
        migration_path = Path(__file__).resolve().parents[1] / 'alembic/versions/20261015_0019_shared_activity_access.py'
        spec = importlib.util.spec_from_file_location('group_migration', migration_path)
        migration = importlib.util.module_from_spec(spec); spec.loader.exec_module(migration)
        statements = []; monkeypatch.setattr(migration.op, 'execute', statements.append)
        migration.upgrade()
        for statement in statements:
            connection.exec_driver_sql(statement.replace('mangrovision.', 'pg_temp.').replace('CREATE TABLE pg_temp.', 'CREATE TEMP TABLE pg_temp.'))
        connection.exec_driver_sql('''CREATE TEMP TABLE activity_schedule_emails(schedule_id BIGINT, status TEXT);
            CREATE TEMP TABLE like_appointment_requests(id BIGINT, schedule_id BIGINT);
            CREATE TEMP TABLE like_appointment_emails(request_id BIGINT, status TEXT);''')
        trigger = connection.scalar(text("SELECT pg_get_functiondef('mangrovision.maintain_planting_event()'::regprocedure)"))
        connection.execute(text(trigger.replace('mangrovision', 'pg_temp')))
        connection.execute(text('''CREATE TRIGGER test_planting_event AFTER UPDATE OF status
            ON pg_temp.planter_assignment_points FOR EACH ROW EXECUTE FUNCTION pg_temp.maintain_planting_event()'''))

        class TempConnection(CompatConnection):
            def __init__(self):
                super().__init__(connection); self.savepoint = connection.begin_nested()
            def commit(self):
                self.savepoint.commit(); self.savepoint = connection.begin_nested()
            def rollback(self):
                if self.savepoint.is_active: self.savepoint.rollback()
            def close(self):
                self.rollback()

        monkeypatch.setattr(db, '_get_connection', TempConnection)
        monkeypatch.setattr(group_access, 'get_connection', TempConnection)
        monkeypatch.setattr(db, '_load_warning_zone_geometries', lambda: [])
        monkeypatch.setattr(db, '_is_point_inside_eroded_zone', lambda *_: False)
        monkeypatch.setenv('LIKE_EMAIL_ENCRYPTION_KEY', Fernet.generate_key().decode())
        connection.execute(text('''
            INSERT INTO organizations(id,name,normalized_name,inspection_interval_days) VALUES(1,'Field test','field test',14);
            INSERT INTO project_sites(id,name,organization_id,polygon_geojson) VALUES(1,'Test site',1,
                '{"type":"Polygon","coordinates":[[[122,10],[123,10],[123,11],[122,11],[122,10]]]}'::jsonb);
            INSERT INTO analyses(id,analysis_number,image_name,analyzed_at,project_site_id,species)
                VALUES(1,1,'fictional.jpg',CURRENT_TIMESTAMP,1,'Rhizophora');
            INSERT INTO planting_points(id,analysis_id,point_num,latitude,longitude)
                SELECT n,1,n,10.5,122.5+n*.00001 FROM generate_series(1,100) n;
            INSERT INTO planting_schedules(id,organization,organization_id,title,start_at,end_at,expected_planters,status,appointment_type)
                VALUES(1,'Field test',1,'Tree planting',clock_timestamp()-interval '1 hour',
                    clock_timestamp()+interval '2 hours',10,'confirmed','tree_planting');
        '''))
        db.create_organization_assignment(1, list(range(1, 101)), site_zone_id=1, species='Rhizophora', planting_schedule_id=1)
        code = group_access.manage_activity_code(1, action='issue')['access']['code']
        monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
        auth = importlib.import_module('api.routes.planter_auth')
        security = importlib.import_module('api.security')
        app = FastAPI(); app.add_middleware(security.SessionSecurityMiddleware)
        app.include_router(auth.router, prefix='/api/planter-auth')
        yield connection, app, code
    finally:
        transaction.rollback(); connection.close()


def post(client, path, payload=None):
    return client.post(f'/api/planter-auth{path}', json=payload,
        headers={'X-CSRF-Token': client.cookies.get('mv_csrf', '')})


def login(client, code, person=1, browser=None, mode='return', **extra):
    return post(client, '/login', {'access_code': code, 'username': f'Planter{person}',
        'browser_key': f'{browser or person:048x}', 'mode': mode, **extra})


def test_ten_planters_record_progress_logout_switch_and_expiry_preserve_exact_points(workspace):
    connection, app, code = workspace
    origin = 'https://field-progress.example.test'
    clients = [TestClient(app, base_url=origin, headers={'Origin': origin}) for _ in range(11)]
    try:
        for number, client in enumerate(clients[:10], 1):
            response = login(client, code, number, mode='first')
            assert response.status_code == 200, response.text
            assert response.json()['planter']['participant_slot'] == number
        original = []
        for number, client in enumerate(clients[:10], 1):
            response = client.get('/api/planter-auth/me/field-points')
            assert response.status_code == 200, response.text
            points = response.json()['points']
            assert len(points) == 10 and {point['participant_slot'] for point in points} == {number}
            original.append({point['assignment_point_id'] for point in points})
        assert len(set.union(*original)) == 100
        assert login(clients[10], code, 11, mode='first').status_code == 409
        first = clients[0]; owned = sorted(original[0])
        response = first.patch(f'/api/planter-auth/me/points/{owned[0]}/status', json={'status': 'completed'},
            headers={'X-CSRF-Token': first.cookies.get('mv_csrf')})
        assert response.status_code == 200, response.text
        response = post(first, '/me/points/mark-all-completed', {'assignment_point_ids': owned[1:3]})
        assert response.status_code == 200, response.text
        assert connection.scalar(text('SELECT count(*) FROM planting_events')) == 3
        assert post(clients[1], '/me/points/mark-all-completed', {'assignment_point_ids': [owned[0]]}).status_code == 403
        for _ in range(2):
            assert post(first, '/logout').status_code == 200
            assert login(first, code).status_code == 200
            points = first.get('/api/planter-auth/me/field-points').json()['points']
            assert {point['assignment_point_id'] for point in points} == original[0]
            assert Counter(point['assignment_status'] for point in points) == {'completed': 3, 'pending': 7}
        replacement = clients[10]
        conflict = login(replacement, code, 1, browser=11)
        assert conflict.status_code == 409 and conflict.json()['detail']['kind'] == 'session_in_use'
        assert login(replacement, code, 1, browser=11, switch_token=conflict.json()['detail']['switch_token']).status_code == 200
        assert first.get('/api/planter-auth/me/field-points').status_code == 401
        assert post(first, '/me/points/mark-all-completed', {'assignment_point_ids': owned[3:4]}).status_code == 401
        points = replacement.get('/api/planter-auth/me/field-points').json()['points']
        assert {point['assignment_point_id'] for point in points} == original[0]
        assert Counter(point['assignment_status'] for point in points) == {'completed': 3, 'pending': 7}
        assert post(first, '/logout').status_code == 200
        assert replacement.get('/api/planter-auth/session').status_code == 200
        connection.execute(text("UPDATE planting_schedules SET start_at=clock_timestamp()-interval '4 hours',end_at=clock_timestamp()-interval '1 hour' WHERE id=1"))
        assert replacement.get('/api/planter-auth/session').status_code == 401
        assert post(replacement, '/me/points/mark-all-completed', {'assignment_point_ids': owned[3:4]}).status_code == 401
        assert connection.scalar(text('SELECT count(*) FROM planting_events')) == 3
        assert Counter(connection.execute(text('SELECT participant_slot FROM planter_assignment_points')).scalars()) == {n: 10 for n in range(1,11)}
    finally:
        for client in clients: client.close()
