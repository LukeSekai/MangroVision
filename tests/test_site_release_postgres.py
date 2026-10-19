"""Release and reuse land in isolated PostgreSQL tables; always roll back."""
import json
import os
from datetime import timedelta

import pytest
from sqlalchemy import text
from shapely.geometry import Point, shape

import planting_database as db
from mangrovision_db.compat import CompatConnection, get_engine
from mangrovision_db.health_summary import summarize_current_health
from mangrovision_db.monitoring_locations import release_dead_locations_batch, list_release_candidates
from mangrovision_db.site_actions import available_released_areas, release_dead_locations
from mangrovision_db.planting_performance import summarize_planting_performance
from mangrovision_db.point_status import current_point_status

pytestmark = pytest.mark.skipif(os.getenv('MANGROVISION_RUN_TEMP_ACCOUNT_TESTS') != '1',
                               reason='Rollback-only PostgreSQL release checks are opt-in.')


@pytest.fixture
def workspace(monkeypatch):
    tables = ['users', 'organizations', 'planters', 'project_sites', 'analyses', 'planting_points',
              'planter_assignments', 'planter_assignment_points', 'planting_events', 'point_death_records',
              'replanting_requests', 'map_zones', 'warning_zones', 'activity_logs', 'planting_schedules',
              'project_site_area_releases', 'organization_participants', 'activity_schedule_emails',
              'like_appointment_emails', 'like_appointment_requests']
    try:
        connection = get_engine().connect()
    except Exception:
        pytest.fail('Temporary PostgreSQL connection unavailable; details suppressed.', pytrace=False)
    with connection:
        transaction = connection.begin()
        try:
            connection.exec_driver_sql("SET LOCAL statement_timeout = '20s'")
            for table in tables:
                connection.exec_driver_sql(f'CREATE TEMP TABLE {table} (LIKE mangrovision.{table} INCLUDING ALL)')
            connection.exec_driver_sql('SET LOCAL search_path = pg_temp, mangrovision, extensions, public')
            connection.exec_driver_sql('CREATE TEMP VIEW site_zones AS SELECT * FROM pg_temp.project_sites')
            for table in tables:
                namespace = connection.scalar(text('''SELECT n.nspname FROM pg_class c JOIN pg_namespace n ON n.oid = c.relnamespace
                    WHERE c.oid = to_regclass(:name)'''), {'name': table})
                assert namespace.startswith('pg_temp_')

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
            connection.exec_driver_sql('''
                INSERT INTO users(id,full_name,email,role) VALUES (1,'LGU test officer','officer@example.invalid','planner');
                INSERT INTO organizations(id,name,normalized_name,inspection_interval_days) VALUES (1,'Coastal team','coastal team',14);
                INSERT INTO planters(id,full_name,organization_id) VALUES (1,'Original team',1);
                INSERT INTO project_sites(id,name,organization_id,polygon_geojson) VALUES
                  (1,'Coastal site',1,'{"type":"Polygon","coordinates":[[[122.625,10.779],[122.626,10.779],[122.626,10.781],[122.625,10.781],[122.625,10.779]]]}'::jsonb);
                INSERT INTO analyses(id,analysis_number,image_name,analyzed_at,project_site_id,species)
                  VALUES (1,1,'fixture.jpg',CURRENT_TIMESTAMP,1,'Bungalon');
                INSERT INTO planter_assignments(id,planter_id,title,assignment_date,project_site_id,status,species)
                  VALUES (1,1,'Original planting',CURRENT_DATE-40,1,'completed','Bungalon');
                INSERT INTO planting_points(id,analysis_id,point_num,latitude,longitude,status,planted_at,death_at) VALUES
                  (1,1,1,10.780,122.6251,'planted',CURRENT_TIMESTAMP-INTERVAL '40 days',CURRENT_TIMESTAMP-INTERVAL '2 days'),
                  (2,1,2,10.780,122.6252,'planted',CURRENT_TIMESTAMP-INTERVAL '40 days',CURRENT_TIMESTAMP-INTERVAL '2 days'),
                  (3,1,3,10.780,122.6253,'planted',CURRENT_TIMESTAMP-INTERVAL '40 days',NULL);
                INSERT INTO planter_assignment_points(assignment_id,planting_point_id,sequence_num,status,completed_at)
                  SELECT 1,id,id,'completed',planted_at FROM planting_points;
                INSERT INTO planting_events(source_key,planting_point_id,assignment_id,project_site_id,planter_id,species,planted_at,
                    point_num,latitude,longitude,closed_at,closure_reason)
                  SELECT 'original-'||id,id,1,1,1,'Bungalon',planted_at,point_num,latitude,longitude,death_at,
                    CASE WHEN death_at IS NOT NULL THEN 'recorded_death' END FROM planting_points ORDER BY id;
                INSERT INTO point_death_records(planting_point_id,planting_event_id,death_at,reason_category)
                  SELECT pp.id,pe.id,pp.death_at,'waves' FROM planting_points pp JOIN planting_events pe ON pe.planting_point_id=pp.id WHERE pp.death_at IS NOT NULL;
            ''')
            connection.exec_driver_sql('ALTER TABLE pg_temp.planter_assignments ALTER COLUMN id RESTART WITH 1000')
            for name in ('maintain_planting_event', 'link_replacement_event', 'finish_replacement_event'):
                definition = connection.scalar(text(f"SELECT pg_get_functiondef('mangrovision.{name}()'::regprocedure)"))
                connection.exec_driver_sql(definition.replace('mangrovision', 'pg_temp'))
            connection.exec_driver_sql('''
                CREATE TRIGGER maintain_event AFTER UPDATE OF status ON pg_temp.planter_assignment_points
                  FOR EACH ROW EXECUTE FUNCTION pg_temp.maintain_planting_event();
                CREATE TRIGGER link_replacement BEFORE INSERT ON pg_temp.planting_events
                  FOR EACH ROW EXECUTE FUNCTION pg_temp.link_replacement_event();
                CREATE TRIGGER finish_replacement AFTER INSERT ON pg_temp.planting_events
                  FOR EACH ROW EXECUTE FUNCTION pg_temp.finish_replacement_event();
            ''')
            monkeypatch.setattr(db, '_load_warning_zone_geometries', lambda: [])
            yield connection
        finally:
            transaction.rollback()


def snapshot():
    connection = db._get_connection()
    try:
        points = db._load_current_dashboard_points(connection)
        events = db._load_dashboard_events(connection)
        now = db._manila_now() + timedelta(seconds=1)
        return points, summarize_current_health(points, events, now), summarize_planting_performance(
            events, now - timedelta(days=100), now, now, points=points)
    finally:
        connection.close()


def release(ids=(1, 2), versions=None):
    return release_dead_locations_batch(list(ids), versions or {str(value): 0 for value in ids}, 1)


def test_selected_release_frees_land_and_keeps_every_original_outcome(workspace):
    before_events = workspace.execute(text('SELECT * FROM planting_events ORDER BY id')).all()
    before_deaths = workspace.execute(text('SELECT * FROM point_death_records ORDER BY id')).all()
    assert release((1,))['released_count'] == 1
    points, _, performance = snapshot()
    assert {point['id']: current_point_status(point) for point in points} == {1: 'planned', 2: 'dead', 3: 'planted'}
    assert {point['id']: point['site_id'] for point in points} == {1: None, 2: 1, 3: 1}
    assert performance['summary']['total'] == 3 and performance['summary']['dead'] == 2
    assert performance['summary']['survival_rate_pct'] == 33.33
    assert db._project_site_planting_history(CompatConnection(workspace))[1]['survival_rate_pct'] == 33.33
    assert workspace.execute(text('SELECT * FROM planting_events ORDER BY id')).all() == before_events
    assert workspace.execute(text('SELECT * FROM point_death_records ORDER BY id')).all() == before_deaths
    public = {point['id']: point for point in db.list_planter_assignment_map_points()}
    assert public[1]['map_status'] == 'planned' and public[1]['source_project_site_id'] is None
    assert public[1]['assignment_id'] is None
    boundary = shape(workspace.scalar(text('SELECT polygon_geojson FROM project_sites WHERE id=1')))
    assert not boundary.covers(Point(122.6251, 10.780))
    assert boundary.covers(Point(122.6252, 10.780)) and boundary.covers(Point(122.6253, 10.780))
    assert shape(available_released_areas()['features'][0]['geometry']).covers(Point(122.6251, 10.780))
    assert workspace.scalar(text('SELECT COUNT(*) FROM planter_assignment_points WHERE released_at IS NOT NULL')) == 1
    assert next(row for row in list_release_candidates() if row['planting_event_id'] == 1)['release_status'] == 'released'


def test_another_organization_can_claim_assign_and_plant_a_fresh_cycle(workspace):
    release()
    released_area = available_released_areas()['features'][0]['geometry']
    workspace.execute(text("""INSERT INTO organizations(id,name,normalized_name,inspection_interval_days)
        VALUES(2,'Second organization','second organization',14);
        INSERT INTO planters(id,full_name,organization_id) VALUES(2,'Second team',2);
        INSERT INTO organization_participants(planter_id,slot) VALUES(2,1);
        INSERT INTO planting_schedules(id,organization_id,organization,title,appointment_type,start_at,end_at,status,project_site_id)
        VALUES(1,2,'Second organization','Replacement activity','tree_planting',CURRENT_TIMESTAMP-INTERVAL '1 hour',CURRENT_TIMESTAMP+INTERVAL '1 hour','in_progress',2);"""))
    workspace.execute(text('INSERT INTO project_sites(id,name,organization_id,polygon_geojson) VALUES(2,\'New site\',2,CAST(:geometry AS jsonb))'), {'geometry': json.dumps(released_area)})
    assignment = db.create_planter_assignment(2, [1, 2], assigned_by_user_id=1,
        site_zone_id=2, species='Bungalon', planting_schedule_id=1)
    assert workspace.scalar(text('SELECT assignment_id FROM replanting_requests WHERE planting_event_id=1')) == assignment
    assert workspace.scalar(text('SELECT COUNT(*) FROM planting_events')) == 3
    point_id = workspace.scalar(text('SELECT id FROM planter_assignment_points WHERE assignment_id=:assignment AND planting_point_id=1'), {'assignment': assignment})
    db.update_assignment_point_status(point_id, 'completed')
    conn = db._get_connection()
    try:
        events = db._load_dashboard_events(conn)
        points = db._load_current_dashboard_points(conn)
    finally:
        conn.close()
    now = db._manila_now() + timedelta(seconds=1)
    original = summarize_planting_performance(events, now-timedelta(days=100), now, now, site_id=1, points=points)
    new = summarize_planting_performance(events, now-timedelta(days=100), now, now, site_id=2, points=points)
    assert original['summary']['total'] == 3 and original['summary']['dead'] == 2
    assert original['summary']['survival_rate_pct'] == 33.33
    assert new['summary']['total'] == 1 and new['summary']['dead'] == 0
    assert new['batches'][0]['organization_name'] == 'Second organization'
    assert new['batches'][0]['planting_type'] == 'replacement'
    replacement = workspace.execute(text('SELECT * FROM planting_events WHERE replaces_event_id=1')).mappings().one()
    assert replacement['project_site_id'] == 2 and replacement['planter_id'] == 2
    assert workspace.scalar(text('SELECT project_site_id FROM planting_events WHERE id=1')) == 1
    assert workspace.scalar(text('SELECT COUNT(*) FROM point_death_records')) == 2


@pytest.mark.parametrize('case', ['stale', 'living', 'erosion', 'removed'])
def test_invalid_batch_changes_neither_land_nor_history(workspace, case):
    ids, versions = [1, 2], {'1': 0, '2': 0}
    if case == 'stale':
        versions['2'] = 99
    elif case == 'living':
        ids = [1, 3]
        versions = {'1': 0, '3': 0}
    elif case == 'removed':
        workspace.exec_driver_sql('UPDATE planting_points SET deleted_at=CURRENT_TIMESTAMP WHERE id=2')
    else:
        workspace.exec_driver_sql('''INSERT INTO map_zones(zone_type,name,polygon_geojson)
            VALUES ('eroded','Erosion','{"type":"Polygon","coordinates":[[[122.62515,10.779],[122.626,10.779],[122.626,10.781],[122.62515,10.781],[122.62515,10.779]]]}'::jsonb)''')
    before = workspace.scalar(text('SELECT polygon_geojson FROM project_sites WHERE id=1'))
    with pytest.raises(ValueError):
        release(ids, versions)
    assert workspace.scalar(text('SELECT polygon_geojson FROM project_sites WHERE id=1')) == before
    assert workspace.scalar(text('SELECT COUNT(*) FROM replanting_requests')) == 0
    assert workspace.scalar(text('SELECT COUNT(*) FROM project_site_area_releases')) == 0
    assert workspace.scalar(text('SELECT COUNT(*) FROM planter_assignment_points WHERE released_at IS NOT NULL')) == 0
    assert workspace.scalar(text('SELECT COUNT(*) FROM point_death_records')) == 2


def test_existing_unplanted_reservations_convert_without_creating_a_planting(workspace):
    workspace.exec_driver_sql('''INSERT INTO replanting_requests(planting_event_id,approved_by) VALUES(1,1);
        UPDATE planter_assignment_points SET released_at=CURRENT_TIMESTAMP WHERE planting_point_id=1;
        UPDATE planting_points SET status='planned',death_at=NULL,planted_at=NULL,planted_date=NULL WHERE id=1;''')
    with pytest.raises(ValueError, match='confirmed dead'):
        release((1,), {'1': 1})
    assert release_dead_locations([1], {'1': 1}, 1, prepared=True)['released_count'] == 1
    assert workspace.scalar(text('SELECT COUNT(*) FROM planting_events')) == 3
    assert workspace.scalar(text('SELECT COUNT(*) FROM point_death_records')) == 2
    assert workspace.scalar(text('SELECT COUNT(*) FROM project_site_area_releases')) == 1


def test_repeated_release_cannot_reset_a_new_cycle(workspace):
    release((1,))
    with pytest.raises(ValueError):
        release((1,))
    assert workspace.scalar(text('SELECT COUNT(*) FROM project_site_area_releases')) == 1
