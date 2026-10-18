"""Exercise LGU replanting against isolated, rollback-only PostgreSQL tables."""
import os
from datetime import timedelta

import pytest
from sqlalchemy import text

import planting_database as db
from mangrovision_db.compat import CompatConnection, get_engine
from mangrovision_db.health_summary import summarize_current_health
from mangrovision_db.lgu_replanting import pending_lgu_requests, record_lgu_replanting
from mangrovision_db.monitoring_locations import approve_replanting_batch, list_replanting
from mangrovision_db.planting_performance import summarize_planting_performance
from mangrovision_db.point_status import current_point_status

pytestmark = pytest.mark.skipif(os.getenv('MANGROVISION_RUN_LGU_REPLANTING_TESTS') != '1',
                               reason='Rollback-only LGU replacement checks are opt-in.')


@pytest.fixture
def workspace(monkeypatch):
    tables = ['users', 'organizations', 'planters', 'project_sites', 'analyses', 'planting_points',
              'planter_assignments', 'planter_assignment_points', 'planting_events', 'point_death_records',
              'replanting_requests', 'map_zones', 'warning_zones', 'activity_logs']
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
                CREATE FUNCTION pg_temp.legacy_lgu_insert_trigger() RETURNS trigger LANGUAGE plpgsql AS $fn$
                  BEGIN NEW.replaces_event_id = NULL; RETURN NEW; END $fn$;
                CREATE TRIGGER legacy_link BEFORE INSERT ON pg_temp.planting_events
                  FOR EACH ROW EXECUTE FUNCTION pg_temp.legacy_lgu_insert_trigger();
            ''')
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


def approve(ids=(1, 2)):
    return approve_replanting_batch(list(ids), {str(value): 0 for value in ids}, 1)


def plant(ids=(1,), **overrides):
    return record_lgu_replanting(list(ids), {str(value): 1 for value in ids},
        project_site_id=1, planted_date=db._manila_now().date().isoformat(), species='Bungalon',
        staff_user_id=1, **overrides)


def test_approval_is_an_lgu_assignment_and_does_not_improve_survival(workspace):
    approve()
    points, current, performance = snapshot()
    assert {point['id']: current_point_status(point) for point in points} == {1: 'assigned', 2: 'assigned', 3: 'planted'}
    assert current['summary']['planted'] == 1 and current['summary']['dead'] == 0
    assert performance['summary']['total'] == 3 and performance['summary']['dead'] == 2
    assert performance['summary']['survival_rate_pct'] == 33.33
    assert performance['replanting']['awaiting_review'] == 0
    assert performance['replanting']['ready_lgu'] == 2
    public = db.list_planter_assignment_map_points()
    assert {point['id']: point['map_status'] for point in public} == {1: 'assigned', 2: 'assigned', 3: 'planted'}
    assert all(point['assignment_id'] is None for point in public if point.get('lgu_replanting_pending'))
    queue = list_replanting()
    assert all(row['can_record_replanting'] and row['replanting_status'] == 'ready_lgu' for row in queue)


def test_partial_replanting_updates_map_queue_staff_attribution_and_batch_history(workspace):
    approve()
    result = plant()
    assert result['replanted_points'] == 1
    points, current, performance = snapshot()
    assert {point['id']: current_point_status(point) for point in points} == {1: 'planted', 2: 'assigned', 3: 'planted'}
    assert performance['summary']['total'] == 4 and performance['summary']['dead'] == 2
    assert performance['replanting']['ready_lgu'] == 1
    assert performance['replanting']['replacements_completed'] == 1
    original = next(row for row in performance['batches'] if row['planting_type'] == 'original')
    assert original['survival_rate_pct'] == 33.33 and original['organization_name'] == 'Coastal team'
    replacement = next(row for row in performance['batches'] if row['planting_type'] == 'replacement')
    assert replacement['organization_name'] == 'LGU' and replacement['survival_rate_pct'] == 100
    assert workspace.scalar(text('SELECT count(*) FROM point_death_records')) == 2
    saved = workspace.execute(text('SELECT * FROM planting_events WHERE replaces_event_id=1')).mappings().one()
    assert saved['planted_by_user_id'] == 1 and saved['planter_id'] is None and saved['assignment_id'] is None
    assert workspace.scalar(text('SELECT count(*) FROM activity_logs WHERE action=\'seedling.replanted\'')) == 1
    now = db._manila_now()
    operations = db.get_dashboard_operations(date_from=(now-timedelta(days=100)).date().isoformat(), date_to=now.date().isoformat())
    assert operations['summary']['pending'] == 1
    assert operations['summary']['completed'] == 4
    assert next(row for row in operations['workload'] if row.get('owner_type') == 'lgu')['pending'] == 1
    with pytest.raises(ValueError):
        plant()
    assert workspace.scalar(text('SELECT count(*) FROM planting_events')) == 4


def test_unapproved_deaths_and_living_points_cannot_be_recorded_as_replacements(workspace):
    with pytest.raises(ValueError, match='ready for LGU'):
        plant()
    with pytest.raises(ValueError):
        approve((3,))
    assert workspace.scalar(text('SELECT count(*) FROM replanting_requests')) == 0
    assert workspace.scalar(text('SELECT count(*) FROM planting_events')) == 3


def test_stale_version_rolls_back_every_selected_location(workspace):
    approve()
    with pytest.raises(ValueError, match='changed'):
        record_lgu_replanting([1, 2], {'1': 1, '2': 0}, project_site_id=1,
            planted_date=db._manila_now().date().isoformat(), species='Bungalon', staff_user_id=1)
    assert workspace.scalar(text('SELECT count(*) FROM planting_events')) == 3
    assert workspace.scalar(text('SELECT count(*) FROM replanting_requests WHERE replacement_event_id IS NOT NULL')) == 0


def test_erosion_after_approval_blocks_planting_and_shows_unavailable(workspace):
    approve()
    workspace.exec_driver_sql('''INSERT INTO map_zones(zone_type,name,polygon_geojson)
        VALUES ('eroded','Temporary erosion','{"type":"Polygon","coordinates":[[[122.625,10.779],[122.626,10.779],[122.626,10.781],[122.625,10.781],[122.625,10.779]]]}'::jsonb)''')
    with pytest.raises(ValueError, match='erosion'):
        plant()
    points, _, performance = snapshot()
    assert current_point_status(next(point for point in points if point['id'] == 1)) == 'unavailable'
    assert performance['replanting']['ready_lgu'] == 0
    assert not any(row['can_record_replanting'] for row in list_replanting())


def test_invalid_date_and_wrong_site_make_no_new_plantings(workspace):
    approve()
    with pytest.raises(ValueError, match='before'):
        record_lgu_replanting([1], {'1': 1}, project_site_id=1,
            planted_date=(db._manila_now()-timedelta(days=3)).date().isoformat(), species='Bungalon', staff_user_id=1)
    with pytest.raises(ValueError, match='active project site'):
        record_lgu_replanting([1], {'1': 1}, project_site_id=999,
            planted_date=db._manila_now().date().isoformat(), species='Bungalon', staff_user_id=1)
    assert workspace.scalar(text('SELECT count(*) FROM planting_events')) == 3


def test_approved_lgu_work_cannot_be_reassigned_to_an_organization(workspace):
    approve()
    connection = db._get_connection()
    try:
        with pytest.raises(ValueError, match='reserved for LGU'):
            db._create_planter_assignment(1, [1], species='Bungalon', connection=connection)
        assert len(pending_lgu_requests(connection, [1, 2])) == 2
    finally:
        connection.close()


def test_a_dead_lgu_replacement_keeps_its_own_attribution_and_can_start_another_cycle(workspace):
    approve()
    replacement_id = plant()['replacement_event_ids'][0]
    db.mark_planting_point_dead(1, 'waves')
    death = workspace.execute(text('SELECT * FROM point_death_records WHERE planting_event_id=:event'),
                              {'event': replacement_id}).mappings().one()
    assert death['planter_id'] is None and death['assignment_id'] is None
    assert death['planter_name'] == 'LGU'
    approve_replanting_batch([replacement_id], {str(replacement_id): 0}, 1)
    _, _, performance = snapshot()
    assert performance['summary']['dead'] == 3
    assert performance['replanting']['ready_lgu'] == 2
    assert performance['replanting']['replacements_completed'] == 1
