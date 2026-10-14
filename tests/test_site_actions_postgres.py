"""Use temporary PostgreSQL tables only; never release or delete live sites."""
import os

import pytest
from sqlalchemy import text
from shapely.geometry import Point, shape

import planting_database as db
from mangrovision_db.compat import CompatConnection, get_engine
from mangrovision_db.site_actions import apply_site_action, available_released_areas, preview_site_actions

pytestmark = pytest.mark.skipif(os.getenv('MANGROVISION_RUN_TEMP_ACCOUNT_TESTS') != '1', reason='Temporary Postgres checks are opt-in.')


@pytest.fixture()
def workspace(monkeypatch):
    tables = ['project_sites', 'analyses', 'planting_points', 'planter_assignments', 'planter_assignment_points',
              'planting_events', 'point_death_records', 'planting_schedules', 'activity_logs',
              'replanting_requests', 'project_site_area_releases']
    try:
        isolated_connection = get_engine().connect()
    except Exception:
        pytest.fail('Temporary database connection unavailable; connection details suppressed.', pytrace=False)
    with isolated_connection as connection:
        transaction = connection.begin()
        try:
            connection.execute(text("SET LOCAL statement_timeout = '20s'"))
            for table in tables:
                connection.execute(text(f'CREATE TEMP TABLE {table} (LIKE mangrovision.{table} INCLUDING ALL)'))
            connection.execute(text('SET LOCAL search_path = pg_temp, mangrovision, extensions, public'))
            connection.execute(text('CREATE TEMP VIEW site_zones AS SELECT * FROM pg_temp.project_sites'))
            # Prove all mutation targets resolve to temporary objects.
            for table in tables:
                namespace = connection.scalar(text('SELECT n.nspname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.oid=to_regclass(:name)'), {'name': table})
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
            connection.execute(text("""
                INSERT INTO project_sites(id,name,organization_id,polygon_geojson) VALUES
                  (1,'Original site',1,'{"type":"Polygon","coordinates":[[[122.62505,10.77995],[122.62525,10.77995],[122.62525,10.78005],[122.62505,10.78005],[122.62505,10.77995]]]}'::jsonb);
                INSERT INTO analyses(id,analysis_number,image_name,analyzed_at,project_site_id,species)
                  VALUES (1,1,'fixture.jpg',CURRENT_TIMESTAMP,1,'Rhizophora');
                INSERT INTO planting_points(id,analysis_id,point_num,latitude,longitude,status,planted_at,death_at) VALUES
                  (1,1,1,10.78,122.62510,'planted',CURRENT_TIMESTAMP-INTERVAL '20 days',NULL),
                  (2,1,2,10.78,122.62512,'planted',CURRENT_TIMESTAMP-INTERVAL '20 days',CURRENT_TIMESTAMP-INTERVAL '1 day'),
                  (3,1,3,10.78,122.62515,'planned',NULL,NULL),
                  (4,1,4,10.78,122.62520,'planned',NULL,NULL);
                INSERT INTO planting_events(id,source_key,planting_point_id,project_site_id,species,planted_at,closed_at,closure_reason) VALUES
                  (1,'test-live',1,1,'Rhizophora',CURRENT_TIMESTAMP-INTERVAL '20 days',NULL,NULL),
                  (2,'test-dead',2,1,'Rhizophora',CURRENT_TIMESTAMP-INTERVAL '20 days',CURRENT_TIMESTAMP-INTERVAL '1 day','recorded_death');
                INSERT INTO point_death_records(id,planting_point_id,planting_event_id,death_at,reason_category)
                  VALUES (1,2,2,CURRENT_TIMESTAMP-INTERVAL '1 day','unknown');
                INSERT INTO planter_assignments(id,planter_id,title,assignment_date,project_site_id,species)
                  VALUES (1,1,'Fixture assignment',CURRENT_DATE,1,'Rhizophora');
                INSERT INTO planter_assignment_points(id,assignment_id,planting_point_id,sequence_num,status,completed_at) VALUES
                  (1,1,1,1,'completed',CURRENT_TIMESTAMP-INTERVAL '20 days'),
                  (2,1,2,2,'completed',CURRENT_TIMESTAMP-INTERVAL '20 days'),
                  (3,1,3,3,'pending',NULL);
            """))
            yield connection
        finally:
            transaction.rollback()


def test_release_shrinks_area_releases_only_nonliving_and_preserves_history(workspace):
    before = preview_site_actions(1)
    assert before['counts'] == {'living': 1, 'dead': 1, 'unplanted': 2, 'release': 3, 'historical_planted': 2, 'historical_dead': 1}
    assert before['can_release'] and not before['can_delete']
    kept = shape(before['after_geometry'])
    assert kept.covers(Point(122.62510, 10.78))
    assert not any(kept.covers(Point(lon, 10.78)) for lon in [122.62512, 122.62515, 122.62520])
    event_rows = list(workspace.execute(text('SELECT id,project_site_id,planted_at,closed_at FROM planting_events ORDER BY id')))
    apply_site_action(1, 'release', before['version'], 1)
    assert list(workspace.execute(text('SELECT id,project_site_id,planted_at,closed_at FROM planting_events ORDER BY id'))) == event_rows
    assert workspace.scalar(text('SELECT COUNT(*) FROM point_death_records')) == 1
    assert workspace.scalar(text("SELECT COUNT(*) FROM planting_points WHERE id<>1 AND status='planned' AND death_at IS NULL")) == 3
    assert workspace.scalar(text("SELECT status FROM planting_points WHERE id=1")) == 'planted'
    assert workspace.scalar(text('SELECT COUNT(*) FROM planter_assignment_points WHERE released_at IS NOT NULL')) == 2
    assert workspace.scalar(text('SELECT released_at FROM planter_assignment_points WHERE id=1')) is None
    assert workspace.scalar(text('SELECT COUNT(*) FROM replanting_requests WHERE planting_event_id=2')) == 1
    assert workspace.scalar(text('SELECT COUNT(*) FROM project_site_area_releases')) == 1
    after = preview_site_actions(1)
    assert after['counts']['historical_planted'] == 2 and after['counts']['historical_dead'] == 1
    assert after['counts']['living'] == 1 and after['counts']['release'] == 0
    assert not after['can_delete']


def test_stale_preview_and_living_site_delete_are_rejected_atomically(workspace):
    before = preview_site_actions(1)
    with pytest.raises(ValueError, match='all seedlings'):
        apply_site_action(1, 'delete', before['version'], 1)
    workspace.execute(text("UPDATE planting_points SET death_at=CURRENT_TIMESTAMP WHERE id=1"))
    with pytest.raises(ValueError, match='changed after the preview'):
        apply_site_action(1, 'release', before['version'], 1)
    assert workspace.scalar(text('SELECT COUNT(*) FROM project_site_area_releases')) == 0
    assert workspace.scalar(text('SELECT COUNT(*) FROM planter_assignment_points WHERE released_at IS NOT NULL')) == 0
    assert workspace.scalar(text('SELECT deleted_at FROM project_sites WHERE id=1')) is None


def test_all_dead_site_deletes_from_active_use_but_keeps_every_record(workspace):
    workspace.execute(text('DELETE FROM planter_assignment_points WHERE planting_point_id IN (3,4)'))
    workspace.execute(text('DELETE FROM planting_points WHERE id IN (3,4)'))
    workspace.execute(text("""UPDATE planting_points SET death_at=CURRENT_TIMESTAMP WHERE id=1;
        UPDATE planting_events SET closed_at=CURRENT_TIMESTAMP,closure_reason='recorded_death' WHERE id=1;
        INSERT INTO point_death_records(id,planting_point_id,planting_event_id,death_at,reason_category)
            VALUES(2,1,1,CURRENT_TIMESTAMP,'unknown');"""))
    preview = preview_site_actions(1)
    assert preview['can_delete']
    apply_site_action(1, 'delete', preview['version'], 1)
    assert db.list_project_sites() == []
    assert workspace.scalar(text('SELECT deleted_at FROM project_sites WHERE id=1')) is not None
    assert workspace.scalar(text('SELECT COUNT(*) FROM planting_events WHERE project_site_id=1')) == 2
    assert workspace.scalar(text('SELECT COUNT(*) FROM point_death_records')) == 2
    assert workspace.scalar(text('SELECT project_site_id FROM analyses WHERE id=1')) == 1
    assert workspace.scalar(text('SELECT COUNT(*) FROM planting_points')) == 2
    assert shape(available_released_areas()['features'][0]['geometry']).covers(Point(122.62512,10.78))


def test_released_area_can_be_claimed_without_moving_old_survival_history(workspace):
    apply_site_action(1, 'release', preview_site_actions(1)['version'], 1)
    geometry = available_released_areas()['features'][0]['geometry']
    workspace.execute(text("INSERT INTO project_sites(id,name,organization_id,polygon_geojson) VALUES(2,'New organization area',2,CAST(:geometry AS jsonb))"), {'geometry': __import__('json').dumps(geometry)})
    points = db._resolve_point_project_sites(db._get_connection(), [{'id':2,'latitude':10.78,'longitude':122.62512,'source_site_id':1}])
    assert points[0]['source_project_site_id'] == 2
    assert available_released_areas()['features'] == []
    workspace.execute(text("""UPDATE planting_points SET status='planted',planted_at=CURRENT_TIMESTAMP,death_at=NULL WHERE id=2;
        INSERT INTO planting_events(id,source_key,planting_point_id,project_site_id,species,planted_at)
            VALUES(3,'new-cycle',2,2,'Rhizophora',CURRENT_TIMESTAMP);"""))
    original = preview_site_actions(1)
    assert original['counts']['historical_planted'] == 2 and original['counts']['historical_dead'] == 1
    assert workspace.scalar(text('SELECT project_site_id FROM planting_events WHERE id=3')) == 2


def test_incomplete_death_history_cannot_be_erased_by_release(workspace):
    workspace.execute(text('DELETE FROM point_death_records'))
    plan = preview_site_actions(1)
    assert not plan['can_release']
    with pytest.raises(ValueError, match='incomplete planting history'):
        apply_site_action(1, 'release', plan['version'], 1)
    assert workspace.scalar(text('SELECT death_at FROM planting_points WHERE id=2')) is not None


def test_releasing_pending_replacement_work_removes_its_old_assignment_link(workspace):
    workspace.execute(text("""UPDATE planting_points SET status='planned',death_at=NULL,planted_at=NULL WHERE id=2;
        UPDATE planter_assignment_points SET status='pending',completed_at=NULL WHERE id=2;
        INSERT INTO replanting_requests(planting_event_id,approved_by,assignment_id) VALUES(2,1,1);"""))
    apply_site_action(1, 'release', preview_site_actions(1)['version'], 1)
    assert workspace.scalar(text('SELECT assignment_id FROM replanting_requests WHERE planting_event_id=2')) is None
    assert workspace.scalar(text('SELECT COUNT(*) FROM point_death_records')) == 1
    assert workspace.scalar(text('SELECT project_site_id FROM planting_events WHERE id=2')) == 1
