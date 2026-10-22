"""Mortality warnings follow planting outcomes, not a presumed cause of death."""
import os
from copy import deepcopy

import pytest

from mangrovision_db.site_mortality import mortality_warning, mortality_warning_zones


@pytest.mark.parametrize('total,dead,active', [(0, 0, False), (100, 29, False), (100, 30, True),
                                             (100, 100, True), (10001, 3000, False), (263, 103, True)])
def test_exact_threshold_uses_original_plantings_before_display_rounding(total, dead, active):
    result = mortality_warning({'total': total, 'dead': dead})
    assert result['active'] is active
    assert result['total_planted'] == total
    assert result['dead_count'] == dead
    assert result['mortality_rate_pct'] == (round(dead / total * 100, 2) if total else 0)
    assert result['threshold_percentage'] == 30


def site(history=None, **props):
    return {'type': 'Feature', 'id': 7, 'geometry': {'type': 'Polygon', 'coordinates': [
        [[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}, 'properties': {
        'id': 7, 'name': 'Cabatuan', 'organization_id': 8,
        'mortality_warning': mortality_warning(history or {'total': 263, 'dead': 103}), **props}}


def test_zone_is_derived_once_from_current_site_geometry_and_corrections_remove_it():
    feature = site()
    original = deepcopy(feature)
    warning = mortality_warning_zones([feature])[0]
    assert warning['id'] == -7
    assert warning['properties']['automatic'] is True
    assert warning['properties']['warning_type'] == 'high_mortality'
    assert warning['properties']['project_site_id'] == 7
    assert warning['geometry'] == feature['geometry']
    assert '39.16%' in warning['properties']['notes']
    assert warning['properties']['notes'] == 'High mortality (39.16% dead). Check before planting.'
    assert 'wave' not in warning['properties']['notes'].lower()
    assert feature == original
    feature['geometry']['coordinates'][0][1] = [0.8, 0]
    assert mortality_warning_zones([feature])[0]['geometry'] == feature['geometry']
    feature['properties']['mortality_warning'] = mortality_warning({'total': 100, 'dead': 29})
    assert mortality_warning_zones([feature]) == []


def test_no_zone_is_drawn_on_released_land_and_unknown_outcomes_are_not_deaths():
    released = site()
    released['geometry'] = None
    assert mortality_warning_zones([released]) == []
    result = mortality_warning({'total': 100, 'dead': 29, 'unknown': 50})
    assert result['unknown_count'] == 50
    assert result['active'] is False


@pytest.mark.parametrize('status,death_at', [('planted', None), ('planted', '2026-10-20')])
def test_site_mortality_does_not_flag_each_planting_location(monkeypatch, status, death_at):
    import planting_database as db
    from shapely.geometry import shape
    warning = mortality_warning_zones([site()])[0]
    monkeypatch.setattr(db, '_load_warning_zone_geometries', lambda: [{
        **warning['properties'], 'geometry': shape(warning['geometry']),
    }])
    point = {'id': 1, 'planting_status': status, 'death_at': death_at, 'latitude': .5, 'longitude': .5}
    result = db._annotate_point_advisories([point], eroded_point_ids=set())[0]
    assert result['warning_summary'] is None
    assert result['warning_severity'] is None
    assert result['survival_warning'] is False
    assert result['death_at'] == death_at
    assert result['planting_status'] == status


def test_wave_warnings_match_recorded_point_ids_not_neighboring_coordinates(monkeypatch):
    import planting_database as db
    monkeypatch.setattr(db, '_load_warning_zone_geometries', lambda: [{
        'id': None, 'name': 'Wave-damaged locations', 'warning_type': 'wave_exposure',
        'severity': 'medium', 'notes': 'Wave-caused death recorded here.', 'point_ids': {1},
    }])
    points = [
        {'id': 1, 'planting_status': 'planned', 'death_at': None, 'latitude': .5, 'longitude': .5},
        {'id': 2, 'planting_status': 'planted', 'death_at': None, 'latitude': .5, 'longitude': .5},
        {'id': 3, 'planting_status': 'planted', 'death_at': '2026-10-20', 'latitude': .5, 'longitude': .5},
    ]
    released, living, other_death = db._annotate_point_advisories(points, eroded_point_ids=set())
    assert released['survival_warning'] is True
    assert released['warning_summary'] == 'Wave exposure: Wave-caused death recorded here.'
    assert released['map_status'] == 'planned' and released['death_at'] is None
    assert living['survival_warning'] is False and living['warning_summary'] is None
    assert other_death['survival_warning'] is False and other_death['warning_summary'] is None


def test_warning_repository_combines_manual_and_automatic_zones_without_writing(monkeypatch):
    import planting_database as db
    class Connection:
        closed = False
        def execute(self, sql):
            assert 'SELECT' in sql and 'FROM warning_zones' in sql
            return self
        def fetchall(self):
            return [{'id': 1, 'name': 'Manual warning', 'warning_type': 'deep_mud', 'severity': 'medium',
                     'notes': '', 'polygon_geojson': '{"type":"Polygon","coordinates":[]}',
                     'created_at': None, 'updated_at': None}]
        def close(self):
            self.closed = True
    connection = Connection()
    features = [site()]
    monkeypatch.setattr(db, 'list_project_sites', lambda **kwargs: features if kwargs['_connection'] is connection else pytest.fail('Expected shared connection'))
    result = db.list_warning_zones(_connection=connection)
    assert [row['id'] for row in result] == [1, -7]
    assert connection.closed is False
    features[0]['properties']['mortality_warning']['active'] = False
    assert [row['id'] for row in db.list_warning_zones(_connection=connection)] == [1]


def test_automatic_warning_cannot_be_deleted_or_edited_as_a_manual_zone(monkeypatch):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routes import zones
    monkeypatch.setattr(zones, '_raise_if_processing_active', lambda: None)
    monkeypatch.setattr(zones, '_require_lgu_user', lambda: {'id': 1})
    monkeypatch.setattr(zones, 'get_warning_zone', lambda zone_id: mortality_warning_zones([site()])[0])
    monkeypatch.setattr(zones, 'delete_warning_zone', lambda *args: pytest.fail('Cannot delete automatic warning'))
    monkeypatch.setattr(zones, 'update_warning_zone', lambda *args, **kwargs: pytest.fail('Cannot edit automatic warning'))
    app = FastAPI()
    app.include_router(zones.router)
    with TestClient(app) as client:
        assert client.delete('/warnings/-7').status_code == 409
        assert client.patch('/warnings/-7', json={'name': 'Changed'}).status_code == 409


@pytest.mark.skipif(os.getenv('MANGROVISION_RUN_MORTALITY_WARNING_TESTS') != '1', reason='Rollback-only PostgreSQL test is opt-in.')
def test_all_causes_release_reassignment_and_corrections_on_postgres():
    import planting_database as db
    from mangrovision_db.compat import CompatConnection, get_engine
    try:
        connection = get_engine().connect()
    except Exception:
        pytest.fail('Test database connection unavailable; private details suppressed.', pytrace=False)
    with connection:
        transaction = connection.begin()
        try:
            connection.exec_driver_sql('''
                SET LOCAL search_path = pg_temp, public;
                SET LOCAL statement_timeout = '15s';
                CREATE TEMP TABLE planting_points(id BIGINT PRIMARY KEY, status TEXT, deleted_at TIMESTAMPTZ);
                CREATE TEMP TABLE planting_events(id BIGINT PRIMARY KEY, planting_point_id BIGINT,
                    project_site_id BIGINT, planted_at TIMESTAMPTZ, closed_at TIMESTAMPTZ, closure_reason TEXT);
                CREATE TEMP TABLE point_death_records(id BIGINT PRIMARY KEY, planting_event_id BIGINT UNIQUE,
                    death_at TIMESTAMPTZ, reason_category TEXT);
                INSERT INTO planting_points SELECT n,'planted',NULL FROM generate_series(1,150) n;
                INSERT INTO planting_events SELECT n,n,1,CURRENT_TIMESTAMP - INTERVAL '30 days',NULL,NULL
                    FROM generate_series(1,100) n;
                INSERT INTO point_death_records SELECT n,n,CURRENT_TIMESTAMP - INTERVAL '1 day',
                    CASE WHEN n<=10 THEN 'waves' WHEN n<=20 THEN 'barnacles' ELSE 'unknown' END
                    FROM generate_series(1,30) n;
                INSERT INTO planting_events VALUES (999,150,1,CURRENT_TIMESTAMP,NULL,'completion_reversed');
            ''')
            conn = CompatConnection(connection)
            history = db._project_site_planting_history(conn)
            assert mortality_warning(history[1])['active'] is True
            assert history[1]['total'] == 100
            assert history[1]['dead'] == 30
            # Release the original locations and plant replacements in a different site.
            connection.exec_driver_sql("UPDATE planting_points SET status='planned' WHERE id<=30")
            connection.exec_driver_sql('''INSERT INTO planting_events
                SELECT 1000+n,n,2,CURRENT_TIMESTAMP,NULL,NULL FROM generate_series(1,30) n''')
            history = db._project_site_planting_history(conn)
            assert history[1]['total'] == 100 and history[1]['dead'] == 30
            assert mortality_warning(history[1])['active'] is True
            assert history[2]['total'] == 30 and history[2]['dead'] == 0
            assert mortality_warning(history[2])['active'] is False
            connection.exec_driver_sql('DELETE FROM point_death_records WHERE planting_event_id=30')
            assert mortality_warning(db._project_site_planting_history(conn)[1])['active'] is False
        finally:
            transaction.rollback()
