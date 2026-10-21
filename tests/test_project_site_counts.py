"""Count linked site points once across analyses and unreleased assignments."""
import sqlite3

import pytest

import planting_database as db


def test_grouped_site_counts_preserve_membership_without_join_multiplication():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    try:
        conn.executescript('''
            CREATE TABLE site_zones (id INTEGER, geometry TEXT, deleted_at TEXT, area_released_at TEXT);
            CREATE VIEW project_sites AS SELECT * FROM site_zones;
            CREATE TABLE analyses (id INTEGER, site_zone_id INTEGER);
            CREATE TABLE planter_assignments (id INTEGER, site_zone_id INTEGER);
            CREATE TABLE planting_points (id INTEGER, analysis_id INTEGER, location INTEGER);
            CREATE TABLE planter_assignment_points
                (assignment_id INTEGER, planting_point_id INTEGER, released_at TEXT);
            CREATE TABLE planting_schedules (project_site_id INTEGER);
            INSERT INTO site_zones(id,geometry) VALUES (1,'1'),(2,'2,3'),(3,'');
            INSERT INTO analyses VALUES (1,1),(2,1),(3,NULL);
            INSERT INTO planter_assignments VALUES (1,1),(2,1),(3,2);
            INSERT INTO planting_points VALUES (1,1,1),(2,2,2),(3,3,3),(4,3,4);
            INSERT INTO planter_assignment_points VALUES
                (1,1,NULL),(2,1,NULL),(1,3,NULL),(2,4,'2026-09-01'),
                (3,3,NULL),(3,2,'2026-09-01');
            INSERT INTO planting_schedules VALUES (1),(1),(2);
        ''')
        # SQLite supplies the spatial predicate for this small membership fixture.
        conn.create_function('covers', 2, lambda geometry, location: str(location) in geometry.split(','))
        class SpatialConnection:
            def execute(self, sql, params=()):
                return conn.execute(sql.replace('extensions.ST_Covers', 'covers'), params)
        result = db._project_site_counts(SpatialConnection())
        assert result == {
            1: {'assignment_count': 2, 'analysis_count': 2, 'point_count': 2, 'schedule_count': 2},
            2: {'assignment_count': 1, 'analysis_count': 0, 'point_count': 2, 'schedule_count': 1},
            3: {'assignment_count': 0, 'analysis_count': 0, 'point_count': 0, 'schedule_count': 0},
        }
    finally:
        conn.close()


@pytest.fixture
def site_dashboard(monkeypatch):
    """Exercise the real aggregate with synthetic reads; never open a database."""
    class Rows:
        def fetchall(self):
            return []

    class Connection:
        closed = False

        def execute(self, sql, params=()):
            assert sql.strip().startswith('SELECT')
            assert 'FROM analyses a' in sql or 'FROM planter_assignments' in sql
            return Rows()

        def close(self):
            self.closed = True

    connection = Connection()
    polygon = {'type': 'Polygon', 'coordinates': [[[0, 0], [1, 0], [1, 1], [0, 1], [0, 0]]]}
    points = [dict(id=i, analysis_id=1, site_id=1, point_status='planned',
                   latitude=.5, longitude=.5, species='Bungalon') for i in range(1, 7)]
    points[1].update(assignment_point_status='pending', assignment_id=10)
    points[2].update(point_status='planted')
    points[3].update(point_status='planted', death_at='2026-10-01')
    points[4].update(assignment_point_status='skipped')
    points[5].update(eroded_unavailable=True)
    points += [dict(id=7, analysis_id=1, site_id=None, point_status='planned', latitude=2, longitude=2),
               dict(id=8, analysis_id=1, site_id=2, point_status='planned', latitude=2, longitude=2),
               dict(id=9, analysis_id=1, site_id=1, point_status='planned', latitude=.5, longitude=.5, deleted_at='2026-10-01')]
    features = [dict(id=i, geometry=polygon if i != 3 else None, properties={
        'name': f'Site {i}', 'organization_id': i, 'organization_name': 'LGU' if i == 2 else 'Coastal guardians',
        'point_count': 999, 'analysis_count': 0, 'area_released_at': '2026-10-01' if i == 3 else None,
    }) for i in range(1, 4)]

    def sites(**kwargs):
        assert kwargs['_connection'] is connection
        return features

    monkeypatch.setattr(db, '_get_connection', lambda: connection)
    monkeypatch.setattr(db, '_dashboard_filter_options', lambda conn: {'sites': []})
    monkeypatch.setattr(db, '_load_current_dashboard_points', lambda conn: points + [points[0]])
    monkeypatch.setattr(db, 'list_project_sites', sites)
    monkeypatch.setattr(db, 'list_site_zones', lambda: [])
    monkeypatch.setattr(db, 'list_warning_zones', lambda **kwargs: [
        dict(geometry=polygon, properties={'warning_type': kind, 'severity': 'high'})
        for kind in ['deep_mud', 'deep_mud', 'high_mortality']])
    monkeypatch.setattr(db, '_unique_footprint_metrics', lambda rows: {
        'unique_footprint_area_m2': None, 'unique_footprint_quality': 'unavailable',
        'footprints_included': 0, 'footprints_missing': 0})
    return connection


def test_site_dashboard_counts_current_locations_without_analysis_links(site_dashboard):
    result = db.get_dashboard_sites(date_from='2026-10-01', date_to='2026-10-22')
    first, lgu, released = result['project_sites']
    assert first['organization_name'] == 'Coastal guardians'
    assert first['analysis_count'] == 0
    assert first['point_count'] == 6
    assert first['point_status_counts'] == dict.fromkeys(
        ['planned', 'assigned', 'planted', 'dead', 'skipped', 'unavailable'], 1)
    assert first['eroded_point_count'] == 1
    assert lgu['organization_name'] == 'LGU' and lgu['point_count'] == 1
    assert released['area_released_at'] and released['geometry'] is None
    assert released['point_count'] == 0
    assert result['summary']['mapped_points'] == 7  # Excludes unclaimed/deleted locations.
    assert result['summary']['sites_with_warnings'] == 1
    assert site_dashboard.closed


def test_site_warnings_count_each_location_once_per_category_and_use_map_status(site_dashboard):
    result = db.get_dashboard_sites(site_id=1)
    site = result['project_sites'][0]
    assert site['warning_point_count'] == 6
    assert len(site['warnings']) == 2
    assert all(warning['count'] == 6 for warning in site['warnings'])
    assert result['summary']['warning_exposed_points'] == 6
    for warning in result['warning_exposure']:
        assert warning['count'] == 6
        for status in ['planned', 'assigned', 'planted', 'dead', 'skipped', 'unavailable']:
            assert warning[status] == 1


def test_site_snapshot_is_independent_of_reporting_dates_and_respects_site_filter(site_dashboard):
    past = db.get_dashboard_sites(date_from='2020-01-01', date_to='2020-12-31', site_id=1)
    current = db.get_dashboard_sites(date_from='2026-01-01', date_to='2026-10-22', site_id=1)
    assert past['project_sites'] == current['project_sites']
    assert past['summary']['project_sites_scope'] == 'current_state'
    assert past['summary']['mapped_points'] == 6
    other = db.get_dashboard_sites(site_id=2)
    assert [site['id'] for site in other['project_sites']] == [2]
    assert other['summary']['mapped_points'] == 1
    assert other['summary']['sites_with_warnings'] == 0


def test_site_snapshot_keeps_dimension_filters_for_current_points(site_dashboard):
    result = db.get_dashboard_sites(assignment_id=10)
    assert [site['id'] for site in result['project_sites']] == [1]
    assert result['project_sites'][0]['point_count'] == 1
    assert result['project_sites'][0]['point_status_counts']['assigned'] == 1
