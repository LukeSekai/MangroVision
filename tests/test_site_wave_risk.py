"""Wave-specific 30% site flags, distinct locations and persistent history."""
import os
import sqlite3

import pytest

from mangrovision_db.site_wave_risk import project_site_wave_risks, wave_risk


@pytest.fixture
def records():
    conn = sqlite3.connect(':memory:')
    conn.row_factory = sqlite3.Row
    conn.executescript('''
        CREATE TABLE analyses(id INTEGER PRIMARY KEY, site_zone_id INTEGER);
        CREATE TABLE project_sites(id INTEGER PRIMARY KEY, geometry TEXT, deleted_at TEXT, area_released_at TEXT);
        CREATE TABLE planting_points(id INTEGER PRIMARY KEY, analysis_id INTEGER, status TEXT,
            planted_at TEXT, death_at TEXT, death_reason_category TEXT, location TEXT);
        CREATE TABLE planting_events(id INTEGER PRIMARY KEY, planting_point_id INTEGER, site_zone_id INTEGER, closure_reason TEXT);
        CREATE TABLE point_death_records(id INTEGER PRIMARY KEY, planting_point_id INTEGER, planting_event_id INTEGER, reason_category TEXT);
        INSERT INTO project_sites(id,geometry) VALUES (1,'1'),(2,'2'),(3,'3');
        INSERT INTO analyses VALUES (1,1),(2,2),(3,3);
    ''')
    for site in (1, 2, 3):
        for index in range(1, 151):
            point = site * 1000 + index
            conn.execute('INSERT INTO planting_points VALUES (?,?,?,?,?,?,?)',
                         (point, site, 'planted' if index <= 100 else 'planned', '2026-01-01' if index <= 100 else None, None, None, str(site)))
            if index <= 100:
                conn.execute('INSERT INTO planting_events VALUES (?,?,?,NULL)', (point, point, site))
    conn.create_function('covers', 2, lambda geometry, location: geometry == location)
    class SpatialConnection:
        def execute(self, sql):
            return conn.execute(sql.replace('extensions.ST_Covers', 'covers'))
    yield conn, lambda: project_site_wave_risks(SpatialConnection())
    conn.close()


def death(conn, point, cause, record_id=None, event_id=None):
    conn.execute('INSERT INTO point_death_records VALUES (?,?,?,?)', (record_id or point, point, event_id or point, cause))


@pytest.mark.parametrize('dead,expected', [(0, False), (10, False), (29, False), (30, True), (31, True), (100, True)])
def test_threshold_uses_all_planted_points_and_only_waves(records, dead, expected):
    conn, risks = records
    for index in range(1, 101):
        death(conn, 1000 + index, 'waves' if index <= dead else 'storm')
    result = risks()[1]
    assert result['planted_points'] == 100  # 50 merely planned points do not dilute it.
    assert result['wave_death_points'] == dead
    assert result['active'] is expected


def test_mixed_causes_never_combine_into_a_wave_warning(records):
    conn, risks = records
    for index in range(1, 31):
        death(conn, 1000 + index, 'barnacles' if index <= 10 else 'storm' if index <= 20 else 'waves')
    assert risks()[1] == wave_risk(100, 10)
    assert risks()[2] == wave_risk(100, 0)


def test_repeated_records_and_replanting_keep_distinct_historical_locations(records):
    conn, risks = records
    for index in range(1, 31):
        point = 1000 + index
        death(conn, point, 'waves')
        conn.execute('INSERT INTO planting_events VALUES (?,?,?,NULL)', (point + 10000, point, 1))
        death(conn, point, 'waves', record_id=point + 10000, event_id=point + 10000)
    conn.execute("UPDATE planting_points SET status='planned', planted_at=NULL, death_at=NULL WHERE analysis_id=1")
    assert risks()[1] == wave_risk(100, 30)
    # Correcting every wave record for one point drops below the threshold.
    conn.execute('DELETE FROM point_death_records WHERE planting_point_id=1001')
    assert risks()[1] == wave_risk(100, 29)


def test_event_site_wins_over_later_location_and_analysis_reassignment(records):
    conn, risks = records
    for index in range(1, 31):
        death(conn, 1000 + index, 'waves')
    conn.execute("UPDATE planting_points SET analysis_id=2, location='2' WHERE analysis_id=1")
    assert risks()[1] == wave_risk(100, 30)
    assert risks()[2] == wave_risk(100, 0)


def test_legacy_deaths_without_events_still_count_once(records):
    conn, risks = records
    conn.execute('DELETE FROM planting_events WHERE site_zone_id=1')
    for index in range(1, 31):
        death(conn, 1000 + index, 'waves')
    assert risks()[1] == wave_risk(100, 30)
    conn.execute("UPDATE planting_points SET death_at='2026-01-02', death_reason_category='waves' WHERE id BETWEEN 1031 AND 1035")
    assert risks()[1] == wave_risk(100, 35)


def test_unlocated_reports_and_reversed_completions_do_not_invent_site_deaths(records):
    conn, risks = records
    conn.execute("CREATE TABLE organization_monitoring_records(location_death_count INTEGER, death_reason_category TEXT)")
    conn.execute("INSERT INTO organization_monitoring_records VALUES (1000,'waves')")
    conn.execute("UPDATE planting_events SET closure_reason='completion_reversed' WHERE site_zone_id=1")
    conn.execute("UPDATE planting_points SET status='planned',planted_at=NULL WHERE analysis_id=1")
    assert 1 not in risks()
    assert wave_risk()['active'] is False
    assert wave_risk(10001, 3000)['active'] is False  # Display rounding never triggers a flag.


@pytest.mark.skipif(os.getenv('MANGROVISION_RUN_WAVE_TESTS') != '1', reason='Temporary PostgreSQL check is opt-in.')
def test_query_on_postgres_temporary_tables_only():
    from sqlalchemy import text
    from mangrovision_db.compat import CompatConnection, get_engine
    try:
        connection = get_engine().connect()
    except Exception:
        pytest.fail('Temporary database connection unavailable; details suppressed.', pytrace=False)
    with connection:
        transaction = connection.begin()
        try:
            connection.exec_driver_sql('''
                SET LOCAL search_path = pg_temp, public;
                SET LOCAL statement_timeout = '20s';
                CREATE TEMP TABLE project_sites(id BIGINT PRIMARY KEY, geometry extensions.geometry(Polygon,4326),
                    deleted_at TIMESTAMPTZ, area_released_at TIMESTAMPTZ);
                CREATE TEMP TABLE analyses(id BIGINT PRIMARY KEY, project_site_id BIGINT);
                CREATE TEMP TABLE planting_points(id BIGINT PRIMARY KEY, analysis_id BIGINT, status TEXT,
                    planted_at TIMESTAMPTZ, death_at TIMESTAMPTZ, death_reason_category TEXT, location extensions.geometry(Point,4326));
                CREATE TEMP TABLE planting_events(id BIGINT PRIMARY KEY, planting_point_id BIGINT, project_site_id BIGINT, closure_reason TEXT);
                CREATE TEMP TABLE point_death_records(id BIGINT PRIMARY KEY, planting_point_id BIGINT, planting_event_id BIGINT, reason_category TEXT);
                INSERT INTO project_sites(id,geometry) VALUES
                    (1,extensions.ST_MakeEnvelope(0,0,1,1,4326)), (2,extensions.ST_MakeEnvelope(2,0,3,1,4326));
                INSERT INTO analyses VALUES (1,1),(2,2);
                INSERT INTO planting_points SELECT n,1,'planted',CURRENT_TIMESTAMP,NULL,NULL,
                    extensions.ST_SetSRID(extensions.ST_MakePoint(.5,.5),4326) FROM generate_series(1,100) n;
                INSERT INTO planting_points SELECT n,2,'planted',CURRENT_TIMESTAMP,NULL,NULL,
                    extensions.ST_SetSRID(extensions.ST_MakePoint(2.5,.5),4326) FROM generate_series(101,200) n;
                INSERT INTO planting_events SELECT id,id,analysis_id,NULL FROM planting_points;
                INSERT INTO point_death_records SELECT n,n,n,'waves' FROM generate_series(1,30) n;
                INSERT INTO point_death_records SELECT n,n,n,
                    CASE WHEN n<=110 THEN 'barnacles' WHEN n<=120 THEN 'storm' ELSE 'waves' END
                    FROM generate_series(101,130) n;
            ''')
            result = project_site_wave_risks(CompatConnection(connection))
            assert result[1] == wave_risk(100, 30)
            assert result[2] == wave_risk(100, 10)
            connection.execute(text("UPDATE planting_points SET status='planned',planted_at=NULL WHERE analysis_id=1"))
            assert project_site_wave_risks(CompatConnection(connection))[1]['active'] is True
        finally:
            transaction.rollback()
