"""Reported-death totals and Philippine calendar months on rollback-only tables."""
import os
from datetime import datetime
from zoneinfo import ZoneInfo

import pytest

from mangrovision_db.mortality import mortality_cause_counts, mortality_monthly_counts

pytestmark = pytest.mark.skipif(
    os.getenv('MANGROVISION_RUN_MORTALITY_TESTS') != '1',
    reason='Temporary PostgreSQL mortality checks are opt-in.',
)
MANILA = ZoneInfo('Asia/Manila')
START = datetime(2026, 5, 1, tzinfo=MANILA)
END = datetime(2026, 9, 30, 23, 59, 59, tzinfo=MANILA)


@pytest.fixture
def records():
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
                CREATE TEMP TABLE planting_events(id BIGINT PRIMARY KEY, planting_point_id BIGINT,
                    project_site_id BIGINT, assignment_id BIGINT, planter_id BIGINT, species TEXT,
                    planted_at TIMESTAMPTZ);
                CREATE TEMP TABLE point_death_records(id BIGINT PRIMARY KEY, planting_event_id BIGINT,
                    death_at TIMESTAMPTZ, reason_category TEXT, assignment_id BIGINT, planter_id BIGINT, species TEXT);
                CREATE TEMP TABLE monitoring_death_locations(id BIGINT PRIMARY KEY, record_id BIGINT,
                    death_record_id BIGINT, revoked_at TIMESTAMPTZ);
                CREATE TEMP TABLE organization_monitoring_records(id BIGINT PRIMARY KEY, monitored_at TIMESTAMPTZ,
                    location_death_count BIGINT, dead_count BIGINT, count_snapshot JSONB);
                CREATE TEMP TABLE monitoring_observations(id BIGINT PRIMARY KEY, planting_event_id BIGINT, status TEXT);
                INSERT INTO planting_events VALUES
                    (1,201,7,11,501,'Bungalon','2026-04-01T00:00Z'),
                    (2,202,7,11,501,'Bungalon','2026-04-02T00:00Z'),
                    (3,203,8,12,502,'Rhizophora','2026-05-01T00:00Z');
                INSERT INTO point_death_records VALUES
                    (1,1,'2026-05-31T17:15Z','waves',11,501,'Bungalon'),
                    (2,2,'2026-05-15T01:00Z','unknown',11,501,'Bungalon'),
                    (3,3,'2026-09-10T02:00Z','drying_out',12,502,'Rhizophora');
                INSERT INTO organization_monitoring_records VALUES
                    (7,'2026-06-30T16:30Z',3,30,'{"death_reason_category":"disease"}'),
                    (8,'2026-08-15T02:00Z',0,50,'{"death_reason_category":"waves"}'),
                    (9,'2026-05-10T02:00Z',2,99,'{"death_reason_category":"storm"}');
                INSERT INTO monitoring_death_locations VALUES
                    (1,7,2,NULL),(2,8,3,'2026-09-11T00:00Z');
                INSERT INTO monitoring_observations VALUES (1,2,'dead'),(2,2,'dead');
            ''')
            yield CompatConnection(connection)
        finally:
            transaction.rollback()


def test_deaths_are_counted_once_across_individual_inspections_and_visit_reports(records):
    causes = mortality_cause_counts(records, START, END)
    monthly = mortality_monthly_counts(records, START, END)
    assert causes == {'storm': 2, 'waves': 1, 'disease': 3, 'drying_out': 1}
    assert monthly == {'2026-05-01': 2, '2026-06-01': 1, '2026-07-01': 3, '2026-09-01': 1}
    assert sum(causes.values()) == sum(monthly.values()) == 7


@pytest.mark.parametrize('filters,expected', [
    ({'site_id': 7}, {'2026-06-01': 1, '2026-07-01': 1}),
    ({'assignment_id': 11}, {'2026-06-01': 1, '2026-07-01': 1}),
    ({'planter_id': 501}, {'2026-06-01': 1, '2026-07-01': 1}),
    ({'species': 'bungalon'}, {'2026-06-01': 1, '2026-07-01': 1}),
    ({'site_id': 8}, {'2026-09-01': 1}),
    ({'site_id': 99}, {}),
])
def test_filtered_counts_never_assign_unlocated_deaths_to_a_project_or_species(records, filters, expected):
    monthly = mortality_monthly_counts(records, START, END, **filters)
    assert monthly == expected
    assert sum(mortality_cause_counts(records, START, END, **filters).values()) == sum(monthly.values())


def test_dates_use_the_recorded_visit_and_philippine_calendar_month(records):
    start = datetime(2026, 7, 1, tzinfo=MANILA)
    end = datetime(2026, 7, 31, 23, 59, 59, tzinfo=MANILA)
    assert mortality_monthly_counts(records, start, end) == {'2026-07-01': 3}
    assert mortality_monthly_counts(records, start, end, site_id=7) == {'2026-07-01': 1}
    # Its raw death timestamp was in May; the linked report is July 1 locally.
    june_end = datetime(2026, 6, 30, 23, 59, 59, tzinfo=MANILA)
    assert mortality_monthly_counts(records, START, june_end, site_id=7) == {'2026-06-01': 1}


def test_previous_cycles_and_year_boundaries_remain_in_death_history(records):
    records.execute('''INSERT INTO planting_events VALUES
        (4,201,7,11,501,'Bungalon','2026-12-01T00:00Z')''')
    records.execute('''INSERT INTO point_death_records VALUES
        (4,4,'2026-12-31T16:15Z','waves',11,501,'Bungalon')''')
    end = datetime(2027, 1, 31, 23, 59, 59, tzinfo=MANILA)
    assert mortality_monthly_counts(records, START, end, site_id=7) == {
        '2026-06-01': 1, '2026-07-01': 1, '2027-01-01': 1,
    }
