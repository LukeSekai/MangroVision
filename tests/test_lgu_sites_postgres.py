"""LGU account, four independent sites, and planting access in rollback-only tables."""
import importlib.util
import os
from datetime import timedelta
from pathlib import Path

import pytest
from sqlalchemy import text

import planting_database as db
from mangrovision_db import group_access, lgu_sites
from mangrovision_db.compat import CompatConnection
from mangrovision_db.request_context import planter_session_token
from test_site_release_postgres import workspace as release_workspace

pytestmark = pytest.mark.skipif(os.getenv('MANGROVISION_RUN_TEMP_ACCOUNT_TESTS') != '1', reason='Temporary PostgreSQL tables only.')


@pytest.fixture
def workspace(release_workspace, monkeypatch):
    connection = release_workspace
    for table in ['auth_sessions', 'participant_browser_bindings', 'activity_access_codes', 'planter_access_codes', 'planter_auth_limits', 'monitoring_observations', 'organization_monitoring_records', 'monitoring_death_locations']:
        connection.exec_driver_sql(f'CREATE TEMP TABLE {table} (LIKE mangrovision.{table} INCLUDING ALL)')
    connection.exec_driver_sql('''ALTER TABLE pg_temp.organizations DROP COLUMN IF EXISTS system_key;
        ALTER TABLE pg_temp.organizations ALTER COLUMN id RESTART WITH 1000;
        ALTER TABLE pg_temp.planters ALTER COLUMN id RESTART WITH 1000;''')
    spec = importlib.util.spec_from_file_location('lgu_migration', Path(__file__).resolve().parents[1] / 'alembic/versions/20261019_0020_permanent_lgu_planting_and_site_.py')
    migration = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(migration)
    class Operations:
        def execute(self, sql):
            connection.exec_driver_sql(sql.replace('mangrovision.', 'pg_temp.').replace('search_path = mangrovision,', 'search_path = pg_temp,'))
    migration.op = Operations()
    migration.upgrade()
    for table in ['permanent_planting_access', 'lgu_site_monitoring_records']:
        namespace = connection.scalar(text('SELECT n.nspname FROM pg_class c JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.oid=to_regclass(:name)'), {'name': table})
        assert namespace.startswith('pg_temp_')
    class Connection(CompatConnection):
        def __init__(self):
            super().__init__(connection)
            self.savepoint = connection.begin_nested()
        def commit(self):
            if self.savepoint.is_active:
                self.savepoint.commit()
            self.savepoint = connection.begin_nested()
        def rollback(self):
            if self.savepoint.is_active:
                self.savepoint.rollback()
        def close(self):
            self.rollback()
    monkeypatch.setattr(db, '_get_connection', Connection)
    monkeypatch.setattr(group_access, 'get_connection', Connection)
    monkeypatch.setitem(group_access.BOOKING_POLICY, 'enforceActivityAccessWindow', True)
    account = lgu_sites.ensure_lgu_account()
    org, owner = account['organization_id'], account['planter_id']
    def batch(statement, params):
        for part in statement.text.split(';'):
            if part.strip():
                connection.execute(text(part), params)
    for index, (total, deaths, age) in enumerate([(100, 20, 40), (200, 20, 30), (50, 20, 10), (150, 15, 2)]):
        site = 10 + index
        west = 122.630 + index * .002
        polygon = {'type': 'Polygon', 'coordinates': [[[west,10.779],[west+.001,10.779],[west+.001,10.781],[west,10.781],[west,10.779]]]}
        import json
        batch(text('''INSERT INTO project_sites(id,name,organization_id,polygon_geojson)
            VALUES(:site,:name,:org,CAST(:polygon AS jsonb));
            INSERT INTO planter_assignments(id,planter_id,title,assignment_date,project_site_id,status,species)
            VALUES(:site,:owner,'LGU planting',CURRENT_DATE-:age,:site,'completed','Bungalon');'''),
            {'site': site, 'name': f'LGU site {index+1}', 'org': org, 'owner': owner, 'age': age, 'polygon': json.dumps(polygon)})
        batch(text('''INSERT INTO planting_points(id,analysis_id,point_num,latitude,longitude,status,planted_at,death_at)
            SELECT :site*1000+n,1,:site*1000+n,10.780,:west+n*.000001,'planted',CURRENT_TIMESTAMP-:age*INTERVAL '1 day',
                CASE WHEN n<=:deaths THEN CURRENT_TIMESTAMP-INTERVAL '1 day' END FROM generate_series(1,:total) n;
            INSERT INTO planter_assignment_points(assignment_id,planting_point_id,sequence_num,status,completed_at)
            SELECT :site,id,point_num,'completed',planted_at FROM planting_points WHERE id BETWEEN :site*1000 AND :site*1000+999;
            INSERT INTO planting_events(source_key,planting_point_id,assignment_id,project_site_id,planter_id,species,
                planted_at,latitude,longitude,point_num,closed_at,closure_reason)
            SELECT 'lgu-fixture-'||id,id,:site,:site,:owner,'Bungalon',planted_at,latitude,longitude,point_num,death_at,
                CASE WHEN death_at IS NOT NULL THEN 'recorded_death' END FROM planting_points WHERE id BETWEEN :site*1000 AND :site*1000+999;
            INSERT INTO point_death_records(planting_point_id,planting_event_id,death_at,reason_category,assignment_id)
            SELECT pp.id,pe.id,pp.death_at,'waves',:site FROM planting_points pp JOIN planting_events pe ON pe.planting_point_id=pp.id
            WHERE pp.id BETWEEN :site*1000 AND :site*1000+999 AND pp.death_at IS NOT NULL;'''),
            {'site': site, 'owner': owner, 'west': west+.00001, 'age': age, 'deaths': deaths, 'total': total})
    yield connection


def visit(site, context, ids=()):
    return lgu_sites.record_visit(site, monitored_at=db._manila_now().date().isoformat(), version=context['version'],
        dead_planting_event_ids=list(ids), health_status='good', actions_taken='Checked seedlings and removed debris.',
        inspector_user_id=1, death_reason_category='waves')


def test_four_sites_are_one_lgu_with_weighted_survival_and_separate_due_dates(workspace):
    data = lgu_sites.workspace()
    assert data['summary'] == {'site_count': 4, 'total_planted': 500, 'alive_count': 425,
                               'dead_count': 75, 'unknown_count': 0, 'survival_rate_pct': 85}
    assert [site['survival_rate_pct'] for site in data['sites']] == [80, 90, 60, 90]
    dates = [site['next_monitoring_date'] for site in data['sites']]
    assert len(set(dates)) == 4
    assert data['account']['permanent_access'] and data['account']['expires_at'] is None
    connection = CompatConnection(workspace)
    events = db._load_dashboard_events(connection)
    now = db._manila_now()
    from mangrovision_db.planting_performance import summarize_planting_performance
    report = summarize_planting_performance(events, now-timedelta(days=100), now, now)
    lgu = [row for row in report['organization_outcomes'] if row['name'] == 'LGU']
    assert len(lgu) == 1 and lgu[0]['total'] == 500 and lgu[0]['survival_rate_pct'] == 85
    assert not any(row['name'] == 'LGU' for row in db.list_organization_monitoring_summaries())


def test_monitoring_one_site_preserves_other_sites_and_the_planting_age_anchor(workspace):
    before_b = lgu_sites.site_context(11)
    before = lgu_sites.site_context(10)
    event_id = before['points'][0]['id']
    visit(10, before, [event_id])
    after = lgu_sites.site_context(10)
    assert after['alive_count'] == 79 and after['dead_count'] == 21
    assert len(after['history']) == 1
    assert lgu_sites.site_context(11)['version'] == before_b['version']
    assert lgu_sites.site_context(11)['next_monitoring_date'] == before_b['next_monitoring_date']
    planted = db._parse_dashboard_datetime(before['first_planted_at']).date()
    assert after['next_monitoring_date'] == db._monitoring_workday(planted+timedelta(days=42)).isoformat()
    assert lgu_sites.workspace()['summary']['survival_rate_pct'] == 84.8
    due = db.get_due_monitoring_inspections(as_of=db._manila_now(), project_site_id=10, upcoming_days=0,
        _conn=CompatConnection(workspace), _settings={'inspection_intervals_days': [30,90,180,365], 'inspection_weekdays': [1]})
    assert not due['observations_due']
    assert workspace.scalar(text('SELECT COUNT(*) FROM monitoring_observations')) == 80
    point_id = workspace.scalar(text('SELECT planting_point_id FROM planting_events WHERE id=:id'), {'id': event_id})
    with pytest.raises(ValueError, match='LGU site monitoring'):
        db.restore_planting_point_to_planted(point_id)


def test_early_foreign_stale_and_duplicate_visits_do_not_write(workspace):
    a, b, young = lgu_sites.site_context(10), lgu_sites.site_context(11), lgu_sites.site_context(12)
    with pytest.raises(ValueError, match='not due'):
        visit(12, young)
    with pytest.raises(ValueError, match='this LGU site'):
        visit(10, a, [b['points'][0]['id']])
    with pytest.raises(ValueError, match='changed'):
        visit(10, {**a, 'version': '0'*64})
    assert workspace.scalar(text('SELECT COUNT(*) FROM lgu_site_monitoring_records')) == 0
    visit(10, a)
    with pytest.raises(ValueError, match='changed'):
        visit(10, a)
    assert workspace.scalar(text('SELECT COUNT(*) FROM lgu_site_monitoring_records')) == 1
    with pytest.raises(ValueError, match='belonging to LGU'):
        lgu_sites.site_context(1)


def test_permanent_credentials_keep_working_without_activity_and_can_sign_out(workspace):
    before = lgu_sites.workspace()['account']
    lgu_sites.ensure_lgu_account()
    assert lgu_sites.workspace()['account'] == before
    assert group_access.check_group_code(before['access_code'])['permanent_access']
    account, token = group_access.authenticate_group(before['access_code'], before['username'], 'a'*48, mode='return')
    current = db.get_planter_by_session_token(token)
    assert account['id'] == before['planter_id'] and current['permanent_access']
    assert current['planting_schedule_id'] is None
    assert len(db.get_planter_field_points(account['id'], 1)) == 500
    db.revoke_planter_session(token)
    assert db.get_planter_by_session_token(token) is None
    assert group_access.check_group_code(before['access_code'])['permanent_access']


def test_lgu_identity_and_account_cannot_be_removed_or_deactivated(workspace):
    owner = lgu_sites.workspace()['account']['planter_id']
    for sql in ["DELETE FROM organizations WHERE system_key='lgu'", "UPDATE organizations SET name='Other' WHERE system_key='lgu'",
                f"UPDATE planters SET status='inactive' WHERE id={owner}", f'DELETE FROM planters WHERE id={owner}']:
        with pytest.raises(Exception):
            with workspace.begin_nested():
                workspace.execute(text(sql))
    assert workspace.scalar(text("SELECT COUNT(*) FROM organizations WHERE system_key='lgu'")) == 1


def test_permanent_planter_completes_only_its_own_assigned_points(workspace):
    account = lgu_sites.workspace()['account']
    _, token = group_access.authenticate_group(account['access_code'], account['username'], 'a'*48, mode='return')
    workspace.execute(text('''INSERT INTO planting_points(id,analysis_id,point_num,latitude,longitude,status)
        VALUES(99999,1,99999,10.780,122.6309,'planned')'''))
    assignment = db.create_planter_assignment(account['planter_id'], [99999], site_zone_id=10,
        species='Bungalon', assigned_by_user_id=1)
    pap = workspace.scalar(text('SELECT id FROM planter_assignment_points WHERE assignment_id=:assignment'), {'assignment': assignment})
    context_token = planter_session_token.set(token)
    try:
        with pytest.raises(group_access.GroupAccessError, match='Only points assigned'):
            db.update_assignment_point_status(1, 'completed', actor_planter_id=account['planter_id'], participant_slot=1)
        db.update_assignment_point_status(pap, 'completed', actor_planter_id=account['planter_id'], participant_slot=1)
    finally:
        planter_session_token.reset(context_token)
    event = workspace.execute(text('SELECT * FROM planting_events WHERE planting_point_id=99999')).mappings().one()
    assert event['project_site_id'] == 10 and event['planter_id'] == account['planter_id']
    assert workspace.scalar(text("SELECT status FROM planting_points WHERE id=99999")) == 'planted'


def test_backdated_new_batch_does_not_inherit_an_earlier_visit(workspace):
    context = lgu_sites.site_context(10)
    visit(10, context)
    workspace.execute(text('''INSERT INTO planting_points(id,analysis_id,point_num,latitude,longitude,status,planted_at)
        VALUES(99999,1,99999,10.780,122.6309,'planted',CURRENT_TIMESTAMP-INTERVAL '20 days');
        INSERT INTO planting_events(source_key,planting_point_id,project_site_id,planter_id,species,planted_at,point_num,latitude,longitude)
        SELECT 'late-entered-lgu',99999,10,id,'Bungalon',CURRENT_TIMESTAMP-INTERVAL '20 days',99999,10.780,122.6309
        FROM planters WHERE full_name='LGU';'''))
    updated = lgu_sites.site_context(10)
    assert updated['monitoring_available']
    assert updated['next_monitoring_date'] < db._manila_now().date().isoformat()
