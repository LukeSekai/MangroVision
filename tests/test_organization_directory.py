"""Directory permissions and rollback-only PostgreSQL record projections."""
import json
import os
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy import text

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
from api.routes import organizations as routes, monitoring
from mangrovision_db import organization_directory as directory
import planting_database as planting
from test_like_appointments_postgres import workspace
from test_planter_codes_postgres import seed


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(routes.router, prefix='/api/organizations')
    with TestClient(app) as client:
        yield client


@pytest.mark.parametrize('user,status', [(None, 401), ({'id': 1, 'role': 'viewer'}, 403)])
def test_non_staff_cannot_read_contact_or_member_records(client, monkeypatch, user, status):
    monkeypatch.setattr(monitoring, 'get_user_by_session_token', lambda token: user)
    monkeypatch.setattr(routes, 'list_directory', lambda: pytest.fail('Unauthorized directory read'))
    monkeypatch.setattr(routes, 'organization_detail', lambda org: pytest.fail('Unauthorized member read'))
    assert client.get('/api/organizations').status_code == status
    assert client.get('/api/organizations/1').status_code == status


@pytest.mark.parametrize('role', ['admin', 'lgu', 'planner'])
def test_staff_reads_are_private_and_do_not_accept_writes(client, monkeypatch, role):
    monkeypatch.setattr(monitoring, 'get_user_by_session_token', lambda token: {'id': 1, 'role': role})
    monkeypatch.setattr(routes, 'list_directory', lambda: [{'id': 1, 'name': 'Example group'}])
    monkeypatch.setattr(routes, 'organization_detail', lambda org: {'organization': {'id': org}, 'members': []} if org == 1 else None)
    response = client.get('/api/organizations')
    assert response.json()['organizations'][0]['name'] == 'Example group'
    assert response.headers['cache-control'] == 'private, no-store'
    response = client.get('/api/organizations/1')
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'private, no-store'
    assert client.get('/api/organizations/999').status_code == 404
    assert client.post('/api/organizations/1', json={}).status_code == 405
    assert client.delete('/api/organizations/1').status_code == 405


@pytest.fixture
def records(workspace, monkeypatch):
    # The imported fixture creates only pg_temp tables and rolls back its outer
    # transaction. Add directory fields to its temporary compatibility view.
    workspace.exec_driver_sql('''ALTER TABLE site_zones ADD COLUMN notes TEXT, ADD COLUMN deleted_at TIMESTAMPTZ;
        CREATE OR REPLACE TEMP VIEW project_sites AS SELECT id, name, polygon_geojson,
            organization_id, inspection_interval_days, updated_at, deleted_at,
            NULL::TIMESTAMPTZ AS area_released_at, notes FROM site_zones;''')
    monkeypatch.setattr(directory, 'get_connection', lambda: planting._get_connection())
    org, owner = seed(workspace, name='Example group', participants=5)
    other_org, other_owner = seed(workspace, name='Other group', participants=2)
    workspace.execute(text('''INSERT INTO organizations(name, normalized_name, inspection_interval_days, created_at)
        VALUES ('No activity yet', 'no activity yet', 14, CURRENT_TIMESTAMP)'''))
    workspace.execute(text('''UPDATE organization_participants SET participant_username='Ana', username_key='ana',
        current_session_hash='never-expose-session-hash' WHERE planter_id=:id AND slot=1'''), {'id': owner})
    workspace.execute(text("UPDATE organization_participants SET device_key_hash='never-expose-browser-hash' WHERE planter_id=:id AND slot=2"), {'id': owner})
    workspace.execute(text('''INSERT INTO auth_sessions(subject_type, subject_id, participant_slot,
        token_hash, created_at, expires_at, last_seen_at) VALUES ('planter', :id, 3,
        'never-expose-token-hash', CURRENT_TIMESTAMP - INTERVAL '2 days',
        CURRENT_TIMESTAMP - INTERVAL '1 day', CURRENT_TIMESTAMP - INTERVAL '1 day')'''), {'id': owner})
    workspace.execute(text("UPDATE organization_participants SET participant_username='Other', username_key='other' WHERE planter_id=:id AND slot=1"), {'id': other_owner})
    workspace.execute(text('''INSERT INTO site_zones(id, name, organization_id, inspection_interval_days, notes)
        VALUES (1, 'Current site', :org, 14, 'Example notes'), (2, 'Archived site', :org, 14, NULL),
               (3, 'Other site', :other, 14, NULL)'''), {'org': org, 'other': other_org})
    workspace.exec_driver_sql('UPDATE site_zones SET deleted_at=CURRENT_TIMESTAMP WHERE id=2')
    workspace.execute(text('''INSERT INTO planting_schedules(id, organization, organization_id, title,
        contact, notification_email, start_at, end_at, expected_planters, status, appointment_type)
        VALUES (10, 'Example group', :org, 'Earlier planting', 'Earlier contact', 'old@example.org',
            '2026-01-01', '2026-01-02', 100, 'completed', 'tree_planting'),
        (11, 'Example group', :org, 'Latest planting', 'Current contact, 09123456789', 'current@example.org',
            '2026-02-01', '2026-02-02', 8, 'confirmed', 'tree_planting'),
        (12, 'Other group', :other, 'Other planting', 'Other contact', 'other@example.org',
            '2026-02-01', '2026-02-02', 30, 'confirmed', 'tree_planting')'''), {'org': org, 'other': other_org})
    workspace.execute(text('''INSERT INTO activity_access_codes(schedule_id, planter_id, code_digest, encrypted_code)
        VALUES (11, :owner, :digest, 'never-expose-encrypted-code')'''), {'owner': owner, 'digest': 'a' * 64})
    return workspace, org, owner


@pytest.mark.skipif(os.getenv('MANGROVISION_RUN_TEMP_LIKE_TESTS') != '1', reason='Temporary PostgreSQL checks are opt-in.')
class TestDirectoryDatabase:
    def test_contact_and_counts_do_not_sum_historical_participants_or_reserved_places(self, records):
        _, org, _ = records
        rows = directory.list_directory()
        assert len(rows) == 3
        row = next(row for row in rows if row['id'] == org)
        assert row['contact'] == 'Current contact, 09123456789'
        assert row['notification_email'] == 'current@example.org'
        assert row['scheduled_participants'] == 8
        assert row['participant_capacity'] == 5
        assert row['member_count'] == 3  # Named, legacy browser, expired session.
        assert row['project_site_count'] == 1  # Archived site is excluded.
        empty = next(row for row in rows if row['name'] == 'No activity yet')
        assert empty['member_count'] == empty['project_site_count'] == empty['account_count'] == 0
        assert empty['scheduled_participants'] is None

    def test_detail_is_scoped_and_never_exposes_authentication_material_or_modifies_records(self, records):
        db, org, _ = records
        tables = ('organizations', 'planters', 'organization_participants', 'auth_sessions',
                  'site_zones', 'planting_schedules', 'activity_access_codes')
        before = {table: db.execute(text(f'SELECT * FROM {table}')).fetchall() for table in tables}
        detail = directory.organization_detail(org)
        assert len(detail['members']) == 3
        assert {member['participant_number'] for member in detail['members']} == {1, 2, 3}
        assert all(member['username'] != 'Other' for member in detail['members'])
        assert {site['id'] for site in detail['project_sites']} == {1, 2}
        assert {activity['id'] for activity in detail['activities']} == {10, 11}
        assert next(activity for activity in detail['activities'] if activity['id'] == 11)['has_code'] is True
        serialized = json.dumps(detail, default=str)
        for private in ('never-expose', 'password_hash', 'code_digest', 'encrypted_code', 'current_session_hash', 'device_key_hash', 'token_hash'):
            assert private not in serialized
        assert before == {table: db.execute(text(f'SELECT * FROM {table}')).fetchall() for table in tables}
        assert directory.organization_detail(999999) is None

    def test_merged_accounts_and_cancelled_activity_do_not_override_current_directory(self, records):
        db, org, owner = records
        _, merged = seed(db, name='Merged holder', participants=1)
        db.execute(text('UPDATE planters SET organization_id=:org, merged_into_planter_id=:owner WHERE id=:id'), {'org': org, 'owner': owner, 'id': merged})
        db.execute(text("UPDATE organization_participants SET participant_username='Merged', username_key='merged' WHERE planter_id=:id"), {'id': merged})
        db.exec_driver_sql("UPDATE planting_schedules SET status='cancelled', updated_at=CURRENT_TIMESTAMP + INTERVAL '1 day' WHERE id=10")
        row = directory.organization_detail(org)['organization']
        assert row['member_count'] == 3
        assert row['account_count'] == 1
        assert row['contact_activity_title'] == 'Latest planting'
