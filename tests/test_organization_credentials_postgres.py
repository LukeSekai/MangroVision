"""Login conversion uses only session-local tables; all writes roll back."""
import hashlib
import json
import os

import pytest
from argon2 import PasswordHasher
from sqlalchemy import text

import planting_database as planting
from mangrovision_db.organization_accounts import claim_participant_device, claim_participant_slot
from mangrovision_db.organization_credentials import apply_conversion, prepare_conversion, public_preview
from test_like_appointments_postgres import workspace  # noqa: F401

pytestmark = pytest.mark.skipif(os.getenv('MANGROVISION_RUN_TEMP_LIKE_TESTS') != '1', reason='Temporary PostgreSQL checks are opt-in.')


@pytest.fixture
def accounts(workspace, monkeypatch):
    # Deliberately shadow auth_sessions too, so preservation checks cannot read
    # or modify actual participant sessions.
    workspace.exec_driver_sql('CREATE TEMP TABLE auth_sessions(id BIGINT, subject_type TEXT, subject_id BIGINT, revoked_at TIMESTAMPTZ)')
    def audit(conn, **fields):
        conn.execute('INSERT INTO activity_logs(action, organization_id, details) VALUES (?, ?, CAST(? AS jsonb))',
                     (fields['action'], fields['organization_id'], json.dumps(fields['details'])))
    monkeypatch.setattr(planting, 'append_activity', audit)
    return workspace


def seed(conn, name='Oton Youth', username='old-oton', *, status='active', merged=None):
    org = conn.execute(text('''INSERT INTO organizations(name, normalized_name, inspection_interval_days, created_at)
        VALUES (:name, :normalized, 14, CURRENT_TIMESTAMP) RETURNING id'''),
                       {'name': name, 'normalized': f'{name.lower()}-{username}'}).scalar_one()
    planter = conn.execute(text('''INSERT INTO planters(organization_id, full_name, username, password_hash,
        status, participant_count, merged_into_planter_id) VALUES (:org, :name, :username, :password, :status, 2, :merged) RETURNING id'''),
        {'org': org, 'name': name, 'username': username, 'password': planting._hash_password('Previous-test-password') if username else None,
         'status': status, 'merged': merged}).scalar_one()
    conn.execute(text('INSERT INTO organization_participants VALUES (:id, 1, :device), (:id, 2, NULL)'),
                 {'id': planter, 'device': hashlib.sha256(('ab' * 24).encode()).hexdigest()})
    return planter


def connect_plan():
    conn = planting._get_connection()
    return conn, prepare_conversion(conn)


def test_conversion_preserves_account_records_allocations_devices_and_sessions(accounts):
    planter = seed(accounts)
    accounts.execute(text('INSERT INTO planter_assignments VALUES (10, :id, 3, CURRENT_DATE)'), {'id': planter})
    accounts.exec_driver_sql('INSERT INTO planter_assignment_points VALUES (100, 10, NULL, 1, 1), (101, 10, NULL, 2, 2)')
    accounts.exec_driver_sql("INSERT INTO planting_events(assignment_id, planted_at) VALUES (10, CURRENT_TIMESTAMP)")
    accounts.execute(text("INSERT INTO auth_sessions VALUES (1, 'planter', :id, NULL)"), {'id': planter})
    tables = ('organization_participants', 'planter_assignments', 'planter_assignment_points', 'planting_events', 'auth_sessions')
    before = {table: accounts.exec_driver_sql(f'SELECT * FROM {table} ORDER BY 1, 2').fetchall() for table in tables}
    old = dict(accounts.execute(text('SELECT * FROM planters')).mappings().one())
    conn, plan = connect_plan()
    assert apply_conversion(conn, plan) == {'updated': 1, 'already_applied': False}
    conn.commit()
    conn.close()
    current = dict(accounts.execute(text('SELECT * FROM planters')).mappings().one())
    assert current['username'] == 'oton-youth'
    assert plan['accounts'][0]['password'].startswith('OtonYouth-')
    assert PasswordHasher().verify(current['password_hash'], plan['accounts'][0]['password'])
    assert planting.verify_password(current['password_hash'], 'Previous-test-password')[0] is False
    assert {key: value for key, value in current.items() if key not in {'username', 'password_hash'}} == {
        key: value for key, value in old.items() if key not in {'username', 'password_hash'}}
    assert {table: accounts.exec_driver_sql(f'SELECT * FROM {table} ORDER BY 1, 2').fetchall() for table in tables} == before
    audit = accounts.execute(text('SELECT details FROM activity_logs')).scalar_one()
    assert 'password' not in json.dumps(audit).lower()
    assert plan['accounts'][0]['password'] not in json.dumps(public_preview(plan))


def test_repeat_apply_is_idempotent_and_preparing_again_skips_converted(accounts):
    seed(accounts)
    conn, plan = connect_plan()
    apply_conversion(conn, plan)
    conn.commit()
    conn.close()
    password_hash = accounts.execute(text('SELECT password_hash FROM planters')).scalar_one()
    conn = planting._get_connection()
    assert apply_conversion(conn, plan) == {'updated': 0, 'already_applied': True}
    assert prepare_conversion(conn)['accounts'] == []
    assert accounts.execute(text('SELECT password_hash FROM planters')).scalar_one() == password_hash
    conn.close()


def test_pending_and_merged_are_skipped_inactive_status_is_preserved(accounts):
    primary = seed(accounts)
    seed(accounts, 'Merged group', 'old-merged', status='inactive', merged=primary)
    seed(accounts, 'Pending group', None)
    inactive = seed(accounts, 'Inactive group', 'old-inactive', status='inactive')
    conn, plan = connect_plan()
    assert len(plan['accounts']) == 2 and len(plan['skipped']) == 2
    apply_conversion(conn, plan)
    conn.commit()
    conn.close()
    assert accounts.execute(text('SELECT status FROM planters WHERE id=:id'), {'id': inactive}).scalar_one() == 'inactive'
    assert accounts.execute(text('SELECT username FROM planters WHERE merged_into_planter_id=:id'), {'id': primary}).scalar_one() == 'old-merged'


def test_usernames_resolve_case_insensitive_collisions_without_self_suffix(accounts):
    seed(accounts, 'Canonical group', 'canonical-group')
    seed(accounts, 'Oton Youth', 'oton-youth')
    seed(accounts, 'Oton Youth', 'second-login')
    conn, plan = connect_plan()
    assert [entry['username'] for entry in plan['accounts']] == ['canonical-group', 'oton-youth', 'oton-youth-2']
    apply_conversion(conn, plan)
    conn.commit()
    conn.close()


def test_changed_account_stops_entire_batch(accounts):
    first = seed(accounts)
    second = seed(accounts, 'Another group', 'another-login')
    conn, plan = connect_plan()
    conn.close()
    accounts.execute(text("UPDATE planters SET participant_count = 3 WHERE id=:id"), {'id': second})
    conn = planting._get_connection()
    with pytest.raises(ValueError, match='changed after preparation'):
        apply_conversion(conn, plan)
    conn.rollback()
    conn.close()
    assert accounts.execute(text('SELECT username FROM planters WHERE id=:id'), {'id': first}).scalar_one() == 'old-oton'
    assert accounts.execute(text('SELECT count(*) FROM activity_logs')).scalar_one() == 0


@pytest.mark.parametrize('status', ['pending', 'sending'])
def test_unsent_booking_prevents_password_becoming_stale(accounts, status):
    from test_like_appointments_postgres import values, request_id
    from mangrovision_db import like_appointments
    body = values()
    like_appointments.submit_appointment(body, 'test-peer')
    like_appointments.review_appointment(request_id(accounts), {'action': 'confirmed', 'contacted': True}, 1)
    accounts.execute(text('UPDATE like_appointment_emails SET status=:status'), {'status': status})
    conn, plan = connect_plan()
    previous = accounts.execute(text('SELECT username, password_hash FROM planters')).one()
    with pytest.raises(ValueError, match='unsent booking email'):
        apply_conversion(conn, plan)
    conn.rollback()
    conn.close()
    assert accounts.execute(text('SELECT username, password_hash FROM planters')).one() == previous


def test_renamed_login_recovers_the_same_slot_from_organization_specific_storage(accounts):
    planter = seed(accounts)
    before = accounts.execute(text('SELECT * FROM organization_participants ORDER BY slot')).fetchall()
    identity = 'ab' * 24
    assert claim_participant_device(planter, 'cd' * 24, known_device_keys=[identity, 'ef' * 24]) == (1, identity)
    assert accounts.execute(text('SELECT * FROM organization_participants ORDER BY slot')).fetchall() == before
    assert claim_participant_slot(planter, identity) == 1


def test_unknown_other_group_device_keys_cannot_replace_a_participant_identity(accounts):
    planter = seed(accounts)
    assert claim_participant_device(planter, 'cd' * 24, known_device_keys=['ef' * 24]) == (2, 'cd' * 24)
    assert accounts.execute(text('SELECT device_key_hash FROM organization_participants WHERE slot=1')).scalar_one() == hashlib.sha256(('ab' * 24).encode()).hexdigest()


def test_multiple_old_identities_require_recovery_instead_of_choosing_wrong_slot(accounts):
    planter = seed(accounts)
    second = 'cd' * 24
    accounts.execute(text('UPDATE organization_participants SET device_key_hash=:device WHERE slot=2'),
                     {'device': hashlib.sha256(second.encode()).hexdigest()})
    with pytest.raises(ValueError, match='more than one participant identity'):
        claim_participant_device(planter, second, known_device_keys=['ab' * 24, second])
    assert claim_participant_device(planter, second, resume_device=True, known_device_keys=['ab' * 24]) == (2, second)


def test_failed_audit_rolls_back_all_credential_changes(accounts, monkeypatch):
    seed(accounts)
    previous = accounts.execute(text('SELECT username, password_hash FROM planters')).one()
    def fail(*args, **kwargs):
        raise RuntimeError('Simulated audit failure')
    monkeypatch.setattr(planting, 'append_activity', fail)
    conn, plan = connect_plan()
    with pytest.raises(RuntimeError, match='audit failure'):
        apply_conversion(conn, plan)
    conn.rollback()
    conn.close()
    assert accounts.execute(text('SELECT username, password_hash FROM planters')).one() == previous
