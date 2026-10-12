"""Real PostgreSQL regression checks, isolated in rollback-only temporary tables."""
import hashlib
import json

import pytest
from sqlalchemy import text

from test_like_appointments_postgres import workspace, confirm
from mangrovision_db import planter_codes as codes, appointment_email as delivery


def seed(db, name='Oton Youth', participants=2, *, pending=False):
    org = db.execute(text('''INSERT INTO organizations(name, normalized_name, inspection_interval_days, created_at, updated_at)
        VALUES (:name, :norm, 14, CURRENT_TIMESTAMP, CURRENT_TIMESTAMP) RETURNING id'''),
                     {'name': name, 'norm': name.lower()}).scalar_one()
    account = db.execute(text('''INSERT INTO planters(full_name, organization_id, username, password_hash, participant_count)
        VALUES (:name, :org, :username, :hash, :count) RETURNING id'''),
        {'name': name, 'org': org, 'username': None if pending else name.lower().replace(' ', '-'),
         'hash': None if pending else 'untouched-legacy-hash', 'count': participants}).scalar_one()
    if not pending:
        db.execute(text('INSERT INTO organization_participants(planter_id, slot) SELECT :id, generate_series(1, :count)'),
                   {'id': account, 'count': participants})
    return org, account


def login(value):
    return codes.authenticate_code(value, address='test-peer')


@pytest.mark.parametrize('separator', ['-', ' ', '\u2010', '\u2011', '\u2012', '\u2013', '\u2014', '\u2015', '\u2212', '\ufe63', '\uff0d'])
def test_mobile_pasted_code_separators_preserve_identity(separator):
    assert codes.normalize_code(f' k7mp{separator}4r9x ') == 'K7MP4R9X'
    with pytest.raises(codes.CodeError):
        codes.normalize_code('K7MP/4R9X')


def test_issuing_existing_codes_preserves_all_records_and_is_idempotent(workspace):
    org, account = seed(workspace)
    identity = 'cd' * 24
    workspace.execute(text('UPDATE organization_participants SET device_key_hash=:hash WHERE planter_id=:id AND slot=2'),
        {'hash': hashlib.sha256(identity.encode()).hexdigest(), 'id': account})
    workspace.execute(text('INSERT INTO planter_assignments(id, planter_id) VALUES (10, :id)'), {'id': account})
    workspace.exec_driver_sql('INSERT INTO planter_assignment_points(id, assignment_id, participant_slot) VALUES (9,10,2)')
    before = {table: workspace.execute(text(f'SELECT * FROM {table}')).fetchall()
              for table in ('planters', 'organization_participants', 'planter_assignments', 'planter_assignment_points')}
    first = codes.manage_codes(account, issue=True)
    assert len(first['codes']) == 2
    assert first == codes.manage_codes(account, issue=True)
    assert first == codes.manage_codes(account)
    for table, records in before.items():
        assert workspace.execute(text(f'SELECT * FROM {table}')).fetchall() == records
    assert codes.normalize_code(first['codes'][0]['code']) != codes.normalize_code(first['codes'][1]['code'])
    for row in workspace.execute(text('SELECT code_digest, encrypted_code FROM planter_access_codes')):
        assert first['codes'][0]['code'] not in row.encrypted_code
        assert row.code_digest != hashlib.sha256(codes.normalize_code(first['codes'][0]['code']).encode()).hexdigest()


def test_code_uses_exact_slot_and_stays_valid_across_logins(workspace):
    _, account = seed(workspace)
    entries = codes.manage_codes(account, issue=True)['codes']
    planter, token = login(entries[1]['code'].lower().replace('-', ' '))
    assert planter['id'] == account and planter['participant_slot'] == 2
    assert token
    assert workspace.execute(text('SELECT participant_slot FROM auth_sessions')).scalar_one() == 2
    assert workspace.execute(text('SELECT last_login FROM planters WHERE id=:id'), {'id': account}).scalar_one() is not None
    assert workspace.execute(text('SELECT device_key_hash FROM organization_participants WHERE slot=1')).scalar_one() is None
    assert login(entries[1]['code'])[0]['participant_slot'] == 2


def test_existing_device_bindings_do_not_restrict_code_sign_in(workspace):
    _, account = seed(workspace)
    first, second = codes.manage_codes(account, issue=True)['codes']
    for slot, key in ((1, 'ab' * 24), (2, 'cd' * 24)):
        workspace.execute(text('UPDATE organization_participants SET device_key_hash=:hash WHERE planter_id=:id AND slot=:slot'),
            {'hash': hashlib.sha256(key.encode()).hexdigest(), 'id': account, 'slot': slot})
    before = workspace.execute(text('SELECT * FROM organization_participants ORDER BY slot')).fetchall()
    assert login(second['code'])[0]['participant_slot'] == 2
    assert login(first['code'])[0]['participant_slot'] == 1
    assert workspace.execute(text('SELECT * FROM organization_participants ORDER BY slot')).fetchall() == before


def test_fresh_browsers_resume_same_points_when_all_ten_participants_have_signed_in(workspace):
    import sys
    from pathlib import Path
    from fastapi import FastAPI
    from fastapi.testclient import TestClient

    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
    from api.routes.planter_auth import router

    _, account = seed(workspace, participants=10)
    entries = codes.manage_codes(account, issue=True)['codes']
    workspace.exec_driver_sql("ALTER TABLE pg_temp.planter_assignment_points ADD COLUMN status TEXT DEFAULT 'pending'")
    workspace.execute(text('INSERT INTO planter_assignments(id, planter_id) VALUES (10, :id)'), {'id': account})
    workspace.exec_driver_sql('''INSERT INTO planter_assignment_points(id, assignment_id, participant_slot, status)
        SELECT n, 10, ((n - 1) / 10) + 1, CASE WHEN n <= 3 THEN 'completed' ELSE 'pending' END
        FROM generate_series(1, 100) AS n''')
    before = workspace.execute(text('SELECT * FROM planter_assignment_points ORDER BY id')).fetchall()
    app = FastAPI()
    app.include_router(router, prefix='/api/planter-auth')
    for index, entry in enumerate([*entries, entries[0], entries[0]]):
        # Each cookie jar has no existing device identity or recovery code.
        with TestClient(app, base_url='https://field.example.test') as client:
            body = {'access_code': entry['code']}
            if index == 11:
                # Older cached clients' browser/recovery fields cannot override
                # the code or block a valid participant from signing in.
                body.update(device_key='old-browser-key', known_device_keys=[], resume_device=True)
            response = client.post('/api/planter-auth/login', json=body)
            assert response.status_code == 200
            assert response.json()['planter']['participant_slot'] == entry['participant_slot']
            assert response.headers.get('set-cookie')
            assert 'device_key' not in response.json()
    slots = workspace.execute(text('SELECT participant_slot FROM auth_sessions ORDER BY id')).scalars().all()
    assert slots == [*range(1, 11), 1, 1]
    assert workspace.execute(text('SELECT count(*) FROM organization_participants')).scalar_one() == 10
    assert workspace.execute(text('SELECT * FROM planter_assignment_points ORDER BY id')).fetchall() == before


def test_different_codes_select_distinct_participants_without_device_binding(workspace):
    _, account = seed(workspace)
    first, second = codes.manage_codes(account, issue=True)['codes']
    assert login(first['code'])[0]['participant_slot'] == 1
    assert login(second['code'])[0]['participant_slot'] == 2
    assert login(first['code'])[0]['participant_slot'] == 1
    assert workspace.execute(text('SELECT count(*) FROM organization_participants WHERE device_key_hash IS NOT NULL')).scalar_one() == 0


def test_unknown_inactive_revoked_and_merged_codes_never_bind_a_device(workspace):
    _, account = seed(workspace)
    entry = codes.manage_codes(account, issue=True)['codes'][0]
    with pytest.raises(codes.CodeError, match='incorrect or no longer valid'):
        login('ZZZZ-ZZZZ')
    workspace.execute(text("UPDATE planters SET status='inactive' WHERE id=:id"), {'id': account})
    with pytest.raises(codes.CodeError, match='reactivate'):
        login(entry['code'])
    workspace.execute(text("UPDATE planters SET status='active', merged_into_planter_id=99 WHERE id=:id"), {'id': account})
    with pytest.raises(codes.CodeError, match='not found'):
        login(entry['code'])
    workspace.execute(text('UPDATE planters SET merged_into_planter_id=NULL WHERE id=:id'), {'id': account})
    codes.manage_codes(account, slot=1, action='revoke')
    with pytest.raises(codes.CodeError, match='incorrect or no longer valid'):
        login(entry['code'])
    assert workspace.execute(text('SELECT count(*) FROM auth_sessions')).scalar_one() == 0
    assert workspace.execute(text('SELECT count(*) FROM organization_participants WHERE device_key_hash IS NOT NULL')).scalar_one() == 0


def test_replacement_revokes_only_that_participants_sessions_and_preserves_device(workspace):
    _, account = seed(workspace)
    first, second = codes.manage_codes(account, issue=True)['codes']
    login(first['code'])
    login(first['code'])
    login(second['code'])
    before = workspace.execute(text('SELECT * FROM organization_participants ORDER BY slot')).fetchall()
    changed = codes.manage_codes(account, slot=1, action='regenerate')['codes']
    assert changed[0]['code'] != first['code'] and changed[1] == second
    with pytest.raises(codes.CodeError, match='incorrect or no longer valid'):
        login(first['code'])
    sessions = workspace.execute(text('SELECT participant_slot, revoked_at FROM auth_sessions ORDER BY id')).fetchall()
    assert sessions[0].revoked_at and sessions[1].revoked_at and sessions[2].revoked_at is None
    assert workspace.execute(text('SELECT * FROM organization_participants ORDER BY slot')).fetchall() == before
    assert login(changed[0]['code'])[0]['participant_slot'] == 1


def test_issue_does_not_reactivate_a_revoked_code(workspace):
    _, account = seed(workspace)
    codes.manage_codes(account, issue=True)
    codes.manage_codes(account, slot=1, action='revoke')
    entry = codes.manage_codes(account, issue=True)['codes'][0]
    assert entry['revoked'] is True and entry['code'] is None


def test_same_participant_number_in_other_organization_cannot_change_code_ownership(workspace):
    _, account = seed(workspace, name='First group')
    first, second = codes.manage_codes(account, issue=True)['codes']
    _, other_account = seed(workspace, name='Second group')
    other = codes.manage_codes(other_account, issue=True)['codes'][0]
    assert login(other['code'])[0]['id'] == other_account
    planter, _ = login(first['code'])
    assert planter['id'] == account and planter['participant_slot'] == 1
    assert login(second['code'])[0]['participant_slot'] == 2


def test_failed_session_creation_preserves_participants_and_last_login(workspace, monkeypatch):
    import planting_database as planting
    _, account = seed(workspace)
    entry = codes.manage_codes(account, issue=True)['codes'][0]
    monkeypatch.setattr(planting, '_create_session', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('Session failure')))
    with pytest.raises(RuntimeError, match='Session failure'):
        login(entry['code'])
    assert workspace.execute(text('SELECT count(*) FROM organization_participants WHERE device_key_hash IS NOT NULL')).scalar_one() == 0
    assert workspace.execute(text('SELECT count(*) FROM auth_sessions')).scalar_one() == 0
    assert workspace.execute(text('SELECT last_login FROM planters WHERE id=:id'), {'id': account}).scalar_one() is None
    assert workspace.execute(text('SELECT count(*) FROM planter_auth_limits')).scalar_one() == 2


def test_durable_guess_limit_stops_retries_without_session_or_slot_writes(workspace):
    for _ in range(20):
        with pytest.raises(codes.CodeError) as error:
            login('ZZZZ-ZZZZ')
        assert error.value.status == 401
    with pytest.raises(codes.CodeError) as error:
        login('ZZZZ-ZZZZ')
    assert error.value.status == 429 and error.value.retry_after > 0
    assert workspace.execute(text('SELECT count(*) FROM auth_sessions')).scalar_one() == 0
    workspace.exec_driver_sql("UPDATE planter_auth_limits SET window_start=CURRENT_TIMESTAMP - INTERVAL '11 minutes'")
    with pytest.raises(codes.CodeError) as error:
        login('ZZZZ-ZZZZ')
    assert error.value.status == 401


def test_existing_organization_uses_same_codes_for_later_bookings(workspace):
    first = confirm(workspace, organization='Oton Youth', participants=2)
    account = workspace.execute(text('SELECT id FROM planters')).scalar_one()
    original = codes.manage_codes(account)['codes']
    second = confirm(workspace, organization='Oton Youth', participants=3)
    assert first['schedule_id'] != second['schedule_id']
    assert codes.manage_codes(account)['codes'] == original


def test_replaced_codes_refresh_pending_emails_and_sending_blocks_changes(workspace):
    confirm(workspace, participants=2)
    account = workspace.execute(text('SELECT id FROM planters')).scalar_one()
    original = codes.manage_codes(account)['codes'][0]['code']
    current = codes.manage_codes(account, slot=1, action='regenerate')['codes'][0]['code']
    encrypted = workspace.execute(text('SELECT encrypted_message FROM like_appointment_emails')).scalar_one()
    message = json.loads(delivery._cipher().decrypt(encrypted.encode()))
    assert current in message['body'] and original not in message['body']
    assert 'Username:' not in message['body'] and 'Password:' not in message['body']
    workspace.exec_driver_sql("UPDATE like_appointment_emails SET status='sending'")
    with pytest.raises(codes.CodeError, match='being delivered'):
        codes.manage_codes(account, slot=1, action='regenerate')
    assert codes.manage_codes(account)['codes'][0]['code'] == current


def test_reserved_account_creation_keeps_original_owner_and_allocates_once(workspace):
    org, account = seed(workspace, pending=True)
    result = codes.create_code_account({'full_name': 'Oton Youth', 'organization_id': org, 'participant_count': 3}, 1)
    assert result['planter_id'] == account and len(result['codes']) == 3
    assert workspace.execute(text('SELECT count(*) FROM planters')).scalar_one() == 1
    with pytest.raises(ValueError, match='already has an account'):
        codes.create_code_account({'full_name': 'Oton Youth', 'organization_id': org, 'participant_count': 4}, 1)
    assert len(codes.manage_codes(account)['codes']) == 3


def test_reviewed_existing_conversion_issues_66_codes_without_changing_accounts_or_devices(workspace):
    from scripts.issue_organization_codes import issue_existing
    for index in range(9):
        seed(workspace, name=f'Existing group {index}', participants=2 if index < 3 else 10)
    before = {table: workspace.execute(text(f'SELECT * FROM {table} ORDER BY 1, 2')).fetchall()
              for table in ('planters', 'organization_participants')}
    result = issue_existing(codes.get_connection(), expected_accounts=9, expected_slots=66)
    assert len(result) == 9 and sum(len(account['codes']) for account in result) == 66
    for table, records in before.items():
        assert workspace.execute(text(f'SELECT * FROM {table} ORDER BY 1, 2')).fetchall() == records
    assert len({entry['code'] for account in result for entry in account['codes']}) == 66


def test_conversion_rejects_a_changed_review_before_issuing_any_codes(workspace):
    from scripts.issue_organization_codes import issue_existing
    seed(workspace)
    with pytest.raises(ValueError, match='changed'):
        issue_existing(codes.get_connection(), expected_accounts=9, expected_slots=66)
    assert workspace.execute(text('SELECT count(*) FROM planter_access_codes')).scalar_one() == 0
