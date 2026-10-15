"""Shared access checks run entirely in session-local tables with outer rollback."""
from datetime import datetime, timedelta, timezone

import pytest
from sqlalchemy import text

import planting_database as planting
from mangrovision_db import group_access as group, planter_codes as old_codes, appointment_email as email
from mangrovision_db.request_context import planter_session_token
from test_like_appointments_postgres import workspace, values, request_id
from test_planter_codes_postgres import seed


def activity(db, participants=10, name='Test group'):
    org, owner = seed(db, name=name, participants=participants)
    start = datetime.now(timezone.utc) - timedelta(minutes=15)
    sid = db.execute(text('''INSERT INTO planting_schedules(organization, organization_id, title,
        start_at, end_at, expected_planters, status, appointment_type)
        VALUES (:name, :org, 'Tree planting', :start, :end, :count, 'confirmed', 'tree_planting') RETURNING id'''),
        {'name': name, 'org': org, 'start': start, 'end': start + timedelta(hours=3), 'count': participants}).scalar_one()
    return owner, sid, group.manage_activity_code(sid, action='issue')['access']['code']


def join(code, name='Ana', browser=1, **options):
    return group.authenticate_group(code, name, f'{browser:048x}', address='shared-proxy', **options)


def current(token):
    return planting.get_planter_by_session_token(token)


def test_one_code_for_100_planters_and_no_shared_code_throttling(workspace):
    owner, sid, code = activity(workspace, participants=100)
    for number in range(1, 101):
        participant, token = join(code, f'Planter{number}', number, mode='first')
        assert participant['participant_slot'] == number
    assert workspace.execute(text('SELECT count(*) FROM activity_access_codes')).scalar_one() == 1
    assert workspace.execute(text('SELECT count(*) FROM auth_sessions WHERE revoked_at IS NULL')).scalar_one() == 100
    assert workspace.execute(text('SELECT count(*) FROM organization_participants')).scalar_one() == 100
    with pytest.raises(group.CodeError, match='No unused|full'):
        join(code, 'Planter101', 101, mode='first')
    assert join(code, 'Planter1', 1)[0]['participant_slot'] == 1


def test_logout_and_other_tabs_restore_same_slot_and_keep_allocations(workspace):
    owner, sid, code = activity(workspace)
    workspace.execute(text('INSERT INTO planter_assignments(id, planter_id, planting_schedule_id) VALUES (10,:owner,:sid)'), {'owner': owner, 'sid': sid})
    workspace.exec_driver_sql('INSERT INTO planter_assignment_points(id,assignment_id,participant_slot) VALUES (1,10,1),(2,10,2)')
    before = workspace.execute(text('SELECT * FROM planter_assignment_points ORDER BY id')).fetchall()
    first, token = join(code, mode='first')
    planting.revoke_planter_session(token)
    assert current(token) is None
    returning, next_token = join(code, ' ana ')
    assert first['participant_slot'] == returning['participant_slot'] == 1
    assert current(next_token)['participant_username'] == 'Ana'
    assert workspace.execute(text('SELECT * FROM planter_assignment_points ORDER BY id')).fetchall() == before
    with pytest.raises(group.GroupAccessError) as failure:
        join(code, 'DifferentName', mode='first')
    assert failure.value.kind == 'already_joined'
    assert workspace.execute(text('SELECT count(*) FROM organization_participants WHERE username_key IS NOT NULL')).scalar_one() == 1


def test_username_is_unique_case_insensitively_and_missing_return_never_registers(workspace):
    _, _, code = activity(workspace)
    join(code, mode='first')
    with pytest.raises(group.CodeError, match='already used'):
        join(code, 'aNA', 2, mode='first')
    for name in ('MissingUser', '', 'a b', 'ab', 'a' * 33):
        with pytest.raises(group.CodeError):
            join(code, name, 2)
    assert workspace.execute(text('SELECT count(*) FROM auth_sessions')).scalar_one() == 1


def test_switch_needs_confirmation_and_old_browser_immediately_loses_access(workspace):
    _, _, code = activity(workspace)
    _, old_token = join(code, mode='first')
    with pytest.raises(group.GroupAccessError) as conflict:
        join(code, browser=2)
    assert conflict.value.kind == 'session_in_use'
    assert current(old_token) is not None
    participant, new_token = join(code, browser=2, switch_token=conflict.value.details['switch_token'])
    assert participant['participant_slot'] == 1
    assert current(old_token) is None and current(new_token) is not None
    assert workspace.execute(text('SELECT count(*) FROM auth_sessions WHERE revoked_at IS NULL')).scalar_one() == 1
    planting.revoke_planter_session(old_token)  # A stale tab's logout cannot revoke the replacement.
    assert current(new_token) is not None


def test_stale_or_wrong_browser_switch_confirmation_cannot_replace_latest_session(workspace):
    _, _, code = activity(workspace)
    join(code, mode='first')
    with pytest.raises(group.GroupAccessError) as first:
        join(code, browser=2)
    proof = first.value.details['switch_token']
    with pytest.raises(group.GroupAccessError) as wrong_browser:
        join(code, browser=3, switch_token=proof)
    assert wrong_browser.value.kind == 'session_in_use'
    join(code, browser=3, switch_token=wrong_browser.value.details['switch_token'])
    with pytest.raises(group.GroupAccessError) as stale:
        join(code, browser=2, switch_token=proof)
    assert stale.value.kind == 'session_in_use'


def test_failed_session_creation_rolls_back_username_browser_and_previous_session(workspace, monkeypatch):
    _, _, code = activity(workspace)
    original = planting._create_session
    monkeypatch.setattr(planting, '_create_session', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('Simulated failure')))
    with pytest.raises(RuntimeError):
        join(code, mode='first')
    assert workspace.execute(text('SELECT count(*) FROM participant_browser_bindings')).scalar_one() == 0
    assert workspace.execute(text('SELECT count(*) FROM organization_participants WHERE username_key IS NOT NULL')).scalar_one() == 0
    monkeypatch.setattr(planting, '_create_session', original)
    _, token = join(code, mode='first')
    with pytest.raises(group.GroupAccessError) as conflict:
        join(code, browser=2)
    monkeypatch.setattr(planting, '_create_session', lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError('Simulated failure')))
    with pytest.raises(RuntimeError):
        join(code, browser=2, switch_token=conflict.value.details['switch_token'])
    assert current(token) is not None


@pytest.mark.parametrize('change', ['expired', 'future', 'cancelled', 'completed', 'revoked'])
def test_activity_window_and_status_end_existing_and_new_access(workspace, change):
    _, sid, code = activity(workspace)
    _, token = join(code, mode='first')
    if change == 'expired':
        workspace.execute(text("UPDATE planting_schedules SET start_at=clock_timestamp()-interval '4 hours', end_at=clock_timestamp()-interval '1 hour' WHERE id=:id"), {'id': sid})
    elif change == 'future':
        workspace.execute(text("UPDATE planting_schedules SET start_at=clock_timestamp()+interval '1 hour', end_at=clock_timestamp()+interval '4 hours' WHERE id=:id"), {'id': sid})
    elif change == 'revoked':
        group.manage_activity_code(sid, action='revoke')
    else:
        workspace.execute(text('UPDATE planting_schedules SET status=:state WHERE id=:id'), {'state': change, 'id': sid})
    assert current(token) is None
    with pytest.raises(group.CodeError):
        join(code)
    assert workspace.execute(text('SELECT participant_username FROM organization_participants WHERE slot=1')).scalar_one() == 'Ana'


def test_expiry_uses_schedule_and_extension_restores_existing_session(workspace):
    _, sid, code = activity(workspace)
    _, token = join(code, mode='first')
    workspace.execute(text("UPDATE planting_schedules SET start_at=clock_timestamp()-interval '4 hours', end_at=clock_timestamp()-interval '1 hour' WHERE id=:id"), {'id': sid})
    assert current(token) is None
    workspace.execute(text("UPDATE planting_schedules SET end_at=clock_timestamp()+interval '1 hour' WHERE id=:id"), {'id': sid})
    assert current(token)['participant_slot'] == 1


def test_existing_personal_code_claims_original_slot_without_moving_points(workspace):
    owner, sid, code = activity(workspace, participants=2)
    entries = old_codes.manage_codes(owner, issue=True)['codes']
    old_codes.authenticate_code(entries[1]['code'])
    workspace.execute(text('INSERT INTO planter_assignments(id,planter_id,planting_schedule_id) VALUES(10,:owner,:sid)'), {'owner': owner, 'sid': sid})
    workspace.exec_driver_sql('INSERT INTO planter_assignment_points(id,assignment_id,participant_slot) VALUES(9,10,2)')
    participant, _ = join(code, mode='first', previous_code=entries[1]['code'])
    assert participant['participant_slot'] == 2
    assert workspace.execute(text('SELECT participant_slot FROM planter_assignment_points WHERE id=9')).scalar_one() == 2
    assert workspace.execute(text('SELECT count(*) FROM organization_participants')).scalar_one() == 2


def test_legacy_cookie_proves_original_participant_when_choosing_username(workspace):
    owner, _, code = activity(workspace)
    entry = old_codes.manage_codes(owner, issue=True)['codes'][3]
    _, old_token = old_codes.authenticate_code(entry['code'])
    reset = planter_session_token.set(old_token)
    try:
        participant, token = join(code, mode='first')
        assert participant['participant_slot'] == 4
        assert current(token)['participant_slot'] == 4
    finally:
        planter_session_token.reset(reset)


def test_activity_capacity_limits_new_people_not_logins_and_later_bookings_preserve_names(workspace):
    owner, sid, code = activity(workspace)
    workspace.execute(text('UPDATE planting_schedules SET expected_planters=1 WHERE id=:id'), {'id': sid})
    join(code, mode='first')
    with pytest.raises(group.CodeError, match='full'):
        join(code, 'Ben', 2, mode='first')
    assert join(code)[0]['participant_slot'] == 1
    later = dict(workspace.execute(text('SELECT * FROM planting_schedules WHERE id=:id'), {'id': sid}).mappings().one())
    later.pop('id')
    later_id = workspace.execute(text('''INSERT INTO planting_schedules(organization,organization_id,title,start_at,end_at,expected_planters,status,appointment_type)
        SELECT organization,organization_id,title,start_at,end_at,20,status,appointment_type FROM planting_schedules WHERE id=:id RETURNING id'''), {'id': sid}).scalar_one()
    next_code = group.manage_activity_code(later_id, action='issue')['access']['code']
    assert next_code != code
    assert join(next_code)[0]['participant_slot'] == 1
    assert workspace.execute(text('SELECT participant_count FROM planters WHERE id=:owner'), {'owner': owner}).scalar_one() == 20
    assert workspace.execute(text('SELECT count(*) FROM organization_participants')).scalar_one() == 20


def test_write_guard_rejects_switched_expired_other_participant_and_other_activity(workspace):
    owner, sid, code = activity(workspace)
    _, token = join(code, mode='first')
    workspace.execute(text('INSERT INTO planter_assignments(id,planter_id,planting_schedule_id) VALUES(10,:owner,:sid),(11,:owner,NULL)'), {'owner': owner, 'sid': sid})
    workspace.exec_driver_sql('INSERT INTO planter_assignment_points(id,assignment_id,participant_slot) VALUES(1,10,1),(2,10,2),(3,11,1)')
    reset = planter_session_token.set(token)
    try:
        conn = group.get_connection()
        try:
            group.guard_point_write(conn, owner, 1, [1])
            conn.commit()
        finally:
            conn.close()
        for points in ([2], [3]):
            conn = group.get_connection()
            try:
                with pytest.raises(group.CodeError, match='Only points'):
                    group.guard_point_write(conn, owner, 1, points)
            finally:
                conn.close()
        with pytest.raises(group.GroupAccessError) as conflict:
            join(code, browser=2)
        join(code, browser=2, switch_token=conflict.value.details['switch_token'])
        conn = group.get_connection()
        try:
            with pytest.raises(group.CodeError, match='ended or moved'):
                group.guard_point_write(conn, owner, 1, [1])
        finally:
            conn.close()
    finally:
        planter_session_token.reset(reset)


def test_booking_email_has_exactly_one_shared_code_and_reuses_it(workspace):
    from mangrovision_db.like_appointments import submit_appointment, review_appointment
    import json
    submit_appointment(values(participants=100), 'test')
    rid = request_id(workspace)
    result = review_appointment(rid, {'action': 'confirmed', 'contacted': True}, 1)
    message = json.loads(email._cipher().decrypt(workspace.execute(text('SELECT encrypted_message FROM like_appointment_emails')).scalar_one().encode()))
    assert message['body'].count('Organization code:') == 1
    assert 'Participant 1:' not in message['body'] and 'Participant 100:' not in message['body']
    assert 'username' in message['body'] and 'No password or PIN' in message['body']
    assert 'Organization planting access' in message['html_body']
    access = group.manage_activity_code(result['schedule_id'])['access']
    assert access['code'] in message['body']
    assert workspace.execute(text('SELECT count(*) FROM planter_access_codes')).scalar_one() == 0


def test_manual_planting_email_contains_the_same_one_code(workspace):
    from mangrovision_db.schedule_notifications import activity_message
    _, sid, code = activity(workspace, participants=100)
    schedule = dict(workspace.execute(text('SELECT * FROM planting_schedules WHERE id=:id'), {'id': sid}).mappings().one())
    # Compat rows used by the service serialize timestamps before rendering.
    schedule['start_at'] = schedule['start_at'].isoformat()
    schedule['end_at'] = schedule['end_at'].isoformat()
    conn = group.get_connection()
    try:
        access = group.activity_code(conn, sid)
        message = activity_message(None, schedule, 'confirmation', access=access)
        assert message['body'].count('Organization code:') == 1 and code in message['html_body']
        assert 'Organization planting access' in message['html_body']
        cancelled = activity_message(schedule, {**schedule, 'status': 'cancelled'}, 'cancellation', access=access)
        assert code not in cancelled['body'] and code not in cancelled['html_body']
        assert 'Open planter workspace' not in cancelled['html_body']
    finally:
        conn.close()


def test_code_replacement_refreshes_pending_email_and_preserves_username_and_progress(workspace):
    from mangrovision_db.like_appointments import submit_appointment, review_appointment
    import json
    body = values()
    submit_appointment(body, 'test')
    result = review_appointment(request_id(workspace), {'action': 'confirmed', 'contacted': True}, 1)
    original = group.manage_activity_code(result['schedule_id'])['access']['code']
    replaced = group.manage_activity_code(result['schedule_id'], action='replace')['access']['code']
    assert original != replaced
    message = json.loads(email._cipher().decrypt(workspace.execute(text('SELECT encrypted_message FROM like_appointment_emails')).scalar_one().encode()))
    assert replaced in message['body'] and original not in message['body']
    workspace.exec_driver_sql("UPDATE like_appointment_emails SET status='sending'")
    with pytest.raises(group.CodeError, match='being delivered'):
        group.manage_activity_code(result['schedule_id'], action='revoke')


def test_staff_cannot_request_another_organizations_activity(workspace):
    _, sid, _ = activity(workspace)
    with pytest.raises(group.CodeError, match='another organization'):
        group.manage_activity_code(sid, organization_id=999)


def test_new_private_tables_have_rls_and_no_public_api_role_grants(workspace):
    for table in ('activity_access_codes', 'participant_browser_bindings'):
        assert workspace.execute(text('SELECT relrowsecurity FROM pg_class WHERE oid=to_regclass(:table)'), {'table': f'pg_temp.{table}'}).scalar_one()
        for role in ('anon', 'authenticated'):
            exists = workspace.execute(text('SELECT 1 FROM pg_roles WHERE rolname=:role'), {'role': role}).first()
            if exists:
                for privilege in ('SELECT', 'INSERT', 'UPDATE', 'DELETE'):
                    assert not workspace.execute(text('SELECT has_table_privilege(:role,:table,:privilege)'),
                        {'role': role, 'table': f'pg_temp.{table}', 'privilege': privilege}).scalar_one()


def test_reviewed_preparation_is_idempotent_and_rejects_changed_activities(workspace):
    from scripts.prepare_group_access import preview, prepare
    _, sid, code = activity(workspace)
    conn = group.get_connection()
    try:
        _, fingerprint = preview(conn)
        result = prepare(conn, fingerprint)
        assert len(result) == 1 and result[0]['access']['code'] == code
        conn.commit()
        assert prepare(conn, fingerprint)[0]['access']['code'] == code
        conn.commit()
        workspace.execute(text('UPDATE planting_schedules SET expected_planters=11 WHERE id=:id'), {'id': sid})
        with pytest.raises(ValueError, match='Activities changed'):
            prepare(conn, fingerprint)
    finally:
        conn.close()
