"""Public contract validation and staff authorization, without database access."""

import importlib
from datetime import datetime, timedelta, timezone
from pathlib import Path
from uuid import uuid4

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


def payload():
    start = (datetime.now(timezone(timedelta(hours=8))) + timedelta(days=2)).replace(hour=8, minute=0, second=0, microsecond=0)
    return dict(submission_key=str(uuid4()), organization='Test school', contact_name='Test coordinator',
                phone='09123456789', email='coordinator@example.org', title='Coastal planting', appointment_type='tree_planting',
                start_at=start.isoformat(), end_at=(start + timedelta(hours=2)).isoformat(),
                participants=20, notes='Test request', consent=True, website='')


@pytest.fixture()
def api(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
    routes = importlib.import_module('api.routes.like_appointments')
    staff = importlib.import_module('api.routes.planting_schedules')
    app = FastAPI()
    app.include_router(routes.public_router, prefix='/api/public/like')
    app.include_router(routes.staff_router, prefix='/api/like-appointments')
    app.include_router(staff.router, prefix='/api/planting-schedules')
    monkeypatch.setattr(staff, 'get_user_by_session_token', lambda _: None)
    monkeypatch.setattr(routes, 'run_delivery_cycle', lambda **kwargs: {'configured': False, 'sent': 0, 'failed': 0})
    with TestClient(app) as client:
        yield client, routes, staff


@pytest.mark.parametrize('method', ['put', 'patch'])
def test_locked_activity_returns_conflict_without_sending_email(api, monkeypatch, method):
    from mangrovision_db.activity_rules import ActivityConflict
    client, _, staff = api
    monkeypatch.setattr(staff, 'get_user_by_session_token', lambda _: {'id': 1, 'role': 'lgu'})
    monkeypatch.setattr(staff, 'get_planting_schedule', lambda _: {'id': 1})
    def locked(*args, **kwargs):
        raise ActivityConflict('Planting has already been recorded for this activity.')
    monkeypatch.setattr(staff, 'update_planting_schedule', locked)
    monkeypatch.setattr(staff, 'run_delivery_cycle', lambda **kwargs: pytest.fail('Rejected change started email delivery'))
    response = getattr(client, method)('/api/planting-schedules/1', json={'title': 'New title'})
    assert response.status_code == 409
    assert 'already been recorded' in response.json()['detail']


def test_schedule_revision_runs_queued_delivery_after_success(api, monkeypatch):
    client, _, staff = api
    monkeypatch.setattr(staff, 'get_user_by_session_token', lambda _: {'id': 1, 'role': 'lgu'})
    monkeypatch.setattr(staff, 'get_planting_schedule', lambda _: {'id': 1})
    updates, deliveries = [], []
    def revise(schedule_id, **values):
        updates.append((schedule_id, values))
        return {'id': schedule_id, 'email_update': {'request_id': 7, 'kind': 'update', 'status': 'queued'}}
    monkeypatch.setattr(staff, 'update_planting_schedule', revise)
    monkeypatch.setattr(staff, 'run_delivery_cycle', lambda **kwargs: deliveries.append(kwargs))
    response = client.patch('/api/planting-schedules/1', json={'title': 'Agreed revision'})
    assert response.status_code == 200
    assert updates == [(1, {'title': 'Agreed revision', 'updated_by_user_id': 1})]
    assert deliveries == [{'request_id': 7}]


def test_notes_update_does_not_run_delivery(api, monkeypatch):
    client, _, staff = api
    monkeypatch.setattr(staff, 'get_user_by_session_token', lambda _: {'id': 1, 'role': 'lgu'})
    monkeypatch.setattr(staff, 'get_planting_schedule', lambda _: {'id': 1})
    monkeypatch.setattr(staff, 'update_planting_schedule', lambda *args, **kwargs: {'id': 1, 'email_update': None})
    monkeypatch.setattr(staff, 'run_delivery_cycle', lambda **kwargs: pytest.fail('Notes change started email delivery'))
    assert client.patch('/api/planting-schedules/1', json={'notes': 'Corrected note'}).status_code == 200


def test_staff_schedule_update_runs_the_staff_queue_only(api, monkeypatch):
    client, _, staff = api
    monkeypatch.setattr(staff, 'get_user_by_session_token', lambda _: {'id': 1, 'role': 'lgu'})
    monkeypatch.setattr(staff, 'get_planting_schedule', lambda _: {'id': 1})
    monkeypatch.setattr(staff, 'update_planting_schedule', lambda *args, **kwargs: {'id': 1, 'email_update': {'schedule_id': 1, 'status': 'queued', 'kind': 'update'}})
    monkeypatch.setattr(staff, 'run_delivery_cycle', lambda **kwargs: pytest.fail('Staff activity entered website queue'))
    calls = []
    monkeypatch.setattr(staff, 'run_schedule_delivery_cycle', lambda **kwargs: calls.append(kwargs))
    assert client.patch('/api/planting-schedules/1', json={'notification_email': 'head@example.org'}).status_code == 200
    assert calls == [{'schedule_id': 1}]
    assert client.patch('/api/planting-schedules/1', json={'notification_email': 'broken'}).status_code == 422


def test_missing_staff_recipient_does_not_start_delivery(api, monkeypatch):
    client, _, staff = api
    monkeypatch.setattr(staff, 'get_user_by_session_token', lambda _: {'id': 1, 'role': 'lgu'})
    monkeypatch.setattr(staff, 'get_planting_schedule', lambda _: {'id': 1})
    monkeypatch.setattr(staff, 'update_planting_schedule', lambda *args, **kwargs: {'id': 1, 'email_update': {'schedule_id': 1, 'status': 'missing_recipient'}})
    monkeypatch.setattr(staff, 'run_schedule_delivery_cycle', lambda **kwargs: pytest.fail('No-recipient update started delivery'))
    assert client.patch('/api/planting-schedules/1', json={'title': 'Revised activity'}).status_code == 200


def test_recorded_activity_delete_returns_conflict(api, monkeypatch):
    from mangrovision_db.activity_rules import ActivityConflict
    client, _, staff = api
    monkeypatch.setattr(staff, 'get_user_by_session_token', lambda _: {'id': 1, 'role': 'lgu'})
    def locked(schedule_id, **kwargs):
        assert schedule_id == 1 and kwargs['deleted_by_user_id'] == 1
        raise ActivityConflict('This activity has recorded planting and cannot be deleted.')
    monkeypatch.setattr(staff, 'delete_planting_schedule', locked)
    response = client.delete('/api/planting-schedules/1')
    assert response.status_code == 409 and 'cannot be deleted' in response.json()['detail']


def test_public_receipt_and_timezone(api, monkeypatch):
    client, routes, _ = api
    captured = {}
    def submit(values, peer):
        captured.update(values)
        return {'reference': 'LIKE-TEST', 'status': 'pending', 'timezone': 'Asia/Manila'}
    monkeypatch.setattr(routes, 'submit_appointment', submit)
    response = client.post('/api/public/like/appointments', json=payload())
    assert response.status_code == 201
    assert response.json() == {'reference': 'LIKE-TEST', 'status': 'pending', 'timezone': 'Asia/Manila'}
    assert captured['start_at'].utcoffset() == timedelta(hours=8)


@pytest.mark.parametrize('appointment_type,label', [
    ('field_visit', 'Field visit'), ('clean_up_drive', 'Clean-up drive'), ('tree_planting', 'Tree planting'),
])
@pytest.mark.parametrize('has_custom_title', [False, True])
def test_public_name_is_generated_from_type(api, monkeypatch, appointment_type, label, has_custom_title):
    client, routes, _ = api
    captured = {}
    def submit(values, peer):
        captured.update(values)
        return {'reference': 'LIKE-TEST'}
    monkeypatch.setattr(routes, 'submit_appointment', submit)
    body = payload()
    body['appointment_type'] = appointment_type
    if has_custom_title:
        body['title'] = 'Cached arbitrary activity name'
    else:
        body.pop('title')
    assert client.post('/api/public/like/appointments', json=body).status_code == 201
    assert captured['title'] == label


def manual_payload(**changes):
    public = payload()
    return {**{key: public[key] for key in ('organization', 'contact_name', 'phone', 'email', 'appointment_type', 'start_at', 'end_at', 'notes')},
            'expected_planters': public['participants'], 'status': 'confirmed', **changes}


@pytest.mark.parametrize('appointment_type,label', [
    ('field_visit', 'Field visit'), ('clean_up_drive', 'Clean-up drive'), ('tree_planting', 'Tree planting'),
])
def test_manual_input_matches_public_data_and_generates_name(api, monkeypatch, appointment_type, label):
    client, _, staff = api
    monkeypatch.setattr(staff, 'get_user_by_session_token', lambda _: {'id': 1, 'role': 'lgu'})
    captured = {}
    def create(**values):
        captured.update(values)
        return {'id': 1}
    monkeypatch.setattr(staff, 'create_planting_schedule', create)
    body = manual_payload(appointment_type=appointment_type, title='Ignored custom title')
    response = client.post('/api/planting-schedules', json=body)
    assert response.status_code == 201
    assert captured['appointment_type'] == appointment_type and captured['title'] == label
    assert captured['contact'] == f"{body['contact_name']}, {body['phone']}"
    assert captured['notification_email'] == body['email']
    assert captured['expected_planters'] == body['expected_planters']
    assert captured['inspection_interval_days'] == 14
    assert captured['start_at'] == body['start_at'] and captured['end_at'] == body['end_at']


@pytest.mark.parametrize('changes', [
    {'appointment_type': 'other'}, {'appointment_type': None}, {'contact_name': None}, {'contact_name': ' '},
    {'phone': None}, {'phone': 'abcdefghi'}, {'phone': '123456'}, {'email': None}, {'email': 'broken'},
    {'expected_planters': None}, {'expected_planters': 0}, {'expected_planters': 10001}, {'expected_planters': 1.5},
    {'expected_planters': True}, {'organization': ' '},
    {'start_at': '2020-01-01T08:00:00+08:00', 'end_at': '2020-01-01T09:00:00+08:00'},
    {'start_at': '2035-01-01T08:00:00+08:00', 'end_at': '2035-01-01T07:00:00+08:00'},
    {'start_at': '2035-01-01T08:00:00+08:00', 'end_at': '2035-01-02T09:00:00+08:00'},
    {'appointment_type': 'field_visit', 'expected_seedlings': 10},
])
def test_invalid_manual_inputs_never_reach_storage(api, monkeypatch, changes):
    client, _, staff = api
    monkeypatch.setattr(staff, 'get_user_by_session_token', lambda _: {'id': 1, 'role': 'lgu'})
    monkeypatch.setattr(staff, 'create_planting_schedule', lambda **kwargs: pytest.fail('Invalid input reached storage'))
    assert client.post('/api/planting-schedules', json=manual_payload(**changes)).status_code in (400, 422)


@pytest.mark.parametrize('changes', [
    {'consent': False}, {'participants': 0}, {'participants': 1.5}, {'participants': 10001},
    {'phone': 'abcdefghi'}, {'email': 'invalid-email'}, {'email': ''}, {'email': None}, {'organization': '   '},
    {'appointment_type': 'other'}, {'appointment_type': ''},
    {'status': 'confirmed'}, {'organization_id': 1}, {'submission_key': 'invalid'},
    {'start_at': '2030-01-01T10:00:00'},
    {'start_at': '2030-01-01T10:00:00+08:00', 'end_at': '2030-01-01T09:00:00+08:00'},
    {'start_at': '2030-01-01T10:00:00+08:00', 'end_at': '2030-01-02T10:00:00+08:00'},
])
def test_invalid_public_input_rejected_before_service(api, monkeypatch, changes):
    client, routes, _ = api
    monkeypatch.setattr(routes, 'submit_appointment', lambda *args: pytest.fail('Invalid input reached the service'))
    assert client.post('/api/public/like/appointments', json={**payload(), **changes}).status_code == 422


def test_honeypot_rejected(api, monkeypatch):
    client, routes, _ = api
    monkeypatch.setattr(routes, 'submit_appointment', lambda *args: pytest.fail('Spam reached the service'))
    assert client.post('/api/public/like/appointments', json={**payload(), 'website': 'spam'}).status_code == 400


def test_public_cannot_read_or_review_requests(api, monkeypatch):
    client, routes, staff = api
    monkeypatch.setattr(routes, 'list_appointments', lambda: pytest.fail('Anonymous request read private records'))
    monkeypatch.setattr(routes, 'review_appointment', lambda *args: pytest.fail('Anonymous request reviewed a booking'))
    assert client.get('/api/like-appointments').status_code == 401
    assert client.get('/api/like-appointments/summary').status_code == 401
    assert client.post('/api/like-appointments/1/review', json={'action': 'confirmed'}).status_code == 401
    assert client.get('/api/public/like/appointments').status_code == 405
    monkeypatch.setattr(staff, 'get_user_by_session_token', lambda _: {'id': 1, 'role': 'viewer'})
    assert client.get('/api/like-appointments').status_code == 403
    assert client.get('/api/like-appointments/summary').status_code == 403


def test_lgu_can_review_and_rate_limit_is_clear(api, monkeypatch):
    client, routes, staff = api
    monkeypatch.setattr(staff, 'get_user_by_session_token', lambda _: {'id': 1, 'role': 'lgu'})
    monkeypatch.setattr(routes, 'review_appointment', lambda request_id, values, user_id: {'id': request_id, 'schedule_id': 2})
    assert client.post('/api/like-appointments/1/review', json={'action': 'confirmed', 'contacted': True}).json()['schedule_id'] == 2
    def limited(*args):
        raise routes.AppointmentRateLimit('Please try again later.')
    monkeypatch.setattr(routes, 'submit_appointment', limited)
    response = client.post('/api/public/like/appointments', json=payload())
    assert response.status_code == 429
    assert response.headers['retry-after'] == '3600'


@pytest.mark.parametrize('appointment_type', ['field_visit', 'clean_up_drive', 'tree_planting'])
def test_all_appointment_types_are_accepted(api, monkeypatch, appointment_type):
    client, routes, _ = api
    captured = {}
    def submit(values, peer):
        captured.update(values)
        return {'reference': 'LIKE-TEST', 'status': 'pending', 'timezone': 'Asia/Manila'}
    monkeypatch.setattr(routes, 'submit_appointment', submit)
    assert client.post('/api/public/like/appointments', json={**payload(), 'appointment_type': appointment_type}).status_code == 201
    assert captured['appointment_type'] == appointment_type


@pytest.mark.parametrize('missing', ['email', 'appointment_type'])
def test_required_fields_cannot_be_omitted(api, missing):
    client, _, _ = api
    body = payload()
    del body[missing]
    assert client.post('/api/public/like/appointments', json=body).status_code == 422


@pytest.mark.parametrize('role', ['admin', 'lgu', 'planner'])
def test_dashboard_summary_contains_only_count_and_link(api, monkeypatch, role):
    client, routes, staff = api
    monkeypatch.setattr(staff, 'get_user_by_session_token', lambda _: {'id': 1, 'role': role})
    summary = {'pending_count': 3, 'target_path': '/scheduling?requests=pending'}
    monkeypatch.setattr(routes, 'appointment_summary', lambda: summary)
    assert client.get('/api/like-appointments/summary').json() == summary


def test_confirmation_schedules_delivery_after_review(api, monkeypatch):
    client, routes, staff = api
    monkeypatch.setattr(staff, 'get_user_by_session_token', lambda _: {'id': 1, 'role': 'lgu'})
    calls = []
    monkeypatch.setattr(routes, 'review_appointment', lambda *args: {'id': 9, 'status': 'confirmed'})
    monkeypatch.setattr(routes, 'run_delivery_cycle', lambda **kwargs: calls.append(kwargs))
    assert client.post('/api/like-appointments/9/review', json={'action': 'confirmed', 'contacted': True}).status_code == 200
    assert calls == [{'request_id': 9}]
    assert client.post('/api/like-appointments/9/review', json={'action': 'confirmed', 'email': 'broken'}).status_code == 422


def test_pending_request_cancellation_rejected_before_review_or_delivery(api, monkeypatch):
    client, routes, staff = api
    monkeypatch.setattr(staff, 'get_user_by_session_token', lambda _: {'id': 1, 'role': 'planner'})
    monkeypatch.setattr(routes, 'review_appointment', lambda *args: pytest.fail('Cancellation reached pending-request review'))
    monkeypatch.setattr(routes, 'run_delivery_cycle', lambda **kwargs: pytest.fail('Rejected cancellation started delivery'))
    response = client.post('/api/like-appointments/9/review', json={'action': 'cancelled', 'decision_notes': 'No longer needed.'})
    assert response.status_code == 422


def test_pending_request_cancellation_rejected_before_database_access(monkeypatch):
    from mangrovision_db import like_appointments
    monkeypatch.setattr(like_appointments.planting, '_get_connection', lambda: pytest.fail('Invalid review accessed database'))
    with pytest.raises(ValueError, match='Confirm or decline'):
        like_appointments.review_appointment(9, {'action': 'cancelled'}, 1)
