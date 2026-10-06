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
                phone='09123456789', email='coordinator@example.org', title='Coastal planting',
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
    monkeypatch.setattr(staff, 'get_user_by_session_token', lambda _: None)
    with TestClient(app) as client:
        yield client, routes, staff


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


@pytest.mark.parametrize('changes', [
    {'consent': False}, {'participants': 0}, {'participants': 1.5}, {'participants': 10001},
    {'phone': 'abcdefghi'}, {'email': 'invalid-email'}, {'organization': '   '},
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
    assert client.post('/api/like-appointments/1/review', json={'action': 'confirmed'}).status_code == 401
    assert client.get('/api/public/like/appointments').status_code == 405
    monkeypatch.setattr(staff, 'get_user_by_session_token', lambda _: {'id': 1, 'role': 'viewer'})
    assert client.get('/api/like-appointments').status_code == 403


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
