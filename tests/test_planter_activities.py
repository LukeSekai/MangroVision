"""The participant activity endpoint never exposes another organization or staff contacts."""

import importlib
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def api(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
    routes = importlib.import_module('api.routes.planter_auth')
    monkeypatch.setattr(routes, 'get_planter_by_session_token', lambda _: None)
    monkeypatch.setattr(routes, 'list_planter_activities', lambda _: pytest.fail('Unauthenticated activity read'))
    app = FastAPI()
    app.include_router(routes.router, prefix='/api/planter-auth')
    with TestClient(app) as client:
        yield client, routes


def test_activity_list_requires_active_participant_session(api, monkeypatch):
    client, routes = api
    assert client.get('/api/planter-auth/me/activities').status_code == 401
    monkeypatch.setattr(routes, 'get_planter_by_session_token', lambda _: {'id': 1, 'status': 'inactive', 'organization_id': 4})
    assert client.get('/api/planter-auth/me/activities').status_code == 403
    monkeypatch.setattr(routes, 'get_planter_by_session_token', lambda _: {'id': 1, 'status': 'active', 'organization_id': 4})
    assert client.get('/api/planter-auth/me/activities').status_code == 401


def test_scope_comes_from_the_session_not_query_parameters(api, monkeypatch):
    client, routes = api
    monkeypatch.setattr(routes, 'get_planter_by_session_token', lambda _: {'id': 1, 'status': 'active', 'organization_id': 4, 'participant_slot': 2, 'planting_schedule_id': 3})
    calls = []
    monkeypatch.setattr(routes, 'list_planter_activities', lambda organization_id: calls.append(organization_id) or [{'id': 3, 'title': 'Our activity'}])
    response = client.get('/api/planter-auth/me/activities?organization_id=999')
    assert response.status_code == 200
    assert response.headers['cache-control'] == 'no-store'
    assert calls == [4]
    assert response.json() == {'activities': [{'id': 3, 'title': 'Our activity'}]}


def test_removed_registration_cannot_create_access(api, monkeypatch):
    client, routes = api
    monkeypatch.setattr(routes, 'authenticate_group', lambda *args, **kwargs: pytest.fail('Closed registration wrote access'))
    response = client.post('/api/planter-auth/register', json={'organization_id': 7, 'username': 'takeover', 'password': 'test-only'})
    assert response.status_code == 410
    assert 'organization code' in response.json()['detail']
    assert 'set-cookie' not in response.headers


def test_old_shared_credentials_cannot_bypass_code_sign_in(api, monkeypatch):
    client, routes = api
    monkeypatch.setattr(routes, 'authenticate_group', lambda *args, **kwargs: pytest.fail('Legacy credentials accepted'))
    response = client.post('/api/planter-auth/login', json={'username': 'old-oton', 'password': 'previous', 'device_key': 'ab' * 24})
    assert response.status_code == 422
    assert 'set-cookie' not in response.headers


def test_invalid_code_never_sets_a_session(api, monkeypatch):
    from mangrovision_db.planter_codes import CodeError
    client, routes = api
    def rejected(*args, **kwargs):
        raise CodeError('This planter code is incorrect or no longer valid.', 401)
    monkeypatch.setattr(routes, 'authenticate_group', rejected)
    response = client.post('/api/planter-auth/login', json={'access_code': 'K7MP-4R9X', 'username': 'Ana', 'browser_key': 'ab' * 24})
    assert response.status_code == 401
    assert response.json()['detail'][0]['loc'] == ['body', 'access_code']
    assert 'set-cookie' not in response.headers


def test_login_returns_exact_slot_without_credentials(api, monkeypatch):
    client, routes = api
    calls = []
    def authenticate(*args, **kwargs):
        calls.append((args, kwargs))
        return {'id': 1, 'status': 'active', 'participant_slot': 2, 'participant_username': 'Ana', 'planting_schedule_id': 3,
                'username': 'old-oton', 'password_hash': 'private'}, 'test-session'
    monkeypatch.setattr(routes, 'authenticate_group', authenticate)
    sessions = []
    monkeypatch.setattr(routes, 'set_planter_session', lambda response, token: sessions.append(token))
    response = client.post('/api/planter-auth/login', json={'access_code': 'K7MP-4R9X', 'username': 'Ana', 'browser_key': 'ab' * 24})
    assert response.status_code == 200 and sessions == ['test-session']
    assert response.headers['cache-control'] == 'no-store'
    assert 'browser_key' not in response.json() and 'device_key' not in response.json()
    assert response.json()['planter']['participant_slot'] == 2
    assert not {'username', 'password_hash', 'access_code'} & response.json()['planter'].keys()
    assert calls[0][0] == ('K7MP-4R9X', 'Ana', 'ab' * 24)


def test_throttled_login_has_retry_after(api, monkeypatch):
    from mangrovision_db.planter_codes import CodeError
    client, routes = api
    def rejected(*args, **kwargs):
        raise CodeError('Too many sign-in attempts.', 429, 120)
    monkeypatch.setattr(routes, 'authenticate_group', rejected)
    response = client.post('/api/planter-auth/login', json={'access_code': 'K7MP-4R9X', 'username': 'Ana', 'browser_key': 'ab' * 24})
    assert response.status_code == 429 and response.headers['retry-after'] == '120'
    assert 'set-cookie' not in response.headers
