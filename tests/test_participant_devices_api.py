"""Device tracking remains staff-only; no external services are needed."""
import importlib
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture()
def devices_api(monkeypatch):
    monkeypatch.syspath_prepend(str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
    monitoring = importlib.import_module('api.routes.monitoring')
    monkeypatch.setattr(monitoring, 'get_user_by_session_token', lambda _: None)
    route = importlib.import_module('api.routes.planters')
    summary = {'participant_count': 10, 'registered_devices': 3, 'available_devices': 7, 'devices': []}
    read = Mock(return_value=summary)
    reset = Mock()
    monkeypatch.setattr(route, 'list_participant_devices', read)
    monkeypatch.setattr(route, 'reset_participant_device', reset)
    app = FastAPI()
    app.include_router(route.router, prefix='/api/planters')
    return TestClient(app), monitoring, read, reset, summary


def test_participants_cannot_list_or_reset_device_slots(devices_api):
    client, _, read, reset, _ = devices_api
    assert client.get('/api/planters/1/participants').status_code == 401
    assert client.post('/api/planters/1/participants/2/reset-device').status_code == 401
    read.assert_not_called()
    reset.assert_not_called()


def test_staff_device_summary_and_unknown_account(devices_api, monkeypatch):
    client, monitoring, read, _, summary = devices_api
    monkeypatch.setattr(monitoring, 'get_user_by_session_token', lambda _: {'id': 7, 'role': 'lgu'})
    response = client.get('/api/planters/1/participants')
    assert response.status_code == 200
    assert response.json() == summary
    read.assert_called_once_with(1)
    read.side_effect = ValueError('Organization account not found.')
    assert client.get('/api/planters/999/participants').status_code == 404


def test_codes_are_staff_only_and_never_cached(devices_api, monkeypatch):
    client, monitoring, _, _, _ = devices_api
    route = importlib.import_module('api.routes.planters')
    manage = Mock(return_value={'planter_id': 1, 'codes': [{'participant_slot': 2, 'code': 'K7MP-4R9X', 'revoked': False}]})
    monkeypatch.setattr(route, 'manage_codes', manage)
    paths = ['/api/planters/1/access-codes', '/api/planters/1/participants/2/access-code']
    assert client.get(paths[0]).status_code == 401
    assert client.post(paths[0]).status_code == 401
    assert client.post(paths[1], json={'action': 'regenerate'}).status_code == 401
    manage.assert_not_called()
    monkeypatch.setattr(monitoring, 'get_user_by_session_token', lambda _: {'id': 7, 'role': 'lgu'})
    response = client.get(paths[0])
    assert response.status_code == 200 and response.headers['cache-control'] == 'no-store'
    assert manage.call_args.kwargs == {}
    response = client.post(paths[0])
    assert response.status_code == 200 and response.headers['cache-control'] == 'no-store'
    assert manage.call_args.kwargs == {'issue': True}
    response = client.post(paths[1], json={'action': 'regenerate'})
    assert response.status_code == 200 and response.headers['cache-control'] == 'no-store'
    assert manage.call_args.kwargs == {'slot': 2, 'action': 'regenerate'}


def test_shared_activity_codes_are_staff_only_and_never_cached(devices_api, monkeypatch):
    client, monitoring, _, _, _ = devices_api
    route = importlib.import_module('api.routes.planters')
    manage = Mock(return_value={'access': {'code': 'K7MP-4R9X', 'schedule_id': 3}})
    monkeypatch.setattr(route, 'manage_activity_code', manage)
    path = '/api/planters/organizations/7/activities/3/access'
    assert client.get(path).status_code == 401
    assert client.post(path, json={'action': 'issue'}).status_code == 401
    manage.assert_not_called()
    monkeypatch.setattr(monitoring, 'get_user_by_session_token', lambda _: {'id': 7, 'role': 'lgu'})
    response = client.get(path)
    assert response.status_code == 200 and response.headers['cache-control'] == 'no-store'
    assert manage.call_args.kwargs == {'action': None, 'organization_id': 7}
    response = client.post(path, json={'action': 'replace'})
    assert response.status_code == 200 and response.headers['cache-control'] == 'no-store'
    assert manage.call_args.kwargs == {'action': 'replace', 'organization_id': 7}
