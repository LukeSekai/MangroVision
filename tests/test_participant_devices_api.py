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
