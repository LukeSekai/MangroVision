"""Credentials and LGU monitoring are staff-only and use the session actor."""
import sys
from pathlib import Path
from unittest.mock import Mock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
from api.routes import lgu_sites as routes, monitoring

BODY = {'version': 'a'*64, 'monitored_at': '2026-10-19', 'dead_planting_event_ids': [7],
        'health_status': 'good', 'actions_taken': 'Removed debris.'}


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr(monitoring, 'get_user_by_session_token', lambda _: {'id': 11, 'role': 'lgu'})
    app = FastAPI()
    app.include_router(routes.router, prefix='/api/monitoring/lgu-sites')
    with TestClient(app) as api:
        yield api


@pytest.mark.parametrize('user,status', [(None, 401), ({'id': 5, 'role': 'planter'}, 403)])
def test_credentials_and_visits_are_not_public(client, monkeypatch, user, status):
    monkeypatch.setattr(monitoring, 'get_user_by_session_token', lambda _: user)
    read = Mock()
    monkeypatch.setattr(routes.lgu_sites, 'workspace', read)
    assert client.get('/api/monitoring/lgu-sites').status_code == status
    assert client.get('/api/monitoring/lgu-sites/1/monitoring').status_code == status
    assert client.post('/api/monitoring/lgu-sites/1/monitoring', json=BODY).status_code == status
    read.assert_not_called()


def test_credentials_response_disables_caching(client, monkeypatch):
    monkeypatch.setattr(routes.lgu_sites, 'workspace', lambda: {'account': {'access_code': 'TEST-CODE'}})
    result = client.get('/api/monitoring/lgu-sites')
    assert result.status_code == 200
    assert result.headers['Cache-Control'] == 'private, no-store'


def test_visit_uses_server_actor_and_rejects_extra_fields(client, monkeypatch):
    write = Mock(return_value={'id': 1})
    monkeypatch.setattr(routes.lgu_sites, 'record_visit', write)
    assert client.post('/api/monitoring/lgu-sites/10/monitoring', json=BODY).status_code == 201
    assert write.call_args.args == (10,)
    assert write.call_args.kwargs['inspector_user_id'] == 11
    assert client.post('/api/monitoring/lgu-sites/10/monitoring', json={**BODY, 'inspector_user_id': 99}).status_code == 422
    assert client.post('/api/monitoring/lgu-sites/10/monitoring', json={**BODY, 'average_height_cm': -1}).status_code == 422
    assert write.call_count == 1


def test_conflicts_keep_the_site_form_open_for_review(client, monkeypatch):
    monkeypatch.setattr(routes.lgu_sites, 'record_visit', Mock(side_effect=ValueError('This site changed. Reload its monitoring form before saving.')))
    result = client.post('/api/monitoring/lgu-sites/10/monitoring', json=BODY)
    assert result.status_code == 409
    assert 'Reload' in result.json()['detail']
