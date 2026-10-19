"""Only authenticated staff can release dead locations from project sites."""
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
from api.routes import monitoring

BODY = {'planting_event_ids': [10], 'versions': {'10': 0}}


def client():
    app = FastAPI()
    app.include_router(monitoring.router)
    return TestClient(app)


@pytest.mark.parametrize('user,status', [(None, 401), ({'id': 5, 'role': 'planter'}, 403)])
def test_release_requires_staff(monkeypatch, user, status):
    monkeypatch.setattr(monitoring, 'get_user_by_session_token', lambda _: user)
    with client() as api:
        assert api.post('/release-locations', json=BODY).status_code == status
        assert api.get('/release-candidates').status_code == status


def test_release_uses_session_actor_and_rejects_unexpected_fields(monkeypatch):
    monkeypatch.setattr(monitoring, 'get_user_by_session_token', lambda _: {'id': 17, 'role': 'planner'})
    received = []
    monkeypatch.setattr(monitoring, 'release_dead_locations_batch',
                        lambda *args: received.append(args) or {'released_count': 1})
    with client() as api:
        assert api.post('/release-locations', json=BODY).json() == {'released_count': 1}
        assert received == [([10], {'10': 0}, 17)]
        assert api.post('/release-locations', json={**BODY, 'user_id': 99}).status_code == 422
        assert api.post('/release-locations', json={**BODY, 'planting_event_ids': []}).status_code == 422
        assert len(received) == 1


def test_conflict_keeps_the_error_actionable_and_old_workflow_is_removed(monkeypatch):
    monkeypatch.setattr(monitoring, 'get_user_by_session_token', lambda _: {'id': 17, 'role': 'lgu'})
    def stale(*_):
        raise ValueError('Location changed. Reload the map before releasing it.')
    monkeypatch.setattr(monitoring, 'release_dead_locations_batch', stale)
    with client() as api:
        response = api.post('/release-locations', json=BODY)
        assert response.status_code == 409
        assert 'Reload the map' in response.json()['detail']
        for path in ['/replanting/approve', '/replanting/assign', '/replanting/lgu-plantings']:
            assert api.post(path, json=BODY).status_code == 404


def test_active_processing_prevents_boundary_changes(monkeypatch):
    monkeypatch.setattr(monitoring, 'get_user_by_session_token', lambda _: {'id': 17, 'role': 'lgu'})
    monkeypatch.setattr(monitoring, 'is_processing_active', lambda: True)
    with client() as api:
        assert api.post('/release-locations', json=BODY).status_code == 409
