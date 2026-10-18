"""LGU replacement writes require a staff session and server-owned attribution."""
import sys
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
from api.routes import monitoring

BODY = {'planting_event_ids': [10], 'versions': {'10': 1}, 'project_site_id': 2,
        'planted_date': '2026-10-18', 'species': 'Bungalon'}


@pytest.mark.parametrize('user,status', [(None, 401), ({'id': 5, 'role': 'planter'}, 403)])
def test_field_and_signed_out_users_cannot_record_lgu_replacements(monkeypatch, user, status):
    monkeypatch.setattr(monitoring, 'get_user_by_session_token', lambda _: user)
    app = FastAPI()
    app.include_router(monitoring.router)
    with TestClient(app) as client:
        assert client.post('/replanting/lgu-plantings', json=BODY).status_code == status


def test_staff_attribution_comes_from_the_session_and_body_cannot_override_it(monkeypatch):
    monkeypatch.setattr(monitoring, 'get_user_by_session_token', lambda _: {'id': 17, 'role': 'planner'})
    received = []
    monkeypatch.setattr(monitoring, 'record_lgu_replanting',
                        lambda *args, **kwargs: received.append((args, kwargs)) or {'replanted_points': 1})
    app = FastAPI()
    app.include_router(monitoring.router)
    with TestClient(app) as client:
        response = client.post('/replanting/lgu-plantings', json=BODY)
        assert response.status_code == 201
        assert received[0][1]['staff_user_id'] == 17
        assert client.post('/replanting/lgu-plantings', json={**BODY, 'staff_user_id': 99}).status_code == 422
        assert len(received) == 1
