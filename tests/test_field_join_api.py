"""The code-only welcome step never grants access to a participant workspace."""
import sys
import io
import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
from api.routes import planter_auth
from api.security import SessionSecurityMiddleware
from mangrovision_db.group_access import GroupAccessError
import deploy_testing


@pytest.fixture
def client():
    app = FastAPI()
    app.add_middleware(SessionSecurityMiddleware)
    app.include_router(planter_auth.router, prefix='/api/planter-auth')
    with TestClient(app, base_url='https://field.example.test') as client:
        yield client


def test_code_only_entry_returns_welcome_context_without_a_session(client, monkeypatch):
    calls = []
    def check(code, **options):
        calls.append(code)
        return {'title': 'Tree planting', 'organization_name': 'Test group'}
    monkeypatch.setattr(planter_auth, 'check_group_code', check)
    response = client.post('/api/planter-auth/join', json={'access_code': 'K7MP-4R9X'})
    assert response.status_code == 200
    assert response.json() == {'activity': {'title': 'Tree planting', 'organization_name': 'Test group'}}
    assert response.headers['cache-control'] == 'no-store'
    assert not response.headers.get('set-cookie')
    assert calls == ['K7MP-4R9X']
    assert client.post('/api/planter-auth/login', json={'access_code': 'K7MP-4R9X'}).status_code == 422


def test_invalid_code_stays_a_code_field_error(client, monkeypatch):
    def invalid(*args, **kwargs):
        raise GroupAccessError('Incorrect organization code.', 401, field='access_code')
    monkeypatch.setattr(planter_auth, 'check_group_code', invalid)
    response = client.post('/api/planter-auth/join', json={'access_code': 'K7MP-4R9X'})
    assert response.status_code == 401
    assert response.json()['detail'][0]['loc'] == ['body', 'access_code']
    assert not response.headers.get('set-cookie')


def test_code_entry_accepts_stale_session_cookies_but_rejects_cross_origin(client, monkeypatch):
    monkeypatch.setattr(planter_auth, 'check_group_code', lambda *args, **kwargs: {'title': 'Tree planting', 'organization_name': 'Test group'})
    client.cookies.set('mv_planter_session', 'expired-session')
    response = client.post('/api/planter-auth/join', headers={'Origin': 'https://field.example.test'}, json={'access_code': 'K7MP-4R9X'})
    assert response.status_code == 200
    response = client.post('/api/planter-auth/join', headers={'Origin': 'https://attacker.example'}, json={'access_code': 'K7MP-4R9X'})
    assert response.status_code == 403


def test_deployment_requires_the_code_only_welcome_endpoint(client, monkeypatch):
    document = client.get('/openapi.json').json()
    monkeypatch.setattr(deploy_testing, 'urlopen', lambda *args, **kwargs: io.BytesIO(json.dumps(document).encode()))
    deploy_testing.verify_planter_login_contract('https://backend.example.test')
    del document['paths']['/api/planter-auth/join']
    with pytest.raises(RuntimeError, match='code-only first-time entry'):
        deploy_testing.verify_planter_login_contract('https://backend.example.test')
