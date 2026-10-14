"""API deletion/release rules; physical history preservation has isolated PG tests."""
import sys
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from sqlalchemy.exc import OperationalError

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'MangroVision_New'))
from api.routes import project_sites as routes


@pytest.fixture()
def client():
    app = FastAPI()
    app.include_router(routes.router, prefix='/api/project-sites')
    with patch.object(routes, '_require_lgu_user', return_value={'id': 7, 'role': 'lgu'}), \
         patch.object(routes, 'is_processing_active', return_value=False), \
         TestClient(app) as http:
        yield http


def test_deletion_requires_a_reviewed_version(client):
    with patch.object(routes, 'delete_project_site') as remove:
        assert client.delete('/api/project-sites/1').status_code == 422
        remove.assert_not_called()


def test_all_dead_deletion_passes_version_and_actor_and_keeps_policy_errors(client):
    with patch.object(routes, 'delete_project_site', return_value=True) as remove:
        result = client.delete('/api/project-sites/1?version=' + 'a'*64)
        assert result.status_code == 200
        remove.assert_called_once_with(1, expected_version='a'*64, user_id=7)
    with patch.object(routes, 'delete_project_site', side_effect=ValueError('All seedlings must be dead.')):
        result = client.delete('/api/project-sites/1?version=' + 'a'*64)
        assert result.status_code == 409
        assert result.json()['detail'] == 'All seedlings must be dead.'


def test_release_requires_review_and_passes_actor(client):
    with patch.object(routes, 'apply_site_action', return_value={'status': 'released'}) as release:
        assert client.post('/api/project-sites/1/release', json={}).status_code == 422
        assert client.post('/api/project-sites/1/release', json={'version': 'b'*64}).status_code == 200
        release.assert_called_once_with(1, 'release', 'b'*64, 7)


def test_unknown_site_and_processing_lock_are_clear(client):
    with patch.object(routes, 'preview_site_actions', side_effect=KeyError(999)):
        assert client.get('/api/project-sites/999/actions').status_code == 404
    with patch.object(routes, 'is_processing_active', return_value=True), patch.object(routes, 'apply_site_action') as release:
        assert client.post('/api/project-sites/1/release', json={'version': 'a'*64}).status_code == 409
        release.assert_not_called()


def test_database_failure_never_returns_connection_details(client):
    failure = OperationalError(None, None, Exception('private connection details'))
    with patch.object(routes, 'preview_site_actions', side_effect=failure):
        response = client.get('/api/project-sites/1/actions')
        assert response.status_code == 503
        assert 'private' not in response.text
    with patch.object(routes, 'delete_project_site', side_effect=failure):
        response = client.delete('/api/project-sites/1?version=' + 'a'*64)
        assert response.status_code == 503
        assert 'private' not in response.text


def test_unauthorized_staff_cannot_use_actions(client):
    from fastapi import HTTPException
    with patch.object(routes, '_require_lgu_user', side_effect=HTTPException(403, 'LGU permission is required.')), patch.object(routes, 'apply_site_action') as release:
        assert client.post('/api/project-sites/1/release', json={'version': 'a'*64}).status_code == 403
        release.assert_not_called()
