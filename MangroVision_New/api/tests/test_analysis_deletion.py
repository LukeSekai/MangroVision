"""Deletion policy and API checks use only in-memory fixture records."""
import sqlite3
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import planting_database as database


@pytest.fixture
def workspace(monkeypatch):
    connection = sqlite3.connect(':memory:', check_same_thread=False)
    connection.row_factory = sqlite3.Row
    connection.executescript('''
        PRAGMA foreign_keys = ON;
        CREATE TABLE analyses(id INTEGER PRIMARY KEY);
        CREATE TABLE planting_points(id INTEGER PRIMARY KEY,
            analysis_id INTEGER REFERENCES analyses(id) ON DELETE CASCADE,
            status TEXT DEFAULT 'planned', planted_at TEXT, planted_date TEXT,
            death_at TEXT, death_reason TEXT, deleted_at TEXT);
        CREATE TABLE planter_assignment_points(id INTEGER PRIMARY KEY,
            planting_point_id INTEGER REFERENCES planting_points(id) ON DELETE CASCADE,
            status TEXT, released_at TEXT);
        CREATE TABLE planting_events(id INTEGER PRIMARY KEY,
            planting_point_id INTEGER REFERENCES planting_points(id) ON DELETE SET NULL, closed_at TEXT);
        CREATE TABLE point_death_records(id INTEGER PRIMARY KEY,
            planting_point_id INTEGER REFERENCES planting_points(id) ON DELETE CASCADE);
        CREATE TABLE monitoring_observations(id INTEGER PRIMARY KEY,
            planting_event_id INTEGER REFERENCES planting_events(id) ON DELETE CASCADE);
        CREATE TABLE analysis_assets(id INTEGER PRIMARY KEY,
            analysis_id INTEGER REFERENCES analyses(id) ON DELETE CASCADE, object_key TEXT);
        CREATE TABLE object_cleanup_jobs(object_key TEXT PRIMARY KEY, reason TEXT);
        INSERT INTO analyses VALUES (1),(2),(3);
        INSERT INTO planting_points(id,analysis_id) VALUES (1,1),(2,1),(3,2);
        INSERT INTO analysis_assets VALUES (1,1,'fixture-original'), (2,1,'fixture-overlay'), (3,2,'other-fixture');
    ''')

    class Connection:
        """SQLite checks policy/cascades; PostgreSQL checks the actual row locks."""
        def execute(self, sql, params=()):
            return connection.execute(sql.replace('FOR UPDATE', ''), params)

        def commit(self):
            connection.commit()

        def rollback(self):
            connection.rollback()

        def close(self):
            pass  # Keep the fixture readable after the service closes its handle.

    cleanup = Mock()
    monkeypatch.setattr(database, '_get_connection', Connection)
    monkeypatch.setattr(database, 'process_object_cleanup_jobs', cleanup)
    yield SimpleNamespace(db=connection, cleanup=cleanup)
    connection.close()


def assert_preserved(workspace):
    assert workspace.db.execute('SELECT COUNT(*) FROM analyses').fetchone()[0] == 3
    assert workspace.db.execute('SELECT COUNT(*) FROM planting_points').fetchone()[0] == 3
    assert workspace.db.execute('SELECT COUNT(*) FROM analysis_assets').fetchone()[0] == 3
    assert workspace.db.execute('SELECT COUNT(*) FROM object_cleanup_jobs').fetchone()[0] == 0
    workspace.cleanup.assert_not_called()


@pytest.mark.parametrize('status', ['assigned', 'planted', 'dead', 'skipped', 'completed', 'Planted', '', None])
def test_used_or_unknown_status_is_locked_even_without_a_history_link(workspace, status):
    workspace.db.execute('UPDATE planting_points SET status=? WHERE id=1', (status,))
    workspace.db.commit()
    with pytest.raises(ValueError, match='cannot be deleted'):
        database.delete_analysis(1)
    assert_preserved(workspace)


@pytest.mark.parametrize('column,value', [
    ('planted_at','2026-10-10T08:00:00Z'), ('planted_date','2026-10-10'),
    ('death_at','2026-10-15T08:00:00Z'), ('death_reason','Storm'),
])
def test_legacy_field_evidence_locks_an_apparently_planned_point(workspace, column, value):
    workspace.db.execute(f'UPDATE planting_points SET {column}=? WHERE id=1', (value,))
    workspace.db.commit()
    with pytest.raises(ValueError, match='cannot be deleted'):
        database.delete_analysis(1)
    assert_preserved(workspace)


@pytest.mark.parametrize('table', ['planter_assignment_points','planting_events','point_death_records'])
def test_any_linked_field_history_locks_the_entire_analysis(workspace, table):
    workspace.db.execute(f'INSERT INTO {table}(id,planting_point_id) VALUES (1,1)')
    workspace.db.commit()
    with pytest.raises(ValueError, match='cannot be deleted'):
        database.delete_analysis(1)
    assert workspace.db.execute(f'SELECT COUNT(*) FROM {table}').fetchone()[0] == 1
    assert_preserved(workspace)


def test_released_assignments_and_closed_monitored_plantings_remain_protected(workspace):
    workspace.db.executescript('''
        UPDATE planting_points SET deleted_at='2026-10-15' WHERE id=1;
        INSERT INTO planter_assignment_points VALUES (1,1,'completed','2026-10-15');
        INSERT INTO planting_events VALUES (1,1,'2026-10-15');
        INSERT INTO monitoring_observations VALUES (1,1);
    ''')
    with pytest.raises(ValueError):
        database.delete_analysis(1)
    assert workspace.db.execute('SELECT COUNT(*) FROM monitoring_observations').fetchone()[0] == 1
    assert_preserved(workspace)


def test_unused_analysis_deletes_only_its_planned_points_and_queues_its_images(workspace):
    # A used point on another analysis does not block this unused one.
    workspace.db.execute("UPDATE planting_points SET status='planted' WHERE id=3")
    workspace.db.commit()
    database.delete_analysis(1)
    assert [row[0] for row in workspace.db.execute('SELECT id FROM analyses ORDER BY id')] == [2,3]
    assert [row[0] for row in workspace.db.execute('SELECT id FROM planting_points')] == [3]
    assert [row[0] for row in workspace.db.execute('SELECT object_key FROM analysis_assets')] == ['other-fixture']
    assert {row[0] for row in workspace.db.execute('SELECT object_key FROM object_cleanup_jobs')} == {'fixture-original','fixture-overlay'}
    workspace.cleanup.assert_called_once_with()


def test_an_analysis_without_points_can_be_deleted(workspace):
    database.delete_analysis(3)
    assert workspace.db.execute('SELECT COUNT(*) FROM analyses WHERE id=3').fetchone()[0] == 0
    assert workspace.db.execute('SELECT COUNT(*) FROM planting_points').fetchone()[0] == 3


def test_direct_delete_api_cannot_bypass_the_lock_or_staff_authorization(workspace, monkeypatch):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from api.routes import analyses
    workspace.db.execute("UPDATE planting_points SET death_at='2026-10-15' WHERE id=1")
    workspace.db.commit()
    app = FastAPI()
    app.include_router(analyses.router, prefix='/api/analyses')
    user = Mock(return_value=None)
    monkeypatch.setattr(analyses, 'get_user_by_session_token', user)
    with TestClient(app) as client:
        assert client.delete('/api/analyses/1').status_code == 401
        user.return_value = {'id':7,'role':'planter'}
        assert client.delete('/api/analyses/1').status_code == 403
        user.return_value = {'id':7,'role':'lgu'}
        response = client.delete('/api/analyses/1')
        assert response.status_code == 409
        assert 'cannot be deleted' in response.json()['detail']
    assert_preserved(workspace)
