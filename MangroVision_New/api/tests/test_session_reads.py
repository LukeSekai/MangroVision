"""Concurrent session checks must not write/lock the session on every read."""
import sys
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import planting_database as database
from mangrovision_db.compat import CompatRow


class SessionReadTests(unittest.TestCase):
    def setUp(self):
        self.conn = MagicMock()
        self.row = {'id': 7, 'role': 'lgu', '_auth_session_id': 23,
                    '_auth_participant_slot': 2, '_auth_heartbeat_due': False}
        self.conn.execute.return_value.fetchone.side_effect = lambda: CompatRow(tuple(self.row), tuple(self.row.values()))

    def test_recent_staff_and_planter_reads_do_not_write(self):
        for subject in ('user', 'planter'):
            with self.subTest(subject=subject), patch.object(database, '_get_connection', return_value=self.conn):
                result = database._get_subject_by_session(subject, 'test-session-only')
                self.assertEqual(result['id'], 7)
                self.assertNotIn('_auth_session_id', result)
                if subject == 'planter':
                    self.assertEqual(result['participant_slot'], 2)
                self.conn.execute.assert_called_once()
                self.conn.commit.assert_not_called()
                self.conn.close.assert_called_once()
                self.conn.reset_mock()

    def test_old_session_updates_heartbeat_then_releases_connection(self):
        self.row['_auth_heartbeat_due'] = True
        with patch.object(database, '_get_connection', return_value=self.conn):
            self.assertEqual(database._get_subject_by_session('user', 'test-session-only')['id'], 7)
        self.assertEqual(self.conn.execute.call_count, 2)
        self.conn.commit.assert_called_once()
        self.conn.close.assert_called_once()

    def test_missing_session_and_query_failure_release_connection(self):
        self.conn.execute.return_value.fetchone.side_effect = lambda: None
        with patch.object(database, '_get_connection', return_value=self.conn):
            self.assertIsNone(database._get_subject_by_session('user', 'test-session-only'))
        self.conn.close.assert_called_once()
        self.conn.reset_mock()
        self.conn.execute.side_effect = RuntimeError('database unavailable')
        with patch.object(database, '_get_connection', return_value=self.conn):
            with self.assertRaises(RuntimeError):
                database._get_subject_by_session('user', 'test-session-only')
        self.conn.close.assert_called_once()

    def test_database_pool_timeout_returns_json_without_exception_details(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.error_responses import install_error_responses
        from sqlalchemy.exc import TimeoutError
        app = FastAPI()
        install_error_responses(app)
        @app.get('/api/test')
        def unavailable():
            raise TimeoutError('private diagnostic')
        response = TestClient(app).get('/api/test')
        self.assertEqual(response.status_code, 503)
        self.assertEqual(response.headers['cache-control'], 'no-store')
        self.assertNotIn('private diagnostic', response.text)
        self.assertIn('database is busy', response.json()['detail'])


if __name__ == '__main__':
    unittest.main()
