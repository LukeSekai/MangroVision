"""Hosted field links must coexist with the local Cloudflare workflow."""
import os
from pathlib import Path
import sys
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import MagicMock, patch

from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api.routes import share


class CloudflaredDiscoveryTests(unittest.TestCase):
    def setUp(self):
        self.directory = TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name) / 'MangroVision_New'
        self.binary = self.root / '.dev-server' / 'tools' / 'cloudflared.exe'
        self.binary.parent.mkdir(parents=True)
        self.binary.touch()
        self.addCleanup(patch.stopall)
        patch.object(share, '__file__', str(self.root / 'api' / 'routes' / 'share.py')).start()
        patch.dict(os.environ, dict.fromkeys(
            ['CLOUDFLARED_BIN', 'LOCALAPPDATA', 'PROGRAMFILES', 'PROGRAMFILES(X86)'], '')).start()

    def test_testing_executable_is_found_without_path_installation(self):
        with patch.object(share.shutil, 'which', return_value=None):
            self.assertEqual(Path(share._find_cloudflared()).resolve(), self.binary.resolve())

    def test_explicit_override_keeps_priority_over_path_and_testing_copy(self):
        custom = Path(self.directory.name) / 'custom.exe'
        custom.touch()
        with patch.dict(os.environ, {'CLOUDFLARED_BIN': str(custom)}), \
                patch.object(share.shutil, 'which', return_value=str(self.binary)):
            self.assertEqual(Path(share._find_cloudflared()), custom)

    def test_existing_path_installation_keeps_priority(self):
        installed = Path(self.directory.name) / 'installed.exe'
        installed.touch()
        with patch.object(share.shutil, 'which', return_value=str(installed)):
            self.assertEqual(Path(share._find_cloudflared()), installed)

    def test_missing_executable_keeps_installation_error(self):
        self.binary.unlink()
        with patch.object(share.shutil, 'which', return_value=None), \
                self.assertRaises(HTTPException) as error:
            share._find_cloudflared()
        self.assertEqual(error.exception.status_code, 503)
        self.assertIn('cloudflared was not found', error.exception.detail)


class FieldShareModeTests(unittest.TestCase):
    def test_hosted_status_generate_and_stop_reuse_public_field_route(self):
        with patch.object(share, '_HOSTED_FRONTEND_URL', 'https://workspace.example'), \
                patch.object(share, 'get_user_by_session_token', return_value={'id': 1}), \
                patch.object(share, '_find_cloudflared') as find_binary, \
                patch.object(share.subprocess, 'Popen') as start_process:
            for action in [share.get_field_link_status, share.start_cloudflare_field_link,
                           share.stop_cloudflare_field_link]:
                payload = action()
                self.assertEqual(payload['field_url'], 'https://workspace.example/field')
                self.assertEqual(payload['provider'], 'hosted')
                self.assertTrue(payload['active'])
                self.assertEqual(payload['cloudflare_url'], '')
            find_binary.assert_not_called()
            start_process.assert_not_called()

    def test_local_mode_still_reuses_managed_cloudflare_tunnel(self):
        process = MagicMock()
        process.poll.return_value = None
        with patch.object(share, '_HOSTED_FRONTEND_URL', ''), \
                patch.object(share, 'get_user_by_session_token', return_value={'id': 1}), \
                patch.object(share, '_tunnel_proc', process), \
                patch.object(share, '_tunnel_url', 'https://local-example.trycloudflare.com'), \
                patch.object(share, '_find_cloudflared') as find_binary:
            payload = share.start_cloudflare_field_link()
            self.assertEqual(payload['provider'], 'cloudflare')
            self.assertEqual(payload['field_url'], 'https://local-example.trycloudflare.com/field')
            self.assertTrue(payload['active'])
            find_binary.assert_not_called()

    def test_hosted_mode_still_requires_authenticated_planner(self):
        with patch.object(share, '_HOSTED_FRONTEND_URL', 'https://workspace.example'), \
                patch.object(share, 'get_user_by_session_token', return_value=None):
            for action in [share.get_field_link_status, share.start_cloudflare_field_link,
                           share.stop_cloudflare_field_link]:
                with self.subTest(action=action.__name__), self.assertRaises(HTTPException) as error:
                    action()
                self.assertEqual(error.exception.status_code, 401)


if __name__ == '__main__':
    unittest.main()
