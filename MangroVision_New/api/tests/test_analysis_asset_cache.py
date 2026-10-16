"""Analysis image revisions cannot reuse stale browser or storage contents."""
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from mangrovision_db import storage
import planting_database as database


class AnalysisAssetCacheTests(unittest.TestCase):
    def test_changed_pixels_have_distinct_paths_even_with_the_same_processing_key(self):
        with patch.object(storage, '_put_payload') as upload:
            green = storage._upload_bytes('previews/same-run', 'visualization_preview', b'green', 'image/webp')
            orange = storage._upload_bytes('previews/same-run', 'visualization_preview', b'orange', 'image/webp')
        self.assertNotEqual(green.object_key, orange.object_key)
        self.assertIn(green.sha256, green.object_key)
        self.assertIn(orange.sha256, orange.object_key)
        self.assertEqual(upload.call_count, 2)

    def test_legacy_image_revalidation_gets_a_distinct_signed_url(self):
        storage._SIGNED_URL_CACHE.clear()
        self.addCleanup(storage._SIGNED_URL_CACHE.clear)
        client = MagicMock()
        client.generate_presigned_url.side_effect = lambda _, Params, **kw: (
            'revalidated' if Params.get('ResponseCacheControl') else 'old-cacheable'
        )
        settings = SimpleNamespace(s3_bucket='test', s3_presigned_url_ttl_seconds=3600)
        with patch.object(storage, 's3_client', return_value=client), \
             patch.object(storage, 'get_settings', return_value=settings):
            old = storage.signed_download_url('legacy/visualization.webp')
            fresh = storage.signed_download_url('legacy/visualization.webp', cache_control='private, no-cache')
            self.assertNotEqual(old, fresh)
            self.assertEqual(storage.signed_download_url('legacy/visualization.webp', cache_control='private, no-cache'), fresh)
        self.assertEqual(client.generate_presigned_url.call_count, 2)

    def test_history_preview_and_jpeg_both_request_revalidation(self):
        conn = MagicMock()
        conn.execute.return_value.fetchall.return_value = [
            dict(kind=kind, object_key=kind, content_type='image/jpeg', byte_size=100, sha256='digest')
            for kind in ('original', 'visualization', 'visualization_preview')
        ]
        with patch.object(database, '_get_connection', return_value=conn), \
             patch.object(database, 'get_settings', return_value=SimpleNamespace(s3_presigned_url_ttl_seconds=3600)), \
             patch.object(database, 'signed_download_url', side_effect=lambda key, **kw: key) as sign:
            assets = database.get_analysis_asset_urls(39)
        self.assertEqual(len(assets), 3)
        self.assertTrue(all(call.kwargs['cache_control'] == 'private, no-cache' for call in sign.call_args_list))
        conn.close.assert_called_once()
