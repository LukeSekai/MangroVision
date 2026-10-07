"""History summaries must use result previews without loading full analyses."""
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
from shapely.geometry import Point, box, mapping, shape

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import planting_database as database


class SpatialTestConnection:
    """Evaluate production spatial predicates with GEOS in an isolated fixture."""
    def __init__(self, connection):
        self.connection = connection
        def geometry(value):
            return shape(json.loads(value)) if value else None
        connection.create_function('ST_GeomFromGeoJSON', 1, lambda value: value)
        connection.create_function('ST_SetSRID', 2, lambda value, srid: value)
        connection.create_function('ST_Intersects', 2, lambda a, b: int(geometry(a).intersects(geometry(b))) if a and b else 0)
        connection.create_function('ST_Touches', 2, lambda a, b: int(geometry(a).touches(geometry(b))) if a and b else 0)
        connection.create_function('ST_Covers', 2, lambda a, b: int(geometry(a).covers(geometry(b))) if a and b else 0)

    def execute(self, sql, parameters=()):
        return self.connection.execute(sql.replace('extensions.', ''), parameters)

    def close(self):
        self.connection.close()


class AnalysisHistoryTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.path = Path(directory.name) / 'history.sqlite'
        with self.connect() as conn:
            conn.executescript('''
                CREATE TABLE analyses (id INTEGER PRIMARY KEY, image_name TEXT,
                    analyzed_at TEXT, hexagon_count INTEGER, center_lat REAL, center_lon REAL,
                    plantable_area_m2 REAL, canopy_area_m2 REAL, canopy_coverage_pct REAL,
                    total_area_m2 REAL, analysis_detail_json TEXT,
                    footprint TEXT, footprint_quality TEXT);
                CREATE TABLE analysis_assets (analysis_id INTEGER, kind TEXT,
                    object_key TEXT, lifecycle_state TEXT);
                CREATE TABLE planting_points (id INTEGER PRIMARY KEY, location TEXT, deleted_at TEXT);
            ''')
            conn.executemany('INSERT INTO analyses VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, NULL, NULL)', [
                (52, 'Analysis 52', '2026-10-07', 50, 10.8, 122.6, 175, 90, 30, 300,
                 json.dumps({'source_image_name': 'QJBJ.JPG'})),
                (51, 'ITBH.JPG', '2026-10-06', 0, None, None, 0, None, None, 300, None),
                (50, 'Legacy', '2026-10-05', 10, None, None, 50, 150, 50, 300, None),
            ])
            conn.executemany('INSERT INTO analysis_assets VALUES (?, ?, ?, ?)', [
                (52, 'visualization', 'full-result', 'ready'),
                (52, 'original_preview', 'original-photo', 'ready'),
                (52, 'visualization_preview', 'small-result', 'ready'),
                (51, 'visualization_preview', 'unfinished-result', 'pending'),
                (51, 'original_preview', 'photo-only', 'ready'),
                (50, 'visualization', 'legacy-result', 'ready'),
            ])

    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        self.addCleanup(conn.close)
        return conn

    def spatial_connect(self):
        return SpatialTestConnection(self.connect())

    def test_history_prefers_ready_result_preview_and_falls_back_to_full_result(self):
        with patch.object(database, '_get_connection', self.spatial_connect), \
             patch.object(database, 'signed_download_url', side_effect=lambda key: f'https://preview.example/{key}') as sign:
            results = database.list_analysis_summaries(include_previews=True)
        self.assertEqual([row['id'] for row in results], [52, 51, 50])
        self.assertEqual(results[0]['source_image_name'], 'QJBJ.JPG')
        self.assertEqual(results[0]['canopy_coverage_pct'], 30)
        self.assertTrue(results[0]['result_preview_url'].endswith('/small-result'))
        self.assertIsNone(results[1]['result_preview_url'])
        self.assertTrue(results[2]['result_preview_url'].endswith('/legacy-result'))
        self.assertEqual(sign.call_count, 2)
        self.assertNotIn('object_key', results[0])

    def test_history_only_includes_earlier_overlapping_boundaries(self):
        footprints = {
            52: mapping(box(122, 10, 122.001, 10.001)),
            51: mapping(box(122.0005, 10.0005, 122.002, 10.002)),
            50: mapping(box(122.002, 10, 122.003, 10.001)),
        }
        with self.connect() as conn:
            for analysis_id, footprint in footprints.items():
                conn.execute('UPDATE analyses SET footprint=?, footprint_quality=? WHERE id=?',
                             (json.dumps(footprint), 'matched_image_corners', analysis_id))
        with patch.object(database, '_get_connection', self.spatial_connect), \
             patch.object(database, 'signed_download_url', return_value='preview'):
            rows = {row['id']: row for row in database.list_analysis_summaries(include_previews=True)}
        self.assertEqual([item['id'] for item in rows[52]['previous_analyses']], [51])
        self.assertEqual(rows[51]['previous_analyses'], [])  # 52 was processed later; 50 only touches an edge.
        self.assertEqual(rows[50]['previous_analyses'], [])
        self.assertTrue(rows[52]['area_history_available'])

    def test_equal_dates_use_id_order_and_missing_boundaries_are_reported(self):
        footprint = json.dumps(mapping(box(122, 10, 122.001, 10.001)))
        with self.connect() as conn:
            conn.execute('UPDATE analyses SET footprint=?, analyzed_at=? WHERE id IN (51,52)', (footprint, '2026-10-07'))
        with patch.object(database, '_get_connection', self.spatial_connect), \
             patch.object(database, 'signed_download_url', return_value='preview'):
            rows = {row['id']: row for row in database.list_analysis_summaries(include_previews=True)}
        self.assertEqual([item['id'] for item in rows[52]['previous_analyses']], [51])
        self.assertEqual(rows[51]['previous_analyses'], [])
        self.assertTrue(rows[52]['area_history_incomplete'])
        self.assertFalse(rows[50]['area_history_available'])

    def test_area_context_preserves_polygon_holes_and_counts_boundary_not_deleted_points(self):
        # A point inside the bounding box but outside the actual angled image is excluded.
        footprint = {'type': 'Polygon', 'coordinates': [
            [[122,10], [122.002,10], [122,10.002], [122,10]],
            [[122.0002,10.0002], [122.0002,10.0004], [122.0004,10.0004], [122.0004,10.0002], [122.0002,10.0002]],
        ]}
        points = [(1,122.0001,10.0001,None), (2,122.002,10,None),
                  (3,122.0018,10.0018,None), (4,122.0003,10.0003,None),
                  (5,122.0001,10.0001,'2026-10-07')]
        with self.connect() as conn:
            conn.execute('UPDATE analyses SET footprint=? WHERE id=52', (json.dumps(footprint),))
            for point_id, lon, lat, deleted in points:
                conn.execute('INSERT INTO planting_points VALUES (?,?,?)',
                             (point_id, json.dumps(mapping(Point(lon, lat))), deleted))
        with patch.object(database, '_get_connection', self.spatial_connect):
            result = database.get_analysis_area_context(footprint)
        self.assertEqual(result['saved_point_count'], 2)
        self.assertEqual([item['id'] for item in result['analyses']], [52])

    def test_area_context_supports_multipolygons_and_rejects_invalid_locations(self):
        footprint = {'type': 'MultiPolygon', 'coordinates': [
            mapping(box(122,10,122.001,10.001))['coordinates'],
            mapping(box(123,11,123.001,11.001))['coordinates'],
        ]}
        with self.connect() as conn:
            conn.execute('INSERT INTO planting_points VALUES (1,?,NULL)', (json.dumps(mapping(Point(123.0005,11.0005))),))
        with patch.object(database, '_get_connection', self.spatial_connect):
            self.assertEqual(database.get_analysis_area_context(footprint)['saved_point_count'], 1)
        with patch.object(database, '_get_connection') as connect:
            with self.assertRaises(ValueError):
                database.get_analysis_area_context(mapping(box(181,10,182,11)))
            connect.assert_not_called()

    def test_assignment_summaries_do_not_sign_or_query_assets(self):
        with self.connect() as conn:
            conn.execute('DROP TABLE analysis_assets')
        with patch.object(database, '_get_connection', self.connect), patch.object(database, 'signed_download_url') as sign:
            results = database.list_analysis_summaries()
        sign.assert_not_called()
        self.assertEqual(len(results), 3)
        self.assertNotIn('result_preview_url', results[0])

    def test_preview_endpoint_preserves_staff_authorization(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.routes import analyses
        app = FastAPI()
        app.include_router(analyses.router, prefix='/api/analyses')
        with TestClient(app) as client, patch.object(analyses, 'get_user_by_session_token') as user, \
             patch.object(analyses, 'list_analysis_summaries', return_value=[]) as summaries:
            user.return_value = None
            self.assertEqual(client.get('/api/analyses/?include_previews=true').status_code, 401)
            user.return_value = {'role': 'planter'}
            self.assertEqual(client.get('/api/analyses/?include_previews=true').status_code, 403)
            summaries.assert_not_called()
            user.return_value = {'role': 'lgu'}
            self.assertEqual(client.get('/api/analyses/?include_previews=true').json(), [])
            summaries.assert_called_once_with(include_previews=True)

    def test_area_check_requires_staff_and_returns_actionable_errors(self):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.routes import analyses
        app = FastAPI()
        app.include_router(analyses.router, prefix='/api/analyses')
        request = {'footprint': mapping(box(122,10,122.001,10.001))}
        with TestClient(app) as client, patch.object(analyses, 'get_user_by_session_token') as user, \
             patch.object(analyses, 'get_analysis_area_context') as context:
            user.return_value = None
            self.assertEqual(client.post('/api/analyses/area-context', json=request).status_code, 401)
            user.return_value = {'role': 'planter'}
            self.assertEqual(client.post('/api/analyses/area-context', json=request).status_code, 403)
            context.assert_not_called()
            user.return_value = {'role': 'lgu'}
            context.return_value = {'analyses': [], 'saved_point_count': 7}
            self.assertEqual(client.post('/api/analyses/area-context', json=request).json()['saved_point_count'], 7)
            context.side_effect = ValueError('invalid boundary')
            self.assertEqual(client.post('/api/analyses/area-context', json=request).status_code, 400)
            context.side_effect = RuntimeError('private database error')
            response = client.post('/api/analyses/area-context', json=request)
            self.assertEqual(response.status_code, 503)
            self.assertIn('try Run Analysis again', response.json()['detail'])
            self.assertNotIn('private database error', response.text)


if __name__ == '__main__':
    unittest.main()
