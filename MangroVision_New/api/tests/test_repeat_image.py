"""Repeat images require consent and cannot create or replace planting points."""
import base64
import hashlib
import json
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from fastapi import HTTPException

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import planting_database as database
from api.routes import processing


class RepeatImageTests(unittest.TestCase):
    def setUp(self):
        self.conn = sqlite3.connect(':memory:')
        self.conn.row_factory = sqlite3.Row
        self.addCleanup(self.conn.close)
        self.conn.create_function('nextval', 1, lambda _: 200)
        self.conn.executescript('''
            CREATE TABLE analyses (
                id INTEGER PRIMARY KEY, user_id INTEGER, image_name TEXT, analysis_number INTEGER,
                analyzed_at TEXT, center_lat REAL, center_lon REAL, altitude_m REAL, gsd_cm REAL,
                coverage_w_m REAL, coverage_h_m REAL, total_area_m2 REAL, canopy_count INTEGER,
                canopy_area_m2 REAL, canopy_coverage_pct REAL, polygon_count INTEGER,
                danger_area_m2 REAL, danger_pct REAL, plantable_area_m2 REAL, plantable_pct REAL,
                hexagon_count INTEGER, ai_confidence REAL, canopy_buffer_m REAL, hexagon_size_m REAL,
                forbidden_filtered INTEGER, eroded_filtered INTEGER, analysis_detail_json TEXT,
                species TEXT, planting_distance_m REAL, footprint_geojson TEXT, footprint_quality TEXT,
                site_zone_id INTEGER);
            CREATE TABLE analysis_assets (analysis_id INTEGER, kind TEXT, sha256 TEXT, lifecycle_state TEXT);
            CREATE TABLE planting_points (id INTEGER PRIMARY KEY, analysis_id INTEGER,
                point_num INTEGER, latitude REAL, longitude REAL, pixel_x INTEGER, pixel_y INTEGER,
                buffer_m REAL, area_m2 REAL, status TEXT, deleted_at TEXT);
            CREATE TABLE assignments (point_id INTEGER, status TEXT);
            CREATE TABLE monitoring (point_id INTEGER, status TEXT);
        ''')
        self.identity = {'source_image_sha256': 'a'*64, 'source_original_sha256': 'b'*64}
        self.conn.execute('''INSERT INTO analyses
            (id,image_name,analysis_number,analyzed_at,center_lat,center_lon,hexagon_count,analysis_detail_json)
            VALUES (52,'Analysis 52',52,'2026-10-07',10.8,122.6,1,?)''',
            (json.dumps({'source_image_name': 'QJBJ.JPG', **self.identity}),))
        self.conn.execute("INSERT INTO planting_points VALUES (100,52,1,10.8,122.6,10,10,1,3,'planted',NULL)")
        self.conn.execute("INSERT INTO assignments VALUES (100,'completed')")
        self.conn.execute("INSERT INTO monitoring VALUES (100,'alive')")
        self.conn.commit()

    def find(self, name='QJBJ.JPG', lat=10.8, lon=122.6, identity=None):
        identity = self.identity if identity is None else identity
        return database._find_repeat_image_analyses(self.conn, name, lat, lon,
            identity.get('source_image_sha256'), identity.get('source_original_sha256'))

    def test_content_identity_recognizes_renamed_images(self):
        self.assertEqual([row['id'] for row in self.find('RENAMED.JPG', 11,123)], [52])

    def test_different_image_with_the_same_filename_is_not_a_repeat(self):
        self.assertEqual(self.find(identity={'source_image_sha256': 'c'*64, 'source_original_sha256': 'd'*64}), [])

    def test_legacy_original_asset_matches_content_even_after_renaming(self):
        self.conn.execute('UPDATE analyses SET analysis_detail_json=?', (json.dumps({'source_image_name':'QJBJ.JPG'}),))
        self.conn.execute("INSERT INTO analysis_assets VALUES (52,'original',?,'ready')", ('b'*64,))
        self.assertEqual([row['id'] for row in self.find('RENAMED.JPG')], [52])
        self.assertEqual(self.find(identity={'source_image_sha256':'c'*64, 'source_original_sha256':'d'*64}), [])

    def test_assetless_legacy_fallback_requires_matching_name_and_location(self):
        self.conn.execute('UPDATE analyses SET analysis_detail_json=?', (json.dumps({'source_image_name':'QJBJ.JPG'}),))
        self.assertEqual([row['id'] for row in self.find(identity={})], [52])
        self.assertEqual(self.find(lat=11, identity={}), [])
        self.assertEqual(self.find(name='OTHER.JPG', identity={}), [])

    def test_repeat_save_preserves_original_points_assignments_and_monitoring(self):
        candidates = [{'_gps_lat':10.81, '_gps_lon':122.61, 'center':(20,20)}]
        results = {'_analysis_detail_json': json.dumps(self.identity), 'hexagon_count':1}
        with patch.object(database, '_matching_project_site_id', return_value=None), \
             patch.object(database, '_delete_analysis_rows', side_effect=AssertionError('must not replace earlier analysis')), \
             patch.object(database, '_planting_spacing_indexes', side_effect=AssertionError('must not prepare new points')), \
             patch.object(database, '_cleanup_same_species_spacing_conflicts', side_effect=AssertionError('must not change earlier points')), \
             patch.object(database, '_cleanup_cross_species_spacing_conflicts', side_effect=AssertionError('must not change earlier points')):
            analysis_id, new_points, _ = database._save_analysis_with_connection(
                self.conn,'RENAMED.JPG',10.8,122.6,results,candidates,user_id=1)
        self.assertNotEqual(analysis_id,52)
        self.assertEqual(new_points,0)
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM analyses').fetchone()[0],2)
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM planting_points').fetchone()[0],1)
        self.assertEqual(self.conn.execute('SELECT analysis_id,status FROM planting_points WHERE id=100').fetchone()[:],(52,'planted'))
        self.assertEqual(self.conn.execute('SELECT status FROM assignments').fetchone()[0],'completed')
        self.assertEqual(self.conn.execute('SELECT status FROM monitoring').fetchone()[0],'alive')
        saved = self.conn.execute('SELECT hexagon_count,analysis_detail_json FROM analyses WHERE id=?',(analysis_id,)).fetchone()
        self.assertEqual(saved['hexagon_count'],0)
        self.assertTrue(json.loads(saved['analysis_detail_json'])['repeat_image']['points_not_added'])

    def test_normal_image_can_still_add_its_points(self):
        existing, spacing = MagicMock(), MagicMock()
        existing.has_neighbor.return_value = False
        spacing.has_conflict.return_value = False
        with patch.object(database, '_matching_project_site_id', return_value=None), \
             patch.object(database, '_planting_spacing_indexes', return_value=(existing,spacing)), \
             patch.object(database, '_cleanup_same_species_spacing_conflicts'), \
             patch.object(database, '_cleanup_cross_species_spacing_conflicts'):
            _, count, _ = database._save_analysis_with_connection(self.conn,'OTHER.JPG',10.81,122.61,
                {'_analysis_detail_json': {'source_image_sha256':'c'*64, 'source_original_sha256':'d'*64}},
                [{'_gps_lat':10.81,'_gps_lon':122.61,'center':(20,20)}],user_id=1)
        self.assertEqual(count,1)
        self.assertEqual(self.conn.execute('SELECT COUNT(*) FROM planting_points').fetchone()[0],2)

    def test_identity_matches_the_original_storage_encoding(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'QJBJ.jpg'
            cv2.imwrite(str(path),np.full((16,20,3),120,np.uint8))
            identity = processing._image_identity(path)
            data = processing._encode_image_data_url(cv2.imread(str(path)),'.jpg')
            self.assertEqual(identity['source_original_sha256'],hashlib.sha256(base64.b64decode(data.split(',',1)[1])).hexdigest())

    def test_repeat_requires_confirmation_and_always_excludes_points(self):
        repeats = [{'id':52}]
        with patch.object(processing,'_image_identity',return_value=self.identity), \
             patch.object(processing,'find_repeat_image_analyses',return_value=repeats):
            with self.assertRaises(HTTPException) as error:
                processing._check_repeat_image(Path('photo.jpg'),'photo.jpg',{'latitude':10.8,'longitude':122.6})
            self.assertEqual(error.exception.status_code,409)
            self.assertEqual(processing._check_repeat_image(Path('photo.jpg'),'photo.jpg',{},allow_repeat=True)[1],repeats)
        points = [{'center':(1,2)}]
        self.assertEqual(processing._exclude_repeat_image_points(points,repeats),[])
        self.assertEqual(processing._exclude_repeat_image_points(points,[]),points)

    def test_save_rechecks_an_image_saved_while_the_preview_was_open(self):
        cached = {'owner_id':1,'center_lat':10.8,'center_lon':122.6,'image_name':'QJBJ.JPG',
                  'safe_hexagons':[], 'results':{'_analysis_detail_json':json.dumps(self.identity)}}
        with patch.object(processing,'_require_lgu_user',return_value={'id':1}), \
             patch.object(processing,'_get_processing_cache',return_value=cached), \
             patch.object(processing,'find_repeat_image_analyses',return_value=[{'id':52}]), \
             patch.object(processing,'save_analysis') as save:
            with self.assertRaises(HTTPException) as error:
                processing.save_processed_analysis(processing.SaveProcessedAnalysisRequest(analysis_key='preview'))
        self.assertEqual(error.exception.status_code,409)
        self.assertIn('No planting points will be added',error.exception.detail)
        save.assert_not_called()

    def test_failed_repeat_save_check_is_retryable_and_does_not_write(self):
        cached = {'owner_id':1,'center_lat':10.8,'center_lon':122.6,'image_name':'QJBJ.JPG',
                  'safe_hexagons':[], 'results':{'_analysis_detail_json':json.dumps(self.identity)}}
        with patch.object(processing,'_require_lgu_user',return_value={'id':1}), \
             patch.object(processing,'_get_processing_cache',return_value=cached), \
             patch.object(processing,'find_repeat_image_analyses',side_effect=RuntimeError('private error')), \
             patch.object(processing,'save_analysis') as save:
            with self.assertRaises(HTTPException) as error:
                processing.save_processed_analysis(processing.SaveProcessedAnalysisRequest(analysis_key='preview'))
        self.assertEqual(error.exception.status_code,503)
        self.assertIn('Try saving again',error.exception.detail)
        self.assertNotIn('private error',error.exception.detail)
        save.assert_not_called()


if __name__ == '__main__':
    unittest.main()
