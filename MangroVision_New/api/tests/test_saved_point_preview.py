"""Saved-point exclusions must agree before rendering and when saving."""
import math
import sqlite3
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import planting_database as database
from api.routes import processing


class SavedPointPreviewTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.path = Path(self.directory.name) / 'points.sqlite'
        with self.connect() as conn:
            conn.executescript('''
                CREATE TABLE analyses (id INTEGER PRIMARY KEY, species TEXT, planting_distance_m REAL);
                CREATE TABLE planting_points (
                    analysis_id INTEGER, latitude REAL, longitude REAL, status TEXT, deleted_at TEXT);
            ''')
        self.connection_patch = patch.object(database, '_get_connection', self.connect)
        self.connection_patch.start()
        self.addCleanup(self.connection_patch.stop)

    def connect(self):
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        self.addCleanup(conn.close)
        return conn

    def point(self, north_m):
        return {'_gps_lat': 10.8 + north_m / 111320., '_gps_lon': 122.6,
                'center': (north_m * 10, 10)}

    def save_point(self, species, status='planned', deleted=False):
        spacing = {'rhizophora': 2., 'bungalon': 1.}.get(species)
        with self.connect() as conn:
            conn.execute('INSERT INTO analyses VALUES (1, ?, ?)', (species, spacing))
            conn.execute('INSERT INTO planting_points VALUES (1, 10.8, 122.6, ?, ?)',
                         (status, 'deleted' if deleted else None))

    def test_same_species_occupied_spots_excluded_but_lattice_neighbors_kept(self):
        for species, spacing in [('rhizophora', 2.), ('bungalon', 1.)]:
            for status in ['planned', 'planted']:
                with self.subTest(species=species, status=status):
                    with self.connect() as conn:
                        conn.execute('DELETE FROM planting_points')
                        conn.execute('DELETE FROM analyses')
                    self.save_point(species, status)
                    points = [self.point(0), self.point(spacing * .90), self.point(spacing)]
                    kept, removed = processing._thin_hexagons_against_saved_species_points(points, species, spacing)
                    self.assertEqual(kept, [points[2]])
                    self.assertEqual(removed, points[:2])

    def test_cross_species_spacing_still_applies(self):
        self.save_point('rhizophora')
        points = [self.point(1.5), self.point(2.01)]
        kept, removed = processing._thin_hexagons_against_saved_species_points(points, 'bungalon', 1.)
        self.assertEqual(kept, points[1:])
        self.assertEqual(removed, points[:1])

    def test_deleted_points_do_not_block_new_analysis(self):
        self.save_point('rhizophora', deleted=True)
        points = [self.point(0), self.point(2)]
        self.assertEqual(processing._thin_hexagons_against_saved_species_points(points, 'rhizophora', 2.),
                         (points, []))

    def test_legacy_speciesless_records_still_block_occupied_locations(self):
        self.save_point(None)
        points = [self.point(0), self.point(2.1)]
        kept, removed = processing._thin_hexagons_against_saved_species_points(points, 'rhizophora', 2.)
        self.assertEqual(kept, points[1:])
        self.assertEqual(removed, points[:1])

    def test_visual_overlay_receives_only_surviving_points(self):
        from shapely.geometry import Point
        self.save_point('rhizophora')
        points = [self.point(0), self.point(1), self.point(2), self.point(4)]
        kept, _ = processing._thin_hexagons_against_saved_species_points(points, 'rhizophora', 2.)
        class Detector:
            def create_hexagon(self, x, y, radius):
                return Point(x, y).buffer(radius)
        displayed = processing._build_authoritative_visualization_hexagons(
            Detector(), kept, lambda lat, lon: ((lat - 10.8) * 1113200, 10), 2/math.sqrt(3), .1)
        self.assertEqual(len(displayed), 2)
        self.assertEqual([h['_gps_lat'] for h in displayed], [h['_gps_lat'] for h in points[2:]])
        self.assertAlmostEqual(displayed[0]['center'][0], 20)

    def test_database_failure_does_not_return_unchecked_points(self):
        with patch.object(database, '_get_connection', side_effect=RuntimeError('offline')):
            with self.assertRaises(processing.HTTPException) as error:
                processing._thin_hexagons_against_saved_species_points([self.point(0)], 'rhizophora', 2.)
        self.assertEqual(error.exception.status_code, 503)

    def test_stale_preview_is_rejected_before_save(self):
        self.save_point('rhizophora')
        cached = {'center_lat': 10.8, 'center_lon': 122.6,
                  'safe_hexagons': [self.point(0)],
                  'results': {'species': 'rhizophora', 'planting_distance_m': 2.}}
        with patch.object(processing, '_require_lgu_user', return_value={'id': 1}), \
             patch.object(processing, '_get_processing_cache', return_value=cached), \
             patch.object(processing, 'save_analysis') as save:
            with self.assertRaises(processing.HTTPException) as error:
                processing.save_processed_analysis(processing.SaveProcessedAnalysisRequest(analysis_key='preview'))
            self.assertEqual(error.exception.status_code, 409)
            save.assert_not_called()


if __name__ == '__main__':
    unittest.main()
