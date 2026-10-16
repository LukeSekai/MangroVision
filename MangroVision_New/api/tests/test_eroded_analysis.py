"""Erosion retains planned locations; permanent hazards still exclude them.

The real analysis/hexagon pipeline runs on a synthetic image. All database,
storage and orthophoto I/O is replaced with isolated fixtures.
"""
import io
import math
import sqlite3
import sys
import tempfile
import unittest
from contextlib import ExitStack, redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import cv2
import numpy as np
from shapely.geometry import Point, box, mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import planting_database as database
from api.routes import processing as p


class ErodedAnalysisTests(unittest.TestCase):
    lat, lon, gsd = 10.8, 122.6, .1
    longitude_scale = 111320 * math.cos(math.radians(lat))

    def gps(self, x, y):
        return self.lat + y*self.gsd/111320, self.lon + x*self.gsd/self.longitude_scale

    def pixels(self, lat, lon):
        return (lon-self.lon)*self.longitude_scale/self.gsd, (lat-self.lat)*111320/self.gsd

    def area(self, x0, y0, x1, y1):
        lat0, lon0 = self.gps(x0, y0)
        lat1, lon1 = self.gps(x1, y1)
        return box(lon0, lat0, lon1, lat1)

    def zones(self, polygon=None):
        return p.ForbiddenZoneFilter(safety_buffer_m=0, geojson_data={
            'type': 'FeatureCollection', 'features': [] if polygon is None else [
                {'type': 'Feature', 'properties': {}, 'geometry': mapping(polygon)},
            ],
        })

    def run_analysis(self, erosion=None, forbidden=None, vegetation=False):
        image = np.full((160, 160, 3), (90, 98, 95), np.uint8)
        if vegetation:
            image[45:115, 45:115] = (20, 170, 40)
        footprint = mapping(self.area(0, 0, 160, 160))
        center_lat, center_lon = self.gps(80, 80)
        metadata = {'has_gps': True, 'has_exif': True, 'camera': {},
                    'gps': {'latitude': center_lat, 'longitude': center_lon,
                            'relative_altitude': 20., 'heading': 0.}}
        match = {'success': True, 'H': np.eye(3), 'validated_H': np.eye(3),
                 'registration_validated': True, 'registration_model': 'projective',
                 'refined_gsd_m_per_pixel': self.gsd, 'projection_rebuilt': False}
        fake_assets = [SimpleNamespace(kind=kind, object_key=f'test/{kind}.jpg')
                       for kind in ('original', 'visualization')]
        with tempfile.TemporaryDirectory() as directory, ExitStack() as stack:
            path = Path(directory)/'isolated-erosion-test.jpg'
            cv2.imwrite(str(path), image)
            replacements = {
                '_build_image_location_preflight': {'can_process': True, 'status': 'inside'},
                '_calibrate_preflight_footprint': {'can_process': True, 'status': 'inside'},
                '_check_repeat_image': ({'source_image_name': path.name}, []),
                '_get_image_alignment': match,
                'zone_revision_fingerprint': 'isolated-test',
                '_load_zone_filters': (self.zones(forbidden), self.zones(erosion)),
                '_load_gis_coverage_geometry': self.area(-100, -100, 260, 260),
                'is_inside_any_orthophoto': True,
                '_orthophoto_canopy_recheck': {'is_canopy': False},
                '_thin_hexagons_against_saved_species_points': None,
                '_build_georeferenced_overlay': {'footprint_geojson': footprint},
                'get_analysis_area_context': {'analyses': [], 'saved_point_count': 0},
                'upload_analysis_data_urls': fake_assets,
                'get_settings': SimpleNamespace(s3_presigned_url_ttl_seconds=900),
            }
            for name, value in replacements.items():
                if name == '_thin_hexagons_against_saved_species_points':
                    stack.enter_context(patch.object(p, name, side_effect=lambda points, *_: (points, [])))
                else:
                    stack.enter_context(patch.object(p, name, return_value=value))
            stack.enter_context(patch.object(p.ExifExtractor, 'extract_all_metadata', return_value=metadata))
            stack.enter_context(patch.object(p.GSDCalculator, 'calculate_gsd_from_metadata', return_value=(self.gsd, {})))
            stack.enter_context(patch.object(p, 'drone_pixel_to_gps_via_homography', side_effect=lambda x,y,*_: self.gps(x,y)))
            stack.enter_context(patch.object(p, '_gps_to_drone_pixel_via_homography', side_effect=lambda lat,lon,*_: self.pixels(lat,lon)))
            stack.enter_context(patch.object(p, 'point_visibility_flags', side_effect=lambda _entry,_coverage,points: [True]*len(points)))
            stack.enter_context(patch.object(p.ortho_matcher, '_ensure_active_ortho', return_value={}))
            stack.enter_context(patch.object(p, 'signed_download_url', side_effect=lambda key: f'/test/{key}'))
            original_renderer = p.HexagonDetector.visualize_results
            rendered_images = []
            def capture_render(instance, results):
                rendered = original_renderer(instance, results)
                rendered_images.append(rendered)
                return rendered
            renderer = stack.enter_context(patch.object(p.HexagonDetector, 'visualize_results',
                autospec=True, side_effect=capture_render))
            cache = stack.enter_context(patch.object(p, '_set_processing_cache'))
            with redirect_stdout(io.StringIO()):
                payload = p._execute_canopy_workflow(path, path.name, 20., 'GENERIC_4K',
                    1., 2/math.sqrt(3), .8, 'hsv', '{}', species='rhizophora')
            cached = cache.call_args.args[1]
            cached['rendered_hexagons'] = renderer.call_args.args[1]['hexagons']
            cached['preview_image'] = rendered_images[-1]
            return payload, cached

    def positions(self, payload):
        return [(row['latitude'], row['longitude']) for row in payload['map']['coordinates']]

    def test_partly_eroded_image_retains_all_planned_points_and_exports(self):
        baseline, _ = self.run_analysis()
        erosion = self.area(0, 80, 160, 160)
        result, cached = self.run_analysis(erosion)
        self.assertGreater(len(self.positions(result)), 0)
        self.assertEqual(self.positions(result), self.positions(baseline))
        rows = result['map']['coordinates']
        blocked = [row for row in rows if row['eroded_unavailable']]
        self.assertGreater(len(blocked), 0)
        self.assertLess(len(blocked), len(rows))
        self.assertEqual(result['metrics']['eroded_unavailable_count'], len(blocked))
        self.assertEqual(result['metrics']['safe_hexagon_count'], len(rows)-len(blocked))
        self.assertEqual(result['metrics']['hexagon_count'], len(rows))
        self.assertEqual(result['metrics']['eroded_filtered_count'], 0)
        self.assertEqual(len(result['exports']['waypoints']), len(rows))
        self.assertEqual(len(result['map']['safe_points_geojson']['features']), len(rows))
        for row, waypoint, feature, stored, rendered in zip(rows, result['exports']['waypoints'],
                result['map']['safe_points_geojson']['features'], cached['safe_hexagons'], cached['rendered_hexagons']):
            # Availability uses the actual saved GPS, not its rounded UI label.
            expected = erosion.covers(Point(stored['_gps_lon'], stored['_gps_lat']))
            self.assertEqual(row['eroded_unavailable'], expected)
            self.assertEqual(waypoint['eroded_unavailable'], expected)
            self.assertEqual(feature['properties']['eroded_unavailable'], expected)
            self.assertEqual(stored['_eroded_unavailable'], expected)
            self.assertEqual(rendered['_eroded_unavailable'], expected)
            self.assertEqual(row['status_label'], 'Not Available for Planting' if expected else 'Planned')

    def test_fully_eroded_image_remains_savable_and_points_return_when_zone_removed(self):
        clear, _ = self.run_analysis()
        blocked, cached = self.run_analysis(self.area(-1, -1, 161, 161))
        self.assertEqual(self.positions(clear), self.positions(blocked))
        self.assertTrue(blocked['can_save'])
        self.assertGreater(blocked['metrics']['eroded_unavailable_count'], 0)
        self.assertEqual(blocked['metrics']['safe_hexagon_count'], 0)
        self.assertEqual(len(cached['safe_hexagons']), len(blocked['map']['coordinates']))
        self.assertTrue(all(h['_eroded_unavailable'] for h in cached['safe_hexagons']))
        self.assertEqual(p._mark_eroded_availability(cached['safe_hexagons'], self.zones()), 0)
        self.assertFalse(any(h['_eroded_unavailable'] for h in cached['safe_hexagons']))

    def test_erosion_never_overrides_forbidden_exclusions(self):
        forbidden = self.area(0, 0, 40, 160)
        baseline, _ = self.run_analysis(forbidden=forbidden)
        result, _ = self.run_analysis(self.area(-1, -1, 161, 161), forbidden=forbidden)
        self.assertEqual(self.positions(result), self.positions(baseline))
        self.assertGreater(result['metrics']['forbidden_filtered_count'], 0)
        self.assertTrue(all(not forbidden.covers(Point(row['longitude'], row['latitude']))
                            for row in result['map']['coordinates']))

    def test_erosion_never_overrides_the_canopy_danger_buffer(self):
        result, cached = self.run_analysis(self.area(-1, -1, 161, 161), vegetation=True)
        mask = cached['results']['danger_mask']
        self.assertTrue(np.any(mask))
        self.assertGreater(len(cached['safe_hexagons']), 0)
        for hexagon in cached['safe_hexagons']:
            x, y = map(round, hexagon['center'])
            self.assertEqual(mask[y, x], 0)
            self.assertTrue(hexagon['_eroded_unavailable'])

    def test_current_saved_point_advisories_clear_when_erosion_is_removed(self):
        rows = [{'id': 1, 'status': 'planned', 'latitude': self.lat, 'longitude': self.lon}]
        with patch.object(database, '_load_warning_zone_geometries', return_value=[]):
            blocked = database._annotate_point_advisories(rows, eroded_point_ids={1})[0]
            cleared = database._annotate_point_advisories(rows, eroded_point_ids=set())[0]
        self.assertEqual(blocked['status'], 'planned')
        self.assertEqual(blocked['map_status'], 'unavailable')
        self.assertEqual(cleared['status'], 'planned')
        self.assertEqual(cleared['map_status'], 'planned')

    def test_save_keeps_eroded_points_planned_and_assignment_gate_tracks_zone_removal(self):
        result, cached = self.run_analysis(self.area(-1, -1, 161, 161))
        conn = sqlite3.connect(':memory:')
        self.addCleanup(conn.close)
        conn.row_factory = sqlite3.Row
        conn.create_function('nextval', 1, lambda _: 1)
        conn.executescript('''
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
            CREATE TABLE planting_points (id INTEGER PRIMARY KEY, analysis_id INTEGER,
                point_num INTEGER, latitude REAL, longitude REAL, pixel_x INTEGER, pixel_y INTEGER,
                buffer_m REAL, area_m2 REAL, status TEXT, deleted_at TEXT);
        ''')
        spatial_index = SimpleNamespace(has_neighbor=lambda *_: False, add=lambda *_: None)
        species_index = SimpleNamespace(has_conflict=lambda *_: False, add=lambda *_: None)
        with patch.object(database, '_matching_project_site_id', return_value=None), \
             patch.object(database, '_find_repeat_image_analyses', return_value=[]), \
             patch.object(database, '_planting_spacing_indexes', return_value=(spatial_index, species_index)), \
             patch.object(database, '_cleanup_same_species_spacing_conflicts'), \
             patch.object(database, '_cleanup_cross_species_spacing_conflicts'):
            _, count, skipped = database._save_analysis_with_connection(conn,
                'isolated-test.jpg', self.lat, self.lon, cached['results'], cached['safe_hexagons'], user_id=1)
        self.assertEqual(count, len(result['map']['coordinates']))
        self.assertEqual(skipped, 0)
        rows = conn.execute('SELECT * FROM planting_points').fetchall()
        self.assertTrue(all(row['status']=='planned' for row in rows))
        point_ids = [row['id'] for row in rows]
        active = {'filter': self.zones(self.area(-1, -1, 161, 161))}
        with patch.object(database, '_is_point_inside_eroded_zone',
                side_effect=lambda lat,lon: not active['filter'].is_safe_location(lat,lon)):
            self.assertIsNotNone(database._first_eroded_unavailable_point(conn, point_ids))
            active['filter'] = self.zones()
            self.assertIsNone(database._first_eroded_unavailable_point(conn, point_ids))
        self.assertEqual(conn.execute('SELECT COUNT(*) FROM planting_points').fetchone()[0], count)
        self.assertTrue(all(row['status']=='planned' for row in conn.execute('SELECT * FROM planting_points')))

    def test_field_completion_still_rejects_a_point_covered_by_active_erosion(self):
        conn = MagicMock()
        conn.execute.return_value.fetchone.return_value = {
            'assignment_id': 1, 'point_num': 1, 'image_name': 'isolated-test.jpg',
            'current_assignment_point_status': 'pending', 'latitude': self.lat,
            'longitude': self.lon, 'deleted_at': None,
        }
        with patch.object(database, '_get_connection', return_value=conn), \
             patch.object(database, '_is_point_inside_eroded_zone', return_value=True), \
             patch.object(database, 'lock_assignment_activity'):
            with self.assertRaisesRegex(ValueError, 'eroded zone.*not available for planting'):
                database.update_assignment_point_status(1, 'completed')
        self.assertTrue(all(not call.args[0].strip().upper().startswith(('UPDATE', 'INSERT'))
                            for call in conn.execute.call_args_list))


if __name__ == '__main__':
    unittest.main()
