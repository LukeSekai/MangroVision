"""Initial preview and processed overlay must share measured image corners."""
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from shapely.geometry import box, shape

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api.routes import processing as p


class PreflightAlignmentTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((60, 100, 3), 120, np.uint8)
        self.matrix = np.float64([[1., .03, 10], [.02, 1., 20], [.001, -.0005, 1.]])
        self.match = {'success': True, 'H': self.matrix, 'heading': 2.,
                      'projection_rotation_source': 'sift_validated_projective',
                      'registration_validated': True}

    def test_preflight_uses_perspective_without_requiring_refined_gsd(self):
        for coverage in [box(-1, -1, 1, 1), box(-1, -1, .06, 1)]:
            with self.subTest(partial=coverage.bounds[2] < 1), \
                 patch.object(p.cv2, 'imread', return_value=self.image), \
                 patch.object(p, '_get_image_alignment', return_value=self.match), \
                 patch.object(p, '_load_gis_coverage_geometry', return_value=coverage), \
                 patch.object(p, '_project_site_context', return_value={'location_label': 'Test', 'project_sites': []}), \
                 patch.object(p, 'ortho_pixel_to_gps', side_effect=lambda x, y: (y*.001, x*.001)), \
                 patch.object(p.ortho_matcher, '_ensure_active_ortho', return_value={}), \
                 patch.object(p, 'overlay_visibility_mask', side_effect=lambda *a: np.full((a[-1], a[-2]), 255, np.uint8)):
                initial = {'status': 'inside', 'latitude': .04, 'longitude': .03,
                           'gsd_m_per_pixel': .01, 'heading_deg': 0., 'coverage_m': [1., .6]}
                preview = p._calibrate_preflight_footprint(Path('photo.jpg'), initial)
                overlay = p._build_georeferenced_overlay(self.image, self.matrix)
                expected = coverage.intersection(shape(overlay['footprint_geojson']))
                self.assertTrue(shape(preview['map']['analysis_footprint']).equals(expected))
                self.assertEqual(preview['map']['analysis_footprint_quality'], 'matched_image_corners')
                self.assertEqual(preview['requires_confirmation'], coverage.bounds[2] < 1)

    def test_cache_preserves_transform_and_invalidates_when_inputs_change(self):
        with tempfile.TemporaryDirectory() as directory:
            image_path = Path(directory) / 'photo.jpg'
            image_path.write_bytes(b'original image')
            map_path = Path(directory) / 'ortho.tif'
            map_path.write_bytes(b'map')
            entry = {'path': map_path, 'name': 'test'}
            with patch.dict(p._PREFLIGHT_ALIGNMENT_CACHE, clear=True), \
                 patch.object(p.ortho_matcher, '_get_ortho_registry', return_value=[entry]), \
                 patch.object(p.ortho_matcher, '_active_tileset_name', return_value='test'), \
                 patch.object(p.ortho_matcher, 'ORTHO_PATH', map_path), \
                 patch.object(p.ortho_matcher, '_activate_ortho') as activate, \
                 patch.object(p, '_match_drone_to_ortho_robust', side_effect=lambda **_: dict(self.match, H=self.matrix.copy())) as match:
                def resolve(gsd=.01):
                    return p._get_image_alignment(image_path, self.image, 10., 122., gsd, 0.)
                first = resolve()
                first['H'][0, 0] = 99
                second = resolve()
                np.testing.assert_array_equal(second['H'], self.matrix)
                self.assertEqual(match.call_count, 1)
                activate.assert_called_with(entry)
                resolve(.02)
                self.assertEqual(match.call_count, 2)
                image_path.write_bytes(b'different image')
                resolve()
                self.assertEqual(match.call_count, 3)
                map_path.write_bytes(b'replaced orthophoto')
                resolve()
                self.assertEqual(match.call_count, 4)

    def test_failed_visual_match_uses_same_metric_corners(self):
        with patch.object(p.cv2, 'imread', return_value=self.image), \
             patch.object(p, '_get_image_alignment', return_value={'success': False}), \
             patch.object(p, '_build_metric_centered_homography', return_value=self.matrix), \
             patch.object(p, '_load_gis_coverage_geometry', return_value=box(-1, -1, 1, 1)), \
             patch.object(p, '_project_site_context', return_value={'location_label': 'Test', 'project_sites': []}), \
             patch.object(p, 'ortho_pixel_to_gps', side_effect=lambda x, y: (y*.001, x*.001)):
            preview = p._calibrate_preflight_footprint(Path('photo.jpg'), {
                'status': 'inside', 'latitude': .04, 'longitude': .03,
                'gsd_m_per_pixel': .01, 'heading_deg': 0.,
            })
            self.assertFalse(preview['footprint_calibrated'])
            self.assertTrue(shape(preview['map']['analysis_footprint']).equals(
                shape(p._projected_image_footprint(self.image.shape, self.matrix))))

    def test_no_gps_does_not_start_matching(self):
        with patch.object(p, '_get_image_alignment') as match:
            result = {'status': 'no_gps'}
            self.assertIs(p._calibrate_preflight_footprint(Path('photo.jpg'), result), result)
            match.assert_not_called()


if __name__ == '__main__':
    unittest.main()
