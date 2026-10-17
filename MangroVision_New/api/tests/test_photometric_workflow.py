"""The corrected automatic transform must survive every map projection."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
from shapely.geometry import box, shape

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api.routes import processing as p


class PhotometricWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.image = np.full((60, 100, 3), 120, np.uint8)
        self.matrix = np.float64([[1.01, .08, 10.], [-.04, .98, 20.], [0., 0., 1.]])
        self.match = {'success': True, 'H': self.matrix, 'heading': 358.,
                      'photometric_refined': True, 'projection_rebuilt': True,
                      'projection_rotation_source': 'photometric_affine'}

    def test_border_overlay_and_coordinates_use_the_corrected_matrix(self):
        with patch.object(p.cv2, 'imread', return_value=self.image), \
             patch.object(p, '_get_image_alignment', return_value=self.match), \
             patch.object(p, '_load_gis_coverage_geometry', return_value=box(-1, -1, 1, 1)), \
             patch.object(p, '_project_site_context', return_value={'project_sites': []}), \
             patch.object(p, 'ortho_pixel_to_gps', side_effect=lambda x, y: (y*.001, x*.001)), \
             patch.object(p.ortho_matcher, '_ensure_active_ortho', return_value={}), \
             patch.object(p, 'overlay_visibility_mask', side_effect=lambda *a: np.full((a[-1], a[-2]), 255, np.uint8)):
            preflight = p._calibrate_preflight_footprint(Path('photo.jpg'), {
                'status': 'inside', 'latitude': .04, 'longitude': .03,
                'gsd_m_per_pixel': .01, 'heading_deg': 0., 'coverage_m': [1., .6]})
            overlay = p._build_georeferenced_overlay(self.image, p._overlay_homography_for_match(self.match))
        self.assertTrue(shape(preflight['map']['analysis_footprint']).equals(shape(overlay['footprint_geojson'])))
        self.assertEqual(preflight['map']['analysis_footprint_quality'], 'metric_projected_image_corners')
        self.assertTrue(preflight['map']['match']['photometric_refined'])
        self.assertIs(p._coordinate_homography_for_match(self.match), self.matrix)
        probes = np.float64([[[0, 0]], [[100, 0]], [[100, 60]], [[0, 60]]])
        expected = cv2.perspectiveTransform(probes, self.matrix).reshape(-1, 2) * .001
        # GIS intersection may reverse a ring or choose a different first corner.
        actual = preflight['map']['analysis_footprint']['coordinates'][0][:4]
        np.testing.assert_allclose(sorted(actual), sorted(expected.tolist()))

    def test_valid_sift_registration_is_not_replaced(self):
        with patch.object(p, 'match_drone_to_ortho', return_value={'success': True}) as sift, \
             patch.object(p, '_post_process_match', return_value=self.match), \
             patch.object(p, '_refine_failed_match_photometrically') as refinement:
            result = p._match_drone_to_ortho_robust(self.image, 10., 122., .01, 0.)
        self.assertIs(result, self.match)
        sift.assert_called_once()
        refinement.assert_not_called()

    def test_failed_landmark_matches_reach_the_automatic_fallback(self):
        failure = {'success': False, 'error': 'No reliable landmarks'}
        with patch.object(p, 'match_drone_to_ortho', return_value=failure), \
             patch.object(p, '_retry_perspective_registration', return_value=None), \
             patch.object(p, '_refine_failed_match_with_vegetation_scale', return_value=failure), \
             patch.object(p, '_refine_failed_match_photometrically', return_value=self.match), \
             patch.object(p, '_refine_failed_match_with_edge_translation') as edge:
            result = p._match_drone_to_ortho_robust(self.image, 10., 122., .01, 0.)
        self.assertIs(result['H'], self.matrix)
        edge.assert_not_called()

    def test_rejected_refinement_preserves_the_existing_fallback(self):
        failure = {'success': False, 'error': 'No reliable landmarks'}
        with patch.object(p, 'match_drone_to_ortho', return_value=failure), \
             patch.object(p, '_retry_perspective_registration', return_value=None), \
             patch.object(p, '_refine_failed_match_with_vegetation_scale', return_value=failure), \
             patch.object(p, '_refine_failed_match_photometrically', return_value=failure), \
             patch.object(p, '_refine_failed_match_with_edge_translation', return_value=failure) as edge:
            result = p._match_drone_to_ortho_robust(self.image, 10., 122., .01, 0.)
        self.assertFalse(result['success'])
        edge.assert_called_once()

    def test_refinement_is_kept_approximate_and_not_recentred_on_exif(self):
        diagnostics = {'photometric_refined': True, 'photometric_score': .6}
        with patch.object(p.ortho_matcher, 'select_orthophoto', return_value={'name': 'Test map', 'path': 'map.tif'}), \
             patch.object(p.ortho_matcher, 'load_orthophoto', return_value=self.image), \
             patch.object(p, '_build_metric_centered_homography', return_value=np.eye(3)), \
             patch.object(p, '_active_ortho_metric_gsd_xy', return_value=(.03, .03)), \
             patch.object(p, 'refine_photometric_alignment', return_value=(self.matrix, diagnostics)), \
             patch.object(p, '_homography_center_diagnostics', return_value={}), \
             patch.object(p, '_homography_scale_diagnostics', return_value={}):
            result = p._refine_failed_match_photometrically({}, self.image, 10., 122., .01, 0.)
        self.assertIs(result['H'], self.matrix)
        self.assertTrue(result['projection_rebuilt'])
        self.assertFalse(result['registration_validated'])
        self.assertFalse(result['gps_anchored'])
        self.assertIn('approximate', p._match_quality_warning(result))
        self.assertNotIn('GPS-anchored', p._match_quality_warning(result))


if __name__ == '__main__':
    unittest.main()
