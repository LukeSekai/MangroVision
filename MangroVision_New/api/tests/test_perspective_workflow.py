"""Validated perspective must survive projection and footprint rendering."""
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api.routes import processing as p


class PerspectiveWorkflowTests(unittest.TestCase):
    def test_projective_matrix_is_shared_by_coordinates_overlay_and_boundary(self):
        image = np.full((60, 100, 3), 120, np.uint8)
        matrix = np.float64([[1., .03, 10], [.02, 1., 20], [.001, -.0005, 1.]])
        result = {'success': True, 'H': np.eye(3), 'validated_H': matrix,
                  'registration_validated': True, 'registration_model': 'projective',
                  'registration_median_error_px': 1., 'registration_stability_px': 2.}
        with patch.object(p, '_homography_center_diagnostics', return_value={'center_drift_m': .2}), \
             patch.object(p, '_homography_scale_diagnostics', return_value={}), \
             patch.object(p.ortho_matcher, 'ORTHO_GSD', .03):
            result = p._post_process_match(result, image, 10., 122., 0., .03)
        self.assertIs(p._coordinate_homography_for_match(result), matrix)
        self.assertIs(p._overlay_homography_for_match(result), matrix)
        self.assertFalse(result['projection_rebuilt'])
        with patch.object(p.ortho_matcher, '_ensure_active_ortho', return_value={}), \
             patch.object(p, 'overlay_visibility_mask', side_effect=lambda *a: np.full((a[-1], a[-2]), 255, np.uint8)), \
             patch.object(p, 'ortho_pixel_to_gps', side_effect=lambda x, y: (y*.001, x*.001)):
            overlay = p._build_georeferenced_overlay(image, matrix)
        expected = cv2.perspectiveTransform(np.float64([[[0, 0]], [[100, 0]], [[100, 60]], [[0, 60]]]), matrix).reshape(-1, 2) * .001
        ring = overlay['footprint_geojson']['coordinates'][0]
        np.testing.assert_allclose(ring[:4], expected)
        self.assertEqual(ring[0], ring[-1])


if __name__ == '__main__':
    unittest.main()
