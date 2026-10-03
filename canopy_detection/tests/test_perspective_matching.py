"""Perspective must be supported by landmarks, with safe failure on weak scenes."""
import unittest

import cv2
import numpy as np

from canopy_detection.perspective_matching import match_perspective_patch
from canopy_detection.registration import select_registration


class PerspectiveMatchingTests(unittest.TestCase):
    def test_measured_perspective_is_preserved(self):
        rng = np.random.default_rng(7)
        source = rng.uniform([0, 0], [1000, 600], (300, 2))
        expected = np.float64([[.8, .02, 40], [-.03, .8, 80], [.00015, -.00008, 1]])
        target = cv2.perspectiveTransform(source[:, None, :], expected).reshape(-1, 2)
        target += rng.normal(0, .15, target.shape)
        actual, diagnostics = select_registration(source, target, 1000, 600, .8)
        self.assertTrue(diagnostics['registration_validated'])
        self.assertEqual(diagnostics['registration_model'], 'projective')
        corners = np.float64([[[0, 0]], [[1000, 0]], [[1000, 600]], [[0, 600]]])
        np.testing.assert_allclose(cv2.perspectiveTransform(corners, actual),
                                   cv2.perspectiveTransform(corners, expected), atol=1.)

    def test_local_cluster_cannot_define_distant_corners(self):
        rng = np.random.default_rng(8)
        source = rng.uniform([100, 100], [200, 180], (200, 2))
        matrix, diagnostics = select_registration(source, source + [40, 80], 1000, 600, 1.)
        self.assertIsNone(matrix)
        self.assertFalse(diagnostics['registration_validated'])

    def test_blank_scene_has_no_invented_trapezoid(self):
        image = np.full((240, 320, 3), 100, dtype=np.uint8)
        matrix, diagnostics = match_perspective_patch(image, image, np.zeros(2), np.eye(3), 1., .03)
        self.assertIsNone(matrix)
        self.assertFalse(diagnostics['registration_validated'])

    def test_image_retry_recovers_known_warp(self):
        rng = np.random.default_rng(9)
        image = rng.integers(0, 256, (420, 600, 3), dtype=np.uint8)
        image = cv2.GaussianBlur(image, (5, 5), 1.)
        expected = np.float64([[.95, .02, 90], [-.02, .95, 70], [.00018, -.00012, 1.]])
        patch = cv2.warpPerspective(image, expected, (800, 650))
        nominal = np.float64([[.95, 0, 90], [0, .95, 70], [0, 0, 1]])
        actual, diagnostics = match_perspective_patch(image, patch, np.zeros(2), nominal, .95, .03)
        self.assertTrue(diagnostics['registration_validated'])
        probes = np.float64([[[0, 0]], [[600, 0]], [[600, 420]], [[0, 420]]])
        np.testing.assert_allclose(cv2.perspectiveTransform(probes, actual),
                                   cv2.perspectiveTransform(probes, expected), atol=3.)


if __name__ == '__main__':
    unittest.main()
