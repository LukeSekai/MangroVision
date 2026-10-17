"""Automatic fallback must improve a real transform without trusting water."""
import unittest

import cv2
import numpy as np

from canopy_detection.photometric_registration import refine_photometric_alignment


def scene(seed=9):
    rng = np.random.default_rng(seed)
    image = np.full((620, 940, 3), (105, 95, 75), np.uint8)
    for row in range(3):
        for col in range(7):
            x, y = 95 + col * 113, 70 + row * 105
            radius = int(rng.integers(24, 43))
            cv2.circle(image, (x, y), radius, (35, 145, 75), -1)
    noise = rng.normal(0, 16, image.shape[:2]).astype(np.int16)
    return np.clip(image.astype(np.int16) + noise[:, :, None], 0, 255).astype(np.uint8)


class PhotometricRegistrationTests(unittest.TestCase):
    def test_known_affine_correction_is_recovered_with_independent_scale_agreement(self):
        reference = scene()
        expected = np.float64([[1.01, .012, 20.], [-.009, .99, 15.], [0., 0., 1.]])
        image = cv2.warpPerspective(reference, np.linalg.inv(expected), (800, 460))
        nominal = np.float64([[1., 0., 25.], [0., 1., 22.], [0., 0., 1.]])
        matrix, diagnostics = refine_photometric_alignment(image, reference, nominal, (.04, .04))
        self.assertIsNotNone(matrix, diagnostics)
        self.assertTrue(diagnostics['photometric_refined'])
        probes = np.float64([[[0, 0]], [[800, 0]], [[800, 460]], [[0, 460]]])
        np.testing.assert_allclose(cv2.perspectiveTransform(probes, matrix),
                                   cv2.perspectiveTransform(probes, expected), atol=2.)

    def test_blank_water_cannot_create_an_alignment(self):
        image = np.full((300, 500, 3), (110, 100, 80), np.uint8)
        matrix, _ = refine_photometric_alignment(image, image, np.eye(3), (.04, .04))
        self.assertIsNone(matrix)

    def test_unrelated_texture_is_rejected(self):
        first, second = scene(13), scene(27)
        # Keep the same plausible land silhouette but change the actual scene.
        second = np.roll(second, (70, 180), axis=(0, 1))
        matrix, _ = refine_photometric_alignment(first, second, np.eye(3), (.04, .04))
        self.assertIsNone(matrix)

    def test_one_local_cluster_cannot_move_the_full_photo(self):
        image = np.full((400, 650, 3), (110, 100, 80), np.uint8)
        image[:100, :100] = scene()[:100, :100]
        reference = np.roll(image, (8, 9), axis=(0, 1))
        matrix, _ = refine_photometric_alignment(image, reference, np.eye(3), (.04, .04))
        self.assertIsNone(matrix)

    def test_invalid_projection_or_metric_scale_fails_closed(self):
        image = scene()
        for matrix, gsd in [(np.zeros((3, 3)), (.04, .04)),
                            (np.eye(3), (np.nan, .04)), (np.eye(3), (0., .04))]:
            with self.subTest(matrix=matrix, gsd=gsd):
                result, _ = refine_photometric_alignment(image, image, matrix, gsd)
                self.assertIsNone(result)


if __name__ == '__main__':
    unittest.main()
