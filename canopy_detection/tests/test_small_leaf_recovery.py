"""Guard the distinction between small leaves, woody highlights, and algae."""
import unittest
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from canopy_detection.seedling_leaf_evidence import (
    filter_small_canopy_ground_artifacts,
    recover_yellow_leaf_clusters,
)


class SmallCanopyEvidenceTests(unittest.TestCase):
    def woody_scene(self, angle=0, scale=1, leafy=False, short=False):
        image = np.full((240, 240, 3), (100, 105, 103), dtype=np.uint8)
        mask = np.zeros((240, 240), np.uint8)
        half_length = 20 if short else 55
        cv2.rectangle(mask, (120-half_length, 114), (120+half_length, 126), 255, -1)
        image[mask > 0] = (135, 165, 164)
        # Flecks on wood pass the old compact-core check. Multiple substantial
        # leaves along a real branch must still protect its narrow prediction.
        for x in ([90, 120, 150] if leafy else [120]):
            cv2.circle(image, (x, 120), 5 if leafy else 2, (35, 160, 65), -1)
        matrix = cv2.getRotationMatrix2D((120, 120), angle, 1)
        image = cv2.warpAffine(image, matrix, (240, 240), flags=cv2.INTER_NEAREST,
                               borderValue=(100, 105, 103))
        mask = cv2.warpAffine(mask, matrix, (240, 240), flags=cv2.INTER_NEAREST)
        return (cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST),
                cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))

    def test_long_wood_with_colored_flecks_rejected_across_angles_and_scales(self):
        for angle in (0, 45, 90):
            for scale in (1, 2):
                with self.subTest(angle=angle, scale=scale):
                    image, mask = self.woody_scene(angle, scale)
                    filtered, info = filter_small_canopy_ground_artifacts(image, mask, .01/scale)
                    self.assertFalse(np.any(filtered))
                    self.assertEqual(info['canopy_ground_artifact_rejected_reasons'],
                                     {'elongated_wood_prediction': 1})

    def test_leaf_covered_narrow_branch_is_preserved(self):
        for angle in (0, 45, 90):
            for scale in (1, 2):
                image, mask = self.woody_scene(angle, scale, leafy=True)
                filtered, _ = filter_small_canopy_ground_artifacts(image, mask, .01/scale)
                np.testing.assert_array_equal(filtered, mask)

    def test_short_thin_leaf_is_not_rejected_as_long_wood(self):
        image, mask = self.woody_scene(short=True)
        # A short leaf has substantial foliage, rather than one colored spot
        # on an otherwise non-leafy object.
        cv2.ellipse(image, (120, 120), (12, 4), 0, 0, 360, (35, 160, 65), -1)
        filtered, _ = filter_small_canopy_ground_artifacts(image, mask, .01)
        np.testing.assert_array_equal(filtered, mask)

    def test_tiny_leaf_colored_core_does_not_validate_whole_ground_patch(self):
        for scale in (1, 2):
            image, mask = self.scene((55, 90, 88))
            cv2.circle(image, (80, 80), 2, (35, 160, 65), -1)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            filtered, info = filter_small_canopy_ground_artifacts(image, mask, .01 / scale)
            self.assertFalse(np.any(filtered))
            self.assertEqual(info['canopy_ground_artifact_rejected_reasons'], {'sparse_leaf_support': 1})

    def test_visible_edge_ground_rejected_but_edge_foliage_preserved(self):
        for scale in (1, 2):
            for color, expected_reject in [((55, 90, 88), True), ((35, 130, 65), False)]:
                image, mask = self.scene(color)
                image, mask = image[80:, :], mask[80:, :]
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                filtered, info = filter_small_canopy_ground_artifacts(image, mask, .01 / scale)
                if expected_reject:
                    self.assertFalse(np.any(filtered))
                    self.assertTrue(info['canopy_ground_artifact_rejected_evidence'][0]['touches_edge'])
                else:
                    np.testing.assert_array_equal(filtered, mask)

    def test_split_wood_with_moderate_aspect_and_sparse_flecks_rejected(self):
        for scale in (1, 2):
            image = np.full((160, 160, 3), (100, 105, 103), dtype=np.uint8)
            mask = np.zeros((160, 160), np.uint8)
            cv2.rectangle(mask, (40, 71), (120, 89), 255, -1)
            image[mask > 0] = (135, 165, 164)
            cv2.circle(image, (80, 80), 5, (35, 160, 65), -1)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            filtered, info = filter_small_canopy_ground_artifacts(image, mask, .01 / scale)
            self.assertFalse(np.any(filtered))
            self.assertEqual(info['canopy_ground_artifact_rejected_reasons'], {'elongated_wood_prediction': 1})

    def test_dense_leafy_crown_clipped_into_a_strip_at_edge_is_preserved(self):
        for scale in (1, 2):
            image = np.full((160, 160, 3), (100, 105, 103), dtype=np.uint8)
            mask = np.zeros((160, 160), np.uint8)
            cv2.rectangle(mask, (0, 70), (100, 82), 255, -1)
            image[mask > 0] = (35, 130, 65)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            filtered, info = filter_small_canopy_ground_artifacts(image, mask, .01 / scale)
            np.testing.assert_array_equal(filtered, mask)
            self.assertEqual(info['canopy_ground_artifact_rejected_count'], 0)

    def scene(self, color, scale=1):
        image = np.full((160, 160, 3), (100, 105, 103), dtype=np.uint8)
        mask = np.zeros((160, 160), dtype=np.uint8)
        cv2.ellipse(mask, (80, 80), (16, 23), 0, 0, 360, 255, -1)
        image[mask > 0] = color
        return (cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST),
                cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST))

    def test_olive_ground_removed_at_multiple_resolutions(self):
        for scale in (1, 2):
            image, mask = self.scene((55, 90, 88), scale)
            before = mask.copy()
            filtered, info = filter_small_canopy_ground_artifacts(image, mask, .01 / scale)
            self.assertEqual(np.count_nonzero(filtered), 0)
            self.assertEqual(info['canopy_ground_artifact_rejected_count'], 1)
            np.testing.assert_array_equal(mask, before)

    def test_green_shaded_and_yellow_crowns_retained(self):
        for color in [(35, 130, 65), (25, 90, 40), (45, 160, 157)]:
            image, mask = self.scene(color)
            filtered, _ = filter_small_canopy_ground_artifacts(image, mask, .01)
            np.testing.assert_array_equal(filtered, mask)

    def test_nearby_leaf_outside_prediction_cannot_validate_algae(self):
        image, mask = self.scene((55, 90, 88))
        cv2.circle(image, (106, 80), 4, (30, 160, 65), -1)
        filtered, _ = filter_small_canopy_ground_artifacts(image, mask, .01)
        self.assertEqual(np.count_nonzero(filtered), 0)

    def test_large_crowns_and_missing_scale_keep_existing_checks(self):
        image, mask = self.scene((55, 90, 88))
        for gsd in (.04, None):
            filtered, _ = filter_small_canopy_ground_artifacts(image, mask, gsd)
            np.testing.assert_array_equal(filtered, mask)

    def test_partial_crown_at_image_edge_is_not_judged_as_ground(self):
        image, mask = self.scene((35, 130, 65))
        image, mask = image[65:100, 70:110], mask[65:100, 70:110]
        filtered, info = filter_small_canopy_ground_artifacts(image, mask, .01)
        np.testing.assert_array_equal(filtered, mask)
        self.assertEqual(info['canopy_ground_artifact_rejected_count'], 0)

    def test_no_visible_surroundings_keeps_existing_canopy_checks(self):
        image, mask = self.scene((55, 90, 88))
        image, mask = image[72:90, 74:86], mask[72:90, 74:86]
        filtered, _ = filter_small_canopy_ground_artifacts(image, mask, .01)
        np.testing.assert_array_equal(filtered, mask)


@unittest.skipUnless(os.environ.get('MANGROVISION_TEST_IMAGE_DIR'),
                     'Original regression photos are supplied separately')
class CanopyWoodPhotoTests(unittest.TestCase):
    def test_reported_wood_removed_and_real_canopy_preserved(self):
        cases = json.loads(Path(__file__).with_name('canopy_wood_examples.json').read_text())
        for case in cases:
            path = Path(os.environ['MANGROVISION_TEST_IMAGE_DIR']) / case['filename']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), case['sha256'])
            image = cv2.imread(str(path))
            for prediction in case['predictions']:
                with self.subTest(photo=case['filename'], example=prediction['note']):
                    mask = np.zeros(image.shape[:2], np.uint8)
                    cv2.fillPoly(mask, [np.array(prediction['polygon'], np.int32)], 255)
                    filtered, info = filter_small_canopy_ground_artifacts(image, mask, case['gsd'])
                    if prediction['expected'] == 'reject':
                        self.assertFalse(np.any(filtered))
                        self.assertEqual(info['canopy_ground_artifact_rejected_reasons'],
                                         {'elongated_wood_prediction': 1})
                    else:
                        np.testing.assert_array_equal(filtered, mask)


@unittest.skipUnless(os.environ.get('MANGROVISION_TEST_IMAGE_DIR'),
                     'Original regression photos are supplied separately')
class CanopyGroundPhotoTests(unittest.TestCase):
    def test_reviewed_ground_and_wood_removed_without_losing_leafy_plants(self):
        cases = json.loads(Path(__file__).with_name('canopy_ground_review_examples.json').read_text())
        for case in cases:
            with self.subTest(photo=case['filename'], example=case['note']):
                path = Path(os.environ['MANGROVISION_TEST_IMAGE_DIR']) / case['filename']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), case['sha256'],
                                 'Use the original photo, not a resized/exported copy')
                image = cv2.imread(str(path))
                self.assertIsNotNone(image)
                mask = np.zeros(image.shape[:2], np.uint8)
                cv2.fillPoly(mask, [np.array(polygon, np.int32) for polygon in case['polygons']], 255)
                filtered, info = filter_small_canopy_ground_artifacts(image, mask, case['gsd'])
                if case['expected'] == 'reject':
                    self.assertFalse(np.any(filtered))
                    self.assertEqual(info['canopy_ground_artifact_rejected_count'], 1)
                else:
                    np.testing.assert_array_equal(filtered, mask)
                    self.assertEqual(info['canopy_ground_artifact_rejected_count'], 0)


class SmallYellowLeafRecoveryTests(unittest.TestCase):
    def test_small_leaf_recovered_at_multiple_resolutions(self):
        for scale in (1, 2):
            image = np.full((120, 120, 3), (100, 105, 103), dtype=np.uint8)
            cv2.ellipse(image, (60, 60), (4, 2), 20, 0, 360, (40, 160, 159), -1)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            excluded = np.zeros(image.shape[:2], dtype=np.uint8)
            entries, _ = recover_yellow_leaf_clusters(image, excluded, [], .01 / scale)
            self.assertEqual(len(entries), 1)
            self.assertLess(np.linalg.norm(np.array(entries[0]['center']) / scale - (60, 60)), 2)
            # An existing canopy or a previously detected leaf excludes it.
            self.assertEqual(recover_yellow_leaf_clusters(image, np.full_like(excluded, 255), [], .01 / scale)[0], [])
            self.assertEqual(recover_yellow_leaf_clusters(image, excluded, [entries[0]['center']], .01 / scale)[0], [])

    def test_elongated_yellow_stick_is_not_recovered(self):
        image = np.full((120, 120, 3), (100, 105, 103), dtype=np.uint8)
        cv2.line(image, (40, 40), (80, 80), (60, 165, 164), 2)
        self.assertEqual(recover_yellow_leaf_clusters(image, np.zeros((120, 120), np.uint8), [], .01)[0], [])

    def test_dull_olive_fragment_is_not_a_leaf(self):
        image = np.full((120, 120, 3), (100, 105, 103), dtype=np.uint8)
        cv2.ellipse(image, (60, 60), (4, 2), 20, 0, 360, (65, 100, 101), -1)
        self.assertEqual(recover_yellow_leaf_clusters(image, np.zeros((120, 120), np.uint8), [], .01)[0], [])


if __name__ == '__main__':
    unittest.main()
