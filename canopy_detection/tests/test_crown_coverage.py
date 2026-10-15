"""Coverage recovery must preserve crowns and keep ground/wood separate."""
import unittest
import hashlib
import json
import os
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import torch
from detectron2.structures import Instances

from canopy_detection.crown_coverage import recover_canopy_crown_coverage
from canopy_detection.detectree2_proper import ProperDetectree2Detector


class CrownCoverageTests(unittest.TestCase):
    def scene(self, scale=1):
        image = np.full((240, 240, 3), (95, 110, 108), np.uint8)
        mask = np.zeros(image.shape[:2], np.uint8)
        cv2.circle(mask, (120, 120), 70, 255, -1)
        image[mask > 0] = (35, 140, 65)
        return image, mask

    def test_bright_crown_interior_recovered_without_bright_ground(self):
        for scale in (1, 2):
            image, mask = self.scene()
            cv2.circle(mask, (120, 120), 20, 0, -1)
            cv2.circle(image, (120, 120), 20, (205, 220, 215), -1)
            cv2.rectangle(image, (195, 110), (225, 125), (205, 220, 215), -1)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            mask = cv2.resize(mask, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            before = mask.copy()
            recovered, metadata = recover_canopy_crown_coverage(image, mask, .01/scale)
            self.assertEqual(recovered[120*scale, 120*scale], 255)
            self.assertEqual(recovered[115*scale, 210*scale], 0)
            self.assertTrue(np.all(recovered[before > 0] == 255))
            np.testing.assert_array_equal(mask, before)
            self.assertGreater(metadata['canopy_crown_recovered_area_m2'], .1)

    def test_shaded_attached_foliage_recovered_but_shadow_gap_remains(self):
        image, mask = self.scene()
        cv2.circle(mask, (125, 120), 18, 0, -1)
        cv2.circle(image, (125, 120), 18, (15, 45, 25), -1)
        cv2.circle(mask, (95, 120), 8, 0, -1)
        cv2.circle(image, (95, 120), 8, (15, 15, 15), -1)
        recovered, _ = recover_canopy_crown_coverage(image, mask, .01)
        self.assertEqual(recovered[120, 125], 255)
        self.assertEqual(recovered[120, 95], 0)

    def test_unattached_leaf_and_tiny_ai_component_do_not_grow(self):
        image, mask = self.scene()
        cv2.circle(image, (220, 120), 5, (30, 160, 65), -1)
        recovered, _ = recover_canopy_crown_coverage(image, mask, .01)
        self.assertEqual(recovered[120, 220], 0)
        tiny = np.zeros_like(mask)
        cv2.circle(tiny, (120, 120), 8, 255, -1)
        np.testing.assert_array_equal(recover_canopy_crown_coverage(image, tiny, .01)[0], tiny)

    def test_excessive_green_ground_growth_is_rejected(self):
        image, mask = self.scene()
        image[:] = (30, 140, 65)
        recovered, metadata = recover_canopy_crown_coverage(image, mask, .01)
        np.testing.assert_array_equal(recovered, mask)
        self.assertEqual(metadata['canopy_crown_recovery_limited_groups'], 1)

    def test_missing_scale_or_empty_mask_is_unchanged(self):
        image, mask = self.scene()
        for scale in (None, 0, float('nan')):
            np.testing.assert_array_equal(recover_canopy_crown_coverage(image, mask, scale)[0], mask)
        empty = np.zeros_like(mask)
        np.testing.assert_array_equal(recover_canopy_crown_coverage(image, empty, .01)[0], empty)

    def test_diagonally_connected_crowns_keep_polygon_coverage(self):
        mask = np.zeros((100, 100), np.uint8)
        cv2.rectangle(mask, (10, 10), (40, 40), 255, -1)
        cv2.rectangle(mask, (41, 41), (80, 80), 255, -1)
        polygons = ProperDetectree2Detector._mask_to_polygons(mask, 10)
        self.assertGreater(sum(polygon.area for polygon in polygons), 2000)
        self.assertTrue(all(polygon.is_valid for polygon in polygons))

    def test_all_seedling_modes_receive_finished_canopy_without_changing_it(self):
        image, raw_mask = self.scene()
        # A missing pale interior is recovered before seedling inference.
        cv2.circle(raw_mask, (120, 120), 20, 0, -1)
        cv2.circle(image, (120, 120), 20, (205, 220, 215), -1)
        seen = []
        expected = None
        def supplement(image, base_mask, gsd):
            seen.append(base_mask.copy())
            marker = np.zeros_like(base_mask)
            cv2.circle(marker, (220, 120), 3, 255, -1)
            return marker, {'seedling_supplement_count': 1}
        for mode in ('off', 'legacy', 'hybrid', 'hybrid_fallback'):
            detector = ProperDetectree2Detector()
            detector.canopy_class_ids = {1}
            detector.set_runtime_tuning(seedling_detection_mode='hybrid' if mode == 'hybrid_fallback' else mode)
            instances = Instances(image.shape[:2])
            instances.scores = torch.tensor([.99])
            instances.pred_classes = torch.tensor([1])
            instances.pred_masks = torch.from_numpy(raw_mask[None] > 0)
            detector.predictor = lambda tile: {'instances': instances}
            detector._detect_seedling_supplement_mask = supplement
            detector._detect_hybrid_seedling_supplement_mask = supplement
            if mode == 'hybrid_fallback':
                detector._detect_hybrid_seedling_supplement_mask = lambda **kwargs: (
                    np.zeros_like(raw_mask), {'seedling_detection_mode': 'fallback_legacy'},
                )
            # Preserve the intended C-shaped input; model contours otherwise
            # intentionally fill enclosed holes before coverage processing.
            with patch.object(detector, '_polygons_to_mask', return_value=raw_mask.copy()):
                polygons, mask, metadata, seedlings = detector.detect_from_image(image, gsd=.01)
            if expected is None:
                expected = mask.copy()
            np.testing.assert_array_equal(mask, expected)
            self.assertEqual(mask[120, 120], 255)
            self.assertEqual(mask[120, 220], 0)
            self.assertGreater(len(polygons), 0)
            if mode != 'off':
                self.assertEqual(seedlings[120, 220], 255)
                np.testing.assert_array_equal(seen[-1], mask)
            else:
                self.assertFalse(np.any(seedlings))


@unittest.skipUnless(os.environ.get('MANGROVISION_TEST_IMAGE_DIR'), 'Original regression photos are supplied separately')
class CrownPhotoRegressionTests(unittest.TestCase):
    def test_bright_and_shaded_reference_crowns_recover_without_nearby_wood(self):
        root = Path(os.environ['MANGROVISION_TEST_IMAGE_DIR'])
        cases = json.loads(Path(__file__).with_name('canopy_crown_reference_examples.json').read_text())
        for case in cases:
            with self.subTest(photo=case['filename']):
                path = root / case['filename']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), case['sha256'])
                image = cv2.imread(str(path))
                mask = np.zeros(case['shape'], np.uint8)
                for hole in (False, True):
                    for contour in case['baseline_contours']:
                        if contour['hole'] == hole:
                            cv2.fillPoly(mask, [np.array(contour['points'], np.int32)], 0 if hole else 255)
                recovered, _ = recover_canopy_crown_coverage(image, mask, case['gsd'])
                self.assertTrue(np.all(recovered[mask > 0] == 255))
                x0, y0, x1, y1 = case['recover_region']
                added = (recovered[y0:y1, x0:x1] > 0) & (mask[y0:y1, x0:x1] == 0)
                self.assertGreater(np.count_nonzero(added) * case['gsd']**2, .05)
                x0, y0, x1, y1 = case['ground_region']
                np.testing.assert_array_equal(recovered[y0:y1, x0:x1], mask[y0:y1, x0:x1])


if __name__ == '__main__':
    unittest.main()
