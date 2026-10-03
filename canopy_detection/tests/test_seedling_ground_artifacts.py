"""Regression cases for stick-shadow false positives (no model required)."""
import unittest
import hashlib
import json
import os
from pathlib import Path

import cv2
import numpy as np

from canopy_detection.seedling_leaf_evidence import (
    filter_seedling_ground_artifacts,
    filter_small_canopy_ground_artifacts,
    recover_yellow_leaf_clusters,
)


class SeedlingGroundArtifactsTests(unittest.TestCase):
    def scene(self, scale=1):
        # Neutral mud, a dark stick shadow, and its weak green color fringe.
        image = np.full((160, 160, 3), (100, 105, 103), dtype=np.uint8)
        cv2.line(image, (15, 85), (125, 85), (65, 70, 68), 8)
        cv2.rectangle(image, (72, 79), (88, 81), (58, 85, 80), -1)
        cv2.line(image, (125, 85), (143, 66), (170, 185, 186), 4)
        if scale != 1:
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return image, [{"center": (80*scale, 80*scale), "cluster_rescue": True}]

    def test_shadow_next_to_stick_is_removed_at_multiple_resolutions(self):
        for scale in (1, 2):
            with self.subTest(scale=scale):
                image, entries = self.scene(scale)
                kept, metadata = filter_seedling_ground_artifacts(image, entries, .01/scale)
                self.assertEqual(kept, [])
                self.assertEqual(metadata["seedling_ground_artifact_rejected_count"], 1)

    def test_green_and_yellow_leaves_are_kept(self):
        for color in ((35, 130, 65), (45, 160, 157)):
            with self.subTest(color=color):
                image, entries = self.scene()
                cv2.ellipse(image, (80, 80), (5, 3), 30, 0, 360, color, -1)
                kept, _ = filter_seedling_ground_artifacts(image, entries, .01)
                self.assertEqual(kept, entries)

    def test_nearby_leaf_protects_shaded_seedling_fragment(self):
        image, entries = self.scene()
        cv2.ellipse(image, (98, 70), (5, 3), 30, 0, 360, (40, 160, 150), -1)
        kept, _ = filter_seedling_ground_artifacts(image, entries, .01)
        self.assertEqual(kept, entries)

    def test_distant_leaves_do_not_validate_stick_shadow(self):
        image, entries = self.scene()
        cv2.rectangle(image, (5, 5), (30, 30), (30, 180, 60), -1)
        kept, _ = filter_seedling_ground_artifacts(image, entries, .01)
        self.assertEqual(kept, [])

    def test_empty_candidates_and_image_edge(self):
        image, _ = self.scene()
        self.assertEqual(filter_seedling_ground_artifacts(image, [], .01)[0], [])
        cv2.circle(image, (1, 1), 3, (30, 170, 70), -1)
        entries = [{"center": (1., 1.)}]
        self.assertEqual(filter_seedling_ground_artifacts(image, entries, .01)[0], entries)

    def test_scattered_yellow_stick_highlights_do_not_count_as_leaf(self):
        for scale in (1, 2):
            with self.subTest(scale=scale):
                image, entries = self.scene()
                for x, y in ((88, 72), (91, 75), (94, 78), (96, 72)):
                    image[y, x] = (80, 145, 143)
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                entries[0]['center'] = (80*scale, 80*scale)
                self.assertEqual(filter_seedling_ground_artifacts(image, entries, .01/scale)[0], [])

    def test_long_yellow_highlight_is_not_a_compact_leaf(self):
        for angle in (0, 45, 90):
            with self.subTest(angle=angle):
                image, entries = self.scene()
                cv2.ellipse(image, (80, 69), (10, 1), angle, 0, 360, (80, 145, 143), -1)
                self.assertEqual(filter_seedling_ground_artifacts(image, entries, .01)[0], [])

    def test_dark_green_fringe_does_not_protect_mud_patch(self):
        image = np.full((160, 160, 3), (103, 110, 107), dtype=np.uint8)
        cv2.circle(image, (80, 80), 10, (75, 100, 92), -1)
        cv2.circle(image, (80, 80), 3, (70, 100, 88), -1)
        entries = [{'center': (80., 80.)}]
        self.assertEqual(filter_seedling_ground_artifacts(image, entries, .008)[0], [])

    def test_compact_shaded_green_leaf_protects_candidate(self):
        image, entries = self.scene()
        # Darker than the ground, but distinctly green and compact.
        cv2.ellipse(image, (95, 70), (5, 3), 30, 0, 360, (25, 90, 40), -1)
        self.assertEqual(filter_seedling_ground_artifacts(image, entries, .01)[0], entries)

    def test_bright_wood_tip_cannot_bypass_the_final_check(self):
        for scale in (1, 2):
            image = np.full((160, 160, 3), (100, 105, 103), dtype=np.uint8)
            cv2.line(image, (25, 80), (82, 80), (170, 190, 191), 4)
            cv2.ellipse(image, (80, 80), (3, 2), 0, 0, 360, (70, 135, 134), -1)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            kept, metadata = filter_seedling_ground_artifacts(image, [{'center': (80*scale, 80*scale)}], .01/scale)
            self.assertEqual(kept, [])
            self.assertEqual(metadata['seedling_ground_artifact_rejected_count'], 1)

    def test_pale_leaf_with_green_pixels_is_preserved(self):
        for scale in (1, 2):
            image = np.full((160, 160, 3), (100, 105, 103), dtype=np.uint8)
            cv2.ellipse(image, (80, 80), (6, 3), 0, 0, 360, (175, 195, 180), -1)
            cv2.circle(image, (78, 80), 2, (95, 145, 130), -1)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            entries = [{'center': (80*scale, 80*scale)}]
            self.assertEqual(filter_seedling_ground_artifacts(image, entries, .01/scale)[0], entries)

    def test_candidate_without_leaf_pixels_is_rejected(self):
        image = np.full((160, 160, 3), (100, 105, 103), dtype=np.uint8)
        self.assertEqual(filter_seedling_ground_artifacts(image, [{'center': (80., 80.)}], .01)[0], [])


@unittest.skipUnless(os.environ.get('MANGROVISION_TEST_IMAGE_DIR'), 'Original regression photos are supplied separately')
class SeedlingPhotoRegressionTests(unittest.TestCase):
    def test_smaller_leaf_recovery_does_not_admit_reviewed_wood_tips(self):
        root = Path(os.environ['MANGROVISION_TEST_IMAGE_DIR'])
        cases = json.loads(Path(__file__).with_name('seedling_ground_artifact_examples.json').read_text())
        for case in cases:
            if not case.get('recovery_negative_centers'):
                continue
            path = root / case['filename']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), case['sha256'])
            image = cv2.imread(str(path))
            entries, _ = recover_yellow_leaf_clusters(image, np.zeros(image.shape[:2], np.uint8), [], case['gsd'])
            entries, _ = filter_seedling_ground_artifacts(image, entries, case['gsd'])
            for entry in entries:
                for center in case['recovery_negative_centers']:
                    self.assertGreater(np.linalg.norm(np.array(entry['center']) - center) * case['gsd'], .20)

    def test_reviewed_sticks_removed_and_visible_leaves_retained(self):
        root = Path(os.environ['MANGROVISION_TEST_IMAGE_DIR'])
        cases = json.loads(Path(__file__).with_name('seedling_ground_artifact_examples.json').read_text())
        for case in cases:
            with self.subTest(image=case['filename']):
                path = root / case['filename']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), case['sha256'],
                                 'Use the original photo, not a resized/exported copy')
                image = cv2.imread(str(path))
                self.assertIsNotNone(image)
                positives = [{'center': point} for point in case['positive_centers']]
                negatives = [{'center': point} for point in case['negative_centers']]
                kept, metadata = filter_seedling_ground_artifacts(image, positives + negatives, case['gsd'])
                self.assertEqual(kept, positives)
                self.assertEqual(metadata['seedling_ground_artifact_rejected_count'], len(negatives))

    def test_photo_algae_removed_and_missed_leaves_recovered(self):
        root = Path(os.environ['MANGROVISION_TEST_IMAGE_DIR'])
        cases = json.loads(Path(__file__).with_name('seedling_ground_artifact_examples.json').read_text())
        for case in cases:
            if not case.get('canopy_negative_polygon'):
                continue
            path = root / case['filename']
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), case['sha256'])
            image = cv2.imread(str(path))
            mask = np.zeros(image.shape[:2], np.uint8)
            cv2.fillPoly(mask, [np.array(case['canopy_negative_polygon'], np.int32)], 255)
            filtered, info = filter_small_canopy_ground_artifacts(image, mask, case['gsd'])
            self.assertFalse(np.any(filtered))
            self.assertEqual(info['canopy_ground_artifact_rejected_count'], 1)
            entries, _ = recover_yellow_leaf_clusters(image, filtered, [], case['gsd'])
            entries, _ = filter_seedling_ground_artifacts(image, entries, case['gsd'])
            points = np.array([entry['center'] for entry in entries])
            for center in case['recovery_centers']:
                self.assertLess(np.linalg.norm(points - center, axis=1).min() * case['gsd'], .15)
            for center in case['negative_centers']:
                self.assertGreater(np.linalg.norm(points - center, axis=1).min() * case['gsd'], .20)


if __name__ == "__main__":
    unittest.main()
