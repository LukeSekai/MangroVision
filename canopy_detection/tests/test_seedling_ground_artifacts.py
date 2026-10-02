"""Regression cases for stick-shadow false positives (no model required)."""
import unittest

import cv2
import numpy as np

from canopy_detection.seedling_leaf_evidence import filter_seedling_ground_artifacts


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


if __name__ == "__main__":
    unittest.main()
