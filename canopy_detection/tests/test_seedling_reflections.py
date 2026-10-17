"""Glare must not become a seedling through clipped yellow-green pixels."""
import hashlib
import json
import os
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from canopy_detection.seedling_reflections import filter_seedling_reflections


class SeedlingReflectionTests(unittest.TestCase):
    def scene(self, scale=1, *, green=False, shadow=False, neighbors=True):
        image = np.full((300, 300, 3), (110, 130, 130), np.uint8)
        cv2.ellipse(image, (150, 150), (4, 3), 0, 0, 360, (200, 255, 255), -1)
        cv2.ellipse(image, (146, 150), (3, 2), 0, 0, 360, (100, 210, 198), -1)
        if neighbors:
            for x, y in ((180, 130), (120, 175), (185, 180)):
                cv2.ellipse(image, (x, y), (3, 2), 0, 0, 360, (250, 255, 255), -1)
        if green:
            cv2.ellipse(image, (142, 146), (4, 3), 0, 0, 360, (35, 160, 75), -1)
        if shadow:
            cv2.line(image, (143, 152), (133, 162), (45, 60, 55), 3)
        image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
        return image, [{'center': (146*scale, 150*scale), 'yellow_recovery': True}], .01/scale

    def test_colored_glint_fringe_is_removed_at_multiple_ground_scales(self):
        for scale in (1, 2, 3):
            with self.subTest(scale=scale):
                image, entries, gsd = self.scene(scale)
                before = image.copy()
                kept, meta = filter_seedling_reflections(image, entries, gsd)
                self.assertEqual(kept, [])
                self.assertEqual(meta['seedling_reflection_rejected_count'], 1)
                self.assertEqual(meta['seedling_reflection_rejected_evidence'][0]['reason'], 'specular_water_highlight')
                np.testing.assert_array_equal(image, before)

    def test_green_leaf_or_connected_plant_shadow_preserves_a_pale_candidate(self):
        for scale in (1, 2):
            for evidence in ('green', 'shadow'):
                with self.subTest(scale=scale, evidence=evidence):
                    image, entries, gsd = self.scene(scale, **{evidence: True})
                    self.assertEqual(filter_seedling_reflections(image, entries, gsd)[0], entries)

    def test_one_white_leaf_alone_cannot_establish_repeated_water_glare(self):
        image, entries, gsd = self.scene(neighbors=False)
        self.assertEqual(filter_seedling_reflections(image, entries, gsd)[0], entries)

    def test_yellow_leaf_body_away_from_its_clipped_tip_is_preserved(self):
        for scale in (1, 2):
            image, entries, gsd = self.scene()
            cv2.ellipse(image, (140, 150), (9, 5), 0, 0, 360, (65, 180, 172), -1)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            entries[0]['center'] = (146*scale, 150*scale)
            self.assertEqual(filter_seedling_reflections(image, entries, gsd/scale)[0], entries)

    def test_a_group_of_glints_cannot_supply_its_own_plant_evidence(self):
        image, entries, gsd = self.scene()
        for center in ((180,130), (120,175), (185,180)):
            entries.append({'center': center, 'cluster_rescue': True})
        self.assertEqual(filter_seedling_reflections(image, entries, gsd)[0], [])

    def test_bright_foliage_background_is_not_treated_as_water(self):
        image, entries, gsd = self.scene()
        image[110:190, 110:190] = (40, 150, 70)
        cv2.ellipse(image, (150, 150), (4, 3), 0, 0, 360, (200, 255, 255), -1)
        cv2.circle(image, (180, 130), 3, (250, 255, 255), -1)
        self.assertEqual(filter_seedling_reflections(image, entries, gsd)[0], entries)

    def test_disconnected_dark_speckles_do_not_establish_a_stem_or_shadow(self):
        image, entries, gsd = self.scene()
        for x, y in ((140, 145), (143, 156), (150, 143), (151, 160)):
            image[y, x] = (45, 60, 55)
        self.assertEqual(filter_seedling_reflections(image, entries, gsd)[0], [])

    def test_textured_ground_with_bright_tips_is_not_inferred_to_be_open_water(self):
        image, entries, gsd = self.scene()
        noise = np.random.default_rng(7).integers(-20, 21, image.shape[:2], dtype=np.int16)
        yy, xx = np.ogrid[:300, :300]
        ring = ((xx-146)**2+(yy-150)**2 >= 16**2)
        image[ring] = np.clip(image.astype(np.int16)+noise[:, :, None], 0, 255)[ring]
        self.assertEqual(filter_seedling_reflections(image, entries, gsd)[0], entries)

    def test_reflections_at_a_photo_edge_still_use_visible_background(self):
        for scale in (1, 2):
            image, entries, gsd = self.scene(scale)
            cropped = image[:155*scale].copy()
            self.assertEqual(filter_seedling_reflections(cropped, entries, gsd)[0], [])

    def test_empty_or_invalid_scale_preserves_the_existing_noop(self):
        image, entries, gsd = self.scene()
        self.assertEqual(filter_seedling_reflections(image, [], gsd)[0], [])
        for invalid in (None, 0, -1, float('nan')):
            self.assertEqual(filter_seedling_reflections(image, entries, invalid)[0], entries)

    def test_classifier_confidence_cannot_bypass_reflection_evidence(self):
        from canopy_detection.seedling_detector import SeedlingCandidate, SeedlingDetector, SeedlingDetectorConfig
        image, entries, gsd = self.scene()
        candidate = SeedlingCandidate(
            x=entries[0]['center'][0], y=entries[0]['center'][1], area_px=10,
            bbox_x=143, bbox_y=147, bbox_w=6, bbox_h=6,
            mean_saturation=90, mean_value=240, mean_exg=100, mean_chroma=.04,
            value_gain=100, exg_gain=80, strong_color=True,
        )
        detector = SeedlingDetector(SeedlingDetectorConfig(mode='legacy'))
        detector.config.mode = 'hybrid'
        detector.model = object()
        with patch.object(detector, 'generate_candidates', return_value=([candidate], {})), \
             patch.object(detector, '_score_candidates', return_value=[.999]):
            mask, accepted, metadata = detector.detect(image, np.zeros(image.shape[:2], np.uint8), gsd)
        self.assertEqual(accepted, [])
        self.assertFalse(np.any(mask))
        self.assertFalse(candidate.accepted)
        self.assertEqual(candidate.rejection_reason, 'specular_water_highlight')
        self.assertEqual(metadata['seedling_reflection_rejected_count'], 1)


@unittest.skipUnless(os.environ.get('MANGROVISION_TEST_IMAGE_DIR'), 'Original regression photos are supplied separately')
class SeedlingReflectionPhotoTests(unittest.TestCase):
    def test_reviewed_glints_are_removed_and_other_candidates_preserved(self):
        root = Path(os.environ['MANGROVISION_TEST_IMAGE_DIR'])
        cases = json.loads(Path(__file__).with_name('seedling_reflection_examples.json').read_text())
        for case in cases:
            with self.subTest(image=case['filename']):
                path = root / case['filename']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), case['sha256'])
                image = cv2.imread(str(path))
                entries = [{'center': point} for point in case['baseline_centers']]
                kept, meta = filter_seedling_reflections(image, entries, case['gsd'])
                negatives = case['glare_centers']
                self.assertEqual([entry['center'] for entry in kept], [e['center'] for e in entries if e['center'] not in negatives])
                self.assertEqual(meta['seedling_reflection_rejected_count'], len(negatives))

    def test_existing_reviewed_leaf_locations_survive_glare_filter(self):
        root = Path(os.environ['MANGROVISION_TEST_IMAGE_DIR'])
        cases = json.loads(Path(__file__).with_name('seedling_ground_artifact_examples.json').read_text())
        for case in cases:
            if not case['positive_centers']:
                continue
            with self.subTest(image=case['filename']):
                path = root / case['filename']
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), case['sha256'])
                image = cv2.imread(str(path))
                entries = [{'center': point} for point in case['positive_centers']]
                self.assertEqual(filter_seedling_reflections(image, entries, case['gsd'])[0], entries)


if __name__ == '__main__':
    unittest.main()
