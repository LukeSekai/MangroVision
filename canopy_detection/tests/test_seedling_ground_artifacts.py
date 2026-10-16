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

    def test_green_tipped_neutral_debris_in_shade_is_not_a_seedling(self):
        for angle in (0, 45, 90):
            for scale in (1, 2):
                with self.subTest(angle=angle, scale=scale):
                    image = np.full((200, 200, 3), (50, 45, 40), np.uint8)
                    cv2.ellipse(image, (100, 100), (13, 3), angle, 0, 360, (110, 120, 118), -1)
                    tip = (100-round(10*np.cos(np.deg2rad(angle))),
                           100-round(10*np.sin(np.deg2rad(angle))))
                    cv2.ellipse(image, tip, (4, 3), angle, 0, 360, (35, 110, 100), -1)
                    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                    kept, meta = filter_seedling_ground_artifacts(
                        image, [{'center': tuple(p*scale for p in tip)}], .01/scale,
                    )
                    self.assertEqual(kept, [])
                    self.assertEqual(meta['seedling_ground_artifact_rejected_reasons'],
                                     {'elongated_neutral_object': 1})

    def test_shaded_leaf_with_real_green_core_or_compact_pale_body_is_preserved(self):
        for color, axes in (((25, 110, 40), (13, 3)), ((35, 110, 100), (6, 4))):
            for scale in (1, 2):
                with self.subTest(color=color, scale=scale):
                    image = np.full((200, 200, 3), (50, 45, 40), np.uint8)
                    cv2.ellipse(image, (100, 100), axes, 0, 0, 360, (110, 120, 118), -1)
                    cv2.ellipse(image, (95, 100), (4, 3), 0, 0, 360, color, -1)
                    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                    entries = [{'center': (95*scale, 100*scale)}]
                    self.assertEqual(filter_seedling_ground_artifacts(image, entries, .01/scale)[0], entries)

    def test_separate_neutral_stick_does_not_change_a_shaded_leaf_shape(self):
        for scale in (1, 2):
            image = np.full((200, 200, 3), (50, 45, 40), np.uint8)
            cv2.ellipse(image, (90, 100), (4, 3), 0, 0, 360, (35, 110, 100), -1)
            cv2.line(image, (99, 100), (119, 100), (110, 120, 118), 4)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            entries = [{'center': (90*scale, 100*scale)}]
            self.assertEqual(filter_seedling_ground_artifacts(image, entries, .01/scale)[0], entries)

    def test_candidate_without_leaf_pixels_is_rejected(self):
        image = np.full((160, 160, 3), (100, 105, 103), dtype=np.uint8)
        self.assertEqual(filter_seedling_ground_artifacts(image, [{'center': (80., 80.)}], .01)[0], [])

    def isolated_shadow_tint(self, scale=1, ground=(100,105,103), leaf=(8,55,20)):
        image = np.full((320,320,3), ground, np.uint8)
        cv2.ellipse(image,(130,130),(4,3),0,0,360,leaf,-1)
        return cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)

    def test_near_black_green_fringe_cannot_validate_isolated_ground(self):
        for scale in (1,2):
            image = self.isolated_shadow_tint(scale)
            kept,info = filter_seedling_ground_artifacts(image,[{'center':(130*scale,130*scale)}],.01/scale)
            self.assertEqual(kept,[])
            self.assertEqual(info['seedling_ground_artifact_rejected_reasons'],{'shadow_green_fringe':1})

    def test_visible_green_and_deeply_shaded_leaves_keep_their_existing_evidence(self):
        for scale in (1,2):
            for ground,leaf in (((100,105,103),(8,95,20)), ((45,50,48),(8,55,20))):
                image = self.isolated_shadow_tint(scale,ground,leaf)
                entries = [{'center':(130*scale,130*scale)}]
                self.assertEqual(filter_seedling_ground_artifacts(image,entries,.01/scale)[0],entries)

    def test_shadow_fringe_with_real_pale_leaf_support_or_supported_group_is_preserved(self):
        for scale in (1,2):
            for grouped in (False,True):
                image = self.isolated_shadow_tint()
                entries = [{'center':(130*scale,130*scale)}]
                if grouped:
                    cv2.ellipse(image,(210,130),(5,3),0,0,360,(30,170,80),-1)
                    entries.append({'center':(210*scale,130*scale)})
                else:
                    cv2.ellipse(image,(134,130),(5,3),0,0,360,(175,210,195),-1)
                image = cv2.resize(image,None,fx=scale,fy=scale,interpolation=cv2.INTER_NEAREST)
                self.assertEqual(filter_seedling_ground_artifacts(image,entries,.01/scale)[0],entries)

    def test_additional_shadow_fringe_check_preserves_close_crown_context(self):
        image = self.isolated_shadow_tint()
        canopy = np.zeros(image.shape[:2],np.uint8)
        cv2.rectangle(canopy,(80,100),(100,160),255,-1)
        original = canopy.copy()
        entries = [{'center':(130,130)}]
        self.assertEqual(filter_seedling_ground_artifacts(image,entries,.01,canopy_mask=canopy)[0],entries)
        np.testing.assert_array_equal(canopy,original)

    def test_compact_brightened_olive_mud_needs_distinct_leaf_color(self):
        for scale in (1, 2):
            image = np.full((160, 160, 3), (82, 100, 97), np.uint8)
            for x, y in ((76, 78), (82, 78), (79, 84)):
                cv2.circle(image, (x, y), 3, (74, 120, 119), -1)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            kept, metadata = filter_seedling_ground_artifacts(
                image, [{'center': (80*scale, 80*scale)}], .007/scale,
            )
            self.assertEqual(kept, [])
            self.assertEqual(metadata['seedling_ground_artifact_rejected_reasons'], {'weak_leaf_color': 1})

    def test_near_white_leaf_can_pass_without_saturated_green(self):
        image = np.full((160, 160, 3), (100, 105, 103), np.uint8)
        cv2.ellipse(image, (80, 80), (8, 5), 0, 0, 360, (175, 210, 195), -1)
        entries = [{'center': (80., 80.)}]
        self.assertEqual(filter_seedling_ground_artifacts(image, entries, .01)[0], entries)

    def test_adjacent_yellow_fragments_cannot_inflate_a_small_green_core(self):
        for scale in (1, 2):
            for angle in (0, 90):
                with self.subTest(scale=scale, angle=angle):
                    image = np.full((160, 160, 3), (105, 110, 108), np.uint8)
                    cv2.ellipse(image, (80, 80), (2, 1), 0, 0, 360, (45, 160, 159), -1)
                    cv2.ellipse(image, (87, 80), (3, 2), 0, 0, 360, (40, 166, 180), -1)
                    if angle:
                        image = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
                    image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                    kept, meta = filter_seedling_ground_artifacts(
                        image, [{'center': (80*scale, 80*scale)}], .01/scale,
                    )
                    self.assertEqual(kept, [])
                    self.assertEqual(meta['seedling_ground_artifact_rejected_reasons'],
                                     {'fragmented_yellow_support': 1})

    def test_separate_yellow_leaves_keep_their_existing_cluster_support(self):
        for scale in (1, 2):
            image = np.full((160, 160, 3), (105, 110, 108), np.uint8)
            cv2.ellipse(image, (80, 80), (2, 1), 0, 0, 360, (45, 160, 159), -1)
            cv2.ellipse(image, (98, 80), (3, 2), 0, 0, 360, (40, 166, 180), -1)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            entries = [{'center': (80*scale, 80*scale)}]
            self.assertEqual(filter_seedling_ground_artifacts(image, entries, .01/scale)[0], entries)

    def test_small_yellow_fragments_keep_close_canopy_and_clipped_context(self):
        for scale in (1, 2):
            image = np.full((160, 160, 3), (105, 110, 108), np.uint8)
            cv2.ellipse(image, (80, 80), (2, 1), 0, 0, 360, (45, 160, 159), -1)
            cv2.ellipse(image, (87, 80), (3, 2), 0, 0, 360, (40, 166, 180), -1)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            entries = [{'center': (80*scale, 80*scale)}]
            canopy = np.zeros(image.shape[:2], np.uint8)
            canopy[:, 30*scale:40*scale] = 255
            original_mask = canopy.copy()
            self.assertEqual(filter_seedling_ground_artifacts(
                image, entries, .01/scale, canopy_mask=canopy,
            )[0], entries)
            np.testing.assert_array_equal(canopy, original_mask)
            far_canopy = np.zeros_like(canopy)
            far_canopy[:, :2*scale] = 255
            self.assertEqual(filter_seedling_ground_artifacts(
                image, entries, .01/scale, canopy_mask=far_canopy,
            )[0], [])
            # A clipped neighborhood cannot establish a whole isolated patch.
            cropped = image[:, 60*scale:].copy()
            clipped_entries = [{'center': (20*scale, 80*scale)}]
            self.assertEqual(filter_seedling_ground_artifacts(cropped, clipped_entries, .01/scale)[0], clipped_entries)

    def ambiguous_green_scene(self):
        image = np.full((160, 160, 3), (105, 124, 123), np.uint8)
        cv2.ellipse(image, (80, 80), (6, 4), 0, 0, 360, (125, 150, 143), -1)
        cv2.ellipse(image, (80, 80), (3, 2), 0, 0, 360, (55, 155, 135), -1)
        return image

    def test_green_mud_highlight_cannot_use_pale_leaf_recovery(self):
        for scale in (1, 2):
            with self.subTest(scale=scale):
                # A small saturated core surrounded by a weak green film.
                # It is brighter than wet mud, but lacks a sunlit pale leaf.
                image = self.ambiguous_green_scene()
                image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                kept, metadata = filter_seedling_ground_artifacts(
                    image, [{'center': (80*scale, 80*scale)}], .007/scale,
                )
                self.assertEqual(kept, [])
                self.assertEqual(metadata['seedling_ground_artifact_rejected_reasons'],
                                 {'flat_green_ground': 1})

    def test_faint_leaf_with_connected_thin_stem_contrast_is_preserved(self):
        for scale in (1, 2):
            image = self.ambiguous_green_scene()
            cv2.line(image, (80, 86), (80, 119), (55, 65, 61), 2)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            entries = [{'center': (80*scale, 80*scale)}]
            self.assertEqual(filter_seedling_ground_artifacts(image, entries, .007/scale)[0], entries)

    def test_separate_dark_ground_cannot_validate_green_highlight(self):
        for scale in (1, 2):
            image = self.ambiguous_green_scene()
            cv2.circle(image, (112, 80), 7, (55, 65, 61), -1)
            image = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
            entries = [{'center': (80*scale, 80*scale)}]
            kept, metadata = filter_seedling_ground_artifacts(image, entries, .007/scale)
            self.assertEqual(kept, [])
            self.assertEqual(metadata['seedling_ground_artifact_rejected_reasons'], {'flat_green_ground': 1})

    def test_scattered_dark_noise_cannot_validate_green_highlight(self):
        image = self.ambiguous_green_scene()
        for y in range(67, 94, 3):
            for x in range(67, 94, 3):
                if (x-80)**2 + (y-80)**2 >= 8**2:
                    image[y, x] = (55, 65, 61)
        self.assertEqual(filter_seedling_ground_artifacts(image, [{'center': (80, 80)}], .007)[0], [])

    def isolated_ground_scenes(self):
        olive = np.full((160, 160, 3), (100, 105, 103), np.uint8)
        # Multiple fragments from one dull olive patch cannot validate each
        # other as a cluster of real leaves.
        for x, axes in ((76, (6, 4)), (89, (2, 2))):
            cv2.ellipse(olive, (x, 80), axes, 0, 0, 360, (30, 122, 120), -1)
        pale = np.full_like(olive, (100, 105, 103))
        cv2.ellipse(pale, (80, 80), (5, 7), 0, 0, 360, (170, 200, 196), -1)
        cv2.ellipse(pale, (80, 85), (2, 2), 0, 0, 360, (95, 145, 140), -1)
        return [(olive, 'weak_leaf_contrast'), (pale, 'flat_green_ground')]

    def test_isolated_olive_fragments_and_single_pale_tip_are_rejected(self):
        for image, reason in self.isolated_ground_scenes():
            for scale in (1, 2):
                with self.subTest(reason=reason, scale=scale):
                    photo = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_NEAREST)
                    kept, meta = filter_seedling_ground_artifacts(
                        photo, [{'center': (80*scale, 80*scale)}], .01/scale,
                    )
                    self.assertEqual(kept, [])
                    self.assertEqual(meta['seedling_ground_artifact_rejected_reasons'], {reason: 1})

    def test_additional_ground_checks_preserve_close_mature_branch_context(self):
        entries = [{'center': (80, 80)}]
        for image, _ in self.isolated_ground_scenes():
            canopy = np.zeros(image.shape[:2], np.uint8)
            cv2.rectangle(canopy, (15, 20), (40, 145), 255, -1)
            original_mask = canopy.copy()
            kept, _ = filter_seedling_ground_artifacts(image, entries, .01, canopy_mask=canopy)
            self.assertEqual(kept, entries)
            np.testing.assert_array_equal(canopy, original_mask)
            for context in (np.zeros_like(canopy), np.pad(np.ones((160, 3), np.uint8), ((0, 0), (0, 157)))):
                self.assertEqual(filter_seedling_ground_artifacts(
                    image, entries, .01, canopy_mask=context,
                )[0], [])

    def test_mature_context_does_not_bypass_existing_ground_rejection(self):
        image, entries = self.scene()
        canopy = np.zeros(image.shape[:2], np.uint8)
        cv2.rectangle(canopy, (15, 20), (40, 145), 255, -1)
        self.assertEqual(filter_seedling_ground_artifacts(image, entries, .01, canopy_mask=canopy)[0], [])

    def test_independently_supported_plant_group_retains_faint_leaf(self):
        for image, _ in self.isolated_ground_scenes():
            cv2.ellipse(image, (20, 80), (4, 3), 0, 0, 360, (30, 170, 80), -1)
            entries = [{'center': (80, 80)}, {'center': (20, 80)}]
            self.assertEqual(filter_seedling_ground_artifacts(image, entries, .01)[0], entries)

    def test_ambiguous_patches_and_duplicate_fragments_cannot_validate_each_other(self):
        for image, _ in self.isolated_ground_scenes():
            image[60:100, :40] = image[60:100, 60:100].copy()
            image = cv2.copyMakeBorder(image, 60, 60, 60, 60, cv2.BORDER_CONSTANT, value=(100, 105, 103))
            entries = [{'center': (140, 140)}, {'center': (144, 140)}, {'center': (80, 140)}]
            self.assertEqual(filter_seedling_ground_artifacts(image, entries, .01)[0], [])

    def test_uncertain_edge_context_cannot_anchor_ambiguous_interior_ground(self):
        for image, _ in self.isolated_ground_scenes():
            image[60:100, :40] = image[60:100, 60:100].copy()
            edge_entry = {'center': (20, 80)}
            entries = [{'center': (80, 80)}, {'center': (76, 80)}, edge_entry]
            self.assertEqual(filter_seedling_ground_artifacts(image, entries, .01)[0], [edge_entry])

    def test_supported_group_does_not_revive_previously_rejected_ground(self):
        image, entries = self.scene()
        cv2.ellipse(image, (20, 80), (4, 3), 0, 0, 360, (30, 170, 80), -1)
        anchor = {'center': (20, 80)}
        self.assertEqual(filter_seedling_ground_artifacts(image, entries+[anchor], .01)[0], [anchor])

    def test_incomplete_context_at_image_edge_preserves_existing_leaf_checks(self):
        for image, _ in self.isolated_ground_scenes():
            cropped = image[:, 60:].copy()
            entries = [{'center': (20, 80)}]
            self.assertEqual(filter_seedling_ground_artifacts(cropped, entries, .01)[0], entries)
        image, _ = self.scene()
        self.assertEqual(filter_seedling_ground_artifacts(
            image[:, 60:].copy(), [{'center': (20, 80)}], .01,
        )[0], [])


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
