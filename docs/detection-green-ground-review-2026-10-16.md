# Green ground seedling review — October 16, 2026

Continued on the user's requested **Latest-Main** branch, starting at `770638b`.
This follows the [October 15 detector review](detection-reference-review-2026-10-15.md).

The initial results below describe commit `ae06600`. The follow-up at the end
of this document covers the additional isolated DJI_0720 detections reported
after that commit.

## Problem and change

The supplemental detector still accepted a small flat green highlight in
`DJI_0700.JPG`, also visible in neighboring `DJI_0698.JPG`. It had a compact
color core and enough relative brightness to pass the existing checks. Color
alone does not establish that green material on mud is a standing seedling.

The final seedling evidence filter now checks tiny, ambiguous green highlights
on lit ground for additional nearby object contrast or a compact bright pale
leaf. The check applies when there is no strongly green leaf core, fewer than
two compact leaf components, and less than 0.0032 m² of compact leaf support.
It uses the existing GSD rather than fixed pixel dimensions.

Dark object support must be connected, start within 12 cm of the candidate,
and supply at least 0.002 m² within 30 cm. This allows thin stems and shadows
but prevents a separate dark mud patch or scattered dark noise from validating
the candidate. Strong green leaves, larger leaf clusters, pale bright foliage
and deeply shaded scenes retain their existing paths. Rejections are recorded
as `flat_green_ground` in the existing detector telemetry.

The mature canopy code, white/shaded crown recovery, model weights, class IDs,
confidence 0.80, tile settings, EXIF/GSD, alignment, spacing and safety buffers
were not changed. No database, stored analysis, assignment or field record was
written. There was no backend restart, deployment or push.

## Verification

The complete detector pipeline replayed the same cached real model predictions
on checksum-verified original photos at their original EXIF scales:

| Original | Supplemental candidates before → after | Mature canopy |
| --- | --- | --- |
| DJI_0698.JPG | 22 → 21 | 20.7770 m², identical pixels |
| DJI_0700.JPG | 4 → 3 | 11.7619 m², identical pixels |
| DJI_0712.JPG | 79 → 79 | 69.4498 m², identical pixels |
| DJI_0720.JPG | 124 → 124 | 124.1561 m², identical pixels |

Only the two reported ground highlights changed across those four photos.
Comparing the previous and updated final filter directly confirms both were
previously accepted and are now rejected for `flat_green_ground`.

- **40/40 reviewed leaf locations retained**, including the existing 33 and
  seven additional faint leaf examples in DJI_0349 and DJI_0460. Verified on
  both original files and cached history inputs at their saved GSDs.
- Rechecked all **35 previously cached history analyses**, using their
  checksum-verified images, saved scales and accepted candidate centers.
  Supplemental candidates decreased **4,603 → 4,594**. This checks the changed
  final filter; it is not a fresh full model inference on all 35 images.
- **47 tests passed** with separately supplied original regression photos.
  Added the two real ground negatives, seven real leaf positives, and tests for
  flat green film, connected thin stems, separate dark ground and dark noise.
- `git diff --check` passed.

Reproduce the automated regression checks from the repository root:

```powershell
$env:MANGROVISION_TEST_IMAGE_DIR = 'C:/Users/Lenovo-Pc/Documents/1'
.\venv\Scripts\python.exe -X utf8 -m pytest canopy_detection/tests/test_seedling_ground_artifacts.py canopy_detection/tests/test_small_leaf_recovery.py canopy_detection/tests/test_crown_coverage.py -q -s
```

Local ignored review artifacts are in `tmp/seedling_algae_review_20261016/`:

- `index.html`: original/before/after gallery and enlarged ground comparisons.
- `final_full/summary.json`: four full pipeline replays and unchanged canopy checks.
- `verification.json`: previous/current ground rejection and reviewed leaf checks.
- `history/summary.json`: all 35 cached history candidate checks and nine removals.
- `history/removed_0.jpg`: crops of the nine additional candidate removals.
- `all_tests.xml`: test results.

## Limits and using the change

This is a conservative appearance check, not an algae species classifier or a
trained seedling model. Neighboring views are not independent validation sites.
Candidate reductions and the reviewed examples do not establish overall
precision/recall; unlabelled history candidates remain uncertain. The optional
seedling classifier checkpoint is still absent, as documented in the earlier
review, and this work does not silently substitute or train one.

Restart the backend and analyze the original again to see the updated result.
Saved analyses continue displaying their existing saved overlay; this fix does
not automatically overwrite them or rebuild their planting points.

## Follow-up: isolated DJI_0720 ground markers

The user clarified that only the isolated yellow circles on the mud were wrong;
the branch and foliage circles around mature trees should remain. The previous
fix had left two paths open: fragments of one dull olive patch could count as
a leaf cluster, and a single pale object could exempt itself from ground checks.

The additional checks now require corroboration for an isolated dominant olive
patch or single pale tip. A large olive component cannot qualify merely by
fragmenting into several colored pieces. Pale foliage can still qualify with
independent green support, several pale leaf cores, substantial ordinary leaf
support, or the existing stem/shadow evidence.

These additional rules preserve close mature branch context within 0.75 m of
the existing accepted canopy. The caller supplies the finished canopy mask;
the evidence filter reads it without changing it. Near image edges, where local
context is incomplete, the earlier checks remain in charge. Faint candidates in
a supported planting group retain the established recovery: an independently
supported candidate 0.25–1.25 m away can supply context. Ambiguous candidates
cannot anchor one another, and the earlier ground/wood rejections cannot be
overridden by this group support.

Final verification against `ae06600`:

- **DJI_0720: 124 → 121** supplemental markers. Exactly the three reported
  isolated markers are removed; all remaining 121 positions are identical,
  including the branch/foliage markers. Two markers referred to one ground
  patch; the third referred to the separate pale object lower down.
- **DJI_0698: 21 → 21; DJI_0700: 3 → 3; DJI_0712: 79 → 79.** Full pipeline
  replay uses identical checksum-verified original images, cached real model
  predictions, confidence 0.80 and original EXIF scales. All four mature canopy
  masks remain identical at every pixel.
- **43/43 reviewed leaf locations retained**, including three additional faint
  or pale leaf positions identified while checking for regressions. Checked on
  the original files and saved history inputs at their stored scales.
- **55 regression tests passed**, covering isolated olive fragments, a single
  pale tip, mature branch context, independently supported plant groups,
  ambiguous fragments that must not validate each other, incomplete edge
  context, and the earlier canopy/wood/leaf cases.
- All **35 cached histories** were checked with their original inputs, saved
  scales and cached mature masks: **4,594 → 4,588** supplemental candidates.
  The six additional removals are unlabelled candidate changes, not six proven
  false positives. The review retains crops for inspection; it is not a fully
  annotated accuracy evaluation or a fresh full inference on all 35 images.

The follow-up artifacts are in `tmp/seedling_water_wood_review_20261016/`:
`index.html`, `final_full/summary.json`, `history/verification.json`,
`history/removed_0.jpg`, and `tests.xml`. The originally opened
`tmp/seedling_algae_review_20261016/index.html` also displays this follow-up;
its earlier page is preserved as `index_initial_before_followup.html`.

The source change remains local on Latest-Main. No backend restart, deployment,
push, database write, saved overlay replacement or planting-point change was
performed. Restart the backend and reanalyze the original to use the new filter.
