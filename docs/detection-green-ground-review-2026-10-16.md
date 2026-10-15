# Green ground seedling review — October 16, 2026

Continued on the user's requested **Latest-Main** branch, starting at `770638b`.
This follows the [October 15 detector review](detection-reference-review-2026-10-15.md).

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
