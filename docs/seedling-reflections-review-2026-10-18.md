# Seedling sunlight-reflection review — 18 October 2026

The original DJI_0885 and DJI_0312 photos contained accepted seedling markers on
reflected sunlight. Clipped red/green channels can leave yellow-green fringes
that pass the existing color and compact-leaf checks. Some also entered through
yellow-leaf recovery, after the earlier water-context check.

`canopy_detection/seedling_reflections.py` now checks every accepted candidate
after recovery and before the final ground-evidence check. A candidate must be
near a clipped highlight, have another separate highlight nearby, and lie on a
smooth, weakly green background before this guard can reject it. Background
texture is measured after excluding highlights and their immediate fringes.
Independent compact green/yellow leaf bodies and connected dark plant structure
protect real foliage. Distances and areas scale with the original photo's GSD.
Reflections cannot anchor another ambiguous candidate in the final ground check.
The optional learned classifier path uses the same guard before drawing markers.

Fresh full GPU inference used the existing canopy checkpoint, confidence 0.80,
original 4000 × 2250 photos and EXIF ground scale, both before and after the fix:

| Photo | Before candidates | After candidates | Reviewed glare markers removed |
| --- | ---: | ---: | ---: |
| DJI_0885.JPG | 55 | 51 | 4 |
| DJI_0312.JPG | 43 | 33 | 10 |

All 84 other candidate positions and every mature-canopy pixel were identical.
These counts describe detector output, not verified seedling population or
overall accuracy. The runtime's optional seedling classifier was unavailable;
the photo runs used its established legacy fallback. A separate integration test
checks that a high classifier score cannot bypass the glare guard.

The 35 checksum-verified cached history inputs were also checked at their saved
scales. All 4,585 previously retained candidates and 44 reviewed leaf locations
were preserved. Early versions rejected pale leaves and were revised before the
final verification; the additional sunlit-leaf fixtures protect these cases.
The focused reflection, ground-artifact, leaf-recovery and crown suites passed
78 tests, with the original photos provided. Additional photo fixtures were
checked again after adding the preservation cases.

Local review: [before-and-after gallery](../tmp/seedling_reflections_20261018/index.html).
It includes all 14 removed glare crops, preserved leaf examples, fresh inference
results and the cached-history verification. The original images, model weights,
alignment logic, planting rules and saved records were not modified by this
review. No database queries or writes were needed.

Restart the backend and analyze the original photos again to use the source
change. Existing saved analyses are not rewritten. This branch has not been
deployed by the local review; publishing the source and restarting/deploying the
processing backend are separate steps.

## Current analysis-history comparison

All 54 images present in analysis history were subsequently compared using
checksum-verified copies of their stored originals, each saved GSD and confidence,
and fresh canopy-model inference. Both versions share the same canopy prediction;
only the new reflection check differs. All images completed, with no failures.

51 comparisons were unchanged. The other three changed as follows:

| Stored history image | Before candidates | After candidates |
| --- | ---: | ---: |
| DJI_0460.JPG | 732 | 731 |
| DJI_0885.JPG | 43 | 42 |
| DJI_0312.JPG | 38 | 32 |

Totals changed from 4,354 to 4,346 candidates. The additional DJI_0460 removal is
a small green patch beside a bright stake and remains **uncertain**, not a
confirmed glare correction. It is flagged in the gallery for visual review.
The stored image copies can produce different counts from the camera-original
tests above. This comparison did not overwrite any saved analysis or field data.

Local artifact: [54-image comparison](../tmp/seedling_history_comparison_20261018/index.html).
Input and source checksums, saved scales, marker masks and gallery links were
verified. The page includes all images, a changed-only filter and close-ups of
the removed candidates. Storage access used reads only.

Run the focused suites from the repository root:

```powershell
$env:MANGROVISION_TEST_IMAGE_DIR = 'C:/Users/Lenovo-Pc/Documents/1'
.\venv\Scripts\python.exe -X utf8 -m pytest -q -s canopy_detection/tests/test_seedling_reflections.py canopy_detection/tests/test_seedling_ground_artifacts.py canopy_detection/tests/test_small_leaf_recovery.py canopy_detection/tests/test_crown_coverage.py
```
