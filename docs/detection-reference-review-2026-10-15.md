# Mud candidates and incomplete canopy coverage — October 15, 2026

Work is isolated on `fix/detection-mud-bright-canopy-20261015`, based on
`3f89805`. VS Code was opened at `tmp/worktrees/detection-mud-bright-canopy`.
The original checkout remains on `Latest-Main`.

## Reproduced problems and changes

- Some locally brighter olive mud and pale twigs passed the supplemental
  seedling check. Candidates now need a compact, distinct leaf-color core.
  A separate bright, pale-leaf path preserves the reviewed near-white seedlings;
  existing leaf-area and wood-shape checks still apply.
- Pale crown interiors and attached shaded foliage were missing from mature
  coverage. Recovery starts from validated AI canopy components of at least
  1.2 m². Connected foliage can extend up to 1.5 m from those anchors. Pale
  material must also lie within a crown's convex envelope and within 0.75 m of
  accepted canopy pixels. Recovery never fills the entire envelope.
- Recovery preserves every accepted canopy pixel, joins only roughly 2 cm
  gaps, and rejects a connected recovery whose added area exceeds 35% of its
  mature anchor area. Large dark gaps and isolated bright ground remain separate.
- Mature coverage now finishes before seedling inference, so recovered tree
  leaves are excluded from supplemental seedling detection. Seedling markers
  remain separate from canopy area in legacy, hybrid and classifier-fallback modes.
- Polygon conversion now retains valid pieces when a pixel-diagonal contact
  repairs into a `MultiPolygon`, instead of dropping the entire repaired crown.

Model weights, class IDs, confidence 0.80, tile size/overlap, EXIF/GSD,
alignment, species spacing, GIS exclusions and safety-buffer settings were
preserved. The unused class-0 head was not enabled. No database, saved analysis,
email, deployment or running backend was changed.

## Reference comparison

Original full-resolution photos were read from the separately supplied
`Documents/1` folder. Before and after use the same real GPU model predictions
and EXIF scale. Final canopy results were also checked with fresh GPU inference.

| Original | Canopy before → after (m²) | Supplemental candidates before → after |
| --- | ---: | ---: |
| DJI_0698.JPG | 19.02 → 20.78 | 54 → 22 |
| DJI_0700.JPG | 11.33 → 11.76 | 16 → 4 |
| DJI_0712.JPG | 59.95 → 69.45 | 87 → 79 |
| DJI_0720.JPG | 119.72 → 124.16 | 268 → 124 |

Candidate reductions include leaves correctly absorbed into mature canopy;
they are **not** counts of independently confirmed false positives. The new
photo regressions reject the inspected mud/twig locations in DJI_0698 and
DJI_0700, and retain five clearly leafy seedling locations in DJI_0712. The
bright central crown in DJI_0698 and shaded lower-left crown in DJI_0712 have
dedicated coverage regressions.

These nearby views belong to one site group, not independent evaluation
samples. Small ambiguous nubs can still resemble vegetation. This review does
not establish precision/recall, complete crown delineation or species accuracy.
The optional seedling-classifier checkpoint was absent from the default asset
location, so the comparison uses the legacy fallback; no replacement classifier
was trained or installed.

## Verification and local artifacts

- 48 tests passed, including original-photo, pale/yellow/shaded-leaf, wood/mud,
  crown recovery, multipart polygon and all seedling-mode checks.
- Fresh canopy inference also covers DJI_0342 and DJI_0460 as additional scenes.
- All 143 retained October 6 canopy masks preserved every existing canopy pixel.
  Recovery added coverage in 69, and the growth cap blocked one excessive group.
  The 12 largest relative changes were visually inspected. This is a cached-mask
  audit, not a new model-accuracy test.
- Original hashes, model/source hashes and EXIF scale are recorded in the local
  review outputs. The original photos and checkpoint were read without changes.
- Final canopy-only GPU masks exactly match the seeded replays for all four
  references. The existing 2 m safety-buffer calculation covers all accepted
  vegetation; five reviewed ground probes no longer create exclusion halos.

Local outputs are ignored by Git:

- [Before/after gallery](../tmp/detection_review_20261015/review/index.html)
- [Four-photo summary](../tmp/detection_review_20261015/review/summary.json)
- [Integrity and safety verification](../tmp/detection_review_20261015/review/verification.json)
- [Fresh canopy gallery](../tmp/detection_review_20261015/final_canopy/index.html)
- [Wider cached-mask audit](../tmp/detection_review_20261015/wider_canopy_audit/summary.json)
- [Test report](../tmp/detection_review_20261015/final_tests.xml)

From this worktree, use the original checkout's existing Python environment and
model asset for an offline canopy review:

```powershell
& ..\..\..\venv\Scripts\python.exe -X utf8 scripts/review_canopies.py `
  "$env:USERPROFILE\Documents\1" tmp/canopy-review `
  --only DJI_0698 DJI_0700 DJI_0712 DJI_0720 `
  --model ..\..\..\MangroVision_New\try_model\model_final.pth

$env:MANGROVISION_TEST_IMAGE_DIR = "$env:USERPROFILE\Documents\1"
& ..\..\..\venv\Scripts\python.exe -X utf8 -m pytest canopy_detection/tests -q -s
```

The existing Windows API module wraps stdout during import; pytest capture can
close its stream during the mixed geometry/API suite. The recorded complete run
uses `-s` and the JUnit report confirms 48 tests, zero failures/errors/skips.
No production stdout code was changed for this test-runner issue.

These fixes affect new analyses after starting a backend from this branch.
Existing saved overlays retain their original results until explicitly reanalyzed.
