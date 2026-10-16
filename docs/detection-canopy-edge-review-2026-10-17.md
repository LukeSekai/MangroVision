# Isolated mud candidates beside canopy — October 17, 2026

The reported Analysis 40 uses the original DJI_0697.JPG. Its saved settings
specify confidence 0.80, an EXIF GSD of 0.6945434 cm/px and the existing
`fallback_legacy` seedling path. The optional learned seedling checkpoint is
still unavailable. A fresh GPU run reproduced its 45 supplemental candidates.

The two isolated mud markers above and left of the main tree were reviewed
separately from visible branch foliage. This preserves the earlier approved
distinction: the isolated ground marks are unwanted, while the existing
branch/foliage path remains eligible.

## Cause and change

The upper marker's high normalized green chroma comes from near-black mud
pixels, not an independently visible green leaf core. That tiny dark fringe
was sufficient to activate the strong-leaf exemption for an adjacent olive
highlight. The other marker uses pale-leaf recovery to accumulate a larger,
low-contrast green smear without any compact ordinary leaf component or
compact bright pale core.

The final evidence filter now treats these two appearances as ambiguous on
lit, isolated ground:

- A recovered pale region with no compact ordinary or bright pale leaf core
  cannot independently establish a seedling merely by accumulating area.
- For a small single patch, green chroma confined to pixels below the existing
  dark-object threshold cannot independently establish a strong green leaf.
  Substantial or multiple pale leaf cores retain their existing exemption.

These checks use the existing GSD-scaled area floors and dark-object threshold.
They apply outside the accepted canopy's 0.75 m branch context, with more than
0.75 m of image-edge context and a local ground value of at least 80. The
existing independently supported plant-group path at 0.25–1.25 m remains
available. Ambiguous candidates cannot validate each other. Real visible green
cores, deep shade, close crown foliage and existing hard rejections retain
their established behavior. Rejection telemetry uses `shadow_green_fringe`
and `diffuse_pale_ground`.

No mature-canopy logic, white/shaded crown recovery, model weights, confidence,
GSD, alignment, spacing, buffers, forbidden/erosion rules or saved-point checks
were changed.

## Verification

- **Fresh DJI_0697 GPU inference: 45 → 43 accepted candidates.** Exactly the
  two reported isolated ground locations are removed. Every other accepted
  position is identical. Both runs process the original 4000 × 2250 photo,
  66 tiles, confidence 0.80 and its original EXIF scale.
- The mature canopy masks are **identical at every pixel**, with 23.0619 m²
  of detected coverage in both runs.
- **65 regression tests passed**, with original photo fixtures supplied.
  The new cases cover the two real negatives, four visible foliage guards,
  dark fringe rejection at two resolutions, visible/deeply shaded green leaves,
  real pale support, independently supported groups and close crown context.
  All 44 previously reviewed leaf locations are retained; with the four new
  foliage guards, 48 reviewed leaf-appearance locations pass.
- All **35 cached history inputs** were rechecked with their checksum-verified
  images, saved scales and mature masks. Their **4,585 previously accepted
  candidates remain unchanged**, including all 44 prior leaf guards at those
  saved scales. This is a final-filter replay, not fresh model inference on
  all histories.
- Four complete pipeline replays use checksum-verified cached real model
  predictions for DJI_0698, DJI_0700, DJI_0712 and DJI_0720. Their existing
  accepted positions and mature-canopy pixels remain unchanged.

Local [before/after review](../tmp/seedling_canopy_edge_20261017/index.html),
`comparison.json`, `regression_tests.xml`, `four_references/summary.json` and
`history/verification.json` are under `tmp/seedling_canopy_edge_20261017/`.

These counts describe detector candidates, not verified distinct seedlings or
overall precision/recall. The changes address the reviewed appearances; they
are not a trained algae/material classifier.

## Activation

Restart the backend before reanalyzing to use the changed filter. Saved result
images remain snapshots. No saved analysis, planting point, assignment, zone or
field record was modified, and no service restart, deployment or push was
performed. Existing analysis-deletion protections remain unchanged.
