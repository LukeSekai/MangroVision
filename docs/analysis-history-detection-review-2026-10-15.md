# Full Analysis History detection review — October 15, 2026

All **35/35 saved analyses** were run locally on
`fix/detection-mud-bright-canopy-20261015`. This is a read-only comparison;
the live Analysis History and its planting records have not been replaced.

## What was tested

Each analysis uses its stored ground scale, 0.80 confidence, 2 m safety radius,
and stored runtime tuning. All originals are 4000 × 2250. Inputs were matched
by the raw upload hash, by reproducing the saved JPEG identity, or downloaded
from private storage with checksum verification. Missing originals in the
local Documents folder were therefore included, not skipped.

The updated code receives fresh GPU model predictions. The previous approved
code at `3f89805` replays those identical predictions to isolate the changes.
The saved historical overlay is shown separately because historical detector
versions and JPEG encoding can also change the output.

Only image-space detection and safety masks were recomputed. Existing mapped
points, GIS exclusions, assignments, footprints and alignment were not regenerated.

## Results

- 35 successful analyses; no processing failures.
- Zero previously accepted canopy pixels removed in any analysis.
- Canopy recovery added coverage in 19 analyses.
- Supplemental candidates across the overlapping images: 6587 → 4603.
- 25 analyses have fewer candidates; 0 have more.
- The leaf-evidence regression check retained all 33 previously reviewed positive examples at the saved ground scales.
- Every accepted canopy/seedling mask is covered by the stored safety radius.
- The optional seedling branch used: fallback_legacy (35).

Counts are detector candidates, not verified unique seedlings or precision/recall.
Candidate reductions include tree leaves absorbed into mature canopy. Nearby and
overlapping photos are not independent samples. Areas below refer to individual
image masks and must not be summed as unique planted land.

## Stored data and the separate asset issue

The run used PostgreSQL READ ONLY transactions and object-storage GET/HEAD only.
Before/after row counts and content digests are recorded for analyses, assets,
planting points, planting events, assignments and assignment points.
Table comparisons: analyses: unchanged, analysis_assets: unchanged, planting_points: unchanged, planting_events: unchanged, planter_assignments: unchanged, planter_assignment_points: unchanged.
The history links to 5,310 planting points, including
1,154 with planting/death records; none were edited.

**Analysis 53 has a pre-existing image-asset integrity discrepancy.** Its full
visualization and preview no longer match their database hashes, byte sizes or
object metadata. Its original image and original preview metadata are consistent,
and the verified raw original was processed successfully. The unverified saved
overlay is omitted from the comparison. No storage repair was attempted.

## Review artifacts

- [Full original / saved / updated gallery](../tmp/history_detection_review_20261015/index.html)
- [Per-analysis CSV](../tmp/history_detection_review_20261015/summary.csv)
- [Verification summary](../tmp/history_detection_review_20261015/verification_summary.json)
- [Storage integrity details](../tmp/history_detection_review_20261015/storage_integrity_issues.json)

These outputs contain the user's images and stay local/ignored. Source and model
hashes are in `run_sources.json`. Starting a backend from this branch affects new
analyses; this review does not activate the branch or rewrite saved results.

| Saved analysis | Source image | Previous code → updated canopy (m²) | Previous code → updated supplemental candidates |
| --- | --- | ---: | ---: |
| Analysis 1 | DJI_0352.JPG | 245.40 → 256.59 | 90 → 57 |
| Analysis 2 | DJI_0689.JPG | 122.92 → 128.29 | 34 → 22 |
| Analysis 3 | DJI_0588.JPG | 217.49 → 223.68 | 63 → 42 |
| Analysis 4 | DJI_0990.JPG | 4.66 → 4.66 | 754 → 742 |
| Analysis 6 | OMIP7152.JPG | 0.00 → 0.00 | 0 → 0 |
| Analysis 7 | NHQJ1178.JPG | 0.00 → 0.00 | 22 → 19 |
| Analysis 8 | MLGP8724.JPG | 0.00 → 0.00 | 0 → 0 |
| Analysis 9 | LODX3270.JPG | 0.00 → 0.00 | 0 → 0 |
| Analysis 10 | OBHO9191.JPG | 0.00 → 0.00 | 24 → 23 |
| Analysis 13 | RVJM4224.JPG | 81.36 → 83.15 | 203 → 124 |
| Analysis 16 | OOGT6807.JPG | 0.12 → 0.12 | 111 → 108 |
| Analysis 21 | KVST2388.JPG | 0.00 → 0.00 | 0 → 0 |
| Analysis 22 | RFBC5866.JPG | 0.00 → 0.00 | 0 → 0 |
| Analysis 23 | XAJI3412.JPG | 225.53 → 239.87 | 502 → 262 |
| Analysis 25 | YAUW1815.JPG | 372.85 → 401.21 | 403 → 159 |
| Analysis 26 | EMNV7587.JPG | 350.48 → 375.60 | 453 → 179 |
| Analysis 28 | PZQN1713.JPG | 0.00 → 0.00 | 20 → 20 |
| Analysis 29 | AQOJ6438.JPG | 0.00 → 0.00 | 0 → 0 |
| Analysis 30 | DAOR9086.JPG | 0.00 → 0.00 | 0 → 0 |
| Analysis 33 | JTZS4533.JPG | 0.00 → 0.00 | 76 → 75 |
| Analysis 35 | WAJG6300.JPG | 175.56 → 186.15 | 491 → 233 |
| Analysis 36 | MTDG3696.JPG | 339.57 → 363.50 | 556 → 221 |
| Analysis 50 | DJI_0189.JPG | 191.35 → 198.92 | 145 → 67 |
| Analysis 53 | DUZC9360.JPG | 75.53 → 77.86 | 147 → 75 |
| Analysis 55 | DJI_0342.JPG | 38.97 → 39.53 | 465 → 416 |
| Analysis 56 | DJI_0150.JPG | 0.00 → 0.00 | 9 → 9 |
| Analysis 57 | DJI_0243.JPG | 30.17 → 31.26 | 65 → 46 |
| Analysis 59 | DJI_0264.JPG | 9.31 → 9.40 | 36 → 26 |
| Analysis 60 | DJI_0349.JPG | 3.46 → 3.46 | 685 → 666 |
| Analysis 61 | DJI_0460.JPG | 70.06 → 74.40 | 859 → 798 |
| Analysis 63 | DJI_0712.JPG | 59.95 → 69.45 | 87 → 79 |
| Analysis 64 | DJI_0717.JPG | 0.00 → 0.00 | 0 → 0 |
| Analysis 65 | DJI_0809.JPG | 66.18 → 85.19 | 14 → 2 |
| Analysis 66 | DJI_0805.JPG | 50.99 → 56.05 | 5 → 0 |
| Analysis 67 | DJI_0692.JPG | 87.73 → 91.53 | 268 → 133 |
