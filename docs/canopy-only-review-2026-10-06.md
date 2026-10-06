# Canopy-only review — 5–6 October 2026

## Completed baseline run

Ran the current mature-canopy detector on all **143 original JPGs** in
`C:\Users\Lenovo-Pc\Documents\1`, at their original 4000 × 2250 resolution.
All completed successfully. The sum of per-photo processing times, including
preview creation, was **2,452 seconds (40.9 minutes)**; gallery creation adds
wall-clock time beyond that sum.

| Result | Photos |
| --- | ---: |
| At least one accepted canopy-mask component | 83 |
| Empty canopy mask | 60 |
| Failed or skipped | 0 |
| Seedling inference invoked | 0 |

The run used the deployed `MangroVision_New/try_model/model_final.pth`, confidence
0.80, existing canopy filters, 512-pixel tiles with 25% overlap, and EXIF ground
scale. The detector ran on the laptop's NVIDIA GPU. Source and model hashes are
recorded in `run_sources.json`. The seedling methods were explicitly disabled
and guarded against accidental invocation.

No alignment, GIS clipping, planting-point generation, API submission, or
database writes ran. The photos and model weights were unchanged. These are
diagnostic detector outputs, **not accuracy, precision, recall, or tree-population
measurements**. Connected mask components include merged crowns and occasional
tiny cleanup fragments. An empty canopy mask does not establish that seedlings
are absent, because that branch was disabled.

The final integrity check verified all 143 source hashes, full-resolution masks,
matching area measurements, zero-seedling metadata, CSV rows, previews, and
gallery entries.

## Canopy filter fix — 6 October 2026

The production appearance filter in `canopy_detection/seedling_leaf_evidence.py`
has now been strengthened. It runs after AI instance cleanup and before canopy
expansion, coverage measurements and danger buffers:

- Small predicted components (up to 1.2 m²) require compact leaf support across
  at least 3% of their mask. A tiny colored fleck cannot validate a whole patch.
- Long, thin predictions with principal aspect at least 4, length at least
  0.60 m and width at most 0.25 m require 10% compact leaf support. This catches
  the reviewed split wood that passed the former aspect floor of 5.
- Photo edges are checked when surrounding pixels are visible. Dense leaf color
  covering at least 25% of a clipped prediction protects genuine foliage that
  appears elongated because the photo cuts through it.
- Rejection telemetry records the position, reason, leaf-support fraction,
  physical area and whether the component touches the photo edge.

The seedling evidence helpers, model weights, predictor resize and public
confidence threshold remain as used in the baseline. These checks improve
appearance validation; they do not identify algae species or establish overall
detector accuracy. Very green non-foliage, heavily shaded plants and artifacts
connected to larger canopy masks remain limitations of this rule-based check.

### Regression verification

The targeted suites passed **33 tests**, including a new fixture set of **20
original-photo cases: 9 reviewed ground/wood negatives and 11 genuine foliage
positives**. All 9 negatives were removed and all 11 positives retained. The
positive cases include plants beside stakes, the narrow leafy PWGX branch,
and clipped crowns in HZYT, EMNV and KPLO.

Fresh mature-canopy AI inference also completed on **8 original photos**, with
the seedling branch explicitly disabled throughout:

| Photo | Baseline canopy area | Revised canopy area | Result |
| --- | ---: | ---: | --- |
| HTWO9123 | 0.449 m² | 0.000 m² | Reported ground patch removed |
| VSIL7626 | 0.342 m² | 0.000 m² | Ground patch at the photo edge removed |
| QJBJ8805 | 97.323 m² | 97.222 m² | Split woody object removed |
| BMIY0408 | 0.121 m² | 0.121 m² | Leafy plant preserved |
| ITBH6710 | 120.169 m² | 120.169 m² | Foliage mask unchanged |
| EMNV7587 | 350.481 m² | 350.481 m² | Foliage mask unchanged |
| HZYT2806 | 375.721 m² | 375.721 m² | Clipped leafy crowns preserved |
| PWGX9067 | 377.336 m² | 377.126 m² | Narrow leafy branch retained; weak isolated fragments removed |

The revised cleanup was also replayed against the cached AI predictions from
**all 143 photos**. Before changing the filter, the replay reproduced every
baseline mask exactly, establishing that the replay follows the recorded
production cleanup path. With the final filter:

- 37 photos had changed masks; 106 were identical to the baseline.
- 78 photos retained nonempty canopy masks, and 65 were empty. Five previously
  nonempty photos became empty.
- All 9 reviewed negative components were completely removed. All pixels in
  the recorded footprints of the 11 positive cases were retained.
- All 8 fresh AI masks matched their revised replay masks exactly.
- No canopy pixels were added. Total coverage removed across the collection
  was 15.190 m²; this is an output difference, not a false-positive area metric.

Reviewed close-ups of all 19 baseline components at least 0.05 m² that lost more
than 20% of their pixels. They include the reported ground patch, obvious wood,
uncertain olive patches and weak root/shadow regions. The whole collection is
not annotated, so changed masks are not all labeled as confirmed corrections.

Updated local artifacts:

- [Revised all-photo gallery](../tmp/canopy_review_fixed_final/index.html).
- [Revised CSV summary](../tmp/canopy_review_fixed_final/summary.csv).
- [Fresh eight-photo AI gallery](../tmp/canopy_review_final_ai/index.html).
- `tmp/canopy_review_fixed_final/changes.json`: per-photo changes and original
  component retention.
- `tmp/canopy_review_fixed_final/changes_review_01.jpg` and
  `changes_review_02.jpg`: reviewed removal close-ups.

Reproduce cleanup validation from the original audit cache with:

```powershell
.\venv\Scripts\python.exe -X utf8 scripts/replay_canopy_review.py `
  C:/Users/Lenovo-Pc/Documents/1 tmp/canopy_review_all tmp/canopy_replay_repeat
```

This replay invokes neither AI branch. For fresh AI inference on selected
photos, use the original runner with `--only HTWO9123 BMIY0408`, optionally adding
other exact filenames or stems.

The backend was restarted to load the new checks. The existing Vercel testing
website was reconnected, and its public API readiness check passed. Existing
saved analysis images are unchanged; rerun a photo to generate a revised result.

## Baseline findings

Reviewed all 12 overview sheets and selected original-resolution component
crops, including every component in the final weak-evidence shortlist. This is
a visual audit, not an exhaustive annotation of every crown boundary.

| Photo | Original pixel position (x, y) | Observation |
| --- | --- | --- |
| HTWO9123 | 1200, 1702 | Reproduces the user-reported mud false positive with seedlings disabled. The isolated olive ground patch covers approximately 0.449 m². |
| HZAP6304 | 1733, 1780 | Similar ground-patch detection, likely another view of the same object. |
| VSIL7626 | 1161, 2205 | Similar patch is accepted at the image edge. The small-canopy appearance filter exempts edge fragments. |
| FRQF2365 | 3503, 1206 | An elongated split woody object is accepted as canopy. |
| QJBJ8805 | 3507, 803 | Another view of the same split wood is accepted. |
| UQZW5316 | 3492, 1631 | A third view of the split wood is accepted. |
| PFOR6071 | 642, 1722 | A forked woody object on exposed mud is accepted. |
| TWWY5921 | 648, 936 | Another view of the forked wood is accepted. |
| RCMJ6444 | 262, 1749 | A pale rectangular non-foliage object is accepted; appears to be wood or debris. |
| JPMT6854 / NHQJ1178 / SRPA2267 | 2994, 1711 / 2938, 2204 / 2950, 1894 | Similar flat olive-green patches are accepted. No distinct leaf cluster is visible in the reviewed crops; material identity needs confirmation. |
| HYVK5611 | 1088, 2110 | Olive-coated pale ground object is accepted; exact material is uncertain. |
| BMIY0408 / EUPU9294 / TAKQ9970 | 3343, 2057 / 3353, 1651 / 3403, 345 | Visible leaf clusters beside stakes are retained. These are useful positive comparisons when filtering wood. |
| PWGX9067 | 3472, 318 | A long narrow branch has visible foliage. Shape alone would be an unreliable reason to reject it. |
| JTZS4533 / PZQN1713 | — | No canopy output in this pass. This does not assess their small-seedling detections. |

RGB photos alone do not establish whether an olive patch consists of mud,
algae, moss, or other material. The reported HTWO case and obvious woody objects
provide useful negative examples; uncertain objects should be reviewed before
becoming training labels. Multiple views of the same objects must remain grouped
when separating training and evaluation data.

## Why the HTWO case survives

The recorded pre-cleanup candidates include a model confidence of **0.963** on
the ground patch. Increasing the public confidence threshold alone would not
remove it at ordinary operating thresholds.

Its saturation is high enough to bypass the low-saturation rejection rule even
though its excess green is weak. The small-canopy filter asks whether any compact
leaf-like core exists inside the mask. It does not require substantial support
relative to the complete predicted area.

Under the diagnostic evidence calculation, the final 7,146-pixel HTWO mask has
only **12 compact leaf-support pixels (0.17%)**. In comparison, the retained
BMIY and EUPU leafy clusters have approximately **44% and 45%** compact support.
The calculation is on the final coverage mask; application filtering happens
before expansion and merging. This points to a weak object-evidence check, not a
validated universal threshold.

The standalone diagnostic ranking examined 694 small non-edge components and
flagged 16 substantial components for visual inspection using weak leaf support
or elongated shape. That shortlist contains genuine foliage and uncertain
boundaries as well as false positives, so **16 is not a false-positive count**.

These findings motivated the filter revision and regression cases documented
above. The stake-adjacent leaf clusters and narrow leafy branch were preserved
as positive comparisons.

## Controlled resize comparison

The active predictor uses `MIN_SIZE_TEST=800`, `MAX_SIZE_TEST=1333`, while the
checkpoint metadata declares 512 for both. A separate offline three-photo
comparison used those declared sizes with the same weights and filters:

| Photo | Current resize: canopy area | Declared 512 resize: canopy area |
| --- | ---: | ---: |
| HTWO9123 | 0.449 m² | 0.472 m² |
| BMIY0408 | 0.121 m² | 0.133 m² |
| ITBH6710 | 120.169 m² | 124.188 m² |

The HTWO false positive remains at both settings. The comparison does not
establish equivalent precision or recall across the dataset. Production resize
settings and detector logic were unchanged during the baseline diagnostic run.
The subsequent appearance-filter fix is documented above.

## Local artifacts and reproduction

Outputs are local in `tmp/canopy_review_all/`, ignored by Git:

- [Original/overlay gallery](../tmp/canopy_review_all/index.html).
- [Per-photo CSV summary](../tmp/canopy_review_all/summary.csv).
- [HTWO close-up](../tmp/canopy_review_all/HTWO9123_crops.jpg).
- [Wood example](../tmp/canopy_review_all/UQZW5316_crops.jpg).
- `overview_01.jpg` through `overview_12.jpg`: all 143 photos.
- Per-photo JSON: source hash, EXIF scale, accepted components, candidate scores
  and polygons before cleanup, and filter telemetry.
- Full-resolution PNG masks, original/overlay JPG previews, and component crops.
- `review_notes.json`, `component_evidence.json`, and `gate_diagnostics.json`.
- `weak_evidence_01.jpg` and `weak_evidence_02.jpg`: inspected shortlist.
- `checkpoint_resize_512/`: separate three-photo controlled comparison.
- `integrity_check.txt`: final artifact verification.

Run from the repository root, using a new output directory:

```powershell
.\venv\Scripts\python.exe -X utf8 scripts/review_canopies.py `
  C:/Users/Lenovo-Pc/Documents/1 tmp/canopy_review_repeat
```

The runner selects CUDA when available and otherwise uses CPU. Use
`--device cpu` explicitly for a CPU comparison. `--resume` reuses completed
photos only when source hashes and detector settings match the original run.
The reusable runner creates the gallery, summaries, overviews, masks, and
per-photo records; the additional evidence shortlist and resize experiment were
generated locally for this review.
