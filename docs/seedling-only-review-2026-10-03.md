# Seedling-only review — 3 October 2026

## Completed run

Ran all **143 original JPGs** in `C:\Users\Lenovo-Pc\Documents\1` at their
original resolution. All completed successfully; none were skipped or failed.
The sum of per-image processing times was 1,104 seconds (18.4 minutes).

| Result | Photos |
| --- | ---: |
| At least one accepted marker | 98 |
| No accepted markers | 45 |
| Rule-based fallback mode | 143 |

The mature-canopy predictor was never loaded or invoked. The review used the
seedling hybrid/fallback dispatch, default tuning, EXIF camera scale, and an empty
canopy exclusion mask. Consequently, tree foliage also receives markers: these
counts are **not seedling population estimates**, accuracy scores, or directly
comparable to full app analyses. No GIS clipping or planting-point generation ran.

The trained seedling checkpoint at `models/seedling_classifier/best.pt` was
missing, so every photo used `fallback_legacy`. This is separate from the existing
large-canopy model. Detector code, model weights, original images, saved analyses,
and database records were not changed during this diagnostic pass.

## Findings

Reviewed 12 overview sheets covering all 143 images, then inspected selected
candidate crops at original resolution. This is a visual audit, not an exhaustive
annotation of every candidate or every missed seedling.

| Photo | Original pixel coordinates (x, y) | Observation |
| --- | --- | --- |
| LUXA2941 | 3301, 1017 | A visibly split woody object still receives a seedling marker. |
| PBPT6413 | 3327, 518; 3358, 509 | Another view of the same wood receives two markers. |
| RVJM4224 | 3262, 1538 | The same wood is marked in a third view. |
| JZZO6134 | 1344, 429 | An elongated log with yellow-green surface material is marked. |
| VSIL7626 | 646, 265; 663, 282 | Two markers on a flat olive strip beside a stake shadow; suspected ground/algae, with no distinct leaf cluster visible. Material identity needs confirmation. |
| LUXA2941 | 2900, 776 | Small ground-colored feature passes; too ambiguous to use as a training label without review. |
| ALYZ1116 | 1029, 1745; 1039, 1745 | Two nearby centers appear to represent the same leaf cluster. |
| BHOC6267 | 269, 1265; 283, 1275 | Another apparent repeated marker on a leaf cluster. |
| JTZS4533 | 2672, 903 | Reviewed pale seedling leaves remain detected. |
| PZQN1713 | 242, 1345 | A previously recovered small leaf cluster remains detected. |

Many exposed sticks in the mud-only frames are already rejected. The failures
above show that the remaining problem also occurs on yellow-green wood surfaces
and weakly colored ground material. These must be distinguished from small pale
leaves, which the current recovery rules are intended to preserve.

### Filter trace

The LUXA wood passes with 132 apparent leaf-support pixels (about 0.0084 square
metres), despite not satisfying the strong-green check. This exceeds the size
limits under which the current elongated-wood and line-context checks run.
The JZZO log also passes with weak leaf support. The small LUXA ground feature
passes the pale-leaf recovery path. Trace values are in the local
`gate_diagnostics.json` artifact.

The next detector revision should use these recorded wood/ground examples as
negative checks, retain the pale-leaf positive checks, and address repeated
centers separately. Ambiguous objects should be reviewed before being used to
train the missing classifier. Adjacent views of the same objects must stay
together when separating training and evaluation data.

## Review artifacts

Local outputs are in `tmp/seedling_review_all/` (ignored by Git):

- [Original/overlay gallery](../tmp/seedling_review_all/index.html)
- [Per-photo summary](../tmp/seedling_review_all/summary.csv)
- [LUXA close-ups](../tmp/seedling_review_all/LUXA2941_crops.jpg)
- [VSIL ground-strip close-ups](../tmp/seedling_review_all/VSIL7626_crops.jpg)
- `overview_01.jpg` through `overview_12.jpg`: all-photo overview sheets.
- 98 candidate crop sheets, prioritizing detections with less surrounding vegetation.
- Per-photo JSON: accepted coordinates, rejection telemetry, EXIF scale, original-file hash.
- Per-photo PNG masks and original/marked JPG previews.
- `review_notes.json`: visual observations, including uncertain cases.
- `run_sources.json`: hashes of detector source files used for this run.

Reproduce the detector run from the repository root:

```powershell
venv/Scripts/python.exe -X utf8 scripts/review_seedlings.py `
  C:/Users/Lenovo-Pc/Documents/1 tmp/seedling_review_all
```

The reusable runner produces the gallery, summaries, masks, and per-photo
metadata. The additional overview/crop sheets were generated for this review.
