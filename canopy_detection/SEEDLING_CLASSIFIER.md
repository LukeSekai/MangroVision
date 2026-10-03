# Seedling classifier workflow

The runtime detector keeps mature Detectree2 crowns separate from small-seedling candidates.  Hybrid mode is enabled by default, but it falls back to the existing legacy color supplement until a trained checkpoint is available.

## Dataset

For an offline review of all photos in a directory, run the seedling branch
without loading the mature-canopy predictor:

```powershell
python scripts/review_seedlings.py 'C:\path\to\original\photos' tmp/seedling_review_all
```

Open `tmp/seedling_review_all/index.html` for original/overlay pairs. The output
also includes per-photo masks, JSON metadata with coordinates and source hashes,
and `summary.csv`. Magenta rings mark accepted centers. This run uses EXIF scale,
default detector settings, and an empty canopy exclusion mask; tree foliage can
therefore receive seedling markers. Counts must not be interpreted as seedling
population estimates or compared directly to the full analysis. No model weights,
original photos, or database records are changed. Photos without a usable relative
altitude are reported as errors rather than assigned a guessed scale.

Create a JSONL manifest with one record per original flight image:

```json
{"image":"frames/site_a_001.JPG","gsd":0.010,"split":"train","seedlings":[[120,80]],"negatives":[[340,210],[500,90]]}
```

Use `train`, `valid`, and `test` splits by flight/site, not by adjacent frames. Mark hard negatives explicitly in mud, wet sand, water ripples, shadows, rocks, debris, and non-seedling vegetation.

Generate an ImageFolder dataset:

```powershell
python canopy_detection/prepare_seedling_dataset.py `
  --manifest path/to/seedling_points.jsonl `
  --output path/to/seedling_dataset
```

The output must contain `train`, `valid`, and `test`, each with `seedling` and `negative` folders.

## Training

```powershell
python canopy_detection/train_seedling_classifier.py `
  --data-root path/to/seedling_dataset `
  --output models/seedling_classifier/best.pt
```

The selected validation threshold targets at least 90% precision. The saved checkpoint is independent of the Detectree2 weights.

## Runtime

The detector searches for `models/seedling_classifier/best.pt`. Override it with `MANGROVISION_SEEDLING_CLASSIFIER_PATH`. Runtime modes can be supplied through `ai_runtime_tuning`:

- `legacy`: existing color supplement;
- `hybrid`: learned classifier plus conservative candidate gates;
- `off` or `disabled`: no seedling supplement.

If hybrid mode cannot load its checkpoint, it reports `fallback_legacy` and preserves existing behavior.

The legacy/fallback supplement checks accepted candidates against nearby ground
after all color and cluster recovery steps. Weak green fragments in stick shadows
or mud are rejected when they lack local leaf evidence. Bright yellow leaves and
strong green leaves can protect shaded parts of the same seedling. The diagnostic
`seedling_ground_artifact_rejected_count` records these rejections. This is a
conservative color/contrast filter, not a trained wood classifier; validation on
annotated images from other flights is still needed to measure overall accuracy.

Regression check on `OOGT6807.JPG` (4000 x 2250, EXIF GSD 0.00782238 m/px):
the selected canopy model with `fallback_legacy` produced 133 seedling markers
before this gate and 124 after it. All six lower-left stick/mud markers were
removed; six manually checked visible leaf locations were retained. These are
targeted regression observations, not a precision/recall benchmark. Existing
saved overlays are not recalculated; process the image again to apply the gate.

### Follow-up: stick highlights and green shadow fringes

The ground check now requires a connected, compact leaf core to protect weak
candidates. A few scattered yellow pixels on wood are insufficient. Dark green
fringes need stronger green color to qualify as leaf evidence; compact bright
yellow and shaded green leaves remain eligible. Core area is measured in square
metres using the image GSD, and principal-axis aspect rejects elongated highlights
at any orientation. This changes the legacy/fallback seedling filter, not model
weights or mature-canopy masks.

On the original 4000 x 2250 photos, running the supplement with an empty mature
canopy mask gave these targeted before/after results:

| Photo | Previous filter | Updated filter |
| --- | ---: | ---: |
| RFBC5866.JPG | 3 | 0 |
| KVST2388.JPG (Analysis 17) | 1 | 0 |
| OOGT6807.JPG | 125 | 124 |
| TWWY5921.JPG | 343 | 338 |

The full RFBC5866 run with `try_model/model_final.pth`, confidence 0.80, and
33 processed tiles also returned zero seedling markers in `fallback_legacy`
mode (five candidates rejected by the updated gate). It returned zero canopy
instances, matching the reported analysis. The supplement left its input canopy
mask unchanged, and the final seedling mask contained no pixels.

The four reported stick detections were removed. The six previously reviewed
OOGT6807 leaf locations remained, and all six previously reviewed stick/shadow
locations remained excluded. Counts are supplement markers, not an accuracy score;
the other photos are not exhaustively annotated. A separate regression manifest
records these ten negative and six positive locations with original-file hashes.
The photos remain separately supplied assets. Run those checks with:

```powershell
$env:MANGROVISION_TEST_IMAGE_DIR = 'C:\path\to\original\photos'
python -m unittest canopy_detection.tests.test_seedling_ground_artifacts -v
```

Without that environment variable, only the synthetic cases run and the photo
test is skipped. The tests cover scattered wood highlights, elongated highlights,
dark green fringes, green/yellow leaves, shaded leaves, image edges, and scaling.
Restart the development launcher once to load its updated reload configuration.
It now watches both the API and `canopy_detection` directories. Saved analyses
keep their previous results until processed again.

Run the model-independent regression cases with:

```powershell
python -m unittest discover -s canopy_detection/tests -v
```

### Small leaves and olive ground patches (PZQN1713)

Small AI canopy components (up to 1.2 m²) now require a connected leaf core
inside the predicted mask before they produce exclusion buffers. The core
must be brighter than the surrounding ground or distinctly green. This removes
the reviewed olive algae patch in PZQN1713; a nearby leaf outside that prediction
cannot validate it. Larger crowns and components cut off by an image edge
keep the existing canopy checks because their full extent is unknown.

The seedling ground check also covers moderately green wood-shadow fringes.
Yellow-leaf recovery accepts smaller leaf areas and thicknesses, with a stronger
absolute excess-green requirement and the existing local contrast checks.
A compact leaf cluster can remain eligible when connected to a thin bright
stem; a small highlight on a long piece of wood is still rejected.

On the original PZQN1713 photo, three reviewed missed leaf clusters near
(417, 580), (242, 1345), and (397, 2067) are recovered. The reviewed stick/shadow
locations near (897, 849) and (478, 2116) are excluded. The regression fixture
includes the model's algae polygon, photo hash, positive leaf coordinates,
and negative coordinates. Additional KVST2388 wood tips guard against false
positives from the smaller-leaf recovery. Check both regression modules with:

```powershell
$env:MANGROVISION_TEST_IMAGE_DIR = 'C:\path\to\original\photos'
python -m unittest canopy_detection.tests.test_seedling_ground_artifacts canopy_detection.tests.test_small_leaf_recovery -v
```

These checks validate specific examples, not overall precision or recall.
The model weights are unchanged; `canopy_ground_artifact_rejected_count`
records the new canopy exclusions (`ai_canopy_ground_artifact_rejected_count`
in the analysis API). Restart the backend and process the photo again to apply
the changes; saved analysis images retain their original results.

### Broader wood/debris review (ITBH, JTZS, XAJI, DAOR)

Every supplemental seedling candidate now needs compact local leaf evidence,
including bright candidates previously able to bypass the ground check. Weak
leaf-colored highlights are checked for elongated wood and nearby linear bright
material. Pale leaves can qualify through connected green anchors; rounded leaf
clusters attached to stems remain eligible. Rejection reasons are recorded in
`seedling_ground_artifact_rejected_reasons`.

The four original photos were run through the model at confidence 0.80, then
their exact cached canopy masks were used to replay the updated supplement:

| Photo | Previous seedling markers | Updated seedling markers |
| --- | ---: | ---: |
| ITBH6710.JPG | 281 | 251 |
| JTZS4533.JPG | 85 | 76 |
| XAJI3412.JPG | 547 | 501 |
| DAOR9086.JPG | 1 | 0 |

The reviewed isolated wood/debris detections were removed. Mature-canopy masks
were unchanged. The checked real seedlings, including pale leaves in JTZS,
remain eligible; their overlapping danger buffers can still cover much of the
right side of that image. All 23 tests across the two regression modules above
passed with the original-photo fixtures enabled. Counts and reviewed examples
are not overall precision or recall measurements. The model weights were not
retrained, and saved analyses must be reprocessed to show the changes.

### Canopy branch: long wood predictions

The far-right wood in LUXA2941 was confirmed in the mature-canopy mask, separate
from the seedling supplement. Its model component measured approximately 1.34 m
long by 0.17 m wide, with principal-axis aspect 8.6. Only 3.1% of its pixels
provided compact leaf evidence, but that was sufficient for the previous gate.

The small-canopy gate now rejects predictions when all of these hold:

- Principal-axis aspect is at least 5.
- Rotated length is at least 0.60 m and width is at most 0.25 m.
- Compact leaf support occupies less than 10% of the prediction.

This runs before canopy merging and danger buffering, within the existing small
component limit of 1.2 square metres. Edge fragments retain their existing
handling. Wider real crowns, short leaves, and narrow branches with sufficient
compact leaf support remain eligible. Rejection metadata identifies
`elongated_wood_prediction` separately from `insufficient_leaf_support`.

Full original-photo model inference was captured for LUXA2941 and WAJG6300.
Replaying the updated validation and downstream canopy/seedling stages on those
cached predictions removed the LUXA wood: 2,863 pixels before canopy expansion,
and 2,967 in the final canopy mask. All other LUXA canopy pixels were unchanged;
the seedling count stayed at 297. WAJG's final canopy mask was unchanged, with
483 seedling markers. Cached validation masks for ITBH, XAJI, JTZS, and DAOR
were also unchanged. The final overlay was reviewed locally; nearby seedling
buffers still remain and are not evidence that the removed wood is canopy.

All 27 tests in `test_small_leaf_recovery` and `test_seedling_ground_artifacts`
passed with original-photo fixtures enabled. The new photo fixture records the
LUXA model polygon and two real-canopy controls; synthetic checks cover rotation,
scale, short leaves, and leaf-covered narrow branches. This validates the reviewed
wood case, not all wood/algae false positives or overall model accuracy.

Local review artifacts: `tmp/canopy_wood_review/LUXA_wood_before_after.jpg` and
`tmp/canopy_wood_review/verification.json`. Restart/reload the backend and process
the photo again to update its analysis; saved summaries keep their old masks.

## Benchmark

Use the same image set and annotate seedling centers plus optional `area_m2` values:

```powershell
python canopy_detection/evaluate_seedling_benchmark.py `
  --manifest path/to/seedling_benchmark.jsonl `
  --output train_outputs/seedling_benchmark.json
```

The benchmark reports seedling precision, recall, F1, and false positives per square metre for legacy and hybrid modes. Mature-canopy regression should be checked from the same run using canopy counts and areas in the returned detector metadata.
