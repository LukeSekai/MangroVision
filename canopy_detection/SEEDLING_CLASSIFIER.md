# Seedling classifier workflow

The runtime detector keeps mature Detectree2 crowns separate from small-seedling candidates.  Hybrid mode is enabled by default, but it falls back to the existing legacy color supplement until a trained checkpoint is available.

## Dataset

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

Run the model-independent regression cases with:

```powershell
python -m unittest discover -s canopy_detection/tests -v
```

## Benchmark

Use the same image set and annotate seedling centers plus optional `area_m2` values:

```powershell
python canopy_detection/evaluate_seedling_benchmark.py `
  --manifest path/to/seedling_benchmark.jsonl `
  --output train_outputs/seedling_benchmark.json
```

The benchmark reports seedling precision, recall, F1, and false positives per square metre for legacy and hybrid modes. Mature-canopy regression should be checked from the same run using canopy counts and areas in the returned detector metadata.
