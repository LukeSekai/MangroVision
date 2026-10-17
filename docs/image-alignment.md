# Image-to-orthophoto perspective alignment

The footprint, raster overlay and exported planting coordinates use the same
accepted image-to-orthophoto homography. A validated projective transform keeps
its four measured corners; it is not converted into a rectangle.

If both standard SIFT attempts fail, processing now tries root-normalized SIFT
on several smoothing levels and image resolutions. Candidate landmarks must be
plausible against the GPS/heading prior (within 6 m and 35 degrees). The existing
registration validator chooses similarity, affine or perspective using spatially
distributed held-out landmarks, geometry limits and corner stability. Projective
fits are refined on their own consensus before validation. No acceptance limits
were relaxed. An accepted retry must also be within 5 m of the GPS centre.

If discrete landmarks are insufficient, a bounded photometric refinement can
correct the approximate footprint using stable image structure. If that also
fails its checks, the existing GPS estimate remains. A trapezoid is never
synthesized solely to resemble a screenshot.

## Targeted verification, 2026-10-03

Using the original `YAUW1815.JPG` and configured FINAL GeoTIFF, the standard
matcher failed (8 inliers on its wider retry). The new processing alignment
accepted a projective transform with 116 inliers, 29 held-out validation inliers,
and support across 9 image cells. Its centre was 1.50 m from the EXIF position;
corner-fit stability was 0.41 m. The 0.07 m median landmark residual measures
agreement with this orthophoto, not surveyed absolute accuracy. Creek alignment
was also reviewed visually. This is one targeted real-photo regression, not a
benchmark across flights or terrain.

Run focused tests:

```powershell
python -m unittest canopy_detection.tests.test_perspective_matching -v
python -m unittest discover -s MangroVision_New/api/tests -p test_perspective_workflow.py -v
```

Restart the backend and process the original photo again. Saved analyses and
their stored previews retain their previous alignment; this change does not
rewrite existing planting records.

## Initial location preview

The upload location check now runs the same matcher as processing and retains
the complete transform in its bounded cache. Image content, camera projection
inputs, map files and active tileset identify the cache entry. Processing reuses
that transform, including the chosen orthophoto. Both views project the same
four corners and clip the displayed boundary to GIS coverage. Partial-coverage
confirmation is evaluated against the full projected footprint before clipping.

On `YAUW1815.JPG`, the initial preview and processed-overlay footprint were
geometrically identical and the matcher ran once (116 inliers, projective).
The initial check can take longer; it runs in a worker thread and its alignment
work is reused during analysis. Re-select the image after restarting the backend
to replace a preview already held by the browser.

```powershell
python -m unittest discover -s MangroVision_New/api/tests -p test_preflight_alignment.py -v
```

## Approximate automatic refinement, 2026-10-18

`DJI_0312.JPG` failed the standard and perspective landmark matchers, leaving
the application's green border on the GPS/heading rectangle. The separately
generated comparison did not affect that production path.

After landmark and vegetation-scale matching fail, the workflow now tries a
bounded affine photometric refinement. It fits only near vegetation and bank
texture, excluding open-water glare from its fitting mask. A coarse fit seeds
two finer smoothing levels; their corners must agree within 0.75 m. Existing
fold, perspective and physical-scale checks still apply. Additional limits cap
the relative scales at 0.85–1.15, rotation change at 8 degrees, centre correction
at 5.5 m and any corner correction at 8 m. Fixed photo-side comparisons must
improve in at least four cells spanning three columns and two rows. Failed or
ambiguous fits retain the previous fallback. SIFT acceptance limits are unchanged.

This result remains **approximate**, with `projection_rebuilt=true` and
`registration_validated=false`. It does not enable the wider edge extrapolation
reserved for validated landmark matches. Correlation scores are image-agreement
measurements, not seedling accuracy, landmark inlier ratios or surveyed GPS error.

The accepted `H` feeds the upload's green polygon, result raster, planting-point
conversion and saved footprint. It is not reset to the EXIF centre after fitting.
The frontend retains the approximate-alignment label. Alignment revision is part
of both alignment and analysis cache identities. No canopy/seedling thresholds,
spacing, eroded-zone rules, assignments or existing records were changed.

Read-only review of the original photo and configured GeoTIFF:

```powershell
.\venv\Scripts\python.exe -X utf8 scripts/review_photo_alignment.py C:/Users/Lenovo-Pc/Documents/1/DJI_0312.JPG --output tmp/dji0312_alignment_fix_20261018
```

This creates a local comparison and a map image showing the corrected green
boundary alongside the previous dashed amber GPS estimate. It verifies that
upload preview, result overlay and coordinate conversion share the same fit.
It does not run vegetation inference, query organization records, or save data.
For the reviewed DJI_0312 image, fixed-region image correlation increased from
0.215 to 0.524, with eight improved regions. The corrected centre moved 5.01 m
from the metadata estimate; this is a correction distance, not an absolute
position-error measurement. The two fitted corner sets differed by at most
0.48 m. This is a targeted check of this photo, not a flight-wide benchmark.
Restart the API and reselect/reanalyse the original photo to use the change in
the app; older saved analyses keep their stored geometry.

```powershell
.\venv\Scripts\python.exe -m unittest canopy_detection.tests.test_photometric_registration -v
.\venv\Scripts\python.exe -m unittest discover -s MangroVision_New/api/tests -p test_photometric_workflow.py -v
```
