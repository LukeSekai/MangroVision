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

If evidence is still insufficient, the existing approximate GPS footprint is
retained. A trapezoid is never synthesized solely to resemble a screenshot.

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
