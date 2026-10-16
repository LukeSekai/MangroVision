# Hexagon preview projection — October 17, 2026

Analysis 35 (DJI_0809.JPG) shows merged light green tiles, while Analysis 39
(DJI_0308.JPG) shows separate tiles. Both use Bungalon, a 1 m target planting
distance and a 0.57735 m spacing-buffer circumradius. Read-only checks found
1 m nearest-neighbor distances in both retained map point sets.

The difference is a display scale mismatch. Analysis 35's measured map
alignment is approximately 0.850 cm per image pixel, whereas its camera GSD
is 0.726 cm per pixel. Point centers follow the measured alignment; the old
buffer radii still divide by the camera GSD. This makes the visible buffers
about 17% oversized, enough to erase the intended gaps between tiles. Analysis
39's measured and camera scales are effectively equal, so it retains the gaps.

## Change

The final renderer now projects the six metric spacing-buffer vertices through
the same GPS-to-image conversion used for the planting points. It uses the
same latitude reference and flat-top metric lattice as the existing point grid,
including rotation and perspective from the measured alignment.

This changes only the spacing shapes in the result image. Each final center,
evaluated planting core, GPS coordinate, erosion flag and authoritative buffer
radius remains unchanged. The correction runs after the existing forbidden,
danger, coverage and spacing checks. These display polygons never enter the
save cache, assignment logic or waypoint exports. Canopy/danger layers retain
their existing rendering priority. An invalid display projection retains the
original marker rather than dropping or relocating a valid planting point.

## Verification

A display-only replay used the retained pixel/GPS pairs from both saved
analyses. The saved similarity transform was reconstructed from these pairs;
this replay is not a new AI analysis or a modification of the saved assets.

| Saved analysis | Retained points | Connected tile shapes before | After |
| --- | ---: | ---: | ---: |
| 35 | 275 | 1 merged patch | 275 separate tiles |
| 39 | 693 | 693 | 693 |

The local [before/after geometry review](../tmp/hex_style_review_20261017/projection_fix.html)
isolates the spacing shapes so their gaps are easy to compare.

New regressions cover the measured scale mismatch, rotated/projective vertex
projection, invalid projection fallback, overlapping versus separated tile
shapes, and the complete analysis pipeline. The pipeline comparison includes
vegetation, a forbidden zone, partial erosion and a 1.17x alignment scale. It
checks that coordinates, exported waypoints, GeoJSON features, safe/blocked
counts, authoritative cores and danger masks do not change.

**90 backend regression tests passed**, with zero failures, errors or skips.
The run also covers the existing preflight/perspective alignment, erosion,
forbidden zones, saved-point spacing, repeat-image, deletion, asset-cache and
batched-save regressions. Test report:
`tmp/hex_style_review_20261017/regression_tests.xml`.

## Activation and reanalysis

Restart the backend to load the updated renderer before running a new analysis.
Existing JPEGs are retained snapshots and are not rewritten. The user plans to
delete the old analysis first and then analyze its source image again. Deletion
remains blocked for analyses with assignments or field history, and the existing
repeat-image protection continues to prevent duplicate planting records.

No live analysis, planting point, assignment, zone or private storage asset was
changed during implementation or verification. No push, deployment, service
restart or schema change was performed.
