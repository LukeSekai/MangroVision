# Erosion availability regression — October 17, 2026

The approved rule is that erosion temporarily blocks assignment and planting.
It must not erase otherwise valid planned locations. Removing the active zone
returns retained locations to their underlying planting/assignment status.

The current analysis workflow instead treated erosion as a permanent exclusion
in GPS candidate checks, projected masks, final GPS checks and visualization
reconciliation. It then forced every remaining coordinate to Available and the
unavailable count to zero. This explains the empty sections in recent results.

## Restored behavior

- Eroded points participate in normal grid generation, spacing, previews,
  coordinates, exports and the save cache.
- Final GPS positions determine the temporary erosion flags. The existing
  renderer colors these points orange. Total planned points include them;
  the safe/available count excludes them.
- Save continues storing the underlying `planned` status. Existing PostGIS
  reads and assignment/field-write guards consult active erosion polygons.
  No new schema or permanently stored unavailable status was introduced.
- Removing a zone clears temporary unavailability for retained points.
- Forbidden structures, canopy danger buffers, map boundaries, spacing,
  repeated-image protection and saved-point deduplication remain enforced.

## Verification

**59 backend tests and 75 frontend tests passed.** Seven new regressions run
the actual analysis/hexagon pipeline with isolated synthetic imagery and mocked
database, storage and external-map I/O. A SQLite-only save check verifies that
the retained points are saved as Planned, remain blocked by the assignment
erosion gate, and become eligible when that erosion check clears. A separate
test verifies that field completion is rejected while erosion is active.

The previous and fixed workflows were also compared on the same synthetic
image, GPS transform and species settings:

| Case | Previous retained | Fixed retained | Fixed available | Fixed unavailable |
| --- | ---: | ---: | ---: | ---: |
| No erosion | 52 | 52 | 52 | 0 |
| Lower half eroded | 24 | 52 | 28 | 24 |
| Entire image eroded | 0 | 52 | 0 | 52 |

These controlled counts demonstrate the regression; they are not counts for
the user's saved Analysis 39. Tests also cover overlapping permanent forbidden
areas, real canopy danger masks, renderer/map/export agreement, point retention
after zone removal, and the existing repeat-image and saved-point protections.

Local review: [before/after gallery](../tmp/eroded_analysis_fix_20261017/index.html),
with `comparison.json`, `regression_tests.xml` and `final_focused_tests.xml`.

## Existing saved results

No live analysis, zone, planting point, assignment or field record was changed.
This fixes generation for new analyses after the updated backend is started.
Older results whose points were already omitted are not rebuilt automatically.
Reanalyzing an already-saved source still follows the approved repeat-image rule
and cannot silently add or replace its planting records; restoring missing
points in those older records requires a separately reviewed repair.

No backend restart, push or deployment was performed during this fix.
