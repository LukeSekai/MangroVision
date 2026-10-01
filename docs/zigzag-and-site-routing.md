# Zigzag assignments and white-road access

Assignment selection uses the **locations** of points to form zigzag strips.
`zigzag_assignment_points` and the matching frontend `zigzagAssignmentPoints`
pair adjacent geographic latitude levels and alternate the selected level across
longitude columns. On the staggered hexagonal grid this selects a W-shaped strip;
on a rectangular grid it selects two complementary W-shaped strips. The selected
strips are divided into balanced participant shares. Survey grids stay separate,
and coordinates are never moved. Missing grid columns retain their parity.
Short strips at clipped boundaries continue into the next strip to keep counts
balanced; these edge shares can contain more than one zigzag segment.

This differs from reversing the visit order across whole rows, which leaves each
participant owning a block of points. The field app's visiting and routing logic
is separate and unchanged; its action is simply labelled **Next point**.
Completed, skipped and eroded points remain excluded from that action.

New assignments use the corrected selection. For an existing, entirely pending
assignment, `scripts/repair_zigzag_assignment.py --assignment-id ID --plot`
previews the corrected ownership. `--apply` locks and rechecks the assignment,
saves the previous allocation under `validation_outputs`, and updates only
participant ownership while preserving each participant's count, all coordinates,
sequence numbers, sessions and statuses. Any planting history prevents the repair.

## Entrance routing

`MangroVision_New/api/data/site_access_routes.json` stores the white-road trace
identified in the user's screenshots. It was matched against the application's
Google satellite tiles and `MAP/FINAL` georeferenced orthophoto. Coordinates use
`[latitude, longitude]`. The physical-site footprint includes its planting
foreshore and is shared by organization plots within the park; it does not
depend on an organization name or numeric database ID.

For destinations in that footprint, Google Routes is asked for walking
alternatives to the public-road junction. The shortest returned valid route by
distance is followed by the traced white access road to the entrance at
`10.78102952, 122.62457728`. Distance and time include the access-road leg. A
person already on the access road joins its nearest segment instead of being
sent back to the junction. Snap tolerances are 12 m on the access road, 20 m at
the provider's destination and 30 m at its origin.

The blue walking route ends at the entrance. The selected planting point remains
marked, with its remaining straight-line distance explicitly identified.
Internal walking lanes are not mapped: the app instructs participants to follow
the LGU-marked lanes and planting order. It does not connect the gate to a point
with an invented straight walking route. A participant already inside the site
sees the point bearing/distance and planting instruction without being routed
back outside. Routing errors are displayed instead of drawing a direct fallback
across ponds. Navigation requests fresh phone location before using the
organization's configured base if location is unavailable.

To support a different physical site, add its mapped access road and footprint
to the JSON file, add a routing regression, and restart the API. Do not enlarge
the local GPS snap tolerance to bypass a missing mapped road.

## Spatial allocation verification (2026-09-13)

- 45 Python checks passed: geometric shares, pending-assignment repair guards,
  participant authorization and unchanged white-road routing.
- 6 JavaScript assignment checks passed, including the actual location sets for
  both rectangular and staggered hex grids.
- PostgreSQL integration passed using temporary tables and real participant
  sessions; each session received a distinct W-shaped set with balanced counts.
- Production frontend build passed; existing bundle warnings remain.
- INHS assignment 19 was previewed and repaired using its real 100 coordinates.
  91 points changed participant ownership; all 10 shares still contain 10 points.
  A recovery snapshot was saved, and a subsequent read-only check reported zero
  remaining differences from the corrected allocation.
- The restarted API and frontend returned HTTP 200; the running API exposes the
  updated spatial allocation endpoint documentation.

## Earlier routing verification (2026-09-12)

- 42 Python checks passed: ordering, routing, organization authorization and
  shared-link origin security.
- 4 JavaScript assignment checks passed.
- PostgreSQL integration passed using temporary tables: assignment, equal shares,
  ten simultaneous device sessions, eleventh-device rejection and slot recovery.
- Production frontend build passed (existing bundle-size warning remains).
- All 100 existing INHS points matched the white-road entrance.
- Live provider check from `10.7830, 122.6185` to an INHS point returned a 758 m
  route to the entrance. Its overlay was inspected against the satellite image
  in `validation_outputs/white-road-route-check.jpg`.
- Readiness and routing returned HTTP 200 through API port 8000 and frontend
  port 5173 after restarting the stopped API server.
- Targeted ESLint still reports existing React effect/memoization issues in
  FieldApp and PlanterManagement. Interactive phone/browser verification was
  unavailable because no browser was connected.

Google reference: [alternative routes](https://developers.google.com/maps/documentation/routes/alternative-routes).
