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

## GPS-to-point navigation

`MangroVision_New/api/data/site_access_routes.json` stores the white-road trace
identified in the user's screenshots. It was matched against the application's
Google satellite tiles and `MAP/FINAL` georeferenced orthophoto. Coordinates use
`[latitude, longitude]`. The physical-site footprint includes its planting
foreshore and is shared by organization plots within the park; it does not
depend on an organization name or numeric database ID.

For destinations in that footprint, Google Routes is asked for walking
alternatives to the public-road junction. The shortest returned valid route by
distance is followed by the traced white access road to the entrance at
`10.78102952, 122.62457728`. Road distance and time include the access-road leg. A
person already on the access road joins its nearest segment instead of being
sent back to the junction. Snap tolerances are 12 m on the access road and 20 m
at the provider's access destination. A phone at home may be away from a mapped
street; the actual GPS coordinate is retained, with a dashed connection to the
provider's street start instead of rejecting a valid road route.

Every navigation response starts at the actual phone GPS coordinate and ends at
the exact selected planting point. `navigation_path` contains that complete
geometry, while `segments` distinguishes solid-blue mapped roads (`road`) from
orange dashed direct connections (`guidance`). The final entrance-to-point
segment is explicitly drawn as guidance: internal walking lanes are not mapped,
so participants must follow LGU-marked lanes. Dashed connections are not verified
walking paths and have no walking ETA. `polyline` retains road-only geometry for
older consumers. A participant already inside the site sees only GPS-to-point
guidance, without being routed back outside. Navigation requires a fresh phone
location. If GPS is denied or unavailable, it explains how to enable it; it
never substitutes the organization's configured base or the access-road start.

The traced physical footprint accepts planting destinations up to 3 m from its
boundary, including NASUGBAN point #198, which lies about 0.6 m outside the trace.
This tolerance does not change project-site ownership or point coordinates.
For the origin, phone GPS accuracy is sent with the request. A position near the
same footprint uses onsite guidance within `max(3, min(accuracy, 20))` metres;
a poor GPS fix cannot skip the entrance route from farther away. The road snap
tolerances above remain unchanged.

During navigation the phone watches GPS and updates the route's start,
straight-line distance and compass bearing to the selected point. Mapped road
sections already passed are trimmed when the GPS fix is within 12 m of the road.
On arrival inside the physical site, navigation switches to direct point guidance.
The initial view includes both GPS origin and destination; subsequent fixes do
not reset a participant's pan or zoom. When GPS uncertainty exceeds the point's
distance, the panel tells participants to use the marked point. Clearing
navigation, signing out, or leaving the field page stops the location watch.
Errors appear in the point action sheet so participants can retry there.

When testing from another location, Google can return no walking route to the
public-road junction or fail to connect closely enough to the mapped road.
The endpoint returns `partial_route`: a dashed GPS-to-road-start connection,
the blue mapped access road, and a dashed entrance-to-point connection. Road
directions are identified as unavailable; trip distance and duration remain
empty. Provider outages and missing routing configuration use the same guidance.
For a destination without a mapped access road, `point_guidance` draws direct
GPS-to-point guidance without calling it a mapped path. Invalid requests still
fail. The **Open Google Maps to point** action uses the latest GPS coordinate as
origin and the exact planting point as destination, rather than the gate or road
start. The link updates with each GPS fix and opens only after an explicit tap.

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
