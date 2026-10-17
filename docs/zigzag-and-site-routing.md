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

`MangroVision_New/api/data/site_access_routes.json` stores the fixed white-road
and bridge approach shared by all organization plots within the Leganes park.
Coordinates use `[latitude, longitude]`; routing depends on the physical site
footprint, not an organization name or database ID. The road was traced against
the app satellite tiles on 2026-09-12. The bridge was extended on 2026-10-17 from
the user's marked screenshot and `MAP/FINAL` georeferenced orthophoto.

The old road endpoint at `10.78102952, 122.62457728` is now `bridge_start`.
`access_path` continues through the blue-roof entrance, along the white bridge,
and along its shoreline spur to the fixed exit at
`10.77999592, 122.625672`. Every destination in the physical site uses this same
approach. Google Routes supplies only the public-road leg to `access_path[0]`;
the complete traced bridge is appended independently of the target. Distance
and time cover the mapped approach only.

`site_area` identifies destinations within the park. The separate
`planting_area` traces its visible foreshore and excludes the wooded bridge
approach. A GPS fix in the park alone cannot trigger a diagonal shortcut through
the trees. A person on the mapped approach joins its nearest segment within
12 m and follows its remaining length. This takes precedence over onsite
classification until the final 12 m at the shoreline exit, even when phone GPS
accuracy overlaps the foreshore. Provider endpoints must join within 20 m of
the public-road junction; these snap tolerances have not been enlarged.

The map shows the mapped approach in blue and the exact target as a labelled
marker. A local orange direction line starts at the shoreline exit, or at the
latest GPS position once in the planting area. It is omitted when samples at
intervals of at most 3 m leave the traced foreshore, so it does not suggest a
shortcut through trees. Participants still follow marked planting lanes; the
image traces and direction lines are not a ground survey of walkable lanes.
The actual origin and target coordinates are never moved.

Before navigation, the point nearest the physical site's bridge exit has a
**Start at Point #…** map overlay with **Navigate to this point**. **Next point**
uses that same starting point. The field-workspace response includes the traced
entrance and site footprint, so this recommendation needs no GPS permission or
Google Routes request. It uses geographic distance from the planting-area
entrance, independently of point numbers, list order or the phone's location.
Completed, skipped, eroded/unavailable and invalid-coordinate points are excluded.
Sites without mapped entrance data keep their existing manual point navigation.

Starting navigation draws an orange dashed continuation through the remaining
assigned points. The chosen start stays first and the farthest remaining point
from the entrance is the finish. The entire walk is planned together: up to 14
points use the shortest open path between those ends; larger assignments use
nearest-neighbour initialization and bounded 2-opt improvements. Optimizing the
whole path avoids the greedy short hop that leaves nearby points behind and
forces a return trip. The screenshot regression follows
**95 → 100 → 99 → 94 → 88 → 82 → 76 → 70 → 58 → 46** using its relative geometry.
Distances are rounded to millimetres and assignment IDs make ties deterministic.
Navigation changes no ownership, coordinates, point status or planting records.
Connections are drawn only within the known physical site and foreshore, using
the same checks that prevent shortcuts through the wooded bridge approach.

**Next point** starts navigation directly when no route is active. During
navigation it advances to the next eligible point without opening the action
sheet or asking for another Navigate tap. It does not mark a point as planted
or skipped. The same mapped bridge approach is reused for points in the same
physical site; a different site or departure from onsite-only guidance requires
a new route request. A new GPS fix is required when advancing. If that fix
fails, the current route stays visible for retry. The orange continuation is
rebuilt from its saved queue when assignments refresh: completed or unavailable
future points are omitted while the planned order remains stable. The orange
continuation and Next use this same order. Advancing saves
the remaining queue in its updated order so unvisited points are retained and
passed points do not reappear. Recording a single point as planted or skipped keeps navigation
active. The end of the route disables Next rather than cycling indefinitely.

The foreshore's boundary tolerance is 3 m. Onsite GPS classification and local
guidance allow `max(3, min(accuracy, 20))` metres to accommodate phone drift,
including the known NASUGBAN point #198 boundary case. A remote or poor GPS fix
cannot bypass the bridge. Road sections already passed are trimmed as the
phone moves; GPS updates do not reset the participant's pan or zoom. Markers
for the public-road start are shown only while that start remains on the route.

The large **Directions to planting point** panel, trip statistics, compass
bearing, and external Google Maps action have been removed from the planter UI.
A small hint says to follow the white bridge and marked planting lanes. Its
close button dismisses it immediately, or it closes automatically after seven
seconds. Advancing to another point does not reopen a dismissed hint.
Dismissing the hint leaves the route and GPS watch active. A compact
point label and **Stop navigation** button remain. GPS interruption messages
stay visible independently of the optional hint. Stopping navigation, signing
out, or leaving the field page clears the location watch.

Navigation requires a fresh phone location and never substitutes an organization
base or road start when permission is denied. If Google is unavailable or has
no route, `partial_route` retains the full known road/bridge approach and target
without inventing a remote GPS connection. The hint reports that road directions
to the site are unavailable; no total journey estimate is shown. Unmapped sites
retain their own provider route or point marker and are never forced through
Leganes. Invalid requests still fail.

To support another physical site, add its traced access path and footprint to
the JSON file, add routing regressions, and restart the API. Restart the API
after editing this file because its route data is cached per process.

The 2026-10-17 regression checks cover different assigned targets using the same
bridge, live positions on the bridge, wooded park positions, foreshore GPS
uncertainty, provider outages, omission of direction lines through trees, and
manual/automatic hint dismissal while navigation stays active. The traced
bridge overlay was visually checked against the orthophoto in
`validation_outputs/fixed-bridge-route-check.jpg`.

## Google Routes configuration for laptop testing

Set `GOOGLE_ROUTES_API_KEY` in the repository-root private `.env`. Do not put
it in a `VITE_` variable or commit it. A hosted environment variable takes
priority. When the running process has no key, the API also checks the private
`.env` at request time, so a newly entered key can be used without another
restart after this code update is running.

Google Routes attempts are capped at 100 per UTC day and 1,000 per UTC month
by default. Configure `GOOGLE_ROUTES_DAILY_LIMIT` and
`GOOGLE_ROUTES_MONTHLY_LIMIT` on the backend to change those limits; zero stops
all Google calls. Counts persist across API restarts in the ignored
`MangroVision_New/run_logs/google_routes_usage.sqlite3`. Attempts are counted
before Google is contacted, including failed calls. The file stores only a
credential fingerprint and counts, with no key, coordinates, or route content.
Calls stop if the counter cannot be checked. Simultaneous requests share the
same counter transaction.

These are local request limits, not a guarantee about a Google bill: other
servers and projects may use the same billing account. Review the account's
actual usage, restrict the key to Routes API, and configure Google Cloud
quotas for each project using the key. Budget alerts do not stop charges.
See [Google's pricing](https://developers.google.com/maps/billing-and-pricing/pricing)
and [cost controls](https://developers.google.com/maps/billing-and-pricing/manage-costs).

On 2026-10-08, a walking-route check using the private `.env` credential
returned 372 road coordinates after billing was activated. The laptop API
was ready, while the saved Vercel tunnel was unreachable. For remote testing,
start `MangroVision_New/start_testing.py --deploy` with the workspace's venv
as described in [user-testing.md](user-testing.md); local development alone
does not reconnect Vercel to the laptop.

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
