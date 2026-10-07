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
street; the actual GPS coordinate is retained as its own marker without drawing
a shortcut to the provider's street start.

Navigation retains the actual phone GPS origin and exact selected point as
separate markers. Mapped road sections are drawn in blue, followed by an orange
dashed guide from the site entrance to the exact selected point. The dashed
section shows direction, not a surveyed walking lane; participants follow marked
planting lanes. Remote GPS-to-road connections and gaps between mapped road
sections are not drawn across ponds or open water. `segments` keeps road and
guidance sections separate. `navigation_path` is a compatibility list of their
coordinates, not a surveyed continuous path. For a destination outside the known
site footprint, a final guide is limited to the provider's 20 m endpoint tolerance.
Navigation requires a fresh phone location and never substitutes an organization
base or access-road start when GPS is denied.

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
On arrival inside the physical site, the road overlay is replaced by the orange
dashed guide from the latest GPS fix to the exact point. This guide updates with
each fix and disappears when origin and target coordinates coincide.
The initial view includes both GPS origin and destination; subsequent fixes do
not reset a participant's pan or zoom. When GPS uncertainty exceeds the point's
distance, the panel tells participants to use the marked point. Clearing
navigation, signing out, or leaving the field page stops the location watch.
Errors appear in the point action sheet so participants can retry there.

When testing from another location, Google can return no walking route to the
public-road junction or fail to connect closely enough to the mapped road.
The endpoint returns `partial_route` with the blue mapped access road and the
local orange dashed guide from the entrance to the point.
The panel visibly states that road directions from the participant's location
are unavailable; trip distance and duration remain empty. Provider outages and
missing configuration use the same behavior. For an unmapped site,
`point_guidance` retains the point marker without drawing a line. The browser
rebuilds local guidance from the known entrance and target, filtering remote
shortcuts from older responses. Invalid requests still fail.
The **Open Google Maps road directions** action requests walking directions from
the latest GPS to the exact point via the public-road junction and site entrance.
Once onsite, it omits those waypoints to avoid backtracking. The link updates
with each GPS fix and opens only after an explicit tap. Google Maps support for
waypoints varies by product; the in-app traced entrance road remains available.

To support a different physical site, add its mapped access road and footprint
to the JSON file, add a routing regression, and restart the API. Do not enlarge
the local GPS snap tolerance to bypass a missing mapped road.

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
