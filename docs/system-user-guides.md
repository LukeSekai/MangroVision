# MangroVision system guide

Reviewed against the current FastAPI and React application on **October 17, 2026**.

Select **Help / Guide** in the workspace header. Choose the recommended guide for
the current screen, a task guide, or **Full system workflow**. Use **Next**,
**Back**, **Finish**, **End guide**, or **Escape**. Replay a guide from the same menu.
End the guide before using the highlighted controls to perform the task.

## Staff workflow

1. **Scheduling — prepare the organization and activity.** Add a staff-entered
   schedule or review a LIKE website request. Contact the requester before
   approving the requested schedule or rescheduling and approving it. Declining
   requires a reason. Website decisions are **Pending, Approved, Declined**;
   activity progress is **Completed, Pending, Cancelled**. Approval is separate
   from progress: an approved activity stays Pending until completed. Calendar
   shows approved activities; List includes the other groups. Missed means the
   end time passed without completion or cancellation, including unfinished
   planting. It is a follow-up label, not another saved progress status.
2. **Organizations — review the saved information.** Search by name, contact,
   email or phone. The finder shows **four organizations per page**, and search
   covers the whole directory. The profile identifies the activity supplying its
   contact details and expected participants. Expected participants and registered
   members are different counts. **Project sites** includes labelled archived
   sites, with map links for current sites. **Activity codes** lets staff reveal,
   copy and hide an existing code; it does not issue or replace one. **Members**
   lists usernames, participant numbers and recorded joining/last-seen dates,
   with search and 25 members per page. Information reloads on return and
   workspace data-change updates. There is no Refresh information button;
   failed requests retain **Retry**.
3. **Planting Map → Zone Editor — prepare the area.** Choose the owner before
   drawing a project site. An organization's first site can occupy unclaimed
   land inside the drone map. Later sites must fit its area or released land
   available to other organizations. Avoid other owners' areas. Start at the
   left, drag right, trace the boundary, then check the validation before saving.
   Eroded zones retain planting locations but block assignment and planting;
   removing the zone restores eligible locations. Warning zones record hazards
   and severity without blocking assignment. Site **Release area** and **Delete
   site** show eligibility and affected locations before confirmation. Release
   preserves living seedlings and earlier planting, death and monitoring history.
4. **Analyze Image — create and review potential planting locations.** Upload an
   original geotagged drone JPEG or PNG. Check GPS, coverage and alignment.
   Review partial coverage, actual overlaps with earlier analysis boundaries,
   and repeated-image confirmations. Choose species and danger buffer, then run
   analysis. Changing screens or refreshing reconnects while the server job is
   still running. Zone and assignment changes are locked during processing.
   Review the original photo alongside the result: purple canopy, red danger
   buffer, green available hexagons, orange eroded/unavailable hexagons. Eroded
   points are retained for possible later use. Save after checking the result.
5. **Analysis history and map — review saved results.** Filter and browse history,
   open a result, and use **View area in map** to focus its boundary while keeping
   Analyze Image open. The analysis overlay is shown in the review window.
   Display numbering reflects currently saved analyses rather than deleted
   database IDs. Analyses with assigned points or field records are protected
   from deletion. Only eligible unused analyses offer Delete with confirmation.
   Use map layers and Legend to inspect points and zones. Export Saved Points
   supports CSV, GPX, KML and GeoJSON. KML uses small dots with hidden labels;
   selecting a dot shows its name and details in Google Earth.
6. **Scheduling and Assign Points — allocate approved work.** Approve the
   tree-planting activity and choose its planting area. In Assign Points, select
   the organization and approved activity, then review the available
   points and allocation before assigning. An incomplete activity can be
   rescheduled until a linked assignment point is recorded as planted. Once
   planting starts, its date, times and planting plan are protected; permitted
   contact/notes corrections and progress updates remain available. A separate
   future activity at the same site is independent. **Completed activities are
   read-only**: they cannot be edited, reopened, cancelled or deleted.
   The planting area is linked automatically from the activity or organization;
   Assign Points has no separate project-site selector.
7. **Field access — share the link and activity code privately.** Share the
   deployed website's `/field` URL. Organization code manages the selected
   planting activity's shared code. Each participant uses a personal username.
   The scheduled access window is temporarily disabled for testing; see
   [shared planter access](shared-planter-access.md). Replacing or revoking the
   code ends that activity's logins while retaining assignments and records.
   The Organizations sidebar page provides contacts, sites, existing codes and
   member information. Assign Points no longer includes the Cloudflare sharing
   panel or Activity Report button.
8. **Monitoring — record visits.** Plant checks follow actual planting dates at
   14-day intervals, with weekend rounds moved to Monday. In **Due today** or
   **Overdue**, select the organization and Record visit. Check the date and enter
   newly dead seedlings since the last visit, including **0** when none died.
   Record the cause and known dead locations; unknown locations remain in visit
   totals as unlocated deaths. Review alive count, survival, growth estimate,
   overall health and LGU actions before saving. Growth is an estimate to compare
   with field observations. Sites to Monitor keeps **five sites per page**.
   Open a site's map/details, use the clock icon for history, then close to return
   to the list. Counts and visit dates cover the site's organization.
9. **Monitoring map — review replanting.** Choose the organization scope and,
   optionally, Show dead plants only. Select dead locations using Move / click,
   Brush select or Brush erase; selected points turn purple. Escape returns to
   Move / click. Review up to 500 points per batch and check eligibility before
   approval. Approved locations enter the LGU queue and appear blue as Assigned;
   earlier planting and death records remain in history. Changing the organization
   clears the selection. Shared risk-zone layers are hidden in a single-organization
   view. Big-wave danger marks sites where at least 30% of historically planted
   locations have recorded big-wave deaths; it describes recorded losses, so check
   present conditions before planning work.
   Open **Monitoring → LGU Replanting** to review pending deaths, filter by
   project site, and record replacements after LGU staff have planted them.
   Confirm the planting date, site and species. Each confirmed replacement
   becomes yellow and has a new staff-attributed record linked to its original.
10. **Dashboard — interpret progress and download reports.** Set From, To and
    Project site, then use Overview, Planting Work, Seedling Health or Project
    Sites. Read the scope on each chart: period-based activity/history and current
    location status are different. Current location status covers all dates for
    the selected site. Current Planted and historical planting can differ after
    deaths or releases for replanting. Follow-up sites link to their map location
    or responsible organization's Monitoring view. Seedling Health selects
    planting batches by planting date and retains original deaths after
    replanting approval or replacement. Review batch survival, current map
    counts, and replanting progress separately. Awaiting review, Ready for LGU
    replanting, and Replacements planted cover all planting dates for the site.
    Ready locations also count as Assigned on the map; these totals overlap.
    Its monthly
    deaths chart shows new deaths recorded in each Philippine calendar month
    within the selected dates, with a matching data table. Select the year in Planting
    goals to review targets. Download Report and Download Monitoring Report open
    scope/preview flows before download.
11. **Activity Logs and Account.** Use Refresh activity and Load more activity to
    review recorded actions and their actors. Inspect the relevant screen for the
    current record. Account settings changes a staff password through current
    password and email-code verification. The new password needs at least 12
    characters. A successful change ends the account's staff sessions.

## Public booking and field users

**LIKE website** is available at `/landing-page`, also linked by **Open LIKE
website** in Scheduling. Visitors first provide group/contact details, then their
preferred date, start time, participants and optional notes. **Back** is at the
top left of the schedule card. The end time is automatically two hours later.
Planting requests currently allow flexible hours for testing, while requiring
the full visit to finish on the same Philippine date. The shared
`src/like/bookingPolicy.json` setting controls this in the public form, backend
and in-app guide. When operating hours are enforced, tree planting starts
between 6 AM and 3 PM and finishes by 5 PM. Tide suggestions check the full
two-hour visit; dates without a forecast need LGU review. Dates
and scheduling times use Philippine time. Sending the form creates a request
subject to LGU agreement and approval.

**Planters** open the shared field link (`/field`) during their approved activity.
Choose First time, enter the shared activity code, then choose a unique username
with 3–32 letters or numbers, allowing dots, underscores and hyphens. Returning
planter uses the current activity code and that username to restore progress.
Switch here ends another active login for that participant and keeps saved work.
No participant password, PIN or staff device-reset workflow is required.

The field Help / Guide covers assignment checks, Refresh assignments, My points,
My location and GPS accuracy, navigation, point details, **Mark as planted**,
**Skip Point**, and **Choose points to mark** for a reviewed batch of physically
planted locations. Unavailable eroded locations cannot be marked planted. Check
progress before signing out. Access ends when the approved activity window ends
or its code is revoked/replaced; saved work stays. Staff and field sign-in guide
definitions are retained, but their guide buttons remain hidden.

## Maintaining the guides

- `MangroVision_New/client/src/guides/tours.js` defines task text, routes and
  highlighted controls. Full system workflow includes all staff screens, starting
  with activities and organizations before site preparation and assignments.
- `src/guides/tourRuntime.js` finds the visible retained route, waits for rendering,
  reveals collapsed panels and cancels pending work when a guide ends.
- `src/components/GuideButton.jsx` selects the current-screen guide;
  `GuideTour.jsx` presents it using lazily loaded React Joyride 3.2.0.
- `src/components/Panel.jsx` supplies stable `data-guide-panel` / `data-guide-card`
  targets. Prefer actual visible controls over layout-dependent selectors.
- `src/guides/config.js` keeps both sign-in guide buttons disabled. Do not enable
  them incidentally when updating workspace guides.

Guides only navigate and reveal panels. They do not upload, analyze, save,
assign, approve, reveal codes, download, delete or submit records. Highlighted
controls are blocked while a tour runs. Conditional controls, such as an activity
code panel requiring an organization/activity, retain their instructions with a
prerequisite note when absent. Public booking is explained from Scheduling;
the staff tour does not navigate out of the workspace into the separate website.

Update instructions and targets together when workflows change. Keep full-map
steps centered so tooltips fit the viewport. From `MangroVision_New/client`, run:

```powershell
node --test src/guides/tours.test.js src/guides/tourRuntime.test.js src/pages/Organizations.test.js src/components/Panel.test.js src/components/ActivityAndBooking.test.js
npm run build
npx eslint src/guides/tours.js src/guides/tours.test.js src/guides/tourRuntime.test.js src/components/GuideButton.jsx src/pages/Organizations.jsx src/pages/Organizations.test.js
```

Browser checks should cover the menu, route changes with Back/Next, collapsed
panels, missing data, Finish, Escape, replay and phone layouts. Use isolated
sample data; following a guide must never create API mutations.
