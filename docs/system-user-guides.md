# MangroVision system guide

Reviewed against the current FastAPI and React application on **October 22, 2026**.

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
3. **Planting Map -> Zone Editor - review sites and hazards.** Project sites
   are formed automatically from assigned planting points. Use Zone Editor to
   draw eroded and warning areas inside the drone map. Eroded zones can cover
   planted or dead seedlings. In Eroded Zones, **Edit** opens the saved boundary
   as a draft: drag its handles, or focus a handle and use arrow keys. Save asks
   **Save eroded zone?** when recorded seedlings overlap. Cancel keeps the saved
   state unchanged. Records, conditions, causes of death and historical totals
   are preserved; monitoring and condition updates remain available. Erosion
   blocks assignment, new planting and replanting. Current boundaries determine
   restrictions; a location covered by another eroded zone stays blocked.
   Warnings record hazards without blocking assignment. Site
   **Release area** and **Delete site** show eligibility before confirmation.
   Release preserves living seedlings and all earlier planting/death history.
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
   supports CSV, GPX, KMZ and GeoJSON. Choose a format first. In the dialog,
   select **All points**, **Project site**, **Species** or **Saved analysis**.
   The last three reveal a dropdown; choose its value to show those points on
   the map preview. Review the locations, count and filename, then use
   **Download [format]**. Pan/zoom or **Fit selected points** helps inspect the
   locations and does not change the exported points. Switching scope clears
   the earlier choice; downloading stays disabled until points are previewed.
   Preview and export dots use red for dead, orange for eroded, yellow for
   planted, green for Bungalon and magenta for Rhizophora. Color priority is
   **Dead → Eroded → Planted → Species**: dead points inside erosion stay red;
   planted points inside erosion become orange. Other/unrecorded species use
   gray when no status color applies. Saved species and erosion fields remain.
   Hover a point for species, status and erosion details; check the legend.
   Site scope uses
   current ownership, so released locations leave the former site's export.
   **No current project site** and **Not recorded** species let staff select
   unowned or legacy points without guessing. **Other recorded species** groups
   other saved species values, preserving them in files. Unavailable points stay included
   and marked. Map layer switches do not change the export selection. Files
   include current site, source analysis, species and availability fields.
   **KMZ** bundles these solid dot colors in one file for Google Earth.
   GPX is intended for GPS navigation, using short stable IDs and full point/site
   details in comments and descriptions; flag colors depend on the receiving app.
   GeoJSON includes color fields; CSV includes marker color and category as data.
   Filenames use the selected site, analysis or species name and Philippine date.
   Point names include analysis display number, point number and stable point ID.
   KMZ uses small dots with hidden labels;
   selecting a dot shows its name and details in Google Earth.
6. **Scheduling and Assign Points - allocate approved work.** Approve the
   tree-planting activity and open **Assign points**. Choose its organization
   and activity, then click or brush available locations. **Move / click** selects
   individual points. For **Brush select** or **Brush erase**, hold the left
   mouse button and drag (press and drag on touch). Release to stop brushing;
   hovering never changes the selection. Escape returns to **Move / click**.
   Leaving the map, losing focus or switching tools ends the active stroke.
   Markers use the same
   zoom-based sizes as the main map: Bungalon green and Rhizophora magenta,
   with selected points purple. Assigning the selected
   points forms the project site and links the activity automatically. There
   is no site creation step or new/existing-site choice. Planters receive the
   selected work, while survival and monitoring begin from actual planting.
   LGU uses the same tool with its permanent account and no timed booking.
   Existing linked activities keep their saved area and history. Activity
   planning details are protected after planting starts; completed activities
   remain read-only.
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
   The shared navigation under the heading switches between **Monitoring visits**
   and **LGU Sites**. Both pages use the same **Show map** action.
9. **Monitoring map ? release locations.** Choose the organization scope and,
   optionally, Show dead plants only. Select confirmed dead locations using
   Move / click, Brush select, or Brush erase; selected points turn purple.
   Escape returns to Move / click. Choose **Release from project site** to review
   and confirm up to 500 locations. Their planting cells leave the old boundary
   and the locations return to green Planned. The original site's planting
   counts, deaths, visits, and survival results remain in history. Another
   organization can use Assign Points on the released land to form its site and
   continue the normal planting process. Erosion restrictions still apply.
   Changing the organization clears the selection. Shared risk-zone layers are
   hidden in a single-organization view. Mortality warning zones mark sites with at least 30% recorded deaths
   across all causes. Review field conditions before planning more planting.
   **Monitoring → LGU Sites** shows LGU's permanent planter credentials and
   separate site results. Each site follows its own planting-based monitoring
   dates. Use **Record monitoring** when due and **History** to review visits.
   The combined LGU survival uses total surviving and planted seedlings.
   When opening a map from Monitoring, LGU Sites, Dashboard, Organizations,
   or Scheduling, use the **Back to [section]** button at the upper left to return
   to that page with its current filters and tab. Scheduling provides map links
   beside an activity's saved planting area in both List and calendar details.
   Direct visits to the Monitoring map return to Monitoring; direct Planting Map
   visits use the sidebar for navigation.
10. **Dashboard — interpret progress and download reports.** Set From, To and
    Project site, then use Overview, Planting Work, Seedling Health or Project
    Sites. Project Sites compares the responsible organization, current mapped
    locations by status, and mapped warnings for each site. It uses only the
    Project site filter and shows a **Current snapshot**, across all planting
    dates. Released locations leave the old site's counts; released sites retain
    a Monitoring link without a map link when their boundary is gone. Unclaimed
    locations remain on Planting Map. Warning categories can overlap, but each
    category counts a location once. Eroded locations appear separately from
    warning zones; mortality warnings still use recorded planting history.
    **View planting history** opens Seedling Health with your site and reporting
    dates retained. **View map** and **Monitoring** open the site's existing
    workspaces. Image area and canopy measurements are in **Analyze Image**.
    Read the scope on each chart: period-based activity/history and current
    location status are different. Current location status covers all dates for
    the selected site. Current Planted and historical planting can differ after
    deaths or releases for replanting. Follow-up sites link to their map location
    or responsible organization's Monitoring view. Seedling Health selects
    planting batches by planting date and retains original deaths after
    site release or later planting. Review historical batch survival and current
    map counts separately. Released points become Planned until assigned under
    a newly claimed site. Its monthly
    deaths chart shows new deaths recorded in each Philippine calendar month
    within the selected dates, with a matching data table. Select the year in Planting
    goals to review targets. Download Report and Download Monitoring Report open
    scope/preview flows before download.
11. **Activity Logs and Account.** Use Refresh activity and Load more activity to
    review recorded actions and their actors. Inspect the relevant screen for the
    current record. The clock and profile are at the upper right beside Help / Guide
    and notifications. The clock uses Philippine time. Select the profile to open
    Account settings; select the sidebar's MangroVision logo to open Planting Map.
    Account settings changes a staff password through current
    password and email-code verification. The new password needs at least 12
    characters. A successful change ends the account's staff sessions.

## Review before submitting changes

Important actions use the shared confirmation dialog. **Cancel** or **Back**
returns without submitting; the current selection or form stays available.

| Action | Review and final button |
| --- | --- |
| Assign Points | Check organization, activity and point count; **Assign points**. |
| Organization monitoring | Check organization, date, located/unlocated deaths, cause and health; **Save visit**. |
| LGU site monitoring | Check site, date, selected deaths, cause and health; **Save monitoring**. |
| Identify/correct death locations | Check reported/located totals and correction effect; **Save locations**. |
| Annual goals | Check year and both targets, including any unset values; **Save goals**. |
| Change activity to Completed/Cancelled | Review the consequence, then **Mark completed** or **Cancel activity**. |
| Delete an eligible unused appointment | Check the appointment and organization; **Delete appointment**. |
| Replace/revoke an activity code | Check activity and the effect on current logins; **Replace code** or **Revoke code**. |
| Switch a participant's active browser | Confirm ending its other login; **Switch here**. |

Monitoring reviews use the existing dialog and **Back** keeps observations and
selected deaths. Failed saves return to the form with entries retained. Existing
confirmations for erosion, analysis deletion, release/site deletion, planting,
skipping, schedule approval and sign-out remain in place. Ordinary filters,
map navigation and report previews keep their current direct controls.

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
planted locations. Unavailable eroded locations cannot be marked planted. If a
selected batch contains an eroded location at submission, the whole planting
submission is blocked; remove unavailable locations and review the selection
before trying again. Existing saved plantings stay unchanged. Planting Map's
available-location count excludes erosion; analyzed plantable area remains the
saved image measurement. Check
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
