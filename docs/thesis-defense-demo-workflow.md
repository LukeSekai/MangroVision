# MangroVision thesis defense and demonstration workflow

Prepared against the current FastAPI and React source on October 22, 2026.
Includes the erosion rules and confirmation reviews introduced on
`feature/eroded-zone-rules-20261022`. Verify the demonstrated build before each
rehearsal. Preparing this guide did not create demonstration accounts, alter
planting records, send messages, or deploy the application.

## 1. What the audience should understand

**Suggested opening:**

> MangroVision connects drone-image analysis, map-based planting allocation,
> participant planting records, monitoring, and reporting in one workflow.
> Staff can trace a planting location from its source analysis to its current
> project site and its recorded planting history.

Frame the problem using the evidence from your own thesis interviews and
observations. Do not claim measured time savings, model accuracy, increased
survival, or ecological impact unless your evaluated results support them.

Present the workflow in this order:

**Public request → staff scheduling → image analysis → review hazards → assign
points and form a project site → field planting → monitoring → release eligible
dead locations → assign and plant a new cycle → dashboard and reports.**

The permanent LGU organization also uses assignment and planting, with separate
site monitoring and no timed public booking requirement.

## 2. Roles and workspaces

| Role | Workspace | Responsibility |
| --- | --- | --- |
| Visitor or organization representative | LIKE website, `/landing-page` | Learn about the site and request a planting, clean-up, or field visit. |
| Authorized staff: admin, LGU, planner | Staff website, `/` | Review requests, plan locations, manage assignments and zones, monitor, and report. Individual actions enforce their existing server permissions. |
| Organization participant | Field website, `/field` | Join an activity with its shared code and a personal username; view and record assigned planting work. |
| LGU field planter | Same `/field` website | Use the permanent LGU account for LGU assignments. |

Use separate browser profiles or devices for staff and planters. Do not narrate
or display passwords, real access codes, email verification codes, private image
URLs, or private configuration on the projector.

## 3. Prepare a rehearsal pack

Use an approved isolated demo database and private storage configuration for
demonstrating writes. A Git branch alone still uses whichever database the
backend is configured to access. If only shared records are available, inspect
existing records and use prepared screenshots/video for mutations instead.

Prepare these items before the defense; this guide does not create them:

- A staff account with access to its verification mailbox, plus an existing
  signed-in staff session for a smooth main demonstration.
- A fictitious organization, for example **Demo Coastal Volunteers**, with an
  approved tree-planting activity and an expected participant count.
- Its shared activity code and at least two synthetic participant usernames,
  privately stored outside the presentation.
- LGU's existing permanent account in the isolated environment.
- An original geotagged drone image that the demo environment can analyze,
  with the required orthophoto and model checkpoint already installed.
- One saved analysis with usable Bungalon and Rhizophora locations; a prepared
  saved result is the fallback if processing takes too long.
- Separate demonstration locations for available, assigned, planted, skipped,
  and confirmed dead seedlings; include an eroded area and a warning area.
- An ordinary organization with monitoring due, an LGU site with monitoring
  due, and a saved monitoring history. Preparing these timelines belongs to
  the isolated demo data, not to changes of the laptop clock or shared records.
- A dead location eligible for release and a nearby location inside erosion.
- A downloaded KMZ, GPX, GeoJSON, CSV, and report PDF made from synthetic data.
- A short screen recording and screenshots of each critical stage, including
  the new erosion confirmation and blocked planting messages.

Choose a small planting selection for narration, such as six to ten points.
Use a prepared larger dataset only to demonstrate filters, aggregation, and
search. All example counts in this document are illustrative, not measurements
of your live database.

## 4. Runtime check before presenting

1. Follow [groupmate setup](groupmate-setup.md) if using another laptop. Check
   the separately supplied orthophoto and checkpoint, dependencies, and private
   database/storage configuration. Do not paste private settings into slides.
2. Check whether a managed development or public testing session already owns
   port 8000. Do not start duplicate servers or stop the active public tunnel
   as a rehearsal side effect. Agree on the intended demo runtime in advance.
3. For the local development runtime, the supported entry point from the repo
   root is `.\venv\Scripts\python.exe MangroVision_New/start_dev.py`.
   The usual local frontend is `http://localhost:5173`; use its `/field` and
   `/landing-page` routes for the other roles. The launcher coordinates the API
   and frontend. Use its managed stop command when intentionally ending it.
4. Verify staff sign-in, image loading, saved maps, one saved report, and the
   planter assignment view. For a phone, use the launcher-supported sharing
   arrangement and its recorded frontend address; phone localhost is the phone.
5. Verify the demonstrated build and the published deployment revision. Check
   that both the frontend and the running API include the erosion rules.
6. Charge devices; keep the laptop awake if it hosts the API. Verify the actual
   network path, mailbox access, and phone location permission in rehearsal.
   Geolocation generally needs a browser-supported secure context.

Do not promise offline operation. The hosted database, private imagery, and
external services require working connectivity in the configured deployment.

## 5. Suggested 12–15 minute main demonstration

| Time | Presenter action | Main point |
| --- | --- | --- |
| 0:00–1:00 | Introduce the problem, roles, and workflow. | One connected record trail. |
| 1:00–2:00 | Show LIKE request form and staff Scheduling with a prepared request. | A request needs staff agreement and approval. |
| 2:00–4:00 | Open Analyze Image, demonstrate location checks, then show a prepared result and map. | Detection and alignment support planning; staff review the result. |
| 4:00–5:00 | Show zones, existing seedlings under erosion, and cancel an overlapping edit. | Erosion restricts future work without rewriting history. |
| 5:00–6:30 | Assign a small available selection to the demo organization. | Selected points form the site; assignment is not planting. |
| 6:30–8:00 | Switch to `/field`; show points, navigation, planting confirmation and progress. | Actual planting is explicitly recorded. |
| 8:00–10:00 | Open a prepared due visit, death-location selection, and saved history. | Monitoring records observed outcomes and preserves their attribution. |
| 10:00–11:00 | Show eligible dead-location release, blocked erosion, and LGU Sites. | Reuse land through the existing process; keep site histories separate. |
| 11:00–13:00 | Compare Dashboard tabs, open report preview, download a prepared PDF. | Current locations and historical planting performance answer different questions. |
| 13:00–14:00 | Export scoped points; show colored KMZ and other format choices. | Operational data can be inspected and shared. |
| 14:00–15:00 | Show logs, contextual guide and account controls; close with limitations. | Actions are traceable and the system supports its users. |

Only perform the write steps above in the approved isolated environment. For a
strict ten-minute slot, show prepared analysis and monitoring results, then
demonstrate assignment, one planting, erosion cancellation, and reporting.
Keep the comprehensive feature checklist below for follow-up questions.

## 6. Full end-to-end rehearsal script

During rehearsal, demonstrate **Cancel** or **Back** before confirming one write.
Assignment reviews show the recipient, activity and count. Organization and LGU
monitoring reviews show the relevant date, deaths and health; Back preserves the
form. Death-location corrections, annual goals and activity completion or
cancellation also have a final review. Eligible appointment deletion, code
replacement/revocation and switching an active participant browser use the shared
confirmation modal. See the [confirmation audit](workflow-confirmation-audit.md).

### A. Public LIKE website

1. Open `/landing-page`. Show Home, About, the visitor experience/map,
   activities, preparation guidelines, and the appointment section. The public
   map gives visitor context; it is not the private staff planting workspace.
2. Start the request form. Enter synthetic group/contact information, choose
   activity, preferred date/time, participant count, and optional notes.
3. Demonstrate Back between form stages and an empty required field before
   submitting the valid isolated request.
4. Show planting-time/tide suggestions when available. The suggestions consider
   the two-hour visit; a missing forecast needs staff review. They do not
   guarantee that weather, water, or terrain is safe.
5. Explain that submitting creates a request for staff review. It does not
   automatically approve the activity, allocate points, or record planting.

**Current testing configuration:** planting operating-hour enforcement and the
activity access-window restriction are disabled in `bookingPolicy.json`.
Same-date visit validation and other existing checks still apply. Explain the
configuration honestly; do not change it during the defense or claim the
disabled time restriction is currently enforced.

### B. Staff authentication and workspace

1. Explain staff username/password plus email-code verification. Show the
   sign-in screen in the backup recording if your main session is already open.
2. Explain returning planter access separately; field users do not use staff
   passwords or staff email verification.
3. Orient the audience: sidebar, Planting Map tools, Philippine-time clock,
   profile/account link, notifications, and Help / Guide in the header.
4. Open a contextual guide briefly. End it before performing real actions.
   Guides explain controls without submitting records themselves.

**Say:** “Each role sees the workflow relevant to its responsibility. The server
also checks authorization when an action is submitted.”

### C. Scheduling

1. Find the prepared public request in List. Inspect group/contact information,
   preferred schedule, participants, notes, and activity type.
2. Explain contacting the requester before approving or rescheduling and
   approving. In the isolated demo, approve the prepared request.
3. Show Calendar and List. Approval decisions are Pending, Approved, Declined;
   progress is Pending, Completed, Cancelled. Approved is not Completed.
4. Show staff-entered scheduling as the alternative intake path. Use a prepared
   example for another activity type rather than submitting multiple requests.
5. Explain decline-reason validation, rescheduling, cancellation, and Missed as
   a follow-up label. Keep one prepared example of each for questions.
6. Open **Assign points** for the approved planting activity. Later, show its
   saved area/map link and return navigation. Planning details become protected
   after planting starts; completed activities remain read-only.

### D. Organizations and shared field access

1. Open Organizations. Search across the directory and show pagination with
   four organizations per finder page.
2. Inspect the organization profile: source activity/contact details, expected
   participants, project sites including archived entries, and registered members.
   Expected participants and registered usernames are different counts.
3. Explain **Activity codes** as viewing/copying/hiding an existing activity code.
   Do not reveal a real code on the projector. Code management for the selected
   activity remains in its existing staff controls.
4. Explain replacing/revoking a code: affected field sessions end and saved work
   remains. Show a recording made with synthetic credentials if needed.
5. Show Members with usernames, participant numbers, joining/last-seen data,
   search, and pagination. Explain automatic reload on return and Retry after
   a failed request.

### E. Drone image analysis

1. Open **Planting Map → Analyze Image**. Upload the prepared original
   geotagged image. Explain why its location metadata and usable coverage matter.
2. Inspect GPS, coverage, projected footprint, and alignment against the
   orthophoto. Respect partial-coverage, occupied-area and repeated-image
   confirmations; show Cancel before proceeding in the isolated demo.
3. Select Bungalon or Rhizophora and the existing danger-buffer control.
   Describe these as the application's existing planning settings. Do not
   change model thresholds or spacing rules during the presentation.
4. Run analysis and show progress. Explain that navigation/refresh can reconnect
   while the server job is running; zone and assignment changes are locked
   during processing. Have a saved result ready for a time-limited slot.
5. Use **Review analysis** to compare the original photo and result: purple
   canopy, red danger buffer, green available hexagons, orange eroded locations.
   Inspect counts, confidence/context and alignment before saving.
6. Save the reviewed result. Point out that erosion retains planned coordinates
   while excluding them from currently usable planting capacity.
7. Open Analysis history; filter, view a result, and use **View area in map**.
   Explain that display numbers differ from stable database IDs. A used
   analysis is protected from deletion; only eligible unused results offer it.

**Say:** “This is planting-location decision support. Image recognition,
orthophoto registration, and a phone's GPS do not establish surveyed accuracy
or replace a field suitability assessment.”

### F. Planting Map and zones

1. Pan and zoom; show current Planned, Assigned, Planted, Dead, Skipped and
   Unavailable counts, the legend, site boundaries, and point details.
2. Toggle orthomosaic/basemap and point/site/assignment/hazard layers. These
   switches change visibility, not records or export membership.
3. Open **Zone Editor**. Project sites already come from assignments. Locate a
   site by its saved name; do not demonstrate the retired manual site creation.
4. Draw erosion: start on the left, hold, drag right, trace and release inside
   the mapped area. Inspect the outline, covered count, and boundary validation.
5. Cover existing planted/dead demo seedlings. Click Save; show **Save eroded
   zone?** and the short record-preservation message. Click **Cancel** first and
   show that the saved state did not change. Then save the reviewed demo zone.
6. Show existing seedlings still carrying their recorded condition/death and
   original counts. Planned locations stay visible but cannot be newly planted.
7. In **Eroded Zones**, use **Edit**. Drag a vertex handle; arrow keys also move
   a focused handle. Cancel once, then save a reviewed change. Existing holes
   and separate polygon parts are retained. Save validates the current boundary.
8. Demonstrate two overlapping erosion zones using prepared data. Moving or
   deleting one does not unblock a location still covered by the other. After
   the last covering zone is removed, eligible locations recover their
   underlying state; dead seedlings do not become living seedlings.
9. Draw/show a Warning with its existing name, type, severity and notes.
   Warnings remain advisory. Automatic mortality warnings follow recorded
   mortality and cannot be removed like a manual warning.
10. Show erosion GeoJSON export, guarded Clear All/Delete, and site **Release
    area/Delete site** previews. Cancel destructive previews during the main
    defense; explain that area/history eligibility checks still apply.

**Capacity explanation:** the map's **Available planting locations** excludes
eroded locations. **Analyzed plantable area** is the saved image measurement,
not a claim of currently unoccupied or certified restoration area. Existing
historical plantings and deaths remain in their original reporting scope.

### G. Assign Points and automatic project sites

1. Choose the organization and its approved activity. If more than one activity
   is eligible, explicitly choose the intended one.
2. Select available points individually with **Move / click**. Demonstrate
   **Brush select** and **Brush erase** by holding the left mouse button and
   dragging. Release stops the stroke; hovering does not paint. Escape returns
   to Move / click. Touch uses press and drag.
3. Show species colors: Bungalon green, Rhizophora magenta; selected points are
   purple. Zoom changes marker size consistently with the rest of Planting Map.
4. Confirm the small reviewed selection. Assigning points forms and links the
   project site automatically; no separate manual boundary-creation step exists.
5. Inspect the new site/assignment. Explain that allocation is not actual
   planting and does not start seedling survival or monitoring counts.
6. Show erosion unavailable points excluded from selection. Explain that the
   server checks again at submission even if the map was opened before a zone
   changed. Warning points remain subject to field review but are not erosion.

### H. Planter field application

1. Open `/field` in the other browser/device. First-time participants enter the
   activity's shared code and choose a personal username. Returning participants
   use the same code and username to restore their work. No participant password
   or PIN is required. Demonstrate explicit browser switching only in rehearsal.
2. Inspect assignment/site context, My points, planting progress and
   **Refresh assignments**. Different participants receive their allocated work.
3. Tap a point for details: reference, species, coordinates and saved context.
   Show My location and GPS accuracy if permission and conditions allow it.
4. Start navigation, inspect the route/approach and ordered point guide, then
   use Next. Explain that following a route does not mark anything planted.
5. For a point physically planted in the demo scenario, select **Mark as
   Planted**, review the confirmation, then confirm. Show acknowledgment,
   yellow planted state, and progress. In real use, record actual work only.
6. Show **Skip Point** and its required reason on a separate demo location;
   skipped is not a death record. Do not skip a point merely to finish the demo.
7. Use **Choose points to mark**, select a physically completed demo batch,
   review, and confirm. Only selected points are submitted.
8. If any selected point is eroded at submission, show **Planting blocked**.
   The mixed batch saves no new plantings. Remove unavailable points and review
   the allowed selection before submitting again. Earlier saved work is kept.
9. Explain completed-activity acknowledgment and read-only protections. End the
   field session using the existing confirmed sign-out flow after showing that
   returning access restores saved work where the activity remains eligible.

For the short demonstration, use a prepared video for concurrent boundary
changes and batch rejection rather than racing two presenters on stage.

### I. Organization monitoring

1. Open Monitoring. Show **Due today**, **Overdue**, upcoming/site context,
   organization selection and **Sites to Monitor**, with five sites per page.
2. Explain that the cadence follows actual planting at 14-day intervals, with
   weekend rounds moved to Monday. A late visit does not rewrite planting dates.
3. Open **Record visit** on a prepared due organization. Inspect date and prior
   balance, then enter newly dead seedlings since the previous visit, including
   zero if none died. Record cause and known locations using the map selector.
4. Explain unlocated deaths: organization totals may include deaths without a
   mapped location; they must not be invented as particular point deaths.
5. Review alive balance, recorded survival, estimated growth, overall health
   and LGU actions, then save only in the isolated environment.
6. Show the success notification and saved visit History, including recorder
   and date. Open site details/map and return without losing filters.
7. Demonstrate that planted seedlings inside erosion remain monitorable and
   their condition can be updated through existing monitoring controls. Saving
   erosion itself never marks them dead or supplies a cause of death.
8. Show the printable monitoring field sheet from the existing location
   selector, with stable references and coordinates for offline field notes.
   This printed artifact does not provide offline synchronization.

### J. Dead-location release and a new planting cycle

1. On **Monitoring → Show map**, choose the organization and optionally **Show
   dead plants only**. Select a confirmed dead point; purple indicates selection.
2. Choose **Release from project site**. Review the eligible selection and
   confirm in the isolated demo. Living, removed, stale, eroded or subsequently
   reused locations are rejected by the existing checks.
3. Show that the released cell leaves the old site's current boundary and its
   mapped location becomes Planned. Death recording alone never did this.
4. Return to **Assign Points**, choose the intended organization/activity, and
   assign the available released location. The new site/cycle has its own
   attribution. Only actual subsequent planting creates another planted seedling.
5. Try the prepared dead location covered by erosion. Show **Replanting
   blocked**. Neither a new status nor a replacement queue is needed.

**Illustrative explanation:** Site A planted 100 seedlings and recorded 20
deaths. Releasing those locations leaves that original batch at 100 plantings,
20 deaths and 80% recorded survival. If Site B later plants 20 replacements,
they are new planting events for Site B. The original deaths are not erased.
An all-sites result combines a different denominator and can therefore differ.

Do not describe this as restoring dead seedlings. A legitimate correction to
an incorrectly recorded death follows existing guarded correction rules;
planting a new seedling is a new cycle.

### K. LGU Sites

1. Switch Monitoring navigation to **LGU Sites**. Show LGU as one permanent
   organization with multiple separate project sites.
2. Show the permanent planter-access panel without revealing real credentials.
   Explain Returning planter access and the absence of a timed booking for LGU.
3. Use **Assign LGU points** to open Assign Points with LGU selected. Site
   formation and planting follow the normal workflow.
4. Open a prepared due site's **Record monitoring**. Select newly dead
   locations and their cause, record health/actions and, when available, an
   actual measured average height. No selected deaths means zero newly dead.
5. Inspect History. Saving a visit updates that site's records and cadence;
   other LGU sites retain their own dates and outcomes.
6. Explain that combined LGU survival uses total surviving and planted
   seedlings, not an unweighted average of site percentages.

### L. Dashboard, goals and reporting

1. Open Dashboard. Set a valid From/To range and optional Project site for the
   period-based tabs. Show loading, empty and retry states only if useful.
2. **Overview:** inspect planting progress, follow-up and annual-goal context.
   **Planting Goals** has its own reporting year and existing planting/survival
   targets. Show a prepared goal rather than modifying shared targets.
3. **Planting Work:** show organization activity, completed/skipped work and
   pending assignment context. Allocation and completion are separate figures.
4. **Seedling Health:** inspect planting-batch survival, species/site comparisons,
   monthly recorded deaths, and matching table views. The selected planting
   dates identify cohorts; recorded outcomes are evaluated at generation time.
5. **Project Sites:** show the **Current snapshot**, organization, current mapped
   location statuses, erosion exposure and mapped warnings. This tab uses the
   site filter across planting dates, without the period controls.
6. Use **View planting history** to reach Seedling Health with site and dates
   retained. Show **View map**, **Monitoring** and **Back to [section]** links.
   Released sites retain history even when no current boundary remains.
7. Open **Download Report** (or **Download Monitoring Report** in Monitoring).
   Choose report type, dates and optional site; inspect the live preview and
   metadata, then add an optional remark.
8. Cover all four types: **Planting accomplishment**, **Survival and monitoring**,
   **Mortality and replanting**, **Organization activity**. Show one full PDF
   and explain the remaining types using prepared previews.
9. Demonstrate **Save as PDF**, **Download CSV**, and **Print** using synthetic
   data. These reports identify scope, generation time and preparer; they are
   not official certification or a digital signature.

**Recorded survival:** seedlings without a recorded death ÷ seedlings planted
in the selected batches × 100, subject to the existing missing-data rules.
Awaiting-inspection seedlings remain included; call it recorded survival,
not a field-verified ecological survival estimate. Do not add successive visit
balances together or combine different chart scopes as if their totals matched.

### M. Scoped planting-point exports

1. Open **Planting Map → Export Saved Points**. Choose format first: CSV, GPX,
   KMZ or GeoJSON. Then choose All points, Project site, Species, or Saved analysis.
2. For a scoped choice, select its value. Inspect the actual map preview, point
   count, hover details, legend, scope and generated filename. Pan/zoom and
   **Fit selected points** do not change the exported selection.
3. Explain that Project site uses current ownership; released points leave the
   former site's export. Unowned and unrecorded/other species remain explicit.
4. Download a KMZ and open the prepared synthetic file in Google Earth. Solid
   dot colors are packaged with it; selecting a dot shows its name/details.
5. Explain the color priority: **dead red → eroded orange → planted yellow →
   species** (Bungalon green, Rhizophora magenta; other/unrecorded gray).
   An orange planted point is still a recorded planted seedling; color expresses
   erosion exposure without rewriting its condition or history.
6. CSV carries tabular attributes. GeoJSON carries geometry, attributes and
   color hints; QGIS can read it but may require selecting a color field/style.
   GPX carries navigation waypoints with stable short IDs and descriptions;
   displayed colors depend on the receiving GPS application/device.
7. Use the prepared GPX in a rehearsed compatible viewer if asked. Do not promise
   identical styling across viewers or upload private coordinates to a third
   party during the defense. There is no separate QGIS ZIP export button.

### N. Activity logs, notifications, guides and account

1. Open Activity Logs, use **Refresh activity** and **Load more activity**,
   and inspect actor/time/action for the demonstrated work. Logs support
   traceability; they do not replace inspecting the source record.
2. Open the header notification panel. Explain the configured reminders and
   follow-up context. Email delivery depends on the configured sender and
   scheduler; use prepared synthetic examples, not a live send experiment.
3. Open Help / Guide and select **Full system workflow** or a task guide. Show
   Next, Back, Finish/End and Escape. End the guide before using real controls.
4. Open Account from the header profile. Explain current-password plus email
   verification for password changes and existing account recovery. Show a
   recording rather than changing a working account during the defense.
5. Show the confirmed staff sign-out at the end, after recording any evidence
   needed for follow-up questions. Keep the current green/white interface and
   supplied branding; no dark-mode toggle is part of this version.

## 7. Complete feature coverage checklist

Use this to assign presenter responsibilities and record rehearsal evidence.

| Feature group | Evidence to have ready |
| --- | --- |
| LIKE public information, map and activities | Public pages and map dialog. |
| Two-stage booking, validation, tide suggestions | Synthetic request, validation example, unavailable-forecast example. |
| Staff sign-in, email verification and recovery | Redacted recording with no usable codes. |
| Calendar/List, direct schedule, approval/reschedule/decline/cancel | Prepared examples with distinct decision/progress statuses. |
| Organization directory, contacts, archived sites, members, activity codes | Synthetic organization profile with multiple members. |
| Upload metadata, coverage/alignment, partial/occupied/repeat guards | Prepared image and redacted preflight screenshots. |
| Detection, danger buffer, grid, analysis review/save/history | Original/result comparison and protected used-analysis example. |
| Shared map, layers, point/site hover, current counts, return navigation | One map location shown from several source pages. |
| Erosion create/edit/cancel/delete/overlap and boundary checks | Before/after screenshots, preserved seedling record, blocked submission. |
| Manual warnings and automatic mortality warnings | Prepared warning examples; manual warnings stay advisory. |
| Assign Points, held brush/erase, species display, automatic sites | Small reviewed selection, linked activity and resulting site. |
| Shared planter code, usernames, returning access, browser switching | Redacted synthetic login video. |
| Field location/accuracy, routes, Next, details and refresh | Rehearsed phone session and route fallback video. |
| Individual planting, skip reason, reviewed batch, completed activity | Prepared participant assignments and acknowledgment screenshots. |
| Organization due/overdue monitoring, death locations, visits/history | One due organization and known/unlocated-death examples. |
| Monitoring map, field sheet, dead-point release and next cycle | Released boundary comparison plus unchanged historical result. |
| LGU permanent access, multiple sites and site-specific visits | Two LGU sites with different due/history states. |
| Area release/guarded site deletion | Eligibility preview with Cancel, no loss of history. |
| Dashboard four tabs, goals, filters and table/chart scopes | Current versus historical example using the same site. |
| Four report types, preview, PDF, CSV and Print | Synthetic report files with readable scope metadata. |
| Four point formats and four scopes, preview and names | KMZ color example; CSV/GeoJSON/GPX attribute examples. |
| Logs, reminders/notifications, contextual/full guides, account/sign-out | Relevant action entries and redacted controls. |

## 8. Suggested defense questions and answers

**Why is assignment separate from planting?**
Assignment reserves work. An actual completion creates the seedling planting
record that starts planting totals and monitoring. Planned capacity is not an
accomplishment count.

**Why do map totals differ from survival reports?**
The map counts current physical locations and ownership. Reports can count
planting events from selected dates, including earlier cycles at released
locations. Read each scope before comparing figures.

**Does erosion mean the seedling is dead?**
No. Erosion is a spatial restriction on future planting/replanting. Existing
condition, death cause and historical counts stay recorded independently.
Monitoring remains allowed.

**What happens if a zone changes while someone is planting?**
The server validates current saved boundaries at submission. Boundary saves
and planting actions coordinate through transaction locks. A mixed eroded batch
is rejected before any new planting in that batch is saved.

**Can one deleted erosion zone reopen a location covered by another?**
No. Coverage is checked against all current active eroded zones, including
their boundaries. The location remains blocked while any one covers it.

**Does replanting improve the original batch's survival?**
No. Earlier deaths stay with the original batch. A new planted seedling is a
separate event with its own organization/site attribution and outcomes.

**Are all predicted points definitely suitable?**
No. Detection, registration, mapped hazards and species settings support
planning. Soil, tide, access, field conditions and positioning uncertainty still
need appropriate human review. Present model evaluation from your actual
thesis experiment separately from application workflow tests.

**Does the system identify each seedling in a drone photograph?**
Describe the implemented canopy detection and generated planting locations.
Mapped planting/condition records are entered through planting and monitoring;
do not claim automated seedling health or cause-of-death diagnosis.

**Is growth directly measured?**
Distinguish the displayed estimate from recorded field height measurements.
An estimate is not proof of the seedling's actual size.

**Can the reports prove certified restoration area or carbon capture?**
No. This version reports operational planting and monitoring records. It does
not certify land area, ecological success, attendance, carbon or finances.

**Is the exported point color standardized?**
KMZ packages the application colors. GeoJSON and GPX carry useful attributes
and hints, but rendering depends on the receiving application.

**What was tested?**
Separate model evaluation, isolated software regressions, manual workflow
rehearsal and usability results. Do not equate a passing build or synthetic
test suite with validated live field accuracy or complete user acceptance.

## 9. Fallbacks and finish

| Interruption | Prepared response |
| --- | --- |
| Slow image processing | Show progress, then open the prepared saved analysis and original/result comparison. |
| Network/API unavailable | State the dependency; use the prerecorded workflow and prepared synthetic files. |
| GPS denied or inaccurate indoors | Show the accuracy/permission state and recorded outdoor navigation; do not fabricate a fix. |
| Verification email delayed | Continue with the prepared signed-in staff session and redacted sign-in video. |
| No monitoring due today | Open prepared history and show the recording; do not alter real planting dates. |
| Export viewer differs | Show KMZ in the rehearsed viewer and inspect GeoJSON/GPX attributes. |
| Older build on public site | Verify the deployment revision; use the rehearsed runtime and identify its build. |

Suggested closing:

> The demonstration follows a location from planning to recorded planting,
> monitoring and a later planting cycle. Staff can manage present availability
> while keeping the history needed to interpret restoration reports.

Before leaving, record the demonstrated branch/build, completed checklist,
remaining unverified items and audience feedback. Leave the data as agreed for
the isolated demo; do not reset the shared database or publish changes as an
automatic cleanup step.

## 10. Supporting implementation guides

- [System user guide](system-user-guides.md)
- [Shared planter access](shared-planter-access.md)
- [LGU Sites](lgu-sites.md)
- [Project-site location release](project-site-location-release.md)
- [Current seedling health and historical survival](current-seedling-health.md)
- [Project Sites dashboard](dashboard-project-sites.md)
- [Mortality warnings](site-mortality-warnings.md)
- [Restoration reports](restoration-reports.md)
- [Planting-point exports](planting-point-exports.md)
- [Image alignment](image-alignment.md)
- [Tide guidance](tide-planting-guidance.md)
- [Managed laptop setup](groupmate-setup.md)
