# MangroVision — Developer handover for the next chat

Prepared on **October 14, 2026**, timezone **Asia/Manila**. Read this together with the root `AGENTS.md`. This is a source and conversation handover, not a new deployment or live service certification. It contains no private credentials or planter codes.

## 1. Start here

MangroVision is our thesis system for mangrove planting planning, organization scheduling, planting assignments, field recording, and monitoring. It combines drone-image canopy analysis with mapped planting locations and retained field records.

The goals include preserving records for better decisions, coordinating organizations, and supporting seedling survival through appropriate spacing and follow-up monitoring. Do not describe improved survival or AI accuracy as experimentally proven unless an actual evaluation supports that claim.

The user prefers concrete changes and short explanations. Inspect the relevant implementation, preserve earlier approved logic, implement the requested work, and verify the specific remaining risks. Ask only for information or authorization genuinely needed. Do not automatically push, deploy, migrate, reset accounts, send emails, or alter live planting records as a side effect of a UI task.

### Current Git snapshot

Before this handover file was added:

| Item | Observed state |
| --- | --- |
| Repository | `https://github.com/LukeSekai/MangroVision.git` |
| Current branch | **`Latest-Main`**, with this exact capitalization |
| HEAD | **`7679e15`** — Add pagination to image analysis history and improve filter handling |
| Locally recorded upstream | `origin/Latest-Main`, also at `7679e15` |
| Working tree | Clean |
| Latest integration merge | `f158dca` — Merge planter sign-in fixes with Latest-Main UI updates |

The remote comparison above uses the locally stored remote reference; this handover did not perform a new fetch. Recheck state when starting the next task. Adding this Markdown file does not commit or push it.

Recent history:

```text
7679e15  Analysis History pagination
f158dca  Merge of the user's planter fixes and groupmate's UI updates
d989945  Groupmate UI/UX fix
750b1f1  User planter sign-in and request-review fixes
e021ff2  Keep unapproved activities out of calendar; remove expected seedlings
```

Useful preserved branches are `fix/planter-code-review` and `integration/planter-review-ui-20261013`, both at `f158dca` in this snapshot. Older branches such as `master` and `Try_New_Latest` are not the current integration target.

## 2. Architecture and where to develop

The current application is **FastAPI + React**, backed by **PostgreSQL/PostGIS** and private object storage. Extend the API and client projects for application changes.

```text
MangroVision/
  AGENTS.md                         Project instructions
  .env                              Private backend configuration; ignored
  .env.example                      Shareable configuration template
  requirements.txt                  Root Python dependencies
  planting_database.py              Compatibility façade for database operations
  mangrovision_db/                   Database/domain services and storage
  alembic/versions/                  Versioned schema migrations
  canopy_detection/                 Detection, spacing and alignment implementation
  waypoint_export.py                Coordinate/waypoint export
  scripts/                          Setup, maintenance, backup and migration tools
  docs/                             Operational and feature documentation
  MAP/FINAL/final_orthophoto.tif      Separately shared runtime asset
  MangroVision_New/
    start_dev.py                    Managed local API + Vite launcher
    start_testing.py                Managed API + public tunnel launcher
    deploy_testing.py               Matching frontend deployment and checks
    api/main.py                     FastAPI app, middleware, routers, readiness
    api/routes/                     HTTP endpoints
    api/data/                       Shared GIS/routing data
    try_model/model_final.pth        Separately shared detector checkpoint
    client/
      package.json                  Frontend dependencies and commands
      vite.config.js                Local proxy and two frontend entry points
      src/StaffApp.jsx               Staff routes
      src/pages/FieldApp.jsx         Planter workspace and code sign-in
      src/like/                     Public LIKE website and booking form
      src/components/               Shared UI, maps, forms, dialogs and history
      src/stores/                   Zustand auth, map and processing state
      src/utils/                    Shared validation, counts, tides, routing helpers
```

### Main technologies

| Technology | Role / reason for use |
| --- | --- |
| Python / FastAPI | API, image processing integration and server authorization |
| React 19 / Vite 8 / React Router | Staff, planter and public website interfaces |
| Zustand | Shared client session, processing and map state |
| Leaflet / React Leaflet | Interactive map, planting points and zones |
| Recharts | Dashboard and tide charts |
| PostgreSQL / PostGIS / SQLAlchemy / Alembic | Persistent records, geometry, transactions and schema changes |
| Supabase private storage / S3-compatible access | Private saved images and related objects, accessed by backend |
| Detectree2 / Detectron2 / PyTorch | Saved canopy detector; runtime compatibility matters |
| OpenCV | Image feature matching and alignment with the orthophoto |
| Google Routes | Optional walking directions to mapped access routes |
| WorldTides / Open-Meteo Marine | Tide forecast integration and fallback |
| Brevo HTTPS and SMTP configuration | Transactional email delivery; HTTPS was added after SMTP timeout problems |
| Vercel + Cloudflare Quick Tunnel | Hosted frontend reaching the laptop-hosted API during testing |

Use `package.json` and the lockfile for actual dependency versions. Do not upgrade libraries, replace the ML checkpoint, or redesign authentication incidentally while fixing a small feature.

### Routes and panels

| User-facing panel | Route / primary frontend | Primary backend area |
| --- | --- | --- |
| Planting Map | `/map`, `pages/MapAnalytics.jsx`, `components/PlantingMapWorkspace.jsx`, `MapView.jsx` | `routes/seedlings.py`, `analyses.py`, `project_sites.py`, `zones.py` |
| Dashboard | `/dashboard`, `pages/Dashboard.jsx`, `RestorationReports.jsx` | `routes/dashboard.py`, `mangrovision_db/health_summary.py` |
| Scheduling | `/scheduling`, `pages/Scheduling.jsx`, `components/WebsiteRequests.jsx` | `routes/planting_schedules.py`, `like_appointments.py`, domain activity/email services |
| Monitoring | `/monitoring`, `pages/OrganizationMonitoring.jsx`, `SitesToMonitor.jsx`, `MonitoringSiteDetails.jsx` | `routes/monitoring.py`, monitoring/mortality domain services |
| Monitoring map | `/monitoring/map`, `pages/MonitoringMapWorkspace.jsx`, `components/SeedlingLocations.jsx` | Monitoring and seedling endpoints |
| Analyze Image | `/processing`, `pages/ImageProcessing.jsx`, `components/ImageAnalysisHistory.jsx` | `routes/processing.py`, `analyses.py`, `api/processing_jobs.py`, canopy code |
| Planting Assignments | `/planters`, `pages/PlanterManagement.jsx` | `routes/assignments.py`, `planters.py`, planting order and account services |
| Zone Editor | `/zones`, `pages/ErodedZoneEditor.jsx` | `routes/zones.py`, `project_sites.py`, domain zones |
| Activity Logs | `/activity`, `pages/ActivityLog.jsx` | `routes/activity.py`, audit middleware and domain activity |
| Planter side | `/field`, `pages/FieldApp.jsx`, `stores/planterAuthStore.js` | `routes/planter_auth.py`, participant-authorized endpoints |
| LIKE website | `/landing-page`, separate `src/like/main.jsx` / `LikeWebsite.jsx` | Public `/api/public/like` and staff `/api/like-appointments` |

`/` redirects an authenticated staff user to `/map`. `/reports` redirects to `/dashboard`. The public website is a side feature at `/landing-page`, not the primary root page. `/like.html` is retained for compatibility.

## 3. Practical development workflow

### At the beginning of each task

1. Read `AGENTS.md`, this handover, and the feature-specific documentation.
2. Check branch, saved work and current diff before editing:

   ```powershell
   git status --short --branch
   git log -5 --oneline
   git diff --stat
   ```

3. Find the implementation using `rg` / `rg --files`. Trace the UI to its API and domain service before changing business rules.
4. Check whether the running API points to the shared database. A feature branch does **not** isolate database state.
5. Make the smallest complete change that meets the request. Reuse current helpers rather than adding conflicting calculations or parallel authentication paths.
6. Run meaningful existing checks and inspect the diff. Report what changed, what passed, and any specific limitation.

Use PowerShell commands from the repository root unless a command explicitly says otherwise. Prefer the project Python executable over whichever Python happens to be on PATH:

```powershell
.\venv\Scripts\python.exe MangroVision_New/start_dev.py
```

This serves the API on port **8000** and Vite on **5173**. Local URLs include `/`, `/field` and `/landing-page` on `http://localhost:5173`. To stop the managed stack:

```powershell
.\venv\Scripts\python.exe MangroVision_New/start_dev.py --stop
```

Starting another managed launcher can replace the existing managed stack. Check whether a remote testing session is running before starting local development. Avoid a second API process on the same port.

### Frontend conventions to preserve

- Authentication uses cookies and CSRF protection. `src/utils/secureFetch.js` attaches credentials and the CSRF header, removes legacy token query parameters, and integrates read-cache invalidation. Do not add plaintext credentials or session tokens to browser storage, URLs or logs.
- Reuse `useFormFeedback.js`, `formValidation.js` and `components/FormFeedback.jsx` for field errors, success/error notices and loading behavior. Focus the first invalid field, retain entered values after a failure, and prevent duplicate submissions.
- Give network/service errors an actionable explanation. A timeout is not proof that the entered code is wrong. Preserve retry identifiers where an operation might already have succeeded.
- `RetainedRoutes.jsx` uses React `Activity` to preserve visited pages while pausing hidden-page effects. Do not let hidden routes attach competing map handlers, open dialogs, or continue unnecessary polling.
- Shared dialogs may stay mounted. Use unique IDs (`useId`), proper hidden/inert behavior, a scrolling body and visible action footer. Review dialogs use a body portal to appear above the workspace header.
- Reuse `pointStatus.js`, `mapPointStats.js`, `health_summary.py` and related shared functions for status/count behavior.
- `src/guides/` contains user help/tours. Update help when changing a workflow; check for obsolete terminology.

### Backend conventions to preserve

- Keep authorization and business constraints server-side. Hiding a button is not sufficient protection.
- Use domain services under `mangrovision_db/` for changes involving schedules, assignments, accounts, monitoring or email queues. Keep related writes atomic.
- Preserve row-locking and concurrency checks where recording planting races with schedule editing or email delivery.
- Add schema changes through Alembic migrations. Do not modify the shared schema manually during ordinary feature development.
- Keep backend and frontend request/response contracts synchronized. Compare the running API schema when a deployed client fails unexpectedly.
- Health readiness verifies connectivity/schema/storage, not every signed-in workflow. Do not equate a ready health response with a successful code login, email delivery or AI run.

## 4. Rules that must survive future changes

### A. Planter access is an individual reusable code

The user chose **one individual code per planter, valid across the organization's activities**. Do not restore shared username/password sign-in or device-dependent slot allocation.

- A code is eight characters, displayed as `XXXX-XXXX`. It is a private reusable credential, not a six-digit one-time email code.
- The request contains only `access_code`. No username, password, `device_key` or recovery credential is required.
- Backend and frontend normalize lowercase, spaces and common pasted Unicode dash variants consistently.
- The same code restores the same account and `participant_slot` on a new browser, phone, domain or device, with the same assigned points and saved progress.
- Participant limits cap **participant slots**, not browser sessions. Signing in again must not consume a new slot.
- Sign-in creates a session and updates the real `planters.last_login` column. The erroneous `last_login_at` name was fixed in both implementation and fixture.
- Existing account IDs, assigned points, participant numbers, planting history and legacy device bindings are preserved. Legacy bindings do not restrict code login.
- Repeated issuance/bookings retain existing codes. Revoked codes stay revoked until explicitly replaced.
- Revoking/replacing a code ends that participant's sessions while preserving their slot and records. It does not sign out other participants.
- Public planter registration is removed. The old registration API returns **410**.
- Staff use **Planting Assignments → Planter codes** for issuance, private retrieval and replacement/revocation.
- `LIKE_EMAIL_ENCRYPTION_KEY` is the stable backend-only key used by code lookup/vault and encrypted email content. Do not change it casually or put it in the client.
- Durable sign-in limits and `Retry-After` handling remain in place. Staff code responses are `no-store`; codes must not enter public session payloads or activity logs.

The October 12 conversion covered **nine existing organization owners and 66 participant slots**. Those are historical conversion counts, not a current database census. Do not reset or reissue them to repair a deployment mismatch.

Primary files: `mangrovision_db/planter_codes.py`, `api/routes/planter_auth.py`, `client/src/stores/planterAuthStore.js`, `client/src/utils/planterCode.js`, and `client/src/pages/FieldApp.jsx`.

### B. Current map statuses and historical events are different

- Current status cards must agree between Planting Map, Planting Assignments and Dashboard for the same scope.
- Overall Seedling Health has **four cards**: Total seedlings, Planted, Dead, Survival rate.
- **Total seedlings = current Planted + current Dead.** Each location counts once.
- **Survival rate = current Planted / (current Planted + current Dead) × 100.** An empty total has no rate. Current Planted includes locations not yet inspected; do not present this as an inspection-only survival study.
- Overall health covers all planting dates for the selected site. Date filters apply to visit, death-cause and measurement history.
- Planned, Assigned, Skipped, Unavailable and deleted locations are excluded from overall seedling totals.
- Historical planting totals can differ from current Planted because a planted seedling may later be dead or released for replanting. Do not silently substitute historical events into a current-status card.
- Located deaths from organization visits update map status. Organization-wide visit balances stay in history and must not be added again to current deaths.
- Clearing a dead location for replanting removes it from current seedling totals until planting is recorded again. Preserve the earlier planting/death/inspection cycle.
- Do not restore the removed 30-day summary, Recorded survival, Awaiting health record, or inspection-coverage cards.
- Aggregate species case-insensitively: `bungalon` and `Bungalon` belong together. The species chart is vertical and stacked.
- Death-cause percentages are each cause's share of all included deaths; they are not cumulative percentages.

The map's broader point overview still includes Planned, Assigned, Planted, Dead, Skipped and Unavailable. That is a different scope from the four seedling-health cards.

### C. Activities and assignments protect recorded work

- New planting assignments link to an eligible tree-planting activity for the selected organization/site. Ambiguous matches require a selection.
- Once a linked assignment has planting-event history, protect its planning details, organization, site and schedule. Cancellation/deletion and reopening an already completed planted activity are blocked.
- Contact details and notes can still be corrected. Allowed activity progress changes remain available.
- A separate future activity at the same site remains editable; do not lock every activity merely because that site has plants.
- Assigned but unplanted points still prevent inappropriate organization/site changes and deletion until assignments are removed.
- Historical activity-link backfill was deliberately conservative. The October 9 review found unmatched planted batches; do not invent associations or claim all historical batches have a schedule lock.
- Scheduling no longer asks for or displays **expected seedlings**. Historical stored columns may remain for compatibility; do not restore the field because an old test expects it.
- Calendar displays confirmed, in-progress and completed activities. Draft/tentative/cancelled records remain available in List. Pending requests are reviewed separately.

### D. Monitoring and brush behavior

- Monitoring rounds are anchored to 14 calendar days from each point's actual planting date, in Philippine time. Weekend rounds move to Monday while later rounds remain anchored to planting-day multiples of 14.
- Today's and overdue work are separate queues. Backend eligibility remains authoritative.
- Visits may be organization-wide; do not describe an organization-wide visit count as a per-site inspection count.
- Brush select/erase is initially armed after choosing the tool. A click on the map starts painting; a second click stops. Escape or Move/click stops it. Merely moving across the map after choosing Brush must not paint.
- Validate reported/located deaths, unavailable points, prior deaths and concurrent changes. Preserve visit history and avoid duplicate death counts.

### E. Zones, spacing, image analysis and routing

- Project sites link to their responsible organization. Eroded/forbidden areas are excluded from new planting; warning zones show cautions without automatically blocking assignment.
- Canopy inference is not seedling health detection. Preserve canopy buffers, species spacing, GIS exclusions and existing-point checks when generating new candidate points.
- Re-analysis in planted areas must preserve existing field records and avoid duplicate planting locations. Keep confirmation/protection behavior where relevant.
- **GSD** is ground distance per image pixel. It converts image distances/areas to ground units; it does not prove positional accuracy or successful survival.
- Candidate locations use a hexagonal grid. Assignment ownership uses geographic zigzag strips divided into balanced shares; changing visiting order alone is not equivalent to that allocation.
- Assignments never move saved point coordinates. Repairs of ownership are limited to wholly pending assignments with no planting history.
- Image preview, overlay footprint and exported coordinates use the same accepted image-to-orthophoto transform. Preserve validation and approximate-GPS fallback when alignment evidence is insufficient.
- SIFT/RootSIFT and validated similarity/affine/projective transforms are used for alignment. Do not loosen acceptance limits or synthesize a plausible footprint for appearance alone.
- GPS-to-point navigation preserves the traced physical access road and site footprint in `api/data/site_access_routes.json`. Do not enlarge snap tolerances to mask missing routes or draw an invented safe shortcut.
- Routes need fresh phone GPS. If road directions are unavailable, explain partial/local guidance and omit unsupported total trip estimates.

## 5. Scheduling, tide advisories, booking and email

The evaluator requested clearer error handling, reschedule notifications, and tide information for planters. These were implemented before the latest merge.

### Booking and rescheduling

- Public appointment choices: Field visit, Clean-up drive, Tree planting.
- Public/manual booking captures organization, contact person, email, phone, future date, start/end time and participant count. Use the selected activity type to generate the title; do not add a redundant activity-name input.
- Confirmation requires staff coordination with the requester. Review distinguishes confirming the requested time from rescheduling and confirming a newly agreed time.
- Approval creates the linked schedule and queues its confirmation atomically and idempotently. Repeated approval must not create duplicate schedules/accounts/messages.
- Tree planting provisions organization access and participant codes where needed. Field visits and clean-up drives do not create planter accounts or accept a planting-area assignment.
- Relevant staff and website schedule changes queue confirmation/update/cancellation notices. Preserve notification recipient handling, delivery conflict checks, encrypted content and retries.
- Planters can see the latest activity schedule and cancellations. Refresh on focus/manual action and the existing polling interval.

### Tide meaning

- Keep schedule, tide advisory, forecast freshness and estimated elevated-water period available on the planter side.
- The elevated-water period is estimated from sampled forecast threshold crossings. It is not the duration of the high-tide peak.
- Do not invent predictions beyond sample coverage or smooth over missing samples. Stale/gapped/out-of-range data should produce an unavailable/unknown explanation.
- The estimated graph reference is not surveyed ground elevation. Forecast labels are planning estimates, not safety clearance or a long-term suitability assessment.
- Preserve common assessment helpers so calendar, activity advice and graph use consistent thresholds. Groupmate changes added daily/weekly tide presentation without replacing these rules.
- See `docs/tide-planting-guidance.md` for calibration, reference compatibility, forecast freshness and limitations.

### Email and other API status

- Confirmation/update/cancellation emails have a shared HTML design and a plain-text alternative. Dynamic values are escaped and action links accept only HTTP/HTTPS.
- Tree-planting access emails provide individual planter codes; cancellation messages omit codes and the planter button.
- Unsent content is encrypted. Provider acceptance, queued status and actual inbox delivery are different things. Do not mark an email delivered merely because a schedule was saved.
- Brevo HTTPS was added after SMTP connection timeouts. A Gmail sender address does not establish that a separate Gmail OAuth/API integration exists.
- Having keys in configuration does not establish that Brevo, tides or Google Routes work now. Verify the relevant integration with approved checks before making a live-status claim.
- Google Routes attempts have local default limits of 100/day and 1,000/month, persisted in an ignored usage database. These are not a guarantee of zero Google charges across all servers/projects. Keep the key server-side.
- Do not send real test emails, submit live bookings or use real planter codes without authorization covering that action. Use fictional/mocked data for development checks.

## 6. Development setup, assets and private settings

For another laptop, **follow `docs/groupmate-setup.md` in order**. Automated test suites are not required for setup.

Required separately shared runtime assets:

| Asset | Path | Reference size |
| --- | --- | ---: |
| Georeferenced orthophoto | `MAP/FINAL/final_orthophoto.tif` | About 113.8 MiB |
| Preferred detector checkpoint | `MangroVision_New/try_model/model_final.pth` | About 479.8 MiB |

Git does not carry these files. Model metadata is tracked. The full WebODM directory and historical tile sets are not needed; the API can render from the GeoTIFF. Training annotations are needed for retraining/evaluation, not normal checkpoint inference.

Typical installation from the root:

```powershell
py -3.12 -m venv venv
.\venv\Scripts\python.exe -m pip install -r requirements.txt
npm.cmd --prefix MangroVision_New/client ci
.\venv\Scripts\python.exe scripts/check_groupmate_setup.py
.\venv\Scripts\python.exe MangroVision_New/start_dev.py
```

After starting:

```powershell
.\venv\Scripts\python.exe scripts/check_groupmate_setup.py --online
```

Node 22.12+ and Python 3.12 are the documented setup baseline. Detectron2 needs a compatible PyTorch/torchvision build; the requirements file alone does not guarantee full detector support. Follow the guide and report missing assets/dependencies rather than silently claiming the fallback is the preferred detector.

The root private `.env` supplies backend database/storage/auth/email/provider configuration. Environment-provided values take precedence. Never print, commit, paste into the next chat, or put server secrets in `client/.env*` or `VITE_*` variables.

Configuration names to recognize, without copying values:

- Runtime DB: `DATABASE_URL`, `APP_DATABASE_ROLE`, `DB_SCHEMA`, `DB_SSLMODE`.
- Owner connection: `MIGRATION_DATABASE_URL`, separate from runtime access, for the same intended database.
- Private storage: server-only `S3_*` settings.
- Cookies/origins: `TRUSTED_ORIGINS`, `COOKIE_SECURE` and related server settings.
- Code/email encryption: stable `LIKE_EMAIL_ENCRYPTION_KEY`.
- Email delivery: `BREVO_API_KEY`, relevant `SMTP_*` and public planter/booking URL settings described in the email docs.
- Optional providers: `GOOGLE_ROUTES_API_KEY`, `WORLDTIDES_API_KEY`.

Do not copy migration/admin/bootstrap credentials to a collaborator merely to run the app. For access to existing records, their API needs the correct authorized shared database and private storage, or they can use the owner's hosted API. A new empty local database does not acquire records through Git.

## 7. Database changes and activation

The repository migration head in this snapshot is **`20261012_0017`**. Prior sessions reported activation through 0017 in the shared database; recheck readiness before relying on current live state.

| Migration | Feature |
| --- | --- |
| `20261006_0013` | LIKE requests, limits and schedule source |
| `20261007_0014` | Appointment types and confirmation queue |
| `20261009_0015` | Assignment/activity association and recorded-planting guards |
| `20261011_0016` | Staff activity notification recipient and email outbox |
| `20261012_0017` | Individual planter code vault and durable auth limits |

Those activations were explicitly approved and completed in earlier sessions. Their approval does not authorize unrelated future migrations, code resets or data changes.

For development, use a disposable/local PostgreSQL/PostGIS database with its own configuration. `docs/like-website.md` explicitly separates shared database activation from feature development.

Before a newly requested shared activation:

1. Finish the feature and make the proposed migration reviewable.
2. Check the runtime and migration connections target the same intended project, without printing credentials.
3. Obtain approval if existing authorization does not cover the new shared change.
4. Create a fresh encrypted database/private-storage backup and verify the archive, dump and object checksums.
5. Apply the reviewed migration, preserve counts/content and protected records, then coordinate backend and frontend activation.
6. Verify readiness and relevant workflows; do not silently use live planting/email actions as tests.

Earlier verification generally checked archive/dump readability and checksums; it did **not** perform a full restore. Do not report otherwise. A restore rehearsal should target an isolated environment. The production restore tool replaces represented objects and must not be used casually on the shared database.

Read `docs/production-database.md` and the scripts before executing backup/restore commands. Keep backup encryption passwords in private configuration/process environment, not command arguments or chat. Do not downgrade, drop, reseed, re-import SQLite or bootstrap accounts on the already configured shared DB to repair an application error. Branch switching does not roll back its database.

## 8. Deploying the correct version

Known production testing URL:

- Staff: `https://mangrovision-user-testing.vercel.app/`
- Planters: `https://mangrovision-user-testing.vercel.app/field`
- LIKE: `https://mangrovision-user-testing.vercel.app/landing-page`

These are known URLs, not a current uptime guarantee.

**A Git push, frontend deployment and backend restart are separate actions.** The configured Vercel production branch is `Latest-Main`. Git integration can publish the frontend, but it does not restart the laptop's Python API.

Read `docs/user-testing.md` before activating. Once activation is authorized, the managed command is:

```powershell
.\venv\Scripts\python.exe -X utf8 MangroVision_New/start_testing.py --deploy
```

On a new host, use the documented `--frontend-url` option. The launcher starts the API and a Cloudflare Quick Tunnel, configures testing cookies/origins, and publishes routing to that tunnel. Keep the host laptop awake and connected. Closing it or stopping the API/tunnel interrupts remote use although the frontend remains hosted.

Every new Quick Tunnel URL is different; redeployment updates the frontend proxy. Never assume a URL from an old log is the active backend. Managed state/logs are in ignored `MangroVision_New/.dev-server/`.

If the API/tunnel remain running but deployment failed:

```powershell
.\venv\Scripts\python.exe -X utf8 MangroVision_New/deploy_testing.py
```

Stop testing with Ctrl+C or:

```powershell
.\venv\Scripts\python.exe -X utf8 MangroVision_New/start_testing.py --stop
```

Vercel uses public routing settings, including the current `MANGROVISION_BACKEND_URL`, empty `VITE_API_BASE`, and `/tiles` for `VITE_TILE_SERVER`. Same-origin proxying preserves cookie/CSRF behavior. Different preview hosts are not automatically trusted for signed-in testing.

The code-login failure shown on the user's phone was caused by a **stale running backend requiring `device_key`**, despite updated source/frontend. Restarting and publishing matching versions fixed it. `deploy_testing.py` now checks the running `/openapi.json` login contract before publishing. Do not bypass that guard or issue new codes to mask this mismatch.

Long analyses use upload/job creation plus authenticated polling, not SSE or a long synchronous request through the tunnel. Retry the same request identifier after a lost acknowledgement to avoid duplicate analysis. Jobs are user-scoped and temporary; saved results persist, unsaved jobs do not survive an API restart.

## 9. Collaboration and merging without losing work

The user and groupmate ultimately clarified that they work on **different laptops with separate copies**. OneDrive in a local path does not mean they edit the same working tree.

Each should use a separate feature branch. Save editor buffers and inspect/commit the intended changes before integration. Do not discard work, force-push, reset hard or run `git clean` as a routine conflict fix.

Example future workflow, substituting the actual agreed branch names:

```powershell
git status --short --branch
git switch -c feature/my-specific-fix
# Implement and verify, then stage only reviewed files.
git add <reviewed-file-paths>
git diff --cached --name-status
git diff --cached --check
git commit -m "Describe the completed change"
git push -u origin feature/my-specific-fix
```

For integration, begin with a clean saved state, fetch the actual remote, create an integration branch from `origin/Latest-Main`, merge the feature branch, inspect both sides and run relevant checks. Resolve conflicts by preserving both intended behavior changes, not by choosing an entire side blindly. Absence of text conflicts does not prove absence of logic regressions.

For the completed October 13 integration:

- Groupmate: `d989945`; user: `750b1f1`; common parent: `e021ff2`.
- The merge was prepared on `integration/planter-review-ui-20261013` and committed as `f158dca`, then advanced/pushed to `Latest-Main` with approval.
- There were no textual conflicts. Only `WebsiteRequests.jsx` and its CSS were edited by both; independent hunks combined the list/header improvements with the protected scrolling review dialog.
- Preservation checks found 10 user-only files and 21 groupmate-only files unchanged from their respective versions. Backend/core planter logic remained unchanged by the groupmate UI work.
- Groupmate changes included monitoring navigation/Today–Overdue queues, daily/weekly tide presentation, full calendar time ranges, pending requests positioning and modal accessibility.

Review code before staging; private `.env`, handovers, large assets, logs, backups and machine configuration are not commit material. Prefer explicit reviewed paths over an unchecked `git add -A`.

## 10. Verification commands and honest test status

Use existing tests that cover the feature and its domain risks. Avoid broad repeat runs after sufficient verification, and do not introduce implementation-mirroring tests for trivial reversible styling.

Frontend from the repository root:

```powershell
npm.cmd --prefix MangroVision_New/client run lint
npm.cmd --prefix MangroVision_New/client run build
git diff --check
```

Focused example from `MangroVision_New/client`:

```powershell
node --test src/components/Panel.test.js src/utils/analysisHistory.test.js
```

Broader frontend discovery from that same directory:

```powershell
$tests = @(rg --files src -g '*.test.js')
node --test --test-reporter=spec @tests
```

Tests use Node's built-in test runner; do not assume Vitest or `npm test` is configured. Some development tests are ignored by Git, while listed regression exceptions are tracked. Discover files actually present on the current laptop; do not treat a missing local-only test as a product failure.

For relevant PostgreSQL checks, first inspect fixtures and isolation, and use an appropriate development target. The documented LIKE fixture uses session-local temporary tables, actual migrations and an outer rollback. It must not be confused with arbitrary tests that mutate permanent tables:

```powershell
$env:MANGROVISION_RUN_TEMP_LIKE_TESTS = '1'
.\venv\Scripts\python.exe -m pytest tests/test_like_appointments_postgres.py tests/test_planter_codes_postgres.py tests/test_planter_activities.py tests/test_appointment_email_transport.py -q
```

Do not run an unreviewed entire Python suite against private shared database settings. These email tests mock delivery; real email needs separate authorization.

### Latest recorded outcomes, not a new test run for this handover

| Check | Recorded outcome |
| --- | --- |
| Analysis-history change | 51 existing Panel/history checks passed; lint and production build passed |
| Integration selected checks | 99 frontend checks passed; lint/build passed |
| Broad frontend audit at `f158dca` | 223 of 227 passed; four pre-existing stale expectations failed |
| Planter login regression | 27 PostgreSQL checks passed, including normalization cases; frontend checks and build passed |
| Real-code verification | Previously authorized two isolated sign-ins restored the same participant/progress; test sessions signed out |

The four broad failures also occurred at the common baseline `e021ff2`:

1. `components/RetainedRoutes.test.js` uses a tentative schedule fixture but expects a calendar card; unapproved activities are intentionally excluded.
2. The same file's planning-lock test targets an old title selector and the removed expected-seedlings field.
3. Its booking-approval test expects the old `Confirm and create schedule` button/notice; current UI uses `Confirm appointment` and shared feedback.
4. `pages/RestorationReports.test.js` supplies obsolete survival-cohort data/expectations instead of approved overall current-status health totals.

These tests need careful fixture/assertion updates. Do **not** revert calendar approval rules, re-add expected seedlings, or restore 30-day metrics to satisfy them. A broader API activity-catalog test was also previously documented as missing 11 existing routes in its catalog; investigate that separately before claiming every suite passes.

The production build has an existing large-chunk warning. It is not an authentication/database failure. Browser visual verification was unavailable for the pagination task; no live-browser claim should be added retrospectively.

## 11. Most recent feature: Analysis History pagination

Committed in **`7679e15`**. Files:

- `client/src/components/ImageAnalysisHistory.jsx`
- `client/src/components/ImageAnalysisHistory.css`

Behavior:

- Five analyses per page; Previous/Next, page count and visible result range.
- Apply search/date filters and sorting to the full loaded history before slicing.
- Changing filters/sort or resetting filters returns to page one.
- Opening/returning from an analysis preserves the page. Refresh preserves a valid page; deletion clamps it when the last page disappears.
- Reset the list's scroll position for page/filter changes without scrolling the whole workspace.
- Disable navigation during relevant loading/open/delete work. Keep accessible status/navigation labels and mobile wrapping.

This is **client-side pagination**; the API still loads the complete history. No API or database migration was required. A future server-pagination change must preserve filters/sort/count semantics across the full dataset and avoid returning misleading totals.

## 12. Remaining work and things not to assume

No new application change is requested by this handover task. These items are candidates for a later authorized task:

1. Update the four stale frontend test expectations without changing approved behavior.
2. Correct old shared-login wording in `client/src/guides/tours.js`, especially the field-login description and staff tour. Code-only sign-in is already implemented; this is help-copy cleanup.
3. Perform a visual browser check of Analysis History pagination when browser access is available.
4. Check actual live backend/frontend versions before claiming the latest Git commit is active. This handover did not restart, deploy or contact the public services.
5. Keep new feature docs consistent. Some historical documents still describe username/password or device recovery; `planter-code-access.md` and `planter-login-fix-2026-10-12.md` supersede those descriptions for planter sign-in.

Do not assume a clean merge proves complete behavioral equivalence, that all APIs are healthy because their keys exist, or that historical table/point counts remain current.

## 13. Documentation worth reading

| Topic | Document |
| --- | --- |
| New laptop setup | [groupmate-setup.md](groupmate-setup.md) |
| Remote testing / Vercel / tunnel | [user-testing.md](user-testing.md) |
| Database / backups / restores | [production-database.md](production-database.md) |
| LIKE website and booking workflow | [like-website.md](like-website.md) |
| Current individual access | [planter-code-access.md](planter-code-access.md) |
| Login bug and coordinated deployment | [planter-login-fix-2026-10-12.md](planter-login-fix-2026-10-12.md) |
| Activity locks | [activity-and-booking-protection.md](activity-and-booking-protection.md) |
| Evaluator feedback implementation | [evaluator-feedback-improvements-2026-10-11.md](evaluator-feedback-improvements-2026-10-11.md) |
| Current health counting | [current-seedling-health.md](current-seedling-health.md) |
| Tide interpretation | [tide-planting-guidance.md](tide-planting-guidance.md) |
| Assignment/routing rules | [zigzag-and-site-routing.md](zigzag-and-site-routing.md) |
| Image alignment | [image-alignment.md](image-alignment.md) |
| Canopy diagnostic review | [canopy-only-review-2026-10-06.md](canopy-only-review-2026-10-06.md) |
| Monitoring deaths / records | [monitoring-death-locations.md](monitoring-death-locations.md) |
| User help | [system-user-guides.md](system-user-guides.md) |
| Evaluator recording | [evaluator-recording-2026-10-09.md](evaluator-recording-2026-10-09.md) |

Ignored local files under `tmp/` include private planter-code/password handovers, email previews and audit output. They are not public documentation; do not open/print them just to reconstruct context. Old organization-password handovers are obsolete for signing in after the code conversion.

### Presentation and academic context

The requested explanation order is:

**Planting Map → Dashboard (Overview, Planting Work, Seedling Health, Project Sites) → Scheduling → Monitoring → Analyze Image → Planting Assignments → Zone Editor → Activity Logs.**

The user already recorded the video. If asked for narration, provide short generic explanations of each panel in that order, without screen actions, “Show” cues or timing instructions that require rerecording.

Immediately before this handover, the user requested the final activity in `CIT239_Unit5_Risk_Management.pdf` in Downloads. We reviewed the 27-slide PDF and answered slide 26 with a seven-risk MangroVision register, mitigation/contingency/avoid/accept decisions and proposed owners. Main risks covered database loss, AI/GSD/alignment, hosting/network outage, exposed credentials, team knowledge gaps, last-minute scope changes and minor cosmetic differences. Database loss was the proposed highest concern. The answer was in chat, not a committed academic file. The likelihood/impact ratings were qualitative estimates, not measured probabilities. The slide 9 matrix and slide 13 example have a priority inconsistency; the answer followed the matrix. No application changes were made for that academic review.

## 14. Starter message to paste into the next chat

```text
We are continuing MangroVision development. Read AGENTS.md and
docs/developer-chat-handoff-2026-10-14.md first, then inspect current Git status.

The current app is FastAPI + React. Our main
integration branch is Latest-Main. At the handover snapshot it was 7679e15,
with the planter/groupmate integration and Analysis History pagination included.

Preserve individual reusable planter codes, existing participant slots and
planting history, current-map dashboard counts, schedule/assignment protections,
and the approved monitoring, tide and booking rules. Never print or commit
private .env values or planter-code handovers. A feature branch does not isolate
the shared database. Git push does not restart the laptop backend.

Use the handover's verification notes; some old test expectations and tour copy
still need updates. Do not claim all suites passed or that Git HEAD is deployed
without checking. Continue the task I request next without resetting data or
undoing the approved behavior.
```
