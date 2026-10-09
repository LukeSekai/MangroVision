# LIKE website and appointment requests

The public LIKE website is a separate React entry at `/landing-page`. It uses the
same frontend package and FastAPI backend as MangroVision, with its own CSS and
no staff workspace, staff map, or chart imports. Visiting `/` or `/index.html` opens
the MangroVision workspace. Staff can also use `/dashboard`; Map Analytics is at
`/map`. Staff-assisted scheduling remains available. The older `/like.html` URL
continues to open the public website.

## Preview the website

From the repository root:

```powershell
npm --prefix MangroVision_New/client run dev:like
```

Open `http://127.0.0.1:5174/landing-page`. This previews the website. Appointment
submission needs the FastAPI server on port 8000 and the new migration installed
in a separate development database. The frontend never simulates a successful
submission: unavailable API/database errors appear in the form.

The normal managed development stack serves MangroVision at
`http://localhost:5173/` and LIKE at `http://localhost:5173/landing-page`.
Use the existing launcher after configuring a separate development database;
do not start a second API process on an occupied port.

## Public map viewer

The **View map** buttons in the navigation and main banner open an interactive
map dialog without a login. It shows the same Google satellite background and
orthophoto as the staff map, using the shared bounds, native zoom limit, and
`VITE_TILE_SERVER` / `VITE_TILESET_PATH` configuration. Visitors can pan, zoom,
and reset the view. There are no planting points, zones, filters, or editing
controls, and opening the map makes no staff/account/database API requests.

The mapping library loads only when the dialog opens. Closing it preserves the
visitor's scroll position and appointment form; Escape and the close button both
work. Loading failures show a Retry button.

For local development and `vite preview`, Vite serves the exported PNG tiles
directly from `MAP/FINAL` at `/tiles/FINAL/{z}/{x}/{y}.png`. The map therefore
works in the website-only preview with FastAPI stopped, provided that the tile
pyramid is present. Restart an existing preview after changing Vite configuration.
Only numbered PNG tile paths are served; other files are not exposed.

Missing local tiles fall through to the existing backend proxy. A laptop with
only `final_orthophoto.tif` needs the backend to render those tiles. Production
hosting still needs the tile backend or a public tile host with the exported
`MAP/FINAL` pyramid (or the configured tile URL). This feature needs no database
migration, email settings, or private `.env` changes. The existing orthophoto
asset is reused; no map data is copied into the frontend build.

The viewer is implemented in `client/src/like/LikeMapDialog.jsx` and
`client/src/like/publicMap.js`; its styles are in `LikeMapDialog.css`.

## Database and staff workflow

Migration `20261006_0013` adds private request and rate-limit tables and a schedule
source column. `20261007_0014` adds appointment types and a private confirmation
email queue. Existing schedules default to `staff` and `tree_planting`.
The migrations are included as code only; they must not be applied to the shared database during feature
development. Branches do not isolate database state.

Configure a disposable development PostgreSQL/PostGIS database as described in
[production-database.md](production-database.md), with both `DATABASE_URL` and
`MIGRATION_DATABASE_URL` pointing to that development environment. Keep all
credentials in private environment configuration. Only after verifying that
target, run Alembic there and start the managed development stack:

```powershell
.\venv\Scripts\python.exe -m alembic upgrade head
.\venv\Scripts\python.exe MangroVision_New/start_dev.py
```

### Missing booking tables after merging

If `/api/like-appointments/summary` reports that `like_appointment_requests`
does not exist, the code has been updated but the booking migrations have not
been applied to the database used by FastAPI.

For an existing database, create and verify an
[encrypted backup](production-database.md#encrypted-backup-and-restore) first.
Set `MIGRATION_DATABASE_URL` in the private root `.env` to the owner/admin
connection for the same database as `DATABASE_URL`. Keep `DATABASE_URL` on the
restricted application account. If the migration URL is missing, the tools
fall back to the application connection, which cannot change the schema and
can cause `pg_dump` to fail on row-level security.

Once the backup is verified, apply the booking and activity-protection migrations
required by this version and restart FastAPI:

```powershell
.\venv\Scripts\python.exe -m alembic upgrade 20261009_0015
```

These migrations create three booking tables and add schedule fields, the
assignment activity link, and the booking email kind. Existing schedules default
to `staff` and `tree_planting`. Confirmation emails additionally need the private
settings described below. See [activity protection and booking updates](activity-and-booking-protection.md)
for the latest migration's safeguards and activation checks.

The required public appointment choices are **Field visit**, **Clean-up drive**,
and **Tree planting**. Organization, contact person, email, phone, date, time,
participant count, and consent are required. Email syntax is validated; this
does not prove mailbox ownership or guarantee delivery. LGU staff confirm the
contact address while coordinating with the requester.

The LGU/admin dashboard shows a compact appointment button with a pending count,
refreshed every 30 seconds and on focus. Clicking it opens
`/scheduling?requests=pending`, selects the pending filter, and scrolls to the
Website Requests panel. Staff also receive a notification linking to that panel.
Reviewing a request clears its pending count and marks that notification read.

Website requests appear on the Scheduling calendar and in the Website Requests
panel. That panel refreshes every minute and has a manual refresh button. All
pending requests remain visible; the history includes the latest 200 reviewed
requests. Staff can review tide advice and overlapping schedules, agree on a new
time, and select an existing organization or register one on confirmation.
Confirmation requires acknowledging that the requester was contacted.

Approval creates one confirmed schedule and links it to the request in
the same transaction as queuing the confirmation email. Repeated approval cannot
create another schedule, account, or queued email. Tree planting creates a shared
organization planter account if one does not exist, or completes a reserved
account. Field visits and clean-up drives create no planter account and cannot
be assigned a planting area. Planting area assignment and later schedule changes
use the existing staff workflow.
Declined/cancelled pending requests require a reason and create no schedule.
Confirmed website schedules are retained and can be cancelled before planting
starts. Deletion is blocked so their request and email history remain connected.

## Confirmation email configuration

Install the updated Python dependencies, then configure these **server-only**
values in the private `.env` for the development environment:

- `BREVO_API_KEY` and `SMTP_FROM`: use a Brevo API key and a verified sender
  for confirmation delivery over HTTPS. This works with the web port when SMTP
  connections are blocked. Copy an API key from Brevo's **SMTP & API → API keys**
  page and keep it in the private root `.env`. The sender uses the
  [Brevo transactional email API](https://developers.brevo.com/reference/send-transac-email).
- When `BREVO_API_KEY` is empty, confirmations use the existing `SMTP_HOST`,
  `SMTP_PORT`, `SMTP_FROM`, `SMTP_USERNAME`, and `SMTP_PASSWORD` settings.
  Cloud verification-email settings are separate from booking confirmations.
- `LIKE_EMAIL_ENCRYPTION_KEY`: a dedicated, stable Fernet key for the email queue.
  Generate it locally with the following command and copy its output into the
  private `.env`. Never commit it or prefix it with `VITE_`.
- `MANGROVISION_PUBLIC_URL`: the frontend's reachable base URL, for example
  `http://localhost:5173` during local development or the public HTTPS domain when
  hosted. Planting emails link to this URL plus `/field`. A localhost link only
  works on the computer running MangroVision; use a reachable address for planters
  opening the email on other devices.

```powershell
.\venv\Scripts\python.exe -m pip install -r requirements.txt
.\venv\Scripts\python.exe -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Keep the same encryption key in every API/worker process and across restarts.
Changing it while messages are queued makes those messages unreadable.
Missing encryption configuration blocks approval with a clear error and rolls
back account/schedule creation. Planting approval also requires the public URL.
Email-provider downtime leaves an approved appointment queued for retry.

After approval, FastAPI attempts delivery in the background. While the API is
running, a worker retries due messages every minute, with backoff after failures.
An optional one-shot worker is also available:

```powershell
.\venv\Scripts\python.exe scripts/send_like_confirmations.py
```

The email includes the reference, organization, appointment type, agreed date and
time in Philippine time, and participant count. Tree planting also includes the
planter URL and username. Newly created accounts receive a random password;
existing accounts keep their password and participant-device allocations, and
receive instructions to use their existing password. Staff can manage existing
account access through the current organization-account workflow.

Passwords are hashed in account records. Queued message contents are encrypted,
never returned by the API, and erased after successful provider submission. Scheduling
shows whether the email is queued, awaiting retry, or sent. Provider acceptance does
not guarantee inbox delivery. Delivery retries can resend an email if a process
stops after sending but before recording success; the schedule and account remain
unique. Cancelled schedules only deliver a queued cancellation notice; removed
schedules are excluded from delivery.
Emails use the agreed schedule details. Before planting starts, revising the
date, time, title, or participant count refreshes an unsent confirmation or queues
an update after earlier delivery. Eligible cancellations queue a cancellation
notice. Staff still coordinate these changes with the organization.

## API and hosting

- `POST /api/public/like/appointments`: anonymous intake; returns only a reference,
  pending receipt status, and Philippine timezone. It cannot accept schedule
  status, registered organization IDs, or privileged fields.
- `GET /api/like-appointments`: authenticated `admin`, `lgu`, or `planner` access.
- `GET /api/like-appointments/summary`: the same staff roles; returns only the
  pending count and Scheduling link.
- `POST /api/like-appointments/{id}/review`: staff approval, decline, or cancellation.

Public submissions carry a UUID submission key. Identical retries reuse the
original reference; changing the payload under the same key returns 409.
The server rejects past start times, invalid or overnight time windows, invalid
contacts, and participant counts outside 1–10,000. Intake uses a honeypot and
database-backed limits of five successful requests per client IP per UTC hour
and 100 per minute globally. Identical retries do not consume another slot.
Client IPs are hashed with the hourly bucket and are not stored as plain text.

Build both entry points with `npm --prefix MangroVision_New/client run build`.
Host both HTML entry points and the generated assets with a same-origin `/api`
proxy to FastAPI. Rewrite `/landing-page` (including its trailing-slash variant)
to `like.html` before the staff SPA fallback. Vite development, Vite preview, and
the included Vercel configuration already provide this rewrite. The separate
preview port uses the existing Vite proxy. If hosting the
website on a separate origin, set its API base at build time and add that exact
origin to the backend's trusted-origin and CORS configuration. Keep staff cookie
and CSRF protection in place. Public submission omits session cookies.

Configure trusted reverse proxies so the API receives the real client IP.
Otherwise visitors behind one proxy can share the same rate-limit bucket. Never
trust arbitrary forwarded client-IP headers. Database credentials stay server-side;
the request tables have RLS enabled and no `anon` or `authenticated` access.

## Photos and remaining content

Public content lives in `client/src/like/LikeWebsite.jsx`; its gradients and
animations live in `client/src/like/like.css`. The five photos are connected from
`MangroVision_New/client/public/like/`:

| File | Website section |
| --- | --- |
| `hero.jpg` | Main banner |
| `about.jpg` | About LIKE |
| `planting.jpg` | Plant with purpose |
| `mangroves.jpg` | Learn from the coast |
| `community.jpg` | Grow as a community |

Replace an image using the same filename to update that section. These public
photos are included in Git and copied into production builds. Images use
descriptive alternative text and crop to fill their frames; the main banner
loads immediately and the other photos load as visitors approach them.

The Past activities section replaces the former staff profile. Its four photos
are connected from the same folder, with titles taken from their filenames
without the `.jpg` extension:

- `NGO Love Our Own Brethren (LOOB) Inc. Field Visit.jpg`
- `University of the Philippines Visayas Field Visit.jpg`
- `UPV IFPDS Field Visit.jpg`
- `ZSL - Mangrove Caravan.jpg`

Activity images show the entire photo, including group members and event text.
Edit `pastActivities` in `LikeWebsite.jsx` with each activity's `date`
(YYYY-MM-DD), short recap, `photo` filename, and descriptive `alt` text. Titles
automatically follow the configured filenames. Use three or four entries.
The recaps are shortened versions of the LIKE Facebook captions supplied for
these activities. Dates remain placeholders until supplied. Leave `photo` and
`date` as `null` to keep their placeholders. Do not present example events or
dates as actual activity history.

Supply official contact information, hours, visitor guidelines, and approved
activity content before publishing.
Animations honor the visitor's reduced-motion preference.

## Verification

```powershell
.\venv\Scripts\python.exe -m pytest tests/test_like_appointments.py tests/test_appointment_email_transport.py -q
node --test MangroVision_New/client/src/like/booking.test.js
```

Real PostgreSQL checks are opt-in. They create only session-local temporary tables,
apply the migration to those temporary objects, and roll back every test. They do
not modify live records, permanent schemas, or Alembic migration history:

```powershell
$env:MANGROVISION_RUN_TEMP_LIKE_TESTS = '1'
.\venv\Scripts\python.exe -m pytest tests/test_like_appointments_postgres.py -q
```

The suite checks intake privacy, validation, LGU permissions, duplicate retries,
atomic/idempotent approval, account creation only for planting, preserved existing
accounts, encrypted email retries and purging, corrected email addresses,
notifications/counts, rollback, decline/cancellation, rate limits, database
client-role access, and existing manual schedule creation. Email delivery is
mocked; these tests never send real email.
