# LIKE website and appointment requests

The public LIKE website is a separate React entry at `/like.html`. It uses the
same frontend package and FastAPI backend as MangroVision, with its own CSS and
no staff workspace, map, or chart imports. The normal `/` entry and staff-assisted
scheduling remain available.

## Preview the website

From the repository root:

```powershell
npm --prefix MangroVision_New/client run dev:like
```

Open `http://127.0.0.1:5174/like.html`. This previews the website. Appointment
submission needs the FastAPI server on port 8000 and the new migration installed
in a separate development database. The frontend never simulates a successful
submission: unavailable API/database errors appear in the form.

The normal managed development stack also serves `/like.html` on port 5173.
Use the existing launcher after configuring a separate development database;
do not start a second API process on an occupied port.

## Database and staff workflow

Migration `20261006_0013` adds private request and rate-limit tables and a schedule
source column. Existing schedules default to `staff`. The migration is included
as code only; it must not be applied to the shared database during feature
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

Website requests appear on the Scheduling calendar and in the Website Requests
panel. That panel refreshes every minute and has a manual refresh button. All
pending requests remain visible; the history includes the latest 200 reviewed
requests. Staff can review tide advice and overlapping schedules, agree on a new
time, and select an existing organization or register one on confirmation.
Confirmation requires acknowledging that the requester was contacted.

Approval creates one confirmed planting schedule and links it to the request in
the same transaction. Repeated approval cannot create another schedule. Planting
area assignment and later schedule changes use the existing staff workflow.
Declined/cancelled pending requests require a reason and create no schedule.
Deleting a linked schedule retains the request history and does not allow a
repeated approval to recreate the deleted schedule.

## API and hosting

- `POST /api/public/like/appointments`: anonymous intake; returns only a reference,
  pending receipt status, and Philippine timezone. It cannot accept schedule
  status, registered organization IDs, or privileged fields.
- `GET /api/like-appointments`: authenticated `admin`, `lgu`, or `planner` access.
- `POST /api/like-appointments/{id}/review`: staff approval, decline, or cancellation.

Public submissions carry a UUID submission key. Identical retries reuse the
original reference; changing the payload under the same key returns 409.
The server rejects past start times, invalid or overnight time windows, invalid
contacts, and participant counts outside 1–10,000. Intake uses a honeypot and
database-backed limits of five successful requests per client IP per UTC hour
and 100 per minute globally. Identical retries do not consume another slot.
Client IPs are hashed with the hourly bucket and are not stored as plain text.

Build both entry points with `npm --prefix MangroVision_New/client run build`.
Host `like.html` and the generated assets with a same-origin `/api` proxy to
FastAPI. The separate preview port uses the existing Vite proxy. If hosting the
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
.\venv\Scripts\python.exe -m pytest tests/test_like_appointments.py -q
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
atomic/idempotent approval, rollback, decline/cancellation, rate limits, database
client-role access, and existing manual schedule creation.
