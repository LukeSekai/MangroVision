# Evaluation feedback: forms, schedule updates, and planter tides

## Included changes

- Shared field corrections and form messages identify invalid values, announce saving and success, and explain connection failures. Booking forms preserve entered values and reuse their submission key after a failed response.
- LIKE public intake and staff request review use the same inline validation as staff forms. Monitoring, point assignment, zone saving, planting records, death-location corrections, replacement assignments, and tide calibration use shared feedback where they save information.
- Staff activities have an optional **Organization notification email**. Confirming, rescheduling, changing the planting area, or cancelling a confirmed activity queues its current details for that address. A single valid email in a legacy Contact field remains a fallback.
- Website bookings retain their existing encrypted confirmation/update queue, including their existing account access behavior. Staff activity emails use a separate encrypted queue and the same Brevo/SMTP transport.
- The planter **Your activities** panel shows only the signed-in organization's confirmed, ongoing, or cancelled activities, capped at 50. It includes future activities and activities updated during the preceding 30 days. It refreshes every minute, on window focus, and on request. Draft activities and staff contact details are excluded.
- Reschedule notices include the previous time from the saved activity log. Current schedule details and cancellations remain visible even when no notification address is configured or email delivery is delayed.

## Tide interpretation

The planter panel uses the same seven-day coastal forecast and estimated reference as staff Scheduling. **High tide peaks** are times of maximum water level. The **estimated elevated-water period** is the interval above the staff graph's reference, with crossings interpolated between available hourly samples. It is not a claimed duration of the peak itself.

The panel shows periods that overlap the activity, including their estimated duration. A period that extends past the available series boundary is labeled **at least**, because its complete duration is unknown. Stale readings, missing hourly coverage, or activities outside forecast coverage produce an unavailable advisory rather than an invented duration. Source and fetched time are displayed. The forecast-derived reference is not a surveyed site elevation; participants should confirm actual conditions with staff.

## Delivery and record protection

- Saving the activity and queueing its email happen in one transaction.
- Repeated saves with unchanged details do not queue another message. An unsent message is replaced with current details after another change.
- Delivery failures return to the queue with backoff; only the exception type is saved. The body is encrypted while pending and cleared after provider acceptance. Provider acceptance does not prove that the recipient read the email.
- Updating a message already being delivered returns a conflict with instructions to retry. Delivery is at least once: a process failure after provider acceptance can cause a retry.
- Staff can see **Queued**, **Sending**, **Sent**, or **Delivery failed; queued for automatic retry** in activity details.
- Missing recipients produce an explicit saved-but-not-emailed notice. Adding a notification address queues current details, including a cancellation.
- Confirmed activities retain their organization and are cancelled instead of deleted. Existing protection of activities with recorded planting remains in place.
- The existing `LIKE_EMAIL_ENCRYPTION_KEY`, sender credentials, and public website email configuration are reused. No new provider credentials are required.

## Database activation

Migration `20261011_0016` adds `planting_schedules.notification_email` and the private `activity_schedule_emails` table. It does not backfill messages or email existing organizations on upgrade. New notices start when a relevant activity is confirmed or edited after activation.

Shared database activation is separate from feature development, as required by [LIKE website setup](like-website.md). Before applying this migration, create and verify an encrypted production backup using [production database procedures](production-database.md). Apply only the new migration to the verified project, then restart the backend and publish the matching frontend together. Without migration 0016, the updated schedule queries cannot run.

### Activation on October 11

The user approved backup, migration, and activation. The fresh encrypted archive is
`backups/mangrovision-20261011T153341Z.tar.enc`. Its database checksum and all 136
private-object checksums were verified, and the custom database dump is readable.
A full restore was not performed. Both database connections were verified to
target the same database before migration. Migration `20261011_0016` was applied;
counts and content fingerprints of all 30 existing tables were preserved,
excluding the new notification-email field. The new activity outbox is empty.
The testing backend was restarted and the matching frontend published to the
existing Vercel production URL. Readiness, `/field`, and `/landing-page` return
HTTP 200; the organization-activities endpoint rejects anonymous access with
HTTP 401. These checks do not replace a signed-in field visit.

## Verification

Temporary PostgreSQL tests apply the actual migrations to session-local tables and roll back all changes. They cover staff and website notifications, retries, cancellation, no recipient, active delivery conflicts, organization scoping, and existing planting protections. Email delivery is simulated; no real messages are sent.

```powershell
$env:MANGROVISION_RUN_TEMP_LIKE_TESTS = '1'
.\venv\Scripts\python.exe -m pytest tests/test_like_appointments_postgres.py tests/test_like_appointments.py tests/test_planter_activities.py tests/test_appointment_email_transport.py -q
```

Frontend interaction tests cover booking field errors, preserved inputs and retry keys, revised schedules and cancellations, tide coverage and duration, and shared accessible feedback. Run from `MangroVision_New/client`:

```powershell
node --test src/components/ActivityAndBooking.test.js src/components/FormFeedback.test.js src/like/booking.test.js src/utils/formValidation.test.js src/utils/activityTides.test.js src/stores/planterAuthStore.test.js
npm run build
```

A separate, broader API activity-catalog test reports 11 existing routes missing from its catalog. Those routes and the catalog are unchanged by this feature. The focused notification, authorization, form, and tide checks pass. No real email delivery or shared database migration is performed by these verification commands.

## Email design and organization access

The follow-up email change adds the shared MangroVision HTML design to LIKE and
staff activity confirmations, updates, and cancellations. Plain-text alternatives
remain available. Dynamic values are HTML-escaped and buttons accept only HTTP
or HTTPS URLs. Cancelled activities omit credentials and the planter button.
Pending email content remains encrypted, and both versions are purged after
provider acceptance. Already delivered emails are not changed or resent.

New LIKE planting accounts use readable organization-based usernames with a
collision suffix when needed. Passwords start with the organization name and
contain readable words plus independently generated digits. Existing accounts,
passwords, device allocations, and saved planting points are retained.

Public planter registration is closed in the UI and API. An older client receives
HTTP 410 with instructions to use the confirmed group login. The staff assignment
panel offers **Organization access** for groups without a login, using the existing
staff-authorized account-creation endpoint. Website approval still creates planting
accounts automatically. Sign-in and participant-device recovery remain available.

Verification: 120 focused backend checks (including temporary PostgreSQL tests),
38 frontend checks, changed-file ESLint, and the production build pass. Email
delivery is mocked; no real test email is sent. Browser visual verification could
not run because its connection failed. A local HTML preview with fictional details
is available at `tmp/email-preview/confirmation.html`.

The backend was restarted and the matching frontend published to the existing
testing site. The deployed field bundle contains the revised sign-in screen and
no registration request. The old sign-up endpoint returns HTTP 410. Public
readiness, the landing page, and Dashboard return HTTP 200. No additional
migration or existing credential reset was needed for this follow-up.
