# Shared planter access

**Temporary testing:** `enforceActivityAccessWindow` is `false` in
`MangroVision_New/client/src/like/bookingPolicy.json`. Valid activity codes work
before the scheduled start and after the scheduled finish, including codes
previously blocked only by the activity time. This applies to joining, returning
sign-in, existing sessions and planting updates. Revoked/replaced codes, cancelled
or completed activities, inactive accounts, participant ownership and normal
browser-session expiry still apply. A browser whose session expires can sign in
again with the same code and username.

To restore the scheduled window described below, set `enforceActivityAccessWindow`
to `true`, restart the backend and rebuild/redeploy the frontend. The separate
`enforcePlantingHours` flag controls the public 6 AM–5 PM booking restriction.
No migration, schedule extension or code regeneration is needed. Newly generated
emails and staff access instructions follow the flag; previously delivered
emails and downloaded handovers keep their earlier wording.

One organization code is sent to the activity's contact person, whether there
are 10 or 100 planters. It belongs to one confirmed tree-planting activity and
opens at its scheduled start, then closes at its scheduled finish. Rescheduling
uses the saved schedule; there is no timer starting at email delivery or each
participant's first visit. Cancelling/completing the activity or revoking its code
also stops existing sessions and planting updates. Records remain saved.

First time asks only for the organization code. Once it is verified, a welcome
screen inside Field asks the planter to choose a personal username (3–32 letters,
numbers, dots, underscores or hyphens). Code verification does not reserve a slot
or create a participant session. Saving the username rechecks the code and activity
window, then opens the field workspace. The username is unique within the
organization, ignoring letter case. Returning planter uses the group code and
that username. There is no password or PIN. This is group access with a claimed
participant identity, not two-factor authentication: a group member who knows
someone's username can use their profile. Browser remembering prevents ordinary
repeat signup; it cannot prevent deliberate signup through a new browser.

The browser saves a random private identity with a cookie backup. Normal tabs
share it, and logout keeps it. First-time signup from an already enrolled browser
redirects to its existing username. Returning sign-in never allocates another
slot. The activity participant cap counts distinct participants across historical
sessions, including logout and switching; it does not count tabs or logins.
Later bookings retain usernames and slots. Increasing a group's size adds slots
without moving existing points or progress.

Only one current session can update a participant. Another browser gets a
**Switch here** confirmation. Its short-lived ticket is tied to the code,
username, browser and current session. A changed session requires a fresh
confirmation. Successful switching revokes the old participant sessions;
logout from an old page cannot revoke the replacement. Every field write checks
the current session and scheduled window in the same transaction as its update,
using the participant lock also used for switching. Points are scoped to both
the participant and activity. The client checks sessions on focus and every
30 seconds; backend checks apply immediately to reads and updates.

Website and manually entered planting activities both email the one code and
instructions. Non-planting and cancellation emails omit access codes. Staff use
**Planting Assignments → Organization code** for the selected activity. Replacing
or revoking a code refreshes pending emails and ends that activity's sessions.
Already delivered emails retain their earlier contents. An email currently being
delivered blocks code changes. Credentials and browser keys stay out of logs,
public session responses and frontend credential storage.

## Existing participants

Migration `20261015_0019` only adds tables, columns, constraints and indexes. It
does not reset accounts, rename participants, move points, or issue live codes.
Existing anonymous slots with devices, sessions or planted progress cannot be
claimed as unused slots. An existing valid session or remembered legacy browser
identity proves its original slot. The old individual-code input is no longer
shown in the normal sign-in flow. Participants without a remembered legacy browser
identity or valid session should ask LGU staff for help restoring their existing
participant. The compatibility API still accepts a previous individual code with
the activity code and chosen username to name the original slot and keep its
progress. Old codes do not directly
authorize the new field workspace.

## Coordinated activation

The source change is not a live deployment. Before activation, preserve local
edits on the backend laptop and prepare a fresh verified encrypted backup as
described in [production database procedures](production-database.md). Keep the
existing `LIKE_EMAIL_ENCRYPTION_KEY`; all API/worker processes need the same key.

Approving a planting request and signing in check that the shared-access tables
and participant/session columns exist. An incomplete rollout returns an explicit
setup-required response instead of a generic server failure. Rejected approval
keeps the request pending and creates no schedule, organization, account, code
or email. This check reports the problem; it does not apply migrations at runtime.

1. Stop writers and the email workers while coordinating the cutover.
2. Confirm application and migration connections target the intended database;
   apply Alembic revision `20261015_0019` using the owner connection.
3. Preview existing future/ongoing activities with
   `python scripts/prepare_group_access.py`. Review its IDs and fingerprint.
4. Use `--prepare --expected-fingerprint <reviewed fingerprint>` with
   `--verified-backup backups/<verified archive>.enc` and
   `--encrypted-handover backups/<new handover>.enc`. It sends no mail. It reuses
   existing group codes, refreshes pending messages, and keeps revoked codes revoked.
   If a commit fails after the artifact is saved, compare it to staff controls
   before sharing it. Backup checksum/readability verification remains an
   operator step; the script checks the artifact exists, not its restoreability.
5. Restart the matching backend and publish the matching frontend together.
   `deploy_testing.py` rejects a backend missing code-only entry or username/browser/switch support.
6. Share the current activity's code with its contact person. Previously delivered
   individual-code emails are not automatically replaced. Verify the planned
   start/end, joining, logout/return, switched-session rejection and preserved points.

Tests use rollback-only temporary PostgreSQL tables and fictional contacts.
No verification sends real mail or modifies live credentials or planting records.


## Sign-in failure messages

Sign-in identifies an unknown code, a revoked code, an activity awaiting LGU
approval, cancellation, completion, and an unknown participant username.
When scheduled access is enabled, a future activity gives its exact opening
date and time in Philippine time; an ended activity explains that its code has
expired and retains saved progress. Permanent LGU access has no activity expiry.
The temporary testing flag above continues to control whether schedule dates
restrict access; messages never claim a schedule block while that flag is off.

First time verifies the code before asking for a personal username. Returning
planter requires a username already registered in that organization. A missing
username explains how to use First time; it never creates another participant
or moves saved progress automatically. A code that becomes invalid during
username setup returns to the code screen with the actual reason.

Database connection failures and incomplete server setup have separate,
retryable messages. Unknown server failures retain a general service message.
No SQL, connection addresses, keys, codes or browser identifiers are included
in public diagnostic messages or sign-in error logs. Retrying an unavailable
service keeps the entered code and does not require a page reload.
