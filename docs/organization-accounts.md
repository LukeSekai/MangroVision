# Organization planter accounts

Each organization registers once in `/field`, providing a shared username,
password, and participant count. Each device uses that login and receives a
persistent numbered participant slot. Logging out does not release the slot.

Registration records the first device as participant 1. The browser keeps a
random device key across logout, and the database stores its hash against the
organization account and participant number. Returning through the same browser
and field-link origin reuses that number. Point statuses and planting events are
stored in the database, so a new session restores the same points and progress.

For an organization with 10 participants, assigning 100 points creates one
organization assignment with 10 distinct points per participant. Points are stored with their
participant slot; completion never moves someone else's points. If the count
does not divide evenly, the extra points are distributed one at a time. Later
batches balance the existing active/completed allocations without moving them.

Staff select the organization and enter a point count in Planters. Available
points are selected automatically; individual map clicks do not assign points.

Quick Assign includes organizations that have not registered a field account.
Their points are reserved under an organization record with no login credentials.
When that organization registers, the same record receives its credentials and
the reserved points are divided using the participant count supplied at signup.
No participant device can claim a slot before registration. Existing registered
accounts keep their original allocations.

An image's project-site link does not make all of its points belong to that
site. Quick Assign counts and validates individual locations inside the saved
Zone Editor boundary, including points on the boundary. If a point lies outside
its image's linked site, the unique boundary covering the point determines its
organization. One image can therefore supply points to NASUGBAN and OTON. A
covering image link resolves overlapping boundaries; other overlaps remain
unassigned until their ownership is clarified. The map, site point counts,
assignment validation, and deletion checks share this ownership rule.
The organization's first mapped project site is selected and located
automatically; organizations with multiple sites can
switch between their own sites. The map, field app, and activity report use the
same organization color. The activity report and monitoring retain organization
totals. There is no individual assignment action or roster.

If a device is replaced or its browser storage is cleared, staff select the
organization and open **Participant device recovery**. Reset the participant's
number, then select **I am replacing a device after an LGU reset** on the
replacement device and enter that number. Normal sign-in needs only the shared
username and password and automatically chooses a free participant slot. Reset
revokes existing sessions for that slot and preserves its assigned points.

## Existing-account conversion

Alembic revision `20260912_0006` retains the oldest active account per organization
(or the oldest account if none is active). Its username and password stay the
same, and its display name becomes the organization's name. Other accounts are
retained as inactive audit records linked to the surviving login. Assignments,
planting events, and mortality ownership are consolidated under that login;
point and event counts are preserved. The initial participant count equals the
number of legacy accounts. Existing planter sessions must sign in again.

New organization registrations supply their actual participant count. The
participant count is fixed for the account so existing allocations cannot
silently move between devices.

Before applying the conversion, save the affected ownership and account fields
with `scripts/backup_organization_accounts.py`. Then run `python -m alembic
upgrade head` with the application's Python environment. The migration is
transactional. Downgrade deliberately requires restoring the pre-conversion
data rather than guessing former ownership.

## Shared-link login and device limits

The organization participant count is the device limit. A new device takes a
free slot; logging in again on the same device resumes that slot. An occupied
participant number from an older form cannot block normal sign-in while other
slots are free. Explicit device recovery requires the chosen slot to be reset
first, so a replacement phone receives its original points.

Field authentication uses the application's `mangrovision.auth_sessions`
table and organization device slots. It does not use Supabase Auth's
single-session settings. The request-origin check accepts the site's own LAN
or shared-link origin and configured trusted frontend origins. Cross-origin
requests remain blocked, and authenticated mutations still require CSRF tokens.

Regression coverage includes ten separate browser cookie jars signing in with
one username/password, all ten sessions remaining valid, and an eleventh device
being rejected. The registration-to-logout-to-login check reserves 100 points,
registers 10 participants, records three planted points for the first device,
and verifies that its original 10 points and progress return after sign-in.
Browser checks cover device identity after logout, a fresh app load, and session
expiry. Tests use temporary tables and leave live accounts unchanged.
