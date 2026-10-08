# Organization planter accounts

Each organization registers once in `/field`, providing a shared username,
password, and participant count. Each device uses that login and receives a
persistent numbered participant slot. Logging out does not release the slot.

Registration records the first device as participant 1. The browser keeps a
random device key across logout in local storage and a one-year cookie backup;
the database stores only its hash against the organization and participant number.
Existing browser keys are retained. Returning through the same browser and field
link reuses that number. Point statuses and planting events live in the database,
so a new session restores the same points and progress. Ordinary return visits
use only the shared username and password, even after a login session expires:
the browser reuses its saved identity automatically. Recovery codes are optional
backup tools, separate from normal sign-in. Sign-in stops before
claiming a slot if neither browser storage nor cookies can persist the identity.

This tracks browser identities, not physical hardware. A different browser,
private browsing session, cleared site data, or a new shared-link domain can look
like a new device. In particular, a restarted Cloudflare quick tunnel can produce
a new domain, and that domain cannot read the previous link's browser storage.
Prefer one permanent field address; `MANGROVISION_PUBLIC_FRONTEND_URL` makes the
share panel reuse a configured hosted frontend. A cookie backup can restore lost
local storage on the same host, but cannot cross unrelated shared-link domains.

Before a field link changes, participants open their account menu and save their
private **Device recovery code**. On the next link, choose **I have a device recovery
code** and enter it with the organization's username and password. The code proves
which existing participant to resume; it does not replace the password. Incorrect
or reset codes never allocate another slot. After successful recovery, that
organization's identity is saved on the new link without changing identities
for other organizations. Keep each participant's code private and separate.

For an organization with 10 participants, assigning 100 points allocates
10 distinct points per participant. Points are stored with their
participant slot; completion never moves someone else's points. If the count
does not divide evenly, the extra points are distributed one at a time. Later
batches balance the existing active/completed allocations without moving them.

Staff select the organization and enter a point count in Planters. Available
points are selected automatically; individual map clicks do not assign points.
The selected points keep the species recorded on their image analyses. A site
containing both Bungalon and Rhizophora creates separate species batches in one
transaction; both species count toward the organization's participant allocation.
The panel shows the selected species counts before assignment. Missing species
must be recorded on the original analysis; selecting a different species cannot
override a point's recorded species. If any batch fails validation, the complete
selection is rolled back.

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

A project site can be deleted before any planting is recorded, including when
points are already assigned. Recorded planting events (including closed history),
planted point timestamps, mortality history, or completed assignment points block
deletion. Unplanted allocations are released and their batches archived when a
site is deleted; points return to planned, with coordinates, species, analyses,
and planning evidence preserved. A field participant cannot plant a released
point using a stale page. This prevents new planting under a deleted site.

For an explicitly requested organization reset with no remaining project site,
`scripts/reset_organization.py --organization-id ID --expected-name NAME` previews
the exact affected records. `--apply` saves and verifies a private backup, removes
the organization's accounts/sessions/assignments/planting history, and returns
its locations to planned. Other organizations' live points and records prevent
an unsafe reset; hashes of all records outside the captured scope must remain
unchanged before the transaction commits. Normal site deletion does not perform
this organization reset.

If the original browser identity and recovery code are both unavailable, staff
select the organization in Planting Assignments and open **Participant devices**.
The list shows occupied slots, each slot's last activity, and its assigned/planted
point counts. Device counts use occupied slots, not the number of login sessions.
Activity timestamps describe the slot's history, including earlier devices after
a reset; they do not identify a physical phone. No keys or session tokens are
returned to staff.

Identify the replaced or duplicate browser before resetting its participant
number. Confirm the reset, then select **I am replacing a device after an LGU
reset** on the replacement browser and enter that number. Reset revokes sessions
and invalidates the old recovery code while preserving assigned points and
planting history. A browser already bound to another participant is rejected
instead of silently resuming the wrong participant; staff must first reset that
known duplicate slot too. Normal sign-in automatically chooses a free slot.
Nothing automatically resets or merges old occupied slots.

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
Browser checks cover logout, a fresh app load, session expiry, cookie restoration,
blocked storage, recovery on a different domain, and preserving other organizations'
identities. PostgreSQL checks cover resumed planting progress on a new origin even
when all slots are full, invalid codes leaving free slots untouched, device
summaries, and revoked recovery codes after reset. Tests use temporary tables and
leave live accounts unchanged.
