# LGU Sites

LGU is one fixed system organization. Its sites remain separate project-site
records with their own planting cycles, survival results, and monitoring
histories. Organization-level dashboard results combine all LGU seedlings
using total counts; four LGU sites still count as four sites and one organization.

Staff open **Monitoring -> LGU Sites** to see combined results and site cards.
**Assign LGU points** opens Assign Points with LGU selected. Click or brush
available planned locations and assign them; their project site is created
automatically. There is no separate site creation step or site selector.
LGU does not need a timed booking to plant. Recorded planting starts each
seedling's history and monitoring cadence.

## Permanent planter access

The staff-only credentials panel shows the planter username and a revealable
organization code. The account and code have no activity expiration. Existing
session expiry, sign-out, and explicit browser switching still work normally.
Signing in again uses the same permanent credentials and keeps saved progress.
Choose **Returning planter** in `/field`. First-time code entry also recognizes
the LGU account and fills its username without creating another participant.

The code is encrypted at rest and never included in public organization lists,
planting data, activity logs, or source files. The credentials response uses
`private, no-store`. The fixed organization cannot be renamed or removed, and
its planter account cannot be deactivated, deleted, or moved to another owner.
Ordinary organizations continue using activity-scoped codes and participant
usernames. Public booking intake cannot register as the reserved LGU organization.

## Monitoring

Each planting batch starts its own 14-day cadence from actual planting.
Weekend due dates move to Monday. A site's next date is its earliest outstanding
batch check. Late visits do not move the planting anchor. Later plantings, even
when entered with an earlier actual planting date, are not treated as inspected
by a visit whose saved snapshot did not contain those seedlings.

Open **Record monitoring** on a due site. Select newly dead seedlings on its
map, select their cause, record overall health and LGU actions, and optionally
enter an actual measured average height. An empty death selection means zero
new deaths. All LGU deaths in this form have known locations. Saving updates
only that site, writes its death records and applicable individual inspection
observations, and refreshes map and reporting data. Other sites retain their
dates and results. History identifies the recording staff member.

The original planting and death history stays with a site after locations are
released. Later planting under another organization's site has its own results.
An LGU site visit's documented deaths cannot be erased by restoring a map point.

## Installation

The project owner applies Alembic revision `20261019_0020`, then runs:

```powershell
.\venv\Scripts\python.exe scripts/setup_lgu_sites.py
```

The initializer is repeatable and does not rotate existing credentials. It
prints no code or private configuration. Groupmates using the same shared
database need the updated application code; they do not repeat the migration.

Backend checks use temporary tables and an outer rollback:

```powershell
$env:MANGROVISION_RUN_TEMP_ACCOUNT_TESTS='1'
.\venv\Scripts\python.exe -m pytest tests/test_lgu_sites_postgres.py tests/test_lgu_sites_api.py -q --tb=short
```

From `MangroVision_New/client`:

```powershell
node --test src/pages/LGUSites.test.js src/stores/planterAuthStore.test.js
```
