# Planter sign-in server error, October 12, 2026

The code-login implementation updated `planters.last_login_at`, but the actual
database column is `planters.last_login`. A valid code could therefore fail with
a database column error and the API's generic HTTP 500 response. Readiness still
reported ready because database connectivity, migration revision and storage
were available. A read-only PostgreSQL check confirmed `UndefinedColumn` for
the old statement and successful compilation for the corrected statement.

The temporary PostgreSQL test fixture also used the incorrect column name,
which hid the mismatch. The login statement and fixture now both use
`last_login`; the login test verifies that the timestamp is saved.

The coordinated changes also let a planter use their individual code on any
browser or device while restoring the same participant slot and points. The
frontend no longer asks for device recovery. Code revocation/replacement and
sign-in limits remain available.

## Existing records and deployment

No accounts, codes, device bindings, passwords or planting records were reset.
No schema migration or new encryption key is required for this fix. Keep the
running backend's private `.env`, including its existing
`LIKE_EMAIL_ENCRYPTION_KEY`.

The public frontend and backend were still running the older version during
investigation. The public backend runs on the groupmate's laptop, not this
workspace's laptop. Applying local changes here does not update that process.

The fix is delivered through the `Latest-Main` branch. Pushing this branch
triggers Vercel's automatic frontend deployment. Vercel does not update or
restart the FastAPI process on the groupmate's laptop.

On the laptop currently running the public backend, preserve any local edits
and pull the updated branch:

```powershell
git status --short
git switch Latest-Main
git pull --ff-only origin Latest-Main
```

If Git reports local changes or a divergent branch, preserve those edits and
resolve the affected files instead of discarding changes. The key correction is in
`mangrovision_db/planter_codes.py`: use `last_login` in the login update.

Restart the backend using the existing testing setup:

```powershell
.\venv\Scripts\python.exe -X utf8 MangroVision_New/start_testing.py --deploy
```

The launcher stops the previous managed stack and starts the updated backend.
Restarting creates a new Quick Tunnel URL, so `--deploy` updates Vercel's routing
to that URL and publishes a frontend deployment. The Git-triggered frontend
deployment is sufficient while the existing tunnel URL stays the same. Keep the
launcher terminal open. After deployment, refresh `/field`; it should say the
code works on any browser or device and have no device-recovery form.

Sign in with an existing participant code, check the same participant number
and saved planting progress, sign out, and repeat in another browser. Check the
backend log if a request still fails; keep credentials and code values private.

## Validation

All 16 code-access PostgreSQL tests passed after correcting the fixture. They
use temporary tables and an outer rollback. Coverage includes returning to the
same participant after repeated sign-ins, a fresh browser for every participant,
all ten occupied slots, preserved planting points, invalid/revoked codes,
inactive accounts, throttling and transaction rollback. Frontend authentication
and page tests, ESLint and the production build also passed.

The public JavaScript still contained both "Return using the same browser" and
"Using a different browser or field link" before the branch update. The backend
fix still needs to be pulled and restarted on the running laptop even after
Vercel finishes its automatic frontend deployment.
