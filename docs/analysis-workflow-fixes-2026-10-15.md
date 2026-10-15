# Analysis workflow fixes — October 15, 2026

Prepared on `fix/analysis-workflow-20261015` from `266daa9` in the isolated
`tmp/worktrees/analysis-workflow-20261015` checkout. The deployed checkout is
unchanged by this work.

- Removed the post-analysis Review analysis and Assign available points shortcut
  cards. The existing result review button, history and assignment tool remain.
- View area in map closes the saved-result dialog and focuses its saved boundary
  while keeping Analyze Image selected. Browser Back still restores the history
  list with its filters.
- History, analysis details, save responses, area advisories and analysis statistics
  display Analysis 1 through Analysis N in retained chronological order. Deleting
  a result closes numbering gaps at the next read. Database IDs, the original
  sequence, source filenames, point links and historical records remain intact.
- The result warning now checks the final GIS-clipped image boundary instead of
  a 15-metre search around its center. Disjoint areas and edge-only contacts are
  excluded; current points on the actual boundary count. Existing duplicate,
  spacing and repeat-image protections still apply.
- Refresh reconnects to the existing owner-authorized processing job. A per-tab,
  per-user reference is saved before uploading, so a lost upload acknowledgement
  can be recovered by its original request ID. Recovery only reads job endpoints;
  it never reuploads the image. Completed unsaved results are recovered too.
- Logout and account changes stop old UI updates. Saving or discarding clears the
  saved reference. No image bytes, result payloads or credentials are persisted
  in browser storage.

The backend's existing limits still apply: one running analysis, at most 24
retained completed jobs, and one-hour retention. Refresh recovery requires the
same running backend. Restarted or expired jobs show an actionable error rather
than an indefinitely running indicator. An upload interrupted before the server
accepted it must be selected again.

Validation passed 35 backend and 87 frontend checks using temporary SQLite/GEOS
fixtures, mocked processing jobs and real React components under JSDOM. Lint and
the production build passed, with the existing large-bundle warning. A separate
database-enforced read-only check found 35 retained analyses and verified the
latest displayed label is Analysis 35. No live image jobs, planting changes,
emails, migrations or storage writes were performed for verification.

Deploy the matching backend and frontend together after integration; the new
frontend uses `GET /api/analyses/jobs/current` for recovery.

## Protected analysis deletion

History replaces Delete with **Locked—contains field records** whenever an
analysis has assigned points, planting/death evidence, or historical assignment,
planting or mortality links. Released assignments, closed planting cycles and
soft-deleted points with that history remain protected. A point marked Planted,
Dead or Skipped is protected even if an older import lacks a history link.
Only unused, unassigned Planned points are eligible; unknown statuses fail closed.
The protected analysis remains available for viewing and exporting.

History, details, save responses and map statistics use one SQL policy to return
`can_delete`, `deletion_locked` and `delete_lock_reason`. A missing eligibility
result requires Refresh instead of offering Delete. If points become used after
the history loaded, the API returns 409 and the UI returns to refreshed history
with the lock explanation.

The deletion transaction locks the analysis row, then its points in ascending
ID order. It rechecks the policy in a fresh statement after obtaining the locks,
before queuing image cleanup or deleting anything. These locks conflict with
field updates and foreign-key inserts. A protected request rolls back without
starting object cleanup. No application migration is required.

Follow-up validation passed 69 backend checks (including seven real PostgreSQL
race cases) and 73 frontend checks, plus lint and production build. The race
checks observed actual database lock waits: a committed planting/death update
or history insertion blocked deletion; when unused deletion won first, competing
assignment and new-point inserts could not commit against the deleted records.

PostgreSQL race checks are opt-in with
`MANGROVISION_RUN_ANALYSIS_DELETE_PG_TESTS=1`. They use only freshly created,
UUID-named fixture schemas and drop them afterwards. Each transaction verifies
its fixture search path, with no application-schema fallback. They send no mail
and never run live image cleanup. A separate test database URL can be supplied
through `MANGROVISION_ANALYSIS_DELETE_TEST_DATABASE_URL`; otherwise the configured
migration connection is used solely for the isolated fixtures.

A database-enforced read-only audit found 12 protected and 23 eligible analyses
out of 35 retained records. History, detail and statistics flags agreed. The
database uses Read Committed isolation, and no fixture schemas remained after
the tests. Verification did not delete or update application records.
