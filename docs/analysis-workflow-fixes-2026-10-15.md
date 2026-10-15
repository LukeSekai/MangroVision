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
