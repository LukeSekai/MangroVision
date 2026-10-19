# Activity protection and booking updates

## Changes prepared on October 9, 2026

### Activities with recorded planting

- Each assignment batch can reference its planting activity. Planters shows a **Planting activity** selector for confirmed or ongoing tree-planting activities owned by the chosen organization and site.
- The API validates that selection. It can infer a unique activity on the assignment date, but asks for a selection when the match is ambiguous or a site's eligible activity uses another date.
- Once the linked batch has any planting-event history, the activity's organization, area, title, dates, expected counts, and inspection interval cannot change. Cancellation and deletion are rejected.
- Contact and notes remain editable and are audited until completion. Progress can advance to in progress or completed. Every completed activity is read-only, including activities without planting records: dates, counts, contacts, notes, status changes, reopening and deletion are blocked by the API.
- Scheduling displays three progress choices: Completed, Pending and Cancelled.
  Approving a pending schedule approves its agreed time and makes it eligible
  for the calendar and planting assignments; it remains Pending until completed.
  Saving Pending preserves an existing confirmation or ongoing planting state.
  Recorded planting disables cancellation, and every completed activity
  cannot return to Pending. Missed activities retain a follow-up notice within Pending.
- A separate future activity at the same site remains editable. Unscheduled planting remains available where there is no eligible activity.
- An activity with assigned but unplanted points cannot change organization or area, or be deleted, until those assignments are removed.
- An activity remains reschedulable while its assignment has no recorded
  planting. Selecting an activity without an area when assigning points links
  it to the selected organization-owned project site in the same transaction.
  Staff do not have to assign its area separately in Scheduling first.
- Planting and schedule changes lock the activity row. A database trigger also prevents planting into an activity that has been made ineligible.

### Website bookings

- Changing the agreed date, time, title, or expected participants refreshes an unsent confirmation or queues an update after earlier delivery.
- An eligible cancellation queues a cancellation notice. Confirmed website requests retain their linked schedule instead of allowing deletion.
- Updating dates also updates the linked, unplanted assignment date. Past times and bookings spanning two Philippine dates are rejected atomically.
- Notes and contact corrections do not queue another booking email.
- Existing usernames, password hashes, and device registrations remain unchanged. A queued initial password is retained only inside the encrypted message; later messages refer to the existing password.
- Saving changes during an active delivery returns a conflict so staff can retry after that delivery finishes.
- Scheduling explains the queued message, and Website requests distinguishes confirmation, update, and cancellation delivery status.

## Historical records

Migration `20261009_0015` links only one-to-one historical matches by organization, site, and Philippine assignment date. It does not guess between activities.

The read-only assessment on October 9 found **14 assignment batches, no unique matches, and 13 unmatched batches with planting history**. These records remain intact but are not covered by a specific activity lock until their activity association is established. New assignments use the explicit association described above.

## October 14: missing activity-area association

San Isidro's confirmed website activity had no planting area, so its point
assignment was treated as unscheduled and its eight planting events did not
activate the schedule lock. The user-authorized repair restored the audited
October 10, 05:30–06:30 Philippine schedule and linked its existing assignment
and area. Planting dates, points, accounts, codes and email history were retained.
The repair was audited after verifying a fresh encrypted database/storage backup;
it sent no email. Other historical associations were not inferred or changed.

The assignment form includes eligible activities that have no area yet. The API
links the chosen activity to the selected site atomically with point assignment.
Zero recorded planting allows rescheduling before completion; the first planted
point locks planning details, even when the assignment is unfinished. Notes,
contact corrections and allowed progress updates remain available until the
activity is completed. A separate future activity keeps its own lock scope.

## October 15: completed activities are read-only

Completed activities expose only View in Scheduling's List and Calendar.
The API returns a conflict for every edit, reopening, cancellation or deletion,
including notes/contact changes and activities with no recorded planting.
  Completion remains separate from request approval.

At the user's explicit request, all 21 existing activities displayed as Pending
were marked Completed in one audited transaction. This includes three legacy
`scheduled` records; all inspected activities had already started. There are
24 completed schedules and zero pending schedules after this correction.
Only status and update timestamps changed. Dates, planting events, points,
assignments, accounts, individual codes, organizations and email history were
verified unchanged. No email was queued or sent by the correction.

A fresh encrypted database/private-storage backup was made first:
`backups/mangrovision-20261014T165139Z.tar.enc` (the filename uses UTC).
The dump checksum, all 136 object checksums and dump readability were verified;
this was not a full restore rehearsal. The ignored operation report and backup
verification are retained under `backups/pending-activity-completion-*`.

## October 20: automatic planting completion

Confirmed and ongoing tree-planting activities now complete automatically when
every point assigned to that activity is recorded as planted. All assignment
batches are checked together under the activity's row lock, in the same
transaction as the last planting update. Empty activities and activities with
pending or skipped points stay open. Other activities at the same organization
or site are independent. Deaths after planting do not reopen an activity.

The change writes one system activity entry and does not queue a booking email.
Completed planting records cannot be reverted to pending or skipped through
the point-status action. Scheduling shows Completed, exposes only View, and
uses Planting completed instead of current tide advice in its list and details.
The planter recording the final point sees a success acknowledgement before
returning to sign-in, rather than reloading with the now-completed activity code.

For older fully planted activities, run
`python scripts/complete_planted_activity.py --schedule-id ID` to preview one
activity without writes. `--apply` uses the same locked completion rule,
verifies that assignments, planting events and queued email records remain
unchanged, and saves a small report under `run_logs/`.
GET requests never change completion status.

## Verification

Development checks used mocked email delivery and PostgreSQL session-local temporary tables, rolled back afterward:

- 52 API and email-transport checks passed.
- 44 booking/activity database checks passed across the regression run and isolated migration-backfill check.
- 15 assignment checks passed, including completion through a clone of the production planting-event trigger.
- 7 targeted UI checks passed for assignment selection, protected activity fields, and image-result map navigation. Two map checks initially used an outdated `/` test route; their fixture now uses the application's `/map` route.
- Frontend production build and lint of the changed UI files passed. The build still reports the existing large-bundle warning.
- Python source compilation, `git diff --check`, and offline Alembic SQL generation passed.

No permanent database changes, demonstration planting records, or real test emails were made by these checks.

## Activation

The updated source depends on migration `20261009_0015`. The user approved the encrypted backup, shared database migration, and testing activation on October 9.

The fresh backup is `backups/mangrovision-20261009T014248Z.tar.enc`. Its database checksum, all 132 private object checksums, and readable database dump were verified; a full restore was not performed. Migration `20261009_0015` was applied successfully. Counts and fingerprints across all 16 preserved record tables are unchanged, excluding the newly added metadata columns. The activity guard trigger is enabled.

The testing API and tunnel were restarted and the Vercel deployment completed. Read-only checks returned HTTP 200 for `/`, `/field`, `/landing-page`, `/like.html`, and `/api/health/ready`. The root and field routes serve the MangroVision entry; the LIKE routes serve the separate LIKE entry. The public staff bundle contains the new activity lock and assignment activity selector.

Activation procedure:

1. Create a fresh encrypted database/private-storage backup and verify the archive and database dump. Use the existing private backup configuration; do not expose credentials.
2. Record pre-migration counts and fingerprints for planting records, assignments, accounts, device registrations, schedules, and email rows. Avoid running this during a field recording session.
3. Apply **only** `20261009_0015` through Alembic with the configured owner migration connection. It adds the assignment activity link/index, email kind, and planting guard trigger, and performs the conservative backfill in a transaction. It deletes no planting records.
4. Verify the revision, new fields, trigger, and unchanged underlying records. Compare activity-link metadata separately from the preserved assignment fields.
5. Restart the running **testing** launcher and update Vercel to the new tunnel using `MangroVision_New/start_testing.py --deploy`. Starting local development would replace the current testing session.
6. Verify API readiness and the staff, planter, and LIKE routes. Do not create real bookings or send test emails without an approved demonstration recipient.

The [evaluator recording script](evaluator-recording-2026-10-09.md) includes preparation, scene timings, narration, and short technical answers.
