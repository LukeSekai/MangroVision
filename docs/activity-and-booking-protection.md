# Activity protection and booking updates

## Changes prepared on October 9, 2026

### Activities with recorded planting

- Each assignment batch can reference its planting activity. Planters shows a **Planting activity** selector for confirmed or ongoing tree-planting activities owned by the chosen organization and site.
- The API validates that selection. It can infer a unique activity on the assignment date, but asks for a selection when the match is ambiguous or a site's eligible activity uses another date.
- Once the linked batch has any planting-event history, the activity's organization, area, title, dates, expected counts, and inspection interval cannot change. Cancellation and deletion are rejected.
- Contact and notes remain editable and are audited. Progress can advance to in progress or completed; a completed activity with planting history cannot be reopened.
- A separate future activity at the same site remains editable. Unscheduled planting remains available where there is no eligible activity.
- An activity with assigned but unplanted points cannot change organization or area, or be deleted, until those assignments are removed.
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
