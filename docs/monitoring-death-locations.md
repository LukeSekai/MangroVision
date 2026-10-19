# Monitoring death locations and replacement planting

LGU staff enter the total newly dead seedlings, then identify up to that many
planting events on the map. Selections identify deaths already in
the total; they never add deaths. For example, entering 10 and selecting 10
reports 10 deaths, all located. Selecting 6 leaves 4 unlocated. Count-only
visits remain supported, with all reported deaths initially unlocated.

## Field workflow

1. In **Monitoring → Prepare field sheet**, choose an organization, project site,
   and assignment. Zoom to the section being inspected and select **Print visible
   area**. The numbered map and checklist contain the same locations, with
   analysis/point references, coordinates, planting dates, and observation space.
2. After the visit, enter **Newly dead seedlings** and **Cause of death**, then
   identify their locations on the visit map. For many deaths, choose **Brush select**
   and hover across the points inside the circular brush (drag on touch screens).
   Fast sweeps include points along the full path. Repeated passes keep points
   selected; grey, unavailable locations are skipped. Selection stops at the
   entered death total. Use **Brush erase** to remove several selections, or
   **Move / click** for individual corrections and map navigation. **Esc** returns
   to Move / click without closing the visit dialog. Deselect locations before
   reducing the total below the selected count. Selections are saved with the
   visit; moving the brush does not save a death immediately. These controls also
   work when identifying remaining locations in visit history.
   The selected cause applies to the new deaths in this visit, including deaths
   whose locations are unknown. Optional notes describe the observed evidence.
   The visit map has no site/assignment filters, search, checklist or print toolbar;
   printing remains in the separate **Prepare field sheet** workflow.
3. In **Monitoring History → Identify remaining locations**, add or correct
   locations whenever field information becomes available. This does not change
   the visit's reported count, survival calculation, or two-week schedule.
   Removing a selection requires a correction note. Releasing a location
   protects the linked death from location changes.
4. In **Monitoring ? Show map**, optionally enable **Show dead plants only**.
   Select dead locations using Move / click, Brush select, or Brush erase, then
   choose **Release from project site**. Selected planting cells leave the old
   boundary and the locations become green Planned. The selection is atomic:
   stale, living, removed, eroded, or subsequently replanted locations reject
   the entire batch. Original planting counts, deaths, visits, and survival stay
   with the old site. Escape returns to Move / click.
5. Another organization can claim the released land as a new project site,
   assign its points, and record planting through the normal participant flow.
   Each new planting has a new event and independent site reporting. See
   [Release dead locations](project-site-location-release.md).

## Data and API

Alembic revision `20260916_0008` adds visit/death links and their correction audit,
visit versions and location budgets, assignment release metadata, review records,
and replacement event links. A unique index prevents two death records for the
same planting event. Events, rather than reusable map points, identify seedlings.

Existing visit counts and coordinates are preserved. Explicit newly-dead counts
are used when present; older reports use only an unambiguous cumulative increase.
Ambiguous reports are flagged for review and cannot accept invented locations.

Authenticated LGU endpoints:

- `GET /api/monitoring/organizations/{id}/planting-locations?monitored_at=...&record_id=...`
- `POST /api/monitoring/organization-records`, now also accepting
  `dead_planting_event_ids`, `unlocated_dead_count`, `death_reason_category`, and
  `death_reason_notes`. A valid cause is required when new deaths exceed zero.
- `PUT /api/monitoring/organization-records/{id}/death-locations`, accepting IDs,
  `expected_version`, and optional `correction_note`.
- `GET /api/monitoring/release-candidates`
- `POST /api/monitoring/release-locations` with `planting_event_ids` and their `versions`.

Visit history returns `reported_dead_count`, `located_dead_count`,
`unlocated_dead_count`, `dead_planting_event_ids`, `location_version`, and
`location_review_required`. All linked count/location changes are transactional;
stale versions or count baselines are rejected.

Organization summaries expose historical `total_planted` separately from
`current_planted_points` (physical locations still held by the organization).
Monitoring statistics use the historical total so releasing a dead location
does not shrink the original cohort or change its survival rate.

Death causes are stored in the existing visit JSON snapshot and returned in visit
history. Identifying locations later copies that saved cause into new death
records. The dashboard cause chart combines identified deaths and the remaining
unlocated visit counts, without counting a linked death twice. Its date range
uses the visit/report date; location, assignment, species or planter filters
exclude unlocated deaths because their finer attribution is unknown. Replanting
approval preserves the death records and their cause statistics.

## Manual verification

Check map/checklist synchronization after filtering, printing
two different map extents, reconciling a partial historical visit, rejecting stale
edits, and reviewing/assigning/completing a replacement with a blank organization
selector at the start of each assignment.
