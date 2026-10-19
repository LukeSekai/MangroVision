# Release dead locations from a project site

Staff record deaths during Monitoring visits. Recording a death does not
automatically release the location or change the site's boundary.

1. Open **Monitoring → Show map** and select confirmed dead locations using
   Move / click, Brush select, or Brush erase. Up to 500 locations can be selected.
   For either brush, hold the left mouse button and drag (press and drag on a
   touchscreen). Release to stop. Hovering alone never selects or erases points.
   The brush ignores hidden site-detail dialogs retained from earlier pages;
   it pauses only while a dialog is visible. Selected dead points turn purple.
2. Choose **Release from project site** and confirm. The complete selection is
   checked and released in one transaction. Stale, living, removed, eroded,
   or subsequently replanted locations cannot be released through this action.
3. The selected planting cells leave the old project's current boundary.
   Their active allocations are released and the mapped locations become green
   **Planned**. Unselected locations and the rest of the boundary stay in place.
4. LGU can assign the released points to another organization through
   **Assign Points**, which creates its project site. Assignment alone
   does not count as planting. Completing planting creates a new event attributed
   to the new organization and site, linked to the earlier planting cycle.

Historical planting events, deaths, causes, visits, observations, and original
site attribution stay intact. If Organization 1 planted 100 seedlings and 20
died, its original site retains 100 plantings, 20 deaths, and 80% recorded
survival after release. Organization 2's new site has its own planting counts
and survival results. Further deaths among Organization 1's remaining seedlings
can still change that original site's recorded survival.

Current map counts follow current ownership; historical performance follows
planting events. A released location has no current project site until claimed.
There is no staff replacement queue or staff replacement planting endpoint.
**Monitoring → LGU Sites** manages LGU's separate planting sites and monitoring.
Released locations can be assigned through the normal planting process, including
to LGU. See [LGU Sites](lgu-sites.md).
The old `/monitoring/replanting` page redirects there for existing bookmarks.

Staff endpoints:

- `GET /api/monitoring/release-candidates`
- `POST /api/monitoring/release-locations` with `planting_event_ids` and `versions`.

The existing private `replanting_requests` table retains release and next-cycle
links for the planting completion triggers. It does not reserve work for LGU.
The existing `project_site_area_releases` table audits boundary changes.
No schema migration is required.

For an installation with unfinished reservations from the former staff queue,
preview them with `python scripts/release_legacy_lgu_queue.py`. Run the same
command with `--apply` to release their land. The script excludes completed
plantings, current organization allocations, removed points, and newer cycles.
It is repeatable and leaves historical records intact.

Regression checks use isolated temporary PostgreSQL tables and roll back:

```powershell
$env:MANGROVISION_RUN_TEMP_ACCOUNT_TESTS='1'
.\venv\Scripts\python.exe -m pytest tests/test_site_release_postgres.py tests/test_site_release_api.py tests/test_site_actions_postgres.py -q --tb=short
```

Client interaction checks: from `MangroVision_New/client`, run
`node --test src/pages/SiteRelease.test.js src/components/MapNavigation.test.js`.
