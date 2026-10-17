# Project-site area actions

The Zone Editor's three-dot menu opens **Release area** and **Delete site**.
Both actions require a server-generated preview version and are rechecked in
one transaction when the LGU confirms.

- Recording a death keeps the point's site membership and the current boundary.
  All planting-map tools show these locations as dark red dead points. Release
  and approved replanting are separate actions; a death report cannot perform
  either automatically.
- Release returns unplanted and confirmed-dead locations to planned, releases
  their assignment membership, and retains the boundary around living planting
  cells. Released locations between living seedlings are excluded from that
  boundary. Erosion restrictions still apply to these locations.
- Delete is available only when the site has dead seedlings, no living or
  unplanted locations remain, and its recorded planting events are all dead.
  This sets `project_sites.deleted_at`; it never deletes the site's history row.
- Events, death records, observations and their original site attribution remain
  intact. Resetting a physical location cannot increase the original cohort's
  survival rate. Confirmed deaths enter the existing approved replacement
  workflow, so later assignment can link the next planting cycle correctly.
- If the entire area is released, `area_released_at` removes it from spatial
  ownership. The site stays in the list until it qualifies for Delete.
- Released land is available to other registered organizations after excluding
  boundaries that have already claimed it. Existing project sites take priority
  over assignment hulls when checking organization areas.
- An organization without an active project-site boundary can draw its first
  site on unclaimed land inside the drone map. It does not need an assignment
  outline or a prior land release. Other organizations' project sites and
  legacy assignment areas still block overlaps. Once saved, its boundary and
  available released land constrain subsequent sites. The drawing count includes
  unclaimed points for a first site.

Migration `20261014_0018` adds the site lifecycle columns and the private
`project_site_area_releases` audit table. The latter records before/after
boundaries and released point snapshots, with backend-only access and RLS.
Historical reports retain the original site IDs and names.

The API provides `GET /api/project-sites/{id}/actions`,
`POST /api/project-sites/{id}/release` with a preview `version`, and
`DELETE /api/project-sites/{id}?version=...`.
Missing or changed previews cannot cause a mutation.

Database regression tests are opt-in and use temporary tables with an outer
transaction rollback. They assert every mutation target resolves to `pg_temp`.
Run from the repository root in PowerShell:

```powershell
$env:MANGROVISION_RUN_TEMP_ACCOUNT_TESTS='1'
.\venv\Scripts\python.exe -m pytest tests/test_site_actions_postgres.py -q --tb=short -p no:cacheprovider
```

Use short tracebacks for database checks. Never print connection parameters or
private environment values.
