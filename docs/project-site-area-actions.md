# Project-site area actions

## Assign points to form the site

Approve the organization's tree-planting activity in Scheduling, then open
**Assign Points**. Choose the organization and confirmed activity, click or
brush available points on the map, and choose **Assign points**. The server
creates the owned project site around exactly those selected locations, links
the activity, and allocates planting work in one transaction. A failed request
leaves no empty site, partial species allocation, or activity-area link.

There is no separate project-site creation form, site-name input, or choice
between a new and existing site. Zone Editor retains the project-site list,
release/delete actions, and eroded/warning zone tools. The old manual creation
API responds with 410 and directs staff to Assign Points.

Only unclaimed, unassigned planned locations inside the drone map are available
for a new activity. The boundary excludes unselected points and protects other
sites, erosion and forbidden areas, including released holes between living
seedlings. Multiple recorded species receive separate work batches under one
site. Points stay awaiting planting; assignment creates no planting events and
does not start survival or monitoring calculations.

Organizations must have their own confirmed or ongoing tree-planting activity
before available points can be assigned. LGU uses its permanent account and is
exempt: it can assign available points without scheduling an activity.
The assignment form and server both enforce this distinction.
Each allocation creates its site automatically. Existing
activities that already have a linked site retain that area and historical
identity; their remaining available points can be allocated without a site
chooser. No previously saved sites or planting records are converted or moved.

The server rechecks availability under transaction locks. Original planting,
death and monitoring history stays with the original site after locations are
released and later assigned to another organization. Selection boundaries are
clipped to map coverage and excluded areas, with numerical clearance at shared
edges to prevent projection slivers from claiming another owner's land.

Image-processing locks apply while a real job is running. Page-load recovery
checks for old or completed jobs do not lock editing or assignment.

## Release and delete

The Zone Editor's three-dot menu opens **Release area** and **Delete site**.
Both actions require a server-generated preview version and are rechecked in
one transaction when the LGU confirms.

- Recording a death keeps the point's site membership and the current boundary.
  All planting-map tools show these locations as dark red dead points. Release
  and selected-location release are separate actions; a death report cannot perform
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
  survival rate. Confirmed deaths retain release metadata, so a normal later assignment can
  link the next planting cycle correctly. There is no LGU reservation.
- If the entire area is released, `area_released_at` removes it from spatial
  ownership. The site stays in the list until it qualifies for Delete.
- Released land is available to other registered organizations after excluding
  boundaries that have already claimed it. Existing project sites take priority
  over assignment hulls when checking organization areas.

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
