# Current eroded-zone rules

Implemented on October 22, 2026. No migration or new seedling status.

Erosion is a restriction derived from active zone geometry. It does not write
seedling condition, death cause, planting history or recorded survival.
Planned locations stay mapped. Existing planted/dead seedlings keep their
recorded state, and monitoring/condition updates remain available.

Zone Editor reuses the drawing form and shared modal. **Edit** creates a draft
with draggable, keyboard-accessible vertices for all polygon parts and holes.
The saved boundary remains active until Save succeeds. Overlap with actual
planting/seedling records requires **Save eroded zone?**; cancelling makes no
saved change. The server checks overlap within the zone-save transaction.
An internal confirmation header acknowledges that dialog; it is not a new
planning setting or database field. Zone updates retain their stable ID and
reject a supplied stale `updated_at` value.

Planting, assignment and selected-dead-location release read current erosion
using their submitting database connection. Shared transaction advisory locks
coordinate these actions with exclusive eroded-zone writes. The spatial check
uses every active eroded zone and includes boundary points. Another covering
zone continues blocking a location after one zone is edited/deleted.

Individual planting and staff-entered planting reject erosion. A selected
planter batch containing an eroded point rejects before any of that batch's
new plantings are written. Earlier saved plantings remain unchanged. Existing
deleted-point handling is preserved. Replanting keeps the current release →
Assign Points → actual planting process; no retired replacement queue returns.

Available planting capacity excludes current erosion, completed planting and
dead/removed locations. Planting Map names the stored area measurement
**Analyzed plantable area** and shows **Available planting locations** separately.
No historical image measurement or survival calculation is rewritten.
Warning-zone rules remain unchanged.

## Regression coverage

Backend checks use synthetic imagery, in-memory spatial fixtures or mocked
connections; shared database writes are not part of this verification.
They cover confirmation/cancellation, record preservation, current boundaries,
overlapping zones, atomic collection replacement, mixed planting batches,
condition updates and LGU monitoring with/without new deaths under erosion.
Existing release, site selection, field routing, historical counting, health,
planting performance and mortality warning checks are included.

Frontend checks exercise real React/Leaflet components with isolated data:
overlap confirmation, failure recovery, draft cancellation, multi-part/hole
preservation, keyboard editing, individual/bulk planting rejection, and the
short replanting modal. Existing exports, guides and LGU Sites are included.
Synthetic desktop/narrow-layout screenshots check the Edit row and shared
confirmation modal. These do not certify physical field accuracy or live
PostgreSQL concurrency under deployment load.

Run from the repository root:

```powershell
.\venv\Scripts\python.exe -m pytest MangroVision_New/api/tests/test_eroded_zone_rules.py MangroVision_New/api/tests/test_zone_boundaries.py MangroVision_New/api/tests/test_eroded_analysis.py tests/test_field_routing.py tests/test_site_release_api.py tests/test_project_site_selection_api.py tests/test_lgu_sites_api.py tests/test_project_site_counts.py tests/test_current_health_summary.py tests/test_planting_performance.py tests/test_site_mortality_warning.py -q --tb=short
```

From `MangroVision_New/client`:

```powershell
node --test src/components/Panel.test.js src/pages/FieldApp.navigation.test.js src/pages/SiteRelease.test.js src/utils/zoneGeometry.test.js src/guides/tours.test.js src/guides/tourRuntime.test.js src/components/SavedPointExports.test.js src/utils/pointExports.test.js src/pages/LGUSites.test.js src/utils/mapPointStats.test.js
npm run build
```

See [system guides](system-user-guides.md) for user instructions and the
[thesis defense/demo workflow](thesis-defense-demo-workflow.md) for rehearsal.
