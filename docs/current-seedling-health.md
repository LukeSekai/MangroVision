# Overall seedling totals

Dashboard Seedling Health, its comparison charts, and the monitoring report
use four overall cards across all planting dates:

- **Total seedlings:** current Planted plus current Dead locations.
- **Planted:** the same current planting status used by Planting Map.
- **Dead:** the same current death status used by Planting Map.
- **Survival rate:** current Planted divided by Total seedlings, multiplied by
  100. Planted includes uninspected locations. An empty total has no rate.

Each physical location counts once. Planned, assigned, skipped, unavailable,
and deleted locations are excluded. Species, site, and planter comparisons
use these same counts and survival calculation. Project site filters apply to overall totals;
date filters apply to the visit, death-cause, and measurement history.

Located deaths from organization visits already update the map status.
Organization-wide balances are kept in visit history and are not added again.
Locations released for replanting leave these totals until planting is recorded
again. Earlier planting cycles, deaths, and inspections remain in history.

The backend assembles these counts in `mangrovision_db/health_summary.py`
through `get_dashboard_ecology`. The overall survival rate uses current map
status across all planting dates. Inspection-age results and inspection
coverage are not part of the overall summary.
