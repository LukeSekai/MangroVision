# Dashboard project-site comparison

Dashboard → Project Sites compares current site ownership, rather than whole
image analyses. It shows the responsible organization, mapped-location total,
the six current map statuses, locations in eroded zones, and warning categories.
The summary counts sites, their currently owned locations, and sites with mapped
warnings or eroded locations. Unclaimed locations are excluded from site totals.

This tab is labelled **Current snapshot**. It has a project-site filter without
date fields or a report download that could be mistaken for a current-site
export. Other Dashboard tabs retain their dates and report controls. **View
planting history** opens Seedling Health without changing the selected site or
the saved reporting dates. Historical planting outcomes and survival calculations
are unchanged. Image measurements remain accessible through Analyze Image and
its saved analysis review; no analysis links are reassigned.

Current point counts use `_load_current_dashboard_points` and
`current_point_status`, matching the Dashboard/map classification. Removed points
are excluded and each physical point counts once. Sites created from selected
points work without a linked image analysis. Release/reassignment removes a
location from the former site's current ownership; historical attribution stays
with the original planting. Fully released site records retain Monitoring access
but have no map link when they have no current boundary.

Warnings use the existing manual and automatic warning polygons. Overlapping
polygons of the same type and severity count a location once in that category.
A location may appear in different categories; unique exposure totals never add
those category counts together. Warning status breakdowns use all six map
statuses, including dead, assigned, skipped and unavailable. Eroded-zone exposure
is reported separately and can include planted/dead locations whose map status
takes precedence over erosion. Automatic mortality warnings retain the existing
historical threshold and do not imply a cause of death.

`GET /api/dashboard/sites` retains its existing analysis/area response fields for
compatibility; the comparison no longer displays them. Its project-site rows
add organization, current `point_status_counts`, `eroded_point_count`, warnings,
and release state. Site `point_count` now uses the same current point selection
as its status breakdown, rather than the all-records linked-point count.

Regression coverage in `tests/test_project_site_counts.py` uses only synthetic
connections and in-memory SQLite. Dashboard component tests cover the comparison,
empty/released sites, navigation, and snapshot/history filter behavior. No shared
database, private imagery, credentials, or migrations are needed for these checks.
