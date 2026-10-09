# Project site information on maps

Hover inside a visible project-site boundary to see its information. Click or
tap the boundary to keep the information open; close it with the popup's ×.
On staff maps, enable **Project Sites** in Layers if its boundaries are hidden.

The same card appears in Planting Map, Zone Editor, Analyze Image, Planting
Assignments, Monitoring Map, the field workspace, and the independent project
site reference map. It includes the site name, organization, notes, approximate
boundary area, and the saved analysis, assignment, and schedule counts supplied
by the existing site data.

Staff map cards also show current mapped locations by status: Available,
Assigned, Planted, Recorded dead, Skipped, and Unavailable. A donut chart shows
their proportions, with the total at its centre and exact counts in the legend.
Chart colours match the planting point colours on the map. Each location counts
once. These counts cover all planting dates and remain independent of map
filters, including the monitoring map's dead-only filter. They describe current
locations rather than historical planting cohorts or verified survival rates.
Until planting-point data has loaded successfully, the card shows site details
and the supplied linked-point count without inventing zero status counts.

Field cards label their counts **Your assigned points** and cover only the
signed-in participant's work in that site. Reference maps show site information
and supplied counts without requesting the staff planting-point dataset.

Boundary area is estimated locally from the site's Polygon or MultiPolygon,
subtracting interior holes. It describes the mapped boundary, not plantable
area or planting capacity.

Implementation is shared through `client/src/utils/projectSiteInfo.js` and
`client/src/components/ProjectSiteInfo.css` under `MangroVision_New`. Counts use
the existing map status classification. Hover and tap do not fetch additional
data or save any records. Planting-point refreshes update the site cards.
