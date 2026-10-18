# Project site information on maps

Hover inside a visible project-site boundary to see its information. Click or
tap the boundary to keep the information open; close it with the popup's ×.
On staff maps, enable **Project Sites** in Layers if its boundaries are hidden.
Project-site cards take priority over overlapping assignment-zone summaries.
Assignment-zone hover titles show the organization name, without repeating the
assignment title. Names keep their original capitalization and wrap inside the card.
Hovering a planting point still shows that point's individual details; hazard
boundaries retain their warning information.
Sites flagged for big waves keep their permanent danger label while still
opening the full analytics card on hover. Hiding a site's boundary hides its
label too.

The same card appears in Planting Map, Zone Editor, Analyze Image, Planting
Assignments, Monitoring Map, the field workspace, and the independent project
site reference map. It includes the site name, organization, notes, approximate
boundary area, and the saved analysis, assignment, and schedule counts supplied
by the existing site data.

Staff map cards also show current mapped locations by status: Available,
Awaiting planting, Planted, Recorded dead, Skipped, and Unavailable. A donut chart shows
their proportions, with the total at its centre and exact counts in the legend.
Chart colours match the planting point colours on the map. Each location counts
once. These counts cover all planting dates and remain independent of map
filters, including the monitoring map's dead-only filter. They describe current
locations rather than historical planting cohorts. **Assigned points** is a
separate total of locations that still have an assignment, including completed,
dead, skipped and unavailable assigned locations. **Awaiting planting** counts
only locations whose current map status is Assigned. The assignment total is
not added as another donut segment, so each mapped location still counts once.

Survival rate appears above the chart. It uses current
Planted divided by Planted plus Recorded dead, matching Dashboard Seedling Health.
Available, Awaiting planting, Skipped and Unavailable locations do not enter that rate.
Before any seedlings have been planted, it shows **Not available yet**.
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
