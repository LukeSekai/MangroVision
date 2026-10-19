# Automatic mortality warning zones

A project site becomes an automatic warning zone when its recorded mortality
is **at least 30%**, across every recorded cause:

`recorded dead seedlings / originally planted seedlings * 100 >= 30`

The threshold uses integer counts before rounding. Cabatuan's saved example
has 103 deaths among 263 plantings: 39.16% mortality and 60.84% survival. Its
previous wave-only calculation used 79 wave deaths and incorrectly described
the whole site as prone to big waves. The system no longer infers that cause.

Counts come from the site's planting-event history, using the same original
outcomes as its survival summary. Planned and assigned points are excluded.
Reversed completions and future plantings are excluded. Deaths remain attributed
to their original planting event and site when locations are released or planted
by another organization. A replacement is a new planting event in its own site.
Unknown outcomes are reported separately and never assumed dead. Organization
death totals without identified planting events cannot be assigned to a site.

Project-site GeoJSON exposes `properties.mortality_warning`. The Warning Zones
response combines manual warnings with one automatic feature for each qualifying
site, using its current boundary. These features have `automatic: true`,
`warning_type: high_mortality`, `project_site_id`, and a stable negative ID.
They are calculated on read, so corrected monitoring records and changed site
boundaries update the warning without a duplicate polygon, migration, or job.
Warnings disappear below the threshold. Released land has no warning geometry;
the original site's historical planting outcomes remain intact.

Automatic warnings appear in the map's **Warning Zones** layer and Zone Editor
list. They provide caution context on planting points and in warning-exposure
reports. They do not block planting or assignment. Manual warnings remain
editable; automatic warnings cannot be manually edited or deleted. Review and
correct the underlying monitoring records to change their mortality rate.

Site cards, detail dialogs and previews use an amber warning with aligned icon
and text. Map site cards show only a short mortality percentage summary.
High-mortality sites use an amber dashed outline without a colored fill,
including when focused. Their automatic warning no longer adds orange rings
to individual points; planting-status colors and manual warning rings remain.
The map legend identifies the outline, and the site card shows a compact
warning badge with the mortality percentage. Site previews match the outline.
Automatic mortality zones open that same project-site card instead of a separate
warning popup. Point warnings show only the mortality percentage and a short
planting caution, without repeating the site name, counts, or threshold.
No permanent danger label covers the points. The Monitoring map lists the warning sites for
the selected organization. Saving a visit uses a single green busy button with
a spinner instead of a second pale loading box.

Regression checks cover exact 29%/30% thresholds, all causes, unchanged original
outcomes after release/replanting, corrections, automatic/manual zone behavior,
warning hover and preview updates, and the monitoring visit saving state.
The database check uses temporary PostgreSQL tables and rolls back its changes.
