# Automatic big-wave site warnings

A project site is flagged when **at least 30% of its distinct planted locations
have a recorded death whose cause category is `waves` (Big waves)**:

`distinct locations with big-wave deaths / distinct planted locations × 100 ≥ 30`

For 100 planted locations, 30 wave deaths trigger the warning; 29 do not. Ten
barnacle deaths, ten storm deaths and ten wave deaths give 10% wave deaths and
do not trigger it. Other causes and text mentioning waves under another category
do not count. Merely planned, assigned or skipped locations never dilute the
denominator unless they have genuine planting/death history.

The rule uses recorded history, including earlier planting cycles. Each location
counts once per site, even with repeated inspections, deaths or replantings.
Replanting or releasing an assignment does not erase the evidence. Correcting an
erroneous death or changing its cause recalculates the warning. If new distinct
planting locations bring the proportion below 30%, the site no longer qualifies.
The threshold uses integer counts before display rounding.

Planting-event site attribution is preserved when later image links or boundaries
change. Older records without an event use the same unambiguous spatial ownership
as the map. Unlocated organization-wide death totals cannot be attributed to a
particular site and contribute only after their locations are identified.

The server adds `properties.big_wave_risk` to project-site GeoJSON. It calculates
all sites in one grouped query; no warning polygons are inserted, and no schema
migration or background worker is required. Existing death categories keep their
stored keys; `waves` is labelled **Big waves** in death-entry choices.

Qualifying sites get a red dashed boundary, a permanent **Danger: Big waves**
label and a popup showing the evidence. Monitoring site cards, details and previews
also display the warning. The Monitoring map lists flagged sites for the selected
organization. Field maps receive the same site warning, even when their point
counts cover only the current participant. Existing data-change refreshes update
the warning after monitoring saves and corrections. This is a site warning and
does not change planting-point statuses or add a forbidden planting area.

Checks include the two panel examples, 29%/30% boundaries, distinct-count
deduplication, legacy records, replanting, corrections, site attribution, UI refresh,
and the actual query against PostgreSQL temporary tables with rollback.
