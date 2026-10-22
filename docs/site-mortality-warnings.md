# Site mortality alerts and wave-damaged locations

A project site has a mortality alert when recorded deaths are **at least 30%**
of its original planted seedlings, across all recorded causes:

`recorded dead seedlings / originally planted seedlings * 100 >= 30`

Counts use planting-event history. Planned and assigned points, reversed
completions and future plantings are excluded. Deaths remain attributed to the
original site after release or planting by another organization. Unknown
outcomes remain separate. Correcting monitoring records updates these totals.

Cabatuan has 103 recorded deaths among 263 original plantings (39.16%), of
which 79 locations have wave-caused death records. The 30% site mortality alert
does **not** mark every plant or the whole site as dangerous. The Monitoring
map lists clickable site names; clicking one focuses its map location and
restricts release controls to that site. Detailed mortality counts remain in
the site card and monitoring details.

## Point warnings

Wave warnings use the exact planting-point IDs in dated `point_death_records`
whose cause is `waves`, joined to their planting events. They exclude future
deaths, reversed completions and removed points. They do not use the site's
whole boundary, a surrounding buffer, or a neighboring plant's coordinates.
Other causes such as barnacles, drying out, animal damage and storms do not
create a wave warning.

An affected location keeps its warning after release makes it Planned, and
after another organization assigns or replants that location. Changing the
recorded cause or correcting the death updates the warning from its source
records. Warning calculation never changes planting status or ownership.

The map keeps ordinary point fills: green Planned, yellow Planted, and the
existing colors for other statuses. An orange outline identifies only a
location with its own wave warning or another applicable, explicitly drawn
manual warning. Its tooltip describes that location's warning. Unaffected
planted points have their normal black outlines and no mortality/wave warning
in their individual tooltip.

Automatic mortality features retain stable negative IDs for site-summary
navigation and compatibility. They carry `point_advisory: false` and
`render_mode: site_summary`: they are not drawn as orange polygons and are
excluded from individual point advisories. Focusing one opens the actual
project site and its summary. Ordinary project-site boundaries and previews
retain their normal styling, including when focused. No orange mortality
overlay or released-cell boxes are drawn.

The former area-wide Cabatuan wave-exposure zone is retired; it must not be
recreated over the original boundary. Its replacement is the dynamic set of
recorded wave-death locations. Other manually drawn geographic Warning Zones
remain independently editable in Zone Editor and keep their established rules.

Regression checks cover exact 29%/30% site thresholds, point-specific warning
scope, normal living-plant tooltips, orange outlines on affected Planned
points, release/reassignment persistence and corrected causes. Database checks
use isolated temporary PostgreSQL tables and roll back.
