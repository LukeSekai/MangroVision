# Planting-point exports

Open **Planting Map → Export Saved Points** and choose **CSV**, **GPX**, **KMZ**
or **GeoJSON**. KMZ packages the KML and its marker images in one file for Google
Earth. GeoJSON provides GIS data with color fields whose display depends on
the receiving application. The export
dialog opens with **All points** selected. Choose
**Project site**, **Species** or **Saved analysis** to reveal the corresponding
dropdown. Choose its value, review the selected points on the map and their
count/filename, then use **Download [format]**. Switching scope clears the
previous choice. Returning to **All points** includes every saved location.

The map preview shows every selected point over the existing satellite and
orthophoto layers. Each point is a single solid dot, using this color priority:

1. **Red:** dead, including dead points inside erosion.
2. **Orange:** inside an eroded zone, including planted points there.
3. **Yellow:** planted, for either species.
4. **Green:** Bungalon; **magenta:** Rhizophora, for remaining points.
5. **Gray:** other/unrecorded species, when no status color applies.

The preview and files use the same priority. Color changes presentation only;
species, status and erosion membership remain in the downloaded fields, and
hover details retain species, current status and erosion advisories. The legend
shows the colors present in the selection. It fits the selection automatically;
staff can pan, zoom or use **Fit selected points**. The preview has its own
mouse-wheel/trackpad zoom handling: scrolling or a pinch-style Ctrl+wheel gesture
over the map zooms that map and prevents the browser's page-zoom action there.
Outside the map, controls retain normal browser scrolling/zoom behavior. The
preview's viewport does not change the workspace map or its selection. Pan/zoom and
main-map layer switches never change export membership. Selecting a scope
without choosing a value clears the preview and disables downloading. Download
is enabled after the selected points have been plotted. Imagery failures show
a message while preserving the point locations and download option.

Failed initial loading provides **Retry loading points**. A failed download
keeps the scope and map available for retry. The dialog holds a snapshot of
saved points from opening time, so a background refresh cannot silently replace
the reviewed locations. The map draws from the same waypoint array sent to the
download endpoint. Scope controls lock during download. Closing or hiding the
page cancels an in-flight request and removes the preview's map/listeners.
The dialog supports Escape, traps keyboard focus, restores the opening button's
focus, and keeps its download/cancel actions visible while its body scrolls.

## Scope and attribution

- **Project site** uses the map API's resolved current boundary ownership, not
  an analysis-wide site link or an earlier assignment. One analysis can span
  several sites. Released locations leave the former site's export. **No
  current project site** selects locations with no resolved current owner.
- **Saved analysis** uses the stored analysis ID for filtering and the current
  display number/source image for the selector. Repeated image filenames remain
  separate selections.
- **Species** uses recorded species from the map API, preferring the current
  planting snapshot for planted/dead seedlings, including replacements, then
  the resolved map or saved analysis species. Named choices are **Bungalon** and
  **Rhizophora**; other values share **Other recorded species** without being
  reclassified or removed from exports. **Not recorded** selects legacy points
  without a recorded species; spacing is never used to guess it.
- These exports describe current physical locations. They do not calculate
  historical survival, planting totals or monitoring outcomes. Deleted points
  are excluded. Planned, assigned, planted, dead, skipped and unavailable
  locations remain eligible for export. Eroded locations retain their
  availability flag and reason.

All four download choices use the same selected waypoint payload. Existing
coordinate, point-number, buffer and area values are preserved. The preview and
exporters read the same palette in `client/src/config/pointExportStyles.json`.

- **KMZ:** Download KMZ includes `doc.kml` and generated PNG marker icons.
  Solid marker colors follow the same priority as the preview. Small icons and hidden labels keep
  dense areas readable. The archive has no remote icon references and contains
  no drone photos, orthophoto, credentials or private storage links. Each
  seedling remains one Point placemark; styling adds no location geometries.
  The raw `/api/export/kml` endpoint used by Analyze Image stays available with
  the same color priority, fields and descriptions. KMZ bundles the dot icons.
- **GPX:** Descriptions include species/status/erosion, and MangroVision
  extensions preserve marker color/category alongside attribution. Symbol
  names request red for dead, orange for erosion, yellow for planted, then a
  species-related flag. GPS software decides which symbols/colors it supports.
  Each waypoint has a short stable navigation ID, such as `MV-000101`; its full
  readable point name stays in the comment, with project-site details in the
  description. These are individual field-navigation waypoints, not an ordered
  route. Use KMZ for bundled marker styling in Google Earth.
- **GeoJSON:** Properties include `marker-color`, `marker-size`, `marker-symbol`
  and `mangrovision_style` with the resolved marker color and color category.
  Compatible apps can use these hints; GeoJSON itself does not mandate styling.
  Processing previews keep their existing GeoJSON properties unchanged.
- **CSV:** Columns include readable point name, navigation ID, point ID,
  analysis ID/name, project-site ID/name,
  species, erosion availability/reason, **Marker Color** and **Color Category**.
  CSV stores these as data rather than spreadsheet
  cell formatting. Formula-like text in CSV labels is neutralized.

Scope appears in the
preview dialog, filename, and GPX/KML/GeoJSON document metadata. CSV scope can be
identified from its filename and row attribution columns.

## Names and dates

File names use the selected scope's readable name, stable site/analysis ID and
the Philippine export date, for example
`MangroVision_north-site_site-4_planting-points_2026-10-22.kmz`.
The title, filename and date are frozen when opening the dialog; changing scope
updates its name using the same snapshot. The extension matches the downloaded
format, including `.kmz` for Google Earth. Filenames use a sanitized basename
without path separators or header control characters.

Point names use saved display numbers when available, for example
`Analysis 2 — Point 007 (ID 101)`. The stable point ID distinguishes repeated
point numbers across analyses. Missing legacy names receive readable fallbacks
without `undefined` or `null` labels. GPX, KML and GeoJSON document titles describe
the selected scope.

The existing `/api/export/{csv,gpx,kml,geojson}` serializers accept optional
attribution fields; `/api/export/kmz` packages the styled Google Earth file and
requires staff permission. Older Analyze Image export payloads remain supported. The
staff map uses the existing authenticated map-point read and shared secure
fetch/CSRF handling. This feature makes no database writes or schema changes.
Report PDF/CSV flows are unchanged.

## Validation

From `MangroVision_New/client`:

```powershell
node --test src/utils/pointExports.test.js src/components/SavedPointExports.test.js src/components/Panel.test.js src/guides/tours.test.js
npx eslint src/utils/pointExports.js src/utils/pointExports.test.js src/utils/pointMarkerStyle.js src/utils/pointCanvas.js src/components/SavedPointExports.jsx src/components/PointExportMap.jsx src/components/SavedPointExports.test.js src/components/PlantingPointSelector.jsx src/components/MapView.jsx src/components/Panel.test.js src/guides/tours.js
npm run build
```

From the repository root:

```powershell
.\venv\Scripts\python.exe -m pytest MangroVision_New/api/tests/test_point_exports.py MangroVision_New/api/tests/test_eroded_analysis.py -q --tb=short
```

Backend checks also inspect every KMZ icon reference, representative PNG pixels,
unmodified coordinates/counts, text escaping, permission checks, and color
priority while preserving species/status/erosion fields in all formats. Icons are generated using the existing
Pillow dependency; no dependency changes or database migrations are required.

These checks use synthetic data and isolated API fixtures. They do not certify
the live database, hosted application, or behavior in external GPS/GIS software.
Interaction checks exercise real Leaflet bounds and marker membership in an
isolated DOM, including matching each format's downloaded rows to the preview,
scope resets, panning, tile failure, retry, duplicate submission, focus return,
solid colors and their priority after zooming, and map/request
cleanup on hidden retained pages. Desktop/mobile visual review
still requires a connected browser.
The shared map/export canvas cancels superseded animation frames and ignores
removed canvases; map interaction tests fail on asynchronous teardown errors.

The broader activity audit catalog check currently reports 16 older unclassified
routes. The same gaps exist in the published `4d4fd48` catalog; the KMZ route is
classified and adds no new gap. These unrelated auth/report classifications
were left unchanged.
