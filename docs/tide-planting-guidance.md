# Tide-based planting guidance

## What changed

The existing WorldTides / Open-Meteo integration is retained. `/api/tides/forecast`
now also returns `heights` (timestamp in UTC epoch seconds, `height_m` in metres),
actual sample coverage, datum/reference, freshness, fallback status and attribution.
High/low `extremes` remain compatible with existing consumers. No forecasts are
fabricated beyond the returned horizon. The Open-Meteo request remains capped at
the existing eight-day API limit; Scheduling requests seven days.

Scheduling plots the sampled series, not a smooth curve invented between sparse
high/low events. Null samples break the curve. Calendar and form advice uses the
same coastal series and estimated limit as the graph over the full activity.
Fractional-hour boundaries are interpolated linearly. This is still a
forecast approximation, not a guarantee about unobserved between-sample peaks.

Workflow, confirmation and later organization-owned site assignment are unchanged.
No crews, attendance, ecological scoring, override workflow or weather limits were added.

## Scheduling display

The graph uses the default coastal forecast without a project-site dropdown.
Its green/red labels say **Safe to plant (estimated)** and **Not safe to plant
(estimated)**. The explanation states that ground height is estimated and asks
users to check their actual planting area. Labels use everyday terms such as
**time underwater**, **average sea level**, and **days between plant checks**.

Calendar entries keep their compact sizing and green/red backgrounds. Clicking
an activity or tide time opens its details; the compact buttons no longer use
check/cross symbols.

Choosing a date or time that crosses the graph's planting limit immediately opens
a **High tide caution** dialog. A single selected start or end time is checked
against the interpolated graph height; once both are set, the full activity is
checked. The check also runs when a delayed forecast arrives. **Change time**
and **Keep this time** return to the unchanged form without submitting it.
Dismissed warnings do not repeat for an unchanged selection. Saving proceeds
normally after form validation, without a second tide confirmation. Unavailable
data is not called high tide. A synchronous in-flight guard prevents duplicate
submissions, and server errors leave entered values intact. Only the form or
caution dialog is open at one time.

Calendar, list, and form advice now use `assessGraphWindow`, so a planting-area
assignment or saved ground measurement is not required. Green means every part
of the activity is at/below the graph's limit; any time above it makes the activity
red. High/low tide labels also compare their height against that limit, rather
than automatically assuming low means safe. Activity assignment and stored
measurements remain unchanged. The separate surveyed-site assessment helper and
calibration API remain available; their form is no longer part of Scheduling.

The schedule legend only shows **Safe to plant** and **Not safe to plant**.
When the graph cannot cover an activity, plain text explains why: for example,
**Tide forecast not available for this date**, **Set the activity start and end
times**, or **Refresh tides to check this activity**. Missing data never becomes
a green or red prediction. Earlier times still within the returned graph can
be estimated; these are model predictions, not observed historical conditions.

In the separate surveyed-site assessment, green windows require ground at/above MSL, estimated inundation of at
most 30% over the returned forecast, and water at/below the ground. Red means
water above ground or a site that fails the baseline. More than 30% through 50%
is marginal / needs assessment; above 50% is unsuitable. 100% means submerged
throughout this forecast, not proof of permanent submergence outside it.
This is NOT a personnel-safety or long-term ecological-suitability clearance.
No sample value is saved as a default survey elevation. In that assessment, missing calibration,
stale forecasts (six hours), gaps, past windows, incompatible references, and
out-of-coverage activities are Unknown. A provider/model reference change requires
re-verification. Geometry edits clear calibration; re-enter it after reviewing
the revised site. Other site edits preserve calibration.

Without calibration, the graph shows a labelled **reference preview**: the
minimum hypothetical elevation exceeded for at most 30% of forecast time,
constrained to at/above MSL. Green/red compares water to this reference; it does
not measure an actual site. Calendar activities use this same estimated reference.
The percentage integrates duration above ground using linear interpolation of
sample crossings; gaps and invalid samples prevent assessment. Chart colors
meet exactly at interpolated crossings. The chart replaces the upper metadata,
advisory block, and next-high/low cards with a compact legend; attribution and
the scientific reference remain below it.

Calibration is shared server-side in nullable `project_sites.tide_calibration`
JSONB; existing sites remain uncalibrated. Its cohesive fields contain the
measurement, reference, survey note, coordinates and timestamp, not foreign keys.
Only the existing LGU/admin/planner authorization boundary can write it through
`PUT /api/project-sites/{id}/tide-calibration`. JSON `null` clears it. This is a
measurement record, not an audited LGU approval process or confirmation snapshot.
The existing private-schema access model and security-invoker compatibility view
are unchanged. No new public grants or new FK indexes are needed.

## Scientific basis

The [ZSL mangrove rehabilitation manual hosted by DENR](https://faspselib.denr.gov.ph/uploads/materials/fec8cc55-253a-4788-b26b-3104b6be9a6d/586ca897-52d7-4432-b2f3-08387292e2f4.pdf)
describes middle-to-upper intertidal sites at/above MSL, inundated no more than
30% of time (printed page 2), and daytime low-tide planting (printed page 54).
**30% of time is not 30% of tidal range.** The manual does not establish one
universal instantaneous tide height for planting. Long-term hydroperiod and
species suitability are not inferred from this short forecast.

**[uncertain / provisional]** The app's instantaneous boundary is zero predicted
water depth above the estimated ground height (or measured ground height in the
separate surveyed-site assessment): `sea_level <= substrate_elevation`.
This conservative exposure convention is an implementation choice, not a
published universal safety threshold. Exposed mud may still be dangerous; inspect
access, footing, weather, waves, currents and local restrictions independently.

Rice cultivation is different: [IRRI water-management guidance](https://www.knowledgebank.irri.org/step-by-step-production/growth/water-management)
describes approximately 3 cm initially **after** transplanting, later 5–10 cm.
Those freshwater field-management depths are not mangrove tide-height thresholds.
Tides cannot directly predict water depth in bunded or hydraulically isolated paddies.

## Providers and NAMRIA

Recommendation: keep the working API and supplement local verification with
NAMRIA; do not claim one global model is most accurate for all Philippine coasts.

| Provider | Coverage / granularity | Cost / constraints | Philippine reliability considerations |
| --- | --- | --- | --- |
| [Open-Meteo Marine](https://open-meteo.com/en/docs/marine-weather-api) | Global, hourly tide/sea-level model, about 8 km grid; actual response governs horizon | [Free noncommercial access](https://open-meteo.com/en/pricing); commercial use needs appropriate service | Daily tide-model refresh; documented limitations at coastlines. Global MSL reference requires survey compatibility. |
| [WorldTides](https://www.worldtides.info/apidocs) | Global model / station-dependent; requested half-hour heights plus extremes | [Credit-based subscriptions](https://www.worldtides.info/developer); server-side key, required attribution, check multi-user caching licence | Explicit `datum=MSL`, but actual `responseDatum` must be checked. No verified Philippine-wide superiority benchmark. |
| [NAMRIA](https://namria.gov.ph/kiosk/namria02.htm) | Official station predictions, hourly data and datum/benchmark services by request | Request process and applicable product fees/eligibility | Authoritative Philippine verification source. No supported public developer feed was verified. The app links to the request service; it does not scrape or pretend NAMRIA is supplying its live curve. |

To evaluate WorldTides, configure the existing server-only `WORLDTIDES_API_KEY`;
the adapter now requests `heights`, `extremes`, `datum=MSL`, `localtime`, `timezone`
and `step=1800`, mapping `dt/height` to `timestamp/height_m`. Preserve attribution.
WorldTides failure retains last-known data or falls back to Open-Meteo visibly;
stale or reference-incompatible data cannot produce a suitable badge.

For direct NAMRIA integration, obtain an authorized machine-readable feed or
licensed dataset, station identification, supported datum conversion and reuse
terms first. Validate predictions against local observations when possible;
agreement with another prediction alone is not proof of accuracy.

## Deployment

Apply the additive Alembic migration `20260911_0003` before restarting the backend:

```powershell
venv/Scripts/python.exe -m alembic upgrade head
venv/Scripts/python.exe scripts/verify_tide_calibration.py
cd MangroVision_New/client
npm run build
```

Before upgrading from `20260904_0002`, the optional verification script's
`--rehearse` mode checks upgrade/downgrade/upgrade inside a rolled-back transaction.
It never commits changes.

Green `#15803D`, red `#B91C1C`, and gray `#4B5563` all exceed 4.5:1 contrast
on white. Text and symbols convey the same states without relying on colour.
Check the finished layout on desktop and mobile, including the graph table fallback.
