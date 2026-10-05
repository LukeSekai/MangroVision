# Restoration reports

LGU staff open the report dialog from **Download Report** in Dashboard or **Download Monitoring Report** in Monitoring. Dashboard carries its current date range and project site into the dialog. Overview and Project Sites start with planting accomplishment; Planting Work starts with organization activity; Seedling Health and Monitoring start with survival and monitoring. Staff can choose any of the four report types in the dialog.

Reports is no longer a separate sidebar section. Old `/reports` links redirect to Dashboard. The shared generator reads existing planting, dashboard and monitoring APIs. It does not require a database migration or change planting records.

## Available reports

| Report | Scope |
| --- | --- |
| Planting accomplishment | Seedlings planted within the selected dates, grouped by month. Replacement seedlings are included. The cumulative column starts at the beginning of the selected reporting period. |
| Survival and monitoring | Seedlings planted within the selected dates, with LGU observations through the period end. Results remain separated by inspection age, site and species. Measured height is included when available. Organization-level visits during the period appear separately when all sites are selected. |
| Mortality and replanting | Reported seedling deaths by cause, plus individual dead seedlings with recorded death dates in the period and their current replanting status. Cause totals use the visit reporting date where applicable; these dates can differ from the recorded death dates in the location table. |
| Organization activity | Assignment points marked planted or skipped during the period, alongside current pending work. Rows are grouped by the organization responsible for each site; sites without an organization are explicitly identified as unassigned groups. |

Choose year to date, this quarter, last quarter or custom dates, and optionally a project site. Monitoring opens with year to date and all sites because its workspace currently has no date or project-site filter. The preview updates automatically when the report type or a filter changes, and when records change. Controls stay available while loading; requests for an earlier selection are cancelled. Downloads appear when the latest preview is ready. Invalid date ranges show a validation message and do not request a report.

The dialog opens below the workspace header, keeps its Close button visible while its contents scroll, supports Escape and keyboard focus containment, and returns focus to the opening button on close. Closing cancels pending report requests and PDF downloads. Print output includes the document without Dashboard, Monitoring or dialog controls.

The preview records its site, date range, generation time in Asia/Manila and the signed-in officer's display name. Staff can add optional remarks. This identifies the preparer; it does not apply a signature or certify government acceptance.

## Downloads

- **Print** opens the browser's printer dialog. Print styling hides workspace controls, uses landscape A4 and repeats table headers across pages.
- **Save as PDF** downloads a PDF directly, without opening the printer dialog. It uses embedded Inter fonts, the MangroVision logo, repeated section/table headings and page numbers. Wide tables use landscape A4; narrower reports use portrait A4. The document uses the displayed report snapshot, including its metadata, N/A values, reading notes and optional remarks.
- **Download CSV** includes the report metadata, summary, supporting tables, reading notes and remarks. It uses UTF-8 with a BOM for Excel and quotes multiline text. Formula-like text is neutralized before spreadsheet export.

The PDF renderer is an authenticated LGU/admin/planner endpoint at `POST /api/export/report/pdf`. It escapes supplied text, validates table dimensions and limits, returns an attachment with `Cache-Control: no-store`, and makes no database changes. ReportLab is included in `requirements.txt`; its Inter font assets and license are shipped in `MangroVision_New/api/assets/fonts`. This version does not generate XLSX or DOCX files. Existing KML/GeoJSON exports and the monitoring field-sheet workflow remain available in their existing screens.

## Reading the figures

The accomplishment report counts seedlings whose planting was recorded within the selected dates. Each planting record represents one seedling. Replacement seedlings planted at reused locations add to that total. Assigned points alone do not count as seedlings planted, and the total includes seedlings that later died. Monthly labels show a month and year, rather than implying planting occurred on the first day of the month; CSV and PDF exports use the same labels.

Observed survival rate uses alive divided by alive plus dead, multiplied by 100, at a specific scheduled inspection age after planting. Missing seedlings and those awaiting inspection are excluded from that rate. No known alive/dead outcomes produces N/A. Inspection completion rate shows completed inspections divided by inspections due, multiplied by 100, at the reporting period end. Earlier deaths are included in later dead counts and excluded from inspection workload; do not add them again. Keep inspection-age totals separate. The summary cards identify which inspection age they cover.

Organization-level visits show their reported alive/dead balances, health and recorded follow-up actions separately. Those balances are not individual inspection results and must not be summed across visits. They do not identify a particular project site, so they are excluded when a site filter is selected. The loader follows visit pagination until it reaches the end of the period; it does not silently export only the first page.

Reported deaths are counted once. Cause totals use the monitoring visit date when a death is linked to a visit, or the individual recorded death date otherwise. The location table uses recorded death dates, so its period counts can differ from cause totals. Each location-table row identifies a dead seedling record; multiple seedlings can have occupied the same physical point at different times. Do not add the two tables' death counts together. Project-site filters exclude deaths without known locations. Replanting approval and assignment remain preparation steps; completed replanting requires a recorded replacement seedling planting. Replanting status and organization pending work are current when generated, including for reports about earlier periods.

Financial utilization, official quarterly target comparisons, certifications, attendance, carbon estimates and certified restoration area are outside this version. Those require information beyond the operational records used here.

## Verification

From `MangroVision_New/client`:

```powershell
node --test src/utils/restorationReports.test.js src/pages/RestorationReports.test.js
npx eslint src/utils/restorationReports.js src/utils/restorationReports.test.js src/components/RestorationReportDialog.jsx src/pages/RestorationReports.jsx src/pages/RestorationReports.test.js
npm run build
```

The tests check seedling/assignment distinctions, inspection denominators, missing observations, earlier deaths, timezone/date boundaries, replanting filters, consistent report wording and month labels, unassigned organization groups, safe CSV output, escaped report text, rendered report metadata and initial report scope from the source screen. Visual browser and print-preview inspection should be performed when a browser connection is available.

From the repository root, with `pytest` and `pypdf` available in the development environment:

```powershell
.\venv\Scripts\python.exe -m pytest MangroVision_New/api/test_report_pdf.py -q
```

These checks cover PDF attachment headers, LGU access, embedded font weights, repeated table headings, complete row pagination, long remarks, empty data and escaped text. The four report layouts were also rendered to page images and visually reviewed using synthetic records.
