# LGU mangrove restoration reports: research and product recommendation

Research date: 4 October 2026. Scope: Philippine LGU mangrove/coastal restoration reporting and the current FastAPI/React MangroVision application. This is a research and design proposal; formatted report generators described below are not implemented by this document.

Implementation follow-up: the applicable operational reports are now available as described in [Restoration reports](restoration-reports.md), with browser printing/PDF saving and CSV downloads. The research proposal below also discusses formats and data additions beyond that implementation.

## Recommendation

Start with a **Quarterly Restoration Accomplishment Report** and a **Survival and Monitoring Report**, followed by a **Mortality and Replanting Action Report**. Produce a printable PDF and an XLSX workbook of supporting tables. Add maps and available inspection photos as annexes. Offer DOCX later for narrative reports that the LGU needs to edit.

The report template must match the receiving office and funding agreement. The reviewed sources do not establish one universal mangrove restoration form for every LGU program. MangroVision can supply operational figures and evidence for prescribed submissions; the responsible office supplies approved targets, financial records, review and signatures.

## Official formats and their relevance

| Reference | What it establishes | How MangroVision could support it |
| --- | --- | --- |
| DBM 2023 LGU budget manual, LBAc Form 3, printed page 208 | Quarterly physical reporting compares approved outputs with actual and cumulative accomplishment, explains variance, and identifies the responsible office. | Provide a restoration output table that can be transferred into the prescribed form. Add approved project codes and quarterly targets. |
| Same manual, LBAc Form 5, printed page 211 | Semester evaluation combines physical results with allotments and obligations. | Supply physical figures. Financial columns require records from the budget/accounting offices. |
| DENR DAO 2019-03, section 15, document page 12 | For the expanded National Greening Program, DENR field-office quarterly/annual reporting uses spatial files, drone imagery and geotagged photographs as supporting evidence. Annual reporting also involves formal certification. | Assemble a map/photo evidence pack. Add shapefile export and verified photo-location metadata where required. Certification remains with the responsible office. |
| DILG Region XI forms catalog | Contains work accomplishment, project completion and turnover forms associated with the Performance Challenge Fund. | Illustrates why completion reports must follow the particular funded program's template. It is not a universal restoration form. |

Sources: [DBM manual](https://www.dbm.gov.ph/wp-content/uploads/Issuances/2023/Local-Budget-Circular/Budget%20Operations%20Manual%20for%20LGUs,%202023%20Edition.pdf), [original DENR order archived by FAO](https://faolex.fao.org/docs/pdf/phi190722.pdf), and [DILG Region XI catalog](https://region11.dilg.gov.ph/wpv1/downloadable-forms/). The DBM form layouts and relevant DENR section were visually inspected, alongside their text.

A [DENR library entry for a Lebak mangrove rehabilitation completion report](https://faspselib.denr.gov.ph/Materials/Detail/3000a476-b7b6-4460-889f-5a42245bc379) offers a historical example of reporting project results and community implementation. It was published in 2003; the library's 2026 upload date does not make it a current reporting requirement.

## Reports the system's existing data can support

“Available” below means the underlying data or workflow exists. A formal document export still needs to be built.

| Proposed report | Useful contents | Current readiness and remaining work |
| --- | --- | --- |
| Quarterly restoration accomplishment | Planting events during the period, site/species/organization breakdowns, initial and replacement planting, cumulative totals, approved targets, variance and explanations. | Planting history and filtered dashboard aggregates exist. Approved quarterly/site targets, project identifiers, narrative remarks and document layout need work. |
| Survival and monitoring | Planting cohort and inspection age, alive/dead/missing results, uninspected seedlings, observed survival, inspection coverage, measured height, condition, actions and overdue inspections. | Individual LGU observations, monitoring rounds, optional photos and dashboard aggregates exist. Build an export with explicit denominators and dates. |
| Mortality and replanting action | Death causes, known death locations, unlocated deaths, replacements approved, replacements actually planted and outstanding work. | Death links, location corrections, approval records and replacement event links exist. Preserve the distinction between approval and completed planting. |
| Organization activity | Assigned/planted/pending/skipped points, contributions by site and period, declared participant count. | Organization activity and shared-account participant slots exist. A verified attendance roster does not exist. |
| Map and photo evidence annex | Site outlines, planting/inspection locations, coordinates, status legend, imagery date, photo captions and linked observations. | Coordinates, GIS exports, map views and optional monitoring photos exist. Add a print map layout, evidence selection and required metadata. |
| Annual or project completion narrative | Objectives, accomplishments, monitoring findings, maintenance needs, problems, recommendations and annexes. | Operational sections can be drafted from the above reports. Background, approved commitments, verified area, finance and officer review need additional inputs. |

A financial utilization report is a later phase. The inspected reporting code does not provide a budget/allotment/obligation/disbursement ledger. Planting counts cannot substitute for official financial figures.

## Recommended file formats

| Format | Intended use | Current position |
| --- | --- | --- |
| PDF | Review, printing and final signed reports; map/photo annexes. | A formatted report generator is needed. A printable monitoring field sheet already exists. |
| XLSX | Detailed accomplishment and monitoring tables that staff can filter, check and reuse. | A structured workbook export is needed; waypoint CSV is already available. |
| DOCX | Editable narrative, findings, recommendations and completion reports. | Add after the first PDF/XLSX reports. |
| KML / GeoJSON | Spatial annexes and GIS exchange. | Existing waypoint exports can be reused, with appropriate report filters. |
| Zipped shapefile | Submission where the receiving program specifically requires that format. | Add conversion/export, including the associated geometry, attributes and projection files. GeoJSON alone does not fulfill a shapefile request. |

These are product recommendations, not a claim that every receiving office accepts every format.

## Proposed first report pack

1. **Identification:** LGU and office, project/site and barangay, project code, reporting dates, generation time, data cutoff and selected filters.
2. **Accomplishment:** approved targets, actual planting events, cumulative results, variance, site/species/organization breakdown and separate replacement planting.
3. **Monitoring:** results for comparable planting cohorts and inspection ages, coverage, mortality causes and recorded growth measurements.
4. **Follow-up:** inspections due, maintenance or replacement work, explanations and responsible officers supplied by the LGU.
5. **Evidence:** a readable map, coordinates/detail workbook and available photos captioned with observation date and linked location.
6. **Review:** spaces for the LGU's preparer, reviewer and approving officer. A generated draft must not invent signatures or certification.

A short summary PDF with separate detail and evidence annexes will serve both office review and field follow-up. An eventual Reports screen should allow report type, period, project site and organization selection, then preview and download. Label drafts and retain the generation time and filters; retaining an immutable issued snapshot would make later corrections traceable.

## Calculation and data-quality rules

- **Assignment is planned work.** Count completed planting events for planting accomplishment. When a physical point is reused, preserve its original planting/death history and distinguish the replacement event.
- **Observed survival:** for a specified LGU inspection cohort, use `alive / (alive + dead) * 100`, matching the current individual-observation dashboard calculation. Show the denominator, missing count and uninspected count beside the percentage. No observations means N/A. This is not automatically a whole-site or DENR sampling estimate.
- **Inspection coverage:** show completed inspections relative to those due for the selected round. Historical deaths carried forward into later rounds are not new field inspections. Keep their treatment consistent with the existing dashboard.
- **Two mortality workflows:** individual LGU observations and organization-level reported mortality balances have different evidence and denominators. Identify the source; do not silently combine them into one survival percentage.
- **Death locations are a subset of reported deaths.** Adding a location does not add another death. Unlocated organization deaths cannot be attributed to a particular site/species without supporting data.
- **Area needs its own basis.** Image coverage, a project boundary, predicted plantable area and verified area restored are different measures. Do not derive certified hectares restored by summing point buffers or overlapping imagery footprints.
- **Photos are optional.** A photo linked to a mapped observation is not proof that the photo has independently verified GPS metadata. Keep that distinction visible in any evidence annex.
- **Targets must be approved.** An annual target or the dashboard's prorated progress reference does not establish official quarterly targets. Missing targets remain blank or N/A.
- **Participants need evidence.** Shared-account/device slot counts are not signed attendance, individual beneficiary records or a sex-disaggregated roster.
- **Dates and scope must be explicit.** Separate planting within the reporting period from monitoring status as of the cutoff. Compare cohorts at comparable ages; avoid a single unlabeled survival percentage across unrelated rounds.

## Inputs to add before formal submission exports

- LGU/office identifiers, project and funding-program identifiers, official output units and quarterly targets.
- Officer names and designation fields, review status and explanations for variance.
- Field-verified area, survey/reference information and the receiving office's map/evidence requirements.
- Actual attendance/beneficiary records when required by the program.
- Official financial data imports when financial reporting is in scope.
- Sampling methodology and additional ecological measurements if a full plantation assessment is required.

## Existing implementation references

- [`export.py`](../MangroVision_New/api/routes/export.py): CSV, GPX, KML and GeoJSON waypoint downloads.
- [`dashboard.py`](../MangroVision_New/api/routes/dashboard.py) and [`planting_database.py`](../planting_database.py): date/site/species/organization filters, annual settings and operational/ecological aggregates.
- [`monitoring.py`](../MangroVision_New/api/routes/monitoring.py): individual LGU observation inputs and monitoring workflows.
- [`PlanterActivityReport.jsx`](../MangroVision_New/client/src/pages/PlanterActivityReport.jsx): interactive organization activity view.
- [`monitoring-death-locations.md`](monitoring-death-locations.md): printable field sheets, reported versus located mortality, and replacement planting history.
- [`organization-accounts.md`](organization-accounts.md): shared organization accounts and participant-slot meaning.

This assessment reads the repository's data structures and workflows. It does not certify that every existing project record has complete observations, photos or approved reporting metadata.
