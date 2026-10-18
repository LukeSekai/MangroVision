# Seedling health and replanting

## Planting performance

Dashboard Seedling Health and the survival and monitoring report select
planting batches by the chosen planting dates and evaluate their recorded
outcomes as of generation time. Each planting event represents one seedling.

- **Seedlings planted:** actual original and replacement plantings in the dates.
- **Without recorded death:** open planting cycles without a saved death.
  These include seedlings awaiting inspection; survival is not field-verified.
- **Recorded deaths:** saved deaths for the selected seedlings, including those
  later released for replanting or replaced.
- **Recorded survival:** seedlings without a recorded death divided by seedlings
  planted, multiplied by 100. An empty batch has no rate. Missing earlier
  outcomes are shown separately and make the affected group's rate unavailable.

The **Survival by planting batch** table groups by Philippine planting date,
assignment, organization, site, species, and original/replacement planting.
Species, site, and planter comparisons use these planting-cycle totals.
Approving or assigning replacement work changes no historical outcome.
Original deaths remain in their batches. An actual replacement planting adds
a separate seedling; repeated deaths at one location remain separate events.
Reversed planting completions and future plantings are excluded. Historical
attribution comes from the planting event, even if the location is later moved
to another organization or removed from the map.

The overall rate combines the selected batches. For example, 80 surviving
original seedlings and 20 deaths produce 80% recorded survival. Approval stays
at 80%. Planting 20 replacements produces 100 seedlings without recorded death
out of 120 planted, or 83.33% overall; the original batch stays at 80% and the
replacement batch begins at 100% until further outcomes are recorded.

## Replanting progress and current map status

Replanting progress covers all planting dates for the selected dimensions:

- **Awaiting review:** recorded dead seedlings at retained locations awaiting
  approval. They remain dark red on the map.
- **Ready for LGU replanting:** approved, eligible locations reserved for the
  LGU team. They appear blue and count as Assigned on the map. Eroded locations
  appear Unavailable and cannot be recorded as replanted while erosion applies.
- **Replacements planted:** dead seedlings followed by a recorded
  replacement cycle. A replacement that later dies still counts as completed
  work for its predecessor and separately creates pending work for itself.

**Current mapped seedlings** continues to count each current planted or dead
physical location once across all planting dates, matching Planting Map.
Approval reserves a location for LGU work. Its stored physical value is Planned,
while its current map classification is Assigned. It leaves current seedling
totals; its earlier planting and death remain in planting performance.
Current map counts and historical planting totals can therefore differ.

The API keeps its current-map `summary` and comparisons for compatibility and
adds `planting_performance` for batch results and replanting progress. These
are assembled by `mangrovision_db/health_summary.py` and
`mangrovision_db/planting_performance.py` through `get_dashboard_ecology`.
Staff handle replacements in **Monitoring → LGU Replanting** using their
existing login. The queue uses existing approval records, so no new account or
schema migration is needed. Actual planting creates a staff-attributed event
linked to the original seedling. See [LGU replanting](lgu-replanting.md).

## Monthly death history

**Seedling deaths by month** shows newly recorded deaths in each Philippine
calendar month within the selected dates. The chart and its data table use
counts for that month, not a running total. Months with no reported deaths
show zero. The existing project-site, species, assignment and planter filters
apply to the history independently of the planting-date selection used for
batch performance. A May batch can have deaths reported in June; these appear
in May's planting performance and June's death history respectively.

The series uses the same records as **Why seedlings died**. Located deaths
are counted once even when an individual inspection and an organization visit
refer to the same death. Unlocated deaths are the remaining newly reported
visit balance, not cumulative deaths from previous visits. Those unlocated
counts are omitted when a site, species, assignment or planter filter is active
because their attribution is unknown. Linked deaths use the visit's reporting
date; otherwise they use the saved death date. Previous planting cycles stay
in this history after replanting or releasing a location.
