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
Releasing locations or assigning them again changes no historical outcome.
Original deaths remain in their batches. An actual replacement planting adds
a separate seedling; repeated deaths at one location remain separate events.
Reversed planting completions and future plantings are excluded. Historical
attribution comes from the planting event, even if the location is later moved
to another organization or removed from the map.

The overall rate combines the selected batches. For example, 80 surviving
original seedlings and 20 deaths produce 80% recorded survival. Release stays
at 80%. If the 20 replacements belong to another site, that new site has 20 new plantings
and its own survival result. The original site stays at 80%. An all-sites report
combining both batches has 100 seedlings without recorded death out of 120
plantings, or 83.33%; this is a different scope from either individual site.

## Released locations and current map status

**Current mapped seedlings** counts each currently planted or dead physical
location once across all planting dates. **Release from project site** removes
selected dead planting cells from the site's current boundary and returns their
locations to green Planned. Earlier planting and death records remain in the
original site's planting performance. A newly claimed site uses its own new
planting events and survival results.

Map site popups use historical site planting outcomes for survival; their status
donuts continue to show current mapped locations. Releasing dead cells cannot
increase the popup?s survival rate.

There is no dedicated LGU replacement queue or replacement form. Monitoring's
**LGU Sites** tab manages the fixed LGU organization's separate planting sites,
permanent planter access, and site-specific monitoring. See
[LGU Sites](lgu-sites.md) and [Release dead locations](project-site-location-release.md).

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
