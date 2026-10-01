# Mangrove growth categories for monitoring

Research checked 12 September 2026. Review interval: 14 days, chosen for MangroVision's monitoring workflow, not a biological stage duration.

## What the studies support

| Species and setting | Reported height increment | Arithmetic equivalent over 14 days |
| --- | --- | --- |
| Rhizophora apiculata, Guang-guang, Mati, Philippines | 5.55, 6.25 and 5.51 cm in one month at three stations | About 2.6–2.9 cm, using reported increment × 14/30 |
| Avicennia marina (Bungalon), Philippine nursery, three substrates | 0.28, 0.27 and 0.30 cm/week, in an 18-week experiment | 0.56, 0.54 and 0.60 cm, using weekly rate × 2 |

These conversions are estimates derived from published averages, not measured two-week increments or universal healthy-growth limits. The settings and seedling ages differ, so this is not a species performance comparison. R. apiculata findings must not be applied automatically to every Rhizophora species. Bungalon is identified as A. marina in the Philippine handbook below.

Sources:

- [Growth dynamics and survival of mangroves (Rhizophoraceae) seedlings in Guang-guang, Mati City, Davao Oriental, Philippines (2023), Table 1, p. 538](https://bioflux.com.ro/docs/2023.534-545.pdf). Use the table values; the surrounding prose contains a small inconsistent final height.
- [Albarico (2023), Growth and Survival of Avicennia marina and Bruguiera cylindrica in Different Substrates, Table 2, p. 245](https://forestist.org/Content/files/sayilar/450/241-246%281%29.pdf). The table reports treatment means with variation; the displayed range is across means, not a confidence interval. This nursery result is not a prediction for planted coastal sites.
- [Primavera et al., Handbook of Mangroves in the Philippines – Panay, p. 24](https://repository.seafdec.org.ph/bitstream/handle/10862/3053/9718511652.pdf?sequence=1), for the local name Bungalon.

## Earlier observed-size categories (historical records only)

Earlier visits used observed size groups adapted from the [DENR/CRMP Participatory Coastal Resource Assessment Training Guide (Deguit et al., 2004), pp. 74–75](https://oneocean.org/download/db_files/pcra_training_guide.pdf#page=85):

- **Seedling:** up to 1 metre tall; stem less than 4 cm across.
- **Young mangrove:** over 1 metre tall; trunk up to 4 cm across.
- **Larger mangrove:** over 1 metre tall; trunk more than 4 cm across.
- **Mixed sizes:** more than one stage in the organization's planting areas.
- **Not checked this visit:** an explicit missing observation.
- **No living seedlings:** filled automatically when the alive count is zero.

The guide calls these seedling, sapling, and mature tree. Its sapling definition prints "of 4 cm" rather than a complete interval; the app uses an explicitly adapted thin-trunk group up to 4 cm. These are field size groups, not proof of reproductive maturity or age. Borderline plants require a field check; dwarf forms may need specialist assessment. Use a marked reference stick; do not measure trunk circumference as diameter. For trees, the guide shows trunk measurement near 1.3 m or above irregular stems/roots.

Height at planting, nursery age, salinity, water exposure, nutrients, damage and site conditions prevent a reliable conversion from elapsed days alone to actual size/maturity. True height increments need repeated measurements on comparable plants.

## Application behavior

- Growth is automatic, as requested by the user: Philippine calendar days since each actual planting, divided into 14-day groups. Under 14 days is **New seedling**; subsequent groups are **Growing seedling**, with the exact two-week age range. These are application age groups, not scientific size or maturity thresholds. New plantings begin in the first group, independently of older batches.
- Each visit stores its calculation in `growth_snapshot` with the method version and visit date. History and dashboard use that snapshot, so previous visits do not age when today changes. Missing planting dates remain unknown.
- No height or stage is entered by the user. Previously observed stages and measured heights remain in storage. Historical age snapshots are derived separately from planting dates at each original visit.
- Dashboard → Seedling Health shows one latest snapshot per organization in the selected dates and a next check 14 days later. Batch counts are originally planted counts, including past deaths, since aggregate deaths cannot be attributed to individual batches without more evidence.
- The form loads the latest visit when opened and carries forward plant health and LGU actions. Newly dead starts at zero. Previous deaths are cumulative totals and are never summed across visits.
- Alive before a visit equals the saved alive balance plus new planting events. New event IDs are snapshotted, so later additions and replanting are counted once. New deaths reduce this balance and add to cumulative deaths; survival uses the cumulative planted denominator. Legacy inconsistent lower death totals cannot resurrect previously recorded deaths.
- Saving locks the organization and checks both the baseline visit ID and expected alive balance. A stale form must reload. Earlier dates cannot be appended before the latest visit, and future visits cannot be recorded.
- Organization visits cannot be attributed to an individual project site/species. The growth summary asks users to clear narrower filters rather than present organization-wide results as site-specific data.
- The two-week review suggestion does not overwrite scheduled visit dates or block visits on other dates.

Storage: revisions `20260912_0004` and `20260912_0005`. New fields separate new deaths, alive before a visit, baseline record, count context, and automatic growth snapshots. Downgrades refuse to erase saved evidence.

User-confirmed correction: Katunggan record 29 (September 12) reported **50 additional deaths** after record 28's 135 deaths. It now stores 185 cumulative dead and 148 alive out of 333. The previous values and correction reason are retained in `count_snapshot`; the original manual stage and other visit details are preserved. The guarded correction script is `scripts/correct_monitoring_progress_20260912.py`.
