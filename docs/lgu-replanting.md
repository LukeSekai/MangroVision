# LGU replanting

Staff use **Monitoring → LGU Replanting** with their existing staff login.
There is no new planter account, organization, participant slot or access code.
The heading, section navigation and **Show map** action match Monitoring visits.
Use the map's upper-left **Back to LGU Replanting** button to return to the queue
after opening the monitoring map or viewing a specific location.

1. Open **Awaiting review**. Filter by project site or search by photo, point,
   site or original organization. Select eligible deaths, then review and approve.
2. Approval releases the old organization allocation and reserves the location
   for LGU work. It appears blue as **Assigned — LGU replanting**. Organization
   assignment paths reject these reserved locations.
3. After physical planting, select the locations in **LGU queue** and choose
   **Record replacement planting**. All selected points must be inside the
   chosen active project site. Confirm the actual date and species. Initial
   height, condition and notes are optional.
4. Confirmed replacements appear yellow. The replacement event records the
   authenticated staff member and links to the original planting event.
   Original planting attribution, deaths, causes and observations are retained.

The dashboard's **Seedling Health** tab shows Awaiting review, Ready for LGU
replanting, and Replacements planted. These progress counts cover all planting
dates for the selected site. Ready locations also count as Assigned on the map;
they are overlapping views, not additional physical points. Planting Work lists
LGU replacement work separately from organization allocations.

Each original batch keeps its deaths and survival calculation after approval
and replanting. Actual replacements create separate batches. If a replacement
later dies, its death remains a separate event and can begin another review.

Eroded and removed locations cannot be approved or planted. A location covered
by erosion after approval remains reserved, appears orange as Unavailable,
and cannot be planted until eligibility changes. Queue records expose the
reason and disable the action. Stale versions, a different site, an existing
allocation or an earlier planting date reject the whole submitted batch.

The implementation uses existing `replanting_requests` and `planting_events`.
There is no schema migration or automatic record backfill. Old organization
replacement assignments retain their historical ownership; new work belongs
to the LGU queue. Writes are transactional and versioned. Double submission
cannot create two replacements for the same dead seedling.

Development checks use rollback-only temporary PostgreSQL tables:

```powershell
$env:MANGROVISION_RUN_LGU_REPLANTING_TESTS='1'
.\venv\Scripts\python.exe -m pytest tests/test_lgu_replanting_postgres.py tests/test_lgu_replanting_api.py -q
```

From `MangroVision_New/client`, run `node --test src/pages/LGUReplanting.test.js`
for the interactive approval, planting and error flows.
