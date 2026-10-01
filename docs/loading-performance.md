# Workspace loading performance

MangroVision continues to use its online Supabase database. No Docker migration
or new index is required for this change.

## Database reads

- Organization summaries batch latest visits and located deaths, and read
  planting events once for all organizations. Five organizations now use four
  SELECTs instead of 28. Historical planted/alive/dead counts still include
  released locations; these remain separate from current mapped ownership.
- History pages batch visit details and locations: three SELECTs instead of
  one plus two per visit. Date/ID pagination order is unchanged.
- The map selects the preferred assignment using PostgreSQL `DISTINCT ON`,
  preserving active/pending, completed, then other assignment priority, with
  date and ID tie-breakers. This lets the planner recognize one row per point.
- Dashboard filter options use one query instead of four. Overview reuses its
  monitoring settings and unfiltered operational totals, and combines data
  quality counts into one query. Current point selection also uses `DISTINCT ON`.
- Project-site counts aggregate point membership separately from assignments,
  analyses, and schedules. This removes a large OR join while preserving unique
  point counts, including points associated with a site through both paths.

On 2026-09-27, missing planner statistics caused a one-row estimate for 1,170
assignments, producing millions of join comparisons. The configured maintenance
connection ran `ANALYZE` on these four tables:

```sql
ANALYZE mangrovision.planter_assignment_points,
        mangrovision.planter_assignments,
        mangrovision.planters,
        mangrovision.planting_points;
```

This updates PostgreSQL's planner statistics, not application records. The
application role cannot perform that maintenance; verify statistics afterward
instead of assuming a command without an exception succeeded. Normal autovacuum
should maintain statistics afterward; repeat maintenance after major imports or
restores if plans show inaccurate row estimates.

## Browser caching and freshness

`utils/apiReadCache.js`, installed by `secureFetch.js`, caches only an explicit
list of workspace GETs for 30 seconds in memory: map points, analysis statistics,
organization summaries and history, project sites, map zones, dashboard reports
and settings, schedules, planter and assignment lists, assignment points, and
saved analysis lists/details/points. Concurrent callers share one network read
and receive independently readable Response bodies. The cache holds at most 64
URLs; filters and pagination cursors are part of each key.

- Page navigation during the TTL reuses the response; no browser storage or
  shared server cache stores authenticated results.
- API mutations clear entries both before and after the request. Successful
  mutations refresh mounted dashboard, scheduling, planter management, monitoring
  summaries/history, and shared map data where it is displayed.
- Reads started before a mutation cannot restore an older cached snapshot,
  including mutations made while a response body is still downloading.
- Returning focus to the app invalidates cached reads and refreshes mounted
  workspace data, including changes made from another browser or planter device.
- Dashboard, scheduling, planter management, and monitoring Refresh buttons
  invalidate or bypass cached data. Store fetch actions accept `{ force: true }`
  for explicit refreshes.
- Login/logout clear the cache and map state. Old-session requests are rejected.
- Errors are not cached. Cancelling one caller stops that caller from receiving
  the response without aborting another caller's shared request. Explicit
  `no-store` requests bypass the cache.
- Visit baselines, death-location selection data, planter access, authentication,
  and shared-link endpoints are not cached. Server write validation remains
  authoritative. Tide forecasts retain their existing server cache.

The dashboard loads only the report needed by the selected tab. It no longer
mounts a hidden map or fetches map points on entry. Switching tabs can reuse
recent reports within the cache lifetime. After removal of the Data Checks tab,
its useful record notices load through a small, staff-protected
`/api/dashboard/record-notices` read (one aggregate SQL statement), cached under
the same freshness rules. Work, Health, and Sites display only their relevant
missing-information warnings. Overview does not request this additional read.

This is not realtime push: changes from a different device appear on the next
uncached read, manual refresh, or return to the window. TTL expiry alone does
not poll a page that remains open and idle.

## Measurement and validation

Run `venv/Scripts/python.exe scripts/profile_loading.py` for warm-connection,
read-only measurements and query plans. It prints counts/timings rather than
records or credentials. `--baseline-functions` optionally accepts a locally
captured, trusted source snapshot to compare the previous implementation.

Three-run medians on the resumed Supabase project, 2026-09-27:

| Read | Original | Updated | SELECTs before / after |
| --- | ---: | ---: | ---: |
| Five organization summaries | 1.952 s | 0.489 s | 28 / 4 |
| Thirteen history records | 1.893 s | 0.429 s | 27 / 3 |
| 3,206 map points | 1.944 s | 1.408 s | 5 / 5 |
| Dashboard overview | 7.144 s | 3.234 s | 26 / 12 |
| Dashboard operations | 0.881 s | 0.678 s | 6 / 3 |
| Dashboard ecology | 1.461 s | 1.375 s | 10 / 7 |
| Dashboard sites | 3.844 s | 1.826 s | 11 / 8 |

The main map SELECT's server execution fell from 819 ms to 16 ms with the new
query and refreshed statistics. The project-site counts query fell from about
2,850 ms to 5 ms. These are domain-service timings, including
database transport and Python processing, but excluding HTTP authentication,
response transfer to the browser, and rendering. Network timing varies. All
returned summaries, history records, map points, and dashboard reports exactly
matched the original implementation in the comparison. Use `--dashboard` to
profile reports and `--site-id ID` to compare a filtered dashboard.
An additional site-filtered comparison also matched all four original reports;
filtered Overview retains a separate query for global operational totals.

Regression checks cover cache reuse, expiry, refresh, mutation races, logout,
failures, uncached sensitive reads, cancellation isolation, filtered cache keys,
bounded retention, selected dashboard tabs, history pagination, constant query
counts, project-site membership overlaps, and preserved deaths/statistics after
replanting approval. PostgreSQL tests use
disposable schemas without changing application records. Interactive browser
verification was unavailable in this session.

## Further point-loading improvements

A subsequent comparison on 2026-09-27 addressed first loads without extending
cache lifetimes or dropping point fields:

- The map query resolves source-site ownership and erosion coverage in the
  same SQL read as the point/assignment data. Explicit site links still win;
  legacy points resolve only when exactly one polygon covers their coordinates.
  Boundaries remain inclusive, and overlapping legacy sites remain unassigned.
- The point endpoint serializes the already-normalized domain result directly,
  avoiding a second recursive normalization pass through thousands of records.
- Selected map/statistics/dashboard GET responses use gzip level 3. This is
  lossless transport compression: coordinates and numeric precision are intact.
  Authentication, shared access, writes, tiles, and processing streams retain
  their existing transport. Clients without gzip support still receive JSON.
- The map uses Leaflet's canvas renderer and reuses species-spacing calculations
  across selection changes. Its redraw comparison now includes all point fields
  so coordinate, ownership, and popup-detail changes are also rendered.
- Dashboard Overview shares the request's database connection and already-read
  settings with its inspection calculation. Historical report years still use
  the correct settings independently of current operational inspection dates.
  This does not introduce a shared or longer-lived data cache.

Three-run medians compared immediately before and after this second change:

| Read | Before | After | SELECTs before / after |
| --- | ---: | ---: | ---: |
| 3,206 map points | 1.930 s | 0.969 s | 5 / 2 |
| Dashboard overview | 2.917 s | 1.817 s | 12 / 11 |

These remain backend domain timings, not complete browser loading measurements.
The full map JSON measured 5,489,872 bytes before compression and 167,896 bytes
after (about 97% smaller), with a byte-identical decompression check. Compression
took about 13 ms. Add `--transport` to the profiling command to reproduce size
comparisons. Dashboard Overview's measured response shrank from 8,233 to 1,877
bytes; its main improvement comes from fewer reads and connection checkouts.

All 3,206 point objects and all four dashboard reports exactly matched their
previous versions. Additional comparisons covered site filters and a historical
report year different from the inspection year. Disposable PostgreSQL tests
cover explicit/ambiguous/boundary ownership, erosion, and historical statistics
after release for replanting. API tests compare decoded JSON bytes and check that
excluded endpoints retain their transport. Lint, build, and map filter/brush
tests pass; interactive canvas rendering still needs a browser check.
