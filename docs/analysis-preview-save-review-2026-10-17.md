# Analysis preview freshness and save performance — October 17, 2026

The reported DJI_0308.JPG / Analysis 39 screenshot shows an entirely green
image alongside 529 available points and 693 total coordinates. Read-only
verification of the existing database and private image assets found 693
retained points, 164 blocked by active erosion, and a stored preview that
already colors those 164 points orange. All 693 full-resolution point-center
colors agree with the current availability flags. No records were changed.

## Image freshness

Processing previously wrote images to deterministic paths derived from its
cache key. A later run could overwrite those objects while their signed URLs
and browser responses remained cached for up to an hour. The processing key
does not include every processing-code change. This permits the stale-image
symptom and also lets a later run affect assets referenced by earlier history.

- Each processing run now has a separate private storage prefix.
- Uploaded image filenames include a SHA-256 content digest.
- Saved-result image links, JPEG links and history thumbnails request
  `private, no-cache` through a signed storage response override. This produces
  different URLs from the old cacheable responses and requires revalidation.
- The response override was verified with an authenticated signed image read:
  HTTP 200 and `Cache-Control: private, no-cache`.

This fixes stale image reuse; it does not rerun detection or reconstruct missing
points in older analyses. Stored images remain snapshots of their analysis run;
editing a zone afterward dynamically changes map availability, not the pixels
of an existing snapshot. The reviewed Analysis 39 snapshot already agrees with
its current erosion flags.

## Save performance

The save path already uploads the images during processing. On Save it still
checks repeated images, visible map bounds and newly conflicting saved points.
It then previously issued one INSERT per planting point and ran two global
spacing cleanups whose inner loops compared every relevant pair of points.
On a remote database, hundreds of separate INSERT round trips can dominate
the wait; pairwise cleanup also becomes increasingly expensive as history grows.

- Accepted points now use bounded, parameterized multi-row INSERTs, 250 points
  per statement. The real in-memory deduplication and species-spacing checks
  still evaluate and index each candidate before writing.
- The cleanups use spatial buckets to find nearby candidates. Their exact
  distance thresholds, record priority, protected-record rules, deletion
  decisions and final count updates are unchanged.
- Existing transactions, permanent-hazard checks, erosion gates, authorization,
  repeat-image consent and saved-point checks remain in place.

An isolated 3,000-point fixture compared the bucket implementation with an
exhaustive scan using the same cleanup functions:

| Check | Exhaustive | Buckets | Decision agreement |
| --- | ---: | ---: | --- |
| Same-species cleanup | 2.619 s | 0.0369 s | Identical |
| Cross-species cleanup | 0.682 s | 0.0064 s | Identical |
| INSERT commands for 693 accepted points | 693 | 3 | 693 retained |

These local timings are not an end-to-end live save benchmark. Network,
database connection, read queries and asset-linking latency still apply.

## Verification and activation

Regression tests cover image revision paths, signed URL cache separation,
history/full JPEG revalidation, large batched saves with an internal duplicate,
the production SQL compatibility adapter, repeat-image and field-history
preservation, exhaustive cleanup equivalence across latitude bands, and the
existing erosion/forbidden/spacing/history/deletion tests.

**78 backend regression tests passed**, with zero failures, errors or skips.
The SQL adapter check uses an isolated SQLite database through the production
compatibility adapter; it does not insert test data into the shared database.

Local review artifacts: [preview and timings](../tmp/eroded_preview_save_20261017/index.html),
`summary.json`, `benchmark.json` and test XML reports in that same directory.
The review image was downloaded without modifying private storage or records.

The source changes require the backend to load the updated code. No service
restart, push, deployment, migration or live planting-data change was performed.
