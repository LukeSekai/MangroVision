# Organization directory

The staff sidebar's **Organizations** page (`/organizations`) brings together
existing organization records. It does not add a second organization registry.

- Search by organization name, contact person, email or phone. The finder shows
  four organizations per page with Previous/Next controls and a visible result
  range. Search covers the entire directory and returns to the first page.
  Changing pages keeps the selected profile open.
  Records reload when returning to this page or after workspace changes. There
  is no manual Refresh button; failed requests still offer Retry.
- The profile shows contact information from the latest saved non-cancelled
  activity (or the latest cancelled activity when there is no other activity).
  Its title is shown as the source. Existing structured website contact fields
  and the planter account's phone are fallbacks for missing schedule fields.
- **Expected participants** comes from that activity, rather than adding all
  historical bookings together. **Registered members** counts occupied
  participant profiles: a saved username, a registered legacy browser, or a
  previous planter session. Unused reserved places are not members. Merged-away
  owner accounts are excluded.
- Project sites include archived sites, labelled as archived. Only current sites
  offer a map link and count toward the current-site summary.
- Activity codes use the existing staff-only activity access endpoint. Codes
  are revealed on request and can be copied or hidden. This page cannot issue,
  replace or revoke codes, send email, or change activity access windows.
- Members show their saved username/participant number and recorded first/last
  session dates, with search and 25 rows per page. Missing legacy names stay
  explicitly unrecorded.

The new `/api/organizations` and `/api/organizations/{id}` GET endpoints require
an LGU/planner/admin staff session and send `Cache-Control: private, no-store`.
They select only directory fields: password hashes, browser/session keys,
encrypted codes and code digests are never returned. The directory response
does not contain plaintext access codes. No migration or record conversion is
needed. The existing database and code encryption key are preserved.

This feature requires publishing the frontend and restarting its matching API.
A Git branch or frontend deployment alone does not update the running Python API.
