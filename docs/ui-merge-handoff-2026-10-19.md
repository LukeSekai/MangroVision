# Header and logo branch: merge handoff

Branch: `ui/header-profile-clock-20261019`.
Base: `3978b50` on `Latest-Main`.

The user requested publishing this branch now and merging later, following the
groupmate's replanting and area-release logic. Her implementation is the desired
behavior for that workflow; this branch does not propose a competing fix.

## Changes to carry forward

- Move the staff profile and Philippine-time clock beside Help / Guide and
  notifications. The profile opens Account settings. Keep sign-out confirmation.
- Make the sidebar's MangroVision logo open Planting Map.
- Use the supplied `LogoSystem.zip` artwork throughout the public site, staff
  and planter screens, sign-in, onboarding, splash, analysis results, browser
  icons and report PDFs. Keep all seven originals in `client/public/brand/`.
- Retain the responsive header, accessible compact Help button, setup-checker
  asset inventory and documentation for these UI changes.

## Merge boundaries

The base already contains the earlier LGU queue implementation. The changes on
this branch do not alter its replanting rules, project-site release rules,
database records, migrations, assignments or survival calculations. Inheriting
that older implementation is not an instruction to retain it over the
groupmate's replacement.

Review shared files carefully rather than choosing an entire side for every
conflict:

- `client/src/pages/FieldApp.jsx`: carry the secondary logo tone into the
  groupmate's version while keeping her handlers and workflow.
- `client/src/components/Sidebar.jsx` and `AppShell.jsx`: preserve new routes,
  permissions and navigation from her code while applying this header layout.
- `client/src/guides/tours.js` and `docs/system-user-guides.md`: keep the profile
  and logo guidance, and make the replanting instructions match her final logic.
- `api/report_pdf.py`: this branch changes only the logo asset path; preserve
  any report or calculation changes from her code.
- `scripts/check_groupmate_setup.py` and `.gitignore`: retain both sides' needed
  assets and configuration entries.

Fetch the latest shared refs again before the actual merge. Resolve behavior
conflicts in favor of the agreed groupmate workflow, then verify the combined
UI and her end-to-end release/replanting cases before deploying.

## Verification completed before publication

35 existing UI, account, booking, navigation and guide tests passed. Six report
PDF tests passed. Lint for the changed JavaScript files and the production build
passed. Isolated desktop/mobile screenshots verified the new header and logo
placements; all seven supplied PNGs and the four compatibility URLs were checked
against their original artwork. This publication does not merge into main or
deploy to production.
