# MangroVision user guides

Select **Help / Guide** in the workspace header to open the guide menu.
The menu recommends a guide for the current screen and lets staff choose a
task guide or **Full system workflow**. Use **Next**, **Back**, **Finish**,
**End guide**, or **Escape**. Guides can be replayed from the same button.

The staff guides cover:

- Workspace navigation, map controls, notifications and account access.
- Project sites, eroded areas, warnings and forbidden-zone review.
- Geotagged image upload, coverage checks, configuration, analysis, result
  review, saving and analysis history.
- Planting-point statuses, map layers, coverage measurements and exports.
- Tide guidance, organization schedules and LIKE website appointments.
- Organization/site/activity selection, point allocation, field links and
  participant-device management.
- Monitoring visits, newly recorded deaths, seedling locations, health,
  monitoring history, replanting and monitoring reports.
- Dashboard scope, restoration indicators, planting goals and reports.
- Activity history and staff password changes.

Field users have a separate guide for participant assignments, map/GPS controls,
navigation, individual and multiple-point planting confirmation, the point list,
skipping, device recovery and activity history. Staff and field sign-in guides
are currently dormant: their code is retained, but their buttons are hidden.

Guides navigate between staff screens and reveal collapsed panels. They do not
upload, analyze, save, assign, approve, download, delete or submit records.
Highlighted controls are blocked during a tour; end the guide to perform the
task. Instructions for conditional controls stay available with a prerequisite
note when those controls are absent. For example, analysis results require a
processed image and participant-device controls require a selected registered
organization.

## Maintaining the guides

The frontend uses React Joyride **3.2.0**, which supports the project's React 19
version. Joyride loads when a user starts a tour.

- `MangroVision_New/client/src/guides/tours.js` defines the task descriptions,
  routes, targets and steps.
- `src/guides/tourRuntime.js` waits for route rendering, finds visible targets,
  opens panel UI and cancels pending work when a tour ends.
- `src/components/GuideButton.jsx` provides the menu and current-screen guide.
- `src/components/GuideTour.jsx` provides the Joyride tooltips and controls.
- `src/components/Panel.jsx` exposes `data-guide-panel` and `data-guide-card`
  attributes so tour targets survive changes to visual styling.

To restore the sign-in guide buttons, set `SIGN_IN_GUIDES_ENABLED` to `true` in
`MangroVision_New/client/src/guides/config.js`. This enables both staff and field
sign-in guides; the workspace guides are always available.

When changing a workflow, update its instructions and targets together. Keep
automatic tour actions limited to navigation and revealing UI. Keep full-map
steps centered so tooltips fit the viewport.

From `MangroVision_New/client`, run:

```powershell
npm run lint
npm run build
node --test src/guides/tourRuntime.test.js src/components/Panel.test.js src/components/LoginScreen.test.js src/components/RetainedRoutes.test.js
```

Also check the menu, Back/Next across routes, collapsed panels, absent data,
Finish, Escape, replay and phone layouts in a browser. Use isolated sample data
when testing; walking through a guide must not create API mutations.
