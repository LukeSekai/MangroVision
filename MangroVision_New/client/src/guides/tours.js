// These tours describe real workflows. They only navigate and reveal UI;
// users perform uploads, submissions, downloads and other actions themselves.
const step = (path, target, title, content) => ({
  path, target, title, content,
  // Full maps fill the workspace and leave no outside edge for a tooltip.
  placement: ['.map-layer', '.field-map-wrap'].includes(target) ? 'center' : 'auto',
});
const card = (panel, key) => `[data-guide-panel="${panel}"] [data-guide-card="${key}"]`;
const mapCard = (key) => card('Planting Map', key);
const imageCard = (key) => card('Analyze Image', key);
const zoneCard = (key) => card('Zone Editor', key);
const assignCard = (key) => card('Planting Assignments', key);

export const STAFF_TOURS = [
  {
    id: 'workspace', title: 'Workspace basics',
    description: 'Find the main screens, notifications and map controls.',
    steps: [
      step('/map', '.workspace-header', 'Welcome to MangroVision', 'Use MangroVision to plan potential planting locations, coordinate organizations, and follow up on planted seedlings. Next and Back move through this guide. You can end it at any time.'),
      step('/map', '.sidebar-nav', 'Choose a task', 'The sidebar opens Planting Map, Dashboard, Scheduling, Monitoring and Activity Logs. Use the navigation inside Planting Map to switch between Overview, Zone Editor, Assign Points and Analyze Image. Each screen also has its own guide in Help / Guide.'),
      step('/map', '.map-layer', 'Explore the map', 'Drag to move around and use the zoom controls to inspect locations. Select a planting point to see its details. The map stays available beside planning screens.'),
      step('/map', '.panel-toggle', 'Make room for the map', 'Use Hide to collapse the side panel and Show to bring it back. Open a section heading to reveal its controls.'),
      step('/map', '.notification-button', 'Follow notifications', 'Open the notification bell to review updates and tasks that need attention. Check Activity Logs for the history of recorded actions.'),
      step('/map', '.sidebar-user', 'Manage your account', 'Select your name to change your staff password through email verification. Use Sign out at the bottom of the navigation when you finish.'),
    ],
  },
  {
    id: 'zones', title: 'Prepare sites and zones', path: '/zones',
    description: 'Review coverage, draw project sites, and manage eroded or warning areas.',
    steps: [
      step('/zones', '[data-guide-panel="Zone Editor"] .floating-panel-header', 'Prepare the planting area', 'Use Zone Editor to define organization project sites and manage eroded and warning areas. Zone editing is locked while image analysis is running.'),
      step('/zones', zoneCard('Draw Zone'), 'Choose the kind of zone', 'Open Draw Zone. Choose Project Site for an organization planting area, Eroded for unsuitable ground, or Warning to record a field hazard. Select the organization when creating a project site.'),
      step('/zones', '.zone-kind-row', 'Draw and review the boundary', 'After choosing the kind, start drawing. Press and hold on the map, drag around the area, then release to finish the loop. Review the boundary and covered-point count before saving.'),
      step('/zones', zoneCard('Draw Zone'), 'Name and save the zone', 'Once a boundary is ready, enter its name and any relevant notes. Warning zones also have a hazard type and severity. Save only after checking the outline; Cancel discards the current drawing.'),
      step('/zones', zoneCard('Project Sites'), 'Review organization sites', 'Project Sites lists the saved organization areas. Hover a site boundary, or tap it on a phone, to review its owner, area, notes and planting counts before choosing it in Scheduling or Planting Assignments.'),
      step('/zones', zoneCard('Eroded Zones'), 'Handle eroded areas', 'Points inside eroded zones remain visible but become unavailable for planting. Removing an eroded zone makes eligible points available again. Review the zone carefully before deleting it. You can also export eroded zones as GeoJSON.'),
      step('/zones', zoneCard('Warning Zones'), 'Review hazards', 'Warning zones describe conditions such as deep mud and their severity. Review these with the field team before working in the area.'),
      step('/zones', zoneCard('Forbidden Zones'), 'Respect exclusions', 'Forbidden Zones displays the configured planting exclusions. Compare these boundaries with the study area when reviewing analysis and generated planting locations.'),
    ],
  },
  {
    id: 'processing', title: 'Analyze an image', path: '/processing',
    description: 'Upload imagery, check its location, review results and save planting points.',
    steps: [
      step('/processing', imageCard('upload'), 'Choose a geotagged image', 'Open Upload Image and choose a drone JPEG or PNG with GPS EXIF information. Use an original image that belongs to the configured coverage area.'),
      step('/processing', '.upload-area', 'Check the preview and location', 'Confirm that you chose the intended image. The system checks its GPS and coverage before allowing analysis. Images without GPS or entirely outside coverage cannot be processed.'),
      step('/processing', imageCard('upload'), 'Review partial coverage or existing data', 'If the image only partly overlaps coverage, review and explicitly confirm that area. If existing planting data overlap, read the confirmation before continuing. Use the image-location map when you need to inspect its position.'),
      step('/processing', '#canopy-buffer', 'Set the danger buffer', 'Choose the buffer distance around detected vegetation. A larger buffer leaves more space around detections and can reduce the area available for planting.'),
      step('/processing', '#species-select', 'Choose the planting species', 'Select the intended species and check the displayed planting-point distance. The system applies the spacing for that species when generating potential planting locations.'),
      step('/processing', '[data-guide-panel="Analyze Image"] .process-action', 'Run the analysis', 'After location checks pass, use Run Analysis. Follow its progress and wait for completion. Processing continues if you change screens; Zone Editor and assignment changes are locked while it runs.'),
      step('/processing', '[data-guide-panel="Analyze Image"] .process-action', 'Review the analysis overlay', 'When processing finishes, choose Review analysis. Compare the detected vegetation, danger areas, plantable space and generated points with the source image. Check the reported capacity before accepting the result.'),
      step('/processing', '[data-guide-panel="Analyze Image"] .process-action', 'Save the reviewed result', 'In the result view, save the analysis and planting points after checking the overlay and image alignment. Saved results can then be viewed on Planting Map and used for organization assignments.'),
      step('/processing', '.analysis-history-trigger', 'Reopen saved analyses', 'View Image Analysis History lists saved results. Open an entry to inspect its image and planting data. Check carefully before confirming deletion of any saved analysis.'),
    ],
  },
  {
    id: 'map', title: 'Planting map and exports', path: '/map',
    description: 'Read point statuses, control layers, check coverage and download coordinates.',
    steps: [
      step('/map', mapCard('overview'), 'Review saved planting capacity', 'Overview summarizes saved analyses and mapped points by status. Planned points are candidates awaiting assignment; assigned, planted, dead, skipped and unavailable points describe later stages.'),
      step('/map', mapCard('legend'), 'Read map symbols', 'Use Legend to identify point statuses, species and zone boundaries. Orange unavailable points are inside eroded areas. Inspect a point on the map to review its details.'),
      step('/map', '[data-guide-layers]', 'Choose visible layers', 'Open the layer control in the upper-left corner to choose the basemap and show or hide the drone orthomosaic, planting points, assignment zones, project sites, forbidden zones, eroded zones and warning zones. These switches change the view.'),
      step('/map', mapCard('coverage'), 'Check area measurements', 'Coverage summarizes analyzed, plantable and danger areas. Compare these figures with the source imagery and the study criteria before using them for planning.'),
      step('/map', mapCard('export'), 'Export saved planting coordinates', 'Open Export Saved Points and choose a supported format. These downloads contain saved planting locations. Open the file in your GIS or navigation tool and verify its coordinates and reference system.'),
    ],
  },
  {
    id: 'scheduling', title: 'Schedules and website requests', path: '/scheduling',
    description: 'Review tide guidance, create schedules, confirm appointments and choose sites.',
    steps: [
      step('/scheduling', '.schedule-header', 'Coordinate organization activities', 'Scheduling brings together staff-entered activities and requests from the LIKE website. Review upcoming visits, clean-up drives and tree-planting activities here.'),
      step('/scheduling', '.schedule-tide-panel', 'Review when to plant', 'Check the tide forecast and activity guidance when selecting a date and time. Open a tide entry for details. Use the displayed guidance together with local field conditions.'),
      step('/scheduling', '.schedule-toolbar', 'Add an organization schedule', 'Choose Add new schedule. Select or enter the organization, activity type, date, time and participant information, then review the form before saving.'),
      step('/scheduling', '.schedule-view-toggle', 'Use the calendar or list', 'Calendar shows activities and tide times by date. List shows organization schedules with their status and available actions. Open an activity to review its details.'),
      step('/scheduling', '.website-requests .schedule-toolbar', 'Review pending website appointments', 'Pending LIKE website requests appear in the calendar and website-request section. Open a request, review its organization and activity details, and confirm or decline it through the provided review controls.'),
      step('/scheduling', '.schedule-panel[aria-labelledby="organization-schedules-title"] .schedule-toolbar', 'Confirm and choose a planting area', 'After confirming a tree-planting schedule, choose the organization planting area. Create missing areas in Zone Editor first. Review the scheduled activity before assigning available points.'),
    ],
  },
  {
    id: 'planters', title: 'Assign points and prepare field users', path: '/planters',
    description: 'Choose an organization, allocate points, share field access and manage devices.',
    steps: [
      step('/planters', '[data-guide-panel="Planting Assignments"] .floating-panel-header', 'Allocate planting work', 'Planting Assignments connects saved available points with an organization and project site. Prepare the organization through Scheduling and create its site in Zone Editor first.'),
      step('/planters', '#assign-organization', 'Choose the organization', 'Open Assign available points and select the organization. Points may be reserved before its shared field account is registered.'),
      step('/planters', '#assign-site', 'Choose its project site', 'Choose a site owned by that organization. The map locates it and the form checks available points. If none are available, review the site, saved analysis and eroded zones.'),
      step('/planters', '#assign-activity', 'Link the planting activity', 'Choose the relevant confirmed activity when one is available. Check that the activity and site match the planned field visit.'),
      step('/planters', '#organization-point-count', 'Set the allocation size', 'Enter how many available points to assign. Review the count, species and participant allocation before using Assign. Participants receive staggered zigzag strips, with later batches balancing existing allocations.'),
      step('/planters', assignCard('Field Share Link'), 'Share the field workspace', 'Open Field Share Link and copy the field link for participants. On a local setup, use the provided link-generation controls if needed. Keep the current link while participants work.'),
      step('/planters', assignCard('Participant devices'), 'Register and recover participant devices', 'After selecting a registered organization, Participant devices shows its device allocations. Each participant should keep their assigned participant number and device recovery code. Use these controls to manage a lost or replaced device.'),
      step('/planters', '.assignment-activity-report', 'Review organization progress', 'View Activity Report opens the recorded planting activity report. Use the Overview tab for mapped planting totals. Refresh your data after field users record planting progress.'),
    ],
  },
  {
    id: 'monitoring', title: 'Record visits and review plant health', path: '/monitoring',
    description: 'Record deaths and health, locate dead plants, review history and plan replanting.',
    steps: [
      step('/monitoring', '.org-monitoring-kpis', 'Review planted seedlings', 'Monitoring summarizes participating organizations, planted seedlings and saved visits. Monitoring begins two weeks after planting; organizations that are not due show a locked state.'),
      step('/monitoring', '.monitoring-due-filter', 'Find organizations due for a visit', 'Select Show organizations due for a visit to focus on work that can be recorded now. Clear the filter to review upcoming organizations.'),
      step('/monitoring', '[data-guide-card="organization-summary"]', 'Choose the organization to monitor', 'Open Record a monitoring visit and choose an available organization. The visit form loads the previously saved counts and record for that organization.'),
      step('/monitoring', '[data-guide-card="organization-summary"]', 'Record observations for the visit', 'In the visit form, check the date and enter newly dead seedlings since the last visit. Enter 0 when there are none. When deaths are recorded, select their cause and locate known dead seedlings on the map; unknown locations remain recorded as unlocated deaths.'),
      step('/monitoring', '[data-guide-card="organization-summary"]', 'Check health and save the visit', 'Review the calculated alive count, survival rate and automatic growth estimate. Choose overall health and describe the LGU actions taken. Save the visit after checking the figures. Growth estimates should be considered alongside field observations.'),
      step('/monitoring', '[data-guide-card="organization-history"]', 'Review monitoring history', 'Open Monitoring History and select an organization to review its saved visits and seedling records. Use the available history and growth views to follow changes over time.'),
      step('/monitoring/map', '[data-guide-card="monitoring-map-review"]', 'Locate deaths and review replanting', 'Monitoring map shows mapped dead plants and replanting controls. Review reported deaths and replacement candidates, then complete the appropriate approval or assignment flow after checking the site.'),
      step('/monitoring', '.org-monitoring-page-actions', 'Download the monitoring report', 'Use Download Monitoring Report to choose the report scope and review the report before downloading. Show map opens the spatial view again.'),
    ],
  },
  {
    id: 'monitoring-map', title: 'Monitoring map', path: '/monitoring/map',
    description: 'Inspect dead plants, replanting candidates, map layers and statuses.',
    steps: [
      step('/monitoring/map', '[data-guide-card="monitoring-map-review"]', 'Review dead plants and replacements', 'Dead plants & replanting brings together mapped deaths and replanting options. Select the relevant organization or record and check the location before continuing with replacement planting.'),
      step('/monitoring/map', '[data-guide-card="monitoring-map-overview"]', 'Read the monitoring overview', 'Overview summarizes visible mapped points and their status. Review organization monitoring records for the complete visit totals, including any deaths whose locations are unknown.'),
      step('/monitoring/map', '[data-guide-card="monitoring-map-legend"]', 'Identify point statuses', 'Use Legend to distinguish dead, planted and other mapped locations. Select a point on the map to inspect its details.'),
      step('/monitoring/map', '[data-guide-card="monitoring-map-layers"]', 'Control monitoring layers', 'Show or hide layers to compare points with imagery and zones. Return to Monitoring to record a visit or review its history.'),
    ],
  },
  {
    id: 'dashboard', title: 'Dashboard and restoration reports', path: '/dashboard',
    description: 'Filter progress, review restoration indicators, set goals and download reports.',
    steps: [
      step('/dashboard', '.dash-header', 'Review restoration progress', 'Dashboard summarizes saved planting and monitoring information. Review its indicators alongside the organization records and field observations.'),
      step('/dashboard', '.dash-filter-shell', 'Choose the reporting scope', 'Set the start date, end date and project site. The default view covers the current year so far. Check the scope before comparing indicators or downloading a report.'),
      step('/dashboard', '.dash-tabs', 'Explore the dashboard sections', 'Use the tabs to view planting progress, survival, health and other restoration indicators. Each view uses the selected reporting scope.'),
      step('/dashboard', '.dash-goals-button', 'Set planting goals', 'Open Planting goals to select the year and review or enter the annual planting and survival targets. Save goals after checking the values.'),
      step('/dashboard', '.dash-report-button', 'Create a restoration report', 'Use Download Report to choose a report type and review its scope. Download the report after checking the included dates, sites and records.'),
    ],
  },
  {
    id: 'activity', title: 'Activity history', path: '/activity',
    description: 'Review recorded actions and find changes made by staff or field participants.',
    steps: [
      step('/activity', '.activity-log-header', 'Review workspace history', 'Activity Logs shows recorded staff and organization actions. Use it to review changes and follow up on unexpected activity.'),
      step('/activity', '.activity-feed-toolbar', 'Find relevant activity', 'Use Refresh activity to load the latest events and Load more activity to browse older entries when available. Review the time, actor and description, then open the related workspace screen to inspect the current record.'),
    ],
  },
  {
    id: 'account', title: 'Account and password', path: '/account',
    description: 'Change your staff password and complete email verification.',
    steps: [
      step('/account', '.account-card', 'Review your staff account', 'Account settings shows the registered verification email and the password-change form. Use the account provisioned by your administrator.'),
      step('/account', '#account-current-password', 'Enter your current password', 'Enter your existing password, then choose a new password with at least 12 characters and repeat it in the confirmation field.'),
      step('/account', '.account-card', 'Verify the change', 'Use Send verification code, then enter the code sent to the registered email. Completing the change signs out all sessions on this account; sign in again using your new password.'),
    ],
  },
];

export const FIELD_TOURS = [{
  id: 'field', title: 'Field planting workflow', path: '/field',
  description: 'Find your assigned points, navigate, record planting and protect your participant access.',
  steps: [
    step(null, '.field-work-heading', 'Check your participant assignment', 'Confirm the organization, project site and participant number. Each participant receives their own allocated points under the shared organization account.'),
    step(null, '.field-progress-copy', 'Review your planting progress', 'Check how many points are planted, still to plant, skipped or unavailable. Use Refresh assignments to pick up changes from the LGU. If no points appear, ask the LGU to check your allocation.'),
    step(null, '.field-map-tools', 'Find your points and location', 'My points fits the map to your assignment. My location requests your device location; allow browser location access when you need it. Check the GPS accuracy displayed on the map.'),
    step(null, '.field-map-wrap', 'Read and select a point', 'Drag and zoom the map, then tap an assigned point to open its details. Planted points are yellow and skipped points are gray. Points in eroded areas are unavailable for planting.'),
    step(null, '.field-work-actions', 'Go to the next point', 'Next point selects the next eligible assignment. In the point details, choose Navigate to this point for directions. Road directions and within-site guidance serve different parts of the journey; read the route notes.'),
    step(null, '.field-map-tools', 'Use GPS with marked locations', 'Compare the destination, direction and distance with field markers and local conditions. Device GPS can be less precise than planting spacing. Use the marked planting location when the accuracy range is larger than your distance to the point.'),
    step(null, '.field-work-actions', 'Record a planted point', 'After physically planting at an eligible location, open its details and use the planted action. Check the point number and confirm the action when prompted.'),
    step(null, '.field-work-actions', 'Record several planted points', 'Choose points to mark lets you select multiple locations that you have already planted. Review the selected point numbers before confirming. Leave unfinished or unavailable locations unselected.'),
    step(null, '.field-list-toggle', 'Review the point list', 'Open the point list to browse assignments and filter by status. Select a point from the list to inspect it. If a point cannot be planted, use the available skip action and provide the requested reason.'),
    step(null, '.field-avatar-button', 'Keep your device recovery code', 'Open the account menu and choose Device recovery code. Keep it privately with your participant number so the same allocation can be restored if your device or field link changes.'),
    step(null, '.field-avatar-button', 'Review activity and finish', 'The account menu also opens Activity Logs and Sign out. Check that your planted-point progress is updated before signing out.'),
  ],
}];

export const STAFF_LOGIN_TOURS = [{
  id: 'staff-login', title: 'Staff sign-in and recovery',
  description: 'Sign in with your staff account and complete email verification.',
  steps: [
    step(null, '.login-form-header', 'Staff access', 'Use the staff account provisioned by your administrator. Field participants use the shared organization account in the field workspace.'),
    step(null, '.login-form', 'Enter your credentials', 'Enter your staff username and password, then choose Sign in. Correct any highlighted input errors before submitting.'),
    step(null, '.login-form', 'Complete email verification', 'After your credentials are accepted, enter the verification code sent to your registered email. Use Resend when it becomes available if you need another code.'),
    step(null, '.auth-text-button', 'Recover a forgotten password', 'Choose Forgot password? to begin the recovery flow. Follow the email-code instructions, set and confirm a new password, then return to sign in.'),
  ],
}];

export const FIELD_LOGIN_TOURS = [{
  id: 'field-login', title: 'Organization access and registration',
  description: 'Register your team, sign in as a participant or recover a device.',
  steps: [
    step(null, '.field-auth-brand', 'Organization field access', 'Field users share an organization account, while each device uses a separate participant number. Ask the LGU to add your organization through Scheduling if it is missing.'),
    step(null, '.field-tab-row', 'Register the organization once', 'Use Register when your organization does not yet have a field account. Choose the organization, shared username and password, and the number of participants. Coordinate this registration with your team.'),
    step(null, '.field-form', 'Sign in on a participant device', 'Use Sign In with the shared organization credentials. Follow the participant-number instructions to claim or restore the appropriate device slot. Keep the same participant number for your assigned work.'),
    step(null, '.field-form', 'Recover access to an existing allocation', 'If replacing a device or returning through a different field link, use the recovery options and your saved device recovery code. Ask the LGU for help if you cannot recover the participant slot.'),
  ],
}];

export function toursForMode(mode) {
  if (mode === 'field') return FIELD_TOURS;
  if (mode === 'staff-login') return STAFF_LOGIN_TOURS;
  if (mode === 'field-login') return FIELD_LOGIN_TOURS;
  return STAFF_TOURS;
}

// The normal staff journey: prepare -> analyze -> review -> schedule -> assign
// -> monitor -> report. Secondary screens stay available as focused guides.
export function fullWorkflow(tours) {
  const order = ['workspace', 'zones', 'processing', 'map', 'scheduling', 'planters', 'monitoring', 'dashboard', 'activity', 'account'];
  return order.flatMap((id) => tours.find((tour) => tour.id === id)?.steps || []);
}
