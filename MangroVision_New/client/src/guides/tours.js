// These tours describe real workflows. They only navigate and reveal UI;
// users perform uploads, submissions, downloads and other actions themselves.
import bookingPolicy from '../like/bookingPolicy.json' with { type: 'json' };

const plantingHoursGuide = bookingPolicy.enforcePlantingHours
  ? 'Tree planting starts between 6 AM and 3 PM and finishes by 5 PM.'
  : 'Planting requests currently allow flexible hours for testing; the full visit must finish on the same Philippine date.';

const activityAccessGuide = bookingPolicy.enforceActivityAccessWindow
  ? 'If the code is unavailable or outside its scheduled window, ask the LGU to check the activity.'
  : 'Codes can currently be used before and after the scheduled planting time. If a code is unavailable, ask the LGU to check the activity.';

const activityAccessTimingGuide = bookingPolicy.enforceActivityAccessWindow
  ? 'Access follows the approved activity’s time window.'
  : 'Codes can currently be used before and after the scheduled planting time.';

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
      step('/map', '.sidebar-nav', 'Choose a task', 'The sidebar opens Planting Map, Dashboard, Scheduling, Organizations, Monitoring and Activity Logs. Use the navigation inside Planting Map to switch between Overview, Zone Editor, Assign Points and Analyze Image. Each screen also has its own guide in Help / Guide.'),
      step('/map', '.map-layer', 'Explore the map', 'Drag to move around and use the zoom controls to inspect locations. Select a planting point to see its details. The map stays available beside planning screens.'),
      step('/map', '.panel-toggle', 'Make room for the map', 'Use Hide to collapse the side panel and Show to bring it back. Open a section heading to reveal its controls.'),
      step('/map', '.notification-button', 'Follow notifications', 'Open the notification bell to review updates and tasks that need attention. Check Activity Logs for the history of recorded actions.'),
      step('/map', '.sidebar-user', 'Manage your account', 'Select your name to change your staff password through email verification. Use Sign out at the bottom of the navigation when you finish.'),
    ],
  },
  {
    id: 'organizations', title: 'Organization information', path: '/organizations',
    description: 'Find organization contacts, project sites, planting codes and registered members.',
    steps: [
      step('/organizations', '[data-guide="organization-list"]', 'Find an organization', 'Search by organization name, contact person, email or phone, then select a profile. The finder shows four organizations per page. Previous and Next browse the results; searching checks the whole directory.'),
      step('/organizations', '[data-guide="organization-contact"]', 'Check contacts and participant counts', 'Review the contact person, email and phone. Expected participants comes from the latest saved activity shown below the details. Registered members counts people who have joined; it can differ from the expected attendance. Missing information is labelled rather than assumed.'),
      step('/organizations', '.organization-tabs', 'Review project sites', 'Project sites lists the organization’s planting areas. View on map opens a current site on Planting Map. Archived sites remain labelled for reference and are excluded from the current-site count.'),
      step('/organizations', '.organization-tabs', 'View an existing activity code', 'Activity codes lists scheduled activities and their progress. Show code reveals an existing code for that activity; Copy shares it and Hide code conceals it again. A missing or revoked code is labelled. Prepare or manage access in Assign Points. Share codes privately; access follows the approved activity’s start and finish times.'),
      step('/organizations', '.organization-tabs', 'Find registered members', 'Members shows recorded usernames, participant numbers and joining or last-seen dates when available. Search the roster or browse 25 members per page. A participant with no saved username is labelled Name not recorded.'),
      step('/organizations', '.organizations-header', 'Read the latest saved information', 'Information reloads when you return to this screen or the workspace receives a data-change update. If loading fails, use Retry. Contact details and expected participants identify the activity they came from.'),
    ],
  },
  {
    id: 'zones', title: 'Prepare sites and zones', path: '/zones',
    description: 'Review coverage, draw project sites, and manage eroded or warning areas.',
    steps: [
      step('/zones', '[data-guide-panel="Zone Editor"] .floating-panel-header', 'Prepare the planting area', 'Hover a boundary to see its name and description. Select a name in the zone lists to zoom to that area. Use Zone Editor to define organization project sites and manage eroded and warning areas. Zone editing is locked while image analysis is running.'),
      step('/zones', zoneCard('Draw Zone'), 'Choose the kind of zone', 'Choose Project Site for an organization planting area, Eroded for temporarily unsuitable ground, or Warning for a field hazard. Select the site owner. Its first site can be drawn on unclaimed land inside the drone map. Later sites must fit its existing area or released land available to other organizations. Avoid other organizations’ areas.'),
      step('/zones', '.zone-kind-row', 'Draw and review the boundary', 'Start on the left side of the area. Press and hold, drag right, then trace the rest of the outline and release. Zones must fit fully inside the drone map. Save becomes available after the boundary check. Redraw if the zone extends outside the allowed area.'),
      step('/zones', zoneCard('Draw Zone'), 'Name and save the zone', 'Once a boundary is ready, enter its name and any relevant notes. Warning zones also have a hazard type and severity. Save only after checking the outline; Cancel discards the current drawing.'),
      step('/zones', zoneCard('Project Sites'), 'Review organization sites', 'Select a site to focus its area. Its three-dot menu offers Release area or Delete site when eligible. Release frees eligible unplanted and dead locations while keeping living seedlings. Review the affected counts and boundary before confirming. Earlier planting, deaths and monitoring stay in history.'),
      step('/zones', zoneCard('Eroded Zones'), 'Handle eroded areas', 'Eroded zones retain potential planting points but make affected points unavailable for assignment and planting. The analysis preview marks these hexagons orange. Removing the zone makes eligible points available again. Check field conditions before removing it. Eroded zones can also be exported as GeoJSON.'),
      step('/zones', zoneCard('Warning Zones'), 'Review hazards', 'Warning zones describe hazards such as deep mud and their severity. They do not block assignment. Review the recorded hazard and actual conditions with the field team before working there.'),
    ],
  },
  {
    id: 'processing', title: 'Analyze an image', path: '/processing',
    description: 'Upload imagery, check its location, review results and save planting points.',
    steps: [
      step('/processing', imageCard('upload'), 'Choose a geotagged image', 'Open Upload Image and choose a drone JPEG or PNG with GPS EXIF information. Use an original image that belongs to the configured coverage area.'),
      step('/processing', '.upload-area', 'Check the preview and location', 'Confirm that you chose the intended image. The system checks its GPS and coverage before allowing analysis. Images without GPS or entirely outside coverage cannot be processed.'),
      step('/processing', imageCard('upload'), 'Review partial coverage or existing data', 'If the image partly overlaps map coverage, review and confirm that area. A repeated image or an image boundary overlapping a saved analysis can also require confirmation. Nearby areas without an overlap are separate. Check the listed earlier results and image location before continuing.'),
      step('/processing', '#canopy-buffer', 'Set the danger buffer', 'Choose the buffer distance around detected vegetation. A larger buffer leaves more space around detections and can reduce the area available for planting.'),
      step('/processing', '#species-select', 'Choose the planting species', 'Select the intended species and check the displayed planting-point distance. The system applies the spacing for that species when generating potential planting locations.'),
      step('/processing', '[data-guide-panel="Analyze Image"] .process-action', 'Run the analysis', 'After location checks pass, use Run Analysis and follow the progress messages. You can change screens or refresh: the workspace reconnects while the analysis job is still running on the server. Zone Editor and assignment changes are locked during processing.'),
      step('/processing', '[data-guide-panel="Analyze Image"] .process-action', 'Review the analysis overlay', 'Choose Review analysis when processing finishes. Compare the result with the original photo: purple shows detected canopy, red the danger buffer, green available planting hexagons and orange eroded or unavailable hexagons. Eroded points are retained so eligible ones can become available when that zone is removed. Check the capacity and alignment before saving.'),
      step('/processing', '[data-guide-panel="Analyze Image"] .process-action', 'Save the reviewed result', 'In the result view, save the analysis and planting points after checking the overlay and image alignment. Saved results can then be viewed on Planting Map and used for organization assignments.'),
      step('/processing', '.analysis-history-trigger', 'Reopen saved analyses', 'View Image Analysis History lets you filter and browse saved results. Display numbers follow the analyses currently saved; deleted entries do not add to the count. Open a result to compare its overlay, source photo and coordinates. View area in map focuses its boundary while keeping Analyze Image open; the result overlay stays in the review window.'),
      step('/processing', '.analysis-history-trigger', 'Protect analyses used for planting', 'Analyses with assigned points or field records are locked against deletion. Their planting and monitoring history must be preserved. Only unused eligible analyses offer Delete; check the affected result before confirming. A lock explains why an entry must stay.'),
    ],
  },
  {
    id: 'map', title: 'Planting map and exports', path: '/map',
    description: 'Read point statuses, control layers, check coverage and download coordinates.',
    steps: [
      step('/map', mapCard('overview'), 'Review saved planting capacity', 'Overview summarizes saved analyses and mapped points by status. Planned points are candidates awaiting assignment; assigned, planted, dead, skipped and unavailable points describe later stages.'),
      step('/map', mapCard('legend'), 'Read map symbols', 'Use Legend to identify point statuses, species and zone boundaries. Orange unavailable points are blocked by eroded areas; eligible points can return when the zone is removed. Warning zones flag hazards without blocking assignment. Select a point to review its saved details.'),
      step('/map', '[data-guide-layers]', 'Choose visible layers', 'Open the layer control in the upper-left corner to choose the basemap and show or hide the drone orthomosaic, planting points, assignment zones, project sites, forbidden zones, eroded zones and warning zones. These switches change the view.'),
      step('/map', mapCard('coverage'), 'Check area measurements', 'Coverage summarizes analyzed, plantable and danger areas. Compare these figures with the source imagery and the study criteria before using them for planning.'),
      step('/map', mapCard('export'), 'Export saved planting coordinates', 'Export Saved Points downloads CSV, GPX, KML or GeoJSON coordinates. KML uses small dots with labels hidden to keep dense planting areas readable in Google Earth; select a dot for its name and details. Check that the exported locations match the intended area.'),
    ],
  },
  {
    id: 'scheduling', title: 'Schedules and website requests', path: '/scheduling',
    description: 'Review tide guidance, approve or decline requests, create schedules and choose sites.',
    steps: [
      step('/scheduling', '.schedule-header', 'Coordinate organization activities', 'Scheduling brings together staff-entered activities and requests from the LIKE website. Review upcoming visits, clean-up drives and tree-planting activities here.'),
      step('/scheduling', '.website-preview-link', 'Understand the public booking form', `Open LIKE website opens the public site in another tab. Visitors enter group and contact details, then their preferred schedule and participants. The end time is two hours after the start. ${plantingHoursGuide} Check tide guidance for the full visit. Back is at the top left of the schedule card. Sending a request does not approve it. Scheduling times use Philippine time.`),
      step('/scheduling', '.website-requests .schedule-toolbar', 'Approve or decline requests', 'Website requests use Pending, Approved and Declined decisions. Contact the requester, then choose Approve requested schedule, Reschedule and approve, or Decline request. Approval requires agreement on the schedule; declining requires a reason. An approved activity remains Pending until completed. Request approval is separate from activity progress.'),
      step('/scheduling', '.schedule-tide-panel', 'Check the weekly tide forecast', 'The graph shows water levels and estimated planting conditions across the forecast. Daily High tide and Low tide times appear in the calendar below. Click a tide time to open its details and that day’s hourly graph. Check actual water and weather at the site.'),
      step('/scheduling', '.schedule-panel[aria-labelledby="organization-schedules-title"] .schedule-toolbar', 'Add an organization schedule', 'Choose Add schedule. Select or enter the organization, activity type, date, start and end times and participant information, then review the form before saving.'),
      step('/scheduling', '.schedule-kpis', 'Review activity progress', 'Activities use Completed, Pending and Cancelled progress groups. Pending includes activities awaiting approval or completion. Missed activities show a follow-up notice within Pending. Select a group to open its records in List. Completed activities are read-only and offer View.'),
      step('/scheduling', '.schedule-panel[aria-labelledby="organization-schedules-title"] .schedule-view-toggle', 'Use the calendar or list', 'Calendar shows approved activities and tide times by date. List lets you switch between All schedules and the three activity groups. A Missed label asks for progress follow-up; planting records and date locks still apply. Open an activity to review its details.'),
      step('/scheduling', '.schedule-panel[aria-labelledby="organization-schedules-title"] .schedule-toolbar', 'Approve and choose a planting area', 'After approving a tree-planting schedule, choose the organization planting area. Create missing areas in Zone Editor first. Review the scheduled activity before assigning available points.'),
      step('/scheduling', '.schedule-panel[aria-labelledby="organization-schedules-title"] .schedule-toolbar', 'Keep recorded activities protected', 'Completed activities offer View and cannot be edited, reopened, cancelled or deleted. Before completion, an activity can be rescheduled while none of its linked assignment points have been planted. Once planting is recorded, its date, times and planting plan are protected. A separate future activity at the same site remains independent.'),
    ],
  },
  {
    id: 'planters', title: 'Assign points and prepare field users', path: '/planters',
    description: 'Choose an organization, allocate points and share activity codes for field access.',
    steps: [
      step('/planters', '[data-guide-panel="Planting Assignments"] .floating-panel-header', 'Allocate planting work', 'Planting Assignments connects saved available points with an organization and project site. Prepare the organization through Scheduling and create its site in Zone Editor first.'),
      step('/planters', '#assign-organization', 'Choose the organization', 'Open Assign available points and select the organization. Its mapped planting area is located automatically. Points may be reserved before its shared field account is registered. If none are available, review the area, saved analysis and eroded zones in Zone Editor.'),
      step('/planters', '#assign-activity', 'Link the planting activity', 'Choose the relevant approved tree-planting activity. Its linked planting area is used automatically; an activity without an area uses the organization’s mapped area. A Pending activity’s date remains editable until one of its linked points is recorded as planted; completed activities are read-only.'),
      step('/planters', '#organization-point-count', 'Set the allocation size', 'Enter how many available points to assign. Review the count, species and participant allocation before using Assign. Participants receive staggered zigzag strips, with later batches balancing existing allocations.'),
      step('/planters', assignCard('Organization code'), 'Share the activity code', `Choose the organization and planting activity to prepare its shared code. Give the code privately to the contact person. First-time planters choose a unique username; returning planters use their existing username. ${activityAccessTimingGuide} Replacing or revoking its code ends that activity’s logins and preserves records.`),
    ],
  },
  {
    id: 'monitoring', title: 'Record visits and review plant health', path: '/monitoring',
    description: 'Find scheduled visits, browse planting sites, record health and plan replanting.',
    steps: [
      step('/monitoring', '.org-monitoring-kpis', 'Review planted seedlings', 'Monitoring summarizes organizations, planted seedlings and saved visits. Plant checks are anchored to actual planting dates at 14-day intervals; weekend rounds move to Monday. Upcoming dates appear in Sites to Monitor.'),
      step('/monitoring', '.org-monitoring-visit-tabs', 'Find scheduled monitoring visits', 'Use Due today or Overdue to find organizations ready for a visit. Each tab shows its count, and overdue visits show how many days late they are. Dates follow Philippine time.'),
      step('/monitoring', '[data-guide-card="organization-summary"]', 'Choose the organization to monitor', 'Under Record a monitoring visit, choose an organization in Due today or Overdue and open Record visit. The form loads its saved counts. Check the organization before entering observations; visit totals cover that organization’s planting history.'),
      step('/monitoring', '[data-guide-card="organization-summary"]', 'Record observations for the visit', 'In the visit form, check the date and enter newly dead seedlings since the last visit. Enter 0 when there are none. When deaths are recorded, select their cause and locate known dead seedlings on the map; unknown locations remain recorded as unlocated deaths.'),
      step('/monitoring', '[data-guide-card="organization-summary"]', 'Check health and save the visit', 'Review the calculated alive count, survival rate and automatic growth estimate. Choose overall health and describe the LGU actions taken. Save the visit after checking the figures. Growth estimates should be considered alongside field observations.'),
      step('/monitoring', '[data-guide-card="sites-to-monitor"]', 'Browse sites to monitor', 'Sites to Monitor displays five planting areas per page with boundary and point previews. Select an area for its details and larger map. The clock icon switches between details and monitoring history. Closing returns to the same list position. Counts and visit dates cover the site’s organization. A big-wave danger flag reflects recorded wave deaths; check actual field conditions.'),
      step('/monitoring/map', '[data-guide-card="monitoring-map-review"]', 'Locate deaths and review replanting', 'Show map opens mapped dead plants and replanting controls. Select eligible dead locations, review them, and approve their release for replanting when appropriate. Release returns locations to Planned while keeping earlier planting, deaths and monitoring in history.'),
      step('/monitoring', '.org-monitoring-page-actions', 'Download the monitoring report', 'Use Download Monitoring Report to choose the report scope and review the report before downloading. Show map opens the spatial view again.'),
    ],
  },
  {
    id: 'monitoring-map', title: 'Monitoring map', path: '/monitoring/map',
    description: 'Inspect dead plants, replanting candidates, map layers and statuses.',
    steps: [
      step('/monitoring/map', '#monitoring-map-organization', 'Choose the organization scope', 'Show organization filters the mapped points and site information. Choose All organizations for the full view. Changing organizations clears the current selection, so check the scope before selecting dead locations.'),
      step('/monitoring/map', '[data-guide-card="monitoring-map-review"]', 'Select dead locations', 'Show dead plants only hides other point statuses. Move / click selects one dead point; Brush select adds several and Brush erase removes them. Selected points turn purple. Press Escape to return to Move / click. Review up to 500 points in one batch.'),
      step('/monitoring/map', '[data-guide-card="monitoring-map-review"]', 'Review and release for replanting', 'Review for replanting shows the selected locations and eligibility checks before approval. Approved locations return to Planned for a new assignment; earlier organization planting and death records stay in history. Return to Assign Points to allocate the released locations. New deaths must be recorded in a monitoring visit first.'),
      step('/monitoring/map', '[data-guide-card="monitoring-map-overview"]', 'Read the monitoring overview', 'Overview summarizes visible mapped points and their status. Review organization monitoring records for the complete visit totals, including any deaths whose locations are unknown.'),
      step('/monitoring/map', '[data-guide-card="monitoring-map-legend"]', 'Read statuses and wave warnings', 'Legend explains dead, planted and selected points. Big-wave danger marks sites where at least 30% of historically planted locations have recorded big-wave deaths. This warning comes from recorded losses; use field observations and current conditions when planning work.'),
      step('/monitoring/map', '[data-guide-card="monitoring-map-layers"]', 'Control monitoring layers', 'Show or hide layers to compare points with imagery and sites. Shared forbidden, eroded and warning zones are hidden while viewing one organization; choose All organizations to show them. Return to Monitoring to record a visit or review its history.'),
    ],
  },
  {
    id: 'dashboard', title: 'Dashboard and restoration reports', path: '/dashboard',
    description: 'Filter progress, review restoration indicators, set goals and download reports.',
    steps: [
      step('/dashboard', '.dash-header', 'Review restoration progress', 'Dashboard summarizes saved planting and monitoring information. Review its indicators alongside the organization records and field observations.'),
      step('/dashboard', '.dash-filter-shell', 'Choose the reporting scope', 'Set From, To and Project site. The default period is this year so far. Activity and history figures follow their displayed period. Current planting-location status covers all dates for the selected site. Read each chart’s scope before comparing totals or downloading reports.'),
      step('/dashboard', '.dash-tabs', 'Explore the dashboard sections', 'Overview shows progress and current location statuses. Planting Work reviews recorded planting. Seedling Health shows survival by planting batch, monthly deaths, and replanting progress. Original deaths remain in their batches after replacement; current map counts appear separately. Project Sites compares sites.'),
      step('/dashboard', '.dash-kpi-action', 'Find sites needing follow-up', 'Select Sites needing follow-up to see the affected project sites and their flagged conditions. View site on map locates a site. Open Monitoring shows the organization responsible for that site.'),
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
    step(null, '.field-work-heading', 'Check your participant assignment', 'Confirm the organization, project site and participant number. Each participant has allocated points and a personal username. Keep that username for returning visits with the group’s approved activity code.'),
    step(null, '.field-progress-copy', 'Review your planting progress', 'Check how many points are planted, still to plant, skipped or unavailable. Use Refresh assignments to pick up changes from the LGU. If no points appear, ask the LGU to check your allocation.'),
    step(null, '.field-map-tools', 'Find your points and location', 'My points fits the map to your assignment. My location requests your device location; allow browser location access when you need it. Check the GPS accuracy displayed on the map.'),
    step(null, '.field-map-wrap', 'Read and select a point', 'Drag and zoom the map, then tap an assigned point to open its details. Planted points are yellow and skipped points are gray. Points in eroded areas are unavailable for planting.'),
    step(null, '.field-work-actions', 'Go to the next point', 'Next point selects the next eligible assignment. In the point details, choose Navigate to this point for directions. Road directions and within-site guidance serve different parts of the journey; read the route notes.'),
    step(null, '.field-map-tools', 'Use GPS with marked locations', 'Compare the destination, direction and distance with field markers and local conditions. Device GPS can be less precise than planting spacing. Use the marked planting location when the accuracy range is larger than your distance to the point.'),
    step(null, '.field-work-actions', 'Record a planted point', 'After physically planting at an eligible location, open its details and choose Mark as planted. Check the point number before confirming. If planting is not possible, use Skip Point and provide the requested reason.'),
    step(null, '.field-work-actions', 'Record several planted points', 'Choose points to mark lets you select multiple locations that you have already planted. Review the selected point numbers before confirming. Leave unfinished or unavailable locations unselected.'),
    step(null, '.field-list-toggle', 'Review the point list', 'Open the point list to browse assignments and filter by status. Select a point from the list to inspect it. If a point cannot be planted, use the available skip action and provide the requested reason.'),
    step(null, '.field-avatar-button', 'Review activity and finish', `The account menu opens Activity Logs and Sign out. Check that planting progress has updated before signing out. ${activityAccessTimingGuide} Access ends when its code is revoked or replaced; saved work remains. Ask the LGU to review the activity if access closes unexpectedly.`),
  ],
}];

export const STAFF_LOGIN_TOURS = [{
  id: 'staff-login', title: 'Staff sign-in and recovery',
  description: 'Sign in with your staff account and complete email verification.',
  steps: [
    step(null, '.login-form-header', 'Staff access', 'Use the staff account provisioned by your administrator. Field participants use their activity’s shared organization code and a personal username in the field workspace.'),
    step(null, '.login-form', 'Enter your credentials', 'Enter your staff username and password, then choose Sign in. Correct any highlighted input errors before submitting.'),
    step(null, '.login-form', 'Complete email verification', 'After your credentials are accepted, enter the verification code sent to your registered email. Use Resend when it becomes available if you need another code.'),
    step(null, '.auth-text-button', 'Recover a forgotten password', 'Choose Forgot password? to begin the recovery flow. Follow the email-code instructions, set and confirm a new password, then return to sign in.'),
  ],
}];

export const FIELD_LOGIN_TOURS = [{
  id: 'field-login', title: 'Planter sign in',
  description: 'Use the shared activity code and your personal username.',
  steps: [
    step(null, '.field-auth-brand', 'Organization field access', `Get the shared activity code from your group leader. Everyone in the organization uses that code. ${activityAccessTimingGuide}`),
    step(null, '.field-form', 'Choose first time or returning', 'First time asks for the organization code, then opens a welcome screen to choose a unique username with 3–32 letters, numbers, dots, underscores or hyphens. Returning planter uses the current activity code and that same username to restore assigned points and progress. No password or PIN is needed.'),
    step(null, '.field-form', 'Switch browsers when needed', `If your participant is signed in elsewhere, choose Switch here. The other login ends and saved progress stays the same. Keep your username for future activities. ${activityAccessGuide}`),
  ],
}];

export function toursForMode(mode) {
  if (mode === 'field') return FIELD_TOURS;
  if (mode === 'staff-login') return STAFF_LOGIN_TOURS;
  if (mode === 'field-login') return FIELD_LOGIN_TOURS;
  return STAFF_TOURS;
}

// Establish the organization through Scheduling before preparing its first site.
// Include every staff screen, including the directory and monitoring map.
export function fullWorkflow(tours) {
  const order = ['workspace', 'scheduling', 'organizations', 'zones', 'processing', 'map', 'planters', 'monitoring', 'monitoring-map', 'dashboard', 'activity', 'account'];
  return order.flatMap((id) => tours.find((tour) => tour.id === id)?.steps || []);
}
