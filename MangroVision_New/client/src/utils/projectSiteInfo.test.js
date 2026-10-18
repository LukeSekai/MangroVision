import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { bindProjectSiteInfo, createProjectSiteInfo, emptyProjectSiteCounts, projectSiteAreaM2, projectSitePointCounts, projectSiteStyle } from './projectSiteInfo.js';

test('site counts reconcile each current location once using resolved site ownership', () => {
  const counts = projectSitePointCounts([
    { id: 1, source_project_site_id: 7, map_status: 'planned', project_site_id: 9 },
    { id: '1', source_project_site_id: '7', map_status: 'planned' },
    { id: 2, source_project_site_id: '7', assignment_status: 'pending' },
    { id: 3, source_site_id: 7, assignment_status: 'completed' },
    { id: 4, source_project_site_id: 7, death_at: '2026-10-10', planting_status: 'planted', assignment_id: 44 },
    { id: 5, source_project_site_id: 7, assignment_status: 'skipped' },
    { id: 6, source_project_site_id: 7, eroded_unavailable: true },
    { id: 8, source_project_site_id: 7, deleted_at: '2026-10-10' },
    { id: 9, source_project_site_id: 7, is_deleted: true },
    { id: 10, source_project_site_id: 9, map_status: 'planted' },
    { id: 11, source_project_site_id: null, organization_id: 7 },
  ]);
  assert.deepEqual(counts.get('7'), {
    mapped: 6, planned: 1, assigned: 1, planted: 1, dead: 1, skipped: 1, unavailable: 1, assignedTotal: 4,
  });
  assert.equal(counts.get('9').mapped, 1);
  assert.equal(counts.has('null'), false);
});

test('participant counts use assignment sites and distinguish unavailable and dead locations', () => {
  const counts = projectSitePointCounts([
    { planting_point_id: 1, project_site_id: 4, source_project_site_id: 8, assignment_status: 'pending' },
    { planting_point_id: 2, project_site_id: '4', assignment_status: 'completed' },
    { planting_point_id: 3, project_site_id: 4, assignment_status: 'pending', inside_eroded_zone: true },
    { planting_point_id: 4, project_site_id: 4, assignment_status: 'completed', death_at: '2026-10-10' },
    { planting_point_id: 5, project_site_id: 9, assignment_status: 'skipped' },
  ], { scope: 'participant' });
  assert.deepEqual(counts.get('4'), {
    mapped: 4, planned: 0, assigned: 1, planted: 1, dead: 1, skipped: 0, unavailable: 1, assignedTotal: 4,
  });
  assert.equal(counts.has('8'), false);
  assert.equal(counts.get('9').skipped, 1);
});

const rectangle = (west, south, east, north) => [[west, south], [east, south], [east, north], [west, north], [west, south]];

test('approximate area handles winding, polygon holes, disconnected sites and invalid geometry', () => {
  const outside = rectangle(122.625, 10.78, 122.626, 10.781);
  const hole = rectangle(122.6252, 10.7802, 122.6258, 10.7808);
  const polygon = { type: 'Polygon', coordinates: [outside] };
  const area = projectSiteAreaM2(polygon);
  assert(area > 12000 && area < 12200);
  assert.equal(projectSiteAreaM2({ ...polygon, coordinates: [[...outside].reverse()] }), area);
  const hollow = projectSiteAreaM2({ ...polygon, coordinates: [outside, hole] });
  assert(Math.abs(hollow / area - 0.64) < 0.001);
  assert.equal(projectSiteAreaM2({ type: 'MultiPolygon', coordinates: [[outside], [outside]] }), 2 * area);
  assert.equal(projectSiteAreaM2(null), null);
  assert.equal(projectSiteAreaM2({ type: 'Point', coordinates: [122.625, 10.78] }), null);
  assert.equal(projectSiteAreaM2({ type: 'Polygon', coordinates: [[[NaN, 10.78]]] }), null);
});

function card(feature, options) {
  const dom = new JSDOM();
  const result = createProjectSiteInfo(feature, { ...options, document: dom.window.document });
  const text = result.textContent;
  const html = result.outerHTML;
  dom.window.close();
  return { text, html };
}

test('reference maps show supplied information without inventing status or survival counts', () => {
  const { text } = card({ properties: {
    name: 'North site', organization_name: 'Coastal team', point_count: 250,
    analysis_count: 2, assignment_count: 3, schedule_count: 1, notes: 'Use the north entrance.',
  } });
  assert.match(text, /North siteCoastal team/);
  assert.match(text, /Saved analyses2Assignments3Schedules1Linked planting points250/);
  assert.match(text, /Use the north entrance/);
  assert.doesNotMatch(text, /Mapped points|Available|Recorded dead|survival/i);
});

test('empty sites show zero counts; participant cards clearly identify the counting scope', () => {
  const counts = emptyProjectSiteCounts();
  const { text: site } = card({ properties: {} }, { counts });
  assert.match(site, /No organization assigned/);
  assert.match(site, /0Mapped pointsAvailable0/);
  const { text: field } = card({ properties: {} }, { counts, scope: 'participant' });
  assert.match(field, /0Your assigned pointsPending0/);
  assert.match(field, /Counts cover your assigned work in this site/);
  assert.doesNotMatch(field, /Available|Current mapped locations/);
});

test('assigned totals retain completed and dead allocations without counting direct LGU plantings or deleted points', () => {
  const points = [
    { id: 1, source_project_site_id: 7, planting_status: 'planted', assignment_id: 10 },
    { id: 2, source_project_site_id: 7, planting_status: 'planted', death_at: '2026-10-18', assignment_id: 10 },
    { id: 3, source_project_site_id: 7, assignment_status: 'pending', assignment_id: 11 },
    { id: 4, source_project_site_id: 7, planting_status: 'planted' },
    { id: 5, source_project_site_id: 7, assignment_id: 10, deleted_at: '2026-10-18' },
    { id: '2', source_project_site_id: '7', death_at: '2026-10-18', assignment_id: 10 },
  ];
  const counts = projectSitePointCounts(points).get('7');
  assert.equal(counts.assignedTotal, 3);
  assert.equal(counts.mapped, 4);
  assert.equal(counts.assigned, 1);
  const { text } = card({ properties: {} }, { counts });
  assert.match(text, /Assigned points3/);
  assert.match(text, /Awaiting planting1/);
  assert.match(text, /Survival rate66\.67%/);
});

test('unavailable site locations remain separate from assigned seedlings and the survival denominator', () => {
  const counts = { mapped: 256, planned: 0, assigned: 0, planted: 126, dead: 123,
    skipped: 0, unavailable: 7, assignedTotal: 249 };
  const { text } = card({ properties: { name: 'Jaro Mangrove Protectors' } }, { counts });
  assert.match(text, /Assigned points249/);
  assert.match(text, /Survival rate50\.6%/);
  assert.match(text, /249 seedlings/);
  assert.match(text, /256Mapped points/);
  assert.match(text, /Unavailable7/);
});

test('user-entered names, organizations and notes remain plain text in map overlays', () => {
  const payload = '<img src=x onerror=alert(1)>';
  const { text, html } = card({ properties: { name: payload, organization_name: `Owner ${payload}`, notes: payload } });
  assert.equal(text.split(payload).length - 1, 3);
  assert.doesNotMatch(html, /<img|<script/);
  assert.match(html, /&lt;img/);
});

test('site survival uses current planted and dead seedlings rather than all mapped locations', () => {
  const { text } = card({ properties: {} }, { counts: {
    mapped: 300, planned: 50, assigned: 20, planted: 89, dead: 129, skipped: 2, unavailable: 10,
  } });
  assert.match(text, /Survival rate40\.83%/);
  assert.match(text, /89 planted.*129 recorded dead/);
  assert.doesNotMatch(text, /29\.67%/);
});

test('survival is unavailable before planting and zero when every planted seedling has died', () => {
  assert.match(card({ properties: {} }, { counts: emptyProjectSiteCounts() }).text, /Survival rateNot available yet/);
  const counts = { ...emptyProjectSiteCounts(), mapped: 10, dead: 10 };
  assert.match(card({ properties: {} }, { counts }).text, /Survival rate0%/);
  assert.match(card({ properties: {} }, { counts: { ...counts, planted: 8, dead: 2 }, scope: 'participant' }).text, /Survival rate80%/);
});

test('big-wave site warnings show their historical site denominator even on participant maps', () => {
  const feature = { properties: { name: 'North site', big_wave_risk: {
    active: true, planted_points: 100, wave_death_points: 30, percentage: 30,
  } } };
  const { text } = card(feature, { counts: emptyProjectSiteCounts(), scope: 'participant' });
  assert.match(text, /Danger: This area is prone to big waves/);
  assert.match(text, /30 of 100 planted locations \(30%\)/);
  assert.match(text, /whole project site’s recorded history/);
  assert.match(text, /Other causes of death do not count/);
  assert.equal(projectSiteStyle(feature, { weight: 2 }).color, '#b91c1c');
  for (const risk of [undefined, { active: false, wave_death_points: 10, planted_points: 100, percentage: 10 }]) {
    const normal = { properties: { big_wave_risk: risk } };
    assert.doesNotMatch(card(normal).text, /Danger|prone to big waves/);
    assert.deepEqual(projectSiteStyle(normal, { color: '#15803d' }), { color: '#15803d' });
  }
});

test('flagged site boundaries carry a permanent danger label and a detailed popup', () => {
  const calls = {};
  const events = {};
  const dom = new JSDOM();
  const label = { setLatLng: () => label, setContent: (text) => { calls.label = text.textContent; return label; },
    addTo: () => { calls.visible = true; return label; }, remove: () => { calls.visible = false; } };
  const layer = { setStyle: (style) => { calls.style = style; }, bindTooltip: (content, options) => { calls.tooltip = options; },
    bindPopup: (content) => { calls.popup = content; }, on: (event, handler) => { events[event] = handler; },
    getBounds: () => ({ isValid: () => true, getCenter: () => [10.78, 122.625] }) };
  bindProjectSiteInfo(layer, { properties: { big_wave_risk: { active: true } } }, {
    document: dom.window.document, map: { hasLayer: () => true },
    createTooltip: (options) => { calls.warningOptions = options; return label; },
  });
  assert.equal(calls.tooltip.className, 'project-site-tooltip');
  assert.equal(calls.warningOptions.permanent, true);
  assert.equal(calls.warningOptions.interactive, false);
  assert.equal(calls.warningOptions.className, 'project-site-danger-label');
  assert.equal(calls.label, '⚠ Danger: Big waves');
  assert.equal(calls.visible, true);
  events.remove();
  assert.equal(calls.visible, false);
  events.add();
  assert.equal(calls.visible, true);
  assert.equal(calls.style.color, '#b91c1c');
  assert.equal(typeof calls.popup, 'function');
  dom.window.close();
});
