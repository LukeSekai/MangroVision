import test from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { createProjectSiteInfo, emptyProjectSiteCounts, projectSiteAreaM2, projectSitePointCounts } from './projectSiteInfo.js';

test('site counts reconcile each current location once using resolved site ownership', () => {
  const counts = projectSitePointCounts([
    { id: 1, source_project_site_id: 7, map_status: 'planned', project_site_id: 9 },
    { id: '1', source_project_site_id: '7', map_status: 'planned' },
    { id: 2, source_project_site_id: '7', assignment_status: 'pending' },
    { id: 3, source_site_id: 7, assignment_status: 'completed' },
    { id: 4, source_project_site_id: 7, death_at: '2026-10-10', planting_status: 'planted' },
    { id: 5, source_project_site_id: 7, assignment_status: 'skipped' },
    { id: 6, source_project_site_id: 7, eroded_unavailable: true },
    { id: 8, source_project_site_id: 7, deleted_at: '2026-10-10' },
    { id: 9, source_project_site_id: 7, is_deleted: true },
    { id: 10, source_project_site_id: 9, map_status: 'planted' },
    { id: 11, source_project_site_id: null, organization_id: 7 },
  ]);
  assert.deepEqual(counts.get('7'), {
    mapped: 6, planned: 1, assigned: 1, planted: 1, dead: 1, skipped: 1, unavailable: 1,
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
    mapped: 4, planned: 0, assigned: 1, planted: 1, dead: 1, skipped: 0, unavailable: 1,
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

test('user-entered names, organizations and notes remain plain text in map overlays', () => {
  const payload = '<img src=x onerror=alert(1)>';
  const { text, html } = card({ properties: { name: payload, organization_name: `Owner ${payload}`, notes: payload } });
  assert.equal(text.split(payload).length - 1, 3);
  assert.doesNotMatch(html, /<img|<script/);
  assert.match(html, /&lt;img/);
});
