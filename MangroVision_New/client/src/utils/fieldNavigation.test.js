import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { googleMapsDirectionsUrl, navigableFieldPoint, navigationPointQueue, navigationSegments, navigationStartPoint, pointGuidance, retargetSiteRoute, upcomingNavigationPoints } from './fieldNavigation.js';

test('field direction follows geographic compass bearings in every quadrant', () => {
  for (const [destination, direction, bearing] of [
    [[0.001, 0], 'N', 0], [[0, 0.001], 'E', 90],
    [[-0.001, 0], 'S', 180], [[0, -0.001], 'W', 270],
  ]) {
    const result = pointGuidance([0,0], destination);
    assert.equal(result.direction, direction);
    assert.ok(Math.abs(result.bearing - bearing) < 0.001);
    assert.ok(Math.abs(result.distance - 111.195) < 0.1);
  }
});

test('distance decreases with movement toward the actual NASUGBAN target', () => {
  const target = [10.7798913043478, 122.627158010418];
  const distant = pointGuidance([target[0] - 0.000085, target[1]], target);
  const near = pointGuidance([target[0] - 0.00001, target[1]], target);
  assert.ok(distant.distance > near.distance);
  assert.equal(distant.distanceLabel, '9 m');
  assert.equal(near.distanceLabel, '1 m');
  assert.equal(pointGuidance(target, target).distance, 0);
  assert.equal(pointGuidance(null, target), null);
});

test('Maps action uses current GPS and the exact planting point', () => {
  const target = [10.7798913043478,122.627158010418];
  const url = new URL(googleMapsDirectionsUrl(target, [10.70,122.55]));
  assert.equal(url.origin, 'https://www.google.com');
  assert.equal(url.pathname, '/maps/dir/');
  assert.equal(url.searchParams.get('api'), '1');
  assert.equal(url.searchParams.get('dir_action'), 'navigate');
  assert.equal(url.searchParams.get('destination'), target.join(','));
  assert.equal(url.searchParams.get('origin'), '10.7,122.55');
  assert.equal(url.searchParams.get('travelmode'), 'walking');
  assert.equal(url.searchParams.get('avoid'), 'ferries');
  assert.equal(googleMapsDirectionsUrl([91,122], [10.70,122.55]), null);
  assert.equal(googleMapsDirectionsUrl(target, null), null);
  assert.equal(googleMapsDirectionsUrl(null), null);
});

const target = [0.003,0.0005];
const route = { target, segments: [
  { kind:'guidance', polyline:[[-0.001,0],[0,0]] },
  { kind:'road', polyline:[[0,0],[0.001,0],[0.002,0]] },
  { kind:'guidance', polyline:[[0.002,0],target] },
] };

test('remote GPS fixes do not invent a connection across unmapped terrain', () => {
  const current = [-0.001,0.001];
  const segments = navigationSegments(route,current);
  assert.deepEqual(segments,[route.segments[1]]);
});

test('movement along a mapped road drops the already passed road start', () => {
  const current = [0.001,0.00005];
  const segments = navigationSegments(route,current);
  assert.ok(pointGuidance(current,segments[0].polyline[0]).distance <= 12);
  assert.ok(!segments.flatMap(s=>s.polyline).some(p=>p[0]===0 && p[1]===0));
  assert.deepEqual(segments.at(-1).polyline.at(-1),[0.002,0]);
});

test('arriving inside the site draws guidance from current GPS to the exact point', () => {
  const current = [0.0029,0.0005];
  const site_area = [[0.0028,0.0004],[0.0032,0.0004],[0.0032,0.0006],[0.0028,0.0006],[0.0028,0.0004]];
  const segments = navigationSegments({...route,site_area},current,3);
  assert.deepEqual(segments,[{kind:'guidance',polyline:[current,target]}]);
  assert.deepEqual(navigationSegments(route,target,3),[]);
  assert.deepEqual(navigationSegments(route,null),[]);
});

test('provider failure never draws a straight GPS-to-point line', () => {
  assert.deepEqual(navigationSegments({target,segments:[],route_source:'point_guidance'},[-0.1,-0.1]),[]);
  assert.deepEqual(navigationSegments({target,segments:[route.segments[0],route.segments[2]]},[-0.1,-0.1]),[]);
});

const siteRoute = { ...route, entrance:[0.002,0],
  site_area:[[0.002,0],[0.002,0.001],[0.004,0.001],[0.004,0],[0.002,0]] };

test('remote directions retain mapped roads then continue from entrance to point', () => {
  const current = [-0.001,0.001];
  const segments = navigationSegments(siteRoute,current);
  assert.deepEqual(segments,[route.segments[1],{kind:'guidance',polyline:[siteRoute.entrance,target]}]);
  assert.ok(segments.every(segment => segment.kind !== 'guidance' || !segment.polyline.includes(current)));
});

test('road-only cached responses still display the final section to the point', () => {
  const cached = {...siteRoute,segments:[route.segments[1]]};
  assert.deepEqual(navigationSegments(cached,[-0.001,0.001]).at(-1),
    {kind:'guidance',polyline:[siteRoute.entrance,target]});
});

test('provider outage keeps local point guidance without linking distant GPS to site', () => {
  const current = [-0.1,-0.1];
  assert.deepEqual(navigationSegments({...siteRoute,segments:[],route_source:'partial_route'},current),
    [{kind:'guidance',polyline:[siteRoute.entrance,target]}]);
});

test('onsite point guidance follows new GPS fixes without routing back to entrance', () => {
  const first = [0.0026,0.0003];
  const next = [0.0029,0.0004];
  assert.deepEqual(navigationSegments(siteRoute,first,9),[{kind:'guidance',polyline:[first,target]}]);
  assert.deepEqual(navigationSegments(siteRoute,next,9),[{kind:'guidance',polyline:[next,target]}]);
  assert.deepEqual(navigationSegments(siteRoute,target,9),[]);
});

test('Google Maps walking link uses the access road before the exact point', () => {
  const options={access_start:[10.78359236,122.61995316],entrance:[10.78102952,122.62457728],
    site_area:[[10.77,122.62],[10.79,122.62],[10.79,122.64],[10.77,122.64]],travel_mode:'walking'};
  const destination=[10.7799452030183,122.626825821686];
  const remote=new URL(googleMapsDirectionsUrl(destination,[10.70,122.55],options));
  assert.equal(remote.searchParams.get('waypoints'),'10.78359236,122.61995316|10.78102952,122.62457728');
  assert.equal(remote.searchParams.get('destination'),destination.join(','));
  assert.equal(remote.searchParams.get('travelmode'),'walking');
  assert.equal(new URL(googleMapsDirectionsUrl(destination,destination,options)).searchParams.has('waypoints'),false);
});

const bridgeSite = JSON.parse(readFileSync(new URL('../../../api/data/site_access_routes.json', import.meta.url)))[0];
const bridgeRoute = {
  target: [10.78012, 122.62707], entrance: bridgeSite.access_path.at(-1),
  site_area: bridgeSite.site_area, planting_area: bridgeSite.planting_area,
  segments: [{ kind: 'road', polyline: bridgeSite.access_path }],
};

test('the actual bridge remains the approach from its start and live GPS along the walkway', () => {
  for (const current of [bridgeSite.bridge_start, bridgeSite.access_path[20], bridgeSite.access_path.at(-4)]) {
    const segments = navigationSegments(bridgeRoute, current, 20);
    assert.equal(segments[0].kind, 'road');
    assert.deepEqual(segments[0].polyline.at(-1), bridgeRoute.entrance);
    assert.deepEqual(segments.at(-1), { kind: 'guidance', polyline: [bridgeRoute.entrance, bridgeRoute.target] });
    assert.ok(!segments.some(segment => segment.kind === 'guidance' && segment.polyline[0] === current));
  }
});

test('being in the wooded park footprint does not bypass the bridge', () => {
  const current = [10.7806, 122.6257];
  const segments = navigationSegments(bridgeRoute, current, 5);
  assert.deepEqual(segments[0], bridgeRoute.segments[0]);
  assert.deepEqual(segments.at(-1).polyline[0], bridgeRoute.entrance);
});

test('on the foreshore, live guidance uses current GPS without returning to the bridge', () => {
  const current = [10.78005, 122.62707];
  assert.deepEqual(navigationSegments(bridgeRoute, current, 5),
    [{ kind: 'guidance', polyline: [current, bridgeRoute.target] }]);
  const boundary = [10.7798063043478, 122.627158010418];
  const target = [10.7798913043478, 122.627158010418];
  assert.deepEqual(navigationSegments({ ...bridgeRoute, target }, boundary, 12),
    [{ kind: 'guidance', polyline: [boundary, target] }]);
});

test('a straight direction through the trees is omitted while the fixed bridge remains', () => {
  const segments = navigationSegments({ ...bridgeRoute, target: [10.7805, 122.626] }, [10.70, 122.55]);
  assert.deepEqual(segments, bridgeRoute.segments);
});

const assigned = [100, 99, 98, 97].map((point_num, index) => ({
  assignment_point_id: index + 1, point_num, assignment_status: 'pending',
  latitude: 10.78012, longitude: 122.62707 + index * 0.00003,
}));

test('navigation starts at the chosen point then visits its nearest remaining assignments', () => {
  const points = [...assigned, { ...assigned[0], assignment_point_id: 5, assignment_status: 'completed' },
    { ...assigned[0], assignment_point_id: 6, assignment_status: 'skipped' },
    { ...assigned[0], assignment_point_id: 7, eroded_unavailable: true },
    { ...assigned[0], assignment_point_id: 8, latitude: null },
    { ...assigned[0], assignment_point_id: 9, latitude: 91 }];
  const before = structuredClone(points);
  assert.deepEqual(navigationPointQueue(points, 2).map(point => point.assignment_point_id), [2, 1, 3, 4]);
  assert.deepEqual(points, before, 'Navigation must not change planting status or ownership');
  assert.equal(navigableFieldPoint(points[4]), false);
});

test('the full path avoids the longer return trip caused by greedy nearest-neighbour choices', () => {
  const positions = [[0.00012, 0.00010], [0.00006, 0], [0.00006, 0.00010],
    [0.00006, 0.00022], [0, 0], [0, 0.00018]];
  const points = positions.map(([latitude, longitude], index) => ({ ...assigned[0], assignment_point_id: index + 1,
    latitude, longitude }));
  const queue = navigationPointQueue(points, 5);
  assert.deepEqual(queue.map(point => point.assignment_point_id), [5, 2, 1, 3, 6, 4]);
  const length = order => order.slice(1).reduce((sum, point, index) => sum
    + pointGuidance([order[index].latitude, order[index].longitude], [point.latitude, point.longitude]).distance, 0);
  const greedy = [5, 2, 3, 1, 4, 6].map(id => points.find(point => point.assignment_point_id === id));
  assert.ok(length(queue) < length(greedy));
  assert.deepEqual(navigationPointQueue([...points].reverse(), 5), queue);
});

test('equal-distance neighbours have a stable order and an empty assignment has no route', () => {
  const points = [assigned[0], { ...assigned[1], longitude: assigned[0].longitude - 0.00003 }, assigned[1]];
  points[1] = { ...points[1], assignment_point_id: 3 };
  assert.deepEqual(navigationPointQueue(points, 1).map(point => point.assignment_point_id), [1, 2, 3]);
  assert.deepEqual(navigationPointQueue([...points].reverse(), 1).map(point => point.assignment_point_id), [1, 2, 3]);
  assert.deepEqual(navigationPointQueue([], 1), []);
});

test('refresh keeps the planned order while dropping an unavailable stop and passed points', () => {
  const offsets = [0, 2, 5, -3];
  const points = assigned.map((point, index) => ({ ...point, longitude: assigned[0].longitude + offsets[index] * 0.00003 }));
  const route = { assignmentPointId: 1, pointIds: navigationPointQueue(points, 1).map(point => point.assignment_point_id) };
  assert.deepEqual(upcomingNavigationPoints(points, route).map(point => point.assignment_point_id), [4, 2, 3]);
  points[1] = { ...points[1], assignment_status: 'completed' };
  assert.deepEqual(upcomingNavigationPoints(points, route).map(point => point.assignment_point_id), [4, 3]);
  const continued = { assignmentPointId: 4, pointIds: [4, 3] };
  assert.deepEqual(upcomingNavigationPoints(points, continued).map(point => point.assignment_point_id), [3]);
});

test('the screenshot layout follows 95 → 100 → 99 → 94 → 88 and continues outward to 46', () => {
  // Relative locations from the screenshot, expressed as geographic coordinates.
  const positions = [[95,330,255],[100,270,291],[99,270,357],[94,330,323],[88,390,357],
    [82,448,323],[76,507,357],[70,566,323],[58,686,323],[46,802,323]];
  const points = positions.map(([point_num, x, y]) => ({ ...assigned[0], point_num, assignment_point_id: point_num,
    latitude: 10.78014 - (y - 255) * 0.0000001, longitude: 122.62707 + (x - 270) * 0.0000001 }));
  assert.deepEqual(navigationPointQueue([...points].reverse(), 95).map(point => point.point_num), positions.map(row => row[0]));
  const initial = navigationPointQueue(points, 95).map(point => point.assignment_point_id);
  for (let index = 0; index < initial.length; index++) {
    assert.deepEqual(upcomingNavigationPoints([...points].reverse(), { assignmentPointId: initial[index], pointIds: initial })
      .map(point => point.assignment_point_id), initial.slice(index + 1));
  }
});

test('the automatic starting point is nearest to the physical entrance, independent of list order and GPS', () => {
  const sites = [{ name: 'Leganes', label: 'LIKE entrance', entrance: bridgeRoute.entrance, site_area: bridgeSite.site_area }];
  const points = assigned.map(point => ({ ...point }));
  points[0].latitude -= 0.00003; // Row/list order would begin at a different point.
  const recommended = navigationStartPoint([...points].reverse(), sites);
  assert.equal(recommended.point.point_num, 100);
  assert.deepEqual(recommended.entrance, bridgeRoute.entrance);
  assert.equal(navigationPointQueue(points, null, recommended.entrance)[0].point_num, 100);
  points[0].assignment_status = 'completed';
  assert.equal(navigationStartPoint(points, sites).point.point_num, 99);
  assert.equal(navigationStartPoint(points.map(point => ({ ...point, inside_eroded_zone: true })), sites), null);
  assert.equal(navigationStartPoint([{ ...assigned[0], latitude: 11 }], sites), null);
});

test('large assignment planning is deterministic, retains every point once and finishes at the far end', () => {
  const points = Array.from({ length: 35 }, (_, index) => ({ ...assigned[0], assignment_point_id: index + 1,
    longitude: assigned[0].longitude + index * 0.00001, latitude: assigned[0].latitude + (index % 2) * 0.000004 }));
  const queue = navigationPointQueue(points, 1);
  assert.deepEqual(queue.map(point => point.assignment_point_id), points.map(point => point.assignment_point_id));
  assert.deepEqual(navigationPointQueue([...points].reverse(), 1), queue);
});

test('Next advances past the active target and drops completed, skipped, missing or unavailable future points', () => {
  const route = { assignmentPointId: 2, pointIds: [1, 2, 3, 4] };
  assert.deepEqual(upcomingNavigationPoints(assigned, route).map(point => point.assignment_point_id), [3, 4]);
  const changed = assigned.map(point => point.assignment_point_id === 3 ? { ...point, assignment_status: 'completed' } : point);
  assert.deepEqual(upcomingNavigationPoints(changed, route).map(point => point.assignment_point_id), [4]);
  assert.deepEqual(upcomingNavigationPoints(assigned, { ...route, assignmentPointId: 4 }), []);
  assert.deepEqual(upcomingNavigationPoints(assigned, null), assigned);
});

test('the orange path connects the active point to every upcoming point, even on arrival at the active point', () => {
  const route = { ...bridgeRoute, upcomingPoints: assigned.slice(1) };
  const continuations = assigned.slice(1).map((point, index) => ({ kind: 'guidance', polyline: [
    [assigned[index].latitude, assigned[index].longitude], [point.latitude, point.longitude],
  ] }));
  assert.deepEqual(navigationSegments(route, bridgeSite.bridge_start).slice(-3), continuations);
  assert.deepEqual(navigationSegments(route, route.target), continuations);
  assert.deepEqual(navigationSegments({ ...route, upcomingPoints: [] }, route.target), []);
});

test('future point connections keep the bridge and do not invent connections through trees or to other sites', () => {
  const route = { ...bridgeRoute, upcomingPoints: [
    { latitude: 10.7805, longitude: 122.626 },
    { latitude: 11, longitude: 122 },
  ] };
  assert.deepEqual(navigationSegments(route, bridgeSite.bridge_start), navigationSegments(bridgeRoute, bridgeSite.bridge_start));
  const next = retargetSiteRoute(bridgeRoute, assigned[1]);
  assert.deepEqual(next.target, [assigned[1].latitude, assigned[1].longitude]);
  assert.equal(next.segments, bridgeRoute.segments, 'Next point retains the fixed bridge');
  assert.equal(retargetSiteRoute(bridgeRoute, { latitude: 11, longitude: 122 }), null);
  assert.equal(retargetSiteRoute({ ...bridgeRoute, route_source: 'within_site', segments: [] }, assigned[1], bridgeSite.bridge_start), null,
    'Leaving the foreshore requires a new bridge route instead of reusing onsite-only guidance');
});
