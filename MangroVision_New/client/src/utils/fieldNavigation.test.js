import test from 'node:test';
import assert from 'node:assert/strict';
import { googleMapsDirectionsUrl, navigationSegments, pointGuidance } from './fieldNavigation.js';

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
