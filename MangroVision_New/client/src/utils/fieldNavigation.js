// True bearing and distance to a coordinate, not a surveyed walking path.
export function pointGuidance(origin, destination) {
  if (!origin || !destination) return null;
  const [lat1, lon1, lat2, lon2] = [...origin, ...destination].map((value) => value * Math.PI / 180);
  const hav = Math.sin((lat2 - lat1) / 2) ** 2
    + Math.cos(lat1) * Math.cos(lat2) * Math.sin((lon2 - lon1) / 2) ** 2;
  const distance = 6371000 * 2 * Math.asin(Math.sqrt(Math.min(1, hav)));
  const bearing = (Math.atan2(Math.sin(lon2 - lon1) * Math.cos(lat2),
    Math.cos(lat1) * Math.sin(lat2) - Math.sin(lat1) * Math.cos(lat2) * Math.cos(lon2 - lon1)) * 180 / Math.PI + 360) % 360;
  const direction = ['N', 'NE', 'E', 'SE', 'S', 'SW', 'W', 'NW'][Math.round(bearing / 45) % 8];
  return { distance, bearing, direction,
    distanceLabel: distance >= 1000 ? `${(distance / 1000).toFixed(2)} km` : `${Math.round(distance)} m` };
}

// https://developers.google.com/maps/documentation/urls/get-started#directions
// Use the latest device GPS as origin and the exact planting point as target.
// Constructing the link makes no request; it opens only after an explicit tap.
function validCoordinate(point) {
  return Array.isArray(point) && point.length === 2
    && point.every(Number.isFinite) && Math.abs(point[0]) <= 90 && Math.abs(point[1]) <= 180;
}

export function navigableFieldPoint(point) {
  return Boolean(point)
    && !['planted', 'completed', 'skipped'].includes(point.assignment_status)
    && !point.eroded_unavailable && !point.inside_eroded_zone
    && point.latitude != null && point.longitude != null
    && validCoordinate([Number(point.latitude), Number(point.longitude)]);
}

const fieldPointCoordinates = point => [Number(point.latitude), Number(point.longitude)];
const comparePointIds = (a, b) => String(a.assignment_point_id).localeCompare(String(b.assignment_point_id), 'en', { numeric: true });
const distanceMm = (a, b) => Math.round(pointGuidance(a, b).distance * 1000);

export function navigationStartPoint(points, sites = []) {
  let result = null, nearestDistance = Infinity;
  for (const point of points.filter(navigableFieldPoint)) {
    const coordinates = fieldPointCoordinates(point);
    const site = sites.find(candidate => validCoordinate(candidate.entrance) && nearSite(coordinates, candidate.site_area, 3));
    if (!site) continue;
    const distance = distanceMm(site.entrance, coordinates);
    if (distance < nearestDistance || (distance === nearestDistance && comparePointIds(point, result.point) < 0)) {
      result = { point, entrance: site.entrance, label: site.label || site.name };
      nearestDistance = distance;
    }
  }
  return result;
}

function nearestPointOrder(points, origin) {
  const remaining = [...points], ordered = [];
  let current = origin;
  while (remaining.length) {
    let nearestIndex = 0, nearestDistanceMm = Infinity;
    remaining.forEach((point, index) => {
      const distanceMm = Math.round(pointGuidance(current, fieldPointCoordinates(point)).distance * 1000);
      // Sub-millimetre differences on the regular grid are rounding noise.
      // Break equal-distance ties by ID so server/list order cannot flip it.
      if (distanceMm < nearestDistanceMm || (distanceMm === nearestDistanceMm
        && String(point.assignment_point_id).localeCompare(String(remaining[nearestIndex].assignment_point_id), 'en', { numeric: true }) < 0)) {
        nearestIndex = index;
        nearestDistanceMm = distanceMm;
      }
    });
    const [nearest] = remaining.splice(nearestIndex, 1);
    ordered.push(nearest);
    current = fieldPointCoordinates(nearest);
  }
  return ordered;
}

function shortestPointPath(start, middle, finish) {
  const nodes = [start, ...middle, finish], n = middle.length;
  const coordinates = nodes.map(fieldPointCoordinates);
  const distance = coordinates.map(a => coordinates.map(b => distanceMm(a, b)));
  const size = 1 << n, costs = new Float64Array(size * n).fill(Infinity);
  const previous = new Int16Array(size * n).fill(-1);
  for (let i = 0; i < n; i++) costs[(1 << i) * n + i] = distance[0][i + 1];
  for (let mask = 1; mask < size; mask++) {
    for (let last = 0; last < n; last++) {
      const cost = costs[mask * n + last];
      if (!Number.isFinite(cost)) continue;
      for (let next = 0; next < n; next++) {
        if (mask & (1 << next)) continue;
        const index = (mask | (1 << next)) * n + next;
        const candidate = cost + distance[last + 1][next + 1];
        if (candidate < costs[index]) {
          costs[index] = candidate;
          previous[index] = last;
        }
      }
    }
  }
  let mask = size - 1, last = 0;
  for (let i = 1; i < n; i++) {
    if (costs[mask * n + i] + distance[i + 1][n + 1]
      < costs[mask * n + last] + distance[last + 1][n + 1]) last = i;
  }
  const path = [];
  while (mask) {
    path.unshift(middle[last]);
    const next = previous[mask * n + last];
    mask ^= 1 << last;
    last = next;
  }
  return [start, ...path, finish];
}

function improvePointPath(path) {
  // Bounded 2-opt for larger assignments. Keep the entrance end and far end fixed.
  const ordered = [...path];
  const distance = (a, b) => distanceMm(fieldPointCoordinates(a), fieldPointCoordinates(b));
  const passes = ordered.length > 200 ? 2 : 8;
  for (let pass = 0; pass < passes; pass++) {
    let improved = false;
    for (let i = 1; i < ordered.length - 2; i++) {
      const limit = ordered.length > 200 ? Math.min(ordered.length - 1, i + 40) : ordered.length - 1;
      for (let j = i + 1; j < limit; j++) {
        const before = distance(ordered[i - 1], ordered[i]) + distance(ordered[j], ordered[j + 1]);
        const after = distance(ordered[i - 1], ordered[j]) + distance(ordered[i], ordered[j + 1]);
        if (after >= before) continue;
        ordered.splice(i, j - i + 1, ...ordered.slice(i, j + 1).reverse());
        improved = true;
      }
    }
    if (!improved) break;
  }
  return ordered;
}

// Plan the entire walk, rather than making a greedy choice at every stop.
// Fix the near end and far end so a short first leg cannot strand points behind.
export function navigationPointQueue(points, startId, entrance = null) {
  const available = points.filter(navigableFieldPoint);
  const anchor = validCoordinate(entrance) ? entrance : null;
  const start = available.find(point => String(point.assignment_point_id) === String(startId))
    || (anchor ? [...available].sort(comparePointIds).reduce((nearest, point) => !nearest
      || distanceMm(anchor, fieldPointCoordinates(point)) < distanceMm(anchor, fieldPointCoordinates(nearest)) ? point : nearest, null) : available[0]);
  if (!start) return [];
  if (available.length < 3) return [start, ...available.filter(point => point !== start)];
  const origin = anchor || fieldPointCoordinates(start);
  const remaining = available.filter(point => point !== start).sort(comparePointIds);
  const finish = remaining.reduce((farthest, point) => distanceMm(origin, fieldPointCoordinates(point))
    >= distanceMm(origin, fieldPointCoordinates(farthest)) ? point : farthest);
  const middle = remaining.filter(point => point !== finish);
  if (available.length <= 14) return shortestPointPath(start, middle, finish);
  return improvePointPath([start, ...nearestPointOrder(middle, fieldPointCoordinates(start)), finish]);
}

export function upcomingNavigationPoints(points, route) {
  if (!route) return points.filter(navigableFieldPoint);
  const ids = route.pointIds || [route.assignmentPointId];
  const index = ids.findIndex(id => String(id) === String(route.assignmentPointId));
  const byId = new Map(points.filter(navigableFieldPoint).map(point => [String(point.assignment_point_id), point]));
  // Preserve the planned walk on refresh and Next. Drop unavailable points
  // without flipping the route back toward places already passed.
  return ids.slice(Math.max(0, index + 1)).map(id => byId.get(String(id))).filter(Boolean);
}

// All points in the mapped physical site share the same bridge approach.
// Other sites need their own route request instead of reusing this bridge.
export function retargetSiteRoute(route, point, currentPosition = null, accuracy = 3) {
  // A failed road request must be retried, rather than cached for every Next.
  if (route?.road_route_error || ['partial_route', 'point_guidance'].includes(route?.route_source)) return null;
  const target = [Number(point?.latitude), Number(point?.longitude)];
  if (route?.route_source === 'within_site' && validCoordinate(currentPosition)
    && !nearSite(currentPosition, route.planting_area || route.site_area, accuracy)) return null;
  return validCoordinate(target) && validCoordinate(route?.entrance)
    && nearSite(target, route?.site_area, 3) ? { ...route, target } : null;
}

export function googleMapsDirectionsUrl(destination, origin, route = null) {
  if (!validCoordinate(destination) || !validCoordinate(origin)) return null;
  const params = new URLSearchParams({ api: '1', origin: origin.join(','),
    destination: destination.join(','), dir_action: 'navigate', travelmode: route?.travel_mode || 'walking', avoid: 'ferries' });
  if (!nearSite(origin, route?.planting_area || route?.site_area, 3)) {
    const waypoints = [route?.access_start, route?.entrance].filter(validCoordinate);
    if (waypoints.length) params.set('waypoints', waypoints.map((point) => point.join(',')).join('|'));
  }
  return `https://www.google.com/maps/dir/?${params}`;
}

function projectOnSegment(point, a, b) {
  const scale = Math.cos(point[0] * Math.PI / 180);
  const dx = b[0] - a[0];
  const dy = (b[1] - a[1]) * scale;
  const lengthSquared = dx * dx + dy * dy;
  const t = lengthSquared ? Math.max(0, Math.min(1,
    ((point[0] - a[0]) * dx + (point[1] - a[1]) * scale * dy) / lengthSquared)) : 0;
  return [a[0] + t * (b[0] - a[0]), a[1] + t * (b[1] - a[1])];
}

function nearSite(point, polygon, accuracy) {
  if (!polygon?.length) return false;
  let inside = false;
  const tolerance = Math.max(3, Math.min(accuracy || 0, 20));
  for (let i = 0; i < polygon.length; i++) {
    const a = polygon[i];
    const b = polygon[(i + 1) % polygon.length];
    if ((a[1] > point[1]) !== (b[1] > point[1])
      && point[0] < (b[0] - a[0]) * (point[1] - a[1]) / (b[1] - a[1]) + a[0]) inside = !inside;
    if (pointGuidance(point, projectOnSegment(point, a, b)).distance <= tolerance) return true;
  }
  return inside;
}

function localGuidanceAllowed(start, target, area, tolerance = 3) {
  if (!area?.length) return true;
  const steps = Math.max(1, Math.ceil(pointGuidance(start, target).distance / 3));
  for (let step = 0; step <= steps; step++) {
    const point = start.map((value, axis) => value + (target[axis] - value) * step / steps);
    if (!nearSite(point, area, tolerance)) return false;
  }
  return true;
}

// A straight guide can leave a concave shoreline even with both ends inside.
// Bend at visible boundary vertices instead of dropping the final connection.
// This remains direction guidance; participants follow the marked lanes.
const boundaryVisibility = new WeakMap();

export function localGuidancePath(start, target, area, tolerance = 3) {
  if (localGuidanceAllowed(start, target, area, tolerance)) return [start, target];
  if (!nearSite(start, area, tolerance) || !nearSite(target, area, tolerance)) return null;
  const last = area.at(-1);
  const vertices = area[0][0] === last[0] && area[0][1] === last[1] ? area.slice(0, -1) : area;
  const nodes = [start, target, ...vertices];
  // GPS updates and successive assigned points share this fixed boundary.
  // Cache only its vertex connections; actual origin/target checks stay fresh.
  // Limit accuracy variants, and let discarded route areas be garbage-collected.
  let byTolerance = boundaryVisibility.get(area);
  if (!byTolerance) { byTolerance = new Map(); boundaryVisibility.set(area, byTolerance); }
  if (!byTolerance.has(tolerance)) {
    if (byTolerance.size >= 4) byTolerance.delete(byTolerance.keys().next().value);
    byTolerance.set(tolerance, new Map());
  }
  const visibility = byTolerance.get(tolerance);
  const canConnect = (a, b) => {
    if (a < 2 || b < 2) return localGuidanceAllowed(nodes[a], nodes[b], area, tolerance);
    const key = `${Math.min(a, b)}:${Math.max(a, b)}`;
    if (!visibility.has(key)) visibility.set(key, localGuidanceAllowed(nodes[a], nodes[b], area, tolerance));
    return visibility.get(key);
  };
  const costs = nodes.map(() => Infinity), previous = nodes.map(() => null);
  const pending = new Set(nodes.map((_, index) => index));
  costs[0] = 0;
  while (pending.size) {
    let current = null;
    for (const index of pending) {
      if (current == null || costs[index] < costs[current]) current = index;
    }
    if (!Number.isFinite(costs[current])) return null;
    if (current === 1) {
      const path = [];
      while (current != null) {
        path.unshift(nodes[current]);
        current = previous[current];
      }
      return path;
    }
    pending.delete(current);
    for (const neighbor of pending) {
      const cost = costs[current] + pointGuidance(nodes[current], nodes[neighbor]).distance;
      if (cost < costs[neighbor] && canConnect(current, neighbor)) {
        costs[neighbor] = cost;
        previous[neighbor] = current;
      }
    }
  }
  return null;
}

// Retain the road route and restore only local guidance to the active point.
// Never reuse an older server's remote GPS-to-point shortcut across water.
function activeNavigationSegments(route, currentPosition, accuracy) {
  if (!validCoordinate(route?.target) || !validCoordinate(currentPosition)) return [];
  const guide = (start, tolerance = 3) => {
    if (start[0] === route.target[0] && start[1] === route.target[1]) return [];
    // A compass line is only useful on the foreshore. Never draw one through
    // trees just because both ends belong to the larger park footprint.
    const polyline = localGuidancePath(start, route.target, route.planting_area, tolerance);
    return polyline ? [{ kind: 'guidance', polyline }] : [];
  };
  if (currentPosition[0] === route.target[0] && currentPosition[1] === route.target[1]) return [];
  const targetInSite = nearSite(route.target, route.site_area, 3);
  const segments = (route.segments != null ? route.segments : route.polyline?.length >= 2
    ? [{ kind: 'road', polyline: route.polyline }] : []).filter((segment) => segment.kind === 'road' && segment.polyline?.length >= 2);
  let nearest = null;
  segments.forEach((segment, segmentIndex) => {
    if (segment.kind !== 'road') return;
    for (let i = 0; i < segment.polyline.length - 1; i++) {
      const projection = projectOnSegment(currentPosition, segment.polyline[i], segment.polyline[i + 1]);
      const distance = pointGuidance(currentPosition, projection).distance;
      if (!nearest || distance < nearest.distance) nearest = { distance, segmentIndex, index: i, projection };
    }
  });
  let remaining = segments;
  if (nearest && nearest.distance <= 12) {
    const segment = segments[nearest.segmentIndex];
    remaining = [{ kind: 'road', polyline: [nearest.projection, ...segment.polyline.slice(nearest.index + 1)] },
      ...segments.slice(nearest.segmentIndex + 1)];
  }
  const remainingDistance = remaining.reduce((total, segment) => total + segment.polyline.slice(1)
    .reduce((length, point, index) => length + pointGuidance(segment.polyline[index], point).distance, 0), 0);
  if (targetInSite && nearSite(currentPosition, route.planting_area || route.site_area, accuracy)
    && (!nearest || nearest.distance > 12 || remainingDistance <= 12)) return guide(currentPosition, accuracy);
  if (targetInSite && validCoordinate(route.entrance) && nearSite(route.entrance, route.site_area, 3)) {
    return [...remaining, ...guide(route.entrance)];
  }
  const roadEnd = remaining.at(-1)?.polyline.at(-1);
  return roadEnd && pointGuidance(roadEnd, route.target).distance <= 20
    ? [...remaining, ...guide(roadEnd)] : remaining;
}

export function navigationSegments(route, currentPosition, accuracy = null) {
  if (!validCoordinate(route?.target) || !validCoordinate(currentPosition)) return [];
  const segments = activeNavigationSegments(route, currentPosition, accuracy);
  const points = [route.target, ...(route.upcomingPoints || []).map(point => [Number(point.latitude), Number(point.longitude)])];
  for (let index = 1; index < points.length; index++) {
    const start = points[index - 1], target = points[index];
    // Only connect known local planting locations. Never join different sites
    // or draw a continuation through the wooded bridge approach.
    if (!validCoordinate(start) || !validCoordinate(target)
      || !nearSite(start, route.site_area, 3) || !nearSite(target, route.site_area, 3)) continue;
    if (start[0] === target[0] && start[1] === target[1]) continue;
    const polyline = localGuidancePath(start, target, route.planting_area || route.site_area);
    if (polyline) segments.push({ kind: 'guidance', polyline });
  }
  return segments;
}
