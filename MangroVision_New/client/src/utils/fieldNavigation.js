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
// Travel mode is chosen by Google Maps.
// Constructing the link makes no request; it opens only after an explicit tap.
function validCoordinate(point) {
  return Array.isArray(point) && point.length === 2
    && point.every(Number.isFinite) && Math.abs(point[0]) <= 90 && Math.abs(point[1]) <= 180;
}

export function googleMapsDirectionsUrl(destination, origin) {
  if (!validCoordinate(destination) || !validCoordinate(origin)) return null;
  const params = new URLSearchParams({ api: '1', origin: origin.join(','),
    destination: destination.join(','), dir_action: 'navigate' });
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

// Update the first coordinate with each GPS fix, trim roads already reached,
// and switch to the exact point once inside the physical planting site.
export function navigationSegments(route, currentPosition, accuracy = null) {
  if (!route?.target || !validCoordinate(currentPosition)) return [];
  const direct = [{ kind: 'guidance', polyline: [currentPosition, route.target] }];
  if (nearSite(currentPosition, route.site_area, accuracy)
    || pointGuidance(currentPosition, route.target).distance <= Math.max(3, Math.min(accuracy || 0, 20))) return direct;
  const segments = route.segments?.length ? route.segments : route.polyline?.length >= 2
    ? [{ kind: 'road', polyline: route.polyline }] : [];
  if (!segments.some(segment => segment.kind === 'road')) return direct;
  let nearest = null;
  segments.forEach((segment, segmentIndex) => {
    if (segment.kind !== 'road') return;
    for (let i = 0; i < segment.polyline.length - 1; i++) {
      const projection = projectOnSegment(currentPosition, segment.polyline[i], segment.polyline[i + 1]);
      const distance = pointGuidance(currentPosition, projection).distance;
      if (!nearest || distance < nearest.distance) nearest = { distance, segmentIndex, index: i, projection };
    }
  });
  let remaining;
  if (nearest && nearest.distance <= 12) {
    const segment = segments[nearest.segmentIndex];
    remaining = [{ kind: 'guidance', polyline: [currentPosition, nearest.projection] },
      { kind: 'road', polyline: [nearest.projection, ...segment.polyline.slice(nearest.index + 1)] },
      ...segments.slice(nearest.segmentIndex + 1)];
  } else if (segments[0].kind === 'guidance') {
    remaining = [{ ...segments[0], polyline: [currentPosition, ...segments[0].polyline.slice(1)] }, ...segments.slice(1)];
  } else {
    remaining = [{ kind: 'guidance', polyline: [currentPosition, segments[0].polyline[0]] }, ...segments];
  }
  const last = remaining.at(-1).polyline.at(-1);
  if (last[0] !== route.target[0] || last[1] !== route.target[1]) {
    remaining.push({ kind: 'guidance', polyline: [last, route.target] });
  }
  return remaining;
}
