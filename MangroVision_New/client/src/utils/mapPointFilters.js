export function filterMapPoints(points, { includeDead = false, deadOnly = false } = {}) {
  if (includeDead) return deadOnly ? points.filter((point) => Boolean(point.death_at)) : points;
  return points.filter((point) => !point.death_at);
}
