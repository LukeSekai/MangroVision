function ringLocation([x, y], ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const [a, b] = ring[j], [c, d] = ring[i];
    if (Math.abs((x - a) * (d - b) - (y - b) * (c - a)) < 1e-14
        && x >= Math.min(a, c) && x <= Math.max(a, c) && y >= Math.min(b, d) && y <= Math.max(b, d)) return 0;
    if ((b > y) !== (d > y) && x < (c - a) * (y - b) / (d - b) + a) inside = !inside;
  }
  return inside ? 1 : -1;
}

export function geometryCoversPoint(geometry, point) {
  const coordinate = [Number(point.longitude), Number(point.latitude)];
  if (!coordinate.every(Number.isFinite)) return false;
  const polygons = geometry.type === 'Polygon' ? [geometry.coordinates] : geometry.coordinates;
  return polygons.some(([outer, ...holes]) => ringLocation(coordinate, outer) >= 0
    && holes.every((hole) => ringLocation(coordinate, hole) <= 0));
}
