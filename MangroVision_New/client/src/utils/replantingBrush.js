export const REPLANTING_BRUSH_RADIUS = 18;

// Test the whole sweep between pointer events so fast movement cannot skip dots.
export function pointsAlongBrush(points, start, end, radius = REPLANTING_BRUSH_RADIUS) {
  const dx = end.x - start.x;
  const dy = end.y - start.y;
  const lengthSquared = dx * dx + dy * dy;
  return points.filter((point) => {
    const t = lengthSquared ? Math.max(0, Math.min(1,
      ((point.x - start.x) * dx + (point.y - start.y) * dy) / lengthSquared)) : 0;
    return (point.x - start.x - t * dx) ** 2 + (point.y - start.y - t * dy) ** 2 <= radius ** 2;
  }).map((point) => point.id);
}

export function paintReplantingSelection(current, ids, mode, limit = 500) {
  const next = new Set(current);
  for (const id of ids) {
    if (mode === 'deselect') next.delete(id);
    else if (mode === 'select' && next.size < limit) next.add(id);
  }
  const result = [...next];
  return result.length === current.length && result.every((id, index) => id === current[index]) ? current : result;
}
