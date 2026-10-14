import { getMapPointStatus } from './mapPointStats.js';

export function filterMapPoints(points, { deadOnly = false } = {}) {
  // Recording a death changes the marker status; it does not release the
  // location. Only an explicit release/replanting action changes ownership.
  return points.filter((point) => {
    const status = getMapPointStatus(point);
    return status !== null && (!deadOnly || status === 'dead');
  });
}
