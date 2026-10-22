import test from 'node:test';
import assert from 'node:assert/strict';
import { geometryCoversPoint } from './zoneGeometry.js';

test('zone counts follow concave boundaries, holes, islands and covered edges', () => {
  const geometry = { type: 'MultiPolygon', coordinates: [
    [[[0,0],[4,0],[4,4],[2,2],[0,4],[0,0]], [[.5,.5],[1,.5],[1,1],[.5,1],[.5,.5]]],
    [[[6,0],[7,0],[7,1],[6,1],[6,0]]],
  ] };
  const covers = (longitude, latitude) => geometryCoversPoint(geometry, { longitude, latitude });
  assert.equal(covers(2, 1), true);
  assert.equal(covers(2, 3), false);
  assert.equal(covers(.75, .75), false);
  assert.equal(covers(.5, .75), true);
  assert.equal(covers(0, 1), true);
  assert.equal(covers(6.5, .5), true);
  assert.equal(covers(5, .5), false);
});
