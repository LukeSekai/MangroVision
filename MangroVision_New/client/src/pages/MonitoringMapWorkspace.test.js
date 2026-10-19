import test, { after, before } from 'node:test';
import assert from 'node:assert/strict';
import { createElement } from 'react';
import { renderToStaticMarkup } from 'react-dom/server';
import { MemoryRouter } from 'react-router-dom';
import { createServer } from 'vite';

let server;
let MonitoringMapWorkspace;
let MapAnalytics;

before(async () => {
  server = await createServer({
    configFile: false, appType: 'custom',
    server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    cacheDir: 'node_modules/.vite-monitoring-map-tests',
    optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ default: MonitoringMapWorkspace } = await server.ssrLoadModule('/src/pages/MonitoringMapWorkspace.jsx'));
  ({ default: MapAnalytics } = await server.ssrLoadModule('/src/pages/MapAnalytics.jsx'));
});

after(async () => { await server?.close(); });

test('dead-plant review belongs to the Monitoring map, not Map Analytics', () => {
  const render = (Component) => renderToStaticMarkup(createElement(
    MemoryRouter, null, createElement(Component),
  ));
  const monitoring = render(MonitoringMapWorkspace);
  const analytics = render(MapAnalytics);

  assert.match(monitoring, /Monitoring map/);
  assert.match(monitoring, /Show organization/);
  assert.match(monitoring, /All organizations/);
  assert.match(monitoring, /Show dead plants only/);
  assert.match(monitoring, /Brush select/);
  assert.match(monitoring, /Dead plants &amp; site release/);
  assert.match(monitoring, /Record a monitoring visit/);
  assert.doesNotMatch(analytics, /Show dead plants only|Review for replanting/);
});
