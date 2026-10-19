import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
import { readFile } from 'node:fs/promises';

let dom, server, L, cachedTileLayer, TileImagePool, requests, images, autoLoad, maps;
const globals = new Map();
before(async () => {
  dom = new JSDOM('', { url: 'http://localhost', pretendToBeVisual: true });
  for (const [key, value] of Object.entries({ window: dom.window, document: dom.window.document,
    navigator: dom.window.navigator, requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
    cancelAnimationFrame: dom.window.cancelAnimationFrame.bind(dom.window) })) {
    globals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  const source = Object.getOwnPropertyDescriptor(window.HTMLImageElement.prototype, 'src');
  Object.defineProperty(window.HTMLImageElement.prototype, 'src', {
    get() { return source.get.call(this); },
    set(url) {
      source.set.call(this, url);
      requests.set(url, (requests.get(url) || 0) + 1);
      images.set(this, { complete: false, width: 0 });
      if (autoLoad) queueMicrotask(() => finish(this));
    }, configurable: true,
  });
  Object.defineProperty(window.HTMLImageElement.prototype, 'complete', { get() { return images.get(this)?.complete || false; }, configurable: true });
  Object.defineProperty(window.HTMLImageElement.prototype, 'naturalWidth', { get() { return images.get(this)?.width || 0; }, configurable: true });
  L = (await import('leaflet')).default;
  server = await createServer({ configFile: false, appType: 'custom',
    server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    cacheDir: 'node_modules/.vite-tile-cache-tests', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ cachedTileLayer, TileImagePool } = await server.ssrLoadModule('/src/utils/cachedTileLayer.js'));
});
after(async () => {
  await server?.close(); dom?.window.close();
  for (const [key, descriptor] of globals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
  }
});
beforeEach(() => { requests = new Map(); images = new Map(); autoLoad = true; maps = []; });
afterEach(() => { for (const map of maps) map.remove(); document.body.innerHTML = ''; });
const settle = async () => { await new Promise((resolve) => setTimeout(resolve, 30)); };
function finish(tile, failed = false) {
  images.set(tile, { complete: true, width: failed ? 0 : 256 });
  tile.dispatchEvent(new window.Event(failed ? 'error' : 'load'));
}
function mapLayer(options = {}) {
  const element = document.createElement('div'); document.body.append(element);
  Object.defineProperties(element, { clientWidth: { value: 512 }, clientHeight: { value: 512 } });
  const map = L.map(element, { zoomAnimation: false, fadeAnimation: false, zoomControl: false,
    attributionControl: false }).setView([10.78, 122.6253], 18);
  maps.push(map);
  const layer = cachedTileLayer('/tiles/FINAL/{z}/{x}/{y}.png', {
    maxZoom: 24, maxNativeZoom: 22, imagePool: new TileImagePool(), ...options,
  }).addTo(map);
  return { map, layer };
}

test('actual Leaflet zoom and pan loops reuse the original images without setting src again', async () => {
  const { map, layer } = mapLayer(); await settle();
  for (const zoom of [19, 17, 20, 18]) { map.setZoom(zoom, { animate: false }); await settle(); }
  const downloaded = new Map(requests);
  for (const zoom of [19, 17, 20, 18]) { map.setZoom(zoom, { animate: false }); await settle(); }
  assert.deepEqual(requests, downloaded);
  map.panTo([10.784, 122.628], { animate: false }); await settle();
  map.panTo([10.78, 122.6253], { animate: false }); await settle();
  const panned = new Map(requests);
  map.panTo([10.784, 122.628], { animate: false }); await settle();
  map.panTo([10.78, 122.6253], { animate: false }); await settle();
  assert.deepEqual(requests, panned);
  map.removeLayer(layer); map.addLayer(layer); await settle();
  assert.deepEqual(requests, panned);
  assert.equal(layer.isLoading(), false);
  assert.ok([...requests.values()].every((count) => count === 1));
});

test('rapid zoom changes keep pending downloads and reattach them without restarting', async () => {
  autoLoad = false;
  const { map, layer } = mapLayer();
  const original = new Map(requests);
  map.setZoom(19, { animate: false });
  map.setZoom(18, { animate: false });
  for (const [url, count] of original) assert.equal(requests.get(url), count);
  for (const tile of [...images.keys()]) finish(tile);
  await settle();
  assert.equal(layer.isLoading(), false);
  assert.ok([...requests.values()].every((count) => count === 1));
});

test('failed downloads are retried rather than cached as blank tiles', async () => {
  autoLoad = false;
  const { map, layer } = mapLayer();
  for (const tile of [...images.keys()]) finish(tile, true);
  const previous = new Map(requests);
  map.removeLayer(layer); map.addLayer(layer);
  for (const [url, count] of previous) assert.equal(requests.get(url), count + 1);
  for (const tile of Object.values(layer._tiles).map((entry) => entry.el)) finish(tile);
  await settle();
  assert.equal(layer.isLoading(), false);
});

test('overzoom above native resolution reuses the same source tiles', async () => {
  const { map } = mapLayer();
  map.setZoom(22, { animate: false }); await settle();
  const downloaded = new Map(requests);
  for (const zoom of [23, 24, 22, 23, 22]) { map.setZoom(zoom, { animate: false }); await settle(); }
  assert.deepEqual(requests, downloaded);
});

test('retained images are bounded, expire and do not mix revised imagery', async () => {
  let now = 0;
  const pool = new TileImagePool({ maxTiles: 2, maxAge: 100, now: () => now });
  const { map, layer } = mapLayer({ imagePool: pool }); await settle();
  map.removeLayer(layer);
  assert.ok(pool.order.size <= 2);
  now = 101;
  assert.equal(pool.take(pool.order.values().next().value), null);
  map.addLayer(layer); await settle();
  const before = [...requests.values()].reduce((sum, value) => sum + value, 0);
  layer.setUrl('/tiles/FINAL/{z}/{x}/{y}.png?v=new-imagery'); await settle();
  assert.ok([...requests.keys()].some((url) => url.endsWith('?v=new-imagery')));
  assert.ok([...requests.values()].reduce((sum, value) => sum + value, 0) > before);
});

test('simultaneous maps never steal an image that is still visible in another map', async () => {
  const pool = new TileImagePool();
  const first = mapLayer({ imagePool: pool }); await settle();
  const originalTiles = Object.values(first.layer._tiles).map((entry) => entry.el);
  const second = mapLayer({ imagePool: pool }); await settle();
  const secondTiles = new Set(Object.values(second.layer._tiles).map((entry) => entry.el));
  assert.ok(originalTiles.every((tile) => tile.isConnected && !secondTiles.has(tile)));
  first.map.removeLayer(first.layer);
  assert.ok([...secondTiles].every((tile) => tile.isConnected));
});

test('imagery revisions change the configured tile URL without changing coordinates', async () => {
  const source = await readFile(new URL('../config/mapTiles.js', import.meta.url), 'utf8');
  const config = await import(`data:text/javascript;base64,${Buffer.from(`import.meta.env = { VITE_TILESET_REVISION: 'new imagery/2' };\n${source}`).toString('base64')}`);
  assert.equal(config.ORTHOPHOTO_TILE_URL, '/tiles/FINAL/{z}/{x}/{y}.png?v=new%20imagery%2F2');
});
