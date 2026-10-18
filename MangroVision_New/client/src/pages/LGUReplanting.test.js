import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement } from 'react';
import { createRoot } from 'react-dom/client';
import { MemoryRouter } from 'react-router-dom';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
import { getMapPointStatus } from '../utils/mapPointStats.js';
import { replantingProgress, replantingRows } from '../utils/lguReplanting.js';

const initialRows = [
  { planting_event_id: 1, planting_point_id: 11, project_site_id: 1, project_site_name: 'Coastal site', analysis_name: 'Photo A', point_num: 11, species: 'Bungalon', version: 0, replanting_status: 'awaiting_review', can_approve: true, death_at: '2026-10-01' },
  { planting_event_id: 2, planting_point_id: 12, project_site_id: 1, project_site_name: 'Coastal site', analysis_name: 'Photo A', point_num: 12, species: 'Bungalon', version: 4, replanting_status: 'ready_lgu', can_record_replanting: true, death_at: '2026-10-01' },
  { planting_event_id: 3, planting_point_id: 13, project_site_id: 1, project_site_name: 'Coastal site', analysis_name: 'Photo B', point_num: 13, species: 'Bungalon', version: 5, replanting_status: 'completed', death_at: '2026-10-01' },
];
let server, Page, dom, root, rows, requests, failPlanting;
const previous = new Map();

before(async () => {
  server = await createServer({ configFile: false, appType: 'custom',
    server: { middlewareMode: true, hmr: false, watch: null },
    optimizeDeps: { noDiscovery: true, include: [] },
  });
  Page = (await server.ssrLoadModule('/src/pages/LGUReplanting.jsx')).default;
});
after(async () => { await server?.close(); });

beforeEach(() => {
  dom = new JSDOM('<body><div id="root"></div></body>', { url: 'http://localhost/' });
  for (const [key, value] of Object.entries({ window: dom.window, document: dom.window.document,
    navigator: dom.window.navigator, HTMLElement: dom.window.HTMLElement, localStorage: dom.window.localStorage,
    IS_REACT_ACT_ENVIRONMENT: true })) {
    previous.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  rows = structuredClone(initialRows); requests = []; failPlanting = false;
  previous.set('fetch', Object.getOwnPropertyDescriptor(globalThis, 'fetch'));
  globalThis.fetch = async (url, options = {}) => {
    const path = String(url);
    if (options.method === 'POST') {
      const body = JSON.parse(options.body);
      requests.push({ path, body });
      if (path.endsWith('/lgu-plantings')) {
        if (failPlanting) return { ok: false, json: async () => ({ detail: 'Replanting work changed. Reload before recording planting.' }) };
        rows = rows.map((row) => body.planting_event_ids.includes(row.planting_event_id)
          ? { ...row, replanting_status: 'completed', can_record_replanting: false } : row);
        return { ok: true, json: async () => ({ replanted_points: body.planting_event_ids.length }) };
      }
      rows = rows.map((row) => body.planting_event_ids.includes(row.planting_event_id)
        ? { ...row, replanting_status: 'ready_lgu', can_approve: false, can_record_replanting: true, version: 1 } : row);
      return { ok: true, json: async () => ({ approved_count: body.planting_event_ids.length }) };
    }
    const payload = path.endsWith('/replanting') ? { points: structuredClone(rows) }
      : path.endsWith('/project-sites') ? { features: [{ id: 1, geometry: { type: 'Polygon', coordinates: [] }, properties: { name: 'Coastal site' } }] }
        : path.endsWith('/map-points') ? [] : { features: [] };
    return { ok: true, json: async () => payload };
  };
  root = createRoot(document.getElementById('root'));
});
afterEach(async () => {
  await act(async () => root.unmount());
  dom.window.close();
  for (const [key, descriptor] of previous) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
  previous.clear();
});

const settle = () => new Promise((resolve) => setTimeout(resolve, 30));
async function mount() {
  await act(async () => root.render(createElement(MemoryRouter, null, createElement(Page))));
  await act(settle);
}
async function click(text) {
  const button = [...document.querySelectorAll('button')].find((item) => item.textContent.trim() === text);
  assert.ok(button, text);
  await act(async () => { button.click(); await settle(); });
}

test('LGU queue records selected replacement IDs and versions and refreshes its progress', async () => {
  await mount();
  assert.match(document.body.textContent, /Ready for LGU replanting/);
  assert.match(document.body.textContent, /Assigned — LGU replanting/);
  const checkbox = document.querySelector('input[type="checkbox"]');
  await act(async () => checkbox.click());
  await click('Record replacement planting');
  await click('Record planted replacements');
  assert.equal(requests.length, 1);
  assert.ok(requests[0].path.endsWith('/replanting/lgu-plantings'));
  assert.deepEqual(requests[0].body.planting_event_ids, [2]);
  assert.deepEqual(requests[0].body.versions, { 2: 4 });
  assert.equal(requests[0].body.project_site_id, 1);
  assert.equal(requests[0].body.species, 'Bungalon');
  assert.match(document.body.textContent, /1 replacement seedling recorded/);
  assert.equal(document.querySelector('[role="dialog"]'), null);
});

test('approval sends only reviewed dead locations into the LGU queue', async () => {
  await mount();
  await click('Awaiting review');
  await act(async () => document.querySelector('input[type="checkbox"]').click());
  await click('Review for LGU replanting');
  assert.match(document.querySelector('[role="dialog"]').textContent, /LGU replanting queue/);
  await click('Approve 1 selected');
  assert.deepEqual(requests[0].body, { planting_event_ids: [1], versions: { 1: 0 } });
  assert.match(document.body.textContent, /1 locations assigned to the LGU replanting queue/);
});

test('stale work stays in the form with an actionable error and no success message', async () => {
  failPlanting = true;
  await mount();
  await act(async () => document.querySelector('input[type="checkbox"]').click());
  await click('Record replacement planting');
  await click('Record planted replacements');
  assert.match(document.querySelector('[role="alert"]').textContent, /work changed/);
  assert.ok(document.querySelector('[role="dialog"]'));
  assert.doesNotMatch(document.body.textContent, /replacement seedling recorded/);
});

test('progress separates paused and removed locations and map statuses retain erosion priority', () => {
  const paused = { ...initialRows[1], planting_event_id: 4, can_record_replanting: false, unavailable_reason: 'Erosion' };
  const removed = { ...initialRows[0], planting_event_id: 5, point_deleted_at: '2026-10-02' };
  assert.deepEqual(replantingProgress([...initialRows, paused, removed]), { awaiting_review: 1, ready_lgu: 1, replacements_completed: 1 });
  assert.equal(replantingRows(initialRows, { stage: 'awaiting_review', search: 'Photo A' }).length, 1);
  assert.equal(getMapPointStatus({ planting_status: 'planned', lgu_replanting_pending: true }), 'assigned');
  assert.equal(getMapPointStatus({ planting_status: 'planned', lgu_replanting_pending: true, eroded_unavailable: true }), 'unavailable');
});
