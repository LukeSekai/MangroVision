import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

let server, Page, root, dom, createRoot, data, requests, conflict;
const previous = new Map();
const site = (id, total, dead, survival, due) => ({ project_site_id: id, site_name: `LGU site ${id}`,
  total_planted: total, alive_count: total-dead, dead_count: dead, survival_rate_pct: survival,
  next_monitoring_date: due, monitoring_available: due < '2030', version: 'a'.repeat(64),
  points: [{ planting_event_id: id*100, reference: `Seedling ${id*100}` }], history: [] });

before(async () => {
  server = await createServer({ configFile: false, appType: 'custom',
    server: { middlewareMode: true, hmr: false, watch: null, ws: false }, optimizeDeps: { noDiscovery: true, include: [] },
    plugins: [{ name: 'lgu-test-map', load(id) {
      if (id.replaceAll('\\', '/').endsWith('/components/SeedlingLocations.jsx')) return `
        import { createElement } from 'react';
        export default function Map({providedPoints, selected, onChange}) {
          return createElement('div', null, providedPoints.map(p => createElement('label', {key:p.planting_event_id},
            createElement('input', {type:'checkbox', checked:selected.includes(p.planting_event_id),
              onChange:e=>onChange(e.target.checked?[...selected,p.planting_event_id]:selected.filter(id=>id!==p.planting_event_id))}), p.reference)));
        }`;
    } }],
  });
  Page = (await server.ssrLoadModule('/src/pages/LGUSites.jsx')).default;
});
after(async () => { await server?.close(); });
beforeEach(async () => {
  dom = new JSDOM('<body><div id="root"></div></body>', { url: 'http://localhost/' });
  for (const [key, value] of Object.entries({ window: dom.window, document: dom.window.document,
    navigator: dom.window.navigator, HTMLElement: dom.window.HTMLElement, localStorage: dom.window.localStorage,
    Event: dom.window.Event, IS_REACT_ACT_ENVIRONMENT: true })) {
    previous.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  ({ createRoot } = await import('react-dom/client'));
  data = { account: { organization_id: 99, username: 'lgu', access_code: 'TEST-CODE' },
    summary: { site_count: 4, total_planted: 500, alive_count: 425, dead_count: 75, survival_rate_pct: 85 },
    sites: [site(1,100,20,80,'2026-01-01'), site(2,200,20,90,'2026-02-01'), site(3,50,20,60,'2030-01-01'), site(4,150,15,90,'2030-02-01')] };
  requests = []; conflict = false;
  previous.set('fetch', Object.getOwnPropertyDescriptor(globalThis, 'fetch'));
  globalThis.fetch = async (url, options = {}) => {
    const path = String(url);
    requests.push({ path, body: options.body ? JSON.parse(options.body) : null });
    if (options.method === 'POST') {
      if (conflict) return Response.json({ detail: 'This site changed. Reload its monitoring form before saving.' }, { status: 409 });
      data.sites[0] = { ...data.sites[0], dead_count: 21, alive_count: 79, survival_rate_pct: 79, next_monitoring_date: '2030-01-01', monitoring_available: false };
      return Response.json({ id: 1 });
    }
    return Response.json(path.endsWith('/lgu-sites') ? data : data.sites[Number(path.match(/lgu-sites\/(\d+)/)?.[1])-1]);
  };
  const store = (await server.ssrLoadModule('/src/stores/mapStore.js')).useMapStore;
  store.setState({ fetchPoints: async () => {}, fetchStats: async () => {}, fetchZones: async () => {} });
  root = createRoot(document.getElementById('root'));
});
afterEach(async () => {
  await act(async () => root.unmount());
  dom.window.close();
  for (const [key, descriptor] of previous) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
  }
  previous.clear();
});
const settle = () => new Promise((resolve) => setTimeout(resolve, 30));
async function mount() {
  await act(async () => root.render(createElement(MemoryRouter, null, createElement(Page))));
  await act(settle);
}
async function click(text, scope = document) {
  const button = [...scope.querySelectorAll('button')].find((item) => item.textContent.trim() === text);
  assert.ok(button, text);
  await act(async () => { button.click(); await settle(); });
}
async function fillActions() {
  const input = document.querySelector('.lgu-site-form textarea[required]');
  await act(async () => {
    Object.getOwnPropertyDescriptor(dom.window.HTMLTextAreaElement.prototype, 'value').set.call(input, 'Checked seedlings and removed debris.');
    input.dispatchEvent(new window.Event('input', { bubbles: true }));
  });
}

test('four separate sites show weighted LGU totals and masked permanent credentials', async () => {
  await mount();
  assert.equal(document.querySelectorAll('.lgu-site-card').length, 4);
  assert.match(document.querySelector('[aria-label="Combined LGU results"]').textContent, /85%/);
  assert.match(document.body.textContent, /Permanent/);
  assert.doesNotMatch(document.body.textContent, /TEST-CODE/);
  await click('Show code');
  assert.match(document.body.textContent, /TEST-CODE/);
  assert.ok(document.querySelector('a[href="/field"]'));
  const upcoming = document.querySelectorAll('.lgu-site-card')[2];
  assert.equal([...upcoming.querySelectorAll('button')].find((button) => button.textContent === 'Record monitoring').disabled, true);
});

test('LGU Sites opens assignment directly without a manual creation or site-selection link', async () => {
  await mount();
  const link = document.querySelector('a[href="/planters?section=assign&organization_id=99"]');
  assert.ok(link);
  assert.equal(link.textContent,'Assign LGU points');
  assert.equal(document.querySelector('a[href^="/zones?"]'),null);
  assert.equal(document.querySelector('a[href*="site_id="]'),null);
});

test('a visit sends only the chosen site and updates its card without changing another site', async () => {
  await mount();
  const otherBefore = document.querySelectorAll('.lgu-site-card')[1].textContent;
  await click('Record monitoring', document.querySelector('.lgu-site-card'));
  await act(settle);
  await act(async () => document.querySelector('[role="dialog"] input[type="checkbox"]').click());
  await fillActions();
  await click('Save site monitoring', document.querySelector('[role="dialog"]'));
  const write = requests.find((request) => request.body);
  assert.equal(write.path, '/api/monitoring/lgu-sites/1/monitoring');
  assert.deepEqual(write.body.dead_planting_event_ids, [100]);
  assert.equal(write.body.version, 'a'.repeat(64));
  assert.equal(write.body.inspector_user_id, undefined);
  assert.equal(document.querySelector('[role="dialog"]'), null);
  assert.equal(document.querySelectorAll('.lgu-site-card')[1].textContent, otherBefore);
});

test('a stale visit keeps entered actions and selected deaths for review', async () => {
  conflict = true;
  await mount();
  await click('Record monitoring', document.querySelector('.lgu-site-card'));
  await act(settle);
  await act(async () => document.querySelector('[role="dialog"] input[type="checkbox"]').click());
  await fillActions();
  await click('Save site monitoring', document.querySelector('[role="dialog"]'));
  assert.match(document.querySelector('[role="alert"]').textContent, /Reload/);
  assert.equal(document.querySelector('.lgu-site-form textarea[required]').value, 'Checked seedlings and removed debris.');
  assert.equal(document.querySelector('[role="dialog"] input[type="checkbox"]').checked, true);
});
