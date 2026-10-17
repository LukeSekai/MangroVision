import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement } from 'react';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
import { MemoryRouter } from 'react-router-dom';
import { STAFF_TOURS } from '../guides/tours.js';
import { prepareGuideStep } from '../guides/tourRuntime.js';

let server, dom, root, createRoot, Organizations;
const globals = new Map();
before(async () => {
  dom = new JSDOM('<div id="root"></div>', { pretendToBeVisual: true, url: 'http://localhost' });
  for (const [key, value] of Object.entries({ window: dom.window, document: dom.window.document,
    navigator: dom.window.navigator, HTMLElement: dom.window.HTMLElement, Element: dom.window.Element,
    IS_REACT_ACT_ENVIRONMENT: true, requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window), fetch: globalThis.fetch,
  })) {
    globals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  ({ createRoot } = await import('react-dom/client'));
  server = await createServer({ configFile: false, appType: 'custom', server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    cacheDir: 'node_modules/.vite-organization-directory-tests', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ default: Organizations } = await server.ssrLoadModule('/src/pages/Organizations.jsx'));
});
after(async () => {
  await server?.close(); dom?.window.close();
  for (const [key, descriptor] of globals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
  }
});

let calls, members;
const organizations = [
  { id: 1, name: 'Example organization', contact: 'Test coordinator, 09123456789', notification_email: 'contact@example.org', scheduled_participants: 20, member_count: 2, project_site_count: 1, inspection_interval_days: 14, contact_activity_title: 'Community planting' },
  { id: 2, name: 'Second organization', member_count: 0, project_site_count: 0, inspection_interval_days: 14 },
];
function detail(id) {
  return { organization: organizations[id - 1],
    project_sites: id === 1 ? [{ id: 12, name: 'Current planting site' }, { id: 13, name: 'Old planting site', deleted_at: '2026-01-01' }] : [],
    activities: id === 1 ? [
      { id: 10, title: 'Community planting', status: 'confirmed', appointment_type: 'tree_planting', expected_planters: 20, has_code: true, start_at: '2026-11-01T00:00:00Z', end_at: '2026-11-01T02:00:00Z' },
      { id: 11, title: 'Unissued planting', status: 'requested', appointment_type: 'tree_planting', has_code: false },
      { id: 12, title: 'Revoked planting', status: 'cancelled', appointment_type: 'tree_planting', has_code: true, code_revoked: true },
    ] : [], members: id === 1 ? members : [] };
}
beforeEach(() => {
  calls = [];
  members = [{ planter_id: 1, participant_number: 1, username: 'Ana', first_joined_at: '2026-01-01', last_seen_at: '2026-01-02' }, { planter_id: 1, participant_number: 2, username: null }];
  root = createRoot(document.getElementById('root'));
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    assert.ok(!options.method || options.method === 'GET', 'Directory must never issue a code or write a record');
    if (url === '/api/organizations') return Response.json({ organizations });
    if (url === '/api/organizations/1') return Response.json(detail(1));
    if (url === '/api/organizations/2') return Response.json(detail(2));
    if (url === '/api/planters/organizations/1/activities/10/access') return Response.json({ access: { code: 'DEMO-CODE' } });
    assert.fail(`Unexpected request: ${url}`);
  };
});
afterEach(async () => { await act(async () => root.unmount()); });
const settle = async () => act(async () => { await new Promise((resolve) => setTimeout(resolve, 30)); });
const button = (text) => [...document.querySelectorAll('button')].find((item) => item.textContent === text);
const tab = (text) => [...document.querySelectorAll('[role="tab"]')].find((item) => item.textContent.startsWith(text));
async function click(item) { assert.ok(item); await act(async () => item.click()); await settle(); }
async function render() {
  await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ['/organizations'] }, createElement(Organizations))));
  await settle();
}
async function change(control, value) {
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(control, value);
    control.dispatchEvent(new window.Event('input', { bubbles: true }));
  });
}

function mockDirectory(getItems) {
  globalThis.fetch = async (url, options) => {
    calls.push({ url, options });
    if (url === '/api/organizations') return Response.json({ organizations: getItems() });
    const organization = getItems().find((item) => url === `/api/organizations/${item.id}`);
    assert.ok(organization, `Unexpected directory request: ${url}`);
    return Response.json({ organization, project_sites: [], activities: [], members: [] });
  };
}

test('organization finder shows four per page, handles the final page, and searches the whole directory', async () => {
  const items = Array.from({ length: 17 }, (_, index) => ({ ...organizations[0], id: index + 1, name: `Group-${String(index + 1).padStart(2, '0')}` }));
  mockDirectory(() => items);
  await render();
  const pager = () => document.querySelector('[aria-label="Organization pages"]');
  const rows = () => [...document.querySelectorAll('.organization-list-item strong')].map((item) => item.textContent);
  assert.deepEqual(rows(), ['Group-01', 'Group-02', 'Group-03', 'Group-04']);
  assert.equal(pager().querySelector('button').disabled, true);
  assert.match(document.querySelector('.organization-result-count').textContent, /Showing 1–4 of 17/);
  await click([...pager().querySelectorAll('button')][1]);
  assert.deepEqual(rows(), ['Group-05', 'Group-06', 'Group-07', 'Group-08']);
  assert.equal(document.querySelector('.organization-detail h2').textContent, 'Group-01');
  await click([...pager().querySelectorAll('button')][1]);
  await click([...pager().querySelectorAll('button')][1]);
  await click([...pager().querySelectorAll('button')][1]);
  assert.deepEqual(rows(), ['Group-17']);
  assert.equal([...pager().querySelectorAll('button')][1].disabled, true);
  assert.match(pager().textContent, /Page 5 of 5/);
  await click(pager().querySelector('button'));
  assert.equal(rows()[0], 'Group-13');
  await change(document.querySelector('.organization-list input'), 'Group-17');
  assert.deepEqual(rows(), ['Group-17']);
  assert.match(document.querySelector('.organization-result-count').textContent, /Showing 1–1 of 1/);
  assert.equal(pager(), null);
  assert.equal(document.querySelector('.organization-totals strong').textContent, '17');
  await change(document.querySelector('.organization-list input'), '');
  assert.equal(rows()[0], 'Group-01');
  assert.match(pager().textContent, /Page 1 of 5/);
});

test('workspace updates clamp the finder page and restore a valid profile when organizations disappear', async () => {
  let items = Array.from({ length: 12 }, (_, index) => ({ ...organizations[0], id: index + 1, name: `Group-${index + 1}` }));
  mockDirectory(() => items);
  await render();
  const pager = () => document.querySelector('[aria-label="Organization pages"]');
  await click([...pager().querySelectorAll('button')][1]);
  await click([...pager().querySelectorAll('button')][1]);
  await click(document.querySelector('.organization-list-item'));
  assert.equal(document.querySelector('.organization-detail h2').textContent, 'Group-9');
  items = items.slice(0, 3);
  await act(async () => window.dispatchEvent(new window.Event('mv:data-changed'))); await settle();
  assert.equal(document.querySelectorAll('.organization-list-item').length, 3);
  assert.match(document.querySelector('.organization-result-count').textContent, /Showing 1–3 of 3/);
  assert.equal(pager(), null);
  assert.equal(document.querySelector('.organization-detail h2').textContent, 'Group-1');
});

test('directory shows saved contact and distinct counts, with map links only for current sites', async () => {
  await render();
  assert.equal(document.querySelector('h1').textContent, 'Organizations');
  assert.equal(button('Refresh information'), undefined);
  const profile = document.querySelector('.organization-detail');
  assert.match(profile.textContent, /Test coordinator/);
  assert.equal(profile.querySelector('a[href^="mailto:"]').textContent, 'contact@example.org');
  assert.equal(profile.querySelector('a[href^="tel:"]').getAttribute('href'), 'tel:09123456789');
  assert.match(profile.textContent, /Expected participants20/);
  assert.match(profile.textContent, /Registered members2/);
  assert.match(profile.textContent, /Community planting/);
  assert.equal(profile.querySelectorAll('a[href^="/map?"]').length, 1);
  assert.equal(profile.querySelector('a[href^="/map?"]').getAttribute('href'), '/map?project_site_id=12&focus=site_points');
  assert.match(profile.textContent, /Archived project site/);
  assert.ok(calls.every(({ options }) => options.cache === 'no-store'));
});

test('organization guide finds the real controls without revealing codes or changing the profile tab', async () => {
  await render();
  Object.defineProperty(window.HTMLElement.prototype, 'getClientRects', {
    configurable: true, value() { return this.isConnected ? [{ width: 100, height: 40 }] : []; },
  });
  const readsBefore = calls.length;
  const selectedBefore = document.querySelector('[role="tab"][aria-selected="true"]');
  const tour = STAFF_TOURS.find((item) => item.id === 'organizations');
  for (const step of tour.steps) {
    const result = await prepareGuideStep(step, { getPath: () => '/organizations',
      navigate: () => assert.fail('Already on the organization page'),
      signal: new AbortController().signal, timeout: 50 });
    assert.equal(result.unavailable, false, `Missing organization guide control: ${step.target}`);
  }
  assert.equal(calls.length, readsBefore);
  assert.equal(document.querySelector('[role="tab"][aria-selected="true"]'), selectedBefore);
  assert.equal(document.querySelector('.organization-code code'), null);
});

test('organization search and member search retain honest missing-name and empty states', async () => {
  await render();
  await change(document.querySelector('.organization-list input'), 'contact@example.org');
  assert.equal(document.querySelectorAll('.organization-list-item').length, 1);
  await click(tab('Members'));
  assert.match(document.querySelector('[role="tabpanel"]').textContent, /Ana/);
  assert.match(document.querySelector('[role="tabpanel"]').textContent, /Name not recorded/);
  await change(document.querySelector('[role="tabpanel"] input'), 'absent');
  assert.match(document.querySelector('[role="tabpanel"]').textContent, /No members match/);
  await change(document.querySelector('.organization-list input'), 'second');
  await click(document.querySelector('.organization-list-item'));
  assert.equal(document.querySelector('.organization-detail h2').textContent, 'Second organization');
  assert.match(document.querySelector('.organization-detail').textContent, /no project sites yet/);
});

test('codes are fetched only on explicit reveal, using existing read-only access, and can be hidden', async () => {
  await render();
  await click(tab('Activity codes'));
  assert.equal(calls.filter(({ url }) => url.includes('/access')).length, 0);
  assert.equal([...document.querySelectorAll('button')].filter((item) => item.textContent === 'Show code').length, 1);
  assert.match(document.body.textContent, /Code not issued/);
  assert.match(document.body.textContent, /Code revoked/);
  await click(button('Show code'));
  assert.equal(document.querySelector('.organization-code code').textContent, 'DEMO-CODE');
  assert.equal(calls.filter(({ url }) => url.includes('/access')).length, 1);
  await click(button('Hide code'));
  assert.equal(document.querySelector('.organization-code code'), null);
  assert.equal(document.body.textContent.includes('DEMO-CODE'), false);
});

test('a late code response cannot appear under a different organization', async () => {
  const base = globalThis.fetch;
  let release, signal;
  globalThis.fetch = (url, options) => {
    if (url.endsWith('/access')) {
      signal = options.signal;
      return new Promise((resolve) => { release = () => resolve(Response.json({ access: { code: 'DEMO-LATE' } })); });
    }
    return base(url, options);
  };
  await render(); await click(tab('Activity codes'));
  await click(button('Show code'));
  await click([...document.querySelectorAll('.organization-list-item')][1]);
  assert.equal(signal.aborted, true);
  await act(async () => release()); await settle();
  assert.equal(document.querySelector('.organization-detail h2').textContent, 'Second organization');
  assert.equal(document.body.textContent.includes('DEMO-LATE'), false);
});

test('member pagination and searching show all members without creating placeholder people', async () => {
  members = Array.from({ length: 28 }, (_, index) => ({ planter_id: 1, participant_number: index + 1, username: `Member-${String(index + 1).padStart(2, '0')}` }));
  await render(); await click(tab('Members'));
  assert.equal(document.querySelectorAll('tbody tr').length, 25);
  await click(button('Next'));
  assert.equal(document.querySelectorAll('tbody tr').length, 3);
  await change(document.querySelector('[role="tabpanel"] input'), 'Member-01');
  assert.equal(document.querySelectorAll('tbody tr').length, 1);
  assert.equal(document.querySelector('tbody strong').textContent, 'Member-01');
});

test('a failed directory load can be retried and does not invent zero-valued records', async () => {
  const base = globalThis.fetch;
  let failed = false;
  globalThis.fetch = (url, options) => {
    if (url === '/api/organizations' && !failed) { failed = true; return Promise.resolve(Response.json({ detail: 'Connection unavailable' }, { status: 503 })); }
    return base(url, options);
  };
  await render();
  assert.match(document.querySelector('[role="alert"]').textContent, /Connection unavailable/);
  assert.ok([...document.querySelectorAll('.organization-totals strong')].every((item) => item.textContent === '—'));
  await click(button('Retry'));
  assert.equal(document.querySelector('[role="alert"]'), null);
  assert.equal(document.querySelector('.organization-detail h2').textContent, 'Example organization');
});

test('organization information tabs support arrow keys and keep one keyboard tab stop', async () => {
  await render();
  const sites = tab('Project sites');
  await act(async () => { sites.focus(); sites.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'End', bubbles: true })); });
  assert.equal(tab('Members').getAttribute('aria-selected'), 'true');
  assert.equal(document.activeElement, tab('Members'));
  assert.equal(document.querySelectorAll('[role="tab"][tabindex="0"]').length, 1);
  await act(async () => document.activeElement.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true })));
  assert.equal(document.activeElement, tab('Project sites'));
  assert.equal(document.querySelector('[role="tabpanel"]').getAttribute('aria-labelledby'), sites.id);
});
