import test, { after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { findRendered, prepareGuideStep } from './tourRuntime.js';

const dom = new JSDOM('<body></body>', { pretendToBeVisual: true });
const previous = new Map();
for (const [key, value] of Object.entries({ window: dom.window, document: dom.window.document })) {
  previous.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
  Object.defineProperty(globalThis, key, { value, configurable: true });
}
Object.defineProperty(window.HTMLElement.prototype, 'getClientRects', {
  value() { return this.isConnected ? [{ width: 100, height: 40 }] : []; },
});
let path;
beforeEach(() => {
  path = '/map';
  document.body.innerHTML = '<header class="workspace-header">Help</header>';
});
after(() => {
  dom.window.close();
  for (const [key, descriptor] of previous) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
});
const options = (extra = {}) => ({
  navigate: (next) => { path = next; }, getPath: () => path,
  signal: new AbortController().signal, timeout: 150, ...extra,
});

test('targets the active page when retained pages contain the same selector', () => {
  document.body.insertAdjacentHTML('beforeend', '<section style="display:none"><button class="target">Old page</button></section><section><button class="target">Active page</button></section>');
  assert.equal(findRendered('.target').textContent, 'Active page');
});

test('reveals a hidden panel and collapsed section without submitting any data', async () => {
  document.body.insertAdjacentHTML('beforeend', '<button class="panel-toggle">Show</button><div class="floating-panel floating-panel-hidden"><div class="panel-card"><button class="panel-card-header" aria-expanded="false">Section</button><div aria-hidden="true" inert><input id="target"><button id="submit">Save</button></div></div></div>');
  let submissions = 0;
  const panel = document.querySelector('.floating-panel');
  document.querySelector('.panel-toggle').onclick = () => panel.classList.remove('floating-panel-hidden');
  document.querySelector('.panel-card-header').onclick = (event) => {
    event.target.setAttribute('aria-expanded', 'true');
    document.querySelector('[inert]').removeAttribute('inert');
    document.querySelector('[aria-hidden]').setAttribute('aria-hidden', 'false');
  };
  document.querySelector('#submit').onclick = () => { submissions += 1; };
  const result = await prepareGuideStep({ path: '/map', target: '#target' }, options());
  assert.equal(result.element.id, 'target');
  assert.equal(result.unavailable, false);
  assert.equal(panel.classList.contains('floating-panel-hidden'), false);
  assert.equal(submissions, 0);
});

test('waits for a route to render and supports returning to an earlier screen', async () => {
  const navigations = [];
  const navigate = (next) => {
    navigations.push(next);
    path = next;
    document.querySelector('.target')?.remove();
    window.setTimeout(() => document.body.insertAdjacentHTML('beforeend', `<button class="target">${next}</button>`), 15);
  };
  let result = await prepareGuideStep({ path: '/processing', target: '.target' }, options({ navigate }));
  assert.equal(result.element.textContent, '/processing');
  result = await prepareGuideStep({ path: '/map', target: '.target' }, options({ navigate }));
  assert.equal(result.element.textContent, '/map');
  assert.deepEqual(navigations, ['/processing', '/map']);
});

test('anchors section tips to a visible heading even when its records fill the page', async () => {
  document.body.insertAdjacentHTML('beforeend', '<section id="records" class="panel-card"><button class="panel-card-header" aria-expanded="true">Records</button><div style="height:4000px">Many rows</div></section>');
  const result = await prepareGuideStep({ path: '/map', target: '#records' }, options());
  assert.equal(result.element.className, 'panel-card-header');
  assert.equal(result.unavailable, false);
});

test('preserves instructions with a fallback when prerequisite controls are absent', async () => {
  const result = await prepareGuideStep({ path: '/map', target: '#not-yet-available' }, options({ timeout: 20 }));
  assert.equal(result.element.className, 'workspace-header');
  assert.equal(result.unavailable, true);
});

test('ending the tour cancels pending section expansion', async () => {
  const controller = new AbortController();
  const pending = prepareGuideStep({ path: '/map', target: '#later' }, options({ signal: controller.signal, timeout: 1000 }));
  controller.abort();
  assert.equal(await pending, null);
  document.body.insertAdjacentHTML('beforeend', '<div class="panel-card"><button class="panel-card-header" aria-expanded="false">Section</button><input id="later"></div>');
  let expansions = 0;
  document.querySelector('.panel-card-header').onclick = () => { expansions += 1; };
  await new Promise((resolve) => window.setTimeout(resolve, 80));
  assert.equal(expansions, 0);
});

test('an already cancelled tour cannot navigate', async () => {
  const controller = new AbortController();
  controller.abort();
  let navigations = 0;
  const result = await prepareGuideStep({ path: '/processing', target: '.target' }, options({
    signal: controller.signal, navigate: () => { navigations += 1; },
  }));
  assert.equal(result, null);
  assert.equal(navigations, 0);
});
