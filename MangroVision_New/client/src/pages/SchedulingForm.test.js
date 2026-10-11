import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
import { scheduleContactFields } from '../utils/scheduleContact.js';

let server, dom, root, createRoot, Scheduling, useAuthStore;
let schedules, writes;
const globals = new Map();
const date = '2035-06-12';

before(async () => {
  dom = new JSDOM('<div id="root"></div>', { pretendToBeVisual: true, url: 'http://localhost' });
  for (const [key, value] of Object.entries({ window: dom.window, document: dom.window.document,
    navigator: dom.window.navigator, HTMLElement: dom.window.HTMLElement, Element: dom.window.Element,
    SVGElement: dom.window.SVGElement, localStorage: dom.window.localStorage, sessionStorage: dom.window.sessionStorage,
    IS_REACT_ACT_ENVIRONMENT: true, requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window), fetch: globalThis.fetch,
  })) {
    globals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  ({ createRoot } = await import('react-dom/client'));
  server = await createServer({ configFile: false, appType: 'custom', server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    cacheDir: 'node_modules/.vite-manual-scheduling-tests', optimizeDeps: { noDiscovery: true, include: [] } });
  ({ default: Scheduling } = await server.ssrLoadModule('/src/pages/Scheduling.jsx'));
  ({ useAuthStore } = await server.ssrLoadModule('/src/stores/authStore.js'));
});
after(async () => {
  await server?.close(); dom?.window.close();
  for (const [key, descriptor] of globals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
  }
});
beforeEach(() => {
  root = createRoot(document.getElementById('root'));
  schedules = []; writes = [];
  useAuthStore.setState({ token: 'test-cookie', isAuthenticated: true });
  globalThis.fetch = async (url, options = {}) => {
    if (options.body) {
      writes.push({ url, method: options.method, body: JSON.parse(options.body) });
      return Response.json({ id: 1 });
    }
    if (String(url).includes('/planting-schedules')) return Response.json({ schedules, organizations: [], project_sites: [] });
    if (String(url).includes('/like-appointments')) return Response.json({ requests: [] });
    if (String(url).includes('/tides/forecast')) return Response.json({ available: false, message: 'Forecast unavailable.' });
    assert.fail(`Unexpected request: ${url}`);
  };
});
afterEach(async () => { await act(async () => root.unmount()); });

async function render() {
  await act(async () => root.render(createElement(MemoryRouter, { future: { v7_startTransition: true, v7_relativeSplatPath: true } }, createElement(Scheduling))));
}
async function click(label) {
  const button = [...document.querySelectorAll('button')].find((control) => control.textContent === label);
  assert.ok(button, `Missing ${label} button`);
  await act(async () => button.click());
}
async function change(name, value) {
  const control = document.querySelector(`[name="${name}"]`);
  assert.ok(control, `Missing ${name} control`);
  await act(async () => {
    const prototype = control.tagName === 'SELECT' ? window.HTMLSelectElement.prototype
      : control.tagName === 'TEXTAREA' ? window.HTMLTextAreaElement.prototype : window.HTMLInputElement.prototype;
    Object.getOwnPropertyDescriptor(prototype, 'value').set.call(control, value);
    control.dispatchEvent(new window.Event(control.tagName === 'SELECT' ? 'change' : 'input', { bubbles: true }));
  });
}
async function submit() {
  await act(async () => document.querySelector('.schedule-form').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true })));
}
async function fill(type) {
  for (const [name, value] of Object.entries({ appointmentType: type, organizationName: 'Coastal volunteers',
    contactName: 'Test coordinator', phone: '09123456789', email: 'head@example.org', date,
    startTime: '08:00', endTime: '09:30', expectedParticipants: '10', notes: 'Meet at the gate.' })) await change(name, value);
}

for (const [type, label] of [['field_visit', 'Field visit'], ['clean_up_drive', 'Clean-up drive'], ['tree_planting', 'Tree planting']]) {
  test(`manual ${type} requires website details and keeps its activity choice`, async () => {
    await render(); await click('+ Add new schedule');
    assert.equal(document.querySelector('.modal-backdrop').parentElement, document.body);
    assert.equal(document.querySelector('[name="title"]'), null);
    assert.deepEqual([...document.querySelector('[name="appointmentType"]').options].map((option) => option.textContent),
      ['Choose an appointment type', 'Field visit', 'Clean-up drive', 'Tree planting']);
    for (const name of ['organizationName', 'contactName', 'phone', 'email', 'date', 'startTime', 'endTime', 'expectedParticipants']) {
      assert.equal(document.querySelector(`[name="${name}"]`).required, true, `${name} must be required`);
    }
    await fill(type);
    assert.equal(Boolean(document.querySelector('[name="seedlings"]')), type === 'tree_planting');
    if (type === 'tree_planting') await change('seedlings', '100');
    await submit();
    assert.equal(writes.length, 1);
    const body = writes[0].body;
    assert.equal(body.appointment_type, type);
    assert.equal(body.title, undefined, `${label} is named by the backend`);
    assert.equal(body.contact_name, 'Test coordinator');
    assert.equal(body.phone, '09123456789'); assert.equal(body.email, 'head@example.org');
    assert.equal(body.expected_planters, 10);
    assert.equal(body.start_at, `${date}T08:00:00+08:00`); assert.equal(body.end_at, `${date}T09:30:00+08:00`);
    assert.equal(body.expected_seedlings, type === 'tree_planting' ? 100 : null);
    assert.equal(body.notes, 'Meet at the gate.');
  });
}

test('manual errors stop missing, invalid and past booking details before saving', async () => {
  await render(); await click('+ Add new schedule'); await submit();
  assert.equal(writes.length, 0);
  await fill('field_visit');
  for (const [name, bad, good] of [['phone', 'abc1234567', '09123456789'], ['email', 'broken', 'head@example.org'],
    ['expectedParticipants', '0', '10'], ['expectedParticipants', '10001', '10'], ['expectedParticipants', '1.5', '10'],
    ['endTime', '07:00', '09:30'], ['date', '2020-01-01', date]]) {
    await change(name, bad); await submit(); assert.equal(writes.length, 0, `${name} must block saving`);
    assert.equal(document.querySelector(`[name="${name}"]`).getAttribute('aria-invalid'), 'true');
    await change(name, good);
  }
  await submit(); assert.equal(writes.length, 1);
});

test('editing a legacy schedule keeps its title and untouched contact summary', async () => {
  const contact = 'Head: old contact details and instructions. '.repeat(4).trim();
  schedules = [{ id: 7, organization: 'Old school', title: 'Historical community event', contact,
    start_at: `${date}T08:00:00+08:00`, end_at: `${date}T09:30:00+08:00`, status: 'requested', expected_planters: null }];
  await render(); await click('List'); await click('Edit');
  assert.equal(document.querySelector('[name="appointmentType"]').disabled, true);
  assert.equal(document.querySelector('[name="title"]'), null);
  await change('notes', 'Corrected staff note.'); await submit();
  assert.equal(writes.length, 1);
  assert.equal(writes[0].method, 'PUT');
  assert.equal(writes[0].body.title, undefined); assert.equal(writes[0].body.appointment_type, undefined);
  assert.equal(writes[0].body.contact, contact);
  assert.equal(writes[0].body.expected_planters, null);
});

test('website schedule contacts remain read-only and their original summary is preserved', async () => {
  const contact = 'Coordinator, 09123456789, head@example.org';
  schedules = [{ id: 8, source: 'website', appointment_type: 'field_visit', organization: 'School', title: 'Original visit name',
    contact, start_at: `${date}T08:00:00+08:00`, end_at: `${date}T09:30:00+08:00`, status: 'confirmed', expected_planters: 10 }];
  await render(); await click('List'); await click('Edit');
  for (const [name, value] of Object.entries({ contactName: 'Coordinator', phone: '09123456789', email: 'head@example.org' })) {
    const control = document.querySelector(`[name="${name}"]`);
    assert.equal(control.readOnly, true); assert.equal(control.value, value);
  }
  await change('notes', 'Updated staff note.'); await submit();
  assert.equal(writes.length, 1); assert.equal(writes[0].body.contact, contact);
  assert.equal(writes[0].body.notification_email, undefined);
});

test('contact summaries support comma-separated names, embedded emails and a separate recipient', () => {
  assert.deepEqual(scheduleContactFields({ contact: 'Doe, Jane, +63 912 345 6789, old@example.org', notification_email: 'new@example.org' }),
    { contactName: 'Doe, Jane', phone: '+63 912 345 6789', email: 'new@example.org' });
});
