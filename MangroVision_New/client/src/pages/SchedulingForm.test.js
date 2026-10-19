import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement } from 'react';
import { MemoryRouter } from 'react-router-dom';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
import { scheduleContactFields } from '../utils/scheduleContact.js';
import { manilaToday } from '../like/booking.js';

let server, dom, root, createRoot, Scheduling, useAuthStore;
let schedules, writes, requests, onWrite, tidePayload, forecastReads;
const globals = new Map();
const date = '2035-06-12';

before(async () => {
  dom = new JSDOM('<div id="root"></div>', { pretendToBeVisual: true, url: 'http://localhost' });
  for (const [key, value] of Object.entries({ window: dom.window, document: dom.window.document,
    navigator: dom.window.navigator, HTMLElement: dom.window.HTMLElement, Element: dom.window.Element,
    SVGElement: dom.window.SVGElement, Event: dom.window.Event, localStorage: dom.window.localStorage, sessionStorage: dom.window.sessionStorage,
    ResizeObserver: class {
      constructor(callback) { this.callback = callback; }
      observe() { this.callback([{ contentRect: { width: 1000, height: 280 } }]); }
      disconnect() {}
    },
    IS_REACT_ACT_ENVIRONMENT: true, requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
    cancelAnimationFrame: dom.window.cancelAnimationFrame.bind(dom.window), fetch: globalThis.fetch,
  })) {
    globals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  ({ createRoot } = await import('react-dom/client'));
  window.HTMLElement.prototype.scrollTo = () => {};
  server = await createServer({ configFile: false, appType: 'custom', server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    cacheDir: 'node_modules/.vite-manual-scheduling-tests', optimizeDeps: { noDiscovery: true, include: [] },
  });
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
  schedules = []; writes = []; requests = []; onWrite = null;
  tidePayload = { available: false, message: 'Forecast unavailable.' }; forecastReads = 0;
  useAuthStore.setState({ token: 'test-cookie', isAuthenticated: true });
  globalThis.fetch = async (url, options = {}) => {
    if (options.body) {
      writes.push({ url, method: options.method, body: JSON.parse(options.body) });
      onWrite?.(writes.at(-1));
      return Response.json({ id: 1 });
    }
    if (String(url).includes('/planting-schedules')) return Response.json({ schedules, organizations: [], project_sites: [] });
    if (String(url).includes('/like-appointments')) return Response.json({ requests });
    if (String(url).includes('/tides/forecast')) { forecastReads += 1; return Response.json(tidePayload); }
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

test('weekly overview stays visible while calendar tide entries open the correct hourly graph without refetching', async () => {
  const today = manilaToday();
  const start = Date.parse(`${today}T00:00:00+08:00`);
  const tomorrow = manilaToday(new Date(start + 86400000));
  tidePayload = { available: true, stale: false, series_available: true, fetched_at: Date.now() / 1000,
    datum: 'MSL', datum_reference: 'test',
    heights: Array.from({ length: 49 }, (_, hour) => ({ timestamp: start / 1000 + hour * 3600,
      height_m: hour === 2 || hour === 29 ? 4 : 0 })),
    extremes: [2, 14, 29, 42].map((hour) => ({ timestamp: start / 1000 + hour * 3600,
      type: hour === 2 || hour === 29 ? 'High' : 'Low', height_m: hour === 2 || hour === 29 ? 4 : 0 })),
  };
  await render();
  const weekly = document.querySelector('.schedule-tide-panel .schedule-tide-chart');
  assert.ok(weekly.querySelector('.recharts-line-curve')?.getAttribute('d'));
  assert.equal(document.querySelector('.schedule-day-picker'), null);
  assert.equal(document.querySelector('.schedule-tide-times'), null);
  assert.equal(document.querySelector('.schedule-planting-times'), null);
  const tideButtons = [...document.querySelectorAll('.schedule-calendar-tide')];
  assert.equal(tideButtons.length, 4);
  assert.equal(forecastReads, 1);
  const high = tideButtons.find((button) => button.querySelector('time').dateTime === new Date(start + 2 * 3600000).toISOString());
  assert.match(high.querySelector('strong').textContent, /High tide/);
  await act(async () => { high.focus(); high.click(); });
  let modal = document.querySelector('.schedule-details-modal.is-tide');
  assert.match(modal.querySelector('.modal-title').textContent, /High tide/);
  assert.equal(modal.querySelector('.schedule-detail-when strong').textContent, '2:00 AM');
  assert.match(modal.querySelector('.schedule-tide-day-chart').getAttribute('aria-label'), new RegExp(`hourly view for ${today}`));
  assert.ok(modal.querySelector('.recharts-line-curve')?.getAttribute('d'));
  assert.match(modal.querySelector('.schedule-tide-day-chart').textContent, /Time \(Philippine time\)/);
  assert.match(modal.querySelector('.schedule-tide-day-chart').textContent, /High tide/);
  assert.match(modal.querySelector('.schedule-tide-day-chart').textContent, /Low tide/);
  assert.equal(forecastReads, 1);
  await click('Close');
  assert.equal(document.querySelector('.schedule-details-modal'), null);
  assert.ok(document.activeElement === high, 'closing returns focus to the tide entry');
  assert.ok(document.querySelector('.schedule-tide-panel .schedule-tide-chart') === weekly, 'the weekly graph stays mounted');
  const low = tideButtons.find((button) => button.querySelector('time').dateTime === new Date(start + 42 * 3600000).toISOString());
  assert.match(low.querySelector('strong').textContent, /Low tide/);
  await act(async () => { low.focus(); low.click(); });
  modal = document.querySelector('.schedule-details-modal.is-tide');
  assert.match(modal.querySelector('.modal-title').textContent, /Low tide/);
  assert.equal(modal.querySelector('.schedule-detail-when strong').textContent, '6:00 PM');
  assert.match(modal.querySelector('.schedule-tide-day-chart').getAttribute('aria-label'), new RegExp(`hourly view for ${tomorrow}`));
  assert.equal(forecastReads, 1);
  await click('Close');
  assert.ok(document.activeElement === low, 'closing returns focus to the second tide entry');
  tidePayload = { ...tidePayload, stale: true };
  await click('Refresh tides');
  assert.equal(forecastReads, 2);
  assert.match(document.querySelector('.schedule-tide-panel .schedule-message').textContent, /out of date/);
  assert.ok(document.querySelector('.schedule-tide-panel .recharts-line-curve'));
  assert.deepEqual(writes, []);
});

for (const [type, label] of [['field_visit', 'Field visit'], ['clean_up_drive', 'Clean-up drive'], ['tree_planting', 'Tree planting']]) {
  test(`manual ${type} requires website details and keeps its activity choice`, async () => {
    await render(); await click('+ Add schedule');
    assert.equal(document.querySelector('.modal-backdrop').parentElement, document.body);
    assert.equal(document.querySelector('[name="title"]'), null);
    assert.deepEqual([...document.querySelector('[name="appointmentType"]').options].map((option) => option.textContent),
      ['Choose an appointment type', 'Field visit', 'Clean-up drive', 'Tree planting']);
    for (const name of ['organizationName', 'contactName', 'phone', 'email', 'date', 'startTime', 'endTime', 'expectedParticipants']) {
      assert.equal(document.querySelector(`[name="${name}"]`).required, true, `${name} must be required`);
    }
    await fill(type);
    assert.equal(document.querySelector('[name="seedlings"]'), null);
    assert.doesNotMatch(document.querySelector('.schedule-form').textContent, /Expected seedlings/);
    await submit();
    assert.equal(writes.length, 1);
    const body = writes[0].body;
    assert.equal(body.appointment_type, type);
    assert.equal(body.title, undefined, `${label} is named by the backend`);
    assert.equal(body.contact_name, 'Test coordinator');
    assert.equal(body.phone, '09123456789'); assert.equal(body.email, 'head@example.org');
    assert.equal(body.expected_planters, 10);
    assert.equal(body.start_at, `${date}T08:00:00+08:00`); assert.equal(body.end_at, `${date}T09:30:00+08:00`);
    assert.equal(Object.hasOwn(body, 'expected_seedlings'), false);
    assert.equal(body.notes, 'Meet at the gate.');
  });
}

test('manual errors stop missing, invalid and past booking details before saving', async () => {
  await render(); await click('+ Add schedule'); await submit();
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
    start_at: `${date}T08:00:00+08:00`, end_at: `${date}T09:30:00+08:00`, status: 'requested', expected_planters: null,
    expected_seedlings: 360 }];
  await render(); await click('List'); await click('Edit');
  assert.equal(document.querySelector('[name="appointmentType"]').disabled, true);
  assert.equal(document.querySelector('[name="title"]'), null);
  await change('notes', 'Corrected staff note.'); await submit();
  assert.equal(writes.length, 1);
  assert.equal(writes[0].method, 'PUT');
  assert.equal(writes[0].body.title, undefined); assert.equal(writes[0].body.appointment_type, undefined);
  assert.equal(writes[0].body.contact, contact);
  assert.equal(writes[0].body.expected_planters, null);
  assert.equal(Object.hasOwn(writes[0].body, 'expected_seedlings'), false, 'Editing must leave historical seedling counts untouched');
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

test('calendar and upcoming totals exclude unapproved requests while drafts remain manageable', async () => {
  const today = manilaToday();
  schedules = [
    ['requested', 'Staff draft', 999], ['tentative', 'Tentative activity', 998], ['confirmed', 'Approved activity', 10],
    ['in_progress', 'Ongoing activity', 5], ['completed', 'Finished activity', 15], ['cancelled', 'Cancelled activity', 200],
  ].map(([status, title, participants], index) => ({ id: index + 1, status, title, organization: `Group ${index + 1}`,
    appointment_type: 'tree_planting', start_at: `${today}T08:00:00+08:00`, end_at: `${today}T09:30:00+08:00`,
    expected_planters: participants, expected_seedlings: 360 }));
  requests = [{ id: 21, reference: 'LIKE-PENDING-TEST', organization: 'Unapproved website group', contact_name: 'Coordinator',
    phone: '09123456789', email: 'head@example.org', title: 'Website visit', appointment_type: 'field_visit', participants: 10,
    status: 'pending', start_at: `${today}T10:00:00+08:00`, end_at: `${today}T11:00:00+08:00` }];
  await render();
  const calendar = document.querySelector('.schedule-calendar');
  assert.deepEqual([...calendar.querySelectorAll('.schedule-entry-title')].map((card) => card.textContent),
    ['Approved activity', 'Ongoing activity', 'Finished activity']);
  assert.doesNotMatch(calendar.textContent, /Staff draft|Tentative activity|Cancelled activity|Unapproved website group/);
  assert.equal(document.querySelector('.schedule-calendar-request'), null);
  assert.match(document.querySelector('#website-requests').textContent, /Unapproved website group/);
  assert.ok([...document.querySelectorAll('#website-requests button')].find((button) => button.textContent === 'Review request'));
  assert.deepEqual([...document.querySelectorAll('.schedule-kpis .schedule-tracking-card strong')].map((value) => value.textContent), ['1', '4', '1']);
  assert.doesNotMatch(document.body.textContent, /Expected seedlings/);
  const approvedCard = calendar.querySelector('.schedule-calendar-event');
  await act(async () => approvedCard.click());
  assert.doesNotMatch(document.querySelector('.schedule-details-modal').textContent, /Seedlings/);
  await click('Close'); await click('List');
  const list = document.querySelector('[aria-labelledby="organization-schedules-title"] table');
  for (const title of ['Staff draft', 'Tentative activity', 'Cancelled activity']) assert.ok(list.textContent.includes(title));
});

test('a website request enters the calendar only after successful approval and refresh', async () => {
  const tomorrow = manilaToday(new Date(Date.now() + 86400000));
  const request = { id: 22, reference: 'LIKE-APPROVAL-TEST', organization: 'Website volunteers', contact_name: 'Coordinator',
    phone: '09123456789', email: 'head@example.org', title: 'Field visit', appointment_type: 'field_visit', participants: 10,
    status: 'pending', start_at: `${tomorrow}T08:00:00+08:00`, end_at: `${tomorrow}T09:30:00+08:00` };
  requests = [request];
  onWrite = (write) => {
    assert.match(write.url, /\/like-appointments\/22\/review$/);
    assert.equal(write.body.action, 'confirmed');
    requests = [{ ...request, status: 'confirmed', schedule_id: 1 }];
    schedules = [{ id: 1, ...request, status: 'confirmed', source: 'website', expected_planters: request.participants }];
  };
  await render();
  if (tomorrow.slice(0, 7) !== manilaToday().slice(0, 7)) {
    await act(async () => document.querySelector('[aria-label="Next month"]').click());
  }
  assert.equal(document.querySelectorAll('.schedule-calendar-event').length, 0);
  await click('Review request');
  await act(async () => document.querySelector('[name="contacted"]').click());
  await act(async () => document.querySelector('.website-review').dispatchEvent(new window.Event('submit', { bubbles: true, cancelable: true })));
  assert.equal(writes.length, 1);
  assert.equal(document.querySelectorAll('.schedule-calendar-event').length, 1);
  assert.match(document.querySelector('.schedule-calendar').textContent, /Website volunteers/);
  assert.match(document.querySelector('#website-requests').textContent, /No pending website requests/);
});


test('approved tree planting opens Assign Points with the activity instead of an area chooser', async () => {
  schedules = [{ id: 70,organization_id: 2,organization: 'Coastal volunteers',title: 'Approved planting',
    appointment_type: 'tree_planting',status: 'confirmed',start_at: `${date}T08:00:00+08:00`,
    end_at: `${date}T09:30:00+08:00`,project_site_id: null }];
  await render(); await click('List');
  const link=document.querySelector('a[href="/planters?section=assign&organization_id=2&activity_id=70"]');
  assert.ok(link); assert.equal(link.textContent,'Assign points');
  assert.doesNotMatch(document.body.textContent,/Choose area|Add one in the Zone Editor/);
  assert.match(document.body.textContent,/Formed when points are assigned/);
  assert.equal(writes.length,0);
});

test('completed planting shows its result without missed warnings, assignment actions or tide advice', async () => {
  schedules = [{ id: 72, organization_id: 2, organization: 'Hamtic Coastal Organization', title: 'Tree planting',
    appointment_type: 'tree_planting', status: 'completed', start_at: '2026-10-19T20:25:00+08:00',
    end_at: '2026-10-19T22:25:00+08:00', project_site_id: 29, project_site_name: 'Hamtic Site 1',
    planting_started: true, recorded_planting_count: 150 }];
  await render(); await click('List');
  const table = document.querySelector('.schedule-activity-table');
  assert.match(table.textContent, /Completed/);
  assert.match(table.textContent, /Planting completed/);
  assert.doesNotMatch(table.textContent, /Pending|Missed|Assign points|Update notes|Good time to plant|Avoid planting|Refresh tides/);
  await act(async () => table.querySelector('button').click());
  assert.match(document.querySelector('.schedule-details-modal').textContent, /Planting completed/);
  assert.doesNotMatch(document.querySelector('.schedule-details-modal').textContent, /Good time to plant|Tide forecast is unavailable/);
  assert.equal(writes.length, 0);
});
