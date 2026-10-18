import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement, Fragment, useState } from 'react';
import { MemoryRouter, useLocation } from 'react-router-dom';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

let server, dom, root, createRoot, EmailCodeInput, LogoutConfirmation, Sidebar, WorkspaceHeader, useAuthStore, useFormFeedback;
let submitted, resendCount, logoutCount;
const savedGlobals = new Map();

before(async () => {
  dom = new JSDOM('<div id="root"></div>', { url: 'http://auth.example.test', pretendToBeVisual: true });
  for (const [key, value] of Object.entries({ window: dom.window, document: dom.window.document,
    localStorage: dom.window.localStorage, sessionStorage: dom.window.sessionStorage,
    IS_REACT_ACT_ENVIRONMENT: true, requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
    fetch: async () => Response.json({ items: [], unread_count: 0 }),
  })) {
    savedGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  ({ createRoot } = await import('react-dom/client'));
  server = await createServer({ configFile: false, appType: 'custom',
    server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    cacheDir: 'node_modules/.vite-auth-interaction-tests', optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ default: EmailCodeInput } = await server.ssrLoadModule('/src/components/EmailCodeInput.jsx'));
  ({ default: LogoutConfirmation } = await server.ssrLoadModule('/src/components/LogoutConfirmation.jsx'));
  ({ default: Sidebar } = await server.ssrLoadModule('/src/components/Sidebar.jsx'));
  ({ default: WorkspaceHeader } = await server.ssrLoadModule('/src/components/WorkspaceHeader.jsx'));
  ({ useAuthStore } = await server.ssrLoadModule('/src/stores/authStore.js'));
  ({ default: useFormFeedback } = await server.ssrLoadModule('/src/utils/useFormFeedback.js'));
});
after(async () => {
  await server?.close(); dom?.window.close();
  for (const [key, descriptor] of savedGlobals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor); else delete globalThis[key];
  }
});
beforeEach(() => {
  submitted = []; resendCount = 0; logoutCount = 0;
  root = createRoot(document.getElementById('root'));
});
afterEach(async () => { await act(async () => root.unmount()); });

function CodeForm({ busy = false, readyAt = 0, initialCode = '' }) {
  const [code, setCode] = useState(initialCode);
  const feedback = useFormFeedback({ code: { label: 'Email verification code' } });
  return createElement('form', { noValidate: true, onChangeCapture: feedback.onChange, onSubmit: (event) => {
    event.preventDefault(); if (feedback.validate()) submitted.push(code);
  } },
  createElement(EmailCodeInput, { code, onChange: setCode, busy, readyAt, feedback, onResend: () => { resendCount++; setCode(''); feedback.clear(); } }),
  createElement('button', { type: 'submit' }, 'Verify'),
  createElement('button', { type: 'button', onClick: () => feedback.fromServer({ fieldErrors: { code: 'That code is incorrect. Try again.' } }) }, 'Reject code'));
}
const digits = () => [...document.querySelectorAll('.email-code-digit')];
const values = () => digits().map(input => input.value);
const button = (text) => [...document.querySelectorAll('button')].find(input => input.textContent === text);
const render = async (element) => { await act(async () => root.render(element)); };
const click = async (element) => { assert.ok(element); await act(async () => element.click()); };
const settle = async () => { await act(async () => { await new Promise(resolve => setTimeout(resolve, 30)); }); };
async function change(index, value) {
  await act(async () => {
    const input = digits()[index];
    input.focus();
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(input, value);
    input.dispatchEvent(new window.Event('input', { bubbles: true }));
  });
}
async function key(index, value) {
  await act(async () => digits()[index].dispatchEvent(new window.KeyboardEvent('keydown', { key: value, bubbles: true, cancelable: true })));
}
async function paste(index, value) {
  await act(async () => {
    digits()[index].focus();
    const event = new window.Event('paste', { bubbles: true, cancelable: true });
    Object.defineProperty(event, 'clipboardData', { value: { getData: () => value } });
    digits()[index].dispatchEvent(event);
  });
}

test('six labelled digit boxes advance focus and submit one complete code', async () => {
  await render(createElement(CodeForm));
  assert.equal(digits().length, 6);
  assert.equal(document.activeElement, digits()[0]);
  assert.equal(digits()[0].autocomplete, 'one-time-code');
  for (let index = 0; index < 6; index++) {
    assert.equal(digits()[index].getAttribute('aria-label'), `Digit ${index + 1} of 6`);
    await change(index, String(index + 1));
    assert.equal(document.activeElement, digits()[Math.min(index + 1, 5)]);
  }
  await click(button('Verify'));
  assert.deepEqual(submitted, ['123456']);
});

test('incomplete codes and holes remain invalid; correcting a digit clears shared feedback', async () => {
  await render(createElement(CodeForm));
  await change(0, '1');
  await click(button('Verify')); await settle();
  assert.deepEqual(submitted, []);
  assert.equal(document.activeElement, digits()[0]);
  assert.match(document.querySelector('.form-field-error').textContent, /all six digits/);
  assert.ok(digits().every(input => input.getAttribute('aria-invalid') === 'true'));
  await paste(0, '123456');
  await key(2, 'Delete');
  assert.deepEqual(values(), ['1', '2', '', '4', '5', '6']);
  await click(button('Verify')); await settle();
  assert.deepEqual(submitted, []);
  await change(2, '9');
  assert.equal(document.querySelector('.form-field-error'), null);
  await click(button('Verify'));
  assert.deepEqual(submitted, ['129456']);
});

test('pasting a full code replaces all boxes from any position and autofill accepts six digits', async () => {
  await render(createElement(CodeForm));
  await paste(3, 'Code: 083 052');
  assert.deepEqual(values(), ['0', '8', '3', '0', '5', '2']);
  await click(button('Verify'));
  await change(0, '908172');
  await click(button('Verify'));
  assert.deepEqual(submitted, ['083052', '908172']);
  await change(1, 'letters');
  assert.equal(digits()[1].value, '0');
});

test('backspace, arrows, Home and End allow corrections without shifting other digits', async () => {
  await render(createElement(CodeForm));
  await paste(0, '123456');
  await key(3, 'Backspace');
  assert.deepEqual(values(), ['1', '2', '3', '', '5', '6']);
  assert.equal(document.activeElement, digits()[3]);
  await key(3, 'Backspace');
  assert.equal(document.activeElement, digits()[2]);
  assert.deepEqual(values(), ['1', '2', '', '', '5', '6']);
  await key(2, 'ArrowLeft'); assert.equal(document.activeElement, digits()[1]);
  await key(1, 'ArrowRight'); assert.equal(document.activeElement, digits()[2]);
  await key(2, 'End'); assert.equal(document.activeElement, digits()[5]);
  await key(5, 'Home'); assert.equal(document.activeElement, digits()[0]);
});

test('server errors focus the code boxes; resend resets the code and keeps cooldown/disabled behavior', async () => {
  await render(createElement(CodeForm));
  await paste(0, '123456');
  await click(button('Reject code')); await settle();
  assert.equal(document.activeElement, digits()[0]);
  assert.equal(document.getElementById(digits()[5].getAttribute('aria-describedby')).textContent, 'That code is incorrect. Try again.');
  await paste(0, '654321');
  assert.equal(document.querySelector('.form-field-error'), null);
  await click(button('Resend code'));
  assert.equal(resendCount, 1);
  assert.deepEqual(values(), ['', '', '', '', '', '']);
  assert.equal(document.activeElement, digits()[0]);
  await render(createElement(CodeForm, { busy: true, readyAt: Date.now() + 60000 }));
  assert.ok(digits().every(input => input.disabled));
  assert.ok(document.querySelector('.email-code-help button').disabled);
});

function CurrentRoute() {
  const location = useLocation();
  return createElement('output', { 'data-current-route': true }, location.pathname);
}

test('header profile opens Account and the brand opens Planting Map without signing out', async (context) => {
  context.mock.timers.enable({ apis: ['Date'], now: Date.parse('2026-10-19T00:30:00Z') });
  useAuthStore.setState({ user: { full_name: 'Test planner' }, logout: () => { logoutCount++; } });
  await render(createElement(MemoryRouter, { initialEntries: ['/dashboard'] },
    createElement(Fragment, null, createElement(Sidebar), createElement(WorkspaceHeader), createElement(CurrentRoute))));
  await settle();
  const header = document.querySelector('.workspace-header');
  assert.match(header.textContent, /Test planner/);
  assert.equal(header.querySelector('.workspace-clock-time').textContent, '8:30:00 AM');
  assert.match(header.querySelector('.workspace-clock').title, /Philippine time/);
  assert.equal(document.querySelector('aside .workspace-clock, aside [href="/account"]'), null);
  await click(header.querySelector('[aria-label="Account settings"]'));
  assert.equal(document.querySelector('[data-current-route]').textContent, '/account');
  await click(document.querySelector('.sidebar-logo'));
  assert.equal(document.querySelector('[data-current-route]').textContent, '/map');
  assert.equal(logoutCount, 0);
  await click(header.querySelector('[aria-label="Help / Guide"]'));
  assert.match(document.querySelector('[role="dialog"]').textContent, /How to use MangroVision/);
});

test('staff logout requires confirmation; staying signed in or Escape keeps the session', async () => {
  useAuthStore.setState({ user: { full_name: 'Test planner' }, logout: () => { logoutCount++; } });
  await render(createElement(MemoryRouter, null, createElement(Sidebar)));
  const trigger = document.querySelector('[title="Sign out"]');
  await act(async () => trigger.focus());
  await click(trigger);
  assert.equal(logoutCount, 0);
  assert.equal(document.activeElement, button('Stay signed in'));
  assert.equal(document.querySelector('aside [role="dialog"]'), null, 'Dialog is portalled outside the sidebar');
  await click(button('Stay signed in'));
  assert.equal(logoutCount, 0);
  assert.equal(document.activeElement, trigger);
  await click(trigger);
  await act(async () => button('Stay signed in').dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
  assert.equal(document.querySelector('[role="dialog"]'), null);
  assert.equal(logoutCount, 0);
  await click(trigger);
  await click(document.querySelector('[role="dialog"] .btn-primary'));
  assert.equal(logoutCount, 1);
  assert.equal(document.querySelector('[role="dialog"]'), null);
});

test('a pending logout runs once and cannot be dismissed until it finishes', async () => {
  let finish;
  await render(createElement(LogoutConfirmation, { open: true, onCancel: () => assert.fail('Pending logout must not be cancelled'), onConfirm: () => {
    logoutCount++; return new Promise(resolve => { finish = resolve; });
  } }));
  await click(document.querySelector('[role="dialog"] .btn-primary'));
  assert.equal(logoutCount, 1);
  assert.ok([...document.querySelectorAll('[role="dialog"] button')].every(input => input.disabled));
  await click(document.querySelector('[role="dialog"] .btn-primary'));
  await act(async () => document.dispatchEvent(new window.KeyboardEvent('keydown', { key: 'Escape', bubbles: true })));
  assert.equal(logoutCount, 1);
  await act(async () => finish());
});
