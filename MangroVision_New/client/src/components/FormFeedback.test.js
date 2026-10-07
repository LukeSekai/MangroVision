import test, { before, after, beforeEach, afterEach } from 'node:test';
import assert from 'node:assert/strict';
import { act, createElement } from 'react';
import { MemoryRouter, Route, useNavigate } from 'react-router-dom';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';
import { submissionError } from '../utils/formValidation.js';

let server, dom, root, createRoot, useFormFeedback, FieldError, FormErrorSummary, RetainedRoutes, submissions;
const originalGlobals = new Map();
before(async () => {
  dom = new JSDOM('<div id="root"></div>', { pretendToBeVisual: true });
  for (const [key, value] of Object.entries({ window: dom.window, document: dom.window.document,
    IS_REACT_ACT_ENVIRONMENT: true, requestAnimationFrame: dom.window.requestAnimationFrame.bind(dom.window),
  })) {
    originalGlobals.set(key, Object.getOwnPropertyDescriptor(globalThis, key));
    Object.defineProperty(globalThis, key, { configurable: true, writable: true, value });
  }
  ({ createRoot } = await import('react-dom/client'));
  server = await createServer({ configFile: false, appType: 'custom',
    server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    cacheDir: 'node_modules/.vite-form-feedback-tests', optimizeDeps: { noDiscovery: true, include: [] },
  });
  ({ default: useFormFeedback } = await server.ssrLoadModule('/src/utils/useFormFeedback.js'));
  ({ FieldError, FormErrorSummary } = await server.ssrLoadModule('/src/components/FormFeedback.jsx'));
  ({ default: RetainedRoutes } = await server.ssrLoadModule('/src/components/RetainedRoutes.jsx'));
});
after(async () => {
  await server?.close();
  dom?.window.close();
  for (const [key, descriptor] of originalGlobals) {
    if (descriptor) Object.defineProperty(globalThis, key, descriptor);
    else delete globalThis[key];
  }
});
beforeEach(() => { submissions = 0; root = createRoot(document.getElementById('root')); });
afterEach(async () => { await act(async () => root.unmount()); });

function Form({ onServerHandled = () => {} }) {
  const feedback = useFormFeedback({ count: { label: 'Participants', aliases: ['expected_planters'] }, title: { label: 'Activity title' }, code: { label: 'Email code' } });
  return createElement('form', { noValidate: true, onChangeCapture: feedback.onChange, onSubmit: (event) => {
    event.preventDefault();
    if (feedback.validate()) submissions++;
  } },
  createElement(FormErrorSummary, { feedback }),
  createElement('label', null, 'Participants', createElement('input', { ...feedback.props('count'), type: 'number', required: true, min: '1', max: '20', step: '1' }), createElement(FieldError, { feedback, field: 'count' })),
  createElement('label', null, 'Activity title', createElement('input', { ...feedback.props('title'), required: true }), createElement(FieldError, { feedback, field: 'title' })),
  createElement('button', { type: 'submit' }, 'Save'),
  createElement('button', { type: 'button', onClick: () => feedback.fromServer(submissionError([
    { loc: ['body', 'expected_planters'], type: 'less_than_equal', ctx: { le: 20 } },
  ])) }, 'Simulate server validation'),
  createElement('button', { 'data-unmounted-error': true, type: 'button', onClick: () => {
    onServerHandled(feedback.fromServer(submissionError([{ loc: ['body', 'code'], type: 'missing' }])));
  } }, 'Simulate an error for another authentication step'));
}
const settle = async () => { await act(async () => { await new Promise((resolve) => setTimeout(resolve, 35)); }); };
const change = async (control, value) => {
  await act(async () => {
    Object.getOwnPropertyDescriptor(window.HTMLInputElement.prototype, 'value').set.call(control, value);
    control.dispatchEvent(new window.Event('input', { bubbles: true }));
  });
};

function PageTabs() {
  const navigate = useNavigate();
  return createElement('nav', null,
    createElement('button', { 'data-tab': 'other', onClick: () => navigate('/other') }, 'Another tab'),
    createElement('button', { 'data-tab': 'edit', onClick: () => navigate('/edit') }, 'Return to form'));
}

test('switching tabs preserves unfinished inputs without marking them invalid', async () => {
  await act(async () => root.render(createElement(MemoryRouter, { initialEntries: ['/edit'] },
    createElement(PageTabs),
    createElement(RetainedRoutes, { paths: ['/edit', '/other'] },
      createElement(Route, { path: '/edit', element: createElement(Form) }),
      createElement(Route, { path: '/other', element: createElement('p', null, 'Another page') })))));
  const count = document.querySelector('[name="count"]');
  const title = document.querySelector('[name="title"]');
  await change(title, 'Unfinished activity');
  await act(async () => {
    count.focus();
    document.querySelector('[data-tab="other"]').focus();
    document.querySelector('[data-tab="other"]').click();
  });
  await settle();
  await act(async () => document.querySelector('[data-tab="edit"]').click());
  await settle();
  assert.equal(document.querySelector('[name="count"]'), count);
  assert.equal(title.value, 'Unfinished activity');
  assert.equal(count.getAttribute('aria-invalid'), 'false');
  assert.equal(document.querySelector('.form-field-error'), null);
  assert.equal(submissions, 0);
  await act(async () => document.querySelector('[type="submit"]').click());
  await settle();
  assert.equal(count.getAttribute('aria-invalid'), 'true');
  assert.equal(document.activeElement, count);
  assert.equal(submissions, 0);
});

test('invalid submissions focus the first field and link each accessible correction to its input', async () => {
  await act(async () => root.render(createElement(Form)));
  await act(async () => document.querySelector('[type="submit"]').click());
  await settle();
  const count = document.querySelector('[name="count"]');
  const title = document.querySelector('[name="title"]');
  assert.equal(document.activeElement, count);
  assert.equal(count.getAttribute('aria-invalid'), 'true');
  assert.equal(document.getElementById(count.getAttribute('aria-describedby')).textContent, 'Enter participants.');
  assert.equal(document.getElementById(title.getAttribute('aria-describedby')).textContent, 'Enter activity title.');
  assert.equal(submissions, 0);
  await change(count, '4');
  assert.equal(count.getAttribute('aria-invalid'), 'false');
  assert.equal(title.getAttribute('aria-invalid'), 'true');
  await change(title, 'Planting visit');
  await act(async () => document.querySelector('[type="submit"]').click());
  assert.equal(submissions, 1);
  assert.equal(document.querySelector('[role="alert"]'), null);
});

test('server validation stays beside its field and focuses that field', async () => {
  await act(async () => root.render(createElement(Form)));
  await act(async () => document.querySelector('[type="button"]').click());
  await settle();
  const count = document.querySelector('[name="count"]');
  assert.equal(document.activeElement, count);
  assert.equal(document.getElementById(count.getAttribute('aria-describedby')).textContent, 'Enter 20 or less for participants.');
  assert.equal(document.querySelector('[name="title"]').getAttribute('aria-invalid'), 'false');
});

test('errors for a hidden authentication field stay at form level', async () => {
  let handled = null;
  await act(async () => root.render(createElement(Form, { onServerHandled: (value) => { handled = value; } })));
  await act(async () => document.querySelector('[data-unmounted-error]').click());
  assert.equal(handled, false);
  assert.equal(document.querySelector('.form-field-error'), null);
});
