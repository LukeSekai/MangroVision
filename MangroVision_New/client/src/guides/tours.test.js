import test from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { JSDOM } from 'jsdom';
import { STAFF_TOURS, FIELD_TOURS, fullWorkflow, toursForMode } from './tours.js';

test('every staff feature route is reachable through its task guide and the full workflow', () => {
  const app = readFileSync(new URL('../StaffApp.jsx', import.meta.url), 'utf8');
  const featurePaths = [...app.matchAll(/<Route path="([^"]+)" element=\{<(?!Navigate\b)/g)]
    .map((match) => match[1]);
  assert.ok(featurePaths.length >= 11, 'Check the actual staff route declarations');
  const workflow = fullWorkflow(STAFF_TOURS);
  for (const path of featurePaths) {
    const tour = STAFF_TOURS.find((item) => item.path === path);
    assert.ok(tour?.steps.length, `Missing task guide for ${path}`);
    assert.ok(workflow.some((step) => step.path === path), `Full workflow omits ${path}`);
  }
  assert.equal(new Set(STAFF_TOURS.map((tour) => tour.id)).size, STAFF_TOURS.length);
  assert.equal(new Set(workflow).size, workflow.length, 'Do not repeat a task guide');
  assert.equal(workflow.length, STAFF_TOURS.reduce((count, tour) => count + tour.steps.length, 0));
  const first = (path) => workflow.findIndex((step) => step.path === path);
  assert.ok(first('/scheduling') < first('/organizations'));
  assert.ok(first('/organizations') < first('/zones'));
  assert.ok(first('/zones') < first('/planters'));
  assert.ok(first('/processing') < first('/planters'));
  assert.ok(first('/planters') < first('/monitoring'));
});

test('guide targets use valid selectors and stay inside their authenticated workspace', () => {
  const dom = new JSDOM('<body></body>');
  try {
    const app = readFileSync(new URL('../StaffApp.jsx', import.meta.url), 'utf8');
    const allowed = new Set([...app.matchAll(/<Route path="([^"]+)"/g)].map((match) => match[1]));
    for (const tour of STAFF_TOURS) {
      for (const step of tour.steps) {
        assert.ok(allowed.has(step.path), `${tour.id} leaves the staff router: ${step.path}`);
        assert.doesNotThrow(() => dom.window.document.querySelector(step.target), step.target);
        assert.ok(step.title && step.content, `Missing instructions in ${tour.id}`);
      }
    }
    // Field users cannot navigate through staff-only screens while learning.
    for (const tour of FIELD_TOURS) {
      assert.ok(tour.steps.every((step) => step.path == null));
    }
  } finally {
    dom.window.close();
  }
});

test('staff, field and dormant sign-in menus keep their own guide sets', () => {
  assert.equal(toursForMode('staff'), STAFF_TOURS);
  assert.equal(toursForMode('field'), FIELD_TOURS);
  assert.deepEqual(toursForMode('staff-login').map((tour) => tour.id), ['staff-login']);
  assert.deepEqual(toursForMode('field-login').map((tour) => tour.id), ['field-login']);
  assert.ok(!STAFF_TOURS.some((tour) => tour.id.endsWith('-login')));
});
