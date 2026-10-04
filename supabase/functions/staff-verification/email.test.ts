import test from 'node:test';
import assert from 'node:assert/strict';
import { verificationEmail } from './email.ts';

test('all verification workflows include the same copyable code in HTML and plain text', () => {
  for (const purpose of ['login', 'recovery', 'settings'] as const) {
    const email = verificationEmail(purpose, '123456');
    assert.match(email.subject, /^MangroVision .+ code$/);
    assert.match(email.htmlContent, />123456<\/p>/);
    assert.match(email.textContent, /123456/);
    assert.match(email.htmlContent, /10 minutes/);
    assert.match(email.textContent, /10 minutes/);
    assert.doesNotMatch(email.htmlContent, /\{\{|<script|<form|https?:\/\//);
    assert.equal(Buffer.byteLength(email.htmlContent) < 100_000, true);
  }
});

test('invalid purposes and non-digit code input cannot enter the template', () => {
  for (const code of ['12345', '1234567', '<img/>', '１２３４５６', '123456\n']) {
    assert.throws(() => verificationEmail('login', code));
  }
  assert.throws(() => verificationEmail('unknown' as 'login', '123456'));
});
