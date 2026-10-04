import test from 'node:test';
import assert from 'node:assert/strict';
import { handleVerification, digest, codeDigest, type Challenge } from './logic.ts';

const token = 'x'.repeat(43), code = '123456', authorization = 'private test authorization';

async function setup(overrides: Partial<Challenge> = {}) {
  const row: Challenge = {
    user_email:'admin@example.com', purpose:'login', code_hash:await codeDigest(token, code),
    request_key:await digest('admin@example.com'), delivered:false, consumed_at:null,
    email_claimed_at:null, expires_at:new Date(Date.now()+600_000).toISOString(), attempts:0,
    ...overrides,
  };
  let reads = 0, claims = 0;
  const sent: unknown[] = [];
  const deps = {
    authorizationToken:authorization, configured:true,
    findChallenge:async (hash: string) => { reads++; assert.equal(hash, await digest(token)); return row; },
    claimChallenge:async () => { claims++; return claims === 1; },
    sendEmail:async (...args: unknown[]) => { sent.push(args); },
  };
  return { row, deps, sent, counts:() => ({ reads, claims }) };
}

function request(body: unknown = { challenge_token:token, code }, auth = authorization) {
  return new Request('https://example.supabase.co/functions/v1/staff-verification', {
    method:'POST', headers:{ 'x-mangrovision-staff-email-token':auth }, body:JSON.stringify(body),
  });
}

test('requires private authorization before database access', async () => {
  const state = await setup();
  assert.equal((await handleVerification(request(undefined, 'incorrect'), state.deps)).status, 401);
  assert.deepEqual(state.counts(), { reads:0, claims:0 });
});

test('rejects bad payloads without database access', async () => {
  const state = await setup();
  for (const body of [null, {}, { challenge_token:'short', code }, { challenge_token:token, code:'123' }]) {
    assert.equal((await handleVerification(request(body), state.deps)).status, 400);
  }
  assert.deepEqual(state.counts(), { reads:0, claims:0 });
});

test('valid code sends only to the database recipient and database purpose', async () => {
  const state = await setup();
  const response = await handleVerification(request({ challenge_token:token, code, recipient:'attacker@example.com', purpose:'settings' }), state.deps);
  assert.equal(response.status, 200);
  assert.deepEqual(await response.json(), { sent:true });
  assert.deepEqual(state.sent, [['admin@example.com','login',code]]);
});

test('wrong code never claims or sends', async () => {
  const state = await setup();
  assert.equal((await handleVerification(request({ challenge_token:token, code:'000000' }), state.deps)).status, 409);
  assert.equal(state.counts().claims, 0);
  assert.deepEqual(state.sent, []);
});

test('expired, exhausted, consumed or delivered challenges never send', async () => {
  for (const overrides of [{ expires_at:new Date(Date.now()-1).toISOString() }, { expires_at:'invalid' },
      { consumed_at:new Date().toISOString() }, { delivered:true }, { attempts:5 }, { email_claimed_at:new Date().toISOString() }]) {
    const state = await setup(overrides);
    assert.equal((await handleVerification(request(), state.deps)).status, 409);
    assert.equal(state.counts().claims, 0);
  }
});

test('rejects changed recipient and placeholder addresses', async () => {
  for (const user_email of ['changed@example.com', 'admin@mangrovision.local']) {
    const state = await setup({ user_email });
    assert.equal((await handleVerification(request(), state.deps)).status, 409);
    assert.deepEqual(state.sent, []);
  }
});

test('concurrent claims cannot send the same challenge twice', async () => {
  const state = await setup();
  const statuses = await Promise.all([handleVerification(request(), state.deps), handleVerification(request(), state.deps)]);
  assert.deepEqual(statuses.map((response) => response.status).sort(), [200,409]);
  assert.equal(state.sent.length, 1);
});

test('provider errors fail closed without sensitive exception text', async () => {
  const state = await setup();
  state.deps.sendEmail = async () => { throw new Error('private API credential'); };
  const response = await handleVerification(request(), state.deps);
  assert.equal(response.status, 502);
  assert.equal((await response.text()).includes('private API credential'), false);
});
