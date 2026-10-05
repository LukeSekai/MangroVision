import test from 'node:test';
import assert from 'node:assert/strict';
import { readAnalysisResponse, startProcessingJob, waitForProcessingJob } from './processingJobs.js';

const reply = (status, body) => new Response(JSON.stringify(body), {
  status, headers: { 'Content-Type': 'application/json' },
});

test('a long running job reports progress and returns its result using GET only', async () => {
  const updates = [];
  const replies = [reply(200, { status: 'running', stage: 'Detecting', pct: 35 }), reply(200, { status: 'succeeded', payload: { analysis_key: 'ready' } })];
  const result = await waitForProcessingJob({
    jobId: 'one', interval: 0, onProgress: (job) => updates.push(job),
    fetcher: async (url, options) => {
      assert.equal(url, '/api/analyses/jobs/one');
      assert.equal(options.cache, 'no-store');
      assert.equal(options.method, undefined);
      return replies.shift();
    },
  });
  assert.equal(updates[0].pct, 35);
  assert.equal(result.analysis_key, 'ready');
});

test('temporary tunnel failure retries the same job without uploading again', async () => {
  let calls = 0;
  const result = await waitForProcessingJob({ jobId: 'existing', interval: 0, fetcher: async () => {
    calls += 1;
    if (calls === 1) throw new TypeError('Network disconnected');
    if (calls === 2) return reply(502, {});
    return reply(200, { status: 'succeeded', payload: { recovered: true } });
  } });
  assert.equal(calls, 3);
  assert.equal(result.recovered, true);
});

test('expired session and failed analysis are surfaced without retrying', async () => {
  for (const response of [reply(401, { detail: 'Session expired' }), reply(200, { status: 'failed', detail: 'Outside map' })]) {
    let calls = 0;
    await assert.rejects(waitForProcessingJob({ jobId: 'one', interval: 0, fetcher: async () => { calls += 1; return response; } }));
    assert.equal(calls, 1);
  }
});

test('resetting the UI aborts waiting and makes no additional request', async () => {
  const controller = new AbortController();
  let calls = 0;
  const waiting = waitForProcessingJob({ jobId: 'one', signal: controller.signal, onProgress: () => controller.abort(), fetcher: async () => {
    calls += 1;
    return reply(200, { status: 'running' });
  } });
  await assert.rejects(waiting, { name: 'AbortError' });
  assert.equal(calls, 1);
});

test('HTML upload gateway error recovers using the same request ID and image', async () => {
  const formData = new FormData();
  formData.set('image', new Blob(['original image bytes']), 'original.jpg');
  let calls = 0;
  const started = await startProcessingJob({ formData, interval: 0, requestId: 'same-request', fetcher: async (url, options) => {
    calls += 1;
    assert.equal(options.method, 'POST');
    assert.equal(options.body.get('request_id'), 'same-request');
    assert.equal(options.body.get('image').size, 20);
    if (calls === 1) return new Response('<!DOCTYPE html><title>Gateway error</title>', { status: 502, headers: { 'Content-Type': 'text/html' } });
    return reply(202, { job_id: 'accepted-job' });
  } });
  assert.equal(started.job_id, 'accepted-job');
  assert.equal(calls, 2);
});

test('HTML or truncated polling response retries the existing job', async () => {
  let calls = 0;
  const result = await waitForProcessingJob({ jobId: 'existing', interval: 0, fetcher: async () => {
    calls += 1;
    if (calls === 1) return new Response('<!DOCTYPE html>proxy page', { headers: { 'Content-Type': 'text/html' } });
    if (calls === 2) return new Response('{"status":', { headers: { 'Content-Type': 'application/json' } });
    return reply(200, { status: 'succeeded', payload: { recovered: true } });
  } });
  assert.equal(calls, 3);
  assert.equal(result.recovered, true);
});

test('permanent HTML upload rejection shows a readable error without retrying', async () => {
  let calls = 0;
  await assert.rejects(startProcessingJob({ formData: new FormData(), requestId: 'one', interval: 0, fetcher: async () => {
    calls += 1;
    return new Response('<!DOCTYPE html>Payload too large', { status: 413, headers: { 'Content-Type': 'text/html' } });
  } }), /image is too large/);
  assert.equal(calls, 1);
  await assert.rejects(readAnalysisResponse(new Response('<!DOCTYPE html>gateway', { status: 504 })), /HTTP 504/);
});
