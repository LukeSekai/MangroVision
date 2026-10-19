import test, { before, after, beforeEach } from 'node:test';
import assert from 'node:assert/strict';
import { JSDOM } from 'jsdom';
import { createServer } from 'vite';

let server, dom, store, fetcher;
const originalGlobals = new Map();
const result = { analysis_key: 'unsaved-result', uploaded_file_name: 'photo.jpg',
  images: { original_preview_url: '/preview.jpg' }, map: { available: true } };
const key = 'mv_analysis_job_7';
const reply = (body, status = 200) => Response.json(body, { status });

before(async () => {
  dom = new JSDOM('', { url: 'http://workspace.example' });
  for (const [name, value] of Object.entries({ sessionStorage: dom.window.sessionStorage,
    FileReader: dom.window.FileReader, File: dom.window.File, FormData: dom.window.FormData,
    fetch: (url, options) => fetcher(url, options) })) {
    originalGlobals.set(name, Object.getOwnPropertyDescriptor(globalThis, name));
    Object.defineProperty(globalThis, name, { configurable: true, writable: true, value });
  }
  server = await createServer({ configFile: false, appType: 'custom',
    server: { middlewareMode: true, hmr: false, watch: null, ws: false },
    cacheDir: 'node_modules/.vite-processing-tests', optimizeDeps: { noDiscovery: true, include: [] } });
});

async function freshStore() {
  // Discard JS memory, retaining only the browser's per-tab storage, like refresh.
  server.moduleGraph.invalidateAll();
  return (await server.ssrLoadModule('/src/stores/processingStore.js')).useProcessingStore;
}

beforeEach(async () => {
  store?.getState().reset();
  sessionStorage.clear();
  store = await freshStore();
});

after(async () => {
  store?.getState().reset();
  await server?.close();
  dom?.window.close();
  for (const [name, descriptor] of originalGlobals) {
    if (descriptor) Object.defineProperty(globalThis, name, descriptor);
    else delete globalThis[name];
  }
});

async function until(condition) {
  for (let i = 0; i < 100; i += 1) {
    if (condition()) return;
    await new Promise(resolve => setTimeout(resolve, 10));
  }
  assert.fail('Processing did not reach the expected state');
}

test('refresh restores a running job and its result without another upload', async () => {
  let uploads = 0, reads = 0, finish = false;
  fetcher = async (url, options = {}) => {
    if (options.method === 'POST') { uploads += 1; return reply({ job_id: 'job-one' }, 202); }
    assert.equal(url, '/api/analyses/jobs/job-one');
    reads += 1;
    return reply(finish ? { status: 'succeeded', payload: result } : { status: 'running', stage: 'Detecting', pct: 40 });
  };
  const pending = store.getState().startProcess({ ownerId: 7, file: new File(['image'], 'photo.jpg') });
  await until(() => reads === 1);
  const savedReference = JSON.parse(sessionStorage.getItem(key));
  assert.equal(savedReference.jobId, 'job-one');
  assert.deepEqual(Object.keys(savedReference).sort(), ['fileName', 'jobId', 'requestId']);
  store.getState().cancel();
  await pending;
  store = await freshStore();
  finish = true;
  await Promise.all([store.getState().resumeProcessing(7), store.getState().resumeProcessing(7)]);
  assert.equal(uploads, 1);
  assert.equal(reads, 2);
  assert.equal(store.getState().result.analysis_key, 'unsaved-result');
  assert.equal(store.getState().resultPreviewUrl, '/preview.jpg');
  assert.equal(store.getState().overlayOpen, true);
  assert.equal(store.getState().processing, false);
});

test('refresh after a lost upload acknowledgement finds the original request', async () => {
  let requestId, uploads = 0;
  fetcher = async (url, options = {}) => {
    if (options.method === 'POST') {
      uploads += 1;
      requestId = options.body.get('request_id');
      throw new TypeError('Acknowledgement lost');
    }
    if (url.startsWith('/api/analyses/jobs/current')) {
      assert.equal(new URL(url, 'http://example').searchParams.get('request_id'), requestId);
      return reply({ job: { job_id: 'accepted-before-refresh' } });
    }
    assert.equal(url, '/api/analyses/jobs/accepted-before-refresh');
    return reply({ status: 'succeeded', payload: result });
  };
  const pending = store.getState().startProcess({ ownerId: 7, file: new File(['image'], 'photo.jpg') });
  await until(() => uploads === 1);
  store.getState().cancel();
  await pending;
  store = await freshStore();
  await store.getState().resumeProcessing(7);
  assert.equal(uploads, 1);
  assert.equal(store.getState().result.analysis_key, result.analysis_key);
});

test('a completed unsaved result survives refresh and saving clears recovery', async () => {
  sessionStorage.setItem(key, JSON.stringify({ jobId: 'finished-job', fileName: 'photo.jpg' }));
  fetcher = async (url, options = {}) => {
    if (options.method === 'POST') {
      assert.equal(url, '/api/analyses/save');
      return reply({ analysis_id: 69, analysis_name: 'Analysis 35', display_number: 35, new_points: 3 });
    }
    assert.equal(url, '/api/analyses/jobs/finished-job');
    return reply({ status: 'succeeded', payload: result });
  };
  await store.getState().resumeProcessing(7);
  await store.getState().saveCurrentAnalysis(7);
  assert.equal(store.getState().result.uploaded_file_name, 'Analysis 35');
  assert.equal(store.getState().result.analysis_id, 69);
  assert.equal(sessionStorage.getItem(key), null);
});

test('reset and account switching stop old progress without exposing the old result', async () => {
  sessionStorage.setItem(key, JSON.stringify({ jobId: 'private-job' }));
  let release;
  fetcher = async url => {
    if (url === '/api/analyses/jobs/private-job') return new Promise(resolve => { release = resolve; });
    assert.equal(url, '/api/analyses/jobs/current');
    return reply({ job: null });
  };
  const pending = store.getState().resumeProcessing(7);
  await until(() => release);
  store.getState().reset();
  await store.getState().resumeProcessing(8);
  release(reply({ status: 'succeeded', payload: result }));
  await pending;
  assert.equal(store.getState().ownerId, 8);
  assert.equal(store.getState().result, null);
  assert.equal(sessionStorage.getItem(key), null);
});

test('an expired backend job explains recovery failure and clears its stale reference', async () => {
  sessionStorage.setItem(key, JSON.stringify({ jobId: 'expired-job' }));
  fetcher = async () => reply({ detail: 'Analysis job not found or expired. The laptop server may have restarted.' }, 404);
  await store.getState().resumeProcessing(7);
  assert.equal(store.getState().processing, false);
  assert.match(store.getState().error, /expired/);
  assert.equal(sessionStorage.getItem(key), null);
});

test('a save completing after logout cannot restore the previous accounts result', async () => {
  let release;
  store.setState({ ownerId: 7, result });
  fetcher = async () => new Promise(resolve => { release = resolve; });
  const pending = store.getState().saveCurrentAnalysis(7);
  await until(() => release);
  store.getState().reset();
  release(reply({ analysis_id: 69, analysis_name: 'Analysis 35' }));
  assert.equal(await pending, null);
  assert.equal(store.getState().result, null);
  assert.equal(store.getState().saving, false);
});

test('without browser state only this users active job is discovered and retained for another refresh', async () => {
  fetcher = async url => url === '/api/analyses/jobs/current'
    ? reply({ job: { job_id: 'active-job', file_name: 'photo.jpg' } })
    : reply({ status: 'succeeded', payload: result });
  await store.getState().resumeProcessing(7);
  assert.equal(store.getState().result.analysis_key, result.analysis_key);
  assert.equal(JSON.parse(sessionStorage.getItem(key)).jobId, 'active-job');
});

test('checking for an old analysis on page load does not lock editing when no job is running', async () => {
  let replyToRecovery;
  const observed=[];
  const unsubscribe=store.subscribe((state)=>observed.push(state.processing));
  fetcher=async () => new Promise((resolve)=>{ replyToRecovery=resolve; });
  const pending=store.getState().resumeProcessing(7);
  await until(()=>replyToRecovery);
  assert.equal(store.getState().processing,false);
  replyToRecovery(reply({ job:null }));
  await pending;unsubscribe();
  assert.ok(observed.every((processing)=>processing===false));
});

test('an old completed job never briefly locks editing while its result is restored', async () => {
  sessionStorage.setItem(key,JSON.stringify({ jobId:'already-finished' }));
  const observed=[];
  const unsubscribe=store.subscribe((state)=>observed.push(state.processing));
  fetcher=async () => reply({ status:'succeeded',payload:result });
  await store.getState().resumeProcessing(7);unsubscribe();
  assert.equal(store.getState().result.analysis_key,result.analysis_key);
  assert.ok(observed.every((processing)=>processing===false));
});

test('a recovered job locks editing only after the server confirms running and unlocks at completion', async () => {
  sessionStorage.setItem(key,JSON.stringify({ jobId:'really-running' }));
  let finish=false;
  fetcher=async () => reply(finish?{ status:'succeeded',payload:result }:{ status:'running',stage:'Detecting',pct:40 });
  const pending=store.getState().resumeProcessing(7);
  await until(()=>store.getState().processing);
  assert.equal(store.getState().stage,'Detecting');
  finish=true;await pending;
  assert.equal(store.getState().processing,false);
});
