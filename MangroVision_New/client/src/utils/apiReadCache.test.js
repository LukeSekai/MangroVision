import test from 'node:test';
import assert from 'node:assert/strict';
import { createApiReadCache } from './apiReadCache.js';

test('returning hours later keeps the snapshot until the user refreshes', async () => {
  let clock = 0;
  let reads = 0;
  const cache = createApiReadCache({ origin: 'https://workspace.example', now: () => clock,
    fetcher: async () => Response.json({ revision: ++reads }),
  });
  assert.equal((await (await cache.fetch('/api/dashboard/overview?site_id=1')).json()).revision, 1);
  clock += 24 * 60 * 60 * 1000;
  assert.equal((await (await cache.fetch('/api/dashboard/overview?site_id=1')).json()).revision, 1);
  assert.equal(reads, 1);
  cache.invalidate();
  assert.equal((await (await cache.fetch('/api/dashboard/overview?site_id=1')).json()).revision, 2);
  assert.equal((await (await cache.fetch('/api/dashboard/overview?site_id=1', { cache: 'reload' })).json()).revision, 3);
});

test('notification actions and exports preserve the workspace snapshot', async () => {
  let reads = 0;
  let mutations = 0;
  const cache = createApiReadCache({ origin: 'https://workspace.example', onMutation: () => { mutations += 1; },
    fetcher: async (_url, options) => {
      if (!options.method || options.method === 'GET') reads += 1;
      return Response.json({ ok: true });
    },
  });
  await cache.fetch('/api/analyses/stats');
  await cache.fetch('/api/notifications/refresh', { method: 'POST' });
  await cache.fetch('/api/notifications/12/read', { method: 'POST' });
  await cache.fetch('/api/export/kml', { method: 'POST' });
  await cache.fetch('/api/analyses/stats');
  assert.equal(reads, 1);
  assert.equal(mutations, 0);
});

test('signing out discards retained reads and rejects old-session results', async () => {
  let resolveRead;
  let sessions = 0;
  let reads = 0;
  const cache = createApiReadCache({ origin: 'https://workspace.example', onSessionChange: () => { sessions += 1; },
    fetcher: async (_url, options) => {
      if (options.method === 'POST') return Response.json({ ok: true });
      reads += 1;
      if (reads === 1) return new Promise((resolve) => { resolveRead = resolve; });
      return Response.json({ revision: reads });
    },
  });
  const pending = cache.fetch('/api/planters/map-points');
  await Promise.resolve();
  await cache.fetch('/api/auth/logout', { method: 'POST' });
  resolveRead(Response.json({ revision: 1 }));
  await assert.rejects(pending, { name: 'AbortError' });
  assert.ok(sessions > 0);
  assert.equal((await (await cache.fetch('/api/planters/map-points')).json()).revision, 2);
});

test('image preflight preserves cached workspace data; saving an analysis refreshes it', async () => {
  let reads = 0;
  let mutations = 0;
  const cache = createApiReadCache({ origin: 'https://workspace.example', onMutation: () => { mutations += 1; },
    fetcher: async (_url, options) => {
      if (!options.method || options.method === 'GET') reads += 1;
      return Response.json({ ok: true });
    },
  });
  await cache.fetch('/api/analyses/stats');
  await cache.fetch('/api/analyses/preflight', { method: 'POST' });
  await cache.fetch('/api/analyses/stats');
  assert.equal(reads, 1);
  assert.equal(mutations, 0);
  await cache.fetch('/api/analyses/save', { method: 'POST' });
  await cache.fetch('/api/analyses/stats');
  assert.equal(reads, 2);
  assert.equal(mutations, 1);
});
