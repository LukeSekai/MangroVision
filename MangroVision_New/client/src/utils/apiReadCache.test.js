import test from 'node:test';
import assert from 'node:assert/strict';
import { createApiReadCache } from './apiReadCache.js';

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
