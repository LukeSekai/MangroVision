import test from 'node:test';
import assert from 'node:assert/strict';
import { Buffer } from 'node:buffer';
import { createServer } from 'node:http';
import { mkdtemp, mkdir, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { basename, dirname, join, resolve } from 'node:path';
import localOrthophotoTiles from './localOrthophotoTiles.js';

test('localhost tile responses are reusable, support ETags, and expose only numbered images', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'mv-tile-cache-'));
  const filename = join(directory, '1', '0', '0.png');
  await mkdir(join(directory, '1', '0'), { recursive: true });
  const bytes = Buffer.from('fictional tile bytes');
  await writeFile(filename, bytes);
  let handler;
  localOrthophotoTiles({ directory }).configureServer({ middlewares: { use(value) { handler = value; } } });
  const server = createServer((request, response) => handler(request, response, () => { response.statusCode = 404; response.end(); }));
  await new Promise((resolve) => server.listen(0, '127.0.0.1', resolve));
  try {
    const origin = `http://127.0.0.1:${server.address().port}`;
    const response = await fetch(`${origin}/tiles/FINAL/1/0/0.png?v=revision`);
    assert.equal(response.status, 200);
    assert.equal(response.headers.get('cache-control'), 'public, max-age=3600');
    assert.deepEqual(Buffer.from(await response.arrayBuffer()), bytes);
    const etag = response.headers.get('etag');
    assert.ok(etag);
    assert.equal((await fetch(`${origin}/tiles/FINAL/1/0/0.png`, { headers: { 'if-none-match': etag } })).status, 304);
    assert.equal((await fetch(`${origin}/tiles/FINAL/1/0/0.png`, { method: 'HEAD' })).headers.get('content-length'), String(bytes.length));
    assert.equal((await fetch(`${origin}/tiles/FINAL/1/0/0.tif`)).status, 404);
    assert.equal((await fetch(`${origin}/tiles/FINAL/1/2/0.png`)).status, 404);
  } finally {
    server.closeAllConnections();
    await new Promise((resolve) => server.close(resolve));
    const cleanupPath = resolve(directory);
    assert.equal(dirname(cleanupPath), resolve(tmpdir()));
    assert.ok(basename(cleanupPath).startsWith('mv-tile-cache-'));
    await rm(cleanupPath, { recursive: true, force: true });
  }
});
