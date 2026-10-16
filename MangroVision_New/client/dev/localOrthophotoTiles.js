import { createReadStream } from 'node:fs';
import { stat } from 'node:fs/promises';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';

// Serve the existing exported tiles before Vite's API proxy. This lets the
// public website preview display the orthophoto without starting FastAPI.
// Only numbered PNG tiles are exposed; the GeoTIFF and other files stay private.
export default function localOrthophotoTiles({ directory = fileURLToPath(new URL('../../../MAP/FINAL/', import.meta.url)) } = {}) {
  const install = (server) => {
    server.middlewares.use(async (request, response, next) => {
      if (!['GET', 'HEAD'].includes(request.method)) return next();
      const match = /^\/tiles\/FINAL\/(\d{1,2})\/(\d{1,10})\/(\d{1,10})\.png(?:\?.*)?$/.exec(request.url || '');
      if (!match) return next();
      const [, zoom, x, y] = match;
      if (Number(zoom) > 30 || Number(x) >= 2 ** Number(zoom) || Number(y) >= 2 ** Number(zoom)) return next();
      const filename = join(directory, zoom, x, `${y}.png`);
      let info;
      try { info = await stat(filename); } catch { return next(); }
      if (!info.isFile()) return next();

      response.setHeader('Content-Type', 'image/png');
      response.setHeader('Cache-Control', 'public, max-age=3600');
      const etag = `"${info.size.toString(16)}-${Math.trunc(info.mtimeMs).toString(16)}"`;
      response.setHeader('ETag', etag);
      response.setHeader('Last-Modified', info.mtime.toUTCString());
      response.setHeader('X-Content-Type-Options', 'nosniff');
      if (request.headers['if-none-match'] === etag) {
        response.statusCode = 304;
        return response.end();
      }
      response.setHeader('Content-Length', info.size);
      if (request.method === 'HEAD') return response.end();
      const stream = createReadStream(filename);
      stream.on('error', () => response.destroy());
      response.on('close', () => stream.destroy());
      stream.pipe(response);
    });
  };
  return {
    name: 'local-orthophoto-tiles',
    configureServer: install,
    configurePreviewServer: install,
  };
}
