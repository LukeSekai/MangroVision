// A short-lived, memory-only cache for shared workspace reads. Mutation forms,
// authentication, visit baselines, and planter/share links always hit the API.
export const WORKSPACE_READ_PATHS = new Set([
  '/api/analyses/stats', '/api/planters/map-points',
  '/api/monitoring/organizations', '/api/project-sites',
  '/api/zones/forbidden', '/api/zones/eroded', '/api/zones/sites',
  '/api/zones/sites/mortality', '/api/zones/warnings',
  '/api/dashboard/overview', '/api/dashboard/operations', '/api/dashboard/ecology',
  '/api/dashboard/sites', '/api/dashboard/settings',
  '/api/dashboard/record-notices',
  '/api/planting-schedules', '/api/monitoring/organization-records',
  '/api/planters', '/api/planters/', '/api/planters/dashboard',
  '/api/assignments', '/api/assignments/', '/api/analyses', '/api/analyses/',
]);

function isWorkspaceRead(path) {
  return WORKSPACE_READ_PATHS.has(path)
    || /^\/api\/analyses\/\d+(\/points)?$/.test(path)
    || /^\/api\/assignments\/\d+\/points$/.test(path);
}

function waitForRead(promise, signal) {
  if (!signal) return promise;
  return new Promise((resolve, reject) => {
    const abort = () => reject(signal.reason || new DOMException('Aborted', 'AbortError'));
    signal.addEventListener('abort', abort, { once: true });
    promise.then(resolve, reject).finally(() => signal.removeEventListener('abort', abort));
    if (signal.aborted) abort();
  });
}

export function createApiReadCache({ fetcher, origin, ttl = 30_000, now = Date.now,
  onMutation = () => {}, onSessionChange = () => {} }) {
  const entries = new Map();
  let generation = 0;
  let session = 0;

  function invalidate({ sessionChanged = false } = {}) {
    generation += 1;
    entries.clear();
    if (sessionChanged) {
      session += 1;
      onSessionChange();
    }
  }

  async function request(input, init = {}) {
    const url = new URL(input instanceof Request ? input.url : input, origin);
    const method = String(init.method || (input instanceof Request ? input.method : 'GET')).toUpperCase();
    const api = url.origin === origin && url.pathname.startsWith('/api/');
    // PDF rendering accepts a snapshot by POST but does not change records.
    // Announcing a mutation would refresh the report and cancel its download.
    const reportExport = method === 'POST' && url.pathname === '/api/export/report/pdf';
    const mutation = api && !reportExport && !['GET', 'HEAD', 'OPTIONS', 'TRACE'].includes(method);
    const auth = /^\/api\/(auth|planter-auth)(\/|$)/.test(url.pathname);
    if (mutation) {
      // Invalidate before AND after: reads during a write cannot survive its commit.
      invalidate({ sessionChanged: auth });
      try {
        const response = await fetcher(input, init);
        invalidate({ sessionChanged: auth });
        if (response.ok && !auth) onMutation();
        return response;
      } catch (error) {
        invalidate();
        throw error;
      }
    }
    const signal = init.signal || (input instanceof Request ? input.signal : null);
    if (!api || method !== 'GET' || !isWorkspaceRead(url.pathname)) {
      return fetcher(input, init);
    }
    if (signal?.aborted) throw signal.reason || new DOMException('Aborted', 'AbortError');
    if (init.cache === 'no-store') {
      entries.delete(url.href);
      return fetcher(input, init);
    }
    // Cancellation belongs to each caller. Unmounting one page must not abort
    // another page's shared read (including React StrictMode's effect replay).
    return waitForRead(read(input, { ...init, signal: null }), signal);
  }

  async function read(input, init) {
    const url = new URL(input instanceof Request ? input.url : input, origin);
    const currentSession = session;
    const key = url.href;
    const force = ['reload', 'no-store', 'no-cache'].includes(init.cache);
    let entry = entries.get(key);
    if (force || !entry || (entry.response && entry.expires <= now())) {
      const startedGeneration = generation;
      entry = {};
      // Date/site filters create distinct dashboard keys. Bound retained
      // responses even when users explore many combinations in one session.
      if (entries.size >= 64) entries.delete(entries.keys().next().value);
      entries.set(key, entry);
      entry.promise = Promise.resolve().then(() => fetcher(input, init)).then(async (response) => {
        if (response.status === 401 || response.status === 403) {
          invalidate({ sessionChanged: true });
          return response;
        }
        // Fetch resolves at headers. Wait for the complete body before checking
        // generation/session so a write during a slow download cannot restore
        // an old snapshot. Keep the original Response metadata and readable body.
        if (response.ok) await response.clone().arrayBuffer();
        if (entries.get(key) === entry && generation === startedGeneration) {
          if (response.ok) {
            entry.response = response;
            entry.expires = now() + ttl;
          } else {
            entries.delete(key);
          }
        }
        return response;
      }).catch((error) => {
        if (entries.get(key) === entry) entries.delete(key);
        throw error;
      });
      entry.generation = startedGeneration;
    }
    const response = entry.response || await entry.promise;
    // Never deliver an old user's in-flight data into a newly signed-in workspace.
    if (currentSession !== session) throw new DOMException('Session changed. Reload the workspace.', 'AbortError');
    if (entry.generation !== generation || (entries.has(key) && entries.get(key) !== entry)) {
      return read(input, { ...init, cache: 'default' });
    }
    return response.clone();
  }

  return { fetch: request, invalidate };
}
