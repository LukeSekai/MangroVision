import { createApiReadCache } from './apiReadCache';

const nativeFetch = window.fetch.bind(window);

function readCookie(name) {
  const prefix = `${encodeURIComponent(name)}=`;
  const item = document.cookie.split('; ').find((entry) => entry.startsWith(prefix));
  return item ? decodeURIComponent(item.slice(prefix.length)) : '';
}

function sanitizedUrl(input) {
  const raw = input instanceof Request ? input.url : String(input);
  const url = new URL(raw, window.location.origin);
  // Older components may still construct token query parameters. Strip them
  // centrally so opaque credentials can never reach logs or browser history.
  url.searchParams.delete('token');
  return url.toString();
}

export function installSecureFetch() {
  const apiOrigin = new URL(import.meta.env.VITE_API_BASE || window.location.origin, window.location.origin).origin;
  const cache = createApiReadCache({
    fetcher: nativeFetch,
    origin: apiOrigin,
    onMutation: () => window.dispatchEvent(new Event('mv:data-changed')),
    onSessionChange: () => window.dispatchEvent(new Event('mv:session-changed')),
  });
  window.addEventListener('mv:invalidate-reads', () => cache.invalidate());
  // Another tab may record a visit or complete planting. Drop its sibling's
  // cache on focus; mounted pages refresh through the same deduplicated reads.
  window.addEventListener('focus', () => {
    cache.invalidate();
    window.dispatchEvent(new Event('mv:data-changed'));
  });
  window.fetch = (input, init = {}) => {
    const method = String(init.method || (input instanceof Request ? input.method : 'GET')).toUpperCase();
    const headers = new Headers(input instanceof Request ? input.headers : undefined);
    new Headers(init.headers || {}).forEach((value, key) => headers.set(key, value));
    if (!['GET', 'HEAD', 'OPTIONS', 'TRACE'].includes(method)) {
      const csrfToken = readCookie('mv_csrf');
      if (csrfToken) headers.set('X-CSRF-Token', csrfToken);
    }

    const url = sanitizedUrl(input);
    const safeInput = input instanceof Request ? new Request(url, input) : url;
    return cache.fetch(safeInput, {
      ...init,
      headers,
      credentials: 'include',
    });
  };
}
