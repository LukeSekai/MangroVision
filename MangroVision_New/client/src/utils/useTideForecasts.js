import { useEffect, useRef, useState } from 'react';

const API = import.meta.env.VITE_API_BASE || '';

// Bounded concurrency, cancellation and per-location error states. Never reuse
// one site's forecast to colour another site's activities.
export default function useTideForecasts(locationKeys, retryKey) {
  const [records, setRecords] = useState({});
  const loaded = useRef(new Set());
  const signature = [...new Set(locationKeys.filter(Boolean))].sort().join(';');
  useEffect(() => {
    const controller = new AbortController();
    const keys = signature.split(';').filter(Boolean)
      .filter((key) => !loaded.current.has(`${retryKey}:${key}`));
    if (!keys.length) return undefined;
    queueMicrotask(() => {
      if (!controller.signal.aborted) setRecords((current) => ({
        ...current,
        ...Object.fromEntries(keys.map((key) => [key, { ...current[key], loading: true, error: '' }])),
      }));
    });
    let index = 0;
    async function worker() {
      while (index < keys.length && !controller.signal.aborted) {
        const key = keys[index++];
        const [lat, lon] = key.split(',');
        const requestController = new AbortController();
        const abortRequest = () => requestController.abort();
        controller.signal.addEventListener('abort', abortRequest, { once: true });
        const timeout = window.setTimeout(abortRequest, 45_000);
        let record;
        try {
          const response = await fetch(`${API}/api/tides/forecast?lat=${lat}&lon=${lon}&days=7`, { signal: requestController.signal });
          const payload = await response.json();
          if (!response.ok || !payload || typeof payload !== 'object' || Array.isArray(payload)) throw new Error('Could not load the tide forecast.');
          record = { payload, loading: false, error: payload.available === true ? '' : (payload.message || 'Tide forecast unavailable.') };
        } catch (error) {
          record = { loading: false, error: error.name === 'AbortError'
            ? 'The water levels took too long to load. Please try again.'
            : error instanceof SyntaxError ? 'We could not read the water levels. Please try again.'
              : error.message || 'Tide forecast unavailable.' };
        } finally {
          window.clearTimeout(timeout);
          controller.signal.removeEventListener('abort', abortRequest);
        }
        if (!controller.signal.aborted) {
          loaded.current.add(`${retryKey}:${key}`);
          setRecords((current) => ({
            ...current,
            // Keep displayed readings during a refresh or temporary failure.
            [key]: record.payload ? record : { ...current[key], ...record },
          }));
        }
      }
    }
    for (let i = 0; i < Math.min(3, keys.length); i += 1) void worker();
    return () => controller.abort();
  }, [signature, retryKey]);
  return records;
}
