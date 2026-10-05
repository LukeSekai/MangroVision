class AnalysisRequestError extends Error {
  constructor(message, retryable = false) {
    super(message);
    this.retryable = retryable;
  }
}

// Gateways can return an HTML error page even though the API returns JSON.
// Do not show their markup or a JSON parser exception to the user.
export async function readAnalysisResponse(response) {
  const status = response.status;
  const temporary = status >= 500 || status === 429;
  if (!response.headers.get('content-type')?.includes('json')) {
    const message = status === 413 ? 'This image is too large to upload.'
      : `The website could not reach the laptop analysis server (HTTP ${status}). Check that the server and tunnel are running.`;
    throw new AnalysisRequestError(message, temporary || response.ok);
  }
  let payload;
  try {
    payload = await response.json();
  } catch {
    throw new AnalysisRequestError('The connection was interrupted while receiving the analysis response.', true);
  }
  if (!response.ok) {
    throw new AnalysisRequestError(typeof payload?.detail === 'string' ? payload.detail
      : `Analysis request failed (HTTP ${status}).`, temporary);
  }
  if (!payload || typeof payload !== 'object' || Array.isArray(payload)) {
    throw new AnalysisRequestError('The analysis server returned an invalid response.', true);
  }
  return payload;
}

async function fetchAnalysisJson(fetcher, url, options, timeout) {
  const controller = new AbortController();
  const signal = options.signal;
  const abort = () => controller.abort();
  signal?.addEventListener('abort', abort, { once: true });
  if (signal?.aborted) controller.abort();
  const timer = setTimeout(abort, timeout);
  try {
    const response = await fetcher(url, { ...options, signal: controller.signal });
    return await readAnalysisResponse(response);
  } catch (error) {
    if (signal?.aborted) throw new DOMException('Aborted', 'AbortError');
    if (error instanceof AnalysisRequestError) throw error;
    throw new AnalysisRequestError('Connection to the laptop analysis server was interrupted. Check its internet connection.', true);
  } finally {
    clearTimeout(timer);
    signal?.removeEventListener('abort', abort);
  }
}

// Retries carry the same request ID, so a lost acknowledgement cannot start
// another AI run or append another processing activity for this upload.
export async function startProcessingJob({ api = '', formData, signal, onRetry, fetcher = fetch,
  interval = 2000, requestId = crypto.randomUUID() }) {
  formData.set('request_id', requestId);
  for (let attempt = 0; ; attempt += 1) {
    try {
      const started = await fetchAnalysisJson(fetcher, `${api}/api/analyses/jobs`, {
        method: 'POST', body: formData, signal,
      }, 90000);
      if (!started?.job_id) throw new AnalysisRequestError('The server did not return an analysis job. Reload the website and try again.');
      return started;
    } catch (error) {
      if (signal?.aborted || !error.retryable || attempt >= 2) throw error;
      onRetry?.();
      await pause(interval * (attempt + 1), signal);
    }
  }
}

// Progress requests retry without uploading the image again.
export async function waitForProcessingJob({ api = '', jobId, signal, onProgress, fetcher = fetch, interval = 2000 }) {
  let failures = 0;
  while (true) {
    signal?.throwIfAborted();
    let job;
    try {
      job = await fetchAnalysisJson(fetcher, `${api}/api/analyses/jobs/${encodeURIComponent(jobId)}`, {
        signal, cache: 'no-store',
      }, 45000);
    } catch (error) {
      if (signal?.aborted || !error.retryable) throw error;
      failures += 1;
      if (failures >= 30) throw new Error('Lost connection to the laptop server. Check that it and the tunnel are still running.');
      onProgress?.({ stage: 'Reconnecting to analysis...', pct: undefined });
      await pause(Math.min(interval * failures, 10000), signal);
      continue;
    }
    failures = 0;
    if (job.status === 'succeeded') return job.payload;
    if (job.status === 'failed') throw new Error(job.detail || 'Processing failed.');
    if (job.status !== 'running') throw new Error('The server returned an invalid analysis status. Reload the website and try again.');
    onProgress?.(job);
    await pause(interval, signal);
  }
}

function pause(milliseconds, signal) {
  return new Promise((resolve, reject) => {
    const finish = () => {
      signal?.removeEventListener('abort', abort);
      resolve();
    };
    const timer = setTimeout(finish, milliseconds);
    const abort = () => {
      clearTimeout(timer);
      signal?.removeEventListener('abort', abort);
      reject(new DOMException('Aborted', 'AbortError'));
    };
    signal?.addEventListener('abort', abort, { once: true });
    if (signal?.aborted) abort();
  });
}
