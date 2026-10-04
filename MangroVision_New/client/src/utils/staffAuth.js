const API = import.meta.env.VITE_API_BASE || '';

export async function staffAuthRequest(path, body) {
  const response = await fetch(`${API}/api/auth/${path}`, {
    method: body === undefined ? 'GET' : 'POST',
    headers: body === undefined ? {} : { 'Content-Type': 'application/json' },
    body: body === undefined ? undefined : JSON.stringify(body),
    cache: 'no-store',
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    const error = new Error(typeof data.detail === 'string' ? data.detail : 'Please check the form and try again.');
    error.retryAfter = Number(response.headers.get('Retry-After') || 0);
    if (error.retryAfter) error.message += ` Try again in ${error.retryAfter} seconds.`;
    throw error;
  }
  return data;
}
