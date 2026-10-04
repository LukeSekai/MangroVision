/** Private verification sender: the database chooses the recipient and purpose. */

export type Challenge = {
  user_email: string;
  purpose: 'login' | 'recovery' | 'settings';
  code_hash: string;
  request_key: string;
  delivered: boolean;
  consumed_at: string | null;
  email_claimed_at: string | null;
  expires_at: string;
  attempts: number;
};

export type Dependencies = {
  authorizationToken: string;
  configured: boolean;
  findChallenge: (tokenHash: string) => Promise<Challenge | undefined>;
  claimChallenge: (tokenHash: string) => Promise<boolean>;
  sendEmail: (recipient: string, purpose: Challenge['purpose'], code: string) => Promise<void>;
};

const encoder = new TextEncoder();
function hex(bytes: ArrayBuffer): string {
  return Array.from(new Uint8Array(bytes), (byte) => byte.toString(16).padStart(2, '0')).join('');
}

export async function digest(value: string): Promise<string> {
  return hex(await crypto.subtle.digest('SHA-256', encoder.encode(value)));
}

export async function codeDigest(token: string, code: string): Promise<string> {
  const key = await crypto.subtle.importKey('raw', encoder.encode(token), { name:'HMAC', hash:'SHA-256' }, false, ['sign']);
  return hex(await crypto.subtle.sign('HMAC', key, encoder.encode(code)));
}

function matches(left: string, right: string): boolean {
  let difference = left.length ^ right.length;
  for (let index = 0; index < Math.max(left.length, right.length); index++) {
    difference |= (left.charCodeAt(index) || 0) ^ (right.charCodeAt(index) || 0);
  }
  return difference === 0;
}

export async function handleVerification(request: Request, deps: Dependencies): Promise<Response> {
  if (request.method !== 'POST') return new Response('Method not allowed', { status:405 });
  const authorization = request.headers.get('x-mangrovision-staff-email-token') ?? '';
  if (!deps.authorizationToken || authorization.length > 256 || !matches(authorization, deps.authorizationToken)) {
    return new Response('Unauthorized', { status:401 });
  }
  if (!deps.configured) return Response.json({ error:'Email service unavailable.' }, { status:503 });
  try {
    const raw = await request.text();
    if (raw.length > 4096) return new Response('Request too large', { status:413 });
    let body;
    try { body = JSON.parse(raw); } catch { return new Response('Invalid request', { status:400 }); }
    const token = body?.challenge_token, code = body?.code;
    if (typeof token !== 'string' || !/^[A-Za-z0-9_-]{43}$/.test(token)
        || typeof code !== 'string' || !/^[0-9]{6}$/.test(code)) {
      return new Response('Invalid request', { status:400 });
    }
    const tokenHash = await digest(token);
    const row = await deps.findChallenge(tokenHash);
    if (!row || row.delivered || row.consumed_at || row.email_claimed_at || row.attempts >= 5
        || Date.parse(row.expires_at) <= Date.now() || !Number.isFinite(Date.parse(row.expires_at))
        || !['login','recovery','settings'].includes(row.purpose)
        || !/^[^\s@]+@[^\s@]+\.[^\s@]+$/.test(row.user_email)
        || /\.(local|invalid|test)$/i.test(row.user_email)
        || !matches(await digest(row.user_email.trim().toLowerCase()), row.request_key)
        || !matches(await codeDigest(token, code), row.code_hash)) {
      return new Response('Verification unavailable', { status:409 });
    }
    // Claim once in a short SQL write, then send outside the transaction.
    if (!await deps.claimChallenge(tokenHash)) return new Response('Verification already claimed', { status:409 });
    await deps.sendEmail(row.user_email, row.purpose, code);
    return Response.json({ sent:true });
  } catch {
    // Provider/SQL exceptions may contain credentials or request data.
    console.error('Staff verification email failed.');
    return Response.json({ error:'Email delivery failed.' }, { status:502 });
  }
}
