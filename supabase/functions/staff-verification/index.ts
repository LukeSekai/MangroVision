import postgres from 'npm:postgres@3.4.9';
import { handleVerification, type Challenge } from './logic.ts';
import { verificationEmail } from './email.ts';

Deno.serve(async (request: Request): Promise<Response> => {
  const databaseUrl = (Deno.env.get('REMINDER_DATABASE_URL') ?? '').replace(/^postgresql\+psycopg:\/\//, 'postgresql://');
  const apiKey = Deno.env.get('BREVO_API_KEY') ?? '';
  const fromEmail = Deno.env.get('REMINDER_FROM_EMAIL') ?? '';
  const sql = postgres(databaseUrl || 'postgres://localhost/unused', {
    max:1, prepare:false, ssl:'require', connect_timeout:10, idle_timeout:5,
  });
  try {
    return await handleVerification(request, {
      authorizationToken: Deno.env.get('STAFF_AUTH_EMAIL_TOKEN') ?? '',
      configured: Boolean(databaseUrl && apiKey && fromEmail),
      findChallenge: async (tokenHash) => {
        const rows = await sql<Challenge[]>`
          SELECT challenge.code_hash, challenge.request_key, challenge.purpose,
                 challenge.delivered, challenge.consumed_at, challenge.expires_at,
                 challenge.attempts, challenge.email_claimed_at, staff.email AS user_email
          FROM mangrovision.staff_auth_challenges challenge
          JOIN mangrovision.users staff ON staff.id = challenge.user_id
          WHERE challenge.token_hash = ${tokenHash}
        `;
        return rows[0];
      },
      claimChallenge: async (tokenHash) => {
        const rows = await sql`
          UPDATE mangrovision.staff_auth_challenges SET email_claimed_at = CURRENT_TIMESTAMP
          WHERE token_hash = ${tokenHash} AND email_claimed_at IS NULL
            AND consumed_at IS NULL AND NOT delivered AND attempts < 5
            AND expires_at > CURRENT_TIMESTAMP RETURNING token_hash
        `;
        return rows.length === 1;
      },
      sendEmail: async (recipient, purpose, code) => {
        const email = verificationEmail(purpose, code);
        const response = await fetch('https://api.brevo.com/v3/smtp/email', {
          method:'POST',
          headers: { 'api-key':apiKey, 'content-type':'application/json', accept:'application/json' },
          body: JSON.stringify({
            sender: { email:fromEmail, name:'MangroVision' }, to:[{ email:recipient }],
            ...email,
            tags:['mangrovision-staff-verification'],
          }),
          signal:AbortSignal.timeout(20_000),
        });
        if (response.status !== 201) throw new Error('Email provider rejected the request.');
      },
    });
  } finally { await sql.end({ timeout:3 }); }
});
