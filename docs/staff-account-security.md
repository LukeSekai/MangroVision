# Staff email verification and account recovery

Staff accounts remain in the private `mangrovision.users` schema in the existing
Supabase PostgreSQL database. This is MangroVision's cookie authentication,
not a migration to Supabase Auth. The database and configured SMTP sender work
independently of a Codex chat session. Organization field accounts keep their
existing login flow.

## Sign in

1. Enter your staff username and password.
2. Check your registered email for the six-digit sign-in code.
3. Enter the code to open the workspace.

The initial button is labelled **Sign in**. Missing inputs and rejected
credentials show a correction beside the relevant username or password field.
An unrecognized username is distinguished from an incorrect password. The form
uses a staff username; the registered email receives the verification code.
Code-delivery failures and account-change conflicts stay at form level.

Entering a password alone creates no staff session. Each code expires after
10 minutes, can be consumed once, and permits five incorrect attempts. Resends
require 60 seconds and invalidate the previous code. An email address may
receive at most 10 verification emails per hour across the three workflows.
Additional identity and client-address limits are persisted in PostgreSQL.
Limits use the request's validated client address, never arbitrary forwarded
headers.

The existing `admin` account's placeholder address was linked to the approved
LGU mailbox, `mangrovision.lgu@gmail.com`. Other accounts need their own real,
provisioned addresses; `.local`, `.invalid`, and `.test` addresses cannot receive
verification. Never redirect every account's codes to a shared sender address.

## Change credentials while signed in

Select your avatar at the bottom of the sidebar to open **Account settings**.
Enter your current password, a new password, and its confirmation.
Request a code and enter it to save. The pending change is not applied
until verification succeeds. Changing the password signs out all sessions
on that account and invalidates outstanding challenges. Sign in again afterward.

The username, email, user ID, role, and associated planting records are preserved.
Credential settings and recovery can only replace the password. The API rejects
username-change fields, including requests from older clients. An older pending
settings challenge containing only a username cannot be completed; request a
new code with a new password.

## Recover from the login page

Choose **Forgot password?**, then **Send recovery code**. The server
automatically uses `mangrovision.lgu@gmail.com`; the form has no email input and
the API accepts no recipient override. Enter the recovery code, your new
password, and its confirmation. Recovery does not sign you
in automatically. This login-page recovery flow is for the LGU admin account.

New passwords require 12–128 characters.

## Deployment and delivery

Apply the existing Alembic migration workflow:

```powershell
.\venv\Scripts\python.exe -m alembic upgrade head
```

Revision `20261004_0011` adds `staff_auth_challenges` and `staff_auth_limits`.
Revision `20261004_0012` adds a delivery claim and the private cloud transport.
Both have RLS enabled; only the backend database role can use them. The
Supabase `anon` and `authenticated` roles have no table privileges. Challenge
secrets and session tokens stay in HttpOnly cookies. PostgreSQL stores their
hashes and HMAC code digests, never plaintext codes. Pending password changes
contain only Argon2id hashes. Authenticated and pending verification requests
receive the existing CSRF protections; sensitive reads use `no-store`.

On the configured Supabase project, email uses the private **staff-verification**
Edge Function and the existing Brevo HTTPS API setup. SMTP ports on this laptop
are unreachable, so this avoids depending on them. The function validates an
existing challenge's HMAC, takes the recipient and purpose from the private
database, and atomically claims delivery to prevent duplicate sends. It accepts
no caller-provided recipient or subject. Monitoring schedules and their sender
function remain unchanged.

Sign-in, recovery, and account-change messages use the shared branded HTML
template in `supabase/functions/staff-verification/email-template.json`, with
inline styles and a phone layout for Gmail. Both cloud delivery and local SMTP
include a plain-text alternative. The code stays copyable as six consecutive
digits; the email contains no external assets or temporary tunnel links.

The new `STAFF_AUTH_EMAIL_TOKEN` is in Supabase Edge Function Secrets; its matching
`mangrovision_staff_email_token` is in Vault. The backend can obtain only this
specific transport through `staff_email_transport()`. It has no general access
to Vault. No additional API key is needed in the local `.env` or browser.
The sender reuses `BREVO_API_KEY`, `REMINDER_DATABASE_URL`, and
`REMINDER_FROM_EMAIL` already configured for the cloud project.

For another deployment, deploy `supabase/functions/staff-verification`, configure
those private secrets, and use the existing Alembic migration workflow.
`STAFF_EMAIL_TRANSPORT` defaults to `auto`, preferring the configured cloud sender
and using SMTP only when no cloud transport is configured. `cloud` requires the
cloud sender; `smtp` explicitly uses the original SMTP transport. A configured
cloud sender's delivery failure never falls back to a second send.
See [monitoring-email-setup.md](monitoring-email-setup.md) for SMTP configuration
on networks where it is available. Keep credentials private. Email delivery errors
fail closed: the app never skips verification or substitutes a default code.

Credential changes and verified sign-ins are recorded in activity history;
codes, passwords, hashes, and challenge secrets are excluded from that history.

Security regressions use temporary PostgreSQL tables and a fake sender:

```powershell
$env:MANGROVISION_RUN_TEMP_ACCOUNT_TESTS='1'
.\venv\Scripts\python.exe -m pytest tests/test_staff_auth_postgres.py -q -p no:cacheprovider
node --test supabase/functions/staff-verification/logic.test.ts
```
