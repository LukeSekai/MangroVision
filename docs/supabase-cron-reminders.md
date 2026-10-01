# Supabase Cron monitoring emails (Free-plan setup)

This is the cloud alternative to leaving the Windows PC running. Supabase Cron
calls an Edge Function at **08:00, 12:00, and 16:00 Asia/Manila, Monday-Friday**. The function
reads the same private `mangrovision` database used by the app and sends one
LGU digest through Brevo's HTTP API when a visit is due in 3 days, tomorrow,
or today and that email date is a weekday. Weekend reminder phases are
skipped, not moved to Friday. For a Monday visit, this means a Friday 3-day
email and a Monday due-day email, but no Sunday email. The extra same-day runs
retry failures; the `email_reminders` queue
prevents ordinary duplicate sends. Saturday/Sunday visits move to Monday, and
the next 14-day round still starts from the original planting date.

Supabase Free includes 500,000 Edge Function invocations per month; this job
uses about 65. Brevo Free includes 300 email sends per day. Those limits are
more than enough for one LGU digest per reminder date, but **Supabase Free
projects may be paused after low activity**. A paused project will not send
reminders until restored. Do not treat this as guaranteed delivery for a
critical LGU operation without monitoring or a paid always-on plan.

## 1. Prepare Brevo and a private job token

1. In Brevo, open **Settings > SMTP & API > API Keys** and generate an **API
   key** for `MangroVision Supabase reminders`. This is *different* from the
   Standard SMTP key that worked with the local Python test. Keep both private.
2. Confirm your sender email is verified in Brevo. Use the same address that
   you currently set as `SMTP_FROM` in the repository-root `.env`.
3. In PowerShell, from the repository root, generate a random job token:

   ```powershell
   venv\Scripts\python.exe -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

   Copy it into a private password manager. Do not paste any key or token into
   the app, chat, source files, or screenshots.

## 2. Add Supabase Edge Function secrets

In your Supabase project, open **Edge Functions > Secrets**. Add these exact
names and values:

| Name | Value |
| --- | --- |
| `BREVO_API_KEY` | The new Brevo **API key**, not the SMTP key. |
| `REMINDER_DATABASE_URL` | The complete `DATABASE_URL` value from the repository-root `.env`. This project already uses Supabase's port-5432 session pooler; the function accepts its `postgresql+psycopg://` prefix. |
| `REMINDER_FROM_EMAIL` | The verified Brevo sender address used as `SMTP_FROM`. |
| `LGU_REMINDER_EMAIL` | `mangrovision.lgu@gmail.com` |
| `REMINDER_JOB_TOKEN` | The random token generated in step 1. |
| `REMINDER_FROM_NAME` | Optional: `MangroVision LGU`. |

Use the existing restricted `mangrovision` database user, not a Supabase
service-role key or the browser's publishable key. Do not add these values to
`MangroVision_New/client/.env*` or commit them to Git.

## 3. Deploy the Edge Function

This repository contains two function files, so deploy from the repository
root with the Supabase CLI (the Dashboard code editor is not needed). Node.js
20+ is required; this Windows PC already has Node.js. In PowerShell:

```powershell
npx --yes supabase@latest login
npx --yes supabase@latest functions deploy monitoring-reminders --project-ref YOUR_PROJECT_REF --no-verify-jwt --use-api
```

Replace `YOUR_PROJECT_REF` with the project ID in the Supabase Dashboard URL:
`https://supabase.com/dashboard/project/YOUR_PROJECT_REF`. Login opens a
browser. `--use-api` bundles on Supabase's side, so Docker and a local
Supabase database are not required.

The `--no-verify-jwt` setting is intentional: Cron sends the custom
`x-mangrovision-job-token` header instead of a user JWT. The function rejects
requests without the correct token *before* reading the database or sending
email. Do not remove that check.

## 4. Test Brevo delivery from Supabase

Open **Edge Functions > monitoring-reminders > Test** in Supabase. Set:

- Method: `POST`
- Query parameter: `test_email` = `1`
- Header: `x-mangrovision-job-token` = the private token from step 1
- Body: `{}`

Send the request. Expect HTTP 200 with `{"test_email":"sent"}` and a test
message at `mangrovision.lgu@gmail.com`. This sends one actual test email but
does not change monitoring records. If it fails, check **Edge Functions >
monitoring-reminders > Logs** and the Brevo transactional email log; verify
the **API key** and sender address. The earlier Python `--test-email` only
tests SMTP, not this cloud API path.

To check database connectivity and today's schedule *without* creating queue
rows or sending email, repeat the Dashboard test with query parameter
`dry_run` = `1` instead. Expect HTTP 200 with a `date`, a `due` list, and
`sent: 0`.

You may also run a normal `POST` with the same header and no query parameter.
It should return `{"date":"YYYY-MM-DD","queued":0,"sent":0,"failed":0}`
when nothing is due; if monitoring *is* due, it will send real reminders.

## 5. Enable Cron, pg_net, and Vault; create the schedule

1. In Supabase, open **Integrations > Cron** and enable it if prompted.
2. Open **Database > Extensions** and enable `pg_net` if it is not already
   enabled. Supabase Cron uses `pg_cron`; both are needed to call the function.
3. Open **Database > Vault**. Create two secrets with these exact names:
   - `mangrovision_project_url` = `https://YOUR_PROJECT_REF.supabase.co`
     (replace the project ref; no trailing slash).
   - `mangrovision_reminder_token` = the **same** token used for
     `REMINDER_JOB_TOKEN` in Edge Function Secrets.
4. Open **SQL Editor**, paste and run
   [`supabase/cron/monitoring-reminders.sql`](../supabase/cron/monitoring-reminders.sql).
   The file contains *no* secret values. It creates one named Cron job for
   `00:00`, `04:00`, and `08:00` UTC on Monday-Friday, which are 08:00,
   12:00, and 16:00 on Philippine weekdays. Run it only after the function
   and secrets are ready.
5. Check **Integrations > Cron > Jobs / History** for the named job
   `mangrovision-monitoring-reminders`, and **Edge Functions >
   monitoring-reminders > Logs** for actual delivery. Cron's SQL success only
   means the HTTP request was enqueued; the Edge Function log/response shows
   whether email was sent. Brevo's transactional log confirms acceptance.

If you already scheduled `scripts/send_monitoring_reminders.py` elsewhere,
disable that *separate scheduled sender* after the cloud job is verified. You
can keep the local SMTP test command for diagnostics. Do not disable or remove
the app's in-app notification sync.

To stop future cloud runs, use **Integrations > Cron** to disable the named
job. Do not delete any monitoring or reminder records.
