# Monitoring email setup

MangroVision sends one LGU digest for each monitoring date, not one email per
seedling. Visits recur on planting day + 14, +28, +42, and so on. A visit whose
nominal date is Saturday or Sunday is scheduled for the following Monday; the
next round is still anchored to the original planting date. Email phases are
three calendar days before, one calendar day before, and on the scheduled visit
date, but any phase landing on Saturday or Sunday is skipped, not moved.
In-app notices remain one day before, on the day, and when overdue.

For example, if day 14 is Sunday, September 27, the visit moves to Monday,
September 28. Its emails go out Friday, September 25 (three days before the
visit) and Monday, September 28 (due day). The Sunday one-day email is skipped.
The following round is still calculated from planting day + 28, not
14 days after that Monday visit.

## 1. Set up Brevo

1. Create or sign in to a Brevo account. The account login can be
   `mangrovision.lgu@gmail.com`; that is also the **recipient** for reminders.
2. In Brevo, open **Settings > Senders, Domains, IPs > Senders > Add a sender**.
   Set **From name** to `MangroVision LGU` and, for the initial test, **From
   email** to `mangrovision.lgu@gmail.com`. Save it and enter the verification
   code sent to that inbox. Brevo may replace a free Gmail *From* address
   with a Brevo domain in delivered messages. For production, use an address on
   a domain you own and authenticate that domain in Brevo; use that verified
   address for `SMTP_FROM`.
3. Go to **Settings > SMTP & API > SMTP**, generate a **Standard SMTP key**
   named `MangroVision monitoring`, and copy both the **SMTP login** shown on
   that page and the **SMTP key** shown at creation. The SMTP key is not the Brevo login
   password and not an API key. Store it privately; never paste it into chat or
   commit it to Git. If Brevo says transactional sending is not activated for
   the account, complete that account activation before testing.

Brevo's SMTP setup: <https://help.brevo.com/hc/en-us/articles/7924908994450-Send-transactional-emails-using-Brevo-SMTP>

## 2. Configure the backend, not the React client

Set these values in the backend host's private environment. For local testing,
put them in the repository-root `.env` file, which is loaded by
`mangrovision_db.config`. Do not edit the React client's `.env` files.

```dotenv
LGU_REMINDER_EMAIL=mangrovision.lgu@gmail.com
SMTP_HOST=smtp-relay.brevo.com
SMTP_PORT=587
SMTP_USERNAME=<copy the SMTP login from Brevo>
SMTP_PASSWORD=<copy the SMTP key from Brevo>
SMTP_FROM=mangrovision.lgu@gmail.com
```

The job also needs the same `DATABASE_URL` and database schema/settings as the
MangroVision backend. This feature does not add a new database migration; the
existing `staff_notifications` and `email_reminders` tables must already be
present from the application's migrations.

The `SMTP_FROM` value above is for the initial verified-Gmail test. Once you
have an authenticated domain and verified sender on it, replace `SMTP_FROM`
with that sender address; `LGU_REMINDER_EMAIL` stays the LGU Gmail inbox.

## 3. Verify SMTP without changing monitoring records

From the repository root, run:

```powershell
venv\Scripts\python.exe scripts\send_monitoring_reminders.py --test-email
```

This sends one message titled `MangroVision email setup test` to
`mangrovision.lgu@gmail.com`. Check the inbox, Spam folder, and Brevo's
transactional email logs. It does not create or send monitoring reminders.
On a Linux/cloud host, use that host's Python executable instead of the local
Windows virtual-environment path.

## 4. Schedule the real daily job

For this project's Supabase database, use the
[Supabase Cron + Edge Function setup](supabase-cron-reminders.md). It uses a
**Brevo API key**, not the SMTP key above, and does not require the Windows PC
or app backend to stay running. Complete that guide instead of also
scheduling the Python command here; run only one scheduled sender.

The following Python/SMTP option remains available for a different always-on
cloud worker:

The cloud worker needs this repository's Python code, dependencies, private
environment variables, and network access to the **same PostgreSQL database**
used by MangroVision. A worker cannot read a database that exists only on a
switched-off local PC. The whole web app need not be open for the job to run.

Run this command from the repository root at **08:00 Asia/Manila Monday-Friday**:

```text
python scripts/send_monitoring_reminders.py
```

For schedulers with a timezone setting, use `Asia/Manila` and cron expression
`0 8 * * 1-5`. For UTC-only schedulers, use `0 0 * * 1-5` (08:00 Philippine
time on the same weekday).
Keep the scheduler's failure alerts enabled and retry failed runs later the
same day. The database queue prevents normal repeat runs from sending the
same reminder twice. The job prints `configured`, `sent`, and `failed` counts;
it exits nonzero when SMTP is missing or sending fails.

No Brevo or cloud credentials are stored in this repository.
