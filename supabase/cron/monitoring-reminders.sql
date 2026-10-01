-- Run in the Supabase SQL Editor only after the Edge Function and both Vault
-- secrets have been created. No credential values belong in this file.
-- UTC 00:00, 04:00, 08:00 = Asia/Manila 08:00, 12:00, 16:00, Mon-Fri.
-- Same-day retries are safe because sent reminders are uniquely keyed.
SELECT cron.schedule(
  'mangrovision-monitoring-reminders',
  '0 0,4,8 * * 1-5',
  $$
  SELECT net.http_post(
    url := (SELECT decrypted_secret FROM vault.decrypted_secrets
            WHERE name = 'mangrovision_project_url') || '/functions/v1/monitoring-reminders',
    headers := jsonb_build_object(
      'Content-Type', 'application/json',
      'x-mangrovision-job-token',
      (SELECT decrypted_secret FROM vault.decrypted_secrets
       WHERE name = 'mangrovision_reminder_token')
    ),
    body := '{}'::jsonb,
    timeout_milliseconds := 30000
  ) AS request_id;
  $$
);
