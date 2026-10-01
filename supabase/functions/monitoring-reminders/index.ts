/** Supabase Cron invokes this function; Brevo sends the due LGU digests. */

import postgres from "npm:postgres@3.4.9";
import {
  buildMonitoringDigests,
  manilaDay,
  type OrganizationPlanting,
  type SitePlanting,
} from "./logic.ts";

function tokenMatches(provided: string, expected: string): boolean {
  if (!expected) return false;
  const left = new TextEncoder().encode(provided);
  const right = new TextEncoder().encode(expected);
  let difference = left.length ^ right.length;
  for (let index = 0; index < Math.max(left.length, right.length); index++) {
    difference |= (left[index] ?? 0) ^ (right[index] ?? 0);
  }
  return difference === 0;
}

async function sendBrevo(
  apiKey: string, fromEmail: string, recipient: string, subject: string, body: string,
): Promise<void> {
  const response = await fetch("https://api.brevo.com/v3/smtp/email", {
    method: "POST",
    headers: { "api-key": apiKey, "content-type": "application/json", accept: "application/json" },
    body: JSON.stringify({
      sender: { email: fromEmail, name: Deno.env.get("REMINDER_FROM_NAME") ?? "MangroVision LGU" },
      to: [{ email: recipient }],
      subject,
      textContent: body,
      tags: ["mangrovision-monitoring"],
    }),
    signal: AbortSignal.timeout(20_000),
  });
  if (response.status !== 201) throw new Error(`Brevo HTTP ${response.status}`);
}

Deno.serve(async (request: Request): Promise<Response> => {
  if (request.method !== "POST") return new Response("Method not allowed", { status: 405 });
  if (!tokenMatches(request.headers.get("x-mangrovision-job-token") ?? "", Deno.env.get("REMINDER_JOB_TOKEN") ?? "")) {
    return new Response("Unauthorized", { status: 401 });
  }

  // The existing Python backend uses SQLAlchemy's postgresql+psycopg scheme;
  // postgres.js expects the ordinary PostgreSQL URI scheme for the same URL.
  const databaseUrl = (Deno.env.get("REMINDER_DATABASE_URL") ?? "")
    .replace(/^postgresql\+psycopg:\/\//, "postgresql://");
  const apiKey = Deno.env.get("BREVO_API_KEY") ?? "";
  const fromEmail = Deno.env.get("REMINDER_FROM_EMAIL") ?? "";
  const recipient = Deno.env.get("LGU_REMINDER_EMAIL") ?? "mangrovision.lgu@gmail.com";
  if (!databaseUrl || !apiKey || !fromEmail || !recipient.includes("@")) {
    return Response.json({ error: "Reminder job secrets are incomplete." }, { status: 503 });
  }

  // A one-off, authenticated delivery test for the Brevo API key. It does not
  // touch the monitoring database or the durable reminder queue.
  if (new URL(request.url).searchParams.get("test_email") === "1") {
    try {
      await sendBrevo(
        apiKey, fromEmail, recipient, "MangroVision cloud email setup test",
        "The Supabase Edge Function can send monitoring emails through Brevo.",
      );
      return Response.json({ test_email: "sent" });
    } catch (error) {
      console.error("Monitoring test email failed:", error instanceof Error ? error.message : "Unknown error");
      return Response.json({ test_email: "failed" }, { status: 502 });
    }
  }

  // One short-lived connection. The application role has RLS policies on its
  // private schema; no service-role key or public Data API grant is needed.
  const sql = postgres(databaseUrl, {
    max: 1,
    prepare: false,
    ssl: "require",
    connect_timeout: 10,
    idle_timeout: 5,
  });
  const today = manilaDay();
  let sent = 0;
  let failed = 0;
  try {
    const siteRows = await sql<SitePlanting[]>`
      SELECT pe.id::text AS id,
             pe.planting_point_id::text AS planting_point_id,
             pe.project_site_id::text AS project_site_id,
             COALESCE(ps.name, 'Unlinked project site') AS site_name,
             to_char(pe.planted_at AT TIME ZONE 'Asia/Manila', 'YYYY-MM-DD') AS planted_day,
             COALESCE((
               SELECT array_agg(mo.interval_days)
               FROM mangrovision.monitoring_observations mo
               WHERE mo.planting_event_id = pe.id
             ), ARRAY[]::integer[]) AS observed_intervals
      FROM mangrovision.planting_events pe
      JOIN mangrovision.planting_points pp ON pp.id = pe.planting_point_id
      LEFT JOIN mangrovision.project_sites ps ON ps.id = pe.project_site_id
      WHERE pe.planted_at <= CURRENT_TIMESTAMP
        AND pe.closed_at IS NULL
        AND pp.status = 'planted'
        AND pe.id = (
          SELECT newer.id FROM mangrovision.planting_events newer
          WHERE newer.planting_point_id = pe.planting_point_id
          ORDER BY newer.planted_at DESC, newer.id DESC LIMIT 1
        )
        AND NOT EXISTS (
          SELECT 1 FROM mangrovision.point_death_records death
          WHERE death.planting_event_id = pe.id
        )
        AND NOT EXISTS (
          SELECT 1 FROM mangrovision.monitoring_observations dead
          WHERE dead.planting_event_id = pe.id AND dead.status = 'dead'
        )
    `;
    const organizationRows = await sql<OrganizationPlanting[]>`
      SELECT o.id::text AS organization_id,
             o.name AS organization_name,
             to_char(pe.planted_at AT TIME ZONE 'Asia/Manila', 'YYYY-MM-DD') AS planted_day,
             latest.latest_day
      FROM mangrovision.planting_events pe
      JOIN mangrovision.planters p ON p.id = pe.planter_id
      JOIN mangrovision.organizations o ON o.id = p.organization_id
      LEFT JOIN (
        SELECT DISTINCT ON (organization_id) organization_id,
               to_char(monitored_at AT TIME ZONE 'Asia/Manila', 'YYYY-MM-DD') AS latest_day
        FROM mangrovision.organization_monitoring_records
        ORDER BY organization_id, monitored_at DESC, id DESC
      ) latest ON latest.organization_id = o.id
      WHERE pe.planted_at <= CURRENT_TIMESTAMP
        AND COALESCE(pe.closure_reason, '') <> 'completion_reversed'
    `;

    const digests = buildMonitoringDigests(today, siteRows, organizationRows);
    if (new URL(request.url).searchParams.get("dry_run") === "1") {
      return Response.json({ date: today, due: digests.map((item) => item.eventKey), sent: 0 });
    }
    for (const digest of digests) {
      await sql`
        INSERT INTO mangrovision.email_reminders (
          event_key, recipient_email, subject, body, send_on, monitoring_due_date
        ) VALUES (
          ${digest.eventKey}, ${recipient}, ${digest.subject}, ${digest.body},
          ${digest.sendOn}::date, ${digest.dueDay}::date
        )
        ON CONFLICT (recipient_email, event_key) DO UPDATE
          SET subject = EXCLUDED.subject, body = EXCLUDED.body
          WHERE email_reminders.status = 'pending'
      `;
    }

    // Only claim keys still due according to *current* planting and visit data.
    // This prevents a stale pending digest from being sent after monitoring was
    // recorded or planting was reversed before the scheduled send.
    for (const digest of digests) {
      const claimed = await sql`
        WITH chosen AS (
          SELECT id FROM mangrovision.email_reminders
          WHERE send_on = ${today}::date
            AND recipient_email = ${recipient}
            AND event_key = ${digest.eventKey}
            AND (
              status = 'pending'
              OR (status = 'sending' AND claimed_at < CURRENT_TIMESTAMP - INTERVAL '15 minutes')
            )
          ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED
        )
        UPDATE mangrovision.email_reminders reminder
        SET status = 'sending', claimed_at = CURRENT_TIMESTAMP, attempts = attempts + 1
        FROM chosen WHERE reminder.id = chosen.id
        RETURNING reminder.id, reminder.subject, reminder.body
      `;
      if (claimed.length === 0) continue;
      const reminder = claimed[0];
      try {
        await sendBrevo(apiKey, fromEmail, recipient, reminder.subject, reminder.body);
        await sql`
          UPDATE mangrovision.email_reminders
          SET status = 'sent', sent_at = CURRENT_TIMESTAMP, last_error = NULL
          WHERE id = ${reminder.id}
        `;
        sent++;
      } catch (error) {
        const reason = error instanceof Error ? error.message.slice(0, 500) : "Unknown delivery error";
        await sql`
          UPDATE mangrovision.email_reminders
          SET status = 'pending', last_error = ${reason}
          WHERE id = ${reminder.id}
        `;
        failed++;
      }
    }
    return Response.json({ date: today, queued: digests.length, sent, failed }, {
      status: failed ? 502 : 200,
    });
  } catch (error) {
    console.error("Monitoring reminder job failed:", error instanceof Error ? error.message : "Unknown error");
    return Response.json({ error: "Monitoring reminder job failed." }, { status: 500 });
  } finally {
    await sql.end({ timeout: 1 });
  }
});
