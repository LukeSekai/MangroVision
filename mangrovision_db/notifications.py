"""Durable monitoring notices and daily email reminders."""

from __future__ import annotations

import os
import smtplib
from datetime import date, datetime, time, timedelta
from email.message import EmailMessage
from typing import Any
from zoneinfo import ZoneInfo

from .compat import get_connection

MANILA = ZoneInfo("Asia/Manila")
DEFAULT_RECIPIENT = "mangrovision.lgu@gmail.com"
EMAIL_PHASES = (
    (3, "three_days_before", "in 3 days"),
    (1, "tomorrow", "tomorrow"),
    (0, "today", "today"),
)


def _today() -> date:
    return datetime.now(MANILA).date()


def _staff_users(conn: Any) -> list[int]:
    rows = conn.execute("""
        SELECT id FROM users WHERE lower(role) IN ('admin', 'lgu', 'planner')
    """).fetchall()
    return [int(row["id"]) for row in rows]


def _site_groups(now: date) -> tuple[list[dict], list[dict]]:
    from planting_database import get_due_monitoring_inspections, list_organization_monitoring_summaries

    due = get_due_monitoring_inspections(
        as_of=datetime.combine(now, time(9, 0), tzinfo=MANILA).isoformat(), upcoming_days=3,
    )["observations_due"]
    grouped: dict[tuple[int | None, str], dict] = {}
    for item in due:
        due_day = date.fromisoformat(item["due_at"][:10])
        if due_day > now + timedelta(days=3):
            continue
        site_id = item.get("project_site_id")
        key = (site_id, due_day.isoformat() if due_day >= now else "overdue")
        group = grouped.setdefault(key, {
            "site_id": site_id,
            "site_name": item.get("project_site_name") or "Unlinked project site",
            "due_day": due_day,
            "point_ids": set(),
            "overdue": due_day < now,
        })
        group["point_ids"].add(item.get("planting_point_id") or item["planting_event_id"])
        if due_day < group["due_day"]:
            group["due_day"] = due_day

    organizations = []
    for item in list_organization_monitoring_summaries():
        raw = item.get("next_monitoring_date")
        if not raw or int(item.get("total_planted") or 0) <= 0:
            continue
        due_day = date.fromisoformat(raw)
        if due_day <= now + timedelta(days=3):
            organizations.append({"id": int(item["id"]), "name": item["name"],
                                  "due_day": due_day})
    return list(grouped.values()), organizations


def sync_monitoring_reminders(today: date | None = None) -> dict:
    """Materialize reminders from current planting/visit state; safe to rerun."""
    now = today or _today()
    site_groups, organizations = _site_groups(now)
    recipient = os.getenv("LGU_REMINDER_EMAIL", DEFAULT_RECIPIENT).strip()
    if recipient and ("@" not in recipient or "\n" in recipient or "\r" in recipient):
        raise ValueError("LGU_REMINDER_EMAIL is invalid.")
    conn = get_connection()
    created = 0
    try:
        users = _staff_users(conn)
        notices = []
        for group in site_groups:
            site_key = group["site_id"] if group["site_id"] is not None else "unknown"
            due_day = group["due_day"]
            if group["overdue"]:
                phase = "overdue"
                key = f"site:{site_key}:overdue:{now.isoformat()}"
                title = f"Overdue seedling inspection: {group['site_name']}"
                body = f"{len(group['point_ids'])} planted point(s) still need inspection. Earliest due date: {due_day.isoformat()}."
            else:
                if due_day > now + timedelta(days=1):
                    continue  # In-app reminders remain day-before and day-of.
                phase = "today" if due_day == now else "tomorrow"
                key = f"site:{site_key}:{due_day.isoformat()}:{phase}"
                title = f"Seedling inspection {phase}: {group['site_name']}"
                body = f"{len(group['point_ids'])} planted point(s) are due on {due_day.isoformat()}."
            notices.append((key, "site_monitoring", title, body, "/monitoring", due_day))
        for org in organizations:
            due_day = org["due_day"]
            if due_day > now + timedelta(days=1):
                continue
            phase = "overdue" if due_day < now else "today" if due_day == now else "tomorrow"
            key = f"organization:{org['id']}:{due_day.isoformat()}:{phase if phase != 'overdue' else now.isoformat()}"
            title = f"Organization monitoring {phase}: {org['name']}"
            body = f"The 14-day monitoring visit is due on {due_day.isoformat()}."
            notices.append((key, "organization_monitoring", title, body, "/monitoring", due_day))

        for user_id in users:
            for key, kind, title, body, path, due_day in notices:
                cursor = conn.execute("""
                    INSERT INTO staff_notifications (user_id, event_key, kind, title, body, target_path, due_date)
                    VALUES (?, ?, ?, ?, ?, ?, ?) ON CONFLICT (user_id, event_key) DO NOTHING
                """, (user_id, key, kind, title, body, path, due_day.isoformat()))
                created += max(0, cursor.rowcount)

        email_phases = EMAIL_PHASES if now.weekday() < 5 else ()
        for days_before, phase, phrase in email_phases:
            due_day = now + timedelta(days=days_before)
            site_lines = [
                f"- {group['site_name']}: {len(group['point_ids'])} point(s)"
                for group in site_groups if not group["overdue"] and group["due_day"] == due_day
            ]
            org_lines = [
                f"- {org['name']}"
                for org in organizations if org["due_day"] == due_day
            ]
            if not (site_lines or org_lines) or not recipient:
                continue
            body_parts = [
                f"MangroVision monitoring is due {phrase} ({due_day.isoformat()}, Asia/Manila)."
            ]
            if site_lines:
                body_parts.extend(["", "Site seedling inspections:", *site_lines])
            if org_lines:
                body_parts.extend(["", "Organization monitoring visits:", *org_lines])
            body_parts.extend(["", "Open MangroVision to record the monitoring work."])
            body = "\n".join(body_parts)
            conn.execute("""
                INSERT INTO email_reminders (
                    event_key, recipient_email, subject, body, send_on, monitoring_due_date
                ) VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT (recipient_email, event_key) DO UPDATE
                    SET body = excluded.body
                    WHERE email_reminders.status = 'pending'
            """, (
                f"monitoring:{due_day.isoformat()}:{phase}", recipient,
                f"MangroVision monitoring {phrase} - {due_day.isoformat()}", body,
                now.isoformat(), due_day.isoformat(),
            ))
        conn.commit()
        return {"created_notifications": created, "site_groups": len(site_groups),
                "organization_groups": len(organizations)}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_staff_notifications(user_id: int, *, limit: int = 50) -> dict:
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT id, kind, title, body, target_path, due_date, created_at, read_at
            FROM staff_notifications WHERE user_id = ? ORDER BY id DESC LIMIT ?
        """, (int(user_id), max(1, min(int(limit), 100)))).fetchall()
        unread = conn.execute("""
            SELECT count(*) AS total FROM staff_notifications
            WHERE user_id = ? AND read_at IS NULL
        """, (int(user_id),)).fetchone()
        return {"items": [dict(row) for row in rows], "unread_count": int(unread["total"])}
    finally:
        conn.close()


def mark_notification_read(user_id: int, notification_id: int) -> bool:
    conn = get_connection()
    try:
        cursor = conn.execute("""
            UPDATE staff_notifications SET read_at = COALESCE(read_at, CURRENT_TIMESTAMP)
            WHERE id = ? AND user_id = ?
        """, (int(notification_id), int(user_id)))
        conn.commit()
        return cursor.rowcount > 0
    finally:
        conn.close()


def _send_smtp(recipient: str, subject: str, body: str, *, html_body: str | None = None) -> None:
    host = os.getenv("SMTP_HOST", "").strip()
    sender = os.getenv("SMTP_FROM", "").strip()
    if not host or not sender:
        raise RuntimeError("SMTP_HOST and SMTP_FROM must be configured before email can be sent.")
    port = int(os.getenv("SMTP_PORT", "587"))
    username = os.getenv("SMTP_USERNAME", "").strip()
    password = os.getenv("SMTP_PASSWORD", "")
    if not username or not password:
        raise RuntimeError("SMTP_USERNAME and SMTP_PASSWORD must be configured before email can be sent.")
    message = EmailMessage()
    message["From"] = sender
    message["To"] = recipient
    message["Subject"] = subject
    message.set_content(body)
    if html_body is not None:
        message.add_alternative(html_body, subtype="html")
    transport = smtplib.SMTP_SSL if port == 465 else smtplib.SMTP
    with transport(host, port, timeout=20) as client:
        if port != 465:
            client.starttls()
        client.login(username, password)
        client.send_message(message)


def send_pending_reminders(*, today: date | None = None, sender=None, limit: int = 20) -> dict:
    """Send today's queued email once; concurrent workers claim separate rows."""
    now = today or _today()
    if now.weekday() >= 5:
        return {"configured": True, "sent": 0, "failed": 0, "skipped_weekend": True}
    send = sender or _send_smtp
    if sender is None and any(not os.getenv(name) for name in (
        "SMTP_HOST", "SMTP_FROM", "SMTP_USERNAME", "SMTP_PASSWORD",
    )):
        return {"configured": False, "sent": 0, "failed": 0}
    recipient = os.getenv("LGU_REMINDER_EMAIL", DEFAULT_RECIPIENT).strip()
    sent = failed = 0
    for _ in range(max(1, min(int(limit), 100))):
        conn = get_connection()
        try:
            row = conn.execute("""
                WITH chosen AS (
                    SELECT id FROM email_reminders
                    WHERE send_on = ? AND recipient_email = ? AND (
                        status = 'pending'
                        OR (status = 'sending' AND claimed_at < CURRENT_TIMESTAMP - INTERVAL '15 minutes')
                    )
                    ORDER BY id LIMIT 1 FOR UPDATE SKIP LOCKED
                )
                UPDATE email_reminders e
                SET status = 'sending', claimed_at = CURRENT_TIMESTAMP, attempts = attempts + 1
                FROM chosen WHERE e.id = chosen.id
                RETURNING e.id, e.recipient_email, e.subject, e.body
            """, (now.isoformat(), recipient)).fetchone()
            conn.commit()
        finally:
            conn.close()
        if not row:
            break
        error_text = None
        try:
            send(row["recipient_email"], row["subject"], row["body"])
        except Exception as error:  # Keep the item retryable for the current day.
            error_text = str(error)[:500]
        conn = get_connection()
        try:
            if error_text is None:
                conn.execute("""
                    UPDATE email_reminders SET status = 'sent', sent_at = CURRENT_TIMESTAMP,
                        last_error = NULL WHERE id = ?
                """, (row["id"],))
                sent += 1
            else:
                conn.execute("""
                    UPDATE email_reminders SET status = 'pending', last_error = ?
                    WHERE id = ?
                """, (error_text, row["id"]))
                failed += 1
            conn.commit()
        finally:
            conn.close()
        if error_text is not None:
            break
    return {"configured": True, "sent": sent, "failed": failed}
