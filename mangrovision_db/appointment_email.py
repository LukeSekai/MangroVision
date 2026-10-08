"""Private confirmation outbox. Retries reuse the same schedule and credentials."""

from __future__ import annotations

import json
import logging
import os
from uuid import uuid4

from cryptography.fernet import Fernet

from .compat import get_connection
from .notifications import _send_smtp

APPOINTMENT_TYPES = {
    'field_visit': 'Field visit',
    'clean_up_drive': 'Clean-up drive',
    'tree_planting': 'Tree planting',
}


def _cipher():
    key = os.getenv('LIKE_EMAIL_ENCRYPTION_KEY', '').strip()
    try:
        return Fernet(key.encode('ascii'))
    except (ValueError, UnicodeError) as error:
        raise ValueError('Configure LIKE_EMAIL_ENCRYPTION_KEY before confirming website appointments.') from error


def queue_confirmation(conn, request, schedule, email, *, login=None):
    """Called inside the same transaction as approval and account creation."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from urllib.parse import urlsplit

    start = datetime.fromisoformat(schedule['start_at']).astimezone(ZoneInfo('Asia/Manila'))
    end = datetime.fromisoformat(schedule['end_at']).astimezone(ZoneInfo('Asia/Manila'))
    lines = [
        f"Hello {request['contact_name']},", '',
        'Your appointment at Leganes Integrated Katunggan Ecopark (LIKE) is confirmed.', '',
        f"Reference: {request['reference']}",
        f"Organization: {schedule['organization_name']}",
        f"Appointment: {APPOINTMENT_TYPES[request['appointment_type']]}",
        f"Activity: {request['title']}",
        f"Date: {start.strftime('%B %d, %Y')}",
        f"Time: {start.strftime('%I:%M %p')} – {end.strftime('%I:%M %p')} (Philippine time, Asia/Manila)",
        f"Participants: {request['participants']}",
    ]
    if login:
        base_url = os.getenv('MANGROVISION_PUBLIC_URL', '').strip().rstrip('/')
        parsed = urlsplit(base_url)
        if parsed.scheme not in {'https', 'http'} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Configure MANGROVISION_PUBLIC_URL before sending planter login details.')
        lines.extend(['', 'Planter access', f'Open: {base_url}/field', f"Username: {login['username']}"])
        if login.get('password'):
            lines.append(f"Password: {login['password']}")
        else:
            lines.append('Use your existing organization password. Contact LGU staff if you need help accessing the account.')
        lines.extend([f"This shared account supports {login['participant_count']} participant devices.",
                      'Keep these login details private and share them only with your planting group.'])
    lines.extend(['', 'For questions or changes, contact the LGU staff who coordinated your appointment.',
                  '', 'LIKE · Leganes, Iloilo'])
    message = {'subject': f"LIKE appointment confirmed · {request['reference']}", 'body': '\n'.join(lines)}
    encrypted = _cipher().encrypt(json.dumps(message, ensure_ascii=False).encode('utf-8')).decode('ascii')
    conn.execute("""
        INSERT INTO like_appointment_emails(request_id, recipient_email, encrypted_message)
        VALUES (?, ?, ?) ON CONFLICT (request_id) DO NOTHING
    """, (request['id'], email, encrypted))


def send_pending_appointment_emails(*, request_id=None, sender=None, limit=20):
    """Claim durable messages atomically; transient failures retry with backoff."""
    if sender is None and any(not os.getenv(name, '').strip() for name in (
        'SMTP_HOST', 'SMTP_FROM', 'SMTP_USERNAME', 'SMTP_PASSWORD',
    )):
        return {'configured': False, 'sent': 0, 'failed': 0}
    send = sender or _send_smtp
    sent = failed = 0
    for _ in range(max(1, min(int(limit), 100))):
        claim = str(uuid4())
        conn = get_connection()
        try:
            row = conn.execute("""
                WITH chosen AS (
                    SELECT e.id FROM like_appointment_emails e
                    JOIN like_appointment_requests r ON r.id = e.request_id
                    JOIN planting_schedules s ON s.id = r.schedule_id
                    WHERE r.status = 'confirmed' AND s.status = 'confirmed'
                        AND (CAST(? AS bigint) IS NULL OR e.request_id = ?)
                        AND ((e.status = 'pending' AND e.next_attempt_at <= CURRENT_TIMESTAMP)
                             OR (e.status = 'sending' AND e.claimed_at < CURRENT_TIMESTAMP - INTERVAL '15 minutes'))
                    ORDER BY e.next_attempt_at, e.id LIMIT 1 FOR UPDATE OF e SKIP LOCKED
                )
                UPDATE like_appointment_emails e SET status = 'sending', claimed_at = CURRENT_TIMESTAMP,
                    claim_token = ?, attempts = attempts + 1
                FROM chosen WHERE e.id = chosen.id
                RETURNING e.id, e.recipient_email, e.encrypted_message, e.attempts
            """, (request_id, request_id, claim)).fetchone()
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
        if not row:
            break
        error_code = None
        try:
            message = json.loads(_cipher().decrypt(row['encrypted_message'].encode('ascii')))
            send(row['recipient_email'], message['subject'], message['body'])
        except Exception as error:
            # Provider errors can echo recipients, credentials, and server configuration.
            error_code = type(error).__name__
        conn = get_connection()
        try:
            if error_code is None:
                conn.execute("""
                    UPDATE like_appointment_emails SET status = 'sent', sent_at = CURRENT_TIMESTAMP,
                        encrypted_message = NULL, last_error = NULL, claim_token = NULL
                    WHERE id = ? AND claim_token = ? AND status = 'sending'
                """, (row['id'], claim))
                sent += 1
            else:
                delay = min(3600, 60 * 2 ** min(int(row['attempts']) - 1, 6))
                conn.execute("""
                    UPDATE like_appointment_emails SET status = 'pending', last_error = ?, claim_token = NULL,
                        next_attempt_at = CURRENT_TIMESTAMP + (? * INTERVAL '1 second')
                    WHERE id = ? AND claim_token = ? AND status = 'sending'
                """, (error_code, delay, row['id'], claim))
                failed += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    return {'configured': True, 'sent': sent, 'failed': failed}


def run_delivery_cycle(**kwargs):
    """Email outages must not undo approval or expose secrets in API logs."""
    try:
        return send_pending_appointment_emails(**kwargs)
    except Exception as error:
        logging.getLogger(__name__).warning('LIKE email delivery unavailable (%s).', type(error).__name__)
        return {'configured': False, 'sent': 0, 'failed': 1}
