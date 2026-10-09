"""Private confirmation outbox. Retries reuse the same schedule and credentials."""

from __future__ import annotations

import json
import logging
import os
from uuid import uuid4

import httpx
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


def _sender_configured():
    if not os.getenv('SMTP_FROM', '').strip():
        return False
    if os.getenv('BREVO_API_KEY', '').strip():
        return True
    return all(os.getenv(name, '').strip() for name in ('SMTP_HOST', 'SMTP_USERNAME', 'SMTP_PASSWORD'))


def _send_confirmation(recipient, subject, body):
    key = os.getenv('BREVO_API_KEY', '').strip()
    if not key:
        return _send_smtp(recipient, subject, body)
    sender = os.getenv('SMTP_FROM', '').strip()
    if not sender:
        raise ValueError('Configure SMTP_FROM with a verified Brevo sender before sending confirmations.')
    if key.startswith('xsmtpsib-'):
        raise ValueError("BREVO_API_KEY must contain an API key from Brevo's API keys page.")
    with httpx.Client(timeout=30, follow_redirects=False) as client:
        response = client.post(
            'https://api.brevo.com/v3/smtp/email',
            headers={'api-key': key, 'accept': 'application/json'},
            json={
                'sender': {'name': 'MangroVision', 'email': sender},
                'to': [{'email': recipient}],
                'subject': subject,
                'textContent': body,
            },
        )
    if response.status_code != 201:
        raise RuntimeError(f'Brevo did not accept the confirmation email (HTTP {response.status_code}).')
    try:
        result = response.json()
    except ValueError:
        raise RuntimeError('Brevo returned an invalid confirmation-email receipt.') from None
    if not isinstance(result, dict) or not isinstance(result.get('messageId'), str) or not result['messageId'].strip():
        raise RuntimeError('Brevo returned no confirmation-email receipt.')


def _message(request, schedule, *, login=None, access_lines=None, kind='confirmation'):
    """Build current activity details, with credentials kept only in ciphertext."""
    from datetime import datetime
    from zoneinfo import ZoneInfo
    from urllib.parse import urlsplit

    start = datetime.fromisoformat(schedule['start_at']).astimezone(ZoneInfo('Asia/Manila'))
    end = datetime.fromisoformat(schedule['end_at']).astimezone(ZoneInfo('Asia/Manila'))
    announcement = {
        'confirmation': 'Your appointment at Leganes Integrated Katunggan Ecopark (LIKE) is confirmed.',
        'update': 'Your appointment at Leganes Integrated Katunggan Ecopark (LIKE) has been updated. Use the revised details below.',
        'cancellation': 'Your appointment at Leganes Integrated Katunggan Ecopark (LIKE) has been cancelled. Please contact LGU staff before arranging another visit.',
    }[kind]
    lines = [
        f"Hello {request['contact_name']},", '',
        announcement, '',
        f"Reference: {request['reference']}",
        f"Organization: {schedule.get('organization_name') or schedule['organization']}",
        f"Appointment: {APPOINTMENT_TYPES[request['appointment_type']]}",
        f"Activity: {schedule['title']}",
        f"Date: {start.strftime('%B %d, %Y')}",
        f"Time: {start.strftime('%I:%M %p')} – {end.strftime('%I:%M %p')} (Philippine time, Asia/Manila)",
        f"Participants: {schedule.get('expected_planters') if schedule.get('expected_planters') is not None else request['participants']}",
    ]
    access_lines = list(access_lines or [])
    if login and not access_lines and kind != 'cancellation':
        base_url = os.getenv('MANGROVISION_PUBLIC_URL', '').strip().rstrip('/')
        parsed = urlsplit(base_url)
        if parsed.scheme not in {'https', 'http'} or not parsed.netloc or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ValueError('Configure MANGROVISION_PUBLIC_URL before sending planter login details.')
        access_lines = ['Planter access', f'Open: {base_url}/field', f"Username: {login['username']}"]
        if login.get('password'):
            access_lines.append(f"Password: {login['password']}")
        else:
            access_lines.append('Use your existing organization password. Contact LGU staff if you need help accessing the account.')
        access_lines.extend([f"This shared account supports {login['participant_count']} participant devices.",
                      'Keep these login details private and share them only with your planting group.'])
    if access_lines and kind != 'cancellation':
        lines.extend(['', *access_lines])
    lines.extend(['', 'For questions or changes, contact the LGU staff who coordinated your appointment.',
                  '', 'LIKE · Leganes, Iloilo'])
    label = {'confirmation': 'confirmed', 'update': 'updated', 'cancellation': 'cancelled'}[kind]
    return {'subject': f"LIKE appointment {label} · {request['reference']}",
            'body': '\n'.join(lines), 'access_lines': access_lines}


def _encrypted_message(message):
    return _cipher().encrypt(json.dumps(message, ensure_ascii=False).encode('utf-8')).decode('ascii')


def queue_confirmation(conn, request, schedule, email, *, login=None):
    """Called inside the same transaction as approval and account creation."""
    encrypted = _encrypted_message(_message(request, schedule, login=login))
    conn.execute("""
        INSERT INTO like_appointment_emails(request_id, recipient_email, encrypted_message)
        VALUES (?, ?, ?) ON CONFLICT (request_id) DO NOTHING
    """, (request['id'], email, encrypted))


def refresh_schedule_confirmation(conn, current, updated):
    """Replace an unsent message, or queue revised details after earlier delivery."""
    from datetime import datetime, timezone
    from zoneinfo import ZoneInfo
    from .activity_rules import ActivityConflict, changed_fields

    if current.get('source') != 'website':
        return None
    changed = set(changed_fields(current, updated))
    detail_change = bool(changed & {'start_at', 'end_at', 'title', 'expected_planters'})
    cancellation = updated['status'] == 'cancelled' and current['status'] != 'cancelled'
    reactivation = current['status'] == 'cancelled' and updated['status'] in {'confirmed', 'in_progress'}
    if not (detail_change or cancellation or reactivation):
        return None
    if changed & {'start_at', 'end_at'}:
        start = datetime.fromisoformat(updated['start_at'])
        end = datetime.fromisoformat(updated['end_at'])
        manila = ZoneInfo('Asia/Manila')
        if start <= datetime.now(timezone.utc) or start.astimezone(manila).date() != end.astimezone(manila).date():
            raise ValueError('Choose an agreed website appointment time in the future, on one Philippine date.')
    request = conn.execute('SELECT * FROM like_appointment_requests WHERE schedule_id = ?', (current['id'],)).fetchone()
    if not request:
        return None
    existing = conn.execute('SELECT * FROM like_appointment_emails WHERE request_id = ? FOR UPDATE', (request['id'],)).fetchone()
    if existing and existing['status'] == 'sending':
        raise ActivityConflict('The booking email is being delivered. Try saving these changes again after delivery finishes.')
    access_lines = []
    if existing and existing['encrypted_message']:
        try:
            old = json.loads(_cipher().decrypt(existing['encrypted_message'].encode('ascii')))
            access_lines = old.get('access_lines') or []
            if not access_lines and '\nPlanter access\n' in old['body']:
                section = old['body'].split('\nPlanter access\n', 1)[1].split('\n\nFor questions or changes,', 1)[0]
                access_lines = ['Planter access', *section.splitlines()]
        except Exception as error:
            raise ValueError('The queued confirmation could not be refreshed. Check the stable email encryption key before changing this booking.') from error
    login = None
    if not access_lines and request['appointment_type'] == 'tree_planting':
        account = conn.execute("""SELECT username, participant_count FROM planters
            WHERE organization_id = ? AND merged_into_planter_id IS NULL AND status = 'active'
            ORDER BY id LIMIT 1""", (updated['organization_id'],)).fetchone()
        if account and account['username']:
            login = dict(account)
    kind = 'cancellation' if updated['status'] == 'cancelled' else (
        'update' if existing and (existing['status'] == 'sent' or existing['message_kind'] == 'update') else 'confirmation')
    encrypted = _encrypted_message(_message(dict(request), updated, login=login, access_lines=access_lines, kind=kind))
    conn.execute("""
        INSERT INTO like_appointment_emails(request_id, recipient_email, encrypted_message, message_kind)
        VALUES (?, ?, ?, ?)
        ON CONFLICT (request_id) DO UPDATE SET
            encrypted_message = excluded.encrypted_message, message_kind = excluded.message_kind,
            status = 'pending', attempts = 0, next_attempt_at = CURRENT_TIMESTAMP,
            claimed_at = NULL, claim_token = NULL, sent_at = NULL, last_error = NULL
    """, (request['id'], request['email'], encrypted, kind))
    return {'request_id': int(request['id']), 'kind': kind, 'status': 'queued'}


def send_pending_appointment_emails(*, request_id=None, sender=None, limit=20):
    """Claim durable messages atomically; transient failures retry with backoff."""
    if sender is None and not _sender_configured():
        return {'configured': False, 'sent': 0, 'failed': 0}
    send = sender or _send_confirmation
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
                    WHERE r.status = 'confirmed'
                        AND ((e.message_kind = 'cancellation' AND s.status = 'cancelled')
                             OR (e.message_kind <> 'cancellation' AND s.status IN ('confirmed', 'in_progress', 'completed')))
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
