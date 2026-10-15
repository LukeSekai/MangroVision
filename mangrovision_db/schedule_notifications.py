"""Staff activity updates use the same private transport as LIKE bookings."""

from __future__ import annotations

import json
import logging
import os
import re
from datetime import datetime
from uuid import uuid4
from zoneinfo import ZoneInfo

from email_validator import EmailNotValidError, validate_email

from . import appointment_email as delivery
from .activity_rules import ActivityConflict, changed_fields
from .compat import get_connection
from .activity_email_design import render_activity_email, safe_link


def clean_notification_email(value):
    value = str(value or '').strip()
    if not value:
        return None
    try:
        return validate_email(value, check_deliverability=False).normalized
    except EmailNotValidError as error:
        raise ValueError('Enter a valid notification email address for the organization.') from error


def notification_recipient(schedule):
    if schedule.get('notification_email'):
        return clean_notification_email(schedule['notification_email'])
    # Legacy contact fields can contain names and phone numbers. Never guess
    # which recipient to use when more than one address has been entered.
    candidates = re.findall(r'[\w.!#$%&\'*+/=?^`{|}~-]+@[\w.-]+\.[A-Za-z]{2,}', schedule.get('contact') or '')
    if len(set(candidates)) != 1:
        return None
    try:
        return clean_notification_email(candidates[0])
    except ValueError:
        return None


def activity_message(current, updated, kind, *, access=None):
    def time_window(schedule):
        zone = ZoneInfo('Asia/Manila')
        start = datetime.fromisoformat(schedule['start_at']).astimezone(zone)
        end = datetime.fromisoformat(schedule['end_at']).astimezone(zone)
        return f"{start:%B %d, %Y, %I:%M %p} to {end:%B %d, %Y, %I:%M %p} (Philippine time)"

    verb = {'confirmation': 'confirmed', 'update': 'updated', 'cancellation': 'cancelled'}[kind]
    lines = [f"Hello {updated['organization']},", '', f'Your MangroVision activity has been {verb}.', '',
             f"Activity: {updated['title']}", f"Current schedule: {time_window(updated)}",
             f"Status: {updated['status']}"]
    if current and any(current[key] != updated[key] for key in ('start_at', 'end_at')):
        lines.append(f'Previous schedule: {time_window(current)}')
    if updated.get('expected_planters') is not None:
        lines.append(f"Expected participants: {updated['expected_planters']}")
    if updated.get('appointment_type', 'tree_planting') == 'tree_planting':
        lines.append(f"Planting area: {updated.get('project_site_name') or 'Awaiting staff assignment'}")
    if kind == 'cancellation':
        lines.extend(['', 'This activity will not proceed at the time above. Contact LGU staff to arrange another activity.'])
    else:
        viewing = (' Signed-in planters can also view the activity and tide advisory in MangroVision.'
                   if updated.get('appointment_type', 'tree_planting') == 'tree_planting' else '')
        lines.extend(['', f'Use the latest schedule above.{viewing}',
                      'Contact LGU staff if these arrangements need to change.'])
    details = [('Organization', updated['organization']), ('Current schedule', time_window(updated)),
               ('Status', updated['status'].replace('_', ' ').capitalize())]
    if current and any(current[key] != updated[key] for key in ('start_at', 'end_at')):
        details.append(('Previous schedule', time_window(current)))
    if updated.get('expected_planters') is not None:
        details.append(('Expected participants', updated['expected_planters']))
    action_url = None
    access_lines = []
    if updated.get('appointment_type', 'tree_planting') == 'tree_planting':
        details.append(('Planting area', updated.get('project_site_name') or 'Awaiting staff assignment'))
        base = safe_link(os.getenv('MANGROVISION_PUBLIC_URL', '').strip().rstrip('/'))
        action_url = f'{base}/field' if base else None
        if kind != 'cancellation' and access and access.get('code'):
            if not action_url:
                raise ValueError('Configure MANGROVISION_PUBLIC_URL before sending planting access.')
            access_lines = ['Planter access', f'Open: {action_url}', f"Organization code: {access['code']}",
                'Share this one code with all participants in your organization.',
                'First time: choose a personal username. Returning: use the same code and username to restore your progress.',
                'Choose Switch here if you need to move an existing login to this browser.',
                'Access opens at the scheduled start and ends at the scheduled finish. No password or PIN is needed.']
            lines.extend(['', *access_lines])
    return {'subject': f"MangroVision activity {verb}: {updated['title']}", 'body': '\n'.join(lines),
            'html_body': render_activity_email(
                kind=kind, title=updated['title'], greeting=f"Hello {updated['organization']},",
                introduction=f'Your MangroVision activity has been {verb}. Please review the current details below.',
                details=details, action_url=action_url, access_lines=access_lines)}


def queue_schedule_notification(conn, current, updated):
    """Schedule and outbox changes commit together; only the latest unsent details remain."""
    if updated.get('source') == 'website':
        return None  # Website recipients and access details stay in their existing outbox.
    changes = set(changed_fields(current, updated)) if current else set()
    if current and changes & {'contact', 'notification_email'} and notification_recipient(current) != notification_recipient(updated):
        changes.add('notification_email')  # Refresh or remove pending delivery to a corrected legacy contact.
    cancelled = updated['status'] == 'cancelled' and current and (
        current['status'] != 'cancelled' or 'notification_email' in changes)
    confirmed = updated['status'] in {'confirmed', 'in_progress'} and (not current or current['status'] in {'requested', 'cancelled'})
    relevant = changes & {'start_at', 'end_at', 'title', 'expected_planters', 'project_site_id', 'notification_email'}
    if not (cancelled or confirmed or (relevant and updated['status'] in {'confirmed', 'in_progress'})):
        return None
    # Editing a draft or past completed activity does not announce a future visit.
    kind = 'cancellation' if cancelled else 'confirmation' if confirmed else 'update'
    existing = conn.execute('SELECT * FROM activity_schedule_emails WHERE schedule_id = ? FOR UPDATE', (updated['id'],)).fetchone()
    if existing and existing['status'] == 'sending':
        raise ActivityConflict('The activity email is being delivered. Please retry saving after delivery finishes.')
    from .group_access import ensure_activity_code
    access = ensure_activity_code(conn, updated) if kind != 'cancellation' else None
    recipient = notification_recipient(updated)
    if not recipient:
        # An older queued message must never deliver obsolete details.
        conn.execute('DELETE FROM activity_schedule_emails WHERE schedule_id = ? AND status <> ?', (updated['id'], 'sent'))
        return {'status': 'missing_recipient', 'kind': kind, 'schedule_id': int(updated['id'])}
    if kind == 'confirmation' and existing and existing['status'] == 'sent':
        kind = 'update'
    message_schedule = dict(updated)
    if updated.get('project_site_id') is not None:
        site = conn.execute('SELECT name FROM site_zones WHERE id = ?', (updated['project_site_id'],)).fetchone()
        message_schedule['project_site_name'] = site['name'] if site else None
    try:
        encrypted = delivery._encrypted_message(activity_message(current, message_schedule, kind, access=access))
    except ValueError as error:
        raise ValueError('Activity email notifications are unavailable. Ask the administrator to configure the email encryption key, then retry.') from error
    conn.execute("""
        INSERT INTO activity_schedule_emails(schedule_id, organization_id, recipient_email, encrypted_message, message_kind)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT (schedule_id) DO UPDATE SET organization_id = excluded.organization_id,
            recipient_email = excluded.recipient_email, encrypted_message = excluded.encrypted_message,
            message_kind = excluded.message_kind, status = 'pending', attempts = 0,
            next_attempt_at = CURRENT_TIMESTAMP, claimed_at = NULL, claim_token = NULL, sent_at = NULL, last_error = NULL
    """, (updated['id'], updated['organization_id'], recipient, encrypted, kind))
    return {'status': 'queued', 'kind': kind, 'schedule_id': int(updated['id'])}


def send_pending_schedule_emails(*, schedule_id=None, sender=None, limit=20):
    if sender is None and not delivery._sender_configured():
        return {'configured': False, 'sent': 0, 'failed': 0}
    send = sender or delivery._send_confirmation
    sent = failed = 0
    for _ in range(max(1, min(int(limit), 100))):
        claim = str(uuid4())
        conn = get_connection()
        try:
            row = conn.execute("""
                WITH chosen AS (
                    SELECT e.id FROM activity_schedule_emails e JOIN planting_schedules s ON s.id = e.schedule_id
                    WHERE ((e.message_kind = 'cancellation' AND s.status = 'cancelled')
                        OR (e.message_kind <> 'cancellation' AND s.status IN ('confirmed', 'in_progress', 'completed')))
                      AND (CAST(? AS bigint) IS NULL OR e.schedule_id = ?)
                      AND ((e.status = 'pending' AND e.next_attempt_at <= CURRENT_TIMESTAMP)
                        OR (e.status = 'sending' AND e.claimed_at < CURRENT_TIMESTAMP - INTERVAL '15 minutes'))
                    ORDER BY e.next_attempt_at, e.id LIMIT 1 FOR UPDATE OF e SKIP LOCKED
                ) UPDATE activity_schedule_emails e SET status = 'sending', claimed_at = CURRENT_TIMESTAMP,
                    claim_token = ?, attempts = attempts + 1 FROM chosen WHERE e.id = chosen.id
                RETURNING e.id, e.recipient_email, e.encrypted_message, e.attempts
            """, (schedule_id, schedule_id, claim)).fetchone()
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
            message = json.loads(delivery._cipher().decrypt(row['encrypted_message'].encode('ascii')))
            if sender is not None:
                send(row['recipient_email'], message['subject'], message['body'])
            else:
                send(row['recipient_email'], message['subject'], message['body'], html_body=message.get('html_body'))
        except Exception as error:
            error_code = type(error).__name__  # Never store provider text or credentials.
        conn = get_connection()
        try:
            if error_code is None:
                conn.execute("""UPDATE activity_schedule_emails SET status = 'sent', sent_at = CURRENT_TIMESTAMP,
                    encrypted_message = NULL, last_error = NULL, claim_token = NULL
                    WHERE id = ? AND claim_token = ? AND status = 'sending'""", (row['id'], claim))
                sent += 1
            else:
                delay = min(3600, 60 * 2 ** min(int(row['attempts']) - 1, 6))
                conn.execute("""UPDATE activity_schedule_emails SET status = 'pending', last_error = ?, claim_token = NULL,
                    next_attempt_at = CURRENT_TIMESTAMP + (? * INTERVAL '1 second')
                    WHERE id = ? AND claim_token = ? AND status = 'sending'""", (error_code, delay, row['id'], claim))
                failed += 1
            conn.commit()
        except Exception:
            conn.rollback()
            raise
        finally:
            conn.close()
    return {'configured': True, 'sent': sent, 'failed': failed}


def run_schedule_delivery_cycle(**kwargs):
    try:
        return send_pending_schedule_emails(**kwargs)
    except Exception as error:
        logging.getLogger(__name__).warning('Activity email delivery unavailable (%s).', type(error).__name__)
        return {'configured': False, 'sent': 0, 'failed': 1}
