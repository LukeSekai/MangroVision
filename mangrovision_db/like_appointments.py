"""Public intake and atomic LGU approval; public submissions never register organizations."""

from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, timedelta, timezone

import planting_database as planting


class AppointmentConflict(ValueError):
    pass


class AppointmentRateLimit(ValueError):
    pass


def _staff_row(row):
    result = dict(row)
    result.pop('submission_key', None)
    result.pop('payload_hash', None)
    return result


def _row(conn, request_id):
    row = conn.execute("""
        SELECT r.*, s.status AS schedule_status
        FROM like_appointment_requests r
        LEFT JOIN planting_schedules s ON s.id = r.schedule_id
        WHERE r.id = ?
    """, (request_id,)).fetchone()
    if not row:
        return None
    return _staff_row(row)


def _receipt(row):
    # Never return an internal ID, contact information, or organization details publicly.
    return {'reference': row['reference'], 'status': 'pending', 'timezone': 'Asia/Manila'}


def _rate_gate(conn, peer, now):
    hour = now.replace(minute=0, second=0, microsecond=0)
    minute = now.replace(second=0, microsecond=0)
    peer_hash = hashlib.sha256(f'{peer}:{hour.isoformat()}'.encode()).hexdigest()
    for key, window, limit in (
        (f'peer:{peer_hash}', hour, 5),
        (f'global:{minute.isoformat()}', minute, 100),
    ):
        accepted = conn.execute("""
            INSERT INTO like_appointment_rate_limits(bucket_key, window_start, attempts)
            VALUES (?, ?, 1)
            ON CONFLICT (bucket_key) DO UPDATE SET attempts = like_appointment_rate_limits.attempts + 1
            WHERE like_appointment_rate_limits.attempts < ? RETURNING attempts
        """, (key, window.isoformat(), limit)).fetchone()
        if accepted is None:
            raise AppointmentRateLimit('Too many requests. Please try again later or arrange your activity with LGU staff.')
    conn.execute('DELETE FROM like_appointment_rate_limits WHERE window_start < ?',
                 ((now - timedelta(hours=2)).isoformat(),))


def submit_appointment(values: dict, peer: str) -> dict:
    values = dict(values)
    submission_key = str(values.pop('submission_key'))
    values.pop('website', None)
    fingerprint = hashlib.sha256(json.dumps(values, sort_keys=True, default=str).encode()).hexdigest()
    conn = planting._get_connection()
    try:
        # Serialize retries before checking the key; concurrent submissions cannot duplicate intake.
        conn.execute('SELECT pg_advisory_xact_lock(hashtextextended(?, 0))', (submission_key,))
        existing = conn.execute('SELECT * FROM like_appointment_requests WHERE submission_key = ?',
                                (submission_key,)).fetchone()
        if existing:
            if existing['payload_hash'] != fingerprint:
                raise AppointmentConflict('This submission key was already used. Please start a new request.')
            conn.commit()
            return _receipt(existing)
        now = datetime.now(timezone.utc)
        if values['start_at'] <= now:
            raise ValueError('Choose a date and time in the future.')
        _rate_gate(conn, peer, now)
        row = conn.execute("""
            INSERT INTO like_appointment_requests
                (reference, submission_key, payload_hash, organization, contact_name, phone, email,
                 title, start_at, end_at, participants, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?) RETURNING reference
        """, (
            f'LIKE-{secrets.token_hex(6).upper()}', submission_key, fingerprint,
            values['organization'], values['contact_name'], values['phone'], values['email'],
            values['title'], values['start_at'].isoformat(), values['end_at'].isoformat(),
            values['participants'], values['notes'],
        )).fetchone()
        conn.commit()
        return _receipt(row)
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_appointments() -> list[dict]:
    conn = planting._get_connection()
    try:
        rows = conn.execute("""
            SELECT r.*, s.status AS schedule_status
            FROM like_appointment_requests r
            LEFT JOIN planting_schedules s ON s.id = r.schedule_id
            WHERE r.status = 'pending' OR r.id IN (
                SELECT id FROM like_appointment_requests WHERE status <> 'pending'
                ORDER BY created_at DESC, id DESC LIMIT 200
            )
            ORDER BY (r.status = 'pending') DESC, r.created_at DESC, r.id DESC
        """).fetchall()
        return [_staff_row(row) for row in rows]
    finally:
        conn.close()


def review_appointment(request_id: int, values: dict, user_id: int) -> dict:
    conn = planting._get_connection()
    try:
        current = conn.execute('SELECT * FROM like_appointment_requests WHERE id = ? FOR UPDATE',
                               (request_id,)).fetchone()
        if current is None:
            raise LookupError('Appointment request not found.')
        action = values['action']
        if current['status'] != 'pending':
            if current['status'] == action:
                result = _row(conn, request_id)
                conn.commit()
                return result
            raise AppointmentConflict('This request has already been reviewed. Refresh the request list.')
        schedule_id = None
        if action == 'confirmed':
            if not values.get('contacted'):
                raise ValueError('Contact the requester and agree on the schedule before confirming.')
            start = values.get('start_at') or planting._parse_dashboard_datetime(current['start_at'])
            end = values.get('end_at') or planting._parse_dashboard_datetime(current['end_at'])
            if start <= datetime.now(timezone.utc):
                raise ValueError('Choose an agreed planting date and time in the future.')
            if end <= start:
                raise ValueError('End time must be after start time.')
            contact = ', '.join(filter(None, [current['contact_name'], current['phone'], current['email']]))[:300]
            schedule = planting.create_planting_schedule(
                organization=values.get('organization') or current['organization'],
                organization_id=values.get('organization_id'), inspection_interval_days=14,
                title=current['title'], start_at=start, end_at=end, contact=contact,
                expected_planters=current['participants'], status='confirmed',
                notes=current['notes'], created_by_user_id=user_id, _connection=conn,
            )
            schedule_id = schedule['id']
            conn.execute("UPDATE planting_schedules SET source = 'website' WHERE id = ?", (schedule_id,))
        elif not str(values.get('decision_notes') or '').strip():
            raise ValueError('Enter a reason for declining or cancelling the request.')
        conn.execute("""
            UPDATE like_appointment_requests SET status = ?, schedule_id = ?, decision_notes = ?,
                contacted = ?, reviewed_by_user_id = ?, reviewed_at = CURRENT_TIMESTAMP WHERE id = ?
        """, (action, schedule_id, values.get('decision_notes'), bool(values.get('contacted')),
              user_id, request_id))
        result = _row(conn, request_id)
        conn.commit()
        return result
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
