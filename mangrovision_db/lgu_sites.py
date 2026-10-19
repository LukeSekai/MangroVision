"""Permanent LGU field access and independently monitored planting sites."""
from __future__ import annotations

import hashlib
import json
import secrets
from datetime import datetime, time, timedelta

from .activity import append_activity
from .appointment_email import _cipher
from .planter_codes import ALPHABET, _digest


def _db():
    import planting_database
    return planting_database


def is_lgu(conn, organization_id):
    row = conn.execute("SELECT to_jsonb(o)->>'system_key' AS system_key FROM organizations o WHERE id = ?",
                       (organization_id,)).fetchone()
    return bool(row and row['system_key'] == 'lgu')


def ensure_lgu_account():
    """Provision once; existing credentials and participant identities stay stable."""
    db = _db()
    conn = db._get_connection()
    try:
        org = conn.execute("SELECT * FROM organizations WHERE system_key = 'lgu' FOR UPDATE").fetchone()
        if not org:
            raise ValueError('Apply the LGU Sites database update before creating planting access.')
        owner = conn.execute('''SELECT * FROM planters WHERE organization_id = ?
            AND merged_into_planter_id IS NULL FOR UPDATE''', (org['id'],)).fetchone()
        if not owner or owner['username'] is None:
            planter_id = db.create_planter('LGU', f'lgu-system-{org["id"]}', secrets.token_urlsafe(40),
                organization_id=org['id'], participant_count=1, _connection=conn)
        else:
            planter_id = owner['id']
        conn.execute("UPDATE planters SET status = 'active', full_name = 'LGU' WHERE id = ?", (planter_id,))
        conn.execute('INSERT INTO organization_participants(planter_id, slot) VALUES (?, 1) ON CONFLICT DO NOTHING', (planter_id,))
        slot = conn.execute('SELECT participant_username FROM organization_participants WHERE planter_id = ? AND slot = 1', (planter_id,)).fetchone()
        if not slot['participant_username']:
            name = 'lgu'
            if conn.execute("SELECT 1 FROM organization_participants WHERE planter_id = ? AND username_key = 'lgu'", (planter_id,)).fetchone():
                name = 'lgu-staff'
            conn.execute('''UPDATE organization_participants SET participant_username = ?, username_key = ?
                WHERE planter_id = ? AND slot = 1''', (name, name, planter_id))
        if not conn.execute('SELECT 1 FROM permanent_planting_access WHERE planter_id = ?', (planter_id,)).fetchone():
            for _ in range(20):
                raw = ''.join(secrets.choice(ALPHABET) for _ in range(8))
                digest = _digest(raw)
                if not conn.execute('''SELECT 1 FROM activity_access_codes WHERE code_digest = ?
                    UNION ALL SELECT 1 FROM permanent_planting_access WHERE code_digest = ?''', (digest, digest)).fetchone():
                    break
            else:
                raise ValueError('Could not generate planting access. Try again.')
            conn.execute('INSERT INTO permanent_planting_access(planter_id, code_digest, encrypted_code) VALUES (?, ?, ?)',
                         (planter_id, digest, _cipher().encrypt(raw.encode()).decode()))
            append_activity(conn, action='lgu.planting_access_created', actor_type='system', organization_id=org['id'],
                summary='Created permanent LGU planting access.', details={'planter_id': planter_id})
        conn.commit()
        return {'organization_id': org['id'], 'planter_id': planter_id}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def permanent_activity(conn, digest):
    # Existing activity-only installations can finish a rolling code deployment.
    if not conn.execute("SELECT to_regclass('permanent_planting_access') AS relation").fetchone()['relation']:
        return None
    row = conn.execute('''SELECT ac.planter_id, ac.code_digest, p.organization_id, p.participant_count
        FROM permanent_planting_access ac JOIN planters p ON p.id = ac.planter_id
        JOIN organizations o ON o.id = p.organization_id
        WHERE ac.code_digest = ? AND to_jsonb(o)->>'system_key' = 'lgu'
          AND p.status = 'active' AND p.merged_into_planter_id IS NULL FOR SHARE OF ac, p''', (digest,)).fetchone()
    return ({**dict(row), 'schedule_id': None, 'title': 'LGU planting', 'start_at': None,
             'end_at': None, 'expected_planters': row['participant_count'], 'permanent_access': True} if row else None)


def permanent_session(conn, token_hash):
    if not conn.execute("SELECT to_regclass('permanent_planting_access') AS relation").fetchone()['relation']:
        return None
    row = conn.execute('''SELECT op.participant_username FROM auth_sessions auth
        JOIN organization_participants op ON op.planter_id = auth.subject_id AND op.slot = auth.participant_slot
        JOIN permanent_planting_access ac ON ac.planter_id = auth.subject_id AND ac.code_digest = auth.activity_code_digest
        JOIN planters p ON p.id = ac.planter_id JOIN organizations o ON o.id = p.organization_id
        WHERE auth.token_hash = ? AND auth.subject_type = 'planter' AND auth.revoked_at IS NULL
          AND auth.expires_at > clock_timestamp() AND auth.planting_schedule_id IS NULL
          AND op.current_session_hash = auth.token_hash AND p.status = 'active'
          AND p.merged_into_planter_id IS NULL AND to_jsonb(o)->>'system_key' = 'lgu' ''', (token_hash,)).fetchone()
    return ({**dict(row), 'permanent_access': True, 'planting_schedule_id': None,
             'activity_end_at': None, 'activity_title': 'LGU planting'} if row else None)


def _site(conn, site_id, *, lock=False):
    row = conn.execute('''SELECT ps.* FROM project_sites ps JOIN organizations o ON o.id = ps.organization_id
        WHERE ps.id = ? AND ps.deleted_at IS NULL AND to_jsonb(o)->>'system_key' = 'lgu' '''
        + (' FOR UPDATE OF ps' if lock else ''), (site_id,)).fetchone()
    if not row:
        raise ValueError('Choose a project site belonging to LGU.')
    return dict(row)


def _history(conn, site_id):
    return [dict(row) for row in conn.execute('''SELECT r.*, u.full_name AS inspector_name
        FROM lgu_site_monitoring_records r JOIN users u ON u.id = r.inspector_user_id
        WHERE r.project_site_id = ? ORDER BY r.monitored_at DESC, r.id DESC''', (site_id,)).fetchall()]


def _context(conn, site, monitored_at=None):
    db = _db()
    observed = db._parse_dashboard_datetime(monitored_at) if monitored_at else db._manila_now()
    if not observed or observed.date() > db._manila_now().date():
        raise ValueError('Choose a monitoring date that is not in the future.')
    history = _history(conn, site['id'])
    latest = history[0] if history else None
    if latest and observed.date() < db._parse_dashboard_datetime(latest['monitored_at']).date():
        raise ValueError('Choose a date on or after this site’s last monitoring visit.')
    cutoff = min(datetime.combine(observed.date() + timedelta(days=1), time.min, tzinfo=observed.tzinfo), db._manila_now())
    events = [dict(row) for row in conn.execute('''SELECT pe.id, pe.id AS planting_event_id, pe.planting_point_id,
        pe.planted_at, pe.species, pe.assignment_id, pe.latitude, pe.longitude, pe.point_num,
        pe.project_site_id, pe.closed_at, pe.closure_reason, pe.inspection_interval_days, pp.deleted_at,
        a.image_name AS analysis_name, d.death_at,
        EXISTS(SELECT 1 FROM planting_events newer WHERE newer.planting_point_id = pe.planting_point_id
            AND (newer.planted_at, newer.id) > (pe.planted_at, pe.id)
            AND COALESCE(newer.closure_reason, '') <> 'completion_reversed') AS superseded
        FROM planting_events pe JOIN planting_points pp ON pp.id = pe.planting_point_id
        JOIN analyses a ON a.id = pp.analysis_id LEFT JOIN point_death_records d ON d.planting_event_id = pe.id
        WHERE pe.project_site_id = ? AND pe.planted_at <= ?
          AND COALESCE(pe.closure_reason, '') <> 'completion_reversed'
        ORDER BY pe.planted_at, pe.id''', (site['id'], cutoff.isoformat())).fetchall()]
    alive, dead, unknown = [], [], []
    for event in events:
        event.update(project_site_name=site['name'], reference=f"{event['analysis_name']} · Point {event['point_num']}")
        event['selectable'] = not any(event.get(key) for key in ('death_at', 'deleted_at', 'closed_at', 'superseded'))
        (dead if event['death_at'] else alive if event['selectable'] else unknown).append(event)
    checked = {}
    for visit in history:
        for event_id in (db._monitoring_json(visit['snapshot']) or {}).get('event_ids', []):
            checked.setdefault(event_id, visit)
    timings = [db._organization_monitoring_timing(checked.get(event['id']), [event], observed, 1) for event in alive]
    timing = min(timings, key=lambda row: row['next_monitoring_date']) if timings else db._organization_monitoring_timing(None, [], observed, 0)
    fingerprint = json.dumps({'site': site['id'], 'latest': latest['id'] if latest else None,
        'events': [(row['id'], row['death_at'], row['closed_at'], row['deleted_at'], row['superseded']) for row in events]}, default=str)
    total = len(events)
    return {'project_site_id': site['id'], 'site_name': site['name'], 'total_planted': total,
        'geometry': db._monitoring_json(site['polygon_geojson']) if not site.get('area_released_at') else None,
        'area_released_at': site.get('area_released_at'),
        'alive_count': len(alive), 'dead_count': len(dead), 'unknown_count': len(unknown),
        'survival_rate_pct': round(len(alive) / total * 100, 2) if total and not unknown else None,
        'first_planted_at': events[0]['planted_at'] if events else None,
        'last_planted_at': events[-1]['planted_at'] if events else None,
        'latest_record': latest, 'history': history, 'points': alive,
        'version': hashlib.sha256(fingerprint.encode()).hexdigest(), **timing}


def workspace():
    db = _db()
    conn = db._get_connection()
    try:
        account = conn.execute('''SELECT o.id AS organization_id, p.id AS planter_id,
            op.participant_username AS username, ac.encrypted_code
            FROM organizations o JOIN planters p ON p.organization_id = o.id AND p.merged_into_planter_id IS NULL
            JOIN organization_participants op ON op.planter_id = p.id AND op.slot = 1
            JOIN permanent_planting_access ac ON ac.planter_id = p.id WHERE o.system_key = 'lgu' ''').fetchone()
        if not account:
            raise ValueError('The permanent LGU account has not been initialized. Run the LGU Sites setup.')
        raw = _cipher().decrypt(account['encrypted_code'].encode()).decode()
        credentials = {key: account[key] for key in ('organization_id', 'planter_id', 'username')}
        credentials.update(access_code=raw[:4] + '-' + raw[4:], expires_at=None, permanent_access=True)
        sites = [dict(row) for row in conn.execute('''SELECT * FROM project_sites
            WHERE organization_id = ? AND deleted_at IS NULL ORDER BY name, id''', (account['organization_id'],)).fetchall()]
        contexts = [_context(conn, site) for site in sites]
        total, alive, dead, unknown = (sum(row[key] for row in contexts)
            for key in ('total_planted', 'alive_count', 'dead_count', 'unknown_count'))
        return {'account': credentials, 'summary': {'site_count': len(sites), 'total_planted': total,
            'alive_count': alive, 'dead_count': dead, 'unknown_count': unknown,
            'survival_rate_pct': round(alive / total * 100, 2) if total and not unknown else None},
            'sites': [{key: value for key, value in context.items() if key not in {'points', 'history'}}
                      | {'monitoring_record_count': len(context['history'])} for context in contexts]}
    finally:
        conn.close()


def site_context(site_id, monitored_at=None):
    conn = _db()._get_connection()
    try:
        return _context(conn, _site(conn, site_id), monitored_at)
    finally:
        conn.close()


def record_visit(site_id, *, monitored_at, version, dead_planting_event_ids, health_status,
                 actions_taken, inspector_user_id, death_reason_category=None, death_notes='', average_height_cm=None):
    from .monitoring_locations import clean_event_ids
    db = _db()
    ids = clean_event_ids(dead_planting_event_ids)
    health = db._clean_organization_health_status(health_status)
    actions = str(actions_taken or '').strip()
    if not 1 <= len(actions) <= 2000:
        raise ValueError('Describe the LGU actions in 1 to 2000 characters.')
    if ids and death_reason_category not in db.DEATH_REASON_CATEGORIES:
        raise ValueError('Choose a cause for the selected dead seedlings.')
    if len(death_notes or '') > 500:
        raise ValueError('Death notes must be 500 characters or fewer.')
    height = None if average_height_cm is None else float(average_height_cm)
    if height is not None and not 0 <= height <= 1000:
        raise ValueError('Measured height must be between 0 and 1000 cm.')
    conn = db._get_connection()
    try:
        site = _site(conn, site_id, lock=True)
        point_rows = conn.execute('''SELECT DISTINCT planting_point_id FROM planting_events
            WHERE project_site_id = ? ORDER BY planting_point_id''', (site_id,)).fetchall()
        point_ids = [row['planting_point_id'] for row in point_rows]
        if point_ids:
            marks = ','.join('?' for _ in point_ids)
            conn.execute(f'SELECT id FROM planting_points WHERE id IN ({marks}) ORDER BY id FOR UPDATE', point_ids).fetchall()
            conn.execute(f'SELECT id FROM planting_events WHERE planting_point_id IN ({marks}) ORDER BY id FOR UPDATE', point_ids).fetchall()
        context = _context(conn, site, monitored_at)
        if version != context['version']:
            raise ValueError('This site changed. Reload its monitoring form before saving.')
        if not context['monitoring_available']:
            raise ValueError('This site is not due for monitoring yet.')
        eligible = {row['id']: row for row in context['points']}
        if not set(ids).issubset(eligible):
            raise ValueError('Select living seedlings from this LGU site and planting cycle.')
        observed = db._parse_dashboard_datetime(monitored_at)
        snapshot = {'event_ids': sorted(eligible), 'dead_planting_event_ids': ids,
            'scheduled_date': context['next_monitoring_date'], 'total_planted': context['total_planted'],
            'death_reason_category': death_reason_category, 'death_notes': death_notes}
        cursor = conn.execute('''INSERT INTO lgu_site_monitoring_records(project_site_id, monitored_at,
            alive_count, dead_count, new_dead_count, health_status, actions_taken, average_height_cm, snapshot, inspector_user_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS jsonb), ?)''', (site_id, observed.isoformat(),
            context['alive_count']-len(ids), context['dead_count']+len(ids), len(ids), health, actions,
            height, json.dumps(snapshot), inspector_user_id))
        record_id = cursor.lastrowid
        deaths = {}
        for event_id in ids:
            event = eligible[event_id]
            death_at = max(observed, db._parse_dashboard_datetime(event['planted_at'])).isoformat()
            death = conn.execute('''INSERT INTO point_death_records(planting_point_id, assignment_id, planting_event_id,
                death_at, reason_category, reason_label, notes, planter_id, species)
                SELECT planting_point_id, assignment_id, id, ?, ?, ?, ?, planter_id, species
                FROM planting_events WHERE id = ?''', (death_at, death_reason_category,
                db.DEATH_REASON_CATEGORIES[death_reason_category], death_notes, event_id))
            deaths[event_id] = death.lastrowid
            conn.execute("UPDATE planting_events SET closed_at = ?, closure_reason = 'recorded_death' WHERE id = ?", (death_at, event_id))
            conn.execute('''UPDATE planting_points SET death_at = ?, death_reason_category = ?,
                death_reason = ?, death_notes = ? WHERE id = ?''', (death_at, death_reason_category,
                db.DEATH_REASON_CATEGORIES[death_reason_category], death_notes, event['planting_point_id']))
        observations = []
        for event in eligible.values():
            planted = db._parse_dashboard_datetime(event['planted_at'])
            cadence = db._event_inspection_cadence(event)
            interval = (observed.date()-planted.date()).days // cadence * cadence
            while interval > 0 and db._monitoring_workday(planted.date()+timedelta(days=interval)) > observed.date():
                interval -= cadence
            if interval > 0:
                dead = event['id'] in deaths
                observations.append((event['id'], interval, observed.isoformat(), 'dead' if dead else 'alive',
                    actions, death_reason_category if dead else None, inspector_user_id, deaths.get(event['id'])))
        for start in range(0, len(observations), 500):
            batch = observations[start:start+500]
            values = ','.join('(?, ?, ?, ?, ?, ?, ?, ?)' for _ in batch)
            conn.execute(f'''INSERT INTO monitoring_observations(planting_event_id, interval_days, inspected_at,
                status, actions_taken, death_reason_category, inspector_user_id, death_record_id) VALUES {values}
                ON CONFLICT(planting_event_id, interval_days) DO UPDATE SET status = excluded.status,
                    death_reason_category = excluded.death_reason_category, death_record_id = excluded.death_record_id,
                    actions_taken = excluded.actions_taken, inspector_user_id = excluded.inspector_user_id,
                    inspected_at = excluded.inspected_at, updated_at = CURRENT_TIMESTAMP
                WHERE excluded.status = 'dead' AND monitoring_observations.status <> 'dead' ''',
                [value for row in batch for value in row])
        append_activity(conn, action='lgu.site_monitored', actor_type='staff', actor_user_id=inspector_user_id,
            organization_id=site['organization_id'], project_site_id=site_id,
            summary=f"Recorded an LGU monitoring visit for {site['name']}.",
            details={'record_id': record_id, 'new_dead_count': len(ids)})
        conn.commit()
        return {'id': record_id, 'project_site_id': site_id, 'new_dead_count': len(ids)}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def owns_death(conn, event_id):
    if not conn.execute("SELECT to_regclass('lgu_site_monitoring_records') AS relation").fetchone()['relation']:
        return False
    return bool(conn.execute("""SELECT 1 FROM lgu_site_monitoring_records
        WHERE snapshot->'dead_planting_event_ids' @> CAST(? AS jsonb) LIMIT 1""", (json.dumps([event_id]),)).fetchone())


def latest_coverage(conn, event_ids, as_of):
    """Actual site visits retire older operational checks without inventing visits."""
    coverage = {}
    for row in conn.execute('''SELECT monitored_at, snapshot FROM lgu_site_monitoring_records
        WHERE monitored_at <= ? ORDER BY monitored_at DESC, id DESC''', (as_of.isoformat(),)).fetchall():
        for event_id in (_db()._monitoring_json(row['snapshot']) or {}).get('event_ids', []):
            if event_id in event_ids:
                coverage.setdefault(event_id, _db()._parse_dashboard_datetime(row['monitored_at']))
    return coverage
