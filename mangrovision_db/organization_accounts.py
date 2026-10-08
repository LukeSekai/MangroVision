"""Organization device identity and stable participant allocations."""
import hashlib


def is_registration_pending(account):
    """A reserved assignment owner has no login credentials until registration."""
    return account.get('username') is None and account.get('password_hash') is None


def allocate_reserved_points(conn, planter_id, participant_count):
    """Divide pre-registration batches once, before any device claims a slot."""
    rows = conn.execute('''SELECT pap.id FROM planter_assignment_points pap
        JOIN planter_assignments pa ON pa.id = pap.assignment_id
        WHERE pa.planter_id = ? AND pap.released_at IS NULL
        ORDER BY pa.id, pap.sequence_num, pap.id''', (planter_id,)).fetchall()
    allocations = list(zip(rows, allocation_slots(len(rows), participant_count)))
    for offset in range(0, len(allocations), 1000):
        batch = allocations[offset:offset + 1000]
        values = ','.join('(CAST(? AS bigint), CAST(? AS integer))' for _ in batch)
        params = [value for row, slot in batch for value in (row['id'], slot)]
        conn.execute(f'''UPDATE planter_assignment_points pap SET participant_slot = allocation.slot
            FROM (VALUES {values}) AS allocation(id, slot) WHERE pap.id = allocation.id''', params)
    conn.execute('DELETE FROM organization_participants WHERE planter_id = ?', (planter_id,))
    conn.execute('INSERT INTO organization_participants(planter_id, slot) SELECT ?, generate_series(1, ?)',
                 (planter_id, participant_count))


def allocation_slots(point_count, participant_count, existing_counts=None):
    """Balance cumulative allocations without moving points already assigned."""
    if not 1 <= participant_count <= 10000:
        raise ValueError('Participant count must be between 1 and 10000.')
    import heapq
    heap = [(int((existing_counts or {}).get(slot, 0)), slot)
            for slot in range(1, participant_count + 1)]
    heapq.heapify(heap)
    slots = []
    for _ in range(point_count):
        count, slot = heapq.heappop(heap)
        slots.append(slot)
        heapq.heappush(heap, (count + 1, slot))
    # Keep each participant's new points together in the selected spatial strips.
    return sorted(slots)


def claim_participant_slot(planter_id, device_key, requested_slot=None, *, recover_slot=False, resume_device=False):
    from planting_database import _get_connection
    if not device_key or not 16 <= len(device_key) <= 200:
        raise ValueError('A persistent device identity is required. Please reload and sign in again.')
    device_hash = hashlib.sha256(device_key.encode()).hexdigest()
    conn = _get_connection()
    try:
        account = conn.execute('SELECT * FROM planters WHERE id = ? FOR UPDATE', (planter_id,)).fetchone()
        if not account or account['status'] != 'active' or account['merged_into_planter_id'] is not None or is_registration_pending(dict(account)):
            raise ValueError('This organization account is inactive.')
        limit = account['participant_count']
        existing = conn.execute('SELECT slot FROM organization_participants WHERE planter_id = ? AND device_key_hash = ? AND slot <= ?',
                                (planter_id, device_hash, limit)).fetchone()
        if existing:
            if recover_slot and requested_slot != existing['slot']:
                raise ValueError(f'This browser is already Participant {existing["slot"]}. Ask the LGU to reset that device slot before recovering Participant {requested_slot}.')
            return existing['slot']
        if resume_device:
            raise ValueError('This recovery code is not linked to this organization. Check the code or ask the LGU to recover your participant number.')
        if recover_slot and (requested_slot is None or not 1 <= requested_slot <= limit):
            raise ValueError(f'Enter the participant number reset by the LGU, between 1 and {limit}.')
        # Older forms sent a participant number on normal sign-in. Treat it as
        # a preference, never as a reason to reject a device when slots remain.
        slot = conn.execute('''SELECT slot FROM organization_participants
            WHERE planter_id = ? AND device_key_hash IS NULL AND slot <= ?
              AND (NOT CAST(? AS BOOLEAN) OR slot = ?)
            ORDER BY CASE WHEN slot = ? THEN 0 ELSE 1 END, slot LIMIT 1''',
                            (planter_id, limit, recover_slot, requested_slot, requested_slot)).fetchone()
        if not slot:
            if recover_slot:
                raise ValueError(f'Participant {requested_slot} is still linked to another device. Ask the LGU to reset that slot first, or use normal sign-in for a new participant.')
            raise ValueError(f'All {limit} participant slots are in use. This organization allows {limit} devices. Ask the LGU to reset a slot when replacing a device.')
        conn.execute('UPDATE organization_participants SET device_key_hash = ? WHERE planter_id = ? AND slot = ?',
                     (device_hash, planter_id, slot['slot']))
        conn.commit()
        return slot['slot']
    finally:
        conn.close()


def list_participant_devices(planter_id):
    """Report occupied browser slots and their history without exposing keys."""
    from planting_database import _get_connection
    conn = _get_connection()
    try:
        account = conn.execute('SELECT participant_count, merged_into_planter_id FROM planters WHERE id = ?',
                               (planter_id,)).fetchone()
        if not account or account['merged_into_planter_id'] is not None:
            raise ValueError('Organization account not found.')
        rows = conn.execute('''WITH session_activity AS (
                SELECT participant_slot, MIN(created_at) AS first_login_at,
                       MAX(last_seen_at) AS last_seen_at,
                       COUNT(*) FILTER (WHERE revoked_at IS NULL AND expires_at > CURRENT_TIMESTAMP) AS active_sessions
                FROM auth_sessions WHERE subject_type = 'planter' AND subject_id = ?
                GROUP BY participant_slot
            ), point_counts AS (
                SELECT pap.participant_slot, COUNT(*) AS assigned_points,
                       COUNT(*) FILTER (WHERE pap.status = 'completed') AS completed_points
                FROM planter_assignment_points pap
                JOIN planter_assignments pa ON pa.id = pap.assignment_id
                WHERE pa.planter_id = ? AND pa.status IN ('active', 'completed')
                  AND pap.released_at IS NULL
                GROUP BY pap.participant_slot
            )
            SELECT op.slot, op.device_key_hash IS NOT NULL AS registered,
                   sa.first_login_at, sa.last_seen_at,
                   COALESCE(sa.active_sessions, 0) AS active_sessions,
                   COALESCE(pc.assigned_points, 0) AS assigned_points,
                   COALESCE(pc.completed_points, 0) AS completed_points
            FROM organization_participants op
            LEFT JOIN session_activity sa ON sa.participant_slot = op.slot
            LEFT JOIN point_counts pc ON pc.participant_slot = op.slot
            WHERE op.planter_id = ? AND op.slot <= ? ORDER BY op.slot''',
                            (planter_id, planter_id, planter_id, account['participant_count'])).fetchall()
        devices = [dict(row) for row in rows]
        registered = sum(device['registered'] for device in devices)
        return {'participant_count': account['participant_count'], 'registered_devices': registered,
                'available_devices': len(devices) - registered, 'devices': devices}
    finally:
        conn.close()


def reset_participant_device(planter_id, slot):
    from planting_database import _get_connection
    conn = _get_connection()
    try:
        conn.execute('SELECT id FROM planters WHERE id = ? FOR UPDATE', (planter_id,)).fetchone()
        row = conn.execute('UPDATE organization_participants SET device_key_hash = NULL WHERE planter_id = ? AND slot = ? RETURNING slot',
                           (planter_id, slot)).fetchone()
        if not row:
            raise ValueError('Participant slot not found.')
        conn.execute("UPDATE auth_sessions SET revoked_at = CURRENT_TIMESTAMP WHERE subject_type = 'planter' AND subject_id = ? AND participant_slot = ?",
                     (planter_id, slot))
        conn.commit()
    finally:
        conn.close()
