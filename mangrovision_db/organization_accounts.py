"""Organization device identity and stable participant allocations."""
import hashlib


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


def claim_participant_slot(planter_id, device_key, requested_slot=None, *, recover_slot=False):
    from planting_database import _get_connection
    if not device_key or not 16 <= len(device_key) <= 200:
        raise ValueError('A persistent device identity is required. Please reload and sign in again.')
    device_hash = hashlib.sha256(device_key.encode()).hexdigest()
    conn = _get_connection()
    try:
        account = conn.execute('SELECT * FROM planters WHERE id = ? FOR UPDATE', (planter_id,)).fetchone()
        if not account or account['status'] != 'active' or account['merged_into_planter_id'] is not None:
            raise ValueError('This organization account is inactive.')
        limit = account['participant_count']
        existing = conn.execute('SELECT slot FROM organization_participants WHERE planter_id = ? AND device_key_hash = ? AND slot <= ?',
                                (planter_id, device_hash, limit)).fetchone()
        if existing:
            return existing['slot']
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
