"""Append-only activity history shared by staff and field workflows."""

from __future__ import annotations

import json
from typing import Any

from .compat import get_connection


def append_activity(
    conn: Any,
    *,
    action: str,
    actor_type: str,
    summary: str,
    actor_user_id: int | None = None,
    actor_planter_id: int | None = None,
    participant_slot: int | None = None,
    organization_id: int | None = None,
    project_site_id: int | None = None,
    planting_point_id: int | None = None,
    planting_event_id: int | None = None,
    details: dict | None = None,
) -> None:
    conn.execute("""
        INSERT INTO activity_logs (
            action, actor_type, actor_user_id, actor_planter_id, participant_slot,
            organization_id, project_site_id, planting_point_id, planting_event_id,
            summary, details
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, CAST(? AS jsonb))
    """, (
        action, actor_type, actor_user_id, actor_planter_id, participant_slot,
        organization_id, project_site_id, planting_point_id, planting_event_id,
        summary, json.dumps(details or {}, separators=(",", ":")),
    ))


def list_activity(*, organization_id: int | None = None, before_id: int | None = None,
                  limit: int = 50) -> list[dict]:
    clean_limit = max(1, min(int(limit), 100))
    clauses = []
    params: list[Any] = []
    if organization_id is not None:
        clauses.append("al.organization_id = ?")
        params.append(int(organization_id))
    if before_id is not None:
        clauses.append("al.id < ?")
        params.append(int(before_id))
    where = " WHERE " + " AND ".join(clauses) if clauses else ""
    conn = get_connection()
    try:
        rows = conn.execute(f"""
            SELECT al.id, al.action, al.actor_type, al.participant_slot,
                   al.organization_id, al.project_site_id, al.planting_point_id,
                   al.planting_event_id, al.summary, al.details, al.created_at,
                   u.full_name AS staff_name, p.full_name AS planter_name,
                   o.name AS organization_name, ps.name AS project_site_name,
                   pp.point_num
            FROM activity_logs al
            LEFT JOIN users u ON u.id = al.actor_user_id
            LEFT JOIN planters p ON p.id = al.actor_planter_id
            LEFT JOIN organizations o ON o.id = al.organization_id
            LEFT JOIN project_sites ps ON ps.id = al.project_site_id
            LEFT JOIN planting_points pp ON pp.id = al.planting_point_id
            {where}
            ORDER BY al.id DESC LIMIT ?
        """, (*params, clean_limit)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()
