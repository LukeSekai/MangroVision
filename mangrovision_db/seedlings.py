"""Staff-entered planting records for mapped baby seedlings."""

from __future__ import annotations

from datetime import date, datetime, time
from math import isfinite
from typing import Any
from uuid import uuid4

from .activity import append_activity
from .compat import get_connection

CONDITIONS = frozenset({"healthy", "fair", "stressed", "unknown"})


def _details(height_cm: Any, condition: Any, notes: Any) -> tuple[float | None, str | None, str | None]:
    height = None if height_cm in (None, "") else float(height_cm)
    if height is not None and (not isfinite(height) or height < 0 or height > 1000):
        raise ValueError("Initial seedling height must be between 0 and 1,000 cm.")
    clean_condition = str(condition or "").strip().lower() or None
    if clean_condition is not None and clean_condition not in CONDITIONS:
        raise ValueError("Choose a valid initial seedling condition.")
    clean_notes = str(notes or "").strip() or None
    if clean_notes and len(clean_notes) > 1000:
        raise ValueError("Seedling notes must be 1,000 characters or fewer.")
    return height, clean_condition, clean_notes


def _planting_timestamp(day: str) -> datetime:
    from planting_database import _MANILA_TZ, _manila_now

    try:
        chosen = date.fromisoformat(str(day))
    except (TypeError, ValueError) as error:
        raise ValueError("Choose a valid planting date.") from error
    now = _manila_now()
    if chosen > now.date():
        raise ValueError("Planting date cannot be in the future.")
    return now if chosen == now.date() else datetime.combine(chosen, time(12, 0), tzinfo=_MANILA_TZ)


def record_lgu_planting(*, point_ids: list[int], project_site_id: int, planted_date: str,
                        species: str, staff_user_id: int, initial_height_cm: Any = None,
                        initial_condition: Any = None, initial_notes: Any = None) -> dict:
    from planting_database import (
        _canonical_assignment_species, _is_point_inside_eroded_zone,
        _resolve_point_project_sites,
    )

    ids = list(dict.fromkeys(int(value) for value in point_ids))
    if not ids or len(ids) > 500:
        raise ValueError("Choose between 1 and 500 mapped points.")
    site_id = int(project_site_id)
    planted_at = _planting_timestamp(planted_date)
    canonical_species = _canonical_assignment_species(species)
    if not canonical_species:
        raise ValueError("Choose a valid mangrove species.")
    height, condition, notes = _details(initial_height_cm, initial_condition, initial_notes)
    marks = ",".join("?" for _ in ids)
    conn = get_connection()
    try:
        site = conn.execute("SELECT id, name FROM project_sites WHERE id = ?", (site_id,)).fetchone()
        if not site:
            raise ValueError("Selected project site was not found.")
        rows = conn.execute(f"""
            SELECT pp.id, pp.point_num, pp.latitude, pp.longitude, pp.status,
                   pp.deleted_at, pp.death_at, a.project_site_id AS source_site_id
            FROM planting_points pp JOIN analyses a ON a.id = pp.analysis_id
            WHERE pp.id IN ({marks}) ORDER BY pp.id FOR UPDATE OF pp
        """, ids).fetchall()
        if len(rows) != len(ids):
            raise ValueError("One or more mapped planting points were not found.")
        points = _resolve_point_project_sites(conn, rows)
        for point in points:
            if point["source_site_id"] != site_id:
                raise ValueError(f"Point #{point['point_num']} is not inside the selected project site.")
            if point["deleted_at"] is not None or point["status"] != "planned" or point["death_at"] is not None:
                raise ValueError(f"Point #{point['point_num']} is not an unplanted available point.")
            if _is_point_inside_eroded_zone(point["latitude"], point["longitude"]):
                raise ValueError(f"Point #{point['point_num']} is unavailable inside an eroded zone.")
        blocked = conn.execute(f"""
            SELECT pp.point_num FROM planter_assignment_points pap
            JOIN planting_points pp ON pp.id = pap.planting_point_id
            JOIN planter_assignments pa ON pa.id = pap.assignment_id
            WHERE pap.planting_point_id IN ({marks}) AND pap.released_at IS NULL
              AND pa.status IN ('active', 'completed') LIMIT 1
        """, ids).fetchone()
        if blocked:
            raise ValueError(f"Point #{blocked['point_num']} is already assigned to an organization.")
        historical = conn.execute(f"""
            SELECT pp.point_num FROM planting_events pe
            JOIN planting_points pp ON pp.id = pe.planting_point_id
            WHERE pe.planting_point_id IN ({marks}) LIMIT 1
        """, ids).fetchone()
        if historical:
            raise ValueError(f"Point #{historical['point_num']} has planting history; use the reviewed replanting workflow.")
        for point in points:
            conn.execute("""
                UPDATE planting_points SET status = 'planted', planted_at = ?, planted_date = ?,
                    death_at = NULL, death_reason = NULL, death_reason_category = NULL, death_notes = NULL
                WHERE id = ?
            """, (planted_at.isoformat(), planted_at.date().isoformat(), point["id"]))
            cursor = conn.execute("""
                INSERT INTO planting_events (
                    source_key, planting_point_id, project_site_id, species, planted_at,
                    point_num, latitude, longitude, source, site_attribution_source,
                    inspection_interval_days, planted_by_user_id,
                    initial_height_cm, initial_condition, initial_notes
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'lgu_direct', 'lgu_direct', 14, ?, ?, ?, ?)
            """, (
                f"lgu-direct:{uuid4().hex}", point["id"], site_id, canonical_species,
                planted_at.isoformat(), point["point_num"], point["latitude"], point["longitude"],
                staff_user_id, height, condition, notes,
            ))
            append_activity(
                conn, action="seedling.planted", actor_type="staff", actor_user_id=staff_user_id,
                project_site_id=site_id, planting_point_id=point["id"],
                planting_event_id=cursor.lastrowid,
                summary=f"LGU planted seedling at point #{point['point_num']} in {site['name']}.",
                details={"species": canonical_species, "planted_date": planted_at.date().isoformat()},
            )
        conn.commit()
        return {"planted_points": len(points), "project_site_id": site_id, "planted_date": planted_at.date().isoformat()}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


def list_seedling_records(*, limit: int = 100) -> list[dict]:
    conn = get_connection()
    try:
        rows = conn.execute("""
            SELECT pe.id, pe.planting_point_id, pe.point_num, pe.project_site_id,
                   ps.name AS project_site_name, pe.species, pe.planted_at,
                   pe.source, pe.initial_height_cm, pe.initial_condition, pe.initial_notes,
                   pe.planted_by_user_id, u.full_name AS lgu_planter_name,
                   p.organization_id, o.name AS organization_name
            FROM planting_events pe
            JOIN planting_points pp ON pp.id = pe.planting_point_id
            LEFT JOIN project_sites ps ON ps.id = pe.project_site_id
            LEFT JOIN users u ON u.id = pe.planted_by_user_id
            LEFT JOIN planters p ON p.id = pe.planter_id
            LEFT JOIN organizations o ON o.id = p.organization_id
            WHERE pp.status = 'planted' AND pp.deleted_at IS NULL
              AND pe.closed_at IS NULL
            ORDER BY pe.planted_at DESC, pe.id DESC LIMIT ?
        """, (max(1, min(int(limit), 500)),)).fetchall()
        return [dict(row) for row in rows]
    finally:
        conn.close()


def update_seedling_details(event_id: int, *, staff_user_id: int,
                            initial_height_cm: Any, initial_condition: Any,
                            initial_notes: Any) -> dict:
    height, condition, notes = _details(initial_height_cm, initial_condition, initial_notes)
    conn = get_connection()
    try:
        event = conn.execute("""
            SELECT pe.id, pe.planting_point_id, pe.point_num, pe.project_site_id,
                   p.organization_id, pp.status
            FROM planting_events pe
            JOIN planting_points pp ON pp.id = pe.planting_point_id
            LEFT JOIN planters p ON p.id = pe.planter_id
            WHERE pe.id = ? AND pe.closed_at IS NULL FOR UPDATE OF pe
        """, (int(event_id),)).fetchone()
        if not event or event["status"] != "planted":
            raise ValueError("Current planted seedling was not found.")
        conn.execute("""
            UPDATE planting_events
            SET initial_height_cm = ?, initial_condition = ?, initial_notes = ?
            WHERE id = ?
        """, (height, condition, notes, event_id))
        append_activity(
            conn, action="seedling.details_updated", actor_type="staff",
            actor_user_id=staff_user_id, organization_id=event["organization_id"],
            project_site_id=event["project_site_id"], planting_point_id=event["planting_point_id"],
            planting_event_id=event_id,
            summary=f"LGU updated the seedling record for point #{event['point_num']}.",
            details={"initial_height_cm": height, "initial_condition": condition},
        )
        conn.commit()
        return {"id": event_id, "initial_height_cm": height,
                "initial_condition": condition, "initial_notes": notes}
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
