"""Seed a coherent, clearly labelled MangroVision consultation dataset.

This is data-only tooling, separate from Alembic schema migrations. It uses
existing unassigned analysis points, preserves unrelated records, inserts the
relationships in one short transaction, and is safe to preview with
``--dry-run``. A second run is a no-op unless ``--replace`` is explicit.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import text


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mangrovision_db.compat import get_engine
from mangrovision_db.passwords import hash_password


SEED_MARKER = "consultation-demo-v1"
DEMO_PASSWORD = "MangroveDemo2026!"
DEMO_SPECIES = "Bungalon"
POINTS_PER_ORGANIZATION = 360
COMPLETED_PLANTED_COUNTS = (244, 243, 243)
COMPLETED_SKIPPED_COUNTS = (6, 7, 7)
ACTIVE_PLANTED_COUNTS = (90, 90, 90)
MANILA = ZoneInfo("Asia/Manila")

ORGANIZATIONS = (
    {
        "name": "Leganes Mangrove Guardians Association",
        "key": "consultation-demo-leganes-guardians",
        "site": "Leganes Mangrove Guardians Association",
        "species": DEMO_SPECIES,
        "mortality_profile": "healthy",
        "contact": "Maribel Santos",
        "planters": (
            ("Ana Mae Dela Cruz", "demo.ana", "0999-000-1001"),
            ("Roberto Villanueva", "demo.roberto", "0999-000-1002"),
        ),
    },
    {
        "name": "Iloilo Coastal Youth Volunteers",
        "key": "consultation-demo-iloilo-youth",
        "site": "Iloilo Coastal Youth Volunteers",
        "species": DEMO_SPECIES,
        "mortality_profile": "low",
        "contact": "Joshua Lim",
        "planters": (
            ("Joshua Miguel Lim", "demo.joshua", "0999-000-2001"),
            ("Leah Mae Ramos", "demo.leah", "0999-000-2002"),
        ),
    },
    {
        "name": "Katunggan Community Planters Cooperative",
        "key": "consultation-demo-katunggan-coop",
        "site": "Katunggan Community Planters Cooperative",
        "species": DEMO_SPECIES,
        "mortality_profile": "critical",
        "contact": "Elena Garcia",
        "planters": (
            ("Elena Garcia", "demo.elena", "0999-000-3001"),
            ("Mark Anthony Flores", "demo.mark", "0999-000-3002"),
        ),
    },
)

TABLES = (
    "users", "organizations", "project_sites", "analyses", "planting_points",
    "planters", "planter_assignments", "planter_assignment_points",
    "planting_events", "point_death_records", "monitoring_observations",
    "organization_monitoring_records", "planting_schedules", "map_zones",
    "dashboard_settings",
)


def _active_demo_count(connection) -> int:
    return int(connection.execute(
        text("SELECT count(*) FROM organizations WHERE normalized_name LIKE 'consultation-demo-%'")
    ).scalar_one())


def _candidate_analyses(connection) -> list[dict]:
    return [dict(row) for row in connection.execute(text("""
        SELECT a.id, a.image_name, count(pp.id) AS safe_planned_points
        FROM analyses a
        JOIN planting_points pp ON pp.analysis_id = a.id
        WHERE a.deleted_at IS NULL
          AND a.project_site_id IS NULL
          AND pp.deleted_at IS NULL
          AND pp.status = 'planned'
          AND pp.death_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM planter_assignment_points pap
              WHERE pap.planting_point_id = pp.id
          )
          AND NOT EXISTS (
              SELECT 1
              FROM map_zones mz
              WHERE mz.deleted_at IS NULL
                AND mz.zone_type IN ('eroded', 'forbidden')
                AND extensions.ST_Covers(mz.geometry, pp.location)
          )
        GROUP BY a.id, a.image_name, a.analyzed_at
        HAVING count(pp.id) >= :minimum
        ORDER BY count(pp.id) DESC, a.analyzed_at DESC, a.id DESC
        LIMIT :organization_count
    """), {
        "minimum": POINTS_PER_ORGANIZATION,
        "organization_count": len(ORGANIZATIONS),
    }).mappings().all()]


def _inventory(connection) -> None:
    print("Current database inventory")
    for table in TABLES:
        count = connection.execute(text(f"SELECT count(*) FROM {table}")).scalar_one()
        print(f"  {table}: {count}")
    candidates = _candidate_analyses(connection)
    print("Eligible analyses")
    for row in candidates:
        print(
            f"  analysis={row['id']} image={row['image_name']} "
            f"safe_planned_points={row['safe_planned_points']}"
        )
    print(
        f"Planned insert: {len(ORGANIZATIONS)} organizations, "
        f"{len(ORGANIZATIONS) * 2} planters, "
        f"{len(ORGANIZATIONS) * POINTS_PER_ORGANIZATION} assigned points."
    )


def _site_polygon(connection, analysis_id: int) -> tuple[dict, float, float]:
    bounds = connection.execute(text("""
        SELECT min(longitude) AS min_lon, max(longitude) AS max_lon,
               min(latitude) AS min_lat, max(latitude) AS max_lat
        FROM planting_points
        WHERE analysis_id = :analysis_id AND deleted_at IS NULL
    """), {"analysis_id": analysis_id}).mappings().one()
    if any(bounds[key] is None for key in ("min_lon", "max_lon", "min_lat", "max_lat")):
        raise RuntimeError(f"Analysis {analysis_id} has no usable planting-point extent.")
    min_lon, max_lon = float(bounds["min_lon"]), float(bounds["max_lon"])
    min_lat, max_lat = float(bounds["min_lat"]), float(bounds["max_lat"])
    padding = 0.00001
    min_lon, max_lon = min_lon - padding, max_lon + padding
    min_lat, max_lat = min_lat - padding, max_lat + padding
    polygon = {
        "type": "Polygon",
        "coordinates": [[
            [min_lon, min_lat], [max_lon, min_lat], [max_lon, max_lat],
            [min_lon, max_lat], [min_lon, min_lat],
        ]],
    }
    return polygon, (min_lat + max_lat) / 2.0, (min_lon + max_lon) / 2.0


def _safe_points(connection, analysis_id: int) -> list[dict]:
    rows = connection.execute(text("""
        SELECT pp.id, pp.point_num, pp.latitude, pp.longitude
        FROM planting_points pp
        WHERE pp.analysis_id = :analysis_id
          AND pp.deleted_at IS NULL
          AND pp.status = 'planned'
          AND pp.death_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM planter_assignment_points pap
              WHERE pap.planting_point_id = pp.id
          )
          AND NOT EXISTS (
              SELECT 1
              FROM map_zones mz
              WHERE mz.deleted_at IS NULL
                AND mz.zone_type IN ('eroded', 'forbidden')
                AND extensions.ST_Covers(mz.geometry, pp.location)
          )
        ORDER BY pp.point_num, pp.id
        LIMIT :point_limit
    """), {
        "analysis_id": analysis_id,
        "point_limit": POINTS_PER_ORGANIZATION,
    }).mappings().all()
    if len(rows) != POINTS_PER_ORGANIZATION:
        raise RuntimeError(
            f"Analysis {analysis_id} no longer has {POINTS_PER_ORGANIZATION} safe planned points."
        )
    return [dict(row) for row in rows]


def _insert_returning_id(connection, sql: str, values: dict) -> int:
    return int(connection.execute(text(sql), values).scalar_one())


def _cleanup_demo(connection) -> None:
    marker_like = f"%[{SEED_MARKER}]%"
    event_like = f"{SEED_MARKER}:%"
    connection.execute(text("""
        DELETE FROM monitoring_observations
        WHERE planting_event_id IN (
            SELECT id FROM planting_events WHERE source_key LIKE :event_like
        )
    """), {"event_like": event_like})
    connection.execute(text("""
        DELETE FROM point_death_records
        WHERE planting_event_id IN (
            SELECT id FROM planting_events WHERE source_key LIKE :event_like
        )
    """), {"event_like": event_like})
    connection.execute(
        text("DELETE FROM planting_events WHERE source_key LIKE :event_like"),
        {"event_like": event_like},
    )
    connection.execute(text("""
        UPDATE planting_points
        SET status = 'planned', planted_at = NULL, planted_date = NULL,
            death_at = NULL, death_reason = NULL,
            death_reason_category = NULL, death_notes = NULL
        WHERE id IN (
            SELECT pap.planting_point_id
            FROM planter_assignment_points pap
            JOIN planter_assignments pa ON pa.id = pap.assignment_id
            WHERE pa.notes LIKE :marker_like
        )
    """), {"marker_like": marker_like})
    connection.execute(text("""
        DELETE FROM planter_assignments WHERE notes LIKE :marker_like
    """), {"marker_like": marker_like})
    connection.execute(text("""
        DELETE FROM planting_schedules WHERE notes LIKE :marker_like
    """), {"marker_like": marker_like})
    connection.execute(text("""
        DELETE FROM organization_monitoring_records
        WHERE organization_id IN (
            SELECT id FROM organizations WHERE normalized_name LIKE 'consultation-demo-%'
        )
    """))
    connection.execute(text("""
        DELETE FROM planters
        WHERE organization_id IN (
            SELECT id FROM organizations WHERE normalized_name LIKE 'consultation-demo-%'
        )
    """))
    connection.execute(text("""
        UPDATE analyses SET project_site_id = NULL
        WHERE project_site_id IN (
            SELECT ps.id FROM project_sites ps
            JOIN organizations o ON o.id = ps.organization_id
            WHERE o.normalized_name LIKE 'consultation-demo-%'
        )
    """))
    connection.execute(text("""
        DELETE FROM project_sites
        WHERE organization_id IN (
            SELECT id FROM organizations WHERE normalized_name LIKE 'consultation-demo-%'
        )
    """))
    connection.execute(
        text("DELETE FROM organizations WHERE normalized_name LIKE 'consultation-demo-%'")
    )


def _snap_to_visit_day(value: datetime, weekdays: tuple[int, ...] = (2, 5)) -> datetime:
    offset = min((weekday - value.isoweekday()) % 7 for weekday in weekdays)
    return value + timedelta(days=offset)


def _observation_rounds(
    age_days: int,
    position: int,
    mortality_profile: str,
) -> list[tuple[int, str, bool]]:
    due = [day for day in range(30, age_days + 1, 30)]
    if not due:
        return []
    if mortality_profile == "low" and position % 10 < 4:
        return [(30, "dead", False)]
    if mortality_profile == "critical" and position % 2 == 0:
        return [(30, "dead", False)]
    bucket = position % 20
    if bucket == 0:
        return [(30, "dead", False)]
    if bucket == 1:
        if 60 in due:
            return [(30, "alive", False), (60, "dead", False)]
        return [(30, "alive", False)]
    if bucket in {2, 3}:
        # Leave the most recent scheduled round open so the dashboard has a
        # realistic overdue queue without making most plants look neglected.
        return [(day, "alive", False) for day in due[:-1]]
    if bucket == 4:
        return [
            (day, "missing" if day == due[-1] else "alive", day == due[-1])
            for day in due
        ]
    return [(day, "alive", bucket == 5 and day == due[0]) for day in due]


def _create_observations(connection, events: list[dict], inspector_id: int, now: datetime) -> int:
    death_reasons = ("waves", "drying_out", "animal_damage", "barnacles", "storm")
    reason_labels = {
        "waves": "Waves", "drying_out": "Drying out",
        "animal_damage": "Animal damage", "barnacles": "Barnacles",
        "storm": "Storm",
    }
    observation_rows = []
    death_index = 0
    for event in events:
        age_days = max(0, (now.date() - event["planted_at"].date()).days)
        for interval, status, intentionally_late in _observation_rounds(
            age_days, event["position"], event["mortality_profile"]
        ):
            target = event["planted_at"] + timedelta(days=interval)
            inspected = _snap_to_visit_day(target).replace(hour=9, minute=0, second=0)
            if intentionally_late:
                inspected += timedelta(days=4)
            if inspected > now:
                continue
            condition = {
                "alive": "Healthy leaves and stable anchoring",
                "dead": "No living foliage observed",
                "missing": "Marker obscured; verification required",
            }[status]
            actions = {
                "alive": "Removed debris and checked protective stake.",
                "dead": "Recorded mortality and flagged the point for replacement review.",
                "missing": "Scheduled a return visit with GPS verification.",
            }[status]
            height = (
                round(28.0 + (event["position"] % 12) * 1.2 + interval * 0.18, 1)
                if status == "alive" else None
            )
            reason = death_reasons[death_index % len(death_reasons)] if status == "dead" else None
            observation_rows.append({
                "event_id": event["id"], "interval": interval,
                "inspected_at": inspected, "status": status,
                "condition": condition, "height_cm": height,
                "notes": f"[{SEED_MARKER}] Round {interval}-day field inspection.",
                "actions": actions, "reason": reason,
                "inspector_id": inspector_id, "created_at": inspected,
                "updated_at": inspected,
            })
            if status == "dead":
                death_index += 1
                break

    if observation_rows:
        connection.execute(text("""
            INSERT INTO monitoring_observations (
                planting_event_id, interval_days, inspected_at, status,
                condition, height_cm, notes, actions_taken,
                death_reason_category, inspector_user_id, created_at, updated_at
            ) VALUES (
                :event_id, :interval, :inspected_at, :status,
                :condition, :height_cm, :notes, :actions,
                :reason, :inspector_id, :created_at, :updated_at
            )
        """), observation_rows)

    deaths = connection.execute(text("""
        SELECT mo.id AS observation_id, mo.inspected_at, mo.death_reason_category,
               pe.id AS event_id, pe.planting_point_id AS point_id,
               pe.assignment_id, pe.planter_id, pe.species,
               p.full_name AS planter_name
        FROM monitoring_observations mo
        JOIN planting_events pe ON pe.id = mo.planting_event_id
        LEFT JOIN planters p ON p.id = pe.planter_id
        WHERE pe.source_key LIKE :event_like AND mo.status = 'dead'
        ORDER BY mo.id
    """), {"event_like": f"{SEED_MARKER}:%"}).mappings().all()

    death_rows = []
    for death in deaths:
        reason = death["death_reason_category"]
        death_rows.append({
            "point_id": death["point_id"],
            "assignment_id": death["assignment_id"],
            "event_id": death["event_id"],
            "observation_id": death["observation_id"],
            "death_at": death["inspected_at"],
            "reason": reason,
            "reason_label": reason_labels[reason],
            "notes": f"[{SEED_MARKER}] Verified during scheduled monitoring.",
            "planter_id": death["planter_id"],
            "planter_name": death["planter_name"],
            "species": death["species"],
            "created_at": death["inspected_at"],
        })
    if death_rows:
        connection.execute(text("""
            INSERT INTO point_death_records (
                planting_point_id, assignment_id, planting_event_id,
                monitoring_observation_id, death_at, reason_category,
                reason_label, notes, planter_id, planter_name, species, created_at
            ) VALUES (
                :point_id, :assignment_id, :event_id, :observation_id,
                :death_at, :reason, :reason_label, :notes,
                :planter_id, :planter_name, :species, :created_at
            )
        """), death_rows)
        linked_deaths = connection.execute(text("""
            SELECT id AS death_id, monitoring_observation_id AS observation_id,
                   planting_point_id AS point_id, planting_event_id AS event_id,
                   death_at, reason_category AS reason, reason_label
            FROM point_death_records
            WHERE monitoring_observation_id IN (
                SELECT mo.id FROM monitoring_observations mo
                JOIN planting_events pe ON pe.id = mo.planting_event_id
                WHERE pe.source_key LIKE :event_like AND mo.status = 'dead'
            )
        """), {"event_like": f"{SEED_MARKER}:%"}).mappings().all()
        connection.execute(text("""
            UPDATE monitoring_observations SET death_record_id = :death_id
            WHERE id = :observation_id
        """), [dict(row) for row in linked_deaths])
        connection.execute(text("""
            UPDATE planting_points
            SET death_at = :death_at, death_reason = :death_reason,
                death_reason_category = :reason,
                death_notes = 'Verified during scheduled monitoring.'
            WHERE id = :point_id
        """), [{
            **dict(row),
            "death_reason": f"{row['reason_label']}: verified during monitoring",
        } for row in linked_deaths])
        connection.execute(text("""
            UPDATE planting_events
            SET closed_at = :death_at, closure_reason = 'monitoring_death'
            WHERE id = :event_id
        """), [dict(row) for row in linked_deaths])
    return len(observation_rows)


def _seed(connection) -> dict:
    now = datetime.now(MANILA).replace(microsecond=0)
    candidates = _candidate_analyses(connection)
    if len(candidates) < len(ORGANIZATIONS):
        raise RuntimeError(
            f"At least three unlinked analyses with {POINTS_PER_ORGANIZATION} "
            "safe planned points each are required."
        )
    inspector_id = connection.execute(text("""
        SELECT id FROM users
        WHERE role IN ('admin', 'lgu', 'planner')
        ORDER BY CASE role WHEN 'admin' THEN 0 WHEN 'lgu' THEN 1 ELSE 2 END, id
        LIMIT 1
    """)).scalar_one_or_none()
    if inspector_id is None:
        raise RuntimeError("Create an LGU/admin user before seeding consultation data.")
    inspector_id = int(inspector_id)

    connection.execute(text("""
        INSERT INTO dashboard_settings (
            year, annual_planting_target, min_survival_target_pct,
            inspection_intervals_json, inspection_weekdays_json,
            updated_at, updated_by_user_id
        ) VALUES (
            :year, 4000, 80.0, CAST(:intervals AS jsonb), CAST(:weekdays AS jsonb),
            :updated_at, :user_id
        )
        ON CONFLICT (year) DO UPDATE SET
            annual_planting_target = excluded.annual_planting_target,
            min_survival_target_pct = excluded.min_survival_target_pct,
            inspection_intervals_json = excluded.inspection_intervals_json,
            inspection_weekdays_json = excluded.inspection_weekdays_json,
            updated_at = excluded.updated_at,
            updated_by_user_id = excluded.updated_by_user_id
    """), {
        "year": now.year, "intervals": json.dumps([30, 60, 90, 120]),
        "weekdays": json.dumps([2, 5]), "updated_at": now,
        "user_id": inspector_id,
    })

    completed_days_ago = (130, 110, 95)
    recent_days_ago = (24, 18, 12)
    all_events: list[dict] = []
    organization_ids: list[int] = []
    schedule_count = 0

    for org_index, (spec, analysis) in enumerate(zip(ORGANIZATIONS, candidates)):
        organization_id = _insert_returning_id(connection, """
            INSERT INTO organizations (
                name, normalized_name, inspection_interval_days, created_at, updated_at
            ) VALUES (:name, :key, 30, :created_at, :updated_at)
            RETURNING id
        """, {
            "name": spec["name"], "key": spec["key"],
            "created_at": now - timedelta(days=150), "updated_at": now,
        })
        organization_ids.append(organization_id)
        polygon, center_lat, center_lon = _site_polygon(connection, int(analysis["id"]))
        site_id = _insert_returning_id(connection, """
            INSERT INTO project_sites (
                name, notes, polygon_geojson, organization_id,
                inspection_interval_days, created_at, updated_at
            ) VALUES (
                :name, :notes, CAST(:polygon AS jsonb), :organization_id,
                30, :created_at, :updated_at
            ) RETURNING id
        """, {
            "name": spec["site"],
            "notes": f"[{SEED_MARKER}] Demonstration project site for consultation.",
            "polygon": json.dumps(polygon), "organization_id": organization_id,
            "created_at": now - timedelta(days=145), "updated_at": now,
        })
        connection.execute(text("""
            UPDATE analyses SET project_site_id = :site_id WHERE id = :analysis_id
        """), {"site_id": site_id, "analysis_id": int(analysis["id"])})

        planter_rows = []
        for full_name, username, phone in spec["planters"]:
            planter_rows.append({
                "full_name": full_name, "organization_id": organization_id,
                "phone": phone, "base_label": spec["site"],
                "base_lat": center_lat, "base_lon": center_lon,
                "notes": f"[{SEED_MARKER}] Consultation demonstration planter.",
                "username": username, "password_hash": hash_password(DEMO_PASSWORD),
                "created_at": now - timedelta(days=140),
            })
        connection.execute(text("""
            INSERT INTO planters (
                full_name, organization_id, phone, base_label, base_lat, base_lon,
                status, notes, username, password_hash, created_at
            ) VALUES (
                :full_name, :organization_id, :phone, :base_label, :base_lat, :base_lon,
                'active', :notes, :username, :password_hash, :created_at
            )
        """), planter_rows)
        planter_map = {
            row["username"]: dict(row)
            for row in connection.execute(text("""
                SELECT id, username, full_name FROM planters
                WHERE organization_id = :organization_id
            """), {"organization_id": organization_id}).mappings().all()
        }
        points = _safe_points(connection, int(analysis["id"]))
        completed_planted = COMPLETED_PLANTED_COUNTS[org_index]
        completed_skipped = COMPLETED_SKIPPED_COUNTS[org_index]
        completed_assignment_size = completed_planted + completed_skipped
        active_planted = ACTIVE_PLANTED_COUNTS[org_index]
        assignment_specs = (
            {
                "kind": "completed", "planter": spec["planters"][0][1],
                "points": points[:completed_assignment_size],
                "completed": completed_planted, "skipped": completed_skipped,
                "planted_at": now - timedelta(days=completed_days_ago[org_index]),
            },
            {
                "kind": "active", "planter": spec["planters"][1][1],
                "points": points[completed_assignment_size:],
                "completed": active_planted, "skipped": 0,
                "planted_at": now - timedelta(days=recent_days_ago[org_index]),
            },
        )
        for assignment_spec in assignment_specs:
            planter = planter_map[assignment_spec["planter"]]
            assignment_status = (
                "completed" if assignment_spec["kind"] == "completed" else "active"
            )
            assignment_date = assignment_spec["planted_at"].date()
            assignment_id = _insert_returning_id(connection, """
                INSERT INTO planter_assignments (
                    planter_id, assigned_by_user_id, title, assignment_date,
                    travel_mode, status, species, notes, project_site_id, created_at
                ) VALUES (
                    :planter_id, :user_id, :title, :assignment_date,
                    'walking', :status, :species, :notes, :site_id, :created_at
                ) RETURNING id
            """, {
                "planter_id": planter["id"], "user_id": inspector_id,
                "title": (
                    f"{spec['site']} completed planting run"
                    if assignment_status == "completed"
                    else f"{spec['site']} current planting assignment"
                ),
                "assignment_date": assignment_date, "status": assignment_status,
                "species": spec["species"],
                "notes": f"[{SEED_MARKER}] Assigned through the LGU planning workflow.",
                "site_id": site_id,
                "created_at": assignment_spec["planted_at"] - timedelta(days=2),
            })
            assignment_point_rows = []
            for position, point in enumerate(assignment_spec["points"]):
                if position < assignment_spec["completed"]:
                    point_status = "completed"
                    completed_at = assignment_spec["planted_at"] + timedelta(minutes=position * 8)
                    skip_reason = None
                elif position < assignment_spec["completed"] + assignment_spec["skipped"]:
                    point_status = "skipped"
                    completed_at = None
                    skip_reason = "Unsafe footing observed during the field activity."
                else:
                    point_status = "pending"
                    completed_at = None
                    skip_reason = None
                assignment_point_rows.append({
                    "assignment_id": assignment_id, "point_id": point["id"],
                    "sequence": position + 1, "status": point_status,
                    "completed_at": completed_at, "notes": f"[{SEED_MARKER}]",
                    "assigned_at": assignment_spec["planted_at"] - timedelta(days=2),
                    "status_changed_at": completed_at or now,
                    "skip_reason": skip_reason,
                })
            connection.execute(text("""
                INSERT INTO planter_assignment_points (
                    assignment_id, planting_point_id, sequence_num, status,
                    completed_at, notes, assigned_at, status_changed_at, skip_reason
                ) VALUES (
                    :assignment_id, :point_id, :sequence, :status,
                    :completed_at, :notes, :assigned_at, :status_changed_at, :skip_reason
                )
            """), assignment_point_rows)

            saved_points = connection.execute(text("""
                SELECT pap.id AS assignment_point_id, pap.sequence_num, pap.status,
                       pp.id AS point_id, pp.point_num, pp.latitude, pp.longitude
                FROM planter_assignment_points pap
                JOIN planting_points pp ON pp.id = pap.planting_point_id
                WHERE pap.assignment_id = :assignment_id
                ORDER BY pap.sequence_num
            """), {"assignment_id": assignment_id}).mappings().all()
            planted_point_rows = []
            skipped_point_rows = []
            event_rows = []
            event_meta_by_source = {}
            for point in saved_points:
                if point["status"] == "completed":
                    planted_at = assignment_spec["planted_at"] + timedelta(
                        minutes=(int(point["sequence_num"]) - 1) * 8
                    )
                    planted_point_rows.append({
                        "planted_at": planted_at, "planted_date": planted_at.date(),
                        "point_id": point["point_id"],
                    })
                    source_key = (
                        f"{SEED_MARKER}:{organization_id}:{assignment_id}:"
                        f"{point['assignment_point_id']}"
                    )
                    event_rows.append({
                        "source_key": source_key,
                        "point_id": point["point_id"],
                        "assignment_point_id": point["assignment_point_id"],
                        "assignment_id": assignment_id, "site_id": site_id,
                        "planter_id": planter["id"], "species": spec["species"],
                        "planted_at": planted_at, "point_num": point["point_num"],
                        "latitude": point["latitude"], "longitude": point["longitude"],
                        "created_at": planted_at,
                    })
                    event_meta_by_source[source_key] = {
                        "point_id": int(point["point_id"]),
                        "assignment_id": assignment_id,
                        "planter_id": int(planter["id"]),
                        "planter_name": planter["full_name"],
                        "species": spec["species"], "planted_at": planted_at,
                        "mortality_profile": spec["mortality_profile"],
                        "position": int(point["sequence_num"]) - 1,
                    }
                elif point["status"] == "skipped":
                    skipped_point_rows.append({"point_id": point["point_id"]})

            if planted_point_rows:
                connection.execute(text("""
                    UPDATE planting_points
                    SET status = 'planted', planted_at = :planted_at,
                        planted_date = :planted_date
                    WHERE id = :point_id
                """), planted_point_rows)
            if skipped_point_rows:
                connection.execute(text("""
                    UPDATE planting_points SET status = 'skipped'
                    WHERE id = :point_id
                """), skipped_point_rows)
            if event_rows:
                connection.execute(text("""
                    INSERT INTO planting_events (
                        source_key, planting_point_id, assignment_point_id,
                        assignment_id, project_site_id, planter_id, species,
                        planted_at, point_num, latitude, longitude, source,
                        site_attribution_source, inspection_interval_days, created_at
                    ) VALUES (
                        :source_key, :point_id, :assignment_point_id,
                        :assignment_id, :site_id, :planter_id, :species,
                        :planted_at, :point_num, :latitude, :longitude,
                        'field_completion', 'planting_snapshot', 30, :created_at
                    )
                """), event_rows)
                saved_events = connection.execute(text("""
                    SELECT id, source_key FROM planting_events
                    WHERE assignment_id = :assignment_id
                    ORDER BY id
                """), {"assignment_id": assignment_id}).mappings().all()
                for saved_event in saved_events:
                    metadata = event_meta_by_source[saved_event["source_key"]]
                    all_events.append({"id": int(saved_event["id"]), **metadata})

        schedule_rows = (
            (-100, "completed", "Completed community planting activity", 42, 120),
            (-1, "in_progress", "Ongoing shoreline planting activity", 18, 55),
            (7, "confirmed", "Confirmed volunteer planting activity", 30, 90),
            (21, "scheduled", "Upcoming expansion planting activity", 25, 75),
            (35, "tentative", "Tentative follow-up planting activity", 20, 60),
        )
        for day_offset, status, title_value, expected_planters, seedlings in schedule_rows:
            start_at = (now + timedelta(days=day_offset)).replace(hour=7, minute=30)
            if status == "in_progress":
                start_at = now - timedelta(hours=1)
            end_at = start_at + timedelta(hours=4)
            connection.execute(text("""
                INSERT INTO planting_schedules (
                    project_site_id, organization, organization_id,
                    inspection_interval_days, contact, title, start_at, end_at,
                    expected_planters, expected_seedlings, status, notes,
                    created_by_user_id, updated_by_user_id, created_at, updated_at
                ) VALUES (
                    :site_id, :organization, :organization_id, 30, :contact,
                    :title, :start_at, :end_at, :expected_planters, :seedlings,
                    :status, :notes, :user_id, :user_id, :created_at, :updated_at
                )
            """), {
                "site_id": site_id, "organization": spec["name"],
                "organization_id": organization_id, "contact": spec["contact"],
                "title": title_value, "start_at": start_at, "end_at": end_at,
                "expected_planters": expected_planters, "seedlings": seedlings,
                "status": status,
                "notes": f"[{SEED_MARKER}] Consultation scheduling example.",
                "user_id": inspector_id, "created_at": now - timedelta(days=140),
                "updated_at": now,
            })
            schedule_count += 1

    observation_count = _create_observations(connection, all_events, inspector_id, now)
    organization_totals = connection.execute(text("""
        SELECT o.id AS organization_id,
               count(DISTINCT pe.id) AS planted_count,
               count(DISTINCT pdr.id) AS dead_count
        FROM organizations o
        JOIN project_sites ps ON ps.organization_id = o.id
        JOIN planting_events pe ON pe.project_site_id = ps.id
                              AND pe.source_key LIKE :event_like
        LEFT JOIN point_death_records pdr ON pdr.planting_event_id = pe.id
        WHERE o.normalized_name LIKE 'consultation-demo-%'
        GROUP BY o.id
        ORDER BY o.id
    """), {"event_like": f"{SEED_MARKER}:%"}).mappings().all()
    monitoring_rows = []
    for org_index, total in enumerate(organization_totals):
        planted_count = int(total["planted_count"])
        final_dead_count = int(total["dead_count"])
        for visit_index, (days_ago, coverage) in enumerate(((60, 0.45), (30, 0.72), (1, 1.0))):
            monitored_count = round(planted_count * coverage)
            dead_count = round(final_dead_count * coverage)
            monitoring_rows.append({
                "organization_id": total["organization_id"],
                "monitored_at": now - timedelta(days=days_ago),
                "alive": monitored_count - dead_count,
                "dead": dead_count,
                "height": round(33.0 + visit_index * 8.4 + org_index * 2.1, 1),
                "health": ("fair", "good", "good")[visit_index],
                "actions": (
                    f"[{SEED_MARKER}] Cleared debris, checked stakes, and documented survival."
                ),
                "inspector_id": inspector_id,
                "created_at": now - timedelta(days=days_ago),
            })
    connection.execute(text("""
        INSERT INTO organization_monitoring_records (
            organization_id, monitored_at, alive_count, dead_count,
            average_height_cm, health_status, actions_taken,
            inspector_user_id, created_at
        ) VALUES (
            :organization_id, :monitored_at, :alive, :dead,
            :height, :health, :actions, :inspector_id, :created_at
        )
    """), monitoring_rows)
    return {
        "organizations": len(organization_ids),
        "planters": len(organization_ids) * 2,
        "assigned_points": len(organization_ids) * POINTS_PER_ORGANIZATION,
        "planted_points": len(all_events),
        "schedules": schedule_count,
        "observations": observation_count,
        "deaths": sum(int(row["dead_count"]) for row in organization_totals),
    }


def _normalize_species_to_bungalon(connection) -> dict[str, int]:
    """Normalize every species snapshot belonging to the tagged demo only."""
    statements = {
        "analyses": """
            UPDATE analyses a
            SET species = :species
            FROM project_sites ps
            JOIN organizations o ON o.id = ps.organization_id
            WHERE a.project_site_id = ps.id
              AND o.normalized_name LIKE 'consultation-demo-%'
              AND a.species IS DISTINCT FROM :species
        """,
        "planter_assignments": """
            UPDATE planter_assignments
            SET species = :species
            WHERE notes LIKE :marker_like
              AND species IS DISTINCT FROM :species
        """,
        "planting_events": """
            UPDATE planting_events
            SET species = :species
            WHERE source_key LIKE :event_like
              AND species IS DISTINCT FROM :species
        """,
        "point_death_records": """
            UPDATE point_death_records pdr
            SET species = :species
            FROM planting_events pe
            WHERE pdr.planting_event_id = pe.id
              AND pe.source_key LIKE :event_like
              AND pdr.species IS DISTINCT FROM :species
        """,
    }
    values = {
        "species": DEMO_SPECIES,
        "marker_like": f"%[{SEED_MARKER}]%",
        "event_like": f"{SEED_MARKER}:%",
    }
    return {
        table: int(connection.execute(text(sql), values).rowcount or 0)
        for table, sql in statements.items()
    }


def _verify() -> dict:
    engine = get_engine()
    with engine.connect() as connection:
        marker_like = f"%[{SEED_MARKER}]%"
        row = connection.execute(text("""
            SELECT
                (SELECT count(*) FROM organizations
                 WHERE normalized_name LIKE 'consultation-demo-%') AS organizations,
                (SELECT count(*) FROM planters p JOIN organizations o ON o.id=p.organization_id
                 WHERE o.normalized_name LIKE 'consultation-demo-%') AS planters,
                (SELECT count(*) FROM planter_assignments WHERE notes LIKE :marker_like) AS assignments,
                (SELECT count(*) FROM planter_assignment_points pap
                 JOIN planter_assignments pa ON pa.id=pap.assignment_id
                 WHERE pa.notes LIKE :marker_like AND pap.status='pending') AS pending,
                (SELECT count(*) FROM planter_assignment_points pap
                 JOIN planter_assignments pa ON pa.id=pap.assignment_id
                 WHERE pa.notes LIKE :marker_like AND pap.status='skipped') AS skipped,
                (SELECT count(*) FROM planting_events
                 WHERE source_key LIKE :event_like) AS planted,
                (SELECT count(*) FROM monitoring_observations mo
                 JOIN planting_events pe ON pe.id=mo.planting_event_id
                 WHERE pe.source_key LIKE :event_like) AS inspections,
                (SELECT count(*) FROM point_death_records pdr
                 JOIN planting_events pe ON pe.id=pdr.planting_event_id
                 WHERE pe.source_key LIKE :event_like) AS deaths,
                (SELECT count(*) FROM planting_schedules
                 WHERE notes LIKE :marker_like) AS schedules,
                (SELECT count(*) FROM organization_monitoring_records omr
                 JOIN organizations o ON o.id=omr.organization_id
                 WHERE o.normalized_name LIKE 'consultation-demo-%') AS organization_visits,
                (SELECT count(*) FROM project_sites ps
                 JOIN organizations o ON o.id=ps.organization_id
                 WHERE o.normalized_name LIKE 'consultation-demo-%') AS project_sites,
                (SELECT count(*) FROM project_sites ps
                 JOIN organizations o ON o.id=ps.organization_id
                 WHERE o.normalized_name LIKE 'consultation-demo-%'
                   AND ps.name = o.name) AS matching_site_names,
                (SELECT count(*) FROM planter_assignments
                 WHERE notes LIKE :marker_like AND status='active') AS active_assignments,
                (SELECT count(*) FROM planter_assignments
                 WHERE notes LIKE :marker_like AND status='completed') AS completed_assignments,
                (SELECT count(*) FROM analyses a
                 JOIN project_sites ps ON ps.id=a.project_site_id
                 JOIN organizations o ON o.id=ps.organization_id
                 WHERE o.normalized_name LIKE 'consultation-demo-%'
                   AND NULLIF(TRIM(a.species), '') IS DISTINCT FROM :species) AS non_bungalon_analyses,
                (SELECT count(*) FROM planter_assignments
                 WHERE notes LIKE :marker_like
                   AND NULLIF(TRIM(species), '') IS DISTINCT FROM :species) AS non_bungalon_assignments,
                (SELECT count(*) FROM planting_events
                 WHERE source_key LIKE :event_like
                   AND NULLIF(TRIM(species), '') IS DISTINCT FROM :species) AS non_bungalon_events,
                (SELECT count(*) FROM point_death_records pdr
                 JOIN planting_events pe ON pe.id=pdr.planting_event_id
                 WHERE pe.source_key LIKE :event_like
                   AND NULLIF(TRIM(pdr.species), '') IS DISTINCT FROM :species) AS non_bungalon_deaths,
                (SELECT count(*) FROM planting_points pp
                 WHERE pp.deleted_at IS NULL AND pp.status='planned'
                   AND NOT EXISTS (
                       SELECT 1 FROM planter_assignment_points pap
                       JOIN planter_assignments pa ON pa.id=pap.assignment_id
                       WHERE pap.planting_point_id=pp.id
                         AND pa.status IN ('active','completed')
                   )
                   AND NOT EXISTS (
                       SELECT 1 FROM map_zones mz
                       WHERE mz.deleted_at IS NULL AND mz.zone_type='eroded'
                         AND extensions.ST_Covers(mz.geometry, pp.location)
                   )) AS map_analytics_planned
        """), {
            "marker_like": marker_like,
            "event_like": f"{SEED_MARKER}:%",
            "species": DEMO_SPECIES,
        }).mappings().one()
    from planting_database import (
        get_dashboard_ecology,
        get_dashboard_operations,
        get_dashboard_overview,
        get_dashboard_sites,
        get_due_monitoring_inspections,
    )

    due = get_due_monitoring_inspections(upcoming_days=30)["summary"]
    ecology = get_dashboard_ecology()
    dashboard_sections = (
        get_dashboard_overview(),
        get_dashboard_operations(),
        ecology,
        get_dashboard_sites(),
    )
    if any(not isinstance(section, dict) or not section for section in dashboard_sections):
        raise RuntimeError("One or more dashboard data providers returned an empty payload.")
    site_survival_rates = {
        item.get("name") or item.get("site_name"): item.get("survival_rate_pct")
        for item in ecology.get("site_outcomes", [])
        if item.get("name") or item.get("site_name")
        if int(item.get("interval_days") or 0) == 30
    }
    expected_low_sites = {spec["site"] for spec in ORGANIZATIONS[1:]}
    low_survival_sites = sum(
        1 for name, rate in site_survival_rates.items()
        if name in expected_low_sites and rate is not None and float(rate) < 80.0
    )
    result = dict(row)
    result.update({
        "overdue_inspections": int(due["overdue"]),
        "upcoming_inspections": int(due["upcoming"]),
        "low_survival_sites": low_survival_sites,
        "site_survival_rates": site_survival_rates,
        "dashboard_sections": len(dashboard_sections),
    })
    required_positive = (
        "organizations", "planters", "assignments", "pending", "skipped", "planted",
        "inspections", "deaths", "schedules", "overdue_inspections",
        "upcoming_inspections", "organization_visits", "project_sites",
        "matching_site_names", "active_assignments", "completed_assignments", "map_analytics_planned",
        "low_survival_sites", "dashboard_sections",
    )
    missing = [key for key in required_positive if int(result.get(key) or 0) <= 0]
    if missing:
        raise RuntimeError("Demo verification failed for: " + ", ".join(missing))
    if int(result["planted"]) != 1000:
        raise RuntimeError(f"Expected exactly 1000 planted points; found {result['planted']}.")
    if int(result["matching_site_names"]) != len(ORGANIZATIONS):
        raise RuntimeError("One or more project-site names do not match their organization.")
    if int(result["low_survival_sites"]) != len(expected_low_sites):
        raise RuntimeError(
            f"Expected {len(expected_low_sites)} low-survival sites; "
            f"found {result['low_survival_sites']}: {site_survival_rates}"
        )
    species_mismatch_fields = (
        "non_bungalon_analyses", "non_bungalon_assignments",
        "non_bungalon_events", "non_bungalon_deaths",
    )
    species_mismatches = {
        key: int(result[key]) for key in species_mismatch_fields if int(result[key]) != 0
    }
    if species_mismatches:
        raise RuntimeError(f"Non-Bungalon species data remains: {species_mismatches}")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Print current counts and eligible analyses without writing data.",
    )
    parser.add_argument(
        "--replace", action="store_true",
        help="Replace only records tagged with this consultation seed marker.",
    )
    parser.add_argument(
        "--verify-only", action="store_true",
        help="Verify existing consultation data and dashboard providers without writing.",
    )
    parser.add_argument(
        "--normalize-bungalon", action="store_true",
        help="Set all tagged consultation-demo species attribution to Bungalon.",
    )
    args = parser.parse_args()
    engine = get_engine()
    if args.dry_run:
        with engine.connect() as connection:
            _inventory(connection)
        return 0
    if args.verify_only:
        verified = _verify()
        print("Consultation demo verification passed")
        for key, value in verified.items():
            print(f"  verified_{key}: {value}")
        return 0
    if args.normalize_bungalon:
        with engine.begin() as connection:
            updated = _normalize_species_to_bungalon(connection)
        verified = _verify()
        print("Species normalization committed")
        for table, count in updated.items():
            print(f"  updated_{table}: {count}")
        for key in (
            "non_bungalon_analyses", "non_bungalon_assignments",
            "non_bungalon_events", "non_bungalon_deaths",
        ):
            print(f"  verified_{key}: {verified[key]}")
        return 0

    with engine.begin() as connection:
        existing = _active_demo_count(connection)
        if existing and not args.replace:
            print(
                f"Consultation demo data already exists ({existing} organizations); "
                "no changes made. Use --replace to rebuild only tagged demo rows."
            )
            return 0
        if existing:
            _cleanup_demo(connection)
        summary = _seed(connection)

    verified = _verify()
    print("Consultation demo seed committed")
    for key, value in summary.items():
        print(f"  inserted_{key}: {value}")
    for key, value in verified.items():
        print(f"  verified_{key}: {value}")
    print("Demo planter usernames: demo.ana, demo.roberto, demo.joshua, demo.leah, demo.elena, demo.mark")
    print(f"Demo planter password: {DEMO_PASSWORD}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
