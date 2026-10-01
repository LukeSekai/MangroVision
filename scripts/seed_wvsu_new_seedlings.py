"""Add 100 tagged, newly planted WVSU demo seedlings to existing mapped points.

This is data-only tooling. It leaves the existing 1,001 planting events alone,
uses one transaction, and is safe to run again after a successful seed.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

from sqlalchemy import text

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mangrovision_db.compat import get_engine


MARKER = "wvsu-new-seedlings-20260929"
ANALYSIS_ID = 141
SEEDLING_COUNT = 100
MANILA = ZoneInfo("Asia/Manila")


def inventory(connection) -> dict:
    organization = connection.execute(text("""
        SELECT id, name FROM organizations WHERE name = 'WVSU'
    """)).mappings().one()
    organization_id = int(organization["id"])
    existing = int(connection.execute(text("""
        SELECT count(*) FROM planting_events WHERE source_key LIKE :marker
    """), {"marker": f"{MARKER}:%"}).scalar_one())
    wvsu_events = int(connection.execute(text("""
        SELECT count(*) FROM planting_events pe
        JOIN planters p ON p.id = pe.planter_id
        WHERE p.organization_id = :organization_id
    """), {"organization_id": organization_id}).scalar_one())
    all_events = int(connection.execute(text("SELECT count(*) FROM planting_events")).scalar_one())
    analysis = connection.execute(text("""
        SELECT id, project_site_id, species FROM analyses
        WHERE id = :analysis_id AND deleted_at IS NULL
    """), {"analysis_id": ANALYSIS_ID}).mappings().one()
    available = int(connection.execute(text("""
        SELECT count(*) FROM planting_points pp
        WHERE pp.analysis_id = :analysis_id AND pp.deleted_at IS NULL
          AND pp.status = 'planned' AND pp.death_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM planter_assignment_points pap
              WHERE pap.planting_point_id = pp.id
          )
          AND NOT EXISTS (
              SELECT 1 FROM map_zones mz
              WHERE mz.deleted_at IS NULL AND mz.zone_type IN ('eroded', 'forbidden')
                AND extensions.ST_Covers(mz.geometry, pp.location)
          )
    """), {"analysis_id": ANALYSIS_ID}).scalar_one())
    return {
        "organization_id": organization_id,
        "analysis_site_id": analysis["project_site_id"],
        "species": analysis["species"] or "Bungalon",
        "available_points": available,
        "tagged_events": existing,
        "wvsu_events": wvsu_events,
        "all_events": all_events,
    }


def seed(connection) -> dict:
    before = inventory(connection)
    if before["tagged_events"] == SEEDLING_COUNT:
        return {"status": "already seeded", **before}
    if before["tagged_events"] or before["wvsu_events"]:
        raise RuntimeError("WVSU already has planting data; review it before seeding.")
    if before["all_events"] != 1001:
        raise RuntimeError(f"Expected 1,001 existing seedlings; found {before['all_events']}.")
    if before["analysis_site_id"] is not None or before["available_points"] < SEEDLING_COUNT:
        raise RuntimeError("Analysis 3 no longer has 100 unassigned safe points.")

    points = connection.execute(text("""
        SELECT pp.id, pp.point_num, pp.latitude, pp.longitude
        FROM planting_points pp
        WHERE pp.analysis_id = :analysis_id AND pp.deleted_at IS NULL
          AND pp.status = 'planned' AND pp.death_at IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM planter_assignment_points pap
              WHERE pap.planting_point_id = pp.id
          )
          AND NOT EXISTS (
              SELECT 1 FROM map_zones mz
              WHERE mz.deleted_at IS NULL AND mz.zone_type IN ('eroded', 'forbidden')
                AND extensions.ST_Covers(mz.geometry, pp.location)
          )
        ORDER BY pp.point_num, pp.id
        LIMIT :seedling_count FOR UPDATE OF pp
    """), {"analysis_id": ANALYSIS_ID, "seedling_count": SEEDLING_COUNT}).mappings().all()
    if len(points) != SEEDLING_COUNT:
        raise RuntimeError("The 100 WVSU points are no longer available.")

    bounds = connection.execute(text("""
        SELECT min(longitude) AS min_lon, max(longitude) AS max_lon,
               min(latitude) AS min_lat, max(latitude) AS max_lat
        FROM planting_points WHERE analysis_id = :analysis_id AND deleted_at IS NULL
    """), {"analysis_id": ANALYSIS_ID}).mappings().one()
    padding = 0.00001
    west, east = float(bounds["min_lon"]) - padding, float(bounds["max_lon"]) + padding
    south, north = float(bounds["min_lat"]) - padding, float(bounds["max_lat"]) + padding
    polygon = {"type": "Polygon", "coordinates": [[
        [west, south], [east, south], [east, north], [west, north], [west, south],
    ]]}
    now = datetime.now(MANILA).replace(microsecond=0)
    planted_date = now.date()
    site_id = int(connection.execute(text("""
        INSERT INTO project_sites (
            name, notes, polygon_geojson, organization_id, inspection_interval_days
        ) VALUES (
            'WVSU', :notes, CAST(:polygon AS jsonb), :organization_id, 14
        ) RETURNING id
    """), {
        "notes": f"[{MARKER}] Demonstration planting site.",
        "polygon": json.dumps(polygon),
        "organization_id": before["organization_id"],
    }).scalar_one())
    connection.execute(text("""
        UPDATE analyses SET project_site_id = :site_id WHERE id = :analysis_id
    """), {"site_id": site_id, "analysis_id": ANALYSIS_ID})

    planter_id = int(connection.execute(text("""
        INSERT INTO planters (full_name, organization_id, status, notes)
        VALUES ('WVSU demo planting', :organization_id, 'inactive', :notes)
        RETURNING id
    """), {
        "organization_id": before["organization_id"],
        "notes": f"[{MARKER}] Data-only demonstration planter; no login credentials.",
    }).scalar_one())
    assignment_id = int(connection.execute(text("""
        INSERT INTO planter_assignments (
            planter_id, title, assignment_date, status, species, notes, project_site_id
        ) VALUES (
            :planter_id, 'WVSU demonstration planting', :planted_date,
            'completed', :species, :notes, :site_id
        ) RETURNING id
    """), {
        "planter_id": planter_id,
        "planted_date": planted_date,
        "species": before["species"],
        "notes": f"[{MARKER}] 100 newly planted demonstration seedlings.",
        "site_id": site_id,
    }).scalar_one())
    point_rows = [{
        "assignment_id": assignment_id,
        "point_id": int(point["id"]),
        "sequence": index + 1,
        "planted_at": now,
    } for index, point in enumerate(points)]
    connection.execute(text("""
        INSERT INTO planter_assignment_points (
            assignment_id, planting_point_id, sequence_num, status,
            completed_at, notes, status_changed_at
        ) VALUES (
            :assignment_id, :point_id, :sequence, 'completed',
            :planted_at, :notes, :planted_at
        )
    """), [{**row, "notes": f"[{MARKER}]"} for row in point_rows])
    assignment_points = connection.execute(text("""
        SELECT id, planting_point_id FROM planter_assignment_points
        WHERE assignment_id = :assignment_id
    """), {"assignment_id": assignment_id}).mappings().all()
    assignment_point_ids = {
        int(row["planting_point_id"]): int(row["id"]) for row in assignment_points
    }
    connection.execute(text("""
        UPDATE planting_points
        SET status = 'planted', planted_at = :planted_at, planted_date = :planted_date
        WHERE id = :point_id
    """), [{**row, "planted_date": planted_date} for row in point_rows])
    connection.execute(text("""
        INSERT INTO planting_events (
            source_key, planting_point_id, assignment_point_id, assignment_id,
            project_site_id, planter_id, species, planted_at, point_num,
            latitude, longitude, source, site_attribution_source,
            inspection_interval_days, created_at
        ) VALUES (
            :source_key, :point_id, :assignment_point_id, :assignment_id,
            :site_id, :planter_id, :species, :planted_at, :point_num,
            :latitude, :longitude, 'field_completion', 'planting_snapshot',
            14, :planted_at
        )
    """), [{
        "source_key": f"{MARKER}:{point['id']}",
        "point_id": int(point["id"]),
        "assignment_point_id": assignment_point_ids[int(point["id"])],
        "assignment_id": assignment_id,
        "site_id": site_id,
        "planter_id": planter_id,
        "species": before["species"],
        "planted_at": now,
        "point_num": int(point["point_num"]),
        "latitude": point["latitude"],
        "longitude": point["longitude"],
    } for point in points])
    after = inventory(connection)
    if after["tagged_events"] != SEEDLING_COUNT or after["wvsu_events"] != SEEDLING_COUNT:
        raise RuntimeError("WVSU seed verification failed; transaction will roll back.")
    if after["all_events"] != 1101:
        raise RuntimeError("Global planting total is not 1,101; transaction will roll back.")
    return {"status": "seeded", "site_id": site_id, "planter_id": planter_id, **after}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dry-run", action="store_true", help="Show counts without writing data.")
    parser.add_argument("--verify-only", action="store_true", help="Check counts without writing data.")
    args = parser.parse_args()
    engine = get_engine()
    if args.dry_run or args.verify_only:
        with engine.connect() as connection:
            result = inventory(connection)
        print(json.dumps(result, indent=2, default=str))
        if args.verify_only and (result["tagged_events"], result["wvsu_events"], result["all_events"]) != (100, 100, 1101):
            raise RuntimeError("Expected 100 WVSU demo plantings and 1,101 total plantings.")
        return
    with engine.begin() as connection:
        result = seed(connection)
    print(json.dumps(result, indent=2, default=str))


if __name__ == "__main__":
    main()
