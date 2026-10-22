"""Current erosion restrictions, independent of recorded seedling condition."""

PLANTING_BLOCKED = "This location is inside an eroded zone. Choose a location outside the zone."
REPLANTING_BLOCKED = "Replanting is unavailable because this location is inside an eroded zone."
ZONE_OVERLAP_MESSAGE = "This area contains recorded seedlings. Their records will be kept, but new planting and replanting here will be blocked."


class SeedlingOverlap(ValueError):
    pass


def lock_erosion_boundaries(conn, *, editing=False):
    # Readers hold a shared transaction lock through planting/release/assignment.
    # Zone writers take the matching exclusive lock, so a submitted action uses
    # one committed boundary version until its own transaction finishes.
    conn.execute("SET LOCAL lock_timeout = '5s'")
    function = 'pg_advisory_xact_lock' if editing else 'pg_advisory_xact_lock_shared'
    conn.execute(f"SELECT {function}(hashtext('mangrovision.eroded_boundaries'))")


def point_is_eroded(conn, latitude, longitude):
    row = conn.execute("""
        SELECT EXISTS (
            SELECT 1 FROM map_zones mz
            WHERE mz.zone_type = 'eroded' AND mz.deleted_at IS NULL
              AND extensions.ST_Covers(mz.geometry,
                  extensions.ST_SetSRID(extensions.ST_MakePoint(?, ?), 4326))
        ) AS is_covered
    """, (float(longitude), float(latitude))).fetchone()
    return bool(row and row['is_covered'])


def recorded_seedling_count(conn, geometry_json):
    row = conn.execute("""
        SELECT COUNT(*) AS count FROM planting_points pp
        WHERE pp.deleted_at IS NULL
          AND extensions.ST_Covers(
              extensions.ST_SetSRID(extensions.ST_GeomFromGeoJSON(?), 4326), pp.location)
          AND (pp.status = 'planted' OR pp.death_at IS NOT NULL
               OR EXISTS (SELECT 1 FROM planting_events pe WHERE pe.planting_point_id = pp.id))
    """, (geometry_json,)).fetchone()
    return int(row['count'])
