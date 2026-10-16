"""Derive site warnings from distinct planted locations and recorded wave deaths.

Replanting does not erase the environmental history of a location. Storms,
barnacles, other causes and unlocated organization totals never count as waves.
"""

WAVE_WARNING_PERCENT = 30
WAVE_WARNING_MESSAGE = 'Danger: This area is prone to big waves.'


def wave_risk(planted_points=0, wave_death_points=0):
    total, deaths = int(planted_points), int(wave_death_points)
    active = total > 0 and deaths * 100 >= total * WAVE_WARNING_PERCENT
    return {'active': active, 'cause': 'waves', 'planted_points': total,
            'wave_death_points': deaths, 'percentage': round(deaths / total * 100, 2) if total else 0,
            'threshold_percentage': WAVE_WARNING_PERCENT, 'basis': 'recorded_history',
            'message': WAVE_WARNING_MESSAGE if active else None}


def project_site_wave_risks(conn):
    """Batch all sites; event-time ownership wins over later boundary/link edits."""
    import planting_database as db
    rows = conn.execute(f'''
        WITH point_sites AS (
            SELECT pp.id AS point_id, pp.status, pp.planted_at, pp.death_at,
                   pp.death_reason_category,
                   {db._point_project_site_owner_sql('a.site_zone_id')} AS site_id
            FROM planting_points pp JOIN analyses a ON a.id = pp.analysis_id
            LEFT JOIN project_sites ps ON extensions.ST_Covers(ps.geometry, pp.location)
            GROUP BY pp.id
        ), valid_events AS (
            SELECT pe.id, pe.planting_point_id AS point_id,
                   COALESCE(pe.site_zone_id, point.site_id) AS site_id
            FROM planting_events pe LEFT JOIN point_sites point ON point.point_id = pe.planting_point_id
            WHERE COALESCE(pe.closure_reason, '') <> 'completion_reversed'
        ), planted_locations AS (
            SELECT site_id, point_id FROM valid_events
            UNION
            SELECT point.site_id, point.point_id FROM point_sites point
            WHERE (point.status = 'planted' OR point.planted_at IS NOT NULL OR point.death_at IS NOT NULL)
              AND NOT EXISTS (SELECT 1 FROM valid_events event WHERE event.point_id = point.point_id)
            UNION
            SELECT COALESCE(event.site_id, point.site_id), death.planting_point_id
            FROM point_death_records death
            LEFT JOIN valid_events event ON event.id = death.planting_event_id
            LEFT JOIN point_sites point ON point.point_id = death.planting_point_id
        ), wave_locations AS (
            SELECT COALESCE(event.site_id, point.site_id) AS site_id, death.planting_point_id AS point_id
            FROM point_death_records death
            LEFT JOIN valid_events event ON event.id = death.planting_event_id
            LEFT JOIN point_sites point ON point.point_id = death.planting_point_id
            WHERE death.reason_category = 'waves'
            UNION
            SELECT point.site_id, point.point_id FROM point_sites point
            WHERE point.death_at IS NOT NULL AND point.death_reason_category = 'waves'
              AND NOT EXISTS (SELECT 1 FROM point_death_records death WHERE death.planting_point_id = point.point_id)
        ), totals AS (
            SELECT site_id, COUNT(DISTINCT point_id) AS planted_points
            FROM planted_locations WHERE site_id IS NOT NULL AND point_id IS NOT NULL GROUP BY site_id
        ), waves AS (
            SELECT site_id, COUNT(DISTINCT point_id) AS wave_death_points
            FROM wave_locations WHERE site_id IS NOT NULL AND point_id IS NOT NULL GROUP BY site_id
        )
        SELECT totals.site_id, totals.planted_points, COALESCE(waves.wave_death_points, 0) AS wave_death_points
        FROM totals LEFT JOIN waves ON waves.site_id = totals.site_id
    ''').fetchall()
    return {int(row['site_id']): wave_risk(row['planted_points'], row['wave_death_points']) for row in rows}
