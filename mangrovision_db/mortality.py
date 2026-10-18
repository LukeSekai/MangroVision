"""Count reported deaths once, whether or not their locations are identified."""


def _reported_deaths(start, end, *, site_id=None, assignment_id=None,
                     species=None, planter_id=None):
    filters = ['COALESCE(visit.monitored_at, death.death_at) >= ?',
               'COALESCE(visit.monitored_at, death.death_at) <= ?']
    params = [start.isoformat(), end.isoformat()]
    for value, expression in ((site_id, 'event.site_zone_id'),
                              (assignment_id, 'COALESCE(event.assignment_id, death.assignment_id)'),
                              (planter_id, 'COALESCE(event.planter_id, death.planter_id)')):
        if value is not None:
            filters.append(f'{expression} = ?')
            params.append(int(value))
    if species:
        filters.append('LOWER(COALESCE(event.species, death.species)) = LOWER(?)')
        params.append(species.strip())
    # A located death appears once, even when it is linked to a visit and also
    # has a scientific inspection. Approval leaves all these history rows intact.
    located = '''SELECT COALESCE(NULLIF(death.reason_category, 'unknown'),
            visit.count_snapshot ->> 'death_reason_category', 'unknown') AS cause,
            COALESCE(visit.monitored_at, death.death_at) AS reported_at,
            1::bigint AS deaths
        FROM point_death_records death
        LEFT JOIN planting_events event ON event.id = death.planting_event_id
        LEFT JOIN monitoring_death_locations link
            ON link.death_record_id = death.id AND link.revoked_at IS NULL
        LEFT JOIN organization_monitoring_records visit ON visit.id = link.record_id
        WHERE ''' + ' AND '.join(filters)
    # Count-only reports belong to the organization as a whole. Do not invent a
    # site, assignment, species or planter attribution when those filters apply.
    if site_id is None and assignment_id is None and not species and planter_id is None:
        unlocated = '''SELECT COALESCE(visit.count_snapshot ->> 'death_reason_category', 'unknown') AS cause,
                visit.monitored_at AS reported_at,
                GREATEST(0, COALESCE(visit.location_death_count, 0) - COALESCE(located.total, 0)) AS deaths
            FROM organization_monitoring_records visit
            LEFT JOIN (SELECT record_id, COUNT(*) AS total FROM monitoring_death_locations
                WHERE revoked_at IS NULL GROUP BY record_id) located ON located.record_id = visit.id
            WHERE visit.monitored_at >= ? AND visit.monitored_at <= ?'''
        located += ' UNION ALL ' + unlocated
        params.extend([start.isoformat(), end.isoformat()])
    return 'WITH reported_deaths AS (' + located + ') ', params


def mortality_cause_counts(conn, start, end, **filters):
    sql, params = _reported_deaths(start, end, **filters)
    rows = conn.execute(sql + '''SELECT cause, SUM(deaths) AS deaths
        FROM reported_deaths WHERE deaths > 0 GROUP BY cause''', params).fetchall()
    return {row['cause']: int(row['deaths']) for row in rows}


def mortality_monthly_counts(conn, start, end, **filters):
    """New reported deaths per Philippine calendar month, retaining past cycles."""
    sql, params = _reported_deaths(start, end, **filters)
    rows = conn.execute(sql + '''SELECT
        date_trunc('month', reported_at AT TIME ZONE 'Asia/Manila')::date AS period_start,
        SUM(deaths) AS deaths FROM reported_deaths WHERE deaths > 0
        GROUP BY 1 ORDER BY 1''', params).fetchall()
    return {str(row['period_start']): int(row['deaths']) for row in rows}
