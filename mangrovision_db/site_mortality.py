"""Automatic, non-blocking warning zones from each site's planting outcomes."""

MORTALITY_WARNING_PERCENT = 30


def mortality_warning(history=None):
    history = history or {}
    total = int(history.get('total') or 0)
    dead = int(history.get('dead') or 0)
    return {
        'active': total > 0 and dead * 100 >= total * MORTALITY_WARNING_PERCENT,
        'total_planted': total,
        'dead_count': dead,
        'unknown_count': int(history.get('unknown') or 0),
        'mortality_rate_pct': round(dead / total * 100, 2) if total else 0,
        'threshold_percentage': MORTALITY_WARNING_PERCENT,
        'basis': 'planting_history',
    }


def mortality_warning_zones(sites):
    """Derive one zone per active site without storing a second copy of its shape.

    Negative IDs keep automatic zones separate from manually drawn warning IDs.
    Corrections and boundary changes take effect on the next read. Released land
    has no geometry, while its original planting outcomes remain in site history.
    """
    features = []
    for site in sites:
        props = site.get('properties') or {}
        warning = props.get('mortality_warning') or {}
        if not warning.get('active') or not site.get('geometry'):
            continue
        site_id = int(props.get('id') or site['id'])
        notes = f"High mortality ({warning['mortality_rate_pct']}% dead). Check before planting."
        features.append({
            'type': 'Feature', 'id': -site_id, 'geometry': site['geometry'],
            'properties': {
                'id': -site_id, 'zone_type': 'warning', 'name': props.get('name') or f'Site {site_id}',
                'warning_type': 'high_mortality', 'warning_label': 'High mortality (at least 30%)',
                'severity': 'high', 'notes': notes, 'automatic': True,
                'project_site_id': site_id, 'organization_id': props.get('organization_id'),
                'mortality_warning': warning,
            },
        })
    return features
