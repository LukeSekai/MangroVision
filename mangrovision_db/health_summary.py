"""Current mapped seedling health, counted once per physical location."""

from datetime import datetime

from .point_status import current_point_status


def _timestamp(value, timezone):
    if not value:
        return None
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return parsed.replace(tzinfo=timezone) if parsed.tzinfo is None else parsed


def _counts():
    return {'total': 0, 'planted': 0, 'dead': 0}


def _with_survival_rate(counts):
    return {
        **counts,
        'survival_rate_pct': round(counts['planted'] / counts['total'] * 100, 2)
        if counts['total'] else None,
    }


def summarize_current_health(points, events, as_of, *, site_names=None, species_labels=None):
    """Count current planted and dead locations across all planting dates.

    Organization balances and inspection rounds are never added to these
    counts. Located deaths already update the canonical map status. Old
    planting cycles and locations released for replanting stay in history.
    """
    site_names = site_names or {}
    species_labels = species_labels or {}
    latest_events = {}
    for event in events:
        if event.get('planting_point_id') is None:
            continue
        key = int(event['planting_point_id'])
        previous = latest_events.get(key)
        rank = (_timestamp(event['planted_at'], as_of.tzinfo), int(event['id']))
        if previous is None or rank > (
            _timestamp(previous['planted_at'], as_of.tzinfo), int(previous['id'])
        ):
            latest_events[key] = event

    total = _counts()
    groups = {'species_outcomes': {}, 'site_outcomes': {}, 'planter_outcomes': {}}
    seen = set()
    for point in points:
        status = current_point_status(point)
        point_id = int(point['id'])
        if status not in {'planted', 'dead'} or point_id in seen:
            continue
        seen.add(point_id)
        event = latest_events.get(point_id) or {}
        raw_species = str(point.get('species') or '').strip()
        species_key = raw_species.casefold().replace('_', '-')
        species = species_labels.get(species_key) or raw_species or 'Unspecified'
        species_key = species.casefold()
        site_id = point.get('site_id')
        site_name = site_names.get(site_id) or (
            event.get('site_name') if site_id == event.get('site_id') else None
        ) or 'Unassigned site'
        planter_id = point.get('planter_id')
        planter_name = event.get('planter_name') or 'Unattributed'
        metadata = {
            'species_outcomes': (species_key, {'name': species, 'species_name': species}),
            'site_outcomes': (site_id, {'name': site_name, 'site_name': site_name, 'site_id': site_id}),
            'planter_outcomes': ((planter_id, site_id, species_key), {
                'name': planter_name, 'planter_name': planter_name, 'planter_id': planter_id,
                'site_name': site_name, 'site_id': site_id, 'species': species,
            }),
        }
        targets = [total]
        for group_name, (key, labels) in metadata.items():
            targets.append(groups[group_name].setdefault(key, {**labels, **_counts()}))
        for target in targets:
            target['total'] += 1
            target[status] += 1

    return {
        'summary': {**_with_survival_rate(total), 'scope': 'current_map'},
        **{
            name: sorted((_with_survival_rate(row) for row in values.values()), key=lambda row: str(row['name']).casefold())
            for name, values in groups.items()
        },
    }
