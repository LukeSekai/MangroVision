"""Planting-batch outcomes and replacement progress, independent of map status."""

from datetime import datetime
from zoneinfo import ZoneInfo

MANILA = ZoneInfo('Asia/Manila')


def _date(value):
    if not value:
        return None
    parsed = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace('Z', '+00:00'))
    return parsed.replace(tzinfo=MANILA) if parsed.tzinfo is None else parsed


def _counts():
    return {'total': 0, 'planted': 0, 'dead': 0, 'unknown': 0}


def _rates(row):
    # Missing outcomes must not turn into assumed survivors or deaths.
    denominator = row['total'] if not row['unknown'] else 0
    return {**row,
            'survival_rate_pct': round(row['planted'] / denominator * 100, 2) if denominator else None,
            'mortality_rate_pct': round(row['dead'] / denominator * 100, 2) if denominator else None}


def summarize_planting_performance(events, start, end, as_of, *, site_id=None,
                                   assignment_id=None, species=None, planter_id=None,
                                   species_labels=None, points=()):
    """Select batches by planting date; evaluate their saved outcomes as of now.

    A replacement is a separate seedling. Approvals and assignments do not add
    plantings or remove deaths. Replanting progress covers all planting dates
    for the selected dimensions and follows each original seedling's outcome.
    """
    species_labels = species_labels or {}
    unique = {}
    for event in events:
        planted_at = _date(event.get('planted_at'))
        if (planted_at and planted_at <= as_of
                and event.get('closure_reason') != 'completion_reversed'):
            unique[int(event['id'])] = {**event, '_planted_at': planted_at}
    ordered = sorted(unique.values(), key=lambda row: (row['_planted_at'], int(row['id'])))
    previous_by_point = {}
    replacement_for = {}
    replacement_ids = set()
    superseded = set()
    for event in ordered:
        point_id = event.get('planting_point_id')
        previous = previous_by_point.get(point_id) if point_id is not None else None
        parent = unique.get(event.get('replaces_event_id'))
        if parent and (parent.get('planting_point_id') != point_id
                       or (parent['_planted_at'], int(parent['id'])) >= (event['_planted_at'], int(event['id']))):
            parent = None
        # Older records may predate explicit replacement links. A documented
        # death followed by planting at the same point still identifies a cycle.
        if parent is None and previous:
            death_at = _date(previous.get('recorded_death_at'))
            if death_at and death_at <= event['_planted_at']:
                parent = previous
        if parent:
            replacement_for.setdefault(int(parent['id']), event)
            replacement_ids.add(int(event['id']))
        if previous:
            superseded.add(int(previous['id']))
        if point_id is not None:
            previous_by_point[point_id] = event

    def matches(row):
        return (all(value is None or row.get(key) == int(value)
                    for key, value in (('site_id', site_id), ('assignment_id', assignment_id), ('planter_id', planter_id)))
                and (not species or str(row.get('species') or '').strip().casefold() == species.strip().casefold()))

    current_points = {int(point['id']): point for point in points}
    progress = {'awaiting_replanting': 0, 'replacements_completed': 0,
                'awaiting_review': 0, 'approved': 0, 'assigned': 0,
                'unavailable': 0, 'scope': 'all_planting_dates'}
    total = _counts()
    batches = {}
    groups = {key: {} for key in ('species_outcomes', 'site_outcomes', 'planter_outcomes')}
    for event in ordered:
        if not matches(event):
            continue
        event_id = int(event['id'])
        death_at = _date(event.get('recorded_death_at'))
        is_dead = death_at is not None and death_at <= as_of
        point = current_points.get(event.get('planting_point_id'), {})
        if is_dead:
            if event_id in replacement_for:
                progress['replacements_completed'] += 1
            elif not point.get('deleted_at') and not point.get('is_deleted'):
                progress['awaiting_replanting'] += 1
                approved_at = _date(event.get('replanting_approved_at'))
                status = ('assigned' if event.get('replanting_assignment_id') is not None else 'approved') if approved_at and approved_at <= as_of else 'awaiting_review'
                progress[status] += 1
                if point.get('eroded_unavailable') or point.get('inside_eroded_zone'):
                    progress['unavailable'] += 1

        if not start <= event['_planted_at'] <= end:
            continue
        closed_at = _date(event.get('closed_at'))
        outcome_missing = (event_id in superseded or (closed_at and closed_at <= as_of)
                           or point.get('deleted_at') or point.get('is_deleted'))
        status = 'dead' if is_dead else 'unknown' if outcome_missing else 'planted'
        raw_species = str(event.get('species') or '').strip()
        species_name = species_labels.get(raw_species.casefold().replace('_', '-')) or raw_species or 'Unspecified'
        site_name = event.get('site_name') or 'Unassigned site'
        planter_name = event.get('planter_name') or 'Unattributed'
        planting_type = 'replacement' if event_id in replacement_ids else 'original'
        day = event['_planted_at'].astimezone(MANILA).date().isoformat()
        key = (day, event.get('assignment_id'), event.get('site_id'), event.get('planter_id'),
               species_name.casefold(), planting_type)
        batch = batches.setdefault(key, {**_counts(), 'id': 'batch:' + '|'.join(str(value) for value in key),
            'planted_date': day,
            'assignment_id': event.get('assignment_id'), 'site_id': event.get('site_id'),
            'site_name': site_name, 'organization_name': event.get('organization_name') or planter_name,
            'species': species_name, 'planting_type': planting_type})
        metadata = {
            'species_outcomes': (species_name.casefold(), {'name': species_name, 'species_name': species_name}),
            'site_outcomes': (event.get('site_id'), {'name': site_name, 'site_name': site_name, 'site_id': event.get('site_id')}),
            'planter_outcomes': ((event.get('planter_id'), event.get('site_id'), species_name.casefold()), {
                'name': planter_name, 'planter_name': planter_name, 'planter_id': event.get('planter_id'),
                'site_name': site_name, 'site_id': event.get('site_id'), 'species': species_name}),
        }
        targets = [total, batch]
        for name, (group_key, labels) in metadata.items():
            targets.append(groups[name].setdefault(group_key, {**labels, **_counts()}))
        for target in targets:
            target['total'] += 1
            target[status] += 1

    return {
        'summary': {**_rates(total), 'scope': 'planting_cycles'},
        'batches': sorted((_rates(row) for row in batches.values()),
                          key=lambda row: (row['planted_date'], row['site_name'], row['planting_type']), reverse=True),
        'replanting': progress,
        **{name: sorted((_rates(row) for row in values.values()), key=lambda row: str(row['name']).casefold())
           for name, values in groups.items()},
    }
