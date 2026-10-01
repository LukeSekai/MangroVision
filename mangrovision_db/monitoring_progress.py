"""Cumulative visit counts and automatic planting-age groups."""
from datetime import datetime
from zoneinfo import ZoneInfo

MANILA = ZoneInfo('Asia/Manila')


def age_snapshot(plantings, monitored_at, total_planted):
    """Calendar age in 14-day groups, never a measured size or maturity claim."""
    observed_date = monitored_at.astimezone(MANILA).date()
    groups = {}
    known = 0
    for planting in plantings:
        value = planting.get('planted_at')
        try:
            planted = value if isinstance(value, datetime) else datetime.fromisoformat(str(value).replace('Z', '+00:00'))
            if planted.tzinfo is None:
                planted = planted.replace(tzinfo=MANILA)
            days = (observed_date - planted.astimezone(MANILA).date()).days
        except (ValueError, TypeError):
            continue
        if days < 0:
            continue
        cycle = days // 14
        species = planting.get('species') or 'Species not recorded'
        key = (cycle, species)
        group = groups.setdefault(key, {
            'species': species, 'completed_cycles': cycle,
            'age_label': f'{cycle * 2}–{cycle * 2 + 2} weeks',
            'stage_label': 'New seedling' if cycle == 0 else 'Growing seedling',
            'planted_count': 0,
        })
        group['planted_count'] += 1
        known += 1
    cohorts = sorted(groups.values(), key=lambda row: (row['completed_cycles'], row['species']))
    missing = max(0, total_planted - known)
    ages = {row['completed_cycles'] for row in cohorts}
    if not cohorts:
        label = 'Planting date not recorded'
    elif len(ages) > 1 or missing:
        label = 'Mixed planting ages'
    else:
        label = f"{cohorts[0]['stage_label']} · {cohorts[0]['age_label']}"
    return {'method': 'planting_age_14d_v1', 'as_of': observed_date.isoformat(),
            'interval_days': 14, 'label': label, 'cohorts': cohorts,
            'missing_date_count': missing}


def carried_counts(latest, plantings, recorded_total, recorded_dead):
    """Legacy deaths are cumulative; never add totals from different visits."""
    event_ids = {int(row['id']) for row in plantings}
    previous = (latest or {}).get('count_snapshot') or {}
    if latest and previous.get('event_ids') is not None:
        new_planted = len(event_ids - set(previous['event_ids']))
        total = int(latest['alive_count']) + int(latest['dead_count']) + new_planted
    else:
        total = max(int(recorded_total), len(event_ids))
        previous_total = int(latest['alive_count']) + int(latest['dead_count']) if latest else 0
        new_planted = max(0, total - previous_total)
    dead = int(recorded_dead)
    if dead > total:
        raise ValueError('Saved deaths exceed the planting total. Review the monitoring history first.')
    return {'total_planted': total, 'previous_dead_count': dead,
            'alive_before_count': total - dead, 'new_planted_count': new_planted,
            'event_ids': sorted(event_ids | set(previous.get('event_ids') or [])),
            'count_conflict': bool(latest and int(latest['dead_count']) < dead)}
