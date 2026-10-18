"""Replanting must retain the outcomes of each original planting batch."""
from datetime import datetime
from zoneinfo import ZoneInfo

from mangrovision_db.planting_performance import summarize_planting_performance

MANILA = ZoneInfo('Asia/Manila')
START = datetime(2026, 1, 1, tzinfo=MANILA)
END = datetime(2026, 10, 31, 23, 59, 59, tzinfo=MANILA)
AS_OF = datetime(2026, 10, 18, tzinfo=MANILA)


def seedling(event_id, point_id=None, **values):
    return {'id': event_id, 'planting_point_id': point_id if point_id is not None else event_id,
            'planted_at': '2026-05-16T08:00:00+08:00', 'site_id': 1, 'site_name': 'Coastal site',
            'planter_id': 10, 'planter_name': 'Field team', 'organization_name': 'Coastal guardians',
            'assignment_id': 20, 'species': 'Bungalon', **values}


def report(events, **filters):
    return summarize_planting_performance(events, START, END, AS_OF, **filters)


def original_batch():
    return [seedling(i, **({'recorded_death_at': '2026-06-01T08:00:00+08:00',
                           'closed_at': '2026-06-01T08:00:00+08:00',
                           'closure_reason': 'recorded_death'} if i <= 20 else {}))
            for i in range(1, 101)]


def test_approving_and_assigning_replanting_never_improves_original_survival():
    events = original_batch()
    before = report(events)
    for event in events[:20]:
        event.update(replanting_approved_at='2026-07-01T08:00:00+08:00', replanting_assignment_id=30)
    approved = report(events, points=[{'id': i, 'point_status': 'planned'} for i in range(1, 21)])
    assert before['summary'] == approved['summary'] == {
        'total': 100, 'planted': 80, 'dead': 20, 'unknown': 0,
        'survival_rate_pct': 80, 'mortality_rate_pct': 20, 'scope': 'planting_cycles',
    }
    assert approved['batches'][0]['survival_rate_pct'] == 80
    assert approved['replanting']['assigned'] == approved['replanting']['awaiting_replanting'] == 20
    assert approved['replanting']['replacements_completed'] == 0


def test_actual_replacements_are_new_seedlings_and_original_batch_remains_80_percent():
    originals = original_batch()
    replacements = [seedling(100 + i, i, planted_at='2026-08-01T08:00:00+08:00',
                             assignment_id=30, replaces_event_id=i) for i in range(1, 21)]
    result = report(originals + replacements)
    assert result['summary']['total'] == 120
    assert result['summary']['planted'] == 100
    assert result['summary']['dead'] == 20
    assert result['summary']['survival_rate_pct'] == 83.33
    assert result['replanting']['replacements_completed'] == 20
    assert result['replanting']['awaiting_replanting'] == 0
    batches = {row['planting_type']: row for row in result['batches']}
    assert batches['original']['survival_rate_pct'] == 80
    assert batches['replacement']['total'] == 20
    assert batches['replacement']['survival_rate_pct'] == 100
    assert sum(row['total'] for row in result['site_outcomes']) == 120


def test_multiple_deaths_at_one_location_are_separate_seedlings_and_only_latest_is_pending():
    events = [seedling(1, recorded_death_at='2026-06-01T08:00:00+08:00'),
              seedling(2, 1, planted_at='2026-07-01T08:00:00+08:00', replaces_event_id=1,
                       recorded_death_at='2026-08-01T08:00:00+08:00')]
    result = report(events + [events[0]])
    assert result['summary']['total'] == result['summary']['dead'] == 2
    assert result['summary']['survival_rate_pct'] == 0
    assert result['replanting']['replacements_completed'] == 1
    assert result['replanting']['awaiting_replanting'] == 1


def test_batch_date_filters_keep_later_deaths_and_do_not_reclassify_replacements_as_originals():
    events = [seedling(1, recorded_death_at='2026-07-01T08:00:00+08:00'),
              seedling(2, 1, planted_at='2026-08-01T08:00:00+08:00', replaces_event_id=1,
                       assignment_id=30, planter_id=11, site_id=2)]
    may = summarize_planting_performance(events, START, datetime(2026, 5, 31, tzinfo=MANILA), AS_OF)
    assert may['summary']['dead'] == 1
    assert may['summary']['survival_rate_pct'] == 0
    replacement = report(events, site_id=2, assignment_id=30, planter_id=11, species='bungalon')
    assert replacement['batches'][0]['planting_type'] == 'replacement'
    assert replacement['summary']['total'] == 1
    assert report(events, site_id=1)['replanting']['replacements_completed'] == 1
    assert report(events, species='Rhizophora')['summary']['total'] == 0


def test_reversed_future_and_unattributed_outcomes_never_create_survivors():
    events = [seedling(1), seedling(2, 1, planted_at='2026-06-01T08:00:00+08:00'),
              seedling(3, closure_reason='completion_reversed'),
              seedling(4, planted_at='2026-12-01T08:00:00+08:00'),
              seedling(5, closed_at='2026-06-01T08:00:00+08:00')]
    result = report(events)
    assert result['summary']['total'] == 3
    assert result['summary']['unknown'] == 2
    assert result['summary']['planted'] == 1
    assert result['summary']['survival_rate_pct'] is None
    assert result['replanting']['replacements_completed'] == 0


def test_removed_or_eroded_locations_keep_deaths_and_do_not_become_replacements():
    events = [seedling(1, recorded_death_at='2026-06-01T08:00:00+08:00'),
              seedling(2, recorded_death_at='2026-06-01T08:00:00+08:00')]
    result = report(events, points=[{'id': 1, 'deleted_at': '2026-07-01'},
                                    {'id': 2, 'point_status': 'planned', 'eroded_unavailable': True}])
    assert result['summary']['dead'] == 2
    assert result['replanting']['awaiting_replanting'] == 1
    assert result['replanting']['unavailable'] == 1
    missing = report([seedling(3)], points=[{'id': 3, 'deleted_at': '2026-07-01'}])
    assert missing['summary']['unknown'] == 1
    assert missing['summary']['survival_rate_pct'] is None


def test_legacy_replacements_and_philippine_dates_are_recognized():
    result = report([seedling(1, recorded_death_at='2026-06-01T08:00:00+08:00'),
                     seedling(2, 1, planted_at='2026-06-30T16:15:00Z')])
    replacement = next(row for row in result['batches'] if row['planting_type'] == 'replacement')
    assert replacement['planted_date'] == '2026-07-01'
    assert result['replanting']['replacements_completed'] == 1
    assert report([])['summary']['survival_rate_pct'] is None
