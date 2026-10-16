"""LIKE hours and server tide checks agree with the staff graph calculation."""
import json
import shutil
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'MangroVision_New'))
from api.tide_assessment import assess_graph_window
from mangrovision_db.like_appointments import requested_appointment_window

BASE = datetime(2030, 1, 2, tzinfo=timezone(timedelta(hours=8)))


def forecast(heights=None):
    return {'available': True, 'stale': False, 'fetched_at': BASE.timestamp(),
            'datum': 'MSL', 'datum_reference': 'test', 'series_available': True,
            'heights': [{'timestamp': BASE.timestamp() + hour * 3600, 'height_m': height}
                        for hour, height in enumerate(heights if heights is not None else [0.2] * 73)]}


def assess(payload, hour, duration=2):
    return assess_graph_window(payload, BASE + timedelta(hours=hour),
                             BASE + timedelta(hours=hour + duration), now=BASE.timestamp())


def test_whole_visit_checks_middle_peak_not_just_start_and_end():
    heights = [0] * 73
    heights[11] = 1
    payload = forecast(heights)
    assert assess(payload, 6)['status'] == 'safe'
    assert assess(payload, 10)['status'] == 'unsafe'
    assert assess(payload, 12)['status'] == 'safe'


def test_stale_missing_and_out_of_coverage_data_stays_unknown():
    for changes in ({'available': False}, {'stale': True}, {'datum': 'unknown'},
                    {'fetched_at': BASE.timestamp() - 6 * 3600}, {'series_available': False}):
        assert assess({**forecast(), **changes}, 8)['status'] == 'unknown'
    assert assess(forecast(), 75)['status'] == 'unknown'
    for heights in ([None] + [0] * 72, [0], []):
        assert assess(forecast(heights), 8)['status'] == 'unknown'
    payload = forecast()
    del payload['heights'][9]
    assert assess(payload, 8)['status'] == 'unknown'


@pytest.mark.parametrize('hour,minute', [(4, 30), (5, 59), (15, 1), (16, 0), (17, 0)])
def test_service_rejects_planting_outside_hours_even_without_the_public_api(hour, minute):
    with pytest.raises(ValueError, match='6 AM to 5 PM'):
        requested_appointment_window(BASE.replace(hour=hour, minute=minute), appointment_type='tree_planting')


@pytest.mark.parametrize('hour', [6, 15])
def test_service_hours_are_in_philippine_time(hour):
    start = BASE.replace(hour=hour).astimezone(timezone.utc)
    local_start, end = requested_appointment_window(start, appointment_type='tree_planting')
    assert local_start.hour == hour
    assert end.hour == hour + 2


def test_server_assessment_matches_javascript_graph_reference_and_interpolation():
    node = shutil.which('node')
    if not node:
        pytest.skip('Node is needed for comparison with the browser calculation.')
    patterns = [[0.2] * 73, [0, 0.5, 1, 0.5] * 18 + [0],
                [1 if i == 11 else 0 for i in range(73)]]
    cases = [{'payload': forecast(pattern), 'startAt': int((BASE.timestamp() + hour * 3600) * 1000),
              'endAt': int((BASE.timestamp() + (hour + 2) * 3600) * 1000), 'now': int(BASE.timestamp() * 1000)}
             for pattern in patterns for hour in [4.5, 6, 8.25, 10, 12, 15, 15.25]]
    module = (ROOT / 'MangroVision_New/client/src/utils/plantingTides.js').as_uri()
    script = f'import {{ assessGraphWindow }} from {json.dumps(module)};\n'
    script += f'console.log(JSON.stringify({json.dumps(cases)}.map(assessGraphWindow)));'
    result = subprocess.run([node, '--input-type=module', '-'], input=script, text=True,
                            capture_output=True, check=True)
    for case, browser in zip(cases, json.loads(result.stdout), strict=True):
        server = assess_graph_window(case['payload'],
            datetime.fromtimestamp(case['startAt'] / 1000, timezone.utc),
            datetime.fromtimestamp(case['endAt'] / 1000, timezone.utc), now=BASE.timestamp())
        assert server['status'] == browser['status']
        assert server['threshold_m'] == pytest.approx(browser['threshold_m'])
        assert server['maximum_m'] == pytest.approx(browser['maximum_m'])
