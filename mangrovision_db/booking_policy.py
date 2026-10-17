"""Shared, reversible booking and planter-access limits for testing."""

import json
from pathlib import Path


BOOKING_POLICY = json.loads((Path(__file__).resolve().parents[1]
    / 'MangroVision_New/client/src/like/bookingPolicy.json').read_text(encoding='utf-8'))


def activity_access_timing_message():
    if BOOKING_POLICY['enforceActivityAccessWindow']:
        return 'Access opens at the scheduled start and ends at the scheduled finish. No password or PIN is needed.'
    return 'This code can be used before and after the scheduled planting time. No password or PIN is needed.'
