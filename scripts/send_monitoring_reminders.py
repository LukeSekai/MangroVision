"""Run daily (Asia/Manila) on the backend host to email monitoring reminders."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from mangrovision_db.notifications import (
    DEFAULT_RECIPIENT,
    _send_smtp,
    send_pending_reminders,
    sync_monitoring_reminders,
)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Send LGU monitoring email reminders.")
    parser.add_argument(
        "--test-email", action="store_true",
        help="Send one configuration test to the LGU inbox without changing reminder records.",
    )
    args = parser.parse_args(argv)
    if args.test_email:
        recipient = os.getenv("LGU_REMINDER_EMAIL", DEFAULT_RECIPIENT).strip()
        if not recipient:
            parser.error("LGU_REMINDER_EMAIL must be configured for a test email.")
        _send_smtp(
            recipient,
            "MangroVision email setup test",
            "This test confirms that MangroVision can email the LGU inbox. "
            "It is not a monitoring reminder.",
        )
        print(json.dumps({"test_email_sent_to": recipient}, sort_keys=True))
        return 0
    sync_monitoring_reminders()
    outcome = send_pending_reminders()
    print(json.dumps(outcome, sort_keys=True))
    return 0 if outcome["configured"] and outcome["failed"] == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
