"""Optional worker for hosts that run appointment email delivery separately."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from mangrovision_db.appointment_email import run_delivery_cycle

if __name__ == '__main__':
    outcome = run_delivery_cycle()
    print(json.dumps(outcome))
    raise SystemExit(0 if outcome['configured'] and not outcome['failed'] else 2)
