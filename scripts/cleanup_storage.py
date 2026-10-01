"""Run retryable object cleanup and remove abandoned processing previews."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from planting_database import (
    cleanup_abandoned_analysis_previews,
    process_object_cleanup_jobs,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--preview-age-hours", type=int, default=24)
    parser.add_argument("--job-limit", type=int, default=100)
    args = parser.parse_args()
    result = {
        "database_cleanup_jobs": process_object_cleanup_jobs(args.job_limit),
        "abandoned_previews": cleanup_abandoned_analysis_previews(args.preview_age_hours),
    }
    print(json.dumps(result, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
