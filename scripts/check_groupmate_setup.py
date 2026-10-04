"""Check the files and tools needed for the current FastAPI/React workspace.

Run from any directory with the Python environment used for MangroVision:
    python scripts/check_groupmate_setup.py
    python scripts/check_groupmate_setup.py --online  # after starting the API

This prints no database, storage, or API credentials.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
from pathlib import Path
import shutil
import sys
from urllib.error import HTTPError, URLError
from urllib.request import urlopen


ROOT = Path(__file__).resolve().parents[1]
ASSETS = {
    "MAP/FINAL/final_orthophoto.tif": (
        "c0c7a30fd8df59d5d7d4eef6e79f1b9e93c32d59116310410480286df8d90a98"
    ),
    "MangroVision_New/try_model/model_final.pth": (
        "603cd28809241324ba645997aea0cc5744ebda541858c73a4393df706f06e89b"
    ),
}
SHIPPED_FILES = (
    "MangroVision_New/try_model/model_metadata.json",
    "MangroVision_New/api/data/site_access_routes.json",
    "MangroVision_New/api/assets/fonts/Inter-Regular.ttf",
    "MangroVision_New/api/assets/fonts/Inter-Bold.ttf",
    "MangroVision_New/client/public/favicon.png",
    "MangroVision_New/client/public/logo-icon.png",
    "MangroVision_New/client/public/logo-icon-small.png",
    "MangroVision_New/client/public/logo-lockup.png",
    "MangroVision_New/client/public/mangrove.jpg",
)
PYTHON_MODULES = (
    "fastapi", "psycopg", "boto3", "cv2", "rasterio", "torch",
    "detectree2", "detectron2", "reportlab",
)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for block in iter(lambda: source.read(4 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def check_online() -> bool:
    try:
        with urlopen("http://127.0.0.1:8000/api/health/ready", timeout=10) as response:
            payload = json.load(response)
            ready = response.status == 200 and payload.get("status") == "ready"
            print(f"{'OK' if ready else 'MISSING'} API/database/storage readiness")
            return ready
    except (HTTPError, URLError, OSError, ValueError) as error:
        print(f"MISSING API/database/storage readiness ({type(error).__name__})")
        return False


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--online", action="store_true", help="Check the running API, database, and storage")
    args = parser.parse_args()
    missing = False

    for relative, expected_hash in ASSETS.items():
        path = ROOT / relative
        if not path.is_file():
            print(f"MISSING {relative} (copy separately; ignored by Git)")
            missing = True
        elif sha256(path) != expected_hash:
            print(f"DIFFERENT {relative} (not the reference asset used for this setup)")
            missing = True
        else:
            print(f"OK {relative}")

    for relative in SHIPPED_FILES:
        present = (ROOT / relative).is_file()
        print(f"{'OK' if present else 'MISSING'} {relative}")
        missing |= not present

    env_present = (ROOT / ".env").is_file()
    print(f"{'OK' if env_present else 'MISSING'} .env (private database/storage settings)")
    missing |= not env_present

    for module in PYTHON_MODULES:
        present = importlib.util.find_spec(module) is not None
        print(f"{'OK' if present else 'MISSING'} Python module {module}")
        missing |= not present

    node_present = shutil.which("node") is not None
    vite_present = (ROOT / "MangroVision_New/client/node_modules/vite/bin/vite.js").is_file()
    print(f"{'OK' if node_present else 'MISSING'} Node.js")
    print(f"{'OK' if vite_present else 'MISSING'} frontend dependencies (npm ci)")
    missing |= not node_present or not vite_present

    if args.online:
        missing |= not check_online()

    print("Setup ready" if not missing else "Setup needs the items marked above")
    return 1 if missing else 0


if __name__ == "__main__":
    sys.exit(main())
