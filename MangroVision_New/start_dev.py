"""
MangroVision v2 — Development Server Launcher

Starts both:
  1. FastAPI backend on port 8000
  2. Vite React dev server on port 5173

Usage:
  python start_dev.py
"""

import subprocess
import sys
import os
from pathlib import Path

ROOT = Path(__file__).parent
CLIENT_DIR = ROOT / "client"
API_DIR = ROOT / "api"

# Add parent MangroVision directory to PYTHONPATH
PARENT_DIR = ROOT.parent
env = os.environ.copy()
existing_path = env.get("PYTHONPATH", "")
env["PYTHONPATH"] = f"{PARENT_DIR};{ROOT};{existing_path}" if existing_path else f"{PARENT_DIR};{ROOT}"


def main():
    print("=" * 60)
    print("  MangroVision v2 — Development Server")
    print("=" * 60)
    print()
    print(f"  API server:     http://localhost:8000")
    print(f"  API docs:       http://localhost:8000/api/docs")
    print(f"  Frontend:       http://localhost:5173")
    print(f"  Tile server:    http://localhost:8000/tiles  (served by FastAPI)")
    print()
    print("=" * 60)
    print()

    # Start FastAPI
    api_proc = subprocess.Popen(
        [
            sys.executable, "-m", "uvicorn",
            "api.main:app",
            "--host", "0.0.0.0",
            "--port", "8000",
            "--reload",
            "--reload-dir", str(API_DIR),
        ],
        cwd=str(ROOT),
        env=env,
    )

    # Start Vite
    vite_proc = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=str(CLIENT_DIR),
        shell=True,
    )

    try:
        api_proc.wait()
    except KeyboardInterrupt:
        print("\nShutting down...")
        api_proc.terminate()
        vite_proc.terminate()
        api_proc.wait()
        vite_proc.wait()


if __name__ == "__main__":
    main()
