"""
Compatibility entry point for older launch commands.

Some local shortcuts still invoke `server/start_tile_server.py`, while the
active tile server lives at the repository root as `start_tile_server.py`.
This wrapper forwards execution to the current script so both paths work.
"""

from pathlib import Path
import runpy


ROOT_SCRIPT = Path(__file__).resolve().parents[1] / "start_tile_server.py"


if __name__ == "__main__":
    runpy.run_path(str(ROOT_SCRIPT), run_name="__main__")
