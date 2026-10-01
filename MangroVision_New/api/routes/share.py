"""Share-link helpers for temporary field-app access."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
import threading
import time
from pathlib import Path

from fastapi import APIRouter, HTTPException
from planting_database import get_user_by_session_token

router = APIRouter()

_URL_PATTERN = re.compile(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com")
_TARGET_URL = os.getenv("MANGROVISION_SHARE_TARGET_URL", "http://127.0.0.1:5173").rstrip("/")
_FIELD_PATH = os.getenv("MANGROVISION_FIELD_PATH", "/field")

_lock = threading.Lock()
_tunnel_proc: subprocess.Popen[str] | None = None
_tunnel_url: str | None = None
_started_at: float | None = None
_log_lines: list[str] = []


def _require_planner() -> None:
    if not get_user_by_session_token(""):
        raise HTTPException(status_code=401, detail="Invalid or expired planner session.")


def _field_url(base_url: str | None) -> str:
    if not base_url:
        return ""
    return f"{base_url.rstrip('/')}/{_FIELD_PATH.lstrip('/')}"


def _append_log(line: str) -> None:
    global _tunnel_url
    clean = line.strip()
    if not clean:
        return
    with _lock:
        _log_lines.append(clean)
        del _log_lines[:-60]
        match = _URL_PATTERN.search(clean)
        if match:
            _tunnel_url = match.group(0)


def _recent_log() -> str:
    with _lock:
        return " ".join(_log_lines[-8:])


def _is_running(proc: subprocess.Popen[str] | None) -> bool:
    return proc is not None and proc.poll() is None


def _find_cloudflared() -> str:
    candidates: list[str] = []
    env_path = os.getenv("CLOUDFLARED_BIN", "").strip().strip('"')
    if env_path:
        candidates.append(env_path)

    found = shutil.which("cloudflared")
    if found:
        candidates.append(found)

    for env_key in ("LOCALAPPDATA", "PROGRAMFILES", "PROGRAMFILES(X86)"):
        root = os.getenv(env_key)
        if root:
            candidates.append(str(Path(root) / "cloudflared" / "cloudflared.exe"))
            candidates.append(str(Path(root) / "Cloudflare" / "cloudflared.exe"))

    seen: set[str] = set()
    for candidate in candidates:
        if not candidate or candidate in seen:
            continue
        seen.add(candidate)
        resolved = shutil.which(candidate) if os.path.basename(candidate) == candidate else None
        path = resolved or candidate
        if Path(path).is_file():
            return path

    raise HTTPException(
        status_code=503,
        detail=(
            "cloudflared was not found. Install the Cloudflare Tunnel CLI, "
            "or set CLOUDFLARED_BIN to the cloudflared executable path."
        ),
    )


def _reader_thread(proc: subprocess.Popen[str]) -> None:
    stream = proc.stdout
    if stream is None:
        return
    try:
        for line in iter(stream.readline, ""):
            _append_log(line)
    finally:
        try:
            stream.close()
        except Exception:
            pass


def _payload(status: str = "idle") -> dict:
    with _lock:
        active = _is_running(_tunnel_proc) and bool(_tunnel_url)
        cloudflare_url = _tunnel_url if active else ""
        started_at = _started_at if _is_running(_tunnel_proc) else None
    return {
        "status": "active" if active else status,
        "active": active,
        "cloudflare_url": cloudflare_url,
        "field_url": _field_url(cloudflare_url),
        "local_field_url": _field_url(_TARGET_URL),
        "target_url": _TARGET_URL,
        "started_at": started_at,
    }


@router.get("/field-link")
def get_field_link_status():
    """Return the current temporary field link, if a tunnel is active."""
    _require_planner()
    return _payload()


@router.post("/field-link/cloudflare")
def start_cloudflare_field_link():
    """Start or reuse a Cloudflare quick tunnel for the React field app."""
    global _started_at, _tunnel_proc, _tunnel_url

    _require_planner()

    reuse_existing = False
    with _lock:
        if _is_running(_tunnel_proc) and _tunnel_url:
            reuse_existing = True

        elif _tunnel_proc is not None and not _is_running(_tunnel_proc):
            _tunnel_proc = None
            _tunnel_url = None
            _started_at = None
            _log_lines.clear()

    if reuse_existing:
        return _payload("active")

    cloudflared = _find_cloudflared()
    command = [
        cloudflared,
        "tunnel",
        "--url",
        _TARGET_URL,
    ]
    creationflags = subprocess.CREATE_NO_WINDOW if os.name == "nt" else 0

    try:
        proc = subprocess.Popen(
            command,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
            cwd=str(Path(__file__).resolve().parents[2]),
            creationflags=creationflags,
        )
    except OSError as error:
        raise HTTPException(status_code=500, detail=f"Could not start cloudflared: {error}") from error

    with _lock:
        _tunnel_proc = proc
        _tunnel_url = None
        _started_at = time.time()
        _log_lines.clear()

    threading.Thread(target=_reader_thread, args=(proc,), daemon=True).start()

    deadline = time.time() + 25
    while time.time() < deadline:
        with _lock:
            url = _tunnel_url
            current_proc = _tunnel_proc
        if url:
            return _payload("active")
        if current_proc is not None and current_proc.poll() is not None:
            detail = "cloudflared stopped before it returned a link."
            log = _recent_log()
            if log:
                detail = f"{detail} {log}"
            raise HTTPException(status_code=502, detail=detail)
        time.sleep(0.25)

    detail = "cloudflared started, but no public link appeared yet. Try Check Status in a few seconds."
    log = _recent_log()
    if log:
        detail = f"{detail} {log}"
    raise HTTPException(status_code=504, detail=detail)


@router.post("/field-link/stop")
def stop_cloudflare_field_link():
    """Stop the managed quick tunnel."""
    global _started_at, _tunnel_proc, _tunnel_url

    _require_planner()

    with _lock:
        proc = _tunnel_proc
        _tunnel_proc = None
        _tunnel_url = None
        _started_at = None
        _log_lines.clear()

    if _is_running(proc):
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()

    return _payload("stopped")
