"""Shared runtime state for API-wide operation locks."""

from contextlib import contextmanager
from threading import Lock

_state_lock = Lock()
_active_processing_jobs = 0


@contextmanager
def processing_job():
    """Mark an image-processing job as active for mutation guards."""
    global _active_processing_jobs
    with _state_lock:
        _active_processing_jobs += 1
    try:
        yield
    finally:
        with _state_lock:
            _active_processing_jobs = max(0, _active_processing_jobs - 1)


def is_processing_active() -> bool:
    """Return True while any backend image-processing job is running."""
    with _state_lock:
        return _active_processing_jobs > 0
