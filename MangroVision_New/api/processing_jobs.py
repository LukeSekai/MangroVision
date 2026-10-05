"""Short HTTP requests around a single laptop's long image analysis.

Jobs and unsaved previews are deliberately local to one API process. Start the
testing server with one worker and without reload; saved records remain in DB.
"""

from contextvars import copy_context
from threading import Lock, Thread
from time import monotonic
from uuid import uuid4


class ProcessingBusy(Exception):
    pass


class ProcessingAlreadyStarted(Exception):
    def __init__(self, job_id):
        self.job_id = job_id


class ProcessingJobs:
    def __init__(self, *, retention_seconds=3600, max_results=24):
        self._lock = Lock()
        self._jobs = {}
        self._active = None
        self._retention = retention_seconds
        self._max_results = max_results

    def _prune(self):
        now = monotonic()
        completed = [key for key, job in self._jobs.items() if job['finished'] is not None]
        for key in completed:
            if now - self._jobs[key]['finished'] >= self._retention:
                self._jobs.pop(key)
        completed = [key for key in completed if key in self._jobs]
        for key in completed[:max(0, len(completed) - self._max_results)]:
            self._jobs.pop(key)

    def find_request(self, owner_id, request_id):
        if not request_id:
            return None
        with self._lock:
            self._prune()
            return self._find_request(owner_id, request_id)

    def _find_request(self, owner_id, request_id):
        return next((key for key, job in self._jobs.items()
                     if job['owner_id'] == int(owner_id) and job.get('request_id') == request_id), None)

    def submit(self, owner_id, work, request_id=None):
        """Reserve capacity before starting work; retain request context in thread."""
        context = copy_context()
        job_id = uuid4().hex
        with self._lock:
            self._prune()
            existing = self._find_request(owner_id, request_id) if request_id else None
            if existing:
                raise ProcessingAlreadyStarted(existing)
            if self._active is not None:
                raise ProcessingBusy('Another image is being analyzed. Please wait for it to finish.')
            self._active = job_id
            self._jobs[job_id] = {
                'owner_id': int(owner_id), 'request_id': request_id, 'finished': None,
                'public': {'job_id': job_id, 'status': 'running', 'stage': 'Starting analysis...', 'pct': 1},
            }
        thread = Thread(target=lambda: context.run(self._run, job_id, work), daemon=True)
        try:
            thread.start()
        except Exception:
            with self._lock:
                self._jobs.pop(job_id)
                self._active = None
            raise
        return job_id

    def _run(self, job_id, work):
        def progress(event):
            with self._lock:
                public = self._jobs[job_id]['public']
                if isinstance(event.get('stage'), str):
                    public['stage'] = event['stage']
                if isinstance(event.get('pct'), (int, float)):
                    public['pct'] = max(0, min(100, event['pct']))

        try:
            payload = work(progress)
            terminal = {'status': 'succeeded', 'stage': 'Complete!', 'pct': 100, 'payload': payload}
        except Exception as error:
            terminal = {
                'status': 'failed', 'detail': str(getattr(error, 'detail', error)) or 'Processing failed.',
                'status_code': getattr(error, 'status_code', 500),
            }
        with self._lock:
            self._jobs[job_id]['public'].update(terminal)
            self._jobs[job_id]['finished'] = monotonic()
            self._active = None
            self._prune()

    def get(self, job_id, owner_id):
        with self._lock:
            self._prune()
            job = self._jobs.get(job_id)
            if job is None or job['owner_id'] != int(owner_id):
                return None
            return dict(job['public'])
