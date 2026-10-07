"""Verify request-independent execution, isolation, capacity and error recovery."""
import sys
import asyncio
import unittest
from unittest.mock import patch
from contextvars import ContextVar
from pathlib import Path
from threading import Event
from time import monotonic, sleep
from uuid import uuid4

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from api.processing_jobs import ProcessingAlreadyStarted, ProcessingBusy, ProcessingJobs


def completed(manager, job_id, owner):
    deadline = monotonic() + 3
    while monotonic() < deadline:
        job = manager.get(job_id, owner)
        if job and job['status'] != 'running':
            return job
        sleep(0.005)
    raise AssertionError('Worker did not finish')


class JobTests(unittest.TestCase):
    def test_independent_completion_ownership_and_context(self):
        manager = ProcessingJobs()
        context = ContextVar('test_session')
        token = context.set('authenticated request')
        started, finish = Event(), Event()
        def work(progress):
            progress({'stage': 'Detecting', 'pct': 40})
            started.set()
            finish.wait(3)
            return {'context': context.get()}
        try:
            job_id = manager.submit(7, work)
        finally:
            context.reset(token)
        try:
            self.assertTrue(started.wait(3))
            self.assertIsNone(manager.get(job_id, 8))
            self.assertEqual(manager.get(job_id, 7)['pct'], 40)
            self.assertNotIn('payload', manager.get(job_id, 7))
            with self.assertRaises(ProcessingBusy):
                manager.submit(8, lambda progress: {})
        finally:
            finish.set()
        job = completed(manager, job_id, 7)
        self.assertEqual(job['payload']['context'], 'authenticated request')

    def test_failure_releases_capacity(self):
        manager = ProcessingJobs()
        def fail(progress):
            raise ValueError('invalid map bounds')
        failed = completed(manager, manager.submit(7, fail), 7)
        self.assertEqual(failed['status'], 'failed')
        self.assertEqual(failed['detail'], 'invalid map bounds')
        next_job = manager.submit(7, lambda progress: {'ok': True})
        self.assertTrue(completed(manager, next_job, 7)['payload']['ok'])

    def test_result_retention_is_bounded(self):
        manager = ProcessingJobs(max_results=1)
        first = manager.submit(7, lambda progress: {})
        completed(manager, first, 7)
        second = manager.submit(7, lambda progress: {})
        completed(manager, second, 7)
        self.assertIsNone(manager.get(first, 7))
        self.assertIsNotNone(manager.get(second, 7))
        manager._retention = 0
        self.assertIsNone(manager.get(second, 7))

    def test_retry_request_is_bound_to_its_owner_and_never_runs_again(self):
        manager = ProcessingJobs()
        request_id = str(uuid4())
        first = manager.submit(7, lambda progress: {'ok': True}, request_id=request_id)
        completed(manager, first, 7)
        self.assertEqual(manager.find_request(7, request_id), first)
        self.assertIsNone(manager.find_request(8, request_id))
        with self.assertRaises(ProcessingAlreadyStarted) as error:
            manager.submit(7, lambda progress: self.fail('A duplicate job ran'), request_id=request_id)
        self.assertEqual(error.exception.job_id, first)
        manager._retention = 0
        self.assertIsNone(manager.find_request(7, request_id))


class JobEndpointTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from api.routes import processing
        cls.processing = processing
        app = FastAPI()
        app.include_router(processing.router, prefix='/api/analyses')
        @app.get('/live')
        def live():
            return {'status': 'ok'}
        cls.app = app
        cls.client = TestClient(app)

    def test_upload_returns_before_pipeline_finishes_and_result_is_private(self):
        p = self.processing
        manager = ProcessingJobs()
        started, finish = Event(), Event()
        paths, repeat_approvals = [], []
        request_id = str(uuid4())
        def pipeline(**parameters):
            paths.append(parameters['temp_path'])
            repeat_approvals.append(parameters['allow_repeat_image_analysis'])
            parameters['progress_cb']({'stage': 'Detecting', 'pct': 25})
            started.set()
            finish.wait(5)
            return {'analysis_key': 'test-only', 'map': {'coordinates': []}}
        with patch.object(p, '_PROCESSING_JOBS', manager), patch.object(p, '_require_lgu_user', return_value={'id': 7}), \
             patch.object(p, '_execute_canopy_workflow', side_effect=pipeline), patch.object(p, '_record_processed_image') as activity:
            try:
                created = self.client.post('/api/analyses/jobs', data={'request_id': request_id, 'allow_repeat_image_analysis':'true'}, files={'image': ('../../outside.jpg', b'test', 'image/jpeg')})
                self.assertEqual(created.status_code, 202)
                job_id = created.json()['job_id']
                self.assertTrue(started.wait(3))
                self.assertEqual(repeat_approvals,[True])
                status = self.client.get(f'/api/analyses/jobs/{job_id}')
                self.assertEqual(status.headers['cache-control'], 'no-store')
                self.assertEqual(status.json()['pct'], 25)
                self.assertEqual(paths[0].parent, p._TEMP_UPLOADS_DIR)
                # The first acknowledgement could be lost at a gateway. A
                # repeat upload must recover that job, including while running.
                retried = self.client.post('/api/analyses/jobs', data={'request_id': request_id}, files={'image': ('same.jpg', b'test', 'image/jpeg')})
                self.assertEqual(retried.status_code, 202)
                self.assertEqual(retried.json()['job_id'], job_id)
                self.assertEqual(len(paths), 1)
                busy = self.client.post('/api/analyses/jobs', files={'image': ('second.jpg', b'test', 'image/jpeg')})
                self.assertEqual(busy.status_code, 409)
                with patch.object(p, '_require_lgu_user', return_value={'id': 8}):
                    self.assertEqual(self.client.get(f'/api/analyses/jobs/{job_id}').status_code, 404)
            finally:
                finish.set()
            completed(manager, job_id, 7)
            result = self.client.get(f'/api/analyses/jobs/{job_id}')
            self.assertEqual(result.json()['status'], 'succeeded')
            self.assertEqual(result.json()['payload']['analysis_key'], 'test-only')
            self.assertFalse(paths[0].exists())
            activity.assert_called_once()

    def test_waiting_for_database_auth_does_not_block_other_http_requests(self):
        from httpx import ASGITransport, AsyncClient
        started, finish, returned = Event(), Event(), Event()
        def blocking_auth():
            started.set()
            finish.wait(5)
            returned.set()
            return {'id': 7}
        async def scenario():
            async with AsyncClient(transport=ASGITransport(app=self.app), base_url='http://testserver') as client:
                upload = asyncio.create_task(client.post('/api/analyses/jobs', files={'image': ('test.jpg', b'test')}))
                try:
                    while not started.is_set():
                        await asyncio.sleep(0.01)
                    self.assertEqual((await client.get('/live')).status_code, 200)
                    self.assertFalse(returned.is_set(), 'Database wait blocked the HTTP event loop')
                finally:
                    finish.set()
                    await upload
        with patch.object(self.processing, '_require_lgu_user', side_effect=blocking_auth), \
             patch.object(self.processing, '_PROCESSING_JOBS', ProcessingJobs()), \
             patch.object(self.processing, '_run_image_workflow', return_value={}):
            asyncio.run(scenario())

    def test_no_session_cannot_start_or_read_job(self):
        with patch.object(self.processing, 'get_user_by_session_token', return_value=None):
            self.assertEqual(self.client.get('/api/analyses/jobs/unknown').status_code, 401)
            response = self.client.post('/api/analyses/jobs', files={'image': ('test.jpg', b'test', 'image/jpeg')})
            self.assertEqual(response.status_code, 401)

    def test_busy_preflight_returns_without_waiting_for_a_long_analysis(self):
        from fastapi import HTTPException
        p = self.processing
        p._WORKFLOW_LOCK.acquire()
        try:
            with patch.object(p, '_calibrate_preflight_footprint') as calibrate:
                with self.assertRaises(HTTPException) as error:
                    p._run_preflight(Path('test.jpg'), {})
                self.assertEqual(error.exception.status_code, 409)
                calibrate.assert_not_called()
        finally:
            p._WORKFLOW_LOCK.release()


if __name__ == '__main__':
    unittest.main()
