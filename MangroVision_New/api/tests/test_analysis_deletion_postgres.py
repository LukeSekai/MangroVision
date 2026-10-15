"""Real lock races in a disposable schema; never target application tables.

Opt in with MANGROVISION_RUN_ANALYSIS_DELETE_PG_TESTS=1. The configured migration
connection only creates/drops a UUID-named fixture schema. A separate test URL
can be supplied as MANGROVISION_ANALYSIS_DELETE_TEST_DATABASE_URL.
"""
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from queue import Queue
from time import monotonic, sleep
from types import SimpleNamespace
from uuid import uuid4

import pytest
from sqlalchemy import create_engine, text
from sqlalchemy.exc import IntegrityError

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
import planting_database as database
from mangrovision_db.compat import CompatConnection
from mangrovision_db.config import get_settings

pytestmark = pytest.mark.skipif(
    os.getenv('MANGROVISION_RUN_ANALYSIS_DELETE_PG_TESTS') != '1',
    reason='Disposable PostgreSQL lock tests are opt-in.',
)


@pytest.fixture
def workspace(monkeypatch):
    settings = get_settings()
    url = os.getenv('MANGROVISION_ANALYSIS_DELETE_TEST_DATABASE_URL') or settings.migration_database_url
    args = {'connect_timeout':10, 'options':'-csearch_path=pg_catalog -cstatement_timeout=12000 -clock_timeout=8000'}
    if settings.db_sslmode:
        args['sslmode'] = settings.db_sslmode
    engine = create_engine(url, isolation_level='READ COMMITTED', pool_size=4, max_overflow=0, connect_args=args)
    schema = 'mv_analysis_delete_test_' + uuid4().hex
    assert re.fullmatch(r'mv_analysis_delete_test_[a-f0-9]{32}', schema)
    created = False

    def connect():
        try:
            connection = engine.connect()
        except Exception as error:
            raise RuntimeError(f'Fixture connection unavailable ({type(error).__name__}); details suppressed.') from None
        try:
            connection.exec_driver_sql(f'SET LOCAL search_path TO "{schema}", pg_catalog')
            assert connection.scalar(text('SELECT current_schema()')) == schema
            return connection
        except BaseException:
            connection.close()
            raise

    try:
        try:
            setup = engine.connect()
        except Exception as error:
            pytest.fail(f'Fixture connection unavailable ({type(error).__name__}); details suppressed.', pytrace=False)
        with setup:
            setup.exec_driver_sql(f'CREATE SCHEMA "{schema}"')
            setup.commit()
            created = True
        with connect() as connection:
            connection.exec_driver_sql('''
                CREATE TABLE analyses(id BIGINT PRIMARY KEY);
                CREATE TABLE planting_points(id BIGINT PRIMARY KEY,
                    analysis_id BIGINT NOT NULL REFERENCES analyses(id) ON DELETE CASCADE,
                    status TEXT DEFAULT 'planned', planted_at TIMESTAMPTZ, planted_date DATE,
                    death_at TIMESTAMPTZ, death_reason TEXT, deleted_at TIMESTAMPTZ);
                CREATE TABLE planter_assignment_points(id BIGINT PRIMARY KEY,
                    planting_point_id BIGINT REFERENCES planting_points(id) ON DELETE CASCADE);
                CREATE TABLE planting_events(id BIGINT PRIMARY KEY,
                    planting_point_id BIGINT REFERENCES planting_points(id) ON DELETE SET NULL);
                CREATE TABLE point_death_records(id BIGINT PRIMARY KEY,
                    planting_point_id BIGINT REFERENCES planting_points(id) ON DELETE CASCADE);
                CREATE TABLE analysis_assets(id BIGINT PRIMARY KEY,
                    analysis_id BIGINT REFERENCES analyses(id) ON DELETE CASCADE, object_key TEXT);
                CREATE TABLE object_cleanup_jobs(id BIGINT GENERATED ALWAYS AS IDENTITY,
                    object_key TEXT UNIQUE, reason TEXT);
                INSERT INTO analyses VALUES (1);
                INSERT INTO planting_points(id,analysis_id) VALUES (1,1);
                INSERT INTO analysis_assets VALUES (1,1,'fixture-original');
            ''')
            for table in ('analyses','planting_points','planter_assignment_points','planting_events',
                          'point_death_records','analysis_assets','object_cleanup_jobs'):
                namespace = connection.scalar(text('''SELECT n.nspname FROM pg_class c
                    JOIN pg_namespace n ON n.oid=c.relnamespace WHERE c.oid=to_regclass(:name)'''), {'name':table})
                assert namespace == schema  # Every possible write resolves ONLY to fixture objects.
            connection.commit()
        monkeypatch.setattr(database, '_get_connection', lambda: CompatConnection(connect()))
        monkeypatch.setattr(database, 'process_object_cleanup_jobs', lambda: pytest.fail('Live storage cleanup must never run.'))
        yield SimpleNamespace(connect=connect)
    finally:
        if created:
            with engine.begin() as cleanup:
                cleanup.exec_driver_sql(f'DROP SCHEMA "{schema}" CASCADE')
        engine.dispose()


def wait_until_blocked(workspace, pid):
    """Observe PostgreSQL actually waiting for a lock; timing alone is insufficient."""
    deadline = monotonic() + 5
    with workspace.connect() as observer:
        while monotonic() < deadline:
            observer.execute(text('SELECT pg_stat_clear_snapshot()'))
            state = observer.scalar(text('SELECT wait_event_type FROM pg_stat_activity WHERE pid=:pid'), {'pid':pid})
            if state == 'Lock':
                return
            sleep(0.03)
    pytest.fail('The competing transaction never reached a PostgreSQL lock wait.')


def delete(workspace, started):
    with workspace.connect() as connection:
        started.put(connection.scalar(text('SELECT pg_backend_pid()')))
        database._delete_analysis_rows(CompatConnection(connection), 1)
        connection.commit()


def assert_preserved(workspace):
    with workspace.connect() as connection:
        assert connection.scalar(text('SELECT COUNT(*) FROM analyses')) == 1
        assert connection.scalar(text('SELECT COUNT(*) FROM planting_points')) == 1
        assert connection.scalar(text('SELECT COUNT(*) FROM analysis_assets')) == 1
        assert connection.scalar(text('SELECT COUNT(*) FROM object_cleanup_jobs')) == 0


@pytest.mark.parametrize('change', [
    "UPDATE planting_points SET status='planted', planted_at=CURRENT_TIMESTAMP WHERE id=1",
    "UPDATE planting_points SET death_at=CURRENT_TIMESTAMP WHERE id=1",
    'INSERT INTO planter_assignment_points VALUES (1,1)',
    'INSERT INTO planting_events VALUES (1,1)',
    'INSERT INTO point_death_records VALUES (1,1)',
])
def test_field_write_wins_delete_waits_then_preserves_the_committed_evidence(workspace, change):
    writer = workspace.connect()
    executor = ThreadPoolExecutor(max_workers=1)
    started = Queue()
    try:
        writer.execute(text(change))
        future = executor.submit(delete, workspace, started)
        wait_until_blocked(workspace, started.get(timeout=5))
        assert not future.done()
        writer.commit()
        with pytest.raises(ValueError, match='cannot be deleted'):
            future.result(timeout=10)
    finally:
        writer.rollback()
        writer.close()
        executor.shutdown(wait=True, cancel_futures=True)
    assert_preserved(workspace)


@pytest.mark.parametrize('change', [
    'INSERT INTO planter_assignment_points VALUES (1,1)',
    'INSERT INTO planting_points(id,analysis_id) VALUES (2,1)',
])
def test_unused_delete_wins_competing_assignment_or_new_point_cannot_commit(workspace, change):
    deleter = workspace.connect()
    executor = ThreadPoolExecutor(max_workers=1)
    started = Queue()
    def write():
        with workspace.connect() as connection:
            started.put(connection.scalar(text('SELECT pg_backend_pid()')))
            connection.execute(text(change))
            connection.commit()
    try:
        database._delete_analysis_rows(CompatConnection(deleter), 1)
        future = executor.submit(write)
        wait_until_blocked(workspace, started.get(timeout=5))
        deleter.commit()
        with pytest.raises(IntegrityError):
            future.result(timeout=10)
    finally:
        deleter.rollback()
        deleter.close()
        executor.shutdown(wait=True, cancel_futures=True)
    with workspace.connect() as connection:
        assert connection.scalar(text('SELECT COUNT(*) FROM planting_points')) == 0
        assert connection.scalar(text('SELECT COUNT(*) FROM planter_assignment_points')) == 0
        assert connection.scalar(text('SELECT COUNT(*) FROM object_cleanup_jobs')) == 1
