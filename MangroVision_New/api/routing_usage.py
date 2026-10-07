"""Persistent request limits for the laptop's Google Routes calls.

Only counts are stored; credentials, coordinates, and route content are not.
This limits this backend, not other applications on a Google billing account.
"""
from datetime import datetime, timezone
import hashlib
import os
from pathlib import Path
import sqlite3

from fastapi import HTTPException


DEFAULT_USAGE_PATH = Path(__file__).resolve().parents[1] / 'run_logs/google_routes_usage.sqlite3'


def _limit(name, default):
    try:
        value = int(os.getenv(name, str(default)))
        if value < 0:
            raise ValueError
        return value
    except ValueError as error:
        raise HTTPException(status_code=503, detail='Road navigation request limits are not configured correctly. Ask your administrator to check them.') from error


def reserve_google_request(api_key, *, usage_path=None, now=None):
    """Reserve a call atomically before contacting Google; never refund attempts.

    Limits reset by UTC calendar date/month and survive server restarts. A
    database failure stops calls, so an unavailable counter cannot bypass caps.
    """
    daily_limit = _limit('GOOGLE_ROUTES_DAILY_LIMIT', 100)
    monthly_limit = _limit('GOOGLE_ROUTES_MONTHLY_LIMIT', 1000)
    instant = (now or datetime.now(timezone.utc)).astimezone(timezone.utc)
    day, month = instant.strftime('%Y-%m-%d'), instant.strftime('%Y-%m')
    identity = hashlib.sha256(api_key.encode('utf-8')).hexdigest()
    path = Path(usage_path or DEFAULT_USAGE_PATH)
    connection = None
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        connection = sqlite3.connect(path, timeout=5, isolation_level=None)
        connection.execute('''CREATE TABLE IF NOT EXISTS usage (
            credential_id TEXT NOT NULL, period TEXT NOT NULL, count INTEGER NOT NULL,
            PRIMARY KEY (credential_id, period))''')
        connection.execute('BEGIN IMMEDIATE')
        counts = dict(connection.execute(
            'SELECT period, count FROM usage WHERE credential_id = ? AND period IN (?, ?)',
            (identity, day, month),
        ).fetchall())
        if counts.get(day, 0) >= daily_limit:
            raise HTTPException(status_code=429, detail='The daily navigation limit for user testing has been reached. Try after the next daily reset or ask your administrator to review the limit.')
        if counts.get(month, 0) >= monthly_limit:
            raise HTTPException(status_code=429, detail='The monthly navigation limit for user testing has been reached. Ask your administrator to review usage before continuing.')
        for period in (day, month):
            connection.execute('''INSERT INTO usage (credential_id, period, count) VALUES (?, ?, 1)
                ON CONFLICT (credential_id, period) DO UPDATE SET count = count + 1''',
                (identity, period))
        connection.execute('DELETE FROM usage WHERE period < ?', (month,))
        connection.commit()
    except (OSError, sqlite3.Error) as error:
        raise HTTPException(status_code=503, detail='Could not check the navigation request limit. Please retry; no route request was sent.') from error
    finally:
        if connection is not None:
            if connection.in_transaction:
                connection.rollback()
            connection.close()
