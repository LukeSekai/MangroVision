"""SQLAlchemy-backed compatibility boundary for the existing domain services.

The legacy service functions use DB-API-like ``execute``/``fetch`` calls.  This
adapter keeps those stable while routing every supported runtime operation to
PostgreSQL through SQLAlchemy and psycopg.  Schema creation is deliberately not
performed here; Alembic is the sole schema authority.
"""

from __future__ import annotations

import json
import re
from collections.abc import Iterator, Mapping, Sequence
from datetime import date, datetime, time
from decimal import Decimal
from functools import lru_cache
from typing import Any

from sqlalchemy import create_engine, event, text
from sqlalchemy.engine import Connection, Engine, Result
from sqlalchemy.exc import SQLAlchemyError

from .config import get_settings

DatabaseError = SQLAlchemyError

_IDENTITY_INSERT_TARGETS = {
    "activity_logs", "staff_notifications", "email_reminders",
    "monitoring_death_locations", "replanting_requests",
    "users", "organizations", "project_sites", "site_zones", "analyses",
    "planting_points", "planters", "planter_assignments",
    "planter_assignment_points", "auth_sessions", "map_zones", "warning_zones",
    "planting_events", "point_death_records", "monitoring_observations",
    "organization_monitoring_records", "planting_schedules", "analysis_assets",
    "object_cleanup_jobs",
}


def _normalize_value(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, (date, time)):
        return value.isoformat()
    if isinstance(value, Decimal):
        return float(value)
    if isinstance(value, (dict, list)):
        return json.dumps(value, ensure_ascii=False)
    return value


class CompatRow(Mapping[str, Any]):
    """Mapping row that also retains SQLite-style numeric indexing."""

    def __init__(self, keys: Sequence[str], values: Sequence[Any]):
        self._keys = tuple(keys)
        self._values = tuple(_normalize_value(value) for value in values)
        self._mapping = dict(zip(self._keys, self._values))
        # Domain services keep their historical response keys while the
        # production schema uses the unambiguous project_site_id name.
        if "project_site_id" in self._mapping and "site_zone_id" not in self._mapping:
            self._mapping["site_zone_id"] = self._mapping["project_site_id"]
            self._keys = (*self._keys, "site_zone_id")
            self._values = (*self._values, self._mapping["project_site_id"])

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return self._values[key]
        return self._mapping[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._keys)

    def __len__(self) -> int:
        return len(self._keys)

    def keys(self):
        return self._mapping.keys()


def _qmarks_to_named(sql: str, params: Sequence[Any]) -> tuple[str, dict[str, Any]]:
    output: list[str] = []
    bound: dict[str, Any] = {}
    index = 0
    in_single = False
    in_double = False
    cursor = 0
    while cursor < len(sql):
        char = sql[cursor]
        if char == "'" and not in_double:
            output.append(char)
            if in_single and cursor + 1 < len(sql) and sql[cursor + 1] == "'":
                output.append("'")
                cursor += 2
                continue
            in_single = not in_single
        elif char == '"' and not in_single:
            in_double = not in_double
            output.append(char)
        elif char == "?" and not in_single and not in_double:
            if index >= len(params):
                raise ValueError("SQL has more placeholders than supplied values")
            name = f"p{index}"
            output.append(f":{name}")
            bound[name] = params[index]
            index += 1
        else:
            output.append(char)
        cursor += 1
    if index != len(params):
        raise ValueError("SQL has fewer placeholders than supplied values")
    return "".join(output), bound


def _translate_sql(sql: str) -> str:
    translated = sql.strip()
    translated = re.sub(r"\bsite_zone_id\b", "project_site_id", translated)
    translated = re.sub(r"\bCOLLATE\s+NOCASE\b", "", translated, flags=re.IGNORECASE)
    translated = re.sub(
        r"date\(\s*'now'\s*,\s*'-30 days'\s*\)",
        "(CURRENT_DATE - INTERVAL '30 days')",
        translated,
        flags=re.IGNORECASE,
    )
    translated = re.sub(
        r"datetime\(\s*'now'\s*\)", "CURRENT_TIMESTAMP", translated, flags=re.IGNORECASE
    )
    if re.match(r"^INSERT\s+OR\s+IGNORE\s+INTO\b", translated, flags=re.IGNORECASE):
        translated = re.sub(
            r"^INSERT\s+OR\s+IGNORE\s+INTO\b", "INSERT INTO", translated, flags=re.IGNORECASE
        )
        translated = translated.rstrip("; ") + " ON CONFLICT DO NOTHING"
    return translated


class CompatCursor:
    def __init__(
        self,
        owner: "CompatConnection",
        result: Result[Any] | None = None,
        lastrowid: int | None = None,
    ):
        self._owner = owner
        self._result = result
        self.lastrowid = lastrowid
        self.rowcount = result.rowcount if result is not None else -1

    def execute(self, sql: str, params: Sequence[Any] = ()) -> "CompatCursor":
        replacement = self._owner.execute(sql, params)
        self._result = replacement._result
        self.lastrowid = replacement.lastrowid
        self.rowcount = replacement.rowcount
        return self

    def _row(self, row: Any | None) -> CompatRow | None:
        if row is None or self._result is None:
            return None
        return CompatRow(tuple(self._result.keys()), tuple(row))

    def fetchone(self) -> CompatRow | None:
        if self._result is None or not self._result.returns_rows:
            return None
        return self._row(self._result.fetchone())

    def fetchall(self) -> list[CompatRow]:
        if self._result is None or not self._result.returns_rows:
            return []
        keys = tuple(self._result.keys())
        return [CompatRow(keys, tuple(row)) for row in self._result.fetchall()]


class CompatConnection:
    def __init__(self, connection: Connection):
        self._connection = connection
        self.total_changes = 0

    def cursor(self) -> CompatCursor:
        return CompatCursor(self)

    @property
    def in_transaction(self) -> bool:
        return self._connection.in_transaction()

    def execute(self, sql: str, params: Sequence[Any] = ()) -> CompatCursor:
        statement = _translate_sql(sql)
        if re.match(r"^BEGIN(?:\s+IMMEDIATE)?$", statement, flags=re.IGNORECASE):
            if not self._connection.in_transaction():
                self._connection.begin()
            return CompatCursor(self)
        insert_match = re.match(
            r'^INSERT\s+INTO\s+"?([a-zA-Z_][a-zA-Z0-9_]*)"?',
            statement,
            flags=re.IGNORECASE,
        )
        returns_identity = bool(
            insert_match
            and insert_match.group(1).lower() in _IDENTITY_INSERT_TARGETS
            and not re.search(r"\bRETURNING\b", statement, flags=re.IGNORECASE)
        )
        if returns_identity:
            statement = statement.rstrip("; ") + " RETURNING id"
        rendered, bindings = _qmarks_to_named(statement, tuple(params or ()))
        result = self._connection.execute(text(rendered), bindings)
        if result.rowcount and result.rowcount > 0:
            self.total_changes += result.rowcount
        lastrowid = None
        if returns_identity and result.returns_rows:
            inserted = result.fetchone()
            if inserted is not None:
                try:
                    lastrowid = int(inserted[0])
                except (ValueError, TypeError):
                    lastrowid = None
        return CompatCursor(self, result, lastrowid)

    def commit(self) -> None:
        if self._connection.in_transaction():
            self._connection.commit()

    def rollback(self) -> None:
        if self._connection.in_transaction():
            self._connection.rollback()

    def close(self) -> None:
        self._connection.close()


@lru_cache(maxsize=1)
def get_engine() -> Engine:
    settings = get_settings()
    connect_args: dict[str, Any] = {
        "options": f"-csearch_path={settings.db_schema},extensions,public"
    }
    if settings.db_sslmode:
        connect_args["sslmode"] = settings.db_sslmode
    engine = create_engine(
        settings.database_url,
        pool_pre_ping=True,
        pool_size=settings.db_pool_size,
        max_overflow=settings.db_max_overflow,
        pool_recycle=900,
        connect_args=connect_args,
    )

    @event.listens_for(engine, "connect")
    def _set_utc(dbapi_connection, _connection_record) -> None:
        with dbapi_connection.cursor() as cursor:
            cursor.execute("SET TIME ZONE 'UTC'")

    return engine


def get_connection() -> CompatConnection:
    return CompatConnection(get_engine().connect())
