# SPDX-License-Identifier: Apache-2.0
"""Shared DB connection dependency for moonlight FastAPI routes.

Each request gets its own SQLite connection.
"""
from __future__ import annotations

import sqlite3
from typing import Iterator

from fastapi import Request

from ..db import DEFAULT_DB_PATH, init_db


_schema_initialised = False


def _ensure_schema() -> None:
    global _schema_initialised
    if _schema_initialised:
        return
    conn = sqlite3.connect(str(DEFAULT_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    try:
        init_db(conn)
    finally:
        conn.close()
    _schema_initialised = True


def get_db(_request: Request) -> Iterator[sqlite3.Connection]:
    _ensure_schema()
    conn = sqlite3.connect(str(DEFAULT_DB_PATH), check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    conn.row_factory = sqlite3.Row
    try:
        yield conn
    finally:
        conn.close()
