# SPDX-License-Identifier: Apache-2.0
"""GET /api/glossary — browse and search the EN↔DV terminology glossary."""
from __future__ import annotations

import sqlite3
from typing import Optional

from fastapi import APIRouter, Depends, Query

from ..db_dep import get_db

router = APIRouter()


@router.get("/glossary")
def glossary(
    q: Optional[str] = Query(default=None, max_length=200),
    domain: Optional[str] = Query(default=None, max_length=100),
    limit: int = Query(default=50, ge=1, le=200),
    conn: sqlite3.Connection = Depends(get_db),
) -> dict:
    params: list = []
    clauses: list[str] = []

    if q:
        like = f"%{q.lower()}%"
        clauses.append("(LOWER(en_term) LIKE ? OR LOWER(dv_term) LIKE ?)")
        params.extend([like, like])
    if domain:
        clauses.append("domain = ?")
        params.append(domain)

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""

    rows = conn.execute(
        f"""SELECT id, en_term, dv_term, domain, freq, confidence
            FROM translation_glossary
            {where}
            ORDER BY freq DESC, confidence DESC
            LIMIT ?""",
        [*params, limit],
    ).fetchall()

    total_row = conn.execute(
        f"SELECT COUNT(*) FROM translation_glossary {where}", params
    ).fetchone()
    total = total_row[0] if total_row else 0

    cols = ("id", "en_term", "dv_term", "domain", "freq", "confidence")
    terms = [{cols[i]: r[i] for i in range(len(cols))} for r in rows]
    return {"terms": terms, "total": total}
