# SPDX-License-Identifier: Apache-2.0
"""GET /api/translate/history — recent translations from translation_runs."""
from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, Query

from ..db_dep import get_db

router = APIRouter()


@router.get("/translate/history")
def history(
    limit: int = Query(default=20, ge=1, le=100),
    conn: sqlite3.Connection = Depends(get_db),
) -> dict:
    # moonlight uses input_text/output_text; we alias to source_text/translation
    # for workbench.js compatibility
    rows = conn.execute(
        """SELECT id, source_lang, target_lang, input_text, output_text,
                  model, cost_usd, created_at
           FROM translation_runs
           ORDER BY created_at DESC
           LIMIT ?""",
        (limit,),
    ).fetchall()

    cols = ("id", "source_lang", "target_lang", "source_text",
            "translation", "model", "cost_usd", "created_at")
    runs = [{cols[i]: r[i] for i in range(len(cols))} for r in rows]
    return {"runs": runs, "count": len(runs)}
