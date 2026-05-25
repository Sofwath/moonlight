# SPDX-License-Identifier: Apache-2.0
"""GET /api/concordance — corpus concordance search for the workbench."""
from __future__ import annotations

import re
import sqlite3

from fastapi import APIRouter, Depends, Query

from ..db_dep import get_db
from ...corpus import search_articles

router = APIRouter()

_WINDOW = 160


def _extract_snippet(body: str, query: str) -> str:
    if not body:
        return ""
    lo_body = body.lower()
    lo_q = re.escape(query.lower())
    m = re.search(lo_q, lo_body)
    if not m:
        snippet = body[:320].rstrip()
        return snippet + ("…" if len(body) > 320 else "")
    s = max(0, m.start() - _WINDOW)
    e = min(len(body), m.end() + _WINDOW)
    out = body[s:e].strip()
    if s > 0:
        out = "…" + out
    if e < len(body):
        out = out + "…"
    return out


@router.get("/concordance")
def concordance(
    q: str = Query(..., min_length=1, max_length=200),
    lang: str = Query(default="DV", pattern=r"^(EN|DV)$"),
    limit: int = Query(default=5, ge=1, le=10),
    conn: sqlite3.Connection = Depends(get_db),
) -> dict:
    hits = search_articles(conn, q, language=lang, limit=limit)
    results = [
        {
            "article_id":     h["article_id"],
            "title":          h["title"] or "",
            "snippet":        _extract_snippet(h["body_text"] or "", q),
            "published_date": h["published_date"] or "",
            "language":       h["language"],
            "paired_id":      h["paired_id"],
        }
        for h in hits
    ]
    return {"results": results, "q": q, "lang": lang}
