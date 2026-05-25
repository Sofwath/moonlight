# SPDX-License-Identifier: Apache-2.0
"""POST /api/align-batch — word alignment via Haiku, with SQLite cache."""
import json
import logging
import os
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import List, Optional

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..db_dep import get_db
from ..limits import limiter

logger = logging.getLogger(__name__)

router = APIRouter()

_SYSTEM = (
    "You are a word alignment system. Given a source sentence and its translation, "
    "produce a complete word alignment: for each target word, list the source word(s) "
    "that correspond to it. Every source word should appear in at least one alignment. "
    "Return JSON only (no markdown fences): "
    '{"alignments": [{"target_word": "exact target word", "source_words": ["word1"]}]}'
)

_CACHE_TTL_HOURS = 24.0


class AlignBatchRequest(BaseModel):
    source: str = Field(..., min_length=1, max_length=2000)
    translation: str = Field(..., min_length=1, max_length=2000)
    source_lang: str = Field(default="EN", pattern=r"^(EN|DV)$")
    target_lang: str = Field(default="DV", pattern=r"^(EN|DV)$")


AlignBatchRequest.model_rebuild()


def _cache_get(
    conn: sqlite3.Connection,
    source: str,
    translation: str,
    source_lang: str,
    target_lang: str,
) -> Optional[List]:
    cutoff = (datetime.now(timezone.utc) - timedelta(hours=_CACHE_TTL_HOURS)).isoformat()
    row = conn.execute(
        """SELECT alignments FROM alignment_cache
           WHERE source_text = ? AND translation = ?
             AND source_lang = ? AND target_lang = ?
             AND created_at >= ?
           ORDER BY created_at DESC LIMIT 1""",
        (source, translation, source_lang, target_lang, cutoff),
    ).fetchone()
    if row is None:
        return None
    try:
        return json.loads(row[0])
    except Exception as e:
        logger.warning("alignment cache decode failed: %s", e)
        return None


def _cache_put(
    conn: sqlite3.Connection,
    source: str,
    translation: str,
    source_lang: str,
    target_lang: str,
    alignments: list,
) -> None:
    try:
        conn.execute(
            """INSERT INTO alignment_cache
               (source_text, translation, source_lang, target_lang, alignments, created_at)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                source, translation, source_lang, target_lang,
                json.dumps(alignments, ensure_ascii=False),
                datetime.now(timezone.utc).isoformat(),
            ),
        )
        conn.commit()
    except Exception as e:
        logger.warning("alignment cache write failed: %s", e)


@router.post("/align-batch")
@limiter.limit("30/minute")
def align_batch(
    request: Request,
    req: AlignBatchRequest = Body(...),
    conn: sqlite3.Connection = Depends(get_db),
) -> dict:
    cached = _cache_get(conn, req.source, req.translation, req.source_lang, req.target_lang)
    if cached is not None:
        return {"alignments": cached, "cache_hit": True}

    if "ANTHROPIC_API_KEY" not in os.environ:
        raise HTTPException(503, "ANTHROPIC_API_KEY not configured")

    import anthropic
    from ...llm import model_id

    user = (
        f"Source ({req.source_lang}): {req.source}\n"
        f"Translation ({req.target_lang}): {req.translation}\n\n"
        "Produce the complete word alignment. Return JSON only."
    )

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model_id("haiku"),
        max_tokens=600,
        system=_SYSTEM,
        messages=[{"role": "user", "content": user}],
    )
    text = "".join(
        getattr(b, "text", "") for b in resp.content
        if getattr(b, "type", None) == "text"
    ).strip()

    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        m = re.search(r'\{.*\}', text, re.DOTALL)
        try:
            data = json.loads(m.group()) if m else {}
        except Exception:
            data = {}

    alignments = data.get("alignments", [])
    if alignments:
        _cache_put(conn, req.source, req.translation, req.source_lang, req.target_lang, alignments)

    return {"alignments": alignments, "cache_hit": False}
