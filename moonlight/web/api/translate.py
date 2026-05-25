# SPDX-License-Identifier: Apache-2.0
"""POST /api/translate — EN ↔ DV translation via moonlight engine."""
import logging
import os
import sqlite3
from typing import List, Optional

logger = logging.getLogger(__name__)

from fastapi import APIRouter, Body, Depends, HTTPException, Request
from pydantic import BaseModel, Field

from ..db_dep import get_db
from ..limits import TRANSLATE_DAILY_CAP_USD, limiter

router = APIRouter()

_ABLATE_OPTIONS = {
    "few_shot", "glossary", "phrase_contexts",
    "sentence_memory", "genre_routing", "term_locking", "polish",
}


class TranslateRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    target_language: Optional[str] = Field(
        default="auto", pattern=r"^(EN|DV|auto)$"
    )
    verify: bool = Field(default=False)
    mode: str = Field(default="faithful", pattern=r"^(faithful|po_style)$")
    model: str = Field(
        default="sonnet",
        pattern=r"^(sonnet|opus|haiku|claude-sonnet|claude-opus|claude-haiku)$",
    )
    n_candidates: int = Field(
        default=1, ge=1, le=3,
        description="Best-of-N candidate scoring (1=off, 3=+0.06 chrF at ~3× cost)",
    )
    ablate: Optional[List[str]] = Field(
        default=None,
        description=(
            "Research: disable pipeline components to measure their contribution. "
            "Valid values: few_shot, glossary, phrase_contexts, sentence_memory, "
            "genre_routing, term_locking, polish."
        ),
    )


TranslateRequest.model_rebuild()


@router.post("/translate")
@limiter.limit("10/minute")
def translate(
    request: Request,
    req: TranslateRequest = Body(...),
    conn: sqlite3.Connection = Depends(get_db),
) -> dict:
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise HTTPException(503, "ANTHROPIC_API_KEY not configured on server")

    # Check daily spend against cap
    try:
        row = conn.execute(
            "SELECT COALESCE(SUM(cost_usd),0) FROM translation_runs "
            "WHERE created_at >= date('now')"
        ).fetchone()
        daily = float(row[0]) if row else 0.0
    except Exception as e:
        logger.warning("daily spend check failed, assuming $0: %s", e)
        daily = 0.0

    if daily >= TRANSLATE_DAILY_CAP_USD:
        raise HTTPException(
            503,
            f"daily translation budget exhausted "
            f"(${daily:.2f} / ${TRANSLATE_DAILY_CAP_USD:.2f}). Try again tomorrow.",
        )

    # Validate ablate values
    ablate_set: set = set()
    if req.ablate:
        unknown = set(req.ablate) - _ABLATE_OPTIONS
        if unknown:
            raise HTTPException(400, f"unknown ablate values: {sorted(unknown)}")
        ablate_set = set(req.ablate)

    from ...translator import translate as _do_translate

    try:
        res = _do_translate(
            conn,
            req.text,
            target_lang=req.target_language if req.target_language != "auto" else None,
            verify=req.verify,
            mode=req.mode,
            model_alias=req.model,
            n_candidates=req.n_candidates,
            ablate=ablate_set if ablate_set else None,
        )
    except ValueError as e:
        raise HTTPException(400, str(e))
    except RuntimeError as e:
        raise HTTPException(503, str(e))
    except Exception as e:
        raise HTTPException(500, f"translation failed: {e}")

    # Attach actual glossary term pairs (the engine only returns a count)
    src_lower = req.text.lower()
    try:
        rows = conn.execute(
            """SELECT en_term, dv_term, domain, confidence
               FROM translation_glossary
               WHERE INSTR(?, LOWER(en_term)) > 0
               ORDER BY LENGTH(en_term) DESC, freq DESC
               LIMIT 20""",
            (src_lower,),
        ).fetchall()
        res["glossary_terms"] = [
            {"en_term": r[0], "dv_term": r[1], "domain": r[2], "confidence": r[3]}
            for r in rows
        ]
    except Exception as e:
        logger.warning("glossary_terms fetch failed: %s", e)
        res["glossary_terms"] = []

    # Enrich phrase_contexts with target-side snippets from paired articles
    for pc in res.get("phrase_contexts", []):
        paired_id = pc.get("paired_id")
        if paired_id:
            try:
                row = conn.execute(
                    "SELECT body_text FROM articles WHERE id = ?",
                    (paired_id,),
                ).fetchone()
                if row and row[0]:
                    pc["target_snippet"] = row[0][:400]
            except Exception as e:
                logger.debug("phrase_context target_snippet fetch failed: %s", e)

    res["word_count"] = len(req.text.split())
    return res
