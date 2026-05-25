# SPDX-License-Identifier: Apache-2.0
"""POST /api/alternatives — word-level translation alternatives via Haiku."""
import json
import logging
import os
import re

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel, Field

from ..limits import limiter

logger = logging.getLogger(__name__)

router = APIRouter()

_SYSTEM = (
    "You are a Dhivehi–English translation assistant. "
    "Given a target word and the full sentence it appears in, suggest "
    "2-3 alternative words or phrases that could replace it while "
    "preserving the meaning. Keep notes very brief (≤8 words each). "
    "CRITICAL: If the target word is in Thaana script (Dhivehi), all "
    "alternative 'text' values MUST be in Thaana script only — no Latin. "
    "If the word is in Latin script (English), all alternatives must be Latin only. "
    "Return JSON only (no markdown): "
    '{"alternatives": [{"text": "word", "note": "brief usage note"}]}'
)


class AltRequest(BaseModel):
    word: str = Field(..., min_length=1, max_length=200)
    translation: str = Field(..., min_length=1, max_length=2000)
    source: str = Field(default="", max_length=2000)
    target_lang: str = Field(default="DV", pattern=r"^(EN|DV)$")


AltRequest.model_rebuild()


@router.post("/alternatives")
@limiter.limit("20/minute")
def alternatives(request: Request, req: AltRequest = Body(...)) -> dict:
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise HTTPException(503, "ANTHROPIC_API_KEY not configured")

    import anthropic
    from ...llm import model_id

    ctx = f"Source: {req.source}\n" if req.source else ""
    user = (
        f"{ctx}"
        f"Translation ({req.target_lang}): {req.translation}\n"
        f"Word to replace: {req.word}\n\n"
        "Suggest alternatives. Return JSON only."
    )

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model_id("haiku"),
        max_tokens=200,
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
        except Exception as e:
            logger.warning("alternatives JSON parse failed: %s", e)
            data = {}

    return {"alternatives": data.get("alternatives", []), "word": req.word}
