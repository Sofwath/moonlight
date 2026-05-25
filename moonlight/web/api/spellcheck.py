# SPDX-License-Identifier: Apache-2.0
"""POST /api/spellcheck — Dhivehi spell checker via Haiku."""
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
    "You are a Dhivehi (Thaana script) spell checker. "
    "Identify actual spelling errors in the given text — only clear misspellings, "
    "not stylistic or register differences. "
    "Return JSON only (no markdown): "
    '{"issues": [{"word": "misspelled word exactly as it appears", '
    '"suggestion": "correct Dhivehi spelling", "reason": "brief reason ≤10 words"}]}. '
    'Return {"issues": []} if the text has no spelling errors.'
)


class SpellcheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    lang: str = Field(default="DV", pattern=r"^(EN|DV)$")


SpellcheckRequest.model_rebuild()


@router.post("/spellcheck")
@limiter.limit("10/minute")
def spellcheck(request: Request, req: SpellcheckRequest = Body(...)) -> dict:
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise HTTPException(503, "ANTHROPIC_API_KEY not configured")

    import anthropic
    from ...llm import model_id

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model_id("haiku"),
        max_tokens=400,
        system=_SYSTEM,
        messages=[{"role": "user", "content": f"Text ({req.lang}): {req.text}"}],
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
            logger.warning("spellcheck JSON parse failed: %s", e)
            data = {}

    return {"issues": data.get("issues", []), "lang": req.lang}
