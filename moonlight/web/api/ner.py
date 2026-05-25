# SPDX-License-Identifier: Apache-2.0
"""POST /api/ner — lightweight Named Entity Recognition via Haiku."""
import json
import os
import re

from fastapi import APIRouter, Body, HTTPException, Request
from pydantic import BaseModel, Field

from ..limits import limiter

router = APIRouter()

_SYSTEM = (
    "You are a named entity recognition system. Extract named entities from "
    "the given text. Return JSON only (no markdown): "
    '{"entities": [{"text": "entity text as it appears", "type": "PERSON|ORG|LOC|DATE|MISC"}]}. '
    "Multi-word entities should be returned as a single entry with the full phrase. "
    "Only include clearly identifiable named entities, not common nouns."
)


class NERRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=4000)
    lang: str = Field(default="DV")


NERRequest.model_rebuild()


@router.post("/ner")
@limiter.limit("20/minute")
def ner(request: Request, req: NERRequest = Body(...)) -> dict:
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise HTTPException(503, "ANTHROPIC_API_KEY not configured")

    import anthropic
    from ...llm import model_id

    client = anthropic.Anthropic()
    resp = client.messages.create(
        model=model_id("haiku"),
        max_tokens=300,
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
        except Exception:
            data = {}

    return {"entities": data.get("entities", []), "lang": req.lang}
