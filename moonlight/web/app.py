# SPDX-License-Identifier: Apache-2.0
"""FastAPI app for moonlight workbench.

Routes:
  GET  /workbench    → translation workbench UI
  GET  /             → redirect to /workbench

  POST /api/translate
  GET  /api/concordance
  GET  /api/glossary
  POST /api/align-batch
  POST /api/alternatives
  POST /api/ner
  POST /api/spellcheck
  GET  /api/translate/history
  POST /api/fluency
  GET  /api/benchmarks
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, PlainTextResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware

from .api import (
    align_batch,
    alternatives,
    benchmarks as benchmarks_api,
    concordance,
    fluency,
    glossary_api,
    history,
    ner,
    spellcheck,
    translate as translate_api,
)
from .limits import limiter

for name in ("httpx", "httpcore", "anthropic"):
    logging.getLogger(name).setLevel(logging.WARNING)

PKG_DIR = Path(__file__).parent
STATIC_DIR = PKG_DIR / "static"

app = FastAPI(title="moonlight", docs_url="/api/docs", redoc_url=None)
app.state.limiter = limiter
app.add_middleware(SlowAPIMiddleware)


@app.exception_handler(RateLimitExceeded)
def _rate_limit_handler(request, exc):
    return PlainTextResponse("rate limit exceeded — try again shortly", status_code=429)


@app.middleware("http")
async def _no_cache_html(request, call_next):
    response = await call_next(request)
    ctype = response.headers.get("content-type", "")
    if ctype.startswith("text/html"):
        response.headers["Cache-Control"] = "no-store, must-revalidate"
        response.headers["Pragma"] = "no-cache"
    return response


# API routers
app.include_router(translate_api.router, prefix="/api", tags=["translate"])
app.include_router(concordance.router, prefix="/api", tags=["concordance"])
app.include_router(glossary_api.router, prefix="/api", tags=["glossary"])
app.include_router(align_batch.router, prefix="/api", tags=["align"])
app.include_router(alternatives.router, prefix="/api", tags=["alternatives"])
app.include_router(ner.router, prefix="/api", tags=["ner"])
app.include_router(spellcheck.router, prefix="/api", tags=["spellcheck"])
app.include_router(history.router, prefix="/api", tags=["history"])
app.include_router(fluency.router, prefix="/api", tags=["fluency"])
app.include_router(benchmarks_api.router, prefix="/api", tags=["benchmarks"])

# Static assets
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


@app.get("/", include_in_schema=False)
def page_root():
    return RedirectResponse("/workbench")


@app.get("/workbench", include_in_schema=False)
def page_workbench():
    return FileResponse(STATIC_DIR / "workbench.html")


@app.get("/robots.txt", include_in_schema=False)
def robots():
    return PlainTextResponse("User-agent: *\nAllow: /\n")
