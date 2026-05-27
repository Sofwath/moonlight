# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for the moonlight translation engine.

Entry point: ``moonlight`` (registered in pyproject.toml).

Commands
--------
translate TEXT          Translate a single string (auto-detects source language)
build-glossary          Mine the corpus for bilingual term pairs (incremental by default)
build-place-names       Download GeoNames MV data and populate place_names table
build-embeddings        Backfill sentence-level embeddings (requires [embeddings] extra)
db-init                 Initialise (or migrate) the moonlight database
db-stats                Show corpus statistics
models                  List all available models with provider and pricing

build-glossary flags
--------------------
--model gemini-flash    Cheapest option (~$0.0003/pair, 20× cheaper than Haiku)
--budget 10             USD cap per run (default 30; full corpus costs ~$2 with Flash)
--full-rebuild          Reprocess all pairs from scratch instead of skipping covered ones
--sample N              Limit to N pairs (default: entire corpus)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Optional

import click

from moonlight.db import get_connection, corpus_stats
from moonlight.llm import MODELS, list_models

_MODEL_CHOICES = sorted(k for k in MODELS if k not in {"sonnet", "haiku", "opus"})


# ── Top-level group ────────────────────────────────────────────────────────────

@click.group()
@click.option(
    "--db", default=None, metavar="PATH",
    help="Path to moonlight.db.  Defaults to data/moonlight.db next to the package.",
)
@click.pass_context
def cli(ctx: click.Context, db: Optional[str]) -> None:
    """moonlight — EN ↔ DV translation engine for Maldives Presidency Office text."""
    ctx.ensure_object(dict)
    ctx.obj["db"] = db


# ── translate ──────────────────────────────────────────────────────────────────

@cli.command()
@click.argument("text")
@click.option(
    "--target", "-t", default=None,
    help="Target language: 'EN' or 'DV'.  Auto-detected if omitted.",
)
@click.option(
    "--mode", "-m", default="faithful",
    type=click.Choice(["faithful", "po_style"]),
    show_default=True,
    help=(
        "faithful: strict claim-preserving (use for any automated pipeline). "
        "po_style: PO press-release register with style-transfer polish."
    ),
)
@click.option(
    "--model", default="claude-sonnet",
    type=click.Choice(_MODEL_CHOICES),
    show_default=True,
    help="Model alias.  Run `moonlight models` to see all options with pricing.",
)
@click.option(
    "--verify", is_flag=True, default=False,
    help="Run a round-trip back-translation check (doubles cost).",
)
@click.option(
    "--candidates", "-n", default=1, type=int, show_default=True,
    help="Number of candidates to generate (Best-of-N).  Winner is returned.",
)
@click.option(
    "--json-output", "json_output", is_flag=True, default=False,
    help="Print full JSON result instead of just the translation.",
)
@click.pass_context
def translate(
    ctx: click.Context,
    text: str,
    target: Optional[str],
    mode: str,
    model: str,
    verify: bool,
    candidates: int,
    json_output: bool,
) -> None:
    """Translate TEXT between English and Dhivehi.

    Source language is auto-detected.  Set --target to override the default
    (translate to the other language).

    Examples::

        moonlight translate "ދިވެހިރާއްޖެ"
        moonlight translate "The President met with the Cabinet." --target DV
        moonlight translate "..." --mode po_style --verify
    """
    from moonlight.translator import translate as _translate

    conn = get_connection(ctx.obj.get("db"))
    try:
        result = _translate(
            conn, text,
            target_lang=target,
            model_alias=model,
            verify=verify,
            mode=mode,
            n_candidates=candidates,
        )
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        conn.close()

    if json_output:
        click.echo(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        click.echo(result["translation"])
        click.echo(
            f"\n[{result['source_lang']} → {result['target_lang']} | "
            f"model={result['model'].split('-')[1]} | "
            f"mode={result['mode']} | "
            f"cost=${result['cost_usd']:.4f} | "
            f"exemplars={len(result.get('exemplars', []))} | "
            f"glossary={result['glossary_terms_used']}]",
            err=True,
        )
        if result.get("entity_check") and not result["entity_check"]["passed"]:
            click.echo(
                f"⚠  Entity check: {result['entity_check']['summary']}", err=True)
        if verify and result.get("verification"):
            v = result["verification"]
            icon = "✓" if v["passed"] else "✗"
            click.echo(f"{icon} Back-translation check: {'passed' if v['passed'] else 'FAILED'}", err=True)
            if not v["passed"]:
                if v["numbers_lost"]:
                    click.echo(f"   numbers lost: {v['numbers_lost']}", err=True)
                if v["proper_nouns_lost"]:
                    click.echo(f"   proper nouns lost: {v['proper_nouns_lost']}", err=True)


# ── build-glossary ─────────────────────────────────────────────────────────────

@cli.command("build-glossary")
@click.option(
    "--sample", default=99999, type=int, show_default=True,
    help="Max paired articles to process (default: entire corpus).",
)
@click.option(
    "--budget", default=30.0, type=float, show_default=True,
    help="Maximum USD to spend on LLM calls.",
)
@click.option(
    "--model", default="claude-haiku",
    type=click.Choice(_MODEL_CHOICES),
    show_default=True,
    help="Model alias.  Haiku is cheapest for bulk extraction (~$0.003/pair).",
)
@click.option(
    "--full-rebuild", is_flag=True, default=False,
    help="Reprocess ALL pairs from scratch instead of skipping already-covered articles.",
)
@click.pass_context
def build_glossary(
    ctx: click.Context,
    sample: int,
    budget: float,
    model: str,
    full_rebuild: bool,
) -> None:
    """Mine the full corpus for bilingual term pairs and populate translation_glossary.

    By default runs incrementally — only processes article pairs not yet
    represented in the glossary, so re-runs are cheap and fast.

    Use --full-rebuild to reprocess everything from scratch.

    Cost: ~$0.003–0.005 per pair with Haiku. Full corpus (~7 000 pairs) ≈ $20–35.
    """
    from moonlight.translator import build_glossary as _build_glossary

    conn = get_connection(ctx.obj.get("db"))

    def _progress(processed: int, total: int, cost: float) -> None:
        click.echo(
            f"\r  {processed}/{total} articles  ${cost:.3f}",
            nl=False, err=True,
        )

    try:
        result = _build_glossary(
            conn,
            sample_size=sample,
            budget_usd=budget,
            progress_cb=_progress,
            model_alias=model,
            incremental=not full_rebuild,
        )
    except Exception as exc:
        click.echo(f"\nError: {exc}", err=True)
        sys.exit(1)
    finally:
        conn.close()

    click.echo()
    click.echo(
        f"Done — {result['pairs_processed']} new articles processed "
        f"({result.get('skipped', 0)} already covered), "
        f"{result['pairs_in_db']} total pairs in glossary, "
        f"${result['cost_usd']:.3f} spent."
    )


# ── build-place-names ──────────────────────────────────────────────────────────

@cli.command("build-place-names")
@click.option(
    "--data-dir", default=None, metavar="DIR",
    help="Directory to cache GeoNames downloads.  Defaults to data/geonames/.",
)
@click.option(
    "--timeout", default=60, type=int, show_default=True,
    help="HTTP timeout in seconds for GeoNames downloads.",
)
@click.pass_context
def build_place_names(
    ctx: click.Context,
    data_dir: Optional[str],
    timeout: int,
) -> None:
    """Download GeoNames MV data and populate the place_names table.

    Downloads two files from geonames.org:

      • MV.zip   — ~1 200 Maldivian place entries (islands, atolls, reefs)
      • alternatenames/MV.zip — Thaana and romanised name variants

    Already-downloaded files are reused.  Requires an internet connection on
    first run; subsequent runs are offline.
    """
    from moonlight.place_names import build_place_names as _build, init_place_names

    conn = get_connection(ctx.obj.get("db"))
    try:
        init_place_names(conn)
        _dir = Path(data_dir) if data_dir else None
        result = _build(conn, data_dir=_dir, timeout=timeout)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        conn.close()

    click.echo(
        f"Done — {result.get('upserted', 0)} rows upserted into place_names table "
        f"({result.get('with_thaana', 0)} with Thaana, total {result.get('total', 0)})."
    )


# ── build-embeddings ───────────────────────────────────────────────────────────

@cli.command("build-embeddings")
@click.option(
    "--batch", default=256, type=int, show_default=True,
    help="Sentences per embedding batch.",
)
@click.option(
    "--model", "embed_model",
    default="paraphrase-multilingual-MiniLM-L12-v2",
    show_default=True,
    help="sentence-transformers model name.",
)
@click.pass_context
def build_embeddings(
    ctx: click.Context,
    batch: int,
    embed_model: str,
) -> None:
    """Backfill sentence-level embeddings for semantic retrieval.

    Requires the [embeddings] extra (``pip install moonlight-mt[embeddings]``).

    Embeddings enable hybrid BM25 + semantic retrieval in
    ``select_sentence_memory_hybrid()``.  Without this step, the translator
    falls back to pure BM25 — still functional, lower register quality.

    The model ``paraphrase-multilingual-MiniLM-L12-v2`` is 278 MB and
    natively supports both Latin and Thaana script.
    """
    try:
        from moonlight.sentence_memory import backfill_sentence_embeddings
    except ImportError as exc:
        click.echo(
            f"Error: sentence-transformers not installed.\n"
            f"Install the embeddings extra: pip install moonlight-mt[embeddings]\n"
            f"({exc})",
            err=True,
        )
        sys.exit(1)

    conn = get_connection(ctx.obj.get("db"))
    try:
        result = backfill_sentence_embeddings(
            conn, model_name=embed_model, batch_size=batch)
    except Exception as exc:
        click.echo(f"Error: {exc}", err=True)
        sys.exit(1)
    finally:
        conn.close()

    click.echo(
        f"Done — {result.get('processed', 0)} new sentences embedded, "
        f"{result.get('skipped_existing', 0)} already had embeddings."
    )


# ── models ────────────────────────────────────────────────────────────────────

@cli.command("models")
@click.option(
    "--json-output", "json_output", is_flag=True, default=False,
    help="Print raw JSON instead of a table.",
)
def models_cmd(json_output: bool) -> None:
    """List all available models with provider, pricing, and context window.

    Set the corresponding API key environment variable before using a model:

    \b
      ANTHROPIC_API_KEY   — claude-sonnet, claude-haiku, claude-opus
      OPENAI_API_KEY      — gpt-4o, gpt-4o-mini, o3-mini
      GEMINI_API_KEY      — gemini-flash, gemini-pro
      GOOGLE_API_KEY      — alias for Gemini models (also accepted)
      DEEPSEEK_API_KEY    — deepseek, deepseek-r1
      DASHSCOPE_API_KEY   — qwen-turbo, qwen-plus, qwen-max
      MISTRAL_API_KEY     — mistral-large, mistral-small
      GROQ_API_KEY        — llama-3.3-70b
      XAI_API_KEY         — grok-2
    """
    rows = list_models()
    if json_output:
        click.echo(json.dumps(rows, indent=2))
        return

    # Table header
    fmt = "{:<18} {:<14} {:>8} {:>9} {:>10}  {}"
    click.echo(fmt.format("ALIAS", "PROVIDER", "IN $/1M", "OUT $/1M", "CTX (k)", "NOTES"))
    click.echo("-" * 90)
    prev_family = ""
    for r in rows:
        family = r["family"].split("/")[0].strip()
        if family != prev_family:
            if prev_family:
                click.echo()
            prev_family = family
        r["id"].split("-")[0] if r["provider"] == "openai_compat" else "anthropic"
        click.echo(fmt.format(
            r["alias"],
            r["family"].split("/")[-1].strip()[:14],
            f"${r['in_per_m']:.2f}",
            f"${r['out_per_m']:.2f}",
            str(r["context_k"]),
            r["notes"][:60],
        ))


# ── db-init ────────────────────────────────────────────────────────────────────

@cli.command("db-init")
@click.pass_context
def db_init(ctx: click.Context) -> None:
    """Initialise (or migrate) the moonlight database.

    Safe to run on an existing database — all CREATE TABLE statements use
    IF NOT EXISTS and no data is dropped.
    """
    conn = get_connection(ctx.obj.get("db"))
    conn.close()
    db_path = ctx.obj.get("db") or "data/moonlight.db"
    click.echo(f"Database ready: {db_path}")


# ── db-stats ───────────────────────────────────────────────────────────────────

@cli.command("db-stats")
@click.pass_context
def db_stats(ctx: click.Context) -> None:
    """Show corpus statistics."""
    conn = get_connection(ctx.obj.get("db"))
    try:
        stats = corpus_stats(conn)
        glossary_n = conn.execute(
            "SELECT COUNT(*) FROM translation_glossary").fetchone()[0]
        runs_n = conn.execute(
            "SELECT COUNT(*) FROM translation_runs").fetchone()[0]
        place_n = conn.execute(
            "SELECT COUNT(*) FROM place_names").fetchone()[0]
        sp_n = conn.execute(
            "SELECT COUNT(*) FROM sentence_pairs").fetchone()[0]
    finally:
        conn.close()

    click.echo(f"Articles       EN: {stats['en_articles']:>6}   DV: {stats['dv_articles']:>6}")
    click.echo(f"Paired pairs      : {stats['paired_articles']:>6}")
    click.echo(f"Date range        : {stats['date_min']} – {stats['date_max']}")
    click.echo(f"Sentence pairs    : {sp_n:>6}")
    click.echo(f"Glossary terms    : {glossary_n:>6}")
    click.echo(f"Translation runs  : {runs_n:>6}")
    click.echo(f"Place names       : {place_n:>6}")
    if stats["categories"]:
        click.echo("\nTop categories (EN):")
        for cat, n in list(stats["categories"].items())[:8]:
            click.echo(f"  {cat:<30} {n:>5}")
