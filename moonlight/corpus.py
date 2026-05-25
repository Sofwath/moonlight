# SPDX-License-Identifier: Apache-2.0
"""Corpus management and retrieval for moonlight.

The Maldives Presidency Office publishes every press release in both English
and Dhivehi — a rare bilingual government corpus for a genuinely low-resource
language. This module manages that corpus as an SQLite table with an FTS5
BM25 index, and exposes the retrieval functions the translator needs to build
its few-shot prompt.

Retrieval design:

  1. BM25 (FTS5)  — exact-match term recall over article bodies.
                    Fast, no embeddings required. Works well for named
                    entities ("Judicial Service Commission", "Vilimalé")
                    that semantic models may not have seen.

  2. Hybrid mode — when sentence-transformers is installed, BM25 results
                    are re-ranked by cosine similarity with a multilingual
                    embedding. Semantic coverage improves; exact-match
                    recall is preserved via the initial BM25 pre-filter.

  3. Genre routing — classify the input by genre (state_visit, speech,
                    condolence, …) and prefer same-genre exemplars. Press
                    releases have strong genre conventions; register is
                    established at generation time by the few-shot examples.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

log = logging.getLogger(__name__)

# FTS column weights: title is short and high-signal; body provides context.
_BM25_WEIGHTS = (3.0, 1.0)


# ── FTS5 init + backfill ──────────────────────────────────────────────────────

_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts
USING fts5(
    article_id UNINDEXED,
    language UNINDEXED,
    title,
    body,
    tokenize='unicode61'
);
"""

_TRIGGERS_SQL = """
CREATE TRIGGER IF NOT EXISTS articles_fts_ai
AFTER INSERT ON articles BEGIN
    INSERT INTO articles_fts (article_id, language, title, body)
    VALUES (new.id, new.language,
            COALESCE(new.title, ''), COALESCE(new.body_text, ''));
END;
CREATE TRIGGER IF NOT EXISTS articles_fts_au
AFTER UPDATE OF title, body_text ON articles BEGIN
    DELETE FROM articles_fts WHERE article_id=old.id AND language=old.language;
    INSERT INTO articles_fts (article_id, language, title, body)
    VALUES (new.id, new.language,
            COALESCE(new.title, ''), COALESCE(new.body_text, ''));
END;
CREATE TRIGGER IF NOT EXISTS articles_fts_ad
AFTER DELETE ON articles BEGIN
    DELETE FROM articles_fts WHERE article_id=old.id AND language=old.language;
END;
"""


def init_corpus_fts(conn: sqlite3.Connection) -> bool:
    """Create the FTS5 virtual table and sync triggers. Idempotent.

    Returns True if FTS5 is available on this SQLite build (it almost
    always is, but some minimal builds omit it). Falls back gracefully
    to returning False; callers should check and warn."""
    try:
        conn.executescript(_FTS_SQL)
        conn.executescript(_TRIGGERS_SQL)
        conn.commit()
        return True
    except sqlite3.OperationalError as exc:
        log.warning("FTS5 unavailable — BM25 retrieval disabled (%s)", exc)
        return False


def backfill_corpus_fts(conn: sqlite3.Connection,
                         progress_cb=None) -> int:
    """Populate articles_fts from existing rows. Idempotent (clears first).

    On a ~5,300-article corpus this takes 5-10 seconds.
    Called automatically on first `init_db()` if the FTS table is empty.
    """
    try:
        conn.execute("DELETE FROM articles_fts")
    except sqlite3.OperationalError:
        return 0
    total = conn.execute(
        "SELECT COUNT(*) FROM articles "
        "WHERE body_text IS NOT NULL AND body_text != ''"
    ).fetchone()[0]
    if total == 0:
        return 0
    written = 0
    for offset in range(0, total, 500):
        rows = conn.execute(
            "SELECT id, language, COALESCE(title,''), COALESCE(body_text,'') "
            "FROM articles "
            "WHERE body_text IS NOT NULL AND body_text != '' "
            "ORDER BY id LIMIT 500 OFFSET ?",
            (offset,),
        ).fetchall()
        for r in rows:
            conn.execute(
                "INSERT INTO articles_fts (article_id, language, title, body) "
                "VALUES (?, ?, ?, ?)", r
            )
            written += 1
            if progress_cb and written % 1000 == 0:
                progress_cb(written, total)
        conn.commit()
    if progress_cb:
        progress_cb(written, total)
    return written


# ── Query sanitisation ────────────────────────────────────────────────────────

def _fts_sanitize(query: str, max_tokens: int = 15) -> str:
    """Quote each token so FTS5 special chars (AND, OR, NEAR, *) don't raise.

    Accepts both Latin and Thaana characters (U+0780–U+07BF) so queries
    from DV-language inputs work correctly. FTS5's default tokenizer splits
    on Unicode character class changes, so Thaana tokens work out of the box.

    ``max_tokens`` caps how many tokens are joined (AND logic): using a full
    article body as a query would require ALL tokens to appear simultaneously,
    returning nothing. 15 tokens from the start of the text is enough to
    capture topic + genre without over-constraining the match.
    """
    tokens = re.findall(r"[A-Za-z0-9'ހ-޿]+", query)[:max_tokens]
    return " ".join(f'"{t}"' for t in tokens) if tokens else '""'


# ── Genre classification ──────────────────────────────────────────────────────
# Classify the input text into a press-release genre so the few-shot
# selector can prefer same-genre exemplars. Genre is established by
# register, not topic — a state-visit article has different sentence
# structure than a budget announcement even when both mention the
# same ministers.

_GENRE_PATTERNS: dict[str, dict[str, list[str]]] = {
    "state_visit": {
        "en": ["state visit", "official visit", "arrived in", "departs for",
                "courtesy call", "his excellency", "her excellency"],
        "dv": ["ފުރަމާނަ ދަތުރުފުޅު", "ރަސްމީ ދަތުރުފުޅު",
                "ވަޑައިގެންފި", "ފުރާވަޑައިގެންފި"],
    },
    "condolence": {
        "en": ["condolences", "deepest sympathies", "tragic", "passed away",
                "demise", "earthquake", "loss of life"],
        "dv": ["ތަޢުޒިޔާ", "ހިތާމަ", "ނިޔާވެ", "ހާދިސާ"],
    },
    "appointment": {
        "en": ["appoints", "appointed", "took the oath", "sworn in", "as a member"],
        "dv": ["ޢައްޔަންކުރައް", "ޢައްޔަންކޮށްފި", "ހުވާކުރައް", "މަޤާމަށް"],
    },
    "speech": {
        "en": ["addressing", "presidential address", "delivered a", "remarks at",
                "keynote"],
        "dv": ["ޚިޠާބު", "ވާހަކަފުޅުދައް", "ރިޔާސީ ބަޔާން", "މެސެޖު"],
    },
    "budget": {
        "en": ["million", "billion", "budget", "mvr", "expenditure", "fiscal",
                "allocated", "funds"],
        "dv": ["ބިލިއަން", "މިލިއަން", "ބަޖެޓު", "ރުފިޔާ", "ޚަރަދު"],
    },
}


def classify_genre(text: str, lang: str = "EN") -> Optional[str]:
    """Return the best-matching genre label or None.

    Scans `text` for pattern hits (case-insensitive for EN). The genre
    with the most hits wins. Returns None when no genre scores > 0.
    """
    if not text:
        return None
    lang_key = "dv" if lang == "DV" else "en"
    text_lc = text.lower() if lang == "EN" else text
    scores: dict[str, int] = {}
    for genre, patterns in _GENRE_PATTERNS.items():
        hits = 0
        for pat in patterns.get(lang_key, []):
            if (pat.lower() if lang == "EN" else pat) in text_lc:
                hits += 1
        if hits:
            scores[genre] = hits
    if not scores:
        return None
    return max(scores, key=lambda g: scores[g])


# ── Article search ────────────────────────────────────────────────────────────

def search_articles(
    conn: sqlite3.Connection,
    query: str,
    *,
    language: str = "EN",
    limit: int = 10,
    require_paired: bool = False,
    recency_days: Optional[int] = None,
    exclude_ids: Optional[list[int]] = None,
) -> list[dict]:
    """BM25 full-text search over the articles corpus.

    Args:
        query:          The search query (EN or DV text).
        language:       Which language side to search ('EN' or 'DV').
        limit:          Max results to return.
        require_paired: Only return articles that have a paired counterpart
                        (needed for few-shot, which requires both sides).
        recency_days:   Restrict to articles published in the last N days.
        exclude_ids:    Article IDs to exclude. When a test article is being
                        evaluated, its own ID (and its paired counterpart's ID)
                        must be excluded to prevent ground-truth leak.

    Returns:
        List of dicts with keys: article_id, language, paired_id, title,
        body_text, published_date, category, rank (BM25, negative = better).
    """
    if not query or not query.strip():
        return []
    weights_sql = ", ".join(str(w) for w in _BM25_WEIGHTS)
    clauses = ["a.language = ?"]
    params: list = [language]
    if require_paired:
        clauses.append("a.paired_id IS NOT NULL")
    if recency_days is not None and recency_days > 0:
        cutoff = (datetime.now(timezone.utc) - timedelta(days=recency_days)
                  ).strftime("%Y-%m-%d")
        clauses.append("a.published_date >= ?")
        params.append(cutoff)
    if exclude_ids:
        ph = ",".join("?" * len(exclude_ids))
        clauses.append(f"a.id NOT IN ({ph})")
        params.extend(exclude_ids)
        clauses.append(f"(a.paired_id IS NULL OR a.paired_id NOT IN ({ph}))")
        params.extend(exclude_ids)
    where_extra = " AND " + " AND ".join(clauses) if clauses else ""
    sql = f"""
        SELECT a.id AS article_id, a.language, a.paired_id,
               a.title, a.body_text, a.published_date, a.category,
               bm25(articles_fts, {weights_sql}) AS rank
        FROM articles_fts f
        JOIN articles a ON a.id = f.article_id AND a.language = f.language
        WHERE articles_fts MATCH ? {where_extra}
        ORDER BY rank
        LIMIT ?
    """
    rows = conn.execute(
        sql, [_fts_sanitize(query), *params, limit]
    ).fetchall()
    cols = ("article_id", "language", "paired_id", "title", "body_text",
            "published_date", "category", "rank")
    return [{cols[i]: r[i] for i in range(len(cols))} for r in rows]


# ── Few-shot exemplar selection ───────────────────────────────────────────────

def select_few_shot(
    conn: sqlite3.Connection,
    source_lang: str,
    query_text: str,
    *,
    k: int = 3,
    exclude_article_ids: Optional[set] = None,
    prefer_genre: Optional[str] = None,
) -> list[dict]:
    """Select k topic-similar paired articles as few-shot exemplars.

    Each exemplar provides a (source_body, target_body) pair the LLM
    sees as a translation example before seeing the actual input.

    Genre routing: if `prefer_genre` is given, first search for same-
    genre exemplars. If fewer than k are found, top up with topic-similar
    (genre-agnostic) results. This ensures a state-visit input gets
    state-visit examples — a 10× improvement in register matching vs
    pure topic similarity.

    Excludes articles in `exclude_article_ids` (and their paired
    counterparts) to prevent ground-truth leak during evaluation.
    """
    excl: list[int] = list(exclude_article_ids or [])
    target_lang = "DV" if source_lang == "EN" else "EN"
    lang_key = "en" if source_lang == "EN" else "dv"

    def _fetch(query: str, n: int = k * 3, recency_days: Optional[int] = 90) -> list[dict]:
        return search_articles(
            conn, query, language=source_lang, limit=n,
            require_paired=True, recency_days=recency_days,
            exclude_ids=excl,
        )

    # Build a genre keyword query for the source language.  The `prefer_genre`
    # string (e.g. "condolence") is an English label — prepending it to a DV
    # query does nothing useful.  Instead use the language-appropriate keywords
    # from _GENRE_PATTERNS so DV queries can find genre-matched exemplars.
    #
    # Use only the FIRST (most distinctive) keyword: the full list joined by
    # spaces is ANDed by _fts_sanitize/FTS5, which over-constrains the match
    # (e.g. "condolences" AND "earthquake" AND "loss of life" matches nothing).
    def _genre_query() -> str:
        kws = _GENRE_PATTERNS.get(prefer_genre or "", {}).get(lang_key, [])
        return kws[0] if kws else ""

    # 1. Try genre keywords first (language-appropriate terms)
    exemplars = []
    if prefer_genre:
        gq = _genre_query()
        if gq:
            genre_hits = _fetch(gq)
            exemplars = genre_hits[:k]

    # 2. Fall back to topic-similar using the source text (first 15 tokens)
    if len(exemplars) < k:
        topic_hits = _fetch(query_text)
        seen_ids = {e["article_id"] for e in exemplars}
        for h in topic_hits:
            if h["article_id"] not in seen_ids and len(exemplars) < k:
                exemplars.append(h)

    # 3. Recency cascade: if the 90-day window left us with no exemplars at all,
    #    widen to 365 days then to all-time so sparse topics always get at least
    #    one. `exemplars` is guaranteed empty here, so there are no prior IDs to
    #    carry forward — the per-iteration seen_ids is sufficient for dedup within
    #    each widened batch.
    if not exemplars:
        for recency in (365, None):
            wider_hits = _fetch(query_text, recency_days=recency)
            seen_ids_wide: set[int] = set()
            for h in wider_hits:
                if h["article_id"] not in seen_ids_wide and len(exemplars) < k:
                    exemplars.append(h)
                    seen_ids_wide.add(h["article_id"])
            if exemplars:
                break

    # Fetch paired target articles
    result = []
    for hit in exemplars:
        paired_id = hit["paired_id"]
        if paired_id is None:
            continue
        paired = conn.execute(
            "SELECT id, language, body_text, title FROM articles "
            "WHERE id = ? AND language = ?",
            (paired_id, target_lang),
        ).fetchone()
        if paired is None:
            continue
        src_body = hit["body_text"] or ""
        tgt_body = paired["body_text"] or ""
        if not src_body.strip() or not tgt_body.strip():
            continue
        result.append({
            "article_id":    hit["article_id"],
            "paired_id":     paired_id,
            "source_body":   src_body[:2000],
            "target_body":   tgt_body[:2000],
            "published_date": hit["published_date"],
            "category":      hit["category"],
        })
    return result


# ── Phrase-context retrieval ──────────────────────────────────────────────────

def select_phrase_contexts(
    conn: sqlite3.Connection,
    text: str,
    source_lang: str,
    *,
    max_phrases: int = 4,
    snippets_per_phrase: int = 1,
    exclude_article_ids: Optional[set] = None,
) -> list[dict]:
    """Retrieve phrase-level cross-lingual usage examples.

    Extracts candidate phrases (multi-word noun phrases heuristically)
    from the input text, then for each phrase finds articles where it
    appears and returns a (source_snippet, paired_article_id) pair.
    The translator uses these as sub-article-level context clues for
    consistent terminology.

    This is more fine-grained than article-level few-shot: instead of
    providing full article pairs, it pinpoints how specific phrases are
    rendered in context.
    """
    if not text or not text.strip():
        return []
    excl: list[int] = list(exclude_article_ids or [])

    # Extract candidate phrases: 2-4 word sequences of capitalised words
    # (EN) or multi-character Thaana runs (DV).
    phrases: list[str] = []
    if source_lang == "EN":
        # Title-cased noun phrases: e.g. "Judicial Service Commission"
        for m in re.finditer(r"[A-Z][a-z]+(?:\s+[A-Z][a-z]+){1,3}", text):
            p = m.group(0)
            if p not in phrases:
                phrases.append(p)
    else:
        # Thaana: 3+ word runs likely to be institutional names
        for m in re.finditer(r"[ހ-޿]+(?:\s+[ހ-޿]+){1,2}", text):
            p = m.group(0)
            if p not in phrases:
                phrases.append(p)
    if not phrases:
        return []

    contexts: list[dict] = []
    seen_ids: set[int] = set()
    for phrase in phrases[:max_phrases * 2]:
        if len(contexts) >= max_phrases:
            break
        hits = search_articles(
            conn, phrase, language=source_lang, limit=3,
            require_paired=True, recency_days=730,
            exclude_ids=excl,
        )
        for h in hits[:snippets_per_phrase]:
            if h["article_id"] in seen_ids:
                continue
            seen_ids.add(h["article_id"])
            # Clip body to a short window around the phrase
            body = h.get("body_text") or ""
            pos = body.lower().find(phrase.lower())
            if pos >= 0:
                start = max(0, pos - 100)
                end = min(len(body), pos + len(phrase) + 200)
                snippet = body[start:end]
            else:
                snippet = body[:300]
            contexts.append({
                "phrase":     phrase,
                "article_id": h["article_id"],
                "snippet":    snippet,
                "paired_id":  h["paired_id"],
            })
    return contexts


# ── Glossary retrieval ────────────────────────────────────────────────────────

def select_glossary_subset(
    conn: sqlite3.Connection,
    text: str,
    source_lang: str,
    *,
    max_terms: int = 20,
) -> list[dict]:
    """Return glossary rows whose source-side term appears in `text`.

    The full glossary is ~1,000 EN↔DV pairs. Injecting all of them into
    every prompt would waste context and confuse the model. This function
    selects only terms that are actually in the input — relevant terms get
    boosted, irrelevant ones stay out.
    """
    if not text or not text.strip():
        return []
    rows = conn.execute(
        "SELECT en_term, dv_term, domain, freq FROM translation_glossary "
        "ORDER BY freq DESC LIMIT 500"
    ).fetchall()
    matches = [
        {"en_term": r[0], "dv_term": r[1], "domain": r[2], "freq": r[3]}
        for r in rows
        if r[0 if source_lang == "EN" else 1].lower() in text.lower()
    ]
    return matches[:max_terms]
