# SPDX-License-Identifier: Apache-2.0
"""Sentence-level translation memory — Phase B1 of the post-review
roadmap (research reviewer 2026-05-23: "the single biggest
improvement").

The existing translator's few-shot retrieval (`articles_fts`) ranks
WHOLE ARTICLES by topic similarity. The reviewer's critique: that's
too coarse. The LLM gets ~1200 chars of vaguely-related corpus prose
when what it actually needs is the analog of the sentence it's
translating right now.

This module builds the alternative: a sentence-level index. Each
paired EN+DV article is split into sentences; each sentence becomes
a row in the `sentence_pairs` table with article context + an FTS5
index for retrieval. At translation time, the input is split into
sentences, each one is queried against the FTS5 index, and the
top-K nearest sentences (with their paired-language counterparts
from the same article position) are shown to the LLM as
sentence-level exemplars.

This is B1a — the data pipeline + retrieval. Embeddings for fuzzy
similarity (B1b) come next.

DESIGN NOTES

  Sentence segmentation: regex on `.!?` followed by whitespace.
  EN protects common abbreviations (Dr., Mr., etc.). DV uses the
  same Western punctuation (the PO corpus does not use the Arabic
  comma U+060C as a sentence terminator).

  Alignment: we do NOT attempt strict sentence-by-sentence
  alignment between EN and DV articles. The PO writes its two
  language sides independently — they have different sentence
  counts and orderings. Instead, we store each sentence with its
  article context, and at retrieval time return the matching
  source-language sentence + the FULL paired-article body (capped).
  The LLM finds the alignment in context.

  Index size: 2,648 paired articles × ~10-20 sentences/article ≈
  30-50k rows. SQLite FTS5 handles this without issue.
"""
from __future__ import annotations

import logging
import re
import sqlite3
from typing import Optional

logger = logging.getLogger(__name__)


# ── Schema ──────────────────────────────────────────────────────────

_SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS sentence_pairs (
    id INTEGER PRIMARY KEY,
    article_id INTEGER NOT NULL,
    paired_article_id INTEGER,
    lang TEXT NOT NULL,                  -- 'EN' or 'DV'
    sentence_idx INTEGER NOT NULL,        -- position within source article
    text TEXT NOT NULL,
    text_len INTEGER NOT NULL,            -- character count (sanity / filter)
    embedding BLOB,                       -- Phase C1: multilingual embedding
    embedding_model TEXT,                 -- which model produced it (re-run safety)
    UNIQUE (article_id, lang, sentence_idx)
);
CREATE INDEX IF NOT EXISTS idx_sentpair_article ON sentence_pairs(article_id);
CREATE INDEX IF NOT EXISTS idx_sentpair_lang ON sentence_pairs(lang);
"""

# Schema migration for existing DBs (added 2026-05-23 for Phase C1).
# Idempotent — ALTER TABLE ... ADD COLUMN throws OperationalError if
# column exists; we swallow and continue.
_MIGRATIONS_SQL = (
    "ALTER TABLE sentence_pairs ADD COLUMN embedding BLOB",
    "ALTER TABLE sentence_pairs ADD COLUMN embedding_model TEXT",
)

# FTS5 virtual table for BM25 sentence retrieval. Indexes the
# sentence text only; metadata (article_id, lang) lives in the
# base sentence_pairs table and joins by row id.
_FTS_SQL = """
CREATE VIRTUAL TABLE IF NOT EXISTS sentence_pairs_fts USING fts5(
    text,
    content='sentence_pairs',
    content_rowid='id',
    tokenize='unicode61'
);

CREATE TRIGGER IF NOT EXISTS sentence_pairs_ai
AFTER INSERT ON sentence_pairs BEGIN
    INSERT INTO sentence_pairs_fts (rowid, text) VALUES (new.id, new.text);
END;
CREATE TRIGGER IF NOT EXISTS sentence_pairs_ad
AFTER DELETE ON sentence_pairs BEGIN
    INSERT INTO sentence_pairs_fts (sentence_pairs_fts, rowid, text)
    VALUES ('delete', old.id, old.text);
END;
"""


def init_sentence_memory(conn: sqlite3.Connection) -> bool:
    """Create the sentence_pairs table + FTS5 index + triggers.
    Idempotent. Returns True iff FTS5 is available (mirrors the
    articles_fts init pattern)."""
    conn.executescript(_SCHEMA_SQL)
    # Run additive column migrations (idempotent — skip if already
    # added). New columns added 2026-05-23 for Phase C1 hybrid
    # retrieval.
    for stmt in _MIGRATIONS_SQL:
        try:
            conn.execute(stmt)
        except sqlite3.OperationalError:
            pass  # column already exists
    try:
        conn.executescript(_FTS_SQL)
    except sqlite3.OperationalError as e:
        logger.info("sentence_pairs FTS5 unavailable (%s)", e)
        return False
    conn.commit()
    return True


# ── Sentence segmentation ──────────────────────────────────────────


# Common English abbreviations that end with a period and should
# NOT trigger a sentence split. Kept conservative — false positives
# (splitting where we shouldn't) hurt recall less than false
# negatives (failing to split when we should), since the FTS5
# index can still match on multi-sentence chunks.
_EN_ABBREVS = (
    "Dr.", "Mr.", "Mrs.", "Ms.", "St.", "Prof.", "Hon.", "Rev.",
    "Lt.", "Sgt.", "Col.", "Gen.", "Capt.", "U.S.", "U.K.", "U.N.",
    "etc.", "Vol.", "No.", "Inc.", "Ltd.", "Co.", "vs.", "i.e.", "e.g.",
)

# Sentinel char (unlikely to appear in source) we swap into
# abbreviation periods so the splitter ignores them. Restored
# after splitting.
_ABBREV_SENTINEL = "\x01"

# Minimum sentence length to keep. Very short fragments (< 15
# chars) are usually headers, captions, or noise — they pollute
# the BM25 index without giving useful retrieval signal.
_MIN_SENTENCE_CHARS = 15


def split_sentences(text: str, lang: str) -> list[str]:
    """Split `text` into sentences. Returns a list of trimmed
    sentence strings, filtered for minimum length.

    Approach: regex split on `.!?` followed by whitespace + an
    expected sentence start (uppercase letter for EN, any letter
    for DV since Thaana has no case). EN abbreviations are
    protected via sentinel substitution to avoid false splits."""
    if not text:
        return []
    work = text.strip()
    if not work:
        return []

    if lang == "EN":
        for abbrev in _EN_ABBREVS:
            work = work.replace(abbrev, abbrev.replace(".", _ABBREV_SENTINEL))
        # Split on sentence-ending punctuation followed by whitespace
        # and an uppercase letter or a Thaana letter (handles mixed-
        # script PO content).
        parts = re.split(
            r"(?<=[.!?])\s+(?=[A-Zހ-޿])",
            work,
        )
        parts = [p.replace(_ABBREV_SENTINEL, ".") for p in parts]
    else:
        # DV: Thaana has no case. Split on sentence-ending punctuation
        # followed by any whitespace. No abbreviation protection
        # because Thaana doesn't use period-terminated abbreviations
        # the same way.
        parts = re.split(r"(?<=[.!?])\s+", work)

    out: list[str] = []
    for p in parts:
        p = p.strip()
        if len(p) >= _MIN_SENTENCE_CHARS:
            out.append(p)
    return out


# ── Backfill ────────────────────────────────────────────────────────


def backfill_sentence_pairs(
    conn: sqlite3.Connection,
    *,
    progress_cb=None,
    batch_size: int = 200,
) -> dict:
    """Populate sentence_pairs from the existing `articles` table.

    Each paired article (one with paired_id NOT NULL) contributes
    its sentences from both sides. Idempotent: existing rows are
    skipped via the UNIQUE constraint on (article_id, lang,
    sentence_idx).

    Returns {articles_processed, sentences_inserted, articles_skipped}.

    Cost: zero LLM calls. Pure SQLite + Python text processing.
    Wall time for the full 2,648-pair corpus: ~30-60 seconds."""
    cur = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE paired_id IS NOT NULL "
        "AND body_text IS NOT NULL AND body_text != ''"
    )
    total = cur.fetchone()[0]
    if total == 0:
        return {"articles_processed": 0, "sentences_inserted": 0,
                "articles_skipped": 0}

    inserted = 0
    processed = 0
    skipped = 0
    offset = 0
    while True:
        rows = conn.execute(
            "SELECT id, language, paired_id, body_text "
            "FROM articles "
            "WHERE paired_id IS NOT NULL "
            "  AND body_text IS NOT NULL AND body_text != '' "
            "ORDER BY id LIMIT ? OFFSET ?",
            (batch_size, offset),
        ).fetchall()
        if not rows:
            break
        for r in rows:
            article_id = r["id"] if isinstance(r, sqlite3.Row) else r[0]
            lang = r["language"] if isinstance(r, sqlite3.Row) else r[1]
            paired_id = r["paired_id"] if isinstance(r, sqlite3.Row) else r[2]
            body = r["body_text"] if isinstance(r, sqlite3.Row) else r[3]
            sentences = split_sentences(body, lang)
            if not sentences:
                skipped += 1
                continue
            for idx, sent in enumerate(sentences):
                try:
                    conn.execute(
                        "INSERT INTO sentence_pairs "
                        "(article_id, paired_article_id, lang, "
                        " sentence_idx, text, text_len) "
                        "VALUES (?, ?, ?, ?, ?, ?)",
                        (article_id, paired_id, lang, idx,
                         sent, len(sent)),
                    )
                    inserted += 1
                except sqlite3.IntegrityError:
                    # UNIQUE constraint hit — row already exists
                    pass
            processed += 1
            if progress_cb is not None and processed % 50 == 0:
                progress_cb(processed, total, inserted)
        conn.commit()
        offset += batch_size

    if progress_cb is not None:
        progress_cb(processed, total, inserted)
    return {
        "articles_processed": processed,
        "sentences_inserted": inserted,
        "articles_skipped": skipped,
    }


# ── Multilingual embeddings — Phase C1 of post-review roadmap ──────
#
# BM25 retrieval works for English on English-rich queries, but it's
# weak for cross-lingual fuzzy matching and for shorter sentences
# where lexical overlap is thin. The reviewer's recommendation:
# hybrid BM25 + multilingual embeddings rerank.
#
# Model: sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2
#   - 50+ languages, INCLUDING approximate coverage of Thaana via
#     byte-pair tokenization (no explicit DV training but
#     character-level mapping degrades gracefully)
#   - 384-dim vectors (same size as existing LocalEmbedder, fits
#     SQLite BLOB column easily)
#   - Free, runs locally on CPU
#
# Storage: embeddings persisted as numpy float32 BLOB in the
# sentence_pairs.embedding column. Backfill is a one-time pass
# (~5-10 min for 111k sentences on CPU). The model_id is stored
# per row so a re-backfill with a different model doesn't pollute
# cosine comparisons.

_MULTILINGUAL_MODEL = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

_embedder_cache: dict = {}  # module-level cache; load once, reuse


def _get_embedder(model_name: str = _MULTILINGUAL_MODEL):
    """Lazy-load the multilingual sentence-transformer. Returns the
    model or None if sentence-transformers is unavailable (in which
    case hybrid retrieval falls back to pure BM25)."""
    if model_name in _embedder_cache:
        return _embedder_cache[model_name]
    try:
        from sentence_transformers import SentenceTransformer
    except ImportError:
        logger.info("sentence-transformers not installed; "
                     "hybrid retrieval disabled, falling back to BM25")
        _embedder_cache[model_name] = None
        return None
    try:
        model = SentenceTransformer(model_name)
        _embedder_cache[model_name] = model
        return model
    except Exception as e:
        logger.warning("failed to load %s (%s); hybrid disabled",
                        model_name, e)
        _embedder_cache[model_name] = None
        return None


def _encode(texts: list[str], model_name: str = _MULTILINGUAL_MODEL):
    """Embed a list of texts. Returns numpy array (n, dim) or None
    if the embedder is unavailable. CPU-bound; batches the call."""
    model = _get_embedder(model_name)
    if model is None:
        return None
    try:
        return model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,   # cosine via dot product
            show_progress_bar=False,
        )
    except Exception as e:
        logger.warning("embedding failed (%s); falling back to BM25", e)
        return None


def backfill_sentence_embeddings(
    conn: sqlite3.Connection,
    *,
    batch_size: int = 256,
    progress_cb=None,
    model_name: str = _MULTILINGUAL_MODEL,
) -> dict:
    """Compute + persist embeddings for sentence_pairs rows that
    don't have one (or have one from a different model).

    Returns {processed, embedded, skipped_existing, skipped, errors}. Idempotent.

    Cost: zero API spend (local model). Wall time: ~5-10 min for
    111k sentences on a typical CPU; ~1-2 min on M-series Mac."""
    model = _get_embedder(model_name)
    if model is None:
        return {"processed": 0, "embedded": 0, "skipped_existing": 0, "skipped": 0, "errors": 0,
                "note": "sentence-transformers unavailable"}

    import numpy as np

    total = conn.execute(
        "SELECT COUNT(*) FROM sentence_pairs "
        "WHERE embedding IS NULL OR embedding_model != ?",
        (model_name,),
    ).fetchone()[0]
    if total == 0:
        # Calculate skipped_existing
        all_sents = conn.execute("SELECT COUNT(*) FROM sentence_pairs").fetchone()[0]
        return {"processed": 0, "embedded": 0, "skipped_existing": all_sents, "skipped": all_sents, "errors": 0,
                "note": "all sentences already have current embeddings"}

    processed = 0
    errors = 0
    while True:
        rows = conn.execute(
            "SELECT id, text FROM sentence_pairs "
            "WHERE embedding IS NULL OR embedding_model != ? "
            "ORDER BY id LIMIT ? OFFSET 0",
            (model_name, batch_size),
        ).fetchall()
        if not rows:
            break
        ids = [r[0] for r in rows]
        texts = [r[1] for r in rows]
        try:
            vecs = model.encode(
                texts, convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
        except Exception as e:
            logger.warning("encode batch failed: %s", e)
            errors += len(rows)
            break  # Break to avoid infinite loop on persistent encoding errors
        # Persist
        for sid, v in zip(ids, vecs):
            try:
                conn.execute(
                    "UPDATE sentence_pairs "
                    "SET embedding = ?, embedding_model = ? WHERE id = ?",
                    (np.asarray(v, dtype=np.float32).tobytes(),
                     model_name, sid),
                )
                processed += 1
            except Exception as e:
                logger.warning("update id=%d failed: %s", sid, e)
                errors += 1
        conn.commit()
        if progress_cb is not None:
            progress_cb(processed, total)
    if progress_cb is not None:
        progress_cb(processed, total)

    all_sents = conn.execute("SELECT COUNT(*) FROM sentence_pairs").fetchone()[0]
    skipped_existing = all_sents - processed

    return {"processed": processed, "embedded": processed,
            "skipped_existing": skipped_existing, "skipped": skipped_existing,
            "errors": errors}


# ── Retrieval ───────────────────────────────────────────────────────


# FTS5 sanitize for sentence-level retrieval.
#
# Unlike articles_fts (which uses AND-of-phrases because article
# bodies are 1000+ chars and the conjunction is satisfiable), this
# operates on ~50-200-char sentences where the default AND would
# require every query token to appear — too strict. We use OR
# semantics so the BM25 ranker can pick the closest sentences
# even if only a few content words overlap.
#
# Stopwords get dropped before the OR-join to avoid noise — "the
# president" → ["president"] (no "the"), then `"president"` in the
# FTS5 query.
_TOKEN_RE = re.compile(r"[A-Za-z0-9'ހ-޿]+")

# Lowercase English stopwords. Thaana has no equivalent
# closed-class noise list; we don't filter DV tokens.
_STOPWORDS_EN = frozenset({
    "the", "a", "an", "and", "or", "of", "in", "on", "at", "to",
    "for", "with", "by", "from", "as", "is", "was", "were", "are",
    "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "this", "that", "these", "those", "it", "its", "he", "she",
    "they", "their", "his", "her", "i", "we", "you", "your", "our",
    "but", "not", "no", "if", "then", "so", "than", "also", "into",
    "about", "over", "after", "before", "during", "while", "made",
    "make", "said", "say", "says", "today", "tonight",
})


def _fts_sanitize_sentence(query: str, *, lang: str) -> str:
    """Build an FTS5 MATCH expression with OR semantics for
    sentence-level retrieval. Drops EN stopwords. Returns the
    empty-quoted form `""` if no usable tokens remain (caller
    skips the query)."""
    tokens = _TOKEN_RE.findall(query)
    if not tokens:
        return '""'
    if lang == "EN":
        tokens = [t for t in tokens if t.lower() not in _STOPWORDS_EN]
    # Cap at top-12 content tokens to avoid pathological query
    # bloat on long input sentences.
    tokens = tokens[:12]
    if not tokens:
        return '""'
    return " OR ".join(f'"{t}"' for t in tokens)


def select_sentence_memory_hybrid(
    conn: sqlite3.Connection,
    text: str,
    *,
    source_lang: str,
    k: int = 5,
    exclude_article_ids: Optional[set] = None,
    bm25_candidates: int = 30,
) -> list[dict]:
    """Hybrid retrieval — BM25 finds the candidate pool, multilingual
    embeddings rerank by semantic similarity.

    Phase C1 of post-review roadmap. Reviewer 2026-05-23 flagged
    pure BM25 as "lexical-dependent, misses semantically similar
    but differently worded articles". Hybrid is the standard fix:

      1. For each input sentence, BM25 finds top-30 candidate
         sentences (wide net — semantically-similar candidates may
         not have high lexical overlap).
      2. Embed the input sentence ONCE per input sentence with the
         multilingual model (cosine-normalized).
      3. For each candidate, fetch its precomputed embedding and
         compute cosine via dot product.
      4. Sort by cosine score, return top-k.

    Falls back to pure BM25 (select_sentence_memory) if:
      - sentence-transformers not installed
      - the embedder fails to load
      - no candidates have stored embeddings (backfill not run)

    Same return shape as select_sentence_memory."""
    input_sentences = split_sentences(text, source_lang)
    if not input_sentences:
        return []

    # If embeddings unavailable, fall back to BM25-only path
    model = _get_embedder()
    if model is None:
        return select_sentence_memory(
            conn, text, source_lang=source_lang, k=k,
            exclude_article_ids=exclude_article_ids,
        )

    import numpy as np

    out: list[dict] = []
    seen: set[int] = set()
    excl = set(exclude_article_ids or [])

    # Embed all input sentences in one batch
    try:
        input_vecs = model.encode(
            input_sentences, convert_to_numpy=True,
            normalize_embeddings=True, show_progress_bar=False,
        )
    except Exception as e:
        logger.warning("input encode failed (%s); falling back to BM25", e)
        return select_sentence_memory(
            conn, text, source_lang=source_lang, k=k,
            exclude_article_ids=exclude_article_ids,
        )

    for input_sent, input_vec in zip(input_sentences, input_vecs):
        if len(out) >= k:
            break
        sanitized = _fts_sanitize_sentence(input_sent, lang=source_lang)
        if sanitized == '""':
            continue
        try:
            rows = conn.execute(
                """SELECT sp.id, sp.article_id, sp.paired_article_id,
                          sp.text AS source_text, sp.embedding
                   FROM sentence_pairs_fts f
                   JOIN sentence_pairs sp ON sp.id = f.rowid
                   WHERE sentence_pairs_fts MATCH ?
                     AND sp.lang = ?
                   ORDER BY bm25(sentence_pairs_fts)
                   LIMIT ?""",
                (sanitized, source_lang, bm25_candidates),
            ).fetchall()
        except sqlite3.OperationalError as e:
            logger.debug("hybrid FTS5 error on %r: %s",
                          input_sent[:60], e)
            continue
        if not rows:
            continue

        # Rerank candidates by cosine similarity. Skip ones missing
        # embeddings (degraded case — they just lose the rerank
        # boost but stay in the candidate pool ranked by BM25).
        candidates = []
        for row in rows:
            sid, src_aid, paired_aid, source_text, emb_blob = row
            if src_aid in excl or paired_aid in excl or src_aid in seen:
                continue
            if emb_blob:
                cand_vec = np.frombuffer(emb_blob, dtype=np.float32)
                cos = float(np.dot(input_vec, cand_vec))
            else:
                # No embedding — neutral score (still kept, just not boosted)
                cos = 0.0
            candidates.append((cos, sid, src_aid, paired_aid, source_text))

        # Sort by cosine descending, take the top until we fill k
        candidates.sort(key=lambda c: -c[0])
        for cos, sid, src_aid, paired_aid, source_text in candidates:
            if len(out) >= k:
                break
            paired_body_row = conn.execute(
                "SELECT body_text FROM articles WHERE id = ?",
                (paired_aid,),
            ).fetchone()
            if paired_body_row is None or not paired_body_row[0]:
                continue
            out.append({
                "input_sentence":    input_sent,
                "source_text":       source_text,
                "source_article_id": src_aid,
                "paired_article_id": paired_aid,
                "paired_body":       paired_body_row[0][:600],
                "cosine":            cos,
            })
            seen.add(src_aid)

    return out


def select_sentence_memory(
    conn: sqlite3.Connection,
    text: str,
    *,
    source_lang: str,
    k: int = 5,
    exclude_article_ids: Optional[set] = None,
) -> list[dict]:
    """Retrieve top-k sentence-level matches from the corpus.

    For each input sentence (after segmenting `text`), the FTS5
    sentence index returns the closest matches in the SOURCE
    language. For each match, we also return the paired-article
    body so the LLM has cross-lingual context.

    Returns a list of dicts (capped at k total):
      {
        "input_sentence":   str,    # which input sentence triggered this match
        "source_text":      str,    # the matching sentence from the corpus
        "source_article_id": int,
        "paired_article_id": int,
        "paired_body":      str,    # first ~600 chars of the paired article
      }

    exclude_article_ids: skip matches from these article IDs (and
    their paired counterparts). Used by the eval to prevent
    ground-truth leak."""
    input_sentences = split_sentences(text, source_lang)
    if not input_sentences:
        return []

    out: list[dict] = []
    seen: set[int] = set()  # dedupe by source_article_id
    excl = set(exclude_article_ids or [])

    for input_sent in input_sentences:
        if len(out) >= k:
            break
        sanitized = _fts_sanitize_sentence(input_sent, lang=source_lang)
        if sanitized == '""':
            continue
        try:
            rows = conn.execute(
                """SELECT sp.article_id, sp.paired_article_id,
                          sp.text AS source_text,
                          bm25(sentence_pairs_fts) AS rank
                   FROM sentence_pairs_fts f
                   JOIN sentence_pairs sp ON sp.id = f.rowid
                   WHERE sentence_pairs_fts MATCH ?
                     AND sp.lang = ?
                   ORDER BY rank
                   LIMIT 5""",
                (sanitized, source_lang),
            ).fetchall()
        except sqlite3.OperationalError as e:
            logger.debug("sentence_memory FTS5 error on %r: %s",
                          input_sent[:60], e)
            continue
        for row in rows:
            src_aid = row[0] if not isinstance(row, sqlite3.Row) else row["article_id"]
            paired_aid = row[1] if not isinstance(row, sqlite3.Row) else row["paired_article_id"]
            source_text = row[2] if not isinstance(row, sqlite3.Row) else row["source_text"]
            if src_aid in excl or paired_aid in excl:
                continue
            if src_aid in seen:
                continue
            paired_body_row = conn.execute(
                "SELECT body_text FROM articles WHERE id = ?",
                (paired_aid,),
            ).fetchone()
            if paired_body_row is None or not paired_body_row[0]:
                continue
            paired_body = paired_body_row[0][:600]
            out.append({
                "input_sentence":    input_sent,
                "source_text":       source_text,
                "source_article_id": src_aid,
                "paired_article_id": paired_aid,
                "paired_body":       paired_body,
            })
            seen.add(src_aid)
            if len(out) >= k:
                break
    return out
