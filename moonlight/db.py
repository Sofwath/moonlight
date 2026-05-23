# SPDX-License-Identifier: Apache-2.0
"""SQLite schema and connection utilities for moonlight.

The database holds only what the translation engine needs:

  articles          — the paired EN-DV corpus (source of truth for retrieval)
  articles_fts      — FTS5 BM25 index over article bodies
  translation_glossary — extracted EN↔DV term pairs mined from the corpus
  translation_runs  — every translation call (audit trail + LRU cache)
  sentence_pairs    — corpus split into aligned sentence pairs
  sentence_pairs_fts — FTS5 index over sentence_pairs
  place_names       — Maldivian island/atoll/city reference DB (GeoNames)

Nothing else. Claims, fact-checks, manifesto promises, web users, scrape
runs — those belong to the application layer, not the translation engine.
"""
from __future__ import annotations

from dataclasses import dataclass
import json as _json
from pathlib import Path
import sqlite3
from typing import Optional

# Default DB location when the user doesn't specify one
DEFAULT_DB_PATH = Path(__file__).resolve().parents[1] / "data" / "moonlight.db"

# ── Schema ────────────────────────────────────────────────────────────────────

ARTICLES_SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id              INTEGER NOT NULL,
    language        TEXT    NOT NULL CHECK(language IN ('EN', 'DV')),
    paired_id       INTEGER,
    category        TEXT,
    category_id     INTEGER,
    title           TEXT,
    body_text       TEXT,
    body_html       TEXT,
    reference       TEXT,
    published_date  TEXT,
    image_urls      TEXT,     -- JSON array of URL strings
    raw_page_html   TEXT,
    scraped_at      TEXT,
    content_hash    TEXT,
    PRIMARY KEY (id, language)
);
CREATE INDEX IF NOT EXISTS idx_articles_paired   ON articles(paired_id, language);
CREATE INDEX IF NOT EXISTS idx_articles_lang     ON articles(language, published_date);
CREATE INDEX IF NOT EXISTS idx_articles_category ON articles(category, language);
"""

GLOSSARY_SCHEMA = """
CREATE TABLE IF NOT EXISTS translation_glossary (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    en_term       TEXT NOT NULL,
    dv_term       TEXT NOT NULL,
    domain        TEXT,
    freq          INTEGER NOT NULL,
    confidence    REAL,
    sample_en_ids TEXT,   -- JSON array of article ids where this pair was found
    extracted_at  TEXT NOT NULL,
    extracted_by  TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_glossary_en ON translation_glossary(en_term);
CREATE INDEX IF NOT EXISTS idx_glossary_dv ON translation_glossary(dv_term);
"""

TRANSLATION_RUNS_SCHEMA = """
CREATE TABLE IF NOT EXISTS translation_runs (
    id                  INTEGER PRIMARY KEY AUTOINCREMENT,
    source_lang         TEXT NOT NULL,
    target_lang         TEXT NOT NULL,
    input_text          TEXT NOT NULL,
    output_text         TEXT NOT NULL,
    mode                TEXT NOT NULL DEFAULT 'faithful',
    exemplar_ids        TEXT,    -- JSON array of article ids used as few-shot
    phrase_context_ids  TEXT,    -- JSON array
    glossary_terms_used INTEGER,
    n_candidates        INTEGER DEFAULT 1,
    model               TEXT NOT NULL,
    cost_usd            REAL,
    tokens_in           INTEGER,
    tokens_out          INTEGER,
    created_at          TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_translation_runs_cache
    ON translation_runs(target_lang, created_at);
"""

SENTENCE_PAIRS_SCHEMA = """
CREATE TABLE IF NOT EXISTS sentence_pairs (
    id                  INTEGER PRIMARY KEY,
    article_id          INTEGER NOT NULL,
    paired_article_id   INTEGER,
    lang                TEXT NOT NULL,
    sentence_idx        INTEGER NOT NULL,
    text                TEXT NOT NULL,
    text_len            INTEGER NOT NULL,
    embedding           BLOB,
    embedding_model     TEXT,
    UNIQUE(article_id, lang, sentence_idx)
);
CREATE INDEX IF NOT EXISTS idx_sentpair_article ON sentence_pairs(article_id);
CREATE INDEX IF NOT EXISTS idx_sentpair_lang ON sentence_pairs(lang);
"""

PLACE_NAMES_SCHEMA = """
CREATE TABLE IF NOT EXISTS place_names (
    geonameid    INTEGER PRIMARY KEY,
    en_name      TEXT NOT NULL,
    dv_thaana    TEXT,
    dv_latin     TEXT,
    en_name_po   TEXT,
    feature_code TEXT,
    atoll_code   TEXT,
    latitude     REAL,
    longitude    REAL,
    source       TEXT NOT NULL DEFAULT 'geonames'
);
CREATE INDEX IF NOT EXISTS idx_place_names_thaana  ON place_names(dv_thaana);
CREATE INDEX IF NOT EXISTS idx_place_names_en      ON place_names(en_name);
CREATE INDEX IF NOT EXISTS idx_place_names_feature ON place_names(feature_code);
"""

FTS5_ARTICLES_SCHEMA = """
CREATE VIRTUAL TABLE IF NOT EXISTS articles_fts USING fts5(
    article_id UNINDEXED,
    language UNINDEXED,
    title,
    body,
    tokenize='unicode61'
);

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

FTS5_SENTENCES_SCHEMA = """
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


def init_db(conn: sqlite3.Connection) -> None:
    """Create all moonlight tables. Idempotent — safe to call repeatedly."""
    conn.executescript(ARTICLES_SCHEMA)
    conn.executescript(GLOSSARY_SCHEMA)
    conn.executescript(TRANSLATION_RUNS_SCHEMA)
    conn.executescript(SENTENCE_PAIRS_SCHEMA)
    conn.executescript(PLACE_NAMES_SCHEMA)

    # FTS5 requires a CREATE VIRTUAL TABLE statement; executescript handles it
    # but we check existence first to avoid re-running FTS5 tokenizer init.
    existing = {
        row[0]
        for row in conn.execute(
            "SELECT name FROM sqlite_master WHERE type IN ('table','shadow')"
        ).fetchall()
    }
    if "articles_fts" not in existing:
        conn.executescript(FTS5_ARTICLES_SCHEMA)
    if "sentence_pairs_fts" not in existing:
        conn.executescript(FTS5_SENTENCES_SCHEMA)

    conn.commit()


def get_connection(path: Optional[Path] = None) -> sqlite3.Connection:
    """Open (creating if necessary) the moonlight SQLite DB.

    Row factory is set to sqlite3.Row so callers can access columns by name.
    WAL mode is enabled for concurrent read/write performance.
    """
    db_path = Path(path) if path else DEFAULT_DB_PATH
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    init_db(conn)
    return conn


# ── Article dataclass ─────────────────────────────────────────────────────────




@dataclass
class Article:
    id: int
    language: str
    paired_id: Optional[int]
    category: str
    category_id: Optional[int]
    title: str
    body_text: str
    body_html: str
    reference: str
    published_date: str
    image_urls: list
    raw_page_html: str
    scraped_at: Optional[str] = None
    content_hash: Optional[str] = None


def insert_article(conn: sqlite3.Connection, article: Article) -> None:
    """Upsert an article into the corpus. Existing rows are replaced."""
    conn.execute(
        """INSERT INTO articles
           (id, language, paired_id, category, category_id, title,
            body_text, body_html, reference, published_date, image_urls,
            raw_page_html, scraped_at, content_hash)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
           ON CONFLICT(id, language) DO UPDATE SET
             paired_id=excluded.paired_id, category=excluded.category,
             title=excluded.title, body_text=excluded.body_text,
             body_html=excluded.body_html, published_date=excluded.published_date,
             image_urls=excluded.image_urls, raw_page_html=excluded.raw_page_html,
             scraped_at=excluded.scraped_at, content_hash=excluded.content_hash""",
        (
            article.id, article.language, article.paired_id, article.category,
            article.category_id, article.title, article.body_text, article.body_html,
            article.reference, article.published_date,
            _json.dumps(article.image_urls or []), article.raw_page_html,
            article.scraped_at, article.content_hash,
        ),
    )
    conn.commit()


def count_pairs(conn: sqlite3.Connection) -> int:
    """Number of paired EN-DV article pairs in the corpus."""
    return conn.execute(
        """SELECT COUNT(*) FROM articles a
           JOIN articles b ON a.paired_id = b.id
           WHERE a.language='EN' AND b.language='DV'"""
    ).fetchone()[0]


def corpus_stats(conn: sqlite3.Connection) -> dict:
    """Summary statistics about the loaded corpus."""
    en_n = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE language='EN'"
    ).fetchone()[0]
    dv_n = conn.execute(
        "SELECT COUNT(*) FROM articles WHERE language='DV'"
    ).fetchone()[0]
    paired_n = count_pairs(conn)
    date_range = conn.execute(
        "SELECT MIN(published_date), MAX(published_date) FROM articles"
    ).fetchone()
    cats = conn.execute(
        "SELECT category, COUNT(*) FROM articles WHERE language='EN' "
        "GROUP BY category ORDER BY COUNT(*) DESC"
    ).fetchall()
    return {
        "en_articles": en_n,
        "dv_articles": dv_n,
        "paired_articles": paired_n,
        "date_min": date_range[0],
        "date_max": date_range[1],
        "categories": {row[0]: row[1] for row in cats},
    }
