#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Import the articles corpus from an existing kahzaabu database into moonlight.

Usage::

    python scripts/import_from_kahzaabu.py --source /path/to/kahzaabu.db

This script copies only the data that moonlight needs for translation:

  articles          — paired EN-DV press release bodies
  translation_glossary — precomputed bilingual term pairs
  sentence_pairs    — pre-split sentence pairs (optional; saves re-splitting)

It does NOT copy claims, fact-checks, scrape runs, manifesto promises, web
users, or any other application-layer table.  moonlight is a translation
engine; it has no business logic outside of that scope.

After importing you can immediately run::

    moonlight translate "ދިވެހިރާއްޖެ"

and the translator will have a full corpus to retrieve exemplars from.

Design notes
------------
The import is a one-way copy.  No foreign-key constraints link moonlight to
the source database; kahzaabu can be updated, migrated, or deleted without
affecting moonlight.  Re-run this script to refresh moonlight's corpus from
a newer kahzaabu snapshot.

Rows are upserted (INSERT OR REPLACE for articles; INSERT OR IGNORE for
sentence_pairs) so re-running is safe.
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path


def _open_source(path: Path) -> sqlite3.Connection:
    if not path.exists():
        sys.exit(f"Source database not found: {path}")
    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row
    return conn


def _import_articles(src: sqlite3.Connection, dst: sqlite3.Connection) -> int:
    rows = src.execute(
        """SELECT id, language, paired_id, category, category_id, title,
                  body_text, body_html, reference, published_date, image_urls,
                  raw_page_html, scraped_at, content_hash
           FROM articles
           WHERE body_text IS NOT NULL AND body_text != ''
           ORDER BY id"""
    ).fetchall()
    n = 0
    for row in rows:
        dst.execute(
            """INSERT INTO articles
               (id, language, paired_id, category, category_id, title,
                body_text, body_html, reference, published_date, image_urls,
                raw_page_html, scraped_at, content_hash)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
               ON CONFLICT(id, language) DO UPDATE SET
                 paired_id=excluded.paired_id,
                 category=excluded.category,
                 title=excluded.title,
                 body_text=excluded.body_text,
                 body_html=excluded.body_html,
                 published_date=excluded.published_date,
                 image_urls=excluded.image_urls,
                 raw_page_html=excluded.raw_page_html,
                 scraped_at=excluded.scraped_at,
                 content_hash=excluded.content_hash""",
            tuple(row),
        )
        n += 1
        if n % 500 == 0:
            print(f"  articles: {n} rows …", end="\r")
    dst.commit()
    return n


def _import_glossary(src: sqlite3.Connection, dst: sqlite3.Connection) -> int:
    # Check table exists in source (older kahzaabu DBs may not have it).
    tables = {
        r[0]
        for r in src.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "translation_glossary" not in tables:
        print("  translation_glossary: not found in source, skipping")
        return 0
    rows = src.execute(
        """SELECT en_term, dv_term, domain, freq, confidence,
                  sample_en_ids, extracted_at, extracted_by
           FROM translation_glossary"""
    ).fetchall()
    n = 0
    for row in rows:
        dst.execute(
            """INSERT OR IGNORE INTO translation_glossary
               (en_term, dv_term, domain, freq, confidence,
                sample_en_ids, extracted_at, extracted_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            tuple(row),
        )
        n += 1
    dst.commit()
    return n


def _import_sentence_pairs(src: sqlite3.Connection, dst: sqlite3.Connection) -> int:
    tables = {
        r[0]
        for r in src.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        ).fetchall()
    }
    if "sentence_pairs" not in tables:
        print("  sentence_pairs: not found in source, skipping")
        return 0

    cursor = src.execute("PRAGMA table_info(sentence_pairs)")
    columns = {row[1] for row in cursor.fetchall()}

    n = 0
    # Source has the new schema columns
    if "text" in columns and "sentence_idx" in columns:
        rows = src.execute(
            "SELECT article_id, paired_article_id, lang, sentence_idx, text, text_len, embedding, embedding_model "
            "FROM sentence_pairs"
        ).fetchall()
        for row in rows:
            dst.execute(
                """INSERT OR IGNORE INTO sentence_pairs
                   (article_id, paired_article_id, lang, sentence_idx, text, text_len, embedding, embedding_model)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                tuple(row),
            )
            n += 1
            if n % 5000 == 0:
                print(f"  sentence_pairs: {n} rows …", end="\r")
        dst.commit()
        return n

    # Source has the old schema columns (sentence, position)
    rows = src.execute(
        "SELECT article_id, lang, sentence, position FROM sentence_pairs"
    ).fetchall()
    for row in rows:
        article_id, lang, sentence, position = row
        # Fetch paired_id from dest
        paired_row = dst.execute(
            "SELECT paired_id FROM articles WHERE id = ? AND language = ?",
            (article_id, lang)
        ).fetchone()
        paired_id = paired_row[0] if paired_row else None

        dst.execute(
            """INSERT OR IGNORE INTO sentence_pairs
               (article_id, paired_article_id, lang, sentence_idx, text, text_len)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (article_id, paired_id, lang, position, sentence, len(sentence)),
        )
        n += 1
        if n % 5000 == 0:
            print(f"  sentence_pairs: {n} rows …", end="\r")
    dst.commit()
    return n


def _backfill_fts(dst: sqlite3.Connection) -> None:
    """Populate FTS5 indexes from the imported data."""
    print("  FTS5 backfill …")
    try:
        dst.execute("INSERT INTO articles_fts(articles_fts) VALUES('rebuild')")
        dst.commit()
        print("  articles_fts: rebuilt")
    except sqlite3.OperationalError as e:
        print(f"  articles_fts: {e}")

    try:
        dst.execute("INSERT INTO sentence_pairs_fts(sentence_pairs_fts) VALUES('rebuild')")
        dst.commit()
        print("  sentence_pairs_fts: rebuilt")
    except sqlite3.OperationalError as e:
        print(f"  sentence_pairs_fts: {e}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import articles corpus from kahzaabu into moonlight."
    )
    parser.add_argument(
        "--source", required=True, metavar="PATH",
        help="Path to the source kahzaabu.db (or moonlight.db with existing data).",
    )
    parser.add_argument(
        "--dest", default=None, metavar="PATH",
        help="Path to the destination moonlight.db.  Defaults to data/moonlight.db.",
    )
    parser.add_argument(
        "--no-glossary", action="store_true",
        help="Skip importing translation_glossary.",
    )
    parser.add_argument(
        "--no-sentence-pairs", action="store_true",
        help="Skip importing sentence_pairs.",
    )
    args = parser.parse_args()

    src_path = Path(args.source).expanduser().resolve()
    src = _open_source(src_path)

    # Open destination via moonlight's get_connection so schema is initialised.
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from moonlight.db import get_connection
    dest_path = Path(args.dest).expanduser().resolve() if args.dest else None
    dst = get_connection(dest_path)

    print(f"Source: {src_path}")
    print(f"Dest:   {dst.execute('PRAGMA database_list').fetchone()[2]}")
    print()

    n_articles = _import_articles(src, dst)
    print(f"  articles:             {n_articles:>7} rows")

    if not args.no_glossary:
        n_glossary = _import_glossary(src, dst)
        print(f"  translation_glossary: {n_glossary:>7} rows")

    if not args.no_sentence_pairs:
        n_sp = _import_sentence_pairs(src, dst)
        print(f"  sentence_pairs:       {n_sp:>7} rows")

    print()
    _backfill_fts(dst)

    src.close()
    dst.close()
    print("\nDone.")


if __name__ == "__main__":
    main()
