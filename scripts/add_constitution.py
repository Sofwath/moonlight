#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Import the Maldives Constitution (EN + DV) into moonlight.db.

The Constitution is available as a PDF from the Great People's Majlis website.
This script expects you to supply plain-text files (one per language) that you
have extracted from the PDF.  Each article is treated as a document.

Expected input format (--en-file / --dv-file): plain text, one article per
"Chapter" / "Article" boundary.  The script segments by "Article N." headings.
If neither file is provided the script prints usage instructions.

Article IDs use the 2,000,000+ namespace to avoid collision with the
Presidency Office corpus (IDs ~28,000–40,000).

Usage::

    # 1. Extract text from the PDFs (requires pdfminer or pdftotext):
    pdftotext constitution_en.pdf constitution_en.txt
    pdftotext constitution_dv.pdf constitution_dv.txt

    # 2. Import
    python scripts/add_constitution.py \\
        --en-file constitution_en.txt \\
        --dv-file constitution_dv.txt

After running, rebuild embeddings::

    moonlight build-embeddings

Pairing
-------
The EN article N is paired with the DV article N by ordinal position (same
article number).  IDs are assigned as:
    EN article N → id = 2_000_000 + N
    DV article N → id = 2_100_000 + N
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moonlight.db import get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("add_constitution")

# ID namespace: 2,000,000+ to avoid PO article collisions
EN_ID_BASE = 2_000_000
DV_ID_BASE = 2_100_000

PUBLISHED_DATE = "2008-08-07"  # Constitution ratified 7 August 2008


# ── Text segmentation ─────────────────────────────────────────────────────────

def _split_articles(text: str) -> list[tuple[int, str, str]]:
    """Split constitution text into (article_num, title, body) tuples.

    Handles patterns like:
        Article 1.
        Article 1
        ARTICLE 1.
    Falls back to chapter-level splitting if no article headings found.
    """
    # Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")

    pattern = re.compile(
        r"(?:^|\n)((?:Article|ARTICLE)\s+(\d+)\.?[^\n]*)\n",
        re.MULTILINE,
    )
    matches = list(pattern.finditer(text))

    if len(matches) < 5:
        # Try chapter-level splitting
        pattern = re.compile(
            r"(?:^|\n)((?:Chapter|CHAPTER)\s+([IVXLC\d]+)\.?[^\n]*)\n",
            re.MULTILINE,
        )
        matches = list(pattern.finditer(text))

    if not matches:
        # No structure found — treat entire text as one article
        logger.warning("No article/chapter headings found; treating as single document")
        return [(1, "Constitution", text.strip())]

    articles: list[tuple[int, str, str]] = []
    for i, m in enumerate(matches):
        heading = m.group(1).strip()
        # Extract article number
        num_str = m.group(2).strip()
        try:
            # Handle Roman numerals for chapter splits
            num = _roman_to_int(num_str) if not num_str.isdigit() else int(num_str)
        except ValueError:
            num = i + 1

        body_start = m.end()
        body_end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        body = text[body_start:body_end].strip()

        articles.append((num, heading, body))

    return articles


def _roman_to_int(s: str) -> int:
    vals = {"I": 1, "V": 5, "X": 10, "L": 50, "C": 100}
    total = 0
    prev = 0
    for ch in reversed(s.upper()):
        v = vals.get(ch, 0)
        total += v if v >= prev else -v
        prev = v
    return total


# ── DB helpers ────────────────────────────────────────────────────────────────

_UPSERT = """
INSERT INTO articles
    (id, language, paired_id, category, category_id, title,
     body_text, body_html, reference, published_date, image_urls,
     raw_page_html, scraped_at, content_hash)
VALUES
    (:id, :language, :paired_id, :category, :category_id, :title,
     :body_text, :body_html, :reference, :published_date, :image_urls,
     :raw_page_html, :scraped_at, :content_hash)
ON CONFLICT(id, language) DO UPDATE SET
    paired_id      = excluded.paired_id,
    title          = excluded.title,
    body_text      = excluded.body_text,
    body_html      = excluded.body_html,
    published_date = excluded.published_date,
    scraped_at     = excluded.scraped_at,
    content_hash   = excluded.content_hash
"""


def _save(conn, article_id: int, lang: str, paired_id: int, title: str, body: str) -> None:
    now = datetime.utcnow().isoformat()
    content_hash = hashlib.sha256(body.encode()).hexdigest()[:16]
    conn.execute(_UPSERT, {
        "id": article_id,
        "language": lang,
        "paired_id": paired_id,
        "category": "constitution",
        "category_id": None,
        "title": title,
        "body_text": body,
        "body_html": "",
        "reference": None,
        "published_date": PUBLISHED_DATE,
        "image_urls": "[]",
        "raw_page_html": "",
        "scraped_at": now,
        "content_hash": content_hash,
    })


def _save_sentence_pairs(conn, article_id: int, paired_id: int, lang: str, body: str) -> int:
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    for idx, para in enumerate(paragraphs):
        conn.execute(
            """INSERT OR IGNORE INTO sentence_pairs
               (article_id, paired_article_id, lang, sentence_idx, text, text_len)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (article_id, paired_id, lang, idx, para, len(para)),
        )
    return len(paragraphs)


def _rebuild_fts(conn) -> None:
    for table in ("articles_fts", "sentence_pairs_fts"):
        try:
            conn.execute(f"INSERT INTO {table}({table}) VALUES('rebuild')")
            conn.commit()
            logger.info(f"  {table}: rebuilt")
        except Exception as e:
            logger.warning(f"  {table} rebuild failed: {e}")


# ── Main ──────────────────────────────────────────────────────────────────────

def _print_instructions() -> None:
    print(__doc__)
    print()
    print("Steps to get the Constitution text files:")
    print()
    print("  English:")
    print("    curl -L 'https://majlis.gov.mv/en/content/constitutionmaldives' -o constitution_en.pdf")
    print("    pdftotext constitution_en.pdf constitution_en.txt")
    print()
    print("  Dhivehi:")
    print("    # Download the DV PDF from the same page, then:")
    print("    pdftotext constitution_dv.pdf constitution_dv.txt")
    print()
    print("Then run:")
    print("    python scripts/add_constitution.py \\")
    print("        --en-file constitution_en.txt \\")
    print("        --dv-file constitution_dv.txt")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Import Maldives Constitution into moonlight.db"
    )
    parser.add_argument("--en-file", metavar="PATH",
                        help="Plain-text English constitution")
    parser.add_argument("--dv-file", metavar="PATH",
                        help="Plain-text Dhivehi constitution")
    parser.add_argument("--db", default=None, metavar="PATH",
                        help="Path to moonlight.db (defaults to data/moonlight.db)")
    parser.add_argument("--no-fts-rebuild", action="store_true",
                        help="Skip FTS5 rebuild at the end")
    args = parser.parse_args()

    if not args.en_file and not args.dv_file:
        _print_instructions()
        sys.exit(0)

    en_articles: list[tuple[int, str, str]] = []
    dv_articles: list[tuple[int, str, str]] = []

    if args.en_file:
        en_path = Path(args.en_file).expanduser().resolve()
        if not en_path.exists():
            sys.exit(f"EN file not found: {en_path}")
        en_articles = _split_articles(en_path.read_text(encoding="utf-8", errors="replace"))
        logger.info(f"EN: {len(en_articles)} articles parsed from {en_path.name}")

    if args.dv_file:
        dv_path = Path(args.dv_file).expanduser().resolve()
        if not dv_path.exists():
            sys.exit(f"DV file not found: {dv_path}")
        dv_articles = _split_articles(dv_path.read_text(encoding="utf-8", errors="replace"))
        logger.info(f"DV: {len(dv_articles)} articles parsed from {dv_path.name}")

    db_path = Path(args.db).expanduser().resolve() if args.db else None
    conn = get_connection(db_path)
    logger.info(f"DB: {conn.execute('PRAGMA database_list').fetchone()[2]}")

    saved = 0
    total_pairs = 0

    # Build a map from article number → (title, body) for fast paired lookup
    dv_map: dict[int, tuple[str, str]] = {num: (title, body) for num, title, body in dv_articles}
    en_map: dict[int, tuple[str, str]] = {num: (title, body) for num, title, body in en_articles}

    # Import EN articles
    for num, title, body in en_articles:
        if not body:
            continue
        en_id = EN_ID_BASE + num
        dv_id = DV_ID_BASE + num if num in dv_map else None
        _save(conn, en_id, "EN", dv_id, title, body)
        total_pairs += _save_sentence_pairs(conn, en_id, dv_id, "EN", body)
        saved += 1

    # Import DV articles
    for num, title, body in dv_articles:
        if not body:
            continue
        dv_id = DV_ID_BASE + num
        en_id = EN_ID_BASE + num if num in en_map else None
        _save(conn, dv_id, "DV", en_id, title, body)
        total_pairs += _save_sentence_pairs(conn, dv_id, en_id, "DV", body)
        saved += 1

    conn.commit()
    logger.info(f"Saved {saved} articles, {total_pairs} sentence pairs")

    if not args.no_fts_rebuild:
        _rebuild_fts(conn)

    conn.close()
    logger.info("Done. Run `moonlight build-embeddings` to encode new sentences.")


if __name__ == "__main__":
    main()
