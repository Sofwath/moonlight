#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Scrape Presidency Office press releases directly into moonlight.db.

Fetches all four categories (press_release, speech, vp_speech, news_bulletin)
from https://presidency.gov.mv, pairs each EN article with its DV version,
and writes directly to moonlight's articles + sentence_pairs tables.

Usage::

    python scripts/scrape_presidency.py
    python scripts/scrape_presidency.py --category press_release --start-id 28000
    python scripts/scrape_presidency.py --id-range 30000 36000
    python scripts/scrape_presidency.py --resume   # skip already-imported articles

After running, rebuild embeddings::

    moonlight build-embeddings

Design
------
The Presidency Office site assigns sequential integer IDs to articles.  EN and
DV versions of the same article have *different* IDs; the language toggle link
on each page carries the paired ID.  We discover pairs by fetching the EN page
and reading the toggle.

Scraping strategy: iterate IDs in a range rather than walking listing pages.
Listing-page pagination is fragile (changes with new articles); ID iteration
is stable and resumable.  Gaps (404, deleted articles) are silently skipped.

Rate limiting: 2 s between requests as in the original kahzaabu scraper.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import re
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import requests
from bs4 import BeautifulSoup

# Add repo root to path so we can import moonlight
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from moonlight.db import get_connection

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("scrape_presidency")

BASE_URL = "https://presidency.gov.mv"
DELAY_SECONDS = 2.0

# Maps category label to (category_id, listing_tid).
# category_id is the path segment; tid is the listing filter param.
CATEGORIES: dict[str, dict[str, int]] = {
    "press_release": {"id": 11, "tid": 1},
    "speech":        {"id": 12, "tid": 2},
    "vp_speech":     {"id": 13, "tid": 3},
    "news_bulletin": {"id": 290, "tid": 28},
}

# Known ID ranges from existing kahzaabu corpus.
# We start just above the known max to find new articles.
DEFAULT_ID_RANGE = (28000, 40000)


# ── HTTP session ──────────────────────────────────────────────────────────────

def _session() -> requests.Session:
    s = requests.Session()
    s.headers.update({
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/131.0.0.0 Safari/537.36"
        ),
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
    })
    retry = requests.adapters.Retry(
        total=3, backoff_factor=2, status_forcelist=[500, 502, 503, 504]
    )
    s.mount("https://", requests.adapters.HTTPAdapter(max_retries=retry))
    return s


# ── HTML parsing ──────────────────────────────────────────────────────────────

def _parse_date(date_str: str) -> str:
    try:
        dt = datetime.strptime(date_str.strip(), "%d %B %Y")
        return dt.strftime("%Y-%m-%d")
    except ValueError:
        return date_str.strip()


def _parse_article(html: str, article_id: int) -> Optional[dict]:
    """Return a dict matching the articles schema, or None if unparseable."""
    soup = BeautifulSoup(html, "lxml")

    # Title
    title_tag = soup.find("h1") or soup.find("h2")
    if not title_tag:
        return None
    title = title_tag.get_text(strip=True)

    # Presidency site returns HTTP 200 for missing/deleted articles with a 404 title
    if "404" in title or "not found" in title.lower() or "page not found" in title.lower():
        return None

    # Date + reference
    published_date = ""
    reference = None
    area = (
        soup.find("div", class_=re.compile(r"article|content|press", re.I))
        or soup.find("main")
        or soup.body
    )
    if area:
        text = area.get_text()
        m = re.search(r"(\d{1,2}\s+\w+\s+\d{4})", text)
        if m:
            published_date = _parse_date(m.group(1))
        m = re.search(r"Ref:\s*(.+?)(?:\s*$|\s*\n)", text, re.MULTILINE)
        if m:
            reference = m.group(1).strip()

    # Body
    body_html = ""
    body_text = ""
    content_div = (
        soup.find("div", class_=re.compile(r"article-body|entry-content|press-body", re.I))
        or soup.find("div", class_=re.compile(r"col-md-12|col-lg-12", re.I))
    )
    target = content_div or area
    if target:
        paras = target.find_all("p")
        if paras:
            body_html = "\n".join(str(p) for p in paras)
            body_text = "\n\n".join(
                p.get_text(strip=True) for p in paras if p.get_text(strip=True)
            )

    # Images
    image_urls: list[str] = []
    if area:
        for img in area.find_all("img", src=True):
            src = img["src"]
            if "storage.googleapis.com" in src or "presidency.gov.mv" in src:
                image_urls.append(src)

    # Language + paired ID from toggle link
    language = "EN"
    paired_id = None
    for link in soup.find_all("a", href=re.compile(r"/Press/Article/\d+")):
        link_text = link.get_text(strip=True)
        href = link.get("href", "")
        m = re.search(r"/Press/Article/(\d+)", href)
        if not m:
            continue
        linked_id = int(m.group(1))
        if linked_id == article_id:
            continue
        if link_text.strip() == "EN":
            language = "DV"
            paired_id = linked_id
            break
        if re.search(r"[ހ-޿]", link_text):
            language = "EN"
            paired_id = linked_id
            break

    # Category from URL or breadcrumb
    category = "press_release"
    breadcrumb = soup.find("nav", class_=re.compile(r"breadcrumb", re.I))
    if breadcrumb:
        bc_text = breadcrumb.get_text(" ").lower()
        if "speech" in bc_text:
            category = "speech"
        elif "bulletin" in bc_text or "news" in bc_text:
            category = "news_bulletin"
    category_id = CATEGORIES.get(category, CATEGORIES["press_release"])["id"]

    content_hash = hashlib.sha256(body_text.encode()).hexdigest()[:16]
    now = datetime.utcnow().isoformat()

    return {
        "id": article_id,
        "language": language,
        "paired_id": paired_id,
        "category": category,
        "category_id": category_id,
        "title": title,
        "body_text": body_text,
        "body_html": body_html,
        "reference": reference,
        "published_date": published_date,
        "image_urls": json.dumps(image_urls),
        "raw_page_html": html,
        "scraped_at": now,
        "content_hash": content_hash,
    }


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
    paired_id     = excluded.paired_id,
    title         = excluded.title,
    body_text     = excluded.body_text,
    body_html     = excluded.body_html,
    published_date= excluded.published_date,
    image_urls    = excluded.image_urls,
    scraped_at    = excluded.scraped_at,
    content_hash  = excluded.content_hash
"""


def _article_exists(conn, article_id: int, lang: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM articles WHERE id=? AND language=?", (article_id, lang)
    ).fetchone()
    return row is not None


def _save_article(conn, data: dict) -> None:
    conn.execute(_UPSERT, data)
    conn.commit()


def _save_sentence_pairs(conn, article: dict) -> int:
    """Split body_text into paragraphs and insert as sentence_pairs rows."""
    body = article.get("body_text") or ""
    if not body.strip():
        return 0
    paragraphs = [p.strip() for p in body.split("\n\n") if p.strip()]
    n = 0
    for idx, para in enumerate(paragraphs):
        conn.execute(
            """INSERT OR IGNORE INTO sentence_pairs
               (article_id, paired_article_id, lang, sentence_idx, text, text_len)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (
                article["id"],
                article.get("paired_id"),
                article["language"],
                idx,
                para,
                len(para),
            ),
        )
        n += 1
    conn.commit()
    return n


def _rebuild_fts(conn) -> None:
    logger.info("Rebuilding FTS5 indexes …")
    try:
        conn.execute("INSERT INTO articles_fts(articles_fts) VALUES('rebuild')")
        conn.commit()
        logger.info("  articles_fts: rebuilt")
    except Exception as e:
        logger.warning(f"  articles_fts rebuild failed: {e}")
    try:
        conn.execute("INSERT INTO sentence_pairs_fts(sentence_pairs_fts) VALUES('rebuild')")
        conn.commit()
        logger.info("  sentence_pairs_fts: rebuilt")
    except Exception as e:
        logger.warning(f"  sentence_pairs_fts rebuild failed: {e}")


# ── Scraping loop ─────────────────────────────────────────────────────────────

def scrape_range(
    session: requests.Session,
    conn,
    id_start: int,
    id_end: int,
    resume: bool = True,
) -> tuple[int, int]:
    """Iterate article IDs from id_start to id_end (inclusive).

    Returns (articles_saved, pairs_saved).
    """
    saved = 0
    pairs = 0
    skipped = 0

    for article_id in range(id_start, id_end + 1):
        # Skip if already in DB (resume mode)
        if resume and _article_exists(conn, article_id, "EN"):
            skipped += 1
            continue
        if resume and _article_exists(conn, article_id, "DV"):
            skipped += 1
            continue

        url = f"{BASE_URL}/Press/Article/{article_id}"
        try:
            time.sleep(DELAY_SECONDS)
            resp = session.get(url, timeout=30)
        except requests.RequestException as e:
            logger.warning(f"  [{article_id}] request error: {e}")
            continue

        if resp.status_code in (403, 404, 410):
            continue
        if not resp.ok:
            logger.warning(f"  [{article_id}] HTTP {resp.status_code}")
            continue

        data = _parse_article(resp.text, article_id)
        if not data or not data.get("body_text"):
            logger.debug(f"  [{article_id}] no body, skipping")
            continue

        _save_article(conn, data)
        pairs += _save_sentence_pairs(conn, data)
        saved += 1
        logger.info(
            f"  [{article_id}] {data['language']} | {data['title'][:60]}"
        )

        # If this is the EN version, also fetch the paired DV version
        if data["language"] == "EN" and data.get("paired_id"):
            dv_id = data["paired_id"]
            if resume and _article_exists(conn, dv_id, "DV"):
                continue
            time.sleep(DELAY_SECONDS)
            try:
                dv_resp = session.get(f"{BASE_URL}/Press/Article/{dv_id}", timeout=30)
            except requests.RequestException as e:
                logger.warning(f"  [{dv_id}] DV request error: {e}")
                continue
            if not dv_resp.ok:
                continue
            dv_data = _parse_article(dv_resp.text, dv_id)
            if dv_data and dv_data.get("body_text"):
                # Enforce correct pairing
                dv_data["language"] = "DV"
                dv_data["paired_id"] = article_id
                _save_article(conn, dv_data)
                pairs += _save_sentence_pairs(conn, dv_data)
                saved += 1
                logger.info(
                    f"  [{dv_id}] DV | {dv_data['title'][:60]}"
                )

        if saved % 100 == 0 and saved > 0:
            logger.info(f"  Progress: {saved} articles saved so far (skipped {skipped})")

    return saved, pairs


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Scrape Presidency Office articles into moonlight.db"
    )
    parser.add_argument(
        "--id-range", nargs=2, type=int, metavar=("START", "END"),
        default=list(DEFAULT_ID_RANGE),
        help=f"Article ID range to scrape (default: {DEFAULT_ID_RANGE[0]}–{DEFAULT_ID_RANGE[1]})",
    )
    parser.add_argument(
        "--start-id", type=int, default=None,
        help="Override start of range",
    )
    parser.add_argument(
        "--no-resume", action="store_true",
        help="Re-fetch articles already in DB (default: skip existing)",
    )
    parser.add_argument(
        "--no-fts-rebuild", action="store_true",
        help="Skip FTS5 rebuild at the end",
    )
    parser.add_argument(
        "--db", default=None, metavar="PATH",
        help="Path to moonlight.db (defaults to data/moonlight.db)",
    )
    args = parser.parse_args()

    id_start, id_end = args.id_range
    if args.start_id is not None:
        id_start = args.start_id

    db_path = Path(args.db).expanduser().resolve() if args.db else None
    conn = get_connection(db_path)
    session = _session()

    logger.info(f"DB: {conn.execute('PRAGMA database_list').fetchone()[2]}")
    logger.info(f"Scraping article IDs {id_start}–{id_end} …")

    try:
        saved, pairs = scrape_range(
            session, conn, id_start, id_end, resume=not args.no_resume
        )
    except KeyboardInterrupt:
        logger.info("Interrupted.")
        saved, pairs = 0, 0

    logger.info(f"Done: {saved} articles saved, {pairs} sentence pairs inserted")

    if not args.no_fts_rebuild and saved > 0:
        _rebuild_fts(conn)

    conn.close()
    logger.info("Finished. Run `moonlight build-embeddings` to encode new sentences.")


if __name__ == "__main__":
    main()
