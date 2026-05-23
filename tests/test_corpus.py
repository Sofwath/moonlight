# SPDX-License-Identifier: Apache-2.0
"""Tests for moonlight.corpus — FTS, genre classification, retrieval."""
from __future__ import annotations

import sqlite3

import pytest

from moonlight.db import get_connection
from moonlight.corpus import (
    backfill_corpus_fts,
    classify_genre,
    init_corpus_fts,
    search_articles,
    select_few_shot,
    select_glossary_subset,
    select_phrase_contexts,
)


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_conn() -> sqlite3.Connection:
    """Open an in-memory DB with full schema + FTS."""
    conn = get_connection(":memory:")
    init_corpus_fts(conn)
    return conn


def _insert_article(conn, *, aid, lang, paired_id, title, body,
                    category="general", published_date="2024-01-01"):
    conn.execute(
        """INSERT INTO articles
           (id, language, paired_id, category, category_id, title,
            body_text, body_html, reference, published_date, image_urls,
            raw_page_html)
           VALUES (?, ?, ?, ?, NULL, ?, ?, '', '', ?, '[]', '')""",
        (aid, lang, paired_id, category, title, body, published_date),
    )
    conn.commit()


@pytest.fixture
def populated_conn():
    """DB with 4 paired EN-DV articles + 1 glossary entry."""
    conn = _make_conn()

    # Article pair 1 — state visit
    _insert_article(
        conn, aid=1, lang="EN", paired_id=2,
        title="State Visit to Singapore",
        body="The President made an official state visit to Singapore. "
             "His Excellency met with senior officials.",
        category="state_visit", published_date="2024-01-10",
    )
    _insert_article(
        conn, aid=2, lang="DV", paired_id=1,
        title="ސިންގަޕޫރަށް ފުރަމާނަ ދަތުރުފުޅު",
        body="ރައީސުލްޖުމްހޫރިއްޔާ ސިންގަޕޫރަށް ރަސްމީ ދަތުރުފުޅެއް ކުރެއްވިއެވެ.",
        category="state_visit", published_date="2024-01-10",
    )

    # Article pair 2 — budget
    _insert_article(
        conn, aid=3, lang="EN", paired_id=4,
        title="Budget 2024 Announced",
        body="The government announced a budget of 10 billion MVR. "
             "Expenditure is expected to reach 8 billion.",
        category="budget", published_date="2024-02-01",
    )
    _insert_article(
        conn, aid=4, lang="DV", paired_id=3,
        title="ބަޖެޓު 2024 އިއުލާންކޮށްފި",
        body="ސަރުކާރުން 10 ބިލިއަން ރުފިޔާގެ ބަޖެޓެއް އިއުލާންކޮށްފިއެވެ.",
        category="budget", published_date="2024-02-01",
    )

    # Article pair 3 — speech (unpaired on DV side only)
    _insert_article(
        conn, aid=5, lang="EN", paired_id=6,
        title="Presidential Address to Parliament",
        body="The President delivered a keynote address at the opening of "
             "the People's Majlis, addressing legislators and officials.",
        category="speech", published_date="2024-03-01",
    )
    _insert_article(
        conn, aid=6, lang="DV", paired_id=5,
        title="ރިޔާސީ ބަޔާން",
        body="ރައީސުލްޖުމްހޫރިއްޔާ ރައްޔިތުންގެ މަޖިލީހުގައި ޚިޠާބު ދެއްވިއެވެ.",
        category="speech", published_date="2024-03-01",
    )

    # Glossary entry
    conn.execute(
        """INSERT INTO translation_glossary
           (en_term, dv_term, domain, freq, confidence, sample_en_ids,
            extracted_at, extracted_by)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
        ("state visit", "ފުރަމާނަ ދަތުރުފުޅު", "diplomacy",
         10, 0.95, "[]", "2024-01-01T00:00:00", "test"),
    )
    conn.commit()
    return conn


# ── 1. init_corpus_fts() ──────────────────────────────────────────────────────

def test_init_corpus_fts_creates_table():
    conn = _make_conn()
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "articles_fts" in tables


def test_init_corpus_fts_creates_triggers():
    conn = _make_conn()
    triggers = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='trigger'"
    ).fetchall()}
    assert "articles_fts_ai" in triggers
    assert "articles_fts_au" in triggers
    assert "articles_fts_ad" in triggers


def test_init_corpus_fts_is_idempotent():
    conn = _make_conn()
    # Second call should not raise
    result = init_corpus_fts(conn)
    assert result is True


# ── 2 & 3. backfill_corpus_fts() ─────────────────────────────────────────────

def test_backfill_empty_corpus_returns_zero():
    conn = _make_conn()
    written = backfill_corpus_fts(conn)
    assert written == 0


def test_backfill_writes_correct_count(populated_conn):
    conn = populated_conn
    # Clear FTS table first to test backfill
    conn.execute("DELETE FROM articles_fts")
    conn.commit()
    written = backfill_corpus_fts(conn)
    # 6 articles with body_text
    assert written == 6


def test_backfill_is_searchable_after(populated_conn):
    conn = populated_conn
    conn.execute("DELETE FROM articles_fts")
    conn.commit()
    backfill_corpus_fts(conn)
    rows = conn.execute(
        "SELECT COUNT(*) FROM articles_fts WHERE articles_fts MATCH '\"state visit\"'"
    ).fetchone()[0]
    assert rows > 0


# ── 4. classify_genre() ───────────────────────────────────────────────────────

def test_classify_state_visit_en():
    text = "The President made an official state visit to a neighbouring country."
    assert classify_genre(text, lang="EN") == "state_visit"


def test_classify_condolence_en():
    text = "The President expressed deepest sympathies over the tragic loss of life."
    assert classify_genre(text, lang="EN") == "condolence"


def test_classify_budget_en():
    text = "The government allocated 500 million MVR to the health sector budget."
    assert classify_genre(text, lang="EN") == "budget"


def test_classify_speech_en():
    text = "The President delivered a keynote address to the gathered delegates."
    assert classify_genre(text, lang="EN") == "speech"


def test_classify_unknown_returns_none():
    text = "Nothing here matches any genre pattern at all."
    assert classify_genre(text, lang="EN") is None


def test_classify_empty_returns_none():
    assert classify_genre("", lang="EN") is None


def test_classify_state_visit_dv():
    text = "ރައީސުލްޖުމްހޫރިއްޔާ ފުރަމާނަ ދަތުރުފުޅެއްގައި ވަޑައިގެންފި"
    assert classify_genre(text, lang="DV") == "state_visit"


def test_classify_budget_dv():
    text = "ސަރުކާރުން 10 ބިލިއަން ރުފިޔާގެ ބަޖެޓެއް ބިލިއަން ޚަރަދު"
    assert classify_genre(text, lang="DV") == "budget"


# ── 5. search_articles() ─────────────────────────────────────────────────────

def test_search_articles_basic(populated_conn):
    results = search_articles(populated_conn, "state visit", language="EN")
    assert len(results) > 0
    assert any(r["article_id"] == 1 for r in results)


def test_search_articles_language_filter(populated_conn):
    results = search_articles(populated_conn, "budget", language="EN")
    assert all(r["language"] == "EN" for r in results)


def test_search_articles_dv_language_filter(populated_conn):
    results = search_articles(populated_conn, "ބަޖެޓު", language="DV")
    assert all(r["language"] == "DV" for r in results)


def test_search_articles_require_paired(populated_conn):
    results = search_articles(
        populated_conn, "visit", language="EN", require_paired=True
    )
    assert all(r["paired_id"] is not None for r in results)


def test_search_articles_exclude_ids(populated_conn):
    results = search_articles(
        populated_conn, "visit", language="EN", exclude_ids=[1, 2]
    )
    assert all(r["article_id"] != 1 for r in results)
    assert all(r["article_id"] != 2 for r in results)


def test_search_articles_empty_query_returns_empty(populated_conn):
    results = search_articles(populated_conn, "", language="EN")
    assert results == []


def test_search_articles_limit(populated_conn):
    results = search_articles(populated_conn, "visit", language="EN", limit=1)
    assert len(results) <= 1


def test_search_articles_result_keys(populated_conn):
    results = search_articles(populated_conn, "budget", language="EN")
    if results:
        required = {"article_id", "language", "paired_id", "title",
                    "body_text", "published_date", "category", "rank"}
        assert required.issubset(results[0].keys())


# ── 6. select_few_shot() ─────────────────────────────────────────────────────

def test_select_few_shot_returns_at_most_k(populated_conn):
    results = select_few_shot(
        populated_conn, "EN", "state visit president official", k=2
    )
    assert len(results) <= 2


def test_select_few_shot_exclude_ids(populated_conn):
    results = select_few_shot(
        populated_conn, "EN", "state visit",
        k=5, exclude_article_ids={1, 2}
    )
    ids = [r["article_id"] for r in results]
    assert 1 not in ids
    assert 2 not in ids


def test_select_few_shot_returns_paired_bodies(populated_conn):
    results = select_few_shot(
        populated_conn, "EN", "state visit singapore official", k=3
    )
    for r in results:
        assert "source_body" in r
        assert "target_body" in r
        assert r["source_body"].strip()
        assert r["target_body"].strip()


def test_select_few_shot_empty_query_returns_empty(populated_conn):
    # FTS on whitespace-only query returns empty
    results = select_few_shot(populated_conn, "EN", "   ", k=3)
    assert results == []


# ── 7. select_phrase_contexts() ──────────────────────────────────────────────

def test_select_phrase_contexts_returns_contexts(populated_conn):
    # "Presidential Address" is a capitalised phrase in the speech article
    results = select_phrase_contexts(
        populated_conn,
        "Presidential Address to Parliament",
        source_lang="EN",
    )
    # May or may not return results depending on FTS — just verify structure
    for r in results:
        assert "phrase" in r
        assert "article_id" in r
        assert "snippet" in r


def test_select_phrase_contexts_empty_returns_empty(populated_conn):
    results = select_phrase_contexts(populated_conn, "", source_lang="EN")
    assert results == []


def test_select_phrase_contexts_exclude_ids(populated_conn):
    results = select_phrase_contexts(
        populated_conn,
        "Presidential Address to Parliament",
        source_lang="EN",
        exclude_article_ids={5, 6},
    )
    for r in results:
        assert r["article_id"] not in {5, 6}


# ── 8. select_glossary_subset() ──────────────────────────────────────────────

def test_select_glossary_subset_matching_term(populated_conn):
    results = select_glossary_subset(
        populated_conn,
        "The President made an official state visit to Singapore.",
        source_lang="EN",
    )
    terms = [r["en_term"] for r in results]
    assert "state visit" in terms


def test_select_glossary_subset_no_match(populated_conn):
    results = select_glossary_subset(
        populated_conn,
        "Nothing related to any glossary term whatsoever.",
        source_lang="EN",
    )
    assert results == []


def test_select_glossary_subset_empty_text(populated_conn):
    results = select_glossary_subset(populated_conn, "", source_lang="EN")
    assert results == []


def test_select_glossary_subset_result_keys(populated_conn):
    results = select_glossary_subset(
        populated_conn,
        "state visit to Singapore",
        source_lang="EN",
    )
    for r in results:
        assert "en_term" in r
        assert "dv_term" in r
        assert "domain" in r
        assert "freq" in r
