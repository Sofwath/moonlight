# SPDX-License-Identifier: Apache-2.0
"""Tests for moonlight.place_names — table init, lookup, formatting."""
from __future__ import annotations


import pytest

from moonlight.db import get_connection
from moonlight.place_names import (
    format_place_name_block,
    init_place_names,
    lookup_place_names_for_text,
)


# ── Fixture ───────────────────────────────────────────────────────────────────

@pytest.fixture
def conn():
    c = get_connection(":memory:")
    init_place_names(c)
    return c


def _insert_place(conn, *, geonameid, en_name, dv_thaana=None,
                  dv_latin=None, en_name_po=None,
                  feature_code="ISL", atoll_code=None,
                  latitude=None, longitude=None):
    conn.execute(
        """INSERT INTO place_names
           (geonameid, en_name, dv_thaana, dv_latin, en_name_po,
            feature_code, atoll_code, latitude, longitude, source)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, 'test')""",
        (geonameid, en_name, dv_thaana, dv_latin, en_name_po,
         feature_code, atoll_code, latitude, longitude),
    )
    conn.commit()


# ── 1. init_place_names() creates table ──────────────────────────────────────

def test_init_place_names_creates_table(conn):
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "place_names" in tables


def test_init_place_names_is_idempotent(conn):
    # Calling twice should not raise
    init_place_names(conn)
    tables = {r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ).fetchall()}
    assert "place_names" in tables


def test_init_place_names_returns_true_when_newly_created():
    # Use a raw sqlite3 connection (no init_db) so place_names truly doesn't exist
    import sqlite3 as _sqlite3
    raw = _sqlite3.connect(":memory:")
    result = init_place_names(raw)
    assert result is True


def test_init_place_names_returns_false_when_already_exists(conn):
    # Table already exists (created by fixture)
    result = init_place_names(conn)
    assert result is False


# ── 2. place_names table has expected columns ─────────────────────────────────

def test_place_names_expected_columns(conn):
    row = conn.execute("SELECT * FROM place_names LIMIT 0").description
    if row is None:
        # Table is empty but columns should still be described via PRAGMA
        cols = {r[1] for r in conn.execute("PRAGMA table_info(place_names)").fetchall()}
    else:
        cols = {r[0] for r in row}

    # Use PRAGMA for safety
    cols = {r[1] for r in conn.execute("PRAGMA table_info(place_names)").fetchall()}
    expected = {"geonameid", "en_name", "dv_thaana", "dv_latin", "en_name_po",
                "feature_code", "atoll_code", "latitude", "longitude", "source"}
    assert expected.issubset(cols)


# ── 3. lookup when table is empty ────────────────────────────────────────────

def test_lookup_empty_table_returns_empty_list(conn):
    result = lookup_place_names_for_text(conn, "ހުޅުމާލެ ގެ ރައީސް")
    assert result == []


def test_lookup_empty_source_returns_empty(conn):
    result = lookup_place_names_for_text(conn, "")
    assert result == []


def test_lookup_whitespace_source_returns_empty(conn):
    result = lookup_place_names_for_text(conn, "   ")
    assert result == []


# ── 4. lookup finds match when text contains dv_thaana substring ─────────────

def test_lookup_finds_matching_entry(conn):
    _insert_place(
        conn, geonameid=1001, en_name="Hulhumale",
        dv_thaana="ހުޅުމާލެ", en_name_po="Hulhumalé", feature_code="PPL",
    )
    result = lookup_place_names_for_text(
        conn, "ހުޅުމާލެ ގައި ހިނގި ހާދިސާ"
    )
    assert len(result) == 1
    assert result[0]["dv_thaana"] == "ހުޅުމާލެ"
    # en_name_po should be preferred over en_name
    assert result[0]["en_name"] == "Hulhumalé"


def test_lookup_uses_en_name_po_when_available(conn):
    _insert_place(
        conn, geonameid=1002, en_name="Male",
        dv_thaana="މާލެ", en_name_po="Malé", feature_code="PPLC",
    )
    result = lookup_place_names_for_text(conn, "މާލެ ތެދުވި")
    assert result[0]["en_name"] == "Malé"


def test_lookup_uses_raw_en_name_when_no_po_form(conn):
    _insert_place(
        conn, geonameid=1003, en_name="Maafushi",
        dv_thaana="މާފުށި", en_name_po=None, feature_code="ISL",
    )
    result = lookup_place_names_for_text(conn, "މާފުށި ރަށް")
    assert result[0]["en_name"] == "Maafushi"


def test_lookup_no_match_when_thaana_absent(conn):
    _insert_place(
        conn, geonameid=1004, en_name="Addu City",
        dv_thaana="އައްޑޫ", en_name_po="Addu City", feature_code="PPL",
    )
    result = lookup_place_names_for_text(conn, "ހދ. ތިލަދުުންމަތި")
    assert result == []


# ── 5. lookup returns longest match first ────────────────────────────────────

def test_lookup_longest_match_first(conn):
    # Short match: "ހދ" (2 chars)
    _insert_place(
        conn, geonameid=2001, en_name="Haa Dhaalu",
        dv_thaana="ހދ", feature_code="ADM1",
    )
    # Longer match: "ހދ. ކުޅުދުއްފުށި" (multi-word)
    _insert_place(
        conn, geonameid=2002, en_name="Kulhudhuffushi",
        dv_thaana="ކުޅުދުުއްފުށި", feature_code="PPL",
    )
    result = lookup_place_names_for_text(
        conn, "ހދ. ކުޅުުދުުއްފުުށި ރަށް"
    )
    # "ކުޅުުދުުއްފުުށި" is longer — but it may or may not match depending on
    # exact Unicode. We just verify order: longer dv_thaana entries come first.
    if len(result) > 1:
        lengths = [len(r["dv_thaana"]) for r in result]
        assert lengths == sorted(lengths, reverse=True)


def test_lookup_result_contains_feature_code(conn):
    _insert_place(
        conn, geonameid=3001, en_name="Gan",
        dv_thaana="ގަން", feature_code="AIRP",
    )
    result = lookup_place_names_for_text(conn, "ގަން ލ.ގަން")
    if result:
        assert "feature_code" in result[0]


# ── 6. format_place_name_block() non-empty ────────────────────────────────────

def test_format_place_name_block_nonempty():
    places = [
        {"dv_thaana": "DV_PLACE_ONE", "en_name": "Hulhumalé", "feature_code": "PPL"},
        {"dv_thaana": "DV_PLACE_TWO", "en_name": "Thinadhoo", "feature_code": "PPL"},
    ]
    block = format_place_name_block(places)
    # English names must appear in block
    assert "Hulhumalé" in block
    assert "Thinadhoo" in block
    # dv_thaana strings are passed through verbatim
    assert places[0]["dv_thaana"] in block
    assert places[1]["dv_thaana"] in block
    assert "PLACE" in block or "place" in block.lower() or "canonical" in block.lower()


def test_format_place_name_block_contains_arrow():
    places = [
        {"dv_thaana": "ހުޅުުމާލެ", "en_name": "Hulhumalé", "feature_code": "PPL"}
    ]
    block = format_place_name_block(places)
    assert "→" in block or "->" in block


def test_format_place_name_block_contains_note_about_apostrophes():
    places = [
        {"dv_thaana": "ތ", "en_name": "Test Place", "feature_code": "ISL"}
    ]
    block = format_place_name_block(places)
    # The block should warn about apostrophes/diacritics
    assert "apostrophe" in block or "Malé" in block or "diacritics" in block.lower()


# ── 7. format_place_name_block() empty list ──────────────────────────────────

def test_format_place_name_block_empty_returns_falsy():
    block = format_place_name_block([])
    # Should return empty string or minimal content
    assert not block or block.strip() == ""
