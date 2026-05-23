# SPDX-License-Identifier: Apache-2.0
"""Tests for moonlight.translator — private helpers and public translate()."""
from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from moonlight.db import get_connection
from moonlight.translator import (
    _apply_term_locks,
    _build_term_locks,
    _cache_lookup,
    _coerce_exemplars,
    _compose_prompt,
    _is_degenerate,
    _restore_term_locks,
    detect_language,
    validate_entities,
    translate,
)


# ── Mock LLMClient ────────────────────────────────────────────────────────────

class _MockClient:
    model_id = "mock-model-v1"

    class _spec:
        id = "mock-model-v1"
        in_per_m = 3.0
        out_per_m = 15.0

    spec = _spec()

    def __init__(self, response="The Maldives government announced today."):
        self._r = response

    def chat(self, system, user, **kw):
        return self._r, 100, 50

    def cost_usd(self, ti, to):
        return (ti * 3.0 + to * 15.0) / 1_000_000


# ── Fixtures ──────────────────────────────────────────────────────────────────

def _make_conn() -> sqlite3.Connection:
    return get_connection(":memory:")


@pytest.fixture
def conn():
    return _make_conn()


# ── 1. detect_language() ──────────────────────────────────────────────────────

def test_detect_thaana_is_dv():
    thaana_text = "ރައީސުލްޖުމްހޫރިއްޔާ ދިވެހިރާއްޖޭގެ ވެރިކަންކުރާ ރަށަށް ވަޑައިގެންނެވިއެވެ"
    assert detect_language(thaana_text) == "DV"


def test_detect_latin_is_en():
    assert detect_language("The President of the Maldives announced a new policy.") == "EN"


def test_detect_empty_is_en():
    assert detect_language("") == "EN"


def test_detect_none_is_en():
    assert detect_language(None) == "EN"


def test_detect_whitespace_is_en():
    assert detect_language("   \n\t  ") == "EN"


def test_detect_mixed_mostly_thaana_is_dv():
    # Mostly Thaana with a few Latin chars (proper names, numbers)
    mixed = "ރައީސް Mohamed 2024 ގައި ވިދާޅުވިއެވެ ދިވެހިރާއްޖޭގެ"
    assert detect_language(mixed) == "DV"


def test_detect_mixed_mostly_latin_is_en():
    # Mostly Latin text with a single Thaana word
    mixed = "The President visited ރ.ވ. in 2024 and met with officials there."
    assert detect_language(mixed) == "EN"


# ── 2. validate_entities() ────────────────────────────────────────────────────

def test_validate_numbers_preserved_passes():
    result = validate_entities(
        "The government allocated 500 million MVR across 12 sectors.",
        "The government allocated 500 million MVR across 12 sectors.",
        source_lang="EN", target_lang="EN",
    )
    assert result["passed"] is True
    assert result["missing_numbers"] == []


def test_validate_missing_multi_digit_fails():
    result = validate_entities(
        "There are 42 islands in the atoll.",
        "There are islands in the atoll.",
        source_lang="EN", target_lang="DV",
    )
    assert result["passed"] is False
    assert "42" in result["missing_numbers"]


def test_validate_single_digit_not_flagged():
    # Single-digit numbers should not trigger failure
    result = validate_entities(
        "He met 3 ministers today.",
        "He met the ministers today.",
        source_lang="EN", target_lang="DV",
    )
    assert "3" not in result["missing_numbers"]


def test_validate_extra_number_fails():
    result = validate_entities(
        "The budget is 100 million.",
        "The budget is 100 million and 200 million extra.",
        source_lang="EN", target_lang="EN",
    )
    assert result["passed"] is False
    assert "200" in result["extra_numbers"]


def test_validate_all_numbers_match_passes():
    result = validate_entities(
        "In 2024, 150 officials attended the 3rd summit.",
        "In 2024, 150 officials attended the summit.",
        source_lang="EN", target_lang="DV",
    )
    # 3 is single digit — not checked; 2024 and 150 must be present
    assert "2024" not in result["missing_numbers"]
    assert "150" not in result["missing_numbers"]


def test_validate_empty_source_passes():
    result = validate_entities(
        "", "",
        source_lang="EN", target_lang="EN",
    )
    assert result["passed"] is True


# ── 3. _build_term_locks() ────────────────────────────────────────────────────

def test_build_term_locks_glossary_match():
    glossary = [{"en_term": "state visit", "dv_term": "ފުރަމާނަ ދަތުރުފުޅު"}]
    locks = _build_term_locks(
        "The President made a state visit to Singapore.",
        source_lang="EN", glossary=glossary,
    )
    # Should have at least one lock for "state visit"
    targets = [t for _, _, _, t in locks]
    assert "ފުރަމާނަ ދަތުރުފުޅު" in targets


def test_build_term_locks_numbers():
    locks = _build_term_locks(
        "The budget is 500 million MVR.",
        source_lang="EN", glossary=[],
    )
    placeholders = [pl for _, _, pl, _ in locks]
    assert any("⟦K" in p for p in placeholders)


def test_build_term_locks_longer_match_wins():
    """When two glossary entries overlap, longer match should win."""
    glossary = [
        {"en_term": "state visit",          "dv_term": "ފުރަމާނަ ދަތުރުފުޅު"},
        {"en_term": "official state visit",  "dv_term": "ރަސްމީ ދަތުރުފުޅު"},
    ]
    locks = _build_term_locks(
        "The President made an official state visit abroad.",
        source_lang="EN", glossary=glossary,
    )
    # Only one lock should match (no overlapping spans)
    assert len(locks) == 1
    # The longer term should win
    _, _, _, target = locks[0]
    assert target == "ރަސްމީ ދަތުރުފުޅު"


def test_build_term_locks_empty_returns_empty():
    locks = _build_term_locks("Hello world", source_lang="EN", glossary=[])
    # "Hello" and "world" are single tokens — no numeric match, no glossary
    # But numeric RE won't match plain words; result should be empty
    assert locks == []


def test_build_term_locks_no_overlap():
    glossary = [
        {"en_term": "state visit", "dv_term": "A"},
        {"en_term": "Singapore",   "dv_term": "B"},
    ]
    locks = _build_term_locks(
        "The state visit to Singapore was successful.",
        source_lang="EN", glossary=glossary,
    )
    # Both terms are present and non-overlapping — both should be locked
    assert len(locks) == 2


# ── 4. _apply_term_locks() ────────────────────────────────────────────────────

def test_apply_term_locks_replaces_spans():
    glossary = [{"en_term": "state visit", "dv_term": "ފުރަމާނަ ދަތުރުފުޅު"}]
    locks = _build_term_locks(
        "The President made a state visit to Singapore.",
        source_lang="EN", glossary=glossary,
    )
    locked_text, pmap = _apply_term_locks(
        "The President made a state visit to Singapore.", locks
    )
    assert "⟦K" in locked_text
    assert "state visit" not in locked_text
    assert len(pmap) == len(locks)


def test_apply_term_locks_builds_pmap():
    glossary = [{"en_term": "state visit", "dv_term": "ދ"}]
    locks = _build_term_locks("official state visit today", "EN", glossary)
    _, pmap = _apply_term_locks("official state visit today", locks)
    for placeholder, target in pmap.items():
        assert placeholder.startswith("⟦K")
        assert target  # non-empty replacement


def test_apply_term_locks_empty_locks_returns_original():
    text = "Hello world"
    locked, pmap = _apply_term_locks(text, [])
    assert locked == text
    assert pmap == {}


# ── 5. _restore_term_locks() ─────────────────────────────────────────────────

def test_restore_term_locks_restores_all():
    pmap = {"⟦K0⟧": "ފުރަމާނަ", "⟦K1⟧": "500"}
    translated = "The ⟦K0⟧ was noted and ⟦K1⟧ was allocated."
    restored, missing = _restore_term_locks(translated, pmap)
    assert "⟦K0⟧" not in restored
    assert "⟦K1⟧" not in restored
    assert "ފުރަމާނަ" in restored
    assert "500" in restored
    assert missing == []


def test_restore_term_locks_reports_missing():
    pmap = {"⟦K0⟧": "ފ", "⟦K1⟧": "500"}
    # LLM dropped ⟦K1⟧
    translated = "The ⟦K0⟧ was mentioned."
    restored, missing = _restore_term_locks(translated, pmap)
    assert "⟦K1⟧" in missing


def test_restore_term_locks_empty_pmap():
    restored, missing = _restore_term_locks("Hello world", {})
    assert restored == "Hello world"
    assert missing == []


# ── 6. _is_degenerate() ──────────────────────────────────────────────────────

def test_is_degenerate_20_same_tokens():
    text = " ".join(["letzte"] * 20)
    assert _is_degenerate(text) is True


def test_is_degenerate_normal_sentence():
    text = "The President visited the island and met with local officials."
    assert _is_degenerate(text) is False


def test_is_degenerate_empty_string():
    assert _is_degenerate("") is False


def test_is_degenerate_short_text_below_threshold():
    text = " ".join(["word"] * 5)
    assert _is_degenerate(text) is False


def test_is_degenerate_custom_threshold():
    text = " ".join(["tok"] * 10)
    # Default threshold=15 — 10 repetitions should NOT trigger
    assert _is_degenerate(text, threshold=15) is False
    # Lower threshold=5 — should trigger
    assert _is_degenerate(text, threshold=5) is True


# ── 7. _compose_prompt() ─────────────────────────────────────────────────────

# Exemplar format expected by _compose_prompt:
# {"en_body": str, "dv_body": str, "en_article_id": int, "published_date": str}
_EXEMPLAR = {
    "en_body": "The President conducted an official visit.",
    "dv_body": "ރައީސް ދަތުރުފުުޅެއް ކުރެއްވިއެވެ.",
    "en_article_id": 99,
    "published_date": "2024-01-01",
}

# Phrase context format expected by _compose_prompt:
# {"phrase": str, "article_id": int, "source_snippet": str, "target_snippet": str}
_PHRASE_CTX = {
    "phrase": "Judicial Service Commission",
    "article_id": 10,
    "source_snippet": "The Judicial Service Commission convened.",
    "target_snippet": "ޖުޑީޝަލް ސަރވިސް ކޮމިޝަން ބައްދަލުވިއެވެ.",
}

_GLOSSARY = [{"en_term": "President", "dv_term": "ރައީސުލްޖުމްހޫރިއްޔާ"}]


def test_compose_prompt_faithful_mode_has_faithful_text():
    system, user = _compose_prompt(
        "The President announced a new policy.",
        source_lang="EN", target_lang="DV",
        glossary=[], exemplars=[],
        mode="faithful",
    )
    assert "FAITHFUL" in system


def test_compose_prompt_po_style_has_po_style_text():
    system, user = _compose_prompt(
        "The President announced a new policy.",
        source_lang="EN", target_lang="DV",
        glossary=[], exemplars=[],
        mode="po_style",
    )
    assert "Presidency Office" in system or "PO" in system


def test_compose_prompt_glossary_block_present():
    system, user = _compose_prompt(
        "The President announced a new policy.",
        source_lang="EN", target_lang="DV",
        glossary=_GLOSSARY, exemplars=[],
        mode="faithful",
    )
    assert "GLOSSARY" in user
    assert "President" in user
    assert "ރައީސުލްޖުމްހޫރިއްޔާ" in user


def test_compose_prompt_exemplars_block_present():
    system, user = _compose_prompt(
        "The President announced a new policy.",
        source_lang="EN", target_lang="DV",
        glossary=[], exemplars=[_EXEMPLAR],
        mode="faithful",
    )
    assert "EXAMPLES" in user
    assert "2024-01-01" in user


def test_compose_prompt_phrase_contexts_present():
    system, user = _compose_prompt(
        "The President announced a new policy.",
        source_lang="EN", target_lang="DV",
        glossary=[], exemplars=[],
        phrase_contexts=[_PHRASE_CTX],
        mode="faithful",
    )
    assert "PHRASE CONTEXTS" in user
    assert "Judicial Service Commission" in user


def test_compose_prompt_place_names_injected():
    place_names = [
        {"dv_thaana": "ހުޅުމާލެ", "en_name": "Hulhumalé", "feature_code": "PPL"}
    ]
    system, user = _compose_prompt(
        "ހުޅުމާލެ ގެ ރައީސް",
        source_lang="DV", target_lang="EN",
        glossary=[], exemplars=[],
        mode="faithful",
        place_names=place_names,
    )
    assert "Hulhumalé" in system or "PLACE" in system


def test_compose_prompt_no_place_names_uses_generic_rule():
    system, user = _compose_prompt(
        "ހުޅުމާލެ ގެ ރައީސް",
        source_lang="DV", target_lang="EN",
        glossary=[], exemplars=[],
        mode="faithful",
        place_names=None,
    )
    # Generic rule-of-thumb block should be present
    assert "MALDIVIAN PLACE" in system or "romanisation" in system.lower()


def test_compose_prompt_ends_with_translate_instruction():
    _, user = _compose_prompt(
        "Hello world.",
        source_lang="EN", target_lang="DV",
        glossary=[], exemplars=[],
        mode="faithful",
    )
    assert "NOW TRANSLATE THIS TO" in user
    assert "Hello world." in user


def test_compose_prompt_phrase_contexts_accepts_snippet_shape():
    system, user = _compose_prompt(
        "The President announced a new policy.",
        source_lang="EN", target_lang="DV",
        glossary=[], exemplars=[],
        phrase_contexts=[{
            "phrase": "Judicial Service Commission",
            "article_id": 10,
            "snippet": "The Judicial Service Commission convened.",
            "paired_id": 11,
        }],
        mode="faithful",
    )
    assert "PHRASE CONTEXTS" in user
    assert "Judicial Service Commission" in user
    assert "paired context unavailable" in user


def test_coerce_exemplars_maps_corpus_shape_for_en_source():
    raw = [{
        "article_id": 101,
        "paired_id": 202,
        "source_body": "The President met officials.",
        "target_body": "ރައީސް ބައްދަލުކުރެއްވި.",
        "published_date": "2024-01-01",
        "title": "State Visit",
    }]
    out = _coerce_exemplars(raw, source_lang="EN")
    assert out[0]["en_article_id"] == 101
    assert out[0]["dv_article_id"] == 202
    assert out[0]["en_body"] == "The President met officials."
    assert out[0]["dv_body"] == "ރައީސް ބައްދަލުކުރެއްވި."


# ── 8. _cache_lookup() ───────────────────────────────────────────────────────

def test_cache_lookup_miss_on_empty_db(conn):
    result = _cache_lookup(conn, "Some text to translate", "DV")
    assert result is None


def test_cache_lookup_hit_within_ttl(conn):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO translation_runs
           (source_lang, target_lang, input_text, output_text,
            exemplar_ids, glossary_terms_used, model, cost_usd, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("EN", "DV", "Some text to translate", "ތަރުޖަމާ",
         "[]", 0, "test-model", 0.001, now),
    )
    conn.commit()
    result = _cache_lookup(conn, "Some text to translate", "DV", ttl_hours=1.0)
    assert result is not None
    assert result["translation"] == "ތަރުޖަމާ"
    assert result["cache_hit"] is True


def test_cache_lookup_miss_after_ttl(conn):
    # Insert a record with a timestamp older than the TTL
    old_ts = (
        datetime.now(timezone.utc) - timedelta(hours=2)
    ).isoformat()
    conn.execute(
        """INSERT INTO translation_runs
           (source_lang, target_lang, input_text, output_text,
            exemplar_ids, glossary_terms_used, model, cost_usd, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("EN", "DV", "Some text to translate", "ތަރުޖަމާ",
         "[]", 0, "test-model", 0.001, old_ts),
    )
    conn.commit()
    result = _cache_lookup(conn, "Some text to translate", "DV", ttl_hours=1.0)
    assert result is None


def test_cache_lookup_different_target_lang_is_miss(conn):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO translation_runs
           (source_lang, target_lang, input_text, output_text,
            exemplar_ids, glossary_terms_used, model, cost_usd, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("EN", "DV", "Some text", "ތ", "[]", 0, "m", 0.0, now),
    )
    conn.commit()
    result = _cache_lookup(conn, "Some text", "EN", ttl_hours=1.0)
    assert result is None


def test_cache_lookup_respects_model_id(conn):
    now = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """INSERT INTO translation_runs
           (source_lang, target_lang, input_text, output_text,
            exemplar_ids, glossary_terms_used, model, cost_usd, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("EN", "DV", "Some text", "ހަމަ", "[]", 0, "model-a", 0.0, now),
    )
    conn.commit()
    hit = _cache_lookup(conn, "Some text", "DV", model_id="model-a", ttl_hours=1.0)
    miss = _cache_lookup(conn, "Some text", "DV", model_id="model-b", ttl_hours=1.0)
    assert hit is not None
    assert hit["translation"] == "ހަމަ"
    assert miss is None


# ── 9 & 10 & 11. translate() ─────────────────────────────────────────────────

def test_translate_returns_expected_keys(conn):
    mock_llm = _MockClient("The Maldives government announced today.")
    result = translate(
        conn,
        "ރައީސުލްޖުމްހޫރިއްޔާ ދިވެހިރާއްޖޭގެ ވެރިކަންކުރާ ރަށަށް ވަޑައިގެންނެވިއެވެ.",
        target_lang="EN",
        llm=mock_llm,
    )
    required_keys = {
        "translation", "source_lang", "target_lang", "cost_usd",
        "cache_hit", "entity_check", "model",
    }
    assert required_keys.issubset(result.keys())


def test_translate_cache_hit_false_on_first_call(conn):
    mock_llm = _MockClient("The Maldives government announced today.")
    result = translate(
        conn,
        "ދިވެހިރާއްޖެ ވެރިކަންކުރާ ރަށް",
        target_lang="EN",
        llm=mock_llm,
    )
    assert result["cache_hit"] is False


def test_translate_cache_hit_on_second_call(conn):
    mock_llm = _MockClient("The Maldives government announced today.")
    text = "ދިވެހިރާއްޖެ ކެބިނެޓް ވަޒީރު"
    translate(conn, text, target_lang="EN", llm=mock_llm)
    result2 = translate(conn, text, target_lang="EN", llm=mock_llm)
    assert result2["cache_hit"] is True


def test_translate_correct_source_and_target_lang(conn):
    mock_llm = _MockClient("ދިވެހި ތަރުޖަމާ.")
    result = translate(
        conn,
        "The President of the Maldives announced a new policy today.",
        target_lang="DV",
        llm=mock_llm,
    )
    assert result["source_lang"] == "EN"
    assert result["target_lang"] == "DV"


def test_translate_ablate_bypasses_cache(conn):
    mock_llm = _MockClient("Translation output one.")
    text = "ދިވެހިރާއްޖެ ސަރުކާރު"
    # First call — normal, should be cached
    translate(conn, text, target_lang="EN", llm=mock_llm)
    # Second call with ablate — should NOT return cache hit
    result = translate(
        conn, text, target_lang="EN", llm=mock_llm,
        ablate={"few_shot"},
    )
    assert result["cache_hit"] is False


def test_translate_empty_input_raises(conn):
    mock_llm = _MockClient()
    with pytest.raises(ValueError, match="empty"):
        translate(conn, "", llm=mock_llm)


def test_translate_whitespace_only_raises(conn):
    mock_llm = _MockClient()
    with pytest.raises(ValueError):
        translate(conn, "   ", llm=mock_llm)


def test_translate_same_source_and_target_raises(conn):
    mock_llm = _MockClient()
    # detect_language("Hello") = EN, target_lang="EN" → same → ValueError
    with pytest.raises(ValueError):
        translate(
            conn,
            "Hello this is an English sentence.",
            target_lang="EN",
            llm=mock_llm,
        )
