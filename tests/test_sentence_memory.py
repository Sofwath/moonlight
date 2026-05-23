# SPDX-License-Identifier: Apache-2.0
from moonlight.db import get_connection, insert_article, Article
from moonlight.sentence_memory import (
    split_sentences,
    backfill_sentence_pairs,
    select_sentence_memory,
    select_sentence_memory_hybrid
)


def test_split_sentences_english():
    text = "Dr. John met with the President at 10 A.M. It was a successful meeting! Did they agree?"
    # Dr. and A.M. should be protected, meeting! and agree? should trigger split
    sents = split_sentences(text, "EN")
    assert len(sents) == 3
    assert sents[0] == "Dr. John met with the President at 10 A.M."
    assert sents[1] == "It was a successful meeting!"
    assert sents[2] == "Did they agree?"


def test_split_sentences_dhivehi():
    # Example Dhivehi text with periods
    text = "މިއީ ފުރަތަމަ ޖުމްލައެވެ. މިއީ ދެވަނަ ޖުމްލައެވެ! މިއީ ތިންވަނަ ޖުމްލައެވެ؟"
    sents = split_sentences(text, "DV")
    assert len(sents) == 3
    assert sents[0] == "މިއީ ފުރަތަމަ ޖުމްލައެވެ."
    assert sents[1] == "މިއީ ދެވަނަ ޖުމްލައެވެ!"
    assert sents[2] == "މިއީ ތިންވަނަ ޖުމްލައެވެ؟"


def test_backfill_and_retrieve_sentences():
    conn = get_connection(":memory:")
    try:
        # Insert a pair of articles
        en_art = Article(
            id=1,
            language="EN",
            paired_id=1,
            category="press_release",
            category_id=None,
            title="President meets cabinet",
            body_text="The President met with the cabinet members today. They discussed the annual budget of the Maldives.",
            body_html="<p>The President met with the cabinet members today. They discussed the annual budget of the Maldives.</p>",
            reference="press-release-1",
            published_date="2026-05-23",
            image_urls=[],
            raw_page_html="<html>...</html>"
        )
        dv_art = Article(
            id=1,
            language="DV",
            paired_id=1,
            category="press_release",
            category_id=None,
            title="ރައީސް ކެބިނެޓާ ބައްދަލުކުރައްވައިފި",
            body_text="ރައީސުލްޖުމްހޫރިއްޔާ މިއަދު ކެބިނެޓްގެ މެންބަރުންނާ ބައްދަލުކުރެއްވިއެވެ. އެބޭފުޅުން މަޝްވަރާ ކުރެއްވީ ބަޖެޓާ ބެހޭގޮތުންނެވެ.",
            body_html="<p>ރައީސުލްޖުމްހޫރިއްޔާ މިއަދު ކެބިނެޓްގެ މެންބަރުންނާ ބައްދަލުކުރެއްވިއެވެ. އެބޭފުޅުން މަޝްވަރާ ކުރެއްވީ ބަޖެޓާ ބެހޭގޮތުންނެވެ.</p>",
            reference="press-release-1",
            published_date="2026-05-23",
            image_urls=[],
            raw_page_html="<html>...</html>"
        )
        insert_article(conn, en_art)
        insert_article(conn, dv_art)

        # Run backfill
        res = backfill_sentence_pairs(conn)
        assert res["articles_processed"] == 2
        assert res["sentences_inserted"] == 4  # 2 EN sents + 2 DV sents

        # Check sentence pairs table
        cursor = conn.execute("SELECT COUNT(*) FROM sentence_pairs")
        assert cursor.fetchone()[0] == 4

        # Retrieve using select_sentence_memory
        matches = select_sentence_memory(
            conn,
            "President met with the cabinet today.",
            source_lang="EN",
            k=2
        )
        assert len(matches) > 0
        assert matches[0]["source_text"] == "The President met with the cabinet members today."
        assert matches[0]["paired_body"].startswith("ރައީސުލްޖުމްހޫރިއްޔާ")

        # Retrieve using select_sentence_memory_hybrid (should fallback to BM25 if no embeddings)
        matches_hybrid = select_sentence_memory_hybrid(
            conn,
            "President met with the cabinet today.",
            source_lang="EN",
            k=2
        )
        assert len(matches_hybrid) > 0
        assert matches_hybrid[0]["source_text"] == "The President met with the cabinet members today."
    finally:
        conn.close()
