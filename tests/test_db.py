# SPDX-License-Identifier: Apache-2.0
from moonlight.db import get_connection


def test_database_initialization():
    # Use in-memory database to test initialization cleanly
    conn = get_connection(":memory:")
    try:
        # Check all expected tables are created
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        tables = {row[0] for row in cursor.fetchall()}
        
        expected_tables = {
            "articles",
            "translation_glossary",
            "translation_runs",
            "sentence_pairs",
            "place_names"
        }
        for table in expected_tables:
            assert table in tables, f"Expected table '{table}' was not created"

        # Check virtual tables (FTS5) are registered
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table' AND sql LIKE '%fts%'"
        )
        fts_tables = {row[0] for row in cursor.fetchall()}
        assert "articles_fts" in fts_tables
        assert "sentence_pairs_fts" in fts_tables

        # Verify columns of sentence_pairs
        cursor = conn.execute("PRAGMA table_info(sentence_pairs)")
        columns = {row[1]: row[2] for row in cursor.fetchall()}
        expected_columns = {
            "id": "INTEGER",
            "article_id": "INTEGER",
            "paired_article_id": "INTEGER",
            "lang": "TEXT",
            "sentence_idx": "INTEGER",
            "text": "TEXT",
            "text_len": "INTEGER",
            "embedding": "BLOB",
            "embedding_model": "TEXT"
        }
        for col_name, col_type in expected_columns.items():
            assert col_name in columns, f"Column '{col_name}' missing from sentence_pairs"

        # Verify triggers are created
        cursor = conn.execute(
            "SELECT name FROM sqlite_master WHERE type='trigger'"
        )
        triggers = {row[0] for row in cursor.fetchall()}
        expected_triggers = {
            "articles_fts_ai",
            "articles_fts_au",
            "articles_fts_ad",
            "sentence_pairs_ai",
            "sentence_pairs_ad"
        }
        for trigger in expected_triggers:
            assert trigger in triggers, f"Trigger '{trigger}' was not created"
    finally:
        conn.close()
