# SPDX-License-Identifier: Apache-2.0
"""moonlight — EN ↔ DV translation engine for Maldives Presidency Office text.

Named after *Moonlight*, the Maldives' first English-language newspaper,
active during the late Nasir era and ceasing publication in December 1978.

Quick start::

    from moonlight.db import get_connection
    from moonlight.translator import translate

    conn = get_connection()          # opens (or creates) data/moonlight.db
    result = translate(conn, "ދިވެހިރާއްޖެ")
    print(result["translation"])     # "Maldives"
    print(result["cost_usd"])

Public API
----------
The stable surface area of the package:

* :mod:`moonlight.translator` — ``translate()``, ``build_glossary()``,
  ``verify_back_translation()``, ``detect_language()``, ``validate_entities()``
* :mod:`moonlight.corpus` — ``search_articles()``, ``select_few_shot()``,
  ``select_phrase_contexts()``, ``select_glossary_subset()``,
  ``classify_genre()``
* :mod:`moonlight.db` — ``get_connection()``, ``corpus_stats()``,
  ``insert_article()``, ``Article``
* :mod:`moonlight.pricing` — ``model_id()``, ``cost()``, ``MODELS``
* :mod:`moonlight.place_names` — ``build_place_names()``,
  ``lookup_place_names_for_text()``
* :mod:`moonlight.sentence_memory` — ``select_sentence_memory_hybrid()``
"""
__version__ = "0.1.0"
