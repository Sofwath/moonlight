# SPDX-License-Identifier: Apache-2.0
"""EN ↔ DV translation engine for Maldives Presidency Office text.

Background
----------
The Maldives Presidency Office (PO) publishes press releases in both
English and Dhivehi (Thaana script).  The two versions are *not* literal
translations of each other — the PO writes each language in its own
register, with institution-specific terminology and idiomatic phrasings
that a generic LLM translation engine gets wrong.

This module produces translations that match the PO's published voice by
combining four sources of information at inference time:

  1. **Glossary** — a precomputed bilingual term dictionary mined from the
     corpus by ``build_glossary()``.  Provides canonical PO terminology for
     institution names and recurring policy phrases.

  2. **Few-shot exemplars** — 2–5 topic-similar paired articles retrieved
     from the corpus via BM25.  Gives the LLM concrete voice patterns in the
     right genre (state visit, budget, legislation, …).

  3. **Phrase contexts** — sentence-level snippets showing how specific
     phrases from the *input* appear in real PO text.  Tighter than
     article-level exemplars; the key insight behind the "expatriate workers"
     fix (see ADR 0016, §5).

  4. **Sentence-level translation memory** — for each sentence in the input,
     the closest matching PO sentence plus its paired-language article body.
     Reviewers called this "the single biggest improvement" — frontier models
     imitate exact translation analogues far better than vaguely related
     prose.

Translation modes
-----------------
Two modes let callers balance fidelity against register:

* ``mode="faithful"`` (default)
    Strict claim-preserving translation for automated pipelines.  Every
    number, date, name, and political attribution must survive unchanged.
    Style-transfer second pass is disabled; no embellishment.  Use this for
    any downstream consumer that makes factual judgements (claim extraction,
    contradiction detection, fact-checking).

* ``mode="po_style"``
    PO press-release register for human-readable / newsroom output.  Applies
    the DV→EN register rules (no direct quotes, theme-led titles, crisp
    prose) and runs a style-transfer second pass to close the register gap.
    Higher risk of semantic drift; **never use for automated fact-checking**.

This distinction was learned the hard way — see ADR 0016, §4 "Failure modes
of style-first translation".

Research notes
--------------
* ``enable_term_locking`` defaults to False because empirical ablation
  (2026-05-23) showed that placeholder substitution *hurts* Sonnet's output
  by ~0.19 composite score.  Frontier models' own entity priors are strong
  enough on PO content that explicit locking fragments their fluency more
  than it helps fidelity.  The implementation is preserved for research
  purposes and may help weaker models.

* ``n_candidates`` (Best-of-N) scores each candidate with a deterministic
  entity/numeric validator + numeric-F1 metric and returns the winner.  At
  n=3 it improves composite score by ~+0.06 at ~3× cost.

* ``style_transfer`` (second LLM pass) improves PO-register closeness but
  introduces hallucination risk in ``faithful`` mode.  Disabled by default
  unless ``mode="po_style"``.

See docs/adr/ for full design rationale and evaluation results.
"""
from __future__ import annotations

import json
import logging
import re
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Optional

from moonlight import corpus
from moonlight.llm import LLMClient, RateLimitError, model_id as resolve_model_id

logger = logging.getLogger(__name__)


# ── Language detection ─────────────────────────────────────────────────────────
#
# Thaana is the script used to write Dhivehi.  Unicode block: U+0780–U+07BF.
# https://www.unicode.org/charts/PDF/U0780.pdf
#
# The heuristic — >50% of non-whitespace characters are Thaana → DV — is robust
# to mixed text.  A DV press release with embedded English proper nouns (minister
# names, institution abbreviations, numeric strings) stays classified as DV.

_THAANA_RE = re.compile(r"[ހ-޿]")


def detect_language(text: Optional[str]) -> str:
    """Return ``'DV'`` if *text* is dominantly Thaana script, else ``'EN'``.

    Heuristic: more than 50 % of non-whitespace characters fall in the Thaana
    Unicode block (U+0780–U+07BF).  Empty / None input returns ``'EN'`` as a
    safe default — the vast majority of caller inputs are English.
    """
    if not text:
        return "EN"
    non_ws = [c for c in text if not c.isspace()]
    if not non_ws:
        return "EN"
    thaana_n = len(_THAANA_RE.findall(text))
    return "DV" if (thaana_n / len(non_ws)) > 0.5 else "EN"


# ── Prompt constants ───────────────────────────────────────────────────────────
#
# The system prompt is assembled from these building blocks.  They are kept as
# module-level constants so the same text is used everywhere; a change here
# propagates to every translate() call.  See ADR 0016, §3 for the rationale
# behind each rule.

_PO_STYLE_NOTES = (
    "The Maldives Presidency Office's Dhivehi register is formal "
    "and institutional. Distinctive markers:\n"
    "  - 'ރައީސުލްޖުމްހޫރިއްޔާ' (NOT 'ޕްރެޒިޑެންޓް') for 'President'\n"
    "  - 'ދައުލަތުގެ ވަޒީރުންގެ މަޖިލިސް' for 'Cabinet'\n"
    "  - 'ރައްޔިތުންގެ މަޖިލިސް' for 'People's Majlis' (Parliament)\n"
    "  - Full institutional names, not abbreviations\n"
    "  - No colloquial Thaana; preserves classical political register\n"
    "\n"
    "===  TERMINOLOGY FIDELITY RULE (LOAD-BEARING)  ===\n"
    "The PO has a preferred phrasing for many recurring concepts "
    "that DIFFERS from the literal translation of the input. "
    "Before producing your output, scan the EXAMPLES below for the "
    "same concept as anything in the input — if the examples use a "
    "specific phrase, you MUST use that exact phrase in your "
    "translation, even if the input uses different wording for the "
    "same concept.\n"
    "\n"
    "Worked example: an input that says 'undocumented foreign "
    "nationals' must be translated using the PO's actual phrase "
    "for that concept (e.g. 'undocumented expatriate workers' in "
    "the EN direction, or 'ބިދޭސީން' in the DV direction) — NOT a "
    "literal translation of the input's words. The examples carry "
    "the PO's canonical phrasing; defer to them.\n"
    "\n"
    "This rule overrides literal accuracy when the two conflict.\n"
)

# DV→EN register rules.  Surfaced by reviewer feedback against a batch of
# DV→EN translations that read as "Dhivehi shaped into English" rather than
# the PO's own English voice.  The PO publishes EN bilingually and writes it
# fresher than a word-for-word rendering of the Dhivehi.
_DV_TO_EN_REGISTER_RULES = (
    "===  PO ENGLISH REGISTER RULES (DV → EN ONLY)  ===\n"
    "The PO's English releases are its own composition, not a "
    "literal translation of the Dhivehi. Match the English voice, "
    "not the Dhivehi syntax:\n"
    "\n"
    "  1. NO DIRECT QUOTES. The PO never uses quotation marks in "
    "     English. Convert every quoted segment from the Dhivehi "
    "     (including inverted-comma Thaana quotes) into reported "
    "     speech. Use 'He said that…', 'The President noted that…', "
    "     'He emphasised that…' — never 'He said, \"…\"'.\n"
    "\n"
    "  2. THEME-LED TITLES. Headlines/titles are theme-led with "
    "     attribution at the end. Format: '<Topic phrase>: "
    "     President' (or '… : Vice President', etc.). NOT a "
    "     subject-verb-object restatement of the lead sentence. "
    "     Example: 'Economic progress over the past two and a half "
    "     years made possible through holistic policy: President' "
    "     — NOT 'President Muizzu states that…'.\n"
    "\n"
    "  3. TRIM LITERAL FLOURISHES. PO English is crisper than PO "
    "     Dhivehi. Translate the MEANING, not the metaphor:\n"
    "       'plunged into a deep crisis' → 'neglected the economy'\n"
    "       'heavy debts placed upon the shoulders of the people' \n"
    "         → 'longstanding legacy debts'\n"
    "       'no adversaries, working in friendship with all near \n"
    "        and far, large and small' → 'maintained cordial and \n"
    "         respectful relations with all countries'\n"
    "     Do not invent content; do remove ornamental phrasing.\n"
    "\n"
    "  4. MONEY FORMAT: 'USD X million' / 'USD X billion'. NOT "
    "     '$X million', NOT 'X million dollars', NOT 'X mn'. "
    "     Always 'USD' prefix.\n"
    "\n"
    "  5. NUMBERS AND DATES: numbers ≥ 10 always as digits "
    "     (20 beds, 50 million). Numbers 1–9 may be words in "
    "     prose ('two countries') but digits for dates, lists, "
    "     and monetary figures. Preserve ALL dates from the "
    "     source verbatim — omitting a date is a factual error.\n"
)


# Faithful-mode instructions.
#
# Research context: an early design used po_style for *all* outputs, then
# scored them against the PO's published English as the "gold" reference.
# The judge metric rewarded stylistic closeness to the PO's own embellished
# voice, which created a blind spot: the LLM learned to produce PO-sounding
# text even when the source didn't support the claims.
#
# The failure mode:
#   Source:    "maintained cordial and respectful relations with all countries"
#   Output:    "the Maldives no longer has any adversaries"   ← hallucinated
#
#   Source:    "previous administrations had neglected the economy"
#   Output:    "opposition was in government, plunged into deep crisis" ← invented framing
#
# Faithful mode addresses this.  It is the default for any downstream consumer
# that makes factual judgements.
_FAITHFUL_MODE_INSTRUCTIONS = (
    "===  FAITHFUL TRANSLATION MODE (LOAD-BEARING)  ===\n"
    "Translate faithfully. Preserve exact claims, numbers, "
    "entities, and political attribution. Do NOT:\n"
    "  - summarise, embellish, or strengthen claims\n"
    "  - infer causality or motivation not stated in the source\n"
    "  - reinterpret or 'editorialise' political framing\n"
    "  - convert specific claims into rhetorical flourishes (or "
    "vice versa)\n"
    "  - add adjectives, adverbs, or attributions of emotion "
    "(e.g. 'expressed his pleasure', 'with great pride') unless "
    "they appear in the source\n"
    "  - replace 'previous administrations' with 'opposition' "
    "or vice versa — preserve the source's exact political "
    "reference\n"
    "  - turn 'maintained relations with all countries' into "
    "'has no adversaries' — these are NOT equivalent\n"
    "\n"
    "NUMBERS — direction-specific rules:\n"
    "  • Dhivehi (DV) output: ALWAYS write as Arabic numerals "
    "(6, 20, 50). PO Dhivehi press releases never spell numbers "
    "as Thaana words (ހަ, ވިހި). Every count, rank, percentage, "
    "and monetary figure must be a digit.\n"
    "  • English (EN) output: numbers ≥ 10 always as digits "
    "(20 beds, 50 million, 14 weeks). Numbers 1–9 may be "
    "spelled as words in running prose ('two countries', "
    "'three meetings') — but use digits for dates ('4 November'), "
    "specific counts in enumerated lists ('6 key priority "
    "areas'), monetary amounts, and percentages.\n"
    "\n"
    "DATES: preserve ALL specific dates (day, month, year) "
    "explicitly mentioned in the source. Omitting a date is a "
    "factual error. If the source says 'on May 19' or 'on the "
    "4th of November', that date MUST appear in the output.\n"
    "\n"
    "Preserve rhetorical structure as closely as possible. "
    "Quoted speech in the source stays quoted in the output "
    "(do NOT convert quotes to reported speech in faithful "
    "mode).\n"
    "\n"
    "The glossary and exemplars below show the PO's canonical "
    "TERMINOLOGY for institutions and recurring concepts — use "
    "them for vocabulary. They do NOT license stylistic "
    "rewriting of the source's claims.\n"
)


# ── Hard terminology locking (Phase B2) ───────────────────────────────────────
#
# How it works:
#   1. Identify protected spans in the source (glossary terms, numbers) and
#      assign each a unique placeholder  ⟦K0⟧, ⟦K1⟧, …
#   2. Send the placeholder-substituted text to the LLM, with an instruction
#      to treat each marker as an opaque token.
#   3. After translation, replace each placeholder deterministically with its
#      target-language rendering.
#
# Research finding (2026-05-23 ablation):
#   Term locking *hurts* Sonnet's output by ~+0.19 composite score.  The
#   placeholder fragmentation breaks fluency more than it helps fidelity;
#   Sonnet's own entity priors are strong enough on PO content.  The feature
#   is preserved because it may help weaker models or legal text where exact
#   surface forms matter more than register.
#
# Placeholder format: ⟦K0⟧ — Unicode mathematical brackets (U+27E6, U+27E7).
# Chosen because they don't appear in normal political prose.

_PLACEHOLDER_RE = re.compile(r"⟦K(\d+)⟧")

# Conservative numeric lock regex: matches USD amounts, plain numbers with
# optional thousands-separator, percentages, and named magnitudes.
_NUMERIC_LOCK_RE = re.compile(
    r"(?:USD\s*)?\d+(?:,\d{3})*(?:\.\d+)?\s*(?:%|million|billion)?",
    re.IGNORECASE,
)


def _build_term_locks(
    text: str,
    source_lang: str,
    glossary: list[dict],
) -> list[tuple]:
    """Identify protected spans in *text* and assign placeholders.

    Returns a sorted, non-overlapping list of
    ``(start, end, placeholder, target_replacement)`` tuples.

    Lock priority (earlier wins on overlap):
      1. Glossary terms — longer match first (longest-match-wins rule).
      2. Numbers / monetary amounts / percentages.

    For English source, glossary lookup is case-insensitive.  For Dhivehi,
    it is exact (Thaana is case-free; substring matching is already loose
    enough).
    """
    candidates: list[tuple] = []

    # 1. Glossary terms
    for g in glossary:
        source_term = g.get("en_term" if source_lang == "EN" else "dv_term") or ""
        target_term = g.get("dv_term" if source_lang == "EN" else "en_term") or ""
        if not source_term or not target_term:
            continue
        if source_lang == "EN":
            haystack = text.lower()
            needle = source_term.lower()
        else:
            haystack = text
            needle = source_term
        idx = 0
        while True:
            pos = haystack.find(needle, idx)
            if pos < 0:
                break
            candidates.append((pos, pos + len(source_term), target_term, "gloss"))
            idx = pos + len(source_term)

    # 2. Numerics
    for m in _NUMERIC_LOCK_RE.finditer(text):
        if m.end() - m.start() < 1:
            continue
        candidates.append((m.start(), m.end(), m.group(), "num"))

    if not candidates:
        return []

    # Sort by start ascending; break ties by span length descending
    # so the longer match wins on overlap (longest-match-wins rule).
    candidates.sort(key=lambda c: (c[0], -(c[1] - c[0])))

    kept: list[tuple] = []
    last_end = -1
    next_placeholder_idx = 0
    for (start, end, target, _kind) in candidates:
        if start < last_end:
            continue
        placeholder = f"⟦K{next_placeholder_idx}⟧"
        next_placeholder_idx += 1
        kept.append((start, end, placeholder, target))
        last_end = end
    return kept


def _apply_term_locks(text: str, locks: list[tuple]) -> tuple[str, dict]:
    """Replace each locked span with its placeholder.

    Returns ``(locked_text, placeholder_to_replacement_map)``.  Locks must be
    sorted by start position and non-overlapping (guaranteed by
    ``_build_term_locks``).
    """
    if not locks:
        return text, {}
    out: list[str] = []
    pos = 0
    pmap: dict[str, str] = {}
    for (start, end, placeholder, target) in locks:
        out.append(text[pos:start])
        out.append(placeholder)
        pmap[placeholder] = target
        pos = end
    out.append(text[pos:])
    return "".join(out), pmap


def _restore_term_locks(translated: str, pmap: dict) -> tuple[str, list[str]]:
    """Replace placeholders in the translated text with their targets.

    Returns ``(restored_text, missing_placeholders)``.  Any placeholder that
    appeared in *pmap* but was absent from the LLM's output is included in
    ``missing_placeholders`` — visible failure beats silent fact corruption.
    """
    out = translated
    for placeholder, replacement in pmap.items():
        out = out.replace(placeholder, replacement)
    leftover = _PLACEHOLDER_RE.findall(out)
    missing = [p for p in pmap if p not in translated]
    if leftover:
        logger.warning(
            "_restore_term_locks: unrecognised placeholders in output: %s", leftover)
    if missing:
        logger.warning(
            "_restore_term_locks: %d placeholders dropped by LLM: %s",
            len(missing), missing,
        )
    return out, missing


# ── Entity / numeric validator ─────────────────────────────────────────────────
#
# Deterministic post-check: every multi-digit number in the source must appear
# in the translation.  This catches the "4 schools" → "1 school" class of
# errors that LLM judges sometimes miss because they reason about meaning
# rather than surface form.
#
# Single-digit numbers (0–9) are excluded because DV source may spell them as
# Thaana words while EN output renders them as digits; that's correct behaviour
# and would produce false positives.

_PN_STOPWORDS = frozenset({
    "The", "A", "An", "This", "That", "These", "Those",
    "He", "She", "It", "They", "We", "I",
    "Today", "Yesterday", "Tomorrow",
})

_PROPER_NOUN_RE = re.compile(r"\b[A-Z][A-Za-z]{2,}\b")
_NUMBER_RE = re.compile(r"\d[\d,]*\.?\d*")


def _is_single_digit(n: str) -> bool:
    return len(n) == 1 and n.isdigit()


def validate_entities(
    source: str,
    translated: str,
    *,
    source_lang: str,
    target_lang: str,
) -> dict:
    """Deterministic entity/numeric consistency check.

    Returns::

        {
          "passed":           bool,
          "missing_numbers":  [str],  # numbers in source absent from translation
          "extra_numbers":    [str],  # numbers in translation absent from source
          "missing_entities": [str],
          "summary":          str,
        }

    Research note: this check is intentionally conservative.  It only flags
    numbers ≥ 10 (single digits are ambiguous across languages), and proper
    nouns only when both source and target are English (Thaana lacks
    capitalisation so regex-based extraction doesn't apply to DV).
    """
    num_re = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
    src_nums = set(num_re.findall(source or ""))
    pred_nums = set(num_re.findall(translated or ""))
    src_nums_strict = {n for n in src_nums if not _is_single_digit(n)}
    pred_nums_strict = {n for n in pred_nums if not _is_single_digit(n)}
    missing_nums = sorted(src_nums_strict - pred_nums_strict)
    extra_nums = sorted(pred_nums_strict - src_nums_strict)

    missing_ents: list[str] = []
    if source_lang == "EN" and target_lang == "EN":
        src_ents = set(_PROPER_NOUN_RE.findall(source or "")) - _PN_STOPWORDS
        pred_ents = set(_PROPER_NOUN_RE.findall(translated or "")) - _PN_STOPWORDS
        missing_ents = sorted(src_ents - pred_ents)
    # DV→EN: can't check entity presence via regex (Thaana has no case).
    # Entity fidelity in that direction is covered by the caller's claim-check.

    failures = []
    if missing_nums:
        failures.append(f"missing numbers: {missing_nums}")
    if extra_nums:
        failures.append(f"extra numbers: {extra_nums}")
    if missing_ents:
        failures.append(f"missing entities: {missing_ents}")
    passed = not failures
    return {
        "passed": passed,
        "missing_numbers": missing_nums,
        "extra_numbers": extra_nums,
        "missing_entities": missing_ents,
        "summary": "; ".join(failures) if failures else "all entities/numbers preserved",
    }


# ── Utility functions ──────────────────────────────────────────────────────────

def _extract_text(resp) -> str:
    """Concatenate text blocks from an Anthropic API response object.

    Kept for backward compatibility.  New code should use ``LLMClient.chat()``
    which handles extraction internally for all providers.
    """
    return "".join(
        getattr(b, "text", "")
        for b in resp.content if getattr(b, "type", None) == "text"
    )


def _is_degenerate(text: str, *, threshold: int = 15) -> bool:
    """Return True if *text* exhibits repeating-token pathology.

    A healthy translation doesn't produce 15+ identical consecutive
    whitespace-separated tokens.  This catches a rare Sonnet failure mode
    observed during evaluation: the decoder gets stuck on a token and repeats
    it hundreds of times (e.g. ``"letzte letzte letzte …"``).
    """
    if not text:
        return False
    tokens = text.split()
    if len(tokens) < threshold:
        return False
    run = 1
    for i in range(1, len(tokens)):
        if tokens[i] == tokens[i - 1]:
            run += 1
            if run >= threshold:
                return True
        else:
            run = 1
    return False


# ── Style-transfer second pass ─────────────────────────────────────────────────
#
# Research context:
#   The initial four-layer prompt produces competent translations that tend to
#   retain source-language flow patterns.  A second LLM pass with the explicit
#   instruction "match this PO exemplar's voice" closes the register gap.
#
#   Evaluation (2026-05-22): style-transfer improves PO-register closeness by
#   ~+0.15 composite score but introduces hallucination risk when the polish
#   step doesn't have access to the source.  Fixed by also passing the original
#   source to the polish call; the LLM can verify facts weren't dropped.
#
#   Style-transfer is DISABLED in faithful mode (default).  It is the
#   embellishment engine; for claim-extraction consumers we cannot afford drift.


def _style_transfer_polish(
    draft: str,
    exemplars: list[dict],
    *,
    source_text: str,
    source_lang: str,
    target_lang: str,
    llm: "LLMClient",
) -> tuple[str, float, int, int]:
    """Second LLM pass: rewrite *draft* to match the top exemplar's voice.

    Returns ``(polished_text, cost_usd, tokens_in, tokens_out)``.

    If the polish result is empty or degenerate, the original *draft* is
    returned unchanged (safe fallback).  The source is passed alongside the
    draft so the LLM can verify that no facts were dropped during polishing.
    """
    if not exemplars or not draft.strip():
        return draft, 0.0, 0, 0
    top = exemplars[0]
    exemplar_target = (
        top.get("en_body" if target_lang == "EN" else "dv_body") or ""
    )
    if not exemplar_target.strip():
        return draft, 0.0, 0, 0
    target_name = "Dhivehi (Thaana script)" if target_lang == "DV" else "English"
    source_name = "Dhivehi" if source_lang == "DV" else "English"
    exemplar_snippet = exemplar_target[:2000]
    source_snippet = source_text[:2000]
    system = (
        f"You are polishing a translation to match the Maldives "
        f"Presidency Office's actual published voice in {target_name}. "
        f"The draft is competent but reads as source-language-shaped. "
        f"Rewrite it to sound like the PO writing directly — without "
        f"losing any facts.\n\n"
        f"PRIORITY ORDER (when rules conflict, earlier wins):\n"
        f"  1. FIDELITY FIRST. Preserve EVERY fact, number, date, "
        f"name, institution, monetary amount, and percentage from "
        f"the source. Do NOT drop any. Do NOT invent new ones. The "
        f"draft is your reference for what facts exist — cross-check "
        f"against the source text below if in doubt.\n"
        f"  2. NO DIRECT QUOTES. Convert any quoted speech in the "
        f"draft to reported speech. The PO does not use quotation "
        f"marks for speech in its English releases.\n"
        f"  3. TRIM VERBOSITY. PO {target_name} is crisp. Compress "
        f"long explanatory clauses; remove redundant repetition; "
        f"prefer one tight sentence over two loose ones.\n"
        f"  4. MATCH PO IDIOM. Use the reference's vocabulary and "
        f"phrasing patterns, not literal calques of the source.\n"
        f"  5. STRUCTURAL MATCH. Match the reference's paragraph "
        f"length, attribution pattern, and title style (if any).\n"
        f"\n"
        f"If the DRAFT contains placeholder markers of the form "
        f"⟦K0⟧, ⟦K1⟧, etc., they are pre-locked terms — preserve each "
        f"marker VERBATIM in your polished output.\n"
        f"\n"
        f"Output ONLY the polished translation. No commentary, no "
        f"wrapping quotes, no prefixes."
    )
    user = (
        f"SOURCE ({source_name}) — verify nothing from here gets "
        f"lost in your polish:\n\n"
        f"{source_snippet}\n\n"
        f"---\n\n"
        f"REFERENCE — an actual PO press release in {target_name}. "
        f"Match this voice exactly:\n\n"
        f"{exemplar_snippet}\n\n"
        f"---\n\n"
        f"DRAFT TRANSLATION to polish:\n\n"
        f"{draft}\n\n"
        f"---\n\n"
        f"Output the polished version, in {target_name}, matching "
        f"the reference's voice while preserving every fact from "
        f"the source."
    )
    polished, tokens_in, tokens_out = llm.chat(
        system, user, max_tokens=4000, temperature=0.1)
    cost = llm.cost_usd(tokens_in, tokens_out)
    if not polished or _is_degenerate(polished):
        logger.warning("style_transfer: polish returned empty/degenerate; keeping draft")
        return draft, cost, tokens_in, tokens_out
    return polished, cost, tokens_in, tokens_out


# ── Prompt assembly ────────────────────────────────────────────────────────────
#
# The four-layer prompt structure:
#
#   [system prompt]           — mode-specific translation instructions
#   GLOSSARY                  — bilingual term pairs from the corpus
#   PHRASE CONTEXTS           — sentence-level usage examples for input phrases
#   SENTENCE-LEVEL TM         — per-sentence closest matches with paired body
#   EXAMPLES                  — article-level few-shot exemplars
#   NOW TRANSLATE THIS TO … : — the actual input text
#
# Each layer is additive.  Ablation experiments confirm all four layers
# contribute to the composite score (see ADR 0016, Table 1).


def _compose_prompt(
    input_text: str,
    source_lang: str,
    target_lang: str,
    glossary: list[dict],
    exemplars: list[dict],
    phrase_contexts: Optional[list[dict]] = None,
    sentence_memory: Optional[list[dict]] = None,
    mode: str = "faithful",
    place_names: Optional[list[dict]] = None,
) -> tuple[str, str]:
    """Assemble the (system_prompt, user_message) pair for the LLM.

    *mode* selects the instruction block:
      - ``"faithful"`` — strict claim-preserving instructions
      - ``"po_style"`` — PO register instructions with DV→EN register rules

    *place_names* is a list of dicts returned by
    ``moonlight.place_names.lookup_place_names_for_text()``.  When populated,
    exact Thaana→EN mappings for this specific text are injected into the
    system prompt instead of a generic rule-of-thumb.
    """
    target_name = "Dhivehi (Thaana script)" if target_lang == "DV" else "English"
    source_name = "Dhivehi" if source_lang == "DV" else "English"

    if mode == "faithful":
        if source_lang == "DV" and target_lang == "EN":
            if place_names:
                from moonlight.place_names import format_place_name_block
                place_name_block = format_place_name_block(place_names)
            else:
                place_name_block = (
                    "\nMALDIVIAN PLACE-NAME ROMANISATION (DV → EN):\n"
                    "Use the PO's canonical English romanisation:\n"
                    "  - Island names typically end in 'u' and may use "
                    "an apostrophe for the glottal stop "
                    "(e.g. Kan'ditheemu, Hanimaadhoo, Kudahuvadhoo). "
                    "Do NOT drop the final 'u' or apostrophe.\n"
                    "  - Atoll names: 'North/South [Name] Atoll' "
                    "(e.g. 'North Miladhunmadulu Atoll') — not the "
                    "Dhivehi directional form.\n"
                )
        else:
            place_name_block = ""
        system = (
            f"You translate text from {source_name} to {target_name} "
            f"for an automated political fact-checking pipeline. "
            f"Your output will be consumed by claim-extraction and "
            f"verification systems that treat it as authoritative "
            f"evidence — semantic drift introduces FALSE CLAIMS into "
            f"the pipeline.\n\n"
            f"{_FAITHFUL_MODE_INSTRUCTIONS}\n"
            f"Use the PO's canonical terminology (from glossary and "
            f"exemplars below) when translating institutional names "
            f"and recurring concepts — but never let stylistic "
            f"matching override claim fidelity."
            f"{place_name_block}\n"
            f"Output ONLY the translation. No commentary, no "
            f"wrapping quotes around the whole output, no prefixes."
        )
    else:
        register_block = (
            f"\n{_DV_TO_EN_REGISTER_RULES}" if target_lang == "EN" else ""
        )
        system = (
            f"You translate text from {source_name} to {target_name} in "
            f"the style of the Maldives Presidency Office press releases. "
            f"Match their register, terminology, and idiomatic political "
            f"vocabulary exactly.\n\n"
            f"{_PO_STYLE_NOTES}{register_block}\n"
            f"Output ONLY the translation. No commentary, no wrapping "
            f"quotes around the whole output, no prefixes like 'Here is "
            f"the translation:'. Plain text."
        )

    parts: list[str] = []

    if glossary:
        parts.append(
            "GLOSSARY — terms appearing in the input with the press "
            "office's preferred translations:"
        )
        for g in glossary:
            parts.append(f"  - \"{g['en_term']}\" ↔ \"{g['dv_term']}\"")
        parts.append("")

    # Phrase contexts: sentence-level snippets showing how specific phrases
    # from the input are used in real PO text.  More precise than article-level
    # exemplars because they show the term in its actual syntactic context.
    if phrase_contexts:
        parts.append(
            f"PHRASE CONTEXTS — sentences from the press office "
            f"corpus showing how specific phrases in your input "
            f"are used in real {source_name} → {target_name} "
            f"pairs. Match these sentence-level patterns closely:"
        )
        for ctx in phrase_contexts:
            phrase = ctx.get("phrase", "")
            article_id = ctx.get("article_id", "?")
            source_snippet = ctx.get("source_snippet") or ctx.get("snippet") or ""
            target_snippet = (
                ctx.get("target_snippet")
                or ctx.get("paired_snippet")
                or "(paired context unavailable)"
            )
            parts.append(f"\n[\"{phrase}\" — from art #{article_id}]:")
            parts.append(f"  {source_name}: {source_snippet}")
            parts.append(f"  {target_name}: {target_snippet}")
        parts.append("")

    # Sentence-level translation memory: for each input sentence, the closest
    # PO sentence + its paired article body.  These are actual translation
    # analogues — the strongest local reference for each sentence's rendering.
    if sentence_memory:
        parts.append(
            f"SENTENCE-LEVEL TRANSLATION MEMORY — for each input "
            f"sentence, the closest matching PO {source_name} "
            f"sentence + the paired {target_name} article body. "
            f"Use these as the strongest local translation "
            f"reference:"
        )
        for i, sm in enumerate(sentence_memory, start=1):
            parts.append(
                f"\nMatch {i} [from art #{sm['source_article_id']} "
                f"→ paired art #{sm['paired_article_id']}]:"
            )
            parts.append(f"  Your input sentence: {sm['input_sentence'][:160]}")
            parts.append(
                f"  Closest {source_name} sentence in corpus: "
                f"{sm['source_text'][:300]}"
            )
            parts.append(
                f"  Paired {target_name} article body "
                f"(first 400 chars): {sm['paired_body'][:400]}"
            )
        parts.append("")

    if exemplars:
        parts.append(
            f"EXAMPLES of the press office's {source_name}↔{target_name} "
            f"style (broader article-level context):"
        )
        for i, ex in enumerate(exemplars, start=1):
            en_snippet = (ex.get("en_body") or "")[:600]
            dv_snippet = (ex.get("dv_body") or "")[:600]
            en_article_id = ex.get("en_article_id", "?")
            published_date = ex.get("published_date", "")
            parts.append(
                f"\nExample {i} [{published_date}, art #{en_article_id}]:"
            )
            parts.append(f"  EN: {en_snippet}")
            parts.append(f"  DV: {dv_snippet}")
        parts.append("")

    parts.append(f"NOW TRANSLATE THIS TO {target_name.upper()}:")
    parts.append("")
    parts.append(input_text)
    return system, "\n".join(parts)


def _coerce_exemplars(exemplars: list[dict], source_lang: str) -> list[dict]:
    """Normalize exemplar rows into translator's canonical key shape.

    Supports both the newer corpus shape
    (article_id/source_body/target_body/...) and the canonical shape used by
    prompt assembly (en_body/dv_body/en_article_id/...).
    """
    normalized: list[dict] = []
    for ex in exemplars:
        source_body = ex.get("source_body") or ""
        target_body = ex.get("target_body") or ""
        article_id = ex.get("article_id")
        paired_id = ex.get("paired_id")
        title = (ex.get("title") or ex.get("en_title") or "").strip()

        if source_lang == "EN":
            en_article_id = ex.get("en_article_id", article_id)
            dv_article_id = ex.get("dv_article_id", paired_id)
            en_body = ex.get("en_body", source_body)
            dv_body = ex.get("dv_body", target_body)
        else:
            en_article_id = ex.get("en_article_id", paired_id)
            dv_article_id = ex.get("dv_article_id", article_id)
            en_body = ex.get("en_body", target_body)
            dv_body = ex.get("dv_body", source_body)

        normalized.append({
            "en_article_id": en_article_id,
            "dv_article_id": dv_article_id,
            "en_body": en_body or "",
            "dv_body": dv_body or "",
            "en_title": ex.get("en_title", title),
            "published_date": ex.get("published_date", ""),
        })
    return normalized


# ── Translation cache ──────────────────────────────────────────────────────────
#
# ``translation_runs`` doubles as an LRU cache — same input + target language
# within the TTL returns instantly without a fresh LLM call.  Cache key is
# (input_text, target_lang); TTL defaults to 1 hour.
#
# Ablation runs bypass the cache because they carry different pipeline flags
# and a cached "full pipeline" result would silently invalidate the experiment.


def _cache_lookup(
    conn: sqlite3.Connection,
    input_text: str,
    target_lang: str,
    *,
    model_id: Optional[str] = None,
    ttl_hours: float = 1.0,
) -> Optional[dict]:
    """Return a cached translation if key fields match within TTL.

    Cache key is always input text + target language + recency window. When
    ``model_id`` is provided, lookup is additionally scoped to that exact model
    so different model runs do not cross-contaminate cached outputs.
    """
    cutoff = (
        datetime.now(timezone.utc) - timedelta(hours=ttl_hours)
    ).isoformat()
    if model_id is not None:
        row = conn.execute(
            """SELECT source_lang, target_lang, output_text, exemplar_ids,
                      glossary_terms_used, model, cost_usd, created_at
               FROM translation_runs
               WHERE input_text = ? AND target_lang = ?
                 AND model = ? AND created_at >= ?
               ORDER BY created_at DESC LIMIT 1""",
            (input_text, target_lang, model_id, cutoff),
        ).fetchone()
    else:
        row = conn.execute(
            """SELECT source_lang, target_lang, output_text, exemplar_ids,
                      glossary_terms_used, model, cost_usd, created_at
               FROM translation_runs
               WHERE input_text = ? AND target_lang = ? AND created_at >= ?
               ORDER BY created_at DESC LIMIT 1""",
            (input_text, target_lang, cutoff),
        ).fetchone()
    if row is None:
        return None
    try:
        exemplar_ids = json.loads(row[3] or "[]")
    except (json.JSONDecodeError, TypeError):
        exemplar_ids = []
    exemplar_refs: list[dict] = []
    if exemplar_ids:
        placeholders = ",".join("?" * len(exemplar_ids))
        rows = conn.execute(
            f"SELECT id, paired_id, title, published_date "
            f"FROM articles WHERE id IN ({placeholders})",
            list(exemplar_ids),
        ).fetchall()
        by_id = {r[0]: r for r in rows}
        for eid in exemplar_ids:
            r = by_id.get(eid)
            if r is not None:
                exemplar_refs.append({
                    "en_article_id":  r[0],
                    "dv_article_id":  r[1],
                    "title":          (r[2] or "").strip(),
                    "published_date": r[3] or "",
                })
    return {
        "translation":         row[2],
        "source_lang":         row[0],
        "target_lang":         row[1],
        "exemplar_ids":        exemplar_ids,
        "exemplars":           exemplar_refs,
        "glossary_terms_used": row[4] or 0,
        "phrase_contexts":     [],
        "model":               row[5],
        "cost_usd":            row[6] or 0.0,
        "cache_hit":           True,
        "cached_at":           row[7],
    }


# ── Best-of-N helpers (C2) ─────────────────────────────────────────────────────
#
# Research context:
#   Generating N candidates and scoring each deterministically (entity
#   validator + numeric-F1) improves composite score by ~+0.06 at ~3× cost.
#   The deterministic scorer is a better arbiter than temperature alone because
#   it captures exactly the failure mode we care about: number / entity loss.


def _single_llm_call(
    llm: "LLMClient",
    system: str,
    user: str,
) -> tuple[str, float, int, int]:
    """Single LLM call with one degenerate-output retry at higher temperature.

    Returns ``(translation_text, cost_usd, tokens_in, tokens_out)``.
    Temperature quirks (o1/o3, DeepSeek-R1) are handled inside ``LLMClient``.
    """
    translation, tokens_in, tokens_out = llm.chat(system, user, max_tokens=4000)
    cost = llm.cost_usd(tokens_in, tokens_out)

    if _is_degenerate(translation):
        logger.warning(
            "_single_llm_call: degenerate output (model=%s, tokens_out=%d); "
            "retrying at temperature=0.7",
            llm.model_id, tokens_out,
        )
        translation2, r_in, r_out = llm.chat(
            system, user, max_tokens=4000, temperature=0.7)
        cost += llm.cost_usd(r_in, r_out)
        tokens_in += r_in
        tokens_out += r_out
        if not _is_degenerate(translation2):
            translation = translation2

    return translation, cost, tokens_in, tokens_out


def _candidate_score(
    translation: str,
    source: str,
    source_lang: str,
    target_lang: str,
) -> float:
    """Score a single candidate for Best-of-N selection.

    Score = 2.0 × entity_check_passed  +  numeric_F1

    Entity validator pass (boolean, weight 2.0) dominates: a candidate that
    preserves all numbers always beats one that doesn't.  Among passing
    candidates, numeric-F1 breaks ties.
    """
    check = validate_entities(
        source, translation, source_lang=source_lang, target_lang=target_lang)
    num_re = re.compile(r"\d+(?:,\d{3})*(?:\.\d+)?")
    src_nums = set(num_re.findall(source or ""))
    pred_nums = set(num_re.findall(translation or ""))
    if not src_nums and not pred_nums:
        num_f1 = 1.0
    elif not src_nums or not pred_nums:
        num_f1 = 0.0
    else:
        tp = len(src_nums & pred_nums)
        num_f1 = (2 * tp / (len(src_nums) + len(pred_nums))) if tp else 0.0
    return 2.0 * float(check["passed"]) + num_f1


# ── Public translation entry point ─────────────────────────────────────────────

_DISCLAIMER = (
    "Reference-implementation output. This translation was produced "
    "by an LLM in the style of Maldives Presidency Office press "
    "releases. NOT an official translation — review against the "
    "original before publishing or quoting."
)


def translate(
    conn: sqlite3.Connection,
    text: str,
    *,
    target_lang: Optional[str] = None,
    llm: Optional["LLMClient"] = None,
    model_alias: str = "claude-sonnet",
    verify: bool = False,
    exclude_article_ids: Optional[set] = None,
    style_transfer: bool = True,
    mode: str = "faithful",
    ablate: Optional[set] = None,
    enable_term_locking: bool = False,
    n_candidates: int = 1,
) -> dict:
    """Translate *text* between English and Dhivehi.

    Parameters
    ----------
    conn:
        Open SQLite connection to a moonlight database.
    text:
        Source text.  Language is auto-detected; provide ``target_lang``
        explicitly to override the default (translate to the other language).
    target_lang:
        ``'EN'``, ``'DV'``, or ``None`` / ``'auto'`` to infer from source.
    llm:
        An ``anthropic.Anthropic`` client.  If *None*, one is constructed from
        ``ANTHROPIC_API_KEY`` in the environment.
    model_alias:
        One of the aliases in ``moonlight.pricing.MODELS``
        (``"sonnet"``, ``"haiku"``, ``"opus"``).
    verify:
        Run a round-trip back-translation to check for number/entity loss.
        Roughly doubles cost; surfaced via ``--verify`` in the CLI.
    exclude_article_ids:
        Article IDs excluded from few-shot + phrase-context retrieval.  Used
        by the evaluation suite (ADR 0017) to prevent ground-truth leakage
        when the test fixture *is* a corpus article.
    style_transfer:
        Run a second LLM pass to match the top exemplar's voice.  Only active
        when ``mode="po_style"``; ignored in ``"faithful"`` mode.
    mode:
        ``"faithful"`` (default) or ``"po_style"``.  See module docstring.
    ablate:
        Set of component names to disable for ablation experiments:
        ``"few_shot"``, ``"glossary"``, ``"phrase_contexts"``,
        ``"genre_routing"``, ``"term_locking"``, ``"polish"``.
        Empty set / None = full pipeline.  Ablation runs bypass the cache.
    enable_term_locking:
        Enable placeholder-based hard terminology locking.  Off by default —
        see module-level research note.
    n_candidates:
        Generate this many raw candidates and return the highest-scoring one
        (Best-of-N, C2).  Set to 1 for deterministic / cost-controlled runs.

    Returns
    -------
    dict with keys:
        ``translation``, ``source_lang``, ``target_lang``, ``exemplar_ids``,
        ``exemplars``, ``glossary_terms_used``, ``terms_locked``,
        ``lock_misses``, ``entity_check``, ``phrase_contexts``,
        ``sentence_memory_used``, ``model``, ``cost_usd``, ``cache_hit``,
        ``n_candidates``, ``mode``, ``ablate``, ``disclaimer``.
        Plus ``verification`` when *verify* is True.
    """
    if not text or not text.strip():
        raise ValueError("translate(): text is empty")

    source_lang = detect_language(text)
    if target_lang is None or target_lang == "auto":
        target_lang = "DV" if source_lang == "EN" else "EN"
    if target_lang not in ("EN", "DV"):
        raise ValueError(f"target_lang must be 'EN' or 'DV' (got {target_lang!r})")
    if target_lang == source_lang:
        raise ValueError(
            f"source and target languages are both {source_lang} — "
            "detect_language thinks the input is already in the target "
            "language. Specify target_lang explicitly to override."
        )

    # Cache lookup — skipped during ablation because the cache key doesn't
    # encode ablation flags; a cached full-pipeline result would silently
    # invalidate the ablation experiment.
    ablate = set(ablate) if ablate else set()
    requested_model_id = llm.model_id if llm is not None else resolve_model_id(model_alias)
    if not ablate:
        hit = _cache_lookup(conn, text, target_lang, model_id=requested_model_id)
        if hit is not None:
            hit["disclaimer"] = _DISCLAIMER
            hit.setdefault("n_candidates", 1)
            return hit

    # Genre classification (Step 3 of the quality roadmap): classify input
    # text by genre, then prefer within-genre exemplars when filling the
    # few-shot slot.  Genre routing improves register by giving the LLM voice
    # patterns from the right genre, not just topic-similar articles.
    if "genre_routing" not in ablate:
        input_genre = corpus.classify_genre(text, lang=source_lang)
        if input_genre:
            logger.debug("translate: classified input as genre=%s", input_genre)
    else:
        input_genre = None

    # Adaptive context budgeting (C3): exemplar count scales with input length.
    # po_style always gets k=5 (register requires more voice samples).
    # faithful mode buckets by word count: short inputs need fewer exemplars —
    # the glossary + phrase_contexts carry the terminology load; extra exemplars
    # just burn tokens and add noise.
    _wc = len(text.split())
    if mode == "po_style":
        k_few = 5
    elif _wc <= 50:
        k_few = 2
    elif _wc <= 200:
        k_few = 3
    else:
        k_few = 5

    exemplars_raw = [] if "few_shot" in ablate else corpus.select_few_shot(
        conn, source_lang, text, k=k_few,
        exclude_article_ids=exclude_article_ids,
        prefer_genre=input_genre,
    )
    exemplars = _coerce_exemplars(exemplars_raw, source_lang)
    glossary = [] if "glossary" in ablate else corpus.select_glossary_subset(
        conn, text, source_lang, max_terms=20,
    )
    phrase_contexts = [] if "phrase_contexts" in ablate else corpus.select_phrase_contexts(
        conn, text, source_lang,
        max_phrases=4, snippets_per_phrase=1,
        exclude_article_ids=exclude_article_ids,
    )

    # Sentence-level translation memory (Phase B1b).  Hybrid BM25 +
    # multilingual sentence embeddings; auto-falls back to pure BM25 if
    # sentence-transformers is unavailable or the corpus hasn't been
    # embedding-backfilled yet.
    sentence_mem: list[dict] = []
    if "sentence_memory" not in ablate:
        try:
            from moonlight import sentence_memory as _sm
            sentence_mem = _sm.select_sentence_memory_hybrid(
                conn, text,
                source_lang=source_lang, k=5,
                exclude_article_ids=exclude_article_ids,
            )
        except sqlite3.OperationalError:
            # sentence_pairs_fts may not exist on a freshly initialised DB.
            sentence_mem = []

    # Hard terminology locking (Phase B2).  Off by default — see module
    # research note at the top of this section.
    use_locking = enable_term_locking and "term_locking" not in ablate
    if not use_locking:
        term_locks = []
        locked_text = text
        lock_map = {}
    else:
        term_locks = _build_term_locks(text, source_lang, glossary)
        locked_text, lock_map = _apply_term_locks(text, term_locks)
    if term_locks:
        logger.debug("translate: locked %d terms", len(term_locks))

    # Dynamic place-name lookup (Phase B3): scan source text for Thaana
    # substrings matching the place_names table, inject exact mappings into
    # the system prompt.  Falls back to generic rule-of-thumb when the table
    # is empty or the module is unavailable.
    _place_names: list[dict] = []
    if source_lang == "DV" and target_lang == "EN":
        try:
            from moonlight.place_names import lookup_place_names_for_text
            _place_names = lookup_place_names_for_text(conn, text)
        except Exception:
            pass

    system, user = _compose_prompt(
        locked_text, source_lang, target_lang, glossary, exemplars,
        phrase_contexts=phrase_contexts,
        sentence_memory=sentence_mem,
        mode=mode,
        place_names=_place_names,
    )

    if term_locks:
        system += (
            "\n\n=== PLACEHOLDER MARKERS — DO NOT MODIFY ===\n"
            "The text to translate contains markers of the form "
            "⟦K0⟧, ⟦K1⟧, etc. These are pre-locked terms — proper "
            "nouns, institutional names, numbers, monetary amounts "
            "that MUST be preserved verbatim. Treat each marker as "
            "an opaque token: keep it EXACTLY as written. Only "
            "translate the prose around the markers."
        )

    if llm is None:
        llm = LLMClient(model_alias)

    model_id = llm.model_id

    # Best-of-N (C2): force n=1 when ablation is active so the ablation signal
    # isn't confounded by candidate selection.
    _n = 1 if (ablate or n_candidates < 1) else max(1, int(n_candidates))
    _candidates: list[tuple[str, float, int, int]] = []
    for _ in range(_n):
        _candidates.append(_single_llm_call(llm, system, user))

    if _n == 1:
        translation, cost, tokens_in, tokens_out = _candidates[0]
    else:
        _scored = [
            (_candidate_score(t, text, source_lang, target_lang), i, t, c, ti, to)
            for i, (t, c, ti, to) in enumerate(_candidates)
        ]
        _scored.sort(key=lambda x: x[0], reverse=True)
        best_score, _, translation, cost, tokens_in, tokens_out = _scored[0]
        cost = sum(c for _, c, _, _ in _candidates)
        tokens_in = sum(ti for _, _, ti, _ in _candidates)
        tokens_out = sum(to for _, _, _, to in _candidates)
        logger.debug("translate: best-of-%d selected score=%.3f", _n, best_score)

    # Style-transfer second pass — po_style only.  Disabled in faithful mode
    # to prevent the embellishment engine from introducing hallucinated claims.
    if (
        style_transfer
        and exemplars
        and mode == "po_style"
        and "polish" not in ablate
    ):
        polished, polish_cost, polish_in, polish_out = _style_transfer_polish(
            translation, exemplars,
            source_text=text, source_lang=source_lang,
            target_lang=target_lang, llm=llm,
        )
        if term_locks:
            polished_placeholders = set(_PLACEHOLDER_RE.findall(polished))
            expected_idx = {p.strip("⟦⟧K") for p in lock_map}
            if polished_placeholders != expected_idx:
                logger.warning(
                    "translate: polish dropped placeholders "
                    "(expected %s, got %s) — falling back to draft",
                    sorted(expected_idx), sorted(polished_placeholders),
                )
            else:
                if polished != translation:
                    logger.debug(
                        "style_transfer: polished %d → %d chars",
                        len(translation), len(polished),
                    )
                    translation = polished
        else:
            if polished != translation:
                logger.debug(
                    "style_transfer: polished %d → %d chars",
                    len(translation), len(polished),
                )
                translation = polished
        cost += polish_cost
        tokens_in += polish_in
        tokens_out += polish_out

    # Deterministic restore of locked terms.
    lock_misses: list[str] = []
    if lock_map:
        translation, lock_misses = _restore_term_locks(translation, lock_map)

    # Entity/numeric validator gate — deterministic post-check.
    entity_check = validate_entities(
        text, translation,
        source_lang=source_lang, target_lang=target_lang,
    )
    if not entity_check["passed"]:
        logger.warning(
            "translate: entity validator failed: %s", entity_check["summary"])

    exemplar_ids = [
        e["en_article_id"] for e in exemplars if e.get("en_article_id") is not None
    ]
    exemplar_refs = [
        {
            "en_article_id":  e.get("en_article_id"),
            "dv_article_id":  e.get("dv_article_id"),
            "title":          (e.get("en_title") or e.get("title") or "").strip(),
            "published_date": e.get("published_date", ""),
        }
        for e in exemplars
        if e.get("en_article_id") is not None
    ]
    now = datetime.now(timezone.utc).isoformat()

    # Don't persist ablation runs — they're experimental outputs.  Persisting
    # them would let a subsequent normal call return the ablated result on a
    # cache hit, silently degrading production.
    if not ablate:
        conn.execute(
            """INSERT INTO translation_runs
               (source_lang, target_lang, input_text, output_text,
                exemplar_ids, glossary_terms_used, model, cost_usd,
                created_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (source_lang, target_lang, text, translation,
             json.dumps(exemplar_ids), len(glossary), model_id, cost, now),
        )
        conn.commit()

    out = {
        "translation":          translation,
        "source_lang":          source_lang,
        "target_lang":          target_lang,
        "exemplar_ids":         exemplar_ids,
        "exemplars":            exemplar_refs,
        "glossary_terms_used":  len(glossary),
        "terms_locked":         len(term_locks),
        "lock_misses":          lock_misses,
        "entity_check":         entity_check,
        "phrase_contexts":      [
            {"phrase": c.get("phrase", ""), "article_id": c.get("article_id")}
            for c in phrase_contexts
        ],
        "sentence_memory_used": [
            {
                "source_article_id": sm["source_article_id"],
                "paired_article_id": sm["paired_article_id"],
            }
            for sm in sentence_mem
        ],
        "model":                model_id,
        "cost_usd":             cost,
        "cache_hit":            False,
        "n_candidates":         _n,
        "mode":                 mode,
        "ablate":               sorted(ablate) if ablate else [],
        "disclaimer":           _DISCLAIMER,
    }
    if verify:
        verification = verify_back_translation(
            conn, text, translation,
            source_lang=source_lang, target_lang=target_lang,
            llm=llm, model_alias=model_alias,
        )
        out["verification"] = verification
        out["cost_usd"] = out["cost_usd"] + verification["cost_usd"]
    return out


# ── Back-translation verification (opt-in) ────────────────────────────────────
#
# Translates the output *back* to the source language, then compares numbers
# and proper nouns against the original.  A "4 schools" that becomes "1 school"
# through the round trip is exactly the failure mode we want to catch.
#
# Cost: roughly doubles per-translation spend (~$0.04 vs ~$0.02).
# Not on by default; the CLI exposes it as --verify.


def verify_back_translation(
    conn: sqlite3.Connection,
    original_text: str,
    translation: str,
    *,
    source_lang: str,
    target_lang: str,
    llm: Optional["LLMClient"] = None,
    model_alias: str = "claude-sonnet",
) -> dict:
    """Round-trip semantic-preservation check.

    Translates *translation* back to *source_lang* WITHOUT few-shot (we want
    a straight back-translation to compare apples-to-apples with the source).

    Returns::

        {
          "back_translation":   str,
          "numbers_lost":       list,
          "numbers_added":      list,
          "proper_nouns_lost":  list,  # EN source only
          "proper_nouns_added": list,
          "passed":             bool,
          "cost_usd":           float,
        }
    """
    if llm is None:
        llm = LLMClient(model_alias)

    target_name = "English" if source_lang == "EN" else "Dhivehi"
    system = (
        f"Translate the given text to {target_name}. Preserve all "
        f"numbers, proper nouns, dates, and institutional names "
        f"exactly. Output the translation only — no commentary."
    )
    back, tokens_in, tokens_out = llm.chat(
        system, translation, max_tokens=2000, temperature=0.1)
    cost = llm.cost_usd(tokens_in, tokens_out)

    orig_nums = set(_NUMBER_RE.findall(original_text))
    back_nums = set(_NUMBER_RE.findall(back))
    numbers_lost = sorted(orig_nums - back_nums)
    numbers_added = sorted(back_nums - orig_nums)

    if source_lang == "EN":
        orig_pn = set(_PROPER_NOUN_RE.findall(original_text))
        back_pn = set(_PROPER_NOUN_RE.findall(back))
        stop = {
            "The", "And", "For", "But", "With", "From", "When",
            "Where", "What", "Why", "How", "Who", "This",
            "That", "These", "Those", "Today", "Yesterday",
            "Tomorrow", "Maldives", "Maldivian",
        }
        orig_pn -= stop
        back_pn -= stop
        proper_nouns_lost = sorted(orig_pn - back_pn)
        proper_nouns_added = sorted(back_pn - orig_pn)
    else:
        proper_nouns_lost = []
        proper_nouns_added = []

    passed = (
        not numbers_lost
        and not numbers_added
        and not proper_nouns_lost
        and not proper_nouns_added
    )
    return {
        "back_translation":    back,
        "numbers_lost":        numbers_lost,
        "numbers_added":       numbers_added,
        "proper_nouns_lost":   proper_nouns_lost,
        "proper_nouns_added":  proper_nouns_added,
        "passed":              passed,
        "cost_usd":            cost,
    }


# ── Glossary builder (one-shot batch job) ─────────────────────────────────────
#
# This is a corpus-mining job, not a per-translation call.  It samples paired
# articles, asks the LLM to extract bilingual term pairs from each, aggregates
# by frequency, and writes the top-N to translation_glossary.
#
# Design choices:
# - Sampling strategy: most recent paired articles.  Recency matters because
#   the PO's preferred terminology drifts over time; the most recent corpus
#   reflects current style.
# - Budget gate: stops when cumulative cost ≥ budget_usd to prevent accidental
#   large spends.  Default $10 covers ~200 pairs at ~$0.05 each.
# - Idempotent: clears previous rows extracted by the same model before writing.
#   Manual edits (extracted_by='manual') are preserved.

_GLOSSARY_BUILDER_SYSTEM = (
    "You extract bilingual term pairs from paired Maldives Presidency "
    "Office press releases. The English body and Dhivehi body describe "
    "the same content. Identify 5-12 institution names, technical "
    "terms, policy phrases, or proper nouns that appear in BOTH "
    "bodies, and pair them up. Output JSON only.\n\n"
    "Schema:\n"
    "{\"pairs\": [\n"
    "  {\"en\": \"Judicial Service Commission\", "
    "\"dv\": \"ޝަރުޢީ ޚިދުމަތާ ބެހޭ ކޮމިޝަން\", "
    "\"domain\": \"government\", \"confidence\": 0.95},\n"
    "  ...\n"
    "]}\n\n"
    "domain options: government, geography, legal, economic, "
    "diplomatic, general. confidence: 0..1 reflecting how certain "
    "you are this is the canonical PO rendering (vs an ad-hoc one). "
    "Skip generic words ('said', 'meeting', 'today') — only "
    "domain-specific terms."
)

_JSON_RE = re.compile(r'\{.*\}', re.DOTALL)


def _extract_pairs_from_article(
    llm: "LLMClient",
    en_body: str,
    dv_body: str,
    *,
    retries: int = 3,
    timeout_seconds: float = 60.0,
) -> tuple[list[dict], dict]:
    """Extract bilingual term pairs from a single paired article.

    Returns ``(pairs, cost_meta)``.  Retries on :exc:`RateLimitError`
    (exponential backoff) and transient errors.  Bodies truncated to 6 000
    chars to bound prompt size.

    Uses ``LLMClient.chat(timeout=...)`` so the same retry logic works
    regardless of which provider is in use.
    """
    import time as _time
    user = (
        f"EN BODY:\n{en_body[:6000]}\n\n"
        f"DV BODY:\n{dv_body[:6000]}\n\n"
        f"Extract paired terms as JSON per the schema."
    )
    last_err: Optional[str] = None
    for attempt in range(retries):
        try:
            text, tokens_in, tokens_out = llm.chat(
                _GLOSSARY_BUILDER_SYSTEM, user,
                max_tokens=2000, temperature=0.2, timeout=timeout_seconds,
            )
            meta = {
                "tokens_in":  tokens_in,
                "tokens_out": tokens_out,
                "cost_usd":   llm.cost_usd(tokens_in, tokens_out),
            }
            m = _JSON_RE.search(text)
            if not m:
                return [], meta
            try:
                d = json.loads(m.group(0))
            except json.JSONDecodeError:
                return [], meta
            return d.get("pairs", []), meta
        except RateLimitError:
            _time.sleep(2 ** attempt * 2)
            last_err = "rate_limit"
        except Exception as e:
            last_err = str(e)[:200]
            if attempt == retries - 1:
                break
            _time.sleep(2 ** attempt)
    logger.warning(
        "build_glossary: exhausted %d retries (%s); skipping article",
        retries, last_err,
    )
    return [], {"tokens_in": 0, "tokens_out": 0, "cost_usd": 0.0,
                "_error": last_err}


def build_glossary(
    conn: sqlite3.Connection,
    *,
    sample_size: int = 200,
    budget_usd: float = 10.0,
    llm: Optional["LLMClient"] = None,
    progress_cb=None,
    model_alias: str = "claude-sonnet",
) -> dict:
    """Mine the corpus for bilingual term pairs and populate translation_glossary.

    Samples up to *sample_size* of the most recent paired articles with
    non-empty bodies (≥ 500 characters each).  For each pair, calls the LLM
    to extract institution names, policy phrases, and proper nouns.  Aggregates
    by frequency and writes to the glossary table.

    Parameters
    ----------
    conn:
        Open moonlight database connection.
    sample_size:
        Maximum number of paired articles to process.
    budget_usd:
        Hard cap on cumulative LLM cost.  Processing stops when this is
        reached.  Default $10 covers ~200 pairs at ~$0.05 each.
    llm:
        :class:`LLMClient` instance.  If None, constructed from *model_alias*.
    progress_cb:
        Optional callable ``(processed, total, cost_usd)`` called every 10
        articles for progress reporting.
    model_alias:
        Model alias to use when *llm* is None.

    Returns
    -------
    dict:
        ``{"pairs_in_db": int, "pairs_processed": int, "cost_usd": float}``
    """
    if llm is None:
        llm = LLMClient(model_alias)

    pairs = conn.execute(
        """SELECT en.id, en.body_text, dv.body_text
           FROM articles en
           JOIN articles dv ON en.paired_id = dv.id
           WHERE en.language = 'EN' AND dv.language = 'DV'
             AND en.body_text IS NOT NULL AND en.body_text != ''
             AND dv.body_text IS NOT NULL AND dv.body_text != ''
             AND LENGTH(en.body_text) >= 500
           ORDER BY en.published_date DESC
           LIMIT ?""",
        (sample_size,),
    ).fetchall()

    aggregated: dict = {}
    cost_total = 0.0
    processed = 0
    for en_id, en_body, dv_body in pairs:
        if cost_total >= budget_usd:
            logger.info("build_glossary: budget cap reached at $%.2f", cost_total)
            break
        terms, meta = _extract_pairs_from_article(llm, en_body, dv_body)
        cost_total += meta.get("cost_usd", 0.0)
        processed += 1
        for t in terms:
            en = (t.get("en") or "").strip()
            dv = (t.get("dv") or "").strip()
            if not en or not dv:
                continue
            key = (en.lower(), dv)
            if key not in aggregated:
                aggregated[key] = {
                    "en_term":        en,
                    "dv_term":        dv,
                    "domain":         t.get("domain") or "general",
                    "confidence_max": float(t.get("confidence") or 0.5),
                    "sample_ids":     [en_id],
                    "freq":           1,
                }
            else:
                aggregated[key]["freq"] += 1
                aggregated[key]["confidence_max"] = max(
                    aggregated[key]["confidence_max"],
                    float(t.get("confidence") or 0.5),
                )
                aggregated[key]["sample_ids"].append(en_id)
        if progress_cb is not None and processed % 10 == 0:
            progress_cb(processed, len(pairs), cost_total)

    now = datetime.now(timezone.utc).isoformat()
    mid = llm.model_id
    conn.execute(
        "DELETE FROM translation_glossary WHERE extracted_by = ?", (mid,))
    rows = sorted(aggregated.values(), key=lambda r: -r["freq"])
    for r in rows:
        conn.execute(
            """INSERT INTO translation_glossary
               (en_term, dv_term, domain, freq, confidence,
                sample_en_ids, extracted_at, extracted_by)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
            (r["en_term"], r["dv_term"], r["domain"], r["freq"],
             r["confidence_max"], json.dumps(r["sample_ids"][:10]),
             now, mid),
        )
    conn.commit()
    n_in_db = conn.execute(
        "SELECT COUNT(*) FROM translation_glossary"
    ).fetchone()[0]
    if progress_cb is not None:
        progress_cb(processed, len(pairs), cost_total)
    return {
        "pairs_in_db":     n_in_db,
        "pairs_processed": processed,
        "cost_usd":        cost_total,
    }


def _has_anthropic_module() -> bool:
    """Return True if ``anthropic`` is importable.

    Used by tests that mock the LLM client and don't install the real SDK to
    decide whether to skip network-dependent assertions.
    """
    try:
        import anthropic  # noqa: F401
        return True
    except ImportError:
        return False
