# DhivehiMT-Bench: Benchmark Design Specification

**Version**: 0.1 (Phase 2 — Design)  
**Project**: Moonlight / kahzaabu  
**Research question**: How should low-resource morphologically rich languages be benchmarked when automatic MT metrics fail to distinguish high-quality outputs? Dhivehi as case study.

---

## 1. Overview

DhivehiMT-Bench is the first formal evaluation benchmark for English–Dhivehi (EN↔DV) machine translation. It is designed to be:

- **Publishable** as a standalone benchmark paper, with Moonlight as the case study system
- **FLORES+-compatible** — devtest is sized and licensed for submission to the OLDI shared task
- **Register-sensitive** — the first MT benchmark for any language to evaluate politeness-register accuracy at verb-form granularity
- **Practically reproducible** — the full evaluation runs from a single script given API keys

The benchmark has three components evaluated independently:

| Component | Size | Purpose |
|-----------|------|---------|
| **Main evaluation set** | 400 segments | Aggregate quality measurement across genres |
| **Challenge set** | 160 contrastive pairs | Discrimination at the top of the quality range; Dhivehi-specific phenomena |
| **Calibration set** | 50 segments | Human ESA annotation; LLM judge calibration; inter-annotator agreement |

Total: **610 segments** across both directions.

---

## 2. Test Set Sources

### 2.1 Source Principle

Sentences must come from **native Dhivehi text**, not from English→Dhivehi translations of Wikipedia. This directly addresses the "Languages Still Left Behind" critique (EMNLP 2025): translated Wikipedia introduces English-world cultural bias and allows named-entity copying to inflate scores artificially.

### 2.2 Sources by Genre

| Genre | Source | Target count | Notes |
|-------|--------|-------------|-------|
| **Government/institutional** | presidency.gov.mv paired EN+DV corpus (kahzaabu) | 100 segments | The primary domain for Moonlight; highest register; most sensitive to honorific/terminology errors |
| **News** | edition.mv, mihaaru.com (DV originals with EN parallels where available) | 100 segments | General formal register; broad vocabulary |
| **Social/informal** | Twitter/X Dhivehi, Facebook public posts, SMS-style text | 100 segments | Tests colloquial register; exposes formality mismatch failures |
| **Religious** | Friday sermon excerpts, Quran translation passages | 100 segments | Classical register; Arabic loanword density; likely inflated for models trained on eBible corpus — reported separately |

**Total main set**: 400 segments × 2 directions = 800 translation tasks.

### 2.3 Segment Selection Criteria

- Length: 15–60 words (excludes titles and very long paragraphs)
- Both sides available: for DV→EN, the original is Dhivehi; for EN→DV, the original is English
- No overlap with the kahzaabu eval article (#29734 or the existing Moonlight eval set)
- Genre-balanced: 100 segments per genre, sampled randomly from a larger pool with manual quality check

### 2.4 FLORES+ Compatibility

The 400-segment main set is split as:
- **dev**: 200 segments (used for system development and parameter tuning)
- **devtest**: 200 segments (held out; used for all reported results)

This mirrors the FLORES-200 dev/devtest structure and makes the devtest directly submittable to OLDI. License: CC BY 4.0.

---

## 3. Challenge Set

The challenge set provides discrimination where aggregate metrics saturate. Each entry is a **contrastive pair**: a source sentence + two candidate translations, one correct and one containing a targeted error. Systems are scored by whether they produce (or prefer, in LLM judge mode) the correct variant.

### 3.1 Error Taxonomy (8 Categories)

#### Cat-1: Politeness Register (40 pairs)
The novel contribution. Dhivehi encodes three register levels:
- **Formal/classical**: Arabic-derived vocabulary, long honorific titles (ރައީސުލްޖުމްހޫރިއްޔާ), formal verb suffixes (-ވިއެވެ termination)
- **Standard**: neutral vocabulary, shorter forms (ރައީސް), standard suffixes
- **Informal/colloquial**: contracted forms, conversational vocabulary, dropped suffixes

Test design: same semantic content, correct register vs. wrong register. Each of the 40 pairs targets one of:
- Verb suffix register mismatch (20 pairs)
- Pronoun selection mismatch — formal first-person variants (e.g. އަޅުގަނޑު) vs. informal alternatives for first person (10 pairs; pending native-speaker verification)
- Lexical register mismatch — classical vs. colloquial vocabulary choice (10 pairs)

Ground truth: PO-published text is the reference for formal register; informal texts from social media for informal.

#### Cat-2: Honorifics and Titles (20 pairs)
The H.E. problem (documented in the Moonlight eval). Test cases:
- Maldivian president: ރައީސުލްޖުމްހޫރިއްޔާ (correct) vs. ހިޒް އެކްސެލެންсީ transliteration (wrong)
- Foreign heads of state: bare name (correct PO convention) vs. Thaana H.E. transliteration (wrong)
- Minister titles: formal Dhivehi ministerial title vs. transliterated English title
- VP, cabinet, judiciary — each with attested correct PO form

#### Cat-3: Named Entities — Maldivian (20 pairs)
Entities with no English training-data equivalent:
- Atoll names (ހއ. ތ. ނ. etc. — abbreviated forms vs. full names)
- Island names (Malé, Addu, Fuvahmulah — romanisation conventions)
- Islamic institutional terms (ދީނީ ކަންތައްތަކާ ގުޅޭ ވުޒާރާ vs. approximations)
- Political party names

#### Cat-4: Converb Clause-Chaining (20 pairs)
Dhivehi restricts finite verbs: subordinate clauses use converb forms (-އިގެ, -އިފައި), not finite verb forms. A translation that puts a finite verb in each clause is grammatically wrong in formal Dhivehi, though intelligible.

Test: English multi-clause sentence → correct converb chain vs. incorrect finite-verb-per-clause version.

#### Cat-5: Gender-Neutral Pronouns (20 pairs)
Direction: EN→DV  
English he/she → Dhivehi gender-neutral އެ/ހެ  
Test: English sentence with gendered pronoun → correct gender-neutral DV form vs. gender-marked calque that does not exist in Dhivehi

Direction: DV→EN  
Dhivehi gender-neutral → English translation must commit to he/she/they  
Test: DV sentence with neutral pronoun → acceptable EN pronoun choice vs. grammatically wrong EN

#### Cat-6: Numerals and Dates (20 pairs)
Dhivehi uses a mix of Arabic-Indic numerals (٠١٢٣٤٥٦٧٨٩) and Western Arabic numerals depending on register and context. PO convention uses Western Arabic numerals for dates and quantities.  
Test: correct numeral form vs. wrong script numeral; correct date format (DV: day/month/year with Dhivehi month names or Hijri calendar) vs. wrong format.

#### Cat-7: Thaana Script Fidelity (10 pairs)
Documented failure mode (GlotOCR 2025): frontier models produce Arabic when confronted with Thaana.  
Test: source DV text → correct Thaana output vs. Arabic-script output, Arabic-mixed output, or romanised output.  
This is a binary pass/fail per system, not a contrastive preference.

#### Cat-8: Institutional Terminology (10 pairs)
Terms specific to Maldivian government with attested correct translations in the PO corpus:
- ބަޖެޓް / budget vs. incorrect approximations
- ކެބިނެޓް / Cabinet (correct transliteration) vs. variants
- ދައުލަތުގެ ބަޖެޓު / state budget (full form) vs. colloquial shortening in formal context

### 3.2 Challenge Set Scoring

For contrastive pairs (Cat-1 through Cat-6, Cat-8): systems are scored by whether they produce output closer to the correct variant than the incorrect one, measured by chrF difference. A system "passes" a pair if its output's chrF against the correct variant is higher than against the incorrect variant by a margin ≥ 2.0.

For Cat-7 (script fidelity): binary — does the output contain exclusively Thaana codepoints (U+0780–U+07BF plus ASCII punctuation) with no Arabic codepoints (U+0600–U+06FF)?

**Per-category accuracy** is reported alongside aggregate chrF on the main set. This provides the discrimination that aggregate metrics cannot.

---

## 4. Calibration Set and Human Annotation

### 4.1 Purpose

The 50-segment calibration set has three functions:
1. **Ground truth** for the main evaluation — human ESA scores anchor the benchmark's validity claim
2. **LLM judge calibration** — Spearman correlation between LLM judge scores and human scores must reach ≥ 0.60 before LLM judge results appear in any comparative claim
3. **Inter-annotator agreement** — Krippendorff's alpha ≥ 0.60 threshold for the annotation to be reported as reliable

### 4.2 ESA Protocol

Following Amrhein et al. (WMT 2024), each segment receives:
- **Direct Assessment score** (0–100 continuous): overall translation quality
- **Error span annotation**: each error marked with span boundaries + MQM category

Reduced MQM profile (3 categories sufficient for this domain):
- **Accuracy**: meaning errors, omissions, additions, hallucinations
- **Fluency**: grammaticality, Thaana script correctness, morphological errors
- **Terminology**: institutional term errors, register errors, honorific errors

Severity: minor (−1), major (−5), critical (−25).

AI pre-annotation: run GEMBA-MQM on each segment before human review, present as a suggested annotation. Annotator confirms, modifies, or rejects. Halves annotation time per Amrhein et al.

### 4.3 Annotator Requirements

- 2–3 annotators
- Native or near-native Dhivehi speakers with English fluency
- Familiarity with Maldivian government/institutional context (not required to be professional translators, but should recognise PO register norms)
- Trained on the ESA protocol with 10 practice examples before the main annotation

### 4.4 Calibration Set Composition

- 15 segments from government/institutional genre
- 15 segments from news
- 10 segments from social/informal
- 10 segments from religious
- Stratified to cover both directions (25 EN→DV, 25 DV→EN)

---

## 5. Evaluation Stack

### 5.1 Automatic Metrics

| Metric | Role | Implementation | Caveat |
|--------|------|---------------|--------|
| **chrF** | Primary | `sacrebleu.corpus_chrf()`, char order 6, beta=2, default settings | No tokenisation dependency; best available for Thaana morphology; unvalidated against DV human judgements — explicit caveat in paper |
| **BLEU** | Secondary / comparability | `sacrebleu.corpus_bleu()`, SacreBLEU hash reported | Word-level tokenisation undefined for Thaana; high variance; used for comparison with NLLB-200 published scores only |
| **COMET** | Indicative | `wmt22-comet-da` | EN↔DV is zero-shot extrapolation; reported with explicit caveat; not used for ranking closely-scored systems |
| **xCOMET** | Indicative + error spans | `xcomet-xl` | Same zero-shot caveat; error span output used for qualitative analysis |

All metrics reported with **bootstrap confidence intervals** (1,000 resamples) at the 95% level. Differences between closely-ranked systems are only claimed if confidence intervals do not overlap.

### 5.2 LLM Judge

**Architecture**: pairwise preference panel

- **Judge models**: GPT-4o + Gemini 1.5 Pro (or best available). Claude is **excluded** from the judge panel for any comparison that includes Moonlight output (self-preference bias mitigation).
- **Protocol**: swap test mandatory — every pairwise comparison run twice with A/B order reversed; only consistent verdicts count; inconsistent pairs recorded as ties.
- **Prompt**: dialect-guided (following Islam et al. 2025) — the prompt explicitly states the language is Dhivehi (Thaana script, Maldivian government register), provides a one-paragraph context on the PO register norms, and lists the error categories most relevant to this domain (honorifics, register, institutional terminology).
- **Scoring**: 5-point scalar per output (not just pairwise preference) to enable both ranking and correlation with human ESA scores.
- **Calibration gate**: Spearman ≥ 0.60 against human ESA scores on the 50-segment calibration set before any LLM judge result is reported in a comparative claim. If this threshold is not met, LLM judge results are reported in a separate appendix section marked as exploratory.

### 5.3 Human Ground Truth

ESA annotation on 50-segment calibration set is the gold standard. All main claims (system rankings, Moonlight improvement claims) are validated against human scores, not just automatic metrics.

---

## 6. Systems Under Test

| System | Type | Why included |
|--------|------|-------------|
| **Google Translate** | Commercial MT | Strongest publicly available baseline; known to include DV |
| **NLLB-200 (if DV covered)** | Open-source MT | The only published quantitative MT results for Dhivehi; mandatory comparison point |
| **Raw GPT-4o** | Frontier LLM baseline (A) | Establishes frontier LLM quality without any domain adaptation |
| **Raw Claude Sonnet 4.6** | Frontier LLM baseline (A) | Anthropic baseline; same model family as Moonlight |
| **Raw Gemini 1.5 Flash** | Frontier LLM baseline (A) | Google baseline |
| **Moonlight — no corpus** | Pipeline (B) | Isolates prompt engineering contribution |
| **Moonlight — full corpus** | Pipeline (C) | Full RAG pipeline; the case study system |
| **Moonlight po_style** | Pipeline variant | Tests register-optimised mode against faithful mode |

The A/B/C ablation structure mirrors the existing Moonlight eval methodology, extended to the full benchmark dataset.

---

## 7. Statistical Methodology

### 7.1 Significance Testing

- **Metric scores**: bootstrap resampling (1,000 samples, segment-level), 95% CI. Claims of improvement only where CIs do not overlap.
- **Challenge set accuracy**: McNemar's test for paired binary outcomes (correct/incorrect per pair per system).
- **Human annotation**: Krippendorff's alpha for inter-annotator agreement. Pearson/Spearman for LLM judge calibration.

### 7.2 Reporting

All results reported at:
- **Corpus level** (aggregate over 400-segment main set)
- **Genre level** (100 segments per genre)
- **Direction level** (EN→DV and DV→EN separately)
- **Challenge set level** (per-category accuracy, 8 categories)

This produces a rich result table that allows readers to identify where each system fails, not just its aggregate rank.

---

## 8. FLORES+ Contribution

The 200-segment devtest is formatted to be directly compatible with FLORES-200/FLORES+:

- 3-domain split: news (Wikinews-style), encyclopaedic (Wikijunior-style), travel/culture (Wikivoyage-style)
- Professional translation quality (verified by inter-annotator agreement ≥ 0.60)
- License: CC BY 4.0
- OLDI submission: devtest submitted to the Open Language Data Initiative as the formal Dhivehi FLORES+ addition

This gives the paper a community contribution beyond the research findings — a reusable resource for any future Dhivehi MT work.

---

## 9. Open Questions for Phase 3 (Implementation)

The following decisions require implementation-time validation:

1. **Annotator recruitment**: identifying 2–3 qualified Dhivehi-English bilingual annotators with institutional domain familiarity. This is the hardest practical constraint.

2. **Social media source access**: edition.mv and mihaaru.com are freely accessible; Twitter/Facebook Dhivehi content requires scraping decisions and licensing considerations.

3. **NLLB-200 DV coverage**: confirm whether NLLB-200 supports div_Thaa at inference time. If not, Google Translate becomes the sole commercial MT baseline.

4. **Challenge set construction tooling**: the contrastive pair format requires a lightweight annotation interface. Options: Label Studio (open source), a custom script writing pairs to JSONL, or manual Google Sheets with export.

5. **LLM judge calibration threshold**: if Spearman < 0.60 on the calibration set, the fallback is: (a) report LLM judge as exploratory, (b) expand calibration set to 100 segments, (c) restrict LLM judge claims to categories where calibration is above threshold.

---

## 10. Deliverables

| Deliverable | Format | Phase |
|-------------|--------|-------|
| This design spec | `docs/benchmark-design-spec.md` | Phase 2 ✓ |
| Main evaluation set (400 segments) | JSONL, CC BY 4.0 | Phase 3 |
| Challenge set (160 pairs) | JSONL with error category labels | Phase 3 |
| ESA annotation tool + guidelines | Python script + PDF | Phase 3 |
| Annotated calibration set (50 segments) | JSONL with ESA spans | Phase 3 |
| Evaluation harness | `scripts/run_benchmark.py` | Phase 4 |
| LLM judge implementation | `scripts/llm_judge.py` | Phase 4 |
| Benchmark paper | `docs/dhivehimt-bench-paper.md` | Phase 4 |
| FLORES+ devtest submission | OLDI-format JSONL | Phase 4 |
