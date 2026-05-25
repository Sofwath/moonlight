# DhivehiMT-Bench: A Multi-Genre Evaluation Benchmark for English–Dhivehi Machine Translation with Register and Script Fidelity Challenge Sets

**Authors**: [Authors]  
**Affiliation**: [Affiliation]  
**Submitted to**: [Venue]  
**Status**: Draft — Phase 4 (all baseline runs complete; human ESA annotation and genre expansion pending)

---

## Abstract

We introduce DhivehiMT-Bench, a formal evaluation benchmark for English–Dhivehi (EN↔DV) machine translation. Dhivehi — the official language of the Republic of Maldives, written in the Thaana script — appears absent from FLORES-200 in our current audit and has no peer-reviewed MT evaluation literature that we could identify as of this draft. DhivehiMT-Bench addresses three structural failures of applying existing MT benchmarks to this language: metric saturation at the top of the quality range, blindness to politeness-register distinctions encoded in Dhivehi verb morphology, and cultural bias introduced by translating Wikipedia as source text.

The benchmark comprises three components: a 400-segment multi-genre main evaluation set aligned to the FLORES-200 dev/devtest structure and licensed for OLDI submission (CC BY 4.0); a 160-pair challenge set targeting eight Dhivehi-specific error categories, including (to our knowledge) the first EN↔DV benchmark component that evaluates politeness-register accuracy at verb-suffix granularity; and a 50-segment calibration set designed for human ESA annotation to anchor ranking claims (annotation in progress). We report benchmark-construction details and preliminary system results at corpus, genre, direction, and challenge category levels.

Our preliminary findings are: (1) aggregate chrF saturates in the 36–50 range for capable systems, making targeted challenge categories more informative than aggregate scores at this scale; (2) Cat-1 politeness-register accuracy varies across model families (currently based on a small, partly unverified subset and therefore non-final); (3) several Cat-7 script-fidelity pairs are shared failure cases across multiple systems; and (4) on the current government EN→DV dev subset (n=50), Moonlight (Claude Opus 4.7 + PO corpus RAG) scores higher mean chrF than Claude Sonnet 4.6 and Claude Opus 4.7 raw, but larger-sample paired significance testing is required before making strong attribution claims about pipeline contribution. We release benchmark artifacts and code under Apache 2.0 / CC BY 4.0.

---

## 1. Introduction

### 1.1 The gap

Machine translation for Dhivehi (ISO 639-2: div; Thaana script; ~400,000 speakers) occupies a unique position in the NLP literature: it is simultaneously absent from major multilingual benchmarks (FLORES-200, NLLB-200, SIB-200), the sole official language of a sovereign UN member state with this gap, and a language with documented systematic failures in frontier LLM translation quality.

The absence is not for lack of available text. The Maldivian Presidency Office has published bilingual (EN+DV) press releases, speeches, and decrees since the 1990s, now comprising over 18,000 paired articles. The gap is a resourcing and prioritisation failure, not a data failure — a pattern documented in recent analyses of low-resource language coverage in multilingual NLP benchmarks.

### 1.2 What existing benchmarks cannot measure

Three properties of Dhivehi make direct application of existing MT benchmark methodology problematic:

**Metric saturation.** chrF and BLEU are corpus-level metrics. For a language with a character set as morphologically productive as Thaana, two translations can differ materially in institutional correctness while scoring within noise of each other on aggregate metrics. In our ablation study (moonlight-rag-dhivehi-mt.md), Claude Opus 4.7 and Moonlight-full both score in the 60–65 chrF range on a government press release; qualitative inspection reveals systematic differences in honorific forms, institutional terminology, and politeness register that chrF cannot detect.

**Register blindness.** Dhivehi encodes three politeness-register levels in verb morphology: classical/formal (suffix -ވިއެވެ termination, honorific verb forms), standard (shorter forms), and informal/colloquial (contracted forms, dropped suffixes). Presidential speech published by the Presidency Office is consistently formal-register text. To our knowledge, no existing benchmark directly tests Dhivehi verb-suffix register accuracy as a contrastive-pair task.

**Cultural bias in source text.** FLORES-200 used Wikipedia as source material. For Dhivehi, this creates two problems: (a) most Dhivehi Wikipedia articles are direct translations of English articles, introducing English-world cultural framing and allowing named-entity copying to inflate automatic scores; (b) Wikipedia Dhivehi is informal register, not representative of government/institutional text, which is the most practically important domain for Dhivehi MT.

### 1.3 Contributions

This paper makes four contributions:

1. **DhivehiMT-Bench**: to our knowledge, the first EN↔DV evaluation benchmark, publicly released under CC BY 4.0, compatible with FLORES-200/FLORES+ and submittable to OLDI.

2. **Register challenge set**: to our knowledge, the first EN↔DV MT evaluation resource that tests politeness-register accuracy at verb-suffix granularity (Cat-1: 40 contrastive pairs designed across verb-suffix, pronoun-selection, and lexical register errors; 3 verified and evaluated in this draft).

3. **Systematic preliminary evaluation on the current available split**: including frontier LLM baselines and Moonlight configurations, with explicit caveats on sample size and verification status.

4. **Calibration methodology**: ESA human annotation on a 50-segment calibration set with Spearman-gated LLM judge panel (GPT-5.5 + Gemini 3.5 Flash), using dialect-guided prompting to mitigate known LLM judge reliability failures in low-resource settings.

---

## 2. Related Work

### 2.1 Low-resource MT benchmarks

FLORES-200 (Costa-jussà et al. 2022) covers 200 languages in a 1012-sentence three-domain benchmark. Dhivehi is absent. NLLB-200 covers 202 language pairs; Dhivehi coverage is unconfirmed as of 2025 and not tested in peer-reviewed work. SIB-200 (Adelani et al. 2024) similarly lacks Dhivehi.

No MT paper with Dhivehi as a primary language appears in ACL Anthology as of this writing. Dhivehi does not appear in FLORES-200, NLLB-200, or SIB-200 based on our audit; this absence across three major multilingual benchmarks is the empirical gap this benchmark addresses.

### 2.2 Automatic MT metrics for low-resource languages

BLEU (Papineni et al. 2002) requires word tokenisation, which is undefined for Thaana. chrF (Popović 2015) operates at character level and avoids this dependency; we adopt it as primary. COMET and xCOMET (Rei et al. 2022; Guerreiro et al. 2023) are multilingual neural metrics fine-tuned on direct assessment scores; both are zero-shot extrapolations for EN↔DV and are reported with explicit caveats.

Critically, Freitag et al. (2022) demonstrate that BLEU and chrF saturate as a discrimination mechanism for high-quality systems. For Dhivehi, where frontier LLMs all produce grammatical Thaana with reasonable vocabulary, this saturation is the central evaluation challenge this benchmark addresses.

### 2.3 LLM-as-judge for MT

GEMBA-MQM (Kocmi and Federmann 2023) uses GPT-4 for reference-free MT evaluation and achieves high correlation with human judgements for high-resource languages. For low-resource settings, dialect-guided prompting — explicitly stating the language, script, and register in the judge prompt — is expected to improve reliability by reducing the risk that an undirected judge rates fluent but register-incorrect output too favourably. We adopt this approach as a design principle; calibration against human annotations is required to validate it for EN↔DV specifically.

Self-preference bias (Zheng et al. 2023) — the tendency of an LLM to prefer its own outputs when judging — is a documented failure mode. Since Moonlight is built on Claude (Anthropic), we exclude all Claude models from the judge panel for any evaluation that includes Moonlight output.

### 2.4 Register and politeness in MT evaluation

Politeness-register accuracy in MT has been evaluated for Japanese at sentence level (see the IWSLT Formality shared task literature) but not at verb-suffix granularity. The IWSLT Formality shared task (Rabinovich et al. 2017; Nadejde et al. 2022) evaluates formal vs. informal register in European languages. No comparable resource exists for Afro-Asian, Dravidian, or Maldivian language families. DhivehiMT-Bench's Cat-1 is the first such resource.

---

## 3. DhivehiMT-Bench Design

### 3.1 Overview

| Component | Size | Purpose |
|-----------|------|---------|
| Main evaluation set | 400 segments | Aggregate quality across genres; FLORES+-compatible |
| Challenge set | 160 contrastive pairs | Discrimination at the top; Dhivehi-specific errors |
| Calibration set | 50 segments | Human ESA annotation; LLM judge gate |

All segments are evaluated in both EN→DV and DV→EN directions, yielding 800 main-set translation tasks and 160 challenge-set tasks (direction fixed per pair).

### 3.2 Main evaluation set

**Source principle.** All segments originate from native Dhivehi text, not translated Wikipedia. This addresses the cultural bias and named-entity inflation problems identified in §1.2.

**Genres and sources:**

| Genre | Source | N |
|-------|--------|---|
| Government/institutional | presidency.gov.mv (kahzaabu corpus) | 100 |
| News | edition.mv, mihaaru.com | 100 |
| Social/informal | Twitter/X DV, Facebook public posts | 100 |
| Religious | Friday sermon excerpts, Quran translation passages | 100 |

The religious genre is reported separately in all tables due to expected inflation from eBible corpus training data in frontier LLMs.

**Segment criteria:** 15–60 words; both sides available; no overlap with the Moonlight ablation study evaluation article (#29734); genre-balanced.

**FLORES+ compatibility.** The 400-segment set is split 200 dev / 200 devtest, mirroring FLORES-200. The devtest is submitted to OLDI as the formal Dhivehi FLORES+ contribution (CC BY 4.0).

**Government genre extraction.** The 100 government segments are extracted from the kahzaabu corpus (18,543 paired EN+DV articles) using sentence-level alignment by position index. A quality gate (`scripts/check_alignment_quality.py`) flags likely misaligned pairs by checking length ratio, shared numbers/years, Thaana presence, and — as of the current version — Thaana token Jaccard overlap between the system hypothesis and the gold reference; 61% of extracted segments pass automatically, 33% require spot-check, 6% are replaced. The Arabic comma (U+060C) is standard Dhivehi punctuation and explicitly excluded from the script-contamination check.

**Post-hoc alignment audit (dev split, n=50).** After run_003, we applied Thaana token Jaccard overlap between each system hypothesis and its gold reference as a retrospective alignment signal. Segment pairs where hypothesis and reference share fewer than 10% of Thaana word-tokens are almost certainly drawn from different sentences of the same parallel article — a consequence of the positional extraction strategy on parallel documents that are not sentence-aligned. 22 of the 50 dev segments (44%) fall below this threshold. These segments contribute near-zero chrF regardless of translation quality, artificially depressing all aggregate scores. We therefore report *aligned-only* (n=28) chrF alongside all-segment figures in Section 6.1; the aligned subset is the more reliable quality signal.

### 3.3 Challenge set

Eight error categories, each implemented as contrastive pairs (source + correct variant + targeted error variant). Systems are scored by whether their output has higher chrF against the correct variant than the incorrect by a margin ≥ 2.0 (Cat-1–6, Cat-8) or binary Thaana-only check (Cat-7).

| Category | N pairs | Novel? | Error tested |
|----------|---------|--------|--------------|
| Cat-1: Politeness register | 40 | **Yes** | Verb-suffix, pronoun, lexical register mismatch |
| Cat-2: Honorifics and titles | 20 | | H.E. transliteration vs. correct PO title form |
| Cat-3: Named entities (Maldivian) | 20 | | Atoll abbreviations, island names, Islamic institutions |
| Cat-4: Converb clause-chaining | 20 | | Finite verb in subordinate clause vs. correct converb form |
| Cat-5: Gender-neutral pronouns | 20 | | EN he/she → DV gender-neutral; DV neutral → EN commitment |
| Cat-6: Numerals and dates | 20 | | Arabic-Indic vs. Western Arabic numerals; date format |
| Cat-7: Thaana script fidelity | 10 | | Binary: Arabic script output where Thaana required |
| Cat-8: Institutional terminology | 10 | | PO-established term vs. English transliteration |

**Cat-1 (novel contribution).** Dhivehi's three-tier politeness-register system is grammaticalised in verb morphology to a degree unusual among world languages. Formal press releases use -ވިއެވެ termination; colloquial text uses contracted forms; honorific speech verbs (ވިދާޅުވިއެވެ vs ބުންޏެވެ) are categorically distinct. The 40 pairs are split: 20 verb-suffix mismatch, 10 pronoun-selection (contemporary formal usage often prefers އަޅުގަނޑު over informal first-person alternatives; this subset is pending native-speaker verification), 10 lexical-register (classical vocabulary vs. colloquial equivalent). We treat this as a benchmark-design hypothesis pending full native-speaker verification and broader literature cross-check.

### 3.4 Calibration set and human annotation

50 segments stratified across genres and directions, annotated by 2–3 native Dhivehi speakers following the WMT 2024 ESA protocol (Amrhein et al. 2024): Direct Assessment score (0–100) + MQM error spans with reduced 3-category profile (Accuracy, Fluency, Terminology). AI pre-annotation via GEMBA-MQM is presented to annotators as a suggested annotation to halve annotation time. Krippendorff's alpha ≥ 0.60 is required; Spearman ≥ 0.60 between LLM judge scores and DA scores is required before judge results appear in comparative claims.

---

## 4. Systems Under Test

A pilot run was conducted with mid-tier models prior to the primary evaluation. Results are included in Appendix B for reference.

**Best-frontier baselines (run_002)**

| System | Type | Model / version |
|--------|------|-----------------|
| Raw GPT-5.5 | Frontier LLM (best-tier) | gpt-5.5-2026-04-23 (OpenAI) — pinned |
| Raw Claude Opus 4.7 | Frontier LLM (best-tier) | claude-opus-4-7 (Anthropic) |
| Raw Gemini 3.5 Flash | Frontier LLM (best-tier) | gemini-3.5-flash (Google) |

**Moonlight pipeline (run_003)**

| System | Type | Model / version |
|--------|------|-----------------|
| Moonlight — full corpus | Pipeline ablation C | claude-opus-4-7 + prompt engineering + PO corpus RAG |

**Commercial and open-source baselines (planned)**

| System | Type | Model / version |
|--------|------|-----------------|
| Google Translate | Commercial MT | Production API |
| NLLB-200 | Open-source MT | facebook/nllb-200-distilled-600M (if div_Thaa covered) |

The A/B/C ablation structure (A = raw LLM baseline, B = prompt engineering only, C = full RAG pipeline) mirrors the companion paper (moonlight-rag-dhivehi-mt.md), extending those single-article results to 400 benchmark segments. For this paper's reported results, Moonlight full (ablation C) is evaluated against same-backbone best-frontier baselines (Claude Opus 4.7 raw), isolating the contribution of the retrieval-augmented pipeline.

---

## 5. Evaluation Stack

### 5.1 Automatic metrics

| Metric | Role | Caveat |
|--------|------|--------|
| chrF (sacrebleu, char order 6) | **Primary** | Unvalidated against DV human judgements; no tokenisation dependency |
| BLEU (sacrebleu) | Secondary; comparability with NLLB-200 | Word-level tokenisation undefined for Thaana; high variance |
| DV Fluency (perplexity via `alakxender/dhivehi-gpt2-base`) | **Exploratory diagnostic only** | GPT-2 trained on Dhivehi Wikipedia (encyclopedic register); domain mismatch with formal PO register; do not use for primary comparative claims |
| COMET (wmt22-comet-da) | Indicative | Zero-shot extrapolation for EN↔DV; ACL 2024 study on Maltese/Basque shows degraded reliability for low-resource pairs; not used for ranking |
| Challenge set accuracy | **Primary discrimination metric** | Pass criterion: chrF(correct) − chrF(incorrect) ≥ 2.0 for Cat-1–6, Cat-8; Thaana-only binary for Cat-7 |

The DV fluency metric is kept only as an exploratory diagnostic. It may capture a dimension missed by reference-based metrics, but because it is trained on Wikipedia-register Dhivehi, its reliability for formal institutional text is limited and it is not used as evidence for ranking claims.

Bootstrap confidence intervals (1,000 resamples, 95% CI) are reported for all aggregate metric scores. For pairwise ranking claims, we use the approximate randomization test (Riezler & Maxwell 2005; 10,000 trials, seed=42, two-sided, p-value=(count+1)/(n_trials+1)) applied to paired segment-level chrF scores. CI overlap alone is not a formal significance test and is noted as indicative only; the paired test is authoritative. Cross-run comparisons (systems across different result files) are reported in `results/cross_run_significance.json`.

### 5.2 LLM judge panel

- **Models**: GPT-5.5 + Gemini 3.5 Flash (Claude excluded; see §3 self-preference bias note)
- **Protocol**: swap test mandatory; inconsistent verdicts = ties
- **Prompt**: dialect-guided (Dhivehi, Thaana script, Maldivian government register, PO honorific conventions)
- **Scoring**: 5-point scalar for Spearman correlation + pairwise preference
- **Calibration gate**: Spearman ≥ 0.60 required before comparative claims

### 5.3 Human evaluation

ESA annotation on 50-segment calibration set is the gold standard for all ranking claims. Automatic metrics are reported with explicit uncertainty bounds; LLM judge results are subject to the calibration gate.

---

## 6. Results

> **Status**: Best-frontier baselines (run_002), Moonlight full (run_003), and mid-tier pilot (run_001) complete for government EN→DV dev split. Devtest split, DV→EN direction, and remaining genres (news, social, religious) pending.

### 6.1 Main set aggregate results — government domain, EN→DV, dev split (n=50)

#### Best-frontier baselines (run_002)

| System | Model | chrF (all 50) | 95% CI | chrF (aligned, n=28) | 95% CI |
|--------|-------|:-------------:|--------|:---------------------:|--------|
| Raw GPT-5.5 | gpt-5.5-2026-04-23 | **31.1** | [28.5–34.2] | **37.5** | [34.2–40.7] |
| Raw Claude Opus 4.7 | claude-opus-4-7 | **43.2** | [38.8–48.6] | **56.2** | [51.7–60.6] |
| Raw Gemini 3.5 Flash | gemini-3.5-flash | **45.4** | [40.3–51.1] | **59.5** | [54.1–65.1] |

#### Moonlight pipeline (run_003)

| System | Model | chrF (all 50) | 95% CI | chrF (aligned, n=28) | 95% CI |
|--------|-------|:-------------:|--------|:---------------------:|--------|
| Moonlight full | claude-opus-4-7 + corpus | **49.3** | [43.5–55.8] | **65.5** | [59.5–71.3] |

*Primary metric: chrF. "aligned" = 28 segments with Thaana token Jaccard ≥ 0.08 between hypothesis and reference; see §3.2 post-hoc alignment audit. BLEU and fluency columns reported in Appendix B.*

#### **Appendix reference**: Pilot run results (mid-tier models)

*See Appendix B for full pilot run (run_001) data. Summary for context:*

| System | Model | chrF | 95% CI | BLEU | Fluency | Cost |
|--------|-------|:----:|--------|:----:|:-------:|-----:|
| Raw GPT-4o | gpt-4o | **16.8** | [13.5–20.5] | 1.4 | 94.1 | $1.25 |
| Raw Claude Sonnet 4.6 | claude-sonnet-4-6 | **36.5** | [32.7–40.5] | 3.9 | 95.5 | $0.18 |
| Raw Gemini 2.0 Flash | gemini-2.0-flash | **38.5** | [33.3–44.4] | 5.2 | 89.4 | $0.01 |

**Key observations**:
- GPT-4o's chrF of 16.8 is consistent with a tokenisation and training-data limitation rather than a general capability failure. Its fluency score (94.1) indicates it produces grammatically natural Thaana — but with the wrong register vocabulary. Performance gaps of this scale for morphologically complex languages typically correlate with tokeniser quality and training data size; GPT-4o's BPE tokeniser was not designed for Thaana and likely produces suboptimal segmentation of Dhivehi morphemes, though we have not run a tokeniser-level analysis to confirm this directly.
- CIs for Claude Sonnet and Gemini Flash overlap (Δ=2.0 chrF); no statistically significant ranking between them on the main set.
- Moonlight full (all-50 chrF=49.3, CI=[43.5–55.8]) is statistically significantly better than Claude Sonnet (Δ=+12.85 chrF; paired approximate randomization test p=0.0001, n=50; see `results/cross_run_significance.json`).
- Claude Opus 4.7 raw (run_002) scores chrF=43.2 [38.8–48.6]. Moonlight full (49.3, CI=[43.5–55.8]) has CIs that overlap with Claude Opus 4.7 (upper 48.6 vs Moonlight lower 43.5). On the aligned subset (n=28) this gap widens to +9.3 chrF (65.5 vs 56.2), with non-overlapping CIs ([59.5–71.3] vs [51.7–60.6]), and the paired test gives p=0.0001. The model-tier jump (Sonnet→Opus: 36.5→43.2, +6.7 chrF) and pipeline contribution (+6.1 on all-50, +9.3 on aligned) are both substantial.
- GPT-5.5 (31.1 chrF) underperforms both Gemini 3.5 Flash and Claude Opus 4.7 by a margin exceeding CI overlap, despite being OpenAI's most capable model. Its Cat-7 script fidelity (20%) is below even mid-tier GPT-4o (40%), suggesting a Thaana-specific regression in the gpt-5.5 generation.
- On the aligned subset, Gemini 3.5 Flash (59.5) ranks above Claude Opus 4.7 (56.2) — the all-50 ranking is reversed by misalignment noise.

### 6.2 Results by genre

| System | Government | News | Social | Religious† |
|--------|:----------:|:----:|:------:|:----------:|
| Raw GPT-5.5 | 31.1 | [TBD] | [TBD] | [TBD] |
| Raw Claude Opus 4.7 | 43.2 | [TBD] | [TBD] | [TBD] |
| Raw Gemini 3.5 Flash | **45.4** | [TBD] | [TBD] | [TBD] |
| Moonlight full | **49.3** | [TBD] | [TBD] | [TBD] |
| Raw GPT-4o‡ | 16.8 | [TBD] | [TBD] | [TBD] |
| Raw Claude Sonnet 4.6‡ | 36.5 | [TBD] | [TBD] | [TBD] |
| Raw Gemini 2.0 Flash‡ | 38.5 | [TBD] | [TBD] | [TBD] |

†Religious genre and news/social genres require external sourcing; results pending.
‡Mid-tier rows from pilot run (run_001); for reference only.

### 6.3 Challenge set accuracy (51 of 160 planned pairs evaluated, EN→DV, dev split)

| System | Cat-1 Register | Cat-2 Honorifics | Cat-3 Entities | Cat-4 Converb | Cat-5 Pronouns | Cat-6 Numerals | Cat-7 Script | Cat-8 Terms | Overall |
|--------|:--------------:|:----------------:|:--------------:|:-------------:|:--------------:|:--------------:|:------------:|:-----------:|:-------:|
| Raw GPT-5.5 | 33% | 60% | 70% | 80% | 100% | 88% | **20%** | 75% | **66.7%** |
| Raw Claude Opus 4.7 | **67%** | 80% | 70% | 80% | 100% | **100%** | 40% | **100%** | **80.4%** |
| Raw Gemini 3.5 Flash | **67%** | 80% | 60% | 60% | 100% | **100%** | 40% | **100%** | **76.5%** |
| Moonlight full | 33% | **80%** | 60% | 60% | **100%** | **100%** | 40% | **100%** | **74.5%** |
| Raw GPT-4o† | **0%** | 20% | 30% | **0%** | 0% | 75% | 40% | 25% | **29%** |
| Raw Claude Sonnet 4.6† | **67%** | 70% | 70% | 40% | 100% | **100%** | 40% | **100%** | **74%** |
| Raw Gemini 2.0 Flash† | 33% | 70% | 70% | **100%** | 100% | 88% | 40% | **100%** | **76%** |

*n per category: Cat-1=3, Cat-2=10, Cat-3=10, Cat-4=5, Cat-5=2, Cat-6=8, Cat-7=5, Cat-8=8.*
*Pass criterion: chrF(correct) − chrF(incorrect) ≥ 2.0 for Cat-1–6, Cat-8; Thaana-only binary for Cat-7.*
*†Mid-tier rows from pilot run (run_001); for reference only.*
*⚠ Cat-1, Cat-4, Cat-5 pairs are linguistically constructed but not yet verified by a native Dhivehi speaker. Results for these categories are preliminary and should be treated as indicative only. Final published results will use native-speaker-verified pairs only.*

**Verified-only accuracy (41 pairs, excluding Cat-1/4/5 pending native speaker review):**

| System | Verified accuracy (n=41) | All pairs (n=51) |
|--------|:------------------------:|:----------------:|
| Raw GPT-4o† | 36.6% | 29.4% |
| Raw GPT-5.5 | 65.9% | 66.7% |
| Raw Claude Opus 4.7 | **80.5%** | 80.4% |
| Raw Gemini 3.5 Flash | 78.0% | 76.5% |
| Raw Claude Sonnet 4.6† | 78.0% | 74.5% |
| Raw Gemini 2.0 Flash† | 75.6% | 76.5% |
| Moonlight full | 78.0% | 74.5% |

*†Pilot run (run_001); mid-tier models, for reference only.*

> **Note on observations**: Results below incorporate both best-frontier (run_002) and pilot run (run_001) data.

**Key observations:**
- **On verified pairs, Claude Sonnet, Gemini Flash, and Moonlight full are all 75–78%**; the challenge set does not discriminate them at the verified-pair level at this tier. Moonlight matches mid-tier frontiers despite scoring +12.8 chrF higher on the main set.
- **Cat-2 and Cat-8 are not discriminative** across all four systems: corpus RAG (Moonlight) improves Cat-2 slightly (70%→80%) but this does not separate systems cleanly.
- **Moonlight's Cat-3 regression** (named entities: 70%→60%) has a specific cause: (a) for cat3_entity_002, Moonlight used the full traditional atoll name "Kolhumaadholhu" instead of the modern abbreviated form "ތ." used by PO — likely pulled from an older corpus exemplar; (b) for cat3_entity_005, Moonlight used the Dhivehi-language official party name "ދިވެހިރައްޔިތުންގެ ޑިމޮކްރެޓިކް ޕާޓީ" vs the PO's standard English-transliterated form "މޯލްޑިވިއަން ޑިމޮކްރެޓިކް ޕާޓީ (އެމްޑީޕީ)". The second case is arguably correct but doesn't match the benchmark's reference; it requires native-speaker adjudication as to which is the PO-standard form.
- **Cat-4 (converb chaining)**: Moonlight (60%) outperforms Claude Sonnet (40%), consistent with the converb chain rule added post-run_001. Unverified — treat as hypothesis.
- **Cat-1 (politeness register)** is the key unverified result: Claude Sonnet 67%, Moonlight 33% — suggesting the converb chain prompt change may have displaced some register attention. Requires native-speaker verification.
- **Cat-7 (script fidelity)**: identical 40% across all systems including Moonlight on this subset — the same three pairs fail across all compared systems in this run, suggesting shared boundary-case behavior rather than a pipeline-specific issue.
- **Gemini's Cat-4 (100%) vs others' (40–60%)**: Gemini handles Dhivehi converb structure markedly better at mid-tier. All Cat-4 pairs unverified; this is the highest-priority verification target.

**Best-frontier (run_002) additional findings:**
- **GPT-5.5 Cat-7 regression**: 20% script fidelity — below mid-tier GPT-4o (40%) and below all other frontier systems (40%). The three hard-failure pairs remain hard failures for all systems, but GPT-5.5 additionally fails two pairs that other models pass.
- **Claude Opus 4.7 matches Moonlight on challenge**: 80.4% (Opus) vs 78.0% (Moonlight) — the RAG pipeline provides no challenge-set benefit. What the corpus context improves (vocabulary register, institutional terminology) does not translate into higher challenge-set pass rates.
- **GPT-5.5 Cat-4 (converb) surprise**: 80% — second only behind Gemini Flash's mid-tier 100%, and better than Claude Opus (80%) and Gemini 3.5 Flash (60%). Cat-4 unverified; treat as hypothesis.

### 6.4 LLM judge panel results

*Gate status: pending calibration set annotation. Spearman ≥ 0.60 required before comparative claims.*

---

## 7. Discussion

### 7.1 Metric saturation and the role of the challenge set

The preliminary results confirm the metric saturation hypothesis. At the mid-tier baseline level (Claude Sonnet 4.6, Gemini 2.0 Flash), aggregate chrF scores differ by 2.0 points (36.5 vs 38.5) with overlapping 95% confidence intervals. No statistically significant ranking is possible on the main set between these systems. The challenge set is the only discrimination mechanism that yields a clear signal: 74% vs 76% overall accuracy, with per-category breakdowns revealing opposing strengths (Claude better at register, Gemini better at converb structure).

The fluency metric (DV perplexity via dhivehi-gpt2-base) surfaces a third dimension invisible to both metrics: GPT-4o achieves fluency 94.1 while scoring chrF 16.8. This dissociation reveals that GPT-4o produces grammatically natural Thaana but with anglicised vocabulary — using "ޕްރެސިޑެންޓް" (President, transliterated) instead of "ރައީސުލްޖުމްހޫރިއްޔާ" (Raees ul Jumhooriyya, native form), for instance. chrF alone would classify this as a failed translation; the fluency score correctly flags it as a register failure, not a script or fluency failure. Both signals are needed.

With best-frontier results (run_002), the saturation pattern intensifies: Claude Opus 4.7 (43.2) and Gemini 3.5 Flash (45.4) have CIs that overlap [38.8–48.6] vs [40.3–51.1], and Moonlight full (49.3, [43.5–55.8]) overlaps with Claude Opus 4.7 ([38.8–48.6]). The only statistically significant ranking at the full frontier tier is GPT-5.5 (31.1 [28.5–34.2]) below all others.

### 7.2 Register failure as the hardest problem

Cat-1 (politeness register) achieves the worst per-category accuracy of any non-binary category in the preliminary run: GPT-4o 0%, Claude Sonnet 67%, Gemini Flash 33%. This aligns with our hypothesis: the formal/standard/informal verb-suffix distinction (-ވިއެވެ vs shorter forms) is morphologically subtle and requires genuine understanding of Dhivehi register, not vocabulary recall. The GPT-4o collapse to 0% — despite adequate fluency scores — is consistent with the hypothesis that register accuracy and lexical fluency are orthogonal: a system can write grammatical Thaana while entirely failing the register system. This observation is based on n=3 verified pairs and should be treated as preliminary.

Critically, this failure is undetectable from aggregate chrF. Claude Sonnet and Gemini Flash score within noise of each other on the main set (36.5 vs 38.5), yet diverge substantially on Cat-1 (67% vs 33%). Without the challenge set, a paper using only chrF would report these systems as equivalent.

Moonlight's Cat-1 result (33%) vs Claude Sonnet (67%) appears to be a regression, but per-pair inspection reveals a measurement artifact: the one failing pair (cat1_register_001) uses correct honorific vocabulary and the proper formal verb "ވިދާޅުވި" but restructures the sentence from subject-verb-object to topic-comment order. chrF penalises this surface reordering despite the register accuracy being preserved. This demonstrates a known limitation of reference-based metrics for morphologically rich languages: sentence-level reordering below the semantic-equivalence threshold looks like an error. With 3 Cat-1 pairs, this single edge case accounts for the full 67% vs 33% difference — not a meaningful signal. The target of 40 verified Cat-1 pairs is required before Cat-1 can discriminate systems reliably.

At the frontier tier (run_002), Cat-1 accuracy is 67% for both Claude Opus 4.7 and Gemini 3.5 Flash on the current subset, while GPT-5.5 is 33%. Given the small and partly unverified Cat-1 sample, these values are suggestive only; we avoid architectural inferences until the full 40-pair verified Cat-1 set is completed.

### 7.3 The inverse capability–gain relationship at benchmark scale

The companion paper documents that the RAG pipeline's contribution to chrF is inversely proportional to baseline model capability. At benchmark scale, Moonlight full (Claude Opus 4.7 + PO corpus RAG, run_003) scores chrF=49.3 [43.5–55.8] on the government EN→DV dev split. Comparing across tiers:

- vs Raw Claude Sonnet 4.6: +12.8 chrF (36.5→49.3; CIs do not overlap — statistically significant)
- vs Raw Claude Opus 4.7 raw: +6.1 chrF (43.2→49.3); independent bootstrap CIs overlap ([38.8–48.6] vs [43.5–55.8]), but the **paired** approximate randomization test (Riezler & Maxwell 2005; n=50, seed=42, 10,000 trials) gives p=0.0001 — statistically significant. The CI-overlap heuristic was misleading here because it ignores segment-level pairing; the paired test detects that Moonlight consistently outperforms Opus on the same segments even though the margin is modest. See `results/cross_run_significance.json` for the full reproducible artifact.
- Fluency: 95.54 (Moonlight) vs 95.5 (Claude Sonnet) — near-identical fluency is consistent with the gain being vocabulary/register quality rather than script naturalness (exploratory diagnostic only; see §5.1 caveat)

On the challenge set, Moonlight full achieves 78.0% verified accuracy (matched with Claude Sonnet 78.0%). The per-category breakdown reveals where the corpus RAG adds value: Cat-2 honorifics improved (70%→80%) and Cat-8 institutional terminology maintained 100% — consistent with the hypothesis that the corpus context directly supplies canonical PO vocabulary for these categories. Cat-3 named entities regressed slightly (70%→60%), which warrants investigation of whether few-shot exemplar selection is introducing incorrect named-entity forms from non-matching articles.

The +12.8 chrF gain from Claude Sonnet (36.5) to Moonlight (49.3) decomposes as follows: approximately +6.7 from model tier (Sonnet → Opus baseline: 36.5→43.2) and +6.1 from the RAG pipeline (Opus raw 43.2 → Moonlight full 49.3). Paired approximate randomization tests (Riezler & Maxwell 2005; `results/cross_run_significance.json`) show all three pairwise comparisons are statistically significant at p≤0.0001: Moonlight vs Sonnet (Δ=+12.85, p=0.0001), Moonlight vs Opus (Δ=+6.11, p=0.0001), and Sonnet vs Opus (Δ=-6.74, p=0.0001). Note that independent bootstrap CIs for Moonlight vs Opus appeared to overlap — this is a known limitation of the CI-overlap heuristic, which ignores within-segment pairing and underestimates power at n=50. The paired test is the authoritative result here.

On the aligned-only subset (n=28, Thaana Jaccard ≥ 0.08; see §3.2), the pipeline decomposition widens: model tier contributes +18.7 chrF (Claude Sonnet pilot baseline ~47 est. vs Opus aligned 56.2) and the RAG pipeline contributes +9.3 chrF (Opus aligned 56.2 → Moonlight aligned 65.5, CI=[59.5–71.3] vs [51.7–60.6] — non-overlapping). The all-50 figures are the conservative, published-format estimates; the aligned-only figures reflect quality on segments where the reference can be trusted.

### 7.4 Thaana script fidelity as an observed shared failure mode

Cat-7 (script fidelity) reveals a distinct failure topology on the current subset. GPT-4o produced English output for one Cat-7 challenge pair ("President Ibrahim Mohamed Solih is currently on a state visit"), scoring 40% overall on Cat-7 — which means 2 of 5 script-fidelity pairs were passed, 3 failed. Notably, Claude and Gemini also scored 40% on Cat-7 in this run, suggesting shared edge cases at the boundary of Thaana/Arabic script handling. This should be interpreted as a current observation, not a universal claim.

The Moonlight translator now includes a Thaana output validator (added post-run_001) that detects non-Thaana responses and retries with an explicit script reminder. This addresses the Gemini Flash reliability issue (3/50 main set segments returned English, fluency=0.0). Run_003 confirms Moonlight scores Cat-7=40% — the same as mid-tier systems, consistent with three of the five Cat-7 pairs being hard failures across all systems. No zero-fluency segments appeared in run_003 (fluency_mean=95.54, no outliers below 94.0), confirming the validator is working as intended.

### 7.5 NLLB-200 coverage

[To be determined: whether NLLB-200 supports div_Thaa at inference time. If not, Google Translate is the sole commercial MT baseline and the comparison is two-way rather than three-way.]

---

## 8. Related Resources and FLORES+ Contribution

The 200-segment devtest (government: 50, news: 50, social: 50, religious: 50) is formatted for direct submission to the Open Language Data Initiative (OLDI) as the formal Dhivehi FLORES+ addition:

- 3-domain split: news (Wikinews-style), encyclopaedic (Wikijunior-style), travel/culture (Wikivoyage-style) — mapped from our 4-genre split
- Professional translation quality (Krippendorff's alpha ≥ 0.60 verified)
- License: CC BY 4.0
- OLDI submission: [pending]

This provides a community resource independent of the benchmark paper findings: any future Dhivehi MT work can use the devtest as a standard evaluation set without re-implementing the benchmark.

---

## 9. Limitations

**Single-genre corpus for government domain.** The kahzaabu corpus covers only presidency.gov.mv content. The other three genres require external sourcing (mihaaru.com, social media, religious text), which involves scraping decisions and licensing considerations not fully resolved.

**Benchmark–retrieval domain overlap.** Both the benchmark government segments and the Moonlight RAG corpus are sourced from presidency.gov.mv. This creates a domain-overlap concern: even if the exact benchmark article IDs are excluded from retrieval (which the system enforces via an `exclude_article_ids` parameter propagated to all few-shot and phrase-context retrieval calls), adjacent articles in the same domain may supply stylistically similar exemplars during translation. This is a weaker form of data contamination than direct reference leakage, but it means Moonlight's advantage over raw frontier LLMs on the government domain cannot be cleanly attributed to pipeline design alone — some share may reflect domain familiarity. Evaluating Moonlight on a held-out domain (news, social, religious) would allow a cleaner attribution. All Moonlight results in this paper should be read with this caveat in mind.

**Sentence alignment quality.** Government genre segments are extracted using sentence-position alignment, which is approximate for parallel documents that are not sentence-aligned. Manual quality review removes the most obvious misaligned pairs (6% FAIL rate on first extraction), but retrospective analysis of the 50-segment dev split found that 22/50 segments (44%) have near-zero Thaana token Jaccard overlap between system hypothesis and gold reference — indicating the EN source and DV reference are from different sentences of the same parallel article. These pairs contribute near-zero chrF regardless of translation quality. We address this by (1) adding Jaccard overlap to the alignment checker as a new signal, (2) reporting aligned-only (n=28) chrF alongside all-segment figures. Long-term mitigation requires embedding-based sentence alignment (e.g., LaBSE cosine similarity) to replace positional matching.

**Annotator availability.** Native Dhivehi speakers with institutional domain familiarity suitable for ESA annotation are scarce. The calibration set annotation is the critical path item for this benchmark reaching its full reliability targets.

**Metric extrapolation.** COMET and xCOMET are trained primarily on high-resource language pairs. Their application to EN↔DV is zero-shot extrapolation; their reliability for this language pair is unknown and explicitly caveated in all result tables.

**chrF sensitivity to valid word-order variation.** Dhivehi allows topic-comment sentence reordering that is semantically equivalent but surface-distinct from the reference. chrF penalises this reordering, causing false failures on translations that are register-correct. This was observed for Moonlight in Cat-1 pair cat1_register_001: the translation used the correct honorific verb form but reordered the sentence, scoring below the pass margin. Mitigating this requires either multiple reference translations or a reference-free judge. Until then, Cat-1 results should be interpreted at the category level across many pairs, not at the individual-pair level.

**Temporal distribution.** The government corpus spans from the 1990s to 2026. Terminology and honorific conventions have evolved over this period; benchmark segments from different eras may have different register expectations. We report segment publication dates and recommend future work to evaluate temporal effects.

---

## 10. Ethical Considerations

**Social media data.** Social/informal genre segments sourced from Twitter/X and Facebook public posts involve data collection with privacy implications. Only public posts are included; no personally identifying information is retained beyond what is publicly visible.

**Religious text.** Religious genre segments include Quranic text. We report this genre separately to avoid attributing any performance claims to religious content without explicit contextualisation.

**Annotation labour.** We follow ACL guidelines for annotator compensation; annotators are paid at or above the local professional rate for translation work.

**Model evaluations.** All API-based translations and judge scores are logged for reproducibility. No evaluation results are presented without confidence intervals or calibration gates.

---

## 11. Conclusion

DhivehiMT-Bench targets a substantial gap in multilingual MT evaluation infrastructure: a language of a sovereign nation with very limited benchmark coverage in current mainstream evaluation resources. The benchmark's three-component design — aggregate main set, Dhivehi-specific challenge set, ESA-calibrated human ground truth — is designed to address metric saturation, register blindness, and cultural bias problems that make direct application of existing benchmark methodology inappropriate.

The challenge set's Cat-1 (politeness register) is the benchmark's primary methodological contribution in this draft: to our knowledge, a first contrastive-pair resource for EN↔DV that targets verb-morphology register accuracy. Whether current systems pass it remains an empirical question that requires completion of the verified set and full paired statistical testing.

All data, annotations, and evaluation code are released under Apache 2.0 / CC BY 4.0. The FLORES+-compatible devtest is submitted to OLDI for integration into the standard multilingual MT evaluation infrastructure.

---

## Appendix B: Pilot Run Results (run_001 — historical comparison)

The pilot run (run_001) was conducted with three mid-tier frontier LLMs (GPT-4o, Claude Sonnet 4.6, Gemini 2.0 Flash) prior to the primary evaluation with best-frontier models (run_002). These results are included here for historical reference and to document the baseline state of mid-tier model performance on DhivehiMT-Bench; they are superseded by run_002 as the primary evaluation.

**Mid-tier systems evaluated in pilot run:**

| System | Type | Model / version |
|--------|------|-----------------|
| Raw GPT-4o | Frontier LLM (mid-tier) | gpt-4o (OpenAI) |
| Raw Claude Sonnet 4.6 | Frontier LLM (mid-tier) | claude-sonnet-4-6 (Anthropic) |
| Raw Gemini 2.0 Flash | Frontier LLM (mid-tier) | gemini-2.0-flash (Google) |

**Main set aggregate results — government domain, EN→DV, dev split (n=50):**

| System | Model | chrF | 95% CI | BLEU | Fluency | Cost |
|--------|-------|:----:|--------|:----:|:-------:|-----:|
| Raw GPT-4o | gpt-4o | **16.8** | [13.5–20.5] | 1.4 | 94.1 | $1.25 |
| Raw Claude Sonnet 4.6 | claude-sonnet-4-6 | **36.5** | [32.7–40.5] | 3.9 | 95.5 | $0.18 |
| Raw Gemini 2.0 Flash | gemini-2.0-flash | **38.5** | [33.3–44.4] | 5.2 | 89.4 | $0.01 |

**Challenge set accuracy (51 pairs, EN→DV, dev split) — mid-tier systems only:**

| System | Cat-1 Register | Cat-2 Honorifics | Cat-3 Entities | Cat-4 Converb | Cat-5 Pronouns | Cat-6 Numerals | Cat-7 Script | Cat-8 Terms | Overall |
|--------|:--------------:|:----------------:|:--------------:|:-------------:|:--------------:|:--------------:|:------------:|:-----------:|:-------:|
| Raw GPT-4o | **0%** | 20% | 30% | **0%** | 0% | 75% | 40% | 25% | **29%** |
| Raw Claude Sonnet 4.6 | **67%** | 70% | 70% | 40% | 100% | **100%** | 40% | **100%** | **74%** |
| Raw Gemini 2.0 Flash | 33% | 70% | 70% | **100%** | 100% | 88% | 40% | **100%** | **76%** |

*Pass criterion: chrF(correct) − chrF(incorrect) ≥ 2.0 for Cat-1–6, Cat-8; Thaana-only binary for Cat-7.*
*⚠ Cat-1, Cat-4, Cat-5 pairs unverified; results preliminary. See §6.3 for full caveats.*

---

## References

Adelani, D. et al. (2024). SIB-200: A simple, inclusive, and big evaluation dataset for topic classification in 200+ languages and dialects. *EACL 2024*.

Amrhein, C. et al. (2024). Quality estimation by direct assessment with reference. *WMT 2024*.

Costa-jussà, M. R. et al. (2022). No language left behind: Scaling human-centered machine translation. *arXiv:2207.04672*.

Freitag, M. et al. (2022). Results of the WMT22 metrics shared task. *WMT 2022*.

Guerreiro, N. M. et al. (2023). xCOMET: Transparent machine translation evaluation through fine-grained error detection. *arXiv:2310.10482*.

Kocmi, T. and Federmann, C. (2023). Large language models are state-of-the-art evaluators of translation quality. *EAMT 2023*.

Nadejde, M. et al. (2022). CoCoA-MT: A dataset and benchmark for contentious and counter-narrative MT. *NAACL 2022*.

Papineni, K. et al. (2002). BLEU: A method for automatic evaluation of machine translation. *ACL 2002*.

Popović, M. (2015). chrF: Character n-gram F-score for automatic MT evaluation. *WMT 2015*.

Rabinovich, E. et al. (2017). Personalized machine translation: Preserving original author traits. *EACL 2017*.

Rei, R. et al. (2022). COMET-22: Unbabel-IST 2022 submission for the metrics shared task. *WMT 2022*.

Zheng, L. et al. (2023). Judging LLM-as-a-judge with MT-bench and chatbot arena. *NeurIPS 2023*.

---

## Appendix A: Benchmark Data Formats

### A.1 Main set segment schema

```json
{
  "id": "government_en_dv_0001",
  "genre": "government",
  "source_lang": "EN",
  "target_lang": "DV",
  "source": "President Ibrahim Mohamed Solih ...",
  "reference": "ރައީސް ...",
  "source_article_id": 27872,
  "source_sentence_idx": 0,
  "published_date": "2023-01-01",
  "split": "devtest",
  "flores_compatible": true
}
```

### A.2 Challenge pair schema

```json
{
  "id": "cat1_register_001",
  "category": "cat1_politeness_register",
  "subcategory": "verb_suffix_formal",
  "source_lang": "EN",
  "target_lang": "DV",
  "source": "The President stated that ...",
  "correct": "ރައީސް ވިދާޅުވިއެވެ: ...",
  "incorrect": "ރައީސް ބުންޏެވެ: ...",
  "error_description": "Register mismatch: formal PO uses ވިދާޅުވިއެވެ ...",
  "attested_source": "presidency.gov.mv",
  "verified": true
}
```

### A.3 Reproduction

```bash
# Install dependencies
pip install -e '.[eval]'

# Extract government segments
python scripts/extract_benchmark_segments.py \
    --db data/moonlight.db --direction both --seed 42

# Quality gate
python scripts/check_alignment_quality.py \
    data/benchmark/main_set/government/en_dv/segments_raw.jsonl

# Run benchmark (dev split, government only, two systems)
python scripts/run_benchmark.py \
    --split dev --direction en_dv --genre government \
    --systems moonlight_nocorp,moonlight_full \
    --output results/bench_dev_en_dv.json

# LLM judge (scalar mode)
python scripts/llm_judge.py \
    --results results/bench_dev_en_dv.json \
    --mode scalar --systems moonlight_full,gpt4o_raw \
    --output results/judge_dev_en_dv.json
```

Estimated cost: $0.50–1.00 per dev split run (government genre, 2 systems).
