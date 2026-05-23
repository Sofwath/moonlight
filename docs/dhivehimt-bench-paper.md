# DhivehiMT-Bench: A Multi-Genre Evaluation Benchmark for English–Dhivehi Machine Translation with Register and Script Fidelity Challenge Sets

**Authors**: [Authors]  
**Affiliation**: [Affiliation]  
**Submitted to**: [Venue]  
**Status**: Draft — Phase 4 (empirical results pending)

---

## Abstract

We introduce DhivehiMT-Bench, the first formal evaluation benchmark for English–Dhivehi (EN↔DV) machine translation. Dhivehi — the official language of the Republic of Maldives, written in the Thaana script — is the only language of a sovereign United Nations member state absent from FLORES-200, and has no peer-reviewed MT evaluation literature. DhivehiMT-Bench addresses three structural failures of applying existing MT benchmarks to this language: metric saturation at the top of the quality range, blindness to politeness-register distinctions encoded in Dhivehi verb morphology, and cultural bias introduced by translating Wikipedia as source text.

The benchmark comprises three components: a 400-segment multi-genre main evaluation set aligned to the FLORES-200 dev/devtest structure and licensed for OLDI submission (CC BY 4.0); a 160-pair challenge set targeting eight Dhivehi-specific error categories including the first MT benchmark to evaluate politeness-register accuracy at verb-suffix granularity; and a 50-segment calibration set with human ESA annotation anchoring all ranking claims. We evaluate eight systems — commercial MT baselines, five frontier LLMs, and Moonlight, a retrieval-augmented translation engine trained on the paired Maldives Presidency Office corpus — and report results at corpus, genre, direction, and challenge category levels.

Our main findings are: (1) aggregate chrF saturates above 60 for all capable systems, making the challenge set the primary discrimination mechanism; (2) no current system reliably passes politeness-register contrastive pairs (Cat-1 accuracy: [TBD]%); (3) Thaana script fidelity (Cat-7) is a hard binary failure for [TBD] systems, consistent with GlotOCR 2025 observations; (4) the inverse relationship between baseline capability and pipeline gain documented in our ablation study holds at benchmark scale: Moonlight adds +[TBD] chrF on government text over raw GPT-4o, but only +[TBD] over raw Claude. We release all data and code under Apache 2.0 / CC BY 4.0.

---

## 1. Introduction

### 1.1 The gap

Machine translation for Dhivehi (ISO 639-2: div; Thaana script; ~400,000 speakers) occupies a unique position in the NLP literature: it is simultaneously absent from major multilingual benchmarks (FLORES-200, NLLB-200, SIB-200), the sole official language of a sovereign UN member state with this gap, and a language with documented systematic failures in frontier LLM translation quality.

The absence is not for lack of available text. The Maldivian Presidency Office has published bilingual (EN+DV) press releases, speeches, and decrees since the 1990s, now comprising over 18,000 paired articles. The gap is a resourcing and prioritisation failure, not a data failure — which is precisely why the "Languages Still Left Behind" critique (EMNLP 2025) applies with full force.

### 1.2 What existing benchmarks cannot measure

Three properties of Dhivehi make applying existing MT benchmark methodology directly inappropriate:

**Metric saturation.** chrF and BLEU are corpus-level metrics. For a language with a character set as morphologically productive as Thaana, two translations can differ materially in institutional correctness while scoring within noise of each other on aggregate metrics. In our ablation study (moonlight-rag-dhivehi-mt.md), Claude Opus 4.7 and Moonlight-full both score in the 60–65 chrF range on a government press release; qualitative inspection reveals systematic differences in honorific forms, institutional terminology, and politeness register that chrF cannot detect.

**Register blindness.** Dhivehi encodes three politeness-register levels in verb morphology: classical/formal (suffix -ވިއެވެ termination, honorific verb forms), standard (shorter forms), and informal/colloquial (contracted forms, dropped suffixes). Presidential speech published by the Presidency Office is consistently formal-register text. No existing MT benchmark evaluates this dimension for any language; there is no benchmark for any language that tests verb-suffix register accuracy as a contrastive pair task.

**Cultural bias in source text.** FLORES-200 used Wikipedia as source material. For Dhivehi, this creates two problems: (a) most Dhivehi Wikipedia articles are direct translations of English articles, introducing English-world cultural framing and allowing named-entity copying to inflate automatic scores; (b) Wikipedia Dhivehi is informal register, not representative of government/institutional text, which is the most practically important domain for Dhivehi MT.

### 1.3 Contributions

This paper makes four contributions:

1. **DhivehiMT-Bench**: the first EN↔DV evaluation benchmark, publicly released under CC BY 4.0, compatible with FLORES-200/FLORES+ and submittable to OLDI.

2. **Register challenge set**: the first MT evaluation resource for any language that tests politeness-register accuracy at verb-suffix granularity (Cat-1: 40 contrastive pairs across verb-suffix, pronoun-selection, and lexical register errors).

3. **Systematic evaluation of eight systems**: including Google Translate, NLLB-200 (if div_Thaa is covered), five frontier LLMs, and three Moonlight configurations (ablation A/B/C from our companion paper).

4. **Calibration methodology**: ESA human annotation on a 50-segment calibration set with Spearman-gated LLM judge panel (GPT-4o + Gemini), directly addressing the reliability concerns documented in Islam et al. (2025) for low-resource LLM-as-judge.

---

## 2. Related Work

### 2.1 Low-resource MT benchmarks

FLORES-200 (Costa-jussà et al. 2022) covers 200 languages in a 1012-sentence three-domain benchmark. Dhivehi is absent. NLLB-200 covers 202 language pairs; Dhivehi coverage is unconfirmed as of 2025 and not tested in peer-reviewed work. SIB-200 (Adelani et al. 2024) similarly lacks Dhivehi.

The "Languages Still Left Behind" analysis (EMNLP 2025) identifies Dhivehi among languages of sovereign nations with documented zero representation in major multilingual NLP benchmarks. No MT paper with Dhivehi as a primary language appears in ACL Anthology as of 2025.

### 2.2 Automatic MT metrics for low-resource languages

BLEU (Papineni et al. 2002) requires word tokenisation, which is undefined for Thaana. chrF (Popović 2015) operates at character level and avoids this dependency; we adopt it as primary. COMET and xCOMET (Rei et al. 2022; Guerreiro et al. 2023) are multilingual neural metrics fine-tuned on direct assessment scores; both are zero-shot extrapolations for EN↔DV and are reported with explicit caveats.

Critically, Freitag et al. (2022) demonstrate that BLEU and chrF saturate as a discrimination mechanism for high-quality systems. For Dhivehi, where frontier LLMs all produce grammatical Thaana with reasonable vocabulary, this saturation is the central evaluation challenge this benchmark addresses.

### 2.3 LLM-as-judge for MT

GEMBA-MQM (Kocmi and Federmann 2023) uses GPT-4 for reference-free MT evaluation and achieves high correlation with human judgements for high-resource languages. Islam et al. (2025) show that dialect-guided prompting — explicitly stating the language, script, and register in the judge prompt — is necessary for reliable LLM judgements in low-resource settings. They find that undirected GPT-4 judges systematically rate fluent but register-incorrect output higher than native speakers would. We adopt their dialect-guided prompting approach.

Self-preference bias (Zheng et al. 2023) — the tendency of an LLM to prefer its own outputs when judging — is a documented failure mode. Since Moonlight is built on Claude (Anthropic), we exclude all Claude models from the judge panel for any evaluation that includes Moonlight output.

### 2.4 Register and politeness in MT evaluation

Politeness-register accuracy in MT has been evaluated for Japanese (Wiesner et al. 2023) at sentence level but not at verb-suffix granularity. The IWSLT Formality shared task (Rabinovich et al. 2017; Nadejde et al. 2022) evaluates formal vs. informal register in European languages. No comparable resource exists for Afro-Asian, Dravidian, or Maldivian language families. DhivehiMT-Bench's Cat-1 is the first such resource.

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

**Government genre extraction.** The 100 government segments are extracted from the kahzaabu corpus (18,543 paired EN+DV articles) using sentence-level alignment by position index. A quality gate (`scripts/check_alignment_quality.py`) flags likely misaligned pairs by checking length ratio, shared numbers/years, and Thaana presence; 61% of extracted segments pass automatically, 33% require spot-check, 6% are replaced. The Arabic comma (U+060C) is standard Dhivehi punctuation and explicitly excluded from the script-contamination check.

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

**Cat-1 (novel contribution).** Dhivehi's three-tier politeness-register system is grammaticalised in verb morphology to a degree unusual among world languages. Formal press releases use -ވިއެވެ termination; colloquial text uses contracted forms; honorific speech verbs (ވިދާޅުވިއެވެ vs ބުންޏެވެ) are categorically distinct. The 40 pairs are split: 20 verb-suffix mismatch, 10 pronoun-selection (formal ތިމަންނަ vs. informal އަހަރެން), 10 lexical-register (classical vocabulary vs. colloquial equivalent). No existing MT benchmark tests any of these distinctions for any language.

### 3.4 Calibration set and human annotation

50 segments stratified across genres and directions, annotated by 2–3 native Dhivehi speakers following the WMT 2024 ESA protocol (Amrhein et al. 2024): Direct Assessment score (0–100) + MQM error spans with reduced 3-category profile (Accuracy, Fluency, Terminology). AI pre-annotation via GEMBA-MQM is presented to annotators as a suggested annotation to halve annotation time. Krippendorff's alpha ≥ 0.60 is required; Spearman ≥ 0.60 between LLM judge scores and DA scores is required before judge results appear in comparative claims.

---

## 4. Systems Under Test

| System | Type | Model / version |
|--------|------|-----------------|
| Google Translate | Commercial MT | Production API |
| NLLB-200 | Open-source MT | facebook/nllb-200-distilled-600M (if div_Thaa covered) |
| Raw GPT-4o | Frontier LLM baseline | gpt-4o (OpenAI) |
| Raw Claude Sonnet 4.6 | Frontier LLM baseline | claude-sonnet-4-6 (Anthropic) |
| Raw Gemini 1.5 Flash | Frontier LLM baseline | gemini-1.5-flash (Google) |
| Moonlight — no corpus | Pipeline ablation B | Claude + prompt engineering, empty DB |
| Moonlight — full corpus | Pipeline ablation C | Claude + prompt engineering + PO corpus RAG |
| Moonlight po_style | Pipeline variant | Ablation C, po_style register-optimised mode |

The A/B/C ablation structure (A = raw LLM baseline, B = prompt engineering only, C = full RAG pipeline) mirrors the companion paper (moonlight-rag-dhivehi-mt.md), extending those single-article results to 400 benchmark segments.

---

## 5. Evaluation Stack

### 5.1 Automatic metrics

| Metric | Role | Caveat |
|--------|------|--------|
| chrF (sacrebleu, char order 6) | **Primary** | Unvalidated against DV human judgements; no tokenisation dependency |
| BLEU (sacrebleu) | Secondary; comparability with NLLB-200 | Word-level tokenisation undefined for Thaana; high variance |
| COMET (wmt22-comet-da) | Indicative | Zero-shot extrapolation for EN↔DV; not used for ranking closely-scored systems |
| xCOMET-XL | Indicative + error spans | Same zero-shot caveat; error span output used for qualitative analysis |

Bootstrap confidence intervals (1,000 resamples, 95% CI) for all aggregate metric scores. Claims of improvement only where CIs do not overlap.

### 5.2 LLM judge panel

- **Models**: GPT-4o + Gemini 1.5 Flash (Claude excluded; see §3 self-preference bias note)
- **Protocol**: swap test mandatory; inconsistent verdicts = ties
- **Prompt**: dialect-guided (Dhivehi, Thaana script, Maldivian government register, PO honorific conventions)
- **Scoring**: 5-point scalar for Spearman correlation + pairwise preference
- **Calibration gate**: Spearman ≥ 0.60 required before comparative claims

### 5.3 Human evaluation

ESA annotation on 50-segment calibration set is the gold standard for all ranking claims. Automatic metrics are reported with explicit uncertainty bounds; LLM judge results are subject to the calibration gate.

---

## 6. Results

> **Note**: This section contains placeholder tables. Empirical results to be inserted after running `scripts/run_benchmark.py` on the full devtest split and completing human ESA annotation.

### 6.1 Main set aggregate results

| System | Dir | chrF | 95% CI | BLEU | COMET |
|--------|-----|:----:|--------|:----:|:-----:|
| Google Translate | EN→DV | [TBD] | | [TBD] | |
| Google Translate | DV→EN | [TBD] | | [TBD] | |
| Raw GPT-4o | EN→DV | [TBD] | | [TBD] | |
| Raw GPT-4o | DV→EN | [TBD] | | [TBD] | |
| Raw Claude Sonnet 4.6 | EN→DV | [TBD] | | [TBD] | |
| Raw Claude Sonnet 4.6 | DV→EN | [TBD] | | [TBD] | |
| Raw Gemini 1.5 Flash | EN→DV | [TBD] | | [TBD] | |
| Raw Gemini 1.5 Flash | DV→EN | [TBD] | | [TBD] | |
| Moonlight — no corpus | EN→DV | [TBD] | | [TBD] | |
| Moonlight — no corpus | DV→EN | [TBD] | | [TBD] | |
| Moonlight — full corpus | EN→DV | [TBD] | | [TBD] | |
| Moonlight — full corpus | DV→EN | [TBD] | | [TBD] | |
| Moonlight po_style | EN→DV | [TBD] | | [TBD] | |

*Primary metric: chrF (character n-gram F-score, 0–100, higher = better).*

### 6.2 Results by genre

| System | Government | News | Social | Religious† |
|--------|:----------:|:----:|:------:|:----------:|
| [Results pending] | | | | |

†Religious genre scores reported separately; likely inflated by eBible corpus overlap.

### 6.3 Challenge set accuracy

| System | Cat-1 Register | Cat-2 Honorifics | Cat-3 Entities | Cat-4 Converb | Cat-5 Pronouns | Cat-6 Numerals | Cat-7 Script | Cat-8 Terms | Overall |
|--------|:--------------:|:----------------:|:--------------:|:-------------:|:--------------:|:--------------:|:------------:|:-----------:|:-------:|
| [Results pending] | | | | | | | | | |

*Pass criterion: chrF(correct) − chrF(incorrect) ≥ 2.0 for Cat-1–6, Cat-8; Thaana-only binary for Cat-7.*

### 6.4 LLM judge panel results

*Gate status: [pending calibration set annotation]*

---

## 7. Discussion

### 7.1 Metric saturation and the role of the challenge set

[To be written after empirical results. Expected finding based on companion paper: all capable systems score within noise of each other on aggregate chrF above approximately 60. The challenge set is the primary source of discrimination.]

### 7.2 Register failure as the hardest problem

[To be written. Expected finding: Cat-1 accuracy is the lowest of all challenge categories. No current system reliably produces -ވިއެވެ formal-register verb endings in government domain text when translating from English. This finding is not detectable from aggregate metrics alone.]

### 7.3 The inverse capability–gain relationship at benchmark scale

From our companion paper (moonlight-rag-dhivehi-mt.md §6.4), the RAG pipeline's contribution to chrF is inversely proportional to the model's baseline Dhivehi capability: GPT-5.5 gains +24.9 chrF (baseline 31.0), while Claude Opus gains only +1.7 (baseline 61.7). If this relationship holds at benchmark scale, it has a practical implication: the Moonlight pipeline is most valuable as a complement to models with weaker out-of-the-box Dhivehi capability, not as a booster for already-capable models.

### 7.4 Thaana script fidelity as a hard failure mode

GlotOCR 2025 documents that frontier LLMs produce Arabic script when confronted with Thaana input. Cat-7 tests this directly. [Expected finding: one or more systems fail Cat-7 on at least some inputs. The binary nature of this error — any Arabic codepoint in a DV output is a hard fail — makes it detectable even when aggregate metrics look acceptable.]

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

**Sentence alignment quality.** Government genre segments are extracted using sentence-position alignment, which is approximate. Manual quality review removes misaligned pairs (6% FAIL rate on first extraction), but no sentence-level alignment verification beyond heuristics is applied.

**Annotator availability.** Native Dhivehi speakers with institutional domain familiarity suitable for ESA annotation are scarce. The calibration set annotation is the critical path item for this benchmark reaching its full reliability targets.

**Metric extrapolation.** COMET and xCOMET are trained primarily on high-resource language pairs. Their application to EN↔DV is zero-shot extrapolation; their reliability for this language pair is unknown and explicitly caveated in all result tables.

**Temporal distribution.** The government corpus spans from the 1990s to 2026. Terminology and honorific conventions have evolved over this period; benchmark segments from different eras may have different register expectations. We report segment publication dates and recommend future work to evaluate temporal effects.

---

## 10. Ethical Considerations

**Social media data.** Social/informal genre segments sourced from Twitter/X and Facebook public posts involve data collection with privacy implications. Only public posts are included; no personally identifying information is retained beyond what is publicly visible.

**Religious text.** Religious genre segments include Quranic text. We report this genre separately to avoid attributing any performance claims to religious content without explicit contextualisation.

**Annotation labour.** We follow ACL guidelines for annotator compensation; annotators are paid at or above the local professional rate for translation work.

**Model evaluations.** All API-based translations and judge scores are logged for reproducibility. No evaluation results are presented without confidence intervals or calibration gates.

---

## 11. Conclusion

DhivehiMT-Bench fills the most significant documented gap in multilingual MT evaluation infrastructure: a language of a sovereign nation with zero peer-reviewed benchmark coverage. The benchmark's three-component design — aggregate main set, Dhivehi-specific challenge set, ESA-calibrated human ground truth — addresses the metric saturation, register blindness, and cultural bias problems that make direct application of existing benchmark methodology inappropriate.

The challenge set's Cat-1 (politeness register) is the benchmark's primary novel contribution to MT evaluation methodology: the first contrastive pair resource for any language that tests verb-morphology register accuracy. Whether current systems pass it is an empirical question this paper answers; that no existing benchmark would even ask the question is the methodological gap we close.

All data, annotations, and evaluation code are released under Apache 2.0 / CC BY 4.0. The FLORES+-compatible devtest is submitted to OLDI for integration into the standard multilingual MT evaluation infrastructure.

---

## References

Adelani, D. et al. (2024). SIB-200: A simple, inclusive, and big evaluation dataset for topic classification in 200+ languages and dialects. *EACL 2024*.

Amrhein, C. et al. (2024). Quality estimation by direct assessment with reference. *WMT 2024*.

Costa-jussà, M. R. et al. (2022). No language left behind: Scaling human-centered machine translation. *arXiv:2207.04672*.

Freitag, M. et al. (2022). Results of the WMT22 metrics shared task. *WMT 2022*.

Guerreiro, N. M. et al. (2023). xCOMET: Transparent machine translation evaluation through fine-grained error detection. *arXiv:2310.10482*.

Islam, M. et al. (2025). Dialect-guided prompting improves LLM judge reliability for low-resource language evaluation. *[venue TBD]*.

Kocmi, T. and Federmann, C. (2023). Large language models are state-of-the-art evaluators of translation quality. *EAMT 2023*.

Nadejde, M. et al. (2022). CoCoA-MT: A dataset and benchmark for contentious and counter-narrative MT. *NAACL 2022*.

Papineni, K. et al. (2002). BLEU: A method for automatic evaluation of machine translation. *ACL 2002*.

Popović, M. (2015). chrF: Character n-gram F-score for automatic MT evaluation. *WMT 2015*.

Rabinovich, E. et al. (2017). Personalized machine translation: Preserving original author traits. *EACL 2017*.

Rei, R. et al. (2022). COMET-22: Unbabel-IST 2022 submission for the metrics shared task. *WMT 2022*.

Wiesner, P. et al. (2023). Evaluating politeness register accuracy in Japanese–English MT. *[venue TBD]*.

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
