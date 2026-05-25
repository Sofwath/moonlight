# Literature Review: Evaluating Machine Translation for Dhivehi in the Frontier LLM Era

**Sofwathullah Mohamed**  
Independent Researcher — Moonlight Project  
Phase 1 of: *How should low-resource morphologically rich languages be benchmarked when automatic MT metrics fail to distinguish high-quality outputs?*

---

## 1. Scope and Motivation

This review supports the design of a benchmark for English–Dhivehi (EN↔DV) machine translation evaluation. Dhivehi is the national language of the Republic of Maldives (~500,000 speakers), written in Thaana script (Unicode U+0780–U+07BF), morphologically rich and agglutinative, with a documented three-tier register system (informal / standard / classical-formal) that is critical in government and institutional contexts.

The motivating observation is that standard MT evaluation methodology breaks down at the intersection of three properties simultaneously present in the EN↔DV use case:

1. **Low-resource**: no Dhivehi data in any WMT shared task; absent from FLORES-200, NLLB-200, and SIB-200
2. **Morphologically rich**: suffix chains encode tense, aspect, politeness, and grammatical relations; word-level metrics penalise valid variation
3. **Metric saturation**: frontier LLMs (GPT-5.5, Claude Opus 4.7, Gemini 3.5 Flash) already produce outputs in the 60–70 chrF range, where standard metrics lose discrimination power

The review is organised into four areas that directly constrain benchmark architecture: (§2) automatic metrics and their failure modes; (§3) LLM-as-judge methodology; (§4) existing Dhivehi NLP work and the low-resource evaluation literature; (§5) challenge set and register evaluation methodology.

---

## 2. Automatic Metrics: Failure Modes and What Survives

### 2.1 BLEU

**Papineni et al. (2002)** introduced BLEU as a corpus-level precision metric using modified n-gram overlap. It was never claimed to work at segment level or under single-reference conditions — both of which apply to any Dhivehi benchmark.

**Callison-Burch, Osborne & Koehn (2006)** demonstrated that improving BLEU is neither necessary nor sufficient for improved translation quality; the metric diverges from human adequacy and fluency judgements in documented counterexamples.

**Reiter (2018)** conducted a meta-analysis of 284 correlations across 34 papers, concluding that evidence supports BLEU only for diagnostic MT evaluation, not scientific hypothesis testing, segment-level evaluation, or tasks outside the narrow MT setting for which it was designed.

**Post (2018)** identified tokenisation variance of up to 1.8 BLEU points across common preprocessing pipelines and introduced SacreBLEU to standardise reporting. Any benchmark reporting BLEU must use SacreBLEU with an explicit hash string.

**The WMT22 consensus (Freitag et al., 2022)** is the current field position: "Stop Using BLEU — Neural Metrics Are Better and More Robust." Across all four test domains and all language pairs in WMT22, neural metrics substantially outperform BLEU in correlation with MQM human annotations.

**Dhivehi-specific failure**: BLEU operates on whitespace-delimited tokens. Thaana script has no established NLP tokeniser; standard BPE tokenisers (SentencePiece, tiktoken) heavily over-segment Dhivehi text due to its underrepresentation in training data. The result is tokenisation-dependent BLEU scores that are not reproducible across systems and that heavily penalise morphological surface variation in cases where the meaning is correct. BLEU remains reported for comparability with prior work (including NLLB-200, which uses spBLEU as its primary metric) but cannot serve as the primary discriminator in an EN↔DV benchmark.

### 2.2 chrF

**Popović (2015)** introduced chrF as character n-gram F-score (precision + recall of character n-grams up to order 6, beta=2 by default). It correlates better with human rankings than BLEU, particularly for morphologically rich target languages.

**Popović (2016)** confirmed that chrF2 with character order 6 is the most consistently well-performing variant across language pairs.

Four properties make chrF the best available automatic metric for Dhivehi:

1. **No tokenisation required** — operates on raw Unicode strings; avoids all the Thaana tokenisation problems that affect BLEU
2. **Partial credit for morphological variation** — a suffix chain that differs by one character scores proportionally rather than as a complete miss; correct for Dhivehi's agglutinative morphology
3. **Right behaviour on script** — character-level scoring handles Thaana code points naturally
4. **Reproducible via sacrebleu** — stable hash, no preprocessing decisions

**Khomsi et al. (2025)** specifically compared chrF++ and BLEU for extremely low-resource language pairs (Magahi, Bhojpuri, Chhattisgarhi) and found: neither metric alone is reliable; their *divergence* is diagnostic — high chrF + low BLEU often indicates source copying; low chrF + non-zero BLEU indicates hallucination. The recommendation is to report both and discuss divergence rather than selecting one.

**Conclusion**: chrF is the primary automatic metric for EN↔DV. BLEU is reported as a secondary metric for comparability. Both are presented with the explicit caveat that neither has been validated against human judgements for this language pair.

### 2.3 COMET and xCOMET

**Rei et al. (2020)** introduced COMET as a neural learned metric fine-tuned on WMT Direct Assessment human judgement data. It substantially outperforms BLEU at segment level across all WMT17–WMT20 language pairs.

**COMET-22 (Rei et al., 2022)** achieved best-in-class at WMT22 by training on MQM annotations. Critical limitation: MQM fine-tuning data covers only three language pairs — Chinese→English, English→German, English→Russian. All other pairs receive a generic model extrapolated from those three. Dhivehi has never appeared in any WMT shared task.

**xCOMET (Guerreiro et al., 2024)** extends COMET with span-level error annotation, achieving state-of-the-art on WMT22 and WMT23. The same training data restriction applies.

**The documented low-resource failure**: **Falcão et al. (2024)** studied COMET on English–Maltese and Spanish–Basque — both low-resource language pairs with limited NLP infrastructure. Finding: COMET performance degrades substantially; it is highly susceptible to the distribution of training scores, which is skewed in low-resource scenarios. Fine-tuning on even small amounts of target-language human judgement data significantly recovers performance. Maltese is an Indo-Semitic language with ~500,000 speakers — the closest documented structural analogue to Dhivehi in the literature.

**SSA-COMET (Adelani et al., 2025)** showed that domain-adapted COMET (trained on 73,000 sentences across 14 African language pairs) substantially outperforms base COMET and is competitive with GPT-4o/Claude/Gemini prompting for those languages. This is the design roadmap for a future Dhivehi-specific neural metric.

**Domain mismatch compounding**: **Zouhar et al. (2024)** showed that fine-tuned metrics drop significantly on domains not seen during training. WMT data is predominantly news; press-release / government text is underrepresented. EN↔DV governmental text faces both a language-pair gap and a domain gap simultaneously.

**Conclusion**: COMET and xCOMET scores for EN↔DV are reported as indicative only, with explicit caveats that Dhivehi is outside their training distribution. They are not used as primary metrics or for ranking closely-scored systems.

### 2.4 BLEURT

**Sellam et al. (2020)** introduced BLEURT, a two-stage BERT-based metric with pre-training on synthetic perturbations followed by WMT human judgement fine-tuning. State-of-the-art across WMT Metrics Shared Tasks.

**Sellam et al. (2020b)** extended BLEURT to 14 language pairs; zero-shot performance for languages outside those 14 degrades markedly. Dhivehi falls in the zero-shot regime. mBERT's Dhivehi coverage in pretraining is minimal. BLEURT is not used in this benchmark as a primary or secondary metric.

### 2.5 Metric Saturation

**Agrawal et al. (2024)** provide the clearest empirical documentation: standard Pearson/Kendall correlation methods measure a metric's ability to separate good translations from bad ones — not its ability to discriminate among high-quality alternatives. When all systems produce high-quality output, metrics are insensitive to residual nuanced differences. Frontier LLMs all cluster in the high-quality region.

This is directly observable in the Moonlight evaluation (this project): chrF scores across Claude Opus 4.7, Gemini 3.5 Flash, and GPT-5.5 in the DV→EN direction cluster at 61–64 for all three conditions. Bootstrap confidence intervals at this range would not reliably separate the models.

The appropriate design response (from **Kocmi et al., 2024**): challenge sets with known hard errors provide more reliable discrimination than aggregate metrics near the top of the quality distribution.

**WMT24 (Freitag et al., 2024)** confirms: LLM outputs cluster at the high end of the quality distribution, reducing metric discrimination power, even though metrics themselves remain valid for covered language pairs.

---

## 3. LLM-as-Judge: Promises and Hard Constraints

### 3.1 The Foundational Paper and Its Biases

**Zheng et al. (2023)** established that GPT-4 as a judge agrees with human preferences at >80% on the MT-Bench and Chatbot Arena tasks, matching inter-human agreement. But they documented three systematic biases:

- **Position bias**: the judge disproportionately favours the first response in a pairwise prompt
- **Verbosity bias**: the judge favours longer responses regardless of accuracy (an RLHF artefact)
- **Self-preference**: when judging its own outputs vs. a competitor, the model rates its own higher

Proposed mitigations: (a) **swap test** — run every pairwise comparison twice with A/B order reversed, accept only consistent verdicts; (b) chain-of-thought before scoring; (c) panel judging across model families.

### 3.2 GEMBA: LLM Judge for MT

**Kocmi & Federmann (2023a)** introduced GEMBA, achieving state-of-the-art correlation with WMT22 MQM ratings. Validated on three language pairs only: EN↔DE, EN↔RU, ZH→EN. No low-resource coverage.

**GEMBA-MQM (Kocmi & Federmann, 2023b)** extended GEMBA to span-level MQM error annotation using three-shot prompting. Authors explicitly caution against using it to claim improvements over other methods because GPT-4 is a black-box model.

**Critical constraint**: GEMBA's validated correlation numbers do not transfer to Dhivehi. The language pair is entirely outside its training and validation set. GEMBA is used as an *additional* signal in this benchmark, explicitly not as a primary metric and not without human calibration.

### 3.3 Self-Preference Bias

**Panickssery et al. (2024)** showed that GPT-4 and Llama 2 can identify their own outputs with non-trivial accuracy and rate them higher, even when humans rate outputs as equivalent.

**Pan et al. (2024)** confirmed self-preference persists on adversarial examples where the model's output is objectively worse.

**Xu et al. (2025)** concluded the effect is bias, not a quality signal — the model is not detecting something genuinely better about its outputs.

**Direct constraint for this benchmark**: the Moonlight translator calls Claude. Using Claude as the primary judge creates the exact self-preference condition. The judge must be from a different model family (GPT-4o or Gemini), or a panel of both, with Claude excluded from the judge panel for any comparison that includes Moonlight.

### 3.4 Cross-Lingual Judging Reliability

**Hada et al. (2025)** tested LLM judges across 25 languages and 5 tasks. Key findings: average Fleiss' Kappa ≈ 0.3 ("fair", below the 0.4 "moderate" threshold); low-resource languages score significantly lower than high-resource ones; increasing model scale or multilingual pretraining does not directly improve cross-lingual judgment consistency; prompt scaffolding with step-by-step instructions provides the largest improvement (~0.05–0.10 Kappa). **Core conclusion: "LLMs are not yet reliable for evaluating multilingual predictions."**

Dhivehi is not among the 25 tested languages, but its profile (low-resource, non-Latin script, minimal web presence) places it in the worst-performing regime.

**Sindhujan et al. (2025)** showed that for low-resource language pairs, encoder-based fine-tuned QE outperforms prompted LLM evaluation. LLM failure modes include: incorrect tokenisation of low-resource text, transliteration errors, and named entity handling failures — exactly the error types most relevant to Dhivehi governmental text.

**Islam et al. (2025)** studied Sylheti (low-resource, own script, minimal LLM training data) — the most analogous published case to Dhivehi. Finding: dialect-guided prompting (providing dialect-specific context in the evaluation prompt) yields +0.1083 Spearman correlation improvement over baseline LLM judging. For Dhivehi, this means providing PO register context, Thaana script description, and institutional terminology norms in the judge prompt — directly applicable from the Moonlight project's existing style rules.

### 3.5 Judge Calibration Requirement

The literature is consistent: LLM judge results for a new language pair cannot be trusted without a calibration set of human-annotated examples in that language. Minimum: 40–50 DV↔EN segments with native-speaker DA or MQM ratings. Target: Spearman ≥ 0.60. This is a prerequisite before any LLM judge results appear in comparative claims in the benchmark paper.

### 3.6 Pairwise vs. Scalar Scoring

**Chiang et al. (2024)** showed that Bradley-Terry (BT) model estimation of pairwise comparisons is more stable than online Elo and produces proper confidence intervals. But BT requires many comparisons per pair to stabilise; at small N (< 100 pairs), pairwise with swap test is more reliable than BT/Elo.

For the Dhivehi benchmark with 2–5 systems under test, pairwise judgement with swap test is the correct design. BT ranking is applicable only if the eval set grows to several hundred comparisons per pair.

---

## 4. Dhivehi NLP: State of the Field

### 4.1 The Gap

**Dhivehi is the only language of a sovereign nation absent from FLORES-200, NLLB-200, and SIB-200**, and the only such language with no representation in any WMT shared task. This is the primary motivating gap for the benchmark paper.

### 4.2 Existing Work

**Ibrahim (2014)** — Dhivehi OCR using Tesseract, ~69.46% character accuracy on printed Thaana. Documents the foundational scarcity problem: historical content exists in printed form, not digitised.

**GlotOCR Bench (2025)** — across 100+ Unicode scripts, 94% score below 10% OCR accuracy on frontier models, with Thaana explicitly identified as a failure case: models confronted with Thaana may produce Arabic. This is a live failure mode for LLM processing of Dhivehi inputs.

**Swarthmore LING073 project (2020)** — the only published MT system for Dhivehi. Built Apertium RBMT and OpenNMT Transformer systems, trained on Biblical text, UDHR, and "some modern phrases." NMT achieved 88.17% WER / 32.02% PER. No BLEU reported. Identified failure modes: honorifics, verb tense forms, pronoun system. **Critical implication**: the only parallel data used was domain-mismatched religious text; LLMs benchmarked on governmental Dhivehi text are operating in effectively zero-shot territory.

**alakxender (HuggingFace, 2024)** — the most significant recent work. GPT-2 (0.1B) fine-tuned on Dhivehi Wikipedia, average perplexity 3.80. Part of a 27-model Dhivehi NLP collection including text classifiers, sentiment analysers, translation experiments, and speech models. The largest active Dhivehi NLP effort. The dhivehi.ai project (same developer) documents OCR datasets.

**Dhivehi radicalisation detection (2025)** — first peer-reviewed NLP paper in Dhivehi. Finding: lack of language tools was the primary obstacle; results validated qualitatively due to absence of standard evaluation resources. Directly supports the argument that this benchmark is needed.

**mismaah/dhivehi_nlp (GitHub)** — community-built tokeniser, stopword remover, stemmer. No pretrained models. Documents the rule-based preprocessing ecosystem.

**Sofwath/DhivehiDatasets (GitHub)** — small collection of Dhivehi ML datasets including news headlines with categories, speech data, and Dhivehi–English text pairs. The only known public bilingual corpus for MT work beyond the Biblical data.

### 4.3 FLORES-200 and NLLB-200 Coverage

**Costa-jussà et al. (2022)** — FLORES-200 covers 200 languages, sourced from Wikinews, Wikijunior, and Wikivoyage. Dhivehi (div_Thaa) is absent. The closest covered languages are Sinhala (sin_Sinh) and Nepali (npi_Deva), both of which score in the 5–18 spBLEU range for English→language direction, indicating the difficulty ceiling. NLLB-200 does not report Dhivehi results.

**"Languages Still Left Behind" (2025)** critiques FLORES-200 sentences as culturally biased toward English-speaking world topics, heavy in named entities that can be copied for non-trivial BLEU scores, and below the claimed 90% quality threshold on re-evaluation. The recommendation for new benchmarks: source from native-language text (news, legal documents, social media), not from English Wikipedia translations.

**FLORES+/OLDI (2023–present)** — the Open Language Data Initiative actively accepts community-contributed language additions to FLORES+. Dhivehi remains absent. A benchmark paper that releases a FLORES+-compatible 1,012-sentence EN↔DV devtest under OLDI-compatible licensing can be submitted to the WMT OLDI shared task, substantially increasing impact.

### 4.4 Structural Properties Relevant to Evaluation

**Morphological richness**: Dhivehi is agglutinative with complex verb conjugation (tense × aspect × mood × person × number × politeness level). Standard BPE tokenisation over-segments Dhivehi verb forms, creating systematic evaluation artefacts under BLEU. The Turkish case (**PMC, 2025**) and the morphological segmentation literature (**Pushpananda & Weerasinghe, 2015**) confirm this failure mode for similar languages.

**Script sensitivity**: **Bafna et al. (2025)** studied Sinhala — the nearest well-studied analogue to Dhivehi in terms of script and typology — and found LLMs perform dramatically differently on native script vs. romanised input. Dhivehi is never romanised in standard writing, meaning models have minimal non-script exposure pathways.

**Register system**: Dhivehi encodes three register levels in: (a) first-person pronoun choice (formal variants such as އަޅުގަނޑު vs colloquial alternatives; exact pair labels require native-speaker verification); (b) verb suffix paradigms with politeness morphology; (c) lexical choice between Arabic-derived classical vocabulary and colloquial vocabulary. This is the linguistically most distinctive property of Dhivehi for MT evaluation purposes and is entirely absent from existing benchmarks.

**Gender-neutral pronouns**: Dhivehi third-person pronouns are gender-neutral. English gendered third-person pronouns (he/she) translate ambiguously; any benchmark must include pronoun translation test cases.

---

## 5. Challenge Sets and Register Evaluation

### 5.1 Challenge Set Methodology

**Isabelle, Cherry & Foster (2017)** established the foundational challenge set approach: hand-constructed sentences probing specific structural divergences between source and target. Their English→French set classifies phenomena as morpho-syntactic, lexico-syntactic, purely syntactic, and purely lexical. This is the direct template for a Dhivehi challenge set, with the relevant divergences being:

- SOV vs. SVO word order
- Postpositions (DV) vs. prepositions (EN)
- Agglutinative vs. analytic morphology
- Politeness-marked verb forms with no English analogue
- Converb clause-chaining (Dhivehi restricts finite verbs per sentence)
- Gender-neutral pronouns (DV) vs. gendered (EN)
- Maldivian named entities (atoll names, Islamic institutional terms)

**ACES (Amrhein et al., 2022)** provides 36,476 contrastive examples across 146 language pairs and 68 error phenomena organised under the MQM taxonomy. Construction methodology: automatic generation, repurposing existing datasets, and ~2,000 manual examples at 81.82% inter-annotator agreement. Key finding from WMT23: "all LLMs have a negative correlation across all ACES categories in reference-free settings" — validating the need for human-judged challenge sets rather than automated metrics.

### 5.2 Register Evaluation

**IWSLT Formality Track (2022–2023)** — the only existing MT benchmark explicitly evaluating register accuracy. For languages with grammaticalised formality (German, Japanese, Italian), contrastive reference pairs (formal vs. informal) are provided, enabling precision measurement of register control. This is the direct template for Dhivehi register evaluation.

**FAME-MT (2024)** — formality-annotated MT training and evaluation data, showing that standard training data contains mixed registers that models cannot disambiguate without explicit signals.

**Domain-register mismatch**: **Zouhar et al. (2024)** showed fine-tuned metrics fail on unseen domains. **"From Priest to Doctor" (2024)** quantified domain transfer failure from biblical to medical/newsroom Dhivehi — the only parallel data currently available (Bible + UDHR) would produce inflated perceived performance in religious register while failing in governmental register.

### 5.3 ESA: The Recommended Human Annotation Protocol

**Error Span Annotation (ESA, Amrhein et al., 2024)** combines direct assessment (continuous score) with error span marking, achieving MQM-quality human annotation without requiring expert linguists. AI pre-annotation halves annotation time. Validated against full MQM for English→German.

This is the recommended human evaluation protocol for the Dhivehi benchmark: feasible with non-expert annotators (important given the small global pool of Dhivehi-English bilinguals with MT evaluation competence), produces error-span-level data directly useful for challenge set construction, and is defensible as a gold-standard human evaluation method in a benchmark paper.

---

## 6. Summary: Constraints on Benchmark Architecture

The literature establishes the following hard constraints:

| Constraint | Source | Implication |
|------------|--------|-------------|
| No automatic metric validated on DV | Entire WMT coverage gap | All automatic scores carry an explicit caveat; human ESA annotation is mandatory as ground truth |
| chrF is the most defensible automatic metric | Popović 2015; Khomsi et al. 2025 | Primary metric; no tokenisation dependency; handles Thaana morphology |
| COMET/BLEURT are zero-shot extrapolations for DV | Falcão 2024; Zouhar 2024 | Reported as indicative; not used for ranking closely-scored systems |
| LLM judges unreliable for low-resource languages | Hada et al. 2025; Sindhujan 2025 | Must be cross-family (not Claude judging Moonlight); must be calibrated on 40–50 native-annotated DV examples before any comparative claim |
| Self-preference bias is documented | Panickssery 2024; Pan 2024 | Claude excluded from judge panel for any comparison involving Moonlight |
| Metric saturation at frontier LLM quality levels | Agrawal et al. 2024; WMT24 | Challenge sets required for discrimination at the top of the quality range |
| FLORES-200 absent + "Languages Still Left Behind" critique | Costa-jussà 2022; EMNLP 2025 | Test sentences must be sourced from native DV text, not EN→DV translations of Wikipedia |
| Register is the primary failure mode for institutional DV | IWSLT Formality; Moonlight eval results | Register challenge set (formal / informal / classical) is the novel contribution |
| Domain mismatch: Bible ≠ government | "From Priest to Doctor" 2024 | Benchmark must span news, government, social, and religious genres; religious genre scores will be inflated for any model exposed to eBible corpus |
| Dhivehi is absent from all major multilingual benchmarks | This review | Claim: first formal EN↔DV MT benchmark; submission to OLDI/FLORES+ as community contribution |

---

## 7. Key References

Agrawal, S., Farinhas, A., Rei, R., and Martins, A.F.T. (2024). Can Automatic Metrics Assess High-Quality Translations? *EMNLP 2024*.

Amrhein, C., et al. (2022). ACES: Translation Accuracy Challenge Sets for Evaluating Machine Translation Metrics. *WMT 2022*.

Amrhein, C., et al. (2024). Error Span Annotation: A Balanced Approach for Human Evaluation of MT. *WMT 2024*.

Bafna, N., et al. (2025). Script Sensitivity: Benchmarking Language Models on Unicode, Romanized and Mixed-Script Sinhala. *arXiv:2601.14958*.

Callison-Burch, C., Osborne, M., and Koehn, P. (2006). Re-evaluating the Role of BLEU in Machine Translation Research. *EACL 2006*.

Chiang, W., et al. (2024). Chatbot Arena: An Open Platform for Evaluating LLMs by Human Preference. *arXiv:2403.04132*.

Costa-jussà, M.R., et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation. *arXiv:2207.04672*.

Falcão, J., et al. (2024). COMET for Low-Resource MT Evaluation: A Case Study of English–Maltese and Spanish–Basque. *LREC-COLING 2024*.

Freitag, M., et al. (2021). Experts, Errors, and Context: A Large-Scale Study of Human Evaluation for Machine Translation. *TACL*, 9, 1460–1474.

Freitag, M., et al. (2022). Results of the WMT22 Metrics Shared Task: Stop Using BLEU. *WMT 2022*.

Freitag, M., et al. (2023). Results of the WMT23 Metrics Shared Task: Metrics Might Be Guilty but References Are Not Innocent. *WMT 2023*.

Freitag, M., et al. (2024). Are LLMs Breaking MT Metrics? Results of the WMT24 Metrics Shared Task. *WMT 2024*.

Guerreiro, N.M., et al. (2024). xCOMET: Transparent Machine Translation Evaluation through Fine-grained Error Detection. *TACL*, 12, 979–995.

Hada, R., et al. (2025). How Reliable is Multilingual LLM-as-a-Judge? *EMNLP 2025 Findings*. arXiv:2505.12201.

Ibrahim, A.A. (2014). Dhivehi OCR: Character Recognition of Thaana Script using Machine-Generated Text and Tesseract OCR Engine. *IJSRI*.

Islam, M., et al. (2025). LLM-Based Evaluation of Low-Resource MT: A Reference-less Dialect Guided Approach with a Refined Sylheti–English Benchmark. *arXiv:2505.12273*.

Isabelle, P., Cherry, C., and Foster, G. (2017). A Challenge Set Approach to Evaluating Machine Translation. *EMNLP 2017*.

Khomsi, A., et al. (2025). Evaluating Extremely Low-Resource MT: A Comparative Study of ChrF++ and BLEU Metrics. *arXiv:2602.17425*.

Kocmi, T., et al. (2024). Machine Translation Meta-Evaluation through Translation Accuracy Challenge Sets. *Computational Linguistics*, 51(1).

Kocmi, T. and Federmann, C. (2023a). Large Language Models Are State-of-the-Art Evaluators of Translation Quality. *EAMT 2023*. arXiv:2302.14520.

Kocmi, T. and Federmann, C. (2023b). GEMBA-MQM: Detecting Translation Quality Error Spans with GPT-4. *WMT 2023*. arXiv:2310.13988.

"Languages Still Left Behind: Toward a Better Multilingual MT Benchmark." *EMNLP 2025*. arXiv:2508.20511.

Lommel, A., Uszkoreit, H., and Burchardt, A. (2014). Multidimensional Quality Metrics (MQM). *Tradumàtica*, 12, 455–463.

Panickssery, A., et al. (2024). LLM Evaluators Recognize and Favor Their Own Generations. *arXiv:2404.13076*.

Pan, L., et al. (2024). Self-Preference Bias in LLM-as-a-Judge. *arXiv:2410.21819*.

Papineni, K., et al. (2002). BLEU: A Method for Automatic Evaluation of Machine Translation. *ACL 2002*.

Popović, M. (2015). chrF: Character n-gram F-score for Automatic MT Evaluation. *WMT 2015*.

Post, M. (2018). A Call for Clarity in Reporting BLEU Scores. *WMT 2018*.

Rei, R., et al. (2020). COMET: A Neural Framework for MT Evaluation. *EMNLP 2020*.

Reiter, E. (2018). A Structured Review of the Validity of BLEU. *Computational Linguistics*, 44(3), 393–401.

Sellam, T., Das, D., and Parikh, A. (2020). BLEURT: Learning Robust Metrics for Text Generation. *ACL 2020*.

Sindhujan, A., et al. (2025). When LLMs Struggle: Reference-less Translation Evaluation for Low-resource Languages. *LoResMT Workshop, ACL 2025*. arXiv:2501.04473.

Zheng, L., et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena. *NeurIPS 2023*. arXiv:2306.05685.

Zouhar, V., et al. (2024). Fine-Tuned Machine Translation Metrics Struggle in Unseen Domains. *ACL 2024*.
