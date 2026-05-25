# DhivehiMT-Bench Review Response Action Items

This file tracks actions from external draft reviews and current implementation status.

## Priority 0: Integrity gates (must-pass before submission)

- [ ] Complete native-speaker verification for Cat-1, Cat-4, Cat-5 pairs.
- [ ] Run benchmark in publication-safe mode (`--publish-ready`) and regenerate stale result artifacts.
- [ ] Re-run challenge-set scoring after seed updates; archive prior stale outputs.
- [ ] Confirm NLLB `div_Thaa` inference support (or remove placeholder baseline).
- [ ] Replace placeholder references (`[venue TBD]`) with verified citations.

## Priority 1: Methodological rigor

- [ ] Expand evaluation beyond `n=50` government dev subset to full planned set.
- [ ] Add paired significance tests (bootstrap paired / approximate randomization / permutation).
- [ ] Add benchmark leakage controls:
  - [ ] document-level exclusion policy
  - [ ] near-duplicate detection
  - [ ] embedding-similarity filtering
  - [ ] temporal split audit
- [ ] Keep challenge pass criterion but add human/LLM forced-choice validation for fragile categories.

## Priority 2: Paper framing

- [ ] Separate benchmark contribution from system-case-study narrative:
  - [ ] benchmark-first manuscript track
  - [ ] Moonlight evaluation companion track
- [ ] Demote fluency-perplexity metric to exploratory-only interpretation.
- [ ] Remove overclaims (`architectural limit`, universal `hard failure`) unless validated on full set.
- [ ] Convert absolute novelty language to scoped `to our knowledge` claims.

## Priority 3: Data completion

- [ ] Finish sourcing/licensing for news/social/religious genres.
- [ ] Complete DV→EN evaluation direction.
- [ ] Finalize OLDI submission status.

## Implemented in current revision

- [x] Challenge-set verification policy guardrails added in `scripts/run_benchmark.py`.
- [x] Publication-safe mode (`--publish-ready`) added.
- [x] Optional unverified inclusion requires explicit flag (`--include-unverified`).
- [x] Challenge policy metadata persisted in benchmark outputs.
- [x] Upstream Cat-1 pronoun seed corrected to `އަޅުގަނޑު` (still `verified:false` pending sign-off).
- [x] Staleness audit added: `scripts/check_challenge_result_staleness.py`.
- [x] Manuscript language softened for novelty/statistical claims in `docs/dhivehimt-bench-paper.md`.

