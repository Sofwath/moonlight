#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Post-process a benchmark results JSON to add DV fluency scores.

Run this after run_benchmark.py on any results file that lacks fluency scores.

Usage::

    python scripts/add_fluency_scores.py results/run_002_frontier_best.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Add DV fluency scores to existing benchmark results JSON."
    )
    parser.add_argument("results", metavar="PATH", help="results JSON from run_benchmark.py")
    parser.add_argument("--out", default=None, metavar="PATH",
                        help="Output path (default: overwrites input)")
    args = parser.parse_args()

    results_path = Path(args.results).expanduser().resolve()
    out_path = Path(args.out).expanduser().resolve() if args.out else results_path

    with results_path.open(encoding="utf-8") as f:
        data = json.load(f)

    print("Loading Dhivehi fluency scorer …")
    from moonlight.dv_fluency import DvFluencyScorer
    scorer = DvFluencyScorer()

    main_raw = data.get("main_set_raw", {})
    total = sum(len(recs) for recs in main_raw.values())
    done = 0

    for sid, records in main_raw.items():
        for r in records:
            if r.get("target_lang") == "DV":
                hyp = r.get("hypothesis", "")
                r["scores"]["fluency"] = scorer.fluency_score(hyp)
                r["scores"]["perplexity"] = scorer.perplexity(hyp)
            done += 1
            print(f"  [{done}/{total}] {sid} {r.get('segment_id','?'):<30} "
                  f"fluency={r['scores'].get('fluency', 'n/a')}", end="\r")

    print()

    # Recompute aggregate fluency means
    agg = data.get("main_set_aggregate", {})
    for sid, records in main_raw.items():
        fluency_scores = [
            r["scores"]["fluency"] for r in records
            if r["scores"].get("fluency") is not None
        ]
        if sid in agg and fluency_scores:
            agg[sid]["fluency_mean"] = round(sum(fluency_scores) / len(fluency_scores), 2)

    with out_path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

    print(f"\nFluency scores added → {out_path}")

    # Print summary table
    print("\nFluency summary:")
    print(f"  {'System':<25}  fluency_mean")
    for sid, a in agg.items():
        if "fluency_mean" in a:
            print(f"  {sid:<25}  {a['fluency_mean']}")


if __name__ == "__main__":
    main()
