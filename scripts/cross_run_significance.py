#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Cross-run paired significance testing for DhivehiMT-Bench.

Computes approximate randomization tests (Riezler & Maxwell 2005) between
systems that live in different run JSON files (e.g. Moonlight in run_003 vs
raw baselines in run_001/run_002). Results are saved as a reproducible JSON
artifact with full provenance.

Usage::

    python scripts/cross_run_significance.py \\
        --runs run_001_frontier_baseline.json:gpt4o_raw,claude_raw,gemini_raw \\
               run_002_frontier_best.json:gpt5_raw,claude_opus_raw,gemini35_raw \\
               run_003_moonlight_full.json:moonlight_full \\
        --output results/cross_run_significance.json

The --runs argument accepts  <filename>:<system_id>[,<system_id>...]  entries.
All systems in all runs are paired against each other on the segments they
share (intersection by segment_id). Pairs with fewer than 10 shared segments
are skipped.

Output JSON schema::

    {
      "meta": {
        "generated_at": "...",
        "n_trials": 10000,
        "seed": 42,
        "method": "approximate_randomization",
        "reference": "Riezler & Maxwell (2005)"
      },
      "runs_loaded": { "<run_file>": ["<system_id>", ...] },
      "comparisons": [
        {
          "system_a": "moonlight_full",
          "system_b": "claude_opus_raw",
          "run_a": "run_003_moonlight_full.json",
          "run_b": "run_002_frontier_best.json",
          "n_shared_segments": 50,
          "delta_chrf": 6.11,
          "p_value": 0.0001,
          "significant_05": true,
          "significant_01": true
        },
        ...
      ]
    }
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def load_seg_scores(run_path: Path, system_id: str) -> dict[str, float]:
    """Return {segment_id: chrf} for one system in one run file."""
    data = json.loads(run_path.read_text(encoding="utf-8"))
    records = data.get("main_set_raw", {}).get(system_id, [])
    return {r["segment_id"]: r["scores"]["chrf"] for r in records}


def approx_rand_test(
    scores_a: list[float],
    scores_b: list[float],
    n_trials: int = 10_000,
    seed: int = 42,
) -> dict:
    """Two-sided approximate randomization test on paired segment-level chrF."""
    if len(scores_a) != len(scores_b) or not scores_a:
        return {"error": "unequal or empty lists"}
    try:
        import numpy as np
    except ImportError:
        sys.exit("numpy is required: pip install numpy")

    rng = np.random.default_rng(seed)
    a = np.array(scores_a)
    b = np.array(scores_b)
    obs_delta = float(a.mean() - b.mean())
    diffs = a - b
    count = sum(
        1
        for _ in range(n_trials)
        if abs(float((rng.choice([-1.0, 1.0], size=len(diffs)) * diffs).mean()))
        >= abs(obs_delta)
    )
    p = (count + 1) / (n_trials + 1)
    return {
        "delta_chrf": round(obs_delta, 3),
        "p_value": round(p, 4),
        "significant_05": p < 0.05,
        "significant_01": p < 0.01,
        "n_shared_segments": len(scores_a),
        "n_trials": n_trials,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Cross-run paired significance testing for DhivehiMT-Bench."
    )
    parser.add_argument(
        "--runs", nargs="+", required=True, metavar="FILE:SID[,SID]",
        help=(
            "Space-separated entries of the form "
            "<results_json>:<system_id>[,<system_id>...]. "
            "Paths relative to the repo results/ directory if not absolute."
        ),
    )
    parser.add_argument(
        "--output", default="results/cross_run_significance.json", metavar="PATH",
        help="Output JSON path (default: results/cross_run_significance.json).",
    )
    parser.add_argument(
        "--n-trials", type=int, default=10_000, metavar="N",
        help="Number of randomization trials (default: 10000).",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="RNG seed for reproducibility (default: 42).",
    )
    parser.add_argument(
        "--min-shared", type=int, default=10, metavar="N",
        help="Skip pairs with fewer than N shared segments (default: 10).",
    )
    args = parser.parse_args()

    results_dir = ROOT / "results"

    # Parse --runs entries
    all_systems: list[tuple[Path, str]] = []  # (run_path, system_id)
    runs_loaded: dict[str, list[str]] = {}
    for entry in args.runs:
        if ":" not in entry:
            sys.exit(f"Invalid --runs entry (expected FILE:SID): {entry!r}")
        file_part, sids_part = entry.split(":", 1)
        run_path = Path(file_part).expanduser()
        if not run_path.is_absolute():
            run_path = results_dir / run_path
        if not run_path.exists():
            sys.exit(f"Run file not found: {run_path}")
        sids = [s.strip() for s in sids_part.split(",") if s.strip()]
        runs_loaded[run_path.name] = sids
        for sid in sids:
            all_systems.append((run_path, sid))

    # Load segment scores
    print(f"Loading {len(all_systems)} system(s) from {len(runs_loaded)} run file(s) …")
    scores: dict[tuple[str, str], dict[str, float]] = {}  # (run_name, sid) → {seg_id: chrf}
    for run_path, sid in all_systems:
        seg_map = load_seg_scores(run_path, sid)
        if not seg_map:
            print(f"  [warn] no segments found for {sid} in {run_path.name}")
        else:
            print(f"  {run_path.name}/{sid}: {len(seg_map)} segments")
        scores[(run_path.name, sid)] = seg_map

    # All pairwise comparisons
    print(f"\nRunning pairwise tests (n_trials={args.n_trials}, seed={args.seed}) …")
    comparisons = []
    keys = list(scores.keys())
    for i, (run_a, sid_a) in enumerate(keys):
        for run_b, sid_b in keys[i + 1:]:
            segs_a = scores[(run_a, sid_a)]
            segs_b = scores[(run_b, sid_b)]
            common = sorted(set(segs_a) & set(segs_b))
            if len(common) < args.min_shared:
                print(f"  skip  {sid_a} vs {sid_b}: only {len(common)} shared segments")
                continue
            a_list = [segs_a[s] for s in common]
            b_list = [segs_b[s] for s in common]
            result = approx_rand_test(a_list, b_list, n_trials=args.n_trials, seed=args.seed)
            entry = {
                "system_a": sid_a,
                "system_b": sid_b,
                "run_a": run_a,
                "run_b": run_b,
                **result,
            }
            comparisons.append(entry)
            sig = "✓" if result["significant_05"] else "✗"
            print(
                f"  {sig} {sid_a} vs {sid_b:<30} "
                f"n={result['n_shared_segments']}  "
                f"Δ={result['delta_chrf']:+.2f}  "
                f"p={result['p_value']:.4f}"
            )

    output = {
        "meta": {
            "generated_at": datetime.now(timezone.utc).isoformat(),
            "n_trials": args.n_trials,
            "seed": args.seed,
            "method": "approximate_randomization",
            "reference": "Riezler & Maxwell (2005). Bootstrapping and Approximate Randomization Testing for Comparing MT Systems. MT Summit X.",
            "p_value_formula": "(count+1)/(n_trials+1) — standard Monte Carlo correction",
        },
        "runs_loaded": runs_loaded,
        "comparisons": comparisons,
    }

    out_path = Path(args.output).expanduser()
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\nSaved → {out_path}")


if __name__ == "__main__":
    main()
