#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Detect challenge-result drift against current challenge seed.

Compares the challenge pair text embedded in result JSON files with the current
`data/benchmark/challenge_set/challenge_seed.jsonl` source of truth.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_SEED = ROOT / "data" / "benchmark" / "challenge_set" / "challenge_seed.jsonl"
DEFAULT_RESULTS_DIR = ROOT / "results"


def load_seed(seed_path: Path) -> dict[str, dict]:
    pairs: dict[str, dict] = {}
    with seed_path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            item = json.loads(line)
            pairs[item["id"]] = item
    return pairs


def iter_result_files(results_dir: Path) -> list[Path]:
    return sorted(results_dir.glob("*.json"))


def check_file(path: Path, seed: dict[str, dict]) -> list[str]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    raw = payload.get("challenge_set_raw", {})
    mismatches: list[str] = []
    for _, rows in raw.items():
        for row in rows:
            pair_id = row.get("pair_id")
            if not pair_id:
                continue
            if pair_id not in seed:
                mismatches.append(f"{pair_id}: missing from current seed")
                continue
            seed_pair = seed[pair_id]
            got_correct = row.get("correct", "")
            got_incorrect = row.get("incorrect", "")
            if got_correct != seed_pair.get("correct", ""):
                mismatches.append(f"{pair_id}: correct text differs")
            if got_incorrect != seed_pair.get("incorrect", ""):
                mismatches.append(f"{pair_id}: incorrect text differs")
    return sorted(set(mismatches))


def main() -> None:
    parser = argparse.ArgumentParser(description="Check challenge-result drift against seed.")
    parser.add_argument("--seed", default=str(DEFAULT_SEED), help="Path to challenge_seed.jsonl")
    parser.add_argument("--results-dir", default=str(DEFAULT_RESULTS_DIR), help="Directory with result JSON files")
    args = parser.parse_args()

    seed_path = Path(args.seed).expanduser().resolve()
    results_dir = Path(args.results_dir).expanduser().resolve()
    if not seed_path.exists():
        raise SystemExit(f"Seed file not found: {seed_path}")
    if not results_dir.exists():
        raise SystemExit(f"Results directory not found: {results_dir}")

    seed = load_seed(seed_path)
    files = iter_result_files(results_dir)
    if not files:
        raise SystemExit(f"No JSON result files found in: {results_dir}")

    drift_found = False
    print(f"Seed: {seed_path}")
    print(f"Results dir: {results_dir}")
    print()
    for path in files:
        mismatches = check_file(path, seed)
        if mismatches:
            drift_found = True
            print(f"[STALE] {path.name}")
            for m in mismatches:
                print(f"  - {m}")
        else:
            print(f"[OK]    {path.name}")
    if drift_found:
        raise SystemExit(2)


if __name__ == "__main__":
    main()

