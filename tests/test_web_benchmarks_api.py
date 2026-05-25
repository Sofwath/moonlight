"""Regression tests for /api/benchmarks payload shaping."""

import json

import pytest

pytest.importorskip("fastapi")

from moonlight.web.api import benchmarks as benchmarks_api


def _write_json(path, payload):
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_benchmarks_returns_aggregate_only_payload_and_summary(tmp_path, monkeypatch):
    """Ensure endpoint omits heavy raw segment arrays and computes summary correctly."""
    # Only frontier_best + moonlight_full are needed for summary computation.
    _write_json(
        tmp_path / "run_002_frontier_best.json",
        {
            "meta": {"split": "dev"},
            "main_set_aggregate": {
                "gpt5_raw": {"n": 50, "chrf": {"mean": 31.14}, "bleu": {"mean": 2.88}},
                "claude_opus_raw": {
                    "n": 50,
                    "chrf": {"mean": 43.20},
                    "bleu": {"mean": 6.78},
                },
                "_pairwise_significance": {"dummy": {"p_value": 0.1}},
            },
            "challenge_set_aggregate": {"gpt5_raw": {"overall_accuracy": 0.667}},
            # This key should be stripped out by the API response.
            "main_set_raw": {"gpt5_raw": [{"segment_id": "s1", "scores": {"chrf": 11.0}}]},
        },
    )
    _write_json(
        tmp_path / "run_003_moonlight_full.json",
        {
            "meta": {"split": "dev"},
            "main_set_aggregate": {
                "moonlight_full": {
                    "n": 50,
                    "chrf": {"mean": 49.31},
                    "bleu": {"mean": 14.09},
                }
            },
            "challenge_set_aggregate": {"moonlight_full": {"overall_accuracy": 0.745}},
            "main_set_raw": {
                "moonlight_full": [{"segment_id": "s1", "scores": {"chrf": 66.37}}]
            },
        },
    )

    monkeypatch.setattr(benchmarks_api, "_RESULTS_DIR", tmp_path)

    response = benchmarks_api.benchmarks()
    runs = response["runs"]
    summary = response["summary"]

    assert "frontier_best" in runs
    assert "moonlight_full" in runs

    # Payload slimming: endpoint should not return full per-segment raw arrays.
    assert "main_set_raw" not in runs["frontier_best"]["data"]
    assert "main_set_raw" not in runs["moonlight_full"]["data"]

    # Aggregate data remains available for rendering tables/charts.
    assert "main_set_aggregate" in runs["frontier_best"]["data"]
    assert "challenge_set_aggregate" in runs["moonlight_full"]["data"]

    # Summary uses max frontier chrF and moonlight chrF means.
    assert summary["best_frontier_chrf"] == 43.2
    assert summary["moonlight_chrf"] == 49.3
    assert summary["improvement"] == 6.1
    assert summary["n_items"] == 50
