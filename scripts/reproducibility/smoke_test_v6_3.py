#!/usr/bin/env python3
"""Smoke test: assert that the package's S19-S23 outputs reproduce the
v6.3 reference numbers exactly (within float tolerance).

The reference numbers are hard-coded from the v6.3 numbers report so this
test is self-contained and runs without access to the upstream working tree.
Float comparisons use a 1e-6 absolute tolerance; integer and string fields
are compared exactly.

Run from the package root:
    python3 scripts/reproducibility/smoke_test_v6_3.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Dict, List

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
STATISTICS_DIR = PACKAGE_ROOT / "data/management_deep_probe/statistics"

TOL = 1e-6


def _load(name: str) -> Dict:
    with open(STATISTICS_DIR / name, "r", encoding="utf-8") as handle:
        return json.load(handle)


def _check(name: str, actual, expected) -> None:
    if isinstance(expected, float):
        if not isinstance(actual, (int, float)) or abs(actual - expected) > TOL:
            raise AssertionError(
                f"FAIL {name}: expected {expected!r}, got {actual!r}"
            )
    else:
        if actual != expected:
            raise AssertionError(
                f"FAIL {name}: expected {expected!r}, got {actual!r}"
            )


def check_s19() -> None:
    data = _load("S19_PairwiseSFTKappa.json")
    summary = data["sft_kappa_summary"]
    _check("S19 sft kappa min", round(summary["min"], 3), 0.500)
    _check("S19 sft kappa max", round(summary["max"], 3), 0.603)
    _check("S19 sft kappa mean", round(summary["mean"], 3), 0.547)
    _check("S19 sft pairs", summary["n_pairs"], 6)
    pairs: List[Dict] = data["sft_pairwise"]
    _check("S19 number of pairs", len(pairs), 6)
    expected_pairs = {
        ("gpt-4.1-ob", "gpt-4.1-nano-ob"): 0.5002,
        ("gpt-4.1-ob", "qwen3-30b-ob"): 0.5556,
        ("gpt-4.1-ob", "qwen3-4b-ob"): 0.5105,
        ("gpt-4.1-nano-ob", "qwen3-30b-ob"): 0.6035,
        ("gpt-4.1-nano-ob", "qwen3-4b-ob"): 0.6032,
        ("qwen3-30b-ob", "qwen3-4b-ob"): 0.5114,
    }
    for p in pairs:
        key = (p["model_a"], p["model_b"])
        if key not in expected_pairs:
            raise AssertionError(f"FAIL S19 unexpected pair: {key}")
        _check(f"S19 kappa {key}", round(p["kappa"], 4), expected_pairs[key])


def check_s20() -> None:
    data = _load("S20_SFTConsensusPerClass.json")
    s = data["summary"]
    _check("S20 consensus n", s["n_consensus_articles"], 51)
    _check("S20 consensus correct", s["n_consensus_correct"], 36)
    _check(
        "S20 consensus acc",
        round(s["consensus_accuracy_overall"] * 100, 2),
        70.59,
    )
    _check(
        "S20 non-consensus ensemble acc",
        round(s["non_consensus_ensemble_accuracy"] * 100, 2),
        53.62,
    )
    by_tier = {row["tier"]: row for row in data["per_tier"]}
    _check("S20 exceptional consensus n", by_tier["exceptional"]["n_consensus_articles"], 14)
    _check(
        "S20 exceptional consensus acc",
        round(by_tier["exceptional"]["consensus_accuracy"] * 100, 2),
        100.00,
    )
    _check("S20 limited consensus n", by_tier["limited"]["n_consensus_articles"], 16)
    _check(
        "S20 limited consensus acc",
        round(by_tier["limited"]["consensus_accuracy"] * 100, 2),
        62.50,
    )


def check_s21() -> None:
    data = _load("S21_AIHumanComplementarity.json")
    e = data["expert"]
    _check("S21 expert N", e["n_paired"], 89)
    _check("S21 expert both_correct", e["both_correct"], 24)
    _check("S21 expert ai_only_correct", e["ai_only_correct"], 32)
    _check("S21 expert human_only_correct", e["human_only_correct"], 13)
    _check("S21 expert both_wrong", e["both_wrong"], 20)
    _check(
        "S21 expert oracle ceiling pct",
        round(e["oracle_ceiling"] * 100, 2),
        77.53,
    )
    s = data["student"]
    _check("S21 student N", s["n_paired"], 103)
    _check("S21 student both_correct", s["both_correct"], 30)
    _check("S21 student ai_only_correct", s["ai_only_correct"], 34)
    _check("S21 student human_only_correct", s["human_only_correct"], 12)
    _check("S21 student both_wrong", s["both_wrong"], 27)
    _check(
        "S21 student oracle ceiling pct",
        round(s["oracle_ceiling"] * 100, 2),
        73.79,
    )


def check_s22() -> None:
    data = _load("S22_McNemarCompendium.json")
    rows = {r["comparator"]: r for r in data["rows"]}
    fm = rows["Frontier mean (per-article plurality of 11)"]
    _check("S22 frontier-mean N", fm["paired_n"], 114)
    _check(
        "S22 frontier-mean diff_pp",
        round(fm["accuracy_diff_pp"], 2),
        22.81,
    )
    _check(
        "S22 frontier-mean stat",
        round(fm["mcnemar_statistic"], 2),
        10.78,
    )
    bf = rows["Best frontier (Gemini 3.1 Pro)"]
    _check("S22 best-frontier N", bf["paired_n"], 120)
    _check(
        "S22 best-frontier diff_pp",
        round(bf["accuracy_diff_pp"], 2),
        17.50,
    )
    ex = rows["Expert majority (excl. ties)"]
    _check("S22 expert N", ex["paired_n"], 89)
    _check("S22 expert diff_pp", round(ex["accuracy_diff_pp"], 2), 21.35)
    jr = rows["Junior majority (excl. ties)"]
    _check("S22 junior N", jr["paired_n"], 103)
    _check("S22 junior diff_pp", round(jr["accuracy_diff_pp"], 2), 21.36)
    for label, row in rows.items():
        if row["mcnemar_p_value"] >= 0.0125:
            raise AssertionError(
                f"FAIL S22 {label}: p={row['mcnemar_p_value']!r} not below 0.0125"
            )


def check_s23() -> None:
    data = _load("S23_HeadroomCaptured.json")
    rows = {r["evaluator"]: r for r in data["evaluators"]}
    sft = rows["SFT 2-model ensemble (nano + Qwen3-30B)"]
    _check("S23 SFT acc pct", round(sft["accuracy"] * 100, 2), 60.83)
    _check("S23 SFT headroom pct", round(sft["headroom_captured_pct"], 2), 47.78)
    fm = rows["Frontier mean (plurality of 11)"]
    _check("S23 frontier-mean acc pct", round(fm["accuracy"] * 100, 2), 38.60)
    _check("S23 frontier-mean headroom pct", round(fm["headroom_captured_pct"], 2), 18.13)
    bf = rows["Best frontier (Gemini 3.1 Pro)"]
    _check("S23 best-frontier acc pct", round(bf["accuracy"] * 100, 2), 43.33)
    _check("S23 best-frontier headroom pct", round(bf["headroom_captured_pct"], 2), 24.44)
    m = data["multipliers"]
    _check("S23 SFT/best-frontier", round(m["sft_over_best_frontier"], 2), 1.95)
    _check("S23 SFT/frontier-mean", round(m["sft_over_frontier_mean"], 2), 2.64)
    _check("S23 SFT/expert-majority", round(m["sft_over_expert_majority"], 2), 2.16)
    _check("S23 SFT/junior-majority", round(m["sft_over_junior_majority"], 2), 2.27)


def main() -> int:
    check_s19()
    check_s20()
    check_s21()
    check_s22()
    check_s23()
    print("Smoke test passed: all v6.3 reference numbers reproduced from S19-S23.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
