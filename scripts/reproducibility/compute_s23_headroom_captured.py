#!/usr/bin/env python3
"""Reproduce S23: headroom captured = (accuracy - 0.25) / (1.00 - 0.25)
for the SFT 2-model ensemble and four comparators (frontier-mean plurality,
best frontier, expert majority, junior majority).

Backs the headroom note in Supplementary Methods SM4 and the headroom
multiplier reported alongside main-text accuracy.

Inputs (read from this package):
    data/management_deep_probe/predictions/sft_predictions.jsonl
    data/management_deep_probe/predictions/frontier_thinking_11models_singleshot.jsonl
    data/management_deep_probe/human_ratings/expert_ratings_deidentified.jsonl
    data/management_deep_probe/human_ratings/student_ratings_deidentified.jsonl

Output:
    data/management_deep_probe/statistics/S23_HeadroomCaptured.json

Run from the package root:
    python3 scripts/reproducibility/compute_s23_headroom_captured.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Tuple

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    BEST_FRONTIER_KEY,
    EXPERTS_PATH,
    FRONTIER_THINKING_PATH,
    PRIMARY_ENSEMBLE_KEY,
    PRIMARY_PAIR_REQUIRED,
    SFT_PREDICTIONS_PATH,
    STATISTICS_DIR,
    STUDENTS_PATH,
    THINKING_MODELS_11,
    article_ground_truth,
    expert_votes_for_article,
    load_jsonl,
    majority_vote_excluding_ties,
    sft_pred,
    student_votes_for_article,
    thinking_pred,
    write_stable_json,
)

OUTPUT_PATH = STATISTICS_DIR / "S23_HeadroomCaptured.json"

CHANCE = 0.25


def _headroom_captured(accuracy: float) -> float:
    return (accuracy - CHANCE) / (1.0 - CHANCE)


def _accuracy(pairs: List[Tuple[str, str]]) -> Dict:
    n = sum(1 for gt, p in pairs if p is not None)
    correct = sum(1 for gt, p in pairs if p is not None and p == gt)
    return {
        "accuracy": correct / n if n else 0.0,
        "correct": correct,
        "n": n,
    }


def main() -> int:
    sft = load_jsonl(SFT_PREDICTIONS_PATH)
    think = load_jsonl(FRONTIER_THINKING_PATH)
    experts = load_jsonl(EXPERTS_PATH)
    students = load_jsonl(STUDENTS_PATH)

    combo = sft[0]["val_outcome"]["rq_with_context"].get(PRIMARY_ENSEMBLE_KEY, {})
    if sorted(combo.get("models", [])) != sorted(PRIMARY_PAIR_REQUIRED):
        raise AssertionError(
            f"Primary ensemble pair mismatch: best_2_model_combo.models="
            f"{combo.get('models')}"
        )

    sft_pairs: List[Tuple[str, str]] = []
    for art in sft:
        sft_pairs.append((article_ground_truth(art), sft_pred(art, PRIMARY_ENSEMBLE_KEY)))
    sft_acc = _accuracy(sft_pairs)

    fm_pairs: List[Tuple[str, str]] = []
    for art in think:
        gt = article_ground_truth(art)
        votes = []
        for k in THINKING_MODELS_11:
            p = thinking_pred(art, k)
            if p:
                votes.append(p)
        pred, is_tie = majority_vote_excluding_ties(votes)
        if is_tie or pred is None:
            continue
        fm_pairs.append((gt, pred))
    fm_acc = _accuracy(fm_pairs)

    bf_pairs: List[Tuple[str, str]] = []
    for art in think:
        gt = article_ground_truth(art)
        p = thinking_pred(art, BEST_FRONTIER_KEY)
        if p is None:
            continue
        bf_pairs.append((gt, p))
    bf_acc = _accuracy(bf_pairs)

    ex_pairs: List[Tuple[str, str]] = []
    for art in experts:
        gt = article_ground_truth(art)
        pred, is_tie = majority_vote_excluding_ties(expert_votes_for_article(art))
        if is_tie or pred is None:
            continue
        ex_pairs.append((gt, pred))
    ex_acc = _accuracy(ex_pairs)

    jr_pairs: List[Tuple[str, str]] = []
    for art in students:
        gt = article_ground_truth(art)
        pred, is_tie = majority_vote_excluding_ties(student_votes_for_article(art))
        if is_tie or pred is None:
            continue
        jr_pairs.append((gt, pred))
    jr_acc = _accuracy(jr_pairs)

    rows = []
    for label, acc in [
        ("Frontier mean (plurality of 11)", fm_acc),
        ("Best frontier (Gemini 3.1 Pro)", bf_acc),
        ("Expert majority (excl. ties)", ex_acc),
        ("Junior majority (excl. ties)", jr_acc),
        ("SFT 2-model ensemble (nano + Qwen3-30B)", sft_acc),
    ]:
        h = _headroom_captured(acc["accuracy"])
        rows.append(
            {
                "accuracy": acc["accuracy"],
                "correct": acc["correct"],
                "evaluator": label,
                "headroom_captured": h,
                "headroom_captured_pct": h * 100.0,
                "n": acc["n"],
            }
        )

    sft_h = _headroom_captured(sft_acc["accuracy"])
    fm_h = _headroom_captured(fm_acc["accuracy"])
    bf_h = _headroom_captured(bf_acc["accuracy"])
    ex_h = _headroom_captured(ex_acc["accuracy"])
    jr_h = _headroom_captured(jr_acc["accuracy"])
    multipliers = {
        "sft_over_best_frontier": (sft_h / bf_h) if bf_h > 0 else None,
        "sft_over_expert_majority": (sft_h / ex_h) if ex_h > 0 else None,
        "sft_over_frontier_mean": (sft_h / fm_h) if fm_h > 0 else None,
        "sft_over_junior_majority": (sft_h / jr_h) if jr_h > 0 else None,
    }

    output = {
        "evaluators": rows,
        "metadata": {
            "argmax_tie_break": "alphabetical (exceptional<fair<limited<strong)",
            "chance_floor": CHANCE,
            "data_sources": {
                "experts": "data/management_deep_probe/human_ratings/expert_ratings_deidentified.jsonl",
                "frontier_thinking": "data/management_deep_probe/predictions/frontier_thinking_11models_singleshot.jsonl",
                "sft_predictions": "data/management_deep_probe/predictions/sft_predictions.jsonl",
                "students": "data/management_deep_probe/human_ratings/student_ratings_deidentified.jsonl",
            },
            "headroom_formula": "(accuracy - 0.25) / (1.00 - 0.25)",
            "majority_vote_tie_policy": "exclude",
            "primary_ensemble_key": PRIMARY_ENSEMBLE_KEY,
            "primary_pair": PRIMARY_PAIR_REQUIRED,
            "supplementary_section": "SM4",
        },
        "multipliers": multipliers,
    }
    write_stable_json(OUTPUT_PATH, output)
    print(
        f"S23 OK: SFT acc={sft_acc['accuracy']*100:.2f}% "
        f"headroom={sft_h*100:.2f}%; "
        f"SFT/best-frontier={multipliers['sft_over_best_frontier']:.2f}x; "
        f"wrote {OUTPUT_PATH.relative_to(OUTPUT_PATH.parents[3])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
