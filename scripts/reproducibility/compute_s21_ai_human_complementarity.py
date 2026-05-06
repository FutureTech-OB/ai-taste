#!/usr/bin/env python3
"""Reproduce S21: AI-human complementarity (oracle ceiling) for the SFT
2-model ensemble vs expert and student plurality (excluding tied articles).

Backs Supplementary Table ST23. Uses only released package files.

Inputs (read from this package):
    data/management_deep_probe/predictions/sft_predictions.jsonl
    data/management_deep_probe/human_ratings/expert_ratings_deidentified.jsonl
    data/management_deep_probe/human_ratings/student_ratings_deidentified.jsonl

Output:
    data/management_deep_probe/statistics/S21_AIHumanComplementarity.json

Run from the package root:
    python3 scripts/reproducibility/compute_s21_ai_human_complementarity.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List, Optional

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    EXPERTS_PATH,
    PRIMARY_ENSEMBLE_KEY,
    PRIMARY_PAIR_REQUIRED,
    SFT_PREDICTIONS_PATH,
    STATISTICS_DIR,
    STUDENTS_PATH,
    article_ground_truth,
    expert_votes_for_article,
    index_by_title,
    load_jsonl,
    majority_vote_excluding_ties,
    sft_pred,
    student_votes_for_article,
    write_stable_json,
)

OUTPUT_PATH = STATISTICS_DIR / "S21_AIHumanComplementarity.json"


def complementarity_block(
    titles_kept: List[str],
    sft_idx: Dict,
    human_pred: Dict[str, str],
    label: str,
) -> Dict:
    n = len(titles_kept)
    both = ai_only = human_only = neither = 0
    sft_correct = human_correct = 0
    for t in titles_kept:
        art = sft_idx[t]
        gt = article_ground_truth(art)
        ai_pred = sft_pred(art, PRIMARY_ENSEMBLE_KEY)
        h_pred = human_pred[t]
        ai_ok = ai_pred == gt
        h_ok = h_pred == gt
        if ai_ok and h_ok:
            both += 1
        elif ai_ok and not h_ok:
            ai_only += 1
        elif h_ok and not ai_ok:
            human_only += 1
        else:
            neither += 1
        if ai_ok:
            sft_correct += 1
        if h_ok:
            human_correct += 1
    oracle = (both + ai_only + human_only) / n if n else 0.0
    sft_acc = sft_correct / n if n else 0.0
    h_acc = human_correct / n if n else 0.0
    return {
        "ai_only_correct": ai_only,
        "both_correct": both,
        "both_wrong": neither,
        "human_accuracy": h_acc,
        "human_correct": human_correct,
        "human_only_correct": human_only,
        "label": label,
        "n_paired": n,
        "oracle_ceiling": oracle,
        "oracle_lift_over_human": oracle - h_acc,
        "oracle_lift_over_sft": oracle - sft_acc,
        "sft_accuracy": sft_acc,
        "sft_correct": sft_correct,
    }


def _human_pred_by_title(records: List[Dict], votes_fn) -> Dict[str, Optional[str]]:
    out: Dict[str, Optional[str]] = {}
    n_tied = 0
    for art in records:
        pred, is_tie = majority_vote_excluding_ties(votes_fn(art))
        if is_tie or pred is None:
            out[art["title"]] = None
            if is_tie:
                n_tied += 1
        else:
            out[art["title"]] = pred
    return out, n_tied


def main() -> int:
    sft = load_jsonl(SFT_PREDICTIONS_PATH)
    experts = load_jsonl(EXPERTS_PATH)
    students = load_jsonl(STUDENTS_PATH)

    sft_idx = index_by_title(sft)

    combo = sft[0]["val_outcome"]["rq_with_context"].get(PRIMARY_ENSEMBLE_KEY, {})
    if sorted(combo.get("models", [])) != sorted(PRIMARY_PAIR_REQUIRED):
        raise AssertionError(
            f"Primary ensemble pair mismatch: best_2_model_combo.models="
            f"{combo.get('models')}, expected {PRIMARY_PAIR_REQUIRED}"
        )

    expert_pred, n_expert_tied = _human_pred_by_title(experts, expert_votes_for_article)
    student_pred, n_student_tied = _human_pred_by_title(students, student_votes_for_article)

    expert_titles = [t for t, p in expert_pred.items() if p is not None and t in sft_idx]
    student_titles = [t for t, p in student_pred.items() if p is not None and t in sft_idx]

    expert_block = complementarity_block(
        expert_titles,
        sft_idx,
        {t: expert_pred[t] for t in expert_titles},  # type: ignore[arg-type]
        label="expert_majority_non_tied",
    )
    student_block = complementarity_block(
        student_titles,
        sft_idx,
        {t: student_pred[t] for t in student_titles},  # type: ignore[arg-type]
        label="student_majority_non_tied",
    )

    output = {
        "expert": expert_block,
        "metadata": {
            "argmax_tie_break": "alphabetical (exceptional<fair<limited<strong)",
            "data_sources": {
                "experts": "data/management_deep_probe/human_ratings/expert_ratings_deidentified.jsonl",
                "sft_predictions": "data/management_deep_probe/predictions/sft_predictions.jsonl",
                "students": "data/management_deep_probe/human_ratings/student_ratings_deidentified.jsonl",
            },
            "majority_vote_tie_policy": "exclude",
            "n_articles_total": 120,
            "n_expert_kept": len(expert_titles),
            "n_expert_tied_articles": n_expert_tied,
            "n_student_kept": len(student_titles),
            "n_student_tied_articles": n_student_tied,
            "primary_ensemble_key": PRIMARY_ENSEMBLE_KEY,
            "primary_pair": PRIMARY_PAIR_REQUIRED,
            "supplementary_table": "ST23",
        },
        "note": (
            "Numbers reflect the current primary SFT ensemble pair "
            "(GPT-4.1-nano + Qwen3-30B). They do not match a hypothetical "
            "earlier version that used GPT-4.1-nano + Qwen3-4B; recompute "
            "from this package, do not copy."
        ),
        "student": student_block,
    }
    write_stable_json(OUTPUT_PATH, output)
    print(
        f"S21 OK: expert N={expert_block['n_paired']} "
        f"oracle={expert_block['oracle_ceiling']*100:.2f}%; "
        f"student N={student_block['n_paired']} "
        f"oracle={student_block['oracle_ceiling']*100:.2f}%; "
        f"wrote {OUTPUT_PATH.relative_to(OUTPUT_PATH.parents[3])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
