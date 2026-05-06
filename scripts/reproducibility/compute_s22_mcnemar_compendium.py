#!/usr/bin/env python3
"""Reproduce S22: pairwise McNemar tests of the SFT 2-model ensemble vs four
comparators (frontier-mean plurality, best frontier, expert majority, junior
majority).

Backs Supplementary Table ST24.

Inputs (read from this package):
    data/management_deep_probe/predictions/sft_predictions.jsonl
    data/management_deep_probe/predictions/frontier_thinking_11models_singleshot.jsonl
    data/management_deep_probe/human_ratings/expert_ratings_deidentified.jsonl
    data/management_deep_probe/human_ratings/student_ratings_deidentified.jsonl

Output:
    data/management_deep_probe/statistics/S22_McNemarCompendium.json

Run from the package root:
    python3 scripts/reproducibility/compute_s22_mcnemar_compendium.py
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Dict, List, Optional

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
    index_by_title,
    load_jsonl,
    majority_vote_excluding_ties,
    sft_pred,
    student_votes_for_article,
    thinking_pred,
    write_stable_json,
)

OUTPUT_PATH = STATISTICS_DIR / "S22_McNemarCompendium.json"


def _chi2_sf_df1(stat: float) -> float:
    """Survival function (1 - CDF) of chi-square with 1 dof.

    For df=1 the chi-square cumulative is erf(sqrt(stat/2)); the survival
    function is therefore erfc(sqrt(stat/2)). This avoids importing scipy
    so the package's reproduction surface stays standard-library only.
    """
    if stat < 0:
        return 1.0
    return math.erfc(math.sqrt(stat / 2.0))


def mcnemar_statistic(b: int, c: int) -> Dict[str, float]:
    if b + c == 0:
        return {"b": float(b), "c": float(c), "p_value": 1.0, "statistic": 0.0}
    stat = (abs(b - c) - 1) ** 2 / (b + c)
    return {
        "b": float(b),
        "c": float(c),
        "p_value": _chi2_sf_df1(stat),
        "statistic": stat,
    }


def mcnemar_row(
    titles: List[str],
    sft_pred_by_title: Dict[str, str],
    comparator_pred_by_title: Dict[str, Optional[str]],
    gt_by_title: Dict[str, str],
    label: str,
) -> Dict:
    paired_titles = [
        t
        for t in titles
        if comparator_pred_by_title.get(t) is not None
        and sft_pred_by_title.get(t) is not None
    ]
    n = len(paired_titles)
    sft_correct = comp_correct = 0
    b = c = 0
    for t in paired_titles:
        gt = gt_by_title[t]
        s = sft_pred_by_title[t] == gt
        cm = comparator_pred_by_title[t] == gt
        if s:
            sft_correct += 1
        if cm:
            comp_correct += 1
        if s and not cm:
            b += 1
        elif cm and not s:
            c += 1
    mcn = mcnemar_statistic(b, c)
    return {
        "accuracy_diff_pp": (sft_correct - comp_correct) * 100.0 / n if n else 0.0,
        "comparator": label,
        "comparator_accuracy": comp_correct / n if n else 0.0,
        "comparator_correct": comp_correct,
        "discordant_b_sft_only": int(b),
        "discordant_c_comparator_only": int(c),
        "mcnemar_p_value": mcn["p_value"],
        "mcnemar_statistic": mcn["statistic"],
        "paired_n": n,
        "sft_ensemble_accuracy": sft_correct / n if n else 0.0,
        "sft_ensemble_correct": sft_correct,
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

    titles = [a["title"] for a in sft]
    gt_by_title = {a["title"]: article_ground_truth(a) for a in sft}
    sft_pred_by_title = {
        a["title"]: sft_pred(a, PRIMARY_ENSEMBLE_KEY) for a in sft
    }

    frontier_pred_by_title: Dict[str, Optional[str]] = {}
    n_frontier_tied = 0
    for art in think:
        votes = []
        for k in THINKING_MODELS_11:
            p = thinking_pred(art, k)
            if p:
                votes.append(p)
        pred, is_tie = majority_vote_excluding_ties(votes)
        if is_tie or pred is None:
            frontier_pred_by_title[art["title"]] = None
            if is_tie:
                n_frontier_tied += 1
        else:
            frontier_pred_by_title[art["title"]] = pred

    best_frontier_by_title: Dict[str, Optional[str]] = {
        art["title"]: thinking_pred(art, BEST_FRONTIER_KEY) for art in think
    }

    expert_pred_by_title: Dict[str, Optional[str]] = {}
    n_expert_tied = 0
    for art in experts:
        pred, is_tie = majority_vote_excluding_ties(expert_votes_for_article(art))
        if is_tie or pred is None:
            expert_pred_by_title[art["title"]] = None
            if is_tie:
                n_expert_tied += 1
        else:
            expert_pred_by_title[art["title"]] = pred

    student_pred_by_title: Dict[str, Optional[str]] = {}
    n_student_tied = 0
    for art in students:
        pred, is_tie = majority_vote_excluding_ties(student_votes_for_article(art))
        if is_tie or pred is None:
            student_pred_by_title[art["title"]] = None
            if is_tie:
                n_student_tied += 1
        else:
            student_pred_by_title[art["title"]] = pred

    rows = [
        mcnemar_row(
            titles,
            sft_pred_by_title,
            frontier_pred_by_title,
            gt_by_title,
            "Frontier mean (per-article plurality of 11)",
        ),
        mcnemar_row(
            titles,
            sft_pred_by_title,
            best_frontier_by_title,
            gt_by_title,
            "Best frontier (Gemini 3.1 Pro)",
        ),
        mcnemar_row(
            titles,
            sft_pred_by_title,
            expert_pred_by_title,
            gt_by_title,
            "Expert majority (excl. ties)",
        ),
        mcnemar_row(
            titles,
            sft_pred_by_title,
            student_pred_by_title,
            gt_by_title,
            "Junior majority (excl. ties)",
        ),
    ]

    output = {
        "metadata": {
            "argmax_tie_break": "alphabetical (exceptional<fair<limited<strong)",
            "best_frontier_key": BEST_FRONTIER_KEY,
            "data_sources": {
                "experts": "data/management_deep_probe/human_ratings/expert_ratings_deidentified.jsonl",
                "frontier_thinking": "data/management_deep_probe/predictions/frontier_thinking_11models_singleshot.jsonl",
                "sft_predictions": "data/management_deep_probe/predictions/sft_predictions.jsonl",
                "students": "data/management_deep_probe/human_ratings/student_ratings_deidentified.jsonl",
            },
            "frontier_mean_definition": (
                "per-article plurality across the 11 thinking-model "
                "single-shot prediction fields, excluding articles where "
                "the plurality is tied"
            ),
            "majority_vote_tie_policy": "exclude",
            "n_articles_total": 120,
            "n_expert_tied_articles": n_expert_tied,
            "n_frontier_tied_articles": n_frontier_tied,
            "n_student_tied_articles": n_student_tied,
            "primary_ensemble_key": PRIMARY_ENSEMBLE_KEY,
            "primary_pair": PRIMARY_PAIR_REQUIRED,
            "supplementary_table": "ST24",
            "thinking_models_11": THINKING_MODELS_11,
        },
        "rows": rows,
    }
    write_stable_json(OUTPUT_PATH, output)
    fm = rows[0]
    print(
        f"S22 OK: McNemar (frontier-mean) N={fm['paired_n']} "
        f"diff={fm['accuracy_diff_pp']:+.2f}pp P={fm['mcnemar_p_value']:.4g}; "
        f"wrote {OUTPUT_PATH.relative_to(OUTPUT_PATH.parents[3])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
