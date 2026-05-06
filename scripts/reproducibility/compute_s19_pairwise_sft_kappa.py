#!/usr/bin/env python3
"""Reproduce S19: pairwise Cohen's kappa across the 4 management SFT models.

Backs Supplementary Table ST21 (inter-SFT pairwise agreement).

Inputs (read from this package):
    data/management_deep_probe/predictions/sft_predictions.jsonl
    data/management_deep_probe/human_ratings/expert_ratings_deidentified.jsonl
    data/management_deep_probe/human_ratings/student_ratings_deidentified.jsonl

Output:
    data/management_deep_probe/statistics/S19_PairwiseSFTKappa.json

Run from the package root:
    python3 scripts/reproducibility/compute_s19_pairwise_sft_kappa.py
"""

from __future__ import annotations

import sys
from collections import Counter as _Counter
from itertools import combinations
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    EXPERTS_PATH,
    LABEL_ORDER_DISPLAY,
    SFT_DISPLAY,
    SFT_MODELS_PUBLIC,
    SFT_PREDICTIONS_PATH,
    STATISTICS_DIR,
    STUDENTS_PATH,
    expert_votes_for_article,
    load_jsonl,
    sft_pred,
    student_votes_for_article,
    write_stable_json,
)

OUTPUT_PATH = STATISTICS_DIR / "S19_PairwiseSFTKappa.json"


def cohens_kappa(y1: List[str], y2: List[str], labels: List[str]) -> Dict:
    if len(y1) != len(y2):
        raise ValueError("y1 and y2 must have the same length")
    n = len(y1)
    label_idx = {lab: i for i, lab in enumerate(labels)}
    cm = [[0.0] * len(labels) for _ in labels]
    for a, b in zip(y1, y2):
        cm[label_idx[a]][label_idx[b]] += 1
    p_o = sum(cm[i][i] for i in range(len(labels))) / n
    row_marg = [sum(row) / n for row in cm]
    col_marg = [sum(cm[i][j] for i in range(len(labels))) / n for j in range(len(labels))]
    p_e = sum(row_marg[i] * col_marg[i] for i in range(len(labels)))
    if abs(1 - p_e) < 1e-12:
        kappa = 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0
    else:
        kappa = (p_o - p_e) / (1 - p_e)
    return {"kappa": kappa, "p_o": p_o, "p_e": p_e, "n": n}


def mean_ordinal_distance(y1: List[str], y2: List[str]) -> float:
    idx = {lab: i for i, lab in enumerate(LABEL_ORDER_DISPLAY)}
    if not y1:
        return 0.0
    return sum(abs(idx[a] - idx[b]) for a, b in zip(y1, y2)) / len(y1)


def fleiss_kappa_from_votes(
    article_votes: List[List[str]], labels: List[str], n_raters_target: int
) -> Dict:
    label_idx = {lab: i for i, lab in enumerate(labels)}
    rows = []
    for votes in article_votes:
        if len(votes) < n_raters_target:
            continue
        sub = votes[:n_raters_target]
        row = [0] * len(labels)
        for v in sub:
            row[label_idx[v]] += 1
        rows.append(row)
    n_subjects = len(rows)
    if n_subjects == 0 or n_raters_target < 2:
        return {
            "kappa": None,
            "n_subjects": int(n_subjects),
            "n_raters": int(n_raters_target),
        }
    totals = [sum(rows[i][j] for i in range(n_subjects)) for j in range(len(labels))]
    p_j = [t / (n_subjects * n_raters_target) for t in totals]
    p_e = sum(p * p for p in p_j)
    p_i = [
        (sum(c * c for c in row) - n_raters_target)
        / (n_raters_target * (n_raters_target - 1))
        for row in rows
    ]
    p_o = sum(p_i) / n_subjects
    kappa = 0.0 if abs(1 - p_e) < 1e-12 else (p_o - p_e) / (1 - p_e)
    return {
        "kappa": float(kappa),
        "n_raters": int(n_raters_target),
        "n_subjects": int(n_subjects),
        "p_e": p_e,
        "p_o": p_o,
    }


def fleiss_kappa_min_truncation(votes: List[List[str]], labels: List[str]) -> Dict:
    counts = [len(v) for v in votes if v]
    if not counts:
        return {"kappa": None, "n_subjects": 0, "n_raters": 0}
    return fleiss_kappa_from_votes(votes, labels, min(counts))


def fleiss_kappa_modal(votes: List[List[str]], labels: List[str]) -> Dict:
    counts = [len(v) for v in votes if v]
    if not counts:
        return {"kappa": None, "n_subjects": 0, "n_raters": 0}
    modal = _Counter(counts).most_common(1)[0][0]
    return fleiss_kappa_from_votes(votes, labels, modal)


def main() -> int:
    sft = load_jsonl(SFT_PREDICTIONS_PATH)
    experts = load_jsonl(EXPERTS_PATH)
    students = load_jsonl(STUDENTS_PATH)

    preds: Dict[str, List[str]] = {k: [] for k in SFT_MODELS_PUBLIC}
    titles: List[str] = []
    for art in sft:
        titles.append(art["title"])
        for k in SFT_MODELS_PUBLIC:
            p = sft_pred(art, k)
            if p is None:
                raise ValueError(f"Missing SFT prediction for {k} on {art['title']!r}")
            preds[k].append(p)

    pairs_out: List[Dict] = []
    kappa_values: List[float] = []
    for a, b in combinations(SFT_MODELS_PUBLIC, 2):
        ck = cohens_kappa(preds[a], preds[b], LABEL_ORDER_DISPLAY)
        pairs_out.append(
            {
                "agreement_pct": ck["p_o"] * 100.0,
                "display_a": SFT_DISPLAY[a],
                "display_b": SFT_DISPLAY[b],
                "kappa": ck["kappa"],
                "mean_ordinal_distance": mean_ordinal_distance(preds[a], preds[b]),
                "model_a": a,
                "model_b": b,
                "n_articles": ck["n"],
            }
        )
        kappa_values.append(ck["kappa"])

    expert_votes = [expert_votes_for_article(a) for a in experts]
    student_votes = [student_votes_for_article(a) for a in students]
    output = {
        "human_kappa_contrast": {
            "expert_fleiss_min_truncation": fleiss_kappa_min_truncation(
                expert_votes, LABEL_ORDER_DISPLAY
            ),
            "expert_fleiss_modal": fleiss_kappa_modal(expert_votes, LABEL_ORDER_DISPLAY),
            "note": (
                "Fleiss' kappa across human raters per article on the same "
                "120-article 4-class label space. Two variants per pool: "
                "min-truncation (keep all articles, truncate to lowest "
                "rater count) and modal-truncation (drop articles below "
                "modal rater count, truncate to the mode). Both are "
                "reported so the editor can pick whichever the SI table "
                "prefers."
            ),
            "student_fleiss_min_truncation": fleiss_kappa_min_truncation(
                student_votes, LABEL_ORDER_DISPLAY
            ),
            "student_fleiss_modal": fleiss_kappa_modal(student_votes, LABEL_ORDER_DISPLAY),
        },
        "metadata": {
            "argmax_tie_break": "alphabetical (exceptional<fair<limited<strong)",
            "data_sources": {
                "experts": "data/management_deep_probe/human_ratings/expert_ratings_deidentified.jsonl",
                "sft_predictions": "data/management_deep_probe/predictions/sft_predictions.jsonl",
                "students": "data/management_deep_probe/human_ratings/student_ratings_deidentified.jsonl",
            },
            "label_space": LABEL_ORDER_DISPLAY,
            "n_articles": len(titles),
            "sft_model_keys": SFT_MODELS_PUBLIC,
            "supplementary_table": "ST21",
        },
        "sft_kappa_summary": {
            "max": max(kappa_values),
            "mean": sum(kappa_values) / len(kappa_values),
            "min": min(kappa_values),
            "n_pairs": len(pairs_out),
        },
        "sft_pairwise": pairs_out,
    }
    write_stable_json(OUTPUT_PATH, output)
    summary = output["sft_kappa_summary"]
    print(
        f"S19 OK: SFT pairwise kappa min={summary['min']:.3f} "
        f"max={summary['max']:.3f} mean={summary['mean']:.3f} "
        f"(n_pairs={summary['n_pairs']}); wrote {OUTPUT_PATH.relative_to(OUTPUT_PATH.parents[3])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
