#!/usr/bin/env python3
"""Reproduce S20: SFT 4/4 unanimous consensus subset, per-class coverage and accuracy.

Backs Supplementary Table ST22 (per-class consensus accuracy) and the
"42.5% coverage / 70.59% accuracy" line in the SI prose.

Inputs (read from this package):
    data/management_deep_probe/predictions/sft_predictions.jsonl

Output:
    data/management_deep_probe/statistics/S20_SFTConsensusPerClass.json

Run from the package root:
    python3 scripts/reproducibility/compute_s20_sft_consensus_per_class.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _common import (  # noqa: E402
    LABEL_ORDER_DISPLAY,
    PRIMARY_ENSEMBLE_KEY,
    SFT_MODELS_PUBLIC,
    SFT_PREDICTIONS_PATH,
    STATISTICS_DIR,
    article_ground_truth,
    load_jsonl,
    sft_pred,
    write_stable_json,
)

OUTPUT_PATH = STATISTICS_DIR / "S20_SFTConsensusPerClass.json"


def main() -> int:
    sft = load_jsonl(SFT_PREDICTIONS_PATH)

    n_total = len(sft)
    per_tier_total: Dict[str, int] = {t: 0 for t in LABEL_ORDER_DISPLAY}
    per_tier_consensus: Dict[str, int] = {t: 0 for t in LABEL_ORDER_DISPLAY}
    per_tier_consensus_correct: Dict[str, int] = {t: 0 for t in LABEL_ORDER_DISPLAY}
    per_tier_consensus_pred_dist: Dict[str, Dict[str, int]] = {
        t: {p: 0 for p in LABEL_ORDER_DISPLAY} for t in LABEL_ORDER_DISPLAY
    }

    n_consensus_correct = 0
    n_consensus_total = 0
    n_nonconsensus_total = 0
    n_nonconsensus_ensemble_correct = 0

    for art in sft:
        gt = article_ground_truth(art)
        per_tier_total[gt] += 1
        preds = [sft_pred(art, k) for k in SFT_MODELS_PUBLIC]
        if any(p is None for p in preds):
            raise ValueError(f"Missing SFT prediction on {art['title']!r}: {preds}")
        unanimous = len(set(preds)) == 1
        if unanimous:
            consensus_label = preds[0]
            per_tier_consensus[gt] += 1
            per_tier_consensus_pred_dist[gt][consensus_label] += 1
            n_consensus_total += 1
            if consensus_label == gt:
                per_tier_consensus_correct[gt] += 1
                n_consensus_correct += 1
        else:
            n_nonconsensus_total += 1
            ens_pred = sft_pred(art, PRIMARY_ENSEMBLE_KEY)
            if ens_pred == gt:
                n_nonconsensus_ensemble_correct += 1

    per_tier_rows: List[Dict] = []
    for tier in LABEL_ORDER_DISPLAY:
        total = per_tier_total[tier]
        cons = per_tier_consensus[tier]
        cons_correct = per_tier_consensus_correct[tier]
        per_tier_rows.append(
            {
                "consensus_accuracy": (cons_correct / cons) if cons else None,
                "consensus_prediction_distribution": per_tier_consensus_pred_dist[tier],
                "coverage": (cons / total) if total else 0.0,
                "n_consensus_articles": cons,
                "n_consensus_correct": cons_correct,
                "n_total": total,
                "tier": tier,
            }
        )

    output = {
        "metadata": {
            "argmax_tie_break": "alphabetical (exceptional<fair<limited<strong)",
            "data_source": "data/management_deep_probe/predictions/sft_predictions.jsonl",
            "label_space": LABEL_ORDER_DISPLAY,
            "n_total_articles": n_total,
            "primary_ensemble_key": PRIMARY_ENSEMBLE_KEY,
            "sft_model_keys": SFT_MODELS_PUBLIC,
            "supplementary_table": "ST22",
        },
        "per_tier": per_tier_rows,
        "summary": {
            "consensus_accuracy_overall": (
                n_consensus_correct / n_consensus_total if n_consensus_total else None
            ),
            "consensus_coverage_overall": n_consensus_total / n_total,
            "n_consensus_articles": n_consensus_total,
            "n_consensus_correct": n_consensus_correct,
            "n_non_consensus_articles": n_nonconsensus_total,
            "n_non_consensus_ensemble_correct": n_nonconsensus_ensemble_correct,
            "non_consensus_ensemble_accuracy": (
                n_nonconsensus_ensemble_correct / n_nonconsensus_total
                if n_nonconsensus_total
                else None
            ),
        },
    }
    write_stable_json(OUTPUT_PATH, output)
    s = output["summary"]
    print(
        f"S20 OK: consensus n={s['n_consensus_articles']}/{n_total} "
        f"({s['consensus_coverage_overall']*100:.1f}%), "
        f"acc={s['consensus_accuracy_overall']*100:.2f}%; "
        f"wrote {OUTPUT_PATH.relative_to(OUTPUT_PATH.parents[3])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
