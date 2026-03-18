#!/usr/bin/env python3
"""Script 02: Frontier model performance under canonical 8-run protocol.

Protocol:
- Primary metric:
  - stochastic frontier models: article-level ``avg_accuracy`` (avg8 mean)
  - deterministic frontier models: single-pass discrete accuracy
- Discrete analyses (macro-F1, per-tier, confusion, McNemar):
  - majority-vote labels (ties excluded)
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
from collections import Counter
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analysis.utils.constants import (
    AI_TO_HUMAN,
    COLORS,
    FIGURES_DIR,
    FRONTIER_DISPLAY_NAMES,
    FRONTIER_MODELS,
    LABEL_ORDER,
    STATS_DIR,
    TABLES_DIR,
)
from analysis.utils.data_loader import get_ground_truth, get_project_root, load_frontier_data
from analysis.utils.frontier_protocol import (
    assert_frontier_models_present,
    extract_discrete_prediction,
    full_predictions,
    model_mode,
)
from analysis.utils.metrics import (
    compute_accuracy,
    compute_confusion_matrix,
    compute_macro_f1,
    compute_per_class_metrics,
)
from analysis.utils.statistical_tests import bonferroni_correction, cochran_q_test, mcnemar_test
from analysis.utils.visualization import plot_confusion_matrix, plot_heatmap, save_figure, set_nature_style


def _article_primary_score(article: Dict, model_key: str, mode: str) -> Optional[float]:
    model_output = (
        article.get("val_outcome", {})
        .get("rq_with_context", {})
        .get(model_key, {})
    )
    gt = get_ground_truth(article)

    if mode == "stochastic_thinking":
        avg_acc = model_output.get("avg_accuracy")
        try:
            if avg_acc is not None:
                return float(avg_acc)
        except (TypeError, ValueError):
            pass

    pred, _ = extract_discrete_prediction(
        model_output,
        policy="prediction",
        tie_policy="run1_fallback",
    )
    if pred is None:
        return None
    return 1.0 if pred == gt else 0.0


def _paired_vectors(
    y_true: List[str],
    preds: List[Optional[str]],
) -> tuple[List[str], List[str]]:
    paired = [(t, p) for t, p in zip(y_true, preds) if p is not None]
    return [t for t, _ in paired], [p for _, p in paired]


def main():
    print("=" * 70)
    print("Script 02: Frontier Model Performance (Canonical Protocol)")
    print("=" * 70)

    project_root = get_project_root()
    figures_dir = project_root / FIGURES_DIR
    tables_dir = project_root / TABLES_DIR
    stats_dir = project_root / STATS_DIR
    for d in (figures_dir, tables_dir, stats_dir):
        d.mkdir(parents=True, exist_ok=True)

    print("\n[1/6] Loading canonical frontier data...")
    frontier_data = load_frontier_data()
    assert_frontier_models_present(frontier_data, FRONTIER_MODELS)
    y_true_all = [get_ground_truth(rec) for rec in frontier_data]
    print(f"  Records loaded: {len(frontier_data)}")
    print(f"  Ground truth distribution: {Counter(y_true_all)}")

    model_keys = list(FRONTIER_MODELS)
    print(f"  Models found: {len(model_keys)}")
    for mk in model_keys:
        print(f"    - {mk} -> {FRONTIER_DISPLAY_NAMES.get(mk, mk)}")

    print("\n[2/6] Computing protocol-consistent metrics...")
    results_rows = []
    model_stats: Dict[str, Dict] = {}
    predictions_majority: Dict[str, List[Optional[str]]] = {}

    for model_key in model_keys:
        display_name = FRONTIER_DISPLAY_NAMES.get(model_key, model_key)
        mode = model_mode(frontier_data, model_key)

        preds_majority_full = full_predictions(
            frontier_data,
            model_key,
            policy="majority",
            tie_policy="exclude",
        )
        preds_run1_full = full_predictions(
            frontier_data,
            model_key,
            policy="run1",
            tie_policy="run1_fallback",
        )
        predictions_majority[model_key] = preds_majority_full

        y_true_majority, y_pred_majority = _paired_vectors(y_true_all, preds_majority_full)
        y_true_run1, y_pred_run1 = _paired_vectors(y_true_all, preds_run1_full)

        n_total = len(y_true_all)
        n_discrete = len(y_pred_majority)
        n_ties = sum(1 for p in preds_majority_full if p is None)
        tie_rate = n_ties / n_total if n_total else 0.0

        acc_majority = compute_accuracy(y_true_majority, y_pred_majority) if y_pred_majority else 0.0
        acc_run1 = compute_accuracy(y_true_run1, y_pred_run1) if y_pred_run1 else 0.0

        primary_scores: List[float] = []
        for rec in frontier_data:
            score = _article_primary_score(rec, model_key, mode)
            if score is not None:
                primary_scores.append(score)
        acc_primary = float(np.mean(primary_scores)) if primary_scores else 0.0

        macro_f1 = compute_macro_f1(y_true_majority, y_pred_majority) if y_pred_majority else 0.0
        per_class = compute_per_class_metrics(y_true_majority, y_pred_majority) if y_pred_majority else {}
        cm = compute_confusion_matrix(y_true_majority, y_pred_majority) if y_pred_majority else np.zeros((4, 4))
        pred_dist = Counter(y_pred_majority)

        per_tier_acc = {}
        for label in LABEL_ORDER:
            tier_true = [t for t in y_true_majority if t == label]
            tier_pred = [p for t, p in zip(y_true_majority, y_pred_majority) if t == label]
            per_tier_acc[label] = compute_accuracy(tier_true, tier_pred) if tier_true else float("nan")

        row = {
            "Model": display_name,
            "Model Key": model_key,
            "Mode": mode,
            "N Valid": len(primary_scores),
            "N Discrete (excl ties)": n_discrete,
            "Tie Rate": round(tie_rate, 4),
            "Accuracy": round(acc_primary, 4),
            "Accuracy Majority": round(acc_majority, 4),
            "Accuracy Run1": round(acc_run1, 4),
            "Macro F1": round(macro_f1, 4),
        }
        for label in LABEL_ORDER:
            row[f"Acc ({label})"] = round(per_tier_acc.get(label, float("nan")), 4)
        for label in LABEL_ORDER:
            row[f"Pred count ({label})"] = pred_dist.get(label, 0)
        results_rows.append(row)

        model_stats[model_key] = {
            "display_name": display_name,
            "mode": mode,
            "n_valid_primary": len(primary_scores),
            "n_valid_discrete": n_discrete,
            "tie_rate": tie_rate,
            "accuracy": acc_primary,  # backward-compatible key (primary metric)
            "accuracy_primary": acc_primary,
            "accuracy_majority": acc_majority,
            "accuracy_run1": acc_run1,
            "macro_f1": macro_f1,
            "per_class_metrics": {
                k: {mk: round(mv, 4) for mk, mv in v.items()}
                for k, v in per_class.items()
            },
            "per_tier_accuracy": {k: round(v, 4) for k, v in per_tier_acc.items()},
            "prediction_distribution": dict(pred_dist),
            "confusion_matrix": cm.tolist(),
        }

        print(
            f"  {display_name}: primary={acc_primary:.3f}, "
            f"majority={acc_majority:.3f}, run1={acc_run1:.3f}, "
            f"mode={mode}, ties={n_ties}"
        )

    results_rows.sort(key=lambda r: r["Accuracy"], reverse=True)

    table3 = pd.DataFrame(results_rows)
    table3_path = tables_dir / "table3_flagship_performance.csv"
    table3.to_csv(table3_path, index=False)
    print(f"\n  Saved: {table3_path}")

    print("\n[3/6] Running Cochran's Q and pairwise McNemar tests...")
    full_models = {
        k: [p for p in v if p is not None]
        for k, v in predictions_majority.items()
        if all(p is not None for p in v)
    }
    if len(full_models) >= 2:
        cochran_result = cochran_q_test(y_true_all, full_models)
        print(
            f"  Cochran's Q: Q={cochran_result['statistic']:.2f}, "
            f"df={cochran_result['df']}, p={cochran_result['p_value']:.6f}"
        )
    else:
        common_valid = [True] * len(y_true_all)
        for preds in predictions_majority.values():
            common_valid = [cv and (p is not None) for cv, p in zip(common_valid, preds)]
        y_true_common = [t for t, keep in zip(y_true_all, common_valid) if keep]
        preds_common = {
            k: [p for p, keep in zip(preds, common_valid) if keep]
            for k, preds in predictions_majority.items()
        }
        cochran_result = cochran_q_test(y_true_common, preds_common)
        print(
            f"  Cochran's Q (common non-tie subset, n={len(y_true_common)}): "
            f"Q={cochran_result['statistic']:.2f}, df={cochran_result['df']}, "
            f"p={cochran_result['p_value']:.6f}"
        )

    model_key_list = list(predictions_majority.keys())
    mcnemar_results: Dict[str, Dict] = {}
    raw_p_values: List[float] = []
    pair_labels: List[str] = []

    for i in range(len(model_key_list)):
        for j in range(i + 1, len(model_key_list)):
            m1, m2 = model_key_list[i], model_key_list[j]
            p1, p2 = predictions_majority[m1], predictions_majority[m2]

            valid = [a is not None and b is not None for a, b in zip(p1, p2)]
            yt = [t for t, v in zip(y_true_all, valid) if v]
            yp1 = [p for p, v in zip(p1, valid) if v]
            yp2 = [p for p, v in zip(p2, valid) if v]
            if len(yt) < 10:
                continue

            result = mcnemar_test(yt, yp1, yp2)
            pair_key = f"{FRONTIER_DISPLAY_NAMES.get(m1, m1)} vs {FRONTIER_DISPLAY_NAMES.get(m2, m2)}"
            mcnemar_results[pair_key] = {
                "statistic": result["statistic"],
                "p_value": result["p_value"],
                "b": result["b"],
                "c": result["c"],
                "n_paired": len(yt),
            }
            raw_p_values.append(result["p_value"])
            pair_labels.append(pair_key)

    if raw_p_values:
        bonf = bonferroni_correction(raw_p_values)
        for idx, pair_key in enumerate(pair_labels):
            mcnemar_results[pair_key]["p_corrected"] = bonf["corrected_p_values"][idx]
            mcnemar_results[pair_key]["significant"] = bonf["rejected"][idx]
        print(
            f"  Pairwise McNemar: {len(raw_p_values)} pairs tested, "
            f"{sum(bonf['rejected'])} significant after Bonferroni correction"
        )

    print("\n[4/6] Creating figures...")
    set_nature_style()

    sorted_models = sorted(model_stats.items(), key=lambda x: x[1]["accuracy_primary"])
    fig, ax = plt.subplots(figsize=(5.8, 4.2))
    y_pos = np.arange(len(sorted_models))
    bar_height = 0.34

    display_names = [FRONTIER_DISPLAY_NAMES.get(k, k) for k, _ in sorted_models]
    accuracies = [s["accuracy_primary"] for _, s in sorted_models]
    f1_scores = [s["macro_f1"] for _, s in sorted_models]

    bars_acc = ax.barh(
        y_pos + bar_height / 2,
        accuracies,
        bar_height,
        label="Primary Accuracy",
        color=COLORS["ai"],
        edgecolor="white",
        linewidth=0.3,
    )
    bars_f1 = ax.barh(
        y_pos - bar_height / 2,
        f1_scores,
        bar_height,
        label="Macro F1 (majority labels)",
        color=COLORS["strong"],
        edgecolor="white",
        linewidth=0.3,
    )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(display_names, fontsize=7)
    ax.set_xlabel("Score")
    ax.set_xlim(0, 1.0)
    ax.set_title("Frontier model performance comparison", fontsize=9, fontweight="bold")
    ax.legend(fontsize=6.3, loc="lower right")

    for bar_obj in list(bars_acc) + list(bars_f1):
        width = bar_obj.get_width()
        ax.text(width + 0.01, bar_obj.get_y() + bar_obj.get_height() / 2, f"{width:.3f}", va="center", fontsize=5.2)

    fig.tight_layout()
    pdf_path, _ = save_figure(fig, "fig2_flagship_accuracy_comparison")
    plt.close(fig)
    print(f"  Saved: {pdf_path}")

    sorted_by_acc = sorted(model_stats.items(), key=lambda x: x[1]["accuracy_primary"], reverse=True)
    heatmap_models = [FRONTIER_DISPLAY_NAMES.get(k, k) for k, _ in sorted_by_acc]
    heatmap_data = np.array(
        [[s["per_tier_accuracy"].get(label, 0.0) for label in LABEL_ORDER] for _, s in sorted_by_acc]
    )
    fig, _ = plot_heatmap(
        heatmap_data,
        row_labels=heatmap_models,
        col_labels=[l.capitalize() for l in LABEL_ORDER],
        title="Per-tier accuracy: frontier models (majority labels)",
        cmap="RdYlGn",
    )
    pdf_path, _ = save_figure(fig, "fig_s2_pertier_heatmap")
    plt.close(fig)
    print(f"  Saved: {pdf_path}")

    top4 = sorted_by_acc[:4]
    cm_labels = [l.capitalize() for l in LABEL_ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(7.0, 6.0))
    axes_flat = axes.flatten()
    for idx, (model_key, stats_entry) in enumerate(top4):
        cm_array = np.array(stats_entry["confusion_matrix"])
        plot_confusion_matrix(
            cm_array,
            labels=cm_labels,
            title=FRONTIER_DISPLAY_NAMES.get(model_key, model_key),
            ax=axes_flat[idx],
            normalize=True,
        )
    fig.suptitle("Confusion matrices: top 4 frontier models", fontsize=10, fontweight="bold", y=1.02)
    fig.tight_layout()
    pdf_path, _ = save_figure(fig, "fig_s3_confusion_matrices")
    plt.close(fig)
    print(f"  Saved: {pdf_path}")

    print("\n[5/6] Saving statistics JSON...")
    all_stats = {
        "protocol": {
            "frontier_primary_metric": "avg8_article_mean_for_stochastic__discrete_for_deterministic",
            "frontier_discrete_policy": "majority_vote_labels_excluding_ties",
        },
        "n_articles": len(frontier_data),
        "n_models": len(model_keys),
        "model_keys": model_keys,
        "model_performance": {
            FRONTIER_DISPLAY_NAMES.get(k, k): {
                "model_key": k,
                "mode": v["mode"],
                "accuracy": round(v["accuracy_primary"], 4),  # backward-compatible primary track
                "accuracy_primary": round(v["accuracy_primary"], 4),
                "accuracy_majority": round(v["accuracy_majority"], 4),
                "accuracy_run1": round(v["accuracy_run1"], 4),
                "macro_f1": round(v["macro_f1"], 4),
                "n_valid_primary": v["n_valid_primary"],
                "n_valid_discrete": v["n_valid_discrete"],
                "tie_rate": round(v["tie_rate"], 4),
                "per_tier_accuracy": v["per_tier_accuracy"],
                "prediction_distribution": v["prediction_distribution"],
                "per_class_metrics": v["per_class_metrics"],
                "confusion_matrix": v["confusion_matrix"],
            }
            for k, v in model_stats.items()
        },
        "ranking_by_accuracy": [
            {
                "rank": i + 1,
                "model": FRONTIER_DISPLAY_NAMES.get(k, k),
                "accuracy_primary": round(v["accuracy_primary"], 4),
                "accuracy_majority": round(v["accuracy_majority"], 4),
                "accuracy_run1": round(v["accuracy_run1"], 4),
                "macro_f1": round(v["macro_f1"], 4),
            }
            for i, (k, v) in enumerate(sorted_by_acc)
        ],
        "cochran_q_test": {
            "statistic": cochran_result["statistic"],
            "p_value": cochran_result["p_value"],
            "df": cochran_result["df"],
            "n_classifiers": cochran_result["n_classifiers"],
            "interpretation": (
                "Significant differences exist among models"
                if cochran_result["p_value"] < 0.05
                else "No significant differences among models"
            ),
        },
        "pairwise_mcnemar": mcnemar_results,
        "bonferroni_n_tests": len(raw_p_values),
        "bonferroni_n_significant": sum(1 for v in mcnemar_results.values() if v.get("significant", False)),
    }

    stats_path = stats_dir / "02_flagship.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {stats_path}")

    print("\n[6/6] Complete.")
    print("=" * 70)


if __name__ == "__main__":
    main()
