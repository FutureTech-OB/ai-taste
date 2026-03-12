#!/usr/bin/env python3
"""Script 24: Student accuracy/reliability vs panel size (robust bootstrap).

Purpose:
    Estimate how student panel size (k raters/article) affects:
    1) majority-vote accuracy (ties excluded)
    2) inter-rater reliability (Fleiss' kappa)
    3) tie rate

Method:
    Cluster bootstrap by article:
    - Resample articles with replacement
    - For each resampled article, sample k student ratings without replacement
    - Compute metrics on that bootstrap sample
    - Repeat B times for percentile CIs

Outputs:
    results/tables/table_s24_student_reliability_vs_size.csv
    results/figures/fig_s24_student_accuracy_reliability_vs_size.png
    results/statistics/24_student_reliability_vs_size.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
import random
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from analysis.utils.constants import LABEL_ORDER, COLORS, FIGURES_DIR, TABLES_DIR, STATS_DIR
from analysis.utils.data_loader import (
    get_project_root,
    load_student_filtered_ratings,
    load_student_merged_ratings,
    normalize_level,
)
from analysis.utils.metrics import compute_accuracy
from analysis.utils.voting import majority_vote
from analysis.utils.statistical_tests import fleiss_kappa
from analysis.utils.visualization import save_figure, set_nature_style


def _extract_article_ratings() -> List[Dict]:
    """Load the filtered combined junior panel and return per-article rating lists."""
    data = load_student_filtered_ratings()
    articles = []
    for art in data:
        gt = normalize_level(art["level"])
        preds = []
        for r in art.get("ratings", []):
            p = normalize_level(r.get("q2_rating", ""))
            if p in LABEL_ORDER:
                preds.append(p)
        if preds:
            articles.append(
                {
                    "title": art.get("title", ""),
                    "gt": gt,
                    "predictions": preds,
                    "n_raters": len(preds),
                }
            )
    return articles


def _sample_k_predictions(preds: List[str], k: int, rng: np.random.RandomState) -> List[str]:
    """Sample k ratings without replacement from one article."""
    idx = rng.choice(len(preds), size=k, replace=False)
    return [preds[i] for i in idx]


def _compute_metrics_one_sample(sampled_articles: List[Dict], k: int, rng: np.random.RandomState) -> Dict[str, float]:
    """Compute metrics for one bootstrap sample at panel size k."""
    y_true = []
    y_pred = []
    tie_count = 0
    counts_matrix = []
    n_eligible = 0

    for art in sampled_articles:
        preds = art["predictions"]
        if len(preds) < k:
            continue
        n_eligible += 1
        sampled = _sample_k_predictions(preds, k, rng)

        # Reliability input: fixed-k category counts per article
        counts_matrix.append([sampled.count(lbl) for lbl in LABEL_ORDER])

        vote, _ = majority_vote(sampled, tie_policy="exclude")
        if vote is None:
            tie_count += 1
            continue
        y_true.append(art["gt"])
        y_pred.append(vote)

    if n_eligible == 0:
        return {
            "accuracy": float("nan"),
            "kappa": float("nan"),
            "tie_rate": float("nan"),
            "clear_vote_n": 0.0,
            "eligible_n": 0.0,
        }

    acc = compute_accuracy(y_true, y_pred) if y_true else float("nan")
    tie_rate = tie_count / n_eligible

    if len(counts_matrix) >= 2:
        kappa = fleiss_kappa(np.asarray(counts_matrix, dtype=int))["kappa"]
    else:
        kappa = float("nan")

    return {
        "accuracy": float(acc),
        "kappa": float(kappa),
        "tie_rate": float(tie_rate),
        "clear_vote_n": float(len(y_true)),
        "eligible_n": float(n_eligible),
    }


def _ci(values: List[float], alpha: float = 0.05) -> Dict[str, float]:
    """Return mean and percentile CI for finite values."""
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "ci_lower": float(np.percentile(arr, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(arr, 100 * (1 - alpha / 2))),
    }


def main():
    parser = argparse.ArgumentParser(description="Student reliability-vs-size robust bootstrap analysis")
    parser.add_argument(
        "--sample-sizes",
        nargs="+",
        type=int,
        default=[2, 3, 4, 5, 8, 10, 15, 20],
        help="Panel sizes k to evaluate",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=2000,
        help="Number of cluster-bootstrap resamples per k",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    args = parser.parse_args()

    set_nature_style()
    root = get_project_root()
    (root / FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    (root / TABLES_DIR).mkdir(parents=True, exist_ok=True)
    (root / STATS_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Script 24: Student Accuracy/Reliability vs Panel Size")
    print("=" * 70)

    articles = _extract_article_ratings()
    n_articles = len(articles)
    min_raters = min(a["n_raters"] for a in articles)
    max_raters = max(a["n_raters"] for a in articles)
    mean_raters = np.mean([a["n_raters"] for a in articles])
    print(f"\nLoaded {n_articles} articles from the filtered combined junior panel")
    print(f"Raters/article: min={min_raters}, max={max_raters}, mean={mean_raters:.2f}")

    sample_sizes = sorted(set(args.sample_sizes))
    n_boot = args.n_bootstrap
    seed = args.seed
    master_rng = np.random.RandomState(seed)

    rows = []
    stats_dump = {
        "meta": {
            "n_articles": int(n_articles),
            "sample_sizes": sample_sizes,
            "n_bootstrap": int(n_boot),
            "seed": int(seed),
            "min_raters_per_article": int(min_raters),
            "max_raters_per_article": int(max_raters),
            "mean_raters_per_article": float(mean_raters),
        },
        "results": {},
    }

    article_indices = np.arange(n_articles)

    for k in sample_sizes:
        print(f"\n[k={k}] Bootstrapping...", end="", flush=True)

        boot_acc = []
        boot_kappa = []
        boot_tie = []
        boot_clear_n = []
        boot_eligible_n = []

        for b in range(n_boot):
            # Cluster bootstrap: resample articles with replacement
            sampled_idx = master_rng.choice(article_indices, size=n_articles, replace=True)
            sampled_articles = [articles[i] for i in sampled_idx]

            # Separate RNG stream per bootstrap draw for within-article sampling
            trial_rng = np.random.RandomState(seed + 100000 * k + b)
            metrics = _compute_metrics_one_sample(sampled_articles, k, trial_rng)

            boot_acc.append(metrics["accuracy"])
            boot_kappa.append(metrics["kappa"])
            boot_tie.append(metrics["tie_rate"])
            boot_clear_n.append(metrics["clear_vote_n"])
            boot_eligible_n.append(metrics["eligible_n"])

        acc_ci = _ci(boot_acc)
        kappa_ci = _ci(boot_kappa)
        tie_ci = _ci(boot_tie)
        clear_n_ci = _ci(boot_clear_n)
        eligible_n_ci = _ci(boot_eligible_n)

        row = {
            "k": int(k),
            "accuracy_mean": acc_ci["mean"],
            "accuracy_ci_lower": acc_ci["ci_lower"],
            "accuracy_ci_upper": acc_ci["ci_upper"],
            "fleiss_kappa_mean": kappa_ci["mean"],
            "fleiss_kappa_ci_lower": kappa_ci["ci_lower"],
            "fleiss_kappa_ci_upper": kappa_ci["ci_upper"],
            "tie_rate_mean": tie_ci["mean"],
            "tie_rate_ci_lower": tie_ci["ci_lower"],
            "tie_rate_ci_upper": tie_ci["ci_upper"],
            "clear_vote_n_mean": clear_n_ci["mean"],
            "clear_vote_n_ci_lower": clear_n_ci["ci_lower"],
            "clear_vote_n_ci_upper": clear_n_ci["ci_upper"],
            "eligible_n_mean": eligible_n_ci["mean"],
            "eligible_n_ci_lower": eligible_n_ci["ci_lower"],
            "eligible_n_ci_upper": eligible_n_ci["ci_upper"],
            "n_bootstrap": int(n_boot),
        }
        rows.append(row)
        stats_dump["results"][str(k)] = row

        print(
            f" done. acc={row['accuracy_mean']:.4f} "
            f"[{row['accuracy_ci_lower']:.4f}, {row['accuracy_ci_upper']:.4f}], "
            f"kappa={row['fleiss_kappa_mean']:.4f} "
            f"[{row['fleiss_kappa_ci_lower']:.4f}, {row['fleiss_kappa_ci_upper']:.4f}]"
        )

    # Save table
    rows_sorted = sorted(rows, key=lambda r: r["k"])
    table_path = root / TABLES_DIR / "table_s24_student_reliability_vs_size.csv"
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    # Plot: two-panel (accuracy + reliability)
    x = np.asarray([r["k"] for r in rows_sorted], dtype=float)
    acc = np.asarray([r["accuracy_mean"] for r in rows_sorted], dtype=float)
    acc_lo = np.asarray([r["accuracy_ci_lower"] for r in rows_sorted], dtype=float)
    acc_hi = np.asarray([r["accuracy_ci_upper"] for r in rows_sorted], dtype=float)
    kap = np.asarray([r["fleiss_kappa_mean"] for r in rows_sorted], dtype=float)
    kap_lo = np.asarray([r["fleiss_kappa_ci_lower"] for r in rows_sorted], dtype=float)
    kap_hi = np.asarray([r["fleiss_kappa_ci_upper"] for r in rows_sorted], dtype=float)
    tie = np.asarray([r["tie_rate_mean"] for r in rows_sorted], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # Panel A: Accuracy
    ax0 = axes[0]
    ax0.plot(x, acc, "-o", color=COLORS["ai"], linewidth=1.2, markersize=3)
    ax0.fill_between(x, acc_lo, acc_hi, color=COLORS["ai"], alpha=0.2)
    ax0.axhline(0.25, color="gray", linestyle=":", linewidth=0.8, alpha=0.8)
    ax0.set_title("A. Accuracy vs Panel Size", fontsize=9, fontweight="bold")
    ax0.set_xlabel("Student panel size (k)")
    ax0.set_ylabel("Majority-vote accuracy")
    ax0.set_xticks(x)

    # Panel B: Reliability + tie rate
    ax1 = axes[1]
    ax1.plot(x, kap, "-o", color=COLORS["student"], linewidth=1.2, markersize=3, label="Fleiss' kappa")
    ax1.fill_between(x, kap_lo, kap_hi, color=COLORS["student"], alpha=0.2)
    ax1.axhline(0.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.8)
    ax1.set_title("B. Reliability vs Panel Size", fontsize=9, fontweight="bold")
    ax1.set_xlabel("Student panel size (k)")
    ax1.set_ylabel("Fleiss' kappa")
    ax1.set_xticks(x)

    # Tie-rate overlay as a secondary axis
    ax1b = ax1.twinx()
    ax1b.plot(x, tie, "--s", color="#555555", linewidth=1.0, markersize=2.8, label="Tie rate")
    ax1b.set_ylabel("Tie rate")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="best")

    fig.tight_layout()
    fig_path, _ = save_figure(fig, "fig_s24_student_accuracy_reliability_vs_size")
    plt.close(fig)

    # Save stats JSON
    stats_path = root / STATS_DIR / "24_student_reliability_vs_size.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_dump, f, indent=2, ensure_ascii=False)

    print("\nSaved outputs:")
    print(f"  {table_path}")
    print(f"  {fig_path}")
    print(f"  {stats_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
