#!/usr/bin/env python3
"""Script 25: Expert accuracy/reliability vs panel size (robust bootstrap).

Purpose:
    Quantify how expert panel size k affects:
    1) majority-vote accuracy (ties excluded)
    2) inter-rater reliability (Fleiss' kappa)
    3) tie rate and effective denominator

Method:
    Cluster bootstrap by article:
    - Resample articles with replacement
    - Per resampled article, sample k expert ratings without replacement
    - Compute metrics for that bootstrap sample
    - Repeat to obtain percentile confidence intervals

Outputs:
    results/tables/table_s25_expert_reliability_vs_size.csv
    results/figures/fig_s25_expert_accuracy_reliability_vs_size.png
    results/statistics/25_expert_reliability_vs_size.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import argparse
import csv
import json
from collections import defaultdict
from typing import Dict, List

import numpy as np
import matplotlib.pyplot as plt

from analysis.utils.constants import LABEL_ORDER, COLORS, FIGURES_DIR, TABLES_DIR, STATS_DIR
from analysis.utils.data_loader import get_project_root, load_expert_ratings, normalize_level
from analysis.utils.metrics import compute_accuracy
from analysis.utils.voting import majority_vote
from analysis.utils.statistical_tests import fleiss_kappa
from analysis.utils.visualization import save_figure, set_nature_style


def _extract_expert_articles() -> List[Dict]:
    """Return per-article expert vote lists."""
    data = load_expert_ratings()
    articles = []
    for art in data:
        gt = normalize_level(art.get("level", ""))
        preds = []
        for r in art.get("ratings", []):
            raw = r.get("q2_rating", "")
            if not raw:
                continue
            p = normalize_level(raw)
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


def _expert_individual_mean_accuracy() -> Dict[str, float]:
    """Compute per-rater mean accuracy (matches Script 07 definition)."""
    data = load_expert_ratings()
    rater_scores = defaultdict(lambda: {"correct": 0, "total": 0})

    for art in data:
        gt = normalize_level(art.get("level", ""))
        for r in art.get("ratings", []):
            raw = r.get("q2_rating", "")
            if not raw:
                continue
            p = normalize_level(raw)
            if p not in LABEL_ORDER:
                continue
            name = r.get("rater_id", r.get("rater_name", "unknown"))
            rater_scores[name]["total"] += 1
            if p == gt:
                rater_scores[name]["correct"] += 1

    accs = []
    for s in rater_scores.values():
        if s["total"] > 0:
            accs.append(s["correct"] / s["total"])

    if not accs:
        return {"n_raters": 0.0, "mean_accuracy": float("nan"), "std_accuracy": float("nan")}

    arr = np.asarray(accs, dtype=float)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return {
        "n_raters": float(arr.size),
        "mean_accuracy": float(arr.mean()),
        "std_accuracy": std,
    }


def _sample_k(preds: List[str], k: int, rng: np.random.RandomState) -> List[str]:
    idx = rng.choice(len(preds), size=k, replace=False)
    return [preds[i] for i in idx]


def _compute_metrics_one_boot(sampled_articles: List[Dict], k: int, rng: np.random.RandomState) -> Dict[str, float]:
    y_true = []
    y_pred = []
    n_correct_clear = 0
    tie_count = 0
    counts_matrix = []
    n_eligible = 0

    for art in sampled_articles:
        preds = art["predictions"]
        if len(preds) < k:
            continue
        n_eligible += 1
        sampled = _sample_k(preds, k, rng)

        counts_matrix.append([sampled.count(lbl) for lbl in LABEL_ORDER])

        vote, _ = majority_vote(sampled, tie_policy="exclude")
        if vote is None:
            tie_count += 1
            continue
        y_true.append(art["gt"])
        y_pred.append(vote)
        if vote == art["gt"]:
            n_correct_clear += 1

    if n_eligible == 0:
        return {
            "accuracy": float("nan"),
            "resolved_accuracy": float("nan"),
            "kappa": float("nan"),
            "tie_rate": float("nan"),
            "clear_vote_n": 0.0,
            "eligible_n": 0.0,
        }

    acc = compute_accuracy(y_true, y_pred) if y_true else float("nan")
    resolved_acc = n_correct_clear / n_eligible
    tie_rate = tie_count / n_eligible

    if len(counts_matrix) >= 2 and k >= 2:
        kappa = fleiss_kappa(np.asarray(counts_matrix, dtype=int))["kappa"]
    else:
        kappa = float("nan")

    return {
        "accuracy": float(acc),
        "resolved_accuracy": float(resolved_acc),
        "kappa": float(kappa),
        "tie_rate": float(tie_rate),
        "clear_vote_n": float(len(y_true)),
        "eligible_n": float(n_eligible),
    }


def _ci(values: List[float], alpha: float = 0.05) -> Dict[str, float]:
    arr = np.asarray([v for v in values if np.isfinite(v)], dtype=float)
    if arr.size == 0:
        return {"mean": float("nan"), "ci_lower": float("nan"), "ci_upper": float("nan")}
    return {
        "mean": float(np.mean(arr)),
        "ci_lower": float(np.percentile(arr, 100 * alpha / 2)),
        "ci_upper": float(np.percentile(arr, 100 * (1 - alpha / 2))),
    }


def main():
    parser = argparse.ArgumentParser(description="Expert reliability-vs-size robust bootstrap analysis")
    parser.add_argument("--sample-sizes", nargs="+", type=int, default=[1, 2, 3, 4])
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    set_nature_style()
    root = get_project_root()
    (root / FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    (root / TABLES_DIR).mkdir(parents=True, exist_ok=True)
    (root / STATS_DIR).mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("Script 25: Expert Accuracy/Reliability vs Panel Size")
    print("=" * 70)

    articles = _extract_expert_articles()
    n_articles = len(articles)
    min_raters = min(a["n_raters"] for a in articles)
    max_raters = max(a["n_raters"] for a in articles)
    mean_raters = np.mean([a["n_raters"] for a in articles])
    indiv = _expert_individual_mean_accuracy()

    print(f"\nLoaded {n_articles} expert-rated articles")
    print(f"Raters/article: min={min_raters}, max={max_raters}, mean={mean_raters:.2f}")
    print(
        f"Individual expert mean accuracy: {indiv['mean_accuracy']:.4f} "
        f"(n_raters={int(indiv['n_raters'])})"
    )

    sample_sizes = [k for k in sorted(set(args.sample_sizes)) if k >= 1]
    n_boot = args.n_bootstrap
    seed = args.seed
    master_rng = np.random.RandomState(seed)
    article_indices = np.arange(n_articles)

    rows = []
    stats = {
        "meta": {
            "n_articles": int(n_articles),
            "sample_sizes": sample_sizes,
            "n_bootstrap": int(n_boot),
            "seed": int(seed),
            "min_raters_per_article": int(min_raters),
            "max_raters_per_article": int(max_raters),
            "mean_raters_per_article": float(mean_raters),
            "individual_expert_mean_accuracy": float(indiv["mean_accuracy"]),
            "individual_expert_n_raters": int(indiv["n_raters"]),
        },
        "results": {},
    }

    for k in sample_sizes:
        print(f"\n[k={k}] Bootstrapping...", end="", flush=True)

        boot_acc = []
        boot_resolved = []
        boot_kappa = []
        boot_tie = []
        boot_clear = []
        boot_eligible = []

        for b in range(n_boot):
            sampled_idx = master_rng.choice(article_indices, size=n_articles, replace=True)
            sampled_articles = [articles[i] for i in sampled_idx]
            trial_rng = np.random.RandomState(seed + 200000 * k + b)
            out = _compute_metrics_one_boot(sampled_articles, k, trial_rng)

            boot_acc.append(out["accuracy"])
            boot_resolved.append(out["resolved_accuracy"])
            boot_kappa.append(out["kappa"])
            boot_tie.append(out["tie_rate"])
            boot_clear.append(out["clear_vote_n"])
            boot_eligible.append(out["eligible_n"])

        acc_ci = _ci(boot_acc)
        res_ci = _ci(boot_resolved)
        kap_ci = _ci(boot_kappa)
        tie_ci = _ci(boot_tie)
        clear_ci = _ci(boot_clear)
        elig_ci = _ci(boot_eligible)

        row = {
            "k": int(k),
            "accuracy_mean": acc_ci["mean"],
            "accuracy_ci_lower": acc_ci["ci_lower"],
            "accuracy_ci_upper": acc_ci["ci_upper"],
            "resolved_accuracy_mean": res_ci["mean"],
            "resolved_accuracy_ci_lower": res_ci["ci_lower"],
            "resolved_accuracy_ci_upper": res_ci["ci_upper"],
            "fleiss_kappa_mean": kap_ci["mean"],
            "fleiss_kappa_ci_lower": kap_ci["ci_lower"],
            "fleiss_kappa_ci_upper": kap_ci["ci_upper"],
            "tie_rate_mean": tie_ci["mean"],
            "tie_rate_ci_lower": tie_ci["ci_lower"],
            "tie_rate_ci_upper": tie_ci["ci_upper"],
            "clear_vote_n_mean": clear_ci["mean"],
            "clear_vote_n_ci_lower": clear_ci["ci_lower"],
            "clear_vote_n_ci_upper": clear_ci["ci_upper"],
            "eligible_n_mean": elig_ci["mean"],
            "eligible_n_ci_lower": elig_ci["ci_lower"],
            "eligible_n_ci_upper": elig_ci["ci_upper"],
            "n_bootstrap": int(n_boot),
        }
        rows.append(row)
        stats["results"][str(k)] = row

        kappa_msg = (
            f"{row['fleiss_kappa_mean']:.4f} "
            f"[{row['fleiss_kappa_ci_lower']:.4f}, {row['fleiss_kappa_ci_upper']:.4f}]"
            if np.isfinite(row["fleiss_kappa_mean"])
            else "NA"
        )
        print(
            f" done. acc={row['accuracy_mean']:.4f} "
            f"[{row['accuracy_ci_lower']:.4f}, {row['accuracy_ci_upper']:.4f}], "
            f"resolved={row['resolved_accuracy_mean']:.4f} "
            f"[{row['resolved_accuracy_ci_lower']:.4f}, {row['resolved_accuracy_ci_upper']:.4f}], "
            f"kappa={kappa_msg}, tie={row['tie_rate_mean']:.3f}"
        )

    rows_sorted = sorted(rows, key=lambda r: r["k"])
    table_path = root / TABLES_DIR / "table_s25_expert_reliability_vs_size.csv"
    with open(table_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows_sorted[0].keys()))
        writer.writeheader()
        writer.writerows(rows_sorted)

    # Plot
    x = np.asarray([r["k"] for r in rows_sorted], dtype=float)
    acc = np.asarray([r["accuracy_mean"] for r in rows_sorted], dtype=float)
    acc_lo = np.asarray([r["accuracy_ci_lower"] for r in rows_sorted], dtype=float)
    acc_hi = np.asarray([r["accuracy_ci_upper"] for r in rows_sorted], dtype=float)
    res = np.asarray([r["resolved_accuracy_mean"] for r in rows_sorted], dtype=float)
    res_lo = np.asarray([r["resolved_accuracy_ci_lower"] for r in rows_sorted], dtype=float)
    res_hi = np.asarray([r["resolved_accuracy_ci_upper"] for r in rows_sorted], dtype=float)
    kap = np.asarray([r["fleiss_kappa_mean"] for r in rows_sorted], dtype=float)
    kap_lo = np.asarray([r["fleiss_kappa_ci_lower"] for r in rows_sorted], dtype=float)
    kap_hi = np.asarray([r["fleiss_kappa_ci_upper"] for r in rows_sorted], dtype=float)
    tie = np.asarray([r["tie_rate_mean"] for r in rows_sorted], dtype=float)

    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0))

    # A: Accuracy
    ax0 = axes[0]
    ax0.plot(x, acc, "-o", color=COLORS["expert"], linewidth=1.2, markersize=3, label="Clear-vote accuracy")
    ax0.fill_between(x, acc_lo, acc_hi, color=COLORS["expert"], alpha=0.2)
    ax0.plot(x, res, "--o", color="#00897B", linewidth=1.1, markersize=3, label="Resolved accuracy")
    ax0.fill_between(x, res_lo, res_hi, color="#00897B", alpha=0.16)
    ax0.axhline(indiv["mean_accuracy"], color="black", linestyle="--", linewidth=0.9, label="Individual mean")
    ax0.axhline(0.25, color="gray", linestyle=":", linewidth=0.8, alpha=0.8)
    ax0.set_title("A. Expert Accuracy vs Panel Size", fontsize=9, fontweight="bold")
    ax0.set_xlabel("Expert panel size (k)")
    ax0.set_ylabel("Accuracy")
    ax0.set_xticks(x)
    ax0.legend(fontsize=6, loc="best")

    # B: Reliability + tie
    ax1 = axes[1]
    finite = np.isfinite(kap)
    if finite.any():
        ax1.plot(x[finite], kap[finite], "-o", color=COLORS["student"], linewidth=1.2, markersize=3, label="Fleiss' kappa")
        ax1.fill_between(x[finite], kap_lo[finite], kap_hi[finite], color=COLORS["student"], alpha=0.2)
    ax1.axhline(0.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.8)
    ax1.set_title("B. Expert Reliability vs Panel Size", fontsize=9, fontweight="bold")
    ax1.set_xlabel("Expert panel size (k)")
    ax1.set_ylabel("Fleiss' kappa")
    ax1.set_xticks(x)

    ax1b = ax1.twinx()
    ax1b.plot(x, tie, "--s", color="#555555", linewidth=1.0, markersize=2.8, label="Tie rate")
    ax1b.set_ylabel("Tie rate")

    lines1, labels1 = ax1.get_legend_handles_labels()
    lines2, labels2 = ax1b.get_legend_handles_labels()
    if lines1 or lines2:
        ax1.legend(lines1 + lines2, labels1 + labels2, fontsize=6, loc="best")

    fig.tight_layout()
    fig_path, _ = save_figure(fig, "fig_s25_expert_accuracy_reliability_vs_size")
    plt.close(fig)

    stats_path = root / STATS_DIR / "25_expert_reliability_vs_size.json"
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("\nSaved outputs:")
    print(f"  {table_path}")
    print(f"  {fig_path}")
    print(f"  {stats_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()
