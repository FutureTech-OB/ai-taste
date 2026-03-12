#!/usr/bin/env python3
"""Script 23: Gatekeeper Decision Profile (Comprehensive).

Compares decision behavior across:
- all 11 flagship models,
- all SFT models (4 checkpoints + 2-model ensemble),
- human evaluator variants:
  - expert individual (pooled ratings + per-rater mean),
  - expert majority vote,
  - student individual (pooled ratings + per-rater mean),
  - student majority vote,
  - student matched-N vote (student sample size matched to expert counts).

Primary outputs:
- results/statistics/23_gatekeeper_profile.json
- results/tables/table23_gatekeeper_profile_main.csv
- results/tables/table23_gatekeeper_per_class.csv
- results/tables/table23_human_variants_summary.csv
- results/tables/table23_student_matched_n_trials.csv
- results/figures/fig23_gatekeeper_profile_main.png
- results/figures/fig_s23_gatekeeper_binary_slices.png
- results/figures/fig23_all_evaluators_accuracy.png
"""

from __future__ import annotations

import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.utils.constants import (  # noqa: E402
    FIGURES_DIR,
    FRONTIER_DISPLAY_NAMES,
    FRONTIER_MODELS,
    HUMAN_TO_AI,
    LABEL_ORDER,
    STATS_DIR,
    TABLES_DIR,
)
from analysis.utils.data_loader import (  # noqa: E402
    get_ground_truth,
    get_project_root,
    load_chat_data,
    load_expert_ratings,
    load_student_filtered_ratings,
    load_student_merged_ratings,
    load_thinking_data,
    load_trained_data,
    normalize_level,
)
from analysis.utils.frontier_protocol import extract_discrete_prediction  # noqa: E402
from analysis.utils.metrics import (  # noqa: E402
    compute_accuracy,
    compute_macro_f1,
    compute_per_class_metrics,
    compute_prediction_distribution,
    get_top_predictions,
    logp_to_prob,
)
from analysis.utils.visualization import set_nature_style  # noqa: E402
from analysis.utils.voting import majority_vote  # noqa: E402


RANDOM_SEED = 42
MATCHED_N_TRIALS = 5000

FLAGSHIP_KEYS = list(FRONTIER_MODELS)

FLAGSHIP_DISPLAY = {
    key: f"Flagship: {FRONTIER_DISPLAY_NAMES.get(key, key)}"
    for key in FLAGSHIP_KEYS
}

SFT_KEYS = [
    "CYqJRxId",
    "ckpt-step-304",
    "ckppt-380",
    "ckppt-228",
    "best_2_model_combo",
]

SFT_DISPLAY = {
    "CYqJRxId": "SFT: GPT-4.1",
    "ckpt-step-304": "SFT: GPT-4.1-nano",
    "ckppt-380": "SFT: Qwen3-30B",
    "ckppt-228": "SFT: Qwen3-4B",
    "best_2_model_combo": "SFT: 2-Model Ensemble",
}

CHAT_BASELINE_KEY = "gpt-5.2"
CHAT_BASELINE_DISPLAY = "Chat Baseline: GPT-5.2"

HUMAN_VARIANTS = [
    "Expert Individual (Pooled)",
    "Expert Majority (Unfiltered)",
    "Student Individual (Pooled)",
    "Student Majority (Filtered)",
    "Student Matched-N Consensus (Expert Sized)",
]

KEY_FIG_ORDER = [
    "Flagship: Gemini 3.1 Pro",
    "Chat Baseline: GPT-5.2",
    "SFT: 2-Model Ensemble",
    "Expert Majority (Unfiltered)",
    "Student Majority (Filtered)",
]

LABEL_TO_SCORE = {
    "limited": 1,
    "fair": 2,
    "strong": 3,
    "exceptional": 4,
}

GROUP_COLORS = {
    "flagship": "#C62828",
    "chat": "#8E24AA",
    "sft": "#FF8A65",
    "human": "#1565C0",
}


def _normalize_prediction_label(raw: Optional[str]) -> Optional[str]:
    """Normalize prediction text to AI label space."""
    if raw is None:
        return None
    value = str(raw).strip().strip("*").strip().lower()
    mapping = {
        "exceptional": "exceptional",
        "strong": "strong",
        "fair": "fair",
        "limited": "limited",
        "top": "exceptional",
        "top-": "strong",
        "good": "fair",
    }
    return mapping.get(value)


def _pred_from_logp(logp_dict: Optional[Dict[str, float]]) -> Optional[str]:
    """Get prediction from log-probabilities."""
    if not isinstance(logp_dict, dict) or not logp_dict:
        return None
    probs = logp_to_prob(logp_dict)
    top = get_top_predictions(probs, top_k=1)
    if not top:
        return None
    return _normalize_prediction_label(top[0][0])


def _pred_from_thinking_model(model_data: Dict[str, Any]) -> Optional[str]:
    """Get protocol-consistent majority prediction from frontier entry."""
    pred, _ = extract_discrete_prediction(
        model_data,
        policy="majority",
        tie_policy="exclude",
    )
    if pred is not None:
        return _normalize_prediction_label(pred)
    return None


def extract_model_predictions(
    data: List[Dict[str, Any]],
    model_key: str,
    source: str,
) -> Tuple[List[str], List[str]]:
    """Extract y_true/y_pred pairs for one model."""
    y_true: List[str] = []
    y_pred: List[str] = []

    for article in data:
        gt = get_ground_truth(article)
        rq = article.get("val_outcome", {}).get("rq_with_context", {})
        model_data = rq.get(model_key, {})

        if source == "thinking":
            pred = _pred_from_thinking_model(model_data)
        else:
            pred = _pred_from_logp(model_data.get("logp"))

        if pred is None:
            continue
        y_true.append(gt)
        y_pred.append(pred)

    return y_true, y_pred


def build_human_vote_records(articles: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Build per-article records with normalized human votes."""
    records: List[Dict[str, Any]] = []
    for article in articles:
        gt = normalize_level(article.get("level", ""))
        votes: List[str] = []
        for rating in article.get("ratings", []):
            raw = rating.get("q2_rating")
            if raw in HUMAN_TO_AI:
                votes.append(HUMAN_TO_AI[raw])
        records.append({
            "title": article.get("title", ""),
            "gt": gt,
            "votes": votes,
        })
    return records


def compute_human_majority_from_records(
    records: List[Dict[str, Any]],
) -> Tuple[List[str], List[Optional[str]], List[int], int]:
    """Compute majority predictions (tie excluded) across full article list."""
    y_true: List[str] = []
    y_pred: List[Optional[str]] = []
    vote_counts: List[int] = []
    n_ties = 0

    for rec in records:
        gt = rec["gt"]
        votes = rec["votes"]
        y_true.append(gt)
        vote_counts.append(len(votes))

        if not votes:
            y_pred.append(None)
            continue

        pred, is_tie = majority_vote(votes, tie_policy="exclude")
        if is_tie or pred is None:
            n_ties += 1
            y_pred.append(None)
        else:
            y_pred.append(pred)

    return y_true, y_pred, vote_counts, n_ties


def filter_valid_pairs(
    y_true: List[str],
    y_pred: List[Optional[str]],
) -> Tuple[List[str], List[str]]:
    """Drop entries where prediction is None."""
    pairs = [(t, p) for t, p in zip(y_true, y_pred) if p is not None]
    return [t for t, _ in pairs], [p for _, p in pairs]


def compute_human_individual_pooled_and_summary(
    articles: List[Dict[str, Any]],
) -> Dict[str, Any]:
    """Compute pooled individual predictions + per-rater average accuracy."""
    y_true: List[str] = []
    y_pred: List[str] = []
    rater_scores = defaultdict(lambda: {"correct": 0, "total": 0})

    for article in articles:
        gt = normalize_level(article.get("level", ""))
        for rating in article.get("ratings", []):
            raw = rating.get("q2_rating")
            if raw not in HUMAN_TO_AI:
                continue
            pred = HUMAN_TO_AI[raw]
            y_true.append(gt)
            y_pred.append(pred)

            rater = rating.get("rater_id", rating.get("rater_name", "unknown"))
            rater_scores[rater]["total"] += 1
            if pred == gt:
                rater_scores[rater]["correct"] += 1

    per_rater_acc = []
    for _, s in rater_scores.items():
        if s["total"] > 0:
            per_rater_acc.append(s["correct"] / s["total"])

    return {
        "y_true": y_true,
        "y_pred": y_pred,
        "n_raters": len(per_rater_acc),
        "mean_accuracy": float(np.mean(per_rater_acc)) if per_rater_acc else 0.0,
        "std_accuracy": float(np.std(per_rater_acc, ddof=1)) if len(per_rater_acc) > 1 else 0.0,
        "per_rater_accuracies": per_rater_acc,
        "pooled_accuracy": float(compute_accuracy(y_true, y_pred)) if y_true else 0.0,
    }


def compute_student_matched_n_consensus(
    student_records: List[Dict[str, Any]],
    expert_counts: List[int],
    n_trials: int = MATCHED_N_TRIALS,
    seed: int = RANDOM_SEED,
) -> Dict[str, Any]:
    """Monte Carlo student matched-N and consensus predictions by article.

    Uses Script-07-consistent design:
    - sample student votes per article to match expert count on that article,
    - apply majority vote with tie exclusion,
    - track trial accuracy on clear-majority articles.
    """
    np_rng = np.random.RandomState(seed)

    article_pred_counts = [Counter() for _ in student_records]
    trial_accuracies: List[float] = []
    trial_effective_ns: List[int] = []

    for _ in range(n_trials):
        correct = 0
        total = 0

        for i, rec in enumerate(student_records):
            votes = rec["votes"]
            gt = rec["gt"]
            n_sample = expert_counts[i] if i < len(expert_counts) else 0
            n_sample = min(n_sample, len(votes))

            if n_sample <= 0:
                continue

            indices = np_rng.choice(len(votes), size=n_sample, replace=False)
            sampled = [votes[idx] for idx in indices]
            pred, _ = majority_vote(sampled, tie_policy="exclude")

            if pred is None:
                continue

            article_pred_counts[i][pred] += 1
            total += 1
            if pred == gt:
                correct += 1

        trial_accuracies.append(correct / total if total > 0 else 0.0)
        trial_effective_ns.append(total)

    # Consensus prediction per article from trial vote frequency
    consensus_preds: List[Optional[str]] = []
    coverage_by_article: List[float] = []
    for i, rec in enumerate(student_records):
        counts = article_pred_counts[i]
        if not counts:
            consensus_preds.append(None)
            coverage_by_article.append(0.0)
            continue

        top_count = max(counts.values())
        top_labels = [k for k, v in counts.items() if v == top_count]
        if len(top_labels) == 1:
            consensus_preds.append(top_labels[0])
        else:
            consensus_preds.append(None)
        coverage_by_article.append(sum(counts.values()) / n_trials)

    acc_arr = np.array(trial_accuracies, dtype=float)
    n_arr = np.array(trial_effective_ns, dtype=float)

    return {
        "ground_truths": [r["gt"] for r in student_records],
        "consensus_preds": consensus_preds,
        "coverage_by_article": coverage_by_article,
        "trial_accuracies": trial_accuracies,
        "trial_effective_ns": trial_effective_ns,
        "mean_accuracy": float(np.mean(acc_arr)) if len(acc_arr) > 0 else 0.0,
        "std_accuracy": float(np.std(acc_arr, ddof=1)) if len(acc_arr) > 1 else 0.0,
        "ci_lower": float(np.percentile(acc_arr, 2.5)) if len(acc_arr) > 0 else 0.0,
        "ci_upper": float(np.percentile(acc_arr, 97.5)) if len(acc_arr) > 0 else 0.0,
        "mean_effective_n": float(np.mean(n_arr)) if len(n_arr) > 0 else 0.0,
        "n_consensus_ties": int(sum(1 for p in consensus_preds if p is None)),
    }


def _safe_div(num: float, den: float) -> float:
    return float(num / den) if den > 0 else 0.0


def compute_binary_metrics(
    y_true: List[str],
    y_pred: List[str],
    positive_labels: set[str],
) -> Dict[str, float]:
    """Binary metrics for arbitrary tier slice."""
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch in binary metrics.")

    if not y_true:
        return {
            "n": 0,
            "tp": 0,
            "fp": 0,
            "tn": 0,
            "fn": 0,
            "precision": 0.0,
            "recall": 0.0,
            "specificity": 0.0,
            "npv": 0.0,
            "accuracy": 0.0,
            "positive_call_rate": 0.0,
            "negative_call_rate": 0.0,
        }

    tp = fp = tn = fn = 0
    for t, p in zip(y_true, y_pred):
        t_pos = t in positive_labels
        p_pos = p in positive_labels
        if t_pos and p_pos:
            tp += 1
        elif (not t_pos) and p_pos:
            fp += 1
        elif (not t_pos) and (not p_pos):
            tn += 1
        else:
            fn += 1

    n = len(y_true)
    return {
        "n": n,
        "tp": tp,
        "fp": fp,
        "tn": tn,
        "fn": fn,
        "precision": _safe_div(tp, tp + fp),
        "recall": _safe_div(tp, tp + fn),
        "specificity": _safe_div(tn, tn + fp),
        "npv": _safe_div(tn, tn + fn),
        "accuracy": _safe_div(tp + tn, n),
        "positive_call_rate": _safe_div(tp + fp, n),
        "negative_call_rate": _safe_div(tn + fn, n),
    }


def compute_signed_error_profile(y_true: List[str], y_pred: List[str]) -> Dict[str, float]:
    """Compute pessimism/optimism profile via signed ordinal errors."""
    if len(y_true) != len(y_pred):
        raise ValueError("Length mismatch in signed error profile.")

    if not y_true:
        return {
            "mean_signed_error": 0.0,
            "mean_abs_error": 0.0,
            "pessimism_rate": 0.0,
            "optimism_rate": 0.0,
            "exact_rate": 0.0,
        }

    errors = []
    for t, p in zip(y_true, y_pred):
        if t in LABEL_TO_SCORE and p in LABEL_TO_SCORE:
            errors.append(LABEL_TO_SCORE[p] - LABEL_TO_SCORE[t])

    if not errors:
        return {
            "mean_signed_error": 0.0,
            "mean_abs_error": 0.0,
            "pessimism_rate": 0.0,
            "optimism_rate": 0.0,
            "exact_rate": 0.0,
        }

    arr = np.array(errors, dtype=float)
    return {
        "mean_signed_error": float(arr.mean()),
        "mean_abs_error": float(np.abs(arr).mean()),
        "pessimism_rate": float(np.mean(arr < 0)),
        "optimism_rate": float(np.mean(arr > 0)),
        "exact_rate": float(np.mean(arr == 0)),
    }


def build_evaluator_metrics(
    name: str,
    group: str,
    y_true: List[str],
    y_pred: List[str],
    n_ties: int = 0,
    extras: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Compute gatekeeper metrics for one evaluator."""
    per_class = compute_per_class_metrics(y_true, y_pred)
    pred_dist = compute_prediction_distribution(y_true, y_pred)
    signed = compute_signed_error_profile(y_true, y_pred)
    top_bottom = compute_binary_metrics(y_true, y_pred, {"exceptional", "strong"})
    extreme_middle = compute_binary_metrics(y_true, y_pred, {"exceptional", "limited"})

    return {
        "name": name,
        "group": group,
        "n_articles": len(y_true),
        "n_ties_excluded": int(n_ties),
        "overall_accuracy": float(compute_accuracy(y_true, y_pred)),
        "macro_f1": float(compute_macro_f1(y_true, y_pred)),
        "per_class_metrics": per_class,
        "prediction_distribution": pred_dist,
        "tier_focus": {
            "tier1_exceptional_precision": float(per_class["exceptional"]["precision"]),
            "tier1_exceptional_recall": float(per_class["exceptional"]["recall"]),
            "tier4_limited_precision": float(per_class["limited"]["precision"]),
            "tier4_limited_recall": float(per_class["limited"]["recall"]),
            "middle_precision_avg": float(np.mean([per_class["strong"]["precision"], per_class["fair"]["precision"]])),
            "middle_recall_avg": float(np.mean([per_class["strong"]["recall"], per_class["fair"]["recall"]])),
        },
        "top12_vs_bottom34": {
            "n": top_bottom["n"],
            "top12_precision_ppv": top_bottom["precision"],
            "top12_recall_sensitivity": top_bottom["recall"],
            "bottom34_precision_npv": top_bottom["npv"],
            "bottom34_recall_specificity": top_bottom["specificity"],
            "top12_call_rate": top_bottom["positive_call_rate"],
            "bottom34_call_rate": top_bottom["negative_call_rate"],
            "binary_accuracy": top_bottom["accuracy"],
        },
        "extreme14_vs_middle23": {
            "n": extreme_middle["n"],
            "extreme14_precision_ppv": extreme_middle["precision"],
            "extreme14_recall_sensitivity": extreme_middle["recall"],
            "middle23_precision_npv": extreme_middle["npv"],
            "middle23_recall_specificity": extreme_middle["specificity"],
            "extreme14_call_rate": extreme_middle["positive_call_rate"],
            "middle23_call_rate": extreme_middle["negative_call_rate"],
            "binary_accuracy": extreme_middle["accuracy"],
        },
        "signed_error_profile": signed,
        "extras": extras or {},
    }


def make_main_summary_table(
    all_metrics: Dict[str, Dict[str, Any]],
    order: List[str],
) -> pd.DataFrame:
    """Create main summary table."""
    rows = []
    for name in order:
        m = all_metrics[name]
        row = {
            "evaluator": name,
            "group": m["group"],
            "n_articles": m["n_articles"],
            "n_ties_excluded": m["n_ties_excluded"],
            "overall_accuracy": m["overall_accuracy"],
            "macro_f1": m["macro_f1"],
            "middle_tier_bias": m["prediction_distribution"]["middle_tier_bias"],
            "tier1_precision": m["tier_focus"]["tier1_exceptional_precision"],
            "tier1_recall": m["tier_focus"]["tier1_exceptional_recall"],
            "tier4_precision": m["tier_focus"]["tier4_limited_precision"],
            "tier4_recall": m["tier_focus"]["tier4_limited_recall"],
            "top12_precision": m["top12_vs_bottom34"]["top12_precision_ppv"],
            "top12_recall": m["top12_vs_bottom34"]["top12_recall_sensitivity"],
            "bottom34_precision": m["top12_vs_bottom34"]["bottom34_precision_npv"],
            "bottom34_recall": m["top12_vs_bottom34"]["bottom34_recall_specificity"],
            "extreme14_precision": m["extreme14_vs_middle23"]["extreme14_precision_ppv"],
            "extreme14_recall": m["extreme14_vs_middle23"]["extreme14_recall_sensitivity"],
            "middle23_precision": m["extreme14_vs_middle23"]["middle23_precision_npv"],
            "middle23_recall": m["extreme14_vs_middle23"]["middle23_recall_specificity"],
            "mean_signed_error": m["signed_error_profile"]["mean_signed_error"],
            "pessimism_rate": m["signed_error_profile"]["pessimism_rate"],
            "optimism_rate": m["signed_error_profile"]["optimism_rate"],
            "individual_mean_accuracy": m["extras"].get("individual_mean_accuracy"),
            "individual_std_accuracy": m["extras"].get("individual_std_accuracy"),
            "matched_n_mean_accuracy": m["extras"].get("matched_n_mean_accuracy"),
            "matched_n_ci_lower": m["extras"].get("matched_n_ci_lower"),
            "matched_n_ci_upper": m["extras"].get("matched_n_ci_upper"),
            "matched_n_mean_effective_n": m["extras"].get("matched_n_mean_effective_n"),
        }
        rows.append(row)

    return pd.DataFrame(rows)


def make_per_class_table(
    all_metrics: Dict[str, Dict[str, Any]],
    order: List[str],
) -> pd.DataFrame:
    """Create long-form per-class metrics table."""
    rows = []
    for name in order:
        per_class = all_metrics[name]["per_class_metrics"]
        for label in LABEL_ORDER:
            rows.append({
                "evaluator": name,
                "group": all_metrics[name]["group"],
                "label": label,
                "precision": per_class[label]["precision"],
                "recall": per_class[label]["recall"],
                "f1": per_class[label]["f1"],
                "support": per_class[label]["support"],
            })
    return pd.DataFrame(rows)


def make_human_variants_table(all_metrics: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Compact table focused on requested human variants."""
    rows = []
    for name in HUMAN_VARIANTS:
        if name not in all_metrics:
            continue
        m = all_metrics[name]
        rows.append({
            "evaluator": name,
            "n_articles": m["n_articles"],
            "n_ties_excluded": m["n_ties_excluded"],
            "overall_accuracy": m["overall_accuracy"],
            "individual_mean_accuracy": m["extras"].get("individual_mean_accuracy"),
            "individual_std_accuracy": m["extras"].get("individual_std_accuracy"),
            "matched_n_mean_accuracy": m["extras"].get("matched_n_mean_accuracy"),
            "matched_n_ci_lower": m["extras"].get("matched_n_ci_lower"),
            "matched_n_ci_upper": m["extras"].get("matched_n_ci_upper"),
            "matched_n_mean_effective_n": m["extras"].get("matched_n_mean_effective_n"),
            "tier1_precision": m["tier_focus"]["tier1_exceptional_precision"],
            "tier1_recall": m["tier_focus"]["tier1_exceptional_recall"],
            "tier4_precision": m["tier_focus"]["tier4_limited_precision"],
            "tier4_recall": m["tier_focus"]["tier4_limited_recall"],
            "top12_precision": m["top12_vs_bottom34"]["top12_precision_ppv"],
            "bottom34_precision": m["top12_vs_bottom34"]["bottom34_precision_npv"],
            "middle_tier_bias": m["prediction_distribution"]["middle_tier_bias"],
            "mean_signed_error": m["signed_error_profile"]["mean_signed_error"],
        })
    return pd.DataFrame(rows)


def plot_main_figure(
    all_metrics: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot key gatekeeper profile figure for main manuscript set."""
    set_nature_style()
    rows = [r for r in KEY_FIG_ORDER if r in all_metrics]
    metric_labels = [
        "P(T1|pred T1)",
        "R(T1)",
        "P(T4|pred T4)",
        "R(T4)",
        "P(Top1-2|pred Top1-2)",
        "P(Bot3-4|pred Bot3-4)",
        "P(Ext14|pred Ext14)",
        "P(Mid23|pred Mid23)",
    ]

    heat_data = []
    for name in rows:
        m = all_metrics[name]
        heat_data.append([
            m["tier_focus"]["tier1_exceptional_precision"],
            m["tier_focus"]["tier1_exceptional_recall"],
            m["tier_focus"]["tier4_limited_precision"],
            m["tier_focus"]["tier4_limited_recall"],
            m["top12_vs_bottom34"]["top12_precision_ppv"],
            m["top12_vs_bottom34"]["bottom34_precision_npv"],
            m["extreme14_vs_middle23"]["extreme14_precision_ppv"],
            m["extreme14_vs_middle23"]["middle23_precision_npv"],
        ])

    heat = np.array(heat_data, dtype=float)
    fig, axes = plt.subplots(1, 3, figsize=(14, 4.8), gridspec_kw={"width_ratios": [2.8, 1.0, 1.0]})

    ax0 = axes[0]
    im = ax0.imshow(heat, cmap="YlGnBu", vmin=0.0, vmax=1.0, aspect="auto")
    ax0.set_xticks(range(len(metric_labels)))
    ax0.set_xticklabels(metric_labels, rotation=35, ha="right")
    ax0.set_yticks(range(len(rows)))
    ax0.set_yticklabels(rows)
    ax0.set_title("A. Tier-Specific And Slice Precision/Recall", fontsize=9, fontweight="bold")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            ax0.text(j, i, f"{heat[i, j]:.2f}", ha="center", va="center", fontsize=6)
    fig.colorbar(im, ax=ax0, fraction=0.046, pad=0.04)

    ax1 = axes[1]
    middle_bias = [all_metrics[name]["prediction_distribution"]["middle_tier_bias"] for name in rows]
    y_pos = np.arange(len(rows))
    colors = [GROUP_COLORS.get(all_metrics[name]["group"], "#666666") for name in rows]
    ax1.barh(y_pos, middle_bias, color=colors, alpha=0.9)
    ax1.axvline(1.0, color="gray", linestyle="--", linewidth=0.8)
    ax1.set_yticks(y_pos)
    ax1.set_yticklabels([])
    ax1.invert_yaxis()
    ax1.set_xlabel("Pred/True Middle Ratio")
    ax1.set_title("B. Middle-Tier Bias", fontsize=9, fontweight="bold")

    ax2 = axes[2]
    signed = [all_metrics[name]["signed_error_profile"]["mean_signed_error"] for name in rows]
    ax2.barh(y_pos, signed, color=colors, alpha=0.9)
    ax2.axvline(0.0, color="gray", linestyle="--", linewidth=0.8)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels([])
    ax2.invert_yaxis()
    ax2.set_xlabel("Mean Signed Error")
    ax2.set_title("C. Pessimism vs Optimism", fontsize=9, fontweight="bold")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_binary_slices(
    all_metrics: Dict[str, Dict[str, Any]],
    output_path: Path,
) -> None:
    """Plot binary-slice precision comparison for key evaluators."""
    set_nature_style()
    rows = [r for r in KEY_FIG_ORDER if r in all_metrics]
    x = np.arange(len(rows))
    width = 0.18

    top_ppv = [all_metrics[n]["top12_vs_bottom34"]["top12_precision_ppv"] for n in rows]
    bot_ppv = [all_metrics[n]["top12_vs_bottom34"]["bottom34_precision_npv"] for n in rows]
    ext_ppv = [all_metrics[n]["extreme14_vs_middle23"]["extreme14_precision_ppv"] for n in rows]
    mid_ppv = [all_metrics[n]["extreme14_vs_middle23"]["middle23_precision_npv"] for n in rows]

    fig, ax = plt.subplots(figsize=(11.5, 4.5))
    ax.bar(x - 1.5 * width, top_ppv, width, label="P(Top1-2 | pred Top1-2)", color="#E64B35")
    ax.bar(x - 0.5 * width, bot_ppv, width, label="P(Bottom3-4 | pred Bottom3-4)", color="#4DBBD5")
    ax.bar(x + 0.5 * width, ext_ppv, width, label="P(Extreme1/4 | pred Extreme1/4)", color="#00A087")
    ax.bar(x + 1.5 * width, mid_ppv, width, label="P(Middle2/3 | pred Middle2/3)", color="#3C5488")

    ax.set_xticks(x)
    ax.set_xticklabels(rows, rotation=24, ha="right")
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Precision / Decision Reliability")
    ax.set_title("Gatekeeper Slice Precision Comparisons", fontsize=9, fontweight="bold")
    ax.legend(loc="upper right", fontsize=6)
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def plot_all_evaluators_accuracy(
    df_main: pd.DataFrame,
    output_path: Path,
) -> None:
    """Plot accuracy across all evaluators (all flagship + SFT + human variants)."""
    set_nature_style()
    df_plot = df_main.sort_values("overall_accuracy", ascending=True).copy()
    colors = [GROUP_COLORS.get(g, "#666666") for g in df_plot["group"]]

    fig, ax = plt.subplots(figsize=(11.5, 7.2))
    y = np.arange(len(df_plot))
    ax.barh(y, df_plot["overall_accuracy"], color=colors, alpha=0.92)
    ax.set_yticks(y)
    ax.set_yticklabels(df_plot["evaluator"])
    ax.set_xlim(0, 1.0)
    ax.set_xlabel("Overall Accuracy")
    ax.set_title("All Evaluators: Gatekeeper Accuracy", fontsize=10, fontweight="bold")
    ax.axvline(0.25, color="gray", linestyle="--", linewidth=0.8, alpha=0.8)
    ax.text(0.255, len(df_plot) - 1.2, "Chance (25%)", fontsize=6.5, color="gray")

    for yi, acc in zip(y, df_plot["overall_accuracy"]):
        ax.text(acc + 0.01, yi, f"{acc:.3f}", va="center", ha="left", fontsize=6)

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    project_root = get_project_root()
    fig_dir = project_root / FIGURES_DIR
    tab_dir = project_root / TABLES_DIR
    stats_dir = project_root / STATS_DIR
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    print("[23] Loading data...")
    thinking_data = load_thinking_data()
    chat_data = load_chat_data()
    trained_data = load_trained_data()
    expert_data = load_expert_ratings()
    student_data = load_student_filtered_ratings()

    all_metrics: Dict[str, Dict[str, Any]] = {}
    evaluator_order: List[str] = []

    print("[23] Extracting flagship model predictions...")
    for key in FLAGSHIP_KEYS:
        display = FLAGSHIP_DISPLAY[key]
        y_true, y_pred = extract_model_predictions(thinking_data, key, source="thinking")
        if not y_true:
            continue
        all_metrics[display] = build_evaluator_metrics(display, "flagship", y_true, y_pred)
        evaluator_order.append(display)

    print("[23] Extracting chat baseline...")
    y_true_chat, y_pred_chat = extract_model_predictions(chat_data, CHAT_BASELINE_KEY, source="logp")
    if y_true_chat:
        all_metrics[CHAT_BASELINE_DISPLAY] = build_evaluator_metrics(
            CHAT_BASELINE_DISPLAY, "chat", y_true_chat, y_pred_chat
        )
        evaluator_order.append(CHAT_BASELINE_DISPLAY)

    print("[23] Extracting SFT model predictions...")
    for key in SFT_KEYS:
        display = SFT_DISPLAY[key]
        y_true, y_pred = extract_model_predictions(trained_data, key, source="logp")
        if not y_true:
            continue
        all_metrics[display] = build_evaluator_metrics(display, "sft", y_true, y_pred)
        evaluator_order.append(display)

    print("[23] Computing human majority and individual variants...")
    expert_records = build_human_vote_records(expert_data)
    student_records = build_human_vote_records(student_data)

    gt_exp_full, pred_exp_full, expert_counts, expert_ties = compute_human_majority_from_records(expert_records)
    gt_exp, pred_exp = filter_valid_pairs(gt_exp_full, pred_exp_full)
    all_metrics["Expert Majority (Unfiltered)"] = build_evaluator_metrics(
        "Expert Majority (Unfiltered)", "human", gt_exp, pred_exp, n_ties=expert_ties
    )
    evaluator_order.append("Expert Majority (Unfiltered)")

    gt_stu_full, pred_stu_full, _student_counts, student_ties = compute_human_majority_from_records(student_records)
    gt_stu, pred_stu = filter_valid_pairs(gt_stu_full, pred_stu_full)
    all_metrics["Student Majority (Filtered)"] = build_evaluator_metrics(
        "Student Majority (Filtered)", "human", gt_stu, pred_stu, n_ties=student_ties
    )
    evaluator_order.append("Student Majority (Filtered)")

    exp_indiv = compute_human_individual_pooled_and_summary(expert_data)
    all_metrics["Expert Individual (Pooled)"] = build_evaluator_metrics(
        "Expert Individual (Pooled)",
        "human",
        exp_indiv["y_true"],
        exp_indiv["y_pred"],
        extras={
            "individual_mean_accuracy": exp_indiv["mean_accuracy"],
            "individual_std_accuracy": exp_indiv["std_accuracy"],
            "n_raters": exp_indiv["n_raters"],
            "pooled_accuracy": exp_indiv["pooled_accuracy"],
        },
    )
    evaluator_order.append("Expert Individual (Pooled)")

    stu_indiv = compute_human_individual_pooled_and_summary(student_data)
    all_metrics["Student Individual (Pooled)"] = build_evaluator_metrics(
        "Student Individual (Pooled)",
        "human",
        stu_indiv["y_true"],
        stu_indiv["y_pred"],
        extras={
            "individual_mean_accuracy": stu_indiv["mean_accuracy"],
            "individual_std_accuracy": stu_indiv["std_accuracy"],
            "n_raters": stu_indiv["n_raters"],
            "pooled_accuracy": stu_indiv["pooled_accuracy"],
        },
    )
    evaluator_order.append("Student Individual (Pooled)")

    print(f"[23] Running student matched-N Monte Carlo ({MATCHED_N_TRIALS} trials)...")
    matched = compute_student_matched_n_consensus(
        student_records,
        expert_counts=expert_counts,
        n_trials=MATCHED_N_TRIALS,
        seed=RANDOM_SEED,
    )
    gt_match, pred_match = filter_valid_pairs(matched["ground_truths"], matched["consensus_preds"])
    all_metrics["Student Matched-N Consensus (Expert Sized)"] = build_evaluator_metrics(
        "Student Matched-N Consensus (Expert Sized)",
        "human",
        gt_match,
        pred_match,
        n_ties=matched["n_consensus_ties"],
        extras={
            "matched_n_mean_accuracy": matched["mean_accuracy"],
            "matched_n_ci_lower": matched["ci_lower"],
            "matched_n_ci_upper": matched["ci_upper"],
            "matched_n_mean_effective_n": matched["mean_effective_n"],
            "matched_n_trials": MATCHED_N_TRIALS,
        },
    )
    evaluator_order.append("Student Matched-N Consensus (Expert Sized)")

    print("[23] Saving tables...")
    df_main = make_main_summary_table(all_metrics, evaluator_order)
    df_per_class = make_per_class_table(all_metrics, evaluator_order)
    df_human = make_human_variants_table(all_metrics)
    df_trials = pd.DataFrame({
        "trial": np.arange(1, MATCHED_N_TRIALS + 1, dtype=int),
        "accuracy": matched["trial_accuracies"],
        "effective_n": matched["trial_effective_ns"],
    })

    main_table_path = tab_dir / "table23_gatekeeper_profile_main.csv"
    per_class_table_path = tab_dir / "table23_gatekeeper_per_class.csv"
    human_table_path = tab_dir / "table23_human_variants_summary.csv"
    trials_table_path = tab_dir / "table23_student_matched_n_trials.csv"

    df_main.to_csv(main_table_path, index=False)
    df_per_class.to_csv(per_class_table_path, index=False)
    df_human.to_csv(human_table_path, index=False)
    df_trials.to_csv(trials_table_path, index=False)

    print("[23] Saving statistics JSON...")
    stats_path = stats_dir / "23_gatekeeper_profile.json"
    stats_out = {
        "meta": {
            "evaluator_order": evaluator_order,
            "flagship_keys": FLAGSHIP_KEYS,
            "sft_keys": SFT_KEYS,
            "chat_baseline_key": CHAT_BASELINE_KEY,
            "random_seed": RANDOM_SEED,
            "matched_n_trials": MATCHED_N_TRIALS,
            "notes": {
                "human_majority_tie_policy": "exclude",
                "tier_mapping": "1=exceptional, 2=strong, 3=fair, 4=limited",
                "top12_semantics": "tenure-like standards",
            },
        },
        "evaluator_metrics": all_metrics,
        "human_individual_summary": {
            "expert": {
                "n_raters": exp_indiv["n_raters"],
                "mean_accuracy": exp_indiv["mean_accuracy"],
                "std_accuracy": exp_indiv["std_accuracy"],
                "pooled_accuracy": exp_indiv["pooled_accuracy"],
            },
            "student": {
                "n_raters": stu_indiv["n_raters"],
                "mean_accuracy": stu_indiv["mean_accuracy"],
                "std_accuracy": stu_indiv["std_accuracy"],
                "pooled_accuracy": stu_indiv["pooled_accuracy"],
            },
        },
        "student_matched_n_details": {
            "mean_accuracy": matched["mean_accuracy"],
            "std_accuracy": matched["std_accuracy"],
            "ci_lower": matched["ci_lower"],
            "ci_upper": matched["ci_upper"],
            "mean_effective_n": matched["mean_effective_n"],
            "n_consensus_ties": matched["n_consensus_ties"],
            "coverage_by_article": matched["coverage_by_article"],
            "trial_accuracies": matched["trial_accuracies"],
            "trial_effective_ns": matched["trial_effective_ns"],
        },
    }
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump(stats_out, f, indent=2, ensure_ascii=False)

    print("[23] Saving figures...")
    fig_main = fig_dir / "fig23_gatekeeper_profile_main.png"
    fig_supp = fig_dir / "fig_s23_gatekeeper_binary_slices.png"
    fig_all = fig_dir / "fig23_all_evaluators_accuracy.png"
    plot_main_figure(all_metrics, fig_main)
    plot_binary_slices(all_metrics, fig_supp)
    plot_all_evaluators_accuracy(df_main, fig_all)

    print("[23] Done.")
    print(f"  - {main_table_path}")
    print(f"  - {per_class_table_path}")
    print(f"  - {human_table_path}")
    print(f"  - {trials_table_path}")
    print(f"  - {stats_path}")
    print(f"  - {fig_main}")
    print(f"  - {fig_supp}")
    print(f"  - {fig_all}")

    print("\n[23] Human Variant Snapshot")
    for name in HUMAN_VARIANTS:
        if name not in all_metrics:
            continue
        m = all_metrics[name]
        extra = m["extras"]
        print(
            f"  {name:32s} "
            f"Acc={m['overall_accuracy']:.3f} "
            f"T1(P/R)={m['tier_focus']['tier1_exceptional_precision']:.3f}/{m['tier_focus']['tier1_exceptional_recall']:.3f} "
            f"T4(P/R)={m['tier_focus']['tier4_limited_precision']:.3f}/{m['tier_focus']['tier4_limited_recall']:.3f} "
            f"MeanPerRater={extra.get('individual_mean_accuracy', float('nan')):.3f} "
            f"MatchedNMean={extra.get('matched_n_mean_accuracy', float('nan')):.3f}"
        )


if __name__ == "__main__":
    main()
