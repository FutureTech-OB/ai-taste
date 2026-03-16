#!/usr/bin/env python3
"""Script 07: AI vs Human Comparison - The Grand Comparison (Hero Analysis)

This is the central analysis for the Nature/Science paper. It compares
AI model performance (flagship models, GPT-5.2 baseline, SFT single model,
SFT ensembles) against human evaluators (expert majority vote, student
majority vote, individual averages) on the 120-article research quality
evaluation task.

All human majority votes use tie_policy='exclude' for consistency:
articles without a clear majority are excluded from accuracy computation.

Outputs:
    - results/figures/fig1_grand_comparison.pdf    -- Hero figure (10 bars)
    - results/figures/fig_s10_agreement_heatmap.pdf -- Cohen's kappa heatmap
    - results/figures/fig_s11_all_flagships.pdf     -- All 11 flagship models
    - results/figures/fig8_complementarity_venn.pdf -- Complementarity analysis
    - results/tables/table8_ai_vs_human.csv        -- Main results table
    - results/statistics/07_ai_vs_human.json       -- All statistical results
"""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from itertools import combinations

from analysis.utils.constants import *
from analysis.utils.data_loader import *
from analysis.utils.frontier_protocol import (
    assert_frontier_models_present,
    best_model_by_primary,
    extract_discrete_prediction,
    full_predictions,
    model_primary_accuracy,
)
from analysis.utils.metrics import *
from analysis.utils.voting import *
from analysis.utils.statistical_tests import *
from analysis.utils.visualization import *


# ===================================================================
# Configuration
# ===================================================================

# Keep defaults aligned with contamination-clean reporting.
# Runtime logic still recomputes this from canonical frontier protocol.
BEST_FLAGSHIP_KEY = 'x-ai/grok-4.1-fast'   # overwritten from canonical primary metric at runtime
BEST_FLAGSHIP_DISPLAY = 'Grok 4.1 Fast'    # overwritten at runtime

FLAGSHIP_KEYS = list(FRONTIER_MODELS)

MONTE_CARLO_TRIALS = 5000
BOOTSTRAP_N = 10000
RANDOM_SEED = 42

EVALUATOR_ORDER = []
EVALUATOR_SHORT = []
EVAL_COLORS = {}


def _configure_evaluator_labels(best_flagship_display, best_sft_single_key):
    """Set evaluator labels/colors with dynamic best-flagship/SFT naming."""
    global EVALUATOR_ORDER, EVALUATOR_SHORT, EVAL_COLORS

    flagship_multiline = f'Best Flagship\n({best_flagship_display})'
    flagship_short = f'Best Flagship ({best_flagship_display})'
    best_sft_display = MODEL_DISPLAY_NAMES.get(best_sft_single_key, best_sft_single_key)
    best_sft_multiline = f'Best SFT\nSingle ({best_sft_display})'
    best_sft_short = f'Best SFT Single ({best_sft_single_key})'

    EVALUATOR_ORDER = [
        flagship_multiline,
        'Flagship\nAverage',
        'GPT-5.2\n(pre-SFT)',
        best_sft_multiline,
        'SFT 2-Model\nEnsemble',
        'Expert Indiv.\nAverage',
        'Expert Majority\n(excl. ties)',
        'Student Indiv.\nAverage',
        'Student Matched\nN (excl. ties)',
        'Student Majority\n(full, excl. ties)',
    ]

    EVALUATOR_SHORT = [
        flagship_short,
        'Flagship Average (11 models)',
        'GPT-5.2 (pre-SFT baseline)',
        best_sft_short,
        'SFT 2-Model Ensemble',
        'Expert Individual Average',
        'Expert Majority Vote (excl. ties)',
        'Student Individual Average',
        'Student Matched-N Vote (excl. ties)',
        'Student Majority Vote (full, excl. ties)',
    ]

    # Color scheme: reds/warm for AI, blues/cool for human
    EVAL_COLORS = {
        flagship_multiline: '#B71C1C',
        'Flagship\nAverage': '#C62828',
        'GPT-5.2\n(pre-SFT)': '#E53935',
        best_sft_multiline: '#FF7043',
        'SFT 2-Model\nEnsemble': '#FF8A65',
        'Expert Indiv.\nAverage': '#0D47A1',
        'Expert Majority\n(excl. ties)': '#1565C0',
        'Student Indiv.\nAverage': '#1E88E5',
        'Student Matched\nN (excl. ties)': '#42A5F5',
        'Student Majority\n(full, excl. ties)': '#90CAF9',
    }


# ===================================================================
# Prediction Extraction Helpers
# ===================================================================

def _extract_val_models(article):
    """Extract model prediction data from a val-data article.

    val_outcome.rq_with_context is a dict whose keys are model names
    (e.g. 'gpt-5.2', 'ckpt-step-304', 'best_2_model_combo', etc.).
    """
    return article.get('val_outcome', {}).get('rq_with_context', {})


def _pred_from_logp(logp_dict):
    """Get predicted label from a logp dictionary."""
    prob = logp_to_prob(logp_dict)
    top = get_top_predictions(prob, top_k=1)
    if top:
        return top[0][0]
    return None


def _pred_from_thinking_model(model_data, policy='majority', tie_policy='exclude'):
    """Get protocol-consistent discrete label from frontier-model entry."""
    pred, _ = extract_discrete_prediction(
        model_data,
        policy=policy,
        tie_policy=tie_policy,
    )
    return pred


def extract_flagship_predictions(frontier_data, model_key, policy='majority'):
    """Extract predictions from a flagship model in canonical frontier data."""
    y_true = []
    y_pred = []
    for article in frontier_data:
        gt = get_ground_truth(article)
        rq = article.get('val_outcome', {}).get('rq_with_context', {})
        model_data = rq.get(model_key, {})
        pred = _pred_from_thinking_model(model_data, policy=policy, tie_policy='exclude')
        if pred is not None:
            y_true.append(gt)
            y_pred.append(pred)
    return y_true, y_pred


def extract_val_model_predictions(val_data, model_key):
    """Extract predictions from a model in the val data (uses logp)."""
    y_true = []
    y_pred = []
    for article in val_data:
        gt = get_ground_truth(article)
        models = _extract_val_models(article)
        model_data = models.get(model_key, {})
        logp = model_data.get('logp')
        if logp is not None:
            pred = _pred_from_logp(logp)
            if pred is not None:
                y_true.append(gt)
                y_pred.append(pred)
    return y_true, y_pred


def compute_best_sft_single_key(val_data):
    """Return the best-performing canonical SFT checkpoint by accuracy."""
    best_key = None
    best_score = None
    for model_key in SFT_MODELS:
        y_true, y_pred = extract_val_model_predictions(val_data, model_key)
        if not y_pred:
            continue
        score = (
            compute_accuracy(y_true, y_pred),
            compute_macro_f1(y_true, y_pred),
            model_key,
        )
        if best_score is None or score > best_score:
            best_key = model_key
            best_score = score
    if best_key is None:
        raise ValueError('No valid SFT single-model predictions found in val data.')
    return best_key


def extract_human_voting_predictions(val_data):
    """Extract human_voting predictions from val data."""
    y_true = []
    y_pred = []
    for article in val_data:
        gt = get_ground_truth(article)
        models = _extract_val_models(article)
        hv = models.get('human_voting', {})
        logp = hv.get('logp')
        if logp is not None:
            pred = _pred_from_logp(logp)
            if pred is not None:
                y_true.append(gt)
                y_pred.append(pred)
    return y_true, y_pred


# ===================================================================
# Human Majority Vote Computation
# ===================================================================

def compute_expert_majority_votes(expert_data):
    """Compute expert majority vote predictions from the public expert file.

    Uses tie_policy='exclude': articles without a clear majority get
    pred=None and are excluded from accuracy computation.

    Returns (y_true, y_pred, expert_counts_per_article, n_ties).
    y_true and y_pred always have len(expert_data) entries.
    pred is None for tied articles.
    expert_counts_per_article has one entry per article (for matched-N use).
    """
    y_true = []
    y_pred = []
    expert_counts = []
    n_ties = 0

    for article in expert_data:
        gt = normalize_level(article.get('level', ''))
        ratings_list = article.get('ratings', [])
        votes = []
        for r in ratings_list:
            raw = r.get('q2_rating', '')
            if raw:
                votes.append(normalize_level(raw))
        if not votes:
            expert_counts.append(0)
            y_true.append(gt)
            y_pred.append(None)
            continue
        expert_counts.append(len(votes))
        pred, is_tie = majority_vote(votes, tie_policy='exclude')
        if is_tie:
            n_ties += 1
        y_true.append(gt)
        y_pred.append(pred)

    return y_true, y_pred, expert_counts, n_ties


def compute_student_majority_votes(student_data):
    """Compute student majority vote predictions from the filtered combined junior panel.

    Uses tie_policy='exclude' for consistency with expert analysis.

    Returns (y_true, y_pred, student_counts_per_article, n_ties).
    y_true and y_pred always have len(student_data) entries.
    pred is None for tied articles.
    """
    y_true = []
    y_pred = []
    student_counts = []
    n_ties = 0

    for article in student_data:
        gt = normalize_level(article.get('level', ''))
        ratings_list = article.get('ratings', [])
        votes = []
        for r in ratings_list:
            raw = r.get('q2_rating', '')
            if raw:
                votes.append(normalize_level(raw))
        if not votes:
            student_counts.append(0)
            y_true.append(gt)
            y_pred.append(None)
            continue
        student_counts.append(len(votes))
        pred, is_tie = majority_vote(votes, tie_policy='exclude')
        if is_tie:
            n_ties += 1
        y_true.append(gt)
        y_pred.append(pred)

    return y_true, y_pred, student_counts, n_ties


def compute_student_matched_n(student_data, expert_counts, n_trials=5000, seed=42):
    """Monte Carlo: sample students to match expert count per article.

    For each trial, for each article, sample N_expert students from the
    available student pool, compute majority vote (tie_policy='exclude'),
    then measure accuracy across articles with clear majority.

    Returns dict with mean_accuracy, ci, effective N stats, representative preds.
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    article_data = []
    for i, article in enumerate(student_data):
        gt = normalize_level(article.get('level', ''))
        ratings_list = article.get('ratings', [])
        votes = []
        for r in ratings_list:
            raw = r.get('q2_rating', '')
            if raw:
                votes.append(normalize_level(raw))
        n_experts = expert_counts[i] if i < len(expert_counts) else 3
        article_data.append({
            'gt': gt,
            'votes': votes,
            'n_sample': min(n_experts, len(votes)),
        })

    trial_accuracies = []
    trial_effective_ns = []
    representative_preds = None

    for trial in range(n_trials):
        correct = 0
        total = 0
        trial_preds = []
        for ad in article_data:
            if not ad['votes'] or ad['n_sample'] == 0:
                trial_preds.append(None)
                continue
            indices = np_rng.choice(
                len(ad['votes']), size=ad['n_sample'], replace=False
            )
            sampled = [ad['votes'][idx] for idx in indices]
            pred, _ = majority_vote(sampled, tie_policy='exclude')
            trial_preds.append(pred)
            if pred is not None:
                total += 1
                if pred == ad['gt']:
                    correct += 1
        acc = correct / total if total > 0 else 0.0
        trial_accuracies.append(acc)
        trial_effective_ns.append(total)
        if representative_preds is None:
            representative_preds = trial_preds

    acc_arr = np.array(trial_accuracies)
    eff_n_arr = np.array(trial_effective_ns)
    return {
        'mean_accuracy': float(np.mean(acc_arr)),
        'std': float(np.std(acc_arr, ddof=1)),
        'ci_lower': float(np.percentile(acc_arr, 2.5)),
        'ci_upper': float(np.percentile(acc_arr, 97.5)),
        'n_trials': n_trials,
        'mean_effective_n': float(np.mean(eff_n_arr)),
        'representative_preds': representative_preds,
        'trial_accuracies': trial_accuracies,
        'ground_truths': [ad['gt'] for ad in article_data],
    }


# ===================================================================
# Flagship Average
# ===================================================================

def _primary_score_from_model_data(gt, model_data):
    """Primary metric score for one article/model under canonical protocol."""
    mode = model_data.get('mode', 'deterministic_logp')
    if mode == 'stochastic_thinking':
        avg_acc = model_data.get('avg_accuracy')
        try:
            if avg_acc is not None:
                return float(avg_acc)
        except (TypeError, ValueError):
            pass

    pred, _ = extract_discrete_prediction(
        model_data,
        policy='prediction',
        tie_policy='run1_fallback',
    )
    if pred is None:
        return None
    return 1.0 if pred == gt else 0.0


def compute_flagship_average(frontier_data):
    """Compute per-article frontier-average primary accuracy.

    Returns:
        per_article_correctness: list of floats (mean primary score
            per article, for bootstrapping CI)
        per_model_accuracies: dict {model_key: primary_accuracy}
        avg_accuracy: float (frontier average under primary metric)
    """
    per_model_accuracies = {}
    for key in FLAGSHIP_KEYS:
        per_model_accuracies[key] = model_primary_accuracy(frontier_data, key)

    per_article_correctness = []
    for article in frontier_data:
        gt = get_ground_truth(article)
        rq = article.get('val_outcome', {}).get('rq_with_context', {})
        scores = []
        for key in FLAGSHIP_KEYS:
            model_data = rq.get(key, {})
            score = _primary_score_from_model_data(gt, model_data)
            if score is not None:
                scores.append(score)
        frac = float(np.mean(scores)) if scores else 0.0
        per_article_correctness.append(frac)

    avg_accuracy = float(np.mean(per_article_correctness))
    return per_article_correctness, per_model_accuracies, avg_accuracy


def compute_flagship_average_macro_f1(frontier_data, n_bootstrap=10000, seed=42):
    """Compute frontier-average macro-F1 and bootstrap CI.

    Macro-F1 is computed per flagship model from discrete majority predictions
    (ties excluded), then averaged across the flagship model set.
    Bootstrap replicates resample articles within each model.
    """
    per_model_predictions = {}
    per_model_macro_f1 = {}
    for key in FLAGSHIP_KEYS:
        y_true, y_pred = extract_flagship_predictions(frontier_data, key, policy='majority')
        if y_true:
            per_model_predictions[key] = (np.array(y_true, dtype=object), np.array(y_pred, dtype=object))
            per_model_macro_f1[key] = float(compute_macro_f1(y_true, y_pred))

    if not per_model_predictions:
        return {
            'estimate': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'std': 0.0,
            'per_model_macro_f1': {},
            'n_models': 0,
        }

    estimate = float(np.mean(list(per_model_macro_f1.values())))
    rng = np.random.RandomState(seed)
    boot = np.zeros(n_bootstrap, dtype=float)
    model_keys = list(per_model_predictions.keys())

    for b in range(n_bootstrap):
        model_vals = []
        for key in model_keys:
            yt_arr, yp_arr = per_model_predictions[key]
            n = len(yt_arr)
            idx = rng.choice(n, size=n, replace=True)
            yt = yt_arr[idx].tolist()
            yp = yp_arr[idx].tolist()
            model_vals.append(compute_macro_f1(yt, yp))
        boot[b] = float(np.mean(model_vals))

    return {
        'estimate': estimate,
        'ci_lower': float(np.percentile(boot, 2.5)),
        'ci_upper': float(np.percentile(boot, 97.5)),
        'std': float(np.std(boot, ddof=1)),
        'per_model_macro_f1': {k: float(v) for k, v in per_model_macro_f1.items()},
        'n_models': len(model_keys),
    }


# ===================================================================
# Individual Rater Accuracy
# ===================================================================

def compute_individual_accuracy(data, rating_field='q2_rating'):
    """Compute per-rater accuracy and return aggregate stats.

    Returns dict with: n_raters, mean_accuracy, std, accuracies (list).
    """
    rater_results = defaultdict(lambda: {'correct': 0, 'total': 0})

    for article in data:
        gt = normalize_level(article.get('level', ''))
        for r in article.get('ratings', []):
            raw = r.get(rating_field, '')
            if not raw:
                continue
            pred = normalize_level(raw)
            name = r.get('rater_id', r.get('rater_name', 'unknown'))
            rater_results[name]['total'] += 1
            if pred == gt:
                rater_results[name]['correct'] += 1

    accuracies = []
    for name, data_r in rater_results.items():
        if data_r['total'] > 0:
            accuracies.append(data_r['correct'] / data_r['total'])

    acc_arr = np.array(accuracies) if accuracies else np.array([0.0])
    return {
        'n_raters': len(accuracies),
        'mean_accuracy': float(np.mean(acc_arr)),
        'std': float(np.std(acc_arr, ddof=1)) if len(acc_arr) > 1 else 0.0,
        'accuracies': accuracies,
    }


# ===================================================================
# Bootstrap CI for Accuracy
# ===================================================================

def bootstrap_accuracy_ci(y_true, y_pred, n_bootstrap=10000, seed=42):
    """Bootstrap 95% CI for accuracy."""
    correct = np.array([1.0 if t == p else 0.0 for t, p in zip(y_true, y_pred)])
    result = bootstrap_ci(
        correct,
        statistic_fn=np.mean,
        n_bootstrap=n_bootstrap,
        ci=0.95,
        seed=seed,
    )
    return result


def bootstrap_macro_f1_ci(y_true, y_pred, n_bootstrap=10000, seed=42):
    """Bootstrap 95% CI for macro-F1 with stratified article resampling."""
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} items, "
            f"y_pred has {len(y_pred)} items."
        )
    if len(y_true) == 0:
        raise ValueError("Cannot bootstrap macro-F1 on empty lists.")

    y_true_arr = np.array(y_true, dtype=object)
    y_pred_arr = np.array(y_pred, dtype=object)
    rng = np.random.RandomState(seed)
    idx_by_label = {
        label: np.where(y_true_arr == label)[0]
        for label in LABEL_ORDER
        if np.any(y_true_arr == label)
    }

    boot = np.zeros(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sampled_parts = []
        for _, idx in idx_by_label.items():
            sampled_parts.append(rng.choice(idx, size=len(idx), replace=True))
        sample_idx = np.concatenate(sampled_parts) if sampled_parts else rng.choice(
            len(y_true_arr), size=len(y_true_arr), replace=True
        )
        rng.shuffle(sample_idx)
        yt = y_true_arr[sample_idx].tolist()
        yp = y_pred_arr[sample_idx].tolist()
        boot[i] = compute_macro_f1(yt, yp)

    return {
        'estimate': float(compute_macro_f1(y_true, y_pred)),
        'ci_lower': float(np.percentile(boot, 2.5)),
        'ci_upper': float(np.percentile(boot, 97.5)),
        'std': float(np.std(boot, ddof=1)),
    }


def extract_individual_pooled_predictions(data, rating_field='q2_rating'):
    """Build pooled individual predictions and article clusters for bootstrap."""
    pooled_true = []
    pooled_pred = []
    article_clusters = []

    for article in data:
        gt = normalize_level(article.get('level', ''))
        preds = []
        for r in article.get('ratings', []):
            raw = r.get(rating_field, '')
            if not raw:
                continue
            pred = normalize_level(raw)
            preds.append(pred)
            pooled_true.append(gt)
            pooled_pred.append(pred)
        if preds:
            article_clusters.append({'gt': gt, 'preds': preds})

    return pooled_true, pooled_pred, article_clusters


def bootstrap_macro_f1_ci_cluster(article_clusters, n_bootstrap=10000, seed=42):
    """Cluster bootstrap macro-F1 CI for pooled individual ratings."""
    if not article_clusters:
        return {'estimate': 0.0, 'ci_lower': 0.0, 'ci_upper': 0.0, 'std': 0.0}

    by_label = defaultdict(list)
    for idx, cluster in enumerate(article_clusters):
        by_label[cluster['gt']].append(idx)

    def _flatten(indices):
        y_true = []
        y_pred = []
        for i in indices:
            gt = article_clusters[i]['gt']
            preds = article_clusters[i]['preds']
            y_true.extend([gt] * len(preds))
            y_pred.extend(preds)
        return y_true, y_pred

    full_indices = list(range(len(article_clusters)))
    y_true_full, y_pred_full = _flatten(full_indices)
    estimate = float(compute_macro_f1(y_true_full, y_pred_full)) if y_true_full else 0.0

    rng = np.random.RandomState(seed)
    boot = np.zeros(n_bootstrap, dtype=float)
    label_order_present = [k for k in LABEL_ORDER if k in by_label]
    if not label_order_present:
        label_order_present = list(by_label.keys())

    for b in range(n_bootstrap):
        sampled = []
        for label in label_order_present:
            idx_list = by_label[label]
            sampled.extend(rng.choice(idx_list, size=len(idx_list), replace=True).tolist())
        rng.shuffle(sampled)
        yt, yp = _flatten(sampled)
        boot[b] = compute_macro_f1(yt, yp) if yt else 0.0

    return {
        'estimate': estimate,
        'ci_lower': float(np.percentile(boot, 2.5)),
        'ci_upper': float(np.percentile(boot, 97.5)),
        'std': float(np.std(boot, ddof=1)),
    }


# ===================================================================
# Per-Tier Accuracy
# ===================================================================

def per_tier_accuracy(y_true, y_pred, tiers=None):
    """Compute accuracy for each tier."""
    if tiers is None:
        tiers = LABEL_ORDER
    results = {}
    for tier in tiers:
        tier_true = [t for t, p in zip(y_true, y_pred) if t == tier]
        tier_pred = [p for t, p in zip(y_true, y_pred) if t == tier]
        if tier_true:
            results[tier] = compute_accuracy(tier_true, tier_pred)
        else:
            results[tier] = 0.0
    return results


# ===================================================================
# Complementarity Analysis
# ===================================================================

def complementarity_analysis(y_true, y_pred_ai, y_pred_human):
    """Analyze complementarity between AI and human predictions.

    Returns counts of: both_correct, ai_only, human_only, neither.
    Also computes a combined ensemble accuracy.
    """
    both_correct = 0
    ai_only = 0
    human_only = 0
    neither = 0
    articles_detail = []

    for i, (gt, ai, human) in enumerate(zip(y_true, y_pred_ai, y_pred_human)):
        ai_ok = (ai == gt)
        human_ok = (human == gt)
        if ai_ok and human_ok:
            both_correct += 1
            articles_detail.append('both')
        elif ai_ok and not human_ok:
            ai_only += 1
            articles_detail.append('ai_only')
        elif not ai_ok and human_ok:
            human_only += 1
            articles_detail.append('human_only')
        else:
            neither += 1
            articles_detail.append('neither')

    n = len(y_true)

    # Combined ensemble: majority vote of AI pred + human pred
    combined_correct = 0
    combined_preds = []
    rng = random.Random(RANDOM_SEED)
    for gt, ai, human in zip(y_true, y_pred_ai, y_pred_human):
        if ai == human:
            combined_pred = ai
        else:
            combined_pred = rng.choice([ai, human])
        combined_preds.append(combined_pred)
        if combined_pred == gt:
            combined_correct += 1

    return {
        'both_correct': both_correct,
        'ai_only_correct': ai_only,
        'human_only_correct': human_only,
        'neither_correct': neither,
        'total': n,
        'both_correct_pct': both_correct / n * 100,
        'ai_only_pct': ai_only / n * 100,
        'human_only_pct': human_only / n * 100,
        'neither_pct': neither / n * 100,
        'shared_error_pct': neither / (neither + ai_only + human_only) * 100
            if (neither + ai_only + human_only) > 0 else 0.0,
        'complementary_error_pct': (ai_only + human_only)
            / (neither + ai_only + human_only) * 100
            if (neither + ai_only + human_only) > 0 else 0.0,
        'combined_accuracy': combined_correct / n,
        'combined_preds': combined_preds,
        'articles_detail': articles_detail,
    }


# ===================================================================
# Figures
# ===================================================================

def plot_hero_figure(evaluator_results, output_dir):
    """Create the hero figure: grouped bar chart of all evaluator accuracies.

    This is Figure 1 for the Nature/Science paper.
    Shows 10 bars: 5 AI models + 5 human evaluators.
    """
    set_nature_style()
    fig, ax = plt.subplots(figsize=(8.5, 4.5))

    names = []
    accuracies = []
    ci_lowers = []
    ci_uppers = []
    colors = []

    for entry in evaluator_results:
        names.append(entry['display_name'])
        acc = entry['accuracy'] * 100
        accuracies.append(acc)
        ci_lo = entry.get('ci_lower', acc)
        ci_hi = entry.get('ci_upper', acc)
        if ci_lo <= 1.0:
            ci_lo *= 100
        if ci_hi <= 1.0:
            ci_hi *= 100
        ci_lowers.append(ci_lo)
        ci_uppers.append(ci_hi)
        colors.append(EVAL_COLORS.get(entry['display_name'], '#888888'))

    x = np.arange(len(names))
    err_lower = [a - lo for a, lo in zip(accuracies, ci_lowers)]
    err_upper = [hi - a for a, hi in zip(accuracies, ci_uppers)]

    bars = ax.bar(
        x, accuracies, width=0.65,
        color=colors, edgecolor='white', linewidth=0.5,
        yerr=[err_lower, err_upper],
        capsize=3, error_kw={'linewidth': 0.8, 'color': '#333333'},
    )

    # Add value labels on bars
    for i, (bar, acc) in enumerate(zip(bars, accuracies)):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.5,
            f'{acc:.1f}%',
            ha='center', va='bottom', fontsize=5.5, fontweight='bold',
        )

    # Chance level
    ax.axhline(y=25, color='#666666', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.text(
        len(names) - 0.5, 26, 'Chance (25%)',
        ha='right', va='bottom', fontsize=6, color='#666666', style='italic',
    )

    # Category separator between AI (5 bars) and Human (5 bars)
    ax.axvline(x=4.5, color='#CCCCCC', linestyle='-', linewidth=0.5, alpha=0.6)

    # Category labels
    ax.text(2.0, -12, 'AI Models', ha='center', va='top', fontsize=8,
            fontweight='bold', color='#B71C1C',
            transform=ax.get_xaxis_transform())
    ax.text(7.0, -12, 'Human Evaluators', ha='center', va='top', fontsize=8,
            fontweight='bold', color='#1565C0',
            transform=ax.get_xaxis_transform())

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=0, ha='center', fontsize=5.5)
    ax.set_ylabel('Accuracy (%)', fontsize=9)
    ax.set_title(
        'AI vs Human Performance on Research Quality Evaluation',
        fontsize=10, fontweight='bold', pad=12,
    )
    ax.set_ylim(0, max(accuracies) + 15)
    ax.set_xlim(-0.6, len(names) - 0.4)

    # Light horizontal gridlines
    ax.yaxis.grid(True, alpha=0.2, linewidth=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    fig.subplots_adjust(bottom=0.24)

    save_figure(fig, 'fig1_grand_comparison', output_dir)
    plt.close(fig)


def plot_agreement_heatmap(kappa_matrix, evaluator_names, output_dir):
    """Create heatmap of Cohen's kappa between all evaluator pairs."""
    set_nature_style()

    n = len(evaluator_names)
    fig, ax = plt.subplots(figsize=(6.5, 5.5))

    # Use short labels
    short_labels = []
    for name in evaluator_names:
        short = name.replace('\n', ' ').strip()
        if len(short) > 20:
            short = short[:18] + '..'
        short_labels.append(short)

    data = np.array(kappa_matrix)

    im = ax.imshow(data, cmap='RdYlBu_r', aspect='auto', vmin=-0.1, vmax=0.8)

    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(short_labels, rotation=45, ha='right', fontsize=6)
    ax.set_yticklabels(short_labels, fontsize=6)

    # Annotate cells
    for i in range(n):
        for j in range(n):
            val = data[i, j]
            color = 'white' if val > 0.5 else 'black'
            text = f'{val:.2f}' if i != j else '1.00'
            ax.text(j, i, text, ha='center', va='center', fontsize=5.5, color=color)

    ax.set_title("Cohen's Kappa Between All Evaluator Pairs",
                 fontsize=9, fontweight='bold')
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Cohen's Kappa")
    fig.tight_layout()

    save_figure(fig, 'fig_s10_agreement_heatmap', output_dir)
    plt.close(fig)


def plot_complementarity(comp_results, output_dir):
    """Create a complementarity visualization (stacked/quadrant bar)."""
    set_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.5))

    # Left panel: stacked bar showing article distribution
    ax = axes[0]
    categories = ['Both\nCorrect', 'AI\nOnly', 'Human\nOnly', 'Neither']
    values = [
        comp_results['both_correct'],
        comp_results['ai_only_correct'],
        comp_results['human_only_correct'],
        comp_results['neither_correct'],
    ]
    pcts = [
        comp_results['both_correct_pct'],
        comp_results['ai_only_pct'],
        comp_results['human_only_pct'],
        comp_results['neither_pct'],
    ]
    bar_colors = ['#4CAF50', '#FF7043', '#42A5F5', '#BDBDBD']

    bars = ax.bar(range(len(categories)), values, color=bar_colors,
                  edgecolor='white', linewidth=0.5, width=0.6)
    for bar, val, pct in zip(bars, values, pcts):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                f'{val}\n({pct:.1f}%)',
                ha='center', va='bottom', fontsize=7)

    ax.set_xticks(range(len(categories)))
    ax.set_xticklabels(categories, fontsize=7)
    ax.set_ylabel('Number of Articles', fontsize=8)
    ax.set_title('SFT 2-Model Ensemble vs Expert Majority',
                 fontsize=8, fontweight='bold')

    # Right panel: error type breakdown
    ax2 = axes[1]
    total_errors = (comp_results['ai_only_correct']
                    + comp_results['human_only_correct']
                    + comp_results['neither_correct'])
    if total_errors > 0:
        shared = comp_results['neither_correct']
        wedge_sizes = [shared, comp_results['ai_only_correct'],
                       comp_results['human_only_correct']]
        wedge_labels = [
            f'Shared errors\n({shared})',
            f'AI correct, Human wrong\n({comp_results["ai_only_correct"]})',
            f'Human correct, AI wrong\n({comp_results["human_only_correct"]})',
        ]
        wedge_colors = ['#BDBDBD', '#FF7043', '#42A5F5']
        wedges, texts, autotexts = ax2.pie(
            wedge_sizes, labels=wedge_labels, colors=wedge_colors,
            autopct='%1.1f%%', startangle=90,
            textprops={'fontsize': 6},
        )
        for at in autotexts:
            at.set_fontsize(6)
    else:
        ax2.text(0.5, 0.5, 'No errors', ha='center', va='center',
                 transform=ax2.transAxes, fontsize=9)

    ax2.set_title('Error Distribution\n(Among Non-Agreement Articles)',
                  fontsize=8, fontweight='bold')

    fig.tight_layout()
    save_figure(fig, 'fig8_complementarity_venn', output_dir)
    plt.close(fig)


def plot_all_flagships(per_model_accuracies, output_dir):
    """Create supplementary bar chart showing all 11 flagship models sorted by accuracy."""
    set_nature_style()
    fig, ax = plt.subplots(figsize=(7.0, 4.0))

    # Sort by accuracy descending
    sorted_models = sorted(per_model_accuracies.items(), key=lambda x: x[1], reverse=True)
    names = []
    accs = []
    for key, acc in sorted_models:
        display = key.split('/')[-1] if '/' in key else key
        names.append(display)
        accs.append(acc * 100)

    x = np.arange(len(names))
    colors = plt.cm.RdYlBu_r(np.linspace(0.2, 0.8, len(names)))

    bars = ax.bar(x, accs, width=0.6, color=colors, edgecolor='white', linewidth=0.5)

    for bar, acc_val in zip(bars, accs):
        ax.text(
            bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8,
            f'{acc_val:.1f}%', ha='center', va='bottom', fontsize=6.5, fontweight='bold',
        )

    ax.axhline(y=25, color='#666666', linestyle='--', linewidth=0.7, alpha=0.7)
    ax.text(len(names) - 0.5, 26, 'Chance (25%)',
            ha='right', va='bottom', fontsize=6, color='#666666', style='italic')

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=45, ha='right', fontsize=6.5)
    ax.set_ylabel('Accuracy (%)', fontsize=9)
    ax.set_title('All 11 Flagship Models on Research Quality Evaluation\n'
                 '(No token prediction probabilities available)',
                 fontsize=9, fontweight='bold', pad=8)
    ax.set_ylim(0, max(accs) + 10)

    ax.yaxis.grid(True, alpha=0.2, linewidth=0.3)
    ax.set_axisbelow(True)

    fig.tight_layout()
    save_figure(fig, 'fig_s11_all_flagships', output_dir)
    plt.close(fig)


# ===================================================================
# Main Analysis
# ===================================================================

def main():
    project_root = get_project_root()
    figures_dir = project_root / FIGURES_DIR
    tables_dir = project_root / TABLES_DIR
    stats_dir = project_root / STATS_DIR

    figures_dir.mkdir(parents=True, exist_ok=True)
    tables_dir.mkdir(parents=True, exist_ok=True)
    stats_dir.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------
    # 1. Load all data
    # -----------------------------------------------------------------
    print('[07] Loading data...')
    val_data = load_val_data()
    frontier_data = load_frontier_data()
    assert_frontier_models_present(frontier_data, FLAGSHIP_KEYS)
    expert_data = load_expert_ratings()
    student_data = load_student_filtered_ratings()

    n_val = len(val_data)
    n_think = len(frontier_data)
    n_expert = len(expert_data)
    n_student = len(student_data)
    print(f'  Val: {n_val}, Thinking: {n_think}, '
          f'Expert: {n_expert}, Student: {n_student}')

    # -----------------------------------------------------------------
    # 2. Extract all predictions
    # -----------------------------------------------------------------
    print('[07] Extracting predictions...')

    # 2a. Best flagship under primary frontier metric
    global BEST_FLAGSHIP_KEY, BEST_FLAGSHIP_DISPLAY
    BEST_FLAGSHIP_KEY, best_flagship_primary = best_model_by_primary(frontier_data, FLAGSHIP_KEYS)
    BEST_FLAGSHIP_DISPLAY = FRONTIER_DISPLAY_NAMES.get(BEST_FLAGSHIP_KEY, BEST_FLAGSHIP_KEY)
    best_sft_single_key = compute_best_sft_single_key(val_data)
    _configure_evaluator_labels(BEST_FLAGSHIP_DISPLAY, best_sft_single_key)

    gt_flagship, pred_flagship = extract_flagship_predictions(
        frontier_data, BEST_FLAGSHIP_KEY, policy='majority'
    )
    pred_flagship_full = full_predictions(
        frontier_data,
        BEST_FLAGSHIP_KEY,
        policy='majority',
        tie_policy='exclude',
    )
    best_flagship_primary_scores = []
    for article in frontier_data:
        model_data = article.get('val_outcome', {}).get('rq_with_context', {}).get(BEST_FLAGSHIP_KEY, {})
        score = _primary_score_from_model_data(get_ground_truth(article), model_data)
        if score is not None:
            best_flagship_primary_scores.append(score)
    print(
        f'  Best Flagship ({BEST_FLAGSHIP_DISPLAY}): '
        f'primary={best_flagship_primary*100:.1f}%, '
        f'discrete N={len(pred_flagship)}'
    )

    # 2b. Flagship average (all 11 models)
    flagship_per_article, flagship_per_model, flagship_avg_acc = (
        compute_flagship_average(frontier_data)
    )
    flagship_macro_f1 = compute_flagship_average_macro_f1(
        frontier_data, n_bootstrap=BOOTSTRAP_N, seed=RANDOM_SEED
    )
    print(f'  Flagship average (11 models): {flagship_avg_acc*100:.1f}%')
    print(f'  Flagship average macro-F1: {flagship_macro_f1["estimate"]:.3f} '
          f'[{flagship_macro_f1["ci_lower"]:.3f}, {flagship_macro_f1["ci_upper"]:.3f}]')
    for key, acc in sorted(flagship_per_model.items(), key=lambda x: -x[1]):
        print(f'    {key}: {acc*100:.1f}%')

    # 2c. GPT-5.2 baseline (from chat data — moved from old 120_val.jsonl)
    try:
        chat_data = load_chat_data()
    except FileNotFoundError:
        chat_data = []
    gt_gpt, pred_gpt = extract_val_model_predictions(chat_data, 'gpt-5.2')
    if not pred_gpt:
        # Fallback: try trained data
        gt_gpt, pred_gpt = extract_val_model_predictions(val_data, 'gpt-5.2')
    print(f'  GPT-5.2 (pre-SFT): {len(pred_gpt)} predictions')

    # 2d. Best SFT single model (from val data)
    gt_sft1, pred_sft1 = extract_val_model_predictions(
        val_data, best_sft_single_key
    )
    print(f'  SFT single ({best_sft_single_key}): {len(pred_sft1)} predictions')

    # 2e. SFT 2-model ensemble (from val data)
    gt_sft2, pred_sft2 = extract_val_model_predictions(
        val_data, 'best_2_model_combo'
    )
    print(f'  SFT 2-model ensemble: {len(pred_sft2)} predictions')

    # 2f. Expert majority vote (exclude ties)
    gt_expert, pred_expert, expert_counts, expert_n_ties = (
        compute_expert_majority_votes(expert_data)
    )
    expert_valid = [(t, p) for t, p in zip(gt_expert, pred_expert) if p is not None]
    gt_expert_clean = [t for t, _ in expert_valid]
    pred_expert_clean = [p for _, p in expert_valid]
    print(f'  Expert majority: {len(pred_expert)} total, '
          f'{len(gt_expert_clean)} with clear majority, '
          f'{expert_n_ties} ties excluded')

    # 2h. Expert individual average
    expert_indiv = compute_individual_accuracy(expert_data)
    expert_pooled_true, expert_pooled_pred, expert_clusters = extract_individual_pooled_predictions(
        expert_data
    )
    expert_pooled_macro = bootstrap_macro_f1_ci_cluster(
        expert_clusters, n_bootstrap=BOOTSTRAP_N, seed=RANDOM_SEED
    )
    print(f'  Expert individual: {expert_indiv["n_raters"]} raters, '
          f'mean={expert_indiv["mean_accuracy"]*100:.1f}%')
    print(f'  Expert pooled macro-F1: {expert_pooled_macro["estimate"]:.3f} '
          f'[{expert_pooled_macro["ci_lower"]:.3f}, {expert_pooled_macro["ci_upper"]:.3f}] '
          f'from {len(expert_pooled_true)} pooled ratings')

    # 2i. Student majority vote (exclude ties)
    gt_student, pred_student, student_counts, student_n_ties = (
        compute_student_majority_votes(student_data)
    )
    student_valid = [(t, p) for t, p in zip(gt_student, pred_student) if p is not None]
    gt_student_clean = [t for t, _ in student_valid]
    pred_student_clean = [p for _, p in student_valid]
    print(f'  Student majority (full): {len(pred_student)} total, '
          f'{len(gt_student_clean)} with clear majority, '
          f'{student_n_ties} ties excluded')

    # 2j. Student individual average
    student_indiv = compute_individual_accuracy(student_data)
    student_pooled_true, student_pooled_pred, student_clusters = extract_individual_pooled_predictions(
        student_data
    )
    student_pooled_macro = bootstrap_macro_f1_ci_cluster(
        student_clusters, n_bootstrap=BOOTSTRAP_N, seed=RANDOM_SEED
    )
    print(f'  Student individual: {student_indiv["n_raters"]} raters, '
          f'mean={student_indiv["mean_accuracy"]*100:.1f}%')
    print(f'  Student pooled macro-F1: {student_pooled_macro["estimate"]:.3f} '
          f'[{student_pooled_macro["ci_lower"]:.3f}, {student_pooled_macro["ci_upper"]:.3f}] '
          f'from {len(student_pooled_true)} pooled ratings')

    # 2k. Student matched-N (Monte Carlo, exclude ties)
    print(f'  Running Monte Carlo for matched-N student '
          f'({MONTE_CARLO_TRIALS} trials)...')
    mc_result = compute_student_matched_n(
        student_data, expert_counts,
        n_trials=MONTE_CARLO_TRIALS, seed=RANDOM_SEED,
    )
    print(f'  Student matched-N: mean accuracy = '
          f'{mc_result["mean_accuracy"]:.3f} '
          f'[{mc_result["ci_lower"]:.3f}, {mc_result["ci_upper"]:.3f}] '
          f'mean effective N = {mc_result["mean_effective_n"]:.1f}')

    # Build matched-N representative predictions
    gt_matched = mc_result['ground_truths']
    pred_matched = mc_result['representative_preds']
    valid_matched = [
        (g, p) for g, p in zip(gt_matched, pred_matched) if p is not None
    ]
    gt_matched_clean = [g for g, _ in valid_matched]
    pred_matched_clean = [p for _, p in valid_matched]

    # -----------------------------------------------------------------
    # 3. Compute metrics for all evaluators
    # -----------------------------------------------------------------
    print('[07] Computing metrics...')

    all_evaluators = []   # Per-article evaluators (for kappa, McNemar)
    hero_entries = []     # All 10 entries for the hero figure

    def _add_evaluator(name, display, y_true, y_pred, ci_override=None):
        """Compute and store metrics for one evaluator with per-article preds."""
        acc = compute_accuracy(y_true, y_pred)
        f1 = compute_macro_f1(y_true, y_pred)
        f1_boot = bootstrap_macro_f1_ci(y_true, y_pred, BOOTSTRAP_N, RANDOM_SEED)
        tier_acc = per_tier_accuracy(y_true, y_pred)
        boot = bootstrap_accuracy_ci(y_true, y_pred, BOOTSTRAP_N, RANDOM_SEED)

        entry = {
            'name': name,
            'display_name': display,
            'accuracy': acc,
            'macro_f1': f1,
            'macro_f1_ci_lower': f1_boot['ci_lower'],
            'macro_f1_ci_upper': f1_boot['ci_upper'],
            'ci_lower': ci_override['ci_lower'] if ci_override else boot['ci_lower'],
            'ci_upper': ci_override['ci_upper'] if ci_override else boot['ci_upper'],
            'per_tier': tier_acc,
            'y_true': y_true,
            'y_pred': y_pred,
            'n': len(y_true),
        }
        all_evaluators.append(entry)
        hero_entries.append(entry)
        return entry

    def _add_aggregate_entry(
        name, display, accuracy, ci_lower, ci_upper, n,
        macro_f1=None, macro_f1_ci_lower=None, macro_f1_ci_upper=None
    ):
        """Add an aggregate-only entry (no per-article preds) to hero figure."""
        entry = {
            'name': name,
            'display_name': display,
            'accuracy': accuracy,
            'macro_f1': macro_f1,
            'macro_f1_ci_lower': macro_f1_ci_lower,
            'macro_f1_ci_upper': macro_f1_ci_upper,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'per_tier': None,
            'n': n,
        }
        hero_entries.append(entry)
        return entry

    # --- AI Models (5 bars) ---
    best_flagship_entry = _add_evaluator(
        'flagship', EVALUATOR_ORDER[0], gt_flagship, pred_flagship
    )
    best_flagship_boot = bootstrap_ci(
        np.array(best_flagship_primary_scores),
        np.mean,
        BOOTSTRAP_N,
        0.95,
        RANDOM_SEED,
    )
    best_flagship_entry['accuracy'] = best_flagship_primary
    best_flagship_entry['ci_lower'] = best_flagship_boot['ci_lower']
    best_flagship_entry['ci_upper'] = best_flagship_boot['ci_upper']
    best_flagship_entry['n'] = len(best_flagship_primary_scores)
    best_flagship_entry['accuracy_discrete_majority'] = compute_accuracy(gt_flagship, pred_flagship)
    best_flagship_entry['metric_protocol'] = 'primary'

    # Flagship average (aggregate)
    flagship_boot = bootstrap_ci(
        np.array(flagship_per_article), np.mean,
        BOOTSTRAP_N, 0.95, RANDOM_SEED,
    )
    _add_aggregate_entry(
        'flagship_avg', EVALUATOR_ORDER[1],
        flagship_avg_acc, flagship_boot['ci_lower'], flagship_boot['ci_upper'],
        n=len(frontier_data),
        macro_f1=flagship_macro_f1['estimate'],
        macro_f1_ci_lower=flagship_macro_f1['ci_lower'],
        macro_f1_ci_upper=flagship_macro_f1['ci_upper'],
    )

    _add_evaluator(
        'gpt52', EVALUATOR_ORDER[2], gt_gpt, pred_gpt
    )
    _add_evaluator(
        'sft_single', EVALUATOR_ORDER[3], gt_sft1, pred_sft1
    )
    _add_evaluator(
        'sft_2ensemble', EVALUATOR_ORDER[4], gt_sft2, pred_sft2
    )

    # --- Human Evaluators (5 bars) ---

    # Expert individual average (aggregate)
    expert_indiv_boot = bootstrap_ci(
        np.array(expert_indiv['accuracies']), np.mean,
        BOOTSTRAP_N, 0.95, RANDOM_SEED,
    )
    _add_aggregate_entry(
        'expert_indiv', EVALUATOR_ORDER[5],
        expert_indiv['mean_accuracy'],
        expert_indiv_boot['ci_lower'], expert_indiv_boot['ci_upper'],
        n=expert_indiv['n_raters'],
        macro_f1=expert_pooled_macro['estimate'],
        macro_f1_ci_lower=expert_pooled_macro['ci_lower'],
        macro_f1_ci_upper=expert_pooled_macro['ci_upper'],
    )

    _add_evaluator(
        'expert', EVALUATOR_ORDER[6],
        gt_expert_clean, pred_expert_clean
    )

    # Student individual average (aggregate)
    student_indiv_boot = bootstrap_ci(
        np.array(student_indiv['accuracies']), np.mean,
        BOOTSTRAP_N, 0.95, RANDOM_SEED,
    )
    _add_aggregate_entry(
        'student_indiv', EVALUATOR_ORDER[7],
        student_indiv['mean_accuracy'],
        student_indiv_boot['ci_lower'], student_indiv_boot['ci_upper'],
        n=student_indiv['n_raters'],
        macro_f1=student_pooled_macro['estimate'],
        macro_f1_ci_lower=student_pooled_macro['ci_lower'],
        macro_f1_ci_upper=student_pooled_macro['ci_upper'],
    )

    _add_evaluator(
        'student_matched', EVALUATOR_ORDER[8],
        gt_matched_clean, pred_matched_clean,
        ci_override={
            'ci_lower': mc_result['ci_lower'],
            'ci_upper': mc_result['ci_upper'],
        },
    )

    _add_evaluator(
        'student_full', EVALUATOR_ORDER[9],
        gt_student_clean, pred_student_clean
    )

    # Print summary
    print('\n  === Grand Comparison (Hero Figure) ===')
    for ev in hero_entries:
        ci_lo = ev['ci_lower'] * 100 if ev['ci_lower'] <= 1.0 else ev['ci_lower']
        ci_hi = ev['ci_upper'] * 100 if ev['ci_upper'] <= 1.0 else ev['ci_upper']
        dn = ev['display_name'].replace('\n', ' ')
        print(f"  {dn:35s}  "
              f"Acc={ev['accuracy']*100:.1f}%  "
              f"N={ev['n']}  "
              f"CI=[{ci_lo:.1f}, {ci_hi:.1f}]")

    # -----------------------------------------------------------------
    # 4. Grand comparison table (table8_ai_vs_human.csv)
    # -----------------------------------------------------------------
    print('\n[07] Building results table...')
    table_rows = []
    for i, ev in enumerate(hero_entries):
        row = {
            'Evaluator': EVALUATOR_SHORT[i],
            'N': ev['n'],
            'Accuracy (%)': round(ev['accuracy'] * 100, 1),
            'Macro F1': round(ev['macro_f1'], 3) if ev['macro_f1'] is not None else 'N/A',
            'Macro F1 95% CI Lower': (
                round(ev['macro_f1_ci_lower'], 3)
                if ev.get('macro_f1_ci_lower') is not None
                else 'N/A'
            ),
            'Macro F1 95% CI Upper': (
                round(ev['macro_f1_ci_upper'], 3)
                if ev.get('macro_f1_ci_upper') is not None
                else 'N/A'
            ),
            '95% CI Lower': round(
                ev['ci_lower'] * 100 if ev['ci_lower'] <= 1.0
                else ev['ci_lower'], 1
            ),
            '95% CI Upper': round(
                ev['ci_upper'] * 100 if ev['ci_upper'] <= 1.0
                else ev['ci_upper'], 1
            ),
        }
        if ev.get('per_tier') is not None:
            for tier in LABEL_ORDER:
                row[f'Acc {tier.capitalize()} (%)'] = round(
                    ev['per_tier'].get(tier, 0.0) * 100, 1
                )
        else:
            for tier in LABEL_ORDER:
                row[f'Acc {tier.capitalize()} (%)'] = 'N/A'
        table_rows.append(row)

    df_table = pd.DataFrame(table_rows)
    table_path = tables_dir / 'table8_ai_vs_human.csv'
    df_table.to_csv(table_path, index=False)
    print(f'  Saved: {table_path}')

    # -----------------------------------------------------------------
    # 5. Agreement analysis (Cohen's kappa heatmap)
    # -----------------------------------------------------------------
    print('\n[07] Computing agreement (Cohen\'s kappa)...')

    # Build title-based prediction dicts
    val_titles = [a['title'] for a in val_data]

    def _build_pred_by_title(data_list, y_pred):
        """Map titles to predictions (may include None for ties)."""
        result = {}
        for article, pred in zip(data_list, y_pred):
            result[article['title']] = pred
        return result

    flagship_by_title = _build_pred_by_title(frontier_data, pred_flagship_full)
    gpt_by_title = _build_pred_by_title(val_data, pred_gpt)
    sft1_by_title = _build_pred_by_title(val_data, pred_sft1)
    sft2_by_title = _build_pred_by_title(val_data, pred_sft2)
    expert_by_title = _build_pred_by_title(expert_data, pred_expert)
    student_by_title = _build_pred_by_title(student_data, pred_student)
    matched_by_title = _build_pred_by_title(student_data, pred_matched)

    # Kappa matrix: use per-article evaluators only (no individual averages
    # or flagship average since they lack single per-article predictions)
    kappa_eval_names = [
        EVALUATOR_ORDER[0],   # Best Flagship
        EVALUATOR_ORDER[2],   # GPT-5.2
        EVALUATOR_ORDER[3],   # Best SFT Single
        EVALUATOR_ORDER[4],   # SFT 2-Model Ensemble
        EVALUATOR_ORDER[6],   # Expert Majority
        EVALUATOR_ORDER[9],   # Student Majority (full)
        EVALUATOR_ORDER[8],   # Student Matched N
    ]
    kappa_pred_dicts = [
        flagship_by_title, gpt_by_title, sft1_by_title,
        sft2_by_title,
        expert_by_title, student_by_title, matched_by_title,
    ]

    # Find common titles across all sources
    common_titles = set(val_titles)
    for pd_dict in kappa_pred_dicts:
        common_titles &= set(pd_dict.keys())
    common_titles = sorted(common_titles)
    print(f'  Common titles for kappa: {len(common_titles)}')

    # Build aligned prediction lists
    aligned_preds = []
    for pd_dict in kappa_pred_dicts:
        aligned_preds.append([pd_dict[t] for t in common_titles])

    n_kappa = len(kappa_eval_names)
    kappa_matrix = np.ones((n_kappa, n_kappa))

    kappa_results = {}
    for i in range(n_kappa):
        for j in range(i + 1, n_kappa):
            # Filter out None predictions (ties)
            valid = [
                (a, b) for a, b in zip(aligned_preds[i], aligned_preds[j])
                if a is not None and b is not None
            ]
            if valid:
                y1 = [a for a, _ in valid]
                y2 = [b for _, b in valid]
                kappa = compute_cohens_kappa(y1, y2)
            else:
                kappa = 0.0
            kappa_matrix[i, j] = kappa
            kappa_matrix[j, i] = kappa
            pair_name = f'{kappa_eval_names[i]} vs {kappa_eval_names[j]}'
            kappa_results[pair_name] = kappa

    # Print notable kappa values
    print('  Notable kappa values:')
    for pair, k in sorted(kappa_results.items(), key=lambda x: -x[1])[:10]:
        print(f'    {pair}: {k:.3f}')

    # -----------------------------------------------------------------
    # 6. Complementarity analysis
    # -----------------------------------------------------------------
    print('\n[07] Complementarity analysis...')

    # SFT 2-model ensemble vs Expert majority
    # Only include articles where both have non-None predictions
    comp_titles = sorted(
        set(sft2_by_title.keys()) & set(expert_by_title.keys())
    )
    comp_gt = []
    comp_ai = []
    comp_human = []
    for t in comp_titles:
        if sft2_by_title.get(t) is None or expert_by_title.get(t) is None:
            continue
        for article in val_data:
            if article['title'] == t:
                comp_gt.append(get_ground_truth(article))
                break
        comp_ai.append(sft2_by_title[t])
        comp_human.append(expert_by_title[t])

    comp = complementarity_analysis(comp_gt, comp_ai, comp_human)
    print(f'  Both correct:       {comp["both_correct"]} ({comp["both_correct_pct"]:.1f}%)')
    print(f'  AI only correct:    {comp["ai_only_correct"]} ({comp["ai_only_pct"]:.1f}%)')
    print(f'  Human only correct: {comp["human_only_correct"]} ({comp["human_only_pct"]:.1f}%)')
    print(f'  Neither correct:    {comp["neither_correct"]} ({comp["neither_pct"]:.1f}%)')
    print(f'  Combined accuracy:  {comp["combined_accuracy"]*100:.1f}%')

    # -----------------------------------------------------------------
    # 7. Statistical significance tests
    # -----------------------------------------------------------------
    print('\n[07] Statistical significance tests...')

    mcnemar_pairs = [
        ('sft_2ensemble', 'expert', 'SFT 2-Ensemble vs Expert'),
        ('sft_2ensemble', 'student_full', 'SFT 2-Ensemble vs Student (Full)'),
        ('expert', 'student_full', 'Expert vs Student (Full)'),
        ('sft_2ensemble', 'gpt52', 'SFT 2-Ensemble vs GPT-5.2'),
        ('sft_single', 'gpt52', 'SFT Single vs GPT-5.2'),
        ('sft_2ensemble', 'sft_single', 'SFT 2-Ensemble vs SFT Single'),
        ('sft_2ensemble', 'flagship', 'SFT 2-Ensemble vs Flagship'),
        ('flagship', 'expert', 'Flagship vs Expert'),
        ('flagship', 'gpt52', 'Flagship vs GPT-5.2'),
    ]

    pred_dicts_mcn = {
        'flagship': flagship_by_title,
        'gpt52': gpt_by_title,
        'sft_single': sft1_by_title,
        'sft_2ensemble': sft2_by_title,
        'expert': expert_by_title,
        'student_full': student_by_title,
        'student_matched': matched_by_title,
    }

    # Build ground-truth map by title
    gt_map = {}
    for a in val_data:
        gt_map[a['title']] = get_ground_truth(a)
    for a in frontier_data:
        if a['title'] not in gt_map:
            gt_map[a['title']] = get_ground_truth(a)

    mcnemar_results = {}
    raw_p_values = []
    pair_labels = []

    for name1, name2, label in mcnemar_pairs:
        d1 = pred_dicts_mcn[name1]
        d2 = pred_dicts_mcn[name2]
        shared = sorted(set(d1.keys()) & set(d2.keys()))

        # Filter: ground truth known, both predictions non-None
        valid_titles = [
            t for t in shared
            if t in gt_map and d1.get(t) is not None and d2.get(t) is not None
        ]

        yt = [gt_map[t] for t in valid_titles]
        yp1 = [d1[t] for t in valid_titles]
        yp2 = [d2[t] for t in valid_titles]

        if len(yt) > 0:
            result = mcnemar_test(yt, yp1, yp2)
        else:
            result = {'statistic': 0.0, 'p_value': 1.0, 'b': 0, 'c': 0}

        mcnemar_results[label] = result
        raw_p_values.append(result['p_value'])
        pair_labels.append(label)

    # Bonferroni correction
    bonf = bonferroni_correction(raw_p_values, alpha=0.05)

    print('  McNemar test results (with Bonferroni correction):')
    for i, label in enumerate(pair_labels):
        res = mcnemar_results[label]
        corrected_p = bonf['corrected_p_values'][i]
        sig = '*' if bonf['rejected'][i] else 'n.s.'
        print(f'    {label:40s}  chi2={res["statistic"]:.2f}  '
              f'p={res["p_value"]:.4f}  p_corr={corrected_p:.4f}  {sig}')

    # Bootstrap CIs for accuracy differences (key pairs)
    print('\n  Bootstrap CIs for accuracy differences:')
    bootstrap_diff_results = {}
    for name1, name2, label in mcnemar_pairs[:6]:
        d1 = pred_dicts_mcn[name1]
        d2 = pred_dicts_mcn[name2]
        shared = sorted(set(d1.keys()) & set(d2.keys()))
        shared = [
            t for t in shared
            if t in gt_map and d1.get(t) is not None and d2.get(t) is not None
        ]

        if not shared:
            continue

        correct1 = np.array([
            1.0 if d1[t] == gt_map[t] else 0.0 for t in shared
        ])
        correct2 = np.array([
            1.0 if d2[t] == gt_map[t] else 0.0 for t in shared
        ])

        diff = correct1 - correct2
        boot_diff = bootstrap_ci(
            diff, statistic_fn=np.mean,
            n_bootstrap=BOOTSTRAP_N, ci=0.95, seed=RANDOM_SEED,
        )
        bootstrap_diff_results[label] = boot_diff
        print(f'    {label:40s}  diff={boot_diff["estimate"]*100:.1f}%  '
              f'CI=[{boot_diff["ci_lower"]*100:.1f}, '
              f'{boot_diff["ci_upper"]*100:.1f}]')

    # -----------------------------------------------------------------
    # 8. Build statistics output
    # -----------------------------------------------------------------
    print('\n[07] Building statistics output...')

    stats_output = {
        'protocol': {
            'frontier_primary_metric': 'avg8_article_mean_for_stochastic__discrete_for_deterministic',
            'frontier_discrete_policy_for_tests': 'majority_vote_excluding_ties',
        },
        'grand_comparison': [],
        'best_flagship': {
            'model_key': BEST_FLAGSHIP_KEY,
            'display_name': BEST_FLAGSHIP_DISPLAY,
            'accuracy_primary': best_flagship_primary,
            'accuracy_majority': compute_accuracy(gt_flagship, pred_flagship) if pred_flagship else 0.0,
            'ci_lower_primary': best_flagship_boot['ci_lower'],
            'ci_upper_primary': best_flagship_boot['ci_upper'],
            'n_primary': len(best_flagship_primary_scores),
            'n_discrete_majority': len(pred_flagship),
        },
        'flagship_average': {
            'accuracy': flagship_avg_acc,
            'macro_f1': flagship_macro_f1['estimate'],
            'macro_f1_ci_lower': flagship_macro_f1['ci_lower'],
            'macro_f1_ci_upper': flagship_macro_f1['ci_upper'],
            'per_model_accuracies': {
                k: round(v, 4) for k, v in flagship_per_model.items()
            },
            'per_model_macro_f1': {
                k: round(v, 6) for k, v in flagship_macro_f1['per_model_macro_f1'].items()
            },
            'ci_lower': flagship_boot['ci_lower'],
            'ci_upper': flagship_boot['ci_upper'],
        },
        'individual_averages': {
            'expert': {
                'n_raters': expert_indiv['n_raters'],
                'mean_accuracy': expert_indiv['mean_accuracy'],
                'std': expert_indiv['std'],
                'ci_lower': expert_indiv_boot['ci_lower'],
                'ci_upper': expert_indiv_boot['ci_upper'],
                'pooled_macro_f1': expert_pooled_macro['estimate'],
                'pooled_macro_f1_ci_lower': expert_pooled_macro['ci_lower'],
                'pooled_macro_f1_ci_upper': expert_pooled_macro['ci_upper'],
                'n_pooled_ratings': len(expert_pooled_true),
            },
            'student': {
                'n_raters': student_indiv['n_raters'],
                'mean_accuracy': student_indiv['mean_accuracy'],
                'std': student_indiv['std'],
                'ci_lower': student_indiv_boot['ci_lower'],
                'ci_upper': student_indiv_boot['ci_upper'],
                'pooled_macro_f1': student_pooled_macro['estimate'],
                'pooled_macro_f1_ci_lower': student_pooled_macro['ci_lower'],
                'pooled_macro_f1_ci_upper': student_pooled_macro['ci_upper'],
                'n_pooled_ratings': len(student_pooled_true),
            },
        },
        'expert_majority_vote_details': {
            'n_total': len(expert_data),
            'n_ties': expert_n_ties,
            'n_clear_majority': len(gt_expert_clean),
            'accuracy_excl_ties': compute_accuracy(gt_expert_clean, pred_expert_clean),
        },
        'student_majority_vote_details': {
            'n_total': len(student_data),
            'n_ties': student_n_ties,
            'n_clear_majority': len(gt_student_clean),
            'accuracy_excl_ties': compute_accuracy(gt_student_clean, pred_student_clean),
        },
        'kappa_matrix': {
            'labels': [n.replace('\n', ' ') for n in kappa_eval_names],
            'values': kappa_matrix.tolist(),
        },
        'kappa_pairwise': {
            k.replace('\n', ' '): v for k, v in kappa_results.items()
        },
        'complementarity': {
            k: v for k, v in comp.items()
            if k not in ('combined_preds', 'articles_detail')
        },
        'mcnemar_tests': {},
        'bonferroni_correction': {
            'alpha_corrected': bonf['alpha_corrected'],
            'n_tests': bonf['n_tests'],
        },
        'bootstrap_accuracy_differences': {},
        'student_matched_n_monte_carlo': {
            'mean_accuracy': mc_result['mean_accuracy'],
            'std': mc_result['std'],
            'ci_lower': mc_result['ci_lower'],
            'ci_upper': mc_result['ci_upper'],
            'n_trials': mc_result['n_trials'],
            'mean_effective_n': mc_result['mean_effective_n'],
        },
    }

    for ev in hero_entries:
        entry = {
            'evaluator': ev['display_name'].replace('\n', ' '),
            'n': ev['n'],
            'accuracy': ev['accuracy'],
            'ci_lower': ev['ci_lower'],
            'ci_upper': ev['ci_upper'],
        }
        if ev.get('macro_f1') is not None:
            entry['macro_f1'] = ev['macro_f1']
        if ev.get('macro_f1_ci_lower') is not None:
            entry['macro_f1_ci_lower'] = ev['macro_f1_ci_lower']
        if ev.get('macro_f1_ci_upper') is not None:
            entry['macro_f1_ci_upper'] = ev['macro_f1_ci_upper']
        if ev.get('per_tier') is not None:
            entry['per_tier_accuracy'] = ev['per_tier']
        stats_output['grand_comparison'].append(entry)

    for i, label in enumerate(pair_labels):
        stats_output['mcnemar_tests'][label] = {
            'statistic': mcnemar_results[label]['statistic'],
            'p_value': mcnemar_results[label]['p_value'],
            'p_corrected': bonf['corrected_p_values'][i],
            'significant': bonf['rejected'][i],
            'b': mcnemar_results[label]['b'],
            'c': mcnemar_results[label]['c'],
        }

    for label, boot in bootstrap_diff_results.items():
        stats_output['bootstrap_accuracy_differences'][label] = {
            'estimate': boot['estimate'],
            'ci_lower': boot['ci_lower'],
            'ci_upper': boot['ci_upper'],
            'std': boot['std'],
        }

    stats_path = stats_dir / '07_ai_vs_human.json'
    with open(stats_path, 'w', encoding='utf-8') as fh:
        json.dump(stats_output, fh, indent=2, ensure_ascii=False, default=str)
    print(f'  Saved: {stats_path}')

    # -----------------------------------------------------------------
    # 9. Generate figures
    # -----------------------------------------------------------------
    print('\n[07] Generating figures...')

    # Hero figure (10 bars)
    plot_hero_figure(hero_entries, figures_dir)
    print(f'  Saved: {figures_dir}/fig1_grand_comparison.pdf')

    # Agreement heatmap
    plot_agreement_heatmap(kappa_matrix, kappa_eval_names, figures_dir)
    print(f'  Saved: {figures_dir}/fig_s10_agreement_heatmap.pdf')

    # Complementarity figure
    plot_complementarity(comp, figures_dir)
    print(f'  Saved: {figures_dir}/fig8_complementarity_venn.pdf')

    # Supplementary: all 11 flagship models
    plot_all_flagships(flagship_per_model, figures_dir)
    print(f'  Saved: {figures_dir}/fig_s11_all_flagships.pdf')

    # -----------------------------------------------------------------
    # 10. Per-tier comparison
    # -----------------------------------------------------------------
    print('\n[07] Per-tier accuracy comparison:')
    for tier in LABEL_ORDER:
        line = f'  {tier:12s}  '
        for ev in hero_entries:
            if ev.get('per_tier') is not None:
                acc = ev['per_tier'].get(tier, 0.0) * 100
            else:
                acc = float('nan')
            short = ev['display_name'].replace('\n', ' ')[:15]
            line += f'{short}={acc:.0f}%  '
        print(line)

    # -----------------------------------------------------------------
    # 11. Chat model supplementary comparison
    # -----------------------------------------------------------------
    print('\n[07] Chat model supplementary comparison...')
    try:
        chat_data = load_chat_data()
        chat_model_results = {}
        for art in chat_data:
            gt = get_ground_truth(art)
            rq = art.get('val_outcome', {}).get('rq_with_context', {})
            for model_key in CHAT_MODELS:
                entry = rq.get(model_key, {})
                logp = entry.get('logp')
                if logp:
                    pred = _pred_from_logp(logp)
                    if pred is not None:
                        if model_key not in chat_model_results:
                            chat_model_results[model_key] = {'y_true': [], 'y_pred': []}
                        chat_model_results[model_key]['y_true'].append(gt)
                        chat_model_results[model_key]['y_pred'].append(pred)

        chat_rows = []
        for model_key, data_cm in chat_model_results.items():
            yt, yp = data_cm['y_true'], data_cm['y_pred']
            acc = compute_accuracy(yt, yp) if yt else 0.0
            f1 = compute_macro_f1(yt, yp) if yt else 0.0
            boot = bootstrap_accuracy_ci(yt, yp, BOOTSTRAP_N, RANDOM_SEED) if yt else {'ci_lower': 0, 'ci_upper': 0}
            f1_boot = bootstrap_macro_f1_ci(yt, yp, BOOTSTRAP_N, RANDOM_SEED) if yt else {'ci_lower': 0, 'ci_upper': 0}
            display = MODEL_DISPLAY_NAMES.get(model_key, model_key)
            print(f'  {display}: Acc={acc*100:.1f}%, N={len(yt)}, '
                  f'CI=[{boot["ci_lower"]*100:.1f}, {boot["ci_upper"]*100:.1f}]')
            chat_rows.append({
                'Evaluator': f'{display} (chat, logp)',
                'N': len(yt),
                'Accuracy (%)': round(acc * 100, 1),
                'Macro F1': round(f1, 3) if f1 else 'N/A',
                'Macro F1 95% CI Lower': round(f1_boot['ci_lower'], 3) if yt else 'N/A',
                'Macro F1 95% CI Upper': round(f1_boot['ci_upper'], 3) if yt else 'N/A',
                '95% CI Lower': round(boot['ci_lower'] * 100, 1),
                '95% CI Upper': round(boot['ci_upper'] * 100, 1),
            })
            stats_output[f'chat_model_{model_key}'] = {
                'accuracy': acc, 'macro_f1': f1, 'n': len(yt),
                'ci_lower': boot['ci_lower'], 'ci_upper': boot['ci_upper'],
                'macro_f1_ci_lower': f1_boot['ci_lower'], 'macro_f1_ci_upper': f1_boot['ci_upper'],
            }

        if chat_rows:
            df_chat = pd.DataFrame(chat_rows)
            chat_table_path = tables_dir / 'table_s_chat_models.csv'
            df_chat.to_csv(chat_table_path, index=False)
            print(f'  Saved: {chat_table_path}')
    except FileNotFoundError:
        print('  WARNING: 120_chat.jsonl not found, skipping chat models')

    # -----------------------------------------------------------------
    # 12. Sensitivity: Filtered expert comparison
    # -----------------------------------------------------------------
    print('\n[07] Sensitivity: Filtered expert comparison...')
    try:
        expert_filtered = load_expert_filtered_ratings()
        gt_ef, pred_ef, ef_counts, ef_ties = compute_expert_majority_votes(expert_filtered)
        ef_valid = [(t, p) for t, p in zip(gt_ef, pred_ef) if p is not None]
        gt_ef_clean = [t for t, _ in ef_valid]
        pred_ef_clean = [p for _, p in ef_valid]
        ef_acc = compute_accuracy(gt_ef_clean, pred_ef_clean) if gt_ef_clean else 0.0

        ef_indiv = compute_individual_accuracy(expert_filtered)

        # Count unique filtered raters
        ef_rater_names = set()
        for art in expert_filtered:
            for r in art.get('ratings', []):
                ef_rater_names.add(r.get('rater_id', r.get('rater_name', r.get('name', 'unknown'))))

        print(f'  Filtered experts: {len(ef_rater_names)} raters')
        print(f'  Filtered majority vote: {ef_acc*100:.1f}% '
              f'(N={len(gt_ef_clean)}, {ef_ties} ties)')
        print(f'  Filtered individual avg: {ef_indiv["mean_accuracy"]*100:.1f}% '
              f'(N={ef_indiv["n_raters"]})')

        # Count unfiltered raters for comparison
        uf_rater_names = set()
        for art in expert_data:
            for r in art.get('ratings', []):
                uf_rater_names.add(r.get('rater_id', r.get('rater_name', r.get('name', 'unknown'))))

        stats_output['sensitivity_filtered_experts'] = {
            'n_raters_unfiltered': len(uf_rater_names),
            'n_raters_filtered': len(ef_rater_names),
            'majority_vote_unfiltered': {
                'accuracy': compute_accuracy(gt_expert_clean, pred_expert_clean),
                'n_clear_majority': len(gt_expert_clean),
                'n_ties': expert_n_ties,
            },
            'majority_vote_filtered': {
                'accuracy': ef_acc,
                'n_clear_majority': len(gt_ef_clean),
                'n_ties': ef_ties,
            },
            'individual_avg_unfiltered': {
                'mean_accuracy': expert_indiv['mean_accuracy'],
                'n_raters': expert_indiv['n_raters'],
            },
            'individual_avg_filtered': {
                'mean_accuracy': ef_indiv['mean_accuracy'],
                'n_raters': ef_indiv['n_raters'],
            },
        }
    except FileNotFoundError:
        print('  WARNING: Filtered expert file not found, skipping sensitivity')

    # Re-save stats with chat model + sensitivity data
    with open(stats_path, 'w', encoding='utf-8') as fh:
        json.dump(stats_output, fh, indent=2, ensure_ascii=False, default=str)
    print(f'  Updated stats: {stats_path}')

    print('\n[07] Analysis complete.')
    print(f'  Figures: {figures_dir}/')
    print(f'  Tables:  {tables_dir}/')
    print(f'  Stats:   {stats_dir}/')


if __name__ == '__main__':
    main()
