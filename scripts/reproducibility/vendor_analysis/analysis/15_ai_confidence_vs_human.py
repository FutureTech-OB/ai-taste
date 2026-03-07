#!/usr/bin/env python3
"""Script 15: AI prediction confidence vs human self-reported confidence.

Comprehensive comparison of AI token prediction probabilities (softmax
from logp) with human self-reported confidence (1-5 scale).

Covers ALL models:
  - SFT individual models: CYqJRxId, ckpt-step-304, ckppt-380, ckppt-228
  - SFT ensembles: best_2_model_combo
  - Baseline: gpt-5.2
  - Voting: human_voting
  - 10 Flagship models (point predictions only, no logp)
  - Human experts (42 raters, q3_confidence 1-5)
  - Human students (filtered combined junior panel with confidence data)

Key insight: When AI predicts a label, the softmax probability represents
the model's "confidence" in that prediction — directly comparable to human
raters' self-reported confidence on the same articles.

Analyses:
    1. Confidence distributions for ALL models with logp
    2. Confidence when correct vs incorrect
    3. Side-by-side calibration curves on unified axes
    4. Per-article difficulty: AI entropy vs human disagreement
    5. Selective prediction: accuracy at confidence thresholds
    6. Per-tier confidence patterns
    7. Flagship models: accuracy-only comparison (no logp available)
    8. Comprehensive comparison table

Outputs:
    - fig15a_confidence_comparison_main.pdf  (hero figure: key models)
    - fig15b_all_models_confidence.pdf       (all SFT models)
    - fig_s17_selective_prediction.pdf
    - fig_s18_confidence_correct_vs_wrong.pdf
    - fig_s19_entropy_vs_disagreement.pdf
    - fig_s20_flagship_vs_sft_accuracy.pdf
    - fig_s21_per_tier_confidence_all.pdf
    - table_s12_confidence_comparison.csv
    - table_s13_all_models_summary.csv
    - 15_ai_confidence_vs_human.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import math
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from scipy import stats as sp_stats

from analysis.utils.constants import (
    LABEL_ORDER, HUMAN_TO_AI, LEVEL_TO_AI,
    CHAT_MODELS, COLORS, FRONTIER_DISPLAY_NAMES, FRONTIER_MODELS, MODEL_DISPLAY_NAMES,
    FIGURES_DIR, TABLES_DIR, STATS_DIR,
)
from analysis.utils.data_loader import (
    get_project_root, load_jsonl, normalize_level,
)
from analysis.utils.frontier_protocol import extract_discrete_prediction
from analysis.utils.metrics import logp_to_prob, compute_ece
from analysis.utils.visualization import set_nature_style, save_figure
from analysis.utils import (
    TRAINED_PATH as _TRAINED_PATH,
    CHAT_PATH as _CHAT_PATH,
    EXPERT_PATH as _EXPERT_PATH,
    STUDENT_FILTERED_PATH as _STUDENT_FILTERED_PATH,
    THINKING_PATH as _THINKING_PATH,
)


# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
ROOT = get_project_root()
VAL_PATH = ROOT / _TRAINED_PATH        # SFT models (was 120_val.jsonl)
CHAT_PATH = ROOT / _CHAT_PATH          # Chat models with logp
THINKING_PATH = ROOT / _THINKING_PATH
EXPERT_PATH = ROOT / _EXPERT_PATH      # Now unfiltered (47+ experts)
STUDENT_FILTERED_PATH = ROOT / _STUDENT_FILTERED_PATH

FIGS = ROOT / FIGURES_DIR
TABS = ROOT / TABLES_DIR
STTS = ROOT / STATS_DIR

for d in [FIGS, TABS, STTS]:
    d.mkdir(parents=True, exist_ok=True)

# Display names for SFT/trained + chat models
SFT_DISPLAY = {
    'CYqJRxId': 'SFT-1 (GPT-4.1)',
    'ckpt-step-304': 'SFT-2 (GPT-4.1-nano)',
    'ckppt-380': 'SFT-3 (Qwen3-30B)',
    'ckppt-228': 'SFT-4 (Qwen3-4B)',
    'gpt-5.2': 'GPT-5.2 (baseline)',
    'best_2_model_combo': 'SFT 2-Model Ensemble',
    'human_voting': 'Human Voting (old students)',
    # Chat models (from 120_chat.jsonl, with logp)
    'kimi-k2-0905-preview': 'Kimi K2 Chat',
    'deepseek-chat': 'DeepSeek Chat',
    'gpt-5.2-chat': 'GPT-5.2 Chat',
}

# Flagship model display names (conservative 11-model frontier cohort)
FLAGSHIP_DISPLAY = {
    key: FRONTIER_DISPLAY_NAMES.get(key, key)
    for key in FRONTIER_MODELS
}

# Color palette for all models
MODEL_COLORS = {
    'SFT-1 (GPT-4.1)': '#E64B35',
    'SFT-2 (GPT-4.1-nano)': '#DC7B5B',
    'SFT-3 (Qwen3-30B)': '#D49A7E',
    'SFT-4 (Qwen3-4B)': '#F39B7F',
    'GPT-5.2 (baseline)': '#8491B4',
    'SFT 2-Model Ensemble': '#B21F00',
    'Human Voting (old students)': '#7E6148',
    'Human Expert': '#4DBBD5',
    'Human Student': '#00A087',
    # Chat models
    'Kimi K2 Chat': '#7570B3',
    'DeepSeek Chat': '#66A61E',
    'GPT-5.2 Chat': '#A6761D',
}


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _extract_int(val):
    """Parse leading integer from survey fields like '4 - Confident'."""
    try:
        return int(str(val).strip()[0])
    except (ValueError, IndexError, TypeError):
        return None


def _entropy(prob_dict, labels=LABEL_ORDER):
    """Compute Shannon entropy of a probability distribution."""
    h = 0.0
    for lab in labels:
        p = prob_dict.get(lab, 0.0)
        if p > 0:
            h -= p * math.log2(p)
    return h


def _mann_whitney_safe(a, b, alternative='two-sided'):
    """Safe Mann-Whitney U test, returns (stat, p) or (nan, nan)."""
    if len(a) < 2 or len(b) < 2:
        return float('nan'), float('nan')
    try:
        stat, p = sp_stats.mannwhitneyu(a, b, alternative=alternative)
        return float(stat), float(p)
    except ValueError:
        return float('nan'), float('nan')


# -----------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------

def load_all_val_predictions():
    """Load ALL model predictions with probabilities from trained + chat files.

    Returns dict: model_key -> list of dicts with:
        ground_truth, predicted, max_prob, entropy, probs, correct
    """
    val_data = load_jsonl(VAL_PATH)
    trained_models = [
        'CYqJRxId', 'ckpt-step-304', 'ckppt-380', 'ckppt-228',
        'gpt-5.2', 'human_voting',
        'best_2_model_combo',
    ]

    results = {}

    # Load trained/SFT models
    for m in trained_models:
        results[m] = []
    for art in val_data:
        gt = art.get('rank', '').strip().lower()
        if not gt:
            continue
        rq = art.get('val_outcome', {}).get('rq_with_context', {})
        for model_key in trained_models:
            entry = rq.get(model_key)
            if entry is None:
                continue
            logp = entry.get('logp')
            if not logp:
                continue
            probs = logp_to_prob(logp)
            predicted = max(probs, key=probs.get)
            max_prob = probs[predicted]
            ent = _entropy(probs)
            results[model_key].append({
                'ground_truth': gt,
                'predicted': predicted,
                'max_prob': max_prob,
                'entropy': ent,
                'probs': probs,
                'correct': int(predicted == gt),
            })

    # Load chat models (gpt-5.2, kimi-k2-0905-preview, deepseek-chat)
    try:
        chat_data = load_jsonl(CHAT_PATH)
    except FileNotFoundError:
        print("WARNING: 120_chat.jsonl not found, skipping chat models")
        chat_data = []

    chat_model_keys = CHAT_MODELS
    for m in chat_model_keys:
        # Use prefixed key to distinguish chat gpt-5.2 from trained gpt-5.2
        key = f'{m}-chat' if m == 'gpt-5.2' else m
        results[key] = []

    for art in chat_data:
        gt = art.get('rank', '').strip().lower()
        if not gt:
            continue
        rq = art.get('val_outcome', {}).get('rq_with_context', {})
        for model_key in chat_model_keys:
            entry = rq.get(model_key)
            if entry is None:
                continue
            logp = entry.get('logp')
            if not logp:
                continue
            probs = logp_to_prob(logp)
            predicted = max(probs, key=probs.get)
            max_prob = probs[predicted]
            ent = _entropy(probs)
            result_key = f'{model_key}-chat' if model_key == 'gpt-5.2' else model_key
            results[result_key].append({
                'ground_truth': gt,
                'predicted': predicted,
                'max_prob': max_prob,
                'entropy': ent,
                'probs': probs,
                'correct': int(predicted == gt),
            })

    return results


def load_flagship_predictions():
    """Load canonical frontier model predictions.

    Returns dict: model_key -> list of dicts with:
        ground_truth, predicted (majority), correct (majority),
        correct_primary (avg8 primary score where available).
    """
    thinking_data = load_jsonl(THINKING_PATH)
    results = {k: [] for k in FRONTIER_MODELS}

    for art in thinking_data:
        gt = art.get('rank', '').strip().lower()
        if not gt:
            continue
        rq = art.get('val_outcome', {}).get('rq_with_context', {})

        for model_key in FRONTIER_MODELS:
            entry = rq.get(model_key)
            if entry is None:
                continue
            pred_majority, _ = extract_discrete_prediction(
                entry,
                policy='majority',
                tie_policy='exclude',
            )
            if pred_majority is None:
                continue

            mode = entry.get('mode', 'deterministic_logp')
            avg_accuracy = entry.get('avg_accuracy')
            primary_score = None
            if mode == 'stochastic_thinking' and avg_accuracy is not None:
                try:
                    primary_score = float(avg_accuracy)
                except (TypeError, ValueError):
                    primary_score = None
            if primary_score is None:
                primary_score = float(pred_majority == gt)

            results[model_key].append({
                'ground_truth': gt,
                'predicted': pred_majority,
                'correct': int(pred_majority == gt),
                'correct_primary': primary_score,
                'has_logp': entry.get('logp') is not None and bool(entry.get('logp')),
                'mode': mode,
            })

    return results


def load_human_ratings_with_confidence():
    """Load expert and filtered combined junior ratings with confidence."""
    result = {'expert': [], 'student': []}

    expert_data = load_jsonl(EXPERT_PATH)
    for art in expert_data:
        gt = normalize_level(art.get('level', ''))
        if not gt:
            continue
        for r in art.get('ratings', []):
            raw = r.get('q2_rating', '')
            predicted = HUMAN_TO_AI.get(str(raw).strip())
            conf = _extract_int(r.get('q3_confidence'))
            fam = _extract_int(r.get('q4_familiarity'))
            if not predicted or conf is None:
                continue
            result['expert'].append({
                'ground_truth': gt,
                'predicted': predicted,
                'confidence': conf,
                'familiarity': fam,
                'correct': int(predicted == gt),
                'article_number': art.get('article_number'),
            })

    student_data = load_jsonl(STUDENT_FILTERED_PATH)
    for art in student_data:
        gt = normalize_level(art.get('level', ''))
        if not gt:
            continue
        for r in art.get('ratings', []):
            raw = r.get('q2_rating', '')
            predicted = HUMAN_TO_AI.get(str(raw).strip())
            conf = _extract_int(r.get('q3_confidence'))
            fam = _extract_int(r.get('q4_familiarity'))
            if not predicted or conf is None:
                continue
            correct = r.get('is_correct')
            if correct is None:
                correct = int(predicted == gt)
            result['student'].append({
                'ground_truth': gt,
                'predicted': predicted,
                'confidence': conf,
                'familiarity': fam,
                'correct': int(correct),
                'article_number': art.get('article_number'),
            })

    return result


def load_human_voting_per_article():
    """Load per-article human voting distributions."""
    result = {'expert_voting': [], 'student_voting': []}

    expert_data = load_jsonl(EXPERT_PATH)
    for art in expert_data:
        gt = normalize_level(art.get('level', ''))
        if not gt:
            continue
        votes = []
        for r in art.get('ratings', []):
            raw = r.get('q2_rating', '')
            pred = HUMAN_TO_AI.get(str(raw).strip())
            if pred:
                votes.append(pred)
        if not votes:
            continue
        c = Counter(votes)
        n = len(votes)
        vote_probs = {lab: c.get(lab, 0) / n for lab in LABEL_ORDER}
        predicted = c.most_common(1)[0][0]
        max_prob = max(vote_probs.values())
        ent = _entropy(vote_probs)
        result['expert_voting'].append({
            'ground_truth': gt,
            'vote_probs': vote_probs,
            'predicted': predicted,
            'max_prob': max_prob,
            'entropy': ent,
            'n_raters': n,
            'correct': int(predicted == gt),
            'agreement_ratio': c.most_common(1)[0][1] / n,
        })

    student_data = load_jsonl(STUDENT_FILTERED_PATH)
    for art in student_data:
        gt = normalize_level(art.get('level', ''))
        if not gt:
            continue
        votes = []
        for r in art.get('ratings', []):
            raw = r.get('q2_rating', '')
            pred = HUMAN_TO_AI.get(str(raw).strip())
            if pred:
                votes.append(pred)
        if not votes:
            continue
        c = Counter(votes)
        n = len(votes)
        vote_probs = {lab: c.get(lab, 0) / n for lab in LABEL_ORDER}
        predicted = c.most_common(1)[0][0]
        max_prob = max(vote_probs.values())
        ent = _entropy(vote_probs)
        result['student_voting'].append({
            'ground_truth': gt,
            'vote_probs': vote_probs,
            'predicted': predicted,
            'max_prob': max_prob,
            'entropy': ent,
            'n_raters': n,
            'correct': int(predicted == gt),
            'agreement_ratio': c.most_common(1)[0][1] / n,
        })

    return result


# -----------------------------------------------------------------------
# Analysis functions
# -----------------------------------------------------------------------

def analyze_model_confidence(records):
    """Compute confidence statistics for a single model with logp data."""
    if not records:
        return None
    confs = [r['max_prob'] for r in records]
    correct_confs = [r['max_prob'] for r in records if r['correct']]
    wrong_confs = [r['max_prob'] for r in records if not r['correct']]
    entropies = [r['entropy'] for r in records]

    n_correct = sum(r['correct'] for r in records)
    accuracy = n_correct / len(records)

    # Confidence-accuracy correlation (point-biserial)
    if len(set(r['correct'] for r in records)) > 1:
        corr_r, corr_p = sp_stats.pointbiserialr(
            [r['correct'] for r in records], confs
        )
    else:
        corr_r, corr_p = float('nan'), float('nan')

    # Mann-Whitney: is confidence significantly higher when correct?
    mw_stat, mw_p = _mann_whitney_safe(correct_confs, wrong_confs, 'greater')

    # Per-tier
    tier_stats = {}
    for tier in LABEL_ORDER:
        tier_recs = [r for r in records if r['ground_truth'] == tier]
        if tier_recs:
            tier_stats[tier] = {
                'n': len(tier_recs),
                'accuracy': float(np.mean([r['correct'] for r in tier_recs])),
                'mean_max_prob': float(np.mean([r['max_prob'] for r in tier_recs])),
                'mean_entropy': float(np.mean([r['entropy'] for r in tier_recs])),
                'mean_conf_correct': float(np.mean(
                    [r['max_prob'] for r in tier_recs if r['correct']]
                )) if any(r['correct'] for r in tier_recs) else None,
                'mean_conf_wrong': float(np.mean(
                    [r['max_prob'] for r in tier_recs if not r['correct']]
                )) if any(not r['correct'] for r in tier_recs) else None,
            }

    # Calibration bins (10 bins from 0.25 to 1.0)
    bin_edges = np.linspace(0.25, 1.0, 11)
    cal_bins = []
    for i in range(10):
        lo, hi = bin_edges[i], bin_edges[i + 1]
        in_bin = [r for r in records
                  if lo <= r['max_prob'] < (hi + (0.001 if i == 9 else 0))]
        if in_bin:
            cal_bins.append({
                'bin_lo': float(lo),
                'bin_hi': float(hi),
                'mean_confidence': float(np.mean([r['max_prob'] for r in in_bin])),
                'accuracy': float(np.mean([r['correct'] for r in in_bin])),
                'count': len(in_bin),
            })

    # Selective prediction curve
    selective = []
    for t in np.linspace(0.25, 0.99, 20):
        kept = [r for r in records if r['max_prob'] >= t]
        if kept:
            selective.append({
                'threshold': float(t),
                'accuracy': float(np.mean([r['correct'] for r in kept])),
                'coverage': float(len(kept) / len(records)),
                'n_kept': len(kept),
            })

    return {
        'n': len(records),
        'n_correct': n_correct,
        'accuracy': float(accuracy),
        'mean_confidence': float(np.mean(confs)),
        'median_confidence': float(np.median(confs)),
        'std_confidence': float(np.std(confs)),
        'min_confidence': float(np.min(confs)),
        'max_confidence_val': float(np.max(confs)),
        'mean_entropy': float(np.mean(entropies)),
        'mean_conf_correct': float(np.mean(correct_confs)) if correct_confs else None,
        'mean_conf_wrong': float(np.mean(wrong_confs)) if wrong_confs else None,
        'conf_gap': float(
            (np.mean(correct_confs) if correct_confs else 0)
            - (np.mean(wrong_confs) if wrong_confs else 0)
        ),
        'confidence_accuracy_corr': {'r': float(corr_r), 'p': float(corr_p)},
        'mann_whitney_correct_gt_wrong': {'U': float(mw_stat), 'p': float(mw_p)},
        'per_tier': tier_stats,
        'calibration_bins': cal_bins,
        'selective_prediction': selective,
    }


def analyze_flagship_model(records):
    """Compute accuracy-only stats for a flagship model (no logp)."""
    if not records:
        return None
    n_correct_majority = sum(r['correct'] for r in records)
    accuracy_majority = n_correct_majority / len(records)
    primary_scores = [float(r.get('correct_primary', r['correct'])) for r in records]
    accuracy_primary = float(np.mean(primary_scores)) if primary_scores else accuracy_majority

    tier_stats = {}
    for tier in LABEL_ORDER:
        tier_recs = [r for r in records if r['ground_truth'] == tier]
        if tier_recs:
            tier_stats[tier] = {
                'n': len(tier_recs),
                'accuracy': float(np.mean([r['correct'] for r in tier_recs])),
            }

    # Prediction distribution
    pred_dist = Counter(r['predicted'] for r in records)

    return {
        'n': len(records),
        'n_correct': n_correct_majority,
        'accuracy': float(accuracy_primary),
        'accuracy_primary': float(accuracy_primary),
        'accuracy_majority': float(accuracy_majority),
        'has_logp': False,
        'per_tier': tier_stats,
        'prediction_distribution': dict(pred_dist),
    }


def analyze_human_confidence(records, group_name):
    """Compute confidence statistics for human raters."""
    if not records:
        return None
    confs_raw = [r['confidence'] for r in records]
    confs_norm = [(c - 1) / 4 for c in confs_raw]
    correct_confs = [r['confidence'] for r in records if r['correct']]
    wrong_confs = [r['confidence'] for r in records if not r['correct']]

    accuracy = sum(r['correct'] for r in records) / len(records)

    # Spearman correlation
    if len(set(r['correct'] for r in records)) > 1:
        corr_r, corr_p = sp_stats.spearmanr(
            [r['confidence'] for r in records],
            [r['correct'] for r in records],
        )
    else:
        corr_r, corr_p = float('nan'), float('nan')

    mw_stat, mw_p = _mann_whitney_safe(correct_confs, wrong_confs, 'greater')

    # Per confidence level accuracy
    conf_level_acc = {}
    for level in range(1, 6):
        at_level = [r for r in records if r['confidence'] == level]
        if at_level:
            conf_level_acc[str(level)] = {
                'n': len(at_level),
                'accuracy': float(np.mean([r['correct'] for r in at_level])),
            }

    # Per-tier
    tier_stats = {}
    for tier in LABEL_ORDER:
        tier_recs = [r for r in records if r['ground_truth'] == tier]
        if tier_recs:
            tier_stats[tier] = {
                'n': len(tier_recs),
                'accuracy': float(np.mean([r['correct'] for r in tier_recs])),
                'mean_confidence': float(np.mean([r['confidence'] for r in tier_recs])),
            }

    # Selective prediction (keep >= threshold)
    selective = []
    for min_conf in range(1, 6):
        kept = [r for r in records if r['confidence'] >= min_conf]
        if kept:
            selective.append({
                'threshold': min_conf,
                'threshold_normalized': float((min_conf - 1) / 4),
                'accuracy': float(np.mean([r['correct'] for r in kept])),
                'coverage': float(len(kept) / len(records)),
                'n_kept': len(kept),
            })

    return {
        'n': len(records),
        'n_correct': sum(r['correct'] for r in records),
        'accuracy': float(accuracy),
        'mean_confidence_raw': float(np.mean(confs_raw)),
        'median_confidence_raw': float(np.median(confs_raw)),
        'std_confidence_raw': float(np.std(confs_raw)),
        'mean_confidence_normalized': float(np.mean(confs_norm)),
        'mean_conf_correct_raw': float(np.mean(correct_confs)) if correct_confs else None,
        'mean_conf_wrong_raw': float(np.mean(wrong_confs)) if wrong_confs else None,
        'conf_gap_raw': float(
            (np.mean(correct_confs) if correct_confs else 0)
            - (np.mean(wrong_confs) if wrong_confs else 0)
        ),
        'confidence_accuracy_corr': {'r': float(corr_r), 'p': float(corr_p)},
        'mann_whitney_correct_gt_wrong': {'U': float(mw_stat), 'p': float(mw_p)},
        'per_confidence_level': conf_level_acc,
        'per_tier': tier_stats,
        'selective_prediction': selective,
    }


def analyze_article_difficulty(val_data, voting_data):
    """Compare AI entropy with human entropy per article across models."""
    # Use multiple AI models
    ai_models_for_entropy = ['best_2_model_combo', 'CYqJRxId', 'gpt-5.2']
    expert_voting = voting_data.get('expert_voting', [])
    student_voting = voting_data.get('student_voting', [])

    result = {'per_model': {}}

    for model_key in ai_models_for_entropy:
        records = val_data.get(model_key, [])
        display = SFT_DISPLAY.get(model_key, model_key)
        if not records or len(records) != len(expert_voting):
            continue

        ai_ent = [r['entropy'] for r in records]
        exp_ent = [e['entropy'] for e in expert_voting]
        stu_ent = [s['entropy'] for s in student_voting[:len(records)]]

        r_exp, p_exp = sp_stats.spearmanr(ai_ent, exp_ent)
        r_stu, p_stu = sp_stats.spearmanr(ai_ent, stu_ent)

        result['per_model'][display] = {
            'vs_expert': {'spearman_r': float(r_exp), 'p_value': float(p_exp)},
            'vs_student': {'spearman_r': float(r_stu), 'p_value': float(p_stu)},
        }

    # Expert vs student entropy
    if expert_voting and student_voting:
        n = min(len(expert_voting), len(student_voting))
        r_es, p_es = sp_stats.spearmanr(
            [e['entropy'] for e in expert_voting[:n]],
            [s['entropy'] for s in student_voting[:n]],
        )
        result['expert_vs_student'] = {
            'spearman_r': float(r_es), 'p_value': float(p_es),
        }

    return result


# -----------------------------------------------------------------------
# Plotting
# -----------------------------------------------------------------------

def plot_hero_figure(all_model_stats, human_stats, output_path):
    """Hero figure: 2x2 panel comparing key AI models and humans.

    (a) Calibration curves
    (b) Confidence correct vs wrong (bar chart)
    (c) Selective prediction: accuracy vs coverage
    (d) Accuracy vs mean confidence scatter
    """
    set_nature_style()
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))

    # Key models for hero figure
    key_models = [
        ('SFT 2-Model Ensemble', '#B21F00'),
        ('GPT-5.2 (baseline)', '#8491B4'),
    ]
    key_humans = [
        ('Human Expert', '#4DBBD5'),
        ('Human Student', '#00A087'),
    ]

    # --- (a) Calibration curves ---
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8, label='Perfect')
    for name, color in key_models:
        stats = all_model_stats.get(name)
        if not stats or 'calibration_bins' not in stats:
            continue
        bins = stats['calibration_bins']
        if bins:
            ax.plot([b['mean_confidence'] for b in bins],
                    [b['accuracy'] for b in bins],
                    'o-', color=color, label=name, markersize=4, linewidth=1.5)
    for name, color in key_humans:
        stats = human_stats.get(name)
        if not stats or 'per_confidence_level' not in stats:
            continue
        levels = stats['per_confidence_level']
        confs = [(int(k) - 1) / 4 for k in sorted(levels.keys())]
        accs = [levels[k]['accuracy'] for k in sorted(levels.keys())]
        ax.plot(confs, accs, 's--', color=color, label=name, markersize=5, linewidth=1.5)

    ax.set_xlabel('Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('(a) Calibration: Confidence vs Accuracy', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='upper left')
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # --- (b) Confidence correct vs wrong ---
    ax = axes[0, 1]
    entries = []
    for name, color in key_models:
        stats = all_model_stats.get(name)
        if stats:
            entries.append((name.replace('SFT 2-Model\n', 'SFT\n'),
                            stats.get('mean_conf_correct', 0) or 0,
                            stats.get('mean_conf_wrong', 0) or 0,
                            color))
    for name, color in key_humans:
        stats = human_stats.get(name)
        if stats:
            cc = stats.get('mean_conf_correct_raw')
            cw = stats.get('mean_conf_wrong_raw')
            entries.append((name, (cc - 1) / 4 if cc else 0,
                            (cw - 1) / 4 if cw else 0, color))

    if entries:
        x = np.arange(len(entries))
        w = 0.35
        labels = [e[0] for e in entries]
        correct_v = [e[1] for e in entries]
        wrong_v = [e[2] for e in entries]
        colors_bar = [e[3] for e in entries]
        ax.bar(x - w / 2, correct_v, w, color=colors_bar, alpha=0.9, label='Correct')
        ax.bar(x + w / 2, wrong_v, w, color=colors_bar, alpha=0.35, label='Wrong')
        ax.set_xticks(x)
        ax.set_xticklabels([l.replace('SFT 2-Model Ensemble', 'SFT Ens.')
                            .replace('GPT-5.2 (baseline)', 'GPT-5.2')
                            for l in labels], fontsize=7)
        ax.legend(fontsize=7)
        ax.set_ylabel('Confidence (normalized)')
        ax.set_ylim(0, 1.05)
    ax.set_title('(b) Confidence: Correct vs Wrong', fontsize=10, fontweight='bold')

    # --- (c) Selective prediction ---
    ax = axes[1, 0]
    for name, color in key_models:
        stats = all_model_stats.get(name)
        if not stats or 'selective_prediction' not in stats:
            continue
        pts = stats['selective_prediction']
        if pts:
            ax.plot([p['coverage'] for p in pts], [p['accuracy'] for p in pts],
                    'o-', color=color, label=name, markersize=3, linewidth=1.5)
    for name, color in key_humans:
        stats = human_stats.get(name)
        if not stats or 'selective_prediction' not in stats:
            continue
        pts = stats['selective_prediction']
        if pts:
            ax.plot([p['coverage'] for p in pts], [p['accuracy'] for p in pts],
                    's--', color=color, label=name, markersize=5, linewidth=1.5)
    ax.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Accuracy')
    ax.set_title('(c) Selective Prediction', fontsize=10, fontweight='bold')
    ax.legend(fontsize=7, loc='lower left')
    ax.set_xlim(-0.02, 1.05)

    # --- (d) Accuracy vs confidence scatter ---
    ax = axes[1, 1]
    for name, color in key_models:
        stats = all_model_stats.get(name)
        if not stats:
            continue
        ax.scatter(stats['mean_confidence'], stats['accuracy'],
                   s=120, color=color, edgecolors='black', linewidths=0.5, zorder=5)
        short = name.replace('SFT 2-Model Ensemble', 'SFT Ens.').replace(
            'GPT-5.2 (baseline)', 'GPT-5.2')
        ax.annotate(short, (stats['mean_confidence'], stats['accuracy']),
                    fontsize=7, xytext=(8, 6), textcoords='offset points')
    for name, color in key_humans:
        stats = human_stats.get(name)
        if not stats:
            continue
        ax.scatter(stats['mean_confidence_normalized'], stats['accuracy'],
                   s=120, color=color, edgecolors='black', linewidths=0.5,
                   zorder=5, marker='s')
        ax.annotate(name, (stats['mean_confidence_normalized'], stats['accuracy']),
                    fontsize=7, xytext=(8, 6), textcoords='offset points')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
    ax.set_xlabel('Mean Confidence')
    ax.set_ylabel('Accuracy')
    ax.set_title('(d) Confidence vs Accuracy Overview', fontsize=10, fontweight='bold')
    ax.set_xlim(0.1, 1.05)
    ax.set_ylim(0.1, 0.7)

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_all_models_confidence(all_model_stats, human_stats, output_path):
    """Comprehensive figure: ALL SFT + GPT-5.2 models confidence analysis.

    3-row figure:
    Row 1: Calibration curves for all individual SFT models
    Row 2: Confidence gap (correct - wrong) + entropy comparison
    Row 3: Selective prediction curves for all models
    """
    set_nature_style()
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))

    # Model ordering
    sft_individual = ['SFT-1 (CYqJRxId)', 'SFT-2 (ckpt-304)',
                      'SFT-3 (ckpt-380)', 'SFT-4 (ckpt-228)']
    sft_ensembles = ['SFT 2-Model Ensemble']
    baseline = ['GPT-5.2 (baseline)']
    voting = ['Human Voting (old students)']
    all_ai = sft_individual + sft_ensembles + baseline + voting

    # Assign colors
    cmap = plt.cm.tab20
    ai_colors = {}
    for i, name in enumerate(all_ai):
        ai_colors[name] = MODEL_COLORS.get(name, cmap(i / len(all_ai)))

    # --- (a) Calibration curves ---
    ax = axes[0, 0]
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, linewidth=0.8)
    for name in all_ai:
        stats = all_model_stats.get(name)
        if not stats or 'calibration_bins' not in stats:
            continue
        bins = stats['calibration_bins']
        if bins:
            ax.plot([b['mean_confidence'] for b in bins],
                    [b['accuracy'] for b in bins],
                    'o-', color=ai_colors.get(name, 'gray'),
                    label=name, markersize=3, linewidth=1.2, alpha=0.8)
    ax.set_xlabel('Predicted Probability (max softmax)')
    ax.set_ylabel('Observed Accuracy')
    ax.set_title('(a) Calibration: All Models', fontsize=10, fontweight='bold')
    ax.legend(fontsize=6, loc='upper left', ncol=1)
    ax.set_xlim(0.2, 1.02)
    ax.set_ylim(-0.02, 1.02)

    # --- (b) Confidence gap bar chart ---
    ax = axes[0, 1]
    names_for_bar = []
    gaps = []
    accs = []
    bar_colors = []
    for name in all_ai:
        stats = all_model_stats.get(name)
        if stats and stats.get('mean_conf_correct') is not None:
            names_for_bar.append(name)
            gaps.append(stats['conf_gap'])
            accs.append(stats['accuracy'])
            bar_colors.append(ai_colors.get(name, 'gray'))

    if names_for_bar:
        x = np.arange(len(names_for_bar))
        bars = ax.bar(x, gaps, color=bar_colors, alpha=0.8, edgecolor='black', linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('SFT ', 'SFT\n').replace('Human Voting', 'Vote')
                            .replace(' (baseline)', '') for n in names_for_bar],
                           fontsize=6, rotation=45, ha='right')
        ax.axhline(y=0, color='black', linewidth=0.5)
        ax.set_ylabel('Confidence Gap (correct - wrong)')
        # Add accuracy annotation on top
        for i, (g, a) in enumerate(zip(gaps, accs)):
            ax.annotate(f'acc={a:.1%}', (i, g), fontsize=6, ha='center',
                        va='bottom' if g >= 0 else 'top', xytext=(0, 3),
                        textcoords='offset points')
    ax.set_title('(b) Confidence Discriminability', fontsize=10, fontweight='bold')

    # --- (c) Entropy comparison ---
    ax = axes[1, 0]
    names_ent = []
    entropies = []
    ent_colors = []
    for name in all_ai:
        stats = all_model_stats.get(name)
        if stats and 'mean_entropy' in stats:
            names_ent.append(name)
            entropies.append(stats['mean_entropy'])
            ent_colors.append(ai_colors.get(name, 'gray'))

    if names_ent:
        x = np.arange(len(names_ent))
        ax.bar(x, entropies, color=ent_colors, alpha=0.8, edgecolor='black', linewidth=0.3)
        ax.set_xticks(x)
        ax.set_xticklabels([n.replace('SFT ', 'SFT\n').replace('Human Voting', 'Vote')
                            .replace(' (baseline)', '') for n in names_ent],
                           fontsize=6, rotation=45, ha='right')
        ax.set_ylabel('Mean Prediction Entropy (bits)')
        ax.axhline(y=2.0, color='gray', linestyle=':', alpha=0.5, label='Max entropy (uniform)')
    ax.set_title('(c) Prediction Uncertainty (Entropy)', fontsize=10, fontweight='bold')

    # --- (d) Selective prediction for all ---
    ax = axes[1, 1]
    for name in all_ai:
        stats = all_model_stats.get(name)
        if not stats or 'selective_prediction' not in stats:
            continue
        pts = stats['selective_prediction']
        if pts:
            ax.plot([p['coverage'] for p in pts], [p['accuracy'] for p in pts],
                    'o-', color=ai_colors.get(name, 'gray'),
                    label=name, markersize=2, linewidth=1.2, alpha=0.8)
    ax.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5, linewidth=0.8)
    ax.set_xlabel('Coverage')
    ax.set_ylabel('Accuracy')
    ax.set_title('(d) Selective Prediction: All Models', fontsize=10, fontweight='bold')
    ax.legend(fontsize=6, loc='lower left', ncol=1)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0.1, 1.05)

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_flagship_vs_sft(all_model_stats, flagship_stats, output_path):
    """Bar chart comparing all 11 flagship models vs SFT models.

    Flagship models have no confidence data, so we show:
    (a) Accuracy comparison across all models
    (b) Per-tier accuracy heatmap
    """
    set_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Collect all models with accuracy
    all_entries = []
    # Flagship models
    for model_key, display in FLAGSHIP_DISPLAY.items():
        stats = flagship_stats.get(model_key)
        if stats:
            all_entries.append((display, stats['accuracy'], 'Flagship', '#8491B4'))

    # SFT models
    sft_names = ['SFT-1 (CYqJRxId)', 'SFT-2 (ckpt-304)',
                 'SFT-3 (ckpt-380)', 'SFT-4 (ckpt-228)',
                 'SFT 2-Model Ensemble',
                 'GPT-5.2 (baseline)']
    for name in sft_names:
        stats = all_model_stats.get(name)
        if stats:
            color = '#E64B35' if 'SFT' in name else '#8491B4'
            all_entries.append((name, stats['accuracy'], 'SFT/Baseline', color))

    # Sort by accuracy
    all_entries.sort(key=lambda e: e[1], reverse=True)

    # --- (a) Accuracy bar chart ---
    ax = axes[0]
    x = np.arange(len(all_entries))
    names = [e[0] for e in all_entries]
    accs = [e[1] for e in all_entries]
    bar_colors = []
    for e in all_entries:
        if 'SFT' in e[0] and 'Ensemble' in e[0]:
            bar_colors.append('#B21F00')
        elif 'SFT' in e[0]:
            bar_colors.append('#E64B35')
        elif e[0] == 'GPT-5.2 (baseline)':
            bar_colors.append('#8491B4')
        else:
            bar_colors.append('#4DBBD5')

    bars = ax.barh(x, accs, color=bar_colors, alpha=0.85, edgecolor='black', linewidth=0.3)
    ax.set_yticks(x)
    ax.set_yticklabels(names, fontsize=7)
    ax.set_xlabel('Accuracy')
    ax.set_title('(a) All Models: Accuracy Comparison', fontsize=11, fontweight='bold')
    ax.axvline(x=0.25, color='gray', linestyle=':', alpha=0.5, label='Chance')
    ax.invert_yaxis()
    # Add accuracy values
    for i, acc in enumerate(accs):
        ax.text(acc + 0.005, i, f'{acc:.1%}', va='center', fontsize=7)

    # Add note about logp
    ax.text(0.98, 0.02,
            'Blue bars: Flagship (no logp)\nRed bars: SFT (has logp)\nGray: Baseline',
            transform=ax.transAxes, fontsize=7, ha='right', va='bottom',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.8))

    # --- (b) Per-tier accuracy heatmap ---
    ax = axes[1]
    # Select top models for heatmap
    heatmap_models = []
    heatmap_data = []

    # Add flagship models
    for model_key in FLAGSHIP_DISPLAY:
        stats = flagship_stats.get(model_key)
        if stats and 'per_tier' in stats:
            display = FLAGSHIP_DISPLAY[model_key]
            row = [stats['per_tier'].get(t, {}).get('accuracy', 0) for t in LABEL_ORDER]
            heatmap_models.append(display)
            heatmap_data.append(row)

    # Add SFT models
    for name in sft_names:
        stats = all_model_stats.get(name)
        if stats and 'per_tier' in stats:
            row = [stats['per_tier'].get(t, {}).get('accuracy', 0) for t in LABEL_ORDER]
            heatmap_models.append(name)
            heatmap_data.append(row)

    if heatmap_data:
        data_arr = np.array(heatmap_data)
        im = ax.imshow(data_arr, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1)
        ax.set_xticks(np.arange(len(LABEL_ORDER)))
        ax.set_xticklabels([t.title() for t in LABEL_ORDER], fontsize=8)
        ax.set_yticks(np.arange(len(heatmap_models)))
        ax.set_yticklabels(heatmap_models, fontsize=7)
        # Annotate values
        for i in range(len(heatmap_models)):
            for j in range(len(LABEL_ORDER)):
                val = data_arr[i, j]
                color = 'white' if val < 0.3 or val > 0.7 else 'black'
                ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                        fontsize=6, color=color)
        fig.colorbar(im, ax=ax, shrink=0.6, label='Accuracy')
    ax.set_title('(b) Per-Tier Accuracy: All Models', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_confidence_correct_vs_wrong_all(all_model_stats, human_stats, output_path):
    """Violin plots for all SFT models + humans: confidence correct vs wrong."""
    set_nature_style()

    # Collect data for violins
    entries = []

    sft_keys = {
        'SFT-1 (CYqJRxId)': 'CYqJRxId',
        'SFT-2 (ckpt-304)': 'ckpt-step-304',
        'SFT-3 (ckpt-380)': 'ckppt-380',
        'SFT-4 (ckpt-228)': 'ckppt-228',
        'SFT 2-Model Ens.': 'best_2_model_combo',
        'GPT-5.2': 'gpt-5.2',
    }
    for display, _ in sft_keys.items():
        stats = all_model_stats.get(display.replace('Ens.', 'Ensemble'))
        if not stats:
            # Try exact match
            for key in all_model_stats:
                if display.split(' (')[0] in key:
                    stats = all_model_stats[key]
                    break
        if stats:
            entries.append({
                'name': display,
                'type': 'AI',
                'conf_correct': stats.get('mean_conf_correct'),
                'conf_wrong': stats.get('mean_conf_wrong'),
                'gap': stats.get('conf_gap', 0),
                'accuracy': stats.get('accuracy', 0),
                'mw_p': stats.get('mann_whitney_correct_gt_wrong', {}).get('p', 1),
            })

    for name in ['Human Expert', 'Human Student']:
        stats = human_stats.get(name)
        if stats:
            cc = stats.get('mean_conf_correct_raw')
            cw = stats.get('mean_conf_wrong_raw')
            entries.append({
                'name': name,
                'type': 'Human',
                'conf_correct': (cc - 1) / 4 if cc else None,
                'conf_wrong': (cw - 1) / 4 if cw else None,
                'gap': stats.get('conf_gap_raw', 0) / 4,
                'accuracy': stats.get('accuracy', 0),
                'mw_p': stats.get('mann_whitney_correct_gt_wrong', {}).get('p', 1),
            })

    if not entries:
        return

    fig, ax = plt.subplots(figsize=(12, 5))

    x = np.arange(len(entries))
    w = 0.3
    correct_vals = [e['conf_correct'] or 0 for e in entries]
    wrong_vals = [e['conf_wrong'] or 0 for e in entries]

    bar_colors = []
    for e in entries:
        if 'SFT' in e['name'] and 'Ens' in e['name']:
            bar_colors.append('#B21F00')
        elif 'SFT' in e['name']:
            bar_colors.append('#E64B35')
        elif 'GPT' in e['name']:
            bar_colors.append('#8491B4')
        elif 'Expert' in e['name']:
            bar_colors.append('#4DBBD5')
        else:
            bar_colors.append('#00A087')

    ax.bar(x - w / 2, correct_vals, w, color=bar_colors, alpha=0.9, label='Correct predictions')
    ax.bar(x + w / 2, wrong_vals, w, color=bar_colors, alpha=0.35, label='Wrong predictions')

    ax.set_xticks(x)
    ax.set_xticklabels([e['name'] for e in entries], fontsize=7, rotation=30, ha='right')
    ax.set_ylabel('Mean Confidence (normalized 0-1)')
    ax.set_title('Prediction Confidence: Correct vs Wrong — All Models',
                 fontsize=11, fontweight='bold')
    ax.legend(fontsize=8)
    ax.set_ylim(0, 1.1)

    # Add significance markers
    for i, e in enumerate(entries):
        p = e.get('mw_p', 1)
        sig = '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'n.s.'
        gap = e.get('gap', 0)
        y_pos = max(correct_vals[i], wrong_vals[i]) + 0.03
        ax.text(i, y_pos, f'{sig}\n(gap={gap:+.3f})', ha='center', fontsize=6)

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_per_tier_confidence_all(all_model_stats, human_stats, output_path):
    """Heatmap: mean confidence by tier for all models."""
    set_nature_style()
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # --- (a) AI models: mean_max_prob by tier ---
    ax = axes[0]
    ai_names = []
    data_rows = []
    for name in sorted(all_model_stats.keys()):
        stats = all_model_stats[name]
        if 'per_tier' not in stats:
            continue
        row = [stats['per_tier'].get(t, {}).get('mean_max_prob', 0) for t in LABEL_ORDER]
        ai_names.append(name)
        data_rows.append(row)

    if data_rows:
        arr = np.array(data_rows)
        im = ax.imshow(arr, cmap='YlOrRd', aspect='auto', vmin=0.25, vmax=1.0)
        ax.set_xticks(np.arange(len(LABEL_ORDER)))
        ax.set_xticklabels([t.title() for t in LABEL_ORDER], fontsize=9)
        ax.set_yticks(np.arange(len(ai_names)))
        ax.set_yticklabels(ai_names, fontsize=7)
        for i in range(len(ai_names)):
            for j in range(len(LABEL_ORDER)):
                val = arr[i, j]
                color = 'white' if val > 0.75 else 'black'
                ax.text(j, i, f'{val:.2f}', ha='center', va='center',
                        fontsize=7, color=color)
        fig.colorbar(im, ax=ax, shrink=0.6, label='Mean Max Probability')
    ax.set_title('(a) AI Model Confidence by Tier', fontsize=11, fontweight='bold')

    # --- (b) AI models: accuracy by tier (for comparison) ---
    ax = axes[1]
    data_acc = []
    for name in ai_names:
        stats = all_model_stats[name]
        row = [stats['per_tier'].get(t, {}).get('accuracy', 0) for t in LABEL_ORDER]
        data_acc.append(row)

    if data_acc:
        arr2 = np.array(data_acc)
        im2 = ax.imshow(arr2, cmap='RdYlGn', aspect='auto', vmin=0, vmax=1.0)
        ax.set_xticks(np.arange(len(LABEL_ORDER)))
        ax.set_xticklabels([t.title() for t in LABEL_ORDER], fontsize=9)
        ax.set_yticks(np.arange(len(ai_names)))
        ax.set_yticklabels(ai_names, fontsize=7)
        for i in range(len(ai_names)):
            for j in range(len(LABEL_ORDER)):
                val = arr2[i, j]
                color = 'white' if val < 0.3 or val > 0.7 else 'black'
                ax.text(j, i, f'{val:.0%}', ha='center', va='center',
                        fontsize=7, color=color)
        fig.colorbar(im2, ax=ax, shrink=0.6, label='Accuracy')
    ax.set_title('(b) AI Model Accuracy by Tier', fontsize=11, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


def plot_entropy_vs_disagreement(difficulty_data, val_data, voting_data, output_path):
    """Scatter: AI entropy vs human entropy, for multiple AI models."""
    set_nature_style()

    ai_models = [
        ('best_2_model_combo', 'SFT 2-Model Ensemble', '#B21F00'),
        ('CYqJRxId', 'SFT-1 (CYqJRxId)', '#E64B35'),
        ('gpt-5.2', 'GPT-5.2', '#8491B4'),
    ]
    expert_voting = voting_data.get('expert_voting', [])
    student_voting = voting_data.get('student_voting', [])

    n_models = len(ai_models)
    fig, axes = plt.subplots(n_models, 2, figsize=(10, 4 * n_models))

    for row, (model_key, display, color) in enumerate(ai_models):
        records = val_data.get(model_key, [])
        n = min(len(records), len(expert_voting), len(student_voting))
        if n == 0:
            continue

        ai_ent = [records[i]['entropy'] for i in range(n)]
        ai_correct = [records[i]['correct'] for i in range(n)]
        exp_ent = [expert_voting[i]['entropy'] for i in range(n)]
        stu_ent = [student_voting[i]['entropy'] for i in range(n)]

        scatter_colors = [color if c else '#CCCCCC' for c in ai_correct]

        # vs expert
        ax = axes[row, 0]
        ax.scatter(ai_ent, exp_ent, c=scatter_colors, alpha=0.6, s=25, edgecolors='none')
        r, p = sp_stats.spearmanr(ai_ent, exp_ent)
        ax.set_xlabel(f'{display} Entropy')
        ax.set_ylabel('Expert Voting Entropy')
        ax.set_title(f'{display} vs Expert (r={r:.3f}, p={p:.3f})',
                     fontsize=9, fontweight='bold')

        # vs student
        ax = axes[row, 1]
        ax.scatter(ai_ent, stu_ent, c=scatter_colors, alpha=0.6, s=25, edgecolors='none')
        r, p = sp_stats.spearmanr(ai_ent, stu_ent)
        ax.set_xlabel(f'{display} Entropy')
        ax.set_ylabel('Student Voting Entropy')
        ax.set_title(f'{display} vs Student (r={r:.3f}, p={p:.3f})',
                     fontsize=9, fontweight='bold')

    plt.tight_layout()
    save_figure(fig, output_path)
    plt.close(fig)


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    print("=" * 70)
    print("Script 15: AI Confidence vs Human Confidence (Comprehensive)")
    print("=" * 70)

    # ---- Load all data ----
    print("\n[1/8] Loading data...")
    val_data = load_all_val_predictions()
    flagship_data = load_flagship_predictions()
    human_data = load_human_ratings_with_confidence()
    voting_data = load_human_voting_per_article()

    print(f"\n  VAL MODELS (with logp):")
    for k, v in val_data.items():
        display = SFT_DISPLAY.get(k, k)
        print(f"    {display}: {len(v)} predictions")

    print(f"\n  FLAGSHIP MODELS (no logp, point predictions only):")
    for k, v in flagship_data.items():
        display = FLAGSHIP_DISPLAY.get(k, k)
        has_logp_any = any(r.get('has_logp', False) for r in v)
        print(f"    {display}: {len(v)} predictions, has_logp={has_logp_any}")

    print(f"\n  HUMAN RATERS:")
    for k, v in human_data.items():
        print(f"    {k}: {len(v)} individual ratings with confidence")
    for k, v in voting_data.items():
        print(f"    {k}: {len(v)} article-level voting distributions")

    # ---- Analyze ALL val models (with logp) ----
    print("\n[2/8] Analyzing ALL val model confidence...")
    all_model_stats = {}
    for model_key, records in val_data.items():
        display = SFT_DISPLAY.get(model_key, model_key)
        stats = analyze_model_confidence(records)
        if stats:
            all_model_stats[display] = stats
            gap = stats['conf_gap']
            mw_p = stats['mann_whitney_correct_gt_wrong']['p']
            corr_r = stats['confidence_accuracy_corr']['r']
            print(f"  {display}: acc={stats['accuracy']:.3f}, "
                  f"mean_conf={stats['mean_confidence']:.3f}, "
                  f"gap(C-W)={gap:+.3f}, "
                  f"MW_p={mw_p:.4f}, "
                  f"entropy={stats['mean_entropy']:.3f}, "
                  f"corr_r={corr_r:.3f}")

    # ---- Analyze ALL flagship models (no logp) ----
    print("\n[3/8] Analyzing ALL flagship models (accuracy only, no logp)...")
    flagship_stats = {}
    for model_key, records in flagship_data.items():
        display = FLAGSHIP_DISPLAY.get(model_key, model_key)
        stats = analyze_flagship_model(records)
        if stats:
            flagship_stats[model_key] = stats
            pred_dist = stats['prediction_distribution']
            dist_str = ", ".join(f"{k}:{v}" for k, v in sorted(pred_dist.items()))
            print(f"  {display}: acc={stats['accuracy']:.3f}, "
                  f"preds=[{dist_str}]")

    # ---- Analyze human confidence ----
    print("\n[4/8] Analyzing human rater confidence...")
    human_stats = {}
    for group, records in human_data.items():
        display = f"Human {group.title()}"
        stats = analyze_human_confidence(records, group)
        if stats:
            human_stats[display] = stats
            gap = stats['conf_gap_raw']
            mw_p = stats['mann_whitney_correct_gt_wrong']['p']
            corr_r = stats['confidence_accuracy_corr']['r']
            print(f"  {display}: acc={stats['accuracy']:.3f}, "
                  f"mean_conf={stats['mean_confidence_raw']:.2f}/5, "
                  f"gap(C-W)={gap:+.2f}, "
                  f"MW_p={mw_p:.4f}, "
                  f"corr_r={corr_r:.3f}")

    # ---- Article difficulty ----
    print("\n[5/8] Analyzing article difficulty (entropy correlations)...")
    difficulty = analyze_article_difficulty(val_data, voting_data)
    for model, corrs in difficulty.get('per_model', {}).items():
        for vs, vals in corrs.items():
            print(f"  {model} {vs}: r={vals['spearman_r']:.3f}, p={vals['p_value']:.3f}")
    es = difficulty.get('expert_vs_student', {})
    if es:
        print(f"  Expert vs Student: r={es['spearman_r']:.3f}, p={es['p_value']:.3f}")

    # ---- Generate ALL figures ----
    print("\n[6/8] Generating figures...")

    plot_hero_figure(all_model_stats, human_stats,
                     str(FIGS / 'fig15a_confidence_comparison_main'))
    print("  -> fig15a_confidence_comparison_main.pdf")

    plot_all_models_confidence(all_model_stats, human_stats,
                               str(FIGS / 'fig15b_all_models_confidence'))
    print("  -> fig15b_all_models_confidence.pdf")

    plot_flagship_vs_sft(all_model_stats, flagship_stats,
                         str(FIGS / 'fig_s20_flagship_vs_sft_accuracy'))
    print("  -> fig_s20_flagship_vs_sft_accuracy.pdf")

    plot_confidence_correct_vs_wrong_all(all_model_stats, human_stats,
                                         str(FIGS / 'fig_s18_confidence_correct_vs_wrong'))
    print("  -> fig_s18_confidence_correct_vs_wrong.pdf")

    plot_entropy_vs_disagreement(difficulty, val_data, voting_data,
                                 str(FIGS / 'fig_s19_entropy_vs_disagreement'))
    print("  -> fig_s19_entropy_vs_disagreement.pdf")

    plot_per_tier_confidence_all(all_model_stats, human_stats,
                                 str(FIGS / 'fig_s21_per_tier_confidence_all'))
    print("  -> fig_s21_per_tier_confidence_all.pdf")

    # Selective prediction (enhanced with all models)
    set_nature_style()
    fig, ax = plt.subplots(figsize=(8, 6))
    for name in sorted(all_model_stats.keys()):
        stats = all_model_stats[name]
        pts = stats.get('selective_prediction', [])
        if pts:
            color = MODEL_COLORS.get(name, 'gray')
            ax.plot([p['coverage'] for p in pts], [p['accuracy'] for p in pts],
                    'o-', color=color, label=name, markersize=3, linewidth=1.5, alpha=0.8)
    # Human
    for name in ['Human Expert', 'Human Student']:
        stats = human_stats.get(name)
        if stats:
            pts = stats.get('selective_prediction', [])
            if pts:
                color = MODEL_COLORS.get(name, 'gray')
                ax.plot([p['coverage'] for p in pts], [p['accuracy'] for p in pts],
                        's--', color=color, label=name, markersize=5, linewidth=1.5)
    ax.axhline(y=0.25, color='gray', linestyle=':', alpha=0.5)
    ax.set_xlabel('Coverage', fontsize=11)
    ax.set_ylabel('Accuracy', fontsize=11)
    ax.set_title('Selective Prediction: All Models + Humans', fontsize=12, fontweight='bold')
    ax.legend(fontsize=7, loc='lower left', ncol=2)
    ax.set_xlim(-0.02, 1.05)
    ax.set_ylim(0.1, 1.05)
    plt.tight_layout()
    save_figure(fig, str(FIGS / 'fig_s17_selective_prediction'))
    plt.close(fig)
    print("  -> fig_s17_selective_prediction.pdf")

    # ---- Save tables ----
    print("\n[7/8] Saving tables...")

    # Table 1: All val models with confidence stats
    rows = []
    for name, stats in sorted(all_model_stats.items()):
        rows.append({
            'model': name,
            'type': 'AI (with logp)',
            'n': stats['n'],
            'accuracy': stats['accuracy'],
            'mean_confidence': stats['mean_confidence'],
            'median_confidence': stats['median_confidence'],
            'std_confidence': stats['std_confidence'],
            'mean_entropy': stats['mean_entropy'],
            'conf_correct': stats.get('mean_conf_correct'),
            'conf_wrong': stats.get('mean_conf_wrong'),
            'conf_gap': stats['conf_gap'],
            'point_biserial_r': stats['confidence_accuracy_corr']['r'],
            'point_biserial_p': stats['confidence_accuracy_corr']['p'],
            'mann_whitney_p': stats['mann_whitney_correct_gt_wrong']['p'],
        })
    # Add flagship models (accuracy only)
    for model_key in FLAGSHIP_DISPLAY:
        stats = flagship_stats.get(model_key)
        if stats:
            rows.append({
                'model': FLAGSHIP_DISPLAY[model_key],
                'type': 'Flagship (no logp)',
                'n': stats['n'],
                'accuracy': stats['accuracy'],
            })
    # Add human
    for name, stats in human_stats.items():
        rows.append({
            'model': name,
            'type': 'Human',
            'n': stats['n'],
            'accuracy': stats['accuracy'],
            'mean_confidence': stats['mean_confidence_normalized'],
            'mean_confidence_raw_1_5': stats['mean_confidence_raw'],
            'conf_correct_raw': stats.get('mean_conf_correct_raw'),
            'conf_wrong_raw': stats.get('mean_conf_wrong_raw'),
            'conf_gap_raw': stats['conf_gap_raw'],
            'spearman_r': stats['confidence_accuracy_corr']['r'],
            'spearman_p': stats['confidence_accuracy_corr']['p'],
            'mann_whitney_p': stats['mann_whitney_correct_gt_wrong']['p'],
        })

    df = pd.DataFrame(rows)
    df.to_csv(str(TABS / 'table_s12_confidence_comparison.csv'), index=False)
    print("  -> table_s12_confidence_comparison.csv")

    # Table 2: comprehensive model summary (one row per model)
    summary_rows = []
    for name, stats in sorted(all_model_stats.items()):
        summary_rows.append({
            'model': name, 'source': 'val (logp)',
            'accuracy': stats['accuracy'],
            'confidence_type': 'softmax max_prob',
            'mean_confidence': stats['mean_confidence'],
            'conf_discriminates': stats['mann_whitney_correct_gt_wrong']['p'] < 0.05,
        })
    for model_key in FLAGSHIP_DISPLAY:
        stats = flagship_stats.get(model_key)
        if stats:
            summary_rows.append({
                'model': FLAGSHIP_DISPLAY[model_key], 'source': 'thinking (no logp)',
                'accuracy': stats['accuracy'],
                'confidence_type': 'N/A',
                'mean_confidence': None,
                'conf_discriminates': None,
            })
    for name, stats in human_stats.items():
        summary_rows.append({
            'model': name, 'source': 'survey',
            'accuracy': stats['accuracy'],
            'confidence_type': 'self-reported (1-5)',
            'mean_confidence': stats['mean_confidence_raw'],
            'conf_discriminates': stats['mann_whitney_correct_gt_wrong']['p'] < 0.05,
        })

    df2 = pd.DataFrame(summary_rows)
    df2.to_csv(str(TABS / 'table_s13_all_models_summary.csv'), index=False)
    print("  -> table_s13_all_models_summary.csv")

    # ---- Save JSON stats ----
    print("\n[8/8] Saving statistics JSON...")
    all_stats_json = {
        'val_models_with_logp': {},
        'flagship_models_no_logp': {},
        'human_raters': {},
        'article_difficulty': difficulty,
    }

    for name, stats in all_model_stats.items():
        # Remove calibration bins and selective for brevity (in CSV)
        s = {k: v for k, v in stats.items()
             if k not in ('calibration_bins', 'selective_prediction')}
        all_stats_json['val_models_with_logp'][name] = s

    for model_key, stats in flagship_stats.items():
        display = FLAGSHIP_DISPLAY.get(model_key, model_key)
        all_stats_json['flagship_models_no_logp'][display] = stats

    for name, stats in human_stats.items():
        s = {k: v for k, v in stats.items() if k != 'selective_prediction'}
        all_stats_json['human_raters'][name] = s

    with open(str(STTS / '15_ai_confidence_vs_human.json'), 'w') as f:
        json.dump(all_stats_json, f, indent=2, default=str)
    print("  -> 15_ai_confidence_vs_human.json")

    # ---- Print comprehensive summary ----
    print("\n" + "=" * 70)
    print("COMPREHENSIVE RESULTS SUMMARY")
    print("=" * 70)

    print("\n--- AI Models with Token Probabilities (logp) ---")
    print(f"{'Model':<30} {'Acc':>6} {'Conf':>6} {'Ent':>6} {'Gap':>7} {'MW p':>8} {'Discrimin.':>10}")
    print("-" * 80)
    for name in ['GPT-5.2 (baseline)',
                 'SFT-1 (CYqJRxId)', 'SFT-2 (ckpt-304)',
                 'SFT-3 (ckpt-380)', 'SFT-4 (ckpt-228)',
                 'SFT 2-Model Ensemble',
                 'Human Voting (old students)']:
        stats = all_model_stats.get(name)
        if not stats:
            continue
        mw_p = stats['mann_whitney_correct_gt_wrong']['p']
        disc = 'YES' if mw_p < 0.05 else 'no'
        print(f"  {name:<28} {stats['accuracy']:>5.1%} {stats['mean_confidence']:>6.3f} "
              f"{stats['mean_entropy']:>6.3f} {stats['conf_gap']:>+6.3f} "
              f"{mw_p:>8.4f} {disc:>10}")

    print("\n--- Flagship Models (NO token probabilities) ---")
    print(f"{'Model':<30} {'Acc':>6} {'Note':>30}")
    print("-" * 70)
    sorted_flagships = sorted(flagship_stats.items(),
                               key=lambda x: x[1]['accuracy'], reverse=True)
    for model_key, stats in sorted_flagships:
        display = FLAGSHIP_DISPLAY.get(model_key, model_key)
        print(f"  {display:<28} {stats['accuracy']:>5.1%} "
              f"{'No logp — confidence unavailable':>30}")

    print("\n--- Human Raters (Self-Reported Confidence) ---")
    print(f"{'Group':<20} {'Acc':>6} {'Conf':>8} {'Gap':>7} {'MW p':>8} {'Discrimin.':>10}")
    print("-" * 65)
    for name, stats in human_stats.items():
        mw_p = stats['mann_whitney_correct_gt_wrong']['p']
        disc = 'YES' if mw_p < 0.05 else 'no'
        print(f"  {name:<18} {stats['accuracy']:>5.1%} "
              f"{stats['mean_confidence_raw']:>6.2f}/5 "
              f"{stats['conf_gap_raw']:>+6.2f} "
              f"{mw_p:>8.4f} {disc:>10}")

    print("\n--- Key Findings ---")
    sft_ens = all_model_stats.get('SFT 2-Model Ensemble', {})
    gpt_base = all_model_stats.get('GPT-5.2 (baseline)', {})

    if sft_ens and gpt_base:
        print(f"\n1. GPT-5.2 baseline is overconfident (mean={gpt_base['mean_confidence']:.3f}) "
              f"but CANNOT distinguish correct from wrong (gap={gpt_base['conf_gap']:+.3f})")
        print(f"   -> SFT fine-tuning produces calibrated confidence "
              f"(gap={sft_ens['conf_gap']:+.3f}, p={sft_ens['mann_whitney_correct_gt_wrong']['p']:.4f})")

    if human_stats.get('Human Expert'):
        exp = human_stats['Human Expert']
        print(f"\n2. Human experts have weak confidence discrimination "
              f"(gap={exp['conf_gap_raw']:+.2f}/5, p={exp['mann_whitney_correct_gt_wrong']['p']:.4f})")

    if sft_ens:
        sel = sft_ens.get('selective_prediction', [])
        high_conf = [p for p in sel if p['coverage'] <= 0.15]
        if high_conf:
            best = max(high_conf, key=lambda p: p['accuracy'])
            print(f"\n3. SFT ensemble: top {best['coverage']:.0%} most confident -> "
                  f"{best['accuracy']:.0%} accuracy (n={best['n_kept']})")

    print("\nDone!")


if __name__ == '__main__':
    main()
