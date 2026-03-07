#!/usr/bin/env python3
"""Script 10: Calibration analysis.

Computes Expected Calibration Error (ECE), Brier scores, and reliability
diagrams for the SFT 2-model ensemble, GPT-5.2 baseline, expert voting,
and student voting.  Examines per-tier calibration quality.

Outputs:
    - fig10_calibration_diagram.pdf
    - fig_s13_ece_comparison.pdf
    - table_s7_calibration_metrics.csv
    - 10_calibration.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from analysis.utils.constants import *
from analysis.utils.data_loader import *
from analysis.utils.metrics import *
from analysis.utils.voting import *
from analysis.utils.statistical_tests import *
from analysis.utils.visualization import *


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------

def _get_model_probs(val_data, model_key):
    """Extract probability dicts and ground truth for a model.

    Returns:
        y_true: list of ground-truth labels (AI label space).
        y_probs: list of probability dicts.
    """
    y_true = []
    y_probs = []

    for art in val_data:
        gt = art.get('rank', '').strip().lower()
        if not gt:
            continue
        entry = art.get('val_outcome', {}).get(
            'rq_with_context', {}
        ).get(model_key)
        if entry is None:
            continue
        logp = entry.get('logp', {})
        if not logp:
            continue
        probs = logp_to_prob(logp)
        y_true.append(gt)
        y_probs.append(probs)

    return y_true, y_probs


def _get_human_voting_probs(val_data):
    """Extract voting-based probability distributions from human_voting.

    The 'confidence' for human voting = proportion assigned to majority.

    Returns:
        y_true: list of ground-truth labels.
        y_probs: list of probability dicts (normalised vote proportions).
    """
    y_true = []
    y_probs = []

    for art in val_data:
        gt = art.get('rank', '').strip().lower()
        if not gt:
            continue
        hv = art.get('val_outcome', {}).get(
            'rq_with_context', {}
        ).get('human_voting')
        if hv is None:
            continue
        ratings = hv.get('ratings', {})
        if not ratings:
            continue
        # Convert rating counts to probabilities in AI label space
        total = sum(ratings.values())
        if total == 0:
            continue
        probs = {}
        for label in LABEL_ORDER:
            human_key = AI_TO_HUMAN.get(label, '')
            count = ratings.get(human_key, 0)
            probs[label] = count / total
        y_true.append(gt)
        y_probs.append(probs)

    return y_true, y_probs


def _get_expert_voting_probs(expert_data):
    """Build expert-only voting probabilities from expert.jsonl.

    Returns:
        y_true: list of ground-truth labels.
        y_probs: list of probability dicts.
    """
    y_true = []
    y_probs = []

    for art in expert_data:
        gt = normalize_level(art.get('level', ''))
        if not gt:
            continue
        surveys = art.get('ratings', art.get('surveys', []))
        labels = []
        for entry in surveys:
            raw = entry.get('q2_rating', entry.get('rating', ''))
            label = normalize_level(raw)
            if label in set(LABEL_ORDER):
                labels.append(label)
        if not labels:
            continue
        total = len(labels)
        counts = Counter(labels)
        probs = {lab: counts.get(lab, 0) / total for lab in LABEL_ORDER}
        y_true.append(gt)
        y_probs.append(probs)

    return y_true, y_probs


def _get_student_voting_probs(student_data):
    """Build student voting probabilities from student_merged.jsonl.

    Returns:
        y_true: list of ground-truth labels.
        y_probs: list of probability dicts.
    """
    y_true = []
    y_probs = []

    for art in student_data:
        gt = normalize_level(art.get('level', ''))
        if not gt:
            continue
        surveys = art.get('ratings', art.get('surveys', []))
        labels = []
        for entry in surveys:
            raw = entry.get('q2_rating', entry.get('rating', ''))
            label = normalize_level(raw)
            if label in set(LABEL_ORDER):
                labels.append(label)
        if not labels:
            continue
        total = len(labels)
        counts = Counter(labels)
        probs = {lab: counts.get(lab, 0) / total for lab in LABEL_ORDER}
        y_true.append(gt)
        y_probs.append(probs)

    return y_true, y_probs


def _compute_calibration_bins(y_true, y_probs, n_bins=10):
    """Compute calibration bins for a reliability diagram.

    Returns:
        bins: list of (mean_confidence, fraction_correct, bin_count).
    """
    if not y_true:
        return []

    confidences = []
    corrects = []
    for true_label, prob_dict in zip(y_true, y_probs):
        pred_label = max(prob_dict, key=prob_dict.get)
        max_conf = prob_dict[pred_label]
        confidences.append(max_conf)
        corrects.append(1 if pred_label == true_label else 0)

    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    bins = []

    for i in range(n_bins):
        lower = bin_boundaries[i]
        upper = bin_boundaries[i + 1]
        if i == n_bins - 1:
            in_bin = [
                j for j, c in enumerate(confidences)
                if lower <= c <= upper
            ]
        else:
            in_bin = [
                j for j, c in enumerate(confidences)
                if lower <= c < upper
            ]
        if not in_bin:
            continue
        mean_conf = float(np.mean([confidences[j] for j in in_bin]))
        frac_correct = float(np.mean([corrects[j] for j in in_bin]))
        bins.append((mean_conf, frac_correct, len(in_bin)))

    return bins


def _brier_decomposition(y_true, y_probs, labels=LABEL_ORDER):
    """Compute Brier score decomposition: reliability, resolution, uncertainty.

    Uses the Murphy (1973) decomposition:
        BS = reliability - resolution + uncertainty

    Returns:
        dict with brier_score, reliability, resolution, uncertainty.
    """
    n = len(y_true)
    if n == 0:
        return {
            'brier_score': 0.0,
            'reliability': 0.0,
            'resolution': 0.0,
            'uncertainty': 0.0,
        }

    brier = compute_brier_score(y_true, y_probs, labels)

    # Base rate per class
    class_counts = Counter(y_true)
    base_rates = {lab: class_counts.get(lab, 0) / n for lab in labels}

    # Uncertainty: sum over classes of base_rate * (1 - base_rate)
    uncertainty = sum(p * (1 - p) for p in base_rates.values())

    # For a proper decomposition we need to bin predictions.
    # We use a simplified approach: group by predicted class and confidence bin.
    n_bins = 10
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)

    reliability = 0.0
    resolution = 0.0

    # Bin by max predicted class and confidence
    confidences = []
    pred_labels = []
    for prob_dict in y_probs:
        pred = max(prob_dict, key=prob_dict.get)
        pred_labels.append(pred)
        confidences.append(prob_dict[pred])

    for bin_idx in range(n_bins):
        lo = bin_boundaries[bin_idx]
        hi = bin_boundaries[bin_idx + 1]
        if bin_idx == n_bins - 1:
            in_bin = [
                i for i, c in enumerate(confidences) if lo <= c <= hi
            ]
        else:
            in_bin = [
                i for i, c in enumerate(confidences) if lo <= c < hi
            ]
        if not in_bin:
            continue

        n_k = len(in_bin)
        bin_acc = sum(
            1 for i in in_bin if pred_labels[i] == y_true[i]
        ) / n_k
        bin_conf = float(np.mean([confidences[i] for i in in_bin]))

        # Reliability contribution
        reliability += (n_k / n) * (bin_conf - bin_acc) ** 2

        # Resolution contribution: how much this bin's accuracy differs
        # from the overall base rate for the predicted class
        overall_acc = sum(
            1 for t, p in zip(y_true, pred_labels) if t == p
        ) / n
        resolution += (n_k / n) * (bin_acc - overall_acc) ** 2

    return {
        'brier_score': brier,
        'reliability': reliability,
        'resolution': resolution,
        'uncertainty': uncertainty,
    }


def _per_tier_ece(y_true, y_probs, n_bins=10):
    """Compute ECE separately for each ground-truth tier.

    Returns:
        dict mapping tier -> {ece, n}.
    """
    tier_data = defaultdict(lambda: {'true': [], 'probs': []})
    for t, p in zip(y_true, y_probs):
        tier_data[t]['true'].append(t)
        tier_data[t]['probs'].append(p)

    result = {}
    for tier in LABEL_ORDER:
        td = tier_data.get(tier, {'true': [], 'probs': []})
        if len(td['true']) < 2:
            result[tier] = {'ece': None, 'n': len(td['true'])}
            continue
        ece_val = compute_ece(td['true'], td['probs'], n_bins=max(n_bins, 3))
        result[tier] = {'ece': ece_val, 'n': len(td['true'])}

    return result


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

def main():
    root = get_project_root()
    fig_dir = root / FIGURES_DIR
    tbl_dir = root / TABLES_DIR
    stat_dir = root / STATS_DIR
    for d in (fig_dir, tbl_dir, stat_dir):
        d.mkdir(parents=True, exist_ok=True)

    # Load data
    val_data = load_val_data()
    expert_data = load_expert_ratings()
    student_merged_data = load_student_merged_ratings()

    # ---- Extract probabilities for each evaluator -----------------------
    evaluators = {}

    # 1. SFT 2-model ensemble
    sft_true, sft_probs = _get_model_probs(val_data, 'best_2_model_combo')
    evaluators['SFT 2-Model Ensemble'] = {
        'y_true': sft_true,
        'y_probs': sft_probs,
        'color': COLORS['sft'],
    }

    # 2. GPT-5.2 baseline — now in chat data, handled by chat model section below

    # 3. Expert voting
    exp_true, exp_probs = _get_expert_voting_probs(expert_data)
    evaluators['Expert voting'] = {
        'y_true': exp_true,
        'y_probs': exp_probs,
        'color': COLORS['expert'],
    }

    # 4. Student voting (full panel)
    stu_true, stu_probs = _get_student_voting_probs(student_merged_data)
    evaluators['Student voting'] = {
        'y_true': stu_true,
        'y_probs': stu_probs,
        'color': COLORS['student'],
    }

    # 5-7. Chat models with logp (from 120_chat.jsonl)
    chat_colors = {'gpt-5.2': '#A6761D', 'kimi-k2-0905-preview': '#7570B3',
                   'deepseek-chat': '#66A61E'}
    try:
        chat_data = load_chat_data()
        for model_key in CHAT_MODELS:
            ct, cp = _get_model_probs(chat_data, model_key)
            display = MODEL_DISPLAY_NAMES.get(model_key, model_key)
            if model_key == 'gpt-5.2':
                display = 'GPT-5.2 (chat)'
            evaluators[display] = {
                'y_true': ct,
                'y_probs': cp,
                'color': chat_colors.get(model_key, '#999999'),
            }
            print(f'  Chat model {display}: {len(ct)} articles with logp')
    except FileNotFoundError:
        print('  WARNING: 120_chat.jsonl not found, skipping chat models')

    # ---- Compute metrics for each evaluator -----------------------------
    metrics = {}
    for name, ev in evaluators.items():
        yt = ev['y_true']
        yp = ev['y_probs']
        n = len(yt)

        if n == 0:
            metrics[name] = {
                'ece': None, 'brier_score': None,
                'brier_decomposition': None, 'n': 0,
                'per_tier_ece': {},
            }
            continue

        ece_val = compute_ece(yt, yp)
        brier_val = compute_brier_score(yt, yp)
        brier_decomp = _brier_decomposition(yt, yp)
        tier_ece = _per_tier_ece(yt, yp)

        # Predicted accuracy (from argmax)
        preds = [max(p, key=p.get) for p in yp]
        acc = compute_accuracy(yt, preds)

        metrics[name] = {
            'ece': ece_val,
            'brier_score': brier_val,
            'brier_decomposition': brier_decomp,
            'accuracy': acc,
            'n': n,
            'per_tier_ece': tier_ece,
        }

    # ---- 5: Reliability diagram (main figure) ---------------------------
    set_nature_style()
    fig_cal, ax_cal = plt.subplots(figsize=(4.5, 4.5))

    # Diagonal perfect calibration line
    ax_cal.plot([0, 1], [0, 1], 'k--', linewidth=0.5, alpha=0.6,
                label='Perfectly calibrated')

    markers = ['o', 's', '^', 'D']
    for idx, (name, ev) in enumerate(evaluators.items()):
        yt = ev['y_true']
        yp = ev['y_probs']
        if not yt:
            continue
        bins = _compute_calibration_bins(yt, yp, n_bins=10)
        if not bins:
            continue
        x_vals = [b[0] for b in bins]
        y_vals = [b[1] for b in bins]
        sizes = [b[2] for b in bins]
        marker = markers[idx % len(markers)]
        ax_cal.plot(x_vals, y_vals, f'{marker}-', color=ev['color'],
                    linewidth=1.2, markersize=4, label=name)
        # Show bin counts as tiny annotations
        for xv, yv, sz in zip(x_vals, y_vals, sizes):
            ax_cal.annotate(str(sz), (xv, yv),
                            textcoords='offset points',
                            xytext=(3, 3), fontsize=4, color=ev['color'])

    ax_cal.set_xlim(0, 1)
    ax_cal.set_ylim(0, 1)
    ax_cal.set_xlabel('Mean predicted confidence')
    ax_cal.set_ylabel('Observed accuracy')
    ax_cal.set_aspect('equal')
    ax_cal.set_title('Calibration (reliability) diagram', fontsize=9,
                     fontweight='bold')
    ax_cal.legend(fontsize=6, loc='lower right')
    fig_cal.tight_layout()
    save_figure(fig_cal, 'fig10_calibration_diagram', output_dir=fig_dir)
    plt.close(fig_cal)

    # ---- Supplementary: ECE comparison bar chart + per-tier --------------
    set_nature_style()
    fig_ece, axes_ece = plt.subplots(1, 2, figsize=(8.0, 3.5))

    # Panel A: Overall ECE comparison
    ax_ea = axes_ece[0]
    eval_names = list(metrics.keys())
    ece_vals = [metrics[n]['ece'] if metrics[n]['ece'] is not None else 0
                for n in eval_names]
    bar_colors = [evaluators[n]['color'] for n in eval_names]
    x_pos = np.arange(len(eval_names))
    ax_ea.bar(x_pos, ece_vals, width=0.6, color=bar_colors,
              edgecolor='white', linewidth=0.5)
    ax_ea.set_xticks(x_pos)
    ax_ea.set_xticklabels(eval_names, rotation=35, ha='right', fontsize=6)
    ax_ea.set_ylabel('ECE')
    ax_ea.set_title('a  Expected Calibration Error', fontsize=9,
                    fontweight='bold', loc='left')
    for i, v in enumerate(ece_vals):
        ax_ea.text(i, v + 0.005, f'{v:.3f}', ha='center', va='bottom',
                   fontsize=6)

    # Panel B: Per-tier ECE as grouped bars
    ax_eb = axes_ece[1]
    tier_labels = [AI_TO_HUMAN.get(t, t) for t in LABEL_ORDER]
    n_evals = len(eval_names)
    w = 0.8 / n_evals
    x_tier = np.arange(len(LABEL_ORDER))

    for idx, name in enumerate(eval_names):
        tier_ece = metrics[name]['per_tier_ece']
        vals = []
        for tier in LABEL_ORDER:
            te = tier_ece.get(tier, {})
            vals.append(te.get('ece', 0) if te.get('ece') is not None else 0)
        offset = (idx - n_evals / 2 + 0.5) * w
        ax_eb.bar(x_tier + offset, vals, w, label=name,
                  color=evaluators[name]['color'],
                  edgecolor='white', linewidth=0.3)

    ax_eb.set_xticks(x_tier)
    ax_eb.set_xticklabels(tier_labels, rotation=45, ha='right')
    ax_eb.set_ylabel('ECE')
    ax_eb.set_title('b  ECE by ground-truth tier', fontsize=9,
                    fontweight='bold', loc='left')
    ax_eb.legend(fontsize=5, loc='best')

    fig_ece.tight_layout()
    save_figure(fig_ece, 'fig_s13_ece_comparison', output_dir=fig_dir)
    plt.close(fig_ece)

    # ---- CSV table ------------------------------------------------------
    table_rows = []
    for name in eval_names:
        m = metrics[name]
        bd = m.get('brier_decomposition') or {}
        row = {
            'evaluator': name,
            'n': m['n'],
            'accuracy': round(m.get('accuracy', 0), 4) if m.get('accuracy') is not None else None,
            'ece': round(m['ece'], 4) if m['ece'] is not None else None,
            'brier_score': (
                round(m['brier_score'], 4)
                if m['brier_score'] is not None else None
            ),
            'brier_reliability': (
                round(bd.get('reliability', 0), 4) if bd else None
            ),
            'brier_resolution': (
                round(bd.get('resolution', 0), 4) if bd else None
            ),
            'brier_uncertainty': (
                round(bd.get('uncertainty', 0), 4) if bd else None
            ),
        }
        # Add per-tier ECE columns
        for tier in LABEL_ORDER:
            te = m['per_tier_ece'].get(tier, {})
            ece_t = te.get('ece')
            row[f'ece_{tier}'] = round(ece_t, 4) if ece_t is not None else None
        table_rows.append(row)

    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(tbl_dir / 'table_s7_calibration_metrics.csv',
                    index=False)

    # ---- JSON summary ---------------------------------------------------
    # Make metrics JSON-serialisable
    def _clean(obj):
        if isinstance(obj, dict):
            return {k: _clean(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_clean(v) for v in obj]
        if isinstance(obj, (np.integer,)):
            return int(obj)
        if isinstance(obj, (np.floating,)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    summary = {
        'metrics': _clean(metrics),
        'calibration_bins': {},
    }
    for name, ev in evaluators.items():
        if not ev['y_true']:
            continue
        bins = _compute_calibration_bins(ev['y_true'], ev['y_probs'])
        summary['calibration_bins'][name] = [
            {'mean_conf': b[0], 'frac_correct': b[1], 'count': b[2]}
            for b in bins
        ]

    with open(stat_dir / '10_calibration.json', 'w') as f:
        json.dump(summary, f, indent=2, default=str)

    # ---- Print key results ----------------------------------------------
    print('=== Script 10: Calibration Analysis ===')
    print()
    print(f'{"Evaluator":<25s} {"N":>4s}  {"Acc":>6s}  {"ECE":>6s}  '
          f'{"Brier":>6s}  {"Reliab":>7s}  {"Resol":>6s}  {"Uncert":>7s}')
    print('-' * 80)
    for name in eval_names:
        m = metrics[name]
        bd = m.get('brier_decomposition') or {}
        acc_str = f'{m["accuracy"]:.3f}' if m.get('accuracy') is not None else 'N/A'
        ece_str = f'{m["ece"]:.3f}' if m['ece'] is not None else 'N/A'
        brier_str = (
            f'{m["brier_score"]:.3f}'
            if m['brier_score'] is not None else 'N/A'
        )
        rel_str = f'{bd.get("reliability", 0):.4f}' if bd else 'N/A'
        res_str = f'{bd.get("resolution", 0):.4f}' if bd else 'N/A'
        unc_str = f'{bd.get("uncertainty", 0):.4f}' if bd else 'N/A'
        print(f'{name:<25s} {m["n"]:>4d}  {acc_str:>6s}  {ece_str:>6s}  '
              f'{brier_str:>6s}  {rel_str:>7s}  {res_str:>6s}  {unc_str:>7s}')

    print()
    print('Per-tier ECE:')
    for tier in LABEL_ORDER:
        print(f'  {AI_TO_HUMAN.get(tier, tier):>6s}: ', end='')
        for name in eval_names:
            te = metrics[name]['per_tier_ece'].get(tier, {})
            ece_t = te.get('ece')
            e_str = f'{ece_t:.3f}' if ece_t is not None else 'N/A'
            print(f'{name}={e_str}  ', end='')
        print()

    print()
    print('Outputs saved.')


if __name__ == '__main__':
    main()
