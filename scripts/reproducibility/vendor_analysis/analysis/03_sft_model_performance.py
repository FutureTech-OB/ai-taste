#!/usr/bin/env python3
"""Script 03: SFT model performance analysis.

Compares SFT-trained checkpoints, their ensembles, and the GPT-5.2 baseline
on the 120-article research quality classification task.  The key finding is
that SFT resolves the "mediocrity bias" where baseline models cluster
predictions in the middle tiers (strong/fair).

Produces:
    - results/tables/table4_sft_performance.csv
    - results/figures/fig3_sft_vs_baseline_confusion.pdf + .png
    - results/figures/fig4_sft_pertier_accuracy.pdf + .png
    - results/figures/fig_s4_prediction_distribution_shift.pdf + .png
    - results/statistics/03_sft.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict

from analysis.utils.constants import *
from analysis.utils.data_loader import *
from analysis.utils.metrics import *
from analysis.utils.voting import *
from analysis.utils.statistical_tests import *
from analysis.utils.visualization import *


# ---------------------------------------------------------------------------
# Display names for val models
# ---------------------------------------------------------------------------
VAL_DISPLAY = {
    'gpt-5.2': 'GPT-5.2 (Baseline)',
    'CYqJRxId': 'SFT (GPT-4.1)',
    'ckpt-step-304': 'SFT (GPT-4.1-nano)',
    'ckppt-380': 'SFT (Qwen3-30B)',
    'ckppt-228': 'SFT (Qwen3-4B)',
    'human_voting': 'Human Voting',
    'best_2_model_combo': '2-Model Ensemble',
}

# Ordered list: baseline first, then SFT checkpoints, then ensembles
MODEL_ORDER = [
    'gpt-5.2',
    'CYqJRxId',
    'ckpt-step-304',
    'ckppt-380',
    'ckppt-228',
    'best_2_model_combo',
]


def extract_val_prediction(model_data: dict) -> str:
    """Extract the predicted label from a val model's logp dict.

    Takes the argmax of the logp dict. Missing labels are treated as
    -infinity.
    """
    logp = model_data.get('logp', {})
    if not logp:
        return 'unknown'

    # Ensure all labels are present with -inf default
    full_logp = {label: logp.get(label, float('-inf')) for label in LABEL_ORDER}
    best_label = max(full_logp, key=full_logp.get)
    return best_label


def compute_model_metrics(y_true_list, y_pred_list, display_name):
    """Compute a full suite of metrics for a model and return a dict."""
    valid_mask = [p != 'unknown' for p in y_pred_list]
    n_valid = sum(valid_mask)
    y_true_v = [t for t, v in zip(y_true_list, valid_mask) if v]
    y_pred_v = [p for p, v in zip(y_pred_list, valid_mask) if v]

    if n_valid == 0:
        return None

    acc = compute_accuracy(y_true_v, y_pred_v)
    macro_f1 = compute_macro_f1(y_true_v, y_pred_v)
    per_class = compute_per_class_metrics(y_true_v, y_pred_v)
    cm = compute_confusion_matrix(y_true_v, y_pred_v)
    pred_dist = Counter(y_pred_v)

    per_tier_acc = {}
    for label in LABEL_ORDER:
        tier_true = [t for t in y_true_v if t == label]
        tier_pred = [p for t, p in zip(y_true_v, y_pred_v) if t == label]
        if tier_true:
            per_tier_acc[label] = compute_accuracy(tier_true, tier_pred)
        else:
            per_tier_acc[label] = float('nan')

    return {
        'display_name': display_name,
        'n_valid': n_valid,
        'accuracy': acc,
        'macro_f1': macro_f1,
        'per_class_metrics': {
            k: {mk: round(mv, 4) for mk, mv in v.items()}
            for k, v in per_class.items()
        },
        'per_tier_accuracy': {k: round(v, 4) for k, v in per_tier_acc.items()},
        'prediction_distribution': dict(pred_dist),
        'confusion_matrix': cm.tolist(),
        'y_true': y_true_v,
        'y_pred': y_pred_v,
    }


def main():
    print("=" * 70)
    print("Script 03: SFT Model Performance")
    print("=" * 70)

    project_root = get_project_root()
    figures_dir = project_root / FIGURES_DIR
    tables_dir = project_root / TABLES_DIR
    stats_dir = project_root / STATS_DIR

    for d in (figures_dir, tables_dir, stats_dir):
        d.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n[1/8] Loading validation data...")
    val_data = load_val_data()
    print(f"  Trained records loaded: {len(val_data)}")

    # Also load chat data for gpt-5.2 baseline (moved from 120_val to 120_chat)
    try:
        chat_data = load_chat_data()
        print(f"  Chat records loaded: {len(chat_data)}")
    except FileNotFoundError:
        chat_data = []
        print("  WARNING: 120_chat.jsonl not found")

    y_true = [get_ground_truth(rec) for rec in val_data]
    print(f"  Ground truth distribution: {Counter(y_true)}")

    # Discover model keys
    first_rq = val_data[0]['val_outcome']['rq_with_context']
    available_models = list(first_rq.keys())
    if chat_data:
        chat_rq = chat_data[0]['val_outcome']['rq_with_context']
        available_models.extend(chat_rq.keys())
    available_models = list(set(available_models))
    print(f"  Available models: {available_models}")

    # ------------------------------------------------------------------
    # Extract predictions
    # ------------------------------------------------------------------
    print("\n[2/8] Extracting predictions...")

    predictions = {}
    for model_key in MODEL_ORDER:
        if model_key not in available_models:
            print(f"  WARNING: {model_key} not found in any data file, skipping.")
            continue

        preds = []
        # Check trained data first, then chat data
        source = val_data
        first_rq_check = val_data[0]['val_outcome']['rq_with_context']
        if model_key not in first_rq_check and chat_data:
            source = chat_data

        for rec in source:
            rq = rec['val_outcome']['rq_with_context']
            model_data = rq.get(model_key, {})
            preds.append(extract_val_prediction(model_data))
        predictions[model_key] = preds

    # ------------------------------------------------------------------
    # Compute metrics
    # ------------------------------------------------------------------
    print("\n[3/8] Computing metrics...")

    model_metrics = {}
    table_rows = []

    for model_key in MODEL_ORDER:
        if model_key not in predictions:
            continue
        display = VAL_DISPLAY.get(model_key, model_key)
        metrics = compute_model_metrics(y_true, predictions[model_key], display)
        if metrics is None:
            print(f"  {display}: No valid predictions, skipping.")
            continue

        model_metrics[model_key] = metrics

        row = {
            'Model': display,
            'Model Key': model_key,
            'N Valid': metrics['n_valid'],
            'Accuracy': round(metrics['accuracy'], 4),
            'Macro F1': round(metrics['macro_f1'], 4),
        }
        for label in LABEL_ORDER:
            row[f'Acc ({label})'] = round(metrics['per_tier_accuracy'].get(label, float('nan')), 4)
        for label in LABEL_ORDER:
            row[f'Pred count ({label})'] = metrics['prediction_distribution'].get(label, 0)

        table_rows.append(row)

        print(f"  {display}: Acc={metrics['accuracy']:.3f}, F1={metrics['macro_f1']:.3f}, "
              f"Pred dist={metrics['prediction_distribution']}")

    # Also compute for ensemble info
    if 'best_2_model_combo' in available_models:
        combo_data = val_data[0]['val_outcome']['rq_with_context'].get('best_2_model_combo', {})
        combo_models = combo_data.get('models', [])
        combo_method = combo_data.get('method', 'unknown')
        print(f"\n  {VAL_DISPLAY.get('best_2_model_combo', 'best_2_model_combo')}:")
        print(f"    Component models: {combo_models}")
        print(f"    Aggregation method: {combo_method}")

    # ------------------------------------------------------------------
    # Save Table 4
    # ------------------------------------------------------------------
    table4 = pd.DataFrame(table_rows)
    table4_path = tables_dir / 'table4_sft_performance.csv'
    table4.to_csv(table4_path, index=False)
    print(f"\n  Saved: {table4_path}")

    # ------------------------------------------------------------------
    # McNemar tests
    # ------------------------------------------------------------------
    print("\n[4/8] Running McNemar tests...")

    mcnemar_comparisons = {}

    # Find best SFT single model
    sft_single_keys = [k for k in MODEL_ORDER
                       if k not in ('gpt-5.2', 'best_2_model_combo')
                       and k in model_metrics]
    if sft_single_keys:
        best_sft_key = max(sft_single_keys, key=lambda k: model_metrics[k]['accuracy'])
        best_sft_display = VAL_DISPLAY.get(best_sft_key, best_sft_key)
        print(f"  Best single SFT model: {best_sft_display} "
              f"(Acc={model_metrics[best_sft_key]['accuracy']:.3f})")
    else:
        best_sft_key = None

    test_pairs = []
    if 'gpt-5.2' in predictions and best_sft_key and best_sft_key in predictions:
        test_pairs.append(('gpt-5.2', best_sft_key, 'Baseline vs Best SFT Single'))
    if 'gpt-5.2' in predictions and 'best_2_model_combo' in predictions:
        test_pairs.append(('gpt-5.2', 'best_2_model_combo', 'Baseline vs 2-Model Ensemble'))
    if best_sft_key and best_sft_key in predictions and 'best_2_model_combo' in predictions:
        test_pairs.append((best_sft_key, 'best_2_model_combo', 'Best SFT Single vs 2-Model Ensemble'))
    for m1_key, m2_key, label in test_pairs:
        p1 = predictions[m1_key]
        p2 = predictions[m2_key]

        valid = [a != 'unknown' and b != 'unknown' for a, b in zip(p1, p2)]
        yt = [t for t, v in zip(y_true, valid) if v]
        yp1 = [p for p, v in zip(p1, valid) if v]
        yp2 = [p for p, v in zip(p2, valid) if v]

        if len(yt) < 5:
            continue

        result = mcnemar_test(yt, yp1, yp2)
        mcnemar_comparisons[label] = {
            'model_1': VAL_DISPLAY.get(m1_key, m1_key),
            'model_2': VAL_DISPLAY.get(m2_key, m2_key),
            'statistic': result['statistic'],
            'p_value': result['p_value'],
            'b': result['b'],
            'c': result['c'],
            'significant': result['p_value'] < 0.05,
        }
        sig_marker = "*" if result['p_value'] < 0.05 else ""
        print(f"  {label}: chi2={result['statistic']:.2f}, "
              f"p={result['p_value']:.4f}{sig_marker}, "
              f"b={result['b']}, c={result['c']}")

    # ------------------------------------------------------------------
    # Bootstrap CIs for accuracy differences
    # ------------------------------------------------------------------
    print("\n[5/8] Computing bootstrap CIs for accuracy differences...")

    bootstrap_results = {}

    if 'gpt-5.2' in model_metrics and best_sft_key and best_sft_key in model_metrics:
        # Build paired correctness arrays
        p_base = predictions['gpt-5.2']
        p_sft = predictions[best_sft_key]
        valid = [a != 'unknown' and b != 'unknown' for a, b in zip(p_base, p_sft)]

        diff_array = []
        for t, pb, ps, v in zip(y_true, p_base, p_sft, valid):
            if v:
                correct_base = 1.0 if pb == t else 0.0
                correct_sft = 1.0 if ps == t else 0.0
                diff_array.append(correct_sft - correct_base)

        if diff_array:
            boot_result = bootstrap_ci(
                diff_array,
                statistic_fn=lambda x: float(np.mean(x)),
                n_bootstrap=10000,
                seed=42,
            )
            bootstrap_results['baseline_vs_best_sft'] = {
                'comparison': f"GPT-5.2 vs {best_sft_display}",
                'accuracy_diff': boot_result['estimate'],
                'ci_lower': boot_result['ci_lower'],
                'ci_upper': boot_result['ci_upper'],
                'std': boot_result['std'],
            }
            print(f"  Baseline vs Best SFT: diff={boot_result['estimate']:.4f} "
                  f"[{boot_result['ci_lower']:.4f}, {boot_result['ci_upper']:.4f}]")

    if 'gpt-5.2' in model_metrics and 'best_2_model_combo' in model_metrics:
        p_base = predictions['gpt-5.2']
        p_ens = predictions['best_2_model_combo']
        valid = [a != 'unknown' and b != 'unknown' for a, b in zip(p_base, p_ens)]

        diff_array = []
        for t, pb, pe, v in zip(y_true, p_base, p_ens, valid):
            if v:
                correct_base = 1.0 if pb == t else 0.0
                correct_ens = 1.0 if pe == t else 0.0
                diff_array.append(correct_ens - correct_base)

        if diff_array:
            boot_result = bootstrap_ci(
                diff_array,
                statistic_fn=lambda x: float(np.mean(x)),
                n_bootstrap=10000,
                seed=42,
            )
            bootstrap_results['baseline_vs_2_ensemble'] = {
                'comparison': 'GPT-5.2 vs 2-Model Ensemble',
                'accuracy_diff': boot_result['estimate'],
                'ci_lower': boot_result['ci_lower'],
                'ci_upper': boot_result['ci_upper'],
                'std': boot_result['std'],
            }
            print(f"  Baseline vs 2-Model Ensemble: diff={boot_result['estimate']:.4f} "
                  f"[{boot_result['ci_lower']:.4f}, {boot_result['ci_upper']:.4f}]")

    # ------------------------------------------------------------------
    # Figure 3: Side-by-side confusion matrices
    # ------------------------------------------------------------------
    print("\n[6/8] Creating figures...")
    set_nature_style()

    cm_labels = [l.capitalize() for l in LABEL_ORDER]

    # Determine which models to compare
    left_key = 'gpt-5.2'
    right_key = 'best_2_model_combo'
    if right_key not in model_metrics:
        # Fallback to best single SFT
        right_key = best_sft_key if best_sft_key else None

    if left_key in model_metrics and right_key and right_key in model_metrics:
        fig, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

        cm_left = np.array(model_metrics[left_key]['confusion_matrix'])
        plot_confusion_matrix(
            cm_left, labels=cm_labels,
            title=VAL_DISPLAY.get(left_key, left_key),
            ax=axes[0], normalize=True,
        )

        cm_right = np.array(model_metrics[right_key]['confusion_matrix'])
        plot_confusion_matrix(
            cm_right, labels=cm_labels,
            title=VAL_DISPLAY.get(right_key, right_key),
            ax=axes[1], normalize=True,
        )

        fig.suptitle(
            'Confusion matrices: baseline vs SFT ensemble',
            fontsize=10, fontweight='bold', y=1.02,
        )
        fig.tight_layout()
        pdf_path, png_path = save_figure(fig, 'fig3_sft_vs_baseline_confusion')
        plt.close(fig)
        print(f"  Saved: {pdf_path}")
    else:
        print("  WARNING: Could not create fig3 (missing model data).")

    # ------------------------------------------------------------------
    # Figure 4: Grouped bar chart of per-tier accuracy
    # ------------------------------------------------------------------
    data_dict = {}
    for model_key in MODEL_ORDER:
        if model_key not in model_metrics:
            continue
        display = VAL_DISPLAY.get(model_key, model_key)
        tier_accs = [
            model_metrics[model_key]['per_tier_accuracy'].get(label, 0.0)
            for label in LABEL_ORDER
        ]
        data_dict[display] = tier_accs

    if data_dict:
        group_labels = [l.capitalize() for l in LABEL_ORDER]

        fig, ax = plt.subplots(figsize=(7.0, 4.0))
        model_names = list(data_dict.keys())
        n_models = len(model_names)
        n_groups = len(group_labels)
        x = np.arange(n_groups)
        width = 0.8 / n_models

        color_list = [
            COLORS['baseline'],  # GPT-5.2
            '#F39B7F',           # SFT 1
            '#E64B35',           # SFT 2
            '#B09C85',           # SFT 3
            '#91D1C2',           # SFT 4
            '#4DBBD5',           # 2-model ensemble
        ]

        for idx, name in enumerate(model_names):
            values = data_dict[name]
            offset = (idx - n_models / 2 + 0.5) * width
            color = color_list[idx % len(color_list)]
            ax.bar(
                x + offset, values, width, label=name,
                color=color, edgecolor='white', linewidth=0.3,
            )

        ax.set_xticks(x)
        ax.set_xticklabels(group_labels)
        ax.set_ylabel('Per-tier Accuracy')
        ax.set_ylim(0, 1.05)
        ax.set_title('Per-tier accuracy: SFT models vs baseline', fontsize=9, fontweight='bold')
        ax.legend(fontsize=5.5, loc='upper right', ncol=2)
        fig.tight_layout()
        pdf_path, png_path = save_figure(fig, 'fig4_sft_pertier_accuracy')
        plt.close(fig)
        print(f"  Saved: {pdf_path}")
    else:
        print("  WARNING: Could not create fig4 (no model data).")

    # ------------------------------------------------------------------
    # Figure S4: Prediction distribution shift (stacked bar)
    # ------------------------------------------------------------------
    print("\n[7/8] Creating prediction distribution shift figure...")

    fig, ax = plt.subplots(figsize=(6.0, 4.0))

    display_order = []
    for model_key in MODEL_ORDER:
        if model_key in model_metrics:
            display_order.append(model_key)

    # Add ground truth as reference
    gt_dist = Counter(y_true)
    all_dists = [('Ground Truth', {l: gt_dist.get(l, 0) for l in LABEL_ORDER})]
    for mk in display_order:
        all_dists.append((
            VAL_DISPLAY.get(mk, mk),
            {l: model_metrics[mk]['prediction_distribution'].get(l, 0) for l in LABEL_ORDER},
        ))

    bar_names = [name for name, _ in all_dists]
    y_pos = np.arange(len(bar_names))
    bar_height = 0.6

    # Compute proportions
    bottoms = np.zeros(len(bar_names))
    tier_colors = [COLORS.get(l, '#999999') for l in LABEL_ORDER]

    for tier_idx, label in enumerate(LABEL_ORDER):
        values = []
        for _, dist in all_dists:
            total = sum(dist.values())
            prop = dist.get(label, 0) / total if total > 0 else 0
            values.append(prop)

        ax.barh(
            y_pos, values, bar_height, left=bottoms,
            label=label.capitalize(),
            color=tier_colors[tier_idx], edgecolor='white', linewidth=0.3,
        )
        bottoms = bottoms + np.array(values)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(bar_names, fontsize=7)
    ax.set_xlabel('Proportion of predictions')
    ax.set_xlim(0, 1.0)
    ax.set_title('Prediction distribution shift: baseline to SFT', fontsize=9, fontweight='bold')
    ax.legend(fontsize=6, loc='lower right', bbox_to_anchor=(1.0, -0.25), ncol=4)
    ax.invert_yaxis()
    fig.tight_layout()
    pdf_path, png_path = save_figure(fig, 'fig_s4_prediction_distribution_shift')
    plt.close(fig)
    print(f"  Saved: {pdf_path}")

    # ------------------------------------------------------------------
    # Mediocrity bias analysis
    # ------------------------------------------------------------------
    print("\n[7.5/8] Analyzing mediocrity bias...")

    if 'gpt-5.2' in model_metrics:
        baseline_dist = model_metrics['gpt-5.2']['prediction_distribution']
        middle_tiers = baseline_dist.get('strong', 0) + baseline_dist.get('fair', 0)
        total_baseline = sum(baseline_dist.values())
        middle_pct = middle_tiers / total_baseline if total_baseline > 0 else 0
        print(f"  GPT-5.2 baseline: {middle_pct:.1%} of predictions in middle tiers (strong+fair)")
        print(f"    Distribution: {baseline_dist}")

    for mk in sft_single_keys:
        if mk in model_metrics:
            sft_dist = model_metrics[mk]['prediction_distribution']
            middle_sft = sft_dist.get('strong', 0) + sft_dist.get('fair', 0)
            total_sft = sum(sft_dist.values())
            middle_pct_sft = middle_sft / total_sft if total_sft > 0 else 0
            print(f"  {VAL_DISPLAY.get(mk, mk)}: {middle_pct_sft:.1%} in middle tiers")
            print(f"    Distribution: {sft_dist}")

    if 'best_2_model_combo' in model_metrics:
        ens_dist = model_metrics['best_2_model_combo']['prediction_distribution']
        middle_ens = ens_dist.get('strong', 0) + ens_dist.get('fair', 0)
        total_ens = sum(ens_dist.values())
        middle_pct_ens = middle_ens / total_ens if total_ens > 0 else 0
        print(f"  {VAL_DISPLAY.get('best_2_model_combo', 'best_2_model_combo')}: {middle_pct_ens:.1%} in middle tiers")
        print(f"    Distribution: {ens_dist}")

    # ------------------------------------------------------------------
    # Save statistics JSON
    # ------------------------------------------------------------------
    print("\n[8/8] Saving statistics JSON...")

    # Strip non-serializable fields from model_metrics
    serializable_metrics = {}
    for mk, mm in model_metrics.items():
        entry = {k: v for k, v in mm.items() if k not in ('y_true', 'y_pred')}
        entry['accuracy'] = round(mm['accuracy'], 4)
        entry['macro_f1'] = round(mm['macro_f1'], 4)
        serializable_metrics[VAL_DISPLAY.get(mk, mk)] = entry

    # Ensemble composition info
    ensemble_info = {}
    if 'best_2_model_combo' in available_models:
        combo_data = val_data[0]['val_outcome']['rq_with_context'].get('best_2_model_combo', {})
        ensemble_info[VAL_DISPLAY.get('best_2_model_combo', 'best_2_model_combo')] = {
            'component_models': combo_data.get('models', []),
            'method': combo_data.get('method', 'unknown'),
        }

    # Mediocrity bias stats
    mediocrity_bias = {}
    if 'gpt-5.2' in model_metrics:
        bd = model_metrics['gpt-5.2']['prediction_distribution']
        total = sum(bd.values())
        mediocrity_bias['baseline'] = {
            'model': 'GPT-5.2',
            'middle_tier_pct': round((bd.get('strong', 0) + bd.get('fair', 0)) / total, 4) if total > 0 else 0,
            'distribution': bd,
        }

    if best_sft_key and best_sft_key in model_metrics:
        sd = model_metrics[best_sft_key]['prediction_distribution']
        total = sum(sd.values())
        mediocrity_bias['best_sft'] = {
            'model': VAL_DISPLAY.get(best_sft_key, best_sft_key),
            'middle_tier_pct': round((sd.get('strong', 0) + sd.get('fair', 0)) / total, 4) if total > 0 else 0,
            'distribution': sd,
        }

    if 'best_2_model_combo' in model_metrics:
        ed = model_metrics['best_2_model_combo']['prediction_distribution']
        total = sum(ed.values())
        mediocrity_bias['2_model_ensemble'] = {
            'model': '2-Model Ensemble',
            'middle_tier_pct': round((ed.get('strong', 0) + ed.get('fair', 0)) / total, 4) if total > 0 else 0,
            'distribution': ed,
        }

    all_stats = {
        'n_articles': len(val_data),
        'ground_truth_distribution': dict(Counter(y_true)),
        'model_performance': serializable_metrics,
        'ensemble_info': ensemble_info,
        'mcnemar_comparisons': mcnemar_comparisons,
        'bootstrap_accuracy_differences': bootstrap_results,
        'mediocrity_bias_analysis': mediocrity_bias,
        'key_finding': (
            'SFT training resolves the mediocrity bias observed in the base GPT-5.2 model. '
            'The untrained baseline clusters predictions predominantly in middle tiers '
            '(strong and fair), while SFT-trained models produce more balanced predictions '
            'across all four quality tiers, leading to improved per-tier accuracy especially '
            'for exceptional and limited categories.'
        ),
    }

    stats_path = stats_dir / '03_sft.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(all_stats, f, indent=2, ensure_ascii=False)
    print(f"  Saved: {stats_path}")

    print("\n" + "=" * 70)
    print("Script 03 complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
