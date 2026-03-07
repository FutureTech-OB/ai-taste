#!/usr/bin/env python3
"""Script 04: Human Expert Performance Analysis.

Analyzes expert rater performance on the 120-article research quality
dataset. Experts only -- students are analyzed in script 05.

Outputs:
    results/tables/table5_expert_performance.csv
    results/tables/table_s2_individual_expert.csv
    results/figures/fig5_expert_accuracy_distribution.pdf
    results/figures/fig_s5_expert_confusion.pdf
    results/figures/fig_s6_expert_confidence_accuracy.pdf
    results/statistics/04_expert.json
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import json
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter, defaultdict
from scipy import stats as scipy_stats

from analysis.utils.constants import *
from analysis.utils.data_loader import *
from analysis.utils.metrics import *
from analysis.utils.voting import *
from analysis.utils.statistical_tests import *
from analysis.utils.visualization import *


# ============================================================================
# Helpers
# ============================================================================

def _parse_confidence(conf_str):
    """Extract integer confidence from string like '3 - Moderately Confident'."""
    try:
        return int(str(conf_str).strip()[0])
    except (ValueError, IndexError, TypeError):
        return None


def _parse_familiarity(fam_str):
    """Extract integer familiarity from string like '4 - Somewhat familiar'."""
    try:
        return int(str(fam_str).strip()[0])
    except (ValueError, IndexError, TypeError):
        return None


def _ensure_dirs():
    """Create output directories if they don't exist."""
    root = get_project_root()
    (root / FIGURES_DIR).mkdir(parents=True, exist_ok=True)
    (root / TABLES_DIR).mkdir(parents=True, exist_ok=True)
    (root / STATS_DIR).mkdir(parents=True, exist_ok=True)


# ============================================================================
# Main analysis
# ============================================================================

def main():
    print("=" * 70)
    print("Script 04: Human Expert Performance Analysis")
    print("=" * 70)

    _ensure_dirs()
    root = get_project_root()
    set_nature_style()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n[1/9] Loading data...")
    expert_data = load_expert_ratings()
    print(f"  Loaded {len(expert_data)} articles from the public expert reproducibility file")

    # ------------------------------------------------------------------
    # Extract all individual expert ratings
    # ------------------------------------------------------------------
    print("\n[2/9] Extracting individual ratings...")

    # Per-article: ground truth and list of (rater_name, predicted_level)
    article_ratings = []           # [{gt, predictions: [level, ...], raters: [...]}]
    individual_records = []        # flat list of dicts per rating

    for article in expert_data:
        gt = normalize_level(article['level'])
        ratings_list = article.get('ratings', [])
        preds = []
        for r in ratings_list:
            pred = normalize_level(r['q2_rating'])
            rater = r.get('rater_id', r.get('rater_name', 'unknown'))
            conf = _parse_confidence(r.get('q3_confidence'))
            fam = _parse_familiarity(r.get('q4_familiarity'))
            read_before = r.get('q1_read_before', 'No')
            is_read = read_before.strip().lower() == 'yes' if isinstance(read_before, str) else False

            preds.append(pred)
            individual_records.append({
                'rater': rater,
                'ground_truth': gt,
                'prediction': pred,
                'correct': int(pred == gt),
                'confidence': conf,
                'familiarity': fam,
                'read_before': is_read,
                'article_number': article.get('article_number'),
                'package': article.get('package'),
                'title': article.get('title', ''),
            })

        article_ratings.append({
            'gt': gt,
            'predictions': preds,
            'title': article.get('title', ''),
            'article_number': article.get('article_number'),
            'package': article.get('package'),
            'n_raters': len(preds),
        })

    total_ratings = len(individual_records)
    unique_experts = set(r['rater'] for r in individual_records)
    print(f"  Total individual ratings: {total_ratings}")
    print(f"  Unique experts: {len(unique_experts)}")
    print(f"  Raters per article: {min(a['n_raters'] for a in article_ratings)}-"
          f"{max(a['n_raters'] for a in article_ratings)}")

    # ------------------------------------------------------------------
    # 1. Expert majority vote accuracy
    # ------------------------------------------------------------------
    print("\n[3/9] Computing majority vote accuracy...")

    results_by_policy = {}
    for policy in ['exclude', 'random', 'wrong']:
        rng = random.Random(42)
        y_true_list = []
        y_pred_list = []
        n_ties = 0
        n_excluded = 0

        for art in article_ratings:
            vote, is_tie = majority_vote(art['predictions'], tie_policy=policy, rng=rng)
            if is_tie:
                n_ties += 1
            if vote is None:
                n_excluded += 1
                continue
            y_true_list.append(art['gt'])
            y_pred_list.append(vote)

        acc = compute_accuracy(y_true_list, y_pred_list)
        macro_f1 = compute_macro_f1(y_true_list, y_pred_list)
        results_by_policy[policy] = {
            'accuracy': acc,
            'macro_f1': macro_f1,
            'n_articles': len(y_true_list),
            'n_ties': n_ties,
            'n_excluded': n_excluded,
            'y_true': y_true_list,
            'y_pred': y_pred_list,
        }
        print(f"  tie_policy='{policy}': accuracy={acc:.4f}, "
              f"macro_f1={macro_f1:.4f}, ties={n_ties}, excluded={n_excluded}")

    # Primary results use 'exclude' policy
    primary = results_by_policy['exclude']
    y_true = primary['y_true']
    y_pred = primary['y_pred']

    # ------------------------------------------------------------------
    # 2. Per-class metrics
    # ------------------------------------------------------------------
    print("\n[4/9] Computing per-class metrics...")

    per_class = compute_per_class_metrics(y_true, y_pred)
    for label in LABEL_ORDER:
        m = per_class[label]
        print(f"  {label:12s}: P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  support={int(m['support'])}")

    # Per-tier accuracy
    tier_accuracy = {}
    for label in LABEL_ORDER:
        tier_true = [t for t, p in zip(y_true, y_pred) if t == label]
        tier_pred = [p for t, p in zip(y_true, y_pred) if t == label]
        if tier_true:
            tier_accuracy[label] = compute_accuracy(tier_true, tier_pred)
        else:
            tier_accuracy[label] = 0.0
        print(f"  {label:12s} accuracy: {tier_accuracy[label]:.3f}")

    # ------------------------------------------------------------------
    # 3. Confusion matrix
    # ------------------------------------------------------------------
    print("\n[5/9] Computing confusion matrix...")

    cm = compute_confusion_matrix(y_true, y_pred)
    print(f"  Confusion matrix shape: {cm.shape}")

    fig_cm, ax_cm = plot_confusion_matrix(
        cm, LABEL_ORDER,
        title='Expert Majority Vote Confusion Matrix',
        normalize=True,
    )
    pdf_path, _ = save_figure(fig_cm, 'fig_s5_expert_confusion')
    plt.close(fig_cm)
    print(f"  Saved: {pdf_path}")

    # ------------------------------------------------------------------
    # 4. Individual expert accuracy
    # ------------------------------------------------------------------
    print("\n[6/9] Computing individual expert accuracy...")

    expert_articles = defaultdict(list)
    for rec in individual_records:
        expert_articles[rec['rater']].append(rec)

    expert_stats_list = []
    for rater, recs in expert_articles.items():
        n_rated = len(recs)
        n_correct = sum(r['correct'] for r in recs)
        acc = n_correct / n_rated if n_rated > 0 else 0.0
        expert_stats_list.append({
            'expert': rater,
            'n_articles': n_rated,
            'n_correct': n_correct,
            'accuracy': acc,
        })

    expert_stats_list.sort(key=lambda x: x['accuracy'], reverse=True)
    expert_accuracies = [e['accuracy'] for e in expert_stats_list]

    print(f"  Number of experts: {len(expert_stats_list)}")
    print(f"  Mean accuracy:   {np.mean(expert_accuracies):.4f}")
    print(f"  Median accuracy: {np.median(expert_accuracies):.4f}")
    print(f"  Std accuracy:    {np.std(expert_accuracies, ddof=1):.4f}")
    print(f"  Min accuracy:    {np.min(expert_accuracies):.4f}")
    print(f"  Max accuracy:    {np.max(expert_accuracies):.4f}")

    # Histogram of individual expert accuracies
    fig_hist, ax_hist = plt.subplots(figsize=(4.0, 3.0))
    ax_hist.hist(expert_accuracies, bins=15, color=COLORS['expert'],
                 edgecolor='white', linewidth=0.5, alpha=0.8)
    ax_hist.axvline(np.mean(expert_accuracies), color='black',
                    linestyle='--', linewidth=0.8, label=f'Mean={np.mean(expert_accuracies):.2f}')
    ax_hist.axvline(0.25, color='red', linestyle=':', linewidth=0.8,
                    label='Chance (25%)')
    ax_hist.set_xlabel('Accuracy')
    ax_hist.set_ylabel('Number of experts')
    ax_hist.set_title('Individual Expert Accuracy Distribution', fontsize=9, fontweight='bold')
    ax_hist.legend(fontsize=7)
    fig_hist.tight_layout()
    pdf_path, _ = save_figure(fig_hist, 'fig5_expert_accuracy_distribution')
    plt.close(fig_hist)
    print(f"  Saved: {pdf_path}")

    # ------------------------------------------------------------------
    # 5. Accuracy by confidence level
    # ------------------------------------------------------------------
    print("\n[7/9] Computing accuracy by confidence / familiarity / read_before...")

    # Confidence
    conf_groups = defaultdict(list)
    for rec in individual_records:
        if rec['confidence'] is not None:
            conf_groups[rec['confidence']].append(rec['correct'])

    conf_accuracy = {}
    conf_x = []
    conf_y = []
    conf_n = []
    for level in sorted(conf_groups.keys()):
        vals = conf_groups[level]
        acc_val = np.mean(vals)
        conf_accuracy[level] = acc_val
        conf_x.append(level)
        conf_y.append(acc_val)
        conf_n.append(len(vals))
        print(f"  Confidence {level}: accuracy={acc_val:.3f} (n={len(vals)})")

    # Familiarity
    fam_groups = defaultdict(list)
    for rec in individual_records:
        if rec['familiarity'] is not None:
            fam_groups[rec['familiarity']].append(rec['correct'])

    fam_accuracy = {}
    fam_x = []
    fam_y = []
    fam_n = []
    for level in sorted(fam_groups.keys()):
        vals = fam_groups[level]
        acc_val = np.mean(vals)
        fam_accuracy[level] = acc_val
        fam_x.append(level)
        fam_y.append(acc_val)
        fam_n.append(len(vals))
        print(f"  Familiarity {level}: accuracy={acc_val:.3f} (n={len(vals)})")

    # Read before
    read_groups = defaultdict(list)
    for rec in individual_records:
        key = 'Yes' if rec['read_before'] else 'No'
        read_groups[key].append(rec['correct'])

    for key in ['Yes', 'No']:
        if key in read_groups:
            vals = read_groups[key]
            print(f"  Read before={key}: accuracy={np.mean(vals):.3f} (n={len(vals)})")

    # Plot confidence vs accuracy
    fig_conf, axes = plt.subplots(1, 2, figsize=(7.0, 3.0))

    axes[0].bar(conf_x, conf_y, color=COLORS['expert'], edgecolor='white', linewidth=0.5)
    for i, (x_val, y_val, n_val) in enumerate(zip(conf_x, conf_y, conf_n)):
        axes[0].text(x_val, y_val + 0.01, f'n={n_val}', ha='center', va='bottom', fontsize=6)
    axes[0].axhline(0.25, color='red', linestyle=':', linewidth=0.8, alpha=0.7)
    axes[0].set_xlabel('Confidence Level')
    axes[0].set_ylabel('Accuracy')
    axes[0].set_title('Accuracy by Confidence', fontsize=9, fontweight='bold')
    axes[0].set_ylim(0, min(1.0, max(conf_y) + 0.15))

    axes[1].bar(fam_x, fam_y, color=COLORS['expert'], edgecolor='white', linewidth=0.5)
    for i, (x_val, y_val, n_val) in enumerate(zip(fam_x, fam_y, fam_n)):
        axes[1].text(x_val, y_val + 0.01, f'n={n_val}', ha='center', va='bottom', fontsize=6)
    axes[1].axhline(0.25, color='red', linestyle=':', linewidth=0.8, alpha=0.7)
    axes[1].set_xlabel('Familiarity Level')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title('Accuracy by Familiarity', fontsize=9, fontweight='bold')
    axes[1].set_ylim(0, min(1.0, max(fam_y) + 0.15))

    fig_conf.tight_layout()
    pdf_path, _ = save_figure(fig_conf, 'fig_s6_expert_confidence_accuracy')
    plt.close(fig_conf)
    print(f"  Saved: {pdf_path}")

    # ------------------------------------------------------------------
    # 6. Mediocrity bias
    # ------------------------------------------------------------------
    print("\n[8/9] Analyzing mediocrity bias...")

    actual_dist = Counter(y_true)
    pred_dist = Counter(y_pred)

    print("  Actual distribution vs Predicted distribution:")
    for label in LABEL_ORDER:
        a_count = actual_dist.get(label, 0)
        p_count = pred_dist.get(label, 0)
        n_total = len(y_true)
        print(f"    {label:12s}: actual={a_count} ({a_count/n_total:.1%})  "
              f"predicted={p_count} ({p_count/n_total:.1%})")

    # Check if middle tiers (strong, fair) are over-predicted
    middle_actual = actual_dist.get('strong', 0) + actual_dist.get('fair', 0)
    middle_pred = pred_dist.get('strong', 0) + pred_dist.get('fair', 0)
    print(f"  Middle tiers (strong+fair): actual={middle_actual}, predicted={middle_pred}")

    # ------------------------------------------------------------------
    # 7. Statistical tests
    # ------------------------------------------------------------------
    print("\n[9/9] Running statistical tests...")

    stats_results = {}

    # Spearman: confidence vs accuracy (per-rating level)
    conf_vals = []
    acc_vals = []
    for rec in individual_records:
        if rec['confidence'] is not None:
            conf_vals.append(float(rec['confidence']))
            acc_vals.append(float(rec['correct']))

    if len(conf_vals) > 2:
        spearman_conf = spearman_correlation(conf_vals, acc_vals)
        stats_results['spearman_confidence_accuracy'] = spearman_conf
        print(f"  Spearman (confidence vs accuracy): rho={spearman_conf['correlation']:.4f}, "
              f"p={spearman_conf['p_value']:.4e}")

    # Spearman: familiarity vs accuracy
    fam_vals = []
    acc_vals_fam = []
    for rec in individual_records:
        if rec['familiarity'] is not None:
            fam_vals.append(float(rec['familiarity']))
            acc_vals_fam.append(float(rec['correct']))

    if len(fam_vals) > 2:
        spearman_fam = spearman_correlation(fam_vals, acc_vals_fam)
        stats_results['spearman_familiarity_accuracy'] = spearman_fam
        print(f"  Spearman (familiarity vs accuracy): rho={spearman_fam['correlation']:.4f}, "
              f"p={spearman_fam['p_value']:.4e}")

    # Mann-Whitney: read_before effect
    read_yes = [float(r['correct']) for r in individual_records if r['read_before']]
    read_no = [float(r['correct']) for r in individual_records if not r['read_before']]
    if len(read_yes) > 1 and len(read_no) > 1:
        mw_read = mann_whitney_u(read_yes, read_no)
        stats_results['mann_whitney_read_before'] = mw_read
        print(f"  Mann-Whitney (read_before): U={mw_read['statistic']:.1f}, "
              f"p={mw_read['p_value']:.4e}, r={mw_read['rank_biserial']:.4f}")

    # One-sample t-test: expert accuracy vs chance (25%)
    ttest = one_sample_ttest(expert_accuracies, popmean=0.25)
    stats_results['ttest_vs_chance'] = ttest
    print(f"  One-sample t-test vs 25%: t={ttest['statistic']:.3f}, "
          f"p={ttest['p_value']:.4e}, d={ttest['cohens_d']:.3f}")

    # Summary statistics
    stats_results['majority_vote'] = {
        policy: {
            'accuracy': res['accuracy'],
            'macro_f1': res['macro_f1'],
            'n_articles': res['n_articles'],
            'n_ties': res['n_ties'],
            'n_excluded': res['n_excluded'],
        }
        for policy, res in results_by_policy.items()
    }

    stats_results['per_class_metrics'] = {
        label: {k: float(v) for k, v in per_class[label].items()}
        for label in LABEL_ORDER
    }

    stats_results['tier_accuracy'] = {k: float(v) for k, v in tier_accuracy.items()}

    stats_results['individual_expert_stats'] = {
        'n_experts': len(expert_stats_list),
        'mean_accuracy': float(np.mean(expert_accuracies)),
        'median_accuracy': float(np.median(expert_accuracies)),
        'std_accuracy': float(np.std(expert_accuracies, ddof=1)),
        'min_accuracy': float(np.min(expert_accuracies)),
        'max_accuracy': float(np.max(expert_accuracies)),
    }

    stats_results['confidence_accuracy'] = {
        str(k): {'accuracy': float(v), 'n': int(len(conf_groups[k]))}
        for k, v in conf_accuracy.items()
    }
    stats_results['familiarity_accuracy'] = {
        str(k): {'accuracy': float(v), 'n': int(len(fam_groups[k]))}
        for k, v in fam_accuracy.items()
    }
    stats_results['read_before_accuracy'] = {
        key: {'accuracy': float(np.mean(vals)), 'n': len(vals)}
        for key, vals in read_groups.items()
    }

    stats_results['mediocrity_bias'] = {
        'actual_distribution': {label: int(actual_dist.get(label, 0)) for label in LABEL_ORDER},
        'predicted_distribution': {label: int(pred_dist.get(label, 0)) for label in LABEL_ORDER},
    }

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print("\nSaving outputs...")

    # Table 5: expert performance
    table5_rows = []
    table5_rows.append({
        'metric': 'Overall Accuracy',
        'value': f"{primary['accuracy']:.4f}",
    })
    table5_rows.append({
        'metric': 'Macro F1',
        'value': f"{primary['macro_f1']:.4f}",
    })
    for label in LABEL_ORDER:
        table5_rows.append({
            'metric': f'{label} Accuracy',
            'value': f"{tier_accuracy[label]:.4f}",
        })
    for label in LABEL_ORDER:
        m = per_class[label]
        table5_rows.append({
            'metric': f'{label} Precision',
            'value': f"{m['precision']:.4f}",
        })
        table5_rows.append({
            'metric': f'{label} Recall',
            'value': f"{m['recall']:.4f}",
        })
        table5_rows.append({
            'metric': f'{label} F1',
            'value': f"{m['f1']:.4f}",
        })

    df_table5 = pd.DataFrame(table5_rows)
    table5_path = root / TABLES_DIR / 'table5_expert_performance.csv'
    df_table5.to_csv(table5_path, index=False)
    print(f"  Saved: {table5_path}")

    # Table S2: individual expert stats
    df_expert = pd.DataFrame(expert_stats_list)
    table_s2_path = root / TABLES_DIR / 'table_s2_individual_expert.csv'
    df_expert.to_csv(table_s2_path, index=False)
    print(f"  Saved: {table_s2_path}")

    # Statistics JSON
    stats_path = root / STATS_DIR / '04_expert.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {stats_path}")

    print("\n" + "=" * 70)
    print("Script 04 complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
