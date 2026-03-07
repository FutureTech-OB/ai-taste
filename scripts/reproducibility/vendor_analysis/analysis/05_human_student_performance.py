#!/usr/bin/env python3
"""Script 05: Human Student Performance Analysis.

Analyzes student rater performance on the 120-article research quality
dataset. Uses all students (old + new combined from student_merged.jsonl).
Old/new split is a data collection artifact with no significant difference
(p=0.319). Students only -- experts are analyzed in script 04.

Outputs:
    results/tables/table6_student_performance.csv
    results/tables/table_s3_monte_carlo.csv
    results/figures/fig6_monte_carlo_accuracy_curve.pdf
    results/figures/fig_s7_student_distribution.pdf
    results/figures/fig_s8_student_confusion.pdf
    results/statistics/05_student.json
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
    print("Script 05: Human Student Performance Analysis")
    print("=" * 70)

    _ensure_dirs()
    root = get_project_root()
    set_nature_style()

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    print("\n[1/8] Loading data...")
    student_merged = load_student_merged_ratings()
    print(f"  Loaded {len(student_merged)} articles from student_merged.jsonl")

    # ------------------------------------------------------------------
    # Extract all individual student ratings from merged data
    # ------------------------------------------------------------------
    print("\n[2/8] Extracting individual ratings from merged data...")

    article_ratings = []
    individual_records = []

    for article in student_merged:
        gt = normalize_level(article['level'])
        ratings_list = article.get('ratings', [])
        preds = []
        for r in ratings_list:
            pred = normalize_level(r['q2_rating'])
            rater = r.get('rater_id', r.get('rater_name', 'unknown'))
            conf = _parse_confidence(r.get('q3_confidence'))
            fam = _parse_familiarity(r.get('q4_familiarity'))
            read_before_raw = r.get('q1_read_before', 'No')
            is_read = (read_before_raw.strip().lower() == 'yes'
                       if isinstance(read_before_raw, str) else False)
            preds.append(pred)
            rec = {
                'rater': rater,
                'ground_truth': gt,
                'prediction': pred,
                'correct': int(pred == gt),
                'confidence': conf,
                'familiarity': fam,
                'read_before': is_read,
                'article_number': article.get('article_number'),
                'package': article.get('package'),
            }
            individual_records.append(rec)

        article_ratings.append({
            'gt': gt,
            'predictions': preds,
            'title': article.get('title', ''),
            'article_number': article.get('article_number'),
            'package': article.get('package'),
            'n_raters': len(preds),
        })

    total_ratings = len(individual_records)
    unique_students = set(r['rater'] for r in individual_records)
    print(f"  Total individual ratings (merged): {total_ratings}")
    print(f"  Unique students (merged): {len(unique_students)}")
    raters_per_article = [a['n_raters'] for a in article_ratings]
    print(f"  Raters per article: min={min(raters_per_article)}, "
          f"max={max(raters_per_article)}, mean={np.mean(raters_per_article):.1f}")

    # ------------------------------------------------------------------
    # 1. Student majority vote accuracy (all students from merged)
    # ------------------------------------------------------------------
    print("\n[3/8] Computing majority vote accuracy...")

    rng = random.Random(42)
    y_true = []
    y_pred = []
    n_ties = 0

    for art in article_ratings:
        vote, is_tie = majority_vote(art['predictions'], tie_policy='exclude', rng=rng)
        if is_tie:
            n_ties += 1
        if vote is None:
            continue
        y_true.append(art['gt'])
        y_pred.append(vote)

    acc = compute_accuracy(y_true, y_pred)
    macro_f1 = compute_macro_f1(y_true, y_pred)
    print(f"  Overall accuracy (exclude ties): {acc:.4f}")
    print(f"  Macro F1: {macro_f1:.4f}")
    print(f"  Ties: {n_ties}, excluded: {len(article_ratings) - len(y_true)}")

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
    # 2. Monte Carlo subsampling
    # ------------------------------------------------------------------
    print("\n[4/8] Running Monte Carlo subsampling...")

    sample_sizes = [2, 3, 4, 5, 8, 10, 15, 20]
    mc_results = []

    for k in sample_sizes:
        print(f"  k={k} ...", end='', flush=True)
        np_rng = np.random.RandomState(42)
        py_rng = random.Random(42)
        n_trials = 5000
        trial_accuracies = []

        for trial in range(n_trials):
            trial_true = []
            trial_pred = []
            n_skipped = 0

            for art in article_ratings:
                n_available = len(art['predictions'])
                if n_available < k:
                    n_skipped += 1
                    continue
                indices = np_rng.choice(n_available, size=k, replace=False)
                sampled = [art['predictions'][i] for i in indices]
                vote, is_tie = majority_vote(sampled, tie_policy='exclude', rng=py_rng)
                if vote is None:
                    n_skipped += 1
                    continue
                trial_true.append(art['gt'])
                trial_pred.append(vote)

            if trial_true:
                trial_acc = compute_accuracy(trial_true, trial_pred)
                trial_accuracies.append(trial_acc)

        acc_array = np.array(trial_accuracies)
        mean_acc = float(np.mean(acc_array))
        std_acc = float(np.std(acc_array, ddof=1))
        ci_lower = float(np.percentile(acc_array, 2.5))
        ci_upper = float(np.percentile(acc_array, 97.5))

        mc_results.append({
            'k': k,
            'mean_accuracy': mean_acc,
            'std': std_acc,
            'ci_lower': ci_lower,
            'ci_upper': ci_upper,
            'n_trials': len(trial_accuracies),
        })
        print(f" accuracy={mean_acc:.4f} [{ci_lower:.4f}, {ci_upper:.4f}]")

    # Plot Monte Carlo accuracy curve
    mc_x = [r['k'] for r in mc_results]
    mc_y = [r['mean_accuracy'] for r in mc_results]
    mc_ci = [(r['ci_lower'], r['ci_upper']) for r in mc_results]

    fig_mc, ax_mc = plot_accuracy_curve(
        mc_x, mc_y, mc_ci,
        title='Student Majority Vote Accuracy vs Sample Size',
        xlabel='Number of students (k)',
        ylabel='Accuracy',
    )
    ax_mc.axhline(0.25, color='gray', linestyle=':', linewidth=0.8, alpha=0.7)
    ax_mc.set_xticks(mc_x)
    fig_mc.tight_layout()
    pdf_path, _ = save_figure(fig_mc, 'fig6_monte_carlo_accuracy_curve')
    plt.close(fig_mc)
    print(f"  Saved: {pdf_path}")

    # ------------------------------------------------------------------
    # 3. Per-class metrics and confusion matrix
    # ------------------------------------------------------------------
    print("\n[5/8] Computing per-class metrics and confusion matrix...")

    per_class = compute_per_class_metrics(y_true, y_pred)
    for label in LABEL_ORDER:
        m = per_class[label]
        print(f"  {label:12s}: P={m['precision']:.3f}  R={m['recall']:.3f}  "
              f"F1={m['f1']:.3f}  support={int(m['support'])}")

    cm = compute_confusion_matrix(y_true, y_pred)
    fig_cm, ax_cm = plot_confusion_matrix(
        cm, LABEL_ORDER,
        title='Student Majority Vote Confusion Matrix',
        normalize=True,
    )
    pdf_path, _ = save_figure(fig_cm, 'fig_s8_student_confusion')
    plt.close(fig_cm)
    print(f"  Saved: {pdf_path}")

    # ------------------------------------------------------------------
    # 4. Individual student accuracy
    # ------------------------------------------------------------------
    print("\n[6/8] Computing individual student accuracy...")

    student_articles = defaultdict(list)
    for rec in individual_records:
        student_articles[rec['rater']].append(rec)

    student_stats_list = []
    for rater, recs in student_articles.items():
        n_rated = len(recs)
        n_correct = sum(r['correct'] for r in recs)
        acc_val = n_correct / n_rated if n_rated > 0 else 0.0
        student_stats_list.append({
            'student': rater,
            'n_articles': n_rated,
            'n_correct': n_correct,
            'accuracy': acc_val,
        })

    student_stats_list.sort(key=lambda x: x['accuracy'], reverse=True)
    student_accuracies = [s['accuracy'] for s in student_stats_list]

    print(f"  Number of students: {len(student_stats_list)}")
    print(f"  Mean accuracy:   {np.mean(student_accuracies):.4f}")
    print(f"  Median accuracy: {np.median(student_accuracies):.4f}")
    print(f"  Std accuracy:    {np.std(student_accuracies, ddof=1):.4f}")
    print(f"  Min accuracy:    {np.min(student_accuracies):.4f}")
    print(f"  Max accuracy:    {np.max(student_accuracies):.4f}")

    # Histogram
    fig_hist, ax_hist = plt.subplots(figsize=(4.0, 3.0))
    ax_hist.hist(student_accuracies, bins=20, color=COLORS['student'],
                 edgecolor='white', linewidth=0.5, alpha=0.8)
    ax_hist.axvline(np.mean(student_accuracies), color='black',
                    linestyle='--', linewidth=0.8,
                    label=f'Mean={np.mean(student_accuracies):.2f}')
    ax_hist.axvline(0.25, color='red', linestyle=':', linewidth=0.8,
                    label='Chance (25%)')
    ax_hist.set_xlabel('Accuracy')
    ax_hist.set_ylabel('Number of students')
    ax_hist.set_title('Individual Student Accuracy Distribution', fontsize=9, fontweight='bold')
    ax_hist.legend(fontsize=7)
    fig_hist.tight_layout()
    pdf_path, _ = save_figure(fig_hist, 'fig_s7_student_distribution')
    plt.close(fig_hist)
    print(f"  Saved: {pdf_path}")

    # ------------------------------------------------------------------
    # 5. Accuracy by confidence/familiarity (all students)
    # ------------------------------------------------------------------
    print("\n[7/8] Computing accuracy by confidence/familiarity (all students)...")

    conf_groups = defaultdict(list)
    for rec in individual_records:
        if rec['confidence'] is not None:
            conf_groups[rec['confidence']].append(rec['correct'])

    conf_accuracy = {}
    for level in sorted(conf_groups.keys()):
        vals = conf_groups[level]
        acc_val = np.mean(vals)
        conf_accuracy[level] = acc_val
        print(f"  Confidence {level}: accuracy={acc_val:.3f} (n={len(vals)})")

    fam_groups = defaultdict(list)
    for rec in individual_records:
        if rec['familiarity'] is not None:
            fam_groups[rec['familiarity']].append(rec['correct'])

    fam_accuracy = {}
    for level in sorted(fam_groups.keys()):
        vals = fam_groups[level]
        acc_val = np.mean(vals)
        fam_accuracy[level] = acc_val
        print(f"  Familiarity {level}: accuracy={acc_val:.3f} (n={len(vals)})")

    # ------------------------------------------------------------------
    # 6. Mediocrity bias
    # ------------------------------------------------------------------
    print("\n  Mediocrity bias analysis...")

    actual_dist = Counter(y_true)
    pred_dist = Counter(y_pred)

    print("  Actual distribution vs Predicted distribution:")
    for label in LABEL_ORDER:
        a_count = actual_dist.get(label, 0)
        p_count = pred_dist.get(label, 0)
        n_total = len(y_true)
        print(f"    {label:12s}: actual={a_count} ({a_count/n_total:.1%})  "
              f"predicted={p_count} ({p_count/n_total:.1%})")

    # ------------------------------------------------------------------
    # 7. Statistical tests
    # ------------------------------------------------------------------
    print("\n[8/8] Running statistical tests...")

    stats_results = {}

    # Spearman: confidence vs accuracy (all students)
    conf_vals = [float(r['confidence']) for r in individual_records
                 if r['confidence'] is not None]
    acc_vals_conf = [float(r['correct']) for r in individual_records
                     if r['confidence'] is not None]

    if len(conf_vals) > 2:
        spearman_conf = spearman_correlation(conf_vals, acc_vals_conf)
        stats_results['spearman_confidence_accuracy'] = spearman_conf
        print(f"  Spearman (confidence vs accuracy): rho={spearman_conf['correlation']:.4f}, "
              f"p={spearman_conf['p_value']:.4e}")

    # Spearman: familiarity vs accuracy (all students)
    fam_vals = [float(r['familiarity']) for r in individual_records
                if r['familiarity'] is not None]
    acc_vals_fam = [float(r['correct']) for r in individual_records
                    if r['familiarity'] is not None]

    if len(fam_vals) > 2:
        spearman_fam = spearman_correlation(fam_vals, acc_vals_fam)
        stats_results['spearman_familiarity_accuracy'] = spearman_fam
        print(f"  Spearman (familiarity vs accuracy): rho={spearman_fam['correlation']:.4f}, "
              f"p={spearman_fam['p_value']:.4e}")

    # Mann-Whitney: read_before effect (all students)
    read_yes = [float(r['correct']) for r in individual_records if r['read_before']]
    read_no = [float(r['correct']) for r in individual_records if not r['read_before']]
    if len(read_yes) > 1 and len(read_no) > 1:
        mw_read = mann_whitney_u(read_yes, read_no)
        stats_results['mann_whitney_read_before'] = mw_read
        print(f"  Mann-Whitney (read_before): U={mw_read['statistic']:.1f}, "
              f"p={mw_read['p_value']:.4e}")

    # One-sample t-test: student accuracy vs chance (25%)
    ttest = one_sample_ttest(student_accuracies, popmean=0.25)
    stats_results['ttest_vs_chance'] = ttest
    print(f"  One-sample t-test vs 25%: t={ttest['statistic']:.3f}, "
          f"p={ttest['p_value']:.4e}, d={ttest['cohens_d']:.3f}")

    # Store summary
    stats_results['majority_vote'] = {
        'accuracy': float(acc),
        'macro_f1': float(macro_f1),
        'n_articles': len(y_true),
        'n_ties': n_ties,
    }

    stats_results['per_class_metrics'] = {
        label: {k: float(v) for k, v in per_class[label].items()}
        for label in LABEL_ORDER
    }

    stats_results['tier_accuracy'] = {k: float(v) for k, v in tier_accuracy.items()}

    stats_results['individual_student_stats'] = {
        'n_students': len(student_stats_list),
        'mean_accuracy': float(np.mean(student_accuracies)),
        'median_accuracy': float(np.median(student_accuracies)),
        'std_accuracy': float(np.std(student_accuracies, ddof=1)),
        'min_accuracy': float(np.min(student_accuracies)),
        'max_accuracy': float(np.max(student_accuracies)),
    }

    stats_results['monte_carlo'] = [
        {k: v for k, v in r.items()}
        for r in mc_results
    ]

    stats_results['confidence_accuracy'] = {
        str(k): {'accuracy': float(v), 'n': int(len(conf_groups[k]))}
        for k, v in conf_accuracy.items()
    }
    stats_results['familiarity_accuracy'] = {
        str(k): {'accuracy': float(v), 'n': int(len(fam_groups[k]))}
        for k, v in fam_accuracy.items()
    }

    stats_results['mediocrity_bias'] = {
        'actual_distribution': {label: int(actual_dist.get(label, 0)) for label in LABEL_ORDER},
        'predicted_distribution': {label: int(pred_dist.get(label, 0)) for label in LABEL_ORDER},
    }

    # ------------------------------------------------------------------
    # Save outputs
    # ------------------------------------------------------------------
    print("\nSaving outputs...")

    # Table 6: student performance
    table6_rows = []
    table6_rows.append({'metric': 'Overall Accuracy', 'value': f"{acc:.4f}"})
    table6_rows.append({'metric': 'Macro F1', 'value': f"{macro_f1:.4f}"})
    for label in LABEL_ORDER:
        table6_rows.append({
            'metric': f'{label} Accuracy',
            'value': f"{tier_accuracy[label]:.4f}",
        })
    for label in LABEL_ORDER:
        m = per_class[label]
        table6_rows.append({
            'metric': f'{label} Precision', 'value': f"{m['precision']:.4f}",
        })
        table6_rows.append({
            'metric': f'{label} Recall', 'value': f"{m['recall']:.4f}",
        })
        table6_rows.append({
            'metric': f'{label} F1', 'value': f"{m['f1']:.4f}",
        })

    df_table6 = pd.DataFrame(table6_rows)
    table6_path = root / TABLES_DIR / 'table6_student_performance.csv'
    df_table6.to_csv(table6_path, index=False)
    print(f"  Saved: {table6_path}")

    # Table S3: Monte Carlo results
    df_mc = pd.DataFrame(mc_results)
    table_s3_path = root / TABLES_DIR / 'table_s3_monte_carlo.csv'
    df_mc.to_csv(table_s3_path, index=False)
    print(f"  Saved: {table_s3_path}")

    # Statistics JSON
    stats_path = root / STATS_DIR / '05_student.json'
    with open(stats_path, 'w', encoding='utf-8') as f:
        json.dump(stats_results, f, indent=2, ensure_ascii=False, default=str)
    print(f"  Saved: {stats_path}")

    print("\n" + "=" * 70)
    print("Script 05 complete.")
    print("=" * 70)


if __name__ == '__main__':
    main()
