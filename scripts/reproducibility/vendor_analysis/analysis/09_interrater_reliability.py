#!/usr/bin/env python3
"""Script 09: Inter-rater reliability analysis.

Computes Fleiss' kappa and Krippendorff's alpha for expert and student
raters, examines agreement by tier and domain, and compares reliability
between rater groups using bootstrap CIs and permutation tests.

Outputs:
    - fig_s12_reliability_comparison.pdf
    - table_s6_interrater_reliability.csv
    - 09_interrater.json
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

CATEGORY_LIST = LABEL_ORDER  # ['exceptional', 'strong', 'fair', 'limited']


def _build_fleiss_matrix(articles, rater_type_filter=None):
    """Build a Fleiss' kappa rating matrix from article records.

    Returns:
        ratings_matrix: np.ndarray of shape (n_articles, 4) where columns
            are counts of raters assigning each of the 4 categories.
        article_meta: list of dicts with article_number, level, domain.
    """
    cat_to_idx = {cat: i for i, cat in enumerate(CATEGORY_LIST)}
    rows = []
    meta = []

    for art in articles:
        surveys = art.get('ratings', art.get('surveys', []))
        counts = np.zeros(len(CATEGORY_LIST), dtype=int)
        n_valid = 0

        for entry in surveys:
            if rater_type_filter and entry.get('rater_type') != rater_type_filter:
                continue
            raw = entry.get('q2_rating', entry.get('rating', ''))
            label = normalize_level(raw)
            if label in cat_to_idx:
                counts[cat_to_idx[label]] += 1
                n_valid += 1

        if n_valid >= 2:
            rows.append(counts)
            meta.append({
                'article_number': art.get('article_number'),
                'level': normalize_level(art.get('level', '')),
                'domain': art.get('domain', ''),
            })

    if not rows:
        return np.zeros((0, len(CATEGORY_LIST)), dtype=int), []

    return np.array(rows, dtype=int), meta


def _build_krippendorff_matrix(articles, rater_type_filter=None):
    """Build a rater x unit matrix for Krippendorff's alpha.

    Each cell contains an ordinal value (0-3 mapping to CATEGORY_LIST)
    or np.nan for missing.

    Returns:
        data: np.ndarray of shape (n_raters, n_articles).
    """
    cat_to_ord = {cat: i for i, cat in enumerate(CATEGORY_LIST)}

    # Collect all unique rater names
    all_raters = []
    for art in articles:
        surveys = art.get('ratings', art.get('surveys', []))
        for entry in surveys:
            if rater_type_filter and entry.get('rater_type') != rater_type_filter:
                continue
            name = entry.get('rater_id', entry.get('rater_name', entry.get('name', 'unknown')))
            if name not in all_raters:
                all_raters.append(name)

    rater_to_idx = {name: i for i, name in enumerate(all_raters)}
    n_raters = len(all_raters)
    n_articles = len(articles)

    data = np.full((n_raters, n_articles), np.nan)

    for art_idx, art in enumerate(articles):
        surveys = art.get('ratings', art.get('surveys', []))
        for entry in surveys:
            if rater_type_filter and entry.get('rater_type') != rater_type_filter:
                continue
            name = entry.get('rater_id', entry.get('rater_name', entry.get('name', 'unknown')))
            raw = entry.get('q2_rating', entry.get('rating', ''))
            label = normalize_level(raw)
            if label in cat_to_ord and name in rater_to_idx:
                data[rater_to_idx[name], art_idx] = float(cat_to_ord[label])

    return data


def fleiss_kappa_simple(ratings_matrix):
    """Compute Fleiss' kappa from a ratings matrix.

    Parameters:
        ratings_matrix: np.ndarray of shape (n_subjects, n_categories).
            Cell value = count of raters choosing that category.

    Returns:
        float: kappa value.
    """
    table = np.asarray(ratings_matrix, dtype=float)
    n, k = table.shape
    N_per_row = table.sum(axis=1)

    # Filter out rows with fewer than 2 raters
    valid = N_per_row >= 2
    if valid.sum() < 2:
        return float('nan')

    table = table[valid]
    N_per_row = N_per_row[valid]
    n = table.shape[0]

    # Proportion per category (weighted by number of ratings)
    total_ratings = table.sum()
    p_j = table.sum(axis=0) / total_ratings

    # Per-subject agreement
    P_i = (np.sum(table ** 2, axis=1) - N_per_row) / (
        N_per_row * (N_per_row - 1)
    )

    P_bar = float(np.mean(P_i))
    P_e = float(np.sum(p_j ** 2))

    if abs(1.0 - P_e) < 1e-12:
        return 1.0 if abs(P_bar - 1.0) < 1e-12 else 0.0

    return (P_bar - P_e) / (1.0 - P_e)


def _pairwise_agreement_rate(articles, rater_type_filter=None):
    """Compute what fraction of rater pairs agree on each article.

    Returns:
        per_article: list of (article_number, agreement_rate, n_pairs).
        overall: float, weighted mean agreement rate.
    """
    results = []
    total_agree = 0
    total_pairs = 0

    for art in articles:
        surveys = art.get('ratings', art.get('surveys', []))
        labels = []
        for entry in surveys:
            if rater_type_filter and entry.get('rater_type') != rater_type_filter:
                continue
            raw = entry.get('q2_rating', entry.get('rating', ''))
            label = normalize_level(raw)
            if label in set(CATEGORY_LIST):
                labels.append(label)

        n = len(labels)
        if n < 2:
            continue

        n_pairs_local = n * (n - 1) // 2
        agree_local = 0
        for i in range(n):
            for j in range(i + 1, n):
                if labels[i] == labels[j]:
                    agree_local += 1

        rate = agree_local / n_pairs_local if n_pairs_local > 0 else 0.0
        results.append({
            'article_number': art.get('article_number'),
            'agreement_rate': rate,
            'n_pairs': n_pairs_local,
        })
        total_agree += agree_local
        total_pairs += n_pairs_local

    overall = total_agree / total_pairs if total_pairs > 0 else 0.0
    return results, overall


def _get_primary_domain(domain_str):
    """Extract the first domain from a comma-separated domain string."""
    if not domain_str:
        return 'Unknown'
    parts = [p.strip() for p in domain_str.split(',')]
    return parts[0] if parts else 'Unknown'


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
    expert_data = load_expert_ratings()
    student_merged_data = load_student_merged_ratings()

    # ---- 1: Fleiss' kappa / Krippendorff's alpha for experts ------------
    expert_fleiss_mat, expert_meta = _build_fleiss_matrix(expert_data)
    expert_kappa = fleiss_kappa_simple(expert_fleiss_mat)

    expert_kripp_mat = _build_krippendorff_matrix(expert_data)
    expert_alpha_result = krippendorff_alpha(
        expert_kripp_mat.tolist(), level='ordinal'
    )
    expert_alpha = expert_alpha_result['alpha']

    # ---- 2: Fleiss' kappa / Krippendorff's alpha for students -----------
    student_fleiss_mat, student_meta = _build_fleiss_matrix(student_merged_data)
    student_kappa = fleiss_kappa_simple(student_fleiss_mat)

    student_kripp_mat = _build_krippendorff_matrix(student_merged_data)
    student_alpha_result = krippendorff_alpha(
        student_kripp_mat.tolist(), level='ordinal'
    )
    student_alpha = student_alpha_result['alpha']

    # ---- 3: Agreement by tier -------------------------------------------
    tier_reliability = {}
    for tier in LABEL_ORDER:
        # Expert subset
        tier_indices_exp = [
            i for i, m in enumerate(expert_meta) if m['level'] == tier
        ]
        if len(tier_indices_exp) >= 2:
            tier_mat_exp = expert_fleiss_mat[tier_indices_exp]
            tier_kappa_exp = fleiss_kappa_simple(tier_mat_exp)
        else:
            tier_kappa_exp = float('nan')

        # Student subset
        tier_indices_stu = [
            i for i, m in enumerate(student_meta) if m['level'] == tier
        ]
        if len(tier_indices_stu) >= 2:
            tier_mat_stu = student_fleiss_mat[tier_indices_stu]
            tier_kappa_stu = fleiss_kappa_simple(tier_mat_stu)
        else:
            tier_kappa_stu = float('nan')

        tier_reliability[tier] = {
            'expert_kappa': tier_kappa_exp,
            'expert_n': len(tier_indices_exp),
            'student_kappa': tier_kappa_stu,
            'student_n': len(tier_indices_stu),
        }

    # ---- 4: Agreement by domain -----------------------------------------
    domain_reliability = {}
    # Use primary domain for grouping
    exp_domains = defaultdict(list)
    for i, m in enumerate(expert_meta):
        primary = _get_primary_domain(m['domain'])
        exp_domains[primary].append(i)

    stu_domains = defaultdict(list)
    for i, m in enumerate(student_meta):
        primary = _get_primary_domain(m['domain'])
        stu_domains[primary].append(i)

    all_domains = sorted(set(list(exp_domains.keys()) + list(stu_domains.keys())))
    for domain in all_domains:
        exp_idx = exp_domains.get(domain, [])
        stu_idx = stu_domains.get(domain, [])

        if len(exp_idx) >= 2:
            d_mat_exp = expert_fleiss_mat[exp_idx]
            d_kappa_exp = fleiss_kappa_simple(d_mat_exp)
        else:
            d_kappa_exp = float('nan')

        if len(stu_idx) >= 2:
            d_mat_stu = student_fleiss_mat[stu_idx]
            d_kappa_stu = fleiss_kappa_simple(d_mat_stu)
        else:
            d_kappa_stu = float('nan')

        domain_reliability[domain] = {
            'expert_kappa': d_kappa_exp,
            'expert_n': len(exp_idx),
            'student_kappa': d_kappa_stu,
            'student_n': len(stu_idx),
        }

    # ---- 5: Expert vs student reliability comparison --------------------
    reliability_comparison = {
        'expert_fleiss_kappa': expert_kappa,
        'expert_krippendorff_alpha': expert_alpha,
        'student_fleiss_kappa': student_kappa,
        'student_krippendorff_alpha': student_alpha,
        'kappa_difference': expert_kappa - student_kappa,
        'alpha_difference': expert_alpha - student_alpha,
    }

    # ---- 6: Pairwise agreement rates ------------------------------------
    expert_pw, expert_pw_overall = _pairwise_agreement_rate(expert_data)
    student_pw, student_pw_overall = _pairwise_agreement_rate(
        student_merged_data
    )

    # ---- 7: Bootstrap CI and permutation test ---------------------------
    # Bootstrap CI for expert kappa
    def _bootstrap_kappa(matrix, n_bootstrap=5000, seed=42):
        """Bootstrap CI for Fleiss' kappa."""
        rng = np.random.RandomState(seed)
        n_subjects = matrix.shape[0]
        boot_kappas = []
        for _ in range(n_bootstrap):
            idx = rng.choice(n_subjects, size=n_subjects, replace=True)
            boot_mat = matrix[idx]
            k = fleiss_kappa_simple(boot_mat)
            if not np.isnan(k):
                boot_kappas.append(k)
        if not boot_kappas:
            return {'ci_lower': float('nan'), 'ci_upper': float('nan'),
                    'std': float('nan')}
        arr = np.array(boot_kappas)
        return {
            'ci_lower': float(np.percentile(arr, 2.5)),
            'ci_upper': float(np.percentile(arr, 97.5)),
            'std': float(np.std(arr, ddof=1)),
        }

    expert_kappa_ci = _bootstrap_kappa(expert_fleiss_mat)
    student_kappa_ci = _bootstrap_kappa(student_fleiss_mat)

    # Permutation test: is expert kappa significantly different from student?
    def _permutation_kappa_diff(exp_mat, stu_mat, n_perms=5000, seed=42):
        """Permutation test for difference in Fleiss' kappa."""
        rng = np.random.RandomState(seed)
        observed_diff = fleiss_kappa_simple(exp_mat) - fleiss_kappa_simple(stu_mat)

        # To permute: combine all articles, shuffle group assignments
        # Both matrices must have same number of columns (categories)
        # But they may have different numbers of raters per row.
        # We use per-article pairwise agreement as the statistic instead.
        exp_pw_rates = []
        for row in exp_mat:
            n_r = row.sum()
            if n_r < 2:
                continue
            agree = float(np.sum(row ** 2) - n_r) / (n_r * (n_r - 1))
            exp_pw_rates.append(agree)

        stu_pw_rates = []
        for row in stu_mat:
            n_r = row.sum()
            if n_r < 2:
                continue
            agree = float(np.sum(row ** 2) - n_r) / (n_r * (n_r - 1))
            stu_pw_rates.append(agree)

        exp_arr = np.array(exp_pw_rates)
        stu_arr = np.array(stu_pw_rates)
        observed_mean_diff = float(np.mean(exp_arr) - np.mean(stu_arr))

        combined = np.concatenate([exp_arr, stu_arr])
        n_exp = len(exp_arr)
        count_extreme = 0

        for _ in range(n_perms):
            rng.shuffle(combined)
            perm_diff = float(np.mean(combined[:n_exp]) - np.mean(combined[n_exp:]))
            if abs(perm_diff) >= abs(observed_mean_diff):
                count_extreme += 1

        p_value = (count_extreme + 1) / (n_perms + 1)

        return {
            'observed_kappa_diff': float(observed_diff),
            'observed_agreement_diff': observed_mean_diff,
            'p_value': float(p_value),
            'n_permutations': n_perms,
        }

    perm_result = _permutation_kappa_diff(expert_fleiss_mat, student_fleiss_mat)

    # ---- Figure: Reliability comparison ---------------------------------
    set_nature_style()
    fig, axes = plt.subplots(1, 3, figsize=(9.0, 3.5))

    # Panel A: Overall kappa comparison
    ax_a = axes[0]
    groups = ['Expert', 'Student']
    kappas = [expert_kappa, student_kappa]
    ci_lows = [expert_kappa_ci['ci_lower'], student_kappa_ci['ci_lower']]
    ci_highs = [expert_kappa_ci['ci_upper'], student_kappa_ci['ci_upper']]
    colors_bar = [COLORS['expert'], COLORS['student']]
    x_pos = np.arange(len(groups))
    bars = ax_a.bar(x_pos, kappas, width=0.5, color=colors_bar,
                    edgecolor='white', linewidth=0.5)
    errs_low = [k - cl for k, cl in zip(kappas, ci_lows)]
    errs_high = [ch - k for k, ch in zip(kappas, ci_highs)]
    ax_a.errorbar(x_pos, kappas,
                  yerr=[errs_low, errs_high],
                  fmt='none', ecolor='black', capsize=3, linewidth=0.8)
    ax_a.set_xticks(x_pos)
    ax_a.set_xticklabels(groups)
    ax_a.set_ylabel("Fleiss' kappa")
    ax_a.set_title('a  Overall reliability', fontsize=9,
                    fontweight='bold', loc='left')
    ax_a.set_ylim(0, max(ci_highs) * 1.3 if max(ci_highs) > 0 else 0.5)
    for i, (k, g) in enumerate(zip(kappas, groups)):
        ax_a.text(i, k + errs_high[i] + 0.01, f'{k:.3f}',
                  ha='center', va='bottom', fontsize=7)

    # Panel B: Kappa by tier
    ax_b = axes[1]
    tier_labels = [AI_TO_HUMAN.get(t, t) for t in LABEL_ORDER]
    exp_tier_kappas = [tier_reliability[t]['expert_kappa'] for t in LABEL_ORDER]
    stu_tier_kappas = [tier_reliability[t]['student_kappa'] for t in LABEL_ORDER]
    x_tier = np.arange(len(tier_labels))
    w = 0.35
    ax_b.bar(x_tier - w / 2, exp_tier_kappas, w, label='Expert',
             color=COLORS['expert'], edgecolor='white', linewidth=0.5)
    ax_b.bar(x_tier + w / 2, stu_tier_kappas, w, label='Student',
             color=COLORS['student'], edgecolor='white', linewidth=0.5)
    ax_b.set_xticks(x_tier)
    ax_b.set_xticklabels(tier_labels, rotation=45, ha='right')
    ax_b.set_ylabel("Fleiss' kappa")
    ax_b.set_title('b  Reliability by tier', fontsize=9,
                    fontweight='bold', loc='left')
    ax_b.legend(fontsize=6)

    # Panel C: Pairwise agreement distribution
    ax_c = axes[2]
    exp_pw_vals = [r['agreement_rate'] for r in expert_pw]
    stu_pw_vals = [r['agreement_rate'] for r in student_pw]
    bp = ax_c.boxplot(
        [exp_pw_vals, stu_pw_vals],
        tick_labels=['Expert', 'Student'],
        patch_artist=True,
        widths=0.5,
        medianprops={'color': 'black', 'linewidth': 1.0},
        flierprops={'markersize': 3},
    )
    for patch, color in zip(bp['boxes'], colors_bar):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    ax_c.set_ylabel('Pairwise agreement rate')
    ax_c.set_title('c  Pairwise agreement', fontsize=9,
                    fontweight='bold', loc='left')
    ax_c.text(
        0.98, 0.02,
        f'Expert mean: {expert_pw_overall:.3f}\n'
        f'Student mean: {student_pw_overall:.3f}',
        transform=ax_c.transAxes, fontsize=6,
        ha='right', va='bottom',
    )

    fig.tight_layout()
    save_figure(fig, 'fig_s12_reliability_comparison', output_dir=fig_dir)
    plt.close(fig)

    # ---- CSV table ------------------------------------------------------
    table_rows = []
    table_rows.append({
        'group': 'Expert (overall)',
        'fleiss_kappa': round(expert_kappa, 4),
        'kappa_ci_lower': round(expert_kappa_ci['ci_lower'], 4),
        'kappa_ci_upper': round(expert_kappa_ci['ci_upper'], 4),
        'krippendorff_alpha': round(expert_alpha, 4),
        'pairwise_agreement': round(expert_pw_overall, 4),
        'n_articles': expert_fleiss_mat.shape[0],
    })
    table_rows.append({
        'group': 'Student (overall)',
        'fleiss_kappa': round(student_kappa, 4),
        'kappa_ci_lower': round(student_kappa_ci['ci_lower'], 4),
        'kappa_ci_upper': round(student_kappa_ci['ci_upper'], 4),
        'krippendorff_alpha': round(student_alpha, 4),
        'pairwise_agreement': round(student_pw_overall, 4),
        'n_articles': student_fleiss_mat.shape[0],
    })
    for tier in LABEL_ORDER:
        tr = tier_reliability[tier]
        table_rows.append({
            'group': f'Expert ({AI_TO_HUMAN.get(tier, tier)} tier)',
            'fleiss_kappa': (
                round(tr['expert_kappa'], 4)
                if not np.isnan(tr['expert_kappa']) else None
            ),
            'kappa_ci_lower': None,
            'kappa_ci_upper': None,
            'krippendorff_alpha': None,
            'pairwise_agreement': None,
            'n_articles': tr['expert_n'],
        })
        table_rows.append({
            'group': f'Student ({AI_TO_HUMAN.get(tier, tier)} tier)',
            'fleiss_kappa': (
                round(tr['student_kappa'], 4)
                if not np.isnan(tr['student_kappa']) else None
            ),
            'kappa_ci_lower': None,
            'kappa_ci_upper': None,
            'krippendorff_alpha': None,
            'pairwise_agreement': None,
            'n_articles': tr['student_n'],
        })

    df_table = pd.DataFrame(table_rows)
    df_table.to_csv(tbl_dir / 'table_s6_interrater_reliability.csv',
                    index=False)

    # ---- JSON summary ---------------------------------------------------
    summary = {
        'expert': {
            'fleiss_kappa': expert_kappa,
            'fleiss_kappa_ci': expert_kappa_ci,
            'krippendorff_alpha': expert_alpha_result,
            'pairwise_agreement_overall': expert_pw_overall,
            'n_articles': int(expert_fleiss_mat.shape[0]),
        },
        'student': {
            'fleiss_kappa': student_kappa,
            'fleiss_kappa_ci': student_kappa_ci,
            'krippendorff_alpha': student_alpha_result,
            'pairwise_agreement_overall': student_pw_overall,
            'n_articles': int(student_fleiss_mat.shape[0]),
        },
        'reliability_by_tier': {
            k: {kk: (vv if not isinstance(vv, float) or not np.isnan(vv) else None)
                for kk, vv in v.items()}
            for k, v in tier_reliability.items()
        },
        'reliability_by_domain': {
            k: {kk: (vv if not isinstance(vv, float) or not np.isnan(vv) else None)
                for kk, vv in v.items()}
            for k, v in domain_reliability.items()
        },
        'comparison': reliability_comparison,
        'permutation_test': perm_result,
    }

    with open(stat_dir / '09_interrater.json', 'w',
              encoding='utf-8') as f:
        json.dump(summary, f, indent=2, default=str, ensure_ascii=False)

    # ---- Print key results ----------------------------------------------
    print('=== Script 09: Inter-Rater Reliability ===')
    print(f"Expert Fleiss' kappa:        {expert_kappa:.4f} "
          f"[{expert_kappa_ci['ci_lower']:.4f}, "
          f"{expert_kappa_ci['ci_upper']:.4f}]")
    print(f"Expert Krippendorff's alpha: {expert_alpha:.4f}")
    print(f"Expert pairwise agreement:   {expert_pw_overall:.4f}")
    print()
    print(f"Student Fleiss' kappa:        {student_kappa:.4f} "
          f"[{student_kappa_ci['ci_lower']:.4f}, "
          f"{student_kappa_ci['ci_upper']:.4f}]")
    print(f"Student Krippendorff's alpha: {student_alpha:.4f}")
    print(f"Student pairwise agreement:   {student_pw_overall:.4f}")
    print()
    print('Reliability by tier:')
    for tier in LABEL_ORDER:
        tr = tier_reliability[tier]
        print(f'  {AI_TO_HUMAN.get(tier, tier):>6s}: '
              f'Expert={tr["expert_kappa"]:.4f} (n={tr["expert_n"]}), '
              f'Student={tr["student_kappa"]:.4f} (n={tr["student_n"]})')
    print()
    print(f'Permutation test (expert vs student):')
    print(f'  Kappa diff: {perm_result["observed_kappa_diff"]:.4f}, '
          f'p={perm_result["p_value"]:.4f}')
    print()
    print('Outputs saved.')


if __name__ == '__main__':
    main()
