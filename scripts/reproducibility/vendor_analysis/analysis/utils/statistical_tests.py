"""Statistical test utilities for the analysis pipeline.

All test functions return dictionaries containing the test statistic,
p-value, and effect size (where applicable).  Standard tests delegate
to ``scipy.stats``; Fleiss' kappa and Krippendorff's alpha are
implemented from scratch.
"""

from __future__ import annotations

from collections import Counter
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
from scipy import stats


# ---------------------------------------------------------------------------
# Paired classifier comparison
# ---------------------------------------------------------------------------

def mcnemar_test(
    y_true: List[str],
    y_pred1: List[str],
    y_pred2: List[str],
) -> Dict[str, float]:
    """McNemar's test comparing two classifiers on paired samples.

    Parameters:
        y_true: Ground-truth labels.
        y_pred1: Predictions from classifier 1.
        y_pred2: Predictions from classifier 2.

    Returns:
        Dict with ``'statistic'``, ``'p_value'``, ``'b'`` (pred1 right,
        pred2 wrong), and ``'c'`` (pred1 wrong, pred2 right).
    """
    if not (len(y_true) == len(y_pred1) == len(y_pred2)):
        raise ValueError("All input lists must have the same length.")

    # b = pred1 correct, pred2 incorrect
    # c = pred1 incorrect, pred2 correct
    b = sum(
        1 for t, p1, p2 in zip(y_true, y_pred1, y_pred2)
        if p1 == t and p2 != t
    )
    c = sum(
        1 for t, p1, p2 in zip(y_true, y_pred1, y_pred2)
        if p1 != t and p2 == t
    )

    # Use continuity correction for small samples
    if b + c == 0:
        return {
            'statistic': 0.0,
            'p_value': 1.0,
            'b': float(b),
            'c': float(c),
        }

    statistic = (abs(b - c) - 1) ** 2 / (b + c)
    p_value = float(1 - stats.chi2.cdf(statistic, df=1))

    return {
        'statistic': float(statistic),
        'p_value': p_value,
        'b': float(b),
        'c': float(c),
    }


# ---------------------------------------------------------------------------
# Multiple classifier comparison
# ---------------------------------------------------------------------------

def cochran_q_test(
    y_true: List[str],
    predictions_dict: Dict[str, List[str]],
) -> Dict[str, Any]:
    """Cochran's Q test for comparing multiple classifiers.

    Parameters:
        y_true: Ground-truth labels.
        predictions_dict: Mapping from classifier name to its prediction
            list (same length as *y_true*).

    Returns:
        Dict with ``'statistic'``, ``'p_value'``, ``'df'``, and
        ``'n_classifiers'``.
    """
    names = list(predictions_dict.keys())
    k = len(names)
    n = len(y_true)

    if k < 2:
        raise ValueError("Need at least 2 classifiers for Cochran's Q test.")

    for name in names:
        if len(predictions_dict[name]) != n:
            raise ValueError(
                f"Classifier '{name}' has {len(predictions_dict[name])} "
                f"predictions but y_true has {n} items."
            )

    # Build binary correctness matrix: (n_samples, k)
    correct = np.zeros((n, k), dtype=float)
    for j, name in enumerate(names):
        for i in range(n):
            correct[i, j] = 1.0 if predictions_dict[name][i] == y_true[i] else 0.0

    # Row and column sums
    row_sums = correct.sum(axis=1)  # T_i
    col_sums = correct.sum(axis=0)  # C_j
    grand_total = correct.sum()

    numerator = (k - 1) * (k * float(np.sum(col_sums ** 2)) - grand_total ** 2)
    denominator = k * grand_total - float(np.sum(row_sums ** 2))

    if denominator == 0:
        return {
            'statistic': 0.0,
            'p_value': 1.0,
            'df': k - 1,
            'n_classifiers': k,
        }

    q_stat = numerator / denominator
    p_value = float(1 - stats.chi2.cdf(q_stat, df=k - 1))

    return {
        'statistic': float(q_stat),
        'p_value': p_value,
        'df': k - 1,
        'n_classifiers': k,
    }


# ---------------------------------------------------------------------------
# Chi-square test
# ---------------------------------------------------------------------------

def chi_square_test(
    observed: np.ndarray,
    expected: Optional[np.ndarray] = None,
) -> Dict[str, float]:
    """Chi-square goodness-of-fit or independence test.

    Parameters:
        observed: Observed counts.  If 1-D, a goodness-of-fit test is
            performed; if 2-D, a test of independence.
        expected: Expected counts (only used for 1-D goodness-of-fit).
            If ``None`` for 1-D, a uniform distribution is assumed.

    Returns:
        Dict with ``'statistic'``, ``'p_value'``, ``'df'``, and
        ``'cramers_v'`` (effect size, only for 2-D contingency tables).
    """
    observed = np.asarray(observed, dtype=float)

    if observed.ndim == 1:
        if expected is None:
            expected_arr = np.full_like(observed, observed.sum() / len(observed))
        else:
            expected_arr = np.asarray(expected, dtype=float)
        stat, p = stats.chisquare(observed, f_exp=expected_arr)
        return {
            'statistic': float(stat),
            'p_value': float(p),
            'df': float(len(observed) - 1),
        }

    # 2-D contingency table
    stat, p, dof, _ = stats.chi2_contingency(observed)
    n = observed.sum()
    min_dim = min(observed.shape) - 1
    cramers_v = float(np.sqrt(stat / (n * min_dim))) if (n * min_dim) > 0 else 0.0

    return {
        'statistic': float(stat),
        'p_value': float(p),
        'df': float(dof),
        'cramers_v': cramers_v,
    }


# ---------------------------------------------------------------------------
# Bootstrap & permutation
# ---------------------------------------------------------------------------

def bootstrap_ci(
    data: Union[List[float], np.ndarray],
    statistic_fn: Callable[[np.ndarray], float],
    n_bootstrap: int = 10000,
    ci: float = 0.95,
    seed: int = 42,
) -> Dict[str, float]:
    """Compute a bootstrap confidence interval for a statistic.

    Parameters:
        data: 1-D array-like of observations.
        statistic_fn: Function mapping an array to a scalar.
        n_bootstrap: Number of bootstrap resamples.
        ci: Confidence level (e.g. 0.95 for 95 % CI).
        seed: Random seed.

    Returns:
        Dict with ``'estimate'`` (point estimate), ``'ci_lower'``,
        ``'ci_upper'``, ``'std'``, and ``'n_bootstrap'``.
    """
    data_arr = np.asarray(data, dtype=float)
    rng = np.random.RandomState(seed)

    point_estimate = float(statistic_fn(data_arr))
    boot_stats: List[float] = []

    for _ in range(n_bootstrap):
        sample = rng.choice(data_arr, size=len(data_arr), replace=True)
        boot_stats.append(float(statistic_fn(sample)))

    boot_array = np.array(boot_stats)
    alpha = 1 - ci
    lower = float(np.percentile(boot_array, 100 * alpha / 2))
    upper = float(np.percentile(boot_array, 100 * (1 - alpha / 2)))

    return {
        'estimate': point_estimate,
        'ci_lower': lower,
        'ci_upper': upper,
        'std': float(np.std(boot_array, ddof=1)),
        'n_bootstrap': n_bootstrap,
    }


def permutation_test(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
    statistic_fn: Callable[[np.ndarray, np.ndarray], float],
    n_permutations: int = 10000,
    seed: int = 42,
) -> Dict[str, float]:
    """Two-sample permutation test.

    Parameters:
        group1: Observations from group 1.
        group2: Observations from group 2.
        statistic_fn: Function ``(g1, g2) -> scalar`` computing the test
            statistic (e.g. difference in means).
        n_permutations: Number of permutations.
        seed: Random seed.

    Returns:
        Dict with ``'observed_statistic'``, ``'p_value'``, and
        ``'n_permutations'``.
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)
    rng = np.random.RandomState(seed)

    observed = float(statistic_fn(g1, g2))
    combined = np.concatenate([g1, g2])
    n1 = len(g1)

    count_extreme = 0
    for _ in range(n_permutations):
        rng.shuffle(combined)
        perm_stat = statistic_fn(combined[:n1], combined[n1:])
        if abs(perm_stat) >= abs(observed):
            count_extreme += 1

    p_value = (count_extreme + 1) / (n_permutations + 1)

    return {
        'observed_statistic': observed,
        'p_value': float(p_value),
        'n_permutations': n_permutations,
    }


# ---------------------------------------------------------------------------
# Non-parametric tests (scipy wrappers)
# ---------------------------------------------------------------------------

def mann_whitney_u(
    group1: Union[List[float], np.ndarray],
    group2: Union[List[float], np.ndarray],
) -> Dict[str, float]:
    """Mann-Whitney U test (two-sided).

    Parameters:
        group1: Observations from group 1.
        group2: Observations from group 2.

    Returns:
        Dict with ``'statistic'``, ``'p_value'``, and
        ``'rank_biserial'`` (effect size r = 1 - 2U/(n1*n2)).
    """
    g1 = np.asarray(group1, dtype=float)
    g2 = np.asarray(group2, dtype=float)

    stat, p = stats.mannwhitneyu(g1, g2, alternative='two-sided')

    n1, n2 = len(g1), len(g2)
    rank_biserial = 1 - (2 * stat) / (n1 * n2) if (n1 * n2) > 0 else 0.0

    return {
        'statistic': float(stat),
        'p_value': float(p),
        'rank_biserial': float(rank_biserial),
    }


def kruskal_wallis(*groups: Union[List[float], np.ndarray]) -> Dict[str, float]:
    """Kruskal-Wallis H test for independent samples.

    Parameters:
        *groups: Two or more groups of observations.

    Returns:
        Dict with ``'statistic'``, ``'p_value'``, and ``'n_groups'``.
    """
    if len(groups) < 2:
        raise ValueError("Need at least 2 groups for Kruskal-Wallis test.")

    arrays = [np.asarray(g, dtype=float) for g in groups]
    stat, p = stats.kruskal(*arrays)

    return {
        'statistic': float(stat),
        'p_value': float(p),
        'n_groups': len(groups),
    }


def spearman_correlation(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
) -> Dict[str, float]:
    """Spearman rank correlation.

    Parameters:
        x: First variable.
        y: Second variable.

    Returns:
        Dict with ``'correlation'`` (rho) and ``'p_value'``.
    """
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)

    rho, p = stats.spearmanr(x_arr, y_arr)

    return {
        'correlation': float(rho),
        'p_value': float(p),
    }


def one_sample_ttest(
    data: Union[List[float], np.ndarray],
    popmean: float,
) -> Dict[str, float]:
    """One-sample t-test.

    Parameters:
        data: Sample observations.
        popmean: Hypothesised population mean.

    Returns:
        Dict with ``'statistic'``, ``'p_value'``, ``'mean'``, ``'std'``,
        and ``'cohens_d'`` (effect size).
    """
    data_arr = np.asarray(data, dtype=float)
    stat, p = stats.ttest_1samp(data_arr, popmean)

    sample_mean = float(np.mean(data_arr))
    sample_std = float(np.std(data_arr, ddof=1)) if len(data_arr) > 1 else 0.0
    cohens_d = (sample_mean - popmean) / sample_std if sample_std > 0 else 0.0

    return {
        'statistic': float(stat),
        'p_value': float(p),
        'mean': sample_mean,
        'std': sample_std,
        'cohens_d': float(cohens_d),
    }


# ---------------------------------------------------------------------------
# Multiple testing corrections
# ---------------------------------------------------------------------------

def bonferroni_correction(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Apply Bonferroni correction for multiple comparisons.

    Parameters:
        p_values: List of uncorrected p-values.
        alpha: Family-wise error rate.

    Returns:
        Dict with ``'corrected_p_values'`` (list), ``'rejected'``
        (list of booleans), ``'alpha_corrected'``, and ``'n_tests'``.
    """
    p_arr = np.asarray(p_values, dtype=float)
    n = len(p_arr)
    alpha_corrected = alpha / n if n > 0 else alpha

    corrected = np.minimum(p_arr * n, 1.0)
    rejected = corrected < alpha

    return {
        'corrected_p_values': corrected.tolist(),
        'rejected': rejected.tolist(),
        'alpha_corrected': float(alpha_corrected),
        'n_tests': n,
    }


def fdr_correction(
    p_values: Union[List[float], np.ndarray],
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Apply Benjamini-Hochberg FDR correction.

    Parameters:
        p_values: List of uncorrected p-values.
        alpha: Desired false discovery rate.

    Returns:
        Dict with ``'corrected_p_values'`` (list), ``'rejected'``
        (list of booleans), ``'alpha'``, and ``'n_tests'``.
    """
    p_arr = np.asarray(p_values, dtype=float)
    n = len(p_arr)

    if n == 0:
        return {
            'corrected_p_values': [],
            'rejected': [],
            'alpha': alpha,
            'n_tests': 0,
        }

    # Sort p-values and keep track of original indices
    sorted_indices = np.argsort(p_arr)
    sorted_p = p_arr[sorted_indices]

    # Compute adjusted p-values (step-up procedure)
    adjusted = np.zeros(n)
    adjusted[n - 1] = sorted_p[n - 1]

    for i in range(n - 2, -1, -1):
        adjusted[i] = min(
            adjusted[i + 1],
            sorted_p[i] * n / (i + 1),
        )

    adjusted = np.minimum(adjusted, 1.0)

    # Map back to original order
    corrected = np.zeros(n)
    corrected[sorted_indices] = adjusted

    rejected = corrected < alpha

    return {
        'corrected_p_values': corrected.tolist(),
        'rejected': rejected.tolist(),
        'alpha': alpha,
        'n_tests': n,
    }


# ---------------------------------------------------------------------------
# Inter-rater reliability (custom implementations)
# ---------------------------------------------------------------------------

def fleiss_kappa(ratings_matrix: Union[List[List[int]], np.ndarray]) -> Dict[str, float]:
    """Compute Fleiss' kappa for multiple raters.

    Parameters:
        ratings_matrix: 2-D array of shape ``(n_subjects, n_categories)``
            where ``ratings_matrix[i][j]`` is the number of raters who
            assigned subject *i* to category *j*.

    Returns:
        Dict with ``'kappa'``, ``'p_e'`` (chance agreement), and
        ``'p_o'`` (observed agreement).
    """
    table = np.asarray(ratings_matrix, dtype=float)
    n_subjects, n_categories = table.shape
    n_raters = table[0].sum()

    if n_raters <= 1:
        raise ValueError("Need at least 2 raters per subject for Fleiss' kappa.")

    # Proportion of all assignments to each category
    p_j = table.sum(axis=0) / (n_subjects * n_raters)

    # Expected chance agreement
    p_e = float(np.sum(p_j ** 2))

    # Per-subject agreement
    p_i = (np.sum(table ** 2, axis=1) - n_raters) / (n_raters * (n_raters - 1))

    # Overall observed agreement
    p_o = float(np.mean(p_i))

    # Fleiss' kappa
    if abs(1.0 - p_e) < 1e-12:
        kappa = 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0
    else:
        kappa = (p_o - p_e) / (1.0 - p_e)

    return {
        'kappa': float(kappa),
        'p_e': p_e,
        'p_o': p_o,
    }


def krippendorff_alpha(
    reliability_data: Union[List[List[Optional[float]]], np.ndarray],
    level: str = 'ordinal',
) -> Dict[str, float]:
    """Compute Krippendorff's alpha for inter-rater reliability.

    Parameters:
        reliability_data: 2-D array of shape ``(n_raters, n_units)``
            where ``reliability_data[r][u]`` is the value assigned by
            rater *r* to unit *u*.  Use ``None`` or ``np.nan`` for
            missing values.
        level: Measurement level -- ``'nominal'``, ``'ordinal'``,
            ``'interval'``, or ``'ratio'``.

    Returns:
        Dict with ``'alpha'``, ``'d_o'`` (observed disagreement), and
        ``'d_e'`` (expected disagreement).

    Raises:
        ValueError: On unknown measurement level.
    """
    valid_levels = {'nominal', 'ordinal', 'interval', 'ratio'}
    if level not in valid_levels:
        raise ValueError(f"Unknown level '{level}'. Choose from {valid_levels}.")

    # Convert to numpy, replacing None with nan
    data = np.array(reliability_data, dtype=float)
    n_raters, n_units = data.shape

    # Build value-pair coincidence matrix
    # Collect all non-missing values per unit
    values_per_unit: List[List[float]] = []
    for u in range(n_units):
        vals = [data[r, u] for r in range(n_raters) if not np.isnan(data[r, u])]
        values_per_unit.append(vals)

    # Total number of pairable values
    all_values: List[float] = []
    for vals in values_per_unit:
        all_values.extend(vals)

    if len(all_values) == 0:
        return {'alpha': 0.0, 'd_o': 0.0, 'd_e': 0.0}

    unique_values = sorted(set(all_values))
    val_to_idx = {v: i for i, v in enumerate(unique_values)}
    n_vals = len(unique_values)

    # Difference function
    def _delta_squared(v1: float, v2: float) -> float:
        if level == 'nominal':
            return 0.0 if v1 == v2 else 1.0
        if level == 'interval':
            return (v1 - v2) ** 2
        if level == 'ratio':
            denom = v1 + v2
            return ((v1 - v2) / denom) ** 2 if denom != 0 else 0.0
        # Ordinal
        i1 = val_to_idx[v1]
        i2 = val_to_idx[v2]
        if i1 > i2:
            i1, i2 = i2, i1

        # Build frequency distribution
        freq = Counter(all_values)
        n_c = sum(freq.get(unique_values[k], 0) for k in range(i1, i2 + 1))
        # Subtract endpoints halved
        n_c = n_c - (freq.get(v1, 0) + freq.get(v2, 0)) / 2
        return n_c ** 2

    # Observed disagreement
    d_o_numerator = 0.0
    d_o_denominator = 0.0

    for vals in values_per_unit:
        m_u = len(vals)
        if m_u < 2:
            continue
        for i in range(m_u):
            for j in range(i + 1, m_u):
                d_o_numerator += _delta_squared(vals[i], vals[j])
                d_o_denominator += 1.0

    if d_o_denominator == 0:
        return {'alpha': 1.0, 'd_o': 0.0, 'd_e': 0.0}

    d_o = d_o_numerator / d_o_denominator

    # Expected disagreement (based on marginal distribution)
    n_total = len(all_values)
    d_e_numerator = 0.0
    d_e_denominator = 0.0

    freq = Counter(all_values)
    for i, v1 in enumerate(unique_values):
        for j, v2 in enumerate(unique_values):
            if i < j:
                n1 = freq[v1]
                n2 = freq[v2]
                d_e_numerator += n1 * n2 * _delta_squared(v1, v2)
                d_e_denominator += n1 * n2

    if d_e_denominator == 0:
        return {'alpha': 1.0, 'd_o': d_o, 'd_e': 0.0}

    d_e = d_e_numerator / d_e_denominator

    if d_e == 0:
        alpha = 1.0
    else:
        alpha = 1.0 - d_o / d_e

    return {
        'alpha': float(alpha),
        'd_o': float(d_o),
        'd_e': float(d_e),
    }
