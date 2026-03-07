"""Metric computation utilities for the analysis pipeline.

Provides accuracy, per-class metrics, confusion matrices, calibration
metrics, and inter-rater agreement measures used to evaluate AI model
and human rater performance on the 120-article research quality dataset.
"""

from __future__ import annotations

import math
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np

from .constants import LABEL_ORDER


# ---------------------------------------------------------------------------
# Probability helpers
# ---------------------------------------------------------------------------

def logp_to_prob(
    logp_dict: Dict[str, float],
    labels: List[str] = LABEL_ORDER,
) -> Dict[str, float]:
    """Convert log-probabilities to probabilities via softmax.

    Applies the log-sum-exp trick for numerical stability.

    Parameters:
        logp_dict: Mapping from label to log-probability.
        labels: Ordered label list.  Only these keys are considered;
            missing labels receive ``-inf`` (probability 0).

    Returns:
        Dict mapping each label to its normalised probability.
    """
    raw_values = [logp_dict.get(label, float('-inf')) for label in labels]
    max_val = max(raw_values)

    # Guard against all -inf
    if math.isinf(max_val) and max_val < 0:
        uniform = 1.0 / len(labels)
        return {label: uniform for label in labels}

    exp_values = [math.exp(v - max_val) for v in raw_values]
    total = sum(exp_values)

    return {
        label: exp_val / total
        for label, exp_val in zip(labels, exp_values)
    }


def get_top_predictions(
    prob_dict: Dict[str, float],
    top_k: int = 2,
) -> List[Tuple[str, float]]:
    """Return the top-k predictions sorted by descending probability.

    Parameters:
        prob_dict: Mapping from label to probability.
        top_k: Number of top predictions to return.

    Returns:
        List of ``(label, probability)`` tuples sorted descending.
    """
    sorted_items = sorted(prob_dict.items(), key=lambda x: x[1], reverse=True)
    return sorted_items[:top_k]


# ---------------------------------------------------------------------------
# Classification metrics
# ---------------------------------------------------------------------------

def compute_accuracy(
    y_true: List[str],
    y_pred: List[str],
) -> float:
    """Compute overall classification accuracy.

    Parameters:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.

    Returns:
        Accuracy as a float in [0, 1].

    Raises:
        ValueError: If the input lists have different lengths or are empty.
    """
    if len(y_true) != len(y_pred):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} items, "
            f"y_pred has {len(y_pred)} items."
        )
    if len(y_true) == 0:
        raise ValueError("Cannot compute accuracy on empty lists.")

    correct = sum(1 for t, p in zip(y_true, y_pred) if t == p)
    return correct / len(y_true)


def compute_per_class_metrics(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = LABEL_ORDER,
) -> Dict[str, Dict[str, float]]:
    """Compute per-class precision, recall, F1, and support.

    Parameters:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered label list.

    Returns:
        Dict mapping each label to a dict with keys ``'precision'``,
        ``'recall'``, ``'f1'``, and ``'support'``.
    """
    results: Dict[str, Dict[str, float]] = {}

    for label in labels:
        tp = sum(1 for t, p in zip(y_true, y_pred) if t == label and p == label)
        fp = sum(1 for t, p in zip(y_true, y_pred) if t != label and p == label)
        fn = sum(1 for t, p in zip(y_true, y_pred) if t == label and p != label)
        support = sum(1 for t in y_true if t == label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0
            else 0.0
        )

        results[label] = {
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'support': float(support),
        }

    return results


def compute_macro_f1(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = LABEL_ORDER,
) -> float:
    """Compute macro-averaged F1 score.

    Parameters:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered label list.

    Returns:
        Macro F1 as a float in [0, 1].
    """
    per_class = compute_per_class_metrics(y_true, y_pred, labels)
    f1_values = [per_class[label]['f1'] for label in labels]
    return sum(f1_values) / len(f1_values) if f1_values else 0.0


def compute_weighted_f1(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = LABEL_ORDER,
) -> float:
    """Compute weighted-averaged F1 score (weighted by support).

    Parameters:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered label list.

    Returns:
        Weighted F1 as a float in [0, 1].
    """
    per_class = compute_per_class_metrics(y_true, y_pred, labels)
    total_support = sum(per_class[label]['support'] for label in labels)
    if total_support == 0:
        return 0.0
    return sum(
        per_class[label]['f1'] * per_class[label]['support']
        for label in labels
    ) / total_support


def compute_balanced_accuracy(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = LABEL_ORDER,
) -> float:
    """Compute balanced accuracy (mean of per-class recall).

    Immune to class-biased predictions: a model predicting all items
    as one class scores 1/K (e.g. 0.25 for 4 classes).

    Parameters:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered label list.

    Returns:
        Balanced accuracy as a float in [0, 1].
    """
    per_class = compute_per_class_metrics(y_true, y_pred, labels)
    recalls = [per_class[label]['recall'] for label in labels]
    return sum(recalls) / len(recalls) if recalls else 0.0


def compute_prediction_distribution(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = LABEL_ORDER,
) -> Dict[str, Dict[str, Any]]:
    """Compare predicted vs actual label distributions.

    Parameters:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered label list.

    Returns:
        Dict with 'actual' and 'predicted' counts plus a 'bias_index'
        (ratio of middle-tier predictions to actual middle-tier count).
    """
    actual = Counter(y_true)
    predicted = Counter(y_pred)
    middle_labels = labels[1:-1] if len(labels) >= 3 else labels
    actual_middle = sum(actual.get(l, 0) for l in middle_labels)
    pred_middle = sum(predicted.get(l, 0) for l in middle_labels)
    bias_index = pred_middle / actual_middle if actual_middle > 0 else 0.0
    return {
        'actual': {l: actual.get(l, 0) for l in labels},
        'predicted': {l: predicted.get(l, 0) for l in labels},
        'middle_tier_bias': float(bias_index),
    }


def compute_confusion_matrix(
    y_true: List[str],
    y_pred: List[str],
    labels: List[str] = LABEL_ORDER,
) -> np.ndarray:
    """Compute a confusion matrix.

    Parameters:
        y_true: Ground-truth labels.
        y_pred: Predicted labels.
        labels: Ordered label list defining row/column order.

    Returns:
        2-D numpy array of shape ``(len(labels), len(labels))`` where
        ``cm[i][j]`` is the count of samples with true label ``labels[i]``
        predicted as ``labels[j]``.
    """
    label_to_idx = {label: idx for idx, label in enumerate(labels)}
    n = len(labels)
    cm = np.zeros((n, n), dtype=int)

    for t, p in zip(y_true, y_pred):
        if t in label_to_idx and p in label_to_idx:
            cm[label_to_idx[t], label_to_idx[p]] += 1

    return cm


# ---------------------------------------------------------------------------
# Calibration metrics
# ---------------------------------------------------------------------------

def compute_ece(
    y_true: List[str],
    y_probs: List[Dict[str, float]],
    n_bins: int = 10,
) -> float:
    """Compute Expected Calibration Error (ECE).

    Uses equal-width binning on the predicted confidence (maximum
    probability) of each sample.

    Parameters:
        y_true: Ground-truth labels.
        y_probs: List of probability dicts (one per sample), mapping
            each label to its predicted probability.
        n_bins: Number of calibration bins.

    Returns:
        ECE as a float in [0, 1].
    """
    if len(y_true) != len(y_probs):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} items, "
            f"y_probs has {len(y_probs)} items."
        )

    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0

    # Extract max confidence and whether prediction is correct
    confidences: List[float] = []
    correct: List[int] = []

    for true_label, prob_dict in zip(y_true, y_probs):
        pred_label = max(prob_dict, key=prob_dict.get)  # type: ignore[arg-type]
        max_conf = prob_dict[pred_label]
        confidences.append(max_conf)
        correct.append(1 if pred_label == true_label else 0)

    # Equal-width binning
    bin_boundaries = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0

    for bin_idx in range(n_bins):
        lower = bin_boundaries[bin_idx]
        upper = bin_boundaries[bin_idx + 1]

        # Include right boundary for last bin
        if bin_idx == n_bins - 1:
            in_bin = [
                i for i, c in enumerate(confidences)
                if lower <= c <= upper
            ]
        else:
            in_bin = [
                i for i, c in enumerate(confidences)
                if lower <= c < upper
            ]

        if len(in_bin) == 0:
            continue

        bin_acc = sum(correct[i] for i in in_bin) / len(in_bin)
        bin_conf = sum(confidences[i] for i in in_bin) / len(in_bin)
        ece += (len(in_bin) / n_samples) * abs(bin_acc - bin_conf)

    return ece


def compute_brier_score(
    y_true: List[str],
    y_probs: List[Dict[str, float]],
    labels: List[str] = LABEL_ORDER,
) -> float:
    """Compute the multi-class Brier score.

    .. math::

        BS = \\frac{1}{N} \\sum_{i=1}^{N} \\sum_{k=1}^{K}
             (p_{ik} - y_{ik})^2

    where *y_{ik}* is 1 if sample *i* has true label *k*, else 0.

    Parameters:
        y_true: Ground-truth labels.
        y_probs: List of probability dicts.
        labels: Ordered label list.

    Returns:
        Brier score (lower is better).
    """
    if len(y_true) != len(y_probs):
        raise ValueError(
            f"Length mismatch: y_true has {len(y_true)} items, "
            f"y_probs has {len(y_probs)} items."
        )

    n_samples = len(y_true)
    if n_samples == 0:
        return 0.0

    total = 0.0
    for true_label, prob_dict in zip(y_true, y_probs):
        for label in labels:
            p = prob_dict.get(label, 0.0)
            y = 1.0 if label == true_label else 0.0
            total += (p - y) ** 2

    return total / n_samples


# ---------------------------------------------------------------------------
# Inter-rater agreement
# ---------------------------------------------------------------------------

def compute_cohens_kappa(
    y1: List[str],
    y2: List[str],
) -> float:
    """Compute Cohen's kappa between two rater lists.

    Parameters:
        y1: Ratings from rater 1.
        y2: Ratings from rater 2.

    Returns:
        Cohen's kappa coefficient in [-1, 1].  1 indicates perfect
        agreement, 0 indicates agreement equal to chance, and negative
        values indicate agreement worse than chance.

    Raises:
        ValueError: If lists have different lengths or are empty.
    """
    if len(y1) != len(y2):
        raise ValueError(
            f"Length mismatch: y1 has {len(y1)} items, "
            f"y2 has {len(y2)} items."
        )
    if len(y1) == 0:
        raise ValueError("Cannot compute Cohen's kappa on empty lists.")

    n = len(y1)
    all_labels = sorted(set(y1) | set(y2))
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    k = len(all_labels)

    # Build confusion matrix
    cm = np.zeros((k, k), dtype=float)
    for a, b in zip(y1, y2):
        cm[label_to_idx[a], label_to_idx[b]] += 1

    # Observed agreement
    p_o = float(np.trace(cm)) / n

    # Expected agreement (product of marginals)
    row_sums = cm.sum(axis=1)
    col_sums = cm.sum(axis=0)
    p_e = float(np.sum(row_sums * col_sums)) / (n * n)

    if abs(1.0 - p_e) < 1e-12:
        return 1.0 if abs(p_o - 1.0) < 1e-12 else 0.0

    return float((p_o - p_e) / (1.0 - p_e))
