"""Voting and sampling utilities for the analysis pipeline.

Provides majority-vote aggregation, probability distribution conversion,
and Monte Carlo sampling to estimate the accuracy distribution of
human-rater subsets on the 120-article research quality dataset.
"""

from __future__ import annotations

import random
from collections import Counter
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from .constants import HUMAN_TO_AI, LABEL_ORDER


# ---------------------------------------------------------------------------
# Label normalisation helpers
# ---------------------------------------------------------------------------

def rating_to_level(rating: str) -> str:
    """Convert a human survey rating to a lowercase level string.

    ``'Top'`` -> ``'top'``, ``'Top-'`` -> ``'top-'``,
    ``'Good'`` -> ``'good'``, ``'Fair'`` -> ``'fair'``.

    Parameters:
        rating: Human survey rating string (e.g. ``'Top'``).

    Returns:
        Lowercase level string.
    """
    mapping = {
        'Top': 'top',
        'Top-': 'top-',
        'Good': 'good',
        'Fair': 'fair',
    }
    return mapping.get(rating, rating.lower())


def norm_level(level: str) -> str:
    """Normalise a level string to lowercase.

    Parameters:
        level: Level string from any source.

    Returns:
        Lowercase-trimmed level string.
    """
    return level.strip().lower()


# ---------------------------------------------------------------------------
# Majority voting
# ---------------------------------------------------------------------------

def majority_vote(
    votes: List[str],
    tie_policy: str = 'exclude',
    rng: Optional[random.Random] = None,
) -> Tuple[Optional[str], bool]:
    """Compute a majority vote from a list of rating labels.

    Parameters:
        votes: List of label strings (any label space).
        tie_policy: How to handle ties:
            - ``'exclude'``: return ``(None, True)``
            - ``'wrong'``: return the *least common* label among tied
              (effectively always wrong), useful as a pessimistic baseline.
            - ``'half'``: return the first tied label alphabetically
              (deterministic, roughly 50 % correct on balanced ties).
            - ``'random'``: pick uniformly at random among tied labels.
        rng: Optional ``random.Random`` instance for reproducibility
            when ``tie_policy='random'``.

    Returns:
        Tuple of ``(prediction, is_tie)``.  ``prediction`` is ``None``
        only when ``tie_policy='exclude'`` and there is a tie.

    Raises:
        ValueError: On empty votes or unknown tie_policy.
    """
    if not votes:
        raise ValueError("Cannot compute majority vote on empty list.")

    valid_policies = {'exclude', 'wrong', 'half', 'random'}
    if tie_policy not in valid_policies:
        raise ValueError(
            f"Unknown tie_policy '{tie_policy}'. Choose from {valid_policies}."
        )

    counts = Counter(votes)
    max_count = max(counts.values())
    top_labels = [label for label, cnt in counts.items() if cnt == max_count]

    is_tie = len(top_labels) > 1

    if not is_tie:
        return (top_labels[0], False)

    # Tie-breaking
    if tie_policy == 'exclude':
        return (None, True)

    if tie_policy == 'wrong':
        # Return the label with the lowest count overall (pessimistic)
        min_count = min(counts.values())
        worst_labels = [label for label, cnt in counts.items() if cnt == min_count]
        worst_labels.sort()
        return (worst_labels[0], True)

    if tie_policy == 'half':
        top_labels.sort()
        return (top_labels[0], True)

    # tie_policy == 'random'
    if rng is None:
        rng = random.Random()
    return (rng.choice(top_labels), True)


# ---------------------------------------------------------------------------
# Probability distribution from ratings
# ---------------------------------------------------------------------------

def ratings_to_prob_distribution(
    ratings_list: List[str],
    labels: List[str] = LABEL_ORDER,
) -> Dict[str, float]:
    """Convert a list of rating strings to a probability distribution.

    Each rating string is first normalised to the AI label space using
    :data:`constants.HUMAN_TO_AI` and :data:`constants.LABEL_ORDER`.

    Parameters:
        ratings_list: List of rating strings (any label convention).
        labels: Ordered label list for the output distribution.

    Returns:
        Dict mapping each label to its empirical probability.  Labels
        not present in *ratings_list* receive probability 0.
    """
    if not ratings_list:
        return {label: 0.0 for label in labels}

    normalised: List[str] = []
    for r in ratings_list:
        stripped = r.strip()
        if stripped in HUMAN_TO_AI:
            normalised.append(HUMAN_TO_AI[stripped])
        elif stripped.lower() in {l.lower() for l in labels}:
            # Already in AI label space (case-insensitive match)
            matched = next(l for l in labels if l.lower() == stripped.lower())
            normalised.append(matched)
        else:
            normalised.append(stripped)

    counts = Counter(normalised)
    total = len(normalised)

    return {
        label: counts.get(label, 0) / total
        for label in labels
    }


# ---------------------------------------------------------------------------
# Monte Carlo sampling
# ---------------------------------------------------------------------------

def monte_carlo_sample_vote(
    votes: List[Tuple[str, str]],
    sample_size: int,
    n_trials: int = 5000,
    tie_policy: str = 'exclude',
    seed: int = 42,
) -> Dict[str, Any]:
    """Run Monte Carlo sampling to estimate majority-vote accuracy.

    Repeatedly draws a random subset of raters, computes majority vote
    per article, and measures accuracy against ground truth.

    Parameters:
        votes: List of ``(ground_truth, rating)`` tuples for a single
            article, where both are in the same label space.  For
            multi-article analysis, group by article externally and
            aggregate results.
        sample_size: Number of raters to sample per trial.
        n_trials: Number of Monte Carlo trials.
        tie_policy: Tie-breaking policy passed to :func:`majority_vote`.
        seed: Random seed for reproducibility.

    Returns:
        Dict with keys:
        - ``'mean_accuracy'``: mean accuracy across trials.
        - ``'std'``: standard deviation of accuracy.
        - ``'ci_lower'``: lower bound of 95 % CI.
        - ``'ci_upper'``: upper bound of 95 % CI.
        - ``'n_trials'``: number of trials executed.
        - ``'n_excluded'``: mean number of excluded articles per trial
          (only relevant for ``tie_policy='exclude'``).
    """
    rng = random.Random(seed)
    np_rng = np.random.RandomState(seed)

    if sample_size > len(votes):
        raise ValueError(
            f"sample_size ({sample_size}) exceeds number of votes "
            f"({len(votes)})."
        )

    # Group votes by ground truth for clarity
    # Each trial: sample `sample_size` votes, do majority vote, check match
    accuracies: List[float] = []
    excluded_counts: List[int] = []

    for _ in range(n_trials):
        indices = np_rng.choice(len(votes), size=sample_size, replace=False)
        sampled = [votes[i] for i in indices]

        ground_truths = [gt for gt, _ in sampled]
        ratings = [r for _, r in sampled]

        prediction, is_tie = majority_vote(ratings, tie_policy=tie_policy, rng=rng)

        if prediction is None:
            # Excluded due to tie
            excluded_counts.append(1)
            continue

        excluded_counts.append(0)

        # Check against the most common ground truth in the sample
        gt_counts = Counter(ground_truths)
        true_label = gt_counts.most_common(1)[0][0]
        accuracies.append(1.0 if prediction == true_label else 0.0)

    if not accuracies:
        return {
            'mean_accuracy': 0.0,
            'std': 0.0,
            'ci_lower': 0.0,
            'ci_upper': 0.0,
            'n_trials': n_trials,
            'n_excluded': float(sum(excluded_counts)) / n_trials if excluded_counts else 0.0,
        }

    acc_array = np.array(accuracies)
    mean_acc = float(np.mean(acc_array))
    std_acc = float(np.std(acc_array, ddof=1)) if len(acc_array) > 1 else 0.0
    ci_lower = float(np.percentile(acc_array, 2.5))
    ci_upper = float(np.percentile(acc_array, 97.5))

    return {
        'mean_accuracy': mean_acc,
        'std': std_acc,
        'ci_lower': ci_lower,
        'ci_upper': ci_upper,
        'n_trials': n_trials,
        'n_excluded': float(sum(excluded_counts)) / n_trials,
    }
