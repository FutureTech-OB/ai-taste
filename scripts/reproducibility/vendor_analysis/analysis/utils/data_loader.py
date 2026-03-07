"""Data loading utilities for the analysis pipeline.

Provides functions to load all data files (articles, model predictions,
human ratings) and to normalise label conventions across data sources.

Field conventions across data sources:
    - Articles JSONL:  ``level`` uses 'top', 'top-', 'good', 'fair'
    - Val / Thinking:  ``rank``  uses 'exceptional', 'strong', 'fair', 'limited'
    - Human surveys:   ``q2_rating`` uses 'Top', 'Top-', 'Good', 'Fair'
                       ``level``    uses 'top', 'top-', 'good', 'fair'
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

from .constants import (
    ARTICLES_PATH,
    ALL_COMBINED_PATH,
    CHAT_PATH,
    ENRICHED_EXPERT_FILTERED_PATH,
    ENRICHED_EXPERT_PATH,
    ENRICHED_STUDENT_FILTERED_PATH,
    ENRICHED_STUDENT_PATH,
    EXPERT_FILTERED_PATH,
    EXPERT_PATH,
    HUMAN_TO_AI,
    LEVEL_TO_AI,
    FRONTIER_PATH,
    STUDENT_FILTERED_PATH,
    STUDENT_MERGED_PATH,
    STUDENT_NEW_PATH,
    STUDENT_OLD_PATH,
    THINKING_LEGACY_PATH,
    THINKING_PATH,
    TRAINED_PATH,
    UNFILTERED_EXPERT_PATH,
    UNFILTERED_STUDENT_PATH,
    VAL_PATH,
)


# ---------------------------------------------------------------------------
# Project root detection
# ---------------------------------------------------------------------------

def get_project_root() -> Path:
    """Return the absolute ``Path`` to the project root directory.

    Detection strategy (in order):
    1. Walk up from this file's location looking for a ``data/`` directory.
    2. Walk up from the current working directory looking for ``data/``.
    3. Fall back to the current working directory.

    Returns:
        Path to the project root.
    """
    # Strategy 1: relative to this source file
    current = Path(__file__).resolve().parent
    for ancestor in [current] + list(current.parents):
        if (ancestor / 'data').is_dir():
            return ancestor

    # Strategy 2: relative to cwd
    cwd = Path.cwd()
    for ancestor in [cwd] + list(cwd.parents):
        if (ancestor / 'data').is_dir():
            return ancestor

    # Fallback
    return cwd


# ---------------------------------------------------------------------------
# Generic JSONL loader
# ---------------------------------------------------------------------------

def load_jsonl(path: Union[str, Path]) -> List[Dict[str, Any]]:
    """Load a JSONL file and return a list of dictionaries.

    Parameters:
        path: Path to the JSONL file (absolute or relative to project root).

    Returns:
        List of parsed JSON records.

    Raises:
        FileNotFoundError: If the file does not exist.
        json.JSONDecodeError: If a line contains invalid JSON.
    """
    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = get_project_root() / resolved

    records: List[Dict[str, Any]] = []
    with open(resolved, 'r', encoding='utf-8') as fh:
        for line_no, line in enumerate(fh, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                records.append(json.loads(stripped))
            except json.JSONDecodeError as exc:
                raise json.JSONDecodeError(
                    f"Invalid JSON on line {line_no} of {resolved}: {exc.msg}",
                    exc.doc,
                    exc.pos,
                ) from exc
    return records


# ---------------------------------------------------------------------------
# Typed data loaders
# ---------------------------------------------------------------------------

def load_articles() -> List[Dict[str, Any]]:
    """Load ``selected_120_articles.jsonl``.

    Returns:
        List of article dicts (each contains at minimum ``title``,
        ``level``, and article content fields).
    """
    return load_jsonl(ARTICLES_PATH)


def load_trained_data() -> List[Dict[str, Any]]:
    """Load ``120_trained.jsonl`` (SFT models, ensembles, human voting).

    Returns:
        List of prediction records with ``rank`` and ``logp`` fields.
    """
    return load_jsonl(TRAINED_PATH)


def load_val_data() -> List[Dict[str, Any]]:
    """Load SFT/trained model predictions (``120_trained.jsonl``).

    Backward-compatible alias for :func:`load_trained_data`.
    Note: the original ``120_val.jsonl`` no longer exists.

    Returns:
        List of prediction records with ``rank`` and ``logp`` fields.
    """
    return load_jsonl(VAL_PATH)


def load_chat_data() -> List[Dict[str, Any]]:
    """Load ``120_chat.jsonl`` (chat models with logp: gpt-5.2, kimi-k2, deepseek-chat).

    Returns:
        List of prediction records with ``rank`` and ``logp`` fields.
    """
    return load_jsonl(CHAT_PATH)


def load_thinking_data() -> List[Dict[str, Any]]:
    """Load thinking/frontier model predictions.

    Returns:
        List of prediction records with ``rank`` and ``logprobs`` fields.
    """
    candidates = [THINKING_PATH, FRONTIER_PATH, THINKING_LEGACY_PATH]
    last_exc: Optional[Exception] = None
    for candidate in candidates:
        try:
            return load_jsonl(candidate)
        except FileNotFoundError as exc:
            last_exc = exc
            continue
    if last_exc is not None:
        raise last_exc
    raise FileNotFoundError(f"Could not locate any thinking/frontier file in: {candidates}")


def load_frontier_data() -> List[Dict[str, Any]]:
    """Load canonical frontier protocol file (``120_frontier.jsonl``)."""
    return load_jsonl(FRONTIER_PATH)


def load_expert_ratings() -> List[Dict[str, Any]]:
    """Load unfiltered expert ratings (47+ experts, primary dataset).

    Returns:
        List of article records containing expert survey responses.
    """
    return load_jsonl(EXPERT_PATH)


def load_expert_filtered_ratings() -> List[Dict[str, Any]]:
    """Load filtered expert ratings (39 experts, for sensitivity analysis).

    Returns:
        List of article records containing filtered expert survey responses.
    """
    return load_jsonl(EXPERT_FILTERED_PATH)


def load_student_new_ratings() -> List[Dict[str, Any]]:
    """Load the filtered combined junior ratings.

    Deprecated compatibility alias. Despite the historical name, this
    loader returns the full filtered junior panel (old + new pooled),
    not a new-only subset.

    Returns:
        List of article records containing student survey responses.
    """
    return load_jsonl(STUDENT_NEW_PATH)


def load_student_merged_ratings() -> List[Dict[str, Any]]:
    """Load the filtered combined junior ratings.

    Returns:
        List of article records containing filtered junior survey responses.
    """
    return load_jsonl(STUDENT_MERGED_PATH)


def load_student_old_ratings() -> List[Dict[str, Any]]:
    """Load the unfiltered combined junior ratings.

    Deprecated compatibility alias. Despite the historical name, this
    loader returns the full unfiltered junior panel, not an old-only subset.

    Returns:
        List of article records containing unfiltered junior survey responses.
    """
    return load_jsonl(STUDENT_OLD_PATH)


def load_student_filtered_ratings() -> List[Dict[str, Any]]:
    """Load the final filtered combined junior panel (old + new pooled)."""
    return load_jsonl(STUDENT_FILTERED_PATH)


def load_all_combined() -> List[Dict[str, Any]]:
    """Load ``all_combined.jsonl`` (all human ratings combined).

    Returns:
        List of article records containing all survey responses.
    """
    return load_jsonl(ALL_COMBINED_PATH)


def load_unfiltered_expert_ratings() -> List[Dict[str, Any]]:
    """Load unfiltered expert ratings (47+ experts).

    Note: now identical to :func:`load_expert_ratings` since unfiltered
    is the primary dataset.
    """
    return load_jsonl(UNFILTERED_EXPERT_PATH)


def load_unfiltered_student_ratings() -> List[Dict[str, Any]]:
    """Load unfiltered student ratings (190 students)."""
    return load_jsonl(UNFILTERED_STUDENT_PATH)


def load_enriched_expert(filtered: bool = True) -> List[Dict[str, Any]]:
    """Load enriched expert data.

    Parameters:
        filtered: If True, load filtered (39 experts); else all 48.
    """
    path = ENRICHED_EXPERT_FILTERED_PATH if filtered else ENRICHED_EXPERT_PATH
    return load_jsonl(path)


def load_enriched_student(filtered: bool = True) -> List[Dict[str, Any]]:
    """Load enriched student data.

    Parameters:
        filtered: If True, load filtered (175 students); else all.
    """
    path = ENRICHED_STUDENT_FILTERED_PATH if filtered else ENRICHED_STUDENT_PATH
    return load_jsonl(path)


# ---------------------------------------------------------------------------
# Rating extraction
# ---------------------------------------------------------------------------

def extract_individual_ratings(
    article_data: Dict[str, Any],
    rater_type_filter: Optional[str] = None,
) -> List[Tuple[str, str, Optional[float], Optional[float], Optional[bool], Optional[float]]]:
    """Extract individual ratings from a single article record.

    Parameters:
        article_data: A single article dict from the human-ratings JSONL.
            Expected to contain a ``ratings`` (or ``surveys``) list where
            each entry has at least ``rater_name``, ``q2_rating`` (or
            ``rating``), and optionally ``confidence``, ``familiarity``,
            ``read_before``, ``duration``, and ``rater_type``.
        rater_type_filter: If provided, only include ratings from raters
            whose ``rater_type`` matches this value (``'expert'`` or
            ``'student'``).

    Returns:
        List of tuples ``(rater_name, rating_level, confidence,
        familiarity, read_before, duration)``.  ``rating_level`` is
        normalised to AI label space (e.g. ``'exceptional'``).
    """
    surveys: List[Dict[str, Any]] = article_data.get(
        'ratings', article_data.get('surveys', [])
    )

    results: List[Tuple[str, str, Optional[float], Optional[float], Optional[bool], Optional[float]]] = []

    for entry in surveys:
        # Filter by rater type when requested
        if rater_type_filter is not None:
            rater_type = entry.get('rater_type', '')
            if rater_type != rater_type_filter:
                continue

        rater_name: str = entry.get('rater_name', entry.get('name', 'unknown'))

        # Resolve the rating string from various field names
        raw_rating: str = entry.get(
            'q2_rating',
            entry.get('rating', entry.get('level', '')),
        )
        rating_level: str = normalize_level(raw_rating)

        confidence: Optional[float] = entry.get('confidence')
        familiarity: Optional[float] = entry.get('familiarity')
        read_before: Optional[bool] = entry.get('read_before')
        duration: Optional[float] = entry.get('duration')

        results.append((
            rater_name,
            rating_level,
            confidence,
            familiarity,
            read_before,
            duration,
        ))

    return results


# ---------------------------------------------------------------------------
# Index building
# ---------------------------------------------------------------------------

def build_article_index(
    articles: List[Dict[str, Any]],
) -> Dict[str, Dict[str, Any]]:
    """Build lookup dictionaries for articles.

    Parameters:
        articles: List of article dicts (from :func:`load_articles`).

    Returns:
        A dict with two sub-dicts:
        ``{'by_title': {title: article}, 'by_number': {number: article}}``.
        ``number`` keys are stored as strings for uniform access.
    """
    by_title: Dict[str, Dict[str, Any]] = {}
    by_number: Dict[str, Dict[str, Any]] = {}

    for article in articles:
        title = article.get('title', '')
        if title:
            by_title[title] = article

        number = article.get('article_number', article.get('number'))
        if number is not None:
            by_number[str(number)] = article

    return {
        'by_title': by_title,
        'by_number': by_number,
    }


# ---------------------------------------------------------------------------
# Label normalisation
# ---------------------------------------------------------------------------

def normalize_level(level_str: str) -> str:
    """Normalise a level/rating string to the AI label space.

    Handles all source conventions:
    - Lowercase level fields: ``'top'`` -> ``'exceptional'``
    - Human survey ratings:   ``'Top'`` -> ``'exceptional'``
    - Already in AI space:    ``'exceptional'`` -> ``'exceptional'``

    Parameters:
        level_str: Raw level or rating string from any data source.

    Returns:
        Normalised label from :data:`constants.LABEL_ORDER`.
    """
    if not level_str:
        return level_str

    stripped = level_str.strip()

    # Already in AI label space
    if stripped.lower() in {'exceptional', 'strong', 'limited'}:
        return stripped.lower()

    # Check lowercase level mapping first (handles 'top', 'top-', 'good', 'fair')
    lower = stripped.lower()
    if lower in LEVEL_TO_AI:
        return LEVEL_TO_AI[lower]

    # Check human survey mapping (handles 'Top', 'Top-', 'Good', 'Fair')
    if stripped in HUMAN_TO_AI:
        return HUMAN_TO_AI[stripped]

    # Edge case: 'fair' is both a level key and an AI label.
    # LEVEL_TO_AI maps 'fair' -> 'limited', which is correct for article
    # levels. If the value is already 'fair' in AI label space, it would
    # have been caught by the first check only if the original string is
    # exactly 'fair'. Since LEVEL_TO_AI takes precedence for article-
    # sourced data, callers working with AI-space predictions should not
    # pass raw 'fair' through this function.
    return stripped.lower()


def get_ground_truth(article: Dict[str, Any]) -> str:
    """Extract the ground-truth label from an article dict.

    Handles both ``rank`` fields (already in AI label space) and ``level``
    fields (need normalisation).

    Parameters:
        article: A single article dict.

    Returns:
        Ground-truth label in AI label space (e.g. ``'exceptional'``).

    Raises:
        KeyError: If neither ``rank`` nor ``level`` is present.
    """
    if 'rank' in article:
        return article['rank'].strip().lower()

    if 'level' in article:
        return normalize_level(article['level'])

    raise KeyError(
        f"Article has neither 'rank' nor 'level' field. "
        f"Available keys: {list(article.keys())}"
    )
