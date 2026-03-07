"""Shared protocol helpers for canonical frontier-model evaluation.

Primary protocol:
- stochastic frontier models: article-level avg8 accuracy (``avg_accuracy``)
- deterministic frontier models: single-pass discrete prediction

Discrete analyses (confusion/per-tier/McNemar) use majority labels by default.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .constants import FRONTIER_MODELS, FRONTIER_PATH, LABEL_ORDER
from .data_loader import get_ground_truth, load_jsonl


VALID_LABELS = set(LABEL_ORDER)

# Legacy aliases observed across historical files.
MODEL_KEY_ALIASES: Dict[str, str] = {
    "openai/gpt-5.2": "openai/gpt-5.2-high",
    "openai/gpt-5.2-pro": "openai/gpt-5.2-high",
    "deepseek/deepseek-v3.2": "deepseek/deepseek-v3.2-speciale",
    "google/gemini-2.5-pro-preview-05-06": "google/gemini-2.5-pro",
}

CANONICAL_TO_SOURCE_KEYS: Dict[str, List[str]] = {k: [k] for k in FRONTIER_MODELS}
for alias, canonical in MODEL_KEY_ALIASES.items():
    CANONICAL_TO_SOURCE_KEYS.setdefault(canonical, [canonical])
    if alias not in CANONICAL_TO_SOURCE_KEYS[canonical]:
        CANONICAL_TO_SOURCE_KEYS[canonical].append(alias)


def normalize_label(raw: Any) -> Optional[str]:
    """Normalize raw label text into canonical 4-tier label space."""
    if raw is None:
        return None
    x = str(raw).strip().strip("*").strip().lower()
    mapping = {
        "exceptional": "exceptional",
        "strong": "strong",
        "fair": "fair",
        "limited": "limited",
        "top": "exceptional",
        "top-": "strong",
        "good": "fair",
    }
    return mapping.get(x)


def canonicalize_model_key(model_key: str) -> str:
    """Map known aliases to canonical frontier model keys."""
    return MODEL_KEY_ALIASES.get(model_key, model_key)


def _prediction_from_logp(logp: Any) -> Optional[str]:
    if not isinstance(logp, dict) or not logp:
        return None
    best_label = None
    best_score = float("-inf")
    for raw_label, raw_score in logp.items():
        label = normalize_label(raw_label)
        if label not in VALID_LABELS:
            continue
        try:
            score = float(raw_score)
        except (TypeError, ValueError):
            continue
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def _compute_majority_from_counts(vote_counts: Any) -> Tuple[Optional[str], bool]:
    """Return (majority_label, is_tie) from vote_counts."""
    if not isinstance(vote_counts, dict) or not vote_counts:
        return (None, False)
    normalized: Dict[str, int] = {}
    for raw_label, raw_count in vote_counts.items():
        label = normalize_label(raw_label)
        if label not in VALID_LABELS:
            continue
        try:
            count = int(raw_count)
        except (TypeError, ValueError):
            continue
        normalized[label] = count
    if not normalized:
        return (None, False)
    top_count = max(normalized.values())
    top_labels = sorted([k for k, v in normalized.items() if v == top_count])
    is_tie = len(top_labels) > 1
    return (None if is_tie else top_labels[0], is_tie)


def _infer_mode(model_output: Dict[str, Any]) -> str:
    mode = model_output.get("mode")
    if mode in {"stochastic_thinking", "deterministic_logp"}:
        return mode
    has_votes = (
        isinstance(model_output.get("vote_predictions"), list)
        or model_output.get("avg_accuracy") is not None
        or model_output.get("vote_valid_n") is not None
    )
    return "stochastic_thinking" if has_votes else "deterministic_logp"


def _resolve_model_output(article: Dict[str, Any], model_key: str) -> Dict[str, Any]:
    rq = article.get("val_outcome", {}).get("rq_with_context", {})
    for source_key in CANONICAL_TO_SOURCE_KEYS.get(model_key, [model_key]):
        if source_key in rq:
            return rq[source_key] or {}
    return {}


def extract_discrete_prediction(
    model_output: Dict[str, Any],
    *,
    policy: str = "majority",
    tie_policy: str = "exclude",
) -> Tuple[Optional[str], bool]:
    """Extract one discrete prediction from canonical model output.

    Args:
        model_output: per-model payload in canonical frontier file.
        policy:
            - ``majority`` (default): use ``prediction_majority``.
            - ``run1``: use the first valid vote from ``vote_predictions``.
            - ``prediction``: use backward-compatible ``prediction`` field.
        tie_policy:
            - ``exclude`` (default): if tied majority, return ``None``.
            - ``run1_fallback``: if tied, fall back to run1/prediction.
    """
    vote_counts = model_output.get("vote_counts")
    computed_majority, computed_tie = _compute_majority_from_counts(vote_counts)

    majority = normalize_label(model_output.get("prediction_majority"))
    if majority is None:
        majority = computed_majority

    tie_flag = model_output.get("vote_is_tie")
    if isinstance(tie_flag, bool):
        is_tie = tie_flag
    else:
        is_tie = computed_tie

    run1 = normalize_label(model_output.get("prediction_run1"))
    if run1 is None:
        vp = model_output.get("vote_predictions")
        if isinstance(vp, list):
            for raw in vp:
                run1 = normalize_label(raw)
                if run1 is not None:
                    break

    prediction = normalize_label(model_output.get("prediction"))
    if prediction is None:
        prediction = _prediction_from_logp(model_output.get("logp"))

    if policy == "majority":
        if not is_tie and majority is not None:
            return (majority, False)
        if tie_policy == "run1_fallback":
            return (run1 or prediction, True)
        return (None, True)

    if policy == "run1":
        return (run1 or prediction or majority, is_tie)

    if policy == "prediction":
        return (prediction or majority or run1, is_tie)

    raise ValueError(f"Unsupported discrete policy: {policy}")


def load_frontier_records(path: str = FRONTIER_PATH) -> List[Dict[str, Any]]:
    """Load canonical frontier records."""
    return load_jsonl(path)


def list_frontier_models(records: Sequence[Dict[str, Any]]) -> List[str]:
    """Return canonical model keys from frontier records."""
    if not records:
        return []
    rq = records[0].get("val_outcome", {}).get("rq_with_context", {})
    return [canonicalize_model_key(k) for k in rq.keys()]


def assert_frontier_models_present(
    records: Sequence[Dict[str, Any]],
    expected_models: Optional[Iterable[str]] = None,
) -> None:
    expected = list(expected_models or FRONTIER_MODELS)
    observed = set(list_frontier_models(records))
    missing = [m for m in expected if m not in observed]
    if missing:
        raise ValueError(f"Missing canonical frontier models: {missing}")


def model_primary_accuracy(records: Sequence[Dict[str, Any]], model_key: str) -> float:
    """Compute model accuracy under primary protocol metric."""
    values: List[float] = []
    for rec in records:
        gt = get_ground_truth(rec)
        model_output = _resolve_model_output(rec, model_key)
        mode = _infer_mode(model_output)

        if mode == "stochastic_thinking":
            avg_acc = model_output.get("avg_accuracy")
            if avg_acc is not None:
                try:
                    values.append(float(avg_acc))
                    continue
                except (TypeError, ValueError):
                    pass

        pred, _ = extract_discrete_prediction(
            model_output,
            policy="prediction",
            tie_policy="run1_fallback",
        )
        if pred is not None:
            values.append(1.0 if pred == gt else 0.0)

    return float(sum(values) / len(values)) if values else 0.0


def frontier_average_primary(
    records: Sequence[Dict[str, Any]],
    model_keys: Optional[Sequence[str]] = None,
) -> float:
    """Mean primary accuracy across frontier models."""
    keys = list(model_keys or FRONTIER_MODELS)
    if not keys:
        return 0.0
    per_model = [model_primary_accuracy(records, k) for k in keys]
    return float(sum(per_model) / len(per_model))


def build_prediction_vectors(
    records: Sequence[Dict[str, Any]],
    model_key: str,
    *,
    policy: str = "majority",
    tie_policy: str = "exclude",
) -> Tuple[List[str], List[str], Dict[str, int]]:
    """Build paired prediction vectors for one frontier model.

    Returns:
        y_true, y_pred, metadata with tie/coverage counters.
    """
    y_true: List[str] = []
    y_pred: List[str] = []
    n_ties = 0
    n_missing = 0
    n_total = 0

    for rec in records:
        n_total += 1
        gt = get_ground_truth(rec)
        model_output = _resolve_model_output(rec, model_key)
        pred, is_tie = extract_discrete_prediction(
            model_output,
            policy=policy,
            tie_policy=tie_policy,
        )
        if is_tie:
            n_ties += 1
        if pred is None:
            n_missing += 1
            continue
        y_true.append(gt)
        y_pred.append(pred)

    meta = {
        "n_total": n_total,
        "n_valid": len(y_pred),
        "n_ties": n_ties,
        "n_missing": n_missing,
    }
    return y_true, y_pred, meta


def full_predictions(
    records: Sequence[Dict[str, Any]],
    model_key: str,
    *,
    policy: str = "majority",
    tie_policy: str = "exclude",
) -> List[Optional[str]]:
    """Return per-article predictions aligned to input record order."""
    out: List[Optional[str]] = []
    for rec in records:
        model_output = _resolve_model_output(rec, model_key)
        pred, _ = extract_discrete_prediction(
            model_output,
            policy=policy,
            tie_policy=tie_policy,
        )
        out.append(pred)
    return out


def model_mode(records: Sequence[Dict[str, Any]], model_key: str) -> str:
    """Return dominant protocol mode for a frontier model."""
    if not records:
        return "deterministic_logp"
    modes: List[str] = []
    for rec in records:
        mode = _infer_mode(_resolve_model_output(rec, model_key))
        modes.append(mode)
    if not modes:
        return "deterministic_logp"
    n_stochastic = sum(1 for m in modes if m == "stochastic_thinking")
    return "stochastic_thinking" if n_stochastic >= (len(modes) / 2.0) else "deterministic_logp"


def predictions_by_title(
    records: Sequence[Dict[str, Any]],
    model_key: str,
    *,
    policy: str = "majority",
    tie_policy: str = "exclude",
) -> Dict[str, Optional[str]]:
    """Map article title -> prediction under a given frontier discrete policy."""
    out: Dict[str, Optional[str]] = {}
    for rec in records:
        title = rec.get("title", "")
        model_output = _resolve_model_output(rec, model_key)
        pred, _ = extract_discrete_prediction(
            model_output,
            policy=policy,
            tie_policy=tie_policy,
        )
        out[title] = pred
    return out


def best_model_by_primary(
    records: Sequence[Dict[str, Any]],
    model_keys: Optional[Sequence[str]] = None,
) -> Tuple[str, float]:
    """Return the best frontier model under primary metric."""
    keys = list(model_keys or FRONTIER_MODELS)
    if not keys:
        raise ValueError("No model keys provided.")
    scored = [(k, model_primary_accuracy(records, k)) for k in keys]
    scored.sort(key=lambda x: x[1], reverse=True)
    return scored[0]
