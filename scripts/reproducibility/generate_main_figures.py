#!/usr/bin/env python3
"""Generate clean final main figures (Figure 2-5) for the Nature draft.

This script rebuilds figures from numeric data only and does not reuse any prior
image assets.

Outputs:
- PNG/PDF saved to: ../reproduced/figures/main

Usage:
    python3 scripts/reproducibility/generate_main_figures.py
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from figure_style_policy import (
    apply_title_policy,
    panel_title,
    place_reference_label_safely,
)


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "README.md").exists() and (candidate / "scripts").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError("Could not locate package root (expected README.md + scripts/ + data/).")


ROOT = find_project_root()
DATA_DIR = ROOT / "data"
STATS_DIR = DATA_DIR / "statistics"
TABLES_DIR = DATA_DIR / "tables"
OUTPUT_DIR = ROOT / "reproduced" / "figures" / "main"
REPORTS_ROOT = ROOT

SFT_FILE = REPORTS_ROOT / "data" / "predictions" / "sft_predictions.jsonl"
CHAT_FILE = REPORTS_ROOT / "data" / "predictions" / "chat_predictions.jsonl"

FIG_WIDTH = 7.2  # Nature double-column width equivalent (inches)

PALETTE = {
    "sft": "#0072B2",
    "frontier": "#D55E00",
    "expert": "#009E73",
    "junior": "#E69F00",
    "calibration": "#CC79A7",
    "neutral": "#4D4D4D",
    "chance": "#999999",
    "tier_exceptional": "#0072B2",
    "tier_strong": "#56B4E9",
    "tier_fair": "#E69F00",
    "tier_limited": "#D55E00",
}

TIER_ORDER = ["exceptional", "strong", "fair", "limited"]
TIER_LABELS = ["Exceptional", "Strong", "Fair", "Limited"]
PAIRWISE_ROOT = DATA_DIR / "pairwise"
PAIRWISE_METRIC_DIRS = {
    "SFT GPT-4.1": PAIRWISE_ROOT / "sft_gpt4_1",
    "Gemini 3.1 Pro": PAIRWISE_ROOT / "frontier_gemini3_1_pro",
    "GPT-5.2 High": PAIRWISE_ROOT / "frontier_gpt5_2_high",
    "GPT-4.1 Baseline": PAIRWISE_ROOT / "baseline_gpt4_1",
}
PAIRWISE_SHORT_LABELS = {
    "SFT GPT-4.1": "SFT GPT-4.1",
    "Gemini 3.1 Pro": "Gemini 3.1",
    "GPT-5.2 High": "GPT-5.2 High",
    "GPT-4.1 Baseline": "GPT-4.1 base",
}
PAIRWISE_COLORS = {
    "SFT GPT-4.1": PALETTE["sft"],
    "Gemini 3.1 Pro": "#009E73",
    "GPT-5.2 High": "#56B4E9",
    "GPT-4.1 Baseline": PALETTE["frontier"],
}


def set_style() -> None:
    """Set a compact Nature-like style for all figures."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.5,
            "axes.titlesize": 7.5,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.0,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_json(name: str) -> Dict:
    with (STATS_DIR / f"{name}.json").open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(TABLES_DIR / name)


def _safe_float(value: object) -> float:
    try:
        if value is None:
            return float("nan")
        if isinstance(value, str) and value.strip().lower() in {"", "na", "n/a", "none"}:
            return float("nan")
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def _to_percent_maybe(value: object) -> float:
    v = _safe_float(value)
    if not np.isfinite(v):
        return float("nan")
    return v * 100.0 if abs(v) <= 1.0 else v


def resolve_existing(candidates: List[Path], label: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not locate {label}. Checked: {[str(p) for p in candidates]}")


def _supports_person_level_linkage(path: Path) -> bool:
    """Return True when a ratings file contains repeatable per-person IDs."""
    seen: Dict[str, int] = {}
    try:
        with path.open("r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                rec = json.loads(line)
                for rating in rec.get("ratings", []):
                    rid = str(rating.get("rater_name", "")).strip()
                    if not rid:
                        rid = str(rating.get("response_id", "")).strip()
                    if not rid:
                        rid = str(rating.get("rater_id", "")).strip()
                    if not rid:
                        continue
                    seen[rid] = seen.get(rid, 0) + 1
                    if seen[rid] > 1:
                        return True
    except OSError:
        return False
    return False


def resolve_person_level_source(candidates: List[Path], label: str) -> Path:
    """Resolve a human-ratings file that supports per-person aggregation."""
    checked: List[str] = []
    for p in candidates:
        if not p.exists():
            checked.append(f"{p} (missing)")
            continue
        if _supports_person_level_linkage(p):
            return p
        checked.append(f"{p} (no repeatable person id)")
    raise RuntimeError(
        f"Could not locate person-linked source for {label}. Checked: {checked}"
    )


def save_figure(fig: plt.Figure, basename: str) -> Path:
    apply_title_policy(fig, family="main")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUTPUT_DIR / f"{basename}.png"
    pdf_path = OUTPUT_DIR / f"{basename}.pdf"
    fig.savefig(png_path, bbox_inches="tight", dpi=300)
    fig.savefig(pdf_path, bbox_inches="tight")
    return png_path


def binomial_ci(p: np.ndarray, n: np.ndarray, z: float = 1.96) -> np.ndarray:
    """Normal approximation 95% CI half-width for binomial proportions."""
    n_safe = np.maximum(n, 1)
    return z * np.sqrt((p * (1.0 - p)) / n_safe)


def _format_p_value(p: float) -> str:
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _softmax_from_logp(logp_dict: Dict[str, float]) -> Dict[str, float]:
    labels = list(logp_dict.keys())
    vals = np.array([float(logp_dict[k]) for k in labels], dtype=float)
    vals = vals - np.max(vals)
    probs = np.exp(vals)
    probs = probs / probs.sum()
    return {label: float(prob) for label, prob in zip(labels, probs)}


def _canonical_label(level: str) -> str:
    mapping = {
        "top": "exceptional",
        "top-": "strong",
        "good": "fair",
        "fair": "limited",
    }
    return mapping.get(str(level).strip().lower(), "")


def _to_binary(value: object) -> int | None:
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (int, float)):
        return int(value > 0)
    if isinstance(value, str):
        raw = value.strip().lower()
        if raw in {"true", "1", "yes"}:
            return 1
        if raw in {"false", "0", "no"}:
            return 0
    return None


def _extract_leading_int(value: object) -> int | None:
    if value is None:
        return None
    raw = str(value).strip()
    if not raw:
        return None
    first = raw[0]
    if first.isdigit():
        parsed = int(first)
        if 1 <= parsed <= 5:
            return parsed
    return None


def _normalize_confidence_1_to_5(value: float) -> float:
    """Map 1-5 confidence to 0-1 via (x - 1) / 4."""
    return (float(value) - 1.0) / 4.0


def _extract_truth_label(record: Dict) -> str:
    rank = str(record.get("rank", "")).strip().lower()
    if rank:
        return rank
    return _canonical_label(record.get("level", ""))


def _prediction_from_logp_label_order(logp_dict: Dict[str, object]) -> str | None:
    best_label: str | None = None
    best_score = float("-inf")
    for label in TIER_ORDER:
        try:
            score = float(logp_dict.get(label, float("-inf")))
        except (TypeError, ValueError):
            score = float("-inf")
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def _load_logp_predictions(path: Path, model_key: str) -> Tuple[np.ndarray, np.ndarray]:
    idx = {label: i for i, label in enumerate(TIER_ORDER)}
    y_true: List[int] = []
    y_pred: List[int] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gt = _extract_truth_label(rec)
            if gt not in idx:
                continue
            entry = rec.get("val_outcome", {}).get("rq_with_context", {}).get(model_key, {})
            logp = entry.get("logp", {}) if isinstance(entry, dict) else {}
            if not isinstance(logp, dict) or not logp:
                continue
            pred = _prediction_from_logp_label_order(logp)
            if pred not in idx:
                continue
            y_true.append(idx[gt])
            y_pred.append(idx[pred])
    return np.asarray(y_true, dtype=int), np.asarray(y_pred, dtype=int)


def _macro_f1_from_codes(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    if y_true.size == 0 or y_pred.size == 0:
        return float("nan")
    n_classes = len(TIER_ORDER)
    f1_scores = np.zeros(n_classes, dtype=float)
    for cls_idx in range(n_classes):
        tp = float(np.sum((y_true == cls_idx) & (y_pred == cls_idx)))
        fp = float(np.sum((y_true != cls_idx) & (y_pred == cls_idx)))
        fn = float(np.sum((y_true == cls_idx) & (y_pred != cls_idx)))
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        if (precision + recall) > 0:
            f1_scores[cls_idx] = 2.0 * precision * recall / (precision + recall)
    return float(np.mean(f1_scores))


def _bootstrap_classifier_cis(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    n_bootstrap: int = 3000,
    seed: int = 42,
) -> Dict[str, Tuple[float, float]]:
    if y_true.size == 0 or y_pred.size == 0:
        return {"acc": (float("nan"), float("nan")), "macro_f1": (float("nan"), float("nan"))}
    n = min(y_true.size, y_pred.size)
    y_true = y_true[:n]
    y_pred = y_pred[:n]

    rng = np.random.default_rng(seed)
    acc_samples = np.empty(n_bootstrap, dtype=float)
    f1_samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        sample_idx = rng.integers(0, n, size=n)
        yt = y_true[sample_idx]
        yp = y_pred[sample_idx]
        acc_samples[i] = float(np.mean(yt == yp))
        f1_samples[i] = _macro_f1_from_codes(yt, yp)

    acc_lo, acc_hi = np.percentile(acc_samples, [2.5, 97.5])
    f1_lo, f1_hi = np.percentile(f1_samples, [2.5, 97.5])
    return {
        "acc": (float(acc_lo), float(acc_hi)),
        "macro_f1": (float(f1_lo), float(f1_hi)),
    }


def _summarize_logp_model(path: Path, model_key: str, *, seed: int) -> Dict[str, float]:
    y_true, y_pred = _load_logp_predictions(path, model_key)
    if y_true.size == 0:
        raise ValueError(f"No valid logp predictions found for model '{model_key}' in {path}")
    acc = float(np.mean(y_true == y_pred))
    macro_f1 = _macro_f1_from_codes(y_true, y_pred)
    cis = _bootstrap_classifier_cis(y_true, y_pred, seed=seed)
    acc_lo, acc_hi = cis["acc"]
    f1_lo, f1_hi = cis["macro_f1"]
    return {
        "n": float(y_true.size),
        "acc": acc,
        "acc_lo": acc_lo,
        "acc_hi": acc_hi,
        "macro_f1": macro_f1,
        "macro_f1_lo": f1_lo,
        "macro_f1_hi": f1_hi,
    }


def _collect_logp_confidence_records(path: Path, model_key: str) -> List[Tuple[float, int]]:
    """Collect (max_confidence, correct) tuples for one model with logp."""
    rows: List[Tuple[float, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gt = _extract_truth_label(rec)
            if not gt:
                continue
            entry = rec.get("val_outcome", {}).get("rq_with_context", {}).get(model_key, {})
            logp = entry.get("logp", {})
            if not logp:
                continue
            probs = _softmax_from_logp(logp)
            pred = max(probs, key=probs.get)
            rows.append((float(probs[pred]), int(pred == gt)))
    return rows


def _collect_human_voting_confidence_records(path: Path) -> List[Tuple[float, int]]:
    """Collect article-level (agreement confidence, correct) tuples for human voting."""
    rows: List[Tuple[float, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gt = _canonical_label(rec.get("level", ""))
            if not gt:
                continue
            labels: List[str] = []
            for rating in rec.get("ratings", []):
                pred = _canonical_label(rating.get("q2_rating", ""))
                if pred:
                    labels.append(pred)
            if not labels:
                continue
            counts = Counter(labels)
            pred = max(counts, key=counts.get)
            conf = counts[pred] / max(len(labels), 1)
            rows.append((float(conf), int(pred == gt)))
    return rows


def _collect_human_rating_confidence_records(path: Path) -> List[Tuple[float, int, int]]:
    """Collect rating-level (conf_norm, conf_raw, correct) tuples for humans."""
    rows: List[Tuple[float, int, int]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gt = _canonical_label(rec.get("level", ""))
            if not gt:
                continue
            for rating in rec.get("ratings", []):
                pred = _canonical_label(rating.get("q2_rating", ""))
                conf_raw = _extract_leading_int(rating.get("q3_confidence"))
                if not pred or conf_raw is None:
                    continue
                conf_norm = _normalize_confidence_1_to_5(conf_raw)
                rows.append((float(conf_norm), int(conf_raw), int(pred == gt)))
    return rows


def _selective_curve_from_confidence(records: List[Tuple[float, int]]) -> pd.DataFrame:
    """Cumulative accuracy of highest-confidence predictions first."""
    if not records:
        return pd.DataFrame(columns=["coverage", "accuracy"])

    arr = np.asarray(records, dtype=float)
    order = np.argsort(-arr[:, 0])
    sorted_rows = arr[order]
    correct = sorted_rows[:, 1]

    n = len(sorted_rows)
    coverage = np.arange(1, n + 1) / n * 100.0
    accuracy = np.cumsum(correct) / np.arange(1, n + 1) * 100.0
    return pd.DataFrame({"coverage": coverage, "accuracy": accuracy})


def _format_pct_label(value: float) -> str:
    """Format a percentage for compact annotation labels."""
    rounded = round(float(value))
    if abs(float(value) - rounded) < 0.05:
        return f"{rounded:.0f}%"
    return f"{float(value):.1f}%"


def _selective_curve_callouts(curve: pd.DataFrame) -> List[Dict[str, object]]:
    """Pick stable, data-driven callouts for the SFT selective-prediction curve."""
    if len(curve) == 0:
        return []

    curve = curve.reset_index(drop=True)
    callouts: List[Dict[str, object]] = []

    perfect = curve[np.isclose(curve["accuracy"].to_numpy(dtype=float), 100.0)]
    if len(perfect) > 0:
        row = perfect.iloc[-1]
        callouts.append(
            {
                "coverage": float(row["coverage"]),
                "accuracy": float(row["accuracy"]),
                "xytext": (7, -2),
                "ha": "left",
                "va": "top",
            }
        )

    high_cov = curve[curve["coverage"] >= 10.0]
    if len(high_cov) == 0:
        high_cov = curve
    above_80 = high_cov[high_cov["accuracy"] >= 80.0]
    if len(above_80) > 0:
        row = above_80.iloc[-1]
    else:
        idx = int(np.argmin(np.abs(high_cov["accuracy"].to_numpy(dtype=float) - 80.0)))
        row = high_cov.iloc[idx]
    callouts.append(
        {
            "coverage": float(row["coverage"]),
            "accuracy": float(row["accuracy"]),
            "xytext": (8, 10),
            "ha": "left",
            "va": "bottom",
        }
    )

    row = curve.iloc[-1]
    callouts.append(
        {
            "coverage": float(row["coverage"]),
            "accuracy": float(row["accuracy"]),
            "xytext": (-8, 6),
            "ha": "right",
            "va": "bottom",
        }
    )

    deduped: List[Dict[str, object]] = []
    seen: set[Tuple[int, int]] = set()
    for item in callouts:
        key = (int(round(float(item["coverage"]) * 10)), int(round(float(item["accuracy"]) * 10)))
        if key in seen:
            continue
        seen.add(key)
        deduped.append(item)
    return deduped


def _human_calibration_by_level(records: List[Tuple[float, int, int]]) -> List[Dict[str, float]]:
    """Return calibration points by self-reported confidence level (1..5)."""
    if not records:
        return []

    points: List[Dict[str, float]] = []
    for level in range(1, 6):
        kept = [r for r in records if r[1] == level]
        if not kept:
            continue
        points.append(
            {
                "level": level,
                "mean_conf": _normalize_confidence_1_to_5(level),
                "frac_correct": float(np.mean([r[2] for r in kept])),
                "count": len(kept),
            }
        )
    return points


def _human_selective_curve(records: List[Tuple[float, int, int]]) -> pd.DataFrame:
    """Selective curve using confidence thresholds >= {1,2,3,4,5}."""
    if not records:
        return pd.DataFrame(columns=["coverage", "accuracy", "threshold"])

    n_total = len(records)
    rows: List[Tuple[float, float, int]] = []
    for threshold in range(1, 6):
        kept = [r for r in records if r[1] >= threshold]
        if not kept:
            continue
        coverage = len(kept) / n_total * 100.0
        accuracy = float(np.mean([r[2] for r in kept])) * 100.0
        rows.append((coverage, accuracy, threshold))

    rows.sort(key=lambda x: x[0])  # low coverage/high threshold first
    return pd.DataFrame(
        {
            "coverage": [r[0] for r in rows],
            "accuracy": [r[1] for r in rows],
            "threshold": [r[2] for r in rows],
        }
    )


def _load_figure5_pairwise_bundle() -> Dict[str, object]:
    stats = load_json("S14_ED2PairwiseRawPValues")
    order = [str(model) for model in stats.get("summary", {}).get("plotted_models", [])]
    if not order:
        raise ValueError("S14_ED2PairwiseRawPValues.json is missing plotted_models for Figure 5.")

    metrics: Dict[str, Dict[str, float]] = {}
    for model in order:
        model_dir = PAIRWISE_METRIC_DIRS.get(model)
        if model_dir is None:
            raise ValueError(f"Unrecognized Figure 5 pairwise model label: {model}")
        metrics_path = model_dir / "metrics.json"
        if not metrics_path.exists():
            raise FileNotFoundError(f"Missing Figure 5 pairwise metrics file: {metrics_path}")
        payload = json.loads(metrics_path.read_text(encoding="utf-8"))
        metrics[model] = {
            "weighted_accuracy": float(payload["weighted_accuracy"]),
            "total": float(payload["total"]),
        }

    comparisons = {
        str(item["evaluator_2"]): item
        for item in stats.get("comparisons", [])
        if str(item.get("evaluator_1", "")) == "SFT GPT-4.1"
    }
    return {"stats": stats, "order": order, "metrics": metrics, "comparisons": comparisons}


def _spearman_rho(x: np.ndarray, y: np.ndarray) -> float:
    if len(x) < 2 or len(y) < 2:
        return float("nan")
    rx = pd.Series(x).rank(method="average").to_numpy(dtype=float)
    ry = pd.Series(y).rank(method="average").to_numpy(dtype=float)
    if np.std(rx) == 0 or np.std(ry) == 0:
        return float("nan")
    return float(np.corrcoef(rx, ry)[0, 1])


def _bootstrap_spearman_ci(
    x: np.ndarray,
    y: np.ndarray,
    n_bootstrap: int = 3000,
    seed: int = 42,
) -> Tuple[float, float]:
    if len(x) == 0 or len(y) == 0:
        return float("nan"), float("nan")
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    rng = np.random.default_rng(seed)
    samples = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        samples[i] = _spearman_rho(x[idx], y[idx])

    samples = samples[np.isfinite(samples)]
    if len(samples) == 0:
        return float("nan"), float("nan")
    lo, hi = np.percentile(samples, [2.5, 97.5])
    return float(lo), float(hi)


def _spearman_permutation_p(
    x: np.ndarray,
    y: np.ndarray,
    n_permutations: int = 5000,
    seed: int = 42,
) -> float:
    """Two-sided permutation p-value for Spearman rho."""
    if len(x) < 3 or len(y) < 3:
        return float("nan")
    n = min(len(x), len(y))
    x = x[:n]
    y = y[:n]

    obs = _spearman_rho(x, y)
    if not np.isfinite(obs):
        return float("nan")
    obs_abs = abs(obs)

    rng = np.random.default_rng(seed)
    extreme = 0
    for _ in range(n_permutations):
        yp = y[rng.permutation(n)]
        rho_p = _spearman_rho(x, yp)
        if np.isfinite(rho_p) and abs(rho_p) >= obs_abs:
            extreme += 1

    return float((extreme + 1) / (n_permutations + 1))


def _load_rater_level_xy(path: Path, predictor_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (predictor_mean, accuracy) per rater from human ratings JSONL.

    Supports both enriched schema (`rater_name`, `is_correct`, `confidence_int`,
    `familiarity_int`) and de-identified schema (`rater_id`, `q2_rating`,
    `q3_confidence`, `q4_familiarity`).
    """
    by_rater: Dict[str, Dict[str, object]] = {}

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gt = _canonical_label(rec.get("level", ""))
            for rating in rec.get("ratings", []):
                name = str(rating.get("rater_name", "")).strip()
                if not name:
                    name = str(rating.get("rater_id", "")).strip()
                if not name:
                    continue

                correct = _to_binary(rating.get("is_correct"))
                if correct is None and gt:
                    pred_label = _canonical_label(rating.get("q2_rating", ""))
                    if pred_label:
                        correct = int(pred_label == gt)
                if correct is None:
                    continue

                pred = rating.get(predictor_key)
                if pred is None and predictor_key == "confidence_int":
                    pred = _extract_leading_int(rating.get("q3_confidence"))
                elif pred is None and predictor_key == "familiarity_int":
                    pred = _extract_leading_int(rating.get("q4_familiarity"))

                if name not in by_rater:
                    by_rater[name] = {"correct": 0, "total": 0, "preds": []}

                by_rater[name]["correct"] = int(by_rater[name]["correct"]) + int(correct)
                by_rater[name]["total"] = int(by_rater[name]["total"]) + 1
                if pred is not None:
                    try:
                        pred_f = float(pred)
                    except (TypeError, ValueError):
                        pred_f = float("nan")
                    if np.isfinite(pred_f):
                        by_rater[name]["preds"].append(pred_f)  # type: ignore[index]

    x_vals: List[float] = []
    y_vals: List[float] = []
    for info in by_rater.values():
        total = int(info["total"])
        preds = info["preds"]  # type: ignore[assignment]
        if total <= 0 or len(preds) == 0:
            continue
        x_vals.append(float(np.mean(preds)))
        y_vals.append(float(int(info["correct"]) / total))

    return np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float)


def _load_rating_level_xy(path: Path, predictor_key: str) -> Tuple[np.ndarray, np.ndarray]:
    """Return (predictor_value, correct) per rating from human ratings JSONL."""
    x_vals: List[float] = []
    y_vals: List[float] = []

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gt = _canonical_label(rec.get("level", ""))
            for rating in rec.get("ratings", []):
                correct = _to_binary(rating.get("is_correct"))
                if correct is None and gt:
                    pred_label = _canonical_label(rating.get("q2_rating", ""))
                    if pred_label:
                        correct = int(pred_label == gt)
                if correct is None:
                    continue

                pred = rating.get(predictor_key)
                if pred is None and predictor_key == "confidence_int":
                    pred = _extract_leading_int(rating.get("q3_confidence"))
                elif pred is None and predictor_key == "familiarity_int":
                    pred = _extract_leading_int(rating.get("q4_familiarity"))

                try:
                    pred_f = float(pred) if pred is not None else float("nan")
                except (TypeError, ValueError):
                    pred_f = float("nan")
                if not np.isfinite(pred_f):
                    continue

                x_vals.append(pred_f)
                y_vals.append(float(correct))

    return np.asarray(x_vals, dtype=float), np.asarray(y_vals, dtype=float)


def compute_sft_selective_curve() -> pd.DataFrame:
    """Compute selective prediction curve from article-level SFT ensemble logp."""
    rows: List[Tuple[float, int]] = []

    with SFT_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            true_label = _canonical_label(rec.get("level", ""))
            if not true_label:
                continue
            rq_out = rec.get("val_outcome", {}).get("rq_with_context", {})
            ens = rq_out.get("best_2_model_combo", {})
            logp = ens.get("logp", {})
            if not logp:
                continue

            probs = _softmax_from_logp(logp)
            pred = max(probs, key=probs.get)
            conf = probs[pred]
            correct = int(pred == true_label)
            rows.append((conf, correct))

    if not rows:
        return pd.DataFrame(columns=["coverage", "accuracy"])

    arr = np.array(rows, dtype=float)
    # Sort by confidence descending.
    idx = np.argsort(-arr[:, 0])
    conf_sorted = arr[idx, 0]
    correct_sorted = arr[idx, 1]

    n = len(correct_sorted)
    cumsum_correct = np.cumsum(correct_sorted)
    coverage = np.arange(1, n + 1) / n * 100.0
    accuracy = cumsum_correct / np.arange(1, n + 1) * 100.0

    return pd.DataFrame(
        {
            "coverage": coverage,
            "accuracy": accuracy,
            "confidence": conf_sorted,
        }
    )


def make_figure2() -> plt.Figure:
    """Figure 2: frontier collapse with leaderboard, distribution shift, and
    confusion anatomy for top flagship examples."""
    # Delegate to the dedicated Figure2 generator to keep panel-a metric pairing
    # (Accuracy CI + Macro-F1 CI) synchronized with the canonical Figure 2.
    try:
        from generate_main_figure2 import make_figure2

        return make_figure2()
    except Exception:
        # Fallback to in-file implementation below if the dedicated import fails.
        pass

    stats02 = load_json("S01_FlagshipStats")
    table3 = load_csv("T02_FlagshipPerformance.csv")

    df = table3.copy()
    df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

    short_name = {
        "Gemini 2.5 Pro": "Gemini 2.5",
        "Gemini 3.1 Pro": "Gemini 3.1*",
        "Kimi K2.5": "Kimi K2.5",
        "GPT-5.2 High": "GPT-5.2",
        "GPT-5.2 Pro": "GPT-5.2",
        "Grok 4.1 Fast": "Grok 4.1",
        "GLM-5": "GLM-5",
        "Claude Opus 4.6": "Opus 4.6",
        "MiniMax M2.5": "MiniMax M2.5",
        "Seed 2.0": "Seed 2.0",
        "DeepSeek V3.2": "DeepSeek V3.2",
        "Qwen 3.5 Plus": "Qwen 3.5",
    }

    def _resolve_model_name(name: str) -> str:
        aliases = {
            "GPT-5.2 High": "GPT-5.2 Pro",
            "GPT-5.2 Pro": "GPT-5.2 High",
        }
        perf = stats02["model_performance"]
        if name in perf:
            return name
        alt = aliases.get(name)
        if alt and alt in perf:
            return alt
        return name

    fig = plt.figure(figsize=(FIG_WIDTH, 8.0))
    outer = gridspec.GridSpec(2, 2, height_ratios=[1.0, 1.55], hspace=0.42, wspace=0.45)

    # Panel a: accuracy + macro-F1 leaderboard
    ax_a = fig.add_subplot(outer[0, 0])
    ax_a.set_title("a  Frontier leaderboard", loc="left")

    acc = df["Accuracy"].to_numpy(dtype=float)
    macro_f1 = df["Macro F1"].to_numpy(dtype=float)
    n_valid = df["N Valid"].to_numpy(dtype=float)
    ci = binomial_ci(acc, n_valid)
    y = np.arange(len(df))

    ax_a.errorbar(
        acc * 100,
        y,
        xerr=ci * 100,
        fmt="o",
        color=PALETTE["frontier"],
        ecolor=PALETTE["neutral"],
        elinewidth=0.8,
        capsize=2,
        markersize=3.8,
        zorder=3,
    )
    ax_a.axvline(25, color=PALETTE["chance"], linestyle="--", linewidth=0.8)
    ax_a.text(25.4, -0.85, "chance", fontsize=5.2, color=PALETTE["chance"])
    ax_a.set_yticks(y)
    ax_a.set_yticklabels([short_name.get(m, m) for m in df["Model"].tolist()])
    ax_a.set_xlabel("Accuracy (%)")
    ax_a.set_xlim(0, 62)
    ax_a.invert_yaxis()

    ax_a.text(61.5, -0.85, "Acc  |  Macro-F1", ha="right", va="bottom", fontsize=5.2)
    for i, (a, f1) in enumerate(zip(acc, macro_f1)):
        ax_a.text(61.5, i, f"{a*100:>4.1f}% | {f1:.3f}", ha="right", va="center", fontsize=5.0)

    mean_frontier = acc.mean() * 100
    ax_a.text(
        0.98,
        0.04,
        f"Frontier mean = {mean_frontier:.1f}%",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.4,
        bbox={"boxstyle": "round,pad=0.2", "facecolor": "white", "edgecolor": "#CCCCCC"},
    )

    # Panel b: prediction distributions with external legend
    ax_b = fig.add_subplot(outer[0, 1])
    ax_b.set_title("b  Predicted-tier distribution", loc="left")

    pred_cols = {
        "exceptional": "Pred count (exceptional)",
        "strong": "Pred count (strong)",
        "fair": "Pred count (fair)",
        "limited": "Pred count (limited)",
    }
    pred_counts = df[[pred_cols[t] for t in TIER_ORDER]].to_numpy(dtype=float)
    totals = pred_counts.sum(axis=1, keepdims=True)
    totals[totals == 0] = 1
    pred_pct = pred_counts / totals * 100.0

    x = np.arange(len(df))
    bottom = np.zeros(len(df))
    tier_colors = [
        PALETTE["tier_exceptional"],
        PALETTE["tier_strong"],
        PALETTE["tier_fair"],
        PALETTE["tier_limited"],
    ]
    for i, tier in enumerate(TIER_ORDER):
        ax_b.bar(
            x,
            pred_pct[:, i],
            bottom=bottom,
            width=0.72,
            color=tier_colors[i],
            edgecolor="white",
            linewidth=0.25,
            label=tier.capitalize(),
        )
        bottom += pred_pct[:, i]

    ax_b.axhline(25, color=PALETTE["chance"], linestyle="--", linewidth=0.7)
    ax_b.set_ylabel("Predicted share (%)")
    ax_b.set_ylim(0, 100)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([short_name.get(m, m) for m in df["Model"].tolist()], rotation=50, ha="right")
    ax_b.legend(
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        fontsize=5.2,
        title="Predicted tier",
        title_fontsize=5.3,
        borderaxespad=0.0,
    )

    # Panel c: top-4 confusion matrices with one shared colorbar
    ax_c_host = fig.add_subplot(outer[1, :])
    ax_c_host.axis("off")
    panel_title(ax_c_host, "c", None, family="main")

    inner = outer[1, :].subgridspec(2, 2, hspace=0.40, wspace=0.28)
    model_grid = [
        ("Gemini 3.1 Pro", "Best frontier*"),
        ("GPT-5.2 High", "Middle-tier clustering"),
        ("Claude Opus 4.6", "Strong-tier ceiling"),
        ("Qwen 3.5 Plus", "Best open-weight"),
    ]
    axis_labels = ["Top", "Top-", "Good", "Fair"]
    cm_axes = []
    im = None

    for idx, (model_name, subtitle) in enumerate(model_grid):
        r = idx // 2
        c = idx % 2
        ax = fig.add_subplot(inner[r, c])
        cm_axes.append(ax)

        resolved_name = _resolve_model_name(model_name)
        cm = np.array(stats02["model_performance"][resolved_name]["confusion_matrix"], dtype=float)
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm_norm = cm / row_sum

        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_title(f"{short_name.get(model_name, model_name)}\n{subtitle}", fontsize=5.4, pad=2)
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        if r == 0:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xticklabels(axis_labels, rotation=40, ha="right", fontsize=5.2)
        if c == 0:
            ax.set_yticklabels(axis_labels, fontsize=5.2)
            ax.set_ylabel("True", fontsize=5.6)
        else:
            ax.set_yticklabels(axis_labels, fontsize=5.2)
        if r == 1:
            ax.set_xlabel("Predicted", fontsize=5.6, labelpad=1)

        for rr in range(4):
            for cc in range(4):
                val = cm_norm[rr, cc]
                ax.text(
                    cc,
                    rr,
                    f"{val:.2f}",
                    ha="center",
                    va="center",
                    fontsize=5.0,
                    color="white" if val >= 0.5 else "black",
                )

    if im is not None:
        cbar = fig.colorbar(im, ax=cm_axes, fraction=0.015, pad=0.01)
        cbar.ax.tick_params(labelsize=5.0)
        cbar.set_label("Row-normalized share", fontsize=5.4)

    fig.subplots_adjust(left=0.10, right=0.86, bottom=0.08, top=0.94)
    return fig


def make_figure3() -> plt.Figure:
    """Figure 3: Human expertise has signal, but noise remains dominant."""
    table_exp = load_csv("T10_IndividualExpertAccuracy_Anonymized.csv")
    table_stu = load_csv("T21_StudentDescriptive.csv")
    table8 = load_csv("T04_AIvsHumanSummary.csv")
    table3 = load_csv("T02_FlagshipPerformance.csv")
    table_s24 = load_csv("T19_StudentReliabilityByPanelSize.csv")
    stats23 = load_json("S12_GatekeeperStats")

    expert_acc = table_exp["accuracy"].dropna().to_numpy(dtype=float) * 100.0
    student_acc = table_stu["accuracy"].dropna().to_numpy(dtype=float) * 100.0

    exp_majority_row = table8.loc[
        table8["Evaluator"].str.contains("Expert Majority Vote", regex=False)
    ].iloc[0]
    stu_majority_row = table8.loc[
        table8["Evaluator"].str.contains("Student Majority Vote", regex=False)
    ].iloc[0]
    exp_majority = float(exp_majority_row["Accuracy (%)"])
    stu_majority = float(stu_majority_row["Accuracy (%)"])

    fig = plt.figure(figsize=(FIG_WIDTH, 7.25))
    gs = gridspec.GridSpec(2, 2, hspace=0.60, wspace=0.35)

    # Panel a: Individual distributions (violin + raw points) + majority markers
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_title("a  Individual accuracy distributions", loc="left")

    rng = np.random.default_rng(42)
    pos_exp, pos_stu = 1.0, 2.0
    x_exp = np.ones_like(expert_acc) * pos_exp + rng.normal(0, 0.05, size=len(expert_acc))
    x_stu = np.ones_like(student_acc) * pos_stu + rng.normal(0, 0.05, size=len(student_acc))

    violin = ax_a.violinplot(
        [expert_acc, student_acc],
        positions=[pos_exp, pos_stu],
        widths=0.7,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body, color in zip(violin["bodies"], [PALETTE["expert"], PALETTE["junior"]]):
        body.set_facecolor(color)
        body.set_edgecolor("#333333")
        body.set_linewidth(0.6)
        body.set_alpha(0.35)
    if "cmedians" in violin:
        violin["cmedians"].set_color("#222222")
        violin["cmedians"].set_linewidth(1.0)

    ax_a.scatter(x_exp, expert_acc, s=10, alpha=0.35, color=PALETTE["expert"], linewidth=0, zorder=3)
    ax_a.scatter(x_stu, student_acc, s=8, alpha=0.20, color=PALETTE["junior"], linewidth=0, zorder=3)

    ax_a.errorbar(
        [pos_exp, pos_stu],
        [exp_majority, stu_majority],
        yerr=[
            [
                exp_majority - float(exp_majority_row["95% CI Lower"]),
                stu_majority - float(stu_majority_row["95% CI Lower"]),
            ],
            [
                float(exp_majority_row["95% CI Upper"]) - exp_majority,
                float(stu_majority_row["95% CI Upper"]) - stu_majority,
            ],
        ],
        fmt="D",
        markersize=4.3,
        color="#111111",
        ecolor="#222222",
        capsize=2,
        linewidth=0.9,
        zorder=5,
    )
    ax_a.text(0.02, 0.93, "Diamonds: majority vote (95% CI)", transform=ax_a.transAxes, fontsize=5.1)

    ax_a.axhline(25, linestyle="--", color=PALETTE["chance"], linewidth=0.8)
    ax_a.set_xticks([pos_exp, pos_stu])
    ax_a.set_xticklabels(
        [f"Experts\n(N={len(expert_acc)})", f"Juniors\n(N={len(student_acc)})"]
    )
    ax_a.set_ylabel("Individual accuracy (%)")
    ax_a.set_xlim(0.45, 2.55)
    ax_a.set_ylim(-5, 105)

    # Panel b: Accuracy + macro-F1 paired view for humans and model references.
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_title("b  Human summary vs frontier", loc="left")

    eval_metrics = stats23.get("evaluator_metrics", {})
    expert_individual_macro_fallback = _safe_float(eval_metrics.get("Expert Individual (Pooled)", {}).get("macro_f1")) * 100.0
    student_individual_macro_fallback = _safe_float(eval_metrics.get("Student Individual (Pooled)", {}).get("macro_f1")) * 100.0

    def _row_by_evaluator(evaluator: str) -> pd.Series:
        return table8.loc[table8["Evaluator"] == evaluator].iloc[0]

    row_expert_majority = _row_by_evaluator("Expert Majority Vote (excl. ties)")
    row_student_majority = _row_by_evaluator("Student Majority Vote (full, excl. ties)")
    row_expert_individual = _row_by_evaluator("Expert Individual Average")
    row_student_individual = _row_by_evaluator("Student Individual Average")
    row_frontier_avg = _row_by_evaluator("Flagship Average (11 models)")
    frontier_macro_f1_fallback = float(np.nanmean(table3["Macro F1"].to_numpy(dtype=float))) * 100.0
    expert_individual_n = int(_safe_float(row_expert_individual.get("N")))
    student_individual_n = int(_safe_float(row_student_individual.get("N")))

    expert_individual_macro = _to_percent_maybe(row_expert_individual.get("Macro F1"))
    if not np.isfinite(expert_individual_macro):
        expert_individual_macro = expert_individual_macro_fallback
    student_individual_macro = _to_percent_maybe(row_student_individual.get("Macro F1"))
    if not np.isfinite(student_individual_macro):
        student_individual_macro = student_individual_macro_fallback
    frontier_macro_f1 = _to_percent_maybe(row_frontier_avg.get("Macro F1"))
    if not np.isfinite(frontier_macro_f1):
        frontier_macro_f1 = frontier_macro_f1_fallback

    metric_rows = [
        {
            "label": "Expert majority (N=89)",
            "acc": _safe_float(row_expert_majority["Accuracy (%)"]),
            "f1": _safe_float(row_expert_majority["Macro F1"]) * 100.0,
            "f1_ci_lo": _to_percent_maybe(row_expert_majority.get("Macro F1 95% CI Lower")),
            "f1_ci_hi": _to_percent_maybe(row_expert_majority.get("Macro F1 95% CI Upper")),
            "ci_lo": _safe_float(row_expert_majority["95% CI Lower"]),
            "ci_hi": _safe_float(row_expert_majority["95% CI Upper"]),
            "color": PALETTE["expert"],
            "marker": "D",
        },
        {
            "label": "Junior majority (N=103)",
            "acc": _safe_float(row_student_majority["Accuracy (%)"]),
            "f1": _safe_float(row_student_majority["Macro F1"]) * 100.0,
            "f1_ci_lo": _to_percent_maybe(row_student_majority.get("Macro F1 95% CI Lower")),
            "f1_ci_hi": _to_percent_maybe(row_student_majority.get("Macro F1 95% CI Upper")),
            "ci_lo": _safe_float(row_student_majority["95% CI Lower"]),
            "ci_hi": _safe_float(row_student_majority["95% CI Upper"]),
            "color": PALETTE["junior"],
            "marker": "D",
        },
        {
            "label": f"Expert individual (N={expert_individual_n})",
            "acc": _safe_float(row_expert_individual["Accuracy (%)"]),
            "f1": expert_individual_macro,
            "f1_ci_lo": _to_percent_maybe(row_expert_individual.get("Macro F1 95% CI Lower")),
            "f1_ci_hi": _to_percent_maybe(row_expert_individual.get("Macro F1 95% CI Upper")),
            "ci_lo": _safe_float(row_expert_individual["95% CI Lower"]),
            "ci_hi": _safe_float(row_expert_individual["95% CI Upper"]),
            "color": PALETTE["expert"],
            "marker": "o",
        },
        {
            "label": f"Junior individual (N={student_individual_n})",
            "acc": _safe_float(row_student_individual["Accuracy (%)"]),
            "f1": student_individual_macro,
            "f1_ci_lo": _to_percent_maybe(row_student_individual.get("Macro F1 95% CI Lower")),
            "f1_ci_hi": _to_percent_maybe(row_student_individual.get("Macro F1 95% CI Upper")),
            "ci_lo": _safe_float(row_student_individual["95% CI Lower"]),
            "ci_hi": _safe_float(row_student_individual["95% CI Upper"]),
            "color": PALETTE["junior"],
            "marker": "o",
        },
        {
            "label": "Frontier avg (11)",
            "acc": _safe_float(row_frontier_avg["Accuracy (%)"]),
            "f1": frontier_macro_f1,
            "f1_ci_lo": _to_percent_maybe(row_frontier_avg.get("Macro F1 95% CI Lower")),
            "f1_ci_hi": _to_percent_maybe(row_frontier_avg.get("Macro F1 95% CI Upper")),
            "ci_lo": _safe_float(row_frontier_avg["95% CI Lower"]),
            "ci_hi": _safe_float(row_frontier_avg["95% CI Upper"]),
            "color": "#7A7A7A",
            "marker": "o",
        },
    ]

    y_base = np.arange(len(metric_rows), dtype=float)
    y_acc = y_base - 0.16
    y_f1 = y_base + 0.16

    for y_i, row in enumerate(metric_rows):
        acc = float(row["acc"])
        f1 = float(row["f1"])
        ci_lo = float(row["ci_lo"])
        ci_hi = float(row["ci_hi"])
        f1_ci_lo = float(row.get("f1_ci_lo", float("nan")))
        f1_ci_hi = float(row.get("f1_ci_hi", float("nan")))
        color = str(row["color"])

        if np.isfinite(ci_lo) and np.isfinite(ci_hi):
            ax_b.plot([ci_lo, ci_hi], [y_acc[y_i], y_acc[y_i]], color=color, linewidth=0.9, alpha=0.35, zorder=1)
        if np.isfinite(f1):
            if np.isfinite(f1_ci_lo) and np.isfinite(f1_ci_hi):
                ax_b.plot([f1_ci_lo, f1_ci_hi], [y_f1[y_i], y_f1[y_i]], color=color, linewidth=0.9, alpha=0.35, zorder=1)
            ax_b.scatter(
                [f1],
                [y_f1[y_i]],
                s=22,
                marker="s",
                facecolors="white",
                edgecolors=color,
                linewidths=0.9,
                zorder=3,
            )

        ax_b.scatter(
            [acc],
            [y_acc[y_i]],
            s=30,
            marker=str(row["marker"]),
            color=color,
            edgecolors="#222222",
            linewidths=0.35,
            zorder=4,
        )

    ax_b.axvline(25, color=PALETTE["chance"], linestyle="--", linewidth=0.8)
    ax_b.set_yticks(y_base)
    ax_b.set_yticklabels([r["label"] for r in metric_rows])
    ax_b.set_xlabel("Performance (%)")
    ax_b.set_xlim(18, 54)
    ax_b.set_ylim(-0.6, len(metric_rows) - 0.4)
    ax_b.invert_yaxis()
    ax_b.legend(
        handles=[
            plt.Line2D(
                [0],
                [0],
                marker="o",
                linestyle="-",
                color="#777777",
                markerfacecolor="#333333",
                markeredgecolor="#333333",
                markersize=4.4,
                linewidth=0.8,
                label="Accuracy (95% CI)",
            ),
            plt.Line2D(
                [0],
                [0],
                marker="s",
                linestyle="-",
                markerfacecolor="white",
                markeredgecolor="#333333",
                color="#333333",
                markersize=4.4,
                linewidth=0.8,
                label="Macro-F1 (95% CI)",
            ),
        ],
        loc="lower right",
        fontsize=4.9,
    )
    # Panel c: Predictor effects with CIs (rating-level records)
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_title("c  Predictor effect sizes (rating-level)", loc="left")

    expert_candidates = [
        ROOT / "data" / "human_ratings" / "reproducibility" / "expert_reproducibility.jsonl",
    ]
    student_candidates = [
        ROOT / "data" / "human_ratings" / "reproducibility" / "student_reproducibility_filtered.jsonl",
    ]

    # Keep Fig. 3c on rating-level records for direct comparability with the
    # reported confidence/familiarity analyses.
    expert_path = resolve_existing(expert_candidates, "expert reproducibility ratings")
    student_path = resolve_existing(student_candidates, "student reproducibility ratings")
    expert_loader = _load_rating_level_xy
    student_loader = _load_rating_level_xy

    exp_conf_x, exp_conf_y = expert_loader(expert_path, "confidence_int")
    exp_fam_x, exp_fam_y = expert_loader(expert_path, "familiarity_int")
    stu_conf_x, stu_conf_y = student_loader(student_path, "confidence_int")
    stu_fam_x, stu_fam_y = student_loader(student_path, "familiarity_int")

    exp_conf_r = _spearman_rho(exp_conf_x, exp_conf_y)
    exp_fam_r = _spearman_rho(exp_fam_x, exp_fam_y)
    stu_conf_r = _spearman_rho(stu_conf_x, stu_conf_y)
    stu_fam_r = _spearman_rho(stu_fam_x, stu_fam_y)

    exp_conf_p = _spearman_permutation_p(exp_conf_x, exp_conf_y, seed=42)
    exp_fam_p = _spearman_permutation_p(exp_fam_x, exp_fam_y, seed=43)
    stu_conf_p = _spearman_permutation_p(stu_conf_x, stu_conf_y, seed=44)
    stu_fam_p = _spearman_permutation_p(stu_fam_x, stu_fam_y, seed=45)

    exp_conf_ci = _bootstrap_spearman_ci(exp_conf_x, exp_conf_y, seed=42)
    exp_fam_ci = _bootstrap_spearman_ci(exp_fam_x, exp_fam_y, seed=43)
    stu_conf_ci = _bootstrap_spearman_ci(stu_conf_x, stu_conf_y, seed=44)
    stu_fam_ci = _bootstrap_spearman_ci(stu_fam_x, stu_fam_y, seed=45)

    rows = [
        {
            "label": "Expert confidence (rho)",
            "effect": float(exp_conf_r),
            "ci_lo": float(exp_conf_ci[0]),
            "ci_hi": float(exp_conf_ci[1]),
            "p": float(exp_conf_p),
            "n": int(len(exp_conf_x)),
        },
        {
            "label": "Expert familiarity (rho)",
            "effect": float(exp_fam_r),
            "ci_lo": float(exp_fam_ci[0]),
            "ci_hi": float(exp_fam_ci[1]),
            "p": float(exp_fam_p),
            "n": int(len(exp_fam_x)),
        },
        {
            "label": "Junior confidence (rho)",
            "effect": float(stu_conf_r),
            "ci_lo": float(stu_conf_ci[0]),
            "ci_hi": float(stu_conf_ci[1]),
            "p": float(stu_conf_p),
            "n": int(len(stu_conf_x)),
        },
        {
            "label": "Junior familiarity (rho)",
            "effect": float(stu_fam_r),
            "ci_lo": float(stu_fam_ci[0]),
            "ci_hi": float(stu_fam_ci[1]),
            "p": float(stu_fam_p),
            "n": int(len(stu_fam_x)),
        },
    ]

    y = np.arange(len(rows))
    ax_c.axvline(0, color=PALETTE["chance"], linestyle="--", linewidth=0.8)
    ax_c.axhline(1.5, color="#DDDDDD", linewidth=0.8)
    all_ci = np.array([[r["ci_lo"], r["ci_hi"]] for r in rows], dtype=float)
    if np.isfinite(all_ci).any():
        effect_x_min = float(np.nanmin(all_ci[:, 0]) - 0.03)
        effect_x_max = float(np.nanmax(all_ci[:, 1]) + 0.03)
    else:
        effect_x_min, effect_x_max = -0.5, 0.5
    # Keep a dedicated right-side lane for text labels to avoid overlap.
    label_x = effect_x_max + 0.04
    x_max = label_x + 0.10

    for i, row in enumerate(rows):
        color = PALETTE["expert"] if row["label"].startswith("Expert") else PALETTE["junior"]
        finite_triplet = np.isfinite([row["effect"], row["ci_lo"], row["ci_hi"]]).all()
        if not finite_triplet:
            ax_c.plot(0.0, i, marker="x", markersize=4.0, color="#888888")
            ax_c.text(
                label_x,
                i,
                "p=NA",
                ha="left",
                va="center",
                fontsize=5.1,
                bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.12, "alpha": 0.9},
            )
            continue
        xerr = [[row["effect"] - row["ci_lo"]], [row["ci_hi"] - row["effect"]]]
        ax_c.errorbar(
            row["effect"],
            i,
            xerr=xerr,
            fmt="o",
            color=color,
            ecolor=color,
            elinewidth=1.0,
            capsize=2.2,
            markersize=4.2,
            zorder=3,
        )
        p_txt = "p=NA" if not np.isfinite(row["p"]) else f"p={row['p']:.3g}"
        p_txt = f"{p_txt}, N={row['n']}"
        ax_c.text(
            label_x,
            i,
            p_txt,
            ha="left",
            va="center",
            fontsize=5.1,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.12, "alpha": 0.9},
        )

    ax_c.set_yticks(y)
    ax_c.set_yticklabels([r["label"] for r in rows])
    ax_c.set_xlabel("Effect size (Spearman rho; rating-level)")
    ax_c.set_xlim(effect_x_min, x_max)
    ax_c.set_ylim(len(rows) - 0.55, -0.45)
    ax_c.text(
        0.01,
        0.03,
        "Unit: individual ratings (not per-person averages).",
        transform=ax_c.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.0,
        color="#666666",
    )

    # Panel d: Matched-N curve
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_title("d  Junior panel-size scaling", loc="left")

    ax_d.plot(
        table_s24["k"],
        table_s24["accuracy_mean"] * 100,
        "o-",
        color=PALETTE["junior"],
        markersize=3,
        linewidth=1.1,
    )
    ax_d.fill_between(
        table_s24["k"],
        table_s24["accuracy_ci_lower"] * 100,
        table_s24["accuracy_ci_upper"] * 100,
        color=PALETTE["junior"],
        alpha=0.2,
    )
    ax_d.axhline(exp_majority, color=PALETTE["expert"], linestyle="--", linewidth=0.9)
    ax_d.text(
        table_s24["k"].max() - 0.2,
        exp_majority + 1.5,
        f"Expert majority = {exp_majority:.1f}%",
        ha="right",
        fontsize=5.3,
        color=PALETTE["expert"],
    )
    ax_d.axhline(25, color=PALETTE["chance"], linestyle="--", linewidth=0.7)
    ax_d.set_xlabel("Junior panel size (k)")
    ax_d.set_ylabel("Majority-vote accuracy (%)")
    ax_d.set_ylim(20, 55)

    fig.tight_layout()
    return fig


def make_figure4() -> plt.Figure:
    """Figure 4: SFT recovers tier discrimination across architectures."""
    table4 = load_csv("T03_SFTPerformance.csv")
    table_base = load_csv("extended_data/ExtendedDataTable1_BaseModelControls.csv")
    table8 = load_csv("T04_AIvsHumanSummary.csv")
    stats23 = load_json("S12_GatekeeperStats")
    best_flagship_eval = (
        table8.loc[table8["Evaluator"].str.startswith("Best Flagship ("), "Evaluator"].iloc[0]
    )
    best_flagship_name = str(best_flagship_eval)[len("Best Flagship (") : -1]
    flagship_rows = [
        k for k in stats23["evaluator_metrics"].keys() if str(k).startswith("Flagship: ")
    ]
    best_flagship_metric_key = max(
        flagship_rows,
        key=lambda k: float(stats23["evaluator_metrics"][k]["overall_accuracy"]),
    )

    fig = plt.figure(figsize=(FIG_WIDTH, 7.1))
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.58, width_ratios=[1.05, 1.20])

    # Panel a: paired architecture view with accuracy above macro-F1.
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_title("a  Base-to-SFT gains by architecture", loc="left")

    panel_a_specs = [
        ("GPT-4.1-nano", "gpt-4.1-nano", "ckpt-step-304"),
        ("Qwen3-4B", "qwen3-4b", "ckppt-228"),
        ("Qwen3-30B", "qwen3-30b-a3b", "ckppt-380"),
        ("GPT-4.1", "gpt-4.1", "CYqJRxId"),
    ]
    arch = [item[0] for item in panel_a_specs]
    base_map: Dict[str, float] = {}
    base_f1_map: Dict[str, float] = {}
    for _, r in table_base.iterrows():
        model = str(r["Model"])
        if model.startswith("Base (") and model.endswith(")"):
            base_map[model[len("Base (") : -1]] = float(r["Accuracy (%)"])
            base_f1_map[model[len("Base (") : -1]] = float(r["Macro F1"]) * 100.0

    sft_map: Dict[str, float] = {}
    sft_f1_map: Dict[str, float] = {}
    for _, r in table4.iterrows():
        model = str(r["Model"])
        if model.startswith("SFT (") and model.endswith(")"):
            sft_map[model[len("SFT (") : -1]] = float(r["Accuracy"]) * 100.0
            sft_f1_map[model[len("SFT (") : -1]] = float(r["Macro F1"]) * 100.0

    missing_base = [name for name in arch if name not in base_map]
    missing_sft = [name for name in arch if name not in sft_map]
    if missing_base or missing_sft:
        raise ValueError(
            f"Figure 4 panel-a inputs missing values (base={missing_base}, sft={missing_sft})."
        )

    panel_a_rows = []
    for idx, (name, base_key, sft_key) in enumerate(panel_a_specs):
        base_metrics = _summarize_logp_model(CHAT_FILE, base_key, seed=110 + idx)
        sft_metrics = _summarize_logp_model(SFT_FILE, sft_key, seed=210 + idx)
        if abs(base_metrics["acc"] * 100.0 - base_map[name]) > 0.11:
            raise ValueError(f"Base accuracy mismatch for {name}: table={base_map[name]:.1f}, raw={base_metrics['acc'] * 100.0:.2f}")
        if abs(sft_metrics["acc"] * 100.0 - sft_map[name]) > 0.11:
            raise ValueError(f"SFT accuracy mismatch for {name}: table={sft_map[name]:.1f}, raw={sft_metrics['acc'] * 100.0:.2f}")
        if abs(base_metrics["macro_f1"] * 100.0 - base_f1_map[name]) > 0.11:
            raise ValueError(
                f"Base macro-F1 mismatch for {name}: table={base_f1_map[name]:.1f}, raw={base_metrics['macro_f1'] * 100.0:.2f}"
            )
        if abs(sft_metrics["macro_f1"] * 100.0 - sft_f1_map[name]) > 0.11:
            raise ValueError(
                f"SFT macro-F1 mismatch for {name}: table={sft_f1_map[name]:.1f}, raw={sft_metrics['macro_f1'] * 100.0:.2f}"
            )
        panel_a_rows.append({"arch": name, "base": base_metrics, "sft": sft_metrics})

    y_center = np.arange(len(panel_a_rows), dtype=float) * 1.75
    y_acc = y_center - 0.22
    y_f1 = y_center + 0.22
    gain_x = 72.0
    marker_size = 5.0

    for i, row in enumerate(panel_a_rows):
        base_acc = float(row["base"]["acc"]) * 100.0
        sft_acc = float(row["sft"]["acc"]) * 100.0
        base_f1 = float(row["base"]["macro_f1"]) * 100.0
        sft_f1 = float(row["sft"]["macro_f1"]) * 100.0

        for y_val, base_val, sft_val in [
            (y_acc[i], base_acc, sft_acc),
            (y_f1[i], base_f1, sft_f1),
        ]:
            ax_a.plot([base_val, sft_val], [y_val, y_val], color="#B5B5B5", linewidth=1.15, zorder=1)

        ax_a.errorbar(
            base_acc,
            y_acc[i],
            xerr=[[base_acc - float(row["base"]["acc_lo"]) * 100.0], [float(row["base"]["acc_hi"]) * 100.0 - base_acc]],
            fmt="o",
            ms=marker_size,
            color=PALETTE["frontier"],
            ecolor=PALETTE["frontier"],
            elinewidth=0.85,
            capsize=2.0,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )
        ax_a.errorbar(
            sft_acc,
            y_acc[i],
            xerr=[[sft_acc - float(row["sft"]["acc_lo"]) * 100.0], [float(row["sft"]["acc_hi"]) * 100.0 - sft_acc]],
            fmt="o",
            ms=marker_size + 0.2,
            color=PALETTE["sft"],
            ecolor=PALETTE["sft"],
            elinewidth=0.9,
            capsize=2.0,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=4,
        )
        ax_a.errorbar(
            base_f1,
            y_f1[i],
            xerr=[[base_f1 - float(row["base"]["macro_f1_lo"]) * 100.0], [float(row["base"]["macro_f1_hi"]) * 100.0 - base_f1]],
            fmt="s",
            ms=marker_size - 0.4,
            color=PALETTE["frontier"],
            ecolor=PALETTE["frontier"],
            elinewidth=0.85,
            capsize=2.0,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=3,
        )
        ax_a.errorbar(
            sft_f1,
            y_f1[i],
            xerr=[[sft_f1 - float(row["sft"]["macro_f1_lo"]) * 100.0], [float(row["sft"]["macro_f1_hi"]) * 100.0 - sft_f1]],
            fmt="s",
            ms=marker_size - 0.2,
            color=PALETTE["sft"],
            ecolor=PALETTE["sft"],
            elinewidth=0.9,
            capsize=2.0,
            markeredgecolor="white",
            markeredgewidth=0.45,
            zorder=4,
        )

        ax_a.text(gain_x, y_acc[i], f"+{sft_acc - base_acc:.1f} pp", va="center", ha="left", fontsize=5.0, color="#333333")
        ax_a.text(gain_x, y_f1[i], f"+{sft_f1 - base_f1:.1f} pp", va="center", ha="left", fontsize=5.0, color="#555555")

    ax_a.set_yticks(y_center)
    ax_a.set_yticklabels([row["arch"] for row in panel_a_rows])
    ax_a.invert_yaxis()
    ax_a.set_ylabel("Architecture")
    ax_a.set_xlabel("Score (%)")
    ax_a.set_xlim(0, 80)
    ax_a.set_ylim(y_center[-1] + 0.85, -1.15)
    ax_a.set_xticks(np.arange(0, 81, 10))
    ax_a.axvline(25, color=PALETTE["chance"], linestyle="--", linewidth=0.8)
    ax_a.grid(axis="x", color="#ECECEC", linewidth=0.55)

    color_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=4.8, color=PALETTE["frontier"], label="Base"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=4.8, color=PALETTE["sft"], label="SFT"),
    ]
    metric_handles = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=4.6, color=PALETTE["neutral"], markerfacecolor="white", label="Accuracy"),
        plt.Line2D([0], [0], marker="s", linestyle="", markersize=4.2, color=PALETTE["neutral"], markerfacecolor="white", label="Macro-F1"),
    ]
    legend_color = ax_a.legend(
        handles=color_handles,
        loc="upper left",
        bbox_to_anchor=(0.03, 0.985),
        ncol=2,
        fontsize=4.5,
        borderaxespad=0.0,
        handletextpad=0.25,
        columnspacing=0.8,
        labelspacing=0.2,
    )
    ax_a.add_artist(legend_color)
    legend_a = ax_a.legend(
        handles=metric_handles,
        loc="upper left",
        bbox_to_anchor=(0.43, 0.985),
        ncol=2,
        fontsize=4.5,
        borderaxespad=0.0,
        handletextpad=0.25,
        columnspacing=0.8,
        labelspacing=0.2,
    )
    place_reference_label_safely(
        ax_a,
        "Chance (25%)",
        preferred_anchor="lower left",
        blockers=[legend_color, legend_a, getattr(ax_a, "_left_title", None)],
        color=PALETTE["chance"],
        fontsize=5.0,
    )
    ax_a.text(
        0.98,
        0.03,
        "95% bootstrap CI",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.9,
        color="#666666",
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "#D8D8D8"},
    )

    # Panel b: Expanded comparison with requested evaluators + CI
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_title("b  Evaluator comparison", loc="left")

    table8_map = {
        "SFT 2-Model Ensemble": ("SFT 2-model ens.", "sft"),
        best_flagship_eval: (f"Best flagship ({best_flagship_name})", "frontier"),
        "Flagship Average (11 models)": ("Frontier average (11 models)", "frontier"),
        "Expert Individual Average": ("Expert individual avg", "expert"),
        "Expert Majority Vote (excl. ties)": ("Expert majority vote", "expert"),
        "Student Individual Average": ("Junior individual avg", "junior"),
        "Student Majority Vote (full, excl. ties)": ("Junior majority vote", "junior"),
    }
    sft_single_map = {
        "SFT (GPT-4.1-nano)": "SFT GPT-4.1-nano",
        "SFT (Qwen3-4B)": "SFT Qwen3-4B",
        "SFT (Qwen3-30B)": "SFT Qwen3-30B",
        "SFT (GPT-4.1)": "SFT GPT-4.1",
    }

    rows = []
    for _, r in table8.iterrows():
        ev = str(r["Evaluator"])
        if ev in table8_map:
            label, cat = table8_map[ev]
            rows.append(
                {
                    "evaluator": label,
                    "accuracy": float(r["Accuracy (%)"]),
                    "ci_lo": float(r["95% CI Lower"]),
                    "ci_hi": float(r["95% CI Upper"]),
                    "cat": cat,
                }
            )

    # Add all four SFT single models (CI by binomial approximation).
    for _, r in table4.iterrows():
        model_name = str(r["Model"])
        if model_name in sft_single_map:
            acc = float(r["Accuracy"]) * 100.0
            n_valid = float(r["N Valid"])
            ci_half = float(binomial_ci(np.array([acc / 100.0]), np.array([n_valid]))[0] * 100.0)
            rows.append(
                {
                    "evaluator": sft_single_map[model_name],
                    "accuracy": acc,
                    "ci_lo": acc - ci_half,
                    "ci_hi": acc + ci_half,
                    "cat": "sft",
                }
            )

    comp = pd.DataFrame(rows).sort_values("accuracy", ascending=True).reset_index(drop=True)

    y = np.arange(len(comp))
    for i, r in comp.iterrows():
        color = PALETTE[r["cat"]]
        ax_b.plot([r["ci_lo"], r["ci_hi"]], [i, i], color=color, linewidth=1.1)
        ax_b.scatter(r["accuracy"], i, color=color, s=24, zorder=3, edgecolor="white", linewidth=0.4)
        label_x = min(r["ci_hi"] + 0.7, 69.0)
        ax_b.text(
            label_x,
            i,
            f'{r["accuracy"]:.1f}',
            va="center",
            ha="left",
            fontsize=4.9,
            bbox={"facecolor": "white", "edgecolor": "none", "pad": 0.08},
        )

    ax_b.set_yticks(y)
    ax_b.set_yticklabels(comp["evaluator"].tolist())
    ax_b.tick_params(axis="y", labelsize=5.1, pad=1)
    ax_b.set_xlabel("Accuracy (%)")
    ax_b.axvline(25, color=PALETTE["chance"], linestyle="--", linewidth=0.7)
    ax_b.set_xlim(20, 70)
    ax_b.set_ylim(-0.8, len(comp) - 0.2)

    handles_b = [
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5, color=PALETTE["sft"], label="SFT"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5, color=PALETTE["expert"], label="Experts"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5, color=PALETTE["junior"], label="Juniors"),
        plt.Line2D([0], [0], marker="o", linestyle="", markersize=5, color=PALETTE["frontier"], label="Frontier"),
    ]
    ax_b.legend(handles=handles_b, loc="lower right", fontsize=5.1, ncol=2)

    # Panel c: Per-tier recall heatmap
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_title("c  Tier recall heatmap", loc="left")

    eval_order = [
        "SFT 2-Model Ensemble",
        best_flagship_eval,
        "Flagship Average (11 models)",
        "Expert Majority Vote (excl. ties)",
        "Student Majority Vote (full, excl. ties)",
    ]

    frontier_avg_recall_pct = {
        tier: float(
            np.mean(
                [
                    float(stats23["evaluator_metrics"][k]["per_class_metrics"][tier]["recall"]) * 100.0
                    for k in flagship_rows
                ]
            )
        )
        for tier in TIER_ORDER
    }

    hm = []
    for ev in eval_order:
        if ev == "Flagship Average (11 models)":
            hm.append(
                [
                    frontier_avg_recall_pct["exceptional"],
                    frontier_avg_recall_pct["strong"],
                    frontier_avg_recall_pct["fair"],
                    frontier_avg_recall_pct["limited"],
                ]
            )
        else:
            row = table8.loc[table8["Evaluator"] == ev].iloc[0]
            hm.append(
                [
                    float(row["Acc Exceptional (%)"]),
                    float(row["Acc Strong (%)"]),
                    float(row["Acc Fair (%)"]),
                    float(row["Acc Limited (%)"]),
                ]
            )
    hm_arr = np.array(hm, dtype=float)

    im = ax_c.imshow(hm_arr, cmap="YlOrRd", vmin=0, vmax=80, aspect="auto")
    ax_c.set_xticks(range(4))
    ax_c.set_xticklabels(TIER_LABELS)
    ax_c.set_yticks(range(len(eval_order)))
    ax_c.set_yticklabels([
        "SFT ensemble",
        "Best frontier",
        "Frontier average",
        "Expert majority",
        "Junior majority",
    ])

    for r in range(hm_arr.shape[0]):
        for c in range(hm_arr.shape[1]):
            val = hm_arr[r, c]
            ax_c.text(c, r, f"{val:.1f}", ha="center", va="center", fontsize=5.1)

    cbar = fig.colorbar(im, ax=ax_c, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(labelsize=5.2)
    cbar.set_label("Recall (%)", fontsize=5.8)

    # Panel d: Boundary-tier F1 (exceptional + limited)
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_title("d  Boundary-tier F1", loc="left")

    frontier_avg_exc = float(
        np.mean(
            [
                float(stats23["evaluator_metrics"][k]["per_class_metrics"]["exceptional"]["f1"])
                for k in flagship_rows
            ]
        )
    )
    frontier_avg_lim = float(
        np.mean(
            [
                float(stats23["evaluator_metrics"][k]["per_class_metrics"]["limited"]["f1"])
                for k in flagship_rows
            ]
        )
    )

    metric_names = [
        ("SFT: 2-Model Ensemble", "SFT", PALETTE["sft"]),
        (best_flagship_metric_key, "Best frontier", PALETTE["frontier"]),
        ("__frontier_avg__", "Frontier average", PALETTE["neutral"]),
        ("Expert Majority (Unfiltered)", "Expert majority", PALETTE["expert"]),
        ("Student Majority (Filtered)", "Junior majority", PALETTE["junior"]),
    ]

    x = np.arange(len(metric_names))
    width = 0.34

    f1_exc = []
    f1_lim = []
    colors = []
    labels = []
    for key, label, color in metric_names:
        if key == "__frontier_avg__":
            f1_exc.append(frontier_avg_exc)
            f1_lim.append(frontier_avg_lim)
        else:
            per = stats23["evaluator_metrics"][key]["per_class_metrics"]
            f1_exc.append(float(per["exceptional"]["f1"]))
            f1_lim.append(float(per["limited"]["f1"]))
        colors.append(color)
        labels.append(label)

    for i in range(len(metric_names)):
        ax_d.bar(i - width / 2, f1_exc[i], width=width, color=colors[i], alpha=0.9)
        ax_d.bar(i + width / 2, f1_lim[i], width=width, color=colors[i], alpha=0.45)

    ax_d.set_xticks(x)
    ax_d.set_xticklabels(labels, rotation=20, ha="right")
    ax_d.set_ylabel("F1 score")
    ax_d.set_ylim(0, 0.75)

    handles = [
        plt.Line2D([0], [0], marker="s", linestyle="", color="#666666", markerfacecolor="#666666", alpha=0.9, label="Exceptional tier"),
        plt.Line2D([0], [0], marker="s", linestyle="", color="#666666", markerfacecolor="#666666", alpha=0.45, label="Limited tier"),
    ]
    ax_d.legend(handles=handles, loc="upper right", fontsize=5.3)

    fig.tight_layout()
    return fig


def make_figure5() -> plt.Figure:
    """Figure 5: Self-knowledge, selective triage, and pairwise ranking generalization."""
    table_s12 = load_csv("T12_ConfidenceComparison.csv")

    trained_path = SFT_FILE
    chat_path = REPORTS_ROOT / "data" / "predictions" / "chat_predictions.jsonl"
    expert_path = REPORTS_ROOT / "data" / "human_ratings" / "reproducibility" / "expert_reproducibility.jsonl"
    student_filtered_path = (
        REPORTS_ROOT / "data" / "human_ratings" / "reproducibility" / "student_reproducibility_filtered.jsonl"
    )

    style = {
        "SFT 2-Model Ensemble": {
            "color": PALETTE["sft"],
            "label": "SFT ensemble",
            "marker": "o",
            "linestyle": "-",
        },
        "GPT-5.2 (chat)": {
            "color": PALETTE["frontier"],
            "label": "GPT-5.2 chat",
            "marker": "o",
            "linestyle": "-",
        },
        "Kimi K2 Chat": {
            "color": "#7570B3",
            "label": "Kimi K2 chat",
            "marker": "o",
            "linestyle": "-",
        },
        "DeepSeek Chat": {
            "color": "#66A61E",
            "label": "DeepSeek chat",
            "marker": "o",
            "linestyle": "-",
        },
        "Human Expert": {
            "color": "#1F78B4",
            "label": "Human expert (1-5)",
            "marker": "s",
            "linestyle": "--",
        },
        "Human Student": {
            "color": "#33A02C",
            "label": "Human junior (1-5)",
            "marker": "s",
            "linestyle": "--",
        },
    }

    panel_b_order = [
        "SFT 2-Model Ensemble",
        "GPT-5.2 (chat)",
        "Kimi K2 Chat",
        "DeepSeek Chat",
    ]

    ai_confidence_records = {
        "SFT 2-Model Ensemble": _collect_logp_confidence_records(trained_path, "best_2_model_combo"),
        "GPT-5.2 (chat)": _collect_logp_confidence_records(chat_path, "gpt-5.2"),
        "Kimi K2 Chat": _collect_logp_confidence_records(chat_path, "kimi-k2-0905-preview"),
        "DeepSeek Chat": _collect_logp_confidence_records(chat_path, "deepseek-chat"),
    }
    human_records = {
        "Human Expert": _collect_human_rating_confidence_records(expert_path),
        "Human Student": _collect_human_rating_confidence_records(student_filtered_path),
    }
    pairwise_bundle = _load_figure5_pairwise_bundle()
    pairwise_stats = pairwise_bundle["stats"]  # type: ignore[assignment]
    pairwise_order = pairwise_bundle["order"]  # type: ignore[assignment]
    pairwise_metrics = pairwise_bundle["metrics"]  # type: ignore[assignment]
    pairwise_comparisons = pairwise_bundle["comparisons"]  # type: ignore[assignment]

    fig = plt.figure(figsize=(FIG_WIDTH, 7.0))
    gs = gridspec.GridSpec(2, 2, hspace=0.60, wspace=0.35)

    # Panel a: Confidence discrimination
    ax_a = fig.add_subplot(gs[0, 0])
    ax_a.set_title("a  Confidence separation", loc="left")

    keep_order = [
        "SFT 2-Model Ensemble",
        "GPT-5.2 Chat",
        "Human Expert",
        "Human Student",
    ]
    rows = []
    for name in keep_order:
        row = table_s12.loc[table_s12["model"] == name].iloc[0]
        corr = row["conf_correct"]
        wrong = row["conf_wrong"]
        if pd.isna(corr) or pd.isna(wrong):
            corr = _normalize_confidence_1_to_5(float(row["conf_correct_raw"]))
            wrong = _normalize_confidence_1_to_5(float(row["conf_wrong_raw"]))
        rows.append(
            {
                "name": name,
                "correct": float(corr),
                "wrong": float(wrong),
                "p": float(row["mann_whitney_p"]),
            }
        )

    x = np.arange(len(rows))
    width = 0.34
    colors = [PALETTE["sft"], PALETTE["frontier"], PALETTE["expert"], PALETTE["junior"]]
    for i, row in enumerate(rows):
        ax_a.bar(i - width / 2, row["correct"], width=width, color=colors[i], alpha=0.9)
        ax_a.bar(i + width / 2, row["wrong"], width=width, color=colors[i], alpha=0.35)
        ax_a.text(i, max(row["correct"], row["wrong"]) + 0.03, f"p={row['p']:.3g}", ha="center", fontsize=5)

    ax_a.set_xticks(x)
    ax_a.set_xticklabels(["SFT\nEnsemble", "GPT-5.2\nChat", "Experts", "Juniors"])
    ax_a.set_ylabel("Mean confidence (0-1)")
    ax_a.set_ylim(0.0, 1.05)
    ax_a.text(
        0.98,
        0.92,
        "Human conf: (x-1)/4 -> 0-1\np: one-sided Mann-Whitney U\n(correct > incorrect, rating-level)",
        transform=ax_a.transAxes,
        ha="right",
        va="center",
        fontsize=4.1,
        color="#555555",
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.85},
    )

    # Panel b: Selective prediction
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.set_title("b  Selective prediction", loc="left")

    sft_curve = pd.DataFrame(columns=["coverage", "accuracy"])
    for key in panel_b_order:
        curve = _selective_curve_from_confidence(ai_confidence_records.get(key, []))
        if len(curve) == 0:
            continue
        st = style[key]
        ax_b.plot(
            curve["coverage"],
            curve["accuracy"],
            color=st["color"],
            linestyle=st["linestyle"],
            linewidth=1.0,
            label=st["label"],
        )
        step = max(1, len(curve) // 10)
        ax_b.scatter(
            curve["coverage"].iloc[::step],
            curve["accuracy"].iloc[::step],
            color=st["color"],
            marker=st["marker"],
            s=9,
            alpha=0.8,
        )
        if key == "SFT 2-Model Ensemble":
            sft_curve = curve

    for human_name in ["Human Expert", "Human Student"]:
        curve = _human_selective_curve(human_records[human_name])
        if len(curve) == 0:
            continue
        st = style[human_name]
        ax_b.plot(
            curve["coverage"],
            curve["accuracy"],
            color=st["color"],
            linestyle=st["linestyle"],
            marker=st["marker"],
            markersize=3.4,
            linewidth=1.0,
            label=st["label"],
        )

    if len(sft_curve) > 0:
        for item in _selective_curve_callouts(sft_curve):
            cov_val = float(item["coverage"])
            acc_val = float(item["accuracy"])
            ax_b.scatter([cov_val], [acc_val], color="#111111", s=14, zorder=4)
            ax_b.annotate(
                f"{_format_pct_label(cov_val)} -> {acc_val:.1f}%",
                xy=(cov_val, acc_val),
                xytext=item["xytext"],
                textcoords="offset points",
                ha=str(item["ha"]),
                va=str(item["va"]),
                fontsize=4.8,
                bbox={"boxstyle": "round,pad=0.12", "facecolor": "white", "edgecolor": "none", "alpha": 0.88},
                annotation_clip=False,
                zorder=5,
            )

    ax_b.axhline(25, color=PALETTE["chance"], linestyle="--", linewidth=0.7)
    ax_b.set_xlabel("Coverage (% highest-confidence predictions)")
    ax_b.set_ylabel("Accuracy (%)")
    ax_b.set_xlim(0, 100)
    ax_b.set_ylim(20, 100)
    legend_b = ax_b.legend(
        loc="upper right",
        bbox_to_anchor=(0.985, 0.90),
        fontsize=4.4,
        ncol=1,
        columnspacing=0.8,
        labelspacing=0.25,
        handlelength=1.8,
        borderaxespad=0.25,
        frameon=True,
    )
    legend_b.get_frame().set_facecolor("white")
    legend_b.get_frame().set_alpha(0.92)
    legend_b.get_frame().set_edgecolor("#C8C8C8")

    # Panel c: Overall pairwise accuracy with raw paired tests
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.set_title("c  Pairwise ranking generalizes beyond tier labels", loc="left")

    pairwise_colors = [PAIRWISE_COLORS[model] for model in pairwise_order]
    x = np.arange(len(pairwise_order))
    acc_vals = []
    ci_vals = []
    for model in pairwise_order:
        p = float(pairwise_metrics[model]["weighted_accuracy"])
        n = float(pairwise_metrics[model]["total"])
        acc_vals.append(p * 100.0)
        ci_vals.append(float(binomial_ci(np.array([p]), np.array([n]))[0]) * 100.0)

    bars = ax_c.bar(x, acc_vals, yerr=ci_vals, capsize=3, color=pairwise_colors, edgecolor="white")
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([PAIRWISE_SHORT_LABELS[model] for model in pairwise_order], rotation=18, ha="right")
    ax_c.set_ylabel("Weighted accuracy (%)")
    ax_c.set_ylim(65, 90)
    for i, bar in enumerate(bars):
        ax_c.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + ci_vals[i] + 0.25,
            f"{acc_vals[i]:.1f}",
            ha="center",
            fontsize=5.8,
        )
    for i, model in enumerate(pairwise_order[1:], start=1):
        comp = pairwise_comparisons.get(model)
        if comp is None:
            continue
        ax_c.text(
            x[i],
            66.0,
            f"p={_format_p_value(float(comp['p_value_raw']))}",
            ha="center",
            va="bottom",
            fontsize=5.1,
            color="white",
        )
    ax_c.text(
        0.98,
        0.97,
        "95% binomial CI\nraw exact McNemar p vs SFT",
        transform=ax_c.transAxes,
        ha="right",
        va="top",
        fontsize=4.6,
        bbox={"boxstyle": "round,pad=0.18", "facecolor": "white", "edgecolor": "#CCCCCC", "alpha": 0.9},
    )

    # Panel d: Hard-boundary pairwise summary
    ax_d = fig.add_subplot(gs[1, 1])
    ax_d.set_title("d  Pairwise gains peak at the hardest boundaries", loc="left")

    pair_type_accuracy = pairwise_stats["pair_type_accuracy_pct"]  # type: ignore[index]
    x_pair_type = "strong_exceptional"
    y_pair_type = "fair_strong"
    label_offsets = {
        "SFT GPT-4.1": (0.8, 0.2),
        "Gemini 3.1 Pro": (0.6, 0.3),
        "GPT-5.2 High": (0.6, 0.1),
        "GPT-4.1 Baseline": (0.6, -0.4),
    }
    for model in pairwise_order:
        x_val = float(pair_type_accuracy[model][x_pair_type]["accuracy_pct"])
        y_val = float(pair_type_accuracy[model][y_pair_type]["accuracy_pct"])
        color = PAIRWISE_COLORS[model]
        ax_d.scatter(x_val, y_val, s=64, color=color, edgecolor="white", linewidth=0.6, zorder=3)
        dx, dy = label_offsets.get(model, (0.6, 0.2))
        ax_d.text(x_val + dx, y_val + dy, PAIRWISE_SHORT_LABELS[model], fontsize=5.4, color=color, va="center")

    ax_d.set_xlim(48, 66.5)
    ax_d.set_ylim(64, 86.5)
    ax_d.set_xlabel("Strong vs Exceptional accuracy (%)")
    ax_d.set_ylabel("Fair vs Strong accuracy (%)")
    ax_d.grid(linestyle=":", alpha=0.4, zorder=0)
    ax_d.text(
        0.98,
        0.02,
        "50 pairs per boundary\nhardest adjacent and top-end contrasts",
        transform=ax_d.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.8,
        color="#444444",
        bbox={"boxstyle": "round,pad=0.14", "facecolor": "white", "edgecolor": "none", "alpha": 0.82},
    )

    fig.tight_layout()
    return fig


def main() -> None:
    set_style()

    figures = [
        ("Figure2", make_figure2),
        ("Figure3", make_figure3),
        ("Figure4", make_figure4),
        ("Figure5", make_figure5),
    ]

    print(f"Output directory: {OUTPUT_DIR}")
    for name, fn in figures:
        fig = fn()
        png = save_figure(fig, name)
        plt.close(fig)
        print(f"- {png}")

    print("Done.")


if __name__ == "__main__":
    main()
