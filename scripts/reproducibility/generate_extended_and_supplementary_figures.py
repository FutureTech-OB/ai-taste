#!/usr/bin/env python3
"""Generate Extended Data and Supplementary figures/tables.

Design goals:
- High information density with paneled layouts.
- Avoid duplicating main-figure headline panels.
- Trace pairwise figure directly to original pairwise experiment files.
"""

from __future__ import annotations

import json
import itertools
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import matplotlib.transforms as mtransforms
import numpy as np
import pandas as pd
from matplotlib.patches import Patch

from figure_style_policy import apply_title_policy


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "README.md").exists() and (candidate / "scripts").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError("Could not locate package root (expected README.md + scripts/ + data/).")


ROOT = find_project_root()
PROJECT_ROOT = ROOT
OUT_EXT = ROOT / "reproduced" / "figures" / "extended_data"
OUT_SUPP = ROOT / "reproduced" / "figures" / "supplementary"
OUT_NOTES = ROOT / "reproduced" / "notes"

STATS_DIR = ROOT / "data" / "statistics"
TABLES_DIR = ROOT / "data" / "tables"
PAIRWISE_RAW_STATS_PATH = STATS_DIR / "S14_ED2PairwiseRawPValues.json"

PROMPT_SIMPLE = PROJECT_ROOT / "data" / "predictions" / "prompt_variants" / "simple_prompt_predictions.jsonl"
PROMPT_JOURNAL = PROJECT_ROOT / "data" / "predictions" / "prompt_variants" / "journal_prompt_predictions.jsonl"
PROMPT_EXPERT = PROJECT_ROOT / "data" / "predictions" / "prompt_variants" / "expert_prompt_predictions.jsonl"
PROMPT_EXPERT_GEMINI31_FALLBACK = PROJECT_ROOT / "data" / "predictions" / "gemini_3_1_pro_standalone.jsonl"
THINKING_PATH = PROJECT_ROOT / "data" / "predictions" / "frontier_10models_8runs.jsonl"
TRAINED_PATH = PROJECT_ROOT / "data" / "predictions" / "sft_predictions.jsonl"
CHAT_PATH = PROJECT_ROOT / "data" / "predictions" / "chat_predictions.jsonl"
RL_PATH = PROJECT_ROOT / "data" / "predictions" / "rl_predictions.jsonl"
TEMPORAL_OLD_PATH = PROJECT_ROOT / "data" / "predictions" / "sft_temporal_old_predictions.jsonl"
CORE_RQ_SHORT_TRANSFER_PATH = PROJECT_ROOT / "data" / "predictions" / "core_rq_short_transfer_predictions.jsonl"
CORE_RQ_SHORT_TRANSFER_STATS_PATH = STATS_DIR / "S15_CoreRQShortTransferStats.json"

PAIRWISE_ROOT = PROJECT_ROOT / "data" / "pairwise"
PAIRWISE_FRONTIER_GEMINI = PAIRWISE_ROOT / "frontier_gemini3_1_pro"

LABELS = ["exceptional", "strong", "fair", "limited"]
LABELS_SHORT = ["Exc", "Str", "Fair", "Ltd"]
LABELS_TITLE = ["Exceptional", "Strong", "Fair", "Limited"]
PAIR_TYPE_ORDER = [
    "strong_exceptional",
    "fair_strong",
    "fair_exceptional",
    "limited_fair",
    "limited_exceptional",
    "limited_strong",
]
PAIR_TYPE_LABELS = {
    "strong_exceptional": "Strong vs Exceptional",
    "fair_strong": "Fair vs Strong",
    "fair_exceptional": "Fair vs Exceptional",
    "limited_fair": "Limited vs Fair",
    "limited_exceptional": "Limited vs Exceptional",
    "limited_strong": "Limited vs Strong",
}

COLORS = {
    "sft": "#0072B2",
    "gemini": "#009E73",
    "grok": "#009E73",
    "gpt52": "#56B4E9",
    "baseline": "#D55E00",
    "expert_prompt": "#0072B2",
    "simple_prompt": "#E69F00",
    "journal_prompt": "#CC79A7",
    "neutral": "#666666",
    "chance": "#9A9A9A",
    "expert": "#4DBBD5",
    "student": "#00A087",
}


def _set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.titlesize": 8.2,
            "axes.labelsize": 7.0,
            "xtick.labelsize": 6.0,
            "ytick.labelsize": 6.0,
            "axes.linewidth": 0.7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def _save(fig: plt.Figure, out_base: Path) -> Tuple[Path, Path]:
    apply_title_policy(fig, family="edsi")
    out_base.parent.mkdir(parents=True, exist_ok=True)
    png = out_base.with_suffix(".png")
    pdf = out_base.with_suffix(".pdf")
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def _binom_ci95(p: float, n: int) -> float:
    if n <= 0:
        return 0.0
    return 1.96 * math.sqrt(max(p * (1.0 - p), 0.0) / n)


def _format_p_value(p: float) -> str:
    if p < 0.001:
        return f"{p:.2e}"
    return f"{p:.4f}"


def _bootstrap_mean_ci(values: np.ndarray, n_boot: int = 4000, seed: int = 42) -> Tuple[float, float]:
    arr = np.asarray(values, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return float("nan"), float("nan")
    if arr.size == 1:
        v = float(arr[0])
        return v, v
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, arr.size, size=(n_boot, arr.size))
    means = arr[idx].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(lo), float(hi)


def _normalize_pred_label(raw: str) -> str | None:
    x = str(raw).strip().lower()
    mapping = {
        "top": "exceptional",
        "top-": "strong",
        "good": "fair",
        "fair": "fair",
        "exceptional": "exceptional",
        "strong": "strong",
        "limited": "limited",
    }
    return mapping.get(x)


def _extract_prediction_from_model_output(model_output: Dict[str, object]) -> str | None:
    """Robustly recover a canonical tier label from model output payload."""
    pred = _normalize_pred_label(str(model_output.get("prediction", "")))
    if pred in LABELS:
        return pred

    # Some files only store per-label log-probabilities without a final label.
    logp = model_output.get("logp")
    if isinstance(logp, dict) and logp:
        best_label: str | None = None
        best_score = float("-inf")
        for raw_label, raw_score in logp.items():
            label = _normalize_pred_label(str(raw_label))
            if label not in LABELS:
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if score > best_score:
                best_score = score
                best_label = label
        return best_label
    return None


def _compute_confusion_from_records(records: List[Tuple[str, str]]) -> np.ndarray:
    cm = np.zeros((4, 4), dtype=int)
    idx = {k: i for i, k in enumerate(LABELS)}
    for gt, pred in records:
        if gt in idx and pred in idx:
            cm[idx[gt], idx[pred]] += 1
    return cm


def _collect_prompt_records(path: Path, model_key: str) -> Dict[str, object]:
    records: List[Tuple[str, str]] = []
    n_total = 0
    n_tie = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gt = _normalize_pred_label(row.get("rank", ""))
            if gt not in LABELS:
                continue
            model_out = row.get("val_outcome", {}).get("rq_with_context", {}).get(model_key, {})
            if not isinstance(model_out, dict) or not model_out:
                continue

            n_total += 1
            pred: str | None = None
            is_tie = bool(model_out.get("vote_is_tie", False))
            tie_detected = is_tie
            if is_tie:
                n_tie += 1
            else:
                pred = _normalize_pred_label(model_out.get("prediction_majority") or model_out.get("prediction"))

            # Use strict majority only; unresolved/tied rows remain incorrect.
            if pred not in LABELS and not is_tie:
                vote_counts = model_out.get("vote_counts", {})
                if isinstance(vote_counts, dict) and vote_counts:
                    top = max(vote_counts.values())
                    winners = [k for k, v in vote_counts.items() if v == top]
                    if len(winners) == 1:
                        pred = _normalize_pred_label(winners[0])
                    else:
                        tie_detected = True
                        n_tie += 1
                        pred = None

            if pred not in LABELS and not tie_detected:
                pred = _extract_prediction_from_model_output(model_out)
            if pred in LABELS:
                records.append((gt, pred))
    return {
        "records": records,
        "n_total": n_total,
        "n_tie": n_tie,
    }


def _summarize_prompt_records(bundle: Dict[str, object]) -> Dict[str, object]:
    records: List[Tuple[str, str]] = bundle.get("records", [])  # type: ignore[assignment]
    cm = _compute_confusion_from_records(records)
    n_total = int(bundle.get("n_total", int(cm.sum())))
    n_resolved = int(cm.sum())
    n_correct = int(np.trace(cm))
    n_unresolved = max(n_total - n_resolved, 0)
    acc = float(n_correct / n_total) if n_total > 0 else float("nan")
    tp = np.diag(cm).astype(float)
    row_sum = cm.sum(axis=1).astype(float)
    col_sum = cm.sum(axis=0).astype(float)
    recall = np.divide(tp, row_sum, out=np.zeros_like(tp), where=row_sum > 0)
    precision = np.divide(tp, col_sum, out=np.zeros_like(tp), where=col_sum > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    macro_f1 = float(np.mean(f1))
    per_tier = [float(x) for x in recall]
    pred_dist = [float(cm[:, j].sum() / n_total) if n_total > 0 else float("nan") for j in range(4)]
    return {
        "records": records,
        "cm": cm,
        "n": n_total,
        "n_resolved": n_resolved,
        "n_unresolved": n_unresolved,
        "n_correct": n_correct,
        "tie_n": int(bundle.get("n_tie", 0)),
        "unresolved_pct": float(n_unresolved / n_total * 100.0) if n_total > 0 else float("nan"),
        "acc": acc,
        "macro_f1": macro_f1,
        "ci": _binom_ci95(acc, n_total),
        "per_tier_recall": per_tier,
        "pred_dist": pred_dist,
    }


def load_prompt_sensitivity_data() -> Dict[str, Dict[str, object]]:
    model_candidates = [
        ("google/gemini-3.1-pro-preview", "Gemini 3.1 Pro"),
        ("google/gemini-2.5-pro", "Gemini 2.5 Pro"),
    ]
    expert_sources_by_model = {
        # Prefer canonical expert prompt file; fall back to legacy archive only if unavailable.
        "google/gemini-3.1-pro-preview": [PROMPT_EXPERT, PROMPT_EXPERT_GEMINI31_FALLBACK],
        "google/gemini-2.5-pro": [PROMPT_EXPERT],
    }

    selected_model_key = ""
    selected_model_label = ""
    selected_expert_source = ""
    expert_bundle: Dict[str, object] = {}

    for model_key, model_label in model_candidates:
        for source in expert_sources_by_model.get(model_key, [PROMPT_EXPERT]):
            if not source.exists():
                continue
            bundle = _collect_prompt_records(source, model_key)
            if int(bundle.get("n_total", 0)) > 0:
                selected_model_key = model_key
                selected_model_label = model_label
                selected_expert_source = str(source)
                expert_bundle = bundle
                break
        if expert_bundle:
            break

    if not expert_bundle:
        raise ValueError(
            "No prompt-sensitivity records found for Gemini standalone diagnostic in available expert sources."
        )

    prompt_data: Dict[str, Dict[str, object]] = {"Expert": _summarize_prompt_records(expert_bundle)}

    for prompt_name, path in [("Simplified", PROMPT_SIMPLE), ("Journal-anchored", PROMPT_JOURNAL)]:
        bundle = _collect_prompt_records(path, selected_model_key)
        prompt_data[prompt_name] = _summarize_prompt_records(bundle)

    prompt_data["_meta"] = {
        "model_key": selected_model_key,
        "model_label": selected_model_label,
        "expert_source": selected_expert_source,
    }
    return prompt_data


def _load_sft_ensemble_diagnostics() -> Dict[str, object]:
    stats03 = json.loads((STATS_DIR / "S02_SFTStats.json").read_text(encoding="utf-8"))
    obj = stats03.get("model_performance", {}).get("2-Model Ensemble", {})
    cm = np.asarray(obj.get("confusion_matrix", np.zeros((4, 4), dtype=int)), dtype=float)
    n = int(cm.sum())
    acc = float(obj.get("accuracy", float(np.trace(cm) / n if n > 0 else float("nan")))) if n > 0 else float("nan")
    macro_f1 = float(obj.get("macro_f1", float("nan")))
    return {"cm": cm, "n": n, "acc": acc, "macro_f1": macro_f1}


def _load_gemini25_prompt_diagnostics() -> Dict[str, Dict[str, object]]:
    out: Dict[str, Dict[str, object]] = {}
    for prompt_name, path in [("Expert", PROMPT_EXPERT), ("Simplified", PROMPT_SIMPLE), ("Journal-anchored", PROMPT_JOURNAL)]:
        bundle = _collect_prompt_records(path, "google/gemini-2.5-pro")
        out[prompt_name] = _summarize_prompt_records(bundle)
    return out


def load_prompt_sensitivity_all_models() -> Dict[str, object]:
    prompt_files = {
        "Simple": PROMPT_SIMPLE,
        "Journal": PROMPT_JOURNAL,
        "Expert": PROMPT_EXPERT,
    }
    model_name_map = {
        "anthropic/claude-opus-4.6": "Claude Opus 4.6",
        "deepseek/deepseek-v3.2": "DeepSeek V3.2",
        "deepseek/deepseek-v3.2-speciale": "DeepSeek V3.2",
        "google/gemini-2.5-pro": "Gemini 2.5 Pro",
        "google/gemini-3-pro-preview": "Gemini 3 Pro",
        "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
        "minimax/minimax-m2.5": "MiniMax M2.5",
        "moonshotai/kimi-k2.5": "Kimi K2.5",
        "openai/gpt-5.2": "GPT-5.2",
        "openai/gpt-5.2-high": "GPT-5.2 High",
        "qwen/qwen3.5-plus-02-15": "Qwen 3.5 Plus",
        "x-ai/grok-4.1-fast": "Grok 4.1 Fast",
        "z-ai/glm-5": "GLM-5",
        "ft:gpt-4.1-2025-04-14:personal:rq1106-2:CYqJRxId": "SFT GPT-4.1",
        "ft:gpt-4.1-nano-2025-04-14:personal::CdeuJ22y": "SFT GPT-4.1-nano",
    }

    model_alias = {
        "openai/gpt-5.2": "openai/gpt-5.2-high",
        "deepseek/deepseek-v3.2": "deepseek/deepseek-v3.2-speciale",
    }

    rows_by_prompt: Dict[str, List[Tuple[str, Dict[str, object]]]] = {}
    all_models: set[str] = set()
    for pname, p in prompt_files.items():
        if not p.exists():
            raise FileNotFoundError(f"Prompt-sensitivity file not found: {p}")
        rows: List[Tuple[str, Dict[str, object]]] = []
        with p.open("r", encoding="utf-8") as f:
            for line in f:
                rec = json.loads(line)
                gt = _normalize_pred_label(rec.get("rank", ""))
                if gt not in LABELS:
                    continue
                out = rec.get("val_outcome", {}).get("rq_with_context", {})
                if not isinstance(out, dict):
                    continue
                out_norm: Dict[str, object] = {}
                for mk, mobj in out.items():
                    canon = model_alias.get(mk, mk)
                    if canon not in out_norm:
                        out_norm[canon] = mobj
                rows.append((gt, out_norm))
                all_models.update(out_norm.keys())
        rows_by_prompt[pname] = rows

    def _argmax_logp(logp: Dict[str, object]) -> str | None:
        best_label: str | None = None
        best_score = float("-inf")
        for raw_label, raw_score in logp.items():
            label = _normalize_pred_label(raw_label)
            if label not in LABELS:
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                continue
            if score > best_score:
                best_score = score
                best_label = label
        return best_label

    def _single_pass_prediction(mobj: Dict[str, object]) -> Tuple[str | None, str, bool]:
        """Single-pass protocol.
        - Thinking/frontier entries: use run-1 from vote_predictions; if missing, fallback to first valid later run.
        - SFT entries: use argmax(logp) because they are deterministic single-pass classifiers.
        """
        vote_predictions = mobj.get("vote_predictions")
        if isinstance(vote_predictions, list):
            if len(vote_predictions) > 0:
                first = _normalize_pred_label(vote_predictions[0])
                if first in LABELS:
                    return first, "run1", False
            for raw in vote_predictions[1:]:
                cand = _normalize_pred_label(raw)
                if cand in LABELS:
                    return cand, "run1_fallback", True
            return None, "run1_missing", False

        logp = mobj.get("logp")
        if isinstance(logp, dict) and logp:
            pred = _argmax_logp(logp)
            if pred in LABELS:
                return pred, "logp", False

        pred = _normalize_pred_label(mobj.get("prediction", ""))
        if pred in LABELS:
            return pred, "prediction", False
        return None, "missing", False

    def _metrics(y_true: List[str], y_pred: List[str]) -> Tuple[float, float]:
        if not y_true:
            return float("nan"), float("nan")
        cm = _compute_confusion_from_records(list(zip(y_true, y_pred)))
        n = int(cm.sum())
        acc = float(np.trace(cm) / n) if n > 0 else float("nan")
        tp = np.diag(cm).astype(float)
        row_sum = cm.sum(axis=1).astype(float)
        col_sum = cm.sum(axis=0).astype(float)
        recall = np.divide(tp, row_sum, out=np.zeros_like(tp), where=row_sum > 0)
        precision = np.divide(tp, col_sum, out=np.zeros_like(tp), where=col_sum > 0)
        f1 = np.divide(
            2.0 * precision * recall,
            precision + recall,
            out=np.zeros_like(tp),
            where=(precision + recall) > 0,
        )
        return acc, float(np.mean(f1))

    per_model: Dict[str, Dict[str, Dict[str, object]]] = {}
    for model in sorted(all_models):
        per_model[model] = {}
        for pname, rows in rows_by_prompt.items():
            rows_with_model = 0
            first_missing_or_null = 0
            fallback_used = 0
            source_run1 = 0
            source_logp = 0
            source_prediction = 0
            y_true: List[str] = []
            y_pred: List[str] = []
            for gt, out in rows:
                mobj = out.get(model)
                if not isinstance(mobj, dict):
                    continue
                rows_with_model += 1
                pred, source, used_fallback = _single_pass_prediction(mobj)
                if source == "run1_missing":
                    first_missing_or_null += 1
                if used_fallback:
                    fallback_used += 1
                if source == "run1":
                    source_run1 += 1
                elif source == "logp":
                    source_logp += 1
                elif source == "prediction":
                    source_prediction += 1
                if pred not in LABELS:
                    continue
                y_true.append(gt)
                y_pred.append(pred)
            if rows_with_model == 0:
                continue
            cm = _compute_confusion_from_records(list(zip(y_true, y_pred))) if y_true else np.zeros((4, 4), dtype=int)
            acc, macro_f1 = _metrics(y_true, y_pred)
            per_model[model][pname] = {
                "rows_with_model": float(rows_with_model),
                "n": float(len(y_true)),
                "usable_rate_pct": float(len(y_true) / rows_with_model * 100.0),
                "first_missing_or_null": float(first_missing_or_null),
                "fallback_used": float(fallback_used),
                "fallback_rate_pct": float(fallback_used / rows_with_model * 100.0),
                "source_run1": float(source_run1),
                "source_logp": float(source_logp),
                "source_prediction": float(source_prediction),
                "acc_pct": float(acc * 100.0) if np.isfinite(acc) else float("nan"),
                "macro_f1_pct": float(macro_f1 * 100.0) if np.isfinite(macro_f1) else float("nan"),
                # Keep confusion for downstream paneling in ED1 (Simple/Journal prompt error structure).
                "cm": cm.tolist(),
            }

    # Sort display by Expert-prompt accuracy if available.
    sorted_models = sorted(
        per_model.keys(),
        key=lambda m: per_model[m].get("Expert", {}).get("acc_pct", -1.0),
        reverse=True,
    )
    return {
        "prompts": ["Simple", "Journal", "Expert"],
        "models": sorted_models,
        "name_map": model_name_map,
        "metrics": per_model,
    }


def make_ed1_prompt_sensitivity_v2(
    out_base: Path | None = None,
    suptitle: str | None = None,
) -> Tuple[Path, Path]:
    _set_style()
    d = load_prompt_sensitivity_data()
    order = ["Expert", "Simplified", "Journal-anchored"]
    colors = [COLORS["expert_prompt"], COLORS["simple_prompt"], COLORS["journal_prompt"]]

    fig = plt.figure(figsize=(10.8, 8.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.26)

    # a) Accuracy with CI + Macro-F1
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(order))
    w = 0.36
    acc = np.array([d[k]["acc"] * 100 for k in order])  # type: ignore[index]
    ci = np.array([d[k]["ci"] * 100 for k in order])  # type: ignore[index]
    macro_f1 = np.array([d[k]["macro_f1"] * 100 for k in order])  # type: ignore[index]
    bars_acc = ax.bar(
        x - w / 2,
        acc,
        width=w,
        yerr=ci,
        capsize=3,
        color=colors,
        edgecolor="white",
        label="Accuracy",
    )
    bars_f1 = ax.bar(
        x + w / 2,
        macro_f1,
        width=w,
        color=colors,
        alpha=0.35,
        hatch="//",
        edgecolor=colors,
        linewidth=0.8,
        label="Macro-F1",
    )
    ax.axhline(25, color=COLORS["chance"], linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Score (%)")
    model_label = str(d.get("_meta", {}).get("model_label", "Gemini"))  # type: ignore[index]
    ax.set_title(f"a  Prompt sensitivity ({model_label})", loc="left")
    ax.legend(fontsize=6, loc="upper right")
    upper = np.nanmax(np.concatenate([acc + ci, macro_f1]))
    if not np.isfinite(upper):
        upper = 60.0
    ax.set_ylim(0, float(upper) + 8.0)
    for i, b in enumerate(bars_acc):
        n = d[order[i]]["n"]  # type: ignore[index]
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + ci[i] + 0.8,
            f"{acc[i]:.1f}% (n={n})",
            ha="center",
            fontsize=6,
        )
    for i, b in enumerate(bars_f1):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.8,
            f"F1 {macro_f1[i]:.1f}",
            ha="center",
            fontsize=5.8,
            color="#222222",
        )

    # b) Per-tier recall heatmap
    ax = fig.add_subplot(gs[0, 1])
    mat = np.array([d[k]["per_tier_recall"] for k in order], dtype=float) * 100.0  # type: ignore[index]
    # Use a slightly expanded upper bound so the 100% cell is less visually saturated.
    im = ax.imshow(mat, cmap="YlGnBu", vmin=0, vmax=120, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(LABELS_TITLE, rotation=35, ha="right")
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels(order)
    ax.set_title("b  Tier recall by prompt", loc="left")
    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):
            val = float(mat[i, j])
            if val >= 70.0:
                ax.text(
                    j,
                    i,
                    f"{val:.1f}",
                    ha="center",
                    va="center",
                    fontsize=6,
                    color="white",
                    bbox=dict(boxstyle="round,pad=0.10", facecolor="#1A2A57", edgecolor="none", alpha=0.35),
                )
            else:
                t = ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6, color="#111111")
                t.set_path_effects([pe.withStroke(linewidth=1.0, foreground="#ffffff")])
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cb.set_label("Recall (%)")

    # c) Prediction distribution shift
    ax = fig.add_subplot(gs[1, 0])
    bottoms = np.zeros(len(order))
    tier_colors = ["#5B8FF9", "#61DDAA", "#F6BD16", "#E8684A"]
    for i, tier in enumerate(LABELS):
        vals = np.array([d[k]["pred_dist"][i] * 100 for k in order], dtype=float)  # type: ignore[index]
        ax.bar(order, vals, bottom=bottoms, color=tier_colors[i], edgecolor="white", label=LABELS_TITLE[i])
        bottoms += vals
    ax.set_ylim(0, 100)
    ax.set_ylabel("Predicted share (%)")
    ax.set_title("c  Predicted-tier distribution by prompt", loc="left")
    ax.legend(ncol=2, fontsize=6, loc="upper left")

    # d) Delta vs expert by tier
    ax = fig.add_subplot(gs[1, 1])
    expert = np.array(d["Expert"]["per_tier_recall"], dtype=float) * 100.0  # type: ignore[index]
    simp = np.array(d["Simplified"]["per_tier_recall"], dtype=float) * 100.0  # type: ignore[index]
    jour = np.array(d["Journal-anchored"]["per_tier_recall"], dtype=float) * 100.0  # type: ignore[index]
    w = 0.34
    x = np.arange(4)
    ax.bar(x - w / 2, simp - expert, width=w, color=COLORS["simple_prompt"], label="Simplified - Expert")
    ax.bar(x + w / 2, jour - expert, width=w, color=COLORS["journal_prompt"], label="Journal - Expert")
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(LABELS_TITLE, rotation=25, ha="right")
    ax.set_ylabel("Recall delta (pp)")
    ax.set_title("d  Tier-wise delta vs expert prompt", loc="left")
    ax.legend(fontsize=6, loc="lower left")

    if out_base is None:
        out_base = OUT_EXT / "ed_fig1"
    return _save(fig, out_base)


def make_sf7_gemini_prompt_diagnostic_v2(
    out_base: Path | None = None,
    suptitle: str | None = None,
) -> Tuple[Path, Path]:
    _set_style()
    gem31 = load_prompt_sensitivity_data()
    gem25 = _load_gemini25_prompt_diagnostics()
    sft = _load_sft_ensemble_diagnostics()

    order = ["Expert", "Simplified", "Journal-anchored"]
    gem25_color = "#56B4E9"
    sft_acc = float(sft["acc"]) * 100.0
    sft_n = int(sft["n"])
    sft_ci = _binom_ci95(sft_acc / 100.0, sft_n) * 100.0 if sft_n > 0 else 0.0
    sft_lo = max(0.0, sft_acc - sft_ci)
    sft_hi = sft_acc + sft_ci
    panel_a_ymin = 10.0
    panel_a_ymax = 80.0

    fig = plt.figure(figsize=(12.6, 8.3))
    gs = gridspec.GridSpec(2, 4, figure=fig, height_ratios=[0.95, 1.0], hspace=0.36, wspace=0.34)

    # a) Prompt-variant accuracy with Gemini 2.5 completeness reference and SFT benchmark line.
    ax = fig.add_subplot(gs[0, :])
    x = np.arange(len(order))
    w = 0.30
    acc31 = np.array([float(gem31[k]["acc"]) * 100.0 for k in order], dtype=float)  # type: ignore[index]
    ci31 = np.array([float(gem31[k]["ci"]) * 100.0 for k in order], dtype=float)  # type: ignore[index]
    acc25 = np.array([float(gem25[k]["acc"]) * 100.0 for k in order], dtype=float)
    ci25 = np.array([float(gem25[k]["ci"]) * 100.0 for k in order], dtype=float)

    bars31 = ax.bar(
        x - w / 2,
        acc31,
        width=w,
        yerr=ci31,
        capsize=3,
        color=COLORS["gemini"],
        edgecolor="white",
        label="Gemini 3.1 Pro",
    )
    bars25 = ax.bar(
        x + w / 2,
        acc25,
        width=w,
        yerr=ci25,
        capsize=3,
        color=gem25_color,
        edgecolor="white",
        label="Gemini 2.5 Pro",
    )
    ax.axhspan(sft_lo, min(panel_a_ymax, sft_hi), color=COLORS["sft"], alpha=0.12, label="SFT 95% CI")
    ax.axhline(25, color=COLORS["chance"], linestyle="--", linewidth=0.9, label="Chance (25%)")
    ax.axhline(sft_acc, color=COLORS["sft"], linestyle="-", linewidth=1.1, label="SFT 2-model ensemble")
    ax.axhline(sft_lo, color=COLORS["sft"], linestyle=":", linewidth=0.85)
    if sft_hi <= panel_a_ymax:
        ax.axhline(sft_hi, color=COLORS["sft"], linestyle=":", linewidth=0.85)
    ax.text(
        0.99,
        0.96,
        f"SFT {sft_acc:.1f}% (95% CI {sft_lo:.1f}-{sft_hi:.1f})",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        color=COLORS["sft"],
    )
    gem25_upper = float(np.nanmax(acc25 + ci25))
    ci_gap_gem25 = sft_lo - gem25_upper
    if ci_gap_gem25 >= 0:
        ci_note = f"SFT lower CI {sft_lo:.1f}% > Gemini 2.5 best upper CI {gem25_upper:.1f}%"
    else:
        ci_note = f"SFT lower CI {sft_lo:.1f}% vs Gemini 2.5 best upper CI {gem25_upper:.1f}%"
    ax.text(
        0.99,
        0.03,
        ci_note,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.18", facecolor="white", edgecolor="#DDDDDD"),
    )
    ax.set_xticks(x)
    ax.set_xticklabels(order)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("a  Prompt-variant performance: Gemini 3.1 vs Gemini 2.5", loc="left")
    ax.legend(fontsize=6, ncol=5, loc="upper left")
    ax.set_ylim(panel_a_ymin, panel_a_ymax)
    for i, b in enumerate(bars31):
        y = max(panel_a_ymin + 0.8, b.get_height() - 1.0)
        ax.text(
            b.get_x() + b.get_width() / 2,
            y,
            f"{acc31[i]:.1f}",
            ha="center",
            va="top",
            fontsize=6.0,
            color="white",
        )
    for i, b in enumerate(bars25):
        y = max(panel_a_ymin + 0.8, b.get_height() - 1.0)
        ax.text(
            b.get_x() + b.get_width() / 2,
            y,
            f"{acc25[i]:.1f}",
            ha="center",
            va="top",
            fontsize=6.0,
            color="white",
        )

    # b-e) Confusion matrices for Gemini prompts and SFT reference.
    conf_specs = [
        (
            "b  Gemini 3.1 | Expert",
            np.asarray(gem31["Expert"]["cm"], dtype=float),  # type: ignore[index]
            acc31[0],
            int(gem31["Expert"]["n"]),  # type: ignore[index]
            int(gem31["Expert"]["n_resolved"]),  # type: ignore[index]
            int(gem31["Expert"]["n_unresolved"]),  # type: ignore[index]
        ),
        (
            "c  Gemini 3.1 | Simplified",
            np.asarray(gem31["Simplified"]["cm"], dtype=float),  # type: ignore[index]
            acc31[1],
            int(gem31["Simplified"]["n"]),  # type: ignore[index]
            int(gem31["Simplified"]["n_resolved"]),  # type: ignore[index]
            int(gem31["Simplified"]["n_unresolved"]),  # type: ignore[index]
        ),
        (
            "d  Gemini 3.1 | Journal-anchored",
            np.asarray(gem31["Journal-anchored"]["cm"], dtype=float),  # type: ignore[index]
            acc31[2],
            int(gem31["Journal-anchored"]["n"]),  # type: ignore[index]
            int(gem31["Journal-anchored"]["n_resolved"]),  # type: ignore[index]
            int(gem31["Journal-anchored"]["n_unresolved"]),  # type: ignore[index]
        ),
        ("e  SFT 2-model ensemble", np.asarray(sft["cm"], dtype=float), sft_acc, int(sft["n"]), int(sft["n"]), 0),
    ]

    cm_norm_stack = np.stack([_row_norm(cm) * 100.0 for _, cm, _, _, _, _ in conf_specs], axis=0)
    vmax = float(max(np.nanmax(cm_norm_stack), 55.0))

    im = None
    conf_axes = [fig.add_subplot(gs[1, i]) for i in range(4)]
    for ax, (title, cm, acc, n_total, n_resolved, n_unresolved) in zip(conf_axes, conf_specs):
        cmn = _row_norm(cm) * 100.0
        im = ax.imshow(cmn, cmap="YlOrBr", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(LABELS_SHORT, fontsize=6.0, rotation=25, ha="right")
        ax.set_yticklabels(LABELS_SHORT, fontsize=6.0)
        ax.set_xlabel("Predicted tier")
        ax.set_title(f"{title}\nacc={acc:.1f}% (n={n_total})", loc="left")
        for i in range(4):
            for j in range(4):
                val = float(cmn[i, j])
                txt_col = "white" if val >= 0.58 * vmax else "#111111"
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=5.7, color=txt_col)
        if n_unresolved > 0:
            ax.text(
                0.98,
                0.03,
                f"resolved {n_resolved}\nunresolved {n_unresolved}",
                transform=ax.transAxes,
                ha="right",
                va="bottom",
                fontsize=5.4,
                color="#444444",
                bbox=dict(boxstyle="round,pad=0.17", facecolor="white", edgecolor="#DDDDDD"),
            )
    conf_axes[0].set_ylabel("True tier")
    for ax in conf_axes[1:]:
        ax.set_ylabel("")
        ax.set_yticklabels([])
    if im is not None:
        cb = fig.colorbar(im, ax=conf_axes, fraction=0.020, pad=0.02)
        cb.set_label("Row-normalized share (%)")

    expert_src = Path(str(gem31.get("_meta", {}).get("expert_source", ""))).name  # type: ignore[index]
    fig.text(
        0.5,
        0.01,
        f"Gemini 3.1 expert arm source: {expert_src}; simplified/journal sources: 120_simple.jsonl and 120_journal.jsonl. "
        "Panels b-d normalize confusion on resolved predictions only; unresolved rows are counted incorrect in panel-a accuracy.",
        ha="center",
        va="bottom",
        fontsize=6.1,
        color="#555555",
    )

    if out_base is None:
        out_base = OUT_SUPP / "legacy_gemini_prompt_diagnostic"
    return _save(fig, out_base)


def make_sf7_prompt_sensitivity_all_models_v2(
    out_base: Path | None = None,
    suptitle: str | None = None,
) -> Tuple[Path, Path]:
    _set_style()
    d = load_prompt_sensitivity_all_models()
    prompts: List[str] = d["prompts"]  # type: ignore[assignment]
    models: List[str] = d["models"]  # type: ignore[assignment]
    name_map: Dict[str, str] = d["name_map"]  # type: ignore[assignment]
    metrics: Dict[str, Dict[str, Dict[str, object]]] = d["metrics"]  # type: ignore[assignment]

    prompt_colors = {
        "Simple": COLORS["simple_prompt"],
        "Journal": COLORS["journal_prompt"],
        "Expert": COLORS["expert_prompt"],
    }

    def _exclude_from_ed1(model_key: str) -> bool:
        k = model_key.lower()
        return "google/gemini-3-pro" in k

    complete_models = [
        m
        for m in models
        if (not _exclude_from_ed1(m))
        and all((p in metrics.get(m, {}) and metrics[m][p]["n"] > 0) for p in prompts)
    ]
    if not complete_models:
        raise ValueError("No models with complete (Simple/Journal/Expert) coverage after exclusion rules.")

    display_models = [name_map.get(m, m) for m in complete_models]
    fig = plt.figure(figsize=(13.8, 8.0))
    gs = gridspec.GridSpec(2, 6, figure=fig, hspace=0.42, wspace=0.35)

    # a) Accuracy by prompt (models with complete 3-prompt coverage; includes Gemini 3.1)
    ax = fig.add_subplot(gs[0, 0:3])
    x = np.arange(len(complete_models))
    w = 0.25
    for i, p in enumerate(prompts):
        vals = np.array([metrics[m][p]["acc_pct"] for m in complete_models], dtype=float)
        ax.bar(
            x + (i - 1) * w,
            vals,
            width=w,
            color=prompt_colors[p],
            edgecolor="white",
            label=p,
        )
    ax.axhline(25, color=COLORS["chance"], linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(display_models, rotation=24, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title(
        "a  Accuracy by prompt (models with complete Simple/Journal/Expert coverage)",
        loc="left",
    )
    ax.legend(fontsize=6, ncol=3, loc="upper left")
    ax.set_ylim(0, 50)

    # b) Macro-F1 by prompt (same complete-coverage model set).
    ax = fig.add_subplot(gs[0, 3:6])
    x = np.arange(len(complete_models))
    for i, p in enumerate(prompts):
        vals = np.array([metrics[m][p]["macro_f1_pct"] for m in complete_models], dtype=float)
        ax.bar(
            x + (i - 1) * w,
            vals,
            width=w,
            color=prompt_colors[p],
            edgecolor="white",
            label=p,
        )
    ax.axhline(25, color=COLORS["chance"], linestyle="--", linewidth=0.9)
    ax.set_xticks(x)
    ax.set_xticklabels(display_models, rotation=24, ha="right")
    ax.set_ylabel("Macro-F1 (%)")
    ax.set_title("b  Macro-F1 by prompt (same model set)", loc="left")
    ax.set_ylim(0, 50)

    def _pooled_prompt_confusion(prompt_name: str) -> Tuple[np.ndarray, int, int]:
        cm_sum = np.zeros((4, 4), dtype=float)
        model_n = 0
        pred_n = 0
        for m in complete_models:
            pobj = metrics.get(m, {}).get(prompt_name, {})
            cm_raw = pobj.get("cm") if isinstance(pobj, dict) else None
            if cm_raw is None:
                continue
            cm = np.asarray(cm_raw, dtype=float)
            if cm.shape != (4, 4):
                continue
            cm_sum += cm
            model_n += 1
            pred_n += int(cm.sum())
        return cm_sum, model_n, pred_n

    cm_expert, n_expert_models, n_expert_pred = _pooled_prompt_confusion("Expert")
    cm_simple, n_simple_models, n_simple_pred = _pooled_prompt_confusion("Simple")
    cm_journal, n_journal_models, n_journal_pred = _pooled_prompt_confusion("Journal")
    cm_expert_norm = _row_norm(cm_expert) * 100.0
    cm_simple_norm = _row_norm(cm_simple) * 100.0
    cm_journal_norm = _row_norm(cm_journal) * 100.0
    vmax = float(max(np.nanmax(cm_expert_norm), np.nanmax(cm_simple_norm), np.nanmax(cm_journal_norm), 40.0))

    # c) Pooled confusion matrix under Expert prompt (baseline for comparison).
    ax_c = fig.add_subplot(gs[1, 0:2])
    im_c = ax_c.imshow(cm_expert_norm, cmap="YlOrBr", vmin=0, vmax=vmax, aspect="auto")
    ax_c.set_xticks(range(4))
    ax_c.set_yticks(range(4))
    ax_c.set_xticklabels(LABELS_SHORT, fontsize=6.2, rotation=25, ha="right")
    ax_c.set_yticklabels(LABELS_SHORT, fontsize=6.2)
    ax_c.set_xlabel("Predicted tier")
    ax_c.set_ylabel("True tier")
    ax_c.set_title("c  Pooled confusion (Expert prompt baseline)", loc="left")
    for i in range(4):
        for j in range(4):
            val = float(cm_expert_norm[i, j])
            txt_col = "white" if val >= 0.58 * vmax else "#111111"
            ax_c.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6.0, color=txt_col)
    acc_expert = float(np.trace(cm_expert) / max(cm_expert.sum(), 1.0) * 100.0)
    ax_c.text(
        0.99,
        0.03,
        f"{n_expert_models} models, n={n_expert_pred}\nacc={acc_expert:.1f}%",
        transform=ax_c.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#DDDDDD"),
    )

    # d) Pooled confusion matrix under Simple prompt.
    ax_d = fig.add_subplot(gs[1, 2:4])
    im_d = ax_d.imshow(cm_simple_norm, cmap="YlOrBr", vmin=0, vmax=vmax, aspect="auto")
    ax_d.set_xticks(range(4))
    ax_d.set_yticks(range(4))
    ax_d.set_xticklabels(LABELS_SHORT, fontsize=6.2, rotation=25, ha="right")
    ax_d.set_yticklabels(LABELS_SHORT, fontsize=6.2)
    ax_d.set_xlabel("Predicted tier")
    ax_d.set_ylabel("True tier")
    ax_d.set_title("d  Pooled confusion (Simple prompt)", loc="left")
    for i in range(4):
        for j in range(4):
            val = float(cm_simple_norm[i, j])
            txt_col = "white" if val >= 0.58 * vmax else "#111111"
            ax_d.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6.0, color=txt_col)
    acc_simple = float(np.trace(cm_simple) / max(cm_simple.sum(), 1.0) * 100.0)
    ax_d.text(
        0.99,
        0.03,
        f"{n_simple_models} models, n={n_simple_pred}\nacc={acc_simple:.1f}%",
        transform=ax_d.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#DDDDDD"),
    )

    # e) Pooled confusion matrix under Journal prompt.
    ax_e = fig.add_subplot(gs[1, 4:6])
    im_e = ax_e.imshow(cm_journal_norm, cmap="YlOrBr", vmin=0, vmax=vmax, aspect="auto")
    ax_e.set_xticks(range(4))
    ax_e.set_yticks(range(4))
    ax_e.set_xticklabels(LABELS_SHORT, fontsize=6.2, rotation=25, ha="right")
    ax_e.set_yticklabels(LABELS_SHORT, fontsize=6.2)
    ax_e.set_xlabel("Predicted tier")
    ax_e.set_ylabel("True tier")
    ax_e.set_title("e  Pooled confusion (Journal prompt)", loc="left")
    for i in range(4):
        for j in range(4):
            val = float(cm_journal_norm[i, j])
            txt_col = "white" if val >= 0.58 * vmax else "#111111"
            ax_e.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=6.0, color=txt_col)
    acc_journal = float(np.trace(cm_journal) / max(cm_journal.sum(), 1.0) * 100.0)
    ax_e.text(
        0.99,
        0.03,
        f"{n_journal_models} models, n={n_journal_pred}\nacc={acc_journal:.1f}%",
        transform=ax_e.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.8,
        color="#444444",
        bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#DDDDDD"),
    )

    cb = fig.colorbar(im_e, ax=[ax_c, ax_d, ax_e], fraction=0.018, pad=0.02)
    cb.set_label("Row-normalized share (%)")

    fig.text(
        0.5,
        0.01,
        "Single-pass protocol: thinking/frontier = run-1 (fallback to first valid run if run-1 missing); SFT = argmax(logp). "
        "Panels a-e use models with complete Simple/Journal/Expert coverage under the conservative 11-model frontier cohort.",
        ha="center",
        va="bottom",
        fontsize=6.2,
        color="#555555",
    )
    if out_base is None:
        out_base = OUT_EXT / "ed_fig1"
    return _save(fig, out_base)


def _compute_classifier_metrics(records: List[Tuple[str, str]]) -> Dict[str, object]:
    cm = _compute_confusion_from_records(records)
    n = int(cm.sum())
    tp = np.diag(cm).astype(float)
    row_sum = cm.sum(axis=1).astype(float)
    col_sum = cm.sum(axis=0).astype(float)
    acc = float(tp.sum() / n) if n > 0 else float("nan")
    recall = np.divide(tp, row_sum, out=np.zeros_like(tp), where=row_sum > 0)
    precision = np.divide(tp, col_sum, out=np.zeros_like(tp), where=col_sum > 0)
    f1 = np.divide(
        2.0 * precision * recall,
        precision + recall,
        out=np.zeros_like(tp),
        where=(precision + recall) > 0,
    )
    return {
        "records": records,
        "cm": cm,
        "n": n,
        "acc": acc,
        "acc_ci": _binom_ci95(acc, n),
        "macro_f1": float(np.mean(f1)),
        "recall": recall,
        "precision": precision,
        "pred_dist": np.divide(cm.sum(axis=0), n, out=np.zeros(4, dtype=float), where=n > 0),
        "exceptional_recall": float(recall[0]) if recall.size > 0 else float("nan"),
        "exceptional_precision": float(precision[0]) if precision.size > 0 else float("nan"),
        "strong_to_exceptional": float(cm[1, 0] / row_sum[1]) if row_sum[1] > 0 else float("nan"),
        "one_step_over": float((cm[1, 0] + cm[2, 1] + cm[3, 2]) / n) if n > 0 else float("nan"),
        "one_step_under": float((cm[0, 1] + cm[1, 2] + cm[2, 3]) / n) if n > 0 else float("nan"),
    }


def _bootstrap_classifier_metrics(records: List[Tuple[str, str]], n_boot: int = 4000, seed: int = 42) -> Dict[str, Tuple[float, float]]:
    if not records:
        nan_pair = (float("nan"), float("nan"))
        return {
            "macro_f1": nan_pair,
            "exceptional_precision": nan_pair,
            "exceptional_recall": nan_pair,
            "strong_to_exceptional": nan_pair,
        }

    rng = np.random.default_rng(seed)
    arr = np.asarray(records, dtype=object)
    out = {
        "macro_f1": [],
        "exceptional_precision": [],
        "exceptional_recall": [],
        "strong_to_exceptional": [],
    }
    for _ in range(n_boot):
        idx = rng.integers(0, len(arr), size=len(arr))
        sample = [(str(gt), str(pred)) for gt, pred in arr[idx]]
        metrics = _compute_classifier_metrics(sample)
        for key in out:
            out[key].append(float(metrics[key]))
    return {
        key: (float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5)))
        for key, values in out.items()
    }


def _load_prediction_records(path: Path, allowed_keys: set[str] | None = None) -> Dict[str, List[Tuple[str, str]]]:
    per_model: Dict[str, List[Tuple[str, str]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gt = _normalize_pred_label(row.get("rank", ""))
            if gt not in LABELS:
                continue
            rq = row.get("val_outcome", {}).get("rq_with_context", {})
            if not isinstance(rq, dict):
                continue
            for model_key, model_out in rq.items():
                if allowed_keys is not None and model_key not in allowed_keys:
                    continue
                if not isinstance(model_out, dict):
                    continue
                pred = _normalize_pred_label(model_out.get("prediction_majority") or model_out.get("prediction"))
                if pred not in LABELS:
                    pred = _extract_prediction_from_model_output(model_out)
                if pred in LABELS:
                    per_model.setdefault(model_key, []).append((gt, pred))
    return per_model


def _prediction_from_logp_label_order(logp: Dict[str, object]) -> str | None:
    """Resolve a log-probability prediction with canonical label-order tie handling."""
    best_label: str | None = None
    best_score = float("-inf")
    for label in LABELS:
        score = float("-inf")
        for raw_label, raw_score in logp.items():
            if _normalize_pred_label(str(raw_label)) != label:
                continue
            try:
                score = float(raw_score)
            except (TypeError, ValueError):
                score = float("-inf")
            break
        if score > best_score:
            best_score = score
            best_label = label
    return best_label


def _load_prediction_records_for_eval_key(
    path: Path,
    eval_key: str,
    allowed_keys: set[str] | None = None,
) -> Dict[str, List[Tuple[str, str]]]:
    """Load fixed-label predictions using the package's canonical label-order tie rule."""
    per_model: Dict[str, List[Tuple[str, str]]] = {}
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gt = _normalize_pred_label(row.get("rank", ""))
            if gt not in LABELS:
                continue
            rq = row.get("val_outcome", {}).get(eval_key, {})
            if not isinstance(rq, dict):
                continue
            for model_key, model_out in rq.items():
                if allowed_keys is not None and model_key not in allowed_keys:
                    continue
                if not isinstance(model_out, dict):
                    continue
                pred = _normalize_pred_label(model_out.get("prediction_majority") or model_out.get("prediction"))
                if pred not in LABELS:
                    logp = model_out.get("logp")
                    if isinstance(logp, dict) and logp:
                        pred = _prediction_from_logp_label_order(logp)
                    else:
                        pred = _extract_prediction_from_model_output(model_out)
                if pred in LABELS:
                    per_model.setdefault(model_key, []).append((gt, pred))
    return per_model


def _load_prediction_records_label_order(path: Path, allowed_keys: set[str] | None = None) -> Dict[str, List[Tuple[str, str]]]:
    """Load fixed-label predictions using the package's canonical label-order tie rule."""
    return _load_prediction_records_for_eval_key(path, "rq_with_context", allowed_keys)


def _logp_to_probability_vector(logp: Dict[str, object]) -> np.ndarray:
    vals = np.array([float(logp.get(label, -1e10)) for label in LABELS], dtype=float)
    vals = vals - np.max(vals)
    probs = np.exp(vals)
    denom = float(np.sum(probs))
    if denom <= 0:
        return np.zeros(len(LABELS), dtype=float)
    return probs / denom


def _build_probability_averaged_ensemble(
    path: Path,
    model_keys: Tuple[str, ...],
    key: str,
    label: str,
    color: str,
    marker: str,
) -> Dict[str, object]:
    records: List[Tuple[str, str]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gt = _normalize_pred_label(row.get("rank", ""))
            if gt not in LABELS:
                continue
            rq = row.get("val_outcome", {}).get("rq_with_context", {})
            if not isinstance(rq, dict):
                continue
            probs: List[np.ndarray] = []
            for model_key in model_keys:
                model_out = rq.get(model_key, {})
                logp = model_out.get("logp") if isinstance(model_out, dict) else None
                if not isinstance(logp, dict) or not logp:
                    probs = []
                    break
                probs.append(_logp_to_probability_vector(logp))
            if not probs:
                continue
            pred = LABELS[int(np.argmax(np.mean(probs, axis=0)))]
            records.append((gt, pred))

    if not records:
        raise ValueError(f"No rows available to build ensemble {label} from {path.name}.")

    metrics = _compute_classifier_metrics(records)
    metrics["macro_f1_ci"] = _bootstrap_classifier_metrics(records)["macro_f1"]
    return {
        "key": key,
        "label": label,
        "color": color,
        "marker": marker,
        "member_keys": list(model_keys),
        **metrics,
    }


def load_temporal_trace_comparison_data() -> Dict[str, object]:
    recent_specs = [
        ("ckpt-step-304", "Recent GPT-4.1-nano", COLORS["sft"], "o"),
        ("ckppt-380", "Recent Qwen3-30B", COLORS["sft"], "o"),
    ]
    recent_keys = {key for key, _, _, _ in recent_specs}
    recent_records = _load_prediction_records_label_order(TRAINED_PATH, allowed_keys=recent_keys)

    recent_items: List[Dict[str, object]] = []
    for model_key, label, color, marker in recent_specs:
        records = recent_records.get(model_key, [])
        if not records:
            continue
        metrics = _compute_classifier_metrics(records)
        metrics["macro_f1_ci"] = _bootstrap_classifier_metrics(records)["macro_f1"]
        recent_items.append(
            {
                "key": model_key,
                "label": label,
                "color": color,
                "marker": marker,
                **metrics,
            }
        )

    if not recent_items:
        raise ValueError("No recent SFT records found in package-local sft_predictions.jsonl.")

    ensemble = _build_probability_averaged_ensemble(
        TRAINED_PATH,
        ("ckpt-step-304", "ckppt-380"),
        "matched_2_model_combo",
        "Recent matched 2-model ensemble",
        "#003B73",
        "D",
    )
    recent_items.append(ensemble)

    old_specs = [
        ("ft:gpt-4.1-nano-2025-04-14:personal:ob-rqcontext-old:DI3q8ijY", "Older-trace GPT-4.1-nano", "#C44E52", "s"),
        ("old_qwen30b_checkpoint_178", "Older-trace Qwen3-30B", "#8172B2", "s"),
    ]
    old_keys = {key for key, _, _, _ in old_specs}
    old_records_by_key = _load_prediction_records_label_order(TEMPORAL_OLD_PATH, allowed_keys=old_keys)
    if not old_records_by_key:
        raise ValueError(f"No old-trace prediction records found in {TEMPORAL_OLD_PATH}")
    old_items: List[Dict[str, object]] = []
    for model_key, label, color, marker in old_specs:
        records = old_records_by_key.get(model_key, [])
        if not records:
            continue
        metrics = _compute_classifier_metrics(records)
        metrics["macro_f1_ci"] = _bootstrap_classifier_metrics(records)["macro_f1"]
        old_items.append(
            {
                "key": model_key,
                "label": label,
                "color": color,
                "marker": marker,
                **metrics,
            }
        )

    if not old_items:
        raise ValueError(f"No recognized old-trace prediction tracks found in {TEMPORAL_OLD_PATH}")

    old_focus = _build_probability_averaged_ensemble(
        TEMPORAL_OLD_PATH,
        ("ft:gpt-4.1-nano-2025-04-14:personal:ob-rqcontext-old:DI3q8ijY", "old_qwen30b_checkpoint_178"),
        "matched_2_model_combo",
        "Older-trace matched 2-model ensemble",
        "#8C2D04",
        "D",
    )
    old_items.append(old_focus)

    t04 = pd.read_csv(TABLES_DIR / "T04_AIvsHumanSummary.csv")
    ref_specs = [
        ("Best frontier", "Best Flagship (Gemini 3.1 Pro)", "#4D4D4D"),
        ("Frontier average", "Flagship Average (11 models)", "#999999"),
        ("Expert majority", "Expert Majority Vote (excl. ties)", "#009E73"),
        ("Student majority", "Student Majority Vote (full, excl. ties)", "#E69F00"),
    ]
    references: List[Dict[str, object]] = []
    for label, evaluator, color in ref_specs:
        row = t04.loc[t04["Evaluator"] == evaluator]
        if row.empty:
            continue
        rec = row.iloc[0]
        references.append(
            {
                "label": label,
                "color": color,
                "acc": float(rec["Accuracy (%)"]) / 100.0,
                "acc_lo": float(rec["95% CI Lower"]) / 100.0,
                "acc_hi": float(rec["95% CI Upper"]) / 100.0,
                "macro_f1": float(rec["Macro F1"]),
                "macro_f1_lo": float(rec["Macro F1 95% CI Lower"]),
                "macro_f1_hi": float(rec["Macro F1 95% CI Upper"]),
            }
        )

    return {
        "recent_items": recent_items,
        "recent_ensemble": ensemble,
        "old_items": old_items,
        "old_focus": old_focus,
        "references": references,
        "old_singleton_only": False,
    }


def _augment_transfer_metrics(metrics: Dict[str, object], macro_f1_ci: Tuple[float, float]) -> Dict[str, object]:
    out = dict(metrics)
    cm = np.asarray(metrics["cm"], dtype=int)
    over_n = int(sum(cm[i, j] for i in range(4) for j in range(0, i)))
    under_n = int(sum(cm[i, j] for i in range(4) for j in range(i + 1, 4)))
    pred_counts = cm.sum(axis=0).astype(int)
    out["macro_f1_ci"] = macro_f1_ci
    out["pred_counts"] = pred_counts
    out["over_n"] = over_n
    out["under_n"] = under_n
    out["middle_n"] = int(pred_counts[1] + pred_counts[2])
    out["extreme_n"] = int(pred_counts[0] + pred_counts[3])
    return out


def _serialize_transfer_metrics(metrics: Dict[str, object]) -> Dict[str, object]:
    cm = np.asarray(metrics["cm"], dtype=int)
    recall = np.asarray(metrics["recall"], dtype=float)
    precision = np.asarray(metrics["precision"], dtype=float)
    pred_dist = np.asarray(metrics["pred_dist"], dtype=float)
    pred_counts = np.asarray(metrics["pred_counts"], dtype=int)
    acc = float(metrics["acc"])
    acc_ci = float(metrics["acc_ci"])
    return {
        "n": int(metrics["n"]),
        "correct": int(np.trace(cm)),
        "accuracy": acc,
        "accuracy_pct": acc * 100.0,
        "accuracy_ci_pct": [max(0.0, (acc - acc_ci) * 100.0), min(100.0, (acc + acc_ci) * 100.0)],
        "macro_f1": float(metrics["macro_f1"]),
        "macro_f1_ci": [float(metrics["macro_f1_ci"][0]), float(metrics["macro_f1_ci"][1])],
        "precision": {label: float(precision[i]) for i, label in enumerate(LABELS)},
        "recall": {label: float(recall[i]) for i, label in enumerate(LABELS)},
        "predicted_counts": {label: int(pred_counts[i]) for i, label in enumerate(LABELS)},
        "predicted_share": {label: float(pred_dist[i]) for i, label in enumerate(LABELS)},
        "confusion_matrix": cm.tolist(),
        "over_errors": int(metrics["over_n"]),
        "under_errors": int(metrics["under_n"]),
    }


def load_core_rq_short_transfer_data() -> Dict[str, object]:
    if not CORE_RQ_SHORT_TRANSFER_PATH.exists():
        raise FileNotFoundError(f"Missing package-local core_rq_short transfer file: {CORE_RQ_SHORT_TRANSFER_PATH}")

    specs = [
        {
            "label": "GPT-4.1 base",
            "family": "GPT-4.1",
            "training": "Base",
            "short_key": "gpt-4.1-2025-04-14",
            "context_key": "gpt-4.1",
            "color": COLORS["baseline"],
        },
        {
            "label": "GPT-4.1 SFT",
            "family": "GPT-4.1",
            "training": "SFT",
            "short_key": "ft:gpt-4.1-2025-04-14:personal:ob-ob-rqcontext:DHnLrzmY",
            "context_key": "CYqJRxId",
            "color": COLORS["sft"],
        },
        {
            "label": "GPT-4.1-nano base",
            "family": "GPT-4.1-nano",
            "training": "Base",
            "short_key": "gpt-4.1-nano-2025-04-14",
            "context_key": "gpt-4.1-nano",
            "color": "#E69F00",
        },
        {
            "label": "GPT-4.1-nano SFT",
            "family": "GPT-4.1-nano",
            "training": "SFT",
            "short_key": "ft:gpt-4.1-nano-2025-04-14:personal:ob-ob-rqcontext:DHKeHMNB",
            "context_key": "ckpt-step-304",
            "color": "#56B4E9",
        },
    ]

    short_records = _load_prediction_records_for_eval_key(
        CORE_RQ_SHORT_TRANSFER_PATH,
        "core_rq_short",
        allowed_keys={str(spec["short_key"]) for spec in specs},
    )
    context_chat = _load_prediction_records_label_order(
        CHAT_PATH,
        allowed_keys={str(spec["context_key"]) for spec in specs if spec["training"] == "Base"},
    )
    context_sft = _load_prediction_records_label_order(
        TRAINED_PATH,
        allowed_keys={str(spec["context_key"]) for spec in specs if spec["training"] == "SFT"},
    )

    items: List[Dict[str, object]] = []
    payload_models: Dict[str, Dict[str, object]] = {}
    for spec in specs:
        short = short_records.get(str(spec["short_key"]), [])
        context_bundle = context_chat if spec["training"] == "Base" else context_sft
        context = context_bundle.get(str(spec["context_key"]), [])
        if not short or not context:
            raise ValueError(f"Missing transfer records for {spec['label']}")

        short_metrics = _augment_transfer_metrics(
            _compute_classifier_metrics(short),
            _bootstrap_classifier_metrics(short)["macro_f1"],
        )
        context_metrics = _augment_transfer_metrics(
            _compute_classifier_metrics(context),
            _bootstrap_classifier_metrics(context)["macro_f1"],
        )
        item = {
            **spec,
            "short": short_metrics,
            "context": context_metrics,
            "delta_acc_pp": (float(short_metrics["acc"]) - float(context_metrics["acc"])) * 100.0,
            "delta_macro_f1": float(short_metrics["macro_f1"]) - float(context_metrics["macro_f1"]),
            "recall_delta_pp": (
                np.asarray(short_metrics["recall"], dtype=float) - np.asarray(context_metrics["recall"], dtype=float)
            )
            * 100.0,
        }
        items.append(item)
        payload_models[str(spec["label"])] = {
            "family": str(spec["family"]),
            "training": str(spec["training"]),
            "rq_with_context": _serialize_transfer_metrics(context_metrics),
            "core_rq_short": _serialize_transfer_metrics(short_metrics),
            "delta_accuracy_pp": float(item["delta_acc_pp"]),
            "delta_macro_f1": float(item["delta_macro_f1"]),
        }

    CORE_RQ_SHORT_TRANSFER_STATS_PATH.write_text(
        json.dumps(
            {
                "figure": "ExtendedDataFigure7",
                "evaluation": "core_rq_short transfer from rq_with_context supervision",
                "scoring_rule": {
                    "prediction_rule": "argmax(logp)",
                    "missing_labels": "treated as -inf",
                    "tie_rule": "canonical label order exceptional > strong > fair > limited",
                },
                "models": payload_models,
            },
            indent=2,
        )
        + "\n",
        encoding="utf-8",
    )

    return {
        "items": items,
        "chance_acc_pct": 25.0,
        "stats_path": CORE_RQ_SHORT_TRANSFER_STATS_PATH,
    }


def load_pairwise_metrics() -> Dict[str, Dict[str, object]]:
    pairwise_files = {
        "SFT GPT-4.1": PAIRWISE_ROOT / "sft_gpt4_1",
        "GPT-5.2 High": PAIRWISE_ROOT / "frontier_gpt5_2_high",
        "GPT-4.1 Baseline": PAIRWISE_ROOT / "baseline_gpt4_1",
    }
    if not PAIRWISE_FRONTIER_GEMINI.exists():
        raise FileNotFoundError(f"Missing required pairwise directory: {PAIRWISE_FRONTIER_GEMINI}")
    pairwise_files["Gemini 3.1 Pro"] = PAIRWISE_FRONTIER_GEMINI

    out: Dict[str, Dict[str, object]] = {}
    for model_name, model_dir in pairwise_files.items():
        metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
        rows: List[Dict[str, object]] = []
        with (model_dir / "pair_results.jsonl").open("r", encoding="utf-8") as f:
            for line in f:
                row = json.loads(line)
                rows.append(row)
        out[model_name] = {"metrics": metrics, "rows": rows}
    return out


def _pairwise_item_key(row: Dict[str, object]) -> Tuple[str, int, str]:
    pair_id = str(row.get("pair_id", "")).strip()
    if not pair_id:
        raise ValueError("Pairwise row is missing required public pair_id field.")
    return (
        str(row.get("pair_type", "")),
        int(row.get("distance", 0) or 0),
        pair_id,
    )


def _exact_mcnemar_p_raw(left_only: int, right_only: int) -> float:
    n = left_only + right_only
    if n <= 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(left_only, right_only) + 1)) / (2**n)
    return float(min(1.0, 2.0 * tail))


def _write_ed2_pairwise_raw_stats(pairwise: Dict[str, Dict[str, object]], frontier_label: str) -> Dict[str, object]:
    plotted_models = ["SFT GPT-4.1", frontier_label, "GPT-5.2 High", "GPT-4.1 Baseline"]
    correctness = {}
    for model in plotted_models:
        rows = pairwise[model]["rows"]  # type: ignore[index]
        correctness[model] = {_pairwise_item_key(row): bool(row.get("is_correct", False)) for row in rows}

    reference_keys = set(correctness["SFT GPT-4.1"])
    for model, vector in correctness.items():
        if set(vector) != reference_keys:
            raise ValueError(f"ED2 pairwise item mismatch for {model}: expected {len(reference_keys)} shared items")

    comparisons = []
    significant = []
    discordance_by_pair_type: Dict[str, Dict[str, Dict[str, int]]] = {}
    for comparator in plotted_models[1:]:
        sft_only = 0
        comparator_only = 0
        both_correct = 0
        both_wrong = 0
        pair_type_breakdown: Dict[str, Dict[str, int]] = {}
        for item_key in reference_keys:
            sft_ok = correctness["SFT GPT-4.1"][item_key]
            comparator_ok = correctness[comparator][item_key]
            pair_type = str(item_key[0])
            bucket = pair_type_breakdown.setdefault(
                pair_type,
                {
                    "sft_only_correct": 0,
                    "comparator_only_correct": 0,
                    "both_correct": 0,
                    "both_wrong": 0,
                    "net_sft_advantage": 0,
                },
            )
            if sft_ok and not comparator_ok:
                sft_only += 1
                bucket["sft_only_correct"] += 1
                bucket["net_sft_advantage"] += 1
            elif comparator_ok and not sft_ok:
                comparator_only += 1
                bucket["comparator_only_correct"] += 1
                bucket["net_sft_advantage"] -= 1
            elif sft_ok and comparator_ok:
                both_correct += 1
                bucket["both_correct"] += 1
            else:
                both_wrong += 1
                bucket["both_wrong"] += 1

        n_items = len(reference_keys)
        sft_acc = sum(correctness["SFT GPT-4.1"].values()) / n_items
        comparator_acc = sum(correctness[comparator].values()) / n_items
        p_raw = _exact_mcnemar_p_raw(sft_only, comparator_only)
        record = {
            "evaluator_1": "SFT GPT-4.1",
            "evaluator_2": comparator,
            "n_paired_items": n_items,
            "accuracy_1": sft_acc,
            "accuracy_2": comparator_acc,
            "accuracy_diff_pp": (sft_acc - comparator_acc) * 100.0,
            "sft_only_correct": sft_only,
            "comparator_only_correct": comparator_only,
            "both_correct": both_correct,
            "both_wrong": both_wrong,
            "test": "Exact McNemar (two-sided binomial on discordant pairs)",
            "p_value_raw": p_raw,
            "p_value_adjustment": "none",
            "significant_at_0_05_raw": p_raw < 0.05,
        }
        comparisons.append(record)
        if record["significant_at_0_05_raw"]:
            significant.append(comparator)
        discordance_by_pair_type[comparator] = {
            pair_type: pair_type_breakdown.get(
                pair_type,
                {
                    "sft_only_correct": 0,
                    "comparator_only_correct": 0,
                    "both_correct": 0,
                    "both_wrong": 0,
                    "net_sft_advantage": 0,
                },
            )
            for pair_type in PAIR_TYPE_ORDER
        }

    pair_type_accuracy_pct: Dict[str, Dict[str, Dict[str, float]]] = {}
    for model in plotted_models:
        rows = pairwise[model]["rows"]  # type: ignore[index]
        by_pair_type: Dict[str, Dict[str, float]] = {}
        for pair_type in PAIR_TYPE_ORDER:
            subset = [row for row in rows if str(row.get("pair_type")) == pair_type]
            n_rows = len(subset)
            correct = sum(int(bool(row.get("is_correct", False))) for row in subset)
            by_pair_type[pair_type] = {
                "n": float(n_rows),
                "correct": float(correct),
                "accuracy_pct": (100.0 * correct / n_rows) if n_rows else float("nan"),
            }
        pair_type_accuracy_pct[model] = by_pair_type

    payload = {
        "figure": "ExtendedDataFigure2",
        "summary": {
            "plotted_models": plotted_models,
            "frontier_comparator_plotted": frontier_label,
            "source_root": "data/pairwise",
            "paired_item_alignment": "300 shared canonical pairwise items",
            "test_method": "Exact McNemar (two-sided binomial on discordant paired correctness outcomes)",
            "p_values_reported": "raw_unadjusted_only",
            "alpha_reference": 0.05,
            "significant_comparisons_raw": significant,
        },
        "pair_type_order": [{"key": pair_type, "label": PAIR_TYPE_LABELS[pair_type]} for pair_type in PAIR_TYPE_ORDER],
        "pair_type_accuracy_pct": pair_type_accuracy_pct,
        "discordance_by_pair_type": discordance_by_pair_type,
        "comparisons": comparisons,
    }
    PAIRWISE_RAW_STATS_PATH.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return payload


def load_rl_metrics() -> Dict[str, object]:
    rows: List[Dict[str, object]] = []
    with RL_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))
    if not rows:
        raise ValueError(f"No RL rows found in {RL_PATH}")

    model_keys = sorted(
        {
            k
            for r in rows
            for k in (r.get("val_outcome", {}).get("rq_with_context", {}) or {}).keys()
        }
    )
    if not model_keys:
        raise ValueError(f"No RL model keys found in {RL_PATH}")
    model_key = model_keys[0]
    idx = {k: i for i, k in enumerate(LABELS)}

    cm_major = np.zeros((4, 4), dtype=int)
    cm_run = np.zeros((4, 4), dtype=int)
    valid_run_hist: Dict[int, int] = {k: 0 for k in range(9)}
    article_avg8: List[float] = []

    run_correct = 0
    run_total = 0
    majority_correct = 0
    majority_total = 0
    tie_n = 0

    for r in rows:
        gt = str(r.get("rank"))
        if gt not in idx:
            continue
        out = (r.get("val_outcome", {}).get("rq_with_context", {}) or {}).get(model_key, {})
        if not isinstance(out, dict):
            continue

        vv = int(out.get("vote_valid_n", 0) or 0)
        valid_run_hist[vv] = valid_run_hist.get(vv, 0) + 1
        if out.get("avg_accuracy") is not None:
            article_avg8.append(float(out["avg_accuracy"]))

        # Run-level pooled accuracy/confusion
        for pred in out.get("vote_predictions", []) or []:
            if pred in idx:
                cm_run[idx[gt], idx[pred]] += 1
                run_total += 1
                if pred == gt:
                    run_correct += 1

        # Majority-vote accuracy/confusion (ties excluded)
        vote_counts = out.get("vote_counts", {}) or {}
        if isinstance(vote_counts, dict) and vote_counts:
            top = max(vote_counts.values())
            modes = [k for k, v in vote_counts.items() if v == top and k in idx]
            if len(modes) == 1:
                pred = modes[0]
                cm_major[idx[gt], idx[pred]] += 1
                majority_total += 1
                if pred == gt:
                    majority_correct += 1
            elif len(modes) > 1:
                tie_n += 1

    run_acc = float(run_correct / run_total) if run_total > 0 else float("nan")
    avg8_article_acc = float(np.mean(article_avg8)) if article_avg8 else float("nan")
    major_acc = float(majority_correct / majority_total) if majority_total > 0 else float("nan")
    major_pred_dist = (cm_major.sum(axis=0) / max(cm_major.sum(), 1)).astype(float)
    run_pred_dist = (cm_run.sum(axis=0) / max(cm_run.sum(), 1)).astype(float)

    return {
        "model_key": model_key,
        "n_articles": len(rows),
        "run_acc": run_acc,
        "avg8_article_acc": avg8_article_acc,
        "article_avg8_values": article_avg8,
        "majority_acc": major_acc,
        "run_correct": run_correct,
        "run_total": run_total,
        "majority_correct": majority_correct,
        "majority_total": majority_total,
        "majority_ties": tie_n,
        "cm_major": cm_major,
        "cm_run": cm_run,
        "major_pred_dist": major_pred_dist,
        "run_pred_dist": run_pred_dist,
        "valid_run_hist": valid_run_hist,
    }


def make_ed2_pairwise_intrinsic_v2() -> Tuple[Path, Path]:
    _set_style()
    d = load_pairwise_metrics()
    frontier_label = "Gemini 3.1 Pro"
    frontier_color = COLORS["gemini"]
    stats = _write_ed2_pairwise_raw_stats(d, frontier_label)
    order = ["SFT GPT-4.1", frontier_label, "GPT-5.2 High", "GPT-4.1 Baseline"]
    colors = [COLORS["sft"], frontier_color, COLORS["gpt52"], COLORS["baseline"]]
    short_names = {
        "SFT GPT-4.1": "SFT GPT-4.1",
        "Gemini 3.1 Pro": "Gemini 3.1",
        "GPT-5.2 High": "GPT-5.2 High",
        "GPT-4.1 Baseline": "GPT-4.1 base",
    }

    fig = plt.figure(figsize=(11.0, 4.8))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.38)

    # a) Pair-type heatmap
    ax = fig.add_subplot(gs[0, 0])
    heat = np.array(
        [
            [float(stats["pair_type_accuracy_pct"][model][pair_type]["accuracy_pct"]) for model in order]  # type: ignore[index]
            for pair_type in PAIR_TYPE_ORDER
        ]
    )
    im = ax.imshow(heat, cmap="Blues", vmin=50, vmax=100, aspect="auto")
    ax.set_xticks(np.arange(len(order)))
    ax.set_xticklabels([short_names[k] for k in order], rotation=18, ha="right")
    ax.set_yticks(np.arange(len(PAIR_TYPE_ORDER)))
    ax.set_yticklabels([PAIR_TYPE_LABELS[key] for key in PAIR_TYPE_ORDER])
    ax.set_title("a  Six pair types localize the real boundary gains", loc="left")
    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            value = heat[i, j]
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=5.8, color="white" if value >= 78 else "black")
    ax.axhline(2.5, color="white", linewidth=0.9)
    ax.text(0.98, 0.02, "Accuracy (%) on 50 pairs per row", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.4, color="#444444")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cbar.ax.tick_params(labelsize=5.4)
    cbar.set_label("Accuracy (%)", fontsize=6)

    # b) Net paired discordance by pair type
    ax = fig.add_subplot(gs[0, 1])
    comparators = [frontier_label, "GPT-5.2 High", "GPT-4.1 Baseline"]
    offsets = [-0.24, 0.0, 0.24]
    y_base = np.arange(len(PAIR_TYPE_ORDER))
    for offset, model, color in zip(offsets, comparators, colors[1:]):
        vals = np.array(
            [int(stats["discordance_by_pair_type"][model][pair_type]["net_sft_advantage"]) for pair_type in PAIR_TYPE_ORDER]  # type: ignore[index]
        )
        ax.barh(y_base + offset, vals, height=0.22, color=color, alpha=0.9, label=f"SFT vs {short_names[model]}")
        for yv, xv in zip(y_base + offset, vals):
            if xv != 0:
                ax.text(xv + (0.18 if xv >= 0 else -0.18), yv, f"{xv:+d}", va="center", ha="left" if xv >= 0 else "right", fontsize=5.5)
    ax.axvline(0, color="#444444", linewidth=0.8)
    ax.set_yticks(y_base)
    ax.set_yticklabels([PAIR_TYPE_LABELS[key] for key in PAIR_TYPE_ORDER])
    ax.invert_yaxis()
    ax.set_xlim(-4.5, 10.5)
    ax.set_xlabel("Net paired advantage for SFT on discordant pairs")
    ax.set_title("b  Raw paired wins cluster in the hardest pair types", loc="left")
    ax.legend(fontsize=5.1, loc="lower right")
    ax.text(
        0.02,
        0.02,
        "Positive values = SFT-only correct exceeds comparator-only correct",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=5.3,
        color="#444444",
    )

    return _save(fig, OUT_EXT / "ed_fig2")


def _row_norm(cm: np.ndarray) -> np.ndarray:
    cm = np.asarray(cm, dtype=float)
    s = cm.sum(axis=1, keepdims=True)
    s[s == 0] = 1.0
    return cm / s


def _plot_confusion(ax: plt.Axes, cm: np.ndarray, title: str, cmap: str = "Blues") -> None:
    cmn = _row_norm(cm)
    ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(LABELS_SHORT, fontsize=4.5, rotation=35, ha="right")
    ax.set_yticklabels(LABELS_SHORT, fontsize=4.5)
    ax.set_title(title, fontsize=5.4, pad=1.5)
    for i in range(4):
        for j in range(4):
            v = cmn[i, j]
            ax.text(j, i, f"{v:.0%}", ha="center", va="center", fontsize=3.4, color="white" if v > 0.48 else "black")


def _load_chat_baseline_confusions() -> Dict[str, Dict[str, object]]:
    """Compute base-model confusion matrices from the canonical chat predictions file."""
    model_map = [
        ("gpt-4.1", "Base GPT-4.1"),
        ("gpt-4.1-nano", "Base GPT-4.1-nano"),
        ("qwen3-30b-a3b", "Base Qwen3-30B"),
        ("qwen3-4b", "Base Qwen3-4B"),
    ]
    idx = {k: i for i, k in enumerate(LABELS)}
    rows: List[Dict[str, object]] = []
    with CHAT_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    out: Dict[str, Dict[str, object]] = {}
    for model_key, display_name in model_map:
        cm = np.zeros((4, 4), dtype=int)
        n_valid = 0
        for row in rows:
            gt = _normalize_pred_label(str(row.get("rank", "")))
            if gt not in idx:
                continue
            model_out = (row.get("val_outcome", {}).get("rq_with_context", {}) or {}).get(model_key, {})
            if not isinstance(model_out, dict):
                continue
            pred = _extract_prediction_from_model_output(model_out)
            if pred not in idx:
                continue
            cm[idx[gt], idx[pred]] += 1
            n_valid += 1
        if n_valid == 0:
            raise ValueError(f"No valid predictions found for base model key '{model_key}' in {CHAT_PATH}")
        acc = float(np.trace(cm) / n_valid)
        out[display_name] = {
            "model_key": model_key,
            "confusion_matrix": cm,
            "accuracy": acc,
            "n_valid": n_valid,
        }
    return out


def _load_st7_ensemble_table() -> pd.DataFrame:
    model_names = {
        "CYqJRxId": "GPT-4.1",
        "ckpt-step-304": "GPT-4.1-nano",
        "ckppt-380": "Qwen3-30B",
        "ckppt-228": "Qwen3-4B",
    }
    sft_keys = list(model_names.keys())

    def _logp_to_probs(logp: Dict[str, object]) -> Dict[str, float]:
        vals = np.array([float(logp.get(label, -1e10)) for label in LABELS], dtype=float)
        vals = vals - np.max(vals)
        probs = np.exp(vals)
        denom = float(np.sum(probs))
        if denom <= 0:
            return {label: 0.0 for label in LABELS}
        probs = probs / denom
        return {label: float(prob) for label, prob in zip(LABELS, probs)}

    rows: List[Dict[str, object]] = []
    with TRAINED_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    data = []
    for combo in itertools.combinations(sft_keys, 2):
        correct = 0
        total = 0
        for row in rows:
            rq = (row.get("val_outcome", {}).get("rq_with_context", {}) or {})
            gt = _normalize_pred_label(str(row.get("rank", "")))
            if gt not in LABELS:
                continue
            logps = []
            for model_key in combo:
                model_out = rq.get(model_key, {})
                logp = model_out.get("logp") if isinstance(model_out, dict) else None
                if not isinstance(logp, dict) or not logp:
                    logps = []
                    break
                logps.append(logp)
            if not logps:
                continue
            probs = [_logp_to_probs(logp) for logp in logps]
            avg = {label: float(np.mean([prob[label] for prob in probs])) for label in LABELS}
            pred = max(avg.items(), key=lambda item: (item[1], item[0]))[0]
            total += 1
            correct += int(pred == gt)
        acc = (correct / total * 100.0) if total else float("nan")
        data.append((model_names[combo[0]], model_names[combo[1]], acc))

    return pd.DataFrame(data, columns=["model_1", "model_2", "accuracy_pct"])


def make_ed3_confusion_ensemble_v2() -> Tuple[Path, Path]:
    _set_style()
    stats02 = json.loads((STATS_DIR / "S01_FlagshipStats.json").read_text(encoding="utf-8"))
    stats03 = json.loads((STATS_DIR / "S02_SFTStats.json").read_text(encoding="utf-8"))
    t3 = pd.read_csv(TABLES_DIR / "T03_SFTPerformance.csv")
    t8 = pd.read_csv(TABLES_DIR / "T04_AIvsHumanSummary.csv")
    st7 = _load_st7_ensemble_table()

    fig = plt.figure(figsize=(12.5, 9.8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.24, wspace=0.24)

    # a) All flagship confusions (title row + 3x4 grid to prevent overlap)
    gs_a = gridspec.GridSpecFromSubplotSpec(
        4, 4, subplot_spec=gs[:, 0], height_ratios=[0.18, 1.0, 1.0, 1.0], hspace=0.42, wspace=0.32
    )
    ax_title_a = fig.add_subplot(gs_a[0, :])
    ax_title_a.axis("off")
    ax_title_a.text(
        0.0,
        0.35,
        "a  Full 11-model flagship confusion atlas",
        transform=ax_title_a.transAxes,
        ha="left",
        va="center",
    )
    rank = stats02.get("ranking_by_accuracy", [])
    perf = stats02.get("model_performance", {})
    for i in range(12):
        ax = fig.add_subplot(gs_a[1 + i // 4, i % 4])
        if i >= len(rank):
            ax.axis("off")
            continue
        mk = rank[i]["model"]
        cm = np.array(perf[mk]["confusion_matrix"], dtype=float)
        acc = float(perf[mk]["accuracy"]) * 100.0
        _plot_confusion(ax, cm, f"{mk}\n{acc:.1f}%")

    # b) Matched base-vs-SFT confusions by architecture (title row + 2x4 grid)
    gs_b = gridspec.GridSpecFromSubplotSpec(
        3, 4, subplot_spec=gs[0, 1], height_ratios=[0.24, 1.0, 1.0], hspace=0.58, wspace=0.40
    )
    ax_title_b = fig.add_subplot(gs_b[0, :])
    ax_title_b.axis("off")
    ax_title_b.text(
        0.0,
        0.35,
        "b  Matched base-vs-SFT confusion maps (4 architectures)",
        transform=ax_title_b.transAxes,
        ha="left",
        va="center",
    )
    perf_base = _load_chat_baseline_confusions()
    perf_sft = stats03.get("model_performance", {})
    architecture_pairs = [
        ("Base GPT-4.1", "SFT (GPT-4.1)"),
        ("Base GPT-4.1-nano", "SFT (GPT-4.1-nano)"),
        ("Base Qwen3-30B", "SFT (Qwen3-30B)"),
        ("Base Qwen3-4B", "SFT (Qwen3-4B)"),
    ]
    panel_b_entries: List[Tuple[str, np.ndarray, float, str]] = []
    for base_name, sft_name in architecture_pairs:
        panel_b_entries.append(
            (
                base_name,
                np.array(perf_base[base_name]["confusion_matrix"], dtype=float),
                float(perf_base[base_name]["accuracy"]) * 100.0,
                "Blues",
            )
        )
        panel_b_entries.append(
            (
                sft_name,
                np.array(perf_sft[sft_name]["confusion_matrix"], dtype=float),
                float(perf_sft[sft_name]["accuracy"]) * 100.0,
                "Reds",
            )
        )
    for i, (name, cm, acc, cmap) in enumerate(panel_b_entries):
        ax = fig.add_subplot(gs_b[1 + i // 4, i % 4])
        display_name = name.replace("SFT (", "SFT ").replace(")", "")
        _plot_confusion(ax, cm, f"{display_name}\n{acc:.1f}%", cmap=cmap)

    # d) Singles vs all pairwise ensembles
    single_acc: Dict[str, float] = {}
    for _, r in t3.iterrows():
        model = str(r["Model"])
        if not (model.startswith("SFT (") and model.endswith(")")):
            continue
        name = model[len("SFT (") : -1]
        single_acc[name] = float(r["Accuracy"]) * 100.0

    expected_single = {"GPT-4.1", "GPT-4.1-nano", "Qwen3-30B", "Qwen3-4B"}
    missing_single = sorted(expected_single - set(single_acc.keys()))
    if missing_single:
        raise ValueError(f"Missing SFT single-model rows in T03_SFTPerformance.csv: {missing_single}")

    ax = fig.add_subplot(gs[1, 1])
    bars = []
    labels = []
    vals = []
    for k, v in sorted(single_acc.items(), key=lambda kv: kv[1], reverse=True):
        labels.append(k + " (single)")
        vals.append(v)
        bars.append("single")
    for _, r in st7.sort_values("accuracy_pct", ascending=False).iterrows():
        labels.append(f"{r['model_1']} + {r['model_2']}")
        vals.append(float(r["accuracy_pct"]))
        bars.append("pair")
    y = np.arange(len(labels))
    color_map = ["#4C78A8" if b == "single" else "#F58518" for b in bars]
    ax.barh(y, vals, color=color_map, edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.2)
    ax.invert_yaxis()
    ax.set_xlabel("Accuracy (%)")
    ax.set_title("c  Pairwise ensemble landscape (non-overlap detail)", loc="left")
    best_flagship = float(
        t8.loc[t8["Evaluator"].str.startswith("Best Flagship ("), "Accuracy (%)"].iloc[0]
    )
    expert_mv = float(
        t8.loc[t8["Evaluator"] == "Expert Majority Vote (excl. ties)", "Accuracy (%)"].iloc[0]
    )
    ax.axvline(best_flagship, color=COLORS["gemini"], linestyle="--", linewidth=1.0)
    ax.axvline(expert_mv, color=COLORS["expert"], linestyle=":", linewidth=1.0)
    ax.set_xlim(0, max(vals) + 5.0)
    ax.text(
        0.98,
        0.07,
        f"Best flagship {best_flagship:.1f}\nExpert majority {expert_mv:.1f}",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.7,
        bbox=dict(boxstyle="round,pad=0.2", facecolor="white", edgecolor="#cccccc"),
        color="#444444",
    )

    return _save(fig, OUT_EXT / "ed_fig3")


def _frontier_primary_label(model_out: Dict[str, object]) -> str | None:
    """Resolve a single primary label from frontier model output."""
    pred = None
    if not model_out.get("vote_is_tie", False):
        pred = model_out.get("prediction_majority") or model_out.get("prediction")
    pred_norm = _normalize_pred_label(pred)
    if pred_norm in LABELS:
        return pred_norm

    vote_counts = model_out.get("vote_counts")
    if isinstance(vote_counts, dict) and vote_counts:
        top = max(vote_counts.values())
        modes = sorted([k for k, v in vote_counts.items() if v == top and _normalize_pred_label(k) in LABELS])
        if modes:
            mode_norm = _normalize_pred_label(modes[0])
            if mode_norm in LABELS:
                return mode_norm

    vote_predictions = model_out.get("vote_predictions")
    if isinstance(vote_predictions, list):
        for raw in vote_predictions:
            cand = _normalize_pred_label(raw)
            if cand in LABELS:
                return cand

    pred_fallback = _extract_prediction_from_model_output(model_out)
    return pred_fallback if pred_fallback in LABELS else None


def _compute_frontier_average_error_profile() -> Dict[str, float]:
    """Average error profile across the conservative 11-model frontier cohort."""
    rows: List[Dict[str, object]] = []
    with THINKING_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    t3 = pd.read_csv(TABLES_DIR / "T02_FlagshipPerformance.csv")
    model_keys = [str(v) for v in t3["Model Key"].dropna().tolist()]
    idx = {k: i for i, k in enumerate(LABELS)}

    per_model: List[Dict[str, float]] = []
    for mk in model_keys:
        exact = off1 = off2 = over = under = total = 0
        for r in rows:
            gt = _normalize_pred_label(r.get("rank", ""))
            if gt not in idx:
                continue
            out = (r.get("val_outcome", {}).get("rq_with_context", {}) or {}).get(mk, {})
            if not isinstance(out, dict) or not out:
                continue
            pred = _frontier_primary_label(out)
            if pred not in idx:
                continue
            total += 1
            diff = idx[pred] - idx[gt]
            if diff == 0:
                exact += 1
            elif abs(diff) == 1:
                off1 += 1
            else:
                off2 += 1
            if diff < 0:
                over += 1
            elif diff > 0:
                under += 1
        if total > 0:
            per_model.append(
                {
                    "exact_pct": 100.0 * exact / total,
                    "off1_pct": 100.0 * off1 / total,
                    "off2_pct": 100.0 * off2 / total,
                    "over_pct": 100.0 * over / total,
                    "under_pct": 100.0 * under / total,
                }
            )

    if not per_model:
        raise ValueError("Could not compute frontier-average error profile: no valid model predictions.")

    def _mean(key: str) -> float:
        return float(np.mean([m[key] for m in per_model]))

    return {
        "exact_pct": _mean("exact_pct"),
        "off1_pct": _mean("off1_pct"),
        "off2_pct": _mean("off2_pct"),
        "over_pct": _mean("over_pct"),
        "under_pct": _mean("under_pct"),
    }


def make_ed4_calibration_error_reliability_v2(
    out_base: Path | None = None,
    include_accuracy_ci: bool = False,
) -> Tuple[Path, Path]:
    _set_style()
    calib_tbl = pd.read_csv(TABLES_DIR / "T13_CalibrationMetrics.csv")
    err_tbl = pd.read_csv(TABLES_DIR / "T14_ErrorAnalysis.csv")
    s24 = pd.read_csv(TABLES_DIR / "T19_StudentReliabilityByPanelSize.csv")
    s25 = pd.read_csv(TABLES_DIR / "T20_ExpertReliabilityByPanelSize.csv")
    frontier_avg = _compute_frontier_average_error_profile()

    fig = plt.figure(figsize=(10.8, 8.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.27)

    # a) Over/under error direction profile (non-overlap with main calibration panels)
    ax = fig.add_subplot(gs[0, 0])
    keep = ["SFT 2-Model", "Best Flagship", "Frontier Average", "Expert Majority", "Student Majority"]
    sub_err = err_tbl[err_tbl["Evaluator"].isin(keep)].copy()
    frontier_err = pd.DataFrame(
        [
            {
                "Evaluator": "Frontier Average",
                "Over %": frontier_avg["over_pct"],
                "Under %": frontier_avg["under_pct"],
            }
        ]
    )
    sub_err = pd.concat([sub_err, frontier_err], ignore_index=True)
    sub_err["Evaluator"] = pd.Categorical(sub_err["Evaluator"], categories=keep, ordered=True)
    sub_err = sub_err.sort_values("Evaluator")
    y = np.arange(len(sub_err))
    under = -sub_err["Under %"].to_numpy(dtype=float)
    over = sub_err["Over %"].to_numpy(dtype=float)
    ax.barh(y, under, color="#4E79A7", edgecolor="white", label="Under-estimation")
    ax.barh(y, over, color="#E15759", edgecolor="white", label="Over-estimation")
    ax.axvline(0.0, color="#444444", linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([str(v).replace("Student", "Junior") for v in sub_err["Evaluator"]], fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Directional error share (%)")
    ax.set_title("a  Error direction profile (over vs under)", loc="left")
    for i, (u, o) in enumerate(zip(under, over)):
        ax.text(u - 1.0, i, f"{abs(u):.1f}", ha="right", va="center", fontsize=5.2)
        ax.text(o + 1.0, i, f"{o:.1f}", ha="left", va="center", fontsize=5.2)
    ax.legend(fontsize=5.3, loc="lower right")

    # b) ECE-Brier landscape
    ax = fig.add_subplot(gs[0, 1])
    ax.scatter(calib_tbl["ece"], calib_tbl["brier_score"], s=55, color="#4E79A7", alpha=0.85, edgecolors="white")
    for _, r in calib_tbl.iterrows():
        label = str(r["evaluator"]).replace("Student", "Junior")
        ax.text(float(r["ece"]) + 0.006, float(r["brier_score"]) + 0.01, label, fontsize=5.5)
    ax.set_xlabel("ECE (lower is better)")
    ax.set_ylabel("Brier score (lower is better)")
    ax.set_title("b  Calibration-error landscape (ECE vs Brier)", loc="left")

    # c) Error severity + direction
    ax = fig.add_subplot(gs[1, 0])
    keep = ["SFT 2-Model", "Best Flagship", "Frontier Average", "Expert Majority", "Student Majority"]
    sub = err_tbl[err_tbl["Evaluator"].isin(keep)].copy()
    frontier_sev = pd.DataFrame(
        [
            {
                "Evaluator": "Frontier Average",
                "Exact Match %": frontier_avg["exact_pct"],
                "Off-by-One %": frontier_avg["off1_pct"],
                "Off-by-Two+ %": frontier_avg["off2_pct"],
                "Over %": frontier_avg["over_pct"],
                "Under %": frontier_avg["under_pct"],
            }
        ]
    )
    sub = pd.concat([sub, frontier_sev], ignore_index=True)
    sub["Evaluator"] = pd.Categorical(sub["Evaluator"], categories=keep, ordered=True)
    sub = sub.sort_values("Evaluator")
    x = np.arange(len(sub))
    exact = sub["Exact Match %"].to_numpy(dtype=float)
    off1 = sub["Off-by-One %"].to_numpy(dtype=float)
    off2 = sub["Off-by-Two+ %"].to_numpy(dtype=float)
    ax.bar(x, exact, color="#59A14F", edgecolor="white", label="Exact")
    ax.bar(x, off1, bottom=exact, color="#F28E2B", edgecolor="white", label="Off-by-1")
    ax.bar(x, off2, bottom=exact + off1, color="#E15759", edgecolor="white", label="Off-by-2+")
    ax.set_xticks(x)
    xt = [str(v).replace("Student", "Junior") for v in sub["Evaluator"]]
    ax.set_xticklabels(xt, rotation=24, ha="right")
    ax.set_ylabel("Share (%)")
    ax.set_ylim(0, 100)
    ax.set_title("c  Error severity structure", loc="left")
    ax.legend(fontsize=5.5, ncol=3, loc="upper right")
    for i, r in sub.reset_index(drop=True).iterrows():
        over = float(r["Over %"])
        under = float(r["Under %"])
        ax.text(i, 3.2, f"O:{over:.1f} U:{under:.1f}", ha="center", fontsize=5.2)

    # d) Panel-size reliability and tie-rate
    ax = fig.add_subplot(gs[1, 1])
    ax.plot(
        s24["k"],
        s24["accuracy_mean"] * 100,
        "-o",
        color=COLORS["student"],
        markersize=3,
        linewidth=1.1,
        label="Junior accuracy",
    )
    ax.plot(
        s25["k"],
        s25["accuracy_mean"] * 100,
        "-o",
        color=COLORS["expert"],
        markersize=3,
        linewidth=1.1,
        label="Expert accuracy",
    )
    if include_accuracy_ci:
        if {"accuracy_ci_lower", "accuracy_ci_upper"}.issubset(s24.columns):
            ax.fill_between(
                s24["k"],
                s24["accuracy_ci_lower"] * 100,
                s24["accuracy_ci_upper"] * 100,
                color=COLORS["student"],
                alpha=0.14,
            )
        if {"accuracy_ci_lower", "accuracy_ci_upper"}.issubset(s25.columns):
            ax.fill_between(
                s25["k"],
                s25["accuracy_ci_lower"] * 100,
                s25["accuracy_ci_upper"] * 100,
                color=COLORS["expert"],
                alpha=0.14,
            )
    ax.axhline(25, color=COLORS["chance"], linestyle="--", linewidth=0.7)
    ax.set_xlabel("Panel size (k)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("d  Panel-size scaling with tie-rate context", loc="left")
    ax2 = ax.twinx()
    ax2.plot(
        s24["k"],
        s24["tie_rate_mean"] * 100,
        "--",
        color=COLORS["student"],
        linewidth=0.9,
        alpha=0.9,
        label="Junior tie rate",
    )
    ax2.plot(
        s25["k"],
        s25["tie_rate_mean"] * 100,
        "--",
        color=COLORS["expert"],
        linewidth=0.9,
        alpha=0.9,
        label="Expert tie rate",
    )
    ax2.set_ylabel("Tie rate (%)")
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2, fontsize=5.3, loc="upper right")

    if out_base is None:
        out_base = OUT_EXT / "ed_fig4"
    return _save(fig, out_base)


def make_ed4_calibration_error_reliability_v3() -> Tuple[Path, Path]:
    return make_ed4_calibration_error_reliability_v2(
        out_base=OUT_EXT / "ed_fig4",
        include_accuracy_ci=True,
    )


def make_ed5_rl_checkpoint_diagnostics_v2(
    out_base: Path | None = None,
    include_ci: bool = False,
) -> Tuple[Path, Path]:
    _set_style()
    rl = load_rl_metrics()
    t3 = pd.read_csv(TABLES_DIR / "T02_FlagshipPerformance.csv")
    t4 = pd.read_csv(TABLES_DIR / "T03_SFTPerformance.csv")

    sft_row = t4.loc[t4["Model"] == "2-Model Ensemble"].iloc[0]
    sft_ens = float(sft_row["Accuracy"]) * 100.0
    sft_n = int(sft_row["N Valid"])
    best_frontier_row = t3.sort_values("Accuracy", ascending=False).iloc[0]
    best_frontier_name = str(best_frontier_row["Model"])
    best_frontier_acc = float(best_frontier_row["Accuracy"]) * 100.0
    best_frontier_n = int(best_frontier_row["N Valid"])
    gpt52_row = t3.loc[t3["Model"].isin(["GPT-5.2 High", "GPT-5.2 Pro"])].iloc[0]
    gpt52 = float(gpt52_row["Accuracy"]) * 100.0
    gpt52_n = int(gpt52_row["N Valid"])
    frontier_mean = float(t3["Accuracy"].mean()) * 100.0
    frontier_acc_values = t3["Accuracy"].to_numpy(dtype=float) * 100.0
    rl_run = float(rl["run_acc"]) * 100.0
    rl_major = float(rl["majority_acc"]) * 100.0
    rl_avg8_article = float(rl["avg8_article_acc"]) * 100.0

    fig = plt.figure(figsize=(10.8, 8.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.28)

    # a) Summary comparison vs frontier and SFT
    ax = fig.add_subplot(gs[0, 0])
    names = ["SFT 2-Model", "RL avg8", "Best frontier", "GPT-5.2 High", "Frontier avg"]
    vals = [sft_ens, rl_run, best_frontier_acc, gpt52, frontier_mean]
    cols = [COLORS["sft"], "#7A7A7A", COLORS["gemini"], "#9C755F", "#4E79A7"]
    x_a = np.arange(len(names))
    bars = ax.bar(x_a, vals, color=cols, edgecolor="white")
    ax.set_xticks(x_a)
    ax.set_xticklabels(names)
    if include_ci:
        rl_run_ci = _binom_ci95(float(rl["run_acc"]), int(rl["run_total"])) * 100.0
        sft_ci = _binom_ci95(sft_ens / 100.0, sft_n) * 100.0
        best_ci = _binom_ci95(best_frontier_acc / 100.0, best_frontier_n) * 100.0
        gpt52_ci = _binom_ci95(gpt52 / 100.0, gpt52_n) * 100.0
        mean_lo, mean_hi = _bootstrap_mean_ci(frontier_acc_values, seed=17)
        lo_a = [sft_ci, rl_run_ci, best_ci, gpt52_ci, max(frontier_mean - mean_lo, 0.0)]
        hi_a = [sft_ci, rl_run_ci, best_ci, gpt52_ci, max(mean_hi - frontier_mean, 0.0)]
        ax.errorbar(
            x_a,
            vals,
            yerr=[lo_a, hi_a],
            fmt="none",
            ecolor="#333333",
            elinewidth=0.85,
            capsize=2.0,
            capthick=0.85,
            zorder=3,
        )
    ax.axhline(25, color=COLORS["chance"], linestyle="--", linewidth=0.8)
    if include_ci:
        upper_a = max(v + h for v, h in zip(vals, hi_a)) + 2.0
        lower_a = min(v - l for v, l in zip(vals, lo_a)) - 2.0
        ax.set_ylim(min(20.0, lower_a), max(65.0, upper_a))
    else:
        ax.set_ylim(20, 65)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("a  RL checkpoint improves over frontier but trails SFT", loc="left")
    ax.tick_params(axis="x", rotation=15)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + 0.6, f"{v:.1f}", ha="center", fontsize=6)

    # b) RL metric definitions (avg8 vs majority)
    ax = fig.add_subplot(gs[0, 1])
    m_names = ["Run-pooled avg8", "Article-mean avg8", "Majority vote"]
    m_vals = [rl_run, rl_avg8_article, rl_major]
    m_ns = [int(rl["run_total"]), int(rl["n_articles"]), int(rl["majority_total"])]
    m_cols = ["#7A7A7A", "#A0A0A0", "#5B8FF9"]
    x = np.arange(3)
    bars = ax.bar(x, m_vals, color=m_cols, edgecolor="white")
    if include_ci:
        ci_run = _binom_ci95(float(rl["run_acc"]), int(rl["run_total"])) * 100.0
        ci_major = _binom_ci95(float(rl["majority_acc"]), int(rl["majority_total"])) * 100.0
        avg8_values = np.asarray(rl.get("article_avg8_values", []), dtype=float) * 100.0
        lo_avg8, hi_avg8 = _bootstrap_mean_ci(avg8_values, seed=23)
        lo_b = [ci_run, max(rl_avg8_article - lo_avg8, 0.0), ci_major]
        hi_b = [ci_run, max(hi_avg8 - rl_avg8_article, 0.0), ci_major]
        ax.errorbar(
            x,
            m_vals,
            yerr=[lo_b, hi_b],
            fmt="none",
            ecolor="#333333",
            elinewidth=0.85,
            capsize=2.0,
            capthick=0.85,
            zorder=3,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(m_names, rotation=20, ha="right")
    ax.set_ylabel("Accuracy (%)")
    # Keep panel-b scale aligned with panel-a for direct visual comparison.
    ax.set_ylim(20, 60)
    ax.set_title("b  RL protocol sensitivity (8-run definitions)", loc="left")
    for i, b in enumerate(bars):
        label_x = b.get_x() + b.get_width() / 2 + 0.05
        ax.text(label_x, m_vals[i] + 0.25, f"{m_vals[i]:.2f}%\n(n={m_ns[i]})", ha="left", fontsize=5.8)
    ax.text(
        0.98,
        0.04,
        f"Majority ties: {int(rl['majority_ties'])}/120",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.7,
        color="#444444",
    )

    # c) RL confusion structure
    ax = fig.add_subplot(gs[1, 0])
    _plot_confusion(
        ax,
        np.array(rl["cm_major"], dtype=float),
        f"c  RL majority confusion (n={int(rl['majority_total'])}, ties={int(rl['majority_ties'])})",
        cmap="Greys",
    )

    # d) Predicted-tier distribution comparison
    ax = fig.add_subplot(gs[1, 1])
    best_frontier_dist_row = best_frontier_row
    ens_row = t4.loc[t4["Model"] == "2-Model Ensemble"].iloc[0]
    dist = np.vstack(
        [
            np.array(rl["run_pred_dist"], dtype=float),
            best_frontier_dist_row[
                [
                    "Pred count (exceptional)",
                    "Pred count (strong)",
                    "Pred count (fair)",
                    "Pred count (limited)",
                ]
            ].to_numpy(dtype=float)
            / float(best_frontier_dist_row["N Valid"]),
            ens_row[
                [
                    "Pred count (exceptional)",
                    "Pred count (strong)",
                    "Pred count (fair)",
                    "Pred count (limited)",
                ]
            ].to_numpy(dtype=float)
            / float(ens_row["N Valid"]),
        ]
    )
    classes = ["RL avg8", "Best frontier", "SFT 2-Model"]
    bottoms = np.zeros(len(classes))
    tier_cols = ["#5B8FF9", "#61DDAA", "#F6BD16", "#E8684A"]
    for i in range(4):
        vals = dist[:, i] * 100.0
        ax.bar(classes, vals, bottom=bottoms, color=tier_cols[i], edgecolor="white", label=LABELS_TITLE[i])
        bottoms += vals
    ax.set_ylim(0, 100)
    ax.set_ylabel("Predicted share (%)")
    ax.set_title("d  RL collapse profile vs best frontier and SFT ensemble", loc="left")
    ax.legend(ncol=2, fontsize=5.8, loc="upper left")

    if out_base is None:
        out_base = OUT_EXT / "ed_fig5"
    return _save(fig, out_base)


def make_ed5_rl_checkpoint_diagnostics_v3() -> Tuple[Path, Path]:
    return make_ed5_rl_checkpoint_diagnostics_v2(
        out_base=OUT_EXT / "ed_fig5",
        include_ci=True,
    )


def make_ed6_temporal_trace_persistence_v1(
    out_base: Path | None = None,
) -> Tuple[Path, Path]:
    _set_style()
    d = load_temporal_trace_comparison_data()
    recent_items: List[Dict[str, object]] = d["recent_items"]  # type: ignore[assignment]
    recent_ensemble: Dict[str, object] = d["recent_ensemble"]  # type: ignore[assignment]
    old_items: List[Dict[str, object]] = d["old_items"]  # type: ignore[assignment]
    old_focus: Dict[str, object] = d["old_focus"]  # type: ignore[assignment]
    references: List[Dict[str, object]] = d["references"]  # type: ignore[assignment]

    recent_by_key = {str(item["key"]): item for item in recent_items}
    old_by_key = {str(item["key"]): item for item in old_items}
    matched_specs = [
        ("ckpt-step-304", "ft:gpt-4.1-nano-2025-04-14:personal:ob-rqcontext-old:DI3q8ijY", "GPT-4.1-nano"),
        ("ckppt-380", "old_qwen30b_checkpoint_178", "Qwen3-30B"),
        ("matched_2_model_combo", "matched_2_model_combo", "Matched\n2-model ensemble"),
    ]
    matched_groups: List[Dict[str, object]] = []
    for recent_key, old_key, label in matched_specs:
        recent_item = recent_by_key.get(recent_key)
        old_item = old_by_key.get(old_key)
        if recent_item is None or old_item is None:
            continue
        matched_groups.append({"label": label, "recent": recent_item, "old": old_item})

    if not matched_groups:
        raise ValueError("ED6 requires matched recent-vs-old groups.")

    benchmark_refs = [ref for ref in references if str(ref["label"]) in {"Frontier average", "Expert majority"}]
    reference_styles = {
        "Frontier average": ("--", 1.2),
        "Expert majority": ("-.", 1.2),
    }

    recent_span = "2020-2025"
    old_span = "2015-2020"
    recent_label = f"Recent training ({recent_span})"
    old_label = f"Older training ({old_span})"
    recent_ensemble_title = f"c  Recent matched ensemble\n({recent_span})"
    old_ensemble_title = f"Older-trace matched ensemble\n({old_span})"

    recent_color = COLORS["sft"]
    old_color = "#D55E00"
    fig = plt.figure(figsize=(11.9, 9.0))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.28)
    x = np.arange(len(matched_groups))
    width = 0.34

    def _plot_grouped_metric(
        ax: plt.Axes,
        metric_key: str,
        title: str,
        ymax: float,
    ) -> None:
        recent_vals = np.array([float(group["recent"][metric_key]) * 100.0 for group in matched_groups])
        old_vals = np.array([float(group["old"][metric_key]) * 100.0 for group in matched_groups])
        if metric_key == "acc":
            recent_yerr = np.array([float(group["recent"]["acc_ci"]) * 100.0 for group in matched_groups])
            old_yerr = np.array([float(group["old"]["acc_ci"]) * 100.0 for group in matched_groups])
        else:
            recent_yerr = np.array(
                [
                    [
                        recent_vals[i] - float(group["recent"]["macro_f1_ci"][0]) * 100.0,
                        float(group["recent"]["macro_f1_ci"][1]) * 100.0 - recent_vals[i],
                    ]
                    for i, group in enumerate(matched_groups)
                ],
                dtype=float,
            ).T
            old_yerr = np.array(
                [
                    [
                        old_vals[i] - float(group["old"]["macro_f1_ci"][0]) * 100.0,
                        float(group["old"]["macro_f1_ci"][1]) * 100.0 - old_vals[i],
                    ]
                    for i, group in enumerate(matched_groups)
                ],
                dtype=float,
            ).T

        recent_bars = ax.bar(
            x - width / 2,
            recent_vals,
            width=width,
            yerr=recent_yerr,
            capsize=2.3,
            color=recent_color,
            edgecolor="white",
            linewidth=0.8,
            label=recent_label,
        )
        old_bars = ax.bar(
            x + width / 2,
            old_vals,
            width=width,
            yerr=old_yerr,
            capsize=2.3,
            color=old_color,
            edgecolor="white",
            linewidth=0.8,
            label=old_label,
        )

        label_transform = mtransforms.blended_transform_factory(ax.transAxes, ax.transData)
        for ref in benchmark_refs:
            linestyle, line_width = reference_styles[str(ref["label"])]
            y_ref = float(ref[metric_key]) * 100.0
            ax.axhline(
                y_ref,
                color=str(ref["color"]),
                linestyle=linestyle,
                linewidth=line_width,
                alpha=1.0,
                zorder=5,
            )
            ax.text(
                1.01,
                y_ref,
                str(ref["label"]),
                transform=label_transform,
                ha="left",
                va="center",
                fontsize=5.2,
                color=str(ref["color"]),
                clip_on=False,
                bbox=dict(boxstyle="round,pad=0.12", facecolor="white", edgecolor="none", alpha=0.9),
            )

        label_dx = width * 0.16
        for bars in [recent_bars, old_bars]:
            for bar in bars:
                ax.text(
                    bar.get_x() + bar.get_width() / 2 + label_dx,
                    bar.get_height() + 1.0,
                    f"{bar.get_height():.1f}",
                    ha="left",
                    va="bottom",
                    fontsize=5.6,
                )

        ax.set_xticks(x)
        ax.set_xticklabels([str(group["label"]) for group in matched_groups])
        ax.set_ylabel("Score (%)")
        ax.set_ylim(0, ymax)
        ax.set_title(title, loc="left")

    ax_acc = fig.add_subplot(gs[0, 0])
    _plot_grouped_metric(ax_acc, "acc", "a  Accuracy", 75)
    ax_acc.legend(
        handles=[
            plt.Rectangle((0, 0), 1, 1, facecolor=recent_color, edgecolor="white", label=recent_label),
            plt.Rectangle((0, 0), 1, 1, facecolor=old_color, edgecolor="white", label=old_label),
        ],
        fontsize=5.2,
        loc="upper left",
        ncol=1,
    )

    ax_f1 = fig.add_subplot(gs[0, 1])
    _plot_grouped_metric(ax_f1, "macro_f1", "b  Macro-F1", 75)

    # c) Confusion comparison for the matched ensemble summary.
    gs_c = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 0], wspace=0.18)
    cm_recent = np.asarray(recent_ensemble["cm"], dtype=float)
    cm_old = np.asarray(old_focus["cm"], dtype=float)
    vmax = float(max(np.nanmax(_row_norm(cm_recent) * 100.0), np.nanmax(_row_norm(cm_old) * 100.0), 40.0))
    c_axes = [fig.add_subplot(gs_c[0, 0]), fig.add_subplot(gs_c[0, 1])]
    for ax, cm, title in [
        (c_axes[0], cm_recent, recent_ensemble_title),
        (c_axes[1], cm_old, old_ensemble_title),
    ]:
        cm_norm = _row_norm(cm) * 100.0
        im = ax.imshow(cm_norm, cmap="YlOrBr", vmin=0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(LABELS_SHORT, rotation=22, ha="right")
        ax.set_yticklabels(LABELS_SHORT if ax is c_axes[0] else [])
        ax.set_xlabel("Predicted tier")
        ax.set_ylabel("True tier" if ax is c_axes[0] else "")
        ax.set_title(title, loc="left")
        for i in range(4):
            for j in range(4):
                val = float(cm_norm[i, j])
                ax.text(j, i, f"{val:.0f}", ha="center", va="center", fontsize=5.5, color="white" if val >= 0.58 * vmax else "#111111")
    cb = fig.colorbar(im, ax=c_axes, fraction=0.035, pad=0.03)
    cb.set_label("Row-normalized share (%)")

    # d) Predicted-tier shift and top-tier inflation diagnostics.
    gs_d = gridspec.GridSpecFromSubplotSpec(1, 2, subplot_spec=gs[1, 1], wspace=0.30)
    ax = fig.add_subplot(gs_d[0, 0])
    dist = np.vstack([np.asarray(recent_ensemble["pred_dist"], dtype=float), np.asarray(old_focus["pred_dist"], dtype=float)]) * 100.0
    names = [f"Recent\nmatched ens.\n({recent_span})", f"Older-trace\nmatched ens.\n({old_span})"]
    bottoms = np.zeros(2)
    tier_cols = ["#0072B2", "#56B4E9", "#E69F00", "#D55E00"]
    for i in range(4):
        vals = dist[:, i]
        ax.bar(names, vals, bottom=bottoms, color=tier_cols[i], edgecolor="white", label=LABELS_TITLE[i])
        bottoms += vals
    ax.set_ylim(0, 100)
    ax.set_ylabel("Predicted share (%)")
    ax.set_title("d  Old training over-predicts exceptional", loc="left")
    ax.legend(fontsize=5.4, loc="upper left", ncol=2)

    ax = fig.add_subplot(gs_d[0, 1])
    metric_labels = ["Exceptional\nprecision", "Fair\nrecall", "Strong→Exceptional"]
    recent_vals = np.array(
        [
            float(recent_ensemble["exceptional_precision"]),
            float(recent_ensemble["recall"][2]),
            float(recent_ensemble["strong_to_exceptional"]),
        ]
    ) * 100.0
    old_vals = np.array(
        [
            float(old_focus["exceptional_precision"]),
            float(old_focus["recall"][2]),
            float(old_focus["strong_to_exceptional"]),
        ]
    ) * 100.0
    xx = np.arange(len(metric_labels))
    ax.bar(xx - width / 2, recent_vals, width=width, color=recent_color, edgecolor="white")
    ax.bar(xx + width / 2, old_vals, width=width, color=old_color, edgecolor="white")
    ax.set_xticks(xx)
    ax.set_xticklabels(metric_labels)
    ax.set_ylabel("Rate (%)")
    ax.set_ylim(0, 80)
    for i, (rv, ov) in enumerate(zip(recent_vals, old_vals)):
        ax.text(i - width / 2, rv + 1.0, f"{rv:.1f}", ha="center", fontsize=5.5)
        ax.text(i + width / 2, ov + 1.0, f"{ov:.1f}", ha="center", fontsize=5.5)

    if out_base is None:
        out_base = OUT_EXT / "ed_fig6"
    return _save(fig, out_base)


def make_ed7_core_rq_short_transfer_v1(
    out_base: Path | None = None,
) -> Tuple[Path, Path]:
    _set_style()
    data = load_core_rq_short_transfer_data()
    items: List[Dict[str, object]] = data["items"]  # type: ignore[assignment]
    if not items:
        raise ValueError("ED7 requires non-empty core_rq_short transfer items.")

    display_names = [
        "GPT-4.1\nBase",
        "GPT-4.1\nFine-tuned",
        "GPT-4.1 nano\nBase",
        "GPT-4.1 nano\nFine-tuned",
    ]
    compact_names = [
        "GPT-4.1\nbase",
        "GPT-4.1\nfine-tuned",
        "GPT-4.1 nano\nbase",
        "GPT-4.1 nano\nfine-tuned",
    ]
    colors = [str(item["color"]) for item in items]
    x = np.arange(len(items))
    chance_acc = float(data["chance_acc_pct"])

    fig = plt.figure(figsize=(12.4, 8.9))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.40, wspace=0.30)

    gs_a = gs[0, :].subgridspec(1, 2, wspace=0.20)
    ax = fig.add_subplot(gs_a[0, 0])
    short_acc = np.array([float(item["short"]["acc"]) * 100.0 for item in items], dtype=float)
    short_acc_ci = np.array([float(item["short"]["acc_ci"]) * 100.0 for item in items], dtype=float)
    context_acc = np.array([float(item["context"]["acc"]) * 100.0 for item in items], dtype=float)
    bars = ax.bar(x, short_acc, color=colors, edgecolor="white", linewidth=0.8)
    ax.errorbar(x, short_acc, yerr=short_acc_ci, fmt="none", ecolor="#333333", elinewidth=0.85, capsize=2.0, zorder=3)
    ax.scatter(x, context_acc, s=34, facecolors="white", edgecolors="#222222", linewidths=0.9, zorder=4)
    ax.axhline(chance_acc, color=COLORS["chance"], linestyle="--", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.set_ylim(20, 62)
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("a  Performance on one-sentence idea statements", loc="left", pad=12)
    for bar, val in zip(bars, short_acc):
        ax.text(
            bar.get_x() + bar.get_width() / 2 + 0.13,
            val + 0.7,
            f"{val:.1f}",
            ha="left",
            fontsize=5.8,
        )

    ax = fig.add_subplot(gs_a[0, 1])
    short_macro = np.array([float(item["short"]["macro_f1"]) for item in items], dtype=float)
    context_macro = np.array([float(item["context"]["macro_f1"]) for item in items], dtype=float)
    macro_yerr = np.array(
        [
            [
                short_macro[i] - float(item["short"]["macro_f1_ci"][0]),
                float(item["short"]["macro_f1_ci"][1]) - short_macro[i],
            ]
            for i, item in enumerate(items)
        ],
        dtype=float,
    ).T
    bars = ax.bar(x, short_macro, color=colors, edgecolor="white", linewidth=0.8)
    ax.errorbar(x, short_macro, yerr=macro_yerr, fmt="none", ecolor="#333333", elinewidth=0.85, capsize=2.0, zorder=3)
    ax.scatter(x, context_macro, s=34, facecolors="white", edgecolors="#222222", linewidths=0.9, zorder=4)
    ax.set_xticks(x)
    ax.set_xticklabels(display_names)
    ax.set_ylim(0.15, 0.66)
    ax.set_ylabel("Macro-F1")
    for bar, val in zip(bars, short_macro):
        ax.text(
            bar.get_x() + bar.get_width() / 2 + 0.13,
            val + 0.012,
            f"{val:.3f}",
            ha="left",
            fontsize=5.6,
        )
    panel_a_handles = [
        Patch(facecolor=colors[1], edgecolor="white"),
        plt.Line2D([0], [0], marker="o", color="#222222", markerfacecolor="white", linewidth=0),
        plt.Line2D([0], [0], color=COLORS["chance"], linestyle="--", linewidth=0.8),
    ]
    ax.legend(
        panel_a_handles,
        ["One-sentence idea statement", "Full idea summary", "25% chance"],
        fontsize=5.4,
        loc="upper left",
        bbox_to_anchor=(0.02, 0.98),
        frameon=False,
        borderpad=0.20,
        handletextpad=0.45,
        labelspacing=0.35,
        handlelength=1.1,
    )

    ax = fig.add_subplot(gs[1, 0])
    recall_delta = np.vstack([np.asarray(item["recall_delta_pp"], dtype=float) for item in items])
    im = ax.imshow(recall_delta, cmap="RdBu_r", vmin=-50, vmax=50, aspect="auto")
    ax.set_xticks(np.arange(len(LABELS_TITLE)))
    ax.set_xticklabels(LABELS_TITLE)
    ax.set_yticks(np.arange(len(items)))
    ax.set_yticklabels(compact_names)
    ax.set_title("b  Which tiers lose recall under one-sentence input", loc="left", pad=12)
    ax.set_xlabel("True tier")
    for i in range(recall_delta.shape[0]):
        for j in range(recall_delta.shape[1]):
            val = recall_delta[i, j]
            ax.text(
                j,
                i,
                f"{val:+.0f}",
                ha="center",
                va="center",
                fontsize=5.5,
                color="white" if abs(val) > 24 else "#111111",
            )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("Recall change (percentage points)", fontsize=6.0)

    ax = fig.add_subplot(gs[1, 1])
    tier_cols = ["#5B8FF9", "#61DDAA", "#F6BD16", "#E8684A"]
    positions: List[float] = []
    xticklabels: List[str] = []
    model_centers: List[float] = []
    for idx, item in enumerate(items):
        base = idx * 2.6
        positions.extend([base, base + 0.82])
        xticklabels.extend(["Full\nsummary", "One-\nsentence"])
        model_centers.append(base + 0.41)
    dist = np.vstack([np.asarray(item["context"]["pred_dist"], dtype=float) * 100.0 for item in items])
    dist_short = np.vstack([np.asarray(item["short"]["pred_dist"], dtype=float) * 100.0 for item in items])
    combined = np.empty((len(positions), 4), dtype=float)
    combined[0::2] = dist
    combined[1::2] = dist_short
    bottoms = np.zeros(len(positions), dtype=float)
    for tier_idx, color in enumerate(tier_cols):
        vals = combined[:, tier_idx]
        ax.bar(positions, vals, bottom=bottoms, width=0.70, color=color, edgecolor="white", label=LABELS_TITLE[tier_idx])
        bottoms += vals
    ax.set_ylim(0, 112)
    ax.set_ylabel("Predicted share (%)")
    ax.set_xticks(positions)
    ax.set_xticklabels(xticklabels)
    ax.set_title("c  Predicted tier mix under the two input formats", loc="left", pad=12)
    transform = mtransforms.blended_transform_factory(ax.transData, ax.transAxes)
    for center, label in zip(model_centers, display_names):
        ax.text(center, -0.23, label.replace("\n", " "), transform=transform, ha="center", va="top", fontsize=5.6)
    for idx in range(1, len(items)):
        ax.axvline(idx * 2.6 - 0.52, color="#DDDDDD", linewidth=0.8)
    ax.legend(
        fontsize=5.3,
        loc="upper center",
        bbox_to_anchor=(0.50, 0.99),
        ncol=2,
        columnspacing=0.9,
        frameon=True,
        facecolor="white",
        edgecolor="#DDDDDD",
        framealpha=0.95,
        borderpad=0.35,
    )
    fig.subplots_adjust(top=0.88, bottom=0.13, left=0.08, right=0.98)

    if out_base is None:
        out_base = OUT_EXT / "ed_fig7"
    return _save(fig, out_base)


def make_supp_table_figures_v2() -> List[Tuple[Path, Path]]:
    _set_style()
    outputs: List[Tuple[Path, Path]] = []

    # ST4-like pairwise detailed table figure
    pairwise = load_pairwise_metrics()
    frontier_label = "Gemini 3.1 Pro"
    rows = []
    for model in ["SFT GPT-4.1", frontier_label, "GPT-5.2 High", "GPT-4.1 Baseline"]:
        m = pairwise[model]["metrics"]  # type: ignore[index]
        d1 = m["per_distance"]["distance_1"]  # type: ignore[index]
        d2 = m["per_distance"]["distance_2"]  # type: ignore[index]
        d3 = m["per_distance"]["distance_3"]  # type: ignore[index]
        rows.append(
            [
                model,
                int(m["total"]),  # type: ignore[index]
                int(m["total_correct"]),  # type: ignore[index]
                f"{float(m['weighted_accuracy']) * 100:.2f}",  # type: ignore[index]
                f"{int(d1['correct'])}/{int(d1['n'])} ({float(d1['accuracy']) * 100:.2f})",  # type: ignore[index]
                f"{int(d2['correct'])}/{int(d2['n'])} ({float(d2['accuracy']) * 100:.2f})",  # type: ignore[index]
                f"{int(d3['correct'])}/{int(d3['n'])} ({float(d3['accuracy']) * 100:.2f})",  # type: ignore[index]
            ]
        )
    fig, ax = plt.subplots(figsize=(11.2, 2.2))
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=[
            "Model",
            "Total",
            "Correct",
            "Weighted Acc (%)",
            "Distance 1",
            "Distance 2",
            "Distance 3",
        ],
        bbox=[0.0, 0.00, 1.0, 0.82],
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.35)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#E6EEF8")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#FAFAFA")
    ax.set_title("Supplementary Table Figure ST4 | Pairwise discrimination by tier distance", fontsize=10, pad=6)
    fig.subplots_adjust(top=0.86, bottom=0.02, left=0.02, right=0.98)
    st4_png, st4_pdf = _save(fig, OUT_SUPP / "SupplementaryTableFigure_ST4_pairwise")
    st4_png.unlink(missing_ok=True)
    outputs.append((st4_png, st4_pdf))

    # ST7 table figure
    st7 = _load_st7_ensemble_table().sort_values("accuracy_pct", ascending=False).reset_index(drop=True)
    st7_rows = []
    table8 = pd.read_csv(TABLES_DIR / "T04_AIvsHumanSummary.csv")
    best_flagship = float(
        table8
        .loc[lambda d: d["Evaluator"].str.startswith("Best Flagship ("), "Accuracy (%)"]
        .iloc[0]
    )
    for _, r in st7.iterrows():
        acc = float(r["accuracy_pct"])
        st7_rows.append(
            [
                str(r["model_1"]),
                str(r["model_2"]),
                f"{acc:.1f}",
                f"{acc - best_flagship:+.1f}",
            ]
        )
    fig, ax = plt.subplots(figsize=(8.8, 2.5))
    ax.axis("off")
    table = ax.table(
        cellText=st7_rows,
        colLabels=["Model 1", "Model 2", "Ensemble Acc (%)", "Delta vs Best Flagship (pp)"],
        bbox=[0.0, 0.00, 1.0, 0.84],
        cellLoc="center",
        colLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7)
    table.scale(1.0, 1.3)
    for (r, c), cell in table.get_celld().items():
        if r == 0:
            cell.set_facecolor("#E6EEF8")
            cell.set_text_props(weight="bold")
        elif r % 2 == 0:
            cell.set_facecolor("#FAFAFA")
    ax.set_title("Supplementary Table Figure ST7 | All pairwise SFT ensemble combinations", fontsize=10, pad=6)
    fig.subplots_adjust(top=0.86, bottom=0.02, left=0.02, right=0.98)
    st7_png, st7_pdf = _save(fig, OUT_SUPP / "SupplementaryTableFigure_ST7_ensembles")
    st7_png.unlink(missing_ok=True)
    outputs.append((st7_png, st7_pdf))

    return outputs


def _human_majority_prediction_distribution(path: Path) -> Tuple[np.ndarray, int]:
    mapping = {"Top": "exceptional", "Top-": "strong", "Good": "fair", "Fair": "limited"}
    idx = {k: i for i, k in enumerate(LABELS)}
    counts = np.zeros(4, dtype=int)
    n_clear = 0
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            votes: Dict[str, int] = {}
            for r in row.get("ratings", []):
                pred = mapping.get(str(r.get("q2_rating", "")).strip())
                if pred is None:
                    continue
                votes[pred] = votes.get(pred, 0) + 1
            if not votes:
                continue
            top = max(votes.values())
            modes = sorted([k for k, v in votes.items() if v == top])
            if len(modes) != 1:
                continue
            pred = modes[0]
            counts[idx[pred]] += 1
            n_clear += 1
    return counts, n_clear


def _normalize_human_tier(raw: object) -> str | None:
    x = str(raw).strip().lower()
    mapping = {
        "top": "exceptional",
        "top-": "strong",
        "good": "fair",
        "fair": "limited",
        "exceptional": "exceptional",
        "strong": "strong",
        "limited": "limited",
    }
    return mapping.get(x)


def _compute_human_confusions(path: Path) -> Dict[str, object]:
    idx = {k: i for i, k in enumerate(LABELS)}
    cm_pooled = np.zeros((4, 4), dtype=int)
    cm_majority = np.zeros((4, 4), dtype=int)

    n_articles_with_votes = 0
    n_majority_clear = 0
    n_majority_ties = 0
    n_pooled_votes = 0

    with path.open("r", encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            gt = _normalize_human_tier(row.get("level", row.get("rank", "")))
            if gt not in idx:
                continue

            votes: List[str] = []
            for rating in row.get("ratings", []):
                pred = _normalize_human_tier(rating.get("q2_rating", ""))
                if pred not in idx:
                    continue
                votes.append(pred)
                cm_pooled[idx[gt], idx[pred]] += 1
                n_pooled_votes += 1

            if not votes:
                continue

            n_articles_with_votes += 1
            counts = Counter(votes)
            top = max(counts.values())
            modes = [k for k, v in counts.items() if v == top]
            if len(modes) != 1:
                n_majority_ties += 1
                continue

            cm_majority[idx[gt], idx[modes[0]]] += 1
            n_majority_clear += 1

    return {
        "cm_pooled": cm_pooled,
        "cm_majority": cm_majority,
        "n_pooled_votes": n_pooled_votes,
        "n_articles_with_votes": n_articles_with_votes,
        "n_majority_clear": n_majority_clear,
        "n_majority_ties": n_majority_ties,
    }


def make_sf8_human_confusions_v2() -> Tuple[Path, Path]:
    _set_style()

    expert_candidates = [
        PROJECT_ROOT / "data" / "human_ratings" / "reproducibility" / "expert_reproducibility.jsonl",
    ]
    student_candidates = [
        PROJECT_ROOT / "data" / "human_ratings" / "reproducibility" / "student_reproducibility_filtered.jsonl",
    ]

    expert_path = next((p for p in expert_candidates if p.exists()), None)
    student_path = next((p for p in student_candidates if p.exists()), None)
    if expert_path is None or student_path is None:
        raise FileNotFoundError("Could not locate expert/student human rating sources for SF8.")

    expert = _compute_human_confusions(expert_path)
    student = _compute_human_confusions(student_path)

    cm_exp_pool = np.asarray(expert["cm_pooled"], dtype=float)
    cm_stu_pool = np.asarray(student["cm_pooled"], dtype=float)
    cm_exp_maj = np.asarray(expert["cm_majority"], dtype=float)
    cm_stu_maj = np.asarray(student["cm_majority"], dtype=float)

    cm_exp_pool_n = _row_norm(cm_exp_pool) * 100.0
    cm_stu_pool_n = _row_norm(cm_stu_pool) * 100.0
    cm_exp_maj_n = _row_norm(cm_exp_maj) * 100.0
    cm_stu_maj_n = _row_norm(cm_stu_maj) * 100.0
    vmax = float(
        max(
            np.nanmax(cm_exp_pool_n),
            np.nanmax(cm_stu_pool_n),
            np.nanmax(cm_exp_maj_n),
            np.nanmax(cm_stu_maj_n),
            45.0,
        )
    )

    fig = plt.figure(figsize=(10.4, 7.6))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.32, wspace=0.22)

    panels = [
        ("a  Expert individual pooled", cm_exp_pool_n, f"n={int(expert['n_pooled_votes'])} ratings"),
        ("b  Junior individual pooled", cm_stu_pool_n, f"n={int(student['n_pooled_votes'])} ratings"),
        (
            "c  Expert majority (ties excluded)",
            cm_exp_maj_n,
            f"n={int(expert['n_majority_clear'])} articles, ties={int(expert['n_majority_ties'])}",
        ),
        (
            "d  Junior majority (ties excluded)",
            cm_stu_maj_n,
            f"n={int(student['n_majority_clear'])} articles, ties={int(student['n_majority_ties'])}",
        ),
    ]

    axes: List[plt.Axes] = []
    im = None
    for i, (title, cmn, note) in enumerate(panels):
        ax = fig.add_subplot(gs[i // 2, i % 2])
        axes.append(ax)
        im = ax.imshow(cmn, cmap="YlGnBu", vmin=0.0, vmax=vmax, aspect="auto")
        ax.set_xticks(range(4))
        ax.set_yticks(range(4))
        ax.set_xticklabels(LABELS_SHORT, fontsize=6.2, rotation=25, ha="right")
        ax.set_yticklabels(LABELS_SHORT, fontsize=6.2)
        ax.set_xlabel("Predicted tier")
        ax.set_ylabel("True tier")
        ax.set_title(title, loc="left")
        for rr in range(4):
            for cc in range(4):
                val = float(cmn[rr, cc])
                txt_col = "white" if val >= 0.60 * vmax else "#111111"
                ax.text(cc, rr, f"{val:.0f}", ha="center", va="center", fontsize=6.1, color=txt_col)
        ax.text(
            0.99,
            0.03,
            note,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=5.7,
            color="#444444",
            bbox=dict(boxstyle="round,pad=0.20", facecolor="white", edgecolor="#DDDDDD"),
        )

    if im is not None:
        cb = fig.colorbar(im, ax=axes, fraction=0.018, pad=0.02)
        cb.set_label("Row-normalized share (%)")

    fig.text(
        0.5,
        0.012,
        "Majority panels use strict clear-majority voting (ties excluded), aligned with Table 8.",
        ha="center",
        va="bottom",
        fontsize=6.0,
        color="#555555",
    )
    return _save(fig, OUT_SUPP / "si_fig6")


def make_supplementary_figures_core_v2() -> List[Tuple[Path, Path]]:
    outputs: List[Tuple[Path, Path]] = []
    _set_style()

    # SF2: Prediction distribution comparison
    t3 = pd.read_csv(TABLES_DIR / "T02_FlagshipPerformance.csv")
    t4 = pd.read_csv(TABLES_DIR / "T03_SFTPerformance.csv")

    pred_cols = [
        "Pred count (exceptional)",
        "Pred count (strong)",
        "Pred count (fair)",
        "Pred count (limited)",
    ]
    # Frontier average (11 flagship models)
    frontier = t3[pred_cols].to_numpy(dtype=float)
    frontier_share = (frontier / t3["N Valid"].to_numpy(dtype=float)[:, None]).mean(axis=0)

    # Chat average (3 chat models)
    chat_profiles: List[np.ndarray] = []
    chat_models = ["gpt-5.2", "kimi-k2-0905-preview", "deepseek-chat"]
    with CHAT_PATH.open("r", encoding="utf-8") as f:
        rows = [json.loads(x) for x in f]
    for m in chat_models:
        c = np.zeros(4, dtype=float)
        n_valid = 0
        for row in rows:
            model_out = row.get("val_outcome", {}).get("rq_with_context", {}).get(m, {})
            if not isinstance(model_out, dict):
                continue
            pred = _extract_prediction_from_model_output(model_out)
            if pred in LABELS:
                c[LABELS.index(pred)] += 1
                n_valid += 1
        if n_valid > 0:
            chat_profiles.append(c / n_valid)
    chat_share = np.mean(np.vstack(chat_profiles), axis=0) if chat_profiles else np.zeros(4, dtype=float)

    # SFT single average (4 SFT models)
    sft_rows = t4[t4["Model"].str.startswith("SFT (")].copy()
    sft = sft_rows[pred_cols].to_numpy(dtype=float)
    sft_share = (sft / sft_rows["N Valid"].to_numpy(dtype=float)[:, None]).mean(axis=0)

    # SFT ensemble
    ens_row = t4[t4["Model"] == "2-Model Ensemble"].iloc[0]
    ens_share = ens_row[pred_cols].to_numpy(dtype=float) / float(ens_row["N Valid"])

    # Human majority distributions (clear majority only)
    expert_counts, n_exp = _human_majority_prediction_distribution(
        PROJECT_ROOT / "data" / "human_ratings" / "reproducibility" / "expert_reproducibility.jsonl"
    )
    student_counts, n_stu = _human_majority_prediction_distribution(
        PROJECT_ROOT / "data" / "human_ratings" / "reproducibility" / "student_reproducibility_filtered.jsonl"
    )
    expert_share = expert_counts / max(n_exp, 1)
    student_share = student_counts / max(n_stu, 1)

    classes = [
        "Frontier avg\n(11)",
        "Chat avg\n(3)",
        "SFT single avg\n(4)",
        "SFT ensemble",
        "Expert majority",
        "Junior majority",
    ]
    shares = np.vstack([frontier_share, chat_share, sft_share, ens_share, expert_share, student_share])
    share_sums = shares.sum(axis=1, keepdims=True)
    shares = np.divide(shares, share_sums, out=np.zeros_like(shares), where=share_sums > 0)

    fig = plt.figure(figsize=(10.4, 6.8))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.24)

    ax = fig.add_subplot(gs[0, :])
    bottoms = np.zeros(len(classes))
    tier_cols = ["#5B8FF9", "#61DDAA", "#F6BD16", "#E8684A"]
    for i, tier in enumerate(LABELS):
        vals = shares[:, i] * 100
        ax.bar(classes, vals, bottom=bottoms, color=tier_cols[i], edgecolor="white", label=LABELS_TITLE[i])
        bottoms += vals
    ax.set_ylim(0, 100)
    ax.set_ylabel("Predicted share (%)")
    ax.set_title("a  Prediction distribution across evaluator classes", loc="left")
    ax.legend(ncol=4, fontsize=6, loc="upper center")

    ax = fig.add_subplot(gs[1, 0])
    # Deviation from balanced 25%
    dev = (shares * 100) - 25.0
    x = np.arange(len(classes))
    w = 0.18
    for i in range(4):
        ax.bar(x + (i - 1.5) * w, dev[:, i], width=w, color=tier_cols[i], label=LABELS_SHORT[i])
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=15, ha="right")
    ax.set_ylabel("Delta vs balanced 25% (pp)")
    ax.set_title("b  Tier-bias profile by class", loc="left")

    ax = fig.add_subplot(gs[1, 1])
    # Distribution concentration entropy (higher means more balanced)
    entropy = -np.sum(np.clip(shares, 1e-9, 1.0) * np.log2(np.clip(shares, 1e-9, 1.0)), axis=1)
    max_h = np.log2(4)
    ax.barh(np.arange(len(classes)), entropy / max_h * 100, color="#4E79A7", edgecolor="white")
    ax.set_yticks(np.arange(len(classes)))
    ax.set_yticklabels(classes, fontsize=6)
    ax.invert_yaxis()
    ax.set_xlabel("Normalized entropy (%)")
    ax.set_title("c  Distribution concentration (higher = less collapse)", loc="left")

    outputs.append(_save(fig, OUT_SUPP / "si_fig1"))

    # SF3: Expert individual accuracy distribution
    s2 = pd.read_csv(TABLES_DIR / "T10_IndividualExpertAccuracy_Anonymized.csv")
    vals = s2["accuracy"].to_numpy(dtype=float) * 100
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.6))
    ax = axes[0]
    ax.hist(vals, bins=12, color=COLORS["expert"], edgecolor="white", alpha=0.85)
    ax.axvline(vals.mean(), color="#333333", linestyle="--", linewidth=1.0)
    ax.axvline(25, color=COLORS["chance"], linestyle=":", linewidth=1.0)
    ax.set_xlabel("Individual expert accuracy (%)")
    ax.set_ylabel("Count")
    ax.set_title("a  Histogram", loc="left")
    ax.text(0.98, 0.95, f"Mean={vals.mean():.1f}\nSD={vals.std(ddof=1):.1f}\nN={len(vals)}", transform=ax.transAxes, ha="right", va="top", fontsize=6)
    ax = axes[1]
    srt = np.sort(vals)
    ecdf = np.arange(1, len(srt) + 1) / len(srt)
    ax.plot(srt, ecdf, color=COLORS["expert"], linewidth=1.4)
    ax.axvline(25, color=COLORS["chance"], linestyle=":", linewidth=1.0)
    ax.set_xlabel("Individual expert accuracy (%)")
    ax.set_ylabel("ECDF")
    ax.set_title("b  Empirical CDF", loc="left")
    outputs.append(_save(fig, OUT_SUPP / "si_fig2"))

    # SF4: Junior Monte Carlo curve
    s3 = pd.read_csv(TABLES_DIR / "T11_MonteCarloMatchedPanel.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10.4, 3.8))
    ax = axes[0]
    ax.plot(s3["k"], s3["mean_accuracy"] * 100, "-o", color=COLORS["student"], markersize=3.2, linewidth=1.2)
    ax.fill_between(s3["k"], s3["ci_lower"] * 100, s3["ci_upper"] * 100, color=COLORS["student"], alpha=0.16)
    ax.axhline(25, color=COLORS["chance"], linestyle=":", linewidth=1.0)
    ax.set_xlabel("Panel size k")
    ax.set_ylabel("Majority-vote accuracy (%)")
    ax.set_title("a  Accuracy vs panel size", loc="left")
    ax = axes[1]
    # Marginal gain per +1 rater
    k = s3["k"].to_numpy(dtype=float)
    y = s3["mean_accuracy"].to_numpy(dtype=float) * 100
    dy = np.diff(y) / np.diff(k)
    ax.plot(k[1:], dy, "-o", color="#2F4B7C", markersize=3.2, linewidth=1.2)
    ax.axhline(0, color="#444444", linewidth=0.8)
    ax.set_xlabel("Panel size k")
    ax.set_ylabel("Marginal gain (pp per +1 rater)")
    ax.set_title("b  Diminishing returns profile", loc="left")
    outputs.append(_save(fig, OUT_SUPP / "si_fig3"))

    # SF5: Flagship collapse-metric landscape (non-overlap with ED3 confusion atlas)
    stats02 = json.loads((STATS_DIR / "S01_FlagshipStats.json").read_text(encoding="utf-8"))
    rank = stats02.get("ranking_by_accuracy", [])
    perf = stats02.get("model_performance", {})
    labels = [str(r["model"]) for r in rank]
    acc = np.array([float(perf[m]["accuracy"]) * 100.0 for m in labels], dtype=float)
    pred_dist = np.array(
        [
            [
                float(perf[m]["prediction_distribution"].get("exceptional", 0.0)),
                float(perf[m]["prediction_distribution"].get("strong", 0.0)),
                float(perf[m]["prediction_distribution"].get("fair", 0.0)),
                float(perf[m]["prediction_distribution"].get("limited", 0.0)),
            ]
            for m in labels
        ],
        dtype=float,
    )
    pred_norm = pred_dist / np.maximum(pred_dist.sum(axis=1, keepdims=True), 1.0)
    middle_share = (pred_norm[:, 1] + pred_norm[:, 2]) * 100.0
    entropy = -np.sum(np.clip(pred_norm, 1e-9, 1.0) * np.log2(np.clip(pred_norm, 1e-9, 1.0)), axis=1)
    entropy_norm = entropy / np.log2(4.0) * 100.0
    recall_mat = np.array(
        [
            [
                float(perf[m]["per_tier_accuracy"].get("exceptional", 0.0)) * 100.0,
                float(perf[m]["per_tier_accuracy"].get("strong", 0.0)) * 100.0,
                float(perf[m]["per_tier_accuracy"].get("fair", 0.0)) * 100.0,
                float(perf[m]["per_tier_accuracy"].get("limited", 0.0)) * 100.0,
            ]
            for m in labels
        ],
        dtype=float,
    )

    fig = plt.figure(figsize=(10.8, 7.2))
    gs = gridspec.GridSpec(2, 2, figure=fig, hspace=0.34, wspace=0.30)

    # a) Middle-tier concentration by model
    ax = fig.add_subplot(gs[0, 0])
    y = np.arange(len(labels))
    ax.barh(y, middle_share, color="#4E79A7", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=5.2)
    ax.invert_yaxis()
    ax.set_xlabel("Predictions in strong+fair tiers (%)")
    ax.set_title("a  Middle-tier concentration across 11 flagship models", loc="left")
    for i, v in enumerate(middle_share):
        ax.text(v + 0.8, i, f"{v:.1f}", va="center", fontsize=5.0)
    ax.set_xlim(0, min(100.0, max(middle_share) + 12.0))

    # b) Tier-recall heatmap (all 11 models)
    ax = fig.add_subplot(gs[0, 1])
    im = ax.imshow(recall_mat, cmap="YlGnBu", vmin=0, vmax=100, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_xticklabels(LABELS_TITLE, rotation=30, ha="right")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=5.2)
    ax.set_title("b  Tier-recall structure by model", loc="left")
    for i in range(recall_mat.shape[0]):
        for j in range(recall_mat.shape[1]):
            v = recall_mat[i, j]
            ax.text(j, i, f"{v:.0f}", ha="center", va="center", fontsize=4.6, color="white" if v > 55 else "black")
    cb = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.02)
    cb.set_label("Recall (%)")

    # c) Tradeoff: middle-tier concentration vs overall accuracy
    ax = fig.add_subplot(gs[1, 0])
    ax.scatter(middle_share, acc, s=40, color="#F28E2B", edgecolors="white")
    for x0, y0, name in zip(middle_share, acc, labels):
        ax.text(x0 + 0.6, y0 + 0.25, name, fontsize=5.0)
    ax.set_xlabel("Middle-tier concentration (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("c  Concentration-accuracy tradeoff", loc="left")

    # d) Distribution entropy ranking
    ax = fig.add_subplot(gs[1, 1])
    ord_idx = np.argsort(entropy_norm)[::-1]
    y = np.arange(len(labels))
    ax.barh(y, entropy_norm[ord_idx], color="#59A14F", edgecolor="white")
    ax.set_yticks(y)
    ax.set_yticklabels([labels[i] for i in ord_idx], fontsize=5.2)
    ax.invert_yaxis()
    ax.set_xlabel("Prediction entropy (normalized, %)")
    ax.set_title("d  Distribution balance ranking (higher = less collapse)", loc="left")

    outputs.append(_save(fig, OUT_SUPP / "si_fig4"))

    # SF6: AI-human complementarity diagnostics on expert-comparable set.
    stats07 = json.loads((STATS_DIR / "S05_AIvsHumanStats.json").read_text(encoding="utf-8"))
    comp = stats07.get("complementarity", {})
    both = int(comp.get("both_correct", 0))
    ai_only = int(comp.get("ai_only_correct", 0))
    human_only = int(comp.get("human_only_correct", 0))
    neither = int(comp.get("neither_correct", 0))
    n_total = int(comp.get("total", both + ai_only + human_only + neither))
    if n_total <= 0:
        raise ValueError("Invalid complementarity totals in 07_ai_vs_human.json")

    ai_acc = (both + ai_only) / n_total * 100.0
    human_acc = (both + human_only) / n_total * 100.0
    # Upper bound if either AI or expert is correct on an article.
    oracle_acc = (both + ai_only + human_only) / n_total * 100.0
    err_total = ai_only + human_only + neither
    shared_err_pct = float(comp.get("shared_error_pct", (neither / err_total * 100.0) if err_total > 0 else 0.0))
    comp_err_pct = float(
        comp.get("complementary_error_pct", ((ai_only + human_only) / err_total * 100.0) if err_total > 0 else 0.0)
    )

    fig, axes = plt.subplots(1, 2, figsize=(10.6, 4.0))

    # a) 4-way overlap decomposition
    ax = axes[0]
    cats = ["Both correct", "AI only", "Human only", "Neither"]
    vals = np.array([both, ai_only, human_only, neither], dtype=float)
    colors = ["#59A14F", COLORS["sft"], COLORS["expert"], "#9D9D9D"]
    bars = ax.bar(np.arange(4), vals, color=colors, edgecolor="white")
    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(cats, rotation=15, ha="right")
    ax.set_ylabel("Articles (n)")
    ax.set_title("a  Error-overlap decomposition (SFT 2-model vs expert majority)", loc="left")
    ax.set_ylim(0, max(vals) * 1.26)
    for b, v in zip(bars, vals):
        ax.text(
            b.get_x() + b.get_width() / 2,
            b.get_height() + 0.6,
            f"{int(v)}\n({v / n_total * 100:.1f}%)",
            ha="center",
            va="bottom",
            fontsize=6.2,
        )
    ax.text(0.98, 0.96, f"Comparable set n={n_total}", transform=ax.transAxes, ha="right", va="top", fontsize=6.3)

    # b) Accuracy comparison + complementarity upper bound
    ax = axes[1]
    labels = ["SFT 2-model", "Expert majority", "Oracle upper bound\n(AI or expert correct)"]
    acc_vals = [ai_acc, human_acc, oracle_acc]
    bar_colors = [COLORS["sft"], COLORS["expert"], "#59A14F"]
    bars = ax.bar(np.arange(3), acc_vals, color=bar_colors, edgecolor="white")
    ax.set_xticks(np.arange(3))
    ax.set_xticklabels(labels, rotation=10, ha="right")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("b  Complementarity ceiling on expert-comparable articles", loc="left")
    ax.axhline(25, color=COLORS["chance"], linestyle="--", linewidth=1.0)
    ax.set_ylim(20, max(acc_vals) + 12)
    for b, v in zip(bars, acc_vals):
        ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.6, f"{v:.1f}%", ha="center", va="bottom", fontsize=6.2)
    ax.text(
        0.98,
        0.93,
        f"Among error-involved cases (n={err_total}):\nComplementary {comp_err_pct:.1f}%  |  Shared {shared_err_pct:.1f}%",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=6.0,
        bbox=dict(boxstyle="round,pad=0.25", facecolor="white", edgecolor="#cccccc"),
    )

    outputs.append(_save(fig, OUT_SUPP / "si_fig5"))

    outputs.append(make_sf8_human_confusions_v2())

    return outputs


def write_design_notes(generated_ext: List[Tuple[Path, Path]], generated_supp: List[Tuple[Path, Path]]) -> Path:
    OUT_NOTES.mkdir(parents=True, exist_ok=True)
    p = OUT_NOTES / "extended_and_supplementary_figures_notes.md"
    lines = [
        "# Extended and Supplementary Figure Notes",
        "",
        "Design principles applied:",
        "- Avoid direct duplication of headline panels from main figures.",
        "- Push detail to dense multi-panel diagnostics (full confusion atlases, matched base-vs-SFT maps, ensemble landscape ranking, calibration/error/reliability internals).",
        "- Trace pairwise figure directly to `data/pairwise/*` trial files and `metrics.json` summaries.",
        "- Add ED6 as a temporal persistence diagnostic using the package-local recent SFT predictions, both verified older-trace checkpoints, and the old 2-model ensemble.",
        "- Add ED7 as a compressed-input transfer diagnostic using the bundled one-sentence idea-statement prediction file and package-local fuller-input reference predictions.",
        "",
        "Generated Extended Data figures:",
    ]
    for png, pdf in generated_ext:
        lines.append(f"- {png.name}, {pdf.name}")
    lines += ["", "Generated Supplementary table-figures:"]
    for png, pdf in generated_supp:
        lines.append(f"- {png.name}, {pdf.name}")
    lines += [
        "",
        "Supplementary generation planning:",
        "- Generated core SI figures tied to current refined SI headings (SF2-SF6) using canonical reanalysis tables/statistics.",
        "- Added SF6 (AI-human error complementarity diagnostics) to directly support Results/Discussion claims on complementary errors.",
        "- Moved cross-model prompt-sensitivity landscape to ED1 (availability-limited; mixed prompt protocols).",
        "- Added SF6 to show expert/junior confusion structure (individual pooled + strict majority panels).",
        "- Prompt protocol caveat: Gemini 3.1 uses canonical `gemini_3_1_pro_standalone.jsonl` merged into the conservative 11-model frontier cohort; ties/unresolved outputs are counted incorrect in fixed-n reporting.",
        "- SF1 (training-loss curves) not generated because raw step-wise training logs are not present in workspace.",
    ]
    lines += [
        "",
        "Pairwise design rationale:",
        "- Uses intrinsic pairwise task with a six-pair-type heatmap and paired discordance decomposition in ED2, while Main Fig. 5 now carries the headline overall-accuracy and hard-boundary panels.",
        "- Matches manuscript method/results claims: overall weighted accuracy, concentration of gains at the fair-strong and strong-exceptional boundaries, and raw exact paired significance against the plotted comparator set.",
        "- Supersedes earlier ad-hoc ED11/ST11 pairwise artifacts generated from evaluator-vs-evaluator significance tables.",
        "",
        "RL update (new data ingested):",
        "- Added ED5 from `data/predictions/rl_predictions.jsonl` (run-pooled avg8, majority-vote robustness, confusion structure, and distribution profile).",
        "- Current workspace contains one RL checkpoint key; architecture-specific mapping for ED Table 3 still requires metadata confirmation if two RL architectures are to be reported separately.",
        "- ED6 packages the historical comparison as recent singles + recent ensemble + two old singles + the old 2-model ensemble, while keeping the old-trace result in a supporting role.",
        "",
        "Non-overlap examples:",
        "- Main Fig. 2 shows selected collapse exemplars; ED3 now shows all 11 flagship confusions + matched base-vs-SFT confusion maps + pairwise ensemble landscape ranking.",
        "- Main Fig. 5 now combines confidence separation, selective prediction, overall pairwise accuracy, and hard-boundary pairwise generalization; ED4 emphasizes calibration/error internals and ED2 keeps the detailed pairwise decomposition.",
    ]
    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def main() -> None:
    OUT_EXT.mkdir(parents=True, exist_ok=True)
    OUT_SUPP.mkdir(parents=True, exist_ok=True)

    generated_ext = [
        make_sf7_prompt_sensitivity_all_models_v2(
            out_base=OUT_EXT / "ed_fig1",
        ),
        make_ed2_pairwise_intrinsic_v2(),
        make_ed3_confusion_ensemble_v2(),
        make_ed4_calibration_error_reliability_v2(),
        make_ed5_rl_checkpoint_diagnostics_v2(),
        make_ed6_temporal_trace_persistence_v1(),
        make_ed7_core_rq_short_transfer_v1(),
    ]
    generated_supp = []
    generated_supp.extend(make_supp_table_figures_v2())
    generated_supp.extend(make_supplementary_figures_core_v2())
    note_path = write_design_notes(generated_ext, generated_supp)

    print("Extended/Supplementary figures generated:")
    for png, pdf in generated_ext + generated_supp:
        print(f"  - {png}")
        print(f"  - {pdf}")
    print(f"Notes: {note_path}")


if __name__ == "__main__":
    main()
