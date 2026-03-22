#!/usr/bin/env python3
"""Build economics extension figures and supporting release artifacts.

Outputs:
- reproduced/figures/main/Figure7.(png|pdf)
- reproduced/figures/extended_data/ed_fig8.(png|pdf)
- reproduced/figures/supplementary/si_fig7.(png|pdf)
- data/statistics/S16_EconomicsExtensionStats.json
- data/statistics/S17_PooledFieldExtensionStats.json
- data/statistics/S18_CrossFieldTransferStats.json
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from scipy.stats import binomtest

from figure_style_policy import apply_title_policy, panel_title


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "README.md").exists() and (candidate / "scripts").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError("Could not locate package root.")


ROOT = find_project_root()
FIG7_OUT_DIR = ROOT / "reproduced" / "figures" / "main"
ED8_OUT_DIR = ROOT / "reproduced" / "figures" / "extended_data"
SF7_OUT_DIR = ROOT / "reproduced" / "figures" / "supplementary"
STATS_DIR = ROOT / "data" / "statistics"

ECON_PATH = ROOT / "data" / "predictions" / "economics_predictions.jsonl"
POOLED_2FIELD_PATH = ROOT / "data" / "predictions" / "pooled_management_economics_predictions.jsonl"

LABELS = ["exceptional", "strong", "fair", "limited"]
LABELS_SHORT = ["Exc", "Str", "Fair", "Ltd"]
LABELS_TITLE = ["Exceptional", "Strong", "Fair", "Limited"]
PALETTE = {
    "base": "#D55E00",
    "sft": "#0072B2",
    "sft_alt": "#56B4E9",
    "chance": "#999999",
    "frontier_ref": "#009E73",
    "tier_exceptional": "#0072B2",
    "tier_strong": "#56B4E9",
    "tier_fair": "#E69F00",
    "tier_limited": "#D55E00",
    "mixed_30b": "#0072B2",
    "mixed_nano": "#56B4E9",
    "link": "#CFCFCF",
}

TIER_COLORS = [
    PALETTE["tier_exceptional"],
    PALETTE["tier_strong"],
    PALETTE["tier_fair"],
    PALETTE["tier_limited"],
]

FIELD_CONFIG = {
    "economics": {
        "path": ECON_PATH,
        "display": "Economics",
        "models": [
            {
                "display": "Qwen3-30B",
                "base": "qwen3_30b_base",
                "sft": "qwen3_30b_sft_economics",
            },
            {
                "display": "Qwen3-4B",
                "base": "qwen3_4b_base",
                "sft": "qwen3_4b_sft_economics",
            },
            {
                "display": "GPT-4.1-nano",
                "base": "gpt_4_1_nano_base",
                "sft": "gpt_4_1_nano_sft_economics",
            },
        ],
    },
}

POOLED_2FIELD_MODELS = {
    "Pooled Qwen3-30B": "qwen3_30b_sft_management_economics",
    "Pooled GPT-4.1-nano": "gpt_4_1_nano_sft_management_economics",
}

POOLED_FIELD_ORDER = ["management", "economics"]
POOLED_FIELD_DISPLAY = {"management": "Management", "economics": "Economics"}

FRONTIER_MODELS = [
    "gpt_4_1_base",
    "grok_4_1_fast",
    "qwen_3_5_plus",
    "gemini_3_1_pro",
]


def normalize_label(value: object) -> str | None:
    if value is None:
        return None
    raw = str(value).strip().lower().strip("*")
    mapping = {
        "exc": "exceptional",
        "exceptional": "exceptional",
        "str": "strong",
        "strong": "strong",
        "fair": "fair",
        "limited": "limited",
        "ltd": "limited",
    }
    return mapping.get(raw)


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.6,
            "axes.titlesize": 7.5,
            "axes.labelsize": 6.8,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "axes.linewidth": 0.7,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def load_jsonl(path: Path) -> List[dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def logsumexp(vals: Iterable[float]) -> float:
    vals = list(vals)
    max_val = max(vals)
    if not math.isfinite(max_val):
        return float("-inf")
    return max_val + math.log(sum(math.exp(v - max_val) for v in vals))


def probs_from_sparse_logp(logp: Dict[str, object]) -> Dict[str, float]:
    parsed: Dict[str, float] = {}
    for label, raw in logp.items():
        try:
            parsed[label] = float(raw)
        except (TypeError, ValueError):
            continue
    if not parsed:
        uniform = 1.0 / len(LABELS)
        return {label: uniform for label in LABELS}
    lse = logsumexp(parsed.values())
    if not math.isfinite(lse):
        uniform = 1.0 / len(LABELS)
        return {label: uniform for label in LABELS}
    out = {label: 0.0 for label in LABELS}
    for label, value in parsed.items():
        out[label] = math.exp(value - lse)
    total = sum(out.values())
    if total <= 0:
        uniform = 1.0 / len(LABELS)
        return {label: uniform for label in LABELS}
    return {label: value / total for label, value in out.items()}


def get_model_probs(row: dict, model_key: str) -> Dict[str, float]:
    info = row["val_outcome"]["rq_with_context"][model_key]
    logp = info.get("logp")
    if not isinstance(logp, dict):
        raise ValueError(f"No logp payload for {model_key}")
    return probs_from_sparse_logp(logp)


def argmax_label(probs: Dict[str, float]) -> str:
    best_label = LABELS[0]
    best_score = probs.get(best_label, float("-inf"))
    for label in LABELS[1:]:
        score = probs.get(label, float("-inf"))
        if score > best_score:
            best_label = label
            best_score = score
    return best_label


def compute_metrics(rows: List[dict], model_key: str) -> Dict[str, object]:
    return metrics_from_records(compute_records(rows, model_key))


def compute_records(rows: List[dict], model_key: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for row in rows:
        probs = get_model_probs(row, model_key)
        pred = argmax_label(probs)
        records.append(
            {
                "truth": normalize_label(row["rank"]) or str(row["rank"]),
                "pred": pred,
                "confidence": probs.get(pred, 0.0),
            }
        )
    return records


def frontier_primary_label(model_out: Dict[str, object]) -> str | None:
    pred = normalize_label(model_out.get("response_text"))
    if pred in LABELS:
        return pred

    vote_predictions = model_out.get("vote_predictions")
    if isinstance(vote_predictions, list):
        for raw in vote_predictions:
            pred = normalize_label(raw)
            if pred in LABELS:
                return pred

    vote_counts = model_out.get("vote_counts")
    if isinstance(vote_counts, dict) and vote_counts:
        top = max(vote_counts.values())
        winners = sorted({normalize_label(k) for k, v in vote_counts.items() if v == top and normalize_label(k) in LABELS})
        if len(winners) == 1:
            return winners[0]

    logp = model_out.get("logp")
    if isinstance(logp, dict) and logp:
        best_label = None
        best_score = float("-inf")
        for label, score in logp.items():
            cand = normalize_label(label)
            if cand not in LABELS:
                continue
            try:
                value = float(score)
            except (TypeError, ValueError):
                continue
            if value > best_score:
                best_score = value
                best_label = cand
        if best_label in LABELS:
            return best_label

    return None


def compute_frontier_records(rows: List[dict], model_key: str) -> List[Dict[str, object]]:
    records: List[Dict[str, object]] = []
    for row in rows:
        outcomes = row.get("val_outcome", {}).get("rq_with_context", {})
        model_out = outcomes.get(model_key)
        if not isinstance(model_out, dict):
            continue
        pred = frontier_primary_label(model_out)
        if pred not in LABELS:
            continue
        records.append(
            {
                "truth": normalize_label(row["rank"]) or str(row["rank"]),
                "pred": pred,
                "confidence": 1.0,
            }
        )
    return records


def best_frontier_metrics(rows: List[dict]) -> Dict[str, Dict[str, object]]:
    candidates: List[Tuple[str, Dict[str, object]]] = []
    total = len(rows)
    for model_key in FRONTIER_MODELS:
        records = compute_frontier_records(rows, model_key)
        if len(records) != total:
            continue
        candidates.append((model_key, metrics_from_records(records)))
    if not candidates:
        return {}
    best_acc_model, best_acc_metrics = max(candidates, key=lambda item: (item[1]["accuracy"], item[1]["macro_f1"]))
    best_f1_model, best_f1_metrics = max(candidates, key=lambda item: (item[1]["macro_f1"], item[1]["accuracy"]))
    return {
        "accuracy": {"model": best_acc_model, **best_acc_metrics},
        "macro_f1": {"model": best_f1_model, **best_f1_metrics},
    }


def metrics_from_records(records: List[Dict[str, object]]) -> Dict[str, object]:
    tp = Counter()
    fp = Counter()
    fn = Counter()
    pred_counts = Counter()
    n = len(records)
    correct = 0
    for rec in records:
        pred = str(rec["pred"])
        truth = str(rec["truth"])
        pred_counts[pred] += 1
        if pred == truth:
            correct += 1
            tp[truth] += 1
        else:
            fp[pred] += 1
            fn[truth] += 1
    f1s = []
    for label in LABELS:
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        f1s.append(f1)
    return {
        "accuracy": correct / n if n else 0.0,
        "macro_f1": sum(f1s) / len(LABELS) if f1s else 0.0,
        "pred_counts": dict(pred_counts),
        "n": n,
    }


def compute_confusion(rows: List[dict], model_key: str) -> np.ndarray:
    return compute_confusion_from_records(compute_records(rows, model_key))


def compute_confusion_from_records(records: List[Dict[str, object]]) -> np.ndarray:
    matrix = np.zeros((4, 4), dtype=float)
    label_to_idx = {label: idx for idx, label in enumerate(LABELS)}
    for rec in records:
        matrix[label_to_idx[str(rec["truth"])], label_to_idx[str(rec["pred"])]] += 1.0
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def ensemble_metrics(rows: List[dict], model_keys: Tuple[str, str]) -> Dict[str, object]:
    tp = Counter()
    fp = Counter()
    fn = Counter()
    pred_counts = Counter()
    confidence_records: List[Tuple[float, int]] = []
    correct = 0
    n = 0
    for row in rows:
        avg = {label: 0.0 for label in LABELS}
        for key in model_keys:
            probs = get_model_probs(row, key)
            for label in LABELS:
                avg[label] += probs.get(label, 0.0)
        for label in LABELS:
            avg[label] /= len(model_keys)
        pred = argmax_label(avg)
        conf = avg[pred]
        truth = row["rank"]
        confidence_records.append((conf, int(pred == truth)))
        pred_counts[pred] += 1
        n += 1
        if pred == truth:
            correct += 1
            tp[truth] += 1
        else:
            fp[pred] += 1
            fn[truth] += 1
    f1s = []
    for label in LABELS:
        precision = tp[label] / (tp[label] + fp[label]) if tp[label] + fp[label] else 0.0
        recall = tp[label] / (tp[label] + fn[label]) if tp[label] + fn[label] else 0.0
        f1s.append(2 * precision * recall / (precision + recall) if precision + recall else 0.0)
    confidence_records.sort(key=lambda item: item[0], reverse=True)
    return {
        "accuracy": correct / n if n else 0.0,
        "macro_f1": sum(f1s) / len(LABELS) if f1s else 0.0,
        "pred_counts": dict(pred_counts),
        "records": confidence_records,
    }


def compute_coverage_curve(records: List[Tuple[float, int]]) -> Tuple[np.ndarray, np.ndarray]:
    coverages = []
    accuracies = []
    total = len(records)
    correct_running = 0
    for idx, (_, is_correct) in enumerate(records, start=1):
        correct_running += int(is_correct)
        coverages.append(idx / total * 100.0)
        accuracies.append(correct_running / idx * 100.0)
    return np.asarray(coverages), np.asarray(accuracies)


def compute_selective_records(records: List[Dict[str, object]]) -> List[Tuple[float, int]]:
    selective = [(float(rec["confidence"]), int(rec["pred"] == rec["truth"])) for rec in records]
    selective.sort(key=lambda item: item[0], reverse=True)
    return selective


def bootstrap_metric_cis(records: List[Dict[str, object]], n_boot: int = 1200, seed: int = 0) -> Dict[str, Tuple[float, float]]:
    if not records:
        return {"accuracy": (0.0, 0.0), "macro_f1": (0.0, 0.0)}
    rng = np.random.default_rng(seed)
    n = len(records)
    accuracy_samples = []
    f1_samples = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, size=n)
        sample = [records[int(i)] for i in idx]
        metrics = metrics_from_records(sample)
        accuracy_samples.append(metrics["accuracy"])
        f1_samples.append(metrics["macro_f1"])
    acc_lo, acc_hi = np.percentile(accuracy_samples, [2.5, 97.5])
    f1_lo, f1_hi = np.percentile(f1_samples, [2.5, 97.5])
    return {
        "accuracy": (float(acc_lo), float(acc_hi)),
        "macro_f1": (float(f1_lo), float(f1_hi)),
    }


def pick_best_pair(rows: List[dict], model_keys: List[str]) -> Tuple[Tuple[str, str], Dict[str, object]]:
    best_pair: Tuple[str, str] | None = None
    best_metrics: Dict[str, object] | None = None
    for idx in range(len(model_keys)):
        for jdx in range(idx + 1, len(model_keys)):
            pair = (model_keys[idx], model_keys[jdx])
            metrics = ensemble_metrics(rows, pair)
            if best_metrics is None or (
                metrics["accuracy"],
                metrics["macro_f1"],
            ) > (
                best_metrics["accuracy"],
                best_metrics["macro_f1"],
            ):
                best_pair = pair
                best_metrics = metrics
    if best_pair is None or best_metrics is None:
        raise RuntimeError("Could not identify best 2-model ensemble.")
    return best_pair, best_metrics


def save_package_figure(fig: plt.Figure, out_dir: Path, basename: str, family: str) -> Tuple[Path, Path]:
    apply_title_policy(fig, family=family)
    out_dir.mkdir(parents=True, exist_ok=True)
    png = out_dir / f"{basename}.png"
    pdf = out_dir / f"{basename}.pdf"
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight")
    plt.close(fig)
    return png, pdf


def write_json(path: Path, obj: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_csv(path: Path, fieldnames: List[str], rows: List[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def pct(value: float) -> float:
    return round(value * 100.0, 1)


def pct_ci(ci: Tuple[float, float]) -> Tuple[float, float]:
    return (round(ci[0] * 100.0, 1), round(ci[1] * 100.0, 1))


def plot_base_vs_sft(ax: plt.Axes, rows: List[dict], model_specs: List[Dict[str, str]], frontier_refs: Dict[str, Dict[str, object]] | None = None) -> None:
    y_center = np.arange(len(model_specs), dtype=float) * 1.85
    y_acc = y_center - 0.25
    y_f1 = y_center + 0.25

    for idx, spec in enumerate(model_specs):
        base_records = compute_records(rows, spec["base"])
        sft_records = compute_records(rows, spec["sft"])
        base_metrics = metrics_from_records(base_records)
        sft_metrics = metrics_from_records(sft_records)
        base_cis = bootstrap_metric_cis(base_records, seed=100 + idx)
        sft_cis = bootstrap_metric_cis(sft_records, seed=200 + idx)

        base_acc = base_metrics["accuracy"] * 100.0
        sft_acc = sft_metrics["accuracy"] * 100.0
        base_f1 = base_metrics["macro_f1"] * 100.0
        sft_f1 = sft_metrics["macro_f1"] * 100.0

        for y_val, base_val, sft_val in [
            (y_acc[idx], base_acc, sft_acc),
            (y_f1[idx], base_f1, sft_f1),
        ]:
            ax.plot([base_val, sft_val], [y_val, y_val], color=PALETTE["link"], linewidth=0.9, zorder=1)

        ax.errorbar(
            base_acc,
            y_acc[idx],
            xerr=[[base_acc - base_cis["accuracy"][0] * 100.0], [base_cis["accuracy"][1] * 100.0 - base_acc]],
            fmt="o",
            color=PALETTE["base"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=4.1,
            elinewidth=0.9,
            capsize=2.0,
            zorder=3,
        )
        ax.errorbar(
            sft_acc,
            y_acc[idx],
            xerr=[[sft_acc - sft_cis["accuracy"][0] * 100.0], [sft_cis["accuracy"][1] * 100.0 - sft_acc]],
            fmt="o",
            color=PALETTE["sft"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=4.1,
            elinewidth=0.9,
            capsize=2.0,
            zorder=4,
        )
        ax.errorbar(
            base_f1,
            y_f1[idx],
            xerr=[[base_f1 - base_cis["macro_f1"][0] * 100.0], [base_cis["macro_f1"][1] * 100.0 - base_f1]],
            fmt="s",
            color=PALETTE["base"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=3.7,
            elinewidth=0.9,
            capsize=2.0,
            zorder=3,
        )
        ax.errorbar(
            sft_f1,
            y_f1[idx],
            xerr=[[sft_f1 - sft_cis["macro_f1"][0] * 100.0], [sft_cis["macro_f1"][1] * 100.0 - sft_f1]],
            fmt="s",
            color=PALETTE["sft"],
            markeredgecolor="white",
            markeredgewidth=0.45,
            markersize=3.7,
            elinewidth=0.9,
            capsize=2.0,
            zorder=4,
        )

        label_box = {"boxstyle": "round,pad=0.08", "facecolor": "white", "edgecolor": "none", "alpha": 0.92}
        ax.annotate(
            f"{base_acc:.1f}",
            xy=(base_acc, y_acc[idx]),
            xytext=(-9, 4),
            textcoords="offset points",
            ha="right",
            va="bottom",
            fontsize=4.7,
            color=PALETTE["base"],
            bbox=label_box,
        )
        ax.annotate(
            f"{sft_acc:.1f}",
            xy=(sft_acc, y_acc[idx]),
            xytext=(9, 4),
            textcoords="offset points",
            ha="left",
            va="bottom",
            fontsize=4.7,
            color=PALETTE["sft"],
            bbox=label_box,
        )
        ax.annotate(
            f"{base_f1:.1f}",
            xy=(base_f1, y_f1[idx]),
            xytext=(-9, -4),
            textcoords="offset points",
            ha="right",
            va="top",
            fontsize=4.7,
            color=PALETTE["base"],
            bbox=label_box,
        )
        ax.annotate(
            f"{sft_f1:.1f}",
            xy=(sft_f1, y_f1[idx]),
            xytext=(9, -4),
            textcoords="offset points",
            ha="left",
            va="top",
            fontsize=4.7,
            color=PALETTE["sft"],
            bbox=label_box,
        )

    if frontier_refs:
        frontier_acc = frontier_refs["accuracy"]["accuracy"] * 100.0
        frontier_f1 = frontier_refs["macro_f1"]["macro_f1"] * 100.0
        ax.axvline(
            frontier_acc,
            color=PALETTE["frontier_ref"],
            linestyle=(0, (4, 2)),
            linewidth=0.95,
            alpha=0.95,
            zorder=0,
        )
        ax.axvline(
            frontier_f1,
            color=PALETTE["frontier_ref"],
            linestyle=(0, (1.2, 1.2)),
            linewidth=0.95,
            alpha=0.95,
            zorder=0,
        )

    ax.axvline(25.0, color=PALETTE["chance"], linestyle="--", linewidth=0.8)
    ax.text(25.8, y_center[-1] + 0.70, "Chance (25%)", fontsize=5.0, color=PALETTE["chance"], ha="left", va="bottom")
    ax.set_yticks(y_center)
    ax.set_yticklabels([spec["display"] for spec in model_specs])
    ax.invert_yaxis()
    ax.set_xlim(0, 86)
    ax.set_ylim(y_center[-1] + 1.02, -1.02)
    ax.set_xticks(np.arange(0, 81, 10))
    ax.set_xlabel("Score (%)")
    ax.grid(axis="x", color="#E6E6E6", linewidth=0.5)
    ax.set_axisbelow(True)


def plot_confusion(ax: plt.Axes, matrix: np.ndarray, model_display: str, accuracy_pct: float) -> None:
    matrix_pct = matrix * 100.0
    im = ax.imshow(matrix_pct, cmap="YlOrRd", vmin=0, vmax=80, aspect="auto")
    ax.set_xticks(range(4))
    ax.set_yticks(range(4))
    ax.set_xticklabels(LABELS_SHORT, rotation=0)
    ax.set_yticklabels(LABELS_SHORT)
    ax.set_xlabel("Predicted tier")
    ax.set_ylabel("True tier")
    for i in range(4):
        for j in range(4):
            value = matrix_pct[i, j]
            color = "white" if value >= 45 else "#1A1A1A"
            ax.text(j, i, f"{value:.0f}", ha="center", va="center", fontsize=5.0, color=color)
    ax.text(0.5, -0.18, f"{model_display}  |  accuracy {accuracy_pct:.1f}%", transform=ax.transAxes, ha="center", va="top", fontsize=5.4)
    return im


def plot_coverage(ax: plt.Axes, records: List[Dict[str, object]], model_label: str, overall_acc_pct: float) -> None:
    coverages, accuracies = compute_coverage_curve(compute_selective_records(records))
    ax.plot(coverages, accuracies, color=PALETTE["sft"], linewidth=1.6)
    ax.axhline(overall_acc_pct, color=PALETTE["chance"], linestyle="--", linewidth=0.8)
    ax.scatter([coverages[-1]], [accuracies[-1]], color=PALETTE["sft"], s=13, zorder=3)
    ax.set_xlim(0, 100)
    ax.set_ylim(55, 101)
    ax.set_xlabel("Coverage (%)")
    ax.set_ylabel("Accuracy (%)")
    ax.grid(color="#E6E6E6", linewidth=0.5)
    ax.set_axisbelow(True)
    ax.text(0.98, 0.06, f"100% -> {overall_acc_pct:.1f}%", transform=ax.transAxes, ha="right", va="bottom", fontsize=5.2, color=PALETTE["chance"])
    ax.text(0.98, 0.01, model_label, transform=ax.transAxes, ha="right", va="bottom", fontsize=5.0, color="#666666")


def build_figure7() -> Tuple[Path, Path]:
    set_style()
    fig = plt.figure(figsize=(7.2, 3.2))
    gs = gridspec.GridSpec(
        1,
        4,
        figure=fig,
        left=0.08,
        right=0.96,
        top=0.84,
        bottom=0.20,
        width_ratios=[1.15, 1.0, 0.09, 1.03],
        wspace=0.48,
    )

    panel_specs = [
        ("a", "Economics: base-to-SFT scores", "economics", 0, 0),
        ("b", "Economics: best-model confusion", "economics", 0, 1),
        ("c", "Economics: selective prediction", "economics", 0, 3),
    ]

    field_cache: Dict[str, Dict[str, object]] = {}
    needed_fields = {fk for _, _, fk, _, _ in panel_specs}
    for field_key, cfg in FIELD_CONFIG.items():
        if field_key not in needed_fields:
            continue
        rows = load_jsonl(cfg["path"])
        single_metrics = []
        for spec in cfg["models"]:
            records = compute_records(rows, spec["sft"])
            metrics = metrics_from_records(records)
            single_metrics.append((spec["display"], spec["sft"], records, metrics))
        best_single_display, best_single_key, best_single_records, best_single_summary = max(
            single_metrics,
            key=lambda item: (item[3]["accuracy"], item[3]["macro_f1"]),
        )
        field_cache[field_key] = {
            "rows": rows,
            "best_single_display": best_single_display,
            "best_single_key": best_single_key,
            "best_single_records": best_single_records,
            "best_single_summary": best_single_summary,
            "frontier_refs": best_frontier_metrics(rows),
        }

    for letter, subtitle, field_key, row_idx, col_idx in panel_specs:
        ax = fig.add_subplot(gs[row_idx, col_idx])
        panel_title(ax, letter, subtitle, family="main")
        cfg = FIELD_CONFIG[field_key]
        rows = field_cache[field_key]["rows"]
        if col_idx == 0:
            plot_base_vs_sft(ax, rows, cfg["models"], field_cache[field_key]["frontier_refs"])
        elif col_idx == 1:
            best_single_key = field_cache[field_key]["best_single_key"]
            best_single_display = field_cache[field_key]["best_single_display"]
            best_single_summary = field_cache[field_key]["best_single_summary"]
            im = plot_confusion(
                ax,
                compute_confusion(rows, best_single_key),
                best_single_display,
                best_single_summary["accuracy"] * 100.0,
            )
            cax = fig.add_subplot(gs[row_idx, 2])
            pos = cax.get_position()
            cax.set_position([pos.x0 - 0.010, pos.y0, pos.width * 0.62, pos.height])
            cbar = fig.colorbar(im, cax=cax)
            cbar.set_label("Recall (%)", fontsize=6.0)
            cbar.ax.tick_params(labelsize=5.4)
            cbar.ax.yaxis.set_label_position("left")
            cbar.ax.yaxis.tick_left()
        else:
            best_single_display = field_cache[field_key]["best_single_display"]
            best_single_records = field_cache[field_key]["best_single_records"]
            best_single_summary = field_cache[field_key]["best_single_summary"]
            plot_coverage(
                ax,
                best_single_records,
                best_single_display,
                best_single_summary["accuracy"] * 100.0,
            )

    all_handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["base"], markeredgecolor=PALETTE["base"], markersize=5, label="Base"),
        Line2D([0], [0], marker="o", color="none", markerfacecolor=PALETTE["sft"], markeredgecolor=PALETTE["sft"], markersize=5, label="In-domain SFT"),
        Line2D([0], [0], marker="o", linestyle="", markersize=4.6, color="#666666", markerfacecolor="white", label="Accuracy"),
        Line2D([0], [0], marker="s", linestyle="", markersize=4.2, color="#666666", markerfacecolor="white", label="Macro-F1"),
        Line2D([0], [0], color=PALETTE["frontier_ref"], linestyle=(0, (4, 2)), linewidth=1.0, label="Best frontier acc. (Gemini 3.1 Pro)"),
        Line2D([0], [0], color=PALETTE["frontier_ref"], linestyle=(0, (1.2, 1.2)), linewidth=1.0, label="Best frontier F1 (Gemini 3.1 Pro)"),
    ]
    fig.legend(
        handles=all_handles,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.02),
        ncol=6,
        fontsize=5.4,
        handletextpad=0.3,
        columnspacing=0.8,
        handlelength=1.6,
        frameon=False,
    )

    return save_package_figure(fig, FIG7_OUT_DIR, "Figure7", family="main")


def build_extended_data_figure8() -> Tuple[Path, Path]:
    set_style()
    rows = load_jsonl(POOLED_2FIELD_PATH)

    field_order = POOLED_FIELD_ORDER
    field_display = POOLED_FIELD_DISPLAY

    metrics_by_model: Dict[str, Dict[str, Dict[str, object]]] = {}
    ci_by_model: Dict[str, Dict[str, Dict[str, Tuple[float, float]]]] = {}
    for display, key in POOLED_2FIELD_MODELS.items():
        metrics_by_model[display] = {}
        ci_by_model[display] = {}
        for field in field_order:
            subset = [row for row in rows if row["subject"] == field]
            records = compute_records(subset, key)
            metrics_by_model[display][field] = metrics_from_records(records)
            ci_by_model[display][field] = bootstrap_metric_cis(records, seed=500 + len(field))

    fig = plt.figure(figsize=(7.2, 4.8))
    gs = gridspec.GridSpec(
        2,
        2,
        figure=fig,
        left=0.08,
        right=0.96,
        top=0.84,
        bottom=0.11,
        height_ratios=[1.0, 1.05],
        wspace=0.38,
        hspace=0.60,
    )

    # Panel a: accuracy
    ax_a = fig.add_subplot(gs[0, 0])
    panel_title(ax_a, "a", "Pooled-training accuracy across test fields", family="edsi")
    x = np.arange(len(field_order))
    width = 0.32
    for idx, (display, color_key) in enumerate([("Pooled Qwen3-30B", "mixed_30b"), ("Pooled GPT-4.1-nano", "mixed_nano")]):
        vals = [metrics_by_model[display][field]["accuracy"] * 100.0 for field in field_order]
        yerr = np.array(
            [
                [vals[j] - ci_by_model[display][field]["accuracy"][0] * 100.0 for j, field in enumerate(field_order)],
                [ci_by_model[display][field]["accuracy"][1] * 100.0 - vals[j] for j, field in enumerate(field_order)],
            ]
        )
        xpos = x + (idx - 0.5) * width
        ax_a.bar(xpos, vals, width=width, color=PALETTE[color_key], edgecolor="white", linewidth=0.5)
        ax_a.errorbar(xpos, vals, yerr=yerr, fmt="none", ecolor="#444444", elinewidth=0.8, capsize=2.0)
        for xpos_i, val in zip(xpos, vals):
            ax_a.text(xpos_i, val + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=5.1)
    ax_a.axhline(25.0, color=PALETTE["chance"], linestyle="--", linewidth=0.8)
    ax_a.set_xticks(x)
    ax_a.set_xticklabels([field_display[field] for field in field_order])
    ax_a.set_ylabel("Accuracy (%)")
    ax_a.set_ylim(0, 78)
    ax_a.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    ax_a.set_axisbelow(True)

    # Panel b: macro-F1
    ax_b = fig.add_subplot(gs[0, 1])
    panel_title(ax_b, "b", "Pooled-training macro-F1 across test fields", family="edsi")
    for idx, (display, color_key) in enumerate([("Pooled Qwen3-30B", "mixed_30b"), ("Pooled GPT-4.1-nano", "mixed_nano")]):
        vals = [metrics_by_model[display][field]["macro_f1"] * 100.0 for field in field_order]
        yerr = np.array(
            [
                [vals[j] - ci_by_model[display][field]["macro_f1"][0] * 100.0 for j, field in enumerate(field_order)],
                [ci_by_model[display][field]["macro_f1"][1] * 100.0 - vals[j] for j, field in enumerate(field_order)],
            ]
        )
        xpos = x + (idx - 0.5) * width
        ax_b.bar(xpos, vals, width=width, color=PALETTE[color_key], edgecolor="white", linewidth=0.5)
        ax_b.errorbar(xpos, vals, yerr=yerr, fmt="none", ecolor="#444444", elinewidth=0.8, capsize=2.0)
        for xpos_i, val in zip(xpos, vals):
            ax_b.text(xpos_i, val + 1.2, f"{val:.1f}", ha="center", va="bottom", fontsize=5.1)
    ax_b.axhline(25.0, color=PALETTE["chance"], linestyle="--", linewidth=0.8)
    ax_b.set_xticks(x)
    ax_b.set_xticklabels([field_display[field] for field in field_order])
    ax_b.set_ylabel("Macro-F1 (%)")
    ax_b.set_ylim(0, 78)
    ax_b.grid(axis="y", color="#E6E6E6", linewidth=0.5)
    ax_b.set_axisbelow(True)

    fig.legend(
        handles=[
            Patch(facecolor=PALETTE["mixed_30b"], edgecolor="white", label="Pooled Qwen3-30B"),
            Patch(facecolor=PALETTE["mixed_nano"], edgecolor="white", label="Pooled GPT-4.1-nano"),
        ],
        loc="upper center",
        bbox_to_anchor=(0.50, 0.905),
        ncol=2,
        fontsize=5.6,
        handlelength=1.2,
        columnspacing=1.0,
    )

    # Panel c: prediction composition for pooled 30B
    ax_c = fig.add_subplot(gs[1, 0])
    panel_title(ax_c, "c", "Pooled Qwen3-30B predicted-tier composition", family="edsi")
    bottom = np.zeros(len(field_order))
    for label, color in zip(LABELS, TIER_COLORS):
        vals = []
        for field in field_order:
            pred_counts = metrics_by_model["Pooled Qwen3-30B"][field]["pred_counts"]
            total = sum(pred_counts.values())
            vals.append(pred_counts.get(label, 0) / total if total else 0.0)
        ax_c.bar(x, vals, bottom=bottom, color=color, edgecolor="white", linewidth=0.5, width=0.6)
        bottom += np.asarray(vals)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels([field_display[field] for field in field_order])
    ax_c.set_ylim(0, 1.0)
    ax_c.set_ylabel("Prediction share")

    # Panel d: prediction composition for pooled nano
    ax_d = fig.add_subplot(gs[1, 1])
    panel_title(ax_d, "d", "Pooled GPT-4.1-nano predicted-tier composition", family="edsi")
    bottom = np.zeros(len(field_order))
    for label, color in zip(LABELS, TIER_COLORS):
        vals = []
        for field in field_order:
            pred_counts = metrics_by_model["Pooled GPT-4.1-nano"][field]["pred_counts"]
            total = sum(pred_counts.values())
            vals.append(pred_counts.get(label, 0) / total if total else 0.0)
        ax_d.bar(x, vals, bottom=bottom, color=color, edgecolor="white", linewidth=0.5, width=0.6)
        bottom += np.asarray(vals)
    ax_d.set_xticks(x)
    ax_d.set_xticklabels([field_display[field] for field in field_order])
    ax_d.set_ylim(0, 1.0)
    ax_d.set_ylabel("Prediction share")

    fig.legend(
        handles=[Patch(facecolor=color, edgecolor="white", label=label) for color, label in zip(TIER_COLORS, LABELS_TITLE)],
        loc="upper center",
        bbox_to_anchor=(0.50, 0.47),
        ncol=4,
        fontsize=5.2,
        handlelength=1.0,
        columnspacing=0.8,
    )

    return save_package_figure(fig, ED8_OUT_DIR, "ed_fig8", family="edsi")


MGMT_SFT_KEY = "gpt_4_1_sft_management"
GPT41_BASE_KEY = "gpt_4_1_base"


def build_supplementary_figure7() -> Tuple[Path, Path]:
    """Cross-field transfer: management-trained SFT GPT-4.1 on economics benchmark."""
    set_style()
    rows = load_jsonl(ECON_PATH)

    # Compute records for management SFT and base on economics
    mgmt_sft_records = compute_records(rows, MGMT_SFT_KEY)
    base_records = compute_records(rows, GPT41_BASE_KEY)
    mgmt_sft_metrics = metrics_from_records(mgmt_sft_records)
    base_metrics = metrics_from_records(base_records)

    # Also get best in-domain economics SFT for reference
    econ_sft_key = FIELD_CONFIG["economics"]["models"][0]["sft"]  # Qwen3-30B (best)
    econ_sft_records = compute_records(rows, econ_sft_key)
    econ_sft_metrics = metrics_from_records(econ_sft_records)

    fig = plt.figure(figsize=(7.2, 3.2))
    gs = gridspec.GridSpec(
        1, 3, figure=fig,
        left=0.08, right=0.96, top=0.84, bottom=0.20,
        width_ratios=[1.0, 1.0, 1.0], wspace=0.40,
    )

    # --- Panel a: accuracy comparison bars ---
    ax_a = fig.add_subplot(gs[0, 0])
    panel_title(ax_a, "a", "Cross-field transfer accuracy", family="edsi")

    evaluators = [
        ("GPT-4.1\nbase", base_metrics["accuracy"] * 100, PALETTE["base"]),
        ("Mgmt SFT\nGPT-4.1", mgmt_sft_metrics["accuracy"] * 100, "#E69F00"),
        ("Econ SFT\nQwen3-30B", econ_sft_metrics["accuracy"] * 100, PALETTE["sft"]),
    ]
    x_pos = range(len(evaluators))
    bars = ax_a.bar(
        x_pos,
        [e[1] for e in evaluators],
        color=[e[2] for e in evaluators],
        width=0.6, edgecolor="white", linewidth=0.5,
    )
    ax_a.set_xticks(list(x_pos))
    ax_a.set_xticklabels([e[0] for e in evaluators], fontsize=6.0)
    ax_a.set_ylabel("Accuracy (%)", fontsize=7)
    ax_a.set_ylim(0, 80)
    ax_a.axhline(25, color=PALETTE["chance"], linestyle="--", linewidth=0.8, label="Chance (25%)")
    for bar_obj, (_, val, _) in zip(bars, evaluators):
        ax_a.text(bar_obj.get_x() + bar_obj.get_width() / 2, val + 1.5,
                  f"{val:.1f}%", ha="center", va="bottom", fontsize=6.0, fontweight="bold")
    ax_a.tick_params(axis="both", labelsize=6)

    # Add significance annotation
    ax_a.annotate("", xy=(0, 52), xytext=(1, 52),
                  arrowprops=dict(arrowstyle="-", color="black", lw=0.8))
    ax_a.text(0.5, 53, "+14.0 pp\n$p < 10^{-8}$", ha="center", fontsize=5.0, style="italic")

    # --- Panel b: confusion matrix for management SFT on economics ---
    ax_b = fig.add_subplot(gs[0, 1])
    panel_title(ax_b, "b", "Mgmt SFT on economics: confusion", family="edsi")
    cm = compute_confusion_from_records(mgmt_sft_records)
    im = plot_confusion(ax_b, cm, "Mgmt SFT GPT-4.1", mgmt_sft_metrics["accuracy"] * 100)

    # --- Panel c: per-tier recall comparison ---
    ax_c = fig.add_subplot(gs[0, 2])
    panel_title(ax_c, "c", "Per-tier recall comparison", family="edsi")

    per_tier_base = {}
    per_tier_mgmt = {}
    per_tier_econ = {}
    for t in LABELS:
        base_t = [r for r in base_records if r["truth"] == t]
        mgmt_t = [r for r in mgmt_sft_records if r["truth"] == t]
        econ_t = [r for r in econ_sft_records if r["truth"] == t]
        per_tier_base[t] = sum(1 for r in base_t if r["pred"] == t) / max(len(base_t), 1) * 100
        per_tier_mgmt[t] = sum(1 for r in mgmt_t if r["pred"] == t) / max(len(mgmt_t), 1) * 100
        per_tier_econ[t] = sum(1 for r in econ_t if r["pred"] == t) / max(len(econ_t), 1) * 100

    x = np.arange(len(LABELS))
    w = 0.25
    ax_c.bar(x - w, [per_tier_base[t] for t in LABELS], w, label="GPT-4.1 base", color=PALETTE["base"], edgecolor="white", linewidth=0.5)
    ax_c.bar(x, [per_tier_mgmt[t] for t in LABELS], w, label="Mgmt SFT GPT-4.1", color="#E69F00", edgecolor="white", linewidth=0.5)
    ax_c.bar(x + w, [per_tier_econ[t] for t in LABELS], w, label="Econ SFT Qwen3-30B", color=PALETTE["sft"], edgecolor="white", linewidth=0.5)
    ax_c.set_xticks(x)
    ax_c.set_xticklabels(LABELS_SHORT, fontsize=6)
    ax_c.set_ylabel("Recall (%)", fontsize=7)
    ax_c.set_ylim(0, 100)
    ax_c.tick_params(axis="both", labelsize=6)
    ax_c.legend(fontsize=5.2, loc="upper right", frameon=False)

    return save_package_figure(fig, SF7_OUT_DIR, "si_fig7", family="edsi")


def write_economics_extension_release_artifacts() -> None:
    econ_rows = load_jsonl(ECON_PATH)
    pooled_rows = load_jsonl(POOLED_2FIELD_PATH)

    # Figure 7 + SF7 summary table
    econ_table_rows: List[dict] = []
    econ_stats_models: Dict[str, dict] = {}
    for spec in FIELD_CONFIG["economics"]["models"]:
        for role, model_key in [("base", spec["base"]), ("sft_economics", spec["sft"])]:
            records = compute_records(econ_rows, model_key)
            metrics = metrics_from_records(records)
            cis = bootstrap_metric_cis(records, seed=1000 + len(econ_table_rows))
            econ_table_rows.append(
                {
                    "model_family": spec["display"],
                    "evaluator": model_key,
                    "role": role,
                    "n": len(records),
                    "accuracy_pct": pct(metrics["accuracy"]),
                    "accuracy_ci_lower_pct": pct_ci(cis["accuracy"])[0],
                    "accuracy_ci_upper_pct": pct_ci(cis["accuracy"])[1],
                    "macro_f1_pct": pct(metrics["macro_f1"]),
                    "macro_f1_ci_lower_pct": pct_ci(cis["macro_f1"])[0],
                    "macro_f1_ci_upper_pct": pct_ci(cis["macro_f1"])[1],
                }
            )
            econ_stats_models[model_key] = {
                "display": spec["display"],
                "role": role,
                "n": len(records),
                "accuracy_pct": pct(metrics["accuracy"]),
                "accuracy_ci_pct": list(pct_ci(cis["accuracy"])),
                "macro_f1_pct": pct(metrics["macro_f1"]),
                "macro_f1_ci_pct": list(pct_ci(cis["macro_f1"])),
            }

    frontier_model_rows = []
    for model_key in FRONTIER_MODELS:
        records = compute_frontier_records(econ_rows, model_key)
        metrics = metrics_from_records(records)
        frontier_model_rows.append(
            {
                "model_family": model_key,
                "evaluator": model_key,
                "role": "frontier_reference",
                "n": len(records),
                "accuracy_pct": pct(metrics["accuracy"]),
                "accuracy_ci_lower_pct": "",
                "accuracy_ci_upper_pct": "",
                "macro_f1_pct": pct(metrics["macro_f1"]),
                "macro_f1_ci_lower_pct": "",
                "macro_f1_ci_upper_pct": "",
            }
        )
        econ_stats_models[model_key] = {
            "display": model_key,
            "role": "frontier_reference",
            "n": len(records),
            "accuracy_pct": pct(metrics["accuracy"]),
            "macro_f1_pct": pct(metrics["macro_f1"]),
        }

    base_records = compute_records(econ_rows, GPT41_BASE_KEY)
    mgmt_sft_records = compute_records(econ_rows, MGMT_SFT_KEY)
    econ_sft_key = FIELD_CONFIG["economics"]["models"][0]["sft"]
    econ_sft_records = compute_records(econ_rows, econ_sft_key)
    transfer_gap_pct = pct(metrics_from_records(mgmt_sft_records)["accuracy"] - metrics_from_records(base_records)["accuracy"])
    transfer_p = binomtest(
        sum(1 for rec in mgmt_sft_records if rec["truth"] == rec["pred"]),
        n=len(mgmt_sft_records),
        p=0.25,
        alternative="greater",
    ).pvalue

    per_tier_recall = {}
    for label, records in {
        "gpt_4_1_base": base_records,
        "gpt_4_1_sft_management": mgmt_sft_records,
        econ_sft_key: econ_sft_records,
    }.items():
        label_map = {}
        for truth_label in LABELS:
            subset = [rec for rec in records if rec["truth"] == truth_label]
            label_map[truth_label] = round(
                100.0 * sum(1 for rec in subset if rec["pred"] == truth_label) / max(len(subset), 1),
                1,
            )
        per_tier_recall[label] = label_map

    write_json(
        STATS_DIR / "S16_EconomicsExtensionStats.json",
        {
            "figure": "Figure7",
            "benchmark_n": len(econ_rows),
            "models": econ_stats_models,
            "frontier_reference_best": {
                "accuracy": best_frontier_metrics(econ_rows)["accuracy"],
                "macro_f1": best_frontier_metrics(econ_rows)["macro_f1"],
            },
            "best_single_economics_model": {
                "evaluator": econ_sft_key,
                "accuracy_pct": pct(metrics_from_records(econ_sft_records)["accuracy"]),
                "macro_f1_pct": pct(metrics_from_records(econ_sft_records)["macro_f1"]),
                "confusion_matrix_pct": compute_confusion_from_records(econ_sft_records).round(4).tolist(),
            },
            "selective_prediction_best_model": {
                "evaluator": econ_sft_key,
                "coverage_pct": np.round(compute_coverage_curve(compute_selective_records(econ_sft_records))[0], 2).tolist(),
                "accuracy_pct": np.round(compute_coverage_curve(compute_selective_records(econ_sft_records))[1], 2).tolist(),
            },
        },
    )

    write_json(
        STATS_DIR / "S18_CrossFieldTransferStats.json",
        {
            "figure": "SupplementaryFigure7",
            "benchmark_n": len(econ_rows),
            "gpt_4_1_base": {
                "accuracy_pct": pct(metrics_from_records(base_records)["accuracy"]),
                "macro_f1_pct": pct(metrics_from_records(base_records)["macro_f1"]),
            },
            "gpt_4_1_sft_management": {
                "accuracy_pct": pct(metrics_from_records(mgmt_sft_records)["accuracy"]),
                "macro_f1_pct": pct(metrics_from_records(mgmt_sft_records)["macro_f1"]),
                "exact_binomial_p_vs_chance": transfer_p,
                "confusion_matrix_pct": compute_confusion_from_records(mgmt_sft_records).round(4).tolist(),
            },
            "best_economics_sft": {
                "evaluator": econ_sft_key,
                "accuracy_pct": pct(metrics_from_records(econ_sft_records)["accuracy"]),
                "macro_f1_pct": pct(metrics_from_records(econ_sft_records)["macro_f1"]),
            },
            "transfer_gain_over_base_pct_points": transfer_gap_pct,
            "per_tier_recall_pct": per_tier_recall,
        },
    )

    # ED8 pooled summary stats
    pooled_table_rows: List[dict] = []
    pooled_stats_models: Dict[str, dict] = {}
    for display, model_key in POOLED_2FIELD_MODELS.items():
        pooled_stats_models[model_key] = {"display": display, "fields": {}}
        for field in POOLED_FIELD_ORDER:
            subset = [row for row in pooled_rows if row["subject"] == field]
            records = compute_records(subset, model_key)
            metrics = metrics_from_records(records)
            cis = bootstrap_metric_cis(records, seed=2000 + len(pooled_table_rows))
            shares = {}
            total = sum(metrics["pred_counts"].values())
            for label in LABELS:
                shares[f"{label}_share_pct"] = round(
                    100.0 * metrics["pred_counts"].get(label, 0) / max(total, 1),
                    1,
                )
            row = {
                "model": display,
                "evaluator": model_key,
                "field": field,
                "n": len(records),
                "accuracy_pct": pct(metrics["accuracy"]),
                "accuracy_ci_lower_pct": pct_ci(cis["accuracy"])[0],
                "accuracy_ci_upper_pct": pct_ci(cis["accuracy"])[1],
                "macro_f1_pct": pct(metrics["macro_f1"]),
                "macro_f1_ci_lower_pct": pct_ci(cis["macro_f1"])[0],
                "macro_f1_ci_upper_pct": pct_ci(cis["macro_f1"])[1],
                **shares,
            }
            pooled_table_rows.append(row)
            pooled_stats_models[model_key]["fields"][field] = row

    write_json(
        STATS_DIR / "S17_PooledFieldExtensionStats.json",
        {
            "figure": "ExtendedDataFigure8",
            "benchmark_n": len(pooled_rows),
            "models": pooled_stats_models,
        },
    )


def main() -> None:
    build_figure7()
    build_extended_data_figure8()
    build_supplementary_figure7()
    write_economics_extension_release_artifacts()


if __name__ == "__main__":
    main()
