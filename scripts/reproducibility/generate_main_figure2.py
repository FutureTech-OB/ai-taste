#!/usr/bin/env python3
"""Generate main Figure 2 as PNG/PDF.

Design goals (user-requested):
- Panel a is primary and spans full width horizontally.
- Panel b keeps tier distribution view with more room.
- Panel c uses one-column confusion matrices for key flagship examples.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import gridspec

from figure_style_policy import apply_title_policy, panel_title


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
OUT_PNG = ROOT / "reproduced" / "figures" / "main" / "Figure2.png"
OUT_PDF = ROOT / "reproduced" / "figures" / "main" / "Figure2.pdf"
REPORTS_ROOT = ROOT
FRONTIER_PRED_FILE = REPORTS_ROOT / "data" / "predictions" / "frontier_10models_8runs.jsonl"

TIER_ORDER = ["exceptional", "strong", "fair", "limited"]
TIER_COLORS = {
    "exceptional": "#0072B2",
    "strong": "#56B4E9",
    "fair": "#E69F00",
    "limited": "#D55E00",
}


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.5,
            "axes.titlesize": 7.6,
            "axes.labelsize": 7,
            "xtick.labelsize": 5.8,
            "ytick.labelsize": 5.8,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 0.7,
        }
    )


def binomial_ci_halfwidth(p: np.ndarray, n: np.ndarray, z: float = 1.96) -> np.ndarray:
    n_safe = np.maximum(n, 1)
    return z * np.sqrt(p * (1.0 - p) / n_safe)


def short_name_map() -> dict[str, str]:
    return {
        "Gemini 2.5 Pro": "Gemini 2.5",
        "Gemini 3.1 Pro": "Gemini 3.1*",
        "Kimi K2.5": "Kimi K2.5",
        "GPT-5.2 High": "GPT-5.2 High",
        "GPT-5.2 Pro": "GPT-5.2 High",
        "Grok 4.1 Fast": "Grok 4.1",
        "GLM-5": "GLM-5",
        "Claude Opus 4.6": "Opus 4.6",
        "MiniMax M2.5": "MiniMax M2.5",
        "Seed 2.0": "Seed 2.0",
        "DeepSeek V3.2": "DeepSeek V3.2",
        "Qwen 3.5 Plus": "Qwen 3.5",
    }


def resolve_model_name(model_name: str, model_performance: dict) -> str:
    """Resolve display-name aliases across old/new table exports."""
    aliases = {
        "GPT-5.2 High": "GPT-5.2 Pro",
        "GPT-5.2 Pro": "GPT-5.2 High",
    }
    if model_name in model_performance:
        return model_name
    alt = aliases.get(model_name)
    if alt and alt in model_performance:
        return alt
    return model_name


def _macro_f1_from_pairs(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
) -> float:
    f1s = []
    for lbl in labels:
        tp = np.sum((y_true == lbl) & (y_pred == lbl))
        fp = np.sum((y_true != lbl) & (y_pred == lbl))
        fn = np.sum((y_true == lbl) & (y_pred != lbl))
        denom = (2 * tp + fp + fn)
        f1 = (2 * tp / denom) if denom > 0 else 0.0
        f1s.append(f1)
    return float(np.mean(f1s))


def _load_majority_pairs(frontier_file: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    """Return majority-vote (true, pred) arrays per model with ties excluded."""
    per_model: dict[str, list[tuple[str, str]]] = {}
    with frontier_file.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            y_true = str(rec.get("rank", "")).strip().lower()
            if y_true not in TIER_ORDER:
                continue
            outcomes = rec.get("val_outcome", {}).get("rq_with_context", {})
            for model_key, out in outcomes.items():
                if out.get("vote_is_tie") is True or out.get("vote_tied") is True:
                    continue
                y_pred = out.get("prediction_majority") or out.get("prediction")
                if not y_pred:
                    counts = out.get("vote_counts") or {}
                    if counts:
                        max_count = max(counts.values())
                        winners = [k for k, v in counts.items() if v == max_count]
                        if len(winners) == 1:
                            y_pred = winners[0]
                y_pred = str(y_pred).strip().lower() if y_pred else ""
                if y_pred not in TIER_ORDER:
                    continue
                per_model.setdefault(model_key, []).append((y_true, y_pred))

    arr_map: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for model_key, pairs in per_model.items():
        if not pairs:
            continue
        y_t = np.array([p[0] for p in pairs], dtype=object)
        y_p = np.array([p[1] for p in pairs], dtype=object)
        arr_map[model_key] = (y_t, y_p)
    return arr_map


def _bootstrap_macro_f1_ci(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    labels: list[str],
    n_bootstrap: int = 5000,
    seed: int = 42,  # deterministic seed for reproducibility
) -> tuple[float, float]:
    n = len(y_true)
    if n == 0:
        return float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    boot = np.empty(n_bootstrap, dtype=float)
    idx_base = np.arange(n)
    for i in range(n_bootstrap):
        idx = rng.choice(idx_base, size=n, replace=True)
        boot[i] = _macro_f1_from_pairs(y_true[idx], y_pred[idx], labels)
    lo, hi = np.percentile(boot, [2.5, 97.5])
    return float(lo), float(hi)


def make_figure2() -> plt.Figure:
    with (STATS_DIR / "S01_FlagshipStats.json").open("r", encoding="utf-8") as f:
        stats02 = json.load(f)

    df = pd.read_csv(TABLES_DIR / "T02_FlagshipPerformance.csv")
    df = df.sort_values("Accuracy", ascending=False).reset_index(drop=True)

    short = short_name_map()
    model_labels = [short.get(m, m) for m in df["Model"].tolist()]
    acc = df["Accuracy"].to_numpy(dtype=float) * 100.0
    f1 = df["Macro F1"].to_numpy(dtype=float)
    n_valid = df["N Valid"].to_numpy(dtype=float)
    ci = binomial_ci_halfwidth(acc / 100.0, n_valid) * 100.0
    model_keys = df["Model Key"].tolist()

    # Compute per-model macro-F1 bootstrap CIs from majority-vote predictions.
    majority_pairs = _load_majority_pairs(FRONTIER_PRED_FILE)
    f1_ci_lo = np.full(len(df), np.nan, dtype=float)
    f1_ci_hi = np.full(len(df), np.nan, dtype=float)
    for i, model_key in enumerate(model_keys):
        pair = majority_pairs.get(model_key)
        if pair is None:
            continue
        y_t, y_p = pair
        # Per-model deterministic seed (123 + model_index) ensures reproducible CIs
        lo, hi = _bootstrap_macro_f1_ci(y_t, y_p, TIER_ORDER, n_bootstrap=4000, seed=123 + i)
        f1_ci_lo[i] = lo * 100.0
        f1_ci_hi[i] = hi * 100.0

    fig = plt.figure(figsize=(9.4, 8.8))
    outer = gridspec.GridSpec(
        2,
        2,
        height_ratios=[1.08, 1.32],
        width_ratios=[1.22, 0.95],
        hspace=0.36,
        wspace=0.34,
    )

    # ----------------------------
    # Panel a (top, full-width)
    # ----------------------------
    ax_a = fig.add_subplot(outer[0, :])
    panel_title(ax_a, "a", "Frontier leaderboard", family="main")

    x = np.arange(len(df))
    bar_w = 0.34
    acc_x = x - bar_w / 2
    f1_x = x + bar_w / 2
    acc_color = "#4E79A7"
    f1_color = "#F28E2B"

    acc_bars = ax_a.bar(
        acc_x,
        acc,
        width=bar_w,
        color=acc_color,
        edgecolor="white",
        linewidth=0.35,
        label="Accuracy",
        zorder=2,
    )
    f1_bars = ax_a.bar(
        f1_x,
        f1 * 100.0,
        width=bar_w,
        color=f1_color,
        edgecolor="white",
        linewidth=0.35,
        label="Macro-F1",
        zorder=2,
    )
    ax_a.errorbar(acc_x, acc, yerr=ci, fmt="none", ecolor="#4D4D4D", elinewidth=0.8, capsize=2, zorder=3)

    # Macro-F1 CI may be unavailable for rare parsing failures; skip safely.
    f1_center = f1 * 100.0
    f1_yerr = np.vstack(
        [
            np.where(np.isfinite(f1_ci_lo), f1_center - f1_ci_lo, np.nan),
            np.where(np.isfinite(f1_ci_hi), f1_ci_hi - f1_center, np.nan),
        ]
    )
    mask = np.isfinite(f1_yerr[0]) & np.isfinite(f1_yerr[1])
    if np.any(mask):
        ax_a.errorbar(
            f1_x[mask],
            f1_center[mask],
            yerr=f1_yerr[:, mask],
            fmt="none",
            ecolor="#4D4D4D",
            elinewidth=0.8,
            capsize=2,
            zorder=3,
        )

    for b, val in zip(acc_bars, acc):
        ax_a.text(
            b.get_x() + b.get_width() / 2,
            val + 0.9,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=5.7,
        )
    for b, val in zip(f1_bars, f1_center):
        ax_a.text(
            b.get_x() + b.get_width() / 2,
            val + 0.9,
            f"{val:.1f}%",
            ha="center",
            va="bottom",
            fontsize=5.5,
            color="#2E2E2E",
        )

    ax_a.axhline(25, color="#8C8C8C", linestyle="--", linewidth=0.9)
    ax_a.text(len(df) - 0.2, 25.7, "Chance (25%)", ha="right", va="bottom", fontsize=6, color="#6F6F6F")

    ax_a.set_ylabel("Score (%)")
    ax_a.set_xticks(x)
    ax_a.set_xticklabels(model_labels, rotation=45, ha="right")
    ax_a.set_ylim(0, 60)
    ax_a.grid(axis="y", color="#E6E6E6", linewidth=0.6, zorder=1)
    ax_a.legend(loc="upper left", fontsize=5.6, frameon=True, framealpha=0.9)
    ax_a.text(
        0.995,
        0.02,
        "* Contamination-risk diagnostic retained for conservative benchmark reporting.",
        transform=ax_a.transAxes,
        ha="right",
        va="bottom",
        fontsize=5.0,
        color="#666666",
    )

    # ----------------------------
    # Panel b (bottom-left)
    # ----------------------------
    ax_b = fig.add_subplot(outer[1, 0])
    panel_title(ax_b, "b", "Predicted-tier distribution", family="main")

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

    xb = np.arange(len(df))
    bottom = np.zeros(len(df))
    for t in TIER_ORDER:
        vals = pred_pct[:, TIER_ORDER.index(t)]
        ax_b.bar(
            xb,
            vals,
            width=0.68,
            bottom=bottom,
            color=TIER_COLORS[t],
            label=t.capitalize(),
            edgecolor="white",
            linewidth=0.28,
        )
        bottom += vals

    ax_b.axhline(25, color="#8C8C8C", linestyle="--", linewidth=0.8)
    ax_b.set_ylabel("Predicted share (%)")
    ax_b.set_ylim(0, 100)
    ax_b.set_xticks(xb)
    ax_b.set_xticklabels(model_labels, rotation=45, ha="right")
    # Keep panel-b legend inside its own panel so it never intrudes into panel c.
    ax_b.legend(
        loc="upper right",
        fontsize=5.2,
        title="Predicted tier",
        title_fontsize=5.4,
        frameon=True,
        framealpha=0.9,
        borderpad=0.3,
        labelspacing=0.25,
        handlelength=1.4,
    )

    # ----------------------------
    # Panel c (bottom-right): one-column confusion matrices
    # ----------------------------
    host = fig.add_subplot(outer[1, 1])
    host.axis("off")
    panel_title(host, "c", "Confusion matrices", family="main")

    # Reserve a dedicated narrow column for panel-c color scale (no overlap).
    inner = outer[1, 1].subgridspec(
        5,
        2,
        height_ratios=[0.22, 1.0, 1.0, 1.0, 1.0],
        width_ratios=[24.0, 1.15],
        wspace=0.16,
        hspace=0.28,
    )
    cm_models = [
        "Gemini 3.1 Pro",
        "GPT-5.2 High",
        "Claude Opus 4.6",
        "Qwen 3.5 Plus",
    ]

    axs = []
    im = None
    for i, model_name in enumerate(cm_models):
        ax = fig.add_subplot(inner[i + 1, 0])
        axs.append(ax)

        resolved = resolve_model_name(model_name, stats02["model_performance"])
        cm = np.array(stats02["model_performance"][resolved]["confusion_matrix"], dtype=float)
        row_sum = cm.sum(axis=1, keepdims=True)
        row_sum[row_sum == 0] = 1
        cm_norm = cm / row_sum

        im = ax.imshow(cm_norm, vmin=0, vmax=1, cmap="Blues", aspect="auto")
        ax.set_title(short.get(model_name, model_name), fontsize=5.2, pad=1)

        ax.set_xticks(range(4))
        ax.set_yticks(range(4))

        if i < 3:
            ax.set_xticklabels([])
            ax.tick_params(axis="x", length=0)
        else:
            ax.set_xticklabels(["Top", "Top-", "Good", "Fair"], rotation=35, ha="right", fontsize=4.6)
            ax.set_xlabel("Predicted", fontsize=5)

        ax.set_yticklabels(["Top", "Top-", "Good", "Fair"], fontsize=4.6)
        if i == 1:
            ax.set_ylabel("True", fontsize=5)

        for r in range(4):
            for c in range(4):
                val = cm_norm[r, c]
                ax.text(c, r, f"{val:.2f}", ha="center", va="center", fontsize=4.4, color="white" if val >= 0.5 else "black")

    if im is not None:
        cax = fig.add_subplot(inner[1:, 1])
        cbar = fig.colorbar(im, cax=cax)
        cbar.ax.tick_params(labelsize=4.8)
        cbar.set_label("Row-normalized", fontsize=5.0)

    fig.subplots_adjust(left=0.07, right=0.93, bottom=0.08, top=0.95)
    return fig


def main() -> None:
    set_style()
    OUT_PNG.parent.mkdir(parents=True, exist_ok=True)
    fig = make_figure2()
    apply_title_policy(fig, family="main")
    fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
    fig.savefig(OUT_PDF, bbox_inches="tight")
    plt.close(fig)
    print(f"Generated: {OUT_PNG}")
    print(f"Generated: {OUT_PDF}")


if __name__ == "__main__":
    main()
