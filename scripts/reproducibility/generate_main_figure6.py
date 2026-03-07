#!/usr/bin/env python3
"""Generate main Figure 6 voting-consistency summary with panel-a/b CIs."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from figure_style_policy import apply_title_policy, place_reference_label_safely


def find_project_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "README.md").exists() and (candidate / "scripts").is_dir() and (candidate / "data").is_dir():
            return candidate
    raise RuntimeError("Could not locate package root (expected README.md + scripts/ + data/).")


ROOT = find_project_root()
OUTPUT_DIR = ROOT / "reproduced" / "figures" / "main"
REPORTS_ROOT = ROOT

# SFT voting data (4 SFT checkpoints on 120 held-out samples)
AI_FILE = REPORTS_ROOT / "data" / "predictions" / "sft_predictions.jsonl"
AI_MODELS = ["CYqJRxId", "ckpt-step-304", "ckppt-380", "ckppt-228"]
AI_TO_HUMAN = {"exceptional": "top", "strong": "top-", "fair": "good", "limited": "fair"}

# Human voting data (package-local deidentified archive copies)
STUDENT_FILE = REPORTS_ROOT / "data" / "human_ratings" / "archive" / "survey_all_student.jsonl"
# Use unfiltered expert file to stay consistent with Figure 4b majority definition.
EXPERT_FILE = REPORTS_ROOT / "data" / "human_ratings" / "archive" / "survey_all_expert.jsonl"
HUMAN_TO_AI = {"Top": "top", "Top-": "top-", "Good": "good", "Fair": "fair"}

# Frontier voting data (conservative 11-model cohort)
FRONTIER_FILE = REPORTS_ROOT / "data" / "predictions" / "frontier_10models_8runs.jsonl"
FRONTIER_TOP4_FIXED = [
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-opus-4.6",
    "openai/gpt-5.2-high",
    "z-ai/glm-5",
]

# Same-model 8-run summary for frontier models (workspace variants)
AVG8_SUMMARY_FILE = REPORTS_ROOT / "data" / "tables" / "T01_FrontierProtocolAudit.csv"


def set_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
            "font.size": 6.6,
            "axes.titlesize": 7.4,
            "axes.labelsize": 6.9,
            "xtick.labelsize": 5.9,
            "ytick.labelsize": 5.9,
            "axes.linewidth": 0.7,
            "lines.linewidth": 1.1,
            "figure.dpi": 300,
            "savefig.dpi": 300,
            "legend.frameon": False,
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def majority_vote(labels: list[str]) -> tuple[str | None, int, int, bool]:
    if not labels:
        return None, 0, 0, False
    cnt = Counter(labels)
    top_count = max(cnt.values())
    modes = sorted([k for k, v in cnt.items() if v == top_count])
    pred = modes[0]
    return pred, top_count, len(labels), len(modes) == 1


def resolve_existing(candidates: list[Path], label: str) -> Path:
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(f"Could not locate {label}. Checked: {[str(p) for p in candidates]}")


def wilson_interval(successes: int, total: int, z: float = 1.959963984540054) -> tuple[float, float]:
    if total <= 0:
        return 0.0, 1.0
    p = float(successes) / float(total)
    denom = 1.0 + (z * z) / float(total)
    center = (p + (z * z) / (2.0 * float(total))) / denom
    half = (
        z
        * np.sqrt((p * (1.0 - p) + (z * z) / (4.0 * float(total))) / float(total))
        / denom
    )
    return max(0.0, center - half), min(1.0, center + half)


def load_sft_cross_model_metrics() -> dict:
    rows = []
    single_correct_total = 0
    single_total_votes = 0
    with AI_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gt = AI_TO_HUMAN.get(rec.get("rank", ""))
            if not gt:
                continue

            rq = rec.get("val_outcome", {}).get("rq_with_context", {})
            preds = []
            per_model_correct = []
            ok = True
            for key in AI_MODELS:
                logp = rq.get(key, {}).get("logp", {})
                if not logp:
                    ok = False
                    break
                pred = AI_TO_HUMAN[max(logp, key=logp.get)]
                preds.append(pred)
                per_model_correct.append(int(pred == gt))
            if not ok:
                continue

            pred, agree, _, _ = majority_vote(preds)
            single_correct_total += int(sum(per_model_correct))
            single_total_votes += len(per_model_correct)
            rows.append(
                {
                    "gt": gt,
                    "pred": pred,
                    "agree": agree,
                    "single_acc": float(sum(per_model_correct)) / 4.0,
                }
            )

    def _metric(cond) -> dict:
        subset = [r for r in rows if cond(r)]
        n = len(subset)
        correct = sum(1 for r in subset if r["pred"] == r["gt"])
        acc = (float(correct) / float(n)) if n else 0.0
        return {"n": n, "correct": correct, "acc": acc}

    n_total = len(rows)
    single_avg = float(np.mean([r["single_acc"] for r in rows])) if rows else 0.0

    return {
        "n_total": n_total,
        "single_avg": single_avg,
        "single_correct": single_correct_total,
        "single_total": single_total_votes,
        "vote_all": _metric(lambda r: True),
        "ge3": _metric(lambda r: r["agree"] >= 3),
        "eq4": _metric(lambda r: r["agree"] == 4),
    }


def load_frontier_top4_cross_model_metrics() -> dict:
    rows = []
    with FRONTIER_FILE.open("r", encoding="utf-8") as f:
        for line in f:
            rows.append(json.loads(line))

    model_keys = set(rows[0]["val_outcome"]["rq_with_context"].keys())
    for r in rows[1:]:
        model_keys &= set(r["val_outcome"]["rq_with_context"].keys())
    model_keys = sorted(model_keys)
    perf = []
    for mk in model_keys:
        vals = []
        for r in rows:
            out = r["val_outcome"]["rq_with_context"].get(mk)
            if out is None:
                continue
            mode = out.get("mode", "deterministic_logp")
            if mode == "stochastic_thinking" and out.get("avg_accuracy") is not None:
                vals.append(float(out["avg_accuracy"]))
                continue
            pred = out.get("prediction")
            if not pred:
                continue
            vals.append(float(pred == r["rank"]))
        acc = float(np.mean(vals)) if vals else 0.0
        perf.append((mk, acc, len(vals), int(round(acc * len(vals))) if vals else 0))
    perf.sort(key=lambda x: x[1], reverse=True)
    available = {p[0] for p in perf}
    missing = [m for m in FRONTIER_TOP4_FIXED if m not in available]
    if missing:
        raise ValueError(f"Missing fixed frontier models in source file: {missing}")

    voted = []
    for r in rows:
        gt = r["rank"]
        preds = []
        for mk in FRONTIER_TOP4_FIXED:
            model_out = r["val_outcome"]["rq_with_context"].get(mk)
            if model_out is None:
                preds = []
                break
            if model_out.get("vote_is_tie", False):
                preds = []
                break
            pred = model_out.get("prediction_majority")
            if not pred:
                preds = []
                break
            preds.append(pred)
        if len(preds) != 4:
            continue
        pred, agree, _, _ = majority_vote(preds)
        voted.append(
            {
                "gt": gt,
                "pred": pred,
                "agree": agree,
            }
        )

    def _metric(cond) -> dict:
        subset = [r for r in voted if cond(r)]
        n = len(subset)
        correct = sum(1 for r in subset if r["pred"] == r["gt"])
        acc = (float(correct) / float(n)) if n else 0.0
        return {"n": n, "correct": correct, "acc": acc}

    n_total = len(voted)
    top4_primary = [x[1] for x in perf if x[0] in FRONTIER_TOP4_FIXED]
    top4_primary_correct = sum(x[3] for x in perf if x[0] in FRONTIER_TOP4_FIXED)
    top4_primary_total = sum(x[2] for x in perf if x[0] in FRONTIER_TOP4_FIXED)
    single_avg = float(np.mean(top4_primary)) if top4_primary else 0.0
    best_single = perf[0]

    return {
        "n_total": n_total,
        "single_avg": single_avg,
        "single_correct": top4_primary_correct,
        "single_total": top4_primary_total,
        "best_single": best_single,
        "vote_all": _metric(lambda r: True),
        "ge3": _metric(lambda r: r["agree"] >= 3),
        "eq4": _metric(lambda r: r["agree"] == 4),
    }


def load_human_metrics(path: Path, min_unanimous_raters: int = 2) -> dict:
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            gt = str(rec.get("level", "")).strip().lower()
            ratings = []
            for r in rec.get("ratings", []):
                mapped = HUMAN_TO_AI.get(r.get("q2_rating", ""))
                if mapped:
                    ratings.append(mapped)
            pred, top_count, n_raters, unique = majority_vote(ratings)
            if pred is None:
                continue
            rows.append(
                {
                    "gt": gt,
                    "pred": pred,
                    "unique": unique,
                    "share": float(top_count) / float(n_raters) if n_raters else 0.0,
                    "n_raters": n_raters,
                }
            )

    def _metric(cond) -> dict:
        subset = [r for r in rows if cond(r)]
        n = len(subset)
        correct = sum(1 for r in subset if r["pred"] == r["gt"])
        acc = (float(correct) / float(n)) if n else 0.0
        return {"n": n, "correct": correct, "acc": acc}

    eligible_ge2 = sum(1 for r in rows if r["n_raters"] >= min_unanimous_raters)

    return {
        "n_total": len(rows),
        "eligible_ge2": eligible_ge2,
        "majority": _metric(lambda r: r["unique"]),
        "ge50": _metric(lambda r: r["unique"] and r["share"] >= 0.5),
        "ge60": _metric(lambda r: r["unique"] and r["share"] >= 0.6),
        "unanimous": _metric(
            lambda r: r["unique"] and r["share"] == 1.0 and r["n_raters"] >= min_unanimous_raters
        ),
    }


def load_frontier_avg8_gains() -> list[tuple[str, float]]:
    name_map = {
        "google/gemini-2.5-pro": "Gemini 2.5 Pro",
        "google/gemini-3.1-pro-preview": "Gemini 3.1 Pro",
        "openai/gpt-5.2-high": "GPT-5.2 High",
        "x-ai/grok-4.1-fast": "Grok 4.1 Fast",
        "z-ai/glm-5": "GLM-5",
        "moonshotai/kimi-k2.5": "Kimi K2.5",
        "minimax/minimax-m2.5": "MiniMax M2.5",
        "deepseek/deepseek-v3.2-speciale": "DeepSeek V3.2",
        "anthropic/claude-opus-4.6": "Claude Opus 4.6",
        "qwen/qwen3.5-plus-02-15": "Qwen 3.5 Plus",
        "doubao-seed-2-0-pro-260215": "Seed 2.0",
    }
    rows = []
    with AVG8_SUMMARY_FILE.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            model = row["model"]
            if not row.get("avg8_acc"):
                continue
            delta_pp = (float(row["majority_acc_excl_ties"]) - float(row["avg8_acc"])) * 100.0
            rows.append((name_map.get(model, model), delta_pp))
    rows.sort(key=lambda x: x[1], reverse=True)
    return rows


def save_figure(fig: plt.Figure, basename: str) -> tuple[Path, Path]:
    apply_title_policy(fig, family="main")
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    png = OUTPUT_DIR / f"{basename}.png"
    pdf = OUTPUT_DIR / f"{basename}.pdf"
    fig.savefig(png, bbox_inches="tight", dpi=300)
    fig.savefig(pdf, bbox_inches="tight")
    return png, pdf


def main() -> None:
    set_style()

    sft = load_sft_cross_model_metrics()
    frontier = load_frontier_top4_cross_model_metrics()
    student = load_human_metrics(STUDENT_FILE, min_unanimous_raters=2)
    expert_unfiltered = load_human_metrics(EXPERT_FILE, min_unanimous_raters=2)
    gains = load_frontier_avg8_gains()

    fig = plt.figure(figsize=(7.6, 6.35))
    gs = fig.add_gridspec(2, 2, height_ratios=[1.0, 1.05], hspace=0.58, wspace=0.46)

    # --------------------------------------------------
    # Panel a: SFT vs frontier aggregation frontier
    # --------------------------------------------------
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_title("a  Aggregation Frontier: SFT vs Frontier")

    def _cov(n: int, total: int) -> float:
        return 100.0 * float(n) / float(max(total, 1))

    sft_pts = [
        ("single avg", 100.0, 100.0 * sft["single_avg"]),
        (">=3/4", _cov(sft["ge3"]["n"], sft["n_total"]), 100.0 * sft["ge3"]["acc"]),
        ("4/4", _cov(sft["eq4"]["n"], sft["n_total"]), 100.0 * sft["eq4"]["acc"]),
    ]
    fr_pts = [
        ("single avg", 100.0, 100.0 * frontier["single_avg"]),
        (
            "vote all",
            _cov(frontier["vote_all"]["n"], frontier["n_total"]),
            100.0 * frontier["vote_all"]["acc"],
        ),
        (">=3/4", _cov(frontier["ge3"]["n"], frontier["n_total"]), 100.0 * frontier["ge3"]["acc"]),
        ("4/4", _cov(frontier["eq4"]["n"], frontier["n_total"]), 100.0 * frontier["eq4"]["acc"]),
    ]

    ax1.plot([p[1] for p in sft_pts], [p[2] for p in sft_pts], color="#0072B2", marker="o", ms=4, lw=1.0)
    ax1.plot([p[1] for p in fr_pts], [p[2] for p in fr_pts], color="#D55E00", lw=1.0)

    sft_ci = {
        "single avg": {"n": sft["single_total"], "correct": sft["single_correct"]},
        ">=3/4": sft["ge3"],
        "4/4": sft["eq4"],
    }
    frontier_ci = {
        "single avg": {"n": frontier["single_total"], "correct": frontier["single_correct"]},
        "vote all": frontier["vote_all"],
        ">=3/4": frontier["ge3"],
        "4/4": frontier["eq4"],
    }

    def _draw_ci(x: float, y: float, ci_data: dict, color: str) -> None:
        n = int(ci_data.get("n", 0))
        k = int(ci_data.get("correct", 0))
        if n <= 0:
            return
        lo, hi = wilson_interval(k, n)
        y_lo = 100.0 * lo
        y_hi = 100.0 * hi
        ax1.errorbar(
            x,
            y,
            yerr=[[max(y - y_lo, 0.0)], [max(y_hi - y, 0.0)]],
            fmt="none",
            ecolor=color,
            elinewidth=0.85,
            capsize=1.8,
            capthick=0.85,
            alpha=0.85,
            zorder=2,
        )

    for label, x, y in sft_pts:
        _draw_ci(x, y, sft_ci[label], "#0072B2")
    for label, x, y in fr_pts:
        _draw_ci(x, y, frontier_ci[label], "#D55E00")
    # Draw frontier points explicitly so overlapping x-values (single avg vs vote-all at ~100%) remain visible.
    for label, x, y in fr_pts:
        if label == "single avg":
            ax1.plot(
                x,
                y,
                marker="s",
                ms=4.4,
                mfc="white",
                mec="#D55E00",
                mew=1.0,
                linestyle="None",
                zorder=4,
            )
        else:
            ax1.plot(x, y, marker="s", ms=4.0, mfc="#D55E00", mec="#D55E00", linestyle="None", zorder=4)

    sft_offsets = {"single avg": (1.2, 0.8), ">=3/4": (1.2, 1.0), "4/4": (1.2, 1.0)}
    for label, x, y in sft_pts:
        dx, dy = sft_offsets[label]
        ax1.text(x + dx, y + dy, f"SFT {label}", fontsize=5.0, color="#0072B2")
    frontier_offsets = {
        "single avg": (-5.5, -3.2, "right", "center"),
        "vote all": (-2.8, 5.2, "right", "center"),
        ">=3/4": (2.0, 2.2, "left", "center"),
        "4/4": (4.0, 1.1, "left", "center"),
    }
    frontier_labels = {
        "single avg": "Frontier single",
        "vote all": "Frontier all",
        ">=3/4": "Frontier >=3/4",
        "4/4": "Frontier 4/4",
    }
    for label, x, y in fr_pts:
        dx, dy, ha, va = frontier_offsets[label]
        ax1.text(
            x + dx,
            y + dy,
            frontier_labels[label],
            fontsize=4.8,
            color="#D55E00",
            ha=ha,
            va=va,
        )
    ax1.text(
        0.01,
        0.98,
        "Error bars: 95% Wilson CI",
        transform=ax1.transAxes,
        ha="left",
        va="top",
        fontsize=4.6,
        color="#666666",
    )

    ax1.axhline(25, color="#888888", lw=0.8, ls="--")
    place_reference_label_safely(
        ax1,
        "Chance (25%)",
        preferred_anchor="lower left",
        color="#666666",
        fontsize=5.2,
    )
    ax1.set_xlim(0, 105)
    ax1.set_ylim(0, 100)
    ax1.set_xlabel("Coverage (%)")
    ax1.set_ylabel("Accuracy (%)")

    # --------------------------------------------------
    # Panel b: Human policy check (Figure 4-consistent majority definitions)
    # --------------------------------------------------
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_title("b  Human Consensus Policies")

    human_rows = [
        ("Expert maj.", expert_unfiltered["majority"], expert_unfiltered["n_total"], "#009E73"),
        ("Junior maj.", student["majority"], student["n_total"], "#E69F00"),
        ("Junior >=50%", student["ge50"], student["n_total"], "#E69F00"),
        ("Expert unanm.", expert_unfiltered["unanimous"], expert_unfiltered["n_total"], "#009E73"),
    ]
    human_rows = [r for r in human_rows if r[1]["n"] > 0]

    y = np.arange(len(human_rows))
    vals = [100.0 * r[1]["acc"] for r in human_rows]
    covs = [100.0 * r[1]["n"] / max(r[2], 1) for r in human_rows]
    colors = [r[3] for r in human_rows]
    xerr_lo = []
    xerr_hi = []
    for _, metric, _, _ in human_rows:
        lo, hi = wilson_interval(int(metric["correct"]), int(metric["n"]))
        center = 100.0 * float(metric["acc"])
        xerr_lo.append(max(center - 100.0 * lo, 0.0))
        xerr_hi.append(max(100.0 * hi - center, 0.0))

    ax2.barh(y, vals, color=colors, height=0.62)
    ax2.errorbar(
        vals,
        y,
        xerr=[xerr_lo, xerr_hi],
        fmt="none",
        ecolor="#333333",
        elinewidth=0.9,
        capsize=2.0,
        capthick=0.9,
        zorder=4,
    )
    for yi, v, cov, xe in zip(y, vals, covs, xerr_hi):
        ax2.text(v + xe + 0.7, yi, f"{v:.1f}% ({cov:.1f}% cov.)", va="center", ha="left", fontsize=5.0)

    ax2.set_yticks(y)
    ax2.set_yticklabels([r[0] for r in human_rows])
    ax2.invert_yaxis()
    ax2.set_xlim(20, 100)
    ax2.set_xlabel("Accuracy (%)")
    ax2.axvline(25, color="#888888", lw=0.8, ls="--")
    ax2.text(
        0.99,
        0.98,
        "Error bars: 95% Wilson CI",
        transform=ax2.transAxes,
        ha="right",
        va="top",
        fontsize=4.6,
        color="#666666",
    )

    # --------------------------------------------------
    # Panel c: same-model 8-run majority-vote gain for frontier models
    # --------------------------------------------------
    ax3 = fig.add_subplot(gs[1, :])
    ax3.set_title("c  Frontier Same-Model 8-Run Voting Gain")

    names = [n for n, _ in gains]
    deltas = [d for _, d in gains]
    y3 = np.arange(len(names))
    colors3 = ["#D55E00" if d >= 0 else "#999999" for d in deltas]
    ax3.barh(y3, deltas, color=colors3, height=0.62)
    ax3.axvline(0, color="#444444", lw=0.8)

    for yi, d in zip(y3, deltas):
        if d >= 0:
            ax3.text(d + 0.08, yi, f"+{d:.2f}", va="center", ha="left", fontsize=5.2)
        else:
            ax3.text(d - 0.08, yi, f"{d:.2f}", va="center", ha="right", fontsize=5.2)

    ax3.set_yticks(y3)
    ax3.set_yticklabels(names, fontsize=5.2)
    ax3.invert_yaxis()
    ax3.set_xlabel("Gain (pp): majority-vote accuracy - mean single-run accuracy (8 runs)")
    xmax = max(max(deltas), 0.5) + 0.35
    xmin = min(min(deltas), -0.5) - 0.35
    ax3.set_xlim(xmin, xmax)

    mean_gain = float(np.mean(deltas))
    median_gain = float(np.median(deltas))
    ax3.text(
        0.995,
        1.035,
        f"Across {len(deltas)} models\nmean {mean_gain:+.2f} pp; median {median_gain:+.2f} pp",
        transform=ax3.transAxes,
        ha="right",
        va="bottom",
        fontsize=4.7,
        color="#444444",
        clip_on=False,
        bbox={"facecolor": "white", "edgecolor": "#dddddd", "pad": 0.18},
    )

    png, pdf = save_figure(fig, "Figure6")
    plt.close(fig)
    print(f"Saved: {png}")
    print(f"Saved: {pdf}")


if __name__ == "__main__":
    main()
