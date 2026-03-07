#!/usr/bin/env python3
"""Rebuild core derived tables/statistics from package-local inputs only.

This script refreshes the package tables/statistics that are directly derived
from the public prediction and rating files:

- `data/tables/T05_SignificanceMain.csv`
- `data/tables/T09_PairwiseMcNemarCompendium.csv`
- `data/tables/T14_ErrorAnalysis.csv`
- `data/tables/T15_HardestArticles.csv`
- `data/tables/T16_DomainPerformance.csv`
- `data/statistics/S08_ErrorStats.json`
- `data/statistics/S09_DomainStats.json`
- `data/statistics/S10_SignificanceStats.json`

The public package still ships additional frozen summary tables/statistics used
by figure scripts. This script focuses on the subset that can be regenerated
unambiguously from the package-local raw inputs.
"""

from __future__ import annotations

import csv
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from scipy.stats import binomtest, chi2, chi2_contingency, kruskal

DEFAULT_BEST_FLAGSHIP_KEY = "x-ai/grok-4.1-fast"
DEFAULT_BEST_FLAGSHIP_NAME = "Grok 4.1 Fast"

CHINESE_TO_DISPLAY = {
    "领导力/管理者": "Leadership",
    "团队与组织": "Teams & Organizations",
    "员工行为与态度": "Employee Behavior",
    "其他管理学研究": "Other Management",
    "社会心理/人际": "Social Psychology",
    "绩效与成果": "Performance & Outcomes",
    "创新与创业": "Innovation & Entrepreneurship",
    "AI与技术": "AI & Technology",
    "知识与学习": "Knowledge & Learning",
    "多元化与公平": "Diversity & Equity",
    "人力资源管理": "Human Resource Management",
    "战略与决策": "Strategy & Decision-Making",
    "伦理与道德": "Ethics & Morality",
    "职业发展": "Career Development",
    "工作压力与健康": "Work Stress & Health",
}

ENGLISH_TO_DISPLAY = {
    "Leadership / Management": "Leadership",
    "Team & Organization": "Teams & Organizations",
    "Employee Behavior & Attitudes": "Employee Behavior",
    "Other Management Research": "Other Management",
    "Social Psychology / Interpersonal": "Social Psychology",
    "Performance & Outcomes": "Performance & Outcomes",
    "Innovation & Entrepreneurship": "Innovation & Entrepreneurship",
    "AI & Technology": "AI & Technology",
    "Knowledge & Learning": "Knowledge & Learning",
    "Diversity & Equity": "Diversity & Equity",
    "Human Resource Management": "Human Resource Management",
    "Strategy & Decision-Making": "Strategy & Decision-Making",
    "Ethics & Morality": "Ethics & Morality",
    "Career Development": "Career Development",
    "Work Stress & Health": "Work Stress & Health",
}

ENGLISH_TO_CHINESE = {
    "Leadership / Management": "领导力/管理者",
    "Team & Organization": "团队与组织",
    "Employee Behavior & Attitudes": "员工行为与态度",
    "Other Management Research": "其他管理学研究",
    "Social Psychology / Interpersonal": "社会心理/人际",
    "Performance & Outcomes": "绩效与成果",
    "Innovation & Entrepreneurship": "创新与创业",
    "AI & Technology": "AI与技术",
    "Knowledge & Learning": "知识与学习",
    "Diversity & Equity": "多元化与公平",
    "Human Resource Management": "人力资源管理",
    "Strategy & Decision-Making": "战略与决策",
    "Ethics & Morality": "伦理与道德",
    "Career Development": "职业发展",
    "Work Stress & Health": "工作压力与健康",
}

TIER_TO_ORD = {
    "limited": 0,
    "fair": 1,
    "strong": 2,
    "exceptional": 3,
}


def find_root() -> Path:
    here = Path(__file__).resolve()
    for candidate in [here.parent, *here.parents]:
        if (candidate / "README.md").exists() and (candidate / "data").is_dir() and (candidate / "scripts").is_dir():
            return candidate
    raise RuntimeError("Could not locate repository root")


def title_key(title: str) -> str:
    return " ".join(str(title).strip().casefold().split())


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def normalize_label(raw: Any, *, human: bool = False) -> Optional[str]:
    if raw is None:
        return None
    text = str(raw).strip()
    if not text:
        return None
    if human:
        return {
            "Top": "exceptional",
            "Top-": "strong",
            "Good": "fair",
            "Fair": "limited",
        }.get(text)

    low = text.lower()
    return {
        "exceptional": "exceptional",
        "strong": "strong",
        "fair": "fair",
        "limited": "limited",
        "top": "exceptional",
        "top-": "strong",
        "good": "fair",
    }.get(low)


def normalize_domain_display(token: str) -> str:
    token = token.strip()
    if token in ENGLISH_TO_DISPLAY:
        return ENGLISH_TO_DISPLAY[token]
    if token in CHINESE_TO_DISPLAY:
        return CHINESE_TO_DISPLAY[token]
    return token


def normalize_domain_chinese(token: str) -> str:
    token = token.strip()
    if token in ENGLISH_TO_CHINESE:
        return ENGLISH_TO_CHINESE[token]
    if token in CHINESE_TO_DISPLAY:
        return token
    return token


def parse_domains(domain_raw: str) -> Tuple[str, List[str], str]:
    parts = [part.strip() for part in str(domain_raw).split(",") if part.strip()]
    if not parts:
        return "Unknown", [], ""
    display = [normalize_domain_display(part) for part in parts]
    chinese = [normalize_domain_chinese(part) for part in parts]
    return display[0], display, ", ".join(chinese)


def pred_from_logp(logp: Any) -> Optional[str]:
    if not isinstance(logp, dict) or not logp:
        return None
    best_label = None
    best_score = float("-inf")
    for label, score in logp.items():
        norm = normalize_label(label)
        if norm is None:
            continue
        try:
            value = float(score)
        except (TypeError, ValueError):
            continue
        if value > best_score:
            best_score = value
            best_label = norm
    return best_label


def majority_vote(labels: Iterable[Optional[str]]) -> Optional[str]:
    clean = [label for label in labels if label]
    if not clean:
        return None
    counts = Counter(clean)
    top = max(counts.values())
    winners = sorted([label for label, count in counts.items() if count == top])
    if len(winners) != 1:
        return None
    return winners[0]


def model_majority_prediction(model_output: Dict[str, Any]) -> Optional[str]:
    if not isinstance(model_output, dict):
        return None
    if model_output.get("vote_is_tie", False):
        return None

    pred_majority = normalize_label(model_output.get("prediction_majority"))
    if pred_majority:
        return pred_majority

    vote_counts = model_output.get("vote_counts") or {}
    if isinstance(vote_counts, dict) and vote_counts:
        top = max(vote_counts.values())
        tied = sorted(
            normalize_label(label)
            for label, count in vote_counts.items()
            if count == top
        )
        tied = [label for label in tied if label]
        if len(set(tied)) == 1:
            return tied[0]
        if len(tied) > 1:
            return None

    return normalize_label(model_output.get("prediction"))


def accuracy_stats(rows: List[Dict[str, Any]], key: str) -> Tuple[int, int, float]:
    valid = [row for row in rows if row.get(key) is not None]
    n = len(valid)
    correct = sum(1 for row in valid if row[key] == row["ground_truth"])
    acc = (correct / n) if n else float("nan")
    return n, correct, acc


def paired_results(rows: List[Dict[str, Any]], key_a: str, key_b: str) -> Dict[str, Any]:
    n = 0
    correct_a = 0
    correct_b = 0
    b = 0
    c = 0

    for row in rows:
        pred_a = row.get(key_a)
        pred_b = row.get(key_b)
        gt = row["ground_truth"]
        if pred_a is None or pred_b is None:
            continue

        n += 1
        a_ok = pred_a == gt
        b_ok = pred_b == gt
        correct_a += int(a_ok)
        correct_b += int(b_ok)
        if a_ok and not b_ok:
            b += 1
        elif b_ok and not a_ok:
            c += 1

    stat = ((abs(b - c) - 1) ** 2) / (b + c) if (b + c) else 0.0
    p_value = float(chi2.sf(stat, 1)) if (b + c) else 1.0
    odds_ratio: Optional[float]
    if c == 0:
        odds_ratio = float("inf") if b > 0 else None
    else:
        odds_ratio = b / c

    return {
        "n": n,
        "correct_a": correct_a,
        "correct_b": correct_b,
        "acc_a": (correct_a / n) if n else float("nan"),
        "acc_b": (correct_b / n) if n else float("nan"),
        "b": b,
        "c": c,
        "stat": stat,
        "p_raw": p_value,
        "odds_ratio": odds_ratio,
    }


def cramers_v(table: List[List[int]]) -> Dict[str, float]:
    stat, p_value, _, _ = chi2_contingency(table)
    n = float(sum(sum(row) for row in table))
    if n <= 0:
        return {"statistic": float("nan"), "p_value": float("nan"), "cramers_v": float("nan")}
    r = len(table)
    c = len(table[0]) if table else 0
    denom = min(r - 1, c - 1)
    value = math.sqrt(stat / (n * denom)) if denom > 0 else float("nan")
    return {"statistic": float(stat), "p_value": float(p_value), "cramers_v": float(value)}


def format_float(
    value: Optional[float],
    *,
    digits: int = 4,
    scientific_below: Optional[float] = None,
) -> str:
    if value is None:
        return ""
    try:
        num = float(value)
    except (TypeError, ValueError):
        return ""
    if not math.isfinite(num):
        return ""
    if scientific_below is not None and abs(num) < scientific_below and num != 0:
        return f"{num:.2e}"
    return f"{num:.{digits}f}"


def write_csv(path: Path, fieldnames: List[str], rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def read_frontier_summary(root: Path) -> Tuple[float, float, str, str, int]:
    frontier_rows = load_jsonl(root / "data" / "predictions" / "frontier_10models_8runs.jsonl")

    def _raw_model_mean(model_key: str) -> float:
        values = []
        for row in frontier_rows:
            out = row["val_outcome"]["rq_with_context"].get(model_key, {})
            avg_accuracy = out.get("avg_accuracy")
            if avg_accuracy is not None:
                values.append(float(avg_accuracy))
        return sum(values) / len(values)

    model_keys = list(frontier_rows[0]["val_outcome"]["rq_with_context"].keys())
    model_means: List[Tuple[str, float]] = []
    for model_key in model_keys:
        mean_value = _raw_model_mean(model_key)
        model_means.append((model_key, mean_value))
    model_means.sort(key=lambda item: item[1], reverse=True)
    best_key, best_primary = model_means[0]
    best_name = best_key

    # Use T02 only as a display-name lookup layer if available.
    table_path = root / "data" / "tables" / "T02_FlagshipPerformance.csv"
    if table_path.exists():
        try:
            with table_path.open("r", encoding="utf-8", newline="") as f:
                rows = list(csv.DictReader(f))
            by_key = {str(row.get("Model Key", "")): str(row.get("Model", "")) for row in rows}
            best_name = by_key.get(best_key) or best_name
        except Exception:
            best_name = best_name

    return (
        sum(v for _, v in model_means) / len(model_means),
        best_primary,
        best_key,
        best_name,
        len(model_means),
    )


def build_records(root: Path, best_flagship_key: str) -> List[Dict[str, Any]]:
    benchmark = {
        title_key(row["title"]): row
        for row in load_jsonl(root / "data" / "benchmark" / "benchmark_articles_120.jsonl")
    }
    frontier = {
        title_key(row["title"]): row
        for row in load_jsonl(root / "data" / "predictions" / "frontier_10models_8runs.jsonl")
    }
    experts = {
        title_key(row["title"]): row
        for row in load_jsonl(root / "data" / "human_ratings" / "reproducibility" / "expert_reproducibility.jsonl")
    }
    students = {
        title_key(row["title"]): row
        for row in load_jsonl(root / "data" / "human_ratings" / "reproducibility" / "student_reproducibility_filtered.jsonl")
    }

    records: List[Dict[str, Any]] = []
    for row in load_jsonl(root / "data" / "predictions" / "sft_predictions.jsonl"):
        key = title_key(row["title"])
        bench = benchmark[key]
        frontier_row = frontier[key]
        expert_row = experts[key]
        student_row = students[key]

        rq_sft = row["val_outcome"]["rq_with_context"]
        rq_frontier = frontier_row["val_outcome"]["rq_with_context"]

        primary_domain, all_domains, chinese_domain = parse_domains(bench["domain"])

        records.append(
            {
                "title": row["title"],
                "title_key": key,
                "ground_truth": normalize_label(row.get("rank") or bench.get("level")),
                "sft_single": pred_from_logp(rq_sft.get("ckpt-step-304", {}).get("logp")),
                "sft2": pred_from_logp(rq_sft.get("best_2_model_combo", {}).get("logp")),
                "gpt52": pred_from_logp(rq_sft.get("gpt-5.2", {}).get("logp")),
                "best_flagship": model_majority_prediction(rq_frontier.get(best_flagship_key, {})),
                "expert_majority": majority_vote(
                    normalize_label(rating.get("q2_rating"), human=True)
                    for rating in expert_row.get("ratings", [])
                ),
                "junior_majority": majority_vote(
                    normalize_label(rating.get("q2_rating"), human=True)
                    for rating in student_row.get("ratings", [])
                ),
                "domain_raw": bench["domain"],
                "domain_chinese": chinese_domain,
                "primary_domain": primary_domain,
                "all_domains": all_domains,
                "journal": bench["journal"],
            }
        )

    return records


def compute_significance(
    rows: List[Dict[str, Any]],
    frontier_average_accuracy: float,
    best_flagship_primary_accuracy: float,
    best_flagship_key: str,
    best_flagship_name: str,
    frontier_model_count: int,
) -> Dict[str, Any]:
    n_sft, sft_correct, sft_acc = accuracy_stats(rows, "sft2")

    comparisons = []

    frontier_p = float(
        binomtest(sft_correct, n_sft, frontier_average_accuracy, alternative="greater").pvalue
    )
    comparisons.append(
        {
            "comparison": f"Frontier average ({frontier_model_count} models)",
            "n": n_sft,
            "acc_sft2": sft_acc,
            "acc_cmp": frontier_average_accuracy,
            "acc_diff_pp": (sft_acc - frontier_average_accuracy) * 100.0,
            "test": "Exact binomial (one-sided)",
            "stat": None,
            "p_raw": frontier_p,
            "odds_ratio": None,
            "short_name": "Frontier Average",
        }
    )

    pair_specs = [
        (
            f"Best-performing frontier model ({best_flagship_name})",
            "best_flagship",
            "Best Flagship",
        ),
        ("Expert majority (excl. ties)", "expert_majority", "Expert Majority"),
        ("Junior majority (full, excl. ties)", "junior_majority", "Junior Majority (full)"),
    ]

    evaluator_accuracies = {
        "SFT 2-Model": sft_acc,
        "Frontier Average": frontier_average_accuracy,
    }

    for label, key, short_name in pair_specs:
        paired = paired_results(rows, "sft2", key)
        comparisons.append(
            {
                "comparison": label,
                "n": paired["n"],
                "acc_sft2": paired["acc_a"],
                "acc_cmp": paired["acc_b"],
                "acc_diff_pp": (paired["acc_a"] - paired["acc_b"]) * 100.0,
                "test": "McNemar (continuity-corrected)",
                "stat": paired["stat"],
                "p_raw": paired["p_raw"],
                "odds_ratio": paired["odds_ratio"],
                "short_name": short_name,
            }
        )
        evaluator_accuracies[short_name] = paired["acc_b"]

    evaluator_accuracies["Best Flagship"] = comparisons[1]["acc_cmp"]

    pairwise_results = []
    for row in comparisons:
        eval_2 = row.get("short_name")
        if eval_2 is None:
            eval_2 = "Frontier Average"
        pairwise_results.append(
            {
                "evaluator_1": "SFT 2-Model",
                "evaluator_2": eval_2,
                "n_paired": int(row["n"]),
                "acc_1": float(row["acc_sft2"]),
                "acc_2": float(row["acc_cmp"]),
                "test": row["test"],
                "statistic": None if row["stat"] is None else float(row["stat"]),
                "p_value": float(row["p_raw"]),
                "odds_ratio": None if row["odds_ratio"] is None else float(row["odds_ratio"]),
                "acc_diff": float(row["acc_sft2"] - row["acc_cmp"]),
            }
        )

    return {
        "best_flagship_model": best_flagship_key,
        "best_flagship_name": best_flagship_name,
        "frontier_model_count": int(frontier_model_count),
        "best_flagship_primary_accuracy": best_flagship_primary_accuracy,
        "frontier_average_accuracy": frontier_average_accuracy,
        "n_articles": n_sft,
        "evaluator_accuracies": evaluator_accuracies,
        "n_pairwise_tests": len(pairwise_results),
        "pairwise_results": pairwise_results,
        "summary": {
            "rule": "Simplified core comparisons only; raw p-values only (no Bonferroni/FDR columns).",
            "best_flagship_name": best_flagship_name,
        },
        "comparisons": comparisons,
    }


def compute_error_tables(
    rows: List[Dict[str, Any]],
    best_flagship_primary_accuracy: float,
    best_flagship_key: str,
    best_flagship_name: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    evaluator_map = [
        ("SFT 2-Model", "sft2"),
        ("Best Flagship", "best_flagship"),
        ("Expert Majority", "expert_majority"),
        ("Student Majority", "junior_majority"),
    ]

    t14_rows: List[Dict[str, Any]] = []
    error_magnitude: Dict[str, Dict[str, Any]] = {}
    error_direction: Dict[str, Dict[str, Any]] = {}

    pattern_table: List[List[int]] = []
    direction_table: List[List[int]] = []

    difficulty_distribution = {str(i): 0 for i in range(5)}
    tier_difficulty: Dict[str, List[int]] = {tier: [0, 0, 0, 0, 0] for tier in TIER_TO_ORD}

    hardest_candidates: List[Dict[str, Any]] = []
    complete_records = 0
    easy_articles = 0
    hard_articles = 0

    ai_right_expert_wrong_articles: List[Dict[str, Any]] = []
    expert_right_ai_wrong_articles: List[Dict[str, Any]] = []

    for row in rows:
        gt = row["ground_truth"]
        article_correct: Dict[str, Optional[bool]] = {}
        valid_predictions = 0
        correct_predictions = 0

        for _, key in evaluator_map:
            pred = row[key]
            if pred is None:
                article_correct[key] = None
                continue
            valid_predictions += 1
            is_correct = pred == gt
            article_correct[key] = is_correct
            correct_predictions += int(is_correct)

        if valid_predictions == 4:
            complete_records += 1
        if valid_predictions > 0 and correct_predictions == valid_predictions:
            easy_articles += 1
        if correct_predictions == 0:
            hard_articles += 1

        difficulty_distribution[str(correct_predictions)] += 1
        tier_difficulty[gt][correct_predictions] += 1

        if row["expert_majority"] is not None:
            if article_correct["sft2"] and article_correct["expert_majority"] is False:
                ai_right_expert_wrong_articles.append(
                    {
                        "title": row["title"],
                        "tier": gt,
                        "domain": row["domain_chinese"],
                        "journal": row["journal"],
                    }
                )
            elif article_correct["expert_majority"] and article_correct["sft2"] is False:
                expert_right_ai_wrong_articles.append(
                    {
                        "title": row["title"],
                        "tier": gt,
                        "domain": row["domain_chinese"],
                        "journal": row["journal"],
                    }
                )

        hardest_candidates.append(
            {
                "category": "",
                "title": row["title"],
                "ground_truth": gt,
                "domain": row["domain_chinese"],
                "journal": row["journal"],
                "n_correct": correct_predictions,
                "n_evaluators": valid_predictions,
                "sft2_correct": int(bool(article_correct["sft2"])),
                "flagship_correct": int(bool(article_correct["best_flagship"])),
                "expert_correct": int(bool(article_correct["expert_majority"])),
                "student_correct": int(bool(article_correct["junior_majority"])),
            }
        )

    for label, key in evaluator_map:
        valid = [row for row in rows if row[key] is not None]
        n = len(valid)
        exact = 0
        off_one = 0
        off_two_plus = 0
        over = 0
        under = 0

        for row in valid:
            gt_ord = TIER_TO_ORD[row["ground_truth"]]
            pred_ord = TIER_TO_ORD[row[key]]
            diff = pred_ord - gt_ord
            if diff == 0:
                exact += 1
            elif abs(diff) == 1:
                off_one += 1
            else:
                off_two_plus += 1

            if diff > 0:
                over += 1
            elif diff < 0:
                under += 1

        bias = "Balanced"
        if over > under:
            bias = "Optimistic"
        elif under > over:
            bias = "Pessimistic"

        pattern_table.append([exact, off_one, off_two_plus])
        direction_table.append([over, exact, under])

        t14_rows.append(
            {
                "Evaluator": label,
                "N": n,
                "Exact Match": exact,
                "Exact Match %": format_float(exact * 100.0 / n if n else None, digits=1),
                "Off-by-One": off_one,
                "Off-by-One %": format_float(off_one * 100.0 / n if n else None, digits=1),
                "Off-by-Two+": off_two_plus,
                "Off-by-Two+ %": format_float(off_two_plus * 100.0 / n if n else None, digits=1),
                "Overestimation": over,
                "Over %": format_float(over * 100.0 / n if n else None, digits=1),
                "Underestimation": under,
                "Under %": format_float(under * 100.0 / n if n else None, digits=1),
                "Bias": bias,
            }
        )

        error_magnitude[label] = {
            "exact": exact,
            "off_by_one": off_one,
            "off_by_two_plus": off_two_plus,
            "total": n,
        }
        error_direction[label] = {
            "overestimation": over,
            "underestimation": under,
            "exact": exact,
            "total": n,
            "over_pct": (over * 100.0 / n) if n else float("nan"),
            "under_pct": (under * 100.0 / n) if n else float("nan"),
        }

    hard_rows = sorted(
        (row for row in hardest_candidates if row["n_correct"] == 0),
        key=lambda row: (row["n_correct"], -row["n_evaluators"], row["title"]),
    )[:20]
    for row in hard_rows:
        row["category"] = "hard"

    easy_rows = sorted(
        (row for row in hardest_candidates if row["n_correct"] >= 3),
        key=lambda row: (-row["n_correct"], -row["n_evaluators"], row["title"]),
    )[:20]
    for row in easy_rows:
        row["category"] = "easy"

    t15_rows = hard_rows + easy_rows

    ai_right_expert_wrong_tiers = Counter(article["tier"] for article in ai_right_expert_wrong_articles)
    expert_right_ai_wrong_tiers = Counter(article["tier"] for article in expert_right_ai_wrong_articles)

    tier_order = ["exceptional", "strong", "fair", "limited"]
    tier_table = [tier_difficulty[tier] for tier in tier_order]

    s08 = {
        "best_flagship_model": best_flagship_key,
        "best_flagship_name": best_flagship_name,
        "best_flagship_accuracy": best_flagship_primary_accuracy,
        "n_articles": len(rows),
        "n_complete_records": complete_records,
        "difficulty_distribution": difficulty_distribution,
        "n_easy_articles": easy_articles,
        "n_hard_articles": hard_articles,
        "error_magnitude": error_magnitude,
        "error_direction": error_direction,
        "ai_right_expert_wrong": {
            "count": len(ai_right_expert_wrong_articles),
            "tier_distribution": {tier: ai_right_expert_wrong_tiers.get(tier, 0) for tier in tier_order},
            "articles": ai_right_expert_wrong_articles,
        },
        "expert_right_ai_wrong": {
            "count": len(expert_right_ai_wrong_articles),
            "tier_distribution": {tier: expert_right_ai_wrong_tiers.get(tier, 0) for tier in tier_order},
            "articles": expert_right_ai_wrong_articles,
        },
        "chi_square_error_patterns": cramers_v(pattern_table),
        "chi_square_error_direction": cramers_v(direction_table),
        "chi_square_tier_difficulty": cramers_v(tier_table),
    }

    return t14_rows, t15_rows, s08


def compute_domain_tables(
    rows: List[Dict[str, Any]],
    best_flagship_primary_accuracy: float,
    best_flagship_key: str,
    best_flagship_name: str,
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    primary_counts: Counter[str] = Counter()
    all_counts: Counter[str] = Counter()
    grouped: Dict[str, List[Dict[str, Any]]] = {}

    for row in rows:
        primary_counts[row["primary_domain"]] += 1
        grouped.setdefault(row["primary_domain"], []).append(row)
        for domain in row["all_domains"]:
            all_counts[domain] += 1

    analysed_domains = [domain for domain, count in primary_counts.items() if count >= 3]
    analysed_domains.sort(key=lambda domain: (-primary_counts[domain], domain))

    t16_rows: List[Dict[str, Any]] = []
    domain_perf_stats: Dict[str, Dict[str, Any]] = {}
    kruskal_stats: Dict[str, Dict[str, Any]] = {}

    evaluator_map = [
        ("SFT 2-Model", "sft2"),
        ("Best Flagship", "best_flagship"),
        ("Expert Majority", "expert_majority"),
        ("Student Majority", "junior_majority"),
    ]

    sft_domain_table: List[List[int]] = []

    for domain in analysed_domains:
        domain_rows = grouped[domain]
        record = {
            "Domain": domain,
            "N Articles": len(domain_rows),
        }
        domain_perf_stats[domain] = {"domain": domain, "n_articles": len(domain_rows)}

        for label, key in evaluator_map:
            valid = [row for row in domain_rows if row[key] is not None]
            correct = sum(1 for row in valid if row[key] == row["ground_truth"])
            acc = (correct / len(valid)) if valid else float("nan")
            record[f"{label} Accuracy"] = format_float(acc, digits=3)
            domain_perf_stats[domain][label] = acc

            if label == "SFT 2-Model":
                sft_domain_table.append([correct, len(valid) - correct])

        t16_rows.append(record)

    for label, key in evaluator_map:
        samples = []
        sample_names = []
        for domain in analysed_domains:
            values = [
                1.0 if row[key] == row["ground_truth"] else 0.0
                for row in grouped[domain]
                if row[key] is not None
            ]
            if values:
                samples.append(values)
                sample_names.append(domain)
        if len(samples) >= 2:
            stat, p_value = kruskal(*samples)
            kruskal_stats[label] = {
                "statistic": float(stat),
                "p_value": float(p_value),
                "n_groups": len(samples),
                "group_names": sample_names,
            }

    s09 = {
        "best_flagship_model": best_flagship_key,
        "best_flagship_name": best_flagship_name,
        "best_flagship_primary_accuracy": best_flagship_primary_accuracy,
        "n_articles": len(rows),
        "domain_mapping": CHINESE_TO_DISPLAY,
        "primary_domain_counts": dict(primary_counts),
        "all_domain_counts": dict(all_counts),
        "domains_analysed": analysed_domains,
        "kruskal_wallis": kruskal_stats,
        "chi_square_domain_accuracy": cramers_v(sft_domain_table),
        "domain_performance": domain_perf_stats,
    }

    return t16_rows, s09


def write_outputs(
    root: Path,
    significance: Dict[str, Any],
    t14_rows: List[Dict[str, Any]],
    t15_rows: List[Dict[str, Any]],
    s08: Dict[str, Any],
    t16_rows: List[Dict[str, Any]],
    s09: Dict[str, Any],
) -> None:
    tables_dir = root / "data" / "tables"
    stats_dir = root / "data" / "statistics"

    t05_rows = []
    for row in significance["comparisons"]:
        t05_rows.append(
            {
                "Comparison": row["comparison"],
                "N": row["n"],
                "SFT 2-Model Acc": format_float(row["acc_sft2"], digits=4),
                "Comparator Acc": format_float(row["acc_cmp"], digits=4),
                "Acc Diff (pp)": format_float(row["acc_diff_pp"], digits=2),
                "Test": row["test"],
                "Statistic": format_float(row["stat"], digits=4),
                "p (raw)": format_float(row["p_raw"], digits=6, scientific_below=1e-4),
            }
        )

    t09_rows = []
    for row in significance["comparisons"]:
        comparator = str(row.get("short_name") or row["comparison"])
        t09_rows.append(
            {
                "Evaluator 1": "SFT 2-Model",
                "Evaluator 2": comparator,
                "N Paired": row["n"],
                "Acc 1": format_float(row["acc_sft2"], digits=4),
                "Acc 2": format_float(row["acc_cmp"], digits=4),
                "Test": row["test"],
                "Statistic": format_float(row["stat"], digits=4),
                "p-value (raw)": format_float(row["p_raw"], digits=6, scientific_below=1e-4),
                "Odds Ratio": format_float(row["odds_ratio"], digits=3),
                "Acc Diff": format_float(row["acc_sft2"] - row["acc_cmp"], digits=4),
            }
        )

    write_csv(
        tables_dir / "T05_SignificanceMain.csv",
        [
            "Comparison",
            "N",
            "SFT 2-Model Acc",
            "Comparator Acc",
            "Acc Diff (pp)",
            "Test",
            "Statistic",
            "p (raw)",
        ],
        t05_rows,
    )
    write_csv(
        tables_dir / "T09_PairwiseMcNemarCompendium.csv",
        [
            "Evaluator 1",
            "Evaluator 2",
            "N Paired",
            "Acc 1",
            "Acc 2",
            "Test",
            "Statistic",
            "p-value (raw)",
            "Odds Ratio",
            "Acc Diff",
        ],
        t09_rows,
    )
    write_csv(
        tables_dir / "T14_ErrorAnalysis.csv",
        [
            "Evaluator",
            "N",
            "Exact Match",
            "Exact Match %",
            "Off-by-One",
            "Off-by-One %",
            "Off-by-Two+",
            "Off-by-Two+ %",
            "Overestimation",
            "Over %",
            "Underestimation",
            "Under %",
            "Bias",
        ],
        t14_rows,
    )
    write_csv(
        tables_dir / "T15_HardestArticles.csv",
        [
            "category",
            "title",
            "ground_truth",
            "domain",
            "journal",
            "n_correct",
            "n_evaluators",
            "sft2_correct",
            "flagship_correct",
            "expert_correct",
            "student_correct",
        ],
        t15_rows,
    )
    write_csv(
        tables_dir / "T16_DomainPerformance.csv",
        [
            "Domain",
            "N Articles",
            "SFT 2-Model Accuracy",
            "Best Flagship Accuracy",
            "Expert Majority Accuracy",
            "Student Majority Accuracy",
        ],
        t16_rows,
    )

    stats_dir.mkdir(parents=True, exist_ok=True)
    (stats_dir / "S08_ErrorStats.json").write_text(json.dumps(s08, indent=2, ensure_ascii=False), encoding="utf-8")
    (stats_dir / "S09_DomainStats.json").write_text(json.dumps(s09, indent=2, ensure_ascii=False), encoding="utf-8")

    s10 = {
        "best_flagship_model": significance["best_flagship_model"],
        "best_flagship_name": significance["best_flagship_name"],
        "frontier_model_count": significance["frontier_model_count"],
        "best_flagship_primary_accuracy": significance["best_flagship_primary_accuracy"],
        "frontier_average_accuracy": significance["frontier_average_accuracy"],
        "n_articles": significance["n_articles"],
        "evaluator_accuracies": significance["evaluator_accuracies"],
        "n_pairwise_tests": significance["n_pairwise_tests"],
        "pairwise_results": significance["pairwise_results"],
        "summary": significance["summary"],
    }
    (stats_dir / "S10_SignificanceStats.json").write_text(
        json.dumps(s10, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def main() -> None:
    root = find_root()
    (
        frontier_average_accuracy,
        best_flagship_primary_accuracy,
        best_flagship_key,
        best_flagship_name,
        frontier_model_count,
    ) = read_frontier_summary(root)
    rows = build_records(root, best_flagship_key)
    significance = compute_significance(
        rows,
        frontier_average_accuracy,
        best_flagship_primary_accuracy,
        best_flagship_key,
        best_flagship_name,
        frontier_model_count,
    )
    t14_rows, t15_rows, s08 = compute_error_tables(
        rows,
        best_flagship_primary_accuracy,
        best_flagship_key,
        best_flagship_name,
    )
    t16_rows, s09 = compute_domain_tables(
        rows,
        best_flagship_primary_accuracy,
        best_flagship_key,
        best_flagship_name,
    )
    write_outputs(root, significance, t14_rows, t15_rows, s08, t16_rows, s09)
    print("Recomputed core tables/statistics:")
    for rel in [
        "data/tables/T05_SignificanceMain.csv",
        "data/tables/T09_PairwiseMcNemarCompendium.csv",
        "data/tables/T14_ErrorAnalysis.csv",
        "data/tables/T15_HardestArticles.csv",
        "data/tables/T16_DomainPerformance.csv",
        "data/statistics/S08_ErrorStats.json",
        "data/statistics/S09_DomainStats.json",
        "data/statistics/S10_SignificanceStats.json",
    ]:
        print(f"- {rel}")


if __name__ == "__main__":
    main()
