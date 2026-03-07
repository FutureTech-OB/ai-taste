#!/usr/bin/env python3
"""Recompute package-local support tables not covered by vendored scripts."""

from __future__ import annotations

import csv
import json
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, List


ROOT = Path(__file__).resolve().parents[2]
TABLES_DIR = ROOT / "data" / "tables"
HUMAN_DIR = ROOT / "data" / "human_ratings" / "reproducibility"
HUMAN_TO_AI = {"Top": "exceptional", "Top-": "strong", "Good": "fair", "Fair": "limited"}


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def write_csv(path: Path, fieldnames: List[str], rows: Iterable[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_ground_truth(raw: Any) -> str:
    text = str(raw).strip()
    lower = text.lower()
    level_map = {
        "top": "exceptional",
        "top-": "strong",
        "good": "fair",
        "fair": "limited",
        "exceptional": "exceptional",
        "strong": "strong",
        "limited": "limited",
    }
    return level_map.get(lower, lower)


def parse_confidence(raw: Any) -> int | None:
    try:
        return int(str(raw).strip()[0])
    except (TypeError, ValueError, IndexError):
        return None


def parse_phd_year(raw: Any) -> int | None:
    text = str(raw).strip().lower()
    if not text:
        return None
    if "postdoc" in text or "博士后" in text or "博后" in text:
        return 7
    if "master" in text or "硕士" in text:
        return 0
    lookup = {
        "1st year": 1,
        "first year": 1,
        "2nd year": 2,
        "second year": 2,
        "3rd year": 3,
        "third year": 3,
        "4th year": 4,
        "fourth year": 4,
        "5th year": 5,
        "fifth year": 5,
        "博士一年级": 1,
        "博士二年级": 2,
        "博士三年级": 3,
        "博士四年级": 4,
        "博士五年级": 5,
    }
    for token, year in lookup.items():
        if token in text:
            return year
    return None


def parse_publications(raw: Any) -> float | None:
    text = str(raw).strip()
    if not text:
        return None
    digits = [int(part) for part in "".join(ch if ch.isdigit() else " " for ch in text).split()]
    if not digits:
        return None
    if len(digits) == 1:
        return float(digits[0])
    return float(sum(digits[:2]) / 2.0)


def normalize_student_background(raw: Any) -> Dict[str, Any]:
    source = dict(raw or {})
    if not source:
        return {}

    publications = source.get("publications")
    if publications in (None, ""):
        publications = parse_publications(source.get("published_papers"))

    phd_year = source.get("phd_year")
    if phd_year in (None, ""):
        phd_year = parse_phd_year(source.get("current_identity"))

    normalized = {
        "gender": source.get("gender", ""),
        "phd_year": phd_year,
        "publications": publications,
        "review_experience": source.get("review_experience", ""),
        "ai_familiarity": source.get("ai_familiarity", ""),
    }
    return {key: value for key, value in normalized.items() if value not in (None, "", [])}


def flatten_human_records(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for article in load_jsonl(path):
        gt = normalize_ground_truth(article.get("level"))
        title = str(article.get("title", ""))
        for rating in article.get("ratings", []):
            pred = HUMAN_TO_AI.get(str(rating.get("q2_rating", "")).strip())
            if pred is None:
                continue
            out.append(
                {
                    "rater_id": rating.get("rater_id"),
                    "article_title": title,
                    "ground_truth": gt,
                    "predicted": pred,
                    "correct_int": int(pred == gt),
                    "confidence_int": parse_confidence(rating.get("q3_confidence")),
                    "familiarity_int": parse_confidence(rating.get("q4_familiarity")),
                    "student_cohort": rating.get("student_cohort", ""),
                    "background": normalize_student_background(rating.get("background")),
                }
            )
    return out


def phd_year_group(value: float | None) -> str:
    if value is None:
        return ""
    if value <= 2:
        return "yr1-2"
    if value <= 4:
        return "yr3-4"
    return "yr5+"


def publications_group(value: float | None) -> str:
    if value is None:
        return ""
    if value == 0:
        return "0"
    if value <= 2:
        return "1-2"
    if value <= 5:
        return "3-5"
    return "6+"


def format_numeric(value: Any) -> str:
    if value in (None, ""):
        return ""
    return str(float(value))


def recompute_t24() -> None:
    records = flatten_human_records(HUMAN_DIR / "student_reproducibility_filtered.jsonl")
    by_rater: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    for record in records:
        rater_id = str(record["rater_id"])
        background = dict(record.get("background") or {})
        if rater_id not in by_rater:
            by_rater[rater_id] = {
                "cohort": record.get("student_cohort") or "",
                "correct": 0,
                "total": 0,
                "background": background,
                "has_background": bool(background),
            }
        by_rater[rater_id]["total"] += 1
        by_rater[rater_id]["correct"] += int(record["correct_int"])
        if not by_rater[rater_id]["background"] and background:
            by_rater[rater_id]["background"] = background
        if background:
            by_rater[rater_id]["has_background"] = True

    rows: List[Dict[str, Any]] = []
    for idx, info in enumerate(by_rater.values(), start=1):
        total = int(info["total"])
        correct = int(info["correct"])
        bg = dict(info["background"] or {})
        phd_year = bg.get("phd_year")
        publications = bg.get("publications")
        rows.append(
            {
                "student_id": f"Student_{idx:03d}",
                "cohort": info["cohort"],
                "accuracy": correct / total if total else 0.0,
                "n_ratings": total,
                "gender": bg.get("gender", ""),
                "phd_year": format_numeric(phd_year),
                "phd_group": phd_year_group(float(phd_year)) if phd_year not in (None, "") else "",
                "publications": format_numeric(publications),
                "pub_group": publications_group(float(publications)) if publications not in (None, "") else "",
                "review_experience": bg.get("review_experience", ""),
                "ai_familiarity": bg.get("ai_familiarity", ""),
                "has_background": bool(info["has_background"]),
            }
        )

    write_csv(
        TABLES_DIR / "T21_StudentDescriptive.csv",
        [
            "student_id",
            "cohort",
            "accuracy",
            "n_ratings",
            "gender",
            "phd_year",
            "phd_group",
            "publications",
            "pub_group",
            "review_experience",
            "ai_familiarity",
            "has_background",
        ],
        rows,
    )


def main() -> None:
    recompute_t24()
    print("Recomputed T21")


if __name__ == "__main__":
    main()
