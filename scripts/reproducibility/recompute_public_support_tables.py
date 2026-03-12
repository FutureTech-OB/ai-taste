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


def flatten_human_records(path: Path) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for article in load_jsonl(path):
        gt = normalize_ground_truth(article.get("level"))
        for rating in article.get("ratings", []):
            pred = HUMAN_TO_AI.get(str(rating.get("q2_rating", "")).strip())
            if pred is None:
                continue
            out.append(
                {
                    "rater_id": rating.get("rater_id"),
                    "ground_truth": gt,
                    "predicted": pred,
                    "correct_int": int(pred == gt),
                }
            )
    return out


def recompute_t21() -> None:
    records = flatten_human_records(HUMAN_DIR / "student_reproducibility_filtered.jsonl")
    by_rater: "OrderedDict[str, Dict[str, int]]" = OrderedDict()

    for record in records:
        rater_id = str(record["rater_id"])
        if rater_id not in by_rater:
            by_rater[rater_id] = {
                "correct": 0,
                "total": 0,
            }
        by_rater[rater_id]["total"] += 1
        by_rater[rater_id]["correct"] += int(record["correct_int"])

    rows: List[Dict[str, Any]] = []
    for info in by_rater.values():
        total = int(info["total"])
        correct = int(info["correct"])
        rows.append(
            {
                "accuracy": correct / total if total else 0.0,
                "n_ratings": total,
            }
        )

    write_csv(
        TABLES_DIR / "T21_StudentDescriptive.csv",
        [
            "accuracy",
            "n_ratings",
        ],
        rows,
    )


def main() -> None:
    recompute_t21()
    print("Recomputed T21")


if __name__ == "__main__":
    main()
