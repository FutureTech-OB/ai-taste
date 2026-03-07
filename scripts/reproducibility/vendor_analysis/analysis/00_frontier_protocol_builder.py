#!/usr/bin/env python3
"""Build the canonical frontier protocol file and protocol-audit table.

Current canonical sources:
  - data/predictions/prompt_variants/expert_prompt_predictions.jsonl
  - data/predictions/gemini_3_1_pro_standalone.jsonl

Protocol rules:
  - fixed frontier cohort from ``FRONTIER_MODELS`` (11 models)
  - fixed 8-run stochastic protocol per model/article (strict first-8 normalization)
  - primary frontier metric: per-article avg8 (mean run accuracy over valid votes)
  - discrete diagnostics: majority vote with ties excluded

Outputs:
  - data/predictions/frontier_10models_8runs.jsonl
  - results/tables/table_frontier_protocol_audit.csv
"""

from __future__ import annotations

import csv
import json
import sys
from collections import Counter
from copy import deepcopy
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from analysis.utils.constants import (
    FRONTIER_MODELS,
    FRONTIER_PATH,
    GEMINI31_STANDALONE_PATH,
    PROMPT_EXPERT_PATH,
    TABLES_DIR,
)
from analysis.utils.data_loader import get_ground_truth, get_project_root, load_jsonl
from analysis.utils.frontier_protocol import CANONICAL_TO_SOURCE_KEYS, normalize_label

VALID_LABELS = {"exceptional", "strong", "fair", "limited"}
GEMINI31_KEY = "google/gemini-3.1-pro-preview"


def _title_key(title: Any) -> str:
    return " ".join(str(title).strip().casefold().split())


def _source_model_data(rq: Dict[str, Any], canonical_model_key: str) -> Dict[str, Any]:
    """Resolve one canonical model payload from aliased source keys."""
    for source_key in CANONICAL_TO_SOURCE_KEYS.get(canonical_model_key, [canonical_model_key]):
        if source_key in rq:
            data = rq[source_key]
            return data if isinstance(data, dict) else {}
    return {}


def _normalize_votes(raw_votes: Any) -> List[Optional[str]]:
    """Normalize to exactly 8 votes (strict first-8 protocol)."""
    votes: List[Optional[str]] = []
    if isinstance(raw_votes, list):
        for x in raw_votes[:8]:
            votes.append(normalize_label(x))
    while len(votes) < 8:
        votes.append(None)
    return votes


def _vote_counts(votes: List[Optional[str]]) -> Dict[str, int]:
    counts = Counter([v for v in votes if v in VALID_LABELS])
    return dict(counts)


def _majority_from_counts(vote_counts: Dict[str, int]) -> Tuple[Optional[str], bool, List[str]]:
    if not vote_counts:
        return (None, False, [])
    top_count = max(vote_counts.values())
    top_labels = sorted([k for k, v in vote_counts.items() if v == top_count])
    is_tie = len(top_labels) > 1
    return (None if is_tie else top_labels[0], is_tie, top_labels if is_tie else [])


def _first_valid(votes: List[Optional[str]]) -> Optional[str]:
    for v in votes:
        if v in VALID_LABELS:
            return v
    return None


def _normalize_stochastic_payload(gt: str, src: Dict[str, Any]) -> Dict[str, Any]:
    """Build strict canonical stochastic payload from one source payload."""
    votes = _normalize_votes(src.get("vote_predictions"))
    counts = _vote_counts(votes)
    maj_from_counts, is_tie, _ = _majority_from_counts(counts)

    pred_majority = normalize_label(src.get("prediction_majority"))
    if pred_majority is None:
        pred_majority = maj_from_counts
    if is_tie:
        pred_majority = None

    pred_run1 = normalize_label(src.get("prediction_run1"))
    if pred_run1 is None:
        pred_run1 = _first_valid(votes)
    if pred_run1 is None:
        pred_run1 = normalize_label(src.get("prediction"))

    pred = pred_majority or pred_run1
    vote_valid_n = int(sum(1 for v in votes if v in VALID_LABELS))
    avg_accuracy: Optional[float] = None
    if vote_valid_n > 0:
        avg_accuracy = float(sum(1 for v in votes if v == gt) / vote_valid_n)

    return {
        "mode": "stochastic_thinking",
        "prediction": pred,
        "prediction_majority": pred_majority,
        "avg_accuracy": avg_accuracy,
        "vote_valid_n": vote_valid_n,
        "vote_counts": counts,
        "vote_is_tie": bool(is_tie),
        "vote_predictions": votes,
    }


def _build_gemini31_index(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """Index Gemini 3.1 standalone rows by canonical title for robust merge."""
    index: Dict[str, Dict[str, Any]] = {}
    for row in rows:
        rq = row.get("val_outcome", {}).get("rq_with_context", {}) or {}
        model_payload = rq.get(GEMINI31_KEY)
        if not isinstance(model_payload, dict):
            continue
        tkey = _title_key(row.get("title", ""))
        index[tkey] = model_payload
    return index


def main() -> None:
    root = get_project_root()
    source_path = root / PROMPT_EXPERT_PATH
    frontier_path = root / FRONTIER_PATH
    tables_dir = root / TABLES_DIR
    tables_dir.mkdir(parents=True, exist_ok=True)
    audit_path = tables_dir / "table_frontier_protocol_audit.csv"

    source_rows = load_jsonl(source_path)
    if not source_rows:
        raise RuntimeError(f"No source rows found in: {source_path}")
    gemini31_rows = load_jsonl(root / GEMINI31_STANDALONE_PATH)
    gemini31_idx = _build_gemini31_index(gemini31_rows)
    gemini31_fallback_from_expert = 0
    gemini31_merged_from_standalone = 0

    canonical_rows: List[Dict[str, Any]] = []
    for src_row in source_rows:
        rq = src_row.get("val_outcome", {}).get("rq_with_context", {})
        gt = get_ground_truth(src_row)
        canonical_rq: Dict[str, Any] = {}

        for model_key in FRONTIER_MODELS:
            if model_key == GEMINI31_KEY:
                lookup_key = _title_key(src_row.get("title", ""))
                src_model = gemini31_idx.get(lookup_key)
                if isinstance(src_model, dict):
                    gemini31_merged_from_standalone += 1
                else:
                    src_model = _source_model_data(rq, model_key)
                    gemini31_fallback_from_expert += 1
            else:
                src_model = _source_model_data(rq, model_key)
            canonical_rq[model_key] = _normalize_stochastic_payload(gt, src_model)

        rec = deepcopy(src_row)
        rec.setdefault("val_outcome", {})
        rec["val_outcome"]["rq_with_context"] = canonical_rq
        canonical_rows.append(rec)

    frontier_path.parent.mkdir(parents=True, exist_ok=True)
    with open(frontier_path, "w", encoding="utf-8") as f:
        for rec in canonical_rows:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    # ------------------------------------------------------------------
    # Protocol audit table
    # ------------------------------------------------------------------
    audit_rows: List[Dict[str, Any]] = []

    for model_key in FRONTIER_MODELS:
        model_entries = [rec["val_outcome"]["rq_with_context"].get(model_key, {}) for rec in canonical_rows]
        gts = [get_ground_truth(rec) for rec in canonical_rows]

        n_articles = len(model_entries)
        n_with_8_runs = sum(1 for e in model_entries if len(e.get("vote_predictions") or []) == 8)

        vote_valid_vals: List[float] = []
        avg8_vals: List[float] = []
        maj_correct = 0
        maj_total = 0
        tie_count = 0
        run1_correct = 0
        run1_total = 0
        missing_count = 0

        for gt, e in zip(gts, model_entries):
            vote_valid = e.get("vote_valid_n")
            try:
                vote_valid_vals.append(float(vote_valid))
            except (TypeError, ValueError):
                pass

            avg_acc = e.get("avg_accuracy")
            try:
                avg8_vals.append(float(avg_acc))
            except (TypeError, ValueError):
                pass

            if bool(e.get("vote_is_tie")):
                tie_count += 1

            maj_pred = normalize_label(e.get("prediction_majority"))
            if maj_pred is not None and not bool(e.get("vote_is_tie", False)):
                maj_total += 1
                if maj_pred == gt:
                    maj_correct += 1

            run1_pred = _first_valid(_normalize_votes(e.get("vote_predictions")))
            if run1_pred is None:
                run1_pred = normalize_label(e.get("prediction"))
            if run1_pred is not None:
                run1_total += 1
                if run1_pred == gt:
                    run1_correct += 1

            pred = normalize_label(e.get("prediction"))
            if pred is None:
                missing_count += 1

        audit_rows.append(
            {
                "model": model_key,
                "mode": "stochastic_thinking",
                "n_articles": n_articles,
                "n_with_8_runs": n_with_8_runs,
                "avg_vote_valid_n": round(float(np.mean(vote_valid_vals)), 4) if vote_valid_vals else 0.0,
                "avg8_acc": round(float(np.mean(avg8_vals)), 4) if avg8_vals else "",
                "majority_acc_excl_ties": round(maj_correct / maj_total, 4) if maj_total else "",
                "tie_rate": round(tie_count / n_articles, 4) if n_articles else 0.0,
                "run1_acc": round(run1_correct / run1_total, 4) if run1_total else "",
                "missing_count": missing_count,
            }
        )

    audit_rows.sort(key=lambda r: float(r["avg8_acc"]) if r["avg8_acc"] != "" else -1.0, reverse=True)

    with open(audit_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "model",
                "mode",
                "n_articles",
                "n_with_8_runs",
                "avg_vote_valid_n",
                "avg8_acc",
                "majority_acc_excl_ties",
                "tie_rate",
                "run1_acc",
                "missing_count",
            ],
        )
        writer.writeheader()
        writer.writerows(audit_rows)

    print(f"Saved canonical frontier file: {frontier_path}")
    print(f"Saved protocol audit table:    {audit_path}")
    print(
        "Gemini 3.1 merge summary: "
        f"standalone rows used={gemini31_merged_from_standalone}, "
        f"fallback rows from expert source={gemini31_fallback_from_expert}"
    )


if __name__ == "__main__":
    main()
