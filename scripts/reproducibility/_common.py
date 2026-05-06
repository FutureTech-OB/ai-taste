"""Shared helpers for the v6.3 reproduction scripts in this package.

Conventions
-----------
- Label space (4-class): exceptional, strong, fair, limited.
- argmax over per-label logp uses alphabetical tie-break in the ordering
  exceptional < fair < limited < strong (the canonical AI-label-space
  alphabetical tie-break used throughout the manuscript). Implemented via
  ``sorted()`` to avoid the dict-insertion-order trap.
- Plurality vote among rater labels uses ``tie_policy='exclude'``: tied
  articles are dropped from the denominator.

JSON outputs
------------
The wrappers write JSON with ``sort_keys=True`` and rounded floats so the
files are byte-stable across machines and Python versions.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple


PACKAGE_ROOT = Path(__file__).resolve().parents[2]

SFT_PREDICTIONS_PATH = (
    PACKAGE_ROOT
    / "data/management_deep_probe/predictions/sft_predictions.jsonl"
)
FRONTIER_THINKING_PATH = (
    PACKAGE_ROOT
    / "data/management_deep_probe/predictions/frontier_thinking_11models_singleshot.jsonl"
)
EXPERTS_PATH = (
    PACKAGE_ROOT
    / "data/management_deep_probe/human_ratings/expert_ratings_deidentified.jsonl"
)
STUDENTS_PATH = (
    PACKAGE_ROOT
    / "data/management_deep_probe/human_ratings/student_ratings_deidentified.jsonl"
)
STATISTICS_DIR = PACKAGE_ROOT / "data/management_deep_probe/statistics"


AI_LABELS_ALPHABETICAL: List[str] = ["exceptional", "fair", "limited", "strong"]
LABEL_ORDER_DISPLAY: List[str] = ["exceptional", "strong", "fair", "limited"]

LEVEL_TO_AI: Dict[str, str] = {
    "top": "exceptional",
    "top-": "strong",
    "good": "fair",
    "fair": "limited",
}

HUMAN_TO_AI: Dict[str, str] = {
    "Top": "exceptional",
    "Top-": "strong",
    "Good": "fair",
    "Fair": "limited",
}


THINKING_MODELS_11: List[str] = [
    "z-ai/glm-5",
    "moonshotai/kimi-k2.5",
    "bytedance-seed/seed-1.6",
    "google/gemini-3.1-pro-preview",
    "google/gemini-3-pro-preview",
    "anthropic/claude-opus-4.6",
    "openai/gpt-5.2-pro",
    "x-ai/grok-4.1-fast",
    "minimax/minimax-m2.5",
    "deepseek/deepseek-v3.2-speciale",
    "google/gemini-2.5-pro",
]

BEST_FRONTIER_KEY = "google/gemini-3.1-pro-preview"

SFT_MODELS_PUBLIC: List[str] = [
    "gpt-4.1-ob",
    "gpt-4.1-nano-ob",
    "qwen3-30b-ob",
    "qwen3-4b-ob",
]

PRIMARY_ENSEMBLE_KEY = "best_2_model_combo"
PRIMARY_PAIR_REQUIRED = ["gpt-4.1-nano-ob", "qwen3-30b-ob"]

SFT_DISPLAY: Dict[str, str] = {
    "gpt-4.1-ob": "SFT GPT-4.1",
    "gpt-4.1-nano-ob": "SFT GPT-4.1-nano",
    "qwen3-30b-ob": "SFT Qwen3-30B",
    "qwen3-4b-ob": "SFT Qwen3-4B",
}


def load_jsonl(path: Path) -> List[Dict]:
    out: List[Dict] = []
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                out.append(json.loads(line))
    return out


def index_by_title(records: List[Dict]) -> Dict[str, Dict]:
    return {r["title"]: r for r in records}


def normalize_to_ai_label(raw: Optional[str]) -> Optional[str]:
    if raw is None:
        return None
    s = str(raw).strip().strip("*").strip()
    if not s:
        return None
    if s in HUMAN_TO_AI:
        return HUMAN_TO_AI[s]
    low = s.lower()
    if low in LEVEL_TO_AI:
        return LEVEL_TO_AI[low]
    if low in {"exceptional", "strong", "fair", "limited"}:
        return low
    return None


def article_ground_truth(article: Dict) -> str:
    if "rank" in article and article["rank"]:
        gt = article["rank"].strip().lower()
        if gt in {"exceptional", "strong", "fair", "limited"}:
            return gt
    if "level" in article:
        norm = normalize_to_ai_label(article["level"])
        if norm is not None:
            return norm
    raise KeyError(
        f"Cannot derive ground truth from article keys={list(article.keys())}"
    )


def argmax_logp_alphabetical(logp_dict: Optional[Dict[str, float]]) -> Optional[str]:
    if not logp_dict or not isinstance(logp_dict, dict):
        return None
    NEG_INF = float("-inf")
    enriched = []
    for label in AI_LABELS_ALPHABETICAL:
        v = logp_dict.get(label, NEG_INF)
        if v is None:
            v = NEG_INF
        enriched.append((label, float(v)))
    enriched.sort(key=lambda kv: (-kv[1], kv[0]))
    top_label, top_value = enriched[0]
    if top_value == NEG_INF:
        return None
    return top_label


def majority_vote_excluding_ties(votes: List[str]) -> Tuple[Optional[str], bool]:
    if not votes:
        return (None, False)
    counts = Counter(votes)
    top_n = max(counts.values())
    top_labels = [lab for lab, c in counts.items() if c == top_n]
    if len(top_labels) > 1:
        return (None, True)
    return (top_labels[0], False)


def sft_pred(article: Dict, model_key: str) -> Optional[str]:
    rq = article.get("val_outcome", {}).get("rq_with_context", {})
    md = rq.get(model_key, {})
    if not isinstance(md, dict):
        return None
    return argmax_logp_alphabetical(md.get("logp"))


def thinking_pred(article: Dict, model_key: str) -> Optional[str]:
    rq = article.get("val_outcome", {}).get("rq_with_context", {})
    md = rq.get(model_key, {})
    if not isinstance(md, dict):
        return None
    return normalize_to_ai_label(md.get("prediction"))


def expert_votes_for_article(article: Dict) -> List[str]:
    out: List[str] = []
    for r in article.get("ratings", []):
        rt = r.get("rater_type")
        if rt is not None and rt != "expert":
            continue
        norm = normalize_to_ai_label(r.get("q2_rating"))
        if norm is not None:
            out.append(norm)
    return out


def student_votes_for_article(article: Dict) -> List[str]:
    out: List[str] = []
    for r in article.get("ratings", []):
        rt = r.get("rater_type")
        if rt is not None and rt != "student":
            continue
        norm = normalize_to_ai_label(r.get("q2_rating"))
        if norm is not None:
            out.append(norm)
    return out


JSON_FLOAT_PRECISION = 8


def _round_floats(obj):
    if isinstance(obj, float):
        if obj != obj:  # NaN
            return None
        if obj == float("inf") or obj == float("-inf"):
            return str(obj)
        return round(obj, JSON_FLOAT_PRECISION)
    if isinstance(obj, dict):
        return {k: _round_floats(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round_floats(v) for v in obj]
    if isinstance(obj, tuple):
        return [_round_floats(v) for v in obj]
    return obj


def write_stable_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rounded = _round_floats(payload)
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(rounded, handle, indent=2, sort_keys=True, ensure_ascii=False)
        handle.write("\n")
