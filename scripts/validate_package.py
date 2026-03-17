#!/usr/bin/env python3
"""Validate the public release repository against required acceptance criteria."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, List, Set

FORBIDDEN_ID_KEYS = {
    "rater_name",
    "response_id",
    "ip_address",
    "start_date",
    "end_date",
}

FORBIDDEN_MANUSCRIPT_SUBSTRINGS = [
    "/Users/",
    "Nutstore Files/reports",
    "paper/refined/",
    "paper/final_fig_nature_clean/",
    "results/tables/",
    "results/statistics/",
    "data/model_predictions/",
    "data/human_ratings/",
]

FORBIDDEN_PATH_SUBSTRINGS = [
    "/home/",
    "/workspace/",
]

FORBIDDEN_TABLE_COLUMNS = {
    "rater_name",
    "response_id",
    "ip_address",
    "start_date",
    "end_date",
    "expert_id",
    "student_id",
    "cohort",
    "gender",
    "phd_year",
    "phd_group",
    "publications",
    "pub_group",
    "review_experience",
    "ai_familiarity",
    "has_background",
}

FORBIDDEN_PUBLIC_JSON_KEYS = {
    "entries",
    "package",
    "file_name",
    "group",
    "total_raters",
    "response_text",
    "reasoning_content",
    "reasoning_meta",
    "prediction_run1",
    "vote_n",
    "vote_tied",
    "error",
    "metadata",
    "status",
    "split",
    "trial_count",
    "doi",
    "doi_first",
    "doi_second",
    "raw_response",
    "background",
    "student_cohort",
    "career_stage",
    "current_identity",
    "published_papers",
    "on_editor_list",
    "duration_minutes",
    "duration_seconds",
    "finished",
    "has_background",
    "confidence_int",
    "familiarity_int",
    "read_before_bool",
    "cohort",
}

EXPECTED_FRONTIER_MODELS = {
    "z-ai/glm-5",
    "moonshotai/kimi-k2.5",
    "google/gemini-2.5-pro",
    "google/gemini-3.1-pro-preview",
    "anthropic/claude-opus-4.6",
    "openai/gpt-5.2-high",
    "x-ai/grok-4.1-fast",
    "minimax/minimax-m2.5",
    "deepseek/deepseek-v3.2-speciale",
    "qwen/qwen3.5-plus-02-15",
    "doubao-seed-2-0-pro-260215",
}

EXPECTED_SFT_KEYS = {
    "CYqJRxId",
    "ckpt-step-304",
    "ckppt-380",
    "ckppt-228",
}

EXPECTED_ECONOMICS_EXTENSION_KEYS = {
    "qwen3_30b_base",
    "qwen3_30b_sft_economics",
    "qwen3_4b_base",
    "qwen3_4b_sft_economics",
    "gpt_4_1_nano_base",
    "gpt_4_1_nano_sft_economics",
    "gpt_4_1_base",
    "gpt_4_1_sft_management",
    "qwen_3_5_plus",
    "gemini_3_1_pro",
    "grok_4_1_fast",
}

EXPECTED_POOLED_FIELD_KEYS = {
    "qwen3_30b_sft_management_economics",
    "gpt_4_1_nano_sft_management_economics",
}

EXPECTED_OLD_SFT_KEYS = {
    "ft:gpt-4.1-nano-2025-04-14:personal:ob-rqcontext-old:DI3q8ijY",
    "old_qwen30b_checkpoint_178",
}

EXPECTED_CORE_RQ_SHORT_TRANSFER_KEYS = {
    "ft:gpt-4.1-2025-04-14:personal:ob-ob-rqcontext:DHnLrzmY",
    "ft:gpt-4.1-nano-2025-04-14:personal:ob-ob-rqcontext:DHKeHMNB",
    "gpt-4.1-nano-2025-04-14",
    "gpt-4.1-2025-04-14",
}

GEMINI_31_KEY = "google/gemini-3.1-pro-preview"
EXPECTED_PAPER_TITLE = "Fine-tuned language models learn tacit scientific judgment from institutional traces"

EXPECTED_PAIRWISE_DIRS = {
    "sft_gpt4_1",
    "frontier_gemini3_1_pro",
    "frontier_gpt5_2_high",
    "baseline_gpt4_1",
}

EXPECTED_FIGURE_INDEX_ROWS = {
    "Figure1",
    "Figure2",
    "Figure3",
    "Figure4",
    "Figure5",
    "Figure6",
    "Figure7",
    "ExtendedDataFigure1",
    "ExtendedDataFigure2",
    "ExtendedDataFigure3",
    "ExtendedDataFigure4",
    "ExtendedDataFigure5",
    "ExtendedDataFigure6",
    "ExtendedDataFigure7",
    "ExtendedDataFigure8",
    "SupplementaryFigure1",
    "SupplementaryFigure2",
    "SupplementaryFigure3",
    "SupplementaryFigure4",
    "SupplementaryFigure5",
    "SupplementaryFigure6",
    "SupplementaryFigure7",
}

REQUIRED_T04_EVALUATORS_FOR_CI = {
    "Expert Majority Vote (excl. ties)",
    "Student Majority Vote (full, excl. ties)",
    "Expert Individual Average",
    "Student Individual Average",
    "Flagship Average (11 models)",
    "SFT 2-Model Ensemble",
}

REQUIRED_CHAT_CONFIDENCE_ROWS = {
    "DeepSeek Chat",
    "GPT-5.2 Chat",
    "Kimi K2 Chat",
}

REQUIRED_HUMAN_JSONL = {
    "expert_ratings_deidentified.jsonl",
    "student_ratings_deidentified.jsonl",
}

REQUIRED_REPRO_HUMAN_JSONL = {
    "expert_reproducibility.jsonl",
    "expert_reproducibility_filtered.jsonl",
    "student_reproducibility.jsonl",
    "student_reproducibility_filtered.jsonl",
}

ALLOWED_HUMAN_ROOT = {
    "README.md",
    "expert_ratings_deidentified.jsonl",
    "student_ratings_deidentified.jsonl",
    "reproducibility",
}

REQUIRED_HUMAN_RECORD_KEYS = {
    "title",
    "journal",
    "level",
    "ratings",
}

OPTIONAL_HUMAN_RECORD_KEYS = {
    "domain",
}

ALLOWED_HUMAN_RATING_KEYS = {
    "rater_id",
    "rater_type",
    "q1_read_before",
    "q2_rating",
    "q3_confidence",
    "q4_familiarity",
}

TRAIN_DATA_PLACEHOLDER_README = "README.md"

def fail(msg: str, errors: List[str]) -> None:
    errors.append(msg)


def _has_forbidden_keys(obj: Any, forbidden: Set[str]) -> bool:
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in forbidden:
                return True
            if _has_forbidden_keys(v, forbidden):
                return True
    elif isinstance(obj, list):
        for item in obj:
            if _has_forbidden_keys(item, forbidden):
                return True
    return False


def _check_jsonl_no_forbidden_keys(path: Path, errors: List[str]) -> None:
    if not path.exists():
        fail(f"Missing de-identified file: {path}", errors)
        return

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"Invalid JSON in {path} line {i}: {exc}", errors)
                continue
            if _has_forbidden_keys(obj, FORBIDDEN_ID_KEYS):
                fail(f"Forbidden identifier key found in {path} line {i}", errors)
                return


def _check_human_jsonl_schema(path: Path, errors: List[str]) -> None:
    if not path.exists():
        return

    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"Invalid JSON in {path} line {i}: {exc}", errors)
                return
            if not isinstance(obj, dict):
                fail(f"Human ratings row must be an object in {path} line {i}", errors)
                return
            keys = set(obj.keys())
            missing = sorted(REQUIRED_HUMAN_RECORD_KEYS - keys)
            extra = sorted(keys - REQUIRED_HUMAN_RECORD_KEYS - OPTIONAL_HUMAN_RECORD_KEYS)
            if missing or extra:
                fail(
                    f"Unexpected human record schema in {path} line {i}: "
                    f"missing={missing or '[]'} extra={extra or '[]'}",
                    errors,
                )
                return
            ratings = obj.get("ratings")
            if not isinstance(ratings, list):
                fail(f"'ratings' must be a list in {path} line {i}", errors)
                return
            for j, rating in enumerate(ratings, start=1):
                if not isinstance(rating, dict):
                    fail(f"Rating entry must be an object in {path} line {i} rating {j}", errors)
                    return
                rating_keys = set(rating.keys())
                missing_rating = sorted(ALLOWED_HUMAN_RATING_KEYS - rating_keys)
                extra_rating = sorted(rating_keys - ALLOWED_HUMAN_RATING_KEYS)
                if missing_rating or extra_rating:
                    fail(
                        f"Unexpected rating schema in {path} line {i} rating {j}: "
                        f"missing={missing_rating or '[]'} extra={extra_rating or '[]'}",
                        errors,
                    )
                    return
                rid = str(rating.get("rater_id", "")).strip()
                if not rid:
                    fail(f"Blank rater_id in {path} line {i} rating {j}", errors)
                    return


def _first_row_model_keys(path: Path, eval_key: str = "rq_with_context") -> Set[str]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            return set(obj.get("val_outcome", {}).get(eval_key, {}).keys())
    return set()


def _count_nonempty_lines(path: Path) -> int:
    with path.open("r", encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


def _check_readme_title_contract(root: Path, errors: List[str]) -> None:
    readme_path = root / "README.md"
    if not readme_path.exists():
        return

    expected_line = f"**Paper title:** {EXPECTED_PAPER_TITLE}"
    actual_line = None
    for line in readme_path.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if stripped.startswith("**Paper title:**"):
            actual_line = stripped
            break

    if actual_line is None:
        fail("README.md is missing the '**Paper title:**' line", errors)
        return

    if actual_line != expected_line:
        fail(
            "README paper title mismatch. "
            f"Expected exactly: {expected_line!r}; got: {actual_line!r}",
            errors,
        )


def _check_core_rq_short_transfer_schema(path: Path, errors: List[str]) -> None:
    required_text_fields = {"one_sentence_idea_statement", "full_idea_summary"}
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                obj = json.loads(line)
            except json.JSONDecodeError as exc:
                fail(f"Invalid JSON in {path} line {i}: {exc}", errors)
                return
            for field in sorted(required_text_fields):
                if not str(obj.get(field, "")).strip():
                    fail(f"Missing required transfer text field '{field}' in {path.name} line {i}", errors)
                    return
            transfer = obj.get("val_outcome", {}).get("core_rq_short")
            if not isinstance(transfer, dict) or not transfer:
                fail(f"Missing val_outcome.core_rq_short payload in {path.name} line {i}", errors)
                return


def _check_tables_no_forbidden_columns(tables_dir: Path, errors: List[str]) -> None:
    for p in sorted(tables_dir.glob("*.csv")):
        if p.name in {"TABLE_INDEX.csv", "FIGURE_DATA_INDEX.csv"}:
            continue
        with p.open("r", encoding="utf-8", newline="") as f:
            reader = csv.reader(f)
            try:
                header = next(reader)
            except StopIteration:
                fail(f"Empty table file: {p.name}", errors)
                continue
        normalized = {h.strip().lower() for h in header}
        overlap = sorted(normalized.intersection(FORBIDDEN_TABLE_COLUMNS))
        if overlap:
            fail(f"Forbidden table columns in {p.name}: {overlap}", errors)


def _read_csv_rows(path: Path, errors: List[str]) -> List[dict]:
    if not path.exists():
        fail(f"Missing CSV file: {path}", errors)
        return []
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        rows = []
        for idx, row in enumerate(reader, start=2):
            if None in row:
                fail(f"Malformed CSV row in {path.name} row {idx}: extra unnamed fields detected", errors)
            rows.append(row)
    return rows


def _check_csv_row_width(path: Path, errors: List[str]) -> None:
    if not path.exists():
        fail(f"Missing CSV file: {path}", errors)
        return
    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.reader(f)
        try:
            header = next(reader)
        except StopIteration:
            fail(f"Empty CSV file: {path.name}", errors)
            return
        expected_width = len(header)
        for idx, row in enumerate(reader, start=2):
            if len(row) != expected_width:
                fail(
                    f"Inconsistent CSV row width in {path.name} row {idx}: "
                    f"expected {expected_width}, got {len(row)}",
                    errors,
                )
                break


def _check_ci_contract(tables_dir: Path, errors: List[str]) -> None:
    # T04 must include filled CI columns for evaluator rows used in main figures.
    t04 = tables_dir / "T04_AIvsHumanSummary.csv"
    rows = _read_csv_rows(t04, errors)
    if rows:
        by_eval = {str(r.get("Evaluator", "")): r for r in rows}
        for evaluator in sorted(REQUIRED_T04_EVALUATORS_FOR_CI):
            if evaluator not in by_eval:
                fail(f"Missing evaluator row in T04_AIvsHumanSummary.csv: {evaluator}", errors)
                continue
            row = by_eval[evaluator]
            for col in [
                "95% CI Lower",
                "95% CI Upper",
                "Macro F1 95% CI Lower",
                "Macro F1 95% CI Upper",
            ]:
                if str(row.get(col, "")).strip() == "":
                    fail(f"Blank CI field in T04_AIvsHumanSummary.csv ({evaluator}, {col})", errors)

    # T11 (SI Fig 3) must include Monte Carlo CI bounds.
    t11 = tables_dir / "T11_MonteCarloMatchedPanel.csv"
    rows11 = _read_csv_rows(t11, errors)
    if rows11:
        for col in ["ci_lower", "ci_upper"]:
            if col not in rows11[0]:
                fail(f"Missing required column in T11_MonteCarloMatchedPanel.csv: {col}", errors)

    # T19/T20 (ED4 + Fig3d support) must include non-empty accuracy CI columns.
    for name in ["T19_StudentReliabilityByPanelSize.csv", "T20_ExpertReliabilityByPanelSize.csv"]:
        p = tables_dir / name
        rows_k = _read_csv_rows(p, errors)
        if not rows_k:
            continue
        for col in ["accuracy_ci_lower", "accuracy_ci_upper"]:
            if col not in rows_k[0]:
                fail(f"Missing required column in {name}: {col}", errors)
        for idx, row in enumerate(rows_k, start=1):
            if str(row.get("accuracy_ci_lower", "")).strip() == "" or str(row.get("accuracy_ci_upper", "")).strip() == "":
                fail(f"Blank accuracy CI fields in {name} row {idx}", errors)
                break

    for name in ["T12_ConfidenceComparison.csv", "T17_AllModelsSummary.csv"]:
        p = tables_dir / name
        rows_conf = _read_csv_rows(p, errors)
        if not rows_conf:
            continue
        present = {str(row.get("model", "")).strip() for row in rows_conf}
        missing = sorted(REQUIRED_CHAT_CONFIDENCE_ROWS - present)
        if missing:
            fail(f"Missing expected chat evaluator rows in {name}: {missing}", errors)


def _check_pairwise_contract(root: Path, errors: List[str]) -> None:
    pairwise_dir = root / "data" / "pairwise"
    if not pairwise_dir.exists():
        fail("Missing data/pairwise directory (required for ExtendedDataFigure2 traceability)", errors)
        return

    readme = pairwise_dir / "README.md"
    if not readme.exists():
        fail("Missing pairwise README: data/pairwise/README.md", errors)

    existing_dirs = {p.name for p in pairwise_dir.iterdir() if p.is_dir()}
    missing_dirs = sorted(EXPECTED_PAIRWISE_DIRS - existing_dirs)
    if missing_dirs:
        fail(f"Missing pairwise model directories: {missing_dirs}", errors)

    for d in sorted(EXPECTED_PAIRWISE_DIRS):
        model_dir = pairwise_dir / d
        if not model_dir.exists():
            continue
        for req in ["metrics.json", "pair_results.jsonl"]:
            if not (model_dir / req).exists():
                fail(f"Missing pairwise file: data/pairwise/{d}/{req}", errors)
        pair_results = model_dir / "pair_results.jsonl"
        if pair_results.exists():
            with pair_results.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    required_keys = {
                        "pair_id",
                        "pair_type",
                        "distance",
                        "correct_position",
                        "prediction",
                        "is_correct",
                    }
                    missing = sorted(required_keys - set(row.keys()))
                    if missing:
                        fail(
                            f"Pairwise row missing required keys in data/pairwise/{d}/pair_results.jsonl line {i}: {missing}",
                            errors,
                        )
                    forbidden = sorted(
                        {
                            "alignment_fail",
                            "doi_first",
                            "doi_second",
                            "rank_first",
                            "rank_second",
                            "raw_response",
                        }.intersection(row.keys())
                    )
                    if forbidden:
                        fail(
                            f"Pairwise row leaked forbidden keys in data/pairwise/{d}/pair_results.jsonl line {i}: {forbidden}",
                            errors,
                        )
                    break


def _check_figure_index_contract(root: Path, errors: List[str]) -> None:
    index_path = root / "data" / "tables" / "FIGURE_DATA_INDEX.csv"
    if not index_path.exists():
        fail("Missing figure index: data/tables/FIGURE_DATA_INDEX.csv", errors)
        return

    _check_csv_row_width(index_path, errors)
    rows = _read_csv_rows(index_path, errors)

    figures = {str(r.get("figure", "")).strip() for r in rows}
    missing = sorted(EXPECTED_FIGURE_INDEX_ROWS - figures)
    if missing:
        fail(f"FIGURE_DATA_INDEX.csv is missing figure rows: {missing}", errors)


def _check_train_data_contract(root: Path, errors: List[str]) -> None:
    train_dir = root / "data" / "train_data"
    if not train_dir.exists():
        fail("Missing data/train_data directory", errors)
        return

    readme = train_dir / TRAIN_DATA_PLACEHOLDER_README
    if not readme.exists():
        fail("Missing data/train_data/README.md placeholder", errors)


def _check_benchmark_contract(path: Path, *, expected_rows: int, expected_subject: str | None, errors: List[str]) -> None:
    if not path.exists():
        fail(f"Missing benchmark file: {path}", errors)
        return

    rows_seen = 0
    with path.open("r", encoding="utf-8") as f:
        for i, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            rows_seen += 1
            rec = json.loads(line)
            if "entries" in rec:
                fail(f"Legacy benchmark prompt wrapper found in {path.name} line {i}; use top-level rq_with_context only", errors)
            if "rq_with_context" not in rec:
                fail(f"Missing rq_with_context field in {path.name} line {i}", errors)
            if "file_path" in rec:
                fail(f"PII leak: 'file_path' field present in {path.name} line {i}", errors)
            if _has_forbidden_keys(rec, FORBIDDEN_PUBLIC_JSON_KEYS):
                fail(f"Forbidden public-schema key found in {path.name} line {i}", errors)
            rec_text = json.dumps(rec)
            for marker in FORBIDDEN_PATH_SUBSTRINGS:
                if marker in rec_text:
                    fail(f"Host-specific path leak '{marker}' found in {path.name} line {i}", errors)
            if expected_subject is not None and rec.get("subject") != expected_subject:
                fail(
                    f"Unexpected subject in {path.name} line {i}: "
                    f"expected {expected_subject!r}, got {rec.get('subject')!r}",
                    errors,
                )
    if rows_seen != expected_rows:
        fail(f"Unexpected row count in {path.name}: expected {expected_rows}, got {rows_seen}", errors)


def main() -> int:
    here = Path(__file__).resolve()
    root = None
    for candidate in [here.parent, *here.parents]:
        if (candidate / "README.md").exists() and (candidate / "scripts").is_dir() and (candidate / "data").is_dir():
            root = candidate
            break
    if root is None:
        print("VALIDATION FAILED")
        print("- Could not locate package root (expected README.md + scripts/ + data/)")
        return 1

    errors: List[str] = []

    # 1) Folder shape and top-level contract
    required_top = {"README.md", "LICENSE", "manuscript", "figures", "data", "scripts", "requirements.txt"}
    optional_top = {"reproduced", ".gitignore", ".venv"}
    allowed_top = required_top | optional_top
    actual_top = {p.name for p in root.iterdir()}
    missing_top = sorted(required_top - actual_top)
    extra_top = sorted(actual_top - allowed_top)
    if missing_top or extra_top:
        fail(
            "Top-level mismatch. "
            f"Missing required: {missing_top or '[]'}; "
            f"unexpected extras: {extra_top or '[]'}; "
            f"actual: {sorted(actual_top)}",
            errors,
        )
    _check_readme_title_contract(root, errors)

    # 2) Manuscript contract
    manuscript_dir = root / "manuscript"
    if not manuscript_dir.exists():
        fail("Missing manuscript directory", errors)
    else:
        actual_manuscript = {p.name for p in manuscript_dir.iterdir() if p.is_file()}
        required_files = {"README.md"}
        missing_required = sorted(required_files - actual_manuscript)
        extra_files = sorted(actual_manuscript - required_files)
        if missing_required or extra_files:
            fail(
                "Manuscript placeholder mismatch. "
                f"Missing required: {missing_required or '[]'}; "
                f"unexpected extras: {extra_files or '[]'}",
                errors,
            )

    # 3) Figures: exactly 44 assets and sequential naming
    expected_figures: List[Path] = []
    for i in range(1, 8):
        expected_figures.append(root / "figures" / "main" / f"Figure{i}.png")
        expected_figures.append(root / "figures" / "main" / f"Figure{i}.pdf")
    for i in range(1, 9):
        expected_figures.append(root / "figures" / "extended_data" / f"ExtendedDataFigure{i}.png")
        expected_figures.append(root / "figures" / "extended_data" / f"ExtendedDataFigure{i}.pdf")
    for i in range(1, 8):
        expected_figures.append(root / "figures" / "supplementary" / f"SupplementaryFigure{i}.png")
        expected_figures.append(root / "figures" / "supplementary" / f"SupplementaryFigure{i}.pdf")

    missing_figs = [p for p in expected_figures if not p.exists()]
    if missing_figs:
        fail(f"Missing figure assets: {len(missing_figs)}", errors)

    actual_fig_files = []
    for sub in ["main", "extended_data", "supplementary"]:
        d = root / "figures" / sub
        if d.exists():
            actual_fig_files.extend([p for p in d.iterdir() if p.is_file() and p.suffix.lower() in {".png", ".pdf"}])
    if len(actual_fig_files) != 44:
        fail(f"Figure asset count mismatch. Expected 44, got {len(actual_fig_files)}", errors)

    # 4) Human ratings + reproducibility inputs are de-identified
    human_dir = root / "data" / "human_ratings"
    actual_human_root = {p.name for p in human_dir.iterdir()} if human_dir.exists() else set()
    extra_human_root = sorted(actual_human_root - ALLOWED_HUMAN_ROOT)
    if extra_human_root:
        fail(f"Unexpected files or directories in data/human_ratings: {extra_human_root}", errors)
    for name in sorted(REQUIRED_HUMAN_JSONL):
        _check_jsonl_no_forbidden_keys(human_dir / name, errors)
        _check_human_jsonl_schema(human_dir / name, errors)
        path = human_dir / name
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if _has_forbidden_keys(obj, FORBIDDEN_PUBLIC_JSON_KEYS):
                        fail(f"Forbidden public-schema key found in {path.relative_to(root)} line {i}", errors)
                        break
    repro_human_dir = human_dir / "reproducibility"
    actual_repro_files = {p.name for p in repro_human_dir.glob("*.jsonl")} if repro_human_dir.exists() else set()
    missing_repro = sorted(REQUIRED_REPRO_HUMAN_JSONL - actual_repro_files)
    extra_repro = sorted(actual_repro_files - REQUIRED_REPRO_HUMAN_JSONL)
    if missing_repro:
        fail(f"Missing reproducibility human files: {missing_repro}", errors)
    if extra_repro:
        fail(f"Unexpected reproducibility human files: {extra_repro}", errors)
    for name in sorted(REQUIRED_REPRO_HUMAN_JSONL):
        _check_jsonl_no_forbidden_keys(repro_human_dir / name, errors)
        _check_human_jsonl_schema(repro_human_dir / name, errors)
        path = repro_human_dir / name
        if path.exists():
            with path.open("r", encoding="utf-8") as f:
                for i, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue
                    obj = json.loads(line)
                    if _has_forbidden_keys(obj, FORBIDDEN_PUBLIC_JSON_KEYS):
                        fail(f"Forbidden public-schema key found in {path.relative_to(root)} line {i}", errors)
                        break

    # 5) Any optional manuscript markdown sources must be free of internal paths
    for path in sorted((root / "manuscript").glob("*.md")):
        text = path.read_text(encoding="utf-8")
        for marker in FORBIDDEN_MANUSCRIPT_SUBSTRINGS:
            if marker in text:
                fail(f"Internal path marker '{marker}' found in manuscript/{path.name}", errors)

    # 6) Strict exclusions
    forbidden_dirs = {"old", "backup"}
    for p in root.rglob("*"):
        if any(part in forbidden_dirs for part in p.parts):
            fail(f"Forbidden directory marker found: {p}", errors)
            break
        if p.suffix.lower() == ".csv" and "raw_csv" in str(p):
            fail(f"Raw CSV found in package: {p}", errors)
            break

    # 7) Script presence contract
    expected_scripts = {
        "reproduce_tables.sh",
        "reproduce_figures.sh",
        "validate_package.py",
    }
    scripts_dir = root / "scripts"
    if scripts_dir.exists():
        actual_scripts = {p.name for p in scripts_dir.iterdir() if p.is_file()}
        if actual_scripts != expected_scripts:
            fail(f"Scripts mismatch. Expected {sorted(expected_scripts)}, got {sorted(actual_scripts)}", errors)

        repro_dir = scripts_dir / "reproducibility"
        if repro_dir.exists():
            required_repro_scripts = {
                "build_consensus_figure.py",
                "build_core_result_figures.py",
                "build_economics_extension_figures.py",
                "build_extended_and_supplementary_figures.py",
                "build_frontier_diagnostics_figure.py",
                "build_release_tables_and_stats.py",
                "figure_style_policy.py",
                "_generate_frontier_diagnostics_figure.py",
                "_generate_core_result_figures.py",
                "_generate_consensus_figure.py",
                "_generate_extended_and_supplementary_figures.py",
                "_recompute_core_tables_and_stats.py",
                "_recompute_public_support_tables.py",
                "_recompute_public_tables_and_stats.py",
                "README.md",
            }
            actual_repro_files = {p.name for p in repro_dir.iterdir() if p.is_file()}
            missing_repro = sorted(required_repro_scripts - actual_repro_files)
            if missing_repro:
                fail(f"Missing reproducibility sources: {missing_repro}", errors)
            unexpected_repro = sorted(actual_repro_files - required_repro_scripts)
            if unexpected_repro:
                fail(f"Unexpected reproducibility files present: {unexpected_repro}", errors)
            vendor_analysis_dir = repro_dir / "vendor_analysis" / "analysis"
            if not vendor_analysis_dir.exists():
                fail("Missing vendored analysis directory: scripts/reproducibility/vendor_analysis/analysis", errors)
            else:
                required_vendor_scripts = {
                    "00_frontier_protocol_builder.py",
                    "02_flagship_model_performance.py",
                    "03_sft_model_performance.py",
                    "04_human_expert_performance.py",
                    "05_human_student_performance.py",
                    "07_ai_vs_human_comparison.py",
                    "09_interrater_reliability.py",
                    "10_calibration_analysis.py",
                    "15_ai_confidence_vs_human.py",
                    "23_gatekeeper_profile.py",
                    "24_student_reliability_vs_size.py",
                    "25_expert_reliability_vs_size.py",
                }
                required_vendor_utils = {
                    "__init__.py",
                    "constants.py",
                    "data_loader.py",
                    "frontier_protocol.py",
                    "metrics.py",
                    "statistical_tests.py",
                    "visualization.py",
                    "voting.py",
                }
                actual_vendor_scripts = {p.name for p in vendor_analysis_dir.glob("*.py")}
                missing_vendor_scripts = sorted(required_vendor_scripts - actual_vendor_scripts)
                if missing_vendor_scripts:
                    fail(f"Missing vendored analysis scripts: {missing_vendor_scripts}", errors)
                utils_dir = vendor_analysis_dir / "utils"
                if not utils_dir.exists():
                    fail("Missing vendored analysis utils directory", errors)
                else:
                    actual_vendor_utils = {p.name for p in utils_dir.glob("*.py")}
                    missing_vendor_utils = sorted(required_vendor_utils - actual_vendor_utils)
                    if missing_vendor_utils:
                        fail(f"Missing vendored analysis utils: {missing_vendor_utils}", errors)
            bad_names = sorted(p.name for p in repro_dir.glob("*_v[0-9]*.py"))
            if bad_names:
                fail(
                    "Version-tagged reproducibility scripts are not allowed. "
                    f"Rename these files: {bad_names}",
                    errors,
                )
    else:
        fail("Missing scripts directory", errors)

    # 8) Prediction package contract
    predictions_dir = root / "data" / "predictions"
    required_prediction_files = {
        "core_rq_short_transfer_predictions.jsonl",
        "economics_predictions.jsonl",
        "frontier_10models_8runs.jsonl",
        "pooled_management_economics_predictions.jsonl",
        "sft_predictions.jsonl",
        "sft_temporal_old_predictions.jsonl",
        "chat_predictions.jsonl",
        "rl_predictions.jsonl",
        "gemini_3_1_pro_standalone.jsonl",
    }
    expected_prediction_rows = {
        "core_rq_short_transfer_predictions.jsonl": 120,
        "economics_predictions.jsonl": 200,
        "frontier_10models_8runs.jsonl": 120,
        "pooled_management_economics_predictions.jsonl": 320,
        "sft_predictions.jsonl": 120,
        "sft_temporal_old_predictions.jsonl": 120,
        "chat_predictions.jsonl": 120,
        "rl_predictions.jsonl": 120,
        "gemini_3_1_pro_standalone.jsonl": 120,
    }
    for req in required_prediction_files:
        p_req = predictions_dir / req
        if not p_req.exists():
            fail(f"Missing prediction file: data/predictions/{req}", errors)
            continue
        n_rows = _count_nonempty_lines(p_req)
        expected_rows = expected_prediction_rows[req]
        if n_rows != expected_rows:
            fail(
                f"Unexpected row count in data/predictions/{req}: "
                f"expected {expected_rows}, got {n_rows}",
                errors,
            )

    prompt_variant_files = {
        "expert_prompt_predictions.jsonl",
        "simple_prompt_predictions.jsonl",
        "journal_prompt_predictions.jsonl",
    }
    prompt_dir = predictions_dir / "prompt_variants"
    for req in prompt_variant_files:
        if not (prompt_dir / req).exists():
            fail(f"Missing prompt-variant file: data/predictions/prompt_variants/{req}", errors)
            continue
        n_rows = _count_nonempty_lines(prompt_dir / req)
        if n_rows != 120:
            fail(f"Unexpected row count in data/predictions/prompt_variants/{req}: expected 120, got {n_rows}", errors)

    frontier_file = predictions_dir / "frontier_10models_8runs.jsonl"
    if frontier_file.exists():
        frontier_keys = _first_row_model_keys(frontier_file)
        if frontier_keys != EXPECTED_FRONTIER_MODELS:
            fail(
                "Frontier model set mismatch in frontier_10models_8runs.jsonl. "
                f"Expected {sorted(EXPECTED_FRONTIER_MODELS)}, got {sorted(frontier_keys)}",
                errors,
            )

    sft_file = predictions_dir / "sft_predictions.jsonl"
    if sft_file.exists():
        sft_keys = _first_row_model_keys(sft_file)
        missing = sorted(EXPECTED_SFT_KEYS - sft_keys)
        if missing:
            fail(f"SFT prediction file missing expected SFT model keys: {missing}", errors)

    old_sft_file = predictions_dir / "sft_temporal_old_predictions.jsonl"
    if old_sft_file.exists():
        old_sft_keys = _first_row_model_keys(old_sft_file)
        missing = sorted(EXPECTED_OLD_SFT_KEYS - old_sft_keys)
        if missing:
            fail(f"Old-trace SFT prediction file missing expected model keys: {missing}", errors)

    short_transfer_file = predictions_dir / "core_rq_short_transfer_predictions.jsonl"
    if short_transfer_file.exists():
        short_transfer_keys = _first_row_model_keys(short_transfer_file, eval_key="core_rq_short")
        if short_transfer_keys != EXPECTED_CORE_RQ_SHORT_TRANSFER_KEYS:
            fail(
                "Core short-input transfer file model set mismatch. "
                f"Expected {sorted(EXPECTED_CORE_RQ_SHORT_TRANSFER_KEYS)}, got {sorted(short_transfer_keys)}",
                errors,
            )
        _check_core_rq_short_transfer_schema(short_transfer_file, errors)

    gemini31_file = predictions_dir / "gemini_3_1_pro_standalone.jsonl"
    if gemini31_file.exists():
        gemini31_keys = _first_row_model_keys(gemini31_file)
        if GEMINI_31_KEY not in gemini31_keys:
            fail(
                "Gemini 3.1 standalone file does not contain expected model key "
                f"'{GEMINI_31_KEY}'",
                errors,
            )

    economics_file = predictions_dir / "economics_predictions.jsonl"
    if economics_file.exists():
        economics_keys = _first_row_model_keys(economics_file)
        if economics_keys != EXPECTED_ECONOMICS_EXTENSION_KEYS:
            fail(
                "Economics extension model set mismatch. "
                f"Expected {sorted(EXPECTED_ECONOMICS_EXTENSION_KEYS)}, got {sorted(economics_keys)}",
                errors,
            )

    pooled_file = predictions_dir / "pooled_management_economics_predictions.jsonl"
    if pooled_file.exists():
        pooled_keys = _first_row_model_keys(pooled_file)
        if pooled_keys != EXPECTED_POOLED_FIELD_KEYS:
            fail(
                "Pooled management+economics model set mismatch. "
                f"Expected {sorted(EXPECTED_POOLED_FIELD_KEYS)}, got {sorted(pooled_keys)}",
                errors,
            )

    # Ensure prediction files are free of direct identifier keys
    for pred_file in sorted(predictions_dir.rglob("*.jsonl")):
        _check_jsonl_no_forbidden_keys(pred_file, errors)
        with pred_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                obj = json.loads(line)
                if _has_forbidden_keys(obj, FORBIDDEN_PUBLIC_JSON_KEYS):
                    fail(f"Forbidden public-schema key found in {pred_file.relative_to(root)} line {i}", errors)
                    break

    # 9) Table package contract
    tables_dir = root / "data" / "tables"
    required_table_indexes = {"TABLE_INDEX.csv", "FIGURE_DATA_INDEX.csv"}
    if tables_dir.exists():
        table_files = {p.name for p in tables_dir.glob("*.csv")}
        missing_indexes = sorted(required_table_indexes - table_files)
        if missing_indexes:
            fail(f"Missing table index files: {missing_indexes}", errors)
        for index_name in sorted(required_table_indexes):
            _check_csv_row_width(tables_dir / index_name, errors)

        # Only allow Txx_* plus index files in external package
        for name in sorted(table_files):
            if name in required_table_indexes:
                continue
            if not name.startswith("T"):
                fail(f"Non-external table naming found: {name}", errors)
            if name.startswith("table_"):
                fail(f"Internal table naming leaked into package: {name}", errors)

        _check_tables_no_forbidden_columns(tables_dir, errors)
        _check_ci_contract(tables_dir, errors)
    else:
        fail("Missing data/tables directory", errors)

    shipped_stats = {
        "S13_Figure6NumbersAudit.json": None,
        "S14_ED2PairwiseRawPValues.json": None,
        "S15_CoreRQShortTransferStats.json": "ExtendedDataFigure7",
        "S16_EconomicsExtensionStats.json": "Figure7",
        "S17_PooledFieldExtensionStats.json": "ExtendedDataFigure8",
        "S18_CrossFieldTransferStats.json": "SupplementaryFigure7",
    }
    for filename, figure_name in shipped_stats.items():
        path = root / "data" / "statistics" / filename
        if not path.exists():
            fail(f"Missing shipped statistics artifact: data/statistics/{filename}", errors)
            continue
        try:
            stats_obj = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            fail(f"Invalid JSON in data/statistics/{filename}: {exc}", errors)
            continue
        if figure_name is not None and stats_obj.get("figure") != figure_name:
            fail(f"{filename} has unexpected figure identifier", errors)

    # 10) Figure trace index + pairwise contract (ED2)
    _check_figure_index_contract(root, errors)
    _check_pairwise_contract(root, errors)

    # 11) Training-data release contract
    _check_train_data_contract(root, errors)

    # 12) Benchmark data integrity
    benchmark_file = root / "data" / "benchmark" / "benchmark_articles_120.jsonl"
    if benchmark_file.exists():
        _check_benchmark_contract(
            benchmark_file,
            expected_rows=120,
            expected_subject=None,
            errors=errors,
        )
        journals: set[str] = set()
        domains: set[str] = set()
        with benchmark_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                rec = json.loads(line)
                if "entries" in rec:
                    fail(f"Legacy benchmark prompt wrapper found in line {i}; use top-level rq_with_context only", errors)
                if "rq_with_context" not in rec:
                    fail(f"Missing rq_with_context field in benchmark line {i}", errors)
                if "file_path" in rec:
                    fail(f"PII leak: 'file_path' field present in benchmark line {i}", errors)
                if _has_forbidden_keys(rec, FORBIDDEN_PUBLIC_JSON_KEYS):
                    fail(f"Forbidden public-schema key found in benchmark line {i}", errors)
                rec_text = json.dumps(rec)
                for marker in FORBIDDEN_PATH_SUBSTRINGS:
                    if marker in rec_text:
                        fail(f"Host-specific path leak '{marker}' found in benchmark line {i}", errors)
                domain_str = rec.get("domain", "")
                for ch in domain_str:
                    if '\u4e00' <= ch <= '\u9fff':
                        fail(f"Chinese characters in domain field at benchmark line {i}: {domain_str}", errors)
                        break
                journals.add(rec.get("journal", ""))
                for d in domain_str.split(", "):
                    domains.add(d.strip())
        if len(journals) != 17:
            fail(f"Expected 17 unique journals in benchmark, found {len(journals)}", errors)
        if len(domains) != 15:
            fail(f"Expected 15 unique domains in benchmark, found {len(domains)}", errors)
    economics_benchmark_file = root / "data" / "benchmark" / "economics_benchmark_articles_200.jsonl"
    _check_benchmark_contract(
        economics_benchmark_file,
        expected_rows=200,
        expected_subject="economics",
        errors=errors,
    )

    # 13) No stale compatibility directories
    stale_dirs = ["data/model_predictions", "data/repro_compat", "data/pair_wise_outcome", "data/articles"]
    for d in stale_dirs:
        if (root / d).exists():
            fail(f"Stale compatibility directory still present: {d}", errors)

    # 14) No host-specific path leaks in JSONL across the package
    for jsonl_file in sorted(root.rglob("*.jsonl")):
        with jsonl_file.open("r", encoding="utf-8") as f:
            for i, line in enumerate(f, start=1):
                for marker in FORBIDDEN_PATH_SUBSTRINGS:
                    if marker in line:
                        fail(
                            f"Host-specific path leak: '{marker}' in {jsonl_file.relative_to(root)} line {i}",
                            errors,
                        )
                        break
                else:
                    continue
                break

    # 15) No Python cache artifacts in the public package
    cache_dirs = sorted(
        p.relative_to(root)
        for p in root.rglob("__pycache__")
        if ".venv" not in p.relative_to(root).parts
    )
    if cache_dirs:
        fail(f"Python cache directories should not be shipped: {cache_dirs}", errors)
    pyc_files = sorted(
        p.relative_to(root)
        for p in root.rglob("*.pyc")
        if ".venv" not in p.relative_to(root).parts
    )
    if pyc_files:
        fail(f"Python bytecode files should not be shipped: {pyc_files}", errors)
    ds_store_files = sorted(
        p.relative_to(root)
        for p in root.rglob(".DS_Store")
        if ".venv" not in p.relative_to(root).parts
    )
    if ds_store_files:
        fail(f"Finder metadata files should not be shipped: {ds_store_files}", errors)

    # 16) Headline numbers spot-check
    t04 = root / "data" / "tables" / "T04_AIvsHumanSummary.csv"
    if t04.exists():
        t04_rows = _read_csv_rows(t04, errors)
        by_eval = {str(r.get("Evaluator", "")): r for r in t04_rows}
        sft_row = by_eval.get("SFT 2-Model Ensemble")
        if sft_row:
            sft_acc = float(sft_row.get("Accuracy (%)", 0))
            if abs(sft_acc - 60.8) > 0.5:
                fail(f"T04 SFT 2-Model Ensemble accuracy {sft_acc}% deviates from expected 60.8%", errors)
        frontier_row = by_eval.get("Flagship Average (11 models)")
        if frontier_row:
            frontier_acc = float(frontier_row.get("Accuracy (%)", 0))
            if abs(frontier_acc - 31.0) > 2.0:
                fail(f"T04 Flagship Average accuracy {frontier_acc}% deviates from expected ~31.0%", errors)

    s10 = root / "data" / "statistics" / "S10_SignificanceStats.json"
    if s10.exists():
        s10_data = json.loads(s10.read_text(encoding="utf-8"))
        best_model = s10_data.get("best_flagship_model", "")
        if not str(best_model).strip():
            fail("S10 best_flagship_model is blank", errors)

    if errors:
        print("VALIDATION FAILED")
        for e in errors:
            print(f"- {e}")
        return 1

    print("VALIDATION PASSED")
    print("- Folder shape matches required structure")
    print("- README paper title matches the release title contract")
    print("- Manuscript placeholder contract satisfied")
    print("- Figure naming/count checks satisfied (44 assets)")
    print("- Human ratings and reproducibility inputs are de-identified")
    print("- Manuscript has no internal path markers")
    print("- Exclusion rules satisfied")
    print("- Required scripts and vendored analysis sources are present")
    print("- Prediction package contract satisfied (frontier/SFT/chat/RL/short-transfer/economics)")
    print("- Public JSON schemas stripped of compatibility-only fields")
    print("- Table index and naming contract satisfied")
    print("- No forbidden identifier columns in packaged tables")
    print("- CI field contract satisfied for reported analyses")
    print("- Figure trace index contract satisfied (Main/ED/SI)")
    print("- Pairwise data contract satisfied (ED2)")
    print("- Training-data placeholder present")
    print("- Benchmark integrity: management + economics files validated, no PII, no host paths")
    print("- No stale compatibility directories")
    print("- Requirements metadata present")
    print("- No host-specific paths in any JSONL file")
    print("- No Python cache artifacts shipped")
    print("- No Finder metadata files shipped")
    print("- Headline numbers spot-check passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
