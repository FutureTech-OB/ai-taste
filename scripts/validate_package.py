#!/usr/bin/env python3
"""Validate the public reproducibility release package."""

from __future__ import annotations

import csv
import json
import re
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

TEXT_SUFFIXES = {".bib", ".csv", ".json", ".jsonl", ".md", ".py", ".txt"}

REQUIRED_FILES = [
    "README.md",
    "LICENSE",
    "requirements.txt",
    ".gitignore",
    "data/README.md",
    "data/management_deep_probe/README.md",
    "data/management_deep_probe/benchmark/README.md",
    "data/management_deep_probe/predictions/README.md",
    "data/management_deep_probe/human_ratings/README.md",
    "data/management_deep_probe/pairwise/README.md",
    "data/management_deep_probe/statistics/README.md",
    "data/management_deep_probe/support_tables/README.md",
    "data/cross_field_standardized/README.md",
    "data/cross_field_standardized/benchmarks/README.md",
    "data/cross_field_standardized/predictions/README.md",
    "data/cross_field_standardized/journal_tiers/README.md",
    "data/supplementary_tables/README.md",
    "figures/main/README.md",
    "figures/supplementary/README.md",
    "scripts/README.md",
    "scripts/reproduce_figures.py",
    "scripts/reproduce_tables.py",
    "scripts/build_release_manifest.py",
    "scripts/validate_package.py",
    "scripts/reproducibility/README.md",
    "scripts/reproducibility/_common.py",
    "scripts/reproducibility/compute_s19_pairwise_sft_kappa.py",
    "scripts/reproducibility/compute_s20_sft_consensus_per_class.py",
    "scripts/reproducibility/compute_s21_ai_human_complementarity.py",
    "scripts/reproducibility/compute_s22_mcnemar_compendium.py",
    "scripts/reproducibility/compute_s23_headroom_captured.py",
    "scripts/reproducibility/smoke_test_management_stats.py",
    "docs/README.md",
    "docs/availability.md",
    "docs/data_dictionary.md",
    "docs/reproducibility.md",
    "docs/reproducibility_checklist.md",
    "docs/release_manifest.csv",
]

REQUIRED_DIRS = [
    "figures/main",
    "figures/supplementary",
    "figures/provenance",
    "data/management_deep_probe/benchmark",
    "data/management_deep_probe/predictions",
    "data/management_deep_probe/human_ratings",
    "data/management_deep_probe/pairwise",
    "data/management_deep_probe/statistics",
    "data/management_deep_probe/support_tables",
    "data/cross_field_standardized/benchmarks",
    "data/cross_field_standardized/predictions",
    "data/cross_field_standardized/statistics",
    "data/cross_field_standardized/journal_tiers",
    "data/supplementary_tables",
    "data/figure_support",
    "docs",
    "scripts",
]

MAIN_FIGURES = [
    "figures/main/Figure1.png",
    "figures/main/Figure2.png",
    "figures/main/Figure2.pdf",
    "figures/main/Figure3.png",
    "figures/main/Figure3.pdf",
    "figures/main/Figure4.png",
    "figures/main/Figure4.pdf",
    "figures/main/Figure5.png",
    "figures/main/Figure5.pdf",
]

SUPPLEMENTARY_FIGURES = [
    f"figures/supplementary/SupplementaryFigure{i}.{ext}"
    for i in range(1, 12)
    for ext in ("png", "pdf")
]

PROVENANCE_FILES = [
    "figures/provenance/fig1_caption.md",
    "figures/provenance/fig2_caption.md",
    "figures/provenance/fig2_provenance.csv",
    "figures/provenance/fig2_panelC_stats.json",
    "figures/provenance/fig2_panelD_stats.json",
    "figures/provenance/fig3_caption.md",
    "figures/provenance/fig3_provenance.csv",
    "figures/provenance/fig4_caption.md",
    "figures/provenance/fig4_provenance.csv",
    "figures/provenance/fig5_caption.md",
    "figures/provenance/fig5_provenance.csv",
] + [
    item
    for i in range(1, 12)
    for item in (
        f"figures/provenance/sf{i}_caption.md",
        f"figures/provenance/sf{i}_provenance.csv",
    )
]

CROSS_FIELDS = [
    "business_finance",
    "communication",
    "economics",
    "political_science",
    "psychology_multidisciplinary",
    "public_administration",
    "sociology",
]

MANAGEMENT_PREDICTIONS = [
    "chat_predictions.jsonl",
    "core_rq_short_transfer_predictions.jsonl",
    "frontier_10models_8runs.jsonl",
    "frontier_thinking_11models_singleshot.jsonl",
    "frontier_prompt_sensitivity_expert_12models_8runs.jsonl",
    "frontier_prompt_sensitivity_simple_8models_8runs.jsonl",
    "frontier_prompt_sensitivity_journal_7models_8runs.jsonl",
    "rl_predictions.jsonl",
    "sft_predictions.jsonl",
    "sft_temporal_old_predictions.jsonl",
]

DATA_FILES = [
    "data/management_deep_probe/benchmark/management_benchmark_120.jsonl",
    "data/management_deep_probe/human_ratings/expert_ratings_deidentified.jsonl",
    "data/management_deep_probe/human_ratings/student_ratings_deidentified.jsonl",
    "data/management_deep_probe/human_ratings/reproducibility/expert_reproducibility.jsonl",
    "data/management_deep_probe/human_ratings/reproducibility/expert_reproducibility_filtered.jsonl",
    "data/management_deep_probe/human_ratings/reproducibility/student_reproducibility.jsonl",
    "data/management_deep_probe/human_ratings/reproducibility/student_reproducibility_filtered.jsonl",
    "data/management_deep_probe/support_tables/TABLE_INDEX.csv",
    "data/management_deep_probe/statistics/S01_FlagshipStats.json",
    "data/management_deep_probe/statistics/S05_AIvsHumanStats.json",
    "data/management_deep_probe/statistics/S14_PairwiseRawPValues.json",
    "data/management_deep_probe/statistics/S15_CoreRQShortTransferStats.json",
    "data/management_deep_probe/statistics/S19_PairwiseSFTKappa.json",
    "data/management_deep_probe/statistics/S20_SFTConsensusPerClass.json",
    "data/management_deep_probe/statistics/S21_AIHumanComplementarity.json",
    "data/management_deep_probe/statistics/S22_McNemarCompendium.json",
    "data/management_deep_probe/statistics/S23_HeadroomCaptured.json",
    "data/cross_field_standardized/predictions/economics_predictions.jsonl",
    "data/cross_field_standardized/predictions/management_sft_transfer_to_seven_fields.jsonl",
    "data/cross_field_standardized/predictions/seven_field_core_predictions.jsonl",
    "data/cross_field_standardized/statistics/benchmark_summary_seven_fields.csv",
    "data/cross_field_standardized/statistics/cross_field_calibration_data.json",
    "data/cross_field_standardized/statistics/ensemble_cross_field_results.csv",
    "data/cross_field_standardized/statistics/training_dataset_sizes_seven_fields.csv",
    "data/cross_field_standardized/statistics/training_dataset_sizes_six_additional_fields.csv",
    "data/cross_field_standardized/statistics/rank_split_counts_six_additional_fields.csv",
    "data/cross_field_standardized/predictions/gpt52_chat_logp_all_fields.jsonl",
    "data/cross_field_standardized/statistics/gpt52_chat_logp_selective_prediction.csv",
    "data/cross_field_standardized/statistics/gpt52_chat_logp_band_data.json",
    "data/cross_field_standardized/predictions/gpt55_chat_logp_all_fields.jsonl",
    "data/cross_field_standardized/predictions/gpt55_high_reasoning_item_summary.jsonl",
    "data/cross_field_standardized/predictions/gpt55_high_reasoning_runs.jsonl",
    "data/cross_field_standardized/statistics/gpt55_chat_logp_selective_prediction.csv",
    "data/cross_field_standardized/statistics/gpt55_chat_logp_band_data.json",
    "data/cross_field_standardized/statistics/gpt55_high_reasoning_field_summary.csv",
    "data/cross_field_standardized/statistics/gpt55_reasoning_comparison_stats.json",
    "data/cross_field_standardized/journal_tiers/journal_rank_lookup.csv",
    "data/supplementary_tables/TABLE_INDEX.csv",
    "data/supplementary_tables/ST2b_gpt55_all_field_comparison.csv",
    "data/supplementary_tables/ST21_cross_architecture_agreement_management.csv",
    "data/supplementary_tables/ST22_sft_consensus_per_class_management.csv",
    "data/supplementary_tables/ST23_ai_human_error_complementarity_management.csv",
    "data/supplementary_tables/ST24_management_mcnemar_compendium.csv",
    "data/figure_support/FIGURE_DATA_INDEX.csv",
] + [
    f"data/management_deep_probe/predictions/{name}" for name in MANAGEMENT_PREDICTIONS
] + [
    f"data/cross_field_standardized/benchmarks/{field}_benchmark_200.jsonl" for field in CROSS_FIELDS
]

EXPECTED_JSONL_COUNTS = {
    "data/management_deep_probe/benchmark/management_benchmark_120.jsonl": 120,
    "data/management_deep_probe/human_ratings/expert_ratings_deidentified.jsonl": 120,
    "data/management_deep_probe/human_ratings/student_ratings_deidentified.jsonl": 120,
    "data/management_deep_probe/human_ratings/reproducibility/expert_reproducibility.jsonl": 120,
    "data/management_deep_probe/human_ratings/reproducibility/expert_reproducibility_filtered.jsonl": 120,
    "data/management_deep_probe/human_ratings/reproducibility/student_reproducibility.jsonl": 120,
    "data/management_deep_probe/human_ratings/reproducibility/student_reproducibility_filtered.jsonl": 120,
    "data/cross_field_standardized/predictions/economics_predictions.jsonl": 200,
    "data/cross_field_standardized/predictions/management_sft_transfer_to_seven_fields.jsonl": 5600,
    "data/cross_field_standardized/predictions/seven_field_core_predictions.jsonl": 11200,
    "data/cross_field_standardized/predictions/gpt52_chat_logp_all_fields.jsonl": 1520,
    "data/cross_field_standardized/predictions/gpt55_chat_logp_all_fields.jsonl": 1520,
    "data/cross_field_standardized/predictions/gpt55_high_reasoning_item_summary.jsonl": 1520,
    "data/cross_field_standardized/predictions/gpt55_high_reasoning_runs.jsonl": 12160,
}
EXPECTED_JSONL_COUNTS.update(
    {f"data/management_deep_probe/predictions/{name}": 120 for name in MANAGEMENT_PREDICTIONS}
)
EXPECTED_JSONL_COUNTS.update(
    {f"data/cross_field_standardized/benchmarks/{field}_benchmark_200.jsonl": 200 for field in CROSS_FIELDS}
)

PAIRWISE_DIRS = [
    "baseline_gpt4_1",
    "frontier_gemini3_1_pro",
    "frontier_gpt5_2_high",
    "frontier_grok4_1_fast",
    "sft_gpt4_1",
]

FORBIDDEN_PATTERNS = [
    (re.compile("/" + "Users/"), "local macOS home path"),
    (re.compile("C:" + r"\\\\Users", re.IGNORECASE), "local Windows home path"),
    (re.compile("Nature" + "_Submission" + "_Package"), "template-package path"),
    (re.compile(r"\\.repo" + "_release"), "nonpublic release-workspace path"),
    (re.compile(r"\.\./\.\./analysis/"), "nonpublic upward-relative analysis path"),
    (re.compile("science-repro" + "ducibility/data/"), "non-package-root data path"),
    (re.compile("all" + "_subjects"), "local training-output path"),
    (re.compile("data/" + "model_predictions"), "legacy local prediction path"),
    (re.compile(":per" + "sonal:"), "provider-local fine-tune identifier"),
    (re.compile("ft:" + "gpt"), "provider-local fine-tune identifier"),
    (re.compile("OPENAI" + "_API_KEY|" + r"sk-[A-Za-z0-9_-]{20,}"), "API secret marker"),
    (
        re.compile(
            "Cla" + "ude/" + "Co" + "dex|" + r"\bTO" + r"DO\b|\bhand" + r"off\b|tempor" + r"ary file",
            re.IGNORECASE,
        ),
        "development note wording",
    ),
]

JOURNAL_TARGET_PATTERNS = [
    (re.compile("Sci" + "ence reproducibility package", re.IGNORECASE), "journal-target package label"),
    (re.compile("Sci" + "ence release-package", re.IGNORECASE), "journal-target package label"),
    (re.compile("Sci" + "ence supplementary", re.IGNORECASE), "journal-target table label"),
    (re.compile("Sci" + "ence table", re.IGNORECASE), "journal-target table label"),
    (re.compile("Sci" + "ence manuscript", re.IGNORECASE), "journal-target manuscript label"),
    (re.compile("data/" + "science"), "old data path"),
    (re.compile("science" + "_submission"), "old manuscript filename"),
    (re.compile("release_package" + "_audit"), "old report filename"),
]

JOURNAL_TARGET_CHECK_PATHS = {
    "README.md",
    "data/README.md",
    "scripts/README.md",
    "scripts/build_release_manifest.py",
    "scripts/reproduce_figures.py",
    "scripts/reproduce_tables.py",
    "scripts/validate_package.py",
    "docs/README.md",
    "docs/availability.md",
    "docs/data_dictionary.md",
    "docs/deidentification.md",
    "docs/journal_tier_methodology.md",
    "docs/reproducibility.md",
    "docs/reproducibility_checklist.md",
}


def rel(path: Path) -> str:
    return path.relative_to(ROOT).as_posix()


def count_jsonl(path: Path) -> int:
    with path.open() as handle:
        return sum(1 for line in handle if line.strip())


def iter_public_text_files() -> list[Path]:
    skip_parts = {".git", ".venv", "__pycache__", "reproduced"}
    files: list[Path] = []
    for path in ROOT.rglob("*"):
        if not path.is_file():
            continue
        relative = path.relative_to(ROOT)
        if any(part in skip_parts for part in relative.parts):
            continue
        if path.suffix.lower() in TEXT_SUFFIXES:
            files.append(path)
    return files


def validate_paths(failures: list[str]) -> None:
    for dirname in REQUIRED_DIRS:
        if not (ROOT / dirname).is_dir():
            failures.append(f"missing directory: {dirname}")

    for filename in REQUIRED_FILES + MAIN_FIGURES + SUPPLEMENTARY_FIGURES + PROVENANCE_FILES + DATA_FILES:
        path = ROOT / filename
        if not path.is_file():
            failures.append(f"missing file: {filename}")
        elif path.stat().st_size == 0:
            failures.append(f"empty file: {filename}")

    for path in ROOT.rglob("*"):
        relative = path.relative_to(ROOT).as_posix()
        if ".git/" in relative:
            continue
        if any(
            bad in relative
            for bad in (
                "science" + "_predictions",
                "science" + "_tables",
                "science" + "_statistics",
                "science" + "_submission",
                "release_package" + "_audit",
            )
        ):
            failures.append(f"old journal-target path remains: {relative}")


def validate_counts(failures: list[str]) -> None:
    for filename, expected in EXPECTED_JSONL_COUNTS.items():
        path = ROOT / filename
        if path.exists():
            observed = count_jsonl(path)
            if observed != expected:
                failures.append(f"row count mismatch: {filename} has {observed}, expected {expected}")

    for dirname in PAIRWISE_DIRS:
        metrics = ROOT / "data/management_deep_probe/pairwise" / dirname / "metrics.json"
        pairs = ROOT / "data/management_deep_probe/pairwise" / dirname / "pair_results.jsonl"
        if not metrics.is_file():
            failures.append(f"missing pairwise metrics: {rel(metrics)}")
        if not pairs.is_file():
            failures.append(f"missing pairwise results: {rel(pairs)}")
        elif count_jsonl(pairs) != 300:
            failures.append(f"pairwise row count mismatch: {rel(pairs)}")

    index = ROOT / "data/supplementary_tables/TABLE_INDEX.csv"
    if index.exists():
        with index.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        table_ids = {row["table_id"] for row in rows}
        expected_ids = {f"ST{i}" for i in range(1, 25)}
        missing = sorted(expected_ids - table_ids)
        if missing:
            failures.append(f"supplementary table index missing ids: {', '.join(missing)}")
        for row in rows:
            table_path = ROOT / "data/supplementary_tables" / row["filename"]
            if not table_path.is_file():
                failures.append(f"supplementary table listed but missing: {rel(table_path)}")

    figure_index = ROOT / "data/figure_support/FIGURE_DATA_INDEX.csv"
    if figure_index.exists():
        with figure_index.open(newline="") as handle:
            figures = {row["figure_id"] for row in csv.DictReader(handle)}
        expected_figures = {"Figure1", "Figure2", "Figure3", "Figure4", "Figure5"} | {
            f"SF{i}" for i in range(1, 12)
        }
        missing = sorted(expected_figures - figures)
        if missing:
            failures.append(f"figure-data index missing ids: {', '.join(missing)}")


def validate_cross_field_schema(failures: list[str]) -> None:
    expected_keys = [
        "benchmark_id",
        "field",
        "field_display",
        "title",
        "journal",
        "published_year",
        "rank",
        "rq_with_context",
        "core_rq_short",
        "source_baseid",
    ]
    for field in CROSS_FIELDS:
        path = ROOT / f"data/cross_field_standardized/benchmarks/{field}_benchmark_200.jsonl"
        if not path.exists():
            continue
        rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        for index, row in enumerate(rows, start=1):
            if list(row.keys()) != expected_keys:
                failures.append(f"benchmark schema mismatch: {rel(path)}:{index}")
                break
            if row.get("field") != field:
                failures.append(f"benchmark field mismatch: {rel(path)}:{index}")
                break
            if not row.get("rq_with_context"):
                failures.append(f"missing released prompt text: {rel(path)}:{index}")
                break
        tiers = {row.get("rank") for row in rows}
        if tiers != {"exceptional", "strong", "fair", "limited"}:
            failures.append(f"benchmark tiers incomplete: {rel(path)}")


def validate_json(failures: list[str]) -> None:
    for path in iter_public_text_files():
        if path.suffix.lower() == ".json":
            try:
                json.loads(path.read_text())
            except Exception as exc:  # pragma: no cover - diagnostic path
                failures.append(f"invalid JSON: {rel(path)} ({exc})")
        elif path.suffix.lower() == ".jsonl":
            with path.open() as handle:
                for line_no, line in enumerate(handle, start=1):
                    if not line.strip():
                        continue
                    try:
                        json.loads(line)
                    except Exception as exc:  # pragma: no cover - diagnostic path
                        failures.append(f"invalid JSONL: {rel(path)}:{line_no} ({exc})")
                        break


def validate_text_hygiene(failures: list[str]) -> None:
    for path in iter_public_text_files():
        text = path.read_text(errors="ignore")
        for pattern, description in FORBIDDEN_PATTERNS:
            match = pattern.search(text)
            if match:
                failures.append(f"{description}: {rel(path)}")
                break

        relative = rel(path)
        if relative in JOURNAL_TARGET_CHECK_PATHS:
            for pattern, description in JOURNAL_TARGET_PATTERNS:
                if pattern.search(text):
                    failures.append(f"{description}: {relative}")
                    break


def main() -> int:
    failures: list[str] = []
    validate_paths(failures)
    validate_counts(failures)
    validate_cross_field_schema(failures)
    validate_json(failures)
    validate_text_hygiene(failures)

    if failures:
        print("Release-package validation failed:")
        for failure in failures:
            print(f"- {failure}")
        return 1

    checked_files = len(REQUIRED_FILES + MAIN_FIGURES + SUPPLEMENTARY_FIGURES + PROVENANCE_FILES + DATA_FILES)
    print("Release-package validation passed.")
    print(f"Checked {checked_files} required files, {len(REQUIRED_DIRS)} directories, JSON syntax, row counts, seven-field benchmark schema, and path hygiene.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
