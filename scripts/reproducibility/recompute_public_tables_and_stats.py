#!/usr/bin/env python3
"""Recompute the public submission tables/statistics from package-local inputs."""

from __future__ import annotations

import csv
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
REPRO_DIR = ROOT / "scripts" / "reproducibility"
VENDOR_ANALYSIS_DIR = REPRO_DIR / "vendor_analysis" / "analysis"
TABLES_DIR = ROOT / "data" / "tables"
STATS_DIR = ROOT / "data" / "statistics"


def run_python(script: Path, *, env: dict[str, str] | None = None) -> None:
    merged_env = os.environ.copy()
    merged_env.setdefault("PYTHONNOUSERSITE", "1")
    merged_env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    if env:
        merged_env.update(env)
    subprocess.run([sys.executable, str(script)], cwd=ROOT, env=merged_env, check=True)


def copy_file(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, dst)


def write_t10_anonymized(src: Path, dst: Path) -> None:
    with src.open("r", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))

    fieldnames = ["expert_id", "n_articles", "n_correct", "accuracy"]
    with dst.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for idx, row in enumerate(rows, start=1):
            writer.writerow(
                {
                    "expert_id": f"Expert_{idx:03d}",
                    "n_articles": row.get("n_articles", ""),
                    "n_correct": row.get("n_correct", ""),
                    "accuracy": row.get("accuracy", ""),
                }
            )


def normalize_t23(src: Path, dst: Path) -> None:
    with src.open("r", encoding="utf-8", newline="") as fh:
        reader = csv.DictReader(fh)
        fieldnames = list(reader.fieldnames or [])
        rows = list(reader)

    for row in rows:
        for key, value in list(row.items()):
            if value == "nan":
                row[key] = ""

    with dst.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    temp_parent = ROOT / "reproduced"
    temp_parent.mkdir(parents=True, exist_ok=True)
    temp_dir = Path(tempfile.mkdtemp(prefix="table_rebuild_", dir=temp_parent))
    keep_temp = os.environ.get("KEEP_VENDOR_RESULTS") == "1"

    try:
        env = os.environ.copy()
        env["RESULTS_ROOT"] = str(temp_dir)

        run_python(VENDOR_ANALYSIS_DIR / "00_frontier_protocol_builder.py", env=env)
        run_python(REPRO_DIR / "recompute_core_tables_and_stats.py")
        run_python(REPRO_DIR / "recompute_public_support_tables.py")

        for name in [
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
        ]:
            run_python(VENDOR_ANALYSIS_DIR / name, env=env)

        table_map = {
            "table_frontier_protocol_audit.csv": TABLES_DIR / "T01_FrontierProtocolAudit.csv",
            "table3_flagship_performance.csv": TABLES_DIR / "T02_FlagshipPerformance.csv",
            "table4_sft_performance.csv": TABLES_DIR / "T03_SFTPerformance.csv",
            "table8_ai_vs_human.csv": TABLES_DIR / "T04_AIvsHumanSummary.csv",
            "table23_gatekeeper_profile_main.csv": TABLES_DIR / "T06_GatekeeperProfileMain.csv",
            "table23_gatekeeper_per_class.csv": TABLES_DIR / "T07_GatekeeperPerClass.csv",
            "table23_human_variants_summary.csv": TABLES_DIR / "T08_HumanVariantsSummary.csv",
            "table_s3_monte_carlo.csv": TABLES_DIR / "T11_MonteCarloMatchedPanel.csv",
            "table_s12_confidence_comparison.csv": TABLES_DIR / "T12_ConfidenceComparison.csv",
            "table_s7_calibration_metrics.csv": TABLES_DIR / "T13_CalibrationMetrics.csv",
            "table_s13_all_models_summary.csv": TABLES_DIR / "T17_AllModelsSummary.csv",
            "table_s_chat_models.csv": TABLES_DIR / "T18_ChatModelSummary.csv",
            "table_s24_student_reliability_vs_size.csv": TABLES_DIR / "T19_StudentReliabilityByPanelSize.csv",
        }
        for src_name, dst_path in table_map.items():
            copy_file(temp_dir / "tables" / src_name, dst_path)

        write_t10_anonymized(
            temp_dir / "tables" / "table_s2_individual_expert.csv",
            TABLES_DIR / "T10_IndividualExpertAccuracy_Anonymized.csv",
        )
        normalize_t23(
            temp_dir / "tables" / "table_s25_expert_reliability_vs_size.csv",
            TABLES_DIR / "T20_ExpertReliabilityByPanelSize.csv",
        )

        stats_map = {
            "02_flagship.json": STATS_DIR / "S01_FlagshipStats.json",
            "03_sft.json": STATS_DIR / "S02_SFTStats.json",
            "04_expert.json": STATS_DIR / "S03_ExpertStats.json",
            "05_student.json": STATS_DIR / "S04_StudentStats.json",
            "07_ai_vs_human.json": STATS_DIR / "S05_AIvsHumanStats.json",
            "09_interrater.json": STATS_DIR / "S06_InterraterStats.json",
            "10_calibration.json": STATS_DIR / "S07_CalibrationStats.json",
            "15_ai_confidence_vs_human.json": STATS_DIR / "S11_ConfidenceStats.json",
            "23_gatekeeper_profile.json": STATS_DIR / "S12_GatekeeperStats.json",
        }
        for src_name, dst_path in stats_map.items():
            copy_file(temp_dir / "statistics" / src_name, dst_path)

        print("Recomputed public submission tables/statistics from package-local inputs.")
    finally:
        if not keep_temp and temp_dir.exists():
            shutil.rmtree(temp_dir, ignore_errors=True)


if __name__ == "__main__":
    main()
