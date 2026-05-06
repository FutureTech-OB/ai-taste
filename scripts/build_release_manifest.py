#!/usr/bin/env python3
"""Build the public release manifest."""

from __future__ import annotations

import csv
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "docs/release_manifest.csv"

SKIP_PARTS = {".git", ".venv", "__pycache__", "reproduced"}
SKIP_NAMES = {".DS_Store"}


def category(path: Path) -> str:
    parts = path.parts
    suffix = path.suffix.lower()
    if parts[0] == "manuscript":
        return "manuscript"
    if parts[0] == "figures":
        return "figure" if suffix in {".png", ".pdf"} else "figure provenance"
    if parts[0] == "scripts":
        return "code"
    if parts[0] == "docs" or path.name in {"README.md", "LICENSE", "requirements.txt"}:
        return "documentation"
    if parts[0] == "data":
        return "source data" if suffix in {".jsonl", ".csv", ".json"} else "data documentation"
    return "support"


def support(path: Path) -> str:
    rel = path.as_posix()
    if rel.startswith("manuscript/"):
        return "Paper source text and references"
    if rel.startswith("figures/main/"):
        return "Main manuscript figures"
    if rel.startswith("figures/supplementary/"):
        return "Supplementary figures"
    if rel.startswith("figures/provenance/"):
        return "Figure caption and provenance support"
    if rel.startswith("data/management_deep_probe/benchmark/"):
        return "Management 120-item benchmark"
    if rel.startswith("data/management_deep_probe/predictions/"):
        return "Management deep-probe prediction records"
    if rel.startswith("data/management_deep_probe/human_ratings/"):
        return "De-identified management human-rating analyses"
    if rel.startswith("data/management_deep_probe/pairwise/"):
        return "Management pairwise ranking comparisons"
    if rel.startswith("data/management_deep_probe/statistics/"):
        return "Management summary statistics used by tables and figures"
    if rel.startswith("data/management_deep_probe/support_tables/"):
        return "Management support tables"
    if rel.startswith("data/cross_field_standardized/benchmarks/"):
        return "Seven-field cross-field benchmark surfaces"
    if rel.startswith("data/cross_field_standardized/predictions/"):
        return "Cross-field prediction records and all-eight-field GPT-5.5 audit records"
    if rel.startswith("data/cross_field_standardized/statistics/"):
        return "Cross-field calibration, ensemble, transfer, benchmark summary, and GPT-5.5 audit statistics"
    if rel.startswith("data/cross_field_standardized/journal_tiers/"):
        return "Journal-to-tier mappings used for cross-field benchmark labels"
    if rel.startswith("data/supplementary_tables/"):
        return "Supplementary tables ST1-ST24 and ST2b"
    if rel.startswith("data/figure_support/"):
        return "Figure-to-data support index"
    if rel.startswith("scripts/"):
        return "Validation and deterministic reproduction commands"
    if rel.startswith("docs/"):
        return "External release documentation"
    return "Package support"


def main() -> int:
    rows: list[dict[str, str]] = []
    for path in sorted(ROOT.rglob("*")):
        if not path.is_file():
            continue
        rel = path.relative_to(ROOT)
        if path.name in SKIP_NAMES:
            continue
        if any(part in SKIP_PARTS for part in rel.parts):
            continue
        if rel.as_posix() == "docs/release_manifest.csv":
            continue
        rows.append(
            {
                "path": rel.as_posix(),
                "kind": category(rel),
                "supports": support(rel),
                "bytes": str(path.stat().st_size),
            }
        )

    OUT.parent.mkdir(exist_ok=True)
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["path", "kind", "supports", "bytes"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT.relative_to(ROOT)} with {len(rows)} files")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
