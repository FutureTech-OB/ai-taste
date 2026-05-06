#!/usr/bin/env python3
"""Reproduce the released machine-readable supplementary tables."""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data/supplementary_tables"
OUT_DIR = ROOT / "reproduced/tables"


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def row_count(path: Path) -> int:
    with path.open(newline="") as handle:
        return max(sum(1 for _ in csv.reader(handle)) - 1, 0)


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    copied: list[dict[str, str]] = []
    for source in sorted(SOURCE_DIR.glob("*.csv")):
        destination = OUT_DIR / source.name
        shutil.copy2(source, destination)
        copied.append(
            {
                "filename": source.name,
                "rows": str(row_count(source)),
                "sha256": sha256(destination),
            }
        )

    checksum_path = OUT_DIR / "table_checksums.csv"
    with checksum_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["filename", "rows", "sha256"])
        writer.writeheader()
        writer.writerows(copied)

    print(f"Reproduced {len(copied)} table files in {OUT_DIR.relative_to(ROOT)}")
    print(f"Wrote {checksum_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
