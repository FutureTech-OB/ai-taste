#!/usr/bin/env python3
"""Reproduce the released figure asset bundle and checksums."""

from __future__ import annotations

import csv
import hashlib
import shutil
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "reproduced/figures"
FIGURE_DIRS = [ROOT / "figures/main", ROOT / "figures/supplementary"]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, str]] = []
    for source_dir in FIGURE_DIRS:
        target_dir = OUT_DIR / source_dir.name
        target_dir.mkdir(exist_ok=True)
        for source in sorted(source_dir.iterdir()):
            if source.suffix.lower() not in {".png", ".pdf"}:
                continue
            destination = target_dir / source.name
            shutil.copy2(source, destination)
            rows.append(
                {
                    "source": source.relative_to(ROOT).as_posix(),
                    "reproduced": destination.relative_to(ROOT).as_posix(),
                    "bytes": str(destination.stat().st_size),
                    "sha256": sha256(destination),
                }
            )

    checksum_path = OUT_DIR / "figure_asset_checksums.csv"
    with checksum_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["source", "reproduced", "bytes", "sha256"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Reproduced {len(rows)} figure assets in {OUT_DIR.relative_to(ROOT)}")
    print(f"Wrote {checksum_path.relative_to(ROOT)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
