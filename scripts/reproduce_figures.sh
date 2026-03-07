#!/usr/bin/env bash
set -euo pipefail
export PYTHONNOUSERSITE="${PYTHONNOUSERSITE:-1}"
export PYTHONDONTWRITEBYTECODE="${PYTHONDONTWRITEBYTECODE:-1}"
PYTHON_BIN="${PYTHON_BIN:-python3}"

ROOT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

bad_named=$(find "$ROOT_DIR/scripts/reproducibility" -maxdepth 1 -type f -name '*_v[0-9]*.py' | wc -l | tr -d ' ')
if [ "$bad_named" -ne 0 ]; then
  echo "Naming check: FAILED (version-tagged reproducibility scripts found)"
  find "$ROOT_DIR/scripts/reproducibility" -maxdepth 1 -type f -name '*_v[0-9]*.py' | sort
  exit 1
fi

main_count=$(find "$ROOT_DIR/figures/main" -maxdepth 1 -type f \( -name '*.png' -o -name '*.pdf' \) | wc -l | tr -d ' ')
ext_count=$(find "$ROOT_DIR/figures/extended_data" -maxdepth 1 -type f \( -name '*.png' -o -name '*.pdf' \) | wc -l | tr -d ' ')
sup_count=$(find "$ROOT_DIR/figures/supplementary" -maxdepth 1 -type f \( -name '*.png' -o -name '*.pdf' \) | wc -l | tr -d ' ')
total=$((main_count + ext_count + sup_count))

echo "Main figure assets: $main_count"
echo "Extended data figure assets: $ext_count"
echo "Supplementary figure assets: $sup_count"
echo "Total figure assets: $total"

echo "Rebuilding figures from package-local scripts..."
"$PYTHON_BIN" "$ROOT_DIR/scripts/reproducibility/generate_main_figure2.py"
"$PYTHON_BIN" "$ROOT_DIR/scripts/reproducibility/generate_main_figures.py"
"$PYTHON_BIN" "$ROOT_DIR/scripts/reproducibility/generate_main_figure6.py"
"$PYTHON_BIN" "$ROOT_DIR/scripts/reproducibility/generate_extended_and_supplementary_figures.py"

"$PYTHON_BIN" - <<'PY' "$ROOT_DIR"
import sys
from pathlib import Path

root = Path(sys.argv[1])
repro = root / "reproduced" / "figures"

expected = []
for name in ["Figure2", "Figure3", "Figure4", "Figure5", "Figure6"]:
    expected.append(repro / "main" / f"{name}.png")
    expected.append(repro / "main" / f"{name}.pdf")
for i in range(1, 6):
    expected.append(repro / "extended_data" / f"ed_fig{i}.png")
    expected.append(repro / "extended_data" / f"ed_fig{i}.pdf")
for i in range(1, 7):
    expected.append(repro / "supplementary" / f"si_fig{i}.png")
    expected.append(repro / "supplementary" / f"si_fig{i}.pdf")

missing = [str(p.relative_to(root)) for p in expected if not p.exists()]
if missing:
    print("Rebuild check: FAILED")
    for p in missing:
        print(f"- missing {p}")
    raise SystemExit(1)

print(f"Rebuild check: PASS ({len(expected)} reproduced assets found)")
PY

"$PYTHON_BIN" - <<'PY' "$ROOT_DIR"
import csv
import glob
import sys
from pathlib import Path

root = Path(sys.argv[1])
index_path = root / "data" / "tables" / "FIGURE_DATA_INDEX.csv"
if not index_path.exists():
    raise SystemExit("Missing FIGURE_DATA_INDEX.csv")

missing = []
with index_path.open("r", encoding="utf-8", newline="") as f:
    reader = csv.DictReader(f)
    rows = list(reader)

for row in rows:
    figure = row.get("figure", "").strip()
    raw = row.get("primary_data_files", "")
    deps = [x.strip() for x in raw.split(";") if x.strip()]
    for dep in deps:
        if "*" in dep:
            matches = glob.glob(str(root / dep))
            if not matches:
                missing.append((figure, dep))
        else:
            if not (root / dep).exists():
                missing.append((figure, dep))

if missing:
    print("Trace check: FAILED")
    for fig, dep in missing:
        print(f"- {fig}: missing {dep}")
    raise SystemExit(1)

print(f"Trace check: PASS ({len(rows)} figure rows verified)")
PY

echo "Figure check completed"
